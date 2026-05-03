##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import logging
import numpy as np
import psutil
import torch

from .parameter import Parameter
from .propagate import AngularSpectrumMethod
from .element import DiffractiveOpticalElement
from .sensor import SensorArray

logger = logging.getLogger("optimiser")


def memory(device):
    """ Return the amount of memory available on the device in bytes. """
    if device.type == "cuda":
        return torch.cuda.get_device_properties(0).total_memory
    return psutil.virtual_memory().total


def power_spectra(setup, sensor):
    """ Return the normalized power spectra of all specimen weighted by the spectral sensor efficiency. """

    # Extract sizes
    Nk = len(setup.wavelengths)
    Ni = len(setup.sources)

    # Stack con taining the power spectrum of each specimen i
    power = np.zeros((Nk, Ni), dtype=float)
    for i in range(Ni):
        power[:, i] = setup.sources[i].values

    # Multiply spectra by the spectral power efficiency of the detector
    power = np.einsum("ki,k->ki", power, sensor.eta.values)

    # Normalize spectra
    power /= power.sum(axis=0, keepdims=True)

    # Return normalized power spectra (Nk, Ni)
    return power


class Ema(Parameter):
    """ Exponential moving average class for loss recording. """

    count: int
    loss: float
    bestLoss: float
    alpha: float
    threshold: float
    patience: int

    def __init__(self, data):
        super().__init__(data)
        self.counter = 0
        self.start()

    def start(self):
        self.counter = 0
        self.loss = 1e99
        self.bestLoss = 1e99

    def step(self, loss):
        if self.loss is None:
            self.loss = loss
        else:
            self.loss = (self.alpha * loss) + (1 - self.alpha) * self.loss

        if self.bestLoss is None:
            improvement = 1.0
        else:
            improvement = (self.bestLoss - self.loss) / (abs(self.bestLoss) + 1e-9)

        if improvement > self.threshold:
            self.bestLoss = self.loss
            self.counter = 0
            return True
        self.counter += 1
        return False

    @property
    def has_finished(self):
        return self.counter >= self.patience


class Optimizer:
    def __init__(self, exp):
        self.exp = exp
        self.jitter = exp.optimizer.jitter

        # Initialise PyTorch environment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Running on {self.device.type.upper()} with {memory(self.device) / 1024 ** 3:.2f} GB")
        if self.device.type == "cuda":
            logger.debug(f"Device Name: {torch.cuda.get_device_name(0)}")
            major, minor = torch.cuda.get_device_capability(0)
            logger.debug(f"Compute Capability: {major}.{minor}")
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r - a  # free inside reserved
            logger.debug(f"Total VRAM: {t / 1024 ** 3:.2f} GB")
            logger.debug(f"Reserved:   {r / 1024 ** 3:.2f} GB")
            logger.debug(f"Allocated:  {a / 1024 ** 3:.2f} GB")

        # Initialise DOE
        self.doe = DiffractiveOpticalElement(self.exp.setup.wavelengths, self.exp.doe.material.values, self.device)

        # Initialise the sensor array
        self.sensor = SensorArray(self.exp.sensor, self.exp.grid)
        self.weight_distance = torch.tensor(self.sensor.next_distance ** 2, device=self.device, dtype=torch.float32)
        self.sensor_masks = torch.tensor(self.sensor.masks, device=self.device, dtype=torch.float32)  # (Ns, N, N)

        # Spectra of all specimen weighted by spectral sensor efficiency
        self.power = torch.tensor(power_spectra(self.exp.setup, self.exp.sensor),
                                  device=self.device, dtype=torch.float32)  # (Nk, Ni)

        # Maximum height of the DOE profile
        self.h_max = float(self.exp.doe.maxHeight)

        # Initialise angular spectrum method
        self.asm = AngularSpectrumMethod(self.exp.grid.count, self.exp.grid.pitch, self.exp.setup.distance,
                                         self.exp.setup.wavelengths, self.device)

    def get_height(self, h_raw):
        """ Generate DOE height profile (0...h_max) from raw height tensor (soft constraint). """

        return torch.sigmoid(h_raw) * self.h_max

    def propagate(self, height, method, count_s, jitter):
        """ Differentiable ASM propagation if plane unit input field using PyTorch. """

        # Propagate field from DOE to sensor plane
        Uo = self.doe.fields_from_height(height)
        Nk = Uo.shape[2]
        Us = torch.empty((count_s, count_s, Nk), dtype=torch.complex64, device=self.device)
        method.propagate(Uo, Us, jitter)

        # Power matrix for all wavelengths
        H = Us.abs() ** 2

        # Contract to signal matrix P (Ns, Ni)
        P_sk = torch.einsum('sij,ijk->sk', self.sensor_masks, H)
        P = torch.matmul(P_sk, self.power)

        return H, P

    def run(self):
        logger.debug("Starting Optimization")

        N = self.exp.grid.count

        # Random initialisation of raw height tensor stretching from -inf to +inf
        h_raw = torch.randn((N, N), device=self.device, dtype=torch.float32, requires_grad=True)
        best_raw = h_raw.detach().clone()

        # Initialize optimiser
        opt = self.exp.optimizer
        optimizer = torch.optim.Adam([h_raw], lr=opt.learningRate)

        # Initialize EMA smoothing (exponential moving average)
        ema = self.exp.optimizer.ema
        ema.start()

        for i in range(opt.maxLoops):

            # Reset gradients
            optimizer.zero_grad()

            # Determine DOE height profile from raw height tensor
            height = self.get_height(h_raw)

            # Power transfer matrix from DOE to sensor plane
            H, P = self.propagate(height, self.asm, N, self.jitter)

            # Singular values of the signal matrix
            S = torch.linalg.svdvals(P) / N ** 2

            # Loss function for orthogonal solution
            l_ortho = opt.weightOrtho * S[0] / (S[-1] + 1e-9)

            # Loss function for maximized power efficiency
            l_eta = -opt.weightEta * torch.log(S + 1e-9).sum()

            # Loss function for centering the light on the sensors
            l_center = opt.weightCenter * torch.mean(H.sum(dim=2) * self.weight_distance) / N ** 2

            # Total loss function with weights
            loss = l_ortho + 0 * l_eta + l_center

            # Backpropagation
            loss.backward()
            optimizer.step()

            # EMA smoothing step
            if ema.step(loss.item()):
                best_raw = h_raw.detach().clone()

            # Logging
            if i % 1 == 0:
                max_h = height.max().item() - height.min().item()
                Psum = (torch.sum(P, dim=0) / N ** 2).tolist()
                Psum = ", ".join([f"{p:.3f}" for p in Psum])
                logger.debug(
                    f"{i:5d} | {ema.counter:3d} | {l_ortho.item():6.2f} | {(l_eta).item():6.2f} | {(l_center).item():6.2f} | {max_h:6.2f} µm | {Psum}")

            if ema.has_finished:
                logger.debug(f"Converged: No improvement > {ema.threshold * 100}% for {ema.patience} iterations.")
                break

        if ema.has_finished:
            height = self.get_height(best_raw)
            height = height.detach().cpu().numpy()
            height -= np.min(height)
        else:
            height = None
        return height

    def step(self, height, method, count_s):
        """ Calculate H, P, and Ps for a given physical height profile illuminated by unit fields. """

        # Prepare height tensor
        if isinstance(height, torch.Tensor):
            height_tensor = height.to(self.device)
        else:
            height_tensor = torch.tensor(height, device=self.device, dtype=torch.float32)

        # Propagate unit fields to the sensor plane
        N = count_s
        with torch.no_grad():
            H, P = self.propagate(height_tensor, method, N, jitter=False)
            Ps = torch.matmul(H, self.power)

        # Normalise powers as numpy arrays
        H = H.cpu().numpy() / N ** 2
        P = P.cpu().numpy() / N ** 2
        Ps = Ps.cpu().numpy() / N ** 2

        # Return results
        return H, P, Ps

    def interpolate_height(self, height, target_count):
        """ Tensor-based spectral interpolation of a height profile. """

        N = height.shape[0]
        M = target_count

        # Pixel sizes must be powers of 2
        assert N > 0 and (N & (N - 1)) == 0
        assert M > 0 and (M & (M - 1)) == 0

        # Move height profile to GPU if possible
        height_tensor = torch.tensor(height, device=self.device, dtype=torch.float32)

        # Spatial spectrum
        height_spectrum = torch.fft.fft2(height_tensor)
        height_spectrum = torch.fft.fftshift(height_spectrum)

        # Zero-padded spacial spectrum
        padded_spectrum = torch.zeros((M, M), dtype=torch.complex64, device=self.device)
        start = (M - N) // 2
        end = start + N
        padded_spectrum[start:end, start:end] = height_spectrum

        # Back transformation to interpolated height profile
        padded_spectrum = torch.fft.ifftshift(padded_spectrum)
        height = torch.fft.ifft2(padded_spectrum)

        # Scale and shift real part
        height = height.real * (M / N) ** 2
        height -= height.min()

        # Return interpolated height profile as numpy array
        return height.cpu().numpy()
