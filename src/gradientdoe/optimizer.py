##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import logging
import time

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


def soft_clip(x, xmax, xfuzz):
    """ Clip the values of x to the range 0..xmax smoothly using a circular arc transition with radius xfuzz."""

    assert xmax > 0
    assert xfuzz > 0
    assert xmax > 2 * xfuzz

    def soft_edge(val, fuzz):
        arc = torch.sqrt(torch.clamp(2 * val * fuzz - val ** 2, min=1e-8))
        val = torch.where(val < 0, torch.zeros_like(val), val)
        return torch.where(val < fuzz, arc, val)

    x = xmax - soft_edge(xmax - x, xfuzz)
    x = soft_edge(x, xfuzz)
    return x


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
    count: int
    pitch: float
    sensor: SensorArray
    sensor_masks: torch.Tensor
    asm: AngularSpectrumMethod

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
        self.sensor = SensorArray(self.exp.sensor)
        self.set_grid(self.exp.grid.count, self.exp.grid.pitch)

        # Spectra of all specimen weighted by spectral sensor efficiency
        self.power = torch.tensor(power_spectra(self.exp.setup, self.exp.sensor),
                                  device=self.device, dtype=torch.float32)  # (Nk, Ni)

        # Maximum height of the DOE profile
        self.h_max = float(self.exp.doe.maxHeight)
        # self.h_max = float(self.exp.doe.maxHeight * self.exp.optimizer.maxHeightFactor)

    def set_grid(self, count, pitch):
        self.count = count
        self.pitch = pitch
        self.sensor.set_grid(count, pitch)
        self.sensor_masks = torch.tensor(self.sensor.masks, device=self.device, dtype=torch.float32)  # (Ns, N, N)
        self.asm = AngularSpectrumMethod(count, pitch, self.exp.setup.distance, self.exp.setup.wavelengths, self.device)

    def init_height(self):
        """ Return random height profile in the range [0.25 * h_max, 0.75 * h_max]. """
        height = (np.random.rand(self.count, self.count) + 0.5) * 0.5 * self.h_max
        logger.debug(f"Initial height profile: {np.min(height):.2f} - {np.max(height):.2f} µm")
        return height
        # x = torch.rand((self.count, self.count), device=self.device, dtype=torch.float32, requires_grad=True)
        # return x * self.h_max

    def get_height(self, h_raw):
        """ Generate DOE height profile (0...h_max) from raw height tensor (soft constraint). """

        if isinstance(h_raw, torch.Tensor):
            return torch.sigmoid(h_raw) * self.h_max
        return self.h_max / (1 + np.exp(-h_raw))

    def get_raw(self, height):
        if isinstance(height, torch.Tensor):
            return torch.logit(height / self.h_max)
        x = np.clip(height / self.h_max, 1e-9, 1 - 1e-9)
        return np.log(x / (1 - x))

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

    def propagate(self, height, method, count_s, jitter):
        """ Differentiable ASM propagation if plane unit input field using PyTorch. """

        # Propagate field from DOE to sensor plane
        Uo = self.doe.fields_from_height(height)
        Nk = Uo.shape[2]
        Us = torch.empty((count_s, count_s, Nk), dtype=torch.complex64, device=self.device)
        method.propagate(Uo, Us, jitter)
        # print(f"Power in:  {torch.sum(torch.abs(Uo) ** 2) / Uo.numel()}", Uo.shape)
        # print(f"Power out: {torch.sum(torch.abs(Us) ** 2) / Us.numel()}", Us.shape)

        # Power matrix for all wavelengths
        H = Us.abs() ** 2

        # Contract to signal matrix P (Ns, Ni)
        P_sk = torch.einsum('sij,ijk->sk', self.sensor_masks, H)
        P = torch.matmul(P_sk, self.power)

        return H, P

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

    def clip_height(self, height, fuzz, h_max=None):
        if h_max is None:
            h_max = self.h_max
        if isinstance(height, torch.Tensor):
            return soft_clip(height, h_max, fuzz * h_max)
        height = torch.tensor(height, device=self.device, dtype=torch.float32)
        height_clipped = self.clip_height(height, fuzz, h_max)
        return height_clipped.detach().cpu().numpy()

    def run(self, height):
        logger.debug("Starting Optimization")
        assert isinstance(height, np.ndarray)

        # Random initialisation of raw height tensor stretching from -inf to +inf
        # height_raw = torch.randn((N, N), device=self.device, dtype=torch.float32, requires_grad=True)

        # Initialize optimiser target
        self.h_max = float(max(np.max(height) * (1 + self.exp.optimizer.maxHeightFactor), self.exp.doe.maxHeight))
        logger.debug(f"Damping maxHeight: {self.h_max:.2f} -> {self.exp.doe.maxHeight:.2f} µm")
        height_raw = torch.tensor(self.get_raw(height), device=self.device, dtype=torch.float32, requires_grad=True)
        best_raw = height_raw.detach().clone()
        # best_height = height.copy()
        # height = torch.tensor(height, device=self.device, dtype=torch.float32, requires_grad=True)

        # Initialize optimiser
        opt = self.exp.optimizer
        optimizer = torch.optim.Adam([height_raw], lr=opt.learningRate)
        # optimizer = torch.optim.Adam([height], lr=opt.learningRate)

        # Initialize EMA smoothing (exponential moving average)
        ema = self.exp.optimizer.ema
        ema.start()

        t = time.time()
        log = ""
        for i in range(opt.maxLoops):

            # Reset gradients
            optimizer.zero_grad()

            # Power transfer matrix from DOE to sensor plane
            # height_clipped = self.clip_height(height, 0.01)
            # H, P = self.propagate(height_clipped, self.asm, self.count, self.jitter)
            H, P = self.propagate(self.get_height(height_raw), self.asm, self.count, self.jitter)

            # Loss function for orthogonal solution using singular values of the signal matrix
            S = torch.linalg.svdvals(P)
            S_rel = S[0] / (S[-1] + 1e-9) - 1
            l_ortho = opt.weightOrtho * S_rel ** opt.expOrtho

            # Power efficiency (P: dim=0 is sensor dim=1 is specimen)
            P_eta = -torch.log((P.mean(dim=1)).sum() / self.count ** 2 + 1e-9)
            l_eta = opt.weightEta * P_eta

            # Maximum height
            l_height = self.h_max - self.exp.doe.maxHeight

            # Total loss function with weights
            loss = l_ortho + l_eta + l_height

            # Backpropagation
            loss.backward()
            optimizer.step()

            h_max = self.h_max - self.exp.optimizer.maxHeightFactor * (self.h_max - self.exp.doe.maxHeight)
            self.h_max = float(max(h_max, self.exp.doe.maxHeight))
            delta_h = self.h_max - self.exp.doe.maxHeight

            # EMA smoothing step
            if ema.step(loss.item()):
                # best_height = height_clipped.detach().cpu().numpy()
                best_raw = height_raw.detach().cpu().numpy()
                P_over = P.mean() / (self.count ** 2 * self.sensor.area_ratio)
                log = f"{l_ortho.item():7.2f} | {l_eta.item():7.2f} || {S_rel:7.3f} | {P_over:7.3f} | {delta_h:7.3f}"

            # Logging
            if ema.has_finished or time.time() - t > 2:
                t = time.time()
                if log:
                    logger.debug(f"{self.count:5d} | {i:5d} | {ema.counter:3d} || {log}")

                if ema.has_finished:
                    logger.debug(
                        f"Converged [{self.count}]: Improvement < {ema.threshold * 100}% for {ema.patience} iterations.")
                    break

        if ema.has_finished:
            # height = best_height
            height = self.get_height(best_raw)
        else:
            height = None
        return height
