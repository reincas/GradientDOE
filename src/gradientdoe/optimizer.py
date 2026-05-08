##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import logging
import numpy as np
import psutil
import time
import torch
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

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


def get_kernel_size(radius):
    """ Return the Gaussian kernel size for a given radius. """

    return 2 * round(4 * radius) + 1


def gaussian_blur(input, radius: float) -> torch.Tensor:
    """ Applies Gaussian blur to a 2D tensor. """

    is_tensor = isinstance(input, torch.Tensor)
    if not is_tensor:
        input = torch.Tensor(input, device="cpu")

    assert input.dim() == 2
    assert radius >= 0

    input = input.unsqueeze(0).unsqueeze(0)

    # Odd kernel size
    kernel_size = get_kernel_size(radius)
    if kernel_size <= 1:
        blurred = input

    else:
        # Normalised 1D Gaussian distribution
        coords = torch.arange(kernel_size, device=input.device).float() - (kernel_size - 1) / 2
        g_1d = torch.exp(-(coords ** 2) / (2 * radius ** 2))
        g_1d = g_1d / g_1d.sum()

        # 2D Gaussian kernel
        g_2d = g_1d.view(-1, 1) @ g_1d.view(1, -1)
        kernel = g_2d.view(1, 1, kernel_size, kernel_size)

        # Convolution with padding to maintain spatial dimensions
        padding = kernel_size // 2
        blurred = F.conv2d(input, kernel, padding=padding)

    blurred = blurred.squeeze()
    if not is_tensor:
        blurred = blurred.numpy()

    return blurred


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

    def __init__(self, exp, device=None):

        self.exp = exp
        self.jitter = exp.optimizer.jitter

        # Initialise PyTorch environment
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Optimizer running on {self.device.type.upper()} with {memory(self.device) / 1024 ** 3:.2f} GB")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info(f"    Device Name: {torch.cuda.get_device_name(0)}")
            major, minor = torch.cuda.get_device_capability(0)
            logger.info(f"    Compute Capability: {major}.{minor}")
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            logger.info(f"    Total VRAM: {t / 1024 ** 3:.2f} GB")
            logger.info(f"    Reserved:   {r / 1024 ** 3:.2f} GB")
            logger.info(f"    Allocated:  {a / 1024 ** 3:.2f} GB")

        # Initialise DOE
        self.doe = DiffractiveOpticalElement(self.exp.setup.wavelengths, self.exp.doe.refractiveIndex.values, self.device)

        # Initialise the sensor array
        self.sensor = SensorArray(self.exp.sensor)
        self.set_grid(self.exp.grid.count, self.exp.grid.pitch)

        # Spectra of all specimen weighted by spectral sensor efficiency
        self.power = torch.tensor(power_spectra(self.exp.setup, self.exp.sensor),
                                  device=self.device, dtype=torch.float32)
        # Maximum height of the DOE profile
        self.h_max = float(self.exp.doe.maxHeight)

    def set_grid(self, count, pitch):
        # Total memory allocation: 1408 MB (self.sensor_masks, self.asm.kernels)

        self.count = count
        self.pitch = pitch

        self.sensor.set_grid(count, pitch)
        self.sensor_masks = torch.tensor(self.sensor.masks, device=self.device, dtype=torch.float32)

        self.asm = AngularSpectrumMethod(count, pitch, self.exp.setup.distance, self.exp.setup.wavelengths, self.device)

    def init_height(self):
        """ Return random height profile in the range [0.25 * h_max, 0.75 * h_max]. """
        height = (np.random.rand(self.count, self.count) + 0.5) * 0.5 * self.h_max
        logger.info(f"Initial height profile: {np.min(height):.2f} - {np.max(height):.2f} µm")
        return height

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

        # Apply Gaussian blur filter
        pitch = self.pitch * height.shape[0] / self.count
        radius = self.exp.doe.blurRadius / pitch
        height = gaussian_blur(height, radius)

        # Scale and shift real part
        height = height.real * (M / N) ** 2
        height -= height.min()

        # Return interpolated height profile as numpy array
        return height.cpu().numpy()

    def step(self, height, method, count_s):
        """ Calculate H, P, and Ps for a given physical height profile illuminated by unit fields. """

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            # Prepare height tensor
            if isinstance(height, torch.Tensor):
                height_tensor = height.to(self.device)
            else:
                height_tensor = torch.tensor(height, device=self.device, dtype=torch.float32)

            # Propagate unit fields to the sensor plane
            U = self.doe.fields_from_height(height_tensor)
            U = method.propagate(U, jitter=False)
            del height_tensor

            # Power distribution in the sensor plane for all specimen (N, N, Ni)
            Ps = torch.einsum('xyk,ki->xyi', U.abs() ** 2, self.power)
            del U

            # Sensor power vs. specimen matrix (Ns, Ni)
            P = torch.einsum('sxy,xyi->si', self.sensor_masks, Ps)

        # Normalise powers as numpy arrays
        P = P.cpu().numpy() / count_s ** 2
        Ps = Ps.cpu().numpy() / count_s ** 2

        # Return results
        return P, Ps

    def run(self, height, learning_rate):
        assert isinstance(height, np.ndarray)

        def forward_propagate(h):
            U = self.doe.fields_from_height(h)
            return self.asm.propagate(U, self.jitter)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info("Starting Optimization")
        lr = format(float(format(learning_rate, ".2g")), "f").rstrip('0').rstrip('.')
        logger.info(f"    Learning Rate: {lr}")

        height -= np.min(height)
        if np.max(height) > self.exp.doe.maxHeight:
            self.h_max = float(max(np.max(height) * (1 + self.exp.optimizer.maxHeightFactor), self.exp.doe.maxHeight))
            logger.info(f"    Damping maximum height: {self.h_max:.2f} -> {self.exp.doe.maxHeight:.2f} µm")
        else:
            self.h_max = self.exp.doe.maxHeight
            logger.info(f"    Maximum height: {np.max(height):.2f} µm")

        # Initialize optimiser target
        height_opt = torch.tensor(height, device=self.device, dtype=torch.float32, requires_grad=True)
        best_height = height

        # Initialize optimiser
        opt = self.exp.optimizer
        optimizer = torch.optim.Adam([height_opt], lr=learning_rate)

        use_checkpoint = self.count >= opt.checkpointThreshold
        logger.info(f"    Using checkpoint: {use_checkpoint}")

        blur_radius = self.exp.doe.blurRadius / self.pitch
        logger.info(f"    Gaussian blur kernel size: {get_kernel_size(blur_radius)}")

        # Initialize EMA smoothing (exponential moving average)
        ema = self.exp.optimizer.ema
        ema.start()

        t = time.time()
        log = ""
        for i in range(opt.maxLoops):

            # Reset gradients
            optimizer.zero_grad(set_to_none=True)

            # Height adjustment
            height_tensor = gaussian_blur(height_opt, blur_radius)
            height_tensor = height_tensor - height_tensor.min()
            if height_tensor.max() > self.h_max:
                height_tensor = height_tensor * (self.h_max / height_tensor.max())

            # ASM field propagation
            if self.device.type != "cpu" and use_checkpoint:
                U = checkpoint(forward_propagate, height_tensor, use_reentrant=False)
            else:
                U = forward_propagate(height_tensor)

            # Sensor power matrix (Ns, Ni)
            P = torch.einsum('sxy,xyk,ki->si', self.sensor_masks, U.abs() ** 2, self.power)
            del U

            # Loss function for orthogonal solution using singular values of the signal matrix
            S = torch.linalg.svdvals(P)
            S_rel = S[0] / (S[-1] + 1e-9) - 1
            l_ortho = opt.weightOrtho * S_rel ** opt.expOrtho

            # Power efficiency (P: dim=0 is sensor dim=1 is specimen)
            P_eta = -torch.log(P.mean() / self.count ** 2 + 1e-9)
            l_eta = opt.weightEta * P_eta

            # Minimize pixel gradient
            diff_x = torch.abs(height_tensor[:, 1:] - height_tensor[:, :-1]).mean()
            diff_y = torch.abs(height_tensor[1:, :] - height_tensor[:-1, :]).mean()
            l_grad = opt.weightGrad * (diff_x + diff_y) / self.pitch

            # Maximum height limit
            h_max = height_tensor.detach().max()
            if h_max <= self.exp.doe.maxHeight:
                self.h_max = self.exp.doe.maxHeight
            else:
                self.h_max = h_max - opt.maxHeightFactor * (h_max - self.exp.doe.maxHeight)

            # Total loss function with weights
            loss = l_ortho + l_eta + l_grad

            # EMA smoothing step
            if ema.step(loss.item()) or i == 0:
                best_height = height_tensor.detach().cpu().numpy()
                P_over = P.mean() / (self.count ** 2 * self.sensor.area_ratio)
                diff_x = np.max(np.abs(best_height[:, 1:] - best_height[:, :-1]))
                diff_y = np.max(np.abs(best_height[1:, :] - best_height[:-1, :]))
                max_grad = max(diff_x, diff_y) / self.pitch
                h_rel = h_max / self.exp.doe.maxHeight
                log = f"{l_ortho.item():7.2f} | {l_eta.item():7.2f} | {l_grad.item():7.2f}" + \
                      f" || {S_rel:7.3f} | {P_over:7.3f} | {max_grad:7.3f} | {h_rel:7.3f}"

            # Logging
            if i == 0 or ema.has_finished or time.time() - t > 2:
                t = time.time()
                if log:
                    logger.info(f"{self.count:5d} | {i:5d} | {ema.counter:3d} || {log}")

                if ema.has_finished and (h_max / self.exp.doe.maxHeight - 1) < opt.maxHeightThreshold:
                    logger.info(
                        f"Converged [{self.count}]: Improvement < {ema.threshold * 100}% for {ema.patience} iterations.")
                    break

            # Backpropagation
            del height_tensor
            loss.backward()
            optimizer.step()

        if ema.has_finished:
            height = best_height
            logger.info(f"Final maximum height: {np.max(height):.2f} µm")
        else:
            height = None
        return height
