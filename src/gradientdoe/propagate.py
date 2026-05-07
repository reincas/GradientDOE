##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import numpy as np
import torch


##########################################################################
# Angular spectrum method
##########################################################################

def etf_kernel(z, lam, f_sq):
    r""" Return the spectral kernel $H = \exp{i k z \sqrt{1 - \lambda^2 (f_x^2 + f_y^2)}}$ of the exact transfer
    function for the given wavelength lam and the squared frequencies f_sq. """

    k = 2 * np.pi / lam

    # Suppress evanescent waves to prevent exponential growth and numerical instability
    argument = 1 - (lam ** 2 * f_sq)
    mask = argument > 0

    phase = np.zeros_like(argument)
    phase[mask] = k * z * np.sqrt(argument[mask])
    return np.exp(1j * phase) * mask


def ftf_kernel(z, lam, f_sq):
    r""" Return the spectral kernel $H = \exp{i k z (1 - \lambda^2 (f_x^2 + f_y^2) / 2})$ of the Fresnel transfer
    function for the given wavelength lam and the squared frequencies f_sq. """

    k = 2 * np.pi / lam
    phase = k * z * (1 - lam ** 2 * f_sq / 2)
    return np.exp(1j * phase)


def spectral_kernels(pixel_count, pixel_size, z, wavelengths, f_kernel):
    """ Return the spectral kernels for the given wavelengths. """

    if isinstance(wavelengths, torch.Tensor):
        wavelengths = wavelengths.detach().cpu().numpy()

    N = pixel_count
    p = pixel_size
    Nk = len(wavelengths)

    # Spatial frequency grid mesh from -1/(2p) to 1/(2p)
    f = np.fft.fftfreq(N, d=p).astype(np.float32)
    fx, fy = np.meshgrid(f, f)
    f_sq = fx ** 2 + fy ** 2

    # Initialize phase transfer kernels
    kernels = np.empty((N, N, Nk), dtype=np.complex64)

    # Calculate kernel for each wavelength
    for i, lam in enumerate(wavelengths):
        kernels[:, :, i] = f_kernel(z, lam, f_sq)

    # Return spectral ETF kernels
    return kernels


class AngularSpectrumMethod:
    def __init__(self, pixel_count, pixel_pitch, z, wavelengths, device, kernel="ETF"):
        # Total memory allocation: 576 MB (self.kernels)

        self.pixel_count = pixel_count
        self.pixel_size = pixel_pitch
        self.z = z
        self.wavelengths = wavelengths
        self.device = device

        # Fresnel number check
        NF = (pixel_count * pixel_pitch / 2) ** 2 / (z * min(wavelengths))
        if NF < 0.25:
            raise NotImplementedError(f"Fresnel number {NF:.2f} < 0.25 not implemented.")

        # Resolution check
        zc = pixel_count * pixel_pitch ** 2 / min(wavelengths)
        if z < zc:
            raise ValueError(f"Propagation distance {z:.0f} µm below minimum for given grid ({zc:.0f} µm).")

        # Spatial frequency grid exponent
        self.fexp = -2j * torch.pi * torch.fft.fftfreq(pixel_count, device=self.device)

        # Pre-calculation of spectral kernels
        if kernel == "ETF":
            f_kernel = etf_kernel
        elif kernel.upper() == "FTF":
            f_kernel = ftf_kernel
        else:
            raise ValueError(f"Unknown kernel {kernel}!")
        self.kernels = torch.tensor(spectral_kernels(pixel_count, pixel_pitch, z, wavelengths, f_kernel),
                                    device=self.device, dtype=torch.complex64)

    def get_jitter(self):

        shift_x = torch.rand(1, device=self.device, dtype=torch.float32) - 0.5
        shift_y = torch.rand(1, device=self.device, dtype=torch.float32) - 0.5
        ramp_x = torch.exp(self.fexp * shift_x)
        ramp_y = torch.exp(self.fexp * shift_y)
        jitter = ramp_y[:, None] * ramp_x
        return jitter

    def propagate(self, Uo, jitter):
        """ Propagate source field Uo to image field Us. Add a grid jitter if jitter == True. """

        # Calculate image field
        Uf = torch.fft.fft2(Uo, dim=(0, 1))
        if jitter:
            return torch.fft.ifft2(Uf * self.get_jitter()[:, :, None] * self.kernels, dim=(0, 1))
        return torch.fft.ifft2(Uf * self.kernels, dim=(0, 1))


##########################################################################
# First Rayleigh-Sommerfeld method
##########################################################################

def pixel_kernel(z2, kx_o, ky_o, kx_sm, ky_sn, U_o, U_s):
    for n, ky_s in enumerate(ky_sn):
        dkzky2 = z2 + (ky_s - ky_o[:, None]) ** 2
        for m, kx_s in enumerate(kx_sm):
            dkx2 = (kx_s - kx_o[None, :]) ** 2
            phi = torch.sqrt(dkzky2 + dkx2)
            g = (1.0 / phi - 1j) * torch.exp(1j * phi) / (phi ** 2)
            U_s[n, m] = (U_o * g).sum()
            # U_s[n, m] = torch.vdot(U_o.flatten(), g.flatten().conj())


def rayleigh_sommerfeld(k, z, px_o, py_o, px_s, py_s, U_o, Nx_s, Ny_s, device):
    # Dimensionless coordinates
    kz = z * k
    kpx_o = px_o * k
    kpy_o = py_o * k
    kpx_s = px_s * k
    kpy_s = py_s * k

    # Size of source grid
    Ny_o, Nx_o = U_o.shape

    # Pixel offsets
    dkx_off = ((Nx_o - 1) * kpx_o - (Nx_s - 1) * kpx_s) / 2
    dky_off = ((Ny_o - 1) * kpy_o - (Ny_s - 1) * kpy_s) / 2
    kx_o = torch.arange(Nx_o, device=device, dtype=torch.float32) * kpx_o - dkx_off
    ky_o = torch.arange(Ny_o, device=device, dtype=torch.float32) * kpy_o - dky_off

    # Initialise image field
    U_s = torch.empty((Ny_s, Nx_s), device=device, dtype=torch.complex64)
    z2 = kz ** 2

    # Build each image pixel as superposition of Huygens waves from all source pixels
    kx_sm = torch.arange(Nx_s, device=device, dtype=torch.float32) * kpx_s
    ky_sn = torch.arange(Ny_s, device=device, dtype=torch.float32) * kpy_s
    pixel_kernel(z2, kx_o, ky_o, kx_sm, ky_sn, U_o, U_s)

    # Global scaling
    U_s *= kz * kpx_o * kpy_o / (2 * torch.pi)
    return U_s


class RayleighSommerfeldMethod:
    def __init__(self, pixel_size_source, pixel_size_image, z, wavelengths, device):
        self.pixel_size_source = pixel_size_source
        self.pixel_size_image = pixel_size_image
        self.z = z
        self.wavelengths = wavelengths
        self.device = device

    def propagate(self, Uo, Us, jitter, k_select=None):
        """ Propagate source field Uo to image field Us for the given set of wavelength indices. Default is all
        wavelengths. """

        assert Us.shape[2] == Uo.shape[2]

        if jitter:
            raise NotImplementedError

        # Default is all wavelengths
        if not k_select:
            k_select = range(Uo.shape[2])

        px_o = py_o = self.pixel_size_source
        px_s = py_s = self.pixel_size_image
        Nx_s = Us.shape[1]
        Ny_s = Us.shape[0]

        # Calculate sensor power matrix for each wavelength
        for k in k_select:
            print(f"Calculating {self.wavelengths[k] * 1000:.3f} nm")
            kn = 2 * torch.pi / self.wavelengths[k]
            Us[:, :, k] = rayleigh_sommerfeld(kn, self.z, px_o, py_o, px_s, py_s, Uo[:, :, k], Nx_s, Ny_s, self.device)
