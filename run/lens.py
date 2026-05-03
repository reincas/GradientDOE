##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gradientdoe.element import DiffractiveOpticalElement
from gradientdoe.propagate import RayleighSommerfeldMethod, AngularSpectrumMethod
from gradientdoe.spectrum import IndexSpectrum
from run.main import MATERIALS


def spherical_lens(pitch, count, focus_distance, delta_n):
    """ Return a 2D grid of height values for a convex spherical lens. """

    # Radius of curvature
    R = focus_distance * delta_n

    # Pixel grid
    limit = ((count - 1) * pitch) / 2
    x = np.linspace(-limit, limit, count)
    y = np.linspace(-limit, limit, count)
    X, Y = np.meshgrid(x, y)

    # Squared radial distances
    r2 = X ** 2 + Y ** 2

    # Spherical Sagitta formula: sag = R - sqrt(R^2 - r^2)
    sag = R - np.sqrt(np.maximum(R ** 2 - r2, 0))

    # Return height grid
    return sag.max() - sag


def plot(P_asm, P_rs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def add_colorbar(im, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        return fig.colorbar(im, cax=cax)

    v_min = min(P_asm.min(), P_rs.min())
    v_max = max(P_asm.max(), P_rs.max())

    # Plot first array
    im1 = ax1.imshow(P_asm, cmap='magma', vmin=v_min, vmax=v_max)
    ax1.set_title('Power (ASM)')
    add_colorbar(im1, ax1)

    # Plot second array
    im2 = ax2.imshow(P_rs, cmap='magma', vmin=v_min, vmax=v_max)
    ax2.set_title('Power (RS)')
    add_colorbar(im2, ax2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wavelengths = np.array([0.640])
    material = IndexSpectrum(MATERIALS["IP-S"], wavelengths)
    n = np.array(material.values)
    Nk = len(wavelengths)

    pitch = 20
    count = 64
    zc = count * pitch ** 2 / wavelengths[0]
    distance = 8 * zc
    focus = 1.2 * distance

    doe = DiffractiveOpticalElement(wavelengths, n, device)

    height = torch.tensor(spherical_lens(pitch, count, focus, n[0] - 1), dtype=torch.float32, device=device)
    Uo = doe.fields_from_height(height)
    Us = torch.empty((count, count, Nk), dtype=torch.complex64, device=device)

    asm = AngularSpectrumMethod(count, pitch, distance, wavelengths, device, "FTF")
    asm.propagate(Uo, Us, jitter=False)
    U_asm = Us[:, :, 0].cpu().numpy()
    P_asm = (np.abs(U_asm) / count) ** 2

    count_o = count * 4
    pitch_o = pitch * count / count_o
    count_s = count
    pitch_s = pitch * count / count_s
    height_fab = torch.tensor(spherical_lens(pitch_o, count_o, focus, n[0] - 1), dtype=torch.float32, device=device)
    Uo_fab = doe.fields_from_height(height_fab)
    Us_fab = torch.empty((count_s, count_s, Nk), dtype=torch.complex64, device=device)

    rs = RayleighSommerfeldMethod(pitch_o, pitch_s, distance, wavelengths, device)
    rs.propagate(Uo_fab, Us_fab, jitter=False)
    U_rs = Us_fab[:, :, 0].cpu().numpy()
    P_rs = (np.abs(U_rs) / count) ** 2

    plot(P_asm, P_rs)
