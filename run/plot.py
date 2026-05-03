##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import torch
from matplotlib import pyplot as plt, patches as patches
import numpy as np
from PIL import Image
from typing import cast, Any

from gradientdoe.experiment import Experiment
from gradientdoe.optimizer import Optimizer
from gradientdoe.propagate import RayleighSommerfeldMethod
from gradientdoe.sensor import SensorArray


def height_image(height, pitch, cmap, name, method, path):
    """ Stores a visualization of the optimized DOE height profile in microns. """

    # Spacial grid
    assert len(height.shape) == 2
    assert height.shape[0] == height.shape[1]
    count = height.shape[0]
    limit = (count * pitch) / 2
    h_max = np.max(height)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot image figure
    im = ax.imshow(height, cmap=cmap, vmin=0, vmax=h_max,
                   extent=(-limit, limit, -limit, limit), origin='lower')

    # Labels and titles
    ax.set_title(f"DOE Height Profile ({name})")
    ax.set_xlabel("x / µm")
    ax.set_ylabel("y / µm")
    plt.colorbar(im, ax=ax, label=r"Height / $\mu$m")

    # Save image figure
    filepath = path.format(method, "h")
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Stored height profile image: {filepath}")


def power_images(Ps, pitch, sensor, cmap, names, method, path):
    """ Stores sensor plane power images. """

    # Spatial grid
    assert len(height.shape) == 2
    assert height.shape[0] == height.shape[1]
    count = height.shape[0]

    # Window edges
    limit = (count * pitch) / 2

    # Sensor locations
    radius = sensor.diameter / 2
    centers = sensor.center
    Ns = len(centers)

    # Global normalisation over all specimen
    Ps /= np.max(Ps)
    Ni = Ps.shape[2]

    # Plot power image of each specimen
    for i in range(Ni):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot image figure with sensor outlines
        im = ax.imshow(Ps[:, :, i], cmap=cmap, vmin=0, vmax=1.0,
                       extent=(-limit, limit, -limit, limit), origin='lower')
        for s in range(Ns):
            xc, yc = centers[s]
            circle = patches.Circle((xc, yc), radius, linewidth=1, edgecolor='red', facecolor='none', alpha=0.7)
            ax.add_patch(circle)

        # Labels and titles
        ax.set_title(f"Sensor Power Image ({names[i]})")
        ax.set_xlabel("x / µm")
        ax.set_ylabel("y / µm")
        plt.colorbar(im, ax=ax, label='Power (relative)')

        # Save image figure
        assert "{0}" in path
        filepath = path.format(method, f"{i:02d}")
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Stored sensor power image: {filepath}")


def store_height(height, step_size, path):
    """ Store the height profile as a 16-bit PNG. """

    # Represent height profile as integer
    h_scaled = np.round(height / step_size)
    h_int16 = np.clip(h_scaled, 0, 2 ** 16 - 1).astype(np.uint16)

    # Store image
    img = Image.fromarray(h_int16)
    filepath = path.format("fab", "h")
    img.save(filepath, format="PNG", compress_level=6)
    print(f"Fabrication file: {path}")
    print(f"Maximum value: {np.max(h_int16)}")


if __name__ == "__main__":
    exp = Experiment.read("result.json")
    optimizer = Optimizer(exp)

    pitch = exp.grid.pitch
    count = exp.grid.count

    height = np.array(exp.height)
    H, P, Ps = optimizer.step(height, optimizer.asm, count)

    # cmap = "viridis"
    cmap = "inferno"
    path = "plots/result_{0}_{1}.png"

    name = exp.doe.material.model
    height_image(height, pitch, cmap, name, "opt", path)

    names = [x.model for x in exp.setup.sources]
    power_images(Ps, pitch, optimizer.sensor, cmap, names, "asm", path)

    # Transformation matrix
    A = np.linalg.pinv(P, rcond=1e-2)
    np.set_printoptions(formatter=cast(Any, {'float': '{: 6.3f}'.format}), linewidth=120)
    print(A)
    print(P / P.sum(axis=0, keepdims=True))
    np.set_printoptions(formatter=cast(Any, {'float': '{: .3f}'.format}), linewidth=120)
    print(A @ P)

    # names = [f"{lam*1000:.3f} nm" for lam in optimizer.doe.wavelengths]
    # path = "plots/result_w{0}.png"
    # power_images(H, pitch, optimizer.sensor, cmap, names, path)

    M = 2
    count_fab = count * M
    pitch_fab = pitch / M
    height_fab = optimizer.interpolate_height(height, count_fab)
    height_image(height_fab, pitch_fab, cmap, name, "fab", path)

    rs = RayleighSommerfeldMethod(pitch_fab, pitch, exp.setup.distance, optimizer.doe.wavelengths, optimizer.device)
    H, P, Ps = optimizer.step(height_fab, rs, count)
    power_images(Ps, pitch, optimizer.sensor, cmap, names, "rs", path)

    # Transformation matrix
    A = np.linalg.pinv(P, rcond=1e-2)
    np.set_printoptions(formatter=cast(Any, {'float': '{: 6.3f}'.format}), linewidth=120)
    print(A)
    print(P / P.sum(axis=0, keepdims=True))
    np.set_printoptions(formatter=cast(Any, {'float': '{: .3f}'.format}), linewidth=120)
    print(A @ P)

    print(f"Grid pitch: {exp.grid.pitch:.1f} µm")
    print(f"Grid count: {exp.grid.count}")
    print(f"Distance: {exp.setup.distance:.0f} µm")

    path = "plots/result.png"
    store_height(height_fab, 0.0002, path)
