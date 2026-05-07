##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import h5py
import math
from matplotlib import pyplot as plt, patches as patches
import numpy as np
from pathlib import Path
from PIL import Image
import sys

from gradientdoe.experiment import Experiment, next_power_of_2
from gradientdoe.optimizer import Optimizer


def get_next_preferred_number(x: float) -> float:
    """ Return the next larger 1*10^N, 2*10^N, or 5*10^N for a given positive float x. """

    if x <= 0:
        raise ValueError("x must be a positive float.")

    exponent = math.floor(math.log10(x))
    fraction = x / (10 ** exponent)

    if fraction <= 1:
        multiplier = 1
    elif fraction <= 2:
        multiplier = 2
    elif fraction <= 5:
        multiplier = 5
    else:
        multiplier = 10

    return float(multiplier * (10 ** exponent))


def get_counts(filename):
    """ Return sorted list of pixel counts of all height profiles. """

    with h5py.File(filename, "r") as fp:
        counts = sorted([int(x.split("_", 1)[1]) for x in fp.keys() if x.startswith("height_")])
    return list(counts)


def read_height(filename, count=None):
    """ Read given (or largest) height profile. """

    with h5py.File(filename, "r") as fp:
        if count is None:
            count = max([int(x.split("_", 1)[1]) for x in fp.keys() if x.startswith("height_")])
        name = f"height_{count}"
        assert name in fp.keys()
        return np.array(fp[name])


def store_height_plot(height, pitch, cmap, name, path):
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
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def store_power_plots(Ps, P, pitch, sensor, cmap, names, path):
    """ Stores sensor plane power images. """

    # Window edges
    count = Ps.shape[0]
    limit = (count * pitch) / 2

    # Sensor locations
    radius = sensor.diameter / 2
    centers = sensor.centers
    Ns = sensor.num_sensors

    # Normalise power
    Ps /= np.max(Ps)

    # Prepare figure
    Ni = Ps.shape[2]
    fig, axes = plt.subplots(1, Ni, figsize=(5 * Ni, 5), constrained_layout=True)
    fig.suptitle("Optical power distribution in the sensor plane")
    if Ni == 1:
        axes = [axes]

    # Plot power image of each specimen
    for i in range(Ni):
        ax = axes[i]

        # Plot image figure with sensor outlines
        im = ax.imshow(Ps[:, :, i], cmap=cmap, vmin=0, vmax=1.0,
                       extent=(-limit, limit, -limit, limit), origin='lower')
        for s in range(Ns):
            xc, yc = centers[s]
            circle = patches.Circle((xc, yc), radius, linewidth=1, edgecolor='red', facecolor='none', alpha=0.7)
            ax.add_patch(circle)
            if P is not None:
                ax.text(xc, yc - radius - (radius * 0.3), f"{P[s, i] * 100:.1f} %",
                        color='white', ha='center', va='top', fontsize=8, fontweight='normal')

        # Labels and titles
        ax.set_title(f"Specimen: {names[i]}")
        ax.set_xlabel("x / µm")
        ax.set_ylabel("y / µm")
        if i == Ni - 1:
            plt.colorbar(im, ax=ax, label='Power (relative)')

    # Save image figure
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def store_height_profile(height, path):
    """ Store the height profile as a 16-bit PNG. """

    step_size = get_next_preferred_number(np.max(height) / (2 ** 16 - 1))
    h_int16 = np.round(height / step_size).astype(np.uint16)
    assert np.max(h_int16) < 2 ** 16 - 1

    img = Image.fromarray(h_int16)
    img.save(path, format="PNG", compress_level=6)
    return step_size


def plot(optimizer, count, root):
    exp = optimizer.exp
    if not isinstance(root, Path):
        root = Path(root)

    # Height profile
    height_path = root / "height.h5"
    counts = get_counts(height_path)
    if count in counts:
        height = read_height(height_path, count)
        print(f"Optimised height profile:")
    else:
        assert count == next_power_of_2(count), f"Pixel count {count} is not a power of 2."
        assert count > max(counts)
        count_opt = max(counts)
        height_opt = read_height(height_path, count_opt)
        height = optimizer.interpolate_height(height_opt, count)
        print(f"Interpolated height profile:")

    # Pixel pitch
    M = count // exp.grid.count
    pitch = exp.grid.pitch / M
    print(f"    Pixel pitch:     {pitch:.1f} µm")
    print(f"    Pixel count:     {count}")
    print(f"    Sensor distance: {exp.setup.distance / 1000:.1f} mm")
    print(f"    Heights:         {np.min(height):.2f} - {np.max(height):.2f} µm")

    # Prepare diagram formatting
    cmap = "viridis"
    # cmap = "inferno"

    # Store height profile as 16-bit PNG image
    path = root / f"profile_{count}.png"
    step_size = store_height_profile(height, path)
    print(f"    Stored fabrication file: {path} with step size: {step_size} µm")

    # Store height profile as plot
    name = exp.doe.material.model
    path = root / f"height_{count}.png"
    store_height_plot(height, pitch, cmap, name, path)
    print(f"    Stored height profile image: {path}")

    # Sensor power for every specimen
    optimizer.set_grid(count, pitch)
    P, Ps = optimizer.step(height, optimizer.asm, count)
    names = [x.model for x in exp.setup.sources]
    assert len(names) == P.shape[1]
    size = max(len(name) for name in names)
    fmt = f"    {{0:{size}s}}: {{1}}"
    for i, name in enumerate(names):
        values = ", ".join(f"{x:6.3f}" for x in P[:, i])
        print(fmt.format(name, values))

    # Store power image with sensor outlines
    path = root / f"power_{count}.png"
    store_power_plots(Ps, P, pitch, optimizer.sensor, cmap, names, path)
    print(f"    Stored sensor power image: {path}")


def get_path():
    """ Returns the first command line argument as a Path object. """
    try:
        return Path(sys.argv[1])
    except IndexError:
        return None


if __name__ == "__main__":
    root = get_path()
    if root is None:
        print("Result folder required as command line argument.")
        sys.exit(1)
    if not root.exists():
        print(f"Result folder {root} does not exist.")
        sys.exit(2)

    # Initialise optimizer and load height profile
    exp = Experiment.read(root / "parameters.json")
    optimizer = Optimizer(exp)
    count = exp.grid.count
    while count <= 8192:
        plot(optimizer, count, root)
        count *= 2
