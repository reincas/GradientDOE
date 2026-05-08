##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Requires a command line argument with the result folder.
#
##########################################################################

import h5py
import logging
import math
from matplotlib import pyplot as plt, patches as patches
import numpy as np
from pathlib import Path
from PIL import Image, PngImagePlugin
import sys

from gradientdoe.experiment import Experiment
from gradientdoe.optimizer import Optimizer

logger = logging.getLogger("plot")


def init_logger(root_path, level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file = root_path / "plot.log"
    file_h = logging.FileHandler(file, mode="a")
    file_h.setFormatter(log_format)
    file_h.setLevel(level)
    root.addHandler(file_h)

    console_h = logging.StreamHandler()
    console_h.setFormatter(log_format)
    console_h.setLevel(level)
    root.addHandler(console_h)

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


def store_height_profile(height, path, uuid=None):
    """ Store the height profile as a 16-bit PNG with optional reference UUID. """

    step_size = get_next_preferred_number(np.max(height) / (2 ** 16 - 1))
    h_int16 = np.round(height / step_size).astype(np.uint16)
    assert np.max(h_int16) < 2 ** 16 - 1

    img = Image.fromarray(h_int16)
    if uuid is None:
        img.save(path, format="PNG", compress_level=6)
    else:
        meta = PngImagePlugin.PngInfo()
        meta.add_text("UUID", uuid)
        img.save(path, format="PNG", compress_level=6, pnginfo=meta)
    return step_size


def plot(optimizer, count, root):
    exp = optimizer.exp
    if not isinstance(root, Path):
        root = Path(root)

    # Height profile
    height_path = root / "height.h5"
    height = read_height(height_path, count)
    logger.info(f"Phase plate height profile:")

    # Pixel pitch
    M = count // exp.grid.count
    pitch = exp.grid.pitch / M
    logger.info(f"    Pixel pitch:     {pitch:.1f} µm")
    logger.info(f"    Pixel count:     {count}")
    logger.info(f"    Sensor distance: {exp.setup.distance / 1000:.1f} mm")
    logger.info(f"    Heights:         {np.min(height):.2f} - {np.max(height):.2f} µm")

    # Prepare diagram formatting
    cmap = "viridis"
    # cmap = "inferno"

    # Store height profile as 16-bit PNG image
    path = root / f"profile_{count}.png"
    step_size = store_height_profile(height, path)
    logger.info(f"    Stored fabrication file: {path} with step size: {step_size} µm")

    # Store height profile as plot
    name = exp.doe.material.model
    path = root / f"height_{count}.png"
    store_height_plot(height, pitch, cmap, name, path)
    logger.info(f"    Stored height profile image: {path}")

    # Sensor power for every specimen
    logger.info(f"    Calculation device: {optimizer.device.type}")
    optimizer.set_grid(count, pitch)
    P, Ps = optimizer.step(height, optimizer.asm, count)
    names = [x.model for x in exp.setup.sources]
    assert len(names) == P.shape[1]
    size = max(len(name) for name in names)
    fmt = f"    {{0:{size}s}}: {{1}}"
    for i, name in enumerate(names):
        values = ", ".join(f"{x:6.3f}" for x in P[:, i])
        logger.info(fmt.format(name, values))

    # Store power image with sensor outlines
    path = root / f"power_{count}.png"
    store_power_plots(Ps, P, pitch, optimizer.sensor, cmap, names, path)
    logger.info(f"    Stored sensor power image: {path}")


def get_path():
    """ Returns the first command line argument as a Path object. """
    try:
        return Path(sys.argv[1])
    except IndexError:
        return None


if __name__ == "__main__":
    root = get_path()
    if root is None:
        logger.info("Result folder required as command line argument.")
        sys.exit(1)
    if not root.exists():
        logger.info(f"Result folder {root} does not exist.")
        sys.exit(2)

    init_logger(root)

    # Initialise optimizer and load height profile
    exp = Experiment.read(root / "parameters.json")
    count = exp.grid.count
    while count <= exp.grid.countFinal:
        device = None if count < exp.optimizer.checkpointThreshold else "cpu"
        optimizer = Optimizer(exp, device)
        plot(optimizer, count, root)
        count *= 2
