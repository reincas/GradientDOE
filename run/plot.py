##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from matplotlib import pyplot as plt, patches as patches
import numpy as np
from PIL import Image
from typing import cast, Any

from gradientdoe.experiment import Experiment
from gradientdoe.optimizer import Optimizer
from gradientdoe.propagate import RayleighSommerfeldMethod, AngularSpectrumMethod

M_FAB = []  # 2, 4, 8]
RS = 0
STORE_WL = False


def store_height_plot(height, pitch, cmap, name, method, path):
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
    filepath = path.format(method, f"{count}")
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Stored height profile image: {filepath}")


def store_power_plots(Ps, P, pitch, sensor, cmap, names, method, path):
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
    filepath = path.format(method, f"{count}")
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Stored sensor power image: {filepath}")


def store_height_profile(height, step_size, path):
    """ Store the height profile as a 16-bit PNG. """

    # Represent height profile as integer
    h_scaled = np.round(height / step_size)
    h_int16 = np.clip(h_scaled, 0, 2 ** 16 - 1).astype(np.uint16)

    # Store image
    img = Image.fromarray(h_int16)
    count = height.shape[0]
    filepath = path.format(f"{count}")
    img.save(filepath, format="PNG", compress_level=6)
    print(f"    Fabrication file: {path}")
    print(f"    Maximum value: {np.max(h_int16)}")
    assert np.max(h_int16) < 2 ** 16 - 1


if __name__ == "__main__":
    np.set_printoptions(formatter=cast(Any, {'float': '{: .3f}'.format}), linewidth=120)

    # Initialise optimizer and load height profile
    exp = Experiment.read("result.json")
    optimizer = Optimizer(exp)
    height = np.array(exp.height)

    pitch = exp.grid.pitch
    count = exp.grid.count
    M = height.shape[0] // count
    count *= M
    pitch /= M

    print(f"Height profile:")
    print(f"    Grid pitch: {pitch:.1f} µm")
    print(f"    Grid count: {count}")
    print(f"    Distance: {exp.setup.distance / 1000:.1f} mm")

    # Prepare diagram formatting and storage
    cmap = "viridis"
    # cmap = "inferno"
    height_path = "plots/height_{0}_{1}.png"
    power_path = "plots/power_{0}_{1}.png"
    spectrum_path = "plots/spectrum_{0}_{1}.png"
    profile_path = "plots/profile_{0}.png"
    name = exp.doe.material.model
    names = [x.model for x in exp.setup.sources]

    print(f"Optimised height profile ({count} pixels):")
    filepath = "plots/result.png"
    store_height_profile(height, 0.0001, profile_path)
    store_height_plot(height, pitch, cmap, name, "opt", height_path)

    optimizer.set_grid(count, pitch)
    H, P, Ps = optimizer.step(height, optimizer.asm, count)
    print("    " + str(P.T).replace("\n", "\n    "))
    store_power_plots(Ps, P, pitch, optimizer.sensor, cmap, names, "opt", power_path)

    if STORE_WL:
        names = [f"{lam*1000:.3f} nm" for lam in optimizer.doe.wavelengths]
        store_power_plots(H, None, pitch, optimizer.sensor, cmap, names, "opt", spectrum_path)

    for M in M_FAB:
        count_fab = count * M
        pitch_fab = pitch / M

        print(f"Interpolated height profile ({count_fab} pixels):")
        height_fab = optimizer.interpolate_height(height, count_fab)
        #height_fab = optimizer.clip_height(height_fab, 0.01)
        store_height_profile(height_fab, 0.0001, profile_path)
        store_height_plot(height_fab, pitch_fab, cmap, name, "ip", height_path)

        print(f"    ASM propagation")
        optimizer.set_grid(count_fab, pitch_fab)
        H, P, Ps = optimizer.step(height_fab, optimizer.asm, count_fab)
        print("    " + str(P.T).replace("\n", "\n    "))
        store_power_plots(Ps, P, pitch_fab, optimizer.sensor, cmap, names, "asm", power_path)

        if RS:
            count_out = count_fab
            pitch_out = pitch_fab

            print(f"    RS propagation")
            optimizer.set_grid(count_out, pitch_out)
            rs = RayleighSommerfeldMethod(pitch_fab, pitch_out, exp.setup.distance, optimizer.doe.wavelengths,
                                          optimizer.device)
            H, P, Ps = optimizer.step(height_fab, rs, count_out)
            print("    " + str(P.T).replace("\n", "\n    "))
            store_power_plots(Ps, P, pitch_out, optimizer.sensor, cmap, names, "rs", power_path)

    # masks = optimizer.sensor.masks # (Ns, N, N)
    # power = optimizer.power.detach().cpu().numpy() # (Nk, Nc)
    # print(count)
    # print(power.sum(axis=0))
    # print(np.einsum("sij->s", masks) / count ** 2)
    # print(np.einsum("ijk->k", H))
    # print(np.einsum("sij,ijk->sk", masks, H))
    # print(np.einsum("sij,ijk,kc->sc", masks, H, power))
