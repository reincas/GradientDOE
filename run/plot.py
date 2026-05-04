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

OPT = 0
M = 2
RS = 0


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


def power_images(Ps, P, pitch, sensor, cmap, names, method, path):
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
                ax.text(xc, yc - radius - (radius * 0.3), f"{P[s, i]*100:.1f} %",
                        color='white', ha='center', va='top', fontsize=8, fontweight='normal')

        # Labels and titles
        ax.set_title(f"Specimen: {names[i]}")
        ax.set_xlabel("x / µm")
        ax.set_ylabel("y / µm")
        if i == Ni - 1:
            plt.colorbar(im, ax=ax, label='Power (relative)')

    # Save image figure
    filepath = path.format(method, "all")
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
    height = np.array(exp.height)

    pitch = exp.grid.pitch
    count = exp.grid.count
    M = height.shape[0] // count
    count *= M
    pitch /= M

    print(f"Grid pitch: {exp.grid.pitch:.1f} µm")
    print(f"Grid count: {exp.grid.count}")
    print(f"Distance: {exp.setup.distance:.0f} µm")

    cmap = "viridis"
    # cmap = "inferno"
    path = "plots/result_{0}_{1}.png"
    name = exp.doe.material.model
    names = [x.model for x in exp.setup.sources]

    AP_list = []

    optimizer.set_grid(count, pitch)
    H, P, Ps = optimizer.step(height, optimizer.asm, count)
    AP_list.append((np.linalg.pinv(P, rcond=1e-2), P))
    power_images(Ps, P, pitch, optimizer.sensor, cmap, names, "opt", path)

    # names = [f"{lam*1000:.3f} nm" for lam in optimizer.doe.wavelengths]
    # power_images(H, None, pitch, optimizer.sensor, cmap, names, "lam", path)

    height_image(height, pitch, cmap, name, "opt", path)

    if OPT:
        count_fab = count * M
        pitch_fab = pitch / M
        height_fab = optimizer.interpolate_height(height, count_fab)
        height_image(height_fab, pitch_fab, cmap, name, "fab", path)

        if RS:
            count_out = count_fab
            pitch_out = pitch_fab
            optimizer.set_grid(count_out, pitch_out)
            rs = RayleighSommerfeldMethod(pitch_fab, pitch_out, exp.setup.distance, optimizer.doe.wavelengths,
                                          optimizer.device)
            H, P, Ps = optimizer.step(height_fab, rs, count_out)
            AP_list.append((np.linalg.pinv(P, rcond=1e-2), P))
            power_images(Ps, P, pitch_out, optimizer.sensor, cmap, names, "rs", path)

        optimizer.set_grid(count_fab, pitch_fab)
        H, P, Ps = optimizer.step(height_fab, optimizer.asm, count_fab)
        AP_list.append((np.linalg.pinv(P, rcond=1e-2), P))
        power_images(Ps, P, pitch_fab, optimizer.sensor, cmap, names, "fab", path)

        np.set_printoptions(formatter=cast(Any, {'float': '{: .3f}'.format}), linewidth=120)
        Ni = len(exp.setup.sources)
        Ns = optimizer.sensor.num_sensors
        for i in range(Ni):
            P = np.empty((len(AP_list), Ns), dtype=float)
            for j, (_, M) in enumerate(AP_list):
                P[j, :] = M[:, i]  # / sum(M[:, i])
            print(f"Specimen {i}:")
            print(P)
        AP = np.empty((Ni, len(AP_list) * Ni), dtype=float)
        for j, (A, P) in enumerate(AP_list):
            AP[:, j * 3:j * 3 + Ni] = A @ P
        print("A @ P:")
        print(AP)
    else:
        np.set_printoptions(formatter=cast(Any, {'float': '{: .3f}'.format}), linewidth=120)
        print(P.T)
        P_norm = P - P.mean(axis=0, keepdims=True)
        print(P_norm.T)

        # masks = optimizer.sensor.masks # (Ns, N, N)
        # power = optimizer.power.detach().cpu().numpy() # (Nk, Nc)
        # print(count)
        # print(power.sum(axis=0))
        # print(np.einsum("sij->s", masks) / count ** 2)
        # print(np.einsum("ijk->k", H))
        # print(np.einsum("sij,ijk->sk", masks, H))
        # print(np.einsum("sij,ijk,kc->sc", masks, H, power))

    path = "plots/result.png"
    store_height(height, 0.0001, path)
