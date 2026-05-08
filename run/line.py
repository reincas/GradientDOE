##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Plot a given row segment for all resolutions
#
##########################################################################

import h5py
import matplotlib.pyplot as plt
from pathlib import Path

from gradientdoe.experiment import next_power_of_2


def pow2range(start, stop):
    count = next_power_of_2(start)
    counts = [count]
    while count < stop:
        count *= 2
        counts.append(count)
    return counts


def get_pos(count, pitch, i):
    start_pos = -count * pitch / 2
    end_pos = count * pitch / 2
    step = (end_pos - start_pos) / (count - 1)
    return start_pos + (i * step)


def hdf5_rows(root, counts, pitch, size, off=0.0):
    with h5py.File(root / "height.h5", 'r') as h5_file:
        lines = []
        for N, count in enumerate(counts):
            name = f"height_{count}"
            imin = round(count * ((1 - size) / 2 + off))
            imax = round(count * ((1 + size) / 2 + off))
            x = [get_pos(count, pitch, i) for i in range(imin, imax)]
            y = h5_file[name][count // 2, imin:imax] + 3.0 * N
            lines.append((x, y))
            pitch /= 2
    return lines


def plot_hdf5_row(lines, path=None):
    plt.figure(figsize=(10, 10))
    for x, y in lines:
        plt.plot(x, y, color='blue', marker='x', markeredgecolor='red', linestyle='-', markersize=6)
    #plt.title(f"Dataset: {dataset_name} | Row Index: {row_index}")
    plt.xlabel("Position / µm")
    plt.ylabel("Height / µm")
    plt.grid(True, linestyle='--', alpha=0.7)
    if path is not None:
        plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    root = Path("result_18")
    size = 0.05
    counts = pow2range(128, 8*1024)
    lines = hdf5_rows(root, counts, 16.0, size, off=-0.2)
    plot_hdf5_row(lines, root)
    plot_hdf5_row(lines, root / "lines.png")