##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import numpy as np
from scipy.special import erf


def get_center(sensor):
    """ Get center coordinates of all sensors. """

    # Center positions of the sensor array
    x_center = (np.arange(sensor.horizontalCount) - (sensor.horizontalCount - 1) / 2) * sensor.pitch
    y_center = (np.arange(sensor.verticalCount) - (sensor.verticalCount - 1) / 2) * sensor.pitch

    # List or sensor coordinates
    centers = []
    for yc in y_center:
        for xc in x_center:
            if sensor.skipCenter and np.isclose(xc, 0) and np.isclose(yc, 0):
                continue
            centers.append([xc, yc])

    # Return sensor coordinates
    return centers


def get_distance(grid, sensor):
    """ Generate a 3D stack of distances to the center of each sensor. """

    # Sanity checks
    unit = grid.pitchUnit
    assert sensor.pitchUnit == unit
    assert sensor.diameterUnit == unit

    p = grid.pitch
    N = grid.count

    # Mesh grid of coordinates
    limit = (N * p) / 2
    coords = np.linspace(-limit, limit, N)
    x, y = np.meshgrid(coords, coords)

    # Sensor center coordinates
    centers = get_center(sensor)
    Ns = len(centers)

    # Build fuzzy masks
    distance = np.zeros((Ns, N, N), dtype=float)
    for s in range(Ns):
        xc, hc = centers[s]
        distance[s] = np.sqrt((x - xc) ** 2 + (y - hc) ** 2)

    # Return stack of masks (Ns, N, N)
    return distance


def next_distance(grid, sensor):
    """ Distance the to the nearest sensor center map. """

    distance = get_distance(grid, sensor)
    return distance.min(axis=0)


def get_masks(grid, sensor):
    """ Generate a 3D stack of fuzzy sensor masks. """

    # Initialise sensor masks by distance to the sensor center
    masks = get_distance(grid, sensor)

    # Sensor radius
    radius = sensor.diameter / 2
    fuzz = sensor.fuzzyRadius

    # Build fuzzy masks
    Ns = masks.shape[0]
    for s in range(Ns):
        r = masks[s]
        masks[s] = 0.5 * (1 - erf((r - radius) / (fuzz / 2)))

    # Return stack of masks (Ns, N, N)
    return masks
