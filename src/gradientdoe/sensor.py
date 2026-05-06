##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import numpy as np
from scipy.special import erf


class SensorArray:
    def __init__(self, sensor):
        self.count_x = sensor.horizontalCount
        self.count_y = sensor.verticalCount
        self.pitch = sensor.pitch
        self.diameter = sensor.diameter
        self.fuzzy_radius = sensor.fuzzyRadius
        self.skip_center = sensor.skipCenter

        self.centers = self.get_centers()
        self.num_sensors = len(self.centers)

        self.distance = None
        self.next_distance = None
        self.masks = None
        self.area_ratio = None

    def set_grid(self, count, pitch):
        self.distance = self.get_distance(count, pitch)
        self.next_distance = self.distance.min(axis=0)
        self.masks = self.get_masks()
        self.area_ratio = np.sum(self.masks[0]) / count ** 2

    def get_centers(self):
        """ Get center coordinates of all sensors. """

        # Center positions of the sensor array
        x_center = (np.arange(self.count_x, dtype=np.float32) - (self.count_x - 1) / 2) * self.pitch
        y_center = (np.arange(self.count_y, dtype=np.float32) - (self.count_y - 1) / 2) * self.pitch

        # List or sensor coordinates
        centers = []
        for yc in y_center:
            for xc in x_center:
                if self.skip_center and np.isclose(xc, 0) and np.isclose(yc, 0):
                    continue
                centers.append([xc, yc])

        # Return sensor coordinates
        return centers

    def get_distance(self, count, pitch):
        """ Generate a 3D stack of distances to the center of each sensor. """

        # Mesh grid of coordinates
        limit = (count * pitch) / 2
        coords = np.linspace(-limit, limit, count, dtype=np.float32)
        x, y = np.meshgrid(coords, coords)

        # Sensor center coordinates
        Ns = len(self.centers)

        # Build distance stack
        distance = np.empty((Ns, count, count), dtype=np.float32)
        for s in range(Ns):
            xc, hc = self.centers[s]
            distance[s] = np.sqrt((x - xc) ** 2 + (y - hc) ** 2)

        # Return distance stack (Ns, N, N)
        return distance

    def get_masks(self):
        """ Generate a 3D stack of fuzzy sensor masks. """

        # Build fuzzy masks
        masks = np.empty(self.distance.shape, dtype=np.float32)
        Ns = masks.shape[0]
        for s in range(Ns):
            r = self.distance[s]
            masks[s] = 0.5 * (1 - erf((r - self.diameter / 2) / (self.fuzzy_radius / 2)))

        # Return stack of masks (Ns, N, N)
        return masks
