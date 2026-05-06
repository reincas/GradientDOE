##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import torch


class DiffractiveOpticalElement:
    def __init__(self, wavelengths, n, device):
        # Total memory allocation: ~0

        self.device = device
        self.wavelengths = torch.tensor(wavelengths, device=device, dtype=torch.float32)
        self.delta_n = torch.tensor(n, device=device, dtype=torch.float32) - 1

    def fields_from_height(self, height):
        """ Return DOE source field height profile. """

        # Dimension hint:    float(N, N) -> complex(N, N, k)
        # Memory allocation: 1152 MB = 4k * 4k * 9 * 8 (U)
        U = torch.exp(1j * (2 * torch.pi / self.wavelengths) * self.delta_n * height[:, :, None])
        return U
