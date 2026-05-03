##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import torch


class DiffractiveOpticalElement:
    def __init__(self, wavelengths, n, device):
        self.device = device
        self.wavelengths = torch.tensor(wavelengths, device=device, dtype=torch.float32)
        self.delta_n = torch.tensor(n, device=device, dtype=torch.float32) - 1

    def fields_from_height(self, height):
        """ Return DOE source field height profile. """

        phase = (2 * torch.pi / self.wavelengths) * self.delta_n * height[:, :, None]
        return torch.exp(1j * phase)
