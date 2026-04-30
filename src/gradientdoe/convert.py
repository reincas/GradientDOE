##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

def convert_factor(src_unit, dst_unit, base):
    """ Returns the factor to multiply by to convert from src_unit to dst_unit. """

    prefixes = {
        'a': 1e-18, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6,
        'm': 1e-3, 'c': 1e-2, 'd': 1e-1, '': 1,
        'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12, 'P': 1e15, 'E': 1e18,
    }

    def get_multiplier(unit, base):
        if not unit.endswith(base):
            raise ValueError(f"Unit '{unit}' does not end with base '{base}'")
        prefix = unit[:-len(base)]
        if prefix not in prefixes:
            raise ValueError(f"Unknown SI prefix: '{prefix}'")
        return prefixes[prefix]

    src_mult = get_multiplier(src_unit, base)
    dst_mult = get_multiplier(dst_unit, base)

    return src_mult / dst_mult
