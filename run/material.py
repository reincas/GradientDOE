##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Script to store material data files
#
##########################################################################

import json
from pathlib import Path

from gradientdoe.parameter import Parameter

MATERIALS = {
    "IP-S": {
        "description": "Polymer for laser lithography",
        "type": "material",
        "model": "IP-S",
        "manufacturer": "Nanoscribe",
        "sellmeier": {
            "unit": "µm",
            "B1": 0.74320827,
            "C1": 0.00000000,
            "B2": 0.49798931,
            "C2": 0.03596280,
            "B3": 0.00123045,
            "C3": 4.77717770,
        },
    },
}


def write_json(parameter, path):
    """ Stores the Parameter object in a pretty-printed JSON file. """

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(parameter.to_dict(), f, indent=4, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    root = Path("../materials")
    for name in MATERIALS:
        material = Parameter(MATERIALS[name])
        write_json(material, root / f"{name}.json")
