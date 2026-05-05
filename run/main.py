##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import logging

import numpy as np

from gradientdoe.experiment import Experiment
from gradientdoe.optimizer import Optimizer
from gradientdoe.spectrum import opt_spectra, show_opt

logger = logging.getLogger("main")

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
        "data": [],
        "dataColumns": ("Wavelength", "Refractive Index"),
        "dataUnits": ("µm", ""),
    },
}

EXPERIMENT = {
    "doe": {
        "material": None,
        "pitch": 2.0,
        "pitchUnit": "µm",
        "count": 1024,
        "maxHeight": 6,
        "maxHeightUnit": "µm",
    },
    "grid": {
        "pitch": 0,
        "pitchUnit": "µm",
        "count": 0,
        "countFinal": 0,
    },
    "setup": {
        "wavelengths": [],
        "wavelengthsUnit": "µm",
        "sources": [],
        "distance": 150000.0,
        "distanceUnit": "µm",
    },
    "sensor": {
        "horizontalCount": 2,
        "verticalCount": 2,
        "pitch": 1000,
        "pitchUnit": "µm",
        "diameter": 250,
        "diameterUnit": "µm",
        "fuzzyRadius": 5,
        "fuzzyRadiusUnit": "µm",
        "skipCenter": True,
        "eta": None,
        "minOversample": 16,
        "oversample": 0,
    },
    "optimizer": {
        "maxLoops": 1000000,
        "learningRate": 0.05,
        "weightOrtho": 2.0,
        "expOrtho": 2,
        "weightEta": 50.0,
        "jitter": True,
        "ema": {
            "patience": 200,
            "threshold": 1e-4,
            "alpha": 0.05,
            "loss": None,
            "bestLoss": None,
        },
    }
}


def init_logger():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file = "gradientdoe.log"
    file_h = logging.FileHandler(file, mode="a")
    file_h.setFormatter(log_format)
    file_h.setLevel(logging.DEBUG)
    root.addHandler(file_h)

    console_h = logging.StreamHandler()
    console_h.setFormatter(log_format)
    console_h.setLevel(logging.DEBUG)
    root.addHandler(console_h)


if __name__ == '__main__':
    init_logger()

    # Artificial specimen spectra
    src_model = "CSL1"
    models = ["FBH05590-10", "FBH05620-10", "FBH05640-10"]
    coverage = 0.99
    wavelengths, spectra = opt_spectra(src_model, models, coverage)
    show_opt(spectra)

    # DOE material
    material = MATERIALS["IP-S"]

    # Experimental setup
    exp = Experiment(EXPERIMENT)
    exp.adjust_parameters(wavelengths, spectra, material)

    # Determine suitable height profile
    count = exp.grid.count
    pitch = exp.grid.pitch
    optimizer = Optimizer(exp)
    height = optimizer.init_height()
    height = optimizer.run(height)
    exp.add_parameter(f"height.{count}", height.tolist())
    while count < exp.grid.countFinal:
        count *= 2
        pitch /= 2
        height = optimizer.interpolate_height(height, count)
        print(f"Interpolation to {count} pixel: height = {np.min(height):.2f} - {np.max(height):.2f} µm")
        height /= np.max(height) * optimizer.h_max
        #height = optimizer.clip_height(height, 0.01)
        optimizer.set_grid(count, pitch)
        height = optimizer.run(height)
        exp.add_parameter(f"height.{count}", height.tolist())

    # Store result
    path = "result.json"
    exp.write(path)
    logger.info(f"Optimization result stored in {path}")
