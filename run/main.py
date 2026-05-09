##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# Requires a command line argument with the result folder.
#
##########################################################################

import h5py
import logging
import math
from pathlib import Path
import sys
from scidatacontainer import Container
from zipfile import ZIP_STORED

from gradientdoe import __version__ as software_version
from gradientdoe.experiment import Experiment
from gradientdoe.optimizer import Optimizer
from gradientdoe.optimizer import __version__ as optimizer_version
from gradientdoe.spectrum import opt_spectra, show_opt

logger = logging.getLogger("main")

# Pixel count which requires checkpoint on the Adam optimiser
CHECKPOINT = 4 * 1024

EXPERIMENT = {
    "doe": {
        "material": "IP-S",
        "refractiveIndex": None,  # Determined by Experiment.adjust_parameters()
        "pitch": 0.25,
        "pitchUnit": "µm",
        "count": 8 * 1024,
        "maxHeight": 6,
        "maxHeightUnit": "µm",
        "blurRadius": 0.5,
        "blurRadiusUnit": "µm",
    },
    "grid": {
        "pitch": 0,  # Determined by Experiment.adjust_parameters()
        "pitchUnit": "µm",
        "count": 0,  # Determined by Experiment.adjust_parameters()
        "countFinal": 0,  # Determined by Experiment.adjust_parameters()
    },
    "setup": {
        "wavelengths": [],  # Determined by Experiment.adjust_parameters()
        "wavelengthsUnit": "µm",
        "sources": [],  # Determined by Experiment.adjust_parameters()
        "distance": 150000.0,
        "distanceUnit": "µm",
    },
    "sensor": {
        "model": "a2A3536-31umBAS",
        "eta": None,  # Determined by Experiment.adjust_parameters()
        "horizontalCount": 2,
        "verticalCount": 2,
        "pitch": 1000,
        "pitchUnit": "µm",
        "diameter": 250,
        "diameterUnit": "µm",
        "fuzzyRadius": 5,
        "fuzzyRadiusUnit": "µm",
        "skipCenter": True,
        "minOversample": 16,
    },
    "optimizer": {
        "version": optimizer_version,
        "maxLoops": 1000000,
        "initialLearningRate": 0.05,
        "finalLearningRate": 0.05,
        "maxHeightFactor": 0.02,
        "maxHeightThreshold": 0.005,
        "weightOrtho": 5.0,
        "expOrtho": 2,
        "weightEta": 50.0,
        "weightGrad": 50.0,
        "jitter": False,
        "ema": {
            "patience": 200,
            "threshold": 1e-2,
            "alpha": 0.2,
        },
    }
}


def init_logger(file_name, level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_h = logging.FileHandler(file_name, mode="a")
    file_h.setFormatter(log_format)
    file_h.setLevel(level)
    root.addHandler(file_h)

    console_h = logging.StreamHandler()
    console_h.setFormatter(log_format)
    console_h.setLevel(level)
    root.addHandler(console_h)


def write_height(height, name, attrs, path):
    """ Store height profile in HDF5 file. """

    with h5py.File(path, "a") as fp:
        if name in fp:
            del fp[name]
        dataset = fp.create_dataset(name, data=height, dtype='float32')
        dataset.attrs.update(attrs)
        # logger.info(f"Height profile {name} stored in {path}")


def get_path():
    """ Returns the first command line argument as a Path object. """
    try:
        return Path(sys.argv[1])
    except IndexError:
        return None


if __name__ == '__main__':
    root = get_path()
    if root is None:
        print("Result folder required as command line argument.")
        sys.exit(1)

    dc_path = Path(str(root) + ".zdc")
    if dc_path.exists():
        print(f"Data container {dc_path} already exists.")
        sys.exit(2)

    if root.exists():
        print(f"Folder {root} already exists.")
        sys.exit(3)

    root.mkdir(parents=True)
    log_path = root / "tmp_main.log"
    init_logger(log_path, logging.INFO)

    # Artificial specimen spectra
    src_model = "CSL1"
    models = ["FBH05590-10", "FBH05620-10", "FBH05640-10"]
    coverage = 0.99
    wavelengths, spectra = opt_spectra(src_model, models, coverage)
    show_opt(spectra)

    # Experimental setup
    exp = Experiment(EXPERIMENT)
    exp.adjust_parameters(wavelengths, spectra)
    exp_path = root / "tmp_parameters.json"
    exp.write(exp_path)
    # logger.info(f"Optimization parameters stored in {exp_path}")

    # Determine suitable height profile
    count = exp.grid.count
    pitch = exp.grid.pitch
    steps = math.log2(exp.grid.countFinal) - math.log2(count) + 1

    initial_learning_rate = exp.optimizer.initialLearningRate
    final_learning_rate = exp.optimizer.finalLearningRate
    rate_base = 10 ** (math.log10(final_learning_rate / initial_learning_rate) / (steps - 1))

    height_path = root / "tmp_height.hdf5"
    height = None
    opt_height = None
    opt_count = None
    opt_name = None
    i = 0
    while count <= exp.grid.countFinal:
        optimizer = Optimizer(exp, checkpoint=CHECKPOINT)
        name = f"height_{count:06d}"
        if count <= CHECKPOINT:
            if height is None:
                src = "random"
                height = optimizer.init_height()
            else:
                src = opt_name
                height = optimizer.interpolate_height(height, count)
            optimizer.set_grid(count, pitch)
            learning_rate = initial_learning_rate * rate_base ** i
            height = optimizer.run(height, learning_rate)
            opt_height = height
            opt_count = count
            opt_name = name
            mode = "optimized"
        else:
            src = opt_name
            height = optimizer.interpolate_height(opt_height, count)
            mode = "interpolated"
        attrs = {
            "count": count,
            "pitch": pitch,
            "unit": exp.doe.pitchUnit,
            "mode": mode,
            "source": src,
        }
        write_height(height, name, attrs, height_path)
        count *= 2
        pitch /= 2
        i += 1

    logger.info(f"Preparing data container file: {dc_path}")
    logging.shutdown()
    with open(log_path, "r") as fp:
        logs = fp.read()

    items = {
        "content.json": {
            "containerType": {"name": "ameliOperator"},
            "usedSoftware": [{"name": "GradientDOE", "version": software_version,
                              "id": "https://github.com/reincas/GradientDOE", "idType": "URL"}],
        },
        "meta.json": {
            "title": "GradientDOE",
            "description": "Optimized height profile of a optical multi-wavelength phase plate",
            "license": "cc-by-sa-4.0",
        },
        "data/parameters.json": exp.to_dict(),
        "data/main.log": logs,
        "data/height.hdf5": {"path": height_path, "compression": ZIP_STORED}
    }
    dc = Container(items=items)
    with h5py.File(height_path, 'a') as f:
        f.attrs["refUUID"] = dc.uuid
    dc.write(str(dc_path))

    exp_path.unlink()
    print(f"Deleted {exp_path}")
    log_path.unlink()
    print(f"Deleted {log_path}")
    height_path.unlink()
    print(f"Deleted {height_path}")
    root.rmdir()
    print(f"Deleted {root}")
