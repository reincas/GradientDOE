##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import logging
import math

import numpy as np

from gradientdoe.optimizer import Ema
from gradientdoe.parameter import Parameter
from gradientdoe.spectrum import IndexSpectrum, Spectrum, Source

logger = logging.getLogger("experiment")


class DoeParameter(Parameter):
    material: IndexSpectrum
    pitch: float
    pitchUnit: str
    size: float
    sizeUnit: str
    maxHeight: float
    maxHeightUnit: str

    def __init__(self, data):
        super().__init__(data)
        if self.material is not None:
            self.material = IndexSpectrum(self.material)


class GridParameter(Parameter):
    pitch: float
    pitchUnit: str
    count: int


class SetupParameter(Parameter):
    wavelengths: list
    wavelengthsUnit: str
    sources: list
    distance: float
    distanceUnit: str

    def __init__(self, data):
        super().__init__(data)
        self.sources = [Source(x) for x in self.sources]


class SensorParameter(Parameter):
    horizontalCount: int
    verticalCount: int
    pitch: float
    pitchUnit: str
    diameter: float
    diameterUnit: str
    fuzzyRadius: float
    fuzzyRadiusUnit: str
    skipCenter: bool
    eta: Spectrum
    minOversample: int
    oversample: int

    def __init__(self, data):
        super().__init__(data)
        if self.eta is not None:
            self.eta = Spectrum(self.eta)


class OptParameter(Parameter):
    maxLoops: int
    learningRate: float
    weightOrtho: float
    weightEta: float
    weightCenter: float
    ema: Ema

    def __init__(self, data):
        super().__init__(data)
        if self.ema is not None:
            self.ema = Ema(self.ema)


class Experiment(Parameter):
    doe: DoeParameter
    grid: GridParameter
    setup: SetupParameter
    sensor: SensorParameter
    optimizer: OptParameter

    def __init__(self, data):
        super().__init__(data)
        self.doe = DoeParameter(self.doe)
        self.grid = GridParameter(self.grid)
        self.setup = SetupParameter(self.setup)
        self.sensor = SensorParameter(self.sensor)
        self.optimizer = OptParameter(self.optimizer)

    def adjust_parameters(self, wavelengths, spectra, material):
        logger.debug("Initializing experiment")

        # List of wavelengths
        self.setup.wavelengths = wavelengths

        # Spectra of all specimen
        self.setup.sources = spectra
        name = ", ".join([x.model for x in spectra])
        logger.debug(f"    Specimen: {name}")

        # Width of the calculation window
        w = max(self.doe.size, self.sensor.horizontalCount * self.sensor.pitch,
                self.sensor.verticalCount * self.sensor.pitch)
        logger.debug(f"    Calculation window: {w * 1e-3:.3f} mm")

        # Determine power-of-2 pixel count based on minimum given
        N = self.sensor.minOversample * w / self.sensor.diameter
        N = 2 ** math.ceil(math.log2(N))
        self.grid.count = N
        logger.debug(f"    Pixel count: {N}")

        p = w / N
        self.grid.pitch = p
        logger.debug(f"    Pixel size: {p:.2f} µm")

        Ne = N * self.sensor.diameter / w
        self.sensor.oversample = Ne
        logger.debug(f"    Sensor oversample: {Ne:.1f}")

        z = N * p ** 2 / (2 * max(wavelengths))
        self.setup.distance = z
        logger.debug(f"    Sensor distance: {z * 1e-3:.3f} mm")

        self.doe.material = IndexSpectrum(material, wavelengths)
        mean = np.mean(np.array(self.doe.material.values))
        logger.debug(f"    Mean refractive index: {mean:.4f}")

        model = "a2A3536-31umBAS"
        eta = Spectrum.read(f"../sensors/{model}.json")
        eta = spectra[0].interpolate(eta)
        self.sensor.eta = eta
        mean = np.mean(np.array(self.sensor.eta.values))
        logger.debug(f"    Mean sensor efficiency: {mean:.3f}")
