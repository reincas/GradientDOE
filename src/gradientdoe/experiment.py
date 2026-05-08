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
    count: int
    maxHeight: float
    maxHeightUnit: str
    blurRadius: float
    blurRadiusUnit: str

    def __init__(self, data):
        super().__init__(data)
        if self.material is not None:
            self.material = IndexSpectrum(self.material)


class GridParameter(Parameter):
    pitch: float
    pitchUnit: str
    count: int
    countFinal: int


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
    model: str
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

    def __init__(self, data):
        super().__init__(data)
        if self.eta is not None:
            self.eta = Spectrum(self.eta)


class OptParameter(Parameter):
    maxLoops: int
    checkpointThreshold: int
    initialLearningRate: float
    finalLearningRate: float
    maxHeightFactor: float
    maxHeightThreshold: float
    weightOrtho: float
    expOrtho: int
    weightEta: float
    weightGrad: float
    jitter: bool
    ema: Ema

    def __init__(self, data):
        super().__init__(data)
        if self.ema is not None:
            self.ema = Ema(self.ema)


def next_power_of_2(x):
    if x <= 1:
        return 1
    n = math.ceil(x)
    return 1 << (n - 1).bit_length()


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
        logger.info("Initializing experiment")

        # List of wavelengths
        self.setup.wavelengths = wavelengths

        # Spectra of all specimen
        self.setup.sources = spectra
        name = ", ".join([x.model for x in spectra])
        logger.info(f"    Specimen: {name}")

        # Width of the calculation window
        sensor_size = self.sensor.horizontalCount * self.sensor.pitch
        count_final = next_power_of_2(self.doe.count)
        pitch_final = self.doe.pitch
        while count_final < sensor_size / pitch_final:
            count_final *= 2
        logger.info(f"    Calculation window: {count_final * pitch_final * 1e-3:.3f} mm")

        # Initial pixel count
        count_initial = count_final
        pitch_initial = pitch_final
        while self.sensor.diameter / pitch_initial > self.sensor.minOversample:
            count_initial //= 2
            pitch_initial *= 2
        self.grid.pitch = pitch_initial
        self.grid.count = count_initial
        self.grid.countFinal = count_final
        logger.info(f"    Initial pixel count / pitch: {count_initial} / {pitch_initial:.2f} µm")
        logger.info(f"    Final pixel count / pitch: {count_final} / {pitch_final:.2f} µm")

        z = self.setup.distance
        zc = count_initial * pitch_initial ** 2 / min(wavelengths)
        assert z >= zc, f"Propagation distance {z:.0f} µm below minimum for given grid ({zc:.0f} µm)."
        logger.info(f"    Sensor distance: {z * 1e-3:.3f} mm")

        self.doe.material = IndexSpectrum(material, wavelengths)
        mean = np.mean(np.array(self.doe.material.values, dtype=np.float32))
        logger.info(f"    Mean refractive index: {mean:.4f}")

        model = self.sensor.model
        eta = Spectrum.read(f"../sensors/{model}.json")
        eta = spectra[0].interpolate(eta)
        self.sensor.eta = eta
        mean = np.mean(np.array(self.sensor.eta.values, dtype=np.float32))
        logger.info(f"    Mean sensor efficiency: {mean:.3f}")
