##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import logging
import numpy as np
from scipy.interpolate import PchipInterpolator
from typing import cast, Any

from .convert import convert_factor
from .parameter import Parameter

logger = logging.getLogger("spectrum")


class Spectrum(Parameter):
    description: str
    type: str
    model: str
    manufacturer: str
    data: list
    dataColumns: list
    dataUnits: list

    def __init__(self, data):
        if isinstance(data, Parameter):
            data = data.to_dict()
        super().__init__(data)

    def clone(self):
        """ Return a copy of this object. """

        obj = self.__class__.__new__(self.__class__)
        Parameter.__init__(obj, self.to_dict())
        return obj

    @property
    def wavelengths(self):
        return list(list(zip(*self.data))[0])

    @property
    def unit(self):
        return self.dataUnits[0]

    @property
    def values(self):
        return list(list(zip(*self.data))[1])

    @property
    def size(self):
        return len(self.data)

    def to_unit(self, unit):
        if unit == self.unit:
            return
        factor = convert_factor(self.unit, unit, "m")
        wavelengths = [v * factor for v in self.wavelengths]
        self.data = list(zip(wavelengths, self.values))
        self.dataUnits[0] = unit

    def normalise(self):
        total = sum(self.values)
        values = [value / total for value in self.values]
        self.data = list(zip(self.wavelengths, values))

    def interpolate(self, other):
        """ Return other spectrum interpolated to wavelength grid of this spectrum. """

        assert isinstance(other, Spectrum)

        factor = convert_factor(other.unit, self.unit, "m")
        other_wavelengths = np.array(other.wavelengths, dtype=np.float32) * factor
        interpolate = PchipInterpolator(other_wavelengths, other.values)
        values = interpolate(self.wavelengths)

        other.data = list(zip(self.wavelengths, values))
        other.dataUnits[0] = self.unit
        return other


class Source(Spectrum):
    def __init__(self, data, source=None):
        super().__init__(data)

        # Normalize source spectrum
        if source is None:
            assert self.type == "source"

        else:
            assert self.type == "filter"
            assert isinstance(source, Source)
            source.interpolate(self)
            values = np.array(self.values, dtype=np.float32) * source.values
            self.data = list(zip(self.wavelengths, values))
            self.type = "source"
            self.model = f"{source.model} | {self.model}"
            self.description = f"{source.description} | {self.description}"


class Sellmeier(Parameter):
    unit: str
    B1: float
    C1: float
    B2: float
    C2: float
    B3: float
    C3: float


class IndexSpectrum(Spectrum):
    sellmeier: Sellmeier

    def __init__(self, data, wavelengths=None):

        # Prepare parameter dictionary
        if isinstance(data, Parameter):
            data = data.to_dict()
        data["data"] = []
        data["dataColumns"] = ("Wavelength", "Refractive Index")
        data["dataUnits"] = (data["sellmeier"]["unit"], "")

        Parameter.__init__(self, data)
        self.sellmeier = Sellmeier(self.sellmeier)
        if wavelengths is not None:
            self.interpolate(wavelengths)

    def normalise(self):
        raise NotImplementedError()

    def interpolate(self, wavelengths):
        if isinstance(wavelengths, np.ndarray):
            assert len(wavelengths) == 1
        elif isinstance(wavelengths, Spectrum):
            spectrum = wavelengths
            wavelengths = np.array(spectrum.wavelengths, dtype=np.float32)
            if spectrum.unit != self.unit:
                factor = convert_factor(spectrum.unit, self.unit, "m")
                wavelengths *= factor
        else:
            assert isinstance(wavelengths, list), type(wavelengths)
            wavelengths = np.array(wavelengths, dtype=np.float32)

        lam2 = wavelengths ** 2
        if self.unit != self.sellmeier.unit:
            factor = convert_factor(self.unit, self.sellmeier.unit, "m")
            lam2 *= factor ** 2

        values = np.ones(len(wavelengths), dtype=np.float32)
        for i in range(1, 4):
            Bi = getattr(self.sellmeier, f"B{i}")
            Ci = getattr(self.sellmeier, f"C{i}")
            values += Bi * lam2 / (lam2 - Ci)
        values = np.sqrt(values).tolist()
        self.data = list(zip(wavelengths, values))


def opt_spectra(source, models, coverage):
    """ Apply each filter model to the given source spectrum and return each spectrum reduced to the shortest list
    of wavelengths which cover at least the given intensity fraction for each of the spectra."""

    # Source spectrum
    source = Source.read(f"../sources/{source}.json")
    source.normalise()

    # Generate normalized spectra from filtered source
    spectra = {}
    for model in models:
        if model == source.model:
            spectrum = source.clone()
        else:
            spectrum = Source.read(f"../filters/{model}.json", source)
        spectrum.add_parameter("power", sum(spectrum.values))
        spectrum.add_parameter("powerUnit", "")
        spectrum.normalise()
        spectra[model] = spectrum

    # Extract wavelengths and intensities
    wavelengths = np.array(source.wavelengths, dtype=np.float32)
    intensities = np.empty((source.size, len(models)), dtype=np.float32)
    for i, model in enumerate(models):
        intensities[:, i] = spectra[model].values

    # Order wavelength indices by their importance (maximum intensity in any spectrum)
    max_intensity = np.max(intensities, axis=1)
    indices = np.argsort(max_intensity)

    # Reorder wavelengths and spectra by importance
    wavelengths = np.array(wavelengths, dtype=np.float32)[indices]
    max_intensity = max_intensity[indices]
    intensities = intensities[indices, :]

    # Skip all wavelengths which do not contribute to the minimum coverage of the total intensity of any spectrum
    i = np.searchsorted(max_intensity, 1 - coverage)
    wavelengths = wavelengths[i:]
    intensities = intensities[i:, :]

    # Sort by wavelengths
    indices = np.argsort(wavelengths)
    wavelengths = wavelengths[indices]
    intensities = intensities[indices, :]

    # Update all spectra with reduced range of wavelengths and normalise them again
    for i, model in enumerate(models):
        spectra[model].data = list(zip(wavelengths.tolist(), intensities[:, i].tolist()))
        spectra[model].add_parameter("coverage", sum(spectra[model].values))
        spectra[model].normalise()
        spectra[model].to_unit("µm")

    # Return spectra with reduced range of wavelengths
    spectra = list(spectra.values())
    wavelengths = spectra[0].wavelengths
    return wavelengths, spectra


def show_opt(spectra):
    wavelengths = spectra[0].wavelengths
    unit = spectra[0].unit
    assert unit == "µm"
    np.set_printoptions(formatter=cast(Any, {'float': '{: .3f}'.format}), linewidth=120)
    for i in range(len(wavelengths)):
        head = f"{wavelengths[i]:.6f} {unit}:"
        intensities = ", ".join([f"{spectra[j].values[i]:.3f}" for j in range(len(spectra))])
        logger.info(f"{head:12s} {intensities}")
    intensities = ", ".join([f"{spectra[j].coverage:.3f}" for j in range(len(spectra))])
    head = f"coverage[{len(wavelengths)}]:"
    logger.info(f"{head:12s} {intensities}")
