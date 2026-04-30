##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import itertools
from pathlib import Path
import numpy as np

from gradientdoe.spectrum import Source


def get_models():
    return [path.stem for path in Path("../filters").glob("*.json")]


def get_spectra(source, models):
    source = Source.read(f"../sources/{source}.json")
    source.normalise()

    data = np.zeros((source.size, len(models)), dtype=float)
    for i, model in enumerate(models):
        if model == source.model:
            values = source.values
        else:
            values = Source.read(f"../filters/{model}.json", source).values
        data[:, i] = values

    return source.wavelengths, data


def show_power():
    models = get_models()
    wavelengths, spectra = get_spectra("CSL1", models)

    indices = list(range(len(models)))
    indices.sort(key=lambda i: -sum(spectra[:, i]))
    print("Total Intensities:")
    for i in indices:
        print(f"    {(models[i] + ":"):12s} {sum(spectra[:, i]):.3f}")


def show_dot():
    models = get_models()
    wavelengths, spectra = get_spectra("CSL1", models)

    indices = list(range(len(models)))
    indices.sort(key=lambda i: -sum(spectra[:, i]))

    array = np.zeros((len(models), len(models)), dtype=float)
    for row in indices:
        for col in indices:
            row_spectrum = np.sqrt(spectra[:, row] / sum(spectra[:, row]))
            col_spectrum = np.sqrt(spectra[:, col] / sum(spectra[:, col]))
            array[row, col] = row_spectrum @ col_spectrum

    np.set_printoptions(formatter={'float': '{: .3f}'.format}, linewidth=160)
    print("Spectral Scalar Products:")
    for row in range(len(array)):
        model = models[indices[row]]
        print(f"    {(model + ":"):12s} {array[row, :]}")


def show_selection(num):
    models = get_models()
    wavelengths, spectra = get_spectra("CSL1", models)

    indices = list(range(len(models)))
    indices.sort(key=lambda i: -sum(spectra[:, i]))

    array = np.zeros((len(models), len(models)), dtype=float)
    for row in indices:
        for col in indices:
            row_spectrum = np.sqrt(spectra[:, row] / sum(spectra[:, row]))
            col_spectrum = np.sqrt(spectra[:, col] / sum(spectra[:, col]))
            array[row, col] = row_spectrum @ col_spectrum

    matches = []
    for selection in itertools.combinations(indices, num):
        match = max([array[i, j] for i, j in itertools.combinations(selection, 2)])
        intensity = min([sum(spectra[:, i]) for i in selection])
        value = match / intensity
        matches.append((value, match, intensity, selection))
    matches.sort(key=lambda x: -x[1])
    print("Filter Sets:")
    for value, match, intensity, selection in matches:
        names = ", ".join([models[indices[i]] for i in selection])
        print(f"    {names}: {value:.3f} {match:.3f} {intensity:.3f}")


if __name__ == '__main__':
    show_power()
    show_dot()
    #show_selection(3)
