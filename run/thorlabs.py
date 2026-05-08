##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import json
from pathlib import Path
import openpyxl

from gradientdoe.spectrum import Spectrum


def import_neon(file_path):
    """ Import Neon spectral lamp spectrum data, model name, and description from a Thorlabs Excel file. """

    # Load the workbook and select the "Transmission" sheet
    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet = wb["Neon Spectral Lines"]

    # Model string
    model = sheet["B10"].value

    # Description string
    desc_parts = []
    for row in sheet["A8":"B9"]:
        for cell in row:
            if cell.value is not None:
                desc_parts.append(str(cell.value))
    description = " ".join(desc_parts)

    # Spectrum Data starting at C3 and D3
    spectrum_data = []
    for row in sheet.iter_rows(min_row=3, min_col=3, max_col=4):
        wavelength = row[0].value  # Column C
        intensity = row[1].value  # Column D

        # Store values if they are numbers
        if isinstance(wavelength, (int, float)) and isinstance(intensity, (int, float)):
            spectrum_data.append([float(wavelength), float(intensity)])

    # Sort spectrum
    spectrum_data.sort(key=lambda x: x[0])

    spectrum = {
        "description": description,
        "type": "source",
        "model": model,
        "manufacturer": "Thorlabs",
        "data": spectrum_data,
        "dataColumns": ["Wavelength", "Intensity"],
        "dataUnits": ["nm", ""],
    }
    return Spectrum(spectrum)


def import_transmission(file_path):
    """ Import transmission spectrum data, model name, and description from a Thorlabs Excel file. """

    # Load the workbook and select the "Transmission" sheet
    wb = openpyxl.load_workbook(file_path, data_only=True)
    names = wb.sheetnames
    if "Transmission" in names:
        sheet = wb["Transmission"]
        model = sheet["B9"].value
        wavelength_col = 2
        transmission_col = 3

    elif "Sheet1" in names:
        sheet = wb["Sheet1"]
        model = sheet["B9"].value.split(",")[0]
        wavelength_col = 2
        transmission_col = 3

    else:
        sheet = wb.worksheets[0]
        model = sheet["B9"].value
        wavelength_col = 2
        transmission_col = 4

    # Description string
    desc_parts = []
    for row in sheet["A7":"B8"]:
        for cell in row:
            if cell.value is not None:
                desc_parts.append(str(cell.value))
    description = " ".join(desc_parts)

    # Spectrum Data starting at C2 and D2
    spectrum_data = []
    for row in sheet.iter_rows(min_row=2):
        wavelength = row[wavelength_col].value
        transmission = row[transmission_col].value

        # Store values if they are numbers and convert transmission from percent to factor
        if isinstance(wavelength, (int, float)) and isinstance(transmission, (int, float)):
            wavelength = float(wavelength)
            transmission = float(transmission) / 100
            if transmission < 0:
                assert abs(transmission) < 6e-4, f"Transmission: {transmission}"
                transmission = 0.0
            spectrum_data.append([wavelength, transmission])

    # Sort spectrum
    spectrum_data.sort(key=lambda x: x[0])

    # Return spectrum as Parameter object
    spectrum = {
        "description": description,
        "type": "filter",
        "model": model,
        "manufacturer": "Thorlabs",
        "data": spectrum_data,
        "dataColumns": ["Wavelength", "Transmission"],
        "dataUnits": ["nm", ""],
    }
    return Spectrum(spectrum)


def write_spectrum(spectrum, output_path):
    """ Stores the Parameter object in a pretty-printed JSON file. """

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(spectrum.to_dict(), f, indent=4, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    for file_path in Path("../private").glob("*.xlsx"):
        if file_path.name.startswith("csl1"):
            spectrum = import_neon(file_path)
            path = f"../sources/{spectrum.model}.json"
        else:
            spectrum = import_transmission(file_path)
            path = f"../filters/{spectrum.model}.json"

        spectrum.write(path)
        print(f"Converted {spectrum.model} -> {path}")
