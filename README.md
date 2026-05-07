# GradientDOE 0.2.0

This is a Python 3 package to determine the height profile of a multi-wavelength phase plate or Diffractive Optical
Element (DOE) using the nonlinear optimiser Adam from the PyTorch package. 

## Application

The phase plate diffracts a multi-wavelength plane wave. This forward propagation is simulated using the angular
spectrum method (ASM). A detector array at a certain distance behind the DOE is recording the signal. The height
profile of the phase plate is optimized to get orthogonal detector signals for certain input spectra while maximising
the detector power. The software currently supports only full-window calculation, both DOE and the sensor are thus of
the same size.

## Configuration

Modify the values in the dictionary `EXPERIMENT` on top of the script `run/main.py`.

## Optimize Phase Plates

Optimised height structures will be calculated and stored in the folder `result` when you run

```
python run/main.py result
```

A GPU with 8 GB VRAM is able to run the optimisation for pixel counts up to 4096.

## Plot Results

To generate plots of the height structures, power images in the sensor plane and image files for the grayscale
lithography from Nanoscribe in the folder `result`, run

```
python run/plot.py result
```
