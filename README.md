# GradientDOE 0.2.0

This is a Python 3 package to determine the height profile of a multi-wavelength Diffractive Optical Element (DOE)
using the nonlinear optimiser Adam from the PyTorch package. The software is currently running either on a CPU or
the GPU of my RTX 3070, CUDA capability 8.6, with 8 GB of VRAM.

## Configuration

Modify the values in the dictionary `EXPERIMENT` on top of the script `run/main.py`.

## Optimize Phase Maps (DOE)

When you run

```
python run/main.py result
```

optimised height structures will be calculated and stored in the folder `result`.

## Plot Results

Run

```
python run/plot.py result
```

to generate plots of the height structures, power images in the sensor plane and image files for the grayscale
lithography from Nanoscribe in the folder `result`.
