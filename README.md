# PyAimRec

## Particle tracking via Adaptive Image Reconstruction

PyAimRec is a Python package for adaptive image-reconstruction–based tracking of spherical colloids in bright-field microscopy images, with a particular focus on situations where particle images strongly overlap and conventional centroid or threshold-based tracking fails.

The method represents each particle by a learned, radially symmetric shape function and iteratively reconstructs the full image from particle positions and shapes. Particle positions are refined through optimisation of the reconstruction error, while particle shapes are updated self-consistently from image residuals. This adaptive procedure enables nanometre-precision localisation even under severe image overlap.

In addition to tracking particle positions, PyAimRec can resolve time-dependent changes in particle shape, making it suitable for studying stimuli-responsive or dynamically evolving colloids. The framework supports both intensity-based and gradient-based reconstruction models and is well suited for quantitative applications such as interparticle distance measurements, interaction-potential extraction, and optical-tweezers experiments involving closely spaced colloidal particles.

It supports two complementary reconstruction engines:

Intensity-based reconstruction (AimRecIntensity)

Gradient-based reconstruction (AimRecGradient)

A lightweight wrapper class (AimRec) provides a unified user-facing interface while keeping all algorithmic logic inside the engine classes.



## Key Features

Adaptive reconstruction of particle images using learned radial shape functions

Robust tracking under particle overlap

Two interchangeable reconstruction modalities:

Intensity (pixel residual–based)

Gradient (vector field–based, baseline-free)

Iterative shape refinement during tracking

Golden-section search–based position optimisation


## Installation

Requirements

Python ≥ 3.10

### Install from GitHub (recommended)

Install the latest version directly from GitHub:

```md
pip install git+https://github.com/genglaodemao/PyAimRec.git
```

### Install from source (development mode)

Clone the repository and install in editable mode:

```md
git clone https://github.com/genglaodemao/PyAimRec.git

cd PyAimRec

pip install -e .
```

### Verify installation

```python
from PyAimRec import AimRec
```
If the import succeeds, PyAimRec is installed correctly.


## Basic Usage

### Intensity-based reconstruction

```python
from PyAimRec import AimRec

from tifffile import imread

im = imread("image.tif")

rec = AimRec(
    im=im,
    mode="intensity",
    kwargs=dict(
        ifbaseline=True,
        Gkernel=0.5,
        Rcut=25,
        ifplot=True,
    ),
)

rec.run()

rec.summary(px_to_nm=73.8)
```

### Gradient-based reconstruction

```python
rec = AimRec(
    im=im,
    mode="gradient",
    kwargs=dict(
        Gkernel=0.5,
        Rcut=25,
        ifplot=True,
    ),
)

rec.run()

rec.summary()
```

### Refinement (optional)

After a full reconstruction, it is often useful to reapeat the reconstruction, starting with the learned shape and re-optimise positions.

Both engines implement run_shape() for this purpose, and the wrapper exposes it directly.

Default behaviour

```python
rec.run_shape(step=0.5, do_plot=True)
```

This enforces internally:

intensity mode:

S = S_accept

PosGuess = Pos_final

gradient mode:

gr = gr_accept

PosGuess = Pos_final



## Tests / Example Scripts

The tests/ directory contains runnable scripts demonstrating and validating the algorithms:

```python
python tests/Runme_AimRec.py

python tests/Runme_Multi.py
```

These scripts serve as:

usage examples

regression tests

debugging / development entry points

They are intentionally verbose and explicit.



## Design Philosophy

Engines own behaviour

AimRecIntensity and AimRecGradient contain all algorithmic logic.

Wrapper routes only

AimRec does not modify state or logic — it only delegates calls.

Explicit over implicit

No hidden state resets. Every workflow step is visible and controllable.

Research-first API

The code prioritises transparency and hackability over consumer convenience.


## Typical Use Cases

Overlapping colloidal particles 

Optical tweezers potential measurements

Thermoresponsive microgels

Active or driven particle systems

High-precision interparticle distance measurements



## Notes

This package is not a general-purpose tracker.

It assumes approximate radial symmetry and reasonable SNR.

Convergence depends on initial guesses and chosen modality.



## Citation

If you use PyAimRec in scientific work, please cite the associated methodological publication (in preparation).
