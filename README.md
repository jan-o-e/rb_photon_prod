# rb_photon_prod
[![Docker](https://github.com/jan-o-e/rb_photon_prod/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/jan-o-e/rb_photon_prod/actions/workflows/docker-publish.yml)

[![Generic badge](https://img.shields.io/badge/arXiv-2305.04899-<COLOR>.svg)](https://arxiv.org/abs/2305.04899)

[![DOI](https://zenodo.org/badge/679003178.svg)](https://zenodo.org/badge/latestdoi/679003178)

This respository has two parts: A [generalised simulation toolbox](./Source_Code) for atom-laser-cavity interactions with single Rb^87 atoms for Quantum Infrmation Processing and [data](./Plots) accompanying the paper describing an ideal photon burst productions scheme. All the results presented in Ernst et. al. *"Bursts of Polarised Photons from Atom-Cavity Sources"* can be produced with the generalised simulation toolbox. The code has been organised into several classes and files for various parts of the simulation, view the General_Rb_Simulator.ipynb for an example of how to simulate a particular sequence (namely the ideal one documented in the paper), but feel free to construct your own notebooks for your own purposes.

The code presented here was inspired by the original [rb_cqed package for modelling cavity-QED](https://github.com/tomdbar/rb-cqed), the Mathematica scripts used for calculating the Clebsch-Gordan Coefficients and energy level splittings are taken from there and some functions are adapted or taken straight from that repository too - if you will - view is as rbcqedv2.

To run virtual experiments with Rb requires the following:
<details>
<summary>📦 Package Requirements</summary>
   
- qutip==4.7.0
- python==3.6+
- numpy==1.16+
- scipy==1.0+
- matplotlib==1.2.1++
- cython==0.29.20, <30.0.0
- C++ compiler (for mac install the xcode command line tools: xcode-select --install)
- ipython==8+
</details>

Included is an environment.yml file (for conda) as well as a docker image.

# Installation Guide

## Install local poetry environment (recommended)

Follow these steps to create a poetry environment for this project (deps specified for Python 3.8):
Poetry can be installed as documented [here](https://python-poetry.org/docs/).

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/jan-o-e/rb_photon_prod
   cd rb_photon_prod
2. **Setup poetry environment (make sure poetry is installed):**
   ```bash
   poetry install
3. **Activate poetry shell:**
   ```bash
   poetry shell

## Install local conda environment

Follow these steps to create a conda environment for this project:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/jan-o-e/rb_photon_prod
   cd rb_photon_prod
2. **Setup conda environment:**
   ```bash
   conda env create -f environment.yml
3. **Activate conda environmen:**
   ```bash
   conda activate rb_photon_prod

Note that this was tested on a Mac with M1 Apple Silicon chip and I have seen some dependency issues when using the older Macs or other operating systems. You can alternatively follow the [qutip installation guide](https://qutip.org/docs/latest/installation.html) and also install matplotlib, cython (important caveat: [cython version compatibility](https://github.com/qutip/qutip/issues/2198) ) and notebook (jupyter notebook) with a package manager of your choice.

## Run docker container (recommended to avoid dependency issues)

Ther is also a [docker image](https://github.com/jan-o-e/rb_photon_prod/pkgs/container/rb_photon_prod), to install this image locally run:
```bash
docker pull ghcr.io/jan-o-e/rb_photon_prod:main
```

In case of suggestions, suspected errors or enquiries please get in touch with me at jan.ernst@physics.ox.ac.uk, or submit a pull request.

## References

Jan Ole Ernst et al 2023 J. Phys. B: At. Mol. Opt. Phys. 56 205003
