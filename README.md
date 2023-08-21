# rb_photon_prod
<div style="border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;">
    <p><strong>📄 Corresponding Paper:</strong> <a href="https://arxiv.org/abs/2305.04899">Bursts of Polarised Photons from Atom-Cavity Sources</a></p>
</div>

This respository has two parts: A [generalised simulation toolbox](./Source_Code) for atom-laser-cavity interactions with single Rb^87 atoms for Quantum Infrmation Processing and [data](./Plots) accompanying the paper describing an ideal photon burst productions scheme. All the results presented in Ernst et. al. *"Bursts of Polarised Photons from Atom-Cavity Sources"* can be produced with the generalised simulation toolbox. The code has been organised into several classes and files for various parts of the simulation, view the General_Rb_Simulator.ipynb for an example of how to simulate a particular sequence (namely the ideal one documented in the paper), but feel free to construct your own notebooks for your own purposes.

The code presented here was inspired by the original [rb_cqed package for modelling cavity-QED](https://github.com/tomdbar/rb-cqed), the Mathematica scripts used for calculating the Clebsch-Gordan Coefficients and energy level splittings are taken from there and some functions are adapted or taken straight from that repository too - if you will - view is as rbcqedv2.

To run virtual experiments with Rb requires the following:
<summary>📦 Package Requirements</summary>

```plaintext
- qutip==4.7.0
- python==3.6+
- numpy==1.16+
- scipy==1.0+
- matplotlib==1.2.1++
- cython==0.29.20, <3.0.0
- C++ compiler (for mac install the xcode command line tools: xcode-select --install)
- ipython==8+
- itertools
- functools
```
Included is an environment.yml file (for conda). I would highly recommend using the newest version of conda on your machine and running the following for setting up a conda environment.

## Installation Guide

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

Note that this was tested on a Mac with M1 Apple Silicon chip and I have seen some dependency issues when using the older Macs or other operating systems. In that case follow the [qutip installation guide](https://qutip.org/docs/latest/installation.html) and also install matplotlib, cython (important caveat [cython version compatibility](https://github.com/qutip/qutip/issues/2198) and notebook (jupyter notebook) with an environment manager like venv or conda.

In case of suggestions, suspected errors or enquiries please get in touch with me at jan.ernst@physics.ox.ac.uk, or submit a pull request.
