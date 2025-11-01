# rb_photon_prod

[![arXiv](https://img.shields.io/badge/arXiv-2305.04899-b31b1b.svg)](https://arxiv.org/abs/2305.04899)
[![DOI](https://img.shields.io/badge/DOI-10.1088%2F1361--6455%2Facf79e-blue.svg)](https://doi.org/10.1088/1361-6455/acf79e)

## Overview

Atom-cavity systems provide robust platforms for quantum information processing, offering strong light-matter coupling and efficient photon collection. This repository presents a generalised simulation toolbox for modeling atom-laser-cavity interactions with single Rb⁸⁷ atoms, specifically designed for investigating photon generation schemes.

The simulation framework enables comprehensive modeling of cavity quantum electrodynamics (cQED) experiments, including full hyperfine structure, arbitrary pulse sequences, spontaneous emission, and photon correlation functions. This toolbox was developed to explore optimal schemes for generating bursts of polarised single photons from atom-cavity sources, as well as investigating the generation of time-bin entangled photonic states.

All results presented in Ernst et al. *"Bursts of Polarised Photons from Atom-Cavity Sources"* (J. Phys. B: At. Mol. Opt. Phys. **56** 205003, 2023) as well as in *Controlling Quantum Systems at the Pulse Level: Cavity QED & Beyond (forthcoming)* 2025 can be reproduced using this simulation toolbox. The code has been organised into modular classes and functions for different aspects of the simulation, making it straightforward to construct custom pulse sequences and explore novel quantum protocols.

The code presented here was inspired by the original [rb_cqed package for modelling cavity-QED](https://github.com/tomdbar/rb-cqed). The Mathematica scripts used for calculating Clebsch-Gordan coefficients and energy level splittings are adapted from that repository—consider this as rb_cqed v2.

# Installation Guide

## Install using UV (recommended)

[UV](https://docs.astral.sh/uv/) is a fast, modern Python package and project manager written in Rust. It provides significantly faster dependency resolution and installation compared to traditional tools.

1. **Install UV:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or on macOS/Linux with Homebrew:
   ```bash
   brew install uv
   ```

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/jan-o-e/rb_photon_prod
   cd rb_photon_prod
   ```

3. **Create and sync virtual environment:**
   ```bash
   uv sync
   ```
   This command automatically creates a virtual environment (`.venv`), installs all dependencies specified in `pyproject.toml`, and generates a `uv.lock` file for reproducible builds.

4. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

5. **Install development dependencies (optional):**
   ```bash
   uv sync --group dev
   ```

**UV Commands Reference:**
- `uv sync` - Install/update dependencies and sync environment
- `uv add <package>` - Add a new dependency
- `uv remove <package>` - Remove a dependency
- `uv pip install <package>` - Install package using pip (legacy mode)
- `uv run <command>` - Run a command in the virtual environment without activation

# Repository Structure

## Directory Overview

### `src/`
Core simulation modules and example notebooks:
- `modules/` - Main simulation library containing:
  - `simulation.py` - Core `Simulation` class for atom-cavity dynamics
  - `atom_config.py` - Rb⁸⁷ atomic structure with hyperfine levels and Clebsch-Gordan coefficients
  - `ketbra_config.py` - Hilbert space basis states and ket-bra operators
  - `cavity.py` - Cavity mode operators and collapse operators
  - `ham_sim_source.py` - Hamiltonian construction for cavity, laser, and Raman interactions
  - `laser_pulses.py` - Pulse shape functions and Rabi-to-power conversions
  - `differential_light_shifts.py` - AC Stark shift calculations for D1/D2 transitions
  - `photon_correlation_calc.py` - Multi-photon correlation function calculations (g⁽ⁿ⁾)
  - `photon_correlation_utils.py` - Utilities for correlation analysis and visualisation
  - `correlation_functions.py` - Time-dependent correlation function evaluators
  - `integration_functions.py` - Numerical integration for complex-valued functions
  - `tensor_functions.py` - Custom tensor operators for composite Hilbert spaces
  - `vector_functions.py` - Vector algebra for polarisation and geometry
- `tests/` - Unit tests for core functionality

## Example Notebooks

For interactive exploration:
- [Photon_Correlation_Example_Calc_n1.ipynb](src/Photon_Correlation_Example_Calc_n1.ipynb) - Guided single-photon correlation analysis
- [CQED_Rb_Simulator.ipynb](src/CQED_Rb_Simulator.ipynb) - General simulation examples
- [FarDetuned_Raman.ipynb](src/FarDetuned_Raman.ipynb) - Simulation example for far detuned Raman pulses

### `experiments/`
Various experiments:
- `calibrate_fd_raman_pulses/` - Far-detuned Raman pulse calibration
- `fullsimulation/` - Complete end-to-end simulation sequences
- `stirap_rotations/` - STIRAP pulse optimisations
- `vstirap/` - v-STIRAP protocols for photon generation

### `run_correlation_calc/`
Production scripts for running photon correlation calculations:
- Parameter sweep scripts for n=1, n=2, and n=3 photon generation
- Magnetic field, detuning, and pulse shape optimisation

### `saved_data_bursts/`
Saved data for photon burst simulation results and output data.

### `saved_data_timebin/`
Saved data for time-bin encoded photonic states simulation results and data.

# Running Photon Correlation Calculations

The `run_correlation_calc/` directory contains production-ready scripts for calculating photon correlation functions (g⁽²⁾(τ)) and analyzing multi-photon generation schemes. Various calculations for different photon numbers and experimental parameters are available.

Two very useful references to understand these calculations are Bauch, David, et al. "Time-bin entanglement in the deterministic generation of linear photonic cluster states." APL Quantum 1.3 (2024) and Tóth, Géza, and Otfried Gühne. "Detecting genuine multipartite entanglement with two local measurements." Physical review letters 94.6 (2005): 060501.

The calculations for photonic cluster states are work in progress.

# Code Quality and Formatting

This project uses modern Python tooling for code quality and consistency.

## Formatting with Black and Ruff

The codebase follows strict formatting standards enforced by:
- **[Black](https://black.readthedocs.io/)** - Opinionated code formatter ensuring consistent style
- **[Ruff](https://docs.astral.sh/ruff/)** - Fast Python linter covering hundreds of rules (replaces flake8, isort, etc.)

### Manual Formatting

Format code manually with:

```bash
# Format all Python files with Black
black .

# Run Ruff linter
ruff check .

# Auto-fix Ruff issues where possible
ruff check --fix .
```

### Development Dependencies

Install formatting tools with UV:

```bash
uv sync --group dev
```

This installs:
- `black>=24.0.0`
- `ruff>=0.6.0`
- `pytest>=8.4.2`

### Pre-commit Hooks

While this repository doesn't currently have a `.pre-commit-config.yaml` file configured, you can set up pre-commit hooks to automatically format code before each commit:

1. **Install pre-commit:**
   ```bash
   uv pip install pre-commit
   ```

2. **Create `.pre-commit-config.yaml`:**
   ```yaml
   repos:
     - repo: https://github.com/psf/black
       rev: 24.0.0
       hooks:
         - id: black
           language_version: python3.10

     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.6.0
       hooks:
         - id: ruff
           args: [--fix]
         - id: ruff-format
   ```

3. **Install the hooks:**
   ```bash
   pre-commit install
   ```

Now Black and Ruff will automatically run on staged files before each commit, ensuring code quality standards are maintained.

## Contributing

Please submit a PR if you are interested in contributing and submit an issue if you find any errors.

I always wanted to modularise this for different atomic species but never got round to it.

## References

Jan Ole Ernst et al 2023 J. Phys. B: At. Mol. Opt. Phys. **56** 205003

Jan Ole Ernst. *Controlling Quantum Systems at the Pulse Level: Cavity QED & Beyond*. 2025. PhD Thesis (forthcoming). University of Oxford
