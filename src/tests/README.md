# Rb Photon Production Simulation Tests

This directory contains regression tests for the Rubidium atom photon production simulations. These tests ensure that core simulation functionality remains unchanged across refactoring and code additions.

## Test Files

### 1. `test_general_rb_simulator.py`
Tests the **VSTIRAP + STIRAP Repumping** sequence simulation from `General_Rb_Simulator.ipynb`.

**What it tests:**
- Full VSTIRAP (Vacuum-Stimulated Rapid Adiabatic Passage) pulse sequence
- STIRAP (Stimulated Raman Adiabatic Passage) repumping
- Cavity-atom coupling with photonic Hilbert space
- Final density matrix after pulse sequence
- Spontaneous emission on D1 and D2 lines

**Initial state:** `|g2, 0_x, 0_y>` (F=2, mF=0, zero cavity photons)

**Key parameters:**
- B-field splitting: 0 MHz
- Cavity decay κ: 2.1×2π MHz
- Atom-cavity coupling: 11.1 MHz
- VSTIRAP pulse length: 0.35 μs
- Repump pulse length: 0.15 μs

### 2. `test_far_detuned_raman.py`
Tests the **Far-Detuned Raman Pulse Pair** simulation from `Test_FarDetuned_Raman.ipynb`.

**What it tests:**
- Two-photon Raman transition via far-detuned intermediate state
- Ground state coherence generation
- Spontaneous emission suppression in far-detuned regime
- Population transfer and coherence evolution

**Initial state:** `|g2>` (F=2, mF=0, no photonic space)

**Key parameters:**
- Detuning: -500 GHz (far-detuned)
- Pulse length: ~0.8 μs
- Flattop pulse with Blackman window edges
- Pulse polarizations: π and σ⁻

## Running the Tests

### Quick Start

The tests are **ready to use immediately** with full density matrix regression checking:

```bash
# Run all tests
pytest src/tests/ -v

# Run with detailed output
pytest src/tests/ -v -s
```

The reference density matrices are stored in:
- `reference_general_rb_sim_rho.npy` (128×128 complex matrix, 256 KB)
- `reference_far_detuned_raman_rho.npy` (32×32 complex matrix, 16 KB)

### Regenerating Reference Data

If you need to regenerate reference data (e.g., after fixing a bug or changing physics):

```bash
# Generate new reference density matrices
python src/tests/generate_reference_data.py

# Then run tests to verify
pytest src/tests/ -v
```

**Warning:** Only regenerate reference data if you understand why the results changed!

### Regular Usage

After setup, run tests before and after making changes:

```bash
# Run all tests
pytest src/tests/ -v

# Run specific test file
pytest src/tests/test_general_rb_simulator.py -v

# Run with detailed output
pytest src/tests/ -v -s

# Run specific test function
pytest src/tests/test_far_detuned_raman.py::TestFarDetunedRaman::test_far_detuned_raman_density_matrix -v
```

## What These Tests Protect Against

### Full Density Matrix Regression
**Both tests now compare the complete final density matrix** against stored reference values:
- Every element of the density matrix is checked (128×128 = 16,384 elements for VSTIRAP test)
- Detects any numerical changes at the ~10⁻⁶ relative tolerance level
- Guards against subtle bugs that might only affect off-diagonal coherences
- Ensures complete reproducibility of quantum state evolution

### Numerical Regression
- Changes to numerical integration that alter results
- Unintended modifications to physical constants
- Bugs in Hamiltonian construction
- Issues with collapse operator definitions
- Solver parameter changes
- Floating-point arithmetic variations

### API Changes
- Breaking changes to module interfaces
- Parameter ordering changes
- Default value modifications

### Physical Correctness (Always Checked)
- Trace preservation (unitarity + dissipation)
- Hermiticity of density matrices
- Non-negative populations (within numerical tolerance)
- Proper spontaneous emission rates

## Test Structure

Both tests follow a similar pattern:

1. **Fixtures** (`@pytest.fixture`):
   - Set up atomic configuration
   - Define pulse parameters
   - Create Hamiltonians
   - Configure collapse operators

2. **Test Functions**:
   - Run simulation with `mesolve`
   - Extract final density matrix
   - Check physical constraints (sanity checks)
   - Load reference density matrix from `.npy` file
   - Compare full matrices element-wise

3. **Sanity Checks** (always active):
   - Trace = 1 (±1e-6)
   - Hermiticity (±1e-10)
   - Positive populations (≥ -1e-6, allowing for numerical noise)
   - Physical behavior (e.g., population transfer occurred)

4. **Regression Checks** (active when reference files exist):
   - Full density matrix comparison using `np.allclose()`
   - Tolerance: rtol=1e-6, atol=1e-9
   - Reports max absolute and relative differences
   - Fails test if any element exceeds tolerance

## Updating Reference Values

If you **intentionally** change the simulation (e.g., fix a bug, improve accuracy):

1. Review the change carefully - understand why results changed
2. Regenerate reference data:
   ```bash
   python src/tests/generate_reference_data.py
   ```
3. Run tests to verify the new references work:
   ```bash
   pytest src/tests/ -v -s
   ```
4. Verify the new values are physically reasonable (check printed observables)
5. Commit the updated `.npy` files along with your code changes
6. Document the change in git commit message

The reference files to commit are:
- `src/tests/reference_general_rb_sim_rho.npy`
- `src/tests/reference_far_detuned_raman_rho.npy`

## Tips for Debugging Test Failures

If a test fails:

1. **Check the error message**: Which observable failed? By how much?

2. **Small differences (< 1e-5)**: Likely numerical precision or solver settings
   - Check if QuTiP version changed
   - Check if numerical tolerances were modified

3. **Large differences**: Likely a real bug
   - Review recent code changes
   - Check if physical parameters were accidentally modified
   - Verify Hamiltonian construction

4. **Systematic shifts**: All populations shifted by same amount
   - Check normalization
   - Check for missing/extra terms in Hamiltonian

5. **Use `-s` flag**: See detailed output including actual vs expected values
   ```bash
   pytest src/tests/test_general_rb_simulator.py -v -s
   ```

## Test Coverage

These tests cover the **core simulation engine** but not:
- Plotting functions
- Parameter sweeps
- Optimization routines
- Data analysis utilities

For those, consider adding integration tests or visual inspection workflows.

## Dependencies

Tests require:
- `pytest`
- `numpy`
- `qutip` (≥5.0)
- All modules in `src/modules/`

Install with:
```bash
pip install pytest
# or if using uv:
uv pip install pytest
```

## Future Enhancements

Consider adding:
- [ ] Tests for other pulse sequences
- [ ] Parameter sweep validation
- [ ] Performance benchmarks
- [ ] Energy conservation checks
- [ ] Photon number distribution tests
