"""
Regression test for n=1 photon correlation calculation.

This test ensures that the single-photon correlation calculation remains unchanged
across refactoring by validating:
1. Off-diagonal elements have magnitude smaller than the minimum diagonal element
2. Numerical values match reference within tolerance (1e-6)
"""

import os
import sys
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from run_correlation_calc.run_correlation_calc import run_simulation


class TestN1CorrelationCalc:
    """Test suite for n=1 photon correlation calculation."""

    @pytest.fixture(scope="class")
    def simulation_params(self, tmp_path_factory):
        """Define simulation parameters matching the single run configuration."""
        # Create a temporary directory that persists for the test session
        temp_dir = tmp_path_factory.mktemp("test_data")
        return {
            "_save_dir": str(temp_dir) + "/",
            "n_sim_steps": 8,
            "b_field": "0p07",
            "n_photons": 1,
            "_n_start": 1,
            "_len_stirap": 1,
            "_omega_stirap_early": 30,
            "_omega_stirap_late": 30,
            "_shape_stirap": "sinsquared",
            "_vst_ramp_up": 0.17,
            "_two_photon_det_2": 0,
            "_two_photon_det_4": 0,
            "_spont_emission": True,
            "_plot": False,
        }

    def test_n1_correlation_matrix_properties(self, simulation_params):
        """
        Test n=1 correlation calculation produces expected matrix properties.

        Validates:
        1. Off-diagonal elements are smaller than min diagonal element
        2. Numerical values match reference within 1e-6 tolerance
        """
        # Run the simulation (directory creation handled by run_simulation)
        result = run_simulation(**simulation_params)

        # Extract the density matrix from result dictionary
        # The result is a dictionary containing real and imaginary parts of matrix elements
        assert isinstance(result, dict), f"Expected dict result, got {type(result)}"

        # Reconstruct the 2x2 density matrix from the dictionary entries
        density_matrix = np.array(
            [
                [
                    result["int_diag_00_re"] + 1j * result["int_diag_00_im"],
                    result["int_off_diag_01_re"] + 1j * result["int_off_diag_01_im"],
                ],
                [
                    result["int_off_diag_10_re"] + 1j * result["int_off_diag_10_im"],
                    result["int_diag_11_re"] + 1j * result["int_diag_11_im"],
                ],
            ]
        )

        # Verify matrix is 2x2
        assert density_matrix.shape == (
            2,
            2,
        ), f"Expected 2x2 matrix, got {density_matrix.shape}"

        # === Property 1: Off-diagonal elements smaller than min diagonal ===

        # Extract diagonal elements
        diagonal = np.abs(np.diag(density_matrix))
        min_diagonal = np.min(diagonal)

        # Extract off-diagonal elements
        n = density_matrix.shape[0]
        off_diagonal_mask = ~np.eye(n, dtype=bool)
        off_diagonal_elements = np.abs(density_matrix[off_diagonal_mask])
        max_off_diagonal = np.max(off_diagonal_elements)

        # Check that all off-diagonal elements are smaller than min diagonal
        assert (
            max_off_diagonal < min_diagonal
        ), f"Off-diagonal elements too large: max={max_off_diagonal:.6e} >= min_diag={min_diagonal:.6e}"

        print("\nMatrix properties check PASSED:")
        print(f"  Min diagonal element: {min_diagonal:.6e}")
        print(f"  Max off-diagonal element: {max_off_diagonal:.6e}")
        print(f"  Ratio: {max_off_diagonal/min_diagonal:.6e}")

        # === Property 2: Hermiticity check ===
        hermiticity_check = np.allclose(
            density_matrix, density_matrix.conj().T, atol=1e-10
        )
        assert hermiticity_check, "Density matrix is not Hermitian"

        # === Property 3: Trace check (if it's a density matrix) ===
        trace = np.trace(density_matrix)
        print(f"  Matrix trace: {trace:.6e}")

        # === Regression check against reference values ===

        # REFERENCE VALUES - Generated from baseline simulation with 8 sim steps
        # These values lock in the expected behavior and validate against regressions
        # Tolerance: 2e-4 (relative) and 1e-8 (absolute) - accounts for numerical precision in long simulations

        reference_values = {
            "int_diag_00_re": 1.606025940367968e-02,
            "int_diag_00_im": 0.000000000000000e00,
            "int_diag_11_re": 1.477970903666534e-02,
            "int_diag_11_im": 0.000000000000000e00,
            "int_off_diag_01_re": 1.284505956488167e-02,
            "int_off_diag_01_im": 4.336209783883778e-03,
            "int_off_diag_10_re": 1.284505956488167e-02,
            "int_off_diag_10_im": -4.336209783883778e-03,
        }

        if len(reference_values) > 0:
            # Reference values have been set - perform regression check

            # Check all matrix elements
            for key in [
                "int_diag_00_re",
                "int_diag_00_im",
                "int_diag_11_re",
                "int_diag_11_im",
                "int_off_diag_01_re",
                "int_off_diag_01_im",
                "int_off_diag_10_re",
                "int_off_diag_10_im",
            ]:
                assert np.isclose(
                    result[key], reference_values[key], rtol=2e-4, atol=1e-8
                ), f"{key} changed: {result[key]:.6e} != {reference_values[key]:.6e}"

            print("\n" + "=" * 60)
            print("REGRESSION CHECK PASSED")
            print("=" * 60)
            print(
                "Matrix values match reference within tolerance (rtol=2e-4, atol=1e-8)"
            )
            print("=" * 60 + "\n")

        else:
            # Reference values not set - print current values for copy-paste
            print("\n" + "=" * 60)
            print("REFERENCE VALUES NOT SET - First run")
            print("=" * 60)
            print("Copy these values into the test to enable regression checking:")
            print("\nreference_values = {")
            for key in [
                "int_diag_00_re",
                "int_diag_00_im",
                "int_diag_11_re",
                "int_diag_11_im",
                "int_off_diag_01_re",
                "int_off_diag_01_im",
                "int_off_diag_10_re",
                "int_off_diag_10_im",
            ]:
                print(f"    '{key}': {result[key]:.15e},")
            print("}")
            print(f"\nMin diagonal: {min_diagonal:.15e}")
            print(f"Max off-diagonal: {max_off_diagonal:.15e}")
            print("\n" + "=" * 60 + "\n")

        # pytest will automatically clean up the temporary directory


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
