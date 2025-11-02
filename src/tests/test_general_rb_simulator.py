"""
Regression test for General Rb Simulator - VSTIRAP + STIRAP Repumping sequence.

This test ensures that core simulation functionality remains unchanged across refactoring
and code additions by comparing the final density matrix against a known reference value.
"""

import os
import numpy as np
import pytest
from itertools import chain
from qutip import mesolve

from modules.atom_config import RbAtom
from modules.cavity import cav_collapse_ops, quant_axis_cavbasis_mapping
from modules.ketbra_config import RbKetBras

# Path to reference data
REFERENCE_FILE = os.path.join(
    os.path.dirname(__file__), "reference_general_rb_sim_rho.npy"
)


class TestGeneralRbSimulator:
    """Test suite for General Rb Simulator VSTIRAP sequence."""

    @pytest.fixture(scope="class")
    def atom_config(self):
        """Configure atomic states and system parameters."""
        # Define ground states to include in simulation
        atomStates = {
            "g1M": 0,
            "g1": 1,
            "g1P": 2,  # F=1, mF=-1,0,+1
            "g2MM": 3,
            "g2M": 4,
            "g2": 5,
            "g2P": 6,
            "g2PP": 7,  # F=2, mF=-2,..,+2
        }

        # Define excited levels
        xlvls = [
            "x0",
            "x1M",
            "x1",
            "x1P",
            "x2MM",
            "x2M",
            "x2",
            "x2P",
            "x2PP",
            "x3MMM",
            "x3MM",
            "x3M",
            "x3",
            "x3P",
            "x3PP",
            "x3PPP",
            "x1M_d1",
            "x1_d1",
            "x1P_d1",
            "x2MM_d1",
            "x2M_d1",
            "x2_d1",
            "x2P_d1",
            "x2PP_d1",
        ]

        # Configure ketbra class with photonic Hilbert space
        kb_class = RbKetBras(atomStates, xlvls, True)
        ketbras = kb_class.getrb_ketbras()

        # Configure Rb atom with B-field splitting
        bfieldsplit = "0"
        rb_atom = RbAtom(bfieldsplit, kb_class)

        return {
            "atomStates": atomStates,
            "kb_class": kb_class,
            "ketbras": ketbras,
            "rb_atom": rb_atom,
        }

    @pytest.fixture(scope="class")
    def cavity_params(self, atom_config):
        """Define cavity parameters and coupling."""
        kappa = 2.1 * 2.0 * np.pi  # Cavity decay rate
        deltaP = 0 * 2.0 * np.pi  # Cavity birefringence

        # Get Clebsch-Gordan coefficient for desired transition
        rb_atom = atom_config["rb_atom"]
        cav_transition = rb_atom.CGg1Mx1
        coupling_factor = 11.1 * 2 * np.pi / cav_transition

        # Define quantization and cavity axes
        cav_axis = [1, 0, 0]
        quant_axis = [0, 1, 0]

        return {
            "kappa": kappa,
            "deltaP": deltaP,
            "coupling_factor": coupling_factor,
            "cav_axis": cav_axis,
            "quant_axis": quant_axis,
        }

    @pytest.fixture(scope="class")
    def vstirap_hamiltonian(self, atom_config, cavity_params):
        """Create VSTIRAP Hamiltonian."""
        rb_atom = atom_config["rb_atom"]
        ketbras = atom_config["ketbras"]
        atomStates = atom_config["atomStates"]

        # VSTIRAP pulse parameters
        lengthStirap = 0.35
        OmegaStirap = 230 * 2 * np.pi
        delta_cav = 0
        delta_laser = 0

        # Pulse shape arguments
        args_omega_stirap = dict(
            [("T", lengthStirap), ("wStirap", np.pi / lengthStirap)]
        )
        vst_driving_shape = "(np.sin(wStirap*t)**2)"

        # VSTIRAP transition levels
        F_vst_start = 2
        F_vst_final = 1
        F_vst_exc = 1

        # Generate VSTIRAP Hamiltonian
        quant_axis_mapping = quant_axis_cavbasis_mapping(
            cavity_params["quant_axis"], cavity_params["cav_axis"]
        )

        H_VStirap, args_hams_VStirap = rb_atom.gen_H_VSTIRAP_D2(
            ketbras,
            atomStates,
            delta_cav,
            delta_laser,
            F_vst_start,
            F_vst_final,
            F_vst_exc,
            "pi",
            OmegaStirap,
            cavity_params["coupling_factor"],
            cavity_params["deltaP"],
            quant_axis_mapping,
            args_omega_stirap,
            vst_driving_shape,
        )

        return H_VStirap, args_hams_VStirap, lengthStirap

    @pytest.fixture(scope="class")
    def repump_hamiltonian(self, atom_config):
        """Create STIRAP repumping Hamiltonian."""
        rb_atom = atom_config["rb_atom"]
        ketbras = atom_config["ketbras"]
        atomStates = atom_config["atomStates"]

        # Repumping pulse parameters
        A_rep = 41
        CGg2x1 = rb_atom.CGg2x1
        CGg1Px1 = rb_atom.CGg1Px1
        A_s = abs(A_rep / CGg2x1) * 2 * np.pi
        A_p = abs(A_rep / CGg1Px1) * 2 * np.pi
        lengthRepump = 0.15

        # Pulse shape parameters
        a = 11
        n = 6
        c = 0.05
        args_repump = dict([("n", n), ("c", c), ("a", a), ("T", lengthRepump)])

        pump_shape = "np.exp(-((t - (T/2))/c)**(2*n))*np.sin(np.pi/2*(1/(1 + np.exp((-a*(t - T/2))/T))))"
        stokes_shape = "np.exp(-((t - (T/2))/c)**(2*n))*np.cos(np.pi/2*(1/(1 + np.exp((-a*(t - T/2))/T))))"

        # STIRAP transition parameters
        delta_sti = 0
        F_pump_start = 1
        F_pump_exc = 1
        F_stokes_start = 2
        F_stokes_exc = 1

        stokes_pol = "pi"
        pump_pol_1 = "sigmaP"
        pump_pol_2 = "sigmaM"

        # Generate repumping Hamiltonian components
        H_Stirap_Stokes = rb_atom.gen_H_Pulse_D1(
            ketbras,
            atomStates,
            delta_sti,
            F_stokes_start,
            F_stokes_exc,
            stokes_pol,
            A_s,
            args_repump,
            stokes_shape,
        )
        H_Stirap_Pump_1 = rb_atom.gen_H_Pulse_D1(
            ketbras,
            atomStates,
            delta_sti,
            F_pump_start,
            F_pump_exc,
            pump_pol_1,
            A_p,
            args_repump,
            pump_shape,
        )
        H_Stirap_Pump_2 = rb_atom.gen_H_Pulse_D1(
            ketbras,
            atomStates,
            delta_sti,
            F_pump_start,
            F_pump_exc,
            pump_pol_2,
            A_p,
            args_repump,
            pump_shape,
        )

        # Combine Hamiltonians
        H_Repump = list(
            chain(*[H_Stirap_Stokes[0], H_Stirap_Pump_1[0], H_Stirap_Pump_2[0]])
        )
        args_hams_Repump = {
            **H_Stirap_Stokes[1],
            **H_Stirap_Pump_1[1],
            **H_Stirap_Pump_2[1],
        }

        return H_Repump, args_hams_Repump, lengthRepump

    @pytest.fixture(scope="class")
    def collapse_operators(self, atom_config, cavity_params):
        """Create collapse operators for the system."""
        rb_atom = atom_config["rb_atom"]
        atomStates = atom_config["atomStates"]

        # Cavity collapse operators
        c_op_list = cav_collapse_ops(cavity_params["kappa"], atomStates)

        # Spontaneous emission collapse operators (D2 and D1 lines)
        c_op_list += rb_atom.spont_em_ops(atomStates)[0]  # D2 line
        c_op_list += rb_atom.spont_em_ops(atomStates)[1]  # D1 line

        return c_op_list

    def test_vstirap_repump_sequence_density_matrix(
        self, atom_config, vstirap_hamiltonian, repump_hamiltonian, collapse_operators
    ):
        """
        Test VSTIRAP + STIRAP repumping sequence produces expected final density matrix.

        This test runs the full simulation sequence and compares the final density matrix
        against a reference value to detect any changes in core functionality.
        """
        kb_class = atom_config["kb_class"]

        # Unpack Hamiltonians
        H_VStirap, args_hams_VStirap, lengthStirap = vstirap_hamiltonian
        H_Repump, args_hams_Repump, lengthRepump = repump_hamiltonian

        # Define time arrays
        tVStirap = np.linspace(0, lengthStirap, 200)
        tRepump = np.linspace(0, lengthRepump, 150)

        # Initial state: |g2, 0_x, 0_y>
        psi0 = kb_class.get_ket("g2", 0, 0)

        # Run VSTIRAP simulation
        output_vst = mesolve(
            H_VStirap, psi0, tVStirap, collapse_operators, [], args=args_hams_VStirap
        )

        # Run repumping simulation with final state from VSTIRAP
        psi_after_vst = output_vst.states[-1]
        output_repump = mesolve(
            H_Repump,
            psi_after_vst,
            tRepump,
            collapse_operators,
            [],
            args=args_hams_Repump,
        )

        # Get final density matrix
        final_rho = output_repump.states[-1]

        # Convert to numpy array for comparison
        final_rho_array = final_rho.full()

        # Reference values - these are the expected values from the original simulation
        # Key diagonal elements that should be preserved
        expected_diagonal_sum = 1.0  # Trace should be 1

        # Check trace preservation
        actual_trace = np.trace(final_rho_array)
        assert np.isclose(
            actual_trace, expected_diagonal_sum, atol=1e-6
        ), f"Trace not preserved: {actual_trace} != {expected_diagonal_sum}"

        # Reference density matrix from original simulation
        # This is a snapshot of the final density matrix to detect regression
        # Format: Real part followed by imaginary part for key elements

        # NOTE: Run this test once to generate reference values, then uncomment below
        # and replace with actual values from the first run
        # Store key diagonal populations (real only, as they should be real)
        # expected_populations = {
        #     "g1M": 0.0,  # These will be updated after first run
        #     "g1": 0.0,
        #     "g1P": 0.0,
        #     "g2MM": 0.0,
        #     "g2M": 0.0,
        #     "g2": 0.0,
        #     "g2P": 0.0,
        #     "g2PP": 0.0,
        # }

        # REFERENCE VALUES - Generated from baseline simulation
        # These values lock in the expected behavior of the simulation
        # Update these after verifying the simulation works correctly

        # For now, perform basic sanity checks
        actual_populations = {
            "g1M": np.real(final_rho_array[0, 0]),
            "g1": np.real(final_rho_array[1, 1]),
            "g1P": np.real(final_rho_array[2, 2]),
            "g2MM": np.real(final_rho_array[3, 3]),
            "g2M": np.real(final_rho_array[4, 4]),
            "g2": np.real(final_rho_array[5, 5]),
            "g2P": np.real(final_rho_array[6, 6]),
            "g2PP": np.real(final_rho_array[7, 7]),
        }

        # Sanity checks
        # 1. All populations should be non-negative
        for state, pop in actual_populations.items():
            assert pop >= -1e-10, f"Negative population for {state}: {pop}"

        # 2. Trace should be 1 (most important check for cavity systems)
        # Note: In cavity QED, significant population can be in photon states

        # 3. Density matrix should be Hermitian
        hermiticity_check = np.allclose(
            final_rho_array, final_rho_array.conj().T, atol=1e-10
        )
        assert hermiticity_check, "Final density matrix is not Hermitian"

        # 4. Some population should have transferred (not all in initial state)
        assert (
            actual_populations["g2"] < 0.99
        ), "No population transfer occurred from initial state"

        # === REGRESSION CHECK AGAINST REFERENCE DENSITY MATRIX ===

        if os.path.exists(REFERENCE_FILE):
            # Load reference density matrix
            reference_rho = np.load(REFERENCE_FILE)

            # Check that shapes match
            assert (
                final_rho_array.shape == reference_rho.shape
            ), f"Shape mismatch: {final_rho_array.shape} != {reference_rho.shape}"

            # Compare density matrices element-wise
            # Use relative tolerance for large elements, absolute for small ones
            max_diff = np.max(np.abs(final_rho_array - reference_rho))
            rel_diff = np.max(
                np.abs(final_rho_array - reference_rho)
                / (np.abs(reference_rho) + 1e-10)
            )

            # Check overall agreement
            matrices_match = np.allclose(
                final_rho_array, reference_rho, rtol=1e-6, atol=1e-9
            )

            assert matrices_match, (
                f"Density matrix regression failed!\n"
                f"Max absolute difference: {max_diff:.3e}\n"
                f"Max relative difference: {rel_diff:.3e}\n"
                f"This indicates the simulation results have changed."
            )

            print("\n" + "=" * 60)
            print("REGRESSION CHECK PASSED")
            print("=" * 60)
            print("Density matrix matches reference within tolerance")
            print(f"Max absolute difference: {max_diff:.3e}")
            print(f"Max relative difference: {rel_diff:.3e}")
            print("=" * 60 + "\n")

        else:
            # Reference file doesn't exist - print warning
            print("\n" + "=" * 60)
            print("WARNING: Reference file not found!")
            print("=" * 60)
            print(f"Expected: {REFERENCE_FILE}")
            print("Run 'python src/tests/generate_reference_data.py' to create it.")
            print("\nCurrent populations:")
            for state, pop in actual_populations.items():
                print(f"    '{state}': {pop:.15e}")
            print("=" * 60 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
