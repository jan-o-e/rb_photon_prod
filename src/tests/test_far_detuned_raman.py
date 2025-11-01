"""
Regression test for Far-Detuned Raman simulation.

This test ensures that the far-detuned Raman pulse pair simulation functionality
remains unchanged across refactoring and code additions by comparing the final
density matrix against a known reference value.
"""

import os
import numpy as np
import pytest
from qutip import mesolve

from modules.atom_config import RbAtom
from modules.ketbra_config import RbKetBras
from modules.laser_pulses import create_flattop_blackman

# Path to reference data
REFERENCE_FILE = os.path.join(
    os.path.dirname(__file__), "reference_far_detuned_raman_rho.npy"
)


class TestFarDetunedRaman:
    """Test suite for Far-Detuned Raman simulation."""

    @pytest.fixture(scope="class")
    def atom_config(self):
        """Configure atomic states without photonic Hilbert space."""
        # Define ground states
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

        # Define excited levels (D1 and D2)
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

        # Configure ketbra class WITHOUT photonic Hilbert space (False)
        kb_class = RbKetBras(atomStates, xlvls, False)
        ketbras = kb_class.getrb_ketbras()

        # Configure Rb atom with zero B-field splitting
        bfieldsplit = "0"
        rb_atom = RbAtom(bfieldsplit, kb_class)

        return {
            "atomStates": atomStates,
            "kb_class": kb_class,
            "ketbras": ketbras,
            "rb_atom": rb_atom,
        }

    @pytest.fixture(scope="class")
    def laser_pulses(self):
        """Define laser pulse parameters and generate pulse shapes."""
        # Pulse parameters
        two_phot_det = 0.0
        pulse_length = 5 / (2 * np.pi) + 0.005

        # Time array
        pulse_time = np.linspace(0, pulse_length, 5000)

        # Generate flattop pulses with Blackman window
        pulse_1 = create_flattop_blackman(pulse_time, 1, 0.025, 0.025)
        pulse_2 = create_flattop_blackman(pulse_time, 1, 0.025, 0.025)

        return {
            "pulse_time": pulse_time,
            "pulse_1": pulse_1,
            "pulse_2": pulse_2,
            "two_phot_det": two_phot_det,
            "pulse_length": pulse_length,
        }

    @pytest.fixture(scope="class")
    def raman_hamiltonian(self, atom_config, laser_pulses):
        """Create far-detuned Raman Hamiltonian and parameters."""
        rb_atom = atom_config["rb_atom"]
        ketbras = atom_config["ketbras"]
        atomStates = atom_config["atomStates"]

        pulse_time = laser_pulses["pulse_time"]
        pulse_1 = laser_pulses["pulse_1"]
        pulse_2 = laser_pulses["pulse_2"]
        two_phot_det = laser_pulses["two_phot_det"]

        # Laser polarizations
        pol_1 = "pi"
        pol_2 = "sigmaM"

        # Detunings
        det_1 = -500000 * 2 * np.pi
        det_2 = det_1 - two_phot_det * 2 * np.pi - rb_atom.getrb_gs_splitting()

        # Amplitudes
        amp_1 = 1 * np.sqrt(2 * 500000) * 2 * np.pi / rb_atom.CG_d1g2x1
        amp_2 = 1 * np.sqrt(2 * 500000) * 2 * np.pi / rb_atom.CG_d1g1Px1

        # Generate Hamiltonian
        ham, args = rb_atom.gen_H_FarDetuned_Raman_PulsePair_D1(
            ketbras,
            atomStates,
            det_1,
            det_2,
            pol_1,
            pol_2,
            amp_1,
            amp_2,
            pulse_time,
            pulse_1,
            pulse_2,
        )

        return {
            "hamiltonian": ham,
            "args": args,
            "pol_1": pol_1,
            "pol_2": pol_2,
            "amp_1": amp_1,
            "amp_2": amp_2,
            "det_1": det_1,
            "det_2": det_2,
        }

    @pytest.fixture(scope="class")
    def collapse_operators(self, atom_config, raman_hamiltonian):
        """Create spontaneous emission collapse operators."""
        rb_atom = atom_config["rb_atom"]
        atomStates = atom_config["atomStates"]

        pol_1 = raman_hamiltonian["pol_1"]
        pol_2 = raman_hamiltonian["pol_2"]
        amp_1 = raman_hamiltonian["amp_1"]
        amp_2 = raman_hamiltonian["amp_2"]
        det_1 = raman_hamiltonian["det_1"]
        det_2 = raman_hamiltonian["det_2"]

        # Add far-detuned spontaneous emission operators for both pulses
        c_op_list = []
        c_op_list += rb_atom.spont_em_ops_far_detuned(atomStates, pol_1, amp_1, det_1)
        c_op_list += rb_atom.spont_em_ops_far_detuned(atomStates, pol_2, amp_2, det_2)

        return c_op_list

    def test_far_detuned_raman_density_matrix(
        self, atom_config, laser_pulses, raman_hamiltonian, collapse_operators
    ):
        """
        Test far-detuned Raman simulation produces expected final density matrix.

        This test runs the Raman pulse pair simulation and verifies:
        1. The final density matrix diagonal elements (populations)
        2. Key coherence terms between ground states
        3. Spontaneous emission losses
        4. Physical constraints (trace, Hermiticity, positive populations)
        """
        kb_class = atom_config["kb_class"]
        atomStates = atom_config["atomStates"]
        pulse_time = laser_pulses["pulse_time"]

        ham = raman_hamiltonian["hamiltonian"]

        # Initial state: |g2> (F=2, mF=0)
        psi0 = kb_class.get_ket_atomic("g2")

        # Run simulation
        output_mesolve = mesolve(ham, psi0, pulse_time, collapse_operators)

        # Get final density matrix
        final_rho = output_mesolve.states[-1]
        final_rho_array = final_rho.full()

        # === BASIC PHYSICAL CONSTRAINTS ===

        # 1. Check trace preservation
        actual_trace = np.trace(final_rho_array)
        assert np.isclose(
            actual_trace, 1.0, atol=1e-6
        ), f"Trace not preserved: {actual_trace} != 1.0"

        # 2. Density matrix should be Hermitian
        hermiticity_check = np.allclose(
            final_rho_array, final_rho_array.conj().T, atol=1e-10
        )
        assert hermiticity_check, "Final density matrix is not Hermitian"

        # === EXTRACT KEY OBSERVABLES ===

        # Ground state populations (diagonal elements)
        state_labels = list(atomStates.keys())
        actual_populations = {}
        for i, state in enumerate(state_labels):
            actual_populations[state] = np.real(final_rho_array[i, i])

        # Key coherence term: rho_{g1P,g2} (element [2, 5])
        coherence_g1P_g2_real = np.real(final_rho_array[2, 5])
        coherence_g1P_g2_imag = np.imag(final_rho_array[2, 5])

        # Calculate spontaneous emission
        sigma_spontDecayOp = sum([x.dag() * x for x in collapse_operators])
        exp_spontDecay = np.abs(
            np.array([(x * sigma_spontDecayOp).tr() for x in output_mesolve.states])
        )
        n_spont = np.trapezoid(exp_spontDecay, dx=pulse_time[1] - pulse_time[0])

        # === SANITY CHECKS ===

        # 1. All populations should be non-negative (allow small numerical errors)
        for state, pop in actual_populations.items():
            assert pop >= -1e-6, f"Negative population for {state}: {pop}"

        # 2. Populations should sum to ~1
        total_population = sum(actual_populations.values())
        assert np.isclose(
            total_population, 1.0, atol=1e-6
        ), f"Total population != 1: {total_population}"

        # 3. Spontaneous emission should be very small (far-detuned regime)
        assert (
            n_spont < 1e-6
        ), f"Spontaneous emission too high for far-detuned regime: {n_spont}"

        # 4. Initial state g2 should have transferred some population
        assert (
            actual_populations["g2"] < 1.0
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
            max_diff = np.max(np.abs(final_rho_array - reference_rho))
            rel_diff = np.max(
                np.abs(final_rho_array - reference_rho)
                / (np.abs(reference_rho) + 1e-10)
            )

            # Check overall agreement
            # Use rtol=1e-5 for larger elements and atol=1e-5 for near-zero elements
            # This accounts for numerical precision in quantum dynamics simulations
            matrices_match = np.allclose(
                final_rho_array, reference_rho, rtol=1e-5, atol=1e-5
            )

            assert matrices_match, (
                f"Density matrix regression failed!\n"
                f"Max absolute difference: {max_diff:.3e}\n"
                f"Max relative difference: {rel_diff:.3e}\n"
                f"This indicates the simulation results have changed."
            )

            print("\n" + "=" * 70)
            print("REGRESSION CHECK PASSED")
            print("=" * 70)
            print("Density matrix matches reference within tolerance")
            print(f"Max absolute difference: {max_diff:.3e}")
            print(f"Max relative difference: {rel_diff:.3e}")
            print("\nKey observables:")
            print(f"  g2 population: {actual_populations['g2']:.6f} (initial state)")
            print(f"  g1P population: {actual_populations['g1P']:.6f} (target state)")
            print(
                f"  Coherence |ρ_g1P,g2|: {np.abs(coherence_g1P_g2_real + 1j*coherence_g1P_g2_imag):.6f}"
            )
            print(f"  Spontaneous emission: {n_spont:.3e}")
            print("=" * 70 + "\n")

        else:
            # Reference file doesn't exist - print warning
            print("\n" + "=" * 70)
            print("WARNING: Reference file not found!")
            print("=" * 70)
            print(f"Expected: {REFERENCE_FILE}")
            print("Run 'python src/tests/generate_reference_data.py' to create it.")
            print("\nKey observables:")
            print(f"  g2 population: {actual_populations['g2']:.15e}")
            print(f"  g1P population: {actual_populations['g1P']:.15e}")
            print(f"  Coherence (real): {coherence_g1P_g2_real:.15e}")
            print(f"  Coherence (imag): {coherence_g1P_g2_imag:.15e}")
            print(f"  Spontaneous emission: {n_spont:.15e}")
            print("=" * 70 + "\n")

    def test_far_detuned_raman_coherence_evolution(
        self, atom_config, laser_pulses, raman_hamiltonian, collapse_operators
    ):
        """
        Test that coherence between ground states develops during simulation.

        This verifies that the Raman process creates coherent superposition states.
        """
        kb_class = atom_config["kb_class"]
        pulse_time = laser_pulses["pulse_time"]
        ham = raman_hamiltonian["hamiltonian"]

        # Initial state: |g2>
        psi0 = kb_class.get_ket_atomic("g2")

        # Run simulation
        output = mesolve(ham, psi0, pulse_time, collapse_operators)

        # Check coherence at final time
        final_rho = output.states[-1].full()
        coherence_magnitude = np.abs(final_rho[2, 5])  # |rho_{g1P,g2}|

        # Coherence should develop (be non-zero)
        assert (
            coherence_magnitude > 1e-3
        ), f"Insufficient coherence developed: {coherence_magnitude}"

        # Check that coherence evolves over time (not constant)
        mid_time_idx = len(pulse_time) // 2
        mid_rho = output.states[mid_time_idx].full()
        mid_coherence = np.abs(mid_rho[2, 5])

        # At least some evolution should occur
        coherence_change = np.abs(coherence_magnitude - mid_coherence)
        assert coherence_change > 1e-6, "Coherence does not evolve during pulse"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
