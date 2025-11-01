import concurrent.futures
import multiprocessing
import sys

import numpy as np
from scipy.integrate import trapezoid
from qutip import mesolve, tensor, qeye, destroy

from src.modules.cavity import (
    cav_collapse_ops,
)
from src.modules.simulation import Simulation
from src.modules.correlation_functions import (
    exp_eval_fixed_start,
    rho_evo_fixed_start,
    rho_evo_floating_start_finish,
    exp_eval_floating_start_finish,
)
from src.modules.integration_functions import trapz_integral_real_imaginary
from src.modules.photon_correlation_utils import (
    generate_time_correlator_eval,
    plot_density_matrix_correlations,
    plot_n2_density_matrix_correlation,
)

# Define the ground states
ground_states = {
    "g1M": 0,
    "g1": 1,
    "g1P": 2,
    "g2MM": 3,
    "g2M": 4,
    "g2": 5,
    "g2P": 6,
    "g2PP": 7,
}

# Define the photon Fock states
fock_states = ["|0,0⟩", "|0,1⟩", "|1,0⟩", "|1,1⟩"]

# Generate the tensor product state list
tensor_product_states = []
for ground_state in ground_states:
    for fock_state in fock_states:
        tensor_product_states.append(f"|{ground_state}⟩ ⊗ {fock_state}")

# Initialize a matrix to store ket-bra combinations
matrix_size = len(tensor_product_states)
ket_bra_matrix = np.empty((matrix_size, matrix_size), dtype=object)

# Fill the matrix with ket-bra combinations
for i, bra in enumerate(tensor_product_states):
    for j, ket in enumerate(tensor_product_states):
        ket_bra_matrix[i, j] = f"⟨{bra}|{ket}⟩"


class TimeBinPhotonCorrelations:

    def __init__(
        self,
        save_dir,
        _x_states,
        _ground_states,
        _omega_stirap_early,
        _omega_stirap_late,
        _length_stirap,
        _n_steps_vst,
        _cubic_spline_trapz_smoothing=True,
        _vstirap_pulse_shape="sinsquared",
        _vst_ramp_up_time=0.2,
        _vst_ramp_down_time=0.2,
        _rot_omega_1=573.4,
        _rot_omega_2=754.1772283803055,
        _rot_omega_3=1139,
        _rot_omega_4=1459,
        _two_photon_det_1=0,
        _two_photon_det_2=0,
        _two_photon_det_3=0,
        _two_photon_det_4=0,
        _n_steps_rot=2500,
        _bfield_split="0p07",
        _n_start=2,
        _spont_emission=True,
        _far_detuned_raman=False,
        _chirped=True,
    ):

        self.save_dir = save_dir
        self._x_states = _x_states
        self._ground_states = _ground_states
        self.omega_stirap = [_omega_stirap_early, _omega_stirap_late]
        self.length_stirap = _length_stirap
        self.n_steps_rot = _n_steps_rot
        self.n_steps_vst = _n_steps_vst
        self.bfield_split = _bfield_split
        self.n_start = _n_start
        self.two_photon_det = [
            _two_photon_det_1,
            _two_photon_det_2,
            _two_photon_det_3,
            _two_photon_det_4,
        ]
        self.omega_rot = [_rot_omega_1, _rot_omega_2, _rot_omega_3, _rot_omega_4]
        self.spont_emission = _spont_emission
        self.chirped_pulses = _chirped
        self.vstirap_pulse_shape = _vstirap_pulse_shape
        self.far_detuned_raman = _far_detuned_raman
        self.vst_rise_time = _vst_ramp_up_time
        self.vst_fall_time = _vst_ramp_down_time
        self.cubic_spline_smoothing = _cubic_spline_trapz_smoothing

        # Initialize simulation
        self.rb_atom_sim = Simulation(
            cavity=True,
            bfieldsplit=self.bfield_split,
            ground_states=self._ground_states,
            x_states=self._x_states,
            show_details=True,
        )

        # Create collapse operator list
        self.c_op_list = []
        self.create_collapse_operators()
        # we need at least 12 collapse operators for up to 3 photons
        self.full_c_op_list = [self.c_op_list[:] for _ in range(12)]

        # Create photon operators
        self.create_photon_operators()

        # Create Hamiltonians and time evolution lists
        self.H_list, self.H_sim_time_list = self.create_hamiltonians()

        # Set multiprocessing start method
        if sys.platform == "win32":
            multiprocessing.set_start_method(
                "spawn", force=True
            )  # Use 'spawn' for Windows
        else:
            try:
                multiprocessing.set_start_method("fork")
            except RuntimeError:
                # If the start method is already set, we can ignore this
                pass  # Use 'fork' for Unix-based systems

        # Define the final expectation operator for the off-diagonal elements of the two photon density matrix
        self.exp_final_a_dag = (
            self.rb_atom_sim.kb_class.get_ket("g2MM", 0, 0)
            * self.rb_atom_sim.kb_class.get_ket("g2PP", 0, 1).dag()
        )
        self.exp_final_a = self.exp_final_a_dag.dag()

    def reset_simulation(self):
        # Simulation initialization logic here
        self.rb_atom_sim.reset()

    def create_collapse_operators(self):
        # Add cavity collapse operators
        self.c_op_list += cav_collapse_ops(self.rb_atom_sim.kappa, self._ground_states)

        # Add spontaneous emission operators for both D2 and D1 lines
        if self.spont_emission:
            self.c_op_list += self.rb_atom_sim.rb_atom.spont_em_ops(
                self._ground_states
            )[
                0
            ]  # D2 line
            self.c_op_list += self.rb_atom_sim.rb_atom.spont_em_ops(
                self._ground_states
            )[
                1
            ]  # D1 line

    def create_far_detuned_collapse_operators(self, pol, det, omega):
        return self.rb_atom_sim.rb_atom.spont_em_ops_far_detuned(
            self._ground_states, pol, omega, det
        )

    def create_photon_operators(self):
        # Define the truncation of Fock states
        N = 2
        M = len(self._ground_states)

        # Define photon creation/annihilation operators
        self.aX = tensor(qeye(M), destroy(N), qeye(N))
        self.aY = tensor(qeye(M), qeye(N), destroy(N))
        self.anX = self.aX.dag() * self.aX
        self.anY = self.aY.dag() * self.aY

    def create_hamiltonians(self):
        # Define parameters for the Hamiltonians
        vst_params = {
            "OmegaStirap_1": self.omega_stirap[0],
            "OmegaStirap_2": self.omega_stirap[1],
            "pulse_shape": self.vstirap_pulse_shape,
            "lengthStirap": self.length_stirap,
            "delta_VST_1": 0,
            "delta_VST_2": 0,
        }

        if self.far_detuned_raman:
            first_params = {
                "cg_1": self.rb_atom_sim.rb_atom.CG_d1g2x1,
                "pol_1": "pi",
                "det_zeeman_1": 0,
                "cg_2": self.rb_atom_sim.rb_atom.CG_d1g1Mx1,
                "pol_2": "sigmaP",
                "det_zeeman_2": -self.rb_atom_sim.rb_atom.deltaZ,
                "psi_0": self.rb_atom_sim.kb_class.get_ket_atomic("g2"),
                "psi_des": 1
                / np.sqrt(2)
                * (
                    self.rb_atom_sim.kb_class.get_ket_atomic("g1M")
                    - self.rb_atom_sim.kb_class.get_ket_atomic("g2")
                ),
                "coherence_indices": [0, 5],
                "two_photon_det": self.two_photon_det[0],
                "det_centre": -100000,
                "rise_time": 0.025,
                "amp_scaling": 1.65587099,
                "pulse_length": 0.125,
                # "rel_phase": 1.595729601823387,
                "rel_phase": self.two_photon_det[0],
                "dead_time": 0.125,
            }

            first_params["det_1"] = (
                first_params["det_centre"] * 2 * np.pi + first_params["det_zeeman_1"]
            )
            first_params["amp_1"] = (
                first_params["amp_scaling"]
                * np.sqrt(2 * np.abs(first_params["det_centre"]))
                * 2
                * np.pi
                / first_params["cg_1"]
            )
            first_params["amp_2"] = (
                first_params["amp_scaling"]
                * np.sqrt(2 * np.abs(first_params["det_centre"]))
                * 2
                * np.pi
                / first_params["cg_2"]
            )
            first_params["det_2"] = (
                first_params["det_1"]
                - self.rb_atom_sim.rb_atom.getrb_gs_splitting()
                + first_params["det_zeeman_2"]
            )

            # Repeat the same structure for second_params, third_params, and fourth_params
            second_params = {
                "cg_1": self.rb_atom_sim.rb_atom.CG_d1g2x1,
                "pol_1": "pi",
                "det_zeeman_1": 0,
                "cg_2": self.rb_atom_sim.rb_atom.CG_d1g1Px1,
                "pol_2": "sigmaM",
                "det_zeeman_2": self.rb_atom_sim.rb_atom.deltaZ,
                "psi_0": 1
                / np.sqrt(2)
                * (
                    self.rb_atom_sim.kb_class.get_ket_atomic("g2")
                    - self.rb_atom_sim.kb_class.get_ket_atomic("g2MM")
                ),
                "psi_des": 1
                / np.sqrt(2)
                * (
                    self.rb_atom_sim.kb_class.get_ket_atomic("g1P")
                    - self.rb_atom_sim.kb_class.get_ket_atomic("g2MM")
                ),
                "coherence_indices": [2, 3],
                "two_photon_det": self.two_photon_det[1],
                "det_centre": -100000,
                "rise_time": 0.025,
                "amp_scaling": 2.21831585,
                "pulse_length": 0.125,
                "dead_time": 0.125,
                # "rel_phase": 3.9893240045584677,
                "rel_phase": self.two_photon_det[1],
            }

            second_params["det_1"] = (
                second_params["det_centre"] * 2 * np.pi + second_params["det_zeeman_1"]
            )
            second_params["amp_1"] = (
                second_params["amp_scaling"]
                * np.sqrt(2 * np.abs(second_params["det_centre"]))
                * 2
                * np.pi
                / second_params["cg_1"]
            )
            second_params["amp_2"] = (
                second_params["amp_scaling"]
                * np.sqrt(2 * np.abs(second_params["det_centre"]))
                * 2
                * np.pi
                / second_params["cg_2"]
            )
            second_params["det_2"] = (
                second_params["det_1"]
                - self.rb_atom_sim.rb_atom.getrb_gs_splitting()
                + second_params["det_zeeman_2"]
            )

            third_params = {
                "cg_1": self.rb_atom_sim.rb_atom.CG_d1g2MMx2MM,
                "pol_1": "pi",
                "det_zeeman_1": 0,
                "cg_2": self.rb_atom_sim.rb_atom.CG_d1g1Mx2MM,
                "pol_2": "sigmaM",
                "det_zeeman_2": -np.pi * self.rb_atom_sim.rb_atom.deltaZ,
                "psi_0": 1
                / np.sqrt(2)
                * (
                    self.rb_atom_sim.kb_class.get_ket_atomic("g2MM")
                    - self.rb_atom_sim.kb_class.get_ket_atomic("g2PP")
                ),
                "psi_des": 1
                / np.sqrt(2)
                * (
                    self.rb_atom_sim.kb_class.get_ket_atomic("g1M")
                    - self.rb_atom_sim.kb_class.get_ket_atomic("g2PP")
                ),
                "coherence_indices": [0, 7],
                "two_photon_det": self.two_photon_det[2],
                "det_centre": -100000,
                "rise_time": 0.025,
                "amp_scaling": 1.52701423,
                "pulse_length": 0.25,
                # "rel_phase": 4.9,
                "rel_phase": self.two_photon_det[2],
            }

            third_params["det_1"] = (
                third_params["det_centre"] * 2 * np.pi + third_params["det_zeeman_1"]
            )
            third_params["amp_1"] = (
                third_params["amp_scaling"]
                * np.sqrt(2 * np.abs(third_params["det_centre"]))
                * 2
                * np.pi
                / third_params["cg_1"]
            )
            third_params["amp_2"] = (
                third_params["amp_scaling"]
                * np.sqrt(2 * np.abs(third_params["det_centre"]))
                * 2
                * np.pi
                / third_params["cg_2"]
            )
            third_params["det_2"] = (
                third_params["det_1"]
                - self.rb_atom_sim.rb_atom.getrb_gs_splitting()
                + third_params["det_zeeman_2"]
            )

            fourth_params = {
                "cg_1": self.rb_atom_sim.rb_atom.CG_d1g2PPx2PP,
                "pol_1": "pi",
                "det_zeeman_1": 0,
                "cg_2": self.rb_atom_sim.rb_atom.CG_d1g1Px2PP,
                "pol_2": "sigmaP",
                "det_zeeman_2": np.pi * self.rb_atom_sim.rb_atom.deltaZ,
                "psi_0": 1
                / np.sqrt(2)
                * (
                    self.rb_atom_sim.kb_class.get_ket_atomic("g2MM")
                    - self.rb_atom_sim.kb_class.get_ket_atomic("g2PP")
                ),
                "psi_des": 1
                / np.sqrt(2)
                * (
                    self.rb_atom_sim.kb_class.get_ket_atomic("g1P")
                    - self.rb_atom_sim.kb_class.get_ket_atomic("g2MM")
                ),
                "coherence_indices": [2, 3],
                "two_photon_det": self.two_photon_det[3],
                "det_centre": -100000,
                "rise_time": 0.025,
                "amp_scaling": 1.52701423,
                "pulse_length": 0.25,
                # "rel_phase": 4.9,
                "rel_phase": self.two_photon_det[3],
            }

            fourth_params["det_1"] = (
                fourth_params["det_centre"] * 2 * np.pi + fourth_params["det_zeeman_1"]
            )
            fourth_params["amp_1"] = (
                fourth_params["amp_scaling"]
                * np.sqrt(2 * np.abs(fourth_params["det_centre"]))
                * 2
                * np.pi
                / fourth_params["cg_1"]
            )
            fourth_params["amp_2"] = (
                fourth_params["amp_scaling"]
                * np.sqrt(2 * np.abs(fourth_params["det_centre"]))
                * 2
                * np.pi
                / fourth_params["cg_2"]
            )
            fourth_params["det_2"] = (
                fourth_params["det_1"]
                - self.rb_atom_sim.rb_atom.getrb_gs_splitting()
                + fourth_params["det_zeeman_2"]
            )

            # add collapse ops to terms as necessary
            if self.n_start == 1:
                # currently throwing an error

                self.full_c_op_list[0] += self.create_far_detuned_collapse_operators(
                    first_params["pol_1"], first_params["det_1"], first_params["amp_1"]
                )
                self.full_c_op_list[0] += self.create_far_detuned_collapse_operators(
                    first_params["pol_2"], first_params["det_2"], first_params["amp_2"]
                )

                self.full_c_op_list[2] += self.create_far_detuned_collapse_operators(
                    second_params["pol_1"],
                    second_params["det_1"],
                    second_params["amp_1"],
                )
                self.full_c_op_list[2] += self.create_far_detuned_collapse_operators(
                    second_params["pol_2"],
                    second_params["det_2"],
                    second_params["amp_2"],
                )

                self.full_c_op_list[4] += self.create_far_detuned_collapse_operators(
                    third_params["pol_1"], third_params["det_1"], third_params["amp_1"]
                )
                self.full_c_op_list[4] += self.create_far_detuned_collapse_operators(
                    third_params["pol_2"], third_params["det_2"], third_params["amp_2"]
                )

                self.full_c_op_list[6] += self.create_far_detuned_collapse_operators(
                    fourth_params["pol_1"],
                    fourth_params["det_1"],
                    fourth_params["amp_1"],
                )
                self.full_c_op_list[6] += self.create_far_detuned_collapse_operators(
                    fourth_params["pol_2"],
                    fourth_params["det_2"],
                    fourth_params["amp_2"],
                )

                self.full_c_op_list[8] += self.create_far_detuned_collapse_operators(
                    third_params["pol_1"], third_params["det_1"], third_params["amp_1"]
                )
                self.full_c_op_list[8] += self.create_far_detuned_collapse_operators(
                    third_params["pol_2"], third_params["det_2"], third_params["amp_2"]
                )

                self.full_c_op_list[10] += self.create_far_detuned_collapse_operators(
                    fourth_params["pol_1"],
                    fourth_params["det_1"],
                    fourth_params["amp_1"],
                )
                self.full_c_op_list[10] += self.create_far_detuned_collapse_operators(
                    fourth_params["pol_2"],
                    fourth_params["det_2"],
                    fourth_params["amp_2"],
                )

            elif self.n_start == 2:
                # currently throwing an error

                self.full_c_op_list[1] += self.create_far_detuned_collapse_operators(
                    third_params["pol_1"], third_params["det_1"], third_params["amp_1"]
                )
                self.full_c_op_list[1] += self.create_far_detuned_collapse_operators(
                    third_params["pol_2"], third_params["det_2"], third_params["amp_2"]
                )

                self.full_c_op_list[3] += self.create_far_detuned_collapse_operators(
                    fourth_params["pol_1"],
                    fourth_params["det_1"],
                    fourth_params["amp_1"],
                )
                self.full_c_op_list[3] += self.create_far_detuned_collapse_operators(
                    fourth_params["pol_2"],
                    fourth_params["det_2"],
                    fourth_params["amp_2"],
                )

                self.full_c_op_list[5].extend(
                    self.create_far_detuned_collapse_operators(
                        third_params["pol_1"],
                        third_params["det_1"],
                        third_params["amp_1"],
                    )
                )
                self.full_c_op_list[5].extend(
                    self.create_far_detuned_collapse_operators(
                        third_params["pol_2"],
                        third_params["det_2"],
                        third_params["amp_2"],
                    )
                )

                self.full_c_op_list[7].extend(
                    self.create_far_detuned_collapse_operators(
                        fourth_params["pol_1"],
                        fourth_params["det_1"],
                        fourth_params["amp_1"],
                    )
                )
                self.full_c_op_list[7].extend(
                    self.create_far_detuned_collapse_operators(
                        fourth_params["pol_2"],
                        fourth_params["det_2"],
                        fourth_params["amp_2"],
                    )
                )

                self.full_c_op_list[9].extend(
                    self.create_far_detuned_collapse_operators(
                        third_params["pol_1"],
                        third_params["det_1"],
                        third_params["amp_1"],
                    )
                )
                self.full_c_op_list[9].extend(
                    self.create_far_detuned_collapse_operators(
                        third_params["pol_2"],
                        third_params["det_2"],
                        third_params["amp_2"],
                    )
                )

                self.full_c_op_list[11].extend(
                    self.create_far_detuned_collapse_operators(
                        fourth_params["pol_1"],
                        fourth_params["det_1"],
                        fourth_params["amp_1"],
                    )
                )
                self.full_c_op_list[11].extend(
                    self.create_far_detuned_collapse_operators(
                        fourth_params["pol_2"],
                        fourth_params["det_2"],
                        fourth_params["amp_2"],
                    )
                )

        else:
            first_params = {
                "param_1": 0.7293,
                "laser_amplitude": self.omega_rot[0],
                "detuning": -37.92,
                "two_photon_detuning": self.two_photon_det[0],
                "duration": 0.1603,
                "detuning_magn": 1,
                "pulse_shape": "fstirap",
                "rotation_number": 1,
            }

            second_params = {
                "param_1": 10.32611058680373,
                "laser_amplitude": self.omega_rot[1],
                "detuning": -1.3177928786079605,
                "two_photon_detuning": self.two_photon_det[1],
                "duration": 0.25,
                "detuning_magn": 1,
                "_n": 3,
                "_c": 0.25 / 3,
                "pulse_shape": "masked",
                "rotation_number": 2,
            }

            third_params = {
                "param_1": 14.92,
                "laser_amplitude": self.omega_rot[2],
                "detuning": -33.82,
                "two_photon_detuning": self.two_photon_det[2],
                "duration": 0.25,
                "detuning_magn": 1,
                "_n": 6,
                "_c": 0.25 / 3,
                "pulse_shape": "masked",
                "rotation_number": 3,
            }

            fourth_params = {
                "param_1": 14.53,
                "laser_amplitude": self.omega_rot[3],
                "detuning": -40.28,
                "two_photon_detuning": self.two_photon_det[3],
                "duration": 0.25,
                "detuning_magn": 1,
                "_n": 6,
                "_c": 0.25 / 3,
                "pulse_shape": "masked",
                "rotation_number": 4,
            }

        # Generate the Hamiltonians for VST and rotations
        H_VStirap_1, H_VStirap_2, t_vst, args_hams_VStirap_1, args_hams_VStirap_2 = (
            self.rb_atom_sim.generate_timebin_vst_hamiltonian(
                vst_params,
                _n_steps_vst=self.n_steps_vst,
                _ramp_up_time=self.vst_rise_time,
                _ramp_down_time=self.vst_fall_time,
            )
        )

        if self.far_detuned_raman:
            H_rot_1, t_rot1, args_hams_rot_1 = (
                self.rb_atom_sim.generate_timebin_far_detuned_raman_hamiltonian(
                    first_params
                )
            )
            H_rot_2, t_rot2, args_hams_rot_2 = (
                self.rb_atom_sim.generate_timebin_far_detuned_raman_hamiltonian(
                    second_params
                )
            )
            H_rot_3, t_rot3, args_hams_rot_3 = (
                self.rb_atom_sim.generate_timebin_far_detuned_raman_hamiltonian(
                    third_params
                )
            )
            H_rot_4, t_rot4, args_hams_rot_4 = (
                self.rb_atom_sim.generate_timebin_far_detuned_raman_hamiltonian(
                    fourth_params
                )
            )
        else:
            H_rot_1, t_rot1, args_hams_rot_1 = (
                self.rb_atom_sim.generate_timebin_rotation_hamiltonian(
                    first_params,
                    _chirped=self.chirped_pulses,
                    _n_steps=self.n_steps_rot,
                )
            )
            H_rot_2, t_rot2, args_hams_rot_2 = (
                self.rb_atom_sim.generate_timebin_rotation_hamiltonian(
                    second_params,
                    _chirped=self.chirped_pulses,
                    _n_steps=self.n_steps_rot,
                )
            )
            H_rot_3, t_rot3, args_hams_rot_3 = (
                self.rb_atom_sim.generate_timebin_rotation_hamiltonian(
                    third_params,
                    _chirped=self.chirped_pulses,
                    _n_steps=self.n_steps_rot,
                )
            )
            H_rot_4, t_rot4, args_hams_rot_4 = (
                self.rb_atom_sim.generate_timebin_rotation_hamiltonian(
                    fourth_params,
                    _chirped=self.chirped_pulses,
                    _n_steps=self.n_steps_rot,
                )
            )

        # Combine Hamiltonians and time lists
        if self.n_start == 2:
            H_list = [H_VStirap_1, H_rot_4, H_VStirap_2, H_rot_3]
            H_sim_time_list = [t_vst, t_rot4, t_vst, t_rot3]
        elif self.n_start == 1:
            H_list = [
                H_rot_1,
                H_VStirap_1,
                H_rot_2,
                H_VStirap_2,
                H_rot_3,
                H_VStirap_1,
                H_rot_4,
                H_VStirap_2,
                H_rot_3,
            ]
            H_sim_time_list = [
                t_rot1,
                t_vst,
                t_rot2,
                t_vst,
                t_rot3,
                t_vst,
                t_rot4,
                t_vst,
                t_rot3,
            ]
        else:
            raise ValueError("Invalid value for n_start. Must be 1 or 2.")

        return H_list, H_sim_time_list

    # Move the compute_rho_off_diag method to the class level
    def compute_rho_off_diag_n1(
        self, _j, _t2, _H_list, _H_sim_time_list, _rho_start_list, _t_bin
    ):
        return rho_evo_floating_start_finish(
            _H_list,
            _H_sim_time_list,
            _rho_start_list[_j],
            _t2,
            _t2 + _t_bin,
            self.full_c_op_list,
            1,
            self.aY.dag(),
        )

    def gen_n1_density_matrix(
        self, _n_off_diag_correlator_eval, _parall=False, _plot=False
    ):
        # Define initial state
        if self.n_start == 2:
            psi0 = (
                1
                / np.sqrt(2)
                * (
                    self.rb_atom_sim.kb_class.get_ket("g2PP", 0, 0)
                    - self.rb_atom_sim.kb_class.get_ket("g1M", 0, 0)
                )
            )
            # List of Hamiltonians and times
            H_list = [self.H_list[0], self.H_list[1], self.H_list[2], self.H_list[3]]
            H_sim_time_list = self.H_sim_time_list
        elif self.n_start == 1:
            psi0 = mesolve(
                self.H_list[0],
                self.rb_atom_sim.kb_class.get_ket("g2", 0, 0),
                self.H_sim_time_list[0],
                self.full_c_op_list,
            ).states[-1]
            H_list = [self.H_list[1], self.H_list[2], self.H_list[3], self.H_list[4]]
            H_sim_time_list = self.H_sim_time_list[1:]
        else:
            raise ValueError("Invalid value for n_start. Must be 1 or 2.")

        # Time bins and correlator evaluation times
        # sample more finely across the photon duration and add final point corresponding to the end of the time bin
        t_correlator_eval, t_bin = generate_time_correlator_eval(
            _n_off_diag_correlator_eval, t_vst=self.length_stirap, t_rot=0.25
        )

        # Calculate density matrices
        rho_start_list = rho_evo_fixed_start(
            H_list, H_sim_time_list, psi0, t_correlator_eval, self.full_c_op_list, 1, 1
        )
        rho_start_list.insert(0, psi0)  # Add the initial density matrix

        # self.rb_atom_sim.rb_atom.plotter_atomstate_population(self.rb_atom_sim.ketbras, rho_start_list, t_correlator_eval, True)

        # Diagonal density matrix expectations
        exp_values_diag_zero = mesolve(
            H_list[0], psi0, H_sim_time_list[0], self.full_c_op_list[0], [self.anY]
        ).expect[0]
        exp_values_diag_one = mesolve(
            H_list[2],
            rho_start_list[-1],
            H_sim_time_list[2],
            self.full_c_op_list[2],
            [self.anY],
        ).expect[0]

        # Off-diagonal density matrices and coherence extraction
        # exp_values_off_diag_zero = []
        exp_values_off_diag_one = []

        if _parall:
            # Parallel execution using ProcessPoolExecutor
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Pass the arguments explicitly to the method
                results = executor.map(
                    self.compute_rho_off_diag_n1,
                    range(len(t_correlator_eval)),
                    t_correlator_eval,
                    [H_list] * len(t_correlator_eval),
                    [H_sim_time_list] * len(t_correlator_eval),
                    [rho_start_list] * len(t_correlator_eval),
                    [t_bin] * len(t_correlator_eval),
                )

                exp_values_off_diag_one = list(results)
        else:
            # Sequential execution
            for j, t2 in enumerate(t_correlator_eval):
                rho_off_diag_one = self.compute_rho_off_diag_n1(
                    j, t2, H_list, H_sim_time_list, rho_start_list, t_bin
                )
                exp_values_off_diag_one.append(rho_off_diag_one)

            # rho_off_diag_zero = rho_evo_floating_start_finish(H_list, H_sim_time_list, rho_start_list[j], t2, t2 + t_bin, self.c_op_list, self.aY, 1)
            # exp_values_off_diag_zero.append(rho_off_diag_zero)

        # TODO adjust indices based on input states
        coherence_01 = [
            exp_values_off_diag_one[time_point][29, 12]
            for time_point in range(len(exp_values_off_diag_one))
        ]
        # coherence_10 = [exp_values_off_diag_zero[time_point][12, 29] for time_point in range(len(exp_values_off_diag_zero))]

        # Calculate integrals for diagonal and off-diagonal elements
        int_off_diag_01_re, int_off_diag_01_im = trapz_integral_real_imaginary(
            t_correlator_eval, coherence_01
        )
        int_off_diag_10_re, int_off_diag_10_im = trapz_integral_real_imaginary(
            t_correlator_eval, np.conjugate(coherence_01)
        )
        int_diag_one_re, int_diag_one_im = trapz_integral_real_imaginary(
            H_sim_time_list[2], exp_values_diag_one
        )
        int_diag_zero_re, int_diag_zero_im = trapz_integral_real_imaginary(
            H_sim_time_list[0], exp_values_diag_zero
        )

        # Optionally generate the plot
        if _plot:
            plot_density_matrix_correlations(
                H_sim_time_list,
                exp_values_diag_zero,
                exp_values_diag_one,
                t_correlator_eval,
                coherence_01,
                self.save_dir,
                self.n_start,
                self.length_stirap,
                self.bfield_split,
            )

        # Return all the computed integrals
        return {
            "n_start": self.n_start,
            "bfield_split": self.bfield_split,
            "length_stirap": self.length_stirap,
            "two_photon_det": self.two_photon_det,
            "omega_stirap": self.omega_stirap,
            "omega_rot_stirap": self.omega_rot,
            "spontaneous_emission": self.spont_emission,
            "chirped_pulse": self.chirped_pulses,
            "vstirap_pulse_shape": self.vstirap_pulse_shape,
            "vst_rise_time": self.vst_rise_time,
            "vst_fall_time": self.vst_fall_time,
            "int_diag_00_re": int_diag_zero_re,
            "int_diag_00_im": int_diag_zero_im,
            "int_diag_11_re": int_diag_one_re,
            "int_diag_11_im": int_diag_one_im,
            "int_off_diag_01_re": int_off_diag_01_re,
            "int_off_diag_01_im": int_off_diag_01_im,
            "int_off_diag_10_re": int_off_diag_10_re,
            "int_off_diag_10_im": int_off_diag_10_im,
            "n_time_steps": _n_off_diag_correlator_eval,
        }

    # Define a wrapper function for parallel execution for n=2
    def compute_rho_diag_eeee(
        self,
        i,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_early,
        t_correlator_eval_early,
        t_bin,
    ):
        # Step 1: Evolve the initial density matrix
        rho_evolve_1 = rho_evo_floating_start_finish(
            H_list[:5],
            H_sim_time_list[:5],
            rho_start_list_early[i],
            t1,
            2 * t_bin,
            self.c_op_list,
            self.aY,
            self.aY.dag(),
        )
        # Step 2: Evaluate the final expectation values
        final_exp_values = exp_eval_fixed_start(
            H_list[4:],
            H_sim_time_list[4:],
            rho_evolve_1,
            t_correlator_eval_early,
            self.c_op_list,
            1,
            1,
            self.anY,
        )
        return i, final_exp_values

    def compute_rho_diag_llll(
        self,
        i,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_late,
        t_correlator_eval_late,
        t_bin,
    ):
        # Step 1: Evolve the initial density matrix
        rho_evolve_1 = rho_evo_floating_start_finish(
            H_list[:5],
            H_sim_time_list[:5],
            rho_start_list_late[i],
            t1,
            2 * t_bin,
            self.c_op_list,
            self.aY,
            self.aY.dag(),
        )
        # Step 2: Evaluate the final expectation values
        final_exp_values = exp_eval_fixed_start(
            H_list[4:],
            H_sim_time_list[4:],
            rho_evolve_1,
            t_correlator_eval_late,
            self.c_op_list,
            1,
            1,
            self.anY,
        )

        return i, final_exp_values

    def compute_rho_leel(
        self,
        i,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_late,
        t_correlator_eval_early,
        t_bin,
    ):
        # Evolve the initial density matrix
        rho_evolve_1 = rho_evo_floating_start_finish(
            H_list[:5],
            H_sim_time_list[:5],
            rho_start_list_late[i],
            t1,
            2 * t_bin,
            self.c_op_list,
            self.aY,
            self.aY.dag(),
        )

        # Evaluate the final expectation values
        final_exp_values = exp_eval_fixed_start(
            H_list[4:],
            H_sim_time_list[4:],
            rho_evolve_1,
            t_correlator_eval_early,
            self.c_op_list,
            1,
            1,
            self.anY,
        )
        return i, final_exp_values

    def compute_rho_elle(
        self,
        i,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_early,
        t_correlator_eval_late,
        t_bin,
    ):
        # Evolve the initial density matrix
        rho_evolve_1 = rho_evo_floating_start_finish(
            H_list[:5],
            H_sim_time_list[:5],
            rho_start_list_early[i],
            t1,
            2 * t_bin,
            self.c_op_list,
            self.aY,
            self.aY.dag(),
        )

        # Evaluate the final expectation values
        final_exp_values = exp_eval_fixed_start(
            H_list[4:],
            H_sim_time_list[4:],
            rho_evolve_1,
            t_correlator_eval_late,
            self.c_op_list,
            1,
            1,
            self.anY,
        )
        return i, final_exp_values

    def compute_rho_eell(
        self,
        _index,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_early,
        t_correlator_eval_early,
        t_bin,
    ):
        # Evolve the initial density matrix
        rho_evolve_1 = rho_evo_floating_start_finish(
            H_list[:5],
            H_sim_time_list[:5],
            rho_start_list_early[_index],
            t1,
            t1 + t_bin,
            self.c_op_list,
            1,
            self.aY.dag(),
        )

        final_exp_values = np.zeros((len(t_correlator_eval_early)), dtype=np.complex128)

        for j, t2 in enumerate(t_correlator_eval_early):
            rho_evolve_2 = rho_evo_floating_start_finish(
                H_list[:7],
                H_sim_time_list[:7],
                rho_evolve_1,
                t1 + t_bin,
                2 * t_bin + t2,
                self.c_op_list,
                self.aY,
                1,
            )
            final_rho = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_2,
                2 * t_bin + t2,
                3 * t_bin + t2,
                self.c_op_list,
                1,
                self.aY.dag(),
                debug=False,
            )
            # here we are evaluating an annihilation operator at the final point
            final_exp_values[j] = final_rho[29, 12]

        return _index, final_exp_values

    def compute_rho_elel(
        self,
        _index,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_early,
        t_correlator_eval_early,
        t_bin,
    ):
        # Evolve the initial density matrix
        rho_evolve_1 = rho_evo_floating_start_finish(
            H_list[:5],
            H_sim_time_list[:5],
            rho_start_list_early[_index],
            t1,
            t1 + t_bin,
            self.c_op_list,
            1,
            self.aY.dag(),
        )

        final_exp_values = np.zeros((len(t_correlator_eval_early)), dtype=np.complex128)

        for j, t2 in enumerate(t_correlator_eval_early):
            rho_evolve_2 = rho_evo_floating_start_finish(
                H_list[:7],
                H_sim_time_list[:7],
                rho_evolve_1,
                t1 + t_bin,
                2 * t_bin + t2,
                self.c_op_list,
                self.aY,
                1,
            )
            final_rho = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_2,
                2 * t_bin + t2,
                3 * t_bin + t2,
                self.c_op_list,
                self.aY,
                1,
                debug=False,
            )
            # here we are evaluating a creation operator at the final point
            final_exp_values[j] = final_rho[12, 29]

        return _index, final_exp_values

    def gen_n2_density_matrix_element(
        self, _n_correlator_eval, _element, _plot=False, _parall=False, _debug=False
    ):
        """generate two photon density matrix elements for the given _element
        _n_correlator_eval: number of points to evaluate the correlator over
        _element: the element of the two photon density matrix to calculate IN OPERATOR ORDERING (e.g. EEEE, LLLL, EELL, LEEL)
        _plot: whether to plot the results
        _parall: whether to use parallel processing
        _debug: whether to print debug information
        """
        # Define initial state
        if self.n_start == 2:
            psi0 = (
                1
                / np.sqrt(2)
                * (
                    self.rb_atom_sim.kb_class.get_ket("g2PP", 0, 0)
                    - self.rb_atom_sim.kb_class.get_ket("g1M", 0, 0)
                )
            )
            # List of Hamiltonians and times
            H_list = [
                self.H_list[0],
                self.H_list[1],
                self.H_list[2],
                self.H_list[3],
                self.H_list[0],
                self.H_list[1],
                self.H_list[2],
                self.H_list[3],
                self.H_list[0],
            ]
            H_sim_time_list = (
                self.H_sim_time_list + self.H_sim_time_list + [self.H_sim_time_list[0]]
            )
        elif self.n_start == 1:
            psi0 = mesolve(
                self.H_list[0],
                self.rb_atom_sim.kb_class.get_ket("g2", 0, 0),
                self.H_sim_time_list[0],
                self.c_op_list,
            ).states[-1]
            H_list = self.H_list[1:] + self.H_list[0]
            H_sim_time_list = self.H_sim_time_list[1:] + [self.H_sim_time_list[1]]
        else:
            raise ValueError("Invalid value for n_start. Must be 1 or 2.")

        # Time bins and correlator evaluation times
        # sample more finely across the photon duration and add final point corresponding to the end of the time bin
        t_correlator_eval_early, t_bin = generate_time_correlator_eval(
            _n_correlator_eval, t_vst=self.length_stirap, t_rot=0.25
        )
        t_correlator_eval_late, t_bin = generate_time_correlator_eval(
            _n_correlator_eval, t_vst=self.length_stirap, t_rot=0.25
        )
        t_correlator_eval_late = np.array([x + t_bin for x in t_correlator_eval_late])

        # Calculate density matrices for the evolution
        rho_start_list_early = rho_evo_fixed_start(
            H_list, H_sim_time_list, psi0, t_correlator_eval_early, self.c_op_list, 1, 1
        )
        rho_start_list_early.insert(0, psi0 * psi0.dag())

        # make sure the hamiltonian list is sliced correctly so the evolution is done with the correct hamiltonian elements
        rho_start_list_late = rho_evo_fixed_start(
            H_list, H_sim_time_list, psi0, t_correlator_eval_late, self.c_op_list, 1, 1
        )
        rho_start_list_late.insert(0, rho_start_list_early[-1])

        ########################################################################################################################################################

        # Diagonal density matrix expectations
        if _element == "EEEE":

            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_early + 2 * t_bin
            exp_values_eeee = np.zeros(
                (len(t_correlator_eval_early), len(t_correlator_eval_early)),
                dtype=np.complex128,
            )

            if _parall:
                # Parallel execution using ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_diag_eeee,
                        range(len(t_correlator_eval_early)),  # For the indices
                        t_correlator_eval_early,  # For each t1 in parallel
                        [H_list] * len(t_correlator_eval_early),
                        [H_sim_time_list] * len(t_correlator_eval_early),
                        [rho_start_list_early] * len(t_correlator_eval_early),
                        [t_correlator_eval_early] * len(t_correlator_eval_early),
                        [t_bin] * len(t_correlator_eval_early),
                    )

                    # Collect the results into the exp_values matrix
                    for _ind, final_exp_values in results:
                        exp_values_eeee[_ind][:] = final_exp_values

            else:
                # Sequential execution
                for _ind, t1 in enumerate(t_correlator_eval_early):
                    _, final_exp_values = self.compute_rho_diag_eeee(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_early[_ind],
                        t_correlator_eval_early,
                        t_bin,
                    )
                    exp_values_eeee[_ind][:] = final_exp_values

            exp_values = exp_values_eeee

        elif _element == "LLLL":
            t1_label = t_correlator_eval_late
            t2_label = t_correlator_eval_late + 2 * t_bin
            exp_values_llll = np.zeros(
                (len(t_correlator_eval_late), len(t_correlator_eval_late)),
                dtype=np.complex128,
            )

            if _parall:

                # Parallel execution using ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_diag_llll,
                        range(len(t_correlator_eval_late)),  # For the indices
                        t_correlator_eval_late,  # For each t1 in parallel
                        [H_list] * len(t_correlator_eval_late),
                        [H_sim_time_list] * len(t_correlator_eval_late),
                        [rho_start_list_late] * len(t_correlator_eval_late),
                        [t_correlator_eval_late] * len(t_correlator_eval_late),
                        [t_bin] * len(t_correlator_eval_late),
                    )

                    # Collect the results into the exp_values matrix
                    for _ind, final_exp_values in results:
                        exp_values_llll[_ind][:] = final_exp_values

            else:
                # Sequential execution
                for _ind, t1 in enumerate(t_correlator_eval_late):
                    _, final_exp_values = self.compute_rho_diag_llll(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_late[_ind],
                        t_correlator_eval_late,
                        t_bin,
                    )
                    exp_values_llll[_ind][:] = final_exp_values

            exp_values = exp_values_llll

        elif _element == "ELLE":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_late + 2 * t_bin
            exp_values_elle = np.zeros(
                (len(t_correlator_eval_early), len(t_correlator_eval_late)),
                dtype=np.complex128,
            )

            if _parall:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_elle,
                        range(
                            len(t_correlator_eval_early)
                        ),  # Parallelizing over indices
                        t_correlator_eval_early,  # Each t1
                        [H_list] * len(t_correlator_eval_early),
                        [H_sim_time_list] * len(t_correlator_eval_early),
                        [rho_start_list_late] * len(t_correlator_eval_early),
                        [t_correlator_eval_early] * len(t_correlator_eval_early),
                        [t_bin] * len(t_correlator_eval_early),
                    )

                    for i, final_exp_values in results:
                        exp_values_elle[i][:] = final_exp_values

            else:
                for _ind, t1 in enumerate(t_correlator_eval_early):
                    _, final_exp_values = self.compute_rho_diag_eeee(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_early[_ind],
                        t_correlator_eval_early,
                        t_bin,
                    )
                    exp_values_elle[i][:] = final_exp_values

            exp_values = exp_values_elle

        elif _element == "LEEL":
            t1_label = t_correlator_eval_late
            t2_label = t_correlator_eval_early + 2 * t_bin
            exp_values_leel = np.zeros(
                (len(t_correlator_eval_late), len(t_correlator_eval_early)),
                dtype=np.complex128,
            )

            if _parall:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_leel,
                        range(
                            len(t_correlator_eval_late)
                        ),  # Parallelizing over indices
                        t_correlator_eval_late,  # Each t1
                        [H_list] * len(t_correlator_eval_late),
                        [H_sim_time_list] * len(t_correlator_eval_late),
                        [rho_start_list_late] * len(t_correlator_eval_late),
                        [t_correlator_eval_early] * len(t_correlator_eval_late),
                        [t_bin] * len(t_correlator_eval_late),
                    )

                    for i, final_exp_values in results:
                        exp_values_leel[i][:] = final_exp_values
            else:
                for i, t1 in enumerate(t_correlator_eval_late):
                    rho_evolve_1 = rho_evo_floating_start_finish(
                        H_list[:5],
                        H_sim_time_list[:5],
                        rho_start_list_late[i],
                        t1,
                        2 * t_bin,
                        self.c_op_list,
                        self.aY,
                        self.aY.dag(),
                    )
                    final_exp_values = exp_eval_fixed_start(
                        H_list[3:],
                        H_sim_time_list[3:],
                        rho_evolve_1,
                        t_correlator_eval_early,
                        self.c_op_list,
                        1,
                        1,
                        self.anY,
                    )
                    exp_values_leel[i][:] = final_exp_values

            exp_values = exp_values_leel

        ##########################################################################################################################################################
        # Off-diagonal elements

        elif _element == "ELEL":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_early + 2 * t_bin
            exp_values_elel = np.zeros(
                (len(t_correlator_eval_early), len(t_correlator_eval_early)),
                dtype=np.complex128,
            )

            if _parall:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_elel,
                        range(
                            len(t_correlator_eval_early)
                        ),  # Parallelizing over indices
                        t_correlator_eval_early,  # Each t1
                        [H_list] * len(t_correlator_eval_early),
                        [H_sim_time_list] * len(t_correlator_eval_early),
                        [rho_start_list_early] * len(t_correlator_eval_early),
                        [t_correlator_eval_early] * len(t_correlator_eval_early),
                        [t_bin] * len(t_correlator_eval_early),
                    )

                    # TODO bit of a hack to make the lengths of the arrays match
                    for _ind, final_exp_values in results:
                        final_exp_values = np.array(final_exp_values)
                        if len(final_exp_values) < len(t_correlator_eval_early):
                            final_exp_values = np.insert(final_exp_values, 0, 0)
                        exp_values_elel[_ind][:] = final_exp_values

            else:
                for _ind, t1 in enumerate(t_correlator_eval_early):

                    rho_evolve_1 = rho_evo_floating_start_finish(
                        H_list[:5],
                        H_sim_time_list[:5],
                        rho_start_list_early[_ind],
                        t1,
                        t1 + t_bin,
                        self.c_op_list,
                        1,
                        self.aY.dag(),
                    )

                    # Loop through elements of rho_evolve_1 and print those with absolute value > 0.1
                    if _debug:
                        for row_idx, row in enumerate(rho_evolve_1):
                            for col_idx, val in enumerate(row[0]):
                                if abs(val) > 0.05:
                                    if row_idx <= 31 and col_idx <= 31:
                                        print(
                                            f"time_1: {t1+t_bin}, {ket_bra_matrix[row_idx,col_idx]} = {val}"
                                        )

                    for j, t2 in enumerate(t_correlator_eval_early):
                        rho_evolve_2 = rho_evo_floating_start_finish(
                            H_list[:7],
                            H_sim_time_list[:7],
                            rho_evolve_1,
                            t1 + t_bin,
                            2 * t_bin + t2,
                            self.c_op_list,
                            self.aY,
                            1,
                        )

                        # Loop through elements of rho_evolve_1 and print those with absolute value > 0.1
                        if _debug:
                            for row_idx, row in enumerate(rho_evolve_2):
                                for col_idx, val in enumerate(row[0]):
                                    if abs(val) > 0.05:
                                        if row_idx <= 31 and col_idx <= 31:
                                            print(
                                                f"time_2: {2 * t_bin + t2}, {ket_bra_matrix[row_idx,col_idx]} = {val}"
                                            )
                            final_exp_values = rho_evo_floating_start_finish(
                                H_list,
                                H_sim_time_list,
                                rho_evolve_2,
                                2 * t_bin + t2,
                                t2 + 3 * t_bin,
                                self.c_op_list,
                                self.aY,
                                1,
                            )
                            for row_idx, row in enumerate(final_exp_values):
                                for col_idx, val in enumerate(row[0]):
                                    if abs(val) > 0.05:
                                        if row_idx <= 31 and col_idx <= 31:
                                            print(
                                                f"time_3: {t2 + 3*t_bin} , {ket_bra_matrix[row_idx,col_idx]} = {val}"
                                            )

                        else:
                            final_exp_values = exp_eval_floating_start_finish(
                                H_list,
                                H_sim_time_list,
                                rho_evolve_2,
                                2 * t_bin + t2,
                                t2_label + t_bin,
                                self.c_op_list,
                                self.aY,
                                1,
                                self.exp_final_a_dag,
                            )

                            # TODO bit of a hack to make the lengths of the arrays match
                            if len(final_exp_values) < len(t_correlator_eval_early):
                                final_exp_values = np.insert(final_exp_values, 0, 0)

                            exp_values_elel[_ind][:] = final_exp_values
            exp_values = exp_values_elel

        elif _element == "EELL":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_early + 2 * t_bin
            exp_values_eell = np.zeros(
                (len(t_correlator_eval_early), len(t_correlator_eval_early)),
                dtype=np.complex128,
            )

            if _parall:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_eell,
                        range(
                            len(t_correlator_eval_early)
                        ),  # Parallelizing over indices
                        t_correlator_eval_early,  # Each t1
                        [H_list] * len(t_correlator_eval_early),
                        [H_sim_time_list] * len(t_correlator_eval_early),
                        [rho_start_list_early] * len(t_correlator_eval_early),
                        [t_correlator_eval_early] * len(t_correlator_eval_early),
                        [t_bin] * len(t_correlator_eval_early),
                    )

                    # TODO bit of a hack to make the lengths of the arrays match
                    for _ind, final_exp_values in results:
                        final_exp_values = np.array(final_exp_values)
                        if len(final_exp_values) < len(t_correlator_eval_early):
                            final_exp_values = np.insert(final_exp_values, 0, 0)
                        exp_values_eell[_ind][:] = final_exp_values

            else:
                for _ind, t1 in enumerate(t_correlator_eval_early):

                    rho_evolve_1 = rho_evo_floating_start_finish(
                        H_list[:5],
                        H_sim_time_list[:5],
                        rho_start_list_early[_ind],
                        t1,
                        t1 + t_bin,
                        self.c_op_list,
                        1,
                        self.aY.dag(),
                    )

                    # Loop through elements of rho_evolve_1 and print those with absolute value > 0.1
                    if _debug:
                        for row_idx, row in enumerate(rho_evolve_1):
                            for col_idx, val in enumerate(row[0]):
                                if abs(val) > 0.05:
                                    if row_idx <= 31 and col_idx <= 31:
                                        print(
                                            f"time_1: {t1+t_bin}, {ket_bra_matrix[row_idx,col_idx]} = {val}"
                                        )

                    for j, t2 in enumerate(t_correlator_eval_early):
                        rho_evolve_2 = rho_evo_floating_start_finish(
                            H_list[:7],
                            H_sim_time_list[:7],
                            rho_evolve_1,
                            t1 + t_bin,
                            2 * t_bin + t2,
                            self.c_op_list,
                            self.aY,
                            1,
                        )

                        # Loop through elements of rho_evolve_1 and print those with absolute value > 0.1
                        if _debug:
                            for row_idx, row in enumerate(rho_evolve_2):
                                for col_idx, val in enumerate(row[0]):
                                    if abs(val) > 0.05:
                                        if row_idx <= 31 and col_idx <= 31:
                                            print(
                                                f"time_2: {2 * t_bin + t2}, {ket_bra_matrix[row_idx,col_idx]} = {val}"
                                            )

                            final_exp_values = rho_evo_floating_start_finish(
                                H_list,
                                H_sim_time_list,
                                rho_evolve_2,
                                2 * t_bin + t2,
                                t2 + 3 * t_bin,
                                self.c_op_list,
                                1,
                                self.aY.dag(),
                            )
                            for row_idx, row in enumerate(final_exp_values):
                                for col_idx, val in enumerate(row[0]):
                                    if abs(val) > 0.01:
                                        if row_idx <= 31 and col_idx <= 31:
                                            print(
                                                f"time_3: {t2 + 3*t_bin} , {ket_bra_matrix[row_idx,col_idx]} = {val}"
                                            )
                                            print(row_idx, col_idx)

                        else:
                            final_exp_values = exp_eval_floating_start_finish(
                                H_list,
                                H_sim_time_list,
                                rho_evolve_2,
                                2 * t_bin + t2,
                                t2_label + t_bin,
                                self.c_op_list,
                                1,
                                self.aY.dag(),
                                self.exp_final_a,
                            )

                            # TODO bit of a hack to make the lengths of the arrays match
                            if len(final_exp_values) < len(t_correlator_eval_early):
                                final_exp_values = np.insert(final_exp_values, 0, 0)

                            exp_values_eell[_ind][:] = final_exp_values

            exp_values = exp_values_eell

        else:
            raise ValueError(
                "Invalid matrix element specified. Choose from 'EEEE', 'LLLL', 'ELEL', 'LELE', 'ELLE', 'EELL'."
            )

        # Split exp_values into real and imaginary parts
        real_values = np.real(exp_values)
        imag_values = np.imag(exp_values)

        # Compute the integral over the first axis, then the second axis for real part
        real_integral_t1 = trapezoid(
            real_values, x=t1_label, axis=1
        )  # Integrate real over t1 (axis=1)
        real_final_integral = trapezoid(
            real_integral_t1, x=t2_label, axis=0
        )  # Then integrate real over t2 (axis=0)

        # Compute the integral over the first axis, then the second axis for imaginary part
        imag_integral_t1 = trapezoid(
            imag_values, x=t1_label, axis=1
        )  # Integrate imaginary over t1 (axis=1)
        imag_final_integral = trapezoid(
            imag_integral_t1, x=t2_label, axis=0
        )  # Then integrate imaginary over t2 (axis=0)

        print(f"2D Trapezoidal Integral Value (Real part): {real_final_integral}")
        print(f"2D Trapezoidal Integral Value (Imaginary part): {imag_final_integral}")

        if _plot:
            plot_n2_density_matrix_correlation(
                t1_label,
                t2_label,
                real_values,
                imag_values,
                _element,
                self.save_dir,
                self.n_start,
                self.length_stirap,
                self.bfield_split,
                self.two_photon_det,
            )

        return {
            "n_start": self.n_start,
            "bfield_split": self.bfield_split,
            "length_stirap": self.length_stirap,
            "two_photon_det": self.two_photon_det,
            "omega_stirap": self.omega_stirap,
            "omega_rot_stirap": self.omega_rot,
            "element": _element,
            "spontaneous_emission": self.spont_emission,
            "chirped_pulse": self.chirped_pulses,
            "vstirap_pulse_shape": self.vstirap_pulse_shape,
            "vst_rise_time": self.vst_rise_time,
            "vst_fall_time": self.vst_fall_time,
            "real_integral_value": real_final_integral,
            "imag_integral_value": imag_final_integral,
            "n_time_steps": _n_correlator_eval,
        }

    def gen_n2_density_matrix(
        self, _n_correlator_eval, _plot=False, _parall=True, only_diag=False
    ):
        # Define the elements you're interested in
        if only_diag:
            elements = ["EEEE", "LLLL", "ELLE", "LEEL"]
        else:
            elements = ["EEEE", "LLLL", "ELLE", "LEEL", "EELL", "ELEL"]

        # Initialize the dictionary to store results
        density_matrix_data = {}

        # Iterate over the elements and generate the density matrix for each
        for _elem in elements:
            result = self.gen_n2_density_matrix_element(
                _n_correlator_eval, _elem, _plot=_plot, _parall=_parall, _debug=False
            )

            # Store the relevant fields from the result in the dictionary
            density_matrix_data[_elem] = {
                "n_start": self.n_start,
                "bfield_split": self.bfield_split,
                "length_stirap": self.length_stirap,
                "two_photon_det": self.two_photon_det,
                "omega_stirap": self.omega_stirap,
                "omega_rot_stirap": self.omega_rot,
                "spontaneous_emission": self.spont_emission,
                "chirped_pulse": self.chirped_pulses,
                "vstirap_pulse_shape": self.vstirap_pulse_shape,
                "vst_rise_time": self.vst_rise_time,
                "vst_fall_time": self.vst_fall_time,
                "real_integral_value": result["real_integral_value"],
                "imag_integral_value": result["imag_integral_value"],
                "n_time_steps": result["n_time_steps"],
            }

            if _elem == "ELEL":
                density_matrix_data["LELE"] = {
                    "n_start": self.n_start,
                    "bfield_split": self.bfield_split,
                    "length_stirap": self.length_stirap,
                    "two_photon_det": self.two_photon_det,
                    "omega_stirap": self.omega_stirap,
                    "omega_rot_stirap": self.omega_rot,
                    "spontaneous_emission": self.spont_emission,
                    "chirped_pulse": self.chirped_pulses,
                    "vstirap_pulse_shape": self.vstirap_pulse_shape,
                    "vst_rise_time": self.vst_rise_time,
                    "vst_fall_time": self.vst_fall_time,
                    "real_integral_value": result["real_integral_value"],
                    "imag_integral_value": np.conjugate(result["imag_integral_value"]),
                    "n_time_steps": result["n_time_steps"],
                }

            elif _elem == "EELL":
                density_matrix_data["LLEE"] = {
                    "n_start": self.n_start,
                    "bfield_split": self.bfield_split,
                    "length_stirap": self.length_stirap,
                    "two_photon_det": self.two_photon_det,
                    "omega_stirap": self.omega_stirap,
                    "omega_rot_stirap": self.omega_rot,
                    "spontaneous_emission": self.spont_emission,
                    "chirped_pulse": self.chirped_pulses,
                    "vstirap_pulse_shape": self.vstirap_pulse_shape,
                    "vst_rise_time": self.vst_rise_time,
                    "vst_fall_time": self.vst_fall_time,
                    "real_integral_value": result["real_integral_value"],
                    "imag_integral_value": np.conjugate(result["imag_integral_value"]),
                    "n_time_steps": result["n_time_steps"],
                }

        return density_matrix_data

    # Define functions for parallel execution for n=3

    ##################### DIAGONAL ELEMENTS ############################
    def compute_rho_diag_eeeeee(
        self,
        ind,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_early,
        t_correlator_eval_early,
        t_bin,
    ):

        final_exp_values = np.zeros(
            (len(t_correlator_eval_early), len(t_correlator_eval_early)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_early):
            # Step 1: Evolve the initial density matrix
            rho_evolve_1 = rho_evo_floating_start_finish(
                H_list[:8],
                H_sim_time_list[:8],
                rho_start_list_early[ind],
                t1,
                2 * t_bin + t2,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )
            # Step 2: Evaluate the final expectation values
            rho_evolve_2 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_1,
                2 * t_bin + t2,
                4 * t_bin,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )

            # Step 3: Evaluate the final expectation values
            final_exp_values[j][:] = exp_eval_fixed_start(
                H_list[8:],
                H_sim_time_list[8:],
                rho_evolve_2,
                t_correlator_eval_early,
                self.c_op_list,
                1,
                1,
                self.anY,
            )

        return ind, final_exp_values

    def compute_rho_diag_llllll(
        self,
        ind,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_late,
        t_correlator_eval_late,
        t_bin,
    ):
        final_exp_values = np.zeros(
            (len(t_correlator_eval_late), len(t_correlator_eval_late)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_late):
            # Step 1: Evolve the initial density matrix
            rho_evolve_1 = rho_evo_floating_start_finish(
                H_list[:8],
                H_sim_time_list[:8],
                rho_start_list_late[ind],
                t1,
                2 * t_bin + t2,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )
            # Step 2: Evaluate the final expectation values
            rho_evolve_2 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_1,
                2 * t_bin + t2,
                4 * t_bin,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )

            # Step 3: Evaluate the final expectation values
            final_exp_values[j][:] = exp_eval_fixed_start(
                H_list[8:],
                H_sim_time_list[8:],
                rho_evolve_2,
                t_correlator_eval_late,
                self.c_op_list,
                1,
                1,
                self.anY,
            )

        return ind, final_exp_values

    def compute_rho_eellee(
        self,
        ind,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_early,
        t_correlator_eval_late,
        t_bin,
    ):
        final_exp_values = np.zeros(
            (len(t_correlator_eval_late), len(t_correlator_eval_late)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_late):
            # Step 1: Evolve the initial density matrix
            # second time eval is actually early so subtract a time bin
            rho_evolve_1 = rho_evo_floating_start_finish(
                H_list[:8],
                H_sim_time_list[:8],
                rho_start_list_early[ind],
                t1,
                2 * t_bin + t2 - t_bin,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )
            # Step 2: Evaluate the final expectation values
            rho_evolve_2 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_1,
                2 * t_bin + t2 - t_bin,
                4 * t_bin,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )

            # Step 3: Evaluate the final expectation values
            final_exp_values[j][:] = exp_eval_fixed_start(
                H_list[8:],
                H_sim_time_list[8:],
                rho_evolve_2,
                t_correlator_eval_late,
                self.c_op_list,
                1,
                1,
                self.anY,
            )

        return ind, final_exp_values

    def compute_rho_eleele(
        self,
        ind,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_early,
        t_correlator_eval_early,
        t_bin,
    ):
        final_exp_values = np.zeros(
            (len(t_correlator_eval_early), len(t_correlator_eval_early)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_early):
            # Step 1: Evolve the initial density matrix
            # we are evolving late for the second photon so need to add a bin
            rho_evolve_1 = rho_evo_floating_start_finish(
                H_list[:8],
                H_sim_time_list[:8],
                rho_start_list_early[ind],
                t1,
                3 * t_bin + t2,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )
            # Step 2: Evaluate the final expectation values
            rho_evolve_2 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_1,
                3 * t_bin + t2,
                4 * t_bin,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )

            # Step 3: Evaluate the final expectation values
            final_exp_values[j][:] = exp_eval_fixed_start(
                H_list[8:],
                H_sim_time_list[8:],
                rho_evolve_2,
                t_correlator_eval_early,
                self.c_op_list,
                1,
                1,
                self.anY,
            )

        print(f"ind: {i}, final_exp_values: {final_exp_values}")
        return ind, final_exp_values

    def compute_rho_elllle(
        self,
        ind,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_early,
        t_correlator_eval_late,
        t_bin,
    ):
        final_exp_values = np.zeros(
            (len(t_correlator_eval_late), len(t_correlator_eval_late)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_late):
            # Step 1: Evolve the initial density matrix
            rho_evolve_1 = rho_evo_floating_start_finish(
                H_list[:8],
                H_sim_time_list[:8],
                rho_start_list_early[ind],
                t1,
                2 * t_bin + t2,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )
            # Step 2: Evaluate the final expectation values
            rho_evolve_2 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_1,
                2 * t_bin + t2,
                4 * t_bin,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )

            # Step 3: Evaluate the final expectation values
            final_exp_values[j][:] = exp_eval_fixed_start(
                H_list[8:],
                H_sim_time_list[8:],
                rho_evolve_2,
                t_correlator_eval_late,
                self.c_op_list,
                1,
                1,
                self.anY,
            )

        return ind, final_exp_values

    def compute_rho_lellel(
        self,
        ind,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_late,
        t_correlator_eval_late,
        t_bin,
    ):
        final_exp_values = np.zeros(
            (len(t_correlator_eval_late), len(t_correlator_eval_late)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_late):
            # Step 1: Evolve the initial density matrix
            rho_evolve_1 = rho_evo_floating_start_finish(
                H_list[:8],
                H_sim_time_list[:8],
                rho_start_list_late[ind],
                t1,
                2 * t_bin + t2 - t_bin,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )
            # Step 2: Evaluate the final expectation values
            rho_evolve_2 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_1,
                2 * t_bin + t2 - t_bin,
                4 * t_bin,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )

            # Step 3: Evaluate the final expectation values
            final_exp_values[j][:] = exp_eval_fixed_start(
                H_list[8:],
                H_sim_time_list[8:],
                rho_evolve_2,
                t_correlator_eval_late,
                self.c_op_list,
                1,
                1,
                self.anY,
            )

        return ind, final_exp_values

    def compute_rho_lleell(
        self,
        ind,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_late,
        t_correlator_eval_early,
        t_bin,
    ):
        final_exp_values = np.zeros(
            (len(t_correlator_eval_early), len(t_correlator_eval_early)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_early):
            # Step 1: Evolve the initial density matrix
            # we are evaluting in the late bin for the second photon so need to add a bin here
            rho_evolve_1 = rho_evo_floating_start_finish(
                H_list[:8],
                H_sim_time_list[:8],
                rho_start_list_late[ind],
                t1,
                3 * t_bin + t2,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )
            # Step 2: Evaluate the final expectation values
            rho_evolve_2 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_1,
                3 * t_bin + t2,
                4 * t_bin,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )

            # Step 3: Evaluate the final expectation values
            final_exp_values[j][:] = exp_eval_fixed_start(
                H_list[8:],
                H_sim_time_list[8:],
                rho_evolve_2,
                t_correlator_eval_early,
                self.c_op_list,
                1,
                1,
                self.anY,
            )

        return ind, final_exp_values

    def compute_rho_leeeel(
        self,
        ind,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_late,
        t_correlator_eval_early,
        t_bin,
    ):
        final_exp_values = np.zeros(
            (len(t_correlator_eval_early), len(t_correlator_eval_early)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_early):
            # Step 1: Evolve the initial density matrix
            rho_evolve_1 = rho_evo_floating_start_finish(
                H_list[:8],
                H_sim_time_list[:8],
                rho_start_list_late[ind],
                t1,
                2 * t_bin + t2,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )
            # Step 2: Evaluate the final expectation values
            rho_evolve_2 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_1,
                2 * t_bin + t2,
                4 * t_bin,
                self.c_op_list,
                self.aY,
                self.aY.dag(),
            )

            # Step 3: Evaluate the final expectation values
            final_exp_values[j][:] = exp_eval_fixed_start(
                H_list[8:],
                H_sim_time_list[8:],
                rho_evolve_2,
                t_correlator_eval_early,
                self.c_op_list,
                1,
                1,
                self.anY,
            )

        return ind, final_exp_values

    ##################### OFF-DIAGONAL ELEMENTS ############################

    def compute_rho_eeelll(
        self,
        _index,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_early,
        t_correlator_eval_early,
        t_bin,
    ):
        # Evolve the initial density matrix
        rho_evolve_1 = rho_evo_floating_start_finish(
            H_list[:5],
            H_sim_time_list[:5],
            rho_start_list_early[_index],
            t1,
            t1 + t_bin,
            self.c_op_list,
            1,
            self.aY.dag(),
        )

        final_exp_values = np.zeros(
            (len(t_correlator_eval_early), len(t_correlator_eval_early)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_early):
            rho_evolve_2 = rho_evo_floating_start_finish(
                H_list[:7],
                H_sim_time_list[:7],
                rho_evolve_1,
                t1 + t_bin,
                2 * t_bin + t2,
                self.c_op_list,
                self.aY,
                1,
            )

            rho_evolve_3 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_2,
                2 * t_bin + t2,
                3 * t_bin + t2,
                self.c_op_list,
                1,
                self.aY.dag(),
                debug=False,
            )

            for k, t3 in enumerate(t_correlator_eval_early):
                rho_evolve_4 = rho_evo_floating_start_finish(
                    H_list,
                    H_sim_time_list,
                    rho_evolve_3,
                    3 * t_bin + t2,
                    4 * t_bin + t3,
                    self.c_op_list,
                    self.aY,
                    1,
                    debug=False,
                )

                final_rho = rho_evo_floating_start_finish(
                    H_list,
                    H_sim_time_list,
                    rho_evolve_4,
                    4 * t_bin + t3,
                    5 * t_bin + t3,
                    self.c_op_list,
                    1,
                    self.aY.dag(),
                    debug=False,
                )

                # here we are evaluating an annihilation operator at the final point
                final_exp_values[j][k] = final_rho[29, 12]

        return _index, final_exp_values

    def compute_rho_elelel(
        self,
        _index,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_early,
        t_correlator_eval_early,
        t_bin,
    ):
        # Evolve the initial density matrix
        rho_evolve_1 = rho_evo_floating_start_finish(
            H_list[:5],
            H_sim_time_list[:5],
            rho_start_list_early[_index],
            t1,
            t1 + t_bin,
            self.c_op_list,
            1,
            self.aY.dag(),
        )

        final_exp_values = np.zeros(
            (len(t_correlator_eval_early), len(t_correlator_eval_early)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_early):
            rho_evolve_2 = rho_evo_floating_start_finish(
                H_list[:7],
                H_sim_time_list[:7],
                rho_evolve_1,
                t1 + t_bin,
                2 * t_bin + t2,
                self.c_op_list,
                self.aY,
                1,
            )

            rho_evolve_3 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_2,
                2 * t_bin + t2,
                3 * t_bin + t2,
                self.c_op_list,
                self.aY,
                1,
                debug=False,
            )

            for k, t3 in enumerate(t_correlator_eval_early):
                rho_evolve_4 = rho_evo_floating_start_finish(
                    H_list,
                    H_sim_time_list,
                    rho_evolve_3,
                    3 * t_bin + t2,
                    4 * t_bin + t3,
                    self.c_op_list,
                    1,
                    self.aY.dag(),
                    debug=False,
                )

                final_rho = rho_evo_floating_start_finish(
                    H_list,
                    H_sim_time_list,
                    rho_evolve_4,
                    4 * t_bin + t3,
                    5 * t_bin + t3,
                    self.c_op_list,
                    1,
                    self.aY.dag(),
                    debug=False,
                )

                # here we are evaluating an annihilation operator at the final point
                final_exp_values[j][k] = final_rho[29, 12]

        return _index, final_exp_values

    def compute_rho_eelell(
        self,
        _index,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_early,
        t_correlator_eval_early,
        t_bin,
    ):

        # Evolve the initial density matrix
        rho_evolve_1 = rho_evo_floating_start_finish(
            H_list[:5],
            H_sim_time_list[:5],
            rho_start_list_early[_index],
            t1,
            t1 + t_bin,
            self.c_op_list,
            1,
            self.aY.dag(),
        )

        final_exp_values = np.zeros(
            (len(t_correlator_eval_early), len(t_correlator_eval_early)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_early):
            rho_evolve_2 = rho_evo_floating_start_finish(
                H_list[:7],
                H_sim_time_list[:7],
                rho_evolve_1,
                t1 + t_bin,
                2 * t_bin + t2,
                self.c_op_list,
                self.aY,
                1,
            )

            rho_evolve_3 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_2,
                2 * t_bin + t2,
                3 * t_bin + t2,
                self.c_op_list,
                1,
                self.aY.dag(),
                debug=False,
            )

            for k, t3 in enumerate(t_correlator_eval_early):
                rho_evolve_4 = rho_evo_floating_start_finish(
                    H_list,
                    H_sim_time_list,
                    rho_evolve_3,
                    3 * t_bin + t2,
                    4 * t_bin + t3,
                    self.c_op_list,
                    self.aY,
                    1,
                    debug=False,
                )

                final_rho = rho_evo_floating_start_finish(
                    H_list,
                    H_sim_time_list,
                    rho_evolve_4,
                    4 * t_bin + t3,
                    5 * t_bin + t3,
                    self.c_op_list,
                    self.aY,
                    1,
                    debug=False,
                )

                # here we are evaluating an creation operator at the final point
                final_exp_values[j][k] = final_rho[12, 29]

        return _index, final_exp_values

    def compute_rho_elleel(
        self,
        _index,
        t1,
        H_list,
        H_sim_time_list,
        rho_start_list_early,
        t_correlator_eval_early,
        t_bin,
    ):

        # Evolve the initial density matrix
        rho_evolve_1 = rho_evo_floating_start_finish(
            H_list[:5],
            H_sim_time_list[:5],
            rho_start_list_early[_index],
            t1,
            t1 + t_bin,
            self.c_op_list,
            1,
            self.aY.dag(),
        )

        final_exp_values = np.zeros(
            (len(t_correlator_eval_early), len(t_correlator_eval_early)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_early):
            rho_evolve_2 = rho_evo_floating_start_finish(
                H_list[:7],
                H_sim_time_list[:7],
                rho_evolve_1,
                t1 + t_bin,
                2 * t_bin + t2,
                self.c_op_list,
                self.aY,
                1,
            )

            rho_evolve_3 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_2,
                2 * t_bin + t2,
                3 * t_bin + t2,
                self.c_op_list,
                self.aY,
                1,
                debug=False,
            )

            for k, t3 in enumerate(t_correlator_eval_early):
                rho_evolve_4 = rho_evo_floating_start_finish(
                    H_list,
                    H_sim_time_list,
                    rho_evolve_3,
                    3 * t_bin + t2,
                    4 * t_bin + t3,
                    self.c_op_list,
                    1,
                    self.aY.dag(),
                    debug=False,
                )

                final_rho = rho_evo_floating_start_finish(
                    H_list,
                    H_sim_time_list,
                    rho_evolve_4,
                    4 * t_bin + t3,
                    5 * t_bin + t3,
                    self.c_op_list,
                    self.aY,
                    1,
                    debug=False,
                )

                # here we are evaluating a creation operator at the final point
                final_exp_values[j][k] = final_rho[12][0][29]

        return _index, final_exp_values

    def gen_n3_density_matrix_element(
        self, _n_correlator_eval, _element, _parall=False, _debug=False
    ):
        """generate two photon density matrix elements for the given _element
        _n_correlator_eval: number of points to evaluate the correlator over
        _element: the element of the two photon density matrix to calculate IN OPERATOR ORDERING (e.g. EEEEEE, LLLLLL, EEELLL)
        _plot: whether to plot the results
        _parall: whether to use parallel processing
        _debug: whether to print debug information
        """
        # Define initial state
        if self.n_start == 2:
            psi0 = (
                1
                / np.sqrt(2)
                * (
                    self.rb_atom_sim.kb_class.get_ket("g2PP", 0, 0)
                    - self.rb_atom_sim.kb_class.get_ket("g1M", 0, 0)
                )
            )
            # List of Hamiltonians and times
            H_list = [
                self.H_list[0],
                self.H_list[1],
                self.H_list[2],
                self.H_list[3],
                self.H_list[0],
                self.H_list[1],
                self.H_list[2],
                self.H_list[3],
                self.H_list[0],
                self.H_list[1],
                self.H_list[2],
                self.H_list[3],
                self.H_list[0],
            ]
            H_sim_time_list = (
                self.H_sim_time_list
                + self.H_sim_time_list
                + self.H_sim_time_list
                + [self.H_sim_time_list[0]]
            )
        elif self.n_start == 1:
            psi0 = mesolve(
                self.H_list[0],
                self.rb_atom_sim.kb_class.get_ket("g2", 0, 0),
                self.H_sim_time_list[0],
                self.c_op_list,
            ).states[-1]
            H_list = self.H_list[1:] + self.H_list[5:] + self.H_list[5]
            H_sim_time_list = (
                self.H_sim_time_list[1:]
                + self.H_sim_time_list[5:]
                + [self.H_sim_time_list[5]]
            )
        else:
            raise ValueError("Invalid value for n_start. Must be 1 or 2.")

        # Time bins and correlator evaluation times
        # sample more finely across the photon duration and add final point corresponding to the end of the time bin
        t_correlator_eval_early, t_bin = generate_time_correlator_eval(
            _n_correlator_eval, t_vst=self.length_stirap, t_rot=0.25
        )
        t_correlator_eval_late, t_bin = generate_time_correlator_eval(
            _n_correlator_eval, t_vst=self.length_stirap, t_rot=0.25
        )
        t_correlator_eval_late = np.array([x + t_bin for x in t_correlator_eval_late])

        # Calculate density matrices for the evolution
        rho_start_list_early = rho_evo_fixed_start(
            H_list, H_sim_time_list, psi0, t_correlator_eval_early, self.c_op_list, 1, 1
        )
        rho_start_list_early.insert(0, psi0 * psi0.dag())

        # make sure the hamiltonian list is sliced correctly so the evolution is done with the correct hamiltonian elements
        rho_start_list_late = rho_evo_fixed_start(
            H_list, H_sim_time_list, psi0, t_correlator_eval_late, self.c_op_list, 1, 1
        )
        rho_start_list_late.insert(0, rho_start_list_early[-1])

        ##################### DIAGONAL ELEMENTS ############################
        if _element == "EEEEEE":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_early + 2 * t_bin
            t3_label = t_correlator_eval_early + 4 * t_bin

            exp_values = np.zeros(
                (
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_early),
                ),
                dtype=np.complex128,
            )

            if _parall:
                # Parallel execution using ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_diag_eeeeee,
                        range(len(t_correlator_eval_early)),  # For the indices
                        t_correlator_eval_early,  # For each t1 in parallel
                        [H_list] * len(t_correlator_eval_early),
                        [H_sim_time_list] * len(t_correlator_eval_early),
                        [rho_start_list_early] * len(t_correlator_eval_early),
                        [t_correlator_eval_early] * len(t_correlator_eval_early),
                        [t_bin] * len(t_correlator_eval_early),
                    )

                    # Collect the results into the exp_values matrix
                    for _ind, final_exp_values in results:
                        exp_values[_ind][:][:] = final_exp_values

            else:
                # Sequential execution
                for _ind, t1 in enumerate(t_correlator_eval_early):
                    _, final_exp_values = self.compute_rho_diag_eeeeee(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_early[_ind],
                        t_correlator_eval_early,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        elif _element == "LLLLLL":
            t1_label = t_correlator_eval_late
            t2_label = t_correlator_eval_late + 2 * t_bin
            t3_label = t_correlator_eval_late + 4 * t_bin

            exp_values = np.zeros(
                (
                    len(t_correlator_eval_late),
                    len(t_correlator_eval_late),
                    len(t_correlator_eval_late),
                ),
                dtype=np.complex128,
            )

            if _parall:
                # Parallel execution using ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_diag_llllll,
                        range(len(t_correlator_eval_late)),  # For the indices
                        t_correlator_eval_late,  # For each t1 in parallel
                        [H_list] * len(t_correlator_eval_late),
                        [H_sim_time_list] * len(t_correlator_eval_late),
                        [rho_start_list_late] * len(t_correlator_eval_late),
                        [t_correlator_eval_late] * len(t_correlator_eval_late),
                        [t_bin] * len(t_correlator_eval_late),
                    )

                    # Collect the results into the exp_values matrix
                    for _ind, final_exp_values in results:
                        exp_values[_ind][:][:] = final_exp_values

            else:
                # Sequential execution
                for _ind, t1 in enumerate(t_correlator_eval_late):
                    _, final_exp_values = self.compute_rho_diag_llllll(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_early[_ind],
                        t_correlator_eval_late,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        elif _element == "EELLEE":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_early + 2 * t_bin
            t3_label = t_correlator_eval_late + 4 * t_bin

            exp_values = np.zeros(
                (
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_late),
                ),
                dtype=np.complex128,
            )

            if _parall:
                # Parallel execution using ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_eellee,
                        range(len(t_correlator_eval_early)),  # For the indices
                        t_correlator_eval_early,  # For each t1 in parallel
                        [H_list] * len(t_correlator_eval_early),
                        [H_sim_time_list] * len(t_correlator_eval_early),
                        [rho_start_list_early] * len(t_correlator_eval_early),
                        [t_correlator_eval_late] * len(t_correlator_eval_early),
                        [t_bin] * len(t_correlator_eval_early),
                    )

                    # Collect the results into the exp_values matrix
                    for _ind, final_exp_values in results:
                        exp_values[_ind][:][:] = final_exp_values

            else:
                # Sequential execution
                for _ind, t1 in enumerate(t_correlator_eval_early):
                    _, final_exp_values = self.compute_rho_eellee(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_early[_ind],
                        t_correlator_eval_early,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        elif _element == "LELLEL":
            t1_label = t_correlator_eval_late
            t2_label = t_correlator_eval_early + 2 * t_bin
            t3_label = t_correlator_eval_late + 4 * t_bin

            exp_values = np.zeros(
                (
                    len(t_correlator_eval_late),
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_late),
                ),
                dtype=np.complex128,
            )

            if _parall:
                # Parallel execution using ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_lellel,
                        range(len(t_correlator_eval_late)),  # For the indices
                        t_correlator_eval_late,  # For each t1 in parallel
                        [H_list] * len(t_correlator_eval_late),
                        [H_sim_time_list] * len(t_correlator_eval_late),
                        [rho_start_list_late] * len(t_correlator_eval_late),
                        [t_correlator_eval_late] * len(t_correlator_eval_late),
                        [t_bin] * len(t_correlator_eval_late),
                    )

                    # Collect the results into the exp_values matrix
                    for _ind, final_exp_values in results:
                        exp_values[_ind][:][:] = final_exp_values

            else:
                # Sequential execution
                for _ind, t1 in enumerate(t_correlator_eval_late):
                    _, final_exp_values = self.compute_rho_lellel(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_late[_ind],
                        t_correlator_eval_late,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        elif _element == "ELLLLE":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_late + 2 * t_bin
            t3_label = t_correlator_eval_late + 4 * t_bin

            exp_values = np.zeros(
                (
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_late),
                    len(t_correlator_eval_early),
                ),
                dtype=np.complex128,
            )

            if _parall:
                # Parallel execution using ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_elllle,
                        range(len(t_correlator_eval_early)),  # For the indices
                        t_correlator_eval_early,  # For each t1 in parallel
                        [H_list] * len(t_correlator_eval_early),
                        [H_sim_time_list] * len(t_correlator_eval_early),
                        [rho_start_list_early] * len(t_correlator_eval_early),
                        [t_correlator_eval_late] * len(t_correlator_eval_early),
                        [t_bin] * len(t_correlator_eval_early),
                    )

                    # Collect the results into the exp_values matrix
                    for _ind, final_exp_values in results:
                        exp_values[_ind][:][:] = final_exp_values

            else:
                # Sequential execution
                for _ind, t1 in enumerate(t_correlator_eval_late):
                    _, final_exp_values = self.compute_rho_elllle(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_early[_ind],
                        t_correlator_eval_late,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        elif _element == "LLEELL":
            t1_label = t_correlator_eval_late
            t2_label = t_correlator_eval_early + 2 * t_bin
            t3_label = t_correlator_eval_early + 4 * t_bin

            exp_values = np.zeros(
                (
                    len(t_correlator_eval_late),
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_early),
                ),
                dtype=np.complex128,
            )

            if _parall:
                # Parallel execution using ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_lleell,
                        range(len(t_correlator_eval_early)),  # For the indices
                        t_correlator_eval_early,  # For each t1 in parallel
                        [H_list] * len(t_correlator_eval_early),
                        [H_sim_time_list] * len(t_correlator_eval_early),
                        [rho_start_list_late] * len(t_correlator_eval_early),
                        [t_correlator_eval_early] * len(t_correlator_eval_early),
                        [t_bin] * len(t_correlator_eval_early),
                    )

                    # Collect the results into the exp_values matrix
                    for _ind, final_exp_values in results:
                        exp_values[_ind][:][:] = final_exp_values

            else:
                # Sequential execution
                for _ind, t1 in enumerate(t_correlator_eval_late):
                    _, final_exp_values = self.compute_rho_elllle(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_early[_ind],
                        t_correlator_eval_late,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        elif _element == "ELEELE":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_late + 2 * t_bin
            t3_label = t_correlator_eval_early + 4 * t_bin

            exp_values = np.zeros(
                (
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_late),
                    len(t_correlator_eval_early),
                ),
                dtype=np.complex128,
            )
            print(f"exp_values {exp_values}")

            if _parall:
                # Parallel execution using ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_eleele,
                        range(len(t_correlator_eval_early)),  # For the indices
                        t_correlator_eval_early,  # For each t1 in parallel
                        [H_list] * len(t_correlator_eval_early),
                        [H_sim_time_list] * len(t_correlator_eval_early),
                        [rho_start_list_early] * len(t_correlator_eval_early),
                        [t_correlator_eval_early] * len(t_correlator_eval_early),
                        [t_bin] * len(t_correlator_eval_early),
                    )

                    # Collect the results into the exp_values matrix
                    for _ind, final_exp_values in results:
                        exp_values[_ind][:][:] = final_exp_values

            else:
                # Sequential execution
                for _ind, t1 in enumerate(t_correlator_eval_early):
                    _, final_exp_values = self.compute_rho_eleele(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_early[_ind],
                        t_correlator_eval_early,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        elif _element == "LEEEEL":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_early + 2 * t_bin
            t3_label = t_correlator_eval_early + 4 * t_bin

            exp_values = np.zeros(
                (
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_early),
                ),
                dtype=np.complex128,
            )

            if _parall:
                # Parallel execution using ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_leeeel,
                        range(len(t_correlator_eval_late)),  # For the indices
                        t_correlator_eval_late,  # For each t1 in parallel
                        [H_list] * len(t_correlator_eval_late),
                        [H_sim_time_list] * len(t_correlator_eval_late),
                        [rho_start_list_late] * len(t_correlator_eval_late),
                        [t_correlator_eval_early] * len(t_correlator_eval_late),
                        [t_bin] * len(t_correlator_eval_late),
                    )

                    # Collect the results into the exp_values matrix
                    for _ind, final_exp_values in results:
                        exp_values[_ind][:][:] = final_exp_values

            else:
                # Sequential execution
                for _ind, t1 in enumerate(t_correlator_eval_early):
                    _, final_exp_values = self.compute_rho_leeeel(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_late[_ind],
                        t_correlator_eval_early,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        ##################### OFF-DIAGONAL ELEMENTS ############################
        elif _element == "EEELLL":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_early + 2 * t_bin
            t3_label = t_correlator_eval_early + 4 * t_bin

            exp_values = np.zeros(
                (
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_early),
                ),
                dtype=np.complex128,
            )

            if _parall:
                # Parallel execution using ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_eeelll,
                        range(len(t_correlator_eval_early)),  # For the indices
                        t_correlator_eval_early,  # For each t1 in parallel
                        [H_list] * len(t_correlator_eval_early),
                        [H_sim_time_list] * len(t_correlator_eval_early),
                        [rho_start_list_early] * len(t_correlator_eval_early),
                        [t_correlator_eval_early] * len(t_correlator_eval_early),
                        [t_bin] * len(t_correlator_eval_early),
                    )

                    # Collect the results into the exp_values matrix
                    for _ind, final_exp_values in results:
                        exp_values[_ind][:][:] = final_exp_values

            else:
                # Sequential execution
                for _ind, t1 in enumerate(t_correlator_eval_early):
                    _, final_exp_values = self.compute_rho_eeelll(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_early[_ind],
                        t_correlator_eval_early,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        elif _element == "ELELEL":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_early + 2 * t_bin
            t3_label = t_correlator_eval_early + 4 * t_bin

            exp_values = np.zeros(
                (
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_early),
                ),
                dtype=np.complex128,
            )

            if _parall:
                # Parallel execution using ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_elelel,
                        range(len(t_correlator_eval_early)),  # For the indices
                        t_correlator_eval_early,  # For each t1 in parallel
                        [H_list] * len(t_correlator_eval_early),
                        [H_sim_time_list] * len(t_correlator_eval_early),
                        [rho_start_list_early] * len(t_correlator_eval_early),
                        [t_correlator_eval_early] * len(t_correlator_eval_early),
                        [t_bin] * len(t_correlator_eval_early),
                    )

                    # Collect the results into the exp_values matrix
                    for _ind, final_exp_values in results:
                        exp_values[_ind][:][:] = final_exp_values

            else:
                # Sequential execution
                for _ind, t1 in enumerate(t_correlator_eval_early):
                    _, final_exp_values = self.compute_rho_elelel(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_early[_ind],
                        t_correlator_eval_early,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        elif _element == "EELELL":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_early + 2 * t_bin
            t3_label = t_correlator_eval_early + 4 * t_bin

            exp_values = np.zeros(
                (
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_early),
                ),
                dtype=np.complex128,
            )

            if _parall:
                # Parallel execution using ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_elelel,
                        range(len(t_correlator_eval_early)),  # For the indices
                        t_correlator_eval_early,  # For each t1 in parallel
                        [H_list] * len(t_correlator_eval_early),
                        [H_sim_time_list] * len(t_correlator_eval_early),
                        [rho_start_list_early] * len(t_correlator_eval_early),
                        [t_correlator_eval_early] * len(t_correlator_eval_early),
                        [t_bin] * len(t_correlator_eval_early),
                    )

                    # Collect the results into the exp_values matrix
                    for _ind, final_exp_values in results:
                        exp_values[_ind][:][:] = final_exp_values

            else:
                # Sequential execution
                for _ind, t1 in enumerate(t_correlator_eval_early):
                    _, final_exp_values = self.compute_rho_elelel(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_early[_ind],
                        t_correlator_eval_early,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        elif _element == "ELLEEL":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_early + 2 * t_bin
            t3_label = t_correlator_eval_early + 4 * t_bin

            exp_values = np.zeros(
                (
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_early),
                    len(t_correlator_eval_early),
                ),
                dtype=np.complex128,
            )

            if _parall:
                # Parallel execution using ProcessPoolExecutor
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_elleel,
                        range(len(t_correlator_eval_early)),  # For the indices
                        t_correlator_eval_early,  # For each t1 in parallel
                        [H_list] * len(t_correlator_eval_early),
                        [H_sim_time_list] * len(t_correlator_eval_early),
                        [rho_start_list_early] * len(t_correlator_eval_early),
                        [t_correlator_eval_early] * len(t_correlator_eval_early),
                        [t_bin] * len(t_correlator_eval_early),
                    )

                    # Collect the results into the exp_values matrix
                    for _ind, final_exp_values in results:
                        exp_values[_ind][:][:] = final_exp_values

            else:
                # Sequential execution
                for _ind, t1 in enumerate(t_correlator_eval_early):
                    _, final_exp_values = self.compute_rho_elleel(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_early[_ind],
                        t_correlator_eval_early,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        else:
            raise ValueError(
                "Invalid matrix element specified. Choose from 'EEEEEE', 'LLLLLL', 'EELLEE', 'LEELLE', 'EEEELL', 'ELELEL', 'EELELL', 'ELLEEL'."
            )

        # Split exp_values into real and imaginary parts
        real_values = np.real(exp_values)
        imag_values = np.imag(exp_values)

        # Compute the integral over the first axis, then the second axis for real part
        real_integral_t1 = trapezoid(real_values, x=t1_label, axis=2)
        real_integral_t2 = trapezoid(real_integral_t1, x=t2_label, axis=1)
        real_final_integral = trapezoid(real_integral_t2, x=t3_label, axis=0)

        # Compute the integral over the first axis, then the second axis for imaginary part
        imag_integral_t1 = trapezoid(imag_values, x=t1_label, axis=2)
        imag_integral_t2 = trapezoid(imag_integral_t1, x=t2_label, axis=1)
        imag_final_integral = trapezoid(imag_integral_t2, x=t3_label, axis=0)

        print(f"3D Trapezoidal Integral Value (Real part): {real_final_integral}")
        print(f"3D Trapezoidal Integral Value (Imaginary part): {imag_final_integral}")

        return {
            "element": _element,
            "n_start": self.n_start,
            "bfield_split": self.bfield_split,
            "length_stirap": self.length_stirap,
            "two_photon_det": self.two_photon_det,
            "omega_stirap": self.omega_stirap,
            "omega_rot_stirap": self.omega_rot,
            "spontaneous_emission": self.spont_emission,
            "chirped_pulse": self.chirped_pulses,
            "vstirap_pulse_shape": self.vstirap_pulse_shape,
            "vst_rise_time": self.vst_rise_time,
            "vst_fall_time": self.vst_fall_time,
            "real_integral_value": real_final_integral,
            "imag_integral_value": imag_final_integral,
            "n_time_steps": _n_correlator_eval,
        }

    def gen_n3_density_matrix(
        self, _n_correlator_eval, _parall=True, only_diag=False, only_off_diag=False
    ):
        # Define the elements you're interested in
        if only_diag:
            elements = [
                "EEEEEE",
                "LLLLLL",
                "EELLEE",
                "ELEELE",
                "ELLLLE",
                "LELLEL",
                "LLEELL",
                "LEEEEL",
            ]
        elif only_off_diag:
            elements = ["EEELLL", "ELELEL", "EELELL", "ELLEEL"]
        elif only_diag and only_off_diag:
            raise ValueError(
                "Cannot choose both only_diag and only_off_diag options. Choose one or neither"
            )
        else:
            elements = [
                "EEEEEE",
                "LLLLLL",
                "EELLEE",
                "ELEELE",
                "ELLLLE",
                "LELLEL",
                "LLEELL",
                "LEEEEL",
                "EEELLL",
                "ELELEL",
                "EELELL",
                "ELLEEL",
            ]

        # Initialize the dictionary to store results
        density_matrix_data = {}

        # Iterate over the elements and generate the density matrix for each
        for _elem in elements:
            result = self.gen_n3_density_matrix_element(
                _n_correlator_eval, _elem, _parall=_parall, _debug=False
            )

            # Store the relevant fields from the result in the dictionary
            density_matrix_data[_elem] = {
                "n_start": self.n_start,
                "bfield_split": self.bfield_split,
                "length_stirap": self.length_stirap,
                "two_photon_det": self.two_photon_det,
                "omega_stirap": self.omega_stirap,
                "omega_rot_stirap": self.omega_rot,
                "spontaneous_emission": self.spont_emission,
                "chirped_pulse": self.chirped_pulses,
                "vstirap_pulse_shape": self.vstirap_pulse_shape,
                "vst_rise_time": self.vst_rise_time,
                "vst_fall_time": self.vst_fall_time,
                "real_integral_value": result["real_integral_value"],
                "imag_integral_value": result["imag_integral_value"],
                "n_time_steps": result["n_time_steps"],
            }

            if _elem == "EEELLL":
                density_matrix_data["LLLEEE"] = {
                    "n_start": self.n_start,
                    "bfield_split": self.bfield_split,
                    "length_stirap": self.length_stirap,
                    "two_photon_det": self.two_photon_det,
                    "omega_stirap": self.omega_stirap,
                    "omega_rot_stirap": self.omega_rot,
                    "spontaneous_emission": self.spont_emission,
                    "chirped_pulse": self.chirped_pulses,
                    "vstirap_pulse_shape": self.vstirap_pulse_shape,
                    "vst_rise_time": self.vst_rise_time,
                    "vst_fall_time": self.vst_fall_time,
                    "real_integral_value": result["real_integral_value"],
                    "imag_integral_value": np.conjugate(result["imag_integral_value"]),
                    "n_time_steps": result["n_time_steps"],
                }

            elif _elem == "ELELEL":
                density_matrix_data["LELELE"] = {
                    "n_start": self.n_start,
                    "bfield_split": self.bfield_split,
                    "length_stirap": self.length_stirap,
                    "two_photon_det": self.two_photon_det,
                    "omega_stirap": self.omega_stirap,
                    "omega_rot_stirap": self.omega_rot,
                    "spontaneous_emission": self.spont_emission,
                    "chirped_pulse": self.chirped_pulses,
                    "vstirap_pulse_shape": self.vstirap_pulse_shape,
                    "vst_rise_time": self.vst_rise_time,
                    "vst_fall_time": self.vst_fall_time,
                    "real_integral_value": result["real_integral_value"],
                    "imag_integral_value": np.conjugate(result["imag_integral_value"]),
                    "n_time_steps": result["n_time_steps"],
                }

            elif _elem == "EELELL":
                density_matrix_data["LLELEE"] = {
                    "n_start": self.n_start,
                    "bfield_split": self.bfield_split,
                    "length_stirap": self.length_stirap,
                    "two_photon_det": self.two_photon_det,
                    "omega_stirap": self.omega_stirap,
                    "omega_rot_stirap": self.omega_rot,
                    "spontaneous_emission": self.spont_emission,
                    "chirped_pulse": self.chirped_pulses,
                    "vstirap_pulse_shape": self.vstirap_pulse_shape,
                    "vst_rise_time": self.vst_rise_time,
                    "vst_fall_time": self.vst_fall_time,
                    "real_integral_value": result["real_integral_value"],
                    "imag_integral_value": np.conjugate(result["imag_integral_value"]),
                    "n_time_steps": result["n_time_steps"],
                }

            elif _elem == "ELLEEL":
                density_matrix_data["LEELLE"] = {
                    "n_start": self.n_start,
                    "bfield_split": self.bfield_split,
                    "length_stirap": self.length_stirap,
                    "two_photon_det": self.two_photon_det,
                    "omega_stirap": self.omega_stirap,
                    "omega_rot_stirap": self.omega_rot,
                    "spontaneous_emission": self.spont_emission,
                    "chirped_pulse": self.chirped_pulses,
                    "vstirap_pulse_shape": self.vstirap_pulse_shape,
                    "vst_rise_time": self.vst_rise_time,
                    "vst_fall_time": self.vst_fall_time,
                    "real_integral_value": result["real_integral_value"],
                    "imag_integral_value": np.conjugate(result["imag_integral_value"]),
                    "n_time_steps": result["n_time_steps"],
                }

        return density_matrix_data
