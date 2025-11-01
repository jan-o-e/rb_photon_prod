import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

import concurrent.futures
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy import interpolate
from scipy.integrate import trapezoid
from qutip import mesolve, tensor, qeye, destroy, expect

from modules.cavity import cav_collapse_ops
from modules.simulation import Simulation
from modules.correlation_functions import (
    exp_eval_fixed_start,
    rho_evo_fixed_start,
    rho_evo_floating_start_finish,
)
from modules.integration_functions import trapz_integral_real_imaginary


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


class TimeBinPhotonCorrelationsCluster:

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
        H_rot_1, t_rot1, args_hams_rot_1 = (
            self.rb_atom_sim.generate_timebin_rotation_hamiltonian(
                first_params, _chirped=self.chirped_pulses, _n_steps=self.n_steps_rot
            )
        )
        H_rot_2, t_rot2, args_hams_rot_2 = (
            self.rb_atom_sim.generate_timebin_rotation_hamiltonian(
                second_params, _chirped=self.chirped_pulses, _n_steps=self.n_steps_rot
            )
        )
        H_rot_3, t_rot3, args_hams_rot_3 = (
            self.rb_atom_sim.generate_timebin_rotation_hamiltonian(
                third_params, _chirped=self.chirped_pulses, _n_steps=self.n_steps_rot
            )
        )
        H_rot_4, t_rot4, args_hams_rot_4 = (
            self.rb_atom_sim.generate_timebin_rotation_hamiltonian(
                fourth_params, _chirped=self.chirped_pulses, _n_steps=self.n_steps_rot
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

    def generate_time_correlator_eval(
        self,
        _n_off_diag_correlator_eval,
        frac_fine=2 / 3,
        fine_interval_frac=(0.1, 0.6),
        coarse_interval_frac=(0, 0.8),
        t_vst=0.5,
        t_rot=0.25,
    ):
        """
        Generate a list of time correlator evaluation points with specified sampling characteristics.

        Parameters:
        _n_off_diag_correlator_eval (int): Total number of points to generate.
        frac_fine (float): Fraction of points to be sampled finely in the fine_interval.
        fine_interval_frac (tuple): Fractions of t_vst for fine sampling (start, end).
        coarse_interval_frac (tuple): Fractions of t_vst + t_rot for coarse sampling (start, end).
        t_vst (float): Time for vst in microseconds.
        t_rot (float): Time for additional rotations in microseconds.

        Returns:
        numpy.ndarray: Sorted array of evaluation points.

        Raises:
        ValueError: If any of the interval bounds exceed t_vst + t_rot.
        """

        # Compute the actual intervals based on fractions
        fine_interval = (fine_interval_frac[0] * t_vst, fine_interval_frac[1] * t_vst)
        # Don't sample all the way to the end of a time bin (there will be no correlation) and it causes a problem with list indexing
        coarse_interval = (
            coarse_interval_frac[0] * (t_vst + t_rot),
            coarse_interval_frac[1] * (t_vst + t_rot - 0.01 * t_rot),
        )

        # Check if any interval bounds exceed t_vst + t_rot
        if fine_interval[1] > t_vst + t_rot or coarse_interval[1] > t_vst + t_rot:
            raise ValueError("Interval bounds exceed t_vst + t_rot.")

        # Compute the number of points for each segment
        n_fine_points = int(frac_fine * _n_off_diag_correlator_eval)
        n_coarse_points = _n_off_diag_correlator_eval - n_fine_points

        # Generate finely sampled points in the fine_interval
        fine_points = np.linspace(fine_interval[0], fine_interval[1], n_fine_points)

        # Generate coarsely sampled points in the coarse_interval
        coarse_points = np.linspace(
            coarse_interval[0], coarse_interval[1], n_coarse_points
        )

        # Combine the points and ensure there are no duplicates
        all_points = np.unique(np.concatenate((fine_points, coarse_points)))

        # Include a single point at t=t_final, if not already included
        final_time = float(t_vst + t_rot)
        if final_time not in all_points:
            all_points = np.append(all_points, final_time)

        # Sort the final list of points
        t_correlator_eval = np.sort(all_points)
        return t_correlator_eval, final_time

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
            self.c_op_list,
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
                self.c_op_list,
            ).states[-1]
            H_list = [self.H_list[1], self.H_list[2], self.H_list[3], self.H_list[4]]
            H_sim_time_list = self.H_sim_time_list[1:]
        else:
            raise ValueError("Invalid value for n_start. Must be 1 or 2.")

        # Time bins and correlator evaluation times
        # sample more finely across the photon duration and add final point corresponding to the end of the time bin
        t_correlator_eval, t_bin = self.generate_time_correlator_eval(
            _n_off_diag_correlator_eval, t_vst=self.length_stirap, t_rot=0.25
        )

        # Calculate density matrices
        rho_start_list = rho_evo_fixed_start(
            H_list, H_sim_time_list, psi0, t_correlator_eval, self.c_op_list, 1, 1
        )
        rho_start_list.insert(0, psi0)  # Add the initial density matrix

        # self.rb_atom_sim.rb_atom.plotter_atomstate_population(self.rb_atom_sim.ketbras, rho_start_list, t_correlator_eval, True)

        # Diagonal density matrix expectations
        exp_values_diag_zero = mesolve(
            H_list[0], psi0, H_sim_time_list[0], self.c_op_list, [self.anY]
        ).expect[0]
        exp_values_diag_one = mesolve(
            H_list[2],
            rho_start_list[-1],
            H_sim_time_list[2],
            self.c_op_list,
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
            exp_values_off_diag_one[time_point][29][0][12]
            for time_point in range(len(exp_values_off_diag_one))
        ]

        if self.cubic_spline_smoothing:
            # Fit a cubic spline to coherence_01
            spline_coherence_01 = interpolate.CubicSpline(
                t_correlator_eval, coherence_01
            )

            # Optionally, define a finer time grid for better integration accuracy
            fine_t_grid = np.linspace(t_correlator_eval[0], t_correlator_eval[-1], 1000)

            # Evaluate the spline on the fine grid
            fine_coherence_01 = spline_coherence_01(fine_t_grid)

            # Calculate integrals for diagonal and off-diagonal elements
            int_off_diag_01_re, int_off_diag_01_im = trapz_integral_real_imaginary(
                fine_t_grid, fine_coherence_01
            )
            int_off_diag_10_re, int_off_diag_10_im = trapz_integral_real_imaginary(
                fine_t_grid, np.conjugate(fine_coherence_01)
            )
        else:
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
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            # Plot for rho_00
            ax[0, 0].plot(
                H_sim_time_list[0], np.real(exp_values_diag_zero), label="Real"
            )
            ax[0, 0].plot(
                H_sim_time_list[0], np.imag(exp_values_diag_zero), label="Imaginary"
            )
            ax[0, 0].set_title("rho_00")
            ax[0, 0].set_xlabel("Time")
            ax[0, 0].set_ylabel("Correlation")
            ax[0, 0].legend()

            # Plot for rho_11
            ax[0, 1].plot(
                H_sim_time_list[2], np.real(exp_values_diag_one), label="Real"
            )
            ax[0, 1].plot(
                H_sim_time_list[2], np.imag(exp_values_diag_one), label="Imaginary"
            )
            ax[0, 1].set_title("rho_11")
            ax[0, 1].set_xlabel("Time")
            ax[0, 1].set_ylabel("Correlation")
            ax[0, 1].legend()

            # Plot for rho_10
            ax[1, 0].plot(
                t_correlator_eval, np.real(np.conj(coherence_01)), label="Real"
            )
            ax[1, 0].plot(
                t_correlator_eval, np.imag(np.conj(coherence_01)), label="Imaginary"
            )
            ax[1, 0].set_title("rho_10")
            ax[1, 0].set_xlabel("Time")
            ax[1, 0].set_ylabel("Correlation")
            ax[1, 0].legend()

            # Plot for rho_01
            ax[1, 1].plot(t_correlator_eval, np.real(coherence_01), label="Real")
            ax[1, 1].plot(t_correlator_eval, np.imag(coherence_01), label="Imaginary")
            ax[1, 1].set_title("rho_01")
            ax[1, 1].set_xlabel("Time")
            ax[1, 1].set_ylabel("Correlation")
            ax[1, 1].legend()

            # plt.show()
            plt.tight_layout()
            # Save the plot as a single file
            # Get the absolute path of the current file (photon_crrelation.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Move up to the parent directory ('modules' and src)
            modules_dir = os.path.abspath(os.path.join(current_dir, ".."))
            src_dir = os.path.abspath(os.path.join(modules_dir, ".."))

            # Define the path where you want to save
            save_path = os.path.join(src_dir, self.save_dir)

            # Create the directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)

            plot_path = os.path.join(
                save_path,
                f"density_matrix_n1_correlations_n_start_{self.n_start}_vst{self.length_stirap}_b{self.bfield_split}.pdf",
            )
            plt.savefig(plot_path)
            # plt.show()

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
    # MATRIX ELEMENTS SPECIFIED IN OPERATOR ORDERING NOT IN QUBIT ORDERING

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

    def compute_rho_diag_leel(
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

    def compute_rho_diag_elle(
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

    def compute_rho_eeel(
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

            # here we are evaluating an number operator at the final point
            # TODO DEBUG
            final_exp_values[j] = expect(self.anY, rho_evolve_2)

        return _index, final_exp_values

    def compute_rho_elll(
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
                H_list[:],
                H_sim_time_list[:],
                rho_evolve_1,
                t1 + t_bin,
                3 * t_bin + t2,
                self.c_op_list,
                self.aY,
                1,
            )

            # here we are evaluating an number operator at the final point
            # TODO DEBUG
            final_exp_values[j] = expect(self.anY, rho_evolve_2)

        return _index, final_exp_values

    def gen_n2_density_matrix_element(
        self, _n_correlator_eval, _element, _plot=False, _parall=False, _debug=False
    ):
        """generate two photon density matrix elements for the given _element
        _n_correlator_eval: number of points to evaluate the correlator over
        _element: the element of the two photon density matrix to calculate IN OPERATOR ORDERING (e.g. EEEE, LLLL, EELL, LEEL, EEEL, LLLE)
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
        t_correlator_eval_early, t_bin = self.generate_time_correlator_eval(
            _n_correlator_eval, t_vst=self.length_stirap, t_rot=0.25
        )
        t_correlator_eval_late, t_bin = self.generate_time_correlator_eval(
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
                        self.compute_rho_diag_elle,
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
                        self.compute_rho_diag_leel,
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

        elif _element == "ELLL":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_late + 2 * t_bin
            exp_values_elll = np.zeros(
                (len(t_correlator_eval_late), len(t_correlator_eval_early)),
                dtype=np.complex128,
            )

            if _parall:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_elll,
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
                        exp_values_elll[_ind][:] = final_exp_values

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
                            H_list[:],
                            H_sim_time_list[:],
                            rho_evolve_1,
                            t1 + t_bin,
                            3 * t_bin + t2,
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
                                                f"time_2: {3 * t_bin + t2}, {ket_bra_matrix[row_idx,col_idx]} = {val}"
                                            )
                            exp_values_elll[_ind][j] = expect(self.anY, rho_evolve_2)

                        else:
                            exp_values_elll[_ind][j] = expect(self.anY, rho_evolve_2)

            exp_values = exp_values_elll
            print(exp_values.shape)

        elif _element == "EEEL":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_early + 2 * t_bin
            exp_values_eeel = np.zeros(
                (len(t_correlator_eval_early), len(t_correlator_eval_early)),
                dtype=np.complex128,
            )

            if _parall:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    results = executor.map(
                        self.compute_rho_eeel,
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
                        exp_values_eeel[_ind][:] = final_exp_values

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
                            H_list[:],
                            H_sim_time_list[:],
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

                            exp_values_eeel[_ind][j] = expect(self.anY, rho_evolve_2)

                        else:
                            exp_values_eeel[_ind][j] = expect(self.anY, rho_evolve_2)

            exp_values = exp_values_eeel

        else:
            raise ValueError(
                "Invalid matrix element specified. Choose from 'EEEE', 'LLLL', 'ELLL', 'LELE', 'ELLE', 'EEEL'."
            )

        # Split exp_values into real and imaginary parts
        real_values = np.real(exp_values)
        imag_values = np.imag(exp_values)

        if self.cubic_spline_smoothing:

            # Assuming real_values is your input 2D array and t1_label, t2_label are the axes
            oversample_factor = 10  # Increase resolution by this factor

            # Define finer grids for t1 and t2
            t1_fine = np.linspace(
                t1_label[0], t1_label[-1], len(t1_label) * oversample_factor
            )
            t2_fine = np.linspace(
                t2_label[0], t2_label[-1], len(t2_label) * oversample_factor
            )

            # Create a 2D cubic spline interpolator for real_values
            spline_real = interpolate.RectBivariateSpline(
                t1_label, t2_label, real_values
            )
            smoothed_real_values_fine = spline_real(t1_fine, t2_fine)

            # If you have an imaginary part and want to do the same:
            spline_imag = interpolate.RectBivariateSpline(
                t1_label, t2_label, imag_values
            )
            smoothed_imag_values_fine = spline_imag(t1_fine, t2_fine)

            # Compute the integral over the first axis (t1), then the second axis (t2) for the real part
            real_integral_t1 = trapezoid(
                smoothed_real_values_fine, x=t1_fine, axis=1
            )  # Integrate over t1_fine (axis=1)
            real_final_integral = trapezoid(
                real_integral_t1, x=t2_fine, axis=0
            )  # Then integrate over t2_fine (axis=0)

            # Compute the integral over the first axis (t1), then the second axis (t2) for the imaginary part
            imag_integral_t1 = trapezoid(
                smoothed_imag_values_fine, x=t1_fine, axis=1
            )  # Integrate over t1_fine (axis=1)
            imag_final_integral = trapezoid(
                imag_integral_t1, x=t2_fine, axis=0
            )  # Then integrate over t2_fine (axis=0)

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
            # Step 1: Contour Plot with Custom X and Y Labels for Real and Imaginary parts side by side
            X, Y = np.meshgrid(t1_label, t2_label)

            # Create a 1x2 subplot
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))

            # Plot real part
            c1 = ax[0].contourf(X, Y, real_values, cmap="viridis")
            fig.colorbar(c1, ax=ax[0], label="Real Correlation Value")
            ax[0].set_title(
                f"Real Correlation Values for N=2 Matrix Element: {_element}"
            )
            ax[0].set_xlabel("t1")
            ax[0].set_ylabel("t2")
            ax[0].set_xticks(np.linspace(min(t1_label), max(t1_label), num=5))
            ax[0].set_yticks(np.linspace(min(t2_label), max(t2_label), num=5))

            # Plot imaginary part
            c2 = ax[1].contourf(X, Y, imag_values, cmap="plasma")
            fig.colorbar(c2, ax=ax[1], label="Imaginary Correlation Value")
            ax[1].set_title(
                f"Imaginary Correlation Values for N=2 Matrix Element: {_element}"
            )
            ax[1].set_xlabel("t1")
            ax[1].set_ylabel("t2")
            ax[1].set_xticks(np.linspace(min(t1_label), max(t1_label), num=5))
            ax[1].set_yticks(np.linspace(min(t2_label), max(t2_label), num=5))

            # Adjust layout to fit everything nicely
            plt.tight_layout()

            # Save the plot as a single file
            # Get the absolute path of the current file (photon_correlation.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Move up to the parent directory ('modules' and src)
            modules_dir = os.path.abspath(os.path.join(current_dir, ".."))
            src_dir = os.path.abspath(os.path.join(modules_dir, ".."))

            # Define the path where you want to save
            save_path = os.path.join(src_dir, self.save_dir)

            # Create the directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)

            plot_path = os.path.join(
                self.save_dir,
                f"n2_correlations_nstart{self.n_start}_vstlength{self.length_stirap}_b{self.bfield_split}_twophotdet{np.round(self.two_photon_det[0],3)}_{np.round(self.two_photon_det[1],3)}{_element}.pdf",
            )
            plt.savefig(plot_path)

            # Show the plot
            # plt.show()

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
            elements = ["EEEE", "LLLL", "ELLE", "LEEL", "EEEL", "ELLL"]

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

            if _elem == "ELLL":
                density_matrix_data["LLLE"] = {
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

            elif _elem == "EEEL":
                density_matrix_data["LEEE"] = {
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

    def compute_rho_lellll(
        self,
        _index,
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
            rho_start_list_late[_index],
            t1 + t_bin,
            t1 + 2 * t_bin,
            self.c_op_list,
            self.aY,
            self.aY.dag(),
        )

        final_exp_values = np.zeros(
            (len(t_correlator_eval_early), len(t_correlator_eval_early)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_early):

            rho_evolve_3 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_1,
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
                    5 * t_bin + t3,
                    self.c_op_list,
                    self.aY,
                    1,
                    debug=False,
                )

                final_exp_values[j][k] = expect(self.anY, rho_evolve_4)

        return _index, final_exp_values

    def compute_rho_eellle(
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
            t1 + 2 * t_bin,
            self.c_op_list,
            self.aY,
            self.aY.dag(),
        )

        final_exp_values = np.zeros(
            (len(t_correlator_eval_early), len(t_correlator_eval_early)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_early):

            rho_evolve_3 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_1,
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

                final_exp_values[j][k] = expect(self.anY, rho_evolve_4)

        return _index, final_exp_values

    def compute_rho_leeell(
        self,
        _index,
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
            rho_start_list_late[_index],
            t1 + t_bin,
            t1 + 2 * t_bin,
            self.c_op_list,
            self.aY,
            self.aY.dag(),
        )

        final_exp_values = np.zeros(
            (len(t_correlator_eval_early), len(t_correlator_eval_early)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_early):

            rho_evolve_3 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_1,
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

                final_exp_values[j][k] = expect(self.anY, rho_evolve_4)

        return _index, final_exp_values

    def compute_rho_eeeele(
        self,
        _index,
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
            rho_start_list_late[_index],
            t1 + t_bin,
            t1 + 2 * t_bin,
            self.c_op_list,
            self.aY,
            self.aY.dag(),
        )

        final_exp_values = np.zeros(
            (len(t_correlator_eval_early), len(t_correlator_eval_early)),
            dtype=np.complex128,
        )

        for j, t2 in enumerate(t_correlator_eval_early):

            rho_evolve_3 = rho_evo_floating_start_finish(
                H_list,
                H_sim_time_list,
                rho_evolve_1,
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
                    5 * t_bin + t3,
                    self.c_op_list,
                    self.aY,
                    1,
                    debug=False,
                )

                final_exp_values[j][k] = expect(self.anY, rho_evolve_4)

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
        t_correlator_eval_early, t_bin = self.generate_time_correlator_eval(
            _n_correlator_eval, t_vst=self.length_stirap, t_rot=0.25
        )
        t_correlator_eval_late, t_bin = self.generate_time_correlator_eval(
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
        elif _element == "LEEELL":
            t1_label = t_correlator_eval_early + t_bin
            t2_label = t_correlator_eval_early + 2 * t_bin
            t3_label = t_correlator_eval_early + 5 * t_bin

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
                        self.compute_rho_leeell,
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
                for _ind, t1 in enumerate(t_correlator_eval_early):
                    _, final_exp_values = self.compute_rho_leeell(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_late[_ind],
                        t_correlator_eval_early,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        elif _element == "EELLLE":
            t1_label = t_correlator_eval_early
            t2_label = t_correlator_eval_early + 2 * t_bin
            t3_label = t_correlator_eval_early + 5 * t_bin

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
                        self.compute_rho_eellle,
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
                    _, final_exp_values = self.compute_rho_eellle(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_early[_ind],
                        t_correlator_eval_early,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        elif _element == "LELLLL":
            t1_label = t_correlator_eval_early + t_bin
            t2_label = t_correlator_eval_early + 2 * t_bin
            t3_label = t_correlator_eval_early + 5 * t_bin

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
                        self.compute_rho_lellll,
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
                for _ind, t1 in enumerate(t_correlator_eval_early):
                    _, final_exp_values = self.compute_rho_lellll(
                        _ind,
                        t1,
                        H_list,
                        H_sim_time_list,
                        rho_start_list_late[_ind],
                        t_correlator_eval_early,
                        t_bin,
                    )
                    exp_values[_ind][:][:] = final_exp_values

        elif _element == "EEEELE":
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
                        self.compute_rho_eeeeele,
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
                    _, final_exp_values = self.compute_rho_eeeeele(
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
                "Invalid matrix element specified. Choose from 'EEEEEE', 'LLLLLL', 'EELLEE', 'LEEELL', 'EELLLE', 'ELELEL', 'LELLLL', 'EEEELE'."
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


if __name__ == "__main__":
    # Initialize the class\

    _ground_states = {
        "g1M": 0,
        "g1": 1,
        "g1P": 2,  # F=1,mF=-1,0,+1 respectively
        "g2MM": 3,
        "g2M": 4,
        "g2": 5,
        "g2P": 6,
        "g2PP": 7,  # F=2,mF=-2,..,+2 respectively
    }

    # List the excited levels to include in the simulation. the _d1 levels correspond to the D1 line levels, the other levels are by default the d2 levels
    _x_states = [
        #'x0', 'x1M', 'x1', 'x1P',
        "x2MM",
        "x2M",
        "x2",
        "x2P",
        "x2PP",
        "x3MMM",
        #'x3MM', 'x3M', 'x3', 'x3P', 'x3PP', 'x3PPP',
        "x1M_d1",
        "x1_d1",
        "x1P_d1",
        "x2MM_d1",
        "x2M_d1",
        "x2_d1",
        "x2P_d1",
        "x2PP_d1",
    ]

    correlator_class = TimeBinPhotonCorrelationsCluster(
        "", _x_states, _ground_states, 50, 50, 1, 200
    )

    # test_n2_off_diag_1 = correlator_class.gen_n2_density_matrix_element(
    # 7, "ELLL", _plot=True, _debug=False, _parall=True)

    # print(test_n2_off_diag_1)

    test_n2_off_diag_2 = correlator_class.gen_n2_density_matrix_element(
        7, "EEEL", _plot=True, _debug=False, _parall=True
    )

    print(test_n2_off_diag_2)
