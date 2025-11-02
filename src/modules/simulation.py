import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from itertools import chain
from qutip import mesolve, Qobj

from modules.laser_pulses import (
    create_masked,
    create_fstirap,
    create_single_blackman,
    create_flattop_blackman,
    create_flattop_gaussian,
)
from modules.atom_config import RbAtom
from modules.cavity import quant_axis_cavbasis_mapping, cav_collapse_ops
from modules.ketbra_config import RbKetBras
from modules.differential_light_shifts import DifferentialStarkShifts


class Simulation:

    def __init__(
        self,
        cavity=True,
        _kappa=2.1,
        bfieldsplit: str = None,
        ground_states: dict = None,
        x_states: list = None,
        show_details=False,
    ) -> None:
        """
        Creates an instance of the simulation class with the specified parameters.

        Args:
            cavity: Whether to include couplings to the cavity
            _kappa: Decay rate of the cavity (MHz)
            bfieldsplit: B-field ground state splitting (MHz). Default is "0"
            ground_states: Dictionary of ground states to include (name: index)
            x_states: List of excited state names
            show_details: Whether to display simulation details
        """
        self.show_details = show_details
        self.cavity = cavity
        self.photonic_space = cavity

        self.bfieldsplit = bfieldsplit or "0"

        self.ground_states = ground_states or {
            "g1M": 0,
            "g1": 1,
            "g1P": 2,  # F=1, mF=-1,0,+1
            "g2MM": 3,
            "g2M": 4,
            "g2": 5,
            "g2P": 6,
            "g2PP": 7,  # F=2, mF=-2,..,+2
        }

        self.x_states = x_states or [
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

        self.initialise()

        if self.cavity:
            # Cavity parameters
            self.kappa = _kappa * 2 * np.pi  # Cavity field decay rate
            self.deltaP = 0.0  # Cavity birefringence (polarization splitting)

            # Atom-cavity coupling configuration
            self.desired_vst_line = "d2"
            norm = np.abs(self.get_CG("CGg1Mx1"))
            self.cav_transition = self.get_CG("CGg2MMx2MM")
            self.coupling_factor = 11.1 * 2 * np.pi / norm

            # Calculate cooperativity
            gamma = self.gamma_d2 if self.desired_vst_line == "d2" else self.gamma_d1
            self.coop = (self.cav_transition * self.coupling_factor) ** 2 / (
                2 * self.kappa * gamma
            )

            if self.show_details:
                print(
                    f"Atom-cavity coupling in {self.bfieldsplit} MHz split hyperfine states: "
                    f"gLev={np.round(self.cav_transition * self.coupling_factor / (2*np.pi), 3)} MHz"
                )
                print(
                    f"Cooperativity C={self.coop:.3f}, theoretical max efficiency "
                    f"eta={2*self.coop/(2*self.coop+1):.3f}"
                )

            # Quantization and cavity axes (3D vectors)
            self.cav_axis = [1, 0, 0]
            self.quant_axis = [0, 1, 0]

    def initialise(self) -> bool:
        """Initialize atomic parameters, Clebsch-Gordan coefficients, and energy splittings."""
        self.kb_class = RbKetBras(
            self.ground_states, self.x_states, self.photonic_space
        )
        self.ketbras = self.kb_class.getrb_ketbras()  # Precompute for speed
        self.atom_states = self.kb_class.atom_states

        # Configure Rb atom with CG coefficients and splittings
        self.rb_atom = RbAtom(self.bfieldsplit, self.kb_class)
        self.splittings_dict = self.rb_atom.get_splittings_dict()
        self.CG_coeffs_dict = self.rb_atom.get_cg_dict()

        # Atomic decay rates and dipole moments
        self.gamma_d2, self.gamma_d1, self.d_d2, self.d_d1 = self.rb_atom.getrb_rates()

        return True

    def reset(self) -> None:
        """Resets all parameters and attributes to their initial state"""
        # Reset attributes
        self.kb_class = None
        self.ketbras = None
        self.atom_states = None
        self.rb_atom = None
        self.splittings_dict = None
        self.CG_coeffs_dict = None
        self.gamma_d2 = None
        self.gamma_d1 = None
        self.d_d2 = None
        self.d_d1 = None
        self.coop = None
        self.cav_transition = None
        self.coupling_factor = None
        self.kappa = None
        self.deltaP = None
        self.desired_vst_line = None
        self.cav_axis = None
        self.quant_axis = None

    def get_CG(self, CG_coeff):
        """Return Clebsch-Gordan coefficient value."""
        try:
            return self.CG_coeffs_dict[CG_coeff]
        except KeyError:
            raise Exception(f"Invalid Clebsch-Gordan coefficient: {CG_coeff}")

    def get_splitting(self, splitting):
        """Return energy splitting value."""
        try:
            return self.splittings_dict[splitting]
        except KeyError:
            raise Exception(f"Invalid splitting: {splitting}")

    def _prepare_pulse_params(
        self,
        delta,
        delta_p,
        delta_s,
        pump_array,
        stokes_array,
        repump_t_array,
        pump_det,
        stokes_det,
    ):
        """Prepare and normalize pulse parameters."""
        delta_p_total = (delta + delta_p) * 2 * np.pi
        delta_s_total = (delta + delta_s) * 2 * np.pi

        # Normalize arrays
        omega_p_max = np.max(np.absolute(pump_array))
        omega_s_max = np.max(np.absolute(stokes_array))
        norm_pump_arr = pump_array / omega_p_max
        norm_stokes_arr = stokes_array / omega_s_max

        # Default detuning arrays to zero if not provided
        pump_det = pump_det if pump_det is not None else np.zeros(len(repump_t_array))
        stokes_det = (
            stokes_det if stokes_det is not None else np.zeros(len(repump_t_array))
        )

        stokes_args = {
            "_array": True,
            "_amp": norm_stokes_arr,
            "_t": repump_t_array,
            "_phase": stokes_det,
        }
        pump_args = {
            "_array": True,
            "_amp": norm_pump_arr,
            "_t": repump_t_array,
            "_phase": pump_det,
        }

        return (
            delta_p_total,
            delta_s_total,
            omega_p_max,
            omega_s_max,
            stokes_args,
            pump_args,
        )

    def gen_repreparation(
        self,
        delta: float,
        repump_t_array: np.ndarray,
        delta_p: float,
        delta_s: float,
        pol_p: str,
        pol_s: str,
        pump_array: np.ndarray,
        stokes_array: np.ndarray,
        F_pump_start: int,
        F_pump_exc: int,
        F_stokes_start: int,
        F_stokes_exc: int,
        pump_det: np.ndarray = None,
        stokes_det: np.ndarray = None,
        raman_pulses: bool = False,
        show_detuning: bool = False,
        include_cavity=False,
    ):
        """
        Generate repreparation Hamiltonian.

        Args:
            delta: Two-photon detuning from atomic resonance (MHz)
            repump_t_array: Time array for simulation
            delta_p, delta_s: Fixed detunings for pump and Stokes (MHz)
            pol_p, pol_s: Polarizations for pump and Stokes
            pump_array, stokes_array: Pulse amplitudes (MHz)
            F_pump_start, F_pump_exc: F states for pump transition
            F_stokes_start, F_stokes_exc: F states for Stokes transition
            pump_det, stokes_det: Time-dependent detunings (MHz)
            raman_pulses: Whether to include couplings to both ground levels
            show_detuning: Whether to plot detuning
            include_cavity: Whether to include cavity coupling

        Returns:
            H_repump: Hamiltonian list
            args_hams_repump: Arguments dictionary
        """
        (
            delta_p_total,
            delta_s_total,
            omega_p_max,
            omega_s_max,
            stokes_args,
            pump_args,
        ) = self._prepare_pulse_params(
            delta,
            delta_p,
            delta_s,
            pump_array,
            stokes_array,
            repump_t_array,
            pump_det,
            stokes_det,
        )

        if self.show_details and show_detuning:
            y_vals = np.exp(1j * delta_s_total * repump_t_array)
            plt.plot(repump_t_array, np.real(y_vals), label="real")
            plt.plot(repump_t_array, np.imag(y_vals), label="imag")
            plt.legend()
            plt.show()

        # Generate Hamiltonians
        gen_method = (
            self.rb_atom.gen_H_RamanPulse_D1
            if raman_pulses
            else self.rb_atom.gen_H_Pulse_D1
        )

        H_Stirap_Stokes = gen_method(
            self.ketbras,
            self.atom_states,
            delta_s_total,
            F_stokes_start,
            F_stokes_exc,
            pol_s,
            omega_s_max,
            {},
            **stokes_args,
        )
        H_Stirap_Pump = gen_method(
            self.ketbras,
            self.atom_states,
            delta_p_total,
            F_pump_start,
            F_pump_exc,
            pol_p,
            omega_p_max,
            {},
            **pump_args,
        )

        if include_cavity:
            H_VStirap, args_hams_VStirap = self.rb_atom.gen_H_VSTIRAP_D2(
                self.ketbras,
                self.atom_states,
                0,
                0,
                1,
                2,
                2,
                "sigmaM",
                0,
                self.coupling_factor,
                self.deltaP,
                quant_axis_cavbasis_mapping(self.quant_axis, self.cav_axis),
                None,
                {},
                _array=True,
                _amp=np.zeros(len(stokes_args["_amp"])),
                _t=repump_t_array,
                _phase=np.zeros(len(stokes_args["_amp"])),
            )
            H_repump = list(chain(*[H_Stirap_Stokes[0], H_Stirap_Pump[0], H_VStirap]))
        else:
            H_repump = list(chain(*[H_Stirap_Stokes[0], H_Stirap_Pump[0]]))

        args_hams_repump = {**H_Stirap_Stokes[1], **H_Stirap_Pump[1]}
        return H_repump, args_hams_repump

    def gen_far_detuned_raman(self, config, n_steps=1000):
        """
        Generate far-detuned Raman Hamiltonian.

        Args:
            config: Configuration dictionary with pulse parameters
            n_steps: Number of time steps

        Returns:
            ham: Hamiltonian list
            sim_pulse_time: Time array
            args: Arguments dictionary
        """
        two_phot_det = config["two_photon_det"]
        dead_time = config.get("dead_time", 0)
        time_per_step = config["pulse_length"] / n_steps

        pulse_time = np.linspace(0, config["pulse_length"], n_steps)

        # Create pulses with both ramp up and down times
        ramp_time = config["rise_time"]
        pulse_1 = create_flattop_blackman(pulse_time, 1, ramp_time, ramp_time)
        pulse_2 = create_flattop_blackman(pulse_time, 1, ramp_time, ramp_time)

        # Calculate detunings and amplitudes
        det_1 = (config["det_centre"] * 2 * np.pi) + config["det_zeeman_1"]
        det_2 = (
            det_1
            - two_phot_det * 2 * np.pi
            - self.rb_atom.getrb_gs_splitting()
            + config["det_zeeman_2"]
        )
        amp_1 = (
            config["amp_scaling"]
            * np.sqrt(2 * np.abs(config["det_centre"]))
            * 2
            * np.pi
            / config["cg_1"]
        )
        amp_2 = (
            config["amp_scaling"]
            * np.sqrt(2 * np.abs(config["det_centre"]))
            * 2
            * np.pi
            / config["cg_2"]
        )

        # Add dead time if specified
        sim_pulse_time = np.linspace(
            0,
            config["pulse_length"] + dead_time,
            n_steps + int(dead_time / time_per_step),
        )
        pulse_1 = np.concatenate((pulse_1, np.zeros(int(dead_time / time_per_step))))
        pulse_2 = np.concatenate((pulse_2, np.zeros(int(dead_time / time_per_step))))

        ham, args = self.rb_atom.gen_H_FarDetuned_Raman_PulsePair_D1(
            self.ketbras,
            self.atom_states,
            det_1,
            det_2,
            config["pol_1"],
            config["pol_2"],
            amp_1,
            amp_2,
            sim_pulse_time,
            pulse_1,
            pulse_2,
        )

        return ham, sim_pulse_time, args

    def run_repreparation(
        self,
        delta: float,
        repump_t_array: np.ndarray,
        delta_p: float,
        delta_s: float,
        pol_p: str,
        pol_s: str,
        pump_array: np.ndarray,
        stokes_array: np.ndarray,
        psi0: Qobj,
        F_pump_start: int,
        F_pump_exc: int,
        F_stokes_start: int,
        F_stokes_exc: int,
        pump_det: np.ndarray = None,
        stokes_det: np.ndarray = None,
        raman_pulses: bool = False,
        d2_decay: bool = False,
        show_detuning: bool = False,
    ):
        """
        Run repreparation simulation.

        Args:
            Same as gen_repreparation, plus:
            psi0: Initial atomic state
            d2_decay: Whether to include D2 line decay

        Returns:
            If show_details: (output_states_list, t_list)
            Otherwise: final state (Qobj)
        """
        # Generate Hamiltonian using helper
        H_repump, args_hams_repump = self.gen_repreparation(
            delta,
            repump_t_array,
            delta_p,
            delta_s,
            pol_p,
            pol_s,
            pump_array,
            stokes_array,
            F_pump_start,
            F_pump_exc,
            F_stokes_start,
            F_stokes_exc,
            pump_det,
            stokes_det,
            raman_pulses,
            show_detuning,
            include_cavity=False,
        )

        # Setup collapse operators
        c_op_list = []
        if d2_decay:
            c_op_list += self.rb_atom.spont_em_ops(self.atom_states)[0]  # D2 line
        c_op_list += self.rb_atom.spont_em_ops(self.atom_states)[1]  # D1 line

        # Run simulation
        output = mesolve(
            H_repump, psi0, tlist=repump_t_array, c_ops=c_op_list, args=args_hams_repump
        )

        if self.show_details:
            return output.states, repump_t_array
        else:
            return output.states[-1]

    def gen_vstirap(
        self,
        stirap_length: float,
        laser_amp,
        laser_delta: float,
        laser_pol,
        cav_delta,
        stirap_pulse,
        detuning_array=None,
        F_start=1,
        F_end=2,
        F_exc=2,
    ):
        """
        Generate VSTIRAP Hamiltonian.

        Args:
            stirap_length: Duration of the process
            laser_amp: Maximum Rabi frequency of the laser pulse (MHz)
            laser_delta: Fixed detuning of the laser from the transition
            laser_pol: Polarization of the laser
            cav_delta: Cavity detuning
            stirap_pulse: Shape of the laser pulse (numpy array)
            detuning_array: Time-dependent laser detuning
            F_start, F_end, F_exc: Initial, final and excited F values

        Returns:
            H_VStirap: Hamiltonian list
            args_hams_VStirap: Arguments dictionary
        """
        tVST, tVSTStep = np.linspace(0, stirap_length, len(stirap_pulse), retstep=True)
        detuning_array = (
            detuning_array if detuning_array is not None else np.zeros(len(tVST))
        )

        assert len(tVST) == len(
            stirap_pulse
        ), "Time array and pulse array must have same length"

        stirap_args = {
            "_array": True,
            "_amp": stirap_pulse,
            "_t": tVST,
            "_phase": detuning_array,
        }

        H_VStirap, args_hams_VStirap = self.rb_atom.gen_H_VSTIRAP_D2(
            self.ketbras,
            self.atom_states,
            cav_delta,
            laser_delta,
            F_start,
            F_end,
            F_exc,
            laser_pol,
            laser_amp,
            self.coupling_factor,
            self.deltaP,
            quant_axis_cavbasis_mapping(self.quant_axis, self.cav_axis),
            None,
            **stirap_args,
        )

        return H_VStirap, args_hams_VStirap

    def run_vstirap(
        self,
        stirap_length: float,
        laser_amp,
        laser_delta: float,
        laser_pol,
        cav_delta,
        psi0,
        stirap_pulse,
        detuning_array=None,
        F_start=1,
        F_end=2,
        F_exc=2,
    ):
        """
        Run VSTIRAP simulation.

        Args:
            Same as gen_vstirap, plus:
            psi0: Initial state

        Returns:
            If show_details: (output_states_list, t_list)
            Otherwise: final state (Qobj)
        """
        tVST, _ = np.linspace(0, stirap_length, len(stirap_pulse), retstep=True)

        # Generate Hamiltonian
        H_VStirap, args_hams_VStirap = self.gen_vstirap(
            stirap_length,
            laser_amp,
            laser_delta,
            laser_pol,
            cav_delta,
            stirap_pulse,
            detuning_array,
            F_start,
            F_end,
            F_exc,
        )

        # Setup collapse operators
        c_op_list = []
        c_op_list += cav_collapse_ops(self.kappa, self.atom_states)
        c_op_list += self.rb_atom.spont_em_ops(self.atom_states)[0]  # D2 line
        c_op_list += self.rb_atom.spont_em_ops(self.atom_states)[1]  # D1 line

        # Run simulation
        output = mesolve(H_VStirap, psi0, tVST, c_op_list, [], args=args_hams_VStirap)

        if self.show_details:
            return output.states, tVST
        else:
            return output.states[-1]

    def _get_state(self, state_name, include_cavity):
        """Helper to get state with or without cavity."""
        if include_cavity:
            return self.kb_class.get_ket(state_name, 0, 0)
        else:
            return self.kb_class.get_ket_atomic(state_name)

    def get_rotation_dict(self, n_rot, _two_photon_det, include_cavity=False):
        """
        Get rotation dictionary for specified Raman rotation.

        Args:
            n_rot: Rotation number (1, 2, 3, or 4)
            _two_photon_det: Two-photon detuning
            include_cavity: Whether to include cavity in the simulation

        Returns:
            Dictionary with rotation parameters
        """
        sqrt2 = 1 / np.sqrt(2)

        if n_rot == 1:
            psi_init = self._get_state("g2", include_cavity)
            psi_des = sqrt2 * (
                self._get_state("g2", include_cavity)
                - self._get_state("g1M", include_cavity)
            )
            return {
                "psi_init": psi_init,
                "psi_des": psi_des,
                "state_i": "g2",
                "state_f": "g1M",
                "state_x": "x1M_d1",
                "delta_p": self.get_splitting("deltaZx1M_d1") + _two_photon_det,
                "delta_s": self.get_splitting("deltaZx1M_d1")
                - self.get_splitting("deltaZ"),
                "cg_pump": self.get_CG("CG_d1g2x1M"),
                "cg_stokes": self.get_CG("CG_d1g1Mx1M"),
                "pump_pol": "sigmaM",
                "stokes_pol": "pi",
                "F_x": 1,
                "F_i": 2,
                "F_f": 1,
            }

        elif n_rot == 2:
            psi_init = sqrt2 * (
                self._get_state("g2", include_cavity)
                - self._get_state("g2MM", include_cavity)
            )
            psi_des = sqrt2 * (
                self._get_state("g1P", include_cavity)
                - self._get_state("g2MM", include_cavity)
            )
            return {
                "psi_init": psi_init,
                "psi_des": psi_des,
                "state_i": "g2",
                "state_f": "g1P",
                "state_x": "x1_d1",
                "delta_p": _two_photon_det,
                "delta_s": self.get_splitting("deltaZ"),
                "cg_pump": self.get_CG("CG_d1g2x1"),
                "cg_stokes": self.get_CG("CG_d1g1Px1"),
                "pump_pol": "pi",
                "stokes_pol": "sigmaM",
                "F_x": 1,
                "F_i": 2,
                "F_f": 1,
            }

        elif n_rot == 3:
            psi_init = sqrt2 * (
                self._get_state("g2PP", include_cavity)
                - self._get_state("g2MM", include_cavity)
            )
            psi_des = sqrt2 * (
                self._get_state("g2PP", include_cavity)
                - self._get_state("g1M", include_cavity)
            )
            return {
                "psi_init": psi_init,
                "psi_des": psi_des,
                "state_i": "g2MM",
                "state_f": "g1M",
                "state_x": "x1M_d1",
                "delta_p": self.get_splitting("deltaZx1M_d1")
                + 2 * self.get_splitting("deltaZ")
                + _two_photon_det,
                "delta_s": -self.get_splitting("deltaZ")
                + self.get_splitting("deltaZx1M_d1"),
                "cg_pump": self.get_CG("CG_d1g2MMx1M"),
                "cg_stokes": self.get_CG("CG_d1g1Mx1M"),
                "pump_pol": "sigmaP",
                "stokes_pol": "pi",
                "F_x": 1,
                "F_i": 2,
                "F_f": 1,
            }

        elif n_rot == 4:
            psi_init = sqrt2 * (
                self._get_state("g2PP", include_cavity)
                - self._get_state("g2MM", include_cavity)
            )
            psi_des = sqrt2 * (
                self._get_state("g1P", include_cavity)
                - self._get_state("g2MM", include_cavity)
            )
            return {
                "psi_init": psi_init,
                "psi_des": psi_des,
                "state_i": "g2PP",
                "state_f": "g1P",
                "state_x": "x1P_d1",
                "delta_p": self.get_splitting("deltaZx1P_d1")
                - 2 * self.get_splitting("deltaZ")
                + _two_photon_det,
                "delta_s": self.get_splitting("deltaZ")
                + self.get_splitting("deltaZx1P_d1"),
                "cg_pump": self.get_CG("CG_d1g2PPx1P"),
                "cg_stokes": self.get_CG("CG_d1g1Px1P"),
                "pump_pol": "sigmaM",
                "stokes_pol": "pi",
                "F_x": 1,
                "F_i": 2,
                "F_f": 1,
            }

        else:
            raise ValueError(
                f"Invalid rotation number: {n_rot}. Must be 1, 2, 3, or 4."
            )

    def generate_timebin_rotation_hamiltonian(
        self, params, _include_cavity=True, _chirped=True, _n_steps=2500
    ):
        """
        Generate Hamiltonian for the specified Raman rotation.

        Args:
            params: Dictionary of rotation parameters
            _include_cavity: Whether to include cavity in simulation
            _chirped: Whether to include chirped pulses
            _n_steps: Number of time steps for simulation

        Returns:
            H_rot: Hamiltonian list
            t_rot: Time array
            args_hams: Arguments dictionary
        """
        # Extract optimization parameters
        _a = params["param_1"]
        laser_amp = params["laser_amplitude"]
        const_det = params["detuning"]
        two_photon_det = params["two_photon_detuning"]
        _length_repump = params["duration"]
        detuning_magn = params["detuning_magn"]
        pulse_shape = params["pulse_shape"]
        rotation_number = params["rotation_number"]
        rotation_dict = self.get_rotation_dict(
            rotation_number, two_photon_det, _include_cavity
        )

        # Extract rotation specific details
        _delta_p = rotation_dict["delta_p"]
        _delta_s = rotation_dict["delta_s"]
        cg_pump = rotation_dict["cg_pump"]
        cg_stokes = rotation_dict["cg_stokes"]
        _pol_p = rotation_dict["pump_pol"]
        _pol_s = rotation_dict["stokes_pol"]
        state_i = rotation_dict["state_i"]
        state_f = rotation_dict["state_f"]
        state_x = rotation_dict["state_x"]
        F_i = rotation_dict["F_i"]
        F_x = rotation_dict["F_x"]
        F_f = rotation_dict["F_f"]

        # Calculate amplitudes
        stokes_amp = laser_amp / cg_stokes
        pump_amp = laser_amp / cg_pump
        n_steps = _n_steps
        t_rot = np.linspace(0, _length_repump, n_steps)
        t_diff = np.linspace(0, _length_repump)

        # Generate pump and stokes pulses
        if pulse_shape == "masked":
            _n = params["_n"]
            _c = params["_c"]
            pump_pulse, stokes_pulse = create_masked(
                t_rot, pump_amp, stokes_amp, _a, n=_n, c=_c
            )
            pump_pulse_diff, stokes_pulse_diff = create_masked(
                t_diff, pump_amp, stokes_amp, _a, n=_n, c=_c
            )
        elif pulse_shape == "fstirap":
            pump_pulse, stokes_pulse = create_fstirap(t_rot, _a, pump_amp, stokes_amp)
            pump_pulse_diff, stokes_pulse_diff = create_fstirap(
                t_diff, _a, pump_amp, stokes_amp
            )
        else:
            raise ValueError(
                f"Invalid pulse shape: {pulse_shape}. Choose 'masked' or 'fstirap'."
            )

        # Calculate Stark shifts
        if _chirped:
            b = 1 / (2 * np.pi)  # Normalization factor
            diff_shift = DifferentialStarkShifts("d1", self.rb_atom, self.atom_states)

            # Calculate time-dependent detuning (pulses in MHz not rad/s)
            shift_dict_stokes = diff_shift.calculate_td_detuning(
                F_f, b * stokes_pulse_diff, const_det, _pol_s
            )
            shift_dict_pump = diff_shift.calculate_td_detuning(
                F_i, b * pump_pulse_diff, const_det, _pol_p
            )

            # Find state evolution
            init_shift = diff_shift.find_state_evolution(
                b * pump_pulse_diff, shift_dict_pump, state_i
            )
            x_shift_p = diff_shift.find_state_evolution(
                b * pump_pulse_diff, shift_dict_pump, state_x
            )
            fin_shift = diff_shift.find_state_evolution(
                b * stokes_pulse_diff, shift_dict_stokes, state_f
            )
            x_shift_s = diff_shift.find_state_evolution(
                b * stokes_pulse_diff, shift_dict_stokes, state_x
            )
            x_shift_tot = x_shift_p + x_shift_s

            # Time-varying detuning via spline interpolation
            pump_det_spline = interpolate.CubicSpline(
                t_diff, (x_shift_tot - init_shift) * detuning_magn * 2 * np.pi
            )
            stokes_det_spline = interpolate.CubicSpline(
                t_diff, (x_shift_tot - fin_shift) * detuning_magn * 2 * np.pi
            )

            _pump_det = pump_det_spline(t_rot)
            _stokes_det = stokes_det_spline(t_rot)
        else:
            _pump_det = np.ones_like(pump_pulse)
            _stokes_det = np.ones_like(pump_pulse)

        # Generate Hamiltonian
        H_rot, args_hams = self.gen_repreparation(
            const_det,
            t_rot,
            _delta_p,
            _delta_s,
            _pol_p,
            _pol_s,
            pump_pulse,
            stokes_pulse,
            F_i,
            F_x,
            F_f,
            F_x,
            pump_det=_pump_det,
            stokes_det=_stokes_det,
            raman_pulses=False,
            include_cavity=_include_cavity,
        )
        return H_rot, t_rot, args_hams

    def generate_timebin_far_detuned_raman_hamiltonian(self, params, _n_steps=5000):
        return self.gen_far_detuned_raman(params, n_steps=_n_steps)

    def generate_timebin_vst_hamiltonian(
        self, vst_params, _n_steps_vst=5000, _ramp_down_time=0.2, _ramp_up_time=0.2
    ):
        """
        Generate Hamiltonian for the specified VSTIRAP process.

        Args:
            vst_params: Dictionary of VSTIRAP parameters
            _n_steps_vst: Number of time steps for simulation
            _ramp_down_time: Ramp down time for flattop pulses
            _ramp_up_time: Ramp up time for flattop pulses

        Returns:
            H_VStirap_1, H_VStirap_2: Hamiltonian lists
            t_vst: Time array
            args_hams_VStirap_1, args_hams_VStirap_2: Arguments dictionaries
        """
        # VST parameters and calculations
        delta_VST_1 = vst_params["delta_VST_1"]
        delta_VST_2 = vst_params["delta_VST_2"]
        delta_cav = (
            2 * self.get_splitting("deltaZ")
            + self.get_splitting("deltaZx2MM")
            + delta_VST_1
        )
        delta_vst_laser_1 = (
            -self.get_splitting("deltaZ")
            + self.get_splitting("deltaZx2MM")
            + delta_VST_1
        )
        delta_vst_laser_2 = (
            self.get_splitting("deltaZ")
            + self.get_splitting("deltaZx2PP")
            + delta_VST_2
        )
        pulse_shape = vst_params["pulse_shape"]

        # Define pulse length and driving array
        pulse_len = vst_params["lengthStirap"]
        OmegaStirap1 = vst_params["OmegaStirap_1"]
        OmegaStirap2 = vst_params["OmegaStirap_2"]
        wStirap = np.pi / pulse_len
        t_vst = np.linspace(0, pulse_len, _n_steps_vst)

        if pulse_shape == "sinsquared":
            vst_driving_array = np.sin(wStirap * t_vst) ** 2
        elif pulse_shape == "blackman":
            vst_driving_array = create_single_blackman(t_vst, 1)
        elif pulse_shape == "flattop_gaussian":
            vst_driving_array = create_flattop_gaussian(
                t_vst, 1, _ramp_up_time, _ramp_down_time
            )
        elif pulse_shape == "flattop_blackman":
            vst_driving_array = create_flattop_blackman(
                t_vst, 1, _ramp_up_time, _ramp_down_time
            )
        else:
            raise ValueError(
                f"Invalid pulse shape: {pulse_shape}. Choose 'sinsquared', 'blackman', 'flattop_gaussian', or 'flattop_blackman'."
            )

        # Generate Hamiltonians for VST pulses
        H_VStirap_1, args_hams_VStirap_1 = self.gen_vstirap(
            pulse_len,
            OmegaStirap1,
            delta_vst_laser_1,
            "sigmaM",
            delta_cav,
            vst_driving_array,
            F_start=1,
            F_exc=2,
            F_end=2,
        )
        H_VStirap_2, args_hams_VStirap_2 = self.gen_vstirap(
            pulse_len,
            OmegaStirap2,
            delta_vst_laser_2,
            "sigmaP",
            delta_cav,
            vst_driving_array,
            F_start=1,
            F_exc=2,
            F_end=2,
        )
        return H_VStirap_1, H_VStirap_2, t_vst, args_hams_VStirap_1, args_hams_VStirap_2
