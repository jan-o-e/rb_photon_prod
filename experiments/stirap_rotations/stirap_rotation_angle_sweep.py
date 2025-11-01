
import numpy as np
from scipy import interpolate
import pickle

from modules.ham_sim_source import *
from modules.simulation import Simulation
from modules.differential_light_shifts import DifferentialStarkShifts
from modules.laser_pulses import *


def save_object_to_pkl(obj, filename):
    """
    Save an object to a .pkl file using pickle.

    Parameters:
        obj (object): The object to be saved.
        filename (str): The name of the .pkl file.

    Returns:
        None
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filename}")


# INSERT OPTIMAL PARAMETERS HERE
opt = {
    "param_1": 14.92,
    "laser_amplitude": 1459,
    "detuning": -33.82,
    "duration": 0.25,
    "detuning_magn": 1,
    "_n": 6,
    "_c": 0.25 / 3,
    "pulse_shape": "masked",
    "rotation_number": 3,
}

cav = False

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
    #'x0',
    #'x1M','x1','x1P',
    #'x2MM','x2M','x2','x2P','x2PP',
    #'x3MMM', 'x3MM','x3M','x3','x3P','x3PP', 'x3PPP',
    "x1M_d1",
    "x1_d1",
    "x1P_d1",
    "x2MM_d1",
    "x2M_d1",
    "x2_d1",
    "x2P_d1",
    "x2PP_d1",
]

rb_atom_sim = Simulation(
    cavity=cav,
    bfieldsplit="0p07",
    ground_states=_ground_states,
    x_states=_x_states,
    show_details=True,
)

# initial and final desired states, can get density matrix from psi*psi.dag()

saved_output = []

delta_sweep_list = np.linspace(-10, 10, 100)
for delta_sweep in delta_sweep_list:
    # first rotation
    psi_init_1 = (
        rb_atom_sim.kb_class.get_ket("g2", 0, 0)
        if cav
        else rb_atom_sim.kb_class.get_ket_atomic("g2")
    )
    psi_des_1 = (
        1
        / (np.sqrt(2))
        * (
            rb_atom_sim.kb_class.get_ket("g2", 0, 0)
            - rb_atom_sim.kb_class.get_ket("g1M", 0, 0)
        )
        if cav
        else 1
        / (np.sqrt(2))
        * (
            rb_atom_sim.kb_class.get_ket_atomic("g2")
            - rb_atom_sim.kb_class.get_ket_atomic("g1M")
        )
    )
    state_i_1 = "g2"
    F_i_1 = 2
    state_f_1 = "g1M"
    F_f_1 = 1
    state_x_1 = "x1M_d1"
    F_x_1 = 1
    state_u_1 = "g2"
    delta_p_1 = rb_atom_sim.get_splitting("deltaZx1M_d1") + delta_sweep
    delta_s_1 = rb_atom_sim.get_splitting("deltaZx1M_d1") - rb_atom_sim.get_splitting(
        "deltaZ"
    )
    cg_pump_1 = "CG_d1g2x1M"
    cg_stokes_1 = "CG_d1g1Mx1M"
    pump_pol_1 = "sigmaM"
    stokes_pol_1 = "pi"

    rotation_dict_1 = {
        "psi_init": psi_init_1,
        "psi_des": psi_des_1,
        "state_i": state_i_1,
        "state_f": state_f_1,
        "state_x": state_x_1,
        "delta_p": delta_p_1,
        "delta_s": delta_s_1,
        "cg_pump": cg_pump_1,
        "cg_stokes": cg_stokes_1,
        "pump_pol": pump_pol_1,
        "stokes_pol": stokes_pol_1,
        "F_x": F_x_1,
        "F_i": F_i_1,
        "F_f": F_f_1,
        "state_u": state_u_1,
    }

    # second rotation
    psi_init_2 = (
        1
        / (np.sqrt(2))
        * (
            rb_atom_sim.kb_class.get_ket("g2", 0, 0)
            - rb_atom_sim.kb_class.get_ket("g2MM", 0, 0)
        )
        if cav
        else 1
        / (np.sqrt(2))
        * (
            rb_atom_sim.kb_class.get_ket_atomic("g2")
            - rb_atom_sim.kb_class.get_ket_atomic("g2MM")
        )
    )
    psi_des_2 = (
        1
        / (np.sqrt(2))
        * (
            rb_atom_sim.kb_class.get_ket("g1P", 0, 0)
            - rb_atom_sim.kb_class.get_ket("g2MM", 0, 0)
        )
        if cav
        else 1
        / (np.sqrt(2))
        * (
            rb_atom_sim.kb_class.get_ket_atomic("g1P")
            - rb_atom_sim.kb_class.get_ket_atomic("g2MM")
        )
    )
    state_i_2 = "g2"
    F_i_2 = 2
    state_f_2 = "g1P"
    F_f_2 = 1
    state_x_2 = "x1_d1"
    F_x_2 = 1
    state_u_2 = "g2MM"
    delta_p_2 = 0 + delta_sweep
    delta_s_2 = rb_atom_sim.get_splitting("deltaZ")
    cg_pump_2 = "CG_d1g2x1"
    cg_stokes_2 = "CG_d1g1Px1"
    pump_pol_2 = "pi"
    stokes_pol_2 = "sigmaM"

    rotation_dict_2 = {
        "psi_init": psi_init_2,
        "psi_des": psi_des_2,
        "state_i": state_i_2,
        "state_f": state_f_2,
        "state_x": state_x_2,
        "delta_p": delta_p_2,
        "delta_s": delta_s_2,
        "cg_pump": cg_pump_2,
        "cg_stokes": cg_stokes_2,
        "pump_pol": pump_pol_2,
        "stokes_pol": stokes_pol_2,
        "F_x": F_x_2,
        "F_i": F_i_2,
        "F_f": F_f_2,
        "state_u": state_u_2,
    }

    # third rotation
    psi_init_3 = (
        1
        / (np.sqrt(2))
        * (
            rb_atom_sim.kb_class.get_ket("g2PP", 0, 0)
            - rb_atom_sim.kb_class.get_ket("g2MM", 0, 0)
        )
        if cav
        else 1
        / (np.sqrt(2))
        * (
            rb_atom_sim.kb_class.get_ket_atomic("g2PP")
            - rb_atom_sim.kb_class.get_ket_atomic("g2MM")
        )
    )
    psi_des_3 = (
        1
        / (np.sqrt(2))
        * (
            rb_atom_sim.kb_class.get_ket("g2PP", 0, 0)
            - rb_atom_sim.kb_class.get_ket("g1M", 0, 0)
        )
        if cav
        else 1
        / (np.sqrt(2))
        * (
            rb_atom_sim.kb_class.get_ket_atomic("g2PP")
            - rb_atom_sim.kb_class.get_ket_atomic("g1M")
        )
    )
    state_i_3 = "g2MM"
    F_i_3 = 2
    state_f_3 = "g1M"
    F_f_3 = 1
    state_x_3 = "x1M_d1"
    F_x_3 = 1
    state_u_3 = "g2PP"
    delta_p_3 = (
        rb_atom_sim.get_splitting("deltaZx1M_d1")
        + 2 * rb_atom_sim.get_splitting("deltaZ")
    ) + delta_sweep
    delta_s_3 = -rb_atom_sim.get_splitting("deltaZ") + rb_atom_sim.get_splitting(
        "deltaZx1M_d1"
    )
    cg_pump_3 = "CG_d1g2MMx1M"
    cg_stokes_3 = "CG_d1g1Mx1M"
    pump_pol_3 = "sigmaP"
    stokes_pol_3 = "pi"

    rotation_dict_3 = {
        "psi_init": psi_init_3,
        "psi_des": psi_des_3,
        "state_i": state_i_3,
        "state_f": state_f_3,
        "state_x": state_x_3,
        "delta_p": delta_p_3,
        "delta_s": delta_s_3,
        "cg_pump": cg_pump_3,
        "cg_stokes": cg_stokes_3,
        "pump_pol": pump_pol_3,
        "stokes_pol": stokes_pol_3,
        "F_x": F_x_3,
        "F_i": F_i_3,
        "F_f": F_f_3,
        "state_u": state_u_3,
    }

    # fourth rotation
    psi_init_4 = (
        1
        / (np.sqrt(2))
        * (
            rb_atom_sim.kb_class.get_ket("g2PP", 0, 0)
            - rb_atom_sim.kb_class.get_ket("g2MM", 0, 0)
        )
        if cav
        else 1
        / (np.sqrt(2))
        * (
            rb_atom_sim.kb_class.get_ket_atomic("g2PP")
            - rb_atom_sim.kb_class.get_ket_atomic("g2MM")
        )
    )
    psi_des_4 = (
        1
        / (np.sqrt(2))
        * (
            rb_atom_sim.kb_class.get_ket("g1P", 0, 0)
            - rb_atom_sim.kb_class.get_ket("g2MM", 0, 0)
        )
        if cav
        else 1
        / (np.sqrt(2))
        * (
            rb_atom_sim.kb_class.get_ket_atomic("g1P")
            - rb_atom_sim.kb_class.get_ket_atomic("g2MM")
        )
    )
    state_i_4 = "g2PP"
    F_i_4 = 2
    state_f_4 = "g1P"
    F_f_4 = 1
    state_x_4 = "x1P_d1"
    F_x_4 = 1
    state_u_4 = "g2MM"
    delta_p_4 = (
        rb_atom_sim.get_splitting("deltaZx1P_d1")
        - 2 * rb_atom_sim.get_splitting("deltaZ")
    ) + delta_sweep
    delta_s_4 = +rb_atom_sim.get_splitting("deltaZ") + rb_atom_sim.get_splitting(
        "deltaZx1P_d1"
    )
    cg_pump_4 = "CG_d1g2PPx1P"
    cg_stokes_4 = "CG_d1g1Px1P"
    pump_pol_4 = "sigmaM"
    stokes_pol_4 = "pi"

    rotation_dict_4 = {
        "psi_init": psi_init_4,
        "psi_des": psi_des_4,
        "state_i": state_i_4,
        "state_f": state_f_4,
        "state_x": state_x_4,
        "delta_p": delta_p_4,
        "delta_s": delta_s_4,
        "cg_pump": cg_pump_4,
        "cg_stokes": cg_stokes_4,
        "pump_pol": pump_pol_4,
        "stokes_pol": stokes_pol_4,
        "F_x": F_x_4,
        "F_i": F_i_4,
        "F_f": F_f_4,
        "state_u": state_u_4,
    }

    # convert optimisation parameters to variables
    pulse_param_1 = opt["param_1"]
    laser_amp = opt["laser_amplitude"]
    const_det = opt["detuning"]
    _length_repump = opt["duration"]
    detuning_magn = opt["detuning_magn"]
    pulse_shape = opt["pulse_shape"]
    rot_number = opt["rotation_number"]

    # select the correct rotation details
    if rot_number == 1:
        params = rotation_dict_1
    elif rot_number == 2:
        params = rotation_dict_2
    elif rot_number == 3:
        params = rotation_dict_3
    elif rot_number == 4:
        params = rotation_dict_4
    else:
        raise Exception("Invalid rotation number")

    psi_init = params["psi_init"]
    psi_des = params["psi_des"]

    # defining parameters that will be the same for all runs
    rho_des = psi_des * psi_des.dag()
    a = 1 / np.sqrt(2)
    b = 1 / (2 * np.pi)
    _delta_p = params["delta_p"]
    _delta_s = params["delta_s"]
    cg_pump_str = params["cg_pump"]
    cg_stokes_str = params["cg_stokes"]
    cg_pump = rb_atom_sim.get_CG(cg_pump_str)
    cg_stokes = rb_atom_sim.get_CG(cg_stokes_str)
    _pol_p = params["pump_pol"]
    _pol_s = params["stokes_pol"]
    state_i = params["state_i"]
    state_f = params["state_f"]
    state_x = params["state_x"]
    state_u = params["state_u"]
    F_i = params["F_i"]
    F_x = params["F_x"]
    F_f = params["F_f"]

    stokes_amp = laser_amp / cg_stokes
    pump_amp = laser_amp / cg_pump
    t = np.linspace(0, _length_repump, 10000)
    t_diff = np.linspace(0, _length_repump)

    # generate stokes and pump pulses WITH RADIAL FREQUENCY AS UNITS FOR AMPLITUDE
    if pulse_shape == "blackman":
        pump_pulse, stokes_pulse = create_blackman(
            t, pulse_param_1, pump_amp, stokes_amp
        )
    elif pulse_shape == "fstirap":
        pump_pulse, stokes_pulse = create_fstirap(
            t, pulse_param_1, pump_amp, stokes_amp
        )
        pump_pulse_diff, stokes_pulse_diff = create_fstirap(
            t_diff, pulse_param_1, pump_amp, stokes_amp
        )
    elif pulse_shape == "masked":
        _n = opt["_n"]
        _c = opt["_c"]
        pump_pulse, stokes_pulse = create_masked(
            t, pump_amp, stokes_amp, pulse_param_1, n=_n, c=_c
        )
        pump_pulse_diff, stokes_pulse_diff = create_masked(
            t_diff, pump_amp, stokes_amp, pulse_param_1, n=_n, c=_c
        )

    else:
        raise Exception("Invalid pulse shape")

    # calculate shifts from the stokes and pump laser pulses
    diff_shift = DifferentialStarkShifts(
        "d1", rb_atom_sim.rb_atom, rb_atom_sim.atom_states
    )
    shift_dict_stokes = diff_shift.calculate_td_detuning(
        F_f, b * stokes_pulse_diff, const_det, _pol_s
    )
    shift_dict_pump = diff_shift.calculate_td_detuning(
        F_i, b * pump_pulse_diff, const_det, _pol_p
    )
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

    # calculate time varying detuning from the shifts of the levels
    pump_det_spline = interpolate.CubicSpline(
        t_diff, (x_shift_tot - init_shift) * detuning_magn * 2 * np.pi
    )
    stokes_det_spline = interpolate.CubicSpline(
        t_diff, (x_shift_tot - fin_shift) * detuning_magn * 2 * np.pi
    )

    _pump_det = pump_det_spline(t)
    _stokes_det = stokes_det_spline(t)

    # run the simulation to find the final state density matrix
    (output_states_list, t_list) = rb_atom_sim.run_repreparation(
        const_det,
        t,
        _delta_p,
        _delta_s,
        _pol_p,
        _pol_s,
        pump_pulse,
        stokes_pulse,
        psi_init,
        F_i,
        F_x,
        F_f,
        F_x,
        pump_det=_pump_det,
        stokes_det=_stokes_det,
        raman_pulses=True,
        show_detuning=False,
    )

    psi_fin = output_states_list[-1]
    saved_output.append(psi_fin)

save_object_to_pkl(
    saved_output,
    f"saved_data/stirap_rotation_angle/angle_sweep/rotation_{rot_number}_{np.round(delta_sweep_list[0],3)}to{np.round(delta_sweep_list[-1],3)}.pkl",
)
