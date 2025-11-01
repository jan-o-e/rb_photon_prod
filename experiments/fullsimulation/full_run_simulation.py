import pickle

import matplotlib.pyplot as plt
import numpy as np
from qutip import fidelity
from scipy import interpolate

from src.modules.cavity import plotter_cavemission
from src.modules.simulation import Simulation
from src.modules.differential_light_shifts import DifferentialStarkShifts
from src.modules.laser_pulses import *

# INSERT OPTIMAL PARAMETERS HERE
first_params = {
    "param_1": 0.7293,
    "laser_amplitude": 573.4,
    "detuning": -37.92,
    "duration": 0.1603,
    "detuning_magn": 1,
    "pulse_shape": "fstirap",
    "rotation_number": 1,
}

second_params = {
    "param_1": 10.32611058680373,
    "laser_amplitude": 754.1772283803055,
    "detuning": -1.3177928786079605,
    "duration": 0.25,
    "detuning_magn": 1,
    "_n": 3,
    "_c": 0.25 / 3,
    "pulse_shape": "masked",
    "rotation_number": 2,
}

third_params = {
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

fourth_params = {
    "param_1": 14.53,
    "laser_amplitude": 1139,
    "detuning": -40.28,
    "duration": 0.25,
    "detuning_magn": 1,
    "_n": 6,
    "_c": 0.25 / 3,
    "pulse_shape": "masked",
    "rotation_number": 4,
}

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

rb_atom_sim = Simulation(
    cavity=True,
    bfieldsplit="0p07",
    ground_states=_ground_states,
    x_states=_x_states,
    show_details=True,
)

# initial and final desired states, can get density matrix from psi*psi.dag()

# first rotation
psi_init_1 = rb_atom_sim.kb_class.get_ket("g2", 0, 0)
psi_des_1 = (
    1
    / (np.sqrt(2))
    * (
        rb_atom_sim.kb_class.get_ket("g2", 0, 0)
        - rb_atom_sim.kb_class.get_ket("g1M", 0, 0)
    )
)
state_i_1 = "g2"
F_i_1 = 2
state_f_1 = "g1M"
F_f_1 = 1
state_x_1 = "x1M_d1"
F_x_1 = 1
delta_p_1 = rb_atom_sim.get_splitting("deltaZx1M_d1")
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
}

# second rotation
psi_init_2 = (
    1
    / (np.sqrt(2))
    * (
        rb_atom_sim.kb_class.get_ket("g2", 0, 0)
        - rb_atom_sim.kb_class.get_ket("g2MM", 0, 0)
    )
)
psi_des_2 = (
    1
    / (np.sqrt(2))
    * (
        rb_atom_sim.kb_class.get_ket("g1P", 0, 0)
        - rb_atom_sim.kb_class.get_ket("g2MM", 0, 0)
    )
)
state_i_2 = "g2"
F_i_2 = 2
state_f_2 = "g1P"
F_f_2 = 1
state_x_2 = "x1_d1"
F_x_2 = 1
delta_p_2 = 0
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
}

# third rotation
psi_init_3 = (
    1
    / (np.sqrt(2))
    * (
        rb_atom_sim.kb_class.get_ket("g2PP", 0, 0)
        - rb_atom_sim.kb_class.get_ket("g2MM", 0, 0)
    )
)
psi_des_3 = (
    1
    / (np.sqrt(2))
    * (
        rb_atom_sim.kb_class.get_ket("g2PP", 0, 0)
        - rb_atom_sim.kb_class.get_ket("g1M", 0, 0)
    )
)
state_i_3 = "g2MM"
F_i_3 = 2
state_f_3 = "g1M"
F_f_3 = 1
state_x_3 = "x1M_d1"
F_x_3 = 1
delta_p_3 = rb_atom_sim.get_splitting("deltaZx1M_d1") + 2 * rb_atom_sim.get_splitting(
    "deltaZ"
)
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
}


# fourth rotation
psi_init_4 = (
    1
    / (np.sqrt(2))
    * (
        rb_atom_sim.kb_class.get_ket("g2PP", 0, 0)
        - rb_atom_sim.kb_class.get_ket("g2MM", 0, 0)
    )
)
psi_des_4 = (
    1
    / (np.sqrt(2))
    * (
        rb_atom_sim.kb_class.get_ket("g1P", 0, 0)
        - rb_atom_sim.kb_class.get_ket("g2MM", 0, 0)
    )
)
state_i_4 = "g2PP"
F_i_4 = 2
state_f_4 = "g1P"
F_f_4 = 1
state_x_4 = "x1P_d1"
F_x_4 = 1
delta_p_4 = rb_atom_sim.get_splitting("deltaZx1P_d1") - 2 * rb_atom_sim.get_splitting(
    "deltaZ"
)
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
}


# params fixed for all runs
a = 1 / np.sqrt(2)
b = 1 / (2 * np.pi)
lengthStirap = 0.5
# delta_VST = -3*2*np.pi
delta_VST = 0
deltaZ = rb_atom_sim.get_splitting("deltaZ")
deltaZx2MM = rb_atom_sim.get_splitting("deltaZx2MM")
deltaZx2PP = rb_atom_sim.get_splitting("deltaZx2PP")
delta_cav = 2 * deltaZ + deltaZx2MM + delta_VST


########   #######  ########    ###    ######## ####  #######  ##    ##       ##
##     ## ##     ##    ##      ## ##      ##     ##  ##     ## ###   ##     ####
##     ## ##     ##    ##     ##   ##     ##     ##  ##     ## ####  ##       ##
########  ##     ##    ##    ##     ##    ##     ##  ##     ## ## ## ##       ##
##   ##   ##     ##    ##    #########    ##     ##  ##     ## ##  ####       ##
##    ##  ##     ##    ##    ##     ##    ##     ##  ##     ## ##   ###       ##
##     ##  #######     ##    ##     ##    ##    ####  #######  ##    ##     ######
# convert optimisation parameters to variables

tau_1 = first_params["param_1"]
laser_amp_1 = first_params["laser_amplitude"]
const_det_1 = first_params["detuning"]
_length_repump_1 = first_params["duration"]
detuning_magn_1 = first_params["detuning_magn"]
pulse_shape_1 = first_params["pulse_shape"]
rot_number_1 = first_params["rotation_number"]

# select the correct rotation details
assert rot_number_1 == 1
params = rotation_dict_1

psi_init = params["psi_init"]
psi_des = params["psi_des"]
rho_des = psi_des * psi_des.dag()

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
F_i = params["F_i"]
F_x = params["F_x"]
F_f = params["F_f"]

stokes_amp = laser_amp_1 / cg_stokes
pump_amp = laser_amp_1 / cg_pump
t = np.linspace(0, _length_repump_1, 3000)
t_diff = np.linspace(0, _length_repump_1)

# generate stokes and pump pulses WITH RADIAL FREQUENCY AS UNITS FOR AMPLITUDE
pump_pulse, stokes_pulse = create_fstirap(t, tau_1, pump_amp, stokes_amp)
pump_pulse_diff, stokes_pulse_diff = create_fstirap(t_diff, tau_1, pump_amp, stokes_amp)

# calculate shifts from the stokes and pump laser pulses
diff_shift = DifferentialStarkShifts("d1", rb_atom_sim.rb_atom, rb_atom_sim.atom_states)
shift_dict_stokes = diff_shift.calculate_td_detuning(
    F_f, b * stokes_pulse_diff, const_det_1, _pol_s
)
shift_dict_pump = diff_shift.calculate_td_detuning(
    F_i, b * pump_pulse_diff, const_det_1, _pol_p
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
    t_diff, (x_shift_tot - init_shift) * detuning_magn_1 * 2 * np.pi
)
stokes_det_spline = interpolate.CubicSpline(
    t_diff, (x_shift_tot - fin_shift) * detuning_magn_1 * 2 * np.pi
)

_pump_det = pump_det_spline(t)
_stokes_det = stokes_det_spline(t)
with open("src/visualisation/pulse_chirps/fstirap_chirp.pkl", "wb") as f:
    pickle.dump([_pump_det, _stokes_det, t], f)

# run the simulation to find the final state density matrix
(output_states_list_1, t_list_1) = rb_atom_sim.run_repreparation(
    const_det_1,
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
)

plot_output = output_states_list_1
plot_timelist = t_list_1

"""
rb_atom_sim.rb_atom.plotter_spontdecay_channels(_ground_states, plot_output,plot_timelist)
rb_atom_sim.rb_atom.plotter_atomstate_population(rb_atom_sim.kb_class.getrb_ketbras(), plot_output,plot_timelist, True)
plt.show()
"""

psi_fin_1 = output_states_list_1[-1]

rho_fin = psi_fin_1 * psi_fin_1.dag()
fid_1 = fidelity(rho_fin, rho_des)
print(f"fidelity after rotation {rot_number_1} is {fid_1}")


##     ##  ######  ######## #### ########     ###    ########        ##
##     ## ##    ##    ##     ##  ##     ##   ## ##   ##     ##     ####
##     ## ##          ##     ##  ##     ##  ##   ##  ##     ##       ##
##     ##  ######     ##     ##  ########  ##     ## ########        ##
##   ##        ##    ##     ##  ##   ##   ######### ##              ##
## ##   ##    ##    ##     ##  ##    ##  ##     ## ##              ##
###     ######     ##    #### ##     ## ##     ## ##            ######


psi_init_v1 = psi_fin_1

t = np.linspace(0, lengthStirap, 1000)

OmegaStirap_v1 = 70 * 2 * np.pi
delta_laser_v1 = -deltaZ + deltaZx2MM + delta_VST
laser_pulse_v1 = create_single_sinsquared(t, 1)

(output_states_list_v1, t_list_v1) = rb_atom_sim.run_vstirap(
    lengthStirap,
    OmegaStirap_v1,
    delta_laser_v1,
    "sigmaM",
    delta_cav,
    psi_init_v1,
    laser_pulse_v1,
    F_start=1,
    F_exc=2,
    F_end=2,
)


plot_output = output_states_list_v1
plot_timelist = t_list_v1
"""
rb_atom_sim.rb_atom.plotter_spontdecay_channels(_ground_states, plot_output,plot_timelist)
rb_atom_sim.rb_atom.plotter_atomstate_population(rb_atom_sim.kb_class.getrb_ketbras(), plot_output,plot_timelist, True)
plt.show()
"""

psi_fin_v1 = output_states_list_v1[-1]

rho_fin = psi_fin_v1 * psi_fin_v1.dag()
# print(f"fidelity after stirap process 1 is {fidelity(rho_fin, rho_des)}")


########   #######  ########    ###    ######## ####  #######  ##    ##     #######
##     ## ##     ##    ##      ## ##      ##     ##  ##     ## ###   ##    ##     ##
##     ## ##     ##    ##     ##   ##     ##     ##  ##     ## ####  ##           ##
########  ##     ##    ##    ##     ##    ##     ##  ##     ## ## ## ##     #######
##   ##   ##     ##    ##    #########    ##     ##  ##     ## ##  ####    ##
##    ##  ##     ##    ##    ##     ##    ##     ##  ##     ## ##   ###    ##
##     ##  #######     ##    ##     ##    ##    ####  #######  ##    ##    #########
# convert optimisation parameters to variables
_a_2 = second_params["param_1"]
_n_2 = second_params["_n"]
_c_2 = second_params["_c"]
laser_amp_2 = second_params["laser_amplitude"]
const_det_2 = second_params["detuning"]
_length_repump_2 = second_params["duration"]
detuning_magn_2 = second_params["detuning_magn"]
pulse_shape_2 = second_params["pulse_shape"]
rot_number_2 = second_params["rotation_number"]

# select the correct rotation details
assert rot_number_2 == 2
params = rotation_dict_2

psi_init = psi_fin_v1  #      CHANGE INITIAL STATE FOR THIS ROTATION HERE!
psi_des = params["psi_des"]

# defining parameters that will be the same for all runs
rho_des = psi_des * psi_des.dag()
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
F_i = params["F_i"]
F_x = params["F_x"]
F_f = params["F_f"]

stokes_amp = laser_amp_2 / cg_stokes
pump_amp = laser_amp_2 / cg_pump
t = np.linspace(0, _length_repump_2, 3000)
t_diff = np.linspace(0, _length_repump_2)

# generate stokes and pump pulses WITH RADIAL FREQUENCY AS UNITS FOR AMPLITUDE
pump_pulse, stokes_pulse = create_masked(t, pump_amp, stokes_amp, _a_2, n=_n_2, c=_c_2)
pump_pulse_diff, stokes_pulse_diff = create_masked(
    t_diff, pump_amp, stokes_amp, _a_2, n=_n_2, c=_c_2
)

# calculate shifts from the stokes and pump laser pulses
diff_shift = DifferentialStarkShifts("d1", rb_atom_sim.rb_atom, rb_atom_sim.atom_states)
shift_dict_stokes = diff_shift.calculate_td_detuning(
    F_f, b * stokes_pulse_diff, const_det_2, _pol_s
)
shift_dict_pump = diff_shift.calculate_td_detuning(
    F_i, b * pump_pulse_diff, const_det_2, _pol_p
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
    t_diff, (x_shift_tot - init_shift) * detuning_magn_2 * 2 * np.pi
)
stokes_det_spline = interpolate.CubicSpline(
    t_diff, (x_shift_tot - fin_shift) * detuning_magn_2 * 2 * np.pi
)

_pump_det = pump_det_spline(t)
_stokes_det = stokes_det_spline(t)

with open("src/visualisation/pulse_chirps/rot_2_chirp.pkl", "wb") as f:
    pickle.dump([_pump_det, _stokes_det, t], f)

# run the simulation to find the final state density matrix
(output_states_list_2, t_list_2) = rb_atom_sim.run_repreparation(
    const_det_2,
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
)

psi_fin_2 = output_states_list_2[-1]

rho_fin = psi_fin_2 * psi_fin_2.dag()
print(f"fidelity after rotation 2 is {fidelity(rho_fin, rho_des)}")


##     ##  ######  ######## #### ########     ###    ########      #######
##     ## ##    ##    ##     ##  ##     ##   ## ##   ##     ##    ##     ##
##     ## ##          ##     ##  ##     ##  ##   ##  ##     ##           ##
##     ##  ######     ##     ##  ########  ##     ## ########      #######
##   ##        ##    ##     ##  ##   ##   ######### ##           ##
## ##   ##    ##    ##     ##  ##    ##  ##     ## ##           ##
###     ######     ##    #### ##     ## ##     ## ##           #########
psi_init_v2 = psi_fin_2

t = np.linspace(0, lengthStirap, 1000)

OmegaStirap_v2 = 70 * 2 * np.pi
delta_laser_v2 = deltaZ + deltaZx2PP + delta_VST
laser_pulse_v2 = create_single_sinsquared(t, 1)

(output_states_list_v2, t_list_v2) = rb_atom_sim.run_vstirap(
    lengthStirap,
    OmegaStirap_v2,
    delta_laser_v2,
    "sigmaP",
    delta_cav,
    psi_init_v2,
    laser_pulse_v2,
    F_start=1,
    F_exc=2,
    F_end=2,
)


psi_fin_v2 = output_states_list_v2[-1]

rho_fin = psi_fin_v2 * psi_fin_v2.dag()
# print(f"fidelity after stirap process 2 is {fidelity(rho_fin, rho_des)}")


########   #######  ########    ###    ######## ####  #######  ##    ##     #######
##     ## ##     ##    ##      ## ##      ##     ##  ##     ## ###   ##    ##     ##
##     ## ##     ##    ##     ##   ##     ##     ##  ##     ## ####  ##           ##
########  ##     ##    ##    ##     ##    ##     ##  ##     ## ## ## ##     #######
##   ##   ##     ##    ##    #########    ##     ##  ##     ## ##  ####           ##
##    ##  ##     ##    ##    ##     ##    ##     ##  ##     ## ##   ###    ##     ##
##     ##  #######     ##    ##     ##    ##    ####  #######  ##    ##     #######
# convert optimisation parameters to variables
_a_3 = third_params["param_1"]
_n_3 = third_params["_n"]
_c_3 = third_params["_c"]
laser_amp_3 = third_params["laser_amplitude"]
const_det_3 = third_params["detuning"]
_length_repump_3 = third_params["duration"]
detuning_magn_3 = third_params["detuning_magn"]
pulse_shape_3 = third_params["pulse_shape"]
rot_number_3 = third_params["rotation_number"]

# select the correct rotation details
assert rot_number_3 == 3
params = rotation_dict_3

psi_init = psi_fin_v2  #      CHANGE INITIAL STATE FOR THIS ROTATION HERE!
psi_des = params["psi_des"]

# defining parameters that will be the same for all runs
rho_des = psi_des * psi_des.dag()
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
F_i = params["F_i"]
F_x = params["F_x"]
F_f = params["F_f"]

stokes_amp = laser_amp_3 / cg_stokes
pump_amp = laser_amp_3 / cg_pump
t = np.linspace(0, _length_repump_3, 3000)
t_diff = np.linspace(0, _length_repump_3)

# generate stokes and pump pulses WITH RADIAL FREQUENCY AS UNITS FOR AMPLITUDE
pump_pulse, stokes_pulse = create_masked(t, pump_amp, stokes_amp, _a_3, n=_n_3, c=_c_3)
pump_pulse_diff, stokes_pulse_diff = create_masked(
    t_diff, pump_amp, stokes_amp, _a_3, n=_n_3, c=_c_3
)

# calculate shifts from the stokes and pump laser pulses
diff_shift = DifferentialStarkShifts("d1", rb_atom_sim.rb_atom, rb_atom_sim.atom_states)
shift_dict_stokes = diff_shift.calculate_td_detuning(
    F_f, b * stokes_pulse_diff, const_det_3, _pol_s
)
shift_dict_pump = diff_shift.calculate_td_detuning(
    F_i, b * pump_pulse_diff, const_det_3, _pol_p
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
    t_diff, (x_shift_tot - init_shift) * detuning_magn_3 * 2 * np.pi
)
stokes_det_spline = interpolate.CubicSpline(
    t_diff, (x_shift_tot - fin_shift) * detuning_magn_3 * 2 * np.pi
)

_pump_det = pump_det_spline(t)
_stokes_det = stokes_det_spline(t)
with open("src/visualisation/pulse_chirps/rot_3_chirp.pkl", "wb") as f:
    pickle.dump([_pump_det, _stokes_det, t], f)

# run the simulation to find the final state density matrix
(output_states_list_3, t_list_3) = rb_atom_sim.run_repreparation(
    const_det_3,
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
)

psi_fin_3 = output_states_list_3[-1]


rho_fin = psi_fin_3 * psi_fin_3.dag()
print(f"fidelity after rotation 3 is {fidelity(rho_fin, rho_des)}")


#  ##     ##  ######  ######## #### ########     ###    ########      #######
#  ##     ## ##    ##    ##     ##  ##     ##   ## ##   ##     ##    ##     ##
#  ##     ## ##          ##     ##  ##     ##  ##   ##  ##     ##           ##
#  ##     ##  ######     ##     ##  ########  ##     ## ########      #######
#   ##   ##        ##    ##     ##  ##   ##   ######### ##                  ##
#    ## ##   ##    ##    ##     ##  ##    ##  ##     ## ##           ##     ##
#     ###     ######     ##    #### ##     ## ##     ## ##            #######
psi_init_v3 = psi_fin_3

t = np.linspace(0, lengthStirap, 1000)

OmegaStirap_v3 = 70 * 2 * np.pi

delta_laser_v3 = -deltaZ + deltaZx2MM + delta_VST

laser_pulse_v3 = create_single_sinsquared(t, 1)

(output_states_list_v3, t_list_v3) = rb_atom_sim.run_vstirap(
    lengthStirap,
    OmegaStirap_v3,
    delta_laser_v3,
    "sigmaM",
    delta_cav,
    psi_init_v3,
    laser_pulse_v1,
    F_start=1,
    F_exc=2,
    F_end=2,
)

psi_fin_v3 = output_states_list_v3[-1]

rho_fin = psi_fin_v3 * psi_fin_v3.dag()
# print(f"fidelity after stirap process 3 is {fidelity(rho_fin, rho_des)}")


#  ########   #######  ########    ###    ######## ####  #######  ##    ##    ##
#  ##     ## ##     ##    ##      ## ##      ##     ##  ##     ## ###   ##    ##    ##
#  ##     ## ##     ##    ##     ##   ##     ##     ##  ##     ## ####  ##    ##    ##
#  ########  ##     ##    ##    ##     ##    ##     ##  ##     ## ## ## ##    ##    ##
#  ##   ##   ##     ##    ##    #########    ##     ##  ##     ## ##  ####    #########
#  ##    ##  ##     ##    ##    ##     ##    ##     ##  ##     ## ##   ###          ##
#  ##     ##  #######     ##    ##     ##    ##    ####  #######  ##    ##          ##
# convert optimisation parameters to variables
_a_4 = fourth_params["param_1"]
_n_4 = fourth_params["_n"]
_c_4 = fourth_params["_c"]
laser_amp_4 = fourth_params["laser_amplitude"]
const_det_4 = fourth_params["detuning"]
_length_repump_4 = fourth_params["duration"]
detuning_magn_4 = fourth_params["detuning_magn"]
pulse_shape_4 = fourth_params["pulse_shape"]
rot_number_4 = fourth_params["rotation_number"]

# select the correct rotation details
assert rot_number_4 == 4
params = rotation_dict_4

psi_init = psi_fin_v3  #      CHANGE INITIAL STATE FOR THIS ROTATION HERE!
psi_des = params["psi_des"]

# defining parameters that will be the same for all runs
rho_des = psi_des * psi_des.dag()
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
F_i = params["F_i"]
F_x = params["F_x"]
F_f = params["F_f"]

stokes_amp = laser_amp_4 / cg_stokes
pump_amp = laser_amp_4 / cg_pump
t = np.linspace(0, _length_repump_4, 3000)
t_diff = np.linspace(0, _length_repump_4)

# generate stokes and pump pulses WITH RADIAL FREQUENCY AS UNITS FOR AMPLITUDE
pump_pulse, stokes_pulse = create_masked(t, pump_amp, stokes_amp, _a_4, n=_n_4, c=_c_4)
pump_pulse_diff, stokes_pulse_diff = create_masked(
    t_diff, pump_amp, stokes_amp, _a_4, n=_n_4, c=_c_4
)

# calculate shifts from the stokes and pump laser pulses
diff_shift = DifferentialStarkShifts("d1", rb_atom_sim.rb_atom, rb_atom_sim.atom_states)
shift_dict_stokes = diff_shift.calculate_td_detuning(
    F_f, b * stokes_pulse_diff, const_det_4, _pol_s
)
shift_dict_pump = diff_shift.calculate_td_detuning(
    F_i, b * pump_pulse_diff, const_det_4, _pol_p
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
    t_diff, (x_shift_tot - init_shift) * detuning_magn_4 * 2 * np.pi
)
stokes_det_spline = interpolate.CubicSpline(
    t_diff, (x_shift_tot - fin_shift) * detuning_magn_4 * 2 * np.pi
)

_pump_det = pump_det_spline(t)
_stokes_det = stokes_det_spline(t)
with open("src/visualisation/pulse_chirps/rot_4_chirp.pkl", "wb") as f:
    pickle.dump([_pump_det, _stokes_det, t], f)

# run the simulation to find the final state density matrix
(output_states_list_4, t_list_4) = rb_atom_sim.run_repreparation(
    const_det_4,
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
)

psi_fin_4 = output_states_list_4[-1]

rho_fin = psi_fin_4 * psi_fin_4.dag()
print(f"fidelity after rotation 4 is {fidelity(rho_fin, rho_des)}")

psi_init_v4 = psi_fin_4

t = np.linspace(0, lengthStirap, 1000)

OmegaStirap_v4 = 70 * 2 * np.pi
delta_laser_v4 = deltaZ + deltaZx2PP + delta_VST
laser_pulse_v4 = create_single_sinsquared(t, 1)

(output_states_list_v4, t_list_v4) = rb_atom_sim.run_vstirap(
    lengthStirap,
    OmegaStirap_v4,
    delta_laser_v4,
    "sigmaP",
    delta_cav,
    psi_init_v4,
    laser_pulse_v4,
    F_start=1,
    F_exc=2,
    F_end=2,
)

psi_fin_v4 = output_states_list_v4[-1]
rho_fin = psi_fin_v4 * psi_fin_v4.dag()

plt.style.use("ggplot")
"""
amp_fig, amp_ax = plt.subplots(1,1,figsize=(10,10))
amp_ax.title.set_text("amplitude pulses")
amp_ax.plot(t,stokes_pulse,label="stokes",color="green")
amp_ax.plot(t,pump_pulse,label="pump",color="darkred")
amp_ax.legend(loc="best")


det_fig, det_ax = plt.subplots(1,1,figsize=(10,10))
det_ax.title.set_text("detuning evolution")
det_ax.plot(t,_stokes_det+const_det, label="stokes",color="green")
det_ax.plot(t,_pump_det+const_det,label="pump",color="darkred")
det_ax.legend(loc="best")


score_fig, score_ax = plt.subplots(1,1)
score_ax.title.set_text("fidelity change")
score = []
for state in output_states_list:
    rho = state*state.dag()
    score.append(fidelity(rho_des,rho))
score_ax.plot(t, score)
score_ax.set_ylim(0,1)
"""


def combine_timelist(time_lists: tuple[list, ...]):
    """
    Combines a tuple of time lists into one continuous array of times
    """
    arrays = [np.array(time_lists[0])]
    for i in range(1, len(time_lists)):
        arrays.append(np.array(time_lists[i]) + arrays[-1][-1])

    return np.concatenate(arrays, axis=None)


t_list_full = combine_timelist(
    (t_list_1, t_list_v1, t_list_2, t_list_v2, t_list_3, t_list_v3, t_list_4, t_list_v4)
)

output_states_list_full = (
    output_states_list_1
    + output_states_list_v1
    + output_states_list_2
    + output_states_list_v2
    + output_states_list_3
    + output_states_list_v3
    + output_states_list_4
    + output_states_list_v4
)

# save timelists individually and the concatenated one
# with open ("/Users/ernst/Desktop/rb_photon_prod_dev/saved_data/full_sim_data/no_kappa/timelists.pkl", "wb") as f:
#    pickle.dump([t_list_1, t_list_v1, t_list_2, t_list_v2,
#                 t_list_3, t_list_v3, t_list_4, t_list_v4, t_list_full], f)

# save all desnity matrix evolution for detailed phase analysis
# with open ("/Users/ernst/Desktop/rb_photon_prod_dev/saved_data/full_sim_data/no_kappa/density_matrix_evolution.pkl", "wb") as f:
#    pickle.dump([output_states_list_1, output_states_list_v1, output_states_list_2, output_states_list_v2, output_states_list_3, output_states_list_v3, output_states_list_4, output_states_list_v4], f)

plot_output = output_states_list_full
plot_timelist = t_list_full

rb_atom_sim.rb_atom.plotter_spontdecay_channels(
    _ground_states, plot_output, plot_timelist
)
rb_atom_sim.rb_atom.plotter_atomstate_population(
    rb_atom_sim.kb_class.getrb_ketbras(), plot_output, plot_timelist, True
)
rb_atom_sim.rb_atom.plotter_atomstate_population(
    rb_atom_sim.kb_class.getrb_ketbras(), plot_output, plot_timelist, False
)

plotter_cavemission(
    rb_atom_sim.ketbras,
    rb_atom_sim.atom_states,
    output_states_list_full,
    t_list_full,
    rb_atom_sim.kappa,
    show_plts=True,
)


plt.show(block=False)
"""

rho_d_4 = np.around(rho_des.full(),4)
rho_f_4 = np.around(rho_fin.full(),4)

i, f = 2,3

print(rho_d_4[i][i], rho_d_4[i][f], rho_d_4[f][i], rho_d_4[f][f])
print(rho_f_4[i][i], rho_f_4[i][f], rho_f_4[f][i], rho_f_4[f][f])

print(fidelity(rho_fin,rho_des))
"""
input("Press enter to close the plots\n")

plt.close("all")
