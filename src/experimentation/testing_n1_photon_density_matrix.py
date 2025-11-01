import numpy as np
from scipy import interpolate
from scipy.integrate import trapezoid
from qutip import (
    mesolve,
    tensor,
    qeye,
    destroy,
)

from src.modules.cavity import (
    cav_collapse_ops,
)
from src.modules.simulation import Simulation
from src.modules.differential_light_shifts import DifferentialStarkShifts
from src.modules.laser_pulses import (
    create_masked,
)
from src.modules.correlation_functions import (
    rho_evo_fixed_start,
    rho_evo_floating_start_finish,
)
from src.modules.photon_correlation_utils import plot_density_matrix_correlations

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

# _ground_states = {
#    "g1M":0, "g1P":1, # F=1,mF=-1,0,+1 respectively
#    "g2MM":2,"g2PP":3 # F=2,mF=-2,..,+2 respectively
# }

print(len(_ground_states))

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

# define list of collapse operators
c_op_list = []
c_op_list += cav_collapse_ops(rb_atom_sim.kappa, _ground_states)

# by default we are adding the collapse operators for both the d2 and d1 line, but comment out either if only one is desired
c_op_list += rb_atom_sim.rb_atom.spont_em_ops(_ground_states)[0]  # d2 line
c_op_list += rb_atom_sim.rb_atom.spont_em_ops(_ground_states)[1]  # d1 line

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

# VST PARAMS
delta_VST = 0 * 2 * np.pi
delta_cav = 2 * deltaZ + deltaZx2MM + delta_VST
delta_vst_laser_1 = -deltaZ + deltaZx2MM + delta_VST
delta_vst_laser_2 = deltaZ + deltaZx2PP + delta_VST
g_mhz = np.round(rb_atom_sim.coupling_factor, 3)
F_vst_start = 1
F_vst_final = 2
F_vst_exc = 2
n_steps_vst = 5000

# Define the pulse length and the number of time steps to be used in the simulation
pulse_len = 0.5
OmegaStirap = 70 * 2 * np.pi

wStirap = np.pi / pulse_len


def amp_shape(t):
    return np.sin(wStirap * t) ** 2


t_vst = np.linspace(0, pulse_len, n_steps_vst)
vst_driving_array = np.array(amp_shape(t_vst))

H_VStirap_1, args_hams_VStirap_1 = rb_atom_sim.gen_vstirap(
    lengthStirap,
    OmegaStirap,
    delta_vst_laser_1,
    "sigmaM",
    delta_cav,
    vst_driving_array,
    F_start=1,
    F_exc=2,
    F_end=2,
)
H_VStirap_2, args_hams_VStirap_2 = rb_atom_sim.gen_vstirap(
    lengthStirap,
    OmegaStirap,
    delta_vst_laser_2,
    "sigmaP",
    delta_cav,
    vst_driving_array,
    F_start=1,
    F_exc=2,
    F_end=2,
)

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
params_3 = rotation_dict_3

psi_des = params_3["psi_des"]

# defining parameters that will be the same for all runs
rho_des = psi_des * psi_des.dag()
_delta_p = params_3["delta_p"]
_delta_s = params_3["delta_s"]
cg_pump_str = params_3["cg_pump"]
cg_stokes_str = params_3["cg_stokes"]
cg_pump = rb_atom_sim.get_CG(cg_pump_str)
cg_stokes = rb_atom_sim.get_CG(cg_stokes_str)
_pol_p = params_3["pump_pol"]
_pol_s = params_3["stokes_pol"]
state_i = params_3["state_i"]
state_f = params_3["state_f"]
state_x = params_3["state_x"]
F_i = params_3["F_i"]
F_x = params_3["F_x"]
F_f = params_3["F_f"]

stokes_amp = laser_amp_3 / cg_stokes
pump_amp = laser_amp_3 / cg_pump
n_steps_rot3 = 2500
t_rot3 = np.linspace(0, _length_repump_3, n_steps_rot3)
t_diff = np.linspace(0, _length_repump_3)

# generate stokes and pump pulses WITH RADIAL FREQUENCY AS UNITS FOR AMPLITUDE
pump_pulse, stokes_pulse = create_masked(
    t_rot3, pump_amp, stokes_amp, _a_3, n=_n_3, c=_c_3
)
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

_pump_det = pump_det_spline(t_rot3)
_stokes_det = stokes_det_spline(t_rot3)

# _pump_det=np.ones_like(pump_pulse)
# _stokes_det=np.ones_like(pump_pulse)

# run the simulation to find the final state density matrix
H_rot_3, args_hams_rot3 = rb_atom_sim.gen_repreparation(
    const_det_3,
    t_rot3,
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
    include_cavity=False,
)

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
params_4 = rotation_dict_4

psi_des_4 = params_4["psi_des"]

# defining parameters that will be the same for all runs
rho_des = psi_des * psi_des.dag()
_delta_p = params_4["delta_p"]
_delta_s = params_4["delta_s"]
cg_pump_str = params_4["cg_pump"]
cg_stokes_str = params_4["cg_stokes"]
cg_pump = rb_atom_sim.get_CG(cg_pump_str)
cg_stokes = rb_atom_sim.get_CG(cg_stokes_str)
_pol_p = params_4["pump_pol"]
_pol_s = params_4["stokes_pol"]
state_i = params_4["state_i"]
state_f = params_4["state_f"]
state_x = params_4["state_x"]
F_i = params_4["F_i"]
F_x = params_4["F_x"]
F_f = params_4["F_f"]

stokes_amp = laser_amp_4 / cg_stokes
pump_amp = laser_amp_4 / cg_pump
n_steps_rot4 = 2500
t_rot4 = np.linspace(0, _length_repump_4, n_steps_rot4)
t_diff_4 = np.linspace(0, _length_repump_4)

# generate stokes and pump pulses WITH RADIAL FREQUENCY AS UNITS FOR AMPLITUDE
pump_pulse, stokes_pulse = create_masked(
    t_rot4, pump_amp, stokes_amp, _a_4, n=_n_4, c=_c_4
)
pump_pulse_diff, stokes_pulse_diff = create_masked(
    t_diff_4, pump_amp, stokes_amp, _a_4, n=_n_4, c=_c_4
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
    t_diff_4, (x_shift_tot - init_shift) * detuning_magn_4 * 2 * np.pi
)
stokes_det_spline = interpolate.CubicSpline(
    t_diff_4, (x_shift_tot - fin_shift) * detuning_magn_4 * 2 * np.pi
)

_pump_det = pump_det_spline(t_rot4)
_stokes_det = stokes_det_spline(t_rot4)

# _pump_det=np.ones_like(pump_pulse)
# _stokes_det=np.ones_like(pump_pulse)
# run the simulation to find the final state density matrix
H_rot_4, args_hams_rot4 = rb_atom_sim.gen_repreparation(
    const_det_4,
    t_rot4,
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
    include_cavity=False,
)

# truncating Fock states
N = 2
M = len(_ground_states)
print(f"States: {_ground_states}")
# Create the photon operators
aX = tensor(qeye(M), destroy(N), qeye(N))
aY = tensor(qeye(M), qeye(N), destroy(N))
anX = aX.dag() * aX
anY = aY.dag() * aY


psi0 = (
    1
    / (np.sqrt(2))
    * (
        rb_atom_sim.kb_class.get_ket("g2PP", 0, 0)
        - rb_atom_sim.kb_class.get_ket("g1M", 0, 0)
    )
)
H_list = [H_VStirap_1, H_rot_4, H_VStirap_2, H_rot_3]
H_sim_time_list = [t_vst, t_rot4, t_vst, t_rot3]

t_correlator_eval = np.linspace(0, 0.6, 30)
t_correlator_eval = np.append(t_correlator_eval, 0.75)
t_bin = np.array(0.75)

rho_start_list = rho_evo_fixed_start(
    H_list, H_sim_time_list, psi0, t_correlator_eval, c_op_list, 1, 1
)
# append psi0*psi0.dag() to the list of density matrices at 0th index
rho_start_list.insert(0, psi0 * psi0.dag())
exp_values_diag_zero = mesolve(
    H_VStirap_1, psi0 * psi0.dag(), t_vst, c_op_list, [anY]
).expect[0]
exp_values_diag_one = mesolve(
    H_VStirap_2, rho_start_list[-1], t_vst, c_op_list, [anY]
).expect[0]
# rho_full_list = rho_evo_fixed_start(H_list, H_sim_time_list, psi0, np.linspace(0,1.5,1000), c_op_list, 1, 1)
# plot=rb_atom_sim.rb_atom.plotter_atomstate_population(rb_atom_sim.rb_atom.ketbras.getrb_ketbras(),rho_full_list, np.linspace(0,1.5,1000)[1:], True)
exp_values_off_diag_zero = []
exp_values_off_diag_one = []


for j, t2 in enumerate(t_correlator_eval):
    rho_off_diag_one = rho_evo_floating_start_finish(
        H_list,
        H_sim_time_list,
        rho_start_list[j],
        t2,
        t2 + t_bin,
        c_op_list,
        1,
        aY.dag(),
    )
    # exp_values_off_diag_one.append(exp_off_diag_one[-1]
    exp_values_off_diag_one.append(rho_off_diag_one)

    # Loop over the elements of the three-dimensional array using enumerate
    # for k, row in enumerate(rho_off_diag_one):
    #    for l, sub_row in enumerate(row[0]):
    #        # Check if the absolute value of the element is greater than 0.1
    #        if abs(sub_row) > 0.05:
    #            # Print the element and its indices
    #            print(f"Value rho01: {sub_row}, Index: ({k}, {l})")

    # print(f"g2PP,1 g2MM,0 {rho_off_diag_one[12][0]}")
    # print(f"g2MM,1 g2PP,0 {rho_off_diag_one[8][0]}")

    rho_off_diag_zero = rho_evo_floating_start_finish(
        H_list, H_sim_time_list, rho_start_list[j], t2, t2 + t_bin, c_op_list, aY, 1
    )

    # exp_values_off_diag_zero.append(exp_off_diag_zero[-1])
    exp_values_off_diag_zero.append(rho_off_diag_zero)

    # Loop over the elements of the three-dimensional array using enumerate
    # for k, row in enumerate(rho_off_diag_zero):
    #    for l, sub_row in enumerate(row[0]):
    #        # Check if the absolute value of the element is greater than 0.1
    #        if abs(sub_row) > 0.1:
    # Print the element and its indices
    #            print(f"Value rho10: {sub_row}, Index: ({k}, {l})")
    # print(f"g2PP,1 g2MM,0 {rho_off_diag_zero[12][0]}")
    # print(f"g2MM,1 g2PP,0 {rho_off_diag_zero[8][0]}")

# Extract the real and imaginary parts for element [13,8] over time
coherence_01 = [
    exp_values_off_diag_one[time_point][29][0][12]
    for time_point in range(len(exp_values_off_diag_one))
]
coherence_10 = [
    exp_values_off_diag_zero[time_point][12][0][29]
    for time_point in range(len(exp_values_off_diag_zero))
]


# Integrate the real and imaginary parts
def integrate_real_imaginary(time_points, values):
    if len(time_points) != len(values):
        raise ValueError("Length of time_points and values must be equal")
    real_part = np.real(values)
    imag_part = np.imag(values)
    real_integral = trapezoid(real_part, time_points)
    imag_integral = trapezoid(imag_part, time_points)
    return real_integral, imag_integral


# Calculate integrals
int_off_diag_01_re, int_off_diag_01_im = integrate_real_imaginary(
    t_correlator_eval, coherence_01
)
int_off_diag_10_re, int_off_diag_10_im = integrate_real_imaginary(
    t_correlator_eval, coherence_10
)
int_diag_one_re, int_diag_one_im = integrate_real_imaginary(t_vst, exp_values_diag_one)
int_diag_zero_re, int_diag_zero_im = integrate_real_imaginary(
    t_vst, exp_values_diag_zero
)
# Print integrals
print("Integral of rho_00 (Real, Imaginary):", int_diag_zero_re, int_diag_zero_im)
print("Integral of rho_11 (Real, Imaginary):", int_diag_one_re, int_diag_one_im)
print("Integral of rho_01 (Real, Imaginary):", int_off_diag_01_re, int_off_diag_01_im)
print("Integral of rho_10 (Real, Imaginary):", int_off_diag_10_re, int_off_diag_10_im)


# Plot using the utility function
plot_density_matrix_correlations(
    H_sim_time_list=[t_vst, t_vst, t_vst, t_rot3],
    exp_values_diag_zero=exp_values_diag_zero,
    exp_values_diag_one=exp_values_diag_one,
    t_correlator_eval=t_correlator_eval,
    coherence_01=coherence_01,
    save_dir="saved_data_timebin/photon_correlations/n_1",
    n_start=1,
    length_stirap=lengthStirap,
    bfield_split=rb_atom_sim.bfieldsplit,
)
