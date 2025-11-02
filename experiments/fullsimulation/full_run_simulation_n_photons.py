import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import time
import pickle
import sys
import os

import numpy as np
from scipy import interpolate

from modules.cavity import plotter_cavemission, cav_emission_from_state
from modules.simulation import Simulation
from modules.differential_light_shifts import DifferentialStarkShifts
from modules.laser_pulses import *


output_states_list_full=[]
t_list_full=[]
emission_list=[]

# INSERT OPTIMAL PARAMETERS HERE
first_params = {
    "param_1": 0.7293,
    "laser_amplitude": 573.4,
    "detuning": -37.92,
    "duration": 0.1603,
    "detuning_magn": 1,
    "pulse_shape": "fstirap",
    "rotation_number":1
}

second_params = {
    "param_1":   10.32611058680373,
    "laser_amplitude": 754.1772283803055,
    "detuning": -1.3177928786079605,
    "duration": 0.25,
    "detuning_magn": 1,
    "_n": 3,
    "_c": 0.25/3,
    "pulse_shape": "masked",
    "rotation_number":2
}

third_params = {
    "param_1": 14.92,
    "laser_amplitude": 1139,
    "detuning": -33.82,
    "duration": 0.25,
    "detuning_magn": 1,
    "_n": 6,
    "_c": 0.25/3,
    "pulse_shape": "masked",
    "rotation_number":3
}

fourth_params = {
    "param_1": 14.53,
    "laser_amplitude": 1139,
    "detuning": -40.28,
    "duration": 0.25,
    "detuning_magn": 1,
    "_n": 6,
    "_c": 0.25/3,
    "pulse_shape": "masked",
    "rotation_number":4
}

_ground_states = {
    "g1M":0, "g1":1, "g1P":2, # F=1,mF=-1,0,+1 respectively
    "g2MM":3, "g2M":4, "g2":5, "g2P":6, "g2PP":7 # F=2,mF=-2,..,+2 respectively
}


# List the excited levels to include in the simulation. the _d1 levels correspond to the D1 line levels, the other levels are by default the d2 levels

_x_states = [
'x0',
'x1M','x1','x1P',
'x2MM','x2M','x2','x2P','x2PP',
'x3MMM', 'x3MM','x3M','x3','x3P','x3PP', 'x3PPP',
'x1M_d1','x1_d1','x1P_d1',
'x2MM_d1','x2M_d1','x2_d1','x2P_d1','x2PP_d1'
]

rb_atom_sim = Simulation(cavity=True, bfieldsplit="0p07",ground_states=_ground_states, x_states=_x_states, show_details=True)


#first rotation
psi_init_1   = rb_atom_sim.kb_class.get_ket("g2",0,0)
psi_des_1    = 1/(np.sqrt(2))*(rb_atom_sim.kb_class.get_ket("g2",0,0) - rb_atom_sim.kb_class.get_ket("g1M",0,0))
state_i_1    = "g2"
F_i_1        = 2
state_f_1    = "g1M"
F_f_1        = 1
state_x_1    = "x1M_d1"
F_x_1        = 1
delta_p_1    = rb_atom_sim.get_splitting("deltaZx1M_d1")
delta_s_1    = (rb_atom_sim.get_splitting("deltaZx1M_d1")-rb_atom_sim.get_splitting("deltaZ"))
cg_pump_1    = "CG_d1g2x1M"
cg_stokes_1  = "CG_d1g1Mx1M"
pump_pol_1   = 'sigmaM'
stokes_pol_1 = 'pi'

rotation_dict_1 = {"psi_init":psi_init_1, "psi_des":psi_des_1, "state_i":state_i_1, "state_f":state_f_1,\
                   "state_x":state_x_1, "delta_p":delta_p_1, "delta_s":delta_s_1, "cg_pump":cg_pump_1,\
                    "cg_stokes":cg_stokes_1, "pump_pol":pump_pol_1, "stokes_pol":stokes_pol_1,
                    "F_x":F_x_1, "F_i":F_i_1, "F_f":F_f_1}

#second rotation
psi_init_2   = 1/(np.sqrt(2))*(rb_atom_sim.kb_class.get_ket('g2',0,0)-rb_atom_sim.kb_class.get_ket('g2MM',0,0))
psi_des_2    = 1/(np.sqrt(2))*(rb_atom_sim.kb_class.get_ket('g1P',0,0)-rb_atom_sim.kb_class.get_ket('g2MM',0,0))
state_i_2    = "g2"
F_i_2        = 2
state_f_2    = "g1P"
F_f_2        = 1
state_x_2    = "x1_d1"
F_x_2        = 1
delta_p_2    = 0
delta_s_2    = rb_atom_sim.get_splitting("deltaZ")
cg_pump_2    = "CG_d1g2x1"
cg_stokes_2  = "CG_d1g1Px1"
pump_pol_2   = "pi"
stokes_pol_2 = "sigmaM"


rotation_dict_2 = {"psi_init":psi_init_2, "psi_des":psi_des_2, "state_i":state_i_2, "state_f":state_f_2,\
                   "state_x":state_x_2, "delta_p":delta_p_2, "delta_s":delta_s_2, "cg_pump":cg_pump_2,\
                    "cg_stokes":cg_stokes_2, "pump_pol":pump_pol_2, "stokes_pol":stokes_pol_2,
                    "F_x":F_x_2, "F_i":F_i_2, "F_f":F_f_2}

#third rotation
psi_init_3   = 1/(np.sqrt(2))*(rb_atom_sim.kb_class.get_ket('g2PP',0,0)-rb_atom_sim.kb_class.get_ket('g2MM',0,0))
psi_des_3    = 1/(np.sqrt(2))*(rb_atom_sim.kb_class.get_ket('g2PP',0,0)-rb_atom_sim.kb_class.get_ket('g1M',0,0))
state_i_3    = "g2MM"
F_i_3        = 2
state_f_3    = "g1M"
F_f_3        = 1
state_x_3    = "x1M_d1"
F_x_3        = 1
delta_p_3    = (rb_atom_sim.get_splitting("deltaZx1M_d1")+2*rb_atom_sim.get_splitting("deltaZ"))
delta_s_3    = (-rb_atom_sim.get_splitting("deltaZ")+rb_atom_sim.get_splitting("deltaZx1M_d1"))
cg_pump_3    = "CG_d1g2MMx1M"
cg_stokes_3  = "CG_d1g1Mx1M"
pump_pol_3   = "sigmaP"
stokes_pol_3 = "pi"


rotation_dict_3 = {"psi_init":psi_init_3, "psi_des":psi_des_3, "state_i":state_i_3, "state_f":state_f_3,\
                   "state_x":state_x_3, "delta_p":delta_p_3, "delta_s":delta_s_3, "cg_pump":cg_pump_3,\
                    "cg_stokes":cg_stokes_3, "pump_pol":pump_pol_3, "stokes_pol":stokes_pol_3,
                    "F_x":F_x_3, "F_i":F_i_3, "F_f":F_f_3}


#fourth rotation
psi_init_4   = 1/(np.sqrt(2))*(rb_atom_sim.kb_class.get_ket('g2PP',0,0)-rb_atom_sim.kb_class.get_ket('g2MM',0,0))
psi_des_4    = 1/(np.sqrt(2))*(rb_atom_sim.kb_class.get_ket('g1P',0,0)-rb_atom_sim.kb_class.get_ket('g2MM',0,0))
state_i_4    = "g2PP"
F_i_4        = 2
state_f_4    = "g1P"
F_f_4        = 1
state_x_4    = "x1P_d1"
F_x_4        = 1
delta_p_4    = (rb_atom_sim.get_splitting("deltaZx1P_d1")-2*rb_atom_sim.get_splitting("deltaZ"))
delta_s_4    = (+rb_atom_sim.get_splitting("deltaZ")+rb_atom_sim.get_splitting("deltaZx1P_d1"))
cg_pump_4    = "CG_d1g2PPx1P"
cg_stokes_4  = "CG_d1g1Px1P"
pump_pol_4   = "sigmaM"
stokes_pol_4 = "pi"


rotation_dict_4 = {"psi_init":psi_init_4, "psi_des":psi_des_4, "state_i":state_i_4, "state_f":state_f_4,\
                   "state_x":state_x_4, "delta_p":delta_p_4, "delta_s":delta_s_4, "cg_pump":cg_pump_4,\
                    "cg_stokes":cg_stokes_4, "pump_pol":pump_pol_4, "stokes_pol":stokes_pol_4,
                    "F_x":F_x_4, "F_i":F_i_4, "F_f":F_f_4}


# params fixed for all runs
a = 1/np.sqrt(2)
b = 1/(2*np.pi)
lengthStirap = 0.5
OmegaStirap=60*2*np.pi
#delta_VST = -3*2*np.pi
delta_VST=0
deltaZ = rb_atom_sim.get_splitting("deltaZ")
deltaZx2MM = rb_atom_sim.get_splitting("deltaZx2MM")
deltaZx2PP = rb_atom_sim.get_splitting("deltaZx2PP")
delta_cav = 2*deltaZ+deltaZx2MM+delta_VST
diff_shift=DifferentialStarkShifts('d1', rb_atom_sim.rb_atom, rb_atom_sim.atom_states)

def run_rot_1_2(input_states, n_steps_rot, n_steps_vst):

    t_vst = np.linspace(0, lengthStirap, n_steps_vst)
    stirap_shape = "flattop_blackman"
    laser_pulse_vst = create_flattop_blackman(t_vst, 1, 0.17, 0.13)

    ########   #######  ########    ###    ######## ####  #######  ##    ##       ##   
    ##     ## ##     ##    ##      ## ##      ##     ##  ##     ## ###   ##     ####   
    ##     ## ##     ##    ##     ##   ##     ##     ##  ##     ## ####  ##       ##   
    ########  ##     ##    ##    ##     ##    ##     ##  ##     ## ## ## ##       ##   
    ##   ##   ##     ##    ##    #########    ##     ##  ##     ## ##  ####       ##   
    ##    ##  ##     ##    ##    ##     ##    ##     ##  ##     ## ##   ###       ##   
    ##     ##  #######     ##    ##     ##    ##    ####  #######  ##    ##     ######
    #convert optimisation parameters to variables

    tau_1 = first_params["param_1"]
    laser_amp_1 = first_params["laser_amplitude"]
    const_det_1 = first_params["detuning"]
    _length_repump_1 = first_params["duration"]
    detuning_magn_1 = first_params["detuning_magn"]

    psi_init_1 = input_states

    cg_pump_1 = rb_atom_sim.get_CG("CG_d1g2x1M")
    cg_stokes_1 = rb_atom_sim.get_CG("CG_d1g1Mx1M")

    stokes_amp_1 = laser_amp_1/cg_stokes_1
    pump_amp_1 = laser_amp_1/cg_pump_1
    t_1 = np.linspace(0,_length_repump_1,n_steps_rot)
    t_diff_1 = np.linspace(0, _length_repump_1)

    # generate stokes and pump pulses WITH RADIAL FREQUENCY AS UNITS FOR AMPLITUDE
    pump_pulse_1, stokes_pulse_1 = create_fstirap(t_1,tau_1,pump_amp_1,stokes_amp_1)
    pump_pulse_diff_1, stokes_pulse_diff_1 = create_fstirap(t_diff_1, tau_1, pump_amp_1, stokes_amp_1)

    #calculate shifts from the stokes and pump laser pulses
    shift_dict_stokes_1=diff_shift.calculate_td_detuning(F_f_1,b*stokes_pulse_diff_1, const_det_1,stokes_pol_1)
    shift_dict_pump_1=diff_shift.calculate_td_detuning(F_i_1,b*pump_pulse_diff_1, const_det_1,pump_pol_1)
    init_shift_1=diff_shift.find_state_evolution(b*pump_pulse_diff_1, shift_dict_pump_1, state_i_1)
    x_shift_p_1 = diff_shift.find_state_evolution(b*pump_pulse_diff_1, shift_dict_pump_1, state_x_1)
    fin_shift_1 = diff_shift.find_state_evolution(b*stokes_pulse_diff_1, shift_dict_stokes_1, state_f_1)
    x_shift_s_1 = diff_shift.find_state_evolution(b*stokes_pulse_diff_1, shift_dict_stokes_1, state_x_1)
    x_shift_tot_1 = x_shift_p_1 + x_shift_s_1

    #calculate time varying detuning from the shifts of the levels
    pump_det_spline_1 = interpolate.CubicSpline(t_diff_1, (x_shift_tot_1-init_shift_1)*detuning_magn_1*2*np.pi)
    stokes_det_spline_1 = interpolate.CubicSpline(t_diff_1, (x_shift_tot_1-fin_shift_1)*detuning_magn_1*2*np.pi)

    _pump_det_1 = pump_det_spline_1(t_1)
    _stokes_det_1 = stokes_det_spline_1(t_1)


    #run the simulation to find the final state density matrix
    start_time=time.time()
    (output_states_list_1, t_list_1) = rb_atom_sim.run_repreparation(const_det_1, t_1,
                                                delta_p_1, delta_s_1, pump_pol_1, stokes_pol_1, pump_pulse_1, 
                                                stokes_pulse_1, psi_init_1, F_i_1, F_x_1, F_f_1, F_x_1,
                                                pump_det=_pump_det_1, stokes_det=_stokes_det_1,
                                                raman_pulses=False)
    print("First rotation run in: ", time.time()-start_time)
    output_states_list_full.append(output_states_list_1)
    t_list_full.append(t_list_1)
    psi_fin_1 = output_states_list_1[-1]


    ##     ##  ######  ######## #### ########     ###    ########        ##   
    ##     ## ##    ##    ##     ##  ##     ##   ## ##   ##     ##     ####   
    ##     ## ##          ##     ##  ##     ##  ##   ##  ##     ##       ##   
    ##     ##  ######     ##     ##  ########  ##     ## ########        ##   
    ##   ##        ##    ##     ##  ##   ##   ######### ##              ##   
    ## ##   ##    ##    ##     ##  ##    ##  ##     ## ##              ##   
    ###     ######     ##    #### ##     ## ##     ## ##            ######



    psi_init_v1 = psi_fin_1

    t_vst = np.linspace(0, lengthStirap, n_steps_vst)

    delta_laser_v1 = -deltaZ+deltaZx2MM+delta_VST

    start_time = time.time()
    (output_states_list_v1, t_list_v1) = rb_atom_sim.run_vstirap(lengthStirap, OmegaStirap, delta_laser_v1, "sigmaM",
                                                                delta_cav, psi_init_v1, laser_pulse_vst, F_start=1, F_exc=2, F_end=2)
    print("First VSTIRAP run in seconds: ", time.time()-start_time)
    output_states_list_full.append(output_states_list_v1)
    t_list_full.append(t_list_v1)
    psi_fin_v1 = output_states_list_v1[-1]

    emission_list.append([cav_emission_from_state(_ground_states, 3, output_states_list_v1, t_list_v1, rb_atom_sim.kappa),
                           plotter_cavemission(rb_atom_sim.ketbras,_ground_states, output_states_list_v1, t_list_v1, rb_atom_sim.kappa, show_plts=False)])

    ########   #######  ########    ###    ######## ####  #######  ##    ##     #######  
    ##     ## ##     ##    ##      ## ##      ##     ##  ##     ## ###   ##    ##     ## 
    ##     ## ##     ##    ##     ##   ##     ##     ##  ##     ## ####  ##           ## 
    ########  ##     ##    ##    ##     ##    ##     ##  ##     ## ## ## ##     #######  
    ##   ##   ##     ##    ##    #########    ##     ##  ##     ## ##  ####    ##        
    ##    ##  ##     ##    ##    ##     ##    ##     ##  ##     ## ##   ###    ##        
    ##     ##  #######     ##    ##     ##    ##    ####  #######  ##    ##    #########
    #convert optimisation parameters to variables
    _a_2 = second_params["param_1"]
    _n_2 = second_params["_n"]
    _c_2 = second_params["_c"]
    laser_amp_2 = second_params["laser_amplitude"]
    const_det_2 = second_params["detuning"]
    _length_repump_2 = second_params["duration"]
    detuning_magn_2 = second_params["detuning_magn"]
    psi_init_2 = psi_fin_v1 #      CHANGE INITIAL STATE FOR THIS ROTATION HERE!

    cg_pump_2 = rb_atom_sim.get_CG("CG_d1g2x1")
    cg_stokes_2 = rb_atom_sim.get_CG("CG_d1g1Px1")
    stokes_amp_2 = laser_amp_2/cg_stokes_2
    pump_amp_2 = laser_amp_2/cg_pump_2
    t_2 = np.linspace(0,_length_repump_2,n_steps_rot)
    t_diff_2 = np.linspace(0, _length_repump_2)

    # generate stokes and pump pulses WITH RADIAL FREQUENCY AS UNITS FOR AMPLITUDE
    pump_pulse_2, stokes_pulse_2 = create_masked(t_2, pump_amp_2, stokes_amp_2, _a_2, n=_n_2, c=_c_2)
    pump_pulse_diff_2, stokes_pulse_diff_2 = create_masked(t_diff_2, pump_amp_2, stokes_amp_2, _a_2, n=_n_2, c=_c_2)

    #calculate shifts from the stokes and pump laser pulses
    shift_dict_stokes_2=diff_shift.calculate_td_detuning(F_f_2,b*stokes_pulse_diff_2, const_det_2,stokes_pol_2)
    shift_dict_pump_2=diff_shift.calculate_td_detuning(F_i_2,b*pump_pulse_diff_2, const_det_2,pump_pol_2)
    init_shift_2=diff_shift.find_state_evolution(b*pump_pulse_diff_2, shift_dict_pump_2, state_i_2)
    x_shift_p_2 = diff_shift.find_state_evolution(b*pump_pulse_diff_2, shift_dict_pump_2, state_x_2)
    fin_shift_2 = diff_shift.find_state_evolution(b*stokes_pulse_diff_2, shift_dict_stokes_2, state_f_2)
    x_shift_s_2 = diff_shift.find_state_evolution(b*stokes_pulse_diff_2, shift_dict_stokes_2, state_x_2)
    x_shift_tot_2 = x_shift_p_2 + x_shift_s_2

    #calculate time varying detuning from the shifts of the levels
    pump_det_spline = interpolate.CubicSpline(t_diff_2, (x_shift_tot_2-init_shift_2)*detuning_magn_2*2*np.pi)
    stokes_det_spline = interpolate.CubicSpline(t_diff_2, (x_shift_tot_2-fin_shift_2)*detuning_magn_2*2*np.pi)

    _pump_det = pump_det_spline(t_2)
    _stokes_det = stokes_det_spline(t_2)

    #run the simulation to find the final state density matrix
    start_time=time.time()
    (output_states_list_2, t_list_2) = rb_atom_sim.run_repreparation(const_det_2, t_2,
                                                delta_p_2,delta_s_2, pump_pol_2, stokes_pol_2, pump_pulse_2, 
                                                stokes_pulse_2, psi_init_2, F_i_2, F_x_2, F_f_2, F_x_2,
                                                pump_det=_pump_det, stokes_det=_stokes_det,
                                                raman_pulses=True)
    print("Second rotation run in: ", time.time()-start_time)
    
    output_states_list_full.append(output_states_list_2)
    t_list_full.append(t_list_2)

    psi_fin_2 = output_states_list_2[-1]

    ##     ##  ######  ######## #### ########     ###    ########      #######  
    ##     ## ##    ##    ##     ##  ##     ##   ## ##   ##     ##    ##     ## 
    ##     ## ##          ##     ##  ##     ##  ##   ##  ##     ##           ## 
    ##     ##  ######     ##     ##  ########  ##     ## ########      #######  
    ##   ##        ##    ##     ##  ##   ##   ######### ##           ##        
    ## ##   ##    ##    ##     ##  ##    ##  ##     ## ##           ##        
    ###     ######     ##    #### ##     ## ##     ## ##           #########
    psi_init_v2 = psi_fin_2

    t_vst2 = np.linspace(0, lengthStirap, n_steps_rot)

    delta_laser_v2=deltaZ+deltaZx2PP+delta_VST

    start_time = time.time()
    (output_states_list_v2, t_list_v2) = rb_atom_sim.run_vstirap(lengthStirap, OmegaStirap, delta_laser_v2, "sigmaP",
                                                                delta_cav, psi_init_v2, laser_pulse_vst, F_start=1, F_exc=2, F_end=2)
    print("Second VSTIRAP run in seconds: ", time.time()-start_time)
    psi_fin_v2 = output_states_list_v2[-1]
    output_states_list_full.append(output_states_list_v2)
    t_list_full.append(t_list_v2)

    emission_list.append([cav_emission_from_state(_ground_states, 7, output_states_list_v2, t_list_v2, rb_atom_sim.kappa),
                           plotter_cavemission(rb_atom_sim.ketbras,_ground_states, output_states_list_v2, t_list_v2, rb_atom_sim.kappa, show_plts=False)])

    return psi_fin_v2

def run_rot_3_4(input_states, n_steps_rot, n_steps_vst, n_times):

    t_vst = np.linspace(0, lengthStirap, n_steps_vst)
    stirap_shape = "flattop_blackman"
    laser_pulse_vst = create_flattop_blackman(t_vst, 1, 0.17, 0.13)

    ########   #######  ########    ###    ######## ####  #######  ##    ##     #######  
    ##     ## ##     ##    ##      ## ##      ##     ##  ##     ## ###   ##    ##     ## 
    ##     ## ##     ##    ##     ##   ##     ##     ##  ##     ## ####  ##           ## 
    ########  ##     ##    ##    ##     ##    ##     ##  ##     ## ## ## ##     #######  
    ##   ##   ##     ##    ##    #########    ##     ##  ##     ## ##  ####           ## 
    ##    ##  ##     ##    ##    ##     ##    ##     ##  ##     ## ##   ###    ##     ## 
    ##     ##  #######     ##    ##     ##    ##    ####  #######  ##    ##     ####### 
    #convert optimisation parameters to variables
    _a_3 = third_params["param_1"]
    _n_3 = third_params["_n"]
    _c_3 = third_params["_c"]
    laser_amp_3 = third_params["laser_amplitude"]
    const_det_3 = third_params["detuning"]
    _length_repump_3 = third_params["duration"]
    detuning_magn_3 = third_params["detuning_magn"]

    # select the correct rotation details
    psi_init_3 = input_states #      CHANGE INITIAL STATE FOR THIS ROTATION HERE!

    stokes_amp_3 = laser_amp_3/rb_atom_sim.get_CG("CG_d1g1Mx1M")
    pump_amp_3 = laser_amp_3/rb_atom_sim.get_CG("CG_d1g2MMx1M")
    t_3 = np.linspace(0,_length_repump_3,n_steps_rot)
    t_diff_3 = np.linspace(0, _length_repump_3)

    # generate stokes and pump pulses WITH RADIAL FREQUENCY AS UNITS FOR AMPLITUDE
    pump_pulse_3, stokes_pulse_3 = create_masked(t_3, pump_amp_3, stokes_amp_3, _a_3, n=_n_3, c=_c_3)
    pump_pulse_diff_3, stokes_pulse_diff_3 = create_masked(t_diff_3, pump_amp_3, stokes_amp_3, _a_3, n=_n_3, c=_c_3)

    #calculate shifts from the stokes and pump laser pulses
    shift_dict_stokes_3=diff_shift.calculate_td_detuning(F_f_3,b*stokes_pulse_diff_3, const_det_3,stokes_pol_3)
    shift_dict_pump_3=diff_shift.calculate_td_detuning(F_i_3,b*pump_pulse_diff_3, const_det_3,pump_pol_3)
    init_shift_3=diff_shift.find_state_evolution(b*pump_pulse_diff_3, shift_dict_pump_3, state_i_3)
    x_shift_p_3 = diff_shift.find_state_evolution(b*pump_pulse_diff_3, shift_dict_pump_3, state_x_3)
    fin_shift_3 = diff_shift.find_state_evolution(b*stokes_pulse_diff_3, shift_dict_stokes_3, state_f_3)
    x_shift_s_3 = diff_shift.find_state_evolution(b*stokes_pulse_diff_3, shift_dict_stokes_3, state_x_3)
    x_shift_tot_3 = x_shift_p_3 + x_shift_s_3

    #calculate time varying detuning from the shifts of the levels
    pump_det_spline = interpolate.CubicSpline(t_diff_3, (x_shift_tot_3-init_shift_3)*detuning_magn_3*2*np.pi)
    stokes_det_spline = interpolate.CubicSpline(t_diff_3, (x_shift_tot_3-fin_shift_3)*detuning_magn_3*2*np.pi)

    _pump_det_3 = pump_det_spline(t_3)
    _stokes_det_3 = stokes_det_spline(t_3)

    #run the simulation to find the final state density matrix
    start_time=time.time()
    (output_states_list_3, t_list_3) = rb_atom_sim.run_repreparation(const_det_3, t_3,
                                                delta_p_3, delta_s_3, pump_pol_3, stokes_pol_3, pump_pulse_3, 
                                                stokes_pulse_3, psi_init_3, F_i_3, F_x_3, F_f_3, F_x_3,
                                                pump_det=_pump_det_3, stokes_det=_stokes_det_3,
                                                raman_pulses=True)
    print("Third rotation run in: ", time.time()-start_time)
    output_states_list_full.append(output_states_list_3)
    t_list_full.append(t_list_3)

    psi_fin_3 = output_states_list_3[-1]


    #  ##     ##  ######  ######## #### ########     ###    ########      #######  
    #  ##     ## ##    ##    ##     ##  ##     ##   ## ##   ##     ##    ##     ## 
    #  ##     ## ##          ##     ##  ##     ##  ##   ##  ##     ##           ## 
    #  ##     ##  ######     ##     ##  ########  ##     ## ########      #######  
    #   ##   ##        ##    ##     ##  ##   ##   ######### ##                  ## 
    #    ## ##   ##    ##    ##     ##  ##    ##  ##     ## ##           ##     ## 
    #     ###     ######     ##    #### ##     ## ##     ## ##            #######  
    psi_init_v3 = psi_fin_3

    t_vst3 = np.linspace(0, lengthStirap, n_steps_vst)

    delta_laser_v3=-deltaZ+deltaZx2MM+delta_VST

    laser_pulse_vst = create_single_sinsquared(t_vst3, 1)
    start_time = time.time()
    (output_states_list_v3, t_list_v3) = rb_atom_sim.run_vstirap(lengthStirap, OmegaStirap, delta_laser_v3, "sigmaM",
                                                                delta_cav, psi_init_v3, laser_pulse_vst, F_start=1, F_exc=2, F_end=2)
    print("Third VSTIRAP run in seconds: ", time.time()-start_time)
    output_states_list_full.append(output_states_list_v3)
    t_list_full.append(t_list_v3)
    psi_fin_v3 = output_states_list_v3[-1]

    emission_list.append([cav_emission_from_state(_ground_states, 3, output_states_list_v3, t_list_v3, rb_atom_sim.kappa),
                           plotter_cavemission(rb_atom_sim.ketbras,_ground_states, output_states_list_v3, t_list_v3, rb_atom_sim.kappa, show_plts=False)])

    #  ########   #######  ########    ###    ######## ####  #######  ##    ##    ##        
    #  ##     ## ##     ##    ##      ## ##      ##     ##  ##     ## ###   ##    ##    ##  
    #  ##     ## ##     ##    ##     ##   ##     ##     ##  ##     ## ####  ##    ##    ##  
    #  ########  ##     ##    ##    ##     ##    ##     ##  ##     ## ## ## ##    ##    ##  
    #  ##   ##   ##     ##    ##    #########    ##     ##  ##     ## ##  ####    ######### 
    #  ##    ##  ##     ##    ##    ##     ##    ##     ##  ##     ## ##   ###          ##  
    #  ##     ##  #######     ##    ##     ##    ##    ####  #######  ##    ##          ##  
    #convert optimisation parameters to variables
    _a_4 = fourth_params["param_1"]
    _n_4 = fourth_params["_n"]
    _c_4 = fourth_params["_c"]
    laser_amp_4 = fourth_params["laser_amplitude"]
    const_det_4 = fourth_params["detuning"]
    _length_repump_4 = fourth_params["duration"]
    detuning_magn_4 = fourth_params["detuning_magn"]

    psi_init_4 = psi_fin_v3 #      CHANGE INITIAL STATE FOR THIS ROTATION HERE!    
    stokes_amp_4 = laser_amp_4/rb_atom_sim.get_CG("CG_d1g1Px1P")
    pump_amp_4 = laser_amp_4/rb_atom_sim.get_CG("CG_d1g2PPx1P")
    t_4 = np.linspace(0,_length_repump_4,n_steps_rot)
    t_diff_4 = np.linspace(0, _length_repump_4)

    # generate stokes and pump pulses WITH RADIAL FREQUENCY AS UNITS FOR AMPLITUDE
    pump_pulse_4, stokes_pulse_4 = create_masked(t_4, pump_amp_4, stokes_amp_4, _a_4, n=_n_4, c=_c_4)
    pump_pulse_diff_4, stokes_pulse_diff_4 = create_masked(t_diff_4, pump_amp_4, stokes_amp_4, _a_4, n=_n_4, c=_c_4)

    #calculate shifts from the stokes and pump laser pulses
    shift_dict_stokes_4=diff_shift.calculate_td_detuning(F_f_4,b*stokes_pulse_diff_4, const_det_4,pump_pol_4)
    shift_dict_pump_4=diff_shift.calculate_td_detuning(F_i_4,b*pump_pulse_diff_4, const_det_4,pump_pol_4)
    init_shift_4=diff_shift.find_state_evolution(b*pump_pulse_diff_4, shift_dict_pump_4, state_i_4)
    x_shift_p_4 = diff_shift.find_state_evolution(b*pump_pulse_diff_4, shift_dict_pump_4, state_x_4)
    fin_shift_4 = diff_shift.find_state_evolution(b*stokes_pulse_diff_4, shift_dict_stokes_4, state_f_4)
    x_shift_s_4 = diff_shift.find_state_evolution(b*stokes_pulse_diff_4, shift_dict_stokes_4, state_x_4)
    x_shift_tot_4 = x_shift_p_4 + x_shift_s_4

    #calculate time varying detuning from the shifts of the levels
    pump_det_spline = interpolate.CubicSpline(t_diff_4, (x_shift_tot_4-init_shift_4)*detuning_magn_4*2*np.pi)
    stokes_det_spline = interpolate.CubicSpline(t_diff_4, (x_shift_tot_4-fin_shift_4)*detuning_magn_4*2*np.pi)

    _pump_det_4 = pump_det_spline(t_4)
    _stokes_det_4 = stokes_det_spline(t_4)

    #run the simulation to find the final state density matrix
    start_time=time.time()
    (output_states_list_4, t_list_4) = rb_atom_sim.run_repreparation(const_det_4, t_4,
                                                delta_p_4, delta_s_4, pump_pol_4, stokes_pol_4, pump_pulse_4, 
                                                stokes_pulse_4, psi_init_4, F_i_4, F_x_4, F_f_4, F_x_4,
                                                pump_det=_pump_det_4, stokes_det=_stokes_det_4,
                                                raman_pulses=True)
    print("Fourth rotation run in: ", time.time()-start_time)
    output_states_list_full.append(output_states_list_4)
    t_list_full.append(t_list_4)

    psi_fin_4 = output_states_list_4[-1]

    psi_init_v4 = psi_fin_4

    t = np.linspace(0, lengthStirap, n_steps_vst)

    delta_laser_v4=deltaZ+deltaZx2PP+delta_VST
    start_time = time.time()
    (output_states_list_v4, t_list_v4) = rb_atom_sim.run_vstirap(lengthStirap, OmegaStirap, delta_laser_v4, "sigmaP",
                                                                delta_cav, psi_init_v4, laser_pulse_vst, F_start=1, F_exc=2, F_end=2)
    print("Fourth VSTIRAP run in seconds: ", time.time()-start_time)
    output_states_list_full.append(output_states_list_v4[-1])
    t_list_full.append(t_list_v4)
    psi_fin_v4 = output_states_list_v4[-1]

    emission_list.append([cav_emission_from_state(_ground_states, 7, output_states_list_v4, t_list_v4, rb_atom_sim.kappa),
                           plotter_cavemission(rb_atom_sim.ketbras,_ground_states, output_states_list_v4, t_list_v4, rb_atom_sim.kappa, show_plts=False)])

    if n_times==1:
        return psi_fin_v4
    else:
        psi_fin=psi_fin_v4
        counter=1
        def run_again(_input):
            start_time=time.time()
            (output_states_list_3, t_list_3) = rb_atom_sim.run_repreparation(const_det_3, t_3,
                                                delta_p_3, delta_s_3, pump_pol_3, stokes_pol_3, pump_pulse_3, 
                                                stokes_pulse_3, _input, F_i_3, F_x_3, F_f_3, F_x_3,
                                                pump_det=_pump_det_3, stokes_det=_stokes_det_3,
                                                raman_pulses=True)
            print("Third rotation run again in: ", time.time()-start_time)
            psi_fin_3 = output_states_list_3[-1]
            output_states_list_full.append(output_states_list_3)
            t_list_full.append(t_list_3)
            start_time = time.time()
            (output_states_list_v3, t_list_v3) = rb_atom_sim.run_vstirap(lengthStirap, OmegaStirap, delta_laser_v3, "sigmaM",
                                                                delta_cav, psi_fin_3, laser_pulse_vst, F_start=1, F_exc=2, F_end=2)
            print("Third VSTIRAP run again in: ", time.time()-start_time)
            output_states_list_full.append(output_states_list_v3)
            t_list_full.append(t_list_v3)
            psi_fin_v3 = output_states_list_v3[-1]
            emission_list.append([cav_emission_from_state(_ground_states, 3, output_states_list_v3, t_list_v3, rb_atom_sim.kappa),
                           plotter_cavemission(rb_atom_sim.ketbras,_ground_states, output_states_list_v3, t_list_v3, rb_atom_sim.kappa, show_plts=False)])
            start_time=time.time()
            (output_states_list_4, t_list_4) = rb_atom_sim.run_repreparation(const_det_4, t_4,
                                                delta_p_4, delta_s_4, pump_pol_4, stokes_pol_4, pump_pulse_4, 
                                                stokes_pulse_4, psi_fin_v3, F_i_4, F_x_4, F_f_4, F_x_4,
                                                pump_det=_pump_det_4, stokes_det=_stokes_det_4,
                                                raman_pulses=True)
            print("Fourth rotation run again in: ", time.time()-start_time)
            output_states_list_full.append(output_states_list_4)
            t_list_full.append(t_list_4)
            psi_fin_4 = output_states_list_4[-1]
            start_time = time.time()
            (output_states_list_v4, t_list_v4) = rb_atom_sim.run_vstirap(lengthStirap, OmegaStirap, delta_laser_v4, "sigmaP",
                                                                delta_cav, psi_fin_4, laser_pulse_vst, F_start=1, F_exc=2, F_end=2)
            print("Fourth VSTIRAP run again in: ", time.time()-start_time)
            output_states_list_full.append(output_states_list_v4)
            t_list_full.append(t_list_v4)
            psi_fin_return = output_states_list_v4[-1]
            emission_list.append([cav_emission_from_state(_ground_states, 7, output_states_list_v4, t_list_v4, rb_atom_sim.kappa),
                           plotter_cavemission(rb_atom_sim.ketbras,_ground_states, output_states_list_v4, t_list_v4, rb_atom_sim.kappa, show_plts=False)])

            return psi_fin_return
        
        while counter<n_times:
            psi_fin=run_again(psi_fin)
            counter+=1
        
        else:
            return psi_fin

if __name__ == "__main__":

    #run the first rotation
    psi_fin_1 = run_rot_1_2(psi_init_1, 3000, 1000)
    psi_fin_2 = run_rot_3_4(psi_fin_1, 3000, 1000, 9)
    
    #save the emission
    with open ("emission_list.pkl", "wb") as f:
        pickle.dump(emission_list, f)
    #save timelists individually and the concatenated one
    with open ("timelists.pkl", "wb") as f:
        pickle.dump(t_list_full, f)
    #save all desnity matrix evolution for detailed phase analysis
    with open ("density_matrix_evolution.pkl", "wb") as f:
        pickle.dump(output_states_list_full, f)

