import time
import pickle

import numpy as np

from qutip import mcsolve

from src.modules.atom_config import RbAtom
from src.modules.cavity import (
    cav_collapse_ops,
    quant_axis_cavbasis_mapping,
)
from src.modules.ketbra_config import RbKetBras

# List the groundstates to be included in the simulation

atomStates = {
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
    #'x1M_d1','x1_d1','x1P_d1',
    #'x2MM_d1','x2M_d1','x2_d1','x2P_d1','x2PP_d1'
]

# configure the atm state dictionary such that it includes the desired excited states
kb_class = RbKetBras(atomStates, xlvls, True)

# precompute ketbras for speed
ketbras = kb_class.getrb_ketbras()

# specify system b field groundstate splitting in MHz
bfieldsplit = "0p07"
rb_atom = RbAtom(bfieldsplit, kb_class)

# List the coupling rates of the system.
#   gamma:  Decay of the atomic amplitude listed for d1 and d2 transitions.
# d: Dipole moment of either d1 or d2 transition
[gamma_d2, gamma_d1, d_d2, d_d1] = rb_atom.getrb_rates()

# specify cavity parameters

#   kappa:  Decay of the electric field out of the cavity.
#   deltaP: polarisation splitting, i..e birefringence of the cavity
kappa = 2.1 * 2.0 * np.pi
deltaP = 0 * 2.0 * np.pi

# configure the atom-cavity coupling strength, for a 10 MHz coupling on your transition of choice --> coupling_factor=10*2*np.pi/cav_transition,
# By default we are considering the D2 line for photon production with an optical cavity, but simply use a configured CG coefficient for D1 if you want to consider this instead, e.g. CG_d1g1Mx1
# pick the Clebsch Gordan coefficient of the desired cavity transition e.g. CGg1Mx1, g1M <-> ground state F=1, m_F=-1, x1 <-> excited state F'=1 m_F'=0

desired_vst_line = "d2"
norm = abs(rb_atom.CGg1Mx1)
coupling_factor = 11.1 * 2 * np.pi / (norm)
cav_transition = abs(rb_atom.CGg2MMx2MM)

if desired_vst_line == "d2":
    coop = (cav_transition * coupling_factor) ** 2 / (2 * kappa * gamma_d2)
else:
    coop = (cav_transition * coupling_factor) ** 2 / (2 * kappa * gamma_d1)

"""
DEFINE QUANTISATION AXIS and Cavity Axis as 3-d vectors to define polarisation basis supported by the cavity
"""
cav_axis = [1, 0, 0]
quant_axis = [0, 1, 0]

# be very careful with the sign of delta and the tuning of the VSTIRAP resonance, deltaZx.. has plus minus signs, deltaZ is always positive
# recomenned drawing an energy level diagram to check the signs
delta_VST = -3 * 2 * np.pi
delta_cav = 2 * rb_atom.deltaZ + rb_atom.deltaZx2MM + delta_VST

delta_laser_1 = -rb_atom.deltaZ + rb_atom.deltaZx2MM + delta_VST
delta_laser_2 = rb_atom.deltaZ + rb_atom.deltaZx2PP + delta_VST

g_mhz = np.round(coupling_factor, 3)

F_vst_start = 1
F_vst_final = 2
F_vst_exc = 2
n_time_steps = 1000

# define list of collapse operators
c_op_list = []
c_op_list += cav_collapse_ops(kappa, atomStates)

# by default we are adding the collapse operators for both the d2 and d1 line, but comment out either if only one is desired
c_op_list += rb_atom.spont_em_ops(atomStates)[0]  # d2 line
# c_op_list+=rb_atom.spont_em_ops(atomStates)[1] #d1 line

_n_traj = 1000
length_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# length_list=[1.5,2]
# area_list=[ 25, 50, 75]
area_list = [75]
for area in area_list:
    area *= 2 * np.pi
    saved_processed_data = []
    for pulse_len in length_list:
        OmegaStirap = area / pulse_len

        wStirap = np.pi / pulse_len

        def amp_shape(t):
            return np.sin(wStirap * t) ** 2

        time_list = np.linspace(0, pulse_len, n_time_steps)
        vst_driving_array = np.array([amp_shape(t) for t in time_list])
        vst_phase_array = np.array([0 for t in time_list], dtype=float)

        [H_VStirap_1, args_hams_VStirap_1] = rb_atom.gen_H_VSTIRAP_D2(
            ketbras,
            atomStates,
            delta_cav,
            delta_laser_1,
            F_vst_start,
            F_vst_final,
            F_vst_exc,
            "sigmaM",
            OmegaStirap,
            coupling_factor,
            deltaP,
            quant_axis_cavbasis_mapping(quant_axis, cav_axis),
            {},
            "",
            True,
            vst_driving_array,
            time_list,
            vst_phase_array,
        )
        [H_VStirap_2, args_hams_VStirap_2] = rb_atom.gen_H_VSTIRAP_D2(
            ketbras,
            atomStates,
            delta_cav,
            delta_laser_2,
            F_vst_start,
            F_vst_final,
            F_vst_exc,
            "sigmaP",
            OmegaStirap,
            coupling_factor,
            deltaP,
            quant_axis_cavbasis_mapping(quant_axis, cav_axis),
            {},
            "",
            True,
            vst_driving_array,
            time_list,
            vst_phase_array,
        )

        # define length of each Hamiltonian, it might be interesting to pick a length longer than the individual pulse duration to consider the effect of non-unitary losses
        tVStirap, tVStirapStep = np.linspace(0, pulse_len, n_time_steps, retstep=True)

        # Initial state of the system in array format with dictionary, atomic state and cavity fock states (atomStates, 'atomic state', N_cavx, N_cavy]
        psi0_list = [
            1
            / (np.sqrt(2))
            * (kb_class.get_ket("g2PP", 0, 0) - kb_class.get_ket("g1M", 0, 0)),
            1
            / (np.sqrt(2))
            * (kb_class.get_ket("g2MM", 0, 0) - kb_class.get_ket("g1P", 0, 0)),
        ]

        H_list = [
            (H_VStirap_1, tVStirap, {}, "H_VStirap_1"),
            (H_VStirap_2, tVStirap, {}, "H_VStirap_2"),
        ]

        # we return the density operators after each simulation run for each timestep in the simulation and save it in a list called output states list
        ind, n_hams = 0, len(H_list)
        output_states_list = []
        t_list = []
        for H, t, args, label in H_list:

            t_start = time.time()

            output = mcsolve(
                H, psi0_list[ind], t, c_op_list, [], args=args, ntraj=_n_traj
            )

            print(
                "Simulation {0}/{1} with {2} timesteps completed in {3} seconds".format(
                    ind, n_hams, t.size, np.round(time.time() - t_start, 3)
                )
            )

            mc_ctimes = output.col_times
            mc_cops = output.col_which
            scattering_t = 0
            emissions_x = 0
            emissions_y = 0
            false_ems = 0

            for i, el in enumerate(mc_cops):
                if len(el) == 1:
                    if el[0] == 0:
                        emissions_x += 1
                    elif el[0] == 1:
                        emissions_y += 1

            for i, el in enumerate(mc_ctimes):
                if len(el) >= 2:
                    if mc_cops[i][-1] == 0 and mc_cops[i][0] > 1:
                        false_ems += 1

            for i in mc_cops:
                for index in i:
                    if index > 1:
                        scattering_t += 1

            print("False emission events: {0}".format(false_ems))
            print(
                "Emission in x: {0}, Emission in y: {1}".format(
                    emissions_x, emissions_y
                )
            )
            print("Total spontaneous emission events: {0}".format(scattering_t))
            saved_processed_data.append(
                [
                    label,
                    pulse_len,
                    OmegaStirap,
                    scattering_t / _n_traj,
                    false_ems / _n_traj,
                    emissions_x / _n_traj,
                    emissions_y / _n_traj,
                ]
            )

            if ind == 0:
                ind = 1
            else:
                ind = 0

    # Save the output states list
    with open(
        f"/Users/ernst/Desktop/rb_photon_prod_dev/saved_data/vstirap_length_rate/mc/data_lensweep_omegaL:{np.round(area/(2*np.pi),2)}.pkl",
        "wb",
    ) as f:
        pickle.dump(saved_processed_data, f)
