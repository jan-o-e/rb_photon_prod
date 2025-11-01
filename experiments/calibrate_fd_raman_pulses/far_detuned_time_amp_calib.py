
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import numpy as np
from qutip import Qobj, mesolve, fidelity
from scipy.optimize import minimize

from src.modules.atom_config import RbAtom
from src.modules.ketbra_config import RbKetBras
from src.modules.laser_pulses import create_flattop_blackman

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
    #'x0',
    #'x1M','x1','x1P',
    #'x2MM','x2M','x2','x2P','x2PP',
    #'x3MMM', 'x3MM','x3M','x3','x3P','x3PP', 'x3PPP',
    #'x1M_d1','x1_d1','x1P_d1',
    #'x2MM_d1','x2M_d1','x2_d1','x2P_d1','x2PP_d1'
]

kb_class = RbKetBras(atomStates, xlvls, False)
ketbras = kb_class.getrb_ketbras()

# specify system b field groundstate splitting in MHz
bfieldsplit = "0p07"
rb_atom = RbAtom(bfieldsplit, kb_class)

# Pulse params
det_centre = -100000
rise_time = 0.025
# amp_scaling = 1.466
amp_scaling = 2
number_of_rabi_cycles = 1
# pulse_length = 1 / amp_scaling**2 * number_of_rabi_cycles
# PULSE LENGTH FIXED FOR GRADIENT OPTIMIZATION
pulse_length = 0.125


rotation_configs = {
    1: {
        "cg_1": rb_atom.CG_d1g2x1,
        "pol_1": "pi",
        "det_zeeman_1": 0,
        "cg_2": rb_atom.CG_d1g1Mx1,
        "pol_2": "sigmaP",
        "det_zeeman_2": -rb_atom.deltaZ,
        "psi_0": kb_class.get_ket_atomic("g2"),
        "psi_des": 1
        / np.sqrt(2)
        * (kb_class.get_ket_atomic("g1M") - kb_class.get_ket_atomic("g2")),
        "coherence_indices": [0, 5],
        "det_centre": det_centre,
        "rise_time": rise_time,
        "amp_scaling": amp_scaling,
        "pulse_length": pulse_length,
        "rel_phase": 0,
    },
    2: {
        "cg_1": rb_atom.CG_d1g2x1,
        "pol_1": "pi",
        "det_zeeman_1": 0,
        "cg_2": rb_atom.CG_d1g1Px1,
        "pol_2": "sigmaM",
        "det_zeeman_2": rb_atom.deltaZ,
        "psi_0": 1
        / np.sqrt(2)
        * (kb_class.get_ket_atomic("g2") - kb_class.get_ket_atomic("g2MM")),
        "psi_des": 1
        / np.sqrt(2)
        * (kb_class.get_ket_atomic("g1P") - kb_class.get_ket_atomic("g2MM")),
        "coherence_indices": [2, 3],
        "det_centre": det_centre,
        "rise_time": rise_time,
        "amp_scaling": amp_scaling,
        "pulse_length": pulse_length,
        "rel_phase": 0,
    },
    3: {
        "cg_1": rb_atom.CG_d1g2MMx2MM,
        "pol_1": "pi",
        "det_zeeman_1": 0,
        "cg_2": rb_atom.CG_d1g1Mx2MM,
        "pol_2": "sigmaM",
        "det_zeeman_2": -np.pi * rb_atom.deltaZ,
        "psi_0": 1
        / np.sqrt(2)
        * (kb_class.get_ket_atomic("g2MM") - kb_class.get_ket_atomic("g2PP")),
        "psi_des": 1
        / np.sqrt(2)
        * (kb_class.get_ket_atomic("g1M") - kb_class.get_ket_atomic("g2PP")),
        "coherence_indices": [0, 7],
        "det_centre": det_centre,
        "rise_time": rise_time,
        "amp_scaling": amp_scaling,
        "pulse_length": pulse_length,
        "rel_phase": 0,
    },
    4: {
        "cg_1": rb_atom.CG_d1g2PPx2PP,
        "pol_1": "pi",
        "det_zeeman_1": 0,
        "cg_2": rb_atom.CG_d1g1Px2PP,
        "pol_2": "sigmaP",
        "det_zeeman_2": np.pi * rb_atom.deltaZ,
        "psi_0": 1
        / np.sqrt(2)
        * (kb_class.get_ket_atomic("g2MM") - kb_class.get_ket_atomic("g2PP")),
        "psi_des": 1
        / np.sqrt(2)
        * (kb_class.get_ket_atomic("g1P") - kb_class.get_ket_atomic("g2MM")),
        "coherence_indices": [2, 3],
        "det_centre": det_centre,
        "rise_time": rise_time,
        "amp_scaling": amp_scaling,
        "pulse_length": pulse_length,
        "rel_phase": 0,
    },
}

# Global parameters
rotation_number = 2

# Select the correct configuration
if rotation_number not in rotation_configs:
    raise ValueError("Rotation number not recognised")

rotation_config = rotation_configs[rotation_number]


def run_mesolve(config):
    """
    Runs the mesolve simulation with the given configuration.
    """
    # Define the laser pulse parameters
    two_phot_det = 0.0
    pulse_time = np.linspace(0, config["pulse_length"], 5000)

    pulse_1 = create_flattop_blackman(pulse_time, 1, config["rise_time"])
    det_1 = (config["det_centre"] * 2 * np.pi) + config["det_zeeman_1"]
    amp_1 = (
        config["amp_scaling"]
        * np.sqrt(2 * np.abs(config["det_centre"]))
        * 2
        * np.pi
        / config["cg_1"]
    )

    pulse_2 = create_flattop_blackman(pulse_time, 1, config["rise_time"])
    det_2 = (
        det_1
        - two_phot_det * 2 * np.pi
        - rb_atom.getrb_gs_splitting()
        + config["det_zeeman_2"]
    )
    amp_2 = (
        config["amp_scaling"]
        * np.sqrt(2 * np.abs(config["det_centre"]))
        * 2
        * np.pi
        / config["cg_2"]
    )

    ham, args = rb_atom.gen_H_FarDetuned_Raman_PulsePair_D1(
        ketbras,
        atomStates,
        det_1,
        det_2,
        config["pol_1"],
        config["pol_2"],
        amp_1,
        amp_2,
        pulse_time,
        pulse_1,
        pulse_2,
    )

    # Add decay operators
    c_op_list = []
    c_op_list += rb_atom.spont_em_ops_far_detuned(
        atomStates, config["pol_1"], amp_1, det_1
    )
    c_op_list += rb_atom.spont_em_ops_far_detuned(
        atomStates, config["pol_2"], amp_2, det_2
    )

    return mesolve(ham, config["psi_0"], pulse_time, c_op_list)


def find_fidelity(output, config):
    """
    Finds the fidelity between the output states and psi_des up to a phase shift.

    Parameters:
        output (QobjEvo or Result): The quantum simulation output from QuTiP.
        config (dict): Configuration dictionary containing "psi_des".

    Returns:
        float: The fidelity between the output states and psi_des.
    """
    # Ensure psi_des is converted correctly
    psi_des_abs = Qobj(np.abs(config["psi_des"].full()))

    return fidelity(Qobj(np.abs(output.states[-1].full())), psi_des_abs)


def find_max_fidelity_in_time(output, config):
    """
    Finds the index and time where the fidelity between output states and psi_des is maximized.

    Parameters:
        output (QobjEvo or Result): The quantum simulation output from QuTiP.
        config (dict): Configuration dictionary containing "psi_des" and "pulse_length".

    Returns:
        tuple: (max_fidelity_index, max_fidelity_time, max_fidelity_value)
    """
    pulse_time = np.linspace(0, config["pulse_length"], len(output.states))

    # Ensure psi_des is also converted correctly
    psi_des_abs = Qobj(np.abs(config["psi_des"].full()))

    fidelities = [
        fidelity(Qobj(np.abs(state.full())), psi_des_abs) for state in output.states
    ]

    max_fidelity_index = np.argmax(fidelities)
    max_fidelity_time = pulse_time[max_fidelity_index]
    max_fidelity_value = fidelities[max_fidelity_index]

    return max_fidelity_index, max_fidelity_time, max_fidelity_value


# max_index, max_time, max_fid= find_max_fidelity_in_time(run_mesolve(rotation_config), rotation_config)
# print(f"Max fidelity: {max_fid} at time {max_time} s")
# THE PROBLEM WITH THIS IS THAT IT RETURNS MAX FID WITHOUT ANY PULSE FALL TIME; SO THE TIME CANNOT BE EXTRACTED FOR USING AS A PULSE LENGTH PARAMETER IN THE NEXT RUN


def optimize_fidelity_gradient(amp_min, amp_max, rotation_config):
    """
    Gradient-based optimization using L-BFGS-B to maximize fidelity.
    Optimizes only amp_scaling while keeping det_zeeman_2 fixed.
    """

    def fidelity_objective(amp_scaling):
        rotation_config["amp_scaling"] = np.float64(amp_scaling)

        result = run_mesolve(rotation_config)
        fid = find_fidelity(result, rotation_config)

        print(
            f"Gradient optimizer Intermediate result: amp_scaling={amp_scaling}, fidelity={fid}"
        )

        return -fid  # Minimizing negative fidelity to maximize fidelity

    initial_guess = [2.2]
    bounds = [(amp_min, amp_max)]

    opt_result = minimize(
        fidelity_objective, x0=initial_guess, bounds=bounds, method="L-BFGS-B"
    )

    return {"optimal_amp": opt_result.x[0], "optimal_fidelity": -opt_result.fun}


# Example usage
amp_min = 1
amp_max = 15

gradient_optimal_values = optimize_fidelity_gradient(amp_min, amp_max, rotation_config)
print(
    f"Gradient Optimization Results for rot{rotation_number}:", gradient_optimal_values
)


'''
def optimize_fidelity_global(amp_min, amp_max, lower_det_bound, upper_det_bound, rotation_config):
    """
    Global optimization using differential evolution to maximize fidelity.
    """
    
    def fidelity_objective(params):
        amp_scaling, det_zeeman_2 = params
        rotation_config["amp_scaling"] = amp_scaling
        rotation_config["det_zeeman_2"] = det_zeeman_2
        
        result = run_mesolve(rotation_config)
        fid = find_fidelity(result, rotation_config)
        
        print(f"Intermediate result: amp_scaling={amp_scaling}, det_zeeman_2={det_zeeman_2}, fidelity={fid}")
        
        return -fid  # Minimizing negative fidelity to maximize fidelity
    
    opt_result = differential_evolution(fidelity_objective, bounds=[(amp_min, amp_max), (lower_det_bound, upper_det_bound)], strategy='best1bin', tol=1e-5)
    
    return {"optimal_amp": opt_result.x[0], "optimal_det_zeeman_2": opt_result.x[1], "optimal_fidelity": -opt_result.fun}

def optimize_fidelity_gradient(amp_min, amp_max, lower_det_bound, upper_det_bound, rotation_config):
    """
    Gradient-based optimization using L-BFGS-B to maximize fidelity.
    """
    
    def fidelity_objective(params):
        amp_scaling, det_zeeman_2 = params
        rotation_config["amp_scaling"] = amp_scaling
        rotation_config["det_zeeman_2"] = det_zeeman_2
        
        result = run_mesolve(rotation_config)
        fid = find_fidelity(result, rotation_config)
        
        print(f"Gradient optimizer Intermediate result: amp_scaling={amp_scaling}, det_zeeman_2={det_zeeman_2}, fidelity={fid}")
        
        return -fid  # Minimizing negative fidelity to maximize fidelity
    
    initial_guess = [(amp_min + amp_max) / 2, (lower_det_bound + upper_det_bound) / 2]
    bounds = [(amp_min, amp_max), (lower_det_bound, upper_det_bound)]
    
    opt_result = minimize(fidelity_objective, x0=initial_guess, bounds=bounds, method='L-BFGS-B')
    
    return {"optimal_amp": opt_result.x[0], "optimal_det_zeeman_2": opt_result.x[1], "optimal_fidelity": -opt_result.fun}

# Example usage
amp_min = 0.5
amp_max = 7.5
lower_det_bound = -3.14
upper_det_bound = 3.14

#global_optimal_values = optimize_fidelity_global(amp_min, amp_max, lower_det_bound, upper_det_bound, rotation_config)
gradient_optimal_values = optimize_fidelity_gradient(amp_min, amp_max, lower_det_bound, upper_det_bound, rotation_config)

#print("Global Optimization Results:", global_optimal_values)
print("Gradient Optimization Results:", gradient_optimal_values)'
'''
