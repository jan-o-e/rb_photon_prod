from functools import partial
from concurrent.futures import ProcessPoolExecutor

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import matplotlib.pyplot as plt
import numpy as np
from qutip import mesolve, fidelity

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
amp_scaling = 4
# number_of_rabi_cycles = 4
# pulse_length = 1 / amp_scaling**2 * number_of_rabi_cycles
pulse_length = 0.125
phase_list = np.linspace(0, 2 * np.pi, 64)


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
        "amp_scaling": 1.65587099,
        "pulse_length": 0.125,
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
        "amp_scaling": 2.21831585,
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
        "amp_scaling": 1.52701423,
        "pulse_length": 0.25,
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
        "amp_scaling": 1.52701423,
        "pulse_length": 0.25,
        "rel_phase": 0,
    },
}


def run_mesolve(config, phase):
    """
    Runs the mesolve simulation with the given configuration.
    """
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
        _rel_phase=phase,
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


def compute_fidelity(phase, config):
    result = run_mesolve(config, phase)
    return fidelity(result.states[-1], config["psi_des"])


if __name__ == "__main__":
    # Global parameters
    rotation_number = 4

    # Select the correct configuration
    if rotation_number not in rotation_configs:
        raise ValueError("Rotation number not recognised")

    rotation_config = rotation_configs[rotation_number]

    # Ensure rotation_config is passed explicitly
    compute_fidelity_with_config = partial(compute_fidelity, config=rotation_config)

    with ProcessPoolExecutor(max_workers=8) as executor:
        fidelities = list(executor.map(compute_fidelity_with_config, phase_list))

    max_fid = max(fidelities)
    max_fid_idx = fidelities.index(max_fid)
    max_fid_phase = phase_list[max_fid_idx]

    print(f"Max fidelity: {max_fid}")
    print(f"Max fidelity phase: {max_fid_phase}")

    plt.plot(phase_list, fidelities)
    plt.xlabel("Relative phase")
    plt.ylabel("Fidelity")
    plt.title("Fidelity vs relative phase")
    plt.savefig(f"far_detuned_phase_calibration_rot_{rotation_number}.pdf")
    plt.show()
