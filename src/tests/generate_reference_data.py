"""
Script to generate and save reference density matrices for regression tests.
Run this script to create reference data files that the tests will use.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from itertools import chain
from qutip import mesolve

from src.modules.atom_config import RbAtom
from src.modules.cavity import cav_collapse_ops, quant_axis_cavbasis_mapping
from src.modules.ketbra_config import RbKetBras
from src.modules.laser_pulses import create_flattop_blackman


def generate_general_rb_simulator_reference():
    """Generate reference data for General Rb Simulator test."""
    print("=" * 70)
    print("GENERATING REFERENCE DATA FOR GENERAL RB SIMULATOR")
    print("=" * 70)

    # Configure atomic states
    atomStates = {
        "g1M": 0,
        "g1": 1,
        "g1P": 2,
        "g2MM": 3,
        "g2M": 4,
        "g2": 5,
        "g2P": 6,
        "g2PP": 7,
    }

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
        "x1M_d1",
        "x1_d1",
        "x1P_d1",
        "x2MM_d1",
        "x2M_d1",
        "x2_d1",
        "x2P_d1",
        "x2PP_d1",
    ]

    kb_class = RbKetBras(atomStates, xlvls, True)
    ketbras = kb_class.getrb_ketbras()

    bfieldsplit = "0"
    rb_atom = RbAtom(bfieldsplit, kb_class)

    # Cavity parameters
    kappa = 2.1 * 2.0 * np.pi
    deltaP = 0 * 2.0 * np.pi
    cav_transition = rb_atom.CGg1Mx1
    coupling_factor = 11.1 * 2 * np.pi / cav_transition
    cav_axis = [1, 0, 0]
    quant_axis = [0, 1, 0]

    # VSTIRAP
    lengthStirap = 0.35
    OmegaStirap = 230 * 2 * np.pi
    delta_cav = 0
    delta_laser = 0
    args_omega_stirap = dict([("T", lengthStirap), ("wStirap", np.pi / lengthStirap)])
    vst_driving_shape = "(np.sin(wStirap*t)**2)"

    F_vst_start = 2
    F_vst_final = 1
    F_vst_exc = 1

    quant_axis_mapping = quant_axis_cavbasis_mapping(quant_axis, cav_axis)
    H_VStirap, args_hams_VStirap = rb_atom.gen_H_VSTIRAP_D2(
        ketbras,
        atomStates,
        delta_cav,
        delta_laser,
        F_vst_start,
        F_vst_final,
        F_vst_exc,
        "pi",
        OmegaStirap,
        coupling_factor,
        deltaP,
        quant_axis_mapping,
        args_omega_stirap,
        vst_driving_shape,
    )

    # Repumping
    A_rep = 41
    CGg2x1 = rb_atom.CGg2x1
    CGg1Px1 = rb_atom.CGg1Px1
    A_s = abs(A_rep / CGg2x1) * 2 * np.pi
    A_p = abs(A_rep / CGg1Px1) * 2 * np.pi
    lengthRepump = 0.15

    a = 11
    n = 6
    c = 0.05
    args_repump = dict([("n", n), ("c", c), ("a", a), ("T", lengthRepump)])

    pump_shape = "np.exp(-((t - (T/2))/c)**(2*n))*np.sin(np.pi/2*(1/(1 + np.exp((-a*(t - T/2))/T))))"
    stokes_shape = "np.exp(-((t - (T/2))/c)**(2*n))*np.cos(np.pi/2*(1/(1 + np.exp((-a*(t - T/2))/T))))"

    delta_sti = 0
    F_pump_start = 1
    F_pump_exc = 1
    F_stokes_start = 2
    F_stokes_exc = 1

    stokes_pol = "pi"
    pump_pol_1 = "sigmaP"
    pump_pol_2 = "sigmaM"

    H_Stirap_Stokes = rb_atom.gen_H_Pulse_D1(
        ketbras,
        atomStates,
        delta_sti,
        F_stokes_start,
        F_stokes_exc,
        stokes_pol,
        A_s,
        args_repump,
        stokes_shape,
    )
    H_Stirap_Pump_1 = rb_atom.gen_H_Pulse_D1(
        ketbras,
        atomStates,
        delta_sti,
        F_pump_start,
        F_pump_exc,
        pump_pol_1,
        A_p,
        args_repump,
        pump_shape,
    )
    H_Stirap_Pump_2 = rb_atom.gen_H_Pulse_D1(
        ketbras,
        atomStates,
        delta_sti,
        F_pump_start,
        F_pump_exc,
        pump_pol_2,
        A_p,
        args_repump,
        pump_shape,
    )

    H_Repump = list(
        chain(*[H_Stirap_Stokes[0], H_Stirap_Pump_1[0], H_Stirap_Pump_2[0]])
    )
    args_hams_Repump = {
        **H_Stirap_Stokes[1],
        **H_Stirap_Pump_1[1],
        **H_Stirap_Pump_2[1],
    }

    # Collapse operators
    c_op_list = cav_collapse_ops(kappa, atomStates)
    c_op_list += rb_atom.spont_em_ops(atomStates)[0]
    c_op_list += rb_atom.spont_em_ops(atomStates)[1]

    # Run simulation
    tVStirap = np.linspace(0, lengthStirap, 200)
    tRepump = np.linspace(0, lengthRepump, 150)
    psi0 = kb_class.get_ket("g2", 0, 0)

    print("Running VSTIRAP simulation...")
    output_vst = mesolve(
        H_VStirap, psi0, tVStirap, c_op_list, [], args=args_hams_VStirap
    )

    print("Running repumping simulation...")
    psi_after_vst = output_vst.states[-1]
    output_repump = mesolve(
        H_Repump, psi_after_vst, tRepump, c_op_list, [], args=args_hams_Repump
    )

    final_rho = output_repump.states[-1]
    final_rho_array = final_rho.full()

    # Save reference data
    np.save("src/tests/reference_general_rb_sim_rho.npy", final_rho_array)

    print("\nSaved reference density matrix to: reference_general_rb_sim_rho.npy")
    print(f"Shape: {final_rho_array.shape}")
    print(f"Trace: {np.trace(final_rho_array)}")
    print(f"Max element: {np.max(np.abs(final_rho_array))}")
    print("=" * 70 + "\n")

    return final_rho_array


def generate_far_detuned_raman_reference():
    """Generate reference data for Far-Detuned Raman test."""
    print("=" * 70)
    print("GENERATING REFERENCE DATA FOR FAR-DETUNED RAMAN")
    print("=" * 70)

    # Configure atomic states
    atomStates = {
        "g1M": 0,
        "g1": 1,
        "g1P": 2,
        "g2MM": 3,
        "g2M": 4,
        "g2": 5,
        "g2P": 6,
        "g2PP": 7,
    }

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
        "x1M_d1",
        "x1_d1",
        "x1P_d1",
        "x2MM_d1",
        "x2M_d1",
        "x2_d1",
        "x2P_d1",
        "x2PP_d1",
    ]

    kb_class = RbKetBras(atomStates, xlvls, False)
    ketbras = kb_class.getrb_ketbras()

    bfieldsplit = "0"
    rb_atom = RbAtom(bfieldsplit, kb_class)

    # Pulse parameters
    two_phot_det = 0.0
    pulse_length = 5 / (2 * np.pi) + 0.005
    pulse_time = np.linspace(0, pulse_length, 5000)

    pulse_1 = create_flattop_blackman(pulse_time, 1, 0.025, 0.025)
    pulse_2 = create_flattop_blackman(pulse_time, 1, 0.025, 0.025)

    pol_1 = "pi"
    pol_2 = "sigmaM"

    det_1 = -500000 * 2 * np.pi
    det_2 = det_1 - two_phot_det * 2 * np.pi - rb_atom.getrb_gs_splitting()

    amp_1 = 1 * np.sqrt(2 * 500000) * 2 * np.pi / rb_atom.CG_d1g2x1
    amp_2 = 1 * np.sqrt(2 * 500000) * 2 * np.pi / rb_atom.CG_d1g1Px1

    ham, args = rb_atom.gen_H_FarDetuned_Raman_PulsePair_D1(
        ketbras,
        atomStates,
        det_1,
        det_2,
        pol_1,
        pol_2,
        amp_1,
        amp_2,
        pulse_time,
        pulse_1,
        pulse_2,
    )

    # Collapse operators
    c_op_list = []
    c_op_list += rb_atom.spont_em_ops_far_detuned(atomStates, pol_1, amp_1, det_1)
    c_op_list += rb_atom.spont_em_ops_far_detuned(atomStates, pol_2, amp_2, det_2)

    # Run simulation
    psi0 = kb_class.get_ket_atomic("g2")

    print("Running far-detuned Raman simulation...")
    output_mesolve = mesolve(ham, psi0, pulse_time, c_op_list)

    final_rho = output_mesolve.states[-1]
    final_rho_array = final_rho.full()

    # Save reference data
    np.save("src/tests/reference_far_detuned_raman_rho.npy", final_rho_array)

    print("\nSaved reference density matrix to: reference_far_detuned_raman_rho.npy")
    print(f"Shape: {final_rho_array.shape}")
    print(f"Trace: {np.trace(final_rho_array)}")
    print(f"Max element: {np.max(np.abs(final_rho_array))}")
    print("=" * 70 + "\n")

    return final_rho_array


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("REFERENCE DATA GENERATION SCRIPT")
    print("=" * 70 + "\n")

    rho1 = generate_general_rb_simulator_reference()
    rho2 = generate_far_detuned_raman_reference()

    print("\n" + "=" * 70)
    print("REFERENCE DATA GENERATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - src/tests/reference_general_rb_sim_rho.npy")
    print("  - src/tests/reference_far_detuned_raman_rho.npy")
    print("\nThese files will be loaded by the tests for regression checking.")
    print("=" * 70 + "\n")
