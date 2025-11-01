
import matplotlib.pyplot as plt
import numpy as np

from qutip import mesolve

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

# precompute ketbras for speed
ketbras = kb_class.getrb_ketbras()

# specify system b field groundstate splitting in MHz
bfieldsplit = "0p07"
# configure rb atom with desired CG coefficients and splittings
rb_atom = RbAtom(bfieldsplit, kb_class)

# Define the laser pulse parameters
final_coherences = []
pulse_length = 1 / (2 * np.pi) + 0.005
phase_list = np.linspace(0, 2 * np.pi, 20)

# The pulse is a flattop pulse with a blackman window
pol_1 = "pi"
pulse_time = np.linspace(0, pulse_length, 50000)
pulse_1 = create_flattop_blackman(pulse_time, 1, 0.025)
amp_1 = 1 * np.sqrt(2 * 500000) * 2 * np.pi / rb_atom.CG_d1g2x1

for phase in phase_list:

    det_1 = -500000 * 2 * np.pi

    pol_2 = "sigmaM"
    pulse_2 = create_flattop_blackman(pulse_time, 1, 0.025)
    det_2 = det_1 - rb_atom.getrb_gs_splitting()
    amp_2 = 1 * np.sqrt(2 * 500000) * 2 * np.pi / rb_atom.CG_d1g1Px1

    # add hamiltonian
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
        phase,
    )

    # add decay ops
    c_op_list = []
    c_op_list += rb_atom.spont_em_ops_far_detuned(atomStates, pol_1, amp_1, det_1)
    c_op_list += rb_atom.spont_em_ops_far_detuned(atomStates, pol_2, amp_2, det_2)

    #
    psi0 = kb_class.get_ket_atomic("g2")
    output_mesolve = mesolve(ham, psi0, pulse_time, c_op_list)

    # real part of coherence
    final_real_coherence = np.real(output_mesolve.states[-1][2][0][5])
    final_imag_coherence = np.imag(output_mesolve.states[-1][2][0][5])

    print(final_real_coherence)
    print(final_imag_coherence)

    final_coherences.append(
        (final_real_coherence, final_imag_coherence, np.round(phase, 4))
    )

# Extract data from the final_coherences list
plot_real_coherences = [item[0] for item in final_coherences]
plot_imag_coherences = [item[1] for item in final_coherences]
plot_two_photon_detunings = [item[2] for item in final_coherences]

# Plot the real and imaginary parts of coherence
plt.figure(figsize=(10, 6))

# Real part
plt.plot(
    plot_two_photon_detunings,
    plot_real_coherences,
    label="Real Part of Coherence",
    color="blue",
    marker="o",
)

# Imaginary part
plt.plot(
    plot_two_photon_detunings,
    plot_imag_coherences,
    label="Imaginary Part of Coherence",
    color="red",
    marker="s",
)

# Add labels and title
plt.xlabel("Relative Phase /rad ")
plt.ylabel("Coherence")
plt.title("Real and Imaginary Coherence vs Two-Photon Detuning")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)  # Reference line
plt.legend()
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.savefig("calibrate_pi2_pulse.pdf")
plt.show()
