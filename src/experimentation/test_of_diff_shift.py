import matplotlib.pyplot as plt
import numpy as np


# importing sys

from src.modules.simulation import Simulation
from src.modules.differential_light_shifts import DifferentialStarkShifts


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
"""
atomStates = {"g1M":0, "g2":1}
"""

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
    cavity=False, bfieldsplit="0", ground_states=_ground_states, x_states=_x_states
)


t = np.linspace(0, 0.1, 100)
T = 0.1 / 9

cg_pump_str = "CG_d1g2x1M"
cg_stokes_str = "CG_d1g1Mx1M"
cg_pump = rb_atom_sim.get_CG(cg_pump_str)
cg_stokes = rb_atom_sim.get_CG(cg_stokes_str)

stokes_amp = 30 / cg_stokes
pump_amp = 30 / cg_pump

# create arrays of the stokes and pump pulses
stokes_pulse = np.zeros(len(t))
pump_pulse = np.zeros(len(t))


def stokes_fct(t, tot_t):
    return np.exp(-((t - 4.5 * T - 0.6 * T) ** 2) / (T**2)) * 1 / np.sqrt(2)


def pump_fct(t, tot_t):
    return np.exp(-((t - 4.5 * T + 0.6 * T) ** 2) / (T**2)) + np.exp(
        -((t - 4.5 * T - 0.6 * T) ** 2) / T**2
    ) * 1 / np.sqrt(2)


for i in range(len(t)):
    stokes_pulse[i] = stokes_amp * pump_fct(t[i], t[-1])
    pump_pulse[i] = pump_amp * stokes_fct(t[i], t[-1])

# calculate shifts from the stokes and pump laser pulses
diff_shift = DifferentialStarkShifts("d1", rb_atom_sim.rb_atom, rb_atom_sim.atom_states)
shift_dict_stokes = diff_shift.calculate_td_detuning(
    2, stokes_pulse, -500 * 2 * np.pi, "pi"
)
shift_dict_pump = diff_shift.calculate_td_detuning(
    1, pump_pulse, -500 * 2 * np.pi, "sigmaM"
)
init_shift = diff_shift.find_state_evolution(pump_pulse, shift_dict_pump, "g2")
x_shift_p = diff_shift.find_state_evolution(pump_pulse, shift_dict_pump, "x1M_d1")
fin_shift = diff_shift.find_state_evolution(stokes_pulse, shift_dict_stokes, "g1M")
x_shift_s = diff_shift.find_state_evolution(stokes_pulse, shift_dict_stokes, "x1M_d1")
x_shift_tot = x_shift_p + x_shift_s

# calculate time varying detuning from the shifts of the levels
_pump_det = x_shift_tot - init_shift
_stokes_det = x_shift_tot - fin_shift


plt.plot(t, stokes_pulse, label="stokes")
plt.plot(t, pump_pulse, label="pump")
plt.legend()
plt.show()

plt.plot(t, init_shift, label="initial state")
plt.plot(t, fin_shift, label="final state")
plt.plot(t, x_shift_tot, label="excited state")
plt.legend()
plt.show()

plt.plot(t, _pump_det, label="pump detuning")
plt.plot(t, _stokes_det, label="stokes detuning")
plt.legend()
plt.show()
