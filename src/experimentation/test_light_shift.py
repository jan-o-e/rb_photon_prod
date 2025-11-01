from scipy.signal.windows import blackman

from src.modules.atom_config import RbAtom
from src.modules.ketbra_config import RbKetBras
from src.modules.differential_light_shifts import DifferentialStarkShifts

# Define the atom and ketbra objects
# List the groundstates to be included in the simulation

ground_states = {
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
x_states = [
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

# configure the atm state dictionary such that it includes the desired excited states, this includes the desired ground and excited states as arguments,
# as well as a boolean for whether to include the photonic Hilbert space

kb_class = RbKetBras(ground_states, x_states, False)

# precompute ketbras for speed
ketbras = kb_class.getrb_ketbras()

atom_states = kb_class.atomStates

rb_atom = RbAtom(0, kb_class)

# Define the differential light shift object
omega_array = blackman(200)
omega_array *= 200
diff_shift = DifferentialStarkShifts("d1", rb_atom, atom_states)
shift_dict = diff_shift.calculate_td_detuning(2, omega_array, -400, "sigmaM")
td_evolution = diff_shift.find_state_evolution(omega_array, shift_dict, "g2")

print(td_evolution)
