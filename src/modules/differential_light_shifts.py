import numpy as np
from sympy.physics.wigner import wigner_6j
from scipy.constants import hbar, h
from src.modules.atom_config import RbAtom


class DifferentialStarkShifts:
    def __init__(self, d_line: str, atom: RbAtom, atom_states: dict):
        """Calculates the differential Stark shift for a given transition in the near resonant case where only one line is considered at the same time and the detuning from the excited levels is smaller than the ground level splitting
        Args:
            d_line (str): transition line (d1 or d2)
            atom (RbAtom): atom object containing the relevant atomic data
            atom_states (dict): dict of atomic states"""

        self.d_line = d_line
        self.atom_states = atom_states
        self.rb_atom = atom
        if d_line == "d2":
            self.d0 = self.rb_atom.getrb_rates()[2]
        elif d_line == "d1":
            self.d0 = self.rb_atom.getrb_rates()[3]
        else:
            raise ValueError("No valid transition line specified")

    def omega_to_e0(self, omega):
        """Converts the rabi frequency of the laser to the electric field strength
        Args:
            omega (float): rabi frequency of the laser in MHz WITHOUT angular CG dependence
        """
        return hbar * omega * 10 ** (6) / self.d0

    def str_to_mF(self, input_str: str):
        """Converts the mF state from a string to an integer"""
        if len(input_str) > 2:
            process_str = input_str[2:]
            if process_str == "PPP":
                return 3
            elif process_str == "PP" or process_str == "PP_d1":
                return 2
            elif process_str == "P" or process_str == "P_d1":
                return 1
            elif process_str == "M" or process_str == "M_d1":
                return -1
            elif process_str == "MM" or process_str == "MM_d1":
                return -2
            elif process_str == "MMM":
                return -3
            elif process_str == "_d1":
                return 0
            else:
                raise ValueError("No valid mF state given")
        else:
            return 0

    def str_to_Fe(self, input_str: str):
        """Converts the Fe state from a string to an integer
        input args: input_str (str): string of the state"""
        process_str = input_str[1]
        if process_str == "0":
            return 0
        elif process_str == "1":
            return 1
        elif process_str == "2":
            return 2
        elif process_str == "3":
            return 3
        else:
            raise ValueError("No valid Fe state given")

    def get_efield_couplings(self, F_g, omega, delta, pol):
        """Calculates the electric field strength for a laser of given polarisation and rabi frequency
        Args:
            F_g (int): total angular momentum of the ground state
            omega (float): rabi frequency of the laser in MHz WITHOUT angular CG dependence
            delta (float): detuning of the transition in MHz
            pol (str): polarisation of the laser (pi, sigmaP, sigmaM)"""
        e0 = self.omega_to_e0(omega) ** 2
        if pol == "pi":
            e1 = 0
            e2 = 2 * self.omega_to_e0(omega) ** 2
            if self.d_line == "d2":
                if F_g == 1:
                    if abs(delta) >= 1000:
                        couplings = self.rb_atom.getCouplingsF1_Pi(
                            2 * np.pi * delta
                        ) + self.rb_atom.getCouplingsF2_Pi(
                            2 * np.pi * (delta) + self.rb_atom.getrb_gs_splitting()
                        )
                        return [couplings, e0, e1, e2]
                    else:
                        couplings = self.rb_atom.getCouplingsF1_Pi(2 * np.pi * delta)
                        return [couplings, e0, e1, e2]
                if F_g == 2:
                    if abs(delta) >= 1000:
                        couplings = self.rb_atom.getCouplingsF1_Pi(
                            2 * np.pi * (delta) - self.rb_atom.getrb_gs_splitting()
                        ) + self.rb_atom.getCouplingsF2_Pi(2 * np.pi * (delta))
                        return [couplings, e0, e1, e2]
                    else:
                        couplings = self.rb_atom.getCouplingsF2_Pi(2 * np.pi * delta)
                        return [couplings, e0, e1, e2]

            elif self.d_line == "d1":
                if F_g == 1:
                    if abs(delta) >= 1000:
                        couplings = self.rb_atom.getD1CouplingsF1_Pi(
                            2 * np.pi * delta
                        ) + self.rb_atom.getD1CouplingsF2_Pi(
                            2 * np.pi * (delta) + self.rb_atom.getrb_gs_splitting()
                        )
                        return [couplings, e0, e1, e2]
                    else:
                        couplings = self.rb_atom.getD1CouplingsF1_Pi(2 * np.pi * delta)
                        return [couplings, e0, e1, e2]
                if F_g == 2:
                    if abs(delta) >= 1000:
                        couplings = self.rb_atom.getD1CouplingsF1_Pi(
                            2 * np.pi * (delta) - self.rb_atom.getrb_gs_splitting()
                        ) + self.rb_atom.getD1CouplingsF2_Pi(2 * np.pi * (delta))
                        return [couplings, e0, e1, e2]
                    else:
                        couplings = self.rb_atom.getD1CouplingsF2_Pi(2 * np.pi * delta)
                        return [couplings, e0, e1, e2]

        elif pol == "sigmaP":
            e1 = self.omega_to_e0(omega) ** 2
            e2 = -self.omega_to_e0(omega) ** 2
            if self.d_line == "d2":
                if F_g == 1:
                    if abs(delta) >= 1000:
                        couplings = self.rb_atom.getCouplingsF1_Sigma_Plus(
                            2 * np.pi * delta
                        ) + self.rb_atom.getCouplingsF2_Sigma_Plus(
                            2 * np.pi * (delta) + self.rb_atom.getrb_gs_splitting()
                        )
                        return [couplings, e0, e1, e2]
                    else:
                        couplings = self.rb_atom.getCouplingsF1_Sigma_Plus(
                            2 * np.pi * delta
                        )
                        return [couplings, e0, e1, e2]
                if F_g == 2:
                    if abs(delta) >= 1000:
                        couplings = self.rb_atom.getCouplingsF1_Sigma_Plus(
                            2 * np.pi * (delta) - self.rb_atom.getrb_gs_splitting()
                        ) + self.rb_atom.getCouplingsF2_Sigma_Plus(2 * np.pi * (delta))
                        return [couplings, e0, e1, e2]
                    else:
                        couplings = self.rb_atom.getCouplingsF2_Sigma_Plus(
                            2 * np.pi * delta
                        )
                        return [couplings, e0, e1, e2]
            elif self.d_line == "d1":
                if F_g == 1:
                    if abs(delta) >= 1000:
                        couplings = self.rb_atom.getD1CouplingsF1_Sigma_Plus(
                            2 * np.pi * delta
                        ) + self.rb_atom.getD1CouplingsF2_Sigma_Plus(
                            2 * np.pi * (delta + self.rb_atom.getrb_gs_splitting())
                        )
                        return [couplings, e0, e1, e2]
                    else:
                        couplings = self.rb_atom.getD1CouplingsF1_Sigma_Plus(
                            2 * np.pi * delta
                        )
                        return [couplings, e0, e1, e2]
                if F_g == 2:
                    if abs(delta) >= 1000:
                        couplings = self.rb_atom.getD1CouplingsF1_Sigma_Plus(
                            2 * np.pi * (delta) - self.rb_atom.getrb_gs_splitting()
                        ) + self.rb_atom.getD1CouplingsF2_Sigma_Plus(
                            2 * np.pi * (delta)
                        )
                        return [couplings, e0, e1, e2]
                    else:
                        couplings = self.rb_atom.getD1CouplingsF2_Sigma_Plus(
                            2 * np.pi * delta
                        )
                        return [couplings, e0, e1, e2]

        elif pol == "sigmaM":
            e1 = -self.omega_to_e0(omega) ** 2
            e2 = -self.omega_to_e0(omega) ** 2
            if self.d_line == "d2":
                if F_g == 1:
                    if abs(delta) >= 1000:
                        couplings = self.rb_atom.getCouplingsF1_Sigma_Minus(
                            2 * np.pi * delta
                        ) + self.rb_atom.getCouplingsF2_Sigma_Minus(
                            2 * np.pi * (delta) + self.rb_atom.getrb_gs_splitting()
                        )
                        return [couplings, e0, e1, e2]
                    else:
                        couplings = self.rb_atom.getCouplingsF1_Sigma_Minus(
                            2 * np.pi * delta
                        )
                        return [couplings, e0, e1, e2]
                if F_g == 2:
                    if abs(delta) >= 1000:
                        couplings = self.rb_atom.getCouplingsF1_Sigma_Minus(
                            2 * np.pi * (delta) - self.rb_atom.getrb_gs_splitting()
                        ) + self.rb_atom.getCouplingsF2_Sigma_Minus(2 * np.pi * (delta))
                        return [couplings, e0, e1, e2]
                    else:
                        couplings = self.rb_atom.getCouplingsF2_Sigma_Minus(
                            2 * np.pi * delta
                        )
                        return [couplings, e0, e1, e2]
            elif self.d_line == "d1":
                if F_g == 1:
                    if abs(delta) >= 1000:
                        couplings = self.rb_atom.getD1CouplingsF1_Sigma_Minus(
                            2 * np.pi * delta
                        ) + self.rb_atom.getD1CouplingsF2_Sigma_Minus(
                            2 * np.pi * (delta) + self.rb_atom.getrb_gs_splitting()
                        )
                        return [couplings, e0, e1, e2]
                    else:
                        couplings = self.rb_atom.getD1CouplingsF1_Sigma_Minus(
                            2 * np.pi * delta
                        )
                        return [couplings, e0, e1, e2]
                if F_g == 2:
                    if abs(delta) >= 1000:
                        couplings = self.rb_atom.getD1CouplingsF1_Sigma_Minus(
                            2 * np.pi * (delta) - self.rb_atom.getrb_gs_splitting()
                        ) + self.rb_atom.getD1CouplingsF2_Sigma_Minus(
                            2 * np.pi * (delta)
                        )
                        return [couplings, e0, e1, e2]
                    else:
                        couplings = self.rb_atom.getD1CouplingsF2_Sigma_Minus(
                            2 * np.pi * delta
                        )
                        return [couplings, e0, e1, e2]
        else:
            raise ValueError("No valid polarisation given")

    def calculate_stark_shift(self, F_g, omega, delta, pol):
        """Calculates the differential Stark shift for a given transition in the near resonant case where only one line is considered at the same time
        Args:
            omega (float): rabi frequency of the laser (not including angular CG coefficient) in MHz
            F_g(int): total angular momentum of the ground level
            delta (float): detuning of the transition in MHz
            pol (str): polarisation of the laser (pi, sigmaP, sigmaM)

        Returns:
            e_splitting (dict): dictionary of the Stark shift for each configured state in MHz
        """

        def transition_freq_g_x(g, e):
            """Calculates the transition frequency from the ground level to excited level in 2pi MHz
            Args:
            g (int): total angular momentum of the ground level
            e (int): total angular momentum of the excited level"""
            return self.rb_atom.getrb_transition_line_freq(g, e, self.d_line)

        def a0(angfrequ_xg, angfreq_l, dip_me):
            """Calculates the linear shift coefficient for a given excited state
            Args:
                F_e (int): total angular momentum of the excited state
                frequ_xg (float): frequency of the transition in 2piMHz
                freq_l (float): frequency of the laser in 2piMHz
                dip_me (float): dipole matrix element for transitions"""

            return (
                2
                * angfrequ_xg
                / (3 * hbar * (angfrequ_xg**2 - angfreq_l**2) * 10 ** (6))
                * dip_me**2
            )

        def a1(F_e, angfrequ_xg, angfreq_l, dip_me):
            """Calculates the quadratic shift coefficient for a given excited state
            Args:
                F_e (int): total angular momentum of the excited state
                frequ_xg (float): frequency of the transition in 2piMHz
                freq_l (float): frequency of the laser in 2piMHz
                dip_me (float): dipole matrix element for transitions"""
            return (
                (-1) ** (F_g + F_e + 1)
                * np.sqrt(6 * F_g * (2 * F_g + 1) / (F_g + 1))
                * wigner_6j(1, 1, 1, F_g, F_g, F_e)
                * angfrequ_xg
                / (hbar * (angfrequ_xg**2 - angfreq_l**2) * 10 ** (6))
                * dip_me**2
            )

        def a2(F_e, angfrequ_xg, angfreq_l, dip_me):
            """Calculates the cubic shift coefficient for a given excited state
            Args:
                F_e (int): total angular momentum of the excited state
                frequ_xg (float): frequency of the transition in 2piMHz
                freq_l (float): frequency of the laser in 2piMH
                dip_me (float): dipole matrix element for transitions"""
            return (
                (-1) ** (F_g + F_e)
                * np.sqrt(
                    (40 * F_g * (2 * F_g + 1) * (2 * F_g - 1))
                    / (3 * (F_g + 1) * (2 * F_g + 3))
                )
                * wigner_6j(1, 1, 2, F_g, F_g, F_e)
                * angfrequ_xg
                / (hbar * (angfrequ_xg**2 - angfreq_l**2) * 10 ** (6))
                * dip_me**2
            )

        def e_shift(F_e, m_F, angfrequ_xg, angfreq_l, dip_me):
            """Calculates the differential Stark shift for a given m_F state
            Args:
                F_e (int): total angular momentum of the excited state
                m_F (int): magnetic quantum number of the ground state
                frequ_xg (float): frequency of the transition in 2piMHz
                freq_l (float): frequency of the laser in 2piMHz
                dip_me (float): dipole matrix element for transitions
            """
            return (
                -a0(angfrequ_xg, angfreq_l, dip_me) * e0
                - e1 * a1(F_e, angfrequ_xg, angfreq_l, dip_me) * m_F / F_g
                - a2(F_e, angfrequ_xg, angfreq_l, dip_me)
                * e2
                / 2
                * (3 * m_F**2 - F_g * (F_g + 1) / (F_g * (2 * F_g - 1)))
            )

        couplings, e0, e1, e2 = self.get_efield_couplings(F_g, omega, delta, pol)

        kb_keys = self.atom_states.keys()
        e_splitting = {}
        for key in kb_keys:
            e_splitting[key] = 0

        for cg, gs, es, delta, delta_mf in couplings:
            # return in MHz
            e_splitting[gs] += e_shift(
                self.str_to_Fe(es),
                self.str_to_mF(es),
                transition_freq_g_x(F_g, self.str_to_Fe(es)),
                transition_freq_g_x(F_g, self.str_to_Fe(es)) + delta * 2 * np.pi,
                cg * self.d0,
            ) / (h * 10**6)
            e_splitting[es] += e_shift(
                self.str_to_Fe(es),
                self.str_to_mF(es),
                transition_freq_g_x(F_g, self.str_to_Fe(es)),
                transition_freq_g_x(F_g, self.str_to_Fe(es)) + delta * 2 * np.pi,
                cg * self.d0,
            ) / (h * 10**6)

        return e_splitting

    def calculate_td_detuning(self, F_g, omega_t_array, delta, pol):
        """Calculates the time dependent detuning of the transition
        Args:
            F_g (int): total angular momentum of the ground state
            omega_t_array List(float): list of rabi frequency values in MHz for laser WITHOUT angular CG dependence
            delta (float): detuning of the transition in MHz
            pol (str): polarisation of the laser (pi, sigmaP, sigmaM)

        Returns:
            e_splitting_list (list(dict)): list of dictionaries containing the time dependent detunings for each state at every timestep in MHz
        """

        e_splitting_list = []
        for omega in omega_t_array:
            e_splitting_list.append(self.calculate_stark_shift(F_g, omega, delta, pol))

        return e_splitting_list

    def find_state_evolution(self, omega_t_array, dict_list, desired_state: str):
        """Finds the evolution of the desired state for a given list of dictionaries and return array of detunings
        Args:
            dict_list (list(dict)): list of dictionaries containing the time dependent detunings for each state at every timestep in MHz
                                    fetched from self.calculated_td_detuning()
            desired_state (str): atomic state for which stark shift is returne
        Returns:
            delta_list (list(float)): list of detunings for the desired state at each timestep in MHz
        """

        delta_list = np.zeros(len(omega_t_array))
        for i, e_dict in enumerate(dict_list):
            if desired_state not in e_dict.keys():
                raise ValueError("No valid state given")
            for key, value in e_dict.items():
                if key == desired_state:
                    delta_list[i] = value

        return delta_list
