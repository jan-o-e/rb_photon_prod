from src.modules.vector_functions import perpendicular_vector
import numpy as np
from qutip import tensor, destroy, qeye, expect, qdiags
from matplotlib import pyplot as plt


def quant_axis_cavbasis_mapping(b_vec, cav_vec):
    """define the basis transformation between atomic polarisation basis (defined by quantisation axis, i.e. magnetic field direction) and cavity polarisation basis
    input args:
    b_vec: three dimensional vector
    cav_vec: three dimensional vector
    linear cavity basis is spanned by two vectors perpendicular to the cavity axis"""
    cav_pol_1 = perpendicular_vector(cav_vec)
    cav_pol_2 = np.cross(cav_pol_1, cav_vec)

    # define plane perpendicular to the quantisation axis
    b_perp_1 = perpendicular_vector(b_vec)
    b_perp_2 = np.cross(b_perp_1, b_vec)

    # transform between lab and atomic basis
    sigma_plus_lab = (
        1 / np.linalg.norm(b_perp_1 + 1j * b_perp_2) * (b_perp_1 + 1j * b_perp_2)
    )
    sigma_minus_lab = (
        1 / np.linalg.norm(b_perp_1 - 1j * b_perp_2) * (b_perp_1 - 1j * b_perp_2)
    )

    # express two-dimensional cavity basis in terms of contributions in atomic basis [pi,sigmaPlus,sigmaMinus]
    cav_x_atom = [
        np.dot(cav_pol_1, b_vec),
        np.dot(cav_pol_1, sigma_plus_lab),
        np.dot(cav_pol_1, sigma_minus_lab),
    ]
    cav_y_atom = [
        np.dot(cav_pol_2, b_vec),
        np.dot(cav_pol_2, sigma_plus_lab),
        np.dot(cav_pol_2, sigma_minus_lab),
    ]

    return [cav_x_atom, cav_y_atom]


def cav_basis_trans(alpha, phi1, phi2):
    """define cavity basis transformation used for the cavity emission plotter
    input args:
    alpha: rotation param for diagonal transformation matrix elements
    phi1: phase transformation angle for off-diagonal transformation matrix elements
    phi2: phase transformation angle for diagnoal transformation matrix elements"""

    alpha = alpha
    beta = np.sqrt(1 - alpha**2)
    phi1 = phi1
    phi2 = phi2
    # returns arguments of a 2x2 transformation matrix
    return [alpha, beta, phi1, phi2]


def cav_collapse_ops(kappa, atomStates):
    """define caviy collapse operators
    input arguments:
    kappa: cavity decay rate
    atomStates: list of atomic states as dict"""

    # truncating Fock states
    N = 2
    M = len(atomStates)
    # Create the photon operators
    aX = tensor(qeye(M), destroy(N), qeye(N))
    aY = tensor(qeye(M), qeye(N), destroy(N))
    """
    The c_op_list defines the collapse operators of the system. Namely
    - the rate of photon decay from the cavity
    - spontaneous decay of the excited atomic states
    """
    # Define collapse operators
    c_op_list = []

    # Cavity decay rate
    c_op_list.append(np.sqrt(2 * kappa) * aX)
    c_op_list.append(np.sqrt(2 * kappa) * aY)

    return c_op_list


def plotter_cavemission(
    ketbras,
    atomStates,
    output,
    t_list,
    kappa,
    deltaP=0.0,
    rotated_basis=False,
    angles_pol=[0, 0, 0, 0],
    show_plts=True,
):
    """
    cavity emission plotter with inputs:
    ketbras: dictionary of ketbras of the atomic states
    atomStates: list of atomic states as dict
    output: output of the simulation
    t_list: list of simulation time
    angles_pol: list of angles for the cavity basis transformation
    kappa: cavity decay rate
    deltaP: detuning between the two cavity polarisation modes
    show_plts (bool): whether to run plt.show()"""
    [alpha, beta, phi1, phi2] = angles_pol
    t = t_list
    output_states = output
    tStep = (t[-1] - t[0]) / (len(t) - 1)

    # truncating Fock states
    N = 2
    M = len(atomStates)
    # Create the photon operators
    aX = tensor(qeye(M), destroy(N), qeye(N))
    aY = tensor(qeye(M), qeye(N), destroy(N))
    anX = aX.dag() * aX
    anY = aY.dag() * aY

    """
    Consider two polarisation bases:
        - linear cavity basis {|X>,|Y>} (i.e. default)
        - rotated {|X'>,|Y'>}
    Photon number operators for calculating the population in the linear and rotated bases.
    """

    exp_anX, exp_anY = np.abs(expect([anX, anY], output_states))

    # Total cavity emission.
    n_ph = np.trapz(2 * kappa * (exp_anX + exp_anY), dx=tStep)
    print("Total cavity emission:", np.round(n_ph, 3))
    n_X = np.trapz(2 * kappa * (exp_anX), dx=tStep)
    n_Y = np.trapz(2 * kappa * (exp_anY), dx=tStep)

    if show_plts:
        # Plotting
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.plot(t, 2 * kappa * exp_anX, "b", label="X photon emission")
        ax.plot(t, 2 * kappa * exp_anY, "g", label="Y photon emission")
        ax.legend(loc="best")
        plt.show()
    print("Total cavity emission in linear basis:", np.round(n_X, 3), np.round(n_Y, 3))

    if rotated_basis:
        # Calculating rotated cavity basis decay rates
        allAtomicStates = list(atomStates)

        def kb(xLev, y):
            return ketbras[str([xLev, y])]

        an_fast_1 = sum(
            map(
                lambda s: kb([s, 1, 0], [s, 1, 0]) + kb([s, 1, 1], [s, 1, 1]),
                allAtomicStates,
            )
        )
        an_fast_2 = sum(
            map(
                lambda s: kb([s, 0, 1], [s, 0, 1]) + kb([s, 1, 1], [s, 1, 1]),
                allAtomicStates,
            )
        )
        an_fast_3 = sum(map(lambda s: kb([s, 0, 1], [s, 1, 0]), allAtomicStates))
        an_fast_4 = sum(map(lambda s: kb([s, 1, 0], [s, 0, 1]), allAtomicStates))

        def anRotP_fast(t, alpha=alpha, phi1=phi1, phi2=phi2):
            beta = np.sqrt(1 - alpha**2)
            delta_phi = phi2 - phi1
            return (alpha**2 * an_fast_1 + beta**2 * an_fast_2) + alpha * beta * (
                np.exp(-1j * deltaP * t) * np.exp(1j * delta_phi) * an_fast_3
                + np.exp(1j * deltaP * t) * np.exp(-1j * delta_phi) * an_fast_4
            )

        def anRotM_fast(t, alpha=alpha, phi1=phi1, phi2=phi2):
            beta = np.sqrt(1 - alpha**2)
            delta_phi = phi2 - phi1
            return (alpha**2 * an_fast_2 + beta**2 * an_fast_1) - alpha * beta * (
                np.exp(-1j * deltaP * t) * np.exp(1j * delta_phi) * an_fast_3
                + np.exp(1j * deltaP * t) * np.exp(-1j * delta_phi) * an_fast_4
            )

        anP_t = [anRotP_fast(time, alpha=alpha, phi1=phi1, phi2=phi2) for time in t]
        anM_t = [anRotM_fast(time, alpha=alpha, phi1=phi1, phi2=phi2) for time in t]

        exp_anP = np.abs(
            np.array([(x[0] * x[1]).tr() for x in zip(output_states, anP_t)])
        )
        exp_anM = np.abs(
            np.array([(x[0] * x[1]).tr() for x in zip(output_states, anM_t)])
        )

        n_P = np.trapezoid(2 * kappa * (exp_anP), dx=tStep)
        n_M = np.trapezoid(2 * kappa * (exp_anM), dx=tStep)

        if show_plts:
            fig_r, ax_r = plt.subplots(figsize=(14, 12))
            ax_r.plot(t, 2 * kappa * exp_anP, "b--", label="X* photon emission")
            ax_r.plot(t, 2 * kappa * exp_anM, "g--", label="Y* photon emission")
            ax.legend(loc="best")
            plt.show()
        print(
            "Rotated cavity basis with params: alpha = {0}, phi1 = {1}, phi2 = {2}".format(
                alpha, phi1, phi2
            )
        )
        print(
            "Total cavity emission in rotated basis:",
            np.round(n_P, 3),
            np.round(n_M, 3),
        )

    return (n_X, n_Y)


def cav_emission_from_state(
    atomStates, atom_key, output, t_list, kappa, show_plts=False
):
    """calculate cavity emission from a particular atomic state
    input args:
    ketbras: dictionary of ketbras of the atomic states
    atomStates: list of atomic states as dict
    atom_key: key of the atomic state
    output: output of the simulation
    t_list: list of simulation time
    show_plts (bool): whether to run plt.show()"""

    tStep = (t_list[-1] - t_list[0]) / (len(t_list) - 1)

    # truncating Fock states
    N = 2
    M = len(atomStates)
    diags = np.array([0] * M)
    diags[atom_key] = 1
    # Create the photon operators
    aX = tensor(qdiags(diags, 0), destroy(N), qeye(N))
    aY = tensor(qdiags(diags, 0), qeye(N), destroy(N))
    anX = aX.dag() * aX
    anY = aY.dag() * aY

    exp_anX, exp_anY = np.abs(expect([anX, anY], output))

    # Total cavity emission.
    n_ph = np.trapz(2 * kappa * (exp_anX + exp_anY), dx=tStep)
    print("Total cavity emission:", np.round(n_ph, 3))
    n_X = np.trapz(2 * kappa * (exp_anX), dx=tStep)
    n_Y = np.trapz(2 * kappa * (exp_anY), dx=tStep)

    if show_plts:
        # Plotting
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.plot(t_list, 2 * kappa * exp_anX, "b", label="X photon emission")
        ax.plot(t_list, 2 * kappa * exp_anY, "g", label="Y photon emission")
        ax.legend(loc="best")
        plt.show()
    print("Total cavity emission in linear basis:", np.round(n_X, 3), np.round(n_Y, 3))

    return (n_X, n_Y)


def cav_emission_from_state_rot(
    ketbras,
    atomStates,
    atom_key,
    output,
    t_list,
    kappa,
    deltaP=0.0,
    rotated_basis=False,
    angles_pol=[0, 0, 0, 0],
    show_plts=False,
):
    """calculate cavity emission from a particular atomic state
    input args:
    ketbras: dictionary of ketbras of the atomic states
    atomStates: list of atomic states as dict
    atom_key: key of the atomic state
    output: output of the simulation
    t_list: list of simulation time
    angles_pol: list of angles for the cavity basis transformation
    kappa: cavity decay rate
    deltaP: detuning between the two cavity polarisation modes
    show_plts (bool): whether to run plt.show()"""

    [alpha, beta, phi1, phi2] = angles_pol
    tStep = (t_list[-1] - t_list[0]) / (len(t_list) - 1)

    # truncating Fock states
    N = 2
    M = len(atomStates)
    diags = np.array([0] * M)
    diags[atom_key] = 1
    # Create the photon operators
    aX = tensor(qdiags(diags, 0), destroy(N), qeye(N))
    aY = tensor(qdiags(diags, 0), qeye(N), destroy(N))
    anX = aX.dag() * aX
    anY = aY.dag() * aY

    exp_anX, exp_anY = np.abs(expect([anX, anY], output))

    # Total cavity emission.
    n_ph = np.trapezoid(2 * kappa * (exp_anX + exp_anY), dx=tStep)
    print("Total cavity emission:", np.round(n_ph, 3))
    n_X = np.trapezoid(2 * kappa * (exp_anX), dx=tStep)
    n_Y = np.trapezoid(2 * kappa * (exp_anY), dx=tStep)

    if show_plts:
        # Plotting
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.plot(t_list, 2 * kappa * exp_anX, "b", label="X photon emission")
        ax.plot(t_list, 2 * kappa * exp_anY, "g", label="Y photon emission")
        ax.legend(loc="best")
        plt.show()
    print("Total cavity emission in linear basis:", np.round(n_X, 3), np.round(n_Y, 3))

    if rotated_basis:

        # Calculating rotated cavity basis decay rates
        allAtomicStates = list(atomStates)
        desired_state = allAtomicStates[atom_key]

        def kb(xLev, y):
            return ketbras[str([xLev, y])]

        an_fast_1 = kb([desired_state, 1, 0], [desired_state, 1, 0]) + kb(
            [desired_state, 1, 1], [desired_state, 1, 1]
        )
        an_fast_2 = kb([desired_state, 0, 1], [desired_state, 0, 1]) + kb(
            [desired_state, 1, 1], [desired_state, 1, 1]
        )
        an_fast_3 = kb([desired_state, 0, 1], [desired_state, 1, 0])
        an_fast_4 = kb([desired_state, 1, 0], [desired_state, 0, 1])

        def anRotP_fast(t, alpha=alpha, phi1=phi1, phi2=phi2):
            beta = np.sqrt(1 - alpha**2)
            delta_phi = phi2 - phi1
            return (alpha**2 * an_fast_1 + beta**2 * an_fast_2) + alpha * beta * (
                np.exp(-1j * deltaP * t) * np.exp(1j * delta_phi) * an_fast_3
                + np.exp(1j * deltaP * t) * np.exp(-1j * delta_phi) * an_fast_4
            )

        def anRotM_fast(t, alpha=alpha, phi1=phi1, phi2=phi2):
            beta = np.sqrt(1 - alpha**2)
            delta_phi = phi2 - phi1
            return (alpha**2 * an_fast_2 + beta**2 * an_fast_1) - alpha * beta * (
                np.exp(-1j * deltaP * t) * np.exp(1j * delta_phi) * an_fast_3
                + np.exp(1j * deltaP * t) * np.exp(-1j * delta_phi) * an_fast_4
            )

        anP_t = [
            anRotP_fast(time, alpha=alpha, phi1=phi1, phi2=phi2) for time in t_list
        ]
        anM_t = [
            anRotM_fast(time, alpha=alpha, phi1=phi1, phi2=phi2) for time in t_list
        ]
        exp_anP = np.abs(np.array([(x[0] * x[1]).tr() for x in zip(output, anP_t)]))
        exp_anM = np.abs(np.array([(x[0] * x[1]).tr() for x in zip(output, anM_t)]))
        n_P = np.trapz(2 * kappa * (exp_anP), dx=tStep)
        n_M = np.trapz(2 * kappa * (exp_anM), dx=tStep)

        if show_plts:
            fig_r, ax_r = plt.subplots(figsize=(14, 12))
            ax_r.plot(t_list, 2 * kappa * exp_anP, "b--", label="X* photon emission")
            ax_r.plot(t_list, 2 * kappa * exp_anM, "g--", label="Y* photon emission")
            ax.legend(loc="best")
            plt.show()
        print(
            "Rotated cavity basis with params: alpha = {0}, phi1 = {1}, phi2 = {2}".format(
                alpha, phi1, phi2
            )
        )
        print(
            "Total cavity emission in rotated basis:",
            np.round(n_P, 3),
            np.round(n_M, 3),
        )

    return (n_X, n_Y)
