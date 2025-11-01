from itertools import chain
import numpy as np


def laserCoupling(
    photonicSpace: bool,
    ketbras,
    Omega,
    gLev,
    xLev,
    deltaL,
    args_list,
    pulseShape="np.sin(w*t)**2",
    array=False,
    amp_array=[],
    t_array=[],
    phase_array=[],
):
    """
    Create a laser coupling.
    Parameters:
        photonicSpace - Boolean to determine if the system has photonic Hilbert space for cavity
        ketbras - dictionary of ketbras
        Omega - The peak rabi frequency of the pump pulse in angular frequency
        gLev - The ground atomic atomic level.
        xLev - The excited atomic level.
        deltaL - The detuning of the pump laser in angular frequency.
        args_list - A dictionary of arguments for the qutip simulation.
        pulseShape - The shape of the pump pulse.
        array - Boolean to determine if the pulse envelope is an array of values or a string function.
        amp_array - Array of amplitude timesteps for the pulse envelope.
        t_array - Array of timesteps for the pulse envelope.
        phase_array - Array of detunings of the laser from resonance.

    Returns:
        (List of cython-ready Hamiltonian terms,
        args_list with relevant parameters added)
    """

    def kb(x, y):
        return ketbras[str([x, y])]

    deltaL_lab = "omegaL_{0}{1}".format(gLev, xLev)
    args_list[deltaL_lab] = deltaL

    if array:
        assert len(amp_array) == len(t_array)
        if len(phase_array) == 0:
            phase_array = np.zeros(len(amp_array))
        else:
            assert len(amp_array) == len(phase_array)

        couple_p = np.empty(len(amp_array), dtype=complex)
        couple_m = np.empty(len(amp_array), dtype=complex)
        for i in range(len(t_array)):
            # changed delta is defined to allow the detuning to be varied through the phase array
            changed_delta = deltaL + phase_array[i]
            couple_p[i] = amp_array[i] * np.exp(1j * changed_delta * t_array[i])
            couple_m[i] = amp_array[i] * np.exp(-1j * changed_delta * t_array[i])

        assert len(couple_m) == len(couple_p) == len(t_array)

    if photonicSpace:
        if array:
            return (
                [
                    [
                        -(Omega / 2)
                        * (
                            (
                                kb([gLev, 0, 0], [xLev, 0, 0])
                                + kb([gLev, 0, 1], [xLev, 0, 1])
                                + kb([gLev, 1, 0], [xLev, 1, 0])
                                + kb([gLev, 1, 1], [xLev, 1, 1])
                            )
                        ),
                        couple_p,
                    ],
                    [
                        -(Omega / 2)
                        * (
                            (
                                kb([xLev, 0, 0], [gLev, 0, 0])
                                + kb([xLev, 0, 1], [gLev, 0, 1])
                                + kb([xLev, 1, 0], [gLev, 1, 0])
                                + kb([xLev, 1, 1], [gLev, 1, 1])
                            )
                        ),
                        couple_m,
                    ],
                ],
                {},
            )

        else:
            return (
                [
                    [
                        -(Omega / 2)
                        * (
                            (
                                kb([gLev, 0, 0], [xLev, 0, 0])
                                + kb([gLev, 0, 1], [xLev, 0, 1])
                                + kb([gLev, 1, 0], [xLev, 1, 0])
                                + kb([gLev, 1, 1], [xLev, 1, 1])
                            )
                        ),
                        "{0}* np.exp(+1j*{1}*t)".format(pulseShape, deltaL_lab),
                    ],
                    [
                        -(Omega / 2)
                        * (
                            (
                                kb([xLev, 0, 0], [gLev, 0, 0])
                                + kb([xLev, 0, 1], [gLev, 0, 1])
                                + kb([xLev, 1, 0], [gLev, 1, 0])
                                + kb([xLev, 1, 1], [gLev, 1, 1])
                            )
                        ),
                        "{0}* np.exp(-1j*{1}*t)".format(pulseShape, deltaL_lab),
                    ],
                ],
                args_list,
            )

    else:
        if array:
            return (
                [
                    [-(Omega / 2) * ((kb([gLev], [xLev]))), couple_p],
                    [-(Omega / 2) * ((kb([xLev], [gLev]))), couple_m],
                ],
                {},
            )
        else:
            return (
                [
                    [
                        -(Omega / 2) * ((kb([gLev], [xLev]))),
                        "{0}* np.exp(+1j*{1}*t)".format(pulseShape, deltaL_lab),
                    ],
                    [
                        -(Omega / 2) * ((kb([xLev], [gLev]))),
                        "{0}* np.exp(-1j*{1}*t)".format(pulseShape, deltaL_lab),
                    ],
                ],
                args_list,
            )


def laserDetunedRamanCoupling(
    photonic_space,
    ketbras,
    omega_1,
    delta_1,
    omega_2,
    delta_2,
    g1Lev,
    g2Lev,
    t_array,
    amp_array_1,
    amp_array_2,
    rel_phase=0.0,
    phase_array_1=np.array([]),
    phase_array_2=np.array([]),
):
    """
    Create a detuned Raman coupling of a Pump kind which is detuned -delta from average detuned in a pair.
    Parameters:
        ketbras - dictionary of ketbras
        omega_1 - The Rabi frequency of the first laser in angular frequency.
        delta_1 - The detuning of the first laser in angular frequency.
        omega_2 - The Rabi frequency of the second laser in angular frequency.
        delta_2 - The detuning of the second laser in angular frequency.
        g1Lev - The ground1 atomic atomic level.
        g2Lev - The ground2 atomic level.
        t_array - Array of timesteps for the pulse envelope.
        amp_array_1 - Array of amplitude timesteps for the first pulse envelope.
        amp_array_2 - Array of amplitude timesteps for the second pulse envelope.
        rel_phase - The relative phase between the two lasers.
        phase_array_1 - Array of phases of the first laser.
        phase_array_2 - Array of phases of the second laser.
    """

    def kb(x, y):
        return ketbras[str([x, y])]

    assert len(amp_array_1) == len(amp_array_2) == len(t_array)

    omega_1 = omega_1.astype(complex)
    omega_2 = omega_2.astype(complex)

    omega_1 = omega_1 * amp_array_1
    omega_2 = np.exp(1j * rel_phase) * omega_2 * amp_array_2

    Delta = (delta_1 + delta_2) / 2 * np.ones(len(t_array))
    delta = (delta_1 - delta_2) * np.ones(len(t_array))

    if len(phase_array_1) == 0 and len(phase_array_2) == 0:
        phase_array_1 = phase_array_2 = np.zeros(len(t_array))

    elif len(phase_array_1) == 0 and len(phase_array_2) != 0:
        raise ValueError("phase_array_1 is empty but phase_array_2 is not")
    elif len(phase_array_1) != 0 and len(phase_array_2) == 0:
        raise ValueError("phase_array_1 is empty but phase_array_2 is not")
    else:
        assert (
            len(amp_array_1)
            == len(amp_array_2)
            == len(t_array)
            == len(phase_array_1)
            == len(phase_array_2)
        )
        for i in range(0, len(phase_array_1)):
            omega_1[i] = omega_1[i] * np.exp(1j * phase_array_1[i])
            omega_2[i] = omega_2[i] * np.exp(1j * phase_array_2[i])

    couple_01 = np.zeros(len(amp_array_1), dtype=complex)
    couple_10 = np.zeros(len(amp_array_2), dtype=complex)
    phase_full_array = np.zeros(len(t_array))
    for i, t in enumerate(t_array):
        phase = (
            -1
            / 2
            * (
                -2 * delta[i]
                + (np.abs(omega_2[i]) ** 2 - np.abs(omega_1[i]) ** 2) / (2 * Delta[i])
            )
        )
        phase_full_array[i] = phase
        couple_01[i] = (
            omega_1[i] * np.conj(omega_2[i]) / (2 * Delta[i]) * np.exp(1j * phase * t)
        )
        couple_10[i] = (
            np.conj(omega_1[i]) * omega_2[i] / (2 * Delta[i]) * np.exp(-1j * phase * t)
        )

    if photonic_space:
        return (
            [
                [-(1 / 2) * ((kb([g1Lev, 0, 0], [g2Lev, 0, 0]))), couple_01],
                [-(1 / 2) * ((kb([g2Lev, 0, 0], [g1Lev, 0, 0]))), couple_10],
            ],
            {},
        )
    else:
        return (
            [
                [-(1 / 2) * ((kb([g1Lev], [g2Lev]))), couple_01],
                [-(1 / 2) * ((kb([g2Lev], [g1Lev]))), couple_10],
            ],
            {},
        )


def cavityCoupling(
    ketbras,
    quant_bas_x,
    quant_bas_y,
    deltaP,
    g0,
    g,
    x,
    deltaC,
    deltaM,
    args_list,
    array=False,
    t_array=[],
    phase_array=[],
):
    """
    Create a cavity coupling.

    Parameters:
        ketbras - dictionary of ketbras
        quant_bas_x/y - arbitrary quantisation axes cavity bases contributions as defined in the function quant_axis_cavbasis_mapping
        g0 - The atom-cavity coupling rate.
        gLev - The ground atomic atomic level.
        xLev - The excited atomic level.
        deltaP - birefringence of the cavity in the linear cavity basis in angular frequency.
        omegaC - The detuning of the cavity resonance in angular frequency.
        deltaM - The angular momentum change from gLev --> xLev.
        args_list - A dictionary of arguments for the qutip simulation.
        array - Boolean to determine if the pulse envelope is an array of values or a string function.
        t_array - Array of timesteps for the pulse envelope.
        phase_array - Array of detunings of the laser from resonance.

    Returns:
        (List of cython-ready Hamiltonian terms,
        args_list with relevant parameters added)
    """

    def kb(x, y):
        return ketbras[str([x, y])]

    pi_x = quant_bas_x[0]
    sigmaP_x = quant_bas_x[1]
    sigmaM_x = quant_bas_x[2]

    pi_y = quant_bas_y[0]
    sigmaP_y = quant_bas_y[1]
    sigmaM_y = quant_bas_y[2]

    deltaC_X = deltaC + deltaP / 2
    deltaC_X_lab = "deltaC_X_{0}{1}".format(g, x)
    args_list[deltaC_X_lab] = deltaC_X
    deltaC_Y = deltaC - deltaP / 2
    deltaC_Y_lab = "deltaC_Y_{0}{1}".format(g, x)
    args_list[deltaC_Y_lab] = deltaC_Y

    if array is False:
        if deltaM == 0:
            H_coupling = (
                [
                    [
                        -g0
                        * pi_x
                        * (kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1])),
                        "np.exp(1j*{0}*t)".format(deltaC_X_lab),
                    ],
                    [
                        -g0
                        * pi_x
                        * (kb([x, 0, 0], [g, 1, 0]) + kb([x, 0, 1], [g, 1, 1])),
                        "np.exp(-1j*{0}*t)".format(deltaC_X_lab),
                    ],
                    [
                        -g0
                        * pi_y
                        * (kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0])),
                        "np.exp(1j*{0}*t)".format(deltaC_Y_lab),
                    ],
                    [
                        -g0
                        * pi_y
                        * (kb([x, 0, 0], [g, 0, 1]) + kb([x, 1, 0], [g, 1, 1])),
                        "np.exp(-1j*{0}*t)".format(deltaC_Y_lab),
                    ],
                ],
                args_list,
            )

        elif deltaM == 1:
            H_coupling = (
                [
                    [
                        -g0
                        * sigmaP_x
                        * (kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1])),
                        "np.exp(1j*{0}*t)".format(deltaC_X_lab),
                    ],
                    [
                        -g0
                        * sigmaP_x
                        * (kb([x, 0, 0], [g, 1, 0]) + kb([x, 0, 1], [g, 1, 1])),
                        "np.exp(-1j*{0}*t)".format(deltaC_X_lab),
                    ],
                    [
                        -g0
                        * sigmaP_y
                        * (kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0])),
                        "np.exp(1j*{0}*t)".format(deltaC_Y_lab),
                    ],
                    [
                        -g0
                        * sigmaP_y
                        * (kb([x, 0, 0], [g, 0, 1]) + kb([x, 1, 0], [g, 1, 1])),
                        "np.exp(-1j*{0}*t)".format(deltaC_Y_lab),
                    ],
                ],
                args_list,
            )

        elif deltaM == -1:
            H_coupling = (
                [
                    [
                        -g0
                        * sigmaM_x
                        * (kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1])),
                        "np.exp(1j*{0}*t)".format(deltaC_X_lab),
                    ],
                    [
                        -g0
                        * sigmaM_x
                        * (kb([x, 0, 0], [g, 1, 0]) + kb([x, 0, 1], [g, 1, 1])),
                        "np.exp(-1j*{0}*t)".format(deltaC_X_lab),
                    ],
                    [
                        -g0
                        * sigmaM_y
                        * (kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0])),
                        "np.exp(1j*{0}*t)".format(deltaC_Y_lab),
                    ],
                    [
                        -g0
                        * sigmaM_y
                        * (kb([x, 0, 0], [g, 0, 1]) + kb([x, 1, 0], [g, 1, 1])),
                        "np.exp(-1j*{0}*t)".format(deltaC_Y_lab),
                    ],
                ],
                args_list,
            )

        return H_coupling

    else:
        if len(phase_array) == 0:
            phase_array = np.zeros(len(t_array))

        assert len(phase_array) == len(t_array)

        output_array_plus_x = np.empty(len(t_array), dtype=complex)
        output_array_minus_x = np.empty(len(t_array), dtype=complex)
        output_array_minus_y = np.empty(len(t_array), dtype=complex)
        output_array_plus_y = np.empty(len(t_array), dtype=complex)

        for i, t in enumerate(t_array):
            # changed delta is defined to allow the detuning to be varied through the phase array
            delta_x = phase_array[i] + deltaC_X
            delta_y = phase_array[i] + deltaC_Y
            output_array_plus_x[i] = np.exp(1j * delta_x * t)
            output_array_minus_x[i] = np.exp(-1j * delta_x * t)
            output_array_plus_y[i] = np.exp(1j * delta_y * t)
            output_array_minus_y[i] = np.exp(-1j * delta_y * t)

        assert len(output_array_minus_y) == len(output_array_plus_y) == len(t_array)
        if deltaM == 0:
            H_coupling = (
                [
                    [
                        -g0
                        * pi_x
                        * (kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1])),
                        output_array_plus_x,
                    ],
                    [
                        -g0
                        * pi_x
                        * (kb([x, 0, 0], [g, 1, 0]) + kb([x, 0, 1], [g, 1, 1])),
                        output_array_minus_x,
                    ],
                    [
                        -g0
                        * pi_y
                        * (kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0])),
                        output_array_plus_y,
                    ],
                    [
                        -g0
                        * pi_y
                        * (kb([x, 0, 0], [g, 0, 1]) + kb([x, 1, 0], [g, 1, 1])),
                        output_array_minus_y,
                    ],
                ],
                {},
            )

        elif deltaM == 1:
            H_coupling = (
                [
                    [
                        -g0
                        * sigmaP_x
                        * (kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1])),
                        output_array_plus_x,
                    ],
                    [
                        -g0
                        * sigmaP_x
                        * (kb([x, 0, 0], [g, 1, 0]) + kb([x, 0, 1], [g, 1, 1])),
                        output_array_minus_x,
                    ],
                    [
                        -g0
                        * sigmaP_y
                        * (kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0])),
                        output_array_plus_y,
                    ],
                    [
                        -g0
                        * sigmaP_y
                        * (kb([x, 0, 0], [g, 0, 1]) + kb([x, 1, 0], [g, 1, 1])),
                        output_array_minus_y,
                    ],
                ],
                {},
            )

        elif deltaM == -1:
            H_coupling = (
                [
                    [
                        -g0
                        * sigmaM_x
                        * (kb([g, 1, 0], [x, 0, 0]) + kb([g, 1, 1], [x, 0, 1])),
                        output_array_plus_x,
                    ],
                    [
                        -g0
                        * sigmaM_x
                        * (kb([x, 0, 0], [g, 1, 0]) + kb([x, 0, 1], [g, 1, 1])),
                        output_array_minus_x,
                    ],
                    [
                        -g0
                        * sigmaM_y
                        * (kb([g, 0, 1], [x, 0, 0]) + kb([g, 1, 1], [x, 1, 0])),
                        output_array_plus_y,
                    ],
                    [
                        -g0
                        * sigmaM_y
                        * (kb([x, 0, 0], [g, 0, 1]) + kb([x, 1, 0], [g, 1, 1])),
                        output_array_minus_y,
                    ],
                ],
                {},
            )

        return H_coupling


def couplingsToLaserHamiltonian(
    ketbras,
    atomStates,
    photonicSpace: bool,
    couplings,
    rabiFreq,
    pulseShape="np.sin(w*t)**2",
    _array=False,
    _amp=[],
    _t=[],
    _phase=[],
):
    """returns Hamiltonian and ham args for a particular laser coupling
    inputs are: ketbras - dictionary of ketbras
                atomStates - dictionary of atomic states
                photonicSpace - Boolean to determine if the system has photonic Hilbert space for cavity
                couplings - requires list of (CG coefficient, ground state string, excited state string, detuning)
                            as specified in the rb_atom_config class with the GetSigmaPlusCouplings(delta).. functions
                rabiFreq - peak Rabi frequency/CG for transitions in angular frequency
                pulseShape - string defining the time dependent function of the laser pulse
                _array - boolean to determine if the pulse envelope is an array of values or a string function
                _amp - array of amplitude timesteps for the pulse envelope
                _t - array of timesteps for the pulse envelope
                _phase - array of detunings of the laser from resonance"""
    hams, args_hams = [], dict()
    for coupling in couplings:
        # Check if this is a coupling between configured states.
        if coupling[1] in atomStates and coupling[2] in atomStates:
            ham, args_ham = laserCoupling(
                photonicSpace,
                ketbras,
                rabiFreq * coupling[0],
                coupling[1],
                coupling[2],
                coupling[3],
                args_hams,
                pulseShape,
                _array,
                _amp,
                _t,
                _phase,
            )
            hams.append(ham)
            if isinstance(args_ham, dict):
                args_hams.update(args_ham)
    return list(chain(*hams)), args_hams


def couplingsToCavHamiltonian(
    quant_bas_x,
    quant_bas_y,
    ketbras,
    atomStates,
    deltaP,
    g0,
    couplings,
    _array=False,
    _t=[],
    _phase=[],
):
    """returns Hamiltonian and ham args for a particular cavity coupling
    inputs are: quant_bas_x/y - quantisation axes cavity bases contributions as defined in the function quant_axis_cavbasis_mapping
        ketbras - dictionary of ketbras
        atomStates - dictionary of atomic states
        deltaP - polarisation splitting of the cavity modes (birefringence of the)
        g0 - atom-cavity coupling rate in angular frequency
        couplings - requires list of (CG coefficient, ground state string, excited state string, detuning)
                    as specified in the rb_atom_config class with the GetSigmaPlusCouplings(delta).. functions
        _array - boolean to determine if the pulse envelope is an array of values or a string function
        _t - array of timesteps for the pulse envelope
        _phase - array of detunings of the laser from resonance"""

    hams, args_hams = [], dict()
    for coupling in couplings:
        # Check if this is a coupling between configured states.
        if coupling[1] in atomStates and coupling[2] in atomStates:
            ham, args_ham = cavityCoupling(
                ketbras,
                quant_bas_x,
                quant_bas_y,
                deltaP,
                g0 * coupling[0],
                coupling[1],
                coupling[2],
                coupling[3],
                coupling[4],
                args_hams,
                _array,
                _t,
                _phase,
            )
            hams.append(ham)

            # Update args_hams with elements from args_ham
            if isinstance(args_ham, dict):
                args_hams.update(args_ham)
    return list(chain(*hams)), args_hams


def couplingsToFarDetunedRamanHamiltonian(
    photonic_space,
    ketbras,
    atomStates,
    couplings1,
    couplings2,
    omega_1,
    omega_2,
    t_array,
    amp_array_1,
    amp_array_2,
    rel_phase=0.0,
    det_array_1=np.array([]),
    det_array_2=np.array([]),
):
    """returns Hamiltonian and ham args for a particular fat detuned Raman laser coupling
    inputs are: photonic_space - Boolean to determine if the system has photonic Hilbert space for cavity
                ketbras - dictionary of ketbras
                atomStates - dictionary of atomic states
                couplings1 - requires list of (CG coefficient, ground state string, excited state string, detuning)
                            as specified in the rb_atom_config class with the GetSigmaPlusCouplings(delta).. functions
                couplings2 - requires list of (CG coefficient, ground state string, excited state string, detuning)
                            as specified in the rb_atom_config class with the GetSigmaPlusCouplings(delta).. functions
                omega_1 - The Rabi frequency of the first laser in angular frequency.
                delta_1 - The detuning of the first laser in angular frequency.
                omega_2 - The Rabi frequency of the second laser in angular frequency.
                delta_2 - The detuning of the second laser in angular frequency.
                t_array - Array of timesteps for the pulse envelope.
                amp_array_1 - Array of amplitude timesteps for the first pulse envelope.
                amp_array_2 - Array of amplitude timesteps for the second pulse envelope.
                det_array_1 - Array of detunings of the first laser from resonance.
                det_array_2 - Array of detunings of the second laser from resonance."""

    # print(f"couplings1: {couplings1}")
    # print(f"couplings2: {couplings2}")

    # couplings are tuples of the form (CG_coeff, g_level, x_level, detuning, polarisation)

    hams, args_hams = [], dict()

    # Helper function to process unique pairs
    def process_unique_pairs(
        couplings_a,
        couplings_b,
        omega_a,
        omega_b,
        amp_array_a,
        amp_array_b,
        rel_phase,
        det_array_a,
        det_array_b,
    ):
        # check if the two lists of couplings are the same
        same_couplings = couplings_a == couplings_b

        # Ensure couplings_a is the shorter or equal-length list
        if len(couplings_a) > len(couplings_b):
            couplings_a, couplings_b = couplings_b, couplings_a
            omega_a, omega_b = omega_b, omega_a
            amp_array_a, amp_array_b = amp_array_b, amp_array_a
            det_array_a, det_array_b = det_array_b, det_array_a

        for i in range(len(couplings_a)):
            if same_couplings:
                start = i + 1
            else:
                start = 0

            for j in range(start, len(couplings_b)):

                coup_a = couplings_a[i]
                coup_b = couplings_b[j]

                # Skip duplicate pairs if both lists are the same
                if coup_a is coup_b:
                    continue

                # Compare the second indexed elements
                if (
                    coup_a[2] == coup_b[2]
                    and coup_a[1] in atomStates
                    and coup_b[1] in atomStates
                ):

                    ham, args_ham = laserDetunedRamanCoupling(
                        photonic_space,
                        ketbras,
                        omega_a * coup_a[0],
                        coup_a[3],
                        omega_b * coup_b[0],
                        coup_b[3],
                        coup_a[1],
                        coup_b[1],
                        t_array,
                        amp_array_a,
                        amp_array_b,
                        rel_phase,
                        det_array_a,
                        det_array_b,
                    )
                    hams.append(ham)
                    args_hams.update(args_ham)

    # Process unique pairs within couplings1
    process_unique_pairs(
        couplings1,
        couplings1,
        omega_1,
        omega_1,
        amp_array_1,
        amp_array_1,
        rel_phase,
        det_array_1,
        det_array_1,
    )

    # Process unique pairs within couplings2
    process_unique_pairs(
        couplings2,
        couplings2,
        omega_2,
        omega_2,
        amp_array_2,
        amp_array_2,
        rel_phase,
        det_array_2,
        det_array_2,
    )

    # Process unique pairs between couplings1 and couplings2
    process_unique_pairs(
        couplings1,
        couplings2,
        omega_1,
        omega_2,
        amp_array_1,
        amp_array_2,
        rel_phase,
        det_array_1,
        det_array_2,
    )

    return list(chain(*hams)), args_hams
