from itertools import chain
from functools import reduce
import numpy as np
import math

def laserCoupling(photonicSpace:bool, ketbras, Omega,gLev,xLev,deltaL,args_list,pulseShape='np.sin(w*t)**2',array=False,amp_array=[], t_array=[]):
    '''
    Create a laser coupling.
    Parameters:
        ketbras - dictionary of ketbras
        Omega - The peak rabi frequency of the pump pulse.
        gLev - The ground atomic atomic level.
        xLev - The excited atomic level.
        deltaL - The detuning of the pump laser.
        args_list - A dictionary of arguments for the qutip simulation.
        pulseShape - The shape of the pump pulse.
        
    Returns:
        (List of cython-ready Hamiltonian terms,
        args_list with relevant parameters added)
    '''
    def kb(x,y):
        return ketbras[str([x,y])]
    deltaL_lab = 'omegaL_{0}{1}'.format(gLev,xLev)
    args_list[deltaL_lab] = deltaL

    if photonicSpace:
        if array:
            assert len(amp_array)==len(t_array)
            output_array_plus=np.empty(len(amp_array),dtype=complex)
            output_array_minus=np.empty(len(amp_array),dtype=complex)
            for i in range(len(t_array)):
                output_array_plus[i]=amp_array[i]*np.exp(1j*deltaL*t_array[i])
                output_array_minus[i]=amp_array[i]*np.exp(-1j*deltaL*t_array[i])
            #no arguments needed here so just return an empty args dictionary
            assert len(output_array_minus)==len(output_array_plus)==len(t_array)
            return (
                     [
                [ -(Omega/2)*(
                        ( kb([gLev,0,0],[xLev,0,0]) + kb([gLev,0,1],[xLev,0,1]) + kb([gLev,1,0],[xLev,1,0]) + kb([gLev,1,1],[xLev,1,1]) )
                ),output_array_plus],
                [ -(Omega/2)*(
                        ( kb([xLev,0,0],[gLev,0,0]) + kb([xLev,0,1],[gLev,0,1]) + kb([xLev,1,0],[gLev,1,0]) + kb([xLev,1,1],[gLev,1,1]) )
                ),output_array_minus]
                ],
                {}
            )
        
        else:
            return (
                [
                [ -(Omega/2)*(
                        ( kb([gLev,0,0],[xLev,0,0]) + kb([gLev,0,1],[xLev,0,1]) + kb([gLev,1,0],[xLev,1,0]) + kb([gLev,1,1],[xLev,1,1]) )
                ),'{0}* np.exp(+1j*{1}*t)'.format(pulseShape,deltaL_lab)],
                [ -(Omega/2)*(
                        ( kb([xLev,0,0],[gLev,0,0]) + kb([xLev,0,1],[gLev,0,1]) + kb([xLev,1,0],[gLev,1,0]) + kb([xLev,1,1],[gLev,1,1]) )
                ),'{0}* np.exp(-1j*{1}*t)'.format(pulseShape,deltaL_lab)]
                ],
            args_list
            )

    else:
        if array:
            output_array_plus=np.empty(len(amp_array),dtype=complex)
            output_array_minus=np.empty(len(amp_array),dtype=complex)
            for i,t in enumerate(t_array):
                output_array_plus[i]=(amp_array[i]*np.exp(1j*deltaL*t))
                output_array_minus[i]=(amp_array[i]*np.exp(-1j*deltaL*t))
            #no arguments needed here so just return an empty args dictionary
            assert len(output_array_minus)==len(output_array_plus)==len(t_array)
            return (
                                [
                [ -(Omega/2)*(
                        ( kb([gLev],[xLev]) + kb([gLev],[xLev]) + kb([gLev],[xLev]) + kb([gLev],[xLev]) )
                ),output_array_plus],
                [ -(Omega/2)*(
                        ( kb([xLev],[gLev]) + kb([xLev],[gLev]) + kb([xLev],[gLev]) + kb([xLev],[gLev]) )
                ),output_array_minus]
                ],
                {}
            )
        else:
            return (
                [
                [ -(Omega/2)*(
                        ( kb([gLev],[xLev]) + kb([gLev],[xLev]) + kb([gLev],[xLev]) + kb([gLev],[xLev]) )
                ),'{0}* np.exp(+1j*{1}*t)'.format(pulseShape,deltaL_lab)],
                [ -(Omega/2)*(
                        ( kb([xLev],[gLev]) + kb([xLev],[gLev]) + kb([xLev],[gLev]) + kb([xLev],[gLev]) )
                ),'{0}* np.exp(-1j*{1}*t)'.format(pulseShape,deltaL_lab)]
                ],
            args_list
            )

def cavityCoupling(ketbras,quant_bas_x,quant_bas_y, deltaP, g0,g,x,deltaC,deltaM,args_list):
    '''
    Create a cavity coupling.

    Parameters:
        ketbras - dictionary of ketbras
        quant_bas_x/y - arbitrary quantisation axes cavity bases contributions as defined in the function quant_axis_cavbasis_mapping
        g0 - The atom-cavity coupling rate.
        gLev - The ground atomic atomic level.
        xLev - The excited atomic level.
        deltaP - birefringence of the cavity in the linear cavity basis
        omegaC - The detuning of the cavity resonance.
        deltaM - The angular momentum change from gLev --> xLev.
        args_list - A dictionary of arguments for the qutip simulation.
        
    Returns:
        (List of cython-ready Hamiltonian terms,
        args_list with relevant parameters added)    
    '''
    def kb(x,y):
        return ketbras[str([x,y])]
    
    pi_x = quant_bas_x[0]
    sigmaP_x = quant_bas_x[1]
    sigmaM_x = quant_bas_x[2]

    pi_y = quant_bas_y[0]
    sigmaP_y = quant_bas_y[1]
    sigmaM_y = quant_bas_y[2]

    deltaC_X = deltaC + deltaP/2
    deltaC_X_lab = 'deltaC_X_{0}{1}'.format(g,x)
    args_list[deltaC_X_lab] = deltaC_X
    deltaC_Y = deltaC - deltaP/2
    deltaC_Y_lab = 'deltaC_Y_{0}{1}'.format(g,x)
    args_list[deltaC_Y_lab] = deltaC_Y
    
    if deltaM==0:
        H_coupling = (
            [
                [ -g0*pi_x*(
                    kb([g,1,0],[x,0,0])+ kb([g,1,1],[x,0,1])
                ),'np.exp(1j*{0}*t)'.format(deltaC_X_lab)],
                
                [ -g0*pi_x*(
                    kb([x,0,0],[g,1,0]) + kb([x,0,1],[g,1,1])
                ),'np.exp(-1j*{0}*t)'.format(deltaC_X_lab)],


                [ -g0*pi_y*(
                    kb([g,0,1],[x,0,0]) + kb([g,1,1],[x,1,0])
                ),'np.exp(1j*{0}*t)'.format(deltaC_Y_lab)], 
        
                [ -g0*pi_y*(
                    kb([x,0,0],[g,0,1]) + kb([x,1,0],[g,1,1])
                ),'np.exp(-1j*{0}*t)'.format(deltaC_Y_lab)]
            ],
            args_list
        )

    elif deltaM==1:
        H_coupling = (
            [
                [ -g0*sigmaP_x*(
                    kb([g,1,0],[x,0,0])+ kb([g,1,1],[x,0,1])
                ),'np.exp(1j*{0}*t)'.format(deltaC_X_lab)],
                
                [ -g0*sigmaP_x*(
                    kb([x,0,0],[g,1,0]) + kb([x,0,1],[g,1,1])
                ),'np.exp(-1j*{0}*t)'.format(deltaC_X_lab)],


                [ -g0*sigmaP_y*(
                    kb([g,0,1],[x,0,0]) + kb([g,1,1],[x,1,0])
                ),'np.exp(1j*{0}*t)'.format(deltaC_Y_lab)], 
        
                [ -g0*sigmaP_y*(
                    kb([x,0,0],[g,0,1]) + kb([x,1,0],[g,1,1])
                ),'np.exp(-1j*{0}*t)'.format(deltaC_Y_lab)]
            ],
            args_list
        )


    elif deltaM==-1:
          H_coupling = (
            [
                [ -g0*sigmaM_x*(
                    kb([g,1,0],[x,0,0])+ kb([g,1,1],[x,0,1])
                ),'np.exp(1j*{0}*t)'.format(deltaC_X_lab)],
                
                [ -g0*sigmaM_x*(
                    kb([x,0,0],[g,1,0]) + kb([x,0,1],[g,1,1])
                ),'np.exp(-1j*{0}*t)'.format(deltaC_X_lab)],


                [ -g0*sigmaM_y*(
                    kb([g,0,1],[x,0,0]) + kb([g,1,1],[x,1,0])
                ),'np.exp(1j*{0}*t)'.format(deltaC_Y_lab)], 
        
                [ -g0*sigmaM_y*(
                    kb([x,0,0],[g,0,1]) + kb([x,1,0],[g,1,1])
                ),'np.exp(-1j*{0}*t)'.format(deltaC_Y_lab)]
            ],
            args_list
        )

    return H_coupling

def couplingsToLaserHamiltonian(ketbras,atomStates, photonicSpace:bool,couplings, rabiFreq, pulseShape='np.sin(w*t)**2', _array=False, _amp=[], _t=[]):
    '''returns Hamiltonian and ham args for a particular laser coupling
    inputs are: ketbras - dictionary of ketbras
                atomStates - dictionary of atomic states
                couplings - requires list of (CG coefficient, ground state string, excited state string, detuning) as specified in the rb_atom_config class with the GetSigmaPlusCouplings(delta).. functions
                rabiFreq - peak Rabi frequency/CG for transitions in angular frequency
                pulseShape - string defining the time dependent function of the laser pulse'''
    hams, args_hams = [], dict()
    for xLev in couplings:
        # Check if this is a coupling between configured states.
        if  xLev[1] in atomStates and xLev[2] in atomStates:
            ham, args_ham = laserCoupling(photonicSpace,ketbras, rabiFreq*xLev[0], xLev[1], xLev[2], xLev[3], 
                                          args_hams, pulseShape, _array, _amp,_t)
            hams.append(ham)
    return list(chain(*hams)), args_ham

def couplingsToCavHamiltonian(quant_bas_x,quant_bas_y,ketbras, atomStates,deltaP, g0, couplings):
    '''returns Hamiltonian and ham args for a particular cavity coupling
        inputs are: quant_bas_x/y - quantisation axes cavity bases contributions as defined in the function quant_axis_cavbasis_mapping 
            quant_args - angles defining the polarisation basis as defined in the function quant_arg requires array of phi1, phi2, alpha, beta
            ketbras - dictionary of ketbras
            atomStates - dictionary of atomic states
            deltaP - polarisation splitting of the cavity modes (birefringence of the)
            pulseShape - string defining the time dependent function of the laser pulse
            couplings - requires list of (CG coefficient, ground state string, excited state string, detuning) as specified in the rb_atom_config class with the GetSigmaPlusCouplings(delta).. functions'''
    hams, args_hams = [], dict()
    for xLev in couplings:
        # Check if this is a coupling between configured states.
        if  xLev[1] in atomStates and xLev[2] in atomStates:
            ham, args_ham = cavityCoupling(ketbras, quant_bas_x,quant_bas_y, deltaP,g0*xLev[0], xLev[1], xLev[2], xLev[3], xLev[4], args_hams) 
            hams.append(ham)
    return list(chain(*hams)), args_ham


