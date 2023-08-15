from vector_functions import *
import numpy as np
from qutip import tensor, destroy, qeye, expect
from matplotlib import pyplot as plt


#define the basis transformation between atomic polarisation basis (defined by quantisation axis, i.e. magnetic field direction) and cavity polarisation basis

def quant_axis_cavbasis_mapping(b_vec,cav_vec):
    #linear cavity basis is spanned by two vectors perpendicular to the cavity axis
    cav_pol_1=perpendicular_vector(cav_vec)
    cav_pol_2=np.cross(cav_pol_1,cav_vec)

    #define plane perpendicular to the quantisation axis
    b_perp_1=perpendicular_vector(b_vec)
    b_perp_2=np.cross(b_perp_1,b_vec)

    #transform between lab and atomic basis
    sigma_plus_lab= 1/np.linalg.norm(b_perp_1+1j*b_perp_2)*(b_perp_1+1j*b_perp_2)
    sigma_minus_lab= 1/np.linalg.norm(b_perp_1-1j*b_perp_2)*(b_perp_1-1j*b_perp_2)

    #express two-dimensional cavity basis in terms of contributions in atomic basis [pi,sigmaPlus,sigmaMinus]
    cav_x_atom=[np.dot(cav_pol_1,b_vec),np.dot(cav_pol_1,sigma_plus_lab),np.dot(cav_pol_1,sigma_minus_lab)]
    cav_y_atom=[np.dot(cav_pol_2,b_vec),np.dot(cav_pol_2,sigma_plus_lab),np.dot(cav_pol_2,sigma_minus_lab)]

    return [cav_x_atom, cav_y_atom]

#define cavity basis transformation used for the cavity emission plotter
def cav_basis_trans(alpha, phi1, phi2):
    alpha=alpha
    beta=np.sqrt(1-alpha**2)
    phi1=phi1
    phi2=phi2
    
    return ([alpha, beta, phi1, phi2])

#define caviy collapse operators  by inputting the photonic field decay rate and an atomStates dictionary
def cav_collapse_ops(kappa,atomStates):
    N=2
    M=len(atomStates)
    # Create the photon operators
    aX = tensor(qeye(M), destroy(N), qeye(N))
    aY = tensor(qeye(M), qeye(N), destroy(N))
    anX = aX.dag()*aX
    anY = aY.dag()*aY
    '''
    The c_op_list is the collapse operators of the system. Namely
    - the rate of photon decay from the cavity
    - spontaneous decay of the excited atomic states
    '''
    # Define collapse operators
    c_op_list = []

    # Cavity decay rate
    c_op_list.append(np.sqrt(2*kappa) * aX)
    c_op_list.append(np.sqrt(2*kappa) * aY)

    return c_op_list

#cavity emission plotter with inputs:
#ketbras: dictionary of ketbras of the atomic states
#atomStates: list of atomic states as dict
#output: output of the simulation
#t_list: list of simulation time
#angles_pol: list of angles for the cavity basis transformation
#kappa: cavity decay rate
#deltaP: detuning between the two cavity polarisation modes

def plotter_cavemission( ketbras, atomStates, output, t_list, angles_pol,kappa,deltaP):
    [alpha, beta, phi1, phi2] = angles_pol
    t=t_list
    output_states=output
    tStep=(t[-1]-t[0])/(len(t)-1)

    #truncating Fock states
    N=2
    M=len(atomStates)
    # Create the photon operators
    aX = tensor(qeye(M), destroy(N), qeye(N))
    aY = tensor(qeye(M), qeye(N), destroy(N))
    anX = aX.dag()*aX
    anY = aY.dag()*aY

    '''
    Consider two polarisation bases:
        - linear cavity basis {|X>,|Y>} (i.e. default)
        - rotated {|X'>,|Y'>}
    Photon number operators for calculating the population in the linear and rotated bases.
    '''

    exp_anX,exp_anY =\
        np.abs( expect([anX,anY], output_states) )
    
    #Calculating rotated cavity basis decay rates
    allAtomicStates = list(atomStates)

    def kb(xLev,y):
        return ketbras[str([xLev,y])]

    an_fast_1  = sum(map(lambda s: kb([s,1,0],[s,1,0]) + kb([s,1,1],[s,1,1]),allAtomicStates))
    an_fast_2  = sum(map(lambda s: kb([s,0,1],[s,0,1]) + kb([s,1,1],[s,1,1]),allAtomicStates))
    an_fast_3  = sum(map(lambda s: kb([s,0,1],[s,1,0]),allAtomicStates))
    an_fast_4  = sum(map(lambda s: kb([s,1,0],[s,0,1]),allAtomicStates))


    def anRotP_fast(t, alpha=alpha, phi1=phi1, phi2=phi2):
        beta = np.sqrt(1-alpha**2)
        delta_phi = phi2 - phi1
        return \
            (alpha**2 * an_fast_1 + beta**2 * an_fast_2) + \
            alpha*beta * (
                np.exp(-1j*deltaP*t) * np.exp(1j*delta_phi) * an_fast_3 + \
                np.exp(1j*deltaP*t) * np.exp(-1j*delta_phi) * an_fast_4
            )

    def anRotM_fast(t, alpha=alpha, phi1=phi1, phi2=phi2):
        beta = np.sqrt(1-alpha**2)
        delta_phi = phi2 - phi1
        return \
            (alpha**2 * an_fast_2 + beta**2 * an_fast_1) - \
            alpha*beta * (
                np.exp(-1j*deltaP*t) * np.exp(1j*delta_phi) * an_fast_3 + \
                np.exp(1j*deltaP*t) * np.exp(-1j*delta_phi) * an_fast_4
            )
    
    anP_t = [anRotP_fast(time, alpha=alpha, phi1=phi1, phi2=phi2) for time in t]
    anM_t = [anRotM_fast(time, alpha=alpha, phi1=phi1, phi2=phi2) for time in t]
    
    exp_anP = np.abs(np.array([(x[0]*x[1]).tr() for x in zip(output_states, anP_t)]))
    exp_anM = np.abs(np.array([(x[0]*x[1]).tr() for x in zip(output_states, anM_t)]))

    # Total cavity emission.
    n_ph = np.trapz(2*kappa*(exp_anX+exp_anY), dx=tStep)
    print('Total cavity emission:', np.round(n_ph,3))
    n_P = np.trapz(2*kappa*(exp_anP), dx=tStep)
    n_M = np.trapz(2*kappa*(exp_anM), dx=tStep)
    n_X = np.trapz(2*kappa*(exp_anX), dx=tStep)
    n_Y = np.trapz(2*kappa*(exp_anY), dx=tStep)

    print("Showing linear cavity basis and modified cavity basis * emission rates, with rotation angles: alpha = {0}, phi1 = {1}, phi2 = {2}".format(alpha, phi1, phi2))

    # Plotting
    fig, (a1,a2) = plt.subplots(2,1,figsize=(14,12))
    a1.plot(t, 2*kappa * exp_anX, 'b', label='X photon emission')
    a1.plot(t, 2*kappa * exp_anY, 'g', label='Y photon emission')
    a1.legend(loc='best')

    a2.plot(t, 2*kappa * exp_anP, 'b', label='X* photon emission')
    a2.plot(t, 2*kappa * exp_anM, 'g', label='Y* photon emission')
    a2.legend(loc='best')
    print("Rotated cavity basis with params: alpha = {0}, phi1 = {1}, phi2 = {2}".format(alpha, phi1, phi2))
    return(fig)
