import csv
import matplotlib.pyplot as plt
from itertools import chain, product
import numpy as np
from qutip import tensor, qeye, basis
from ham_sim_source import couplingsToCavHamiltonian, couplingsToLaserHamiltonian
from ketbra_config import rb_atom_ketbras

class rb_atom_config:
    '''
    Import coupling factors and energy level splittings from file.

    The notation used is to denote the ground/excited 'F' levels with gLev'F'/xLev'F'
    The magnetic 'mF' sublevels -3,-2,-1,0,1,2,3 are denoted MMM,MM,M,[nothing],P,PP,PPP.
    Examples: the F'=2,mF'=3 sublevel is denoted x2PPP,
            the F=1,mF'=-1 sublevel is denoted g1M.
            
    Note: By default the parameter file has the energy splitting given wrt the F'=2 level. 
    '''  
    def __init__(self, bfieldsplit:str, ketbras:rb_atom_ketbras):
        self.bfieldsplit=bfieldsplit
        self.ketbras = ketbras
        self.photonic_space = ketbras.photonicSpace
        imports = []
        imports_d1 = []

        with open("d2_params/exp_params_{0}MHz.csv".format(self.bfieldsplit)) as file:
            reader = csv.reader(file)
            for row in reader:
                imports.append(row)
                
        imports = dict(map(lambda xLev: (str(xLev[0]), float(xLev[1])), imports))
 
        with open("d1_params/exp_params_D1_{0}MHz.csv".format(self.bfieldsplit)) as file:
            reader = csv.reader(file)
            for row in reader:
                imports_d1.append(row)
                
        imports_d1 = dict(map(lambda x: (str(x[0]), float(x[1])), imports_d1))

        self.deltaZ,self.deltaEx3,self.deltaEx1,self.deltaEx0,\
        self.deltaZx3MMM,\
        self.deltaZx3MM,self.deltaZx2MM,\
        self.deltaZx3M,self.deltaZx2M,self.deltaZx1M,\
        self.deltaZx3,self.deltaZx2,self.deltaZx1,self.deltaZx0,\
        self.deltaZx3P,self.deltaZx2P,self.deltaZx1P,\
        self.deltaZx3PP,self.deltaZx2PP,\
        self.deltaZx3PPP = \
            [imports[delta]*2*np.pi for delta in 
                ["deltaZ", "deltaEx3","deltaEx1","deltaEx0",
                "deltaZx3MMM",
                "deltaZx3MM","deltaZx2MM",
                "deltaZx3M","deltaZx2M", "deltaZx1M",
                "deltaZx3","deltaZx2", "deltaZx1", "deltaZx0",
                "deltaZx3P","deltaZx2P", "deltaZx1P",
                "deltaZx3PP","deltaZx2PP",
                "deltaZx3PPP"]
            ]

        self.CGg1Mx3MM, self.CGg1Mx2MM, \
        self.CGg1x3M, self.CGg1x2M, self.CGg1x1M, self.CGg1Mx3M, self.CGg1Mx2M, self.CGg1Mx1M, \
        self.CGg1Px3, self.CGg1Px2, self.CGg1Px1, self.CGg1Px0, self.CGg1x3, self.CGg1x2, self.CGg1x1, self.CGg1x0, self.CGg1Mx3, self.CGg1Mx2, self.CGg1Mx1, self.CGg1Mx0, \
        self.CGg1Px3P, self.CGg1Px2P, self.CGg1Px1P, self.CGg1x3P, self.CGg1x2P, self.CGg1x1P, \
        self.CGg1Px3PP,self.CGg1Px2PP = [imports[CG] for CG in [
            "CGg1Mx3MM", "CGg1Mx2MM",
            "CGg1x3M", "CGg1x2M", "CGg1x1M", "CGg1Mx3M", "CGg1Mx2M", "CGg1Mx1M",
            "CGg1Px3", "CGg1Px2", "CGg1Px1", "CGg1Px0", "CGg1x3", "CGg1x2", "CGg1x1", "CGg1x0", "CGg1Mx3", "CGg1Mx2", "CGg1Mx1", "CGg1Mx0",
            "CGg1Px3P", "CGg1Px2P", "CGg1Px1P", "CGg1x3P", "CGg1x2P", "CGg1x1P", 
            "CGg1Px3PP","CGg1Px2PP"]]

        self.CGg2MMx3MMM, \
        self.CGg2Mx3MM, self.CGg2Mx2MM, self.CGg2MMx3MM, self.CGg2MMx2MM, \
        self.CGg2x3M, self.CGg2x2M, self.CGg2x1M, self.CGg2Mx3M, self.CGg2Mx2M, self.CGg2Mx1M, self.CGg2MMx3M, self.CGg2MMx2M, self.CGg2MMx1M, \
        self.CGg2Px3, self.CGg2Px2, self.CGg2Px1, self.CGg2Px0, self.CGg2x3,  self.CGg2x2, self.CGg2x1, self.CGg2x0, self.CGg2Mx3, self.CGg2Mx2, self.CGg2Mx1, self.CGg2Mx0, \
        self.CGg2PPx3P, self.CGg2PPx2P, self.CGg2PPx1P, self.CGg2Px3P, self.CGg2Px2P,  self.CGg2Px1P, self.CGg2x3P, self.CGg2x2P, self.CGg2x1P, \
        self.CGg2PPx3PP, self.CGg2PPx2PP, self.CGg2Px3PP, self.CGg2Px2PP, \
        self.CGg2PPx3PPP = [imports[CG] for CG in [
            "CGg2MMx3MMM",
            "CGg2Mx3MM","CGg2Mx2MM", "CGg2MMx3MM", "CGg2MMx2MM",
            "CGg2x3M", "CGg2x2M", "CGg2x1M", "CGg2Mx3M", "CGg2Mx2M", "CGg2Mx1M", "CGg2MMx3M", "CGg2MMx2M", "CGg2MMx1M",
            "CGg2Px3", "CGg2Px2", "CGg2Px1", "CGg2Px0", "CGg2x3", "CGg2x2", "CGg2x1", "CGg2x0", "CGg2Mx3", "CGg2Mx2", "CGg2Mx1", "CGg2Mx0",
            "CGg2PPx3P", "CGg2PPx2P", "CGg2PPx1P", "CGg2Px3P", "CGg2Px2P", "CGg2Px1P", "CGg2x3P", "CGg2x2P", "CGg2x1P",
            "CGg2PPx3PP", "CGg2PPx2PP", "CGg2Px3PP", "CGg2Px2PP",
            "CGg2PPx3PPP"]]

        self.deltaZ_d1,self.deltaEx1_d1,\
        self.deltaZx2MM_d1,\
        self.deltaZx2M_d1,self.deltaZx1M_d1,\
        self.deltaZx2_d1,self.deltaZx1_d1,\
        self.deltaZx2P_d1,self.deltaZx1P_d1,\
        self.deltaZx2PP_d1 = \
            [imports_d1[delta_d1]*2*np.pi for delta_d1 in 
                ["deltaZ","deltaEx1",
                "deltaZx2MM",
                "deltaZx2M", "deltaZx1M",
                "deltaZx2", "deltaZx1",
                "deltaZx2P", "deltaZx1P",
                "deltaZx2PP",
                ]
            ]

        self.CG_d1g1Mx2MM, \
        self.CG_d1g1x2M, self.CG_d1g1x1M, self.CG_d1g1Mx2M, self.CG_d1g1Mx1M, \
        self.CG_d1g1Px2, self.CG_d1g1Px1, self.CG_d1g1x2, self.CG_d1g1x1,  self.CG_d1g1Mx2, self.CG_d1g1Mx1, \
        self.CG_d1g1Px2P, self.CG_d1g1Px1P, self.CG_d1g1x2P, self.CG_d1g1x1P, \
        self.CG_d1g1Px2PP = [imports_d1[CG_d1] for CG_d1 in [
            "CGg1Mx2MM",
            "CGg1x2M", "CGg1x1M", "CGg1Mx2M", "CGg1Mx1M",
            "CGg1Px2", "CGg1Px1",  "CGg1x2", "CGg1x1",  "CGg1Mx2", "CGg1Mx1", 
            "CGg1Px2P", "CGg1Px1P", "CGg1x2P", "CGg1x1P", 
            "CGg1Px2PP"]]

        self.CG_d1g2Mx2MM, self.CG_d1g2MMx2MM, \
        self.CG_d1g2x2M, self.CG_d1g2x1M,self.CG_d1g2Mx2M, self.CG_d1g2Mx1M, self.CG_d1g2MMx2M, self.CG_d1g2MMx1M, \
        self.CG_d1g2Px2, self.CG_d1g2Px1,self.CG_d1g2x2, self.CG_d1g2x1, self.CG_d1g2Mx2, self.CG_d1g2Mx1,  \
        self.CG_d1g2PPx2P, self.CG_d1g2PPx1P, self.CG_d1g2Px2P,  self.CG_d1g2Px1P, self.CG_d1g2x2P, self.CG_d1g2x1P, \
        self.CG_d1g2PPx2PP, self.CG_d1g2Px2PP = [imports_d1[CG_d1] for CG_d1 in [
            "CGg2Mx2MM",  "CGg2MMx2MM",
            "CGg2x2M", "CGg2x1M","CGg2Mx2M", "CGg2Mx1M",  "CGg2MMx2M", "CGg2MMx1M",
            "CGg2Px2", "CGg2Px1", "CGg2x2", "CGg2x1",  "CGg2Mx2", "CGg2Mx1",
            "CGg2PPx2P", "CGg2PPx1P", "CGg2Px2P",  "CGg2Px1P", "CGg2x2P", "CGg2x1P",
            "CGg2PPx2PP",  "CGg2Px2PP"]]\

    def getrb_splittings_couplings(self):
        
        return(    [[self.deltaZ,self.deltaEx3,self.deltaEx1,self.deltaEx0,\
        self.deltaZx3MMM,\
        self.deltaZx3MM,self.deltaZx2MM,\
        self.deltaZx3M,self.deltaZx2M,self.deltaZx1M,\
        self.deltaZx3,self.deltaZx2,self.deltaZx1,self.deltaZx0,\
        self.deltaZx3P,self.deltaZx2P,self.deltaZx1P,\
        self.deltaZx3PP,self.deltaZx2PP,\
        self.deltaZx3PPP,
        self.deltaZ_d1,self.deltaEx1_d1,\
        self.deltaZx2MM_d1,\
        self.deltaZx2M_d1,self.deltaZx1M_d1,\
        self.deltaZx2_d1,self.deltaZx1_d1,\
        self.deltaZx2P_d1,self.deltaZx1P_d1,\
        self.deltaZx2PP_d1],
        [self.CGg1Mx3MM, self.CGg1Mx2MM, \
        self.CGg1x3M, self.CGg1x2M, self.CGg1x1M, self.CGg1Mx3M, self.CGg1Mx2M, self.CGg1Mx1M, \
        self.CGg1Px3, self.CGg1Px2, self.CGg1Px1, self.CGg1Px0, self.CGg1x3, self.CGg1x2, self.CGg1x1, self.CGg1x0, self.CGg1Mx3, self.CGg1Mx2, self.CGg1Mx1, self.CGg1Mx0, \
        self.CGg1Px3P, self.CGg1Px2P, self.CGg1Px1P, self.CGg1x3P, self.CGg1x2P, self.CGg1x1P, \
        self.CGg1Px3PP,self.CGg1Px2PP,
        self.CGg2MMx3MMM, \
        self.CGg2Mx3MM, self.CGg2Mx2MM, self.CGg2MMx3MM, self.CGg2MMx2MM, \
        self.CGg2x3M, self.CGg2x2M, self.CGg2x1M, self.CGg2Mx3M, self.CGg2Mx2M, self.CGg2Mx1M, self.CGg2MMx3M, self.CGg2MMx2M, self.CGg2MMx1M, \
        self.CGg2Px3, self.CGg2Px2, self.CGg2Px1, self.CGg2Px0, self.CGg2x3,  self.CGg2x2, self.CGg2x1, self.CGg2x0, self.CGg2Mx3, self.CGg2Mx2, self.CGg2Mx1, self.CGg2Mx0, \
        self.CGg2PPx3P, self.CGg2PPx2P, self.CGg2PPx1P, self.CGg2Px3P, self.CGg2Px2P,  self.CGg2Px1P, self.CGg2x3P, self.CGg2x2P, self.CGg2x1P, \
        self.CGg2PPx3PP, self.CGg2PPx2PP, self.CGg2Px3PP, self.CGg2Px2PP, \
        self.CGg2PPx3PPP,
        self.CG_d1g1Mx2MM, \
        self.CG_d1g1x2M, self.CG_d1g1x1M, self.CG_d1g1Mx2M, self.CG_d1g1Mx1M, \
        self.CG_d1g1Px2, self.CG_d1g1Px1, self.CG_d1g1x2, self.CG_d1g1x1,  self.CG_d1g1Mx2, self.CG_d1g1Mx1, \
        self.CG_d1g1Px2P, self.CG_d1g1Px1P, self.CG_d1g1x2P, self.CG_d1g1x1P, \
        self.CG_d1g1Px2PP, \
        self.CG_d1g2Mx2MM, self.CG_d1g2MMx2MM, \
        self.CG_d1g2x2M, self.CG_d1g2x1M,self.CG_d1g2Mx2M, self.CG_d1g2Mx1M, self.CG_d1g2MMx2M, self.CG_d1g2MMx1M, \
        self.CG_d1g2Px2, self.CG_d1g2Px1,self.CG_d1g2x2, self.CG_d1g2x1, self.CG_d1g2Mx2, self.CG_d1g2Mx1,  \
        self.CG_d1g2PPx2P, self.CG_d1g2PPx1P, self.CG_d1g2Px2P,  self.CG_d1g2Px1P, self.CG_d1g2x2P, self.CG_d1g2x1P, \
        self.CG_d1g2PPx2PP, self.CG_d1g2Px2PP]])

    @staticmethod     
    def getrb_xlvls():
        # List the default excited levels
        xlvls = [
        'x0',
        'x1M','x1','x1P',
        'x2MM','x2M','x2','x2P','x2PP',
        'x3MMM', 'x3MM','x3M','x3','x3P','x3PP', 'x3PPP',
        'x1M_d1','x1_d1','x1P_d1',
        'x2MM_d1','x2M_d1','x2_d1','x2P_d1','x2PP_d1'
        ] 
        return xlvls
    
    @staticmethod     
    def getrb_groundstates():
        #list default atomic ground states
        atomStates = {
            "g1M":0, "g1":1, "g1P":2, # F=1,mF=-1,0,+1 respectively
            "g2MM":3, "g2M":4, "g2":5, "g2P":6, "g2PP":7 # F=2,mF=-2,..,+2 respectively
        }
        return atomStates
    
    @staticmethod     
    def getrb_rates():    
        # List the coupling rates of the system.
        #   gamma:  Decay of the atomic amplitude listed for d1 and d2 transitions.
        # d: Dipole moment of either d1 or d2 transition
        gamma_d2 = 3 * 2.*np.pi
        gamma_d1 = 5.746*np.pi
        d_d2 = 3.584*10**(-29)
        d_d1 = 2.537*10**(-29) 
        return ([gamma_d2,gamma_d1,d_d2,d_d1])
           
    def spont_em_ops(self, atomStates):
        #this function returns a list of collapse operators for spontaneous emission via the d1 and d2 lines respectively as an array
        N=2 # Number of Fock states
        M = len(atomStates)    
        [gamma_d1, gamma_d2]=[self.getrb_rates()[0], self.getrb_rates()[1]]
        
    #write function for appending sponEmChannels based on xLvls and atomStates for more modularity

        spontEmmChannels = [
            # |F',mF'> --> |F=1,mF=-1>
            ('g1M','x0',self.CGg1Mx0),
            ('g1M','x1M',self.CGg1Mx1M),('g1M','x1',self.CGg1Mx1), 
            ('g1M','x2MM',self.CGg1Mx2MM),('g1M','x2M',self.CGg1Mx2M),('g1M','x2',self.CGg1Mx2),
            ('g1M','x3MM',self.CGg1Mx3MM),('g1M','x3M',self.CGg1Mx3M),('g1M','x3',self.CGg1Mx3),

            # |F',mF'> --> |F=1,mF=0>
            ('g1','x0',self.CGg1x0),
            ('g1','x1M',self.CGg1x1M),('g1','x1',self.CGg1x1),('g1','x1P',self.CGg1x1P),
            ('g1','x2M',self.CGg1x2M),('g1','x2',self.CGg1x2),('g1','x2P',self.CGg1x2P),
            ('g1','x3M',self.CGg1x3M),('g1','x3',self.CGg1x3),('g1','x3P',self.CGg1x3P),

            # |F',mF'> --> |F=1,mF=+1>
            ('g1P','x0',self.CGg1Px0),
            ('g1P','x1',self.CGg1Px1),('g1P','x1P',self.CGg1Px1P),
            ('g1P','x2',self.CGg1Px2),('g1P','x2P',self.CGg1Px2P),('g1P','x2PP',self.CGg1Px2PP),
            ('g1P','x3',self.CGg1Px3),('g1P','x3P',self.CGg1Px3P),('g1P','x3PP',self.CGg1Px3PP),

            # |F',mF'> --> |F=2,mF=-2>
            ('g2MM','x1M',self.CGg2MMx1M),
            ('g2MM','x2MM',self.CGg2MMx2MM),('g2MM','x2M',self.CGg2MMx2M),
            ('g2MM','x3MMM',self.CGg2MMx3MMM),('g2MM','x3MM',self.CGg2MMx3MM),('g2MM','x3M',self.CGg2MMx3M),

            # |F',mF'> --> |F=2,mF=-1>
            ('g2M','x0',self.CGg2Mx0),
            ('g2M','x1M',self.CGg2Mx1M),('g2M','x1',self.CGg2Mx1),
            ('g2M','x2MM',self.CGg2Mx2MM),('g2M','x2M',self.CGg2Mx2M),('g2M','x2',self.CGg2Mx2),
            ('g2M','x3MM',self.CGg2Mx3MM),('g2M','x3M',self.CGg2Mx3M),('g2M','x3',self.CGg2Mx3),

            # |F',mF'> --> |F=2,mF=0>
            ('g2','x0',self.CGg2x0),
            ('g2','x1M',self.CGg2x1M),('g2','x1',self.CGg2x1),('g2','x1P',self.CGg2x1P),
            ('g2','x2M',self.CGg2x2M),('g2','x2',self.CGg2x2),('g2','x2P',self.CGg2x2P),
            ('g2','x3M',self.CGg2x3M),('g2','x3',self.CGg2x3),('g2','x3P',self.CGg2x3P),

            # |F',mF'> --> |F=2,mF=+1>
            ('g2P','x0',self.CGg2Px0),
            ('g2P','x1',self.CGg2Px1),('g2P','x1P',self.CGg2Px1P),
            ('g2P','x2',self.CGg2Px2),('g2P','x2P',self.CGg2Px2P),('g2P','x2PP',self.CGg2Px2PP),
            ('g2P','x3',self.CGg2Px3),('g2P','x3P',self.CGg2Px3P),('g2P','x3PP',self.CGg2Px3PP),

            # |F',mF'> --> |F=2,mF=+2>
            ('g2PP','x1P',self.CGg2PPx1P),
            ('g2PP','x2P',self.CGg2PPx2P),('g2PP','x2PP',self.CGg2PPx2PP),
            ('g2PP','x3P',self.CGg2PPx3P),('g2PP','x3PP',self.CGg2PPx3PP),('g2PP','x3PPP',self.CGg2PPx3PPP)
        ]

        spontEmmChannels_d1 = [
                            # |F',mF'> --> |F=1,mF=-1>
                            ('g1M','x1M_d1',self.CG_d1g1Mx1M),('g1M','x1_d1',self.CG_d1g1Mx1), 
                            ('g1M','x2MM_d1',self.CG_d1g1Mx2MM),('g1M','x2M_d1',self.CG_d1g1Mx2M),('g1M','x2_d1',self.CG_d1g1Mx2),
            
                            # |F',mF'> --> |F=1,mF=0>
                            ('g1','x1M_d1',self.CG_d1g1x1M),('g1','x1_d1',self.CG_d1g1x1),('g1','x1P_d1',self.CG_d1g1x1P),
                            ('g1','x2M_d1',self.CG_d1g1x2M),('g1','x2_d1',self.CG_d1g1x2),('g1','x2P_d1',self.CG_d1g1x2P),
            
                            # |F',mF'> --> |F=1,mF=+1>
                            ('g1P','x1_d1',self.CG_d1g1Px1),('g1P','x1P_d1',self.CG_d1g1Px1P),
                            ('g1P','x2_d1',self.CG_d1g1Px2),('g1P','x2P_d1',self.CG_d1g1Px2P),('g1P','x2PP_d1',self.CG_d1g1Px2PP),
            
                            # |F',mF'> --> |F=2,mF=-2>
                            ('g2MM','x1M_d1',self.CG_d1g2MMx1M),
                            ('g2MM','x2MM_d1',self.CG_d1g2MMx2MM),('g2MM','x2M_d1',self.CG_d1g2MMx2M),
            
                            # |F',mF'> --> |F=2,mF=-1>
                            ('g2M','x1M_d1',self.CG_d1g2Mx1M),('g2M','x1_d1',self.CG_d1g2Mx1),
                            ('g2M','x2MM_d1',self.CG_d1g2Mx2MM),('g2M','x2M_d1',self.CG_d1g2Mx2M),('g2M','x2_d1',self.CG_d1g2Mx2),
            
                            # |F',mF'> --> |F=2,mF=0>
                            ('g2','x1M_d1',self.CG_d1g2x1M),('g2','x1_d1',self.CG_d1g2x1),('g2','x1P_d1',self.CG_d1g2x1P),
                            ('g2','x2M_d1',self.CG_d1g2x2M),('g2','x2_d1',self.CG_d1g2x2),('g2','x2P_d1',self.CG_d1g2x2P),
            
                            # |F',mF'> --> |F=2,mF=+1>
                            ('g2P','x1_d1',self.CG_d1g2Px1),('g2P','x1P_d1',self.CG_d1g2Px1P),
                            ('g2P','x2_d1',self.CG_d1g2Px2),('g2P','x2P_d1',self.CG_d1g2Px2P),('g2P','x2PP_d1',self.CG_d1g2Px2PP),
            
                            # |F',mF'> --> |F=2,mF=+2>
                            ('g2PP','x1P_d1',self.CG_d1g2PPx1P),
                            ('g2PP','x2P_d1',self.CG_d1g2PPx2P),('g2PP','x2PP_d1',self.CG_d1g2PPx2PP),
                            ]

        spontDecayOps_d2 = []
        # np.sqrt(2) in font of trans strength is because sum of strengths 
        # is 1/2 for D2 but splitting ratios need to sum to 1

        for xLev in spontEmmChannels:
            if self.photonic_space:
                try:
                    spontDecayOps_d2.append(np.sqrt(2) * xLev[2] * np.sqrt(2*gamma_d2) * 
                                tensor(
                                    basis(M, atomStates[xLev[0]]) * basis(M, atomStates[xLev[1]]).dag(), qeye(N), qeye(N)))
                except KeyError:
                    pass
            else:
                try:
                    spontDecayOps_d2.append(np.sqrt(2) * xLev[2] * np.sqrt(2*gamma_d2) * 
                                basis(M, atomStates[xLev[0]]) * basis(M, atomStates[xLev[1]]).dag())
                except KeyError:
                    pass


        spontDecayOps_d1 = []

        for xLev in spontEmmChannels_d1:
            if self.photonic_space:
                try:
                    spontDecayOps_d1.append( xLev[2] * np.sqrt(2*gamma_d1) * 
                                tensor(
                                    basis(M, atomStates[xLev[0]]) * basis(M, atomStates[xLev[1]]).dag(), qeye(N), qeye(N)))
                except KeyError:
                    pass
            else:
                try:
                    spontDecayOps_d1.append(xLev[2] * np.sqrt(2*gamma_d1) * 
                                basis(M, atomStates[xLev[0]]) * basis(M, atomStates[xLev[1]]).dag())
                except KeyError:
                    pass


    

        return ([spontDecayOps_d2,spontDecayOps_d1])


    #return couplings for various levels and laser polarisations which only takes delta - detuning from resonance as an input argument
    def getCouplingsF1_Sigma_Plus(self, delta):
        return [ 
        # For |F,mF>=|1,mF> <--> |F',mF'>=|3,mF+/-1>
        (self.CGg1Mx3,   'g1M', 'x3',   delta + self.deltaZ - self.deltaZx3 - self.deltaEx3, +1),
        (self.CGg1x3P,   'g1',  'x3P',  delta - self.deltaZx3P - self.deltaEx3, +1),
        (self.CGg1Px3PP, 'g1P', 'x3PP', delta - self.deltaZ - self.deltaZx3PP - self.deltaEx3, +1),
        # For |F,mF>=|1,mF> <--> |F',mF'>=|2,mF+/-1>
        (self.CGg1Mx2,   'g1M', 'x2',   delta + self.deltaZ - self.deltaZx2, +1),
        (self.CGg1x2P,   'g1',  'x2P',  delta - self.deltaZx2P, +1),
        (self.CGg1Px2PP, 'g1P', 'x2PP', delta - self.deltaZ - self.deltaZx2PP, +1),
        # For |F,mF>=|1,mF> <--> |F',mF'>=|1,mF+/-1>
        (self.CGg1Mx1,   'g1M', 'x1',   delta + self.deltaZ - self.deltaZx1 - self.deltaEx1, +1),
        (self.CGg1x1P,   'g1',  'x1P',  delta - self.deltaZx1P - self.deltaEx1, +1),
        # For |F,mF>=|1,mF> <--> |F',mF'>=|0,mF+/-1>
        (self.CGg1Mx0,   'g1M', 'x0',   delta + self.deltaZ - self.deltaZx0 - self.deltaEx0, +1),
        ]

    def getCouplingsF1_Sigma_Minus(self, delta):
        return [ 
        # For |F,mF>=|1,mF> <--> |F',mF'>=|3,mF+/-1>
        (self.CGg1Mx3MM, 'g1M', 'x3MM', delta + self.deltaZ - self.deltaZx3MM - self.deltaEx3, -1),
        (self.CGg1x3M,   'g1',  'x3M',  delta - self.deltaZx3M - self.deltaEx3, -1),
        (self.CGg1Px3,   'g1P', 'x3',   delta - self.deltaZ - self.deltaZx3 - self.deltaEx3, -1),
        # For |F,mF>=|1,mF> <--> |F',mF'>=|2,mF+/-1>
        (self.CGg1Mx2MM, 'g1M', 'x2MM', delta + self.deltaZ - self.deltaZx2MM, -1),
        (self.CGg1x2M,   'g1',  'x2M',  delta - self.deltaZx2M, -1),
        (self.CGg1Px2,   'g1P', 'x2',   delta - self.deltaZ - self.deltaZx2, -1),
        # For |F,mF>=|1,mF> <--> |F',mF'>=|1,mF+/-1>
        (self.CGg1x1M,   'g1',  'x1M',  delta - self.deltaZx1M - self.deltaEx1, -1),
        (self.CGg1Px1,   'g1P', 'x1',   delta - self.deltaZ - self.deltaZx1 - self.deltaEx1, -1),
        # For |F,mF>=|1,mF> <--> |F',mF'>=|0,mF+/-1>
        (self.CGg1Px0,   'g1P', 'x0',   delta - self.deltaZ - self.deltaZx0 - self.deltaEx0, -1),
        ]

    def getCouplingsF2_Pi(self, delta):
        return [ 
        # For |F,mF>=|2,mF> <--> |F',mF'>=|3,mF>
        (self.CGg2MMx3MM, 'g2MM', 'x3MM', delta - 2*self.deltaZ - self.deltaZx3MM - self.deltaEx3, 0),
        (self.CGg2Mx3M,   'g2M',  'x3M',  delta -   self.deltaZ - self.deltaZx3M - self.deltaEx3, 0),
        (self.CGg2x3,     'g2',   'x3',   delta - self.deltaZx3 - self.deltaEx3, 0),
        (self.CGg2Px3P,   'g2P',  'x3P',  delta +   self.deltaZ - self.deltaZx3P - self.deltaEx3, 0),
        (self.CGg2PPx3PP, 'g2PP', 'x3PP', delta + 2*self.deltaZ - self.deltaZx3PP - self.deltaEx3, 0),
        # For |F,mF>=|2,mF> <--> |F',mF'>=|2,mF>
        (self.CGg2MMx2MM, 'g2MM', 'x2MM', delta - 2*self.deltaZ - self.deltaZx2MM, 0),
        (self.CGg2Mx2M,   'g2M',  'x2M',  delta -   self.deltaZ - self.deltaZx2M, 0),
        (self.CGg2x2,     'g2',   'x2',   delta - self.deltaZx2, 0),
        (self.CGg2Px2P,   'g2P',  'x2P',  delta +   self.deltaZ - self.deltaZx2P, 0),
        (self.CGg2PPx2PP, 'g2PP', 'x2PP', delta + 2*self.deltaZ - self.deltaZx2PP, 0),
        # For |F,mF>=|2,mF> <--> |F',mF'>=|1,mF>
        (self.CGg2Mx1M,   'g2M',  'x1M',  delta -   self.deltaZ - self.deltaZx1M - self.deltaEx1, 0),
        (self.CGg2x1,     'g2',   'x1',   delta - self.deltaZx1 - self.deltaEx1, 0),
        (self.CGg2Px1P,   'g2P',  'x1P',  delta +   self.deltaZ -self.deltaZx1P - self.deltaEx1, 0),
        # For |F,mF>=|2,mF> <--> |F',mF'>=|0,mF>
        (self.CGg2x0,     'g2',   'x0',   delta - self.deltaZx0 - self.deltaEx0, 0),
        ]

    def getCouplingsF1_Pi(self, delta):
        return [ 
        # For |F,mF>=|1,mF> <--> |F',mF'>=|3,mF>
        (self.CGg1Mx3M, 'g1M', 'x3M', delta + self.deltaZ - self.deltaZx3M - self.deltaEx3, 0),
        (self.CGg1x3,   'g1',  'x3',  delta - self.deltaZx3 - self.deltaEx3, 0),
        (self.CGg1Px3P, 'g1P', 'x3P', delta - self.deltaZ - self.deltaZx3P - self.deltaEx3, 0),
        # For |F,mF>=|1,mF> <--> |F',mF'>=|2,mF>
        (self.CGg1Mx2M, 'g1M', 'x2M', delta + self.deltaZ - self.deltaZx2M, 0),
        (self.CGg1x2,   'g1',  'x2',  delta - self.deltaZx2, 0),
        (self.CGg1Px2P, 'g1P', 'x2P', delta - self.deltaZ - self.deltaZx2P, 0),
        # For |F,mF>=|1,mF> <--> |F',mF'>=|1,mF>
        (self.CGg1Mx1M, 'g1M', 'x1M', delta + self.deltaZ - self.deltaZx1M - self.deltaEx1, 0),
        (self.CGg1x1,   'g1',  'x1',  delta - self.deltaZx1 - self.deltaEx1, 0),
        (self.CGg1Px1P, 'g1P', 'x1P', delta - self.deltaZ - self.deltaZx1P - self.deltaEx1, 0),
        # For |F,mF>=|1,mF> <--> |F',mF'>=|0,mF>
        (self.CGg1x0,   'g1',  'x0',  delta - self.deltaZx0 - self.deltaEx0, 0),
        ]

    def getCouplingsF2_Sigma_Plus(self, delta):
        return [
        # For |F,mF>=|2,mF> <--> |F',mF'>=|3,mF+/-1>
        (self.CGg2MMx3M,   'g2MM', 'x3M',   delta - 2 * self.deltaZ - self.deltaZx3M - self.deltaEx3, +1),
        (self.CGg2Mx3,     'g2M',  'x3',    delta - self.deltaZ - self.deltaZx3 - self.deltaEx3, +1),
        (self.CGg2x3P,     'g2',   'x3P',   delta - self.deltaZx3P - self.deltaEx3, +1),
        (self.CGg2Px3PP,   'g2P',  'x3PP',  delta + self.deltaZ - self.deltaZx3PP - self.deltaEx3, +1),
        (self.CGg2PPx3PPP, 'g2PP', 'x3PPP', delta + 2 * self.deltaZ - self.deltaZx3PPP - self.deltaEx3, +1),
        # For |F,mF>=|2,mF> <--> |F',mF'>=|2,mF+/-1>
        (self.CGg2MMx2M,   'g2MM', 'x2M',   delta - 2 * self.deltaZ - self.deltaZx2M, +1),
        (self.CGg2Mx2,     'g2M',  'x2',    delta - self.deltaZ - self.deltaZx2, +1),
        (self.CGg2x2P,     'g2',   'x2P',   delta - self.deltaZx2P, +1),
        (self.CGg2Px2PP,   'g2P',  'x2PP',  delta + self.deltaZ - self.deltaZx2PP, +1),
        # For |F,mF>=|2,mF> <--> |F',mF'>=|1,mF+/-1>
        (self.CGg2MMx1M,   'g2MM', 'x1M',   delta - 2 * self.deltaZ - self.deltaZx1M - self.deltaEx1, +1),
        (self.CGg2Mx1,     'g2M',  'x1',    delta - self.deltaZ - self.deltaZx1 - self.deltaEx1, +1),
        (self.CGg2x1P,     'g2',   'x1P',   delta - self.deltaZx1P - self.deltaEx1, +1),
        # For |F,mF>=|2,mF> <--> |F',mF'>=|0,mF+/-1>
        (self.CGg2Mx0,     'g2M',  'x0',    delta - self.deltaZ - self.deltaZx0 - self.deltaEx0, +1),
        ]

    def getCouplingsF2_Sigma_Minus(self,delta):
        return [
        # For |F,mF>=|2,mF> <--> |F',mF'>=|3,mF+/-1>
        (self.CGg2MMx3MMM, 'g2MM', 'x3MMM', delta - 2 * self.deltaZ - self.deltaZx3MMM - self.deltaEx3, -1),
        (self.CGg2Mx3MM,   'g2M',  'x3MM',  delta - self.deltaZ - self.deltaZx3MM - self.deltaEx3, -1),
        (self.CGg2x3M,     'g2',   'x3M',   delta - self.deltaZx3M - self.deltaEx3, -1),
        (self.CGg2Px3,     'g2P',  'x3',    delta + self.deltaZ - self.deltaZx3 - self.deltaEx3, -1),
        (self.CGg2PPx3P,   'g2PP', 'x3P',   delta + 2 * self.deltaZ - self.deltaZx3P - self.deltaEx3, -1),
        # For |F,mF>=|2,mF> <--> |F',mF'>=|2,mF+/-1>
        (self.CGg2Mx2MM,   'g2M',  'x2MM',  delta - self.deltaZ - self.deltaZx2MM, -1),
        (self.CGg2x2M,     'g2',   'x2M',   delta - self.deltaZx2M, -1),
        (self.CGg2Px2,     'g2P',  'x2',    delta + self.deltaZ - self.deltaZx2, -1),
        (self.CGg2PPx2P,   'g2PP', 'x2P',   delta + 2 * self.deltaZ - self.deltaZx2P, -1),
        # For |F,mF>=|2,mF> <--> |F',mF'>=|1,mF+/-1>
        (self.CGg2Mx1,     'g2M',  'x1',    delta - self.deltaZ - self.deltaZx1 - self.deltaEx1, -1),
        (self.CGg2x1M,     'g2',   'x1M',   delta - self.deltaZx1M - self.deltaEx1, -1),
        (self.CGg2Px1,     'g2P',  'x1',    delta + self.deltaZ - self.deltaZx1 - self.deltaEx1, -1),
        (self.CGg2PPx1P,   'g2PP', 'x1P',   delta + 2 * self.deltaZ - self.deltaZx1P - self.deltaEx1, -1),
        # For |F,mF>=|2,mF> <--> |F',mF'>=|0,mF+/-1>
        (self.CGg2Px0,     'g2P',  'x0',    delta + self.deltaZ - self.deltaZx0 - self.deltaEx0, -1)
        ]

    def getD1CouplingsF1_Sigma_Plus(self,delta):
        return [ 
        # For |F,mF>=|1,mF> <--> |F',mF'>=|2,mF+/-1>
        (self.CG_d1g1Mx2,   'g1M', 'x2_d1',   delta + self.deltaZ_d1 - self.deltaZx2_d1,1),
        (self.CG_d1g1x2P,   'g1',  'x2P_d1',  delta - self.deltaZx2P_d1,1),
        (self.CG_d1g1Px2PP, 'g1P', 'x2PP_d1', delta - self.deltaZ_d1 - self.deltaZx2PP_d1,1),
        # For |F,mF>=|1,mF> <--> |F',mF'>=|1,mF+/-1>
        (self.CG_d1g1Mx1,   'g1M', 'x1_d1',   delta + self.deltaZ_d1 - self.deltaZx1_d1 - self.deltaEx1_d1,1),
        (self.CG_d1g1x1P,   'g1',  'x1P_d1',  delta - self.deltaZx1P_d1 - self.deltaEx1_d1,1),
        ]

    def getD1CouplingsF1_Sigma_Minus(self,delta):
        return [ 
        # For |F,mF>=|1,mF> <--> |F',mF'>=|2,mF+/-1>
        (self.CG_d1g1Mx2MM, 'g1M', 'x2MM_d1', delta + self.deltaZ_d1 - self.deltaZx2MM_d1,-1),
        (self.CG_d1g1x2M,   'g1',  'x2M_d1',  delta - self.deltaZx2M_d1,-1),
        (self.CG_d1g1Px2,   'g1P', 'x2_d1',   delta - self.deltaZ_d1 - self.deltaZx2_d1,-1),
        # For |F,mF>=|1,mF> <--> |F',mF'>=|1,mF+/-1>
        (self.CG_d1g1x1M,   'g1',  'x1M_d1',  delta - self.deltaZx1M_d1 - self.deltaEx1_d1,-1),
        (self.CG_d1g1Px1,   'g1P', 'x1_d1',   delta - self.deltaZ_d1 - self.deltaZx1_d1 - self.deltaEx1_d1,-1),
        ]

    def getD1CouplingsF2_Pi(self,delta):
        return [ 
        # For |F,mF>=|2,mF> <--> |F',mF'>=|2,mF>
        (self.CG_d1g2MMx2MM, 'g2MM', 'x2MM_d1', delta - 2*self.deltaZ_d1 - self.deltaZx2MM_d1,0),
        (self.CG_d1g2Mx2M,   'g2M',  'x2M_d1',  delta -   self.deltaZ_d1 - self.deltaZx2M_d1,0),
        (self.CG_d1g2x2,     'g2',   'x2_d1',   delta - self.deltaZx2_d1,0),
        (self.CG_d1g2Px2P,   'g2P',  'x2P_d1',  delta +   self.deltaZ_d1 - self.deltaZx2P_d1,0),
        (self.CG_d1g2PPx2PP, 'g2PP', 'x2PP_d1', delta + 2*self.deltaZ_d1 - self.deltaZx2PP_d1,0),
        # For |F,mF>=|2,mF> <--> |F',mF'>=|1,mF>
        (self.CG_d1g2Mx1M,   'g2M',  'x1M_d1',  delta -   self.deltaZ_d1 - self.deltaZx1M_d1 - self.deltaEx1_d1,0),
        (self.CG_d1g2x1,     'g2',   'x1_d1',   delta - self.deltaZx1_d1 - self.deltaEx1_d1,0),
        (self.CG_d1g2Px1P,   'g2P',  'x1P_d1',  delta +   self.deltaZ_d1 - self.deltaZx1P_d1 - self.deltaEx1_d1,0),

        ]

    def getD1CouplingsF1_Pi(self, delta):
        return [ 
        # For |F,mF>=|1,mF> <--> |F',mF'>=|2,mF>
        (self.CG_d1g1Mx2M, 'g1M', 'x2M_d1', delta + self.deltaZ_d1 - self.deltaZx2M_d1,0),
        (self.CG_d1g1x2,   'g1',  'x2_d1',  delta - self.deltaZx2_d1,0),
        (self.CG_d1g1Px2P, 'g1P', 'x2P_d1', delta - self.deltaZ_d1 - self.deltaZx2P_d1,0),
        # For |F,mF>=|1,mF> <--> |F',mF'>=|1,mF>
        (self.CG_d1g1Mx1M, 'g1M', 'x1M_d1', delta + self.deltaZ_d1 - self.deltaZx1M_d1 - self.deltaEx1_d1,0),
        (self.CG_d1g1x1,   'g1',  'x1_d1',  delta - self.deltaZx1_d1 - self.deltaEx1_d1,0),
        (self.CG_d1g1Px1P, 'g1P', 'x1P_d1', delta - self.deltaZ_d1 - self.deltaZx1P_d1 - self.deltaEx1_d1,0),
        ]

    def getD1CouplingsF2_Sigma_Plus(self,delta):
        return [
        # For |F,mF>=|2,mF> <--> |F',mF'>=|2,mF+/-1>
        (self.CG_d1g2MMx2M,   'g2MM', 'x2M_d1',   delta - 2 * self.deltaZ_d1 - self.deltaZx2M_d1,1),
        (self.CG_d1g2Mx2,     'g2M',  'x2_d1',    delta - self.deltaZ_d1 - self.deltaZx2_d1,1),
        (self.CG_d1g2x2P,     'g2',   'x2P_d1',   delta - self.deltaZx2P_d1,1),
        (self.CG_d1g2Px2PP,   'g2P',  'x2PP_d1',  delta + self.deltaZ_d1 - self.deltaZx2PP_d1,1),
        # For |F,mF>=|2,mF> <--> |F',mF'>=|1,mF+/-1>
        (self.CG_d1g2MMx1M,   'g2MM', 'x1M_d1',   delta - 2 * self.deltaZ_d1 - self.deltaZx1M_d1 - self.deltaEx1_d1,1),
        (self.CG_d1g2Mx1,     'g2M',  'x1_d1',    delta - self.deltaZ_d1 - self.deltaZx1_d1 - self.deltaEx1_d1,1),
        (self.CG_d1g2x1P,     'g2',   'x1P_d1',   delta - self.deltaZx1P_d1 - self.deltaEx1_d1,1)
        ]

    def getD1CouplingsF2_Sigma_Minus(self,delta):
        return [
        # For |F,mF>=|2,mF> <--> |F',mF'>=|2,mF+/-1>
        (self.CG_d1g2Mx2MM,   'g2M',  'x2MM_d1',  delta - self.deltaZ_d1 - self.deltaZx2MM_d1,-1),
        (self.CG_d1g2x2M,     'g2',   'x2M_d1',   delta - self.deltaZx2M_d1,-1),
        (self.CG_d1g2Px2,     'g2P',  'x2_d1',    delta + self.deltaZ_d1 - self.deltaZx2_d1,-1),
        (self.CG_d1g2PPx2P,   'g2PP', 'x2P_d1',   delta + 2 * self.deltaZ_d1 - self.deltaZx2P_d1,-1),
        # For |F,mF>=|2,mF> <--> |F',mF'>=|1,mF+/-1>
        (self.CG_d1g2x1M,     'g2',   'x1M_d1',   delta - self.deltaZx1M_d1 - self.deltaEx1_d1,-1),
        (self.CG_d1g2Px1,     'g2P',  'x1_d1',    delta + self.deltaZ_d1 - self.deltaZx1_d1 - self.deltaEx1_d1,-1),
        (self.CG_d1g2PPx1P,   'g2PP', 'x1P_d1',   delta + 2 * self.deltaZ_d1 - self.deltaZx1P_d1 - self.deltaEx1_d1,-1)    
        ]    

    def gen_H_VSTIRAP_D1(self,ketbras, atomStates, delta_cav, delta_laser,F_start,F_final,F_exc,laser_pol, omega_s, driving_shape, shape_args, coupling_factor, deltaP, quant_bas):
    #create Hamiltonian for V-STIRAP photon production w.r.t. D1 transitions
    #arguments: 
    # xlvls, atomStates - excited levels and ground states to be included in simulation
    # delta, F_start, F_final, F_exc - detuning w.r.t. desired excited state (assuming two photon resonance condition for both laser and cavity), initial F level, final F level, and desired excited level F' w.r.t. adiabatic passage is performed
    # laser_pol, omega_s, driving_shape, shape_args - polarisation of the laser is given as string 'sigmaP', 'sigmaM' or 'Pi', driving shape in qutip compatible string format, and dictionar list of required args for the driving pulse
    # coupling_factor, deltaP, quant_axis - cavity coupling in angular frequeny with included CG coefficient for the required transition, cavity birefringence and cav basis vectors

        args_hams_driving_pulse=shape_args
        quant_bas_x=quant_bas[0]
        quant_bas_y=quant_bas[1]

  
        if F_exc==1:
            delta_laser+=self.deltaEx1_d1
            delta_cav+=self.deltaEx1_d1
        
        if F_exc==0:
            delta_laser+= self.deltaEx0
            delta_cav+=self.deltaEx0
        elif F_exc==1:
            delta_laser+=self.deltaEx1
            delta_cav+=self.deltaEx1
        elif F_exc==3:
            delta_laser+=self.deltaEx3
            delta_cav+=self.deltaEx3
        
        if F_start==1:
            if laser_pol=='pi':
                laserCouplings_VStirap = self.getD1CouplingsF1_Pi(delta_laser)
            elif laser_pol=='sigmaP':
                laserCouplings_VStirap = self.getD1CouplingsF1_Sigma_Plus(delta_laser)
            elif laser_pol=='sigmaM':
                laserCouplings_VStirap = self.getD1CouplingsF1_Sigma_Minus(delta_laser)
        elif F_start==2:
            if laser_pol=='pi':
                laserCouplings_VStirap = self.getD1CouplingsF2_Pi(delta_laser)
            if laser_pol=='sigmaP':
                laserCouplings_VStirap = self.getD1CouplingsF2_Sigma_Plus(delta_laser)
            if laser_pol=='sigmaM':
                laserCouplings_VStirap = self.getD1CouplingsF2_Sigma_Minus(delta_laser)

        if F_final==1:
                cavityCouplings_plus = self.getD1CouplingsF1_Sigma_Plus(delta_cav)
                cavityCouplings_minus = self.getD1CouplingsF1_Sigma_Minus(delta_cav)
                cavityCouplings_pi = self.getD1CouplingsF1_Pi(delta_cav)
        elif F_final==2:
                cavityCouplings_plus = self.getD1CouplingsF2_Sigma_Plus(delta_cav)
                cavityCouplings_minus = self.getD1CouplingsF2_Sigma_Minus(delta_cav)
                cavityCouplings_pi = self.getD1CouplingsF2_Pi(delta_cav)


        hams_cavity_pi, args_hams_cavity_pi = couplingsToCavHamiltonian(quant_bas_x,quant_bas_y, ketbras, atomStates,deltaP, coupling_factor,cavityCouplings_pi)
        hams_cavity_plus, args_hams_cavity_plus = couplingsToCavHamiltonian(quant_bas_x,quant_bas_y, ketbras, atomStates,deltaP, coupling_factor,cavityCouplings_plus)
        hams_cavity_minus, args_hams_cavity_minus = couplingsToCavHamiltonian(quant_bas_x,quant_bas_y, ketbras, atomStates,deltaP, coupling_factor,cavityCouplings_minus)
        hams_VStirap_laser, args_hams_VStirap_laser = couplingsToLaserHamiltonian(ketbras, atomStates,self.photonic_space,laserCouplings_VStirap,omega_s,driving_shape)
        args_hams_VStirap = {**args_hams_VStirap_laser,**args_hams_cavity_plus,**args_hams_cavity_minus,**args_hams_cavity_pi, **args_hams_driving_pulse}

        H_VStirap = list(chain(*[hams_cavity_plus,hams_cavity_minus, hams_cavity_pi,hams_VStirap_laser]))

        return [H_VStirap, args_hams_VStirap]

    
    def gen_H_VSTIRAP_D2(self, ketbras, atomStates, delta_cav,delta_laser,F_start,F_final,F_exc,laser_pol, omega_s, driving_shape, shape_args, coupling_factor, deltaP, quant_bas):
    #create Hamiltonian for V-STIRAP photon production w.r.t. D2 transitions
    #arguments: 
    # xlvls, atomStates - excited levels and ground states to be included in simulation
    # delta_cav,delta_laser, F_start, F_final, F_exc - detuning w.r.t. desired excited state (assuming two photon resonance condition for both laser and cavity), initial F level, final F level, and desired excited level F' w.r.t. adiabatic passage is performed
    # laser_pol, omega_s, driving_shape, shape_args - polarisation of the laser is given as string 'sigmaP', 'sigmaM' or 'Pi', driving shape in qutip compatible string format, and dictionar list of required args for the driving pulse
    # coupling_factor, deltaP, quant_bas - cavity coupling in angular frequeny with included CG coefficient for the required transition, cavity birefringence and basis for the quantisation axis and cavity axis

        args_hams_driving_pulse=shape_args
        quant_bas_x=quant_bas[0]
        quant_bas_y=quant_bas[1]

        if F_exc==0:
            delta_laser+= self.deltaEx0
            delta_cav+=self.deltaEx0
        elif F_exc==1:
            delta_laser+=self.deltaEx1
            delta_cav+=self.deltaEx1
        elif F_exc==3:
            delta_laser+=self.deltaEx3
            delta_cav+=self.deltaEx3
        
        if F_start==1:
            if laser_pol=='pi':
                laserCouplings_VStirap = self.getCouplingsF1_Pi(delta_laser)
            elif laser_pol=='sigmaP':
                laserCouplings_VStirap = self.getCouplingsF1_Sigma_Plus(delta_laser)
            elif laser_pol=='sigmaM':
                laserCouplings_VStirap = self.getCouplingsF1_Sigma_Minus(delta_laser)
        elif F_start==2:
            if laser_pol=='pi':
                laserCouplings_VStirap = self.getCouplingsF2_Pi(delta_laser)
            elif laser_pol=='sigmaP':
                laserCouplings_VStirap = self.getCouplingsF2_Sigma_Plus(delta_laser)
            elif laser_pol=='sigmaM':
                laserCouplings_VStirap = self.getCouplingsF2_Sigma_Minus(delta_laser)

        if F_final==1:
                cavityCouplings_plus = self.getCouplingsF1_Sigma_Plus(delta_cav)
                cavityCouplings_minus = self.getCouplingsF1_Sigma_Minus(delta_cav)
                cavityCouplings_pi = self.getCouplingsF1_Pi(delta_cav)
        elif F_final==2:
                cavityCouplings_plus = self.getCouplingsF2_Sigma_Plus(delta_cav)
                cavityCouplings_minus = self.getCouplingsF2_Sigma_Minus(delta_cav)
                cavityCouplings_pi = self.getCouplingsF2_Pi(delta_cav)
          
            
        hams_cavity_pi, args_hams_cavity_pi = couplingsToCavHamiltonian(quant_bas_x,quant_bas_y, ketbras, atomStates,deltaP, coupling_factor,cavityCouplings_pi)
        hams_cavity_plus, args_hams_cavity_plus = couplingsToCavHamiltonian(quant_bas_x,quant_bas_y, ketbras, atomStates,deltaP, coupling_factor,cavityCouplings_plus)
        hams_cavity_minus, args_hams_cavity_minus = couplingsToCavHamiltonian(quant_bas_x,quant_bas_y, ketbras, atomStates,deltaP, coupling_factor,cavityCouplings_minus)
        hams_VStirap_laser, args_hams_VStirap_laser = couplingsToLaserHamiltonian(ketbras, atomStates,self.photonic_space,laserCouplings_VStirap,omega_s,driving_shape)
        args_hams_VStirap = {**args_hams_VStirap_laser,**args_hams_cavity_plus,**args_hams_cavity_minus,**args_hams_cavity_pi, **args_hams_driving_pulse}

        H_VStirap = list(chain(*[hams_cavity_plus,hams_cavity_minus, hams_cavity_pi,hams_VStirap_laser]))

        return [H_VStirap, args_hams_VStirap]
    

    def gen_H_Pulse_D1(self,ketbras, atomStates, delta,F_start,F_exc,laser_pol, omega, shape_args, driving_shape='np.sin(w*t)**2', _array=False, _amp=[], _t=[]):
        #create Hamiltonian for laser pulse w.r.t. D2 transitions
        #arguments: 
        # xlvls, atomStates - excited levels and ground states to be included in simulation
        # delta, F_start, F_exc - detuning w.r.t. desired excited stat, initial F level and desired excited level F' 
        # laser_pol, omega, driving_shape, shape_args - polarisation of the laser is given as string 'sigmaP', 'sigmaM' or 'Pi', driving shape in qutip compatible string format, and dictionar list of required args for the driving pulse
        args_hams_driving_pulse=shape_args

        delta_laser=delta    
        if F_exc==1:
            delta_laser+=self.deltaEx1_d1

        if F_start==1:
            if laser_pol=='pi':
                laserCouplings_Pulse = self.getD1CouplingsF1_Pi(delta_laser)
            elif laser_pol=='sigmaP':
                laserCouplings_Pulse = self.getD1CouplingsF1_Sigma_Plus(delta_laser)
            elif laser_pol=='sigmaM':
                laserCouplings_Pulse = self.getD1CouplingsF1_Sigma_Minus(delta_laser)

        elif F_start==2:
            if laser_pol=='pi':
                laserCouplings_Pulse = self.getD1CouplingsF2_Pi(delta_laser)
            elif laser_pol=='sigmaP':
                laserCouplings_Pulse = self.getD1CouplingsF2_Sigma_Plus(delta_laser)
            elif laser_pol=='sigmaM':
                laserCouplings_Pulse = self.getD1CouplingsF2_Sigma_Minus(delta_laser)
            

        hams_laser, args_hams_laser = couplingsToLaserHamiltonian(ketbras, atomStates,self.photonic_space,laserCouplings_Pulse,omega,
                                                                                    driving_shape,  _array, _amp, _t)
        args_hams_laser_r={**args_hams_driving_pulse, **args_hams_laser}
        return [hams_laser, args_hams_laser_r]
    

    def gen_H_Pulse_D2(self,ketbras, atomStates, delta,F_start,F_exc,laser_pol, omega,shape_args, driving_shape='np.sin(w*t)**2',  _array=False, _amp=[], _t=[]):
        #create Hamiltonian for laser pulse w.r.t. D1 transitions
        #arguments: 
        # xlvls, atomStates - excited levels and ground states to be included in simulation
        # delta, F_start, F_exc - detuning w.r.t. desired excited stat, initial F level and desired excited level F' 
        # laser_pol, omega, driving_shape, shape_args - polarisation of the laser is given as string 'sigmaP', 'sigmaM' or 'Pi', driving shape in qutip compatible string
        # Create Hamiltonian terms for laser pulse w.r.t. d1 transitions, returns hamiltonian and hamiltonian arguments
        args_hams_driving_pulse=shape_args

        delta_laser=delta    
        if F_exc==0:
            delta_laser+=self.deltaEx0
        elif F_exc==1:
            delta_laser+=self.deltaEx1
        elif F_exc==3:
            delta_laser+=self.deltaEx3
        
        if F_start==1:
            if laser_pol=='pi':
                laserCouplings_Pulse = self.getCouplingsF1_Pi(delta_laser)
            elif laser_pol=='sigmaP':
                laserCouplings_Pulse = self.getCouplingsF1_Sigma_Plus(delta_laser)
            elif laser_pol=='sigmaM':
                laserCouplings_Pulse = self.getCouplingsF1_Sigma_Minus(delta_laser)
            
        elif F_start==2:
            if laser_pol=='pi':
                laserCouplings_Pulse = self.getCouplingsF2_Pi(delta_laser)
            elif laser_pol=='sigmaP':
                laserCouplings_Pulse = self.getCouplingsF2_Sigma_Plus(delta_laser)
            elif laser_pol=='sigmaM':
                laserCouplings_Pulse = self.getCouplingsF2_Sigma_Minus(delta_laser)
            

        hams_laser, args_hams_laser = couplingsToLaserHamiltonian(ketbras, atomStates,self.photonic_space,laserCouplings_Pulse,omega,
                                                                                    driving_shape,  _array, _amp, _t)
        return [hams_laser, args_hams_laser]
    
        
    def gen_H_OpticalPump_D2(self,ketbras, atomStates, delta1,delta2,F_start_1,F_start_2,F_exc_1,F_exc_2,laser_pol_1, laser_pol_2, omega1,omega2, driving_shape, shape_args):
        [hams_pulse1, args_hams_pulse_1]=self.gen_H_Pulse_D2(ketbras, atomStates, delta1,F_start_1,F_exc_1,laser_pol_1, omega1, driving_shape, shape_args)
        [hams_pulse2, args_hams_pulse_2]=self.gen_H_Pulse_D2(ketbras, atomStates, delta2,F_start_2,F_exc_2,laser_pol_2, omega2, driving_shape, shape_args)

        return [list(chain(*[hams_pulse1, hams_pulse2])),{**args_hams_pulse_1, **args_hams_pulse_2}]


    def plotter_spontdecay_channels(self, atomStates, output, t_list):

        output_states=output
        t=t_list
        tStep=(t[-1]-t[0])/(len(t)-1)

        # Spontaneous emission from D1 and D2 lines
        sigma_spontDecayOp_d2 = sum([x.dag()*x for x in self.spont_em_ops( atomStates)[0]])
        sigma_spontDecayOp_d1=sum([x.dag()*x for x in self.spont_em_ops( atomStates)[1]])
        exp_spontDecay_d1 = np.abs( np.array([(x*sigma_spontDecayOp_d1).tr() for x in output_states]) )
        exp_spontDecay_d2 = np.abs( np.array([(x*sigma_spontDecayOp_d2).tr() for x in output_states]) )

        # Total spontaneous emission.
        n_spont_d1 = np.trapz(exp_spontDecay_d1, dx=tStep)
        n_spont_d2 = np.trapz(exp_spontDecay_d2, dx=tStep)
        print('Total spontaneous emission from D1:', np.round(n_spont_d1/2,3))
        print('Total spontaneous emission from D2:', np.round(n_spont_d2/2,3))
        
        # Plotting
        fig, ax = plt.subplots(1,1,figsize=(14,12))
        ax.plot(t, exp_spontDecay_d1, 'b', label='Spont. Emission Rate D1')
        ax.plot(t, exp_spontDecay_d2, 'g', label='Spont. Emission Rate D2')
        ax.legend(loc='best')
        
        return(fig)


    def plotter_atomstate_population(self, ketbras, output, t_list, bol_d1:bool):
        '''Returns plots of ground state populations and excited state populations for either the D1 or D2 transition lines
            input args
            ketbras: precomputed ketbras for the simulation
            output: output from numerical qutip solver for run of a particular hamiltonian
            t_list: list of timesteps in the hamiltonian simulation
            bol_d1: boolean for whether to print the d1 or d2 excited state populations'''
        output_states=output
        t=t_list
        tStep=(t[-1]-t[0])/(len(t)-1)

        if bol_d1:
            [
                ag1M, ag1, ag1P,
                ag2MM, ag2M, ag2, ag2P, ag2PP,
                ax1M_d1, ax1_d1, ax1P_d1,
                ax2MM_d1, ax2M_d1, ax2_d1, ax2P_d1, ax2PP_d1
            ] = self.ketbras.getrb_aList(ketbras)

            [exp_ax1M_d1,exp_ax1_d1,exp_ax1P_d1,
            exp_ax2MM_d1,exp_ax2M_d1,exp_ax2_d1,exp_ax2P_d1,exp_ax2PP_d1] = [
                np.real( np.array([(xLev*a).tr() for xLev in output_states]) )
                for a in [ax1M_d1,ax1_d1,ax1P_d1,
                        ax2MM_d1,ax2M_d1,ax2_d1,ax2P_d1,ax2PP_d1]
            ]

            [exp_ag1,exp_ag1P,exp_ag1M,
            exp_ag2MM,exp_ag2M,exp_ag2,exp_ag2P,exp_ag2PP] = [
                np.real( np.array([(xLev*a).tr() for xLev in output_states]) ) 
                for a in [ag1,ag1P,ag1M,
                        ag2MM,ag2M,ag2,ag2P,ag2PP]
            ]

            fig_d1, (a1,a2,b1,b2) = plt.subplots(4,1,figsize=(15,12))

            a1.plot(t, exp_ag1M,   'r', label='$g1M: |F,mF>=|1,-1>$')
            a1.plot(t, exp_ag1,  '--c', label='$g1:  |F,mF>=|1,0>$')
            a1.plot(t, exp_ag1P, '--m', label='$g1P: |F,mF>=|1,+1>$')
            a1.legend(loc='best')

            a2.plot(t, exp_ag2MM,  'b', label='$g2MM:|F,mF>=|2,-2>$')
            a2.plot(t, exp_ag2M, '--r', label='$g2M: |F,mF>=|2,-1>$')
            a2.plot(t, exp_ag2,  '--y', label='$g2:  |F,mF>=|2,0>$')
            a2.plot(t, exp_ag2P, '--m', label='$g2P: |F,mF>=|2,+1>$')
            a2.plot(t, exp_ag2PP,'--k', label='$g2PP:|F,mF>=|2,+2>$')
            a2.legend(loc='best')


            b1.plot(t, exp_ax1M_d1,   'r', label='$x1M_{D1}: |F,mF>=|1,-1>$')
            b1.plot(t, exp_ax1_d1,  '--c', label='$x1_{D1}:  |F,mF>=|1,0>$')
            b1.plot(t, exp_ax1P_d1, '--m', label='$x1P_{D1}: |F,mF>=|1,+1>$')
            b1.legend(loc='best')

            b2.plot(t, exp_ax2MM_d1,  'b', label='$x2MM_{D1}:|F,mF>=|2,-2>$')
            b2.plot(t, exp_ax2M_d1, '--r', label='$x2M_{D1}: |F,mF>=|2,-1>$')
            b2.plot(t, exp_ax2_d1,  '--y', label='$x2_{D1}:  |F,mF>=|2,0>$')
            b2.plot(t, exp_ax2P_d1, '--m', label='$x2P_{D1}: |F,mF>=|2,+1>$')
            b2.plot(t, exp_ax2PP_d1,'--k', label='$2PP_{D1}:|F,mF>=|2,+2>$')
            b2.legend(loc='best')

            return fig_d1


        else:
            [
                ag1M, ag1, ag1P,
                ag2MM, ag2M, ag2, ag2P, ag2PP,
                ax0,
                ax1M, ax1, ax1P,
                ax2MM, ax2M, ax2, ax2P, ax2PP,
                ax3MMM, ax3MM, ax3M, ax3, ax3P, ax3PP, ax3PPP
            ] = self.ketbras.getrb_aList(ketbras)
            [exp_ax0,
            exp_ax1,exp_ax1P,exp_ax1M,
            exp_ax2MM,exp_ax2M,exp_ax2,exp_ax2P,exp_ax2PP,
            exp_ax3MMM,exp_ax3MM,exp_ax3M,exp_ax3,exp_ax3P,exp_ax3PP,exp_ax3PPP] = [
                np.real( np.array([(xLev*a).tr() for xLev in output_states]) )
                for a in [ax0,
                        ax1,ax1P,ax1M,
                        ax2MM,ax2M,ax2,ax2P,ax2PP,
                        ax3MMM,ax3MM,ax3M,ax3,ax3P,ax3PP,ax3PPP]
            ]
        
            [exp_ag1,exp_ag1P,exp_ag1M,
            exp_ag2MM,exp_ag2M,exp_ag2,exp_ag2P,exp_ag2PP] = [
                np.real( np.array([(xLev*a).tr() for xLev in output_states]) ) 
                for a in [ag1,ag1P,ag1M,
                        ag2MM,ag2M,ag2,ag2P,ag2PP]
            ]

            # Plotting
            fig_d2, (a1,a2,a3,a4,a5,a6) = plt.subplots(6,1,figsize=(14,12))

            a1.plot(t, exp_ag1M,   'r', label='$g1M: |F,mF>=|1,-1>$')
            a1.plot(t, exp_ag1,  '--c', label='$g1:  |F,mF>=|1,0>$')
            a1.plot(t, exp_ag1P, '--m', label='$g1P: |F,mF>=|1,+1>$')
            a1.legend(loc='best')

            a2.plot(t, exp_ag2MM,  'b', label='$g2MM:|F,mF>=|2,-2>$')
            a2.plot(t, exp_ag2M, '--r', label='$g2M: |F,mF>=|2,-1>$')
            a2.plot(t, exp_ag2,  '--y', label='$g2:  |F,mF>=|2,0>$')
            a2.plot(t, exp_ag2P, '--m', label='$g2P: |F,mF>=|2,+1>$')
            a2.plot(t, exp_ag2PP,'--k', label='$g2PP:|F,mF>=|2,+2>$')
            a2.legend(loc='best')

            a3.plot(t, exp_ax0,  'b', label='$x0_{D2}:  |F,mF>=|0,0>$')
            a3.legend(loc='best')

            a4.plot(t, exp_ax1M,   'r', label='$x1M_{D2}: |F,mF>=|1,-1>$')
            a4.plot(t, exp_ax1,  '--c', label='$x1_{D2}:  |F,mF>=|1,0>$')
            a4.plot(t, exp_ax1P, '--m', label='$x1P_{D2}: |F,mF>=|1,+1>$')
            a4.legend(loc='best')

            a5.plot(t, exp_ax2MM,  'b', label='$x2MM_{D2}:|F,mF>=|2,-2>$')
            a5.plot(t, exp_ax2M, '--r', label='$x2M_{D2}: |F,mF>=|2,-1>$')
            a5.plot(t, exp_ax2,  '--y', label='$x2_{D2}:  |F,mF>=|2,0>$')
            a5.plot(t, exp_ax2P, '--m', label='$x2P_{D2}: |F,mF>=|2,+1>$')
            a5.plot(t, exp_ax2PP,'--k', label='$x2PP_{D2}:|F,mF>=|2,+2>$')
            a5.legend(loc='best')

            a6.plot(t, exp_ax3MMM,  'b', label='$x3MMM_{D2}:|F,mF>=|3,-3>$')
            a6.plot(t, exp_ax3MM, '--r', label='$x3MM_{D2}: |F,mF>=|3,-2>$')
            a6.plot(t, exp_ax3M,  '--y', label='$x3M_{D2}:  |F,mF>=|3,-1>$')
            a6.plot(t, exp_ax3,  '--c', label='$x3_{D2}:  |F,mF>=|3,0>$')
            a6.plot(t, exp_ax3P, '--m', label='$x3P_{D2}: |F,mF>=|3,+1>$')
            a6.plot(t, exp_ax3PP,'--k', label='$x3PP_{D2}:|F,mF>=|3,+2>$')
            a6.plot(t, exp_ax3PPP,'--g', label='$x3PPP_{D2}:|F,mF>=|3,+3>$')
            a6.legend(loc='best')

            return fig_d2

