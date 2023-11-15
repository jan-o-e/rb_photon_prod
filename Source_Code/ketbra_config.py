
from itertools import product
from qutip import tensor, basis

class rb_atom_ketbras:
    def __init__(self, atomStates:dict, xlvls, photonicSpace:bool):
        '''setup Hilbert space for rb atom
        input args:
        atomStates: dictionary of atomic states
        xlvls: list of excited states
        photonicSpace: boolean as to whether we want to include the photonic Hilbert space for the cavity'''
        
        self.atomStates = atomStates
        self.xlvls = xlvls
        # Add the excited-states already configured.
        for k,v in zip(self.xlvls, range(len(self.atomStates), len(self.atomStates)+len(self.xlvls))):
            self.atomStates[k]=v
        self.totalstate_arr= ["g1M", "g1", "g1P", "g2MM", "g2M", "g2", "g2P", "g2PP"] + xlvls
        self.totalstate_arr_d1=["g1M", "g1", "g1P", "g2MM", "g2M", "g2", "g2P", "g2PP", 'x1M_d1','x1_d1','x1P_d1','x2MM_d1','x2M_d1','x2_d1','x2P_d1','x2PP_d1']
        self.totalstate_arr_d2=["g1M", "g1", "g1P", "g2MM", "g2M", "g2", "g2P", "g2PP",'x0','x1M','x1','x1P','x2MM','x2M','x2','x2P','x2PP','x3MMM', 'x3MM','x3M','x3','x3P','x3PP', 'x3PPP']
        self.totalstate_arr_a=['a' + item for item in self.totalstate_arr]
        #boolean as to whether we want to include the photonic Hilbert space
        self.photonicSpace = photonicSpace

    def get_ket_atomic(self, atom):
        # A dictionary of the atomic states.
        M = len(self.atomStates) 
        return tensor(basis(M, self.atomStates[atom]))
 
    def get_ket(self, atom, cavX, cavY):
        '''returns ket for a particular atomStates dictionary and input strings for the atomic and photonic states atom, cavX and cavY'''

        #This defines where we truncate the Fock Space for the cavity
        N = 2
        return tensor(self.get_ket_atomic(atom), basis(N, cavX), basis(N, cavY))

    def getrb_ketbras(self):
        '''returns a dictionary of ketbras from an excited state array and complete atomStates dictionary'''

        if self.photonicSpace:

            def ket(atom, cavX, cavY):
                return self.get_ket(atom , cavX, cavY)

            def bra(atom, cavX, cavY):
                return ket(atom, cavX, cavY).dag()

            ketbras = {}
            s=[ ["g1M", "g1", "g1P", "g2MM", "g2M", "g2", "g2P", "g2PP"] + self.xlvls, [0, 1], [0,1] ]
            states = list(map(list, list(product(*s))))

            for xLev in list(map(list, list(product(*[states,states])))):
                ketbras[str(xLev)] = ket(*xLev[0])*bra(*xLev[1])

        else:
            def ket(atom):
                return self.get_ket_atomic(atom)

            def bra(atom):
                return ket(atom).dag()

            ketbras = {}
            s=self.totalstate_arr
                
            for xLev in [[[x],[y]] for x in s for y in s]:
                ketbras[str(xLev)] = ket(*xLev[0])*bra(*xLev[1])

        return ketbras



    #faster if you use precomputed ketbras than rerunning the function get_ketbras
    def getrb_aDict(self,ketbras,d_line='both'):
        '''returns a dictionary of atomic population operators from an excited state array and complete atomStates dictionary
        input args:
        ketbras: dictionary of ketbras from getrb_ketbras
        d_line: 'both' for d1 and d2, 'd1' for d1 only, 'd2' for d2 only'''

        def kb(xLev,y):
            return ketbras[str([xLev,y])]
        aDict = {}
        def createStateOp(s):
            try:
                if self.photonicSpace:
                    aOp = kb([s,0,0],[s,0,0]) + kb([s,1,0],[s,1,0]) + kb([s,0,1],[s,0,1])+ kb([s,1,1],[s,1,1])
                    aDict[s]=aOp
                else:
                    aOp = kb([s],[s])
                    aDict[s]=aOp
            except KeyError:
                aOp =  None
            return aOp

        if d_line=='both':
            for s in self.totalstate_arr:
                createStateOp(s)
        elif d_line=='d1':
            for s in self.totalstate_arr_d1:
                createStateOp(s)
        elif d_line=='d2':
            for s in self.totalstate_arr_d2:
                createStateOp(s)

        return aDict

#get list of atomic population operators
    def getrb_aList(self, ketbras, d_line='both'):
        '''returns a list of atomic population operators from an excited state array and complete atomStates dictionary
        input args:
        ketbras: dictionary of ketbras from getrb_ketbras
        d_line: 'both' for d1 and d2, 'd1' for d1 only, 'd2' for d2 only'''

        def kb(xLev,y):
            return ketbras[str([xLev,y])]
        aDict = {}
        def createStateOp(s):
            try:
                if self.photonicSpace:
                    aOp = kb([s,0,0],[s,0,0]) + kb([s,1,0],[s,1,0]) + kb([s,0,1],[s,0,1])+ kb([s,1,1],[s,1,1])
                    aDict[s]=aOp
                else:
                    aOp = kb([s],[s])
                    aDict[s]=aOp
            except KeyError:
                aOp =  None
            return aOp
        
        if d_line=='both':
            return([createStateOp(s) for s in self.totalstate_arr])
        elif d_line=='d1':
            return([createStateOp(s) for s in self.totalstate_arr_d1])
        elif d_line=='d2':
            return([createStateOp(s) for s in self.totalstate_arr_d2])

    

#returns a dictionary of ketbras from an excited state array and complete atomStates dictionary, faster is you use precomputed ketbras than rerunning the function get_ketbras
    def getrb_arhoDict(self, ketbras):

        def kb(xLev,y):
            return ketbras[str([xLev,y])]
        aRhoDict = {}

        if self.photonicSpace:
            def createStateOp_rho(i,j,x,y):
                try:
                    a = kb([i,x,y],[j,x,y])
                    aRhoDict[i,j,x,y]=a
                except KeyError:
                    a =  None
                return a
            
            for i in self.totalstate_arr:
                for j in self.totalstate_arr:
                    for m in [0,1]:
                        for n in [0,1]:
                            createStateOp_rho(i,j,m,n)
        else:
            def createStateOp_rho(i,j):
                try:
                    a = kb([i],[j])
                    aRhoDict[i,j]=a
                except KeyError:
                    a =  None
                return a
            
            for i in self.totalstate_arr:
                for j in self.totalstate_arr:
                    createStateOp_rho(i,j)

        return aRhoDict


    #define function to recompute initial state from intermediary simulation run which take input as an output_states[-1] array from mesolve
    def recompute_psi(self,ketbras,input):
        '''returns the new initial state from an intermediary simulation run
        input args:
        ketbras: dictionary of ketbras from
        input: output_states[-1] array from mesolve
        '''
        psiList=[]
        aDict=self.getrb_aDict(ketbras)
        arhoDict=self.getrb_arhoDict(ketbras)
        if self.photonicSpace:
            for k in aDict:
                for b in aDict:
                    for x in [0,1]:
                        for y in [0,1]:
                            psiList.append(((input*arhoDict[k,b,x,y]).tr() ,ketbras["[['{0}', {2}, {3}], ['{1}', {2}, {3}]]".format(k,b,x,y)]))    
        else:
            for k in aDict:
                for b in aDict:
                    psiList.append(((input*arhoDict[k,b]).tr() ,ketbras["[['{0}'], ['{1}']]".format(k,b)]))
        
        return (sum([x*y for x,y in psiList]))