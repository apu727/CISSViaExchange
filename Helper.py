import scipy.integrate as integrate
import numpy as np

#Fundamental constants
#SI
# e = 1.60217663e-19
# h = 6.62607015e-34#2*np.pi
# hbar = h/(2*np.pi)
# me = 9.1093837e-31
# meff = me
# kb = 1.380649e-23
# T = 0.01 #Kelvin
# E_0 = 8.8541878188e-12 # vaccum permittivity

#Atomic Units
e = 1
h = 2*np.pi
hbar = 1
me = 1
meff = me
EnergyUnit = 4.3597447222060e-18 # Hartree
LengthUnit = 5.29177210544e-11# Bohr radius in metres
ElectricPotentialUnit = 27.211386245981 # in Volts
kb = 1.380649e-23 / EnergyUnit # Hartrees per kelvin
T = 237 #Kelvin
E_0 = 0.25/np.pi # vaccum permittivity
sigmax = np.array([[0,1],[1,0]])
sigmay = np.array([[0,-1j],[1j,0]])
sigmaz = np.array([[1,0],[0,-1]])


def integrateComplex(func, boundary1, boundary2, **kwargs):
    return integrate.quad(lambda x : func(x).real,boundary1,boundary2, **kwargs)[0] + 1j*integrate.quad(lambda x : func(x).imag,boundary1,boundary2, **kwargs)[0]

def integrateRealVector(func, boundary1, boundary2, **kwargs):
    return integrate.quad_vec(func ,boundary1,boundary2, **kwargs)[0]

def fermiFunction(E,mu):
    if (E-mu)/(kb*T) > 30:
        return 0 # e^-30 ~ 0
    return 1/(np.exp((E-mu)/(kb*T))+1)
def bose_EinsteinDistribution(omega):
    if (hbar*omega)/(kb*T) > 30:
        return 0 # e^-30 ~ 0
    return 1/(np.exp((hbar*omega)/(kb*T))-1)

def deltaFunction(x,Precision):
    return np.exp(-0.5*(x**2)/(Precision**2))/(Precision*np.sqrt(2*np.pi))

def linearInterpolate(Data, Energy, lookup,stepsize):
    """interpolate between floor(Energy) and ceil(Energy)"""
    index = (Energy - lookup[0])/stepsize
    substep = index - np.floor(index)
    indexFloor = int(np.floor(index))
    indexCeil = int(np.ceil(index))
    if (Data.shape[0] == 1):
        return Data[0] #Not initialised
    if indexCeil >= len(Data):
        #print(f"Ran out of data upwards, need to extrapolate Value = {Data[-1]}")
        return Data[-1]
    elif indexFloor < 0:
        #print(f"Ran out of data downwards, need to extrapolate Value = {Data[0]}")
        return Data[0]
    
    if not (lookup[indexFloor] <= Energy and lookup[indexCeil] >= Energy):
        print("Error in interpoalte function, guessed wrong")
    
    
    return Data[indexFloor]*(1-substep) + substep*Data[indexCeil]

def deltaij(i,j):
    if i==j:
        return 1
    return 0

class CachingClass:
    def __init__(self,EnergyScale, Gn):
        self.Gn = Gn
        self.GnEnergyScale = EnergyScale
        self.GnCacheValid=False
    
    def __call__(self,E):
        return self.get(E)    
    
    def computeAndCacheFunc(self, EnergyScale, Gn):
        self.GnCache = np.array([Gn(E) for E in EnergyScale])
        self.GnEnergyScale = EnergyScale
        self.GnCacheValid = True
        self.Gn = Gn
    
    def setEnergyScale(self, EnergyScale):
        #self.computeAndCacheFunc(EnergyScale,self.Gn)
        self.GnEnergyScale = EnergyScale
        self.GnCacheValid=False

    def get(self,E):
        if (not self.GnCacheValid):
            self.computeAndCacheFunc(self.GnEnergyScale,self.Gn)
        
        return linearInterpolate(self.GnCache, E, self.GnEnergyScale, self.GnEnergyScale[1]-self.GnEnergyScale[0])
    
    def invalidate(self):
        self.GnCacheValid = False

class GreensFunction:
    def __init__(self, EigenValueMatrix,EnergyRange):
        self.setNewOperator(EigenValueMatrix,EnergyRange)
    def __call__(self, E):
        return self.get(E)
    def setNewOperator(self, operatorMatrix,EnergyRange):
        self.operatorMatrix = operatorMatrix
        self.EnergyRange = EnergyRange
        self.Eig = [np.linalg.eig(operatorMatrix(E)) for E in EnergyRange]
        self.adjointEig = [np.linalg.eig(np.conj(operatorMatrix(E))) for E in EnergyRange]

        for e1,e2 in zip(self.Eig[0][0],self.adjointEig[0][0]):
            assert(np.abs(e1-np.conj(e2)) < 1e-9)

        self.matrices = np.array([[np.outer(psi,phi) for psi,phi in zip(self.Eig[i][1],self.adjointEig[i][1])] for i in range(len(self.Eig))])
        
    def update(self,EnergyRange):
        self.setNewOperator(self.operatorMatrix,EnergyRange)
    def get(self,Energy):
        index = (Energy - self.EnergyRange[0])/(self.EnergyRange[1]-self.EnergyRange[0])
        
        indexFloor = int(np.floor(index))
        indexCeil = int(np.ceil(index))

        if indexCeil >= len(self.matrices):
            #print(f"Ran out of data upwards, need to extrapolate Value = {Data[-1]}")
            return np.sum(self.matrices[-1]/(Energy-self.Eig[-1][0]),axis=0)
        elif indexFloor < 0:
            #print(f"Ran out of data downwards, need to extrapolate Value = {Data[0]}")
            return np.sum(self.matrices[0]/(Energy-self.Eig[0][0]),axis=0)
        
        if not (self.EnergyRange[indexFloor] <= Energy and self.EnergyRange[indexCeil] >= Energy):
            print("Error in interpoalte function, guessed wrong")

        substep = index - np.floor(index)

        return (1-substep)*np.sum(self.matrices[indexFloor]/(Energy-self.Eig[indexFloor][0]),axis=0) + substep*np.sum(self.matrices[indexCeil]/(Energy-self.Eig[indexCeil][0]),axis=0)
    