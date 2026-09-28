
from cppNEGF import BasisManager,ComplexDualSelfAdjointMatrix,SpinSymmetry,ComplexSelfAdjointMatrix,SparseTensor_4_Complex,ComplexDirectSelfAdjointMatrix,FockHamiltonian,MatrixSelfEnergy,ComplexDirectMatrix,ComplexMatrix,InterfaceSelfEnergy,InterfaceLeadType,SubsystemFlags

from Plots import *
from loadQChemOutput import *
from simpleLeads import *
from pyscfInterface import generateIntegrals, pyscfGen, getSpinorVersionBAD
import numpy as np
import itertools as it

import time
import os.path
from dataclasses import dataclass, field
import copy


np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)
def getSpinorVersion(Mat):
    return Mat.toBasis(type(Mat)(),BasisManager.removeSpinTags(Mat.getBasis()))
#User Settings
@dataclass
class GreensFunctionMethodParameters:
    activeOrbitals = None
    frozenOrbitals = set()
    outputName = ""

    #Lead parameters
    startGamma = 0
    maxGamma = 0
    RightGamma = 0 # None to set the same as left
    GammaSteps = 1
    muLeft = 0#-0.171847#5
    startVoltage = 0*globals.electricPotentialUnit #10
    EndVoltage = 0*globals.electricPotentialUnit #10
    VoltageSteps = 2

    TStart = globals.T
    TEnd = globals.T
    TStep = 0

    maintainElectronNumber = None
    muMax = 200 # Pointless?

    coupleBasedOnElement = True
    ElementCoupleLeft = [] 
    ElementCoupleRight = [] 
    SpinDifference = np.array([0,0,0.00])*2 # This is divided by 2 for some reason
    RightSpinDifference = np.array([0,0,0.0])*2

    #PhotoElectronLeads
    photoElectronGamma = 1e-4
    photoElectronBandBottom = None # None means no bottom
    photoElectronSpin = [0,0,0.00]
    photoElectronElementCouple = []
    photoElectronExcitationEnergy = 0#125.9/globals.electricPotentialUnit # 5.9eV # gets added to muLeft

    ScatteringGamma = 1e-2
    ScatteringBandBottom = None # AKA vaccum level
    ScatteringElementCouple = [17]
    ScatteringSpin = [0,0,0]
    ScatteringMu = muLeft




    #less common user settings
    atomString = None
    subsystems = None
    subsystemViaInterface = False
    SubsystemOpt = SubsystemFlags.false
    basis = "sto-3g"
    charge = 0
    spin = 0
    useHFOrbitals = False
    useOrthogonalOrbitals = False # Takes effect only if youre not using HF orbitals
    usingDualSpace : bool = field(init=False)
    workInLowdinBasis = False
    forceSelfAdjointSelfEnergy = False
    loadFromQChemData = True
    
    useEDIIS = False
    CDIISInitialAlpha = 0.3

    doOldLeads = False # Couples to leads via an identity in the AO basis. This creates very large complex eigenvalues. 
    twoLeads = True
    doPhotoElectronLeads = False
    doPhotoElectronSCF = False

    doSOC = False
    spinLambda = 1 #Rescale the non spin symmetric part of the exchange matrix by this factor. 
    # SOCScaleFactor = 1# Multiplies the SOC contribution by a scale factor. This is plain wrong but can give useful results. Now in PyscfInterface

    computeEnergy = None 
    SCFRead = False # Reads a finalDensity file if it can find one
    doSCF = True # Actually do SCF or just use the loaded density. Usually you want to use SCF
    doRestrictedCalc = True
    SCFConvergence = 10 # Same as QCHEM, Converged when trace changes by less than 10**-SCFConvergence
    doAintegral = False
    doGnIntegral = None

    saveFrozenCore = False
    loadFrozenCore = False
    frozenCoreEnergy = -1
    doPlots = True
    doCurrentPlots = True
    doSpectralPlots = True
    doNonEqGnIntegral = False
    NonEqGnStart = 0
    NonEqGnEnd = 0

    AllowFailConverge = False

    def finalise(self):
        if self.maintainElectronNumber is None: 
            self.maintainElectronNumber = True if self.startVoltage == self.EndVoltage and not self.muLeft is None and self.startVoltage + self.muLeft == 0 else False
        
        self.useHFOrbitals = self.useHFOrbitals or len(self.frozenOrbitals) > 0
        self.usingDualSpace = not (self.useHFOrbitals or self.useOrthogonalOrbitals) # Only use dual space if both are false.
        if self.loadFromQChemData:
            self.frozenOrbitals = set()
        if self.computeEnergy is None:
            self.computeEnergy = False if self.doSOC else True
        if self.doGnIntegral is None:
            self.doGnIntegral = False if self.doSOC else True
        if self.loadFromQChemData:
            self.frozenOrbitals = set()
        if self.forceSelfAdjointSelfEnergy:
            if self.startGamma != 0 or self.maxGamma != 0 or self.RightGamma != 0 or self.doPhotoElectronLeads:
                print("Are you sure this is a self adjoint problem?!!")
                print("##############################################")
        if not self.subsystems is None and len(self.subsystems) < 2:
            print(f"Subsystem length is {len(self.subsystems)}")
            print(f"Since this is less than 2, we are ignoring and pretending it doesnt exist")
        if self.subsystemViaInterface:
            self.workInLowdinBasis = True
        
        
    def validate(self):
        ok = True
        if len(self.outputName) == 0:
            print(f"outputname invalid {self.outputName}")
            ok = False
        if len(self.ElementCoupleLeft) == 0:
            print(f"ElementCoupleLeft invalid {self.ElementCoupleLeft}")
            ok = False
        if not self.loadFromQChemData and self.atomString is None:
            print("Need to specify atomstring if not loading from qchem")
            ok = False
        if self.doNonEqGnIntegral and self.NonEqGnEnd == self.NonEqGnStart:
            print("self.doNonEqGnIntegral and self.NonEqGnEnd == self.NonEqGnStart is meaningless")
            ok = False
        return ok

@dataclass
class GreensFunctionResults:
    CalcEnergy = None
    SCFEnergy = None
    InterfacesEnergy = None


class GreensFunctionMethod:
   

    def __init__(self,params : GreensFunctionMethodParameters):
        self.params = params
        self.results = GreensFunctionResults()
        if not params.validate():
            print("ok = false")
            quit(1)

        print(f"Settings:\nBasis: {self.params.basis}\nCharge:{self.params.charge}\nSpin:{self.params.spin}\nUseHFOrbitals:{self.params.useHFOrbitals}\nOrthogonalOrbitals:{self.params.useOrthogonalOrbitals}\nUsingDualSpace:{self.params.usingDualSpace}\nQchem?:{self.params.loadFromQChemData}\n\n")

    def load(self):
        if self.params.loadFromQChemData:
            S,F,D,eri,oneEInts,fchkParams,success = loadQChem(self.params.outputName)
        else:
            RHF = pyscfGen(self.params.atomString,self.params.basis,self.params.charge,self.params.spin)
            D = RHF.make_rdm1()*0.5 # We want only the Alpha block not the full `spatial` density matrix
            vj,vk = RHF.get_jk()
            F = vj-0.5*vk # Only want Alpha Alpha block
            
            fchkParams = {}
            fchkParams["Basis"] = self.params.basis
            fchkParams["NBasis"] = F.shape[0]
            fchkParams["NoAtoms"] = RHF.mol.natm
            fchkParams["AtomicNumbers"] =  RHF.mol.atom_charges()
            fchkParams["NoElectrons"] = RHF.mol.nelec[0] + RHF.mol.nelec[1]
            fchkParams["SCFEnergy"] = RHF.energy_tot()
            fchkParams["NoAlphaElectrons"] = RHF.mol.nelec[0]
            fchkParams["NoBetaElectrons"] = RHF.mol.nelec[1]
            fchkParams["BetaMoCoeff"]  = fchkParams["AlphaMoCoeff"] = RHF.mo_coeff
            fchkParams["Comment"] = RHF.mol.atom
            self.AtomToAOMap = []
            for sh0,sh1,AO0,AO1 in RHF.mol.aoslice_by_atom():
                self.AtomToAOMap.append(list(range(AO0,AO1)))
            


        Generate2eInts = False
        if Generate2eInts:
            memEstimate = 8*fchkParams["NBasis"]**4/1e9
            print(f"2eInt memoryEstimate: {memEstimate}GB for {fchkParams["NBasis"]} Basis functions")
            if (memEstimate > 8):
                cont = input("Continue?, Y/n")
                if (cont.lower() != "y"):
                    print("Quitting")
                    quit()
        isECP = "ECPMatrix" in fchkParams
        if self.params.loadFromQChemData:
            atomString = generatePyscfAtomString(fchkParams)        
        else:
            atomString = self.params.atomString
        hcore,eri,nuclearEnergy,S_AO,hecp,makeJK,hSOC = generateIntegrals(atomString,fchkParams["Basis"],isECP,Generate2eInts=Generate2eInts,BreitIntegrals=self.params.doSOC,atomicCharges=fchkParams["AtomicNumbers"],namedBasis="AO",spinLambda=self.params.spinLambda)
        S_AO.asMetric()
        if isECP:
            hcore += hecp
        #Generate transformation to make these compatible
        if self.params.loadFromQChemData:
            permutationMatrix,cbs,state = computeQChemToPyscfPermutation(self.params.outputName,fchkParams) # P^{i}_{j} = c'^i c_j where c'^i is in the pyscf basis, c_j is in the qchem basis. A normal vector is c^j\ket{e_j} so this works as expected
        else:
           oneEInts = hcore.toNumpy()
           S = S_AO
           permutationMatrix = np.identity(fchkParams["NBasis"])
        
        PInv = np.linalg.inv(permutationMatrix).T
        #oneEInts is the spatial one E Integrals. D is the Alpha Alpha Density matrix. F is the Alpha Alpha block of the Fock matrix built from this AlphaAlpha Density matrix. 
        # I.e. H^\t{1e}_{\alpha \alpha}, D_{\alpha\alpha} F_\alpha\alpha
        print(f"E before Transformation: {np.trace(F@D) + 2*np.trace(oneEInts@D)+nuclearEnergy}")
        D = np.einsum("ip,jq,pq->ij",permutationMatrix,permutationMatrix,D) # P^{i}_{p} D^{p\Bar{q}} P^{j}_{q}
        F = np.einsum("ip,jq,pq->ij",PInv,PInv,F) #  (PInv)_{i}^{p} (PInv)_{j}^{q} F_{pq}
        oneEInts = np.einsum("ip,jq,pq->ij",PInv,PInv,oneEInts) #  (PInv)_{i}^{p} (PInv)_{j}^{q} F_{pq}            
        print(f"E After Transformation: {np.trace(F@D) + 2*np.trace(oneEInts@D)+nuclearEnergy}")
        if D.shape[-1] != S_AO.rows():
            SpinorD = getSpinorVersion(ComplexDualSelfAdjointMatrix(D,"AO" + BasisManager.Alpha_Block,SpinSymmetry.RHF)) 
        else:
            SpinorD = getSpinorVersion(ComplexDualSelfAdjointMatrix(D,"AO",SpinSymmetry.NoSpinSym)) 
        spinorHcore = getSpinorVersion(hcore)
        FSpinor = makeJK(SpinorD) + spinorHcore
        # print(f"E using new integrals: {0.5*(np.trace(spinorHcore.toNumpy()@SpinorD.toNumpy()) + np.trace(FSpinor.toNumpy() @ SpinorD.toNumpy()))+nuclearEnergy}")
        print(f"E using new integrals: {0.5*((SpinorD * spinorHcore).trace() + (SpinorD * FSpinor).trace())  +nuclearEnergy}")

        print("using NEW Integrals!!!!!")
        S = S_AO
        oneEInts = spinorHcore

        #Setup HOMO Lumo
        # FockEigenValues = np.linalg.eigvals(np.linalg.inv(getSpinorVersion(S_AO))@FSpinor)
        FockEigenValues = FSpinor.toBasis(ComplexSelfAdjointMatrix(),"AO").getEigenValues()
        FockEigenValues = np.sort(FockEigenValues.toNumpy())
        self.Homo = FockEigenValues[fchkParams["NoElectrons"]-1]
        if (fchkParams["NoElectrons"] == len(FockEigenValues)):
            #There is no Lumo
            self.Lumo = self.Homo + 2
        else:
            self.Lumo = FockEigenValues[fchkParams["NoElectrons"]]
        print(f"Homo:{self.Homo}, Lumo: {self.Lumo}")
        if self.params.muLeft is None:
            self.params.muLeft = 0.5*(self.Homo+self.Lumo)

        #Need to setup Atom -> AO Map now while we have `logical` access to cbs
        if self.params.loadFromQChemData:
            AOBoundaries = cbs.function_indices_for_item_starts
            AOBoundaries.append(cbs.n_functions) # stop index
            self.AtomToAOMap = []
            for idx in range(len(AOBoundaries)-1):
                self.AtomToAOMap.append(list(range(AOBoundaries[idx],AOBoundaries[idx+1])))
        self.AtomicNumbers = fchkParams["AtomicNumbers"]
        #TODO pretty names for atoms? 
        # Can be extracted from cbs.basis_items
        

        self.makeJK = makeJK
    
        
            
            
        if "SCFEnergy" in fchkParams:
            print(f"QCHEM: SCFEnergy: {fchkParams["SCFEnergy"]}")
            self.results.SCFEnergy = fchkParams["SCFEnergy"]

        self.S = S
        self.F = F
        self.DSpinor = SpinorD # Assumes restricted calculation
        if not eri is None:
            self.eri = eri
        self.oneEH = oneEInts
        if self.params.doSOC:
            self.oneEHSOC = hSOC
        
        self.Source = "QCHEM" if self.params.loadFromQChemData else "PYSCFGUESS"
        
        self.Molecule = fchkParams["Comment"]
        self.AlphaOccupations = list(range(fchkParams["NoAlphaElectrons"]))
        self.BetaOccupations =  list(range(fchkParams["NoBetaElectrons"]))
        self.NumAlphaElectrons = fchkParams["NoAlphaElectrons"]
        self.NumBetaElectrons = fchkParams["NoBetaElectrons"]
        if self.params.subsystemViaInterface:
            assert(self.params.charge == 0)
            assert(self.params.spin == 0)
            self.NumElectrons = np.sum(fchkParams["AtomicNumbers"][slice(*self.params.subsystems[0])])
            self.NumAlphaElectrons = self.NumElectrons/2
            self.NumBetaElectrons = self.NumElectrons/2
        else:
            self.NumElectrons = fchkParams["NoElectrons"]
        if self.params.useHFOrbitals or self.params.useOrthogonalOrbitals:
            if self.params.useHFOrbitals:
                AllOrbs = fchkParams["AlphaMoCoeff"]    
                self.AOToMoBasisChangeMatrix = AllOrbs
                self.Basis = "HF"
            
            if self.params.useOrthogonalOrbitals:
                S_AO = self.S
                O,V = np.linalg.eigh(S_AO)
                O = np.power(O,-0.5)
                AllOrbs = V @ np.diag(O) @ V.T
                self.AOToMoBasisChangeMatrix = AllOrbs
                self.Basis = "ORTH"
            #Transform all the matrices
            self.F = np.einsum("pi,pq,qj->ij", AllOrbs, self.F, AllOrbs)

            D = np.einsum("pi,pq,qj->ij", AllOrbs, D, AllOrbs)
            self.DSpinor = getSpinorVersion(D) # Assumes restricted calculation

            self.eri = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri,AllOrbs, AllOrbs, AllOrbs, AllOrbs)
            self.oneEH = np.einsum("pi,pq,qj->ij", AllOrbs, self.oneEH, AllOrbs)
        elif self.params.loadFrozenCore and os.path.isfile(outputName + "FrozenCore.npz"):
            # Frozen Core nonOrthogonalBasis
            assert(eri is None)
            FrozenCoreNPZ = np.load(outputName + "FrozenCore.npz") # These are all in the AO dual basis. 
            LProj = FrozenCoreNPZ["LProj"]
            LInvProj = FrozenCoreNPZ["LInvProj"]
            CProj = FrozenCoreNPZ["CProj"]
            CInvProj = FrozenCoreNPZ["CInvProj"]
            LGn = FrozenCoreNPZ["LGn"]
            CGn = FrozenCoreNPZ["CGn"]
            LE = FrozenCoreNPZ["LE"]
            CE = FrozenCoreNPZ["CE"]
            print(f"Frozen core energy:{np.real(LE[0,0])}")
            nuclearEnergy += np.real(LE)[0,0] # Electronic energy of the core
            
            #AO metrics
            loweringMetricSpatial = self.S
            loweringMetric = getSpinorVersion(loweringMetricSpatial)
            raisingMetricSpatial = np.linalg.inv(loweringMetricSpatial)
            raisingMetric = getSpinorVersion(raisingMetricSpatial)

            self.S = None
            self.SSpinor = np.conj(CProj.T) @ loweringMetric @ CProj # C^\bar{i}_\bar{l} S_{\bar{i}j} C^j_m, The metric for two direct basis kets in the projected basis
            self.SRaisingSpinor = np.linalg.inv(self.SSpinor)

            self.Basis = "FC"
            self.CProj = CProj # (AO x Active dimension) transformation matrix. R^-1 H R is the transformation to get to the basis
            self.CInvProj = CInvProj
            self.CInvProjDirecttoDirect =  self.SSpinor @ CInvProj @ raisingMetric # Projects from the AO basis to the direct basis of the frozen core
            self.LProj = LProj
            self.LInvProj = LInvProj
            
            oneEInts = np.array(getSpinorVersion(oneEInts),dtype=np.complex128)
            
            # oneEInts += makeJK(LGn @ raisingMetric) # Frozen core JK, makeJK works in the AO basis
            oneEInts = self.CInvProjDirecttoDirect @ oneEInts @ CProj # (C^{-1})^i_l S^{lm} h_{mn} C^n_j
            self.oneEH = oneEInts # Need to keep everything in the Direct basis
            if self.params.doSOC:
                self.oneEHSOC = self.CInvProjDirecttoDirect@ hSOC @ CProj # keep in direct basis

            
            self.F = None
            
            self.JKFC = self.CInvProjDirecttoDirect @ makeJK(LGn @ raisingMetric)@ CProj# Frozen core JK, makeJK works in the AO basis, needs to be a self energy or we get thermal energy wrong
            if os.path.isfile(outputName + "finalGn.npy") and self.params.SCFRead:            
                self.DSpinor = np.load(outputName + "finalGn.npy")
                if self.DSpinor.shape[0] != CProj.shape[1]:
                    self.DSpinor = CInvProj @ CGn @ CProj @ self.SRaisingSpinor # end up in the double direct basis
                else:
                    print("loading Density from File")
            else:
                self.DSpinor = CInvProj @ CGn @ CProj @ self.SRaisingSpinor # end up in the double direct basis
            #Replaces Fock Hamiltonian
            # self.JKC = self.CInvProjDirecttoDirect @ makeJK(CGn @ raisingMetric)@ CProj# Frozen core JK, makeJK works in the AO basis, needs to be a self energy or we get thermal energy wrong

            # dm will be provided as   D^{\mu\nu} so need C^i_\mu D^{\mu\nu} (S_C)_{\nu k} (C^{-1})^k_l (S_AO)^{lj} to get back to AO basis
            # Some of these will cancel
            self.makeJK = lambda dm : self.CInvProjDirecttoDirect @ makeJK(CProj @ dm @ self.CInvProjDirecttoDirect) @ CProj

            self.AlphaOccupations = list(range(fchkParams["NoAlphaElectrons"] - LProj.shape[0]//2)) # Is this even right?
            self.BetaOccupations =  list(range(fchkParams["NoBetaElectrons"] - LProj.shape[0]//2))
            self.NumAlphaElectrons = fchkParams["NoAlphaElectrons"] - LProj.shape[0]//2
            self.NumBetaElectrons = fchkParams["NoBetaElectrons"] - LProj.shape[0]//2
            self.NumElectrons = fchkParams["NoElectrons"] - LProj.shape[0] # This is right
            self.NBasis = CProj.shape[1]
        else:
            #Use AOS
            self.Basis = "AO"
            self.NBasis = fchkParams["NBasis"]*2
            self.AOToMoBasisChangeMatrix = np.identity(fchkParams["NBasis"])
            if os.path.isfile(self.params.outputName + "finalGn.npy") and self.params.SCFRead:
                print(f"loading Density from File, assuming basis:{self.Basis}, ComplexSelfAdjointMatrix")
                self.DSpinor = ComplexSelfAdjointMatrix(np.load(self.params.outputName + "finalGn.npy"), self.Basis, SpinSymmetry.NoSpinSym)
                self.DSpinor = self.DSpinor.toBasis(ComplexDualSelfAdjointMatrix(), self.Basis)


        # Assumes restricted calculation
        assert(np.max(np.abs(fchkParams["BetaMoCoeff"] - fchkParams["AlphaMoCoeff"])) == 0)

        assert(len(self.params.frozenOrbitals) == 0) #Not Yet implemented
        #FIXME rethink these and what size is what
        self.NumActiveOrbitals = self.NBasis
        self.NumFrozenOrbitals = 0
        self.NAO = fchkParams["NBasis"]
        self.ActiveOrbitalsList = np.array(list(range(self.NAO)))
        self.FrozenOrbitalsList = np.array([])
        
        self.eri_ijIndexes = {(i,j):idx for idx,(i,j) in enumerate(it.product(self.ActiveOrbitalsList,repeat=2))}

        #Compute nuclear repulsion energy
        NoAtoms = fchkParams["NoAtoms"]
        if not "nuclearEnergy" in locals():
            raise NotImplementedError("Find Nuclear Energy")
        
        self.ConstantEnergyOffset = nuclearEnergy # Since we cant do frozen core
        
    def createFHamFromEri(self):
        #Create FHam from two electron tensor
        activeOrbitalsList  = self.ActiveOrbitalsList
        numFrozenOrbitals = self.NumFrozenOrbitals
        twoElectronIntegrals = self.eri
        ijIndexes = self.eri_ijIndexes


        twoElectronTensor = SparseTensor_4_Complex()
        #twoElectronTensor.setSize(numActiveOrbitals)
        npTwoElectronTensor = np.zeros(shape=(self.NAO,self.NAO,self.NAO,self.NAO))
        for (p,q,r,s) in it.product(activeOrbitalsList,repeat=4):
            #<qp|V|rs>
            idx1 = ijIndexes[(q,s)]
            idx2 = ijIndexes[(p,r)]
            p -= numFrozenOrbitals
            q -= numFrozenOrbitals
            r -= numFrozenOrbitals
            s -= numFrozenOrbitals
            if (abs(twoElectronIntegrals[idx1,idx2]) > 0):
                # twoElectronTensor.setCoeff([p,q,r,s],twoElectronIntegrals[idx1,idx2]) 
                npTwoElectronTensor[p,q,r,s] = twoElectronIntegrals[idx1,idx2]    
        twoElectronTensor.setCoeffs(npTwoElectronTensor)

        # ZeroHam = np.zeros(shape=(self.NBasis,self.NBasis))
        print("Constructing Fock Hamiltonian")
        # FHam = FockHamiltonian.FockHamiltonian(self.DSpinor,ZeroHam,twoElectronTensor,self.makeJK) #For debug, compares the two methods. These seem to give slightly different eigenvectors??
        DM = self.DSpinor
        ZeroHam = ComplexDirectSelfAdjointMatrix().setZero(self.NBasis,self.NBasis,self.Basis,SpinSymmetry.NoSpinSym)
        FHam = FockHamiltonian.FockHamiltonian(DM,ZeroHam,twoElectronTensor) #FHam expects in AO basis if non-orthogonal so this is handeled correctly.
        print("Done constructing Fock Hamiltonian")
        return FHam
    def runHamGen(self) -> GreensFunctionResults:
        #Sets up the myMolObject
        self.load()
        if self.params.subsystemViaInterface:
            paramsCopy = copy.copy(self.params)
            AtomsFound = 0
            paramsCopy.atomString = ""

            print(self.params.atomString)
            for i,c in enumerate(self.params.atomString.split(' ')):
                if not str.isalpha(c):
                    paramsCopy.atomString += c + ' '
                elif AtomsFound < self.params.subsystems[1][0] or AtomsFound >= self.params.subsystems[1][1]:
                    paramsCopy.atomString += "X-" + c + ' '
                    AtomsFound += 1
                else:
                    paramsCopy.atomString += c + ' '
                    AtomsFound += 1
            
            print(paramsCopy.atomString)
            paramsCopy.subsystems = None
            paramsCopy.subsystemViaInterface = False
            LeadGreensFunctionMethod = GreensFunctionMethod(paramsCopy)
            LeadGreensFunctionMethod.load()
        
        
        
        assert(hasattr(self,"S"))
        if self.Basis == "AO":
            AllOrbs = self.AOToMoBasisChangeMatrix
        elif self.Basis == "FC":
            raise NotImplemented("FC Basis")                
        else:
            raise NotImplemented(f"not handled basis{self.Basis}")
        
        if not self.params.subsystems is None and len(self.params.subsystems) >= 2:
            for i,(start,stop) in enumerate(self.params.subsystems):
                idxs = np.array(list(range(start,stop)))
                AOs = [2*AO + x for idx in idxs for AO in self.AtomToAOMap[idx] for x in range(2)] #Nightmare fuel
                
                transform = np.zeros((self.NBasis,len(AOs)))
                transform[AOs,list(range(len(AOs)))] = 1
                transform = ComplexMatrix(transform,["AO_LB" if self.params.workInLowdinBasis else "AO",f"Mol{i}"])
                # transform = transform.toBasis(ComplexMatrix(),["AO",f"Mol{i}"])
                if self.params.workInLowdinBasis:
                    transform.asBasisChangeMatrix()
                else:
                    transform.asProjectionMatrix()
            # self.Basis = "Mol0"
            

            
        
        numFrozenOrbitals = self.NumFrozenOrbitals
        #numActiveOrbitals_ = self.NumActiveOrbitals # May or may not be spin oprbitals
        numActiveSpinOrbitals = self.NBasis
        frozenOrbitalsList = self.FrozenOrbitalsList
        activeOrbitalsList  = self.ActiveOrbitalsList
        hcore = self.oneEH

        #Two electron Integrals in the active space
        if hasattr(self,"eri"):
            FHam = self.createFHamFromEri()
        else:
            assert(hasattr(self,"makeJK"))
            assert(self.params.usingDualSpace) # We need to be in the AO basis otherwise this wont work. In principle it could but there is no need
            DM = self.DSpinor
            #Create FHam using the makeJK object. 
            if self.Basis != "FC":
                # ZeroHam = np.zeros(shape=(numActiveSpinOrbitals,numActiveSpinOrbitals))
                ZeroHam = ComplexDirectSelfAdjointMatrix()
                ZeroHam.setZero(self.NBasis,self.NBasis,"AO",SpinSymmetry.NoSpinSym)
            else:
                ZeroHam = self.JKFC
            print("Constructing Fock Hamiltonian")
            FHam = FockHamiltonian.FockHamiltonian(DM,ZeroHam,self.makeJK)
            print("Done constructing Fock Hamiltonian")

        FHam.setIntegrationParameters(-8,1,500)
        FHam.setUseEDIIS(self.params.useEDIIS)
        FHam.setCDIISInitialAlpha(self.params.CDIISInitialAlpha)
        
        


        ExtendedHCore = ComplexDirectSelfAdjointMatrix(self.NBasis,self.NBasis,self.Basis,SpinSymmetry.RHF)
        if self.Basis != "FC" and hasattr(self,"eri"):
            twoElectronIntegrals = self.eri
            ijIndexes = self.eri_ijIndexesij
            for i in activeOrbitalsList:
                for j in activeOrbitalsList:
                    ijSameSpinFrozenFockElement = 0
                    for k in frozenOrbitalsList:
                        #TODO Frozen core in non-orthogonal basis
                        ijSameSpinFrozenFockElement += 2*twoElectronIntegrals[ijIndexes[(i,j)],ijIndexes[(k,k)]] # <ik|kj> So k=s can be A or B
                        ijSameSpinFrozenFockElement -= twoElectronIntegrals[ijIndexes[(i,k)],ijIndexes[(k,j)]] # <ik|jk> So i=k & k=j so 1 choice
                        
                    Shiftedi = i - numFrozenOrbitals
                    Shiftedj = j - numFrozenOrbitals
                    
                    ExtendedHCore[2*Shiftedi,2*Shiftedj] = hcore[i,j] + ijSameSpinFrozenFockElement
                    ExtendedHCore[2*Shiftedi+1,2*Shiftedj+1] = hcore[i,j] + ijSameSpinFrozenFockElement
        else:
            assert(ExtendedHCore.rows() == hcore.rows())
            ExtendedHCore = hcore
        ExtendedHCore = ExtendedHCore.toBasis(ComplexSelfAdjointMatrix(),ExtendedHCore.getBasis())
        
        ##Need to fix the basis transformation from AO to Mol0 when it is not a projection
        if self.params.workInLowdinBasis:
            ExtendedHCore = ExtendedHCore.toBasis(ComplexSelfAdjointMatrix(),self.Basis+ BasisManager.LowdinOrthogonaliseBasis)
        else:
            ExtendedHCore = ExtendedHCore.toBasis(ComplexSelfAdjointMatrix(),self.Basis)
        oneElecHam = MatrixSelfEnergy.SelfAdjointMatrixSelfEnergy(ExtendedHCore)        
        

        if self.params.doSOC:
            SOCHam = MatrixSelfEnergy.SelfAdjointMatrixSelfEnergy(self.oneEHSOC)
        
        

        #Couple to LHS of molecule
        LHSCoupleMatrix = np.zeros(self.NAO)
        RHSCoupleMatrix = np.zeros(self.NAO)
        for Atom,AOOnAtom in enumerate(self.AtomToAOMap):
            if self.params.coupleBasedOnElement:
                if self.AtomicNumbers[Atom] in self.params.ElementCoupleLeft:
                    LHSCoupleMatrix[AOOnAtom] = 1
                if self.AtomicNumbers[Atom] in self.params.ElementCoupleRight:
                    RHSCoupleMatrix[AOOnAtom] = 1
            else:
                if Atom == 0:
                    LHSCoupleMatrix[AOOnAtom] = 1
                elif Atom == 1:
                    RHSCoupleMatrix[AOOnAtom] = 1
        if self.Basis == "FC":
            LHSCoupleMatrix = np.diag(LHSCoupleMatrix)
            if (np.all(RHSCoupleMatrix == 0)):
                print("Setting to one Lead mode since RHSCouple = 0")
                self.params.twoLeads = False
            RHSCoupleMatrix = np.diag(RHSCoupleMatrix)
            #Basis transformation done in Simple leads after construction in the AO basis
        else:
            #TODO This should use the basis manager
            LHSCoupleMatrix = np.diag(LHSCoupleMatrix) #C_{pi} M_{pq} C_{qj}
            if self.params.doOldLeads:
                LHSCoupleMatrix = ComplexDirectMatrix(LHSCoupleMatrix[numFrozenOrbitals:,numFrozenOrbitals:],"AO" + BasisManager.Alpha_Block,SpinSymmetry.RHF)
            else:
                LHSCoupleMatrix = ComplexDirectMatrix(LHSCoupleMatrix,"AO" + BasisManager.Alpha_Block + BasisManager.LowdinOrthogonaliseBasis,SpinSymmetry.RHF)
            if (np.all(RHSCoupleMatrix == 0)):
                print("Setting to one Lead mode since RHSCouple = 0")
                self.params.twoLeads = False
            RHSCoupleMatrix = np.einsum("pi,pq,qj->ij", AllOrbs,np.diag(RHSCoupleMatrix),AllOrbs) #C_{pi} M_{pq} C_{qj}
            if self.params.doOldLeads:
                RHSCoupleMatrix = ComplexDirectMatrix(RHSCoupleMatrix[numFrozenOrbitals:,numFrozenOrbitals:],"AO" + BasisManager.Alpha_Block,SpinSymmetry.RHF)
            else:
                RHSCoupleMatrix = ComplexDirectMatrix(RHSCoupleMatrix[numFrozenOrbitals:,numFrozenOrbitals:],"AO" + BasisManager.Alpha_Block+ BasisManager.LowdinOrthogonaliseBasis,SpinSymmetry.RHF)


        #AtomProjectors 
        AtomProjectionName = "AtomBasis"
        AtomProjectionMatrixSpatial = np.zeros((len(self.AtomToAOMap),self.NAO)) 
        for Atom,AOOnAtom in enumerate(self.AtomToAOMap):
            AtomProjectionMatrixSpatial[Atom,AOOnAtom] = 1
        if self.Basis == "FC":
            AtomProjectionMatrix = getSpinorVersion(AtomProjectionMatrixSpatial)
            AtomProjectionMatrixInv = self.CInvProj @ AtomProjectionMatrix.T
            AtomProjectionMatrix = AtomProjectionMatrix @ self.CProj
        else:
            AtomProjectionMatrixSpatial = np.einsum("pq,qj->pj", AtomProjectionMatrixSpatial,AllOrbs)
            #This is a basis change matrix and so the basis manager cant handle deblocking it. 
            AtomProjectionMatrix = ComplexMatrix(getSpinorVersionBAD(AtomProjectionMatrixSpatial.T),["AO",AtomProjectionName],SpinSymmetry.RHF)
        
        
        AtomProjectionMatrix.asProjectionMatrix()
        

        AtomLabels = [str(AN) for AN in self.AtomicNumbers]


        


        #Legacy, needs to be refactored
        HamParams = secQuantHam.Parameters()
        HamParams.HilbertSpaceSize = numActiveSpinOrbitals
        HamParams.t_0 = 1
        if self.params.maintainElectronNumber:
            if self.params.twoLeads:
                print("Maintaining electron number and there are two leads may lead to unexpected behaviour")
            muInitial = (self.Homo + self.Lumo)/2
            muInitial = np.real(muInitial)
            assert(muInitial <= self.params.muMax or print(f"{muInitial} <= {self.params.muMax}"))
            if self.Basis == "AO":
                lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE =  setupSimpleLeads(HamParams, CouplingMatrixLHS = LHSCoupleMatrix, CouplingMatrixRHS=RHSCoupleMatrix
                                                                                    ,VoltageDifference=self.params.startVoltage,SpinDifference=self.params.SpinDifference,muDifference = 0,BottomofBandLead1=-0.2,
                                                                                Gamma=self.params.startGamma,numberOfElectrons=self.NumElectrons,muMax=self.params.muMax,muInitial=muInitial,
                                                                                    RightGamma=self.params.RightGamma, RightSpinDifference = self.params.RightSpinDifference)
            elif self.Basis == "FC":
                lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE =  setupSimpleLeads(HamParams, CouplingMatrixLHS = LHSCoupleMatrix, CouplingMatrixRHS=RHSCoupleMatrix
                                                                                    ,VoltageDifference=self.params.startVoltage,SpinDifference=self.params.SpinDifference,muDifference = 0,BottomofBandLead1=-0.2,
                                                                                Gamma=self.params.startGamma,numberOfElectrons=self.NumElectrons,muMax=self.params.muMax,muInitial=muInitial,
                                                                                    RightGamma=self.params.RightGamma, RightSpinDifference = self.params.RightSpinDifference,
                                                                                    CProj=self.CProj, CInvProj=self.CInvProjDirecttoDirect )
            else:
                print("not implemented")
                quit()
        else:
            if self.Basis == "FC":
                lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE =  setupSimpleLeads(HamParams, CouplingMatrixLHS = LHSCoupleMatrix, CouplingMatrixRHS=RHSCoupleMatrix
                                                                                ,VoltageDifference=self.params.startVoltage,SpinDifference=self.params.SpinDifference,muDifference = 0,BottomofBandLead1=-0.2,
                                                                                Gamma=self.params.startGamma,muInitial=self.params.muLeft,RightGamma=self.params.RightGamma, RightSpinDifference = self.params.RightSpinDifference,
                                                                                CProj=self.CProj, CInvProj=self.CInvProjDirecttoDirect)
            else:
                lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE =  setupSimpleLeads(HamParams, CouplingMatrixLHS = LHSCoupleMatrix, CouplingMatrixRHS=RHSCoupleMatrix
                                                                                ,VoltageDifference=self.params.startVoltage,SpinDifference=self.params.SpinDifference,muDifference = 0,BottomofBandLead1=-0.2,
                                                                                Gamma=self.params.startGamma,muInitial=self.params.muLeft,RightGamma=self.params.RightGamma, RightSpinDifference = self.params.RightSpinDifference)

        if (self.params.doPhotoElectronLeads):
            photoElectronCoupleMatrix = np.zeros(self.NAO)
            ScatteringCoupleMatrix = np.zeros(self.NAO)
            for Atom,AOOnAtom in enumerate(self.AtomToAOMap):
                if self.AtomicNumbers[Atom] in self.params.photoElectronElementCouple:
                    photoElectronCoupleMatrix[AOOnAtom] = 1
                if self.AtomicNumbers[Atom] in self.params.ScatteringElementCouple:
                    ScatteringCoupleMatrix[AOOnAtom] = 1
            if self.Basis == "AO":
                    photoElectronCoupleMatrix = np.einsum("pi,pq,qj->ij", AllOrbs,np.diag(photoElectronCoupleMatrix),AllOrbs) #C_{pi} M_{pq} C_{qj}
                    photoElectronCoupleMatrix = photoElectronCoupleMatrix.copy()[numFrozenOrbitals:,numFrozenOrbitals:]

                    ScatteringCoupleMatrix = np.einsum("pi,pq,qj->ij", AllOrbs,np.diag(ScatteringCoupleMatrix),AllOrbs) #C_{pi} M_{pq} C_{qj}
                    ScatteringCoupleMatrix = ScatteringCoupleMatrix.copy()[numFrozenOrbitals:,numFrozenOrbitals:]
            elif self.Basis == "FC":
                photoElectronCoupleMatrix = np.diag(photoElectronCoupleMatrix)
                ScatteringCoupleMatrix = np.diag(ScatteringCoupleMatrix)
            if self.params.doOldLeads:
                photoElectronCoupleMatrix = ComplexDirectMatrix(photoElectronCoupleMatrix,self.Basis + BasisManager.Alpha_Block,SpinSymmetry.RHF)
                ScatteringCoupleMatrix = ComplexDirectMatrix(ScatteringCoupleMatrix,self.Basis + BasisManager.Alpha_Block,SpinSymmetry.RHF)
            else:
                photoElectronCoupleMatrix = ComplexDirectMatrix(photoElectronCoupleMatrix,self.Basis + BasisManager.Alpha_Block+BasisManager.LowdinOrthogonaliseBasis,SpinSymmetry.RHF)
                ScatteringCoupleMatrix = ComplexDirectMatrix(ScatteringCoupleMatrix,self.Basis + BasisManager.Alpha_Block+BasisManager.LowdinOrthogonaliseBasis,SpinSymmetry.RHF)

            PhotoElecParameterDict = {"numberOfSpinOrbitals":numActiveSpinOrbitals,
                                "photoElectronGamma":self.params.photoElectronGamma,
                                "photoElectronBandBottom":self.params.photoElectronBandBottom,
                                "photoElectronSpin":self.params.photoElectronSpin,
                                "photoElectronCoupleMatrix":photoElectronCoupleMatrix,
                                "photoElectronMu":self.params.photoElectronExcitationEnergy + self.params.muLeft,
                                "ScatteringGamma":self.params.ScatteringGamma,
                                "ScatteringBandBottom":self.params.ScatteringBandBottom,
                                "scatteringCoupleMatrix":ScatteringCoupleMatrix,
                                "ScatteringSpin":self.params.ScatteringSpin,
                                "ScatteringMu":self.params.ScatteringMu
                            }
            if self.Basis == "FC":
                PhotoElecParameterDict["CProj"] = self.CProj
                PhotoElecParameterDict["CInvProj"] = self.CInvProj
            print("PhotoElectron parameters")
            print(PhotoElecParameterDict)
            
            PEAlpha,PEBeta,ScatAlpha,ScatBeta = setupPhotoElectronLeads(**PhotoElecParameterDict)
            photoElectronLeads = [PEAlpha,PEBeta,ScatAlpha,ScatBeta]
            
        else:
            photoElectronLeads = []

        global isConverged, count
        count = 0
        
        
        #Interface self energy
        self.InterfaceSelfEnergies = []
        if not self.params.subsystems is None and len(self.params.subsystems) >= 2 and self.params.subsystemViaInterface:
            #May be useful in the future, currently unused and broken.
            
            #Dont ignore the nucleus anywhere#
            # OneEHamMol = self.oneEH.toBasisCast(ComplexSelfAdjointMatrix(),"AO_LB")
            # OneEHamCross = OneEHamMol

            #Dont ignore nucleus of system in cross term
            OneEHamMol = LeadGreensFunctionMethod.oneEH.toBasisCast(ComplexSelfAdjointMatrix(),"AO_LB")
            OneEHamCross = self.oneEH.toBasisCast(ComplexSelfAdjointMatrix(),"AO_LB")

            #Ignore nucleus of system everywhere
            # OneEHamMol = LeadGreensFunctionMethod.oneEH.toBasisCast(ComplexSelfAdjointMatrix(),"AO_LB")
            # OneEHamCross = OneEHamMol

            #Include nucleus of system everywhere
            # OneEHamMol = self.oneEH.toBasisCast(ComplexSelfAdjointMatrix(),"AO_LB")
            # OneEHamCross = OneEHamMol

            
            FHamMol = FockHamiltonian.FockHamiltonian(LeadGreensFunctionMethod.DSpinor.toBasis(self.DSpinor,"AO_LB"),ZeroHam,self.makeJK)
            FHamMol.setUseEDIIS(self.params.useEDIIS)
            FHamMol.setCDIISInitialAlpha(self.params.CDIISInitialAlpha)
            FHamMol.setBasis("AO_LB")

            MolGfuncCalc = GreensFunction.GreensFunction()
            MolGfuncCalc.setHamiltonian(MatrixSelfEnergy.SelfAdjointMatrixSelfEnergy(OneEHamMol.toBasisCast(ComplexSelfAdjointMatrix(),"Mol1")))
            MolGfuncCalc.setSelfEnergies([FHamMol])
            MolGfuncCalc.forceSelfAdjointSelfEnergy(self.params.forceSelfAdjointSelfEnergy)
            MolGfuncCalc.setRestrictedCalculation(self.params.doRestrictedCalc)
            MolGfuncCalc.mu = 0
            FHamMol.setGreensFunction(MolGfuncCalc)

            SCFMol = SCFSolver.SCFSolver()
            SCFMol.setGreensFunction(MolGfuncCalc)
            SCFMol.setSelfConsistentSelfEnergy([FHamMol])
            def convergenceFuncMol():
                global isConverged,count
                count += 1
                print(f"SCFITER:{count}")
                errorEstimate = FHamMol.getErrorEstimate()
                isConverged = errorEstimate < pow(10,-self.params.SCFConvergence)
                return isConverged
            SCFMol.run(convergenceFuncMol,2000,self.params.TStart,self.params.TEnd,self.params.TStep)
            ThermalEnergyParameters = QuantityCalc.ThermalEnergyParameters()
            ThermalEnergyParameters.numberOfIntegrationPoints = 500
            ThermalEnergyParameters.startE = -40
            ThermalEnergyParameters.endE = 1
            # Adds on the nuclear + any frozen part
            ThermalEnergyParameters.extraEnergy = 0

            ThermalEnergyCalc = QuantityCalc.ThermalEnergy(ThermalEnergyParameters)
            ThermalEnergyCalc.setGreensFunction(MolGfuncCalc)
            self.results.InterfacesEnergy = ThermalEnergyCalc.getEnergy()
            
            count = 0
            
            FHamCross = FockHamiltonian.FockHamiltonian(FHamMol.getGn().toBasis(self.DSpinor,"AO_LB"),OneEHamCross.toBasis(ZeroHam,"AO_LB"),self.makeJK)
            FHamCross.setBasis("AO_LB") # Cross couple term due to density here
            FHamCross2 = FockHamiltonian.FockHamiltonian(FHamMol.getGn().toBasis(self.DSpinor,"AO_LB"),ZeroHam,self.makeJK)
            FHamCross2.setBasis("AO_LB") # interact term due to density here

            FHamCross3 = FockHamiltonian.FockHamiltonian(FHamMol.getGn().toBasis(self.DSpinor,"AO_LB"),ZeroHam,lambda dm : self.makeJK(FHam.getGn())) # Cross couple term due to density THERE
            FHam.setBasis("AO_LB")
            FHamCross3.setBasis("AO_LB")

            SEParams = InterfaceSelfEnergy.Parameters()
            #TODO set parameters automagically. 
            SEParams.m_type = InterfaceLeadType.GreensFunction
            SEParams.mu = 0 
            SEParams.Gfunc = MolGfuncCalc
            SEParams.SECouple = [FHamCross,FHamCross3] ## If we multiply OneEHamMol01 by 2 we get the correct energy but the wrong spectrum.... 
            self.InterfaceSelfEnergies.append(InterfaceSelfEnergy.InterfaceSelfEnergy(SEParams))
            self.InterfaceSelfEnergies.append(FHamCross2)
                





        
        GfuncCalc = GreensFunction.GreensFunction()
        if not self.params.subsystems is None and len(self.params.subsystems) >= 2 and not self.params.subsystemViaInterface:
            GfuncCalc.setSubsystems(["Mol0","Mol1"],self.params.SubsystemOpt)
        elif self.params.subsystemViaInterface:
            oneElecHam.setBasis("Mol0")

        GfuncCalc.setRestrictedCalculation(self.params.doRestrictedCalc)
        if self.params.forceSelfAdjointSelfEnergy:
            GfuncCalc.forceSelfAdjointSelfEnergy(True)
        print(f"Restricted Calculation: {self.params.doRestrictedCalc}",flush=True)
        GfuncCalc.setHamiltonian(oneElecHam)
        
        if (self.params.twoLeads):
            if (self.params.doPhotoElectronSCF and self.params.doPhotoElectronLeads):
                # GfuncCalc.setSelfEnergies([FHam,lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE] + photoElectronLeads + self.InterfaceSelfEnergies)
                self.GFuncSelfEnergies = [FHam,lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE] + photoElectronLeads
            else:
                # GfuncCalc.setSelfEnergies([FHam,lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE] + self.InterfaceSelfEnergies)
                self.GFuncSelfEnergies = [FHam,lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE]

            if (self.params.maintainElectronNumber):
                lead1AlphaSE.setGreensFunction(GfuncCalc) # No Harm in setting it if it is not used. 
                lead1BetaSE.setGreensFunction(GfuncCalc)
                lead2AlphaSE.setGreensFunction(GfuncCalc)
                lead2BetaSE.setGreensFunction(GfuncCalc)
        else:
            if (self.params.doPhotoElectronSCF and self.params.doPhotoElectronLeads):
                # GfuncCalc.setSelfEnergies([FHam,lead1AlphaSE,lead1BetaSE]+photoElectronLeads + self.InterfaceSelfEnergies)
                self.GFuncSelfEnergies = [FHam,lead1AlphaSE,lead1BetaSE]+photoElectronLeads
            else:    
                # GfuncCalc.setSelfEnergies([FHam,lead1AlphaSE,lead1BetaSE] + self.InterfaceSelfEnergies)
                self.GFuncSelfEnergies = [FHam,lead1AlphaSE,lead1BetaSE]
            if (self.params.maintainElectronNumber):
                lead1AlphaSE.setGreensFunction(GfuncCalc)
                lead1BetaSE.setGreensFunction(GfuncCalc)
        
        GfuncCalc.setSelfEnergies(self.GFuncSelfEnergies + self.InterfaceSelfEnergies)

        if self.params.doSOC:
            GfuncCalc.setPerturbativeSelfEnergies([SOCHam])
            
            
        
        #GfuncCalc.setSelfEnergies([FHam])
        FHam.setGreensFunction(GfuncCalc)
        
        SCF = SCFSolver.SCFSolver()
        SCF.setGreensFunction(GfuncCalc)

        if self.params.subsystemViaInterface:
            FHamCross3.setGreensFunction(GfuncCalc)
            SCF.setSelfConsistentSelfEnergy([FHam,FHamCross3])
        else:
            SCF.setSelfConsistentSelfEnergy([FHam])

        ThermalEnergyParameters = QuantityCalc.ThermalEnergyParameters()
        ThermalEnergyParameters.numberOfIntegrationPoints = 500
        ThermalEnergyParameters.startE = -40
        ThermalEnergyParameters.endE = 1
        # Adds on the nuclear + any frozen part
        ThermalEnergyParameters.extraEnergy = self.ConstantEnergyOffset

        ThermalEnergyCalc = QuantityCalc.ThermalEnergy(ThermalEnergyParameters)
        ThermalEnergyCalc.setGreensFunction(GfuncCalc)

        spectralParameters = QuantityCalc.SpectralDensityCalcParameters()
        spectralParameters.numberOfIntegrationPoints = 1000000
        spectralParameters.startE = -4
        spectralParameters.endE = 4

        spectralDensity = QuantityCalc.SpectralDensityCalc(spectralParameters)
        spectralDensity.setGreensFunction(GfuncCalc)

        densityParameters = QuantityCalc.ElectronDensityParameters()
        densityParameters.numberOfIntegrationPoints = 500
        densityParameters.startE = -4
        densityParameters.endE = 4

        electronDensity = QuantityCalc.ElectronDensityCalc(densityParameters)
        electronDensity.setGreensFunction(GfuncCalc)

        DivcurrentParameters = QuantityCalc.DivCurrentParameters()
        DivcurrentParameters.numberOfIntegrationPoints = 5000
        DivcurrentParameters.startE = self.params.muLeft-1.5*self.params.EndVoltage/globals.electricPotentialUnit
        DivcurrentParameters.endE = self.params.muLeft+0.5*self.params.EndVoltage/globals.electricPotentialUnit

        divCurrentLead1Alpha = QuantityCalc.DivCurrentCalc(DivcurrentParameters)
        divCurrentLead1Alpha.setGreensFunction(GfuncCalc,[lead1AlphaSE])

        divCurrentLead1Beta = QuantityCalc.DivCurrentCalc(DivcurrentParameters)
        divCurrentLead1Beta.setGreensFunction(GfuncCalc,[lead1BetaSE])

        divCurrentLead2Alpha = QuantityCalc.DivCurrentCalc(DivcurrentParameters)
        divCurrentLead2Alpha.setGreensFunction(GfuncCalc,[lead2AlphaSE])

        divCurrentLead2Beta = QuantityCalc.DivCurrentCalc(DivcurrentParameters)
        divCurrentLead2Beta.setGreensFunction(GfuncCalc,[lead2BetaSE])

        if (self.params.doPhotoElectronLeads):
            DivcurrentParameters = QuantityCalc.DivCurrentParameters()
            DivcurrentParameters.numberOfIntegrationPoints = 50000
            DivcurrentParameters.startE = self.params.muLeft-0.5 if PhotoElecParameterDict["photoElectronBandBottom"] is None else PhotoElecParameterDict["photoElectronBandBottom"]-1
            DivcurrentParameters.endE = PhotoElecParameterDict["photoElectronMu"]+1

            divCurrentPEAlpha = QuantityCalc.DivCurrentCalc(DivcurrentParameters)
            divCurrentPEAlpha.setGreensFunction(GfuncCalc,[PEAlpha])

            divCurrentPEBeta = QuantityCalc.DivCurrentCalc(DivcurrentParameters)
            divCurrentPEBeta.setGreensFunction(GfuncCalc,[PEBeta])

            divCurrentScatAlpha= QuantityCalc.DivCurrentCalc(DivcurrentParameters)
            divCurrentScatAlpha.setGreensFunction(GfuncCalc,[ScatAlpha])

            divCurrentScatBeta = QuantityCalc.DivCurrentCalc(DivcurrentParameters)
            divCurrentScatBeta.setGreensFunction(GfuncCalc,[ScatBeta])

        # if (True or SOPeturbative):
        #     divCurrentSO = QuantityCalc.DivCurrentCalc(DivcurrentParameters)
        #     divCurrentSO.setGreensFunction(GfuncCalc,[PerturbSE])
        Steps = 5000

        Gr = GfuncCalc.getGrAtE(-0.48444)

        def doPlots():
            plotRange = QuantityCalc.autoRange(Steps,GfuncCalc)[Steps//10:Steps*9//10]
            plotRange[0] -= 2
            plotRange[-1] += 2
            t0 = time.time()
            if self.params.computeEnergy:
                TE = ThermalEnergyCalc.getEnergy()
                print(f"Thermal Energy {TE}") 
            else:   
                print(f"Thermal Energy Not Computing Energy") 
            t1 = time.time()
            total = t1-t0
            print(f"time taken:{total}")
            #Fast to slow
            if self.params.doPhotoElectronLeads and self.params.doCurrentPlots:
                photoElecBottomPlot = self.params.muLeft-0.5 if PhotoElecParameterDict["photoElectronBandBottom"] is None else PhotoElecParameterDict["photoElectronBandBottom"]-1
                plotdivCurrentE(self.params.outputName+ "PE", np.linspace(photoElecBottomPlot,PhotoElecParameterDict["photoElectronMu"]+1,1000),
                                divCurrentPEAlpha,divCurrentPEBeta,divCurrentScatAlpha,divCurrentScatBeta,True,None)
                plotdivCurrentE(self.params.outputName+ "SCATONLY", np.linspace(photoElecBottomPlot,PhotoElecParameterDict["photoElectronMu"]+1,1000),
                                divCurrentScatAlpha,divCurrentScatBeta,None,None,False,None)
                plotdivCurrentE(self.params.outputName+ "PE2", np.linspace(photoElecBottomPlot,PhotoElecParameterDict["photoElectronMu"]+1,1000),
                                divCurrentPEAlpha,divCurrentPEBeta,divCurrentLead1Alpha,divCurrentLead1Beta,True,None)
            if self.params.doCurrentPlots:
                plotdivCurrentE(self.params.outputName , np.linspace(self.params.muLeft-1.5*self.params.EndVoltage/globals.electricPotentialUnit,self.params.muLeft+0.5*self.params.EndVoltage/globals.electricPotentialUnit,1000),
                            divCurrentLead1Alpha,divCurrentLead1Beta,divCurrentLead2Alpha,divCurrentLead2Beta,self.params.twoLeads,None)
            if self.params.doSpectralPlots:
                plotEnergySpectralDensity(self.params.outputName, self.NAO, plotRange, spectralDensity, AtomProjectionName, GfuncCalc, AtomLabels)
                plotEnergyDensity(self.params.outputName, self.NAO,plotRange,electronDensity,AtomProjectionName,GfuncCalc,AtomLabels)
            if self.params.doGnIntegral:
                plotSpinDensity(self.params.outputName,electronDensity,AtomLabels,AtomProjectionName,GfuncCalc)
                plotDensity(self.params.outputName,electronDensity,AtomProjectionName,AtomLabels,GfuncCalc)
                if self.params.doNonEqGnIntegral:
                    NonEQGn = electronDensity.getGn(self.params.doSOC,self.params.NonEqGnStart,self.params.NonEqGnEnd)
                    NonEQGn = NonEQGn.toBasis(NonEQGn,self.Basis)
                    print(f"Saving NonEqGn in basis:{self.Basis} as ComplexSelfAdjointMatrix")
                    np.save(self.params.outputName + "finalNonEqGn.npy",NonEQGn.toNumpy())
            else:
                print("Not doing Gn Integral")
            
        
        def convergenceFunc():
            global isConverged,count
            count += 1
            print(f"SCFITER:{count}")
            errorEstimate = FHam.getErrorEstimate()
            isConverged = errorEstimate < pow(10,-self.params.SCFConvergence)
            return isConverged
        
        #SCF.setCallBackFunction(doPlots)
        #SCF.setCallBackFunction(lambda : print(f"Thermal Energy{ThermalEnergyCalc.getEnergy() if computeEnergy else "Not Computing Energy"}")  )
        def saveDensity():
            if not self.params.doSCF:
                return
            #Gn = electronDensity.getGn(False)
            Gn = FHam.getGn()
            Gn = Gn.toBasis(Gn,self.Basis)
            print(f"Saving Gn in basis:{self.Basis} as ComplexSelfAdjointMatrix")
            np.save(self.params.outputName + "finalGn.npy",Gn.toNumpy())
            # Gn2 = ComplexSelfAdjointMatrix(np.load(self.params.outputName + "finalGn.npy"), self.Basis, SpinSymmetry.NoSpinSym)
            # Gn2 = Gn2.toBasis(ComplexDualSelfAdjointMatrix(), self.Basis)
            # Gn2 = Gn2.toBasis(ComplexSelfAdjointMatrix(), self.Basis)
            # print(f"SaveLoadError:{np.linalg.norm((Gn - Gn2).toNumpy())}")

        SCF.setCallBackFunction(lambda : saveDensity())
        print(f"Thermal Energy{ThermalEnergyCalc.getEnergy() if self.params.computeEnergy else "Not Computing Energy"}")
        if self.params.startGamma != self.params.maxGamma:
            for t in np.logspace(np.log10(self.params.startGamma*HamParams.t_0),np.log10(self.params.maxGamma*HamParams.t_0),self.params.GammaSteps):
                print(f"Thermal Energy{ThermalEnergyCalc.getEnergy() if self.params.computeEnergy else "Not Computing Energy"}")
                lastDensityTrace = -1
                if self.params.doSCF:
                    FHam.resetDIISCache()
                    SCF.run(convergenceFunc,50,self.params.TStart,self.params.TEnd,self.params.TStep)
                print(f"Setting Lead coupling to {t}")

                lead1AlphaSE.setT(t)
                lead1BetaSE.setT(t)
                if self.params.RightGamma is None:
                    lead2AlphaSE.setT(t)
                    lead2BetaSE.setT(t)
        isConverged = False    

        if self.params.doSCF:
            FHam.resetDIISCache()
            SCF.run(convergenceFunc,2000,self.params.TStart,self.params.TEnd,self.params.TStep)
            if not isConverged and not self.params.AllowFailConverge:
                print("SCF not converged, try again")
                quit()
        print(f"Thermal Energy{ThermalEnergyCalc.getEnergy() if self.params.computeEnergy else "Not Computing Energy"}")
        
        
        if self.params.startVoltage != self.params.EndVoltage and self.params.maintainElectronNumber == False:
            for V in np.linspace(self.params.startVoltage,self.params.EndVoltage,self.params.VoltageSteps):
                lastDensityTrace = -1
                FHam.resetDIISCache()
                SCF.run(convergenceFunc,3,self.params.TStart,self.params.TEnd,self.params.TStep)
                print(f"Thermal Energy{ThermalEnergyCalc.getEnergy() if self.params.computeEnergy else "Not Computing Energy"}")
                print(f"Setting Voltage  to {V}")
                lead2AlphaSE.setChemicalPotential(self.params.muLeft-V*globals.e/globals.electricPotentialUnit)
                lead2BetaSE.setChemicalPotential(self.params.muLeft-V*globals.e/globals.electricPotentialUnit)
        elif self.params.startVoltage != self.params.EndVoltage:
            print("Cannot change voltage and also maintain electron number")
        isConverged = False
        lastDensityTrace = -1
        if self.params.doSCF:
            FHam.resetDIISCache()
            SCF.run(convergenceFunc,2000,self.params.TStart,self.params.TEnd,self.params.TStep)
            if not isConverged and not self.params.AllowFailConverge:
                print("SCF not converged, try again")
                quit()

        
        saveDensity()
        if self.params.saveFrozenCore and self.Basis == "AO" and not self.params.loadFrozenCore:
            LProj,LInvProj,CProj,CInvProj,LGn,CGn,LE,CE = GfuncCalc.getFrozenCoreTransformation(self.params.frozenCoreEnergy)
            print(f"Frozen Core Energy:{LE[0,0]}, Active Space Energy{CE[0,0]}")
            np.savez(self.params.outputName + "FrozenCore.npz",LProj=LProj,LInvProj=LInvProj,CProj=CProj,CInvProj=CInvProj,LGn=LGn,CGn=CGn,LE=LE,CE=CE)
        if self.params.doPlots:
            doPlots()

        if self.params.doAintegral:
            t0 = time.time()
            Density = spectralDensity.getAR()
            print(Density/(2*np.pi))
            t1 = time.time()

            total = t1-t0
            print(f"time taken:{total}")
        else:
            print("not Doing A integral")
        if self.params.computeEnergy:
            if self.params.subsystemViaInterface:
                ##Multiply this by 2 too
                # FHamCross = FockHamiltonian.FockHamiltonian(FHamMol.getGn().toBasis(self.DSpinor,"AO_LB")*2,OneEHamCross.toBasis(ZeroHam,"AO_LB"),self.makeJK)
                # FHamCross.setBasis("AO_LB") # Cross couple term due to density here
                # FHamCross2 = FockHamiltonian.FockHamiltonian(FHamMol.getGn().toBasis(self.DSpinor,"AO_LB")*2,ZeroHam,self.makeJK)
                # FHamCross2.setBasis("AO_LB") # interact term due to density here
                ##
                SEParams.SECouple = [FHamCross,MatrixSelfEnergy.SelfAdjointMatrixSelfEnergy(OneEHamCross)] 
                self.InterfaceSelfEnergies = []
                self.InterfaceSelfEnergies.append(InterfaceSelfEnergy.InterfaceSelfEnergy(SEParams))
                self.InterfaceSelfEnergies.append(FHamCross2)
                GfuncCalc.setSelfEnergies(self.GFuncSelfEnergies + self.InterfaceSelfEnergies)

            self.results.CalcEnergy = ThermalEnergyCalc.getEnergy()
        
        #Cleanup
        if self.params.subsystemViaInterface:
            MolGfuncCalc.setSelfEnergies([])
            self.InterfaceSelfEnergies = []
        GfuncCalc.setSelfEnergies([])
        BasisManager.deleteAllBasis()
        return self.results
        
        
    
       
        

if __name__ == "__main__":
    params = GreensFunctionMethodParameters()
    params.activeOrbitals = None
    params.frozenOrbitals = set()
    params.outputName = "Calculations/NHCLF/NHCLF"

    #Lead parameters
    params.startGamma = 0
    params.maxGamma = 0
    params.RightGamma = 0e-6 # None to set the same as left
    params.GammaSteps = 1
    params.muLeft = -0.382#-0.171847#5
    params.startVoltage = 0*globals.electricPotentialUnit #10
    params.EndVoltage = 0*globals.electricPotentialUnit #10
    params.VoltageSteps = 2

    # maintainElectronNumber = True if startVoltage == EndVoltage and startVoltage + muLeft == 0 else False
    params.muMax = 200 # Pointless?

    params.coupleBasedOnElement = True
    params.ElementCoupleLeft = [9,17,7,1] 
    params.ElementCoupleRight = [] 
    params.SpinDifference = np.array([0,0,0.00])*2 # This is divided by 2 for some reason
    params.RightSpinDifference = np.array([0,0,0.0])*2

    #PhotoElectronLeads
    params.photoElectronGamma = 1e-4
    params.photoElectronBandBottom = None # None means no bottom
    params.photoElectronSpin = [0,0,0.00]
    params.photoElectronElementCouple = [9]
    params.photoElectronExcitationEnergy = 125.9/globals.electricPotentialUnit # 5.9eV # gets added to muLeft

    params.ScatteringGamma = 1e-2
    params.ScatteringBandBottom = None # AKA vaccum level
    params.ScatteringElementCouple = [17]
    params.ScatteringSpin = [0,0,0]
    params.ScatteringMu = params.muLeft




    #less common user settings
    params.basis = "sto-3g"
    params.charge = 0
    params.spin = 0
    params.useHFOrbitals = False or len(params.frozenOrbitals) > 0
    params.useOrthogonalOrbitals = False # Takes effect only if youre not using HF orbitals
    #Auto determined
    # params.usingDualSpace = not (params.useHFOrbitals or params.useOrthogonalOrbitals) # Only use dual space if both are false.
    params.loadFromQChemData = True

    params.twoLeads = True
    params.doPhotoElectronLeads = False
    params.doPhotoElectronSCF = False

    params.doSOC = False
    # SOCScaleFactor = 1# Multiplies the SOC contribution by a scale factor. This is plain wrong but can give useful results. Now in PyscfInterface

    params.computeEnergy = False if params.doSOC else True
    params.SCFRead = True # Reads a finalDensity file if it can find one
    params.doSCF = True # Actually do SCF or just use the loaded density. Usually you want to use SCF
    params.doRestrictedCalc = True
    params.SCFConvergence = 10 # Same as QCHEM, Converged when trace changes by less than 10**-SCFConvergence
    params.doAintegral = False
    params.doGnIntegral = True

    params.saveFrozenCore = False
    params.loadFrozenCore = False
    params.frozenCoreEnergy = -1

    params.finalise()

    GF = GreensFunctionMethod(params)
    GF.runHamGen()




