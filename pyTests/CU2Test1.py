from cppNEGF import SecondQuantisedHamiltonian as secQuantHam
from cppNEGF import SimpleLeadSelfEnergy,QuantityCalc,SCFSolver,GreensFunction,PhononSelfEnergy, MatrixSelfEnergy, FockHamiltonian, SparseTensor_4_Complex
from cppNEGF import globals
from Plots import *
import Plots

Plots.Validate = True # Variable defined in plots
from loadQChemOutput import *

import numpy as np
from simpleLeads import *
from .Util import *

from SOCIntegrals import make_h1_soc, koseki_charge
from pyscf import gto, scf, ao2mo, lib, tools
import pyscf 

import numpy as np
import itertools as it

import random

from typing import List
import time
import os.path



np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)


#User Settings

activeOrbitals = None
frozenOrbitals = set()
outputName = "pyTests/Resources/CUTest1"
resourceName = "pyTests/Resources/CU"



spinLocked = True
EField = [0,0,0]

#Lead parameters
startGamma = 1e-6
maxGamma = 1e-6
RightGamma = None # None to set the same as left
GammaSteps = 1
muLeft = -0.117769#-0.171847#5
startVoltage = 0.0*globals.electricPotentialUnit #10
EndVoltage = 0.0*globals.electricPotentialUnit #10
VoltageSteps = 2

maintainElectronNumber = True if startVoltage == EndVoltage and startVoltage + muLeft == 0 else False
muMax = 200 # Pointless?

coupleBasedOnElement = True
ElementCoupleLeft = [29] 
ElementCoupleRight = [] 
SpinDifference = np.array([0,0,0.00])*2 # This is divided by 2 for some reason
RightSpinDifference = np.array([0,0,0.0])*2

#PhotoElectronLeads
photoElectronGamma = 1e-4
photoElectronBandBottom = None # None means no bottom
photoElectronSpin = [0,0,0.00]
photoElectronElementCouple = [9]
photoElectronExcitationEnergy = 1/globals.electricPotentialUnit # 5.9eV # gets added to muLeft

ScatteringGamma = 1e-2
ScatteringBandBottom = -0.3 # AKA vaccum level
ScatteringElementCouple = [17]
ScatteringSpin = [0,0,0]
ScatteringMu = muLeft




#less common user settings
basis = "sto-3g"
charge = 0
spin = 0
Headless = True
useHFOrbitals = False or len(frozenOrbitals) > 0
useOrthogonalOrbitals = False # Takes effect only if youre not using HF orbitals
usingDualSpace = not (useHFOrbitals or useOrthogonalOrbitals) # Only use dual space if both are false.
loadFromQChemData = True
if loadFromQChemData:
    frozenOrbitals = set()

twoLeads = True
doPhotoElectronLeads = False
doPhotoElectronSCF = False

doSOC = False
SOCScaleFactor = 1# Multiplies the SOC contribution by a scale factor. This is plain wrong but can give useful results

computeEnergy = False if doSOC else True
SCFRead = False # Reads a finalDensity file if it can find one
doSCF = True # Actually do SCF or just use the loaded density. Usually you want to use SCF
doRestrictedCalc = False
SCFConvergence = 9.6 # Same as QCHEM, Converged when trace changes by less than 10**-SCFConvergence
doAintegral = False
doGnIntegral = True if doSOC else True

saveFrozenCore = False
loadFrozenCore = False
frozenCoreEnergy = -1


print(f"Settings:\nBasis: {basis}\nCharge:{charge}\nSpin:{spin}\nUseHFOrbitals:{useHFOrbitals}\nOrthogonalOrbitals:{useOrthogonalOrbitals}\nUsingDualSpace:{usingDualSpace}\nQchem?:{loadFromQChemData}\n\n")
if SOCScaleFactor != 1:
    print("#####################################")
    print(f"Note SOCScaleFactor:{SOCScaleFactor}")
    print("#####################################")
    print("#####################################")
def getSpinorVersion(Mat):
    ret = np.zeros(shape=np.array(Mat.shape)*2)
    for i in range(Mat.shape[0]):
        for j in range(Mat.shape[1]):
            ret[2*i,2*j] = Mat[i,j]
            ret[2*i+1,2*j+1] = Mat[i,j]
    return ret

def convertToPyScfSpinorOrdering(Mat):
    #Converts from the spin index running fastest to slowest
    spatialSize = len(Mat)//2

    mat2 = Mat.reshape((spatialSize,2,spatialSize,2))
    return np.transpose(mat2,(1,0,3,2)).reshape((len(Mat),len(Mat)))

def convertFromPyscfSpinorOrdering(Mat):
    spatialSize = len(Mat)//2

    mat2 = Mat.reshape((2,spatialSize,2,spatialSize))
    return np.transpose(mat2,(1,0,3,2)).reshape((len(Mat),len(Mat)))

myMolObject = {}


def generateIntegrals(atomString,basis,isECP,Generate2eInts, BreitIntegrals = False, atomicCharges = None):
    mol = gto.Mole()
    mol.atom = atomString

    mol.basis = basis
    if isECP:
        mol.ecp = basis
    mol.charge = 0 # TODO get valid parameters automatically. Unsure when Pyscf checks these for validity
    mol.spin = 0
    mol.build()
    hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    if isECP:
        hecp = mol.intor_symmetric("ECPscalar")
    else:
        hecp = None
    S_AO = mol.intor_symmetric("int1e_ovlp")
    
    
    if BreitIntegrals:
        # class fakezfsObj:
        #     sso = False
        #     soo = False # Two electron interaction, We've kinda fudged this with an effective charge instead & ECPs TODO?
        #     so_eff_charge = True
        factor = SOCScaleFactor*(7.2973525664e-3)**2  /4 # 1/4c^2 if you use sigma, 1/2c^2 if you use shat. Note the factor for spin 1/2
        # h1aa, h1ab, h1ba, h1bb = make_h1_soc(fakezfsObj,mol,mo_coeff=None,mo_occ=None) # h1aa, h1ab, h1ba, h1bb
        # hSOCSpinor = np.zeros((mol.nao,mol.nao))
        # hSOCSpinor[::2,::2] = h1aa
        # hSOCSpinor[1::2,1::2] = h1bb
        # hSOCSpinor[::2,1::2] = h1ab
        # hSOCSpinor[1::2,::2] = h1ba
        hso1e = np.zeros((3,mol.nao,mol.nao),dtype=np.complex128)
        for ia in range(mol.natm):
            mol.set_rinv_origin(mol.atom_coord(ia))
            #Want to use True charge not ECP charge JPCA, 102, 10430, I think.
            Z = koseki_charge(atomicCharges[ia])
            hso1e += -Z * mol.intor('int1e_prinvxp', 3)*1j # See equation (24) in JCP, 122, 034107. The two p vs nabla give factor -1
        #hso1e is a 3,nao,nao array where the 3 corresponds to the pauli matrices X Y Z
        if (not np.any(hso1e)):
            print("SOC Integral isnt doing anything????????")
        hSOC = np.zeros((mol.nao*2,mol.nao*2),dtype=np.complex128)
        #sigmaZ
        hSOC[::2,::2] = hso1e[2,:,:]*factor
        hSOC[1::2,1::2] = -hso1e[2,:,:]*factor
        #sigmaX
        hSOC[1::2,::2] = hSOC[::2,1::2] = hso1e[0,:,:]*factor
        #sigmaY
        hSOC[::2,1::2] =  -1j*hso1e[1,:,:]*factor
        hSOC[1::2,0::2] =  1j*hso1e[1,:,:]*factor

        if isECP:
            #SOCECP
            # SO-ECP integrals in real spherical Gaussian basis computes
            #       <i| 1j * l U(r)|j>
            # have three components, corresponding to the lx, ly, lz operators
            #
            hso1e = mol.intor('ECPso')
            if (np.any(hso1e)):
                print("ECPSO is doing something!!")
            #
            # The SOC contribution to Hamiltonian is < s dot l U(r) > (s = 1/2 Pauli matrix)
            # Note the phase 1j was introduced to make ECPso integrals real. Removing it
            # with multipler -1j.
            # I Guess the 1/c^2 factor is already included here?
            s = .5 * lib.PauliMatrices
            hSOC += -1j * lib.einsum('sxy,spq->pxqy', s, hso1e).reshape(hSOC.shape)
    else:
        hSOC = None


    


    # #For RHF
    # def makejk(dm):
    #     J,K = pyscf.scf.jk.get_jk(mol,(dm,dm),scripts=('ijkl,ji->kl','ijkl,li->kj'),intor="int2e")
    #     return 2*J-K
    #For SPINDENSITY
    def makejk(dm):
        dm = convertToPyScfSpinorOrdering(dm)
        # J,K = pyscf.scf.ghf.get_jk(mol,SpinorD)
        #GHF Has a bug so copy the function here and fix it
        
        nso = dm.shape[-1]
        nao = nso // 2

        dmaa = dm[:nao,:nao]
        dmab = dm[:nao,nao:]
        dmbb = dm[nao:,nao:]
        dmba = dm[nao:,:nao]
        dms = np.stack((dmaa, dmbb, dmab, dmba))
        
        j1, k1 = pyscf.scf.hf.get_jk(mol, dms,hermi=0)

        vj = vk = None
        vj = np.zeros((nso,nso), dm.dtype)
        vj[:nao,:nao] = vj[nao:,nao:] = j1[0] + j1[1]

        vk = np.zeros((nso,nso), dm.dtype)
        vk[:nao,:nao] = k1[0]
        vk[nao:,nao:] = k1[1]
        vk[:nao,nao:] = k1[2]
        vk[nao:,:nao] = k1[3]

        return convertFromPyscfSpinorOrdering(vj-vk)
    
    if Generate2eInts:
        print("Generating 2e Integrals")
        twoElectronIntegrals = ao2mo.kernel(mol,np.identity(mol.nao),aosym=1) 
        print("Done Generating 2e Integrals")
        # eri4 = mol.intor('int2e', aosym='s1')
        return (hcore,twoElectronIntegrals,mol.energy_nuc(),S_AO,hecp, makejk,hSOC)
    else:        
        return (hcore,None,mol.energy_nuc(),S_AO,hecp, makejk,hSOC)


def pyscfGen(mol = None, initialDensityGuess = None):
    #Not implemented checks
    assert(doSOC == False)

    if mol is None:
        mol = gto.Mole()
        mol.atom = atomString

        mol.basis = basis
        mol.charge = charge
        mol.spin = spin
        mol.build()
    

    #mol.symmetry = 1 // do symmetry
    rhf_H2 = scf.RHF(mol)
    rhf_H2.verbose = 10000
    if initialDensityGuess is None:
        e_H2 = rhf_H2.kernel()
    else:
        e_H2 = rhf_H2.kernel(dm0=initialDensityGuess)
    
    if useHFOrbitals:
        AllOrbs = rhf_H2.mo_coeff

    #Rotate to make the game harder
    if not useHFOrbitals:
        if (useOrthogonalOrbitals):
            S_AO = mol.intor_symmetric("int1e_ovlp")
            O,V = np.linalg.eigh(S_AO)
            O = np.power(O,-0.5)
            AllOrbs = V @ np.diag(O) @ np.conj(V.T)
            print(np.einsum("pi,pq,qj->ij", AllOrbs,S_AO,AllOrbs)) #C_{pi} H_{pq} C_{qj})    
        else:
            #Use AOs and setup metric
            loweringMetricSpatial = mol.intor_symmetric("int1e_ovlp")
            AllOrbs = np.identity(len(loweringMetricSpatial)) # Identity matrix to allow rest to be happy
            

    
    print(AllOrbs)

    #MAKE MEP Cube file
    # dm = np.zeros(shape=(len(AllOrbs),len(AllOrbs)))
    # for i in range(mol.nelec[0]):
    #     dm[i,i] = 1
    # tools.cubegen.mep(mol,f"{outputName}MEP.cube",dm)


    h_core_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    if mol.has_ecp():
        h_core_ao +=  mol.intor_symmetric("ECPscalar")
    hcore_mo = np.einsum("pi,pq,qj->ij", AllOrbs,h_core_ao,AllOrbs) #C_{pi} H_{pq} C_{qj}

    #DIPOLE integrals
    # h_r_ao = mol.intor_symmetric('int1e_r')
    # h_r_mo = np.einsum("pi,xpq,qj->xij", AllOrbs,h_r_ao,AllOrbs)#*lib.param.BOHR #C_{pi} H_{pq} C_{qj}
    # h_epotential_mo = np.einsum("x,xij->ij",EField,h_r_mo)
    
    #ANG MOMENTUM?
    # h_AngMom_ao = np.zeros((mol.nao,mol.nao),dtype=np.complex128)
    # index = 0
    # for ib in range(mol.nbas):
    #     ia = mol.bas_atom(ib)
    #     l = mol.bas_angular(ib)
    #     if (l == 1):
    #         h_AngMom_ao[index,index+1] = 1j
    #         h_AngMom_ao[index+1,index] = -1j
    #     index += 2*l+1
    
    # #Grad VXP in AO basis
    # h_GradVxP_ao = np.zeros((3,mol.nao,mol.nao),dtype=np.complex128)
    # for i in range(mol.natm):
    #     with mol.with_rinv_origin(mol.atom_coord(i)):
    #         h_GradVxP_ao += mol.intor('int1e_ia01p')*1j*mol.atom_charge(i)
    #         # int1e_ia01p := (#C(0 1) \| nabla-rinv \| cross p\) is the same as <|  -i \nabla{1/r} \times -i \nabla  |> see https://github.com/sunqm/libcint/blob/master/scripts/parser.cl
    # #asser Hermitian
    # assert(np.all(np.isclose(h_GradVxP_ao[2], np.conj(h_GradVxP_ao[2].T))))
    # #Grad VXP in MO basis
    # h_GradVxP_mo = np.einsum("pi,xpq,qj->xij", AllOrbs,h_GradVxP_ao,AllOrbs)#*lib.param.BOHR #C_{pi} H_{pq} C_{qj}
    # h_GradVxP_Spinor = np.zeros((2*mol.nao,2*mol.nao),dtype=np.complex128)
    
    # #Maybe you can do this with tensor notation but its late and im confused 
    # # h_GradVxP_Spinor = h_GradVxP_mo_{xij}sigma_{x}
    # #X component
    # h_GradVxP_Spinor[mol.nao:,:mol.nao] = h_GradVxP_mo[0,:,:]
    # h_GradVxP_Spinor[:mol.nao,mol.nao:] = h_GradVxP_mo[0,:,:]
    # #Y Component
    # h_GradVxP_Spinor[:mol.nao,mol.nao:] = h_GradVxP_mo[1,:,:]*-1j
    # h_GradVxP_Spinor[mol.nao:,:mol.nao] = h_GradVxP_mo[1,:,:]*1j
    # #Z Component
    # h_GradVxP_Spinor[:mol.nao,:mol.nao] = h_GradVxP_mo[2,:,:]
    # h_GradVxP_Spinor[mol.nao:,mol.nao:] = h_GradVxP_mo[2,:,:]*-1
    
    

    # h_GradVxP_Spinor *= SOCMagnitude
    
    #FCI energy -74.54899494196664 C2

    # h_angMom_mo*=0
    # h_epotential_mo *=0

    
    #orbs0 = np.reshape(orbs[:,0],(2,1))
    #orbs1 = np.reshape(orbs[:,1],(2,1))
    #shape = (ij_pairs, kl_pairs). 
    #No deduplication so ni*nj = ij_pairs. gives the value of (ij|kl) for the given ij,kl. 
    # index0 = ni*i+j
    twoElectronIntegrals = ao2mo.kernel(mol,AllOrbs,aosym=1)

    numSpatialOrbitals = mol.nao
    spatialorbitals = np.array(list(range(numSpatialOrbitals)))

    global frozenOrbitals
    frozenOrbitals = frozenOrbitals.intersection(spatialorbitals)
    global activeOrbitals
    activeOrbitals = activeOrbitals.intersection(spatialorbitals) if not activeOrbitals is None and not len(activeOrbitals) == 0 else set(spatialorbitals)-frozenOrbitals
    numActiveOrbitals = len(activeOrbitals)
    numFrozenOrbitals = len(frozenOrbitals)
    if usingDualSpace:
        assert(numFrozenOrbitals == 0)


    aoLabels = mol.ao_labels()
    print(f"AOLabels:\n{list(enumerate(aoLabels))}")
    print(f"Frozen orbitals:")
    for f in frozenOrbitals:
        print(f"{aoLabels[f]}")
    print("Active orbitals:")
    for a in activeOrbitals:
        print(f"{aoLabels[a]}")

    activeOrbitalsList = np.array(sorted(activeOrbitals))
    frozenOrbitalsList = np.array(sorted(frozenOrbitals))

    numAlphaelectrons,numBetaElectrons = mol.nelec
    numAlphaelectrons -= len(frozenOrbitals)
    numBetaElectrons -= len(frozenOrbitals)

    alphaOccupations = np.array(list(set(activeOrbitalsList[:numAlphaelectrons]))) - numFrozenOrbitals
    alphaOccupations *= 2
    betaOccupations = np.array(list(set(activeOrbitalsList[:numBetaElectrons]))) - numFrozenOrbitals
    betaOccupations = 2*betaOccupations +1
    Occupations = list(betaOccupations) + list(alphaOccupations) # Occupations in the active space
    

    #construct TwoElectronTensor
    ijIndexes = {(i,j):idx for idx,(i,j) in enumerate(it.product(spatialorbitals,repeat=2))} #lookup table for ij -> 2 electron integral index
    #Chemists notation so ij have the same electron coordinate. I want it in physicists so <qp|V|rs> therefore q,s and p,r have the same electron coordinate

    #The fock Hamiltonian is  F_{ij} = h_{ij} + \sum_{p}^occ <ip|V|pj> - <ip|V|jp> with the sum over occupied orbitals 
    #In the frozen core approximation some of ps are deemed always occupied.     
    # So we can split apart the sum    
    #F_{ij} = h_{ij} + \sum_{p}^occActive <ip|V|pj> - <ip|V|jp> + \sum_{p}^occFrozen <ip|V|pj> - <ip|V|jp>
    # If you use HF orbitals then F is diagonal so the overlap over the fock matrix of a frozen and active orbital is zero. Therefore
    # F_{ia} = h_{ia} + \sum_{ps}^occ <pi|V|js> - <pi|V|sj> = 0
    # As long as you dont change the frozen orbitals this will always be true for any other orbital rotation of the active orbitals. 
    # Therefore we can neglect the off diagonal blocks which link the frozen and active blocks and get, for the fock matrix, valid for the active orbitals
    # #F_{ij} = h_{ij} + \sum_{p}^occActive <ip|V|pj> - <ip|V|jp> + \sum_{p}^occFrozen <ip|V|pj> - <ip|V|jp>
    #
    # we can write the total energy as E = \sum_{occ}1/2 h_{ii} + 1/2 F_{ii}
    # Therefore we get a constant additional contribution to the energy of the frozen orbitals equal to E_extra =  \sum_{Frozenocc} 1/2h_{ii} + 1/2 F_{ii}

    FrozenConstantPart = 0
    #Add on constant part of the energy, Frozen electronic Energy + Nuclear energy
    
    #Frozen Electronic energy is assumed to always be a singlet state
    #h_{ii}
    for i in frozenOrbitalsList:
        FrozenConstantPart += 2*hcore_mo[i,i] #always spin allowed
    
    #sum_i sum_j 2(ii|jj) - (ij|ij)
    # 4 choices for coulomb AABB AAAA BBBB BBAA, 2 choices for exchange AAAA BBBB since ABAB and BABA are not allowed
    for i in frozenOrbitalsList:
        for j in frozenOrbitalsList:
            FrozenConstantPart += 2*twoElectronIntegrals[ijIndexes[(i,i)],ijIndexes[(j,j)]] - twoElectronIntegrals[ijIndexes[(i,j)],ijIndexes[(i,j)]]
    nuclearEnergy = mol.energy_nuc()
    FrozenConstantPart += nuclearEnergy

    
    if usingDualSpace:
        MOS = rhf_H2.mo_coeff
        SpinMOs = getSpinorVersion(MOS)
        SpinMOs[:,numAlphaelectrons*2::2] *= 0
        SpinMOs[:,numBetaElectrons*2+1::2] *= 0
        initalDensity = SpinMOs @ SpinMOs.T
    else:
        initalDensity = np.zeros(shape=(numActiveOrbitals*2,numActiveOrbitals*2))
        for occ in Occupations:
            initalDensity[occ,occ] = 1

    elecEnergy = rhf_H2.energy_elec()
    #print(f"ElecEnergy: {elecEnergy[0]} Coulomb Energy:{elecEnergy[1]}, nucEnergy:{nuclearEnergy}, Fock Energy:{elecEnergy[0] + elecEnergy[1]}")
    print(f"PYSCF: ElecEnergy: {elecEnergy[0]}, Total Energy:{elecEnergy[0] + nuclearEnergy}")


    myMolObject["S"] = loweringMetricSpatial
    myMolObject["DSpinor"] = initalDensity
    myMolObject["eri"] = twoElectronIntegrals
    myMolObject["oneEH"] = hcore_mo
    myMolObject["Basis"] = "HF" if useHFOrbitals else "GUESS" if useOrthogonalOrbitals else "AO" if usingDualSpace else "Unknown"
    # TODO
    myMolObject["Molecule"] = outputName
    # myMolObject["AtomString"] = atomString
    myMolObject["Source"] = "PySCF"
    myMolObject["AlphaOccupations"] = alphaOccupations
    myMolObject["BetaOccupations"] = betaOccupations 
    myMolObject["NumAlphaElectrons"] = numAlphaelectrons
    myMolObject["NumBetaElectrons"] = numBetaElectrons
    myMolObject["NumElectrons"] = numAlphaelectrons + numBetaElectrons
    myMolObject["AOToMoBasisChangeMatrix"] = AllOrbs
    myMolObject["NumActiveOrbitals"] = numActiveOrbitals
    myMolObject["NumFrozenOrbitals"] = numFrozenOrbitals
    myMolObject["ActiveOrbitalsList"] = activeOrbitalsList
    myMolObject["FrozenOrbitalsList"] = frozenOrbitalsList
    myMolObject["NBasis"] = mol.nao
    myMolObject["NAO"] = mol.nao
    myMolObject["eri_ijIndexes"] = ijIndexes
    myMolObject["ConstantEnergyOffset"] = FrozenConstantPart
    myMolObject["AtomToAOMap"] = [range(*sliceIDXS) for sliceIDXS in mol.aoslice_by_atom()[:,2:]]
    myMolObject["AtomicNumbers"] = mol.elements
    myMolObject["Homo"] = 0
    myMolObject["Lumo"] = 0 # Todo


def loadFromQChem():
    
    S,F,D,eri,oneEInts,fchkParams,success = loadQChem(resourceName)

    if not success or doSOC or True:
        #Used to be an error case, but we want to go this way usually

        #Could not find the Eri file or something else went wrong with reading this in
        # print("Could not load Eri file satisfactorily, Reverting to using QCHEM D,H-1elec and transforming to pyscf basis via RevQCMagic")
        # global atomString,activeOrbitals,frozenOrbitals,charge,spin,basis
        # atomString = generatePyscfAtomString(fchkParams)
        # activeOrbitals = None
        # frozenOrbitals = set()
        # charge = fchkParams["Charge"]
        # spin = (fchkParams["Multiplicity"]-1)//2
        # basis = fchkParams["Basis"]
        # pyscfGen()
        # return

        #REVQCMAGIC
        # Density,mol = loadWithRevQCMagic(outputName)
        # pyscfGen(mol,Density)
        # return
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
        atomString = generatePyscfAtomString(fchkParams)        
        hcore,eri,nuclearEnergy,S_AO,hecp,makeJK,hSOC = generateIntegrals(atomString,fchkParams["Basis"],isECP,Generate2eInts=Generate2eInts,BreitIntegrals=doSOC,atomicCharges=fchkParams["AtomicNumbers"])
        if isECP:
            hcore += hecp
        #Generate transformation to make these compatible
    
        permutationMatrix,cbs = computeQChemToPyscfPermutation(resourceName,fchkParams) # P^{i}_{j} = c'^i c_j where c'^i is in the pyscf basis, c_j is in the qchem basis. A normal vector is c^j\ket{e_j} so this works as expected
        PInv = np.linalg.inv(permutationMatrix).T

        print(f"E before Transformation: {np.trace(F@D) + np.trace(oneEInts@D)+nuclearEnergy}")
        D = np.einsum("ip,jq,pq->ij",permutationMatrix,permutationMatrix,D) # P^{i}_{p} D^{p\Bar{q}} P^{j}_{q}
        F = np.einsum("ip,jq,pq->ij",PInv,PInv,F) #  (PInv)_{i}^{p} (PInv)_{j}^{q} F_{pq}
        oneEInts = np.einsum("ip,jq,pq->ij",PInv,PInv,oneEInts) #  (PInv)_{i}^{p} (PInv)_{j}^{q} F_{pq}            
        print(f"E After Transformation: {np.trace(F@D) + np.trace(oneEInts@D)+nuclearEnergy}")

        SpinorD = getSpinorVersion(D)
        spinorHcore = getSpinorVersion(hcore)
        FSpinor = makeJK(SpinorD) + spinorHcore
        print(f"E using new integrals: {0.5*(np.trace(spinorHcore@SpinorD) + np.trace(FSpinor @ SpinorD))+nuclearEnergy}")

        S = S_AO
        print("using NEW Integrals!!!!!")
        oneEInts = hcore

        #Setup HOMO Lumo
        FockEigenValues = np.linalg.eigvals(np.linalg.inv(getSpinorVersion(S_AO))@FSpinor)
        FockEigenValues = np.sort(FockEigenValues)
        myMolObject["Homo"] = FockEigenValues[fchkParams["NoElectrons"]-1]
        if (fchkParams["NoElectrons"] == len(FockEigenValues)):
            #There is no Lumo
            myMolObject["Lumo"] = myMolObject["Homo"] + 2
        else:
            myMolObject["Lumo"] = FockEigenValues[fchkParams["NoElectrons"]]
        print(f"Homo:{myMolObject["Homo"]}, Lumo: {myMolObject["Lumo"]}")

        #Need to setup Atom -> AO Map now while we have `logical` access to cbs
        AOBoundaries = cbs.function_indices_for_item_starts
        AOBoundaries.append(cbs.n_functions) # stop index
        myMolObject["AtomToAOMap"] = []
        for idx in range(len(AOBoundaries)-1):
            myMolObject["AtomToAOMap"].append(list(range(AOBoundaries[idx],AOBoundaries[idx+1])))
        myMolObject["AtomicNumbers"] = fchkParams["AtomicNumbers"]
        #TODO pretty names for atoms? 
        # Can be extracted from cbs.basis_items
        

        myMolObject["makeJK"] = makeJK
    
    
        
        
    if "SCFEnergy" in fchkParams:
        print(f"QCHEM: SCFEnergy: {fchkParams["SCFEnergy"]}")

    myMolObject["S"] = S
    myMolObject["F"] = F
    myMolObject["DSpinor"] = getSpinorVersion(D) # Assumes restricted calculation
    if not eri is None:
        myMolObject["eri"] = eri
    myMolObject["oneEH"] = oneEInts
    if doSOC:
        myMolObject["oneEHSOC"] = hSOC
    
    myMolObject["Source"] = "QCHEM"
    
    myMolObject["Molecule"] = fchkParams["Comment"]
    myMolObject["AlphaOccupations"] = list(range(fchkParams["NoAlphaElectrons"]))
    myMolObject["BetaOccupations"] =  list(range(fchkParams["NoBetaElectrons"]))
    myMolObject["NumAlphaElectrons"] = fchkParams["NoAlphaElectrons"]
    myMolObject["NumBetaElectrons"] = fchkParams["NoBetaElectrons"]
    myMolObject["NumElectrons"] = fchkParams["NoElectrons"]
    if useHFOrbitals or useOrthogonalOrbitals:
        if useHFOrbitals:
            AllOrbs = fchkParams["AlphaMoCoeff"]    
            myMolObject["AOToMoBasisChangeMatrix"] = AllOrbs
            myMolObject["Basis"] = "HF"
        
        if useOrthogonalOrbitals:
            S_AO = myMolObject["S"]
            O,V = np.linalg.eigh(S_AO)
            O = np.power(O,-0.5)
            AllOrbs = V @ np.diag(O) @ V.T
            myMolObject["AOToMoBasisChangeMatrix"] = AllOrbs
            myMolObject["Basis"] = "ORTH"
        #Transform all the matrices
        myMolObject["F"] = np.einsum("pi,pq,qj->ij", AllOrbs, myMolObject["F"], AllOrbs)

        D = np.einsum("pi,pq,qj->ij", AllOrbs, D, AllOrbs)
        myMolObject["DSpinor"] = getSpinorVersion(D) # Assumes restricted calculation

        myMolObject["eri"] = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri,AllOrbs, AllOrbs, AllOrbs, AllOrbs)
        myMolObject["oneEH"] = np.einsum("pi,pq,qj->ij", AllOrbs, myMolObject["oneEH"], AllOrbs)
    elif loadFrozenCore and os.path.isfile(outputName + "FrozenCore.npz"):
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
        loweringMetricSpatial = myMolObject["S"]
        loweringMetric = getSpinorVersion(loweringMetricSpatial)
        raisingMetricSpatial = np.linalg.inv(loweringMetricSpatial)
        raisingMetric = getSpinorVersion(raisingMetricSpatial)

        myMolObject["S"] = None
        myMolObject["SSpinor"] = np.conj(CProj.T) @ loweringMetric @ CProj # C^\bar{i}_\bar{l} S_{\bar{i}j} C^j_m, The metric for two direct basis kets in the projected basis
        myMolObject["SRaisingSpinor"] = np.linalg.inv(myMolObject["SSpinor"])

        myMolObject["Basis"] = "FC"
        myMolObject["CProj"] = CProj # (AO x Active dimension) transformation matrix. R^-1 H R is the transformation to get to the basis
        myMolObject["CInvProj"] = CInvProj
        myMolObject["CInvProjDirecttoDirect"] =  myMolObject["SSpinor"] @ CInvProj @ raisingMetric # Projects from the AO basis to the direct basis of the frozen core
        myMolObject["LProj"] = LProj
        myMolObject["LInvProj"] = LInvProj
        
        oneEInts = np.array(getSpinorVersion(oneEInts),dtype=np.complex128)
        
        # oneEInts += makeJK(LGn @ raisingMetric) # Frozen core JK, makeJK works in the AO basis
        oneEInts = myMolObject["CInvProjDirecttoDirect"] @ oneEInts @ CProj # (C^{-1})^i_l S^{lm} h_{mn} C^n_j
        myMolObject["oneEH"] = oneEInts # Need to keep everything in the Direct basis
        if doSOC:
            myMolObject["oneEHSOC"] = myMolObject["CInvProjDirecttoDirect"]@ hSOC @ CProj # keep in direct basis

        
        myMolObject["F"] = None
        
        myMolObject["JKFC"] = myMolObject["CInvProjDirecttoDirect"] @ makeJK(LGn @ raisingMetric)@ CProj# Frozen core JK, makeJK works in the AO basis, needs to be a self energy or we get thermal energy wrong
        if os.path.isfile(outputName + "finalGn.npy") and SCFRead:            
            myMolObject["DSpinor"] = np.load(outputName + "finalGn.npy")
            if myMolObject["DSpinor"].shape[0] != CProj.shape[1]:
                myMolObject["DSpinor"] = CInvProj @ CGn @ CProj @ myMolObject["SRaisingSpinor"] # end up in the double direct basis
            else:
                print("loading Density from File")
        else:
            myMolObject["DSpinor"] = CInvProj @ CGn @ CProj @ myMolObject["SRaisingSpinor"] # end up in the double direct basis
        #Replaces Fock Hamiltonian
        # myMolObject["JKC"] = myMolObject["CInvProjDirecttoDirect"] @ makeJK(CGn @ raisingMetric)@ CProj# Frozen core JK, makeJK works in the AO basis, needs to be a self energy or we get thermal energy wrong

        # dm will be provided as   D^{\mu\nu} so need C^i_\mu D^{\mu\nu} (S_C)_{\nu k} (C^{-1})^k_l (S_AO)^{lj} to get back to AO basis
        # Some of these will cancel
        myMolObject["makeJK"] = lambda dm : myMolObject["CInvProjDirecttoDirect"] @ makeJK(CProj @ dm @ myMolObject["CInvProjDirecttoDirect"]) @ CProj

        myMolObject["AlphaOccupations"] = list(range(fchkParams["NoAlphaElectrons"] - LProj.shape[0]//2)) # Is this even right?
        myMolObject["BetaOccupations"] =  list(range(fchkParams["NoBetaElectrons"] - LProj.shape[0]//2))
        myMolObject["NumAlphaElectrons"] = fchkParams["NoAlphaElectrons"] - LProj.shape[0]//2
        myMolObject["NumBetaElectrons"] = fchkParams["NoBetaElectrons"] - LProj.shape[0]//2
        myMolObject["NumElectrons"] = fchkParams["NoElectrons"] - LProj.shape[0] # This is right
        myMolObject["NBasis"] = CProj.shape[1]
    else:
        #Use AOS
        myMolObject["Basis"] = "AO"
        myMolObject["NBasis"] = fchkParams["NBasis"]*2
        myMolObject["AOToMoBasisChangeMatrix"] = np.identity(fchkParams["NBasis"])
        if os.path.isfile(outputName + "finalGn.npy") and SCFRead:
            print("loading Density from File")
            myMolObject["DSpinor"] = np.load(outputName + "finalGn.npy")


    # Assumes restricted calculation
    assert(np.max(np.abs(fchkParams["BetaMoCoeff"] - fchkParams["AlphaMoCoeff"])) == 0)

    assert(len(frozenOrbitals) == 0) #Not Yet implemented
    #FIXME rethink these and what size is what
    myMolObject["NumActiveOrbitals"] = myMolObject["NBasis"]
    myMolObject["NumFrozenOrbitals"] = 0
    myMolObject["NAO"] = fchkParams["NBasis"]
    myMolObject["ActiveOrbitalsList"] = np.array(list(range(myMolObject["NAO"])))
    myMolObject["FrozenOrbitalsList"] = np.array([])
    
    myMolObject["eri_ijIndexes"] = {(i,j):idx for idx,(i,j) in enumerate(it.product(myMolObject["ActiveOrbitalsList"],repeat=2))}

    #Compute nuclear repulsion energy
    NoAtoms = fchkParams["NoAtoms"]
    if not "nuclearEnergy" in locals():
        
        nuclearEnergy = 0
        
        for i in range(NoAtoms):
            for j in range(i+1,NoAtoms):
                Z1 = fchkParams["AtomicNumbers"][i]
                Z2 = fchkParams["AtomicNumbers"][j]
                r1 = fchkParams["CartCoords"][i]
                r2 = fchkParams["CartCoords"][j]
                r12 = np.linalg.norm(r1-r2)
                nuclearEnergy += Z1*Z2/(r12)
    myMolObject["ConstantEnergyOffset"] = nuclearEnergy # Since we cant do frozen core
    
    
    #AtomToAOMap
    if not "AtomToAOMap" in myMolObject:
        assert(myMolObject["Basis"] != "FC")
        myMolObject["AtomToAOMap"] = [[] for i in range (NoAtoms)]
        count = 0
        for atom,shellAngMom in zip(fchkParams["ShellToAtomMap"],fchkParams["ShellAngMom"]):
            myMolObject["AtomToAOMap"][atom -1] += range(count,abs(shellAngMom)*2+1+count)
            count += abs(shellAngMom)*2+1
        myMolObject["AtomicNumbers"] = fchkParams["AtomicNumbers"]
    
def runHamGen():
    #Sets up the myMolObject
    if loadFromQChemData:
        loadFromQChem()
    else:
        pyscfGen()
    
    

    if myMolObject["Basis"] == "AO":
        loweringMetricSpatial = myMolObject["S"]
        loweringMetric = getSpinorVersion(loweringMetricSpatial)
        raisingMetricSpatial = np.linalg.inv(loweringMetricSpatial)
        raisingMetric = getSpinorVersion(raisingMetricSpatial)
        globals.setMetric(loweringMetric,raisingMetric)
        
        AllOrbs = myMolObject["AOToMoBasisChangeMatrix"]
    elif myMolObject["Basis"] == "FC":
        loweringMetric = myMolObject["SSpinor"]
        globals.setMetric(loweringMetric)
        raisingMetric = myMolObject["SRaisingSpinor"]
    else:
        print(f"not handled basis{myMolObject["Basis"]}")
        quit()
        
    
    numFrozenOrbitals = myMolObject["NumFrozenOrbitals"]
    #numActiveOrbitals_ = myMolObject["NumActiveOrbitals"] # May or may not be spin oprbitals
    numActiveSpinOrbitals = myMolObject["NBasis"]
    frozenOrbitalsList = myMolObject["FrozenOrbitalsList"]
    activeOrbitalsList  = myMolObject["ActiveOrbitalsList"]

    ijIndexes = myMolObject["eri_ijIndexes"]
    
    hcore = myMolObject["oneEH"]


    #Two electron Integrals in the active space
    if ("eri" in myMolObject):
        #Create FHam from two electron tensor

        twoElectronIntegrals = myMolObject["eri"]
        twoElectronTensor = SparseTensor_4_Complex()
        #twoElectronTensor.setSize(numActiveOrbitals)
        npTwoElectronTensor = np.zeros(shape=(myMolObject["NAO"],myMolObject["NAO"],myMolObject["NAO"],myMolObject["NAO"]))
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

        ZeroHam = np.zeros(shape=(myMolObject["NBasis"],myMolObject["NBasis"]))
        print("Constructing Fock Hamiltonian")
        # FHam = FockHamiltonian.FockHamiltonian(myMolObject["DSpinor"],ZeroHam,twoElectronTensor,myMolObject["makeJK"]) #For debug, compares the two methods. These seem to give slightly different eigenvectors??
        FHam = FockHamiltonian.FockHamiltonian(myMolObject["DSpinor"],ZeroHam,twoElectronTensor) #FHam expects in AO basis if non-orthogonal so this is handeled correctly.
        print("Done constructing Fock Hamiltonian")

    else:
        assert("makeJK" in myMolObject)
        assert(usingDualSpace) # We need to be in the AO basis otherwise this wont work. In principle it could but there is no need

        #Create FHam using the makeJK object. 
        if myMolObject["Basis"] != "FC":
            ZeroHam = np.zeros(shape=(numActiveSpinOrbitals,numActiveSpinOrbitals))
        else:
            ZeroHam = myMolObject["JKFC"]
        print("Constructing Fock Hamiltonian")
        FHam = FockHamiltonian.FockHamiltonian(myMolObject["DSpinor"],ZeroHam,myMolObject["makeJK"]) #FHam expects in AO basis if non-orthogonal so this is handeled correctly.
       
        print("Done constructing Fock Hamiltonian")

    FHam.setIntegrationParameters(-8,1,500)
    
    


    ExtendedHCore = np.zeros(shape=(numActiveSpinOrbitals,numActiveSpinOrbitals))
    if myMolObject["Basis"] != "FC":
        for i in activeOrbitalsList:
            for j in activeOrbitalsList:
                ijSameSpinFrozenFockElement = 0
                #ijDiffSpinFrozenFockElement = 0
                for k in frozenOrbitalsList:
                    #TODO Frozen core in non-orthogonal basis
                    ijSameSpinFrozenFockElement += 2*twoElectronIntegrals[ijIndexes[(i,j)],ijIndexes[(k,k)]] # <ik|kj> So k=s can be A or B
                    ijSameSpinFrozenFockElement -= twoElectronIntegrals[ijIndexes[(i,k)],ijIndexes[(k,j)]] # <ik|jk> So i=k & k=j so 1 choice
                    #ijDiffSpinFrozenFockElement += twoElectronIntegrals[ijIndexes[(i,j)],ijIndexes[(k,s)]] # <ik|kj> Not allowed since Spin i = spin j but this is not true here
                    #ijDiffSpinFrozenFockElement -= twoElectronIntegrals[ijIndexes[(i,s)],ijIndexes[(k,j)]] # <ik|jk> i = k & k = j so not Allowed
                Shiftedi = i - numFrozenOrbitals
                Shiftedj = j - numFrozenOrbitals
                
                ExtendedHCore[2*Shiftedi,2*Shiftedj] = hcore[i,j] + ijSameSpinFrozenFockElement
                ExtendedHCore[2*Shiftedi+1,2*Shiftedj+1] = hcore[i,j] + ijSameSpinFrozenFockElement
    else:
        assert(ExtendedHCore.shape[0] == hcore.shape[0])
        ExtendedHCore = hcore
    oneElecHam = MatrixSelfEnergy.MatrixSelfEnergy(ExtendedHCore)

    if doSOC:
        SOCHam = MatrixSelfEnergy.MatrixSelfEnergy(myMolObject["oneEHSOC"])
    
    

    #Couple to LHS of molecule
    global twoLeads
    LHSCoupleMatrix = np.zeros(myMolObject["NAO"])
    RHSCoupleMatrix = np.zeros(myMolObject["NAO"])
    for Atom,AOOnAtom in enumerate(myMolObject["AtomToAOMap"]):
        if coupleBasedOnElement:
            if myMolObject["AtomicNumbers"][Atom] in ElementCoupleLeft:
                LHSCoupleMatrix[AOOnAtom] = 1
            if myMolObject["AtomicNumbers"][Atom] in ElementCoupleRight:
                RHSCoupleMatrix[AOOnAtom] = 1
        else:
            if Atom == 0:
                LHSCoupleMatrix[AOOnAtom] = 1
            elif Atom == 1:
                RHSCoupleMatrix[AOOnAtom] = 1
    if myMolObject["Basis"] == "AO":
        LHSCoupleMatrix = np.einsum("pi,pq,qj->ij", AllOrbs,np.diag(LHSCoupleMatrix),AllOrbs) #C_{pi} M_{pq} C_{qj}
        LHSCoupleMatrix = LHSCoupleMatrix.copy()[numFrozenOrbitals:,numFrozenOrbitals:]
        
        if (np.all(RHSCoupleMatrix == 0)):
            print("Setting to one Lead mode since RHSCouple = 0")
            twoLeads = False
        RHSCoupleMatrix = np.einsum("pi,pq,qj->ij", AllOrbs,np.diag(RHSCoupleMatrix),AllOrbs) #C_{pi} M_{pq} C_{qj}
        RHSCoupleMatrix = RHSCoupleMatrix.copy()[numFrozenOrbitals:,numFrozenOrbitals:]
    
    elif myMolObject["Basis"] == "FC":
        LHSCoupleMatrix = np.diag(LHSCoupleMatrix)
        if (np.all(RHSCoupleMatrix == 0)):
            print("Setting to one Lead mode since RHSCouple = 0")
            twoLeads = False
        RHSCoupleMatrix = np.diag(RHSCoupleMatrix)
        #Basis transformation done in Simple leads after construction in the AO basis

    #AtomProjectors 
    AtomProjectionMatrixSpatial = np.zeros((len(myMolObject["AtomToAOMap"]),myMolObject["NAO"])) # P^i_j
    for Atom,AOOnAtom in enumerate(myMolObject["AtomToAOMap"]):
        AtomProjectionMatrixSpatial[Atom,AOOnAtom] = 1
    if myMolObject["Basis"] == "AO":
        AtomProjectionMatrixSpatial = np.einsum("pq,qj->pj", AtomProjectionMatrixSpatial,AllOrbs)
        AtomProjectionMatrix = getSpinorVersion(AtomProjectionMatrixSpatial)
        AtomProjectionMatrixInv = AtomProjectionMatrix.T
    elif myMolObject["Basis"] == "FC":
        AtomProjectionMatrix = getSpinorVersion(AtomProjectionMatrixSpatial)
        AtomProjectionMatrixInv = myMolObject["CInvProj"] @ AtomProjectionMatrix.T
        AtomProjectionMatrix = AtomProjectionMatrix @ myMolObject["CProj"] 
        

    AtomLabels = [str(AN) for AN in myMolObject["AtomicNumbers"]]


    


    #Legacy, needs to be refactored
    HamParams = secQuantHam.Parameters()
    HamParams.HilbertSpaceSize = numActiveSpinOrbitals
    HamParams.t_0 = 1
    if maintainElectronNumber:
        if twoLeads:
            print("Maintaining electron number and there are two leads may lead to unexpected behaviour")
        muInitial = (myMolObject["Homo"] + myMolObject["Lumo"])/2
        muInitial = np.real(muInitial)
        assert(muInitial <= muMax or print(f"{muInitial} <= {muMax}"))
        if myMolObject["Basis"] == "AO":
            lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE =  setupSimpleLeads(HamParams, CouplingMatrixLHS = LHSCoupleMatrix, CouplingMatrixRHS=RHSCoupleMatrix
                                                                                ,VoltageDifference=startVoltage,SpinDifference=SpinDifference,muDifference = 0,BottomofBandLead1=-0.2,
                                                                            Gamma=startGamma,numberOfElectrons=myMolObject["NumElectrons"],muMax=muMax,muInitial=muInitial,
                                                                                RightGamma=RightGamma, RightSpinDifference = RightSpinDifference)
        elif myMolObject["Basis"] == "FC":
            lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE =  setupSimpleLeads(HamParams, CouplingMatrixLHS = LHSCoupleMatrix, CouplingMatrixRHS=RHSCoupleMatrix
                                                                                ,VoltageDifference=startVoltage,SpinDifference=SpinDifference,muDifference = 0,BottomofBandLead1=-0.2,
                                                                            Gamma=startGamma,numberOfElectrons=myMolObject["NumElectrons"],muMax=muMax,muInitial=muInitial,
                                                                                RightGamma=RightGamma, RightSpinDifference = RightSpinDifference,
                                                                                CProj=myMolObject["CProj"], CInvProj=myMolObject["CInvProjDirecttoDirect"] )
        else:
            print("not implemented")
            quit()
    else:
        if myMolObject["Basis"] == "AO":
            lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE =  setupSimpleLeads(HamParams, CouplingMatrixLHS = LHSCoupleMatrix, CouplingMatrixRHS=RHSCoupleMatrix
                                                                            ,VoltageDifference=startVoltage,SpinDifference=SpinDifference,muDifference = 0,BottomofBandLead1=-0.2,
                                                                            Gamma=startGamma,muInitial=muLeft,RightGamma=RightGamma, RightSpinDifference = RightSpinDifference)
        elif myMolObject["Basis"] == "FC":
            lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE =  setupSimpleLeads(HamParams, CouplingMatrixLHS = LHSCoupleMatrix, CouplingMatrixRHS=RHSCoupleMatrix
                                                                            ,VoltageDifference=startVoltage,SpinDifference=SpinDifference,muDifference = 0,BottomofBandLead1=-0.2,
                                                                            Gamma=startGamma,muInitial=muLeft,RightGamma=RightGamma, RightSpinDifference = RightSpinDifference,
                                                                            CProj=myMolObject["CProj"], CInvProj=myMolObject["CInvProjDirecttoDirect"])
        else:
            print("not implemented")
            quit()

    if (doPhotoElectronLeads):
        photoElectronCoupleMatrix = np.zeros(myMolObject["NAO"])
        ScatteringCoupleMatrix = np.zeros(myMolObject["NAO"])
        for Atom,AOOnAtom in enumerate(myMolObject["AtomToAOMap"]):
            if myMolObject["AtomicNumbers"][Atom] in photoElectronElementCouple:
                photoElectronCoupleMatrix[AOOnAtom] = 1
            if myMolObject["AtomicNumbers"][Atom] in ScatteringElementCouple:
                ScatteringCoupleMatrix[AOOnAtom] = 1
        if myMolObject["Basis"] == "AO":
            photoElectronCoupleMatrix = np.einsum("pi,pq,qj->ij", AllOrbs,np.diag(photoElectronCoupleMatrix),AllOrbs) #C_{pi} M_{pq} C_{qj}
            photoElectronCoupleMatrix = photoElectronCoupleMatrix.copy()[numFrozenOrbitals:,numFrozenOrbitals:]

            ScatteringCoupleMatrix = np.einsum("pi,pq,qj->ij", AllOrbs,np.diag(ScatteringCoupleMatrix),AllOrbs) #C_{pi} M_{pq} C_{qj}
            ScatteringCoupleMatrix = ScatteringCoupleMatrix.copy()[numFrozenOrbitals:,numFrozenOrbitals:]
        elif myMolObject["Basis"] == "Fc":
            photoElectronCoupleMatrix = np.diag(photoElectronCoupleMatrix)
            ScatteringCoupleMatrix = np.diag(ScatteringCoupleMatrix)
        
        PhotoElecParameterDict = {"numberOfSpinOrbitals":numActiveSpinOrbitals,
                            "photoElectronGamma":photoElectronGamma,
                            "photoElectronBandBottom":photoElectronBandBottom,
                            "photoElectronSpin":photoElectronSpin,
                            "photoElectronCoupleMatrix":photoElectronCoupleMatrix,
                            "photoElectronMu":photoElectronExcitationEnergy + muLeft,
                            "ScatteringGamma":ScatteringGamma,
                            "ScatteringBandBottom":ScatteringBandBottom,
                            "scatteringCoupleMatrix":ScatteringCoupleMatrix,
                            "ScatteringSpin":ScatteringSpin,
                            "ScatteringMu":ScatteringMu
                        }
        if myMolObject["Basis"] == "FC":
            PhotoElecParameterDict["CProj"] = myMolObject["CProj"]
            PhotoElecParameterDict["CInvProj"] = myMolObject["CInvProj"]
        print("PhotoElectron parameters")
        print(PhotoElecParameterDict)
        
        PEAlpha,PEBeta,ScatAlpha,ScatBeta = setupPhotoElectronLeads(**PhotoElecParameterDict)
        photoElectronLeads = [PEAlpha,PEBeta,ScatAlpha,ScatBeta]
        
    else:
        photoElectronLeads = []
    GfuncCalc = GreensFunction.GreensFunction()
    GfuncCalc.setRestrictedCalculation(doRestrictedCalc)
    print(f"Restricted Calculation: {doRestrictedCalc}",flush=True)
    GfuncCalc.setHamiltonian(oneElecHam)
    if (twoLeads):
        if (doPhotoElectronSCF and doPhotoElectronLeads):
            GfuncCalc.setSelfEnergies([FHam,lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE] + photoElectronLeads)
        else:
            GfuncCalc.setSelfEnergies([FHam,lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE])
        if (maintainElectronNumber):
            lead1AlphaSE.setGreensFunction(GfuncCalc) # No Harm in setting it if it is not used. 
            lead1BetaSE.setGreensFunction(GfuncCalc)
            lead2AlphaSE.setGreensFunction(GfuncCalc)
            lead2BetaSE.setGreensFunction(GfuncCalc)
    else:
        if (doPhotoElectronSCF and doPhotoElectronLeads):
            GfuncCalc.setSelfEnergies([FHam,lead1AlphaSE,lead1BetaSE]+photoElectronLeads)
        else:    
            GfuncCalc.setSelfEnergies([FHam,lead1AlphaSE,lead1BetaSE])
        if (maintainElectronNumber):
            lead1AlphaSE.setGreensFunction(GfuncCalc)
            lead1BetaSE.setGreensFunction(GfuncCalc)
    if doSOC:
        GfuncCalc.setPerturbativeSelfEnergies([SOCHam])

    
    #GfuncCalc.setSelfEnergies([FHam])
    FHam.setGreensFunction(GfuncCalc)
    
    SCF = SCFSolver.SCFSolver()
    SCF.setGreensFunction(GfuncCalc)

    SCF.setSelfConsistentSelfEnergy(FHam)

    ThermalEnergyParameters = QuantityCalc.ThermalEnergyParameters()
    ThermalEnergyParameters.numberOfIntegrationPoints = 500
    ThermalEnergyParameters.startE = -40
    ThermalEnergyParameters.endE = 1
    # Adds on the nuclear + any frozen part
    ThermalEnergyParameters.extraEnergy = myMolObject["ConstantEnergyOffset"] 

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
    DivcurrentParameters.startE = muLeft-1.5*EndVoltage/globals.electricPotentialUnit
    DivcurrentParameters.endE = muLeft+0.5*EndVoltage/globals.electricPotentialUnit

    divCurrentLead1Alpha = QuantityCalc.DivCurrentCalc(DivcurrentParameters)
    divCurrentLead1Alpha.setGreensFunction(GfuncCalc,[lead1AlphaSE])

    divCurrentLead1Beta = QuantityCalc.DivCurrentCalc(DivcurrentParameters)
    divCurrentLead1Beta.setGreensFunction(GfuncCalc,[lead1BetaSE])

    divCurrentLead2Alpha = QuantityCalc.DivCurrentCalc(DivcurrentParameters)
    divCurrentLead2Alpha.setGreensFunction(GfuncCalc,[lead2AlphaSE])

    divCurrentLead2Beta = QuantityCalc.DivCurrentCalc(DivcurrentParameters)
    divCurrentLead2Beta.setGreensFunction(GfuncCalc,[lead2BetaSE])

    if (doPhotoElectronLeads):
        DivcurrentParameters = QuantityCalc.DivCurrentParameters()
        DivcurrentParameters.numberOfIntegrationPoints = 5000
        DivcurrentParameters.startE = muLeft-0.5 if PhotoElecParameterDict["photoElectronBandBottom"] is None else PhotoElecParameterDict["photoElectronBandBottom"]-1
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

    
    GfuncCalc.getGrAtE(-1)
    
    
    
   # print(f"Spectral Density{Density/(2*np.pi)}")

    def doPlots():
        plotRange = QuantityCalc.autoRange(Steps,GfuncCalc)[Steps//10:Steps*9//10]
        plotRange[0] -= 2
        plotRange[-1] += 2
        t0 = time.time()
        print(f"Thermal Energy{ThermalEnergyCalc.getEnergy() if computeEnergy else "Not Computing Energy"}") 
        t1 = time.time()
        total = t1-t0
        print(f"time taken:{total}")
        #Fast to slow
        if (doPhotoElectronLeads):
            photoElecBottomPlot = muLeft-0.5 if PhotoElecParameterDict["photoElectronBandBottom"] is None else PhotoElecParameterDict["photoElectronBandBottom"]-1
            plotdivCurrentE(outputName+ "PE", np.linspace(photoElecBottomPlot,PhotoElecParameterDict["photoElectronMu"]+1,1000),
                            divCurrentPEAlpha,divCurrentPEBeta,divCurrentScatAlpha,divCurrentScatBeta,True,None)
            plotdivCurrentE(outputName+ "SCATONLY", np.linspace(photoElecBottomPlot,PhotoElecParameterDict["photoElectronMu"]+1,1000),
                            divCurrentScatAlpha,divCurrentScatBeta,None,None,False,None)
            plotdivCurrentE(outputName+ "PE2", np.linspace(photoElecBottomPlot,PhotoElecParameterDict["photoElectronMu"]+1,1000),
                            divCurrentPEAlpha,divCurrentPEBeta,divCurrentLead1Alpha,divCurrentLead1Beta,True,None)
        plotdivCurrentE(outputName , np.linspace(muLeft-1.5*EndVoltage/globals.electricPotentialUnit,muLeft+0.5*EndVoltage/globals.electricPotentialUnit,1000),
                        divCurrentLead1Alpha,divCurrentLead1Beta,divCurrentLead2Alpha,divCurrentLead2Beta,twoLeads,None)
        
        plotEnergySpectralDensity(outputName, myMolObject["NAO"],plotRange,spectralDensity,AtomProjectionMatrix,AtomProjectionMatrixInv,GfuncCalc,AtomLabels)
        plotEnergyDensity(outputName, myMolObject["NAO"],plotRange,electronDensity,AtomProjectionMatrix,AtomProjectionMatrixInv,GfuncCalc,AtomLabels)
        if (doGnIntegral):
            plotSpinDensity(outputName,electronDensity,AtomLabels,AtomProjectionMatrixSpatial,AtomProjectionMatrixSpatial.T)
            plotDensity(outputName,myMolObject["NAO"],electronDensity,AtomProjectionMatrix,AtomProjectionMatrixInv,AtomLabels)
        else:
            print("Not doing Gn Integral")
        
    
    global isConverged, count
    count = 0
    def convergenceFunc():
        global isConverged,count
        count += 1
        print(f"SCFITER:{count}")
        errorEstimate = FHam.getErrorEstimate()
        print(f"Error Estimate:{errorEstimate}")
        isConverged = errorEstimate < pow(10,-SCFConvergence)
        return isConverged and count > 1

    #SCF.setCallBackFunction(doPlots)
    #SCF.setCallBackFunction(lambda : print(f"Thermal Energy{ThermalEnergyCalc.getEnergy() if computeEnergy else "Not Computing Energy"}")  )
    def saveDensity():
        if not doSCF:
            return
        #Gn = electronDensity.getGn(False)
        Gn = FHam.getGn()
        if (usingDualSpace):
            Gn = Gn @ raisingMetric
        np.save(outputName + "finalGn.npy",Gn)

    SCF.setCallBackFunction(lambda : saveDensity())
    print(f"Thermal Energy{ThermalEnergyCalc.getEnergy() if computeEnergy else "Not Computing Energy"}")
    if startGamma != maxGamma:
        for t in np.logspace(np.log10(startGamma*HamParams.t_0),np.log10(maxGamma*HamParams.t_0),GammaSteps):
            print(f"Thermal Energy{ThermalEnergyCalc.getEnergy() if computeEnergy else "Not Computing Energy"}")
            if doSCF:
                SCF.run(convergenceFunc,50)
                count = 0
            print(f"Setting Lead coupling to {t}")

            lead1AlphaSE.setT(t)
            lead1BetaSE.setT(t)
            if RightGamma is None:
                lead2AlphaSE.setT(t)
                lead2BetaSE.setT(t)
    isConverged = False    
    if doSCF:
        SCF.run(convergenceFunc,2000)
        count = 0
        if not isConverged:
            print("SCF not converged, try again")
            quit()
    print(f"Thermal Energy{ThermalEnergyCalc.getEnergy() if computeEnergy else "Not Computing Energy"}")
    
    
    if startVoltage != EndVoltage and maintainElectronNumber == False:
        for V in np.linspace(startVoltage,EndVoltage,VoltageSteps):
            SCF.run(convergenceFunc,3)
            count = 0
            print(f"Thermal Energy{ThermalEnergyCalc.getEnergy() if computeEnergy else "Not Computing Energy"}")
            print(f"Setting Voltage  to {V}")
            lead2AlphaSE.setChemicalPotential(muLeft-V*globals.e/globals.electricPotentialUnit)
            lead2BetaSE.setChemicalPotential(muLeft-V*globals.e/globals.electricPotentialUnit)
    elif startVoltage != EndVoltage:
        print("Cannot change voltage and also maintain electron number")
    isConverged = False
    if doSCF:
        SCF.run(convergenceFunc,2000)
        count = 0
        if not isConverged:
            print("SCF not converged, try again")
            quit()

    
    saveDensity()
    if saveFrozenCore and myMolObject["Basis"] == "AO" and not loadFrozenCore:
        LProj,LInvProj,CProj,CInvProj,LGn,CGn,LE,CE = GfuncCalc.getFrozenCoreTransformation(frozenCoreEnergy)
        print(f"Frozen Core Energy:{LE[0,0]}, Active Space Energy{CE[0,0]}")
        np.savez(outputName + "FrozenCore.npz",LProj=LProj,LInvProj=LInvProj,CProj=CProj,CInvProj=CInvProj,LGn=LGn,CGn=CGn,LE=LE,CE=CE)
    doPlots()

    if (doAintegral):
        t0 = time.time()
        Density = spectralDensity.getAR()
        print(Density/(2*np.pi))
        t1 = time.time()

        total = t1-t0
        print(f"time taken:{total}")
    else:
        print("not Doing A integral")
    
       
        

# if __name__ == "__main__":
#     runHamGen()



