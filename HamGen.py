from pyscf import gto, scf, ao2mo, lib

import numpy as np
import itertools as it
from typing import List

import matplotlib.pyplot as plt
import random
import os.path
import pickle

from scipy.integrate import solve_ivp

np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)



#User Settings
bondlen = 1.5


zposA = np.linspace(0.1,bondlen-0.1,8)
posA = [[0.2,0.1+z%0.1,z] for z in zposA]
posA = [[random.randint(0,50)/100,random.randint(0,50)/100,random.randint(0,50)/100] for i in range(len(posA))]

zposB = np.linspace(-1,bondlen+1,6)
posB = [[0.1,0.1,z] for z in zposB]
slaterDeterminantPositions = [posA,posB]

# atomString = "N 0 0 0 ; N 0 0 1.09"
# activeOrbitals = None#N2
# frozenOrbitals = {0,1,2,3}#N2

# outputName = "H2"
# atomString = f"H 0 0 0 ; H 0 0 {bondlen}"
# frozenOrbitals = set()
# activeOrbitals = None

# outputName = f"C2_singlet"
# atomString = f"C 0 0 -{bondlen/2} ; C 0 0 {bondlen/2}"
# activeOrbitals = None
# frozenOrbitals = set({0,1,2,3})#LiH

# outputName = f"C"
# atomString = f"C 0 0 0 ;"
# activeOrbitals = None
# frozenOrbitals = set()#LiH

# outputName = f"NHCLF_eq_singlet_planar"
# atomString = ("N         0.0000000000   -0.0901980079    0.3876186255;"
#              "H         0.0000000000   -0.1644803083    1.4090660101;"
#              "Cl        0.0000000000    1.4587250969   -0.3925061006;"
#              "F         0.0000000000   -1.2040467807   -0.4041785350;")
# activeOrbitals = None
# frozenOrbitals = set(range(0,13))#NHCLF



outputName = f"NHCLF_eq_singlet_tetrahedral"
atomString = ("N        -0.1252750742   -0.1890201929    0.0473047357;"
             "H        -0.3990709952   -0.2054706954    1.0717545993;"
             "Cl        0.5894735807    1.4636694458   -0.0960256887;"
             "F         0.9348724887   -1.0691785574   -0.0230336463;")
activeOrbitals = None
frozenOrbitals = set(range(0,13))#NHCLF




# atomString = f"He 0 0 0 ; He 0 0 {bondlen}"
# activeOrbitals = {1,2,3,4}#He2
# frozenOrbitals = {0}#He2

# atomString = f"N 0 0 0 ; N 0 0 {bondlen}"
# activeOrbitals = set()
# frozenOrbitals = set()

# atomString = f"Li 0 0 0 ; Li 0 0 {bondlen}"
# activeOrbitals = set()
# frozenOrbitals = {0,1}


spinLocked = True
dryRun = False
evaluateHam = True
shift = 0
EField = [0,0,1]
SOCMagnitude = 1/(2*137**2) # https://doi.org/10.1063/1.1829047 for constant
TimeEvolveLength = 50
TimeStepCount = 500
decayRate = 0



#less common user settings
basis = "sto-3g"
charge = 0
spin = 0

evaluateSlaterDeterminant = False

def runHamGen(outputName = outputName, atomString = atomString, frozenOrbitals = frozenOrbitals,
              activeOrbitals = activeOrbitals, spinLocked = spinLocked, dryRun = dryRun,
              evaluateHam = evaluateHam, shift = shift, basis = basis, 
              charge = charge, spin = spin, evaluateSlaterDeterminant = evaluateSlaterDeterminant):
    mol = gto.Mole()
    mol.atom = atomString

    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    #mol.symmetry = 1 // do symmetry
    rhf_H2 = scf.RHF(mol)
    e_H2 = rhf_H2.kernel()


    h_core_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    hcore_mo = np.einsum("pi,pq,qj->ij", rhf_H2.mo_coeff,h_core_ao,rhf_H2.mo_coeff) #C_{pi} H_{pq} C_{qj}

    h_r_ao = mol.intor_symmetric('int1e_r')
    h_r_mo = np.einsum("pi,xpq,qj->xij", rhf_H2.mo_coeff,h_r_ao,rhf_H2.mo_coeff)#*lib.param.BOHR #C_{pi} H_{pq} C_{qj}
    h_epotential_mo = np.einsum("x,xij->ij",EField,h_r_mo)
    
    # h_AngMom_ao = np.zeros((mol.nao,mol.nao),dtype=np.complex128)
    # index = 0
    # for ib in range(mol.nbas):
    #     ia = mol.bas_atom(ib)
    #     l = mol.bas_angular(ib)
    #     if (l == 1):
    #         h_AngMom_ao[index,index+1] = 1j
    #         h_AngMom_ao[index+1,index] = -1j
    #     index += 2*l+1
    h_GradVxP_ao = np.zeros((3,mol.nao,mol.nao),dtype=np.complex128)
    for i in range(mol.natm):
        with mol.with_rinv_origin(mol.atom_coord(i)):
            h_GradVxP_ao += mol.intor('int1e_ia01p')*1j*mol.atom_charge(i)
            # int1e_ia01p := (#C(0 1) \| nabla-rinv \| cross p\) is the same as <|  -i \nabla{1/r} \times -i \nabla  |> see https://github.com/sunqm/libcint/blob/master/scripts/parser.cl
        
        
    assert(np.all(np.isclose(h_GradVxP_ao[2], np.conj(h_GradVxP_ao[2].T))))
    
    
    

    h_GradVxP_mo = np.einsum("pi,xpq,qj->xij", rhf_H2.mo_coeff,h_GradVxP_ao,rhf_H2.mo_coeff)#*lib.param.BOHR #C_{pi} H_{pq} C_{qj}
    h_GradVxP_Spinor = np.zeros((2*mol.nao,2*mol.nao),dtype=np.complex128)
    
    #Maybe you can do this with tensor notation but its late and im confused 
    # h_GradVxP_Spinor = h_GradVxP_mo_{xij}sigma_{x}
    #X component
    h_GradVxP_Spinor[mol.nao:,:mol.nao] = h_GradVxP_mo[0,:,:]
    h_GradVxP_Spinor[:mol.nao,mol.nao:] = h_GradVxP_mo[0,:,:]
    #Y Component
    h_GradVxP_Spinor[:mol.nao,mol.nao:] = h_GradVxP_mo[1,:,:]*-1j
    h_GradVxP_Spinor[mol.nao:,:mol.nao] = h_GradVxP_mo[1,:,:]*1j
    #Z Component
    h_GradVxP_Spinor[:mol.nao,:mol.nao] = h_GradVxP_mo[2,:,:]
    h_GradVxP_Spinor[mol.nao:,mol.nao:] = h_GradVxP_mo[2,:,:]*-1
    
    

    h_GradVxP_Spinor *= SOCMagnitude
    
    #FCI energy -74.54899494196664 C2

    # h_angMom_mo*=0
    # h_epotential_mo *=0


    #eri_4fold_ao = mol.intor('int2e_sph', aosym=4)
    eri_4fold_mo = mol.ao2mo(rhf_H2.mo_coeff)

    #print(ao2mo.kernel(mol,rhf_H2.mo_coeff,aosym=1))

    orbs = rhf_H2.mo_coeff
    #orbs0 = np.reshape(orbs[:,0],(2,1))
    #orbs1 = np.reshape(orbs[:,1],(2,1))
    #shape = (ij_pairs, kl_pairs). 
    #No deduplication so ni*nj = ij_pairs. gives the value of (ij|kl) for the given ij,kl. 
    # index0 = ni*i+j
    twoElectronIntegrals = ao2mo.kernel(mol,orbs,aosym=1)

    numSpatialOrbitals = mol.nao
    spatialorbitals = np.array(list(range(numSpatialOrbitals)))

    frozenOrbitals = frozenOrbitals.intersection(spatialorbitals)
    activeOrbitals = activeOrbitals.intersection(spatialorbitals) if not activeOrbitals is None and not len(activeOrbitals) == 0 else set(spatialorbitals)-frozenOrbitals


    aoLabels = mol.ao_labels()
    print(f"AOLabels:\n{list(enumerate(aoLabels))}")
    print(f"Frozen orbitals:")
    for f in frozenOrbitals:
        print(f"{aoLabels[f]}")
    print("Active orbitals:")
    for a in activeOrbitals:
        print(f"{aoLabels[a]}")

    # from pyscf.tools import cubegen
    # for i in range(orbs.shape[1]):
    #     cubegen.orbital(mol, f'LiH{i+1}.cube', orbs[:,i])

    if not dryRun:
        HamFileName = f"{outputName}_{basis}_{bondlen}_E{EField}_SOC{SOCMagnitude}.pkl"
        
        if not os.path.isfile(HamFileName):

            activeOrbitalsList = np.array(sorted(activeOrbitals))
            frozenOrbitalsList = np.array(sorted(frozenOrbitals))

            numAlphaelectrons,numBetaElectrons = mol.nelec
            numAlphaelectrons -= len(frozenOrbitals)
            numBetaElectrons -= len(frozenOrbitals)

            alphaOccupations = [set(x).union(frozenOrbitalsList) for x in it.combinations(activeOrbitalsList,numAlphaelectrons)]
            betaOccupations = [set(x).union(numSpatialOrbitals+frozenOrbitalsList) for x in it.combinations(numSpatialOrbitals+activeOrbitalsList,numBetaElectrons)]

            if spinLocked:
                hfStates = [a.union(b) for a,b in it.product(alphaOccupations,betaOccupations)]
            else:
                TotalActiveOrbitals = np.array([[a,b] for a,b in zip(activeOrbitalsList,numSpatialOrbitals+activeOrbitalsList)])
                TotalActiveOrbitals = TotalActiveOrbitals.flatten()
                hfStates = [set(x).union(frozenOrbitalsList).union(numSpatialOrbitals+frozenOrbitalsList) for x in 
                            it.combinations(TotalActiveOrbitals,numBetaElectrons+numAlphaelectrons)]

            #hfStates[80],hfStates[1] = hfStates[1],hfStates[80]
            #hfStates[0], hfStates[1] = hfStates[1], hfStates[0]

            ijIndexes = {(i,j):idx for idx,(i,j) in enumerate(it.product(spatialorbitals,repeat=2))} #lookup table for ij -> 2 electron integral index
            nuclearEnergy = mol.energy_nuc() 
            
            if evaluateHam:
                
                def getSpatialFromSpin(i):
                    return i%numSpatialOrbitals

                def isSpinAllowed(i,a):
                    return i//numSpatialOrbitals == a//numSpatialOrbitals
                def getTwoElectronIntegral(i,a,j,b):
                    """Excitation from i->a j->b"""
                    #i a,j,b dont have to be disjoint

                    iaAllowed = isSpinAllowed(i,a)
                    jbAllowed = isSpinAllowed(j,b)
                    ibAllowed = isSpinAllowed(i,b)
                    jaAllowed = isSpinAllowed(j,a)
                    i = getSpatialFromSpin(i)
                    j = getSpatialFromSpin(j)
                    a = getSpatialFromSpin(a)
                    b = getSpatialFromSpin(b)

                    ret = 0
                    if (iaAllowed and jbAllowed):
                        iaIndex = ijIndexes[(i,a)]
                        jbIndex = ijIndexes[(j,b)]
                        ret += twoElectronIntegrals[iaIndex,jbIndex]
                    if (ibAllowed and jaAllowed):
                        ibIndex = ijIndexes[(i,b)]
                        jaIndex = ijIndexes[(j,a)]
                        ret -= twoElectronIntegrals[ibIndex,jaIndex]
                    return ret

                def getFockMatrixElem(i,a,occ : List[int]):
                    #assert(i!=a)

                    if isSpinAllowed(i,a):
                        oneElectronTerm = hcore_mo[getSpatialFromSpin(a),getSpatialFromSpin(i)]
                        oneElectronTerm += h_epotential_mo[getSpatialFromSpin(a),getSpatialFromSpin(i)]
                        
                        moment = h_r_mo[:,getSpatialFromSpin(a),getSpatialFromSpin(i)]

                        alphaSpinMoment = moment if i < numSpatialOrbitals else np.zeros(3)
                        betaSpinMoment  = moment if i >= numSpatialOrbitals else np.zeros(3)
                        

                    else:
                        oneElectronTerm = 0
                        alphaSpinMoment = 0
                        betaSpinMoment = 0
                    
                    oneElectronTerm += h_GradVxP_Spinor[a,i]

                    # if (isinstance(oneElectronTerm,complex)):
                    #     print("complex value in 1 electron integrals. May have got it the wrong way around")

                    twoElectronTerm = 0
                    for j in occ:
                        twoElectronTerm += getTwoElectronIntegral(i,a,j,j)
                    return (oneElectronTerm + twoElectronTerm,alphaSpinMoment,betaSpinMoment)

                def getEnergy(occ : List[int]):
                    ret = 0
                    alphaSpinMoment = 0
                    betaSpinMoment = 0
                    #hii
                    for i in occ:
                        ret += hcore_mo[getSpatialFromSpin(i),getSpatialFromSpin(i)] #always spin allowed
                        ret += h_epotential_mo[getSpatialFromSpin(i),getSpatialFromSpin(i)]
                        
                        ret += h_GradVxP_Spinor[i,i]

                        moment = h_r_mo[:,getSpatialFromSpin(i),getSpatialFromSpin(i)]
                        alphaSpinMoment += moment if i < numSpatialOrbitals else np.zeros(3)
                        betaSpinMoment  += moment if i >= numSpatialOrbitals else np.zeros(3)
                    
                    #sum_i sum_i>j (ii|jj) - (ij|ij)
                    for i in occ:
                        for j in occ:
                            if i <= j:
                                continue
                            ret += getTwoElectronIntegral(i,i,j,j)
                    ret += nuclearEnergy
                    return (ret,alphaSpinMoment,betaSpinMoment)


                Ham = np.zeros((len(hfStates),len(hfStates)),dtype=np.complex128)
                # piggyback on the computation of Hamiltonian matrix elements to get spin resolved |i> c^{\alpha/\beta}_{ij} <j| representation for \frac{\v{r}\cdot\v{E}}{|\v{E}|} operator
                momentMatrixAlphaSpin = np.zeros((3,len(hfStates),len(hfStates)),dtype=np.complex128) 
                momentMatrixBetaSpin = np.zeros((3,len(hfStates),len(hfStates)),dtype=np.complex128)

                for i,state1 in enumerate(hfStates):
                    for j,state2 in enumerate(hfStates):
                        #print(f"{i},{j}")
                        # state1 = {0,8,6,7}
                        # state2 = {0,1,2,6}

                        bothOccupied = list(state1.intersection(state2))
                        bothOccupied.sort()

                        if state1 == state2:
                            pass#get Energy                   

                        
                        topElements = list(state1 - state2) # occupied in state1 and not in state2
                        topElements.sort()
                        totalTop = len(topElements)

                        bottomElements = list(state2 - state1) # occupied in state2 and not in state1
                        bottomElements.sort()
                        totalBottom = len(bottomElements)

                        state1List = list(state1)
                        state1List.sort()

                        state2List = list(state2)
                        state2List.sort()

                        signChange = 1 # sign change from permutations
                        count = 0
                        diffIdxs = []

                        for idx in range(len(state1List)):
                            s1 = state1List[idx]
                            s2 = state2List[idx]
                            if s1 != s2:
                                if s1 in state2List:
                                    signChange *= -1 #align these terms
                                    s1InState2Idx = state2List.index(s1)
                                    state2List[idx], state2List[s1InState2Idx] = state2List[s1InState2Idx], state2List[idx] #swap
                                else:
                                    assert(s1 in topElements)
                                    if s2 != bottomElements[count]: #s2 is not the pair for s1
                                        signChange *= -1 #align these terms
                                        BottomElementInState2Idx = state2List.index(bottomElements[count])
                                        state2List[idx], state2List[BottomElementInState2Idx] = state2List[BottomElementInState2Idx], state2List[idx] #swap
                                    else: #s2 in the correct pair for s1
                                        pass
                                    diffIdxs.append(idx)
                                    count += 1

                        assert([state1List[idx] for idx in diffIdxs] == topElements)
                        assert(bottomElements == [state2List[idx] for idx in diffIdxs])

                        if totalTop + totalBottom > 4:
                            continue # this element is 0. only double excitations have overlap
                        
                        elif totalTop == 2 and totalBottom == 2:
                            #(b1t1|b2t2)-(b1t2|b2t1). Only two electron part survives
                            if Ham[i,j] != 0:
                                print(f"Ham already done:{i},{j}")
                            Ham[i,j] = signChange*getTwoElectronIntegral(bottomElements[0],topElements[0],bottomElements[1],topElements[1])
                        
                        elif totalTop == 1 and totalBottom == 1:
                            #single excitation. Fock Matrix element
                            FME = getFockMatrixElem(bottomElements[0],topElements[0],bothOccupied)
                            Ham[i,j] = signChange*FME[0]
                            momentMatrixAlphaSpin[:,i,j] = signChange*FME[1]
                            momentMatrixBetaSpin[:,i,j] = signChange*FME[2]
                        
                        elif totalTop == 0 and totalBottom == 0:
                            #Energy of MO
                            assert(i==j)
                            assert(signChange == 1)
                            EME = getEnergy(bothOccupied)
                            Ham[i,j] = signChange*EME[0]
                            momentMatrixAlphaSpin[:,i,j] = signChange*EME[1]
                            momentMatrixBetaSpin[:,i,j] = signChange*EME[2]
                        else:
                            print("unknown case")


                if (shift != 0):
                    # Ham += np.diag([-np.average(np.diag(Ham))+0.01]*len(Ham))
                    Ham += np.diag([shift]*len(Ham))
                with open(HamFileName,'wb') as f:
                    pickle.dump((Ham,momentMatrixAlphaSpin,momentMatrixBetaSpin),f)
        else:
            """File exists"""
            with open(HamFileName,"rb") as f:
                Ham,momentMatrixAlphaSpin,momentMatrixBetaSpin = pickle.load(f)
                
        NoEFieldHam = Ham - (np.einsum("xij,x->ij",momentMatrixAlphaSpin,EField) + np.einsum("xij,x->ij",momentMatrixBetaSpin,EField))
        # FakeSpinHam = Ham - 0.5*(momentMatrixBetaSpin*np.linalg.norm(EField)) # changes the energy of the alpha electrons by a different amount
        
        # FakeSpinHam,Ham = (Ham,FakeSpinHam) # switcharoo
        if not os.path.isfile("EV_"+ HamFileName):
            E,V = np.linalg.eigh(Ham) # Ham @ V[:,i] = E[i] * V[:,i]
            NoEFieldE,NoEFieldV = np.linalg.eigh(NoEFieldHam)
            assert(np.all(np.isclose(Ham,np.conj(Ham.T))))
            with open("EV_"+ HamFileName,'wb') as f:
                pickle.dump((E,V,NoEFieldE,NoEFieldV),f)
        else:
            with open("EV_"+ HamFileName,"rb") as f:
                E,V,NoEFieldE,NoEFieldV = pickle.load(f)

        
        """Start State is SO Hamiltonian ground state"""
        startState = NoEFieldV[:,0]
        """Start State is HF state, 
        time evolving from this is not a relaxation to a ground state with fake spin orbit coupling but instead adding S-O coupling as a pertubation at each time step."""

        #startState = np.zeros(len(Ham))
        #startState[0] = 1
        timeIndependentHamiltonian = False

        if timeIndependentHamiltonian:
            TimeIndependentHamiltonianEvolution(mol, Ham, momentMatrixAlphaSpin, momentMatrixBetaSpin, NoEFieldHam, E, V, NoEFieldE, startState)
        else:
            """Hamiltonian is HNoEField + E(t)"""
            #1s = (4.34*10**-18) Eh^-1
            startTime = 0 / (4.34*10**-6)
            endTime = 9 / (4.34*10**-6) # 9ps in a.u
            timestepSize = 0.000001 / (4.34*10**-6) # 0.1 ps in au
            

            timesteps = np.arange(startTime,endTime,timestepSize)
            velocity = 3 *(4.34*10**-6)#Angstroms per picosecond(4.34 *10^-6) in Angstroms/(Eh^-1)

            func_Efield = lambda x :  2*(27-x)*np.array([0,0,1/27]) if x < 27 else [0,0,0]#1 e_h/{e} per angstrom = 27V per angstrom
            currState = startState.astype(np.complex128)
            TimeEvolvedStates = []
            #TimeEvolvedStates.append(currState)
            global EShift 
            EShift = NoEFieldE[0]

            def deltaPsiFunc(t,psi):
                currentEfield = func_Efield(velocity*t)
                global EShift
                mag = np.linalg.norm(psi)
                # if (np.linalg.norm(psi) > 1):
                #     EShift += 0.1*mag
                # elif (np.linalg.norm(psi) < 1):
                #     EShift -= 0.1/mag
                Hamt = NoEFieldHam + (np.einsum("xij,x->ij",momentMatrixAlphaSpin,currentEfield) + np.einsum("xij,x->ij",momentMatrixBetaSpin,currentEfield)) - np.diag(np.ones(len(Ham))*EShift)
                deltaState = -1j * Hamt @ psi #\frac{d}{dt}\psi = H(t)\psi
                print(f"{t},{np.linalg.norm(psi)},{EShift}")
                return deltaState


            if not os.path.isfile("timeEvolve.pkl"):
                ret = solve_ivp(deltaPsiFunc,(startTime,10000),currState,method="RK45")
                with open("timeEvolve.pkl",'wb') as f:
                    pickle.dump(ret,f)
            else:
                with open("timeEvolve.pkl",'rb') as f:
                    ret = pickle.load(f)

            timesteps = ret.t
            TimeEvolvedStates = ret.y.T
            print(len(timesteps))

            
            

            # for t in timesteps:
            #     """Starts at t = 0"""
            #     #TODO verlet algorithm
            #     currentEfield = func_Efield(velocity*t)
            #     Hamt = NoEFieldHam + (np.einsum("xij,x->ij",momentMatrixAlphaSpin,currentEfield) + np.einsum("xij,x->ij",momentMatrixBetaSpin,currentEfield)) - np.diag(np.ones(len(Ham))*NoEFieldE[0])
            #     deltaState = timestepSize * Hamt @ currState #\frac{d}{dt}\psi = H(t)\psi
            #     currState += deltaState
            #     currState /= np.linalg.norm(currState)
            #     TimeEvolvedStates.append(currState)
            #     print(t)

            TimeEvolvedStatesMag = np.einsum("ti,ti->t",np.conj(TimeEvolvedStates),TimeEvolvedStates) 
            AlphaMomentPosition = lib.param.BOHR*np.einsum("ti,xij,tj->xt",np.conj(TimeEvolvedStates),momentMatrixAlphaSpin,TimeEvolvedStates,optimize='optimal')/(TimeEvolvedStatesMag*mol.nelec[0])
            BetaMomentPosition = lib.param.BOHR*np.einsum("ti,xij,tj->xt",np.conj(TimeEvolvedStates),momentMatrixBetaSpin,TimeEvolvedStates,optimize='optimal')/(TimeEvolvedStatesMag*mol.nelec[0])

            plt.plot(timesteps,AlphaMomentPosition[0].real,label="Alpha-X")
            plt.plot(timesteps,BetaMomentPosition[0].real,label="Beta-X")
            plt.plot(timesteps,AlphaMomentPosition[1].real,label="Alpha-Y")
            plt.plot(timesteps,BetaMomentPosition[1].real,label="Beta-Y")
            plt.plot(timesteps,AlphaMomentPosition[2].real,label="Alpha-Z")
            plt.plot(timesteps,BetaMomentPosition[2].real,label="Beta-Z")
            plt.legend()
            plt.show()

            plt.plot(timesteps,AlphaMomentPosition[0].real-BetaMomentPosition[0].real,label="Alpha-Beta X")
            plt.plot(timesteps,AlphaMomentPosition[1].real-BetaMomentPosition[1].real,label="Alpha-Beta Y")
            plt.plot(timesteps,AlphaMomentPosition[2].real-BetaMomentPosition[2].real,label="Alpha-Beta Z")
            plt.legend()
            plt.show()



def TimeIndependentHamiltonianEvolution(mol, Ham, momentMatrixAlphaSpin, momentMatrixBetaSpin, NoEFieldHam, E, V, NoEFieldE, startState):
    print(f"Energies: HF:{NoEFieldHam[0,0].real}, HF+EField:{Ham[0,0].real},FCI{NoEFieldE[0].real},FCI+EField{E[0].real}")
            
    TimeSteps = np.linspace(0,TimeEvolveLength,TimeStepCount)
    if (decayRate == 0):
        TimeModifiedHam = np.einsum("i,t->ti",E, -1j*TimeSteps)
    else:
        TimeModifiedHam = np.einsum("i,t->ti",E-E[0]+0.00001, (-1j-decayRate)*TimeSteps)
            
    TimeEvolvedEnergy = np.exp(TimeModifiedHam)
            #TimeEvolvedStates = np.einsum("ji,tjk,kl,l->til",V,TimeEvolvedMatrices,V,startState)
            
    TimeEvolvedStates = np.einsum("ij,tj->ti",V, np.einsum("ti,i->ti",TimeEvolvedEnergy, np.einsum("ji,j->i",np.conj(V) , startState)))
            # print(TimeEvolvedStates[-1,:])
            # print(TimeEvolvedStates[0,:])
    TimeEvolvedStatesMag = np.einsum("ti,ti->t",np.conj(TimeEvolvedStates),TimeEvolvedStates) 

            #Position in angstroms
            # a = np.einsum("ik,xkj->xij",Ham,momentMatrixAlphaSpin-momentMatrixBetaSpin) - np.einsum("xik,kj->xij",momentMatrixAlphaSpin-momentMatrixBetaSpin,Ham)
            # commStart = 1j*lib.param.BOHR*np.einsum("i,xij,j->x",np.conj(startState),a,startState,optimize='optimal')/(mol.nelectron)
            # comm100 = 1j*lib.param.BOHR*np.einsum("i,xij,j->x",np.conj(TimeEvolvedStates[100]),a,TimeEvolvedStates[100],optimize='optimal')/(TimeEvolvedStatesMag[100]*mol.nelectron)

    AlphaMomentPosition = lib.param.BOHR*np.einsum("ti,xij,tj->xt",np.conj(TimeEvolvedStates),momentMatrixAlphaSpin,TimeEvolvedStates,optimize='optimal')/(TimeEvolvedStatesMag*mol.nelec[0])
    BetaMomentPosition = lib.param.BOHR*np.einsum("ti,xij,tj->xt",np.conj(TimeEvolvedStates),momentMatrixBetaSpin,TimeEvolvedStates,optimize='optimal')/(TimeEvolvedStatesMag*mol.nelec[0])

    plt.plot(TimeSteps,AlphaMomentPosition[0].real,label="Alpha-X")
    plt.plot(TimeSteps,BetaMomentPosition[0].real,label="Beta-X")
    plt.plot(TimeSteps,AlphaMomentPosition[1].real,label="Alpha-Y")
    plt.plot(TimeSteps,BetaMomentPosition[1].real,label="Beta-Y")
    plt.plot(TimeSteps,AlphaMomentPosition[2].real,label="Alpha-Z")
    plt.plot(TimeSteps,BetaMomentPosition[2].real,label="Beta-Z")
    plt.legend()
    plt.show()

    plt.plot(TimeSteps,AlphaMomentPosition[0].real-BetaMomentPosition[0].real,label="Alpha-Beta X")
    plt.plot(TimeSteps,AlphaMomentPosition[1].real-BetaMomentPosition[1].real,label="Alpha-Beta Y")
    plt.plot(TimeSteps,AlphaMomentPosition[2].real-BetaMomentPosition[2].real,label="Alpha-Beta Z")
    plt.legend()
    plt.show()
        
def generateHams(bondLengths,shift=0):
    for L in bondLengths:
        runHamGen(outputName=f"/home/bence/AnsatzSynthesis/AnsatzSynth/cppAnsatzSynth/Hams/LiH_Backwards/LiH{L}",
                   atomString=f"Li 0 0 0 ; H 0 0 {L}",shift=shift)


if __name__ == "__main__":
    runHamGen()
    #bondlengths = np.concatenate((np.arange(1.0,1.5,0.2),np.arange(1.6,4.5,0.2),np.arange(4.6,5.3,0.2)))
    #generateHams(bondlengths,shift=0)
