from pyscf import gto, scf,ao2mo

from SOCIntegrals import make_h1_soc, koseki_charge
from cppNEGF import ComplexDualSelfAdjointMatrix,  SpinSymmetry, BasisManager,ComplexDirectSelfAdjointMatrix,ComplexDualMatrix,ComplexDirectMatrix,LoweringMetric
import numpy as np

SOCScaleFactor = 1

def getSpinorVersionBAD(Mat):
    #This is needed to Bootstrap the metric until BasisManager can support a spin conversion in a basis which doesnt exist
    ret = np.zeros(shape=np.array(Mat.shape)*2)
    for i in range(Mat.shape[0]):
        for j in range(Mat.shape[1]):
            ret[2*i,2*j] = Mat[i,j]
            ret[2*i+1,2*j+1] = Mat[i,j]
    return ret

def getSpinorVersion(Mat):
    return Mat.toBasisCast(type(Mat)(),BasisManager.removeSpinTags(Mat.getBasis()))

def generateIntegrals(atomString,basis,isECP,Generate2eInts, BreitIntegrals = False, atomicCharges = None, namedBasis = "AO", spinLambda = 1):

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
        factor = SOCScaleFactor*(7.2973525664e-3)**2  /4 # 1/4c^2 if you use sigma, 1/2c^2 if you use shat. Note the factor for spin 1/2
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
        hSOC[::2,1::2] +=  -1j*hso1e[1,:,:]*factor
        hSOC[1::2,0::2] +=  1j*hso1e[1,:,:]*factor

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
        print(f"DM is RHF?:{dm.getSpinSym() == SpinSymmetry.RHF}")
        if dm.getSpinSym() == SpinSymmetry.RHF:
            dmaa = dm.toBasis(ComplexDualSelfAdjointMatrix(),namedBasis + BasisManager.Alpha_Block).toNumpy()
            j1, k1 = scf.hf.get_jk(mol, dmaa,hermi=0)
            return ComplexDirectSelfAdjointMatrix(2*j1 - k1,namedBasis+BasisManager.Alpha_Block,SpinSymmetry.RHF).toBasis(ComplexDirectSelfAdjointMatrix(),namedBasis)
            #return 2*j1 - k
        else:
            # dm = convertToPyScfSpinorOrdering(dm)
            # J,K = pyscf.scf.ghf.get_jk(mol,SpinorD)
            #GHF Has a bug so copy the function here and fix it
            
            nso = dm.rows()
            nao = nso // 2

            # dmaa = dm[:nao,:nao]
            # dmab = dm[:nao,nao:]
            # dmbb = dm[nao:,nao:]
            # dmba = dm[nao:,:nao]

            dmaa = dm.toBasis(ComplexDualMatrix(),namedBasis + BasisManager.Alpha_Block).toNumpy()
            dmab = dm.toBasis(ComplexDualMatrix(),[namedBasis + BasisManager.Alpha_Block,namedBasis + BasisManager.Beta_Block]).toNumpy()
            dmbb = dm.toBasis(ComplexDualMatrix(),namedBasis + BasisManager.Beta_Block).toNumpy()
            dmba = dm.toBasis(ComplexDualMatrix(),[namedBasis + BasisManager.Beta_Block,namedBasis + BasisManager.Alpha_Block]).toNumpy()
            
            dms = np.stack((dmaa, dmbb, dmab, dmba))
            
            j1, k1 = scf.hf.get_jk(mol, dms,hermi=0)
            print(f"spinLambda:{spinLambda}")
            if spinLambda != 1:
                #Rescale the Non identity spin parts by spinLambda. 
                # i.e. 
                # [KAA,KAB] =  [KAA/2+KBB/2,0] +  Lambda [KAA/2-KBB/2,KAB         ] = KI/2 + Lambda [KZ/2.   ,KX/2-iKY/2]
                # [KBA,KBB]    [0,KAA/2+KBB/2]           [KBA.       , KBB/2-KAA/2]                 [KX/2+iKY/2, -KZ/2  ]
                KI = k1[0] + k1[1]
                # KX = k1[2] + k1[3]
                # KY = 1j*(k1[2] -k1[3])
                KZ = k1[0] - k1[1]

                

                k1[0] = KI/2 + KZ*(spinLambda/2)
                k1[1] = KI/2 - KZ*(spinLambda/2)
                k1[2] *= spinLambda
                k1[3] *= spinLambda

                
            # vj = vk = None
            # vj = np.zeros((nso,nso), dm.dtype)
            # vj[:nao,:nao] = vj[nao:,nao:] = j1[0] + j1[1]
            vj = ComplexDirectMatrix(j1[0] + j1[1],namedBasis+BasisManager.Alpha_Block,SpinSymmetry.RHF)
            vj = vj.toBasis(ComplexDirectMatrix(),namedBasis)
            
            # vk = np.zeros((nso,nso), dm.dtype)
            # vk[:nao,:nao] = k1[0]
            # vk[nao:,nao:] = k1[1]
            # vk[:nao,nao:] = k1[2]
            # vk[nao:,:nao] = k1[3]
            vj -= ComplexDirectMatrix(k1[0],namedBasis+BasisManager.Alpha_Block,SpinSymmetry.NoSpinSym).toBasis(ComplexDirectMatrix(),namedBasis,True)
            vj -= ComplexDirectMatrix(k1[1],namedBasis+BasisManager.Beta_Block,SpinSymmetry.NoSpinSym).toBasis(ComplexDirectMatrix(),namedBasis,True)
            vj -= ComplexDirectMatrix(k1[2],[namedBasis + BasisManager.Alpha_Block,namedBasis + BasisManager.Beta_Block],SpinSymmetry.NoSpinSym).toBasis(ComplexDirectMatrix(),namedBasis,True)
            vj -= ComplexDirectMatrix(k1[3],[namedBasis + BasisManager.Beta_Block,namedBasis + BasisManager.Alpha_Block],SpinSymmetry.NoSpinSym).toBasis(ComplexDirectMatrix(),namedBasis,True)


            # return convertFromPyscfSpinorOrdering(vj-vk)
            return vj.to(ComplexDirectSelfAdjointMatrix())
    #The "" basis always exists and is orthogonal. So this will succeed and can be used to bootstrap the metric.
    
    S_AO = getSpinorVersionBAD(S_AO)
    S_AO = LoweringMetric(S_AO,namedBasis,SpinSymmetry.RHF)
    if Generate2eInts:
        print("Generating 2e Integrals")
        twoElectronIntegrals = ao2mo.kernel(mol,np.identity(mol.nao),aosym=1) 
        print("Done Generating 2e Integrals")
        # eri4 = mol.intor('int2e', aosym='s1')
        hcore = ComplexDirectSelfAdjointMatrix(hcore,namedBasis + BasisManager.Alpha_Block,SpinSymmetry.RHF)
        if not hecp is None:
            hecp = ComplexDirectSelfAdjointMatrix(hecp,namedBasis + BasisManager.Alpha_Block,SpinSymmetry.RHF)
        if not hSOC is None:
            hSOC = LoweringMetric(hSOC,"namedBasis",SpinSymmetry.NoSpinSym)
        return (hcore,twoElectronIntegrals,mol.energy_nuc(),S_AO,hecp, makejk,hSOC)
    else:
        hcore = ComplexDirectSelfAdjointMatrix(hcore,namedBasis + BasisManager.Alpha_Block,SpinSymmetry.RHF)
        if not hecp is None:
            hecp = ComplexDirectSelfAdjointMatrix(hecp,namedBasis + BasisManager.Alpha_Block,SpinSymmetry.RHF)
        if not hSOC is None:
            hSOC = ComplexDirectSelfAdjointMatrix(hSOC,namedBasis,SpinSymmetry.NoSpinSym)    
        
        return (hcore,None,mol.energy_nuc(),S_AO,hecp, makejk,hSOC)





def pyscfGen(atomString,basis,charge=0,spin=0, initialDensityGuess = None,doSOC=False,):
    """Actually do a RHF calculation. Returns the pyscf SCF object"""
    #Not implemented checks
    assert(doSOC == False)

    mol = gto.Mole()
    mol.atom = atomString

    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()
    

    #mol.symmetry = 1 // do symmetry
    rhf_H2 = scf.RHF(mol)
    # rhf_H2.verbose = 10000
    if initialDensityGuess is None:
        e_H2 = rhf_H2.kernel()
    else:
        e_H2 = rhf_H2.kernel(dm0=initialDensityGuess)
    
    #MAKE MEP Cube file
    # dm = np.zeros(shape=(len(AllOrbs),len(AllOrbs)))
    # for i in range(mol.nelec[0]):
    #     dm[i,i] = 1
    # tools.cubegen.mep(mol,f"{outputName}MEP.cube",dm)
    # print(rhf_H2.energy_elec()[0]/2)
    return rhf_H2
