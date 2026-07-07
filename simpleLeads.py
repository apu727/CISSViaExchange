from cppNEGF import SecondQuantisedHamiltonian as secQuantHam
from cppNEGF import SimpleLeadSelfEnergy,QuantityCalc,SCFSolver,GreensFunction,ExplicitLeadSelfEnergy
from cppNEGF import globals
import numpy as np



def setupPhotoElectronLeads(numberOfSpinOrbitals,
                            photoElectronGamma, photoElectronBandBottom,photoElectronSpin,photoElectronCoupleMatrix,photoElectronMu,
                            ScatteringGamma,ScatteringBandBottom,scatteringCoupleMatrix,ScatteringSpin,ScatteringMu = None,CProj = None,CInvProj = None):
    """By default works in Hartree Units"""
    lead1AlphaParams = SimpleLeadSelfEnergy.Parameters() #photoelectron
    lead1AlphaParams.alpha1 = 1 # No clue tbh
    lead1AlphaParams.t = photoElectronGamma
    
    lead1AlphaParams.mu = photoElectronMu 
    lead1AlphaParams.couplingPoints = photoElectronCoupleMatrix
    lead1AlphaParams.MatrixDimension = numberOfSpinOrbitals
    lead1AlphaParams.magnetisationDir = photoElectronSpin
    lead1AlphaParams.upSpin = True
    lead1AlphaParams.hasBottom = False if photoElectronBandBottom is None else True
    if not photoElectronBandBottom is None:
        lead1AlphaParams.bottomOfBand = photoElectronBandBottom
    if not CProj is None:
        lead1AlphaParams.hasCProj = True
        lead1AlphaParams.CProj = CProj
        lead1AlphaParams.CInvProjDirecttoDirect = CInvProj

    lead1BetaParams = SimpleLeadSelfEnergy.Parameters(lead1AlphaParams)
    lead1BetaParams.upSpin = False

    lead2AlphaParams = SimpleLeadSelfEnergy.Parameters(lead1AlphaParams) # scattering
    lead2AlphaParams.t = ScatteringGamma
    lead2AlphaParams.mu = ScatteringBandBottom-1 if ScatteringMu is None else ScatteringMu 
    lead2AlphaParams.couplingPoints = scatteringCoupleMatrix
    lead2AlphaParams.magnetisationDir = ScatteringSpin
    lead2AlphaParams.upSpin = True
    if not ScatteringBandBottom is None:
        lead2AlphaParams.hasBottom = True 
        lead2AlphaParams.bottomOfBand = ScatteringBandBottom
        
    


    lead2BetaParams = SimpleLeadSelfEnergy.Parameters(lead2AlphaParams)
    lead2BetaParams.upSpin = False

    lead1AlphaSE = SimpleLeadSelfEnergy.SimpleLeadSelfEnergy(lead1AlphaParams)
    lead1BetaSE = SimpleLeadSelfEnergy.SimpleLeadSelfEnergy(lead1BetaParams)
    lead2AlphaSE = SimpleLeadSelfEnergy.SimpleLeadSelfEnergy(lead2AlphaParams)
    lead2BetaSE = SimpleLeadSelfEnergy.SimpleLeadSelfEnergy(lead2BetaParams)

    lead1AlphaSE.name = "PEAlpha"
    lead1BetaSE.name = "PEBeta"
    lead2AlphaSE.name = "ScatAlpha"
    lead2BetaSE.name = "ScatBeta"

    return (lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE)


def setupSimpleLeads(HamParams, CouplingMatrixLHS, CouplingMatrixRHS, VoltageDifference = 0.1, SpinDifference = 0,muDifference = 0,BottomofBandLead1 = -0.2,
                     Gamma = 0.1, numberOfElectrons = None, muMax = None, muInitial = 0.0,RightGamma = None,RightSpinDifference = None,CProj = None,CInvProj = None):   
    if RightGamma is None:
        RightGamma = Gamma

    lead1AlphaParams = SimpleLeadSelfEnergy.Parameters()
    lead1AlphaParams.alpha1 = 1# +SpinDifference/2
    lead1AlphaParams.t = HamParams.t_0*Gamma

    lead1AlphaParams.mu = muInitial - muDifference*HamParams.t_0 
    lead1AlphaParams.couplingPoints = CouplingMatrixLHS
    lead1AlphaParams.MatrixDimension = HamParams.HilbertSpaceSize
    lead1AlphaParams.magnetisationDir = [0,SpinDifference/2,0] if not isinstance(SpinDifference, np.ndarray) else SpinDifference/2
    lead1AlphaParams.upSpin = True
    if not numberOfElectrons is None:
        lead1AlphaParams.adjustMuToElectrons = True
        lead1AlphaParams.muMax = lead1AlphaParams.mu if muMax == None else muMax
        lead1AlphaParams.numberOfElectrons = numberOfElectrons
    if not CProj is None:
        lead1AlphaParams.hasCProj = True
        lead1AlphaParams.CProj = CProj
        lead1AlphaParams.CInvProjDirecttoDirect = CInvProj

    lead1BetaParams = SimpleLeadSelfEnergy.Parameters(lead1AlphaParams)
    lead1BetaParams.couplingPoints = CouplingMatrixLHS
    lead1BetaParams.alpha1 = 1# - SpinDifference/2
    lead1BetaParams.mu = muInitial + muDifference*HamParams.t_0
    lead1BetaParams.upSpin = False

    lead2AlphaParams = SimpleLeadSelfEnergy.Parameters(lead1AlphaParams)
    lead2AlphaParams.t = HamParams.t_0*RightGamma
    lead2AlphaParams.adjustMuToElectrons = False
    lead2AlphaParams.mu = muInitial-VoltageDifference*globals.e/globals.electricPotentialUnit
    lead2AlphaParams.alpha1 = 1
    lead2AlphaParams.couplingPoints = CouplingMatrixRHS
    lead2AlphaParams.magnetisationDir = [0,0,0] if RightSpinDifference is None else RightSpinDifference
    lead2AlphaParams.upSpin = True


    lead2BetaParams = SimpleLeadSelfEnergy.Parameters(lead2AlphaParams)
    lead2BetaParams.couplingPoints = CouplingMatrixRHS
    lead2BetaParams.alpha1 = 1
    lead2BetaParams.upSpin = False

    lead1AlphaSE = SimpleLeadSelfEnergy.SimpleLeadSelfEnergy(lead1AlphaParams)
    lead1BetaSE = SimpleLeadSelfEnergy.SimpleLeadSelfEnergy(lead1BetaParams)
    lead2AlphaSE = SimpleLeadSelfEnergy.SimpleLeadSelfEnergy(lead2AlphaParams)
    lead2BetaSE = SimpleLeadSelfEnergy.SimpleLeadSelfEnergy(lead2BetaParams)

    lead1AlphaSE.name = "lead1AlphaSE"
    lead1BetaSE.name = "lead1BetaSE"
    lead2AlphaSE.name = "lead2AlphaSE"
    lead2BetaSE.name = "lead2BetaSE"

    return (lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE)

def setupExplicitLeads(HamParams, VoltageDifference = 0.1,SpinDifference = 0, muDifference = 0,BottomofBandLead1 = -0.2, t_0 = None):

    lead1AlphaParams = ExplicitLeadSelfEnergy.Parameters()
    lead1AlphaParams.a = HamParams.R
    lead1AlphaParams.alpha1 = 1
    lead1AlphaParams.t = HamParams.t_0 if t_0 is None else t_0
    lead1AlphaParams.bottomOfBand = BottomofBandLead1 - SpinDifference/2
    lead1AlphaParams.sign = 1
    lead1AlphaParams.mu = -muDifference/2
    lead1AlphaParams.couplingPoints = [0]
    lead1AlphaParams.MatrixDimension = HamParams.HilbertSpaceSize

    lead1BetaParams = ExplicitLeadSelfEnergy.Parameters(lead1AlphaParams)
    lead1BetaParams.bottomOfBand += SpinDifference
    lead1BetaParams.couplingPoints = [1]
    lead1BetaParams.mu = muDifference/2

    lead2AlphaParams = ExplicitLeadSelfEnergy.Parameters(lead1AlphaParams)
    lead2AlphaParams.alpha1 = 0.9
    lead2AlphaParams.bottomOfBand = BottomofBandLead1 - VoltageDifference*globals.e/globals.electricPotentialUnit
    lead2AlphaParams.mu = - VoltageDifference*globals.e/globals.electricPotentialUnit
    lead2AlphaParams.couplingPoints = [HamParams.HilbertSpaceSize-2]
    lead2AlphaParams.sign = 1

    lead2BetaParams = ExplicitLeadSelfEnergy.Parameters(lead2AlphaParams)
    lead2AlphaParams.alpha1 = 0.9
    lead2BetaParams.bottomOfBand = BottomofBandLead1 - VoltageDifference*globals.e/globals.electricPotentialUnit
    lead2BetaParams.mu = - VoltageDifference*globals.e/globals.electricPotentialUnit
    lead2BetaParams.couplingPoints = [HamParams.HilbertSpaceSize-1]
    lead2BetaParams.sign = 1
    

    lead1AlphaSE = ExplicitLeadSelfEnergy.ExplicitLeadSelfEnergy(lead1AlphaParams)
    lead1BetaSE = ExplicitLeadSelfEnergy.ExplicitLeadSelfEnergy(lead1BetaParams)
    lead2AlphaSE = ExplicitLeadSelfEnergy.ExplicitLeadSelfEnergy(lead2AlphaParams)
    lead2BetaSE = ExplicitLeadSelfEnergy.ExplicitLeadSelfEnergy(lead2BetaParams)
    
    lead1AlphaSE.name = "lead1AlphaSE"
    lead1BetaSE.name = "lead1BetaSE"
    lead2AlphaSE.name = "lead2AlphaSE"
    lead2BetaSE.name = "lead2BetaSE"

    return (lead1AlphaSE,lead1BetaSE,lead2AlphaSE,lead2BetaSE)




