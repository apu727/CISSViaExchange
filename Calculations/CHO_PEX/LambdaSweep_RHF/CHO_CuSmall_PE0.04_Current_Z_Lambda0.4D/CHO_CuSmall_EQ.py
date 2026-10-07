from hartree_fock import GreensFunctionMethodParameters,GreensFunctionMethod
from cppNEGF import globals
import numpy as np
spinLambda = 0.4
spinLambdaStr = "0.4D"
params = GreensFunctionMethodParameters()
params.activeOrbitals = None
params.frozenOrbitals = set()
params.outputName = f"Calculations/CHO_PEX/LambdaSweep_RHF/CHO_CuSmall_PE0.04_Current_Z_Lambda{spinLambdaStr}/CHO"

#Lead parameters
params.startGamma = 1e-2
params.maxGamma = 1e-2
params.RightGamma = 0 # None to set the same as left
params.GammaSteps = 1
params.muLeft = -0.15 
params.startVoltage = 0*globals.electricPotentialUnit #10
params.EndVoltage = 0*globals.electricPotentialUnit #10
params.VoltageSteps = 2

params.TStart = 300
params.TEnd = 300
params.TStep = 0

maintainElectronNumber = True
params.muMax = 200 # Pointless?

params.coupleBasedOnElement = True
params.ElementCoupleLeft = [29] 
params.ElementCoupleRight = [-1] 
params.SpinDifference = np.array([0.04687, 0.13027, 0.99037])*0.04*2 # This is divided by 2 for some reason
params.RightSpinDifference = np.array([0,0,0])*2

#PhotoElectronLeads
params.photoElectronGamma = 1e-4
params.photoElectronBandBottom = None # None means no bottom
params.photoElectronSpin = np.array([0.04687, 0.13027, 0.99037])*0.04
params.photoElectronElementCouple = [29]
params.photoElectronExcitationEnergy = 5.9/globals.electricPotentialUnit # 5.9eV # gets added to muLeft

params.ScatteringGamma = 1e-2
params.ScatteringBandBottom = 0 # AKA vaccum level
params.ScatteringElementCouple = [8,6,1]
params.ScatteringSpin = np.array([0.04687, 0.13027, 0.99037])*1e-8
params.ScatteringMu = params.ScatteringBandBottom-1
params.useEDIIS = True



#less common user settings
params.basis = "sto-3g"
params.charge = 0
params.spin = 0
params.workInLowdinBasis = True
params.forceSelfAdjointSelfEnergy = False
params.useHFOrbitals = False or len(params.frozenOrbitals) > 0
params.useOrthogonalOrbitals = False # Takes effect only if youre not using HF orbitals
#Auto determined
# params.usingDualSpace = not (params.useHFOrbitals or params.useOrthogonalOrbitals) # Only use dual space if both are false.
params.loadFromQChemData = True

params.twoLeads = True
params.doPhotoElectronLeads = True
params.doPhotoElectronSCF = True

params.doSOC = False
params.spinLambda = spinLambda
# SOCScaleFactor = 1# Multiplies the SOC contribution by a scale factor. This is plain wrong but can give useful results. Now in PyscfInterface

params.computeEnergy = False if params.doSOC else True
params.SCFRead = True # Reads a finalDensity file if it can find one
# params.CDIISInitialAlpha = 1 # default 0.3
params.doSCF = True # Actually do SCF or just use the loaded density. Usually you want to use SCF
params.doRestrictedCalc = False
params.SCFConvergence = 5 # Same as QCHEM, 10**-SCFConvergence
params.doAintegral = False
params.doGnIntegral = True
params.doNonEqGnIntegral = True
params.doPlots = False
params.doCurrentPlots = True 
params.doSpectralPlots = True

doNonEqGnIntegral = True
params.NonEqGnStart = 1e-10
params.NonEqGnEnd = params.muLeft+params.photoElectronExcitationEnergy+1;

params.saveFrozenCore = False
params.loadFrozenCore = False
params.frozenCoreEnergy = -1

params.finalise()

GF = GreensFunctionMethod(params)
GF.runHamGen()

params.useEDIIS = False
params.SCFConvergence = 8
params.doPlots = True
GF = GreensFunctionMethod(params)
GF.runHamGen()
