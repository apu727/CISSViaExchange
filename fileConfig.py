from cppNEGF import SecondQuantisedHamiltonian as secQuantHam
from cppNEGF import SimpleLeadSelfEnergy,QuantityCalc,SCFSolver,GreensFunction,PhononSelfEnergy,ConfigFileLoader
from cppNEGF import globals

import matplotlib.pyplot as plt
import numpy as np
from simpleLeads import *
import sys

PathToSaveTo = sys.argv[1] if len(sys.argv) > 1 else "./"
Headless = True

cfl = ConfigFileLoader.ConfigFileLoader(PathToSaveTo + "Config.txt")
Ham = cfl.getHam()
HilbertSpaceSize = Ham.getHilbertSpaceSize()
SCF = cfl.getSCF()
currentCalcs = cfl.getCurrentCalcs()
for cc in currentCalcs:
    if (cc.selfEnergyName == "lead1AlphaSE"):
        lead1AlphaCurrent = cc
    if (cc.selfEnergyName == "lead2AlphaSE"):
        lead2AlphaCurrent = cc
    if (cc.selfEnergyName == "lead1BetaSE"):
        lead1BetaCurrent = cc
    if (cc.selfEnergyName == "lead2BetaSE"):
        lead2BetaCurrent = cc
    if (cc.selfEnergyName == "PhononSE"):
        PhononCurrent = cc
electronDensity = cfl.getElectronDensityCalcs()[0]
spectralDensity = cfl.getSpectraFunctionCalcs()[0]


def plotiE():
    ESpace = np.linspace(-0.01,0.01,1000)
    I1Alpha = np.array(lead1AlphaCurrent.getiE(ESpace))
    I1Beta = np.array(lead1BetaCurrent.getiE(ESpace))
    # I2Alpha = np.array(lead2AlphaCurrent.getiE(ESpace))
    # I2Beta = np.array(lead2BetaCurrent.getiE(ESpace))
    IPhi = np.array(PhononCurrent.getiE(ESpace))


    #plt.plot(ESpace,I1Alpha-I1Beta)

    Total1Alpha = lead1AlphaCurrent.getI()
    Total1Beta =  lead1BetaCurrent.getI()

    print(f"Polarisation:{100*(Total1Alpha-Total1Beta)/(Total1Alpha+Total1Beta)}")
    plt.plot(ESpace,I1Alpha,label="Alpha")
    plt.plot(ESpace,I1Beta,label="Beta")
    #plt.plot(ESpace,I2Alpha,label="Alpha2")
    #plt.plot(ESpace,I2Beta,label="Beta2")
    plt.plot(ESpace,IPhi,label="IPhi")
    #plt.ylim(-25,25)
    plt.legend()
    with open(PathToSaveTo + "iE.png","wb") as f:
        plt.savefig(f,dpi=400)
    if not Headless:
        plt.show()
    else:
        plt.clf()

def plotSpinPolarisedTransmission():
    ESpace = np.linspace(-0.1,0.1,1000)
    I1Alpha = np.array(lead1AlphaCurrent.getiE(ESpace))
    I1Beta = np.array(lead1BetaCurrent.getiE(ESpace))
    I2Alpha = np.array(lead2AlphaCurrent.getiE(ESpace))
    I2Beta = np.array(lead2BetaCurrent.getiE(ESpace))
    #IPhi = np.array(PhononCurrent.getiE(ESpace))


    #plt.plot(ESpace,I1Alpha-I1Beta)
    plt.plot(ESpace,100*(I1Alpha-I1Beta)/(I1Alpha + I1Beta),label="lead1 polarisation %")
    plt.plot(ESpace,100*(I2Alpha-I2Beta)/(I2Alpha + I2Beta),label="lead2 polarisation %")
    #plt.plot(ESpace,IPhi,label="IPhi")
    #plt.ylim(-100,100)
    plt.legend()
    with open(PathToSaveTo + "SpinPolarisedTransmission.png","wb") as f:
        plt.savefig(f,dpi=400)
    if not Headless:
        plt.show()
    else:
        plt.clf()

# def plotDensity():
#     #nER = lambda E,r: GfuncCalc.getGnAtE(E)[r,r]/(2*np.pi) # the 2pi comes from the energy time fourier transform?
#     positions = list(range(HamParams.HilbertSpaceSize//2))
#     ERange = np.linspace(-0.01,0.01,5000)
#     EStep = ERange[1] - ERange[0]
#     E,R = np.meshgrid(ERange,positions)
#     nE = []
#     for e in ERange:
#         nE.append(GfuncCalc.getGnAtE(e)*EStep/(2*np.pi))

#     NAlpha = np.array([[nE[idx][2*r,2*r].real for r in positions] for idx in range(len(ERange))]).T  
#     NBeta = np.array([[nE[idx][2*r+1,2*r+1].real for r in positions] for idx in range(len(ERange))]).T  

#     fig, axs = plt.subplots(3)

#     img1 = axs[0].contourf(R, E, NAlpha, 100)
#     img2 = axs[1].contourf(R, E, NBeta, 100)
#     img3 = axs[2].contourf(R, E, NAlpha-NBeta, 100)
#     fig.colorbar(img1)
#     fig.colorbar(img2)
#     fig.colorbar(img3)

#     # for Eigval in np.linalg.eigvals(HamMatrix):
#     #     plt.axhline(y = Eigval, color = 'r', linestyle = '-') 
#     axs[1].set_xlabel("Position ith site")
#     axs[1].set_ylabel("Energy")
#     plt.show()

def plotDensity():
    positions = list(range(HilbertSpaceSize//2))
    Density = electronDensity.getnR()

    #print(f"AlphaDensity:{Density[::2]}\n BetaDensity:{Density[1::2]}")


    NAlpha = Density[::2]
    NBeta = Density[1::2]

    plt.plot(positions,NAlpha,label="N Alpha")
    plt.plot(positions,NBeta,label="N Beta")
    plt.legend()
    with open(PathToSaveTo + "ElectronDensity.png","wb") as f:
        plt.savefig(f,dpi=400)
    if not Headless:
        plt.show()
    else:
        plt.clf()

def plotSpectralDensity():
    positions = list(range(HilbertSpaceSize//2))
    Density = spectralDensity.getAR()

    #print(f"AlphaDensity:{Density[::2]}\n BetaDensity:{Density[1::2]}")


    NAlpha = Density[::2]
    NBeta = Density[1::2]

    plt.plot(positions,NAlpha,label="N Alpha")
    plt.plot(positions,NBeta,label="N Beta")
    plt.legend()
    with open(PathToSaveTo + "States.png","wb") as f:
        plt.savefig(f,dpi=400)
    if not Headless:
        plt.show()
    else:
        plt.clf()

def myplot():
    ERange = np.linspace(-0.01,0.01,5000)
    EStep = ERange[1] - ERange[0]
    AE = np.array(spectralDensity.getAEr(ERange))*EStep    

    NAlpha = np.array([AE[idx][0].real for idx in range(len(ERange))])  
    NBeta = np.array([AE[idx][1].real  for idx in range(len(ERange))])

    plt.plot(ERange,NAlpha,label="N Alpha")
    plt.plot(ERange,NBeta,label="N Beta")
    plt.legend()
    with open(PathToSaveTo + "SpectralDensityOfStatesAtSite1.png","wb") as f:
        plt.savefig(f,dpi=400)
    if not Headless:
        plt.show()
    else:
        plt.clf()

def plotEnergySpectralDensity():
    positions = list(range(HilbertSpaceSize//2))
    ERange = np.linspace(-0.01,0.01,5000)
    EStep = ERange[1] - ERange[0]
    E,R = np.meshgrid(ERange,positions)
    AE = np.array(spectralDensity.getAEr(ERange))*EStep    


    NAlpha = np.array([[AE[idx][2*r].real for r in positions] for idx in range(len(ERange))]).T  
    NBeta = np.array([[AE[idx][2*r+1].real for r in positions] for idx in range(len(ERange))]).T  

    fig, axs = plt.subplots(3)

    img1 = axs[0].contourf(R, E, NAlpha, 100)
    img2 = axs[1].contourf(R, E, NBeta, 100)
    img3 = axs[2].contourf(R, E, NAlpha-NBeta, 100)
    fig.colorbar(img1)
    fig.colorbar(img2)
    fig.colorbar(img3)

    # for Eigval in np.linalg.eigvals(HamMatrix):
    #     plt.axhline(y = Eigval, color = 'r', linestyle = '-') 
    axs[1].set_xlabel("Position ith site")
    axs[1].set_ylabel("Energy")
    with open(PathToSaveTo + "SpectralDensityOfStates.png","wb") as f:
        plt.savefig(f,dpi=400)
    if not Headless:
        plt.show()
    else:
        plt.clf()


def plotEnergyDensity():
    positions = list(range(HilbertSpaceSize//2))
    ERange = np.linspace(-0.01,0.01,5000)
    EStep = ERange[1] - ERange[0]
    E,R = np.meshgrid(ERange,positions)
    nE = np.array(electronDensity.getnEr(ERange))*EStep    


    NAlpha = np.array([[nE[idx][2*r].real for r in positions] for idx in range(len(ERange))]).T  
    NBeta = np.array([[nE[idx][2*r+1].real for r in positions] for idx in range(len(ERange))]).T  

    fig, axs = plt.subplots(3)

    img1 = axs[0].contourf(R, E, NAlpha, 100)
    img2 = axs[1].contourf(R, E, NBeta, 100)
    img3 = axs[2].contourf(R, E, NAlpha-NBeta, 100)
    fig.colorbar(img1)
    fig.colorbar(img2)
    fig.colorbar(img3)

    # for Eigval in np.linalg.eigvals(HamMatrix):
    #     plt.axhline(y = Eigval, color = 'r', linestyle = '-') 
    axs[1].set_xlabel("Position ith site")
    axs[1].set_ylabel("Energy")
    with open(PathToSaveTo + "ElectronSpectralDensity.png","wb") as f:
        plt.savefig(f,dpi=400)
    if not Headless:
        plt.show()
    else:
        plt.clf()
    

#SCF.setCallBackFunction(lambda :plotDensity() or plotEnergyDensity() or plotEnergySpectralDensity() or myplot())
cfl.setFileName(PathToSaveTo + "Parameters.txt")
cfl.save()
SCF.setCallBackFunction(lambda : 1)
SCF.run()
plotDensity()
plotEnergyDensity()
plotSpectralDensity()
plotEnergySpectralDensity()
myplot()
plotiE()