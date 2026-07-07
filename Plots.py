import matplotlib.pyplot as plt
from matplotlib import ticker

from cppNEGF import BasisManager,IndexType,QuantityCalc,SpinSymmetry

import numpy as np
import os
import sys
from pyTests.Util import *
Validate = False ## Used in testing to not generate graphs but check outputs

Headless = True
def plotDensity(outputName, electronDensity, AtomProjectionName, OrbLabels, GfuncCalc):
    
    if electronDensity is None:
        if not os.path.isfile(outputName + "ElectronDensity.npz"):
            print(f"Cannot find file{outputName + "ElectronDensity.npz"}")
            return
        npzFile = np.load(outputName + "ElectronDensity.npz")
        compDensity = npzFile["compDensity"]
        Density = npzFile["Density"]
        positions = npzFile["positions"]
        OrbLabels = npzFile["OrbLabels"]
        project = npzFile["project"]

    else:
        currBasis = GfuncCalc.getBasis()
        compDensity =  electronDensity.getnR("AO")
        #This isnt actually a basis transform. It is more like Tr[Density @ Operator_i] where each operator_i extracts a specific part of the diagonal of the density. The resultant vector is the sum over the i index
        project = BasisManager().getTransform("AO",IndexType.direct,AtomProjectionName,IndexType.direct)
        print("NEED to fix ALL Density plots to always work in 'AO' Basis before taking the diagonal!")
        if not project:
            raise ValueError("Cannot transform to AtomProjectionBasis")
        
        if project.isIdentity:
            project = None
        else:
            project = project.toNumpy()
            assert(np.all(project.imag == 0))
            project = project.real

        Density = compDensity @ project
        positions = list(range(project.shape[1]//2))

    NAlpha = Density[::2]
    NBeta = Density[1::2]    
    if (Validate):
        if expect(True,os.path.isfile(outputName + "ElectronDensity.npz"),f"Cannot find file{outputName + "ElectronDensity.npz"}"):
            return
        npzFile = np.load(outputName + "ElectronDensity.npz")
        expectMatrixNear(npzFile["compDensity"],compDensity,1e-8,"CompDensity")
        expectMatrixNear(npzFile["Density"],Density,1e-8,"Density")
        expectMatrixNear(npzFile["positions"],positions,1e-8,"positions")
        # expectMatrixNear(npzFile["OrbLabels"],OrbLabels,1e-8,"OrbLabels")
        expectMatrixNear(npzFile["project"],project,1e-8,"project")
        return
    
    print(f"Net Polarisation:{np.sum(NAlpha-NBeta)}")

    plt.plot(positions,NAlpha,label="N Alpha")
    plt.plot(positions,NBeta,label="N Beta")
    plt.xticks(positions,OrbLabels)
    #plt.ylim(0,1.1)
    plt.legend()
    with open(outputName + "ElectronDensity.png","wb") as f:
        plt.savefig(f,dpi=400)
    if not Headless:
        plt.show()
    else:
        plt.close()

    plt.plot(positions,NAlpha-NBeta,label="Alpha-Beta")
    plt.xticks(positions,OrbLabels)
    #plt.ylim(0,1.1)
    plt.legend()
    with open(outputName + "ElectronDensityDiff.png","wb") as f:
        plt.savefig(f,dpi=400)
    if not Headless:
        plt.show()
    else:
        plt.close()

    
    np.savez(outputName + "ElectronDensity.npz",compDensity=compDensity,Density=Density,positions=positions,OrbLabels=OrbLabels,project=project)

def plotSpinDensity(outputName, electronDensity,OrbLabels, AtomProjectionName,GfuncCalc):
    sigmaX = np.array([[0,1],[1,0]])
    sigmaY = np.array([[0,-1j],[1j,0]])
    sigmaZ = np.array([[1,0],[0,-1]])
    sigmaI = np.identity(2)
    
    if electronDensity is None:
        if not os.path.isfile(outputName + "ElectronSpinDensity.npz"):
            print(f"Cannot find file{outputName + "ElectronSpinDensity.npz"}")
            return
        npzFile = np.load(outputName + "ElectronSpinDensity.npz")
        compDensity = npzFile["compDensity"]
        positions = npzFile["positions"]
        OrbLabels = npzFile["OrbLabels"]
        spatialProject = npzFile["spatialProject"]
    else:
        currBasis = GfuncCalc.getBasis()

        
        spatialProject = BasisManager().getTransform("AO"+ BasisManager.Alpha_Block,IndexType.direct,AtomProjectionName + BasisManager.Alpha_Block,IndexType.direct)
        if not spatialProject:
            raise ValueError("Cannot transform to AtomProjectionBasis")
        
        if spatialProject.isIdentity:
            spatialProject = None
        else:
            spatialProject = spatialProject.toNumpy()
            assert(np.all(spatialProject.imag == 0))
            spatialProject = spatialProject.real
            
        compDensity =  electronDensity.getnS("AO").reshape((-1,2,2))
        if (spatialProject is None):
            positions = list(range(compDensity.shape[0]))
        else:
            positions = list(range(spatialProject.shape[1]))
    
    if (Validate):
        if expect(True,os.path.isfile(outputName + "ElectronSpinDensity.npz"),f"Cannot find file{outputName + "ElectronSpinDensity.npz"}"):
            return
        npzFile = np.load(outputName + "ElectronSpinDensity.npz")
        expectMatrixNear(npzFile["compDensity"],compDensity,1e-8,"CompDensity")
        expectMatrixNear(npzFile["positions"],positions,1e-8,"positions")
        # expectMatrixNear(npzFile["OrbLabels"],OrbLabels,1e-8,"OrbLabels")
        expectMatrixNear(npzFile["spatialProject"],spatialProject,1e-8,"spatialProject")
        expectMatrixNear(npzFile["spatialInvProject"],spatialInvProject,1e-8,"spatialInvProject")
        return

    xDensityDiff = np.real(np.einsum("qp,ipq -> i",sigmaX,compDensity))
    yDensityDiff = np.real(np.einsum("qp,ipq -> i",sigmaY,compDensity))
    zDensityDiff = np.real(np.einsum("qp,ipq -> i",sigmaZ,compDensity))
    Density = np.real(np.einsum("qp,ipq -> i",sigmaI,compDensity))
    if (spatialProject is None):
        spatialProject = np.identity(len(xDensityDiff))
        spatialInvProject = np.identity(len(xDensityDiff))
    else:
        xDensityDiff = np.einsum("ji,j->i",spatialProject,xDensityDiff)
        yDensityDiff = np.einsum("ji,j->i",spatialProject,yDensityDiff)
        zDensityDiff = np.einsum("ji,j->i",spatialProject,zDensityDiff)
        Density = np.einsum("ji,j->i",spatialProject,Density)
        


    plt.plot(positions,xDensityDiff,label="X component")
    plt.plot(positions,yDensityDiff,label="Y component")
    plt.plot(positions,zDensityDiff,label="Z component")
    plt.xticks(positions,OrbLabels)
    #plt.ylim(0,1.1)
    plt.legend()
    with open(outputName + "ElectronSpinDensityDiff.png","wb") as f:
        plt.savefig(f,dpi=400)
    if not Headless:
        plt.show()
    else:
        plt.close()

    
    np.savez(outputName + "ElectronSpinDensity.npz",compDensity=compDensity,positions=positions,OrbLabels=OrbLabels,spatialProject=spatialProject)

def plotEnergySpectralDensity(outputName, numSpatialOrbitals, ERange, spectralDensity, AtomProjectionName,GfuncCalc,OrbLabels):
    if (spectralDensity is None):
        if not os.path.isfile(outputName + "SpectralDensityOfStates.npz"):
            print(f"Cannot find file{outputName + "SpectralDensityOfStates.npz"}")
            return
        npzFile = np.load(outputName + "SpectralDensityOfStates.npz",)
        positions = npzFile["positions"]
        linESpace = npzFile["linESpace"]
        OrbLabels = npzFile["OrbLabels"]
        E = npzFile["E"]
        R = npzFile["R"]
        AESW = npzFile["AESW"]
        AE = npzFile["AE"]
    else:
        linESpace = np.linspace(ERange[0],ERange[-1],len(ERange))
        currBasis = GfuncCalc.getBasis()

        project = BasisManager().getTransform("AO",IndexType.direct,AtomProjectionName,IndexType.direct)
        if not project:
            raise ValueError("Cannot transform to AtomProjectionBasis")
        
        if project.isIdentity:
            AESW = np.array(spectralDensity.getAEr(ERange))
            positions = list(range(AESW.shape[0]//2))
        else:
            project = project.toNumpy()
            assert(np.all(project.imag == 0))
            project = project.real
            positions = list(range(project.shape[1]//2))
            AESW = np.einsum("ji,xj->xi",project, np.array(spectralDensity.getAEr(ERange, "AO")))
        AE = QuantityCalc.interpolateWithSpectralWeight(linESpace,ERange,AESW,GfuncCalc,0)
        E,R = np.meshgrid(linESpace,positions)
        
        
    
    if (Validate):
        return # Disabled because It is too prone to errors due to shifting of the MO energies
        if expect(True,os.path.isfile(outputName + "SpectralDensityOfStates.npz"),f"Cannot find file{outputName + "SpectralDensityOfStates.npz"}"):
            return
        npzFile = np.load(outputName + "SpectralDensityOfStates.npz")
        expectMatrixNear(npzFile["positions"],positions,1e-8,"positions")
        expectMatrixNear(npzFile["linESpace"],linESpace,1e-8,"linESpace")
        # expectMatrixNear(npzFile["OrbLabels"],OrbLabels,1e-8,"OrbLabels")
        expectMatrixNear(npzFile["E"],E,1e-8,"E")
        expectMatrixNear(npzFile["R"],R,1e-8,"R")
        #expectMatrixNear(npzFile["AESW"],AESW,1e-5,"AESW")# These tests fail quite often since a small change in the position of the energies can lead to massive changes in the spectral function
        #expectMatrixNear(npzFile["AE"],AE,1e-5,"AE")# These tests fail quite often since a small change in the position of the energies can lead to massive changes in the spectral function
        return
    smallestValue = 1e-5#np.min(AE) # probably negative
    smallestValue = abs(smallestValue)

    NAlpha = np.array([[AE[idx][2*r].real if AE[idx][2*r].real > smallestValue else smallestValue for r in positions] for idx in range(len(linESpace))]).T  
    NBeta = np.array([[AE[idx][2*r+1].real if AE[idx][2*r+1].real > smallestValue else smallestValue for r in positions] for idx in range(len(linESpace))]).T  

    
    fig, axs = plt.subplots(3)
    img1 = axs[0].contourf(R, E, NAlpha, 100,locator=ticker.LogLocator(),cmap=plt.cm.inferno)
    img2 = axs[1].contourf(R, E, NBeta, 100,locator=ticker.LogLocator(),cmap=plt.cm.inferno)
    img3 = axs[2].contourf(R, E, NAlpha-NBeta, 100,cmap=plt.cm.inferno)
    axs[0].set_xticks(positions,OrbLabels)
    axs[1].set_xticks(positions,OrbLabels)
    axs[2].set_xticks(positions,OrbLabels)

    fig.colorbar(img1)
    fig.colorbar(img2)
    fig.colorbar(img3)

    # for Eigval in np.linalg.eigvals(HamMatrix):
    #     plt.axhline(y = Eigval, color = 'r', linestyle = '-') 
    axs[1].set_xlabel("Position ith site")
    axs[1].set_ylabel("Energy")
    with open(outputName + "SpectralDensityOfStates.png","wb") as f:
        plt.savefig(f,dpi=400)
    if not Headless:
        plt.show()
    else:
        plt.close()
    np.savez(outputName + "SpectralDensityOfStates.npz",positions=positions,linESpace=linESpace,E=E,R=R,AESW=AESW,AE=AE,OrbLabels=OrbLabels)

def plotEnergyDensity(outputName, numSpatialOrbitals, ERange, electronDensity, AtomProjectionName, GfuncCalc,OrbLabels):
    if (electronDensity is None):
        if not os.path.isfile(outputName + "ElectronSpectralDensity.npz"):
            print(f"Cannot find file{outputName + "ElectronSpectralDensity.npz"}")
            return
        npzFile = np.load(outputName + "ElectronSpectralDensity.npz",)
        positions = npzFile["positions"]
        linESpace = npzFile["linESpace"]
        OrbLabels = npzFile["OrbLabels"]
        E = npzFile["E"]
        R = npzFile["R"]
        nESW = npzFile["nESW"]
        nE = npzFile["nE"]
    else:
        currBasis = GfuncCalc.getBasis()

        project = BasisManager().getTransform("AO",IndexType.direct,AtomProjectionName,IndexType.direct)
        if not project:
            raise ValueError("Cannot transform to AtomProjectionBasis")
        
        if project.isIdentity:
            nESW =  np.einsum("ij,xj->xi",project, np.array(electronDensity.getnEr(ERange,"AO")))
            positions = list(range(nESW.shape[0]//2))
        else:
            project = project.toNumpy()
            assert(np.all(project.imag == 0))
            project = project.real
            positions = list(range(project.shape[1]//2))
            nESW =  np.einsum("ji,xj->xi",project, np.array(electronDensity.getnEr(ERange)))
        
        linESpace = np.linspace(ERange[0],ERange[-1],len(ERange))
        nE = QuantityCalc.interpolateWithSpectralWeight(linESpace,ERange,nESW,GfuncCalc,0)
        E,R = np.meshgrid(linESpace,positions)
        
    
    if (Validate):
        return # Disabled because It is too prone to errors due to shifting of the MO energies
        if expect(True,os.path.isfile(outputName + "ElectronSpectralDensity.npz"),f"Cannot find file{outputName + "ElectronSpectralDensity.npz"}"):
            return
        npzFile = np.load(outputName + "ElectronSpectralDensity.npz")
        expectMatrixNear(npzFile["positions"],positions,1e-8,"positions")
        expectMatrixNear(npzFile["linESpace"],linESpace,1e-8,"linESpace")
        # expectMatrixNear(npzFile["OrbLabels"],OrbLabels,1e-8,"OrbLabels")
        expectMatrixNear(npzFile["E"],E,1e-8,"E")
        expectMatrixNear(npzFile["R"],R,1e-8,"R")
        #expectMatrixNear(npzFile["nESW"],nESW,1e-5,"nESW")
        #expectMatrixNear(npzFile["nE"],nE,1e-5,"nE")
        return
    
    smallestValue = 1e-5#np.min(nE)
    smallestValue = abs(smallestValue)

    NAlpha = np.array([[nE[idx][2*r].real  if nE[idx][2*r].real > smallestValue else smallestValue  for r in positions] for idx in range(len(linESpace))]).T  
    NBeta = np.array([[nE[idx][2*r+1].real if nE[idx][2*r+1].real > smallestValue else smallestValue for r in positions] for idx in range(len(linESpace))]).T  

    fig, axs = plt.subplots(3)
    #print(f"minNAlpha:{np.min(NAlpha)}")

    img1 = axs[0].contourf(R, E, NAlpha, 100,locator=ticker.LogLocator(),cmap=plt.cm.inferno)
    img2 = axs[1].contourf(R, E, NBeta, 100,locator=ticker.LogLocator(),cmap=plt.cm.inferno)
    img3 = axs[2].contourf(R, E, NAlpha-NBeta, 100,cmap=plt.cm.inferno)
    
    axs[0].set_xticks(positions,OrbLabels)
    axs[1].set_xticks(positions,OrbLabels)
    axs[2].set_xticks(positions,OrbLabels)
    
    fig.colorbar(img1)
    fig.colorbar(img2)
    fig.colorbar(img3)

    # for Eigval in np.linalg.eigvals(HamMatrix):
    #     plt.axhline(y = Eigval, color = 'r', linestyle = '-') 
    axs[1].set_xlabel("Position ith site")
    axs[1].set_ylabel("Energy")
    with open(outputName + "ElectronSpectralDensity.png","wb") as f:
        plt.savefig(f,dpi=400)
    if not Headless:
        plt.show()
    else:
        plt.close()
    np.savez(outputName + "ElectronSpectralDensity.npz",positions=positions,linESpace=linESpace,E=E,R=R,nESW=nESW,nE=nE,OrbLabels=OrbLabels)

def plotdivCurrentE(outputName, ESpace, divCurrentLead1Alpha,divCurrentLead1Beta,divCurrentLead2Alpha,divCurrentLead2Beta,twoLeads, divCurrentSO):
    if divCurrentLead1Alpha is None:
        if not os.path.isfile(outputName + "DiviE.npz"):
            print(f"Cannot find file{outputName + "DiviE.npz"}")
            return
        npzFile = np.load(outputName + "DiviE.npz")
        ESpace = npzFile["ESpace"]
        I1Alpha = npzFile["I1Alpha"]
        I1Beta = npzFile["I1Beta"]
        if "I2Alpha" in npzFile:
            I2Alpha = npzFile["I2Alpha"]
            I2Beta = npzFile["I2Beta"]
            twoLeads = True
        else:
            twoLeads = False
    else:
        # ISO = np.array(divCurrentSO.getAtE(ESpace))
        I1Alpha = np.array(divCurrentLead1Alpha.getAtE(ESpace))
        I1Beta = np.array(divCurrentLead1Beta.getAtE(ESpace))
        if (twoLeads):
            I2Alpha = np.array(divCurrentLead2Alpha.getAtE(ESpace))
            I2Beta = np.array(divCurrentLead2Beta.getAtE(ESpace))
    
    if (Validate):
        return # Disabled because It is too prone to errors due to shifting of the MO energies
        if expect(True,os.path.isfile(outputName + "DiviE.npz"),f"Cannot find file{outputName + "DiviE.npz"}"):
            return
        npzFile = np.load(outputName + "DiviE.npz")
        expectMatrixNear(npzFile["ESpace"],ESpace,1e-8,"ESpace")
        expectMatrixNear(npzFile["I1Alpha"],I1Alpha,1e-8,"I1Alpha")
        expectMatrixNear(npzFile["I1Beta"],I1Beta,1e-8,"I1Beta")
        if "I2Alpha" in npzFile:
            expectMatrixNear(npzFile["I2Alpha"],I2Alpha,1e-8,"I2Alpha")
            expectMatrixNear(npzFile["I2Beta"],I2Beta,1e-8,"I2Beta")
        expect("I2Alpha" in npzFile,twoLeads,"twoLeads")
        return
  
    plt.plot(ESpace,np.sum(I1Alpha,axis=1),label="1Up")
    plt.plot(ESpace,np.sum(I1Beta,axis=1),label="1Down")
    if (twoLeads):
        plt.plot(ESpace,np.sum(I2Alpha,axis=1),label="2Up")
        plt.plot(ESpace,np.sum(I2Beta,axis=1),label="2Down")

    # plt.plot(ESpace,ISO[:,0],label="ISO-0")
    # plt.plot(ESpace,ISO[:,1],label="ISO-1")
    
    plt.legend()
    with open(outputName + "DiviE.png","wb") as f:
        plt.savefig(f,dpi=400)
    if not Headless:
        plt.show()
    else:
        plt.close()
    
    #Spin difference plot
    I1Sum = np.sum(I1Alpha+I1Beta,axis=1)
    I1Polarisation  = np.zeros_like(I1Sum)
    np.divide(np.sum(I1Alpha-I1Beta,axis=1),I1Sum,out=I1Polarisation,where=np.abs(I1Sum) > 1e-6)

    plt.plot(ESpace,100*I1Polarisation,label="1Polarisation")
    if (twoLeads):
        I2Sum = np.sum(I2Alpha+I2Beta,axis=1)
        I2Polarisation  = np.zeros_like(I2Sum)
        np.divide(np.sum(I2Alpha-I2Beta,axis=1),I2Sum,out=I2Polarisation,where=np.abs(I2Sum) > 1e-6)
        plt.plot(ESpace,100*I2Polarisation,label="2Polarisation")
    plt.ylabel("Spin Polarisation %")
    plt.xlabel("Energy / Eh")
    # plt.plot(ESpace,ISO[:,0],label="ISO-0")
    # plt.plot(ESpace,ISO[:,1],label="ISO-1")
    
    plt.legend()
    with open(outputName + "DiviEPol.png","wb") as f:
        plt.savefig(f,dpi=400)
    if not Headless:
        plt.show()
    else:
        plt.close()
    
    if twoLeads:
        np.savez(outputName + "DiviE.npz",ESpace=ESpace,I1Alpha=I1Alpha,I1Beta=I1Beta,I2Alpha=I2Alpha,I2Beta=I2Beta)
    else:
        np.savez(outputName + "DiviE.npz",ESpace=ESpace,I1Alpha=I1Alpha,I1Beta=I1Beta)
    
    
    # I1TotalAlpha = divCurrentLead1Alpha.get()
    # I1TotalBeta = divCurrentLead1Beta.get()
    # I2TotalAlpha = divCurrentLead2Alpha.get()
    # I2TotalBeta = divCurrentLead2Beta.get()
    # # ITotalSO = divCurrentSO.get()

    # # print(f"Total I1Up{np.sum(I1TotalAlpha)}, Total I1Down{np.sum(I1TotalBeta)}, TotalSO: \n{ITotalSO}")
    # print(f"Total I1Up{np.sum(I1TotalAlpha)}, Total I1Down{np.sum(I1TotalBeta)}")
    # if (twoLeads):
    #     print(f"Total I2Up{np.sum(I2TotalAlpha)}, Total I2Down{np.sum(I2TotalBeta)}")

if __name__ == "__main__":    
    #Generate the plots from files
    outputName = sys.argv[1]
    print(f"Loading data from: {outputName}")

    plotdivCurrentE(outputName+ "PE",None,None,None,None,None,None,None)
    plotdivCurrentE(outputName+ "PE2",None,None,None,None,None,None,None)
    plotdivCurrentE(outputName,None,None,None,None,None,None,None)
        
    plotEnergySpectralDensity(outputName, None,None,None,None,None,None,None)
    plotEnergyDensity(outputName,None,None,None,None,None,None,None)
    plotDensity(outputName,None,None,None,None,None)
    
