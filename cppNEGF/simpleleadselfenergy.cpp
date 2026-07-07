#include "simpleleadselfenergy.h"
#include "Eigen/Eigenvalues"
#include "complexmatrixbuffer.h"
#include "configfileloader.h"
#include "integrator.h"

template<class SelfEnergyMatrixType>
void SimpleLeadSelfEnergy<SelfEnergyMatrixType>::getGamma(numType E, SelfEnergyMatrixType& dest)
{
    getSigmaR(E,dest);
    SelfEnergyMatrixType temp;
    getSigmaA(E,temp);
    dest -= temp;
    dest *= iu;
    dest = zeroOrMatrix(E,dest);
}

template<class SelfEnergyMatrixType>
void SimpleLeadSelfEnergy<SelfEnergyMatrixType>::clearCache()
{
    if (!m_params.adjustMuToElectrons)
        return; // nothing to do
    releaseAssert(m_GFunc != nullptr, "m_GFunc is Nullptr, You probabily forgot to set it");

    complexType HomoComplex;
    numType Homo = 0;
    numType Lumo = 0;
    int numberOfElectronsLeft = m_params.numberOfElectrons;

    if (m_GFunc->getComputationMethod() == GreensFunction::GrComputationType::AnalyticSOPRegions)
    {
        std::vector<complexType> poles;
        auto regions = m_GFunc->getAnalyticRegions();
        for (size_t regionIndex = 0; regionIndex < regions.size()-1; regionIndex++)
        {//loop up the regions and find a place where we have satisfied the correct number of electrons and the divergences are accurate

            m_GFunc->getEstimateForDivergencesOfGr(m_GFunc->getRegionRepresentativeEnergy(regionIndex),poles);

            std::sort(poles.begin(),poles.end(),[](const complexType& a, const complexType& b){return a.real() < b.real();});

            releaseAssert(m_params.numberOfElectrons <= poles.size(), "Too many electrons given in Simple Lead self energy");
            releaseAssert(m_params.numberOfElectrons > 0, "0 electrons given in Simple Lead self energy");

            for (auto& p : poles)
            {
                if (p.real() > regions[regionIndex+1])
                    break; //outside region so break
                if (p.real() < regions[regionIndex])
                    continue; // below region so doesnt count

                numberOfElectronsLeft -=1;

                if (numberOfElectronsLeft == 0)
                {
                    Homo = p.real(); // in region and the last electron
                    HomoComplex = p;
                }
                else if (numberOfElectronsLeft == -1)
                {
                    Lumo = p.real(); // the lastest electron
                    break; // no more electrons to deal with
                }
            }
        }
        if (numberOfElectronsLeft > 0)
            releaseAssert(false, "Couldnt place all the electrons?");
        if (numberOfElectronsLeft == 0)
        {
            //There is no LUMO
            Lumo = Homo + 4*abs(HomoComplex.imag())+2;
        }
    }
    else if (m_GFunc->getComputationMethod() == GreensFunction::GrComputationType::AnalyticSOP)
    {
        std::vector<complexType> poles;

        m_GFunc->getEstimateForDivergencesOfGr(0,poles);

        std::sort(poles.begin(),poles.end(),[](const complexType& a, const complexType& b){return a.real() < b.real();});

        releaseAssert(m_params.numberOfElectrons <= poles.size(), "Too many electrons given in Simple Lead self energy");
        releaseAssert(m_params.numberOfElectrons > 0, "0 electrons given in Simple Lead self energy");

        for (auto& p : poles)
        {
            numberOfElectronsLeft -=1;

            if (numberOfElectronsLeft == 0)
                Homo = p.real(); // in region and the last electron
            else if (numberOfElectronsLeft == -1)
            {
                Lumo = p.real(); // the lastest electron
                break; // no more electrons to deal with
            }
        }

        if (numberOfElectronsLeft > 0)
            releaseAssert(false, "Couldnt place all the electrons?");
        if (numberOfElectronsLeft == 0)
        {
            //There is no LUMO
            Lumo = Homo + 4*abs(poles[m_params.numberOfElectrons-1].imag())+2;
        }
    }
    else
    {
        ComplexSelfAdjointMatrix Gn;
        m_GFunc->getGnIntegral(0,Gn,false);
        numType foundNumElec = Gn.trace()/(2*M_PI);
        if (std::abs(foundNumElec-m_params.numberOfElectrons) > 0.1 )
            m_params.mu += -(foundNumElec-m_params.numberOfElectrons)*0.001;
        logger().log("Setting mu to", m_params.mu);
        return;
    }


    logger().log("Homo Energy", Homo);
    logger().log("Lumo Energy", Lumo);
    m_params.mu = std::min((Homo + Lumo)/2,m_params.muMax);
    logger().log("Setting mu to", m_params.mu);
}

template<class SelfEnergyMatrixType>
void SimpleLeadSelfEnergy<SelfEnergyMatrixType>::setBasis(const std::string &bas)
{
    const std::string& SEBasis = BasisManager::removeSpinTags(m_params.couplingPoints.getBasisS());
    //Construct in the natural SE basis.
    m_workingBasis = {SEBasis,SEBasis};
    computeSigmaR(m_params.bottomOfBand+1,m_SigmaR);
    computeSigmaA(m_params.bottomOfBand+1,m_SigmaA);
    m_ZeroMatrix.setZero(m_params.MatrixDimension,m_params.MatrixDimension,m_workingBasis,m_params.sym);

    //Transform if needed
    if (m_workingBasis[0] == bas)
        return;
    m_workingBasis = {bas,bas};

    m_SigmaR.toBasisC(m_SigmaR,m_workingBasis);
    m_SigmaA.toBasisC(m_SigmaA,m_workingBasis);
    m_ZeroMatrix.setZero(m_params.MatrixDimension,m_params.MatrixDimension,m_workingBasis,m_params.sym);



}

template<class SelfEnergyMatrixType>
numType SimpleLeadSelfEnergy<SelfEnergyMatrixType>::fermiFunction(numType E)
{
    if ((E-m_params.mu)/(kb*T) > 30)
        return 0;
    else
        return 1.0/(exp((E-m_params.mu)/(kb*T)) + 1);


}

template<class SelfEnergyMatrixType>
SimpleLeadSelfEnergy<SelfEnergyMatrixType>::SimpleLeadSelfEnergy(Parameters params)
{
    m_params = params;
    if (m_params.hasBottom)
    {
        this->m_type = selfEnergyType::ConstantPiecewise;
        m_ConstantRegions = {negInf,m_params.bottomOfBand,posInf};
    }
    else
    {
        this->m_type = selfEnergyType::Constant;
        m_ConstantRegions = {negInf,posInf};
    }

    if (m_params.adjustMuToElectrons)
        logger().log("Initial mu is",m_params.mu);
    setBasis(BasisManager::removeSpinTags(m_params.couplingPoints.getBasisS()));

}

template<class SelfEnergyMatrixType>
void SimpleLeadSelfEnergy<SelfEnergyMatrixType>::getSigmaIn(numType E, SelfEnergyMatrixType& dest)
{
    getGamma(E,dest);
    dest *= fermiFunction(E);
}

template<class SelfEnergyMatrixType>
void SimpleLeadSelfEnergy<SelfEnergyMatrixType>::getSigmaOut(numType E, SelfEnergyMatrixType& dest)
{
    getGamma(E,dest);
    dest*=(1-fermiFunction(E));
}

template<class SelfEnergyMatrixType>
void SimpleLeadSelfEnergy<SelfEnergyMatrixType>::getSigmaR(SelfEnergyMatrixType &dest)
{
    if (this->m_type == selfEnergyType::ConstantPiecewise)
        releaseAssert(false,"ConstantPieceWise does not have Energy Independent Self energy");
    else
        return getSigmaR(0,dest);}

template<class SelfEnergyMatrixType>
void SimpleLeadSelfEnergy<SelfEnergyMatrixType>::getSigmaA(SelfEnergyMatrixType &dest)
{
    if (this->m_type == selfEnergyType::ConstantPiecewise)
        releaseAssert(false,"ConstantPieceWise does not have Energy Independent Self energy");
    else
        return getSigmaA(0,dest);
}


template<class SelfEnergyMatrixType>
void SimpleLeadSelfEnergy<SelfEnergyMatrixType>::computeSigmaR(numType E, SelfEnergyMatrixType& out)
{
    complexType val = -m_params.alpha1*iu*m_params.t/2.;
    int matDim = m_params.couplingPoints.rows();
    BasisManager::possibleTags tag = BasisManager::possibleTags::NullBasis;
    std::string basis = BasisManager::removeSpinTags(m_params.couplingPoints.getBasisS(),&tag);
    releaseAssert(tag != BasisManager::possibleTags::NullBasis,"Coupling Points matrix not in a spin blocked Basis. Make sure this is a spatial Matrix and appropriately tagged");

    ComplexDirectMatrix dest;
    dest.setZero(2*matDim,2*matDim,basis,m_params.sym);

    HermitianMatrix2cd spinMat = pauliMatrices::sigmaX * m_params.magnetisationDir[0]
                               + pauliMatrices::sigmaY * m_params.magnetisationDir[1]
                               + pauliMatrices::sigmaZ * m_params.magnetisationDir[2];

    // Eigen::SelfAdjointEigenSolver<Eigen::Matrix2cd> es(spinMat,Eigen::DecompositionOptions::ComputeEigenvectors);
    Matrix<complexType,2,2,enums::MatrixProperties::Unitary,enums::IndexType::dual,enums::IndexType::eigenValue> EigenVectors;
    Matrix<numType,2,1,enums::MatrixProperties::None,enums::IndexType::eigenValue,enums::IndexType::invalid> eigenValues;
    releaseAssert(spinMat.getEigenValuesAndVectors(eigenValues,EigenVectors),"Diagonalising spin matrix failed");
    int myIndex = 0;
    if (eigenValues(0) == 0)
    {// no clear direction so pick the basis directions to avoid problems
        spinMat.setZero();
        if (m_params.upSpin)
            spinMat(0,0) = 1;
        else
            spinMat(1,1) = 1;

    }
    else
    {//Clear direction \Gamma = \Gamma + \Gamma \sigma\cdot \v{m}
        if (eigenValues(0) > 0)
            myIndex = 0;
        else
            myIndex = 1;

        if (!m_params.upSpin)
            myIndex = 1-myIndex;
        spinMat = HermitianMatrix2cd(EigenVectors.col(myIndex)*EigenVectors.col(myIndex).adjoint(),"",enums::SpinSymmetry::SpinMatrix); // Spin degrees of freedom are orthogonal for now!
        spinMat *= (eigenValues(myIndex)+1);
    }

    for (long i = 0; i < m_params.couplingPoints.rows(); i++)
    {
        for (long j = 0; j < m_params.couplingPoints.cols(); j++)
        {// idx in spatial basis
            complexType coeff = m_params.couplingPoints.coeff(i,j);
            if (coeff == complexType(0))
                continue;
            dest.coeffRef(2*i,2*j) = val*spinMat(0,0)*coeff;
            dest.coeffRef(2*i,2*j+1) = val*spinMat(0,1)*coeff;
            dest.coeffRef(2*i+1,2*j) = val*spinMat(1,0)*coeff;
            dest.coeffRef(2*i+1,2*j+1) = val*spinMat(1,1)*coeff;

        }
    }
    //This is now in the basis manager
    // if (m_params.hasCProj)
    // {
    //     dest = (m_params.CInvProjDirecttoDirect * dest * m_params.CProj).sparseView();
    // }


    if constexpr(SelfEnergyMatrixType::isSparse)
    {
        ComplexMatrix temp;
        dest.toBasisC(temp,m_workingBasis);
        out = temp.sparseView();
    }
    else
    {
        dest.toBasisC(out,m_workingBasis);
    }


}

template<class SelfEnergyMatrixType>
void SimpleLeadSelfEnergy<SelfEnergyMatrixType>::computeSigmaA(numType E, SelfEnergyMatrixType& dest)
{
    SelfEnergyMatrixType sr;
    getSigmaR(E,sr);
    dest = sr.adjoint();
}

template<class SelfEnergyMatrixType>
void SimpleLeadSelfEnergy<SelfEnergyMatrixType>::getSigmaK(numType E, SelfEnergyMatrixType &dest)
{
    getGamma(E,dest);
    dest *= -iu*(1-2*fermiFunction(E));
}

// template<class SelfEnergyMatrixType>
// void SimpleLeadSelfEnergy<SelfEnergyMatrixType>::save(std::string filename) const
// {
//     logger logFile = logger(filename,true);
//     logFile.log("SimpleLeadSelfEnergy<SelfEnergyMatrixType>::Parameters\n{");
//     logFile.log("alpha1",m_params.alpha1);
//     logFile.log("t",m_params.t);
//     logFile.log("mu",m_params.mu);
//     logFile.log("couplingPoints",m_params.couplingPoints);
//     logFile.log("MatrixDimension",m_params.MatrixDimension);
//     logFile.log("Name",this->m_prettyName);
//     logFile.log("}");
// }

// template<class SelfEnergyMatrixType>
// void SimpleLeadSelfEnergy<SelfEnergyMatrixType>::load(FILE *file)
// {
//     while (true)
//     {
//         char objectName[30];
//         int ret = fscanf(file," %29[^{}:\n ] : ",objectName);
//         if (ret <= 0)
//             break;
//         std::string objectName_s(objectName);
//         if (objectName_s == "alpha1")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.alpha1);
//         }
//         else if (objectName_s == "t")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.t);
//         }
//         else if (objectName_s == "mu")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.mu);
//         }
//         else if (objectName_s == "couplingPoints")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.couplingPoints);
//         }
//         else if (objectName_s == "MatrixDimension")
//         {
//             uint32_t dummy;
//             ConfigFileLoader::loadParameter(file,dummy);
//         }
//         else if (objectName_s == "Name")
//         {
//             ConfigFileLoader::loadParameter(file,this->m_prettyName);
//         }
//         else
//         {
//             logger().log("Unknown SimpleLeadSelfEnergy parameter: ",objectName_s);
//         }
//     }
// }

template class SimpleLeadSelfEnergy<ComplexSparseMatrix>;
template class SimpleLeadSelfEnergy<ComplexMatrix>;

