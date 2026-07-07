#include "explicitleadselfenergy.h"
#include "linalg.h"


template<class SelfEnergyMatrixType>
numType ExplicitLeadSelfEnergy<SelfEnergyMatrixType>::KFromE(numType E)
{
    if (E > m_params.bottomOfBand && E < m_params.bottomOfBand + 4*m_params.t)
    {
        //E = BottomOfBand + 2*t (1 - cos(k*a))
        return acos(1-((E - m_params.bottomOfBand)/(2*m_params.t)))/m_params.a;
    }
    return 0;
}

template<class SelfEnergyMatrixType>
void ExplicitLeadSelfEnergy<SelfEnergyMatrixType>::getGamma(numType E, SelfEnergyMatrixType& dest)
{
    getSigmaR(E,dest);
    SelfEnergyMatrixType temp;
    getSigmaA(E,temp);
    dest -= temp;
    dest *= iu;
}



template<class SelfEnergyMatrixType>
numType ExplicitLeadSelfEnergy<SelfEnergyMatrixType>::fermiFunction(numType E)
{
    if (E-m_params.mu > 30*kb*T)
        return 0;
    else
        return 1.0/(exp((E-m_params.mu)/(kb*T)) + 1);

}

template<class SelfEnergyMatrixType>
ExplicitLeadSelfEnergy<SelfEnergyMatrixType>::ExplicitLeadSelfEnergy(Parameters params)
{
    m_params = params;
    // GreensFunction::makeTimeReversalMatrix(m_timeReversalMatrix,m_params.MatrixDimension);
}

template<class SelfEnergyMatrixType>
void ExplicitLeadSelfEnergy<SelfEnergyMatrixType>::getSigmaIn(numType E, SelfEnergyMatrixType& dest)
{
    getGamma(E,dest);
    dest *= fermiFunction(E);
}

template<class SelfEnergyMatrixType>
void ExplicitLeadSelfEnergy<SelfEnergyMatrixType>::getSigmaOut(numType E, SelfEnergyMatrixType& dest)
{
    getGamma(E,dest);
    dest *= 1-fermiFunction(E);
}

template<>
void ExplicitLeadSelfEnergy<ComplexMatrix>::getSigmaR(complexType E, ComplexMatrix& out)
{
    if (E.imag() != 0)
        logger().log("No analytic continuation of this self energy");
    numType k = KFromE(real(E));
    complexType val = -m_params.t*std::exp(iu*k*m_params.a)*m_params.alpha1*m_params.alpha1;
    //dest.resize(m_params.MatrixDimension,m_params.MatrixDimension);
    ComplexMatrix dest;
    dest.setZero(m_params.MatrixDimension,m_params.MatrixDimension,m_params.basis,m_params.sym);

    for (auto idx : m_params.couplingPoints)
    {
        dest(idx,idx) = val;
    }
    dest.toBasisC(out,m_workingBasis);
}

template<>
void ExplicitLeadSelfEnergy<ComplexSparseMatrix>::getSigmaR(complexType E, ComplexSparseMatrix& out)
{
    if (E.imag() != 0)
        logger().log("No analytic continuation of this self energy");
    numType k = KFromE(real(E));
    complexType val = -m_params.t*std::exp(iu*k*m_params.a)*m_params.alpha1*m_params.alpha1;
    //dest.resize(m_params.MatrixDimension,m_params.MatrixDimension);
    ComplexSparseMatrix dest;
    dest.resize(m_params.MatrixDimension,m_params.MatrixDimension,m_params.basis,m_params.sym);

    for (auto idx : m_params.couplingPoints)
    {
        dest.coeffRef(idx,idx) = val;
    }
    dest.toBasisC(out,m_workingBasis);
}

template<class SelfEnergyMatrixType>
void ExplicitLeadSelfEnergy<SelfEnergyMatrixType>::getSigmaA(complexType E, SelfEnergyMatrixType& dest)
{
    SelfEnergyMatrixType sr;
    getSigmaR(std::conj(E),sr);
    dest = sr.adjoint();
}

template<class SelfEnergyMatrixType>
void ExplicitLeadSelfEnergy<SelfEnergyMatrixType>::getSigmaK(numType E, SelfEnergyMatrixType &dest)
{
    getGamma(E,dest);
    dest *= -iu*(1-2*fermiFunction(E));
}

// template<class SelfEnergyMatrixType>
// void ExplicitLeadSelfEnergy<SelfEnergyMatrixType>::save(std::string filename) const
// {
//     logger logFile = logger(filename,true);
//     logFile.log("ExplicitLeadSelfEnergy<SelfEnergyMatrixType>::Parameters\n{");
//     logFile.log("a",m_params.a);
//     logFile.log("alpha1",m_params.alpha1);
//     logFile.log("t",m_params.t);
//     logFile.log("bottomOfBand",m_params.bottomOfBand);
//     logFile.log("mu",m_params.mu);
//     logFile.log("CouplingPoints",m_params.couplingPoints);
//     logFile.log("MatrixDimension",m_params.MatrixDimension);
//     logFile.log("Name",this->m_prettyName);
//     logFile.log("}");
// }

// template<class SelfEnergyMatrixType>
// void ExplicitLeadSelfEnergy<SelfEnergyMatrixType>::load(FILE *file)
// {
//     while (true)
//     {
//         char objectName[30];
//         int ret = fscanf(file," %29[^{}:\n ] : ",objectName);
//         if (ret <= 0)
//             break;
//         std::string objectName_s(objectName);
//         if (objectName_s == "a")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.a);
//         }
//         else if (objectName_s == "alpha1")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.alpha1);
//         }
//         else if (objectName_s == "t")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.t);
//         }
//         else if (objectName_s == "bottomOfBand")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.bottomOfBand);
//         }
//         else if (objectName_s == "mu")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.mu);
//         }
//         else if (objectName_s == "CouplingPoints")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.couplingPoints,"%u");
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
//             logger().log("Unknown ExplicitLeadSelfEnergy parameter: ",objectName_s);
//         }
//     }
// }

template class ExplicitLeadSelfEnergy<ComplexMatrix>;
template class ExplicitLeadSelfEnergy<ComplexSparseMatrix>;
