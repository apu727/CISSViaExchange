#include "matrixselfenergy.h"



template<class SelfEnergyMatrixType,bool b>
MatrixSelfEnergy<SelfEnergyMatrixType,b>::MatrixSelfEnergy(const SparseMatrixType& matrix)
{
    this->m_type = selfEnergyType::Constant;
    m_MatrixProvided = matrix;
    const auto& bas = m_MatrixProvided.getBasis();
    if (b && !(BasisManager::isProjection(bas[0]) || BasisManager::isProjection(bas[1])))
    {
        ComplexMatrix temp = static_cast<ComplexMatrix>(m_MatrixProvided).adjoint();
        ComplexMatrix H;

        H = m_MatrixProvided - temp;
        if (H.norm()/(H.cols()*H.rows()) > 1e-10)
            logger().log("Matrix is not Self-Adjoint, MatrixSelfEnergy, error:", (numType)(H.norm()/(H.cols()*H.rows())));
    }
    setBasis(m_MatrixProvided.getBasis());//If this is wrong then it is reset later. But we may be the Hamiltonian and so need to have a basis
}

template<class SelfEnergyMatrixType,bool b>
MatrixSelfEnergy<SelfEnergyMatrixType,b>::MatrixSelfEnergy(const DenseMatrixType &mat)
{
    SparseMatrixType temp;
    mat.sparseView().toBasisC(temp,mat.getBasis());   
    *this = MatrixSelfEnergy(temp);
}

template<class SelfEnergyMatrixType,bool b>
MatrixSelfEnergy<SelfEnergyMatrixType,b>::MatrixSelfEnergy(SecondQuantisedHamiltonian& Ham)
{
    this->m_type = selfEnergyType::Constant;
    m_MatrixProvided = static_cast<SparseMatrixType>(Ham.getHamMatrix());
    const auto& bas = m_MatrixProvided.getBasis();
    if (b && !(BasisManager::isProjection(bas[0]) || BasisManager::isProjection(bas[1])))
    {
        ComplexMatrix temp = static_cast<ComplexMatrix>(m_MatrixProvided).adjoint();
        ComplexMatrix H;

        H = m_MatrixProvided - temp;
        if (H.norm()/(H.cols()*H.rows()) > 1e-10)
            logger().log("Matrix is not Self-Adjoint, MatrixSelfEnergy, error:", (numType)(H.norm()/(H.cols()*H.rows())));
    }
    setBasis(m_MatrixProvided.getBasis());//If this is wrong then it is reset later. But we may be the Hamiltonian and so need to have a basis
}

template<class SelfEnergyMatrixType, bool selfAdjoint>
void MatrixSelfEnergy<SelfEnergyMatrixType, selfAdjoint>::getSigmaK(numType E, SelfEnergyMatrixType &dest)
{
    if constexpr(selfAdjoint)
        dest.setZero(m_SigmaR.rows(),m_SigmaR.cols(),m_workingBasis,enums::SpinSymmetry::RHF);
    else
    {
        releaseAssert(false,"MatrixSelfEnergy,false does not support SigmaK for non selfadjoint matrices");
    }
}

template<class SelfEnergyMatrixType,bool b>
void MatrixSelfEnergy<SelfEnergyMatrixType,b>::setBasis(const std::string &bas)
{
    setBasis({bas,bas});
}

template<class SelfEnergyMatrixType,bool b>
void MatrixSelfEnergy<SelfEnergyMatrixType,b>::setBasis(const indexBasisT &bas)
{
    if (bas == m_workingBasis)
        return;
    m_workingBasis = bas;
    m_MatrixProvided.toBasisC(m_SigmaRS,m_workingBasis);
    m_SigmaR = static_cast<SelfEnergyMatrixType>(m_SigmaRS);
    if (b && !(BasisManager::isProjection(m_workingBasis[0]) || BasisManager::isProjection(m_workingBasis[1])))
    {
        m_SigmaR.makeSelfAdjoint(false);
        m_SigmaA = m_SigmaR;
        m_SigmaAS = m_SigmaRS;
    }
    else
    {
        m_MatrixProvided.adjoint().toBasisC(m_SigmaAS,m_workingBasis);
        m_SigmaA = static_cast<SelfEnergyMatrixType>(m_SigmaAS);
    }
}

//Self energies must use the no params version so they all have the same base class
template class MatrixSelfEnergy<ComplexMatrix,false>;
template class MatrixSelfEnergy<ComplexSparseMatrix,false>;
template class MatrixSelfEnergy<ComplexMatrix,true>;
template class MatrixSelfEnergy<ComplexSparseMatrix,true>;
