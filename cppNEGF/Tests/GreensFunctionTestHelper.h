#ifndef GREENSFUNCTIONTESTHELPER_H
#define GREENSFUNCTIONTESTHELPER_H
#include "Tests/testutil.hpp"
#include "greensfunction.h"
#include "linalg.h"
#include "matrixselfenergy.h"
#include "selfenergybase.h"
#include "sopgreensfunction.h"
#include <Eigen/LU>
#include <memory>
class GreensFunctionTest
{
public:
    std::shared_ptr<GreensFunction> m_gfunc;
    GreensFunctionTest(){}
    void setup(const ComplexSelfAdjointSparseMatrix& Ham, GreensFunction::integratorType integType = GreensFunction::integratorType::autoDetect)
    {
        auto H = std::make_shared<MatrixSelfEnergy<SelfEnergyMatrix>>(Ham);
        m_gfunc = std::make_shared<GreensFunction>();
        m_gfunc->setHamiltonian(H);
        setRegions({negInf,posInf});
        m_gfunc->m_integratorType = integType;
    }
    void reset()
    {
        m_gfunc = nullptr;
    }
    void setComputationMethod(GreensFunction::GrComputationType comp)
    {
        m_gfunc->m_GrComputationMethod = comp;
    }
    void setEigenValues(std::vector<EigenValueVector> ev)
    {
        if (!m_gfunc->m_SOPGIsBuilt)
            m_gfunc->buildSOP();
        m_gfunc->m_GMM->m_EigenValues = ev;
    }
    void setEigenVectors(std::vector<EigenVectorMatrix> ev)
    {
        if (!m_gfunc->m_SOPGIsBuilt)
            m_gfunc->buildSOP();

        m_gfunc->m_GMM->m_rightEigenVectors = ev;
        m_gfunc->m_GMM->m_InvRightEigenVectors.clear();
        for (auto& e : ev)
            m_gfunc->m_GMM->m_InvRightEigenVectors.push_back(e.inverse());
        m_gfunc->m_GMM->m_GRIsCached = true;
    }
    void setRegions(std::vector<numType> regions)
    {
        m_gfunc->m_analyticRegions = regions;
        if (!m_gfunc->m_SOPGIsBuilt)
            m_gfunc->buildSOP();
        m_gfunc->m_GMM->m_analyticRegions = regions;
    }

    void getEigenVectors(EigenVectorMatrix& rightEigenVectors, InvEigenVectorMatrix & invRightEigenVectors, EigenValueVector& eigenValues)
    {
        expect(2ul,m_gfunc->m_analyticRegions.size(),__PRETTY_FUNCTION__,"GetEigenVectors has regions");
        if (!m_gfunc->m_SOPGIsBuilt)
            m_gfunc->buildSOP();
        rightEigenVectors = m_gfunc->m_GMM->m_rightEigenVectors[0];
        invRightEigenVectors = m_gfunc->m_GMM->m_InvRightEigenVectors[0];
        eigenValues = m_gfunc->m_GMM->m_EigenValues[0];
    }
};

ComplexSelfAdjointSparseMatrix constructPeriodicHamilonian(int numberOfSites, const std::string& basis,numType, numType);
ComplexSelfAdjointSparseMatrix constructPeriodicHamilonian(int numberOfSites, const std::string& basis);
void makeMetric(int numberOfSites, ComplexMatrix& transform, const std::string& basisName, const std::string& newBasisName);

//Need to aligns magnetisation with the Z direction via a rotation. For S^2 = 0 this is always the case. All the tests are S^2 = 0
template<bool (**nearFunc)(const ComplexSelfAdjointMatrix&,const ComplexSelfAdjointMatrix&)>
inline bool densityMatrixNear(const ComplexSelfAdjointMatrix& matA, const ComplexSelfAdjointMatrix& matB)
{
    // Compares if two density matrices are related by a spin rotation, Must be in an orthogonal basis
    // uses the relation that a SU(2) can be written as \exp{i \theta \sigma\cdot \v{n}} = cos(\theta) I + \iu sin(\theta) \sigma\cdot \v{n}
    // top left block
    // This should be doable with the new spinSym parameter
    int ret = 0;
    long spinOrbitals = matA.rows();
    ret += expect(spinOrbitals,matA.cols(),__PRETTY_FUNCTION__,"MatA.rows = matA.cols");
    ret += expect(spinOrbitals,matB.cols(),__PRETTY_FUNCTION__,"MatA.rows = matB.cols");
    ret += expect(spinOrbitals,matB.rows(),__PRETTY_FUNCTION__,"MatA.rows = matA.rows");
    ret += expect(0l,spinOrbitals % 2,__PRETTY_FUNCTION__,"spinOrbitals %2 == 0");
    if (ret)
        return false;

    // Eigen::Matrix<ComplexMatrix,2,2> blockedA;
    // blockedA(0,0) = getBlock(matA,0,0);
    // blockedA(0,1) = getBlock(matA,0,1);
    // blockedA(1,0) = getBlock(matA,1,0);
    // blockedA(1,1) = getBlock(matA,1,1);

    // Eigen::Matrix<ComplexMatrix,2,2> blockedB;
    // blockedB(0,0) = getBlock(matB,0,0);
    // blockedB(0,1) = getBlock(matB,0,1);
    // blockedB(1,0) = getBlock(matB,1,0);
    // blockedB(1,1) = getBlock(matB,1,1);

    //TODO we can know that these are real in compile time maybe
    complexType IComponentA = 0.5*pauliMatrices::sigmaI.cdot(matA);//0.5*(blockedA(0,0).trace() + blockedA(1,1).trace());
    complexType XComponentA = 0.5*pauliMatrices::sigmaX.cdot(matA);//0.5*(blockedA(0,1).trace() + blockedA(1,0).trace());
    complexType YComponentA = 0.5*pauliMatrices::sigmaY.cdot(matA);//0.5*(-iu*blockedA(0,1).trace() + iu*blockedA(1,0).trace());
    complexType ZComponentA = 0.5*pauliMatrices::sigmaZ.cdot(matA);;//0.5*(blockedA(0,0).trace() - blockedA(1,1).trace());

    complexType IComponentB = 0.5*pauliMatrices::sigmaI.cdot(matB);//0.5*(blockedB(0,0).trace() + blockedB(1,1).trace());
    complexType XComponentB = 0.5*pauliMatrices::sigmaX.cdot(matB);//0.5*(blockedB(0,1).trace() + blockedB(1,0).trace());
    complexType YComponentB = 0.5*pauliMatrices::sigmaY.cdot(matB);//0.5*(-iu*blockedB(0,1).trace() + iu*blockedB(1,0).trace());
    complexType ZComponentB = 0.5*pauliMatrices::sigmaZ.cdot(matB);//0.5*(blockedB(0,0).trace() -  blockedB(1,1).trace());

    ret += expectNear(0,IComponentA.imag(),__PRETTY_FUNCTION__,"IComponentA Real");
    ret += expectNear(0,XComponentA.imag(),__PRETTY_FUNCTION__,"XComponentA Real");
    ret += expectNear(0,YComponentA.imag(),__PRETTY_FUNCTION__,"YComponentA Real");
    ret += expectNear(0,ZComponentA.imag(),__PRETTY_FUNCTION__,"ZComponentA Real");

    ret += expectNear(0,IComponentB.imag(),__PRETTY_FUNCTION__,"IComponentB Real");
    ret += expectNear(0,XComponentB.imag(),__PRETTY_FUNCTION__,"XComponentB Real");
    ret += expectNear(0,YComponentB.imag(),__PRETTY_FUNCTION__,"YComponentB Real");
    ret += expectNear(0,ZComponentB.imag(),__PRETTY_FUNCTION__,"ZComponentB Real");
    if (ret)// should not happen for a Hermitian density matrix, Possible in non orthogonal basis?
        return false;

    Eigen::Vector3d nA({XComponentA.real(),YComponentA.real(),ZComponentA.real()});
    numType ThetaA = 0;
    Eigen::Vector3d AxisA;
    AxisA.setZero();
    if (nA.norm() > 1e-16)
    {
        nA.normalize();
        ThetaA = -std::acos(nA(2));
        AxisA = nA.cross(Eigen::Vector3d({0,0,1}));
        AxisA.normalize();
    }
    else
    {
        AxisA = Eigen::Vector3d({0,0,1});
        ThetaA = 0;
    }
    ThetaA /= 2;



    Eigen::Vector3d nB({XComponentB.real(),YComponentB.real(),ZComponentB.real()});
    numType ThetaB = 0;
    Eigen::Vector3d AxisB;
    AxisB.setZero();
    if (nB.norm() > 1e-16)
    {
        nB.normalize();
        ThetaB = -std::acos(nB(2));
        AxisB = nB.cross(Eigen::Vector3d({0,0,1}));
        AxisB.normalize();
    }
    else
    {
        AxisB = Eigen::Vector3d({0,0,1});
        ThetaB = 0;
    }
    ThetaB /= 2;

    //Construct inverse rotations
    Eigen::Vector4cd rotationA;
    rotationA(0) = std::cos(ThetaA);
    rotationA(1) = iu*std::sin(ThetaA)*AxisA(0);
    rotationA(2) = iu*std::sin(ThetaA)*AxisA(1);
    rotationA(3) = iu*std::sin(ThetaA)*AxisA(2);


    UnitaryMatrix2cd rotationMatrixA = static_cast<UnitaryMatrix2cd>(HermitianMatrix2cd::Identity(2,2,"",enums::SpinSymmetry::SpinMatrix) * rotationA(0) + pauliMatrices::sigmaX * rotationA(1) + pauliMatrices::sigmaY * rotationA(2) + pauliMatrices::sigmaZ * rotationA(3));

    Eigen::Vector4cd rotationB;
    rotationB(0) = std::cos(ThetaB);
    rotationB(1) = iu*std::sin(ThetaB)*AxisB(0);
    rotationB(2) = iu*std::sin(ThetaB)*AxisB(1);
    rotationB(3) = iu*std::sin(ThetaB)*AxisB(2);

    UnitaryMatrix2cd rotationMatrixB = static_cast<UnitaryMatrix2cd>(HermitianMatrix2cd::Identity(2,2,"",enums::SpinSymmetry::SpinMatrix) * rotationB(0) + pauliMatrices::sigmaX * rotationB(1) + pauliMatrices::sigmaY * rotationB(2) + pauliMatrices::sigmaZ * rotationB(3));

    // Eigen::Matrix<ComplexMatrix,2,2> RotABlocked;// = rotationMatrixA * blockedA * rotationMatrixA.adjoint();
    // Eigen::Matrix<ComplexMatrix,2,2> temp;
    // for (int i = 0; i < 2; i++)
    // {
    //     for (int j = 0; j < 2; j++)
    //     {
    //         temp(i,j).resize(spinOrbitals/2,spinOrbitals/2,"A_Rotated",enums::SpinSymmetry::NoSpinSym);
    //         for (int k = 0; k < 2; k++)
    //         {
    //             temp(i,j) += rotationMatrixA(i,k)*blockedA(k,j);
    //         }
    //     }
    // }

    // for (int i = 0; i < 2; i++)
    // {
    //     for (int j = 0; j < 2; j++)
    //     {
    //         temp(i,j).resize(spinOrbitals/2,spinOrbitals/2,"A_Rotated",enums::SpinSymmetry::NoSpinSym);
    //         for (int k = 0; k < 2; k++)
    //         {
    //             RotABlocked(i,j) += temp(i,k)*std::conj(rotationMatrixA(j,k));
    //         }
    //     }
    // }

    auto ARotBasis = rotationMatrixA.asTemporaryBasisChangeMatrix();
    ComplexSelfAdjointMatrix RotA;
    matA.toBasisC(RotA,*ARotBasis);
    // RotA = rotationMatrixA * matA * rotationMatrixA.adjoint();
//     RotA.setZero();
// #warning TODO with the basis manager
//     RotA += toBlock(RotABlocked(0,0),0,0);
//     RotA += toBlock(RotABlocked(1,1),1,1); // I + Z
//     RotA += toBlock(RotABlocked(0,1),0,1);// X + Y
//     RotA += toBlock(RotABlocked(1,0),1,0);// X + Y

    // Eigen::Matrix<ComplexMatrix,2,2> RotBBlocked;// = rotationMatrixB * blockedB * rotationMatrixB.adjoint();

    // for (int i = 0; i < 2; i++)
    // {
    //     for (int j = 0; j < 2; j++)
    //     {
    //         temp(i,j) = ComplexMatrix::Zero(spinOrbitals/2,spinOrbitals/2);
    //         for (int k = 0; k < 2; k++)
    //         {
    //             temp(i,j) += rotationMatrixB(i,k)*blockedB(k,j);
    //         }
    //     }
    // }

    // for (int i = 0; i < 2; i++)
    // {
    //     for (int j = 0; j < 2; j++)
    //     {
    //         RotBBlocked(i,j) = ComplexMatrix::Zero(spinOrbitals/2,spinOrbitals/2);
    //         for (int k = 0; k < 2; k++)
    //         {
    //             RotBBlocked(i,j) += temp(i,k)*std::conj(rotationMatrixB(j,k));
    //         }
    //     }
    // }

    // ComplexMatrix RotB(spinOrbitals,spinOrbitals);
    auto BRotBasis = rotationMatrixB.asTemporaryBasisChangeMatrix();
    ComplexSelfAdjointMatrix RotB;
    matB.toBasisC(RotB,*ARotBasis);

    // RotB = rotationMatrixB * matB * rotationMatrixB;
    // RotB.setZero();
    // RotB += toBlock(RotBBlocked(0,0),0,0);
    // RotB += toBlock(RotBBlocked(1,1),1,1); // I + Z
    // RotB += toBlock(RotBBlocked(0,1),0,1);// X + Y
    // RotB += toBlock(RotBBlocked(1,0),1,0);// X + Y

    return (*nearFunc)(RotA,RotB);
}

#endif // GREENSFUNCTIONTESTHELPER_H
