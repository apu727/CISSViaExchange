#include "basismanager.h"
#include "linalg.h"
#include "testutil.hpp"
#include "GreensFunctionTestHelper.h"

template <typename T>
T getAlphaBlock(const T& mat)
{
    T ret;
    ret.setZero(mat.rows()/2, mat.cols()/2,mat.getBasis(),mat.getSpinSym());
    for (long i = 0; i < ret.rows(); i++)
    {
        for (long j = 0; j < ret.rows(); j++)
        {
            ret(i,j) = mat(2*i,2*j);
        }
    }
    return ret;
}
template <typename T>
T fromAlphaBlock(const T& mat)
{
    T ret;
    ret.setZero(mat.rows()*2, mat.cols()*2,mat.getBasis(),mat.getSpinSym());
    for (long i = 0; i < mat.rows(); i++)
    {
        for (long j = 0; j < mat.rows(); j++)
        {
            ret(2*i,2*j) = mat(i,j);
            ret(2*i+1,2*j+1) = mat(i,j);
        }
    }
    return ret;
}

//The metric is special and its eigenvectors can be obtained. But the compile time checks wont allow this (correctly) as it is only true for the metric
ComplexHermitianMatrix toLowdin(const ComplexSelfAdjointMatrix& mat, const Eigen::MatrixXcd& m)
{
    typename ComplexDirectHermitianMatrix::EigenValueType eVals;
    typename ComplexDirectHermitianMatrix::EigenVectorType eVecs;
    ComplexHermitianMatrix metric(m,mat.getBasis(),mat.getSpinSym());

    if (!metric.getEigenValuesAndVectors(eVals,eVecs))
        releaseAssert(false, "getEigenvalues for toLowdin failed");
    for (long i = 0; i < eVals.rows();i++)
    {
        eVals(i) = 1./sqrt(eVals(i));
    }
    auto shouldBeId = eVecs * eVecs.inverse();
    auto transform = eVecs * eVals.asDiagonal(mat.getBasis()) * eVecs.inverse();

    for (long i = 0; i < eVals.rows();i++)
    {
        eVals(i) = 1./eVals(i);
    }
    auto transformInv = eVecs * eVals.asDiagonal(mat.getBasis()) * eVecs.inverse();
    auto result = transformInv * mat * transform;

    return static_cast<ComplexHermitianMatrix>(result);
}

int testBasisTransforms()
{
    int ret = 0;
    BasisManager& bm = BasisManager::getInstance();
    bm.deleteAllBasis();

    Eigen::MatrixXcd transformMatrixEM(2,2);
    transformMatrixEM << 1,    0.5,
        0.5,  1;
    // If i is the parent and mu is the new basis
    //|e_\mu} = L_\mu^{i} |e_i>. Expected to be stored as L_{i\mu}
    Eigen::MatrixXcd transformMatrix2EM(2,2);
    transformMatrix2EM << 1,    0.25,
        0.25,  1;

    Eigen::MatrixXcd myRandomMatrixEM(2,2);
    myRandomMatrixEM << 1,    2,
        3,  4;
    ComplexMatrix myRandomMatrix(myRandomMatrixEM,"1",enums::SpinSymmetry::NoSpin);

    //S_\bar{\nu}\mu} = L^{\bar{j}}_{\bar{\nu}} L^{i}_{\mu} <e_\bar{j}|e_i>
    Eigen::MatrixXcd metric;
    metric.noalias() = transformMatrixEM.adjoint() * transformMatrixEM;

    bm.addOrthogonalBasis("1",transformMatrixEM.cols());
    bm.addBasis("Basis1_2","1",transformMatrixEM);

    ret += expectNear(metric,bm.getMetric("Basis1_2").m_MatBuffer.get(),matrixNear,__PRETTY_FUNCTION__,"Metric calculated wrong");
    if (ret) return ret;

    ComplexMatrix myRandomMatrix2;
    myRandomMatrix.toBasisC(myRandomMatrix2,"Basis1_2");
    ComplexMatrix myRandomMatrixExpt(transformMatrixEM.inverse() * myRandomMatrixEM * transformMatrixEM,"Basis1_2",enums::SpinSymmetry::NoSpin);
    ret += expectNear(myRandomMatrixExpt,myRandomMatrix2,matrixNear,__PRETTY_FUNCTION__,"Transform wrong");
    if (ret) return ret;

    bm.addBasis("Basis2_3","1",transformMatrix2EM);
    myRandomMatrix.toBasisC(myRandomMatrix2,"Basis2_3");
    myRandomMatrixExpt = ComplexMatrix(transformMatrix2EM.inverse() * myRandomMatrixEM *transformMatrix2EM,"Basis2_3",enums::SpinSymmetry::NoSpin);
    ret += expectNear(myRandomMatrixExpt,myRandomMatrix2,matrixNear,__PRETTY_FUNCTION__,"Transform wrong, graph traversal");
    if (ret) return ret;

    //Backwards graph traversal
    myRandomMatrix2.toBasisC(myRandomMatrix2,"1");
    ret += expectNear(myRandomMatrix,myRandomMatrix2,matrixNear,__PRETTY_FUNCTION__,"Transform wrong, Backwards graph traversal");
    if (ret) return ret;

    //Test useless transform
    ComplexMatrix myRandomMatrix3;
    myRandomMatrix2.toBasisC(myRandomMatrix3,"1");
    ret += expect(myRandomMatrix2,myRandomMatrix3,__PRETTY_FUNCTION__,"Useless transform");

    //Test multiple traversal
    bm.addBasis("Basis1_3","Basis1_2",transformMatrixEM);
    myRandomMatrix.toBasisC(myRandomMatrix3,"Basis1_3");
    myRandomMatrixExpt = ComplexMatrix(transformMatrixEM.inverse() * transformMatrixEM.inverse() * myRandomMatrixEM *transformMatrixEM * transformMatrixEM,"Basis1_3",enums::SpinSymmetry::NoSpin);
    ret += expectNear(myRandomMatrixExpt,myRandomMatrix3,matrixNear,__PRETTY_FUNCTION__,"Multiple Transform wrong");
    myRandomMatrix.toBasisC(myRandomMatrix3,"Basis1_3");
    ret += expectNear(myRandomMatrixExpt,myRandomMatrix3,matrixNear,__PRETTY_FUNCTION__,"Multiple Transform wrong, Cached?");

    myRandomMatrix.toBasisC(myRandomMatrix2,"Basis2_3");
    myRandomMatrix.toBasisC(myRandomMatrix3,"Basis1_2");
    ComplexMatrix myRandomMatrix4;
    myRandomMatrix2.toBasisC(myRandomMatrix2,"Basis1_2");

    ret += expectNear(myRandomMatrix3,myRandomMatrix2,matrixNear,__PRETTY_FUNCTION__,"Common Parent Transform wrong");
    if (ret) return ret;

    //Delete and ask for transform, should return false
    bm.deleteBasis("Basis1_2");
    bool success =  myRandomMatrix.toBasis(myRandomMatrix2,"Basis1_2");
    ret += expect(false,success,__PRETTY_FUNCTION__,"Did not fail basis transformation");
    if (ret) return ret;


    success = myRandomMatrix.toBasis(myRandomMatrix3,"Basis1_3");
    ret += expect(true,success,__PRETTY_FUNCTION__,"Failed Transform from 1 => Basis1_3 after intermediate has been deleted. This probably means it was not cached");
    if (ret) return ret;
    myRandomMatrixExpt = ComplexMatrix(transformMatrixEM.inverse() * transformMatrixEM.inverse() * myRandomMatrixEM *transformMatrixEM * transformMatrixEM,"Basis1_3",enums::SpinSymmetry::NoSpin);
    ret += expectNear(myRandomMatrixExpt,myRandomMatrix3,matrixNear,__PRETTY_FUNCTION__,"Incorrect Transform from 1 => Basis1_3 after intermediate has been deleted. This probably means it was not cached");
    if (ret) return ret;

    bm.deleteBasis("Basis1_2");
    bm.deleteBasis("1");
    bm.deleteBasis("Basis1_3");
    bm.deleteBasis("Basis2_3");
    ret += expect(1ul,bm.m_BasisList.size(),__PRETTY_FUNCTION__,"Basis Delete");
    ret += expect(1ul,bm.getBasisObjects(),__PRETTY_FUNCTION__,"Basis numObjects");
    bm.deleteAllBasis();

    return ret;
}

int testHardBasisTransforms()
{
    int ret = 0;
    BasisManager& bm = BasisManager::getInstance();
    bm.deleteAllBasis();
    ComplexMatrix transform;
    std::string orthName = "Orth";

    makeMetric(10,transform,orthName,"blank");
    transform *= std::exp(iu);
    Eigen::MatrixXcd transformEm = *transform;
    Eigen::MatrixXcd transformInvEm = transformEm.inverse();

    bm.addOrthogonalBasis(orthName,20);
    auto nonOrthName = transform.asTemporaryBasisChangeMatrix();

    //4 possible cases

    {
        //M^{i\bar{j}}
        Eigen::MatrixXcd AExpt(20,20);
        Eigen::MatrixXcd AExptTransf(20,20);
        AExpt.setRandom();
        ComplexDualMatrix A(AExpt,orthName, enums::SpinSymmetry::NoSpinSym);
        AExptTransf = transformInvEm * AExpt * transformInvEm.adjoint();
        A.toBasisC(A,*nonOrthName);
        ret += expectNear(AExptTransf,*A,matrixNear,__PRETTY_FUNCTION__,"Dual,improper dual failed");
        A.toBasisC(A,orthName);
        ret += expectNear(AExpt,*A,matrixNear,__PRETTY_FUNCTION__,"Dual,improper dual failed2");

    }
    {
        //M^{i}_{j}
        Eigen::MatrixXcd AExpt(20,20);
        Eigen::MatrixXcd AExptTransf(20,20);
        AExpt.setRandom();
        ComplexMatrix A(AExpt,orthName, enums::SpinSymmetry::NoSpinSym);
        AExptTransf = transformInvEm * AExpt * transformEm;
        A.toBasisC(A,*nonOrthName);
        ret += expectNear(AExptTransf,*A,matrixNear,__PRETTY_FUNCTION__,"Dual,direct failed");
        A.toBasisC(A,orthName);
        ret += expectNear(AExpt,*A,matrixNear,__PRETTY_FUNCTION__,"Dual,direct  failed2");
    }

    {
        //M_{\bar{i}j}
        Eigen::MatrixXcd AExpt(20,20);
        Eigen::MatrixXcd AExptTransf(20,20);
        AExpt.setRandom();
        ComplexDirectMatrix A(AExpt,orthName, enums::SpinSymmetry::NoSpinSym);
        AExptTransf = transformEm.adjoint() * AExpt * transformEm;
        A.toBasisC(A,*nonOrthName);
        ret += expectNear(AExptTransf,*A,matrixNear,__PRETTY_FUNCTION__,"impropert direct,direct failed");
        A.toBasisC(A,orthName);
        ret += expectNear(AExpt,*A,matrixNear,__PRETTY_FUNCTION__,"impropert direct,direct failed2");
    }
    {
        //M_{i}^{j}
        Eigen::MatrixXcd AExpt(20,20);
        Eigen::MatrixXcd AExptTransf(20,20);
        AExpt.setRandom();
        Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::direct,enums::IndexType::dual> A(AExpt,orthName, enums::SpinSymmetry::NoSpinSym);
        AExptTransf = transformEm.transpose() * AExpt * transformInvEm.transpose();
        A.toBasisC(A,*nonOrthName);
        ret += expectNear(AExptTransf,*A,matrixNear,__PRETTY_FUNCTION__,"enums::IndexType::direct,enums::IndexType::dual failed");
        A.toBasisC(A,orthName);
        ret += expectNear(AExpt,*A,matrixNear,__PRETTY_FUNCTION__,"ienums::direct,enums::IndexType::dual failed2");
    }
    //There are some 'illegal' ones which are actually tensor products. Ill only test a couple
    {
        //M_{i}_{j}
        Eigen::MatrixXcd AExpt(20,20);
        Eigen::MatrixXcd AExptTransf(20,20);
        AExpt.setRandom();
        Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::direct,enums::IndexType::direct> A(AExpt,orthName, enums::SpinSymmetry::NoSpinSym);
        AExptTransf = transformEm.transpose() * AExpt * transformEm;
        A.toBasisC(A,*nonOrthName);
        ret += expectNear(AExptTransf,*A,matrixNear,__PRETTY_FUNCTION__,"enums::IndexType::direct,enums::IndexType::direct failed");
        A.toBasisC(A,orthName);
        ret += expectNear(AExpt,*A,matrixNear,__PRETTY_FUNCTION__,"ienums::direct,enums::IndexType::direct failed2");
    }

    {
        //M^{i}^{j}
        Eigen::MatrixXcd AExpt(20,20);
        Eigen::MatrixXcd AExptTransf(20,20);
        AExpt.setRandom();
        Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::dual,enums::IndexType::dual> A(AExpt,orthName, enums::SpinSymmetry::NoSpinSym);
        AExptTransf = transformInvEm * AExpt * transformInvEm.transpose();
        A.toBasisC(A,*nonOrthName);
        ret += expectNear(AExptTransf,*A,matrixNear,__PRETTY_FUNCTION__,"enums::IndexType::dual,enums::IndexType::dual failed");
        A.toBasisC(A,orthName);
        ret += expectNear(AExpt,*A,matrixNear,__PRETTY_FUNCTION__,"ienums::dual,enums::IndexType::dual failed2");
    }

    {
        //M_{i}_{j}
        Eigen::MatrixXcd AExpt(20,20);
        Eigen::MatrixXcd AExptTransf(20,20);
        AExpt.setRandom();
        Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::improperDirect,enums::IndexType::improperDirect> A(AExpt,orthName, enums::SpinSymmetry::NoSpinSym);
        AExptTransf = transformEm.adjoint() * AExpt * transformEm.conjugate();
        A.toBasisC(A,*nonOrthName);
        ret += expectNear(AExptTransf,*A,matrixNear,__PRETTY_FUNCTION__,"enums::IndexType::improperDirect,enums::IndexType::improperDirect failed");
        A.toBasisC(A,orthName);
        ret += expectNear(AExpt,*A,matrixNear,__PRETTY_FUNCTION__,"ienums::improperDirect,enums::IndexType::improperDirect failed2");
    }

    {
        //M^{i}^{j}
        Eigen::MatrixXcd AExpt(20,20);
        Eigen::MatrixXcd AExptTransf(20,20);
        AExpt.setRandom();
        Matrix<complexType,Eigen::Dynamic,Eigen::Dynamic,enums::MatrixProperties::None,enums::IndexType::improperDual,enums::IndexType::improperDual> A(AExpt,orthName, enums::SpinSymmetry::NoSpinSym);
        AExptTransf = transformInvEm.conjugate() * AExpt * transformInvEm.adjoint();
        A.toBasisC(A,*nonOrthName);
        ret += expectNear(AExptTransf,*A,matrixNear,__PRETTY_FUNCTION__,"enums::IndexType::improperDual,enums::IndexType::improperDual failed");
        A.toBasisC(A,orthName);
        ret += expectNear(AExpt,*A,matrixNear,__PRETTY_FUNCTION__,"ienums::improperDual,enums::IndexType::improperDual failed2");
    }
#warning TODO test raising and lowering transform
    bm.deleteBasis(orthName);
    return ret;
}

int testBuiltinBasisTransforms()
{
    int ret = 0;
    BasisManager& bm = BasisManager::getInstance();
    bm.deleteAllBasis();

    Eigen::MatrixXcd transformMatrixEM(2,2);
    transformMatrixEM << 1,    0.5,
        0.5,  1;

    Eigen::MatrixXcd myRandomMatrixEM(2,2);
    myRandomMatrixEM << 1,    2,
        2,  4;
    ComplexSelfAdjointMatrix spatialMatrixOrth(myRandomMatrixEM,"1" + BasisManager::Alpha_Block,enums::SpinSymmetry::RHF);
    ComplexSelfAdjointMatrix spatialMatrixNonOrth;

    bm.addOrthogonalBasis("1",transformMatrixEM.cols()*2);
    ComplexMatrix transformSpatial(transformMatrixEM,"1" + BasisManager::Alpha_Block,enums::SpinSymmetry::RHF);
    ComplexMatrix transform;
    {
        ComplexMatrix transformExpt = fromAlphaBlock(transformSpatial);
        transformExpt.setBasis("1");
        transformSpatial.toBasisC(transform,"1");
        ret += expectNear(transformExpt,transform,makeMatrixNearTol(transformExpt),__PRETTY_FUNCTION__,"Fail transformExpt");
    }
    transform.setBasis({"1",std::string("Basis1_2")});
    transform.asBasisChangeMatrix();
    // bm.addBasis("Basis1_2","1",transformMatrixEM);
    auto metric = bm.getMetric("Basis1_2");

    spatialMatrixOrth.toBasisC(spatialMatrixNonOrth,"Basis1_2" + BasisManager::Alpha_Block);
    {
        ComplexSelfAdjointMatrix spatialMatrixNonOrthExpt = static_cast<ComplexSelfAdjointMatrix>(transformSpatial.inverse() * spatialMatrixOrth * transformSpatial);
        spatialMatrixNonOrthExpt.setBasis("Basis1_2" + BasisManager::Alpha_Block);
        ret += expectNear(spatialMatrixNonOrthExpt,spatialMatrixNonOrthExpt,makeMatrixNearTol(spatialMatrixNonOrthExpt),__PRETTY_FUNCTION__,"Fail spatialMatrixNonOrth");
    }

    ComplexSelfAdjointMatrix fullMatrixOrth;
    ComplexSelfAdjointMatrix fullMatrixOrthExpt = fromAlphaBlock(spatialMatrixOrth);
    fullMatrixOrthExpt.setBasis("1");
    {
        ComplexSelfAdjointMatrix temp;
        spatialMatrixOrth.toBasisC(temp,"1");
        fullMatrixOrth = static_cast<ComplexSelfAdjointMatrix>(temp);
    }
    ret += expectNear(fullMatrixOrthExpt,fullMatrixOrth,makeMatrixNearTol(fullMatrixOrthExpt),__PRETTY_FUNCTION__,"Fail from alpha block Orth");

    ComplexSelfAdjointMatrix fullMatrixNonOrth;
    ComplexSelfAdjointMatrix fullMatrixNonOrthExpt = fromAlphaBlock(spatialMatrixNonOrth);
    fullMatrixNonOrthExpt.setBasis("Basis1_2");
    spatialMatrixNonOrth.toBasisC(fullMatrixNonOrth,"Basis1_2");
    ret += expectNear(fullMatrixNonOrthExpt,fullMatrixNonOrth,makeMatrixNearTol(fullMatrixNonOrthExpt),__PRETTY_FUNCTION__,"Fail from alpha block Non Orth");

    ComplexHermitianMatrix fullMatrixNonOrthLowdin;
    ComplexHermitianMatrix fullMatrixNonOrthLowdin2;
    ComplexHermitianMatrix fullMatrixNonOrthLowdinExpt = toLowdin(fullMatrixNonOrth,metric.m_MatBuffer.get());
    fullMatrixNonOrthLowdinExpt.setBasis("Basis1_2" + BasisManager::LowdinOrthogonaliseBasis);

    {
        ComplexMatrix temp;
        spatialMatrixNonOrth.toBasisC(temp,"Basis1_2" + BasisManager::LowdinOrthogonaliseBasis);
        fullMatrixNonOrthLowdin = static_cast<ComplexHermitianMatrix>(temp);
    }
    ret += expectNear(fullMatrixNonOrthLowdinExpt,fullMatrixNonOrthLowdin,makeMatrixNearTol(fullMatrixNonOrthLowdinExpt),__PRETTY_FUNCTION__,"Fail Lowdin 1");

    bool s = fullMatrixNonOrth.toHermitianBasis(fullMatrixNonOrthLowdin2);
    ret += expectNear(fullMatrixNonOrthLowdinExpt,fullMatrixNonOrthLowdin2,makeMatrixNearTol(fullMatrixNonOrthLowdinExpt),__PRETTY_FUNCTION__,"Fail Lowdin 2");

    ComplexSelfAdjointMatrix fullMatrixNonOrthLowdinAA;
    ComplexSelfAdjointMatrix fullMatrixNonOrthLowdinAAExpt = static_cast<ComplexSelfAdjointMatrix>(getAlphaBlock(fullMatrixNonOrthLowdinExpt));
    fullMatrixNonOrthLowdinAAExpt.setBasis("Basis1_2" + BasisManager::Alpha_Block +BasisManager::LowdinOrthogonaliseBasis);

    fullMatrixNonOrth.toBasisC(fullMatrixNonOrthLowdinAA,"Basis1_2" + BasisManager::Alpha_Block +BasisManager::LowdinOrthogonaliseBasis);
    ret += expectNear(fullMatrixNonOrthLowdinAAExpt,fullMatrixNonOrthLowdinAA,makeMatrixNearTol(fullMatrixNonOrthLowdinAAExpt),__PRETTY_FUNCTION__,"Fail Lowdin AA1");

    fullMatrixNonOrth.toBasisC(fullMatrixNonOrthLowdinAA,"Basis1_2" +BasisManager::LowdinOrthogonaliseBasis + BasisManager::Alpha_Block );
    ret += expectNear(fullMatrixNonOrthLowdinAAExpt,fullMatrixNonOrthLowdinAA,makeMatrixNearTol(fullMatrixNonOrthLowdinAAExpt),__PRETTY_FUNCTION__,"Fail Lowdin AA2");
    //TODO check with vectors also, Both row and column
    //Some of these should fail
    {
        ComplexSelfAdjointMatrix temp;
        Eigen::MatrixXcd Zero = Eigen::MatrixXcd::Zero(spatialMatrixOrth.rows(),spatialMatrixOrth.cols());
        bool success;
        //succeed
        fullMatrixOrth.toBasisC(temp,"1" + BasisManager::Beta_Block);
        ret += expectNear(spatialMatrixOrth.m_MatBuffer.get(),temp.m_MatBuffer.get(),matrixNear,__PRETTY_FUNCTION__,"Beta_Block");
        fullMatrixOrth.toBasisC(temp,"1" + BasisManager::Alpha_Block);
        ret += expectNear(spatialMatrixOrth.m_MatBuffer.get(),temp.m_MatBuffer.get(),matrixNear,__PRETTY_FUNCTION__,"Alpha_Block");
        spatialMatrixOrth.toBasisC(temp,"1" + BasisManager::Beta_Block);
        ret += expectNear(spatialMatrixOrth.m_MatBuffer.get(),temp.m_MatBuffer.get(),matrixNear,__PRETTY_FUNCTION__,"Alpha_Block to Beta_Block");
        ComplexMatrix temp2;
        fullMatrixOrth.toBasisC(temp2,{"1" + BasisManager::Beta_Block,"1"+BasisManager::Alpha_Block});
        ret += expectNear(Zero,temp2.m_MatBuffer.get(),matrixNear,__PRETTY_FUNCTION__,"Beta_Alpha_Block");
        fullMatrixOrth.toBasisC(temp2,{"1" + BasisManager::Alpha_Block,"1"+BasisManager::Beta_Block});
        ret += expectNear(Zero,temp2.m_MatBuffer.get(),matrixNear,__PRETTY_FUNCTION__,"Alpha_Beta_Block");

        //fail
        success = spatialMatrixOrth.toBasis(temp2,"1" + BasisManager::Alpha_Block + BasisManager::Beta_Block);
        ret += expect(false,success,__PRETTY_FUNCTION__,"Double Blocking1");
        // success = spatialMatrixOrth.toBasis(temp2,"1" + BasisManager::Alpha_Block + BasisManager::Beta_Alpha_Block);
        // ret += expect(false,success,__PRETTY_FUNCTION__,"Double Blocking2");
        // success = spatialMatrixOrth.toBasis(temp2,"1" + BasisManager::Alpha_Block + BasisManager::Alpha_Beta_Block);
        // ret += expect(false,success,__PRETTY_FUNCTION__,"Double Blocking3");
        success = spatialMatrixOrth.toBasis(temp2,"1" + BasisManager::Alpha_Block + BasisManager::Alpha_Block);
        ret += expect(false,success,__PRETTY_FUNCTION__,"Double Blocking4");
    }

    bm.deleteBasis("Basis1_2");
    bm.deleteBasis("1");

    ret += expect(1ul,bm.m_BasisList.size(),__PRETTY_FUNCTION__,"Basis Delete");
    ret += expect(1ul,bm.getBasisObjects(),__PRETTY_FUNCTION__,"Basis numObjects");
    bm.deleteAllBasis();

    return ret;
}

int basisManagerTests(int argc, char** argv)
{
    int success = 0;

    success += testBasisTransforms();
    success += testBuiltinBasisTransforms();
    success += testHardBasisTransforms();

    return success > 0 ? -1 : 0;

}
