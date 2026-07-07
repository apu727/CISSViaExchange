
#include "Tests/testutil.hpp"
#include "matrixselfenergy.h"

const std::string testBasis = "MatrixSelfEnergyTests";


template <class MT>
class MatrixSelfEnergyTests
{
public:
    MatrixSelfEnergy<MT> se;
    MT mat;
    MatrixSelfEnergyTests(const ComplexSelfAdjointSparseMatrix& Matrix)
        : se(Matrix)
    {
        mat = static_cast<MT>(Matrix);
        se.setBasis(mat.getBasis());
    }
    int test()
    {
        int ret = 0;

        MT seR;
        se.getSigmaR(0,seR);
        ret += expectNear(mat,seR,matrixNear,__PRETTY_FUNCTION__,"matDim");

        MT seA;
        MT seAExpt;
        se.getSigmaA(0,seA);
        seAExpt = seR.adjoint();
        ret += expectNear(seAExpt,seA,matrixNear,__PRETTY_FUNCTION__,"seA");
        // if the metric is changed after the self energy is constructed this fails
        /*
         setMetric(Blah)
         se.getSigmaA(0,seA);
         adjoint(seR,seAExpt);
         ret += expectNear(seAExpt,seA,matrixNear,__PRETTY_FUNCTION__,"seA"); Fail
         * */

        // MT gamma;
        // MT gammaExpt;
        // se.getGamma(0,gamma);
        // gammaExpt = iu*(seR-seA);
        // ret += expectNear(gammaExpt,gamma,matrixNear,__PRETTY_FUNCTION__,"gamma");
        MT zero(mat.rows(),mat.cols(),testBasis,enums::SpinSymmetry::NoSpinSym);
        zero.setZero();
        MT sigmaK;

        se.getSigmaK(0,sigmaK);
        ret += expectNear(zero,sigmaK,matrixNear,__PRETTY_FUNCTION__,"sigmaK");

        MT sigmaIn;


        se.getSigmaIn(0,sigmaIn);
        ret += expectNear(zero,sigmaIn,matrixNear,__PRETTY_FUNCTION__,"sigmaIn");

        MT sigmaOut;

        se.getSigmaOut(0,sigmaOut);

        ret += expectNear(zero,sigmaOut,matrixNear,__PRETTY_FUNCTION__,"sigmaOut");

        ret += expect(false,se.hasSigmaK(),__PRETTY_FUNCTION__,"hasSigmaK");
        ret += expect(selfEnergyType::Constant,se.m_type,__PRETTY_FUNCTION__,"type");
        ret += expect(true,se.isAnalytic(),__PRETTY_FUNCTION__,"type");
        return ret;
    }
};

int matrixselfenergyTests(int argc, char** argv)
{
    int ret = 0;

    Eigen::MatrixXcd Mat1(2,2);
    Mat1 << 1,2,2,4;
    BasisManager::getInstance().addOrthogonalBasis(testBasis,4);
    ComplexSelfAdjointSparseMatrix Mat(Mat1.sparseView(),testBasis,enums::SpinSymmetry::NoSpinSym);

    MatrixSelfEnergyTests<ComplexMatrix> se1(Mat);
    ret += se1.test();


    MatrixSelfEnergyTests<ComplexSparseMatrix> se2(Mat);
    ret += se2.test();
    BasisManager::getInstance().deleteBasis(testBasis);

    return ret > 0 ? -1 : 0;
}
