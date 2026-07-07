#include "Tests/testutil.hpp"
#include <integrator.h>
#include <Eigen/LU>

static std::string OrthBasis = "LinalgTestsOrth";
static std::string NonOrthBasis = "LinalgTestsNonOrth"; // The number of sites is appended to the end

int testSutractWithInfinities()
{
    int ret = 0;
    ret += expect(0.,subtractWithInfinities(posInf,posInf),__PRETTY_FUNCTION__,"posinf-posinf");
    ret += expect(0.,subtractWithInfinities(negInf,negInf),__PRETTY_FUNCTION__,"neginf-neginf");
    numType a = 1.231231245251535;
    numType b = 3.151251526464374;
    ret += expect(a-b,subtractWithInfinities(a,b),__PRETTY_FUNCTION__,"a-b");
    return ret;
}

int testAdjoint()
{
    int ret = 0;
    Eigen::MatrixXcd overlapMatrixEM(2,2);
    overlapMatrixEM << 1,    0.5,
                     0.5,  1;
    Eigen::MatrixXcd testMatrixEM(2,2);
    testMatrixEM << 1, 2,
                  3, 4;
    testMatrixEM *= 1.+iu;

    BasisManager::getInstance().deleteAllBasis();
    BasisManager::getInstance().addOrthogonalBasis(OrthBasis,testMatrixEM.rows());
    ComplexDirectHermitianMatrix overlapMatrix(overlapMatrixEM,NonOrthBasis,enums::SpinSymmetry::NoSpin);
    overlapMatrix.asMetric();

    ComplexMatrix testMatrixOrth(testMatrixEM,OrthBasis,enums::SpinSymmetry::NoSpin);
    ComplexMatrix testMatrixNonOrth(testMatrixEM,NonOrthBasis,enums::SpinSymmetry::NoSpin);

    ComplexSparseMatrix testMatrixOrthS = testMatrixOrth.sparseView();
    ComplexSparseMatrix testMatrixNonOrthS = testMatrixNonOrth.sparseView();

    {
        ComplexMatrix temp;
        ComplexMatrix Expt = ComplexMatrix(testMatrixEM.adjoint(),OrthBasis,enums::SpinSymmetry::NoSpin);
        ComplexSparseMatrix tempS;
        ComplexSparseMatrix ExptS = Expt.sparseView();

        temp = testMatrixOrth.adjoint();
        tempS = testMatrixOrthS.adjoint();

        ret += expectNear(Expt,temp,matrixNear,__PRETTY_FUNCTION__,"HermitConjugate");
        ret += expectNear(ExptS,tempS,matrixNear,__PRETTY_FUNCTION__,"HermitConjugateS");

        Expt  = ComplexMatrix((overlapMatrixEM*testMatrixEM*overlapMatrixEM.inverse()).adjoint(),NonOrthBasis,enums::SpinSymmetry::NoSpin);
        ExptS = Expt.sparseView();
        temp = testMatrixNonOrth.adjoint();
        tempS = testMatrixNonOrthS.adjoint();

        ret += expectNear(Expt,temp,matrixNear,__PRETTY_FUNCTION__,"adjoint");
        ret += expectNear(ExptS,tempS,matrixNear,__PRETTY_FUNCTION__,"adjointS");
    }
    BasisManager::getInstance().deleteAllBasis();
    return ret;
}



int linalgTests(int argc, char** argv)
{
    int ret = 0;
    ret += testSutractWithInfinities();
    ret += testAdjoint();
    return ret > 0 ? -1 : 0;
}
