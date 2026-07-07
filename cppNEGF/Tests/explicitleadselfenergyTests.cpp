#include "Tests/testutil.hpp"
#include "explicitleadselfenergy.h"



template <class MT>
class ExplicitLeadSelfEnergyTests
{
public:
    ExplicitLeadSelfEnergy<MT> se;
    ExplicitLeadSelfEnergyTests(typename ExplicitLeadSelfEnergy<MT>::Parameters& params1)
        : se(params1)
    {

    }
    int test()
    {
        int ret = 0;
        numType bottomofBand = se.m_params.bottomOfBand;
        ret += expect(0.,se.KFromE(bottomofBand-1),__PRETTY_FUNCTION__,"KFromE");
        numType topOfBand = bottomofBand + 4*se.m_params.t;
        ret += expectNear(M_PI/se.m_params.a,se.KFromE(topOfBand-1e-15),2e-8,__PRETTY_FUNCTION__,"KFromE2");

        ret += expectNear(0.,se.fermiFunction(se.m_params.mu+31*kb*T),__PRETTY_FUNCTION__,"fermiFunction1");
        ret += expectNear(1.,se.fermiFunction(se.m_params.mu-31*kb*T),__PRETTY_FUNCTION__,"fermiFunction2");
        ret += expectNear(0.5,se.fermiFunction(se.m_params.mu),__PRETTY_FUNCTION__,"fermiFunction2");

        MT seR;
        se.getSigmaR(se.m_params.mu,seR);
        ret += expect((long)se.m_params.MatrixDimension,seR.cols(),__PRETTY_FUNCTION__,"matDim");
        MT seA;
        MT seAExpt;
        se.getSigmaA(se.m_params.mu,seA);
        seAExpt = seR.adjoint();
        ret += expectNear(seAExpt,seA,matrixNear,__PRETTY_FUNCTION__,"seA");

        MT gamma;
        MT gammaExpt;
        se.getGamma(se.m_params.mu,gamma);
        gammaExpt = iu*(seR-seA);
        ret += expectNear(gammaExpt,gamma,matrixNear,__PRETTY_FUNCTION__,"gamma");

        MT sigmaK;
        MT sigmaKExpt;
        se.getSigmaK(se.m_params.mu,sigmaK);
        sigmaKExpt = -iu*(1-2*se.fermiFunction(se.m_params.mu))*gammaExpt;
        ret += expectNear(sigmaKExpt,sigmaK,matrixNear,__PRETTY_FUNCTION__,"sigmaK");

        MT sigmaIn;
        MT sigmaInExpt;

        se.getSigmaIn(se.m_params.mu,sigmaIn);
        sigmaInExpt = se.fermiFunction(se.m_params.mu)*gammaExpt;
        ret += expectNear(sigmaInExpt,sigmaIn,matrixNear,__PRETTY_FUNCTION__,"sigmaIn");

        MT sigmaOut;
        MT sigmaOutExpt;

        se.getSigmaOut(se.m_params.mu,sigmaOut);
        sigmaOutExpt = (1-se.fermiFunction(se.m_params.mu))*gammaExpt;
        ret += expectNear(sigmaOutExpt,sigmaOut,matrixNear,__PRETTY_FUNCTION__,"sigmaOut");

        ret += expect(se.m_params.mu,se.getChemicalPotential(),__PRETTY_FUNCTION__,"getChemPot");
        ret += expect(true,se.hasSigmaK(),__PRETTY_FUNCTION__,"hasSigmaK");
        ret += expect(selfEnergyType::noAssumptions,se.m_type,__PRETTY_FUNCTION__,"type");
        ret += expect(false,se.isAnalytic(),__PRETTY_FUNCTION__,"type");
        return ret;
    }
    int testChangeMatrixDimension()
    {
        int ret = 0;
        se.setMatrixDimension(4);
        ret += expect(4u,se.m_params.MatrixDimension,__PRETTY_FUNCTION__,"setMatrixDim");
        ret +=test();
        return ret;
    }
};

int explicitleadselfenergyTests(int argc, char** argv)
{
    int ret = 0;
    typename ExplicitLeadSelfEnergy<ComplexMatrix>::Parameters params1;
    params1.a = 2;
    params1.alpha1 = 1;
    params1.t = 2;
    params1.bottomOfBand = -4;
    params1.mu = 0;
    params1.couplingPoints = {0};
    params1.MatrixDimension = 2;

    ExplicitLeadSelfEnergyTests<ComplexMatrix> se1(params1);
    ret += se1.test();
    ret += se1.testChangeMatrixDimension();

    typename ExplicitLeadSelfEnergy<ComplexSparseMatrix>::Parameters params2;
    params2.a = 2;
    params2.alpha1 = 1;
    params2.t = 2;
    params2.bottomOfBand = -4;
    params2.mu = 0;
    params2.couplingPoints = {0};
    params2.MatrixDimension = 2;

    ExplicitLeadSelfEnergyTests<ComplexSparseMatrix> se2(params2);
    ret += se2.test();
    ret += se2.testChangeMatrixDimension();

    return ret > 0 ? -1 : 0;
}

