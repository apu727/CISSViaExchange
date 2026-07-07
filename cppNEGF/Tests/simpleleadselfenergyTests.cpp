#include "Tests/testutil.hpp"
#include "simpleleadselfenergy.h"

#include "GreensFunctionTestHelper.h"

const std::string simpleLeadSelfEnergyTestBasis = "simpleLeadSelfEnergyTestOrth";

template <class MT>
class SimpleLeadSelfEnergyTests
{
public:
    SimpleLeadSelfEnergy<MT> se;
    typename SimpleLeadSelfEnergy<MT>::Parameters params;
    MT m_seR;
    SimpleLeadSelfEnergyTests(typename SimpleLeadSelfEnergy<MT>::Parameters& params1, MT& sampleSeR)
        : se(params1)
    {
        params=params1;
        m_seR = sampleSeR;
        se.setBasis(sampleSeR.getBasis());
    }
    int test()
    {
        int ret = 0;

        ret += expectNear(0.,se.fermiFunction(se.m_params.mu+31*kb*T),__PRETTY_FUNCTION__,"fermiFunction1");
        ret += expectNear(1.,se.fermiFunction(se.m_params.mu-31*kb*T),__PRETTY_FUNCTION__,"fermiFunction2");
        ret += expectNear(0.5,se.fermiFunction(se.m_params.mu),__PRETTY_FUNCTION__,"fermiFunction2");

        MT seR;
        se.getSigmaR(se.m_params.mu,seR);
        ret += expect((long)se.m_params.MatrixDimension,seR.cols(),__PRETTY_FUNCTION__,"matDim");
        ret += expectNear(m_seR,seR,matrixNear,__PRETTY_FUNCTION__,"seR");
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

        if (se.m_params.hasBottom)
            ret += expect(selfEnergyType::ConstantPiecewise,se.m_type,__PRETTY_FUNCTION__,"type");
        else
            ret += expect(selfEnergyType::Constant,se.m_type,__PRETTY_FUNCTION__,"type");
        ret += expect(true,se.isAnalytic(),__PRETTY_FUNCTION__,"isAnalytic");
        return ret;
    }
    int testChangeT()
    {
        int ret = 0;
        params.t *=2;
        se.setT(params.t);
        m_seR*=2;
        ret += test();
        return ret;
    }

    int testChangemu()
    {
        int ret = 0;
        params.mu += 1;
        se.setChemicalPotential(params.mu);
        ret += expect(params.mu,se.m_params.mu,__PRETTY_FUNCTION__,"Change mu");
        test();
        return ret;
    }

    int testBottomOfBand()
    {
        int ret = 0;
        auto paramsCopy = params;
        paramsCopy.hasBottom = true;
        paramsCopy.bottomOfBand = -1;
        paramsCopy.mu = -2;

        se = SimpleLeadSelfEnergy<MT>(paramsCopy);
        se.setBasis(m_seR.getBasis());
        MT backupSer = m_seR;
        m_seR.setZero();
        ret += test();
        auto regions = se.getContinuousRegions();
        ret += expect(3ul,regions.size(),__PRETTY_FUNCTION__,"RegionCount");
        ret += expect(0.,subtractWithInfinities(regions[0],negInf),__PRETTY_FUNCTION__,"RegionS1");
        ret += expect(0.,subtractWithInfinities(regions[1],-1),__PRETTY_FUNCTION__,"RegionS2");
        ret += expect(0.,subtractWithInfinities(regions[2],posInf),__PRETTY_FUNCTION__,"RegionS3");

        m_seR = backupSer;
        se = SimpleLeadSelfEnergy<MT>(params);
        se.setBasis(m_seR.getBasis());
        return ret;
    }
    int testAdjustElectrons()
    {
        int ret = 0;
        auto paramsCopy = params;
        paramsCopy.adjustMuToElectrons = true;
        paramsCopy.numberOfElectrons = 2;
        paramsCopy.muMax = 1;
        se = SimpleLeadSelfEnergy<MT>(paramsCopy);
        se.setBasis(m_seR.getBasis());
        GreensFunctionTest GF;
        GF.setup(ComplexSelfAdjointSparseMatrix::Zero(4,4,simpleLeadSelfEnergyTestBasis,enums::SpinSymmetry::RHF));
        std::vector<EigenValueVector> eigVals;
        std::vector<EigenVectorMatrix> rightEigenVectors;

        eigVals.resize(1);
        {
            Eigen::VectorXcd eigValsEm;
            eigValsEm.resize(4);
            eigValsEm << -1, 0, -0.5, 1;
            eigVals[0] = EigenValueVector(eigValsEm,"",enums::SpinSymmetry::RHF);
        }



        rightEigenVectors.resize(1);
        rightEigenVectors[0].setIdentity(4,4,simpleLeadSelfEnergyTestBasis,enums::SpinSymmetry::RHF);

        GF.setEigenValues(eigVals);
        GF.setEigenVectors(rightEigenVectors);
        GF.setComputationMethod(GreensFunction::GrComputationType::AnalyticSOP);
        se.setGreensFunction(GF.m_gfunc);

        se.clearCache();
        ret += expect(-0.25,se.m_params.mu,__PRETTY_FUNCTION__,"AutoAdjust mu1");

        {
            Eigen::VectorXcd eigValsEm;
            eigValsEm.resize(4);
            eigValsEm << 2,2,2,2;
            eigVals[0] = EigenValueVector(eigValsEm,"",enums::SpinSymmetry::RHF);
        }

        GF.setEigenValues(eigVals);
        se.clearCache();
        ret += expect(1.,se.m_params.mu,__PRETTY_FUNCTION__,"muMax");

        //Regions Test
        GF.setComputationMethod(GreensFunction::GrComputationType::AnalyticSOPRegions);
        GF.setRegions({negInf,0,posInf});
        eigVals.resize(2);
        {
            Eigen::VectorXcd eigValsEm;
            eigValsEm.resize(4);
            eigValsEm << -1, 0.1, -0.5, 1;
            eigVals[0] = EigenValueVector(eigValsEm,"",enums::SpinSymmetry::RHF);
            eigValsEm << -10, -20, 0.5, 1;
            eigVals[1] = EigenValueVector(eigValsEm,"",enums::SpinSymmetry::RHF);
        }

        rightEigenVectors.push_back(rightEigenVectors[0]);
        GF.setEigenValues(eigVals);
        GF.setEigenVectors(rightEigenVectors);

        se.clearCache();
        ret += expect(0.,se.m_params.mu,__PRETTY_FUNCTION__,"regionsAdjust");
        // No Lumo Test

        GF.reset();
        GF.setup(ComplexSelfAdjointSparseMatrix::Zero(2,2,simpleLeadSelfEnergyTestBasis,enums::SpinSymmetry::RHF));
        eigVals.resize(1);
        rightEigenVectors.resize(1);
        {
            Eigen::VectorXcd eigValsEm;
            eigValsEm.resize(2);
            eigValsEm << -1. - iu,-1. - iu;
            eigVals[0] = EigenValueVector(eigValsEm,"",enums::SpinSymmetry::RHF);
        }

        rightEigenVectors[0].setIdentity(2,2,simpleLeadSelfEnergyTestBasis,enums::SpinSymmetry::RHF);
        GF.setComputationMethod(GreensFunction::GrComputationType::AnalyticSOP);
        GF.setEigenValues(eigVals);
        GF.setEigenVectors(rightEigenVectors);

        se.setGreensFunction(GF.m_gfunc);

        se.clearCache();
        bool (*lt)(const numType&,const numType&) = [](const numType& a, const numType& b){return a < b;};
        ret += expectNear(-1.,se.m_params.mu,lt,__PRETTY_FUNCTION__,"noLumoAdjust");

        // no Lumo with regions
        GF.setComputationMethod(GreensFunction::GrComputationType::AnalyticSOPRegions);
        GF.setRegions({negInf,0,posInf});
        eigVals.resize(2);
        rightEigenVectors.resize(2);
        {
            Eigen::VectorXcd eigValsEm;
            eigValsEm.resize(2);
            eigValsEm << -1. - iu, 0.25 - iu;
            eigVals[0] = EigenValueVector(eigValsEm,"",enums::SpinSymmetry::RHF);
            eigValsEm << 0.5 - iu,-1. - iu;
            eigVals[1] = EigenValueVector(eigValsEm,"",enums::SpinSymmetry::RHF);
        }
        rightEigenVectors[1].setIdentity(2,2,simpleLeadSelfEnergyTestBasis,enums::SpinSymmetry::RHF);

        GF.setEigenValues(eigVals);
        GF.setEigenVectors(rightEigenVectors);

        se.clearCache();

        ret += expectNear(0.5,se.m_params.mu,lt,__PRETTY_FUNCTION__,"noLumoAdjust");

        se = SimpleLeadSelfEnergy<MT>(params);
        se.setBasis(m_seR.getBasis());
        return ret;
    }

};

int simpleleadselfenergyTests(int argc, char** argv)
{
    int ret = 0;
    typename SimpleLeadSelfEnergy<ComplexMatrix>::Parameters params1;
    params1.alpha1 = 1;
    params1.t = 1;
    params1.mu = 0;

    params1.adjustMuToElectrons = false;
    params1.numberOfElectrons = 0;
    params1.muMax = 0;

    params1.hasBottom = false;
    params1.bottomOfBand = 0;


    Eigen::MatrixXcd couplingPoints(2,2);
    couplingPoints << 1,0,0,0;
    params1.couplingPoints = ComplexDirectMatrix(couplingPoints,simpleLeadSelfEnergyTestBasis + BasisManager::Alpha_Block,enums::SpinSymmetry::RHF);
    params1.magnetisationDir = Eigen::Vector3d({0,0,0.5});
    params1.upSpin = true;

    params1.MatrixDimension = 4; // TODO autodetermine
    BasisManager::getInstance().addOrthogonalBasis(simpleLeadSelfEnergyTestBasis,params1.MatrixDimension);

    ComplexMatrix expectedSigmaR(4,4,simpleLeadSelfEnergyTestBasis,enums::SpinSymmetry::NoSpinSym);
    expectedSigmaR.setZero();
    expectedSigmaR(0,0) = -params1.alpha1*iu*params1.t/2.*(1.5);

    SimpleLeadSelfEnergyTests<ComplexMatrix> se1(params1,expectedSigmaR);
    ret += se1.test();
    ret += se1.testChangemu();
    ret += se1.testBottomOfBand();
    ret += se1.testChangeT();
    ret += se1.testAdjustElectrons();


    typename SimpleLeadSelfEnergy<ComplexSparseMatrix>::Parameters params2;
    params2.alpha1 = 1;
    params2.t = 1;
    params2.mu = 0;

    params2.adjustMuToElectrons = false;
    params2.numberOfElectrons = 0;
    params2.muMax = 0;

    params2.hasBottom = false;
    params2.bottomOfBand = 0;

    params2.couplingPoints = ComplexDirectMatrix(couplingPoints,simpleLeadSelfEnergyTestBasis + BasisManager::Alpha_Block,enums::SpinSymmetry::RHF);;
    params2.magnetisationDir = Eigen::Vector3d({0,0,0.5});
    params2.upSpin = true;

    params2.MatrixDimension = 4; // TODO autodetermine


    ComplexSparseMatrix expectedSigmaRS = expectedSigmaR.sparseView();
    SimpleLeadSelfEnergyTests<ComplexSparseMatrix> se2(params2,expectedSigmaRS);
    ret += se2.test();
    ret += se2.testChangemu();
    ret += se2.testBottomOfBand();
    ret += se2.testChangeT();
    ret += se2.testAdjustElectrons();
    BasisManager::deleteBasis(simpleLeadSelfEnergyTestBasis);
    return ret > 0 ? -1 : 0;
}

