#include "GreensFunctionTestHelper.h"
#include "TabulatedIntegrals.h"
#include "testutil.hpp"
#include "fockhamiltonian.h"
#include "scfsolver.h"
#include "simpleleadselfenergy.h"
#include "threadpool.h"
#include <future>
#include <utility>

//Adjusted for different types of integrators. Not pretty
double matrixNearTol = 1e-12;

static bool quickTestsOnly = false;

static std::string OrthBasis = "GreensFunctionTestsOrth";
static std::string NonOrthBasis = "GreensFunctionTestsNonOrth"; // The number of sites is appended to the end

numType HoppingStrength = -1;
numType onSiteEnergy = 0;
numType LeadGamma = 2; // divided by two elsewhere

numType CoulombInteractionStrength = 1;

static numType pseudoFermiFunction(numType E,numType mu)
{
    if ((E-mu)/(kb*300) > 30)
        return 0;
    else
        return 1.0/(exp((E-mu)/(kb*300)) + 1);
    expect(300.,T,__PRETTY_FUNCTION__,"Fermi Function Temperature wrong");
}

struct EigenVectorPairings
{
    complexType pole;
    EigenVector rightEigenVector;
    InvEigenVector rightInvEigenVector;
    int QN;
};
ComplexSelfAdjointSparseMatrix constructPeriodicHamilonian(int numberOfSites, const std::string& basis)
    {return constructPeriodicHamilonian(numberOfSites,basis,HoppingStrength,onSiteEnergy);}

ComplexSelfAdjointSparseMatrix constructPeriodicHamilonian(int numberOfSites, const std::string& basis, numType hoppingStrength, numType OnSiteEnergy)
{
    //Parameters
    ComplexSelfAdjointSparseMatrix ret(numberOfSites*2,numberOfSites*2,basis,enums::SpinSymmetry::RHF); // SO basis
    ret.setZero();
    for (int i = 0; i < numberOfSites; i++)
    {
        if (i >= 1)
        {
            ret.coeffRef(2*i,2*(i-1)) = hoppingStrength;
            ret.coeffRef(2*i+1,2*(i-1)+1) = hoppingStrength;
        }
        if (i < numberOfSites-1)
        {
            ret.coeffRef(2*i,2*(i+1)) = hoppingStrength;
            ret.coeffRef(2*i+1,2*(i+1)+1) = hoppingStrength;
        }
        ret.coeffRef(2*i,2*i) = OnSiteEnergy;
        ret.coeffRef(2*i+1,2*i+1) = OnSiteEnergy;
    }
    ret.coeffRef(0,2*(numberOfSites-1)) += hoppingStrength;
    ret.coeffRef(1,2*(numberOfSites-1)+1) += hoppingStrength;
    ret.coeffRef(2*(numberOfSites-1),0) += hoppingStrength;
    ret.coeffRef(2*(numberOfSites-1)+1,1) += hoppingStrength;

    return ret;
}

std::vector<complexType> getExpectedPoles(int numberOfSites, bool addBroadening)
{
    std::vector<complexType> poles;
    poles.reserve(numberOfSites*2);
    // QN is in [0,numberOfSites) - numberOfSites/2
    // note that numberOfSites/2 =/= numberOfSites-numberOfSites/2 for odd integers

    for (int QN = -numberOfSites/2; QN < numberOfSites - numberOfSites/2; QN++)
    {
        numType Energy = -2*std::cos(2*M_PI*((numType)QN/(numType)numberOfSites));
        if (addBroadening)
        {
            poles.push_back(Energy-iu*LeadGamma/2.);
            poles.push_back(Energy-iu*LeadGamma/2.);
        }
        else
        {
            poles.push_back(Energy);
            poles.push_back(Energy);
        }
    }
    expect(numberOfSites*2,(int)poles.size(),__PRETTY_FUNCTION__,"Poles Calculation wrong");
    return poles;
}

int checkEigenVectors(int numberOfSites, EigenVectorMatrix& foundRightEigenVectors, InvEigenVectorMatrix& foundRightInvEigenVectors, EigenValueVector &foundPoles, std::vector<EigenVectorPairings> &AnalyticEigenVectors)
{// Checks the EigenVectors and accounts for degeneracy and non-uniqueness
    int ret = 0;

    //Group degenerate poles
    std::vector<std::vector<EigenVectorPairings>> groups;

    for (int QN = -numberOfSites/2; QN <= 0; QN++)
    {
        complexType Energy = -2*std::cos(2*M_PI*((numType)QN/(numType)numberOfSites)) -iu*LeadGamma/2.;

        UnitaryEigenVector vecAlpha(2*numberOfSites,OrthBasis + std::to_string(numberOfSites),enums::SpinSymmetry::NoSpinSym);
        UnitaryEigenVector vecBeta(2*numberOfSites,OrthBasis + std::to_string(numberOfSites),enums::SpinSymmetry::NoSpinSym);
        vecAlpha.setZero();
        vecBeta.setZero();

        groups.push_back({});
        const int ks[] = {1,-1};
        for (auto s : ks)
        {
            for (int i = 0; i < numberOfSites; i++)
            {
                vecAlpha(2*i) = std::exp(iu*2.*M_PI*(numType)(QN*i*s)/(numType)numberOfSites);
                vecBeta(2*i+1) = vecAlpha(2*i);
            }
            vecAlpha/= sqrt(numberOfSites);
            vecBeta/= sqrt(numberOfSites);

            UnitaryInvEigenVector InvVecAlpha = vecAlpha.inverse();
            UnitaryInvEigenVector InvVecBeta = vecBeta.inverse();

            groups.back().push_back({Energy,static_cast<EigenVector>(vecAlpha),static_cast<InvEigenVector>(InvVecAlpha),QN});
            groups.back().push_back({Energy,static_cast<EigenVector>(vecBeta),static_cast<InvEigenVector>(InvVecBeta),QN});
            if (QN == 0)
                break;
            if ((numberOfSites % 2) == 0 && QN == -numberOfSites/2)
            {
                break;
            }
        }
    }



    std::vector<std::vector<EigenVectorPairings>> foundGroups;

    for (int i = 0; i < foundPoles.rows(); i++)
    {
        complexType pole = foundPoles(i);
        EigenVector vec = foundRightEigenVectors.getCol(i);
        InvEigenVector vecInv = foundRightInvEigenVectors.getRow(i);
        bool foundGroup = false;
        for (size_t j = 0; j < foundGroups.size(); j++)
        {
            if (abs(pole - foundGroups[j][0].pole) < 1e-13)
            {
                foundGroups[j].push_back({pole,vec,vecInv});
                foundGroup = true;
                break;
            }
        }
        if (!foundGroup)
        {
            foundGroups.push_back({{pole,vec,vecInv}});
        }
    }

    std::sort(groups.begin(),groups.end(),[](std::vector<EigenVectorPairings> &a, std::vector<EigenVectorPairings>& b)
              {
                  return realLessThan(a[0].pole,b[0].pole);
              });
    std::sort(foundGroups.begin(),foundGroups.end(),[](std::vector<EigenVectorPairings> a, std::vector<EigenVectorPairings> b)
              {
                  return realLessThan(a[0].pole,b[0].pole);
              });
    ret += expect(groups.size(),foundGroups.size(),__PRETTY_FUNCTION__,"Number of pole groups found is different");

    if (groups.size() != foundGroups.size())
        return ret;

    for (size_t i = 0; i < groups.size(); i++)
    {
        auto &group = groups[i];
        auto &foundGroup = foundGroups[i];

        //Mathematically the same as constructing:
        //Id = \sum_group group.right * group.InvRight.transpose()
        //foundId = \sum_foundgroup foundgroup.right * foundgroup.InvRight.transpose()
        //Check: foundId = Id * foundId * Id

        ComplexMatrix S1(group.size(),foundGroup.size(),"",enums::SpinSymmetry::NoSpin);
        ComplexMatrix S2(foundGroup.size(),group.size(),"",enums::SpinSymmetry::NoSpin);

        for (int i = 0; i < group.size(); i++)
        {
            for (int j = 0; j < foundGroup.size(); j++)
            {
                S1(i,j) = group[i].rightInvEigenVector * foundGroup[j].rightEigenVector;
                S2(j,i) = foundGroup[j].rightInvEigenVector * group[i].rightEigenVector;
            }
        }
        ComplexMatrix Id = (S1*S2);
        ret += expectNear(ComplexMatrix::Identity(group.size(),group.size(),"",enums::SpinSymmetry::NoSpin),Id,makeMatrixNearTol(Id,matrixNearTol),__PRETTY_FUNCTION__,"Check EigenSpace the same");
    }

    for (auto& group : groups)
    {
        for (auto& g : group)
            AnalyticEigenVectors.push_back(g);
    }

    return ret;
}

ComplexMatrix getGrAtE(complexType E, std::vector<EigenVectorPairings> EV, bool addBroadening)
{
    int numberOfSO = EV[0].rightEigenVector.rows();
    ComplexMatrix GR(numberOfSO,numberOfSO,OrthBasis + std::to_string(numberOfSO/2),enums::SpinSymmetry::RHF);
    GR.setZero();
    for (auto& ev : EV)
    {
        complexType pole = ev.pole;
        if (!addBroadening)
            pole = pole.real();
        const auto& rightEV = ev.rightEigenVector;
        const auto& invRightEV = ev.rightInvEigenVector;

        complexType factor = 1./(E-pole);
        GR += factor * rightEV * invRightEV;
    }
    return GR;
}

int getGnIntegral(int numberOfSites, ComplexSelfAdjointMatrix& gnIntegralOut, const std::vector<EigenVectorPairings>& EV,numType mu, bool addBroadening)
{
    const numType* integralTable = nullptr;
    bool found = false;
    for (auto& it : SiteIntegrals)
    {
        if (it.first == mu)
        {
            for (auto& it2 : it.second)
            {
                if (it2.first == numberOfSites)
                {
                    found = true;
                    integralTable = it2.second;
                    break;
                }
            }
            if (found)
                break;
        }
    }
    expect(true,found,__PRETTY_FUNCTION__,"not Found Integral Table for number of sites");
    ComplexMatrix gnIntegral;
    gnIntegral.setZero(2*numberOfSites,2*numberOfSites,OrthBasis + std::to_string(numberOfSites),enums::SpinSymmetry::RHF);
    if (!found)
        return 1;
    for (auto& E : EV)
    {
        numType Integral = integralTable[E.QN + numberOfSites/2];
        if (addBroadening)
            gnIntegral += Integral * E.rightEigenVector * E.rightInvEigenVector;
        else
            gnIntegral += 2*M_PI*pseudoFermiFunction(E.pole.real(),mu) * E.rightEigenVector * E.rightInvEigenVector;
    }
    gnIntegralOut = static_cast<ComplexSelfAdjointMatrix>(gnIntegral);
    return 0;
}

void makeMetric(int numberOfSites, ComplexMatrix& transform, const std::string& basisName, const std::string& newBasisName)
{
    //L^{i}_{\mu} Is transform where i is the orthogonal basis and \mu is the new basis
    //Transforms the Basis |e_i> via |e_\mu> = L^i_\mu |e_i>
    transform.resize(numberOfSites*2,numberOfSites*2,{basisName,newBasisName},enums::SpinSymmetry::RHF);
    transform.setZero();

    const numType DecayRate = 1;

    for (int i = 0; i < numberOfSites; i++)
    {
        for (int j = 0; j < numberOfSites; j++)
        {
            transform(2*j,2*i) = std::exp(-abs(i-j)/DecayRate);
            transform(2*j+1,2*i+1) = std::exp(-abs(i-j)/DecayRate);
        }
        transform.col(2*i).normalize();
        transform.col(2*i+1).normalize();
    }
    // transform.transposeInPlace();
    // inverseTransform = transform.inverse();
    //transform = L^{i}_{\mu} stored as L_{i\mu}
    // metric = transform.adjoint()*transform; // S_{\bar{\mu}\nu} = \bra{e_\mu}\ket{e_\nu} = <e_i|e_j> L^{\bar{i}}_{\bar{\mu}}L^{j}_{\nu} = L^\dagger L
}

int testCyclicBoundaryConditions(int numberOfSites, bool restrictedCalc, numType mu, const std::string& workingBasis,
                                 GreensFunction::integratorType integType, bool addBroadening)
{
    //transform = L^{i}_{\mu} stored as L_{i\mu}
    //c^\nu \ket{e_\nu} = c^i L^{\nu}_{i} {e_\nu}
    //h_{\bar{\mu}\nu} = L^{\bar{i}}_{\bar{\mu}} h_{\bar{i}j} L^{j}_{\nu}
    //h_{\bar{\mu}\nu} = L^\dagger h L
    //S_{\bar{\mu}\nu} = \bra{e_\mu}\ket{e_\nu} = L^{\bar{i}}_{\bar{\mu}}L^{i}_{\nu} = L^\dagger L

    // GR_{ij} = L^{\bar{\kappa}}_{\bar{i}}S_{\bar{\kappa},\mu}GR^{\mu}_{\nu}L^{\nu}{j}
    // GR_{ij} = L^{\bar{\kappa}}_{\bar{i}}L^{\bar{k}}_{\bar{\kappa}}L^{k}_{\mu}GR^{\mu}_{\nu}L^{\nu}_{j}
    // GR_{ij} = L^{i}_{\mu}GR^{\mu}_{\nu}L^{\nu}_{j}
    // GR_{ij} = L GR L^{-1}

    logger().log(std::string("Running ") +std::string(__PRETTY_FUNCTION__) + std::string("with ") + std::to_string(numberOfSites) + " sites and restrictedCalc=" + std::to_string(restrictedCalc));
    int ret = 0;
    ComplexSelfAdjointSparseMatrix Ham = constructPeriodicHamilonian(numberOfSites,OrthBasis + std::to_string(numberOfSites));
    ComplexSelfAdjointSparseMatrix HamTransformed;

    Ham.toBasisC(HamTransformed,workingBasis);

    //Make self energy
    SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters SEparams;

    SEparams.alpha1 = 1;
    SEparams.t = LeadGamma;
    SEparams.mu = mu;
    auto metric = BasisManager::getInstance().getMetric(workingBasis);
    metric.setSpinSym(enums::SpinSymmetry::RHF);
    metric.toBasisC(SEparams.couplingPoints,workingBasis+BasisManager::Alpha_Block);

    if (!addBroadening)
        SEparams.couplingPoints.setZero();
    SEparams.magnetisationDir.setZero();
    SEparams.upSpin = true;
    SEparams.MatrixDimension = numberOfSites*2;

    auto upSpinSE = std::make_shared<SimpleLeadSelfEnergy<SelfEnergyMatrix>>(SEparams);
    SEparams.upSpin = false;
    auto downSpinSE = std::make_shared<SimpleLeadSelfEnergy<SelfEnergyMatrix>>(SEparams);

    GreensFunctionTest GF;
    GF.setup(HamTransformed,integType);
    ComplexSelfAdjointMatrix HamFound = GF.m_gfunc->getHamiltonian();
    HamFound.toBasisC(HamFound,OrthBasis + std::to_string(numberOfSites));


    ret += expectNear(static_cast<ComplexSelfAdjointMatrix>(Ham),HamFound,makeMatrixNearTol(HamFound,matrixNearTol),__PRETTY_FUNCTION__,"ReadBackHamiltonian");
    GF.m_gfunc->setSelfEnergies({upSpinSE,downSpinSE});
    ret += expect(true,GF.m_gfunc->isConstant(false),__PRETTY_FUNCTION__,"isConstant1");
    ret += expect(true,GF.m_gfunc->isConstant(true),__PRETTY_FUNCTION__,"isConstant1");
    ret += expect({negInf,posInf},GF.m_gfunc->getAnalyticRegions(),__PRETTY_FUNCTION__,"getAnalyticRegions");
    ret += expect({upSpinSE,downSpinSE},GF.m_gfunc->getSelfEnergies(),__PRETTY_FUNCTION__,"getSelfEnergies");

    GF.m_gfunc->setRestrictedCalculation(restrictedCalc);
    ret += expect(restrictedCalc,GF.m_gfunc->getRestrictedCalculation(),__PRETTY_FUNCTION__,"getRestrictedCalculation");

    std::vector<complexType> polesExpt = getExpectedPoles(numberOfSites,addBroadening);
    std::vector<complexType> polesFound;
    GF.m_gfunc->getEstimateForDivergencesOfGr(0,polesFound);
    std::sort(polesExpt.begin(),polesExpt.end(),realLessThan);
    std::sort(polesFound.begin(),polesFound.end(),realLessThan);

    ret += expectNear(polesExpt,polesFound,vectorNear,__PRETTY_FUNCTION__,"Check Poles");

    EigenVectorMatrix rightEigenVectors;
    InvEigenVectorMatrix rightInvEigenVectors;
    EigenValueVector eigenValues;
    std::vector<EigenVectorPairings> analyticEigenVectors;

    GF.getEigenVectors(rightEigenVectors,rightInvEigenVectors,eigenValues);

    // (R^{-1})_{mi} = (R^{-1})_{m\mu} L^{\mu}_{i}
    // rightInvEigenVectors = rightInvEigenVectors * inverseTransform;
    rightInvEigenVectors.toBasisC(rightInvEigenVectors,OrthBasis + std::to_string(numberOfSites));
    // (R)^{i}_{m} = L^{i}_{\mu} (R)^{\mu}_{m}
    // rightEigenVectors = transform * rightEigenVectors;
    rightEigenVectors.toBasisC(rightEigenVectors,OrthBasis + std::to_string(numberOfSites));


    ret += checkEigenVectors(numberOfSites,rightEigenVectors,rightInvEigenVectors,eigenValues,analyticEigenVectors);





    ComplexMatrix GR;
    ComplexMatrix GRExpt;
    GF.m_gfunc->getGrAtE(mu-0.1,GR);
    GRExpt = getGrAtE(mu-0.1,analyticEigenVectors,addBroadening);
    // GR = transform * GR * inverseTransform;
    GR.toBasisC(GR,OrthBasis + std::to_string(numberOfSites));

    if (addBroadening)
        ret += expectNear(GR,GRExpt,makeMatrixNearTol(GRExpt,matrixNearTol),__PRETTY_FUNCTION__,"Check Gr at E=-0.1");
    else
    {//Need to check that mu-0.1 is far enough away from poles to not suffer problems with 1/~0
        bool allowed = true;
        for (auto p : polesExpt)
        {
            if (abs(mu-0.1-p.real()) < 0.02)
            {
                allowed = false;
                logger().log("Not checking Gr at E=mu-0.1 because there is a pole at", p.real());
                break;
            }
        }
        if (allowed)
            ret += expectNear(GR,GRExpt,makeMatrixNearTol(GRExpt,matrixNearTol),__PRETTY_FUNCTION__,"Check Gr at E=-0.1");

    }

    ComplexMatrix GA;
    GF.m_gfunc->getGaAtE(mu-0.1,GA);
    ComplexMatrix GAExpt;
    GAExpt = GR.adjoint();

    GA.toBasisC(GA,OrthBasis + std::to_string(numberOfSites));

    ret += expectNear(GA,GAExpt,makeMatrixNearTol(GAExpt,matrixNearTol),__PRETTY_FUNCTION__,"Check Ga at E=-0.1");

    if (addBroadening)
    {// GK == 0 except where it is infinite if there is no broadening
        ComplexMatrix GK;
        ComplexMatrix GKExpt;
        GF.m_gfunc->getGkAtE(mu-0.1,GK);
        GK.toBasisC(GK,OrthBasis + std::to_string(numberOfSites));

        GKExpt = GR - GA;
        GKExpt *= (1-2*pseudoFermiFunction(mu-0.1,mu));
        ret += expectNear(GKExpt,GK,makeMatrixNearTol(GKExpt,matrixNearTol),__PRETTY_FUNCTION__,"Check Gk at E=-0.1");

        ComplexMatrix SigmaK;
        {
            SelfEnergyMatrix temp;
            upSpinSE->getSigmaK(mu-0.1,temp);
            SigmaK = temp;
            downSpinSE->getSigmaK(mu-0.1,temp);
            SigmaK += temp;
        }
        SigmaK.toBasisC(SigmaK,OrthBasis + std::to_string(numberOfSites));
        ComplexMatrix GKExpt2 = GR * SigmaK * GA;
        ret += expectNear(GKExpt2,GK,makeMatrixNearTol(GKExpt2,matrixNearTol),__PRETTY_FUNCTION__,"Check Gk2 at E=-0.1");
    }

    ComplexSelfAdjointMatrix Gn;
    ComplexSelfAdjointMatrix GnExpt;
    GF.m_gfunc->getGnIntegral(100,Gn,false);
    ret += expect(workingBasis,Gn.getBasis()[0],__PRETTY_FUNCTION__,"Gn in wrong basis");
    Gn.toBasisC(Gn,OrthBasis + std::to_string(numberOfSites));

    getGnIntegral(numberOfSites,GnExpt,analyticEigenVectors,mu,addBroadening);
    //ret += expectNear(GnExpt,Gn,densityMatrixNearenums::SpinSymmetry<&myMatrixNear>,__PRETTY_FUNCTION__,"GnIntegral wrong");
    ret += expectNear(GnExpt,Gn,makeMatrixNearTol(GnExpt,matrixNearTol),__PRETTY_FUNCTION__,"GnIntegral wrong");
    logger().log("Gn Accuracy", (Gn-GnExpt).norm());
    GF.m_gfunc->setSelfEnergies({});
    return ret;
}

int NoCoulombTest()
{
    //Runs a simple test of Green's functions using cyclic boundary conditions. Tests various integrators
    //To Add: Perturbative self energy, regions, clearCache,
    //Doesn't test the Fock Hamiltonian along with SCF theory and DIIS, Doesnt test Energy
    int ret = 0;
    numType mu = 0;
    ComplexMatrix Metric;
    ComplexMatrix transform;
    ComplexMatrix inverseTransform;
    bool hasTransform = false;
    bool restrictedCalc = false;
    bool addBroadening = true; // Computes it without adding broadening, only viable for the equilibrium method
    GreensFunction::integratorType integrator = GreensFunction::integratorType::SumOverPolesEquilibrium;
    BasisManager::getInstance().deleteAllBasis();
    int quickTestMaxSize = 499;

    {
        const int sizes[] = {2,3,4,10,100,500,1000};
        for (auto& i : sizes)
        {
            if (quickTestsOnly && quickTestMaxSize < i)
                continue;
            ComplexMatrix NonOrthBasisTransform;

            BasisManager::getInstance().addOrthogonalBasis(OrthBasis + std::to_string(i),2*i);
            makeMetric(i,NonOrthBasisTransform,OrthBasis + std::to_string(i),NonOrthBasis + std::to_string(i));

            NonOrthBasisTransform.asBasisChangeMatrix();
        }
    }


    auto doLoop = [BY_REF_CAPTURE(quickTestMaxSize),
                   BY_REF_CAPTURE(hasTransform),
                   BY_REF_CAPTURE(ret),
                   BY_REF_CAPTURE(restrictedCalc),
                   BY_REF_CAPTURE(mu),
                   BY_REF_CAPTURE(integrator),
                   BY_REF_CAPTURE(addBroadening)
    ](int maxSize)
    {
        const int sizes[] = {2,3,4,10,100,500,1000};
        if (quickTestsOnly && quickTestMaxSize < maxSize)
            maxSize = quickTestMaxSize;
        for (auto& i : sizes)
        {
            if (maxSize < i)
                return;
            std::string workingBasis;
            if (hasTransform)
                workingBasis = NonOrthBasis + std::to_string(i);
            else
                workingBasis = OrthBasis + std::to_string(i);
            ret += testCyclicBoundaryConditions(i,restrictedCalc,mu,workingBasis,integrator, addBroadening);
            if (ret)
                return;
        }
    };
    auto doLoop2 = [BY_REF_CAPTURE(quickTestMaxSize),
                    BY_REF_CAPTURE(hasTransform),
                    BY_REF_CAPTURE(ret),
                    BY_REF_CAPTURE(restrictedCalc),
                    BY_REF_CAPTURE(mu),
                    BY_REF_CAPTURE(integrator),
                    BY_REF_CAPTURE(addBroadening),
                    BY_REF_CAPTURE(doLoop)
    ](int maxSize = 100000000)
    {
        // Not restricted Calc
        hasTransform = false;
        restrictedCalc = false;
        mu = 0;
        doLoop(maxSize);
        if (ret)
            return ret;
        mu = -0.1;
        doLoop(maxSize);
        if (ret)
            return ret;

        //Restricted Calc
        restrictedCalc = true;
        mu = 0;
        doLoop(maxSize);
        if (ret)
            return ret;
        mu = -0.1;
        doLoop(maxSize);
        if (ret)
            return ret;


        //Add a metric (but its the same problem)
        logger().log("#####################    Running with non-orthogonal basis");

        hasTransform = true;

        restrictedCalc = false;
        mu = 0;
        doLoop(maxSize);
        if (ret)
            return ret;
        mu = -0.1;
        doLoop(maxSize);
        if (ret)
            return ret;

        //Restricted Calc
        restrictedCalc = true;
        mu = 0;
        doLoop(maxSize);
        if (ret)
            return ret;
        mu = -0.1;
        doLoop(maxSize);
        if (ret)
            return ret;
        return ret;
    };

    // Equilibrium integrator without broadening
    logger().log("#####################    No broadening");
    addBroadening = false;
    doLoop2(1000);
    if (ret)
        return ret;

    //Equilibrium integrator with broadening
    logger().log("#####################    Broadening");
    addBroadening = true;
    doLoop2(1000);
    if (ret)
        return ret;

    //nonEquilibrium integrator
    logger().log("#####################    SumOverPolesNonEquilibrium");
    integrator = GreensFunction::integratorType::SumOverPolesNonEquilibrium;
    doLoop2(1000);
    if (ret)
        return ret;
    return ret;
    //Tricks
    logger().log("#####################    Tricks");
    integrator = GreensFunction::integratorType::Tricks;
    doLoop2(100);
    if (ret)
        return ret;

    //Tricks
    logger().log("#####################    NoAssumptions");
    integrator = GreensFunction::integratorType::NoAssumptions;
    double bkpmatrixNearTol = matrixNearTol;
    matrixNearTol = 5e-8;

    doLoop2(10);
    matrixNearTol = bkpmatrixNearTol;
    if (ret)
        return ret;
    return ret;
}


ComplexDirectSelfAdjointMatrix JKBuilder(ComplexDualSelfAdjointMatrix Density)
{ // Builds a J-K matrix for the given SO Density.
    int numberOfSites = Density.rows()/2;
    ComplexSelfAdjointMatrix retOrth(2*numberOfSites,2*numberOfSites,OrthBasis + std::to_string(numberOfSites),enums::SpinSymmetry::RHF);
    ComplexSelfAdjointMatrix DensityOrth;
    Density.toBasisC(DensityOrth,OrthBasis + std::to_string(numberOfSites));
    retOrth.setZero();
    // \sigma_{qs} = G^n_{rp}(h_{pqrs} - h_{qprs})
    // h_{pqrs} = Strength* 1/(|(q+s/2) - (p+r)/2|+1)

    const auto periodicDistance = [=](numType a, numType b)
    {
        numType dist = abs(a-b);
        if (dist > numberOfSites/2.)
            return numberOfSites - dist;
        else
            return dist;
    };
    //Okay to capture retOrth by ref as it is guaranteed to only have one reference
    auto work = [BY_REF_CAPTURE(retOrth),
                 BY_VAL_CAPTURE(numberOfSites),
                 CONST_REF_CAPTURE(periodicDistance),
                 CONST_REF_CAPTURE(DensityOrth)]
        (int q, int s)
    {
        for (int r = 0; r < numberOfSites; r++)
        {
            for (int p = 0; p < numberOfSites; p++)
            {
                numType val = CoulombInteractionStrength/(periodicDistance((q+s)/2.,(p+r)/2.)+1);
                //r&p must be same spin for coulomb term
                //q&s must be same spin for coulomb term

                //r&q must be same spin for Fock term
                //p&s must be same spin for Fock term
                // Chemist notation (AA|AA) + (AA|BB) + (BB|AA) + (BB|BB)

                retOrth(2*q,2*s) +=  val*DensityOrth(2*r,2*p);
                retOrth(2*q,2*s) +=  val*DensityOrth(2*r+1,2*p+1);
                retOrth(2*q+1,2*s+1) +=  val*DensityOrth(2*r,2*p);
                retOrth(2*q+1,2*s+1) +=  val*DensityOrth(2*r+1,2*p+1);
                val = 0.1*CoulombInteractionStrength/(periodicDistance((p+s)/2.,(q+r)/2.)+1);

                retOrth(2*q,2*s) -=  val*DensityOrth(2*r,2*p);
                retOrth(2*q,2*s+1) -=  val*DensityOrth(2*r,2*p+1);
                retOrth(2*q+1,2*s) -=  val*DensityOrth(2*r+1,2*p);
                retOrth(2*q+1,2*s+1) -=  val*DensityOrth(2*r+1,2*p+1);
            }
        }
    };
    std::vector<std::future<void>> futures;
    auto& pool = threadpool::getInstance(NUM_CORES);
    for (int q = 0; q < numberOfSites; q++)
    {
        for (int s = 0; s < numberOfSites; s++)
        {
            futures.push_back(pool.queueWork([=,&work](){work(q,s);}));
        }
    }
    for (auto& f : futures)
        f.wait();

    ComplexDirectSelfAdjointMatrix ret;
    retOrth.toBasisC(ret,Density.getBasis());
    return ret;
}

ComplexSelfAdjointMatrix loadDensity(int numberOfSites, numType mu)
{
    ComplexSelfAdjointMatrix ret(2*numberOfSites,2*numberOfSites,OrthBasis + std::to_string(numberOfSites),enums::SpinSymmetry::RHF);
    ret.setZero();
    const numType* dataptr = nullptr;

    for (auto& s : SiteDensities)
    {
        if (s.first == mu)
        {
            for (auto& s2 : s.second)
            {
                if (s2.first == numberOfSites)
                {
                    dataptr = s2.second;
                }
            }
        }
    }
    if (dataptr == nullptr)
    {
        expect(0,1,__PRETTY_FUNCTION__,"Could not load density");
        return ret;
    }
    for (int i = 0; i < 2*numberOfSites; i++)
    {
        for (int j = 0; j < 2*numberOfSites; j++)
        {
            ret(i,j) = complexType(dataptr[(i*2*numberOfSites + j)*2],dataptr[(i*2*numberOfSites + j)*2+1]);
        }
    }
    return ret;
}
int testCyclicBoundaryConditionsWithCoulomb(int numberOfSites, bool restrictedCalc, numType mu, std::string workingBasis,
                                 GreensFunction::integratorType integType, std::string filename)
{
    //transform = L^{i}_{\mu} stored as L_{i\mu}
    //c^\nu \ket{e_\nu} = c^i L^{\nu}_{i} {e_\nu}
    //h_{\bar{\mu}\nu} = L^{\bar{i}}_{\bar{\mu}} h_{\bar{i}j} L^{j}_{\nu}
    //h_{\bar{\mu}\nu} = L^\dagger h L
    //S_{\bar{\mu}\nu} = \bra{e_\mu}\ket{e_\nu} = L^{\bar{i}}_{\bar{\mu}}L^{i}_{\nu} = L^\dagger L

    // GR_{ij} = L^{\bar{\kappa}}_{\bar{i}}S_{\bar{\kappa},\mu}GR^{\mu}_{\nu}L^{\nu}{j}
    // GR_{ij} = L^{\bar{\kappa}}_{\bar{i}}L^{\bar{k}}_{\bar{\kappa}}L^{k}_{\mu}GR^{\mu}_{\nu}L^{\nu}_{j}
    // GR_{ij} = L^{i}_{\mu}GR^{\mu}_{\nu}L^{\nu}_{j}
    // GR_{ij} = L GR L^{-1}

    logger().log(std::string("Running ") +std::string(__PRETTY_FUNCTION__) + std::string("with ") + std::to_string(numberOfSites) + " sites and restrictedCalc=" + std::to_string(restrictedCalc));
    int ret = 0;
    ComplexSelfAdjointSparseMatrix Ham = constructPeriodicHamilonian(numberOfSites,OrthBasis + std::to_string(numberOfSites));
    ComplexSelfAdjointSparseMatrix HamTransformed;
    Ham.toBasisC(HamTransformed,workingBasis);

    //Make self energy
    SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters SEparams;

    SEparams.alpha1 = 1;
    SEparams.t = LeadGamma;
    SEparams.mu = mu;
    auto metric = BasisManager::getInstance().getMetric(workingBasis);
    metric.setSpinSym(enums::SpinSymmetry::RHF);
    metric.toBasisC(SEparams.couplingPoints,workingBasis+BasisManager::Alpha_Block);

    SEparams.magnetisationDir.setZero();
    SEparams.upSpin = true;
    SEparams.MatrixDimension = numberOfSites*2;

    auto upSpinSE = std::make_shared<SimpleLeadSelfEnergy<SelfEnergyMatrix>>(SEparams);
    SEparams.upSpin = false;
    auto downSpinSE = std::make_shared<SimpleLeadSelfEnergy<SelfEnergyMatrix>>(SEparams);

    std::shared_ptr<FockHamiltonian<SelfEnergyMatrix>> FHam;
    {
        ComplexDualSelfAdjointMatrix Zero1(2*numberOfSites,2*numberOfSites,OrthBasis + std::to_string(numberOfSites),enums::SpinSymmetry::RHF);
        Zero1.setZero();
        ComplexDirectSelfAdjointMatrix Zero2(2*numberOfSites,2*numberOfSites,OrthBasis + std::to_string(numberOfSites),enums::SpinSymmetry::RHF);
        Zero2.setZero();
        FHam = std::make_shared<FockHamiltonian<SelfEnergyMatrix>>(Zero1,Zero2,JKBuilder);
    }


    GreensFunctionTest GF;
    GF.setup(HamTransformed,integType);
    ComplexSelfAdjointMatrix HamFound = GF.m_gfunc->getHamiltonian();
    HamFound.toBasisC(HamFound,OrthBasis + std::to_string(numberOfSites));

    ret += expectNear(static_cast<ComplexSelfAdjointMatrix>(Ham),HamFound,makeMatrixNearTol(HamFound,matrixNearTol),__PRETTY_FUNCTION__,"ReadBackHamiltonian");
    FHam->setGreensFunction(GF.m_gfunc);
    GF.m_gfunc->setSelfEnergies({FHam,upSpinSE,downSpinSE});
    ret += expect(true,GF.m_gfunc->isConstant(false),__PRETTY_FUNCTION__,"isConstant1");
    ret += expect(true,GF.m_gfunc->isConstant(true),__PRETTY_FUNCTION__,"isConstant1");
    ret += expect({negInf,posInf},GF.m_gfunc->getAnalyticRegions(),__PRETTY_FUNCTION__,"getAnalyticRegions");
    ret += expect({FHam,upSpinSE,downSpinSE},GF.m_gfunc->getSelfEnergies(),__PRETTY_FUNCTION__,"getSelfEnergies");

    GF.m_gfunc->setRestrictedCalculation(restrictedCalc);
    ret += expect(restrictedCalc,GF.m_gfunc->getRestrictedCalculation(),__PRETTY_FUNCTION__,"getRestrictedCalculation");

    // logger::silence();
    {
        FHam->resetDIISCache();
        SCFSolver SCF;
        SCF.setGreensFunction(GF.m_gfunc);
        SCF.setSelfConsistentSelfEnergy({FHam});
        if (numberOfSites >= 100)
            FHam->setUseEDIIS(true);
        long itercount = 0;
        // logger::silence();
        SCF.run([CONST_REF_CAPTURE(FHam),BY_REF_CAPTURE(itercount),BY_VAL_CAPTURE(numberOfSites)](){
            itercount++;
            logger().log("error estimate:", FHam->getErrorEstimate());
            return FHam->getErrorEstimate() < 5e-12 && ( itercount > 100 || numberOfSites > 2); // the two site version always has the correct eigenspace since this is fully determined by symmetry. Therefore CDIIS predicts too low of an error
        },10000);
    }
    // logger::unsilence();

    ComplexSelfAdjointMatrix Gn;
    GF.m_gfunc->getGnIntegral(100,Gn,false);
    Gn.toBasisC(Gn,OrthBasis + std::to_string(numberOfSites));

    // ComplexSelfAdjointMatrix GnExpt = loadDensity(numberOfSites,mu);
    // ret += expectNear(GnExpt,Gn,makeMatrixNearTol(Gn,matrixNearTol),__PRETTY_FUNCTION__,"Gn integral wrong");

    logger(filename).logAccurate("Resultant Density", Gn);
    // logger().log("Gn error:", (Gn-GnExpt).norm());
    // logger().log("Gn Trace error:", Gn.trace()-GnExpt.trace());
    logger().log("error estimate:", FHam->getErrorEstimate());
    GF.m_gfunc->setSelfEnergies({});
    return ret;
}
int CoulombTest()
{//TODO with an overlap matrix & different integrators?


    int ret = 0;
    numType mu = 0;
    ComplexMatrix Metric;
    ComplexMatrix transform;
    ComplexMatrix inverseTransform;
    bool hasTransform = false;
    bool restrictedCalc = false;
    GreensFunction::integratorType integrator = GreensFunction::integratorType::SumOverPolesEquilibrium;
    double bkpMatrixNearTol = matrixNearTol;
    matrixNearTol = 5e-8;
    BasisManager::getInstance().deleteAllBasis();
    int quickTestMaxSize = 499;

    {
        const int sizes[] = {2,3,4,10,100/*,500,1000*/};
        for (auto& i : sizes)
        {
            if (quickTestsOnly && quickTestMaxSize < i)
                continue;
            ComplexMatrix NonOrthBasisTransform;

            BasisManager::getInstance().addOrthogonalBasis(OrthBasis + std::to_string(i),2*i);
            makeMetric(i,NonOrthBasisTransform,OrthBasis + std::to_string(i),NonOrthBasis + std::to_string(i));

            NonOrthBasisTransform.asBasisChangeMatrix();
        }
    }

    auto doLoop = [BY_REF_CAPTURE(quickTestMaxSize),
                   BY_REF_CAPTURE(hasTransform),
                   BY_REF_CAPTURE(ret),
                   BY_REF_CAPTURE(restrictedCalc),
                   BY_REF_CAPTURE(mu),
                   BY_REF_CAPTURE(integrator)
    ](int maxSize)
    {
        const int sizes[] = {2,3,4,10,100,500,1000};
        if (quickTestsOnly && quickTestMaxSize < maxSize)
            maxSize = quickTestMaxSize;
        for (auto& i : sizes)
        {
            if (quickTestsOnly && i == 2)
                continue;
            if (maxSize < i)
                return;

            std::string workingBasis;
            if (hasTransform)
                workingBasis = NonOrthBasis + std::to_string(i);
            else
                workingBasis = OrthBasis + std::to_string(i);
            std::string filename = std::to_string(i) + std::string("SitesMu") + (mu == 0 ? std::string("0") : std::string("m0p1")) + std::string("Coulomb.txt");
            ret += testCyclicBoundaryConditionsWithCoulomb(i,restrictedCalc,mu,workingBasis,integrator,filename);
            if (ret)
                return;
        }
    };
    auto doLoop2 = [BY_REF_CAPTURE(quickTestMaxSize),
                    BY_REF_CAPTURE(hasTransform),
                    BY_REF_CAPTURE(ret),
                    BY_REF_CAPTURE(restrictedCalc),
                    BY_REF_CAPTURE(mu),
                    BY_REF_CAPTURE(integrator),
                    BY_REF_CAPTURE(doLoop)](int maxSize = 100000000)
    {
        // //Not restricted Calc
        // // restrictedCalc = false;
        // // mu = 0;
        // // doLoop(maxSize);
        // // if (ret)
        // //     return ret;
        // // mu = -0.1;
        // // doLoop(maxSize);
        // // if (ret)
        // //     return ret;

        //Restricted Calc
        hasTransform = false;
        restrictedCalc = true;
        mu = 0;
        doLoop(maxSize);
        if (ret)
            return ret;
        mu = -0.1;
        doLoop(maxSize);
        if (ret)
            return ret;


        //Add a metric (but its the same problem)
        logger().log("Running with non-orthogonal basis");

        hasTransform = true;

        // restrictedCalc = false;
        // mu = 0;
        // doLoop(maxSize);
        // if (ret)
        //     return ret;
        // mu = -0.1;
        // doLoop(maxSize);
        // if (ret)
        //     return ret;

        //Restricted Calc
        restrictedCalc = true;
        mu = 0;
        doLoop(maxSize);
        if (ret)
            return ret;
        mu = -0.1;
        doLoop(maxSize);
        if (ret)
            return ret;
        return ret;
    };

    //Equilibrium integrator
    doLoop2(100);
    if (ret)
        goto end;

    //nonEquilibrium integrator
    integrator = GreensFunction::integratorType::SumOverPolesNonEquilibrium;
    doLoop2(100);
    if (ret)
        goto end;

    //Tricks
    integrator = GreensFunction::integratorType::Tricks;
    doLoop2(10);
    if (ret)
        goto end;

    //Tricks
    integrator = GreensFunction::integratorType::NoAssumptions;


    doLoop2(10);

end://ugly
    bkpMatrixNearTol = bkpMatrixNearTol;
    if (ret)
        return ret;
    return ret;
}
int greensfunctionTests(int argc, char** argv)
{
    char* quickTestPtr = std::getenv("QUICKTESTS");
    if (quickTestPtr && *quickTestPtr == 'N')
        return -1;
    else if (quickTestPtr && *quickTestPtr != '0')
    {
        logger().log("QuickTests only");
        quickTestsOnly = true;
    }
    BasisManager::getInstance().deleteAllBasis();
    int ret = 0;
    ret += NoCoulombTest();
    if (ret)
        return ret > 0 ? -1 : 0;
    ret += CoulombTest();

    return ret > 0 ? -1 : 0;
}
