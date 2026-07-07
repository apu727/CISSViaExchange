#include "GreensFunctionTestHelper.h"
#include "interfaceselfenergy.h"

const static std::string InfiniteLeadMoleculeOrth = "InfMoleculeOrth";

numType t = -1;
numType U0 = 0;
numType mu = -3*t;

int runTightBindingInfiniteLeadTest(int numberOfSites, bool restrictedCalc, numType mu)
{
    logger().log(std::string("Running ") +std::string(__PRETTY_FUNCTION__) + std::string("with ") + std::to_string(numberOfSites) + " sites and restrictedCalc=" + std::to_string(restrictedCalc));
    int ret = 0;
    ComplexSelfAdjointSparseMatrix Ham = constructPeriodicHamilonian(numberOfSites,InfiniteLeadMoleculeOrth + std::to_string(numberOfSites),t,U0);
    Ham.coeffRef(0,2*(numberOfSites-1)) -= t;
    Ham.coeffRef(1,2*(numberOfSites-1)+1) -= t;
    Ham.coeffRef(2*(numberOfSites-1),0) -= t;
    Ham.coeffRef(2*(numberOfSites-1)+1,1) -= t;

    //Make self energy
    InterfaceSelfEnergy<SelfEnergyMatrix>::Parameters SEparams;

    SEparams.t = t;
    SEparams.mu = mu;
    SEparams.m = 0.5; //Only used for ContinuumInfinite. Means that the dispersion relations ~ match
    SEparams.couplingPoints.setZero(numberOfSites,numberOfSites,InfiniteLeadMoleculeOrth + std::to_string(numberOfSites) + BasisManager::Alpha_Block,enums::SpinSymmetry::RHF);
    SEparams.couplingPoints(0,0) = t;
    SEparams.m_type = InterfaceSelfEnergy<SelfEnergyMatrix>::LeadType::ContinuumInfinite;

    SEparams.MagnetisationDirection.setZero(); // not strictly needed

    auto SEL = std::make_shared<InterfaceSelfEnergy<SelfEnergyMatrix>>(SEparams);
    SEparams.couplingPoints(0,0) = 0;
    SEparams.couplingPoints(numberOfSites-1,numberOfSites-1) = t;
    auto SER = std::make_shared<InterfaceSelfEnergy<SelfEnergyMatrix>>(SEparams);

    GreensFunctionTest GF;
    GF.setup(Ham);


    GF.m_gfunc->setSelfEnergies({SEL,SER});

    GF.m_gfunc->setRestrictedCalculation(restrictedCalc);

    numType EnergyStart = 3*t;
    numType EnergyEnd = -3*t;
    size_t steps = 1000;
    ComplexSelfAdjointMatrix dest;
    std::vector<numType> energies;
    std::vector<numType> densities;
    for (size_t i = 0; i < steps; i++)
    {
        numType Energy = ((EnergyEnd-EnergyStart)/steps)*i + EnergyStart;
        GF.m_gfunc->getGnAtE(Energy,dest);
        densities.push_back(dest.trace()/(2*M_PI));
        energies.push_back(Energy);
    }
    logger().logAccurate("Energies",energies);
    logger().logAccurate("Densities",densities);

    return ret;
}

int infiniteLeadTests(int argc, char** argv)
{
    int ret = 0;
    BasisManager::getInstance().addOrthogonalBasis(InfiniteLeadMoleculeOrth + std::to_string(10),20);
    ret += runTightBindingInfiniteLeadTest(10,true,mu);

    return ret > 0? -1 : 0;
}
