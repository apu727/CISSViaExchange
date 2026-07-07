#include "greensfunction.h"
#include "complexmatrixbuffer.h"
#include "fockhamiltonian.h"
#include "integrator.h"
#include "logger.h"
#include "sopgreensfunction.h"
#include <Eigen/LU>
#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>
#include "Eigen/Eigenvalues"


const bool useDiagonalOfGreensFunc = false;
const complexType phononBroadening = 0;//-0.00036749322175664445*iu;//-0.25*iu;


// const bool allowAnalyticGROptimisation = true;

const bool computeIntegrationNumericallyAlways = false;
const bool finiteTemperatureSum = true; // Chooses whether the Analytic expressions take temperature into account.

const long numResidues = 10./(kb*T); // How many residues of the fermi-function to sum in the sum over poles method. Energy Error is O(1/E) where E is the largest pole that has been summed. E.g. 10^-2 for numResidues = 10/kT

const bool testIntegrator = false; //always does a numerical integral and compares the results
const bool testAgainstNumericalIntegrators = false;

const bool useGK = true;
static numType fermiFunction(numType E, numType mu)
{
    if ((E-mu)/(kb*T) > 40)
        return 0;
    else
        return 1.0/(exp((E-mu)/(kb*T)) + 1);


}

static complexType fermiFunction(complexType E, numType mu)
{
    if ((real(E)-mu)/(kb*T) > 30)
        return 0;
    else
        return 1.0/(exp((E-mu)/(kb*T)) + 1.);


}


void GreensFunction::setupAnalyticRegions()
{
    bool selfEnergiesAreAnalytic = true; //In the complex number sense
    bool selfEnergiesAreConstant = true; //Does it vary with E;
    bool selfEnergiesArePieceWiseConstant = true; //Is it piecewise constant

    bool perturbativeSelfEnergiesAreAnalytic = true; //In the complex number sense
    bool perturbativeSelfEnergiesAreConstant = true; //Does it vary with E;
    bool perturbativeSelfEnergiesArePieceWiseConstant = true; //Is it piecewise constant


    m_analyticRegions.clear();
    std::vector<numType> analyticRegions;
    std::vector<numType> perturbativeAnalyticRegions;
    if (m_selfEnergies.size() != 0)
    {
        for (auto& SE : m_selfEnergies)
        {
            SE->setBasis(m_basis);

            if (!SE->isAnalytic())
            {
                selfEnergiesAreAnalytic = false;
                selfEnergiesAreConstant = false;
                selfEnergiesArePieceWiseConstant = false;
                //Cant break because we need to set the basis for all self energies
            }
            selfEnergiesAreConstant = selfEnergiesAreConstant && SE->isConstant();
            selfEnergiesArePieceWiseConstant = selfEnergiesArePieceWiseConstant && SE->isConstantPiecewise();

            auto region = SE->getContinuousRegions();
            for (auto& r : region)
                analyticRegions.push_back(r);

        }
        perturbativeAnalyticRegions = analyticRegions;

        perturbativeSelfEnergiesAreAnalytic = selfEnergiesAreAnalytic;
        perturbativeSelfEnergiesAreConstant = selfEnergiesAreConstant;
        perturbativeSelfEnergiesArePieceWiseConstant = selfEnergiesArePieceWiseConstant;

        for (auto& SE : m_perturbativeSelfEnergies)
        {
            SE->setBasis(m_basis);

            if (!SE->isAnalytic())
            {
                perturbativeSelfEnergiesAreAnalytic = false;
                perturbativeSelfEnergiesAreConstant = false;
                perturbativeSelfEnergiesArePieceWiseConstant = false;
                //Cant break because we need to set the basis for all self energies
            }
            perturbativeSelfEnergiesAreConstant = perturbativeSelfEnergiesAreConstant && SE->isConstant();
            perturbativeSelfEnergiesArePieceWiseConstant = perturbativeSelfEnergiesArePieceWiseConstant && SE->isConstantPiecewise();
            auto region = SE->getContinuousRegions();
            for (auto& r : region)
                perturbativeAnalyticRegions.push_back(r);
        }
        {
            std::sort(analyticRegions.begin(),analyticRegions.end());
            // deal with duplicates
            for (size_t i = 0; i < analyticRegions.size(); i++)
            {
                if (i+1 < analyticRegions.size() && (subtractWithInfinities(analyticRegions[i+1], analyticRegions[i]) < 1e-10))
                    continue;
                m_analyticRegions.push_back(analyticRegions[i]);
            }
        }


        //Currently unused
        {
            std::sort(analyticRegions.begin(),analyticRegions.end());
            // deal with duplicates
            for (size_t i = 0; i < analyticRegions.size(); i++)
            {
                if (i+1 < analyticRegions.size() && (subtractWithInfinities(analyticRegions[i+1], analyticRegions[i]) < 1e-10))
                    continue;
                m_perturbativeAnalyticRegions.push_back(analyticRegions[i]);
            }
        }
    }
    else
    {
        m_analyticRegions = {negInf,posInf};
    }

    if (selfEnergiesAreConstant)
    {
        if (m_GrComputationMethod == GrComputationType::Subsystem)
            m_subSystemComputationMethod = GrComputationType::AnalyticSOP;
        else
            m_GrComputationMethod = GrComputationType::AnalyticSOP;
    }
    else if (selfEnergiesArePieceWiseConstant)
    {
        if (m_GrComputationMethod == GrComputationType::Subsystem)
            m_subSystemComputationMethod = GrComputationType::AnalyticSOPRegions;
        else
            m_GrComputationMethod = GrComputationType::AnalyticSOPRegions;
    }
    else if (selfEnergiesAreAnalytic)
    {
        if (m_GrComputationMethod == GrComputationType::Subsystem)
            m_subSystemComputationMethod = GrComputationType::Analytic;
        else
            m_GrComputationMethod = GrComputationType::Analytic;
    }
    else
    {
        if (m_GrComputationMethod == GrComputationType::Subsystem)
            m_subSystemComputationMethod = GrComputationType::RealAxis;
        else
            m_GrComputationMethod = GrComputationType::RealAxis;
    }


    if (perturbativeSelfEnergiesAreConstant)
    {
        m_perturbativeGrComputationMethod = GrComputationType::AnalyticSOP;
    }
    else if (perturbativeSelfEnergiesArePieceWiseConstant)
    {
        m_perturbativeGrComputationMethod = GrComputationType::AnalyticSOPRegions;
    }
    else if (perturbativeSelfEnergiesAreAnalytic)
    {
        m_perturbativeGrComputationMethod = GrComputationType::Analytic;
    }
    else
        m_perturbativeGrComputationMethod = GrComputationType::RealAxis;
    m_perturbativeGrComputationMethod = GrComputationType::RealAxis; //Disabled For now
    if (m_analyticRegions.size() > 0)
        logger().log("Number of regions", m_analyticRegions.size()-1);
    if (m_perturbativeAnalyticRegions.size() > 0)
        logger().log("Number of Perturbative regions", m_perturbativeAnalyticRegions.size()-1);
}

bool GreensFunction::findIndexOfRegion(complexType E, size_t &index) const
{
    //TODO perturbative regions, currently not needed
    if (E == negInf)
    {
        if (m_analyticRegions[0] == negInf)
        {
            index = 0;
            return true;
        }
        else
            return false;
    }
    if (E == posInf)
    {
        if (m_analyticRegions.back() == posInf)
        {
            index = m_analyticRegions.size()-1;
            return true;
        }
        else
            return false;
    }
    for (size_t i = 0; i < m_analyticRegions.size()-1; i++)
    {
        if (m_analyticRegions[i] <= E.real() && m_analyticRegions[i+1] >= E.real())
        {
            index = i;
            return true;
        }
        // if (E.real() == 0 && m_analyticRegions[i] <= E.real() && m_analyticRegions[i+1] >= E.real())
        // {//Include the boundaries if E = 0 since it is such a common number. Otherwise we dont expect to run into this. 0 is used throughout the code as a throwaway number
        //     index = i;
        //     return true;
        // }
    }
    logger().log("Failed to find region with E",E.real());
    return false; // No such region. In principle this is ridiculous as posInf should be the last one.
}

void GreensFunction::buildSOP() const
{
    std::lock_guard<std::mutex> lock(m_cacheMutex);
    if (!m_SOPGIsBuilt)
    {
        if (m_GrComputationMethod == GrComputationType::Subsystem)
        {
            numType mu;
            {
                numType muMin=1e20;
                numType muMax=-1e20;
                bool foundAMu = false;
                for (auto SE : m_selfEnergies)
                {
                    if (SE->hasSigmaK())
                    {
                        foundAMu = true;
                        numType mu = SE->getChemicalPotential();

                        muMax = std::max(mu,muMax);
                        muMin = std::min(mu,muMin);
                    }
                }
                if (!foundAMu)
                    muMin = muMax = m_mu;
                releaseAssert(abs(muMax-muMin) < 1e-10,"Subsystem must be equilibrium for now");
                mu = muMax;
            }

            if (isSet(m_SubSystemOpt,SubsystemFlags::AOPartitioning))
            {
                releaseAssert(false,"AOPartitioning");
                // ComplexMatrix EIDD;

                // EIDD.setIdentity(m_Ham.rows(),m_Ham.cols(),m_basis,m_spinSym);

                // EIDD *= (E+iu*1e-30);

                // EIDD -= m_Ham;
                // SelfEnergyMatrix temp;
                // for (auto SE : m_selfEnergies)
                // {
                //     SE->getSigmaR(E,temp);
                //     EIDD -= temp;
                // }
                // ComplexDualMatrix GMM,GML,GLM,GLL,GMMBare,GLLBare;
                // ComplexDirectMatrix HMM,HML,HLM,HLL;
                // //converts first and then projects
                // EIDD.toBasisC(HMM,{m_subsystemBases[0],m_subsystemBases[0]});
                // EIDD.toBasisC(HML,{m_subsystemBases[0],m_subsystemBases[1]});
                // EIDD.toBasisC(HLM,{m_subsystemBases[1],m_subsystemBases[0]});
                // EIDD.toBasisC(HLL,{m_subsystemBases[1],m_subsystemBases[1]});

                // GMMBare = HMM.inverse();
                // GLLBare = HLL.inverse();
                // if (isSet(m_SubSystemOpt,SubsystemFlags::FreezeMol))
                //     GMM = GMMBare;
                // else
                //     GMM = (HMM - HML*GLLBare*HLM).inverse();

                // if (isSet(m_SubSystemOpt,SubsystemFlags::FreezeLead))
                //     GLL = GLLBare;
                // else
                //     GLL = (HLL - HLM*GMMBare*HML).inverse();

                // if (!isSet(m_SubSystemOpt,SubsystemFlags::IgnoreGLM))
                // {
                //     GML = GMMBare*HML*GLL;
                //     GLM = GLLBare*HLM*GMM;
                //     GML*=-1;
                //     GLM*=-1;
                // }

                // ComplexMatrix GMMf,GMLf,GLMf,GLLf;
                // GMM.toBasisC(GMMf,m_basis);
                // GLL.toBasisC(GLLf,m_basis);
                // dest = GMMf;
                // dest += GLLf;

                // if (!isSet(m_SubSystemOpt,SubsystemFlags::IgnoreGLM))
                // {
                //     GML.toBasisC(GMLf,m_basis);
                //     GLM.toBasisC(GLMf,m_basis);
                //     dest += GMLf;
                //     dest += GLMf;
                // }
            }
            else
            {
                ComplexMatrix EI = static_cast<ComplexMatrix>(m_Ham);
                SelfEnergyMatrix temp;
                for (auto SE : m_selfEnergies)
                {
                    SE->getSigmaR(temp);
                    EI += temp;
                }
                std::shared_ptr<SOPGreensFunction> GMMBare,GLLBare;
                ComplexMatrix HMM,HML,HLM,HLL;
                ComplexMatrix HMMAdj,HMLAdj,HLMAdj,HLLAdj;
                EI.toBasisC(HMM,{m_subsystemBases[0],m_subsystemBases[0]});
                EI.toBasisC(HML,{m_subsystemBases[0],m_subsystemBases[1]});
                EI.toBasisC(HLM,{m_subsystemBases[1],m_subsystemBases[0]});
                EI.toBasisC(HLL,{m_subsystemBases[1],m_subsystemBases[1]});

                ComplexMatrix EIAdj = EI.adjoint();
                EIAdj.toBasisC(HMMAdj,{m_subsystemBases[0],m_subsystemBases[0]});
                EIAdj.toBasisC(HMLAdj,{m_subsystemBases[0],m_subsystemBases[1]});
                EIAdj.toBasisC(HLMAdj,{m_subsystemBases[1],m_subsystemBases[0]});
                EIAdj.toBasisC(HLLAdj,{m_subsystemBases[1],m_subsystemBases[1]});

                GMMBare = std::make_shared<SOPGreensFunction>(HMM,HMMAdj,mu,m_analyticRegions,m_enforcedSpinSym);

                GLLBare = std::make_shared<SOPGreensFunction>(HLL,HLLAdj,mu,m_analyticRegions,m_enforcedSpinSym);
                if (isSet(m_SubSystemOpt,SubsystemFlags::FreezeMol))
                    m_GMM = GMMBare;
                else
                {
                    m_GMM = std::make_shared<SOPGreensFunction>(EI,mu,m_analyticRegions,m_enforcedSpinSym);
                    m_GMM->setDesiredBasis({m_subsystemBases[0],m_subsystemBases[0]});
                }
                if (isSet(m_SubSystemOpt,SubsystemFlags::FreezeLead))
                    m_GLL = GLLBare;
                else
                {
                    m_GLL = std::make_shared<SOPGreensFunction>(EI,mu,m_analyticRegions,m_enforcedSpinSym);
                    m_GLL->setDesiredBasis({m_subsystemBases[1],m_subsystemBases[1]});
                }

                if (!isSet(m_SubSystemOpt,SubsystemFlags::IgnoreGLM))
                {
                    // m_GML = std::make_shared<SOPGreensFunction>(*GMMBare* HML* *m_GLL); //Old method leads to inconsistent G_{ML} and G_{LM} although the results are ok.
                    m_GML = std::make_shared<SOPGreensFunction>(*m_GMM* HML* *GLLBare); // Does this work when we do it exactly too? about to find out.
                    m_GLM = std::make_shared<SOPGreensFunction>(*GLLBare* HLM* *m_GMM);
                }
                /*
                m_GDEBUG = std::make_shared<SOPGreensFunction>(m_Ham,m_selfEnergies,m_analyticRegions,m_enforcedSpinSym);
                {
                    ComplexMatrix GrExpt,GLL,GMM,GLM,GML,GrFound,M;
                    m_GDEBUG->getGr(-0.4,GrExpt);
                    m_GLL->getGr(-0.4,GLL);
                    m_GLM->getGr(-0.4,GLM);
                    m_GML->getGr(-0.4,GML);
                    m_GMM->getGr(-0.4,GMM);


                    GLL.toBasisC(GLL,m_basis);
                    GLM.toBasisC(GLM,m_basis);
                    GML.toBasisC(GML,m_basis);
                    GMM.toBasisC(GMM,m_basis);
                    GrFound = GLL + GLM + GML + GMM;
                    M = GrExpt-GrFound;
                    numType error = M.norm();
                    logger().log("Error",error);
                }*/
            }
        }
        else
        {
            m_GMM = std::make_shared<SOPGreensFunction>(m_Ham,m_selfEnergies,m_analyticRegions,m_enforcedSpinSym);
        }
        m_SOPGIsBuilt = true;
    }

}

numType GreensFunction::getRegionRepresentativeEnergy(size_t regionIndex) const
{
    assert(regionIndex < m_analyticRegions.size()-1);

    numType regionEnergy = (m_analyticRegions[regionIndex] + m_analyticRegions[regionIndex+1])/2.;
    if (m_analyticRegions[regionIndex] == negInf)
        regionEnergy = m_analyticRegions[regionIndex+1]-1;
    if (m_analyticRegions[regionIndex+1] == posInf)
        regionEnergy = m_analyticRegions[regionIndex]+1;
    if (m_analyticRegions[regionIndex+1] == posInf && m_analyticRegions[regionIndex] == negInf)
        regionEnergy = 0;
    return regionEnergy;
}


GreensFunction::GreensFunction() {}

void GreensFunction::setHamiltonian(std::shared_ptr<HamiltonianBase> H)
{
    m_Ham = H->getHamMatrix();
    m_basis = m_Ham.getBasisS();
    logger().log("Setting GFunc working basis to",m_basis);
}

void GreensFunction::setSubsystems(const std::vector<std::string>& subsystems,SubsystemFlags flags)
{
    releaseAssert(subsystems.size() == 2,"subsystems.size() == 2");
    m_GrComputationMethod = GrComputationType::Subsystem;
    m_subsystemBases=subsystems;
    m_SubSystemOpt = flags;
}

bool GreensFunction::isConstant(bool perturb) const
{
    if (perturb && m_perturbativeSelfEnergies.size() > 0)
        return m_perturbativeGrComputationMethod == GrComputationType::AnalyticSOP || m_perturbativeGrComputationMethod == GrComputationType::AnalyticSOPRegions;
    else
        return (m_GrComputationMethod == GrComputationType::AnalyticSOP || m_GrComputationMethod == GrComputationType::AnalyticSOPRegions) ||
               (m_GrComputationMethod == GrComputationType::Subsystem && (m_subSystemComputationMethod == GrComputationType::AnalyticSOP || m_subSystemComputationMethod == GrComputationType::AnalyticSOPRegions));

}

void GreensFunction::clearCache()
{
    m_SOPGIsBuilt = false;
    m_GMM = m_GLL = m_GLM = m_GML = nullptr;
    m_GnIsCached = false;
    m_EGnIsCached = false;
    m_GnPerturbIsCached = false;
    m_EGnPerturbIsCached = false;
    m_analyticRegions.clear();
    m_perturbativeAnalyticRegions.clear();

    if (m_GrComputationMethod == GrComputationType::Subsystem)
        m_subSystemComputationMethod = GrComputationType::None;
    else
        m_GrComputationMethod = GrComputationType::None;
    m_perturbativeGrComputationMethod = GrComputationType::None;
    setupAnalyticRegions();
}

void GreensFunction::getEstimateForDivergencesOfGr(numType E, std::vector<complexType> &divergences) const
{
    if (isConstant(false) && !BasisManager::isProjection(m_basis) /*This is only an estimate so if non-perturb returns true there is a good chance this will work*/)
    {
        if (!m_SOPGIsBuilt)
            buildSOP();
        m_GMM->getDivergencesOfGr(E,divergences);
        return;
    }
    ComplexMatrix EI;
    EI.setIdentity(m_Ham.rows(),m_Ham.cols(),m_basis,m_spinSym);

    EI *= -(iu*1e-30);

    EI += m_Ham;
    SelfEnergyMatrix temp;
    for (auto SE : m_selfEnergies)
    {
        SE->getSigmaR(E,temp);
        EI += temp;
    }

    EigenValueVector eigenValues;
    EI.getEigenValues(eigenValues);
    divergences.clear();
    for (long i = 0; i < eigenValues.rows(); i++)
    {
        divergences.push_back(eigenValues(i));
        if (eigenValues(i).imag() > 1e-10)
            logger().log("Pole has Imaginary eigenvalue greater than 0");
    }

}

void GreensFunction::getGrAtE(complexType E, ComplexMatrix& dest,bool perturb) const
{

    if (isConstant(false) && !BasisManager::isProjection(m_basis) )
    {
        if (!m_SOPGIsBuilt) buildSOP();

        if (m_GrComputationMethod == GrComputationType::Subsystem)
        {
            ComplexMatrix GMM, GML, GLM, GLL;
            m_GMM->getGr(E,GMM);
            m_GLL->getGr(E,GLL);
            GMM.toBasisC(GMM,m_basis);
            GLL.toBasisC(GLL,m_basis);
            dest = GMM;
            dest += GLL;
            if (!isSet(m_SubSystemOpt,SubsystemFlags::IgnoreGLM))
            {
                m_GLM->getGr(E,GLM);
                m_GML->getGr(E,GML);
                GML.toBasisC(GML,m_basis);
                GLM.toBasisC(GLM,m_basis);
                dest += GML;
                dest += GLM;
            }
        }
        else
        {
            m_GMM->getGr(E,dest);
        }
        if (!perturb)
            return;
    }
    else
    {
        ComplexMatrix EI;

        EI.setIdentity(m_Ham.rows(),m_Ham.cols(),m_basis,m_spinSym);

        EI *= (E+iu*1e-30);

        EI -= m_Ham;
        SelfEnergyMatrix temp;
        for (auto SE : m_selfEnergies)
        {
            SE->getSigmaR(E,temp);
            EI -= temp;
        }

        dest = EI.inverse();
    }

    if (perturb && m_perturbativeSelfEnergies.size() != 0)
    {
        ComplexMatrix EI;
        EI.setZero(m_Ham.rows(),m_Ham.cols(),m_basis,m_spinSym);
        MatrixBufferObject<SelfEnergyMatrix> temp;

        for (auto SE : m_perturbativeSelfEnergies)
        {
            SE->getSigmaR(E,temp);
            EI += temp.get();
        }
        // GR = GR_0 + GR_0 \SigmaR GR_0
        // ComplexMatrixBufferObject temp2;
        // ComplexMatrixBufferObject temp3;
        // temp2 = EI*dest;
        // temp3 = dest * temp2;
        // dest += temp3;

        //Can't save a temporary because this aliases
        dest += dest * EI * dest;
    }
}

void GreensFunction::getGaAtE(complexType E, ComplexMatrix& dest, bool perturb) const
{
    if (m_GrComputationMethod == GrComputationType::Subsystem)
    {
        if (isConstant(false) && !BasisManager::isProjection(m_basis) )
        {
            if (!m_SOPGIsBuilt) buildSOP();

            if (m_GrComputationMethod == GrComputationType::Subsystem)
            {
                ComplexMatrix GMM, GML, GLM, GLL;
                m_GMM->getGa(E,GMM);
                m_GLL->getGa(E,GLL);
                GMM.toBasisC(GMM,m_basis);
                GLL.toBasisC(GLL,m_basis);
                dest = GMM;
                dest += GLL;
                if (!isSet(m_SubSystemOpt,SubsystemFlags::IgnoreGLM))
                {
                    m_GLM->getGa(E,GLM);
                    m_GML->getGa(E,GML);
                    GML.toBasisC(GML,m_basis);
                    GLM.toBasisC(GLM,m_basis);
                    dest += GML;
                    dest += GLM;
                }
            }
            if (!perturb)
                return;
        }
        else
        {
            releaseAssert(false,"isConstant(false) && !BasisManager::isProjection(m_basis)");
        }

        if (perturb && m_perturbativeSelfEnergies.size() != 0)
        {
            ComplexMatrix EI;
            EI.setZero(m_Ham.rows(),m_Ham.cols(),m_basis,m_spinSym);
            MatrixBufferObject<SelfEnergyMatrix> temp;

            for (auto SE : m_perturbativeSelfEnergies)
            {
                SE->getSigmaR(E,temp);
                EI += temp.get();
            }

            //Can't save a temporary because this aliases
            dest += dest * EI * dest;
        }
    }
    else if (BasisManager::isProjection(m_basis))
    {
        getGaAtEFromGr(E,ComplexMatrix(),dest,perturb);
        return;
    }
    else
    {
        ComplexMatrix temp;
        getGrAtE(E,temp,perturb);
        dest = temp.adjoint();
        return;
    }
}

void GreensFunction::getGaAtEFromGr(complexType E,const ComplexMatrix& Gr, ComplexMatrix& dest, bool perturb) const
{
    if (m_GrComputationMethod == GrComputationType::Subsystem)
    {
        getGaAtE(E,dest,perturb);
    }
    else if (BasisManager::isProjection(m_basis))
    {//Gr must be unused in this branch, see getGaAtE

        ComplexMatrix EI;

        EI.setIdentity(m_Ham.rows(),m_Ham.cols(),m_basis,m_spinSym);

        EI *= (E+iu*1e-30);

        EI -= m_Ham; //m_Ham is always self adjoint in the full basis and so this is the same.
        SelfEnergyMatrix temp;
        for (auto SE : m_selfEnergies)
        {
            SE->getSigmaA(E,temp);
            EI -= temp;
        }

        dest = EI.inverse();

        if (perturb && m_perturbativeSelfEnergies.size() != 0)
        {
            ComplexMatrix EI;
            EI.setZero(m_Ham.rows(),m_Ham.cols(),m_basis,m_spinSym);
            MatrixBufferObject<SelfEnergyMatrix> temp;

            for (auto SE : m_perturbativeSelfEnergies)
            {
                SE->getSigmaA(E,temp);
                EI += temp.get();
            }
            // GA = GA_0 + GA_0 \SigmaR GA_0
            //Can't save a temporary because this aliases
            dest += dest * EI * dest;
        }
    }
    else
    {
        dest = Gr.adjoint();
    }
}

void GreensFunction::getAAtE(numType E, ComplexMatrix &dest, bool perturb) const
{
    ComplexMatrix temp;
    getGrAtE(E,dest,perturb);
    getGaAtEFromGr(E,dest,temp,perturb);
    dest -= temp;
    dest *= iu;
}

void GreensFunction::getGkAtE(numType E, ComplexMatrix &dest, bool perturb) const
{
    ComplexMatrix t1;
    ComplexMatrix t2;
    getGkGrGaAtE(E,dest,t1,t2,perturb);
}

void GreensFunction::getGnAtE(numType E, ComplexSelfAdjointMatrix& out, bool perturb) const
{
    ComplexMatrix dest;
    if (useGK)
    {
        ComplexMatrix gr;
        ComplexMatrix ga;
        getGkGrGaAtE(E,dest,gr,ga,perturb);
        // getGkAtE(E,dest, perturb);
        // getGrAtE(E,gr,perturb);
        dest -= gr;
        // getGaFromGr(gr.get(),ga.get(),m_timeReversalMatrix);
        dest += ga;
        dest *= -iu/2.; // Gn = -0.5*\iu (G^K + G^A - G^R)
        out = static_cast<ComplexSelfAdjointMatrix>(dest);
    }
    else
    {
        ComplexMatrix TotalSigmaIn;


        TotalSigmaIn.setZero(m_Ham.rows(),m_Ham.cols(),m_basis,m_spinSym);

        SelfEnergyMatrix temp;
        for (auto SE : m_selfEnergies)
        {
            SE->getSigmaIn(E,temp);
            TotalSigmaIn += temp;
        }
        ComplexMatrix gr;
        ComplexMatrix ga;

        getGrAtE(E,gr, perturb);
        getGaAtE(E,ga, perturb);

        dest = gr*TotalSigmaIn;
        dest *= ga;
        out = static_cast<ComplexSelfAdjointMatrix>(dest);
    }

}

void GreensFunction::getGpAtE(numType E, ComplexSelfAdjointMatrix& dest, bool perturb) const
{

    if (useGK)
    {
        ComplexMatrix gr;
        ComplexMatrix ga;
        ComplexMatrix gk;
        getGkGrGaAtE(E,gk,gr,ga,perturb);
        // getGkAtE(E,dest,perturb);
        // getGrAtE(E,gr,perturb);
        gk += gr;
        // getGaFromGr(gr.get(),ga.get(),m_timeReversalMatrix);
        gk -= ga;
        gk *= iu/2.; // Gp = 0.5*\iu (G^K - G^A + G^R)
        dest = static_cast<ComplexSelfAdjointMatrix>(gk);
    }
    else
    {
        ComplexMatrix TotalSigmaOut;
        TotalSigmaOut.setZero(m_Ham.rows(),m_Ham.cols(),m_basis,m_spinSym);

        SelfEnergyMatrix temp;
        for (auto SE : m_selfEnergies)
        {
            SE->getSigmaOut(E,temp);
            TotalSigmaOut += temp;
        }
        ComplexMatrix temp1;
        ComplexMatrix temp2;
        ComplexMatrix temp3;
        getGrAtE(E,temp1,perturb);
        getGaAtE(E,temp2,perturb);


        temp3 = temp1*TotalSigmaOut;
        temp3 *= temp2;
        dest = static_cast<ComplexSelfAdjointMatrix>(temp3);
    }
}

void GreensFunction::getGkGrGaAtE(numType E, ComplexMatrix &Gk, ComplexMatrix &Gr, ComplexMatrix &Ga, bool perturb) const
{
    if (m_GrComputationMethod == GrComputationType::Subsystem)
    {
        releaseAssert(perturb == false || m_perturbativeSelfEnergies.size() == 0, "perturb == false");
        numType mu;
        if (isEquilibriumGn(&mu))
        {
            ComplexMatrix GR0,GA0,GK0;
            getGrAtE(E,GR0,false);
            getGaAtEFromGr(E,GR0,GA0,false);
            GK0 = (1-2*fermiFunction(E,mu))*(GR0-GA0);
            //TODO perturb
            Gk = GK0;
            Gr = GR0;
            Ga = GA0;
        }
        else
        {
            releaseAssert(false,"Subsystem Not Implemented");
        }
    }
    else
    {
        ComplexMatrix GR0;
        ComplexMatrix GA0;
        ComplexMatrix GK0;

        SelfEnergyMatrix SigmaK;
        SigmaK.resize(m_Ham.rows(),m_Ham.cols(),m_basis,m_spinSym);
        SigmaK.setZero();
        SelfEnergyMatrix temp;
        for (auto SE : m_selfEnergies)
        {
            SE->getSigmaK(E,temp);
            SigmaK += temp;
        }


        getGrAtE(E,GR0,false);
        //GA0 = GR0.adjoint();
        getGaAtEFromGr(E,GR0,GA0,false);

        GK0 = GR0 * SigmaK;
        GK0 *= GA0; // GK = GR SigmaK GA //TODO is this true in a projected basis? Shouldnt this be a 2x2 matrix product

        Gk = GK0;
        Gr = GR0;
        Ga = GA0;

        if (perturb && m_perturbativeSelfEnergies.size() != 0)
        {
            SelfEnergyMatrix petR;
            petR.resize(m_Ham.rows(),m_Ham.cols(),m_basis,m_spinSym);
            petR.setZero();
            SelfEnergyMatrix petA;
            petA.resize(m_Ham.rows(),m_Ham.cols(),m_basis,m_spinSym);
            petA.setZero();
            SelfEnergyMatrix petK;
            petK.resize(m_Ham.rows(),m_Ham.cols(),m_basis,m_spinSym);
            petK.setZero();

            for (auto SE : m_perturbativeSelfEnergies)
            {
                SE->getSigmaK(E,temp);
                petK += temp;

                SE->getSigmaR(E,temp);
                petR += temp;

                SE->getSigmaA(E,temp);
                petA += temp;
            }


            // Gk += GR0 * petR * GK0;
            // Gk += GR0 * petK * GA0;
            // Gk += GK0 * petA * GA0;
            PEAxB(Gk,GR0,petR*GK0);
            PEAxB(Gk,GR0,petK*GA0);
            PEAxB(Gk,GK0,petA*GA0);

            // GR = GR_0 + GR_0 \SigmaR GR_0
            // Gr += GR0 * petR * GR0;
            PEAxB(Gr,GR0,petR * GR0);
            Ga = Gr.adjoint();

        }
    }
}

void GreensFunction::getGnIntegral(unsigned long, ComplexSelfAdjointMatrix& out, bool perturb, numType EStart, numType EEnd) const
{
    if (m_perturbativeSelfEnergies.size() == 0)
        perturb = false;
    if (EStart == negInf && EEnd == posInf)
    {
        if (m_GnIsCached && !perturb)
        {
            out = m_Gn;
            return;
        }
        else if (m_GnPerturbIsCached && perturb)
        {
            out = m_GnPerturb;
            return;
        }
    }

    ComplexSelfAdjointMatrix compareIntegral;
    compareIntegral.resize(getDimension(),getDimension(),m_basis,m_spinSym);
    compareIntegral.setZero();
    out.resize(getDimension(),getDimension(),m_basis,m_spinSym);
    out.setZero(); //important

    if (isConstant(perturb) && !computeIntegrationNumericallyAlways &&
        !BasisManager::isProjection(m_basis) &&
        (m_integratorType & (integratorType::SumOverPolesEquilibrium|integratorType::SumOverPolesNonEquilibrium|integratorType::autoDetect)) != integratorType::False)
    {
        if (!m_SOPGIsBuilt)
            buildSOP();
        bool forceNonEquilibrium = m_integratorType == integratorType::SumOverPolesNonEquilibrium;
        if (m_SOPGIsBuilt)
        {
            if (m_GrComputationMethod == GrComputationType::Subsystem)
            {
                ComplexMatrix GMM, GML, GLM, GLL,dest;
                m_GMM->getGnIntegral(GMM,forceNonEquilibrium,EStart,EEnd);
                m_GLL->getGnIntegral(GLL,forceNonEquilibrium,EStart,EEnd);
                GMM.toBasisC(GMM,m_basis);
                GLL.toBasisC(GLL,m_basis);
                dest = GMM;
                dest += GLL;
                if (!isSet(m_SubSystemOpt,SubsystemFlags::IgnoreGLM))
                {
                    m_GLM->getGnIntegral(GLM,forceNonEquilibrium,EStart,EEnd);
                    m_GML->getGnIntegral(GML,forceNonEquilibrium,EStart,EEnd);
                    GML.toBasisC(GML,m_basis);
                    GLM.toBasisC(GLM,m_basis);
                    dest += GML;
                    dest += GLM;
                }
                out = static_cast<ComplexSelfAdjointMatrix>(dest);
            }
            else
            {
                ComplexMatrix dest;
                m_GMM->getGnIntegral(dest,forceNonEquilibrium,EStart,EEnd);
                out = static_cast<ComplexSelfAdjointMatrix>(dest);
            }
            if (EStart == negInf && EEnd == posInf)
            {
                if (!perturb)
                {
                    m_Gn = out;
                    m_GnIsCached = true;
                }

                else
                {//perturb == true
                    releaseAssert(false,"Should never get here?");
                    m_GnPerturb = out;
                    m_GnPerturbIsCached = true;
                }
            }
            if (testIntegrator)
            {
                compareIntegral = std::move(out);
                out.resize(getDimension(),getDimension(),m_basis,m_spinSym);
                out.setZero();
            }
            else
                return;
        }
        else
            logger().log("Analytic Gr cannot be computed even though optimisations are allowed");
        //fall through
    }

    std::vector<numType> mup;
    numType MuLowest = 1e20;
    numType MuHighest = -1e20;

    for (auto SE : m_selfEnergies)
    {
        if (SE->hasSigmaK())
        {
            mup.push_back(SE->getChemicalPotential());
            MuLowest = std::min(MuLowest,mup.back());
            MuHighest = std::max(MuHighest,mup.back());
        }
    }
    if (mup.size() == 0)
    {
        mup.push_back(m_mu);
        MuHighest = MuLowest = m_mu;
    }

    releaseAssert(MuLowest >= -100,"Phonons integrals?"); // This is a ridiculously low value and so we probably have a self energy which doesnt have a well defined chemical potential, e.g. Phonons
        //Need to specify the computation type for this. Probably Real axis.
    releaseAssert(EStart == negInf && EEnd == posInf,"EStart == negInf && EEnd == posInf, Not Implemented");
    long dim = getDimension();



    if ((m_GrComputationMethod == GrComputationType::Analytic
         || m_GrComputationMethod == GrComputationType::AnalyticSOP
         || m_GrComputationMethod == GrComputationType::AnalyticSOPRegions
         ||  m_GrComputationMethod == GrComputationType::Subsystem)
        && (m_integratorType & (integratorType::Tricks|integratorType::autoDetect)) != integratorType::False)
    {
        ComplexMatrix dest;
        dest.resize(getDimension(),getDimension(),m_basis,m_spinSym);
        dest.setZero(); //important
        logger().log("Gn Integrate with Tricks");
        MuLowest -= 31*kb*T; // boundary above which we integrate along the real axis
        MuHighest += 31*kb*T;
        for (size_t regionIndex = 0; regionIndex < m_analyticRegions.size()-1; regionIndex++)
        {
            numType regionStartEnergy = m_analyticRegions[regionIndex] == negInf ? negInf : m_analyticRegions[regionIndex] + 1e-10; // helps comparisons get the limit right
            numType regionEndEnergy = m_analyticRegions[regionIndex+1] == posInf ? posInf : m_analyticRegions[regionIndex+1] - 1e-10;
            regionEndEnergy = std::min(MuLowest,regionEndEnergy);

            std::vector<complexType> poles;
            getEstimateForDivergencesOfGr(getRegionRepresentativeEnergy(regionIndex),poles);
            complexType smallestPole = 1e20;
            for (auto& p : poles)
            {
                // assert(p.imag() < 1e-10);
                if (m_forceSelfAdjointSelfEnergy)
                    p = p.real();
                else if (p.imag() > 0)
                    p = std::conj(p);

                smallestPole = smallestPole.real() < p.real() ?  smallestPole : p;
            }
            numType integrationStart = 0;
            if (regionStartEnergy == negInf)
            {
                // integrationStart = smallestPole.real() - 4*std::abs(smallestPole.imag())-10;
                integrationStart = -100;
            }
            else
                integrationStart = regionStartEnergy;
            numType integrationEnd = regionEndEnergy;


            ComplexMatrix temp;

            numType R = integrationEnd - integrationStart;

            auto start1 = std::chrono::high_resolution_clock::now();
            /* Up */
            GaussLegendreIntegral(0,R,100,temp,
                                  [&me = std::as_const(*this),BY_VAL_CAPTURE(integrationStart)](numType P,ComplexMatrix& dest){me.getGrAtE(P*iu + integrationStart,dest);dest *= iu;},
                                  [basis = m_basis,sym = m_spinSym,basisSize = dim]()
                                  {
                                      ComplexMatrix temp;
                                      temp.resize(basisSize,basisSize,basis,sym);
                                      temp.setZero();
                                      return temp;
                                  }
                                  );
            dest += temp;
            auto start2 = std::chrono::high_resolution_clock::now();
            /* Across */
            GaussLegendreIntegral(integrationStart,integrationEnd,100,temp,
                                  [&me = std::as_const(*this),BY_VAL_CAPTURE(R)](numType P,ComplexMatrix& dest){me.getGrAtE(P + R*iu,dest);},
                                  [basis = m_basis,sym = m_spinSym,basisSize = dim]()
                                  {
                                      ComplexMatrix temp;
                                      temp.resize(basisSize,basisSize,basis,sym);
                                      temp.setZero();
                                      return temp;
                                  }
                                  );
            dest += temp;
            auto start3 = std::chrono::high_resolution_clock::now();
            /* Down */
            GaussLegendreIntegral(R,0,100,temp,
                                  [&me = std::as_const(*this),BY_VAL_CAPTURE(integrationEnd)](numType P,ComplexMatrix& dest){me.getGrAtE(P*iu + integrationEnd,dest);dest *= iu;},
                                  [basis = m_basis,sym = m_spinSym,basisSize = dim]()
                                  {
                                      ComplexMatrix temp;
                                      temp.resize(basisSize,basisSize,basis,sym);
                                      temp.setZero();
                                      return temp;
                                  }
                                  );
            dest += temp;

            // Lower G^A contour
            if (BasisManager::isProjection(m_basis) || m_GrComputationMethod == GrComputationType::Subsystem)
            {
                /* Up(down) */
                GaussLegendreIntegral(0,R,100,temp,
                                      [&me = std::as_const(*this),BY_VAL_CAPTURE(integrationStart)](numType P,ComplexMatrix& dest){me.getGaAtE(-P*iu + integrationStart,dest);dest *= -iu;},
                                      [basis = m_basis,sym = m_spinSym,basisSize = dim]()
                                      {
                                          ComplexMatrix temp;
                                          temp.resize(basisSize,basisSize,basis,sym);
                                          temp.setZero();
                                          return temp;
                                      }
                                      );
                dest -= temp;
                /* Across */
                GaussLegendreIntegral(integrationStart,integrationEnd,100,temp,
                                      [&me = std::as_const(*this),BY_VAL_CAPTURE(R)](numType P,ComplexMatrix& dest){me.getGaAtE(P + -R*iu,dest);},
                                      [basis = m_basis,sym = m_spinSym,basisSize = dim]()
                                      {
                                          ComplexMatrix temp;
                                          temp.resize(basisSize,basisSize,basis,sym);
                                          temp.setZero();
                                          return temp;
                                      }
                                      );
                dest -= temp;
                /* Down */
                GaussLegendreIntegral(R,0,100,temp,
                                      [&me = std::as_const(*this),BY_VAL_CAPTURE(integrationEnd)](numType P,ComplexMatrix& dest){me.getGaAtE(-P*iu + integrationEnd,dest);dest *= -iu;},
                                      [basis = m_basis,sym = m_spinSym,basisSize = dim]()
                                      {
                                          ComplexMatrix temp;
                                          temp.resize(basisSize,basisSize,basis,sym);
                                          temp.setZero();
                                          return temp;
                                      }
                                      );
                dest -= temp;
                dest *= iu;
            }
            else
            {
                dest *= iu;
                temp = dest.adjoint();
                dest += temp;
            }


            /* negative Infinity to -R integral */
            auto start4 = std::chrono::high_resolution_clock::now();
            if (regionStartEnergy == negInf)
            {
                SpectralWeightedIntegral<true>(negInf,integrationStart,100000,temp,poles,
                                                [&me = std::as_const(*this)](numType E,ComplexMatrix& dest)
                                                {
                                                    me.getAAtE(E,dest);
                                                },
                                                [basis = m_basis,sym = m_spinSym,basisSize = dim]()
                                                {
                                                    ComplexMatrix temp;
                                                    temp.resize(basisSize,basisSize,basis,sym);
                                                    temp.setZero();
                                                    return temp;
                                                }
                                                );
                dest += temp;
            }
            /* Bias Integral */
            auto start5 = std::chrono::high_resolution_clock::now();
            if (integrationEnd == MuLowest)
            {
                ComplexSelfAdjointMatrix temp3;
                SpectralWeightedIntegral<true>(MuLowest,MuHighest,30000,temp3,poles,
                                                [&me = std::as_const(*this)](numType E,ComplexSelfAdjointMatrix& dest){me.getGnAtE(E,dest);},
                                                [basis = m_basis,sym = m_spinSym,basisSize = dim]()
                                                {
                                                    ComplexSelfAdjointMatrix temp;
                                                    temp.resize(basisSize,basisSize,basis,sym);
                                                    temp.setZero();
                                                    return temp;
                                                }
                                                );
                dest += temp3;
                regionIndex =  m_analyticRegions.size()-1;// effectively a break but we need to print some things first
            }
            auto start6 = std::chrono::high_resolution_clock::now();
            auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(start2 - start1);
            auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(start3 - start2);
            auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(start4 - start3);
            auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(start5 - start4);
            auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(start6 - start5);
            logger().log("Numerical Gn Internal Timings us",std::vector<long>({duration1.count(),duration2.count(),duration3.count(),duration4.count(),duration5.count()}));
        }
        out = static_cast<ComplexSelfAdjointMatrix>(dest);
    }// End Can do contour integral tricks
    else
    {// Cannot do contour integration tricks
        logger().log("Gn Integrate without Tricks, number of Points per region:", 0);
        ComplexSelfAdjointMatrix temp;
        for (size_t regionIndex = 0; regionIndex < m_analyticRegions.size()-1; regionIndex++)
        {
            numType regionStartEnergy = m_analyticRegions[regionIndex];
            numType regionEndEnergy = m_analyticRegions[regionIndex+1];
            regionEndEnergy = std::min(MuHighest+31*(kb*T),regionEndEnergy);
            if (regionStartEnergy >= regionEndEnergy)
            {
                continue;
            }

            std::vector<complexType> poles;
            getEstimateForDivergencesOfGr(getRegionRepresentativeEnergy(regionIndex),poles);

            auto start1 = std::chrono::high_resolution_clock::now();
            SpectralWeightedIntegral<false>(regionStartEnergy,regionEndEnergy,0,temp,poles,
                                            [&me = std::as_const(*this)](numType E,ComplexSelfAdjointMatrix& dest){me.getGnAtE(E,dest);},
                                            [basis = m_basis,sym = m_spinSym,basisSize = dim]()
                                            {
                                                ComplexSelfAdjointMatrix temp;
                                                temp.resize(basisSize,basisSize,basis,sym);
                                                temp.setZero();
                                                return temp;
                                            }
                                            );
            out += temp;
            auto start2 = std::chrono::high_resolution_clock::now();
            auto MAYBE_UNUSED duration1 = std::chrono::duration_cast<std::chrono::microseconds>(start2 - start1);
            // logger().log("No Tricks Numerical Gn Internal Timings us",duration1.count());
        }
    }
    if (!perturb)
    {
        m_Gn = out;
        m_GnIsCached = true;
    }
    else
    {//perturb == true
        m_GnPerturb = out;
        m_GnPerturbIsCached = true;
    }
    if (testIntegrator)
    {

        ComplexMatrix dest2;
        dest2.resize(getDimension(),getDimension(),m_basis,m_spinSym);
        dest2.setZero();
        if (testAgainstNumericalIntegrators)
        {
            ComplexSelfAdjointMatrix temp;
            for (size_t regionIndex = 0; regionIndex < m_analyticRegions.size()-1; regionIndex++)
            {
                numType regionStartEnergy = m_analyticRegions[regionIndex];
                numType regionEndEnergy = m_analyticRegions[regionIndex+1];
                regionEndEnergy = std::min(MuLowest,regionEndEnergy);


                std::vector<complexType> poles;
                getEstimateForDivergencesOfGr(getRegionRepresentativeEnergy(regionIndex),poles);
                SpectralWeightedIntegral<false>(regionStartEnergy,regionEndEnergy,1e6,temp,poles,
                                                [&me = std::as_const(*this)](numType E, ComplexSelfAdjointMatrix& d)
                                                {
                                                    me.getGnAtE(E,d);
                                                },
                                                [basis = m_basis,sym = m_spinSym,basisSize = dim]()
                                                {
                                                    ComplexSelfAdjointMatrix temp;
                                                    temp.resize(basisSize,basisSize,basis,sym);
                                                    temp.setZero();
                                                    return temp;
                                                });
                dest2 += temp;
            }
        }
        if (isConstant(perturb)) // This will only be filled out in this case
        {
            logger().log("Numerical Error, matrix Norm", (out - compareIntegral).norm());
            if (testAgainstNumericalIntegrators)
                logger().log("Numerical Error3, matrix Norm", (dest2 - compareIntegral).norm());
        }
        if (testAgainstNumericalIntegrators)
            logger().log("Numerical Error2, matrix Norm", (out - dest2).norm());

    }



}

void GreensFunction::getGnNonEqIntegral(unsigned long m_numberIntegrationSteps, ComplexSelfAdjointMatrix& dest,bool perturb) const
{
    std::vector<numType> mup;
    numType MuLowest = 1e20;
    numType MuHighest = -1e20;

    for (auto SE : m_selfEnergies)
    {
        if (SE->hasSigmaK())
        {
            mup.push_back(SE->getChemicalPotential());
            MuLowest = std::min(MuLowest,mup.back());
            MuHighest = std::max(MuHighest,mup.back());
        }
    }
    if (mup.size() == 0)
    {
        mup.push_back(m_mu);
        MuHighest = MuLowest = m_mu;
    }

    releaseAssert(MuLowest >= -100,"Phonons integrals?"); // This is a ridiculously low value and so we probably have a self energy which doesnt have a well defined chemical potential, e.g. Phonons
    if (MuLowest == MuHighest)
    {
        dest.setZero(getDimension(),getDimension(),m_basis,m_spinSym);
        return;
    }
    else
    {
        getGnIntegral(m_numberIntegrationSteps,dest,perturb,MuLowest-40*kb*T,MuHighest+40*kb*T);
        return;
    }


}

void GreensFunction::getGpIntegral(unsigned long numberIntegrationSteps, ComplexSelfAdjointMatrix &dest, bool perturb) const
{
    ComplexSelfAdjointMatrix GnInt;
    getGnIntegral(numberIntegrationSteps,GnInt, perturb);
    dest.setIdentity(getDimension(),getDimension(),m_basis,m_spinSym);
    dest *= 2*M_PI;
    dest -= GnInt;
}

void GreensFunction::getEGnIntegral(unsigned long, ComplexSelfAdjointMatrix& out, bool perturb) const
{
    if (m_perturbativeSelfEnergies.size() == 0)
        perturb = false;
    if (m_EGnIsCached && !perturb)
    {
        out = m_EGn;
        return;
    }
    else if (m_EGnPerturbIsCached && perturb)
    {
        out = m_EGnPerturb;
        return;
    }

    ComplexSelfAdjointMatrix compareIntegral;
    compareIntegral.resize(getDimension(),getDimension(),m_basis,m_spinSym);
    compareIntegral.setZero();
    out.resize(getDimension(),getDimension(),m_basis,m_spinSym);
    out.setZero();

    if (isConstant(perturb) && !computeIntegrationNumericallyAlways &&
        !BasisManager::isProjection(m_basis) &&
        (m_integratorType & (integratorType::SumOverPolesEquilibrium|integratorType::SumOverPolesNonEquilibrium|integratorType::autoDetect)) != integratorType::False)
    {
        if (!m_SOPGIsBuilt)
            buildSOP();
        if (m_SOPGIsBuilt)
        {
            if (m_GrComputationMethod == GrComputationType::Subsystem)
            {
                ComplexMatrix GMM, GML, GLM, GLL,dest;
                m_GMM->getEGnIntegral(GMM);
                m_GLL->getEGnIntegral(GLL);
                GMM.toBasisC(GMM,m_basis);
                GLL.toBasisC(GLL,m_basis);
                dest = GMM;
                dest += GLL;
                if (!isSet(m_SubSystemOpt,SubsystemFlags::IgnoreGLM))
                {
                    m_GLM->getEGnIntegral(GLM);
                    m_GML->getEGnIntegral(GML);
                    GML.toBasisC(GML,m_basis);
                    GLM.toBasisC(GLM,m_basis);
                    dest += GML;
                    dest += GLM;
                }
                out = static_cast<ComplexSelfAdjointMatrix>(dest);
            }
            else
            {
                ComplexMatrix dest;
                m_GMM->getEGnIntegral(dest);
                out = static_cast<ComplexSelfAdjointMatrix>(dest);
            }

            if (!perturb)
            {
                m_EGn = out;
                m_EGnIsCached = true;
            }
            else
            {//perturb == true
                m_EGnPerturb = out;
                m_EGnPerturbIsCached = true;
            }

            if (testIntegrator)
            {
                compareIntegral = std::move(out);
                out.resize(getDimension(),getDimension(),m_basis,m_spinSym);
                out.setZero();
            }
            else
                return;
        }
        else
            logger().log("Analytic Gr cannot be computed even though optimisations are allowed");
        //fall through
    }

    std::vector<numType> mup;
    numType MuLowest = 1e20;
    numType MuHighest = -1e20;

    for (auto SE : m_selfEnergies)
    {
        if (SE->hasSigmaK())
        {
            mup.push_back(SE->getChemicalPotential());
            MuLowest = std::min(MuLowest,mup.back());
            MuHighest = std::max(MuHighest,mup.back());
        }
    }
    if (mup.size() == 0)
    {
        mup.push_back(m_mu);
        MuHighest = MuLowest = m_mu;
    }

    releaseAssert(MuLowest >= -100,"Phonons integrals?"); // This is a ridiculously low value and so we probably have a self energy which doesnt have a well defined chemical potential, e.g. Phonons
        //Need to specify the computation type for this. Probably Real axis.

    long dim = getDimension();



    if ((m_GrComputationMethod == GrComputationType::Analytic
         || m_GrComputationMethod == GrComputationType::AnalyticSOP
         || m_GrComputationMethod == GrComputationType::AnalyticSOPRegions
         ||  m_GrComputationMethod == GrComputationType::Subsystem)
        && (m_integratorType & (integratorType::Tricks|integratorType::autoDetect)) != integratorType::False)
    {
        ComplexMatrix dest;
        dest.resize(getDimension(),getDimension(),m_basis,m_spinSym);
        dest.setZero(); //important
        logger().log("EGn Integrate with Tricks");
        MuLowest -= 31*kb*T; // boundary above which we integrate along the real axis
        MuHighest += 31*kb*T;
        for (size_t regionIndex = 0; regionIndex < m_analyticRegions.size()-1; regionIndex++)
        {
            numType regionStartEnergy = m_analyticRegions[regionIndex] == negInf ? negInf : m_analyticRegions[regionIndex] + 1e-10; // helps comparisons get the limit right
            numType regionEndEnergy = m_analyticRegions[regionIndex+1] == posInf ? posInf : m_analyticRegions[regionIndex+1] - 1e-10;
            regionEndEnergy = std::min(MuLowest,regionEndEnergy);



            std::vector<complexType> poles;
            getEstimateForDivergencesOfGr(getRegionRepresentativeEnergy(regionIndex),poles);
            complexType smallestPole = 1e20;
            for (auto p : poles)
            {
                if (m_forceSelfAdjointSelfEnergy)
                    p = p.real();
                else if (p.imag() > 0)
                    p = std::conj(p);
                smallestPole = smallestPole.real() < p.real() ?  smallestPole : p;
            }
            numType integrationStart = -100;
            // if (regionStartEnergy == negInf)
            // {
            //     integrationStart = smallestPole.real() - 4*std::abs(smallestPole.imag())-10;
            // }
            // else
            //     integrationStart = regionStartEnergy;
            numType integrationEnd = regionEndEnergy;


            ComplexMatrix temp;

            numType R = integrationEnd - integrationStart;

            auto start1 = std::chrono::high_resolution_clock::now();
            /* Up */
            GaussLegendreIntegral(0,R,100,temp,
                                  [&me = std::as_const(*this),BY_VAL_CAPTURE(integrationStart)](numType P,ComplexMatrix& dest){me.getGrAtE(P*iu + integrationStart,dest);dest *= (P*iu +integrationStart)*iu;},
                                  [basis=m_basis, sym=m_spinSym, basisSize = dim]()
                                  {
                                      ComplexMatrix temp;
                                      temp.resize(basisSize,basisSize,basis,sym);
                                      temp.setZero();
                                      return temp;
                                  }
                                  );
            dest += temp;
            auto start2 = std::chrono::high_resolution_clock::now();
            /* Across */
            GaussLegendreIntegral(integrationStart,integrationEnd,100,temp,
                                  [&me = std::as_const(*this),BY_VAL_CAPTURE(R)](numType P,ComplexMatrix& dest){me.getGrAtE(P + R*iu,dest); dest *= (P + R*iu);},
                                  [basis=m_basis, sym=m_spinSym, basisSize = dim]()
                                  {
                                      ComplexMatrix temp;
                                      temp.resize(basisSize,basisSize,basis,sym);
                                      temp.setZero();
                                      return temp;
                                  }
                                  );
            dest += temp;
            auto start3 = std::chrono::high_resolution_clock::now();
            /* Down */
            GaussLegendreIntegral(R,0,100,temp,
                                  [&me = std::as_const(*this),BY_VAL_CAPTURE(integrationEnd)](numType P,ComplexMatrix& dest){me.getGrAtE(P*iu + integrationEnd,dest);dest *= (P*iu + integrationEnd)*iu;},
                                  [basis=m_basis, sym=m_spinSym, basisSize = dim]()
                                  {
                                      ComplexMatrix temp;
                                      temp.resize(basisSize,basisSize,basis,sym);
                                      temp.setZero();
                                      return temp;
                                  }
                                  );
            dest += temp;

            // Lower G^A contour
            if (BasisManager::isProjection(m_basis))
            {
                /* Up(down) */
                GaussLegendreIntegral(0,R,100,temp,
                                      [&me = std::as_const(*this),BY_VAL_CAPTURE(integrationStart)](numType P,ComplexMatrix& dest){me.getGaAtE(-P*iu + integrationStart,dest);dest *= (-P*iu +integrationStart)*(-iu);},
                                      [basis=m_basis, sym=m_spinSym, basisSize = dim]()
                                      {
                                          ComplexMatrix temp;
                                          temp.resize(basisSize,basisSize,basis,sym);
                                          temp.setZero();
                                          return temp;
                                      }
                                      );
                dest -= temp;
                /* Across */
                GaussLegendreIntegral(integrationStart,integrationEnd,100,temp,
                                      [&me = std::as_const(*this),BY_VAL_CAPTURE(R)](numType P,ComplexMatrix& dest){me.getGaAtE(P + -R*iu,dest); dest *= (P + -R*iu);},
                                      [basis=m_basis, sym=m_spinSym, basisSize = dim]()
                                      {
                                          ComplexMatrix temp;
                                          temp.resize(basisSize,basisSize,basis,sym);
                                          temp.setZero();
                                          return temp;
                                      }
                                      );
                dest -= temp;
                /* Down */
                GaussLegendreIntegral(R,0,100,temp,
                                      [&me = std::as_const(*this),BY_VAL_CAPTURE(integrationEnd)](numType P,ComplexMatrix& dest){me.getGaAtE(-P*iu + integrationEnd,dest);dest *= (-P*iu + integrationEnd)*(-iu);},
                                      [basis=m_basis, sym=m_spinSym, basisSize = dim]()
                                      {
                                          ComplexMatrix temp;
                                          temp.resize(basisSize,basisSize,basis,sym);
                                          temp.setZero();
                                          return temp;
                                      }
                                      );
                dest -= temp;
                dest *= iu;
            }
            else
            {
                dest *= iu; //iu (GR-GA)
                temp = dest.adjoint();
                dest += temp;
            }


            /* negative Infinity to -R integral */
            auto start4 = std::chrono::high_resolution_clock::now();
            /* Bias Integral */
            auto start5 = std::chrono::high_resolution_clock::now();
            if (regionEndEnergy == MuLowest)
            {
                ComplexSelfAdjointMatrix temp2;
                SpectralWeightedIntegral<true>(MuLowest,MuHighest,30000,temp2,poles,
                                                [&me = std::as_const(*this)](numType E,ComplexSelfAdjointMatrix& dest){me.getGnAtE(E,dest); dest *= E;},
                                                [basis=m_basis, sym=m_spinSym, basisSize = dim]()
                                                {
                                                    ComplexSelfAdjointMatrix temp;
                                                    temp.resize(basisSize,basisSize,basis,sym);
                                                    temp.setZero();
                                                    return temp;
                                                }
                                                );
                dest += temp2;
                regionIndex =  m_analyticRegions.size()-1;// effectively a break but we need to print some things first
            }
            auto start6 = std::chrono::high_resolution_clock::now();
            auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(start2 - start1);
            auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(start3 - start2);
            auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(start4 - start3);
            auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(start5 - start4);
            auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(start6 - start5);
            logger().log("Numerical EGn Internal Timings us",std::vector<long>({duration1.count(),duration2.count(),duration3.count(),duration4.count(),duration5.count()}));
        }
        out = static_cast<ComplexSelfAdjointMatrix>(dest);
    }// End Can do contour integral tricks
    else
    {// Cannot do contour integration tricks
        logger().log("Gn Integrate without Tricks, number of Points per region:", 100);
        ComplexSelfAdjointMatrix temp;
        for (size_t regionIndex = 0; regionIndex < m_analyticRegions.size()-1; regionIndex++)
        {
            numType regionStartEnergy = m_analyticRegions[regionIndex];
            numType regionEndEnergy = m_analyticRegions[regionIndex+1];
            regionEndEnergy = std::min(MuLowest,regionEndEnergy);



            std::vector<complexType> poles;
            getEstimateForDivergencesOfGr(getRegionRepresentativeEnergy(regionIndex),poles);

            auto start1 = std::chrono::high_resolution_clock::now();
            SpectralWeightedIntegral<false>(regionStartEnergy,regionEndEnergy,100,temp,poles,
                                            [&me = std::as_const(*this)](numType E,ComplexSelfAdjointMatrix& dest){me.getGnAtE(E,dest); dest *= E;},
                                            [basis=m_basis, sym=m_spinSym, basisSize = dim]()
                                            {
                                                ComplexSelfAdjointMatrix temp;
                                                temp.resize(basisSize,basisSize,basis,sym);
                                                temp.setZero();
                                                return temp;
                                            }
                                            );
            out += temp;
            auto start2 = std::chrono::high_resolution_clock::now();
            auto MAYBE_UNUSED duration = std::chrono::duration_cast<std::chrono::microseconds>(start2 - start1);
            // logger().log("No Tricks Numerical Gn Internal Timings us",duration.count());
        }
    }
    if (!perturb)
    {
        m_EGn = out;
        m_EGnIsCached = true;
    }
    else
    {//perturb == true
        m_EGnPerturb = out;
        m_EGnPerturbIsCached = true;
    }
    if (testIntegrator)
    {

        ComplexSelfAdjointMatrix dest2;
        dest2.resize(getDimension(),getDimension(),m_basis,m_spinSym);
        dest2.setZero();
        if (testAgainstNumericalIntegrators)
        {
            ComplexSelfAdjointMatrix temp;
            for (size_t regionIndex = 0; regionIndex < m_analyticRegions.size()-1; regionIndex++)
            {
                numType regionStartEnergy = m_analyticRegions[regionIndex];
                numType regionEndEnergy = m_analyticRegions[regionIndex+1];
                regionEndEnergy = std::min(MuLowest,regionEndEnergy);


                std::vector<complexType> poles;
                getEstimateForDivergencesOfGr(getRegionRepresentativeEnergy(regionIndex),poles);
                SpectralWeightedIntegral<false>(regionStartEnergy,regionEndEnergy,1e6,temp,poles,
                                                [&me = std::as_const(*this)](numType E, ComplexSelfAdjointMatrix& d)
                                                {
                                                    me.getGnAtE(E,d);
                                                    d*=E;
                                                },
                                                [basis=m_basis, sym=m_spinSym, basisSize = dim]()
                                                {
                                                    ComplexSelfAdjointMatrix temp;
                                                    temp.resize(basisSize,basisSize,basis,sym);
                                                    temp.setZero();
                                                    return temp;
                                                });
                dest2 += temp;
            }
        }
        if (isConstant(perturb)) // This will only be filled out in this case
        {
            logger().log("Numerical Error, matrix Norm", (out - compareIntegral).norm());
            if (testAgainstNumericalIntegrators)
                logger().log("Numerical Error3, matrix Norm", (dest2 - compareIntegral).norm());
        }
        if (testAgainstNumericalIntegrators)
            logger().log("Numerical Error2, matrix Norm", (out - dest2).norm());

    }



}

void GreensFunction::getGrIntegral(bool perturb, ComplexMatrix& dest) const
{
    if (m_GrComputationMethod == GrComputationType::Subsystem)
    {
        releaseAssert(false,"Subsystem Not Implemented");
    }
    else if (perturb == false && isEquilibriumGn() &&
                   m_GrComputationMethod == GrComputationType::AnalyticSOP || m_GrComputationMethod == GrComputationType::AnalyticSOPRegions)
    {
        if (!m_SOPGIsBuilt) buildSOP();
        m_GMM->getGrIntegral(dest);
        return;

    }
    releaseAssert(false, "GrIntegral not implemented in this case");
}

bool GreensFunction::isEquilibriumGn(numType* mu) const
{

    numType smallestMu=1e20;
    numType largestMu=-1e20;

    for (auto SE : m_selfEnergies)
    {
        if (SE->hasSigmaK())
        {
            numType mu = SE->getChemicalPotential();
            largestMu = std::max(mu,largestMu);
            smallestMu = std::min(mu,smallestMu);
        }
    }
    if (abs(largestMu - smallestMu) < 1e-10)
    {
        if (mu)
            *mu = largestMu;
        return true;
    }
    return false;
}

std::function<void(numType,ComplexSelfAdjointMatrix&)> GreensFunction::getBareGnFunc() const
{
    //Removed
    __builtin_trap();   
}

std::function<void(numType,ComplexSelfAdjointMatrix&)> GreensFunction::getBareGpFunc() const
{
    //Removed
    __builtin_trap();
}

template <>
void logger::logAccurate<GreensFunction::GrComputationType>(const std::string& name, GreensFunction::GrComputationType e)
{
    switch (e)
    {
    case GreensFunction::GrComputationType::Analytic:
        log(name,"Analytic"); break;
    case GreensFunction::GrComputationType::AnalyticSOP:
        log(name,"AnalyticSOP"); break;
    case GreensFunction::GrComputationType::AnalyticSOPRegions:
        log(name,"AnalyticSOPRegions"); break;
    case GreensFunction::GrComputationType::None:
        log(name,"None"); break;
    case GreensFunction::GrComputationType::RealAxis:
        log(name,"RealAxis"); break;
    case GreensFunction::GrComputationType::Subsystem:
        log(name,"Subsystem"); break;
    default:
        log(name,static_cast<std::underlying_type_t<GreensFunction::GrComputationType>>(e)); break;

    };
}
