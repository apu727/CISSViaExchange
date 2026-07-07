#ifndef GREENSFUNCTION_H
#define GREENSFUNCTION_H
#include "hamiltonianbase.h"
#include "linalg.h"
#include "selfenergybase.h"
#include <memory>


class GreensFunctionTest;
class SOPGreensFunction;

class GreensFunction
{
public:
    enum class integratorType
    {
        False = 0,
        autoDetect = 1,
        SumOverPolesEquilibrium = 2,
        SumOverPolesNonEquilibrium = 4,
        NoAssumptions = 8,
        Tricks = 16
    };
    enum class EnforcedSpinSymmetryType
    {
        autoDetect,
        GHF,
        RHF,
        UHF,//Not implemented
    };
    enum class GrComputationType
    {
        AnalyticSOP, // We know the full SOP for all energies
        AnalyticSOPRegions, // We know the full SOP between regions
        Analytic, //We can evaluate Gr anywhere(except poles) in the complex plane
        RealAxis, //We can only evaluate Gr on the real axis.
        Subsystem,  // Computes quantities via the subsystem seperation method
        None,
    };
    enum class SubsystemFlags
    {
        False = 0,
        AOPartitioning = 1, // Subsystems are done in the dual direct basis or in the H_{ij} G^{jk} (AO partitioning) basis.
        FreezeLead = 2, // Freezes the GLL subsystem to the bare one
        FreezeMol = 4, // Freezes the GMM subsystem to the bare one
        IgnoreGLM = 8 // Ignores the GLM and GML terms.
    };

private:
    friend class GreensFunctionTest;

    /* Setup objects*/
    ComplexSelfAdjointSparseMatrix m_Ham;

    std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>>> m_selfEnergies;
    std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>>> m_perturbativeSelfEnergies;

    const enums::SpinSymmetry m_spinSym = enums::SpinSymmetry::RHF; // always start with RHF, this is then degraded by other matrices that are added
    std::string m_basis = "";
    numType m_mu = 0; // Only use if this is not determinable from the self energies.

    /* internal state objects */
    //For the SOP representation
    mutable std::shared_ptr<SOPGreensFunction> m_GMM; //circular header dependency due to enums
    //Used in subsystem
    mutable std::shared_ptr<SOPGreensFunction> m_GLL;
    mutable std::shared_ptr<SOPGreensFunction> m_GML;
    mutable std::shared_ptr<SOPGreensFunction> m_GLM;
    mutable std::shared_ptr<SOPGreensFunction> m_GDEBUG;
    mutable std::atomic_bool m_SOPGIsBuilt = false;
    mutable std::mutex m_cacheMutex;


    //Cached Integrals
    mutable ComplexSelfAdjointMatrix m_Gn;
    mutable bool m_GnIsCached = false;
    mutable ComplexSelfAdjointMatrix m_GnPerturb;
    mutable bool m_GnPerturbIsCached = false;

    mutable ComplexSelfAdjointMatrix m_EGn;
    mutable bool m_EGnIsCached = false;
    mutable ComplexSelfAdjointMatrix m_EGnPerturb;
    mutable bool m_EGnPerturbIsCached = false;

    std::vector<numType> m_analyticRegions;
    std::vector<numType> m_perturbativeAnalyticRegions;
    void setupAnalyticRegions();
    bool findIndexOfRegion(complexType E, size_t& index) const;

    void buildSOP() const;





    integratorType m_integratorType = integratorType::autoDetect; // used in deciding which integrator to use. Usually autodetect but this allows for testing of others
    EnforcedSpinSymmetryType m_enforcedSpinSym = EnforcedSpinSymmetryType::autoDetect;
    bool m_forceSelfAdjointSelfEnergy = false;
    GrComputationType m_GrComputationMethod = GrComputationType::None;
    GrComputationType m_perturbativeGrComputationMethod = GrComputationType::None;
    GrComputationType m_subSystemComputationMethod = GrComputationType::None; // Method to compute this specific subsystem via

    std::vector<std::string> m_subsystemBases;
    SubsystemFlags m_SubSystemOpt = SubsystemFlags::False; //options that configure the exact calculation


public:
    GreensFunction();
    virtual ~GreensFunction(){};

    void clearCache();

    /*
     * Returns true if we can do a SOP or SOPRegions expression for the Green's function.
     * For this all self energies need to be constant(piecewise/selfconsistent) and we need to not be perturbative,
     * either due to perturb==false or m_perturbativeSE.size() == 0
     */
    bool isConstant(bool perturb) const;
    bool isEquilibriumGn(numType* mu = nullptr) const;
    const std::string& getBasis(){return m_basis;}

    bool getRestrictedCalculation() const {return m_enforcedSpinSym == EnforcedSpinSymmetryType::RHF;}
    void setRestrictedCalculation(bool isRestricted){  m_enforcedSpinSym = isRestricted ? EnforcedSpinSymmetryType::RHF : EnforcedSpinSymmetryType::GHF;}

    EnforcedSpinSymmetryType getEnforcedSpin() const {return m_enforcedSpinSym;}
    void setEnforcedSpin(EnforcedSpinSymmetryType s){m_enforcedSpinSym = s;}

    enums::SpinSymmetry getSpinSym() const {return m_spinSym;}; // Returns the maximum possible symmetry of the results.

    void forceSelfAdjointSelfEnergy(bool force){m_forceSelfAdjointSelfEnergy = force;}

    void getEstimateForDivergencesOfGr(numType E, std::vector<complexType>& divergences) const;
    const GrComputationType& getComputationMethod() const {return m_GrComputationMethod;}
    const std::vector<numType>& getAnalyticRegions() const {return m_analyticRegions;}
    numType getRegionRepresentativeEnergy(size_t regionIndex) const;

    numType getMu() const {return m_mu;}
    void setMu(numType m) {m_mu = m;}






    const ComplexSelfAdjointSparseMatrix& getHamiltonian() const {return m_Ham;}
    void setHamiltonian(std::shared_ptr<HamiltonianBase> H );

    std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>>> getSelfEnergies() const {return m_selfEnergies;}
    void setSelfEnergies(const std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>>>& SEs){clearCache(); m_selfEnergies = SEs; setupAnalyticRegions();}
    void setPerturbativeSelfEnergies(const std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>>>& SEs){clearCache(); m_perturbativeSelfEnergies = SEs; setupAnalyticRegions();}
    void setSubsystems(const std::vector<std::string>& subsystems,SubsystemFlags flags);

    // std::vector<ComplexMatrix> getFrozenCoreTransformation(numType cutoffE);

    void getGrAtE(complexType E, ComplexMatrix& dest,bool perturb = true) const;
    void getGaAtE(complexType E, ComplexMatrix& dest,bool perturb = true) const;
    void getGaAtEFromGr(complexType E,const ComplexMatrix& Gr, ComplexMatrix& dest, bool perturb = true) const;
    void getAAtE(numType E, ComplexMatrix& dest,bool perturb = true) const;
    void getGkAtE(numType E, ComplexMatrix& dest,bool perturb = true) const;
    void getGnAtE(numType E, ComplexSelfAdjointMatrix& dest,bool perturb = true) const;
    void getGpAtE(numType E, ComplexSelfAdjointMatrix& dest,bool perturb = true) const;

    void getGkGrGaAtE(numType E, ComplexMatrix& Gk, ComplexMatrix& Gr, ComplexMatrix& Ga, bool perturb = true) const;

    void getGrAtEInBasis(numType E, ComplexMatrix& dest, const std::string& basis) const
    {ComplexMatrix temp; getGrAtE(E,temp); temp.toBasisC(dest,basis);}
    void getGaAtEInBasis(numType E, ComplexMatrix& dest, const std::string& basis) const
    {ComplexMatrix temp; getGaAtE(E,temp); temp.toBasisC(dest,basis);}
    void getGnAtEInBasis(numType E, ComplexSelfAdjointMatrix& dest, const std::string& basis) const
    {ComplexSelfAdjointMatrix temp; getGnAtE(E,temp); temp.toBasisC(dest,basis);}
    void getGpAtEInBasis(numType E, ComplexSelfAdjointMatrix& dest,const std::string& basis) const
    {ComplexSelfAdjointMatrix temp; getGpAtE(E,temp); temp.toBasisC(dest,basis);}




    void getGnIntegral(unsigned long m_numberIntegrationSteps, ComplexSelfAdjointMatrix& dest,bool perturb, numType EStart = negInf, numType EEnd = posInf) const;
    void getGnNonEqIntegral(unsigned long m_numberIntegrationSteps, ComplexSelfAdjointMatrix& dest,bool perturb) const;
    void getGpIntegral(unsigned long m_numberIntegrationSteps, ComplexSelfAdjointMatrix& dest, bool perturb) const;
    void getEGnIntegral(unsigned long m_numberIntegrationSteps, ComplexSelfAdjointMatrix& dest, bool perturb) const;
    void getGrIntegral(bool perturb, ComplexMatrix& dest) const; // in Testing



    std::function<void(numType,ComplexSelfAdjointMatrix&)> getBareGnFunc() const;
    std::function<void(numType,ComplexSelfAdjointMatrix&)> getBareGpFunc() const;
    long getDimension() const{return m_Ham.rows();}

    //Python overloads
    ComplexSelfAdjointMatrix getGnAtE(numType E) const {ComplexSelfAdjointMatrix temp; getGnAtE(E,temp); return temp;}
    ComplexSelfAdjointMatrix getGpAtE(numType E) const {ComplexSelfAdjointMatrix temp; getGpAtE(E,temp); return temp;}
    ComplexMatrix getGrAtE(numType E) const {ComplexMatrix temp; getGrAtE(E,temp); return temp;}
    ComplexMatrix getGaAtE(numType E) const {ComplexMatrix temp; getGaAtE(E,temp); return temp;}

    ComplexSelfAdjointMatrix getGnAtEInBasis(numType E, const std::string& basis) const {ComplexSelfAdjointMatrix temp; getGnAtEInBasis(E,temp,basis); return temp;}
    ComplexSelfAdjointMatrix getGpAtEInBasis(numType E, const std::string& basis) const {ComplexSelfAdjointMatrix temp; getGpAtEInBasis(E,temp,basis); return temp;}
    ComplexMatrix getGrAtEInBasis(numType E, const std::string& basis) const {ComplexMatrix temp; getGrAtEInBasis(E,temp,basis); return temp;}
    ComplexMatrix getGaAtEInBasis(numType E, const std::string& basis) const {ComplexMatrix temp; getGaAtEInBasis(E,temp,basis); return temp;}
    // void save(std::string filename) const;




};

inline GreensFunction::integratorType operator|(GreensFunction::integratorType a, GreensFunction::integratorType b)
{
    typedef std::underlying_type_t<GreensFunction::integratorType> actualType;
    return static_cast<GreensFunction::integratorType>(static_cast<actualType>(a) | static_cast<actualType>(b));
}
inline GreensFunction::integratorType operator&(GreensFunction::integratorType a, GreensFunction::integratorType b)
{
    typedef std::underlying_type_t<GreensFunction::integratorType> actualType;
    return static_cast<GreensFunction::integratorType>(static_cast<actualType>(a) & static_cast<actualType>(b));
}

inline GreensFunction::SubsystemFlags operator|(GreensFunction::SubsystemFlags a, GreensFunction::SubsystemFlags b)
{
    typedef std::underlying_type_t<GreensFunction::SubsystemFlags> actualType;
    return static_cast<GreensFunction::SubsystemFlags>(static_cast<actualType>(a) | static_cast<actualType>(b));
}
inline GreensFunction::SubsystemFlags operator&(GreensFunction::SubsystemFlags a, GreensFunction::SubsystemFlags b)
{
    typedef std::underlying_type_t<GreensFunction::SubsystemFlags> actualType;
    return static_cast<GreensFunction::SubsystemFlags>(static_cast<actualType>(a) & static_cast<actualType>(b));
}
inline bool isSet(GreensFunction::SubsystemFlags a, GreensFunction::SubsystemFlags b)
{
    return (a & b) != GreensFunction::SubsystemFlags::False;
}



template <> void logger::logAccurate<GreensFunction::GrComputationType>(const std::string& name, GreensFunction::GrComputationType e);


#endif // GREENSFUNCTION_H
