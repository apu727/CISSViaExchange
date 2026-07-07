#ifndef INTERFACESELFENERGY_H
#define INTERFACESELFENERGY_H
#include "greensfunction.h"
#include "selfenergybase.h"

template<typename SelfEnergyMatrixType>
class InterfaceSelfEnergy : public SelfEnergyBase<SelfEnergyMatrixType>
{
public:
    enum class LeadType
    {
        GreensFunction,
        TightBindingInfinite,
        ContinuumInfinite,
        None
    };
    struct Parameters
    {
        LeadType m_type = LeadType::None;
        numType mu = 0;
        //For the infinite tight binding lead

        numType t = 0; // hopping term
        numType U0 = 0;
        numType a = 1; // lattice spacing, bohr radii

        //For the Continuum Lead
        // numType U0 = 0; - Reused. In order to match the parameters for the tightBindingInfinite. U0->U0+2t
        //numType a = 1; -Reused, Here it denotes where the continuum lead interacts from. I.e.  -----a---| . . . . |---a-----
        numType m = 1; // Units of electron mass

        //TODO this is a basis change matrix but doesnt fit with the current basis manager.
        ComplexDirectMatrix couplingPoints; // This is the matrix either side of GFunc. i.e. \Sigma^R = couplingPoints * G^R * couplingPoints.adjoint
        Eigen::Vector3d MagnetisationDirection;

        //For the Green's function version. Self Energy matrix for the coupling points so they dont have to be static
        //Expect this is either a MatrixSelfEnergy or a FockHamiltonian -> Either way it is assumed to be self adjoint!
        std::shared_ptr<GreensFunction> Gfunc; // The source for G^R(E). Should be analytic => sum over poles representation
        std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrixType>>> SECouple; // Used as \Sigma^R = SECouple.SigmaR * G^R * SECouple.SigmaA. Should be self adjoint and constant

        SelfEnergyMatrixType SECoupleCacheLHS; // The H_{ML} that is cached and the basis transforms managed by this
        SelfEnergyMatrixType SECoupleCacheRHS; // H_{LM}

        SelfEnergyMatrixType SECoupleCacheAdjointLHS; // The H^A_{ML} that is cached and the basis transforms managed by this
        SelfEnergyMatrixType SECoupleCacheAdjointRHS; // H^A_{LM}

        Parameters(){MagnetisationDirection.setZero();}
    };
private:
    Parameters m_params;
    indexBasisT m_currentBasis;
    //Variables for the Greens function version


    //Variables for TightBindingInfinite case
    ComplexSparseMatrix m_SigmaRBase; // couplingPoints*couplingPoints.T Since G^R is a scalar

    complexType getKFromE(complexType E);
    complexType fermiFunction(complexType E);

public:
    InterfaceSelfEnergy(Parameters params);

    void getSigmaIn(numType E, SelfEnergyMatrixType& dest) override;

    void getSigmaOut(numType E, SelfEnergyMatrixType& dest) override;

    void getSigmaR(complexType E, SelfEnergyMatrixType& dest) override;

    void getSigmaA(complexType E, SelfEnergyMatrixType& dest) override;

    void getSigmaK(numType E, SelfEnergyMatrixType& dest) override;

    void getGamma(numType E, SelfEnergyMatrixType& dest) override;

    void recomputeSelfEnergies() override;
    void setMatrixDimension(uint32_t) override;

    bool hasSigmaK()  override {return true;}
    numType getChemicalPotential() override {return m_params.mu;}

    //Lets it know the Green's function has changed so any dependent quantities (mu) need to be updated
    void clearCache() override;
    //Note that This is called after every SCF iteration. So check if it has changed before doing work
    void setBasis(const std::string&) override;
    void setBasis(const indexBasisT& bas)override{releaseAssert(bas[0] == bas[1],"bas[0] == bas[1], InterfaceSelfEnergy"); setBasis(bas[0]);};
    const indexBasisT& getBasis()override{return m_currentBasis;}
    std::vector<numType> getContinuousRegions() override;


};

#endif // INTERFACESELFENERGY_H
