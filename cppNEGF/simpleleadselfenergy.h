#ifndef SIMPLELEADSELFENERGY_H
#define SIMPLELEADSELFENERGY_H

#include "selfenergybase.h"
#include "greensfunction.h"

template<class SelfEnergyMatrixType>
class SimpleLeadSelfEnergyTests;

template<class SelfEnergyMatrixType>
class SimpleLeadSelfEnergy : public SelfEnergyBase<SelfEnergyMatrixType>
{
    friend class SimpleLeadSelfEnergyTests<SelfEnergyMatrixType>;

    numType fermiFunction(numType E);
    void computeSigmaR(numType E, SelfEnergyMatrixType& dest);
    void computeSigmaA(numType E, SelfEnergyMatrixType& dest);
    SelfEnergyMatrixType m_SigmaR; // we can do this because its a constant
    SelfEnergyMatrixType m_SigmaA; // we can do this because its a constant
    SelfEnergyMatrixType m_ZeroMatrix;
    std::shared_ptr<GreensFunction> m_GFunc = nullptr;
    std::vector<numType> m_ConstantRegions;
    indexBasisT m_workingBasis = {};

    const SelfEnergyMatrixType& zeroOrMatrix(complexType E, const SelfEnergyMatrixType& mat)
    {
        return (!m_params.hasBottom || E.real() >= m_params.bottomOfBand) ? mat : m_ZeroMatrix;
    }
public:
    typedef struct Parameters
    {
        numType alpha1; // coupling factor to system ?
        numType t; // finite difference kinetic energy of lead
        numType mu; // chemical potential of lead

        bool adjustMuToElectrons = false; // dynamically adjusts the potential to get ~numberOfElectrons of electrons. Will never go above muMax
        int numberOfElectrons = 0;
        numType muMax;

        bool hasBottom = false;
        numType bottomOfBand = 0; // below which Gamma = 0


        ComplexDirectMatrix couplingPoints;
        Eigen::Vector3d magnetisationDir; // only first three taken into account
        bool upSpin;// upSpin is defined as the direction with positive Eigenvalue
        int MatrixDimension;

        // Can be used to construct a lead in a bigger basis and then project down. Used in Frozen core
        // bool hasCProj = false;
        // ComplexMatrix CProj;
        // ComplexMatrix CInvProjDirecttoDirect; // Direct basis to direct basis version
        enums::SpinSymmetry sym = enums::SpinSymmetry::NoSpinSym; // Cant yet be autodetermined since it would always be NoSpinSym even when there is. Need to know about other SEs to determine this
        // std::string basis = ""; // Autodetermiend from couplingPoints matrix

    } Parameters;
    Parameters m_params;

    SimpleLeadSelfEnergy(Parameters params);
    virtual ~SimpleLeadSelfEnergy(){};

    void getSigmaIn(numType E, SelfEnergyMatrixType& dest) override;
    //virtual void getSigmaIn(SelfEnergyMatrixType& dest) override {return getSigmaIn(0,dest);};

    void getSigmaOut(numType E, SelfEnergyMatrixType& dest) override;
    //virtual void getSigmaOut(SelfEnergyMatrixType& dest) override {return getSigmaOut(0,dest);};

    void getSigmaR(complexType E, SelfEnergyMatrixType& dest) override{dest = zeroOrMatrix(E,m_SigmaR);}
    void getSigmaR(SelfEnergyMatrixType& dest) override;

    void getSigmaA(complexType E, SelfEnergyMatrixType& dest) override{dest = zeroOrMatrix(E,m_SigmaA);}
    void getSigmaA(SelfEnergyMatrixType& dest) override;

    void getSigmaK(numType E, SelfEnergyMatrixType& dest) override;


    // virtual void save(std::string filename) const override;
    // virtual void load(FILE* file) override;
    void setMatrixDimension(uint32_t dim) override {m_params.MatrixDimension = dim;}

    bool hasSigmaK() override {return true;}
    numType getChemicalPotential() override {return m_params.mu;}

    void setChemicalPotential(numType mu){m_params.mu = mu;}
    void getGamma(numType E, SelfEnergyMatrixType& dest) override;
    void clearCache() override;
    std::vector<numType> getContinuousRegions() override {return m_ConstantRegions;}
    void setBasis(const std::string& bas) override;
    void setBasis(const indexBasisT& bas) override {releaseAssert(bas[0] == bas[1],"setBasis,SimpleLeadSelfEnergy"); setBasis((bas[0]));};
    const indexBasisT& getBasis() override {return m_workingBasis;};

    void setT(numType t){m_params.t = t; computeSigmaR(0,m_SigmaR); computeSigmaA(0,m_SigmaA);}

    void setGreensFunction(std::shared_ptr<GreensFunction> gFunc){m_GFunc = gFunc;}
};
extern template class SimpleLeadSelfEnergy<ComplexSparseMatrix>;
extern template class SimpleLeadSelfEnergy<ComplexMatrix>;
#endif // SIMPLELEADSELFENERGY_H
