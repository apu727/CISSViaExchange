#ifndef EXPLICITLEADSELFENERGY_H
#define EXPLICITLEADSELFENERGY_H

#include "selfenergybase.h"
template<class T>
class ExplicitLeadSelfEnergyTests;

template<class SelfEnergyMatrixType>
class ExplicitLeadSelfEnergy : public SelfEnergyBase<SelfEnergyMatrixType>
{
    friend class ExplicitLeadSelfEnergyTests<SelfEnergyMatrixType>;
    //E = BottomOfBand + 2*t (1 - cos(k*a))
    numType KFromE(numType E);

    numType fermiFunction(numType E);
    indexBasisT m_workingBasis = {};
public:
    typedef struct Parameters
    {
        numType a; // lattice spacing in lead
        numType alpha1; // coupling factor to system ?
        numType t; // finite difference kinetic energy of lead
        numType bottomOfBand; // bottom of band in lead
        numType mu; // chemical potential of lead

        std::vector<uint32_t> couplingPoints;
        uint32_t MatrixDimension;
        enums::SpinSymmetry sym = enums::SpinSymmetry::NoSpinSym;
        std::string basis = "";

    } Parameters;
    Parameters m_params;

    ExplicitLeadSelfEnergy(Parameters params);
    virtual ~ExplicitLeadSelfEnergy(){};

    virtual void getSigmaIn(numType E, SelfEnergyMatrixType& dest) override;
    virtual void getSigmaOut(numType E, SelfEnergyMatrixType& dest) override;
    virtual void getSigmaR(complexType E, SelfEnergyMatrixType& dest) override;
    virtual void getSigmaA(complexType E, SelfEnergyMatrixType& dest) override;
    virtual void getSigmaK(numType E, SelfEnergyMatrixType& dest) override;
    // virtual void save(std::string filename) const override;
    // virtual void load(FILE* file) override;
    virtual void setMatrixDimension(uint32_t dim) override {m_params.MatrixDimension = dim;};
    virtual bool hasSigmaK() override {return true;}
    virtual numType getChemicalPotential()override {return m_params.mu;}
    virtual void getGamma (numType E, SelfEnergyMatrixType& dest)override;
    virtual void setBasis(const std::string& bas) override{m_workingBasis = {bas,bas};}
    virtual void setBasis(const indexBasisT& bas) override{releaseAssert(bas[0]==bas[1],"bas[0]==bas[1],ExplicitLeadSelfEnergy"); setBasis(bas[0]);}
    virtual const indexBasisT& getBasis() override{return m_workingBasis;}
};

#endif // EXPLICITLEADSELFENERGY_H
