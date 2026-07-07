#ifndef SELFENERGYBASE_H
#define SELFENERGYBASE_H

#include "linalgForwardDeclarations.h"
#include "logger.h"

typedef ComplexMatrix SelfEnergyMatrix;
// typedef ComplexSparseMatrix SelfEnergyMatrix;
enum class selfEnergyType
{
    noAssumptions = 0,
    Constant = 1,
    ConstantSelfConsistent = 2, //constant within an iteration but can change due to the Green's function changing. E.g. HF Hamiltonian
    ConstantPiecewise = 3, // Constant within an interval but discontinous at the Boundary.
    Analytic = 4, //In the complex number sense
};

template<class SelfEnergyMatrixType>
class SelfEnergyBase
{
public:
    selfEnergyType m_type = selfEnergyType::noAssumptions;
    std::string m_prettyName;
    SelfEnergyBase(std::string name = ""){m_prettyName = name;};
    virtual ~SelfEnergyBase(){};

    virtual void getSigmaIn(numType E, SelfEnergyMatrixType& dest) = 0;
    //virtual void getSigmaIn(SelfEnergyMatrixType& dest){assert(false);} // for constant self energies

    virtual void getSigmaOut(numType E, SelfEnergyMatrixType& dest) = 0;
    //virtual void getSigmaOut(SelfEnergyMatrixType& dest) {assert(false);}

    virtual void getSigmaR(complexType E, SelfEnergyMatrixType& dest) = 0;
    virtual void getSigmaR(SelfEnergyMatrixType& dest){releaseAssert(false,"This Self energy has not implemented Energy Independent SigmaR");}

    virtual void getSigmaA(complexType E, SelfEnergyMatrixType& dest) = 0;
    virtual void getSigmaA(SelfEnergyMatrixType& dest){releaseAssert(false,"This Self energy has not implemented Energy Independent SigmaA");}

    virtual void getSigmaK(numType E, SelfEnergyMatrixType& dest) = 0;
    virtual void getSigmaK(SelfEnergyMatrixType& dest){releaseAssert(false,"This Self energy has not implemented Energy Independent SigmaK");}

    virtual void getGamma(numType E, SelfEnergyMatrixType& dest){releaseAssert(false,"This Self energy has not implemented a Gamma");}

    virtual void recomputeSelfEnergies(){};
    // virtual void save(std::string filename) const =0;//{logger(filename,true).log("Name",m_prettyName);};
    // virtual void load(FILE* file) = 0;
    virtual void setMatrixDimension(uint32_t){};

    virtual bool hasSigmaK() = 0;
    virtual numType getChemicalPotential(){releaseAssert(false,"This Self energy has not implemented a chemical potential"); return 0;}

    //Lets it know the Green's function has changed so any dependent quantities (mu) need to be updated
    virtual void clearCache(){};
    virtual const indexBasisT& getBasis() = 0;

    //Note that This is called after every SCF iteration. So check if it has changed before doing work
    virtual void setBasis(const std::string&) = 0;
    virtual void setBasis(const indexBasisT&) = 0;

    virtual std::vector<numType> getContinuousRegions()
    {
        if (m_type == selfEnergyType::Constant || m_type == selfEnergyType::ConstantSelfConsistent)
        {
            return {negInf,posInf};
        }
        releaseAssert(false, "regions called on non-constant self energy in base self energy, This needs to be implemented in the derived class");
        return {};
    }
    bool isAnalytic() const
    {
        switch (m_type)
        {
        case selfEnergyType::Constant:
        case selfEnergyType::ConstantSelfConsistent:
        case selfEnergyType::ConstantPiecewise:
        case selfEnergyType::Analytic:
            return true;
        default:
            return false;
        }
    }

    bool isConstant() const
    {
        switch (m_type)
        {
        case selfEnergyType::Constant:
        case selfEnergyType::ConstantSelfConsistent:
            return true;
        default:
            return false;
        }
    }

    bool isConstantPiecewise() const // Constant => constant Piecewise.
    {
        switch (m_type)
        {
        case selfEnergyType::Constant:
        case selfEnergyType::ConstantSelfConsistent:
        case selfEnergyType::ConstantPiecewise:
            return true;
        default:
            return false;
        }
    }

};

template <>
inline void logger::logAccurate<selfEnergyType>(const std::string& name,selfEnergyType e)
{
    switch (e)
    {
    case selfEnergyType::noAssumptions:
        log(name,"NoAssumptions"); break;
    case selfEnergyType::Constant:
        log(name,"Constant"); break;
    case selfEnergyType::ConstantSelfConsistent:
        log(name,"ConstantSelfConsistent"); break;
    case selfEnergyType::ConstantPiecewise:
        log(name,"ConstantPiecewise"); break;
    case selfEnergyType::Analytic:
        log(name,"Analytic"); break;
    default:
        log(name,static_cast<std::underlying_type_t<selfEnergyType>>(e)); break;

    };
}

#endif // SELFENERGYBASE_H
