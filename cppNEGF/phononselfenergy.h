#ifndef PHONONSELFENERGY_H
#define PHONONSELFENERGY_H

#include "selfenergybase.h"
#include "greensfunction.h"
#include <memory>

template <class SelfEnergyMatrixType>
class PhononSelfEnergy : public SelfEnergyBase<SelfEnergyMatrixType>
{
    indexBasisT m_workingBasis = {};
    enums::SpinSymmetry m_spinSym = enums::SpinSymmetry::NoSpinSym;

    std::shared_ptr<GreensFunction> m_GFunc;
    ComplexSparseMatrix m_PhononCorrelationMatrix;
    SparseTensor<4,complexType> m_PhononGTensor;
    SparseTensor<2,complexType> m_PhononGDiagonalTensor;
    ComplexSparseMatrix m_PhononGDiagonalMatrix;
    ComplexSparseMatrix m_timeReversalMatrix;

    std::function<void(numType,ComplexSelfAdjointMatrix&)> m_bareGnfunc;
    std::function<void(numType,ComplexSelfAdjointMatrix&)> m_bareGpfunc;

    bool activeBuffer = false; // 0 & 1 binary switch
    std::vector<numType> m_EnergyDataPoints1;
    std::vector<SelfEnergyMatrixType> m_SigmaIn1;
    std::vector<SelfEnergyMatrixType> m_SigmaOut1;
    std::vector<SelfEnergyMatrixType> m_SigmaR1;
    //Pseudo circular buffer to allow temporary computations without realloc
    std::vector<numType> m_EnergyDataPoints2;
    std::vector<SelfEnergyMatrixType> m_SigmaIn2;
    std::vector<SelfEnergyMatrixType> m_SigmaOut2;
    std::vector<SelfEnergyMatrixType> m_SigmaR2;

    std::vector<SelfEnergyMatrixType>& getActiveSigmaIn(){return activeBuffer ? m_SigmaIn2 : m_SigmaIn1;}
    std::vector<SelfEnergyMatrixType>& getActiveSigmaOut(){return activeBuffer ? m_SigmaOut2 : m_SigmaOut1;}
    std::vector<SelfEnergyMatrixType>& getActiveSigmaR(){return activeBuffer ? m_SigmaR2 : m_SigmaR1;}
    std::vector<numType>& getActiveEnergyDataPoints(){return activeBuffer ? m_EnergyDataPoints2 : m_EnergyDataPoints1;}

    void CalcSigmaIn(numType E, SelfEnergyMatrixType& dest);
    void CalcSigmaOut(numType E, SelfEnergyMatrixType& dest);
    numType boseEinsteinDistribution(numType omega);
    void interpolate(const std::vector<SelfEnergyMatrixType>& matrices, const std::vector<numType>& Energies, const numType E, SelfEnergyMatrixType& dest);
    void asyncReComputeSelfE(int numberOfPoints, numType EnergyRange, std::vector<SelfEnergyMatrixType>* SigmaIn, std::vector<SelfEnergyMatrixType>* SigmaOut, std::vector<SelfEnergyMatrixType>* SigmaR, std::vector<numType>* EnergyDataPoints);
    void asyncMakeCausal(int numberOfPoints, std::vector<SelfEnergyMatrixType>* SigmaR);
public:
    typedef struct Parameters
    {
        std::vector<numType> omegas;
        numType lowestEnergy;
        numType largestEnergy;
        numType smallestEnergyFeature;
        int MatrixDimension;
    } Parameters;
    Parameters m_params;

    PhononSelfEnergy(Parameters params);
    PhononSelfEnergy(const PhononSelfEnergy& other) = delete;

    virtual ~PhononSelfEnergy(){};
    void setGreensFunction(std::shared_ptr<GreensFunction> GFunc);
    void setPhononCorrelationMatrix(const ComplexSparseMatrix& phononCorrelationMatrix){m_PhononCorrelationMatrix = phononCorrelationMatrix;}
    void setPhononGTensor(const SparseTensor<3,complexType>& PhononGTensor);

    void recomputeSelfEnergies() override;
    void getSigmaIn(numType E, SelfEnergyMatrixType& dest) override;
    void getSigmaOut(numType E, SelfEnergyMatrixType& dest) override;
    void getSigmaR(complexType E, SelfEnergyMatrixType& dest) override;
    void getSigmaA(complexType E, SelfEnergyMatrixType& dest) override;
    void getSigmaK(numType E, SelfEnergyMatrixType& dest) override;

    // virtual void save(std::string filename) const override;
    // virtual void load(FILE* file) override;
    virtual void setMatrixDimension(uint32_t dim) override {m_params.MatrixDimension = dim;}
    virtual bool hasSigmaK() override {return true;}
    virtual numType getChemicalPotential()override {return -1000;} // Chemical potential is not defined for phonons

    virtual const indexBasisT& getBasis() override {return m_workingBasis;}
    virtual void setBasis(const std::string& bas) override {m_workingBasis = {bas,bas};}
    virtual void setBasis(const indexBasisT& bas) override {releaseAssert(bas[0] == bas[1],"phononSE,setBasis"); setBasis(bas[0]);}


};

#endif // PHONONSELFENERGY_H
