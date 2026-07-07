#ifndef FOCKHAMILTONIAN_H
#define FOCKHAMILTONIAN_H

#include "greensfunction.h"
template <typename errorVectorType, typename quantityType, auto innerProductFunc, bool EDIIS>
class DIIS;

template <typename errorVectorType, typename quantityType, auto innerProductFunc>
using PulayDIIS = DIIS<errorVectorType,quantityType,innerProductFunc,false>;


template <typename errorVectorType, typename quantityType, auto innerProductFunc>
using EDIIS = DIIS<errorVectorType,quantityType,innerProductFunc,true>;

inline numType complexMatrixRealInnerProduct(ComplexMatrix& a, ComplexMatrix& b) {return std::real(a.cdot(b));}

template <typename SelfEnergyMatrixType>
class FockHamiltonian : public SelfEnergyBase<SelfEnergyMatrixType>
{//Everything is in the basis that was originally provided. Probably an MO basis from a converged HF calculation

    ComplexDualSelfAdjointMatrix m_suppliedDensityMatrix;
    ComplexDirectSelfAdjointMatrix m_suppliedOneElectronIntegrals;

    int m_basisSize = 0;
    numType m_lowerEnergyBound = 0;
    numType m_upperEnergyBound = 0;
    unsigned long m_numberIntegrationSteps = 0;
    SelfEnergyMatrixType m_HamMatrix;
    numType m_errorEstimate = NAN;
    indexBasisT m_basis = {};
    indexBasisT m_notProjectedBasis = {};
    bool m_basisIsProjected = false;


    ComplexSelfAdjointSparseMatrix m_oneElectronIntegrals; // h^{p}_{q}

    SparseTensor<4,complexType> m_HFTwoElectronIntegrals; //T^{pq}_{rs} =  h^{pq}_{rs} - h^{qp}_{rs}, prepared for a G^{r}_{q} T^{pq}_{rs} einsum.

    std::vector<int> m_initialOccupations; // only important for the first iteration when we do not have a green's function yet.
    ComplexDualSelfAdjointMatrix m_Gn; // In AO basis for compatibility reasons
    std::shared_ptr<GreensFunction> m_GFunc;

    //Zeros that arent on the stack so can be returned by reference
    ComplexSparseMatrix m_phononCorrelationMatrix;
    SparseTensor<3,complexType> m_phononGTensor;

    // Builds the JK matrix F_{\bar{q}s} from (Gn)^{r\bar{p}} using h_{\bar{p}\bar{q}rs}
    std::function<ComplexDirectSelfAdjointMatrix(ComplexDualSelfAdjointMatrix)> m_JKBuilder;
    ComplexDirectSelfAdjointMatrix buildJK(ComplexDualSelfAdjointMatrix Gn);

    //Gives  F^{q}_{s}
    void constructFockMatrix(const ComplexDualSelfAdjointMatrix& IntGLesser);
    void constructHFTwoElectronIntegrals(const  SparseTensor<4,complexType>& twoElectronIntegrals);
    //Gives back (Gn)^{r\bar{p}}
    void computeGnIntegral(ComplexDualSelfAdjointMatrix& dest);

    //DIIS Stuff
    void DIISStep(ComplexDualSelfAdjointMatrix& Gn); // Does a DIIS step based on the current Fock matrix and Gn in the AO basis (Gn)^{r\bar{p}}
    void DDIISStep(ComplexDualSelfAdjointMatrix& Gn);
    void CDIISStep(ComplexDualSelfAdjointMatrix& Gn); // Can only be used in equilibrium situations
    void RCAStep(ComplexDualSelfAdjointMatrix& Gn);
    std::list<ComplexDualSelfAdjointMatrix> m_pastGn; // used in DIIS & proper Average
    std::list<ComplexMatrix> m_pastGrInt; // used in DIIS & proper Average
    std::list<ComplexMatrix> m_pastEVecs; // used in DIIS
    std::list<ComplexSelfAdjointMatrix> m_pastJKMatrices;

    std::list<numType> m_pastEnergies;
    std::list<ComplexDirectSelfAdjointMatrix> m_pastJKAOMatrices;

    bool m_doneFirstDIIS = false;
    bool m_forceDownhillDIIS = false;
    // ComplexMatrix m_lastGn; // used in downhill diis
    // ComplexSparseMatrix m_lastHamMatrix; //downhill diis
    // state variables for downhill diis
    bool m_doingLineSearch = false;
    bool m_lineSearchFailed = false;

    ComplexMatrix m_LowdinOrthogonalise;
    ComplexMatrix m_LowdinUnOrthogonalise;

    ComplexMatrix m_lastGrInt; // DIIS
    ComplexMatrix m_DIISHam;// Includes all the self energies and stuff
    size_t m_DIISIterCount = 0;
    ComplexDualSelfAdjointMatrix m_snapShotGn;
    ComplexMatrix m_snapShotGrInt;
    ComplexSelfAdjointSparseMatrix m_snapShotHam;
    std::vector<numType> m_stabilityErrors;
    numType CDIISInitialAlpha = 0.3;
    numType m_CDIISAlpha = CDIISInitialAlpha;

    enum class DIISTYPE
    {
        unset = 0,
        DDIIS = 1,
        CDIIS = 2,
        RCA = 3,
    };

    DIISTYPE m_DIISType = DIISTYPE::unset;
    bool m_useEDIIS = false;
    std::shared_ptr<EDIIS<ComplexMatrix,ComplexSelfAdjointMatrix,complexMatrixRealInnerProduct>> m_EDIISPtr;
    std::shared_ptr<PulayDIIS<ComplexMatrix,ComplexSelfAdjointMatrix,complexMatrixRealInnerProduct>> m_PulayDIISPtr;



    void init(const ComplexDualSelfAdjointMatrix& initialDensityMatrix, const ComplexDirectSelfAdjointMatrix& oneElectronIntegrals);
    bool m_debugFockGeneration = false;


public:
    /* Initial Occupations are Spin interleaved i.e. \alpha \beta \alpha \beta virtual\alpha virtual\beta
     *
     * One Electron Integrals are in the AO Spin orbital basis
     * Two Electron Integrals are in the AO Spin orbital basis
     * It all gets converted to the Spin orbital basis internally anyway.
     * If the metric is diagonal then expects the matrix elements to be in the natural representation. i.e. H = h^p_q \psi_p \psi^q + \half  h^{pq}_{rs} \psi_{p}\psi_{q} \psi^s \psi^r and uses them as is
     * If the metric is not diagonal then expects matrix elements in AO basis and calculates h^p_q = S^{p\Bar{i}} h_{\bar{i}q} and h^{pq}_{rs} = S^{p\Bar{i}S^{q\Bar{j}h_{\Bar{i}\Bar{j}rs}
    */
    FockHamiltonian(const ComplexDualSelfAdjointMatrix& initialDensityMatrix, const ComplexDirectSelfAdjointMatrix& oneElectronIntegrals, const  SparseTensor<4,complexType>& twoElectronIntegrals);
    FockHamiltonian(const ComplexDualSelfAdjointMatrix& initialDensityMatrix, const ComplexDirectSelfAdjointMatrix& oneElectronIntegrals, std::function<ComplexDirectSelfAdjointMatrix(ComplexDualSelfAdjointMatrix)> JKBuilder);

    //Used for debugging to compare two methods
    FockHamiltonian(const ComplexDualSelfAdjointMatrix& initialDensityMatrix, const ComplexDirectSelfAdjointMatrix& oneElectronIntegrals, const  SparseTensor<4,complexType>& twoElectronIntegrals,std::function<ComplexDirectSelfAdjointMatrix(ComplexDualSelfAdjointMatrix)> JKBuilder);

    virtual ~FockHamiltonian(){};

    void setGreensFunction(std::shared_ptr<GreensFunction> gFunc){m_GFunc = gFunc;}
    void setIntegrationParameters(numType lower, numType upper, unsigned long steps){m_lowerEnergyBound = lower; m_upperEnergyBound = upper; m_numberIntegrationSteps = steps;}
    ComplexSelfAdjointMatrix getGn(){ComplexSelfAdjointMatrix ret; m_Gn.toBasisC(ret,m_Gn.getBasis()); return ret;}


    //Self Energy Virtual functions
    virtual void getSigmaIn(numType E, SelfEnergyMatrixType& dest) override{dest.setZero(m_basisSize,m_basisSize,m_basis,m_Gn.getSpinSym());};

    virtual void getSigmaOut(numType E, SelfEnergyMatrixType& dest)override {dest.setZero(m_basisSize,m_basisSize,m_basis,m_Gn.getSpinSym());};

    virtual void getSigmaR(complexType E, SelfEnergyMatrixType& dest)override{dest = static_cast<SelfEnergyMatrixType>(m_HamMatrix);}
    virtual void getSigmaR(SelfEnergyMatrixType& dest)override{dest = static_cast<SelfEnergyMatrixType>(m_HamMatrix);}

    virtual void getSigmaA(complexType E, SelfEnergyMatrixType& dest)override{dest = static_cast<SelfEnergyMatrixType>(m_HamMatrix);}
    virtual void getSigmaA(SelfEnergyMatrixType& dest)override{dest = static_cast<SelfEnergyMatrixType>(m_HamMatrix);}

    virtual void getSigmaK(numType E, SelfEnergyMatrixType& dest)override{dest.setZero(m_basisSize,m_basisSize,m_basis,m_Gn.getSpinSym());}
    virtual void getSigmaK(SelfEnergyMatrixType& dest)override{dest.setZero(m_basisSize,m_basisSize,m_basis,m_Gn.getSpinSym());}


    virtual void recomputeSelfEnergies() override;
    // virtual void save(std::string filename) const override{logger().log("not Implemented");}
    // virtual void load(FILE* file)override{logger().log("not Implemented");}
    virtual void setMatrixDimension(uint32_t s)override{if (s != m_basisSize) logger().log("Tried to set basis size to an inconsistent value in Fock Hamiltonian",std::vector<long>({s,m_basisSize}));};
    virtual void getGamma(numType E, SelfEnergyMatrixType& dest) override{dest.setZero(m_basisSize,m_basisSize,m_basis,m_Gn.getSpinSym());}
    virtual bool hasSigmaK() override {return false;}
    virtual void clearCache() override;
    void resetDIISCache();
    virtual void setBasis(const std::string& bas) override{setBasis({bas,bas});}
    virtual void setBasis(const indexBasisT& bas) override;
    virtual const indexBasisT& getBasis() override{return m_basis;}

    numType getErrorEstimate(){return m_errorEstimate;}

    void setUseEDIIS(bool s){m_useEDIIS = s;}
    void setCDIISInitialAlpha(numType s){CDIISInitialAlpha = s;}


};
extern template class FockHamiltonian<SelfEnergyMatrix>;
#endif // FOCKHAMILTONIAN_H
