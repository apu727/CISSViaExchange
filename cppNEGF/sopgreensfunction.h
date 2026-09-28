#ifndef SOPGREENSFUNCTION_H
#define SOPGREENSFUNCTION_H
#include "linalg.h"
#include "selfenergybase.h"
#include "greensfunction.h"

class SOPGreensFunction
{
    friend class GreensFunctionTest;

    ComplexMatrix m_Ham;
    ComplexMatrix m_HamAdj;
    std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>>> m_selfEnergies;
    bool m_isBareBones = false; //Specifies that m_Ham,m_selfEnergies are invalid



    mutable enums::SpinSymmetry m_spinSym = enums::SpinSymmetry::RHF; // always start with RHF, this is then degraded by other matrices that are added. Set after cache is built
    indexBasisT m_basis = {"",""};
    // indexBasisT m_desiredBasis = {"",""};
    numType m_mu = 0; // Only use if this is not determinable from the self energies.

    /* internal state objects */

    /* Variables modified by buildCache() which needs to be callable on a const object*/
    mutable std::mutex m_cacheMutex;
    mutable bool m_GRIsCached = false;
    mutable std::vector<EigenVectorMatrix> m_rightEigenVectors;
    mutable std::vector<InvEigenVectorMatrix> m_InvRightEigenVectors;
    mutable std::vector<EigenValueVector> m_EigenValues;

    //The adjoint map turns the direct into a dual which means the inv and right are now backwards. But since the Adj =/= inv I think its better to call it InvAdj than RightAdj.
    mutable std::vector<InvEigenVectorMatrix> m_rightEigenVectorsAdj;//R^\dagger, needed when in a projected basis
    mutable std::vector<EigenVectorMatrix> m_InvRightEigenVectorsAdj; //R^{-1\dagger}, needed when in a projected basis
    mutable std::vector<EigenValueVector> m_EigenValuesAdj; // Should be the same but perhaps in a different order.

    mutable std::vector<Eigen::Vector<int,Eigen::Dynamic>> m_poleOrder; //\frac{1}{(E-\varepsilon_0)^order} i.e. 1 means 1/x
    mutable std::vector<Eigen::Vector<int,Eigen::Dynamic>> m_poleOrderAdj; //\frac{1}{(E-\varepsilon_0)^order} i.e. 1 means 1/x
    mutable std::vector<numType> m_muP;
    mutable numType m_muMin;
    mutable numType m_muMax;
    mutable std::vector<std::vector<SelfEnergyMatrix>> m_GammaP;
    /* Variables modified by buildCache() which needs to be callable on a const object*/

    std::vector<std::vector<ComplexMatrix>> m_Residues; // only used if m_isBareBones == true
    std::vector<std::vector<ComplexMatrix>> m_ResiduesAdj; // only used if m_isBareBones == true

    std::vector<numType> m_analyticRegions;

    //Cached Integrals
    ComplexMatrix m_Gn;
    bool m_GnIsCached = false;

    ComplexMatrix m_EGn;
    bool m_EGnIsCached = false;

    // bool isEquilibrium = false;
    /* Private functions */

    void buildCache() const;
    void generateAdjoint() const;


    void IntegrateAnalyticEquilibriumGn(ComplexMatrix& dest, numType mu, bool MultiplyByE, bool makeHermitian = true, numType EStart = negInf, numType EEnd = posInf);
    void IntegrateAnalyticNonEquilibriumGnOld(ComplexMatrix& dest, bool MultiplyByE);
    void IntegrateAnalyticNonEquilibriumGn(ComplexMatrix& dest, bool MultiplyByE, numType EStart = negInf, numType EEnd = posInf);


    bool findIndexOfRegion(complexType E, size_t& index) const;
    numType getRegionRepresentativeEnergy(size_t regionIndex) const;

    GreensFunction::EnforcedSpinSymmetryType m_enforcedSpinSym = GreensFunction::EnforcedSpinSymmetryType::autoDetect;
    bool m_forceSelfAdjointSelfEnergy = false;


public:
    SOPGreensFunction(const SOPGreensFunction& other); // I dont like this

    SOPGreensFunction(const ComplexSelfAdjointSparseMatrix& Ham,
                      const std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>>>& SEs,
                      const std::vector<numType>& analyticRegions,
                      GreensFunction::EnforcedSpinSymmetryType enforcedSpinSym);

    SOPGreensFunction(const ComplexMatrix& Ham,
                      numType mu,
                      const std::vector<numType>& analyticRegions,
                      GreensFunction::EnforcedSpinSymmetryType enforcedSpinSym);

    SOPGreensFunction(const ComplexMatrix& Ham,const ComplexMatrix& HamAdj,
                      numType mu,
                      const std::vector<numType>& analyticRegions,
                      GreensFunction::EnforcedSpinSymmetryType enforcedSpinSym);

    //works in the basis of the provided m_Ham
    const indexBasisT& getBasis(){return m_basis;}
    void setDesiredBasis(const indexBasisT& bas);

    void setForceSelfAdjointSelfEnergy(bool f){m_forceSelfAdjointSelfEnergy = f;}
    void setMu(numType mu){m_mu = mu;} // used if the self energies dont have a mu
    numType getMu(){return m_mu;}

    void getGr(complexType E, ComplexMatrix& dest) const;
    void getGa(complexType E, ComplexMatrix& dest);
    void getGk(complexType E, ComplexMatrix& dest);
    void getGkGrGa(complexType E, ComplexMatrix& Gk,ComplexMatrix& Gr,ComplexMatrix& Ga);
    void getGn(complexType E, ComplexMatrix& dest);
    void getGp(complexType E, ComplexMatrix& dest);

    void getGnIntegral(ComplexMatrix& dest, bool forceNonEquilibrium = false, numType EStart = negInf, numType EEnd = posInf);
    void getEGnIntegral(ComplexMatrix& dest, bool forceNonEquilibrium = false);
    void getGrIntegral(ComplexMatrix& dest);

    std::pair<long,long> getSpatialDimension() const; // The non-eigenvalue dimension. Not actually the spatial orbitals.
    long getEigenValueDimension()const {return m_GRIsCached ? m_EigenValues[0].rows() : m_Ham.rows();} // The eigenvalue dimension.
    void getDivergencesOfGr(numType E, std::vector<complexType> &divergences);

    //cant be const as may need to build the cache
    friend SOPGreensFunction operator *(const ComplexMatrix& mat, const SOPGreensFunction& GF);
    friend SOPGreensFunction operator *(const SOPGreensFunction& GF,const ComplexMatrix& mat);
    friend SOPGreensFunction operator *(const SOPGreensFunction& GFA, const SOPGreensFunction& GFB);

};

#endif // SOPGREENSFUNCTION_H
