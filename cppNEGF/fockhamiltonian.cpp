#include "fockhamiltonian.h"
#include "EinSumTemplate.h"
#include "integrator.h"
#include "Eigen/LU"
#include "Eigen/Eigenvalues"
#include "Eigen/Dense"
#include "diis.h"


const bool doDIIS = true; // Density does not commute with the Fock matrix. G^R does.
// 6n = \iu \int f(E) G^R - G^A \td E ish
//G^R commutes with H^R, G^A commutes with H^A so you may guess
// [H^R,G^R] = error. Then since \dagger and - are both linear operations. Extrapolating G^R leads to an extrapolation of the density.
// Requires us to get G^R integral. This is divergent, but ignoring the divergence since it cancels with the divergence in G^A we could do this.
const int CDIISSize = 15; // Increase if convergence problems, QCHEM does 15
const int RCASize = 15; // Increase if convergence problems, QCHEM does 15
const int DDIISSize = 6; // Increase if convergence problems, QCHEM does 15
const bool doDownhillDIIS = false;
const bool doProperAverage = false;
const bool doOrthogonalCommutators = false;
const int properAverageSize = 11;


template<typename SelfEnergyMatrixType>
void FockHamiltonian<SelfEnergyMatrixType>::init(const ComplexDualSelfAdjointMatrix& initialDensityMatrix, const ComplexDirectSelfAdjointMatrix& oneElectronIntegrals)
{


    oneElectronIntegrals.sparseView().toBasisC(m_oneElectronIntegrals,m_basis);

    m_basisSize = m_oneElectronIntegrals.rows();

    if (initialDensityMatrix.rows() != 0)
        initialDensityMatrix.toBasisC(m_Gn,m_notProjectedBasis);
    else
        m_Gn.setZero(m_basisSize,m_basisSize,m_notProjectedBasis,enums::SpinSymmetry::RHF);

    constructFockMatrix(m_Gn);

    m_phononCorrelationMatrix.resize(m_basisSize,m_basisSize,m_notProjectedBasis,enums::SpinSymmetry::NoSpinSym);
    m_phononGTensor.setSize(m_basisSize);
}


template<typename SelfEnergyMatrixType>
FockHamiltonian<SelfEnergyMatrixType>::FockHamiltonian(const ComplexDualSelfAdjointMatrix& initialDensityMatrix, const ComplexDirectSelfAdjointMatrix& oneElectronIntegrals, std::function<ComplexDirectSelfAdjointMatrix(ComplexDualSelfAdjointMatrix)> JKBuilder)
{
    this->m_type = selfEnergyType::ConstantSelfConsistent;
    m_JKBuilder = JKBuilder;
    m_suppliedDensityMatrix = initialDensityMatrix;
    m_suppliedOneElectronIntegrals = oneElectronIntegrals;
}


template<typename SelfEnergyMatrixType>
FockHamiltonian<SelfEnergyMatrixType>::FockHamiltonian(const ComplexDualSelfAdjointMatrix& initialDensityMatrix, const ComplexDirectSelfAdjointMatrix& oneElectronIntegrals, const SparseTensor<4, complexType> &twoElectronIntegrals)
{
    this->m_type = selfEnergyType::ConstantSelfConsistent;
    constructHFTwoElectronIntegrals(twoElectronIntegrals);
    m_JKBuilder = [this](ComplexDualSelfAdjointMatrix Gn) {return this->buildJK(Gn);};
    m_suppliedDensityMatrix = initialDensityMatrix;
    m_suppliedOneElectronIntegrals = oneElectronIntegrals;
}

template<typename SelfEnergyMatrixType>
FockHamiltonian<SelfEnergyMatrixType>::FockHamiltonian(const ComplexDualSelfAdjointMatrix& initialDensityMatrix, const ComplexDirectSelfAdjointMatrix& oneElectronIntegrals, const SparseTensor<4, complexType> &twoElectronIntegrals, std::function<ComplexDirectSelfAdjointMatrix(ComplexDualSelfAdjointMatrix)> JKBuilder)
{
    this->m_type = selfEnergyType::ConstantSelfConsistent;
    m_JKBuilder = JKBuilder;
    constructHFTwoElectronIntegrals(twoElectronIntegrals);
    m_debugFockGeneration = true;
    m_suppliedDensityMatrix = initialDensityMatrix;
    m_suppliedOneElectronIntegrals = oneElectronIntegrals;
}

template<typename SelfEnergyMatrixType>
ComplexDirectSelfAdjointMatrix FockHamiltonian<SelfEnergyMatrixType>::buildJK(ComplexDualSelfAdjointMatrix Gn)
{
    //Gn is in AO Basis i.e.(Gn)^{r\Bar{p}}
    ComplexDirectSelfAdjointMatrix JK;
    SparseTensorView<ComplexDirectSelfAdjointMatrix,complexType,false> HamView(JK);
    SparseTensorView<ComplexDualSelfAdjointMatrix,complexType,false> GLesserView(Gn);
    auto start1 = std::chrono::high_resolution_clock::now();
    Einsum<4,complexType,2,complexType,2,complexType>("pqrs,rp,qs")(&m_HFTwoElectronIntegrals,&GLesserView,&HamView); //h_{\Bar{p}\Bar{q}rs} (G^<)^{r\Bar{p}}
#warning TODO incorporate Tensors and einsum into the dual/direct notation
    auto start2 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(start2 - start1);
    logger().log("FockEinsum Timings us",std::vector<long>({duration1.count()}));
    return JK;
}

template<typename SelfEnergyMatrixType>
void FockHamiltonian<SelfEnergyMatrixType>::constructFockMatrix(const ComplexDualSelfAdjointMatrix &Gn)
{
    if (m_debugFockGeneration)
    {
        ComplexDirectSelfAdjointMatrix JKFromTensor;
        JKFromTensor = buildJK(Gn);

        ComplexDirectSelfAdjointMatrix JKFromFunc;

        JKFromFunc = m_JKBuilder(Gn);

        ComplexDirectSelfAdjointMatrix Diff = JKFromTensor - JKFromFunc;
        ComplexSelfAdjointMatrix  Diff2;
        Diff.toBasisC(Diff2,Diff.getBasis());
        logger().log("JKDiffNorm:", Diff2.norm());
        numType JKEnergyFromTensor;
        JKEnergyFromTensor = (JKFromTensor * Gn).trace().real();

        numType JKEnergyFromFunc;        
        JKEnergyFromFunc = (JKFromFunc * Gn).trace().real();

        logger().log("JKEnergyDiff:", JKEnergyFromTensor-JKEnergyFromFunc);
    }

    if constexpr (SelfEnergyMatrixType::isSparse)
    {
        ComplexDirectSelfAdjointSparseMatrix JK;
        JK =  m_JKBuilder(Gn).sparseView();
        JK.toBasisC(m_HamMatrix,m_basis);
    }
    else
    {
        ComplexDirectSelfAdjointMatrix JK;
        JK = m_JKBuilder(Gn);
        JK.toBasisC(m_HamMatrix,m_basis);
    }


    m_HamMatrix += m_oneElectronIntegrals;

    // logger().log("ElecEnergyCalculated:", (IntGLesser * m_HamMatrix).trace()*(-iu/(2*M_PI)));
}

template<typename SelfEnergyMatrixType>
void FockHamiltonian<SelfEnergyMatrixType>::computeGnIntegral(ComplexDualSelfAdjointMatrix& dest)
{//TODO adaptive Quadrature or something
    assert(m_GFunc.get() != nullptr);
    ComplexSelfAdjointMatrix temp;
    m_GFunc->getGnIntegral(0,temp,false);// Dont want perturbative as this would fold it in to the skeleton expansion.
    logger().logAccurate("IntGLesser Trace Real",temp.trace()*(1./(2*M_PI)));
    // logger().log("IntGLesser Trace Imag",temp.trace()*(1./(2*M_PI)),"%.10lf");
    // temp *= 1./(2*M_PI);
    temp.toBasisC(dest,m_notProjectedBasis);

    dest *= 1./(2*M_PI);
}

template<typename SelfEnergyMatrixType>
void FockHamiltonian<SelfEnergyMatrixType>::DIISStep(ComplexDualSelfAdjointMatrix &GnAO)
{//Do DIIS step based on Fock matrix and AO (Gn)^{r\bar{p}}

    //CHOOSe between diis methods
    bool canComputeEqGr = m_GFunc->isEquilibriumGn() && m_GFunc->getComputationMethod() == GreensFunction::GrComputationType::AnalyticSOP && !m_basisIsProjected;
    if (m_DIISType == DIISTYPE::unset)
    {
        if (canComputeEqGr)
            m_DIISType = DIISTYPE::CDIIS;
        else
            m_DIISType = DIISTYPE::DDIIS;
    }
    if (canComputeEqGr && m_DIISType == DIISTYPE::DDIIS)
    {
        //Switch to CDIIS
        m_DIISType = DIISTYPE::CDIIS;
        resetDIISCache();
    }
    else if (!canComputeEqGr && m_DIISType == DIISTYPE::CDIIS)
    {
        //Switch to DDIIS
        m_DIISType = DIISTYPE::DDIIS;
        resetDIISCache();
    }

    if (m_DIISType == DIISTYPE::CDIIS)
    {
        CDIISStep(GnAO);
    }
    else if (m_DIISType == DIISTYPE::RCA)
    {
        RCAStep(GnAO);
    }
    else if (m_DIISType == DIISTYPE::DDIIS)
    {
        DDIISStep(GnAO);
    }
    else
        releaseAssert(false,"Unhandled dIIS switch case");


}

template<typename SelfEnergyMatrixType>
void FockHamiltonian<SelfEnergyMatrixType>::DDIISStep(ComplexDualSelfAdjointMatrix &GnAO)
{
    ComplexSelfAdjointMatrix Gn,MGn;
    GnAO.toBasisC(Gn,m_notProjectedBasis);



    auto start1 = std::chrono::high_resolution_clock::now();
    ComplexSelfAdjointMatrix currEVec;

    m_Gn.toBasisC(MGn,m_notProjectedBasis);
    currEVec = Gn/(Gn.trace()) - MGn/(MGn.trace());
    // currEVec = Gn - MGn;

    m_errorEstimate = currEVec.norm();
    // //Remove this to reenable
    // logger().log("DDIIS Curr error",m_errorEstimate);
    // m_Gn = (m_Gn * (1-0.05) + GnAO*0.05);
    // constructFockMatrix(m_Gn);
    // return;
    // //End remove
    ComplexMatrix errorMat;
    if (m_useEDIIS)
    {
        if (!m_EDIISPtr) m_EDIISPtr = std::make_shared<std::remove_reference_t<decltype(*m_EDIISPtr)>>(DDIISSize);
        m_EDIISPtr->addNew(Gn,static_cast<ComplexMatrix>(currEVec));
        m_EDIISPtr->getNext(MGn,errorMat);
    }
    else
    {
        if (!m_PulayDIISPtr) m_PulayDIISPtr = std::make_shared<std::remove_reference_t<decltype(*m_PulayDIISPtr)>>(DDIISSize);
        m_PulayDIISPtr->addNew(Gn,static_cast<ComplexMatrix>(currEVec));
        m_PulayDIISPtr->getNext(MGn,errorMat);
    }
    MGn.toBasisC(m_Gn,m_notProjectedBasis);



    auto start2 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(start2 - start1);
    logger().log("DDIIS Timings us",std::vector<long>({duration1.count()}));

    logger().log("DDIIS Extrapolated error",errorMat.norm());
    logger().log("DDIIS Curr error",currEVec.norm());

    constructFockMatrix(m_Gn);
}

template<typename SelfEnergyMatrixType>
void FockHamiltonian<SelfEnergyMatrixType>::CDIISStep(ComplexDualSelfAdjointMatrix &GnAO)
{
    if (m_useEDIIS && !m_EDIISPtr)
    {
        m_EDIISPtr = std::make_shared<std::remove_reference_t<decltype(*m_EDIISPtr)>>(CDIISSize);
    }
    else if (!m_PulayDIISPtr)
    {
        m_PulayDIISPtr = std::make_shared<std::remove_reference_t<decltype(*m_PulayDIISPtr)>>(CDIISSize);
    }

    m_DIISIterCount++;

    ComplexMatrix GrInt;
    m_GFunc->getGrIntegral(false,GrInt);
    GrInt *= -iu;

    auto start1 = std::chrono::high_resolution_clock::now();
    ComplexMatrix currEVec;
    if (m_doneFirstDIIS == false)
    {
        m_DIISHam = static_cast<ComplexMatrix>(m_GFunc->getHamiltonian());
        SelfEnergyMatrixType temp;
        for (auto& se : m_GFunc->getSelfEnergies())
        {
            if (se.get() == static_cast<SelfEnergyBase<SelfEnergyMatrixType>*>(this))
                continue; // ignore me, Assuming only one Fock Hamiltonian
            se->getSigmaR(0,temp);
            m_DIISHam += temp;
        }
        m_doneFirstDIIS = true;
        m_lastGrInt = GrInt;
    }
    else
    {
        GnAO = m_Gn*(1-m_CDIISAlpha) + GnAO*m_CDIISAlpha;
        GrInt = m_lastGrInt*(1-m_CDIISAlpha) + GrInt*m_CDIISAlpha;
    }


    ComplexMatrix nextHam;
    ComplexSelfAdjointMatrix JKNow;
    auto start2 = std::chrono::high_resolution_clock::now();
    m_JKBuilder(GnAO).toBasisC(JKNow,m_notProjectedBasis);
    JKNow.makeSelfAdjoint();
    auto start3 = std::chrono::high_resolution_clock::now();
    nextHam = m_DIISHam + JKNow + m_oneElectronIntegrals;

    // If we split G^R = (G^R + G^A)/2 + (G^R - G^A)/2 and likewise for H^R
    // then we can evaluate the Hermitian commutators as G^H H^H - H^H G^H = G^H H^H - (G^H H^H)^\dagger
    // the Hermitian-antihermitian parts are G^H H^A - H^A G^H = G^H H^A + (G^H H^A)^\dagger
    // and G^A H^H - H^H G^A = = G^A H^H + (G^A H^H)^\dagger
    // the Anti-hermitian and Antihermitian parts are
    // G^A H^A - H^A G^A = (G^A H^A) - (G^A H^A)^\dagger.
    // if we do this in an orthogonal basis, e.g. the Lowdin orthogonal basis then this probably has better stability?


    currEVec =  GrInt* nextHam - nextHam * GrInt;

#warning CHANGEHERE
    checkBasisChangeAssert(currEVec.toOrthogonalBasis(currEVec),GrInt.getBasis(),{"OrthogonalBasis","OrthogonalBasis"});
    if (m_useEDIIS)
        m_EDIISPtr->addNew(JKNow,currEVec);
    else
        m_PulayDIISPtr->addNew(JKNow,currEVec);
    m_pastGrInt.push_back(GrInt);
    m_pastGn.push_back(GnAO);
    if (m_pastGrInt.size() > CDIISSize)
    {
        m_pastGrInt.pop_front();
        m_pastGn.pop_front();
    }

    numType currError = currEVec.norm();
    numType QChemError;
    numType QChemError2;
    {
        ComplexDirectMatrix QCHEMEVEC;
        if (BasisManager::exists("AO"))
            currEVec.toBasisC(QCHEMEVEC,"AO");
        else
            currEVec.toBasisC(QCHEMEVEC,currEVec.getBasis());
        QChemError = QCHEMEVEC.getEigenOptimisation().norm()/QCHEMEVEC.rows();
        QChemError2 = QCHEMEVEC.getEigenOptimisation().lpNorm<Eigen::Infinity>();
    }
    m_errorEstimate = currError + ((GnAO-m_Gn)/m_CDIISAlpha).norm();

    auto start4 = std::chrono::high_resolution_clock::now();
    ComplexMatrix errorMat;
    ComplexSelfAdjointMatrix FinalHamMatrix;
    Eigen::VectorXd CoeffVec;
    if (m_useEDIIS)
        m_EDIISPtr->getNext(FinalHamMatrix,errorMat,&CoeffVec);
    else
        m_PulayDIISPtr->getNext(FinalHamMatrix,errorMat,&CoeffVec);


    auto start5 = std::chrono::high_resolution_clock::now();
    FinalHamMatrix += m_oneElectronIntegrals;

    m_lastGrInt.setZero();
    m_Gn.setZero();
    m_Gn.setSpinSym(GnAO.getSpinSym());
    auto pastGrIntIt = m_pastGrInt.begin();
    auto pastGnIt = m_pastGn.begin();
    assert(CoeffVec.size() == m_pastGrInt.size());
    for (long i = 0; i < CoeffVec.size(); i++)
    {
        m_Gn += *pastGnIt * CoeffVec[i];
        m_lastGrInt += *pastGrIntIt *CoeffVec[i];
        pastGrIntIt++;
        pastGnIt++;
    }

    if constexpr (SelfEnergyMatrixType::isSparse)
        m_HamMatrix = FinalHamMatrix.sparseView();
    else
        m_HamMatrix = static_cast<SelfEnergyMatrixType>(FinalHamMatrix);
    
    //Calc error

    auto start6 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(start2 - start1);
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(start3 - start2);
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(start4 - start3);
    auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(start5 - start4);
    auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(start6 - start5);
    logger().log("CDIIS Timings us",std::vector<long>({duration1.count(),duration2.count(),duration3.count(), duration4.count(),duration5.count()}));

    // logger().log("CDIIS Extrapolated error norm",errorMat.norm());
    logger().log("CDIIS Curr errorEstimate ",m_errorEstimate);
    logger().log("CDIIS Curr QChem style RMS",QChemError);
    logger().log("CDIIS Curr QChem style MaxError",QChemError2);

}
template<typename SelfEnergyMatrixType>
void FockHamiltonian<SelfEnergyMatrixType>::RCAStep(ComplexDualSelfAdjointMatrix &GnAO)
{
    // m_DIISIterCount++;
    // if (!m_EDIISPtr)
    //     m_EDIISPtr = std::make_shared<std::remove_reference_t<decltype(*m_EDIISPtr)>>(RCASize);


    // auto start1 = std::chrono::high_resolution_clock::now();
    // if (m_doneFirstDIIS == false)
    // {
    //     m_DIISHam = static_cast<ComplexMatrix>(m_GFunc->getHamiltonian());
    //     SelfEnergyMatrixType temp;
    //     for (auto& se : m_GFunc->getSelfEnergies())
    //     {
    //         if (se.get() == static_cast<SelfEnergyBase<SelfEnergyMatrixType>*>(this))
    //             continue; // ignore me, Assuming only one Fock Hamiltonian
    //         se->getSigmaR(0,temp);
    //         m_DIISHam += temp;
    //     }
    //     m_doneFirstDIIS = true;
    // }


    // ComplexMatrix nextHam;
    // ComplexSelfAdjointMatrix JKNow;
    // auto start2 = std::chrono::high_resolution_clock::now();
    // ComplexDirectSelfAdjointMatrix JKAONow = m_JKBuilder(GnAO);
    // JKAONow.toBasisC(JKNow,m_basis);
    // auto start3 = std::chrono::high_resolution_clock::now();

    // m_pastJKAOMatrices.push_back(JKAONow);
    // m_pastGn.push_back(GnAO);
    // if (m_pastGn.size() > RCASize)
    // {
    //     m_pastJKMatrices.pop_front();
    //     m_pastJKAOMatrices.pop_front();
    //     m_pastGn.pop_front();
    // }
    // size_t currDIISSize = m_pastGn.size();
    // m_EDIISPtr->addNew(JKNow,ComplexMatrix());

    // auto iIteratorJK = m_pastJKAOMatrices.begin();
    // auto iIteratorGn = m_pastGn.begin();

    // Eigen::MatrixXd B(currDIISSize,currDIISSize);
    // for (size_t i = 0; i < currDIISSize; i++)
    // {
    //     auto jIteratorJK = m_pastJKAOMatrices.begin();
    //     auto jIteratorGn = m_pastGn.begin();
    //     for (size_t j = 0; j < currDIISSize; j++)
    //     {
    //         B(i,j) = (*iIteratorGn - *jIteratorGn).cdot(*iIteratorJK - *jIteratorJK)
    //     }
    // }

    // m_errorEstimate = (GnAO-m_Gn).norm();

    // auto start4 = std::chrono::high_resolution_clock::now();
    // ComplexMatrix errorMat;
    // ComplexSelfAdjointMatrix FinalHamMatrix;
    // Eigen::VectorXd CoeffVec;
    // if (m_useEDIIS)
    //     m_EDIISPtr->getNext(FinalHamMatrix,errorMat,&CoeffVec);
    // else
    //     m_PulayDIISPtr->getNext(FinalHamMatrix,errorMat,&CoeffVec);


    // auto start5 = std::chrono::high_resolution_clock::now();
    // FinalHamMatrix += m_oneElectronIntegrals;

    // m_Gn.setZero();
    // auto pastGrIntIt = m_pastGrInt.begin();
    // auto pastGnIt = m_pastGn.begin();
    // assert(CoeffVec.size() == m_pastGrInt.size());
    // for (long i = 0; i < CoeffVec.size(); i++)
    // {
    //     m_Gn += *pastGnIt * CoeffVec[i];
    //     m_lastGrInt += *pastGrIntIt *CoeffVec[i];
    //     pastGrIntIt++;
    //     pastGnIt++;
    // }

    // m_HamMatrix = FinalHamMatrix.sparseView();
    // //Calc error

    // auto start6 = std::chrono::high_resolution_clock::now();
    // auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(start2 - start1);
    // auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(start3 - start2);
    // auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(start4 - start3);
    // auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(start5 - start4);
    // auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(start6 - start5);
    // logger().log("CDIIS Timings us",std::vector<long>({duration1.count(),duration2.count(),duration3.count(), duration4.count(),duration5.count()}));

    // // logger().log("CDIIS Extrapolated error norm",errorMat.norm());
    // logger().log("CDIIS Curr errorEstimate ",m_errorEstimate);
}

template<typename SelfEnergyMatrixType>
void FockHamiltonian<SelfEnergyMatrixType>::resetDIISCache()
{
    m_pastGn.clear();
    m_pastEVecs.clear();
    m_pastJKMatrices.clear();
    m_pastEnergies.clear();
    m_pastJKAOMatrices.clear();
    m_pastGrInt.clear();
    m_lastGrInt = ComplexMatrix();
    m_doneFirstDIIS = false;
    m_stabilityErrors.clear();
    m_DIISIterCount = 0;
    m_CDIISAlpha = CDIISInitialAlpha;
    m_errorEstimate = NAN;
    m_EDIISPtr = nullptr;
    m_PulayDIISPtr = nullptr;
}

template<typename SelfEnergyMatrixType>
void FockHamiltonian<SelfEnergyMatrixType>::recomputeSelfEnergies()
{
    ComplexDualSelfAdjointMatrix Gn;
    computeGnIntegral(Gn);//(G^<)^r_p = \psi^r><\psi_p
    if (doDIIS)
    {
        DIISStep(Gn); // DIIS handles building of the fock matrix
        return;
    }
    else if (doProperAverage)
    {
        m_pastGn.push_back(Gn);
        if (m_pastGn.size() > properAverageSize)
            m_pastGn.pop_front();

        m_Gn.setZero();
        for (auto& g : m_pastGn)
            m_Gn += g;
        m_Gn /= m_pastGn.size();
        {
            ComplexSelfAdjointMatrix temp;
            (m_Gn - m_pastGn.back()).toBasisC(temp,m_Gn.getBasis());
            m_errorEstimate = temp.norm();
        }

    }
    else
    {
        {
            ComplexSelfAdjointMatrix temp;
            (m_Gn - Gn).toBasisC(temp,m_Gn.getBasis());
            m_errorEstimate = temp.norm();
        }

        m_Gn = (m_Gn * (1-0.3) + Gn * 0.3);
    }
    constructFockMatrix(m_Gn);
}

template<typename SelfEnergyMatrixType>
void FockHamiltonian<SelfEnergyMatrixType>::clearCache()
{
    //DIIS???
    // std::list<ComplexMatrix> m_pastGn; // used in DIIS
    // std::list<ComplexMatrix> m_pastEVecs; // used in DIIS
    // m_DIISHam.setZero();// Includes all the self energies and stuff
}

template<typename SelfEnergyMatrixType>
void FockHamiltonian<SelfEnergyMatrixType>::setBasis(const indexBasisT& bas)
{
    if (bas == m_basis)
        return;
    if (BasisManager::isProjection(bas[0]) || BasisManager::isProjection(bas[1]))
    {
        m_basisIsProjected = true;
        m_notProjectedBasis = m_suppliedDensityMatrix.getBasis();
        m_basis = bas;
    }
    else
    {
        m_notProjectedBasis = bas;
        m_basis = bas;
    }
    init(m_suppliedDensityMatrix,m_suppliedOneElectronIntegrals);
}

template<typename SelfEnergyMatrixType>
void FockHamiltonian<SelfEnergyMatrixType>::constructHFTwoElectronIntegrals(const SparseTensor<4, complexType> &twoElectronIntegrals)
{   /*Construct the Spin basis two electron integral tensor adapted for a HF calculation
     * Assumed we have <qp|\sigma_I V |rs>.
     * Therefore (q & S) and (p & r) must be same spin respectively
     */
    DenseTensor<4, complexType> reformulatedTensor(m_basisSize);
    auto itEnd = twoElectronIntegrals.end();
    for (auto it = twoElectronIntegrals.begin(); *it < *itEnd; ++(*it))
    {
        SparseTensor<4, complexType>::indexArrayType idxs;
        complexType val = it->get(idxs);
        //val is always non-zero
        if (abs(val) < 1e-15)
            continue;
        for (int i = 0; i < 4; i++) // love me some hard-coded magic numbers
            idxs[i]*=2;// change to spin basis

        SparseTensor<4, complexType>::indexArrayType spinIdxs;
        for (int i = 0; i < 4; i++)
            spinIdxs[i] = 0;// I dont trust the ={}

        while (spinIdxs[3] != 2)
        {
            SparseTensor<4, complexType>::indexArrayType currIdxs;
            for (int i = 0; i < 4; i++)
            {
                currIdxs[i] = idxs[i] + spinIdxs[i];
            }
            //h_{pqrs} is spin allowed
            if (spinIdxs[1] == spinIdxs[3] && spinIdxs[0] == spinIdxs[2])
            {   //Hartree term is chosen by T_{pqrs}
                reformulatedTensor.coeffRef(currIdxs) += val;
                //Fock term is chosen by T_{qprs}
                std::swap(currIdxs[0],currIdxs[1]);
                reformulatedTensor.coeffRef(currIdxs) -= val;
                std::swap(currIdxs[0],currIdxs[1]);// not needed probably
            }
            //Increment Spin indexes
            for (int i = 0; i < 4; i++)
            {
                if (spinIdxs[i] == 1 && i < 3)
                    spinIdxs[i] = 0; // set zero and carry
                else
                {
                    spinIdxs[i] += 1; // increment
                    break;
                }
            }
        }
    }

    // if (!MetricIsDiagonal)
    // {
    //     DenseTensor<4, complexType> temp1(m_basisSize);
    //     SparseTensorView<ComplexMatrix,complexType,false> RaisingView(raisingMetric);

    //     Einsum<4,complexType,2,complexType,4,complexType>("ijrs,pi,pjrs")(&reformulatedTensor,&RaisingView,&temp1);
    //     Einsum<4,complexType,2,complexType,4,complexType>("pjrs,qj,pqrs")(&temp1,&RaisingView,&reformulatedTensor);
    // }
    //always unique as all repeated and dest indices are in first.
    reformulatedTensor.prepareForEinsum<4,2,2,EinsumTemplates::firstType>("pqrs,rp,qs",m_HFTwoElectronIntegrals,true);
    logger().log("JK Matrix Size",m_HFTwoElectronIntegrals.getOccupiedSize());
    logger().log("JK Matrix Possible Size",m_HFTwoElectronIntegrals.getSize()*m_HFTwoElectronIntegrals.getSize()*m_HFTwoElectronIntegrals.getSize()*m_HFTwoElectronIntegrals.getSize());
}

template class FockHamiltonian<SelfEnergyMatrix>;
