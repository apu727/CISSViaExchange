#include "sopgreensfunction.h"

#include "integrator.h"
#include <utility>

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

void SOPGreensFunction::buildCache() const
{
    std::lock_guard lock (m_cacheMutex);
    if (!m_GRIsCached) // check now that we're locked in case we waited at the mutex while some other thread did it.
    {
        releaseAssert(!m_isBareBones,"BuildCache for BareBones SOPGreensFunction");
        bool isProjectionBasis = BasisManager::isProjection(m_basis[0]) || BasisManager::isProjection(m_basis[1]);
        m_muMin=1e20;
        m_muMax=-1e20;

        m_muP.clear();
        m_GammaP.clear();


        bool foundAMu = false;
        for (auto SE : m_selfEnergies)
        {
            if (SE->hasSigmaK())
            {
                foundAMu = true;
                numType mu = SE->getChemicalPotential();
                m_muP.push_back(mu);

                m_muMax = std::max(mu,m_muMax);
                m_muMin = std::min(mu,m_muMin);
            }
        }
        if (!foundAMu)
            m_muMin = m_muMax = m_mu;

        m_rightEigenVectors.clear();
        m_InvRightEigenVectors.clear();
        m_EigenValues.clear();

        m_poleOrder.clear();
        m_poleOrderAdj.clear();
        m_rightEigenVectorsAdj.clear();
        m_InvRightEigenVectorsAdj.clear();
        m_EigenValuesAdj.clear();

        for (size_t regionIndex = 0; regionIndex < m_analyticRegions.size()-1; regionIndex++)
        {

            m_rightEigenVectors.emplace_back();
            m_InvRightEigenVectors.emplace_back();
            m_EigenValues.emplace_back();
            m_GammaP.emplace_back();
            m_poleOrder.emplace_back();
            //Allocate the space but dont use it. This is basically free as the default constructor does nothing
            m_rightEigenVectorsAdj.emplace_back();
            m_InvRightEigenVectorsAdj.emplace_back();
            m_EigenValuesAdj.emplace_back();


            numType regionEnergy = getRegionRepresentativeEnergy(regionIndex);

            auto& rightEigenVectorStore = m_rightEigenVectors.back();
            auto& rightInvEigenVectorStore = m_InvRightEigenVectors.back();
            auto& eigenValuesStore = m_EigenValues.back();

            auto& rightEigenVectorAdjStore = m_rightEigenVectorsAdj.back();
            auto& rightInvEigenVectorAdjStore = m_InvRightEigenVectorsAdj.back();
            auto& eigenValuesStoreAdj = m_EigenValuesAdj.back();

            ComplexMatrix EI = m_Ham;
            ComplexMatrix EIAdj = m_HamAdj;

            SelfEnergyMatrix temp;
            for (auto SE : m_selfEnergies)
            {
                SE->getSigmaR(regionEnergy,temp);
                EI += temp;
                if (isProjectionBasis)
                {
                    SE->getSigmaA(regionEnergy,temp);
                    EIAdj += temp;
                }
                if (SE->hasSigmaK())
                {
                    SE->getGamma(regionEnergy,temp);
                    m_GammaP.back().push_back(temp);
                }
            }
            if (m_enforcedSpinSym == GreensFunction::EnforcedSpinSymmetryType::RHF)
                EI.setSpinSym(enums::SpinSymmetry::RHF);
            logger().log("EI spin Sym is RHF",EI.getSpinSym() == enums::SpinSymmetry::RHF);
            m_spinSym = EI.getSpinSym();
            if (m_forceSelfAdjointSelfEnergy)
            {
                ComplexSelfAdjointMatrix EiSA = static_cast<ComplexSelfAdjointMatrix>(EI);
                ComplexSelfAdjointMatrix::EigenValueType eigenValues;
                ComplexSelfAdjointMatrix::EigenVectorType rightEigenVectors;
                releaseAssert(EiSA.getEigenValuesAndVectors(eigenValues,rightEigenVectors),"Diagonalisation failed");

                rightEigenVectorStore =  static_cast<EigenVectorMatrix>(rightEigenVectors);
                rightInvEigenVectorStore = static_cast<InvEigenVectorMatrix>(rightEigenVectors.inverse());
                eigenValuesStore = static_cast<EigenValueVector>(eigenValues);
                if (isProjectionBasis)
                {
                    rightEigenVectorAdjStore = rightInvEigenVectorStore; // not a typo
                    rightInvEigenVectorAdjStore = rightEigenVectorStore;
                }
            }
            else
            {
                EigenValueVector eigenValues;
                EigenVectorMatrix rightEigenVectors;
                bool s = EI.getEigenValuesAndVectors(eigenValues,rightEigenVectors,true);
                releaseAssert(s,"Diagonalisation failed");

                for (long i = 0; i < eigenValues.rows(); i++)
                {
                    auto &e = eigenValues(i);
                    if (abs(e.imag()) < 1e-13)
                    {
                        e = e.real();
                    }
                }

                rightEigenVectorStore =  rightEigenVectors;
                rightInvEigenVectorStore = rightEigenVectors.inverse();
                eigenValuesStore = eigenValues;
                m_poleOrder.back().setConstant(eigenValuesStore.rows(),1);

                if (isProjectionBasis)
                {
                    bool s = EIAdj.getEigenValuesAndVectors(eigenValues,rightEigenVectors,true);
                    releaseAssert(s,"Diagonalisation failed");

                    for (long i = 0; i < eigenValues.rows(); i++)
                    {
                        auto &e = eigenValues(i);
                        if (abs(e.imag()) < 1e-13)
                        {
                            e = e.real();
                        }
                    }

                    rightEigenVectorAdjStore =  rightEigenVectors.inverse();
                    rightInvEigenVectorAdjStore = rightEigenVectors;
                    eigenValuesStoreAdj = eigenValues;

                }
            }


        }
        m_poleOrderAdj = m_poleOrder;
        if (!isProjectionBasis)
        {
            m_rightEigenVectorsAdj.clear();
            m_InvRightEigenVectorsAdj.clear();
            m_EigenValuesAdj.clear();
            m_poleOrderAdj.clear();
        }

        m_GRIsCached = true;
        logger().log("built cache");
        for (auto& evR : m_EigenValues)
        {
            for (long i = 0; i < evR.rows(); i++)
            {
                auto& p = evR(i);
                if (p.imag() > 0)
                    logger().log("Pole has Imaginary eigenvalue greater than 0", p);
            }
        }

    }
}

void SOPGreensFunction::generateAdjoint() const
{
    assert(m_GRIsCached);
    if (m_EigenValuesAdj.size() > 0)
        return;
    m_rightEigenVectorsAdj.clear();
    m_InvRightEigenVectorsAdj.clear();
    m_EigenValuesAdj.clear();

    for (const auto& EV: m_rightEigenVectors)
        m_rightEigenVectorsAdj.emplace_back(EV.adjoint());

    for (const auto& EV: m_InvRightEigenVectors)
        m_InvRightEigenVectorsAdj.emplace_back(EV.adjoint());

    for (auto& EV: m_EigenValues)
        m_EigenValuesAdj.emplace_back(EV.conjugate());

    m_poleOrderAdj = m_poleOrder;
}

void SOPGreensFunction::IntegrateAnalyticEquilibriumGn(ComplexMatrix &dest, numType mu, bool MultiplyByE, bool makeHermitian, numType EStart, numType EEnd)
{
    bool isOrthogonalBasis = BasisManager::getInstance().isOrthogonal(m_basis[0]) && BasisManager::getInstance().isOrthogonal(m_basis[1]);
    bool hasAdjComputed = m_isBareBones? m_poleOrderAdj.size() > 0 : m_EigenValuesAdj.size() > 0;

    complexType (*PoleIntegral)(numType lower, numType upper, complexType polem, numType chemicalPotential, int poleOrder) = nullptr;
    if (!MultiplyByE)
    {
        //GN integrals
        PoleIntegral = [](numType lower, numType upper, complexType polem, numType chemicalPotential, int poleOrder)
        {
            numType ktOffset = 40*kb*T;
            /* bounds are in order lower bottom Top upper*/
            if (lower > chemicalPotential + ktOffset)
                return complexType(0);

            numType Top = 0;
            complexType bottom = 0;
            if (upper > chemicalPotential + ktOffset)
            {
                Top = chemicalPotential + ktOffset;
            }
            else
                //Below fermi level
                Top = upper;

            if (lower < chemicalPotential- ktOffset)
            {
                bottom = std::min(chemicalPotential - ktOffset,upper); // either top of integration of mu-31KT whichever is smaller
                // lower < bottom
            }
            else
            {
                bottom = lower;
            }
            bool poleHasZeroImaginaryPart = abs(polem.imag()) < 1e-12;

            int NumPoles = 1./(2*M_PI*kb*T)+1;
            if (bottom != Top && !poleHasZeroImaginaryPart)
            {//Offset into the complex plane to avoid poles
                bottom += NumPoles*T*(kb*2*M_PI*iu);
            }


            complexType ret = 0;
            if (lower != bottom)
            {
                assert(poleOrder > 0);
                if (poleOrder == 1)
                {
                    ret += iu*std::log(std::abs(bottom-polem));
                    ret -= (argd(bottom - polem));
                    if (lower != negInf)
                        ret -= iu*std::log(std::abs(lower-polem));
                    ret += (argd(lower - polem));
                }
                else if (poleOrder == 2)
                {
                    ret += -iu/(bottom-polem);
                    if (lower != negInf)
                        ret -= -iu/(lower-polem);
                }
                else
                {
                    ret += -iu/(pow(bottom-polem,poleOrder-1)*(poleOrder-1.));
                    if (lower != negInf)
                        ret -= -iu/(pow(lower-polem,poleOrder-1)*(poleOrder-1.));
                }
            }
            complexType FiniteTempBit = 0;
            if (bottom != Top)
            {
                if (!poleHasZeroImaginaryPart)
                {
                    // SpectralWeightedIntegral<false>(bottom.real(),Top,0,FiniteTempBit,std::vector<complexType>({polem}),[&](numType E, complexType& dest){dest = fermiFunction(E,chemicalPotential)*iu/(E-polem);}, [](){return complexType();},false);

                    releaseAssert(bottom.imag()>0,"bottom.imag()>0");
                    // GaussLegendreIntegral(bottom.real(),Top,200,FiniteTempBit,[&](numType E, complexType& dest){dest = fermiFunction(E,chemicalPotential)*iu/(E+bottom.imag()*iu-polem);}, [](){return complexType();},false);
                    AdaptiveGaussLegendreIntegralSingularities(bottom.real(),Top,FiniteTempBit,
                                                               [BY_VAL_CAPTURE(chemicalPotential),BY_VAL_CAPTURE(bottom),BY_VAL_CAPTURE(polem),BY_VAL_CAPTURE(poleOrder)](numType E, complexType& dest)
                                                               {dest = fermiFunction(E,chemicalPotential)*iu/pow(E+bottom.imag()*iu-polem,poleOrder);},
                                                               []()
                                                               {return complexType();},{chemicalPotential + M_PI*kb*T*iu},0,false);
                    if (bottom.imag() < polem.imag() && bottom.real() <polem.real() && Top > polem.real())
                    {
                        //Encircled this pole in an anticlockwise fashion. so add on the clockwise i.e. -2pi iu
                        releaseAssert(false,"Impossible");
                    }
                    if (bottom.imag() > 0)
                    {
                        //Encircled this pole in a clockwise fashion. so add on the anticlockwise i.e. 2pi iu.
                        const complexType pikbTiu = (M_PI*iu*kb)*T;
                        for (int i = 0; i < NumPoles; i++)
                        {
                            //First one is 1*pikbTiu, last one is (2*numPoles-1) < 2*numPoles
                            complexType E = chemicalPotential + numType(2*i+1)*pikbTiu;
                            //Fermi function has residue -kb T
                            FiniteTempBit += -2.*pikbTiu*iu/pow(E-polem,poleOrder);
                        }
                    }
                }
                else if (bottom.real() <= polem.real() && polem.real() < Top && poleOrder == 1) // polem.real in [bottom,top)
                    FiniteTempBit = fermiFunction(polem.real(),chemicalPotential)*M_PI;
            }

            return ret + FiniteTempBit;
        };
    }
    else
    {
        //EGN integrals
        PoleIntegral = [](numType lower, numType upper, complexType polem, numType chemicalPotential,int poleOrder)
        {
            numType ktOffset = 40*kb*T;
            /* bounds are in order lower--NonZero-- bottom--Fermi-- Top--Zero-- upper*/
            if (lower > chemicalPotential + ktOffset)
                return complexType(0);
            numType Top = 0;
            complexType bottom = 0;
            if (upper > chemicalPotential + ktOffset)
            {
                Top = chemicalPotential + ktOffset;
            }
            else
                //Below fermi level
                Top = upper;

            if (lower < chemicalPotential- ktOffset)
            {
                bottom = std::min(chemicalPotential - ktOffset,upper); // either top of integration of mu-31KT whichever is smaller
                // lower < bottom
            }
            else
            {
                bottom = lower;
            }

            bool poleHasZeroImaginaryPart = abs(polem.imag()) < 1e-12;
            int NumPoles = 1./(2*M_PI*kb*T)+1;
            if (bottom != Top && !poleHasZeroImaginaryPart)
            {//Offset into the complex plane to avoid poles
                bottom += NumPoles*T*(kb*2*M_PI*iu);
            }

            complexType ret = 0;
            if (lower != bottom)
            {
                assert(poleOrder > 0);
                if (poleOrder == 1)
                {
                    // Worry about the 1 in a projected basis? E/(E-a) = 1+a/(E-a)? Usually it cancels since it creates an identity matrix but in the projected basis this is not true? -It is . P R R^-1 P = P R^\dagger R^{-1\dagger} P \something like I
                    // ret += bottom;
                    // if (lower != negInf)
                    //   ret -= lower;
                    ret += iu*polem*std::log(std::abs(bottom-polem));
                    ret -= polem*argd(bottom - polem);
                    if (lower != negInf)
                        ret -= iu*polem*std::log(std::abs(lower-polem));

                    ret += polem*argd(lower - polem);
                }
                else if (poleOrder == 2)
                {
                    //Integrate \iu E/(E-polem)^2 = polem/(E-polem)^2 + 1/(E-polem)

                    ret += iu*std::log(std::abs(bottom-polem)) - argd(bottom - polem); //1/(E-polem)
                    // if (!poleHasZeroImaginaryPart)
                    ret += -iu*polem/(bottom-polem); //polem/(E-polem)^2
                    if (lower != negInf)
                    {
                        ret -= iu*std::log(std::abs(lower-polem)) - argd(lower - polem);
                        // if (!poleHasZeroImaginaryPart)
                        ret -= -iu*polem/(lower-polem);
                    }
                }
                else if (!poleHasZeroImaginaryPart)
                {
                    //According to mathematica: \int \iu x/(x-a)^n \td x= \iu \frac{a+(1-n)x}{(n-2)(n-1)(x-a)^{n-1})

                    ret += iu*(polem+(1.-poleOrder)*bottom)/((poleOrder-2.)*(poleOrder-1.)*pow(bottom-polem,poleOrder-1));
                    if (lower != negInf)
                        ret -= iu*(polem+(1.-poleOrder)*lower)/((poleOrder-2.)*(poleOrder-1.)*pow(lower-polem,poleOrder-1));
                }
            }

            //bottom to upper integral
            complexType FiniteTempBit = 0;
            if (bottom != Top)
            {
                if (!poleHasZeroImaginaryPart)
                {
                    // SpectralWeightedIntegral<false>(bottom,Top,0,FiniteTempBit,std::vector<complexType>({polem}),[&](numType E, complexType& dest){dest = fermiFunction(E,chemicalPotential)*iu*polem/(E-polem);}, [](){return complexType();},false);
                    releaseAssert(bottom.imag()>0,"bottom.imag()>0");
                    // GaussLegendreIntegral(bottom.real(),Top,200,FiniteTempBit,[&](numType E, complexType& dest){dest = fermiFunction(E,chemicalPotential)*iu*polem/(E+bottom.imag()*iu-polem);}, [](){return complexType();},false);
                    AdaptiveGaussLegendreIntegralSingularities(bottom.real(),Top,FiniteTempBit,
                                                               [BY_VAL_CAPTURE(chemicalPotential),BY_VAL_CAPTURE(bottom),BY_VAL_CAPTURE(polem),BY_VAL_CAPTURE(poleOrder)](numType E, complexType& dest)
                                                               {dest = fermiFunction(E,chemicalPotential)*iu*polem/pow((E+bottom.imag()*iu-polem),poleOrder);},
                                                               []()
                                                               {return complexType();},{chemicalPotential + M_PI*kb*T*iu},0,false);
                    if (bottom.imag() < polem.imag() && bottom.real() <polem.real() && Top > polem.real())
                    {
                        //Encircled this pole in an anticlockwise fashion. so add on the clockwise i.e. -2pi iu
                        releaseAssert(false,"Impossible");
                    }
                    if (bottom.imag() > 0)
                    {
                        //Encircled this pole in a clockwise fashion. so add on the anticlockwise i.e. 2pi iu.
                        const complexType pikbTiu = (M_PI*iu*kb)*T;
                        for (int i = 0; i < NumPoles; i++)
                        {
                            //First one is 1*pikbTiu, last one is (2*numPoles-1) < 2*numPoles
                            complexType E = chemicalPotential + numType(2*i+1)*pikbTiu;
                            //Fermi function has residue -kb T
                            FiniteTempBit += -2.*pikbTiu*iu*polem/pow(E-polem,poleOrder);
                        }
                    }
                }
                else if (bottom.real() <= polem.real() && polem.real() < Top && poleOrder <= 2)
                {
                    // if poleOrder > 2 then there is no residue. Note how:
                    //\frac{i x}{(x-a)^2} = \frac{i a}{(x-a)^2}+\frac{i}{x-a}
                    //\frac{i x}{x-a} = \frac{i a}{x-a}+i
                    //The constant cancels later, RR^{-1}
                    if (poleOrder == 1)
                        FiniteTempBit = fermiFunction(polem.real(),chemicalPotential)*M_PI*polem;
                    else if (poleOrder == 2)
                        FiniteTempBit = fermiFunction(polem.real(),chemicalPotential)*M_PI;
                    else
                    {
                        releaseAssert(false,"poleOrder <= 0");
                    }
                }
            }
            return ret+FiniteTempBit;
        };

    }

    auto start1 = std::chrono::high_resolution_clock::now();
    auto spatDim  = getSpatialDimension();
    dest.resize(spatDim.first,spatDim.second,m_basis,m_spinSym);
    dest.setZero();

    //This is an O(n^3) cost where n is matrix dimension. as it contains one matrix multiplication and n, scalar matrix multiplication

    std::mutex workerLock;
    std::vector<ComplexMatrix> results;
    //,
    auto work = [   CONST_REF_CAPTURE(PoleIntegral),
                 CONST_REF_CAPTURE(m_rightEigenVectors),
                 CONST_REF_CAPTURE(m_EigenValues),
                 CONST_REF_CAPTURE(m_poleOrder),
                 CONST_REF_CAPTURE(m_InvRightEigenVectors),
                 CONST_REF_CAPTURE(m_EigenValuesAdj),
                 CONST_REF_CAPTURE(m_InvRightEigenVectorsAdj),
                 CONST_REF_CAPTURE(m_rightEigenVectorsAdj),
                 CONST_REF_CAPTURE(m_poleOrderAdj),
                 CONST_REF_CAPTURE(m_basis),
                 BY_VAL_CAPTURE(m_spinSym),
                 CONST_REF_CAPTURE(m_analyticRegions),
                 CONST_REF_CAPTURE(m_isBareBones),
                 CONST_REF_CAPTURE(m_Residues),
                 CONST_REF_CAPTURE(m_ResiduesAdj),
                 BY_REF_CAPTURE(workerLock),
                 BY_REF_CAPTURE(results),
                 BY_VAL_CAPTURE(hasAdjComputed),
                 BY_VAL_CAPTURE(mu),
                 BY_VAL_CAPTURE(EStart),
                 BY_VAL_CAPTURE(EEnd),
                 spatdim = getSpatialDimension()
                ]
        (size_t startm, size_t endM, size_t storeIdx)
    {
        ComplexMatrix workerFinalDest;
        workerFinalDest.resize(spatdim.first,spatdim.second,m_basis,m_spinSym);
        workerFinalDest.setZero();

        for (size_t regionIndex = 0; regionIndex < m_analyticRegions.size()-1; regionIndex++)
        {
            numType regionStartEnergy = m_analyticRegions[regionIndex];
            numType regionEndEnergy = m_analyticRegions[regionIndex+1];
            regionStartEnergy = std::max(EStart,regionStartEnergy);
            regionEndEnergy = std::min(EEnd,regionEndEnergy);
            if (regionEndEnergy <= regionStartEnergy)
                continue;


            // if (m_forceSelfAdjointSelfEnergy && isOrthogonalBasis)
            if (!m_isBareBones)
            {
                Eigen::MatrixXcd workerDest;
                workerDest.resize(spatdim.first,spatdim.second);
                workerDest.setZero();

                for (size_t m = startm; m < endM;m++)
                {
                    complexType poleIntegralVal = PoleIntegral(regionStartEnergy,regionEndEnergy,m_EigenValues[regionIndex](m),mu,m_poleOrder[regionIndex][m]);
                    workerDest += m_rightEigenVectors[regionIndex].col(m) * m_InvRightEigenVectors[regionIndex].row(m) * poleIntegralVal;
                    if (hasAdjComputed)
                    {
                        //unfortunately bc the adj is another diagonalisation potentially, we can't guarantee the order is the sa
                        poleIntegralVal = std::conj(PoleIntegral(regionStartEnergy,regionEndEnergy,std::conj(m_EigenValuesAdj[regionIndex](m)),mu,m_poleOrderAdj[regionIndex][m]));
                        workerDest += m_InvRightEigenVectorsAdj[regionIndex].col(m) * m_rightEigenVectorsAdj[regionIndex].row(m) * poleIntegralVal;
                    }
                }

                // workerFinalDest += ComplexMatrix(std::move(workerDest.get()),m_basis,m_spinSym);
                workerFinalDest += ComplexMatrix(workerDest,m_basis,m_spinSym);
            }
            else
            {
                for (size_t m = startm; m < endM;m++)
                {
                    complexType poleIntegralVal = PoleIntegral(regionStartEnergy,regionEndEnergy,m_EigenValues[regionIndex](m),mu,m_poleOrder[regionIndex][m]);
                    workerFinalDest += m_Residues[regionIndex][m] * poleIntegralVal;
                    if (hasAdjComputed)
                    {
                        //unfortunately bc the adj is another diagonalisation potentially, we can't guarantee the order is the sa
                        poleIntegralVal = std::conj(PoleIntegral(regionStartEnergy,regionEndEnergy,std::conj(m_EigenValuesAdj[regionIndex](m)),mu,m_poleOrderAdj[regionIndex][m]));
                        workerFinalDest += m_ResiduesAdj[regionIndex][m] * poleIntegralVal;
                    }
                }
            }
            // else
            // {
            //     EigenVectorMatrix workerDest;
            //     workerDest.resize(dim,dim,m_basis,m_spinSym);
            //     workerDest.setZero();
            //     for (size_t m = startm; m < endM;m++)
            //     {

                //         workerDest.col(m) = m_rightEigenVectors[regionIndex].col(m)*PoleIntegral(regionStartEnergy,regionEndEnergy,m_EigenValues[regionIndex](m),mu,m_poleOrder[regionIndex][m]);
                //     }
                //     PEAxB(workerFinalDest,workerDest,m_InvRightEigenVectors[regionIndex]);
                // }
        }

        std::unique_lock locker(workerLock);
        results[storeIdx] = workerFinalDest;
    };

    const size_t eigdim = getEigenValueDimension();
    const int stepSize = std::max(eigdim/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    size_t numWorkers  = ceil((numType)eigdim/stepSize);
    results.resize(numWorkers);
    size_t workerIdx = 0;
    Eigen::setNbThreads(1);
    for (size_t m = 0; m < eigdim; m+= stepSize)
    {
        auto endIndex = std::min(m+stepSize,eigdim);
        futures.push_back(pool.queueWork([=,&work](){work(m,endIndex,workerIdx);}));
        workerIdx++;
    }
    releaseAssert(numWorkers == workerIdx,"Invalid calculation of number of workers");
    for (auto& fut : futures)
        fut.wait();
    Eigen::setNbThreads(0);
    workerLock.lock(); //paranoia
    for (auto& r : results)
        dest += r;

    workerLock.unlock();
    //Need to take the adjoint and add it on, This is already done if we're in a projection basis
    if (makeHermitian && !hasAdjComputed)
    {
        dest += dest.adjoint();
        dest.makeSelfAdjoint();
        // if (!MultiplyByE && false)
        // {
        //     ComplexHermitianMatrix hermDensity;
        //     checkBasisChangeAssert(static_cast<ComplexSelfAdjointMatrix>(dest).toHermitianBasis(hermDensity),dest.getBasis(),"Dest.HermBasis");
        //     ComplexHermitianMatrix::EigenValueType densityEigVals;
        //     ComplexHermitianMatrix::EigenVectorType densityEigVecs;
        //     releaseAssert(hermDensity.getEigenValuesAndVectors(densityEigVals,densityEigVecs),"FailedToGetdensityeigenvalues");
        //     bool fixedOne = false;
        //     for (long i = 0; i < densityEigVals.rows(); i++)
        //     {
        //         // logger().log("densityEigVal",densityEigVals(i)/(2*M_PI));
        //         if (densityEigVals(i) < 0 || densityEigVals(i) > 2*M_PI)
        //             fixedOne = true;
        //         densityEigVals(i) = std::max(densityEigVals(i),0.);
        //         densityEigVals(i) = std::min(densityEigVals(i),2*M_PI);
        //     }
        //     // logger().log("densityEigVals.rows()",densityEigVals.rows());
        //     if (fixedOne)
        //     {
        //         logger().log("Fixed one");
        //         dest = densityEigVecs * densityEigVals.asDiagonal(densityEigVecs.getBasis()) * densityEigVecs.inverse();
        //         dest.toBasisC(dest,m_basis);
        //     }
        // }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start1);
    if (!MultiplyByE)
        logger().log("Gn Timings us",std::vector<long>({duration1.count()}));
    else
        logger().log("EGn Timings us",std::vector<long>({duration1.count()}));
}

void SOPGreensFunction::IntegrateAnalyticNonEquilibriumGn(ComplexMatrix &dest, bool MultiplyByE, numType EStart, numType EEnd)
{
    releaseAssert(m_muP.size() == m_GammaP.back().size(),"m_muP.size() == m_GammaP.size()");
    releaseAssert(!m_isBareBones,"!m_isBareBones");
    bool isProjectionBasis = BasisManager::getInstance().isProjection(m_basis[0]) && BasisManager::getInstance().isProjection(m_basis[1]);
    if (!isProjectionBasis)
        generateAdjoint(); // Safe to call even if it is already generated

    //These are the same as the equilibrium integrals. TODO refactor so its not duplicated
    //\int_{-\infinity}^{\mu} (E?) 1/(E-polem)^poleOrder \td E
    // Since we perform the partial fraction decompose before calling this function we dont need the EGn integrals.
    complexType  (*PoleIntegral)(numType lower, numType upper, complexType polem, numType chemicalPotential, int poleOrder) =
        [](numType lower, numType upper, complexType polem, numType chemicalPotential, int poleOrder)
    {
        numType ktOffset = 40*kb*T;
        /* bounds are in order lower bottom Top upper*/
        if (lower > chemicalPotential + ktOffset)
            return complexType(0);

        numType Top = 0;
        complexType bottom = 0;
        if (upper > chemicalPotential + ktOffset)
        {
            Top = chemicalPotential + ktOffset;
        }
        else
            //Below fermi level
            Top = upper;

        if (lower < chemicalPotential- ktOffset)
        {
            bottom = std::min(chemicalPotential - ktOffset,upper); // either top of integration of mu-31KT whichever is smaller
            // lower < bottom
        }
        else
        {
            bottom = lower;
        }
        bool poleHasZeroImaginaryPart = abs(polem.imag()) < 1e-12;

        int NumPoles = 1./(2*M_PI*kb*T)+1;
        if (bottom != Top && !poleHasZeroImaginaryPart)
        {//Offset into the complex plane to avoid poles
            bottom += NumPoles*T*(kb*2*M_PI*iu);
        }


        complexType ret = 0;
        if (lower != bottom)
        {
            assert(poleOrder > 0);
            if (poleOrder == 1)
            {
                ret += iu*std::log(std::abs(bottom-polem));
                ret -= (argd(bottom - polem));
                if (lower != negInf)
                    ret -= iu*std::log(std::abs(lower-polem));
                ret += (argd(lower - polem));
            }
            else if (poleOrder == 2)
            {
                ret += -iu/(bottom-polem);
                if (lower != negInf)
                    ret -= -iu/(lower-polem);
            }
            else
            {
                ret += -iu/(pow(bottom-polem,poleOrder-1)*(poleOrder-1.));
                if (lower != negInf)
                    ret -= -iu/(pow(lower-polem,poleOrder-1)*(poleOrder-1.));
            }
        }
        complexType FiniteTempBit = 0;
        if (bottom != Top)
        {
            if (!poleHasZeroImaginaryPart)
            {
                // SpectralWeightedIntegral<false>(bottom.real(),Top,0,FiniteTempBit,std::vector<complexType>({polem}),[&](numType E, complexType& dest){dest = fermiFunction(E,chemicalPotential)*iu/(E-polem);}, [](){return complexType();},false);

                releaseAssert(bottom.imag()>0,"bottom.imag()>0");
                // GaussLegendreIntegral(bottom.real(),Top,200,FiniteTempBit,[&](numType E, complexType& dest){dest = fermiFunction(E,chemicalPotential)*iu/(E+bottom.imag()*iu-polem);}, [](){return complexType();},false);
                AdaptiveGaussLegendreIntegralSingularities(bottom.real(),Top,FiniteTempBit,
                                                           [BY_VAL_CAPTURE(chemicalPotential),BY_VAL_CAPTURE(bottom),BY_VAL_CAPTURE(polem),BY_VAL_CAPTURE(poleOrder)](numType E, complexType& dest)
                                                           {dest = fermiFunction(E,chemicalPotential)*iu/pow(E+bottom.imag()*iu-polem,poleOrder);},
                                                           []()
                                                           {return complexType();},{chemicalPotential + M_PI*kb*T*iu},0,false);
                if (bottom.imag() < polem.imag() && bottom.real() <polem.real() && Top > polem.real())
                {
                    //Encircled this pole in an anticlockwise fashion. so add on the clockwise i.e. -2pi iu
                    releaseAssert(false,"Impossible");
                }
                if (bottom.imag() > 0)
                {
                    //Encircled this pole in a clockwise fashion. so add on the anticlockwise i.e. 2pi iu.
                    const complexType pikbTiu = (M_PI*iu*kb)*T;
                    for (int i = 0; i < NumPoles; i++)
                    {
                        //First one is 1*pikbTiu, last one is (2*numPoles-1) < 2*numPoles
                        complexType E = chemicalPotential + numType(2*i+1)*pikbTiu;
                        //Fermi function has residue -kb T
                        FiniteTempBit += -2.*pikbTiu*iu/pow(E-polem,poleOrder);
                    }
                }
            }
            else if (bottom.real() <= polem.real() && polem.real() < Top && poleOrder == 1) // polem.real in [bottom,top)
                FiniteTempBit = fermiFunction(polem.real(),chemicalPotential)*M_PI;
        }

        return ret + FiniteTempBit;
    };





    auto start1 = std::chrono::high_resolution_clock::now();
    const auto spatDim = getSpatialDimension();
    dest.resize(spatDim.first,spatDim.second,m_basis,m_spinSym);
    dest.setZero();
    //This is an O(n^3) cost where n is matrix dimension.


    //for work2
    EigenValueVector IntGrEig;
    EigenValueVector IntGaEig;
    IntGrEig.resize(getEigenValueDimension(),1,"",m_EigenValues[0].getSpinSym());
    IntGaEig.resize(getEigenValueDimension(),1,"",m_EigenValues[0].getSpinSym());

    auto work2 = [CONST_REF_CAPTURE(m_analyticRegions),
                 CONST_REF_CAPTURE(m_EigenValuesAdj),
                 CONST_REF_CAPTURE(m_EigenValues),
                 CONST_REF_CAPTURE(PoleIntegral),
                 CONST_REF_CAPTURE(m_muP),
                 CONST_REF_CAPTURE(m_GammaP),
                 CONST_REF_CAPTURE(m_basis),
                 CONST_REF_CAPTURE(m_spinSym),
                 BY_VAL_CAPTURE(MultiplyByE),
                 BY_VAL_CAPTURE(EStart),
                 BY_VAL_CAPTURE(EEnd),
                 BY_REF_CAPTURE(IntGrEig),
                 BY_REF_CAPTURE(IntGaEig)

    ](size_t startm, size_t endM, size_t p, size_t regionIndex)
    {
        //Evaluate \int 1/(E-Em) \td E for all Em for a given p and region index
        numType regionStartEnergy = m_analyticRegions[regionIndex];
        numType regionEndEnergy = m_analyticRegions[regionIndex+1];
        regionStartEnergy = std::max(EStart,regionStartEnergy);
        regionEndEnergy = std::min(EEnd,regionEndEnergy);
        if (regionEndEnergy <= regionStartEnergy)
        {
            for (size_t m = startm; m < endM; m++)
            {
                IntGrEig(m) = 0.;
                IntGaEig(m) = 0.;
            }
            return;
        }
        for (size_t m = startm; m < endM; m++)
        {
            complexType polem = m_EigenValues[regionIndex](m);
            IntGrEig(m) = -iu*PoleIntegral(regionStartEnergy,regionEndEnergy,polem,m_muP[p],1);
            complexType polemPrime = m_EigenValuesAdj[regionIndex](m);
            if (abs(polemPrime - std::conj(polem)) < 1e-14)
                IntGaEig(m) = std::conj(IntGrEig(m));
            else
                IntGaEig(m) = std::conj(-iu*PoleIntegral(regionStartEnergy,regionEndEnergy,std::conj(polemPrime),m_muP[p],1));
        }

    };

    const size_t eigdim = getEigenValueDimension();
    const int stepSize = std::max(eigdim/NUM_CORES,1ul);
    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    MatrixBufferObject<Eigen::MatrixXcd> LeftBit,Mat2,RightBit,MiddleBit,Mat5,EmMinusEmPrime;
    for (size_t regionIndex = 0; regionIndex < m_analyticRegions.size()-1; regionIndex++)
    {
        EmMinusEmPrime->resize(eigdim,eigdim);
        for (size_t m = 0; m < eigdim; m++)
            EmMinusEmPrime->row(m) = m_EigenValues[regionIndex](m) - std::as_const(m_EigenValuesAdj[regionIndex]).getEigenOptimisation().array();

        *EmMinusEmPrime = EmMinusEmPrime->cwiseInverse();

        for (size_t p = 0; p < m_muP.size(); p++)
        {
            for (size_t m = 0; m < eigdim; m+= stepSize)
            {
                auto endIndex = std::min(m+stepSize,eigdim);
                futures.push_back(pool.queueWork([=,&work2](){work2(m,endIndex,p,regionIndex);}));
            }
            for (auto& f : futures)
                f.wait();
            futures.clear();
            //R LambdaInt R^{-1} Gamma_p R^{-1\dagger}1/(em-emprime) R^\dagger

            LeftBit->noalias() = std::as_const(m_rightEigenVectors[regionIndex]).getEigenOptimisation() * IntGrEig.getEigenOptimisation().asDiagonal(); //R LambdaInt

            Mat2->noalias() = std::as_const(m_InvRightEigenVectors[regionIndex]).getEigenOptimisation() * std::as_const(m_GammaP[regionIndex][p]).getEigenOptimisation();
            MiddleBit->noalias() = *Mat2 * std::as_const(m_InvRightEigenVectorsAdj[regionIndex]).getEigenOptimisation(); //R^{-1} Gamma R^{-1\dagger}
            *Mat2 = MiddleBit->cwiseProduct(*EmMinusEmPrime);//R^{-1} Gamma R^{-1\dagger}1/(em-emprime)
            if (MultiplyByE)
            {
                *Mat2 = std::as_const(m_EigenValues[regionIndex]).getEigenOptimisation().asDiagonal()* *Mat2;
            }

            Mat5->noalias() = *Mat2 * std::as_const(m_rightEigenVectorsAdj[regionIndex]).getEigenOptimisation(); //R^{-1} Gamma R^{-1\dagger}1/(em-emprime) R^\dagger
            dest.noalias() += *LeftBit * *Mat5;//R LambdaInt R^{-1} Gamma R^{-1\dagger}1/(em-emprime) R^\dagger

            //R 1/(emPrime-em) R^{-1} Gamma_p R^{-1\dagger} LambdaInt R^\dagger
            LeftBit->noalias() = IntGaEig.getEigenOptimisation().asDiagonal() * std::as_const(m_rightEigenVectorsAdj[regionIndex]).getEigenOptimisation(); //LambdaInt R^\dagger

            //Reuse middle bit
            *Mat2 = -1* MiddleBit->cwiseProduct( *EmMinusEmPrime);//1/(emPrime-em) R_{m.}^{-1} Gamma_p R^{-1\dagger}_{.mprime}
            if (MultiplyByE)
            {
                *Mat2 *= std::as_const(m_EigenValuesAdj[regionIndex]).getEigenOptimisation().asDiagonal();
            }

            Mat5->noalias() = *Mat2 * *LeftBit; //1/(emPrime-em) R_{m.}^{-1} Gamma_p R^{-1\dagger}_{.mprime} LambdaInt R^\dagger
            dest.noalias() += std::as_const(m_rightEigenVectors[regionIndex]).getEigenOptimisation() * *Mat5;//R 1/(emPrime-em) R^{-1} Gamma_p R^{-1\dagger} LambdaInt R^\dagger
        }



    }
    if (!isProjectionBasis)
        dest.makeSelfAdjoint();
    logger().log("(E)GN spin Sym is RHF",dest.getSpinSym() == enums::SpinSymmetry::RHF);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start1);
    if (!MultiplyByE)
        logger().log("Gn Timings us",std::vector<long>({duration1.count()}));
    else
        logger().log("EGn Timings us",std::vector<long>({duration1.count()}));
}

void SOPGreensFunction::IntegrateAnalyticNonEquilibriumGnOld(ComplexMatrix &dest, bool MultiplyByE)
{
    releaseAssert(m_muP.size() == m_GammaP.back().size(),"m_muP.size() == m_GammaP.size()");
    releaseAssert(!m_isBareBones,"!m_isBareBones");
    bool isProjectionBasis = BasisManager::getInstance().isProjection(m_basis[0]) && BasisManager::getInstance().isProjection(m_basis[1]);

    complexType (*PoleIntegral)(numType lower, numType upper,complexType polem, complexType polemPrime, numType chemicalPotential) = nullptr;
    if (!MultiplyByE)
    {

        //GN integrals
        PoleIntegral = [](numType lower, numType upper,complexType polem, complexType polemPrime, numType chemicalPotential)
        {
            numType ktOffset = 40*kb*T;
            /* bounds are in order lower bottom Top upper*/
            if (lower > chemicalPotential + ktOffset)
                return complexType(0);

            numType Top = 0;
            complexType bottom = 0;
            if (upper > chemicalPotential + ktOffset)
            {
                Top = chemicalPotential + ktOffset;
            }
            else
                //Below fermi level
                Top = upper;

            if (lower < chemicalPotential- ktOffset)
            {
                bottom = std::min(chemicalPotential - ktOffset,upper); // either top of integration of mu-31KT whichever is smaller
                // lower < bottom
            }
            else
            {
                bottom = lower;
            }

            int NumPoles = 1./(2*M_PI*kb*T)+1;
            if (bottom != Top && T > 10)
            {//Offset into the complex plane to avoid poles if needed
                if (abs(polem - (bottom+Top/2.)) < 0.5)
                {
                    bottom -= NumPoles*T*(kb*2*M_PI*iu);
                }
                else if (abs(polemPrime - (bottom+Top/2.)) < 0.5)
                {
                    bottom += NumPoles*T*(kb*2*M_PI*iu);
                }
                else
                    ;//Stay on the real axis
            }

            complexType ret = 0;
            //This contour never accumulates poles as it has a branch cut from the pole up/downwards
            if (lower != bottom)
            {
                ret += std::log(std::abs(bottom-polem)/std::abs(bottom-polemPrime)); //TODO handle if this becomes infinity
                ret += iu*(argd(bottom - polem)-argu(bottom - polemPrime));

                if (lower != negInf)
                    ret -= std::log(std::abs(lower-polem)/std::abs(lower-polemPrime));
                ret -= iu*(argd(lower - polem)-argu(lower - polemPrime));

                ret *= 1./(polem - polemPrime);
            }
            complexType FiniteTempBit = 0;
            complexType FiniteTempBitTest = 0;
            if (bottom != Top && T > 10)
            {
                if (bottom.imag() == 0.)
                {
                    // SpectralWeightedIntegral<false>(bottom.real(),Top,10000000,FiniteTempBitTest,std::vector<complexType>({polem,std::conj(polemPrime)}),[&](numType E, complexType& dest){dest = fermiFunction(E,chemicalPotential)/((E-polem)*(E-polemPrime));}, [](){return complexType();},false);
                    //FermiWeightedIntegral<true>(bottom.real(),Top,1000,FiniteTempBit,chemicalPotential,[&](numType E, complexType& dest){dest = 1./((E-polem)*(E-polemPrime));},[](){return complexType();},false);
                    // GaussLegendreIntegral(bottom.real(),Top,200,FiniteTempBit,[&](numType E, complexType& dest){dest = fermiFunction(E,chemicalPotential)/((E-polem)*(E-polemPrime));}, [](){return complexType();},false);
                    AdaptiveGaussLegendreIntegralSingularities(bottom.real(),Top,FiniteTempBit,
                                                               [BY_VAL_CAPTURE(chemicalPotential),BY_VAL_CAPTURE(bottom),BY_VAL_CAPTURE(polem),BY_VAL_CAPTURE(polemPrime)](numType E, complexType& dest)
                                                               {dest = fermiFunction(E,chemicalPotential)/((E-polem)*(E-polemPrime));},
                                                               []()
                                                               {return complexType();},{chemicalPotential + M_PI*kb*T*iu},0,false);
                    // logger().log("integral error",FiniteTempBit-FiniteTempBitTest);
                }
                else
                {
                    // GaussLegendreIntegral(bottom.real(),Top,200,FiniteTempBit,[&](numType E, complexType& dest){dest = fermiFunction(E,chemicalPotential)/((E+bottom.imag()*iu-polem)*(E+bottom.imag()*iu-polemPrime));}, [](){return complexType();},false);
                    AdaptiveGaussLegendreIntegralSingularities(bottom.real(),Top,FiniteTempBit,
                                                               [BY_VAL_CAPTURE(chemicalPotential),BY_VAL_CAPTURE(bottom),BY_VAL_CAPTURE(polem),BY_VAL_CAPTURE(polemPrime)](numType E, complexType& dest)
                                                               {dest = fermiFunction(E,chemicalPotential)/((E+bottom.imag()*iu-polem)*(E+bottom.imag()*iu-polemPrime));},
                                                               []()
                                                               {return complexType();},{chemicalPotential + M_PI*kb*T*iu},0,false);
                    // FermiWeightedIntegral<true>(bottom.real(),Top,200,FiniteTempBit,chemicalPotential,     [&](numType E, complexType& dest){dest = 1./((E+bottom.imag()*iu-polem)*(E+bottom.imag()*iu-polemPrime));}, [](){return complexType();},false);
                    if (bottom.imag() > polemPrime.imag() && bottom.real() <polemPrime.real() && Top > polemPrime.real())
                    {
                        //Encircled this pole in a clockwise fashion. so add on the anticlockwise i.e. 2pi iu
                        FiniteTempBit += 2*M_PI*iu*fermiFunction(polemPrime,chemicalPotential)/((polemPrime-polem));
                    }
                    else if (bottom.imag() < polem.imag() && bottom.real() <polem.real() && Top > polem.real())
                    {
                        //Encircled this pole in an anticlockwise fashion. so add on the clockwise i.e. -2pi iu
                        FiniteTempBit -= 2*M_PI*iu*fermiFunction(polem,chemicalPotential)/((polem-polemPrime));
                    }
                    if (bottom.imag() > 0)
                    {
                        //Encircled this pole in a clockwise fashion. so add on the anticlockwise i.e. 2pi iu.
                        const complexType pikbTiu = (M_PI*iu*kb)*T;
                        for (int i = 0; i < NumPoles; i++)
                        {
                            //First one is 1*pikbTiu, last one is (2*numPoles-1) < 2*numPoles
                            complexType E = chemicalPotential + numType(2*i+1)*pikbTiu;
                            //Fermi function has residue -kb T
                            FiniteTempBit += -2.*pikbTiu/((E-polem)*(E-polemPrime));
                        }
                    }
                    else if (bottom.imag() < 0)
                    {
                        //Encircled this pole in a anticlockwise fashion. so add on the clockwise i.e. -2pi iu.
                        const complexType pikbTiu = (M_PI*iu*kb)*T;
                        for (int i = 0; i > -NumPoles; i--)
                        {
                            //First one is -1*pikbTiu, last one is (-2*numPoles+1) > -2*numPoles
                            complexType E = chemicalPotential + numType(2*i-1)*pikbTiu;
                            //Fermi function has residue -kb T
                            FiniteTempBit -= -2.*pikbTiu/((E-polem)*(E-polemPrime));
                        }
                    }

                }
            }
            return ret + FiniteTempBit;
        };

    }
    else
    {//EGN integrals

        PoleIntegral = [](numType lower, numType upper, complexType polem, complexType polemPrime, numType chemicalPotential)
        {
#warning update to use new integrators
            if (lower > chemicalPotential + 31*kb*T)
                return complexType(0);
            numType Top = 0;
            numType bottom = 0;
            if (upper > chemicalPotential + 31*kb*T)
            {
                Top = chemicalPotential + 31*kb*T;
            }
            else
                //Below fermi level
                Top = upper;

            if (lower < chemicalPotential- 31*kb*T)
            {
                bottom = std::min(chemicalPotential - 31*kb*T,upper); // either top of integration of mu-31KT whichever is smaller
                // lower < bottom
            }
            else
            {
                bottom = lower;
            }

            complexType ret = 0;
            if (lower < bottom)
            {
                ret += polem*std::log(std::abs(bottom-polem))-polemPrime*std::log(std::abs(bottom-polemPrime));
                ret += iu*(polem*argd(bottom - polem)-polemPrime*argu(bottom - polemPrime));
                if (lower != negInf)
                    ret -= polem*std::log(std::abs(lower-polem))-polemPrime*std::log(std::abs(lower-polemPrime));


                if (abs(argd(lower - polem) - M_PI) < 1e-10 && abs(argu(lower - polemPrime) - M_PI) < 1e-10)
                {// There may be numerical problems with the other ordering
                    ret *= 1./(polem - polemPrime);
                    ret -= iu*M_PI;

                }
                else
                {
                    ret -= iu*(polem*argd(lower - polem)-polemPrime*argu(lower - polemPrime));
                    ret *= 1./(polem - polemPrime);
                }

            }

            complexType FiniteTempBit = 0;
            if (bottom < Top)
            {
                SpectralWeightedIntegral<false>(bottom,Top,0,FiniteTempBit,std::vector<complexType>({polem,std::conj(polemPrime)}),
                                                [BY_VAL_CAPTURE(chemicalPotential),BY_VAL_CAPTURE(polem),BY_VAL_CAPTURE(polemPrime)](numType E, complexType& dest)
                                                {dest = fermiFunction(E,chemicalPotential)*E/((E-polem)*(E-polemPrime));},
                                                []()
                                                {return complexType();},false);
            }
            return ret+FiniteTempBit;
        };

    }

    auto start1 = std::chrono::high_resolution_clock::now();
    const auto spatDim = getSpatialDimension();
    dest.resize(spatDim.first,spatDim.second,m_basis,m_spinSym);
    dest.setZero();
    //This is an O(n^4) cost where n is matrix dimension. This is the most expensive part

    std::mutex workerLock;
    std::vector<ComplexMatrix> results;

    auto work = [CONST_REF_CAPTURE(m_analyticRegions),
                 CONST_REF_CAPTURE(m_InvRightEigenVectors),
                 CONST_REF_CAPTURE(m_InvRightEigenVectorsAdj),
                 CONST_REF_CAPTURE(m_EigenValuesAdj),
                 CONST_REF_CAPTURE(m_EigenValues),
                 CONST_REF_CAPTURE(m_rightEigenVectorsAdj),
                 CONST_REF_CAPTURE(PoleIntegral),
                 CONST_REF_CAPTURE(m_rightEigenVectors),
                 CONST_REF_CAPTURE(m_muP),
                 CONST_REF_CAPTURE(m_GammaP),
                 CONST_REF_CAPTURE(m_basis),
                 CONST_REF_CAPTURE(m_spinSym),
                 BY_VAL_CAPTURE(MultiplyByE),
                 BY_VAL_CAPTURE(isProjectionBasis),
                 BY_REF_CAPTURE(workerLock),
                 BY_REF_CAPTURE(results),
                 spatDim = getSpatialDimension(),
                 eigDim = getEigenValueDimension()
    ](size_t startm, size_t endM, size_t p, size_t storeIdx)
    {
        ComplexMatrix workerFinalDest;
        workerFinalDest.resize(spatDim.first,spatDim.second,m_basis,m_spinSym);
        workerFinalDest.setZero();
        EigenVectorMatrix workerTemp;
        workerTemp.resize(spatDim.first,spatDim.second,m_basis,m_spinSym);


        for (size_t regionIndex = 0; regionIndex < m_analyticRegions.size()-1; regionIndex++)
        {
            //Evaluate \int GR \Gamma GA \td E = R \Lambda R^{-1} \Gamma R^{-1\dagger} \Lambda^\dagger R^\dagger
            numType regionStartEnergy = m_analyticRegions[regionIndex];
            numType regionEndEnergy = m_analyticRegions[regionIndex+1];

            ComplexMatrix workerDest;
            workerDest.resize(spatDim.first,spatDim.second,m_basis,m_spinSym);
            workerDest.setZero();

            InvEigenVectorMatrix precompute2;
            precompute2 = m_InvRightEigenVectors[regionIndex] * m_GammaP[regionIndex][p];
            //Break out of the compile time restriction for the sake of optimisation
            ComplexMatrix AmAdjoint;
            ComplexMatrix Am;
            Am.resize(spatDim.first,spatDim.second,m_basis,m_spinSym);
            Eigen::MatrixXcd AmEV(spatDim.first,spatDim.second);
            for (size_t mPrime = startm; mPrime < endM; mPrime++)
            {
                auto __attribute__ ((unused))start1 = std::chrono::high_resolution_clock::now();
                Matrix<complexType,-1,-1,enums::MatrixProperties::None,enums::IndexType::eigenValue,enums::IndexType::direct> precompute;
                if (!isProjectionBasis)
                {
                    Am.noalias() = m_rightEigenVectors[regionIndex].col(mPrime) * m_InvRightEigenVectors[regionIndex].row(mPrime);
                    AmAdjoint = Am.adjoint();
                    precompute = precompute2 * AmAdjoint;
#warning todo optimise to O(n2) same as the precomputed adjoint method
                }
                else
                {
                    // AmAdjoint.noalias() = m_InvRightEigenVectorsAdj[regionIndex].col(mPrime) * m_rightEigenVectorsAdj[regionIndex].row(mPrime);
                    //precompute = precompute2 * AmAdjoint;
                    precompute.noalias() = (precompute2.getEigenOptimisation() * m_InvRightEigenVectorsAdj[regionIndex].col(mPrime)) * m_rightEigenVectorsAdj[regionIndex].row(mPrime); // O(n^2)
                }



                workerTemp.setZero();
                auto __attribute__ ((unused))start2 = std::chrono::high_resolution_clock::now();

                for (size_t m = 0; m < eigDim; m++)
                {
                    if (abs(m_EigenValues[regionIndex](m).imag()) < 1e-8)
                        logger().log("Pole is small and non-equilibrium requested", m_EigenValues[regionIndex](m));

                    complexType adjEigenValue = isProjectionBasis ? m_EigenValuesAdj[regionIndex](mPrime) : std::conj(m_EigenValues[regionIndex](mPrime));
                    if (abs(adjEigenValue.imag()) < 1e-8)
                        logger().log("Pole is small and non-equilibrium requested", adjEigenValue);

                    workerTemp.col(m).noalias() = m_rightEigenVectors[regionIndex].col(m) *
                                                  PoleIntegral(regionStartEnergy,regionEndEnergy,m_EigenValues[regionIndex](m),adjEigenValue, m_muP[p]);
                }
                auto __attribute__ ((unused))start3 = std::chrono::high_resolution_clock::now();
                // workerDest += workerTemp * precompute;
                PEAxB(workerDest,workerTemp,precompute);
                auto __attribute__ ((unused))start4 = std::chrono::high_resolution_clock::now();
                auto __attribute__ ((unused))duration1 = std::chrono::duration_cast<std::chrono::microseconds>(start2 - start1);
                auto __attribute__ ((unused))duration2 = std::chrono::duration_cast<std::chrono::microseconds>(start3 - start2);
                auto __attribute__ ((unused))duration3 = std::chrono::duration_cast<std::chrono::microseconds>(start4 - start3);
                // if (startm == 0)
                //     logger().log("gn Internal Timings us",std::vector<long>({duration1.count(),duration2.count(),duration3.count()}));
            }
            workerFinalDest += workerDest;
            // if (regionIndex == 2 && !MultiplyByE)
            // {
            //     logger().log("Region Boundaries",std::vector<numType>({regionStartEnergy,regionEndEnergy}));
            //     logger().log("GnTrace Here:",workerDest->trace());
            //     logger().log("Number of regions",m_analyticRegions.size()-1);

            // }
        }
        std::unique_lock locker(workerLock);
        results[storeIdx] = workerFinalDest;
    };

    const size_t eigdim = getEigenValueDimension();
    const int stepSize = std::max(eigdim/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    size_t numWorkers  = m_GammaP.back().size()*ceil((numType)eigdim/stepSize);
    results.resize(numWorkers);
    size_t workerIdx = 0;
    Eigen::setNbThreads(1);
    for (size_t p = 0; p < m_GammaP.back().size(); p++)
    {
        for (size_t mPrime = 0; mPrime < eigdim; mPrime+= stepSize)
        {
            auto endIndex = std::min(mPrime+stepSize,eigdim);
            futures.push_back(pool.queueWork([=,&work](){work(mPrime,endIndex,p,workerIdx);}));
            workerIdx++;
        }
    }
    releaseAssert(numWorkers == workerIdx,"Invalid calculation of number of workers");
    for (auto& fut : futures)
        fut.wait();
    Eigen::setNbThreads(0);
    workerLock.lock(); //paranoia
    for (auto& r : results)
        dest += r;
    if (!isProjectionBasis)
        dest.makeSelfAdjoint();
    // if (!MultiplyByE)
    // {
    //     ComplexHermitianMatrix hermDensity;
    //     checkBasisChangeAssert(static_cast<ComplexSelfAdjointMatrix>(dest).toHermitianBasis(hermDensity),dest.getBasis(),"Dest.HermBasis");
    //     ComplexHermitianMatrix::EigenValueType densityEigVals;
    //     ComplexHermitianMatrix::EigenVectorType densityEigVecs;
    //     releaseAssert(hermDensity.getEigenValuesAndVectors(densityEigVals,densityEigVecs),"FailedToGetdensityeigenvalues");
    //     bool fixedOne = false;
    //     for (long i = 0; i < densityEigVals.rows(); i++)
    //     {
    //         logger().log("densityEigVal",densityEigVals(i)/(2*M_PI));
    //         if (densityEigVals(i) < 0 || densityEigVals(i) > 2*M_PI)
    //             fixedOne = true;
    //         densityEigVals(i) = std::max(densityEigVals(i),0.);
    //         densityEigVals(i) = std::min(densityEigVals(i),2*M_PI);
    //     }
    //     if (fixedOne)
    //     {
    //         dest = densityEigVecs * densityEigVals.asDiagonal(densityEigVecs.getBasis()) * densityEigVecs.inverse();
    //         dest.toBasisC(dest,m_basis);
    //     }
    // }
    workerLock.unlock();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start1);
    if (!MultiplyByE)
        logger().log("Gn Timings us",std::vector<long>({duration1.count()}));
    else
        logger().log("EGn Timings us",std::vector<long>({duration1.count()}));
}

bool SOPGreensFunction::findIndexOfRegion(complexType E, size_t &index) const
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

numType SOPGreensFunction::getRegionRepresentativeEnergy(size_t regionIndex) const
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

SOPGreensFunction::SOPGreensFunction(const SOPGreensFunction &other)
{
    m_Ham = other.m_Ham;
    m_HamAdj = other.m_HamAdj;
    m_selfEnergies = other.m_selfEnergies;
    m_muP = other.m_muP;
    m_muMin = other.m_muMin;
    m_muMax = other.m_muMax;
    m_GammaP = other.m_GammaP;
    m_basis = other.m_basis;
    m_mu = other.m_mu; // Only use if this is not determinable from the self energies.

    /* internal state objects */

    m_GRIsCached = other.m_GRIsCached;
    m_rightEigenVectors = other.m_rightEigenVectors;
    m_InvRightEigenVectors = other.m_InvRightEigenVectors;
    m_EigenValues = other.m_EigenValues;
    m_rightEigenVectorsAdj = other.m_rightEigenVectorsAdj;
    m_InvRightEigenVectorsAdj = other.m_InvRightEigenVectorsAdj;
    m_EigenValuesAdj = other.m_EigenValuesAdj;
    m_poleOrder = other.m_poleOrder;
    m_poleOrderAdj = other.m_poleOrder;

    m_Residues = other.m_Residues;
    m_ResiduesAdj = other.m_ResiduesAdj;

    m_isBareBones = other.m_isBareBones;


    m_analyticRegions = other.m_analyticRegions;

    //Cached Integrals
    m_Gn = other.m_Gn;
    m_GnIsCached = other.m_GnIsCached;

    m_EGn = other.m_EGn;
    m_EGnIsCached = other.m_EGnIsCached;

    // isEquilibrium = other.isEquilibrium;
}

SOPGreensFunction::SOPGreensFunction(const ComplexMatrix &Ham, numType mu, const std::vector<numType> &analyticRegions, GreensFunction::EnforcedSpinSymmetryType enforcedSpinSym)
{
    m_Ham = Ham;
    m_basis = Ham.getBasis();
    m_mu = mu;
    m_analyticRegions = analyticRegions;
    m_enforcedSpinSym = enforcedSpinSym;
    releaseAssert(!BasisManager::isProjection(m_basis[0]) && !BasisManager::isProjection(m_basis[1]),"SOPGreensFunction::SOPGreensFunction(const ComplexMatrix &Ham, numType mu, const std::vector<numType> &analyticRegions, GreensFunction::EnforcedSpinSymmetryType enforcedSpinSym) constructor only in non projected basis");
}
SOPGreensFunction::SOPGreensFunction(const ComplexMatrix &Ham, const ComplexMatrix &HamAdj, numType mu, const std::vector<numType> &analyticRegions, GreensFunction::EnforcedSpinSymmetryType enforcedSpinSym)
{
    m_Ham = Ham;
    m_HamAdj = HamAdj;
    m_basis = Ham.getBasis();
    m_mu = mu;
    m_analyticRegions = analyticRegions;
    m_enforcedSpinSym = enforcedSpinSym;
}
void SOPGreensFunction::setDesiredBasis(const indexBasisT &bas)
{
    releaseAssert(!BasisManager::isProjection(m_basis[0]) && !BasisManager::isProjection(m_basis[1]),
                  "setDesiredBasis called when it is already a projection. This means the cache would need to be rebuilt which we cant do");
    if (!m_GRIsCached)
        buildCache();
    m_basis = bas;
    if (m_isBareBones == false)
    {
        if (BasisManager::isProjection(m_basis[0]) || BasisManager::isProjection(m_basis[1]) || m_basis[1] != m_basis[0])
        {
            generateAdjoint();

            for (auto& EV: m_rightEigenVectorsAdj)
                EV.toBasisC(EV,m_basis);
            for (auto& EV: m_InvRightEigenVectorsAdj)
                EV.toBasisC(EV,m_basis);
        }
        for (auto& EV: m_rightEigenVectors)
            EV.toBasisC(EV,m_basis);
        for (auto& EV: m_InvRightEigenVectors)
            EV.toBasisC(EV,m_basis);
    }
    else
    {
        for (auto& Res: m_Residues)
            for (auto& R : Res)
                R.toBasisC(R,m_basis);
    }
}

SOPGreensFunction::SOPGreensFunction(const ComplexSelfAdjointSparseMatrix &Ham, const std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix> > > &SEs, const std::vector<numType> &analyticRegions,
                                     GreensFunction::EnforcedSpinSymmetryType enforcedSpinSym)
{
    m_Ham = static_cast<ComplexMatrix>(Ham);
    m_basis = m_Ham.getBasis();
    m_selfEnergies = SEs;
    m_analyticRegions = analyticRegions;
    m_enforcedSpinSym = enforcedSpinSym;
}



void SOPGreensFunction::getGr(complexType E, ComplexMatrix &dest) const
{
    if (!m_GRIsCached && !m_isBareBones)
        buildCache();

    //cache is good
    size_t regionIndex = 0;
    bool success = findIndexOfRegion(E,regionIndex);
    releaseAssert(success,"Could not find Region for Energy");
    if (m_isBareBones)
    {
        dest.setZero(m_Residues[0].front().rows(),m_Residues[0].front().cols(),m_basis,m_spinSym);
        const EigenValueVector& eigenValueStore = m_EigenValues[regionIndex];
        const std::vector<ComplexMatrix>& residueStore = m_Residues[regionIndex];
        const Eigen::Vector<int,Eigen::Dynamic> poleOrderStore = m_poleOrder[regionIndex];
        // This is n^4. not good
        for (size_t i = 0; i < residueStore.size(); i++)
        {
            if (poleOrderStore[i] == 1)
                dest += residueStore[i]/(E-eigenValueStore(i));
            else
                dest += residueStore[i]/pow((E-eigenValueStore(i)),poleOrderStore[i]);
        }
    }
    else
    {
        const auto& rightEigenVectorStore = m_rightEigenVectors[regionIndex];
        const auto& rightInvEigenVectorStore = m_InvRightEigenVectors[regionIndex];
        const auto& eigenValuesStore = m_EigenValues[regionIndex];

        EigenVectorMatrix temp;
        temp.resize(rightEigenVectorStore.rows(),rightEigenVectorStore.cols(),m_basis,m_spinSym);
        assert(rightEigenVectorStore.cols() == eigenValuesStore.rows());
        temp.setZero();
        for (long i = 0; i < rightEigenVectorStore.cols(); i++)
        {
            temp.col(i) = rightEigenVectorStore.col(i)/(E-eigenValuesStore(i));
        }
        dest = temp * rightInvEigenVectorStore;
    }
}

void SOPGreensFunction::getGa(complexType E, ComplexMatrix &dest)
{
    if (!m_GRIsCached && !m_isBareBones)
        buildCache();

    bool hasAdjComputed = m_isBareBones? m_poleOrderAdj.size() > 0 : m_EigenValuesAdj.size() > 0;
    if (hasAdjComputed)
    {
        size_t regionIndex = 0;
        bool success = findIndexOfRegion(E,regionIndex);
        releaseAssert(success,"Could not find Region for Energy");
        if (m_isBareBones)
        {
            dest.setZero(m_Residues[0].front().rows(),m_Residues[0].front().cols(),m_basis,m_spinSym);
            const EigenValueVector& eigenValueStore = m_EigenValuesAdj[regionIndex];
            const std::vector<ComplexMatrix>& residueStore = m_ResiduesAdj[regionIndex];
            const Eigen::Vector<int,Eigen::Dynamic> poleOrderStore = m_poleOrderAdj[regionIndex];
            // This is n^4. not good
            for (size_t i = 0; i < residueStore.size(); i++)
            {
                if (poleOrderStore[i] == 1)
                    dest += residueStore[i]/(E-eigenValueStore(i));
                else
                    dest += residueStore[i]/pow((E-eigenValueStore(i)),poleOrderStore[i]);
            }
        }
        else
        {
            const auto& rightEigenVectorStore = m_InvRightEigenVectorsAdj[regionIndex];
            const auto& rightInvEigenVectorStore = m_rightEigenVectorsAdj[regionIndex];
            const auto& eigenValuesStore = m_EigenValuesAdj[regionIndex];

            EigenVectorMatrix temp;
            temp.resize(rightEigenVectorStore.rows(),rightEigenVectorStore.cols(),m_basis,m_spinSym);
            assert(rightEigenVectorStore.cols() == eigenValuesStore.rows());
            temp.setZero();
            for (long i = 0; i < rightEigenVectorStore.cols(); i++)
            {
                temp.col(i) = rightEigenVectorStore.col(i)/(E-eigenValuesStore(i));
            }
            dest = temp * rightInvEigenVectorStore;
        }
    }
    else
    {
        getGr(E,dest);
        dest = dest.adjoint();
    }
}

void SOPGreensFunction::getGnIntegral(ComplexMatrix &dest, bool forceNonEquilibrium, numType EStart, numType EEnd)
{

    if (m_GnIsCached && EStart == negInf && EEnd == posInf)
    {
        dest = m_Gn;
        return;
    }
    if (!m_GRIsCached)
        buildCache();

    if (abs(m_muMin - m_muMax) < 1e-10 && !forceNonEquilibrium)
    {
        logger().log("Mu Difference is small, Equilibrium Green's function");

        IntegrateAnalyticEquilibriumGn(dest,m_muMax,false,true,EStart,EEnd);
    }
    else
    {
        logger().log("Mu Difference is large, Non-Equilibrium Green's function");
        IntegrateAnalyticNonEquilibriumGn(dest,false,EStart,EEnd);
    }
    if (EStart == negInf && EEnd == posInf)
    {
        m_Gn = dest;
        m_GnIsCached = true;
    }

    //fall through
}

void SOPGreensFunction::getEGnIntegral(ComplexMatrix &dest,bool forceNonEquilibrium)
{

    if (m_EGnIsCached)
    {
        dest = m_EGn;
        return;
    }

    if (!m_GRIsCached)
        buildCache();

    if (abs(m_muMax - m_muMin) < 1e-10 && !forceNonEquilibrium)
    {
        logger().log("Mu Difference is small, Equilibrium Green's function");
        IntegrateAnalyticEquilibriumGn(dest,m_muMax,true);
    }
    else
    {
        logger().log("Mu Difference is large, Non-Equilibrium Green's function");
        IntegrateAnalyticNonEquilibriumGn(dest,true);
    }

    m_EGn = dest;
    m_EGnIsCached = true;


}

void SOPGreensFunction::getGrIntegral(ComplexMatrix &dest)
{

    if (!m_GRIsCached)
        buildCache();

    if (abs(m_muMax - m_muMin) < 1e-10)
    {
        logger().log("Mu Difference is small, Equilibrium Green's function");
        IntegrateAnalyticEquilibriumGn(dest,m_muMax,false,false);
    }
    else
    {
        releaseAssert(false,"getGrIntegral in NON-eq situation");
    }
}

std::pair<long, long> SOPGreensFunction::getSpatialDimension() const
{
    if (m_GRIsCached)
    {
        if (m_isBareBones)
            return {m_Residues[0][0].rows(),m_Residues[0][0].cols()};
        else
            return {m_rightEigenVectors[0].rows(),m_InvRightEigenVectors[0].cols()};
    }
    else
         return {m_Ham.rows(),m_Ham.cols()};}

void SOPGreensFunction::getDivergencesOfGr(numType E, std::vector<complexType> &divergences)
{
    if (!m_GRIsCached)
    {
        ComplexMatrix t;
        getGr(0,t);
    }
    if (m_GRIsCached)
    {
        size_t regionIndex;
        bool success = findIndexOfRegion(E,regionIndex);

        divergences.clear();
        releaseAssert(success,"Could not find region for Energy");
        for (long i = 0; i < m_EigenValues[regionIndex].rows(); i++)
        {
            divergences.push_back(m_EigenValues[regionIndex](i));
        }
        for (int i = 0; i < m_EigenValues[regionIndex].rows(); i++)
            if (m_EigenValues[regionIndex](i).imag() > 1e-10)
                logger().log("Pole has Imaginary eigenvalue greater than 0");
    }
}

SOPGreensFunction operator *(const ComplexMatrix& mat, const SOPGreensFunction& GF)
{
    if (!GF.m_GRIsCached)
        GF.buildCache();

    if (mat.getBasis()[0] != GF.m_basis[0])
        GF.generateAdjoint();
    SOPGreensFunction ret = GF;
    for (auto& EV : ret.m_rightEigenVectors)
    {
        EV = mat * EV; //Gr = R\Lambdaish R^{-1}
    }
    for (auto& EV : ret.m_InvRightEigenVectorsAdj) // transform if it exists
    {
        EV = mat * EV; //Ga = R^{-1}^\dagger \Lambdaish R
    }
    ret.m_EGnIsCached = ret.m_GnIsCached = false;
    ret.m_EGn = ret.m_Gn = ComplexMatrix();
    ret.m_Ham = ComplexMatrix();
    ret.m_selfEnergies.clear();
    ret.m_isBareBones = false;
    return ret;
}
SOPGreensFunction operator *(const SOPGreensFunction& GF,const ComplexMatrix& mat)
{
    if (!GF.m_GRIsCached)
        GF.buildCache();
    if (mat.getBasis()[1] != GF.m_basis[1])
        GF.generateAdjoint();

    SOPGreensFunction ret = GF;
    for (auto& EV : ret.m_InvRightEigenVectors)
    {
        EV*=mat; //Gr = R\Lambdaish R^{-1}
    }
    for (auto& EV : ret.m_rightEigenVectorsAdj)
    {
        EV *= mat; //Ga = R^{-1}^\dagger \Lambdaish R
    }
    ret.m_EGnIsCached = ret.m_GnIsCached = false;
    ret.m_EGn = ret.m_Gn = ComplexMatrix();
    ret.m_Ham = ComplexMatrix();
    ret.m_HamAdj = ComplexMatrix();
    ret.m_selfEnergies.clear();
    ret.m_isBareBones = false;
    return ret;
}

SOPGreensFunction operator *(const SOPGreensFunction& GFA,const SOPGreensFunction& GFB)
{
    // releaseAssert(!GFA.m_isBareBones && !GFB.m_isBareBones,"Chained multiplications not implemented");
    if (!GFA.m_GRIsCached)
        GFA.buildCache();
    if (!GFB.m_GRIsCached)
        GFB.buildCache();
    if (GFA.m_basis[0] != GFB.m_basis[0] || GFA.m_basis[1] != GFB.m_basis[1])
    {
        GFA.generateAdjoint();
        GFB.generateAdjoint();
    }

    SOPGreensFunction ret = GFA;
    ret.m_Ham = ComplexMatrix();
    ret.m_selfEnergies.clear();
    ret.m_isBareBones = true;
    ret.m_EigenValues.clear();
    ret.m_rightEigenVectors.clear();
    ret.m_InvRightEigenVectors.clear();

    ret.m_EigenValuesAdj.clear();
    ret.m_rightEigenVectorsAdj.clear();
    ret.m_InvRightEigenVectorsAdj.clear();

    ret.m_poleOrder.clear();
    ret.m_poleOrderAdj.clear();

    ret.m_basis= {GFA.m_basis[0],GFB.m_basis[1]};


    if (GFA.m_analyticRegions != GFB.m_analyticRegions)
    {
        std::vector<numType> analyticRegions = GFA.m_analyticRegions;
        analyticRegions.insert(ret.m_analyticRegions.end(),GFB.m_analyticRegions.begin(),GFB.m_analyticRegions.end());
        ret.m_analyticRegions.clear();
        //deduplicate
        std::sort(analyticRegions.begin(),analyticRegions.end());
        // deal with duplicates
        for (size_t i = 0; i < analyticRegions.size(); i++)
        {
            if (i+1 < analyticRegions.size() && (subtractWithInfinities(analyticRegions[i+1], analyticRegions[i]) < 1e-10))
                continue;
            ret.m_analyticRegions.push_back(analyticRegions[i]);
        }
    }

    ret.m_EigenValues.resize(ret.m_analyticRegions.size()-1);
    ret.m_Residues.resize(ret.m_analyticRegions.size()-1);
    ret.m_poleOrder.resize(ret.m_analyticRegions.size()-1);

    //For Gr
    for (long RegionIdx = 0; RegionIdx < ret.m_analyticRegions.size()-1; RegionIdx++)
    {
        numType regionEnergy = ret.getRegionRepresentativeEnergy(RegionIdx);
        size_t ARegionIdx, BRegionIdx;
        GFA.findIndexOfRegion(regionEnergy,ARegionIdx);
        GFB.findIndexOfRegion(regionEnergy,BRegionIdx);

        EigenValueVector& eigenValueStore = ret.m_EigenValues[RegionIdx];
        std::vector<ComplexMatrix>& residueStore = ret.m_Residues[RegionIdx];
        Eigen::Vector<int,Eigen::Dynamic>& poleOrderStore = ret.m_poleOrder[RegionIdx];

        size_t numberOfTermsEstimate = GFA.m_EigenValues[ARegionIdx].rows()*GFB.m_EigenValues[BRegionIdx].rows()*2;
        std::vector<complexType> eigenValuesVec;
        std::vector<int> poleOrderVec;
        eigenValuesVec.reserve(numberOfTermsEstimate);
        poleOrderVec.reserve(numberOfTermsEstimate);
        residueStore.reserve(numberOfTermsEstimate);

        for (long AEVIdx = 0; AEVIdx < GFA.m_EigenValues[ARegionIdx].rows(); AEVIdx++)
        {
            complexType AEigVal = GFA.m_EigenValues[ARegionIdx](AEVIdx);
            ComplexMatrix AResidue  = GFA.m_rightEigenVectors[ARegionIdx].getCol(AEVIdx) * GFA.m_InvRightEigenVectors[ARegionIdx].getRow(AEVIdx);
            int APoleOrder = GFA.m_poleOrder[ARegionIdx](AEVIdx);
            for (long BEVIdx = 0; BEVIdx < GFB.m_EigenValues[BRegionIdx].rows(); BEVIdx++)
            {
                //Coverup method. We have \frac{A}{E-\varepsilon_0} \frac{B}{E-\varepsilon_1} = \frac{AB/(\varepsilon_0-\varepsilon_1)}{E-\varepsilon_0} + \frac{AB/(\varepsilon_1-\varepsilon_0)}{E-\varepsilon_1}
                //We absorb the residues
                ComplexMatrix BResidue  = GFB.m_rightEigenVectors[BRegionIdx].getCol(BEVIdx) * GFB.m_InvRightEigenVectors[BRegionIdx].getRow(BEVIdx);
                complexType BEigVal = GFB.m_EigenValues[BRegionIdx](BEVIdx);
                int BPoleOrder = GFB.m_poleOrder[BRegionIdx](BEVIdx);

                if (abs(AEigVal-BEigVal) < 1e-8)
                {
                    // releaseAssert(false,"ABEigValDegenerate not handled");
                    //Can assume they are all order one as chained not implemented. See releaseAssert
                    eigenValuesVec.push_back((AEigVal+BEigVal)/2.);
                    residueStore.emplace_back(AResidue*BResidue);
                    poleOrderVec.push_back(APoleOrder+BPoleOrder);
                }
                else
                {
                    eigenValuesVec.push_back(AEigVal);
                    eigenValuesVec.push_back(BEigVal);
                    ComplexMatrix ABResidue = AResidue*BResidue;

                    residueStore.emplace_back(ABResidue*(1./pow(AEigVal-BEigVal,BPoleOrder)));
                    residueStore.emplace_back(ABResidue*(1./pow(BEigVal-AEigVal,APoleOrder)));
                    poleOrderVec.push_back(APoleOrder); // div at E = AEigVal
                    poleOrderVec.push_back(BPoleOrder);
                }
            }
        }
        poleOrderStore = Eigen::Map<Eigen::Vector<int,Eigen::Dynamic>>(poleOrderVec.data(),poleOrderVec.size());
        eigenValueStore = EigenValueVector(Eigen::Map<Eigen::Vector<complexType,Eigen::Dynamic>>(eigenValuesVec.data(),eigenValuesVec.size()),eigenValueStore.getBasis(),enums::SpinSymmetry::NoSpinSym);
    }
    //For Ga if different
    releaseAssert(GFA.m_EigenValuesAdj.size() ==  GFB.m_EigenValuesAdj.size(),"GFA.m_EigenValuesAdj.size() ==  GFB.m_EigenValuesAdj.size()"); //Later
    if (GFA.m_EigenValuesAdj.size() != 0)
    {
        ret.m_EigenValuesAdj.resize(ret.m_analyticRegions.size()-1);
        ret.m_ResiduesAdj.resize(ret.m_analyticRegions.size()-1);
        ret.m_poleOrderAdj.resize(ret.m_analyticRegions.size()-1);
        for (long RegionIdx = 0; RegionIdx < ret.m_analyticRegions.size()-1; RegionIdx++)
        {
            numType regionEnergy = ret.getRegionRepresentativeEnergy(RegionIdx);
            size_t ARegionIdx, BRegionIdx;
            GFA.findIndexOfRegion(regionEnergy,ARegionIdx);
            GFB.findIndexOfRegion(regionEnergy,BRegionIdx);

            EigenValueVector& eigenValueStore = ret.m_EigenValuesAdj[RegionIdx];
            std::vector<ComplexMatrix>& residueStore = ret.m_ResiduesAdj[RegionIdx];
            Eigen::Vector<int,Eigen::Dynamic>& poleOrderStore = ret.m_poleOrderAdj[RegionIdx];

            size_t numberOfTermsEstimate = GFA.m_EigenValuesAdj[ARegionIdx].rows()*GFB.m_EigenValuesAdj[BRegionIdx].rows()*2;
            std::vector<complexType> eigenValuesVec;
            std::vector<int> poleOrderVec;
            eigenValuesVec.reserve(numberOfTermsEstimate);
            poleOrderVec.reserve(numberOfTermsEstimate);
            residueStore.reserve(numberOfTermsEstimate);

            for (long AEVIdx = 0; AEVIdx < GFA.m_EigenValuesAdj[ARegionIdx].rows(); AEVIdx++)
            {
                complexType AEigVal = GFA.m_EigenValuesAdj[ARegionIdx](AEVIdx);
                ComplexMatrix AResidue  = GFA.m_InvRightEigenVectorsAdj[ARegionIdx].getCol(AEVIdx) * GFA.m_rightEigenVectorsAdj[ARegionIdx].getRow(AEVIdx);
                int APoleOrder = GFA.m_poleOrderAdj[ARegionIdx](AEVIdx);

                for (long BEVIdx = 0; BEVIdx < GFB.m_EigenValuesAdj[BRegionIdx].rows(); BEVIdx++)
                {
                    //Coverup method. We have \frac{A}{E-\varepsilon_0} \frac{B}{E-\varepsilon_1} = \frac{AB/(\varepsilon_0-\varepsilon_1)}{E-\varepsilon_0} + \frac{AB/(\varepsilon_1-\varepsilon_0)}{E-\varepsilon_1}
                    //We absorb the residues
                    ComplexMatrix BResidue  = GFB.m_InvRightEigenVectorsAdj[BRegionIdx].getCol(BEVIdx) * GFB.m_rightEigenVectorsAdj[BRegionIdx].getRow(BEVIdx);
                    complexType BEigVal = GFB.m_EigenValuesAdj[BRegionIdx](BEVIdx);
                    int BPoleOrder = GFB.m_poleOrder[BRegionIdx](BEVIdx);

                    if (abs(AEigVal-BEigVal) < 1e-8)
                    {
                        // releaseAssert(false,"ABEigValDegenerate not handled");
                        //Can assume they are all order one as chained not implemented. See releaseAssert
                        eigenValuesVec.push_back((AEigVal+BEigVal)/2.);
                        residueStore.emplace_back(AResidue*BResidue);
                        poleOrderVec.push_back(APoleOrder+BPoleOrder);
                    }
                    else
                    {
                        eigenValuesVec.push_back(AEigVal);
                        eigenValuesVec.push_back(BEigVal);
                        ComplexMatrix ABResidue = AResidue*BResidue;

                        residueStore.emplace_back(ABResidue*(1./pow(AEigVal-BEigVal,BPoleOrder)));
                        residueStore.emplace_back(ABResidue*(1./pow(BEigVal-AEigVal,APoleOrder)));
                        poleOrderVec.push_back(APoleOrder); // div at E = AEigVal
                        poleOrderVec.push_back(BPoleOrder);
                    }
                }
            }
            poleOrderStore = Eigen::Map<Eigen::Vector<int,Eigen::Dynamic>>(poleOrderVec.data(),poleOrderVec.size());
            eigenValueStore = EigenValueVector(Eigen::Map<Eigen::Vector<complexType,Eigen::Dynamic>>(eigenValuesVec.data(),eigenValuesVec.size()),eigenValueStore.getBasis(),enums::SpinSymmetry::NoSpinSym);
        }
    }

    return ret;
}
