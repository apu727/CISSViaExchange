#include "phononselfenergy.h"

#include "EinSumTemplate.h"
// #include "complexmatrixbuffer.h"
#include "kiss_fft.h"
#include "logger.h"
#include "threadpool.h"
#include <utility>
// #include "configfileloader.h"


const bool useCorrelationMatrix = false;
const bool onlyUseDiagonalOfGreensFunc = false;
const bool useDiagonalTensor = false;
const bool useBarePropagator = false;

template <class SelfEnergyMatrixType>
void PhononSelfEnergy<SelfEnergyMatrixType>::CalcSigmaIn(numType E, SelfEnergyMatrixType& dest)
{
    if (useCorrelationMatrix)
    {
        ComplexMatrix ret;
        ret.resize(m_params.MatrixDimension,m_params.MatrixDimension,m_workingBasis,m_spinSym);
        ret.setZero();


        ComplexSelfAdjointMatrix temp1;
        ComplexSelfAdjointMatrix temp2;
        for(auto omega : m_params.omegas)
        {
            m_GFunc->getGnAtE(E-hbar*omega,temp1);
            m_GFunc->getGnAtE(E+hbar*omega,temp2);

            ret += boseEinsteinDistribution(omega)*temp1;
            ret += (boseEinsteinDistribution(omega)+1)*temp2;
        }
        dest.resize(m_params.MatrixDimension,m_params.MatrixDimension,m_workingBasis,m_spinSym);
        dest.setZero();
        for (int i = 0; i < m_params.MatrixDimension; i++)
        {
            for (int j = 0; j < m_params.MatrixDimension; j++)
            {
                if (m_PhononCorrelationMatrix.coeff(i,j) != complexType(0))
                    dest.coeffRef(i,j) = ret(i,j)*m_PhononCorrelationMatrix.coeff(i,j);
            }
        }
    }
    else
    {
        ComplexMatrix ret;
        ret.setZero(m_params.MatrixDimension,m_params.MatrixDimension,m_workingBasis,m_spinSym);

        ComplexSelfAdjointMatrix temp1;
        ComplexSelfAdjointMatrix temp2;
        for(auto omega : m_params.omegas)
        {
            if (useBarePropagator)
            {
                m_bareGnfunc(E-hbar*omega,temp1);
                m_bareGnfunc(E+hbar*omega,temp2);
            }
            else
            {
                m_GFunc->getGnAtE(E-hbar*omega,temp1);
                m_GFunc->getGnAtE(E+hbar*omega,temp2);
            }


            ret += boseEinsteinDistribution(omega)*temp1;
            ret += (boseEinsteinDistribution(omega)+1)*temp2;
        }


        //Sigma^in_{ml'} = \sum_{l,m',n} (g_{mln} g_{m'l'n})_{mlop} (G^n_{lo}...)
        if (useDiagonalTensor)
        {
            complexType meanTrace = ret.trace()/(complexType)ret.cols();;
            dest = m_PhononGDiagonalMatrix;
            dest *= meanTrace;
        }
        else
        {
            SparseTensorView<decltype(dest),complexType,false> destView(dest);
            SparseTensorView<decltype(ret),complexType,onlyUseDiagonalOfGreensFunc> retView(ret);
            Einsum<4,complexType,2,complexType,2,complexType>("mlop,lo,mp")(&m_PhononGTensor,&retView,&destView);
        }
    }
}

template <class SelfEnergyMatrixType>
void PhononSelfEnergy<SelfEnergyMatrixType>::CalcSigmaOut(numType E, SelfEnergyMatrixType& dest)
{
    if (useCorrelationMatrix)
    {
        ComplexMatrix ret;
        ret.setZero(m_params.MatrixDimension,m_params.MatrixDimension,m_workingBasis,m_spinSym);

        ComplexSelfAdjointMatrix temp1;
        ComplexSelfAdjointMatrix temp2;

        for (auto omega : m_params.omegas)
        {
            m_GFunc->getGpAtE(E+hbar*omega,temp1);
            m_GFunc->getGpAtE(E-hbar*omega,temp2);

            ret += boseEinsteinDistribution(omega)*temp1;
            ret += (boseEinsteinDistribution(omega)+1)*temp2;
        }
        dest.resize(m_params.MatrixDimension,m_params.MatrixDimension,m_workingBasis,m_spinSym);
        dest.setZero();
        for (int i = 0; i < m_params.MatrixDimension; i++)
        {
            for (int j = 0; j < m_params.MatrixDimension; j++)
            {
                if (m_PhononCorrelationMatrix.coeff(i,j) != complexType(0))
                    dest.coeffRef(i,j) = ret(i,j)*m_PhononCorrelationMatrix.coeff(i,j);
            }
        }
    }
    else
    {
        ComplexMatrix ret;
        ret.setZero(m_params.MatrixDimension,m_params.MatrixDimension,m_workingBasis,m_spinSym);

        ComplexSelfAdjointMatrix temp1;
        ComplexSelfAdjointMatrix temp2;

        for (auto omega : m_params.omegas)
        {
            if(useBarePropagator)
            {
                m_bareGpfunc(E+hbar*omega,temp1);
                m_bareGpfunc(E-hbar*omega,temp2);
            }
            else
            {
                m_GFunc->getGpAtE(E+hbar*omega,temp1);
                m_GFunc->getGpAtE(E-hbar*omega,temp2);
            }

            ret += boseEinsteinDistribution(omega)*temp1;
            ret += (boseEinsteinDistribution(omega)+1)*temp2;
        }
        if (useDiagonalTensor)
        {
            complexType meanTrace = ret.trace()/(complexType)ret.cols();
            dest = m_PhononGDiagonalMatrix;
            dest *= meanTrace;
        }
        else
        {
            //Sigma^in_{ml'} = \sum_{l,m',n} (g_{mln} g_{m'l'n})_{mlop} (G^n_{lo}...)
            SparseTensorView<decltype(dest),complexType,false> destView(dest);
            SparseTensorView<decltype(ret),complexType,onlyUseDiagonalOfGreensFunc> retView(ret);
            Einsum<4,complexType,2,complexType,2,complexType>("mlop,lo,mp")(&m_PhononGTensor,&retView,&destView);
        }
    }
}

template <class SelfEnergyMatrixType>
numType PhononSelfEnergy<SelfEnergyMatrixType>::boseEinsteinDistribution(numType omega)
{
    if (hbar*omega/(kb*T) > 30)
        return 0;
    else
        return 1.0/(exp(hbar*omega/(kb*T)) - 1);
}

template <class SelfEnergyMatrixType>
void PhononSelfEnergy<SelfEnergyMatrixType>::interpolate(const std::vector<SelfEnergyMatrixType> &matrices, const std::vector<numType>& Energies, const numType E, SelfEnergyMatrixType& dest)
{
    dest.resize(m_params.MatrixDimension,m_params.MatrixDimension,m_workingBasis,m_spinSym);
    dest.setZero();
    if (Energies.size() == 0)
    {
        return;
    }
    numType minE = getActiveEnergyDataPoints().front();
    numType maxE = getActiveEnergyDataPoints().back();
    numType EStep = getActiveEnergyDataPoints()[1] - getActiveEnergyDataPoints()[0];
    numType numberOfSteps = (E-minE)/EStep;
    int idxFloor = floor(numberOfSteps);
    int idxCeil = ceil(numberOfSteps);

    if (idxFloor < 0)
    {//Log something
        dest = matrices.front();
        return;
    }
    else if (idxCeil >= matrices.size())
    {//Log something
        dest = matrices.back();
        return;
    }
    else
    {
        numType weight = (numberOfSteps - idxFloor);
        dest = weight*matrices[idxFloor];
        dest += (1-weight)*matrices[idxCeil];
        return;
    }
}


template <class SelfEnergyMatrixType>
PhononSelfEnergy<SelfEnergyMatrixType>::PhononSelfEnergy(Parameters params)
{
    m_params = params;
    // GreensFunction::makeTimeReversalMatrix(m_timeReversalMatrix,m_params.MatrixDimension);
}

template<class SelfEnergyMatrixType>
void PhononSelfEnergy<SelfEnergyMatrixType>::setGreensFunction(std::shared_ptr<GreensFunction> GFunc)
{
    m_GFunc = GFunc;
    if(useBarePropagator)
    {
        m_bareGnfunc = m_GFunc->getBareGnFunc();
        m_bareGpfunc = m_GFunc->getBareGpFunc();
    }
}

template<class SelfEnergyMatrixType>
void PhononSelfEnergy<SelfEnergyMatrixType>::setPhononGTensor(const SparseTensor<3, complexType> &PhononGTensor)
{
    // g_{mln} g_{m'l'n}
    SparseTensor<4,complexType> temp;
    Einsum<3,complexType,3,complexType,4,complexType>("mln,opn,mlop")(&PhononGTensor,&PhononGTensor,&temp);
    temp.prepareForEinsum<4,2,2,EinsumTemplates::firstType>("mlop,lo,mp",m_PhononGTensor);
    //Einsum<3,complexType,3,complexType,4,complexType>("mln,opn,mlop")(&PhononGTensor,&PhononGTensor,&m_PhononGTensor);


    Einsum<3,complexType,3,complexType,2,complexType>("mln,lpn,mp")(&PhononGTensor,&PhononGTensor,&m_PhononGDiagonalTensor);
    //logger().log(m_PhononGDiagonalTensor.toString());

    m_PhononGDiagonalMatrix.resize(m_PhononGDiagonalTensor.getSize(),m_PhononGDiagonalTensor.getSize(),m_workingBasis,m_spinSym);
    m_PhononGDiagonalMatrix.setZero();

    SparseTensorView<decltype(m_PhononGDiagonalMatrix),complexType,false> view(m_PhononGDiagonalMatrix);
    view.copy(m_PhononGDiagonalTensor);
}

template <>
void PhononSelfEnergy<ComplexMatrix>::asyncReComputeSelfE(int numberOfPoints, numType EnergyRange, std::vector<ComplexMatrix>* SigmaIn, std::vector<ComplexMatrix>* SigmaOut, std::vector<ComplexMatrix>* SigmaR,
                                           std::vector<numType>* EnergyDataPoints)
{
    std::atomic_int finishCount = 0;

    auto work = [BY_VAL_CAPTURE(SigmaR),
                 BY_VAL_CAPTURE(SigmaIn),
                 BY_VAL_CAPTURE(SigmaOut),
                 BY_VAL_CAPTURE(EnergyRange),
                 BY_VAL_CAPTURE(numberOfPoints),
                 BY_VAL_CAPTURE(EnergyDataPoints),
                 BY_REF_CAPTURE(finishCount),
                 &me = *this

    ](int count, int endIndex)
    {
        for (;count < endIndex; count++)
        {
            numType E = count*2.*EnergyRange/numberOfPoints  - EnergyRange;

            (*EnergyDataPoints)[count] = E;

            me.CalcSigmaIn(E,(*SigmaIn)[count]); // these should really be const?
            me.CalcSigmaOut(E,(*SigmaOut)[count]);

            (*SigmaR)[count] = (*SigmaIn)[count] + (*SigmaOut)[count]; // note not causal yet
            (*SigmaR)[count] *= -iu;
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max(numberOfPoints/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    for (int i = 0; i<numberOfPoints; i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,numberOfPoints);
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex);}));
    }

    for (auto& fut : futures)
        fut.wait();

    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
}

template <>
void PhononSelfEnergy<ComplexSparseMatrix>::asyncReComputeSelfE(int numberOfPoints, numType EnergyRange, std::vector<ComplexSparseMatrix>* SigmaIn, std::vector<ComplexSparseMatrix>* SigmaOut, std::vector<ComplexSparseMatrix>* SigmaR,
                                                                 std::vector<numType>* EnergyDataPoints)
{
    std::atomic_int finishCount = 0;

    auto work = [BY_VAL_CAPTURE(SigmaR),
                 BY_VAL_CAPTURE(SigmaIn),
                 BY_VAL_CAPTURE(SigmaOut),
                 BY_VAL_CAPTURE(EnergyRange),
                 BY_VAL_CAPTURE(numberOfPoints),
                 BY_VAL_CAPTURE(EnergyDataPoints),
                 BY_REF_CAPTURE(finishCount),
                 &me = *this
    ](int count, int endIndex)
    {
        for (;count < endIndex; count++)
        {
            //logger().log("on",count);
            numType E = count*2.*EnergyRange/numberOfPoints  - EnergyRange;

            (*EnergyDataPoints)[count] = E;

            me.CalcSigmaIn(E,(*SigmaIn)[count]);
            me.CalcSigmaOut(E,(*SigmaOut)[count]);

            (*SigmaR)[count] = (*SigmaIn)[count] + (*SigmaOut)[count]; // note not causal yet
            (*SigmaR)[count] *= -iu;

            (*SigmaR)[count].makeCompressed();
            (*SigmaOut)[count].makeCompressed();
            (*SigmaIn)[count].makeCompressed();
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max(numberOfPoints/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    for (int i = 0; i<numberOfPoints; i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,numberOfPoints);
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex);}));
    }

    for (auto& fut : futures)
        fut.wait();

    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
}

template <>
void PhononSelfEnergy<ComplexMatrix>::asyncMakeCausal(int numberOfPoints, std::vector<ComplexMatrix> *SigmaR)
{
    std::atomic_int finishCount = 0;

    auto work = [BY_VAL_CAPTURE(SigmaR),
                 BY_VAL_CAPTURE(numberOfPoints),
                 BY_REF_CAPTURE(finishCount),
                 &me = *this
    ](unsigned long i)
    {
        kiss_fft_cfg inverseCfg = kiss_fft_alloc( numberOfPoints ,false ,0,0 );// Fourier transform between energy and time is weird
        kiss_fft_cfg forwardCfg = kiss_fft_alloc( numberOfPoints ,true ,0,0 );// Fourier transform between energy and time is weird


        static_assert(std::is_same<kiss_fft_scalar,numType>::value,"NUMTYPE MUST BE SAME AS KISS_FFT_SCALAR");
        std::vector<kiss_fft_cpx> cx_in;
        std::vector<kiss_fft_cpx> cx_out;
        cx_in.resize(numberOfPoints);
        cx_out.resize(numberOfPoints);
        assert(numberOfPoints % 2 == 0);

        for (unsigned long j = 0; j < (*SigmaR)[0].cols(); j++)
        {
            // if (m_PhononCorrelationMatrix.coeff(i,j) == complexType(0))
            //     continue;
            bool foundNonZero = false;
            for (int idx = 0; idx < numberOfPoints/2; idx++)
            {
                cx_in[idx+numberOfPoints/2] = {(*SigmaR)[idx](i,j).real(),(*SigmaR)[idx](i,j).imag()};
                if (cx_in[idx+numberOfPoints/2].r != 0 || cx_in[idx+numberOfPoints/2].i != 0)
                    foundNonZero = true;
            }
            for (int idx = numberOfPoints/2; idx < numberOfPoints; idx++)
            {
                cx_in[idx-numberOfPoints/2] = {(*SigmaR)[idx](i,j).real(),(*SigmaR)[idx](i,j).imag()};
                if (cx_in[idx-numberOfPoints/2].r != 0 || cx_in[idx-numberOfPoints/2].i != 0)
                    foundNonZero = true;
            }
            if (!foundNonZero)
                continue;
            //Computes sum_{E=0}^{E=numberOfPoints} \exp(iu*2*PI*E*t/numberOfPoints) \Sigma^R(E)
            // t in [0,numberOfPoints-1]
            kiss_fft( inverseCfg , &cx_in[0] , &cx_out[0] );
            //cx_out[numberOfPoints/2] is nyquist frequency. i.e. t = numberOfPoints/2
            memset(&cx_out[numberOfPoints/2+1],0,sizeof(kiss_fft_cpx)*(numberOfPoints/2-1));
            cx_out[0].r *= 0.5;
            cx_out[0].i *= 0.5;
            cx_out[numberOfPoints/2].r *= 0.5;
            cx_out[numberOfPoints/2].i *= 0.5;

            //Computes \int \exp(-iu E t) \Sigma^R(t) dt
            kiss_fft( forwardCfg , &cx_out[0] , &cx_in[0] );

            for (int idx = 0; idx < numberOfPoints/2; idx++)
            {
                //kissFFt does not do normalisation so we have to.
                //(*SigmaR)[idx].coeff(i,j) = complexType(cx_in[idx+numberOfPoints/2].r , cx_in[idx+numberOfPoints/2].i)/(double)numberOfPoints;
                (*SigmaR)[idx](i,j) = complexType(cx_in[idx+numberOfPoints/2].r , cx_in[idx+numberOfPoints/2].i)/(double)numberOfPoints;
            }
            for (int idx = numberOfPoints/2; idx < numberOfPoints; idx++)
            {
                //kissFFt does not do normalisation so we have to.
                //(*SigmaR)[idx].coeff(i,j) = complexType(cx_in[idx-numberOfPoints/2].r , cx_in[idx-numberOfPoints/2].i)/(double)numberOfPoints;
                (*SigmaR)[idx](i,j) = complexType(cx_in[idx-numberOfPoints/2].r , cx_in[idx-numberOfPoints/2].i)/(double)numberOfPoints;
            }
        }

        kiss_fft_free(forwardCfg);
        kiss_fft_free(inverseCfg);
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const long stepSize = 1;

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    const long maxI = (*SigmaR)[0].rows();
    for (long i = 0; i<maxI; i++)
    {
        futures.push_back(pool.queueWork([=,&work](){work(i);}));
    }

    for (auto& fut : futures)
        fut.wait();

    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
}

template <>
void PhononSelfEnergy<ComplexSparseMatrix>::asyncMakeCausal(int numberOfPoints, std::vector<ComplexSparseMatrix> *SigmaR)
{
    std::atomic_int finishCount = 0;
    std::mutex mymutex;
    auto work = [BY_VAL_CAPTURE(SigmaR),
                 BY_VAL_CAPTURE(numberOfPoints),
                 BY_REF_CAPTURE(finishCount),
                 BY_REF_CAPTURE(mymutex)
    ](unsigned long i)
    {
        kiss_fft_cfg inverseCfg = kiss_fft_alloc( numberOfPoints ,false ,0,0 );
        kiss_fft_cfg forwardCfg = kiss_fft_alloc( numberOfPoints ,true ,0,0 );


        static_assert(std::is_same<kiss_fft_scalar,numType>::value,"NUMTYPE MUST BE SAME AS KISS_FFT_SCALAR");
        std::vector<kiss_fft_cpx> cx_in;
        std::vector<kiss_fft_cpx> cx_out;
        cx_in.resize(numberOfPoints);
        cx_out.resize(numberOfPoints);
        assert(numberOfPoints % 2 == 0);

        for (unsigned long j = 0; j < (*SigmaR)[0].cols(); j++)
        {
            // if (m_PhononCorrelationMatrix.coeff(i,j) == complexType(0))
            //     continue;
            mymutex.lock();
            bool foundNonZero = false;
            for (int idx = 0; idx < numberOfPoints/2; idx++)
            {
                cx_in[idx+numberOfPoints/2] = {(*SigmaR)[idx].coeff(i,j).real(),(*SigmaR)[idx].coeff(i,j).imag()};
                if (cx_in[idx+numberOfPoints/2].r != 0 || cx_in[idx+numberOfPoints/2].i != 0)
                    foundNonZero = true;

            }
            for (int idx = numberOfPoints/2; idx < numberOfPoints; idx++)
            {
                cx_in[idx-numberOfPoints/2] = {(*SigmaR)[idx].coeff(i,j).real(),(*SigmaR)[idx].coeff(i,j).imag()};
                if (cx_in[idx-numberOfPoints/2].r != 0 || cx_in[idx-numberOfPoints/2].i != 0)
                    foundNonZero = true;
            }
            mymutex.unlock();
            if (!foundNonZero)
                continue;
            //Computes sum_{E=0}^{E=numberOfPoints} \exp(iu*2*PI*E*t/numberOfPoints) \Sigma^R(E)
            // t in [0,numberOfPoints-1]
            kiss_fft( inverseCfg , &cx_in[0] , &cx_out[0] );
            //cx_out[numberOfPoints/2] is nyquist frequency. i.e. t = numberOfPoints/2
            memset(&cx_out[numberOfPoints/2+1],0,sizeof(kiss_fft_cpx)*(numberOfPoints/2-1));
            cx_out[0].r *= 0.5;
            cx_out[0].i *= 0.5;
            cx_out[numberOfPoints/2].r *= 0.5;
            cx_out[numberOfPoints/2].i *= 0.5;

            // for (int idx = 1; idx < numberOfPoints/2 + 1; idx++)
            // {
            //     cx_out[idx].r *= 2;
            //     cx_out[idx].i *= 2;
            // }

            //Computes \int \exp(-iu E t) \Sigma^R(t) dt
            kiss_fft( forwardCfg , &cx_out[0] , &cx_in[0] );
            mymutex.lock();
            for (int idx = 0; idx < numberOfPoints/2; idx++)
            {
                //kissFFt does not do normalisation so we have to.
                //(*SigmaR)[idx].coeff(i,j) = complexType(cx_in[idx+numberOfPoints/2].r , cx_in[idx+numberOfPoints/2].i)/(double)numberOfPoints;
                (*SigmaR)[idx].coeffRef(i,j) = complexType(cx_in[idx+numberOfPoints/2].r , cx_in[idx+numberOfPoints/2].i)/(double)numberOfPoints;
            }
            for (int idx = numberOfPoints/2; idx < numberOfPoints; idx++)
            {
                //kissFFt does not do normalisation so we have to.
                //(*SigmaR)[idx].coeff(i,j) = complexType(cx_in[idx-numberOfPoints/2].r , cx_in[idx-numberOfPoints/2].i)/(double)numberOfPoints;
                (*SigmaR)[idx].coeffRef(i,j) = complexType(cx_in[idx-numberOfPoints/2].r , cx_in[idx-numberOfPoints/2].i)/(double)numberOfPoints;
            }
            mymutex.unlock();
        }

        kiss_fft_free(forwardCfg);
        kiss_fft_free(inverseCfg);
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const long stepSize = 1;

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    const long maxI = (*SigmaR)[0].rows();
    for (long i = 0; i<maxI; i++)
    {
        futures.push_back(pool.queueWork([=,&work](){work(i);}));
    }

    for (auto& fut : futures)
        fut.wait();

    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
    for (ComplexSparseMatrix &SR : *SigmaR)
    {
        SR.makeCompressed();
    }
}

template <class SelfEnergyMatrixType>
void PhononSelfEnergy<SelfEnergyMatrixType>::recomputeSelfEnergies()
{
    assert(m_GFunc != nullptr);
    std::vector<SelfEnergyMatrixType>* SigmaIn = &m_SigmaIn1;
    std::vector<SelfEnergyMatrixType>* SigmaOut = &m_SigmaOut1;
    std::vector<SelfEnergyMatrixType>* SigmaR = &m_SigmaR1;
    std::vector<numType>* EnergyDataPoints = &m_EnergyDataPoints1;

    if (!activeBuffer)
    {

        SigmaIn = &m_SigmaIn2;
        SigmaOut = &m_SigmaOut2;
        SigmaR = &m_SigmaR2;
        EnergyDataPoints = &m_EnergyDataPoints2;
    }


    numType EnergyRange = 0;
    if (abs(m_params.lowestEnergy) > abs(m_params.largestEnergy))
        EnergyRange =  abs(m_params.lowestEnergy);
    else
        EnergyRange = abs(m_params.largestEnergy);

    int numberOfPoints = 2*int(10.*EnergyRange/(m_params.smallestEnergyFeature));
    numberOfPoints = kiss_fft_next_fast_size(numberOfPoints);
    EnergyRange = (m_params.smallestEnergyFeature)*numberOfPoints/20;
    m_params.largestEnergy = -EnergyRange;
    m_params.largestEnergy = EnergyRange;



    SigmaIn->resize(numberOfPoints);
    SigmaOut->resize(numberOfPoints);
    SigmaR->resize(numberOfPoints);
    EnergyDataPoints->resize(numberOfPoints);
    logger().log("Phonon self Energy, Number of points",numberOfPoints);
    logger().log("Phonon self Energy, Energy Resolution",EnergyRange/numberOfPoints);
    logger().log("Phonon self Energy, Energy Range",EnergyRange);

    asyncReComputeSelfE(numberOfPoints,EnergyRange,SigmaIn,SigmaOut,SigmaR,EnergyDataPoints);



    logger().log("Phonon self Energy, Make Causal");
    logger().log("Energy at numberOfPoints/2",(*EnergyDataPoints)[numberOfPoints/2]);
    asyncMakeCausal(numberOfPoints,SigmaR);
    SelfEnergyMatrixType tempSigmaA;
    tempSigmaA = (*SigmaR)[numberOfPoints/2].adjoint();

    ComplexMatrix GammaMAt = iu*((*SigmaR)[numberOfPoints/2]  - tempSigmaA);
    ComplexMatrix SumSigmaInOut = (*SigmaIn)[numberOfPoints/2] + (*SigmaOut)[numberOfPoints/2];
    ComplexMatrix diff = GammaMAt - SumSigmaInOut;
    logger().log("DiffNorm", diff.norm());

    // Sigma^K = Sigma^< + Sigma^>
    // Sigma^K = iu Sigma^in - iu Sigma^out
    ComplexMatrix SigmaK1 = iu*(*SigmaIn)[numberOfPoints/2-1] - iu*(*SigmaOut)[numberOfPoints/2-1];

    // G^K - G^R + G^A &= 2G^<
    tempSigmaA = (*SigmaR)[numberOfPoints/2-1].adjoint();
    ComplexMatrix SigmaK2 = 2.*iu*(*SigmaIn)[numberOfPoints/2-1] + (*SigmaR)[numberOfPoints/2-1] - tempSigmaA;
    ComplexMatrix diff2 = SigmaK1 - SigmaK2;
    logger().log("diff2", diff2.norm());
    //SigmaR is now causal
    activeBuffer = !activeBuffer;

}

template <class SelfEnergyMatrixType>
void PhononSelfEnergy<SelfEnergyMatrixType>::getSigmaIn(numType E, SelfEnergyMatrixType& dest)
{
    interpolate(getActiveSigmaIn(),getActiveEnergyDataPoints(),E,dest);
}

template <class SelfEnergyMatrixType>
void PhononSelfEnergy<SelfEnergyMatrixType>::getSigmaOut(numType E, SelfEnergyMatrixType& dest)
{
    interpolate(getActiveSigmaOut(),getActiveEnergyDataPoints(),E,dest);
}

template <class SelfEnergyMatrixType>
void PhononSelfEnergy<SelfEnergyMatrixType>::getSigmaR(complexType E, SelfEnergyMatrixType& dest)
{
    if (E.imag() != 0)
        logger().log("No analytic continuation of this self energy");
    interpolate(getActiveSigmaR(),getActiveEnergyDataPoints(),E.real(),dest);
}

template <class SelfEnergyMatrixType>
void PhononSelfEnergy<SelfEnergyMatrixType>::getSigmaA(complexType E, SelfEnergyMatrixType& dest)
{
    SelfEnergyMatrixType sr;
    getSigmaR(std::conj(E),sr);
    dest  = sr.adjoint();
}

template<class SelfEnergyMatrixType>
void PhononSelfEnergy<SelfEnergyMatrixType>::getSigmaK(numType E, SelfEnergyMatrixType &dest)
{
    SelfEnergyMatrixType temp;
    // Sigma^K = Sigma^< + Sigma^>
    // Sigma^K = iu Sigma^in - iu Sigma^out
    getSigmaIn(E,dest);
    getSigmaOut(E,temp);
    dest -= temp;
    dest *= iu;
    // G^K - G^R + G^A &= 2G^<
    // ComplexMatrix SigmaK2;
    // ComplexSparseMatrix temp2;

    // getSigmaIn(E,temp2);
    // temp2 *= 2.*iu;
    // SigmaK2 = temp2.toDense();

    // getSigmaR(E,temp2);
    // SigmaK2 += temp2;

    // getSigmaA(E,temp2);
    // SigmaK2 -= temp2;

    // temp2 = dest - SigmaK2;
    // if (temp2.norm() > 1e-10)
    // {
    //     logger().log("now",temp2.norm());
    //     auto Sin = MatrixBuffer<SelfEnergyMatrixType>::getInstance().getAMatrix();
    //     auto Sout = MatrixBuffer<SelfEnergyMatrixType>::getInstance().getAMatrix();
    //     auto SR = MatrixBuffer<SelfEnergyMatrixType>::getInstance().getAMatrix();
    //     getSigmaIn(E,Sin);
    //     getSigmaOut(E,Sout);
    //     getSigmaR(E,SR);

    //     ComplexMatrix SigmaK1 = iu*Sin - iu*Sout;

    //     // G^K - G^R + G^A &= 2G^<
    //     ComplexMatrix SigmaK3 = 2.*iu*Sin.get() + SR.get() - SR->toDense().adjoint();
    //     ComplexMatrix diff2 = SigmaK1 - SigmaK3;
    //     logger().log("diff2", diff2.norm());
    // }
}

// template<class SelfEnergyMatrixType>
// void PhononSelfEnergy<SelfEnergyMatrixType>::save(std::string filename) const
// {
//     logger logFile = logger(filename,true);
//     logFile.log("PhononSelfEnergy<SelfEnergyMatrixType>::Parameters\n{");
//     logFile.log("omegas",m_params.omegas);
//     logFile.log("lowestEnergy",m_params.lowestEnergy);
//     logFile.log("largestEnergy",m_params.largestEnergy);
//     logFile.log("smallestEnergyFeature",m_params.smallestEnergyFeature);
//     logFile.log("MatrixDimension",m_params.MatrixDimension);
//     logFile.log("Name",this->m_prettyName);
//     logFile.log("}");
// }

// template<class SelfEnergyMatrixType>
// void PhononSelfEnergy<SelfEnergyMatrixType>::load(FILE *file)
// {
//     while (true)
//     {
//         char objectName[30];
//         int ret = fscanf(file," %29[^{}:\n ] : ",objectName);
//         if (ret <= 0)
//             break;
//         std::string objectName_s(objectName);
//         if (objectName_s == "omegas")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.omegas,numTypeCode);
//         }
//         else if (objectName_s == "lowestEnergy")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.lowestEnergy);
//         }
//         else if (objectName_s == "largestEnergy")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.largestEnergy);
//         }
//         else if (objectName_s == "smallestEnergyFeature")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.smallestEnergyFeature);
//         }
//         else if (objectName_s == "MatrixDimension")
//         {
//             uint32_t dummy;
//             ConfigFileLoader::loadParameter(file,dummy);
//         }
//         else if (objectName_s == "Name")
//         {
//             ConfigFileLoader::loadParameter(file,this->m_prettyName);
//         }
//         else
//         {
//             logger().log("Unknown PhononSelfEnergy parameter: ",objectName_s);
//         }
//     }
// }

template class PhononSelfEnergy<SelfEnergyMatrix>;

