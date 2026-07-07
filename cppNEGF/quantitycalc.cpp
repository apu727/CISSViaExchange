#include "quantitycalc.h"
#include "integrator.h"
#include "threadpool.h"
#include "configfileloader.h"




std::vector<numType> autoRange(long NumberOfSteps, std::shared_ptr<GreensFunction> gFunc)
{
    std::vector<complexType> poles;
    gFunc->getEstimateForDivergencesOfGr(0,poles);

    //Need to sanity check the poles since the imaginary parts may be zero. In this case add a broadening of 1e-6 to it.
    for(auto& p : poles)
    {
        if (abs(p.imag()) < 1e-12)
            p -= iu*1e-6;
    }
    std::vector<numType> cumWeightFunctionPoints;
    std::vector<numType> EnergyPoints;
    constructSpectralWeightFunction(negInf,posInf,NumberOfSteps,poles,cumWeightFunctionPoints,EnergyPoints);
    return EnergyPoints;

}

std::vector<Eigen::VectorXd> interpolateWithSpectralWeight(std::vector<numType> points,std::vector<numType> existingPoints, std::vector<Eigen::VectorXd> existingData,
                                                           std::shared_ptr<GreensFunction> gFunc, numType artificialBroadening)
{
    std::vector<complexType> poles;
    gFunc->getEstimateForDivergencesOfGr(0,poles);
    for (auto& p : poles)
    {
        p -= artificialBroadening*iu;
    }
    auto weightFunction =
        [CONST_REF_CAPTURE(poles)](long double E)
    {
        long double sum = 0;
        for (std::complex<long double> p : poles)
        {
            sum += -2*(std::complex<long double>(1)/(E-p)).imag();
        }
        return sum;
    };
    std::vector<Eigen::VectorXd> ret(points.size());

    for (size_t i = 0; i < points.size(); i++)
    {
        size_t lowerBoundIndex = -1;
        size_t upperBoundIndex = -1;
        for (size_t idx = 0; idx < existingPoints.size(); idx++)
        {
            if (existingPoints[idx] >= points[i])
            {
                upperBoundIndex = idx;
                if (idx > 0)
                    lowerBoundIndex = idx-1;
                break;
            }
        }
        if (lowerBoundIndex == -1 && upperBoundIndex == 0)
        {// off the bottom
            ret[i] = existingData[upperBoundIndex]*(weightFunction(points[i])/weightFunction(existingPoints[upperBoundIndex]));
        }
        else if (upperBoundIndex == -1)
        {// off the top
            ret[i] = existingData[upperBoundIndex]*(weightFunction(points[i])/weightFunction(existingPoints.back()));
        }
        else
        {//Inside
            ret[i] = existingData[upperBoundIndex]*(weightFunction(points[i])/weightFunction(existingPoints[upperBoundIndex]));
            ret[i] += existingData[lowerBoundIndex]*(weightFunction(points[i])/weightFunction(existingPoints[lowerBoundIndex]));
            ret[i] /= 2;
        }
    }
    return ret;
}
CurrentCalc::CurrentCalc(Parameters param) {m_params = param;}

numType CurrentCalc::getiE(numType E) const
{
    if (m_selfEnergy == nullptr)
        return 0;
    assert(m_GFunc != nullptr);
    SelfEnergyMatrix SigmaIn;
    SelfEnergyMatrix SigmaOut;
    ComplexSelfAdjointMatrix Gp;
    ComplexSelfAdjointMatrix Gn;
    ComplexMatrix temp1;
    ComplexMatrix temp2;

    m_selfEnergy->getSigmaIn(E,SigmaIn);
    m_selfEnergy->getSigmaOut(E,SigmaOut);
    m_GFunc->getGpAtE(E,Gp);
    m_GFunc->getGnAtE(E,Gn);

    temp1 = SigmaIn * static_cast<ComplexMatrix>(Gp);
    temp2 = SigmaOut * static_cast<ComplexMatrix>(Gn);
    temp1 -= temp2;

    complexType ret = (e/h)*(temp1.trace());
    assert (abs(ret.imag()) < 1e-10);
    return ret.real();
}

std::vector<numType> CurrentCalc::getiE(std::vector<numType> Energies) const
{
    std::atomic_int finishCount = 0;
    std::vector<numType> individualSums(Energies.size());

    auto work = [BY_REF_CAPTURE(individualSums),
                 BY_REF_CAPTURE(finishCount),
                 CONST_REF_CAPTURE(Energies),
                     &me = std::as_const(*this)
    ](unsigned long startIndex, unsigned long endIndex)
    {
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            individualSums[idx] = me.getiE(Energies[idx]);
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max(Energies.size()/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    Eigen::setNbThreads(1);
    for (unsigned long i = 0; i<Energies.size(); i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,Energies.size());
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex);}));
    }

    for (auto& fut : futures)
        fut.wait();
    Eigen::setNbThreads(0);
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
    return individualSums;
}

void CurrentCalc::asyncI(std::vector<numType>& individualSums, numType& step) const
{
    std::atomic_int finishCount = 0;
    individualSums.resize(m_params.numberOfIntegrationPoints);
    step = (m_params.endE - m_params.startE)/m_params.numberOfIntegrationPoints;
    if (step < 1e-8)
        logger().log("WARN: Step size is small:", step);

    auto work = [BY_REF_CAPTURE(individualSums),
                 BY_REF_CAPTURE(finishCount),
                 CONST_REF_CAPTURE(m_params),
                 BY_VAL_CAPTURE(step),
                 &me = std::as_const(*this)
    ](unsigned long startIndex, unsigned long endIndex)
    {
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            individualSums[idx] = me.getiE(m_params.startE + idx*step);
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max(m_params.numberOfIntegrationPoints/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;

    Eigen::setNbThreads(1);
    for (int i = 0; i<m_params.numberOfIntegrationPoints; i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,m_params.numberOfIntegrationPoints);
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex);}));
    }

    for (auto& fut : futures)
        fut.wait();
    Eigen::setNbThreads(0);
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
}

numType CurrentCalc::getI() const
{
    if (m_selfEnergy == nullptr)
        return 0;
    //trapezoid rule
    numType sum = 0;

    std::vector<numType> individualSums;
    numType step;
    asyncI(individualSums,step);
    bool warned = false;
    for (unsigned long idx = 1; idx < m_params.numberOfIntegrationPoints-1; idx++)
    {
        //sum += getiE(m_params.startE + idx*step)*step;
        sum += individualSums[idx];
        if (abs(individualSums[idx]) < 1e-10 && !warned)
        {
            logger().log("WARN: small value in integration",individualSums[idx]);
            warned = true;
        }
    }
    //fix endpoints

    //sum += getiE(m_params.startE)*step/2;
    //sum += getiE(m_params.endE)*step/2;
    sum += individualSums.front()/2;
    sum += individualSums.back()/2;
    sum *= step;
    if (abs(individualSums.front()) > 1e-10)
        logger().log("WARN: Start point is not zero",individualSums.front());
    if (abs(individualSums.back()) > 1e-10)
        logger().log("WARN: End point is not zero",individualSums.back());

    logger().log("Current Integration, Number Of points", m_params.numberOfIntegrationPoints);
    return sum;
}

// void CurrentCalc::save(std::string filename) const
// {
//     logger logFile = logger(filename,true);
//     logFile.log("CurrentCalc::Parameters\n{");
//     logFile.log("numberOfIntegrationPoints",m_params.numberOfIntegrationPoints);
//     logFile.log("startE",m_params.startE);
//     logFile.log("endE",m_params.endE);
//     logFile.log("SelfEnergyName", m_selfEnergy->m_prettyName);
//     logFile.log("}");
// }

// void CurrentCalc::load(FILE *file)
// {
//     m_params.startE = 0;
//     m_params.endE = 0;
//     m_params.numberOfIntegrationPoints = 1;
//     while (true)
//     {
//         char objectName[30];
//         int ret = fscanf(file," %29[^{}:\n ] : ",objectName);
//         if (ret <= 0)
//             break;
//         std::string objectName_s(objectName);
//         if (objectName_s == "numberOfIntegrationPoints")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.numberOfIntegrationPoints);
//         }
//         else if (objectName_s == "startE")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.startE);
//         }
//         else if (objectName_s == "endE")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.endE);
//         }
//         else if (objectName_s == "SelfEnergyName")
//         {
//             ConfigFileLoader::loadParameter(file,m_selfEnergyName);
//         }
//         else
//         {
//             logger().log("Unknown CurrentCalc parameter: ",objectName_s);
//         }
//     }
// }

void ElectronDensityCalc::asyncnEr(std::vector<Eigen::VectorXd > &individualSums, numType &step) const
{
    std::atomic_int finishCount = 0;
    individualSums.resize(m_params.numberOfIntegrationPoints);
    step = (m_params.endE - m_params.startE)/m_params.numberOfIntegrationPoints;
    if (step < 1e-8)
        logger().log("WARN: Step size is small:", step);

    auto work = [BY_REF_CAPTURE(individualSums),
                 BY_REF_CAPTURE(finishCount),
                 CONST_REF_CAPTURE(m_params),
                 BY_VAL_CAPTURE(step),
                 &me = std::as_const(*this)
    ](unsigned long startIndex, unsigned long endIndex)
    {
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            individualSums[idx] = me.getnEr(m_params.startE + idx*step);
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max(m_params.numberOfIntegrationPoints/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    Eigen::setNbThreads(1);
    for (int i = 0; i<m_params.numberOfIntegrationPoints; i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,m_params.numberOfIntegrationPoints);
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex);}));
    }

    for (auto& fut : futures)
        fut.wait();
    Eigen::setNbThreads(0);
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
}

void ElectronDensityCalc::asyncnEr(std::vector<Eigen::VectorXd> &individualSums, numType &step, const std::string& basis) const
{
    std::atomic_int finishCount = 0;
    individualSums.resize(m_params.numberOfIntegrationPoints);
    step = (m_params.endE - m_params.startE)/m_params.numberOfIntegrationPoints;
    if (step < 1e-8)
        logger().log("WARN: Step size is small:", step);

    auto work = [BY_REF_CAPTURE(individualSums),
                 BY_REF_CAPTURE(finishCount),
                 CONST_REF_CAPTURE(m_params),
                 CONST_REF_CAPTURE(basis),
                 BY_VAL_CAPTURE(step),
                 &me = std::as_const(*this)
    ](unsigned long startIndex, unsigned long endIndex)
    {
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            individualSums[idx] = me.getnEr(m_params.startE + idx*step,basis);
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max(m_params.numberOfIntegrationPoints/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    Eigen::setNbThreads(1);
    for (int i = 0; i<m_params.numberOfIntegrationPoints; i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,m_params.numberOfIntegrationPoints);
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex);}));
    }

    for (auto& fut : futures)
        fut.wait();
    Eigen::setNbThreads(0);
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
}

ComplexSelfAdjointMatrix ElectronDensityCalc::getGn(bool perturb, numType EStart, numType EEnd) const
// perturb is an option since for Graphs you want perturb = true but for saving you want perturb = false
{
    ComplexSelfAdjointMatrix ret;
    assert(m_GFunc != nullptr);
    m_GFunc->getGnIntegral(m_params.numberOfIntegrationPoints,ret,perturb,EStart,EEnd);
    ret *= 1./(2*M_PI);
    return ret;
}

ComplexSelfAdjointMatrix ElectronDensityCalc::getNonEqGn(bool perturb) const
// perturb is an option since for Graphs you want perturb = true but for saving you want perturb = false
{
    ComplexSelfAdjointMatrix ret;
    assert(m_GFunc != nullptr);
    m_GFunc->getGnNonEqIntegral(m_params.numberOfIntegrationPoints,ret,perturb);
    ret *= 1./(2*M_PI);
    return ret;
}

Eigen::VectorXd ElectronDensityCalc::getnR() const
{
    ComplexSelfAdjointMatrix temp;
    assert(m_GFunc != nullptr);
    m_GFunc->getGnIntegral(m_params.numberOfIntegrationPoints,temp,true);

    return (temp.diagonal()/(2*M_PI)).real();

}

Eigen::VectorXcd ElectronDensityCalc::getnS() const
{
    ComplexSelfAdjointMatrix temp;
    assert(m_GFunc != nullptr);
    m_GFunc->getGnIntegral(m_params.numberOfIntegrationPoints,temp,true);
    Eigen::VectorXcd ret;
    ret.resize(temp.cols()*2);
    for (long i = 0; i < temp.cols()/2; i++)
    {
        ret(4*i) = temp(2*i,2*i);
        ret(4*i+1) = temp(2*i,2*i+1);
        ret(4*i+2) = temp(2*i+1,2*i);
        ret(4*i+3) = temp(2*i+1,2*i+1);
    }
    ret *= 1./(2*M_PI);
    return ret;
}

Eigen::VectorXcd ElectronDensityCalc::getnS(const std::string& basis) const
{
    ComplexSelfAdjointMatrix temp;
    assert(m_GFunc != nullptr);
    m_GFunc->getGnIntegral(m_params.numberOfIntegrationPoints,temp,true);
    temp.toBasisC(temp,basis);
    Eigen::VectorXcd ret;
    ret.resize(temp.cols()*2);
    for (long i = 0; i < temp.cols()/2; i++)
    {
        ret(4*i) = temp(2*i,2*i);
        ret(4*i+1) = temp(2*i,2*i+1);
        ret(4*i+2) = temp(2*i+1,2*i);
        ret(4*i+3) = temp(2*i+1,2*i+1);
    }
    ret *= 1./(2*M_PI);
    return ret;
}

Eigen::VectorXd ElectronDensityCalc::getnR(const std::string& basis) const
{
    ComplexSelfAdjointMatrix temp;
    assert(m_GFunc != nullptr);
    m_GFunc->getGnIntegral(m_params.numberOfIntegrationPoints,temp,true);
    temp.toBasisC(temp,basis);
    return (temp.diagonal()/(2*M_PI)).real();

}

Eigen::VectorXd ElectronDensityCalc::getnEr(numType E) const
{
    assert(m_GFunc != nullptr);
    return (m_GFunc->getGnAtE(E).diagonal()/(2*M_PI)).real();
}

std::vector<Eigen::VectorXd> ElectronDensityCalc::getnEr(std::vector<numType> E) const
{
    assert(m_GFunc != nullptr);
    std::atomic_int finishCount = 0;
    std::vector<Eigen::VectorXd> individualSums;
    individualSums.resize(E.size());

    auto work = [BY_REF_CAPTURE(individualSums),
                 BY_REF_CAPTURE(finishCount),
                 CONST_REF_CAPTURE(E),
                 &me = std::as_const(*this)
    ](unsigned long startIndex, unsigned long endIndex)
    {
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            individualSums[idx] = me.getnEr(E[idx]);
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max(E.size()/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    Eigen::setNbThreads(1);
    for (unsigned long i = 0; i<E.size(); i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,E.size());
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex);}));
    }

    for (auto& fut : futures)
        fut.wait();
    Eigen::setNbThreads(0);
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
    return individualSums;
}

Eigen::VectorXd ElectronDensityCalc::getnEr(numType E, const std::string& basis) const
{
    assert(m_GFunc != nullptr);
    return (m_GFunc->getGnAtEInBasis(E,basis).diagonal()/(2*M_PI)).real();
}

std::vector<Eigen::VectorXd> ElectronDensityCalc::getnEr(std::vector<numType> E, const std::string& basis) const
{
    assert(m_GFunc != nullptr);
    std::atomic_int finishCount = 0;
    std::vector<Eigen::VectorXd> individualSums;
    individualSums.resize(E.size());

    auto work = [BY_REF_CAPTURE(individualSums),
                 BY_REF_CAPTURE(finishCount),
                 CONST_REF_CAPTURE(E),
                 CONST_REF_CAPTURE(basis),
                 &me = std::as_const(*this)
    ](unsigned long startIndex, unsigned long endIndex)
    {
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            individualSums[idx] = me.getnEr(E[idx],basis);
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max(E.size()/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    Eigen::setNbThreads(1);
    for (unsigned long i = 0; i<E.size(); i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,E.size());
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex);}));
    }

    for (auto& fut : futures)
        fut.wait();
    Eigen::setNbThreads(0);
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
    return individualSums;
}

// void ElectronDensityCalc::save(std::string filename) const
// {
//     logger logFile = logger(filename,true);
//     logFile.log("ElectronDensityCalc::Parameters\n{");
//     logFile.log("numberOfIntegrationPoints",m_params.numberOfIntegrationPoints);
//     logFile.log("startE",m_params.startE);
//     logFile.log("endE",m_params.endE);
//     logFile.log("}");
// }

// void ElectronDensityCalc::load(FILE *file)
// {
//     m_params.startE = 0;
//     m_params.endE = 0;
//     m_params.numberOfIntegrationPoints = 1;
//     while (true)
//     {
//         char objectName[30];
//         int ret = fscanf(file," %29[^{}:\n ] : ",objectName);
//         if (ret <= 0)
//             break;
//         std::string objectName_s(objectName);
//         if (objectName_s == "numberOfIntegrationPoints")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.numberOfIntegrationPoints);
//         }
//         else if (objectName_s == "startE")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.startE);
//         }
//         else if (objectName_s == "endE")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.endE);
//         }
//         else
//         {
//             logger().log("Unknown ElectronDensityCalc parameter: ",objectName_s);
//         }
//     }
// }

void SpectralFunction::asyncAEr(std::vector<Eigen::VectorXd> &individualSums, numType &step) const
{
    std::atomic_int finishCount = 0;
    individualSums.resize(m_params.numberOfIntegrationPoints);
    step = (m_params.endE - m_params.startE)/m_params.numberOfIntegrationPoints;
    if (step < 1e-8)
        logger().log("WARN: Step size is small:", step);

    auto work = [BY_REF_CAPTURE(individualSums),
                 BY_REF_CAPTURE(finishCount),
                 CONST_REF_CAPTURE(m_params),
                 BY_VAL_CAPTURE(step),
                 &me = std::as_const(*this)
    ](unsigned long startIndex, unsigned long endIndex)
    {
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            individualSums[idx] = me.getAEr(m_params.startE + idx*step);
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max(m_params.numberOfIntegrationPoints/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    Eigen::setNbThreads(1);
    for (int i = 0; i<m_params.numberOfIntegrationPoints; i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,m_params.numberOfIntegrationPoints);
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex);}));
    }

    for (auto& fut : futures)
        fut.wait();
    Eigen::setNbThreads(0);
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
}

void SpectralFunction::asyncAEr(std::vector<Eigen::VectorXd> &individualSums, numType &step, const std::string& basis) const
{
    std::atomic_int finishCount = 0;
    individualSums.resize(m_params.numberOfIntegrationPoints);
    step = (m_params.endE - m_params.startE)/m_params.numberOfIntegrationPoints;
    if (step < 1e-8)
        logger().log("WARN: Step size is small:", step);

    auto work = [BY_REF_CAPTURE(individualSums),
                 BY_REF_CAPTURE(finishCount),
                 CONST_REF_CAPTURE(m_params),
                 CONST_REF_CAPTURE(basis),
                 BY_VAL_CAPTURE(step),
                 &me = std::as_const(*this)
    ](unsigned long startIndex, unsigned long endIndex)
    {
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            individualSums[idx] = me.getAEr(m_params.startE + idx*step,basis);
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max(m_params.numberOfIntegrationPoints/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    Eigen::setNbThreads(1);
    for (int i = 0; i<m_params.numberOfIntegrationPoints; i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,m_params.numberOfIntegrationPoints);
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex);}));
    }

    for (auto& fut : futures)
        fut.wait();
    Eigen::setNbThreads(0);
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
}

Eigen::VectorXd SpectralFunction::getAR() const
{
    assert(m_GFunc != nullptr);


    std::vector<complexType> poles;
    m_GFunc->getEstimateForDivergencesOfGr(0,poles);

    long dim = m_GFunc->getDimension();
    Eigen::VectorXd sum;
    sum.setZero(dim);

    logger().log("A Integration, Number Of points", m_params.numberOfIntegrationPoints);
    SpectralWeightedIntegral(m_params.numberOfIntegrationPoints,sum,poles,
                             [&me = std::as_const(*this)](numType E, Eigen::VectorXd& dest){dest = me.getAEr(E);},
                             [dim](){    Eigen::VectorXd t;    t.setZero(dim); return t;});
    //TrapezoidalIntegral(m_params.startE,m_params.endE,m_params.numberOfIntegrationPoints,sum,[&](numType E, Eigen::VectorXd& dest){dest = getAEr(E);},[dim](){    Eigen::VectorXd t;    t.setZero(dim); return t;});
    return sum;
}

Eigen::VectorXd SpectralFunction::getAR(const std::string& basis) const
{
    assert(m_GFunc != nullptr);


    std::vector<complexType> poles;
    m_GFunc->getEstimateForDivergencesOfGr(0,poles);

    long dim = m_GFunc->getDimension();
    Eigen::VectorXd sum;
    sum.setZero(dim);

    logger().log("A Integration, Number Of points", m_params.numberOfIntegrationPoints);
    SpectralWeightedIntegral(m_params.numberOfIntegrationPoints,sum,poles,
                             [&me = std::as_const(*this),CONST_REF_CAPTURE(basis)](numType E, Eigen::VectorXd& dest){dest = me.getAEr(E,basis);},
                             [dim](){    Eigen::VectorXd t;    t.setZero(dim); return t;});
    //TrapezoidalIntegral(m_params.startE,m_params.endE,m_params.numberOfIntegrationPoints,sum,[&](numType E, Eigen::VectorXd& dest){dest = getAEr(E,basisChange);},[dim](){    Eigen::VectorXd t;    t.setZero(dim); return t;});
    return sum;
}

Eigen::VectorXd SpectralFunction::getAEr(numType E) const
{
    assert(m_GFunc != nullptr);
    ComplexMatrix GR;
    ComplexMatrix GA;
    m_GFunc->getGrAtE(E,GR);
    //In general see below
    // GA = GR.adjoint();
    m_GFunc->getGaAtEFromGr(E,GR,GA);
    // A = \iu [Gr - GA]
    GR -= GA;
    GR *= iu;
    Eigen::VectorXd A = GR.diagonal().real();
    return A;

    // Alternatively
    // ComplexMatrixBufferObject Gn = ComplexMatrixBuffer::getInstance().getAMatrix();
    // ComplexMatrixBufferObject Gp = ComplexMatrixBuffer::getInstance().getAMatrix();
    // m_GFunc->getGnAtE(E,Gn);
    // m_GFunc->getGpAtE(E,Gp);
    // Eigen::VectorXd A2 = (Gn.get() + Gp.get()).diagonal().real();
    // return A2;

}

Eigen::VectorXd SpectralFunction::getAEr(numType E,const std::string& basis) const
{
    assert(m_GFunc != nullptr);
    ComplexMatrix GR;
    ComplexMatrix GA;
    m_GFunc->getGrAtEInBasis(E,GR,basis);
    //In general see below
    GA = GR.adjoint();
    // A = \iu [Gr - GA]
    GR -= GA;
    GR *= iu;
    Eigen::VectorXd A = GR.diagonal().real();
    return A;

    // Alternatively
    // ComplexMatrixBufferObject Gn = ComplexMatrixBuffer::getInstance().getAMatrix();
    // ComplexMatrixBufferObject Gp = ComplexMatrixBuffer::getInstance().getAMatrix();
    // m_GFunc->getGnAtEInBasis(E,Gn,basisChange);
    // m_GFunc->getGpAtEInBasis(E,Gp,basisChange);
    // Eigen::VectorXd A2 = (Gn.get() + Gp.get()).diagonal().real();
    // return A2;
}

std::vector<Eigen::VectorXd> SpectralFunction::getAEr(std::vector<numType> E) const
{
    assert(m_GFunc != nullptr);
    std::atomic_int finishCount = 0;
    std::vector<Eigen::VectorXd> individualSums;
    individualSums.resize(E.size());

    auto work = [BY_REF_CAPTURE(individualSums),
                 BY_REF_CAPTURE(finishCount),
                 CONST_REF_CAPTURE(E),
                     &me = std::as_const(*this)
    ](unsigned long startIndex, unsigned long endIndex)
    {
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            individualSums[idx] = me.getAEr(E[idx]);
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max(E.size()/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    Eigen::setNbThreads(1);
    for (unsigned long i = 0; i<E.size(); i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,E.size());
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex);}));
    }

    for (auto& fut : futures)
        fut.wait();
    Eigen::setNbThreads(0);
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
    return individualSums;
}

std::vector<Eigen::VectorXd> SpectralFunction::getAEr(std::vector<numType> E, const std::string& basis) const
{
    assert(m_GFunc != nullptr);
    std::atomic_int finishCount = 0;
    std::vector<Eigen::VectorXd> individualSums;
    individualSums.resize(E.size());

    auto work = [BY_REF_CAPTURE(individualSums),
                 BY_REF_CAPTURE(finishCount),
                 CONST_REF_CAPTURE(E),
                 CONST_REF_CAPTURE(basis),
                 &me = std::as_const(*this)
    ](unsigned long startIndex, unsigned long endIndex)
    {
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            individualSums[idx] = me.getAEr(E[idx],basis);
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max(E.size()/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    Eigen::setNbThreads(1);
    for (unsigned long i = 0; i<E.size(); i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,E.size());
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex);}));
    }

    for (auto& fut : futures)
        fut.wait();
    Eigen::setNbThreads(0);
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
    return individualSums;
}

// void SpectralFunction::save(std::string filename) const
// {
//     logger logFile = logger(filename,true);
//     logFile.log("SpectralFunction::Parameters\n{");
//     logFile.log("numberOfIntegrationPoints",m_params.numberOfIntegrationPoints);
//     logFile.log("startE",m_params.startE);
//     logFile.log("endE",m_params.endE);
//     logFile.log("}");
// }

// void SpectralFunction::load(FILE *file)
// {
//     m_params.startE = 0;
//     m_params.endE = 0;
//     m_params.numberOfIntegrationPoints = 1;
//     while (true)
//     {
//         char objectName[30];
//         int ret = fscanf(file," %29[^{}:\n ] : ",objectName);
//         if (ret <= 0)
//             break;
//         std::string objectName_s(objectName);
//         if (objectName_s == "numberOfIntegrationPoints")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.numberOfIntegrationPoints);
//         }
//         else if (objectName_s == "startE")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.startE);
//         }
//         else if (objectName_s == "endE")
//         {
//             ConfigFileLoader::loadParameter(file,m_params.endE);
//         }
//         else
//         {
//             logger().log("Unknown SpectralFunction parameter: ",objectName_s);
//         }
//     }
// }

complexType OperatorExpecationValue::getElecValAtE(numType E, const ComplexMatrix &op) const
{
    assert(m_GFunc != nullptr);
    ComplexSelfAdjointMatrix gn;
    m_GFunc->getGnAtE(E,gn);
    return (gn * op).trace();
}

complexType OperatorExpecationValue::getHoleValAtE(numType E, const ComplexMatrix &op) const
{
    assert(m_GFunc != nullptr);
    ComplexSelfAdjointMatrix gp;
    m_GFunc->getGpAtE(E,gp);
    return (gp * op).trace();
}

std::vector<complexType> OperatorExpecationValue::getElecValE(std::vector<numType> E, const ComplexMatrix &op) const
{
    std::vector<complexType> ret;
    ret.reserve(E.size());
    for (size_t i = 0; i < E.size(); i++)
    {
        ret.push_back(getElecValAtE(E[i],op));
    }
    return ret;
}

std::vector<complexType> OperatorExpecationValue::getHoleValE(std::vector<numType> E, const ComplexMatrix &op) const
{
    std::vector<complexType> ret;
    ret.reserve(E.size());
    for (size_t i = 0; i < E.size(); i++)
    {
        ret.push_back(getHoleValAtE(E[i],op));
    }
    return ret;
}

complexType TransmissionFunc::getTransmissionAtEBetween(numType E, std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix> > p, std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix> > q) const
{
    ComplexMatrix gr;
    ComplexMatrix ga;

    assert(m_GFunc != nullptr);
    m_GFunc->getGrAtE(E,gr);
    ga = gr.adjoint();

    SelfEnergyMatrix GammaP;
    SelfEnergyMatrix temp;
    p->getSigmaR(GammaP);
    p->getSigmaA(temp);
    GammaP -= temp;
    GammaP *= iu;

    SelfEnergyMatrix GammaQ;
    q->getSigmaR(GammaQ);
    q->getSigmaA(temp);
    GammaQ -= temp;
    GammaQ *= iu;

    return (GammaP *  gr * GammaQ * ga).trace();



}

DivCurrentCalc::DivCurrentCalc(Parameters params)
{
    m_params = params;
    m_energies.resize(m_params.numberOfIntegrationPoints);
    numType step = (m_params.endE-m_params.startE)/m_params.numberOfIntegrationPoints;
    for (int i = 0; i < m_params.numberOfIntegrationPoints; i++)
    {
        m_energies[i] = i*step + m_params.startE;
    }
}

void DivCurrentCalc::getAtE(numType E, Eigen::VectorXd& dest) const
{
    assert(m_GFunc != nullptr);
    const ComplexSelfAdjointSparseMatrix& ham = m_GFunc->getHamiltonian();
    enums::SpinSymmetry sym = m_GFunc->getSpinSym();
    const std::string& basis = m_GFunc->getBasis();
    SelfEnergyMatrix SigmaR;
    SigmaR.setZero(ham.rows(),ham.cols(),basis,sym);

    SelfEnergyMatrix SigmaK;
    SigmaK.setZero(ham.rows(),ham.cols(),basis,sym);

    SelfEnergyMatrix SigmaA;
    SigmaA.setZero(ham.rows(),ham.cols(),basis,sym);

    SelfEnergyMatrix temp;
    for (auto& SE : m_selfEnergies)
    {
        SE->getSigmaR(E,temp);
        SigmaR += temp;
        SE->getSigmaK(E,temp);
        SigmaK += temp;
        SE->getSigmaA(E,temp);
        SigmaA += temp;
    }
    // SigmaA = SigmaR.adjoint();

    ComplexMatrix GR;
    ComplexMatrix GK;
    ComplexMatrix GA;
    // m_GFunc->getGrAtE(E,GR,true);// This is like solving for the Green's function by some method and then calculating a perturbative current for all self-energy terms. You lose all nice relations therefore
    // m_GFunc->getGkAtE(E,GK,true);
    // adjoint(GR,GA);
    m_GFunc->getGkGrGaAtE(E,GK,GR,GA,true);


    dest.resize(ham.rows());
    dest.setZero();
    //ret += (ham * GK.get()).diagonal();
    //ret -= (GK.get() * ham).diagonal();

    dest += MulDiagonal(SigmaR, GK).real();
    dest += MulDiagonal(SigmaK, GA).real();
    dest -= MulDiagonal(GK, SigmaA).real();
    dest -= MulDiagonal(GR,  SigmaK).real();

    // assert(dest.imag().norm() < 1e-10);

    dest *= (-e/(4*M_PI));
}

std::vector<Eigen::VectorXd> DivCurrentCalc::getAtE(std::vector<numType> Energies) const
{
    std::atomic_int finishCount = 0;
    std::vector<Eigen::VectorXd> individualSums(Energies.size());

    auto work = [BY_REF_CAPTURE(individualSums),
                 BY_REF_CAPTURE(finishCount),
                 CONST_REF_CAPTURE(Energies),
                 &me = std::as_const(*this)
    ](unsigned long startIndex, unsigned long endIndex)
    {
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            me.getAtE(Energies[idx],individualSums[idx]);
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max(Energies.size()/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    Eigen::setNbThreads(1);
    for (unsigned long i = 0; i<Energies.size(); i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,Energies.size());
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex);}));
    }

    for (auto& fut : futures)
        fut.wait();
    Eigen::setNbThreads(0);
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
    return individualSums;
}

Eigen::VectorXd DivCurrentCalc::get() const
{
    if (m_selfEnergies.size() == 0)
        return Eigen::VectorXd().setZero();

    std::vector<complexType> poles;
    m_GFunc->getEstimateForDivergencesOfGr(0,poles);

    Eigen::VectorXd sum;
    long dim = m_GFunc->getDimension();
    //Gauleg?
    logger().log("Integrating Current between",std::vector<numType>({m_params.startE,m_params.endE}));
    SpectralWeightedIntegral<false>(m_params.startE,m_params.endE,m_params.numberOfIntegrationPoints,sum,poles,
                                    [&me = std::as_const(*this)](numType E, Eigen::VectorXd& dest){me.getAtE(E,dest);},[dim](){return Eigen::VectorXd(dim).setZero();});
    //TrapezoidalIntegral(m_params.startE,m_params.endE,m_params.numberOfIntegrationPoints,sum,[&](numType E, Eigen::VectorXd& dest){dest = getAtE(E);},[dim](){Eigen::VectorXd(dim).setZero();});
    return sum;

}

bool DivCurrentCalc::isConstant() const
{
    bool ret = true;
    for (auto& SE : m_selfEnergies)
        ret = ret && (SE->m_type == selfEnergyType::ConstantSelfConsistent || SE->m_type == selfEnergyType::Constant);
    return ret;
}
numType ThermalEnergy::getAtE(numType E) const
{
    assert(m_GFunc != nullptr);
    //\frac{1}{4\pi} Tr(H G^n + E G^n)
    ComplexSelfAdjointMatrix Gn;
    m_GFunc->getGnAtE(E,Gn);
    //Gn = Projector * Gn * Projector^\dagger


    numType ret = E*(Gn.trace());
    ret += MulDiagonal(m_Ham, Gn).sum().real();//Technically the trace of a product of selfAdjoint matrices must be real. This is annoying! to detect though
    ret *= 1/(4*M_PI);
    return ret;
}
numType ThermalEnergy::getAtE(numType E, const std::string& basis) const
{
    assert(m_GFunc != nullptr);
    //\frac{1}{4\pi} Tr(H G^n + E G^n)
    ComplexSelfAdjointMatrix Gn;
    m_GFunc->getGnAtEInBasis(E,Gn,basis);
    //Gn = Projector * Gn * Projector^\dagger


    numType ret = E*(Gn.trace());
    ret += MulDiagonal(m_Ham, Gn).sum().real();//Technically the trace of a product of selfAdjoint matrices must be real. This is annoying! to detect though
    ret *= 1/(4*M_PI);
    return ret;
}

ThermalEnergy::ThermalEnergy(Parameters params)
{
    m_params = params;
    m_energies.resize(m_params.numberOfIntegrationPoints);
    numType step = (m_params.endE-m_params.startE)/m_params.numberOfIntegrationPoints;
    for (int i = 0; i < m_params.numberOfIntegrationPoints; i++)
    {
        m_energies[i] = i*step + m_params.startE;
    }
}
numType ThermalEnergy::getEnergy() const
{
    assert(m_GFunc != nullptr);

    ComplexSelfAdjointMatrix EGn;
    ComplexSelfAdjointMatrix Gn;

    m_GFunc->getEGnIntegral(m_params.numberOfIntegrationPoints,EGn,true);
    m_GFunc->getGnIntegral(m_params.numberOfIntegrationPoints,Gn,true);

    complexType ret = (EGn.trace());
    logger().log("Sum of MO Energies", ret/(2*M_PI));
    ret += (m_Ham * Gn).trace();
    ret *= 1/(4*M_PI);
    logger().log("Electronic energy", ret);
    logger().log("Extra energy", m_params.extraEnergy);
    logger().log("Total energy", ret.real() + m_params.extraEnergy);
    return ret.real() + m_params.extraEnergy;

    // std::vector<complexType> poles;
    // m_GFunc->getEstimateForDivergencesOfGr(0,poles);

    // numType sum;
    // SpectralWeightedIntegral(m_params.numberOfIntegrationPoints,sum,poles,[&](numType E, numType& dest){dest = getAtE(E);},[](){return 0;});
    //TrapezoidalIntegral(m_params.startE,m_params.endE,m_params.numberOfIntegrationPoints,sum,[&](numType E, numType& dest){dest = getAtE(E,Projector);},[](){return 0;});
    //return sum;
}

numType ThermalEnergy::getEnergy(const std::string& basis) const
{
    assert(m_GFunc != nullptr);

    // std::vector<complexType> poles;
    // m_GFunc->getEstimateForDivergencesOfGr(0,poles);

    // numType sum;
    // SpectralWeightedIntegral(m_params.numberOfIntegrationPoints,sum,poles,[&](numType E, numType& dest){dest = getAtE(E,Projector);},[](){return 0;});
    // //TrapezoidalIntegral(m_params.startE,m_params.endE,m_params.numberOfIntegrationPoints,sum,[&](numType E, numType& dest){dest = getAtE(E,Projector);},[](){return 0;});
    // return sum;
    ComplexSelfAdjointMatrix EGn;
    ComplexSelfAdjointMatrix Gn;

    m_GFunc->getEGnIntegral(m_params.numberOfIntegrationPoints,EGn,true);
    m_GFunc->getGnIntegral(m_params.numberOfIntegrationPoints,Gn,true);
    EGn.toBasisC(EGn,basis);
    Gn.toBasisC(Gn,basis);
    // EGn.get() = LHSProjector * EGn.get() * RHSProjector;
    // Gn.get() = LHSProjector * Gn.get() * RHSProjector;

    complexType ret = (EGn.trace());
    ret += (m_Ham * Gn).trace();
    ret *= 1/(4*M_PI);
    logger().log("XXXThermal energy", ret);
    return ret.real() + m_params.extraEnergy;
}

std::vector<numType> ThermalEnergy::getAtE(std::vector<numType> Energies, const std::string& basis) const
{
    std::atomic_int finishCount = 0;
    std::vector<numType> individualSums(Energies.size());

    auto work = [BY_REF_CAPTURE(individualSums),
                 BY_REF_CAPTURE(finishCount),
                 CONST_REF_CAPTURE(Energies),
                 CONST_REF_CAPTURE(basis),
                 &me = std::as_const(*this)
    ](unsigned long startIndex, unsigned long endIndex)
    {
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            individualSums[idx] = me.getAtE(Energies[idx], basis);
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max(Energies.size()/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    Eigen::setNbThreads(1);
    for (unsigned long i = 0; i<Energies.size(); i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,Energies.size());
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex);}));
    }

    for (auto& fut : futures)
        fut.wait();
    Eigen::setNbThreads(0);
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        logger().log("Wait returned but not all done?");

    std::atomic_thread_fence(std::memory_order_acquire);
    return individualSums;
}
