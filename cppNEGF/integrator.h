#ifndef INTEGRATOR_H
#define INTEGRATOR_H
#include "global.h"
#include "logger.h"
#include "threadpool.h"
#include <cassert>
#include <utility>
#include <vector>

template<typename weightFunc, typename cumWeightFunc>
inline void constructWeightFunctionPointsGauleg(numType start, numType end, unsigned long steps, std::vector<numType>& weights, std::vector<numType>& EnergyPoints,
                                                weightFunc weightFunction, cumWeightFunc cumWeightFunction, numType startGuess = 0);

template< bool preweighted, typename returnType, typename funcType, typename allocatorType>
inline void WeightedIntegral(const std::vector<numType>& weights, const std::vector<numType>& EnergyPoints, returnType& sum, numType chemicalPotential, funcType func,  allocatorType alloc, bool Parallelise = true);

/* Numerical recipes Third edition */
template<typename floatType = numType>
inline void gauleg(const numType x1, const numType x2, std::vector<numType> &x, std::vector<numType> &w)
    /*Given the lower and upper limits of integration x1 and x2, this routine returns arrays x[0..n-1]
        and w[0..n-1] of length n, containing the abscissas and weights of the Gauss-Legendre n-point
                                                                     quadrature formula.*/
{
    constexpr floatType EPS= std::is_same_v<floatType,long double> ? 1.0e-16 : 1e-14; //EPS is the relative precision.
    floatType z1,z,xm,xl,pp,p3,p2,p1;
    int n=x.size(); // n must be odd!
    int m=(n+1)/2;// The roots are symmetric in the interval, so we only have to find half of them.
    xm=0.5*(x2+x1);
    xl=0.5*(x2-x1);
    for (int i=0;i<m;i++)
    {// Loop over the desired roots.
        z=cos(M_PI*(i+0.75)/(n+0.5));
        // Starting with this approximation to the ith root, we enter the main loop of refinement by Newton’s method.
        do {
            p1=1.0;
            p2=0.0;
            for (int j=0;j<n;j++) {// Loop up the recurrence relation to get the Legendre polynomial evaluated at z.
                p3=p2;
                p2=p1;
                p1=((2.0*j+1.0)*z*p2-j*p3)/(j+1);
            }
            // p1 is now the desired Legendre polynomial. We next compute pp, its derivative,
            // by a standard relation involving also p2, the polynomial of one lower order.
            pp=n*(z*p1-p2)/(z*z-1.0);
            z1=z;
            z=z1-p1/pp;// Newton’s method.
        } while (abs(z-z1) > EPS);
        x[i]=xm-xl*z;// Scale the root to the desired interval,
            x[n-1-i]=xm+xl*z;// and put in its symmetric counterpart.
            w[i]=2.0*xl/((1.0-z*z)*pp*pp);// Compute the weight
            w[n-1-i]=w[i];// and its symmetric counterpart.
    }

}
template<typename returnType, typename funcType, typename allocatorType>
inline void GaussLegendreIntegral(numType lower, numType upper, unsigned long steps, returnType& sum, funcType func,  allocatorType alloc, bool parallelise = true)
{
    std::vector<numType> weights;
    std::vector<numType> energyPoints;
    if (steps %2 == 0)
        steps += 1;
    constructWeightFunctionPointsGauleg(lower,upper,steps,weights,energyPoints,[](numType){return 1.;},[](numType E){return E;});
    WeightedIntegral<true>(weights,energyPoints,sum,0,func,alloc,parallelise);
}

template<typename returnType, typename funcType, typename allocatorType>
inline void AdaptiveGaussLegendreIntegral(numType lower, numType upper, returnType& sum, numType& errorEstimate, funcType func,  allocatorType alloc,
                                          size_t recursionCount, const returnType& currentIntegralRegion, size_t minRecursion = 0,
                                                        bool parallelise = false, numType precision = 1e-15,size_t maxRecursion = 40)
{
    if (minRecursion > maxRecursion)
        minRecursion = maxRecursion;
    constexpr numType absError = 1e-17;
    assert(lower < upper);
    constexpr size_t pointsBig = 43;
    constexpr size_t pointsSmall = 21;
    static std::atomic_bool init = false;
    static std::mutex initMutex;
    static std::vector<numType> weightsBig(pointsBig);
    static std::vector<numType> energyPointsBig(pointsBig);

    static std::vector<numType> weightsSmall(pointsSmall);
    static std::vector<numType> energyPointsSmall(pointsSmall);
    if (!init)
    {
        std::lock_guard<std::mutex> lock(initMutex);
        if (!init)
        {
            gauleg<long double>(-1,1,energyPointsSmall,weightsSmall);
            gauleg<long double>(-1,1,energyPointsBig,weightsBig);
            init = true;
        }
    }

    std::vector<numType> scaledWeightsSmall(pointsSmall);
    std::vector<numType> scaledEnergyPointsSmall(pointsSmall);

    std::vector<numType> scaledWeightsBig(pointsBig);
    std::vector<numType> scaledEnergyPointsBig(pointsBig);

    numType domainMidpoint = (lower+upper)/2;
    numType domainSize = upper-lower;

    numType domainSizes[2];
    numType domainMidpoints[2];


    domainSizes[0] = (upper-lower)/2;
    domainSizes[1] = domainSizes[0];
    domainMidpoints[0] = (domainMidpoint + lower)/2;
    domainMidpoints[1] = (domainMidpoint + upper)/2;

    returnType partialSumSmall[2];
    returnType partialSumBig[2];


    if (recursionCount == 0)
    {
        errorEstimate = 0;
        for (size_t j = 0; j < pointsSmall; j++)
        {
            scaledWeightsSmall[j] = weightsSmall[j]*domainSize/2;
            scaledEnergyPointsSmall[j] = energyPointsSmall[j]*domainSize/2 + domainMidpoint;
        }
        for (size_t j = 0; j < pointsBig; j++)
        {
            scaledWeightsBig[j] = weightsBig[j]*domainSize/2;
            scaledEnergyPointsBig[j] = energyPointsBig[j]*domainSize/2 + domainMidpoint;
        }
        WeightedIntegral<true>(scaledWeightsSmall,scaledEnergyPointsSmall,partialSumSmall[0],0,func,alloc,parallelise);
        WeightedIntegral<true>(scaledWeightsBig,scaledEnergyPointsBig,partialSumBig[0],0,func,alloc,parallelise);

        sum = partialSumBig[0];
        if (abs(partialSumBig[0]-partialSumSmall[0]) < precision * abs(partialSumBig[0]))
        {
            errorEstimate = abs(partialSumBig[0]-partialSumSmall[0]);
            return;
        }
        else
        {
            sum = alloc();
            // currentIntegralRegion = alloc();
        }
    }




    for (int i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < pointsSmall; j++)
        {
            scaledWeightsSmall[j] = weightsSmall[j]*domainSizes[i]/2;
            scaledEnergyPointsSmall[j] = energyPointsSmall[j]*domainSizes[i]/2 + domainMidpoints[i];
        }
        for (size_t j = 0; j < pointsBig; j++)
        {
            scaledWeightsBig[j] = weightsBig[j]*domainSizes[i]/2;
            scaledEnergyPointsBig[j] = energyPointsBig[j]*domainSizes[i]/2 + domainMidpoints[i];
        }
        WeightedIntegral<true>(scaledWeightsSmall,scaledEnergyPointsSmall,partialSumSmall[i],0,func,alloc,parallelise);
        WeightedIntegral<true>(scaledWeightsBig,scaledEnergyPointsBig,partialSumBig[i],0,func,alloc,parallelise);
    }

    sum += (partialSumBig[0] + partialSumBig[1] - currentIntegralRegion);

    if (((abs(partialSumBig[0]-partialSumSmall[0]) > precision*abs(sum) && abs(partialSumBig[0]-partialSumSmall[0]) > absError && recursionCount < maxRecursion) || recursionCount < minRecursion) && domainSizes[0] > 1e-10)
    {
        // logger().log("Bisecting, RelativeError",abs(partialSumBig[0]-partialSumSmall[0])/abs(sum));
        // logger().log("Bisecting, AbsError",abs(partialSumBig[0]-partialSumSmall[0]));
        // logger().log("Bisecting, region",std::vector<numType>({lower,domainMidpoint}));
        AdaptiveGaussLegendreIntegral(lower,domainMidpoint,sum,errorEstimate,func,alloc,recursionCount+1,partialSumBig[0],minRecursion,parallelise,precision,maxRecursion);
    }
    else
    {
        errorEstimate += abs(partialSumBig[0]-partialSumSmall[0]);
    }
    if (((abs(partialSumBig[1]-partialSumSmall[1]) > precision*abs(sum) && abs(partialSumBig[1]-partialSumSmall[1]) > absError  && recursionCount < maxRecursion) || recursionCount < minRecursion) && domainSizes[1] > 1e-10)
    {
        // logger().log("Bisecting, RelativeError",abs(partialSumBig[1]-partialSumSmall[1])/abs(sum));
        // logger().log("Bisecting, AbsError",abs(partialSumBig[1]-partialSumSmall[1]));
        // logger().log("Bisecting, region",std::vector<numType>({domainMidpoint,upper}));
        AdaptiveGaussLegendreIntegral(domainMidpoint,upper,sum,errorEstimate,func,alloc,recursionCount+1,partialSumBig[1],minRecursion,parallelise,precision,maxRecursion);
    }
    else
    {
        errorEstimate += abs(partialSumBig[1]-partialSumSmall[1]);
    }


}
template<typename returnType, typename funcType, typename allocatorType>
inline void AdaptiveGaussLegendreIntegralSingularities(numType lower, numType upper, returnType& sum, funcType func,  allocatorType alloc,
                                          std::vector<complexType> singularities, size_t minRecursion = 0, bool parallelise = true, numType precision = 1e-15)
{
    bool toFlip = false;
    if (lower >= upper)
    {
        toFlip = true;
        std::swap(lower,upper);
    }

    std::vector<numType> projectedSingularities;
    std::vector<numType> distanceToSingularities;
    assert(lower < upper);

    std::sort(singularities.begin(),singularities.end(),[](complexType a, complexType b){return a.real() < b.real();});

    projectedSingularities.reserve(singularities.size());
    distanceToSingularities.reserve(singularities.size());
    for (auto& s : singularities)
    {
        projectedSingularities.push_back(s.real());
        distanceToSingularities.push_back(abs(s.imag()));
    }
    std::vector<numType> domains;
    domains.push_back(lower);

    for (size_t i = 0; i < projectedSingularities.size(); i++)
    {
        if (domains.back() >= projectedSingularities[i] - distanceToSingularities[i])
            ;//domains.back() = projectedSingularities[i] - distanceToSingularities[i];
        else
            domains.push_back(projectedSingularities[i] - distanceToSingularities[i]);
        domains.push_back(projectedSingularities[i] + distanceToSingularities[i]);
    }
    if (upper <= domains.back())
        domains.back() = upper;
    else
        domains.push_back(upper);

    assert(domains.size() >= 2);

    returnType partialSum;
    numType partialErrorEstimate;
    numType errorEstimate = 0;
    sum = alloc();
    for (size_t i = 1; i < domains.size(); i++)
    {
        partialSum = alloc();
        AdaptiveGaussLegendreIntegral(domains[i-1],domains[i],partialSum,partialErrorEstimate,func,alloc,0,alloc(),minRecursion,false,precision);
        sum += partialSum;
        // logger().log("Partial sum",partialSum);
        errorEstimate += partialErrorEstimate;
    }
    // logger().log("Integral error estimate:",errorEstimate);
    if (toFlip)
        sum *= -1.;


}
template<typename T>
void fastToSum(const T& a, const T& b, T& s, T& t)
{//computes s = a ~+ b and exact error t = a+b - s. ~+ is rounded sum
// see https://en.wikipedia.org/wiki/2Sum
    s = a + b;
    T aPrime = s - b;
    T bPrime = s - aPrime;
    T deltaA = a-aPrime;
    T deltaB = b-bPrime;
    t = deltaA + deltaB;
}

template<typename returnType, typename funcType, typename allocatorType, bool kahansum = false>
inline void TrapezoidalIntegral(numType lower, numType upper, unsigned long steps, returnType& sum, funcType func,  allocatorType alloc)
{
    numType IntegrationStepSize = (upper-lower)/steps; // note that we actually evaluate on steps+1 steps due to the end points

    sum = alloc();
    const int stepSize = std::max((steps)/NUM_CORES,1ul);
    std::vector<returnType> partialSums;
    partialSums.assign(steps/stepSize + (steps % stepSize == 0 ? 0 : 1 ),alloc());

    auto work = [BY_REF_CAPTURE(alloc),
                 BY_REF_CAPTURE(func),
                 BY_VAL_CAPTURE(IntegrationStepSize),
                 BY_VAL_CAPTURE(lower),
                 BY_REF_CAPTURE(partialSums)
    ](unsigned long startIndex, unsigned long endIndex, size_t workerIndex)
    {
        returnType WorkerSum = alloc();
        returnType WorkerComp = alloc();
        returnType workerTemp = alloc();
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            numType point = idx*IntegrationStepSize + lower;
            func(point,workerTemp);
            if constexpr(kahansum)
            {
                workerTemp += WorkerComp;

                returnType WorkerSumcopy = WorkerSum;
                fastToSum(WorkerSumcopy,workerTemp,WorkerSum,WorkerComp);
            }
            else
            {
                WorkerSum += workerTemp;
            }
        }
        WorkerSum *= IntegrationStepSize;
        partialSums[workerIndex] = std::move(WorkerSum);

    };


    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    size_t workerCount = 0;
    for (unsigned long i = 0; i < steps; i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,steps);
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex,workerCount);}));
        workerCount++;
    }
    releaseAssert(workerCount == partialSums.size(),"Trapezoidal worker calc error");
    for (auto& fut : futures)
        fut.wait();

    for (auto &s : partialSums)
        sum += s;


    returnType temp = alloc();
    func(upper,temp);
    sum += IntegrationStepSize*temp/2.;

    func(lower,temp);
    sum -= IntegrationStepSize*temp/2.;
}


template<typename funcType, typename derivFuncType>
inline numType newtonSolver(funcType func, derivFuncType derivFunc, long double value, long double guess)
{
    long double currValue = func(guess);
    long double step = 1;
    while (std::abs(currValue - value) > 1e-10 && step > 1e-12)
    {
        step = - (currValue - value)/derivFunc(guess);
        guess = guess + step;
        while (std::abs(func(guess) - value) > std::abs(currValue - value))
        {
            //This is a convergent geometric sum which takes us back to where we started in the infinite limit
            step /=2;
            guess -= step;
        }
        currValue = func(guess);
    }
    return guess;
}
// Branch cut in the lower half plane
inline numType argd(complexType z)
{   numType ret = std::atan(z.imag()/z.real());
    if (z.real() < 0)
        ret += M_PI;
    return ret;
};
//Branch cut in the upper half plane
inline numType argu(complexType z)
{   numType ret = std::atan(z.imag()/z.real()) + M_PI;
    if (z.real() > 0)
        ret += M_PI;
    return ret;
}

template<typename weightFunc, typename cumWeightFunc>
inline void constructWeightFunctionPoints(numType start, numType end, unsigned long steps, std::vector<numType>& weights, std::vector<numType>& EnergyPoints,
    weightFunc weightFunction, cumWeightFunc cumWeightFunction, numType startGuess = 0)
{//Sample uniformly with a weight function


    numType wEnd;
    numType wStart;
    wEnd = cumWeightFunction(end);
    wStart = cumWeightFunction(start);

    std::vector<numType> cumWeightFunctionPoints;
    cumWeightFunctionPoints.assign(steps,0);
    EnergyPoints.assign(steps,0);

    for (int i = 0; i < steps; i++)
    {
        cumWeightFunctionPoints[i] = ((numType)i/(numType)(steps-1))*(wEnd-wStart) + wStart;
    }
    weights.assign(steps,(wEnd-wStart)/(numType)(steps-1));
    weights[0] *= 0.5;
    weights[steps-1] *= 0.5;

    if (start == negInf)
    {
        EnergyPoints[0] = -1e10; // negative infinity
        EnergyPoints[1] = newtonSolver(cumWeightFunction,weightFunction,cumWeightFunctionPoints[1],startGuess);
    }
    else
    {
        EnergyPoints[0] = start;
        EnergyPoints[1] = newtonSolver(cumWeightFunction,weightFunction,cumWeightFunctionPoints[1],start);
    }


    for (int i = 2; i < steps-1; i++)
    {
        numType target = cumWeightFunctionPoints[i];
        EnergyPoints[i] = newtonSolver(cumWeightFunction,weightFunction,target,EnergyPoints[i-1]);
    }
    if (end == posInf)
    {
        EnergyPoints[steps-1] = 1e10; // positive infinity
    }
    else
    {
        EnergyPoints[steps-1] = end;
    }
}
template<typename weightFunc, typename cumWeightFunc>
inline void constructWeightFunctionPointsGauleg(numType start, numType end, unsigned long steps, std::vector<numType>& weights, std::vector<numType>& EnergyPoints,
                                          weightFunc weightFunction, cumWeightFunc cumWeightFunction, numType startGuess/* = 0*/)
{


    numType wEnd;
    numType wStart;
    wEnd = cumWeightFunction(end);
    wStart = cumWeightFunction(start);


    std::vector<numType> cumWeightFunctionPoints;
    const int maxSteps = 10000;
    if (steps > maxSteps)
    {
        int numRegions = ceil(steps/(numType)maxSteps);
        int stepsPerRegion = maxSteps;
        if (stepsPerRegion %2 == 0)
            stepsPerRegion += 1;

        numType weightStep = (wEnd-wStart)/numRegions;
        cumWeightFunctionPoints.clear();
        weights.clear();
        weights.reserve(numRegions*stepsPerRegion);
        cumWeightFunctionPoints.reserve(numRegions*stepsPerRegion);

        for (int i = 0; i <numRegions; i++)
        {
            std::vector<numType> tempCumWeightFunctionPoints(stepsPerRegion);
            std::vector<numType> tempWeights(stepsPerRegion);

            gauleg(wStart + i*weightStep,wStart + (i+1)*weightStep,tempCumWeightFunctionPoints,tempWeights);
            cumWeightFunctionPoints.insert(cumWeightFunctionPoints.end(),tempCumWeightFunctionPoints.begin(),tempCumWeightFunctionPoints.end());
            weights.insert(weights.end(),tempWeights.begin(),tempWeights.end());
        }
        EnergyPoints.assign(weights.size(),0);

    }
    else
    {
        weights.assign(steps,0);
        EnergyPoints.assign(steps,0);
        cumWeightFunctionPoints.resize(steps,0);
        gauleg(wStart,wEnd,cumWeightFunctionPoints,weights);
    }

    EnergyPoints[0] = newtonSolver(cumWeightFunction,weightFunction,cumWeightFunctionPoints[0], start == negInf? startGuess : start);

    for (int i = 1; i < EnergyPoints.size(); i++)
    {
        numType target = cumWeightFunctionPoints[i];
        EnergyPoints[i] = newtonSolver(cumWeightFunction,weightFunction,target,EnergyPoints[i-1]);
    }
}

inline void constructFermiWeightFunctionGauleg(numType start, numType end, unsigned long steps, numType chemicalPotential,
                                            std::vector<numType>& weights ,std::vector<numType>& EnergyPoints)
{
    auto cumWeightFunction =
        [mu=chemicalPotential](long double E)
    {
        return -(kb*T)*std::log(1+std::exp(-(E-mu)/(kb*T)));
    };
    auto weightFunction =
        [mu=chemicalPotential](long double E)
    {
        return 1./(std::exp((E-mu)/(kb*T))+1);
    };
    if (steps == 0)
    {//Automatically determine based on range
        long double upper = cumWeightFunction(end);
        long double lower = cumWeightFunction(start);
        steps = std::abs((upper - lower)/(1e-4)); // 1 Million is fine for around 20 poles i.e. a range of 120
        steps = std::max(steps, 10ul);
    }


    if (start == negInf)
    {
        logger().log("Cannot integrate from -inf for fermi function, Weight does not converge at -inf");
        start = -1000;
    }
    if (steps % 2 == 0)
        steps += 1;

    constructWeightFunctionPointsGauleg(start,end,steps,weights,EnergyPoints,weightFunction,cumWeightFunction);

}

//Utility for use in autoranging for plots. Not used in integrators
inline void constructSpectralWeightFunction(numType start, numType end, unsigned long steps, const std::vector<complexType>& poles,
                                            std::vector<numType>& cumWeightFunctionPoints ,std::vector<numType>& EnergyPoints)
{
    auto cumWeightFunction =
        [CONST_REF_CAPTURE(poles)](long double E)
    {
        long double sum = 0;
        for (auto p : poles)
        {
            sum += 2*std::atan(p.imag()/(E-p.real()));
            if (E-p.real() > 0)
                sum += 2*M_PI;
        }
        return sum;
    };
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
    if (steps == 0)
    {//Automatically determine based on range
        long double upper = cumWeightFunction(end);
        long double lower = cumWeightFunction(start);
        steps = std::abs((upper - lower)/(1e-4)); // 1 Million is fine for around 20 poles i.e. a range of 120
        steps = std::max(steps, 10ul);
    }

    if (start == negInf)
        constructWeightFunctionPoints(start,end,steps,cumWeightFunctionPoints,EnergyPoints,weightFunction,cumWeightFunction,-10000);
    else
        constructWeightFunctionPoints(start,end,steps,cumWeightFunctionPoints,EnergyPoints,weightFunction,cumWeightFunction);

}

template<typename returnType, typename funcType, typename allocatorType>
inline void DeltaFunctionIntegral(numType start, numType end, returnType& sum, const std::vector<numType>& points, funcType func,  allocatorType alloc, bool Parallelise = true)
{
    sum = alloc();

    size_t steps = points.size();
    const int stepSize = std::max(steps/NUM_CORES,1ul);
    std::vector<returnType> partialSums;
    partialSums.assign(steps/stepSize + (steps % stepSize != 0),alloc());

    auto work = [BY_REF_CAPTURE(partialSums),
                 BY_REF_CAPTURE(alloc),
                 BY_REF_CAPTURE(func),
                 CONST_REF_CAPTURE(points),
                 BY_VAL_CAPTURE(start),
                 BY_VAL_CAPTURE(end)
    ](unsigned long startIndex, unsigned long endIndex, size_t workerIndex)
    {
        returnType WorkerSum = alloc();
        returnType workerTemp = alloc();
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            numType point = points[idx];
            if (point < start || point > end)
                continue;
            func(point,workerTemp);
            if (point == start || point == end)
                WorkerSum += workerTemp*0.5;
            else
                WorkerSum += workerTemp;
        }
        partialSums[workerIndex] = std::move(WorkerSum);
    };


    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    size_t workerCount = 0;
    for (unsigned long i = 0; i < steps; i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,steps);
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex,workerCount);},!Parallelise));
        workerCount++;
    }

    releaseAssert(workerCount == partialSums.size(),"delta integral worker count wrong");
    for (auto& fut : futures)
        fut.wait();

    for (returnType& s: partialSums)
        sum += s;
}


template<bool gauleg, typename returnType, typename funcType, typename allocatorType>
inline void SpectralWeightedIntegral(numType start, numType end, unsigned long steps, returnType& sum, const std::vector<complexType>& polesIn, funcType func,  allocatorType alloc, bool Parallelise = true)
{
    std::vector<numType> weights;
    std::vector<numType> EnergyPoints;

    //Check if any poles are zero in which case \int f(E) \td E = \sum_p 2\pi \lim_{E->p} f(E)*(E-p)
    std::vector<complexType> poles;
    std::vector<numType> deltaPoles;
    for (auto p : polesIn)
    {
        if (abs(p.imag()) < 1e-14)
        {
            deltaPoles.push_back(p.real());
        }
        else
        {
            poles.push_back(p);
        }
    }
    sum = alloc();
    if (deltaPoles.size() > 0)
    {
        DeltaFunctionIntegral(start,end,sum,deltaPoles,[CONST_REF_CAPTURE(func)](numType E, returnType& dest){func(E+1e-8,dest); dest *= 1e-8;},alloc,Parallelise);
        sum *= 2*M_PI;
    }
    if (poles.size() == 0)
    {
        return;
    }


    auto cumWeightFunction =
        [CONST_REF_CAPTURE(poles)](long double E)
    {
        long double sum = 0;
        for (auto p : poles)
        {
            sum += 2*std::atan(p.imag()/(E-p.real()));
            if (E-p.real() > 0)
                sum += 2*M_PI;
        }
        return sum;
    };
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
    if (steps == 0 && gauleg)
    {//Automatically determine based on range
        long double upper = cumWeightFunction(end);
        long double lower = cumWeightFunction(start);
        steps = std::abs((upper - lower)*100.); // 5000 points for a weight of 75 is fine for a general integration but it gets the fermi-function `wrong'
        steps = std::max(steps, 10ul);
    }
    else if (steps == 0 && !gauleg)
    {
        long double upper = cumWeightFunction(end);
        long double lower = cumWeightFunction(start);
        steps = std::abs((upper - lower)*1e4); // 1million points per 100 weight is fine
        steps = std::max(steps, 10ul);
    }
    if (steps < 10)
        steps = 10;
    if (steps % 2 == 0)
        steps += 1;

   /*Constructs a set of points E where w(E) is spaced according to the gauss-legendre quadrature
    * w'(E) = \iu \sum_m 1/(E-poles[m]) - 1/(E-poles(m)^*)
    * w(E) &= \sum_{m} \begin{cases}
    * 2\arctan(\frac{\Gamma_m}{E - \varepsilon^r_m})  & \t{Re}(E - \varepsilon^r_m) < 0\\
    * 2\arctan(\frac{\Gamma_m}{E - \varepsilon^r_m}) + 2\pi & \t{Re}(E - \varepsilon^r_m) > 0\\
    * \end{cases}
    * Where \Gamma_m = Im(poles[m]) < 0, \varepsilon^r_m = Re(poles[m])
    *
    */
    if (gauleg)
    {
        if (start == negInf)
            constructWeightFunctionPointsGauleg(start,end,steps,weights,EnergyPoints,weightFunction,cumWeightFunction,-10000);
        else
            constructWeightFunctionPointsGauleg(start,end,steps,weights,EnergyPoints,weightFunction,cumWeightFunction);
    }
    else
    {
        if (start == negInf)
            constructWeightFunctionPoints(start,end,steps,weights,EnergyPoints,weightFunction,cumWeightFunction,-10000);
        else
            constructWeightFunctionPoints(start,end,steps,weights,EnergyPoints,weightFunction,cumWeightFunction);
    }
    steps = weights.size();

    const int stepSize = std::max(steps/NUM_CORES,1ul);
    std::vector<returnType> partialSums;
    partialSums.assign(steps/stepSize + (steps % stepSize != 0),alloc());

    auto work = [BY_REF_CAPTURE(partialSums),
                 BY_REF_CAPTURE(alloc),
                 BY_REF_CAPTURE(func),
                 CONST_REF_CAPTURE(EnergyPoints),
                 CONST_REF_CAPTURE(weights),
                 CONST_REF_CAPTURE(weightFunction),
                 BY_VAL_CAPTURE(start),
                 BY_VAL_CAPTURE(end)](unsigned long startIndex, unsigned long endIndex,size_t workerIndex)
    {
        returnType WorkerSum = alloc();
        returnType workerTemp = alloc();
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            numType point = EnergyPoints[idx];
            func(point,workerTemp);
            WorkerSum += weights[idx]*(workerTemp/(double)weightFunction(point));
        }
        partialSums[workerIndex] = std::move(WorkerSum);
    };


    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    size_t workerCount = 0;
    for (unsigned long i = 0; i < steps; i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,steps);
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex,workerCount);},!Parallelise));
        workerCount++;
    }
    releaseAssert(workerCount == partialSums.size(),"Spectral weighted integral worker count wrong");
    for (auto& fut : futures)
        fut.wait();

    for (auto& s: partialSums)
        sum += s;

}
//Overload with infinite limits
template<bool gauleg=false, typename returnType, typename funcType, typename allocatorType>
inline void SpectralWeightedIntegral(unsigned long steps, returnType& sum, const std::vector<complexType>& poles, funcType func,  allocatorType alloc)
{
    SpectralWeightedIntegral<gauleg>(negInf,posInf,steps,sum,poles,func,alloc);
}

template< bool preweighted, typename returnType, typename funcType, typename allocatorType>
inline void FermiWeightedIntegral(numType start, numType end, unsigned long steps, returnType& sum, numType chemicalPotential, funcType func,  allocatorType alloc, bool Parallelise = true)
{
    std::vector<numType> weights;
    std::vector<numType> EnergyPoints;
    auto cumWeightFunction =
        [mu=chemicalPotential](long double E)
    {
            return -(kb*T)*std::log(1+std::exp(-(E-mu)/(kb*T)));
    };
    auto weightFunction =
        [mu=chemicalPotential](long double E)
    {
            return 1./(std::exp((E-mu)/(kb*T))+1);
    };
    if (steps == 0)
    {//Automatically determine based on range
        long double upper = cumWeightFunction(end);
        long double lower = cumWeightFunction(start);
        steps = std::abs((upper - lower)/(1e-4)); // 1 Million is fine for around 20 poles i.e. a range of 120
        steps = std::max(steps, 10ul);
    }


    if (start == negInf)
    {
        logger().log("Cannot integrate from -inf for fermi function, Weight does not converge at -inf");
        start = -1000;
    }
    if (steps % 2 == 0)
        steps += 1;
    constructWeightFunctionPointsGauleg(start,end,steps,weights,EnergyPoints,weightFunction,cumWeightFunction);
    steps = weights.size();
    std::mutex resultMutex;
    sum = alloc();

    const int stepSize = std::max(steps/NUM_CORES,1ul);
    std::vector<returnType> partialSums;
    partialSums.assign(steps/stepSize + (steps % stepSize != 0),alloc());

    auto work = [BY_REF_CAPTURE(partialSums),
                 BY_REF_CAPTURE(alloc),
                 BY_REF_CAPTURE(func),
                 CONST_REF_CAPTURE(EnergyPoints),
                 CONST_REF_CAPTURE(weights),
                 CONST_REF_CAPTURE(weightFunction),
                 BY_VAL_CAPTURE(start),
                 BY_VAL_CAPTURE(end)
    ](unsigned long startIndex, unsigned long endIndex, size_t workerIndex)
    {
        returnType WorkerSum = alloc();
        returnType workerTemp = alloc();
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            numType point = EnergyPoints[idx];
            func(point,workerTemp);
            if (preweighted)
                WorkerSum += weights[idx]*workerTemp;
            else
                WorkerSum += weights[idx]*(workerTemp/(numType)weightFunction(point));
        }
        partialSums[workerIndex] = std::move(WorkerSum);
    };


    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    size_t workerCount = 0;
    for (unsigned long i = 0; i < steps; i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,steps);
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex,workerCount);},!Parallelise));
        workerCount++;
    }
    releaseAssert(workerCount == partialSums.size(),"Spectral weighted integral worker count wrong");
    for (auto& fut : futures)
        fut.wait();
    for (auto& s: partialSums)
        sum += s;
}


template< bool preweighted, typename returnType, typename funcType, typename allocatorType>
inline void WeightedIntegral(const std::vector<numType>& weights, const std::vector<numType>& EnergyPoints, returnType& sum, numType /*todoRemove*/, funcType func,  allocatorType alloc, bool Parallelise/* = true*/)
{
    unsigned long steps = weights.size();
    std::mutex resultMutex;
    sum = alloc();
    const int stepSize = std::max(steps/NUM_CORES,1ul);
    std::vector<returnType> partialSums;
    partialSums.assign(steps/stepSize + (steps % stepSize != 0),alloc());

    auto work = [BY_REF_CAPTURE(partialSums),
                 BY_REF_CAPTURE(alloc),
                 BY_REF_CAPTURE(func),
                 CONST_REF_CAPTURE(EnergyPoints),
                 CONST_REF_CAPTURE(weights)
    ](unsigned long startIndex, unsigned long endIndex, size_t workerIndex)
    {
        returnType WorkerSum = alloc();
        returnType workerTemp = alloc();
        for (unsigned long idx = startIndex; idx < endIndex; idx++)
        {
            numType point = EnergyPoints[idx];
            func(point,workerTemp);
            if (preweighted)
                WorkerSum += weights[idx]*workerTemp;
            else
                static_assert(preweighted, "must be preweighted");
                // WorkerSum += weights[idx]*(workerTemp/(numType)weightFunction(point));
        }
        partialSums[workerIndex] = std::move(WorkerSum);
    };


    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    size_t workerCount = 0;
    for (unsigned long i = 0; i < steps; i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,steps);
        futures.push_back(pool.queueWork([=,&work](){work(i,endIndex,workerCount);},!Parallelise));
        workerCount++;
    }
    releaseAssert(workerCount == partialSums.size(),"Spectral weighted integral worker count wrong");
    for (auto& fut : futures)
        fut.wait();

    for (auto& s: partialSums)
        sum += s;
}

/* Integrate along the arc with radius R, midpoint m, start angle Theta0, EndAngle Theta1 with steps steps.
 * I = \sum_i f( R e^\iu \theta) \iu R e^\iu \theta \delta \theta
 */

template<typename returnType, typename funcType, typename allocatorType>
inline void ArcIntegral(unsigned long steps, numType R, numType Theta0, numType Theta1, returnType& sum, funcType func,  allocatorType alloc)
{
    auto integralFunc = [CONST_REF_CAPTURE(func),BY_VAL_CAPTURE(R)](numType Theta, returnType& dest)
    {
        complexType z = R*std::exp(iu*Theta);
        func(z,dest);
        dest *= iu*z;
    };
    GaussLegendreIntegral(Theta0,Theta1,steps,sum,integralFunc,alloc);
}

#endif // INTEGRATOR_H
