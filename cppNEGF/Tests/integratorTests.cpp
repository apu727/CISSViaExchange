#ifndef FOO_CPP
#define FOO_CPP
#include "integrator.h"
#include "testutil.hpp"
constexpr bool logErrors = true;

void niceFunction(numType E, numType& dest)
{
    dest = E*E;
}

void niceFunctionC(complexType E, complexType& dest)
{
    dest = E*E;
}

numType niceFunctionAntiDerivative(numType E)
{
    return pow(E,3)/3.;
}

void peakedFunction(numType E, numType& dest)
{
    dest = ((1./(E-1e-6*iu)) - (1./(E+1e-6*iu))).imag();
}

void peakedFunctionf(float E, float& dest)
{
    dest = ((1.f/(E-1e-6f*std::complex<float>(0,1))) - (1.f/(E+1e-6f*std::complex<float>(0,1)))).imag();
}

void peakedFunctionC(complexType E, complexType& dest)
{
    dest = (1./(E-1e-6*iu));
}

numType peakedFunctionAntiDerivative(numType E)
{
    return (argu(E-1e-6*iu) -argd(E+1e-6*iu));
}

void peakedComplexFunction(numType E, complexType& dest)
{
    dest = 1./((E-1e-6*iu)*(E-1e-6*iu));
}

void peakedComplexFunctionC(complexType E, complexType& dest)
{
    dest = 1./((E-1e-6*iu)*(E-1e-6*iu));
}

complexType peakedComplexFunctionAntiDerivative(numType E)
{
    return -1./(E-1e-6*iu);
}

complexType complexAlloc()
{
    return 0;
}
numType realAlloc()
{
    return 0;
}

std::vector<complexType> pseudoPoles({-1e-6*iu});

static numType pseudoFermiFunction(numType E)
{
    if (E/(kb*300) > 30)
        return 0;
    else
        return 1.0/(exp((E)/(kb*300)) + 1);
}

bool defaultNear(const numType& A, const numType& B)
{
    return abs((A-B)/B) < 1e-8 || abs(A-B) < 1e-8;
}

bool defaultNearC(const complexType& A, const complexType& B)
{
    return abs((A-B)/B) < 1e-8 || abs(A-B) < 1e-8;
}

int testNiceFunction()
{
    numType realDest;
    complexType complexDest;
    numType realAnswer;

    int success = 0;

    realAnswer = niceFunctionAntiDerivative(10) - niceFunctionAntiDerivative(-10);

    GaussLegendreIntegral(-10,10,100,realDest,niceFunction,realAlloc);
    success += expectNear(realAnswer,realDest,defaultNear,__PRETTY_FUNCTION__,"GaussLegendre Failed");
    if (logErrors) logger().log("GaussLegendre " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - realAnswer);

    AdaptiveGaussLegendreIntegralSingularities(-10,10,realDest,niceFunction,realAlloc,{});
    success += expectNear(realAnswer,realDest,defaultNear,__PRETTY_FUNCTION__,"AdaptiveGaussLegendreIntegral Failed");
    if (logErrors) logger().log("AdaptiveGaussLegendreIntegral " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - realAnswer);

    TrapezoidalIntegral(-10,10,100000,realDest,niceFunction,realAlloc);
    success += expectNear(realAnswer,realDest,defaultNear,__PRETTY_FUNCTION__,"TrapezoidalIntegral Failed");
    if (logErrors) logger().log("TrapezoidalIntegral " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - realAnswer);

    DeltaFunctionIntegral(-10,10,realDest,std::vector<numType>({-10,0,9}),niceFunction,realAlloc);
    success += expect((numType)(10*10*0.5 + 9*9),realDest,__PRETTY_FUNCTION__,"DeltaFunctionIntegral Failed");

    DeltaFunctionIntegral(-10,10,realDest,std::vector<numType>({-20}),niceFunction,realAlloc);
    success += expect(0.,realDest,__PRETTY_FUNCTION__,"DeltaFunctionIntegral2 Failed");

    SpectralWeightedIntegral<false>(-10,10,1000000,realDest,std::vector<complexType> ({-1.*iu}),niceFunction,realAlloc);
    success += expectNear(realAnswer,realDest,defaultNear,__PRETTY_FUNCTION__,"SpectralWeightedIntegral<false> Failed");
    if (logErrors) logger().log("SpectralWeightedIntegral<false> " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - realAnswer);

    SpectralWeightedIntegral<true>(-10,10,1000,realDest,std::vector<complexType> ({-1.*iu}),niceFunction,realAlloc);
    success += expectNear(realAnswer,realDest,defaultNear,__PRETTY_FUNCTION__,"SpectralWeightedIntegral<true> Failed");
    if (logErrors) logger().log("SpectralWeightedIntegral<true> " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - realAnswer);

    FermiWeightedIntegral<false>(-10,10,100,realDest,0,[](numType E, numType& dest) {niceFunction(E,dest);dest *= pseudoFermiFunction(E);},realAlloc);
    success += expectNear(1000./3.,realDest,defaultNear,__PRETTY_FUNCTION__,"FermiWeightedIntegral<false> Failed");
    if (logErrors) logger().log("FermiWeightedIntegral<false> " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - 1000./3.);

    FermiWeightedIntegral<true>(-10,10,100,realDest,0,[](numType E, numType& dest) {niceFunction(E,dest);},realAlloc);
    success += expectNear(1000./3.,realDest,defaultNear,__PRETTY_FUNCTION__,"FermiWeightedIntegral<true> Failed");
    if (logErrors) logger().log("FermiWeightedIntegral<true> " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - 1000./3.);

    std::vector<numType> weights;
    std::vector<numType> EnergyPoints;
    constructFermiWeightFunctionGauleg(-10,10,100,0,weights,EnergyPoints);

    // WeightedIntegral<false>(weights,EnergyPoints,realDest,0,[](numType E, numType& dest) {niceFunction(E,dest);dest *= pseudoFermiFunction(E);},realAlloc);
    // success += expect(1000./3.,realDest,__PRETTY_FUNCTION__,"WeightedIntegral<false> Failed");

    WeightedIntegral<true>(weights,EnergyPoints,realDest,0,[](numType E, numType& dest) {niceFunction(E,dest);},realAlloc);
    success += expectNear(1000./3.,realDest,defaultNear,__PRETTY_FUNCTION__,"WeightedIntegral<true> Failed");
    if (logErrors) logger().log("WeightedIntegral<true> " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - 1000./3.);

    ArcIntegral(100,10,-M_PI,0,complexDest,niceFunctionC,complexAlloc);
    success += expectNear(realAnswer,complexDest.real(),defaultNear,__PRETTY_FUNCTION__,"ArcIntegral Failed");
    if (logErrors) logger().log("ArcIntegral " + std::string(__PRETTY_FUNCTION__) + " diff",complexDest.real() - realAnswer);

    return success;
}

int testpeakedFunction()
{
    numType realDest;
    complexType complexDest;
    numType realAnswerStartEnd;
    numType realAnswerR10;

    int success = 0;
    numType start = -1e-4;
    numType end = 1e-4;

    realAnswerStartEnd = peakedFunctionAntiDerivative(end) - peakedFunctionAntiDerivative(start);
    realAnswerR10 = peakedFunctionAntiDerivative(10) - peakedFunctionAntiDerivative(-10);

    GaussLegendreIntegral(start,end,3000,realDest,peakedFunction,realAlloc);
    success += expectNear(realAnswerStartEnd,realDest,defaultNear,__PRETTY_FUNCTION__,"GaussLegendre Failed");
    if (logErrors) logger().log("GaussLegendre " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - realAnswerStartEnd);

    AdaptiveGaussLegendreIntegralSingularities(start,end,realDest,peakedFunction,realAlloc,{-1e-6*iu});
    success += expectNear(realAnswerStartEnd,realDest,defaultNear,__PRETTY_FUNCTION__,"AdaptiveGaussLegendreIntegral Failed");
    if (logErrors) logger().log("AdaptiveGaussLegendreIntegral " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - realAnswerStartEnd);

    AdaptiveGaussLegendreIntegralSingularities(-10,10,realDest,peakedFunction,realAlloc,{-1e-6*iu});
    success += expectNear(realAnswerR10,realDest,defaultNear,__PRETTY_FUNCTION__,"AdaptiveGaussLegendreIntegral2 Failed");
    if (logErrors) logger().log("AdaptiveGaussLegendreIntegral2 " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - realAnswerR10);

    TrapezoidalIntegral(start,end,1000,realDest,peakedFunction,realAlloc);
    success += expectNear(realAnswerStartEnd,realDest,defaultNear,__PRETTY_FUNCTION__,"TrapezoidalIntegral Failed");
    if (logErrors) logger().log("TrapezoidalIntegral " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - realAnswerStartEnd);

    DeltaFunctionIntegral(-10,10,realDest,std::vector<numType>({start}),peakedFunction,realAlloc);
    {
        numType temp;
        peakedFunction(start,temp);
        success += expect(temp,realDest,__PRETTY_FUNCTION__,"DeltaFunctionIntegral Failed");
    }

    SpectralWeightedIntegral<false>(-10,10,10,realDest,std::vector<complexType> ({-1e-6*iu}),peakedFunction,realAlloc);
    success += expectNear(realAnswerR10,realDest,defaultNear,__PRETTY_FUNCTION__,"SpectralWeightedIntegral<false> Failed");
    if (logErrors) logger().log("SpectralWeightedIntegral<false> " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - realAnswerR10);

    SpectralWeightedIntegral<false>(10,realDest,std::vector<complexType> ({-1e-6*iu}),peakedFunction,realAlloc);
    success += expectNear(2*M_PI,realDest,defaultNear,__PRETTY_FUNCTION__,"SpectralWeightedIntegral<false>2 Failed");
    if (logErrors) logger().log("SpectralWeightedIntegral<false>2 " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - 2*M_PI);

    SpectralWeightedIntegral<true>(-10,10,10,realDest,std::vector<complexType> ({-1e-6*iu}),peakedFunction,realAlloc);
    success += expectNear(realAnswerR10,realDest,defaultNear,__PRETTY_FUNCTION__,"SpectralWeightedIntegral<true> Failed");
    if (logErrors) logger().log("SpectralWeightedIntegral<true> " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - realAnswerR10);

    FermiWeightedIntegral<false>(start,end,3000,realDest,0,[](numType E, numType& dest) {peakedFunction(E,dest);dest *= pseudoFermiFunction(E);},realAlloc);
    success += expectNear(3.121593320216463,realDest,defaultNear,__PRETTY_FUNCTION__,"FermiWeightedIntegral<false> Failed");
    if (logErrors) logger().log("FermiWeightedIntegral<false> " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - 3.121593320216463);

    FermiWeightedIntegral<true>(start,end,3000,realDest,0,[](numType E, numType& dest) {peakedFunction(E,dest);},realAlloc);
    success += expectNear(3.121593320216463,realDest,defaultNear,__PRETTY_FUNCTION__,"FermiWeightedIntegral<true> Failed");
    if (logErrors) logger().log("FermiWeightedIntegral<true> " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - 3.121593320216463);

    std::vector<numType> weights;
    std::vector<numType> EnergyPoints;
    constructFermiWeightFunctionGauleg(start,end,3000,0,weights,EnergyPoints);

    // WeightedIntegral<false>(weights,EnergyPoints,realDest,0,[](numType E, numType& dest) {niceFunction(E,dest);dest *= pseudoFermiFunction(E);},realAlloc);
    // success += expect(1000./3.,realDest,__PRETTY_FUNCTION__,"WeightedIntegral<false> Failed");

    WeightedIntegral<true>(weights,EnergyPoints,realDest,0,[](numType E, numType& dest) {peakedFunction(E,dest);},realAlloc);
    success += expectNear(3.121593320216463,realDest,defaultNear,__PRETTY_FUNCTION__,"WeightedIntegral<true> Failed");
    if (logErrors) logger().log("WeightedIntegral<true> " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - 3.121593320216463);

    constructSpectralWeightFunction(-10,10,10,std::vector<complexType> ({-1e-6*iu}),weights,EnergyPoints);

    WeightedIntegral<true>(weights,EnergyPoints,realDest,0,[](numType E, numType& dest) {peakedFunction(E,dest); dest /= -2*(1./(E-(-1e-6*iu))).imag();},realAlloc);
    success += expectNear(realAnswerR10,realDest,defaultNear,__PRETTY_FUNCTION__,"WeightedIntegral<true> (spectral) Failed");
    if (logErrors) logger().log("WeightedIntegral<true> (spectral) " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - realAnswerR10);

    ArcIntegral(100,10,-M_PI,0,complexDest,peakedFunctionC,complexAlloc);
    success += expectNear(realAnswerR10,2*complexDest.imag(),defaultNear,__PRETTY_FUNCTION__,"ArcIntegral Failed");
     if (logErrors) logger().log("ArcIntegral " + std::string(__PRETTY_FUNCTION__) + " diff",2*complexDest.imag() - realAnswerR10);

    return success;
}

int testpeakedComplexFunction()
{
    complexType complexDest;
    complexType complexAnswerStartEnd;
    complexType complexAnswerR10;

    int success = 0;
    numType start = -1e-4;
    numType end = 1e-4;

    complexAnswerStartEnd = peakedComplexFunctionAntiDerivative(end) - peakedComplexFunctionAntiDerivative(start);
    complexAnswerR10 = peakedComplexFunctionAntiDerivative(10) - peakedComplexFunctionAntiDerivative(-10);

    GaussLegendreIntegral(start,end,3000,complexDest,peakedComplexFunction,complexAlloc);
    success += expectNear(complexAnswerStartEnd,complexDest,defaultNearC,__PRETTY_FUNCTION__,"GaussLegendre Failed");
    if (logErrors) logger().log("GaussLegendre " + std::string(__PRETTY_FUNCTION__) + " diff",complexDest - complexAnswerStartEnd);

    AdaptiveGaussLegendreIntegralSingularities(start,end,complexDest,peakedComplexFunction,complexAlloc,{-1e-6*iu});
    success += expectNear(complexAnswerStartEnd,complexDest,defaultNearC,__PRETTY_FUNCTION__,"AdaptiveGaussLegendreIntegral Failed");
    if (logErrors) logger().log("AdaptiveGaussLegendreIntegral " + std::string(__PRETTY_FUNCTION__) + " diff",complexDest - complexAnswerStartEnd);

    AdaptiveGaussLegendreIntegralSingularities(-10,10,complexDest,peakedComplexFunction,complexAlloc,{-1e-6*iu});
    success += expectNear(complexAnswerR10,complexDest,defaultNearC,__PRETTY_FUNCTION__,"AdaptiveGaussLegendreIntegral2 Failed");
    if (logErrors) logger().log("AdaptiveGaussLegendreIntegral2 " + std::string(__PRETTY_FUNCTION__) + " diff",complexDest - complexAnswerR10);

    TrapezoidalIntegral(start,end,10000,complexDest,peakedComplexFunction,complexAlloc);
    success += expectNear(complexAnswerStartEnd,complexDest,defaultNearC,__PRETTY_FUNCTION__,"TrapezoidalIntegral Failed");
    if (logErrors) logger().log("TrapezoidalIntegral " + std::string(__PRETTY_FUNCTION__) + " diff",complexDest - complexAnswerStartEnd);

    DeltaFunctionIntegral(-10,10,complexDest,std::vector<numType>({start}),peakedComplexFunction,complexAlloc);
    {
        complexType temp;
        peakedComplexFunction(start,temp);
        success += expectNear<complexType,complexType>(temp,complexDest,[=](const complexType& a, const complexType& b){return ((a-b)/b).real() < 1e-16 && ((a-b)/b).imag() < 1e-16;},__PRETTY_FUNCTION__,"DeltaFunctionIntegral Failed");
        // logger().logAccurate("Diff",1e20*(complexDest-temp)/temp);
    }

    SpectralWeightedIntegral<false>(start,end,100000,complexDest,std::vector<complexType> ({-1e-6*iu}),peakedComplexFunction,complexAlloc);
    success += expectNear(complexAnswerStartEnd,complexDest,defaultNearC,__PRETTY_FUNCTION__,"SpectralWeightedIntegral<false> Failed");
    if (logErrors) logger().log("SpectralWeightedIntegral<false> " + std::string(__PRETTY_FUNCTION__) + " diff",complexDest - complexAnswerStartEnd);

    SpectralWeightedIntegral<true>(start,end,10000,complexDest,std::vector<complexType> ({-1e-6*iu}),peakedComplexFunction,complexAlloc);
    success += expectNear(complexAnswerStartEnd,complexDest,defaultNearC,__PRETTY_FUNCTION__,"SpectralWeightedIntegral<true> Failed");
    if (logErrors) logger().log("SpectralWeightedIntegral<true> " + std::string(__PRETTY_FUNCTION__) + " diff",complexDest - complexAnswerStartEnd);

    FermiWeightedIntegral<false>(start,end,9999,complexDest,0,[](numType E, complexType& dest) {peakedComplexFunction(E,dest);dest *= pseudoFermiFunction(E);},complexAlloc);
    success += expectNear(-9999.000099989613-816.1624616246891*iu,complexDest,defaultNearC,__PRETTY_FUNCTION__,"FermiWeightedIntegral<false> Failed");
    if (logErrors) logger().log("FermiWeightedIntegral<false> " + std::string(__PRETTY_FUNCTION__) + " diff",complexDest - (-9999.000099989613-816.1624616246891*iu));

    FermiWeightedIntegral<true>(start,end,9999,complexDest,0,[](numType E, complexType& dest) {peakedComplexFunction(E,dest);},complexAlloc);
    success += expectNear(-9999.000099989613-816.1624616246891*iu,complexDest,defaultNearC,__PRETTY_FUNCTION__,"FermiWeightedIntegral<true> Failed");
    if (logErrors) logger().log("FermiWeightedIntegral<true> " + std::string(__PRETTY_FUNCTION__) + " diff",complexDest - (-9999.000099989613-816.1624616246891*iu));

    std::vector<numType> weights;
    std::vector<numType> EnergyPoints;
    constructFermiWeightFunctionGauleg(start,end,9999,0,weights,EnergyPoints);

    // WeightedIntegral<false>(weights,EnergyPoints,realDest,0,[](numType E, numType& dest) {niceFunction(E,dest);dest *= pseudoFermiFunction(E);},realAlloc);
    // success += expect(1000./3.,realDest,__PRETTY_FUNCTION__,"WeightedIntegral<false> Failed");

    WeightedIntegral<true>(weights,EnergyPoints,complexDest,0,[](numType E, complexType& dest) {peakedComplexFunction(E,dest);},complexAlloc);
    success += expectNear(-9999.000099989613-816.1624616246891*iu,complexDest,defaultNearC,__PRETTY_FUNCTION__,"WeightedIntegral<true> (fermi) Failed");
    if (logErrors) logger().log("WeightedIntegral<true> " + std::string(__PRETTY_FUNCTION__) + " diff",complexDest - (-9999.000099989613-816.1624616246891*iu));

    constructSpectralWeightFunction(start,end,100000,std::vector<complexType> ({-1e-6*iu}),weights,EnergyPoints);

    WeightedIntegral<true>(weights,EnergyPoints,complexDest,0,[](numType E, complexType& dest) {peakedComplexFunction(E,dest); dest /= -2*(1./(E-(-1e-6*iu))).imag();},complexAlloc);
    success += expectNear(complexAnswerStartEnd,complexDest,defaultNearC,__PRETTY_FUNCTION__,"WeightedIntegral<true> (spectral) Failed");
    if (logErrors) logger().log("WeightedIntegral<true> (spectral) " + std::string(__PRETTY_FUNCTION__) + " diff",complexDest - complexAnswerStartEnd);


    ArcIntegral(100,10,-M_PI,0,complexDest,peakedComplexFunctionC,complexAlloc);
    success += expectNear(complexAnswerR10,complexDest,defaultNearC,__PRETTY_FUNCTION__,"ArcIntegral Failed");
    if (logErrors) logger().log("ArcIntegral " + std::string(__PRETTY_FUNCTION__) + " diff",complexDest - complexAnswerR10);

    return success;
}

int testIncreasingStepSize()
{
    double realDest;
    float realDestf;
    numType realAnswerStartEnd;
    numType start = -1e-4;
    numType end = 1e-4;

    realAnswerStartEnd = peakedFunctionAntiDerivative(end) - peakedFunctionAntiDerivative(start);

    int sizes[] = {1,10,100,1000,10000,100000,1000000,10000000,100000000};
    for (auto& s : sizes)
    {
        TrapezoidalIntegral<double,decltype(peakedFunction),decltype(realAlloc),false>(start,end,s,realDest,peakedFunction,realAlloc);
        if (logErrors) logger().log("TrapezoidalIntegral No Kahan sum " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - realAnswerStartEnd);
    }
    for (auto& s : sizes)
    {
        TrapezoidalIntegral<double,decltype(peakedFunction),decltype(realAlloc),true>(start,end,s,realDest,peakedFunction,realAlloc);
        if (logErrors) logger().log("TrapezoidalIntegral Kahan sum " + std::string(__PRETTY_FUNCTION__) + " diff",realDest - realAnswerStartEnd);
    }

    for (auto& s : sizes)
    {
        TrapezoidalIntegral<float,decltype(peakedFunctionf),float(void),false>((float)start,(float)end,s,realDestf,peakedFunctionf,[](){return 0.f;});
        if (logErrors) logger().log("TrapezoidalIntegral(float) No Kahan sum " + std::string(__PRETTY_FUNCTION__) + " diff",realDestf - realAnswerStartEnd);
    }
    for (auto& s : sizes)
    {
        TrapezoidalIntegral<float,decltype(peakedFunctionf),float(void),true>((float)start,(float)end,s,realDestf,peakedFunctionf,[](){return 0.f;});
        if (logErrors) logger().log("TrapezoidalIntegral(float) Kahan sum " + std::string(__PRETTY_FUNCTION__) + " diff",realDestf - realAnswerStartEnd);
    }
    if (logErrors) logger().log("Best float approx " + std::string(__PRETTY_FUNCTION__) + " diff",(float)realAnswerStartEnd - realAnswerStartEnd);
    return 0;
}

int integratorTests(int argc, char** argv)
{
    int success = 0;
    success += testNiceFunction();
    success += testpeakedFunction();
    success += testpeakedComplexFunction();
    testIncreasingStepSize();

    return success > 0 ? -1 : 0;

}
#endif // FOO_CPP
