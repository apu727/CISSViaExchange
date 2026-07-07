#ifndef QUANTITYCALC_H
#define QUANTITYCALC_H


#include "selfenergybase.h"
#include "greensfunction.h"
std::vector<numType> autoRange(long NumberOfSteps, std::shared_ptr<GreensFunction> gFunc);
std::vector<Eigen::VectorXd> interpolateWithSpectralWeight(std::vector<numType> points,std::vector<numType> existingPoints, std::vector<Eigen::VectorXd> existingData,  std::shared_ptr<GreensFunction> gFunc, numType artificialBroadening = 0);

class CurrentCalc
{
    std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>> m_selfEnergy;
    std::shared_ptr<GreensFunction> m_GFunc;
    void asyncI(std::vector<numType>& individualSums,numType& step) const;


public:
    typedef struct
    {
        int numberOfIntegrationPoints;
        numType startE;
        numType endE;
    } Parameters;
    Parameters m_params;
    std::string m_selfEnergyName;

    CurrentCalc(Parameters params);
    virtual ~CurrentCalc(){};
    void setSelfEnergy(std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>> SE){m_selfEnergy = SE; m_selfEnergyName = m_selfEnergy->m_prettyName;}
    void setGreensFunction(std::shared_ptr<GreensFunction> GFunc){m_GFunc = GFunc;}

    numType getiE(numType E) const;
    std::vector<numType> getiE(std::vector<numType> Energies)  const; // For python interface, not efficient in copying
    numType getI() const;
    // virtual void save(std::string filename) const;
    // void load(FILE* file);

};

class DivCurrentCalc
{// computes -e \nabla \cdot J
    std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>>> m_selfEnergies;
    std::shared_ptr<GreensFunction> m_GFunc;
    std::vector<numType> m_energies;


public:
    typedef struct
    {
        int numberOfIntegrationPoints;
        numType startE;
        numType endE;
    } Parameters;
    Parameters m_params;


    DivCurrentCalc(Parameters params);
    virtual ~DivCurrentCalc(){};
    void setGreensFunction(std::shared_ptr<GreensFunction> GFunc, std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>>> SEs){m_GFunc = GFunc; m_selfEnergies = SEs;}

    void getAtE(numType E,Eigen::VectorXd& dest) const;
    std::vector<Eigen::VectorXd> getAtE(std::vector<numType> Energies) const; // For python interface, not efficient in copying
    Eigen::VectorXd get() const;
    // virtual void save(std::string filename) const {logger().log("Not Implemented");};
    // void load(FILE* file){logger().log("Not Implemented");};

    bool isConstant() const;

};

class ElectronDensityCalc
{
    std::shared_ptr<GreensFunction> m_GFunc;
    void asyncnEr(std::vector<Eigen::VectorXd>& individualSums,numType& step) const;
    void asyncnEr(std::vector<Eigen::VectorXd> &individualSums, numType &step, const std::string& basis) const;
public:
    typedef struct
    {
        int numberOfIntegrationPoints;
        numType startE;
        numType endE;
    } Parameters;
    Parameters m_params;

    ElectronDensityCalc(Parameters params){m_params = params;}
    virtual ~ElectronDensityCalc(){};
    void setGreensFunction(std::shared_ptr<GreensFunction> GFunc){m_GFunc = GFunc;}

    ComplexSelfAdjointMatrix getGn(bool perturb, numType EStart = negInf, numType EEnd = posInf) const;
    ComplexSelfAdjointMatrix getNonEqGn(bool perturb) const;
    Eigen::VectorXd getnR() const;
    Eigen::VectorXcd getnS() const; // only traces the spatial degrees of freedom
    Eigen::VectorXcd getnS(const std::string& basis) const; // only traces the spatial degrees of freedom
    Eigen::VectorXd getnR(const std::string& basis) const;

    Eigen::VectorXd getnEr(numType E) const;
    Eigen::VectorXd getnEr(numType E, const std::string& basis) const;

    std::vector<Eigen::VectorXd> getnEr(std::vector<numType> E) const; // for python interface, not efficient in copying
    std::vector<Eigen::VectorXd> getnEr(std::vector<numType> E, const std::string& basise) const; // for python interface, not efficient in copying


    // virtual void save(std::string filename) const;
    // void load(FILE* file);
};

class SpectralFunction
{
    std::shared_ptr<GreensFunction> m_GFunc;
    void asyncAEr(std::vector<Eigen::VectorXd>& individualSums,numType& step) const;
    void asyncAEr(std::vector<Eigen::VectorXd>& individualSums,numType& step, const std::string& basis) const;
public:
    typedef struct
    {
        int numberOfIntegrationPoints;
        numType startE;
        numType endE;
    } Parameters;
    Parameters m_params;

    SpectralFunction(Parameters params){m_params = params;}
    virtual ~SpectralFunction(){};
    void setGreensFunction(std::shared_ptr<GreensFunction> GFunc){m_GFunc = GFunc;}

    Eigen::VectorXd getAR() const;
    Eigen::VectorXd getAR(const std::string& basis) const;
    Eigen::VectorXd getAEr(numType E) const;
    Eigen::VectorXd getAEr(numType E,const std::string& basis) const;
    std::vector<Eigen::VectorXd> getAEr(std::vector<numType> E) const; // for python interface, not efficient in copying
    std::vector<Eigen::VectorXd> getAEr(std::vector<numType> E, const std::string& basis) const; // for python interface, not efficient in copying
    // virtual void save(std::string filename) const;
    // void load(FILE* file);
};

class OperatorExpecationValue
{
    std::shared_ptr<GreensFunction> m_GFunc;
public:

    OperatorExpecationValue(){}
    virtual ~OperatorExpecationValue(){}

    complexType getElecValAtE(numType E, const ComplexMatrix& op) const;
    complexType getHoleValAtE(numType E, const ComplexMatrix& op) const;
    void setGreensFunction(std::shared_ptr<GreensFunction> GFunc){m_GFunc = GFunc;}

    std::vector<complexType> getElecValE(std::vector<numType> E, const ComplexMatrix &op) const;
    std::vector<complexType> getHoleValE(std::vector<numType> E, const ComplexMatrix &op) const;
};

class TransmissionFunc
{
    std::shared_ptr<GreensFunction> m_GFunc;
public:
    TransmissionFunc(){}
    virtual ~TransmissionFunc(){}

    void setGreensFunction(std::shared_ptr<GreensFunction> GFunc){m_GFunc = GFunc;}

    complexType getTransmissionAtEBetween(numType E, std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>> p, std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>> q) const;

};

class ThermalEnergy
{
    std::shared_ptr<GreensFunction> m_GFunc;
    std::vector<numType> m_energies;
    ComplexSelfAdjointSparseMatrix m_Ham;


public:
    typedef struct
    {
        int numberOfIntegrationPoints;
        numType startE;
        numType endE;
        numType extraEnergy;
    } Parameters;

    Parameters m_params;
    ThermalEnergy(Parameters params);
    virtual ~ThermalEnergy(){}

    void setGreensFunction(std::shared_ptr<GreensFunction> GFunc){m_GFunc = GFunc; m_Ham = m_GFunc->getHamiltonian();}
    numType getEnergy() const;
    numType getEnergy(const std::string& basis) const;

    numType getAtE(numType E) const;
    numType getAtE(numType E, const std::string& basis) const;
    std::vector<numType> getAtE(std::vector<numType> Energies, const std::string& basis) const; // For python interface, not efficient in copying
};


#endif // QUANTITYCALC_H
