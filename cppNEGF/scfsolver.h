#ifndef SCFSOLVER_H
#define SCFSOLVER_H

#include "greensfunction.h"
#include "selfenergybase.h"

class SCFSolver
{
    std::shared_ptr<GreensFunction> m_GreensFunction;
    std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>>> m_selfConsistentSelfEnergy;
    std::function<void()> m_callback;
    bool m_isConverged = false;
    int m_numberOfTimes = 100;
public:
    SCFSolver();
    virtual ~SCFSolver(){};
    void setGreensFunction(std::shared_ptr<GreensFunction> G){m_GreensFunction = G;}
    void setSelfConsistentSelfEnergy(std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>>> SE){m_selfConsistentSelfEnergy = SE;}
    void setCallBackFunction(std::function<void()> func){m_callback = func;}
    void run(std::function<bool(void)> convergenceFunc, int numberOfTimes = -1,numType TStart = T, numType TEnd = T, numType TStep = 0);
    std::shared_ptr<GreensFunction> getConvergedGreensFunction();
    // void save(std::string filename) const;
    // void load(FILE* file);

};

#endif // SCFSOLVER_H
