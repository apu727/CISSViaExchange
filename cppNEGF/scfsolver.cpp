#include "scfsolver.h"
#include "logger.h"
#include "configfileloader.h"


SCFSolver::SCFSolver() {}

void SCFSolver::run(std::function<bool(void)> convergenceFunc, int numberOfTimes, numType TStart, numType TEnd, numType TStep)
{
    numType TBkp = T;
    T = TStart;
    if (numberOfTimes > 0)
        m_numberOfTimes = numberOfTimes;
    if (m_selfConsistentSelfEnergy.size() == 0)
    {
        logger().log("No SelfConsistent Self Energy set\n");
        m_isConverged = true;
        T = TBkp;
        return;
    }
    logger().log("Number of loops",m_numberOfTimes);
    m_GreensFunction->clearCache();
    for (auto &se : m_GreensFunction->getSelfEnergies())
    {
        se->clearCache();
    }
    if (m_callback)
        m_callback(); // allow for logging

    for (int idx = 0; idx < m_numberOfTimes && !convergenceFunc(); idx++)
    {
        T -= TStep;
        if (T < TEnd)
            T = TEnd;
        logger().log("T",T);
        for (auto& SCSE : m_selfConsistentSelfEnergy)
            SCSE->recomputeSelfEnergies();
        m_GreensFunction->clearCache();
        for (auto &se : m_GreensFunction->getSelfEnergies())
        {
            se->clearCache();
        }
        if (m_callback)
            m_callback(); // allow for logging
    }
    m_isConverged = true;
    T = TBkp;
}

std::shared_ptr<GreensFunction> SCFSolver::getConvergedGreensFunction()
{
    return m_GreensFunction;
}

// void SCFSolver::save(std::string filename) const
// {
//     logger logFile = logger(filename,true);
//     logFile.log("SCFSolver::Parameters\n{");
//     logFile.log("numberOfTimes",m_numberOfTimes);
//     logFile.log("}");
// }

// void SCFSolver::load(FILE *file)
// {
//     while (true)
//     {
//         char objectName[30];
//         int ret = fscanf(file," %29[^{}:\n ] : ",objectName);
//         if (ret <= 0)
//             break;
//         std::string objectName_s(objectName);
//         if (objectName_s == "numberOfTimes")
//         {
//             ConfigFileLoader::loadParameter(file,m_numberOfTimes);
//         }
//         else
//         {
//             logger().log("Unknown SCFSolver parameter: ",objectName_s);
//         }
//     }
// }
