#ifndef CONFIGFILELOADER_H
#define CONFIGFILELOADER_H

// #include "greensfunction.h"
// #include "logger.h"
// #include "quantitycalc.h"
// #include "scfsolver.h"
// #include "selfenergybase.h"
// #include <string>
// #include <vector>

// class ConfigFileLoader
// {
//     FILE* m_file = nullptr;
//     std::string m_filename;
//     std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>>> m_selfEnergies;
//     std::shared_ptr<GreensFunction> m_greensFunction;
//     std::shared_ptr<HamiltonianBase> m_Ham;
//     std::vector<CurrentCalc> m_currentCalcs;
//     std::vector<ElectronDensityCalc> m_electronDensityCalcs;
//     std::vector<SpectralFunction> m_spectralFunctionCalcs;
//     SCFSolver m_scf;

// public:
//     ConfigFileLoader(std::string filename);
//     virtual ~ConfigFileLoader();
//     void load();
//     bool success(){return m_file != nullptr;}

//     void setFileName(std::string filename);
//     void save();

//     std::vector<std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>>> getSelfEnergies(){return m_selfEnergies;}
//     std::shared_ptr<GreensFunction> getGreensFunction(){return m_greensFunction;};
//     std::shared_ptr<HamiltonianBase> getHam() {return m_Ham;}
//     std::vector<CurrentCalc> getCurrentCalcs() {return m_currentCalcs;}
//     std::vector<ElectronDensityCalc> getElectronDensityCalcs() {return m_electronDensityCalcs;}
//     std::vector<SpectralFunction> getSpectraFunctionCalcs() {return m_spectralFunctionCalcs;}
//     SCFSolver getSCF() {return m_scf;}


//     //Helper functions
//     static void loadParameter(FILE* file, std::string& value);
//     static void loadParameter(FILE* file, numType& value);
//     static void loadParameter(FILE* file, complexType& value);
//     static void loadParameter(FILE* file, uint32_t& value);
//     static void loadParameter(FILE* file, int& value);
//     static void loadParameter(FILE* file, ComplexMatrix& value);

//     static void loadParameter(FILE* file, std::vector<std::complex<double>>& value)
//     {
//         int ret = fgetc(file);
//         if (ret < 0 || ret != '[')
//             logger().log("Error loading [ parameter");
//         std::string formatCode = "%lg:%lg, ";
//         while(ret > 0)
//         {
//             double real;
//             double imag;
//             ret = fscanf(file, formatCode.c_str(),&real,&imag);
//             if (ret)
//                 value.push_back({real,imag});
//         }
//         do
//         {
//             ret = fgetc(file);
//         } while (ret > 0 && ret != ']');
//     }

//     template <typename T>
//     static void loadParameter(FILE* file, std::vector<T>& value, std::string formatCode)
//     {
//         int ret = fgetc(file);
//         if (ret < 0 || ret != '[')
//             logger().log("Error loading int parameter");
//         formatCode += " , ";
//         while(ret > 0)
//         {
//             T loaded;
//             ret = fscanf(file, formatCode.c_str(),&loaded);
//             if (ret)
//                 value.push_back(loaded);
//         }
//         do
//         {
//             ret = fgetc(file);
//         } while (ret > 0 && ret != ']');
//     }


// };

#endif // CONFIGFILELOADER_H
