// #include "configfileloader.h"
// #include "logger.h"
// #include "secondquantisedhamiltonian.h"
// #include "explicitleadselfenergy.h"
// #include "phononselfenergy.h"
// #include "simpleleadselfenergy.h"



// ConfigFileLoader::ConfigFileLoader(std::string filename)
// {
//     m_filename = filename;
//     m_file = fopen(m_filename.c_str(),"r");
//     if (m_file == nullptr)
//     {
//         logger().log("Could not open file for logging",filename);
//         logger().log("Error:",errno);
//         m_file = stderr;
//     }
//     else
//         load();

// }

// ConfigFileLoader::~ConfigFileLoader()
// {
//     if (m_file != nullptr)
//         fclose(m_file);
// }
// void ConfigFileLoader::load()
// {
//     while (true)
//     {
//         //Read object
//         char objectName[100] = {};
//         int ret = fscanf(m_file," %99[^{}\n ] {",objectName);
//         if (ret <= 0)
//             break;
//         std::string objectName_s(objectName);
//         if (objectName_s == "GreensFunction")
//         {
//             m_greensFunction = std::make_shared<GreensFunction>();
//             load();
//             if (m_Ham != nullptr)
//                 m_greensFunction->setHamiltonian(m_Ham);
//             else
//                 logger().log("No Hamiltonian set for Greens function");

//             if (m_selfEnergies.size() > 0)
//                 m_greensFunction->setSelfEnergies(m_selfEnergies);
//             else
//                 logger().log("No SelfEnergies set for Greens function");
//             m_scf.setGreensFunction(m_greensFunction);

//         }
//         else if (objectName_s == "SecondQuantisedHamiltonian::Parameters")
//         {
//             m_Ham = std::make_shared<SecondQuantisedHamiltonian>();
//             m_Ham->load(m_file);
//             if (m_greensFunction == nullptr)
//                 logger().log("Hamiltonian set before Greens function declared");


//         }
//         else if (objectName_s == "ExplicitLeadSelfEnergy<SelfEnergyMatrixType>::Parameters")
//         {
//             std::shared_ptr<ExplicitLeadSelfEnergy<SelfEnergyMatrix>> SE = std::make_shared<ExplicitLeadSelfEnergy<SelfEnergyMatrix>>(ExplicitLeadSelfEnergy<SelfEnergyMatrix>::Parameters());
//             SE->load(m_file);
//             SE->setMatrixDimension(m_Ham->getHilbertSpaceSize());
//             m_selfEnergies.push_back(SE);
//         }
//         else if (objectName_s == "PhononSelfEnergy<SelfEnergyMatrixType>::Parameters")
//         {
//             std::shared_ptr<PhononSelfEnergy<SelfEnergyMatrix>> SE = std::make_shared<PhononSelfEnergy<SelfEnergyMatrix>>(PhononSelfEnergy<SelfEnergyMatrix>::Parameters());
//             SE->load(m_file);
//             SE->setGreensFunction(m_greensFunction);
//             SE->setPhononCorrelationMatrix(m_Ham->getPhononCorrelationMatrix());
//             SE->setPhononGTensor(m_Ham->getPhononGTensor());
//             SE->setMatrixDimension(m_Ham->getHilbertSpaceSize());
//             m_selfEnergies.push_back(SE);
//             m_scf.setSelfConsistentSelfEnergy(SE);

//         }
//         else if (objectName_s == "SimpleLeadSelfEnergy<SelfEnergyMatrixType>::Parameters")
//         {
//             std::shared_ptr<SimpleLeadSelfEnergy<SelfEnergyMatrix>> SE = std::make_shared<SimpleLeadSelfEnergy<SelfEnergyMatrix>>(SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters());
//             SE->load(m_file);
//             SE->setMatrixDimension(m_Ham->getHilbertSpaceSize());
//             m_selfEnergies.push_back(SE);
//         }
//         else if (objectName_s == "CurrentCalc::Parameters")
//         {
//             CurrentCalc::Parameters ccParam;
//             CurrentCalc cc(ccParam);
//             cc.load(m_file);
//             bool found = false;
//             for(auto& se : m_selfEnergies)
//             {
//                 if (se->m_prettyName == cc.m_selfEnergyName)
//                 {
//                     cc.setSelfEnergy(se);
//                     found = true;
//                     break;
//                 }
//             }
//             if (!found)
//             {
//                 logger().log("Didnt find self energy with name, make sure the self energies come before the current calcs: ",cc.m_selfEnergyName);
//             }
//             cc.setGreensFunction(m_greensFunction);
//             m_currentCalcs.push_back(cc);
//         }
//         else if (objectName_s == "ElectronDensityCalc::Parameters")
//         {
//             ElectronDensityCalc::Parameters edcParam;
//             ElectronDensityCalc edc(edcParam);
//             edc.load(m_file);
//             edc.setGreensFunction(m_greensFunction);
//             m_electronDensityCalcs.push_back(edc);
//         }
//         else if (objectName_s == "SpectralFunction::Parameters")
//         {
//             SpectralFunction::Parameters sfcParam;
//             SpectralFunction sfc(sfcParam);
//             sfc.load(m_file);
//             sfc.setGreensFunction(m_greensFunction);
//             m_spectralFunctionCalcs.push_back(sfc);
//         }
//         else if (objectName_s == "SCFSolver::Parameters")
//         {
//             m_scf.load(m_file);
//         }
//         else
//         {
//             logger().log("Unknown object in config file, ignoring: ", objectName_s);
//         }

//         do
//         {
//             ret = fgetc(m_file);
//             //jump to the next }
//         }while(ret > 0 && ret != '}');

//         if (ret < 0)
//             break;
//     }
// }

// void ConfigFileLoader::setFileName(std::string filename)
// {
//     m_filename = filename;
//     //TODO this is UGLY af
// }

// void ConfigFileLoader::save()
// {
//     {
//         //Delete any existing file
//         logger l(m_filename);
//     }

//     m_greensFunction->save(m_filename);

//     for (auto& cc : m_currentCalcs)
//     {
//         cc.save(m_filename);
//     }
//     for (auto& edc : m_electronDensityCalcs)
//         edc.save(m_filename);
//     for (auto& sfc:  m_spectralFunctionCalcs)
//         sfc.save(m_filename);
// }

// void ConfigFileLoader::loadParameter(FILE *file, std::string &value)
// {
//     char objectName[30] = {};
//     int ret = fscanf(file," %s",objectName);
//     value = objectName;
//     if (ret <= 0)
//         logger().log("Error loading string parameter");
// }


// void ConfigFileLoader::loadParameter(FILE *file, numType &value)
// {
//     int ret = fscanf(file," " numTypeCode,&value);
//     if (ret <= 0)
//         logger().log("Error loading numType parameter");
// }

// void ConfigFileLoader::loadParameter(FILE *file, complexType &value)
// {
//     numType real;
//     numType imag;

//     int ret = fscanf(file," " numTypeCode "," numTypeCode ,&real,&imag);
//     if (ret <= 0)
//         logger().log("Error loading numType parameter");
//     value = complexType(real,imag);
// }

// void ConfigFileLoader::loadParameter(FILE *file, uint32_t &value)
// {
//     int ret = fscanf(file," %u",&value);
//     if (ret <= 0)
//         logger().log("Error loading uint32_t parameter");
// }
// void ConfigFileLoader::loadParameter(FILE *file, int &value)
// {
//     int ret = fscanf(file," %i",&value);
//     if (ret <= 0)
//         logger().log("Error loading int parameter");
// }

// void ConfigFileLoader::loadParameter(FILE *file, ComplexMatrix &value)
// {
//     int rows;
//     int cols;
//     loadParameter(file,rows);
//     loadParameter(file,cols);
//     value.resize(rows,cols);
//     value.setZero();

//     for (int i = 0; i < rows; i++)
//     {
//         for (int j = 0; j < cols; j++)
//         {
//             loadParameter(file,value.coeffRef(i,j));
//         }
//     }

// }


