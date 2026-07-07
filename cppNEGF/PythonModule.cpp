
#include "fockhamiltonian.h"
#include "greensfunction.h"
#include "matrixselfenergy.h"
#include "phononselfenergy.h"
#include "quantitycalc.h"
#include "scfsolver.h"
#include "secondquantisedhamiltonian.h"
#include "simpleleadselfenergy.h"
#include "explicitleadselfenergy.h"
#include "interfaceselfenergy.h"
#include "configfileloader.h"
#include "basismanager.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/native_enum.h>
#include <pybind11/operators.h>




namespace py = pybind11;

//py::module abc = m.def_submodule("abc");

void declare_linalg(py::module &m);

PYBIND11_MODULE(cppNEGF, m) {
    //Globals module
    py::module globals = m.def_submodule("globals");
    globals.attr("lengthUnit")= py::float_(lengthUnit);
    globals.attr("electricPotentialUnit") = py::float_(electricPotentialUnit);
    globals.attr("e") = py::float_(e);
    globals.attr("T") = py::float_(T);

    //End Globals module
    //enums
    py::native_enum<enums::IndexType>(m, "IndexType","enum.Enum")
        .value("direct", enums::IndexType::direct)
        .value("dual", enums::IndexType::dual)
        .value("improperDirect",enums::IndexType::improperDirect)
        .value("improperDual",enums::IndexType::improperDual)
        .value("eigenValue",enums::IndexType::eigenValue)
        .value("invalid",enums::IndexType::invalid)
        .finalize();

    py::native_enum<enums::SpinSymmetry>(m, "SpinSymmetry","enum.Enum")
        .value("NoSpin", enums::SpinSymmetry::NoSpin)
        .value("NoSpinSym", enums::SpinSymmetry::NoSpinSym)
        .value("RHF",enums::SpinSymmetry::RHF)
        .value("SpinMatrix",enums::SpinSymmetry::SpinMatrix)
        .finalize();

    py::native_enum<enums::MatrixProperties>(m, "MatrixProperties","enum.Enum")
        .value("None", enums::MatrixProperties::None)
        .value("selfAdjoint", enums::MatrixProperties::selfAdjoint)
        .value("Hermitian",enums::MatrixProperties::Hermitian)
        .value("Unitary",enums::MatrixProperties::Unitary)
        .value("NormPreserving",enums::MatrixProperties::NormPreserving)
        .finalize();

    py::native_enum<GreensFunction::integratorType>(m, "integratorType","enum.Enum")
        .value("autoDetect", GreensFunction::integratorType::autoDetect)
        .value("SumOverPolesEquilibrium", GreensFunction::integratorType::SumOverPolesEquilibrium)
        .value("SumOverPolesNonEquilibrium",GreensFunction::integratorType::SumOverPolesNonEquilibrium)
        .value("NoAssumptions",GreensFunction::integratorType::NoAssumptions)
        .value("Tricks",GreensFunction::integratorType::Tricks)
        .finalize();

    py::native_enum<GreensFunction::EnforcedSpinSymmetryType>(m, "EnforcedSpinSymmetryType","enum.Enum")
        .value("autoDetect", GreensFunction::EnforcedSpinSymmetryType::autoDetect)
        .value("GHF", GreensFunction::EnforcedSpinSymmetryType::GHF)
        .value("RHF",GreensFunction::EnforcedSpinSymmetryType::RHF)
        .value("UHF",GreensFunction::EnforcedSpinSymmetryType::UHF)
        .finalize();

    py::native_enum<GreensFunction::SubsystemFlags>(m, "SubsystemFlags","enum.Flag")
        .value("false", GreensFunction::SubsystemFlags::False)
        .value("AOPartitioning", GreensFunction::SubsystemFlags::AOPartitioning)
        .value("FreezeLead",GreensFunction::SubsystemFlags::FreezeLead)
        .value("FreezeMol",GreensFunction::SubsystemFlags::FreezeMol)
        .value("IgnoreGLM",GreensFunction::SubsystemFlags::IgnoreGLM)
        .finalize();

    py::native_enum<InterfaceSelfEnergy<SelfEnergyMatrix>::LeadType>(m, "InterfaceLeadType","enum.Enum")
        .value("GreensFunction", InterfaceSelfEnergy<SelfEnergyMatrix>::LeadType::GreensFunction)
        .value("TightBindingInfinite", InterfaceSelfEnergy<SelfEnergyMatrix>::LeadType::TightBindingInfinite)
        .value("ContinuumInfinite",InterfaceSelfEnergy<SelfEnergyMatrix>::LeadType::ContinuumInfinite)
        .value("None",InterfaceSelfEnergy<SelfEnergyMatrix>::LeadType::None)
        .finalize();



    //endenums
    //BasisManager
    py::class_<BasisManager,std::unique_ptr<BasisManager, py::nodelete>>(m,"BasisManager")
        .def(py::init(
            []()
            {
                return std::unique_ptr<BasisManager, py::nodelete>(&BasisManager::getInstance());
            }))
        .def("addOrthogonalBasis",py::overload_cast<std::string,long>(&BasisManager::addOrthogonalBasis),py::arg("name"),py::arg("dim"))
        .def("isOrthogonal",py::overload_cast<const std::string& >(&BasisManager::isOrthogonal),py::arg("name"))
        .def("getTransform",&BasisManager::getTransform,py::arg("From"),py::arg("fromIndex"),py::arg("to"),py::arg("toIndex"),py::arg("AllowProjectedRaising") = false)
        .def("getMetric",&BasisManager::getMetric,py::arg("Name"))
        .def_static("deleteBasis",&BasisManager::deleteBasis,py::arg("name"))
        .def_static("deleteAllBasis",&BasisManager::deleteAllBasis)
        .def_static("buildCanonicalBasisString",py::overload_cast<std::string>(&BasisManager::buildCanonicalBasisString),py::arg("basis"))
        .def_static("hasSpinBlockTag", &BasisManager::hasSpinBlockTag, py::arg("basisName"))
        .def_static("removeSpinTags",
             [](const std::string& bas)
             {
                 BasisManager::possibleTags tag = BasisManager::possibleTags::NullBasis;
                 std::string ret = BasisManager::removeSpinTags(bas,&tag);
                 if (tag == BasisManager::possibleTags::NullBasis)
                     logger().log("Basis:" + bas + " did not have a spin tag");
                 return ret;

             })
        .def_static("removeSpinTags",
                    [](const indexBasisT& bas)
                    {
                        BasisManager::possibleTags tag = BasisManager::possibleTags::NullBasis;
                        std::string reti = BasisManager::removeSpinTags(bas[0],&tag);
                        if (tag == BasisManager::possibleTags::NullBasis)
                            logger().log("Basis:" + bas[0] + " did not have a spin tag");
                        std::string retj = BasisManager::removeSpinTags(bas[1],&tag);
                        if (tag == BasisManager::possibleTags::NullBasis)
                            logger().log("Basis:" + bas[1] + " did not have a spin tag");
                        return indexBasisT({reti,retj});

                    })
        .def_property_readonly_static("Alpha_Block",[](py::object){return BasisManager::Alpha_Block;})
        .def_property_readonly_static("Beta_Block",[](py::object){return BasisManager::Beta_Block;})
        .def_property_readonly_static("LowdinOrthogonaliseBasis",[](py::object){return BasisManager::LowdinOrthogonaliseBasis;})

        ;
    //Other methods should be called via linalg
    py::class_<BasisManager::BasisTransform<Eigen::MatrixXcd>>(m,"BasisManagerTransform")
        .def_readonly("isIdentity",&BasisManager::BasisTransform<Eigen::MatrixXcd>::isIdentity)
        .def("__bool__",[](const BasisManager::BasisTransform<Eigen::MatrixXcd>& self){return static_cast<bool>(self);})
        .def("toNumpy",[](BasisManager::BasisTransform<Eigen::MatrixXcd>& self)
             {
                if (self.transform.isEmpty())
                    throw std::runtime_error("Cannot get the numpy matrix for an empty transform");
                return *self;
             })
        ;
    //End BasisManager

    //Linalg
    //See linalgForwardDeclarations.h for typedefs
    //See PythonLinalg.cpp for horrible templates
    declare_linalg(m);




    //EndLinalg

    //Base classes
    py::class_<HamiltonianBase, std::shared_ptr<HamiltonianBase>>(m, "HamiltonianBase")
        .def("getHamMatrix", &HamiltonianBase::getHamMatrix)
        .def("getHilbertSpaceSize",&HamiltonianBase::getHilbertSpaceSize);

    py::class_<SelfEnergyBase<SelfEnergyMatrix>, std::shared_ptr<SelfEnergyBase<SelfEnergyMatrix>>>(m, "SelfEnergyBase")
        .def("getSigmaIn", py::overload_cast<numType,SelfEnergyMatrix&>(&SelfEnergyBase<SelfEnergyMatrix>::getSigmaIn))
        .def("getSigmaOut", py::overload_cast<numType,SelfEnergyMatrix&>(&SelfEnergyBase<SelfEnergyMatrix>::getSigmaOut))
        .def("getSigmaR", py::overload_cast<complexType,SelfEnergyMatrix&>(&SelfEnergyBase<SelfEnergyMatrix>::getSigmaR))
        .def("getSigmaA", py::overload_cast<complexType,SelfEnergyMatrix&>(&SelfEnergyBase<SelfEnergyMatrix>::getSigmaA))
        .def("recomputeSelfEnergies", &SelfEnergyBase<SelfEnergyMatrix>::recomputeSelfEnergies)
        .def_readwrite("name", &SelfEnergyBase<SelfEnergyMatrix>::m_prettyName);
    //End base Classes
    //SecQuantHam module
    py::module secQuantHam_M = m.def_submodule("SecondQuantisedHamiltonian");
    py::class_<SecondQuantisedHamiltonian, HamiltonianBase, std::shared_ptr<SecondQuantisedHamiltonian>>(secQuantHam_M, "SecondQuantisedHamiltonian")
        .def(py::init<const SecondQuantisedHamiltonian::Parameters &>())
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def("getHamMatrix", &SecondQuantisedHamiltonian::getHamMatrix)
        .def("getPhononCorrelationMatrix", &SecondQuantisedHamiltonian::getPhononCorrelationMatrix)
        .def("getPhononGTensor", &SecondQuantisedHamiltonian::getPhononGTensor)
        .def("loadParameters",&SecondQuantisedHamiltonian::loadParameters)
        .def("loadFile",&SecondQuantisedHamiltonian::loadFile)
        // .def("save",&SecondQuantisedHamiltonian::save)
        ;

    py::class_<SparseTensor<3,complexType>>(m, "SparseTensor_3_Complex")
        .def(py::init<>())
        .def("toString",&SparseTensor<3,complexType>::toString, py::arg("indexToOrderBy") = 0);

    py::class_<SecondQuantisedHamiltonian::Parameters>(secQuantHam_M, "Parameters")
        .def(py::init<>())
        .def(py::init<SecondQuantisedHamiltonian::Parameters&>())
        .def_readwrite("U_0Alpha", &SecondQuantisedHamiltonian::Parameters::U_0Alpha,"on site interaction")
        .def_readwrite("U_0Beta", &SecondQuantisedHamiltonian::Parameters::U_0Beta,"on site interaction")
        .def_readwrite("U_1", &SecondQuantisedHamiltonian::Parameters::U_1,"on site interaction phonon")
        .def_readwrite("t_0", &SecondQuantisedHamiltonian::Parameters::t_0,"nearest neighbour hopping")
        .def_readwrite("lambda_0", &SecondQuantisedHamiltonian::Parameters::lambda_0,"phonon assisted hopping")
        .def_readwrite("t_1", &SecondQuantisedHamiltonian::Parameters::t_1,"phonon assisted next nearest neighbour hopping (Spin orbit)")
        .def_readwrite("lambda_1", &SecondQuantisedHamiltonian::Parameters::lambda_1,"ratio of me")
        .def_readwrite("meff",&SecondQuantisedHamiltonian::Parameters::meff,"radius of loops")
        .def_readwrite("R",&SecondQuantisedHamiltonian::Parameters::R,"radius of loops")
        .def_readwrite("P",&SecondQuantisedHamiltonian::Parameters::P,"pitch of loops")
        .def_readwrite("N",&SecondQuantisedHamiltonian::Parameters::N,"number of loops")
        .def_readwrite("M",&SecondQuantisedHamiltonian::Parameters::M,"sites per loop")
        .def_readwrite("HilbertSpaceSize",&SecondQuantisedHamiltonian::Parameters::HilbertSpaceSize,"Dimension Of Matrices")
        .def_readwrite("numberOfElectrons",&SecondQuantisedHamiltonian::Parameters::numberOfElectrons,"Number of electrons, only 1 is supported");

    //End SecQuantHam module
    //Fock Hamiltonian module
    py::module FockHam_M = m.def_submodule("FockHamiltonian");
    py::class_<FockHamiltonian<SelfEnergyMatrix>, SelfEnergyBase<SelfEnergyMatrix>, std::shared_ptr<FockHamiltonian<SelfEnergyMatrix>>>(FockHam_M, "FockHamiltonian")
        .def(py::init<const ComplexDualSelfAdjointMatrix&,const ComplexDirectSelfAdjointMatrix&, const  SparseTensor<4,complexType>& >())
        .def(py::init<const ComplexDualSelfAdjointMatrix&,const ComplexDirectSelfAdjointMatrix&, std::function<ComplexDirectSelfAdjointMatrix(ComplexDualSelfAdjointMatrix)>>())
        .def(py::init<const ComplexDualSelfAdjointMatrix&,const ComplexDirectSelfAdjointMatrix&, const  SparseTensor<4,complexType>&, std::function<ComplexDirectSelfAdjointMatrix(ComplexDualSelfAdjointMatrix)>>())
        .def("setGreensFunction", &FockHamiltonian<SelfEnergyMatrix>::setGreensFunction)
        .def("getGn", &FockHamiltonian<SelfEnergyMatrix>::getGn)
        .def("setIntegrationParameters", &FockHamiltonian<SelfEnergyMatrix>::setIntegrationParameters)
        .def("getErrorEstimate", &FockHamiltonian<SelfEnergyMatrix>::getErrorEstimate)
        .def("resetDIISCache",&FockHamiltonian<SelfEnergyMatrix>::resetDIISCache)
        .def("setUseEDIIS",&FockHamiltonian<SelfEnergyMatrix>::setUseEDIIS)
        .def("setCDIISInitialAlpha",&FockHamiltonian<SelfEnergyMatrix>::setCDIISInitialAlpha)
        .def("getSigmaR",[](FockHamiltonian<SelfEnergyMatrix>& self){SelfEnergyMatrix s; self.getSigmaR(s); return s;})
        .def("setBasis",py::overload_cast<const indexBasisT&>(&FockHamiltonian<SelfEnergyMatrix>::setBasis))
        .def("setBasis",py::overload_cast<const std::string&>(&FockHamiltonian<SelfEnergyMatrix>::setBasis))
        .def("recomputeSelfEnergies",&FockHamiltonian<SelfEnergyMatrix>::recomputeSelfEnergies)
        // .def("save",&FockHamiltonian<SelfEnergyMatrix>::save)
        ;

    py::class_<SparseTensor<4,complexType>>(m, "SparseTensor_4_Complex")
        .def(py::init<>())
        .def("setSize",&SparseTensor<4,complexType>::setSize)
        .def("getSize",&SparseTensor<4,complexType>::getSize)
        .def("setCoeff",    [](SparseTensor<4,complexType> &me, const std::vector<idxType>& indexes, complexType value) // Python suffered with the passing array by reference.
                             {
                                 SparseTensor<4,complexType>::indexArrayType idxs;
                                 assert(indexes.size() >= 4);
                                 for (noIdxType i = 0; i < 4; i++)
                                     idxs[i] = indexes[i];
                                 me.coeffRef(idxs) = value;
                             })
        .def("setCoeffs",[](SparseTensor<4,complexType> &me, const py::array_t<double> &array)
                            {
                                auto r = array.unchecked<4>();//Throws if NDim != 4
                                SparseTensor<4,complexType>::indexArrayType idxs;
                                releaseAssert(r.shape(0) == r.shape(1) && r.shape(0) == r.shape(2) && r.shape(0) == r.shape(3), "Must give a square tensor");
                                me.setSize(r.shape(0));
                                for (py::ssize_t i = 0; i < r.shape(0); i++)
                                {
                                    for (py::ssize_t j = 0; j < r.shape(1); j++)
                                    {
                                        for (py::ssize_t k = 0; k < r.shape(2); k++)
                                        {
                                            for (py::ssize_t l = 0; l < r.shape(3); l++)
                                            {
                                                idxs[0] = i;
                                                idxs[1] = j;
                                                idxs[2] = k;
                                                idxs[3] = l;
                                                me.coeffRef2(idxs,0,true) = r(i,j,k,l);
                                            }
                                        }
                                    }
                                }
                            })
        .def("toString",&SparseTensor<4,complexType>::toString, py::arg("indexToOrderBy") = 0);


    //End Fock Hamiltonian module
    //Greens Function module
    py::module greensFunc_M = m.def_submodule("GreensFunction");

    py::class_<GreensFunction,std::shared_ptr<GreensFunction>> (greensFunc_M, "GreensFunction")
        .def(py::init<>())
        .def("setHamiltonian",&GreensFunction::setHamiltonian)
        .def("setSelfEnergies",&GreensFunction::setSelfEnergies)
        .def("setPerturbativeSelfEnergies",&GreensFunction::setPerturbativeSelfEnergies)
        .def("setRestrictedCalculation",&GreensFunction::setRestrictedCalculation)
        .def("getRestrictedCalculation",&GreensFunction::getRestrictedCalculation)
        .def("getGrAtE",py::overload_cast<numType>(&GreensFunction::getGrAtE,py::const_))
        .def("getGaAtE",py::overload_cast<numType>(&GreensFunction::getGaAtE,py::const_))
        .def("getGnAtE",py::overload_cast<numType>(&GreensFunction::getGnAtE,py::const_))
        .def("getGpAtE",py::overload_cast<numType>(&GreensFunction::getGpAtE,py::const_))
        // .def("getFrozenCoreTransformation",&GreensFunction::getFrozenCoreTransformation)
        .def("getBasis",&GreensFunction::getBasis)
        .def("forceSelfAdjointSelfEnergy",&GreensFunction::forceSelfAdjointSelfEnergy)
        .def_property("mu",&GreensFunction::getMu,&GreensFunction::setMu)
        .def("setSubsystems",&GreensFunction::setSubsystems)
        // .def("save",&GreensFunction::save)
        ;
    //End Greens Function module

    //Phonon Self Energy module
    py::module phononSelfEnergy_M = m.def_submodule("PhononSelfEnergy");

    py::class_<PhononSelfEnergy<SelfEnergyMatrix>, SelfEnergyBase<SelfEnergyMatrix>,std::shared_ptr<PhononSelfEnergy<SelfEnergyMatrix>>> (phononSelfEnergy_M, "PhononSelfEnergy")
        .def(py::init<PhononSelfEnergy<SelfEnergyMatrix>::Parameters>())
        .def("setGreensFunction",&PhononSelfEnergy<SelfEnergyMatrix>::setGreensFunction)
        .def("setPhononCorrelationMatrix",&PhononSelfEnergy<SelfEnergyMatrix>::setPhononCorrelationMatrix)
        .def("setPhononGTensor",&PhononSelfEnergy<SelfEnergyMatrix>::setPhononGTensor)
        .def("recomputeSelfEnergies",&PhononSelfEnergy<SelfEnergyMatrix>::recomputeSelfEnergies)
        .def("getSigmaIn",&PhononSelfEnergy<SelfEnergyMatrix>::getSigmaIn)
        .def("getSigmaOut",&PhononSelfEnergy<SelfEnergyMatrix>::getSigmaOut)
        .def("getSigmaR",&PhononSelfEnergy<SelfEnergyMatrix>::getSigmaR)
        .def("getSigmaA",&PhononSelfEnergy<SelfEnergyMatrix>::getSigmaA)
        // .def("save",&PhononSelfEnergy<SelfEnergyMatrix>::save)
        ;

    py::class_<PhononSelfEnergy<SelfEnergyMatrix>::Parameters> (phononSelfEnergy_M,"Parameters")
        .def(py::init<>())
        .def(py::init<PhononSelfEnergy<SelfEnergyMatrix>::Parameters&>())
        .def_readwrite("omegas",&PhononSelfEnergy<SelfEnergyMatrix>::Parameters::omegas)
        .def_readwrite("lowestEnergy",&PhononSelfEnergy<SelfEnergyMatrix>::Parameters::lowestEnergy)
        .def_readwrite("largestEnergy",&PhononSelfEnergy<SelfEnergyMatrix>::Parameters::largestEnergy)
        .def_readwrite("smallestEnergyFeature",&PhononSelfEnergy<SelfEnergyMatrix>::Parameters::smallestEnergyFeature)
        .def_readwrite("MatrixDimension",&PhononSelfEnergy<SelfEnergyMatrix>::Parameters::MatrixDimension);

    //End Phonon Self Energy module

    //Quantity calc module
    py::module quantityCalc_M = m.def_submodule("QuantityCalc");
    quantityCalc_M.def("autoRange", &autoRange);
    quantityCalc_M.def("interpolateWithSpectralWeight", &interpolateWithSpectralWeight);
    py::class_<CurrentCalc> (quantityCalc_M, "CurrentCalc")
        .def(py::init<CurrentCalc::Parameters>())
        .def("setSelfEnergy",&CurrentCalc::setSelfEnergy)
        .def("setGreensFunction",&CurrentCalc::setGreensFunction)
        .def("getiE",py::overload_cast<numType>(&CurrentCalc::getiE,py::const_))
        .def("getI",&CurrentCalc::getI)
        .def("getiE",py::overload_cast<std::vector<numType>>(&CurrentCalc::getiE,py::const_))
        // .def("save",&CurrentCalc::save)
        .def_readwrite("selfEnergyName",&CurrentCalc::m_selfEnergyName);



    py::class_<CurrentCalc::Parameters> (quantityCalc_M,"CurrentParameters")
        .def(py::init<>())
        .def(py::init<CurrentCalc::Parameters&>())
        .def_readwrite("numberOfIntegrationPoints",&CurrentCalc::Parameters::numberOfIntegrationPoints)
        .def_readwrite("startE",&CurrentCalc::Parameters::startE)
        .def_readwrite("endE",&CurrentCalc::Parameters::endE);

    py::class_<DivCurrentCalc> (quantityCalc_M, "DivCurrentCalc")
        .def(py::init<DivCurrentCalc::Parameters>())
        .def("setGreensFunction",&DivCurrentCalc::setGreensFunction)
        .def("getAtE",py::overload_cast<std::vector<numType>>(&DivCurrentCalc::getAtE,py::const_))
        .def("get",&DivCurrentCalc::get)
        // .def("save",&DivCurrentCalc::save)
        ;

    py::class_<DivCurrentCalc::Parameters> (quantityCalc_M,"DivCurrentParameters")
        .def(py::init<>())
        .def(py::init<DivCurrentCalc::Parameters&>())
        .def_readwrite("numberOfIntegrationPoints",&DivCurrentCalc::Parameters::numberOfIntegrationPoints)
        .def_readwrite("startE",&DivCurrentCalc::Parameters::startE)
        .def_readwrite("endE",&DivCurrentCalc::Parameters::endE);

    py::class_<ThermalEnergy> (quantityCalc_M, "ThermalEnergy")
        .def(py::init<ThermalEnergy::Parameters>())
        .def("setGreensFunction",&ThermalEnergy::setGreensFunction)
        .def("getAtE",py::overload_cast<numType>(&ThermalEnergy::getAtE,py::const_))
        .def("getAtE",py::overload_cast<numType,const std::string&>(&ThermalEnergy::getAtE,py::const_))
        .def("getAtE",py::overload_cast<std::vector<numType>,const std::string&>(&ThermalEnergy::getAtE,py::const_))
        .def("getEnergy",py::overload_cast<>(&ThermalEnergy::getEnergy,py::const_))
        .def("getEnergy",py::overload_cast<const std::string&>(&ThermalEnergy::getEnergy,py::const_));

    py::class_<ThermalEnergy::Parameters> (quantityCalc_M,"ThermalEnergyParameters")
        .def(py::init<>())
        .def(py::init<ThermalEnergy::Parameters&>())
        .def_readwrite("numberOfIntegrationPoints",&ThermalEnergy::Parameters::numberOfIntegrationPoints)
        .def_readwrite("startE",&ThermalEnergy::Parameters::startE)
        .def_readwrite("endE",&ThermalEnergy::Parameters::endE)
        .def_readwrite("extraEnergy",&ThermalEnergy::Parameters::extraEnergy);

    py::class_<ElectronDensityCalc> (quantityCalc_M, "ElectronDensityCalc")
        .def(py::init<ElectronDensityCalc::Parameters>())
        .def("setGreensFunction",&ElectronDensityCalc::setGreensFunction)
        .def("getnEr",py::overload_cast<numType>(&ElectronDensityCalc::getnEr,py::const_))
        .def("getnEr",py::overload_cast<numType,const std::string&>(&ElectronDensityCalc::getnEr,py::const_))
        .def("getGn",&ElectronDensityCalc::getGn,py::arg("perturb") = false, py::arg("EStart") = negInf, py::arg("EEnd") = posInf)
        .def("getNonEqGn", &ElectronDensityCalc::getNonEqGn)
        .def("getnR",py::overload_cast<const std::string&>(&ElectronDensityCalc::getnR,py::const_))
        .def("getnR",py::overload_cast<>(&ElectronDensityCalc::getnR,py::const_))
        .def("getnS",py::overload_cast<>(&ElectronDensityCalc::getnS,py::const_))
        .def("getnS",py::overload_cast<const std::string&>(&ElectronDensityCalc::getnS,py::const_))
        .def("getnEr",py::overload_cast<std::vector<numType>>(&ElectronDensityCalc::getnEr,py::const_))
        .def("getnEr",py::overload_cast<std::vector<numType>,const std::string&>(&ElectronDensityCalc::getnEr,py::const_))
        // .def("save",&ElectronDensityCalc::save)
        ;

    py::class_<ElectronDensityCalc::Parameters> (quantityCalc_M,"ElectronDensityParameters")
        .def(py::init<>())
        .def(py::init<ElectronDensityCalc::Parameters&>())
        .def_readwrite("numberOfIntegrationPoints",&ElectronDensityCalc::Parameters::numberOfIntegrationPoints)
        .def_readwrite("startE",&ElectronDensityCalc::Parameters::startE)
        .def_readwrite("endE",&ElectronDensityCalc::Parameters::endE);

    py::class_<SpectralFunction> (quantityCalc_M, "SpectralDensityCalc")
        .def(py::init<SpectralFunction::Parameters>())
        .def("setGreensFunction",&SpectralFunction::setGreensFunction)
        .def("getAEr",py::overload_cast<numType>(&SpectralFunction::getAEr,py::const_))
        .def("getAEr",py::overload_cast<numType,const std::string&>(&SpectralFunction::getAEr,py::const_))
        .def("getAR",py::overload_cast<const std::string&>(&SpectralFunction::getAR,py::const_))
        .def("getAR",py::overload_cast<>(&SpectralFunction::getAR,py::const_))
        .def("getAEr",py::overload_cast<std::vector<numType>>(&SpectralFunction::getAEr,py::const_))
        .def("getAEr",py::overload_cast<std::vector<numType>,const std::string&>(&SpectralFunction::getAEr,py::const_))
        // .def("save",&SpectralFunction::save)
        ;

    py::class_<SpectralFunction::Parameters> (quantityCalc_M,"SpectralDensityCalcParameters")
        .def(py::init<>())
        .def(py::init<SpectralFunction::Parameters&>())
        .def_readwrite("numberOfIntegrationPoints",&SpectralFunction::Parameters::numberOfIntegrationPoints)
        .def_readwrite("startE",&SpectralFunction::Parameters::startE)
        .def_readwrite("endE",&SpectralFunction::Parameters::endE);

    py::class_<OperatorExpecationValue>(quantityCalc_M,"OperatorExpectationValue")
        .def(py::init<>())
        .def("setGreensFunction",&OperatorExpecationValue::setGreensFunction)
        .def("getElecValE",&OperatorExpecationValue::getElecValE)
        .def("getHoleValE",&OperatorExpecationValue::getHoleValE);

    py::class_<TransmissionFunc>(quantityCalc_M,"TransmissionFunc")
        .def(py::init<>())
        .def("setGreensFunction",&TransmissionFunc::setGreensFunction)
        .def("getTransmissionAtEBetween",&TransmissionFunc::getTransmissionAtEBetween);

    //End Quantity calc module

    //SCF Solver module
    py::module SCFSolver_M = m.def_submodule("SCFSolver");
    py::class_<SCFSolver> (SCFSolver_M, "SCFSolver")
        .def(py::init<>())
        .def("setGreensFunction",&SCFSolver::setGreensFunction)
        .def("setSelfConsistentSelfEnergy",&SCFSolver::setSelfConsistentSelfEnergy)
        .def("setCallBackFunction",&SCFSolver::setCallBackFunction)
        .def("run",&SCFSolver::run, py::arg("Convergence Function"), py::arg("numberOfTimes") = -1,py::arg("TStart") = T, py::arg("TEnd") = T, py::arg("TStep") = 0)
        // .def("save",&SCFSolver::save)
        ;
    //End SCF Solver module

    //Simple Lead self energy module
    py::module simpleLeadSelfEnergy_M = m.def_submodule("SimpleLeadSelfEnergy");
    py::class_<SimpleLeadSelfEnergy<SelfEnergyMatrix>,SelfEnergyBase<SelfEnergyMatrix>,std::shared_ptr<SimpleLeadSelfEnergy<SelfEnergyMatrix>>> (simpleLeadSelfEnergy_M, "SimpleLeadSelfEnergy")
        .def(py::init<SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters>())
        .def("getSigmaIn",py::overload_cast<numType,SelfEnergyMatrix&>(&SimpleLeadSelfEnergy<SelfEnergyMatrix>::getSigmaIn))
        .def("getSigmaOut",py::overload_cast<numType,SelfEnergyMatrix&>(&SimpleLeadSelfEnergy<SelfEnergyMatrix>::getSigmaOut))
        .def("getSigmaR",py::overload_cast<complexType,SelfEnergyMatrix&>(&SimpleLeadSelfEnergy<SelfEnergyMatrix>::getSigmaR))
        .def("getSigmaA",py::overload_cast<complexType,SelfEnergyMatrix&>(&SimpleLeadSelfEnergy<SelfEnergyMatrix>::getSigmaA))
        .def("setChemicalPotential",py::overload_cast<numType>(&SimpleLeadSelfEnergy<SelfEnergyMatrix>::setChemicalPotential))
        .def("setGreensFunction",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::setGreensFunction)
        .def("setT",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::setT)
        // .def("save",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::save)
        ;

    py::class_<SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters>(simpleLeadSelfEnergy_M,"Parameters")
        .def(py::init<>())
        .def(py::init<SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters&>())
        .def_readwrite("alpha1",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::alpha1)
        .def_readwrite("t",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::t)
        .def_readwrite("mu",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::mu)
        .def_readwrite("muMax",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::muMax)
        .def_readwrite("hasBottom",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::hasBottom)
        .def_readwrite("bottomOfBand",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::bottomOfBand)
        .def_readwrite("adjustMuToElectrons",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::adjustMuToElectrons)
        .def_readwrite("numberOfElectrons",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::numberOfElectrons)
        .def_readwrite("magnetisationDir",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::magnetisationDir)
        .def_readwrite("upSpin",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::upSpin)
        .def_readwrite("couplingPoints",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::couplingPoints)
        .def_readwrite("MatrixDimension",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::MatrixDimension)
        // .def_readwrite("hasCProj",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::hasCProj)
        // .def_readwrite("CProj",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::CProj)
        // .def_readwrite("CInvProjDirecttoDirect",&SimpleLeadSelfEnergy<SelfEnergyMatrix>::Parameters::CInvProjDirecttoDirect)
        ;
    //End Simple Lead self energy module
    // Matrix Self Energy
    py::module matrixSelfEnergy_M = m.def_submodule("MatrixSelfEnergy");
    py::class_<MatrixSelfEnergy<SelfEnergyMatrix,true>,HamiltonianBase, SelfEnergyBase<SelfEnergyMatrix>,std::shared_ptr<MatrixSelfEnergy<SelfEnergyMatrix>>> (matrixSelfEnergy_M, "SelfAdjointMatrixSelfEnergy")
        .def(py::init<const ComplexSelfAdjointSparseMatrix&>(),py::arg("Matrix"))
        .def(py::init<const ComplexSelfAdjointMatrix&>(),py::arg("Matrix"))
        .def(py::init<SecondQuantisedHamiltonian&>())
        .def("getSigmaIn",py::overload_cast<numType,SelfEnergyMatrix&>(&MatrixSelfEnergy<SelfEnergyMatrix>::getSigmaIn))
        .def("getSigmaOut",py::overload_cast<numType,SelfEnergyMatrix&>(&MatrixSelfEnergy<SelfEnergyMatrix>::getSigmaOut))
        .def("getSigmaR",py::overload_cast<complexType,SelfEnergyMatrix&>(&MatrixSelfEnergy<SelfEnergyMatrix>::getSigmaR))
        .def("getSigmaA",py::overload_cast<complexType,SelfEnergyMatrix&>(&MatrixSelfEnergy<SelfEnergyMatrix>::getSigmaA))
        .def("setBasis",py::overload_cast<const std::string&>(&MatrixSelfEnergy<SelfEnergyMatrix>::setBasis))
        .def("setBasis",py::overload_cast<const indexBasisT&>(&MatrixSelfEnergy<SelfEnergyMatrix>::setBasis))
        // .def("save",&MatrixSelfEnergy<SelfEnergyMatrix>::save)
        ;

    py::class_<MatrixSelfEnergy<SelfEnergyMatrix,false>,HamiltonianBase, SelfEnergyBase<SelfEnergyMatrix>,std::shared_ptr<MatrixSelfEnergy<SelfEnergyMatrix,false>> > (matrixSelfEnergy_M, "MatrixSelfEnergy")
        .def(py::init<const ComplexSparseMatrix&>(),py::arg("Matrix"))
        .def(py::init<const ComplexMatrix&>(),py::arg("Matrix"))
        .def(py::init<SecondQuantisedHamiltonian&>())
        .def("getSigmaIn",py::overload_cast<numType,SelfEnergyMatrix&>(&MatrixSelfEnergy<SelfEnergyMatrix,false>::getSigmaIn))
        .def("getSigmaOut",py::overload_cast<numType,SelfEnergyMatrix&>(&MatrixSelfEnergy<SelfEnergyMatrix,false>::getSigmaOut))
        .def("getSigmaR",py::overload_cast<complexType,SelfEnergyMatrix&>(&MatrixSelfEnergy<SelfEnergyMatrix,false>::getSigmaR))
        .def("getSigmaA",py::overload_cast<complexType,SelfEnergyMatrix&>(&MatrixSelfEnergy<SelfEnergyMatrix,false>::getSigmaA))
        .def("setBasis",py::overload_cast<const std::string&>(&MatrixSelfEnergy<SelfEnergyMatrix,false>::setBasis))
        .def("setBasis",py::overload_cast<const indexBasisT&>(&MatrixSelfEnergy<SelfEnergyMatrix,false>::setBasis))
        // .def("save",&MatrixSelfEnergy<SelfEnergyMatrix>::save)
        ;
    //End  Matrix Self Energy

    //Explicit lead self energy module
    py::module explicitLeadSelfEnergy_M = m.def_submodule("ExplicitLeadSelfEnergy");
    py::class_<ExplicitLeadSelfEnergy<SelfEnergyMatrix>,SelfEnergyBase<SelfEnergyMatrix>,std::shared_ptr<ExplicitLeadSelfEnergy<SelfEnergyMatrix>>> (explicitLeadSelfEnergy_M, "ExplicitLeadSelfEnergy")
        .def(py::init<ExplicitLeadSelfEnergy<SelfEnergyMatrix>::Parameters>())
        .def("getSigmaIn",py::overload_cast<numType,SelfEnergyMatrix&>(&ExplicitLeadSelfEnergy<SelfEnergyMatrix>::getSigmaIn))
        .def("getSigmaOut",py::overload_cast<numType,SelfEnergyMatrix&>(&ExplicitLeadSelfEnergy<SelfEnergyMatrix>::getSigmaOut))
        .def("getSigmaR",py::overload_cast<complexType,SelfEnergyMatrix&>(&ExplicitLeadSelfEnergy<SelfEnergyMatrix>::getSigmaR))
        .def("getSigmaA",py::overload_cast<complexType,SelfEnergyMatrix&>(&ExplicitLeadSelfEnergy<SelfEnergyMatrix>::getSigmaA))
        // .def("save",&ExplicitLeadSelfEnergy<SelfEnergyMatrix>::save)
        ;

    py::class_<ExplicitLeadSelfEnergy<SelfEnergyMatrix>::Parameters>(explicitLeadSelfEnergy_M,"Parameters")
        .def(py::init<>())
        .def(py::init<ExplicitLeadSelfEnergy<SelfEnergyMatrix>::Parameters&>())
        .def_readwrite("a",&ExplicitLeadSelfEnergy<SelfEnergyMatrix>::Parameters::a)
        .def_readwrite("alpha1",&ExplicitLeadSelfEnergy<SelfEnergyMatrix>::Parameters::alpha1)
        .def_readwrite("t",&ExplicitLeadSelfEnergy<SelfEnergyMatrix>::Parameters::t)
        .def_readwrite("bottomOfBand",&ExplicitLeadSelfEnergy<SelfEnergyMatrix>::Parameters::bottomOfBand)
        .def_readwrite("mu",&ExplicitLeadSelfEnergy<SelfEnergyMatrix>::Parameters::mu)
        .def_readwrite("couplingPoints",&ExplicitLeadSelfEnergy<SelfEnergyMatrix>::Parameters::couplingPoints)
        .def_readwrite("MatrixDimension",&ExplicitLeadSelfEnergy<SelfEnergyMatrix>::Parameters::MatrixDimension)
        ;

    //End Explicit lead self energy module

    //InterfaceSelfEnergy
    py::module InterfaceSelfEnergy_M = m.def_submodule("InterfaceSelfEnergy");
    py::class_<InterfaceSelfEnergy<SelfEnergyMatrix>,SelfEnergyBase<SelfEnergyMatrix>,std::shared_ptr<InterfaceSelfEnergy<SelfEnergyMatrix>>> (InterfaceSelfEnergy_M, "InterfaceSelfEnergy")
        .def(py::init<InterfaceSelfEnergy<SelfEnergyMatrix>::Parameters>())
        .def("getSigmaIn",py::overload_cast<numType,SelfEnergyMatrix&>(&InterfaceSelfEnergy<SelfEnergyMatrix>::getSigmaIn))
        .def("getSigmaOut",py::overload_cast<numType,SelfEnergyMatrix&>(&InterfaceSelfEnergy<SelfEnergyMatrix>::getSigmaOut))
        .def("getSigmaR",py::overload_cast<complexType,SelfEnergyMatrix&>(&InterfaceSelfEnergy<SelfEnergyMatrix>::getSigmaR))
        .def("getSigmaA",py::overload_cast<complexType,SelfEnergyMatrix&>(&InterfaceSelfEnergy<SelfEnergyMatrix>::getSigmaA))
        // .def("save",&ExplicitLeadSelfEnergy<SelfEnergyMatrix>::save)
        ;

    py::class_<InterfaceSelfEnergy<SelfEnergyMatrix>::Parameters>(InterfaceSelfEnergy_M,"Parameters")
        .def(py::init<>())
        .def(py::init<InterfaceSelfEnergy<SelfEnergyMatrix>::Parameters&>())
        .def_readwrite("m_type",&InterfaceSelfEnergy<SelfEnergyMatrix>::Parameters::m_type)
        .def_readwrite("mu",&InterfaceSelfEnergy<SelfEnergyMatrix>::Parameters::mu)
        //For the infinite tight binding lead
        .def_readwrite("t",&InterfaceSelfEnergy<SelfEnergyMatrix>::Parameters::t)
        .def_readwrite("U0",&InterfaceSelfEnergy<SelfEnergyMatrix>::Parameters::U0)
        .def_readwrite("a",&InterfaceSelfEnergy<SelfEnergyMatrix>::Parameters::a)
        //For the Continuum Lead
        // numType U0 = 0; - Reused. In order to match the parameters for the tightBindingInfinite. U0->U0+2t
        //numType a = 1; -Reused, Here it denotes where the continuum lead interacts from. I.e.  -----a---| . . . . |---a-----
        .def_readwrite("m",&InterfaceSelfEnergy<SelfEnergyMatrix>::Parameters::m)
        .def_readwrite("couplingPoints",&InterfaceSelfEnergy<SelfEnergyMatrix>::Parameters::couplingPoints)
        .def_readwrite("MagnetisationDirection",&InterfaceSelfEnergy<SelfEnergyMatrix>::Parameters::MagnetisationDirection)
        //For the Green's function version.
        .def_readwrite("Gfunc",&InterfaceSelfEnergy<SelfEnergyMatrix>::Parameters::Gfunc)
        .def_readwrite("SECouple",&InterfaceSelfEnergy<SelfEnergyMatrix>::Parameters::SECouple)
        ;
    //End InterfaceSelfEnergy
    //config file loader module
    // py::module configFileLoader_M = m.def_submodule("ConfigFileLoader");
    // py::class_<ConfigFileLoader> (configFileLoader_M, "ConfigFileLoader")
    //     .def(py::init<std::string>())
    //     .def("load",&ConfigFileLoader::load)
    //     .def("success",&ConfigFileLoader::success)
    //     .def("setFileName",&ConfigFileLoader::setFileName)
    //     .def("save",&ConfigFileLoader::save)
    //     .def("getSelfEnergies",&ConfigFileLoader::getSelfEnergies)
    //     .def("getGreensFunction",&ConfigFileLoader::getGreensFunction)
    //     .def("getHam",&ConfigFileLoader::getHam)
    //     .def("getCurrentCalcs",&ConfigFileLoader::getCurrentCalcs)
    //     .def("getElectronDensityCalcs",&ConfigFileLoader::getElectronDensityCalcs)
    //     .def("getSpectraFunctionCalcs",&ConfigFileLoader::getSpectraFunctionCalcs)
    //     .def("getSCF",&ConfigFileLoader::getSCF);




}

