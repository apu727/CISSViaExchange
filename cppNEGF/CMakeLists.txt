cmake_minimum_required(VERSION 3.16)

project(cppNEGF LANGUAGES CXX)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CPPNEGF_SOURCES
    greensfunction.h greensfunction.cpp
    hamiltonianbase.h hamiltonianbase.cpp
    secondquantisedhamiltonian.h secondquantisedhamiltonian.cpp
    selfenergybase.h selfenergybase.cpp
    explicitleadselfenergy.h explicitleadselfenergy.cpp
    simpleleadselfenergy.h simpleleadselfenergy.cpp
    phononselfenergy.h phononselfenergy.cpp
    scfsolver.h scfsolver.cpp
    quantitycalc.h quantitycalc.cpp
    global.h global.cpp
    linalg.h linalg.cpp
    logger.h logger.cpp
    complexmatrixbuffer.h complexmatrixbuffer.cpp
    threadpool.cpp threadpool.h
    configfileloader.h configfileloader.cpp
    sparsetensor.cpp sparsetensor.h
    EinSumTemplate.h

    matrixselfenergy.h matrixselfenergy.cpp
    fockhamiltonian.h fockhamiltonian.cpp
    integrator.h integrator.cpp

    basismanager.h basismanager.cpp
    enums.h
    LinalgOperators.h
    LinalgOperatorDef.h
    linalgForwardDeclarations.h
    interfaceselfenergy.h interfaceselfenergy.cpp
    diis.h diis.cpp
    sopgreensfunction.h sopgreensfunction.cpp
)

add_library(cppNEGFLib STATIC  ${CPPNEGF_SOURCES}

)

#Eigen
target_compile_definitions(cppNEGFLib PRIVATE EIGEN_MPL2_ONLY)

#KissFFT
set(kissfft_DIR ./third-party/kissfft/install/lib/cmake/kissfft/)
find_package(kissfft CONFIG REQUIRED COMPONENTS STATIC double)


target_include_directories(cppNEGFLib PRIVATE ./third-party/eigen-3.4.0)
target_link_libraries(cppNEGFLib PRIVATE kissfft::kissfft-double)
set_property(TARGET cppNEGFLib PROPERTY POSITION_INDEPENDENT_CODE ON)

#OtherFlags
include(GNUInstallDirs)
#set(CMAKE_CXX_FLAGS "-fsanitize=address")
# set(CMAKE_CXX_FLAGS "-Wno-stringop-overflow -fuse-ld=gold")
add_compile_options(-Wno-stringop-overflow -fuse-ld=gold)
add_link_options(-fuse-ld=gold)
if (OPENMP)
    add_compile_options(-fopenmp)
    add_link_options(-fopenmp)
endif()
#Decide Python or Standalone build

set(PYTHON_MODULE_SOURCES PythonModule.cpp
    PythonLinalg.hpp
    pythonLinalg/PythonLinalg1.cpp
    pythonLinalg/PythonLinalg2.cpp
    pythonLinalg/PythonLinalg3.cpp
    pythonLinalg/PythonLinalg4.cpp
    pythonLinalg/PythonLinalg5.cpp
    pythonLinalg/PythonLinalg6.cpp
    pythonLinalg/PythonLinalg7.cpp
    pythonLinalg/PythonLinalg8.cpp
    pythonLinalg/PythonLinalg9.cpp
    pythonLinalg/PythonLinalg10.cpp
    pythonLinalg/PythonLinalg11.cpp
    pythonLinalg/PythonLinalg12.cpp
    pythonLinalg/PythonLinalg13.cpp
    pythonLinalg/PythonLinalg14.cpp
    pythonLinalg/PythonLinalg15.cpp
    pythonLinalg/PythonLinalg16.cpp
    pythonLinalg/PythonLinalg17.cpp
    pythonLinalg/PythonLinalg18.cpp
    pythonLinalg/PythonLinalg19.cpp
    pythonLinalg/PythonLinalg20.cpp
    pythonLinalg/PythonLinalg21.cpp
    pythonLinalg/PythonLinalg22.cpp
    pythonLinalg/PythonLinalg23.cpp
    pythonLinalg/PythonLinalg24.cpp
    pythonLinalg/PythonLinalg25.cpp
    pythonLinalg/PythonLinalg26.cpp
    pythonLinalg/PythonLinalg27.cpp
    pythonLinalg/PythonLinalg28.cpp
    pythonLinalg/PythonLinalg29.cpp
    pythonLinalg/PythonLinalg30.cpp
    pythonLinalg/PythonLinalg31.cpp
    pythonLinalg/PythonLinalg32.cpp
    pythonLinalg/PythonLinalg33.cpp
    pythonLinalg/PythonLinalg34.cpp
    pythonLinalg/PythonLinalg35.cpp
)

if (COMPILE_PYTHON_LIBS)
    set(pybind11_DIR ${CMAKE_SOURCE_DIR}/../share/cmake/pybind11/)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(cppNEGF ${PYTHON_MODULE_SOURCES})
    #TODO figure out Generator expressions to turn this off for debug builds
    set_property(TARGET cppNEGF
        PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
    set_property(TARGET cppNEGF
        PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)


    target_include_directories(cppNEGF PRIVATE ./third-party/eigen-3.4.0)
    target_include_directories(cppNEGF PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(cppNEGF PUBLIC cppNEGFLib)
    target_link_options(cppNEGF PRIVATE $<$<CONFIG:Debug>:LINKER:--gdb-index> )
    # add_dependencies(cppNEGF cppNEGFLib) # workaround because the python module needs ~9GB of ram to compile
    # set_property(SOURCE PythonLinalg1.cpp APPEND PROPERTY OBJECT_DEPENDS PythonModule.cpp.o)
    # set_property(SOURCE PythonLinalg2.cpp APPEND PROPERTY OBJECT_DEPENDS PythonLinalg1.cpp.o)
    # set_property(SOURCE PythonLinalg3.cpp APPEND PROPERTY OBJECT_DEPENDS PythonLinalg2.cpp.o)
    #set_source_files_properties(PythonLinalg.cpp PROPERTIES COMPILE_FLAGS -foptimize-sibling-calls)

    #InstallDir
    install(TARGETS cppNEGF DESTINATION ${CMAKE_SOURCE_DIR}/..) # Set Install prefix to python directory

elseif  (COMPILE_STANDALONE)
    add_executable(cppNEGF main.cpp)
    target_include_directories(cppNEGF PRIVATE ./third-party/eigen-3.4.0)

else()
    message(FATAL_ERROR "Specify COMPILE_STANDALONE or COMPILE_PYTHON_LIBS")
endif()

#target_link_libraries(cppNEGF cppNEGFLib)

#Build Tests
if (BUILD_TESTING)
    add_subdirectory(Tests)
endif()






