#Tests

set (TestsToRun
  integratorTests.cpp
  explicitleadselfenergyTests.cpp
  fockhamiltonianTests.cpp
  complexmatrixbufferTests.cpp
  einsumTests.cpp
  greensfunctionTests.cpp
  linalgTests.cpp
  matrixselfenergyTests.cpp
  phononselfenergyTests.cpp
  quantitycalcTests.cpp
  secondquantisedhamiltonianTests.cpp
  simpleleadselfenergyTests.cpp
  sparsetensorTests.cpp
  threadpoolTests.cpp
  basisManagerTests.cpp
  infiniteLeadTests.cpp
)
create_test_sourcelist (Tests cppTests.cpp ${TestsToRun})

# add the executable
add_executable (cppTests ${Tests}
    testutil.hpp
    GreensFunctionTestHelper.h
    TabulatedIntegrals.h

)
target_link_options(cppTests PRIVATE $<$<CONFIG:Debug>:LINKER:--gdb-index> )

target_include_directories(cppTests PUBLIC ../)
target_include_directories(cppTests PRIVATE ../third-party/eigen-3.4.0)
target_link_libraries(cppTests cppNEGFLib)

# Add all the ADD_TEST for each test
foreach (t ${TestsToRun})
  get_filename_component (TName ${t} NAME_WE)
  add_test (NAME ${TName} COMMAND cppTests ${TName})
endforeach ()
