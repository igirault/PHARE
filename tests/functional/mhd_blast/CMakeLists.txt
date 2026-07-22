cmake_minimum_required (VERSION 3.20.1)

project(test-mhd-blast)

if(NOT ${PHARE_PROJECT_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  file(GLOB PYFILES "*.py")
  file(COPY ${PYFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HighFive)

  ## These tests use dump diagnostics so require HighFive!
  ## Heavy level 101: blast1/blast2 are EXPECTED to crash (negative pressure at
  ## low plasma-beta) and must not gate default CI.
  phare_mpi_python3_exec(101 10 test-mhd-blast-reference blast_reference.py ${CMAKE_CURRENT_BINARY_DIR})
  phare_mpi_python3_exec(101 10 test-mhd-blast-blast1 blast_blast1.py ${CMAKE_CURRENT_BINARY_DIR})
  phare_mpi_python3_exec(101 10 test-mhd-blast-blast2 blast_blast2.py ${CMAKE_CURRENT_BINARY_DIR})

  ## blast1/blast2 crash by design (negative pressure at low beta -> NaN abort).
  ## Mark them expected-failure so CTest is GREEN when they crash and RED if one
  ## ever completes (e.g. a Release build with the NaN check compiled out). Guard
  ## matches phare_mpi_python3_exec's level condition so the tests exist here.
  ## Note: the clean NaN abort requires a Debug build or -DPHARE_FORCE_DEBUG_DO.
  if(101 GREATER_EQUAL ${PHARE_EXEC_LEVEL_MIN} AND 101 LESS_EQUAL ${PHARE_EXEC_LEVEL_MAX})
    set_tests_properties(
      py3_test-mhd-blast-blast1_mpi_n_10
      py3_test-mhd-blast-blast2_mpi_n_10
      PROPERTIES WILL_FAIL TRUE)
  endif()
endif()
