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
endif()
