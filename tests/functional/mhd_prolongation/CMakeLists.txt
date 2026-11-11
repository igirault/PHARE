cmake_minimum_required (VERSION 3.20.1)

project(test-mhd_prolongation)

if(NOT ${PHARE_PROJECT_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  file(GLOB PYFILES "*.py")
  file(COPY ${PYFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HighFive)
  phare_python3_exec(9 mhd_sharp_jump_refinement test_sharp_jump_refinement.py ${CMAKE_CURRENT_BINARY_DIR})
endif()
