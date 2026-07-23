cmake_minimum_required (VERSION 3.20.1)

project(test-mhd_cloud_shock)

if(NOT ${PHARE_PROJECT_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  file(GLOB PYFILES "*.py")
  file(COPY ${PYFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HighFive AND testMPI)
  ## This test uses dump diagnostics so requires HighFive!
  # exec level 11: 3-level AMR + Hall + hyper-resistivity, excluded by
  # default, opt in via -DPHARE_EXEC_LEVEL_MAX=11 (or higher)
  # 10 ranks, matching mhd_harris_with_boundaries/mhd_blast
  phare_mpi_python3_exec(11 10 test-mhd_cloud_shock cloud_shock.py ${CMAKE_CURRENT_BINARY_DIR})
endif()
