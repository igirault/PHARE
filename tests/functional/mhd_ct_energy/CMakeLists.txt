cmake_minimum_required (VERSION 3.20.1)

project(test-mhd_ct_energy)

if(NOT ${PHARE_PROJECT_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  file(GLOB PYFILES "*.py")
  file(COPY ${PYFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HighFive)

  ## These test use dump diagnostics so require HighFive!
  phare_python3_exec(9 ct_energy_wenoz_decay test_wenoz_mode_decay.py ${CMAKE_CURRENT_BINARY_DIR})
  phare_python3_exec(9 ct_energy_dt_limit test_dt_limit_noise.py ${CMAKE_CURRENT_BINARY_DIR})
endif()
