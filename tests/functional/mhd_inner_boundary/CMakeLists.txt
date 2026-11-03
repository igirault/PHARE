cmake_minimum_required (VERSION 3.20.1)

project(test-mhd_inner_boundary)

if(NOT ${PHARE_PROJECT_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  file(GLOB PYFILES "*.py")
  file(COPY ${PYFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HighFive)
  phare_mpi_python3_exec(9 4 mhd_inner_boundary_sphere sphere_periodic.py ${CMAKE_CURRENT_BINARY_DIR})
  phare_mpi_python3_exec(9 4 mhd_inner_boundary_sphere_dipole_L2 sphere_periodic.py ${CMAKE_CURRENT_BINARY_DIR} --external-field --levels 2)
  phare_mpi_python3_exec(9 4 mhd_inner_boundary_sphere_restart sphere_periodic.py ${CMAKE_CURRENT_BINARY_DIR} --restart)
  phare_mpi_python3_exec(9 4 mhd_inner_boundary_sphere_under_resolved sphere_periodic.py ${CMAKE_CURRENT_BINARY_DIR} --under-resolved)
  phare_mpi_python3_exec(9 1 mhd_inner_boundary_sphere_decomposition sphere_periodic.py ${CMAKE_CURRENT_BINARY_DIR} --under-resolved --external-field --compare-decomposition)
endif()
