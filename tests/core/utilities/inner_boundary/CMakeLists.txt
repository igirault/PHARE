cmake_minimum_required (VERSION 3.20.1)

project(test-inner_boundary)

function(_add_inner_boundary_test name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${GTEST_INCLUDE_DIRS})
  target_link_libraries(${name} PRIVATE phare_core phare_simulator ${GTEST_LIBS})
  add_no_mpi_phare_test(${name} ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

_add_inner_boundary_test(test-inner_boundary)
_add_inner_boundary_test(test-inner_boundary_mesh_classifier)
