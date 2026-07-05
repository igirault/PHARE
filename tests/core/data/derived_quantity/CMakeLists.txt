cmake_minimum_required (VERSION 3.20.1)

project(test-derived-quantity)

set(SOURCES test_centering.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  ${GTEST_INCLUDE_DIRS}
  )

target_link_libraries(${PROJECT_NAME} PRIVATE
  phare_core
  phare_initializer
  ${GTEST_LIBS})

add_no_mpi_phare_test(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR})


project(test-derived-quantity-interface)
add_executable(${PROJECT_NAME} test_derived_quantity.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  ${GTEST_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE phare_core phare_initializer ${GTEST_LIBS})
add_no_mpi_phare_test(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR})


project(test-derived-scratch)
add_executable(${PROJECT_NAME} test_derived_scratch.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  ${GTEST_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE phare_core phare_initializer ${GTEST_LIBS})
add_no_mpi_phare_test(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR})


project(test-mhd-derived-quantities)
add_executable(${PROJECT_NAME} test_mhd_derived_quantities.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  ${GTEST_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE phare_core phare_initializer ${GTEST_LIBS})
add_no_mpi_phare_test(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR})
