cmake_minimum_required(VERSION 3.20.1)

# Standalone benchmark: builds on its own, without configuring PHARE.
#
#   cmake -S tools/bench/initializer -B /tmp/bench_field_init -DCMAKE_BUILD_TYPE=Release
#   cmake --build /tmp/bench_field_init -j
#   /tmp/bench_field_init/bench_field_init
#
# Only header-only PHARE core is used, so nothing of PHARE needs to be built first.
# googlebench and pybind11 are taken from <phare>/subprojects when present, and cloned into the
# build tree otherwise. Pass -DPHARE_SUBPROJECTS=/path/to/subprojects to point elsewhere.

project(phare_bench_initializer LANGUAGES CXX)

if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
  message(FATAL_ERROR "in-source build would litter the source tree: pass -B <build dir>")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(PHARE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ABSOLUTE)
if (NOT DEFINED PHARE_SUBPROJECTS)
  set(PHARE_SUBPROJECTS ${PHARE_ROOT}/subprojects)
endif()

include(FetchContent)

# --- googlebench ---------------------------------------------------------------------------------
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
if (EXISTS ${PHARE_SUBPROJECTS}/googlebench/CMakeLists.txt)
  add_subdirectory(${PHARE_SUBPROJECTS}/googlebench ${CMAKE_BINARY_DIR}/googlebench)
else()
  FetchContent_Declare(googlebench GIT_REPOSITORY https://github.com/google/benchmark GIT_TAG main GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googlebench)
endif()

# --- pybind11 ------------------------------------------------------------------------------------
if (EXISTS ${PHARE_SUBPROJECTS}/pybind11/CMakeLists.txt)
  add_subdirectory(${PHARE_SUBPROJECTS}/pybind11 ${CMAKE_BINARY_DIR}/pybind11)
else()
  FetchContent_Declare(pybind11 GIT_REPOSITORY https://github.com/pybind/pybind11 GIT_TAG master GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(pybind11)
endif()

# --- cppdict (header only, needed by initializer/data_provider.hpp) -------------------------------
if (EXISTS ${PHARE_SUBPROJECTS}/cppdict/include/dict.hpp)
  set(CPPDICT_INCLUDE ${PHARE_SUBPROJECTS}/cppdict/include)
else()
  FetchContent_Declare(cppdict GIT_REPOSITORY https://github.com/LaboratoryOfPlasmaPhysics/cppdict GIT_TAG master GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(cppdict)
  set(CPPDICT_INCLUDE ${cppdict_SOURCE_DIR}/include)
endif()

# --- the benchmark -------------------------------------------------------------------------------
add_executable(bench_field_init bench_field_init.cpp)

target_include_directories(bench_field_init PRIVATE ${PHARE_ROOT}/src ${PHARE_ROOT} ${CPPDICT_INCLUDE})
target_link_libraries(bench_field_init PRIVATE benchmark pybind11::embed)
target_compile_options(bench_field_init PRIVATE -Wall -Wextra)
