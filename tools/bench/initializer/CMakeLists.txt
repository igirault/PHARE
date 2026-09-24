cmake_minimum_required (VERSION 3.20.1)

project(phare_bench_initializer)

add_phare_cpp_benchmark(10 ${PROJECT_NAME} field_init ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${PROJECT_NAME}_field_init PUBLIC pybind11::embed)
