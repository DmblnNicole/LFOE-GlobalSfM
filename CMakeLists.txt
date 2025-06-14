cmake_minimum_required(VERSION 3.10)

project(glomap_with_filtering)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external/glomap)

include_directories(
    ${CMAKE_SOURCE_DIR}/src/filter 
    ${CMAKE_SOURCE_DIR}/src/python)

find_package(pybind11 REQUIRED)
find_package(colmap REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(glomap_filter
    src/main/glomap_filter.cc
    src/filter/global_mapper_filter.cc
    src/main/run_mapper.cc
    src/filter/filter.cc
)

target_link_libraries(glomap_filter PRIVATE glomap pybind11::embed colmap::colmap yaml-cpp)