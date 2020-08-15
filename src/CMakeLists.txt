cmake_minimum_required(VERSION 2.8.12)
project(algorithms)

find_package(PythonLibs)
find_package(Eigen3 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(algorithms/eigen)

add_subdirectory(algorithms/pybind11 build)

pybind11_add_module(algorithms algorithms/algorithms.cpp)
