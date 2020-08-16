cmake_minimum_required(VERSION 2.8.12)
project(algorithms)

SET(MKLROOT "/opt/intel/compilers_and_libraries_2020.2.258/mac/mkl")
SET(CMAKE_CXX_FLAGS "-DMKL_LP64 -m64 -I${MKLROOT}/include ${MKLROOT}/lib/libmkl_intel_lp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -lpthread -lm -ldl")

find_package(PythonLibs)
find_package(Eigen3 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(algorithms/eigen)

add_subdirectory(algorithms/pybind11 build)
pybind11_add_module(algorithms algorithms/algorithms.cpp)