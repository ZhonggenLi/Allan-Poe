cmake_minimum_required(VERSION 3.10)
project(hybrid_search LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "native")

# Set output directory for the python module
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)


# Define the source files
set(SOURCES
    src/bindings.cpp
    src/index.cu
    src/query.cu
)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Create the Python extension module
pybind11_add_module(hybrid_search MODULE ${SOURCES})

# Link libraries
target_link_libraries(hybrid_search PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(hybrid_search PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --ftz=true
        --prec-div=false
        --prec-sqrt=false
        -Xptxas=-O3
        -w
    >
)

target_compile_options(hybrid_search PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -w
    >
)