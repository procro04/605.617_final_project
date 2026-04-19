cmake_minimum_required(VERSION 3.18)
project(occupancy_map LANGUAGES CXX)

# Phase 1: CPU-only build
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(inc/)

add_executable(occupancy_map_cpu src/occupancy_map_cpu.cpp)
target_compile_options(occupancy_map_cpu PRIVATE -O2 -Wall -Wextra)

# Test data generator (for verifying the pipeline without the real dataset)
add_executable(generate_test_data src/generate_test_data.cpp)
target_compile_options(generate_test_data PRIVATE -O2)
# File for combining intel data to look like test data
add_executable(combine_intel_data src/combine_intel_data.cpp)
target_compile_options(combine_intel_data PRIVATE -O2)

# Phase 2: CUDA build
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

add_executable(occupancy_map_cuda src/occupancy_map_cuda.cu)
target_compile_options(occupancy_map_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -O2
>)
set_target_properties(occupancy_map_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"   # Adjust for your GPU
)
