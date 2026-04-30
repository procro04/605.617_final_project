#pragma once

// ---------------------------------------------------------------------------
// CUDA Error Checking
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>
#include <iostream>

// Wrap every CUDA API call with this to catch errors early.
#define CUDA_CHECK(call)                                                 \
    do                                                                   \
    {                                                                    \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess)                                          \
        {                                                                \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " -- " << cudaGetErrorString(err) << "\n";      \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

inline void print_cuda_device_info()
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found!\n";
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA Device: " << prop.name << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  SMs: " << prop.multiProcessorCount
              << "  Max threads/block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Global memory: " << (prop.totalGlobalMem >> 20) << " MB\n";
}
