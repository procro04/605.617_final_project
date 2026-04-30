#pragma once

// ---------------------------------------------------------------------------
// CUDA Error Checking
// ---------------------------------------------------------------------------

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
