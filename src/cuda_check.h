// Copyright 2026 CUDA Canny Capstone Authors.
//
// Error-checking helpers for the CUDA runtime API.

#ifndef SRC_CUDA_CHECK_H_
#define SRC_CUDA_CHECK_H_

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// Aborts with a readable message when a CUDA runtime call fails. A macro is
// used so that the failing file and line are reported.
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    const cudaError_t cuda_check_status = (call);                          \
    if (cuda_check_status != cudaSuccess) {                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                 \
                   cudaGetErrorName(cuda_check_status), __FILE__, __LINE__, \
                   cudaGetErrorString(cuda_check_status));                 \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

// Checks for errors raised by the most recent kernel launch.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

#endif  // SRC_CUDA_CHECK_H_
