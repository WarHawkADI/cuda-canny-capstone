// Copyright 2026 CUDA Canny Capstone Authors.
//
// Single-threaded CPU reference implementation of the pipeline. It is used
// to verify the GPU output and as the baseline for speedup measurements.

#ifndef SRC_CANNY_CPU_H_
#define SRC_CANNY_CPU_H_

#include "src/canny_types.h"

namespace canny {

// Runs the full pipeline on the CPU. Intermediate stages are stored in
// *result only when keep_stages is true.
void RunCannyCpu(const Image& input, const CannyParams& params,
                 bool keep_stages, CannyResult* result);

}  // namespace canny

#endif  // SRC_CANNY_CPU_H_
