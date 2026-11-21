// Copyright 2026 CUDA Canny Capstone Authors.
//
// CUDA implementation of the edge detection pipeline. Each GpuCanny object
// owns a private CUDA stream plus device and pinned host buffers, so several
// objects driven from different host threads run concurrently on the GPU.

#ifndef SRC_CANNY_GPU_H_
#define SRC_CANNY_GPU_H_

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "src/canny_types.h"

namespace canny {

// Copies the Gaussian weights into __constant__ memory. Must be called once
// before any GpuCanny::Run, and not concurrently with running pipelines.
void UploadGaussianKernel(const std::vector<float>& weights);

class GpuCanny {
 public:
  explicit GpuCanny(const CannyParams& params);
  ~GpuCanny();

  GpuCanny(const GpuCanny&) = delete;
  GpuCanny& operator=(const GpuCanny&) = delete;

  // Runs the full pipeline for one image on this object's stream and blocks
  // until the result is on the host.
  void Run(const Image& input, bool keep_stages, CannyResult* result,
           GpuTiming* timing);

 private:
  void EnsureCapacity(int64_t pixels, int channels);
  void ReleaseBuffers();

  // CUDA events recorded at each stage boundary for the timing breakdown.
  enum Mark {
    kMarkStart = 0,
    kMarkUploaded,
    kMarkGray,
    kMarkBlurred,
    kMarkGradient,
    kMarkThresholded,
    kMarkComputed,
    kMarkDone,
    kNumMarks
  };

  CannyParams params_;
  int radius_ = 0;
  int sm_count_ = 1;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t events_[kNumMarks] = {};

  int64_t capacity_pixels_ = 0;
  int capacity_channels_ = 0;

  // Pinned host staging buffers (enable truly asynchronous copies).
  uint8_t* host_input_ = nullptr;
  uint8_t* host_edges_ = nullptr;
  int* host_changed_ = nullptr;

  // Device buffers.
  uint8_t* d_input_ = nullptr;
  uint8_t* d_gray_ = nullptr;
  uint32_t* d_histogram_ = nullptr;
  uint8_t* d_lut_ = nullptr;
  float* d_blur_rows_ = nullptr;
  float* d_blurred_ = nullptr;
  float* d_magnitude_ = nullptr;
  uint8_t* d_direction_ = nullptr;
  float* d_suppressed_ = nullptr;
  float* d_max_ = nullptr;
  uint8_t* d_labels_ = nullptr;
  int* d_changed_ = nullptr;
  // Ping-pong "tile changed in previous pass" flags for hysteresis.
  uint8_t* d_tile_active_[2] = {nullptr, nullptr};
  int64_t capacity_tiles_ = 0;
};

}  // namespace canny

#endif  // SRC_CANNY_GPU_H_
