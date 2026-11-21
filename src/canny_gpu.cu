// Copyright 2026 CUDA Canny Capstone Authors.
//
// CUDA kernels and host orchestration for the edge detection pipeline:
//
//   RGB -> gray -> [histogram equalization] -> separable Gaussian blur ->
//   Sobel gradient -> non-maximum suppression (+ block max reduction) ->
//   double threshold -> iterative hysteresis -> binary edge map
//
// GPU techniques used: grid-stride loops, shared-memory tiling with halos,
// __constant__ memory for filter weights, shared-memory atomics for the
// histogram, a work-efficient block scan for the CDF, warp-shuffle
// reductions with a global atomicMax, block-local iterative propagation with
// __syncthreads_or, pinned host memory and per-object CUDA streams.

#include "src/canny_gpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <vector>

#include "src/cuda_check.h"
#include "src/pixel_ops.h"

namespace canny {
namespace {

__constant__ float c_gaussian[2 * kMaxGaussianRadius + 1];

constexpr int kThreads1d = 256;
constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
// Hysteresis tile sides in pixels. Large tiles need fewer relaunches and
// skip more work on huge sparse images; small tiles keep more SMs busy on
// ordinary images (chosen from measurements, see README).
constexpr int kHysteresisSmallTile = 32;
constexpr int kHysteresisLargeTile = 64;
constexpr int64_t kLargeTileMinPixels = 16 << 20;
constexpr int kHysteresisThreads = 16;  // Threads per block side.
constexpr int kHysteresisPassesPerCheck = 4;
constexpr int kMaxHysteresisPasses = 1 << 20;

__device__ __forceinline__ int ClampIndex(int value, int hi) {
  return min(max(value, 0), hi);
}

// ---------------------------------------------------------------------------
// Stage 1: color conversion and histogram equalization.
// ---------------------------------------------------------------------------

__global__ void GrayscaleKernel(const uint8_t* __restrict__ input,
                                int channels, int64_t n,
                                uint8_t* __restrict__ gray) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    gray[i] = PixelToGray(input + i * channels, channels);
  }
}

// Privatized histogram: each block accumulates into shared memory first so
// that global atomics are only issued once per bin per block.
__global__ void HistogramKernel(const uint8_t* __restrict__ gray, int64_t n,
                                uint32_t* __restrict__ histogram) {
  __shared__ uint32_t local[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) local[i] = 0;
  __syncthreads();
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    atomicAdd(&local[gray[i]], 1u);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    if (local[i] != 0) atomicAdd(&histogram[i], local[i]);
  }
}

// One block of 256 threads: inclusive Hillis-Steele scan of the histogram
// into a CDF, then the equalization lookup table.
__global__ void EqualizeLutKernel(const uint32_t* __restrict__ histogram,
                                  uint32_t total, uint8_t* __restrict__ lut) {
  __shared__ uint32_t scan[256];
  __shared__ uint32_t cdf_min;
  const int t = threadIdx.x;
  const uint32_t count = histogram[t];
  scan[t] = count;
  if (t == 0) cdf_min = UINT_MAX;
  __syncthreads();
  for (int offset = 1; offset < 256; offset <<= 1) {
    const uint32_t add = t >= offset ? scan[t - offset] : 0u;
    __syncthreads();
    scan[t] += add;
    __syncthreads();
  }
  if (count > 0) atomicMin(&cdf_min, scan[t]);
  __syncthreads();
  lut[t] = EqualizeBin(scan[t], cdf_min, total);
}

__global__ void ApplyLutKernel(const uint8_t* __restrict__ lut, int64_t n,
                               uint8_t* __restrict__ gray) {
  __shared__ uint8_t table[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) table[i] = lut[i];
  __syncthreads();
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    gray[i] = table[gray[i]];
  }
}

// ---------------------------------------------------------------------------
// Stage 2: separable Gaussian blur with shared-memory tiles.
// ---------------------------------------------------------------------------

// Horizontal pass. Each block row loads blockDim.x + 2 * radius pixels.
__global__ void GaussianRowsKernel(const uint8_t* __restrict__ input,
                                   int width, int height, int radius,
                                   float* __restrict__ output) {
  extern __shared__ float tile[];
  const int tile_width = blockDim.x + 2 * radius;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int src_y = min(y, height - 1);
  const int x_origin = blockIdx.x * blockDim.x - radius;
  float* row = tile + threadIdx.y * tile_width;
  for (int j = threadIdx.x; j < tile_width; j += blockDim.x) {
    const int src_x = ClampIndex(x_origin + j, width - 1);
    row[j] = static_cast<float>(input[src_y * width + src_x]);
  }
  __syncthreads();
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= width || y >= height) return;
  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    sum += c_gaussian[k + radius] * row[threadIdx.x + radius + k];
  }
  output[y * width + x] = sum;
}

// Vertical pass. Each block loads blockDim.y + 2 * radius rows.
__global__ void GaussianColumnsKernel(const float* __restrict__ input,
                                      int width, int height, int radius,
                                      float* __restrict__ output) {
  extern __shared__ float tile[];
  const int tile_height = blockDim.y + 2 * radius;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int src_x = min(x, width - 1);
  const int y_origin = blockIdx.y * blockDim.y - radius;
  for (int i = threadIdx.y; i < tile_height; i += blockDim.y) {
    const int src_y = ClampIndex(y_origin + i, height - 1);
    tile[i * blockDim.x + threadIdx.x] = input[src_y * width + src_x];
  }
  __syncthreads();
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    sum += c_gaussian[k + radius] *
           tile[(threadIdx.y + radius + k) * blockDim.x + threadIdx.x];
  }
  output[y * width + x] = sum;
}

// ---------------------------------------------------------------------------
// Stage 3: Sobel gradient.
// ---------------------------------------------------------------------------

__global__ void SobelKernel(const float* __restrict__ blurred, int width,
                            int height, float* __restrict__ magnitude,
                            uint8_t* __restrict__ direction) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  float p[9];
  for (int dy = -1; dy <= 1; ++dy) {
    const int sy = ClampIndex(y + dy, height - 1);
    for (int dx = -1; dx <= 1; ++dx) {
      const int sx = ClampIndex(x + dx, width - 1);
      p[(dy + 1) * 3 + (dx + 1)] = __ldg(&blurred[sy * width + sx]);
    }
  }
  const int i = y * width + x;
  Sobel3x3(p, &magnitude[i], &direction[i]);
}

// ---------------------------------------------------------------------------
// Stage 4: non-maximum suppression fused with a max reduction.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float WarpMax(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__global__ void NonMaxSuppressionKernel(const float* __restrict__ magnitude,
                                        const uint8_t* __restrict__ direction,
                                        int width, int height,
                                        float* __restrict__ suppressed,
                                        float* __restrict__ global_max) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  float value = 0.0f;
  if (x < width && y < height) {
    const int i = y * width + x;
    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
      int dx = 0;
      int dy = 0;
      DirectionOffset(direction[i], &dx, &dy);
      const float center = magnitude[i];
      const float forward = magnitude[(y + dy) * width + (x + dx)];
      const float backward = magnitude[(y - dy) * width + (x - dx)];
      if (IsLocalMaximum(center, forward, backward)) value = center;
    }
    suppressed[i] = value;
  }

  // Block-wide max: warp shuffles, then one warp reduces the partials.
  __shared__ float warp_max[32];
  const int linear = threadIdx.y * blockDim.x + threadIdx.x;
  const int lane = linear & 31;
  const int warp = linear >> 5;
  const int warps = (blockDim.x * blockDim.y + 31) >> 5;
  value = WarpMax(value);
  if (lane == 0) warp_max[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float block_max = lane < warps ? warp_max[lane] : 0.0f;
    block_max = WarpMax(block_max);
    // Non-negative floats order the same as their int bit patterns.
    if (lane == 0 && block_max > 0.0f) {
      atomicMax(reinterpret_cast<int*>(global_max), __float_as_int(block_max));
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 5: double threshold (thresholds derived on the device).
// ---------------------------------------------------------------------------

__global__ void ThresholdKernel(const float* __restrict__ suppressed,
                                int64_t n, const float* __restrict__ max_value,
                                float low_ratio, float high_ratio,
                                uint8_t* __restrict__ labels) {
  const float max_v = *max_value;
  const float high = high_ratio * max_v;
  const float low = low_ratio * max_v;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    labels[i] = ClassifyMagnitude(suppressed[i], low, high);
  }
}

// ---------------------------------------------------------------------------
// Stage 6: hysteresis. Each block loads a tile plus a one-pixel halo and
// propagates "strong" through connected "weak" pixels inside shared memory
// until the tile is stable, then writes back. The host relaunches until no
// block changes anything, which is exactly the flood-fill fixpoint.
//
// Work skipping: a tile can only change if its own pixels or its halo changed
// since it was last evaluated, so a block exits immediately unless it or one
// of its 8 neighbour tiles changed in the previous pass (active_in). Later
// passes therefore only touch the "frontier" of tiles that are still growing.
// ---------------------------------------------------------------------------

template <int kHysteresisTile>
__global__ void HysteresisKernel(uint8_t* labels, int width, int height,
                                 const uint8_t* __restrict__ active_in,
                                 uint8_t* __restrict__ active_out,
                                 int* changed) {
  // Shared tile with a one-pixel halo.
  __shared__ uint8_t tile[kHysteresisTile + 2][kHysteresisTile + 2];
  __shared__ int run_block;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int run = 0;
    for (int dy = -1; dy <= 1 && !run; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const int bx = static_cast<int>(blockIdx.x) + dx;
        const int by = static_cast<int>(blockIdx.y) + dy;
        if (bx >= 0 && by >= 0 && bx < static_cast<int>(gridDim.x) &&
            by < static_cast<int>(gridDim.y) &&
            active_in[by * gridDim.x + bx] != 0) {
          run = 1;
          break;
        }
      }
    }
    run_block = run;
  }
  __syncthreads();
  if (!run_block) return;  // Uniform across the block.

  const int x_origin = blockIdx.x * kHysteresisTile - 1;
  const int y_origin = blockIdx.y * kHysteresisTile - 1;
  for (int i = threadIdx.y; i < kHysteresisTile + 2; i += blockDim.y) {
    for (int j = threadIdx.x; j < kHysteresisTile + 2; j += blockDim.x) {
      const int gx = x_origin + j;
      const int gy = y_origin + i;
      const bool inside = gx >= 0 && gy >= 0 && gx < width && gy < height;
      tile[i][j] = inside ? labels[gy * width + gx] : kNotEdge;
    }
  }
  __syncthreads();

  // Each thread owns several pixels of the tile (strided). Sweeping them in
  // order lets a promotion propagate several steps within one iteration.
  const int tile_x0 = blockIdx.x * kHysteresisTile;
  const int tile_y0 = blockIdx.y * kHysteresisTile;
  bool block_changed = false;
  while (true) {
    int promoted_any = 0;
    for (int py = threadIdx.y; py < kHysteresisTile; py += blockDim.y) {
      for (int px = threadIdx.x; px < kHysteresisTile; px += blockDim.x) {
        const int ty = py + 1;
        const int tx = px + 1;
        if (tile[ty][tx] != kWeakEdge) continue;
        if (tile_x0 + px >= width || tile_y0 + py >= height) continue;
        bool promote = false;
        for (int dy = -1; dy <= 1 && !promote; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (tile[ty + dy][tx + dx] == kStrongEdge) {
              promote = true;
              break;
            }
          }
        }
        if (promote) {
          tile[ty][tx] = kStrongEdge;
          promoted_any = 1;
        }
      }
    }
    if (!__syncthreads_or(promoted_any)) break;
    block_changed = true;
  }
  if (!block_changed) return;  // Uniform: nothing to write back.
  for (int py = threadIdx.y; py < kHysteresisTile; py += blockDim.y) {
    for (int px = threadIdx.x; px < kHysteresisTile; px += blockDim.x) {
      const int x = tile_x0 + px;
      const int y = tile_y0 + py;
      if (x < width && y < height) labels[y * width + x] = tile[py + 1][px + 1];
    }
  }
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    active_out[blockIdx.y * gridDim.x + blockIdx.x] = 1;
    *changed = 1;
  }
}

__global__ void FinalizeEdgesKernel(int64_t n, uint8_t* labels) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    labels[i] = labels[i] == kStrongEdge ? 255 : 0;
  }
}

int GridSize1d(int64_t n, int max_blocks) {
  const int64_t needed = (n + kThreads1d - 1) / kThreads1d;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(needed,
                                                                 max_blocks)));
}

}  // namespace

void UploadGaussianKernel(const std::vector<float>& weights) {
  CUDA_CHECK(cudaMemcpyToSymbol(c_gaussian, weights.data(),
                                weights.size() * sizeof(float)));
}

GpuCanny::GpuCanny(const CannyParams& params)
    : params_(params),
      radius_(static_cast<int>(MakeGaussianKernel(params.sigma).size() / 2)) {
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  for (cudaEvent_t& event : events_) CUDA_CHECK(cudaEventCreate(&event));
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count_,
                                    cudaDevAttrMultiProcessorCount, device));
}

GpuCanny::~GpuCanny() {
  ReleaseBuffers();
  for (cudaEvent_t event : events_) cudaEventDestroy(event);
  cudaStreamDestroy(stream_);
}

void GpuCanny::ReleaseBuffers() {
  cudaFreeHost(host_input_);
  cudaFreeHost(host_edges_);
  cudaFreeHost(host_changed_);
  cudaFree(d_input_);
  cudaFree(d_gray_);
  cudaFree(d_histogram_);
  cudaFree(d_lut_);
  cudaFree(d_blur_rows_);
  cudaFree(d_blurred_);
  cudaFree(d_magnitude_);
  cudaFree(d_direction_);
  cudaFree(d_suppressed_);
  cudaFree(d_max_);
  cudaFree(d_labels_);
  cudaFree(d_changed_);
  cudaFree(d_tile_active_[0]);
  cudaFree(d_tile_active_[1]);
  d_tile_active_[0] = d_tile_active_[1] = nullptr;
  capacity_tiles_ = 0;
  host_input_ = host_edges_ = nullptr;
  host_changed_ = nullptr;
  d_input_ = d_gray_ = d_lut_ = d_direction_ = d_labels_ = nullptr;
  d_histogram_ = nullptr;
  d_blur_rows_ = d_blurred_ = d_magnitude_ = d_suppressed_ = d_max_ = nullptr;
  d_changed_ = nullptr;
  capacity_pixels_ = 0;
  capacity_channels_ = 0;
}

void GpuCanny::EnsureCapacity(int64_t pixels, int channels) {
  if (pixels <= capacity_pixels_ && channels <= capacity_channels_) return;
  ReleaseBuffers();
  const size_t n = static_cast<size_t>(pixels);
  CUDA_CHECK(cudaMallocHost(&host_input_, n * channels));
  CUDA_CHECK(cudaMallocHost(&host_edges_, n));
  CUDA_CHECK(cudaMallocHost(&host_changed_, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_input_, n * channels));
  CUDA_CHECK(cudaMalloc(&d_gray_, n));
  CUDA_CHECK(cudaMalloc(&d_histogram_, 256 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_lut_, 256));
  CUDA_CHECK(cudaMalloc(&d_blur_rows_, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_blurred_, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_magnitude_, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_direction_, n));
  CUDA_CHECK(cudaMalloc(&d_suppressed_, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_max_, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_labels_, n));
  CUDA_CHECK(cudaMalloc(&d_changed_, sizeof(int)));
  capacity_pixels_ = pixels;
  capacity_channels_ = channels;
}

void GpuCanny::Run(const Image& input, bool keep_stages, CannyResult* result,
                   GpuTiming* timing) {
  const int width = input.width;
  const int height = input.height;
  const int64_t n = input.PixelCount();
  const int channels = input.channels;
  EnsureCapacity(n, channels);
  std::memcpy(host_input_, input.pixels.data(), n * channels);

  const int grid_1d = GridSize1d(n, sm_count_ * 8);
  const dim3 block_2d(kBlockX, kBlockY);
  const dim3 grid_2d((width + kBlockX - 1) / kBlockX,
                     (height + kBlockY - 1) / kBlockY);

  CUDA_CHECK(cudaEventRecord(events_[kMarkStart], stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_input_, host_input_, n * channels,
                             cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaEventRecord(events_[kMarkUploaded], stream_));

  GrayscaleKernel<<<grid_1d, kThreads1d, 0, stream_>>>(d_input_, channels, n,
                                                       d_gray_);
  CUDA_CHECK_LAUNCH();

  if (params_.equalize) {
    CUDA_CHECK(cudaMemsetAsync(d_histogram_, 0, 256 * sizeof(uint32_t),
                               stream_));
    HistogramKernel<<<grid_1d, kThreads1d, 0, stream_>>>(d_gray_, n,
                                                         d_histogram_);
    CUDA_CHECK_LAUNCH();
    EqualizeLutKernel<<<1, 256, 0, stream_>>>(
        d_histogram_, static_cast<uint32_t>(n), d_lut_);
    CUDA_CHECK_LAUNCH();
    ApplyLutKernel<<<grid_1d, kThreads1d, 0, stream_>>>(d_lut_, n, d_gray_);
    CUDA_CHECK_LAUNCH();
  }
  CUDA_CHECK(cudaEventRecord(events_[kMarkGray], stream_));

  const size_t row_smem = kBlockY * (kBlockX + 2 * radius_) * sizeof(float);
  const size_t col_smem = kBlockX * (kBlockY + 2 * radius_) * sizeof(float);
  GaussianRowsKernel<<<grid_2d, block_2d, row_smem, stream_>>>(
      d_gray_, width, height, radius_, d_blur_rows_);
  CUDA_CHECK_LAUNCH();
  GaussianColumnsKernel<<<grid_2d, block_2d, col_smem, stream_>>>(
      d_blur_rows_, width, height, radius_, d_blurred_);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaEventRecord(events_[kMarkBlurred], stream_));

  SobelKernel<<<grid_2d, block_2d, 0, stream_>>>(d_blurred_, width, height,
                                                 d_magnitude_, d_direction_);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaEventRecord(events_[kMarkGradient], stream_));

  CUDA_CHECK(cudaMemsetAsync(d_max_, 0, sizeof(float), stream_));
  NonMaxSuppressionKernel<<<grid_2d, block_2d, 0, stream_>>>(
      d_magnitude_, d_direction_, width, height, d_suppressed_, d_max_);
  CUDA_CHECK_LAUNCH();

  ThresholdKernel<<<grid_1d, kThreads1d, 0, stream_>>>(
      d_suppressed_, n, d_max_, params_.low_ratio, params_.high_ratio,
      d_labels_);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaEventRecord(events_[kMarkThresholded], stream_));

  const bool large_tiles = n >= kLargeTileMinPixels;
  const int tile_side = large_tiles ? kHysteresisLargeTile
                                    : kHysteresisSmallTile;
  const dim3 hyst_block(kHysteresisThreads, kHysteresisThreads);
  const dim3 hyst_grid((width + tile_side - 1) / tile_side,
                       (height + tile_side - 1) / tile_side);
  const int64_t tiles = static_cast<int64_t>(hyst_grid.x) * hyst_grid.y;
  if (tiles > capacity_tiles_) {
    cudaFree(d_tile_active_[0]);
    cudaFree(d_tile_active_[1]);
    CUDA_CHECK(cudaMalloc(&d_tile_active_[0], tiles));
    CUDA_CHECK(cudaMalloc(&d_tile_active_[1], tiles));
    capacity_tiles_ = tiles;
  }
  // Every tile is active for the first pass.
  CUDA_CHECK(cudaMemsetAsync(d_tile_active_[0], 1, tiles, stream_));
  int current = 0;
  int passes = 0;
  do {
    CUDA_CHECK(cudaMemsetAsync(d_changed_, 0, sizeof(int), stream_));
    for (int p = 0; p < kHysteresisPassesPerCheck; ++p) {
      CUDA_CHECK(
          cudaMemsetAsync(d_tile_active_[1 - current], 0, tiles, stream_));
      if (large_tiles) {
        HysteresisKernel<kHysteresisLargeTile>
            <<<hyst_grid, hyst_block, 0, stream_>>>(
                d_labels_, width, height, d_tile_active_[current],
                d_tile_active_[1 - current], d_changed_);
      } else {
        HysteresisKernel<kHysteresisSmallTile>
            <<<hyst_grid, hyst_block, 0, stream_>>>(
                d_labels_, width, height, d_tile_active_[current],
                d_tile_active_[1 - current], d_changed_);
      }
      CUDA_CHECK_LAUNCH();
      current = 1 - current;
    }
    passes += kHysteresisPassesPerCheck;
    CUDA_CHECK(cudaMemcpyAsync(host_changed_, d_changed_, sizeof(int),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  } while (*host_changed_ != 0 && passes < kMaxHysteresisPasses);

  FinalizeEdgesKernel<<<grid_1d, kThreads1d, 0, stream_>>>(n, d_labels_);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaEventRecord(events_[kMarkComputed], stream_));

  CUDA_CHECK(cudaMemcpyAsync(host_edges_, d_labels_, n,
                             cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaEventRecord(events_[kMarkDone], stream_));
  CUDA_CHECK(cudaEventSynchronize(events_[kMarkDone]));

  auto elapsed = [this](int from, int to) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, events_[from], events_[to]));
    return ms;
  };
  timing->upload_ms = elapsed(kMarkStart, kMarkUploaded);
  timing->gray_ms = elapsed(kMarkUploaded, kMarkGray);
  timing->blur_ms = elapsed(kMarkGray, kMarkBlurred);
  timing->gradient_ms = elapsed(kMarkBlurred, kMarkGradient);
  timing->suppress_ms = elapsed(kMarkGradient, kMarkThresholded);
  timing->hysteresis_ms = elapsed(kMarkThresholded, kMarkComputed);
  timing->compute_ms = elapsed(kMarkUploaded, kMarkComputed);
  timing->download_ms = elapsed(kMarkComputed, kMarkDone);
  timing->total_ms = elapsed(kMarkStart, kMarkDone);

  result->width = width;
  result->height = height;
  result->hysteresis_passes = passes;
  result->edges.assign(host_edges_, host_edges_ + n);
  result->edge_pixels = std::count(result->edges.begin(),
                                   result->edges.end(), uint8_t{255});

  if (keep_stages) {
    // Debug/visualization copies; deliberately outside the timed region.
    result->gray.resize(n);
    std::vector<float> blurred(n);
    std::vector<float> magnitude(n);
    CUDA_CHECK(cudaMemcpyAsync(result->gray.data(), d_gray_, n,
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(blurred.data(), d_blurred_, n * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(magnitude.data(), d_magnitude_,
                               n * sizeof(float), cudaMemcpyDeviceToHost,
                               stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    result->blurred = ScaleToBytes(blurred, 1.0f);
    const float max_mag = *std::max_element(magnitude.begin(), magnitude.end());
    result->gradient =
        ScaleToBytes(magnitude, max_mag > 0.0f ? 255.0f / max_mag : 0.0f);
  }
}

}  // namespace canny
