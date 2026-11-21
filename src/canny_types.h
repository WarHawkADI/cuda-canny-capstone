// Copyright 2026 CUDA Canny Capstone Authors.
//
// Plain data types shared by the GPU pipeline, the CPU reference and the
// command-line driver.

#ifndef SRC_CANNY_TYPES_H_
#define SRC_CANNY_TYPES_H_

#include <cstdint>
#include <string>
#include <vector>

namespace canny {

// Largest supported Gaussian radius (kernel size 2 * radius + 1).
constexpr int kMaxGaussianRadius = 15;

// Tunable parameters of the edge detection pipeline.
struct CannyParams {
  float sigma = 1.4f;        // Gaussian standard deviation in pixels.
  float low_ratio = 0.10f;   // Weak threshold as a fraction of max gradient.
  float high_ratio = 0.25f;  // Strong threshold as a fraction of max gradient.
  bool equalize = false;     // Apply histogram equalization before blurring.
};

// An 8-bit interleaved image (1 to 4 channels).
struct Image {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<uint8_t> pixels;

  int64_t PixelCount() const {
    return static_cast<int64_t>(width) * height;
  }
};

// Output of one pipeline run. Intermediate stages are only filled when
// requested because copying them back costs PCIe bandwidth.
struct CannyResult {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> edges;     // Final edge map, 0 or 255.
  std::vector<uint8_t> gray;      // Grayscale (after optional equalization).
  std::vector<uint8_t> blurred;   // Gaussian-smoothed image.
  std::vector<uint8_t> gradient;  // Gradient magnitude scaled to 0..255.
  int hysteresis_passes = 0;      // Kernel launches until convergence.
  int64_t edge_pixels = 0;
};

// Wall-clock breakdown of a GPU run, measured with CUDA events.
struct GpuTiming {
  float upload_ms = 0.0f;      // Host to device copy.
  float gray_ms = 0.0f;        // Grayscale (+ histogram equalization).
  float blur_ms = 0.0f;        // Separable Gaussian blur.
  float gradient_ms = 0.0f;    // Sobel gradient.
  float suppress_ms = 0.0f;    // Non-max suppression, max, double threshold.
  float hysteresis_ms = 0.0f;  // Iterative hysteresis + finalize.
  float compute_ms = 0.0f;     // All kernels (sum of the stages above).
  float download_ms = 0.0f;    // Device to host copy.
  float total_ms = 0.0f;
};

// Builds a normalized 1-D Gaussian kernel of size 2 * radius + 1 where
// radius = ceil(3 * sigma), clamped to kMaxGaussianRadius.
std::vector<float> MakeGaussianKernel(float sigma);

// Converts float values to bytes as round(value * scale), clamped to 0..255.
std::vector<uint8_t> ScaleToBytes(const std::vector<float>& values,
                                  float scale);

}  // namespace canny

#endif  // SRC_CANNY_TYPES_H_
