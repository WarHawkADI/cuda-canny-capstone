// Copyright 2026 CUDA Canny Capstone Authors.

#include "src/canny_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "src/pixel_ops.h"

namespace canny {
namespace {

inline int Clamp(int value, int lo, int hi) {
  return std::min(std::max(value, lo), hi);
}

std::vector<uint8_t> ToGray(const Image& input) {
  const int64_t n = input.PixelCount();
  std::vector<uint8_t> gray(n);
  for (int64_t i = 0; i < n; ++i) {
    gray[i] = PixelToGray(&input.pixels[i * input.channels], input.channels);
  }
  return gray;
}

void EqualizeHistogram(std::vector<uint8_t>* gray) {
  uint32_t hist[256] = {0};
  for (uint8_t v : *gray) ++hist[v];
  uint32_t cdf[256];
  uint32_t running = 0;
  uint32_t cdf_min = 0;
  bool found_min = false;
  for (int i = 0; i < 256; ++i) {
    running += hist[i];
    cdf[i] = running;
    if (!found_min && hist[i] > 0) {
      cdf_min = running;
      found_min = true;
    }
  }
  uint8_t lut[256];
  for (int i = 0; i < 256; ++i) lut[i] = EqualizeBin(cdf[i], cdf_min, running);
  for (uint8_t& v : *gray) v = lut[v];
}

// Separable Gaussian blur with clamp-to-edge borders. The accumulation
// order matches the CUDA kernels exactly.
std::vector<float> GaussianBlur(const std::vector<uint8_t>& gray, int width,
                                int height, const std::vector<float>& w) {
  const int radius = static_cast<int>(w.size() / 2);
  std::vector<float> tmp(gray.size());
  std::vector<float> out(gray.size());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -radius; k <= radius; ++k) {
        const int sx = Clamp(x + k, 0, width - 1);
        sum += w[k + radius] * static_cast<float>(gray[y * width + sx]);
      }
      tmp[y * width + x] = sum;
    }
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -radius; k <= radius; ++k) {
        const int sy = Clamp(y + k, 0, height - 1);
        sum += w[k + radius] * tmp[sy * width + x];
      }
      out[y * width + x] = sum;
    }
  }
  return out;
}

}  // namespace

void RunCannyCpu(const Image& input, const CannyParams& params,
                 bool keep_stages, CannyResult* result) {
  const int width = input.width;
  const int height = input.height;
  const int64_t n = input.PixelCount();
  result->width = width;
  result->height = height;

  // 1. Grayscale and optional histogram equalization.
  std::vector<uint8_t> gray = ToGray(input);
  if (params.equalize) EqualizeHistogram(&gray);

  // 2. Gaussian smoothing.
  const std::vector<float> weights = MakeGaussianKernel(params.sigma);
  const std::vector<float> blurred = GaussianBlur(gray, width, height, weights);

  // 3. Sobel gradient magnitude and quantized direction.
  std::vector<float> magnitude(n);
  std::vector<uint8_t> direction(n);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float p[9];
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int sx = Clamp(x + dx, 0, width - 1);
          const int sy = Clamp(y + dy, 0, height - 1);
          p[(dy + 1) * 3 + (dx + 1)] = blurred[sy * width + sx];
        }
      }
      Sobel3x3(p, &magnitude[y * width + x], &direction[y * width + x]);
    }
  }

  // 4. Non-maximum suppression and global maximum.
  std::vector<float> suppressed(n, 0.0f);
  float max_value = 0.0f;
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      const int i = y * width + x;
      int dx = 0;
      int dy = 0;
      DirectionOffset(direction[i], &dx, &dy);
      const float forward = magnitude[(y + dy) * width + (x + dx)];
      const float backward = magnitude[(y - dy) * width + (x - dx)];
      if (IsLocalMaximum(magnitude[i], forward, backward)) {
        suppressed[i] = magnitude[i];
        max_value = std::max(max_value, magnitude[i]);
      }
    }
  }

  // 5. Double threshold.
  const float high = params.high_ratio * max_value;
  const float low = params.low_ratio * max_value;
  std::vector<uint8_t> labels(n);
  std::vector<int64_t> stack;
  for (int64_t i = 0; i < n; ++i) {
    labels[i] = ClassifyMagnitude(suppressed[i], low, high);
    if (labels[i] == kStrongEdge) stack.push_back(i);
  }

  // 6. Hysteresis: flood-fill from strong pixels through weak pixels.
  while (!stack.empty()) {
    const int64_t i = stack.back();
    stack.pop_back();
    const int x = static_cast<int>(i % width);
    const int y = static_cast<int>(i / width);
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = x + dx;
        const int ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const int64_t j = static_cast<int64_t>(ny) * width + nx;
        if (labels[j] == kWeakEdge) {
          labels[j] = kStrongEdge;
          stack.push_back(j);
        }
      }
    }
  }

  result->edges.assign(n, 0);
  result->edge_pixels = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (labels[i] == kStrongEdge) {
      result->edges[i] = 255;
      ++result->edge_pixels;
    }
  }
  result->hysteresis_passes = 0;

  if (keep_stages) {
    result->gray = gray;
    result->blurred = ScaleToBytes(blurred, 1.0f);
    float max_mag = 0.0f;
    for (float m : magnitude) max_mag = std::max(max_mag, m);
    result->gradient =
        ScaleToBytes(magnitude, max_mag > 0.0f ? 255.0f / max_mag : 0.0f);
  }
}

}  // namespace canny
