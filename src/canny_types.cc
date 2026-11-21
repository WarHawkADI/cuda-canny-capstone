// Copyright 2026 CUDA Canny Capstone Authors.

#include "src/canny_types.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace canny {

std::vector<float> MakeGaussianKernel(float sigma) {
  const float safe_sigma = std::max(sigma, 0.1f);
  const int radius = std::min(
      kMaxGaussianRadius, static_cast<int>(std::ceil(3.0f * safe_sigma)));
  std::vector<float> weights(2 * radius + 1);
  double sum = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    const double w =
        std::exp(-(static_cast<double>(i) * i) /
                 (2.0 * static_cast<double>(safe_sigma) * safe_sigma));
    weights[i + radius] = static_cast<float>(w);
    sum += w;
  }
  for (float& w : weights) w = static_cast<float>(w / sum);
  return weights;
}

std::vector<uint8_t> ScaleToBytes(const std::vector<float>& values,
                                  float scale) {
  std::vector<uint8_t> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    const float v = values[i] * scale + 0.5f;
    out[i] = static_cast<uint8_t>(std::min(std::max(v, 0.0f), 255.0f));
  }
  return out;
}

}  // namespace canny
