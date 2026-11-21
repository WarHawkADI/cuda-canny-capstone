// Copyright 2026 CUDA Canny Capstone Authors.
//
// Per-pixel math shared by the CUDA kernels and the CPU reference
// implementation. Keeping a single definition guarantees that the GPU and
// CPU pipelines evaluate exactly the same formulas, which is what makes a
// bit-exact verification of the GPU output possible.

#ifndef SRC_PIXEL_OPS_H_
#define SRC_PIXEL_OPS_H_

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define CANNY_HD __host__ __device__
#else
#define CANNY_HD
#endif

namespace canny {

// Edge classification labels produced by the double-threshold stage.
constexpr uint8_t kNotEdge = 0;
constexpr uint8_t kWeakEdge = 1;
constexpr uint8_t kStrongEdge = 2;

// Quantized gradient directions (in image coordinates, y pointing down).
constexpr uint8_t kDirHorizontal = 0;  // Gradient along x.
constexpr uint8_t kDirDiagonal45 = 1;  // Gradient along (+x, +y).
constexpr uint8_t kDirVertical = 2;    // Gradient along y.
constexpr uint8_t kDirDiagonal135 = 3;  // Gradient along (+x, -y).

// tan(22.5 deg) and tan(67.5 deg), used to bin gradient angles without
// calling atan2 (whose last-ulp results differ between CPU and GPU libm).
constexpr float kTan22_5 = 0.41421356f;
constexpr float kTan67_5 = 2.41421356f;

// Integer BT.601-style luma: (77 R + 150 G + 29 B + 128) >> 8.
CANNY_HD inline uint8_t RgbToGray(uint8_t r, uint8_t g, uint8_t b) {
  return static_cast<uint8_t>((77 * r + 150 * g + 29 * b + 128) >> 8);
}

// Converts one interleaved pixel with 1..4 channels to grayscale.
CANNY_HD inline uint8_t PixelToGray(const uint8_t* px, int channels) {
  if (channels < 3) return px[0];
  return RgbToGray(px[0], px[1], px[2]);
}

// Sobel operator on a 3x3 neighbourhood given in row-major order
// (p0 p1 p2 / p3 p4 p5 / p6 p7 p8). Writes the gradient magnitude and the
// quantized gradient direction.
CANNY_HD inline void Sobel3x3(const float* p, float* magnitude,
                              uint8_t* direction) {
  const float gx = (p[2] + 2.0f * p[5] + p[8]) - (p[0] + 2.0f * p[3] + p[6]);
  const float gy = (p[6] + 2.0f * p[7] + p[8]) - (p[0] + 2.0f * p[1] + p[2]);
  *magnitude = sqrtf(gx * gx + gy * gy);

  const float ax = fabsf(gx);
  const float ay = fabsf(gy);
  if (ay <= ax * kTan22_5) {
    *direction = kDirHorizontal;
  } else if (ay >= ax * kTan67_5) {
    *direction = kDirVertical;
  } else {
    *direction = ((gx > 0.0f) == (gy > 0.0f)) ? kDirDiagonal45
                                              : kDirDiagonal135;
  }
}

// Offset (dx, dy) of the neighbour that lies along the gradient direction.
CANNY_HD inline void DirectionOffset(uint8_t direction, int* dx, int* dy) {
  switch (direction) {
    case kDirHorizontal:
      *dx = 1;
      *dy = 0;
      break;
    case kDirDiagonal45:
      *dx = 1;
      *dy = 1;
      break;
    case kDirVertical:
      *dx = 0;
      *dy = 1;
      break;
    default:
      *dx = 1;
      *dy = -1;
      break;
  }
}

// Non-maximum suppression test. The asymmetric comparison (> forward,
// >= backward) keeps exactly one pixel of a flat ridge.
CANNY_HD inline bool IsLocalMaximum(float center, float forward,
                                    float backward) {
  return center > forward && center >= backward;
}

// Double-threshold classification of a suppressed gradient magnitude.
CANNY_HD inline uint8_t ClassifyMagnitude(float value, float low,
                                          float high) {
  if (value <= 0.0f) return kNotEdge;
  if (value >= high) return kStrongEdge;
  if (value >= low) return kWeakEdge;
  return kNotEdge;
}

// Histogram-equalization lookup value for one bin, using integer math only.
CANNY_HD inline uint8_t EqualizeBin(uint32_t cdf, uint32_t cdf_min,
                                    uint32_t total) {
  if (total <= cdf_min || cdf <= cdf_min) return 0;
  const uint64_t range = total - cdf_min;
  const uint64_t value =
      (static_cast<uint64_t>(cdf - cdf_min) * 255u + range / 2u) / range;
  return static_cast<uint8_t>(value > 255u ? 255u : value);
}

}  // namespace canny

#endif  // SRC_PIXEL_OPS_H_
