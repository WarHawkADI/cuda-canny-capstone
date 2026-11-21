// Copyright 2026 CUDA Canny Capstone Authors.
//
// Image file loading and saving (PNG, JPEG, BMP, TGA, PGM/PPM) built on the
// single-header stb libraries.

#ifndef SRC_IMAGE_IO_H_
#define SRC_IMAGE_IO_H_

#include <cstdint>
#include <string>
#include <vector>

#include "src/canny_types.h"

namespace canny {

// Loads an image from disk. Returns false and fills *error on failure.
bool LoadImage(const std::string& path, Image* image, std::string* error);

// Writes a single-channel 8-bit image as PNG. Returns false on failure.
bool SaveGrayPng(const std::string& path, int width, int height,
                 const std::vector<uint8_t>& pixels);

// Returns true if the path has an extension that LoadImage understands.
bool HasSupportedImageExtension(const std::string& path);

}  // namespace canny

#endif  // SRC_IMAGE_IO_H_
