// Copyright 2026 CUDA Canny Capstone Authors.

#include "src/image_io.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

namespace canny {

bool LoadImage(const std::string& path, Image* image, std::string* error) {
  int width = 0;
  int height = 0;
  int channels = 0;
  stbi_uc* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
  if (data == nullptr) {
    *error = stbi_failure_reason() != nullptr ? stbi_failure_reason()
                                              : "unknown error";
    return false;
  }
  image->width = width;
  image->height = height;
  image->channels = channels;
  const size_t bytes = static_cast<size_t>(width) * height * channels;
  image->pixels.assign(data, data + bytes);
  stbi_image_free(data);
  return true;
}

bool SaveGrayPng(const std::string& path, int width, int height,
                 const std::vector<uint8_t>& pixels) {
  if (pixels.size() != static_cast<size_t>(width) * height) return false;
  return stbi_write_png(path.c_str(), width, height, 1, pixels.data(),
                        width) != 0;
}

bool HasSupportedImageExtension(const std::string& path) {
  const size_t dot = path.find_last_of('.');
  if (dot == std::string::npos) return false;
  std::string ext = path.substr(dot + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  static const char* const kExtensions[] = {"png", "jpg", "jpeg", "bmp",
                                            "tga", "pgm", "ppm",  "pnm"};
  for (const char* candidate : kExtensions) {
    if (ext == candidate) return true;
  }
  return false;
}

}  // namespace canny
