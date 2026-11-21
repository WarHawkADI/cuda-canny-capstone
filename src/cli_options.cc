// Copyright 2026 CUDA Canny Capstone Authors.

#include "src/cli_options.h"

#include <cerrno>
#include <cstdlib>
#include <string>

namespace canny {
namespace {

bool ParseFloat(const std::string& text, float* value) {
  if (text.empty()) return false;
  char* end = nullptr;
  errno = 0;
  const float parsed = std::strtof(text.c_str(), &end);
  if (errno != 0 || end == nullptr || *end != '\0') return false;
  *value = parsed;
  return true;
}

bool ParseInt(const std::string& text, int* value) {
  if (text.empty()) return false;
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(text.c_str(), &end, 10);  // NOLINT
  if (errno != 0 || end == nullptr || *end != '\0') return false;
  *value = static_cast<int>(parsed);
  return true;
}

}  // namespace

bool ParseOptions(int argc, char** argv, Options* options,
                  std::string* error) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    // Flags without a value.
    if (arg == "-h" || arg == "--help") {
      options->show_help = true;
      continue;
    }
    if (arg == "--verify") {
      options->verify = true;
      continue;
    }
    if (arg == "--save-stages") {
      options->save_stages = true;
      continue;
    }
    if (arg == "--equalize") {
      options->params.equalize = true;
      continue;
    }
    if (arg == "--quiet") {
      options->quiet = true;
      continue;
    }
    // Flags that take a value.
    if (i + 1 >= argc) {
      *error = "unknown flag or missing value for '" + arg + "'";
      return false;
    }
    const std::string value = argv[++i];
    bool ok = true;
    if (arg == "-i" || arg == "--input") {
      options->input = value;
    } else if (arg == "-o" || arg == "--output") {
      options->output_dir = value;
    } else if (arg == "--csv") {
      options->csv_path = value;
    } else if (arg == "--sigma") {
      ok = ParseFloat(value, &options->params.sigma);
    } else if (arg == "--low") {
      ok = ParseFloat(value, &options->params.low_ratio);
    } else if (arg == "--high") {
      ok = ParseFloat(value, &options->params.high_ratio);
    } else if (arg == "-w" || arg == "--workers") {
      ok = ParseInt(value, &options->workers);
    } else if (arg == "--device") {
      ok = ParseInt(value, &options->device);
    } else if (arg == "-n" || arg == "--max-images") {
      ok = ParseInt(value, &options->max_images);
    } else if (arg == "-r" || arg == "--repeat") {
      ok = ParseInt(value, &options->repeat);
    } else {
      *error = "unknown flag '" + arg + "'";
      return false;
    }
    if (!ok) {
      *error = "invalid value '" + value + "' for " + arg;
      return false;
    }
  }

  if (options->show_help) return true;
  if (options->input.empty()) {
    *error = "--input is required";
    return false;
  }
  const CannyParams& p = options->params;
  if (p.sigma <= 0.0f || p.sigma > kMaxGaussianRadius / 3.0f) {
    *error = "--sigma must be in (0, 5]";
    return false;
  }
  if (p.low_ratio < 0.0f || p.high_ratio > 1.0f ||
      p.low_ratio > p.high_ratio) {
    *error = "thresholds must satisfy 0 <= low <= high <= 1";
    return false;
  }
  if (options->workers < 1 || options->workers > 64) {
    *error = "--workers must be between 1 and 64";
    return false;
  }
  if (options->repeat < 1 || options->repeat > 1000) {
    *error = "--repeat must be between 1 and 1000";
    return false;
  }
  if (options->max_images < 0) {
    *error = "--max-images must be >= 0";
    return false;
  }
  if (options->csv_path.empty()) {
    options->csv_path = options->output_dir + "/results.csv";
  }
  return true;
}

std::string UsageText(const std::string& program) {
  return "Usage: " + program +
         " --input <image|dir> [options]\n"
         "\n"
         "GPU-accelerated Canny edge detection for batches of images.\n"
         "\n"
         "Options:\n"
         "  -i, --input PATH       Image file or directory (searched "
         "recursively)\n"
         "  -o, --output DIR       Output directory (default: data/output)\n"
         "      --csv PATH         Per-image metrics CSV\n"
         "                         (default: <output>/results.csv)\n"
         "      --sigma F          Gaussian sigma in pixels, (0, 5] "
         "(default: 1.4)\n"
         "      --low F            Weak threshold, fraction of max gradient "
         "(default: 0.10)\n"
         "      --high F           Strong threshold, fraction of max "
         "gradient (default: 0.25)\n"
         "      --equalize         Histogram-equalize before edge "
         "detection\n"
         "  -w, --workers N        Concurrent host threads / CUDA streams "
         "(default: 2)\n"
         "      --device N         CUDA device ordinal (default: 0)\n"
         "  -n, --max-images N     Process at most N images (default: all)\n"
         "  -r, --repeat N         Run the GPU pipeline N times per image "
         "and report\n"
         "                         the fastest run (default: 1)\n"
         "      --verify           Run the CPU reference, compare pixel by "
         "pixel and\n"
         "                         report the speedup\n"
         "      --save-stages      Also save gray, blur and gradient "
         "images\n"
         "      --quiet            Print only the summary\n"
         "  -h, --help             Show this help\n"
         "\n"
         "Supported formats: png, jpg, bmp, tga, pgm, ppm.\n";
}

}  // namespace canny
