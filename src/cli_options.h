// Copyright 2026 CUDA Canny Capstone Authors.
//
// Command-line option parsing for the cuda_canny executable.

#ifndef SRC_CLI_OPTIONS_H_
#define SRC_CLI_OPTIONS_H_

#include <string>

#include "src/canny_types.h"

namespace canny {

struct Options {
  std::string input;                  // Image file or directory (required).
  std::string output_dir = "data/output";
  std::string csv_path;               // Defaults to <output_dir>/results.csv.
  CannyParams params;
  int workers = 2;                    // Host threads, one CUDA stream each.
  int device = 0;                     // CUDA device ordinal.
  int max_images = 0;                 // 0 means no limit.
  int repeat = 1;                     // GPU runs per image (best is kept).
  bool verify = false;                // Compare with the CPU reference.
  bool save_stages = false;           // Also write intermediate images.
  bool quiet = false;                 // Only print the summary.
  bool show_help = false;
};

// Parses argv into *options. Returns false and fills *error on bad input.
bool ParseOptions(int argc, char** argv, Options* options, std::string* error);

// Returns the --help text.
std::string UsageText(const std::string& program);

}  // namespace canny

#endif  // SRC_CLI_OPTIONS_H_
