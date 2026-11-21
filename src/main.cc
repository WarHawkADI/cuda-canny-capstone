// Copyright 2026 CUDA Canny Capstone Authors.
//
// cuda_canny: batch GPU edge detection driver.
//
// Images are distributed over N host worker threads. Each worker owns a
// GpuCanny instance (its own CUDA stream, pinned buffers and device memory),
// so image decoding, PCIe transfers and kernels of different images overlap.
// Optionally every result is checked against the CPU reference.

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstdio>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <mutex>   // NOLINT(build/c++11)
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "src/canny_cpu.h"
#include "src/canny_gpu.h"
#include "src/canny_types.h"
#include "src/cli_options.h"
#include "src/cuda_check.h"
#include "src/image_io.h"

namespace canny {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// Metrics collected for one input image.
struct ImageRecord {
  std::string name;
  bool ok = false;
  std::string error;
  int width = 0;
  int height = 0;
  int channels = 0;
  double load_ms = 0.0;
  GpuTiming gpu;
  int hysteresis_passes = 0;
  int64_t edge_pixels = 0;
  double cpu_ms = 0.0;
  int64_t mismatches = -1;  // -1 when verification was not requested.
};

double MillisecondsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

std::vector<std::string> CollectInputs(const std::string& input,
                                       int max_images) {
  std::vector<std::string> files;
  std::error_code ec;
  if (fs::is_directory(input, ec)) {
    for (const auto& entry : fs::recursive_directory_iterator(input, ec)) {
      if (entry.is_regular_file() &&
          HasSupportedImageExtension(entry.path().string())) {
        files.push_back(entry.path().string());
      }
    }
    std::sort(files.begin(), files.end());
  } else if (fs::is_regular_file(input, ec)) {
    files.push_back(input);
  }
  if (max_images > 0 && static_cast<int>(files.size()) > max_images) {
    files.resize(max_images);
  }
  return files;
}

void PrintDeviceInfo(int device) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int runtime_version = 0;
  int driver_version = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  std::printf(
      "GPU %d: %s | compute %d.%d | %d SMs | %.0f MiB | runtime %d.%d | "
      "driver %d.%d\n",
      device, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
      prop.totalGlobalMem / (1024.0 * 1024.0), runtime_version / 1000,
      (runtime_version % 1000) / 10, driver_version / 1000,
      (driver_version % 1000) / 10);
}

int64_t CountMismatches(const std::vector<uint8_t>& a,
                        const std::vector<uint8_t>& b) {
  if (a.size() != b.size()) return static_cast<int64_t>(std::max(a.size(),
                                                                 b.size()));
  int64_t count = 0;
  for (size_t i = 0; i < a.size(); ++i) count += a[i] != b[i];
  return count;
}

std::string OutputPath(const Options& options, const std::string& input,
                       const std::string& suffix) {
  return (fs::path(options.output_dir) /
          (fs::path(input).stem().string() + suffix + ".png"))
      .string();
}

void ProcessImage(const Options& options, const std::string& path,
                  GpuCanny* gpu, ImageRecord* record) {
  record->name = fs::path(path).filename().string();
  Image image;
  const Clock::time_point load_start = Clock::now();
  if (!LoadImage(path, &image, &record->error)) return;
  record->load_ms = MillisecondsSince(load_start);
  record->width = image.width;
  record->height = image.height;
  record->channels = image.channels;

  // With --repeat N the fastest of N identical runs is reported, which
  // filters out GPU clock ramp-up after idle periods (e.g. while the CPU
  // reference was running).
  CannyResult result;
  for (int rep = 0; rep < options.repeat; ++rep) {
    GpuTiming timing;
    const bool last = rep + 1 == options.repeat;
    gpu->Run(image, options.save_stages && last, &result, &timing);
    if (rep == 0 || timing.total_ms < record->gpu.total_ms) {
      record->gpu = timing;
    }
  }
  record->hysteresis_passes = result.hysteresis_passes;
  record->edge_pixels = result.edge_pixels;

  if (!SaveGrayPng(OutputPath(options, path, "_edges"), image.width,
                   image.height, result.edges)) {
    record->error = "failed to write output image";
    return;
  }
  if (options.save_stages) {
    SaveGrayPng(OutputPath(options, path, "_1_gray"), image.width,
                image.height, result.gray);
    SaveGrayPng(OutputPath(options, path, "_2_blur"), image.width,
                image.height, result.blurred);
    SaveGrayPng(OutputPath(options, path, "_3_gradient"), image.width,
                image.height, result.gradient);
  }

  if (options.verify) {
    CannyResult reference;
    const Clock::time_point cpu_start = Clock::now();
    RunCannyCpu(image, options.params, false, &reference);
    record->cpu_ms = MillisecondsSince(cpu_start);
    record->mismatches = CountMismatches(result.edges, reference.edges);
  }
  record->ok = true;
}

void PrintRecord(const ImageRecord& r, size_t index, size_t total) {
  if (!r.ok) {
    std::printf("[%4zu/%zu] %-24s FAILED: %s\n", index + 1, total,
                r.name.c_str(), r.error.c_str());
    return;
  }
  std::printf(
      "[%4zu/%zu] %-24s %5dx%-5d c%d  gpu %8.3f ms (h2d %.3f, kern %.3f, "
      "d2h %.3f) passes %3d edges %8lld",
      index + 1, total, r.name.c_str(), r.width, r.height, r.channels,
      r.gpu.total_ms, r.gpu.upload_ms, r.gpu.compute_ms, r.gpu.download_ms,
      r.hysteresis_passes, static_cast<long long>(r.edge_pixels));  // NOLINT
  if (r.mismatches >= 0) {
    std::printf("  cpu %9.2f ms  x%6.1f  mismatch %lld", r.cpu_ms,
                r.cpu_ms / std::max(1e-6, static_cast<double>(r.gpu.total_ms)),
                static_cast<long long>(r.mismatches));  // NOLINT
  }
  std::printf("\n");
}

bool WriteCsv(const std::string& path, const std::vector<ImageRecord>& rows) {
  std::ofstream csv(path);
  if (!csv) return false;
  csv << "image,status,width,height,channels,megapixels,load_ms,"
         "gpu_upload_ms,gpu_gray_ms,gpu_blur_ms,gpu_gradient_ms,"
         "gpu_suppress_ms,gpu_hysteresis_ms,gpu_compute_ms,gpu_download_ms,"
         "gpu_total_ms,hysteresis_passes,edge_pixels,cpu_ms,speedup,"
         "mismatched_pixels\n";
  char line[768];
  for (const ImageRecord& r : rows) {
    const double mp = static_cast<double>(r.width) * r.height / 1e6;
    const double speedup =
        r.mismatches >= 0 && r.gpu.total_ms > 0.0f ? r.cpu_ms / r.gpu.total_ms
                                                   : 0.0;
    std::snprintf(line, sizeof(line),
                  "%s,%s,%d,%d,%d,%.4f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                  "%.4f,%.4f,%.4f,%d,%lld,%.3f,%.2f,%lld\n",
                  r.name.c_str(), r.ok ? "ok" : "failed", r.width, r.height,
                  r.channels, mp, r.load_ms, r.gpu.upload_ms, r.gpu.gray_ms,
                  r.gpu.blur_ms, r.gpu.gradient_ms, r.gpu.suppress_ms,
                  r.gpu.hysteresis_ms, r.gpu.compute_ms, r.gpu.download_ms,
                  r.gpu.total_ms, r.hysteresis_passes,
                  static_cast<long long>(r.edge_pixels),  // NOLINT
                  r.cpu_ms, speedup,
                  static_cast<long long>(r.mismatches));  // NOLINT
    csv << line;
  }
  return static_cast<bool>(csv);
}

int Run(const Options& options) {
  const std::vector<std::string> files =
      CollectInputs(options.input, options.max_images);
  if (files.empty()) {
    std::fprintf(stderr, "No supported images found at '%s'\n",
                 options.input.c_str());
    return 1;
  }
  std::error_code ec;
  fs::create_directories(options.output_dir, ec);
  if (ec) {
    std::fprintf(stderr, "Cannot create output directory '%s': %s\n",
                 options.output_dir.c_str(), ec.message().c_str());
    return 1;
  }

  CUDA_CHECK(cudaSetDevice(options.device));
  PrintDeviceInfo(options.device);
  CUDA_CHECK(cudaFree(nullptr));  // Create the context before timing.

  const std::vector<float> weights = MakeGaussianKernel(options.params.sigma);
  UploadGaussianKernel(weights);
  std::printf(
      "Images: %zu | workers/streams: %d | sigma %.2f (kernel %zu taps) | "
      "low %.2f high %.2f | equalize %s | verify %s | repeat %d\n",
      files.size(), options.workers, options.params.sigma, weights.size(),
      options.params.low_ratio, options.params.high_ratio,
      options.params.equalize ? "on" : "off", options.verify ? "on" : "off",
      options.repeat);

  std::vector<ImageRecord> records(files.size());
  std::atomic<size_t> next_index{0};
  std::atomic<size_t> completed{0};
  std::mutex print_mutex;
  const int workers =
      std::min<int>(options.workers, static_cast<int>(files.size()));

  const Clock::time_point wall_start = Clock::now();
  std::vector<std::thread> threads;
  for (int w = 0; w < workers; ++w) {
    threads.emplace_back([&]() {
      CUDA_CHECK(cudaSetDevice(options.device));
      GpuCanny gpu(options.params);
      for (size_t i = next_index++; i < files.size(); i = next_index++) {
        ProcessImage(options, files[i], &gpu, &records[i]);
        const size_t done = completed++;
        if (!options.quiet) {
          std::lock_guard<std::mutex> lock(print_mutex);
          PrintRecord(records[i], done, files.size());
          std::fflush(stdout);
        }
      }
    });
  }
  for (std::thread& t : threads) t.join();
  const double wall_ms = MillisecondsSince(wall_start);

  // Aggregate statistics.
  int ok = 0;
  double megapixels = 0.0;
  double gpu_ms = 0.0;
  double kernel_ms = 0.0;
  double cpu_ms = 0.0;
  int64_t mismatches = 0;
  GpuTiming stages;  // Per-stage sums.
  for (const ImageRecord& r : records) {
    if (!r.ok) continue;
    ++ok;
    megapixels += static_cast<double>(r.width) * r.height / 1e6;
    gpu_ms += r.gpu.total_ms;
    kernel_ms += r.gpu.compute_ms;
    stages.upload_ms += r.gpu.upload_ms;
    stages.gray_ms += r.gpu.gray_ms;
    stages.blur_ms += r.gpu.blur_ms;
    stages.gradient_ms += r.gpu.gradient_ms;
    stages.suppress_ms += r.gpu.suppress_ms;
    stages.hysteresis_ms += r.gpu.hysteresis_ms;
    stages.download_ms += r.gpu.download_ms;
    cpu_ms += r.cpu_ms;
    if (r.mismatches > 0) mismatches += r.mismatches;
  }

  std::printf("\n===== Summary =====\n");
  std::printf("Processed       : %d / %zu images (%.2f megapixels)\n", ok,
              files.size(), megapixels);
  std::printf("Wall time       : %.2f ms (load + GPU + save, %d workers)\n",
              wall_ms, workers);
  std::printf("Throughput      : %.2f images/s, %.2f MP/s end-to-end\n",
              ok / (wall_ms / 1000.0), megapixels / (wall_ms / 1000.0));
  std::printf("GPU time (sum)  : %.2f ms total, %.2f ms kernels -> %.2f MP/s\n",
              gpu_ms, kernel_ms, megapixels / std::max(1e-9, gpu_ms / 1000.0));
  std::printf(
      "GPU stages (ms) : h2d %.2f | gray %.2f | blur %.2f | sobel %.2f | "
      "nms+thresh %.2f | hysteresis %.2f | d2h %.2f\n",
      stages.upload_ms, stages.gray_ms, stages.blur_ms, stages.gradient_ms,
      stages.suppress_ms, stages.hysteresis_ms, stages.download_ms);
  if (options.verify) {
    std::printf("CPU time (sum)  : %.2f ms -> %.2f MP/s\n", cpu_ms,
                megapixels / std::max(1e-9, cpu_ms / 1000.0));
    std::printf("Speedup         : %.1fx (CPU reference / GPU incl. "
                "transfers)\n",
                cpu_ms / std::max(1e-9, gpu_ms));
    std::printf("Verification    : %s (%lld mismatched pixels)\n",
                mismatches == 0 ? "PASSED" : "FAILED",
                static_cast<long long>(mismatches));  // NOLINT
  }
  if (WriteCsv(options.csv_path, records)) {
    std::printf("Metrics CSV     : %s\n", options.csv_path.c_str());
  } else {
    std::fprintf(stderr, "Failed to write CSV '%s'\n",
                 options.csv_path.c_str());
  }
  std::printf("Edge maps       : %s\n", options.output_dir.c_str());

  if (ok != static_cast<int>(files.size())) return 2;
  if (options.verify && mismatches != 0) return 3;
  return 0;
}

}  // namespace
}  // namespace canny

int main(int argc, char** argv) {
  canny::Options options;
  std::string error;
  if (!canny::ParseOptions(argc, argv, &options, &error)) {
    std::fprintf(stderr, "Error: %s\n\n%s", error.c_str(),
                 canny::UsageText(argv[0]).c_str());
    return 1;
  }
  if (options.show_help) {
    std::printf("%s", canny::UsageText(argv[0]).c_str());
    return 0;
  }
  return canny::Run(options);
}
