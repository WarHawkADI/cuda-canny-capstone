# Coursera submission form: copy/paste text

## Project Title

CUDA Canny: Batch GPU Edge Detection with Bit-Exact CPU Verification

## Code repository URL

https://github.com/WarHawkADI/cuda-canny-capstone

## Proof of execution artifacts

Upload `proof_of_execution.zip`, created by `scripts/package_proof.sh`. It
contains:

* `logs/`: console logs of every run (nvidia-smi, verification, throughput
  scaling)
* `*/results.csv`: per-image metrics for 142 images (size, per-stage GPU
  times, CPU time, speedup, mismatched pixels)
* `misc/`: before/after and intermediate images (gray, blur, gradient,
  edges) for the 39 SIPI misc images
* `all_verify/`: edge maps for all 142 images, including the 8192×8192 image
* `report/`: before/after strips and benchmark charts

## Code Project Description

I built cuda_canny, a C++/CUDA command-line tool that runs the full Canny edge
detection pipeline on whole directories of images. The stages are grayscale
conversion, optional histogram equalization, separable Gaussian blur, Sobel
gradient, non-maximum suppression, double thresholding and hysteresis. Each
stage is a hand-written CUDA kernel, and each one uses a different technique
from the specialization:

* grid-stride loops
* a privatized shared-memory histogram with atomics
* a Hillis-Steele block scan for the CDF
* `__constant__` memory for the filter weights
* shared-memory tiles with halos for the convolution
* a warp-shuffle max reduction with atomicMax, fused into the NMS kernel
* a hysteresis kernel that propagates edges inside shared memory with
  `__syncthreads_or` and is relaunched until it converges

The batch driver runs N host threads. Each thread has its own CUDA stream,
pinned host buffers and device buffers, so decoding, PCIe transfers and
kernels for different images overlap. CUDA events at every stage boundary
give a per-stage timing breakdown. The program takes command-line arguments
for input/output paths, sigma, thresholds, equalization, worker count,
repeat count, verification and stage dumps. It writes edge maps, a
per-image CSV and a log.

To prove correctness, I wrote a CPU reference implementation. All per-pixel
math lives in one `__host__ __device__` header shared by both
implementations. Fused multiply-add is disabled (-fmad=false), and gradient
angles are binned without atan2. So the GPU result must match the CPU
**pixel for pixel**, not just approximately. Across 142 images (148
megapixels: the USC-SIPI misc, textures and aerials volumes plus an
8192×8192 synthetic image) there were 0 mismatched pixels. On an RTX 2050
laptop GPU the whole pipeline is 9.8× faster than the CPU overall. On the
67-megapixel image it is 53× faster: 64 ms vs 3.4 s, including transfers.

The most interesting issue was hysteresis. Connectivity is a global
property, so the first tiled version needed 84 kernel relaunches on the big
image and took 272 ms, about 90% of GPU time. Every other stage was
bandwidth-bound and took only a few milliseconds. I added two fixes:

1. Active-tile skipping: blocks whose neighbourhood did not change in the
   previous pass exit immediately.
2. Larger tiles where each thread sweeps several pixels.

Together these cut hysteresis to 16 ms. The large tiles were slower on
512×512 images, though, because there were too few blocks to fill the GPU.
So the tile size became a template parameter chosen per image.

Other lessons:

* Laptop GPUs downclock when idle, so benchmarks need warm-up and
  best-of-N runs (the --repeat flag).
* End-to-end throughput is limited by CPU-side PNG decoding. Overlapping
  images on 4 streams raised it from 15.5 to 29.7 images/s.
* Next steps: union-find connected components for hysteresis, and
  nvJPEG/NPP for decoding.

## Project Presentation/Demonstration URL

Record using docs/PRESENTATION.md as the script, upload the video to YouTube
(Unlisted) or Google Drive (anyone with the link), and paste the link here.
