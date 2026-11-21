# Presentation / demo script (about 7 minutes)

Record your screen (OBS, Zoom or the Xbox Game Bar with Win+Alt+R) and upload
the video to YouTube as *Unlisted* or to Google Drive with *Anyone with the
link*. Paste that URL into the "Project Presentation/Demonstration" field.

---

## 0:00 – 0:45 Intro

* "My capstone is **CUDA Canny**: a batch edge detection tool where every
  stage of the Canny pipeline is a custom CUDA kernel."
* "It processes the USC-SIPI image database plus a 67-megapixel synthetic
  image, and a CPU reference implementation checks every pixel of the GPU
  output."
* Show the README on GitHub: the before/after image at the top.

## 0:45 – 2:30 Pipeline and CUDA techniques (README table)

Walk through the stage table:

1. Grayscale: grid-stride loop.
2. Histogram equalization: privatized shared-memory histogram with atomics,
   then a Hillis-Steele scan in one block for the CDF.
3. Gaussian blur: separable, weights in constant memory, shared-memory tiles
   with halos.
4. Sobel: no atan2, so the angle binning is exact on both CPU and GPU.
5. NMS fused with a warp-shuffle max reduction and atomicMax. The thresholds
   are computed on the GPU, so there is no host round trip.
6. Hysteresis: the interesting one. It is a connected-components problem, so
   blocks propagate inside a shared-memory tile using `__syncthreads_or`, and
   the host relaunches until nothing changes.
7. Batch driver: N threads, each with its own CUDA stream and pinned memory.

Open `src/canny_gpu.cu` and show `HysteresisKernel` and `GaussianRowsKernel`.

## 2:30 – 4:30 Live demo (terminal in WSL / lab)

```bash
make
./bin/cuda_canny --help
./bin/cuda_canny --input data/input/misc --output demo --verify --save-stages --repeat 3 --workers 1
./bin/cuda_canny --input data/input --workers 4 --quiet
```

Point out on screen:
* the per-image line: GPU ms split into h2d/kernels/d2h, hysteresis passes,
  CPU ms, speedup, **mismatch 0**
* the summary: `Verification : PASSED (0 mismatched pixels)`
* open a few files in `demo/`: `_1_gray`, `_2_blur`, `_3_gradient`, `_edges`

## 4:30 – 6:00 Results (results/report/*.png)

* `gpu_vs_cpu_time.png`: the gap grows with image size.
* `gpu_stage_breakdown.png`: **hysteresis dominates**. Every other stage is
  bandwidth-bound and takes only milliseconds.
* `worker_scaling.png`: more streams help end-to-end throughput because PNG
  decode/encode on the CPU overlaps with GPU work.
* `equalize_comparison.png`: equalization brings out edges in dark or
  low-contrast images.

## 6:00 – 7:00 Lessons learned

* **Bit-exact verification needs discipline**: shared `__host__ __device__`
  math, `-fmad=false`, and no transcendental functions on the decision path.
  atan2 and FMA can give last-bit differences that flip pixels sitting on a
  threshold.
* **Profile before optimizing**: the per-stage event timing showed that
  hysteresis took 80–90% of GPU time. Active-tile skipping plus larger
  multi-pixel tiles cut it about 17× on the 8192² image (272 ms to 16 ms).
  But 64 px tiles were *slower* on 512² images (too few blocks for 16 SMs),
  hence the per-image template choice.
* **Laptop GPUs clock down when idle**: timings taken right after a long CPU
  phase are slow. `--repeat` reports the best of N runs.
* Next steps: a union-find connected-components hysteresis (one pass), and
  NVIDIA nvJPEG/NPP for decoding so the CPU stops being the end-to-end
  bottleneck.
