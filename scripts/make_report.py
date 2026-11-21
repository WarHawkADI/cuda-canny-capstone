#!/usr/bin/env python3
"""Builds proof-of-execution figures from the cuda_canny outputs.

Reads the CSV/logs/images written by run.sh and produces, under
results/report/:

  before_after_<name>.png   original | gray | blur | gradient | edges
  equalize_comparison.png   edges without vs. with histogram equalization
  gpu_vs_cpu_time.png       per-image GPU and CPU time vs. image size
  gpu_stage_breakdown.png   where the GPU time goes, by pipeline stage
  worker_scaling.png        end-to-end throughput vs. number of workers
  summary.md                headline numbers in a markdown table

Usage: python3 scripts/make_report.py [--results results] [--data data/input]
"""

import argparse
import csv
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

# Reference categorical palette (dataviz skill, light mode).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

SAMPLES = ["house", "4.2.03", "4.2.07", "boat.512", "5.2.10", "7.1.01"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def read_csv(path):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "ok"]


def find_original(data_dir, stem):
    for path in data_dir.rglob(stem + ".png"):
        return path
    return None


def labeled_strip(paths, labels, height=256):
    tiles = []
    for path, label in zip(paths, labels):
        img = Image.open(path).convert("RGB")
        img = img.resize((int(img.width * height / img.height), height))
        canvas = Image.new("RGB", (img.width, height + 22), SURFACE)
        canvas.paste(img, (0, 22))
        ImageDraw.Draw(canvas).text((4, 4), label, fill=INK)
        tiles.append(canvas)
    gap = 6
    strip = Image.new(
        "RGB",
        (sum(t.width for t in tiles) + gap * (len(tiles) - 1), tiles[0].height),
        SURFACE,
    )
    x = 0
    for t in tiles:
        strip.paste(t, (x, 0))
        x += t.width + gap
    return strip


def before_after(results, data_dir, out_dir):
    stage_dir = results / "misc"
    made = 0
    for stem in SAMPLES:
        original = find_original(data_dir, stem)
        stages = [stage_dir / f"{stem}_{s}.png"
                  for s in ("1_gray", "2_blur", "3_gradient", "edges")]
        if original is None or not all(p.exists() for p in stages):
            continue
        strip = labeled_strip(
            [original] + stages,
            ["BEFORE: original", "grayscale", "gaussian blur",
             "sobel gradient", "AFTER: canny edges"])
        strip.save(out_dir / f"before_after_{stem}.png")
        if made == 0:  # First sample doubles as the README hero image.
            Image.open(original).convert("RGB").save(out_dir / "readme_before.png")
            Image.open(stages[-1]).save(out_dir / "readme_after.png")
        made += 1
    return made


def equalize_comparison(results, data_dir, out_dir):
    eq_dir = results / "equalize"
    rows = []
    for stem in SAMPLES[:3]:
        original = find_original(data_dir, stem)
        plain = results / "misc" / f"{stem}_edges.png"
        eq_gray = eq_dir / f"{stem}_1_gray.png"
        eq_edges = eq_dir / f"{stem}_edges.png"
        if original and plain.exists() and eq_gray.exists() and eq_edges.exists():
            rows.append(labeled_strip(
                [original, plain, eq_gray, eq_edges],
                ["original", "edges", "equalized gray", "edges (--equalize)"]))
    if not rows:
        return False
    sheet = Image.new("RGB", (max(r.width for r in rows),
                              sum(r.height for r in rows) + 6 * len(rows)),
                      SURFACE)
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 6
    sheet.save(out_dir / "equalize_comparison.png")
    return True


def time_chart(rows, out_dir):
    mp = [float(r["megapixels"]) for r in rows]
    gpu = [float(r["gpu_total_ms"]) for r in rows]
    cpu = [float(r["cpu_ms"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    style_axes(ax)
    ax.scatter(mp, cpu, s=36, color=ORANGE, edgecolor=SURFACE, linewidth=1.5,
               label="CPU reference (1 thread)", zorder=3)
    ax.scatter(mp, gpu, s=36, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
               label="GPU (incl. PCIe transfers)", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Image size (megapixels)", color=INK_2)
    ax.set_ylabel("Time per image (ms)", color=INK_2)
    ax.set_title("Per-image pipeline time, GPU vs CPU", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "gpu_vs_cpu_time.png", dpi=150)
    plt.close(fig)


def stage_chart(rows, out_dir):
    stages = [("H2D copy", "gpu_upload_ms"), ("Grayscale", "gpu_gray_ms"),
              ("Gaussian blur", "gpu_blur_ms"), ("Sobel", "gpu_gradient_ms"),
              ("NMS + threshold", "gpu_suppress_ms"),
              ("Hysteresis", "gpu_hysteresis_ms"), ("D2H copy", "gpu_download_ms")]
    totals = [sum(float(r[key]) for r in rows) for _, key in stages]
    grand = sum(totals)
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    names = [name for name, _ in stages][::-1]
    values = totals[::-1]
    bars = ax.barh(names, values, color=BLUE, height=0.6, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + grand * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f} ms ({100 * value / grand:.0f}%)", va="center",
                color=INK_2, fontsize=9)
    ax.set_xlim(0, max(values) * 1.3)
    ax.set_xlabel("Total GPU time over all images (ms)", color=INK_2)
    ax.set_title("GPU time by pipeline stage", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "gpu_stage_breakdown.png", dpi=150)
    plt.close(fig)


def scaling_chart(log_dir, out_dir):
    points = []
    for log in sorted(log_dir.glob("03_throughput_w*.log")):
        workers = int(re.search(r"_w(\d+)", log.name).group(1))
        match = re.search(r"Throughput\s*:\s*([\d.]+) images/s, ([\d.]+) MP/s",
                          log.read_text())
        if match:
            points.append((workers, float(match.group(1)), float(match.group(2))))
    if not points:
        return []
    points.sort()
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="x", visible=False)
    labels = [str(w) for w, _, _ in points]
    bars = ax.bar(labels, [p[1] for p in points], color=BLUE, width=0.55,
                  zorder=3)
    for bar, (_, ips, _) in zip(bars, points):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{ips:.1f}", ha="center", va="bottom", color=INK_2, fontsize=9)
    ax.set_xlabel("Worker threads (= CUDA streams)", color=INK_2)
    ax.set_ylabel("Images per second (end-to-end)", color=INK_2)
    ax.set_title("Batch throughput vs concurrent streams", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "worker_scaling.png", dpi=150)
    plt.close(fig)
    return points


def write_summary(rows, points, out_dir):
    gpu = sum(float(r["gpu_total_ms"]) for r in rows)
    cpu = sum(float(r["cpu_ms"]) for r in rows)
    mp = sum(float(r["megapixels"]) for r in rows)
    mism = sum(int(r["mismatched_pixels"]) for r in rows)
    biggest = max(rows, key=lambda r: float(r["megapixels"]))
    lines = [
        "# Benchmark summary", "",
        "| Metric | Value |", "|---|---|",
        f"| Images verified | {len(rows)} |",
        f"| Total pixels | {mp:.1f} MP |",
        f"| GPU time (sum, incl. transfers) | {gpu:.1f} ms |",
        f"| CPU reference time (sum) | {cpu:.1f} ms |",
        f"| Overall speedup | {cpu / gpu:.1f}x |",
        f"| Mismatched pixels GPU vs CPU | {mism} |",
        f"| Largest image | {biggest['image']} ({biggest['width']}x"
        f"{biggest['height']}): GPU {float(biggest['gpu_total_ms']):.1f} ms, "
        f"CPU {float(biggest['cpu_ms']):.1f} ms |",
    ]
    for workers, ips, mps in points:
        lines.append(f"| Throughput, {workers} worker(s) | {ips:.1f} images/s"
                     f" ({mps:.1f} MP/s end-to-end) |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=pathlib.Path, default="results")
    parser.add_argument("--data", type=pathlib.Path, default="data/input")
    args = parser.parse_args()
    out_dir = args.results / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"before/after strips: {before_after(args.results, args.data, out_dir)}")
    print(f"equalize sheet: {equalize_comparison(args.results, args.data, out_dir)}")
    rows = read_csv(args.results / "all_verify" / "results.csv")
    time_chart(rows, out_dir)
    stage_chart(rows, out_dir)
    points = scaling_chart(args.results / "logs", out_dir)
    write_summary(rows, points, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
