#!/usr/bin/env python3
"""Prepares input data for cuda_canny.

* Converts the SIPI database TIFF files to PNG (stb_image cannot read TIFF).
* Optionally generates large synthetic test images to exercise the GPU on
  big inputs.

Usage:
  python3 scripts/convert_images.py --src <dir with .tiff> --dst <png dir>
  python3 scripts/convert_images.py --synthetic 8192 --dst data/input/synthetic
"""

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image


def convert_tiffs(src: pathlib.Path, dst: pathlib.Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(src.rglob("*")):
        if path.suffix.lower() not in (".tif", ".tiff"):
            continue
        with Image.open(path) as img:
            mode = "RGB" if img.mode in ("RGB", "RGBA", "P", "CMYK") else "L"
            img.convert(mode).save(dst / (path.stem + ".png"), optimize=False)
        count += 1
    return count


def make_synthetic(size: int, dst: pathlib.Path, seed: int = 7) -> pathlib.Path:
    """Procedural RGB scene: gradients, circles, rectangles and noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.stack([xx * 180, yy * 180, (1 - xx) * 120], axis=-1)
    for _ in range(400):
        cx, cy = rng.random(2)
        r = rng.uniform(0.005, 0.06)
        color = rng.uniform(0, 255, 3)
        # Only touch the shape's bounding box to keep generation fast.
        x0, x1 = (max(0, int((cx - r) * size)), min(size, int((cx + r) * size) + 1))
        y0, y1 = (max(0, int((cy - r) * size)), min(size, int((cy + r) * size) + 1))
        bx, by = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
        if rng.random() < 0.5:
            mask = (bx - cx) ** 2 + (by - cy) ** 2 < r * r
        else:
            mask = (np.abs(bx - cx) < r) & (np.abs(by - cy) < r * 0.6)
        img[y0:y1, x0:x1][mask] = color
    img += rng.normal(0, 6, img.shape)
    out = dst / f"synthetic_{size}x{size}.png"
    dst.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=pathlib.Path, help="TIFF source dir")
    parser.add_argument("--dst", type=pathlib.Path, required=True)
    parser.add_argument("--synthetic", type=int, default=0,
                        help="Generate one SIZE x SIZE synthetic image")
    args = parser.parse_args()
    if args.src:
        n = convert_tiffs(args.src, args.dst)
        print(f"Converted {n} TIFF images from {args.src} -> {args.dst}")
    if args.synthetic:
        print(f"Wrote {make_synthetic(args.synthetic, args.dst)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
