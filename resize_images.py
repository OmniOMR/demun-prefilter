"""
resize_images.py — copy a folder of images pre-resized to model input size.

Usage:
    python resize_images.py --input /path/to/images --output /path/to/resized
                            [--variant b4|b0] [--quality 95] [--workers 8]
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

CROP_SIZES = {"b4": 380, "b0": 224}

parser = argparse.ArgumentParser()
parser.add_argument("--input",   required=True)
parser.add_argument("--output",  required=True)
parser.add_argument("--variant", default="b4", choices=["b4", "b0"])
parser.add_argument("--quality", type=int, default=95)
parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
args = parser.parse_args()

CROP_SIZE  = CROP_SIZES[args.variant]
INPUT_DIR  = Path(args.input)
OUTPUT_DIR = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(p for p in INPUT_DIR.iterdir()
               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})

print(f"Input    : {INPUT_DIR}  ({len(files)} images)")
print(f"Output   : {OUTPUT_DIR}")
print(f"Size     : {CROP_SIZE}×{CROP_SIZE} BICUBIC")
print(f"Workers  : {args.workers}\n")


def resize_one(args_tuple):
    src, dst, crop_size, quality = args_tuple
    try:
        img = Image.open(src).convert("RGB").resize((crop_size, crop_size), Image.BICUBIC)
        img.save(dst, format="JPEG", quality=quality, optimize=True)
        return True
    except Exception as e:
        return f"FAILED {src.name}: {e}"


tasks = [
    (p, OUTPUT_DIR / (p.stem + ".jpg"), CROP_SIZE, args.quality)
    for p in files
]

done = 0
errors = 0
import time
t0 = time.perf_counter()

with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(resize_one, t): t for t in tasks}
    for fut in as_completed(futures):
        result = fut.result()
        done += 1
        if result is not True:
            print(f"\n  {result}")
            errors += 1
        if done % 500 == 0 or done == len(tasks):
            elapsed = time.perf_counter() - t0
            print(f"\r  {done}/{len(tasks)}  ({done/len(tasks)*100:.0f}%)  "
                  f"{done/elapsed:.0f} img/s", end="", flush=True)

elapsed = time.perf_counter() - t0
print(f"\n\nDone: {done - errors}/{len(tasks)} images in {elapsed:.1f}s "
      f"({done/elapsed:.0f} img/s)  →  {OUTPUT_DIR}")
if errors:
    print(f"Errors: {errors}")
