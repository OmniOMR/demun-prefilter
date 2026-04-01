"""
demun_local.py — run demun-prefilter inference locally, no HTTP overhead.

Sources:
    folder      Read JPEGs from a local directory
    iipsrv      Fetch from an IIPImage server at a given size

Usage:
    # local folder
    python demun_local.py --source folder --input /path/to/images
                          --output results.csv

    # IIPImage server
    python demun_local.py --source iipsrv
                          --base-url http://iipsrv/fcgi-bin/iipsrv.fcgi
                          --image-list images.txt
                          --output results.csv

Options:
    --variant       b4 | b0           (default: from config.json)
    --checkpoint    path/to/model.pth (default: from config.json)
    --batch-size    GPU batch size    (default: 32)
    --threshold     float [0,1]       (default: 0.5)
    --workers       decode threads    (default: 8)
"""
import argparse
import csv
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B4_Weights
from torchvision.transforms import InterpolationMode

try:
    from turbojpeg import TurboJPEG, TJPF_RGB
    _turbo = TurboJPEG()
except Exception:
    _turbo = None

# ── device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
USE_CUDA = DEVICE.type == "cuda"
if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--source",      default="folder", choices=["folder", "iipsrv"])
parser.add_argument("--input",       default=None,  help="Folder of images (source=folder)")
parser.add_argument("--base-url",    default=None,  help="IIPImage base URL (source=iipsrv)")
parser.add_argument("--image-list",  default=None,  help="Text file with image paths, one per line")
parser.add_argument("--output",      default="results.csv")
parser.add_argument("--variant",     default=None,  choices=["b4", "b0"])
parser.add_argument("--checkpoint",  default=None)
parser.add_argument("--batch-size",  type=int, default=32)
parser.add_argument("--threshold",   type=float, default=None)
parser.add_argument("--workers",     type=int, default=min(8, os.cpu_count() or 4))
parser.add_argument("--config",      default="config.json")
args = parser.parse_args()

# ── load config ───────────────────────────────────────────────────────────────
cfg = {}
if Path(args.config).exists():
    with open(args.config) as f:
        cfg = json.load(f)

VARIANT     = args.variant    or cfg.get("model_variant", "b4")
THRESHOLD   = args.threshold  if args.threshold is not None else cfg.get("threshold", 0.5)
BATCH_SIZE  = args.batch_size
USE_FP16    = cfg.get("use_fp16", True) and USE_CUDA
USE_COMPILE = cfg.get("use_compile", True) and USE_CUDA

MODEL_VARIANTS = {
    "b0": (EfficientNet_B0_Weights.IMAGENET1K_V1, models.efficientnet_b0),
    "b4": (EfficientNet_B4_Weights.IMAGENET1K_V1, models.efficientnet_b4),
}
weights_obj, builder = MODEL_VARIANTS[VARIANT]
t = weights_obj.transforms()
CROP_SIZE = t.crop_size[0]
MEAN, STD = tuple(t.mean), tuple(t.std)

CHECKPOINT = args.checkpoint or cfg.get("checkpoint_path") or f"model_{VARIANT}.pth"
if not Path(CHECKPOINT).exists():
    print(f"ERROR: checkpoint not found at {CHECKPOINT}")
    sys.exit(1)

# ── model loading ─────────────────────────────────────────────────────────────
def load_model() -> Tuple[nn.Module, bool]:
    print(f"Loading {VARIANT} from {CHECKPOINT}...")
    m = builder(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    m.load_state_dict(state, strict=False)
    m.to(DEVICE).eval()
    needs_autocast = False

    if USE_CUDA and USE_COMPILE:
        try:
            print("Compiling with torch.compile...")
            m = torch.compile(m, mode="default")
            dummy = torch.randn(BATCH_SIZE, 3, CROP_SIZE, CROP_SIZE, device=DEVICE)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16):
                    for _ in range(5):
                        m(dummy)
            torch.cuda.synchronize()
            needs_autocast = USE_FP16
            print("Compile + warmup done.")
        except Exception as e:
            print(f"torch.compile failed ({e}), using eager mode.")
    elif USE_CUDA:
        dummy = torch.randn(BATCH_SIZE, 3, CROP_SIZE, CROP_SIZE, device=DEVICE)
        with torch.no_grad():
            for _ in range(5):
                m(dummy)
        torch.cuda.synchronize()

    return m, needs_autocast

model, needs_autocast = load_model()
dtype  = torch.float16 if USE_FP16 else torch.float32
mean_t = torch.tensor(MEAN, device=DEVICE, dtype=dtype).view(1, 3, 1, 1)
std_t  = torch.tensor(STD,  device=DEVICE, dtype=dtype).view(1, 3, 1, 1)

# Pinned memory for fast H2D transfer; separate streams to overlap transfer with compute
if USE_CUDA:
    _transfer_stream = torch.cuda.Stream()
    _compute_stream  = torch.cuda.Stream()
    _pinned = np.zeros((BATCH_SIZE, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    # Register as pinned memory so H2D uses DMA without a staging copy
    _pinned_tensor = torch.from_numpy(_pinned)
    torch.cuda.cudart().cudaHostRegister(
        _pinned_tensor.data_ptr(), _pinned_tensor.nbytes, 0)
else:
    _transfer_stream = _compute_stream = None
    _pinned = None

np_buffer = np.empty((BATCH_SIZE, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)

# ── image source ──────────────────────────────────────────────────────────────
def collect_items() -> List[str]:
    """Return list of identifiers (file paths or image names for iipsrv)."""
    if args.source == "folder":
        if not args.input:
            print("ERROR: --input required for source=folder")
            sys.exit(1)
        d = Path(args.input)
        return [str(p) for p in sorted(d.iterdir())
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
    else:
        if not args.image_list:
            print("ERROR: --image-list required for source=iipsrv")
            sys.exit(1)
        with open(args.image_list) as f:
            return [line.strip() for line in f if line.strip()]


# ── iipsrv fetch stats (thread-safe) ──────────────────────────────────────────
_fetch_lock  = threading.Lock()
_fetch_count = 0      # images successfully fetched from iipsrv
_fetch_bytes = 0      # total bytes received


def _record_fetch(nbytes: int) -> None:
    global _fetch_count, _fetch_bytes
    with _fetch_lock:
        _fetch_count += 1
        _fetch_bytes += nbytes


def _jpeg_to_array(data: bytes) -> np.ndarray:
    """Decode JPEG bytes to a (H, W, 3) uint8 numpy array."""
    if _turbo is not None:
        arr = _turbo.decode(data, pixel_format=TJPF_RGB)
        if arr.shape[0] != CROP_SIZE or arr.shape[1] != CROP_SIZE:
            arr = np.array(Image.fromarray(arr).resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC))
        return arr
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img.resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC), dtype=np.uint8)


def decode_item(item: str) -> Tuple[str, Optional[np.ndarray]]:
    """Decode one item to a (name, numpy_array_or_None) tuple."""
    try:
        if args.source == "folder":
            data = Path(item).read_bytes()
            arr = _jpeg_to_array(data)
        else:
            import urllib.request
            url = (f"{args.base_url}?FIF={item}"
                   f"&WID={CROP_SIZE}&HEI={CROP_SIZE}&CVT=JPEG")
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            _record_fetch(len(data))
            arr = _jpeg_to_array(data)
        return item, arr
    except Exception:
        return item, None


# ── inference ─────────────────────────────────────────────────────────────────
def run_batch(arrays: List[np.ndarray]) -> List[float]:
    n = len(arrays)
    if USE_CUDA:
        # Fill pinned buffer (avoids internal staging copy during H2D)
        pinned = _pinned[:n]
        for i, arr in enumerate(arrays):
            pinned[i] = arr  # numpy → pinned numpy (fast memcpy)
        with torch.no_grad():
            with torch.cuda.stream(_transfer_stream):
                batch = torch.from_numpy(pinned).permute(0, 3, 1, 2).to(DEVICE, non_blocking=True)
            _compute_stream.wait_stream(_transfer_stream)
            with torch.cuda.stream(_compute_stream):
                batch = batch.to(dtype) / 255.0
                batch = (batch - mean_t) / std_t
                if needs_autocast:
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = model(batch)
                else:
                    logits = model(batch)
                probs = torch.sigmoid(logits.float().squeeze(-1))
        _compute_stream.synchronize()
    else:
        buf = np_buffer[:n]
        for i, arr in enumerate(arrays):
            buf[i] = arr
        with torch.no_grad():
            batch = torch.from_numpy(buf).permute(0, 3, 1, 2).to(DEVICE, non_blocking=True)
            batch = batch.to(dtype) / 255.0
            batch = (batch - mean_t) / std_t
            if needs_autocast:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(batch)
            else:
                logits = model(batch)
            probs = torch.sigmoid(logits.float().squeeze(-1))
    result = probs.tolist()
    return result if isinstance(result, list) else [result]


# ── main loop ─────────────────────────────────────────────────────────────────
items = collect_items()
if not items:
    print("No images found.")
    sys.exit(1)

print(f"\nSource    : {args.source}  ({len(items)} images)")
print(f"Model     : {VARIANT}  crop={CROP_SIZE}  fp16={USE_FP16}")
print(f"Batch     : {BATCH_SIZE}  workers={args.workers}")
print(f"Threshold : {THRESHOLD}")
print(f"Output    : {args.output}\n")

out_path = Path(args.output)
csv_file = open(out_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["filename", "prediction", "probability"])

pool = ThreadPoolExecutor(max_workers=args.workers)
t0 = time.perf_counter()
total_done = 0
errors = 0

# Split into chunks of BATCH_SIZE
chunks = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]


def decode_chunk(chunk: List[str]):
    """Decode a full batch in parallel, return (names, arrays)."""
    futures = {pool.submit(decode_item, item): item for item in chunk}
    names, arrays = [], []
    for fut in as_completed(futures):
        name, arr = fut.result()
        names.append(name)
        arrays.append(arr)
    return names, arrays


from concurrent.futures import Future

def submit_decode(chunk):
    return pool.submit(decode_chunk, chunk)


# Two-deep pipeline: while GPU runs chunk N, decode chunk N+1
pending: Optional[Future] = None

chunk_iter = iter(chunks)
try:
    pending = submit_decode(next(chunk_iter))
except StopIteration:
    pending = None

for chunk in chunk_iter:
    next_pending = submit_decode(chunk)

    names, arrays = pending.result()
    pending = next_pending

    valid_names  = [n for n, a in zip(names, arrays) if a is not None]
    valid_arrays = [a for a in arrays if a is not None]
    errors += len(arrays) - len(valid_arrays)

    if valid_arrays:
        probs = run_batch(valid_arrays)
        for name, prob in zip(valid_names, probs):
            label = "YES" if prob >= THRESHOLD else "NO"
            writer.writerow([Path(name).name, label, f"{prob:.6f}"])

    for name, arr in zip(names, arrays):
        if arr is None:
            writer.writerow([Path(name).name, "ERROR", ""])

    total_done += len(names)
    elapsed = time.perf_counter() - t0
    fetch_suffix = ""
    if args.source == "iipsrv":
        with _fetch_lock:
            fc, fb = _fetch_count, _fetch_bytes
        fetch_suffix = (f"  [fetch: {fc/elapsed:.0f} img/s"
                        f"  {fb/elapsed/1e6:.1f} MB/s]")
    print(f"\r  {total_done}/{len(items)}  ({total_done/len(items)*100:.0f}%)  "
          f"{total_done/elapsed:.0f} img/s{fetch_suffix}", end="", flush=True)

# Flush last pending chunk
if pending is not None:
    names, arrays = pending.result()
    valid_names  = [n for n, a in zip(names, arrays) if a is not None]
    valid_arrays = [a for a in arrays if a is not None]
    errors += len(arrays) - len(valid_arrays)
    if valid_arrays:
        probs = run_batch(valid_arrays)
        for name, prob in zip(valid_names, probs):
            label = "YES" if prob >= THRESHOLD else "NO"
            writer.writerow([Path(name).name, label, f"{prob:.6f}"])
    for name, arr in zip(names, arrays):
        if arr is None:
            writer.writerow([Path(name).name, "ERROR", ""])
    total_done += len(names)
    elapsed = time.perf_counter() - t0
    fetch_suffix = ""
    if args.source == "iipsrv":
        with _fetch_lock:
            fc, fb = _fetch_count, _fetch_bytes
        fetch_suffix = (f"  [fetch: {fc/elapsed:.0f} img/s"
                        f"  {fb/elapsed/1e6:.1f} MB/s]")
    print(f"\r  {total_done}/{len(items)}  ({total_done/len(items)*100:.0f}%)  "
          f"{total_done/elapsed:.0f} img/s{fetch_suffix}", end="", flush=True)

elapsed = time.perf_counter() - t0
csv_file.close()
pool.shutdown(wait=False)

print(f"\n\nProcessed : {total_done} images in {elapsed:.1f}s")
print(f"Throughput: {total_done / elapsed:.0f} img/s")
if args.source == "iipsrv":
    with _fetch_lock:
        fc, fb = _fetch_count, _fetch_bytes
    print(f"Fetch rate: {fc/elapsed:.0f} img/s  ({fb/elapsed/1e6:.1f} MB/s  "
          f"{fb/fc/1e3:.1f} KB/img avg)" if fc else "Fetch rate: 0 img/s")
yes = sum(1 for row in csv.reader(open(out_path)) if len(row) > 1 and row[1] == "YES")
print(f"Results   : {yes} YES  /  {total_done - yes - errors} NO  /  {errors} errors")
print(f"Saved     : {out_path}")
