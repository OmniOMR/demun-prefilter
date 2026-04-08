# Demun-prefilter API

This repository hosts a FastAPI application that exposes convolutional neural networks trained to prefilter document images by detecting the presence of musical notation. The service is designed as a high-recall filter that flags pages likely containing notation while maintaining a useful true-negative rate, allowing downstream pipelines to concentrate on genuinely musical content.

Model architecture, checkpoint, and decision threshold are configured via `config.json`. At runtime the service loads a binary classifier fine-tuned on proprietary datasets of scanned pages with and without notation and predicts whether musical notation is present (`YES`) or absent (`NO`) given a single RGB image.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model checkpoints (`model_b0.pth`, `model_b4.pth`) must be present in the working directory (or another location referenced in `config.json`). If a referenced checkpoint is missing at startup the server still boots, and the first request that points at a valid file will trigger the load.

For faster JPEG decoding in `demun_local.py` (requires libjpeg-turbo 2.x):

```bash
sudo apt-get install libturbojpeg0-dev
```

`demun_local.py` automatically uses TurboJPEG when available, falling back to PIL otherwise.

For full `torch.compile` support (higher throughput), install the Python development headers:

```bash
sudo apt-get install python3.12-dev   # or whichever Python version you use
```

Without them the server falls back to `torch.jit.trace`, which still provides significant optimization.

## Configuration

All runtime settings live in `config.json`:

```json
{
  "model_variant": "b4",
  "threshold": 0.5,
  "max_batch_size": 8,
  "max_wait_ms": 3.0,
  "use_compile": true,
  "use_fp16": true
}
```

- `model_variant`: `"b0"` or `"b4"` to select the EfficientNet backbone.
- `threshold`: sigmoid cutoff used to map probabilities to `YES`/`NO`.
- `checkpoint_path` (optional): filesystem path to the `.pth` checkpoint to load if you don't want the default `model_<variant>.pth`.
- `max_batch_size`: maximum number of images to batch together for GPU inference. Optimal values: 32 for B0, 8 for B4.
- `max_wait_ms`: maximum time (ms) the batcher waits to fill a batch before running inference. Lower values reduce latency; higher values improve throughput under load.
- `use_compile`: when `true`, attempts `torch.compile` (falls back to `torch.jit.trace` if Python dev headers are missing).
- `use_fp16`: when `true`, runs inference in half-precision for higher throughput on CUDA.

The file location can be overridden by setting the `CONFIG_PATH` environment variable before launching the app.

## Running the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

On startup the server automatically:
1. Detects the best available device (CUDA > MPS > CPU).
2. On CUDA: enables `cudnn.benchmark` and TF32 matmul precision for tensor core utilization.
3. Loads the configured model, applies `torch.compile` or `torch.jit.trace` + `torch.jit.freeze` with fp16 weights.
4. Runs warmup passes to stabilize kernel selection.
5. Starts the async batching engine with dedicated CUDA streams for transfer and compute.

## API Endpoints

### `GET /health`

Returns server status, device, and whether batching is active.

```json
{
  "status": "ok",
  "device": "cuda",
  "cuda_available": true,
  "batching_enabled": true
}
```

### `GET /config`

Returns the active runtime configuration including all performance fields.

### `POST /config`

Accepts any subset of config fields to update the runtime configuration. Leave a field out to keep the prior value or send `"checkpoint_path": null` to clear a previously set override. Changing the model variant, compile, or fp16 settings triggers a full model reload and batcher rebuild.

```bash
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"model_variant": "b0", "threshold": 0.4, "max_batch_size": 32}'
```

### `POST /predict`

Single-image prediction.

- **Payload:** multipart form with an `image/*` file part named `file` and optional form fields:
  - `model_variant`: overrides the configured variant (`"b0"` or `"b4"`).
  - `threshold`: overrides the sigmoid cutoff; must parse as a float in `[0, 1]`.
  - `checkpoint_path`: points to an alternate checkpoint file.
  - `persist`: set to `"true"` to make the supplied overrides the new defaults for subsequent requests and for `GET /config`.
- **Response:**

```json
{
  "prediction": "YES",
  "probability": 0.87,
  "model_variant": "b4",
  "threshold": 0.5
}
```

When the request uses the active model (no variant override), it is routed through the async batcher for optimal throughput. Requests with variant overrides fall back to a synchronous single-image path.

### `POST /predict_batch`

Batch prediction endpoint for maximum throughput. Send multiple images in a single HTTP request.

- **Payload:** multipart form with one or more `image/*` file parts named `files`, and an optional `threshold` form field.
- **Response:**

```json
{
  "results": [
    {"prediction": "YES", "probability": 0.87, "model_variant": "b4", "threshold": 0.5},
    {"prediction": "NO", "probability": 0.12, "model_variant": "b4", "threshold": 0.5}
  ],
  "model_variant": "b4",
  "threshold": 0.5
}
```

Images are always resized to the model's crop size (224×224 for B0, 380×380 for B4) using GPU bicubic interpolation with antialiasing, matching the training preprocessing pipeline. **Always send images larger than the crop size** — sending an image that is already at exactly crop size skips the downscaling step and produces pixel values that differ from the training distribution, leading to incorrect predictions. A safe minimum is ~20% above crop size (e.g. 460×460 or larger for B4).

## Performance

Measured on NVIDIA RTX 5060 Ti (16 GB VRAM), fp16 inference, `torch.compile` (default mode), EfficientNet-B4.

### Pure model throughput (no HTTP overhead)

| Model | Batch Size | Throughput |
|:------|:-----------|:-----------|
| EfficientNet-B0 | 32 | ~5,400 img/s |
| EfficientNet-B4 | 8 | ~700 img/s |

### HTTP throughput (pre-sized images)

| Model | Endpoint | Batch Size | Concurrency | Throughput |
|:------|:---------|:-----------|:------------|:-----------|
| EfficientNet-B4 | `POST /predict_batch` | 32 | 8 | ~419 img/s |

Measured on 10,620 unique pre-resized 380×380 images (dataset repeated 3×). Without `python3-dev`, the server falls back to `torch.jit.trace` which is slightly slower.

### Local mode throughput (no HTTP)

`demun_local.py` bypasses HTTP entirely:

| Model | Source | Throughput |
|:------|:-------|:-----------|
| EfficientNet-B0 | Pre-resized JPEGs from disk | ~2,285 img/s (steady state) |
| EfficientNet-B4 | Pre-resized JPEGs from disk | ~570 img/s (steady state) |

The remaining gap to the pure model ceiling (~5,400 img/s for B0, ~700 img/s for B4) is per-batch CPU/GPU pipeline overhead (numpy copies, tensor ops, sync) rather than decode. With a real C++ iipsrv the throughput approaches the same disk-mode numbers.

## Evaluation Summary

Performance metrics were computed on two complementary datasets:

1. **Balanced validation suite** with a 50/50 split between notation and non-notation pages.
2. **Edge-case stress test** of 100k pages that earlier model generations all classified as positives. Only 1.5% of this corpus contains actual notation.

### Balanced validation (50/50)

| Model | Threshold | Recall | TNR |
|:------|:----------|:-------|:----|
| EfficientNet-B0 | 0.35 | >= 0.98 | ~0.52 |
| EfficientNet-B4 | 0.60 | ~0.99 | ~0.78 |

### Edge-case corpus (1.5% positives)

| Model | Threshold | Recall | TNR |
|:------|:----------|:-------|:----|
| EfficientNet-B0 | 0.50 | ~0.97 | ~0.36 |
| EfficientNet-B4 | 0.50 | ~0.98 | ~0.79 |

The edge-case dataset is intentionally adversarial: every sample was previously flagged as positive by legacy models, making the operating point of this generation particularly salient. Both current models retain high recall while recovering substantial specificity, with the EfficientNet-B4 variant nearly recovering four-fifths of the false positives.

## Design Notes

- **Objective:** Maximize recall under tight latency constraints while improving TNR on difficult negatives to reduce downstream review costs.
- **Preprocessing:** Inputs are always resized to the model's crop size (224 for B0, 380 for B4) using GPU bicubic interpolation with antialiasing and normalized with ImageNet statistics. Images must be larger than the crop size — sending an already-cropped image bypasses the downscaling and produces pixel statistics that differ from the training distribution.
- **GPU optimization:** On CUDA, the model runs in fp16 with `cudnn.benchmark` and TF32 matmul precision enabled. The model is optimized via `torch.compile(mode="default")` (if Python dev headers are available) or `torch.jit.trace` + `torch.jit.freeze` as fallback. Separate CUDA streams overlap data transfer with compute.
- **Batching:** An async batching engine collects incoming requests into GPU batches, configured via `max_batch_size` and `max_wait_ms`. Image decoding runs in a parallel thread pool.
- **Calibration:** Thresholds are configurable per deployment through `config.json`; values above reflect empirically optimized operating points for the reported datasets.
- **Model selection:** EfficientNet-B0 provides a rapid screening layer suitable for high-throughput scraping pipelines, while EfficientNet-B4 acts as a confirmatory stage where additional latency is acceptable in exchange for sharper discrimination. Switch between them by updating `model_variant` in the configuration.

## Utility Scripts

### `demun_local.py` — local inference, no HTTP

Runs inference directly, bypassing the HTTP server entirely. Two source modes:

```bash
# Read JPEGs from a local folder
python demun_local.py --source folder --input /path/to/images --output results.csv

# Fetch from an IIPImage server at the model's crop size
python demun_local.py --source iipsrv \
    --base-url http://iipsrv/fcgi-bin/iipsrv.fcgi \
    --image-list images.txt \
    --output results.csv
```

Options: `--variant b4|b0`, `--checkpoint path/to/model.pth`, `--batch-size 32`, `--threshold 0.5`, `--workers 8`.

Uses a 2-deep decode pipeline (decode batch N+1 while GPU runs batch N) and `torch.compile` + fp16 for maximum throughput (~414 img/s with pre-resized images on the test GPU).

### `resize_images.py` — pre-resize images to model input size

Pre-resizes a folder of images to the model's crop size (380×380 for B4, 224×224 for B0) using BICUBIC interpolation:

```bash
python resize_images.py --input /path/to/originals \
    --output /path/to/resized \
    --variant b4 --workers 8
```

Achieves ~600 img/s on CPU using `ProcessPoolExecutor`.

## Limitations

- The reported accuracy metrics derive from proprietary datasets; broader generalization depends on the similarity between deployment data and the curated corpora.
- `torch.compile` requires Python development headers (`python3-dev`). Without them the server falls back to `torch.jit.trace` which is still performant but slightly slower.
- The `/predict` endpoint with per-request model variant overrides bypasses the batcher and uses a slower single-image path.
