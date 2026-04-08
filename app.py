import asyncio
import io
import json
import os

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from pydantic import BaseModel, Field
from pydantic import field_validator
from PIL import Image

try:
    from turbojpeg import TurboJPEG as _TurboJPEG, TJPF_RGB as _TJPF_RGB
    _turbo = _TurboJPEG()
except Exception:
    _turbo = None
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B4_Weights


# ---------------------------------------------------------------------------
# CONFIG & DEVICE
# ---------------------------------------------------------------------------

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

USE_CUDA = DEVICE.type == "cuda"

if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")

MODEL_VARIANTS: Dict[str, Dict[str, object]] = {
    "b0": {
        "weights": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "builder": models.efficientnet_b0,
        "title_suffix": "EfficientNet-B0",
    },
    "b4": {
        "weights": EfficientNet_B4_Weights.IMAGENET1K_V1,
        "builder": models.efficientnet_b4,
        "title_suffix": "EfficientNet-B4",
    },
}

CONFIG_LOCK = asyncio.Lock()
_MODEL_CACHE: Dict[Tuple, Tuple[nn.Module, int, Tuple, Tuple, bool]] = {}

# Thread pool for CPU-bound image decoding
_DECODE_POOL = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4)))

# Maximum allowed upload size per image (50 MB)
MAX_IMAGE_BYTES = 50 * 1024 * 1024
# Maximum number of files in a single batch request
MAX_BATCH_FILES = 256
# Maximum total bytes across all files in a single batch request (500 MB)
MAX_BATCH_TOTAL_BYTES = 500 * 1024 * 1024


async def _read_image_bytes(file: UploadFile) -> bytes:
    """Read and validate an uploaded image file."""
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(data)} bytes). Maximum is {MAX_IMAGE_BYTES} bytes.",
        )
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    return data


# ---------------------------------------------------------------------------
# CONFIG LOADING
# ---------------------------------------------------------------------------

_CONFIG_DEFAULTS = {
    "model_variant": "b4",
    "threshold": 0.5,
    "max_batch_size": 32,
    "max_wait_ms": 5.0,
    "use_compile": True,
    "use_fp16": True,
}


def _load_config_or_defaults(path: str) -> Dict[str, object]:
    defaults = dict(_CONFIG_DEFAULTS)
    if not os.path.exists(path):
        print(f"[config] No config file at {path}. Using defaults.")
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as fp:
            cfg = json.load(fp)
            if not isinstance(cfg, dict):
                raise ValueError("Top-level JSON must be an object.")
            for k, v in defaults.items():
                cfg.setdefault(k, v)
            return cfg
    except Exception as exc:
        print(f"[config] Invalid config at {path}: {exc}. Using defaults.")
        return defaults


def normalize_variant(value: object) -> str:
    variant = str(value).lower()
    if variant not in MODEL_VARIANTS:
        supported = ", ".join(sorted(MODEL_VARIANTS))
        raise ValueError(f"Unsupported model_variant '{variant}'. Supported: {supported}")
    return variant


def resolve_checkpoint_path(config: Dict[str, object], variant: str) -> str:
    path = config.get("checkpoint_path")
    return str(path) if path is not None else f"model_{variant}.pth"


def compose_app_title(variant: str) -> str:
    return "Demun-prefilter"


# ---------------------------------------------------------------------------
# MODEL BUILDING & LOADING
# ---------------------------------------------------------------------------

def _get_variant_meta(variant: str) -> Tuple[int, Tuple, Tuple]:
    """Return (crop_size, mean, std) for a variant."""
    weights = MODEL_VARIANTS[variant]["weights"]
    t = weights.transforms()
    return t.crop_size[0], tuple(t.mean), tuple(t.std)


def _build_model(variant: str) -> nn.Module:
    builder = MODEL_VARIANTS[variant]["builder"]
    model = builder(weights=None)
    last_linear = model.classifier[-1]
    in_feats = getattr(last_linear, "in_features")
    model.classifier[-1] = nn.Linear(in_feats, 1)
    return model


def build_model_and_transforms(variant: str) -> Tuple[nn.Module, transforms.Compose]:
    """Legacy CPU transform pipeline (used as fallback on non-CUDA devices)."""
    crop_size, mean, std = _get_variant_meta(variant)
    eval_tf = transforms.Compose([
        transforms.Resize(
            (crop_size, crop_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    model = _build_model(variant)
    return model, eval_tf


def _load_state_dict(checkpoint_path: str):
    try:
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    return state


def load_model_bundle(
    variant: str,
    checkpoint_path: str,
    use_compile: bool = True,
    use_fp16: bool = True,
    max_batch_size: int = 32,
) -> Tuple[nn.Module, int, Tuple, Tuple, bool]:
    """Load model and return (model, crop_size, mean, std, needs_autocast).

    On CUDA the model is optionally compiled with torch.compile and warmed up.
    "needs_autocast" is True when torch.compile succeeded (fp32 weights, use
    autocast at inference) and False for JIT-traced models (fp16 weights baked in).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    model = _build_model(variant)
    state = _load_state_dict(checkpoint_path)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("load_state_dict mismatches:",
              f"\n  missing:   {missing}",
              f"\n  unexpected:{unexpected}")

    model.to(DEVICE).eval()
    crop_size, mean, std = _get_variant_meta(variant)
    needs_autocast = False

    if USE_CUDA and use_compile:
        try:
            print(f"[model] Compiling {variant} with torch.compile (default)...")
            model = torch.compile(model, mode="default")
            _warmup_model(model, crop_size, max_batch_size, use_fp16)
            needs_autocast = use_fp16
        except Exception as exc:
            print(f"[model] torch.compile failed ({exc}), falling back to jit.trace...")
            model = _build_model(variant)
            model.load_state_dict(_load_state_dict(checkpoint_path), strict=False)
            model.to(DEVICE).eval()
            model = _jit_trace_model(model, crop_size, max_batch_size, use_fp16)
    elif USE_CUDA:
        _warmup_model(model, crop_size, max_batch_size, use_fp16)

    return model, crop_size, mean, std, needs_autocast


def _jit_trace_model(model: nn.Module, crop_size: int, batch_size: int, use_fp16: bool) -> nn.Module:
    """Trace the model with torch.jit.trace for optimized inference."""
    print(f"[model] JIT tracing with batch_size={batch_size}, crop={crop_size}, fp16={use_fp16}...")
    if use_fp16:
        model = model.half()
    dtype = torch.float16 if use_fp16 else torch.float32
    dummy = torch.randn(batch_size, 3, crop_size, crop_size, device=DEVICE, dtype=dtype)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, check_trace=False)
        traced = torch.jit.freeze(traced)
        # Warmup the traced model
        for _ in range(5):
            traced(dummy)
    torch.cuda.synchronize()
    print("[model] JIT trace + freeze complete.")
    return traced


def _warmup_model(model: nn.Module, crop_size: int, batch_size: int, use_fp16: bool):
    """Run a few forward passes so torch.compile / cuDNN benchmarking finish."""
    print(f"[model] Warming up with batch_size={batch_size}, crop={crop_size}, fp16={use_fp16}...")
    dummy = torch.randn(batch_size, 3, crop_size, crop_size, device=DEVICE)
    if use_fp16:
        dummy = dummy.half()
    with torch.no_grad():
        if use_fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                for _ in range(5):
                    model(dummy)
        else:
            for _ in range(5):
                model(dummy)
    torch.cuda.synchronize()
    print("[model] Warmup complete.")


def get_or_load_bundle(
    variant: str,
    checkpoint_path: str,
    use_compile: bool = True,
    use_fp16: bool = True,
    max_batch_size: int = 32,
) -> Tuple[nn.Module, int, Tuple, Tuple, bool]:
    key = (variant, checkpoint_path, use_compile, use_fp16)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    bundle = load_model_bundle(variant, checkpoint_path, use_compile, use_fp16, max_batch_size)
    _MODEL_CACHE[key] = bundle
    return bundle


# ---------------------------------------------------------------------------
# GPU PREPROCESSING
# ---------------------------------------------------------------------------

class InvalidImageError(ValueError):
    """Raised when image bytes cannot be decoded by PIL."""


def _decode_image_to_numpy(raw: bytes) -> np.ndarray:
    """Decode image bytes to a [H, W, 3] uint8 numpy array."""
    try:
        if _turbo is not None:
            return _turbo.decode(raw, pixel_format=_TJPF_RGB)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.array(img)
    except Exception as exc:
        raise InvalidImageError(f"Cannot decode image: {exc}") from exc


def preprocess_batch_gpu(
    images_bytes: List[bytes],
    crop_size: int,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    dtype: torch.dtype = torch.float16,
    np_buffer: Optional[np.ndarray] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Decode images on CPU, then resize+normalize on GPU.

    Always applies bicubic+antialias resize to crop_size×crop_size to match
    the training preprocessing pipeline, regardless of input dimensions.
    """
    futures = [_DECODE_POOL.submit(_decode_image_to_numpy, raw) for raw in images_bytes]
    decoded = [f.result() for f in futures]

    ctx = torch.cuda.stream(stream) if stream is not None else _nullcontext()

    with ctx:
        batch_tensors = []
        for arr in decoded:
            t_cpu = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]
            t_gpu = t_cpu.to(DEVICE, non_blocking=True).unsqueeze(0).float()  # [1,3,H,W]
            t_gpu = F.interpolate(
                t_gpu, size=(crop_size, crop_size),
                mode="bicubic", align_corners=False, antialias=True,
            )
            batch_tensors.append(t_gpu)

        stacked = torch.cat(batch_tensors, dim=0)  # [B,3,H,W] on GPU
        batch = stacked.to(dtype) / 255.0
        batch = (batch - mean_t) / std_t

    return batch


class _nullcontext:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ---------------------------------------------------------------------------
# INFERENCE BATCHER (async queue → batched GPU inference)
# ---------------------------------------------------------------------------

class InferenceBatcher:
    """Collects individual prediction requests and runs them in GPU batches."""

    def __init__(
        self,
        model: nn.Module,
        crop_size: int,
        mean: Tuple,
        std: Tuple,
        max_batch_size: int = 32,
        max_wait_ms: float = 5.0,
        use_fp16: bool = True,
        needs_autocast: bool = False,
    ):
        self._model = model
        self._crop_size = crop_size
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_ms / 1000.0
        self._use_fp16 = use_fp16
        self._needs_autocast = needs_autocast

        # Pre-compute GPU-resident normalization tensors
        self._dtype = torch.float16 if use_fp16 else torch.float32
        self._mean_t = torch.tensor(mean, device=DEVICE, dtype=self._dtype).view(1, 3, 1, 1)
        self._std_t = torch.tensor(std, device=DEVICE, dtype=self._dtype).view(1, 3, 1, 1)

        # Pre-allocated numpy buffer for fast-path decoding (same-size images)
        self._np_buffer = np.empty((max_batch_size, crop_size, crop_size, 3), dtype=np.uint8)

        # CUDA streams
        self._transfer_stream = torch.cuda.Stream() if USE_CUDA else None
        self._compute_stream = torch.cuda.Stream() if USE_CUDA else None

        self._queue: asyncio.Queue = None  # set in start()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._queue = asyncio.Queue()
        self._running = True
        self._worker_task = loop.create_task(self._batch_worker())

    async def stop(self):
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def submit(self, image_bytes: bytes) -> float:
        """Submit a single image and return its probability."""
        future = self._loop.create_future()
        await self._queue.put((image_bytes, future))
        return await future

    async def submit_many(self, images_bytes: List[bytes]) -> List[float]:
        """Submit multiple images and return their probabilities."""
        futures = []
        for ib in images_bytes:
            fut = self._loop.create_future()
            await self._queue.put((ib, fut))
            futures.append(fut)
        return list(await asyncio.gather(*futures))

    async def _batch_worker(self):
        while self._running:
            batch_items = []
            try:
                # Block until at least one item arrives
                first = await self._queue.get()
                batch_items.append(first)
            except asyncio.CancelledError:
                return

            # Drain up to max_batch_size within max_wait_ms
            deadline = time.monotonic() + self._max_wait_s
            while len(batch_items) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch_items.append(item)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    # Process what we have, then exit
                    break

            images_bytes = [item[0] for item in batch_items]
            futures = [item[1] for item in batch_items]

            try:
                probabilities = await self._loop.run_in_executor(
                    None, self._run_batch, images_bytes
                )
                for fut, prob in zip(futures, probabilities):
                    if not fut.done():
                        fut.set_result(prob)
            except Exception as e:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)

    def _run_batch(self, images_bytes: List[bytes]) -> List[float]:
        """Preprocess + infer a batch. Runs in a thread."""
        # Preprocess on transfer stream
        batch = preprocess_batch_gpu(
            images_bytes, self._crop_size, self._mean_t, self._std_t,
            dtype=self._dtype,
            np_buffer=self._np_buffer,
            stream=self._transfer_stream,
        )

        # Synchronize: compute waits for transfer
        if self._compute_stream and self._transfer_stream:
            self._compute_stream.wait_stream(self._transfer_stream)

        # Inference on compute stream
        ctx = torch.cuda.stream(self._compute_stream) if self._compute_stream else _nullcontext()
        with ctx:
            with torch.no_grad():
                if self._needs_autocast:
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = self._model(batch)
                else:
                    logits = self._model(batch)
                probs = torch.sigmoid(logits.float().squeeze(-1))

        # Synchronize only the compute stream, not globally
        if self._compute_stream:
            self._compute_stream.synchronize()

        return probs.tolist() if probs.dim() > 0 else [probs.item()]

# ---------------------------------------------------------------------------
# INITIAL RUNTIME CONFIG
# ---------------------------------------------------------------------------

CONFIG = _load_config_or_defaults(CONFIG_PATH)
MODEL_VARIANT = normalize_variant(CONFIG.get("model_variant", "b4"))
THRESHOLD = float(CONFIG.get("threshold", 0.5))
CHECKPOINT_PATH = resolve_checkpoint_path(CONFIG, MODEL_VARIANT)
MAX_BATCH_SIZE = int(CONFIG.get("max_batch_size", 32))
MAX_WAIT_MS = float(CONFIG.get("max_wait_ms", 5.0))
USE_COMPILE = bool(CONFIG.get("use_compile", True))
USE_FP16 = bool(CONFIG.get("use_fp16", True))

# Global batcher instance (set during lifespan)
_batcher: Optional[InferenceBatcher] = None

# Legacy CPU transforms for fallback / non-batched path
_cpu_eval_tf: Optional[transforms.Compose] = None
_active_model: Optional[nn.Module] = None


# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------

class ConfigResponse(BaseModel):
    model_variant: str
    threshold: float
    checkpoint_path: str
    max_batch_size: int
    max_wait_ms: float
    use_compile: bool
    use_fp16: bool


class ConfigUpdateRequest(BaseModel):
    model_variant: Optional[str] = None
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    checkpoint_path: Optional[str] = None
    max_batch_size: Optional[int] = Field(None, ge=1, le=256)
    max_wait_ms: Optional[float] = Field(None, ge=0.0, le=1000.0)
    use_compile: Optional[bool] = None
    use_fp16: Optional[bool] = None

    @field_validator("model_variant")
    @classmethod
    def _validate_variant(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return normalize_variant(value)

    @field_validator("checkpoint_path")
    @classmethod
    def _validate_checkpoint(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        path = value.strip()
        if not path:
            raise ValueError("checkpoint_path cannot be empty")
        return path


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    model_variant: str
    threshold: float


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    model_variant: str
    threshold: float


# ---------------------------------------------------------------------------
# LIFESPAN (startup / shutdown)
# ---------------------------------------------------------------------------

def _build_batcher(model, crop_size, mean, std, needs_autocast=False) -> InferenceBatcher:
    return InferenceBatcher(
        model=model,
        crop_size=crop_size,
        mean=mean,
        std=std,
        max_batch_size=MAX_BATCH_SIZE,
        max_wait_ms=MAX_WAIT_MS,
        use_fp16=USE_FP16,
        needs_autocast=needs_autocast,
    )


async def _setup_batcher_and_model():
    """Load model and create batcher. Called at startup and on config change."""
    global _batcher, _active_model, _cpu_eval_tf

    try:
        model, crop_size, mean, std, needs_autocast = get_or_load_bundle(
            MODEL_VARIANT, CHECKPOINT_PATH, USE_COMPILE, USE_FP16, MAX_BATCH_SIZE,
        )
        _active_model = model

        if USE_CUDA:
            batcher = _build_batcher(model, crop_size, mean, std, needs_autocast)
            batcher.start(asyncio.get_running_loop())
            _batcher = batcher
        else:
            # Fallback: build CPU transforms
            _, _cpu_eval_tf = build_model_and_transforms(MODEL_VARIANT)
            _active_model = model

    except FileNotFoundError:
        print(f"[startup] No checkpoint at {CHECKPOINT_PATH}. "
              f"Server starts; first request with a valid checkpoint will load on demand.")
        model = _build_model(MODEL_VARIANT)
        model.to(DEVICE).eval()
        _active_model = model
        _, _cpu_eval_tf = build_model_and_transforms(MODEL_VARIANT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _setup_batcher_and_model()
    yield
    if _batcher:
        await _batcher.stop()
    _DECODE_POOL.shutdown(wait=False)


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------

app = FastAPI(title=compose_app_title(MODEL_VARIANT), version="1.0", lifespan=lifespan)


def current_config_response() -> ConfigResponse:
    return ConfigResponse(
        model_variant=MODEL_VARIANT,
        threshold=THRESHOLD,
        checkpoint_path=CHECKPOINT_PATH,
        max_batch_size=MAX_BATCH_SIZE,
        max_wait_ms=MAX_WAIT_MS,
        use_compile=USE_COMPILE,
        use_fp16=USE_FP16,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "cuda_available": USE_CUDA,
        "batching_enabled": _batcher is not None,
    }


@app.get("/config", response_model=ConfigResponse)
async def get_runtime_config() -> ConfigResponse:
    return current_config_response()


@app.post("/config", response_model=ConfigResponse)
async def update_runtime_config(update: ConfigUpdateRequest) -> ConfigResponse:
    global MODEL_VARIANT, THRESHOLD, CHECKPOINT_PATH, CONFIG
    global MAX_BATCH_SIZE, MAX_WAIT_MS, USE_COMPILE, USE_FP16
    global _batcher, _active_model, _cpu_eval_tf

    async with CONFIG_LOCK:
        new_variant = MODEL_VARIANT if update.model_variant is None else update.model_variant
        new_threshold = THRESHOLD if update.threshold is None else float(update.threshold)
        new_batch_size = MAX_BATCH_SIZE if update.max_batch_size is None else update.max_batch_size
        new_wait_ms = MAX_WAIT_MS if update.max_wait_ms is None else update.max_wait_ms
        new_compile = USE_COMPILE if update.use_compile is None else update.use_compile
        new_fp16 = USE_FP16 if update.use_fp16 is None else update.use_fp16

        config_snapshot = dict(CONFIG)
        config_snapshot["model_variant"] = new_variant
        config_snapshot["threshold"] = new_threshold
        config_snapshot["max_batch_size"] = new_batch_size
        config_snapshot["max_wait_ms"] = new_wait_ms
        config_snapshot["use_compile"] = new_compile
        config_snapshot["use_fp16"] = new_fp16

        if "checkpoint_path" in update.model_fields_set:
            if update.checkpoint_path is None:
                config_snapshot.pop("checkpoint_path", None)
            else:
                config_snapshot["checkpoint_path"] = update.checkpoint_path

        resolved_checkpoint = resolve_checkpoint_path(config_snapshot, new_variant)

        reload_needed = (
            new_variant != MODEL_VARIANT
            or resolved_checkpoint != CHECKPOINT_PATH
            or new_compile != USE_COMPILE
            or new_fp16 != USE_FP16
        )
        rebatch_needed = reload_needed or new_batch_size != MAX_BATCH_SIZE or new_wait_ms != MAX_WAIT_MS

        if reload_needed:
            try:
                # Run blocking compile/warmup in a thread so the event loop stays live
                bundle = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: get_or_load_bundle(
                        new_variant, resolved_checkpoint, new_compile, new_fp16, new_batch_size,
                    ),
                )
                model, crop_size, mean, std, needs_autocast = bundle
            except (FileNotFoundError, RuntimeError, OSError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Persist config
        CONFIG.clear()
        CONFIG.update(config_snapshot)
        THRESHOLD = new_threshold
        MODEL_VARIANT = new_variant
        CHECKPOINT_PATH = resolved_checkpoint
        MAX_BATCH_SIZE = new_batch_size
        MAX_WAIT_MS = new_wait_ms
        USE_COMPILE = new_compile
        USE_FP16 = new_fp16
        app.title = compose_app_title(MODEL_VARIANT)

        # Rebuild batcher if needed
        if rebatch_needed and USE_CUDA:
            if _batcher:
                await _batcher.stop()
            if reload_needed:
                _active_model = model
                batcher = _build_batcher(model, crop_size, mean, std, needs_autocast)
            else:
                cached = _MODEL_CACHE.get((new_variant, resolved_checkpoint, new_compile, new_fp16))
                if cached:
                    _, cs, m, s, na = cached
                    batcher = _build_batcher(_active_model, cs, m, s, na)
                else:
                    # Cache miss shouldn't happen (model was loaded at startup),
                    # but preserve needs_autocast from the old batcher if available.
                    na = _batcher._needs_autocast if _batcher else False
                    batcher = _build_batcher(
                        _active_model,
                        *_get_variant_meta(new_variant),
                        needs_autocast=na,
                    )
            batcher.start(asyncio.get_running_loop())
            _batcher = batcher
        elif reload_needed and not USE_CUDA:
            _active_model = model
            _, _cpu_eval_tf = build_model_and_transforms(new_variant)

    return current_config_response()


# ---------------------------------------------------------------------------
# SINGLE-IMAGE FALLBACK (non-CUDA or variant override)
# ---------------------------------------------------------------------------

def _predict_single_sync(
    image_bytes: bytes,
    model: nn.Module,
    eval_tf,
    threshold: float,
    needs_autocast: bool = False,
    use_fp16: bool = False,
) -> Tuple[str, float]:
    """Run single-image inference with the CPU transform pipeline.

    Handles three cases:
    - torch.compile path (needs_autocast=True): fp32 input, autocast wrapper
    - JIT-trace path (needs_autocast=False, use_fp16=True): fp16 input (weights are baked fp16)
    - CPU / no-fp16 path: plain fp32 forward pass
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise InvalidImageError(f"Cannot decode image: {exc}") from exc
    tensor = eval_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if needs_autocast and USE_CUDA:
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(tensor).squeeze()
        elif use_fp16 and not needs_autocast and USE_CUDA:
            # JIT-traced model has fp16 weights baked in — only valid on CUDA
            logit = model(tensor.half()).squeeze()
        else:
            logit = model(tensor).squeeze()
        prob = torch.sigmoid(logit.float()).item()
    label = "YES" if prob >= threshold else "NO"
    return label, prob


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
async def predict_notation(
    file: UploadFile = File(...),
    model_variant: Optional[str] = Form(None, description='Optional. "b0" or "b4".'),
    threshold: Optional[str]     = Form(None, description="Optional. Float in [0,1]."),
    checkpoint_path: Optional[str] = Form(None, description="Optional. Path to .pth."),
    persist: bool = Form(False, description="If true, make these overrides the new defaults"),
):
    global MODEL_VARIANT, THRESHOLD, CHECKPOINT_PATH, _active_model, _cpu_eval_tf, CONFIG, _batcher

    image_bytes = await _read_image_bytes(file)

    def _clean(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        s2 = s.strip()
        return None if s2 == "" else s2

    model_variant_clean = _clean(model_variant)
    threshold_str = _clean(threshold)
    checkpoint_path_clean = _clean(checkpoint_path)

    try:
        req_variant = normalize_variant(model_variant_clean) if model_variant_clean is not None else MODEL_VARIANT
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    temp_config = dict(CONFIG)
    temp_config["model_variant"] = req_variant
    if checkpoint_path_clean is not None:
        temp_config["checkpoint_path"] = checkpoint_path_clean
    resolved_ckpt = resolve_checkpoint_path(temp_config, req_variant)

    if threshold_str is None:
        req_threshold = THRESHOLD
    else:
        try:
            req_threshold = float(threshold_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="threshold must be a float (e.g., 0.5)")
        if not (0.0 <= req_threshold <= 1.0):
            raise HTTPException(status_code=400, detail="threshold must be in [0,1]")

    # Determine if we can use the batcher (same model as active)
    uses_default_model = (req_variant == MODEL_VARIANT and resolved_ckpt == CHECKPOINT_PATH)

    if uses_default_model and _batcher is not None:
        # Fast path: batched GPU inference
        try:
            probability = await _batcher.submit(image_bytes)
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
    else:
        # Fallback: load requested model, single-image inference
        try:
            bundle = get_or_load_bundle(req_variant, resolved_ckpt, USE_COMPILE, USE_FP16, MAX_BATCH_SIZE)
            model, crop_size, mean, std, needs_autocast = bundle
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Build CPU transform — BICUBIC direct square resize (empirically best on this dataset)
        eval_tf = transforms.Compose([
            transforms.Resize(
                (crop_size, crop_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        try:
            _, probability = await asyncio.get_running_loop().run_in_executor(
                None, _predict_single_sync, image_bytes, model, eval_tf,
                req_threshold, needs_autocast, USE_FP16,
            )
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    prediction_label = "YES" if probability >= req_threshold else "NO"

    if persist:
        # Trigger a config update through the normal path
        update_payload = ConfigUpdateRequest(model_variant=req_variant, threshold=req_threshold)
        if checkpoint_path_clean is not None:
            update_payload.checkpoint_path = checkpoint_path_clean
        await update_runtime_config(update_payload)

    return {
        "prediction": prediction_label,
        "probability": round(probability, 6),
        "model_variant": req_variant,
        "threshold": req_threshold,
    }


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(..., description="One or more image files"),
    threshold: Optional[str] = Form(None, description="Optional. Float in [0,1]."),
):
    """Batch prediction endpoint — send multiple images in one request."""
    if not _batcher:
        raise HTTPException(
            status_code=503,
            detail="Batching not available (CUDA required).",
        )

    if threshold is not None:
        threshold = threshold.strip()
        if threshold == "":
            threshold = None

    if threshold is None:
        req_threshold = THRESHOLD
    else:
        try:
            req_threshold = float(threshold)
        except ValueError:
            raise HTTPException(status_code=400, detail="threshold must be a float")
        if not (0.0 <= req_threshold <= 1.0):
            raise HTTPException(status_code=400, detail="threshold must be in [0,1]")

    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files ({len(files)}). Maximum is {MAX_BATCH_FILES}.",
        )

    images_bytes = []
    total_bytes = 0
    for f in files:
        data = await _read_image_bytes(f)
        total_bytes += len(data)
        if total_bytes > MAX_BATCH_TOTAL_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Batch too large. Total exceeds {MAX_BATCH_TOTAL_BYTES // (1024 * 1024)} MB.",
            )
        images_bytes.append(data)

    try:
        probabilities = await _batcher.submit_many(images_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    results = []
    for prob in probabilities:
        label = "YES" if prob >= req_threshold else "NO"
        results.append(PredictionResponse(
            prediction=label,
            probability=round(prob, 6),
            model_variant=MODEL_VARIANT,
            threshold=req_threshold,
        ))

    return BatchPredictionResponse(
        results=results,
        model_variant=MODEL_VARIANT,
        threshold=req_threshold,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
