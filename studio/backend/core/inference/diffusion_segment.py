# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text-prompted masks for the Images page: SAM 3 (Segment Anything with Concepts) turns "sofa, red lamp, pillow:2"
into a white-on-black mask the inpaint and localized-edit painters load and keep editable.

Behaviour follows ComfyUI's ``SAM3_Detect`` node and its "Image Segmentation (SAM3)" blueprint
(comfy_extras/nodes_sam3.py, comfy/text_encoders/sam3_clip.py): comma-separated concepts, ``:N`` caps a concept's
detections (default 1), the image is stretched to 1008x1008 with bilinear resampling, a detection is kept when its
score > 0.5, the best ``N`` per concept are kept, and the union of their masks (mask logits upsampled bilinearly to
the source size, > 0) is the result. Two departures: the score is sigmoid(logit) * sigmoid(presence), as in Meta's
SAM 3 and transformers' post-processing (ComfyUI drops the presence term, and then a concept that is not in the
picture, "elephant" in a living room, masks the sofa at 0.72; with it every absent concept scores below 0.02 and no
present one changes), and ComfyUI's optional crop-and-refine pass through the tracker's SAM decoder is not run.

Weights are the official transformers-format ``facebook/sam3`` checkpoint (gated: the license is accepted on Hugging
Face). SAM 3.1 ships only an original-format multiplex checkpoint with no transformers integration, so it is not
loaded here; ``UNSLOTH_SAM3_PATH`` points at any local transformers-format SAM 3 directory instead.

The model is auxiliary to the image model, like the FLUX.3 prompt upsampler: loaded on the first request, kept for
a short keep-alive, unloaded right after a call when the accelerator is tight, and dropped whenever the GPU arbiter
hands the GPU to a load, evicts the image model, or training starts. It runs on the CPU while training holds the GPU
or when the accelerator has no room.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import sys
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from loggers import get_logger

logger = get_logger(__name__)

# ── weights ─────────────────────────────────────────────────────────────────────────────────────────────────────────
SEGMENTER_ID = "sam3"
SEGMENTER_REPO = "facebook/sam3"
SEGMENTER_LICENSE_URL = "https://huggingface.co/facebook/sam3"
SEGMENTER_WEIGHTS_FILE = "model.safetensors"
# Only what Sam3Model and its tokenizer read: the repo also holds sam3.pt, the same weights in the original format.
SEGMENTER_FILES = (
    "config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    SEGMENTER_WEIGHTS_FILE,
)
# fp32 safetensors as published (half of it once loaded in bf16 / fp16), plus the tokenizer and configs.
SEGMENTER_SIZE_BYTES = 3_439_938_512 + 5_060_000
ENV_PATH_OVERRIDE = "UNSLOTH_SAM3_PATH"
ENV_KEEP_ALIVE = "UNSLOTH_SAM3_KEEP_ALIVE"

# ── ComfyUI SAM3_Detect defaults ────────────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 1008
DEFAULT_THRESHOLD = 0.5
MAX_CONCEPTS = 16
# The detector proposes 200 queries per concept.
MAX_DETECTIONS = 200
TEXT_MAX_LENGTH = 32

# ── memory ──────────────────────────────────────────────────────────────────────────────────────────────────────────
KEEP_ALIVE_SECONDS = 120.0
# bf16 weights (~1.7 GB) plus the 1008x1008 ViT activations and the allocator's slack.
_GPU_NEED_MIB = 4096
# After a call, below this much free (or this fraction of the card) the model leaves at once, so the image model
# never denoises next to it. The prompt upsampler's tight rule.
_TIGHT_FREE_MIB = 6144
_TIGHT_FREE_FRACTION = 0.20


class SegmenterUnavailableError(RuntimeError):
    """This install cannot run SAM 3 (transformers without it)."""


class SegmenterNotDownloadedError(RuntimeError):
    """The weights are not in the cache; the UI offers the download."""


class SegmenterBusyError(RuntimeError):
    """The image model is rendering or loading; the GPU is not free for the segmenter."""


class SegmenterReleaseTimeoutError(RuntimeError):
    """A GPU handover waited out a call still running on the accelerator: the card is not free yet."""


# ── prompt syntax ───────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen = True)
class Concept:
    text: str
    max_detections: int


_MAX_SUFFIX = re.compile(r"^(.+?)\s*:\s*([\d.]+)\s*$")


def parse_prompt(text: Optional[str]) -> list[Concept]:
    """ComfyUI's ``_parse_prompts`` (comfy/text_encoders/sam3_clip.py): drop parentheses, split on commas, and read a
    trailing ``:N`` as that concept's maximum detections (rounded, at least 1; 1 when absent). Duplicates are kept,
    as ComfyUI keeps them. Capped at MAX_CONCEPTS concepts and MAX_DETECTIONS per concept."""
    text = (text or "").replace("(", "").replace(")", "")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    concepts: list[Concept] = []
    for part in parts:
        match = _MAX_SUFFIX.match(part)
        if match:
            label = match.group(1).strip()
            try:
                count = max(1, round(float(match.group(2))))
            except (ValueError, OverflowError):  # "1.2.3", or hundreds of digits (float -> inf)
                label, count = part, 1
        else:
            label, count = part, 1
        if label:
            concepts.append(Concept(label, min(count, MAX_DETECTIONS)))
    if len(concepts) > MAX_CONCEPTS:
        raise ValueError(f"At most {MAX_CONCEPTS} concepts per mask (got {len(concepts)}).")
    return concepts


# ── availability and files ──────────────────────────────────────────────────────────────────────────────────────────
def transformers_supports_sam3() -> bool:
    """Whether the installed transformers ships SAM 3 (5.x), read from the package layout: the status route must not
    import the model code, and an attribute probe on transformers' lazy module during another thread's import of it
    can read as missing."""
    try:
        import importlib.util

        # A --no-torch install ships transformers without torch: SAM 3 has the code there and still cannot run.
        if importlib.util.find_spec("torch") is None:
            return False
        return importlib.util.find_spec("transformers.models.sam3.modeling_sam3") is not None
    except Exception:  # noqa: BLE001
        return False


def _override_dir() -> Optional[str]:
    """The override directory when it holds what Sam3Model and its tokenizer load (a config, the weights, single or
    sharded, and a tokenizer), else None: a partial or unrelated folder must not read as downloaded."""
    value = (os.environ.get(ENV_PATH_OVERRIDE) or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    has = lambda *names: any((path / name).is_file() for name in names)  # noqa: E731
    complete = (
        has("config.json")
        and (
            has(SEGMENTER_WEIGHTS_FILE)
            or _shards_complete(path / (SEGMENTER_WEIGHTS_FILE + ".index.json"))
        )
        # AutoTokenizer (CLIPTokenizer for sam3) loads tokenizer.json, else needs vocab.json and merges.txt together.
        and (has("tokenizer.json") or (has("vocab.json") and has("merges.txt")))
    )
    return str(path) if complete else None


def _shards_complete(index: Path) -> bool:
    """A sharded checkpoint's index is present and every shard its weight_map names is beside it."""
    try:
        weight_map = json.loads(index.read_text(encoding = "utf-8")).get("weight_map")
    except Exception:  # noqa: BLE001 -- missing or unreadable: not a checkpoint
        return False
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shards = set(weight_map.values())
    return all(isinstance(name, str) and (index.parent / name).is_file() for name in shards)


def _cached_snapshot(
    model_id: str = SEGMENTER_ID,
    *,
    hub_cache: Optional[Path] = None,
    revision: Optional[str] = None,
) -> Optional[str]:
    """The snapshot directory holding every SEGMENTER_FILES at one revision (the recorded one first), else None."""
    try:
        from huggingface_hub import hf_hub_download

        from core.inference.stt_sidecar import (
            _active_hf_hub_cache,
            _fallback_revisions,
            _read_revision_record,
        )

        root = hub_cache if hub_cache is not None else _active_hf_hub_cache()
        if revision is not None:
            revisions = [revision]
        else:
            recorded = _read_revision_record(SEGMENTER_REPO)
            revisions = ([recorded] if recorded else []) + [
                rev
                for rev in _fallback_revisions(SEGMENTER_REPO, hub_cache = root)
                if rev != recorded
            ]
        for rev in revisions:
            paths = []
            for filename in SEGMENTER_FILES:
                try:
                    paths.append(
                        hf_hub_download(
                            repo_id = SEGMENTER_REPO,
                            filename = filename,
                            revision = rev,
                            local_files_only = True,
                            cache_dir = str(root),
                        )
                    )
                except Exception:  # noqa: BLE001 -- missing at this revision
                    paths = []
                    break
            if paths:
                return str(Path(paths[0]).parent)
    except Exception:  # noqa: BLE001 -- an unreadable cache reads as not downloaded
        return None
    return None


def model_dir() -> Optional[str]:
    """The env override when set (a typo'd override is not silently replaced by the cache), else the cached repo."""
    if (os.environ.get(ENV_PATH_OVERRIDE) or "").strip():
        return _override_dir()
    return _cached_snapshot()


@dataclass(frozen = True)
class _SegmenterSpec:
    repo: str
    files: tuple
    label: str
    model_file: str = SEGMENTER_WEIGHTS_FILE
    mmproj_file: str = "config.json"


SEGMENTER_SPEC = _SegmenterSpec(repo = SEGMENTER_REPO, files = SEGMENTER_FILES, label = "SAM 3 segmenter")


def _resolve_segmenter_id(model_id: Optional[str]) -> str:
    if (model_id or SEGMENTER_ID) != SEGMENTER_ID:
        raise ValueError(f"Unknown segmenter '{model_id}'.")
    return SEGMENTER_ID


def describe_download_error(_model_id: str, error: object) -> Optional[str]:
    """A gated-repo refusal names the license to accept; anything else keeps the generic message."""
    name = type(error).__name__
    text = str(error)
    if (
        name == "GatedRepoError"
        or "gated" in text.lower()
        or "403" in text
        or "401" in text
        or "awaiting a review" in text
    ):
        return (
            f"{SEGMENTER_REPO} is gated on Hugging Face. Accept the SAM 3 license at {SEGMENTER_LICENSE_URL} "
            "with your account, add a Hugging Face token in Settings, then download again."
        )
    return None


_download_state_lock = threading.Lock()
_download_state = None


def _cached_download_paths(
    model_id: str,
    *,
    hub_cache: Optional[Path] = None,
    revision: Optional[str] = None,
):
    path = _cached_snapshot(model_id, hub_cache = hub_cache, revision = revision)
    return None if path is None else (path, path)


def _get_download_state():
    global _download_state
    with _download_state_lock:
        if _download_state is None:
            from core.inference.stt_mtmd_sidecar import _MtmdDownloadState
            _download_state = _MtmdDownloadState(
                specs = {SEGMENTER_ID: SEGMENTER_SPEC},
                cached_paths = _cached_download_paths,
                resolve = _resolve_segmenter_id,
                label = "SAM 3 segmenter",
                describe_error = describe_download_error,
            )
        return _download_state


def start_download(hf_token: Optional[str] = None) -> str:
    return _get_download_state().start(SEGMENTER_ID, hf_token)


def download_status() -> dict:
    return _get_download_state().status()


def cancel_download(download_id: Optional[str] = None) -> bool:
    return _get_download_state().cancel(SEGMENTER_ID, download_id)


def status() -> dict:
    """The status route body: can it run, is it downloaded, is it loaded (and where), and the download progress."""
    supported = transformers_supports_sam3()
    path = model_dir() if supported else None
    segmenter = _segmenter
    return {
        "supported": supported,
        "reason": None
        if supported
        else "Mask from text needs PyTorch and transformers 5 (Python 3.10 or newer); this install lacks one of them.",
        "downloaded": path is not None,
        "loaded": bool(segmenter is not None and segmenter.loaded),
        "device": segmenter.device if segmenter is not None and segmenter.loaded else None,
        "repo": SEGMENTER_REPO,
        "gated": True,
        "license_url": SEGMENTER_LICENSE_URL,
        "size_bytes": SEGMENTER_SIZE_BYTES,
        "override": bool((os.environ.get(ENV_PATH_OVERRIDE) or "").strip()),
        "download": download_status(),
    }


# ── device choice ───────────────────────────────────────────────────────────────────────────────────────────────────
def _resident_cuda_ordinal() -> Optional[int]:
    """The CUDA card the resident image model sits on, else None."""
    try:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        ordinal = getattr(
            getattr(get_active_diffusion_engine(), "_state", None), "placed_ordinal", None
        )
        return None if ordinal is None else int(ordinal)
    except Exception:  # noqa: BLE001
        return None


def _cuda_free_mib(ordinal: int) -> tuple[Optional[int], Optional[int]]:
    try:
        import torch
        free, total = torch.cuda.mem_get_info(ordinal)
        return int(free // (1024 * 1024)), int(total // (1024 * 1024))
    except Exception:  # noqa: BLE001
        return None, None


def vram_is_tight(free_mib: Optional[int], total_mib: Optional[int]) -> bool:
    if free_mib is None:
        return False
    floor = _TIGHT_FREE_MIB
    if total_mib:
        floor = max(floor, int(total_mib * _TIGHT_FREE_FRACTION))
    return free_mib < floor


def _other_owner_on(ordinal: Optional[int]) -> Optional[str]:
    """The GPU arbiter's resident other than the image model on CUDA card ``ordinal`` (one with unknown GPUs counts
    for every card; ``ordinal`` None is Apple's single Metal device, which every resident shares), else None. Called under the run lock after the handover fence: a later handover waits for the
    call (or aborts) instead."""
    try:
        from core.inference import gpu_arbiter
        held = gpu_arbiter.residents()
        physical = None if ordinal is None else gpu_arbiter.physical_gpu_ids([ordinal])
    except Exception:  # noqa: BLE001 -- no arbiter: nothing else is placed through it
        return None
    for owner, gpus in held.items():
        if owner == gpu_arbiter.DIFFUSION:
            continue
        if gpus is None or physical is None or gpus & physical:
            return owner
    return None


def choose_device(allow_gpu: bool = True) -> tuple[str, Any]:
    """(torch device string, dtype): the image model's CUDA card (else the current one) in bf16 (fp16 where bf16 is
    not supported) when it has room for the segmenter, Apple's MPS in fp32, else the CPU in fp32."""
    import torch

    if allow_gpu and torch.cuda.is_available():
        ordinal = _resident_cuda_ordinal()
        if ordinal is None or ordinal >= torch.cuda.device_count():
            ordinal = torch.cuda.current_device()
        other = _other_owner_on(ordinal)
        if other is not None:
            # A chat / video / other model owns the card: it grows its working memory without knowing about SAM 3.
            logger.info("images.segment: cuda:%s is held by %s, running on the CPU", ordinal, other)
            return "cpu", torch.float32
        free_mib, _total = _cuda_free_mib(ordinal)
        if free_mib is None or free_mib >= _GPU_NEED_MIB:
            bf16 = False
            with suppress(Exception):
                # is_bf16_supported reads the CURRENT card; the segmenter runs on ``ordinal``.
                with torch.cuda.device(ordinal):
                    bf16 = bool(torch.cuda.is_bf16_supported())
            return f"cuda:{ordinal}", torch.bfloat16 if bf16 else torch.float16
        logger.info("images.segment: %s MiB free on cuda:%s, running on the CPU", free_mib, ordinal)
    if (
        allow_gpu
        and getattr(getattr(torch, "backends", None), "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        other = _other_owner_on(None)
        if other is None:
            return "mps", torch.float32
        logger.info("images.segment: Metal is held by %s, running on the CPU", other)
    return "cpu", torch.float32


# ── inference ───────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Instance:
    concept: str
    score: float
    box: tuple[float, float, float, float]


@dataclass
class SegmentResult:
    mask_png: bytes
    width: int
    height: int
    instances: list[Instance] = field(default_factory = list)
    concepts: list[dict] = field(default_factory = list)
    device: str = "cpu"
    elapsed_ms: float = 0.0

    def to_json(self) -> dict:
        return {
            "mask": "data:image/png;base64," + base64.b64encode(self.mask_png).decode("ascii"),
            "width": self.width,
            "height": self.height,
            "instances": [
                {
                    "prompt": i.concept,
                    "score": round(i.score, 4),
                    "box": [round(v, 1) for v in i.box],
                }
                for i in self.instances
            ],
            "prompts": self.concepts,
            "device": self.device,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


def model_image_size(model: Any) -> int:
    """The square input the checkpoint was built for (1008 for SAM 3)."""
    try:
        return int(model.config.vision_config.backbone_config.image_size)
    except Exception:  # noqa: BLE001
        return IMAGE_SIZE


def preprocess(
    image: Any,
    device: str,
    dtype: Any,
    size: int = IMAGE_SIZE,
) -> Any:
    """ComfyUI's input: RGB in [0, 1] stretched to 1008x1008 with bilinear interpolation (no crop, no antialias),
    then SAM 3's normalisation (mean 0.5, std 0.5)."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    array = np.array(image.convert("RGB"), dtype = np.uint8)
    tensor = torch.from_numpy(array).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = F.interpolate(tensor, size = (size, size), mode = "bilinear", align_corners = False)
    return ((tensor - 0.5) / 0.5).to(dtype)


def select_detections(
    pred_logits: Any,
    pred_boxes: Any,
    pred_masks: Any,
    *,
    max_detections: int,
    threshold: float,
    presence_logit: Any = None,
) -> tuple[Any, Any, Any]:
    """ComfyUI's keep rule for one concept: score > threshold, best first, at most ``max_detections``, with the score
    sigmoid(logit) * sigmoid(presence) when the presence logit is given. Takes one image's (Q,), (Q, 4) normalised
    xyxy and (Q, h, w) logits; returns the kept scores, boxes, masks."""
    probs = pred_logits.float().sigmoid()
    if presence_logit is not None:
        probs = probs * presence_logit.float().sigmoid().reshape(-1)[0]
    keep = probs > threshold
    probs, boxes, masks = probs[keep], pred_boxes[keep], pred_masks[keep]
    order = probs.argsort(descending = True)[:max_detections]
    return probs[order], boxes[order], masks[order]


def union_mask(masks: Any, height: int, width: int) -> Any:
    """(N, h, w) mask logits -> one (height, width) bool mask: bilinear upsample, > 0, any."""
    import torch
    import torch.nn.functional as F

    if masks is None or masks.shape[0] == 0:
        return torch.zeros(
            height, width, dtype = torch.bool, device = None if masks is None else masks.device
        )
    union = torch.zeros(height, width, dtype = torch.bool, device = masks.device)
    # One instance at a time keeps a 4K source to a single full-size float plane.
    for logits in masks:
        up = F.interpolate(
            logits[None, None].float(), size = (height, width), mode = "bilinear", align_corners = False
        )[0, 0]
        union |= up > 0
    return union


def mask_to_png(mask: Any) -> bytes:
    from PIL import Image

    array = (mask.to("cpu").numpy().astype("uint8")) * 255
    buffer = io.BytesIO()
    Image.fromarray(array, mode = "L").save(buffer, format = "PNG", optimize = True)
    return buffer.getvalue()


def _free_device_cache(device: Optional[str]) -> None:
    """Collect what nothing references any more, then hand ``device``'s cached blocks back to the driver."""
    import gc

    gc.collect()
    if device and device.startswith("cuda"):
        with suppress(Exception):
            import torch
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    elif device == "mps":
        with suppress(Exception):
            import torch
            torch.mps.empty_cache()


class MaskSegmenter:
    """One lazily loaded Sam3Model + its tokenizer. ``_run_lock`` serialises loads, calls and unloads."""

    def __init__(self, keep_alive_seconds: Optional[float] = None) -> None:
        if keep_alive_seconds is None:
            try:
                keep_alive_seconds = float(os.environ.get(ENV_KEEP_ALIVE) or KEEP_ALIVE_SECONDS)
            except ValueError:
                keep_alive_seconds = KEEP_ALIVE_SECONDS
        self.keep_alive_seconds = max(0.0, keep_alive_seconds)
        self._run_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._path: Optional[str] = None
        self.device: Optional[str] = None
        self._dtype = None
        self._timer: Optional[threading.Timer] = None
        self._release_pending = False
        self._release_epoch = 0
        # Bumped by every GPU handover's release: a call that queued before one never runs on the accelerator.
        self._handover_epoch = 0
        # The device of the call in flight: "" until it is chosen, None when no call runs.
        self._call_device: Optional[str] = None
        # Set (under _state_lock) once the call in flight has read its release flag: a release that times out after
        # that is not honoured by the call, so it waits out the call's short tail itself.
        self._call_settled = False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def holds_vram(self) -> bool:
        return self._model is not None and bool(self.device) and self.device != "cpu"

    # -- lifecycle -------------------------------------------------------------------------------------------------
    def _load(self, path: str, device: str, dtype: Any) -> None:
        if (
            self._model is not None
            and self._path == path
            and self.device == device
            and self._dtype == dtype
        ):
            return
        if self._model is not None:
            self._unload_locked("reload")
        from transformers import AutoTokenizer, Sam3Model

        started = time.perf_counter()
        # The tokenizer first: a broken one fails before any weights are placed on the device.
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = None
        try:
            if device.startswith("cuda"):
                # Straight onto the card: about a third faster than loading on the CPU and moving it.
                model = Sam3Model.from_pretrained(path, dtype = dtype, device_map = device)
            else:
                model = Sam3Model.from_pretrained(path, dtype = dtype)
                model.to(device)
            model.eval()
        except BaseException:
            # Weights placed before the failure are referenced nowhere else: drop them and their cached blocks.
            model = None
            _free_device_cache(device)
            raise
        self._model, self._tokenizer = model, tokenizer
        self._path, self.device, self._dtype = path, device, dtype
        logger.info(
            "images.segment: loaded %s on %s (%s) in %.1fs",
            path,
            device,
            str(dtype).replace("torch.", ""),
            time.perf_counter() - started,
        )

    def _unload_locked(self, reason: str) -> bool:
        with self._state_lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        if self._model is None:
            return False
        device = self.device
        self._model = None
        self._tokenizer = None
        self._path = None
        self.device = None
        self._dtype = None
        _free_device_cache(device)
        logger.info("images.segment: unloaded from %s (%s)", device, reason)
        return True

    def _call_on_accelerator(self) -> bool:
        """A call is in flight on the GPU / MPS, or has not chosen its device yet."""
        device = self._call_device
        return device is not None and device != "cpu"

    def release(
        self,
        reason: str = "release",
        *,
        wait: bool = False,
        timeout: float = 30.0,
        handover: bool = False,
    ) -> bool:
        """Unload now if idle. A call in flight unloads when it ends (``wait`` waits up to ``timeout`` for it
        instead). True when the weights left now. ``handover`` (the GPU is changing hands) raises
        SegmenterReleaseTimeoutError when the wait ran out on a call still on the accelerator."""
        with self._state_lock:
            self._release_epoch += 1
            if handover:
                self._handover_epoch += 1
        acquired = (
            self._run_lock.acquire(timeout = timeout)
            if wait
            else self._run_lock.acquire(blocking = False)
        )
        if not acquired:
            with self._state_lock:
                self._release_pending = True
                # Read with the flag: a call that has not settled yet sees the flag and unloads when it ends.
                honoured = self._call_device is not None and not self._call_settled
                on_accelerator = self._call_on_accelerator()
            if wait and not honoured:
                # The call ended (or is ending) without seeing the flag: its tail is short, so wait it out and unload
                # here rather than leave the weights resident for the next owner.
                acquired = self._run_lock.acquire(timeout = _SETTLE_TIMEOUT_SECONDS)
        if not acquired:
            if handover and on_accelerator:
                raise SegmenterReleaseTimeoutError(
                    "A mask from text is still running on the GPU; GPU ownership cannot be transferred yet."
                )
            return False
        try:
            with self._state_lock:
                self._release_pending = False
            return self._unload_locked(reason)
        finally:
            self._run_lock.release()

    def _arm_keep_alive(self) -> None:
        with self._state_lock:
            if self._timer is not None:
                self._timer.cancel()
            epoch = self._release_epoch
            timer = threading.Timer(self.keep_alive_seconds, self._expire, args = (epoch,))
            timer.daemon = True
            self._timer = timer
            timer.start()

    def _expire(self, epoch: int) -> None:
        if not self._run_lock.acquire(blocking = False):
            return  # in use: the call re-arms the keep-alive when it ends
        try:
            with self._state_lock:
                if epoch != self._release_epoch and self._model is None:
                    return
            self._unload_locked("keep-alive expired")
        finally:
            self._run_lock.release()

    def _after_call(self) -> None:
        """Called with ``_run_lock`` held: honour a release that arrived mid-call, drop a GPU copy when the card is
        tight, else keep it for the keep-alive."""
        with self._state_lock:
            pending, self._release_pending = self._release_pending, False
            self._call_settled = True
        if pending:
            self._unload_locked("released during a call")
            return
        if self.device and self.device.startswith("cuda"):
            free_mib, total_mib = _cuda_free_mib(int(self.device.split(":", 1)[1]))
            if vram_is_tight(free_mib, total_mib):
                self._unload_locked(f"tight: {free_mib} of {total_mib} MiB free")
                return
        if self.keep_alive_seconds <= 0:
            self._unload_locked("no keep-alive")
            return
        self._arm_keep_alive()

    # -- inference -------------------------------------------------------------------------------------------------
    def segment(
        self,
        image: Any,
        prompt: str,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        allow_gpu: bool = True,
        path: Optional[str] = None,
    ) -> SegmentResult:
        concepts = parse_prompt(prompt)
        if not concepts:
            raise ValueError("Describe what to mask, for example: sofa, red lamp, pillow:2")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        if not transformers_supports_sam3():
            raise SegmenterUnavailableError(
                "This install's transformers has no SAM 3. It needs transformers 5 (Python 3.10 or newer)."
            )
        path = path or model_dir()
        if path is None:
            raise SegmenterNotDownloadedError(
                f"The SAM 3 segmenter is not downloaded. Download {SEGMENTER_REPO} first."
            )
        queued_at = self._handover_epoch
        with self._run_lock:
            if allow_gpu and (self._handover_epoch != queued_at or _gpu_handover_in_progress()):
                # The GPU changed hands while this call waited, or is changing hands now: the new owner may still be
                # loading onto the card.
                raise SegmenterBusyError("The GPU is changing hands.")
            with self._state_lock:
                self._call_device, self._call_settled = "", False
            try:
                return self._segment_locked(image, concepts, threshold, path, allow_gpu)
            finally:
                with self._state_lock:
                    self._call_device, self._call_settled = None, False

    def _segment_locked(
        self, image: Any, concepts: list[Concept], threshold: float, path: str, allow_gpu: bool
    ) -> SegmentResult:
        started = time.perf_counter()
        device, dtype = choose_device(allow_gpu)
        self._call_device = device
        oom = False
        try:
            result = self._segment_on(image, concepts, threshold, path, device, dtype)
        except Exception as exc:
            if not _is_oom(exc) or device == "cpu":
                self._after_call()
                raise
            oom = True
        if oom:
            # Out of the except block: the OOM's traceback (and the frames pinning the failed attempt's tensors) is
            # gone, so the unload and the cache release below actually free the card before the call reads CPU-only.
            logger.warning("images.segment: out of memory on %s, retrying on the CPU", device)
            self._unload_locked("out of memory")
            _free_device_cache(device)  # also when the OOM hit mid-load, before any model was kept
            import torch

            device, dtype = "cpu", torch.float32
            self._call_device = device
            try:
                result = self._segment_on(image, concepts, threshold, path, device, dtype)
            except Exception:
                # The CPU copy and any release asked for mid-call are settled here too, not left resident.
                self._after_call()
                raise
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._after_call()
        return result

    def _segment_on(
        self,
        image: Any,
        concepts: list[Concept],
        threshold: float,
        path: str,
        device: str,
        dtype: Any,
    ) -> SegmentResult:
        import torch

        self._load(path, device, dtype)
        model, tokenizer = self._model, self._tokenizer
        width, height = image.size
        scale = torch.tensor([width, height, width, height], dtype = torch.float32)
        union = torch.zeros(height, width, dtype = torch.bool, device = device)
        instances: list[Instance] = []
        summary: list[dict] = []
        with torch.inference_mode():
            pixels = preprocess(image, device, dtype, model_image_size(model))
            vision = model.get_vision_features(pixel_values = pixels)
            # Duplicates are encoded once; each concept keeps its own cap, as ComfyUI runs them one by one.
            texts = list(dict.fromkeys(c.text for c in concepts))
            tokens = tokenizer(
                texts,
                padding = "max_length",
                max_length = TEXT_MAX_LENGTH,
                truncation = True,
                return_tensors = "pt",
            ).to(device)
            # Only each text's kept detections (best first, up to its largest cap) outlive its forward, never the raw
            # output's 200 mask planes; a repeat with a smaller cap takes the best of them, as its own run would.
            caps: dict[str, int] = {}
            for concept in concepts:
                caps[concept.text] = max(caps.get(concept.text, 0), concept.max_detections)
            # Per text only its kept scores and boxes outlive the forward: the union of a text's best-``cap`` masks
            # covers every occurrence's best-``k`` (k <= cap), so the masks are OR-ed in at once and dropped.
            kept: dict[str, tuple[Any, Any]] = {}
            for concept in concepts:
                if concept.text not in kept:
                    index = texts.index(concept.text)
                    out = model(
                        vision_embeds = vision,
                        input_ids = tokens["input_ids"][index : index + 1],
                        attention_mask = tokens["attention_mask"][index : index + 1],
                    )
                    presence = getattr(out, "presence_logits", None)
                    scores, boxes, masks = select_detections(
                        out.pred_logits[0],
                        out.pred_boxes[0],
                        out.pred_masks[0],
                        max_detections = caps[concept.text],
                        threshold = threshold,
                        presence_logit = None if presence is None else presence[0],
                    )
                    del out, presence
                    union |= union_mask(masks, height, width)
                    del masks
                    kept[concept.text] = (scores, boxes)
                scores, boxes = (t[: concept.max_detections] for t in kept[concept.text])
                boxes_px = (boxes.float().cpu() * scale).tolist()
                for score, box in zip(scores.tolist(), boxes_px):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    instances.append(
                        Instance(
                            concept.text,
                            float(score),
                            (
                                max(0.0, x1),
                                max(0.0, y1),
                                min(float(width), x2),
                                min(float(height), y2),
                            ),
                        )
                    )
                summary.append(
                    {
                        "text": concept.text,
                        "max_detections": concept.max_detections,
                        "found": int(scores.shape[0]),
                    }
                )
            del kept, vision, pixels
        return SegmentResult(
            mask_png = mask_to_png(union),
            width = width,
            height = height,
            instances = instances,
            concepts = summary,
            device = device,
        )


# How long a timed-out release waits for a call that already settled (unload / keep-alive, then the lock) to let go.
_SETTLE_TIMEOUT_SECONDS = 5.0


def _gpu_handover_in_progress() -> bool:
    """The arbiter is handing the GPU to a new owner (its load still registering). Never imports the arbiter."""
    arbiter = sys.modules.get("core.inference.gpu_arbiter")
    probe = getattr(arbiter, "handover_in_progress", None)
    try:
        return bool(probe()) if callable(probe) else False
    except Exception:  # noqa: BLE001
        return False


def _is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in ("OutOfMemoryError",) or "out of memory" in str(exc).lower()


_segmenter: Optional[MaskSegmenter] = None
_segmenter_lock = threading.Lock()


def get_segmenter() -> MaskSegmenter:
    global _segmenter
    with _segmenter_lock:
        if _segmenter is None:
            _segmenter = MaskSegmenter()
        return _segmenter


def release_segmenter(
    reason: str = "release",
    *,
    wait: bool = False,
    handover: bool = False,
) -> bool:
    """Drop the weights if they were ever loaded. Never creates the segmenter. ``handover``: see
    MaskSegmenter.release."""
    segmenter = _segmenter
    return segmenter is not None and segmenter.release(reason, wait = wait, handover = handover)


_renders_pending = 0
_renders_lock = threading.Lock()


@contextmanager
def render_fence() -> Iterator[None]:
    """Held by an image render from before it releases the segmenter until it ends: no Find starts meanwhile."""
    global _renders_pending
    with _renders_lock:
        _renders_pending += 1
    try:
        yield
    finally:
        with _renders_lock:
            _renders_pending -= 1


def _take_render_slot() -> tuple[Any, str]:
    """(the active image engine's generation lock, acquired) or (None, why not). ``generating`` while a render or
    a model transition holds it, ``loading`` while a load is in flight; an engine with no lock (or none readable)
    runs nothing on torch, so the call proceeds with no slot. Never blocks."""
    try:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        engine = get_active_diffusion_engine()
    except Exception:  # noqa: BLE001 -- no engine: nothing to collide with
        return None, ""
    loading = getattr(engine, "_loading", None)
    if loading is not None and not getattr(loading, "error", None):
        return None, "loading"
    lock = getattr(engine, "_generate_lock", None)
    if lock is None:
        return None, ""
    if not lock.acquire(blocking = False):
        return None, "generating"
    loading = getattr(engine, "_loading", None)
    if loading is not None and not getattr(loading, "error", None):
        lock.release()
        return None, "loading"
    return lock, ""


def segment_image(
    image_b64: str,
    prompt: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    allow_gpu: bool = True,
) -> dict:
    """The segment route's work: decode the upload (EXIF-upright, as inpaint decodes it), refuse while the image
    model renders or loads, run SAM 3, answer the route body."""
    from core.inference.diffusion import decode_b64_image

    if not parse_prompt(prompt):
        raise ValueError("Describe what to mask, for example: sofa, red lamp, pillow:2")
    image = decode_b64_image(image_b64, mode = "RGB")
    slot, why = _take_render_slot()
    if not why and _renders_pending:
        # A render released SAM 3 and is on its way to the slot: a Find taking it first would reload SAM 3 under it.
        if slot is not None:
            slot.release()
        slot, why = None, "generating"
    if why:
        raise SegmenterBusyError(
            "The image model is generating."
            if why == "generating"
            else "An image model is loading."
        )
    try:
        return (
            get_segmenter()
            .segment(image, prompt, threshold = threshold, allow_gpu = allow_gpu)
            .to_json()
        )
    finally:
        if slot is not None:
            slot.release()


__all__ = [
    "Concept",
    "DEFAULT_THRESHOLD",
    "MaskSegmenter",
    "SEGMENTER_REPO",
    "SegmenterBusyError",
    "SegmenterNotDownloadedError",
    "SegmenterReleaseTimeoutError",
    "SegmenterUnavailableError",
    "get_segmenter",
    "parse_prompt",
    "release_segmenter",
    "segment_image",
    "status",
]
