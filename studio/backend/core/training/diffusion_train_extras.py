# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in extras: LoRA EMA (0.99 default decay, exported as a second adapter), conditioning cache
(bit-identical to a fresh encode), aspect bucketing; pure helpers avoid torch at import time"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable, Optional


# Effective decay = min(decay, (1 + n) / (WARMUP_OFFSET + n)).

_EMA_WARMUP_OFFSET = 10.0


class LoRAEMA:
    """Exponential moving average over a model's TRAINABLE parameters only.

    For a LoRA run the trainable set is just the adapter matrices, so the
    shadow costs megabytes, not the gigabytes a full-model EMA would. Shadows
    are stored keyed by parameter name (stable across re-wraps) on the same
    device/dtype as the source params, and updated in-place:

        shadow = decay * shadow + (1 - decay) * param
    """

    def __init__(
        self,
        model: Any,
        decay: float = 0.99,
        warmup: bool = True,
    ):
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError(f"ema decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.updates = 0
        self._shadow: dict[str, Any] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = p.detach().clone()
            shadow.requires_grad = False
            self._shadow[name] = shadow

    def reseed_from(self, model: Any) -> None:
        """For a resume that first enables EMA: shadows would otherwise hold freshly initialised weights."""
        import torch

        with torch.no_grad():
            for name, p in model.named_parameters():
                shadow = self._shadow.get(name)
                if shadow is None or tuple(shadow.shape) != tuple(p.shape):
                    continue
                shadow.copy_(p.detach().to(device = shadow.device, dtype = shadow.dtype))
        self.updates = 0

    def effective_decay(self) -> float:
        """The decay used for the NEXT update (after ``updates`` prior ones)."""
        if not self.warmup:
            return self.decay
        step = self.updates + 1
        return min(self.decay, step / (_EMA_WARMUP_OFFSET + step))

    def update(self, model: Any) -> None:
        """Apply one EMA update from ``model``'s current trainable params."""
        import torch

        decay = self.effective_decay()
        with torch.no_grad():
            for name, p in model.named_parameters():
                shadow = self._shadow.get(name)
                if shadow is None:
                    continue
                shadow.mul_(decay).add_(p.detach(), alpha = 1.0 - decay)
        self.updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {name: t.detach().clone() for name, t in self._shadow.items()}

    def missing_from(self, state: dict[str, Any]) -> tuple[str, ...]:
        """load_state_dict skips missing shadows silently, which would blend restored and fresh weights."""
        saved = state or {}
        missing = []
        for name, shadow in self._shadow.items():
            entry = saved.get(name)
            if entry is None or tuple(entry.shape) != tuple(shadow.shape):
                missing.append(name)
        return tuple(missing)

    def load_state_dict(
        self,
        state: dict[str, Any],
        updates: int = 0,
    ) -> None:
        """Pass updates to keep the warmup ramp position; otherwise the ramp restarts on resume."""
        import torch

        with torch.no_grad():
            for name, shadow in self._shadow.items():
                saved = (state or {}).get(name)
                if saved is None or tuple(saved.shape) != tuple(shadow.shape):
                    continue
                shadow.copy_(saved.to(device = shadow.device, dtype = shadow.dtype))
        self.updates = max(0, int(updates or 0))

    def copy_to(self, model: Any) -> dict[str, Any]:
        """Write the shadow values into ``model``'s params, returning the
        displaced originals so ``restore`` can undo the swap."""
        import torch

        backup: dict[str, Any] = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                shadow = self._shadow.get(name)
                if shadow is None:
                    continue
                backup[name] = p.detach().clone()
                p.copy_(shadow)
        return backup

    def restore(self, model: Any, backup: dict[str, Any]) -> None:
        import torch
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in backup:
                    p.copy_(backup[name])

    def __len__(self) -> int:
        return len(self._shadow)


def save_ema_adapter(ema: "LoRAEMA", transformer: Any, spec_save: Any, out_dir: str) -> str:
    """Temporarily swaps EMA shadows into the live LoRA params for the diffusers export path, then restores the trained weights."""
    from peft.utils import get_peft_model_state_dict

    ema_dir = Path(out_dir) / "ema"
    ema_dir.mkdir(parents = True, exist_ok = True)
    backup = ema.copy_to(transformer)
    try:
        layers = get_peft_model_state_dict(transformer)
        spec_save(None, str(ema_dir), layers)
    finally:
        ema.restore(transformer, backup)
    return str(ema_dir)


_CACHE_VERSION = "1"
# v2 composites transparent images onto white before encoding.
_LATENT_CACHE_VERSION = "2"


def _file_content_hash(path: str) -> str:
    """return a truncated sha256; renames preserve cache entries while edits invalidate them."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:20]


def _text_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _sanitize(token: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", str(token).lower()).strip("-")


def _hub_cache_roots() -> list[str]:
    """HF_HUB_CACHE is an import-time snapshot, so the active Unsloth cache root is searched first."""
    import os  # noqa: PLC0415 - keep the module import list light for the subprocess

    roots: list[str] = []
    try:
        from utils.hf_cache_settings import active_hf_hub_cache  # noqa: PLC0415
        active = str(active_hf_hub_cache() or "").strip()
        if active:
            roots.append(active)
    except Exception:  # noqa: BLE001 -- settings unavailable in the subprocess: fall back
        pass
    for candidate in (os.environ.get("HF_HUB_CACHE"), os.environ.get("HUGGINGFACE_HUB_CACHE")):
        if candidate and candidate.strip() and candidate.strip() not in roots:
            roots.append(candidate.strip())
    try:
        from huggingface_hub import constants  # noqa: PLC0415
        if constants.HF_HUB_CACHE and str(constants.HF_HUB_CACHE) not in roots:
            roots.append(str(constants.HF_HUB_CACHE))
    except Exception:  # noqa: BLE001 -- no hub package: whatever we collected above stands
        pass
    return roots


# In-place edits here must change the fingerprint; "connectors" is LTX-2's cached connector.
_CACHE_SOURCE_SUBDIRS = ("text_encoder", "tokenizer", "vae", "connectors")


def source_revision(ref: Any) -> str:
    """A repo id or path is not a version; the marker tracks content so stale cache is not reused; it must never load the encoders."""
    import os  # noqa: PLC0415 - keep the module import list light for the subprocess
    try:
        name = str(ref or "").strip()
        if not name:
            return "none"
        if os.path.isdir(name):
            parts: list[str] = []
            roots = [name]
            with os.scandir(name) as it:
                roots += [
                    e.path for e in it if e.is_dir() and e.name.startswith(_CACHE_SOURCE_SUBDIRS)
                ]
            for root in roots:
                with os.scandir(root) as it:
                    for e in it:
                        if not e.is_file():
                            continue
                        st = e.stat()
                        parts.append(
                            f"{os.path.relpath(e.path, name)}:{st.st_size}:{st.st_mtime_ns}"
                        )
            return f"dir-{hashlib.sha256('|'.join(sorted(parts)).encode()).hexdigest()[:16]}"
        if "/" in name:
            org, _, repo = name.partition("/")
            for root in _hub_cache_roots():
                base = os.path.join(root, f"models--{org}--{repo}")
                ref_file = os.path.join(base, "refs", "main")
                if os.path.isfile(ref_file):
                    with open(ref_file, encoding = "utf-8") as fh:
                        sha = fh.read().strip()
                    if sha:
                        return f"rev-{sha[:16]}"
                snaps = os.path.join(base, "snapshots")
                if os.path.isdir(snaps):
                    names = sorted(os.listdir(snaps))
                    if len(names) == 1:
                        return f"rev-{names[0][:16]}"
        return "unresolved"
    except Exception:  # noqa: BLE001 - best-effort, never block a run
        return "unresolved"


class PersistentConditioningCache:
    """One safetensors file per cached item under ``cache_dir``.

    Keys carry everything that changes the encoded tensors: the cache format
    version, family, resolution (or explicit bucket shape), the source content
    hash, and -- for latents -- the crop/flip variant. Tensors are stored fp32
    exactly as the trainer's in-memory path holds them, so a reload is
    bit-identical to a fresh encode."""

    def __init__(self, cache_dir: str, family: str, resolution: int):
        self.root = Path(cache_dir).expanduser()
        self.family = _sanitize(family)
        self.resolution = int(resolution)
        self.root.mkdir(parents = True, exist_ok = True)

    def latent_key(
        self,
        image_path: str,
        variant: tuple[float, float, bool],
        shape: Optional[tuple[int, int]] = None,
    ) -> str:
        u_left, u_top, flip = variant
        geom = f"{shape[0]}x{shape[1]}" if shape else str(self.resolution)
        var = f"{u_left:.6f}_{u_top:.6f}_{int(bool(flip))}"
        return (
            f"lat_v{_LATENT_CACHE_VERSION}_{self.family}_{geom}_"
            f"{_file_content_hash(image_path)}_{var}"
        )

    def text_key(self, caption: str) -> str:
        return f"txt_v{_CACHE_VERSION}_{self.family}_{_text_content_hash(caption)}"

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.safetensors"

    def has(self, key: str) -> bool:
        return self.path_for(key).is_file()

    def put(self, key: str, tensors: Iterable[Any]) -> None:
        """store an ordered tensor tuple with None slots recorded for exact reconstruction."""
        from safetensors.torch import save_file

        named: dict[str, Any] = {}
        none_slots: list[int] = []
        for i, t in enumerate(tensors):
            if t is None:
                none_slots.append(i)
            else:
                named[f"t{i}"] = t.detach().cpu().contiguous()
        meta = {
            "version": _CACHE_VERSION,
            "family": self.family,
            "none_slots": json.dumps(none_slots),
            "count": str(len(none_slots) + len(named)),
        }
        tmp = self.path_for(key).with_suffix(".tmp")
        save_file(named, str(tmp), metadata = meta)
        tmp.replace(self.path_for(key))

    def get(self, key: str) -> Optional[tuple]:
        """Load a tuple previously stored with ``put``; None if absent/corrupt."""
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            from safetensors import safe_open
            with safe_open(str(path), framework = "pt", device = "cpu") as f:
                meta = f.metadata() or {}
                count = int(meta.get("count", "0"))
                none_slots = set(json.loads(meta.get("none_slots", "[]")))
                out: list[Any] = []
                for i in range(count):
                    out.append(None if i in none_slots else f.get_tensor(f"t{i}"))
            return tuple(out)
        except Exception:  # noqa: BLE001 -- a corrupt entry is re-encoded, never fatal
            return None


# 64: a multiple of every family's latent grid (16 DiTs, 32 LTX-2, SDXL U-Net) and few compiled shapes.

BUCKET_DIVISOR = 64

# Widest bucket aspect ratio; more extreme images clamp to it, as is common for panoramas.
MAX_BUCKET_RATIO = 2.0


def bucket_resolutions(
    resolution: int,
    divisor: int = BUCKET_DIVISOR,
    max_ratio: float = MAX_BUCKET_RATIO,
) -> list[tuple[int, int]]:
    """The exact legacy square plus kohya BucketManager buckets (tallest grid height within resolution**2)."""
    base = int(resolution)
    area = base * base
    out = {(base, base)}
    w = divisor
    while w * w <= area * max_ratio:
        h = (area // w) // divisor * divisor
        if h >= divisor and 1.0 / max_ratio <= w / h <= max_ratio:
            out.add((w, h))
            out.add((h, w))
        w += divisor
    return sorted(out)


def nearest_bucket(width: int, height: int, buckets: list[tuple[int, int]]) -> tuple[int, int]:
    """Closest aspect in log space; ties: larger area, then sorted order (deterministic)."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    r = math.log(width / height)
    return min(buckets, key = lambda b: (abs(math.log(b[0] / b[1]) - r), -b[0] * b[1], b))


def compute_bucket(
    width: int,
    height: int,
    base_resolution: int,
    divisor: int = BUCKET_DIVISOR,
    max_ratio: float = MAX_BUCKET_RATIO,
) -> tuple[int, int]:
    """The (bucket_w, bucket_h) for an image of ``width`` x ``height``."""
    return nearest_bucket(width, height, bucket_resolutions(base_resolution, divisor, max_ratio))


def assign_buckets(
    sizes: list[tuple[int, int]],
    base_resolution: int,
    divisor: int = BUCKET_DIVISOR,
    max_ratio: float = MAX_BUCKET_RATIO,
) -> dict[tuple[int, int], list[int]]:
    choices = bucket_resolutions(base_resolution, divisor, max_ratio)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (w, h) in enumerate(sizes):
        buckets.setdefault(nearest_bucket(w, h, choices), []).append(i)
    return buckets


def cover_resize_dims(width: int, height: int, canvas_w: int, canvas_h: int) -> tuple[int, int]:
    """Smallest uniform scale covering the canvas (square canvas = legacy short-side resize)."""
    scale = max(canvas_w / width, canvas_h / height)
    return max(canvas_w, round(width * scale)), max(canvas_h, round(height * scale))


def oriented_image_size(path: str) -> tuple[int, int]:
    """(width, height) after EXIF orientation, from the header only (no pixel decode)."""
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
        try:
            orientation = im.getexif().get(0x0112, 1)
        except Exception:  # noqa: BLE001 -- unreadable EXIF is treated as upright, like exif_transpose
            orientation = 1
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)


class BucketBatchSampler:
    """Batch indices so every batch comes from ONE bucket (one latent shape).

    Epoch-style like PermutationBatchSampler: every image once per cycle; a bucket's last partial
    batch wraps within itself so shapes stay fixed."""

    def __init__(self, buckets: dict[tuple[int, int], list[int]], rng: random.Random):
        if not buckets or not any(buckets.values()):
            raise ValueError("BucketBatchSampler needs at least one bucketed item")
        self._rng = rng
        self._shapes = sorted(s for s in buckets if buckets[s])
        self._items = {s: list(buckets[s]) for s in self._shapes}
        self._batches: list[tuple[tuple[int, int], list[int]]] = []
        self._pos = 0

    def _refill(self, k: int) -> None:
        batches: list[tuple[tuple[int, int], list[int]]] = []
        for shape in self._shapes:
            order = list(self._items[shape])
            self._rng.shuffle(order)
            count = -(-len(order) // k)
            stream = (order * (-(-count * k // len(order))))[: count * k]
            batches.extend((shape, stream[b * k : (b + 1) * k]) for b in range(count))
        self._rng.shuffle(batches)
        self._batches, self._pos = batches, 0

    def next_batch(self, k: int) -> tuple[tuple[int, int], list[int]]:
        if self._pos >= len(self._batches) or len(self._batches[self._pos][1]) != k:
            self._refill(k)
        shape, idxs = self._batches[self._pos]
        self._pos += 1
        return shape, list(idxs)

    def _bucket_key(self) -> list[list[Any]]:
        return [[s[0], s[1], list(self._items[s])] for s in self._shapes]

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "bucket",
            "buckets": self._bucket_key(),
            "batches": [[s[0], s[1], list(i)] for s, i in self._batches],
            "pos": int(self._pos),
        }

    def load_state_dict(self, state: Optional[dict[str, Any]]) -> bool:
        """False (never a silent fresh cycle) unless the state matches this exact bucket assignment."""
        if not isinstance(state, dict) or state.get("kind") != "bucket":
            return False
        try:
            if [[int(w), int(h), [int(i) for i in idx]] for w, h, idx in state["buckets"]] != (
                self._bucket_key()
            ):
                return False
            batches = [((int(w), int(h)), [int(i) for i in idx]) for w, h, idx in state["batches"]]
            pos = int(state.get("pos") or 0)
        except (KeyError, TypeError, ValueError):
            return False
        for shape, idxs in batches:
            if shape not in self._items or not set(idxs) <= set(self._items[shape]):
                return False
        if not 0 <= pos <= len(batches):
            return False
        self._batches, self._pos = batches, pos
        return True
