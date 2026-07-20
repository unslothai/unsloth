# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Optional extras for the diffusion LoRA trainers: LoRA EMA, the persistent
conditioning cache, and aspect-ratio bucketing.

Everything here is opt-in and family-agnostic:

- ``LoRAEMA`` keeps an exponential moving average of ONLY the trainable LoRA
  parameters. Short LoRA runs (a few hundred steps) can't afford the classic
  0.9999 decay -- the shadow would still be ~frozen at the init when the run
  ends -- so the default decay is 0.99 with a warmup ramp that keeps early
  updates responsive. The EMA weights export as a SECOND adapter next to the
  primary one, so the user can A/B both.

- ``PersistentConditioningCache`` stores the trainer's precomputed conditioning
  tensors (VAE latent posterior stats + text-encoder embeddings) on disk as one
  safetensors file per item, keyed by content hash + family + resolution. On a
  warm start the trainer can skip loading the VAE and the multi-GB text
  encoders entirely. The cached latent tensors are EXACTLY what the in-memory
  path produces (the family's ``encode_latent_stats`` output with all
  normalisation -- including Qwen's per-channel latents_mean/std -- already
  folded in, held in fp32), so a cache hit is bit-identical to a fresh encode.

- Aspect-ratio bucketing assigns each image a same-area bucket resolution
  (width/height multiples of a divisor the VAE + patching can consume) and
  batches only within a bucket, so a mixed-aspect dataset trains without
  square-cropping away composition.

Pure helpers avoid torch at import time so the CPU unit tests stay light.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable, Optional

# ── LoRA EMA ──────────────────────────────────────────────────────────────────

# Warmup horizon for the EMA decay ramp: effective decay is
# min(decay, (1 + updates) / (WARMUP_OFFSET + updates)), the standard inverse
# ramp. With the offset at 10, step 1 averages aggressively (~0.18) and the
# ramp reaches 0.99 after ~1000 updates, so a 300-step run still ends with a
# shadow that has absorbed most of the trajectory instead of the init.
_EMA_WARMUP_OFFSET = 10.0


class LoRAEMA:
    """Exponential moving average over a model's TRAINABLE parameters only.

    For a LoRA run the trainable set is just the adapter matrices, so the
    shadow costs megabytes, not the gigabytes a full-model EMA would. Shadows
    are stored keyed by parameter name (stable across re-wraps) on the same
    device/dtype as the source params, and updated in-place:

        shadow = decay * shadow + (1 - decay) * param
    """

    def __init__(self, model: Any, decay: float = 0.99, warmup: bool = True):
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
    """Export the EMA weights as a second adapter under ``out_dir``/ema.

    Temporarily swaps the shadow values into the live LoRA params so
    ``get_peft_model_state_dict`` serialises them through the exact same
    (diffusers-format) path as the primary adapter, then restores the trained
    weights. Returns the ema output directory."""
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


# ── persistent conditioning cache ─────────────────────────────────────────────

_CACHE_VERSION = "1"


def _file_content_hash(path: str) -> str:
    """sha256 of the file bytes (truncated hex): renames/moves keep their cache
    entries, edits invalidate them."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:20]


def _text_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _sanitize(token: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", str(token).lower()).strip("-")


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

    # -- keys --
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
            f"lat_v{_CACHE_VERSION}_{self.family}_{geom}_"
            f"{_file_content_hash(image_path)}_{var}"
        )

    def text_key(self, caption: str) -> str:
        return f"txt_v{_CACHE_VERSION}_{self.family}_{_text_content_hash(caption)}"

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.safetensors"

    def has(self, key: str) -> bool:
        return self.path_for(key).is_file()

    # -- IO --
    def put(self, key: str, tensors: Iterable[Any]) -> None:
        """Store an ordered tuple of tensors (None entries allowed: their slot
        indices are recorded in the metadata so ``get`` restores them)."""
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


# ── aspect-ratio bucketing ────────────────────────────────────────────────────

# Pixel-dimension divisor for bucket shapes. The DiT families divide by 8 in
# the VAE and 2 again in latent patching, and regional torch.compile prefers a
# small set of distinct shapes, so buckets snap to multiples of 64 pixels.
BUCKET_DIVISOR = 64

# Widest aspect ratio a bucket may take; anything more extreme clamps to it
# (matching the common multi-tier bucketing practice of capping panoramas).
MAX_BUCKET_RATIO = 2.0


def compute_bucket(
    width: int,
    height: int,
    base_resolution: int,
    divisor: int = BUCKET_DIVISOR,
    max_ratio: float = MAX_BUCKET_RATIO,
) -> tuple[int, int]:
    """The (bucket_w, bucket_h) for an image: preserve aspect (clamped to
    ``max_ratio``), keep area ~= base_resolution**2, snap both dims to
    ``divisor``. A square input maps exactly to (base, base)."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    ratio = width / height
    ratio = max(1.0 / max_ratio, min(max_ratio, ratio))
    area = float(base_resolution) * float(base_resolution)
    bw = math.sqrt(area * ratio)
    bh = bw / ratio
    snap = lambda v: max(divisor, int(round(v / divisor)) * divisor)  # noqa: E731
    return snap(bw), snap(bh)


def assign_buckets(
    sizes: list[tuple[int, int]],
    base_resolution: int,
    divisor: int = BUCKET_DIVISOR,
    max_ratio: float = MAX_BUCKET_RATIO,
) -> dict[tuple[int, int], list[int]]:
    """Group dataset indices by their bucket shape."""
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (w, h) in enumerate(sizes):
        buckets.setdefault(compute_bucket(w, h, base_resolution, divisor, max_ratio), []).append(i)
    return buckets


class BucketBatchSampler:
    """Batch indices so every batch comes from ONE bucket (one latent shape).

    Within each bucket the draw is a reshuffled permutation (each cycle visits
    every image once, like the trainers' PermutationBatchSampler); the bucket
    for each batch is drawn weighted by bucket size so coverage stays uniform
    across the dataset. Seed-deterministic via the caller's ``rng``. A batch
    never mixes buckets; a bucket smaller than the batch size wraps within
    itself so the batch shape stays fixed."""

    def __init__(self, buckets: dict[tuple[int, int], list[int]], rng: random.Random):
        if not buckets or not any(buckets.values()):
            raise ValueError("BucketBatchSampler needs at least one bucketed item")
        self._rng = rng
        self._shapes = sorted(buckets.keys())
        self._items = {s: list(buckets[s]) for s in self._shapes}
        self._weights = [len(self._items[s]) for s in self._shapes]
        self._order: dict[tuple[int, int], list[int]] = {s: [] for s in self._shapes}
        self._pos = {s: 0 for s in self._shapes}

    def next_batch(self, k: int) -> tuple[tuple[int, int], list[int]]:
        """Returns (bucket_shape, indices) with exactly ``k`` indices."""
        shape = self._rng.choices(self._shapes, weights = self._weights, k = 1)[0]
        out: list[int] = []
        while len(out) < k:
            order, pos = self._order[shape], self._pos[shape]
            if pos >= len(order):
                order = list(self._items[shape])
                self._rng.shuffle(order)
                self._order[shape], pos = order, 0
            take = min(k - len(out), len(order) - pos)
            out.extend(order[pos : pos + take])
            self._pos[shape] = pos + take
        return shape, out
