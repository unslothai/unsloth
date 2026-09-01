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


# Effective decay is min(decay, (1 + updates) / (WARMUP_OFFSET + updates)); at offset 10 step 1
# averages aggressively (~0.18) and the ramp reaches 0.99 after ~1000 updates.
# ── LoRA EMA ──────────────────────────────────────────────────────────────────
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
        """Re-point every shadow at the model's CURRENT trainable weights.

        For a resume that turns EMA on for the first time. The trainer builds the EMA before it
        restores the adapter, so the shadow holds freshly initialised LoRA weights, and a
        checkpoint written with EMA off carries no shadow to replace them with -- leaving the
        exported EMA adapter blending the restored weights with initialisation noise. Starting
        from the restored weights is what enabling EMA at step N means."""
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
        """Live shadow names a saved EMA state does not cover, by name or by shape.

        ``load_state_dict`` skips those entries by design (a differently-wrapped model should
        degrade rather than raise), which is exactly why a caller restoring a run has to ask:
        a partial EMA silently blends restored shadows for some parameters with freshly
        initialised ones for the rest, and every later update and the exported EMA adapter
        carry that mixture while the run reports a clean resume."""
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
        """Restore shadows saved by ``state_dict`` (a resume checkpoint), in place.

        ``updates`` restores the warmup ramp position: without it a resumed run would
        restart the ramp and pull the shadow hard towards the current weights. Entries the
        live model does not have are ignored, so a checkpoint from a differently-wrapped
        model degrades to "keep the freshly initialised shadow" instead of raising."""
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


def _hub_cache_roots() -> list[str]:
    """Hub cache roots to look a repo up in, ACTIVE one first.

    Unsloth can move its cache during a session (Settings), and loading follows the live setting,
    but ``huggingface_hub.constants.HF_HUB_CACHE`` is a snapshot of the environment at import
    time. Reading only that constant left the revision "unresolved" (or pinned to a snapshot in
    the previous root) once the cache moved, so pulling a new revision of the same checkpoint no
    longer changed this key and a warm run silently reused the old embeddings and latents. The
    constant stays as a fallback, since the trainer subprocess may run without Unsloth's settings
    module importable."""
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


# An in-place edit of any of these must change the fingerprint: text_encoder*/tokenizer* produce
# the embeddings and vae* the latents. "connectors" is LTX-2's, the only family running a module
# between encode_prompt and the DiT: its connector output, not the raw Gemma3 state, is cached.
_CACHE_SOURCE_SUBDIRS = ("text_encoder", "tokenizer", "vae", "connectors")


def source_revision(ref: Any) -> str:
    """Revision/content marker for a checkpoint reference, resolved without loading it.

    A repo id or directory path is not a version: a Hub repo that advances to a new
    revision, or a directory edited in place, keeps the same string while its encoders
    or VAE change, so cached conditioning from the old ones would be reused. Shared by the
    trainer's cache namespace and the inference-side wrapper; deliberately cheap and
    never touches the encoders, since the point of the cache is that a warm run does
    not load them.
    """
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
            f"lat_v{_CACHE_VERSION}_{self.family}_{geom}_" f"{_file_content_hash(image_path)}_{var}"
        )

    def text_key(self, caption: str) -> str:
        return f"txt_v{_CACHE_VERSION}_{self.family}_{_text_content_hash(caption)}"

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.safetensors"

    def has(self, key: str) -> bool:
        return self.path_for(key).is_file()

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


# Buckets snap to 64 pixels: the DiT families divide by 8 in the VAE and 2 again in latent patching,
# and regional torch.compile prefers few distinct shapes.
# ── aspect-ratio bucketing ────────────────────────────────────────────────────
BUCKET_DIVISOR = 64

# Widest aspect ratio a bucket may take; anything more extreme clamps to it (matching the common
# practice of capping panoramas).
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
