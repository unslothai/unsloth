# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resumable checkpoints for diffusion (image / future video) LoRA training.

A diffusion run used to write exactly one thing, the deployable adapter, at the very end, so a stop
at step 11 left the AdamW moments, the LR-schedule position, the RNG streams and the step counter to
die with the subprocess: restarting the same configuration began again at step 1. This module is the
single writer and the single reader of the state that makes a run continuable.

Layout, under the run's own ``output_dir`` (so the resumed run keeps writing into the same folder
the adapter is published from): ``<output_dir>/pytorch_lora_weights.safetensors`` is the deployable
adapter (unchanged), and each ``checkpoint-<N>/`` holds ``trainer_state.json`` (manifest +
completion marker), ``adapter_model.safetensors`` (peft-format LoRA tensors),
``ema_adapter.safetensors`` (the EMA shadow, when ema_decay > 0), ``optimizer.pt``, ``scheduler.pt``
and ``rng_state.pt`` (torch CPU + per-device CUDA RNG states).

``trainer_state.json`` is written LAST inside a hidden staging directory, and the whole staging
directory is then promoted with a single ``os.replace``. A process killed at any point leaves either
the previous checkpoint or a ``.tmp-checkpoint-*`` directory that no scanner ever matches, never a
half-written ``checkpoint-<N>`` that looks valid.

Identity: every bundle records the training identity it belongs to (family, base repo + revision,
dataset fingerprint, LoRA targets/rank/alpha, precision). ``preflight_resume`` compares that against
the incoming request and refuses a mismatch with a user-facing reason, so the start route can reject
BEFORE it evicts the resident GPU model.

``kind`` ("image" today) is carried in both the manifest and the identity so a future video trainer
can write and validate its own bundles through this same code without a format change.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import random
import re
import shutil
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Iterable

# Bumped only for a breaking layout change; readers refuse unknown versions.
CHECKPOINT_FORMAT = "unsloth-diffusion-checkpoint"
CHECKPOINT_VERSION = 1

CHECKPOINT_PREFIX = "checkpoint-"
# Hidden and not prefixed "checkpoint-", so a half-written bundle is never mistaken for one.
_STAGING_PREFIX = ".tmp-checkpoint-"

TRAINER_STATE_FILENAME = "trainer_state.json"
ADAPTER_FILENAME = "adapter_model.safetensors"
EMA_FILENAME = "ema_adapter.safetensors"
OPTIMIZER_FILENAME = "optimizer.pt"
SCHEDULER_FILENAME = "scheduler.pt"
RNG_FILENAME = "rng_state.pt"

DEFAULT_SAVE_TOTAL_LIMIT = 2


class ResumeError(ValueError):
    """A resume request that cannot be honoured. The message is shown to the user."""


_IDENTITY_LABELS: tuple[tuple[str, str], ...] = (
    ("kind", "training type"),
    ("family", "model family"),
    ("base_model", "base model"),
    ("base_revision", "base model revision"),
    ("dataset_fingerprint", "training images"),
    ("lora_target_modules", "LoRA target modules"),
    ("lora_rank", "LoRA rank"),
    ("lora_alpha", "LoRA alpha"),
    ("lora_dropout", "LoRA dropout"),
    ("cfg_dropout", "caption dropout"),
    # Read from the INCOMING config while moments and scheduler come from the bundle.
    ("flow_shift", "timestep shift"),
    ("weighting_scheme", "loss weighting scheme"),
    ("snr_gamma", "min-SNR gamma"),
    ("lr_scheduler", "learning-rate schedule"),
    ("lr_warmup_steps", "learning-rate warmup"),
    # Latent cache and crop/flip plan are built from these BEFORE the RNG is restored.
    ("seed", "random seed"),
    ("cache_latents", "latent caching"),
    ("cache_mode", "latent cache path"),
    # The mode the LOOP actually ran in: cached and in-loop paths draw crops from different streams.
    ("cache_variants", "cached crop variants"),
    ("center_crop", "centre cropping"),
    ("random_flip", "random flipping"),
    # Buckets change canvases and batch order; a pre-field manifest trained square (_LEGACY_IDENTITY_DEFAULTS).
    ("bucketing", "aspect ratio buckets"),
    # TF32 versus strict fp32 continues the trajectory at a different numeric precision, and off is the documented
    # strict-reproducibility mode.
    ("enable_tf32", "TF32 matmuls"),
    ("train_batch_size", "batch size"),
    ("gradient_accumulation_steps", "gradient accumulation"),
    ("max_grad_norm", "gradient clipping"),
    # Trainers crop to this before the restored sampler sees anything.
    ("resolution", "training resolution"),
    ("precision", "mixed precision"),
    ("base_precision", "base precision"),
    # What the base was ACTUALLY converted to: fp8/mxfp8 can fall back to bf16.
    ("base_precision_effective", "resolved base precision"),
    # LoRAEMA uses the INCOMING decay with restored shadows, so a change makes a hybrid EMA.
    ("ema_decay", "EMA decay"),
)
# Unknown on either side means "cannot tell", not a mismatch.
_OPTIONAL_IDENTITY_FIELDS = frozenset(
    {
        "base_revision",
        "dataset_fingerprint",
        "lora_dropout",
        "cfg_dropout",
        "ema_decay",
        "flow_shift",
        "weighting_scheme",
        "snr_gamma",
        "lr_scheduler",
        "lr_warmup_steps",
        "seed",
        "cache_latents",
        "cache_mode",
        "cache_variants",
        "center_crop",
        "random_flip",
        "bucketing",
        "enable_tf32",
        "train_batch_size",
        "gradient_accumulation_steps",
        "max_grad_norm",
        "base_precision_effective",
    }
)
# Bundle side only: an incoming None is the route's unresolved pre-trainer identity ("cannot tell").
_LEGACY_IDENTITY_DEFAULTS: dict[str, str] = {"bucketing": "off"}
# What source_revision() returns when it cannot resolve a revision offline.
_UNRESOLVED_REVISION = "unresolved"


def _revision_is_comparable(value: Any) -> bool:
    """Only a Hub revision (rev-<sha>) is comparable; dir-<hash> from local sizes and mtimes is brittle."""
    return isinstance(value, str) and value.startswith("rev-")


def _revision_repo(identity: "CheckpointIdentity") -> str:
    """Bundles from before mirrors existed lack base_revision_repo, so this falls back to base_model."""
    return str(getattr(identity, "base_revision_repo", None) or identity.base_model or "").lower()


@dataclass(frozen = True)
class CheckpointIdentity:
    """What a checkpoint was trained as. Two bundles are interchangeable only when every field here
    agrees, so resuming can never continue a FLUX run into an SDXL adapter, or feed rank-16
    moments to a rank-32 optimizer. ``base_revision`` and ``dataset_fingerprint`` are optional:
    the first is resolved from the local Hub cache and reads ``unresolved`` when the repo has not
    been fetched yet, the second is only known once the dataset has been walked. Either being
    unknown skips that comparison rather than failing it."""

    family: str
    base_model: str
    lora_target_modules: tuple[str, ...]
    lora_rank: int
    lora_alpha: int
    precision: str
    base_precision: str
    resolution: int
    kind: str = "image"
    base_revision: Optional[str] = None
    dataset_fingerprint: Optional[str] = None
    lora_dropout: Optional[float] = None
    # The DiT loop draws rng.random() per sample while this is > 0, so a change diverges the RNG.
    cfg_dropout: Optional[float] = None
    # Optional so a bundle from before these were recorded reads unknown, not mismatched.
    flow_shift: Optional[str] = None
    weighting_scheme: Optional[str] = None
    # Text, not float: None DISABLES min-SNR and must stay distinct from "not recorded".
    snr_gamma: Optional[str] = None
    lr_scheduler: Optional[str] = None
    lr_warmup_steps: Optional[int] = None
    seed: Optional[int] = None
    # Booleans as text ("on"/"off"): None is reserved for a manifest that predates the field.
    cache_latents: Optional[str] = None
    # Resolved, not requested; left None by the start route so the pre-eviction preflight skips it.
    cache_mode: Optional[str] = None
    cache_variants: Optional[int] = None
    center_crop: Optional[str] = None
    random_flip: Optional[str] = None
    bucketing: Optional[str] = None
    enable_tf32: Optional[str] = None
    train_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    # Text, so 0.0 (clipping disabled) is a value and None stays "not recorded".
    max_grad_norm: Optional[str] = None
    # Text: 0.0 means EMA off; None is reserved for a manifest that predates the field.
    ema_decay: Optional[str] = None
    base_precision_effective: Optional[str] = None
    # Gated bases fetch from an ungated mirror with different SHAs, so record which repo.
    base_revision_repo: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "family": self.family,
            "base_model": self.base_model,
            "base_revision": self.base_revision,
            "base_revision_repo": self.base_revision_repo,
            "dataset_fingerprint": self.dataset_fingerprint,
            "lora_target_modules": list(self.lora_target_modules),
            "lora_rank": int(self.lora_rank),
            "lora_alpha": int(self.lora_alpha),
            "lora_dropout": self.lora_dropout,
            "cfg_dropout": self.cfg_dropout,
            "flow_shift": self.flow_shift,
            "weighting_scheme": self.weighting_scheme,
            "snr_gamma": self.snr_gamma,
            "lr_scheduler": self.lr_scheduler,
            "lr_warmup_steps": self.lr_warmup_steps,
            "seed": self.seed,
            "cache_latents": self.cache_latents,
            "cache_mode": self.cache_mode,
            "cache_variants": self.cache_variants,
            "center_crop": self.center_crop,
            "random_flip": self.random_flip,
            "bucketing": self.bucketing,
            "enable_tf32": self.enable_tf32,
            "train_batch_size": self.train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "ema_decay": self.ema_decay,
            "base_precision_effective": self.base_precision_effective,
            "precision": self.precision,
            "base_precision": self.base_precision,
            "resolution": int(self.resolution),
        }

    @classmethod
    def from_dict(cls, data: Any) -> Optional["CheckpointIdentity"]:
        """Rebuild an identity from a manifest. Returns None for anything unreadable, so a
        hand-edited or truncated record reads as "no identity" instead of raising."""
        if not isinstance(data, dict):
            return None
        try:
            targets = data.get("lora_target_modules") or []
            if not isinstance(targets, (list, tuple)):
                return None
            return cls(
                family = str(data.get("family") or ""),
                base_model = str(data.get("base_model") or ""),
                lora_target_modules = tuple(str(t) for t in targets),
                lora_rank = int(data.get("lora_rank") or 0),
                lora_alpha = int(data.get("lora_alpha") or 0),
                lora_dropout = _optional_float(data.get("lora_dropout")),
                cfg_dropout = _optional_float(data.get("cfg_dropout")),
                flow_shift = _optional_str(data.get("flow_shift")),
                weighting_scheme = _optional_str(data.get("weighting_scheme")),
                snr_gamma = _optional_str(data.get("snr_gamma")),
                lr_scheduler = _optional_str(data.get("lr_scheduler")),
                lr_warmup_steps = _optional_int(data.get("lr_warmup_steps")),
                seed = _optional_int(data.get("seed")),
                cache_latents = _optional_str(data.get("cache_latents")),
                cache_mode = _optional_str(data.get("cache_mode")),
                cache_variants = _optional_int(data.get("cache_variants")),
                center_crop = _optional_str(data.get("center_crop")),
                random_flip = _optional_str(data.get("random_flip")),
                bucketing = _optional_str(data.get("bucketing")),
                enable_tf32 = _optional_str(data.get("enable_tf32")),
                train_batch_size = _optional_int(data.get("train_batch_size")),
                gradient_accumulation_steps = _optional_int(data.get("gradient_accumulation_steps")),
                max_grad_norm = _optional_str(data.get("max_grad_norm")),
                ema_decay = _optional_str(data.get("ema_decay")),
                base_precision_effective = _optional_str(data.get("base_precision_effective")),
                precision = str(data.get("precision") or ""),
                base_precision = str(data.get("base_precision") or ""),
                resolution = int(data.get("resolution") or 0),
                kind = str(data.get("kind") or "image"),
                base_revision = _optional_str(data.get("base_revision")),
                base_revision_repo = _optional_str(data.get("base_revision_repo")),
                dataset_fingerprint = _optional_str(data.get("dataset_fingerprint")),
            )
        except (TypeError, ValueError):
            return None

    def with_dataset(self, fingerprint: Optional[str]) -> "CheckpointIdentity":
        """A copy carrying the dataset fingerprint, filled in once the images are known."""
        return replace(self, dataset_fingerprint = fingerprint)

    def mismatch_reason(self, other: "CheckpointIdentity") -> Optional[str]:
        """Why ``other`` (the incoming request) cannot continue ``self`` (the checkpoint), or None
        when they are compatible. Reports the FIRST difference so the message names one concrete
        thing to change."""
        mine, theirs = self.as_dict(), other.as_dict()
        for field, label in _IDENTITY_LABELS:
            a, b = mine.get(field), theirs.get(field)
            if a in (None, "") and field in _LEGACY_IDENTITY_DEFAULTS:
                a = _LEGACY_IDENTITY_DEFAULTS[field]
            # None / "" is "cannot tell", NOT falsiness: a lora_dropout of 0.0 is a real value and the commonest one,
            # so truthiness would skip exactly the 0.0-against-0.15 comparison that matters.
            if field in _OPTIONAL_IDENTITY_FIELDS and (a in (None, "") or b in (None, "")):
                continue
            if field == "base_revision" and not (
                _revision_is_comparable(a) and _revision_is_comparable(b)
            ):
                continue
            # SHAs are comparable only within one repo: an ungated mirror has different SHAs.
            if field == "base_revision" and _revision_repo(self) != _revision_repo(other):
                continue
            if a == b:
                continue
            if field == "dataset_fingerprint":
                return (
                    "The training images have changed since this checkpoint was written, so "
                    "the run cannot continue from it. Restore the original dataset, or start "
                    "a new run."
                )
            return (
                f"This checkpoint was trained with a different {label} "
                f"({_render(a)} vs {_render(b)}), so it cannot be resumed into this run."
            )
        return None


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> Optional[float]:
    """A rounded float, or None for anything unreadable (including a manifest that predates the
    field). None is the "cannot tell" the optional-field rule skips."""
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


# Families whose scheduler bakes a static shift into ``sigmas`` (no dynamic shifting, shift != 1): Z-Image's 3 / 6.
_STATIC_SHIFT_FAMILIES = frozenset({"z-image"})


def _flow_shift_key(value: Any, family: str = "") -> str:
    """None keeps the "1.0" earlier builds wrote. Numbers changed meaning only on static-shift
    families, so only those get a new token (older checkpoints are refused there)."""
    if value is None:
        return "1.0"
    if isinstance(value, str):
        return value
    if family in _STATIC_SHIFT_FAMILIES:
        return f"effective:{float(value)}"
    return str(float(value))


def _snr_gamma_key(value: Any) -> str:
    """min-SNR as a comparable token. None DISABLES the weighting, so it gets its own value rather
    than the unknown the optional-field rule skips."""
    number = _optional_float(value)
    return "off" if number is None else f"{number}"


def _flag_key(value: Any) -> Optional[str]:
    """A boolean as a comparable token, so False is a value and None stays "not recorded"."""
    if value is None:
        return None
    return "on" if bool(value) else "off"


def _optional_int(value: Any) -> Optional[int]:
    """An int, or None for anything unreadable. Same "cannot tell" as _optional_float."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _render(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) or "none"
    return str(value) if value not in (None, "") else "unset"


def dataset_fingerprint(pairs: Any) -> str:
    """Built from file name, byte size and caption, so moving the dataset folder does not change it."""
    parts: list[str] = []
    for entry in pairs or ():
        try:
            path, caption = entry[0], entry[1]
        except (IndexError, KeyError, TypeError):
            continue
        parts.append(f"{Path(str(path)).name}:{_content_probe(path)}:{caption}")
    digest = hashlib.sha256("|".join(sorted(parts)).encode("utf-8", "replace")).hexdigest()
    return f"ds-{len(parts)}-{digest[:24]}"


# Enough to tell images apart without reading a huge dataset before the GPU model is evicted.
_PROBE_BYTES = 65536


def _content_probe(path: Any) -> str:
    """Hashes the head and tail, not the whole file: size alone missed same-length in-place overwrites."""
    try:
        size = os.path.getsize(path)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            digest.update(handle.read(_PROBE_BYTES))
            if size > _PROBE_BYTES:
                # Covers every byte past the head, so a same-length replacement sharing a head is detected.
                handle.seek(max(_PROBE_BYTES, size - _PROBE_BYTES))
                digest.update(handle.read(_PROBE_BYTES))
    # ValueError: open() rejects an embedded NUL rather than raising OSError.
    except (OSError, ValueError):
        return "?"
    return f"{size}-{digest.hexdigest()[:16]}"


def _resolve_lora_targets(cfg: Any) -> tuple[str, ...]:
    """Must resolve the same targets the trainer attaches, or every resume fingerprints as a mismatch."""
    configured = tuple(cfg.lora_target_modules)
    if str(getattr(cfg, "resolved_family", "") or "").strip().lower() == "sdxl":
        return configured
    # Not wrapped: a generic fallback would make the route fingerprint differ from the trainer's.
    from core.training.diffusion_dit_trainer import _SPECS, _select_lora_targets

    spec = _SPECS.get(cfg.resolved_family)
    if spec is None:
        return configured
    return tuple(_select_lora_targets(configured, spec.lora_targets))


def with_cache_mode(identity: "CheckpointIdentity", used_cache: bool) -> "CheckpointIdentity":
    """Records the cache path actually taken, since cached and in-loop runs draw different RNG streams."""
    return replace(identity, cache_mode = "cached" if used_cache else "in-loop")


def with_resolved_base_precision(
    identity: "CheckpointIdentity", resolved: Any
) -> "CheckpointIdentity":
    """Records the precision actually used: fp8 or mxfp8 can fall back to bf16 with just a warning."""
    value = str(resolved or "").strip().lower()
    if not value:
        return identity
    return replace(identity, base_precision_effective = value)


def with_resolved_revision(identity: "CheckpointIdentity", base_model: Any) -> "CheckpointIdentity":
    """Re-read after the load, always: from_pretrained can refresh refs/main past the pre-load commit."""
    from core.training.diffusion_train_extras import source_revision

    resolved = source_revision(base_model)
    if not _revision_is_comparable(resolved) or resolved == identity.base_revision:
        return identity
    return replace(identity, base_revision = resolved, base_revision_repo = str(base_model or ""))


def identity_for_config(
    cfg: Any,
    *,
    dataset_pairs: Any = None,
    resolved_targets: Optional[tuple[str, ...]] = None,
    kind: str = "image",
) -> CheckpointIdentity:
    """base_precision is the requested mode, not what auto resolves to, since no model is loaded yet."""
    from core.training.diffusion_train_common import effective_mixed_precision
    from core.training.diffusion_train_extras import source_revision

    targets = tuple(resolved_targets) if resolved_targets else _resolve_lora_targets(cfg)
    # base_model stays canonical; only the revision pair follows the repo the weights come from.
    fetch_base_model = str(getattr(cfg, "fetch_base_model", None) or cfg.base_model or "")
    return CheckpointIdentity(
        family = str(getattr(cfg, "resolved_family", "") or ""),
        base_model = str(cfg.base_model or ""),
        lora_target_modules = targets,
        lora_rank = int(cfg.lora_rank),
        lora_alpha = int(cfg.lora_alpha if cfg.lora_alpha is not None else cfg.lora_rank),
        lora_dropout = round(float(getattr(cfg, "lora_dropout", 0.0) or 0.0), 6),
        cfg_dropout = round(float(getattr(cfg, "cfg_dropout", 0.0) or 0.0), 6),
        flow_shift = _flow_shift_key(
            getattr(cfg, "flow_shift", None), str(getattr(cfg, "resolved_family", "") or "")
        ),
        weighting_scheme = str(getattr(cfg, "weighting_scheme", "") or "none"),
        # Only the SDXL loop applies min-SNR, so a flow-matching run records "cannot tell" rather than the unused default.
        snr_gamma = (
            _snr_gamma_key(getattr(cfg, "snr_gamma", None))
            if str(getattr(cfg, "resolved_family", "") or "sdxl") == "sdxl"
            else None
        ),
        lr_scheduler = str(getattr(cfg, "lr_scheduler", "") or "constant"),
        lr_warmup_steps = int(getattr(cfg, "lr_warmup_steps", 0) or 0),
        seed = int(getattr(cfg, "seed", 0) or 0),
        cache_latents = _flag_key(getattr(cfg, "cache_latents", None)),
        cache_variants = int(getattr(cfg, "cache_variants", 0) or 0),
        center_crop = _flag_key(getattr(cfg, "center_crop", None)),
        random_flip = _flag_key(getattr(cfg, "random_flip", None)),
        bucketing = _flag_key(getattr(cfg, "bucketing", None)),
        enable_tf32 = _flag_key(getattr(cfg, "enable_tf32", None)),
        train_batch_size = int(getattr(cfg, "train_batch_size", 0) or 0),
        gradient_accumulation_steps = int(getattr(cfg, "gradient_accumulation_steps", 0) or 0),
        max_grad_norm = f"{round(float(getattr(cfg, 'max_grad_norm', 0.0) or 0.0), 6)}",
        ema_decay = f"{round(float(getattr(cfg, 'ema_decay', 0.0) or 0.0), 6)}",
        # EFFECTIVE precision: pre-Ampere resolves bf16 to fp16.
        precision = effective_mixed_precision(cfg),
        base_precision = str(getattr(cfg, "base_precision", "") or ""),
        resolution = int(cfg.resolution),
        kind = kind,
        # Revision of the repo actually FETCHED: the canonical one may not be cached.
        base_revision = source_revision(fetch_base_model),
        base_revision_repo = fetch_base_model,
        dataset_fingerprint = dataset_fingerprint(dataset_pairs) if dataset_pairs else None,
    )


def capture_rng_state(streams: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Trainer-owned random.Random streams are not in module random state, so they are passed in."""
    payload: dict[str, Any] = {
        "python": _random_state_to_json(random.getstate()),
        "streams": {},
        "numpy": None,
        # One backend's device generator says nothing on another; the resume preflight reads this.
        "accelerator": _rng_accelerator(),
    }
    for name, stream in (streams or {}).items():
        try:
            payload["streams"][str(name)] = _random_state_to_json(stream.getstate())
        except Exception:  # noqa: BLE001 -- a stream we cannot read simply is not restored
            continue
    try:
        import numpy as np
        kind, keys, pos, has_gauss, cached = np.random.get_state()
        payload["numpy"] = {
            "bit_generator": str(kind),
            "keys": [int(k) for k in keys],
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        }
    except Exception:  # noqa: BLE001 -- no numpy / a non-MT19937 global generator
        payload["numpy"] = None

    tensors: dict[str, Any] = {}
    try:
        import torch
        tensors["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            try:
                for i, state in enumerate(torch.cuda.get_rng_state_all()):
                    tensors[f"torch_cuda_{i}"] = state
            except Exception:  # noqa: BLE001 -- one device erroring loses the whole capture
                # All or nothing on CUDA: a CPU-only state would resume with a freshly seeded CUDA generator.
                tensors = {}
        elif _xpu_available():
            # randn_like draws on the training device, so XPU noise lives in the XPU generator.
            try:
                for i, state in enumerate(torch.xpu.get_rng_state_all()):
                    tensors[f"torch_xpu_{i}"] = state
            except Exception:  # noqa: BLE001 -- one device erroring loses the whole capture
                tensors = {}
    except Exception:  # noqa: BLE001 -- torch RNG capture is best-effort
        tensors = {}
    return {"json": payload, "tensors": tensors}


def restore_rng_state(
    payload: Optional[dict[str, Any]],
    tensors: Optional[dict[str, Any]] = None,
    streams: Optional[dict[str, Any]] = None,
) -> None:
    """Undo ``capture_rng_state``. Every part is independent and best-effort: a checkpoint written
    on a 2-GPU box restored on a 1-GPU box still restores everything else."""
    payload = payload or {}
    state = _random_state_from_json(payload.get("python"))
    if state is not None:
        try:
            random.setstate(state)
        except (TypeError, ValueError):
            pass
    saved_streams = payload.get("streams")
    if isinstance(saved_streams, dict):
        for name, stream in (streams or {}).items():
            got = _random_state_from_json(saved_streams.get(str(name)))
            if got is None:
                continue
            try:
                stream.setstate(got)
            except (TypeError, ValueError):
                continue
    np_state = payload.get("numpy")
    if isinstance(np_state, dict):
        try:
            import numpy as np
            np.random.set_state(
                (
                    str(np_state.get("bit_generator") or "MT19937"),
                    np.array(np_state.get("keys") or [], dtype = np.uint32),
                    int(np_state.get("pos") or 0),
                    int(np_state.get("has_gauss") or 0),
                    float(np_state.get("cached_gaussian") or 0.0),
                )
            )
        except Exception:  # noqa: BLE001 -- no numpy, or an incompatible generator
            pass
    if not tensors:
        return
    try:
        import torch

        cpu = tensors.get("torch_cpu")
        if cpu is not None:
            torch.set_rng_state(cpu.cpu().to(torch.uint8))
        if torch.cuda.is_available():
            # Per device, not set_rng_state_all: that needs one state per visible device and restores
            # nothing when fewer devices were visible at save time.
            for i in range(torch.cuda.device_count()):
                state = tensors.get(f"torch_cuda_{i}")
                if state is None:
                    continue
                torch.cuda.set_rng_state(state.cpu().to(torch.uint8), i)
        elif _xpu_available():
            for i in range(torch.xpu.device_count()):
                state = tensors.get(f"torch_xpu_{i}")
                if state is None:
                    continue
                torch.xpu.set_rng_state(state.cpu().to(torch.uint8), i)
    except Exception:  # noqa: BLE001 -- best-effort restore, never fatal
        pass


def _rng_accelerator() -> str:
    """The backend whose device generator a capture holds: ``cuda``, ``xpu`` or ``cpu``."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 -- probe failure -> no CUDA generator captured
        pass
    return "xpu" if _xpu_available() else "cpu"


def _xpu_available() -> bool:
    """Guarded like ``resolve_train_device``'s probe: an uninitialised driver must not turn a
    best-effort RNG capture into a raise."""
    try:
        import torch
        fn = getattr(getattr(torch, "xpu", None), "is_available", None)
        return bool(fn()) if callable(fn) else False
    except Exception:  # noqa: BLE001 -- probe failure -> no XPU generator to capture
        return False


def _random_state_to_json(state: Any) -> Optional[list[Any]]:
    """``random.Random.getstate()`` is ``(version, tuple[int, ...], gauss)``; JSON has no tuples, so
    store it as nested lists."""
    try:
        version, keys, gauss = state
        return [int(version), [int(k) for k in keys], gauss]
    except (TypeError, ValueError):
        return None


def _random_state_from_json(value: Any) -> Optional[tuple]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (int(value[0]), tuple(int(k) for k in value[1]), value[2])
    except (TypeError, ValueError):
        return None


def save_checkpoint(
    *,
    output_dir: str | os.PathLike[str],
    step: int,
    adapter_state: dict[str, Any],
    identity: CheckpointIdentity,
    target_steps: int,
    optimizer: Any = None,
    lr_scheduler: Any = None,
    ema_state: Optional[dict[str, Any]] = None,
    ema_updates: int = 0,
    rng: Optional[dict[str, Any]] = None,
    sampler_state: Optional[dict[str, Any]] = None,
    progress: Optional[dict[str, Any]] = None,
    save_total_limit: int = DEFAULT_SAVE_TOTAL_LIMIT,
    discard_existing: bool = False,
    source_checkpoint: Optional[str | os.PathLike[str]] = None,
    preexisting: "Optional[Iterable[Any]]" = None,
) -> str:
    """Staged in a hidden directory and promoted with one os.replace, so a kill keeps the old bundle."""
    import torch
    from safetensors.torch import save_file

    step = int(step)
    if step < 0:
        raise ValueError("checkpoint step must be >= 0")
    if not adapter_state:
        # An empty safetensors file fails bundle validation later; fail loudly at write time.
        raise ValueError("refusing to write a checkpoint with no adapter tensors")
    root = Path(output_dir).expanduser()
    root.mkdir(parents = True, exist_ok = True)
    doomed: list[Path] = []
    if discard_existing:
        # Deleted only AFTER the new bundle is promoted, so a failed write keeps a checkpoint.
        doomed = list_checkpoints(root)
    else:
        # Reusing an existing bundle at this step is safe only when it is byte-identical state.
        existing = root / f"{CHECKPOINT_PREFIX}{step}"
        if (
            source_checkpoint is not None
            and Path(source_checkpoint).expanduser().resolve() == existing.resolve()
            and read_checkpoint(existing) is not None
        ):
            return str(existing)
    staging = root / f"{_STAGING_PREFIX}{step}-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents = True, exist_ok = False)

    try:
        _save_tensors(save_file, adapter_state, staging / ADAPTER_FILENAME)
        files = {"adapter": ADAPTER_FILENAME}
        optimizer_class: Optional[str] = None
        optimizer_param_names: Optional[list[str]] = None
        if ema_state:
            _save_tensors(save_file, ema_state, staging / EMA_FILENAME)
            files["ema"] = EMA_FILENAME
        if optimizer is not None:
            # torch.save: optimizer state mixes nested tensors and scalars, which safetensors cannot hold.
            _torch_save(torch, optimizer.state_dict(), staging / OPTIMIZER_FILENAME)
            files["optimizer"] = OPTIMIZER_FILENAME
            optimizer_class = optimizer_key(optimizer)
            # adapter_state follows named_parameters() order, which IS the optimizer's positional order.
            optimizer_param_names = [str(name) for name in adapter_state]
        if lr_scheduler is not None:
            _torch_save(torch, lr_scheduler.state_dict(), staging / SCHEDULER_FILENAME)
            files["scheduler"] = SCHEDULER_FILENAME
        rng_json: Optional[dict[str, Any]] = None
        if rng:
            rng_json = rng.get("json")
            rng_tensors = rng.get("tensors") or {}
            if not rng_tensors:
                # capture_rng_state never raises; fail the WRITE instead of producing an unresumable bundle.
                raise RuntimeError(
                    "the run's random-number state could not be captured, so this checkpoint "
                    "would not be resumable"
                )
            _torch_save(torch, rng_tensors, staging / RNG_FILENAME)
            files["rng"] = RNG_FILENAME

        manifest: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "kind": identity.kind,
            "global_step": step,
            "target_steps": int(target_steps),
            # Recorded so a future mid-accumulation checkpoint is a value change, not a format change.
            "micro_step": 0,
            "created_at": time.time(),
            "identity": identity.as_dict(),
            "sampler": sampler_state or None,
            "rng": rng_json,
            "ema_updates": int(ema_updates),
            # bnb AdamW8bit and torch AdamW load each other's state_dict then KeyError on step one.
            "optimizer_class": optimizer_class,
            # Optimizer state is keyed by POSITION; a traversal-order change silently rebinds moments.
            "optimizer_param_names": optimizer_param_names,
            # Nested so a caller can never shadow a reserved key.
            "progress": dict(progress or {}),
            "files": files,
            # Sizes catch truncation: torch.load then returns uninitialised memory as Adam moments.
            "file_sizes": _file_sizes(staging, files),
        }
        # LAST: the manifest is the completion marker.
        _write_text(staging / TRAINER_STATE_FILENAME, json.dumps(manifest, indent = 2))
        _fsync_dir(staging)
        final = _promote(staging, root, step)
    except BaseException:
        shutil.rmtree(staging, ignore_errors = True)
        raise
    for stale in doomed:
        # _promote already replaced it; removing it would delete the new checkpoint.
        if stale != final:
            shutil.rmtree(stale, ignore_errors = True)
    _prune_staging(root)
    # Pin the source bundle too, else pruning can leave the original run with no resume point.
    prune_checkpoints(
        root,
        keep = save_total_limit,
        protect = final,
        also_protect = _source_bundle_path(root, source_checkpoint),
        preexisting = preexisting,
    )
    return str(final)


def optimizer_key(optimizer: Any) -> str:
    """A stable name for an optimizer implementation, e.g. ``bitsandbytes.optim.adamw.AdamW8bit``.
    Fused vs non-fused torch AdamW share a class and a state layout, so they compare equal."""
    cls = type(optimizer)
    return f"{getattr(cls, '__module__', '?')}.{getattr(cls, '__qualname__', cls.__name__)}"


def _save_tensors(save_file: Any, state: dict[str, Any], path: Path) -> None:
    """safetensors refuses tensors that share storage, so detach/clone every entry onto CPU (a LoRA
    state dict is megabytes, and this runs at most once per save_steps)."""
    payload = {
        str(k): v.detach().to("cpu", copy = True).contiguous()
        for k, v in (state or {}).items()
        if v is not None
    }
    save_file(payload, str(path))
    _fsync_file(path)


def _torch_save(torch: Any, obj: Any, path: Path) -> None:
    torch.save(obj, str(path))
    _fsync_file(path)


def _file_sizes(staging: Path, files: dict[str, str]) -> dict[str, int]:
    """``{role: bytes}`` for everything already written into the staging dir."""
    sizes: dict[str, int] = {}
    for role, name in files.items():
        try:
            sizes[role] = int((staging / name).stat().st_size)
        except OSError:
            continue
    return sizes


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding = "utf-8")
    _fsync_file(path)


# errno meaning "cannot fsync this handle here". EBADF is a deliberate tradeoff: Windows
# also maps real FlushFileBuffers failures to it (see _fsync_file).
_FSYNC_UNSUPPORTED = frozenset(
    code
    for code in (
        getattr(errno, name, None)
        for name in ("EACCES", "EPERM", "EINVAL", "ENOTSUP", "EOPNOTSUPP", "EBADF", "ENOSYS")
    )
    if code is not None
)


def _fsync_file(path: Path) -> None:
    """Raise real fsync failures (ENOSPC surfaces at fsync), except on Windows, where EBADF is ambiguous."""
    try:
        # O_RDWR: Windows' _commit maps to FlushFileBuffers, which needs write access.
        fd = os.open(str(path), os.O_RDWR)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as error:
        if error.errno not in _FSYNC_UNSUPPORTED:
            raise
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):  # Windows cannot open a directory as a file descriptor
        return
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _promote(staging: Path, root: Path, step: int) -> Path:
    """os.replace cannot overwrite a directory, so the occupant is moved aside and kept, not deleted."""
    final = root / f"{CHECKPOINT_PREFIX}{step}"
    displaced: Optional[Path] = None
    if final.exists():
        displaced = root / f"{_STAGING_PREFIX}replaced-{step}-{uuid.uuid4().hex[:8]}"
        os.replace(final, displaced)
        # os.replace does NOT restamp the directory; stamp the swap time.
        with contextlib.suppress(OSError):
            os.utime(displaced, None)
    try:
        os.replace(staging, final)
    except OSError:
        # Restore the moved-aside copy before raising, else the run has no checkpoint at all.
        if displaced is not None:
            with contextlib.suppress(OSError):
                os.replace(displaced, final)
        raise
    _fsync_dir(root)
    return final


# "stale": promotion killed mid-swap; "replaced": displaced by a later write. Step in name.
_STALE_SLOT = re.compile(r"^(?:stale|replaced)-(\d+)-")
_REPLACED_SLOT = re.compile(r"^replaced-(\d+)-")


def _prune_staging(root: Path) -> None:
    """An empty-slot stale orphan is the previous bundle moved aside, so restore it, never delete."""
    try:
        entries = list(root.glob(f"{_STAGING_PREFIX}*"))
    except OSError:
        return

    # Newest first, so a stacked slot gets its immediate predecessor back.
    def _written_at(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    for entry in sorted(entries, key = _written_at, reverse = True):
        suffix = entry.name[len(_STAGING_PREFIX) :]
        if _recover_orphaned_slot(root, entry) or _REPLACED_SLOT.match(suffix):
            continue
        shutil.rmtree(entry, ignore_errors = True)


# Avoids racing _promote's swap window; only replaced- entries wait, writers pass 0.
_LIVE_REPLACEMENT_GRACE_SECONDS = 5.0


def _recover_orphaned_slots(
    root: Path, *, min_age: float = _LIVE_REPLACEMENT_GRACE_SECONDS
) -> None:
    """Never deletes: hands each stale orphan back to its empty slot, newest first, as replacements
    stack."""
    try:
        entries = list(root.glob(f"{_STAGING_PREFIX}*"))
    except OSError:
        return

    def _written_at(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    now = time.time()
    for entry in sorted(entries, key = _written_at, reverse = True):
        in_flight_shape = _REPLACED_SLOT.match(entry.name[len(_STAGING_PREFIX) :]) is not None
        if in_flight_shape and min_age > 0 and (now - _written_at(entry)) < min_age:
            continue  # possibly a promotion in flight; leave it to the writer
        _recover_orphaned_slot(root, entry)


def _recover_orphaned_slot(root: Path, entry: Path) -> bool:
    """True when ``entry`` is a stale bundle that must not be swept up: either it was handed back to
    its empty slot, or the hand-back failed and leaving it on disk beats deleting the only copy
    of the run's last resumable state."""
    match = _STALE_SLOT.match(entry.name[len(_STAGING_PREFIX) :])
    if match is None:
        return False
    slot = root / f"{CHECKPOINT_PREFIX}{int(match.group(1))}"
    if slot.exists():
        return False
    try:
        os.replace(entry, slot)
    except OSError:
        pass
    return True


def _retire_replaced_slots(root: Path, *, restore: bool) -> None:
    """restore=False drops displaced bundles, since the adapter they belong to was just overwritten."""
    try:
        entries = list(root.glob(f"{_STAGING_PREFIX}replaced-*"))
    except OSError:
        return

    def _written_at(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    # NEWEST first per slot: replacements stack, so restore the actual predecessor.
    entries.sort(key = _written_at, reverse = True)
    restored_slots: set[Path] = set()
    for entry in entries:
        match = _REPLACED_SLOT.match(entry.name[len(_STAGING_PREFIX) :])
        slot = root / f"{CHECKPOINT_PREFIX}{int(match.group(1))}" if match else None
        if restore and slot is not None and slot not in restored_slots and not slot.exists():
            try:
                os.replace(entry, slot)
                restored_slots.add(slot)
                continue
            except OSError:
                continue
        shutil.rmtree(entry, ignore_errors = True)


def _source_bundle_path(root: Path, source_checkpoint) -> Optional[Path]:
    """The bundle this run resumed from, when it lives in ``root``. None otherwise."""
    if not source_checkpoint:
        return None
    try:
        source = Path(source_checkpoint).expanduser().resolve()
    except OSError:
        return None
    for candidate in list_checkpoints(root):
        try:
            if candidate.resolve() == source:
                return candidate
        except OSError:
            continue
    return None


def prune_checkpoints(
    output_dir: str | os.PathLike[str],
    keep: int = DEFAULT_SAVE_TOTAL_LIMIT,
    *,
    protect: Optional[Path] = None,
    also_protect: Optional[Path] = None,
    preexisting: "Optional[Iterable[Any]]" = None,
) -> None:
    """Only bundles this run wrote count toward keep; preexisting and protected ones are never pruned."""
    if keep <= 0:
        return
    # Identity, not pathname: a run can overwrite a slot that already existed.
    kept_from_before: set[Path] = set()
    for entry in preexisting or ():
        if isinstance(entry, tuple):
            path, identity = Path(entry[0]), entry[1]
            if identity is not None and _bundle_identity(path) != identity:
                continue
            kept_from_before.add(path)
        else:
            kept_from_before.add(Path(entry))
    survivors = [c for c in list_checkpoints(output_dir) if c not in kept_from_before]
    for pinned in (protect, also_protect):
        if pinned is not None and pinned in survivors:
            survivors = [c for c in survivors if c != pinned]
            keep = max(0, keep - 1)
    for stale in survivors[keep:]:
        shutil.rmtree(stale, ignore_errors = True)


def clear_checkpoints(output_dir: str | os.PathLike[str]) -> None:
    """Remove every ``checkpoint-<N>`` bundle in ``output_dir``. Used when a fresh (non-resumed) run
    takes over an output directory that an earlier run of the same name left checkpoints in."""
    root = Path(output_dir).expanduser()
    for stale in list_checkpoints(root):
        shutil.rmtree(stale, ignore_errors = True)
    _retire_replaced_slots(root, restore = False)


def resumed_into_this_dir(cfg: Any, output_dir: "str | os.PathLike[str]") -> bool:
    """A resumed run shares its source's directory, so clearing the directory would take the source too."""
    source = getattr(cfg, "resume_from_checkpoint", None)
    if not source:
        return False
    try:
        root = Path(output_dir).expanduser().resolve()
        candidate = Path(str(source)).expanduser().resolve()
    except OSError:
        return False
    return candidate == root or candidate.parent == root


def snapshot_checkpoints(output_dir: str | os.PathLike[str]) -> list[tuple[Path, Optional[tuple]]]:
    """Taken before the first write: a path cannot tell a pre-existing bundle from one written over it."""
    return [(path, _bundle_identity(path)) for path in list_checkpoints(output_dir)]


def retire_own_checkpoints(
    output_dir: str | os.PathLike[str],
    preexisting: "Iterable[Any]",
    *,
    resumed_here: bool = True,
) -> None:
    """Drops a finished run's bundles, so a later resume cannot roll back to its last periodic save."""
    if resumed_here:
        clear_own_checkpoints(output_dir, preexisting)
        return
    root = Path(output_dir).expanduser()
    for stale in list_checkpoints(root):
        shutil.rmtree(stale, ignore_errors = True)
    _retire_replaced_slots(root, restore = False)


def discard_preexisting_checkpoints(
    output_dir: str | os.PathLike[str], preexisting: "Iterable[Any]"
) -> None:
    """A fresh retrain that stops with save drops older bundles, which would outrank its lower-step save."""
    root = Path(output_dir).expanduser()
    keep: dict[Path, Optional[tuple]] = {}
    for entry in preexisting:
        if isinstance(entry, tuple):
            path, identity = entry
            keep[Path(path)] = identity
        else:
            keep[Path(entry)] = _bundle_identity(Path(entry))
    for stale in list_checkpoints(root):
        # Identity, not pathname: a bundle this run wrote over an old one is this run's.
        if stale in keep and keep[stale] == _bundle_identity(stale):
            shutil.rmtree(stale, ignore_errors = True)
    _retire_replaced_slots(root, restore = False)


def discard_named_checkpoints(paths: "Iterable[Any]") -> None:
    """Used when a killed child never ran its cleanup; removes exactly the bundles the parent saw
    written."""
    roots: set[Path] = set()
    for value in paths:
        if not value:
            continue
        try:
            path = Path(str(value)).expanduser()
        except (TypeError, ValueError):
            continue
        if checkpoint_step(path) < 0:
            continue  # not a bundle path; never delete something we cannot name
        roots.add(path.parent)
        shutil.rmtree(path, ignore_errors = True)
    for root in roots:
        _retire_replaced_slots(root, restore = True)


def clear_own_checkpoints(output_dir: str | os.PathLike[str], preexisting: "Iterable[Any]") -> None:
    """Removes only bundles this run wrote, identified by the pre-run snapshot, never the resume source."""
    # Keyed by path AND identity: a periodic save can REPLACE a pre-existing bundle.
    keep: dict[Path, Optional[tuple]] = {}
    for entry in preexisting:
        if isinstance(entry, tuple):
            path, identity = entry
            keep[Path(path)] = identity
        else:
            keep[Path(entry)] = _bundle_identity(Path(entry))
    for stale in list_checkpoints(output_dir):
        if stale in keep and keep[stale] == _bundle_identity(stale):
            continue
        shutil.rmtree(stale, ignore_errors = True)
    _retire_replaced_slots(Path(output_dir).expanduser(), restore = True)


def _bundle_identity(path: Path) -> Optional[tuple]:
    """The manifest's start time tells a bundle apart from one later written over the same path."""
    try:
        manifest = json.loads((path / TRAINER_STATE_FILENAME).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    # created_at is the completion marker, so it differs between two writes at the same step.
    return (manifest.get("created_at"), manifest.get("global_step"))


def checkpoint_step(path: Path) -> int:
    """The step encoded in a ``checkpoint-<N>`` directory name, or -1."""
    name = path.name
    if not name.startswith(CHECKPOINT_PREFIX):
        return -1
    try:
        step = int(name[len(CHECKPOINT_PREFIX) :])
    except ValueError:
        return -1
    return step if step >= 0 else -1


def list_checkpoints(output_dir: str | os.PathLike[str]) -> list[Path]:
    """Every ``checkpoint-<N>`` directory under ``output_dir``, newest step first. Does not
    validate; use ``read_checkpoint`` / ``latest_valid_checkpoint`` for that."""
    root = Path(output_dir).expanduser()
    try:
        found = [p for p in root.glob(f"{CHECKPOINT_PREFIX}*") if p.is_dir()]
    except OSError:
        return []
    return sorted((p for p in found if checkpoint_step(p) >= 0), key = checkpoint_step, reverse = True)


def read_checkpoint(path: str | os.PathLike[str]) -> Optional[dict[str, Any]]:
    """Every gate a resume needs: manifest parses, step matches the name, and all listed files parse."""
    directory = Path(path).expanduser()
    if not directory.is_dir():
        return None
    try:
        manifest = json.loads((directory / TRAINER_STATE_FILENAME).read_text(encoding = "utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("format") != CHECKPOINT_FORMAT:
        return None
    version = manifest.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or not 1 <= version <= CHECKPOINT_VERSION
    ):
        return None
    step = manifest.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return None
    named_step = checkpoint_step(directory)
    if named_step >= 0 and named_step != step:
        return None
    files = manifest.get("files")
    if not isinstance(files, dict) or not files.get("adapter"):
        return None
    sizes = manifest.get("file_sizes")
    sizes = sizes if isinstance(sizes, dict) else {}
    for role, name in files.items():
        if not isinstance(name, str) or not name or Path(name).name != name:
            return None
        # Optimizer and scheduler state can validly be tensor-free (SGD, constant LR).
        if not _valid_state_file(directory / name, require_tensor = role in ("adapter", "ema")):
            return None
        expected_size = sizes.get(role)
        if isinstance(expected_size, int) and not isinstance(expected_size, bool):
            try:
                if (directory / name).stat().st_size != expected_size:
                    return None
            except OSError:
                return None
    return manifest


def _valid_state_file(path: Path, require_tensor: bool = True) -> bool:
    """Catches everything: a deflate-corrupt member can raise zlib.error, which the LLM validator misses."""
    from core.training.resume import _valid_state_file as _validate
    try:
        return _validate(path, require_tensor = require_tensor)
    except Exception:  # noqa: BLE001 -- unreadable in any way == not resumable
        return False


def latest_valid_checkpoint(
    output_dir: str | os.PathLike[str],
    not_before: Optional[float] = None,
    not_after: Optional[float] = None,
    usable: "Optional[Callable[[Path, dict], bool]]" = None,
) -> Optional[tuple[Path, dict]]:
    """Bundles outside the run's start and end times are skipped, since runs can share one output dir."""
    # Killed mid-swap leaves the only bundle under the stale name; recover it into empty slots only.
    _recover_orphaned_slots(Path(output_dir).expanduser())
    for candidate in list_checkpoints(output_dir):
        manifest = read_checkpoint(candidate)
        if manifest is None:
            continue
        if not_before is not None:
            try:
                created = float(manifest.get("created_at") or 0.0)
            except (TypeError, ValueError):
                created = 0.0
            if created < float(not_before):
                continue
        if not_after is not None:
            try:
                created = float(manifest.get("created_at") or 0.0)
            except (TypeError, ValueError):
                created = 0.0
            # created_at 0.0 predates the field and must not be fenced out.
            if created and created > float(not_after):
                continue
        if usable is not None and not usable(candidate, manifest):
            continue
        return candidate, manifest
    return None


def iter_valid_checkpoints(output_dir: str | os.PathLike[str]) -> "list[tuple[Path, dict]]":
    """Every complete bundle, newest first, since the preflight's stricter checks may reject the newest."""
    _recover_orphaned_slots(Path(output_dir).expanduser())
    found: list[tuple[Path, dict]] = []
    for candidate in list_checkpoints(output_dir):
        manifest = read_checkpoint(candidate)
        if manifest is not None:
            found.append((candidate, manifest))
    return found


_UNRESUMABLE_STATUS = {
    "completed": "This run finished its full step count, so there is nothing left to train.",
    "running": "This run is still training.",
}


def _fully_loadable(path: Path, manifest: dict[str, Any]) -> bool:
    """Header scan is not enough; the required state and a real torch.load must also succeed."""
    try:
        _assert_required_state(path, manifest)
        _assert_optimizer_buildable(path, manifest)
        _assert_loadable(path, manifest)
    except ResumeError:
        return False
    return True


def _source_checkpoint_bundle(
    source_checkpoint, source_created_at: Optional[float] = None
) -> Optional[tuple[Path, dict[str, Any]]]:
    """A pathname is not identity: a same-named bundle from another run is refused by its timestamp."""
    if not source_checkpoint:
        return None
    try:
        candidate = Path(str(source_checkpoint)).expanduser()
        manifest = read_checkpoint(candidate) if candidate.is_dir() else None
    except OSError:
        return None
    if manifest is None:
        return None
    if source_created_at:
        try:
            written = float(manifest.get("created_at") or 0.0)
        except (TypeError, ValueError):
            written = 0.0
        if not written or abs(written - float(source_created_at)) > 1e-6:
            return None
    # read_checkpoint is only a header scan, so check full loadability before advertising.
    if not _fully_loadable(candidate, manifest):
        return None
    return (candidate, manifest)


def describe_resume_state(
    output_dir: Optional[str],
    *,
    status: Optional[str] = None,
    started_at: Optional[float] = None,
    ended_at: Optional[float] = None,
    source_checkpoint: Optional[str] = None,
    source_created_at: Optional[float] = None,
    total_steps: Optional[int] = None,
) -> dict[str, Any]:
    """Mirrors the LLM can_resume_run rules; an unreadable directory simply reports nothing to resume."""
    blank: dict[str, Any] = {
        "can_resume": False,
        "checkpoint_step": None,
        "checkpoint_path": None,
        "resume_blocked_reason": None,
    }
    if not output_dir:
        return blank
    blocked = _UNRESUMABLE_STATUS.get(str(status or "").strip().lower())
    if blocked:
        return {**blank, "resume_blocked_reason": blocked}
    try:
        root = Path(str(output_dir)).expanduser()
        if not root.is_dir():
            recovered = _source_checkpoint_bundle(source_checkpoint, source_created_at)
            if recovered is None:
                return {
                    **blank,
                    "resume_blocked_reason": "This run's output folder no longer exists.",
                }
            found = recovered
        else:
            found = latest_valid_checkpoint(
                root,
                not_before = started_at,
                not_after = ended_at,
                # The path is sent back as explicit, so it must actually load.
                usable = _fully_loadable,
            )
        if found is None and source_checkpoint:
            # Read the source bundle directly rather than widening the started_at fence.
            found = _source_checkpoint_bundle(source_checkpoint, source_created_at)
    except OSError:
        return blank
    if found is None:
        if not list_checkpoints(root):
            reason = "This run saved no resume checkpoint, so it cannot be continued."
        elif started_at is not None and latest_valid_checkpoint(root) is not None:
            reason = (
                "This run saved no resume checkpoint of its own; the checkpoints in its folder "
                "were left by an earlier run of the same adapter."
            )
        else:
            reason = "This run's checkpoints are incomplete or corrupt, so it cannot be resumed."
        return {**blank, "resume_blocked_reason": reason}
    path, manifest = found
    step = int(manifest.get("global_step") or 0)
    target = int(total_steps or manifest.get("target_steps") or 0)
    if target and step >= target:
        return {
            **blank,
            "checkpoint_step": step,
            "checkpoint_path": str(path),
            "resume_blocked_reason": (
                f"This run's checkpoint is already at step {step} of {target}, so there is "
                "nothing left to train."
            ),
        }
    return {
        "can_resume": True,
        "checkpoint_step": step,
        "checkpoint_path": str(path),
        "resume_blocked_reason": None,
    }


def resolve_resume_dir(path_value: str) -> Path:
    """Confines a client-supplied resume path to the outputs root; escapes raise ResumeError."""
    from core.training.resume import normalize_resume_output_dir
    from utils.paths import outputs_root

    try:
        resolved = Path(normalize_resume_output_dir(str(path_value)))
    except ValueError as error:
        message = str(error)
        if not message.startswith("Resume checkpoint"):
            # Don't leak server paths from the containment resolver's message.
            message = "Resume checkpoint must be inside Unsloth outputs."
        raise ResumeError(message) from error
    # A name that cleans to the outputs ROOT would sweep checkpoints across unrelated runs.
    if resolved.resolve(strict = False) == outputs_root().resolve(strict = False):
        raise ResumeError(
            f"'{path_value}' is the outputs folder itself, not a training run inside it."
        )
    return resolved


# Not read from the manifest: the route preflight must know what a complete bundle carries.
_TRAINER_RNG_STREAMS: tuple[str, ...] = ("loop", "variant")

_REQUIRED_STATE: tuple[tuple[str, str], ...] = (
    ("adapter", "the trained LoRA weights"),
    ("optimizer", "the optimizer moments"),
    ("scheduler", "the learning-rate schedule position"),
    ("rng", "the random-number generator state"),
)


def _assert_required_state(path: Path, manifest: dict[str, Any]) -> None:
    """Refuse a bundle missing state the trainer treats as mandatory, BEFORE teardown."""
    files = manifest.get("files")
    listed = files if isinstance(files, dict) else {}
    missing = [
        label
        for role, label in _REQUIRED_STATE
        if not isinstance(listed.get(role), str) or not listed.get(role)
    ]
    if str(manifest.get("kind") or "image") == "image" and not isinstance(
        manifest.get("sampler"), dict
    ):
        missing.append("the dataset sampler position")
    # restore_rng_state is best-effort per part, so check both random.Random streams exist.
    rng_manifest = manifest.get("rng")
    saved_streams = rng_manifest.get("streams") if isinstance(rng_manifest, dict) else None
    if not isinstance(saved_streams, dict) or not all(
        isinstance(saved_streams.get(name), (list, tuple)) for name in _TRAINER_RNG_STREAMS
    ):
        missing.append("the trainer's random-number streams")
    # A CUDA bundle resumed on XPU (or reverse) leaves the generator freshly seeded. Only a KNOWN
    # mismatch counts, so bundles predating this field still resume.
    saved_accel = rng_manifest.get("accelerator") if isinstance(rng_manifest, dict) else None
    if isinstance(saved_accel, str) and saved_accel and saved_accel != _rng_accelerator():
        missing.append(
            f"the random-number state for this accelerator (written on {saved_accel}, "
            f"resuming on {_rng_accelerator()})"
        )
    if not missing:
        return
    raise ResumeError(
        f"'{path.name}' is missing {_join_clauses(missing)}, so the run cannot be continued "
        "from it. Resume from an earlier checkpoint, or start a new run."
    )


def _join_clauses(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _assert_optimizer_buildable(path: Path, manifest: dict[str, Any]) -> None:
    """Runs before teardown and refuses only the provably impossible: 8-bit moments with no bitsandbytes."""
    saved = manifest.get("optimizer_class")
    if not isinstance(saved, str) or "bitsandbytes" not in saved:
        return
    reason = None
    if os.environ.get("UNSLOTH_DIFFUSION_FP32_OPTIM", "") in ("1", "true"):
        reason = "UNSLOTH_DIFFUSION_FP32_OPTIM forces plain torch AdamW on this host"
    else:
        try:
            import importlib.util
            if importlib.util.find_spec("bitsandbytes") is None:
                reason = "bitsandbytes is not installed on this host"
        except (ImportError, ValueError):
            reason = "bitsandbytes is not installed on this host"
    if reason is None:
        return
    raise ResumeError(
        f"'{path.name}' carries bitsandbytes 8-bit optimizer state, but {reason}, so its "
        "moments cannot be restored. Install bitsandbytes (or unset the override) to continue "
        "this run."
    )


def _assert_loadable(path: Path, manifest: dict[str, Any]) -> None:
    """Opens each state file for real, since the header scan passes pickles that torch.load refuses."""
    loaded = LoadedCheckpoint(path = path, manifest = manifest)
    files = manifest.get("files")
    for role in (files or {}) if isinstance(files, dict) else ():
        try:
            if role in ("adapter", "ema"):
                loaded.tensors(role)
            else:
                state = loaded.torch_state(role)
                if role == "rng" and not (
                    isinstance(state, dict) and state.get("torch_cpu") is not None
                ):
                    raise ResumeError(
                        f"'{path.name}' carries no torch random-number state, so the run "
                        "would continue on a different random stream. Resume from an earlier "
                        "checkpoint, or start a new run."
                    )
        except ResumeError:
            raise
        except Exception as error:  # noqa: BLE001 -- any unreadable state file is a refusal
            raise ResumeError(
                f"The '{role}' file in '{path.name}' could not be read back "
                f"({type(error).__name__}), so this checkpoint cannot be resumed."
            ) from error


def recorded_resume_targets(path_value: Any) -> Optional[tuple[str, ...]]:
    """LoRA targets recorded by the bundle a resume would continue, else None. Never raises."""
    try:
        root = resolve_resume_dir(str(path_value))
        explicit = read_checkpoint(root) if checkpoint_step(root) >= 0 else None
        candidates = [(root, explicit)] if explicit is not None else iter_valid_checkpoints(root)
        for _candidate, manifest in candidates:
            saved = CheckpointIdentity.from_dict((manifest or {}).get("identity"))
            if saved is not None and saved.lora_target_modules:
                return tuple(saved.lora_target_modules)
    except Exception:  # noqa: BLE001 -- best effort; the preflight reports the real problem
        pass
    return None


def recorded_bucketing(path_value: str) -> bool:
    """Whether the resumed bundle trained bucketed (pre-field or unreadable: False; preflight_resume refuses)."""
    try:
        root = resolve_resume_dir(path_value)
    except ResumeError:
        return False
    explicit = read_checkpoint(root) if checkpoint_step(root) >= 0 else None
    candidates = [(root, explicit)] if explicit is not None else iter_valid_checkpoints(root)
    for _path, manifest in candidates:
        identity = manifest.get("identity")
        if isinstance(identity, dict):
            return _optional_str(identity.get("bucketing")) == "on"
    return False


def preflight_resume(
    path_value: str, *, identity: CheckpointIdentity, target_steps: int
) -> tuple[str, int]:
    """Runs before the resident model is evicted, so a rejected resume does not cost the loaded pipeline."""
    root = resolve_resume_dir(path_value)
    # An adapter dir can be named like a bundle; explicit only when it IS a valid bundle.
    explicit = read_checkpoint(root) if checkpoint_step(root) >= 0 else None
    candidates: list[tuple[Path, dict]]
    if explicit is not None:
        candidates = [(root, explicit)]
    else:
        candidates = iter_valid_checkpoints(root)
        if not candidates and checkpoint_step(root) >= 0:
            raise ResumeError(
                f"'{root.name}' is not a complete training checkpoint (it is missing files, "
                "or was left behind by an interrupted save)."
            )
    if not candidates:
        raise ResumeError(
            "No complete training checkpoint was found for this run, so there is nothing to "
            "resume from. Start a new run instead."
        )
    # The newest bundle may fail torch.load while an older retained one is good; keep scanning.
    first_error: Optional[ResumeError] = None
    for path, manifest in candidates:
        try:
            return _validated_resume(path, manifest, identity, target_steps)
        except ResumeError as exc:
            # "Already at the target" is terminal; falling past it would retrain completed work.
            if getattr(exc, "terminal", False):
                raise
            if first_error is None:
                first_error = exc
    assert first_error is not None
    raise first_error


def _terminal(error: ResumeError) -> ResumeError:
    """Mark a refusal as final: it describes the REQUEST, not a damaged bundle, so the directory
    scan must not walk past it to an older checkpoint."""
    error.terminal = True  # type: ignore[attr-defined]
    return error


def _validated_resume(
    path: Path, manifest: dict[str, Any], identity: CheckpointIdentity, target_steps: int
) -> tuple[str, int]:
    """The full per-bundle gate: identity, required state, and a real load of every file."""
    saved = CheckpointIdentity.from_dict(manifest.get("identity"))
    if saved is None:
        raise ResumeError(
            "This checkpoint does not record what it was trained from, so it cannot be "
            "safely resumed."
        )
    reason = saved.mismatch_reason(identity)
    if reason:
        raise ResumeError(reason)
    _assert_required_state(path, manifest)
    _assert_optimizer_buildable(path, manifest)
    _assert_loadable(path, manifest)
    step = int(manifest.get("global_step") or 0)
    if target_steps and step >= int(target_steps):
        raise _terminal(
            ResumeError(
                f"This checkpoint is already at step {step} of {int(target_steps)}, so there "
                "is nothing left to train. Raise the step count to continue it."
            )
        )
    return str(path), step


@dataclass
class LoadedCheckpoint:
    """A validated bundle, with its tensors read lazily so a preflight never pays for them."""

    path: Path
    manifest: dict[str, Any]

    @property
    def step(self) -> int:
        return int(self.manifest.get("global_step") or 0)

    @property
    def target_steps(self) -> int:
        return int(self.manifest.get("target_steps") or 0)

    @property
    def ema_updates(self) -> int:
        return int(self.manifest.get("ema_updates") or 0)

    @property
    def optimizer_class(self) -> Optional[str]:
        value = self.manifest.get("optimizer_class")
        return str(value) if value else None

    @property
    def optimizer_param_names(self) -> Optional[list[str]]:
        """The trainable names in the order the optimizer held them, or None on a bundle from before
        this was recorded (which skips the check rather than failing it)."""
        value = self.manifest.get("optimizer_param_names")
        if not isinstance(value, list) or not value:
            return None
        return [str(name) for name in value]

    @property
    def progress(self) -> dict[str, Any]:
        state = self.manifest.get("progress")
        return state if isinstance(state, dict) else {}

    @property
    def running_loss(self) -> float:
        """The loss total accumulated up to ``step``, so a resumed run's reported average stays an
        average over the WHOLE run rather than restarting at the resume point."""
        try:
            return float(self.progress.get("running_loss") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def sampler_state(self) -> Optional[dict[str, Any]]:
        state = self.manifest.get("sampler")
        return state if isinstance(state, dict) else None

    @property
    def rng_json(self) -> Optional[dict[str, Any]]:
        state = self.manifest.get("rng")
        return state if isinstance(state, dict) else None

    def _file(self, role: str) -> Optional[Path]:
        files = self.manifest.get("files")
        name = files.get(role) if isinstance(files, dict) else None
        if not isinstance(name, str) or not name or Path(name).name != name:
            return None
        candidate = self.path / name
        return candidate if candidate.is_file() else None

    def tensors(
        self,
        role: str,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """The safetensors bundle for ``role`` (``adapter`` / ``ema``), or {}."""
        from safetensors.torch import load_file

        path = self._file(role)
        return load_file(str(path), device = device) if path is not None else {}

    def torch_state(self, role: str) -> Optional[Any]:
        """Loads with weights_only = True: the resume path is client-supplied, so no pickled code
        may execute."""
        import torch

        path = self._file(role)
        if path is None:
            return None
        return torch.load(str(path), map_location = "cpu", weights_only = True)


def load_checkpoint(path: str | os.PathLike[str]) -> LoadedCheckpoint:
    """Open a bundle that ``preflight_resume`` already accepted. Raises ResumeError if it became
    unreadable in between (a concurrent delete, a half-mounted volume)."""
    directory = Path(path).expanduser()
    manifest = read_checkpoint(directory)
    if manifest is None:
        raise ResumeError(
            f"The training checkpoint at '{directory}' could not be read; it may have been "
            "deleted or damaged since the run started."
        )
    return LoadedCheckpoint(path = directory, manifest = manifest)
