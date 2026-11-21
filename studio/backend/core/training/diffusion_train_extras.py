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


def accumulate_loss(acc: Any, loss: Any, accumulation_steps: int) -> Any:
    """``acc + float(loss) / accumulation_steps`` without a per-micro-step host sync: an fp64 device sum is the same
    IEEE arithmetic as the Python float one. Devices without fp64 (MPS, Arc / integrated XPUs) keep the Python float.
    Read with loss_value."""
    detached = loss.detach()
    if detached.device.type not in ("cpu", "cuda"):
        from core.inference.video_minimax_h3_pdd import device_has_fp64
        if not device_has_fp64(detached.device):
            return (acc if acc is not None else 0.0) + float(detached) / accumulation_steps
    term = detached.double() / accumulation_steps
    return term if acc is None else acc.add_(term)


def loss_value(acc: Any) -> float:
    return 0.0 if acc is None else float(acc)


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

    def _update_lists(self, model: Any) -> tuple[list, list]:
        """(shadows, params) for ``model``, cached per model."""
        import weakref

        cached = getattr(self, "_lists", None)
        if cached is not None and cached[0]() is model:
            return cached[1], cached[2]
        shadows, params = [], []
        for name, p in model.named_parameters():
            shadow = self._shadow.get(name)
            if shadow is None:
                continue
            shadows.append(shadow)
            params.append(p.detach())
        try:
            self._lists = (weakref.ref(model), shadows, params)
        except TypeError:  # not weak-referenceable: rebuild each call
            self._lists = None
        return shadows, params

    def update(self, model: Any) -> None:
        """Apply one EMA update from ``model``'s current trainable params."""
        import torch

        decay = self.effective_decay()
        shadows, params = self._update_lists(model)
        if shadows:
            with torch.no_grad():
                torch._foreach_mul_(shadows, decay)
                torch._foreach_add_(shadows, params, alpha = 1.0 - decay)
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
        """Write shadow values into params, returning displaced originals for restore to undo."""
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


def _latent_cache_salt() -> str:
    """Windows ROCm keys apart: latents encoded there before its VAE attention was routed to exact math are wrong."""
    try:
        from core.inference.diffusion_vae_attn_chunked import windows_rocm
        return "w" if windows_rocm() else ""
    except Exception:  # noqa: BLE001 -- unknown keeps the shared key
        return ""


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
# audio_vae: MiniMax-H3 caches its audio latents, so an edited local audio VAE must miss too.
_CACHE_SOURCE_SUBDIRS = ("text_encoder", "tokenizer", "vae", "audio_vae", "connectors")


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


# The persistent training conditioning cache is on by default; these turn the DEFAULT location off
# (an explicit cond_cache_dir is still honoured) and bound what any cache directory may hold.
TRAIN_COND_CACHE_ENV = "UNSLOTH_DIFFUSION_TRAIN_COND_CACHE"
TRAIN_COND_CACHE_MAX_GB_ENV = "UNSLOTH_DIFFUSION_TRAIN_COND_CACHE_MAX_GB"
_DEFAULT_COND_CACHE_MAX_GB = 20.0
# cond_cache_dir values meaning "no persistent cache for this run" (blank means the default instead).
COND_CACHE_OFF_VALUES = frozenset({"off", "none", "false", "0", "no", "disabled"})
# Under the account workspace (beside the other Studio caches), never the outputs root the run listings scan.
DEFAULT_COND_CACHE_SUBDIR = "cache/diffusion-train-cond"
# Only files this cache wrote are ever evicted, so a user directory holding other files is safe: a candidate needs a
# cache key name (kind, then ``_v`` and a version digit) and the cache's own safetensors header metadata.
_CACHE_ENTRY_NAME = re.compile(r"^(?:lat|txt|h3clip|h3img)_v\d[^/\\]*\.safetensors$")
# Written into every entry's header; entries from before the marker carry exactly the legacy metadata keys below.
_CACHE_OWNER_KEY = "unsloth_cond_cache"
_CACHE_OWNER_VALUE = "1"
_CACHE_LEGACY_META_KEYS = frozenset({"version", "family", "none_slots", "count"})
# Our headers list a handful of tensors; a larger header is not one of ours and is never parsed.
_CACHE_MAX_HEADER_BYTES = 1 << 20
# A write that would leave less than this free on the cache's disk is skipped (the run then saves its adapter).
_COND_CACHE_DISK_RESERVE_BYTES = 2 << 30


def cond_cache_is_off(value: Any) -> bool:
    return str(value or "").strip().lower() in COND_CACHE_OFF_VALUES


def default_cond_cache_enabled() -> bool:
    import os  # noqa: PLC0415
    return (os.environ.get(TRAIN_COND_CACHE_ENV) or "").strip().lower() not in COND_CACHE_OFF_VALUES


def default_cond_cache_dir() -> Optional[Path]:
    """The default persistent cache directory for the acting account, or None when the default is switched off
    (or the Studio paths cannot be resolved here, which only ever costs the re-encode)."""
    if not default_cond_cache_enabled():
        return None
    try:
        from utils.paths.storage_roots import account_path  # noqa: PLC0415
        return Path(account_path(DEFAULT_COND_CACHE_SUBDIR))
    except Exception:  # noqa: BLE001 -- the cache is an optimisation, never fatal
        return None


def effective_cond_cache_dir(value: Any) -> Optional[str]:
    """What a trainer opens for ``cfg.cond_cache_dir``: an off value is None, an explicit directory is itself,
    unset (None or blank) is the default directory unless the default is switched off."""
    if cond_cache_is_off(value):
        return None
    text = str(value).strip() if value is not None else ""
    if text:
        return text
    default = default_cond_cache_dir()
    return str(default) if default is not None else None


def cond_cache_max_bytes() -> int:
    """Size cap of one cache directory (oldest entries evicted first); 0 or a negative value means no cap."""
    import os  # noqa: PLC0415

    raw = (os.environ.get(TRAIN_COND_CACHE_MAX_GB_ENV) or "").strip()
    try:
        gb = float(raw) if raw else _DEFAULT_COND_CACHE_MAX_GB
    except ValueError:
        gb = _DEFAULT_COND_CACHE_MAX_GB
    return int(gb * 1024**3) if gb > 0 else 0


def compact_cache_namespace(family: str, *parts: str) -> str:
    """A short, filesystem-safe namespace: the family readable, the rest (repo ids, local paths, revisions) hashed,
    so a long local base path cannot push a key past the 255-byte file-name limit."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:20]
    return f"{_sanitize(family)}-{digest}"


# Longest readable namespace in a key; longer ones keep a prefix plus a hash of the whole.
_MAX_FAMILY_CHARS = 96


def _is_cache_entry(path: str) -> bool:
    """True only for a safetensors file PersistentConditioningCache wrote: its header metadata carries the owner marker,
    or (entries written before the marker) exactly the cache's own metadata keys over ``t<i>`` tensors. Reads the
    header alone; anything unreadable or unexpected is not ours, so it is never evicted."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) != 8:
                return False
            n = int.from_bytes(head, "little")
            if n <= 0 or n > _CACHE_MAX_HEADER_BYTES:
                return False
            raw = f.read(n)
        if len(raw) != n:
            return False
        header = json.loads(raw.decode("utf-8"))
        if not isinstance(header, dict):
            return False
        meta = header.get("__metadata__")
        if not isinstance(meta, dict):
            return False
        if meta.get(_CACHE_OWNER_KEY) == _CACHE_OWNER_VALUE:
            return True
        if set(meta) != _CACHE_LEGACY_META_KEYS:
            return False
        count = int(meta["count"])
        slots = json.loads(meta["none_slots"])
        names = [k for k in header if k != "__metadata__"]
        if not isinstance(slots, list) or len(names) + len(slots) != count:
            return False
        return all(re.fullmatch(r"t\d+", k) for k in names)
    except Exception:  # noqa: BLE001 -- not provably ours: keep it
        return False


class PersistentConditioningCache:
    """One safetensors file per cached item under ``cache_dir``.

    Keys carry everything that changes the encoded tensors: the cache format
    version, family, resolution (or explicit bucket shape), the source content
    hash, and -- for latents -- the crop/flip variant. Tensors are stored fp32
    exactly as the trainer's in-memory path holds them, so a reload is
    bit-identical to a fresh encode.

    The directory is bounded by ``max_bytes`` (``cond_cache_max_bytes()`` by default): a write that takes it over
    the cap evicts the least recently used entries (mtime, refreshed on every hit). Best-effort, never fatal."""

    def __init__(
        self,
        cache_dir: str,
        family: str,
        resolution: int,
        text_encoder: str = "",
        max_bytes: Optional[int] = None,
    ):
        self.root = Path(cache_dir).expanduser()
        self.family = _sanitize(family)
        if len(self.family) > _MAX_FAMILY_CHARS:
            # A local base path in the namespace would push a key past the 255-byte file-name limit.
            digest = hashlib.sha256(self.family.encode("utf-8")).hexdigest()[:20]
            self.family = f"{self.family[:_MAX_FAMILY_CHARS - 21]}-{digest}"
        self.resolution = int(resolution)
        self.text_encoder = (
            f"{_sanitize(text_encoder)[:16]}-{hashlib.sha1(text_encoder.encode()).hexdigest()[:10]}"
            if text_encoder
            else ""
        )
        self.root.mkdir(parents = True, exist_ok = True)
        self.max_bytes = cond_cache_max_bytes() if max_bytes is None else int(max_bytes)
        # Running size of the directory, read once here and kept current by put(); None until measured.
        self._bytes: Optional[int] = None
        # (name, inode, size) of files already confirmed as entries this cache wrote, so a rescan skips the header read.
        self._owned: set[tuple[str, int, int]] = set()

    def _entries(self) -> list[tuple[float, int, Path]]:
        import os  # noqa: PLC0415

        out = []
        with os.scandir(self.root) as it:
            for e in it:
                if not (_CACHE_ENTRY_NAME.match(e.name) and e.is_file(follow_symlinks = False)):
                    continue
                st = e.stat(follow_symlinks = False)
                ident = (e.name, st.st_ino, st.st_size)
                if ident not in self._owned:
                    if not _is_cache_entry(e.path):
                        continue
                    self._owned.add(ident)
                out.append((st.st_mtime, st.st_size, Path(e.path)))
        return out

    def _account(self, added: int, keep: Path) -> None:
        """Evict oldest-first until the directory is under the cap again; ``keep`` (just written) goes last."""
        if self.max_bytes <= 0:
            return
        try:
            if self._bytes is None:
                self._bytes = sum(size for _m, size, _p in self._entries())
            else:
                self._bytes += added
            if self._bytes <= self.max_bytes:
                return
            entries = sorted(self._entries(), key = lambda t: (t[2] == keep, t[0], t[2].name))
            total = sum(size for _m, size, _p in entries)
            # Down to 90% of the cap, so a run writing many entries does not rescan on every put.
            target = int(self.max_bytes * 0.9)
            for _mtime, size, path in entries:
                if total <= target:
                    break
                try:
                    path.unlink()
                    total -= size
                except OSError:
                    pass
            self._bytes = total
        except Exception:  # noqa: BLE001 -- eviction is best-effort; the run still trains
            self._bytes = None

    def _touch(self, path: Path) -> None:
        try:
            import os  # noqa: PLC0415
            os.utime(path)
        except OSError:
            pass

    def entry_key(self, kind: str, version: str, source_path: Optional[str], *parts: Any) -> str:
        """Generic key: ``{kind}_v{version}_{family}_{parts...}_{source content hash}``. Every part is sanitized."""
        fields = [f"{kind}_v{version}", self.family]
        fields += [_sanitize(str(p)) for p in parts]
        if source_path is not None:
            fields.append(_file_content_hash(source_path))
        return "_".join(fields)

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
            f"lat_v{_LATENT_CACHE_VERSION}{_latent_cache_salt()}_{self.family}_{geom}_"
            f"{_file_content_hash(image_path)}_{var}"
        )

    def text_key(self, caption: str) -> str:
        encoder = f"_{self.text_encoder}" if self.text_encoder else ""
        return f"txt_v{_CACHE_VERSION}_{self.family}{encoder}_{_text_content_hash(caption)}"

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.safetensors"

    def has(self, key: str) -> bool:
        try:
            return self.path_for(key).is_file()
        except OSError:
            return False

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
            _CACHE_OWNER_KEY: _CACHE_OWNER_VALUE,
        }
        final = self.path_for(key)
        tmp = final.with_suffix(".tmp")
        import shutil  # noqa: PLC0415

        nbytes = sum(t.numel() * t.element_size() for t in named.values())
        try:
            if shutil.disk_usage(str(self.root)).free < nbytes + _COND_CACHE_DISK_RESERVE_BYTES:
                return
        except OSError:
            pass
        try:
            save_file(named, str(tmp), metadata = meta)
            tmp.replace(final)
        except BaseException:
            # A full disk or a stop mid-write must not leave a partial file the size cap never counts.
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        try:
            size = final.stat().st_size
        except OSError:
            size = 0
        self._account(size, final)

    def get(self, key: str) -> Optional[tuple]:
        """Load a tuple previously stored with ``put``; None if absent/corrupt."""
        path = self.path_for(key)
        try:
            if not path.is_file():
                return None
        except OSError:  # an unusable name is a miss, never fatal
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
            # Least recently USED goes first under the size cap, so a hit refreshes the entry.
            self._touch(path)
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
