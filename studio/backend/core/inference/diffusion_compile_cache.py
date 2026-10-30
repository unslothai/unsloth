# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Torch Mega-cache of denoiser compile; exact-match fingerprint, a mismatch yields a miss not an
error; best-effort; manifest is the commit point over content-addressed bundles; old cache.bin
manifests stay valid, format version not bumped; LRU never evicts open or last-hour keys"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

# UNSLOTH_DIFFUSION_COMPILE_CACHE: auto (default) | 0 | 1; 1 also re-saves on a hit.
# _DIR root, _SAVE=0 load-only, _SYNC=1 save inline, _MAX_GB budget (default 20, 0 = no eviction).
_ENV_MODE = "UNSLOTH_DIFFUSION_COMPILE_CACHE"
_ENV_DIR = "UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR"
_ENV_SAVE = "UNSLOTH_DIFFUSION_COMPILE_CACHE_SAVE"
_ENV_SYNC = "UNSLOTH_DIFFUSION_COMPILE_CACHE_SYNC"
_ENV_MAX_GB = "UNSLOTH_DIFFUSION_COMPILE_CACHE_MAX_GB"
_DEFAULT_MAX_GB = 20.0

_LEGACY_ROOT = Path.home() / ".cache" / "unsloth" / "diffusion_compile_cache"

_MANIFEST_NAME = "manifest.json"
# Legacy bundle name: still read and collected, never written.
_BUNDLE_NAME = "cache.bin"
_BUNDLE_PREFIX = "bundle-"
_BUNDLE_SUFFIX = ".bin"
# What _atomic_write names its in-progress file, as a suffix: ".<final name>.<random>.tmp".
_TEMP_SUFFIX = ".tmp"
_FORMAT_VERSION = 1

_LAST_USED_NAME = "last_used"
_TOUCH_INTERVAL_SECONDS = 600.0
# Must exceed _TOUCH_INTERVAL_SECONDS: another process may be compiling into a key touched this recently.
_EVICT_GRACE_SECONDS = 3600.0
_TOMBSTONE_SUFFIX = ".evicting"

# An unnamed bundle this new may be another process's, published before its manifest.
_GC_GRACE_SECONDS = 60.0


def _bundle_name(digest: str) -> str:
    """Content-addressed file name, so writing a bundle can never damage the live one."""
    return f"{_BUNDLE_PREFIX}{digest[:16]}{_BUNDLE_SUFFIX}"


def _manifest_bundle(cdir: Path, manifest: dict[str, Any]) -> Path:
    """The bundle file a manifest names. Absent "bundle" means a pre-content-addressing manifest."""
    name = str(manifest.get("bundle") or _BUNDLE_NAME)
    # Defence against a manifest naming something outside its own directory.
    return cdir / Path(name).name


def _read_manifest(path: Path) -> Optional[dict[str, Any]]:
    try:
        loaded = json.loads(path.read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001 - unreadable/absent/half-written reads as "no manifest"
        return None
    return loaded if isinstance(loaded, dict) else None


def cache_mode() -> str:
    """``off`` | ``auto`` | ``on`` from the environment. ``auto`` is the default."""
    raw = (os.environ.get(_ENV_MODE) or "auto").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "yes"):
        return "on"
    return "auto"


def _save_enabled(mode: str) -> bool:
    if mode == "off":
        return False
    if mode == "on":
        return True
    # auto: save by default (without a saved bundle no user gets a warm restart); the SAVE env overrides: "0" load-only,
    # "1" on.
    return (os.environ.get(_ENV_SAVE) or "").strip().lower() not in ("0", "off", "false", "no")


def _portable_mode() -> bool:
    try:
        from utils.paths.storage_roots import portable_mode
    except ImportError:
        return False
    try:
        return bool(portable_mode())
    except Exception:  # noqa: BLE001 - never fail a cache lookup over this
        return False


def _toolchain_path_unparseable(value: str) -> bool:
    """Repeats the storage_roots rule in full, so the verdict never depends on sys.path."""
    try:
        from utils.paths.storage_roots import toolchain_path_unparseable
    except ImportError:
        return (
            any(ch.isspace() for ch in value)
            or "'" in value
            or '"' in value
            or (os.name != "nt" and "\\" in value)
        )
    return toolchain_path_unparseable(value)


def _parseable_cache_fallback(key: str, intended: str) -> str | None:
    """storage_roots' ready-to-publish fallback, or None when this module is reached without
    studio/backend on sys.path, where there is no safe directory to offer."""
    try:
        from utils.paths.storage_roots import parseable_cache_fallback
    except ImportError:
        return None
    return parseable_cache_fallback(key, intended)


def _default_root() -> Path:
    """Resolved per call, not a module constant: this module is imported before startup sets
    UNSLOTH_STUDIO_HOME, which storage_roots reads."""
    try:
        from utils.paths.storage_roots import cache_root as studio_cache_root
    except ImportError:
        return _LEGACY_ROOT
    return studio_cache_root() / "diffusion_compile_cache"


def sync_saves() -> bool:
    """Whether ``save_async`` must write inline instead of handing off to the worker."""
    return (os.environ.get(_ENV_SYNC) or "").strip().lower() in ("1", "on", "true", "yes")


def cache_root() -> Path:
    """The root every bundle, manifest and inductor artifact is WRITTEN under."""
    root = os.environ.get(_ENV_DIR)
    if root:
        return Path(root)
    return _default_root()


def legacy_cache_root() -> Optional[Path]:
    """Read-only pre-relocation root for reads; never a write root, so an upgrade does not pin home."""
    if os.environ.get(_ENV_DIR) or _portable_mode():
        return None
    # Equal when storage_roots is unavailable and the default IS the legacy root.
    if _LEGACY_ROOT == _default_root():
        return None
    # Instance-level wrap of pipe.encode_prompt: dies with the pipe, never mutates the class; no-op (False) if env knob unset, no encode_prompt, or setup fails.
    try:
        if not _LEGACY_ROOT.exists():
            return None
    except OSError:
        return None
    return _LEGACY_ROOT


def _triton_version() -> Optional[str]:
    try:
        import triton  # noqa: PLC0415
        return str(getattr(triton, "__version__", None))
    except Exception:  # noqa: BLE001 - triton optional
        return None


def _diffusers_version() -> Optional[str]:
    try:
        import diffusers  # noqa: PLC0415
        return str(getattr(diffusers, "__version__", None))
    except Exception:  # noqa: BLE001
        return None


def environment_fingerprint() -> dict[str, Any]:
    """Hard portability fields; any change invalidates a bundle, mirroring torch's inductor cache key."""
    fp: dict[str, Any] = {
        "format": _FORMAT_VERSION,
        "torch": None,
        "torch_cuda": None,
        "triton": _triton_version(),
        "diffusers": _diffusers_version(),
        "gpu_name": None,
        "gpu_capability": None,
    }
    try:
        import torch  # noqa: PLC0415

        fp["torch"] = str(torch.__version__)
        fp["torch_cuda"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            # Current device, not 0: artifacts are architecture-specific.
            index = torch.cuda.current_device()
            fp["gpu_name"] = torch.cuda.get_device_name(index)
            cap = torch.cuda.get_device_capability(index)
            fp["gpu_capability"] = f"sm_{cap[0]}{cap[1]}"
        # torch keys artifacts on this tag (set by diffusion_fp8_unit_scale); a bundle from another tag would load,
        # miss, recompile and never be rewritten. Only a set tag joins the key, so untagged bundles stay valid.
        tag = getattr(getattr(torch, "compiler", None), "config", None)
        tag = getattr(tag, "cache_key_tag", None)
        if isinstance(tag, str) and tag:
            fp["torch_cache_key_tag"] = tag
    except Exception:  # noqa: BLE001 - best-effort
        pass
    return fp


# Switches that change the traced graph after begin() (swaps in apply_speed_optims / first forward), so the tree can't
# show them. name -> default ("raw" = value-carrying). Keyed only off default; runtime-only knobs are excluded.
_GRAPH_SWITCHES = {
    "UNSLOTH_DIFFUSION_INT8_GEMM": "auto",
    "UNSLOTH_DIFFUSION_INT8_GEMM_CONVROT": "on",
    "UNSLOTH_DIFFUSION_INT8_GEMM_STREAMED": "on",
    "UNSLOTH_DIFFUSION_INT8_GEMM_UNLISTED": "auto",
    "UNSLOTH_DIFFUSION_INT8_ROTQUANT": "auto",
    "UNSLOTH_DIFFUSION_INT8_ROTQUANT_UNLISTED": "auto",
    "UNSLOTH_DIFFUSION_INT8_FUSED": "on",
    "UNSLOTH_DIFFUSION_INT8_TEXT_STREAM": "on",
    "UNSLOTH_DIFFUSION_INT8_MODULATION": "on",
    # Count of dense outlier channels (0 = none, clamped to 16, junk = the default), keyed on the value the reader uses.
    "UNSLOTH_DIFFUSION_INT8_MOD_OUTLIERS": "count:8",
    "UNSLOTH_DIFFUSION_INT8_GEMV": "on",
    "UNSLOTH_DIFFUSION_ZIMAGE_FUSED": "on",
    "UNSLOTH_DIFFUSION_Q21_REAL_ROPE": "on",
    "UNSLOTH_DIFFUSION_QWEN_REAL_ROPE": "on",
    "UNSLOTH_DIFFUSION_EAGER_PATCHES": "on",
    "UNSLOTH_DIFFUSION_ARCH_PATCHES": "on",
    "UNSLOTH_DIFFUSION_FLUX2_FUSED_ROPE": "on",
    "UNSLOTH_DIFFUSION_BLOCK_RESTRIDE": "on",
    "UNSLOTH_DIFFUSION_INDUCTOR_BACKPORTS": "on",
    "UNSLOTH_STATIC_STREAM_MERGE_DETECT": "off",
    "UNSLOTH_TORCHAO_INDUCTOR_CONFIG": "off",
    "UNSLOTH_NVFP4_BACKEND": "raw:auto",
    "UNSLOTH_DIFFUSION_FP8_UNIT_SCALE": "auto",
    "UNSLOTH_DIFFUSION_FP8_UNIT_SCALE_MIN_ROWS": "rows:256",
    # Unset is off; "auto" protects the head + last step, so "auto" is NOT the default here.
    "UNSLOTH_NVFP4_PROTECT_STEPS": "raw",
}
_OFF_WORDS = ("0", "off", "false", "no", "none")
_ON_WORDS = ("1", "on", "true", "yes", "force")
_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+")


def _switch_value(name: str, default: str) -> Optional[str]:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return None
    if default.startswith("rows:"):
        # A row threshold keys on the value the reader uses (max(1, int)); junk falls back to the default there.
        try:
            rows = str(max(1, int(raw)))
        except ValueError:
            return None
        return None if rows == default.partition(":")[2] else rows
    if default.startswith("count:"):
        try:
            count = str(max(0, min(16, int(raw))))
        except ValueError:
            return None
        return None if count == default.partition(":")[2] else count
    if default.startswith("raw"):
        return None if raw in _OFF_WORDS or raw == default.partition(":")[2] else raw
    value = "off" if raw in _OFF_WORDS else "on" if raw in _ON_WORDS else None
    if value is None:
        return None
    return None if value == default else value


def graph_switches() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, default in sorted(_GRAPH_SWITCHES.items()):
        value = _switch_value(name, default)
        if value is not None:
            out[name] = value
    return out


def _tensor_signature(tensor: Any, depth: int = 0) -> str:
    """``Cls/dtype(inner...)``: what a traced op dispatches on, never shapes or values."""
    head = f"{type(tensor).__name__}/{getattr(tensor, 'dtype', None)}"
    if depth > 2:
        return head
    inner: list[str] = []
    cls = type(tensor)
    data_names = [
        *(getattr(cls, "tensor_data_names", None) or ()),
        *(getattr(cls, "optional_tensor_data_names", None) or ()),
    ]
    if data_names:
        for name in data_names:
            value = getattr(tensor, name, None)
            inner.append(
                f"{name}={_tensor_signature(value, depth + 1) if value is not None else None}"
            )
        for attr in ("tensor_attribute_names", "optional_tensor_attribute_names"):
            for name in getattr(cls, attr, None) or ():
                if name == "shape":
                    continue
                if name == "block_size":
                    # Relative to the shape: per-row [1, k] vs per-tensor [n, k] trace different scale broadcasts.
                    shape = tuple(getattr(tensor, "shape", ()) or ())
                    block = tuple(getattr(tensor, name, None) or ())
                    rel = ",".join(
                        "full" if i < len(shape) and b == shape[i] else str(b)
                        for i, b in enumerate(block)
                    )
                    inner.append(f"block_size={rel}")
                    continue
                inner.append(f"{name}={_ADDRESS_RE.sub('', repr(getattr(tensor, name, None)))}")
    elif cls.__name__ not in ("Tensor", "Parameter") and hasattr(tensor, "__tensor_flatten__"):
        # Older wrappers (AffineQuantizedTensor): inner tensors only, the flatten context carries the shape.
        try:
            names, _ctx = tensor.__tensor_flatten__()
            for name in names:
                inner.append(f"{name}={_tensor_signature(getattr(tensor, name, None), depth + 1)}")
        except Exception:  # noqa: BLE001
            inner.append("flatten=unavailable")
    return head + (f"({'; '.join(inner)})" if inner else "")


def _weight_signature(module: Any, weight: Any) -> str:
    """Scale dtype (fp32 hosted vs bf16 runtime) and W8A8 vs weight-only both land here."""
    parts = [type(module).__name__, _tensor_signature(weight)]
    group = getattr(module, "convrot_groupsize", None)
    if group is not None:
        parts.append(f"convrot{group}")
    rot = getattr(module, "rot_group", None)
    if rot:
        parts.append(f"rot{rot}")
    if getattr(module, "act_int8", False):
        parts.append("act_int8")
    fmt = getattr(module, "fp8_format", None)
    if isinstance(fmt, str) and fmt != "float8_e4m3fn":
        # Stored E5M2 codes decode through a different integer graph; E4M3 keeps the key it always had.
        parts.append(fmt)
    fwd = module.__dict__.get("forward") if hasattr(module, "__dict__") else None
    if fwd is not None:
        func = getattr(fwd, "__func__", fwd)
        parts.append(
            f"fwd={getattr(func, '__module__', '')}.{getattr(func, '__qualname__', type(func).__name__)}"
        )
    return "/".join(parts)


def _stored_weight(module: Any) -> tuple[Any, bool]:
    """The stored weight, never a ``weight`` property: native layers dequantize the whole matrix on each read."""
    d = getattr(module, "__dict__", None) or {}
    for store in ("_parameters", "_buffers"):
        weight = (d.get(store) or {}).get("weight")
        if weight is not None:
            return weight, False
    if d.get("weight") is not None:
        return d["weight"], False
    packed = (d.get("_buffers") or {}).get("weight_q")
    return packed, packed is not None


def graph_variant(transformer: Any) -> dict[str, Any]:
    """Structural descriptor of the loaded quant artifact (no weight hashing): stable across processes, distinct for
    plain vs ConvRot, hosted vs ComfyUI vs runtime quantise, torchao vs native. Never raises."""
    variant: dict[str, Any] = {}
    try:
        import torch  # noqa: PLC0415

        counts: dict[str, int] = {}
        for module in transformer.modules():
            weight, packed = _stored_weight(module)
            if not isinstance(weight, torch.Tensor):
                continue
            sig = _weight_signature(module, weight) + ("/packed" if packed else "")
            counts[sig] = counts.get(sig, 0) + 1
        variant["weights"] = dict(sorted(counts.items()))
    except Exception as exc:  # noqa: BLE001 - an unreadable tree keys apart rather than sharing a bundle
        variant["weights"] = f"unavailable: {type(exc).__name__}"
    try:
        from .diffusion_convrot import CONVROT_ATTR  # noqa: PLC0415
        rotation = getattr(transformer, CONVROT_ATTR, None)
        if isinstance(rotation, dict):
            variant["rotation"] = {k: rotation[k] for k in sorted(rotation)}
    except Exception:  # noqa: BLE001
        pass
    return variant


def _has_quantized_layout(variant: dict[str, Any]) -> bool:
    """A ComfyUI single file keys quant as none, so the tree decides."""
    if variant.get("rotation"):
        return True
    weights = variant.get("weights")
    if not isinstance(weights, dict):
        return False
    for sig in weights:
        parts = sig.split("/")
        if len(parts) < 2 or parts[1] not in ("Parameter", "Tensor") or "(" in sig:
            return True
        # An instance forward alone is not quantized: offload hooks install one on dense loads too.
        if any(p.startswith(("convrot", "rot", "act_int8", "packed")) for p in parts[3:]):
            return True
    return False


def model_fingerprint(
    *,
    family: Any,
    transformer: Any,
    dtype: Any,
    quant: Any,
    attention_backend: Any,
    compile_kwargs: dict[str, Any],
    shape_bucket: Any = None,
    reduction_filter: bool = False,
) -> dict[str, Any]:
    """Model-graph key: the transformer class and its repeated blocks, so the key tracks what is
    compiled."""
    blocks = list(getattr(transformer, "_repeated_blocks", []) or [])
    if not blocks and transformer is not None:
        from .diffusion_regional_compile import verified_repeated_blocks
        blocks = list(verified_repeated_blocks(type(transformer).__name__))
    if quant is not None and transformer is not None:
        # Native layers compile to a different graph than torchao under the same scheme name.
        from .diffusion_native_quant import native_quant_signature
        quant = native_quant_signature(transformer) or quant
    fp = {
        "family": str(family),
        "transformer_cls": type(transformer).__name__ if transformer is not None else None,
        "repeated_blocks": sorted(str(b) for b in blocks),
        "dtype": str(dtype),
        "quant": str(quant) if quant is not None else "none",
        "attention_backend": str(attention_backend) if attention_backend is not None else "default",
        # A False vae_decode is what every load keyed before the flag existed, so it is left out of the key.
        "compile_kwargs": {
            k: compile_kwargs[k]
            for k in sorted(compile_kwargs)
            if not (k == "vae_decode" and not compile_kwargs[k])
        },
        "shape_bucket": shape_bucket,
    }
    # INT8 and INT8-ConvRot share "int8" but compile different graphs. Quantized layouts only: dense / GGUF keys hold.
    if str(quant) != "gguf" and transformer is not None:
        variant = graph_variant(transformer)
        if quant is not None or _has_quantized_layout(variant):
            fp["variant"] = variant
    switches = graph_switches()
    if switches:
        fp["switches"] = switches
    # Added only when armed, so other bundles keep their key.
    try:
        from .diffusion_dynamic_text import fingerprint as _dynamic_text_fp
        dynamic_text = _dynamic_text_fp(transformer, compile_kwargs.get("dynamic", True))
    except Exception:  # noqa: BLE001
        dynamic_text = None
    if dynamic_text:
        fp["dynamic_text"] = dynamic_text
    # Inductor keys graphs on dynamic_scale_rblock: an old bundle would hit, miss every graph, and never be rewritten.
    from .diffusion_compile_config import reduction_blocks_pinned, reduction_config_filter_available

    if reduction_blocks_pinned():
        fp["inductor"] = {"dynamic_scale_rblock": False}
        # Only for a family whose compile pins the reduction-config filter, so every other bundle keeps its key.
        if reduction_filter and reduction_config_filter_available():
            fp["inductor"]["force_filter_reduction_configs"] = True
    return fp


def cache_key(env_fp: dict[str, Any], model_fp: dict[str, Any]) -> str:
    payload = json.dumps({"env": env_fp, "model": model_fp}, sort_keys = True, default = str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclasses.dataclass
class CacheContext:
    """Per-load cache state carried between ``begin`` and ``save``/``restore``.

    ``shapes`` tracks the (width, height, batch) tuples whose STATIC-compile artifacts the
    bundle covers (persisted in the manifest). A dynamic compile reuses one artifact; a
    static compile produces NEW artifacts per shape, so the caller registers each shape
    and clears ``saved`` on a new one so the next ``save`` rewrites the enriched set."""

    key: str
    dir: Path
    bundle: Path
    manifest_path: Path
    env_fp: dict[str, Any]
    model_fp: dict[str, Any]
    mode: str
    hit: bool = False
    saved: bool = False
    prev_inductor_dir: Optional[str] = None
    prev_inductor_dir_set: bool = False
    shapes: set = dataclasses.field(default_factory = set)
    # Lets a background save avoid marking a concurrently registered shape persisted.
    dirty_seq: int = 0
    # Recompiled artifacts often hash the same, so overwrite instead of the exists() shortcut.
    rejected_bundle: Optional[str] = None
    last_touch: float = 0.0


def begin(
    *,
    family: Any,
    transformer: Any,
    dtype: Any,
    quant: Any,
    attention_backend: Any,
    compile_kwargs: dict[str, Any],
    shape_bucket: Any = None,
    logger: Any = None,
    reduction_filter: bool = False,
) -> Optional[CacheContext]:
    """Points inductor at a per-key dir and loads a matching bundle before compile; never raises."""
    mode = cache_mode()
    if mode == "off":
        return None
    try:
        import torch  # noqa: PLC0415
        if not (
            hasattr(torch.compiler, "save_cache_artifacts")
            and hasattr(torch.compiler, "load_cache_artifacts")
        ):
            _warn(logger, "Mega-cache API unavailable (need torch >= 2.7); skipping")
            return None
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"torch import failed: {exc}")
        return None

    env_fp = environment_fingerprint()
    model_fp = model_fingerprint(
        family = family,
        transformer = transformer,
        dtype = dtype,
        quant = quant,
        attention_backend = attention_backend,
        compile_kwargs = compile_kwargs,
        shape_bucket = shape_bucket,
        reduction_filter = reduction_filter,
    )
    key = cache_key(env_fp, model_fp)
    cdir = cache_root() / key
    ctx = CacheContext(
        key = key,
        dir = cdir,
        bundle = cdir / _BUNDLE_NAME,
        manifest_path = cdir / _MANIFEST_NAME,
        env_fp = env_fp,
        model_fp = model_fp,
        mode = mode,
    )

    # Same reason as restore(): the previous load's save reads the inductor dir that is about to be repointed.
    if not wait_for_saves():
        _warn(
            logger,
            f"compile-cache: background save still running after {_SAVE_JOIN_TIMEOUT:.0f}s; continuing",
        )

    try:
        cdir.mkdir(parents = True, exist_ok = True)
        inductor_dir = str(cdir / "inductor")
        # Startup applies the same test before pinning TORCHINDUCTOR_CACHE_DIR.
        if _toolchain_path_unparseable(inductor_dir):
            # Keeping the shared fallback would serialise it into every bundle; isolate per cdir.
            isolated = _parseable_cache_fallback("TORCHINDUCTOR_CACHE_DIR", inductor_dir)
            if isolated is None:
                _warn(
                    logger,
                    f"compile-cache: leaving TORCHINDUCTOR_CACHE_DIR as it is: {inductor_dir} "
                    "holds a character the C++ builders cannot paste into a command line "
                    "unquoted, and no safe replacement is available",
                )
            else:
                ctx.prev_inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
                ctx.prev_inductor_dir_set = True
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = isolated
        else:
            ctx.prev_inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
            ctx.prev_inductor_dir_set = True
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_dir
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"could not set TORCHINDUCTOR_CACHE_DIR: {exc}")

    # The manifest is the commit point and only names a fully written bundle.
    published = _read_manifest(ctx.manifest_path)
    if published is not None:
        ctx.bundle = _manifest_bundle(cdir, published)

    _register_live(cdir)
    _touch_last_used(ctx)

    # Also collect here: two saves inside the grace window would leak the first bundle.
    _collect_superseded(cdir, logger)

    # Path.exists() raises when a parent denies traversal.
    try:
        pair_present = published is not None and ctx.bundle.exists()
    except OSError as exc:
        _warn(logger, f"compile-cache: cannot check the bundle at {ctx.dir}: {exc}")
        pair_present = False
    if pair_present:
        ctx.hit = _try_load(ctx, logger)
    if not ctx.hit:
        # An install that predates the relocation may still hold this key under the old root.
        ctx.hit = _load_from_legacy(ctx, logger)
    if not ctx.hit:
        _info(logger, f"compile-cache: no bundle for key {key} (will compile locally)")
    elif mode != "on":
        # Loaded artifacts == on-disk artifacts, so nothing to save. A new static-compile shape
        # re-dirties via register_shape; mode "on" keeps saving.
        ctx.saved = True
    # A hit never saves, so an over-budget cache would otherwise never shrink.
    if _save_enabled(mode):
        _evict_soon(logger)
    return ctx


def _load_from_legacy(ctx: CacheContext, logger: Any) -> bool:
    """Copies a same-key legacy bundle forward, so the read fallback does not become permanent."""
    root = legacy_cache_root()
    if root is None:
        return False
    ldir = root / ctx.key
    manifest_path = ldir / _MANIFEST_NAME
    # The manifest is the commit point here too, so it decides which legacy bundle is live; a
    # pre-content-addressing one names cache.bin.
    published = _read_manifest(manifest_path)
    if published is None:
        return False
    bundle = _manifest_bundle(ldir, published)
    try:
        if not bundle.exists():
            return False
    except OSError:
        # Same reason as legacy_cache_root: a pair we cannot even stat is a miss.
        return False
    if not _try_load(ctx, logger, bundle = bundle, manifest_path = manifest_path):
        return False
    if _save_enabled(ctx.mode):
        try:
            ctx.dir.mkdir(parents = True, exist_ok = True)
            # Under the name the copied manifest names, which is not ctx.bundle unless the legacy
            # pair predates content addressing. Published like begin() does, so the context goes
            # on naming the live bundle in the write root.
            migrated = _manifest_bundle(ctx.dir, published)
            _atomic_copy(bundle, migrated)
            # Bundle first: a manifest without one reads as a miss, not an unservable hit.
            _atomic_copy(manifest_path, ctx.manifest_path)
            ctx.bundle = migrated
            _info(logger, f"compile-cache: migrated legacy bundle for key {ctx.key}")
        except OSError as exc:
            _warn(logger, f"compile-cache: could not migrate legacy bundle: {exc}")
    return True


def register_shape(ctx: Optional[CacheContext], shape: Any, *, static: bool) -> None:
    """Static compiles only: a new shape clears saved so the next save rewrites the bundle with it."""
    if ctx is None or not static:
        return
    try:
        key = tuple(shape)
        # Under the dirty lock so a background save cannot publish saved = True over this clear.
        with _dirty_lock:
            if key not in ctx.shapes:
                ctx.shapes.add(key)
                ctx.saved = False
                ctx.dirty_seq += 1
    except Exception:  # noqa: BLE001 - bookkeeping only
        pass


def mark_recompiled(ctx: Optional[CacheContext]) -> None:
    """A render compiled new graphs for a shape the bundle already covers (automatic dynamic re-specialises on the
    first new text length, keyed by nothing ``register_shape`` sees), so the bundle on disk lacks them: clear
    ``saved`` so the next ``save`` rewrites it. Never raises."""
    if ctx is None:
        return
    try:
        with _dirty_lock:
            ctx.saved = False
            ctx.dirty_seq += 1
    except Exception:  # noqa: BLE001 - bookkeeping only
        pass


def _try_load(
    ctx: CacheContext,
    logger: Any,
    *,
    bundle: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> bool:
    """Validate and load a bundle pair, defaulting to the context's own (write-root) one."""
    bundle = bundle if bundle is not None else ctx.bundle
    manifest_path = manifest_path if manifest_path is not None else ctx.manifest_path
    try:
        manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: unreadable manifest: {exc}")
        return False

    # json.loads may return [] or null; a bad entry must be a miss.
    if not isinstance(manifest, dict):
        _warn(logger, "compile-cache: manifest is not an object; ignoring")
        return False

    # Exact-match guard (defence in depth: torch also validates internally on load).
    if manifest.get("env") != ctx.env_fp or manifest.get("model") != ctx.model_fp:
        _warn(logger, "compile-cache: fingerprint mismatch; falling back to local compile")
        return False

    try:
        data = bundle.read_bytes()
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: cannot read bundle: {exc}")
        return False

    # Integrity check (corruption / truncation; not a security signature).
    digest = hashlib.sha256(data).hexdigest()
    if manifest.get("sha256") and manifest["sha256"] != digest:
        _warn(logger, "compile-cache: bundle checksum mismatch; ignoring")
        ctx.rejected_bundle = ctx.bundle.name
        return False

    try:
        import torch  # noqa: PLC0415

        info = torch.compiler.load_cache_artifacts(data)
        if info is None:
            _warn(logger, "compile-cache: load_cache_artifacts returned None (no hit)")
            return False
        try:
            ctx.shapes = {tuple(s) for s in manifest.get("shapes", [])}
        except Exception:  # noqa: BLE001 - coverage bookkeeping only
            ctx.shapes = set()
        _info(logger, f"compile-cache: loaded bundle for key {ctx.key}")
        return True
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: load failed: {exc}")
        return False


def _atomic_write(path: Path, data: bytes) -> None:
    """Temp file in the same directory plus os.replace, so a killed save never exposes a partial file."""
    tmp: Optional[str] = None
    try:
        fd, tmp = tempfile.mkstemp(
            dir = str(path.parent), prefix = f".{path.name}.", suffix = _TEMP_SUFFIX
        )
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy via a same-directory temp file and os.replace, so a torn migrated manifest is never visible."""
    tmp: Optional[str] = None
    try:
        fd, tmp = tempfile.mkstemp(dir = str(dst.parent), prefix = f".{dst.name}.", suffix = _TEMP_SUFFIX)
        os.close(fd)
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        tmp = None
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _collect_superseded(cdir: Path, logger: Any) -> list[str]:
    """Re-reads the on-disk manifest before deleting, and spares bundles inside the grace window."""
    removed: list[str] = []
    try:
        manifest = _read_manifest(cdir / _MANIFEST_NAME)
        live = _manifest_bundle(cdir, manifest).name if manifest is not None else None
        cutoff = time.time() - _GC_GRACE_SECONDS
        for path in cdir.iterdir():
            name = path.name
            if name == live or not path.is_file():
                continue
            if not (
                name == _BUNDLE_NAME
                or (name.startswith(_BUNDLE_PREFIX) and name.endswith(_BUNDLE_SUFFIX))
                # A killed save leaves a temp file (can be GB); the grace window never takes an in-flight write
                or (name.startswith(".") and name.endswith(_TEMP_SUFFIX))
            ):
                continue
            try:
                if path.stat().st_mtime > cutoff:
                    continue
                path.unlink()
                removed.append(name)
            except OSError:
                # Another process got there first, or it is busy on Windows. Next save tries again.
                continue
    except Exception as exc:  # noqa: BLE001 - housekeeping must never fail a save
        _warn(logger, f"compile-cache: could not collect superseded bundles: {exc}")
    return removed


def _compile_lock() -> Any:
    try:
        from torch._dynamo.convert_frame import compile_lock  # noqa: PLC0415
        return compile_lock
    except Exception:  # noqa: BLE001 - no dynamo lock on this build: save unlocked, as before
        return contextlib.nullcontext()


def _one_group_per_type(data: bytes, logger: Any) -> bytes:
    """One ``(type, artifacts)`` group per type: torch appends a group per save and loads them with ``dict(...)``,
    so a re-saved bundle kept only the newest group of each type."""
    try:
        from torch.compiler._cache import (  # noqa: PLC0415
            CacheArtifactManager,
            _deserialize_single_cache,
            _serialize_single_cache,
        )
        from torch.utils._appending_byte_serializer import AppendingByteSerializer  # noqa: PLC0415
    except ImportError:
        # torch 2.7 writes one flat artifact list and loads it without dict(), so there is nothing to merge
        return data
    try:
        CacheArtifactManager._ensure_cache_artifacts_registered()
        groups = AppendingByteSerializer.to_list(data, deserialize_fn = _deserialize_single_cache)
        # Dedupe on (key, content) only: Inductor / AOT entries sharing a key are distinct guarded candidates
        # (stored under a content hash), and a repeat moves to its latest position so a single-slot type
        # (autotune) still applies its newest bytes last, as torch's in-order load would.
        merged: dict[str, dict] = {}
        for kind, artifacts in groups:
            bucket = merged.setdefault(kind, {})
            for artifact in artifacts:
                content = artifact.content
                if not isinstance(content, (bytes, bytearray, memoryview)):
                    content = json.dumps(
                        content, sort_keys = True, default = repr
                    ).encode()  # autotune: JSON
                ident = (artifact.key, hashlib.sha256(content).digest())
                bucket.pop(ident, None)
                bucket[ident] = artifact
        if len(merged) == len(groups):
            return data
        out = AppendingByteSerializer(serialize_fn = _serialize_single_cache)
        out.extend((kind, list(bucket.values())) for kind, bucket in merged.items())
        return out.to_bytes()
    except Exception as exc:  # noqa: BLE001 - keep torch's own bytes
        _warn(
            logger,
            f"compile-cache: could not merge artifact groups ({exc}); saving torch's bundle as is",
        )
        return data


def _write_bundle(ctx: CacheContext, logger: Any) -> bool:
    """The save itself. Runs on the caller's thread (``save``) or the worker (``save_async``)."""
    if not _save_enabled(ctx.mode) or ctx.saved:
        return False
    # Snapshot before collecting: the request thread may add to the set.
    with _dirty_lock:
        seq = ctx.dirty_seq
        shapes = sorted(list(s) for s in ctx.shapes)
    try:
        import torch  # noqa: PLC0415

        # an artifact recorded mid-serialise is dropped from pending but stays "seen", so it would never be saved
        with _compile_lock():
            result = torch.compiler.save_cache_artifacts()
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: save_cache_artifacts failed: {exc}")
        return False
    if not result or result[0] is None:
        _warn(logger, "compile-cache: nothing to save (empty artifacts)")
        return False

    data = _one_group_per_type(result[0], logger)
    try:
        ctx.dir.mkdir(parents = True, exist_ok = True)
        digest = hashlib.sha256(data).hexdigest()
        bundle = ctx.dir / _bundle_name(digest)
        manifest = {
            "format": _FORMAT_VERSION,
            "key": ctx.key,
            "created": time.time(),
            "bytes": len(data),
            "sha256": digest,
            # The bundle THIS manifest is paired with. Content-addressed, so a save writes a file no reader is
            # using and the previous pair stays whole and addressed until the manifest below commits.
            "bundle": bundle.name,
            "env": ctx.env_fp,
            "model": ctx.model_fp,
            # Static-compile shape coverage (register_shape); unused by dynamic compiles.
            "shapes": shapes,
        }
        # Bundle then manifest: the manifest is the single commit point.
        if not bundle.exists() or bundle.name == ctx.rejected_bundle:
            _atomic_write(bundle, data)
        _atomic_write(
            ctx.manifest_path,
            json.dumps(manifest, indent = 2, sort_keys = True, default = str).encode("utf-8"),
        )
        ctx.bundle = bundle
        _collect_superseded(ctx.dir, logger)
        evict(logger = logger)
        with _dirty_lock:
            # Per-slot layout codes for _flatten/_unflatten (leading int64 tensor): -1 None, -2 a bare tensor, n >= 0 a list of n tensors
            if ctx.dirty_seq == seq:
                ctx.saved = True
        _info(logger, f"compile-cache: saved bundle ({len(data)} bytes) for key {ctx.key}")
        return True
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: could not write bundle: {exc}")
        return False


def save(ctx: Optional[CacheContext], *, logger: Any = None) -> bool:
    """Synchronous bundle write; no-op unless the context is dirty, since a bundle hit starts clean."""
    if ctx is None:
        return False
    return _write_bundle(ctx, logger)


# Single daemon worker serialises all saves.
_dirty_lock = threading.RLock()
_worker_cv = threading.Condition(threading.Lock())
_worker_thread: Optional[threading.Thread] = None
_worker_queue: list[tuple[CacheContext, Any]] = []
_worker_active: Optional[CacheContext] = None

# ~150x the worst measured save (0.19 s / 71 MB on B200); longer looks wedged.
_SAVE_JOIN_TIMEOUT = 30.0


def _worker_loop() -> None:
    global _worker_active
    while True:
        with _worker_cv:
            while not _worker_queue:
                _worker_cv.wait()
            ctx, logger = _worker_queue.pop(0)
            _worker_active = ctx
        try:
            _write_bundle(ctx, logger)
        except Exception as exc:  # noqa: BLE001 - a cache write must never reach a render
            _warn(logger, f"compile-cache: background save failed: {exc}")
        finally:
            with _worker_cv:
                _worker_active = None
                _worker_cv.notify_all()


def _start_worker_locked() -> None:
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return
    _worker_thread = threading.Thread(
        target = _worker_loop, name = "unsloth-compile-cache-save", daemon = True
    )
    _worker_thread.start()


def save_async(ctx: Optional[CacheContext], *, logger: Any = None) -> bool:
    """Queue save on the worker; an in-flight save is re-queued since it cannot include later shapes."""
    if ctx is None:
        return False
    note_use(ctx)
    if not _save_enabled(ctx.mode) or ctx.saved:
        return False
    if sync_saves():
        return _write_bundle(ctx, logger)
    try:
        with _worker_cv:
            if any(queued is ctx for queued, _ in _worker_queue):
                return False
            _worker_queue.append((ctx, logger))
            _start_worker_locked()
            _worker_cv.notify_all()
        return True
    except Exception as exc:  # noqa: BLE001 - best-effort, exactly like the write itself
        _warn(logger, f"compile-cache: could not queue background save: {exc}")
        return False


def wait_for_saves(timeout: float = _SAVE_JOIN_TIMEOUT) -> bool:
    """Block until the worker is idle. True if it drained, False on timeout."""
    deadline = time.monotonic() + timeout
    with _worker_cv:
        while _worker_queue or _worker_active is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _worker_cv.wait(remaining)
    return True


def restore(ctx: Optional[CacheContext], *, logger: Any = None) -> None:
    """Restore ``TORCHINDUCTOR_CACHE_DIR`` to its pre-load value. Call on unload."""
    if ctx is None:
        return
    _unregister_live(ctx.dir)
    if not ctx.prev_inductor_dir_set:
        return
    # Before the env var moves: a running save reads the inductor dir being repointed.
    if not wait_for_saves():
        _warn(
            logger,
            f"compile-cache: background save still running after {_SAVE_JOIN_TIMEOUT:.0f}s; continuing",
        )
    try:
        if ctx.prev_inductor_dir is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = ctx.prev_inductor_dir
    except Exception:  # noqa: BLE001
        pass


# Refcounted: a reload of the same model overlaps its predecessor's teardown.
_live_lock = threading.Lock()
_live_dirs: dict[str, int] = {}


def _register_live(cdir: Path) -> None:
    with _live_lock:
        _live_dirs[str(cdir)] = _live_dirs.get(str(cdir), 0) + 1


def _unregister_live(cdir: Path) -> None:
    with _live_lock:
        count = _live_dirs.get(str(cdir), 0) - 1
        if count > 0:
            _live_dirs[str(cdir)] = count
        else:
            _live_dirs.pop(str(cdir), None)


def note_use(ctx: Optional[CacheContext]) -> None:
    """Throttled touch keeping ``ctx``'s key out of other processes' eviction; call before a render too."""
    if ctx is not None and time.time() - ctx.last_touch > _TOUCH_INTERVAL_SECONDS:
        _touch_last_used(ctx)


def _touch_last_used(ctx: CacheContext) -> None:
    ctx.last_touch = time.time()
    try:
        path = ctx.dir / _LAST_USED_NAME
        path.touch(exist_ok = True)
        os.utime(path, None)
    except OSError:
        pass


def max_cache_bytes() -> Optional[int]:
    raw = (os.environ.get(_ENV_MAX_GB) or "").strip()
    try:
        gb = float(raw) if raw else _DEFAULT_MAX_GB
    except ValueError:
        gb = _DEFAULT_MAX_GB
    if not math.isfinite(gb * (1 << 30)):
        gb = _DEFAULT_MAX_GB
    if not gb > 0:
        return None
    return int(gb * (1 << 30))


def _is_key_dir(path: Path) -> bool:
    name = path.name
    return len(name) == 32 and all(c in "0123456789abcdef" for c in name)


def _dir_bytes(path: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks = False):
        for name in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total


def _last_used(path: Path) -> float:
    for name in (_LAST_USED_NAME, _MANIFEST_NAME):
        try:
            return (path / name).stat().st_mtime
        except OSError:
            continue
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _is_tombstone(path: Path) -> bool:
    name = path.name
    return name.endswith(_TOMBSTONE_SUFFIX) and name[32:33] == "." and _is_key_dir(Path(name[:32]))


def _remove_key_dir(path: Path) -> Optional[int]:
    """Atomic rename then delete, so a concurrent open sees a clean miss. Returns bytes left, None if not taken."""
    tomb = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}{_TOMBSTONE_SUFFIX}")
    try:
        os.rename(path, tomb)
    except OSError:
        return None
    shutil.rmtree(tomb, ignore_errors = True)
    return _dir_bytes(tomb) if tomb.exists() else 0


def _evict_soon(logger: Any) -> None:
    if sync_saves():
        evict(logger = logger)
        return
    try:
        threading.Thread(
            target = evict,
            kwargs = {"logger": logger},
            name = "unsloth-compile-cache-evict",
            daemon = True,
        ).start()
    except Exception:  # noqa: BLE001
        pass


def evict(
    *,
    root: Optional[Path] = None,
    max_bytes: Optional[int] = None,
    logger: Any = None,
) -> list[str]:
    """Delete LRU keys until the root fits its budget (a target: live and recent keys are spared). Never raises."""
    removed: list[str] = []
    try:
        root = root if root is not None else cache_root()
        budget = max_bytes if max_bytes is not None else max_cache_bytes()
        if budget is None or not root.is_dir():
            return removed
        entries = []
        leftover = 0
        for child in root.iterdir():
            if child.is_symlink() or not child.is_dir():
                continue
            if _is_tombstone(child):
                shutil.rmtree(child, ignore_errors = True)
                leftover += _dir_bytes(child) if child.exists() else 0
            elif _is_key_dir(child):
                entries.append((_last_used(child), child, _dir_bytes(child)))
        total = leftover + sum(size for _, _, size in entries)
        if total <= budget:
            return removed
        with _live_lock:
            live = set(_live_dirs)
        cutoff = time.time() - _EVICT_GRACE_SECONDS
        for used, path, size in sorted(entries, key = lambda e: e[0]):
            if total <= budget:
                break
            if str(path) in live or used > cutoff or _last_used(path) > cutoff:
                continue
            left = _remove_key_dir(path)
            if left is None:
                if not path.exists():
                    total -= size
                continue
            total -= size - left
            removed.append(path.name)
        if removed:
            _info(
                logger,
                f"compile-cache: evicted {len(removed)} least recently used key(s); "
                f"{total / (1 << 30):.2f} GB of {budget / (1 << 30):.2f} GB now in use",
            )
    except Exception as exc:  # noqa: BLE001 - housekeeping must never fail a save
        _warn(logger, f"compile-cache: eviction failed: {exc}")
    return removed


def _warn(logger: Any, msg: str) -> None:
    if logger is not None:
        logger.warning("diffusion.compile_cache: %s", msg)


def _info(logger: Any, msg: str) -> None:
    if logger is not None:
        logger.info("diffusion.compile_cache: %s", msg)
