# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-warmed ``torch.compile`` cache for the diffusion denoiser (Mega-cache).

The regional ``torch.compile`` of the repeated denoiser block (``diffusion_speed.py``)
pays a one-time 25-58s compile on the FIRST image after a load. This module lets that
cost be paid ONCE -- by us (the distributor) ahead of time, or by the user on a first
run -- and reused on every later load via torch's portable Mega-cache
(``torch.compiler.save_cache_artifacts`` / ``load_cache_artifacts``, torch >= 2.7).

PORTABILITY IS NOT UNIVERSAL. A compiled artifact is only valid for the SAME torch
version, Triton version, CUDA build, and GPU architecture it was produced on (and the
same model graph: family, dtype, quant scheme, attention backend, compile kwargs, shape
bucket). torch validates these on load and a mismatch simply yields no cache hit -- it
does NOT error. So this layer is built around an EXACT-MATCH fingerprint with a SILENT
FALLBACK to local compile: a miss is normal and never fatal. We therefore ship per-arch
bundles keyed by the full fingerprint, never one universal cache. See
``outputs/compile_cache/DISTRIBUTION.md``.

Lifecycle (driven by the caller, around ``_compile_repeated_blocks``):
  1. ``begin(...)``    -> build the fingerprint, point ``TORCHINDUCTOR_CACHE_DIR`` at a
                          per-key dir, and ``load_cache_artifacts`` if a matching bundle
                          exists. Must run BEFORE the first compiled forward.
  2. (compile + one warmup forward happen as usual; on a hit they reuse the cache.)
  3. ``save(...)``     -> ``save_cache_artifacts`` to the bundle + manifest, AFTER the
                          warmup forward, when in distributor/save mode.
  4. ``restore(...)``  -> put ``TORCHINDUCTOR_CACHE_DIR`` back on unload.

Everything is env-gated and best-effort; torch is imported lazily.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

# ----------------------------------------------------------------------------- env knobs
# UNSLOTH_DIFFUSION_COMPILE_CACHE: auto (default) | 0 | 1
#   auto -> load a matching bundle if present (no automatic save).
#   1    -> load AND save (distributor / first-run warm).
#   0    -> disabled (plain local compile, no cache dir override).
# UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR: root dir for bundles (default under the workspace).
# UNSLOTH_DIFFUSION_COMPILE_CACHE_SAVE: 1 -> force-enable save even in "auto".
_ENV_MODE = "UNSLOTH_DIFFUSION_COMPILE_CACHE"
_ENV_DIR = "UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR"
_ENV_SAVE = "UNSLOTH_DIFFUSION_COMPILE_CACHE_SAVE"

_DEFAULT_ROOT = Path.home() / ".cache" / "unsloth" / "diffusion_compile_cache"

_MANIFEST_NAME = "manifest.json"
_BUNDLE_NAME = "cache.bin"
_FORMAT_VERSION = 1


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
    # auto: save only if explicitly opted in.
    return (os.environ.get(_ENV_SAVE) or "").strip().lower() in ("1", "on", "true", "yes")


def cache_root() -> Path:
    root = os.environ.get(_ENV_DIR)
    return Path(root) if root else _DEFAULT_ROOT


# --------------------------------------------------------------------------- fingerprint
def _triton_version() -> Optional[str]:
    try:
        import triton  # noqa: PLC0415
        return str(getattr(triton, "__version__", None))
    except Exception:  # noqa: BLE001 — triton optional
        return None


def _diffusers_version() -> Optional[str]:
    try:
        import diffusers  # noqa: PLC0415
        return str(getattr(diffusers, "__version__", None))
    except Exception:  # noqa: BLE001
        return None


def environment_fingerprint() -> dict[str, Any]:
    """The HARD-portability dimensions: any difference here invalidates a bundle.

    These mirror what torch's inductor cache itself keys on (torch + triton + CUDA +
    GPU type), plus diffusers (the graph source). We surface them explicitly so the
    manifest is self-describing and a mismatch is obvious to a human, not just to torch.
    """
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
            fp["gpu_name"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            fp["gpu_capability"] = f"sm_{cap[0]}{cap[1]}"
    except Exception:  # noqa: BLE001 — best-effort
        pass
    return fp


def model_fingerprint(
    *,
    family: Any,
    transformer: Any,
    dtype: Any,
    quant: Any,
    attention_backend: Any,
    compile_kwargs: dict[str, Any],
    shape_bucket: Any = None,
) -> dict[str, Any]:
    """The MODEL-graph dimensions that change the compiled artifact.

    ``family`` is the Unsloth family name; ``transformer`` is the live module (we read
    its class name + ``_repeated_blocks`` so the key tracks exactly what gets compiled).
    """
    blocks = list(getattr(transformer, "_repeated_blocks", []) or [])
    return {
        "family": str(family),
        "transformer_cls": type(transformer).__name__ if transformer is not None else None,
        "repeated_blocks": sorted(str(b) for b in blocks),
        "dtype": str(dtype),
        "quant": str(quant) if quant is not None else "none",
        "attention_backend": str(attention_backend) if attention_backend is not None else "default",
        "compile_kwargs": {k: compile_kwargs[k] for k in sorted(compile_kwargs)},
        "shape_bucket": shape_bucket,
    }


def cache_key(env_fp: dict[str, Any], model_fp: dict[str, Any]) -> str:
    payload = json.dumps({"env": env_fp, "model": model_fp}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# ----------------------------------------------------------------------------- lifecycle
@dataclasses.dataclass
class CacheContext:
    """Carries the per-load cache state between ``begin`` and ``save``/``restore``."""

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
) -> Optional[CacheContext]:
    """Point inductor at a per-key dir and load a matching bundle, BEFORE compile.

    Returns a ``CacheContext`` to pass to ``save``/``restore``, or ``None`` when the
    cache is disabled or torch lacks the Mega-cache API. Never raises.
    """
    mode = cache_mode()
    if mode == "off":
        return None
    try:
        import torch  # noqa: PLC0415
        if not (hasattr(torch.compiler, "save_cache_artifacts")
                and hasattr(torch.compiler, "load_cache_artifacts")):
            _warn(logger, "Mega-cache API unavailable (need torch >= 2.7); skipping")
            return None
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"torch import failed: {exc}")
        return None

    env_fp = environment_fingerprint()
    model_fp = model_fingerprint(
        family=family, transformer=transformer, dtype=dtype, quant=quant,
        attention_backend=attention_backend, compile_kwargs=compile_kwargs,
        shape_bucket=shape_bucket,
    )
    key = cache_key(env_fp, model_fp)
    cdir = cache_root() / key
    ctx = CacheContext(
        key=key, dir=cdir, bundle=cdir / _BUNDLE_NAME,
        manifest_path=cdir / _MANIFEST_NAME, env_fp=env_fp, model_fp=model_fp, mode=mode,
    )

    # Isolate inductor's on-disk cache per key so bundles never cross-contaminate.
    try:
        cdir.mkdir(parents=True, exist_ok=True)
        ctx.prev_inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        ctx.prev_inductor_dir_set = True
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cdir / "inductor")
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"could not set TORCHINDUCTOR_CACHE_DIR: {exc}")

    # Try an exact-match load. A miss/mismatch is normal and non-fatal.
    if ctx.bundle.exists() and ctx.manifest_path.exists():
        ctx.hit = _try_load(ctx, logger)
    else:
        _info(logger, f"compile-cache: no bundle for key {key} (will compile locally)")
    return ctx


def _try_load(ctx: CacheContext, logger: Any) -> bool:
    try:
        manifest = json.loads(ctx.manifest_path.read_text())
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: unreadable manifest: {exc}")
        return False

    # Exact-match guard (defence in depth: torch also validates internally on load).
    if manifest.get("env") != ctx.env_fp or manifest.get("model") != ctx.model_fp:
        _warn(logger, "compile-cache: fingerprint mismatch; falling back to local compile")
        return False

    try:
        data = ctx.bundle.read_bytes()
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: cannot read bundle: {exc}")
        return False

    # Integrity check (corruption / truncation; not a security signature).
    digest = hashlib.sha256(data).hexdigest()
    if manifest.get("sha256") and manifest["sha256"] != digest:
        _warn(logger, "compile-cache: bundle checksum mismatch; ignoring")
        return False

    try:
        import torch  # noqa: PLC0415
        info = torch.compiler.load_cache_artifacts(data)
        if info is None:
            _warn(logger, "compile-cache: load_cache_artifacts returned None (no hit)")
            return False
        _info(logger, f"compile-cache: loaded bundle for key {ctx.key}")
        return True
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: load failed: {exc}")
        return False


def save(ctx: Optional[CacheContext], *, logger: Any = None) -> bool:
    """Persist the compiled artifacts to the bundle + manifest, AFTER a warmup forward.

    No-op unless save is enabled (mode ``on`` or ``UNSLOTH_DIFFUSION_COMPILE_CACHE_SAVE``).
    Returns True if a bundle was written.
    """
    if ctx is None or not _save_enabled(ctx.mode) or ctx.saved:
        return False
    try:
        import torch  # noqa: PLC0415
        result = torch.compiler.save_cache_artifacts()
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: save_cache_artifacts failed: {exc}")
        return False
    if not result or result[0] is None:
        _warn(logger, "compile-cache: nothing to save (empty artifacts)")
        return False

    data = result[0]
    try:
        ctx.dir.mkdir(parents=True, exist_ok=True)
        ctx.bundle.write_bytes(data)
        manifest = {
            "format": _FORMAT_VERSION,
            "key": ctx.key,
            "created": time.time(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "env": ctx.env_fp,
            "model": ctx.model_fp,
        }
        ctx.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        ctx.saved = True
        _info(logger, f"compile-cache: saved bundle ({len(data)} bytes) for key {ctx.key}")
        return True
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: could not write bundle: {exc}")
        return False


def restore(ctx: Optional[CacheContext]) -> None:
    """Restore ``TORCHINDUCTOR_CACHE_DIR`` to its pre-load value. Call on unload."""
    if ctx is None or not ctx.prev_inductor_dir_set:
        return
    try:
        if ctx.prev_inductor_dir is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = ctx.prev_inductor_dir
    except Exception:  # noqa: BLE001
        pass


def _warn(logger: Any, msg: str) -> None:
    if logger is not None:
        logger.warning("diffusion.compile_cache: %s", msg)


def _info(logger: Any, msg: str) -> None:
    if logger is not None:
        logger.info("diffusion.compile_cache: %s", msg)
