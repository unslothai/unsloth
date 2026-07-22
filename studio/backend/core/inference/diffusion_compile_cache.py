# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-warmed ``torch.compile`` cache for the diffusion denoiser (Mega-cache).

The regional compile of the repeated denoiser block pays a one-time 25-58s compile on
the FIRST image after a load. This lets that cost be paid ONCE (by the distributor ahead
of time or the user on first run) and reused on every later load via torch's portable
Mega-cache (``save_cache_artifacts`` / ``load_cache_artifacts``, torch >= 2.7).

PORTABILITY IS NOT UNIVERSAL: an artifact is only valid for the SAME torch/Triton/CUDA
build, GPU arch, and model graph (family, dtype, quant, attention backend, compile
kwargs, shape bucket). torch validates these on load; a mismatch yields no hit, not an
error. Hence an EXACT-MATCH fingerprint with SILENT FALLBACK to local compile (a miss is
normal), and per-arch bundles keyed by the full fingerprint. See
``outputs/compile_cache/DISTRIBUTION.md``.

GGUF loads participate too (fingerprinted ``quant="gguf"``, a different compiled graph
than the dense family): ``torch.compile`` mode="default" runs clean over diffusers'
GGUF dequant path (measured on torch 2.10 / diffusers 0.39), but the cold warmup is
HEAVY at batched shapes -- ~159 s first-pass wall on a 12B-class 4-step model at
batch 32, up to ~655 s on a 20B CFG-batched model at 1024px -- which is exactly the
cost this bundle amortises to once-ever. Batched generation registers every distinct
(width, height, batch) chunk shape it runs via ``register_shape`` so a static-compile
bundle grows to cover the batch sizes actually used, OOM-backoff halves included.

Lifecycle (around ``_compile_repeated_blocks``): ``begin`` builds the fingerprint, points
``TORCHINDUCTOR_CACHE_DIR`` at a per-key dir, and loads a matching bundle (before the
first compiled forward); ``save`` writes the bundle + manifest after the warmup forward
(on by default, a hit skips the rewrite); ``restore`` resets the inductor dir on unload.
All env-gated and best-effort; torch imported lazily.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------------- env knobs
# UNSLOTH_DIFFUSION_COMPILE_CACHE: auto (default) | 0 | 1
#   auto -> load a matching bundle AND save one after the first compiled generation.
#           Measured on Qwen-Image (B200, deferred 3rd-gen engage, FBCache armed): compile
#           hitch drops 29.1 -> 22.2 s, bit-identical output, 7.9 MB bundle, ~0.5 s save.
#           Residual warmup is dynamo tracing + guards, which Mega-cache does not capture.
#   1    -> same as auto, also re-saves on a hit (distributor refresh).
#   0    -> disabled (plain local compile, no cache dir override).
# UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR: root dir for bundles (default under the workspace).
# UNSLOTH_DIFFUSION_COMPILE_CACHE_SAVE: 0 disables the auto save (load-only); 1 keeps it.
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
    # auto: save by default (without a saved bundle no user gets a warm restart). SAVE
    # env overrides: "0" -> load-only, "1" -> keep on.
    return (os.environ.get(_ENV_SAVE) or "").strip().lower() not in ("0", "off", "false", "no")


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
    """HARD-portability dimensions: any difference invalidates a bundle.

    Mirrors what torch's inductor cache keys on (torch + triton + CUDA + GPU type) plus
    diffusers (the graph source), surfaced explicitly so the manifest is self-describing.
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
    """MODEL-graph dimensions that change the compiled artifact.

    Reads ``transformer``'s class name + ``_repeated_blocks`` so the key tracks exactly
    what gets compiled.
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
    payload = json.dumps({"env": env_fp, "model": model_fp}, sort_keys = True, default = str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# ----------------------------------------------------------------------------- lifecycle
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

    Returns a ``CacheContext`` for ``save``/``restore``, or ``None`` when disabled or torch
    lacks the Mega-cache API. Never raises.
    """
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

    # Isolate inductor's on-disk cache per key so bundles never cross-contaminate.
    try:
        cdir.mkdir(parents = True, exist_ok = True)
        ctx.prev_inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        ctx.prev_inductor_dir_set = True
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cdir / "inductor")
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"could not set TORCHINDUCTOR_CACHE_DIR: {exc}")

    # Try an exact-match load. A miss/mismatch is normal and non-fatal.
    if ctx.bundle.exists() and ctx.manifest_path.exists():
        ctx.hit = _try_load(ctx, logger)
        if ctx.hit and mode != "on":
            # Loaded artifacts == on-disk artifacts, so nothing to save (~0.5 s for no
            # change). A new static-compile shape re-dirties via register_shape; mode
            # "on" (distributor refresh) keeps saving.
            ctx.saved = True
    else:
        _info(logger, f"compile-cache: no bundle for key {key} (will compile locally)")
    return ctx


def register_shape(ctx: Optional[CacheContext], shape: Any, *, static: bool) -> None:
    """Record a generation's (width, height, batch) against the bundle coverage.

    Only meaningful for a STATIC compile: each new shape triggers its own compile, so the
    existing bundle lacks those artifacts -- clear ``saved`` so the next ``save`` rewrites
    the enriched set. Dynamic compiles reuse one artifact and never dirty the context.
    Never raises."""
    if ctx is None or not static:
        return
    try:
        key = tuple(shape)
        if key not in ctx.shapes:
            ctx.shapes.add(key)
            ctx.saved = False
    except Exception:  # noqa: BLE001 — bookkeeping only
        pass


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
        # Static-compile shapes this bundle covers (see register_shape).
        try:
            ctx.shapes = {tuple(s) for s in manifest.get("shapes", [])}
        except Exception:  # noqa: BLE001 — coverage bookkeeping only
            ctx.shapes = set()
        _info(logger, f"compile-cache: loaded bundle for key {ctx.key}")
        return True
    except Exception as exc:  # noqa: BLE001
        _warn(logger, f"compile-cache: load failed: {exc}")
        return False


def save(ctx: Optional[CacheContext], *, logger: Any = None) -> bool:
    """Persist compiled artifacts to the bundle + manifest, AFTER a warmup forward.

    No-op unless save is enabled and the context is dirty: a bundle HIT starts clean
    (rewriting the just-loaded artifacts costs ~0.5 s for no change); a new static-compile
    shape re-dirties via ``register_shape`` so the bundle grows to cover every shape used.
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
        ctx.dir.mkdir(parents = True, exist_ok = True)
        ctx.bundle.write_bytes(data)
        manifest = {
            "format": _FORMAT_VERSION,
            "key": ctx.key,
            "created": time.time(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "env": ctx.env_fp,
            "model": ctx.model_fp,
            # Static-compile shape coverage (register_shape); unused by dynamic compiles.
            "shapes": sorted(list(s) for s in ctx.shapes),
        }
        ctx.manifest_path.write_text(json.dumps(manifest, indent = 2, sort_keys = True, default = str))
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
