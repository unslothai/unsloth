# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run a first-frame render by swapping the loaded pipeline's class to its image-to-video class.

Not ``from_pipe``: it casts every shared module to float32 (undoing fp32-VAE / bf16-DiT, GGUF,
torchao, offload hooks) and builds a second object lacking what Studio set on the loaded one.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import threading
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

_MISSING = object()
# The per-backend generate lock does not cover a second backend holding the same pipeline.
_SWAP_LOCK = threading.RLock()


def resolve_i2v_pipeline_class(name: str) -> type:
    import diffusers

    from .ltx2_import_compat import ensure_ltx2_pipelines_importable, is_ltx2_pipeline_class

    if is_ltx2_pipeline_class(name):
        ensure_ltx2_pipelines_importable(logger)
    cls = getattr(diffusers, name, None)
    if cls is None:
        raise ValueError(
            f"This diffusers build has no {name}, which image-to-video for this model needs. "
            "Update Unsloth to get a newer diffusers."
        )
    return cls


def i2v_pipeline_available(name: Optional[str]) -> bool:
    """Never raises: a missing class must read as "no first frame", not a 500."""
    if not name:
        return False
    try:
        import diffusers
    except Exception:  # noqa: BLE001
        return False
    key = (id(diffusers), name)
    cached = _AVAILABLE.get(key)
    if cached is None:
        try:
            resolve_i2v_pipeline_class(name)
            cached = True
        except Exception:  # noqa: BLE001 -- an absent or unimportable class is simply unavailable
            cached = False
        _AVAILABLE[key] = cached
    return cached


_AVAILABLE: dict[tuple[int, str], bool] = {}


def _init_kwargs(pipe: Any, cls: type, config_overrides: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(cls.__init__).parameters
    components = dict(getattr(pipe, "components", {}) or {})
    config = dict(getattr(pipe, "config", {}) or {})
    kwargs: dict[str, Any] = {}
    missing = []
    for name, param in params.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in config_overrides:
            kwargs[name] = config_overrides[name]
        elif name in components:
            kwargs[name] = components[name]
        elif name in config and not name.startswith("_"):
            kwargs[name] = config[name]
        elif param.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        raise ValueError(
            f"The loaded {type(pipe).__name__} has no {', '.join(sorted(missing))}, which "
            f"{cls.__name__} needs, so this model cannot take a first frame."
        )
    return kwargs


def i2v_instance_state(
    pipe: Any,
    cls: type,
    config_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    template = cls(**_init_kwargs(pipe, cls, dict(config_overrides or {})))
    own = pipe.__dict__
    return {k: v for k, v in template.__dict__.items() if own.get(k, _MISSING) is not v}


@contextlib.contextmanager
def first_frame_pipeline(
    pipe: Any,
    class_name: Optional[str],
    *,
    config_overrides: Optional[dict[str, Any]] = None,
) -> Iterator[Any]:
    if not class_name:
        yield pipe
        return
    cls = resolve_i2v_pipeline_class(class_name)
    if isinstance(pipe, cls):
        yield pipe
        return
    original_cls = type(pipe)
    with _SWAP_LOCK:
        overlay = i2v_instance_state(pipe, cls, config_overrides)
        saved = {k: pipe.__dict__.get(k, _MISSING) for k in overlay}
        pipe.__dict__.update(overlay)
        try:
            pipe.__class__ = cls
        except TypeError:
            _restore(pipe, saved)
            raise
        try:
            with _offload_vae_after_encode(pipe):
                yield pipe
        finally:
            pipe.__class__ = original_cls
            _carry_component_registrations(pipe, overlay, saved)
            _restore(pipe, saved)


@contextlib.contextmanager
def _offload_vae_after_encode(pipe: Any) -> Iterator[None]:
    """Model offload chains vae after the transformer (the t2v order), so the first-frame encode would leave the VAE
    resident through denoising; send it back once the encode returns."""
    vae = getattr(pipe, "vae", None)
    encode = getattr(vae, "encode", None)
    if not getattr(pipe, "_all_hooks", None) or not callable(encode):
        yield
        return
    try:
        from accelerate.hooks import CpuOffload
    except Exception:  # noqa: BLE001 -- no accelerate, no offload hooks
        yield
        return
    own = vae.__dict__.get("encode", _MISSING)

    def _encode(*args: Any, **kwargs: Any) -> Any:
        out = encode(*args, **kwargs)
        hook = getattr(vae, "_hf_hook", None)
        if isinstance(hook, CpuOffload):
            hook.init_hook(vae)
        return out

    vae.__dict__["encode"] = _encode
    try:
        yield
    finally:
        if vae.__dict__.get("encode") is _encode:
            if own is _MISSING:
                del vae.__dict__["encode"]
            else:
                vae.__dict__["encode"] = own


def _carry_component_registrations(
    pipe: Any, overlay: dict[str, Any], saved: dict[str, Any]
) -> None:
    """Components assigned mid-render register into the overlaid config; copy them to the restored one."""
    live = pipe.__dict__.get("_internal_dict")
    template = overlay.get("_internal_dict")
    original = saved.get("_internal_dict", _MISSING)
    if live is None or live is template or original is _MISSING or original is None:
        return
    try:
        changed = {
            k: v
            for k, v in dict(live).items()
            if k in original
            and (template is None or template.get(k, _MISSING) != v)
            and original[k] != v
        }
        if changed:
            saved["_internal_dict"] = type(original)({**dict(original), **changed})
    except Exception:  # noqa: BLE001 -- bookkeeping only; never fail the render over it
        logger.debug("video_i2v: could not carry component registrations back", exc_info = True)


def _restore(pipe: Any, saved: dict[str, Any]) -> None:
    for key, value in saved.items():
        if value is _MISSING:
            pipe.__dict__.pop(key, None)
        else:
            pipe.__dict__[key] = value
