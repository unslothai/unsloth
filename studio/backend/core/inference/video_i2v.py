# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Serve a first-frame request through a family's image-to-video pipeline, over the loaded one.

A video family's ``pipeline_class`` is its text-to-video pipeline (WanPipeline, LTX2Pipeline). The
image-to-video class for the same checkpoint (WanImageToVideoPipeline, LTX2ImageToVideoPipeline)
takes the very same components, so a first-frame request must not load anything: it runs the I2V
class's ``__call__`` against the pipeline Studio already built, placed, quantised and compiled.

diffusers' ``DiffusionPipeline.from_pipe`` is the obvious tool and the wrong one here: it ends with
``new_pipeline.to(dtype=...)``, defaulting to float32, which would upcast every shared module
(and undo Wan's fp32-VAE / bf16-DiT split, GGUF and torchao layers, offload hooks). It also builds
a second pipeline object, so everything Studio set on the loaded pipeline instance (the prompt
cache's ``encode_prompt``, CUDA-graph bookkeeping, pinning requests) would be missing from it, and
anything the render records on it would be lost.

So instead the I2V class is constructed once over the loaded components purely to learn the
instance state its ``__init__`` sets (its config, its own ``video_processor``, the optional
``image_encoder`` / ``image_processor`` slots), and for the duration of one render the loaded
pipeline takes that class and that state. Afterwards both are put back exactly. Construction moves
no weights; the swap allocates nothing; text-to-video calls never enter this module.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import threading
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

_MISSING = object()
# One swap at a time per process: a pipeline object is shared state, and the generate lock that already serialises
# renders of one backend does not cover a second backend instance holding the same object.
_SWAP_LOCK = threading.RLock()


def resolve_i2v_pipeline_class(name: str) -> type:
    """The diffusers class named by a family's ``i2v_pipeline_class``."""
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
    """Whether this install's diffusers exports ``name``. Never raises: status and request
    validation ask it, and a missing class must read as "no first frame", not as a 500."""
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
    """Constructor kwargs for ``cls`` from ``pipe``'s registered components and config values."""
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


def i2v_instance_state(pipe: Any, cls: type, config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The instance attributes ``cls.__init__`` sets over ``pipe``'s components that differ from
    ``pipe``'s own: its config, processors and empty optional slots. The shared components come
    back as the identical objects, so they are not part of the overlay."""
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
    """``pipe`` running as ``class_name`` for one render, then exactly as it was.

    A no-op when ``class_name`` is empty or ``pipe`` already is that class. Attributes the render
    sets that are not part of the I2V overlay (step counters, CUDA-graph bookkeeping, a swapped
    scheduler) stay on ``pipe``, as they would after a text-to-video render.
    """
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
            yield pipe
        finally:
            pipe.__class__ = original_cls
            _carry_component_registrations(pipe, overlay, saved)
            _restore(pipe, saved)


def _carry_component_registrations(pipe: Any, overlay: dict[str, Any], saved: dict[str, Any]) -> None:
    """A component assigned during the render (``pipe.vae = ...``) re-registers itself in the
    pipeline config through ``DiffusionPipeline.__setattr__``, which, mid-swap, is the overlaid
    config. Copy such entries back onto the config being restored so it keeps naming the
    components the pipeline actually holds."""
    live = pipe.__dict__.get("_internal_dict")
    template = overlay.get("_internal_dict")
    original = saved.get("_internal_dict", _MISSING)
    if live is None or live is template or original is _MISSING or original is None:
        return
    try:
        changed = {
            k: v
            for k, v in dict(live).items()
            if k in original and (template is None or template.get(k, _MISSING) != v) and original[k] != v
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
