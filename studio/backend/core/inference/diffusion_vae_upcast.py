# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""fp32 VAE runs that leave a 16-bit VAE's stored weights in place.

SDXL-class VAEs set ``force_upcast``: diffusers' decode and img2img ``prepare_latents`` (and Studio's own encode
helpers) run ``vae.to(torch.float32)`` and later ``vae.to(float16)``. Each cast replaces every parameter's storage, so
under whole-module CPU offload the kept host weights are lost: the VAE is never pinned (the cast lands before its first
onload), every onload is a per-tensor pageable copy of fp32 weights (twice the bytes), and every offload copies the
weights back. Here the upcast first onloads the VAE through its kept offload hook (pinned chunks, load dtype), casts on
the device, and the cast back puts the original tensor objects back instead of allocating new ones, so the offload
re-points them at the pinned host copy. Values are identical: an fp16 / bf16 -> fp32 cast is exact, and the fp32 math
is the same as the stock upcast. ComfyUI likewise never rewrites a VAE's stored weights for a higher compute dtype.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterable, Optional

# Marks the instance-level ``to`` below; the kept offload hook's pinned onload passes device moves through it.
DTYPE_ONLY_TO_ATTR = "_unsloth_dtype_only_to"
_KEEP_ATTR = "_unsloth_offload_keep_cpu"


def _first_param(module: Any) -> Any:
    try:
        return next(iter(module.parameters()), None)
    except Exception:  # noqa: BLE001 - duck-typed module
        return None


def _onload_through_kept_hook(owner: Any) -> None:
    """Onload a whole-module offloaded ``owner`` the way its next forward would (pinned chunks, load dtype), so the
    upcast happens on the device instead of converting the host weights."""
    hook = getattr(owner, "_hf_hook", None)
    if not getattr(hook, _KEEP_ATTR, False) or not callable(getattr(hook, "pre_forward", None)):
        return
    first = _first_param(owner)
    if first is None or first.device.type != "cpu":
        return
    try:
        hook.pre_forward(owner)
    except Exception:  # noqa: BLE001 - the forward's own hook onloads it as before
        pass


def _float_tensors(parts: Iterable[Any], dtype: Any) -> tuple[list, list]:
    params, buffers = [], []
    for part in parts:
        for p in part.parameters():
            if p.is_floating_point() and p.dtype != dtype:
                params.append(p)
        for module in part.modules():
            for name, b in list(getattr(module, "_buffers", {}).items()):
                if b is not None and b.is_floating_point() and b.dtype != dtype:
                    buffers.append((module, name))
    return params, buffers


def upcast_keeping_storage(
    owner: Any,
    parts: Optional[list] = None,
    dtype: Any = None,
) -> list:
    """Cast ``parts`` (default ``[owner]``) to ``dtype`` (fp32), remembering the tensors they held; returns the token
    for ``restore_storage``."""
    import torch

    dtype = torch.float32 if dtype is None else dtype
    parts = [owner] if parts is None else [p for p in parts if p is not None]
    _onload_through_kept_hook(owner)
    params, buffers = _float_tensors(parts, dtype)
    saved_params = [(p, p.data) for p in params]
    saved_buffers = [(m, name, m._buffers[name]) for m, name in buffers]
    for part in parts:
        # The class method: an instance-level ``to`` (ours, an offload arena's) only handles device moves.
        type(part).to(part, dtype = dtype)
    token: list = []
    for p, orig in saved_params:
        token.append(("p", p, orig, p.data_ptr()))
    for m, name, orig in saved_buffers:
        cast = m._buffers.get(name)
        token.append(("b", (m, name), orig, cast.data_ptr() if cast is not None else None))
    return token


def restore_storage(token: list) -> None:
    """Put back the tensors ``upcast_keeping_storage`` replaced. A tensor that has since moved (another device, a
    reload) is cast back to its load dtype instead, which is what the stock cast does."""
    for kind, ref, orig, cast_ptr in token:
        try:
            if kind == "p":
                p = ref
                if p.data_ptr() == cast_ptr and p.device == orig.device:
                    p.data = orig
                elif p.dtype != orig.dtype:
                    p.data = p.data.to(orig.dtype)
            else:
                m, name = ref
                current = m._buffers.get(name)
                if current is None:
                    continue
                if current.data_ptr() == cast_ptr and current.device == orig.device:
                    m._buffers[name] = orig
                elif current.dtype != orig.dtype:
                    m._buffers[name] = current.to(orig.dtype)
        except Exception:  # noqa: BLE001 - best effort; leaves that tensor as the cast made it
            pass
    token.clear()


@contextlib.contextmanager
def fp32_keeping_storage(owner: Any, parts: Optional[list] = None):
    """Run ``parts`` of ``owner`` in fp32 for the block, then restore the original tensors (see module docstring)."""
    token = upcast_keeping_storage(owner, parts)
    try:
        yield
    finally:
        restore_storage(token)


def _is_16bit(dtype: Any) -> bool:
    import torch
    return dtype in (torch.float16, torch.bfloat16)


@contextlib.contextmanager
def pipeline_upcasts_keep_storage(pipe: Any):
    """For one pipeline call: diffusers' ``vae.to(torch.float32)`` / ``vae.to(load dtype)`` round trip (SDXL decode,
    img2img ``prepare_latents``) goes through ``upcast_keeping_storage`` / ``restore_storage``. Anything else ``to``
    is asked (a device move, another dtype) runs unchanged."""
    vae = getattr(pipe, "vae", None)
    first = _first_param(vae) if vae is not None else None
    config = getattr(vae, "config", None)
    if (
        first is None
        or not _is_16bit(first.dtype)
        or not getattr(config, "force_upcast", False)
        or "to" in getattr(vae, "__dict__", {})
    ):
        yield
        return
    import torch

    load_dtype = first.dtype
    stock_to = type(vae).to
    state: dict = {"token": None}

    def to(*args: Any, **kwargs: Any) -> Any:
        try:
            device, dtype, _non_blocking, fmt = torch._C._nn._parse_to(*args, **kwargs)
        except Exception:  # noqa: BLE001 - not a tensor-style call: the stock method reports it
            return stock_to(vae, *args, **kwargs)
        if device is None and fmt is None and dtype is not None:
            current = getattr(_first_param(vae), "dtype", None)
            if dtype == torch.float32 and state["token"] is None and current == load_dtype:
                state["token"] = upcast_keeping_storage(vae)
                return vae
            if dtype == load_dtype and state["token"] is not None:
                token, state["token"] = state["token"], None
                restore_storage(token)
                return vae
        return stock_to(vae, *args, **kwargs)

    setattr(to, DTYPE_ONLY_TO_ATTR, True)
    vae.__dict__["to"] = to
    try:
        yield
    finally:
        if vae.__dict__.get("to") is to:
            vae.__dict__.pop("to", None)
        if state["token"] is not None:
            # A raise between the upcast and the cast back: leave the VAE in its load dtype, as the next call expects.
            restore_storage(state["token"])
