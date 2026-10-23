# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""fp32 VAE runs that leave a 16-bit VAE's stored weights in place."""

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
    """Onload via the kept hook (pinned chunks) so the upcast happens on device, not on the host weights."""
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
    """``(module, name)`` of every floating parameter and buffer not already in ``dtype``. Keyed by registration, not
    object: sequential offload's leaf hooks register a new Parameter on every onload and offload."""
    params, buffers, seen = [], [], set()
    for part in parts:
        for module in part.modules():
            for name, p in list(getattr(module, "_parameters", {}).items()):
                if (
                    p is not None
                    and p.is_floating_point()
                    and p.dtype != dtype
                    and id(p) not in seen
                ):
                    seen.add(id(p))
                    params.append((module, name))
            for name, b in list(getattr(module, "_buffers", {}).items()):
                if b is not None and b.is_floating_point() and b.dtype != dtype:
                    buffers.append((module, name))
    return params, buffers


def upcast_keeping_storage(
    owner: Any,
    parts: Optional[list] = None,
    dtype: Any = None,
) -> list:
    import torch

    dtype = torch.float32 if dtype is None else dtype
    parts = [owner] if parts is None else [p for p in parts if p is not None]
    _onload_through_kept_hook(owner)
    params, buffers = _float_tensors(parts, dtype)
    saved = [("p", m, name, m._parameters[name].data) for m, name in params]
    saved += [("b", m, name, m._buffers[name]) for m, name in buffers]
    for part in parts:
        # The class method: an instance-level ``to`` (ours, an offload arena's) only handles device moves.
        type(part).to(part, dtype = dtype)
    token: list = []
    for kind, m, name, orig in saved:
        cast = (m._parameters if kind == "p" else m._buffers).get(name)
        token.append((kind, m, name, orig, cast.data_ptr() if cast is not None else None))
    return token


def _restorable(current: Any, orig: Any, cast_ptr: Any) -> bool:
    """``current`` holds ``orig``'s values (diffusers < 0.37 ``upcast_vae`` casts some submodules back)."""
    import torch

    if current.device != orig.device or current.shape != orig.shape:
        return False
    if current.dtype != orig.dtype:
        return current.data_ptr() == cast_ptr
    if current.device.type == "meta" or current.data_ptr() == orig.data_ptr():
        return current.device.type == "meta"
    return bool(torch.equal(current, orig))


def restore_storage(token: list) -> None:
    """Put back the original tensors on the current registrations; one that has since moved or been changed is cast
    back to its load dtype instead, as the stock cast does."""
    for kind, m, name, orig, cast_ptr in token:
        try:
            table = m._parameters if kind == "p" else m._buffers
            current = table.get(name)
            if current is None:
                continue
            data = current.data if kind == "p" else current
            if _restorable(data, orig, cast_ptr):
                if kind == "p":
                    current.data = orig
                else:
                    table[name] = orig
            elif data.dtype != orig.dtype:
                if kind == "p":
                    current.data = data.to(orig.dtype)
                else:
                    table[name] = data.to(orig.dtype)
        except Exception:  # noqa: BLE001 - best effort; leaves that tensor as the cast made it
            pass
    token.clear()


@contextlib.contextmanager
def fp32_keeping_storage(owner: Any, parts: Optional[list] = None):
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
    """Route diffusers' fp32 / load-dtype round trip through ``upcast_keeping_storage``; other ``to`` calls run as is."""
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
