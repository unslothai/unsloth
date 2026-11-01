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


def _kept_state(owner: Any) -> Optional[dict]:
    """The kept offload hook's state on ``owner`` (diffusion_memory._wrap_cpu_offload_hook), else None."""
    if not getattr(getattr(owner, "_hf_hook", None), _KEEP_ATTR, False):
        return None
    state = getattr(owner, "__dict__", {}).get(_KEEP_ATTR)
    return state if isinstance(state, dict) else None


def _names(owner: Any) -> dict:
    try:
        return {id(m): (prefix + "." if prefix else "") for prefix, m in owner.named_modules()}
    except Exception:  # noqa: BLE001 - duck-typed owner: no hook state to keep in step
        return {}


def _host_backed(state: Optional[dict], kind: str, full: Optional[str], current: Any) -> Any:
    """The pinned host tensor ``current`` is an untouched onload of (the hook would re-point it on offload), else None."""
    if state is None or full is None or current.device.type in ("cpu", "meta"):
        return None
    import torch

    if kind == "p":
        kept, seen = state.get("host", {}).get(full), state.get("version", {}).get(full)
        data = current.data
        live = seen is not None and seen[0] is current and seen[1] == current._version
    else:
        kept, seen = (
            state.get("buffer_host", {}).get(full),
            state.get("buffer_version", {}).get(full),
        )
        data = current
        live = seen is not None and seen[0] is current and seen[1] == _version(current)
    if (
        not live
        or type(kept) is not torch.Tensor
        or type(data) is not torch.Tensor
        or seen[2] != data.data_ptr()
        or kept.device.type != "cpu"
        or kept.dtype != data.dtype
        or kept.shape != data.shape
        or kept.stride() != data.stride()
    ):
        return None
    return kept


def _version(t: Any) -> Optional[int]:
    try:
        return None if t.is_inference() else int(t._version)
    except Exception:  # noqa: BLE001
        return None


def _mark_onloaded(
    state: Optional[dict], kind: str, full: Optional[str], m: Any, name: str
) -> None:
    """Mark the cast-back tensor as the hook's untouched onload, so offload re-points it at the pinned host tensor."""
    if state is None or full is None:
        return
    table = state.get("version" if kind == "p" else "buffer_version")
    if not isinstance(table, dict) or full not in table:
        return
    current = (m._parameters if kind == "p" else m._buffers).get(name)
    if current is None:
        return
    if kind == "p":
        table[full] = (current, current._version, current.data_ptr())
    else:
        table[full] = (current, _version(current), current.data_ptr())


def _set(kind: str, m: Any, name: str, tensor: Any) -> None:
    if kind == "p":
        m._parameters[name].data = tensor
    else:
        m._buffers[name] = tensor


def _get(kind: str, m: Any, name: str) -> Any:
    current = (m._parameters if kind == "p" else m._buffers).get(name)
    if current is None:
        return None
    return current.data if kind == "p" else current


def upcast_keeping_storage(
    owner: Any,
    parts: Optional[list] = None,
    dtype: Any = None,
) -> list:
    """Cast ``parts`` to ``dtype`` one tensor at a time (never both copies on device); host tensors are kept for
    ``restore_storage``. Kept-hook onloads are re-pointed at their pinned host tensor first, freeing the shared chunk."""
    import torch

    dtype = torch.float32 if dtype is None else dtype
    parts = [owner] if parts is None else [p for p in parts if p is not None]
    _onload_through_kept_hook(owner)
    params, buffers = _float_tensors(parts, dtype)
    state = _kept_state(owner)
    prefixes = _names(owner) if state is not None else {}
    records: list = []
    for kind, m, name in [("p", m, n) for m, n in params] + [("b", m, n) for m, n in buffers]:
        current = (m._parameters if kind == "p" else m._buffers)[name]
        data = current.data if kind == "p" else current
        prefix = prefixes.get(id(m))
        full = None if prefix is None else prefix + name
        kept = _host_backed(state, kind, full, current)
        if kept is not None:
            how, ref = "host", (data.device, data.dtype, full, state, kept)
        elif data.device.type in ("cpu", "meta"):
            how, ref = "orig", data
        else:
            how, ref = "dev", (data.device, data.dtype, full, state, None)
        # [kind, module, name, how, ref, cast data_ptr, cast _version]; the cast fields stay None until it lands.
        records.append([kind, m, name, how, ref, None, None])
        del current, data
    token: list = []
    try:
        # Host-backed weights first, all of them: they share one device chunk, which frees only once none points into it.
        for rec in records:
            if rec[3] == "host":
                token.append(rec)
                _set(rec[0], rec[1], rec[2], rec[4][4])
        for rec in records:
            kind, m, name, how, ref = rec[:5]
            if how != "host":
                token.append(rec)
            data = _get(kind, m, name)
            device = data.device if how == "orig" else ref[0]
            cast = data.to(device = device, dtype = dtype, non_blocking = data.is_pinned())
            del data
            _set(kind, m, name, cast)
            del cast
            current = _get(kind, m, name)
            rec[5], rec[6] = current.data_ptr(), _version(current)
            del current
    except BaseException:
        # A failed upcast (OOM) puts every tensor back in its original dtype before the error propagates.
        restore_storage(token)
        raise
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


def _restore_one(rec: list) -> None:
    import torch

    kind, m, name, how, ref, cast_ptr, cast_version = rec
    data = _get(kind, m, name)
    if data is None:
        return
    if how == "orig":
        if _restorable(data, ref, cast_ptr):
            _set(kind, m, name, ref)
        elif data.dtype != ref.dtype:
            _set(kind, m, name, data.to(ref.dtype))
        return
    device, load_dtype, full, state, kept = ref
    if how == "host" and rec[5] is None:
        if data.data_ptr() == kept.data_ptr():
            # Never cast: back on the device, as the hook's onload of the pinned host tensor.
            _set(kind, m, name, kept.to(device, non_blocking = kept.is_pinned()))
            _mark_onloaded(state, kind, full, m, name)
        return
    untouched = data.data_ptr() == cast_ptr and _version(data) == cast_version
    if cast_ptr is None:
        # Never cast (a "dev" tensor the failed upcast had not reached): still the original.
        return
    if data.dtype != load_dtype:
        _set(kind, m, name, data.to(load_dtype))
        if untouched and how == "host":
            # fp16 / bf16 -> fp32 -> back is exact, so this is the pinned host tensor's onload again.
            _mark_onloaded(state, kind, full, m, name)
    elif how == "host" and data.device.type != "cpu":
        # Cast back meanwhile by a submodule ``to`` (diffusers < 0.37 ``upcast_vae``): still the host tensor's values?
        if data.shape == kept.shape and bool(torch.equal(data, kept.to(data.device))):
            _mark_onloaded(state, kind, full, m, name)


def restore_storage(token: list) -> None:
    """Put back the original dtype: a kept host tensor itself, a device tensor by an exact cast back; one since moved
    or changed is cast back to its load dtype, as the stock cast does."""
    for rec in token:
        try:
            _restore_one(rec)
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
