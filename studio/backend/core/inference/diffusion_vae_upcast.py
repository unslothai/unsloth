# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import contextlib
from typing import Any, Iterable, Optional

DTYPE_ONLY_TO_ATTR = "_unsloth_dtype_only_to"
_KEEP_ATTR = "_unsloth_offload_keep_cpu"


def _first_param(module: Any) -> Any:
    try:
        return next(iter(module.parameters()), None)
    except Exception:  # noqa: BLE001 - duck-typed module
        return None


def _onload_through_kept_hook(owner: Any) -> None:
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
    """Track registrations: sequential offload replaces Parameter objects."""
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
        # [kind, module, name, storage source, original, cast pointer, cast version]
        records.append([kind, m, name, how, ref, None, None])
        del current, data
    token: list = []
    try:
        # Every host-backed view must leave the shared device chunk before it can be freed.
        for rec in records:
            if rec[3] == "host":
                token.append(rec)
                _set(rec[0], rec[1], rec[2], rec[4][4])
                if rec[0] == "b":
                    rec[4][3]["buffer_version"][rec[4][2]] = None
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
            current = (m._parameters if kind == "p" else m._buffers)[name]
            rec[5], rec[6] = current.data_ptr(), _version(current)
            del current
    except BaseException:
        restore_storage(token)
        raise
    return token


def _restorable(current: Any, orig: Any, cast_ptr: Any) -> bool:
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
            _set(kind, m, name, kept.to(device, non_blocking = kept.is_pinned()))
            _mark_onloaded(state, kind, full, m, name)
        return
    current = (m._parameters if kind == "p" else m._buffers)[name]
    untouched = data.data_ptr() == cast_ptr and _version(current) == cast_version
    if cast_ptr is None:
        return
    if data.dtype != load_dtype:
        try:
            cast = data.to(load_dtype)
        except RuntimeError as exc:
            from .diffusion_vae_oom import is_oom
            if is_oom(exc) and untouched and cast_version is not None and how == "host":
                _set(kind, m, name, kept)
                rec[5], rec[6] = None, None
            raise
        _set(kind, m, name, cast)
        if untouched and how == "host":
            _mark_onloaded(state, kind, full, m, name)
    elif how == "host" and data.device.type != "cpu":
        if data.shape == kept.shape and bool(torch.equal(data, kept.to(data.device))):
            _mark_onloaded(state, kind, full, m, name)


def _release_restore_memory(token: list, exc: Optional[BaseException] = None) -> None:
    import torch
    import traceback

    errors, seen = [exc], set()
    while errors:
        error = errors.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        errors.extend((error.__cause__, error.__context__))
        errors.extend(getattr(error, "exceptions", ()))
    backends = dict.fromkeys(rec[4][0].type for rec in token if rec[3] in ("dev", "host"))
    for backend in backends:
        clear = getattr(getattr(torch, backend, None), "empty_cache", None)
        if callable(clear):
            try:
                clear()
            except Exception:  # noqa: BLE001
                pass


def restore_storage(token: list) -> None:
    import sys

    pending = list(reversed(token))
    exc = sys.exc_info()[1]
    if exc is not None:
        from .diffusion_vae_oom import is_oom
        if is_oom(exc):
            _release_restore_memory(pending, exc)
    while pending:
        failed = []
        released = False
        for rec in pending:
            cast_ptr = rec[5]
            try:
                _restore_one(rec)
            except Exception:  # noqa: BLE001 - retried below; a later cast back can free the room it needs
                failed.append(rec)
                released |= cast_ptr is not None and rec[5] is None
        if released:
            _release_restore_memory(failed)
        elif len(failed) == len(pending):
            break
        pending = failed
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
            restore_storage(state["token"])
