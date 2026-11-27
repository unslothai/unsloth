# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-process cache of the text-encoder dead-tail probe verdict (diffusion_te_truncate).

The probe runs several encode_prompt forwards on the host before placement (~1-7 s per load). Its verdict is
structural, so it is stored under the Studio cache keyed by everything it can depend on: the pipeline class and
the code of every method it can call (instance-level patches included), the encoder classes, their configs,
module classes and forward code, parameter names / shapes / dtypes plus a sample of the weights, the model path,
the family, the library versions and this probe's own sources. Any change is a different key, so a stale verdict
is never read; a miss, an unkeyable pipeline or an unreadable file probes as before.
UNSLOTH_DIFFUSION_TE_TRUNCATE_CACHE=0 turns it off."""

from __future__ import annotations

import hashlib
import json
import marshal
import os
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Optional

ENV = "UNSLOTH_DIFFUSION_TE_TRUNCATE_CACHE"
_FILE_NAME = "diffusion_te_truncate.json"
_FORMAT = 1
_MAX_ENTRIES = 64
# Edits to these change what the probe computes or how a verdict is applied.
_SOURCES = (
    "diffusion_te_truncate.py",
    "diffusion_te_truncate_cache.py",
    "diffusion_text_encoder_trim.py",
)
# Env that can change the encoder forward the probe runs.
_ENV_PREFIXES = ("UNSLOTH_DIFFUSION_TE", "UNSLOTH_FUSED_FORWARD")
_ENV_IGNORED = frozenset({ENV})
_SIMPLE = (bool, int, float, str, bytes, type(None))
_LOCK = threading.Lock()
_SOURCE_DIGEST: list = []


def enabled() -> bool:
    return (os.environ.get(ENV) or "").strip().lower() not in ("0", "false", "no", "off")


def _cache_file() -> Optional[Path]:
    try:
        from utils.paths.storage_roots import cache_root
        return Path(cache_root()) / _FILE_NAME
    except Exception:  # noqa: BLE001 - no resolvable root is a miss
        return None


def _source_digest() -> str:
    if not _SOURCE_DIGEST:
        h = hashlib.sha256()
        here = Path(__file__).resolve().parent
        for name in _SOURCES:
            h.update(name.encode())
            try:
                h.update((here / name).read_bytes())
            except OSError:
                h.update(b"<missing>")
        _SOURCE_DIGEST.append(h.hexdigest())
    return _SOURCE_DIGEST[0]


def _version(dist: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:  # noqa: BLE001
        return None


class _Unkeyable(Exception):
    """Something the verdict may depend on cannot be identified: never cached."""


def _code_digest(code: types.CodeType, h: Any) -> None:
    # marshal covers bytecode, constants (nested code included), names and the file / line table.
    h.update(marshal.dumps(code))


def _callable_digest(
    fn: Any,
    h: Any,
    depth: int = 0,
) -> None:
    """Code of ``fn`` and of everything it wraps, plus simple closure values; refuses what it cannot see into."""
    if depth > 8:
        raise _Unkeyable("wrapper chain too deep")
    if isinstance(fn, (staticmethod, classmethod)):
        fn = fn.__func__
    if isinstance(fn, property):
        for part in (fn.fget, fn.fset, fn.fdel):
            if part is not None:
                _callable_digest(part, h, depth + 1)
        return
    if isinstance(fn, types.MethodType):
        _callable_digest(fn.__func__, h, depth + 1)
        return
    if isinstance(fn, types.FunctionType):
        h.update(f"{fn.__module__}.{fn.__qualname__}".encode())
        _code_digest(fn.__code__, h)
        h.update(repr(fn.__defaults__).encode() if _simple(fn.__defaults__) else b"<defaults>")
        for cell in fn.__closure__ or ():
            try:
                value = cell.cell_contents
            except ValueError:  # an empty cell
                h.update(b"<empty>")
                continue
            if callable(value) and not isinstance(value, type):
                _callable_digest(value, h, depth + 1)
            elif _simple(value):
                h.update(repr(value).encode())
            else:
                h.update(type(value).__qualname__.encode())
        wrapped = getattr(fn, "__wrapped__", None)
        if wrapped is not None:
            _callable_digest(wrapped, h, depth + 1)
        return
    if isinstance(
        fn, (types.BuiltinFunctionType, types.MethodWrapperType, types.WrapperDescriptorType)
    ):
        h.update(repr(fn).encode())
        return
    if isinstance(fn, type):
        h.update(f"{fn.__module__}.{fn.__qualname__}".encode())
        return
    call = getattr(type(fn), "__call__", None)
    if isinstance(call, types.FunctionType):
        h.update(type(fn).__qualname__.encode())
        _callable_digest(call, h, depth + 1)
        return
    raise _Unkeyable(f"opaque callable {type(fn).__name__}")


def _simple(value: Any) -> bool:
    if isinstance(value, _SIMPLE):
        return True
    # Not sets: their repr order follows the per-process string hash.
    if isinstance(value, (tuple, list)):
        return all(_simple(v) for v in value)
    return False


def _class_digest(
    cls: type,
    h: Any,
    *,
    stop: tuple = (),
) -> None:
    """Every function, property and wrapper defined on ``cls`` and its bases (up to ``stop``)."""
    for klass in cls.__mro__:
        if klass is object or klass in stop:
            continue
        h.update(f"{klass.__module__}.{klass.__qualname__}".encode())
        for name in sorted(klass.__dict__):
            value = klass.__dict__[name]
            if isinstance(value, (types.FunctionType, staticmethod, classmethod, property)):
                h.update(name.encode())
                try:
                    _callable_digest(value, h)
                except _Unkeyable:
                    h.update(b"<opaque>")


def _instance_digest(
    obj: Any,
    h: Any,
    *,
    skip: tuple = (),
) -> None:
    """Instance attributes that can steer the code: callables by code, simple values by repr, the rest by type."""
    import torch
    for name in sorted(getattr(obj, "__dict__", {})):
        if name in skip:
            continue
        value = obj.__dict__[name]
        h.update(name.encode())
        if isinstance(value, (torch.nn.Module, torch.Tensor)):
            h.update(type(value).__qualname__.encode())
        elif _simple(value):
            h.update(repr(value).encode())
        elif callable(value) and not isinstance(value, type):
            _callable_digest(value, h)
        else:
            h.update(type(value).__qualname__.encode())


def _config_digest(config: Any) -> str:
    if config is None:
        return "none"
    to_json = getattr(config, "to_json_string", None)
    if callable(to_json):
        try:
            return hashlib.sha256(to_json(use_diff = False).encode()).hexdigest()
        except TypeError:
            return hashlib.sha256(to_json().encode()).hexdigest()
    return hashlib.sha256(
        json.dumps(dict(config), sort_keys = True, default = str).encode()
    ).hexdigest()


def _weights_digest(module: Any, h: Any) -> None:
    """Names, shapes, dtypes and the first few values of every parameter and buffer: views only, never a full copy."""
    import torch
    with torch.no_grad():
        for kind, items in (("p", module.named_parameters()), ("b", module.named_buffers())):
            for name, tensor in items:
                if tensor is None:
                    continue
                if tensor.device.type == "meta":
                    raise _Unkeyable("meta tensor")
                h.update(f"{kind}:{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
                if tensor.numel() == 0:
                    continue
                sample = tensor.detach()
                while sample.ndim > 1:
                    sample = sample[0]
                sample = sample.reshape(-1)[:8]
                if not (
                    sample.is_floating_point()
                    or sample.dtype in (torch.int8, torch.uint8, torch.int32, torch.int64)
                ):
                    raise _Unkeyable(f"unsampled dtype {tensor.dtype}")
                h.update(repr(sample.float().cpu().tolist()).encode())


def fingerprint(
    pipe: Any,
    *,
    family: Optional[str] = None,
    encoder_attrs: tuple = (),
) -> Optional[dict]:
    """Everything the probe's verdict on ``pipe`` depends on, or None when any of it cannot be identified."""
    try:
        import torch

        cls = type(pipe)
        if "<locals>" in cls.__qualname__:
            return None  # a function-local stand-in (e.g. a worker's probe shim): its behaviour lives in closures
        pipe_h = hashlib.sha256()
        _class_digest(cls, pipe_h)
        _instance_digest(pipe, pipe_h, skip = encoder_attrs)
        config = getattr(pipe, "config", None)
        encoders: dict = {}
        for attr in encoder_attrs:
            encoder = getattr(pipe, attr, None)
            if not isinstance(encoder, torch.nn.Module):
                continue
            classes = sorted(
                {type(m) for m in encoder.modules()},
                key = lambda c: f"{c.__module__}.{c.__qualname__}",
            )
            code_h = hashlib.sha256()
            for klass in classes:
                _class_digest(klass, code_h, stop = (torch.nn.Module,))
            inst_h = hashlib.sha256()
            for name, module in encoder.named_modules():
                inst_h.update(name.encode())
                _instance_digest(
                    module,
                    inst_h,
                    skip = (
                        "_parameters",
                        "_buffers",
                        "_modules",
                        "_forward_hooks",
                        "_forward_pre_hooks",
                        "_forward_hooks_with_kwargs",
                        "_forward_pre_hooks_with_kwargs",
                        "_forward_hooks_always_called",
                        "_backward_hooks",
                        "_backward_pre_hooks",
                        "_state_dict_hooks",
                        "_state_dict_pre_hooks",
                        "_load_state_dict_pre_hooks",
                        "_load_state_dict_post_hooks",
                        "_non_persistent_buffers_set",
                        "config",
                    ),
                )
            weights_h = hashlib.sha256()
            _weights_digest(encoder, weights_h)
            encoders[attr] = {
                "class": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
                "name_or_path": str(getattr(encoder, "name_or_path", "") or ""),
                "config": _config_digest(getattr(encoder, "config", None)),
                "code": code_h.hexdigest(),
                "modules": inst_h.hexdigest(),
                "weights": weights_h.hexdigest(),
            }
        if not encoders:
            return None
        env = {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith(_ENV_PREFIXES) and k not in _ENV_IGNORED
        }
        return {
            "format": _FORMAT,
            "family": str(family or ""),
            "pipe": f"{cls.__module__}.{cls.__qualname__}",
            "pipe_code": pipe_h.hexdigest(),
            "pipe_config": _config_digest(dict(config)) if config is not None else "none",
            "pipe_name_or_path": str(
                getattr(config, "_name_or_path", "") or getattr(pipe, "name_or_path", "") or ""
            ),
            "encoders": encoders,
            "python": sys.version,
            "torch": str(torch.__version__),
            "transformers": _version("transformers"),
            "diffusers": _version("diffusers"),
            "source": _source_digest(),
            "env": env,
        }
    except _Unkeyable:
        return None
    except Exception:  # noqa: BLE001 - anything unexpected: probe as before
        return None


def _key(fp: dict) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys = True).encode()).hexdigest()[:32]


def _read_all(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001 - absent / torn / unreadable reads as empty
        return {}
    return data if isinstance(data, dict) else {}


_PLAN_KEYS = {"layers": int, "last_consumed": int, "head_consumed": bool, "stack": str}


def _valid_plan(plan: Any) -> bool:
    if not isinstance(plan, dict):
        return False
    for attr, info in plan.items():
        if not isinstance(attr, str) or not isinstance(info, dict):
            return False
        for name, kind in _PLAN_KEYS.items():
            value = info.get(name)
            if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
                return False
        if not -1 <= info["last_consumed"] < info["layers"]:
            return False
        if info.get("head") is not None and not isinstance(info["head"], str):
            return False
        tuple_len = info.get("tuple_len")
        if tuple_len is not None and (
            not isinstance(tuple_len, int) or isinstance(tuple_len, bool)
        ):
            return False
    return True


def load(fp: Optional[dict]) -> Optional[dict]:
    """The stored plan for exactly this fingerprint, or None."""
    if fp is None or not enabled():
        return None
    path = _cache_file()
    if path is None:
        return None
    entry = _read_all(path).get(_key(fp))
    if not isinstance(entry, dict) or entry.get("fingerprint") != fp:
        return None
    plan = entry.get("plan")
    return plan if _valid_plan(plan) else None


def store(fp: Optional[dict], plan: Any) -> bool:
    """Persist a successful probe's plan. Best-effort; never raises."""
    if fp is None or not enabled() or not _valid_plan(plan):
        return False
    path = _cache_file()
    if path is None:
        return False
    tmp: Optional[Path] = None
    with _LOCK:
        try:
            data = _read_all(path)
            data[_key(fp)] = {"fingerprint": fp, "plan": plan, "t": time.time()}
            if len(data) > _MAX_ENTRIES:
                oldest = sorted(
                    data,
                    key = lambda k: (data[k] or {}).get("t", 0) if isinstance(data[k], dict) else 0,
                )
                for k in oldest[: len(data) - _MAX_ENTRIES]:
                    data.pop(k, None)
            path.parent.mkdir(parents = True, exist_ok = True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(data, sort_keys = True), encoding = "utf-8")
            os.replace(tmp, path)
            return True
        except Exception:  # noqa: BLE001 - a cache write never fails a load
            if tmp is not None:
                try:
                    tmp.unlink()
                except Exception:  # noqa: BLE001
                    pass
            return False
