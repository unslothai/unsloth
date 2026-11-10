# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Host-RAM LRU of text-encoder outputs; hits return copies so in-place edits cannot poison renders."""

from __future__ import annotations

import collections
import hashlib
import inspect
import json
import os
import threading
import weakref
from typing import Any, Optional

_ENV_ENABLE = "UNSLOTH_DIFFUSION_PROMPT_CACHE"
_ENV_BUDGET_MB = "UNSLOTH_DIFFUSION_PROMPT_CACHE_MB"
_DEFAULT_BUDGET_MB = 256
_FALSE_TOKENS = ("0", "false", "no", "off")

_KEY_EXCLUDED_ARGS = frozenset({"device", "generator"})
# Cache the per-text helper so an unchanged CFG negative is reused across prompts. "0" = pair key only.
_ENV_PER_TEXT = "UNSLOTH_DIFFUSION_PROMPT_CACHE_PER_TEXT"
_PER_TEXT_METHODS = {
    "Wan": "_get_t5_prompt_embeds",
    "ZImage": "_encode_prompt",
}
_TEXT_ENCODER_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4")
# "0": H3 calls with vision inputs bypass the cache.
_ENV_H3_VISION = "UNSLOTH_H3_VISION_PROMPT_CACHE"


def enabled() -> bool:
    return (os.environ.get(_ENV_ENABLE) or "").strip().lower() not in _FALSE_TOKENS


def budget_bytes() -> int:
    raw = (os.environ.get(_ENV_BUDGET_MB) or "").strip()
    try:
        mb = float(raw) if raw else float(_DEFAULT_BUDGET_MB)
    except ValueError:
        mb = float(_DEFAULT_BUDGET_MB)
        raw = ""  # malformed = unset, so the automatic cap still applies
    budget = int(max(0.0, mb) * 1024 * 1024)
    if not raw:
        total = _host_ram_bytes()
        if total:
            budget = min(budget, max(16 * 1024 * 1024, total // 64))
    return budget


def _host_ram_bytes() -> int:
    """Entries live in host RAM, so an enforcing cgroup limit caps the budget too."""
    try:
        total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        total = 0
    try:
        from .diffusion_memory import _cgroup_memory_limit_mib
        limit_mib = _cgroup_memory_limit_mib()
    except Exception:  # noqa: BLE001 - unreadable means no limit
        limit_mib = None
    if limit_mib:
        limit = int(limit_mib) * 1024 * 1024
        total = min(total, limit) if total else limit
    return total


def _torch():
    import torch
    return torch


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    torch = _torch()
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    image = _image_fingerprint(value)
    if image is not None:
        return image
    raise TypeError(type(value).__name__)


def _image_fingerprint(value: Any) -> Optional[list]:
    """Edit / reference images key by pixel content, so a repeat edit reuses the encoder output."""
    if not (hasattr(value, "tobytes") and hasattr(value, "mode") and hasattr(value, "size")):
        return None
    module = type(value).__module__ or ""
    if not module.startswith("PIL."):
        return None
    digest = hashlib.sha256(value.tobytes()).hexdigest()
    return ["PIL.Image", str(value.mode), list(value.size), digest]


def _is_tensor(obj: Any) -> bool:
    return hasattr(obj, "detach") and hasattr(obj, "clone") and hasattr(obj, "element_size")


def _map_tensors(obj: Any, fn: Any) -> Any:
    if _is_tensor(obj):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(_map_tensors(o, fn) for o in obj)
    if isinstance(obj, list):
        return [_map_tensors(o, fn) for o in obj]
    if isinstance(obj, dict) and type(obj) is dict:
        return {k: _map_tensors(v, fn) for k, v in obj.items()}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(type(obj).__name__)


def _nbytes(obj: Any) -> int:
    total = [0]

    def count(t: Any) -> Any:
        total[0] += int(t.numel()) * int(t.element_size())
        return t

    _map_tensors(obj, count)
    return total[0]


class PromptCache:
    def __init__(self, budget: Optional[int] = None) -> None:
        self.budget = budget_bytes() if budget is None else int(budget)
        self._entries: "collections.OrderedDict[str, tuple[Any, int]]" = collections.OrderedDict()
        self._lock = threading.Lock()
        self.bytes = 0
        self.stats = {"hits": 0, "misses": 0, "bypassed": 0, "evictions": 0, "stored": 0}

    def get(self, key: str, device: Any) -> Any:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            self.stats["hits"] += 1
            stored = entry[0]
        return _map_tensors(stored, lambda t: _fresh(t, device))

    def put(self, key: str, result: Any) -> bool:
        try:
            size = _nbytes(result)
        except TypeError:
            return False
        if size <= 0 or size > self.budget:
            return False
        stored = _map_tensors(result, _store_copy)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self.bytes -= old[1]
            self._entries[key] = (stored, size)
            self.bytes += size
            self.stats["stored"] += 1
            while self.bytes > self.budget and self._entries:
                _, (_, dropped) = self._entries.popitem(last = False)
                self.bytes -= dropped
                self.stats["evictions"] += 1
        return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.bytes = 0

    def __len__(self) -> int:
        return len(self._entries)

    def describe(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self.bytes,
                "budget": self.budget,
                **self.stats,
            }


def _store_copy(t: Any) -> Any:
    t = t.detach()
    device = t.device
    # Not pinned: pin_memory() costs a slow cudaHostAlloc per entry, to save only ~0.3 ms per hit.
    stored = t.to("cpu", copy = True)
    stored._unsloth_src_device = device
    return stored


def _fresh(t: Any, device: Any) -> Any:
    target = device if device is not None else getattr(t, "_unsloth_src_device", t.device)
    torch = _torch()
    target = torch.device(target)
    if target.type == t.device.type and (target.index is None or target.index == t.device.index):
        return t.clone()
    return t.to(target, copy = True)


def _encoder_identity(pipe: Any) -> list:
    ident = []
    for attr in _TEXT_ENCODER_ATTRS:
        module = getattr(pipe, attr, None)
        if module is not None:
            ident.append([attr, type(module).__name__, id(module)])
    return ident


# "0": every attached LoRA keys the text embeddings, including adapters only the denoiser carries.
_ENV_SCOPED_LORA_KEY = "UNSLOTH_DIFFUSION_TE_SCOPED_LORA_KEY"
_DENOISER_ATTRS = (
    "transformer",
    "transformer_2",
    "transformer_3",
    "unconditional_transformer",
    "unet",
    "prior",
    "decoder",
)
# Reading these off a denoiser runs none of its layers.
_INERT_MODULE_ATTRS = frozenset({"dtype", "device", "config"})
_MAX_FOLLOWED_METHODS = 64
# Marks an instance wrapper that only forwards to the same-named class method and runs no denoiser
TRANSPARENT_WRAPPER_ATTR = "_unsloth_prompt_cache_transparent"


def mark_transparent_wrapper(fn: Any) -> Any:
    """Self-reference, so a ``functools.wraps`` copy on an outer wrapper is not trusted."""
    setattr(fn, TRANSPARENT_WRAPPER_ATTR, fn)
    return fn


def is_transparent_wrapper(fn: Any) -> bool:
    return fn is not None and getattr(fn, TRANSPARENT_WRAPPER_ATTR, None) is fn


def _scoped_lora_key() -> bool:
    return (os.environ.get(_ENV_SCOPED_LORA_KEY) or "").strip().lower() not in _FALSE_TOKENS


def _inert_base_property(cls: Any, attr: str, prop: Any) -> bool:
    """``prop`` is ``DiffusionPipeline``'s own (``_execution_device``, ``device``, ``dtype``): no layer runs."""
    try:
        from diffusers import DiffusionPipeline
    except Exception:  # noqa: BLE001 - no base to compare against: follow the getter instead
        return False
    return cls is not DiffusionPipeline and prop is inspect.getattr_static(
        DiffusionPipeline, attr, None
    )


def _encode_skips_denoisers(
    cls: Any,
    name: str,
    instance: Any = None,
) -> bool:
    """True only when ``cls.<name>`` and the ``self`` methods / properties it reads provably run no denoiser."""
    import ast
    import textwrap

    own = {
        attr: value
        for attr, value in (getattr(instance, "__dict__", None) or {}).items()
        if not _transparent_wrapper(cls, attr, value, instance)
    }
    denoisers = [own.get(attr) for attr in _DENOISER_ATTRS if own.get(attr) is not None]
    seen: set = set()
    todo = [name]
    while todo:
        method = todo.pop()
        if method in seen:
            continue
        seen.add(method)
        if len(seen) > _MAX_FOLLOWED_METHODS:
            return False
        if method in own:
            return False
        func = inspect.getattr_static(cls, method, None)
        if func is None:
            continue
        if isinstance(func, property):
            if _inert_base_property(cls, method, func):
                continue
            func = func.fget
            if func is None:
                return False
        elif isinstance(func, (staticmethod, classmethod)):
            func = func.__func__
        if not inspect.isfunction(inspect.unwrap(func)):
            return False  # a descriptor or callable object whose reads this walk cannot see
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(func))))
        except (OSError, TypeError, SyntaxError, ValueError):
            return False
        fn = tree.body[0] if tree.body else None
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not fn.args.args:
            return False
        me = fn.args.args[0].arg
        parents = {id(c): n for n in ast.walk(fn) for c in ast.iter_child_nodes(n)}
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id == "super":
                return False
            if isinstance(node, ast.Call):
                callee = node.func.id if isinstance(node.func, ast.Name) else None
                if callee in ("isinstance", "issubclass", "type", "id"):
                    continue
                if (
                    callee in ("getattr", "hasattr", "vars", "setattr", "delattr")
                    and node.args
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == me
                ):
                    return False
                if any(
                    isinstance(arg, ast.Name) and arg.id == me
                    for arg in [*node.args, *(k.value for k in node.keywords)]
                ):
                    return False  # self handed to a helper: what it reads is unknown
            if not (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == me
            ):
                continue
            if node.attr in _DENOISER_ATTRS:
                parent = parents.get(id(node))
                if not (
                    isinstance(parent, ast.Attribute)
                    and parent.value is node
                    and parent.attr in _INERT_MODULE_ATTRS
                ):
                    return False
            elif node.attr in own:
                if _bound_code(own[node.attr]) or _opaque_callable(own[node.attr], denoisers):
                    return False
            elif _followable(inspect.getattr_static(cls, node.attr, None)):
                todo.append(node.attr)
    return True


def _bound_code(value: Any) -> bool:
    import functools
    import types
    return isinstance(value, (types.FunctionType, types.MethodType, functools.partial))


def _transparent_wrapper(cls: Any, attr: str, value: Any, instance: Any) -> bool:
    """``value`` is a ``TRANSPARENT_WRAPPER_ATTR`` wrapper of ``instance``'s own ``cls.<attr>``."""
    if not callable(value):
        return False
    for _ in range(8):
        if not is_transparent_wrapper(value):
            break
        value = getattr(value, "__wrapped__", None)
    else:
        return False
    if value is getattr(instance, "__dict__", {}).get(attr):
        return False
    target = inspect.getattr_static(cls, attr, None)
    return (
        getattr(value, "__self__", None) is instance
        and target is not None
        and getattr(value, "__func__", None) is target
    )


def _opaque_callable(value: Any, denoisers: list) -> bool:
    """Callable the walk cannot read; tokenizers/processors and denoiser-free modules are exempt."""
    import types

    if (
        not callable(value)
        or isinstance(value, type)
        or isinstance(value, types.BuiltinFunctionType)
    ):
        return False
    try:
        import torch
        if isinstance(value, torch.nn.Module):
            ids = {id(m) for d in denoisers if isinstance(d, torch.nn.Module) for m in d.modules()}
            return any(id(m) in ids for m in value.modules()) or any(
                v is d for v in vars(value).values() for d in denoisers
            )
    except Exception:  # noqa: BLE001 - unclassifiable
        return True
    try:
        from transformers import PreTrainedTokenizerBase, ProcessorMixin
        from transformers.image_processing_utils import ImageProcessingMixin
        from transformers.feature_extraction_utils import FeatureExtractionMixin

        if isinstance(
            value,
            (PreTrainedTokenizerBase, ProcessorMixin, ImageProcessingMixin, FeatureExtractionMixin),
        ):
            return False
    except Exception:  # noqa: BLE001 - no transformers: nothing to exempt
        pass
    return True


def _followable(value: Any) -> bool:
    return value is not None and (callable(value) or hasattr(type(value), "__get__"))


def _adapter_components(owner: Any) -> Optional[dict]:
    """Adapter name -> the components it patches, or None when the owner cannot say."""
    out: dict = {}
    try:
        listed = (
            owner.get_list_adapters() if callable(getattr(owner, "get_list_adapters", None)) else {}
        )
        for component, names in (listed or {}).items():
            for n in names or ():
                out.setdefault(n, set()).add(str(component))
    except Exception:  # noqa: BLE001 - unclassifiable
        return None
    try:
        from .diffusion_lora_files import KRON_ATTR

        roots = {}
        for attr in (*_DENOISER_ATTRS, *_TEXT_ENCODER_ATTRS):
            module = getattr(owner, attr, None)
            if module is not None:
                roots[id(module)] = attr
        for n, adapters in (owner.__dict__.get(KRON_ATTR, {}) or {}).items():
            for adapter in adapters:
                out.setdefault(n, set()).add(roots.get(id(getattr(adapter, "root", None)), "?"))
        from .diffusion_lora_quant_merge import _MERGED_ATTR

        merged = getattr(owner, _MERGED_ATTR, None)
        if merged:
            # merge() refuses unless every merged layer sits inside the denoisers
            for n in merged.get("weights", {}) or {}:
                out.setdefault(n, set()).add("transformer")
    except Exception:  # noqa: BLE001 - unclassifiable
        return None
    return out


def _lora_state(pipe: Any, skips_denoisers: bool = False) -> Any:
    try:
        loras = tuple(getattr(pipe, "_unsloth_loras", ()) or ())
    except TypeError:
        return None
    if skips_denoisers and loras and _scoped_lora_key():
        components = _adapter_components(pipe)
        if components is not None:
            loras = tuple(
                entry
                for entry in loras
                if not (
                    isinstance(entry, (tuple, list))
                    and entry
                    and components.get(entry[0])
                    and components[entry[0]] <= set(_DENOISER_ATTRS)
                )
            )
    try:
        return _plain(loras)
    except TypeError:
        return None


def _embedding_state(pipe: Any) -> Any:
    try:
        return _plain(tuple(pipe.__dict__.get("_unsloth_embeddings", ()) or ()))
    except (AttributeError, TypeError):
        return None


def _hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys = True, separators = (",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_for(pipe: Any) -> Optional[PromptCache]:
    return getattr(pipe, "_unsloth_prompt_cache", None)


def install(
    pipe: Any,
    *,
    identity: Optional[dict] = None,
    lora_owner: Any = None,
    logger: Any = None,
) -> bool:
    """``lora_owner``: the ``from_pipe`` source that tracks adapters; its store is shared."""
    if not enabled() or pipe is None:
        return False
    if cache_for(pipe) is not None:
        return True
    cache = cache_for(lora_owner) if lora_owner is not None else None
    try:
        if cache is None:
            cache = PromptCache()
    except Exception as exc:  # noqa: BLE001 - optimisation only
        _warn(logger, "setup", exc)
        return False
    if cache.budget <= 0:
        return False
    load_fp = _plain(identity or {})
    wrapped = _wrap_encode_prompt(pipe, cache, load_fp, lora_owner or pipe, logger)
    per_text = _per_text_method(pipe)
    if wrapped and per_text is not None:
        _wrap_encode_prompt(pipe, cache, load_fp, lora_owner or pipe, logger, name = per_text)
    registered = _register_h3(pipe, cache, load_fp, logger)
    if not (wrapped or registered):
        return False
    try:
        pipe._unsloth_prompt_cache = cache
    except Exception as exc:  # noqa: BLE001
        _warn(logger, "attach", exc)
        return False
    if logger is not None:
        logger.info(
            "diffusion.prompt_cache: on (%.0f MB budget; %s)",
            cache.budget / 1024**2,
            "encode_prompt" if wrapped else "modular text encoder",
        )
    return True


def release(pipe: Any) -> None:
    cache = cache_for(pipe)
    if cache is not None:
        cache.clear()


def _per_text_method(pipe: Any) -> Optional[str]:
    if (os.environ.get(_ENV_PER_TEXT) or "").strip().lower() in _FALSE_TOKENS:
        return None
    cls = type(pipe).__name__
    for prefix, name in _PER_TEXT_METHODS.items():
        if cls.startswith(prefix) and callable(getattr(pipe, name, None)):
            return name
    return None


def _wrap_encode_prompt(
    pipe: Any,
    cache: PromptCache,
    load_fp: Any,
    lora_owner: Any,
    logger: Any,
    *,
    name: str = "encode_prompt",
) -> bool:
    encode = getattr(pipe, name, None)
    if not callable(encode):
        return False
    try:
        signature = inspect.signature(encode)
    except (TypeError, ValueError):
        return False
    try:
        skips_denoisers = _encode_skips_denoisers(type(pipe), name, pipe)
    except Exception:  # noqa: BLE001 - unprovable: the full LoRA key
        skips_denoisers = False

    def cached_encode_prompt(*args: Any, **kwargs: Any) -> Any:
        key = None
        try:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            keyed = {
                k: _plain(v) for k, v in bound.arguments.items() if k not in _KEY_EXCLUDED_ARGS
            }
            payload = {
                "load": load_fp,
                "encoders": _encoder_identity(pipe),
                "loras": _lora_state(lora_owner, skips_denoisers),
                "args": keyed,
            }
            if name != "encode_prompt":
                payload["method"] = name
            # Textual-inversion embeddings change what the same text encodes to; absent = the unchanged key.
            embeddings = _embedding_state(pipe)
            if embeddings:
                payload["embeddings"] = embeddings
            key = _hash(payload)
            hit = cache.get(key, bound.arguments.get("device"))
        except TypeError:
            cache.stats["bypassed"] += 1
            key = None
        except Exception as exc:  # noqa: BLE001 - never fail a render over the cache
            _warn(logger, "lookup", exc)
            key = None
        # Encode outside the handlers so an encoder failure is not chained onto the lookup error.
        if key is None:
            return encode(*args, **kwargs)
        if hit is not None:
            return hit
        result = encode(*args, **kwargs)
        cache.stats["misses"] += 1
        try:
            cache.put(key, result)
        except Exception as exc:  # noqa: BLE001 - a failed store only skips reuse
            _warn(logger, "store", exc)
        return result

    cached_encode_prompt.__signature__ = signature
    cached_encode_prompt.__wrapped__ = encode
    setattr(pipe, name, cached_encode_prompt)
    return True


# MiniMax-H3 steps call this module global, so it is shimmed process-wide and routed per text encoder.
_H3_MODULE = "diffusers.modular_pipelines.minimax_h3.encoders"
_H3_FUNC = "get_qwen3vl_prompt_embeds"
_H3_REGISTRY: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_H3_LOCK = threading.Lock()


def _register_h3(pipe: Any, cache: PromptCache, load_fp: Any, logger: Any) -> bool:
    text_encoder = _modular_text_encoder(pipe)
    if text_encoder is None:
        return False
    try:
        import importlib
        module = importlib.import_module(_H3_MODULE)
    except Exception:  # noqa: BLE001 - not an H3-capable diffusers
        return False
    with _H3_LOCK:
        original = getattr(module, _H3_FUNC, None)
        if not callable(original):
            return False
        if not getattr(original, "_unsloth_prompt_cache_shim", False):
            shim = _make_h3_shim(original, logger)
            setattr(module, _H3_FUNC, shim)
        try:
            _H3_REGISTRY[text_encoder] = (cache, load_fp)
        except TypeError:
            return False
    return True


def _modular_text_encoder(pipe: Any) -> Any:
    if callable(getattr(pipe, "encode_prompt", None)):
        return None
    if not hasattr(pipe, "blocks") and "Modular" not in type(pipe).__name__:
        return None
    try:
        return getattr(pipe, "text_encoder", None)
    except Exception:  # noqa: BLE001 - a lazy component that is not loaded
        return None


def _vision_fingerprint(vision: Any) -> list:
    """Host-side content key of H3 vision tensors; TypeError (= bypass) for device tensors or unknown values."""
    if (os.environ.get(_ENV_H3_VISION) or "").strip().lower() in _FALSE_TOKENS:
        raise TypeError("vision inputs")
    if not isinstance(vision, dict):
        raise TypeError(type(vision).__name__)
    torch = _torch()
    out = []
    for name in sorted(vision):
        t = vision[name]
        if not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.is_sparse:
            raise TypeError(f"vision input {name}")
        flat = t.detach().contiguous().reshape(-1)
        digest = hashlib.sha256(memoryview(flat.view(torch.uint8).numpy())).hexdigest()
        out.append([str(name), str(t.dtype), list(t.shape), digest])
    return out


def _make_h3_shim(original: Any, logger: Any) -> Any:
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        signature = None

    def get_qwen3vl_prompt_embeds(*args: Any, **kwargs: Any) -> Any:
        if signature is None:
            return original(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            text_encoder = arguments.get("text_encoder")
            registered = _H3_REGISTRY.get(text_encoder) if text_encoder is not None else None
        except Exception:  # noqa: BLE001
            registered = None
        if registered is None:
            return original(*args, **kwargs)
        cache, load_fp = registered
        key = None
        try:
            payload = {
                "load": load_fp,
                "encoder": [type(text_encoder).__name__, id(text_encoder)],
                "token_ids": _plain(list(arguments.get("token_ids") or ())),
                "layer": _plain(arguments.get("text_encoder_layer")),
                "dtype": _plain(arguments.get("dtype")),
            }
            vision = arguments.get("vision_inputs")
            if vision:
                payload["vision"] = _vision_fingerprint(vision)
            key = _hash(payload)
            hit = cache.get(key, arguments.get("device"))
        except TypeError:
            cache.stats["bypassed"] += 1
            key = None
        except Exception as exc:  # noqa: BLE001
            _warn(logger, "h3 lookup", exc)
            key = None
        if key is None:
            return original(*args, **kwargs)
        if hit is not None:
            return hit
        result = original(*args, **kwargs)
        cache.stats["misses"] += 1
        try:
            cache.put(key, result)
        except Exception as exc:  # noqa: BLE001
            _warn(logger, "h3 store", exc)
        return result

    get_qwen3vl_prompt_embeds._unsloth_prompt_cache_shim = True
    get_qwen3vl_prompt_embeds.__wrapped__ = original
    if signature is not None:
        get_qwen3vl_prompt_embeds.__signature__ = signature
    return get_qwen3vl_prompt_embeds


def _warn(logger: Any, what: str, exc: Any) -> None:
    if logger is not None:
        logger.warning("diffusion.prompt_cache: %s failed: %s", what, exc)
