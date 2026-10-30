# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What an MLX load would occupy, priced before anything is loaded."""

from __future__ import annotations

import contextlib
import contextvars
import glob
import functools
import json
import os
import posixpath
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

__all__ = [
    "MLX_FIT_MIN_CONTEXT",
    "MLX_KV_BLOCK",
    "MLX_PREFILL_CHUNK",
    "MlxMemoryBreakdown",
    "mlx_fit_context",
    "mlx_fit_outcome",
    "mlx_memory_breakdown",
    "mlx_offload_plan",
    "mlx_shard_files",
    "mlx_weight_bytes",
]

MLX_KV_BLOCK = 256

_PROBE_SHORT = 8
_PROBE_LONG = 40
# Above mlx-vlm's decode block (8 tokens), inside which some targets evaluate their forward.
_DRAFTER_PROBE = (16, 48)

MLX_PREFILL_CHUNK = 2048

MLX_FIT_MIN_CONTEXT = 4096
_KV_GROUP_SIZE = 64

_ATTENTION_LIVE = 3.5
_COMPUTE_BASE_BYTES = int(0.64 * 1024**3)

_RECURRENT_DTYPE_SIZE = 4

# Score copies an unfused call holds at the queries' width; Metal measured 1.03 to 1.15 (mlx 0.32.4).
_SCORE_LIVE = 1.25

_WIDTH_NAMES = {"float64": "f64", "float32": "f32", "bfloat16": "bf16", "float16": "f16"}

_CACHE_BOUND_ATTRS = ("max_size", "chunk_size")

_MAX_SAFETENSORS_HEADER = 100_000_000


@dataclass
class MlxMemoryBreakdown:
    weights_bytes: int
    kv_bytes: int
    compute_bytes: int
    total_bytes: int
    gpu_bytes: int
    n_ctx: int
    layer_count: Optional[int] = None
    cache_type_kv: Optional[str] = None
    kv_estimable: bool = True
    kv_on_gpu: bool = True
    n_parallel: int = 1
    gpu_layers: Optional[int] = None
    # The fractions of the PLE tables / routed experts the load reads from disk; None for none.
    ple_offload: Optional[float] = None
    expert_offload: Optional[float] = None
    offloadable: tuple = ()


def _snapshot_config(model_dir: str) -> Optional[dict]:
    path = os.path.join(model_dir, "config.json")
    try:
        with open(path, "r", encoding = "utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:
        logger.debug("MLX estimate could not read %s: %s", path, exc)
        return None
    if not isinstance(config, dict):
        return None
    return _as_the_loader_reads_it(config)


def _as_the_loader_reads_it(config: dict) -> dict:
    import contextlib
    import io

    try:
        from unsloth_zoo.mlx.loader import _mlx_vlm_config_override_data
    except Exception:
        return config
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            patched = _mlx_vlm_config_override_data(config)
    except Exception:
        return config
    return patched if isinstance(patched, dict) else config


def _escapes_snapshot(shard: str) -> bool:
    # Lexical, not realpath: hub snapshot files are symlinks into ../../blobs by design.
    norm = posixpath.normpath(shard.replace("\\", "/"))
    return norm.startswith("/") or norm.split("/")[0] == ".." or norm[1:2] == ":"


def _indexed_shards(model_dir: str) -> list:
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return []
    try:
        with open(index, encoding = "utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
    except (ValueError, OSError):
        return []
    if not isinstance(weight_map, dict):
        raise ValueError("model.safetensors.index.json has no weight map")
    if any(not isinstance(shard, str) for shard in weight_map.values()):
        raise ValueError("model.safetensors.index.json names a non-string shard")
    if any(_escapes_snapshot(shard) for shard in weight_map.values()):
        raise ValueError("model.safetensors.index.json names a shard outside the snapshot")
    named = sorted(set(weight_map.values()))
    return [
        os.path.join(model_dir, shard)
        for shard in named
        if os.path.isfile(os.path.join(model_dir, shard))
    ]


def mlx_shard_files(model_dir: str, config: Optional[dict] = None) -> list:
    """The shards the package that would load this model actually reads."""
    names = [
        (os.path.basename(path), path)
        for path in glob.glob(os.path.join(model_dir, "*.safetensors"))
    ]
    if config is None:
        return [path for _, path in names]
    if _builds_on_mlx_vlm(config):
        return _indexed_shards(model_dir) or [
            path for name, path in names if name != "consolidated.safetensors"
        ]
    return [path for name, path in names if name.startswith("model")]


def _shard_bytes(model_dir: str, config: Optional[dict] = None) -> int:
    total = 0
    for shard in mlx_shard_files(model_dir, config):
        try:
            total += os.path.getsize(os.path.realpath(shard))
        except OSError:
            continue
    return total


_SAFETENSORS_DTYPES = {
    "F64": "float32",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U64": "uint64",
    "U32": "uint32",
    "U16": "uint16",
    "U8": "uint8",
    "BOOL": "bool_",
}


def _checkpoint_tensors(model_dir: str, config: Optional[dict], dtype):
    import mlx.core as mx

    tensors = {}
    for shard in mlx_shard_files(model_dir, config):
        for name, meta in _safetensors_header(shard).items():
            if name == "__metadata__":
                continue
            stored = getattr(mx, _SAFETENSORS_DTYPES.get(meta["dtype"], "float32"))
            if stored in (mx.float32, mx.float16, mx.bfloat16):
                stored = dtype
            tensors[name] = mx.zeros(tuple(meta["shape"]), dtype = stored)
    return tensors


def _resident_bytes(model_dir: str, config: dict, quantize: bool) -> int:
    """Parameter bytes the load leaves resident, at the width it holds them."""
    import contextlib
    import inspect
    import io

    from mlx.utils import tree_flatten
    from unsloth_zoo.mlx.loader import (
        _apply_mlx_quantization,
        _resolve_mlx_quantization_spec,
    )

    # Chat loads quantize embeddings with the rest; an older unsloth_zoo keeps them full width.
    embeddings = (
        {"quantize_embeddings": True}
        if "quantize_embeddings" in inspect.signature(_resolve_mlx_quantization_spec).parameters
        else {}
    )

    spec = (
        None
        if not quantize
        else _resolve_mlx_quantization_spec(
            load_in_4bit = True,
            load_in_8bit = False,
            load_in_16bit = False,
            load_in_fp8 = False,
            load_in_mxfp4 = False,
            load_in_nvfp4 = False,
            full_finetuning = False,
            q_bits = None,
            q_group_size = None,
            q_mode = None,
            mlx_quantization_config = None,
            quantization_config = None,
            quant_predicate = None,
            quantize_modules = None,
            force_requantize = False,
            **embeddings,
        )
    )
    if quantize and not spec.enabled:
        raise ValueError("the loader would quantize nothing")

    dtype = _runtime_dtype()
    model = _whole_model(config, dtype)
    tensors = _checkpoint_tensors(model_dir, config, dtype)
    if not tensors:
        raise ValueError("this checkpoint ships no tensors")
    if hasattr(model, "sanitize"):
        tensors = model.sanitize(tensors)
    model.load_weights(list(tensors.items()), strict = False)
    orphaned = [name for name, _ in tree_flatten(model.parameters()) if name not in tensors]
    if orphaned:
        raise ValueError(
            f"{len(orphaned)} parameters this checkpoint does not supply, "
            f"beginning {orphaned[0]}"
        )
    if quantize:
        with contextlib.redirect_stdout(io.StringIO()):
            model, _ = _apply_mlx_quantization(
                model,
                config,
                spec,
                is_vlm = _builds_on_mlx_vlm(config),
            )
    return sum(
        value.nbytes
        for name, value in tree_flatten(model.parameters())
        if hasattr(value, "nbytes") and not _mapped(name, config, quantize)
    )


def mlx_weight_bytes(
    model_dir: str,
    config: Optional[dict] = None,
    load_in_4bit: bool = False,
) -> int:
    """Weight bytes an MLX load would leave resident."""
    shards = _shard_bytes(model_dir, config)
    if not config:
        return shards
    if config.get("quantization") or config.get("quantization_config"):
        return shards - _mapped_bytes(model_dir, config)
    try:
        return _resident_bytes(model_dir, config, quantize = load_in_4bit)
    except Exception as exc:
        logger.debug("MLX estimate priced %s as stored: %s", model_dir, exc)
        return shards - _mapped_bytes(model_dir, config, load_in_4bit)


def _mapped(
    name: str,
    config: dict,
    quantize: bool = False,
) -> bool:
    """A table mlx-vlm reads row by row from the checkpoint rather than holding it: DeepSeek
    V4.1's Engram, unless the load quantizes it."""
    return not quantize and _maps_engram(config) and ".engram.embed." in name


def _maps_engram(config: dict) -> bool:
    # mlx-vlm offloads Engram for every deepseek_v41 load, whatever the config says of its layers.
    return "deepseek_v41" in (
        config.get("model_type"),
        (config.get("text_config") or {}).get("model_type"),
    )


def _mapped_bytes(
    model_dir: str,
    config: dict,
    quantize: bool = False,
) -> int:
    if quantize or not _maps_engram(config):
        return 0
    total = 0
    for shard in mlx_shard_files(model_dir, config):
        for name, meta in _safetensors_header(shard).items():
            if name != "__metadata__" and _mapped(name, config):
                start, end = meta["data_offsets"]
                total += end - start
    return total


def _safetensors_header(shard: str) -> dict:
    with open(shard, "rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        limit = min(_MAX_SAFETENSORS_HEADER, os.fstat(handle.fileno()).st_size - 8)
        if not 0 < length <= limit:
            raise ValueError(f"{shard} declares a {length}-byte safetensors header")
        return json.loads(handle.read(length))


def _runtime_dtype():
    import mlx.core as mx
    chip = mx.device_info().get("device_name", "") or ""
    return mx.float16 if chip.startswith(("Apple M1", "Apple M2")) else mx.bfloat16


def _generation_settings(config: dict) -> tuple:
    """``(prefill chunk, kv group size)`` a load of this model runs at."""
    try:
        from core.inference.mlx_inference import MLX_KV_GROUP_SIZE, mlx_prefill_chunk
    except Exception as exc:
        logger.debug("MLX estimate cannot reach the loader's generation settings: %s", exc)
        return MLX_PREFILL_CHUNK, _KV_GROUP_SIZE
    vision = _loads_as_vision(config)
    if vision and _routes_to_diffusion(config):
        raise ValueError("mlx-vlm would divert this to a diffusion generator")
    return (
        mlx_prefill_chunk(vision = vision, config = config if vision else None),
        MLX_KV_GROUP_SIZE,
    )


# The route a sizing prices when the load overrides the config's own: a speculative text load goes through mlx-vlm.
_ROUTE_VISION: contextvars.ContextVar = contextvars.ContextVar("mlx_route_vision", default = None)


def _loads_as_vision(config: dict) -> bool:
    forced = _ROUTE_VISION.get()
    if forced is not None:
        return forced
    from types import SimpleNamespace

    try:
        from utils.models.model_config import _is_vlm
    except Exception:
        return any(
            key in config
            for key in ("vision_config", "img_processor", "image_token_index", "projector_config")
        )
    return bool(_is_vlm(SimpleNamespace(**config)))


@contextlib.contextmanager
def _on_route(vision: Optional[bool]):
    token = _ROUTE_VISION.set(vision)
    try:
        yield
    finally:
        _ROUTE_VISION.reset(token)


def _builds_on_mlx_vlm(config: dict) -> bool:
    """Whether mlx-vlm builds the weights: on a vision route, or for a text model unsloth_zoo builds
    there because mlx-lm cannot; that one is still served as text."""
    if _loads_as_vision(config):
        return True
    try:
        from unsloth_zoo.mlx.loader import _mlx_vlm_only_text_model
        return bool(_mlx_vlm_only_text_model(config, config.get("model_type") or ""))
    except Exception:
        return False


def _routes_to_diffusion(config: dict) -> bool:
    """Whether mlx-vlm would divert this load to a diffusion generator."""
    try:
        from mlx_vlm.generate.diffusion import is_diffusion_model
        from mlx_vlm.utils import get_model_and_args
    except ImportError:
        return False
    arch = get_model_and_args(config)[0]
    loader_config = _loader_config(arch, config)
    if (
        getattr(loader_config, "canvas_length", None) is None
        and getattr(loader_config, "mask_token_id", None) is None
    ):
        return False
    return bool(is_diffusion_model(arch.Model(loader_config), {}))


def _declines_to_chunk(model_class, model) -> bool:
    """Whether this model's runtime feeds the whole prompt to a single step."""
    return bool(
        getattr(model_class, "no_chunked_prefill", False)
        or getattr(model, "no_chunked_prefill", False)
    )


def _tower_needs_parent(arch) -> bool:
    import inspect

    tower = getattr(arch, "LanguageModel", None)
    if tower is None or tower.__init__ is object.__init__:
        return True
    try:
        parameters = inspect.signature(tower.__init__).parameters
    except (TypeError, ValueError):
        return False
    required = [
        name
        for name, value in parameters.items()
        if name != "self"
        and value.default is inspect.Parameter.empty
        and value.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(required) > 1


def _loader_config(arch, config: dict):
    """Config resolved as the loader does (``from_dict`` leaves nested sub-configs as dicts)."""
    from mlx_vlm.utils import update_module_configs

    config = dict(config)
    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    model_config = arch.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config,
        arch,
        config,
        ["text", "vision", "perceiver", "projector", "audio"],
    )
    return model_config


def _whole_model(config: dict, dtype):
    if config.get("model_file"):
        raise ValueError("this checkpoint carries its own model module")
    if _builds_on_mlx_vlm(config):
        from mlx_vlm.utils import get_model_and_args

        _install_oq_renames(config)
        arch = get_model_and_args(config)[0]
        model = arch.Model(_loader_config(arch, config))
    else:
        from mlx_lm.utils import _get_classes
        model_class, args_class = _get_classes(config)
        model = model_class(args_class.from_dict(config))
    model.set_dtype(dtype)
    model.eval()
    return model


def _probe_models(config: dict, dtype):
    """Language towers to try, in order, with the cache builder and class owning each."""
    if not config.get("model_type"):
        raise ValueError("config.json declares no model_type")
    if config.get("model_file"):
        raise ValueError("this checkpoint carries its own model module")

    def prepared(build):
        def _build():
            model = build()
            model.set_dtype(dtype)
            model.eval()
            return model

        return _build

    if not _builds_on_mlx_vlm(config):
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.utils import _get_classes

        try:
            model_class, args_class = _get_classes(config)
        except Exception as exc:
            raise ValueError(f"mlx-lm cannot resolve this architecture: {exc}")
        yield (
            prepared(lambda: model_class(args_class.from_dict(config))),
            make_prompt_cache,
            model_class,
        )
        return

    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.utils import get_model_and_args

    try:
        arch = get_model_and_args(config)[0]
    except Exception as exc:
        raise ValueError(f"mlx-vlm cannot resolve this architecture: {exc}")

    if not _tower_needs_parent(arch):

        def _tower():
            text_config = getattr(_loader_config(arch, config), "text_config", None)
            if text_config is None:
                raise ValueError("this architecture declares no text config")
            return arch.LanguageModel(text_config)

        yield prepared(_tower), make_prompt_cache, getattr(arch, "Model", None)

    def _whole_tower():
        whole = arch.Model(_loader_config(arch, config))
        return getattr(whole, "language_model", whole)

    yield prepared(_whole_tower), make_prompt_cache, getattr(arch, "Model", None)


def _quantize_like_runtime(cache, kv_bits: Optional[int]):
    """The converted cache a load ends up with, or None if nothing in it converts."""
    if not kv_bits:
        return None
    from core.inference.mlx_inference import _quantize_kv_entries

    try:
        converted = _quantize_kv_entries(list(cache), kv_bits)
    except Exception:
        return None
    return converted if any(new is not old for new, old in zip(converted, cache)) else None


def _sub_caches(entry):
    inner = getattr(entry, "caches", None)
    if not isinstance(inner, (list, tuple)):
        return (entry,)
    return tuple(sub for cache in inner for sub in _sub_caches(cache))


def _block(entry) -> int:
    """Allocation granularity of the storage behind ``entry``, in tokens."""
    step = getattr(entry, "step", None)
    return step if isinstance(step, int) and step > 0 else 1


def _bound_spec(entry):
    for name in _CACHE_BOUND_ATTRS:
        value = getattr(entry, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return type(entry), name, value
    return None


def _entry_bytes(entry) -> int:
    import mlx.core as mx

    # mlx-lm 0.32's KV `state` is the whole preallocated buffer; `keys_and_values` is the held part.
    held = getattr(entry, "keys_and_values", None)
    state = held() if callable(held) and getattr(entry, "keys", None) is not None else entry.state
    if isinstance(state, mx.array):
        state = (state,)
    elif not isinstance(state, (list, tuple)):
        return 0
    total = 0
    for item in state:
        if isinstance(item, mx.array):
            total += item.size * item.dtype.size
        elif isinstance(item, (list, tuple)):
            total += sum(sub.size * sub.dtype.size for sub in item if isinstance(sub, mx.array))
    return total


def _width_name(dtype) -> str:
    name = str(dtype).rsplit(".", 1)[-1]
    return _WIDTH_NAMES.get(name, name)


def _cache_width_name(
    plan, kv_bits, converted: bool, full_width: str, n_ctx: int, prefill_chunk: int
) -> str:
    """How to caption the cache, from the width each growing entry ends up at."""
    charged: dict = {}
    for slot in plan:
        if slot["slope"] <= 0:
            continue
        quantized = converted and slot["converts"]
        name = f"{kv_bits}-bit" if quantized else full_width
        const, slope = ("quant_const", "quant_slope") if quantized else ("const", "slope")
        charged[name] = (
            charged.get(name, 0.0)
            + slot[const]
            + slot[slope] * _held_tokens(slot, n_ctx, prefill_chunk, True)
        )
    if not charged:
        return full_width
    return "/".join(sorted(charged, key = lambda name: charged[name], reverse = True))


def _conv_width(entry) -> int:
    """Total channel width of a linear-attention layer's convolution states."""
    import mlx.core as mx

    state = entry.state
    if not isinstance(state, (list, tuple)):
        return 0
    # mlx-lm 0.32's ArraysCache state is (arrays, left_padding, lengths).
    items = [
        sub for item in state for sub in (item if isinstance(item, (list, tuple)) else (item,))
    ]
    return sum(
        int(item.shape[-1]) for item in items if isinstance(item, mx.array) and item.ndim == 3
    )


def _tower_layers(model) -> Optional[int]:
    seen = model
    for _ in range(3):
        layers = getattr(seen, "layers", None)
        if isinstance(layers, list) and layers:
            return len(layers)
        seen = getattr(seen, "model", None) or getattr(seen, "language_model", None)
        if seen is None:
            return None
    return None


def _tower_widths(model):
    """The widths in ``_config_widths``' order as the tower was BUILT, or zeros."""
    node, depth = model, 0
    while node is not None and depth < 3:
        for name in ("args", "config"):
            resolved = getattr(node, name, None)
            hidden = (
                0 if resolved is None else _config_width(getattr(resolved, "hidden_size", None))
            )
            if hidden > 0:
                heads = _config_width(getattr(resolved, "num_attention_heads", None))
                return (
                    hidden,
                    _config_width(getattr(resolved, "intermediate_size", None))
                    or _config_width(getattr(resolved, "moe_intermediate_size", None)),
                    heads,
                    _config_width(getattr(resolved, "num_key_value_heads", None)) or heads,
                    _config_width(getattr(resolved, "head_dim", None))
                    or (hidden // heads if heads else 0),
                    _config_width(getattr(resolved, "v_head_dim", None)),
                )
        node = getattr(node, "model", None) or getattr(node, "language_model", None)
        depth += 1
    return (0,) * len(_config_widths({}))


def _config_widths(config: dict):
    text = config.get("text_config") or config
    hidden = _config_width(text.get("hidden_size"))
    heads = _config_width(text.get("num_attention_heads"))
    return (
        hidden,
        _config_width(text.get("intermediate_size"))
        or _config_width(text.get("moe_intermediate_size")),
        heads,
        _config_width(text.get("num_key_value_heads")) or heads,
        _config_width(text.get("head_dim")) or (hidden // heads if heads else 0),
        _config_width(text.get("v_head_dim")),
    )


def _entries(cache, quantized = None) -> list:
    entries = []
    for index, entry in enumerate(cache):
        converts = quantized is not None and quantized[index] is not entry
        for leaf in _sub_caches(entry):
            entries.append(
                {
                    "bytes": _entry_bytes(leaf),
                    "quant_bytes": _entry_bytes(leaf),
                    "bound_spec": _bound_spec(leaf),
                    "conv_width": _conv_width(leaf),
                    "block": _block(leaf),
                    "converts": converts,
                }
            )
    return entries


_DEFERRED_EVAL = threading.local()
_DEFERRABLE_LOCK = threading.Lock()


def _deferrable(evaluate):
    def call(*args, **kwargs):
        if getattr(_DEFERRED_EVAL, "depth", 0):
            return None
        return evaluate(*args, **kwargs)

    call._unsloth_deferrable = True
    return call


@contextlib.contextmanager
def weightless_forward():
    """Defer ``mx.eval`` / ``mx.async_eval`` on this thread while a weightless tower runs. Model code
    evaluates mid-forward (mlx-vlm's per-layer ``async_eval``), which on random weights materializes
    the whole model; the probes read only shapes, which need no evaluation."""
    import mlx.core as mx

    with _DEFERRABLE_LOCK:
        if not getattr(mx.eval, "_unsloth_deferrable", False):
            mx.eval, mx.async_eval = _deferrable(mx.eval), _deferrable(mx.async_eval)
    _DEFERRED_EVAL.depth = getattr(_DEFERRED_EVAL, "depth", 0) + 1
    try:
        yield
    finally:
        _DEFERRED_EVAL.depth -= 1


def _probe(config: dict, dtype, n_tokens: int, kv_bits):
    import mlx.core as mx

    cache = None
    whole_prompt = False
    layers = None
    widths = (0,) * len(_config_widths({}))
    failure = ValueError("no architecture module could build this config")
    with weightless_forward():
        for build, make_prompt_cache, model_class in _probe_models(config, dtype):
            model = build()
            cache = make_prompt_cache(model)
            try:
                model(mx.zeros((1, n_tokens), dtype = mx.int32), cache = cache)
            except Exception as exc:
                failure, cache = exc, None
                continue
            whole_prompt = _declines_to_chunk(model_class, model)
            layers = _tower_layers(model)
            widths = _tower_widths(model)
            break
        if cache is None:
            raise failure
        quantized = _quantize_like_runtime(cache, kv_bits)
    entries = _entries(cache, quantized)
    if quantized is not None:
        converted = [leaf for entry in quantized for leaf in _sub_caches(entry)]
        if len(converted) == len(entries):
            for slot, leaf in zip(entries, converted):
                slot["quant_bytes"] = _entry_bytes(leaf)
    return {
        "entries": entries,
        "quantized": quantized is not None,
        "whole_prompt": whole_prompt,
        "layers": layers,
        "widths": widths,
    }


def _cache_plan(config: dict, dtype, kv_bits):
    """Per-entry ``const + slope * T``, solved from two lazy probes."""
    near = _probe(config, dtype, _PROBE_SHORT, kv_bits)
    far = _probe(config, dtype, _PROBE_LONG, kv_bits)
    if not near["entries"] or len(near["entries"]) != len(far["entries"]):
        raise ValueError("cache probe returned no comparable entries")
    plan = _solved(near["entries"], far["entries"], (_PROBE_SHORT, _PROBE_LONG), near["quantized"])
    return (
        plan,
        0 if near["quantized"] else None,
        {
            "whole_prompt": near["whole_prompt"],
            "layers": near["layers"],
            "widths": near["widths"],
        },
    )


def _solved(near: list, far: list, lengths: tuple, quantized: bool) -> list:
    """Per-entry ``const + slope * T`` from the entries of caches filled to two lengths."""
    short, span = lengths[0], lengths[1] - lengths[0]

    def solve(key):
        return [
            (n[key] - (f[key] - n[key]) / span * short, (f[key] - n[key]) / span)
            for n, f in zip(near, far)
        ]

    return [
        {
            "const": const,
            "slope": slope,
            "quant_const": q_const,
            "quant_slope": q_slope,
            "bound_spec": slot["bound_spec"],
            "conv_width": slot["conv_width"],
            "block": slot["block"],
            "converts": slot["converts"] and quantized,
        }
        for slot, (const, slope), (q_const, q_slope) in zip(
            near, solve("bytes"), solve("quant_bytes")
        )
    ]


def _drafter_sizing(config: dict, dtype, drafter: tuple) -> tuple:
    """``(weight bytes, cache plan, capture bytes per prompt token)`` of the drafter ``(path, builtin)``
    attaches, from a weightless target and a lazily built drafter."""
    from unsloth_zoo.mlx.speculative import companion_drafter, native_mtp_drafter, probe_drafter

    path, builtin = drafter
    if not builtin and (_snapshot_config(path) or {}).get("model_file"):
        raise ValueError("this drafter carries its own model module")
    target = _whole_model(config, dtype)
    built = (native_mtp_drafter if builtin else companion_drafter)(path, target, lazy = True)
    if built is None:
        raise ValueError("no MTP splitter knows this checkpoint")
    near, far = (_entries(probe_drafter(target, built, length)) for length in _DRAFTER_PROBE)
    if len(near) != len(far):
        raise ValueError("drafter cache probe returned no comparable entries")
    # A dflash or eagle3 prefill keeps every chunk's captured layers, then a concatenated copy.
    captured = len(getattr(built, "capture", {}).get("capture_layer_ids", ()))
    hidden = _config_width((config.get("text_config") or config).get("hidden_size"))
    return (
        built.weight_bytes,
        _solved(near, far, _DRAFTER_PROBE, False),
        2 * captured * hidden * dtype.size,
    )


@functools.lru_cache(maxsize = 256)
def _bounded_peak(cache_type, attribute: str, value: int, n_ctx: int, prefill_chunk: int) -> int:
    """Largest slot count a bounded cache reaches, by driving the real class."""
    import mlx.core as mx

    cache = cache_type(**{attribute: value})
    settled = value + 4 * (prefill_chunk + (getattr(cache, "step", 0) or MLX_KV_BLOCK))
    n_ctx = min(n_ctx, settled)
    peak, processed = 0, 0

    def call(width):
        nonlocal peak
        trim = getattr(cache, "maybe_trim_front", None)
        if callable(trim):
            trim()
        cache.update_and_fetch(
            mx.zeros((1, 1, width, 1), dtype = mx.bfloat16),
            mx.zeros((1, 1, width, 1), dtype = mx.bfloat16),
        )
        keys = getattr(cache, "keys", None)
        peak = max(peak, 0 if keys is None else int(keys.shape[2]))

    while n_ctx - processed > 1:
        width = min(prefill_chunk, n_ctx - processed - 1)
        call(width)
        processed += width
    call(1)
    call(1)
    return peak


def _held_tokens(
    entry,
    n_ctx: int,
    prefill_chunk: int,
    decoding: bool = True,
) -> int:
    tokens = n_ctx
    block = entry.get("block") or MLX_KV_BLOCK
    if entry["slope"] > 0:
        tokens += 1 if decoding else 0
        if block > 1:
            tokens = -(-tokens // block) * block
    spec = entry.get("bound_spec")
    if spec:
        tokens = _bounded_peak(*spec, n_ctx, prefill_chunk)
    return tokens


def _line_bytes(
    plan,
    n_ctx: int,
    prefill_chunk: int,
    quantized: bool,
    decoding: bool = True,
) -> int:
    const, slope = ("quant_const", "quant_slope") if quantized else ("const", "slope")
    return int(
        sum(
            entry[const] + entry[slope] * _held_tokens(entry, n_ctx, prefill_chunk, decoding)
            for entry in plan
        )
    )


def _kv_bytes(plan, n_ctx: int, quant_start, prefill_chunk: int):
    """Peak cache bytes across the run, and where the cache is quantized from."""
    if quant_start is None:
        return _line_bytes(plan, n_ctx, prefill_chunk, quantized = False), None
    # The load converts the cache before prefill, so no full-width copy is ever held.
    return _line_bytes(plan, n_ctx, prefill_chunk, quantized = True), quant_start


def _config_width(value) -> int:
    """A width from the config, which is not always a single number."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (list, tuple)):
        widths = [int(v) for v in value if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return max(widths) if widths else 0
    return 0


def _widest_quantized_scores(n_ctx: int, boundary: int, prefill_chunk: int) -> int:
    widest = 0
    remaining = n_ctx - boundary - 1
    if remaining > 0:
        whole, rest = divmod(remaining, prefill_chunk)
        if whole:
            widest = prefill_chunk * (boundary + whole * prefill_chunk)
        if rest:
            widest = max(widest, rest * (n_ctx - 1))
    return max(widest, n_ctx + 1)


def _dequantized_row_bytes(widths, dtype_size: int, kv_bits, kv_group_size, prefill_chunk):
    """Bytes per cached token one attention call holds where the runtime dequantizes the cache."""
    _hidden, _intermediate, heads, kv_heads, head_dim, value_dim = widths
    if not (heads and kv_heads and head_dim and kv_bits):
        return None
    try:
        import mlx.core as mx

        from unsloth_zoo.mlx.attention import (
            _row_bytes,
            dequantizing_is_smaller,
            fused_kernel_exists,
        )
    except Exception as exc:
        logger.debug("MLX estimate cannot reach the quantized-attention wrapper: %s", exc)
        return None
    try:
        dtype = mx.float32 if dtype_size == 4 else mx.bfloat16

        def rows(width):
            return mx.quantize(
                mx.zeros((1, kv_heads, 1, width), dtype = dtype),
                group_size = kv_group_size,
                bits = kv_bits,
            )

        keys, values = rows(head_dim), rows(value_dim or head_dim)
        queries = mx.zeros((1, heads, prefill_chunk, head_dim), dtype = dtype)
        if not fused_kernel_exists(queries, keys, values, kv_group_size, "causal"):
            return None
        if not dequantizing_is_smaller(queries, keys, values, kv_group_size):
            return None
        return kv_heads * sum(_row_bytes(cache, kv_group_size, dtype) for cache in (keys, values))
    except Exception as exc:
        logger.debug("MLX estimate cannot route the quantized attention: %s", exc)
        return None


def _compute_bytes(
    widths,
    dtype_size: int,
    prefill_chunk: int,
    plan,
    n_ctx: int,
    quant_boundary,
    kv_bits = None,
    kv_group_size: int = _KV_GROUP_SIZE,
) -> int:
    """Transient buffers a prefill step holds, on top of weights and cache."""
    hidden, intermediate, heads = widths[:3]
    if hidden <= 0:
        return 0
    intermediate = intermediate or 4 * hidden
    attention_width = (hidden + intermediate) * dtype_size * _ATTENTION_LIVE
    recurrent_width = sum(int(entry["conv_width"]) for entry in plan) * _RECURRENT_DTYPE_SIZE
    total = _COMPUTE_BASE_BYTES + prefill_chunk * (attention_width + recurrent_width)
    if quant_boundary is not None:
        score = heads * dtype_size * _SCORE_LIVE
        row = _dequantized_row_bytes(widths, dtype_size, kv_bits, kv_group_size, prefill_chunk)
        if row is None:
            total += _widest_quantized_scores(n_ctx, quant_boundary, prefill_chunk) * score
        else:
            total += max(row * n_ctx, (n_ctx + 1) * score)
    elif not any(entry.get("bound_spec") for entry in plan):
        # Windowed models are skipped: the config's head_dim need not be the full-context layers' width.
        score = heads * dtype_size * _SCORE_LIVE
        total += _widest_unfused_scores(widths, dtype_size, n_ctx, prefill_chunk) * score
    return int(total)


def _widest_unfused_scores(widths, dtype_size: int, n_ctx: int, prefill_chunk: int) -> int:
    """Query x key cells of the widest call mlx leaves unfused over every prompt the context admits."""
    heads, kv_heads, head_dim = (*widths[2:], 0, 0, 0)[:3]
    route = _zoo_route() if heads and kv_heads and head_dim else None
    if route is None:
        return 0
    widths, chunk = tuple(widths), prefill_chunk

    def unfused(queries, keys):
        return not _attention_fuses(route, widths, dtype_size, queries, keys)

    def longest_tail(whole, limit):
        lengths = _unfused_tails(route, widths, dtype_size, chunk)[min(whole, 1)]
        if limit > _TAIL_SCAN and unfused(limit, limit + whole * chunk):
            return limit
        return max((r for r in lengths if r <= limit), default = 0)

    whole, rest = divmod(max(n_ctx - 1, 0), chunk)
    cells = [n_ctx + 1 if unfused(1, 2) or unfused(1, n_ctx + 1) else 0]
    if whole and unfused(chunk, chunk):
        cells.append(chunk * chunk)
    if whole >= 2 and unfused(chunk, whole * chunk):
        cells.append(chunk * whole * chunk)
    r = longest_tail(whole, rest)
    cells.append(r * (whole * chunk + r))
    if whole:
        r = longest_tail(whole - 1, chunk - 1)
        cells.append(r * ((whole - 1) * chunk + r))
    return max(cells)


# mlx's fused-kernel thresholds all sit below this, so longer tails route as this one does.
_TAIL_SCAN = 4096


@functools.lru_cache(maxsize = 64)
def _unfused_tails(route, widths, dtype_size: int, chunk: int):
    """Tail lengths mlx leaves unfused, as the first call (square) and after a whole chunk."""
    span = range(1, min(chunk - 1, _TAIL_SCAN) + 1)
    return tuple(
        tuple(r for r in span if not _attention_fuses(route, widths, dtype_size, r, r + later))
        for later in (0, chunk)
    )


def _zoo_route():
    try:
        import mlx.core  # noqa: F401

        from unsloth_zoo.mlx.attention import fused_kernel_exists
    except Exception:
        return None
    return fused_kernel_exists


def _attention_fuses(route, widths, dtype_size: int, queries: int, keys: int) -> bool:
    """Whether mlx fuses a full-width causal call; elsewhere it materializes the scores."""
    import mlx.core as mx

    _heads, kv_heads, head_dim, value_dim = (*widths[2:], 0, 0, 0, 0)[:4]
    try:
        dtype = mx.float32 if dtype_size == 4 else mx.bfloat16

        # zoo reads only shapes and dtypes; a group of one makes the scales the cache's width.
        def cache(width):
            scales = mx.zeros((1, kv_heads, keys, width), dtype = dtype)
            return scales, scales, scales

        q = mx.zeros((1, widths[2], queries, head_dim), dtype = dtype)
        return bool(route(q, cache(head_dim), cache(value_dim or head_dim), 1, "causal"))
    except Exception as exc:
        logger.debug("MLX estimate cannot route the attention: %s", exc)
        return True


def _load_is_refused(model_dir: str, config: dict, load_in_4bit: bool) -> Optional[str]:
    try:
        from unsloth_zoo.mlx.loader import (
            _ensure_quantization_compatible,
            _get_existing_mlx_quantization,
            _resolve_mlx_quantization_spec,
        )
    except Exception:
        return None

    import contextlib
    import io

    existing = _get_existing_mlx_quantization(config)
    if isinstance(existing, dict) and existing.get("quant_method") == "bitsandbytes":
        return "bitsandbytes weights mlx-lm cannot read"

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec = _resolve_mlx_quantization_spec(
                load_in_4bit = bool(load_in_4bit),
                load_in_8bit = False,
                load_in_16bit = False,
                load_in_fp8 = False,
                load_in_mxfp4 = False,
                load_in_nvfp4 = False,
                full_finetuning = False,
                q_bits = None,
                q_group_size = None,
                q_mode = None,
                mlx_quantization_config = None,
                quantization_config = None,
                quant_predicate = None,
                quantize_modules = None,
                force_requantize = False,
            )
            _ensure_quantization_compatible(config, spec, "this checkpoint")
    except ValueError as exc:
        return str(exc)
    except Exception:
        return None

    return _refused_by_its_own_tensors(model_dir, config)


def _extra_tensors_refused(model, config: dict, extras: list) -> Optional[str]:
    try:
        from unsloth_zoo.mlx.loader import (
            _KNOWN_MLX_LM_STRICT_FALLBACKS,
            _KNOWN_VLM_EXTRA_WEIGHT_FILTERS,
            _gemma4_unused_shared_kv_weight,
            _message_matches_known_fallback,
            _raise_if_qk_norm_version_gap,
        )
    except Exception:
        return None

    model_type = config.get("model_type")
    message = f"Received {len(extras)} parameters not in model: \n{extras}."
    try:
        _raise_if_qk_norm_version_gap(model_type, message, ValueError(message))
    except ValueError as refusal:
        return str(refusal)

    vision = _builds_on_mlx_vlm(config)
    table = _KNOWN_VLM_EXTRA_WEIGHT_FILTERS if vision else _KNOWN_MLX_LM_STRICT_FALLBACKS
    rule = table.get(model_type)
    if rule is not None and _message_matches_known_fallback(message, rule):
        allowed = rule.get("allowed_extra", frozenset())
        shared_kv = rule.get("allow_shared_kv", False)
        extras = [
            name
            for name in extras
            if name not in allowed
            and not (shared_kv and _gemma4_unused_shared_kv_weight(model, name))
        ]
        if not extras:
            return None

    return f"{len(extras)} tensors this architecture has no home for, " f"beginning {extras[0]}"


def _refused_by_its_own_tensors(model_dir: str, config: dict) -> Optional[str]:
    """Refusals only the checkpoint's own tensors reveal."""
    try:
        from mlx.utils import tree_flatten
        from unsloth_zoo.mlx.loader import _raise_if_qk_norm_version_gap
    except Exception:
        return None

    import contextlib
    import io

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dtype = _runtime_dtype()
            model = _whole_model(config, dtype)
            tensors = _checkpoint_tensors(model_dir, config, dtype)
            if not tensors:
                return None
            if hasattr(model, "sanitize"):
                tensors = model.sanitize(tensors)
            built = {name for name, _ in tree_flatten(model.parameters())}
    except Exception:
        return None

    unhoused = sorted(
        name for name in set(tensors) - built if not name.endswith((".scales", ".biases"))
    )
    if unhoused:
        refusal = _extra_tensors_refused(model, config, unhoused)
        if refusal:
            return refusal

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model.load_weights(list(tensors.items()), strict = True)
    except Exception as exc:
        try:
            _raise_if_qk_norm_version_gap(config.get("model_type"), str(exc), exc)
        except ValueError as refusal:
            return str(refusal)
        except Exception:
            return None
    return None


@functools.lru_cache(maxsize = 1)
def mlx_offload_planner():
    """unsloth_zoo's offload planner, or None when the installed unsloth_zoo cannot offload as
    Studio prices it."""
    import inspect

    try:
        from unsloth_zoo.mlx import offload
        from unsloth_zoo.mlx.loader import FastMLXModel
        hook = "offload" in inspect.signature(FastMLXModel.from_pretrained).parameters
    except Exception:
        return None
    if not (hook and hasattr(offload, "plan_offload") and hasattr(offload, "offloadable")):
        return None
    return offload


def _lazy_checkpoint(model_dir: str, config: dict):
    """``model_dir``'s model on the route a load takes, its weights mapped but never read."""
    from pathlib import Path

    # mlx-vlm imports a checkpoint's own module unasked; pricing must never run it.
    if config.get("model_file"):
        raise ValueError("this checkpoint carries its own model module")
    # Refuses an index naming a shard outside the snapshot, as every other read here does.
    _indexed_shards(model_dir)
    # Not strict: unsloth_zoo loads past known unused tensors (Gemma 4's). Weightless: mlx-vlm's
    # checkpoint conversion (ModelOpt FP8) evaluates its dense weights even on a lazy load.
    with weightless_forward():
        if _builds_on_mlx_vlm(config):
            from mlx_vlm.utils import load_model

            _install_oq_renames(config)
            view, owner = _loader_view(model_dir, config)
            if owner is None:
                return load_model(Path(view), lazy = True, strict = False)
            model = owner.call(load_model, Path(view), lazy = True, strict = False)
            owner.transfer_to(model)
            return model
        from mlx_lm.utils import load_model
        return load_model(Path(model_dir), lazy = True, strict = False)[0]


def _loader_view(model_dir: str, config: dict) -> tuple:
    """``(path, owner)``: unsloth_zoo's config view of ``model_dir`` (owner None without unsloth_zoo)."""
    import io

    try:
        from unsloth_zoo.mlx.loader import (
            _MLXVLMConfigViewOwner,
            _materialize_mlx_vlm_config_override,
        )
    except Exception:
        return model_dir, None
    with contextlib.redirect_stdout(io.StringIO()):
        view, _ = _materialize_mlx_vlm_config_override(model_dir, config)
    return view, _MLXVLMConfigViewOwner(view, model_dir)


def _install_oq_renames(config: dict) -> None:
    """unsloth_zoo's sanitize renames for DeepSeek-V4.1 oQ quants, as its load installs them."""
    try:
        from unsloth_zoo.mlx.loader import _ensure_deepseek_v41_omlx_layout
    except Exception:
        return
    _ensure_deepseek_v41_omlx_layout(config.get("model_type"))


@functools.lru_cache(maxsize = 4)
def _offloadable_cached(model_dir: str, fingerprint: tuple, cast_dtype, on_mlx_vlm: bool):
    from unsloth_zoo.mlx.offload import offloadable
    return offloadable(
        _lazy_checkpoint(model_dir, _snapshot_config(model_dir)), model_dir, cast_dtype = cast_dtype
    )


def _offloadable(model_dir: str, config: dict) -> dict:
    """What unsloth_zoo's offload can take from this checkpoint, sized as the rest of the weights
    are: shards as stored, else at the runtime width."""
    quantized = config.get("quantization") or config.get("quantization_config")
    return _offloadable_cached(
        os.path.realpath(model_dir),
        _checkpoint_fingerprint(model_dir),
        None if quantized else _runtime_dtype(),
        _builds_on_mlx_vlm(config),
    )


def mlx_offloadable_parts(model_dir: str, vision: Optional[bool] = None) -> tuple:
    if mlx_offload_planner() is None:
        return ()
    with _on_route(vision):
        found = _offloadable(model_dir, _snapshot_config(model_dir) or {})
    return tuple(key for key in ("ple", "experts") if found[key] is None)


def _offloaded(model_dir: str, config: dict, offload: Optional[tuple]) -> tuple:
    """``(weights change, (ple, experts) applied)`` of reading ``offload``'s fractions from disk: what
    unsloth_zoo's offload can take swaps its size for what it holds offloaded, and a knob with
    nothing to take applies nothing."""
    planner = mlx_offload_planner()
    if not offload or not any(offload) or planner is None:
        return 0, (None, None)
    found = _offloadable(model_dir, config)
    applied = tuple(
        value if found[key] is None else None for value, key in zip(offload, ("ple", "experts"))
    )
    delta = (
        int(planner.offload_footprint(found, *applied)) - found["ple_bytes"] - found["expert_bytes"]
    )
    return delta, applied


@dataclass
class _MlxSizing:
    """What a load costs before a context is named, so a search over contexts probes once."""

    weights: int
    dtype: object
    plan: list
    quant_start: Optional[int]
    facts: dict
    widths: tuple
    chunk: int
    kv_bits: Optional[int]
    kv_group_size: int
    drafter_weights: int = 0
    drafter_plan: tuple = ()
    capture_per_token: int = 0
    offload: tuple = (None, None)


_SIZING_CACHE_MAX = 16
_sizing_cache: "OrderedDict[tuple, Optional[_MlxSizing]]" = OrderedDict()
_sizing_lock = threading.Lock()


def _checkpoint_fingerprint(model_dir: str) -> tuple:
    """Size and mtime of every config, index and shard file, so a changed download re-sizes."""
    root = os.path.realpath(model_dir)
    found = []
    for pattern in ("*.json", "**/*.safetensors"):
        for path in sorted(glob.glob(os.path.join(root, pattern), recursive = True)):
            try:
                stat = os.stat(path)
            except OSError:
                continue
            found.append((os.path.relpath(path, root), stat.st_size, stat.st_mtime_ns))
    return tuple(found)


def _size_load(
    model_dir: str,
    kv_bits: Optional[int],
    prefill_chunk: Optional[int],
    load_in_4bit: bool,
    vision: Optional[bool] = None,
    drafter: Optional[tuple] = None,
    offload: Optional[tuple] = None,
) -> Optional[_MlxSizing]:
    """Cached ``_size_load_uncached``: every estimate prices the fit and the breakdown, and every
    slider step prices again, each otherwise rebuilding the same weightless architecture."""
    try:
        key = (
            os.path.realpath(model_dir),
            kv_bits,
            prefill_chunk,
            bool(load_in_4bit),
            os.environ.get("UNSLOTH_MLX_PROMPT_CACHE_BYTES"),
            _checkpoint_fingerprint(model_dir),
            vision,
            drafter,
            drafter and _checkpoint_fingerprint(drafter[0]),
            offload,
        )
    except Exception:
        return _size_load_uncached(
            model_dir, kv_bits, prefill_chunk, load_in_4bit, vision, drafter, offload
        )
    with _sizing_lock:
        if key in _sizing_cache:
            _sizing_cache.move_to_end(key)
            return _sizing_cache[key]
    sizing = _size_load_uncached(
        model_dir, kv_bits, prefill_chunk, load_in_4bit, vision, drafter, offload
    )
    with _sizing_lock:
        _sizing_cache[key] = sizing
        while len(_sizing_cache) > _SIZING_CACHE_MAX:
            _sizing_cache.popitem(last = False)
    return sizing


def _clear_sizing_cache() -> None:
    with _sizing_lock:
        _sizing_cache.clear()
    _offloadable_cached.cache_clear()


def _size_load_uncached(
    model_dir: str,
    kv_bits: Optional[int],
    prefill_chunk: Optional[int],
    load_in_4bit: bool,
    vision: Optional[bool] = None,
    drafter: Optional[tuple] = None,
    offload: Optional[tuple] = None,
) -> Optional[_MlxSizing]:
    token = _ROUTE_VISION.set(vision)
    try:
        return _sized(model_dir, kv_bits, prefill_chunk, load_in_4bit, drafter, offload)
    finally:
        _ROUTE_VISION.reset(token)


def _sized(
    model_dir: str,
    kv_bits: Optional[int],
    prefill_chunk: Optional[int],
    load_in_4bit: bool,
    drafter: Optional[tuple],
    offload: Optional[tuple] = None,
) -> Optional[_MlxSizing]:
    """Everything about a load that does not move with the context, or None if it cannot be sized.
    ``offload`` is the ``(ple, experts)`` fractions the load reads from disk."""
    config = _snapshot_config(model_dir)
    if config is None:
        return None
    refusal = _load_is_refused(model_dir, config, load_in_4bit)
    if refusal is not None:
        logger.debug("MLX estimate refuses %s: %s", model_dir, refusal)
        return None
    try:
        weights = mlx_weight_bytes(model_dir, config, load_in_4bit)
    except Exception as exc:
        logger.debug("MLX estimate cannot read the shards of %s: %s", model_dir, exc)
        return None
    if weights <= 0:
        return None
    try:
        delta, offload = _offloaded(model_dir, config, offload)
        weights += delta
    except Exception as exc:
        logger.debug("MLX estimate cannot size the offload of %s: %s", model_dir, exc)
        return None
    try:
        dtype = _runtime_dtype()
        loaded_chunk, loaded_group = _generation_settings(config)
        plan, quant_start, facts = _cache_plan(config, dtype, kv_bits)
        widths = tuple(
            tower or checkpoint
            for tower, checkpoint in zip(facts["widths"], _config_widths(config))
        )
        chunk = prefill_chunk or loaded_chunk
        drafted = (0, (), 0) if drafter is None else _drafter_sizing(config, dtype, drafter)
    except Exception as exc:
        logger.debug(
            "MLX estimate could not size %s (%s): %s", model_dir, config.get("model_type"), exc
        )
        return None
    return _MlxSizing(
        weights = weights,
        dtype = dtype,
        plan = plan,
        quant_start = quant_start,
        facts = facts,
        widths = widths,
        chunk = chunk,
        kv_bits = kv_bits,
        kv_group_size = loaded_group,
        drafter_weights = drafted[0],
        drafter_plan = tuple(drafted[1]),
        capture_per_token = drafted[2],
        offload = offload,
    )


def _priced_at(sizing: _MlxSizing, n_ctx: int) -> Optional[MlxMemoryBreakdown]:
    try:
        context = max(int(n_ctx or 0), 1)
        whole_prompt = sizing.facts["whole_prompt"]
        chunk = context if whole_prompt else sizing.chunk
        kv, quant_boundary = _kv_bytes(sizing.plan, context, sizing.quant_start, chunk)
        compute = _compute_bytes(
            sizing.widths,
            sizing.dtype.size,
            chunk,
            sizing.plan,
            context,
            quant_boundary,
            sizing.kv_bits,
            sizing.kv_group_size,
        )
        drafter_kv = _line_bytes(list(sizing.drafter_plan), context, chunk, quantized = False)
    except Exception as exc:
        logger.debug("MLX estimate could not price %s tokens: %s", n_ctx, exc)
        return None
    capture = sizing.capture_per_token * context
    total = sizing.weights + sizing.drafter_weights + kv + drafter_kv + compute + capture
    return MlxMemoryBreakdown(
        weights_bytes = sizing.weights + sizing.drafter_weights,
        kv_bytes = kv + drafter_kv,
        compute_bytes = compute + capture,
        total_bytes = total,
        gpu_bytes = total,
        n_ctx = context,
        layer_count = sizing.facts["layers"],
        cache_type_kv = _cache_width_name(
            sizing.plan,
            sizing.kv_bits,
            quant_boundary is not None,
            _width_name(sizing.dtype),
            context,
            chunk,
        ),
        ple_offload = sizing.offload[0],
        expert_offload = sizing.offload[1],
    )


def mlx_memory_breakdown(
    model_dir: str,
    *,
    n_ctx: int,
    kv_bits: Optional[int] = None,
    prefill_chunk: Optional[int] = None,
    load_in_4bit: bool = False,
    vision: Optional[bool] = None,
    drafter: Optional[tuple] = None,
    offload: Optional[tuple] = None,
) -> Optional[MlxMemoryBreakdown]:
    """Price an MLX load, or None when it cannot honestly be sized. ``vision`` overrides the loader route
    the config implies; ``drafter`` is the ``(path, builtin)`` drafter a speculative load attaches;
    ``offload`` the ``(ple, experts)`` fractions it reads from disk."""
    sizing = _size_load(model_dir, kv_bits, prefill_chunk, load_in_4bit, vision, drafter, offload)
    return None if sizing is None else _priced_at(sizing, n_ctx)


def mlx_fit_outcome(
    model_dir: str,
    *,
    budget_bytes: int,
    max_ctx: int,
    min_ctx: int = MLX_FIT_MIN_CONTEXT,
    kv_bits: Optional[int] = None,
    prefill_chunk: Optional[int] = None,
    load_in_4bit: bool = False,
    vision: Optional[bool] = None,
    drafter: Optional[tuple] = None,
    offload: Optional[tuple] = None,
) -> tuple:
    """``("unsizable" | "fits" | "fitted" | "no_fit", context)``: the largest context inside
    ``budget_bytes`` for ``"fitted"``, else None. An offloaded load keeps unsloth_zoo's offload
    safety margin of the budget free, as its offload planner does."""
    sizing = _size_load(model_dir, kv_bits, prefill_chunk, load_in_4bit, vision, drafter, offload)
    if sizing is None:
        return "unsizable", None
    budget_bytes = _fit_budget(sizing, budget_bytes)
    ceiling = max(int(max_ctx or 0), MLX_KV_BLOCK)
    at_ceiling = _priced_at(sizing, ceiling)
    if at_ceiling is None:
        return "unsizable", None
    if at_ceiling.total_bytes <= budget_bytes:
        return "fits", None
    floor = max(-(-int(min_ctx or 0) // MLX_KV_BLOCK) * MLX_KV_BLOCK, MLX_KV_BLOCK)
    low, high, best = floor, ceiling, None
    while low <= high:
        middle = (low + high) // 2
        priced = _priced_at(sizing, middle)
        if priced is None:
            return "unsizable", None
        if priced.total_bytes <= budget_bytes:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return ("no_fit", None) if best is None else ("fitted", (best // MLX_KV_BLOCK) * MLX_KV_BLOCK)


def _fit_budget(sizing: _MlxSizing, budget_bytes: int) -> float:
    if not any(sizing.offload):
        return budget_bytes
    least, share = mlx_offload_planner().SAFETY_MARGIN
    return min(budget_bytes - least, budget_bytes / (1 + share))


def mlx_headroom(
    model_dir: str,
    *,
    budget_bytes: int,
    n_ctx: int,
    kv_bits: Optional[int] = None,
    prefill_chunk: Optional[int] = None,
    load_in_4bit: bool = False,
    vision: Optional[bool] = None,
    drafter: Optional[tuple] = None,
    offload: Optional[tuple] = None,
) -> Optional[int]:
    """Bytes of ``budget_bytes`` a load at ``n_ctx`` leaves free (as ``mlx_fit_outcome`` prices it), None if unsized."""
    sizing = _size_load(model_dir, kv_bits, prefill_chunk, load_in_4bit, vision, drafter, offload)
    priced = None if sizing is None else _priced_at(sizing, max(int(n_ctx or 0), MLX_KV_BLOCK))
    if priced is None:
        return None
    return max(int(_fit_budget(sizing, budget_bytes) - priced.total_bytes), 0)


def mlx_offload_plan(
    model_dir: str,
    *,
    budget_bytes: int,
    max_ctx: int,
    context: Optional[int] = None,
    ple = "auto",
    experts = "auto",
    kv_bits: Optional[int] = None,
    prefill_chunk: Optional[int] = None,
    load_in_4bit: bool = False,
    vision: Optional[bool] = None,
    drafter: Optional[tuple] = None,
) -> Optional[tuple]:
    """The ``(ple, experts, prefill_step)`` unsloth_zoo's 50/50 rule takes for this load, priced
    as Studio prices it, with ``drafter`` when given; a knob not ``"auto"`` and a pinned ``context``
    are kept. When nothing fits, everything offloadable goes to disk. None when the load cannot be planned."""
    planner = mlx_offload_planner()
    sizing = _size_load(model_dir, kv_bits, prefill_chunk, load_in_4bit, vision, drafter)
    if sizing is None and drafter is not None:
        # A drafter that cannot be sized cannot attach either: plan the target as if none was asked for.
        drafter = None
        sizing = _size_load(model_dir, kv_bits, prefill_chunk, load_in_4bit, vision)
    if planner is None or sizing is None:
        return None
    with _on_route(vision):
        found = _offloadable(model_dir, _snapshot_config(model_dir))
    sized = {None: sizing, sizing.chunk: sizing}

    def context_cost(tokens, step):
        if step not in sized:
            sized[step] = _size_load(model_dir, kv_bits, step, load_in_4bit, vision, drafter)
        priced = sized[step] and _priced_at(sized[step], tokens)
        if priced is None:
            raise ValueError(f"cannot price {tokens} tokens")
        return priced.total_bytes - sizing.weights

    plan = planner.plan_offload(
        budget = budget_bytes,
        resident = sizing.weights - found["ple_bytes"] - found["expert_bytes"],
        inventory = found,
        context_cost = context_cost,
        ceiling = max_ctx,
        minimum = MLX_FIT_MIN_CONTEXT,
        context = context,
        prefill_step = None if experts and found["expert_bytes"] else sizing.chunk,
        offload = {"ple": ple, "experts": experts},
    )
    if plan["context"] is None:
        return tuple(
            (1.0 if found[key] is None else None) if value == "auto" else value
            for value, key in ((ple, "ple"), (experts, "experts"))
        ) + (None,)
    return plan["ple"], plan["experts"], plan.get("prefill_step") if plan["experts"] else None


def mlx_fit_context(model_dir: str, **kwargs) -> Optional[int]:
    """Largest context whose estimated footprint stays inside ``budget_bytes``; None when the ceiling
    already fits or nothing can be said."""
    return mlx_fit_outcome(model_dir, **kwargs)[1]
