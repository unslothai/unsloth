# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What an MLX load would occupy, priced before anything is loaded."""

from __future__ import annotations

import glob
import functools
import json
import os
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
    "mlx_memory_breakdown",
    "mlx_shard_files",
    "mlx_weight_bytes",
]

MLX_KV_BLOCK = 256

_PROBE_SHORT = 8
_PROBE_LONG = 40

MLX_PREFILL_CHUNK = 2048

MLX_FIT_MIN_CONTEXT = 4096
_KV_GROUP_SIZE = 64

_ATTENTION_LIVE = 3.5
_COMPUTE_BASE_BYTES = int(0.64 * 1024**3)

_RECURRENT_DTYPE_SIZE = 4

_QUANT_SCORE_DTYPE_SIZE = 4

_QUANT_SCORE_LIVE = 1.0

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
    if _loads_as_vision(config):
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
        with open(shard, "rb") as handle:
            length = int.from_bytes(handle.read(8), "little")
            limit = min(_MAX_SAFETENSORS_HEADER, os.fstat(handle.fileno()).st_size - 8)
            if not 0 < length <= limit:
                raise ValueError(f"{shard} declares a {length}-byte safetensors header")
            header = json.loads(handle.read(length))
        for name, meta in header.items():
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
    import io

    from mlx.utils import tree_flatten
    from unsloth_zoo.mlx.loader import (
        _apply_mlx_quantization,
        _resolve_mlx_quantization_spec,
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
                is_vlm = _loads_as_vision(config),
            )
    return sum(
        value.nbytes for _, value in tree_flatten(model.parameters()) if hasattr(value, "nbytes")
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
        return shards
    try:
        return _resident_bytes(model_dir, config, quantize = load_in_4bit)
    except Exception as exc:
        logger.debug("MLX estimate priced %s as stored: %s", model_dir, exc)
        return shards


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


def _loads_as_vision(config: dict) -> bool:
    from types import SimpleNamespace
    try:
        from utils.models.model_config import _is_vlm
    except Exception:
        return any(
            key in config
            for key in ("vision_config", "img_processor", "image_token_index", "projector_config")
        )
    return bool(_is_vlm(SimpleNamespace(**config)))


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
    if _loads_as_vision(config):
        from mlx_vlm.utils import get_model_and_args
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

    if not _loads_as_vision(config):
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

    state = entry.state
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
    return sum(
        int(item.shape[-1]) for item in state if isinstance(item, mx.array) and item.ndim == 3
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


def _probe(config: dict, dtype, n_tokens: int, kv_bits):
    import mlx.core as mx

    cache = None
    whole_prompt = False
    layers = None
    widths = (0,) * len(_config_widths({}))
    failure = ValueError("no architecture module could build this config")
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
    span = _PROBE_LONG - _PROBE_SHORT

    def solve(key):
        return [
            (n[key] - (f[key] - n[key]) / span * _PROBE_SHORT, (f[key] - n[key]) / span)
            for n, f in zip(near["entries"], far["entries"])
        ]

    plan = []
    for slot, (const, slope), (q_const, q_slope) in zip(
        near["entries"], solve("bytes"), solve("quant_bytes")
    ):
        plan.append(
            {
                "const": const,
                "slope": slope,
                "quant_const": q_const,
                "quant_slope": q_slope,
                "bound_spec": slot["bound_spec"],
                "conv_width": slot["conv_width"],
                "block": slot["block"],
                "converts": slot["converts"] and near["quantized"],
            }
        )
    return (
        plan,
        0 if near["quantized"] else None,
        {
            "whole_prompt": near["whole_prompt"],
            "layers": near["layers"],
            "widths": near["widths"],
        },
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
        score = heads * _QUANT_SCORE_DTYPE_SIZE * _QUANT_SCORE_LIVE
        row = _dequantized_row_bytes(widths, dtype_size, kv_bits, kv_group_size, prefill_chunk)
        if row is None:
            total += _widest_quantized_scores(n_ctx, quant_boundary, prefill_chunk) * score
        else:
            total += max(row * n_ctx, (n_ctx + 1) * score)
    return int(total)


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

    vision = _loads_as_vision(config)
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
    model_dir: str, kv_bits: Optional[int], prefill_chunk: Optional[int], load_in_4bit: bool
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
        )
    except Exception:
        return _size_load_uncached(model_dir, kv_bits, prefill_chunk, load_in_4bit)
    with _sizing_lock:
        if key in _sizing_cache:
            _sizing_cache.move_to_end(key)
            return _sizing_cache[key]
    sizing = _size_load_uncached(model_dir, kv_bits, prefill_chunk, load_in_4bit)
    with _sizing_lock:
        _sizing_cache[key] = sizing
        while len(_sizing_cache) > _SIZING_CACHE_MAX:
            _sizing_cache.popitem(last = False)
    return sizing


def _clear_sizing_cache() -> None:
    with _sizing_lock:
        _sizing_cache.clear()


def _size_load_uncached(
    model_dir: str, kv_bits: Optional[int], prefill_chunk: Optional[int], load_in_4bit: bool
) -> Optional[_MlxSizing]:
    """Everything about a load that does not move with the context, or None if it cannot be sized."""
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
        dtype = _runtime_dtype()
        loaded_chunk, loaded_group = _generation_settings(config)
        plan, quant_start, facts = _cache_plan(config, dtype, kv_bits)
        widths = tuple(
            tower or checkpoint
            for tower, checkpoint in zip(facts["widths"], _config_widths(config))
        )
        chunk = prefill_chunk or loaded_chunk
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
    except Exception as exc:
        logger.debug("MLX estimate could not price %s tokens: %s", n_ctx, exc)
        return None
    total = sizing.weights + kv + compute
    return MlxMemoryBreakdown(
        weights_bytes = sizing.weights,
        kv_bytes = kv,
        compute_bytes = compute,
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
    )


def mlx_memory_breakdown(
    model_dir: str,
    *,
    n_ctx: int,
    kv_bits: Optional[int] = None,
    prefill_chunk: Optional[int] = None,
    load_in_4bit: bool = False,
) -> Optional[MlxMemoryBreakdown]:
    """Price an MLX load, or None when it cannot honestly be sized."""
    sizing = _size_load(model_dir, kv_bits, prefill_chunk, load_in_4bit)
    return None if sizing is None else _priced_at(sizing, n_ctx)


def mlx_fit_context(
    model_dir: str,
    *,
    budget_bytes: int,
    max_ctx: int,
    min_ctx: int = MLX_FIT_MIN_CONTEXT,
    kv_bits: Optional[int] = None,
    prefill_chunk: Optional[int] = None,
    load_in_4bit: bool = False,
) -> Optional[int]:
    """Largest context whose estimated footprint stays inside ``budget_bytes``."""
    sizing = _size_load(model_dir, kv_bits, prefill_chunk, load_in_4bit)
    if sizing is None:
        return None
    ceiling = max(int(max_ctx or 0), MLX_KV_BLOCK)
    at_ceiling = _priced_at(sizing, ceiling)
    if at_ceiling is None or at_ceiling.total_bytes <= budget_bytes:
        return None
    floor = max(-(-int(min_ctx or 0) // MLX_KV_BLOCK) * MLX_KV_BLOCK, MLX_KV_BLOCK)
    low, high, best = floor, ceiling, None
    while low <= high:
        middle = (low + high) // 2
        priced = _priced_at(sizing, middle)
        if priced is None:
            return None
        if priced.total_bytes <= budget_bytes:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return None if best is None else (best // MLX_KV_BLOCK) * MLX_KV_BLOCK
