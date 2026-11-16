# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pre-quantized bitsandbytes 4-bit checkpoints under sharded loading (tensor / pipeline parallel, native FSDP2).

transformers builds the model on meta with its own Linear4bit replacement (no weight is read), then every rank
fills it from the safetensors checkpoint: the packed 4-bit bytes it keeps (all, a dim-0 byte range, a tensor
parallel slice, or none) and the QuantState statistics of what it keeps (absmax and maps, 1/64 of the bytes)."""

from __future__ import annotations

import contextlib

import torch

__all__ = [
    "bnb4_quantization",
    "is_linear4bit",
    "quantized_weights",
    "prequantized_skeleton",
]

_STAT_SUFFIXES = (
    "absmax",
    "quant_map",
    "nested_absmax",
    "nested_quant_map",
    "quant_state.bitsandbytes__nf4",
    "quant_state.bitsandbytes__fp4",
)
_SAFETENSORS_DTYPES = {
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
_LOAD_ONLY_KWARGS = (
    "cache_dir",
    "force_download",
    "proxies",
    "local_files_only",
    "token",
    "revision",
    "subfolder",
    "trust_remote_code",
    "device_map",
    "dtype",
    "torch_dtype",
    "attn_implementation",
    "quantization_config",
    "load_in_4bit",
    "load_in_8bit",
    "low_cpu_mem_usage",
    "key_mapping",
    "use_safetensors",
    "weights_only",
    "max_memory",
    "offload_folder",
    "offload_state_dict",
    "offload_buffers",
    "output_loading_info",
    "state_dict",
    "adapter_kwargs",
    "adapter_name",
    "use_kernels",
    "kernel_config",
    "_from_auto",
    "_from_pipeline",
    "_commit_hash",
    "code_revision",
    "mirror",
)
_HUB_KWARGS = (
    "cache_dir",
    "force_download",
    "proxies",
    "local_files_only",
    "token",
    "revision",
    "subfolder",
    "trust_remote_code",
)


def _as_dict(qc):
    if qc is None:
        return None
    if isinstance(qc, dict):
        return qc
    to_dict = getattr(qc, "to_dict", None)
    return to_dict() if callable(to_dict) else dict(vars(qc))


def bnb4_quantization(config):
    """The quantization_config (dict) of a pre-quantized bitsandbytes 4-bit checkpoint config, else None."""
    text = getattr(config, "get_text_config", None)
    for c in (config, text() if callable(text) else None):
        qc = _as_dict(getattr(c, "quantization_config", None))
        if not qc:
            continue
        method = qc.get("quant_method", "")
        method = str(getattr(method, "value", method)).lower()
        if method == "bitsandbytes" and (qc.get("load_in_4bit") or qc.get("_load_in_4bit")):
            return qc
    return None


def is_linear4bit(module):
    return any(
        c.__name__ == "Linear4bit" and c.__module__.startswith("bitsandbytes")
        for c in type(module).__mro__
    )


def quantized_weights(model):
    return {
        id(m._parameters["weight"]): m
        for m in model.modules()
        if is_linear4bit(m) and m._parameters.get("weight") is not None
    }


def _hub_config(cls, name, kwargs):
    """The checkpoint config as from_pretrained resolves it (kwargs naming config attributes update it)."""
    config = kwargs.get("config")
    from transformers import PretrainedConfig

    if isinstance(config, PretrainedConfig):
        return config
    extra = {k: v for k, v in kwargs.items() if k not in _LOAD_ONLY_KWARGS and k != "config"}
    hub = {k: kwargs[k] for k in _HUB_KWARGS if k in kwargs}
    source = config if isinstance(config, str) else name
    config, unused = cls.config_class.from_pretrained(
        source, return_unused_kwargs = True, **hub, **extra
    )
    for key in unused:
        kwargs.pop(key, None)
    for key in extra:
        if key not in unused:
            kwargs.pop(key, None)
    return config


def checkpoint_quantization(cls, name, kwargs):
    """(config, bnb 4-bit quantization dict or None) for `name`; kwargs lose the config attributes they set."""
    config = _hub_config(cls, name, kwargs)
    return config, bnb4_quantization(config)


@contextlib.contextmanager
def _no_missing_key_init():
    from transformers import PreTrainedModel
    import transformers

    original = PreTrainedModel.__dict__.get("_initialize_missing_keys")
    verbosity = transformers.logging.get_verbosity()
    if original is not None:
        PreTrainedModel._initialize_missing_keys = lambda self, *args, **kwargs: None
    transformers.logging.set_verbosity_error()
    try:
        yield
    finally:
        transformers.logging.set_verbosity(verbosity)
        if original is not None:
            PreTrainedModel._initialize_missing_keys = original


def meta_state_dict(reader):
    packed = QuantReads(reader, {}).packed_keys()
    out = {}
    for key in reader.index:
        if key in packed or any(key.endswith("." + s) for s in _STAT_SUFFIXES):
            continue
        view = reader._open(key).get_slice(key)
        out[key] = torch.empty(
            tuple(view.get_shape()), dtype = _SAFETENSORS_DTYPES[view.get_dtype()], device = "meta"
        )
    return out


def prequantized_skeleton(original, cls, name, args, kwargs, config, state_dict):
    """transformers' own model for a pre-quantized bitsandbytes 4-bit checkpoint, on meta, nothing read.
    `state_dict` holds the non-quantized tensors as meta tensors: transformers' loading (and any patch of it)
    gives each its parameter dtype, as the full load does; the 4-bit weights stay unloaded."""
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        not in (
            "config",
            "device_map",
            "state_dict",
            "load_in_4bit",
            "load_in_8bit",
            "key_mapping",
            "low_cpu_mem_usage",
        )
    }
    with torch.random.fork_rng(devices = []), _no_missing_key_init():
        model = original(
            cls, None, *args, config = config, state_dict = state_dict, device_map = "meta", **kwargs
        )
    model.name_or_path = name
    model.config._name_or_path = name
    if callable(getattr(model, "can_generate", None)) and model.can_generate():
        try:
            from transformers import GenerationConfig
            hub = {k: kwargs[k] for k in _HUB_KWARGS if k in kwargs and k != "trust_remote_code"}
            model.generation_config = GenerationConfig.from_pretrained(name, **hub)
        except Exception:
            pass
    return model


class QuantReads:
    def __init__(self, reader, quantized):
        self.reader, self.quantized = reader, quantized
        self._stats = None

    def module(self, param):
        return self.quantized.get(id(param))

    def _stat_index(self):
        if self._stats is None:
            self._stats = {}
            for k in self.reader.index:
                for suffix in _STAT_SUFFIXES:
                    if k.endswith("." + suffix):
                        self._stats.setdefault(k[: -len(suffix) - 1], []).append(k)
                        break
        return self._stats

    def packed_keys(self):
        return set(self._stat_index())

    def stat_keys(self, key):
        keys = self._stat_index().get(key)
        if not keys:
            raise ValueError(
                f"Unsloth: sharded loading found no bitsandbytes 4-bit statistics for '{key}' in this "
                "pre-quantized checkpoint."
            )
        return keys

    def stats(
        self,
        key,
        skip = (),
        device = "cpu",
    ):
        """{checkpoint key: tensor} of the QuantState of `key` (the `skip` statistics as storage-free stand-ins)."""
        out = {}
        for k in self.stat_keys(key):
            if k.rsplit(".", 1)[-1] in skip:
                view = self.reader._open(k).get_slice(k)
                dtype = _SAFETENSORS_DTYPES[view.get_dtype()]
                out[k] = torch.zeros((), dtype = dtype, device = device).expand(tuple(view.get_shape()))
            else:
                out[k] = self.reader.full(k)
        return out

    def quant_state(self, key, device):
        import bitsandbytes as bnb
        return bnb.functional.QuantState.from_dict(qs_dict = self.stats(key), device = device)

    def packed_dtype(self, key):
        return _SAFETENSORS_DTYPES[self.reader._open(key).get_slice(key).get_dtype()]

    def _param(self, data, stats, module, device):
        import bitsandbytes as bnb

        new = bnb.nn.Params4bit.from_prequantized(
            data = data,
            quantized_stats = stats,
            requires_grad = False,
            device = device,
            module = module,
        )
        module._is_hf_initialized = True
        return new

    def whole(self, key, module, device):
        return self._param(self.reader.full(key), self.stats(key), module, device)

    def placeholder(self, key, module, device):
        """A storage-free Params4bit stand-in (packed bytes and absmax zero-stride; maps and shape real)."""
        view = self.reader._open(key).get_slice(key)
        data = torch.zeros((), dtype = self.packed_dtype(key), device = device).expand(
            tuple(view.get_shape())
        )
        stats = self.stats(key, skip = ("absmax", "nested_absmax"), device = device)
        return self._param(data, stats, module, device)

    def rows(self, key, start, stop, module, device):
        """Packed rows [start, stop) of the stored [N, 1] tensor with the whole QuantState: an FSDP2 chunk."""
        return self._param(self.reader.rows(key, start, stop), self.stats(key), module, device)
