# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ComfyUI / kohya LoRA files on diffusers pipelines.

Community adapters are written for ComfyUI's loader, which reads conventions diffusers' converters do not:

- a per-module ``.alpha`` scale (strength = alpha / rank), and ``lora_down`` / ``lora_up`` names;
- an Anima embedding adapter (``llm_adapter.embed``) stored as a Linear pair;
- MOSS-Image adapters named after the reference layout, including ComfyUI's fused ``proj.qkv``;
- LyCORIS LoKR (``lokr_w1`` / ``lokr_w2``), a Kronecker product rather than a low-rank pair.

Plain LoRAs are normalized to the ``lora_A`` / ``lora_B`` layout each family's diffusers loader takes. LoKR modules
are applied by a ``KronLinear`` wrapper adding ``scale * x @ kron(w1, w2).T`` without ever building the dense
delta, the same arithmetic as ComfyUI's ``LokrAdapter``. Both kinds share one adapter-name space, so
:func:`set_adapter_weights` and :func:`unload_adapters` stand in for diffusers' ``set_adapters`` /
``unload_lora_weights`` wherever Studio changes the selection."""

from __future__ import annotations

import re
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

KRON_ATTR = "_unsloth_kron_adapters"
# Adapter names whose file held LoKR modules only: diffusers has no PEFT adapter of that name to set.
KRON_ONLY_ATTR = "_unsloth_kron_only"
_PREFIXES = ("diffusion_model.", "model.diffusion_model.", "transformer.")
_LOKR_SUFFIXES = (
    "lokr_w1",
    "lokr_w2",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_w2_a",
    "lokr_w2_b",
    "lokr_t2",
)
# Families whose diffusers LoRA path needs this normalization (the others keep diffusers' own reading of the file).
NORMALIZED_FAMILIES = ("anima", "krea-2", "moss-image")


def read_lora_file(path: str) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path))


def _strip(key: str) -> str:
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _lokr_stem(key: str) -> Optional[str]:
    for suffix in _LOKR_SUFFIXES:
        if key.endswith("." + suffix):
            return key[: -len(suffix) - 1]
    return None


def split_lokr(state: dict) -> tuple[dict, dict]:
    """``(lokr, rest)``: LoKR tensors grouped by module stem, and every other tensor. A ``.alpha`` goes with its LoKR
    module when that module is LoKR."""
    lokr: dict[str, dict] = {}
    for key, value in state.items():
        stem = _lokr_stem(key)
        if stem is not None:
            lokr.setdefault(stem, {})[key[len(stem) + 1 :]] = value
    rest = {}
    for key, value in state.items():
        if _lokr_stem(key) is not None:
            continue
        if key.endswith(".alpha") and key[: -len(".alpha")] in lokr:
            lokr[key[: -len(".alpha")]]["alpha"] = value
            continue
        rest[key] = value
    return lokr, rest


def normalize_lora(state: dict, family: Optional[str]) -> dict:
    """Plain LoRA tensors in the ``diffusion_model.<module>.lora_A/B.weight`` layout diffusers' converters read, with
    ``.alpha`` folded into ``lora_B`` (ComfyUI: ``strength * alpha / rank``)."""
    import torch

    renamed: dict = {}
    alphas: dict = {}
    for key, value in state.items():
        if key.endswith(".alpha"):
            alphas[key[: -len(".alpha")]] = value
            continue
        key = key.replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B.")
        renamed[key] = value
    for stem, alpha in alphas.items():
        a, b = renamed.get(stem + ".lora_A.weight"), renamed.get(stem + ".lora_B.weight")
        if a is None or b is None:
            continue  # an alpha with no low-rank pair (some converters write it for every module)
        scale = float(alpha) / int(a.shape[0])
        if scale != 1.0:
            renamed[stem + ".lora_B.weight"] = (b.to(torch.float32) * scale).to(b.dtype)
    if family == "anima":
        renamed = _anima_embedding(renamed)
    elif family == "moss-image":
        renamed = _moss_fused_qkv(renamed)
    return renamed


def _anima_embedding(state: dict) -> dict:
    """ComfyUI applies an ``llm_adapter.embed`` pair as a Linear delta ``B @ A`` on the ``[vocab, dim]`` embedding
    table; PEFT's embedding LoRA computes ``(B' @ A').T``, so ``A' = B.T`` and ``B' = A.T``."""
    out = dict(state)
    for key in [k for k in state if ".llm_adapter.embed.lora_A.weight" in k]:
        stem = key[: -len(".lora_A.weight")]
        a, b = out.pop(stem + ".lora_A.weight"), out.pop(stem + ".lora_B.weight", None)
        if b is None:
            continue
        out[stem + ".lora_embedding_A"] = b.T.contiguous()
        out[stem + ".lora_embedding_B"] = a.T.contiguous()
    return out


_MOSS_QKV = re.compile(r"^(.*\.proj\.)qkv\.(lora_[AB])\.weight$")


def _moss_fused_qkv(state: dict) -> dict:
    """ComfyUI's fused ``proj.qkv`` as the in-tree ``proj.q`` / ``k`` / ``v``: ``A`` is shared, ``B``'s rows split."""
    out: dict = {}
    for key, value in state.items():
        match = _MOSS_QKV.match(key)
        if match is None:
            out[key] = value
            continue
        stem, part = match.groups()
        if part == "lora_A":
            for name in ("q", "k", "v"):
                out[f"{stem}{name}.lora_A.weight"] = value
        else:
            rows = int(value.shape[0]) // 3
            for i, name in enumerate(("q", "k", "v")):
                out[f"{stem}{name}.lora_B.weight"] = value[i * rows : (i + 1) * rows].contiguous()
    return out


def _kron_delta(x: Any, w1: Any, w2: Any, rows: Optional[tuple[int, int]]) -> Any:
    """``x @ kron(w1, w2)[rows].T`` without the dense product: row ``i * b1 + k`` and column ``j * b2 + l`` of the
    Kronecker product is ``w1[i, j] * w2[k, l]``, so two small matmuls on a reshaped input give it."""
    a1, a2 = w1.shape
    b1, b2 = w2.shape
    lead = x.shape[:-1]
    t = x.reshape(*lead, a2, b2) @ w2.T  # (..., a2, b1)
    y = (w1 @ t).reshape(*lead, a1 * b1)  # (..., a1, b1) -> rows i * b1 + k
    return y if rows is None else y[..., rows[0] : rows[1]]


def _module_class():
    from torch import nn
    class KronLinear(nn.Module):
        """A Linear plus LoKR deltas, swapped in for the Linear the way a PEFT LoRA layer is. A wrapper rather than a
        forward hook: the fused paths that call a plain Linear's weights directly (MOSS-Image's int8 runner, the shared
        int8 GEMM) step aside for any non-Linear module, as they do for PEFT, and a compiled forward always traces it.
        Each adapter is a slot of non-persistent buffers (``w1``, ``w2`` and a 0-d ``scale`` written in place, so a
        compiled build or a CUDA graph reads a new weight without retracing); the names are the same on every layer, so
        the repeated blocks a regional compile reuses trace to the same code."""

        def __init__(self, base_layer):
            super().__init__()
            self.base_layer = base_layer
            self.slots: list = []

        @property
        def weight(self):
            return self.base_layer.weight

        @property
        def bias(self):
            return self.base_layer.bias

        def get_base_layer(self):
            base = self.base_layer
            get = getattr(base, "get_base_layer", None)
            return get() if callable(get) else base

        def forward(self, x, *args, **kwargs):
            out = self.base_layer(x, *args, **kwargs)
            for prefix, rows in self.slots:
                w1 = getattr(self, prefix + "_w1").to(x.dtype)
                w2 = getattr(self, prefix + "_w2").to(x.dtype)
                out = out + _kron_delta(x, w1, w2, rows) * getattr(self, prefix + "_scale").to(
                    out.dtype
                )
            return out

    return KronLinear


_KRON_CLASS = None


def kron_linear_class():
    global _KRON_CLASS
    if _KRON_CLASS is None:
        _KRON_CLASS = _module_class()
    return _KRON_CLASS


def _parent_of(root: Any, module: Any) -> tuple[Any, str]:
    for name, child in root.named_modules():
        for attr, sub in child.named_children():
            if sub is module:
                return child, attr
    raise ValueError("the LoKR target is not a submodule of its component")


class KronAdapter:
    """One adapter's slot on one :class:`KronLinear` (created around the layer on first use)."""

    def __init__(
        self,
        root: Any,
        module: Any,
        w1: Any,
        w2: Any,
        alpha_scale: float,
        rows: Optional[tuple[int, int]] = None,
    ):
        import torch

        cls = kron_linear_class()
        if not isinstance(module, cls):
            parent, attr = _parent_of(root, module)
            wrapper = cls(module)
            setattr(parent, attr, wrapper)
            module = wrapper
        self.wrapper = module
        self.root = root
        base = module.get_base_layer()
        device = _device_of(base)
        dtype = _dtype_of(base)
        self.prefix = f"kron{len(module.slots)}"
        self.alpha_scale = float(alpha_scale)
        module.register_buffer(
            self.prefix + "_w1", w1.to(device = device, dtype = dtype), persistent = False
        )
        module.register_buffer(
            self.prefix + "_w2", w2.to(device = device, dtype = dtype), persistent = False
        )
        module.register_buffer(
            self.prefix + "_scale",
            torch.tensor(self.alpha_scale, dtype = torch.float32, device = device),
            persistent = False,
        )
        module.slots.append((self.prefix, rows))

    def set_weight(self, weight: float) -> None:
        import torch
        with torch.no_grad():
            getattr(self.wrapper, self.prefix + "_scale").fill_(float(weight) * self.alpha_scale)

    def remove(self) -> None:
        wrapper = self.wrapper
        wrapper.slots = [slot for slot in wrapper.slots if slot[0] != self.prefix]
        for suffix in ("_w1", "_w2", "_scale"):
            wrapper._buffers.pop(self.prefix + suffix, None)
        if not wrapper.slots:
            parent, attr = _parent_of(self.root, wrapper)
            setattr(parent, attr, wrapper.base_layer)


def _kron_factors(parts: dict) -> tuple[Any, Any, float]:
    """``(w1, w2, alpha_scale)`` as ComfyUI computes them: a factor stored as a low-rank pair is its product, and
    ``alpha / rank`` applies only when one is (rank = that pair's inner size)."""
    import torch

    dim = None
    w1 = parts.get("lokr_w1")
    if w1 is None:
        w1 = parts["lokr_w1_a"].float() @ parts["lokr_w1_b"].float()
        dim = int(parts["lokr_w1_b"].shape[0])
    w2 = parts.get("lokr_w2")
    if w2 is None:
        if "lokr_t2" in parts:
            raise ValueError("a LoKR convolution (lokr_t2) cannot apply to a Linear layer")
        w2 = parts["lokr_w2_a"].float() @ parts["lokr_w2_b"].float()
        dim = int(parts["lokr_w2_b"].shape[0])
    if w1.dim() != 2 or w2.dim() != 2:
        raise ValueError("only Linear LoKR modules are supported")
    alpha = parts.get("alpha")
    scale = float(alpha) / dim if alpha is not None and dim else 1.0
    return w1.to(torch.float32), w2.to(torch.float32), scale


def _device_of(module: Any) -> Any:
    weight = getattr(module, "weight", None)
    return weight.device if weight is not None and weight.device.type != "meta" else None


def _dtype_of(module: Any) -> Any:
    """The compute dtype of a layer: a quantized weight reports its storage dtype, so fall back to its bias or bf16."""
    import torch

    for name in ("weight", "bias"):
        tensor = getattr(module, name, None)
        if tensor is not None and tensor.is_floating_point() and tensor.element_size() >= 2:
            return tensor.dtype
    return torch.bfloat16


def _base_layer(module: Any) -> Any:
    get_base = getattr(module, "get_base_layer", None)
    return get_base() if callable(get_base) else module


def _lokr_targets(
    pipe: Any, stem: str, family: Optional[str]
) -> list[tuple[Any, Any, Optional[tuple[int, int]]]]:
    """``[(component, module, row slice)]`` the file's module ``stem`` maps to on ``pipe``."""
    key = _strip(stem)
    if family == "krea-2":
        from .diffusion_single_file_converters import krea2_checkpoint_to_diffusers
        name = next(iter(krea2_checkpoint_to_diffusers({key + ".weight": None})))[: -len(".weight")]
        return [(pipe.transformer, pipe.transformer.get_submodule(name), None)]
    if family == "moss-image":
        root = pipe.transformer
        match = re.match(r"^(.*\.proj\.)qkv$", key)
        if match is None:
            return [(root, root.get_submodule(key), None)]
        modules = [root.get_submodule(match.group(1) + n) for n in ("q", "k", "v")]
        width = int(_base_layer(modules[0]).weight.shape[0])
        return [(root, m, (i * width, (i + 1) * width)) for i, m in enumerate(modules)]
    if family == "anima":
        from diffusers.loaders.lora_conversion_utils import (
            _convert_non_diffusers_anima_lora_to_diffusers,
        )

        name = next(
            iter(
                _convert_non_diffusers_anima_lora_to_diffusers(
                    {"diffusion_model." + key + ".x": None}
                )
            )
        )
        component, _, path = name[: -len(".x")].partition(".")
        root = getattr(pipe, component)
        return [(root, root.get_submodule(path), None)]
    raise ValueError(f"LoKR adapters are not supported for {family or 'this model'}")


def _attach_lokr(pipe: Any, name: str, lokr: dict, family: Optional[str]) -> int:
    adapters: list[KronAdapter] = []
    try:
        for stem, parts in lokr.items():
            w1, w2, scale = _kron_factors(parts)
            for root, module, rows in _lokr_targets(pipe, stem, family):
                base = _base_layer(module)
                out_f, in_f = (
                    (int(d) for d in base.weight.shape) if base.weight.dim() == 2 else (0, 0)
                )
                full = int(w1.shape[0] * w2.shape[0])
                width = rows[1] - rows[0] if rows else full
                if in_f != int(w1.shape[1] * w2.shape[1]) or out_f != width:
                    raise ValueError(
                        f"LoKR module {_strip(stem)} is {full} x {int(w1.shape[1] * w2.shape[1])}, "
                        f"but the layer it targets is {out_f} x {in_f}"
                    )
                adapters.append(KronAdapter(root, module, w1, w2, scale, rows))
    except Exception:
        for adapter in adapters:
            adapter.remove()
        raise
    registry = pipe.__dict__.setdefault(KRON_ATTR, {})
    registry[name] = adapters
    return len(adapters)


def load_adapter(pipe: Any, path: str, name: str, family: Optional[str]) -> None:
    """Load one adapter file under ``name``: plain LoRA tensors through ``pipe.load_lora_weights`` (normalized for
    :data:`NORMALIZED_FAMILIES`), LoKR modules as :class:`KronAdapter` slots on a ``KronLinear`` wrapper."""
    if family not in NORMALIZED_FAMILIES:
        pipe.load_lora_weights(path, adapter_name = name)
        return
    state = read_lora_file(path)
    lokr, rest = split_lokr(state)
    plain = normalize_lora(rest, family)
    if not any(".lora_" in k for k in plain):
        plain = {}
    if plain:
        # PEFT wraps only the Linears it knows: take earlier LoKR wrappers out while it injects, then back around
        # whatever now sits at each path (a PEFT layer, when the same layer gained a LoRA).
        wrappers = _unwrap_kron(pipe)
        try:
            pipe.load_lora_weights(plain, adapter_name = name)
        finally:
            _rewrap_kron(wrappers)
    if lokr:
        attached = _attach_lokr(pipe, name, lokr, family)
        if not plain:
            pipe.__dict__.setdefault(KRON_ONLY_ATTR, set()).add(name)
        logger.info("diffusion.lora: %s applies %d LoKR module(s)", name, attached)
    if not plain and not lokr:
        raise ValueError(f"{name} holds no LoRA or LoKR weights this loader recognises")


def _unwrap_kron(pipe: Any) -> list:
    out = []
    seen: set = set()
    for adapters in pipe.__dict__.get(KRON_ATTR, {}).values():
        for adapter in adapters:
            wrapper = adapter.wrapper
            if id(wrapper) in seen:
                continue
            seen.add(id(wrapper))
            parent, attr = _parent_of(adapter.root, wrapper)
            setattr(parent, attr, wrapper.base_layer)
            out.append((parent, attr, wrapper))
    return out


def _rewrap_kron(wrappers: list) -> None:
    for parent, attr, wrapper in wrappers:
        wrapper.base_layer = getattr(parent, attr)
        setattr(parent, attr, wrapper)


def kron_names(pipe: Any) -> tuple[str, ...]:
    return tuple(pipe.__dict__.get(KRON_ATTR, {}))


def set_adapter_weights(pipe: Any, names: list, weights: list) -> None:
    """``pipe.set_adapters`` across both kinds: LoKR slots take their weight directly; PEFT adapters go through
    diffusers (a name that is LoKR-only has no PEFT adapter to set)."""
    set_kron_weights(pipe, dict(zip(names, weights)))
    kron_only = pipe.__dict__.get(KRON_ONLY_ATTR, set())
    pairs = [(n, w) for n, w in zip(names, weights) if n not in kron_only]
    if pairs or not kron_only:
        pipe.set_adapters([n for n, _w in pairs], adapter_weights = [w for _n, w in pairs])


def set_kron_weights(pipe: Any, weights: dict) -> None:
    """Every LoKR adapter's weight (name -> weight; an omitted name is 0, as a pinned PEFT scale is)."""
    for adapter_name, adapters in pipe.__dict__.get(KRON_ATTR, {}).items():
        for adapter in adapters:
            adapter.set_weight(float(weights.get(adapter_name, 0.0)))


def unload_adapters(pipe: Any) -> None:
    """``pipe.unload_lora_weights`` plus every LoKR wrapper."""
    for adapters in pipe.__dict__.pop(KRON_ATTR, {}).values():
        for adapter in adapters:
            adapter.remove()
    pipe.__dict__.pop(KRON_ONLY_ATTR, None)
    pipe.unload_lora_weights()
