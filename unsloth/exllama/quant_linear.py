# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EXL3 linear layers and the dequantization hooks Unsloth's LoRA relies on."""

from __future__ import annotations

import torch
from typing import Any, Optional

from .utils import require_exllama, is_exllama_available


def _get_linear_exl3_cls():
    """Return the ``LinearEXL3`` class, importing exllamav3 on demand."""
    require_exllama()
    from exllamav3.modules.quant.exl3 import LinearEXL3

    return LinearEXL3


def _get_hf_linear_cls():
    """Return exllamav3's transformers wrapper class, or None if unavailable."""
    if not is_exllama_available():
        return None
    try:
        from exllamav3.integration.transformers import Exl3HfLinear
        return Exl3HfLinear
    except Exception:
        return None


def _inner_exl3(module: Any):
    """Extract the underlying ``LinearEXL3`` from a module, if present.

    Handles ExLlamaV3's ``Exl3HfLinear`` wrapper (``.inner``), a bare
    ``LinearEXL3``, and Unsloth's :class:`ExllamaV3Linear` (which holds the
    source module on ``.exl3_linear``).
    """
    if module is None:
        return None
    # Bare LinearEXL3 exposes get_weight_tensor directly.
    if hasattr(module, "get_weight_tensor") and hasattr(module, "trellis"):
        return module
    inner = getattr(module, "inner", None)
    if inner is not None and hasattr(inner, "get_weight_tensor"):
        return inner
    # Unsloth's ExllamaV3Linear stores the source Exl3HfLinear / LinearEXL3 on
    # ``exl3_linear``; recurse into it (guard against self-reference).
    exl3_linear = getattr(module, "exl3_linear", None)
    if exl3_linear is not None and exl3_linear is not module:
        return _inner_exl3(exl3_linear)
    return None


class Exl3QuantState:
    """bitsandbytes-``quant_state``-compatible descriptor for an EXL3 weight.

    Attached to a base weight tensor as ``weight.quant_state``. Exposes
    ``dequantize()`` returning the dense reconstructed ``[out_features,
    in_features]`` weight in the requested dtype, matching what Unsloth's
    ``fast_dequantize`` expects to hand to the LoRA matmul.
    """

    quant_type: str = "exl3"

    def __init__(
        self,
        exl3_linear: Any,
        *,
        in_features: int,
        out_features: int,
        compute_dtype: torch.dtype = torch.float16,
        bias: Optional[torch.Tensor] = None,
    ):
        self.exl3_linear = exl3_linear
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.dtype = compute_dtype
        self.bias = bias
        # bitsandbytes quant states expose a ``.shape`` used by some code paths.
        self.shape = torch.Size([self.out_features, self.in_features])

    @property
    def device(self) -> torch.device:
        inner = _inner_exl3(self.exl3_linear)
        if inner is not None and getattr(inner, "trellis", None) is not None:
            return inner.trellis.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def dequantize(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Reconstruct the dense weight as ``[out_features, in_features]``.

        ExLlamaV3's ``get_weight_tensor`` returns the weight in
        ``[in_features, out_features]`` (column-major w.r.t. a standard
        ``nn.Linear.weight``), so we transpose to the ``[out, in]`` convention
        used everywhere in Unsloth / PyTorch.
        """
        inner = _inner_exl3(self.exl3_linear)
        if inner is None:
            raise RuntimeError("Unsloth: EXL3 quant state has no reconstructable inner layer.")
        w = inner.get_weight_tensor()  # [in_features, out_features], fp16
        w = w.t().contiguous()  # -> [out_features, in_features]
        out_dtype = dtype or self.dtype
        if out_dtype is not None and w.dtype != out_dtype:
            w = w.to(out_dtype)
        return w

    def to(self, *args, **kwargs):
        # Quant state is device-anchored by its inner CUDA tensors; nothing to move.
        return self

    def __repr__(self) -> str:
        return (
            f"Exl3QuantState(out={self.out_features}, in={self.in_features}, "
            f"dtype={self.dtype})"
        )


def is_exl3_linear(module: Any) -> bool:
    """True if ``module`` is (or wraps) an EXL3 quantized linear layer."""
    return _inner_exl3(module) is not None


def get_exl3_quant_state(module_or_weight: Any) -> Optional[Exl3QuantState]:
    """Return the :class:`Exl3QuantState` for a weight/module, if any."""
    if isinstance(module_or_weight, torch.Tensor):
        return (
            getattr(module_or_weight, "quant_state", None)
            if isinstance(getattr(module_or_weight, "quant_state", None), Exl3QuantState)
            else None
        )
    weight = getattr(module_or_weight, "weight", None)
    if weight is not None:
        qs = getattr(weight, "quant_state", None)
        if isinstance(qs, Exl3QuantState):
            return qs
    return None


@torch.no_grad()
def exl3_fast_dequantize(
    quant_state: Exl3QuantState,
    *,
    transpose: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Reconstruct an EXL3 weight for the LoRA matmul.

    :param transpose:
        When True return ``[in_features, out_features]`` (i.e. ``W.t()``), which
        is what ``matmul_lora`` / ``fast_linear_forward`` want when they call
        ``fast_dequantize(W.t(), ...)``.
    """
    w = quant_state.dequantize(dtype = dtype)  # [out, in]
    if transpose:
        w = w.t().contiguous()
    return w


class ExllamaV3Linear(torch.nn.Linear):
    """An EXL3 quantized layer that *is* an ``nn.Linear`` for discovery.

    This subclasses :class:`torch.nn.Linear` deliberately: Unsloth / PEFT find
    LoRA targets with ``isinstance(module, torch.nn.Linear)``, and Unsloth's
    fast-LoRA kernels read ``base_layer.weight.quant_state``. Subclassing gives
    us both for free while the actual weight data lives in the EXL3 trellis
    tensors (the ``nn.Linear`` weight is a frozen meta placeholder carrying the
    :class:`Exl3QuantState`).

    The forward pass uses ExLlamaV3's fused EXL3 matmul for pure inference; for
    training the LoRA kernels bypass ``forward`` and reconstruct via the quant
    state, so autograd through the trellis kernel is never required. The base
    weight is always frozen.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        exl3_linear: Any,
        bias: Optional[torch.Tensor] = None,
        compute_dtype: torch.dtype = torch.float16,
    ):
        # Build the nn.Linear on the meta device so we never allocate the dense
        # [out, in] weight. bias handling is custom (frozen), so tell the parent
        # there is no bias and attach our own below.
        super().__init__(
            int(in_features),
            int(out_features),
            bias = False,
            device = "meta",
            dtype = compute_dtype,
        )
        # Store the EXL3 module off the nn tree (object.__setattr__) so it adds
        # no state_dict keys; it is only a handle for weight reconstruction.
        object.__setattr__(self, "exl3_linear", exl3_linear)
        self.compute_dtype = compute_dtype

        quant_state = Exl3QuantState(
            exl3_linear,
            in_features = in_features,
            out_features = out_features,
            compute_dtype = compute_dtype,
            bias = bias,
        )
        # Real (not meta) placeholder so PEFT places adapters on the right
        # device (peft #1639), shaped [out, 1] like a bnb packed weight so
        # W.t() has shape[0]==1 (the fast_dequantize transpose signal).
        device = quant_state.device
        placeholder = torch.zeros((1,), dtype = compute_dtype, device = device).expand(
            int(out_features), 1
        )
        placeholder.quant_state = quant_state
        self.weight = torch.nn.Parameter(placeholder, requires_grad = False)
        self.weight.quant_state = quant_state

        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad = False)
        else:
            self.register_parameter("bias", None)

    @property
    def quant_state(self) -> Exl3QuantState:
        return self.weight.quant_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct then F.linear; the fused trellis kernel gave wrong logits.
        dtype = x.dtype
        qs = self.weight.quant_state
        # Move weight/bias to x's device (multi-GPU device_map).
        W = qs.dequantize(dtype = dtype).to(x.device)  # [out, in]
        bias = self.bias.to(device = x.device, dtype = dtype) if self.bias is not None else None
        out = torch.nn.functional.linear(x, W, bias)
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, " f"quant=exl3"


@torch.no_grad()
def attach_exl3_quant_states(
    model: torch.nn.Module, compute_dtype: torch.dtype = torch.float16
) -> int:
    """Walk ``model`` and attach an :class:`Exl3QuantState` to every EXL3 layer.

    ExLlamaV3's transformers integration replaces ``nn.Linear`` modules with
    ``Exl3HfLinear`` wrappers. Those wrappers keep a dummy ``.weight`` meta
    tensor and, crucially, are NOT ``nn.Linear`` instances, so Unsloth / PEFT
    LoRA target discovery (``isinstance(module, nn.Linear)``) skips them. We
    therefore *replace* each wrapper in the module tree with an
    :class:`ExllamaV3Linear` (an ``nn.Linear`` subclass) whose ``weight``
    carries the :class:`Exl3QuantState`. This makes the layer both discoverable
    as a LoRA target and reconstructable by the fast-LoRA kernels.

    Returns the number of layers stamped.
    """
    count = 0

    # Collect (parent_module, attr_name, wrapper) triples first; we must not
    # mutate the module tree while iterating over it.
    replacements = []
    for parent_name, parent in model.named_modules():
        for attr_name, child in list(parent.named_children()):
            if isinstance(child, ExllamaV3Linear):
                count += 1  # already converted (e.g. re-entrant call)
                continue
            inner = _inner_exl3(child)
            if inner is None:
                continue
            in_features = getattr(child, "in_features", None) or getattr(inner, "in_features", None)
            out_features = getattr(child, "out_features", None) or getattr(
                inner, "out_features", None
            )
            if in_features is None or out_features is None:
                continue
            replacements.append((parent, attr_name, child, in_features, out_features))

    for parent, attr_name, wrapper, in_features, out_features in replacements:
        inner = _inner_exl3(wrapper)
        bias = getattr(inner, "bias", None)
        new_layer = ExllamaV3Linear(
            in_features = in_features,
            out_features = out_features,
            exl3_linear = wrapper,  # keep wrapper so .inner is re-read lazily
            bias = bias,
            compute_dtype = compute_dtype,
        )
        # Drop the wrapper's dense placeholder weight (we only need .inner for
        # reconstruction) so each layer doesn't keep a full fp16 copy in VRAM.
        try:
            w = getattr(wrapper, "weight", None)
            if isinstance(w, torch.Tensor) and w.numel() > 1:
                wrapper.weight = torch.nn.Parameter(
                    torch.zeros((1,), dtype = compute_dtype, device = "meta"),
                    requires_grad = False,
                )
        except Exception:
            pass
        setattr(parent, attr_name, new_layer)
        count += 1
    # Reclaim the freed dense placeholder weights.
    if count:
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return count


# Module attribute names of output heads that downstream code (e.g. Unsloth's
# fused cross-entropy loss) accesses via a raw ``.weight`` matmul rather than
# through ``forward`` / ``fast_dequantize``. These must hold a real dense weight.
_HEAD_ATTR_NAMES = ("lm_head",)


@torch.no_grad()
def harmonize_nonquant_dtype(model, target_dtype) -> int:
    """Cast every NON-EXL3 floating-point parameter/buffer to ``target_dtype``.

    ExLlamaV3's HF quantizer defaults unquantized layers to float16. Some
    architectures (e.g. Qwen3.5's Gated-DeltaNet linear attention) require
    bfloat16 and mix dense layers (in_proj_*, conv, norms) with quantized ones;
    a float16 dense layer then hits a dtype-mismatch against bf16 activations.
    We align all dense floating tensors to the model's intended compute dtype.
    EXL3 layers are skipped - their placeholder weight must stay as-is and the
    trellis reconstruction already casts to the requested dtype at use time.
    Returns the number of tensors cast.
    """
    if target_dtype is None:
        return 0
    exl3_modules = set()
    for module in model.modules():
        if is_exl3_linear(module):
            exl3_modules.add(id(module))
    count = 0
    for module in model.modules():
        if id(module) in exl3_modules:
            continue
        for pname, p in list(module.named_parameters(recurse = False)):
            if (
                p is not None
                and p.is_floating_point()
                and p.dtype != target_dtype
                and not p.is_meta
                and getattr(p, "quant_state", None) is None
            ):
                p.data = p.data.to(target_dtype)
                count += 1
        for bname, b in list(module.named_buffers(recurse = False)):
            if b is None or not b.is_floating_point() or b.dtype == target_dtype or b.is_meta:
                continue
            # Keep RoPE inv_freq (and similar) in float32; downcasting degrades
            # positional encoding.
            lname = bname.lower()
            if (
                "inv_freq" in lname
                or "rotary" in lname
                or lname.endswith("_freqs")
                or "freqs" in lname
            ):
                continue
            setattr(module, bname, b.to(target_dtype))
            count += 1
    return count


@torch.no_grad()
def densify_exl3_head(model: torch.nn.Module, compute_dtype: torch.dtype = torch.float16) -> int:
    """Reconstruct quantized output heads into dense ``nn.Linear`` layers.

    Unsloth's fused LM-head + cross-entropy loss reads ``lm_head.weight``
    directly and matmuls with it, so a quantized head (whose ``weight`` is a
    placeholder) breaks it. The head is quality-critical and small relative to
    the decoder, so we materialize it to a real dense ``nn.Linear`` on the
    head's device. Returns the number of heads densified.
    """
    count = 0
    # Collect (parent, attr, quant_state) first, then mutate, to avoid changing
    # the module tree while iterating over named_modules().
    targets = []
    for parent_name, parent in model.named_modules():
        for attr_name in _HEAD_ATTR_NAMES:
            head = getattr(parent, attr_name, None)
            if head is None:
                continue
            qs = None
            if isinstance(head, ExllamaV3Linear):
                qs = head.quant_state
            else:
                w = getattr(head, "weight", None)
                if isinstance(w, torch.Tensor):
                    cand = getattr(w, "quant_state", None)
                    if isinstance(cand, Exl3QuantState):
                        qs = cand
            if qs is not None:
                targets.append((parent, attr_name, qs))

    for parent, attr_name, qs in targets:
        new_head = torch.nn.Linear(
            qs.in_features,
            qs.out_features,
            bias = qs.bias is not None,
            device = qs.device,
            dtype = compute_dtype,
        )
        # Always reconstruct the real quantized head from its trellis; do NOT
        # tie it to the embedding even if tie_word_embeddings (the head is a
        # distinct quantized weight, and tying gave garbage logits).
        dense_w = qs.dequantize(dtype = compute_dtype)  # [out, in] on cuda
        new_head.weight = torch.nn.Parameter(dense_w, requires_grad = False)
        if qs.bias is not None:
            new_head.bias = torch.nn.Parameter(
                qs.bias.to(device = new_head.weight.device, dtype = compute_dtype),
                requires_grad = False,
            )
        setattr(parent, attr_name, new_head)
        count += 1
    return count
