# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""QAT weight fake quantization for packed 3-D expert stacks (transformers 5.x MoE, gpt-oss).

torchao's QAT prepare swaps only nn.Linear / nn.Embedding, so packed (E, N, K) expert parameters are never fake
quantized. Every weight scheme Unsloth ships quantizes along the reduction dim K in rows or blocks, so the whole
stack is fake quantized in one call on the (E * N, K) view, bit-identical to quantizing each expert alone. The
only cross-row statistic, NVFP4's per-tensor scale, is taken per expert (torchao's own 3-D per-expert form)."""

import torch

__all__ = [
    "fake_quantize_expert_stack",
    "fake_quantize_expert_stack_per_expert",
    "ExpertStackFakeQuantize",
    "expert_stack_weight_fake_quantizer",
    "prepare_expert_stacks_for_qat",
    "qat_expert_stack_stats",
    "EXPERT_STACKS_ENV",
]


def _reduction_last(weight, transposed):
    # transposed: (E, K, N) storage used as x @ W[e] (gpt-oss); quantize along K like nn.Linear's in_features.
    return weight.transpose(1, 2) if transposed else weight


class _STE(torch.autograd.Function):
    """Forward the quantized stack, pass the gradient straight through (as torchao's QAT backward does)."""

    @staticmethod
    def forward(ctx, weight, quantized):
        return quantized

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _qdq_2d_or_3d(w, quantizer):
    """Fake quantize `w` (2-D (N, K) or 3-D (E, N, K), K last and contiguous) with a torchao quantizer / config."""
    from torchao.quantization.qat.fake_quantizer import FakeQuantizerBase

    if isinstance(quantizer, FakeQuantizerBase):
        if w.dim() == 2:
            return quantizer(w)
        E, N, K = w.shape
        return quantizer(w.reshape(E * N, K)).view(E, N, K)
    name = type(quantizer).__name__
    if name == "NVFP4FakeQuantizeConfig":
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale

        scale = None
        if quantizer.use_per_tensor_scale:
            dims = tuple(range(1, w.dim())) if w.dim() == 3 else None
            amax = (
                torch.amax(torch.abs(w), dim = dims, keepdim = True)
                if dims
                else torch.max(torch.abs(w))
            )
            scale = per_tensor_amax_to_scale(amax)
        q = NVFP4Tensor.to_nvfp4(w.contiguous(), per_tensor_scale = scale)
        return q.dequantize(w.dtype)
    if name == "MXFakeQuantizeConfig":
        from torchao.prototype.mx_formats.mx_tensor import MXTensor

        shape = w.shape
        q = MXTensor.to_mx(
            w.reshape(-1, shape[-1]).contiguous(),
            elem_dtype = quantizer.dtype,
            block_size = quantizer.block_size,
            scaling_mode = quantizer.scaling_mode,
            kernel_preference = quantizer.kernel_preference,
        )
        return q.dequantize(w.dtype).view(shape)
    raise TypeError(f"Unsloth: no expert-stack fake quantization for {name}")


def _check_batchable(quantizer):
    """Raise for quantizer states whose statistics are not independent per row / K block."""
    cfg = getattr(quantizer, "config", None)
    gran = type(getattr(cfg, "granularity", None)).__name__
    if gran == "PerTensor":
        raise ValueError("Unsloth: a per-tensor scale spans one expert; use the per-expert path")
    if gran == "PerToken":
        raise ValueError(
            "Unsloth: per-token fake quantization is for activations, not expert weights"
        )
    if getattr(cfg, "range_learning", False):
        raise ValueError("Unsloth: range-learning QAT keeps one learned scale set per weight")


def fake_quantize_expert_stack(
    weight,
    quantizer,
    transposed = False,
):
    """One fake-quant call over a packed (E, N, K) expert stack (or (E, K, N) with transposed=True).

    `quantizer` is what the scheme puts on an nn.Linear weight: a torchao FakeQuantizer module (int4, int8,
    fp8-int4, fp8-fp8, int8-int4, cactus) or an NVFP4 / MX FakeQuantizeConfig (nvfp4, mxfp8). Gradient: straight
    through to `weight`."""
    assert weight.dim() == 3, f"expected a 3-D expert stack, got {tuple(weight.shape)}"
    _check_batchable(quantizer)
    w = _reduction_last(weight, transposed)
    if not w.is_contiguous():
        w = w.contiguous()
    if type(quantizer).__name__ in ("NVFP4FakeQuantizeConfig", "MXFakeQuantizeConfig"):
        with torch.no_grad():
            q = _qdq_2d_or_3d(w, quantizer)
        q = _STE.apply(w, q)
    else:
        q = _qdq_2d_or_3d(w, quantizer)
    return _reduction_last(q, transposed)


def fake_quantize_expert_stack_per_expert(
    weight,
    quantizer,
    transposed = False,
):
    """Reference: the same fake quantization one expert at a time (what per-expert nn.Linear QAT computes)."""
    out = []
    for e in range(weight.shape[0]):
        w = weight[e].t() if transposed else weight[e]
        w = w.contiguous()
        if type(quantizer).__name__ in ("NVFP4FakeQuantizeConfig", "MXFakeQuantizeConfig"):
            with torch.no_grad():
                q = _qdq_2d_or_3d(w, quantizer)
            q = _STE.apply(w, q)
        else:
            q = _qdq_2d_or_3d(w, quantizer)
        out.append(q.t() if transposed else q)
    return torch.stack(out, 0)


def expert_stack_weight_fake_quantizer(base_config):
    """The weight quantizer (module or config) torchao's QAT prepare would attach to an nn.Linear for
    `base_config`; None when the scheme has no weight fake quantization."""
    from torchao.quantization.qat.fake_quantize_config import _infer_fake_quantize_configs
    from torchao.quantization.qat.fake_quantizer import FakeQuantizerBase

    _, weight_config = _infer_fake_quantize_configs(base_config)
    if weight_config is None:
        return None
    if type(weight_config).__name__ in ("NVFP4FakeQuantizeConfig", "MXFakeQuantizeConfig"):
        return weight_config
    return FakeQuantizerBase.from_config(weight_config)


class ExpertStackFakeQuantize(torch.nn.Module):
    """A `torch.nn.utils.parametrize` parametrization: reading `experts.gate_up_proj` returns its fake-quantized
    stack, one quantizer call per read (wrap a forward in `parametrize.cached()` for one per step)."""

    def __init__(
        self,
        quantizer,
        transposed = False,
    ):
        super().__init__()
        self.quantizer = quantizer if isinstance(quantizer, torch.nn.Module) else None
        self._config = None if isinstance(quantizer, torch.nn.Module) else quantizer
        self.transposed = transposed

    def forward(self, weight):
        q = self.quantizer if self.quantizer is not None else self._config
        return fake_quantize_expert_stack(weight, q, self.transposed)


# ---- wiring into qat_scheme prepare -----------------------------------------------------------------------------

EXPERT_STACKS_ENV = (
    "UNSLOTH_QAT_EXPERT_STACKS"  # 0 = leave packed expert stacks unquantized (the old behaviour)
)
_STACK_NAMES = (("gate_up_proj", "gate_up"), ("down_proj", "down"))
_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_HOOKS = "_unsloth_qat_expert_hooks"
_ROUNDED = "_unsloth_qat_expert_rounded"
_INCOMING = "_unsloth_qat_expert_incoming"
_META = "unsloth_qat_expert_rounded"  # state dict metadata: fingerprints of this module's rounded stacks
_STATS = {"fake_quantized": 0, "rounded": 0, "skipped": {}}


def qat_expert_stack_stats():
    return {
        "fake_quantized": _STATS["fake_quantized"],
        "rounded": _STATS["rounded"],
        "skipped": dict(_STATS["skipped"]),
    }


def _enabled():
    import os
    return os.environ.get(EXPERT_STACKS_ENV, "1").strip().lower() not in ("0", "false", "off", "no")


def _transposed(owner, attr, proj):
    """True for (E, K, N) storage (gpt-oss), False for (E, N, K) (F.linear layout), None when unknown."""
    try:
        from unsloth_zoo.packed_experts import stack_rows_are_transposed
        return stack_rows_are_transposed(owner, attr, proj)
    except Exception:
        pass
    name = type(owner).__name__
    if "GptOss" in name:
        return True
    if name.endswith("Experts"):
        return False
    return None


def _expert_stacks(model):
    """(qualified name, owner module, attribute, proj type, parameter) for every 16-bit 3-D routed-expert stack,
    including stacks a LoRA wrapper keeps under `base_layer`."""
    try:
        from ._utils import _is_lora_adapter_name
    except Exception:
        _is_lora_adapter_name = lambda fqn: any(p.startswith("lora_") for p in fqn.split("."))
    out = []
    for name, owner in model.named_modules():
        if _is_lora_adapter_name(name):
            continue
        params = owner._parameters
        for attr, proj in _STACK_NAMES:
            p = params.get(attr)
            if p is None or p.dim() != 3:
                continue
            fqn = f"{name}.{attr}" if name else attr
            out.append((fqn, owner, attr, proj, p))
    return out


def _skip(reason, fqn, skipped):
    _STATS["skipped"][reason] = _STATS["skipped"].get(reason, 0) + 1
    skipped.append((fqn, reason))


def _linear_weight_quantizer(torchao_config, fqn, param, transposed, cache):
    """The weight quantizer of the recipe whose filter takes one expert of this stack as a Linear: a stand-in
    nn.Linear(K, N) on the meta device goes through the (config, filter) pairs in order, as torchao's prepare
    walks them, so embedding-only recipes (int8-int4's int8 embeddings), group-size and name filters apply.
    Filters run for every stack (they may read the name); only the quantizer built per recipe is cached."""
    _, A, B = param.shape
    K, N = (A, B) if transposed else (B, A)
    probe = torch.nn.Linear(K, N, bias = False, device = "meta", dtype = param.dtype)
    for base_config, filter_fn in torchao_config.base_config_and_filter_fns:
        try:
            taken = filter_fn is None or bool(filter_fn(probe, fqn))
        except Exception:
            taken = False
        if taken:
            if id(base_config) not in cache:
                try:
                    cache[id(base_config)] = expert_stack_weight_fake_quantizer(base_config)
                except Exception:
                    cache[id(base_config)] = None
            return cache[id(base_config)]
    return None


def _quantize_stack(param, quantizer, transposed):
    try:
        q = fake_quantize_expert_stack(param, quantizer, transposed)
    except ValueError:  # per-tensor statistics: one call per expert
        q = fake_quantize_expert_stack_per_expert(param, quantizer, transposed)
    # Same memory layout as the parameter, so the expert matmuls see what a rounded stack would hold.
    return q.contiguous()


_CHUNK = 1 << 22


def _fingerprint(t):
    """Order-sensitive identity of a stack's values, in bounded float64 chunks: (sum, sum of squares, index-weighted
    sum). Plain floats, so it also travels in state dict metadata."""
    flat = t.detach().reshape(-1)
    total = squares = weighted = 0.0
    for i in range(0, flat.numel(), _CHUNK):
        x = flat[i : i + _CHUNK].double()
        w = (torch.arange(i, i + x.numel(), device = x.device, dtype = torch.float64) % 997) + 1
        total += float(x.sum())
        squares += float((x * x).sum())
        weighted += float((x * w).sum())
        del x, w
    return (total, squares, weighted)


def _recipe(quantizer, transposed):
    """A stable description of the grid a stack is rounded onto (quantizer type, its config, layout)."""
    config = getattr(quantizer, "config", quantizer)
    return f"{type(quantizer).__name__}|{config!r}|transposed={bool(transposed)}"


def _round_stack(owner, attr, quantizer, transposed):
    """Write fake_quant(stack) into the frozen stack once, and keep it rounded across load_state_dict. A load that
    brings rounded values back (this model's own state dict, or another prepared model's, whose state dict marks
    its rounded stacks and their recipe in the metadata, when that recipe matches) is kept as is, so nothing is
    quantized twice (int8 / cactus fake quant is
    not idempotent); any other values (an unrounded checkpoint) are rounded again."""
    p = owner._parameters[attr]
    with torch.no_grad():
        p.data.copy_(_quantize_stack(p.detach(), quantizer, transposed))
    plan = owner.__dict__.setdefault(_ROUNDED, {})
    first = not plan
    plan[attr] = (quantizer, transposed, _fingerprint(p))
    if not first:
        return

    def mark(module, state_dict, prefix, local_metadata):
        rounded = module.__dict__.get(_ROUNDED, {})
        local_metadata[_META] = {name: [*fp, _recipe(q, t)] for name, (q, t, fp) in rounded.items()}

    def before_load(module, state_dict, prefix, local_metadata, *args):
        module.__dict__[_INCOMING] = dict(local_metadata.get(_META) or {})

    def after_load(module, incompatible_keys):
        rounded = module.__dict__.get(_ROUNDED, {})
        incoming = module.__dict__.pop(_INCOMING, None) or {}
        for name, (q, t, own) in list(rounded.items()):
            w = module._parameters.get(name)
            if w is None or w.dim() != 3 or w.requires_grad:
                continue
            fp = _fingerprint(w)
            theirs = incoming.get(name)
            same_recipe = theirs is not None and len(theirs) == 4 and theirs[3] == _recipe(q, t)
            if fp == own or (same_recipe and fp == tuple(theirs[:3])):
                rounded[name] = (q, t, fp)
                continue
            with torch.no_grad():
                w.data.copy_(_quantize_stack(w.detach(), q, t))
            rounded[name] = (q, t, _fingerprint(w))

    owner._register_state_dict_hook(mark)
    owner._register_load_state_dict_pre_hook(before_load, with_module = True)
    owner.register_load_state_dict_post_hook(after_load)


def _install_hooks(owner, attr, quantizer, transposed):
    """Per forward: reads of `owner.<attr>` inside the forward return the fake-quantized stack (straight-through
    gradient to the parameter). The parameter itself stays in `_parameters`, so state dicts, optimizers and
    named_parameters see the 16-bit master weight."""
    plan = owner.__dict__.setdefault(_HOOKS, {})
    if attr in plan:
        return
    plan[attr] = (quantizer, transposed)
    if len(plan) > 1:
        return

    def pre(module, args):
        for name, (q, t) in module.__dict__[_HOOKS].items():
            p = module._parameters.get(name)
            if p is not None:
                module.__dict__[name] = _quantize_stack(p, q, t)

    def post(module, args, output):
        for name in module.__dict__[_HOOKS]:
            module.__dict__.pop(name, None)
        return output

    owner.register_forward_pre_hook(pre)
    owner.register_forward_hook(post, always_call = True)


def prepare_expert_stacks_for_qat(model, torchao_config):
    """Weight fake quantization for packed 3-D expert stacks, which torchao's Linear-only prepare skips.

    Frozen stacks (LoRA, including LoRA on the stacks themselves) are rounded onto the grid once, in place: equal
    to fake quantizing them every forward, and the only form a PEFT ParamWrapper (which parametrizes the stack
    during its forward) keeps. Trainable stacks (full finetuning) are fake quantized on every forward with a
    straight-through gradient. Expert activations are not quantized. Returns the number
    of stacks covered; warns once listing stacks that stay unquantized."""
    if not _enabled():
        return 0
    stacks = _expert_stacks(model)
    if not stacks:
        return 0
    scheme = getattr(torchao_config, "qat_scheme", None)
    scheme = scheme if isinstance(scheme, str) else "custom"
    skipped, covered, cache = [], 0, {}
    for fqn, owner, attr, proj, p in stacks:
        if type(p) is not torch.nn.Parameter or p.dtype not in _FLOAT_DTYPES:
            # NF4 / NVFP4 / FP8 packed stacks stay as stored.
            _skip("packed_or_quantized", fqn, skipped)
            continue
        if p.device.type == "meta":
            _skip("meta", fqn, skipped)
            continue
        transposed = _transposed(owner, attr, proj)
        if transposed is None:
            _skip("unknown_layout", fqn, skipped)
            continue
        quantizer = _linear_weight_quantizer(torchao_config, fqn, p, transposed, cache)
        if quantizer is None:
            # No recipe's filter takes these experts as Linears (embedding-only recipe, group size, ...).
            _skip("no_linear_recipe", fqn, skipped)
            continue
        try:
            with torch.no_grad():
                _quantize_stack(
                    p[:1], quantizer, transposed
                )  # shape / group-size check on one expert
        except Exception:
            _skip("unsupported_shape", fqn, skipped)
            continue
        if not p.requires_grad:
            _round_stack(owner, attr, quantizer, transposed)
            _STATS["rounded"] += 1
        else:
            _install_hooks(owner, attr, quantizer, transposed)
            _STATS["fake_quantized"] += 1
        covered += 1
    if skipped:
        import warnings

        preview = ", ".join(f"{n} ({r})" for n, r in skipped[:6])
        more = f", ... ({len(skipped) - 6} more)" if len(skipped) > 6 else ""
        warnings.warn(
            f"Unsloth: qat_scheme='{scheme}' leaves {len(skipped)} MoE expert weight stack(s) unquantized: "
            f"{preview}{more}",
            stacklevel = 3,
        )
    if covered and any(
        getattr(b, "act_mapping_type", None) is not None or "Activation" in type(b).__name__
        for b, _ in torchao_config.base_config_and_filter_fns
    ):
        import warnings
        warnings.warn(
            f"Unsloth: qat_scheme='{scheme}' fake quantizes the MoE expert weights only; expert activations "
            "stay in 16 bit.",
            stacklevel = 3,
        )
    return covered
