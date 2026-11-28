# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""How much of a LoRA update survives FP8 / FP4 re-quantization after the 16-bit merge (arXiv 2611.32064).

Survival = cos(D, dW) with D = Q(W + dW) - Q(W) on a few sampled layers. Not ||D|| / ||dW||: below the
rounding step D is whole-step flips whose norm is several times ||dW||. Read-only."""

import math

import torch

__all__ = ["lora_quant_survival", "report_lora_quant_survival", "SURVIVAL_WARN_BELOW"]

SURVIVAL_WARN_BELOW = 0.5
_MAX_LAYERS = 8
_MAX_ROWS = 256

_FP8_MAX = 448.0
_FP4_MAX = 6.0

# llm-compressor scheme / torchao kind -> reference weight format.
_SCHEME_FORMATS = {
    "FP8_DYNAMIC": "fp8_channel",
    "FP8": "fp8_tensor",
    "FP8_BLOCK": "fp8_block",
    "MXFP8": "mxfp8",
    "MXFP4": "mxfp4",
    "MXFP4A16": "mxfp4",
    "NVFP4": "nvfp4",
    "NVFP4A16": "nvfp4",
    "torchao_fp8": "fp8_channel_bf16_scale",
}


def _fp8(x):
    return x.clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def _fp4(x):
    # compressed-tensors FP4_E2M1_DATA.cast_to_fp4: ties go to the even code.
    m = x.abs()
    q = torch.where(m <= 5.0, 4.0, 6.0)
    for edge, inclusive, value in (
        (3.5, False, 3.0),
        (2.5, True, 2.0),
        (1.75, False, 1.5),
        (1.25, True, 1.0),
        (0.75, False, 0.5),
        (0.25, True, 0.0),
    ):
        q = torch.where((m <= edge) if inclusive else (m < edge), value, q)
    return q * x.sign()


def _blocks(x, size):
    rows, cols = x.shape
    pad = (-cols) % size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x.view(rows, -1, size), cols


def _pow2_scale(amax, emax):
    # compressed-tensors generate_mx_scales: round_to_power_2 rounds up from 1.75 * 2^e, then - emax.
    amax = amax.clamp(min = 2.0**-126)
    exp = torch.floor(torch.log2(amax))
    exp = exp + (amax >= 1.75 * torch.exp2(exp)).float() - emax
    return torch.exp2(exp)


def _quantize(w, fmt, global_amax):
    """Fake-quantize a [rows, cols] fp32 tensor; scales only from w itself, except the NVFP4 global scale."""
    if fmt == "fp8_tensor":
        scale = (global_amax / _FP8_MAX).clamp(min = 1e-12)
        return _fp8(w / scale) * scale
    if fmt in ("fp8_channel", "fp8_channel_bf16_scale"):
        scale = (w.abs().amax(dim = 1, keepdim = True) / _FP8_MAX).clamp(min = 1e-12)
        if fmt == "fp8_channel_bf16_scale":
            # torchao keeps the per-row scale in the (bfloat16) weight dtype.
            scale = scale.to(torch.bfloat16).float()
        return _fp8(w / scale) * scale
    if fmt == "fp8_block":
        rows, cols = w.shape
        pr, pc = (-rows) % 128, (-cols) % 128
        x = torch.nn.functional.pad(w, (0, pc, 0, pr))
        x = x.view(x.shape[0] // 128, 128, x.shape[1] // 128, 128)
        scale = (x.abs().amax(dim = (1, 3), keepdim = True) / _FP8_MAX).clamp(min = 1e-12)
        out = (_fp8(x / scale) * scale).view(rows + pr, cols + pc)
        return out[:rows, :cols]
    if fmt in ("mxfp4", "mxfp8"):
        x, cols = _blocks(w, 32)
        amax = x.abs().amax(dim = -1, keepdim = True)
        if fmt == "mxfp4":
            scale = _pow2_scale(amax, 2)
            out = _fp4(x / scale) * scale
        else:
            scale = _pow2_scale(amax, 8)
            out = _fp8(x / scale) * scale
        return out.flatten(1)[:, :cols]
    if fmt == "nvfp4":
        # Per-tensor fp32 scale, then per-16 e4m3 block scales.
        global_scale = (global_amax / (_FP8_MAX * _FP4_MAX)).clamp(min = 1e-12)
        x, cols = _blocks(w, 16)
        block = _fp8(x.abs().amax(dim = -1, keepdim = True) / _FP4_MAX / global_scale)
        scale = (block * global_scale).clamp(min = 1e-12)
        out = _fp4(x / scale) * scale
        return out.flatten(1)[:, :cols]
    raise ValueError(f"unknown format {fmt}")


def _lora_layers(model):
    for name, module in model.named_modules():
        lora_A, lora_B = getattr(module, "lora_A", None), getattr(module, "lora_B", None)
        base = getattr(module, "base_layer", None)
        if not isinstance(lora_A, torch.nn.ModuleDict) or not isinstance(
            lora_B, torch.nn.ModuleDict
        ):
            continue
        if base is None or not isinstance(getattr(base, "weight", None), torch.Tensor):
            continue
        adapters = [
            a for a in getattr(module, "active_adapters", []) or [] if a in lora_A and a in lora_B
        ]
        if len(adapters) != 1 or getattr(module, "fan_in_fan_out", False):
            continue
        adapter = adapters[0]
        # DoRA merges through a magnitude rescale this check does not model.
        if adapter in (getattr(module, "lora_magnitude_vector", None) or {}):
            continue
        # PiSSA / OLoRA style residual init: B @ A also carries the initial factors, not only the update.
        if adapter in (getattr(module, "_unsloth_initial_lora", None) or {}):
            continue
        yield name, module, base, adapter


def _base_weight(base):
    # QLoRA exports merge into the original weights; the NF4 dequant is a stand-in (within 0.03 measured).
    weight = base.weight
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is not None:
        from bitsandbytes.functional import dequantize_4bit
        dtype = getattr(quant_state, "dtype", torch.bfloat16)
        return dequantize_4bit(weight.data, quant_state).to(dtype)
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    return weight.data


@torch.no_grad()
def lora_quant_survival(
    model,
    scheme,
    max_layers = _MAX_LAYERS,
    max_rows = _MAX_ROWS,
    skip = None,
):
    """[(layer name, cos(D, dW), ||D|| / ||dW||)] for up to `max_layers` evenly spaced LoRA layers, or None
    when the format is not modelled or no plain 2-D LoRA layer is found. `skip(name, module)` drops
    layers the export leaves unquantized; names are the merged model's."""
    fmt = _SCHEME_FORMATS.get(scheme)
    if fmt is None:
        return None
    layers = [
        (name.removeprefix("base_model.model."), module, base, adapter)
        for name, module, base, adapter in _lora_layers(model)
    ]
    if skip is not None:
        layers = [layer for layer in layers if not skip(layer[0], layer[1])]
    if not layers:
        return None
    if len(layers) > max_layers:
        step = len(layers) / max_layers
        layers = [layers[int(i * step)] for i in range(max_layers)]
    out = []
    for name, module, base, adapter in layers:
        W = _base_weight(base)
        if W is None or W.dim() != 2:
            continue
        A = module.lora_A[adapter].weight
        B = module.lora_B[adapter].weight
        scaling = (
            module.scaling[adapter] if isinstance(getattr(module, "scaling", None), dict) else 1.0
        )
        if A.dim() != 2 or B.dim() != 2 or B.shape[0] != W.shape[0] or A.shape[1] != W.shape[1]:
            continue
        # The export writes the merge in the model's 16-bit dtype, then reloads it as bfloat16 to quantize.
        merge_dtype = W.dtype if W.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

        def staged(x):
            return x.to(merge_dtype).to(torch.bfloat16).float()

        A = A.to(device = W.device, dtype = torch.float32)
        B = B.to(device = W.device, dtype = torch.float32)

        def merged_rows(start, stop):
            return staged(W[start:stop].float() + (B[start:stop] @ A) * scaling)

        rows = min(W.shape[0], max_rows)
        w = W[:rows].to(torch.float32)
        dW = (B[:rows] @ A) * scaling
        dW_norm = dW.norm().item()
        if not math.isfinite(dW_norm) or dW_norm == 0.0:
            continue
        merged = merged_rows(0, rows)
        before = staged(w)
        global_amax = before.abs().amax()
        merged_amax = merged.abs().amax()
        if fmt in ("fp8_tensor", "nvfp4"):
            # Per-tensor scales come from the whole tensor, before and after the merge.
            global_amax = staged(W.abs().amax())
            merged_amax = max(
                merged_rows(i, i + 1024).abs().amax() for i in range(0, W.shape[0], 1024)
            )
        q_delta = _quantize(merged, fmt, merged_amax) - _quantize(before, fmt, global_amax)
        q_norm = q_delta.norm().item()
        cosine = (q_delta * dW).sum().item() / (q_norm * dW_norm) if q_norm > 0.0 else 0.0
        out.append((name, max(cosine, 0.0), q_norm / dW_norm))
    return out or None


def _median(values):
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])


def report_lora_quant_survival(
    model,
    scheme,
    log = print,
    warn = None,
    skip = None,
):
    """Log the median survival of the sampled layers and warn below SURVIVAL_WARN_BELOW. Never raises;
    returns the median, or None when nothing was measured."""
    warn = warn or log
    try:
        ratios = lora_quant_survival(model, scheme, skip = skip)
    except Exception as error:
        log(f"Unsloth: could not estimate how much of the LoRA update survives {scheme}: {error!r}")
        return None
    if not ratios:
        return None
    median = _median([c for _, c, _ in ratios])
    norm_ratio = _median([n for _, _, n in ratios])
    worst = ", ".join(f"{n} {c:.0%}" for n, c, _ in sorted(ratios, key = lambda x: x[1])[:3])
    log(
        f"Unsloth: LoRA survival under {scheme}: about {median:.0%} of the exported weight change follows "
        f"the update (median of {len(ratios)} sampled layers, change / update norm {norm_ratio:.2f}x; "
        f"lowest: {worst})."
    )
    if median < SURVIVAL_WARN_BELOW:
        warn(
            f"Unsloth: most of the LoRA update is lost to {scheme} rounding: about {median:.0%} survives, the "
            "rest of the exported change is rounding noise. "
            "The exported weights may not reproduce the fine-tune: evaluate the export, or use merged_16bit "
            "/ GGUF instead."
        )
    return median
