# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Weighted cross entropy for the diffusion objectives without [N, V] fp32 logits.

Rows are processed in fixed-size chunks (the last one padded with ignored rows), so the compiled chunk kernels
see one shape per vocabulary / chunk size and never recompile as the masked fraction or sequence length moves.
Backward recomputes each chunk's logits instead of keeping them.
"""

import os

import torch
import torch.nn.functional as F

__all__ = [
    "chunked_cross_entropy",
    "diffusion_compile_mode",
    "compile_loss",
    "compile_backbone_layers",
    "fusable_head",
    "head_returns_hidden",
    "is_distributed_wrapper",
]

_CHUNK_BYTES = int(os.environ.get("UNSLOTH_DIFFUSION_CE_CHUNK_BYTES", 256 * 1024 * 1024))
_MIN_ROWS = 16


def diffusion_compile_mode(args):
    """None (eager), "auto" (compiled CE chunks on CUDA), "loss" (compiled CE chunks anywhere) or "full"
    (compiled CE chunks + per-layer backbone compile)."""
    if os.environ.get("UNSLOTH_COMPILE_DISABLE", "0").lower() in ("1", "true", "yes"):
        return None
    value = getattr(args, "diffusion_compile", "auto") if args is not None else "auto"
    if isinstance(value, str):
        value = value.strip().lower()
    if value in (False, None, "false", "0", "no", "off", "none", "eager"):
        return None
    if value in (True, "true", "1", "yes", "on", "full"):
        return "full"
    if value in ("auto", "loss"):
        return value
    raise ValueError(
        f"Unsloth: diffusion_compile must be 'auto', 'loss', 'full', True or False, got {value!r}."
    )


def compile_loss(args, device):
    mode = diffusion_compile_mode(args)
    return mode in ("loss", "full") or (mode == "auto" and torch.device(device).type == "cuda")


def _rows_per_chunk(n, vocab):
    rows = max(_MIN_ROWS, _CHUNK_BYTES // (4 * max(vocab, 1)))
    rows = 1 << (rows.bit_length() - 1)
    # Small inputs use the next power of two, so short batches neither pad to a huge chunk nor recompile per length.
    return min(rows, max(_MIN_ROWS, 1 << max(n - 1, 0).bit_length()))


def _logits(h, weight, bias, softcap):
    z = h if weight is None else F.linear(h, weight, bias)
    z = z.float()
    if softcap is not None:
        z = torch.tanh(z / softcap) * softcap
    return z


def _chunk_forward(h, weight, bias, targets, token_weights, softcap):
    z = _logits(h, weight, bias, softcap)
    lse = torch.logsumexp(z, dim = -1)
    valid = targets != -100
    picked = z.gather(1, targets.clamp_min(0)[:, None]).squeeze(1)
    ce = torch.where(valid, lse - picked, 0.0)
    return (ce * token_weights).sum(), ce.sum(), lse


def _chunk_backward(h, weight, bias, targets, token_weights, lse, grad, softcap):
    z = h if weight is None else F.linear(h, weight, bias)
    z = z.float()
    if softcap is not None:
        tanh = torch.tanh(z / softcap)
        z = tanh * softcap
    valid = targets != -100
    scale = torch.where(valid, token_weights * grad, 0.0)
    dz = torch.exp(z - lse[:, None])
    dz = dz.scatter_add(1, targets.clamp_min(0)[:, None], -valid.to(dz.dtype)[:, None])
    dz = dz * scale[:, None]
    if softcap is not None:
        dz = dz * (1 - tanh * tanh)
    return dz.to(h.dtype)


_COMPILED = {}


def _kernels(compiled):
    if not compiled:
        return _chunk_forward, _chunk_backward
    if not _COMPILED:
        _COMPILED["forward"] = torch.compile(_chunk_forward, fullgraph = True, dynamic = False)
        _COMPILED["backward"] = torch.compile(_chunk_backward, fullgraph = True, dynamic = False)
    return _COMPILED["forward"], _COMPILED["backward"]


def _chunk(tensor, start, rows, fill):
    piece = tensor[start : start + rows]
    short = rows - piece.shape[0]
    if short:
        pad = [0, 0] * (piece.dim() - 1) + [0, short]
        piece = F.pad(piece, pad, value = fill)
    return piece


class _ChunkedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, targets, token_weights, softcap, compiled):
        forward_kernel, _ = _kernels(compiled)
        n = hidden.shape[0]
        vocab = hidden.shape[1] if weight is None else weight.shape[0]
        rows = _rows_per_chunk(n, vocab)
        token_weights = token_weights.float()
        weighted = hidden.new_zeros((), dtype = torch.float32)
        total = hidden.new_zeros((), dtype = torch.float32)
        lses = []
        for start in range(0, n, rows):
            w_sum, ce_sum, lse = forward_kernel(
                _chunk(hidden, start, rows, 0),
                weight,
                bias,
                _chunk(targets, start, rows, -100),
                _chunk(token_weights, start, rows, 0),
                softcap,
            )
            weighted = weighted + w_sum
            total = total + ce_sum
            lses.append(lse)
        ctx.save_for_backward(hidden, weight, bias, targets, token_weights, torch.cat(lses))
        ctx.rows, ctx.softcap, ctx.compiled = rows, softcap, compiled
        # Backward must recompute the logits in the forward's precision, or softmax and logsumexp disagree.
        device_type = hidden.device.type
        ctx.autocast = (
            device_type,
            torch.is_autocast_enabled(device_type),
            torch.get_autocast_dtype(device_type),
        )
        ctx.mark_non_differentiable(total)
        return weighted, total

    @staticmethod
    def backward(ctx, grad_weighted, grad_total):
        device_type, enabled, dtype = ctx.autocast
        with torch.autocast(device_type = device_type, dtype = dtype, enabled = enabled):
            return _ChunkedCrossEntropy._backward(ctx, grad_weighted)

    @staticmethod
    def _backward(ctx, grad_weighted):
        hidden, weight, bias, targets, token_weights, lse = ctx.saved_tensors
        _, backward_kernel = _kernels(ctx.compiled)
        rows, n = ctx.rows, hidden.shape[0]
        need_h, need_w, need_b = ctx.needs_input_grad[:3]
        grad_hidden = torch.empty_like(hidden) if need_h else None
        grad_w = torch.zeros_like(weight, dtype = torch.float32) if need_w else None
        grad_b = (
            torch.zeros_like(bias, dtype = torch.float32) if need_b and bias is not None else None
        )
        grad = grad_weighted.float()
        for start in range(0, n, rows):
            h = _chunk(hidden, start, rows, 0)
            dz = backward_kernel(
                h,
                weight,
                bias,
                _chunk(targets, start, rows, -100),
                _chunk(token_weights, start, rows, 0),
                lse[start : start + rows],  # one padded entry per row, so every slice is full
                grad,
                ctx.softcap,
            )
            take = min(rows, n - start)
            if weight is None:
                if need_h:
                    grad_hidden[start : start + take] = dz[:take]
                continue
            if need_h:
                grad_hidden[start : start + take] = (dz @ weight)[:take]
            if need_w:
                grad_w += (dz.t() @ h).float()
            if grad_b is not None:
                grad_b += dz.sum(0).float()
        if grad_w is not None:
            grad_w = grad_w.to(weight.dtype)
        if grad_b is not None:
            grad_b = grad_b.to(bias.dtype)
        return grad_hidden, grad_w, grad_b, None, None, None, None


def chunked_cross_entropy(
    hidden,
    targets,
    token_weights = None,
    weight = None,
    bias = None,
    softcap = None,
    compiled = False,
):
    """sum(w * CE) and the unweighted CE sum over positions with target != -100.

    ``hidden`` is [..., H] projected by ``weight`` [V, H] (+ ``bias``), or already logits [..., V] when
    ``weight`` is None. Logits are computed in the input dtype and upcast like ``head(h).float()``. The second
    output carries no gradient (metrics). Returns (weighted_sum, ce_sum, count) as 0-dim tensors.
    """
    width = hidden.shape[-1]
    hidden = hidden.reshape(-1, width)
    targets = targets.reshape(-1)
    if token_weights is None:
        token_weights = (targets != -100).float()
    token_weights = token_weights.reshape(-1)
    if weight is not None and hidden.dtype != weight.dtype:
        hidden = hidden.to(weight.dtype)
    weighted, total = _ChunkedCrossEntropy.apply(
        hidden,
        weight,
        bias,
        targets,
        token_weights,
        None if softcap is None else float(softcap),
        bool(compiled),
    )
    return weighted, total, (targets != -100).sum()


def is_distributed_wrapper(model):
    return type(model).__name__ in (
        "DistributedDataParallel",
        "FullyShardedDataParallel",
        "DeepSpeedEngine",
    )


def fusable_head(model, base):
    """The plain nn.Linear output head whose input can be taken instead of logits, or None (dense logits).

    Decided once per head module (a head wrapped later, e.g. by PEFT, is re-checked): a LoRA / quantized /
    offloaded / tied or post-scaled head, or a distributed wrapper (FSDP shards the weight outside its
    forward), keeps the dense path."""
    if is_distributed_wrapper(model):
        return None
    head = base.get_output_embeddings() if hasattr(base, "get_output_embeddings") else None
    cached = base.__dict__.get("_unsloth_diffusion_fused_head")
    if cached is not None and cached[0] is head:
        return cached[1]
    config = getattr(base, "config", None)
    ok = (
        type(head) is torch.nn.Linear
        and not hasattr(head, "_hf_hook")
        and "forward" not in vars(head)
        and not getattr(
            config, "weight_tying", False
        )  # LLaDA: tied head is F.linear on wte, not this module
        and not getattr(config, "scale_logits", False)
        and head.weight.device.type != "meta"
    )
    base.__dict__["_unsloth_diffusion_fused_head"] = (head, head if ok else None)
    return head if ok else None


class head_returns_hidden:
    """While active the output head returns its input, so ``model(...).logits`` are the final hidden states."""

    def __init__(self, head):
        self.head = head

    # Eager even under an outer torch.compile: the swap is plain module plumbing, not math.
    @torch.compiler.disable
    def __enter__(self):
        if self.head is not None:
            self.head.forward = _identity
        return self.head

    @torch.compiler.disable
    def __exit__(self, *exc):
        if self.head is not None:
            self.head.__dict__.pop("forward", None)
        return False


def _identity(x):
    return x


def _decoder_layer_lists(model):
    seen = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue
        if name.rsplit(".", 1)[-1] not in ("layers", "blocks", "h"):
            continue
        if any(name.startswith(prefix + ".") for prefix in seen):
            continue
        seen.add(name)
        yield name, module


def compile_backbone_layers(model, dynamic = None):
    """Compile every decoder layer in place (regional compile); idempotent. Returns the layer count."""
    compiled = 0
    for _, layers in _decoder_layer_lists(model):
        for layer in layers:
            if getattr(layer, "_unsloth_diffusion_compiled", False):
                continue
            layer.compile(dynamic = dynamic)
            layer._unsloth_diffusion_compiled = True
            compiled += 1
    return compiled
