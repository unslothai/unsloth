# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""float16 compute with fixed headroom for Z-Image on Apple Silicon MPS (opt-in: ``UNSLOTH_DIFFUSION_MPS_FP16=1``).

M1 / M2 GPUs have no native bf16, so a torch MPS bf16 GEMM runs ~1.7x slower than fp16. Plain fp16 is not an option:
Z-Image's feed-forward output reaches ~5e5 (fp16 max 65504) and renders NaN. As in mflux's float16 compute mode, only
the inside of each block's attention and feed-forward runs in fp16; the residual stream, AdaLN modulation, the norms
between modules, embedders, final layer, text encoder and VAE stay bf16. The two overflowing sites run at a fixed
power-of-two headroom and are scaled back in bf16:

    feed-forward: w2(silu(w1 x) * (w3 x / 32)) * 32
    attention:    to_out(sdpa(q, k, v / 4)) * 4

Both are identities (bias-free projections, attention linear in v, power-of-two scales exact in floating point), so
the only change is fp16 rounding inside the module. The fp16 result is clamped to the fp16 range before the bf16 cast:
a stray overflow saturates one value instead of sending inf into the stream.

Per instance (FFN ``forward`` override, attention processor subclass), never class-level, so CUDA / CPU pipes and other
models are never touched, and module calls stay at run time so a LoRA injected later still applies. Studio's int8
weight-only layers (``NativeWeightOnlyLinear``) keep their int8 storage and take fp16 activations as they are.
"""

from __future__ import annotations

import os
import types
from typing import Any, Optional

MPS_FP16_ENV = "UNSLOTH_DIFFUSION_MPS_FP16"
FFN_HEADROOM = 32.0
ATTN_HEADROOM = 4.0
_FP16_MAX = 65504.0
_MARK = "_unsloth_mps_fp16"
# torch dtype name the attention / FFN internals run in (tests set float32 to check the headroom identities exactly).
COMPUTE_DTYPE = "float16"

_SUPPORTED_TRANSFORMERS = ("ZImageTransformer2DModel",)
_PROCESSOR_CLS = "ZSingleStreamAttnProcessor"
# The processor below re-implements diffusers' Z-Image attention with the v headroom; a diffusers release that changes
# that attention (gating, biases, a new norm) no longer carries these, and the patch stays off.
_PROCESSOR_TOKENS = (
    "attn.to_q(hidden_states)",
    "attn.to_k(hidden_states)",
    "attn.to_v(hidden_states)",
    "attn.norm_q(query)",
    "attn.norm_k(key)",
    "apply_rotary_emb(",
    "dispatch_attention_fn(",
    "attn.to_out[0](hidden_states)",
)
_FFN_TOKENS = ("self.w1(x)", "self.w3(x)", "self.w2(", "silu")
_ATTN_LINEARS = ("to_q", "to_k", "to_v")
_SOURCE_OK: dict[str, bool] = {}


def mps_fp16_requested() -> bool:
    return (os.environ.get(MPS_FP16_ENV) or "").strip().lower() in ("1", "on", "true", "yes")


def _source_ok(key: str, obj: Any, tokens: tuple[str, ...]) -> bool:
    if key not in _SOURCE_OK:
        try:
            import inspect

            src = inspect.getsource(obj)
            _SOURCE_OK[key] = all(tok in src for tok in tokens)
        except Exception:  # noqa: BLE001 - unreadable source: keep stock bf16
            _SOURCE_OK[key] = False
    return _SOURCE_OK[key]


def _compute_dtype() -> Any:
    import torch

    return getattr(torch, COMPUTE_DTYPE)


def _to_stream(x: Any, dtype: Any, headroom: float) -> Any:
    # x is this module's own fp16 temporary, so clamp in place; x headroom is exact in bf16 (power of two).
    return x.clamp_(-_FP16_MAX, _FP16_MAX).to(dtype).mul_(headroom)


def _ffn_forward(self, x):
    import torch.nn.functional as F

    dtype = x.dtype
    h = x.to(_compute_dtype())
    gate = getattr(self, "_forward_silu_gating", None)
    up = self.w3(h) * (1.0 / FFN_HEADROOM)
    h = gate(self.w1(h), up) if callable(gate) else F.silu(self.w1(h)) * up
    return _to_stream(self.w2(h), dtype, FFN_HEADROOM)


_PROCESSOR: Optional[type] = None


def processor_cls() -> type:
    """``ZSingleStreamAttnProcessor`` with fp16 compute and v / 4 headroom (built lazily: imports diffusers)."""
    global _PROCESSOR
    if _PROCESSOR is not None:
        return _PROCESSOR
    from diffusers.models.transformers import transformer_z_image as zmod

    class ZImageMpsFp16AttnProcessor(zmod.ZSingleStreamAttnProcessor):
        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states = None,
            attention_mask = None,
            freqs_cis = None,
        ):
            dtype = hidden_states.dtype
            x = hidden_states.to(_compute_dtype())
            query = attn.to_q(x).unflatten(-1, (attn.heads, -1))
            key = attn.to_k(x).unflatten(-1, (attn.heads, -1))
            value = (attn.to_v(x) * (1.0 / ATTN_HEADROOM)).unflatten(-1, (attn.heads, -1))
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            if freqs_cis is not None:
                query = _rope(query, freqs_cis)
                key = _rope(key, freqs_cis)
            # A bf16 qk-norm weight (a LoRA or load that bypassed the cast) must not mix dtypes in the kernel.
            query, key = query.to(value.dtype), key.to(value.dtype)
            if attention_mask is not None and attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]
            out = zmod.dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask = attention_mask,
                dropout_p = 0.0,
                is_causal = False,
                backend = self._attention_backend,
                parallel_config = self._parallel_config,
            )
            out = attn.to_out[0](out.flatten(2, 3).to(value.dtype))
            if len(attn.to_out) > 1:
                out = attn.to_out[1](out)
            return _to_stream(out, dtype, ATTN_HEADROOM)

    _PROCESSOR = ZImageMpsFp16AttnProcessor
    return _PROCESSOR


def _rope(x_in: Any, freqs_cis: Any) -> Any:
    # diffusers' Z-Image RoPE: complex rotation in fp32, result back in the input dtype.
    import torch

    x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
    return torch.view_as_real(x * freqs_cis.unsqueeze(2)).flatten(3).type_as(x_in)


def _is_native_quant(mod: Any) -> bool:
    return type(mod).__name__ == "NativeWeightOnlyLinear" and hasattr(mod, "weight_q")


def _castable(mod: Any) -> bool:
    """A plain float module (Linear / RMSNorm, optionally PEFT-wrapped) whose weights can be stored in fp16, or a
    Studio int8 weight-only Linear. Anything carrying quantized storage (GGUF, torchao, bnb) is refused."""
    import torch

    if mod is None:
        return False
    if _is_native_quant(mod):
        return getattr(mod, "scheme", "int8") == "int8" and not getattr(mod, "act_int8", False)
    for sub in mod.modules():
        if _is_native_quant(sub) or hasattr(sub, "compute_dtype"):
            return False
    for t in (*mod.parameters(), *mod.buffers()):
        if type(t.data) is not torch.Tensor or not t.is_floating_point():
            return False
        if t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return False
    return True


def _cast_compute(mod: Any) -> None:
    if _is_native_quant(mod):
        return
    dtype = _compute_dtype()
    for t in (*mod.parameters(), *mod.buffers()):
        if t.dtype is not dtype:
            t.data = t.data.to(dtype)


def _block_targets(block: Any) -> Optional[tuple[Any, Any, list]]:
    attn = getattr(block, "attention", None)
    ffn = getattr(block, "feed_forward", None)
    if attn is None or ffn is None:
        return None
    proc = getattr(attn, "processor", None)
    if type(proc).__name__ != _PROCESSOR_CLS and block.__dict__.get(_MARK) is None:
        return None
    to_out = getattr(attn, "to_out", None)
    leaves = [getattr(attn, n, None) for n in _ATTN_LINEARS]
    leaves += [to_out[0] if to_out is not None and len(to_out) else None]
    leaves += [getattr(ffn, n, None) for n in ("w1", "w2", "w3")]
    # The headroom is exact only through bias-free projections.
    if any(m is None or getattr(m, "bias", None) is not None for m in leaves):
        return None
    norms = [n for n in (getattr(attn, "norm_q", None), getattr(attn, "norm_k", None)) if n is not None]
    return attn, ffn, leaves + norms


def unsupported_reason(root: Any, target: Any) -> Optional[str]:
    """Why ``install_mps_fp16`` would not engage on ``root`` for ``target``, or None when it would."""
    if not mps_fp16_requested():
        return f"{MPS_FP16_ENV} not set"
    if getattr(target, "device", None) != "mps":
        return "not an MPS device"
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "torch unavailable"
    if getattr(target, "dtype", None) is not torch.bfloat16:
        return "stream dtype is not bfloat16"
    if root is None or type(root).__name__ not in _SUPPORTED_TRANSFORMERS:
        return "transformer is not Z-Image"
    try:
        from diffusers.models.transformers import transformer_z_image as zmod
    except Exception:  # noqa: BLE001
        return "diffusers has no Z-Image transformer"
    if not _source_ok("processor", zmod.ZSingleStreamAttnProcessor.__call__, _PROCESSOR_TOKENS):
        return "installed diffusers Z-Image attention changed"
    if not _source_ok("ffn", zmod.FeedForward, _FFN_TOKENS):
        return "installed diffusers Z-Image feed-forward changed"
    blocks = 0
    for _name, block in root.named_modules():
        if not (hasattr(block, "attention") and hasattr(block, "feed_forward")):
            continue
        found = _block_targets(block)
        if found is None:
            return f"block {_name} does not match the Z-Image layout"
        if not all(_castable(m) for m in found[2]):
            return f"block {_name} holds quantized or non-float weights"
        blocks += 1
    if not blocks:
        return "no Z-Image block found"
    return None


def install_mps_fp16(root: Any, target: Any, logger: Any = None) -> int:
    """Run Z-Image attention + FFN internals in fp16 with headroom on MPS. Returns blocks patched (0 = not engaged).
    All or nothing: one block that does not match keeps the whole model in bf16. Idempotent."""
    reason = unsupported_reason(root, target)
    if reason is not None:
        if logger is not None and mps_fp16_requested():
            logger.info("diffusion.mps_fp16: not engaged (%s)", reason)
        return 0
    cls = processor_cls()
    blocks = 0
    for _name, block in root.named_modules():
        if not (hasattr(block, "attention") and hasattr(block, "feed_forward")):
            continue
        attn, ffn, leaves = _block_targets(block)
        if block.__dict__.get(_MARK) is None:
            dtypes = {id(t): t.dtype for m in leaves for t in (*m.parameters(), *m.buffers())}
            for m in leaves:
                _cast_compute(m)
            old = attn.processor
            new = cls()
            new._attention_backend = getattr(old, "_attention_backend", None)
            new._parallel_config = getattr(old, "_parallel_config", None)
            attn.set_processor(new)
            ffn.forward = types.MethodType(_ffn_forward, ffn)
            block.__dict__[_MARK] = (old, leaves, dtypes)
        blocks += 1
    if logger is not None:
        logger.info(
            "diffusion.mps_fp16: blocks=%d (attention + FFN in float16 with x%g / x%g headroom; residual stream, "
            "norms and modulation bf16)",
            blocks,
            ATTN_HEADROOM,
            FFN_HEADROOM,
        )
    root.__dict__[_MARK] = blocks
    return blocks


def is_installed(root: Any) -> bool:
    return bool(root is not None and root.__dict__.get(_MARK))


def remove_mps_fp16(root: Any) -> int:
    """Undo ``install_mps_fp16``: stock processors and FFN forward, weights back in their load dtype."""
    if root is None:
        return 0
    removed = 0
    for _name, block in root.named_modules():
        mark = block.__dict__.pop(_MARK, None)
        if mark is None or _name == "":
            continue
        old, leaves, dtypes = mark
        block.attention.set_processor(old)
        block.feed_forward.__dict__.pop("forward", None)
        for m in leaves:
            if _is_native_quant(m):
                continue
            for t in (*m.parameters(), *m.buffers()):
                want = dtypes.get(id(t))
                if want is not None and t.dtype is not want:
                    t.data = t.data.to(want)
        removed += 1
    root.__dict__.pop(_MARK, None)
    return removed
