# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-architecture eager fusions for the diffusion DiT blocks.

``diffusion_eager_patches.py`` covers the SHARED classes (RMSNorm, AdaLayerNorm*). The remaining
fusible ops -- the gated residual ``x = x + gate * out`` and the inline modulation
``norm * (1 + scale) + shift`` -- are written longhand in each family's PER-MODEL block forward,
so they need per-arch patches. One small patch function per target, applied through the shared,
reversible backend.

The fused op is ``torch.addcmul(a, b, c) == a + b * c`` (one FMA kernel, 1-ULP, MORE accurate).
Only the out-of-place form is used: COMPILE-SAFE (lowers to plain ops, neutral under
torch.compile), it captures the eager win (fewer launches) but not the output allocation. The
in-place ``addcmul_`` would save the allocation, but that measured NEUTRAL here (CUDA allocator
recycles) and is compile-unsafe + aliasing-risky.

Each patch is guarded TWICE: ``can_safely_patch`` checks the SIGNATURE, and ``_body_has`` confirms
the exact rewritten lines are still present, so a changed diffusers block is left UNPATCHED.
Kill-switch: ``UNSLOTH_DIFFUSION_ARCH_PATCHES=0``. Implemented for all five families (see ``_SPECS``).
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Callable, Optional

import torch

from .diffusion_patch_backend import apply_patch, revert_patch

logger = logging.getLogger(__name__)

_ENV_ENABLE = "UNSLOTH_DIFFUSION_ARCH_PATCHES"


def _patches_enabled() -> bool:
    return (os.environ.get(_ENV_ENABLE) or "").strip().lower() not in ("0", "off", "false", "no")


def _body_has(fn: Callable, *needles: str) -> bool:
    """True iff every ``needle`` is in ``fn``'s source -- a body-drift guard that self-disables the
    patch if diffusers changed the rewritten lines."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return all(n in src for n in needles)


# =====================================================================================
# qwen-image: QwenImageTransformerBlock._modulate  (modulation addcmul, all 4 call sites)
# =====================================================================================
def _qwen_modulate(
    self,
    x,
    mod_params,
    index = None,
):
    """``QwenImageTransformerBlock._modulate`` with the final ``x*(1+scale)+shift`` fused to
    ``torch.addcmul`` (both the global and per-token ``index`` branches end in it)."""
    shift, scale, gate = mod_params.chunk(3, dim = -1)

    if index is not None:
        actual_batch = shift.size(0) // 2
        shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
        scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
        gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]
        index_expanded = index.unsqueeze(-1)
        shift_0_exp = shift_0.unsqueeze(1)
        shift_1_exp = shift_1.unsqueeze(1)
        scale_0_exp = scale_0.unsqueeze(1)
        scale_1_exp = scale_1.unsqueeze(1)
        gate_0_exp = gate_0.unsqueeze(1)
        gate_1_exp = gate_1.unsqueeze(1)
        shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
        scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
        gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
    else:
        shift_result = shift.unsqueeze(1)
        scale_result = scale.unsqueeze(1)
        gate_result = gate.unsqueeze(1)

    # fused: x * (1 + scale_result) + shift_result
    return torch.addcmul(shift_result, x, 1 + scale_result), gate_result


def _spec_qwen_modulate():
    try:
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenImageTransformerBlock as cls,
        )
    except Exception:  # noqa: BLE001
        return None
    orig = getattr(cls, "_modulate", None)
    if orig is None or not _body_has(orig, "x * (1 + scale_result) + shift_result"):
        return None
    return (cls, "_modulate", _qwen_modulate)


# =====================================================================================
# z-image: ZImageTransformerBlock.forward  (the 2 gated-residual addcmuls)
# =====================================================================================
def _zimage_forward(
    self,
    x: torch.Tensor,
    attn_mask: torch.Tensor,
    freqs_cis: torch.Tensor,
    adaln_input: torch.Tensor | None = None,
    noise_mask: torch.Tensor | None = None,
    adaln_noisy: torch.Tensor | None = None,
    adaln_clean: torch.Tensor | None = None,
):
    """``ZImageTransformerBlock.forward`` with the two gated residuals ``x = x + gate * sublayer``
    fused to ``torch.addcmul``. The shift-free ``*scale`` modulation and non-gated residuals are
    left as-is."""
    from diffusers.models.transformers.transformer_z_image import select_per_token

    if self.modulation:
        seq_len = x.shape[1]

        if noise_mask is not None:
            mod_noisy = self.adaLN_modulation(adaln_noisy)
            mod_clean = self.adaLN_modulation(adaln_clean)

            scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(
                4, dim = 1
            )
            scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(
                4, dim = 1
            )

            gate_msa_noisy, gate_mlp_noisy = gate_msa_noisy.tanh(), gate_mlp_noisy.tanh()
            gate_msa_clean, gate_mlp_clean = gate_msa_clean.tanh(), gate_mlp_clean.tanh()

            scale_msa_noisy, scale_mlp_noisy = 1.0 + scale_msa_noisy, 1.0 + scale_mlp_noisy
            scale_msa_clean, scale_mlp_clean = 1.0 + scale_msa_clean, 1.0 + scale_mlp_clean

            scale_msa = select_per_token(scale_msa_noisy, scale_msa_clean, noise_mask, seq_len)
            scale_mlp = select_per_token(scale_mlp_noisy, scale_mlp_clean, noise_mask, seq_len)
            gate_msa = select_per_token(gate_msa_noisy, gate_msa_clean, noise_mask, seq_len)
            gate_mlp = select_per_token(gate_mlp_noisy, gate_mlp_clean, noise_mask, seq_len)
        else:
            mod = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim = 2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        # Attention block -- fused gated residual: x + gate_msa * attention_norm2(attn_out)
        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa, attention_mask = attn_mask, freqs_cis = freqs_cis
        )
        x = torch.addcmul(x, gate_msa, self.attention_norm2(attn_out))

        # FFN block -- fused gated residual: x + gate_mlp * ffn_norm2(feed_forward(...))
        x = torch.addcmul(
            x, gate_mlp, self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        )
    else:
        attn_out = self.attention(
            self.attention_norm1(x), attention_mask = attn_mask, freqs_cis = freqs_cis
        )
        x = x + self.attention_norm2(attn_out)
        x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

    return x


def _spec_zimage_forward():
    try:
        from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock as cls
    except Exception:  # noqa: BLE001
        return None
    orig = getattr(cls, "forward", None)
    if orig is None or not _body_has(
        orig,
        "x = x + gate_msa * self.attention_norm2(attn_out)",
        "x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))",
    ):
        return None
    return (cls, "forward", _zimage_forward)


# =====================================================================================
# flux.1: FluxTransformerBlock / FluxSingleTransformerBlock
# (block modulation goes through AdaLayerNormZero, handled by the shared patch; here we fuse the
#  inline norm2 modulation + the gated residual adds.)
# =====================================================================================
def _flux_double_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    temb,
    image_rotary_emb = None,
    joint_attention_kwargs = None,
):
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
        hidden_states, emb = temb
    )
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        self.norm1_context(encoder_hidden_states, emb = temb)
    )
    joint_attention_kwargs = joint_attention_kwargs or {}
    attention_outputs = self.attn(
        hidden_states = norm_hidden_states,
        encoder_hidden_states = norm_encoder_hidden_states,
        image_rotary_emb = image_rotary_emb,
        **joint_attention_kwargs,
    )
    if len(attention_outputs) == 2:
        attn_output, context_attn_output = attention_outputs
    elif len(attention_outputs) == 3:
        attn_output, context_attn_output, ip_attn_output = attention_outputs

    # fused: hidden_states + gate_msa * attn_output
    hidden_states = torch.addcmul(hidden_states, gate_msa.unsqueeze(1), attn_output)

    norm_hidden_states = self.norm2(hidden_states)
    # fused: norm * (1 + scale_mlp) + shift_mlp
    norm_hidden_states = torch.addcmul(
        shift_mlp[:, None], norm_hidden_states, 1 + scale_mlp[:, None]
    )

    ff_output = self.ff(norm_hidden_states)
    hidden_states = torch.addcmul(hidden_states, gate_mlp.unsqueeze(1), ff_output)
    if len(attention_outputs) == 3:
        hidden_states = hidden_states + ip_attn_output

    encoder_hidden_states = torch.addcmul(
        encoder_hidden_states, c_gate_msa.unsqueeze(1), context_attn_output
    )

    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = torch.addcmul(
        c_shift_mlp[:, None], norm_encoder_hidden_states, 1 + c_scale_mlp[:, None]
    )

    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    encoder_hidden_states = torch.addcmul(
        encoder_hidden_states, c_gate_mlp.unsqueeze(1), context_ff_output
    )
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states


def _spec_flux_double():
    try:
        from diffusers.models.transformers.transformer_flux import FluxTransformerBlock as cls
    except Exception:  # noqa: BLE001
        return None
    orig = getattr(cls, "forward", None)
    if orig is None or not _body_has(
        orig,
        "hidden_states = hidden_states + attn_output",
        "norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]",
        "encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output",
    ):
        return None
    return (cls, "forward", _flux_double_forward)


def _flux_single_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    temb,
    image_rotary_emb = None,
    joint_attention_kwargs = None,
):
    text_seq_len = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim = 1)

    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb = temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(
        hidden_states = norm_hidden_states,
        image_rotary_emb = image_rotary_emb,
        **joint_attention_kwargs,
    )

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim = 2)
    gate = gate.unsqueeze(1)
    # fused: residual + gate * proj_out(hidden_states)
    hidden_states = torch.addcmul(residual, gate, self.proj_out(hidden_states))
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    encoder_hidden_states, hidden_states = (
        hidden_states[:, :text_seq_len],
        hidden_states[:, text_seq_len:],
    )
    return encoder_hidden_states, hidden_states


def _spec_flux_single():
    try:
        from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock as cls
    except Exception:  # noqa: BLE001
        return None
    orig = getattr(cls, "forward", None)
    if orig is None or not _body_has(
        orig,
        "hidden_states = gate * self.proj_out(hidden_states)",
        "hidden_states = residual + hidden_states",
    ):
        return None
    return (cls, "forward", _flux_single_forward)


# =====================================================================================
# flux.2-klein: Flux2TransformerBlock / Flux2SingleTransformerBlock
# (modulation is INLINE, so we fuse both modulation and gated residuals; scale/shift/gate are
#  [B,1,dim] so no [:, None].)
# =====================================================================================
def _flux2_double_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    temb_mod_img,
    temb_mod_txt,
    image_rotary_emb = None,
    joint_attention_kwargs = None,
):
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation

    joint_attention_kwargs = joint_attention_kwargs or {}
    (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(
        temb_mod_img, 2
    )
    (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = (
        Flux2Modulation.split(temb_mod_txt, 2)
    )

    norm_hidden_states = self.norm1(hidden_states)
    norm_hidden_states = torch.addcmul(shift_msa, norm_hidden_states, 1 + scale_msa)

    norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
    norm_encoder_hidden_states = torch.addcmul(
        c_shift_msa, norm_encoder_hidden_states, 1 + c_scale_msa
    )

    attention_outputs = self.attn(
        hidden_states = norm_hidden_states,
        encoder_hidden_states = norm_encoder_hidden_states,
        image_rotary_emb = image_rotary_emb,
        **joint_attention_kwargs,
    )
    attn_output, context_attn_output = attention_outputs

    hidden_states = torch.addcmul(hidden_states, gate_msa, attn_output)

    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = torch.addcmul(shift_mlp, norm_hidden_states, 1 + scale_mlp)

    ff_output = self.ff(norm_hidden_states)
    hidden_states = torch.addcmul(hidden_states, gate_mlp, ff_output)

    encoder_hidden_states = torch.addcmul(encoder_hidden_states, c_gate_msa, context_attn_output)

    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = torch.addcmul(
        c_shift_mlp, norm_encoder_hidden_states, 1 + c_scale_mlp
    )

    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    encoder_hidden_states = torch.addcmul(encoder_hidden_states, c_gate_mlp, context_ff_output)
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states


def _spec_flux2_double():
    try:
        from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock as cls
    except Exception:  # noqa: BLE001
        return None
    orig = getattr(cls, "forward", None)
    if orig is None or not _body_has(
        orig,
        "norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa",
        "hidden_states = hidden_states + gate_mlp * ff_output",
        "encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output",
    ):
        return None
    return (cls, "forward", _flux2_double_forward)


def _flux2_single_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    temb_mod,
    image_rotary_emb = None,
    joint_attention_kwargs = None,
    split_hidden_states = False,
    text_seq_len = None,
):
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation

    if encoder_hidden_states is not None:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim = 1)

    mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]

    norm_hidden_states = self.norm(hidden_states)
    norm_hidden_states = torch.addcmul(mod_shift, norm_hidden_states, 1 + mod_scale)

    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(
        hidden_states = norm_hidden_states,
        image_rotary_emb = image_rotary_emb,
        **joint_attention_kwargs,
    )

    hidden_states = torch.addcmul(hidden_states, mod_gate, attn_output)
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    if split_hidden_states:
        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states
    else:
        return hidden_states


def _spec_flux2_single():
    try:
        from diffusers.models.transformers.transformer_flux2 import (
            Flux2SingleTransformerBlock as cls,
        )
    except Exception:  # noqa: BLE001
        return None
    orig = getattr(cls, "forward", None)
    if orig is None or not _body_has(
        orig,
        "norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift",
        "hidden_states = hidden_states + mod_gate * attn_output",
    ):
        return None
    return (cls, "forward", _flux2_single_forward)


# =====================================================================================
# krea-2: Krea2TransformerBlock.forward  (2 inline modulations + 2 gated residuals)
# =====================================================================================
def _krea2_block_forward(
    self,
    hidden_states,
    temb,
    image_rotary_emb,
    attention_mask = None,
):
    """``Krea2TransformerBlock.forward`` with the two inline modulations
    ``(1 + scale) * norm(x) + shift`` and the two gated residuals each fused to ``torch.addcmul``."""
    # temb: (B, 1, 6 * hidden_size), shared across blocks; each block learns an additive table.
    modulation = temb.unflatten(-1, (6, -1)) + self.scale_shift_table
    prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)

    norm1 = self.norm1(hidden_states)
    attn_out = self.attn(
        torch.addcmul(preshift, norm1, 1 + prescale),
        attention_mask = attention_mask,
        image_rotary_emb = image_rotary_emb,
    )
    hidden_states = torch.addcmul(hidden_states, pregate, attn_out)
    norm2 = self.norm2(hidden_states)
    ff_out = self.ff(torch.addcmul(postshift, norm2, 1 + postscale))
    return torch.addcmul(hidden_states, postgate, ff_out)


def _spec_krea2_forward():
    try:
        from diffusers.models.transformers.transformer_krea2 import Krea2TransformerBlock as cls
    except Exception:  # noqa: BLE001
        return None
    orig = getattr(cls, "forward", None)
    if orig is None or not _body_has(
        orig,
        "(1.0 + prescale) * self.norm1(hidden_states) + preshift",
        "hidden_states = hidden_states + pregate * attn_out",
        "(1.0 + postscale) * self.norm2(hidden_states) + postshift",
        "hidden_states = hidden_states + postgate * ff_out",
    ):
        return None
    return (cls, "forward", _krea2_block_forward)


# =====================================================================================
# registry + lifecycle
# =====================================================================================
# Each entry is a zero-arg resolver returning (cls, attr, new_fn) or None. All COMPILE-SAFE.
_SPECS: tuple[Callable[[], Optional[tuple]], ...] = (
    _spec_qwen_modulate,
    _spec_zimage_forward,
    _spec_flux_double,
    _spec_flux_single,
    _spec_flux2_double,
    _spec_flux2_single,
    _spec_krea2_forward,
)

# (cls, attr) pairs we successfully patched, for an exact reverse.
_patched: list[tuple] = []


def install_arch_patches() -> int:
    """Install the per-arch compile-safe fusions (idempotent). Returns the count applied. Safe for
    every non-``off`` tier: they lower to plain ops and are neutral under ``torch.compile``."""
    if not _patches_enabled():
        uninstall_arch_patches()
        return 0
    if _patched:
        return len(_patched)
    for resolve in _SPECS:
        try:
            spec = resolve()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "arch-patch: resolver %s failed: %s", getattr(resolve, "__name__", resolve), exc
            )
            spec = None
        if spec is None:
            continue
        cls, attr, new_fn = spec
        if apply_patch(cls, attr, new_fn, match_level = "relaxed"):
            _patched.append((cls, attr))
        else:
            logger.warning(
                "arch-patch: skipping %s.%s (signature mismatch / unavailable)",
                getattr(cls, "__name__", cls),
                attr,
            )
    logger.info("arch-patch: installed %d/%d per-arch fusions", len(_patched), len(_SPECS))
    return len(_patched)


def uninstall_arch_patches() -> None:
    """Restore every per-arch patched method/forward (idempotent)."""
    for cls, attr in list(_patched):
        revert_patch(cls, attr)
    _patched.clear()


def is_installed() -> bool:
    return bool(_patched)
