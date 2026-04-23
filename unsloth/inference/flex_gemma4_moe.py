# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Gemma 4 MoE inference with flex_attention + paged KV cache.

Sibling of ``flex_moe.py`` (Qwen3-MoE) and ``flex_gpt_oss.py`` (gpt-oss).
Scope: ``unsloth/gemma-4-26B-A4B-it`` (30 layers, 128 experts top-k 8,
hidden 2816, moe_intermediate 704, dense intermediate 2112, ~3.8B active).

Arch-specific pieces on top of the shared engine:

1. **Dual dense MLP + MoE per layer.** Unlike Qwen3/gpt-oss which replace
   the dense MLP with an expert block, Gemma 4 MoE runs both in parallel
   per layer and sums their normed outputs before the residual add:

       residual = h
       h = pre_ffw_norm(h)
       mlp_out = layer.mlp(h)                        # dense SwiGLU
       h1 = post_ffw_norm_1(mlp_out)
       h2 = pre_ffw_norm_2(residual.reshape(-1, H))  # experts input
       h2 = layer.experts(h2, top_k_idx, top_k_w)
       h2 = post_ffw_norm_2(h2).reshape(residual)
       h = post_ffw_norm(h1 + h2)
       h = residual + h
       h *= layer.layer_scalar

2. **Two-tier RoPE.**
   - Sliding layers (25/30): ``rope_type="default"``, theta=10K,
     full ``head_dim=256``.
   - Full-attn layers (5/30): ``rope_type="proportional"``, theta=1M,
     ``global_head_dim=512``, ``partial_rotary_factor=0.25`` (rotate
     only leading 25% of head dim; pass through remainder).
   Walker computes both (cos, sin) tuples per generate entry; attention
   forward picks the right one based on ``self_attn.layer_type``.

3. **Per-head Q/K/V RMSNorm pre-rotary.** ``q_norm`` and ``k_norm`` apply
   RMSNorm on the last (head_dim) axis before RoPE. ``v_norm`` applies
   with ``with_scale=False`` (RMSNorm that just divides, no gain).

4. **K=V alternative on full-attn layers.** ``attention_k_eq_v=True`` +
   full-attn layer ⇒ ``v_proj is None`` and ``value_states`` is the raw
   ``k_proj(hidden)`` output (before ``k_norm`` and before RoPE),
   followed only by ``v_norm``. Sliding layers use the normal q/k/v path.

5. **Expert grouped_mm.** Reuses ``Gemma4TextExperts.forward`` through
   ``unsloth_zoo.temporary_patches.gemma4_moe.patch_gemma4_moe`` — the
   ``per_expert_scale`` is pre-folded into routing weights, so the
   generic ``forward_native_grouped_mm`` (``moe_utils.py``) handles the
   standard ``(E, 2I, H)``/``(E, H, I)`` layout with ``act_fn =
   gelu_pytorch_tanh`` via the default ``elif hasattr(self, 'act_fn')``
   fallback — zero changes needed in moe_utils.

6. **Embedding scale + final ``layer_scalar``.** Embed output gets
   multiplied by ``sqrt(hidden_size)``. Each decoder layer's output is
   multiplied by ``self.layer_scalar`` (a ``torch.ones(1)`` buffer —
   numerically a no-op today, but must not be dropped so the walker
   matches HF bitwise).

Out of scope (this file errors out if the config requests them):
- ``num_kv_shared_layers > 0`` (E2B/E4B KV-share variants)
- ``hidden_size_per_layer_input > 0`` (E2B/E4B per-layer input gate)
- Mixed sliding-window sizes across layers
- bnb-4bit stacked experts (no ``Gemma4TextExpertsBnb4bit`` ships yet)
"""

from __future__ import annotations

import math
import os
import types
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

try:
    from .flex_qwen3_llama import (
        DECODE_KERNEL_OPTIONS_DEFAULT,
        PREFILL_KERNEL_OPTIONS_DEFAULT,
        Sequence,
        flex_attention_compiled,
        refresh_lora_merge_from_pristine,
    )
    from .flex_moe import refresh_moe_lora_merge_from_pristine
    from .flex_paged_attention import PagedKVCache, PageTable
except ImportError:  # script-mode fallback
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from flex_qwen3_llama import (  # noqa: E402
        DECODE_KERNEL_OPTIONS_DEFAULT,
        PREFILL_KERNEL_OPTIONS_DEFAULT,
        Sequence,
        flex_attention_compiled,
        refresh_lora_merge_from_pristine,
    )
    from flex_moe import refresh_moe_lora_merge_from_pristine  # noqa: E402
    from flex_paged_attention import PagedKVCache, PageTable  # noqa: E402


# ---------------------------------------------------------------------------
# Rotary helpers
# ---------------------------------------------------------------------------


def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_full(x, cos, sin):
    """Standard Llama-style RoPE on the full head dim.

    ``cos``/``sin`` shape ``(B, S, D)`` → unsqueeze at head-dim to
    ``(B, 1, S, D)`` so it broadcasts over ``(B, H, S, D)`` q/k.

    For Gemma 4 full-attention layers, ``rope_type="proportional"`` emits
    an ``inv_freq`` with zeros in the tail ``(1 - partial_rotary_factor)``
    fraction of positions. Those zero entries make the corresponding
    cos=1 / sin=0, so ``rotate_half`` passes the tail dims through
    unchanged — no separate partial-RoPE helper is needed.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# Attention forward: paged KV + flex_attention + per-layer sliding + k=v.
# ---------------------------------------------------------------------------


def make_gemma4_moe_attention_forward(page_table: PageTable):
    """Return a ``forward`` method for ``Gemma4TextAttention``.

    Differences from ``flex_gpt_oss``:
    - No sinks; call ``flex_attention_compiled`` without ``return_lse``.
    - ``q_norm`` / ``k_norm`` / ``v_norm`` before rotary / KV write.
    - ``v_proj is None`` (k=v full-attn) ⇒ reuse raw ``k_proj`` as v.
    - Partial RoPE on full-attn layers via ``_partial_rotary_dim``.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        position_embeddings_sliding: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_full: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        flex_block_mask: Optional[BlockMask] = None,
        flex_block_mask_sliding: Optional[BlockMask] = None,
        flex_input_pos: Optional[torch.Tensor] = None,
        flex_batch_idx: Optional[torch.Tensor] = None,
        flex_kernel_options: Optional[dict] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # pick cos/sin for this layer
        if self.is_sliding:
            cos, sin = position_embeddings_sliding
        else:
            cos, sin = position_embeddings_full

        # q projection
        q = self.q_proj(hidden_states).view(hidden_shape)
        q = self.q_norm(q)
        # shape now (B, S, Hq, D)

        # k/v projections
        k_raw = self.k_proj(hidden_states).view(hidden_shape)

        if self.v_proj is not None:
            v_raw = self.v_proj(hidden_states).view(hidden_shape)
        else:
            # k=v alternative: value uses the RAW k projection
            # (before k_norm, before rotary). v_norm then applies with
            # with_scale=False.
            v_raw = k_raw

        k = self.k_norm(k_raw)
        v = self.v_norm(v_raw)

        # Transpose for flex_attention: (B, S, H, D) -> (B, H, S, D).
        # Rotary applied AFTER the transpose so cos/sin broadcast on dim 1.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = _apply_rotary_full(q, cos, sin)
        k = _apply_rotary_full(k, cos, sin)

        # Paged-KV write.
        if self._paged_cache is not None and flex_input_pos is not None:
            cache_dtype = self._paged_cache.k_cache.dtype
            if k.dtype != cache_dtype:
                k = k.to(cache_dtype)
            if v.dtype != cache_dtype:
                v = v.to(cache_dtype)
            k, v = self._paged_cache.update(flex_input_pos, k, v, flex_batch_idx)

        # Per-layer block mask dispatch.
        if self.is_sliding and flex_block_mask_sliding is not None:
            block_mask = flex_block_mask_sliding
        else:
            block_mask = flex_block_mask

        attn_output = flex_attention_compiled(
            q,
            k,
            v,
            scale=self.scaling,
            block_mask=block_mask,
            enable_gqa=True,
            kernel_options=flex_kernel_options,
        )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None

    return forward


def patch_gemma4_moe_attention_forwards(model: torch.nn.Module, page_table: PageTable):
    """Attach a ``PagedKVCache`` + replace ``forward`` on every
    ``Gemma4TextAttention`` layer AND replace ``Gemma4TextExperts.forward``
    with the grouped_mm MoE backend.

    Gemma 4 has per-layer-type head_dim and kv_heads, so each layer's
    cache is sized from its own attrs.

    The stock ``Gemma4TextExperts.forward`` (modeling_gemma4.py:1263) is a
    Python loop over active experts — slow for decode and not
    CUDA-graph-capturable. We rebind to the generic
    ``forward_native_grouped_mm`` (moe_utils.py:771) which handles the
    standard ``(E, 2I, H)`` / ``(E, H, I)`` layout with the ``act_fn``
    fallback activation path (``gelu_pytorch_tanh`` via ``ACT2FN``).

    The router's ``per_expert_scale`` is already folded into
    ``top_k_weights`` by the modeling-native ``Gemma4TextRouter.forward``
    (modeling_gemma4.py:1309), so no separate folding is needed here.
    (The unsloth-zoo ``gemma4_moe.patch_gemma4_moe`` targets a legacy
    ``Gemma4TextMoEBlock`` class that no longer ships in transformers
    5.5+; it no-ops on current transformers.)
    """
    fwd = make_gemma4_moe_attention_forward(page_table)
    text_cfg = getattr(model.config, "text_config", model.config)

    if getattr(text_cfg, "num_kv_shared_layers", 0):
        raise NotImplementedError(
            "Gemma 4 MoE with KV-shared layers not supported yet "
            f"(num_kv_shared_layers={text_cfg.num_kv_shared_layers})."
        )
    if getattr(text_cfg, "hidden_size_per_layer_input", 0):
        raise NotImplementedError(
            "Gemma 4 MoE with per-layer input gate not supported yet "
            f"(hidden_size_per_layer_input={text_cfg.hidden_size_per_layer_input})."
        )

    # Pick grouped_mm MoE backend once; bind per-layer below.
    try:
        from unsloth_zoo.temporary_patches.moe_utils import get_forward_moe_backend
        _moe_forward = get_forward_moe_backend()
    except Exception:  # pragma: no cover - defensive
        _moe_forward = None

    for layer in model.model.layers:
        attn = layer.self_attn
        num_q_heads = attn.q_proj.out_features // attn.head_dim
        num_kv_heads = max(1, num_q_heads // attn.num_key_value_groups)
        attn._paged_cache = PagedKVCache(
            page_table,
            n_heads=num_kv_heads,
            head_dim=attn.head_dim,
            dtype=model.dtype,
        ).to(model.device)
        attn.forward = types.MethodType(fwd, attn)

        if _moe_forward is not None and getattr(layer, "enable_moe_block", False):
            experts = layer.experts
            # The grouped_mm backend inspects ``self.act_fn`` — already
            # set by ``Gemma4TextExperts.__init__`` to
            # ``ACT2FN[config.hidden_activation]``. No other setup needed.
            experts.forward = types.MethodType(_moe_forward, experts)


# ---------------------------------------------------------------------------
# Walker: dual dense MLP + MoE per decoder layer + layer_scalar.
# ---------------------------------------------------------------------------


def _compute_rotary_per_layer_type(base, inputs_embeds, position_ids):
    """Call ``Gemma4TextRotaryEmbedding`` once per known ``layer_type``.

    Returns ``(cos_sliding, sin_sliding), (cos_full, sin_full)``. cos/sin
    for the full-attention type are ALREADY partial-sized in the HF impl
    if ``rope_type="proportional"`` + ``partial_rotary_factor<1.0`` — the
    init fn slices the head dim to ``partial_rotary_factor * head_dim``
    before emitting ``inv_freq``. So we just forward through.
    """
    rot = base.rotary_emb
    layer_types = set(getattr(rot, "layer_types", {"sliding_attention", "full_attention"}))
    pos_sliding = pos_full = None
    if "sliding_attention" in layer_types:
        pos_sliding = rot(inputs_embeds, position_ids, layer_type="sliding_attention")
    if "full_attention" in layer_types:
        pos_full = rot(inputs_embeds, position_ids, layer_type="full_attention")
    # Fallback: if only one type exists, use it for both slots so the
    # attention forward never reads None.
    if pos_sliding is None:
        pos_sliding = pos_full
    if pos_full is None:
        pos_full = pos_sliding
    return pos_sliding, pos_full


def call_gemma4_moe_model_with_flex_kwargs(model, input_ids, position_ids, flex_kwargs):
    """Walk a ``Gemma4TextModel`` manually, injecting flex kwargs into each
    attention call. Per-layer dual MLP + MoE with ``layer_scalar``.

    ``model.model.embed_tokens`` is a ``Gemma4TextScaledWordEmbedding`` that
    already applies ``* sqrt(hidden_size)`` in its forward, so we just
    call it (no manual scale).
    """
    base = model.model
    inputs_embeds = base.embed_tokens(input_ids)
    position_embeddings_sliding, position_embeddings_full = _compute_rotary_per_layer_type(
        base, inputs_embeds, position_ids
    )

    hidden_states = inputs_embeds
    compute_dtype = inputs_embeds.dtype

    for layer in base.layers:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states).to(compute_dtype)

        attn_out, _ = layer.self_attn(
            hidden_states,
            position_embeddings_sliding=position_embeddings_sliding,
            position_embeddings_full=position_embeddings_full,
            **flex_kwargs,
        )
        attn_out = layer.post_attention_layernorm(attn_out).to(compute_dtype)
        hidden_states = residual + attn_out

        residual = hidden_states
        pre_ffw = layer.pre_feedforward_layernorm(hidden_states).to(compute_dtype)
        mlp_out = layer.mlp(pre_ffw)

        if getattr(layer, "enable_moe_block", False):
            h1 = layer.post_feedforward_layernorm_1(mlp_out).to(compute_dtype)

            flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_w, top_k_idx = layer.router(flat)
            h2 = layer.pre_feedforward_layernorm_2(flat).to(compute_dtype)
            h2 = layer.experts(h2, top_k_idx, top_k_w)
            h2 = h2.reshape(residual.shape)
            h2 = layer.post_feedforward_layernorm_2(h2).to(compute_dtype)

            mlp_out = h1 + h2

        mlp_out = layer.post_feedforward_layernorm(mlp_out).to(compute_dtype)
        hidden_states = residual + mlp_out
        hidden_states = hidden_states * layer.layer_scalar

    hidden_states = base.norm(hidden_states)
    return hidden_states


# ---------------------------------------------------------------------------
# Sliding-window block-mask builders. Paged-KV aware.
# ---------------------------------------------------------------------------


def _create_sliding_causal_blockmask(page_table: PageTable, B: int, L: int, W: int):
    def sliding_causal(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < W)

    return create_block_mask(
        sliding_causal,
        B=B,
        H=None,
        Q_LEN=L,
        KV_LEN=L,
        BLOCK_SIZE=page_table.page_size,
        device=page_table.device,
    )


def _create_prefill_sliding_blockmask(
    page_table: PageTable, batch_idx: torch.Tensor, W: int, BLOCK_SIZE: int = 128
):
    assert batch_idx.ndim == 2 and batch_idx.shape[0] == 1
    L = batch_idx.shape[1]
    docs = batch_idx.view(-1)

    def document_causal_sliding(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        window = (q_idx - kv_idx) < W
        document = docs[q_idx] == docs[kv_idx]
        return causal & window & document

    return create_block_mask(
        document_causal_sliding,
        B=1,
        H=None,
        Q_LEN=L,
        KV_LEN=L,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# ---------------------------------------------------------------------------
# FlexGemma4MoEInference
# ---------------------------------------------------------------------------


class FlexGemma4MoEInference:
    """Gemma 4 MoE inference engine. API-compatible with ``FlexMoEInference``
    and ``FlexGptOssInference``.

    Phase 1: eager decode (capture disabled). Phase 3 enables capture.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size=32,
        max_seq_length=2048,
        n_pages=2048,
        page_size=128,
        max_new_tokens=512,
        decode_kernel_options=None,
        prefill_kernel_options=None,
        fa4_prefill=None,
        base_model=None,
        peft_model=None,
        cumem_allocator=None,
        compile_walker=None,
    ):
        assert max_seq_length % page_size == 0
        assert hasattr(model, "model") and hasattr(model.model, "layers"), (
            "FlexGemma4MoEInference expects a HF CausalLM shape (.model.layers)."
        )
        for i, layer in enumerate(model.model.layers):
            if not getattr(layer, "enable_moe_block", False):
                # Dense-only Gemma 4 layers mixed with MoE is not expected
                # for 26B-A4B; if it ever appears we still don't crash —
                # the walker's ``if enable_moe_block`` guard handles it.
                continue
            assert hasattr(layer, "router") and hasattr(layer, "experts"), (
                f"Layer {i} claims enable_moe_block=True but is missing router/experts."
            )

        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.eos_token_id = tokenizer.eos_token_id
        self.base_model = base_model
        self.peft_model = peft_model
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.page_size = page_size
        self.max_new_tokens = max_new_tokens

        # Detect per-layer sliding window. Gemma 4 uses one sliding size
        # across sliding_attention layers.
        sliding_windows = set()
        for layer in model.model.layers:
            sw = getattr(layer.self_attn, "sliding_window", None)
            if sw is not None:
                sliding_windows.add(int(sw))
        if len(sliding_windows) > 1:
            raise NotImplementedError(
                f"Mixed sliding-window sizes not supported: {sliding_windows}"
            )
        self.sliding_window = next(iter(sliding_windows), None)

        # Detect bnb-4bit stacked experts (no such class ships yet for
        # Gemma 4; if it appears, flag and force eager).
        self._has_bnb_experts = any(
            getattr(layer, "enable_moe_block", False)
            and type(layer.experts).__name__.endswith("Bnb4bit")
            for layer in model.model.layers
        )

        # FA4 kernel branch.
        if fa4_prefill is None or fa4_prefill:
            major, _ = torch.cuda.get_device_capability(self.device)
            supported = major >= 9
            if fa4_prefill and not supported:
                import warnings
                warnings.warn(
                    f"--fa4_prefill needs sm_90+; found sm_{major}0. Falling "
                    f"back to the Triton flex_attention backend.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            fa4_prefill = supported
        self.fa4_prefill = fa4_prefill
        self.prefill_q_block = 256 if fa4_prefill else 128
        self.prefill_kv_block = 128
        self.decode_kernel_options = (
            decode_kernel_options
            if decode_kernel_options is not None
            else DECODE_KERNEL_OPTIONS_DEFAULT
        )
        base_prefill_opts = (
            prefill_kernel_options
            if prefill_kernel_options is not None
            else dict(PREFILL_KERNEL_OPTIONS_DEFAULT)
        )
        if fa4_prefill:
            base_prefill_opts = dict(base_prefill_opts)
            base_prefill_opts.pop("FORCE_USE_FLEX_ATTENTION", None)
            base_prefill_opts["BACKEND"] = "FLASH"
        self.prefill_kernel_options = base_prefill_opts

        from .sleep_mode import kv_cache_pool as _kv_cache_pool
        with _kv_cache_pool(cumem_allocator):
            self.page_table = PageTable(
                n_pages=n_pages,
                page_size=page_size,
                max_batch_size=max_batch_size,
                device=self.device.type,
            )
            patch_gemma4_moe_attention_forwards(model, self.page_table)

        self.input_pos_buffer = torch.zeros(
            max_batch_size, dtype=torch.int32, device=self.device
        )
        self.block_mask_logical = self.page_table.create_causal_blockmask(
            B=max_batch_size,
            L=max_seq_length,
        )
        if self.sliding_window is not None:
            self.block_mask_logical_sliding = _create_sliding_causal_blockmask(
                self.page_table,
                B=max_batch_size,
                L=max_seq_length,
                W=self.sliding_window,
            )
        else:
            self.block_mask_logical_sliding = None

        self.cudagraph_captured = False
        self.graphs = {}
        self.graph_vars = {}
        self.graph_bs = None

        # Optional torch.compile walker.
        if compile_walker is None:
            compile_walker = os.environ.get("UNSLOTH_FLEX_COMPILE_WALKER", "") == "1"
        self._moe_walker = call_gemma4_moe_model_with_flex_kwargs
        if compile_walker:
            try:
                self._moe_walker = torch.compile(
                    call_gemma4_moe_model_with_flex_kwargs,
                    fullgraph=False,
                    dynamic=False,
                )
                print(
                    "[flex-gemma4moe] wrapped call_gemma4_moe_model_with_flex_kwargs "
                    "with torch.compile(fullgraph=False, dynamic=False)"
                )
            except Exception as e:
                print(f"[flex-gemma4moe] torch.compile wrap failed: {e}")
                self._moe_walker = call_gemma4_moe_model_with_flex_kwargs

        if self._has_bnb_experts:
            print(
                "[flex-gemma4moe] bnb-4bit experts detected; CUDA graph capture "
                "disabled (decode stays eager)."
            )

    # --- tokenize / prefill / decode --------------------------------------

    def tokenize(self, sequences):
        for seq in sequences:
            if seq.input_ids is not None and seq.input_length > 0:
                continue
            ids = self.tokenizer(seq.text, return_tensors="pt")["input_ids"].squeeze(0)
            seq.input_ids = ids
            seq.input_length = ids.shape[0]

    def _prefill(self, batch: list[Sequence]) -> torch.Tensor:
        input_ids_list = [seq.input_ids.to(self.device) for seq in batch]
        input_pos_list = [
            torch.arange(seq.input_length, dtype=torch.long, device=self.device)
            for seq in batch
        ]
        batch_idx_list = [
            torch.full(
                (seq.input_length,), seq.batch_idx, dtype=torch.long, device=self.device
            )
            for seq in batch
        ]
        input_ids = torch.cat(input_ids_list).view(1, -1)
        input_pos = torch.cat(input_pos_list).view(1, -1)
        batch_idx = torch.cat(batch_idx_list).view(1, -1)

        L = input_ids.shape[1]
        q_block = self.prefill_q_block
        pad = (q_block - L % q_block) % q_block
        if pad > 0:
            input_ids = F.pad(input_ids, (0, pad), value=0)
            input_pos = F.pad(input_pos, (0, pad), value=0)
            batch_idx = F.pad(batch_idx, (0, pad), value=0)

        input_lengths = torch.tensor(
            [s.input_length for s in batch], dtype=torch.long, device=self.device
        )
        logits_positions = input_lengths.cumsum(dim=0) - 1

        prefill_block_size = (
            (self.prefill_q_block, self.prefill_kv_block)
            if self.fa4_prefill
            else self.prefill_q_block
        )
        mask = self.page_table.create_prefill_blockmask_no_paging(
            batch_idx, BLOCK_SIZE=prefill_block_size
        )
        mask_sliding = None
        if self.sliding_window is not None:
            mask_sliding = _create_prefill_sliding_blockmask(
                self.page_table,
                batch_idx,
                W=self.sliding_window,
                BLOCK_SIZE=prefill_block_size,
            )

        flex_kwargs = dict(
            flex_block_mask=mask,
            flex_block_mask_sliding=mask_sliding,
            flex_input_pos=input_pos,
            flex_batch_idx=batch_idx,
            flex_kernel_options=self.prefill_kernel_options,
        )
        position_ids = input_pos
        hidden = self._moe_walker(
            self.model, input_ids, position_ids, flex_kwargs
        )
        return self.model.lm_head(hidden[:, logits_positions, :]).squeeze(0)

    def _decode_block_mask(self, batch_idx: torch.Tensor, *, sliding: bool):
        block_mask = (
            self.block_mask_logical_sliding if sliding else self.block_mask_logical
        )
        input_pos = self.input_pos_buffer[batch_idx]
        assert batch_idx.ndim == 1 and input_pos.ndim == 1
        B = batch_idx.shape[0]
        input_block_idx = input_pos // block_mask.BLOCK_SIZE[0]
        kv_num_blocks = block_mask.kv_num_blocks[batch_idx, :, input_block_idx].view(
            B, 1, 1
        )
        kv_indices = block_mask.kv_indices[batch_idx, :, input_block_idx].view(
            B, 1, 1, -1
        )
        full_num = full_idx = None
        if block_mask.full_kv_num_blocks is not None:
            full_num = block_mask.full_kv_num_blocks[
                batch_idx, :, input_block_idx
            ].view(B, 1, 1)
            full_idx = block_mask.full_kv_indices[
                batch_idx, :, input_block_idx
            ].view(B, 1, 1, -1)

        if sliding:
            W = self.sliding_window

            def mask_fn(off):
                def m(b, h, q_idx, kv_idx):
                    pos = q_idx + off[b]
                    return (pos >= kv_idx) & (pos - kv_idx < W)
                return m
        else:
            def mask_fn(off):
                def m(b, h, q_idx, kv_idx):
                    return q_idx + off[b] >= kv_idx
                return m

        seq_length = (1, block_mask.seq_lengths[1])
        mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_num,
            full_idx,
            BLOCK_SIZE=block_mask.BLOCK_SIZE,
            mask_mod=mask_fn(input_pos),
            seq_lengths=seq_length,
        )
        return mask, input_pos

    def _decode_step_eager(self, batch_idx: torch.Tensor, input_ids: torch.Tensor):
        B = input_ids.shape[0]
        mask_full, input_pos = self._decode_block_mask(batch_idx, sliding=False)
        mask_full = self.page_table.convert_logical_block_mask(mask_full, batch_idx)
        mask_sliding = None
        if self.sliding_window is not None:
            ms, _ = self._decode_block_mask(batch_idx, sliding=True)
            mask_sliding = self.page_table.convert_logical_block_mask(ms, batch_idx)

        position_ids = input_pos.view(B, 1).to(torch.long)
        flex_kwargs = dict(
            flex_block_mask=mask_full,
            flex_block_mask_sliding=mask_sliding,
            flex_input_pos=input_pos.view(B, 1).to(torch.long),
            flex_batch_idx=batch_idx,
            flex_kernel_options=self.decode_kernel_options,
        )
        hidden = self._moe_walker(
            self.model, input_ids.view(B, 1), position_ids, flex_kwargs
        )
        return self.model.lm_head(hidden[:, -1, :])

    def _decode_step(
        self, batch_idx: torch.Tensor, input_ids: torch.Tensor, input_pos: torch.Tensor
    ):
        self.input_pos_buffer.zero_()
        self.input_pos_buffer[batch_idx] = input_pos
        if not self.cudagraph_captured or self.graph_bs is None:
            return self._decode_step_eager(batch_idx, input_ids)
        bs = input_ids.size(0)
        key = next(x for x in self.graph_bs if x >= bs)
        graph = self.graphs[key]
        gv = self.graph_vars
        for k, v in gv.items():
            if k != "outputs":
                v.zero_()
        gv["input_ids"][:bs] = input_ids
        gv["batch_idx"][:bs] = batch_idx
        graph.replay()
        return gv["outputs"][:bs]

    def capture_decode_cudagraph(self):
        """Capture one CUDA graph per bs bucket.

        Phase 3 wires this in. bnb-4bit stays eager either way (no
        stacked bnb experts class ships for Gemma 4 today; guard is
        defensive).
        """
        if self._has_bnb_experts:
            print(
                "[flex-gemma4moe] bnb-4bit experts: skipping cudagraph capture; "
                "decode stays eager."
            )
            return
        try:
            from unsloth_zoo.temporary_patches.moe_utils import select_moe_backend
            backend = select_moe_backend()
        except Exception:
            backend = None
        if backend != "grouped_mm":
            print(
                f"[flex-gemma4moe] MoE CUDA graph capture requires the "
                f"'grouped_mm' backend (got {backend!r}); skipping capture."
            )
            return

        max_bs = self.max_batch_size
        reserved_batches = []
        for bi in range(1, max_bs):
            try:
                allocated = self.page_table.allocate()
                self.page_table.reserve(
                    allocated,
                    torch.tensor([allocated], device=self.device, dtype=torch.long),
                    self.page_size,
                )
                reserved_batches.append(allocated)
            except Exception:
                break

        input_ids = torch.zeros(max_bs, dtype=torch.int64, device=self.device)
        batch_idx = torch.arange(max_bs, dtype=torch.int64, device=self.device)
        outputs = torch.zeros(
            (max_bs, self.model.config.vocab_size),
            dtype=self.model.dtype,
            device=self.device,
        )

        _env_bs = os.environ.get("UNSLOTH_FLEX_GRAPH_BS")
        if _env_bs:
            try:
                self.graph_bs = [int(x) for x in _env_bs.split(",") if x.strip()]
            except ValueError:
                print(
                    f"[flex-gemma4moe] invalid UNSLOTH_FLEX_GRAPH_BS={_env_bs!r}; "
                    f"using default bucket ladder"
                )
                self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        else:
            self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

        pool = None
        for bs in reversed(self.graph_bs):
            if bs > max_bs:
                continue
            print(f"[flex-gemma4moe] capturing CUDA graph for bs={bs}")
            torch.cuda.synchronize()
            _ = self._decode_step_eager(batch_idx[:bs], input_ids[:bs])
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool):
                outputs[:bs] = self._decode_step_eager(batch_idx[:bs], input_ids[:bs])
            if pool is None:
                pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
        for bi in reserved_batches:
            self.page_table.erase(bi)
        self.graph_vars = dict(
            input_ids=input_ids, batch_idx=batch_idx, outputs=outputs
        )

    def refresh_inference_from_base(self):
        """Refresh merged LoRA weights from pristine. Dense refresh on
        q/k/v/o/router/dense-MLP; MoE refresh on ``gate_up_proj`` /
        ``down_proj`` in standard ``(E, 2I, H)``/``(E, H, I)``
        orientation — reuses ``refresh_moe_lora_merge_from_pristine``
        verbatim."""
        if self.base_model is None or self.peft_model is None:
            return 0
        n = refresh_lora_merge_from_pristine(self.base_model, self.peft_model)
        try:
            n += refresh_moe_lora_merge_from_pristine(
                self.base_model, self.peft_model
            )
        except Exception:
            pass
        return n

    @torch.inference_mode()
    def generate(self, sequences: list[Sequence], capture_cudagraph=False):
        self.tokenize(sequences)
        waiting = deque(sequences)
        running = deque()
        done = []

        if capture_cudagraph and not self.cudagraph_captured:
            self.capture_decode_cudagraph()
            if self.graphs:
                self.cudagraph_captured = True

        while waiting or running:
            batch = []
            while waiting and self.page_table.can_reserve(waiting[0].total_length):
                seq = waiting.popleft()
                bi = self.page_table.allocate()
                self.page_table.reserve(
                    bi,
                    torch.tensor([bi], device=self.device, dtype=torch.long),
                    seq.total_length,
                )
                seq.batch_idx = bi
                batch.append(seq)
            if batch:
                logits = self._prefill(batch)
                next_ids = torch.argmax(logits, dim=-1).tolist()
                for i, seq in enumerate(batch):
                    seq.last_token_id = next_ids[i]
                    seq.output_ids.append(next_ids[i])
                    if (
                        seq.last_token_id == self.eos_token_id
                        or len(seq.output_ids) >= seq.max_new_tokens
                    ):
                        seq.finished = True
                        done.append(seq)
                        self.page_table.erase(seq.batch_idx)
                    else:
                        running.append(seq)
                continue

            decode_batch = []
            while running:
                seq = running.popleft()
                if self.page_table.capacity[seq.batch_idx] >= seq.total_length:
                    decode_batch.append(seq)
                elif self.page_table.can_reserve(
                    seq.total_length, batch_idx_int=seq.batch_idx
                ):
                    self.page_table.reserve(
                        seq.batch_idx,
                        torch.tensor(
                            [seq.batch_idx], device=self.device, dtype=torch.long
                        ),
                        seq.total_length,
                    )
                    decode_batch.append(seq)
                else:
                    running.appendleft(seq)
                    newest = running.pop()
                    waiting.appendleft(newest)
                    self.page_table.erase(newest.batch_idx)
            if not decode_batch:
                continue

            B = len(decode_batch)
            bi_tensor = torch.tensor(
                [s.batch_idx for s in decode_batch],
                dtype=torch.long,
                device=self.device,
            )
            last_ids = torch.tensor(
                [s.last_token_id for s in decode_batch],
                dtype=torch.long,
                device=self.device,
            )
            cur_pos = torch.tensor(
                [s.total_length - 1 for s in decode_batch],
                dtype=torch.int32,
                device=self.device,
            )
            logits = self._decode_step(bi_tensor, last_ids, cur_pos)
            next_ids = torch.argmax(logits, dim=-1).tolist()
            for i, seq in enumerate(decode_batch):
                seq.last_token_id = next_ids[i]
                seq.output_ids.append(next_ids[i])
                if (
                    seq.last_token_id == self.eos_token_id
                    or len(seq.output_ids) >= seq.max_new_tokens
                ):
                    seq.finished = True
                    done.append(seq)
                    self.page_table.erase(seq.batch_idx)
                else:
                    running.append(seq)

        return done
