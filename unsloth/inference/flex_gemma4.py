"""Gemma-4-E2B-it inference with flex_attention + paged KV cache + CUDA graphs.

Extends the Qwen3/Llama-3.2 engine in `qwen3_flex_inference.py` to a third
architecture, `unsloth/gemma-4-E2B-it`. Gemma-4 is not a drop-in addition:
its text backbone diverges from Qwen3/Llama in ways that cannot be folded
into a single `hasattr(self, "q_norm")` branch. The divergences, and how
this file handles them:

1. KV-sharing layers. E2B has 35 layers; the upper 15 lack `k_proj`,
   `v_proj`, `k_norm`, `v_norm` entirely and consume the K/V produced
   by a "store" layer further up the stack. We link each shared layer's
   `_paged_cache` to the store layer's `PagedKVCache` so flex_attention
   reads the same pages that the store layer populated a few layers
   earlier -- no sidecar, no SDPA fallback, one block mask per regime
   works for every layer.
2. Dual attention types. Each layer is either `full_attention`
   (`head_dim=512`, `rope_theta=1e6`, `partial_rotary_factor=0.25`) or
   `sliding_attention` (`head_dim=256`, `rope_theta=10000`,
   `sliding_window=512`). We precompute both (cos, sin) pairs once per
   forward and dispatch on `self.layer_type`.
3. Per-layer input embeddings. `embed_tokens_per_layer` produces a
   `[B, S, num_layers, 256]` auxiliary table that enters every layer
   through a `per_layer_input_gate -> act -> mul -> per_layer_projection
   -> post_per_layer_input_norm -> +residual` path after the MLP residual.
4. Four norms per layer. `input_layernorm` / `post_attention_layernorm`
   wrap the attention block (double residual); `pre_feedforward_layernorm`
   / `post_feedforward_layernorm` wrap the MLP (double residual). A scalar
   `layer_scalar` multiplies hidden_states at layer end.
5. Final logit softcap. `logits = tanh(logits / 30.0) * 30.0` applied on
   the lm_head output.

The engine is text-only: `Gemma4ForCausalLM(text_config)` skips the
multimodal `Gemma4ForConditionalGeneration` wrapper and its vision + audio
towers entirely. Shared helpers (`PagedKVCache`, `PageTable`, `Sequence`,
`refresh_lora_merge_from_pristine`, `run_drift_verification`,
`flex_attention_compiled`, `_apply_rotary`, FA4 capability guard) are
imported from `qwen3_flex_inference.py` unchanged.

Run:
    CUDA_VISIBLE_DEVICES=6 python scripts/benchmarks/gemma4_flex_inference.py \
        --n_prompts 64 --max_new_tokens 512 --capture_cudagraph \
        --stats_path logs/flex_gemma4_bf16.json

Requires `transformers>=5.5.0` (the `gemma4` module). The main workspace
env stays on 4.57.6; this file short-circuits with a clear install hint
if the module is missing. Use `isolated_run.py` with
`--extra_packages "transformers>=5.5.0 peft datasets"` to run on that env.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import types
from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

HERE = Path(__file__).resolve().parent

try:
    from .flex_qwen3_llama import (
        DECODE_KERNEL_OPTIONS_DEFAULT,
        PREFILL_KERNEL_OPTIONS_DEFAULT,
        Sequence,
        _apply_rotary,
        _hash_state_dict,
        _lora_needs_peft_fallback,
        flex_attention_compiled,
        refresh_lora_merge_from_pristine,
        run_drift_verification,
    )
    from .flex_paged_attention import PagedKVCache, PageTable
except ImportError:  # script-mode fallback (scripts/benchmarks CLI shim)
    sys.path.insert(0, str(HERE))
    from qwen3_flex_inference import (  # noqa: E402
        DECODE_KERNEL_OPTIONS_DEFAULT,
        PREFILL_KERNEL_OPTIONS_DEFAULT,
        Sequence,
        _apply_rotary,
        _hash_state_dict,
        _lora_needs_peft_fallback,
        flex_attention_compiled,
        refresh_lora_merge_from_pristine,
        run_drift_verification,
    )
    from flex_paged_attention import PagedKVCache, PageTable  # noqa: E402
from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask  # noqa: E402


# --- sliding-window block mask helpers ------------------------------------
#
# Gemma-4's `sliding_attention` layers attend only to the last
# `sliding_window` KV positions (`q_idx - kv_idx < W`) in addition to the
# standard causal mask. The shared helpers in `flex_paged_attention.py`
# only expose the pure-causal builders, so we build sliding variants here
# (local to this file so that file is untouched).


def _causal_blockmask_with_window(
    B: int, L: int, block_size: int, window: int, device: str
):
    def causal_windowed(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < window)

    return _create_block_mask(
        causal_windowed,
        B = B,
        H = None,
        Q_LEN = L,
        KV_LEN = L,
        BLOCK_SIZE = block_size,
        device = device,
    )


def _prefill_blockmask_with_window(batch_idx: torch.Tensor, block_size, window: int):
    assert batch_idx.ndim == 2 and batch_idx.shape[0] == 1
    L = batch_idx.shape[1]
    docs = batch_idx.view(-1)

    def document_causal_windowed(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        window_mask = q_idx - kv_idx < window
        document_mask = docs[q_idx] == docs[kv_idx]
        return causal_mask & window_mask & document_mask

    return _create_block_mask(
        document_causal_windowed,
        B = 1,
        H = None,
        Q_LEN = L,
        KV_LEN = L,
        BLOCK_SIZE = block_size,
    )


# --- transformers version guard --------------------------------------------
#
# Gemma-4 lands in `transformers>=5.5.0`. The main workspace env is on
# 4.57.6, which keeps the Qwen3 + Llama paths in qwen3_flex_inference.py
# working unchanged. We defer the import to call time so `--help` still
# works on 4.57.6.


def _require_gemma4():
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
        from transformers.models.gemma4.configuration_gemma4 import (
            Gemma4Config,
            Gemma4TextConfig,
        )
    except ImportError:
        import transformers

        raise SystemExit(
            f"Gemma-4 requires transformers>=5.5.0 (`gemma4` module). "
            f"Current: transformers=={transformers.__version__}. "
            f"Install: uv pip install 'transformers>=5.5.0'"
        )
    return Gemma4ForCausalLM, Gemma4Config, Gemma4TextConfig


# --- attention forward factory --------------------------------------------


def _apply_rotary_q(q, cos, sin):
    """Rotary on Q alone; used on shared layers where K is read
    pre-rotated from the store layer's paged cache."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim = -1)

    return (q * cos) + (rotate_half(q) * sin)


def make_flex_gemma4_attention_forward(page_table: PageTable):
    """Return a new `forward` method for `Gemma4TextAttention` that routes
    through flex_attention against a paged KV cache.

    Two layer kinds:
      - non-shared (`self.is_kv_shared_layer == False`): standard q/k/v
                   projection; writes new K/V into `self._paged_cache`
                   (which is the layer's own PagedKVCache).
      - shared     (`self.is_kv_shared_layer == True`): no `k_proj`/
                   `v_proj`/`k_norm`/`v_norm`. Reads K/V directly from
                   the *store* layer's paged cache, which the patching
                   helper has already linked onto `self._paged_cache`.
                   No write -- the store layer populated the cache for
                   the same positions earlier in the walker, so the
                   shared layer just attends over those pages with the
                   identical block mask.

    Linking shared layers to the store layer's paged cache keeps one
    block-mask + one KV layout across the whole stack and lets
    flex_attention handle every layer uniformly (no sidecar, no SDPA
    fallback, one CUDA graph capture).

    `position_embeddings` is a dict keyed by `layer_type`; we pick the
    right (cos, sin) pair before rotary.

    `self.v_proj` may be None when the config sets `attention_k_eq_v`
    (global head dim with shared K=V); that branch reuses k_raw.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict,
        attention_mask = None,
        past_key_values = None,
        cache_position = None,
        flex_block_mask: Optional[dict] = None,
        flex_input_pos: Optional[torch.Tensor] = None,
        flex_batch_idx: Optional[torch.Tensor] = None,
        flex_kernel_options: Optional[dict] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings[self.layer_type]

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        if getattr(self, "is_kv_shared_layer", False):
            # Shared layer reads from the paired store layer's K/V.
            # `PagedKVCache.update` returns different shapes for prefill
            # vs decode:
            #   - prefill: the packed k_val/v_val [1, H, L_packed, D]
            #   - decode : the full paged pool k_cache/v_cache
            #              [1, H, n_pages*page_size, D]
            # The prefill block_mask is sized for L_packed and the decode
            # block_mask is sized for the paged pool, so we need to match
            # the same shape here. The store layer stashes its
            # post-rotary k/v as `_last_k_val` / `_last_v_val` during
            # prefill; at decode time we read from its `_paged_cache`
            # (the same buffer the shared layer was linked to at patch).
            q = _apply_rotary_q(q, cos, sin)
            store_attn = self._store_attn
            if q.shape[-2] > 1:
                k = store_attn._last_k_val
                v = store_attn._last_v_val
            else:
                k = self._paged_cache.k_cache
                v = self._paged_cache.v_cache
        else:
            k_raw = self.k_proj(hidden_states).view(hidden_shape)
            k = self.k_norm(k_raw).transpose(1, 2)
            # `v_proj` may be None under Gemma-4's K=V global-attention
            # option; in that case reuse the raw (un-normed) k projection.
            if self.v_proj is not None:
                v = self.v_norm(
                    self.v_proj(hidden_states).view(hidden_shape)
                ).transpose(1, 2)
            else:
                v = k_raw.transpose(1, 2)
            q, k = _apply_rotary(q, k, cos, sin)

            # Store layers stash the post-rotary k/v so any shared
            # successors can read the same packed prefill tensors. This
            # assignment is a pointer rebind, not a copy; CUDA graph
            # capture sees a stable attribute reference. Plain
            # non-shared layers don't need this.
            if getattr(self, "store_full_length_kv", False):
                self._last_k_val = k
                self._last_v_val = v

            if self._paged_cache is not None and flex_input_pos is not None:
                k, v = self._paged_cache.update(flex_input_pos, k, v, flex_batch_idx)

        # flex_block_mask is a dict keyed by layer_type; pick the one
        # matching this layer's regime (full_attention vs sliding_attention).
        block_mask = flex_block_mask[self.layer_type]
        attn_output = flex_attention_compiled(
            q,
            k,
            v,
            scale = self.scaling,
            block_mask = block_mask,
            enable_gqa = True,
            kernel_options = flex_kernel_options,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None

    return forward


def patch_gemma4_attention_forwards(model: torch.nn.Module, page_table: PageTable):
    """Attach a PagedKVCache to every non-shared attention layer, link
    every shared attention layer to its store layer's cache, and swap in
    the flex_attention forward above.

    Three passes:
      1. Allocate a PagedKVCache on each non-shared layer (the cache
         shape depends on that layer's head_dim and num_kv_heads, which
         vary across Gemma-4 layers).
      2. Walk shared layers and set
         `shared._paged_cache = store._paged_cache`, where `store` is
         `model.model.layers[shared.kv_shared_layer_index]`. The shared
         layer's forward reads `k_cache` / `v_cache` directly; the store
         layer's `update()` writes populate the same tensors.
      3. Bind the flex forward.
    """
    fwd = make_flex_gemma4_attention_forward(page_table)
    for layer in model.model.layers:
        attn = layer.self_attn
        if getattr(attn, "is_kv_shared_layer", False):
            continue
        n_kv = attn.k_proj.out_features // attn.head_dim
        attn._paged_cache = PagedKVCache(
            page_table,
            n_heads = n_kv,
            head_dim = attn.head_dim,
            dtype = model.dtype,
        ).to(model.device)
    for layer in model.model.layers:
        attn = layer.self_attn
        if not getattr(attn, "is_kv_shared_layer", False):
            continue
        store_attn = model.model.layers[attn.kv_shared_layer_index].self_attn
        attn._paged_cache = store_attn._paged_cache
        attn._store_attn = store_attn
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(fwd, layer.self_attn)


# --- model forward walker --------------------------------------------------


def call_gemma4_model_with_flex_kwargs(model, input_ids, position_ids, flex_kwargs):
    """Walk the Gemma-4 text model manually so we can inject flex_* kwargs
    into each attention call. Mirrors `call_model_with_flex_kwargs` in
    `qwen3_flex_inference.py` but:

      - precomputes both (cos, sin) variants and passes them as a dict
        keyed by `layer.self_attn.layer_type`;
      - materializes `per_layer_inputs` via the model's own
        `get_per_layer_inputs` + `project_per_layer_inputs` helpers;
      - runs the double-residual around attn, the double-residual around
        MLP, the per-layer-input path, and the `layer_scalar` multiply.

    The final norm is applied here; lm_head + softcap is applied by the
    caller (so we can slice to logits_positions before the vocab matmul).
    """
    base = model.model

    inputs_embeds = base.embed_tokens(input_ids)

    # Per-layer input table. E2B has hidden_size_per_layer_input = 256 and
    # 35 layers, so this is [B, S, 35, 256] -- a local tensor with fixed
    # shape across CUDA graph replays (input_ids is pre-allocated upstream).
    per_layer_inputs = None
    if getattr(base, "hidden_size_per_layer_input", 0):
        per_layer_inputs = base.get_per_layer_inputs(input_ids, inputs_embeds)
        per_layer_inputs = base.project_per_layer_inputs(
            inputs_embeds, per_layer_inputs
        )

    position_embeddings = {
        layer_type: base.rotary_emb(inputs_embeds, position_ids, layer_type)
        for layer_type in base.unique_layer_types
    }

    hidden_states = inputs_embeds
    for i, layer in enumerate(base.layers):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states, _ = layer.self_attn(
            hidden_states,
            position_embeddings = position_embeddings,
            **flex_kwargs,
        )
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = layer.pre_feedforward_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = layer.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if per_layer_inputs is not None and hasattr(layer, "per_layer_input_gate"):
            residual = hidden_states
            hidden_states = layer.per_layer_input_gate(hidden_states)
            hidden_states = layer.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_inputs[:, :, i, :]
            hidden_states = layer.per_layer_projection(hidden_states)
            hidden_states = layer.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        if hasattr(layer, "layer_scalar"):
            hidden_states = hidden_states * layer.layer_scalar

    hidden_states = base.norm(hidden_states)
    return hidden_states


# --- inference engine ------------------------------------------------------


class FlexGemma4Inference:
    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size = 32,
        max_seq_length = 2048,
        n_pages = 2048,
        page_size = 128,
        max_new_tokens = 512,
        decode_kernel_options = None,
        prefill_kernel_options = None,
        fa4_prefill = None,
        base_model = None,
        peft_model = None,
    ):
        assert max_seq_length % page_size == 0
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

        if fa4_prefill is None or fa4_prefill:
            major, _ = torch.cuda.get_device_capability(self.device)
            supported = major >= 9
            if fa4_prefill and not supported:
                import warnings

                warnings.warn(
                    f"--fa4_prefill needs Hopper (sm_90) or Blackwell "
                    f"(sm_100 / sm_120); found sm_{major}0. Falling back "
                    f"to the Triton flex_attention backend.",
                    RuntimeWarning,
                    stacklevel = 2,
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

        self.page_table = PageTable(
            n_pages = n_pages,
            page_size = page_size,
            max_batch_size = max_batch_size,
            device = self.device.type,
        )

        patch_gemma4_attention_forwards(model, self.page_table)

        self.input_pos_buffer = torch.zeros(
            max_batch_size, dtype = torch.int32, device = self.device
        )

        # Detect the sliding window from any sliding-attention layer.
        # We need one block mask per layer type: `full_attention` is pure
        # causal; `sliding_attention` is causal AND q_pos - kv_pos < window.
        sliding_window = None
        for layer in model.model.layers:
            if getattr(layer.self_attn, "is_sliding", False):
                sliding_window = layer.self_attn.sliding_window
                break

        self.sliding_window = sliding_window
        self.block_mask_logical_by_type = {
            "full_attention": self.page_table.create_causal_blockmask(
                B = max_batch_size, L = max_seq_length
            ),
        }
        if sliding_window is not None:
            self.block_mask_logical_by_type["sliding_attention"] = (
                _causal_blockmask_with_window(
                    B = max_batch_size,
                    L = max_seq_length,
                    block_size = page_size,
                    window = sliding_window,
                    device = self.device.type,
                )
            )
        # Legacy alias used by the original page-aware decode slicer.
        self.block_mask_logical = self.block_mask_logical_by_type["full_attention"]

        self.cudagraph_captured = False
        self.graphs = {}
        self.graph_vars = {}

    def tokenize(self, sequences):
        for seq in sequences:
            if seq.input_ids is not None and seq.input_length > 0:
                # Pre-tokenized input (see FlexEngine). Skip.
                continue
            ids = self.tokenizer(seq.text, return_tensors = "pt")["input_ids"].squeeze(0)
            seq.input_ids = ids
            seq.input_length = ids.shape[0]

    def _softcap(self, logits):
        sc = getattr(self.model.config, "final_logit_softcapping", None)
        if sc is not None and sc > 0:
            logits = torch.tanh(logits / sc) * sc
        return logits

    def _prefill(self, batch: list) -> torch.Tensor:
        input_ids_list = [seq.input_ids.to(self.device) for seq in batch]
        input_pos_list = [
            torch.arange(seq.input_length, dtype = torch.long, device = self.device)
            for seq in batch
        ]
        batch_idx_list = [
            torch.full(
                (seq.input_length,),
                seq.batch_idx,
                dtype = torch.long,
                device = self.device,
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
            input_ids = F.pad(input_ids, (0, pad), value = 0)
            input_pos = F.pad(input_pos, (0, pad), value = 0)
            batch_idx = F.pad(batch_idx, (0, pad), value = 0)

        input_lengths = torch.tensor(
            [s.input_length for s in batch], dtype = torch.long, device = self.device
        )
        logits_positions = input_lengths.cumsum(dim = 0) - 1

        prefill_block_size = (
            (self.prefill_q_block, self.prefill_kv_block)
            if self.fa4_prefill
            else self.prefill_q_block
        )
        mask_full = self.page_table.create_prefill_blockmask_no_paging(
            batch_idx, BLOCK_SIZE = prefill_block_size
        )
        masks_by_type = {"full_attention": mask_full}
        if self.sliding_window is not None:
            masks_by_type["sliding_attention"] = _prefill_blockmask_with_window(
                batch_idx,
                block_size = prefill_block_size,
                window = self.sliding_window,
            )

        flex_kwargs = dict(
            flex_block_mask = masks_by_type,
            flex_input_pos = input_pos,
            flex_batch_idx = batch_idx,
            flex_kernel_options = self.prefill_kernel_options,
        )
        hidden = call_gemma4_model_with_flex_kwargs(
            self.model, input_ids, input_pos, flex_kwargs
        )
        logits = self.model.lm_head(hidden[:, logits_positions, :]).squeeze(0)
        return self._softcap(logits)

    def _decode_block_mask(self, batch_idx: torch.Tensor):
        """Slice one row of the logical decode mask per sequence, for
        both full and sliding regimes. Returns a dict keyed by layer_type
        plus the raw `input_pos` tensor (needed for PageTable conversion)."""
        input_pos = self.input_pos_buffer[batch_idx]
        assert batch_idx.ndim == 1 and input_pos.ndim == 1
        B = batch_idx.shape[0]

        def _slice(block_mask, extra_mask_mod):
            input_block_idx = input_pos // block_mask.BLOCK_SIZE[0]
            kv_num_blocks = block_mask.kv_num_blocks[
                batch_idx, :, input_block_idx
            ].view(B, 1, 1)
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

            seq_length = (1, block_mask.seq_lengths[1])
            return BlockMask.from_kv_blocks(
                kv_num_blocks,
                kv_indices,
                full_num,
                full_idx,
                BLOCK_SIZE = block_mask.BLOCK_SIZE,
                mask_mod = extra_mask_mod,
                seq_lengths = seq_length,
            )

        def causal_offset(off):
            def m(b, h, q_idx, kv_idx):
                return q_idx + off[b] >= kv_idx

            return m

        def causal_offset_windowed(off, window):
            def m(b, h, q_idx, kv_idx):
                return (q_idx + off[b] >= kv_idx) & (q_idx + off[b] - kv_idx < window)

            return m

        masks = {
            "full_attention": _slice(
                self.block_mask_logical_by_type["full_attention"],
                causal_offset(input_pos),
            ),
        }
        if self.sliding_window is not None:
            masks["sliding_attention"] = _slice(
                self.block_mask_logical_by_type["sliding_attention"],
                causal_offset_windowed(input_pos, self.sliding_window),
            )
        return masks, input_pos

    def _decode_step_eager(self, batch_idx: torch.Tensor, input_ids: torch.Tensor):
        B = input_ids.shape[0]
        masks, input_pos = self._decode_block_mask(batch_idx)
        # Convert each regime's block mask through the page table so the
        # logical→physical kv page mapping is correct for every layer.
        masks = {
            k: self.page_table.convert_logical_block_mask(m, batch_idx)
            for k, m in masks.items()
        }
        position_ids = input_pos.view(B, 1).to(torch.long)
        flex_kwargs = dict(
            flex_block_mask = masks,
            flex_input_pos = input_pos.view(B, 1).to(torch.long),
            flex_batch_idx = batch_idx,
            flex_kernel_options = self.decode_kernel_options,
        )
        hidden = call_gemma4_model_with_flex_kwargs(
            self.model, input_ids.view(B, 1), position_ids, flex_kwargs
        )
        logits = self.model.lm_head(hidden[:, -1, :])
        return self._softcap(logits)

    def _decode_step(
        self,
        batch_idx: torch.Tensor,
        input_ids: torch.Tensor,
        input_pos: torch.Tensor,
    ):
        self.input_pos_buffer.zero_()
        self.input_pos_buffer[batch_idx] = input_pos
        if not self.cudagraph_captured:
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
        max_bs = self.max_batch_size
        reserved_batches = []
        for bi in range(1, max_bs):
            try:
                allocated = self.page_table.allocate()
                self.page_table.reserve(
                    allocated,
                    torch.tensor([allocated], device = self.device, dtype = torch.long),
                    self.page_size,
                )
                reserved_batches.append(allocated)
            except Exception:
                break

        input_ids = torch.zeros(max_bs, dtype = torch.int64, device = self.device)
        batch_idx = torch.arange(max_bs, dtype = torch.int64, device = self.device)
        outputs = torch.zeros(
            (max_bs, self.model.config.vocab_size),
            dtype = self.model.dtype,
            device = self.device,
        )
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        pool = None
        for bs in reversed(self.graph_bs):
            if bs > max_bs:
                continue
            print(f"[flex-gemma4] capturing CUDA graph for bs={bs}")
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
            input_ids = input_ids, batch_idx = batch_idx, outputs = outputs
        )

    def refresh_inference_from_base(self):
        if self.base_model is None or self.peft_model is None:
            return 0
        return refresh_lora_merge_from_pristine(self.base_model, self.peft_model)

    @torch.inference_mode()
    def generate(self, sequences, capture_cudagraph = False):
        self.tokenize(sequences)
        waiting = deque(sequences)
        running = deque()
        done = []

        if capture_cudagraph and not self.cudagraph_captured:
            self.capture_decode_cudagraph()
            self.cudagraph_captured = True

        while waiting or running:
            batch = []
            while waiting and self.page_table.can_reserve(waiting[0].total_length):
                seq = waiting.popleft()
                bi = self.page_table.allocate()
                self.page_table.reserve(
                    bi,
                    torch.tensor([bi], device = self.device, dtype = torch.long),
                    seq.total_length,
                )
                seq.batch_idx = bi
                batch.append(seq)
            if batch:
                logits = self._prefill(batch)
                next_ids = torch.argmax(logits, dim = -1).tolist()
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
                    seq.total_length, batch_idx_int = seq.batch_idx
                ):
                    self.page_table.reserve(
                        seq.batch_idx,
                        torch.tensor(
                            [seq.batch_idx],
                            device = self.device,
                            dtype = torch.long,
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
                dtype = torch.long,
                device = self.device,
            )
            last_ids = torch.tensor(
                [s.last_token_id for s in decode_batch],
                dtype = torch.long,
                device = self.device,
            )
            cur_pos = torch.tensor(
                [s.total_length - 1 for s in decode_batch],
                dtype = torch.int32,
                device = self.device,
            )
            logits = self._decode_step(bi_tensor, last_ids, cur_pos)
            next_ids = torch.argmax(logits, dim = -1).tolist()
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


# --- CLI -------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default = "unsloth/gemma-4-E2B-it")
    p.add_argument("--n_prompts", type = int, default = 64)
    p.add_argument("--n_rounds", type = int, default = 2)
    p.add_argument("--max_new_tokens", type = int, default = 512)
    p.add_argument("--max_batch_size", type = int, default = 64)
    p.add_argument("--max_seq_length", type = int, default = 2048)
    p.add_argument("--n_pages", type = int, default = 2048)
    p.add_argument("--page_size", type = int, default = 128)
    p.add_argument("--capture_cudagraph", action = "store_true")
    p.add_argument("--lora_adapter", default = None)
    p.add_argument("--decode_kernel_options", default = None)
    p.add_argument("--prefill_kernel_options", default = None)
    p.add_argument(
        "--fa4_prefill",
        default = None,
        action = argparse.BooleanOptionalAction,
        help = (
            "Use BLOCK_SIZE=(256,128) + BACKEND=FLASH on prefill. Default "
            "auto-enables on Hopper (sm_90) and Blackwell (sm_100, sm_120)."
        ),
    )
    p.add_argument("--load_in_4bit", action = "store_true")
    p.add_argument(
        "--no_merge_lora",
        action = "store_true",
        help = "Keep the LoRA adapter as a PEFT wrapper instead of merging.",
    )
    p.add_argument(
        "--verify_no_drift",
        action = "store_true",
        help = "Drift-verify the double-copy LoRA refresh across N cycles.",
    )
    p.add_argument("--verify_iterations", type = int, default = 10)
    p.add_argument("--model_name_4bit", default = None)
    p.add_argument("--stats_path", required = True)
    p.add_argument(
        "--chat_template",
        choices = ["auto", "grpo", "native"],
        default = "auto",
        help = (
            "Which chat template to use. `auto`: native for Gemma-4. "
            "`grpo`: force the GRPO template. `native`: force the "
            "tokenizer's built-in template."
        ),
    )
    args = p.parse_args()

    def _parse_opts(s):
        if s is None:
            return None
        return json.loads(s)

    Gemma4ForCausalLM, Gemma4Config, Gemma4TextConfig = _require_gemma4()

    from transformers import AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration,
    )

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = None
    peft_model = None

    if args.load_in_4bit:
        from transformers import AutoModelForCausalLM
        from huggingface_hub import HfApi

        bnb_model_name = args.model_name_4bit or f"{args.model_name}-unsloth-bnb-4bit"
        # Probe for the 4-bit shard. If missing, the user asked for a
        # quant row we cannot produce; fail loudly rather than silently
        # falling back to bf16 (which would mislabel the stats file).
        try:
            HfApi().model_info(bnb_model_name)
        except Exception as e:
            raise SystemExit(
                f"[flex-gemma4] --load_in_4bit: 4-bit shard {bnb_model_name} "
                f"is not available ({e}). Use --model_name_4bit to override "
                f"or drop --load_in_4bit for bf16."
            )
        print(f"[flex-gemma4] loading 4-bit base: {bnb_model_name}")
        # AutoModelForCausalLM resolves to `Gemma4ForConditionalGeneration`
        # for Gemma-4 -- mirror the bf16 path and move the
        # language_model into a ForCausalLM shell so downstream code can
        # reach `model.model.layers` / `model.model.embed_tokens`.
        full_model = AutoModelForCausalLM.from_pretrained(
            bnb_model_name,
            attn_implementation = "eager",
            device_map = "cuda:0",
        )
        if hasattr(full_model.model, "language_model"):
            lang_model = full_model.model.language_model
            full_model.model.vision_tower = None
            full_model.model.audio_tower = None
            full_model.model.embed_vision = None
            full_model.model.embed_audio = None
            text_cfg = full_model.config.text_config
            model = Gemma4ForCausalLM(text_cfg)
            model.model = lang_model
            model.lm_head.weight = lang_model.embed_tokens.weight
        else:
            model = full_model
            if getattr(model.config, "tie_word_embeddings", False):
                model.lm_head.weight = model.model.embed_tokens.weight
        model.eval()
        del full_model
        if args.lora_adapter:
            from peft import PeftModel

            peft_wrapper = PeftModel.from_pretrained(
                model,
                str(Path(args.lora_adapter).resolve()),
                is_trainable = False,
            )
            model = peft_wrapper.base_model.model
    else:
        # Text-only bf16. The checkpoint stores text weights under the
        # `model.language_model.` prefix (Gemma-4 is natively multimodal).
        # We load the full `Gemma4ForConditionalGeneration`, pluck the
        # language_model, then drop the vision + audio towers before
        # moving to GPU so peak memory reflects text only.
        full_cfg = Gemma4Config.from_pretrained(args.model_name)
        text_cfg = full_cfg.text_config

        full_model = Gemma4ForConditionalGeneration.from_pretrained(
            args.model_name,
            dtype = torch.bfloat16,
            attn_implementation = "eager",
        )
        lang_model = full_model.model.language_model
        # Drop the non-text towers. `embed_vision` / `embed_audio` project
        # from text hidden size -- harmless when not invoked, but we kill
        # them too so deepcopy (below) stays cheap.
        full_model.model.vision_tower = None
        full_model.model.audio_tower = None
        full_model.model.embed_vision = None
        full_model.model.embed_audio = None

        # Build a ForCausalLM shell around the language_model so LoRA /
        # state-dict hashing treat it like any other HF decoder model.
        base_model = Gemma4ForCausalLM(text_cfg)
        base_model.model = lang_model
        base_model.lm_head.weight = lang_model.embed_tokens.weight
        base_model = base_model.to(torch.bfloat16).to("cuda")
        base_model.eval()
        del full_model

        if not args.lora_adapter:
            model = base_model
            base_model = None
        elif args.no_merge_lora:
            from peft import PeftModel

            peft_wrapper = PeftModel.from_pretrained(
                base_model,
                str(Path(args.lora_adapter).resolve()),
                is_trainable = False,
            )
            model = peft_wrapper.base_model.model
            base_model = None
        else:
            from peft import PeftModel

            print("[flex-gemma4] deep-copying base model for double-copy rollout")
            inference_model = copy.deepcopy(base_model)
            inference_model.eval()
            peft_model = PeftModel.from_pretrained(
                inference_model,
                str(Path(args.lora_adapter).resolve()),
                is_trainable = False,
            )
            model = peft_model.base_model.model
            model.eval()

    if args.verify_no_drift:
        if args.load_in_4bit:
            raise SystemExit(
                "--verify_no_drift only applies to the bf16 double-copy path."
            )
        if args.no_merge_lora:
            raise SystemExit("--verify_no_drift is incompatible with --no_merge_lora.")
        if base_model is None or peft_model is None:
            raise SystemExit(
                "--verify_no_drift requires --lora_adapter against the bf16 path."
            )
        print(
            f"[flex-gemma4] running drift verification: "
            f"{args.verify_iterations} perturb+refresh cycles"
        )
        result = run_drift_verification(
            base_model, peft_model, n_iters = args.verify_iterations
        )
        result = {"mode": "verify_no_drift", "model_name": args.model_name, **result}
        os.makedirs(
            os.path.dirname(os.path.abspath(args.stats_path)) or ".",
            exist_ok = True,
        )
        with open(args.stats_path, "w") as f:
            json.dump(result, f, indent = 2)
        print(json.dumps(result, indent = 2))
        os._exit(0)

    from unsloth_grpo_common import SYSTEM_PROMPT, apply_chat_template_to_tokenizer
    from datasets import load_dataset

    if args.chat_template == "auto":
        # Gemma-4 default is the tokenizer native template; only Qwen3
        # in this repo uses GRPO-by-default.
        use_grpo = False
    elif args.chat_template == "grpo":
        use_grpo = True
    else:
        use_grpo = False
    if use_grpo:
        apply_chat_template_to_tokenizer(tok)
        print("[flex-gemma4] chat_template: GRPO")
    else:
        print("[flex-gemma4] chat_template: tokenizer native")
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split = "train")
    ds = ds.shuffle(seed = 3407).select(range(args.n_prompts))
    messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["prompt"]},
        ]
        for x in ds
    ]
    texts = [
        tok.apply_chat_template(m, add_generation_prompt = True, tokenize = False)
        for m in messages
    ]

    inference = FlexGemma4Inference(
        model,
        tok,
        max_batch_size = args.max_batch_size,
        max_seq_length = args.max_seq_length,
        n_pages = args.n_pages,
        page_size = args.page_size,
        max_new_tokens = args.max_new_tokens,
        decode_kernel_options = _parse_opts(args.decode_kernel_options),
        prefill_kernel_options = _parse_opts(args.prefill_kernel_options),
        fa4_prefill = args.fa4_prefill,
        base_model = base_model,
        peft_model = peft_model,
    )

    if inference.base_model is not None and inference.peft_model is not None:
        n = inference.refresh_inference_from_base()
        print(f"[flex-gemma4] double-copy rollout: refreshed {n} LoRA-target layers")

    def make_seqs():
        return [Sequence(text = t, max_new_tokens = args.max_new_tokens) for t in texts]

    torch.cuda.reset_peak_memory_stats()
    print("[flex-gemma4] warmup (16 prompts)...")
    _ = inference.generate(make_seqs()[:16], capture_cudagraph = args.capture_cudagraph)
    torch.cuda.synchronize()

    wall_times = []
    total_decoded = 0
    for r in range(args.n_rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = inference.generate(make_seqs())
        torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - t0)
        total_decoded = sum(len(s.output_ids) for s in out)
        print(
            f"[flex-gemma4] round {r}: {wall_times[-1]:.2f}s, {total_decoded} "
            f"tokens, {total_decoded / wall_times[-1]:.1f} tok/s"
        )

    med = sorted(wall_times)[len(wall_times) // 2]
    best = min(wall_times)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    sample_completions = []
    for s in out[:3]:
        sample_completions.append(
            tok.decode(s.output_ids[:80], skip_special_tokens = True)
        )
    res = {
        "backend": "flex-gemma4",
        "model_name": args.model_name,
        "capture_cudagraph": args.capture_cudagraph,
        "lora_adapter": args.lora_adapter,
        "n_prompts": args.n_prompts,
        "n_decoded_tokens": total_decoded,
        "wall_times_s": wall_times,
        "median_wall_s": med,
        "best_wall_s": best,
        "decode_tps_median": total_decoded / med if med else 0,
        "decode_tps_best": total_decoded / best if best else 0,
        "max_new_tokens": args.max_new_tokens,
        "peak_memory_gb": peak,
        "sample_completions": sample_completions,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.stats_path)) or ".", exist_ok = True)
    with open(args.stats_path, "w") as f:
        json.dump(res, f, indent = 2)
    print(json.dumps(res, indent = 2))
    os._exit(0)


if __name__ == "__main__":
    main()
