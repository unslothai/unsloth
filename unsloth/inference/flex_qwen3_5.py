# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Flex-attention inference backend for Qwen3.5 / Qwen3.6 dense and MoE.

Qwen3.5 / Qwen3.6 text models use a hybrid layer stack: every 4th layer is a
standard full-attention block (partial rotary_factor=0.25, attn_output_gate,
per-head Q/K RMSNorm), the remaining 3 of 4 are Gated DeltaNet linear
attention blocks backed by ``flash-linear-attention``'s Triton kernels
(``fused_recurrent_gated_delta_rule`` for decode,
``chunk_gated_delta_rule`` for prefill).

Flex Attention can only express softmax attention, so the 75% linear_attn
layers can't use it directly. This engine therefore dispatches per layer:

- ``full_attention`` layers: swap ``Qwen3_5Attention.forward`` with a
  flex_attention-based forward bound to a shared ``PageTable`` (same
  machinery as :mod:`flex_qwen3_llama`).
- ``linear_attention`` layers: keep HF's ``Qwen3_5GatedDeltaNet.forward``
  intact. It calls FLA directly when the fast path is installed and falls
  back to the pure-torch reference kernels otherwise.

State management for linear_attn layers is delegated to HF's
``DynamicCache`` populated with ``LinearAttentionAndFullAttentionLayer``
layers. That class owns both a conv_state (size ``[B, conv_dim, 4]``)
and a recurrent_state (``[B, num_v_heads, head_k_dim, head_v_dim]``
float32) per layer. Full-attn layers use the same layer type but the
``DynamicLayer`` half is bypassed because the flex forward writes
directly into the PageTable-backed paged KV cache.

This is a minimal first-iteration backend. See
tests/qwen3_5_flex_parity.py for the smoke. Follow-ups:
- CUDA graph capture for the decode step.
- torch.compile on the walker.
- Qwen3.5-MoE: expert dispatch uses the existing ``forward_moe_backend``
  from moe_utils; the layer-level dispatch in this file is already
  arch-agnostic to the expert kernel.
"""
from __future__ import annotations

import os
import types
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from .flex_paged_attention import PagedKVCache, PageTable


# Same pattern as flex_qwen3_llama: compile once at import for warm caches.
_FLEX_COMPILE_MODE = os.environ.get("FLEX_COMPILE_MODE", None)
if _FLEX_COMPILE_MODE:
    flex_attention_compiled = torch.compile(
        flex_attention, fullgraph=True, mode=_FLEX_COMPILE_MODE,
    )
else:
    flex_attention_compiled = torch.compile(flex_attention, fullgraph=True)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_partial_rotary(q, k, cos, sin):
    """Qwen3.5's partial_rotary_factor=0.25 form.

    ``cos`` / ``sin`` arrive shaped ``[B, S, rotary_dim]`` where
    ``rotary_dim = head_dim * 0.25 = 64`` for head_dim=256. Split q / k
    along ``head_dim`` into a rotary prefix and an untouched tail,
    apply rotary to the prefix, concat back. Mirrors
    ``modeling_qwen3_5.apply_rotary_pos_emb``.
    """
    cos = cos.unsqueeze(1)  # [B, 1, S, rotary_dim]
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


def make_qwen3_5_flex_attention_forward(page_table: PageTable):
    """Swap-in for ``Qwen3_5Attention.forward`` that uses flex_attention
    + paged KV cache. Handles:
    - attn_output_gate: ``q_proj`` outputs ``2 * hidden_shape`` and we
      chunk into (query, gate); the attn_output is multiplied by
      ``sigmoid(gate)`` before ``o_proj``.
    - per-head Q/K RMSNorm before rotary.
    - partial rotary (factor=0.25) via :func:`_apply_partial_rotary`.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        flex_block_mask: Optional[BlockMask] = None,
        flex_input_pos: Optional[torch.Tensor] = None,
        flex_batch_idx: Optional[torch.Tensor] = None,
        flex_kernel_options: Optional[dict] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # q_proj output: [..., n_heads, 2 * head_dim]; split into q / gate.
        q_and_gate = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)  # [..., n_heads * head_dim]

        # q_norm / k_norm apply per-head RMSNorm on the head_dim axis.
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_partial_rotary(
            query_states, key_states, cos, sin,
        )

        # Write to paged KV cache. For prefill assign_prefill_no_paging
        # writes into [1, H, MAX_S, D]; for decode assign() writes into
        # B decode slots.
        if self._paged_cache is not None and flex_input_pos is not None:
            cache_dtype = self._paged_cache.k_cache.dtype
            if key_states.dtype != cache_dtype:
                key_states = key_states.to(cache_dtype)
            if value_states.dtype != cache_dtype:
                value_states = value_states.to(cache_dtype)
            key_states, value_states = self._paged_cache.update(
                flex_input_pos, key_states, value_states, flex_batch_idx,
            )

        attn_output = flex_attention_compiled(
            query_states,
            key_states,
            value_states,
            scale=self.scaling,
            block_mask=flex_block_mask,
            enable_gqa=True,
            kernel_options=flex_kernel_options,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    return forward


def _patch_qwen3_5_full_attn_forwards(text_model, page_table: PageTable):
    """Walk ``text_model.layers`` and, for each layer whose ``layer_type ==
    'full_attention'``, attach a ``PagedKVCache`` and swap in the flex
    attention forward. Linear_attn layers are left alone (HF forward
    dispatches to FLA directly).

    ``text_model`` is ``Qwen3_5TextModel`` or ``Qwen3_5MoeTextModel`` —
    both expose ``.layers`` / ``.config``. The outer multimodal
    wrapper's ``.model.language_model`` is what the engine resolves and
    passes in.
    """
    cfg = text_model.config
    fwd = make_qwen3_5_flex_attention_forward(page_table)
    for layer_idx, layer in enumerate(text_model.layers):
        if cfg.layer_types[layer_idx] != "full_attention":
            continue
        attn = layer.self_attn
        attn._paged_cache = PagedKVCache(
            page_table,
            n_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            dtype=text_model.dtype,
        ).to(text_model.dtype).to(next(text_model.parameters()).device)
        attn.forward = types.MethodType(fwd, attn)


def _resolve_text_model(model: torch.nn.Module):
    """Qwen3.5 / Qwen3.6 ship the text backbone two levels down:
    ``Qwen3_5ForConditionalGeneration.model.language_model`` is the
    Qwen3_5TextModel (or Qwen3_5MoeTextModel) that owns ``.layers``,
    ``.embed_tokens``, ``.norm``, ``.rotary_emb``.
    """
    base = getattr(model, "model", model)
    lang = getattr(base, "language_model", None)
    if lang is not None and hasattr(lang, "layers"):
        return lang
    if hasattr(base, "layers"):
        return base
    # PEFT wrapper.
    inner = getattr(model, "base_model", None)
    if inner is not None:
        return _resolve_text_model(getattr(inner, "model", inner))
    raise RuntimeError(
        "Cannot locate the Qwen3.5 text backbone (expected "
        "``.model.language_model.layers``)"
    )


def _call_qwen3_5_with_flex_kwargs(
    text_model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    full_attn_flex_kwargs: dict,
    linear_cache,
    linear_attn_mask,
    *,
    lm_head_fn,
):
    """Per-layer walker.

    - ``full_attention`` layers: called with ``**full_attn_flex_kwargs``
      (block_mask / input_pos / batch_idx / kernel_options). The patched
      attention forward consumes these; ``past_key_values`` is ignored.
    - ``linear_attention`` layers: HF's ``Qwen3_5GatedDeltaNet.forward``
      needs ``cache_params=<DynamicCache>`` and the linear_attn_mask
      (left-padding aware, ``None`` during cached decode).
    - Residual / MLP / layernorms are the stock HF modules so we run
      them inline without swapping.
    """
    cfg = text_model.config
    inputs_embeds = text_model.embed_tokens(input_ids)

    # Qwen3_5TextRotaryEmbedding internally expands 2D position_ids to
    # 3D via [None, ...].expand(3, B, S). For text-only, all 3 copies
    # are identical so the result collapses to standard rope on the
    # first ``partial_rotary_factor`` fraction of head_dim.
    position_embeddings = text_model.rotary_emb(inputs_embeds, position_ids)

    hidden_states = inputs_embeds
    layer_types = cfg.layer_types

    for layer_idx, layer in enumerate(text_model.layers):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        if layer_types[layer_idx] == "linear_attention":
            hidden_states = layer.linear_attn(
                hidden_states=hidden_states,
                cache_params=linear_cache,
                attention_mask=linear_attn_mask,
            )
        else:
            hidden_states, _ = layer.self_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                **full_attn_flex_kwargs,
            )

        hidden_states = residual + hidden_states

        # MLP path. For dense Qwen3_5 the layer has ``layer.mlp``; for
        # Qwen3_5Moe it has the same name, but the submodule is a
        # Qwen3_5MoeSparseMoeBlock that owns gate + experts + shared
        # expert. Both implement ``forward(hidden_states) -> hidden``
        # so we don't dispatch here.
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = text_model.norm(hidden_states)
    return lm_head_fn(hidden_states)


@dataclass
class Sequence:
    text: str = ""
    input_ids: Optional[torch.Tensor] = None
    input_length: int = 0
    output_ids: Optional[list] = None
    batch_idx: int = -1
    finished: bool = False
    last_token_id: int = -1
    max_new_tokens: int = 512
    _linear_cache = None  # per-seq DynamicCache for linear_attn state

    def __post_init__(self):
        if self.output_ids is None:
            self.output_ids = []

    @property
    def total_length(self) -> int:
        return self.input_length + len(self.output_ids)


DECODE_KERNEL_OPTIONS_DEFAULT = None
PREFILL_KERNEL_OPTIONS_DEFAULT = {"FORCE_USE_FLEX_ATTENTION": True}


class FlexQwen3_5Inference:
    """Flex-attention backend for Qwen3.5 / Qwen3.6 dense and MoE."""

    arch = "qwen3_5"

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 8,
        max_seq_length: int = 2048,
        n_pages: int = 2048,
        page_size: int = 128,
        max_new_tokens: int = 512,
        decode_kernel_options=None,
        prefill_kernel_options=None,
        fa4_prefill=None,
        base_model=None,
        peft_model=None,
        cumem_allocator=None,
    ):
        assert max_seq_length % page_size == 0
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.eos_token_id = tokenizer.eos_token_id
        self.base_model = base_model
        self.peft_model = peft_model
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.page_size = page_size
        self.max_new_tokens = max_new_tokens

        self.fa4_prefill = bool(fa4_prefill)
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

        # Resolve the text backbone so we walk .layers / .rotary_emb
        # directly, bypassing the outer multimodal wrapper's complex
        # forward. lm_head still lives on the outer wrapper.
        self.text_model = _resolve_text_model(model)
        if hasattr(model, "lm_head"):
            self._lm_head = model.lm_head
        else:
            # Multimodal wrapper routes lm_head through .model.
            self._lm_head = model.model.lm_head

        # Page table for the full-attention layers' paged KV cache.
        from .sleep_mode import kv_cache_pool as _kv_cache_pool
        with _kv_cache_pool(cumem_allocator):
            self.page_table = PageTable(
                n_pages=n_pages,
                page_size=page_size,
                max_batch_size=max_batch_size,
                device=self.device.type,
            )
            _patch_qwen3_5_full_attn_forwards(self.text_model, self.page_table)

        self.input_pos_buffer = torch.zeros(
            max_batch_size, dtype=torch.int32, device=self.device,
        )
        self.block_mask_logical = self.page_table.create_causal_blockmask(
            B=max_batch_size, L=max_seq_length,
        )

        # Pre-allocate per-slot LinearAttention caches. Each sequence's
        # decode loop reuses the same DynamicCache object for its slot;
        # we reset it via .reset() at prefill so the conv/recurrent
        # states start clean.
        self._linear_caches = [
            self._build_linear_cache() for _ in range(max_batch_size)
        ]

    # ----- cache helpers -------------------------------------------------
    def _build_linear_cache(self):
        """Allocate a ``DynamicCache`` seeded from ``model.config`` so
        ``config.layer_types`` dictates the per-layer cache class:
        ``"linear_attention"`` → ``LinearAttentionLayer``
        (owns conv/recurrent state) and ``"full_attention"`` →
        ``DynamicLayer`` (dense KV). The full-attn DynamicLayer half is
        untouched in this engine (the flex forward writes directly into
        the paged KV cache) — only the linear layers actually see use.
        """
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache(config=self.model.config)
        return cache

    # ----- prefill -------------------------------------------------------
    def tokenize(self, sequences):
        for seq in sequences:
            if seq.input_ids is not None and seq.input_length > 0:
                continue
            ids = self.tokenizer(seq.text, return_tensors="pt")["input_ids"].squeeze(0)
            seq.input_ids = ids
            seq.input_length = ids.shape[0]

    @torch.inference_mode()
    def _prefill_single(self, seq: Sequence) -> int:
        """One-sequence prefill. Unlike :class:`FlexInference`, we can't
        trivially pack multiple linear_attn sequences into a [1, L_total]
        run because the conv1d and recurrent state carry independent
        context per sequence. First iteration runs one prefill per seq.
        Full-attn layers still use document-causal paging internally so
        the paged KV cache is populated correctly."""
        device = self.device
        input_ids = seq.input_ids.to(device).view(1, -1)
        L = seq.input_length
        batch_idx = torch.full((1, L), seq.batch_idx, dtype=torch.long, device=device)
        input_pos = torch.arange(L, dtype=torch.long, device=device).view(1, -1)

        # Pad packed prefill to the flex_attention Q-block boundary.
        q_block = self.prefill_q_block
        pad = (q_block - L % q_block) % q_block
        if pad > 0:
            input_ids = F.pad(input_ids, (0, pad), value=0)
            input_pos = F.pad(input_pos, (0, pad), value=0)
            batch_idx = F.pad(batch_idx, (0, pad), value=seq.batch_idx)

        prefill_block_size = (
            (self.prefill_q_block, self.prefill_kv_block)
            if self.fa4_prefill
            else self.prefill_q_block
        )
        mask = self.page_table.create_prefill_blockmask_no_paging(
            batch_idx, BLOCK_SIZE=prefill_block_size,
        )
        full_attn_kwargs = dict(
            flex_block_mask=mask,
            flex_input_pos=input_pos,
            flex_batch_idx=batch_idx,
            flex_kernel_options=self.prefill_kernel_options,
        )
        linear_cache = seq._linear_cache
        # HF uses torch.all(attention_mask == 1) to decide to drop the
        # mask. For a fresh prefill pass None, meaning the DeltaNet
        # layer computes without a padding mask (our single packed
        # sequence has no padding).
        logits = _call_qwen3_5_with_flex_kwargs(
            self.text_model,
            input_ids,
            input_pos,
            full_attn_kwargs,
            linear_cache,
            linear_attn_mask=None,
            lm_head_fn=self._lm_head,
        )
        # Logits at the last real token (before pad) — position L-1.
        return int(torch.argmax(logits[0, L - 1, :]).item())

    @torch.inference_mode()
    def _decode_step_eager(
        self,
        batch_idx: torch.Tensor,
        input_ids: torch.Tensor,
        linear_caches: list,
    ):
        """Single decode step across a batch of sequences. We run the
        DeltaNet layers with each sequence's own cache sequentially —
        the FLA recurrent kernel is per-sequence. The full_attn layers
        are batched (paged KV + flex_attention).

        Returns logits [B, V].
        """
        B = input_ids.shape[0]
        mask, input_pos = self._decode_block_mask(batch_idx)
        mask = self.page_table.convert_logical_block_mask(mask, batch_idx)
        position_ids = input_pos.view(B, 1).to(torch.long)

        full_attn_kwargs = dict(
            flex_block_mask=mask,
            flex_input_pos=input_pos.view(B, 1).to(torch.long),
            flex_batch_idx=batch_idx,
            flex_kernel_options=self.decode_kernel_options,
        )

        # Path B: per-sequence decode. The FLA recurrent kernel expects
        # a single ``initial_state`` tensor per call; the state lives in
        # each sequence's DynamicCache. Batched decode across different
        # cache tensors would require fused batched FLA kernels which
        # aren't wired up yet. Serialize the DeltaNet halves of the
        # step, batch the full-attn halves.
        #
        # Simplest working implementation: decode one sequence at a time
        # end-to-end. The per-seq paged block mask already slices to
        # that row, so the full-attn math still runs correctly.
        logits_list = []
        for i in range(B):
            single_bi = batch_idx[i : i + 1]
            single_ids = input_ids[i : i + 1].view(1, 1)
            single_mask, single_pos = self._decode_block_mask(single_bi)
            single_mask = self.page_table.convert_logical_block_mask(
                single_mask, single_bi,
            )
            single_kwargs = dict(
                flex_block_mask=single_mask,
                flex_input_pos=single_pos.view(1, 1).to(torch.long),
                flex_batch_idx=single_bi,
                flex_kernel_options=self.decode_kernel_options,
            )
            logits = _call_qwen3_5_with_flex_kwargs(
                self.text_model,
                single_ids,
                single_pos.view(1, 1).to(torch.long),
                single_kwargs,
                linear_caches[i],
                linear_attn_mask=None,
                lm_head_fn=self._lm_head,
            )
            logits_list.append(logits[0, -1, :])
        return torch.stack(logits_list, dim=0)

    def _decode_block_mask(self, batch_idx: torch.Tensor):
        """Slice a single-row BlockMask for every seq in the decode batch,
        then translate logical→physical pages. Copy of the same helper
        in ``FlexInference`` — dedup if this pattern grows a third
        consumer."""
        block_mask = self.block_mask_logical
        input_pos = self.input_pos_buffer[batch_idx]
        B = batch_idx.shape[0]
        input_block_idx = input_pos // block_mask.BLOCK_SIZE[0]
        kv_num_blocks = block_mask.kv_num_blocks[batch_idx, :, input_block_idx].view(
            B, 1, 1,
        )
        kv_indices = block_mask.kv_indices[batch_idx, :, input_block_idx].view(
            B, 1, 1, -1,
        )
        full_num = full_idx = None
        if block_mask.full_kv_num_blocks is not None:
            full_num = block_mask.full_kv_num_blocks[
                batch_idx, :, input_block_idx
            ].view(B, 1, 1)
            full_idx = block_mask.full_kv_indices[batch_idx, :, input_block_idx].view(
                B, 1, 1, -1,
            )

        def causal_offset(off):
            def offset(b, h, q_idx, kv_idx):
                return q_idx + off[b] >= kv_idx
            return offset

        seq_length = (1, block_mask.seq_lengths[1])
        mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_num,
            full_idx,
            BLOCK_SIZE=block_mask.BLOCK_SIZE,
            mask_mod=causal_offset(input_pos),
            seq_lengths=seq_length,
        )
        return mask, input_pos

    def _reset_linear_cache(self, seq: Sequence):
        """Fresh linear-attn cache for a newly-scheduled sequence."""
        seq._linear_cache = self._build_linear_cache()

    # ----- generate loop -------------------------------------------------
    @torch.inference_mode()
    def generate(self, sequences: list, capture_cudagraph: bool = False):
        """Main entry; mirror of FlexInference.generate but with the
        per-sequence decode loop described above. CUDA graph capture is
        deferred — the per-seq DeltaNet call makes graph capture
        non-trivial and is a follow-up."""
        del capture_cudagraph  # TODO: add once batched DeltaNet exists.
        self.tokenize(sequences)
        waiting = deque(sequences)
        running = deque()
        done = []

        while waiting or running:
            # Schedule waiting -> running via prefill.
            if waiting and self.page_table.can_reserve(waiting[0].total_length):
                seq = waiting.popleft()
                bi = self.page_table.allocate()
                self.page_table.reserve(
                    bi,
                    torch.tensor([bi], device=self.device, dtype=torch.long),
                    seq.total_length,
                )
                seq.batch_idx = bi
                self._reset_linear_cache(seq)
                next_id = self._prefill_single(seq)
                seq.last_token_id = next_id
                seq.output_ids.append(next_id)
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

            if not running:
                # No running seqs yet and prefill budget exhausted; wait.
                # In practice the page table can always reserve at least
                # one pending prompt thanks to the headroom factor.
                break

            # Decode step for everything running.
            decode_batch = []
            while running:
                seq = running.popleft()
                needed = seq.total_length
                if self.page_table.capacity[seq.batch_idx] >= needed:
                    decode_batch.append(seq)
                elif self.page_table.can_reserve(
                    needed, batch_idx_int=seq.batch_idx,
                ):
                    self.page_table.reserve(
                        seq.batch_idx,
                        torch.tensor(
                            [seq.batch_idx],
                            device=self.device,
                            dtype=torch.long,
                        ),
                        needed,
                    )
                    decode_batch.append(seq)
                else:
                    # Evict: push newest back to waiting.
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
            self.input_pos_buffer.zero_()
            self.input_pos_buffer[bi_tensor] = cur_pos

            linear_caches = [s._linear_cache for s in decode_batch]
            logits = self._decode_step_eager(bi_tensor, last_ids, linear_caches)
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


__all__ = ["FlexQwen3_5Inference", "Sequence"]
