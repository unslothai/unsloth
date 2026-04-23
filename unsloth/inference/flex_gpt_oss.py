# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""gpt-oss inference with flex_attention + paged KV cache + attention sinks.

Sibling of ``flex_moe.py`` (Qwen3-MoE). Three arch-specific pieces:

1. **Attention sinks.** Each layer has a learned per-head ``sinks``
   parameter that biases a virtual sink token in the softmax. We
   implement it the same way ``unsloth_zoo.flex_attention.attention_sink``
   does: call flex_attention with ``return_lse=True``, then scale the
   output by ``sigmoid(lse - sinks[h])``.

2. **Per-layer sliding window.** gpt-oss layer types alternate between
   full attention and sliding-128. The walker passes two ``BlockMask``
   objects (``flex_block_mask`` + ``flex_block_mask_sliding``) and each
   attention forward picks one based on ``self.sliding_window``.

3. **MoE expert stack.** Uses unsloth_zoo's ``GptOssExperts`` forward
   (``forward_native_grouped_mm`` with the ``"GptOssExperts"`` branch
   — interleaved gate/up split + ``gate * sigmoid(gate * 1.702)`` — is
   already there at moe_utils.py:918-972, reused verbatim). The MoE
   LoRA merge reuses ``refresh_moe_lora_merge_from_pristine`` from
   ``flex_moe.py`` — the transposed-orientation branch covers
   gpt-oss's ``(E, H, 2I)`` gate_up_proj layout.

bnb-4bit: ``GptOssExpertsBnb4bit`` uses an ``nn.ModuleList`` per-expert
loop that can't be CUDA-graph-captured. Detect it in ``__init__`` and
set ``capture_cudagraph=False`` — decode still benefits from paged KV
and flex_attention, just not the graph replay.
"""

from __future__ import annotations

import os
import types
from collections import deque
from typing import Optional

import torch
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


# gpt-oss rotary is NOT the Llama ``(q * cos) + (rotate_half(q) * sin)``
# half-dim sin/cos rotation. It does a first/second-half split with
# ``cos`` / ``sin`` sized to ``head_dim / 2``. Import the reference here
# and unsqueeze head-dim once per call — same behaviour as
# ``apply_rotary_pos_emb`` in ``transformers.models.gpt_oss``.
def _gpt_oss_apply_rotary(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # [B, 1, S, D/2]
    sin = sin.unsqueeze(1)

    def rotate(x):
        first, second = torch.chunk(x, 2, dim=-1)
        first_ = first * cos - second * sin
        second_ = second * cos + first * sin
        return torch.cat((first_, second_), dim=-1)

    return rotate(q), rotate(k)


# ---------------------------------------------------------------------------
# Attention forward: paged KV + flex_attention + sinks + per-layer sliding.
# ---------------------------------------------------------------------------


def make_gptoss_attention_forward(page_table: PageTable):
    """Return a ``forward`` method for ``GptOssAttention``.

    Differences from the dense Qwen3/Llama flex forward:
    - No ``q_norm`` / ``k_norm`` (gpt-oss has neither).
    - Uses ``flex_attention(..., return_lse=True)`` and scales output by
      ``sigmoid(lse - sinks[h])`` for the sink token.
    - Picks ``flex_block_mask_sliding`` when the layer has
      ``self.sliding_window is not None``.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
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

        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _gpt_oss_apply_rotary(q, k, cos, sin)

        # Paged-KV write. Match the pre-allocated cache dtype; attention
        # linears may compute in fp32 under autocast.
        if self._paged_cache is not None and flex_input_pos is not None:
            cache_dtype = self._paged_cache.k_cache.dtype
            if k.dtype != cache_dtype:
                k = k.to(cache_dtype)
            if v.dtype != cache_dtype:
                v = v.to(cache_dtype)
            k, v = self._paged_cache.update(flex_input_pos, k, v, flex_batch_idx)

        # Per-layer block mask dispatch. sliding_window is None on full
        # attention layers, an int on sliding layers.
        if self.sliding_window is not None and flex_block_mask_sliding is not None:
            block_mask = flex_block_mask_sliding
        else:
            block_mask = flex_block_mask

        attn_output, logsumexp = flex_attention_compiled(
            q,
            k,
            v,
            scale=self.scaling,
            block_mask=block_mask,
            enable_gqa=True,
            kernel_options=flex_kernel_options,
            return_lse=True,
        )

        # Attention sink. Equivalent to concatenating a sink column
        # ``sinks[h]`` to the attention logits before softmax. With
        # ``return_lse=True`` we have ``lse = log sum_k exp(QK[q, k] *
        # scale)``; the sink-aware softmax scales the output by
        # ``exp(lse) / (exp(lse) + exp(sinks[h])) = sigmoid(lse -
        # sinks[h])``. Mirrors ``flex_attention_add_sinks`` in
        # ``unsloth_zoo/flex_attention/attention_sink.py``.
        logsumexp = logsumexp - self.sinks.view(1, -1, 1)
        sink_scale = torch.sigmoid(logsumexp)
        attn_output = attn_output * sink_scale.unsqueeze(-1).to(attn_output.dtype)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None

    return forward


def patch_gptoss_attention_forwards(model: torch.nn.Module, page_table: PageTable):
    """Attach a ``PagedKVCache`` + replace ``forward`` on every
    ``GptOssAttention`` layer."""
    fwd = make_gptoss_attention_forward(page_table)
    for layer in model.model.layers:
        attn = layer.self_attn
        attn._paged_cache = PagedKVCache(
            page_table,
            n_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            dtype=model.dtype,
        ).to(model.device)
        attn.forward = types.MethodType(fwd, attn)


# ---------------------------------------------------------------------------
# Walker: pass flex kwargs through each layer, unpack MLP tuple return.
# ---------------------------------------------------------------------------


def call_gpt_oss_model_with_flex_kwargs(model, input_ids, position_ids, flex_kwargs):
    """Walk a ``GptOssModel`` manually, injecting flex kwargs into each
    attention call. ``GptOssMLP.forward`` returns
    ``(hidden_states, router_scores)``; we discard router scores at
    inference (no load-balance loss)."""
    base = model.model
    inputs_embeds = base.embed_tokens(input_ids)
    position_embeddings = base.rotary_emb(inputs_embeds, position_ids)
    _cos, _sin = position_embeddings
    if _cos.dim() == 2:
        _cos = _cos[position_ids]
        _sin = _sin[position_ids]
        position_embeddings = (_cos, _sin)
    hidden_states = inputs_embeds
    compute_dtype = inputs_embeds.dtype
    for layer in base.layers:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states).to(compute_dtype)
        hidden_states, _ = layer.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            **flex_kwargs,
        )
        hidden_states = residual + hidden_states.to(compute_dtype)
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states).to(compute_dtype)
        mlp_out = layer.mlp(hidden_states)
        if isinstance(mlp_out, tuple):
            hidden_states = mlp_out[0]
        else:
            hidden_states = mlp_out
        hidden_states = residual + hidden_states.to(compute_dtype)
    hidden_states = base.norm(hidden_states)
    return hidden_states


# ---------------------------------------------------------------------------
# Sliding-window block-mask builders. Paged-KV aware.
# ---------------------------------------------------------------------------


def _create_sliding_causal_blockmask(page_table: PageTable, B: int, L: int, W: int):
    """Full-window causal block mask: ``q >= kv`` and ``q - kv < W``.
    Built against the logical block layout; ``convert_logical_block_mask``
    wraps it to paged indices later for decode."""

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
    """Document-causal + sliding: ``q >= kv``, ``q - kv < W``, and
    ``docs[q] == docs[kv]``. Mirrors
    ``create_prefill_blockmask_no_paging`` with the window term added."""
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
# FlexGptOssInference
# ---------------------------------------------------------------------------


class FlexGptOssInference:
    """gpt-oss inference engine. API-compatible with ``FlexMoEInference``.

    Phase 1: eager decode only (no CUDA graph capture). Phase 3 adds the
    capture path for the bf16 variant; bnb-4bit stays eager because
    ``GptOssExpertsBnb4bit`` uses an nn.ModuleList per-expert loop that
    isn't graph-capturable.
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
            "FlexGptOssInference expects a HF CausalLM shape (.model.layers)."
        )
        for i, layer in enumerate(model.model.layers):
            assert hasattr(layer.self_attn, "sinks"), (
                f"Layer {i}.self_attn has no sinks — not a gpt-oss attention?"
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

        # Detect per-layer sliding window. gpt-oss alternates full /
        # sliding-128; record whichever window is used so block-mask
        # builders pick it up. If no layer has a sliding window, skip
        # the sliding mask construction entirely.
        sliding_windows = {
            int(layer.self_attn.sliding_window)
            for layer in model.model.layers
            if layer.self_attn.sliding_window is not None
        }
        if len(sliding_windows) > 1:
            raise NotImplementedError(
                f"Mixed sliding-window sizes not supported: {sliding_windows}"
            )
        self.sliding_window = next(iter(sliding_windows), None)

        # Detect bnb-4bit experts. The bnb variant uses an nn.ModuleList
        # per-expert loop — disable CUDA graph capture if present.
        self._has_bnb_experts = any(
            type(layer.mlp.experts).__name__ == "GptOssExpertsBnb4bit"
            for layer in model.model.layers
        )

        # FA4 kernel branch (same as dense/MoE).
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
            patch_gptoss_attention_forwards(model, self.page_table)

        self.input_pos_buffer = torch.zeros(
            max_batch_size, dtype=torch.int32, device=self.device
        )
        self.block_mask_logical = self.page_table.create_causal_blockmask(
            B=max_batch_size,
            L=max_seq_length,
        )
        # Sliding-window logical block mask (only if any layer uses it).
        if self.sliding_window is not None:
            self.block_mask_logical_sliding = _create_sliding_causal_blockmask(
                self.page_table,
                B=max_batch_size,
                L=max_seq_length,
                W=self.sliding_window,
            )
        else:
            self.block_mask_logical_sliding = None

        # Cudagraph capture disabled for Phase 1. Phase 3 wires this up
        # for bf16; bnb-4bit stays False either way.
        self.cudagraph_captured = False
        self.graphs = {}
        self.graph_vars = {}

        # Optional torch.compile walker (Phase 5 turns this on).
        if compile_walker is None:
            compile_walker = os.environ.get("UNSLOTH_FLEX_COMPILE_WALKER", "") == "1"
        self._moe_walker = call_gpt_oss_model_with_flex_kwargs
        if compile_walker:
            try:
                self._moe_walker = torch.compile(
                    call_gpt_oss_model_with_flex_kwargs,
                    fullgraph=False,
                    dynamic=False,
                )
                print(
                    "[flex-gptoss] wrapped call_gpt_oss_model_with_flex_kwargs "
                    "with torch.compile(fullgraph=False, dynamic=False)"
                )
            except Exception as e:
                print(f"[flex-gptoss] torch.compile wrap failed: {e}")
                self._moe_walker = call_gpt_oss_model_with_flex_kwargs

        if self._has_bnb_experts:
            print(
                "[flex-gptoss] bnb-4bit experts detected; CUDA graph capture "
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
        """Capture one CUDA graph per bs bucket. bf16 experts only;
        bnb-4bit (nn.ModuleList per-expert loop) can't be captured and
        ``__init__`` already set ``capture_cudagraph`` to False for that
        variant — this method is only reached when the engine-level
        flag is still True."""
        if self._has_bnb_experts:
            print(
                "[flex-gptoss] bnb-4bit experts: skipping cudagraph capture; "
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
                f"[flex-gptoss] MoE CUDA graph capture requires the "
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
                    f"[flex-gptoss] invalid UNSLOTH_FLEX_GRAPH_BS={_env_bs!r}; "
                    f"using default bucket ladder"
                )
                self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        else:
            self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

        pool = None
        for bs in reversed(self.graph_bs):
            if bs > max_bs:
                continue
            print(f"[flex-gptoss] capturing CUDA graph for bs={bs}")
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
        """Refresh merged LoRA weights from pristine. Reuses the dense
        refresh for q/k/v/o/router and the MoE refresh
        (``refresh_moe_lora_merge_from_pristine``) for ``gate_up_proj``
        / ``down_proj`` in the ``(E, H, 2I)`` transposed orientation."""
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
        """Decode loop. Phase 1: always eager."""
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
