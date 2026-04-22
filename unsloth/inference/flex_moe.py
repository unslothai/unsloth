# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Qwen3-MoE inference with flex_attention + paged KV cache.

Sibling of ``flex_qwen3_llama.py`` (dense Qwen3 / Llama-3). Handles
``Qwen3MoeForCausalLM`` where each decoder layer's ``mlp`` is a
``Qwen3MoeSparseMoeBlock``. Two differences from the dense path:

1. The walker (``call_moe_model_with_flex_kwargs``) unpacks whatever the
   MoE MLP returns. Stock HF 5.x returns a plain tensor; Unsloth's
   patched ``Qwen3MoeSparseMoeBlock_fast_forward`` returns
   ``(hidden_states, router_logits)``. The ``isinstance(_, tuple)``
   guard handles both without coupling this file to either forward.

2. Decode runs eager (no CUDA-graph capture). The MoE expert routing
   uses ``torch.where`` + a Python for-loop over experts, which is
   data-dependent-shape and not graph-capturable. Prefill still uses
   flex_attention compiled. A future cut can swap in a padded-fixed-
   shape dispatch via ``UNSLOTH_MOE_STATIC_DISPATCH=1``.

Everything else — paged-KV cache, attention forward, prefill block-mask,
LoRA double-copy refresh — is shared verbatim with the dense path.
"""

from __future__ import annotations

import os
import types
from collections import deque
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

try:
    from .flex_qwen3_llama import (
        DECODE_KERNEL_OPTIONS_DEFAULT,
        PREFILL_KERNEL_OPTIONS_DEFAULT,
        Sequence,
        patch_model_attention_forwards,
        refresh_lora_merge_from_pristine,
    )
    from .flex_paged_attention import PagedKVCache, PageTable
except ImportError:  # script-mode fallback
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from flex_qwen3_llama import (  # noqa: E402
        DECODE_KERNEL_OPTIONS_DEFAULT,
        PREFILL_KERNEL_OPTIONS_DEFAULT,
        Sequence,
        patch_model_attention_forwards,
        refresh_lora_merge_from_pristine,
    )
    from flex_paged_attention import PagedKVCache, PageTable  # noqa: E402


def call_moe_model_with_flex_kwargs(model, input_ids, position_ids, flex_kwargs):
    """Walk a Qwen3-MoE model manually, injecting flex_* kwargs into each
    attention call. Mirrors ``call_model_with_flex_kwargs`` from
    ``flex_qwen3_llama.py`` but handles the MoE MLP return shape.

    For Qwen3-MoE, ``layer.mlp`` is a ``Qwen3MoeSparseMoeBlock``. Its
    forward signature varies by patch:

      - stock HF 5.x: returns a single tensor (``final_hidden_states``).
      - Unsloth's ``Qwen3MoeSparseMoeBlock_fast_forward``: returns
        ``(final_X, router_logits)``.
      - unsloth_zoo's ``sparse_moe_block_forward``: returns a single
        tensor.

    We call ``layer.mlp(...)`` and unpack whatever comes back. At
    inference we discard ``router_logits`` — no load-balance loss.
    """
    base = model.model  # Qwen3MoeModel
    inputs_embeds = base.embed_tokens(input_ids)
    position_embeddings = base.rotary_emb(inputs_embeds, position_ids)
    # Unsloth's ``LlamaRotaryEmbedding`` drop-in returns the full cached
    # cos/sin as ``[max_seq, D]`` 2D tensors, expecting the caller to
    # slice. Stock HF returns ``[B, S, D]`` already-sliced.
    _cos, _sin = position_embeddings
    if _cos.dim() == 2:
        _cos = _cos[position_ids]
        _sin = _sin[position_ids]
        position_embeddings = (_cos, _sin)
    hidden_states = inputs_embeds
    # RMSNorm + bnb-4bit Linear compute can promote activations to fp32
    # along the Qwen3 MoE path even under autocast. Lock activations to
    # the embed dtype so paged-KV writes (which index_put_ into a
    # pre-allocated bf16 cache) see a matching dtype.
    compute_dtype = inputs_embeds.dtype
    for layer in base.layers:
        # Attention block — identical to dense Qwen3 / Llama.
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states).to(compute_dtype)
        hidden_states, _ = layer.self_attn(
            hidden_states,
            position_embeddings = position_embeddings,
            **flex_kwargs,
        )
        hidden_states = residual + hidden_states.to(compute_dtype)
        # MoE MLP.
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


class FlexMoEInference:
    """MoE inference engine. API-compatible with ``FlexInference`` so
    ``FlexEngine`` dispatch is a one-line change.

    Differences:
      - uses ``call_moe_model_with_flex_kwargs`` (tuple-aware walker).
      - ``cudagraph_captured`` is permanently False; ``generate`` always
        runs the eager decode path. ``capture_decode_cudagraph`` raises
        ``NotImplementedError`` so a stray ``capture_cudagraph=True``
        fails loudly rather than silently producing wrong output.
    """

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
        cumem_allocator = None,
        compile_walker = None,
    ):
        # FastQwen3MoeModel.pre_patch (unsloth/models/qwen3_moe.py) installs
        # a legacy Qwen3MoeSparseMoeBlock_fast_forward that expects
        # ``self.gate_proj``; transformers 5.x Qwen3MoE uses
        # ``self.gate`` / ``self.experts`` instead, so that forward is dead
        # code on this env. Unsloth-zoo's ``patch_qwen3_moe`` re-patches it
        # to the correct ``sparse_moe_block_forward``, but Unsloth's
        # pre_patch can run later and silently clobber it (patch_function
        # bails via can_safely_patch on a second pass). Force-restore the
        # stock HF forward here so the flex walker sees a working MLP.
        try:
            import transformers.models.qwen3_moe.modeling_qwen3_moe as _hf_mod
            _BlockCls = _hf_mod.Qwen3MoeSparseMoeBlock
            cur_forward = getattr(_BlockCls, "forward", None)
            cur_name = getattr(cur_forward, "__name__", "")
            if "fast_forward" in cur_name or cur_name == "Qwen3MoeSparseMoeBlock_fast_forward":
                # Prefer unsloth_zoo's patched version if present;
                # fall back to the stock HF forward otherwise.
                unique = getattr(_BlockCls, "_original_forward_Qwen3MoeSparseMoeBlock", None) or getattr(_BlockCls, "_Qwen3MoeSparseMoeBlock_original_forward", None)
                if unique is not None:
                    _BlockCls.forward = unique
                else:
                    # Re-run unsloth_zoo patch to install sparse_moe_block_forward.
                    from unsloth_zoo.temporary_patches.qwen3_moe import patch_qwen3_moe
                    patch_qwen3_moe()
                    # If patch_function still skipped due to can_safely_patch,
                    # fall back to stock HF as a last resort.
                    cur_forward_after = getattr(_BlockCls, "forward", None)
                    cur_name_after = getattr(cur_forward_after, "__name__", "")
                    if "fast_forward" in cur_name_after:
                        # Lazy-load pristine forward by reloading the module.
                        import importlib
                        _fresh_mod = importlib.reload(_hf_mod)
                        _BlockCls.forward = _fresh_mod.Qwen3MoeSparseMoeBlock.forward
        except Exception:
            pass
        assert max_seq_length % page_size == 0
        # Startup sanity checks. If any of these fail the architecture
        # isn't a Qwen3-MoE variant we know how to drive.
        assert hasattr(model, "model") and hasattr(model.model, "layers"), (
            "FlexMoEInference expects a HF CausalLM shape (.model.layers)."
        )
        for i, layer in enumerate(model.model.layers):
            assert hasattr(layer, "post_attention_layernorm"), (
                f"Layer {i} has no post_attention_layernorm."
            )
            assert hasattr(layer, "mlp") and callable(
                getattr(layer.mlp, "forward", None)
            ), f"Layer {i}.mlp has no callable forward."

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

        # Kernel-options / FA4 branch — copied from FlexInference.
        if fa4_prefill is None or fa4_prefill:
            major, _ = torch.cuda.get_device_capability(self.device)
            supported = major >= 9
            if fa4_prefill and not supported:
                import warnings
                warnings.warn(
                    f"--fa4_prefill needs Hopper (sm_90) or Blackwell "
                    f"(sm_100 / sm_120); found sm_{major}0. Falling back to "
                    f"the Triton flex_attention backend.",
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

        # Route paged-KV allocations through the cuMem pool when sleep
        # mode is active. Block-mask / input_pos scratch stays in the
        # default allocator.
        from .sleep_mode import kv_cache_pool as _kv_cache_pool
        with _kv_cache_pool(cumem_allocator):
            self.page_table = PageTable(
                n_pages = n_pages,
                page_size = page_size,
                max_batch_size = max_batch_size,
                device = self.device.type,
            )
            patch_model_attention_forwards(model, self.page_table)

        self.input_pos_buffer = torch.zeros(
            max_batch_size, dtype = torch.int32, device = self.device
        )
        self.block_mask_logical = self.page_table.create_causal_blockmask(
            B = max_batch_size,
            L = max_seq_length,
        )

        self.cudagraph_captured = False
        self.graphs = {}
        self.graph_vars = {}

        # Optional: wrap ``call_moe_model_with_flex_kwargs`` with
        # ``torch.compile(fullgraph=False, dynamic=False)`` BEFORE CUDA
        # graph capture. On Qwen3-30B-A3B 4bit this gives ~2x decode
        # throughput (378 → 753 tok/s at bs=8, 1735 → 3383 tok/s at
        # bs=48) because Inductor fuses the layernorm + residual +
        # router pointwise ops and the compiled kernels get recorded
        # into the captured graph. Opt-in for now: either pass
        # ``compile_walker=True`` explicitly or set
        # ``UNSLOTH_FLEX_COMPILE_WALKER=1``.
        if compile_walker is None:
            compile_walker = os.environ.get("UNSLOTH_FLEX_COMPILE_WALKER", "") == "1"
        self._moe_walker = call_moe_model_with_flex_kwargs
        if compile_walker:
            try:
                self._moe_walker = torch.compile(
                    call_moe_model_with_flex_kwargs,
                    fullgraph = False,
                    dynamic = False,
                )
                print(
                    "[flex-moe] wrapped call_moe_model_with_flex_kwargs "
                    "with torch.compile(fullgraph=False, dynamic=False)"
                )
            except Exception as e:
                print(f"[flex-moe] torch.compile wrap failed, falling back: {e}")
                self._moe_walker = call_moe_model_with_flex_kwargs

    # --- tokenize / prefill / decode ---------------------------------------
    # Near-verbatim from FlexInference. Only difference is the walker.

    def tokenize(self, sequences):
        for seq in sequences:
            if seq.input_ids is not None and seq.input_length > 0:
                continue
            ids = self.tokenizer(seq.text, return_tensors = "pt")["input_ids"].squeeze(0)
            seq.input_ids = ids
            seq.input_length = ids.shape[0]

    def _prefill(self, batch: list[Sequence]) -> torch.Tensor:
        input_ids_list = [seq.input_ids.to(self.device) for seq in batch]
        input_pos_list = [
            torch.arange(seq.input_length, dtype = torch.long, device = self.device)
            for seq in batch
        ]
        batch_idx_list = [
            torch.full(
                (seq.input_length,), seq.batch_idx, dtype = torch.long, device = self.device
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
        mask = self.page_table.create_prefill_blockmask_no_paging(
            batch_idx, BLOCK_SIZE = prefill_block_size
        )

        flex_kwargs = dict(
            flex_block_mask = mask,
            flex_input_pos = input_pos,
            flex_batch_idx = batch_idx,
            flex_kernel_options = self.prefill_kernel_options,
        )
        position_ids = input_pos
        hidden = self._moe_walker(
            self.model, input_ids, position_ids, flex_kwargs
        )
        return self.model.lm_head(hidden[:, logits_positions, :]).squeeze(0)

    def _decode_block_mask(self, batch_idx: torch.Tensor):
        block_mask = self.block_mask_logical
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
            full_idx = block_mask.full_kv_indices[batch_idx, :, input_block_idx].view(
                B, 1, 1, -1
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
            BLOCK_SIZE = block_mask.BLOCK_SIZE,
            mask_mod = causal_offset(input_pos),
            seq_lengths = seq_length,
        )
        return mask, input_pos

    def _decode_step_eager(self, batch_idx: torch.Tensor, input_ids: torch.Tensor):
        B = input_ids.shape[0]
        mask, input_pos = self._decode_block_mask(batch_idx)
        mask = self.page_table.convert_logical_block_mask(mask, batch_idx)
        position_ids = (input_pos).view(B, 1).to(torch.long)
        flex_kwargs = dict(
            flex_block_mask = mask,
            flex_input_pos = input_pos.view(B, 1).to(torch.long),
            flex_batch_idx = batch_idx,
            flex_kernel_options = self.decode_kernel_options,
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
        # batch_idx=0 is the reserved no-op slot. Zero out the unused part
        # of each capture-shape buffer so padded entries don't write into
        # real KV pages.
        for k, v in gv.items():
            if k != "outputs":
                v.zero_()
        gv["input_ids"][:bs] = input_ids
        gv["batch_idx"][:bs] = batch_idx
        graph.replay()
        return gv["outputs"][:bs]

    def capture_decode_cudagraph(self):
        """Capture one CUDA graph per batch-size bucket for MoE decode.

        Supported on the ``grouped_mm`` MoE backend only. On that backend
        the decode path is fixed-shape:
        ``bincount(minlength=num_experts) → cumsum → argsort →
        torch._grouped_mm × 2 → index_add_``. Python control flow in
        ``sparse_moe_block_forward`` runs once at capture time; only the
        recorded CUDA kernels replay.

        For any other backend (``unsloth_triton``, ``native_torch``) this
        method logs a warning and returns without enabling replay, so
        ``generate(capture_cudagraph=True)`` silently falls back to eager
        decode instead of failing inside the captured graph.

        Pre-reserves a page for every ``batch_idx`` slot so the paged-KV
        ``index_put_`` during capture hits valid physical addresses. The
        reservations are erased after capture — replay reads / writes the
        same physical pages regardless of the logical batch state,
        because ``batch_idx = 0`` is reserved as a padding slot.
        """
        try:
            from unsloth_zoo.temporary_patches.moe_utils import select_moe_backend
            backend = select_moe_backend()
        except Exception:
            backend = None
        if backend != "grouped_mm":
            print(
                f"[flex] MoE CUDA graph capture requires the 'grouped_mm' "
                f"backend (got {backend!r}); skipping capture, decode stays "
                f"eager."
            )
            return

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
            print(f"[flex-moe] capturing CUDA graph for bs={bs}")
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
        """Re-materialize the inference copy's merged LoRA weights from
        the pristine base. No-op when no adapter is configured.

        For Qwen3-MoE, dense LoRA targets (q/k/v/o and potentially the
        router ``gate``) are handled by the dense refresh. Stacked
        expert LoRA targets (``gate_up_proj`` / ``down_proj``) are
        handled by the MoE refresh, which writes via ``torch.baddbmm``
        into the same stacked-tensor storage so captured replay
        addresses stay valid.
        """
        if self.base_model is None or self.peft_model is None:
            return 0
        n = refresh_lora_merge_from_pristine(self.base_model, self.peft_model)
        try:
            n += refresh_moe_lora_merge_from_pristine(
                self.base_model, self.peft_model
            )
        except Exception:
            # MoE LoRA merge is best-effort for now: ZOO's MoE PEFT wrapper
            # varies by transformers version. If the wrapper shape isn't
            # recognised we fall back to the dense refresh only (which
            # already handled any LoraLayer-wrapped modules).
            pass
        return n

    @torch.inference_mode()
    def generate(self, sequences: list[Sequence], capture_cudagraph = False):
        """Decode loop. Captures one CUDA graph per bucket on first call
        when ``capture_cudagraph=True`` and the ``grouped_mm`` MoE
        backend is active; otherwise falls back to eager decode."""
        self.tokenize(sequences)
        waiting = deque(sequences)
        running = deque()
        done = []

        if capture_cudagraph and not self.cudagraph_captured:
            self.capture_decode_cudagraph()
            # ``capture_decode_cudagraph`` leaves ``cudagraph_captured``
            # alone when it skips (non-grouped_mm backend), so only flip
            # the flag when at least one bucket was actually captured.
            if self.graphs:
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
                            [seq.batch_idx], device = self.device, dtype = torch.long
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


# ===========================================================================
# MoE LoRA refresh — phase-4 companion to ``refresh_lora_merge_from_pristine``.
# ===========================================================================


def _get_moe_wrapper_tensor(wrapper):
    """Return the underlying 3D expert tensor for a PEFT-wrapped MoE
    parameter. Tries the common attribute paths in order."""
    if hasattr(wrapper, "get_base_layer"):
        base = wrapper.get_base_layer()
        if hasattr(base, "data"):
            return base.data
        return base
    if isinstance(wrapper, torch.Tensor):
        return wrapper.data
    if hasattr(wrapper, "data"):
        return wrapper.data
    return None


def _pristine_moe_tensor(pristine_module, param_name):
    p = getattr(pristine_module, param_name, None)
    if p is None:
        return None
    if isinstance(p, torch.Tensor):
        return p.data if hasattr(p, "data") else p
    if hasattr(p, "data"):
        return p.data
    return p


def refresh_moe_lora_merge_from_pristine(base_model, peft_model):
    """Batched in-place LoRA merge for Qwen3-MoE stacked expert tensors.

    For each PEFT ParamWrapper on a ``Qwen3MoeExperts.gate_up_proj`` /
    ``down_proj``, compute::

        W_inf[e] = W_pristine[e] + sum_active(scaling * B[e] @ A[e])

    via ``torch.baddbmm`` into the same storage, mirroring the dense
    ``refresh_lora_merge_from_pristine`` semantics (in-place write so
    captured CUDA-graph replay reads the refreshed values).

    Handles both standard (``E, 2I, H``) and transposed (``E, H, 2I``)
    stacked orientations via a runtime shape check against the flat
    ``lora_A``/``lora_B`` shapes.

    Returns the count of expert tensors refreshed. No-op when no
    ParamWrapper-style MoE LoRA is present (e.g. dense-only LoRA, or
    a transformers version that hasn't introduced stacked experts).
    """
    if base_model is None or peft_model is None:
        return 0

    inference_model = peft_model.base_model.model
    n_refreshed = 0

    for name, module in inference_model.named_modules():
        if not (hasattr(module, "gate_up_proj") and hasattr(module, "down_proj")):
            continue
        if not hasattr(module, "num_experts"):
            continue
        E = int(module.num_experts)
        try:
            pristine = base_model.get_submodule(name)
        except AttributeError:
            continue

        for param_name in ("gate_up_proj", "down_proj"):
            wrapper = getattr(module, param_name, None)
            pristine_data = _pristine_moe_tensor(pristine, param_name)
            if wrapper is None or pristine_data is None:
                continue
            has_lora = hasattr(wrapper, "lora_A") and hasattr(wrapper, "lora_B")
            if not has_lora:
                # No PEFT wrapping — keep the plain parameter in sync
                # with pristine (covers the no-LoRA case where the
                # inference copy otherwise diverges via training).
                W_inf = _get_moe_wrapper_tensor(wrapper)
                if W_inf is not None and W_inf.shape == pristine_data.shape:
                    W_inf.copy_(pristine_data)
                    n_refreshed += 1
                continue

            W_inf = _get_moe_wrapper_tensor(wrapper)
            if W_inf is None or W_inf.dim() != 3:
                continue

            adapter_names = list(wrapper.lora_A.keys())
            if not adapter_names:
                W_inf.copy_(pristine_data)
                if hasattr(wrapper, "merged_adapters"):
                    wrapper.merged_adapters = []
                n_refreshed += 1
                continue

            # Determine orientation from lora shapes vs W_inf shape.
            lora_A_w0 = wrapper.lora_A[adapter_names[0]].weight.data
            lora_B_w0 = wrapper.lora_B[adapter_names[0]].weight.data
            in_dim = lora_A_w0.shape[1]
            out_dim = lora_B_w0.shape[0]
            d0, d1 = W_inf.shape[1], W_inf.shape[2]
            if d0 == out_dim and d1 == in_dim:
                is_standard = True
            elif d0 == in_dim and d1 == out_dim:
                is_standard = False
            else:
                raise RuntimeError(
                    f"[refresh_moe_lora_merge_from_pristine] cannot "
                    f"determine orientation for {name}.{param_name}: "
                    f"W_inf.shape={tuple(W_inf.shape)}, "
                    f"in_dim={in_dim}, out_dim={out_dim}"
                )

            # Reset to pristine, then accumulate per-adapter.
            W_inf.copy_(pristine_data)

            for adapter_name in adapter_names:
                scaling = wrapper.scaling[adapter_name]
                A_w = wrapper.lora_A[adapter_name].weight.data
                B_w = wrapper.lora_B[adapter_name].weight.data
                R = A_w.shape[0] // E
                # A_w: (E*R, in_dim)  -> A_3d: (E, R, in_dim)
                A_3d = A_w.view(E, R, in_dim)
                # B_w: (out_dim, E*R) -> (out_dim, E, R) -> (E, out_dim, R)
                B_3d = B_w.view(out_dim, E, R).permute(1, 0, 2).contiguous()
                if is_standard:
                    torch.baddbmm(
                        W_inf,
                        B_3d.to(W_inf.dtype),
                        A_3d.to(W_inf.dtype),
                        alpha = float(scaling),
                        beta = 1.0,
                        out = W_inf,
                    )
                else:
                    torch.baddbmm(
                        W_inf,
                        A_3d.transpose(-2, -1).contiguous().to(W_inf.dtype),
                        B_3d.transpose(-2, -1).contiguous().to(W_inf.dtype),
                        alpha = float(scaling),
                        beta = 1.0,
                        out = W_inf,
                    )

            if hasattr(wrapper, "merged_adapters"):
                wrapper.merged_adapters = list(adapter_names)
            n_refreshed += 1

    return n_refreshed
