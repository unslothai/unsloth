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
    for layer in base.layers:
        # Attention block — identical to dense Qwen3 / Llama.
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states, _ = layer.self_attn(
            hidden_states,
            position_embeddings = position_embeddings,
            **flex_kwargs,
        )
        hidden_states = residual + hidden_states
        # MoE MLP.
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(hidden_states)
        if isinstance(mlp_out, tuple):
            hidden_states = mlp_out[0]
        else:
            hidden_states = mlp_out
        hidden_states = residual + hidden_states
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
    ):
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

        # MoE decode is eager-only for this first cut. Set the captured
        # flag to False permanently so ``generate(capture_cudagraph=True)``
        # still runs the eager fallback.
        self.cudagraph_captured = False
        self.graphs = {}
        self.graph_vars = {}

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
        hidden = call_moe_model_with_flex_kwargs(
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
        hidden = call_moe_model_with_flex_kwargs(
            self.model, input_ids.view(B, 1), position_ids, flex_kwargs
        )
        return self.model.lm_head(hidden[:, -1, :])

    def _decode_step(
        self, batch_idx: torch.Tensor, input_ids: torch.Tensor, input_pos: torch.Tensor
    ):
        # MoE path is always eager — no CUDA graph replay. See capture
        # docstring below.
        self.input_pos_buffer.zero_()
        self.input_pos_buffer[batch_idx] = input_pos
        return self._decode_step_eager(batch_idx, input_ids)

    def capture_decode_cudagraph(self):
        """Not supported for MoE. ``Qwen3MoeExperts.forward`` uses
        ``torch.where`` + a data-dependent Python for-loop over experts
        (shapes depend on routing), which cannot be captured. Raising
        here so a stray ``capture_cudagraph=True`` fails loudly.

        Future: a padded-fixed-shape dispatch can be gated behind
        ``UNSLOTH_MOE_STATIC_DISPATCH=1`` to make capture viable — out
        of scope for the first cut.
        """
        raise NotImplementedError(
            "FlexMoEInference does not support CUDA graph capture: MoE "
            "expert routing has data-dependent shapes. Run with "
            "capture_cudagraph=False."
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
        """Decode loop. ``capture_cudagraph`` is ignored for MoE —
        always runs eager."""
        self.tokenize(sequences)
        waiting = deque(sequences)
        running = deque()
        done = []

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
