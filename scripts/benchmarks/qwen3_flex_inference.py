"""Qwen3 inference with flex_attention + paged KV cache + CUDA graphs.

The transformers continuous-batching path tops out at ~10% of vLLM on this
workload because `_generation_step` is Python-heavy (scheduler + paged
attention dispatch per layer + per-request metadata updates). torch.compile
chokes on it (700+ recompile storm, see Phase 4).

flex-nano-vllm (Chang, 2024) hits 90% of vLLM in 1000 lines of pure PyTorch
by building paged attention on top of `torch.nn.attention.flex_attention`:

1. The paged KV cache is a single contiguous [1, H, num_pages*page_size, D]
   tensor; logical<->physical mapping lives in a PageTable.
2. flex_attention's BlockMask lets us route queries to physical pages via
   mask_mod + score_mod callbacks, which compile cleanly.
3. One CUDA graph per batch-size bucket captured during warmup; dispatch to
   the nearest bucket on each decode step and pad with batch_idx=0 (reserved
   as a no-op slot).

This file adapts that architecture to Qwen3-4B. The attention forward is
monkey-patched to use our PagedKVCache, and the inference loop runs
prefill + decode on the main thread (no background worker, graph replay
works end-to-end).

LoRA: the bf16 path uses a **double-copy rollout pattern** when
`--lora_adapter` is set. A pristine `base_model` lives on GPU alongside a
deep-copy `inference_model` (wrapped by PEFT). Before each rollout --
or at setup time, here -- the inference copy's LoRA-target base weights
are restored in-place from pristine, then `merge_adapter()` is called
fresh. We never call `unmerge_adapter()`. This avoids the ~1 ULP bf16
drift per merge/unmerge cycle that would otherwise corrupt the base
model across hundreds of GRPO iterations. `--verify_no_drift` hashes the
base params before and after N cycles and asserts bit-identical.

Run:
    CUDA_VISIBLE_DEVICES=6 python scripts/benchmarks/qwen3_flex_inference.py \
        --n_prompts 32 --max_new_tokens 512 --stats_path logs/qwen3_flex.json

Add `--capture_cudagraph` to capture per-batch-size decode graphs during
warmup.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from flex_paged_attention import PagedKVCache, PageTable  # noqa: E402

# Compile flex_attention once at import for warm caches. `fullgraph=True` is
# required for the decode CUDA graph capture to be worth anything.
# Allow an environment override to try `mode="max-autotune"` for the kernel
# template search -- pays off on steady-state decode but adds ~minutes of
# warmup time at first import.
_FLEX_COMPILE_MODE = os.environ.get("FLEX_COMPILE_MODE", None)
if _FLEX_COMPILE_MODE:
    flex_attention_compiled = torch.compile(
        flex_attention,
        fullgraph = True,
        mode = _FLEX_COMPILE_MODE,
    )
else:
    flex_attention_compiled = torch.compile(flex_attention, fullgraph = True)


def _apply_rotary(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim = -1)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def make_flex_qwen3_attention_forward(page_table: PageTable):
    """Return a new `forward` method for `Qwen3Attention` that uses
    flex_attention against a paged KV cache. The returned closure captures
    the shared PageTable; each layer gets its own PagedKVCache attached to
    the module as `self._paged_cache`.

    Expects the caller to have set on each layer:
        self._paged_cache: PagedKVCache
    and to pass the following kwargs through the model forward:
        flex_block_mask: BlockMask
        flex_input_pos:   Tensor [B, S]
        flex_batch_idx:   Tensor [B] (decode) or [1, S] (packed prefill)
        flex_kernel_options: dict | None
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask = None,
        past_key_values = None,
        cache_position = None,
        flex_block_mask: Optional[BlockMask] = None,
        flex_input_pos: Optional[torch.Tensor] = None,
        flex_batch_idx: Optional[torch.Tensor] = None,
        flex_kernel_options: Optional[dict] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _apply_rotary(q, k, cos, sin)

        # Write to paged KV cache. For prefill, assign_prefill_no_paging
        # writes into [1, H, MAX_S, D]; for decode, assign() writes into the
        # B decode slots.
        if self._paged_cache is not None and flex_input_pos is not None:
            k, v = self._paged_cache.update(flex_input_pos, k, v, flex_batch_idx)

        # Flex attention. The block mask routes each query to the correct
        # pages; enable_gqa handles num_kv_heads < num_q_heads.
        attn_output = flex_attention_compiled(
            q,
            k,
            v,
            scale = self.scaling,
            block_mask = flex_block_mask,
            enable_gqa = True,
            kernel_options = flex_kernel_options,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None

    return forward


def patch_qwen3_model(model: torch.nn.Module, page_table: PageTable):
    """Attach a `PagedKVCache` to every `Qwen3Attention` layer and swap in
    the flex_attention forward above.
    """
    fwd = make_flex_qwen3_attention_forward(page_table)
    for layer in model.model.layers:
        attn = layer.self_attn
        attn._paged_cache = PagedKVCache(
            page_table,
            n_heads = model.config.num_key_value_heads,
            head_dim = model.config.head_dim,
            dtype = model.dtype,
        ).to(model.device)
        # Bind as method.
        import types

        attn.forward = types.MethodType(fwd, attn)


# --- model forward helper that passes flex kwargs through ------------------


def call_model_with_flex_kwargs(model, input_ids, position_ids, flex_kwargs):
    """`model(**inputs, **flex_kwargs)` would error because Qwen3ForCausalLM
    doesn't declare the flex_* kwargs. We walk through the model manually
    to pass them into the attention layers (which now accept them)."""
    base = model.model  # Qwen3Model
    inputs_embeds = base.embed_tokens(input_ids)
    position_embeddings = base.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds
    for layer in base.layers:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states, _ = layer.self_attn(
            hidden_states,
            position_embeddings = position_embeddings,
            **flex_kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
    hidden_states = base.norm(hidden_states)
    return hidden_states


# --- inference engine ------------------------------------------------------


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

    def __post_init__(self):
        if self.output_ids is None:
            self.output_ids = []

    @property
    def total_length(self) -> int:
        return self.input_length + len(self.output_ids)


# --- double-copy LoRA rollout helpers -------------------------------------
#
# PEFT's `merge_adapter` / `unmerge_adapter` pair is asymmetric at bf16:
# merge does `W_bf16 += delta_fp32` (the += upcasts then truncates), while
# unmerge does `W_bf16 -= delta_fp32.to(bf16)` -- the delta is rounded to
# bf16 first, so the round-trip leaves ~1 ULP drift on `base_layer.weight`
# every cycle. Across hundreds of GRPO iterations this corrupts the base
# model; the adapter ends up training against a drifting target.
#
# vLLM avoids this by keeping the base weights pristine and materializing a
# second "base + LoRA" copy for inference. We do the same: keep `base_model`
# (pristine) and `inference_model = deepcopy(base_model)`, wrap the copy
# with PEFT, and before each rollout refresh the LoRA-target base weights
# from pristine in-place and call `merge_adapter()` fresh. Never unmerge --
# we always re-materialize, so there is no round-trip error to accumulate.


def refresh_lora_merge_from_pristine(base_model, peft_model):
    """Copy pristine `base_model` weights into `peft_model`'s LoRA-target
    `base_layer.weight`s in-place, reset the PEFT `merged` flag without the
    unmerge arithmetic, then call `peft_model.merge_adapter()` once.

    In-place `weight.data.copy_(pristine)` writes into the same tensor
    storage, so CUDA graphs captured against the merged weights stay valid
    across refreshes (replay reads the captured address; the new value
    takes effect on the next replay without re-capture).

    Returns the number of LoraLayer modules refreshed.
    """
    from peft.tuners.lora.layer import LoraLayer

    n_refreshed = 0
    for name, module in peft_model.base_model.model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        base_submodule = base_model.get_submodule(name)
        module.base_layer.weight.data.copy_(base_submodule.weight.data)
        module.merged_adapters = []
        n_refreshed += 1
    peft_model.merge_adapter()
    return n_refreshed


def _hash_state_dict(model) -> str:
    """sha256 over all parameter bytes in name-sorted order. Uses the
    bit-level `view(torch.uint8)` reinterpretation so bf16 / int / etc. all
    round-trip without any float casting."""
    h = hashlib.sha256()
    sd = model.state_dict()
    for name in sorted(sd.keys()):
        t = sd[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def run_drift_verification(base_model, peft_model, n_iters: int = 10,
                           noise_scale: float = 0.01):
    """Simulate N GRPO iterations: perturb LoRA weights with random noise,
    call `refresh_lora_merge_from_pristine`, repeat. Assert the pristine
    `base_model`'s parameters are bit-identical before and after.

    Also checks inference-copy determinism: after restoring the LoRA state
    to its initial value, the merged `inference_model` state-dict hash
    should match the hash taken right after the first refresh.
    """
    from peft.tuners.lora.layer import LoraLayer

    inference_model = peft_model.base_model.model

    # Snapshot initial LoRA A/B weights so we can restore at the end.
    initial_lora = {}
    for name, module in inference_model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        for adapter_name in list(module.lora_A.keys()):
            initial_lora[(name, "A", adapter_name)] = (
                module.lora_A[adapter_name].weight.data.clone()
            )
            initial_lora[(name, "B", adapter_name)] = (
                module.lora_B[adapter_name].weight.data.clone()
            )

    base_hash_before = _hash_state_dict(base_model)

    # Initial refresh: establishes merged-state baseline for the inference copy.
    refresh_lora_merge_from_pristine(base_model, peft_model)
    inf_hash_initial_merged = _hash_state_dict(inference_model)

    for _ in range(n_iters):
        for name, module in inference_model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            for adapter_name in list(module.lora_A.keys()):
                a = module.lora_A[adapter_name].weight.data
                b = module.lora_B[adapter_name].weight.data
                a.add_(noise_scale * torch.randn_like(a))
                b.add_(noise_scale * torch.randn_like(b))
        refresh_lora_merge_from_pristine(base_model, peft_model)

    base_hash_after = _hash_state_dict(base_model)

    # Restore initial LoRA weights and re-merge; inference hash must match
    # the initial merged-state hash (determinism of the refresh pipeline).
    for (name, kind, adapter_name), w in initial_lora.items():
        module = inference_model.get_submodule(name)
        tgt = module.lora_A if kind == "A" else module.lora_B
        tgt[adapter_name].weight.data.copy_(w)
    refresh_lora_merge_from_pristine(base_model, peft_model)
    inf_hash_restored = _hash_state_dict(inference_model)

    base_ok = base_hash_before == base_hash_after
    inf_ok = inf_hash_initial_merged == inf_hash_restored

    assert base_ok, (
        f"base model drifted across {n_iters} refreshes\n"
        f"  before: {base_hash_before}\n"
        f"  after : {base_hash_after}"
    )
    assert inf_ok, (
        f"inference model did not revert to deterministic merged-state hash\n"
        f"  initial  : {inf_hash_initial_merged}\n"
        f"  restored : {inf_hash_restored}"
    )
    print(f"[verify] base model bit-identical across {n_iters} refreshes")
    print(f"[verify] inference copy deterministic after LoRA restore")
    print(f"[verify] sha256 base   : {base_hash_before}")
    print(f"[verify] sha256 merged : {inf_hash_initial_merged}")
    return {
        "n_iters": n_iters,
        "noise_scale": noise_scale,
        "base_hash_before": base_hash_before,
        "base_hash_after": base_hash_after,
        "base_bit_identical": base_ok,
        "inference_hash_initial_merged": inf_hash_initial_merged,
        "inference_hash_after_restore": inf_hash_restored,
        "inference_deterministic": inf_ok,
    }


# Default kernel_options per phase. Our defaults stay conservative -- the
# non-default FlexKernelOptions (PRESCALE_QK, ROWS_GUARANTEED_SAFE, USE_TMA)
# are opt-in via CLI because some of them break correctness on our
# paged-attention setup.
#
# Specifically, `ROWS_GUARANTEED_SAFE=True` is unsafe here: we reserve
# batch_idx=0 and page_idx=0 as no-op padding slots. When a decode
# padded batch row maps to only-reserved pages, the block mask returns
# False for every kv_idx, so the row has zero unmasked values. The flag
# tells the kernel to skip the row-has-at-least-one-unmasked check, so
# the softmax NaNs silently -- which manifests as "!!!!!!" token spam.
DECODE_KERNEL_OPTIONS_DEFAULT = None
# Prefill keeps FORCE_USE_FLEX_ATTENTION so we don't auto-dispatch into
# the flex-decoding kernel when the packed q_len gets small.
PREFILL_KERNEL_OPTIONS_DEFAULT = {"FORCE_USE_FLEX_ATTENTION": True}


class FlexInference:
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
        fa4_prefill = False,
        base_model = None,
        peft_model = None,
    ):
        assert max_seq_length % page_size == 0
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.eos_token_id = tokenizer.eos_token_id
        # For double-copy LoRA rollout: `base_model` is the pristine copy
        # (never touched); `peft_model` wraps the inference copy (`model`
        # above is `peft_model.base_model.model`). Both may be None when
        # no LoRA adapter is active, or when the 4-bit naive-wrapper path
        # is used.
        self.base_model = base_model
        self.peft_model = peft_model
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.page_size = page_size
        self.max_new_tokens = max_new_tokens
        self.fa4_prefill = fa4_prefill
        # On SM100 (Blackwell), FA4 via flex_attention requires Q block = 256,
        # KV block = 128. See attention-gym `get_flash_block_size`.
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
            # Use the CuTeDSL FA4 kernel on Blackwell. FORCE_USE_FLEX_ATTENTION
            # must be off because the FLASH backend is the flex_attention kernel.
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
        patch_qwen3_model(model, self.page_table)

        # Pre-allocated decode state.
        self.input_pos_buffer = torch.zeros(
            max_batch_size, dtype = torch.int32, device = self.device
        )
        # Full-length logical causal mask (shared across decode batch).
        self.block_mask_logical = self.page_table.create_causal_blockmask(
            B = max_batch_size,
            L = max_seq_length,
        )

        self.cudagraph_captured = False
        self.graphs = {}
        self.graph_vars = {}

    def tokenize(self, sequences):
        for seq in sequences:
            ids = self.tokenizer(seq.text, return_tensors = "pt")["input_ids"].squeeze(0)
            seq.input_ids = ids
            seq.input_length = ids.shape[0]

    def _prefill(self, batch: list[Sequence]) -> torch.Tensor:
        """Packed prefill: concatenate all sequences into [1, L] with a
        document_causal mask. Return logits at the last position of each
        sequence as [num_seqs, V].
        """
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

        # Pad to multiple of Q block size (flex_attention block alignment).
        # For FA4 on Blackwell, Q block = 256 -- otherwise 128.
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
        logits_positions = input_lengths.cumsum(dim = 0) - 1  # [num_seqs]

        # If FA4 is on, BLOCK_SIZE is a (Q, KV) tuple. Otherwise scalar.
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
        position_ids = input_pos  # Qwen3 uses 0-based; unlike Gemma2
        hidden = call_model_with_flex_kwargs(
            self.model, input_ids, position_ids, flex_kwargs
        )
        return self.model.lm_head(hidden[:, logits_positions, :]).squeeze(0)

    def _decode_block_mask(self, batch_idx: torch.Tensor):
        """Slice a single-row BlockMask for every seq in the decode batch,
        then translate logical→physical pages."""
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
        hidden = call_model_with_flex_kwargs(
            self.model, input_ids.view(B, 1), position_ids, flex_kwargs
        )
        return self.model.lm_head(hidden[:, -1, :])  # [B, V]

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
        """Capture one CUDA graph per batch-size bucket.

        Pre-reserves a page for every batch_idx slot so the KV cache writes
        during capture hit valid physical addresses. After capture we erase
        the batches -- the graph replay reads/writes the same physical
        pages regardless of whether the logical batch currently owns them,
        because batch_idx 0 is reserved as a padding slot.
        """
        max_bs = self.max_batch_size
        # Reserve a dummy page for every slot we're going to use during
        # capture. Without this, assign() does k_cache[:, :, -1, :] = ...
        # and we get an illegal memory access.
        reserved_batches = []
        for bi in range(1, max_bs):
            try:
                allocated = self.page_table.allocate()
                self.page_table.reserve(
                    allocated,
                    torch.tensor([allocated], device = self.device, dtype = torch.long),
                    self.page_size,  # just one page
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
            print(f"[flex] capturing CUDA graph for bs={bs}")
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
        # Release the scratch batches; real requests will re-allocate them.
        for bi in reserved_batches:
            self.page_table.erase(bi)
        self.graph_vars = dict(
            input_ids = input_ids, batch_idx = batch_idx, outputs = outputs
        )

    def refresh_inference_from_base(self):
        """Re-materialize the inference copy's merged LoRA weights from the
        pristine `base_model`. Call this once at setup (before CUDA graph
        capture) and, in a real GRPO loop, once after every training step
        that updates the LoRA adapter. Never call `unmerge_adapter()` --
        we always re-merge from pristine, so no drift accumulates.

        No-op when the double-copy pair wasn't configured (e.g. no LoRA,
        or 4-bit naive PEFT-wrapper path).
        """
        if self.base_model is None or self.peft_model is None:
            return 0
        return refresh_lora_merge_from_pristine(self.base_model, self.peft_model)

    @torch.inference_mode()
    def generate(self, sequences: list[Sequence], capture_cudagraph = False):
        self.tokenize(sequences)
        waiting = deque(sequences)
        running = deque()
        done = []

        if capture_cudagraph and not self.cudagraph_captured:
            self.capture_decode_cudagraph()
            self.cudagraph_captured = True

        while waiting or running:
            # 1. Try to schedule new requests into running.
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

            # 2. Reserve pages for running seqs that need more capacity.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default = "unsloth/Qwen3-4B-Base")
    p.add_argument("--n_prompts", type = int, default = 32)
    p.add_argument("--n_rounds", type = int, default = 2)
    p.add_argument("--max_new_tokens", type = int, default = 512)
    p.add_argument("--max_batch_size", type = int, default = 64)
    p.add_argument("--max_seq_length", type = int, default = 2048)
    p.add_argument("--n_pages", type = int, default = 2048)
    p.add_argument("--page_size", type = int, default = 128)
    p.add_argument("--capture_cudagraph", action = "store_true")
    p.add_argument("--lora_adapter", default = None)
    # Kernel tuning (optional JSON-valued CLI args so we can sweep quickly):
    p.add_argument(
        "--decode_kernel_options",
        default = None,
        help = "JSON for FlexKernelOptions applied in decode, "
        'e.g. \'{"PRESCALE_QK":true,"USE_TMA":true}\'.',
    )
    p.add_argument(
        "--prefill_kernel_options", default = None, help = "Same but for prefill."
    )
    # If set, torch.compile the full attention-stack closure in addition to
    # (or instead of) compiling just flex_attention. `reduce-overhead` is
    # the interesting mode; it nests with our CUDA graph capture.
    p.add_argument(
        "--compile_model_forward",
        default = None,
        choices = [None, "default", "reduce-overhead", "max-autotune-no-cudagraphs"],
    )
    p.add_argument(
        "--fa4_prefill",
        action = "store_true",
        help = (
            "Use BLOCK_SIZE=(256,128) + BACKEND=FLASH on prefill to unlock the "
            "CuTeDSL FA4 kernel on Blackwell (SM100)."
        ),
    )
    p.add_argument(
        "--load_in_4bit",
        action = "store_true",
        help = (
            "Load the base model as bitsandbytes 4-bit. When set with "
            "--lora_adapter, the LoRA is kept as a PEFT wrapper (no merge) "
            "because merging into 4-bit weights is not supported."
        ),
    )
    p.add_argument(
        "--no_merge_lora",
        action = "store_true",
        help = (
            "Reference path: keep the LoRA adapter as a PEFT wrapper "
            "instead of merging it. Runs three matmuls per projection; "
            "slow. Useful for the unmerged row in the writeup's comparison "
            "table. The default is now the double-copy pattern, which is "
            "both merge-speed and drift-free."
        ),
    )
    p.add_argument(
        "--verify_no_drift",
        action = "store_true",
        help = (
            "Drift-verification mode. Hash the pristine base model params, "
            "run N perturb+refresh cycles (simulating N GRPO iterations) "
            "on a copy, re-hash, and assert bit-identical. Requires a "
            "--lora_adapter; skips rollout generation."
        ),
    )
    p.add_argument(
        "--verify_iterations",
        type = int,
        default = 10,
        help = "Number of perturb+refresh cycles for --verify_no_drift.",
    )
    p.add_argument(
        "--model_name_4bit",
        default = None,
        help = (
            "Override the 4-bit shard name. Defaults to "
            "`{model_name}-unsloth-bnb-4bit`."
        ),
    )
    p.add_argument("--stats_path", required = True)
    args = p.parse_args()

    def _parse_opts(s):
        if s is None:
            return None
        return json.loads(s)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = None
    peft_model = None

    if args.load_in_4bit:
        # Load the pre-quantized Unsloth 4-bit shard. Compute dtype comes
        # from the packaged config (bf16 for these shards).
        #
        # 4-bit keeps the naive PEFT-wrapper path: bnb's `Linear4bit` holds
        # packed quantised weights, not regular bf16, so the double-copy
        # refresh (in-place copy of `base_layer.weight`) doesn't apply.
        # Materializing a full bf16 inference copy via dequant would wipe
        # out the memory saving of 4-bit.
        bnb_model_name = args.model_name_4bit or f"{args.model_name}-unsloth-bnb-4bit"
        print(f"[flex] loading 4-bit base: {bnb_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            bnb_model_name,
            attn_implementation = "eager",
            device_map = "cuda:0",
        )
        # See note in cb_vs_vllm_generation.py: tie lm_head to embed_tokens
        # for bnb-4bit shards of tied-embedding models.
        if getattr(model.config, "tie_word_embeddings", False):
            model.lm_head.weight = model.model.embed_tokens.weight
        model.eval()

        if args.lora_adapter:
            from peft import PeftModel

            peft_wrapper = PeftModel.from_pretrained(
                model,
                str(Path(args.lora_adapter).resolve()),
                is_trainable = False,
            )
            # LoRA stays as a wrapper around Params4bit; three matmuls per
            # projection. This is the slow reference row in the writeup.
            model = peft_wrapper.base_model.model
    else:
        # bf16 path -- double-copy LoRA rollout.
        #
        # `base_model` stays pristine; we deep-copy it to `inference_model`,
        # wrap the copy with PEFT, and re-materialize the merged LoRA on
        # the copy whenever the LoRA weights change. Memory cost: +~8 GB
        # for Qwen3-4B bf16 (two copies on GPU) -- well within budget vs
        # vLLM's 156 GB.
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype = torch.bfloat16,
            attn_implementation = "eager",
        ).to("cuda")
        base_model.eval()

        if not args.lora_adapter:
            # No adapter -- use base_model directly, no inference copy.
            model = base_model
            base_model = None
        elif args.no_merge_lora:
            # Reference path: PEFT wrapper on the only model copy,
            # adapter unmerged. Three matmuls per projection. Kept for
            # the comparison row in the writeup.
            from peft import PeftModel

            peft_wrapper = PeftModel.from_pretrained(
                base_model,
                str(Path(args.lora_adapter).resolve()),
                is_trainable = False,
            )
            model = peft_wrapper.base_model.model
            base_model = None
        else:
            # Double-copy rollout path.
            from peft import PeftModel

            print("[flex] deep-copying base model for double-copy LoRA rollout")
            inference_model = copy.deepcopy(base_model)
            inference_model.eval()

            peft_model = PeftModel.from_pretrained(
                inference_model,
                str(Path(args.lora_adapter).resolve()),
                is_trainable = False,
            )
            model = peft_model.base_model.model
            model.eval()
            # Do NOT call merge_adapter here -- FlexInference.refresh_
            # inference_from_base() below handles the initial merge so the
            # same code path runs at setup and on every GRPO refresh.

    # Drift-verification mode: skip rollout generation, just hash-check.
    if args.verify_no_drift:
        if args.load_in_4bit:
            raise SystemExit(
                "--verify_no_drift only applies to the bf16 double-copy path "
                "(4-bit keeps the naive PEFT-wrapper path, no merge refresh)."
            )
        if args.no_merge_lora:
            raise SystemExit(
                "--verify_no_drift is incompatible with --no_merge_lora "
                "(nothing is merged; nothing to drift)."
            )
        if base_model is None or peft_model is None:
            raise SystemExit(
                "--verify_no_drift requires --lora_adapter so there is a "
                "LoRA to merge/refresh against the pristine base."
            )
        print(
            f"[flex] running drift verification: {args.verify_iterations} "
            f"perturb+refresh cycles"
        )
        result = run_drift_verification(
            base_model, peft_model, n_iters = args.verify_iterations
        )
        result = {"mode": "verify_no_drift", **result}
        os.makedirs(
            os.path.dirname(os.path.abspath(args.stats_path)) or ".",
            exist_ok = True,
        )
        with open(args.stats_path, "w") as f:
            json.dump(result, f, indent = 2)
        print(json.dumps(result, indent = 2))
        os._exit(0)

    from unsloth_grpo_common import (
        SYSTEM_PROMPT,
        apply_chat_template_to_tokenizer,
    )
    from datasets import load_dataset

    apply_chat_template_to_tokenizer(tok)
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

    # Make sure the base HF model that Qwen3Attention belongs to isn't wrapped
    # by PeftModel anymore (we merged); `.model` should be Qwen3ForCausalLM.
    inference = FlexInference(
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

    # Initial merge from pristine. Done via `refresh_inference_from_base`
    # (not raw `merge_adapter`) so the exact same code path runs at setup
    # and at every GRPO refresh -- the CUDA graph capture below sees the
    # merged weights already in place. In a real GRPO loop, call
    # `inference.refresh_inference_from_base()` after every training step
    # that updates the LoRA adapter. We skip per-round refresh in this
    # benchmark because the LoRA weights don't change between rounds.
    if inference.base_model is not None and inference.peft_model is not None:
        n = inference.refresh_inference_from_base()
        print(f"[flex] double-copy rollout: refreshed {n} LoRA-target layers")

    # Optionally compile the manual forward walker. This fuses the layer-stack
    # ops around flex_attention. Under CUDA graph capture, the compiled
    # function gets captured into the same graph.
    if args.compile_model_forward:
        torch._dynamo.config.cache_size_limit = 256
        print(
            f"[flex] torch.compile(call_model_with_flex_kwargs, "
            f"mode={args.compile_model_forward!r})"
        )
        import sys as _sys

        _this = _sys.modules[__name__]
        _this.call_model_with_flex_kwargs = torch.compile(
            call_model_with_flex_kwargs,
            mode = args.compile_model_forward,
            dynamic = True,
            fullgraph = False,
        )

    def make_seqs():
        return [Sequence(text = t, max_new_tokens = args.max_new_tokens) for t in texts]

    # Warmup.
    torch.cuda.reset_peak_memory_stats()
    print("[flex] warmup (16 prompts)...")
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
            f"[flex] round {r}: {wall_times[-1]:.2f}s, {total_decoded} tokens, "
            f"{total_decoded / wall_times[-1]:.1f} tok/s"
        )

    med = sorted(wall_times)[len(wall_times) // 2]
    best = min(wall_times)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    # Sample a couple of completions so we can eyeball coherence.
    sample_completions = []
    for s in out[:3]:
        sample_completions.append(
            tok.decode(s.output_ids[:80], skip_special_tokens = True)
        )
    res = {
        "backend": "qwen3_flex",
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
