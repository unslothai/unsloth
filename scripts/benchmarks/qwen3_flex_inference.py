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

Run:
    CUDA_VISIBLE_DEVICES=6 python scripts/benchmarks/qwen3_flex_inference.py \
        --n_prompts 32 --max_new_tokens 512 --stats_path logs/qwen3_flex.json

Add `--capture_cudagraph` to capture per-batch-size decode graphs during
warmup.
"""

from __future__ import annotations

import argparse
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
    ):
        assert max_seq_length % page_size == 0
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.eos_token_id = tokenizer.eos_token_id
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.page_size = page_size
        self.max_new_tokens = max_new_tokens
        self.decode_kernel_options = (
            decode_kernel_options
            if decode_kernel_options is not None
            else DECODE_KERNEL_OPTIONS_DEFAULT
        )
        self.prefill_kernel_options = (
            prefill_kernel_options
            if prefill_kernel_options is not None
            else PREFILL_KERNEL_OPTIONS_DEFAULT
        )

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

        # Pad to multiple of 128 (flex_attention block alignment).
        L = input_ids.shape[1]
        pad = (128 - L % 128) % 128
        if pad > 0:
            input_ids = F.pad(input_ids, (0, pad), value = 0)
            input_pos = F.pad(input_pos, (0, pad), value = 0)
            batch_idx = F.pad(batch_idx, (0, pad), value = 0)

        input_lengths = torch.tensor(
            [s.input_length for s in batch], dtype = torch.long, device = self.device
        )
        logits_positions = input_lengths.cumsum(dim = 0) - 1  # [num_seqs]

        mask = self.page_table.create_prefill_blockmask_no_paging(batch_idx)

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
    p.add_argument("--decode_kernel_options", default = None,
                   help = "JSON for FlexKernelOptions applied in decode, "
                          "e.g. '{\"PRESCALE_QK\":true,\"USE_TMA\":true}'.")
    p.add_argument("--prefill_kernel_options", default = None,
                   help = "Same but for prefill.")
    # If set, torch.compile the full attention-stack closure in addition to
    # (or instead of) compiling just flex_attention. `reduce-overhead` is
    # the interesting mode; it nests with our CUDA graph capture.
    p.add_argument("--compile_model_forward", default = None,
                   choices = [None, "default", "reduce-overhead",
                              "max-autotune-no-cudagraphs"])
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
    # Load eager; we swap attention forward below.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype = torch.bfloat16,
        attn_implementation = "eager",
    ).to("cuda")
    model.eval()

    if args.lora_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(Path(args.lora_adapter).resolve()),
            is_trainable = False,
        )
        # Merge so attention forward below sees merged weights without the
        # PEFT wrapper mangling `self.q_proj` etc.
        model = model.merge_and_unload()
        model.eval()

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
    )

    # Optionally compile the manual forward walker. This fuses the layer-stack
    # ops around flex_attention. Under CUDA graph capture, the compiled
    # function gets captured into the same graph.
    if args.compile_model_forward:
        torch._dynamo.config.cache_size_limit = 256
        print(f"[flex] torch.compile(call_model_with_flex_kwargs, "
              f"mode={args.compile_model_forward!r})")
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
