# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF tensor layout, bucketed by where each tensor is allowed to live.

Split out from ``offload_planner`` on purpose: this half does file IO and knows
about GGUF key names, the other half is pure arithmetic. The planner can then be
tested exhaustively from hand-built layouts with no fixtures on disk.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# blk.<N>.<tail>
_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

# Sparse MoE experts: only expert_used_count of expert_count read per token, so
# host traffic is a small fraction of their size. The cheap thing to spill.
_MOE_EXPERT_RE = re.compile(r"^ffn_(up|gate|down)_exps\.weight$")
# Dense FFN. Fully activated: every byte crosses the bus every token.
_DENSE_FFN_RE = re.compile(r"^ffn_(up|gate|down)\.weight$")


@dataclass(frozen = True)
class BlockLayout:
    """One transformer block, split into what may and may not be spilled."""

    index: int
    # ffn_*_exps (MoE) or plain ffn_* (dense). Safe to push to host RAM.
    spillable_bytes: int
    # attention, norms, routers, shared experts, ssm: on the critical path every
    # token, or the KV cache hangs off them.
    resident_bytes: int


@dataclass(frozen = True)
class ModelLayout:
    """Everything the planner needs, and nothing about files."""

    arch: str = ""
    n_layers: int = 0
    n_attention_layers: int = 0
    blocks: tuple[BlockLayout, ...] = field(default_factory = tuple)
    # Rides the layer list at index n_layer_all, so it is GPU-resident for any
    # -ngl >= 1 and can only be moved with an explicit override.
    lm_head_bytes: int = 0
    # llama-model.cpp pins dev_input to the CPU unconditionally, so this is
    # never charged to VRAM. Tracked because it IS charged to host RAM.
    token_embd_bytes: int = 0
    # output_norm and friends: GPU-resident, too small to be worth spilling.
    other_resident_bytes: int = 0
    # Attention cache for ONE token at f16, across the attention layers only.
    kv_bytes_per_token_f16: int = 0
    # Mamba conv/SSM state. Context independent; follows the layer, which -ot never moves.
    recurrent_bytes: int = 0
    n_ctx_train: int = 0
    is_moe: bool = False
    # Sparse-MoE routing: experts read per token is expert_used/expert_count.
    # Offloaded experts move only that fraction per token, a dense FFN all of it.
    n_expert: int = 0
    n_expert_used: int = 0
    # ``blocks`` drops the trailing nextn/MTP blk.<N> tensors: block_count counts
    # them (llama-model.cpp reads it into n_layer_all) but the target does not use
    # them. They are real blk.<N>.ffn_* weights (models/qwen35moe.cpp,
    # load_block_mtp), so an unbounded ^blk\.\d+\. spill pattern WOULD match them
    # once a draft is loaded. The planner uses this to stay bounded.
    has_excluded_blocks: bool = False
    # Total bytes of those dropped blocks, so a caller that knows a draft WILL
    # engage can charge them back. Dropping them suits the ordinary load: every
    # trailing block gets TENSOR_SKIP unless load_mtp is set
    # (models/glm4-moe.cpp:42-44, the same gate in every embedded-MTP arch) and
    # TENSOR_SKIP returns before the tensor exists
    # (llama-model-loader.cpp:1123-1131). But ``--spec-type draft-mtp`` sets
    # load_mtp on the TARGET's own params (common/common.cpp:1713), so the block is
    # materialised at its layer's buffer type, and i_gpu_start counting back from
    # n_layer_all (llama-model.cpp:1449) puts those blocks on a GPU FIRST.
    # llama.cpp's own fitter widens its offloadable-layer count the same way
    # (common/fit.cpp:139-142). Zero when nothing was dropped.
    excluded_block_bytes: int = 0
    # Sliding-window attention: some layers keep a window-sized cache, some the
    # full context (llama-kv-cache-iswa.cpp:69-104 builds two caches and filters
    # each by hparams.is_swa(il)), interleaved per layer. Every layer is still an
    # attention layer, so n_attention_layers does NOT reveal this. A multi-device
    # split has to know WHERE the big caches land, so the planner abstains.
    has_swa: bool = False
    # False when a needed quantity could not be read. The planner abstains.
    complete: bool = False

    @property
    def spillable_bytes(self) -> int:
        return sum(b.spillable_bytes for b in self.blocks)

    @property
    def block_resident_bytes(self) -> int:
        return sum(b.resident_bytes for b in self.blocks)

    def kv_bytes(
        self,
        n_ctx: int,
        bytes_per_elem: int = 2,
    ) -> int:
        """Attention cache at ``n_ctx``. bytes_per_elem 2 = f16, 1 = q8_0-ish."""
        if self.kv_bytes_per_token_f16 <= 0 or n_ctx <= 0:
            return 0
        return self.kv_bytes_per_token_f16 * n_ctx * bytes_per_elem // 2


def _field(
    reader,
    key: str,
    default = None,
):
    f = reader.fields.get(key)
    if f is None:
        return default
    try:
        return f.contents()
    except Exception:  # a malformed field must not take the whole load down
        return default


def layout_from_gguf(path: str) -> ModelLayout:
    """Read ``path`` into a :class:`ModelLayout`.

    Returns an incomplete layout (``complete = False``) rather than raising when
    anything required is missing, so a surprising GGUF makes the planner abstain
    instead of failing a load that llama.cpp would have handled.
    """
    try:
        from gguf import GGUFReader
        reader = GGUFReader(path)
    except Exception as exc:
        logger.debug("offload layout: cannot read %s (%s)", path, exc)
        return ModelLayout()

    try:
        return _layout_from_reader(reader)
    except Exception as exc:
        logger.debug("offload layout: cannot interpret %s (%s)", path, exc)
        return ModelLayout()


def _layout_from_reader(reader) -> ModelLayout:
    # Split GGUF: llama.cpp loads every sibling shard
    # (llama-model-loader.cpp:590-618), but GGUFReader memmaps only the ONE path
    # it was given. Shard 1 still carries the metadata, so the layout would look
    # complete while undercounting resident and spillable by most of the model --
    # an overstated fit, too few -ot patterns, and a startup OOM with --fit off.
    # Abstain instead; the seam then reproduces --fit on exactly.
    if int(_field(reader, "split.count") or 0) > 1:
        return ModelLayout()

    arch = str(_field(reader, "general.architecture") or "")
    if not arch:
        return ModelLayout()

    blocks_total = _field(reader, f"{arch}.block_count")
    if not blocks_total:
        return ModelLayout()
    blocks_total = int(blocks_total)

    # llama.cpp keeps embedded MTP blocks out of the target context and prices
    # their cache separately, so the attention count must not include them.
    nextn = int(_field(reader, f"{arch}.nextn_predict_layers") or 0)
    n_layers = max(0, blocks_total - nextn)

    # Hybrid: only 1 in full_attention_interval layers carries a KV cache, the
    # rest are recurrent. Absent (or 0) means every layer is attention.
    fai = int(_field(reader, f"{arch}.full_attention_interval") or 0)
    n_attention = -(-n_layers // fai) if fai > 0 else n_layers
    n_recurrent = max(0, n_layers - n_attention)

    n_kv_head = _field(reader, f"{arch}.attention.head_count_kv")
    n_head = _field(reader, f"{arch}.attention.head_count")
    n_embd = _field(reader, f"{arch}.embedding_length")
    key_len = _field(reader, f"{arch}.attention.key_length")
    val_len = _field(reader, f"{arch}.attention.value_length")
    if key_len is None and n_embd and n_head:
        key_len = int(n_embd) // int(n_head)
    if val_len is None:
        val_len = key_len
    if not n_kv_head or not key_len or not val_len:
        return ModelLayout()

    kv_per_token = int(n_attention) * int(n_kv_head) * (int(key_len) + int(val_len)) * 2

    # Charging every layer the full context above is the safe direction for the
    # TOTAL; what it cannot say is which layers hold the big caches.
    has_swa = bool(_field(reader, f"{arch}.attention.sliding_window") or 0)

    # Mamba conv + SSM state, one f32 copy per sequence. Mirrors llama.cpp's
    # own sizing; zero when the model has no recurrent layers.
    d_inner = int(_field(reader, f"{arch}.ssm.inner_size") or 0)
    d_state = int(_field(reader, f"{arch}.ssm.state_size") or 0)
    n_group = int(_field(reader, f"{arch}.ssm.group_count") or 0)
    d_conv = int(_field(reader, f"{arch}.ssm.conv_kernel") or 0)
    recurrent = 0
    if n_recurrent and d_inner and d_state and d_conv:
        n_embd_r = max(0, d_conv - 1) * (d_inner + 2 * n_group * d_state)
        n_embd_s = d_state * d_inner
        recurrent = n_recurrent * (n_embd_r + n_embd_s) * 4

    n_expert = int(_field(reader, f"{arch}.expert_count") or 0)
    n_expert_used = int(_field(reader, f"{arch}.expert_used_count") or 0)
    is_moe = bool(n_expert)

    spill: dict[int, int] = {}
    resident: dict[int, int] = {}
    lm_head = 0
    token_embd = 0
    other_resident = 0

    for tensor in reader.tensors:
        name = str(tensor.name)
        nbytes = int(tensor.n_bytes)
        match = _BLOCK_RE.match(name)
        if match:
            index = int(match.group(1))
            tail = match.group(2)
            # Shared experts (ffn_*_shexp) and routers (ffn_gate_inp*) run on every
            # token: dense-FFN bandwidth for a rounding error of size. Not spillable.
            spillable = _MOE_EXPERT_RE.match(tail) or (not is_moe and _DENSE_FFN_RE.match(tail))
            if spillable:
                spill[index] = spill.get(index, 0) + nbytes
            else:
                resident[index] = resident.get(index, 0) + nbytes
            continue
        if "token_embd" in name:
            token_embd += nbytes
        elif name == "output.weight":
            lm_head += nbytes
        else:
            other_resident += nbytes

    if not spill and not resident:
        return ModelLayout()

    # Tied embeddings duplicate the vocabulary matrix, they do not SAVE it. With no
    # output.weight llama.cpp re-creates the output tensor from token_embd as
    # TENSOR_DUPLICATED (models/llama.cpp:41-45, models/qwen3.cpp:22-25,
    # models/gemma3.cpp:43-47, and ~60 more) and routes a duplicated TOKEN_EMBD
    # through the OUTPUT buffer list (llama-model-loader.cpp:1113-1114). dev_input
    # is CPU-pinned while dev_output follows the layer split (llama-model.cpp:1465,
    # 1474), so the buffer-type contexts differ, the same-context reuse check misses
    # (llama-model-loader.cpp:1309-1314), and ggml_dup_tensor allocates a second
    # full matrix (llama-model-loader.cpp:1318) that load_all_data fills by name
    # with a real host to device copy (:1542, :1583). Counting the one stored tensor
    # as host-only understates VRAM by a whole vocabulary matrix -- the optimistic
    # direction. Resident, not lm_head: the duplicate keeps the name
    # token_embd.weight, so LM_HEAD_PATTERN cannot match and the lm_head rung would
    # credit a spill that moves nothing.
    if not lm_head and token_embd:
        other_resident += token_embd

    # Trailing nextn/MTP blocks are NOT part of the target model and are not loaded
    # unless a draft is engaged, so an -ot naming them moves nothing: measured,
    # spilling only blk.<nextn> leaves the host buffer at exactly token_embd and the
    # device buffer unchanged. Counting them spillable would credit bytes that can
    # never be freed. Unsloth prices the drafter separately anyway.
    all_block_indices = set(spill) | set(resident)
    block_indices = sorted(i for i in all_block_indices if i < n_layers)
    has_excluded = any(i >= n_layers for i in all_block_indices)
    excluded_bytes = sum(
        spill.get(i, 0) + resident.get(i, 0) for i in all_block_indices if i >= n_layers
    )
    blocks = tuple(
        BlockLayout(
            index = i,
            spillable_bytes = spill.get(i, 0),
            resident_bytes = resident.get(i, 0),
        )
        for i in block_indices
    )

    return ModelLayout(
        arch = arch,
        n_layers = n_layers,
        n_attention_layers = int(n_attention),
        has_swa = has_swa,
        blocks = blocks,
        lm_head_bytes = lm_head,
        token_embd_bytes = token_embd,
        other_resident_bytes = other_resident,
        kv_bytes_per_token_f16 = kv_per_token,
        recurrent_bytes = recurrent,
        n_ctx_train = int(_field(reader, f"{arch}.context_length") or 0),
        is_moe = is_moe,
        n_expert = n_expert,
        n_expert_used = n_expert_used,
        has_excluded_blocks = has_excluded,
        excluded_block_bytes = excluded_bytes,
        complete = True,
    )


def spill_pattern_for(layout: ModelLayout, indices: Optional[list[int]] = None) -> str:
    """The anchored ``-ot`` pattern matching the spillable FFN of ``indices``.

    Anchored because llama.cpp matches with ``std::regex_search``: an unanchored
    ``output\\.weight`` also matches every ``blk.N.attn_output.weight``, which
    silently moves 16 attention projections nobody asked to move. The trailing
    ``\\.weight$`` likewise keeps ``ffn_(up|gate|down)\\.`` from matching
    ``ffn_gate_inp.weight``.
    """
    body = "ffn_(up|gate|down)_exps" if layout.is_moe else "ffn_(up|gate|down)"
    if indices is None:
        block = r"\d+"
    else:
        block = "|".join(str(i) for i in sorted(indices))
        block = f"({block})"
    return rf"^blk\.{block}\.{body}\.weight$"


LM_HEAD_PATTERN = r"^output\.weight$"
