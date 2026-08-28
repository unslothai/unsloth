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
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# blk.<N>.<tail>
_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

# Sparse MoE experts: only expert_used_count of expert_count read per token, so
# host traffic is a small fraction of their size. The cheap thing to spill.
# Fused (ffn_gate_up_exps, cohere2moe/deepseek2/dots3note) and chunked
# (ffn_*_chexps, grovemoe) spellings are experts too: created per expert and
# dispatched with GGML_OP_MUL_MAT_ID, so read just as sparsely as the split form.
# NOT ffn_routed_up/down: kimi-k3 creates it {n_embd, n_embd_latent} with no
# expert axis and plain GGML_OP_MUL_MAT, so every token crosses it -- spilling it
# would send a hot tensor to the host at the rate reserved for cold ones.
_MOE_EXPERT_RE = re.compile(r"^ffn_(up|gate|down|gate_up)_(exps|chexps)\.weight$")
# Dense FFN. Fully activated: every byte crosses the bus every token.
_DENSE_FFN_RE = re.compile(r"^ffn_(up|gate|down)\.weight$")


class SpillClass(Enum):
    """One rung's worth of tensors, in the order the ladder gives them up.

    The ladder used to have a single FFN rung: a block's whole spillable FFN
    moved or none of it did. That is coarse in the direction that costs the most,
    because the planner only ever needs to cover a DEFICIT, and a whole-FFN unit
    overshoots it by up to two thirds. Splitting the FFN into its three matrices
    lets the same deficit be covered by moving about a third as many bytes.

    The order below is llama.cpp's, from ``common/fit.cpp:407-440``, where the
    per-layer fractions are named for what STAYS resident and so read backwards:
    ``LAYER_FRACTION_GATE`` moves ``ffn_down`` only, ``_UP`` moves down plus
    gate, ``_ATTN`` moves the whole FFN. llama.cpp applies that gradation to the
    single boundary layer and no other (``fit.cpp:490``); this applies it to
    every layer.

    UNMEASURED, and deliberately so: ``offload_cost_model`` prices all three
    expert matrices identically (same ``Access.SCATTERED``, same routed
    fraction) and they are within a few percent of the same size, so nothing in
    our cost model prefers this order to any other. The GRANULARITY is what pays;
    the ORDER is inherited on the assumption that llama.cpp had a reason.
    ``PlanOptions.ffn_rung_order`` exists so a benchmark can contradict it.
    """

    FFN_DOWN = "ffn_down"
    # Fused ``gate_up`` counts here: it IS the up projection, with the gate
    # welded on, so there is no separate gate rung left for such a model.
    FFN_UP = "ffn_up"
    FFN_GATE = "ffn_gate"
    # Shared experts and any non-expert FFN on a MoE model: fully activated, so
    # dense-FFN bandwidth. Empty on a dense model, whose FFN is already rungs 1-3.
    DENSE_FFN = "dense_ffn"
    # The four big attention projections. NOT the norms, which are on the
    # critical path for a rounding error of size.
    ATTENTION = "attention"


# Ordered by what the ladder gives up first. lm_head and the KV cache are rungs
# too, but they are not per-block and are handled directly by the planner.
FFN_SPILL_CLASSES = (SpillClass.FFN_DOWN, SpillClass.FFN_UP, SpillClass.FFN_GATE)
BLOCK_SPILL_CLASSES = FFN_SPILL_CLASSES + (SpillClass.DENSE_FFN, SpillClass.ATTENTION)

# Which tail names fall in which class, for a MoE and for a dense model. Kept as
# ONE table so the byte accounting and the emitted ``-ot`` pattern can never
# disagree: a plan that credits itself bytes its pattern does not move is the
# failure mode that produced a silent 20 GiB miscount once already.
_CLASS_BODIES: dict[SpillClass, tuple[str, str]] = {
    #                     MoE                              dense
    SpillClass.FFN_DOWN: ("ffn_down_(exps|chexps)", "ffn_down"),
    SpillClass.FFN_UP: ("ffn_(up|gate_up)_(exps|chexps)", "ffn_up"),
    SpillClass.FFN_GATE: ("ffn_gate_(exps|chexps)", "ffn_gate"),
    # Shared experts, and the plain dense FFN that MoE architectures put in
    # their first k layers. Both are read in full every token, unlike the routed
    # experts above, which is why they are a later rung than all three of those.
    SpillClass.DENSE_FFN: ("ffn_(up|gate|down|gate_up)(_shexp)?", ""),
    SpillClass.ATTENTION: ("attn_(q|k|v|qkv|output)", "attn_(q|k|v|qkv|output)"),
}

_CLASS_RES: dict[SpillClass, tuple[re.Pattern, re.Pattern]] = {
    cls: (
        re.compile(rf"^{moe}\.weight$") if moe else re.compile(r"(?!)"),
        re.compile(rf"^{dense}\.weight$") if dense else re.compile(r"(?!)"),
    )
    for cls, (moe, dense) in _CLASS_BODIES.items()
}


def classify_tail(tail: str, is_moe: bool) -> Optional[SpillClass]:
    """Which rung ``blk.N.<tail>`` belongs to, or None for never-spillable.

    None covers the norms, the router (``ffn_gate_inp``), and the recurrent
    state: each is either tiny, on the critical path for every token, or moved
    only by a layer split that ``-ot`` cannot express.
    """
    for cls in BLOCK_SPILL_CLASSES:
        if _CLASS_RES[cls][0 if is_moe else 1].match(tail):
            return cls
    return None


@dataclass(frozen = True)
class BlockLayout:
    """One transformer block, split into what may and may not be spilled."""

    index: int
    # ffn_*_exps (MoE) or plain ffn_* (dense). Safe to push to host RAM.
    spillable_bytes: int
    # attention, norms, routers, shared experts, ssm: on the critical path every
    # token, or the KV cache hangs off them.
    resident_bytes: int
    # The same bytes again, split by rung. Default 0 so every hand-built layout
    # in the existing tests still constructs, and so a layout that could not be
    # graded (an architecture whose tails match none of the class patterns) falls
    # back to the coarse whole-FFN rung rather than silently spilling nothing.
    ffn_down_bytes: int = 0
    ffn_up_bytes: int = 0
    ffn_gate_bytes: int = 0
    dense_ffn_bytes: int = 0
    attn_bytes: int = 0
    # The GGUF quant type of each rung's tensors ("" when unknown), because what
    # a rung costs on the CPU is not a function of its BYTES alone.
    #
    # MEASURED on gemma-4-26B-A4B, same model, three quants, pure CPU (-ngl 0),
    # tokens/s x file size as a per-byte proxy:
    #
    #   Q2_K_XL  IQ4_NL down + IQ2_XS  gate_up   327.6   ladder 1.27-1.33x
    #   Q3_K_XL  IQ4_NL down + IQ3_XXS gate_up   361.6   ladder 1.09-1.34x
    #   Q4_K_XL  Q5_1   down + Q4_K    gate_up   403.3   ladder 0.93-0.96x
    #
    # The IQ mixes reach 81% of the K-quant's per-byte CPU throughput, and the
    # ladder wins exactly where the tensor it keeps RESIDENT is an IQ type. So
    # the rung order is a property of the types, not of the model or the host --
    # three earlier readings (host speed, model family, spill depth) were each
    # falsified by the next batch of cells.
    ffn_down_type: str = ""
    ffn_up_type: str = ""
    ffn_gate_type: str = ""

    @property
    def graded(self) -> bool:
        """Whether the three FFN rungs account for the whole spillable FFN.

        False means the sub-FFN rungs must not be used for this block: the
        breakdown would understate what a pattern moves, and the planner would
        fill VRAM against a deficit it had not really closed.
        """
        graded = self.ffn_down_bytes + self.ffn_up_bytes + self.ffn_gate_bytes
        return self.spillable_bytes > 0 and graded == self.spillable_bytes

    def class_bytes(self, cls: SpillClass) -> int:
        return {
            SpillClass.FFN_DOWN: self.ffn_down_bytes,
            SpillClass.FFN_UP: self.ffn_up_bytes,
            SpillClass.FFN_GATE: self.ffn_gate_bytes,
            SpillClass.DENSE_FFN: self.dense_ffn_bytes,
            SpillClass.ATTENTION: self.attn_bytes,
        }[cls]


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


def _kv_heads_total(n_kv_head, n_attention: int) -> int:
    """Total KV heads across `n_attention` layers, for a scalar OR a per-layer list.

    ``attention.head_count_kv`` is a bare scalar on most architectures and a
    PER-LAYER array on some. gemma-4-26B-A4B ships a 30-entry
    ``[8, 8, 8, 8, 8, 2, ...]`` and gemma-4-31B a 60-entry
    ``[16, 16, 16, 16, 16, 4, ...]``.

    ``int()`` on a list raises, ``layout_from_gguf`` swallows the error to a
    debug log, and the planner then abstains on every quant of both families --
    six of the thirteen models in the sweep -- with nothing visibly failing,
    because an abstain falls through to ``--fit on``. It surfaced only when a
    Kaggle cell finally ran one and reported "layout or device inventory is
    incomplete".

    Summing is the right arithmetic and not merely a type fix: these models mix
    8 with 2, and 16 with 4, so one head count times a layer count is the wrong
    number even where it happens not to raise. ``llama_cpp.py`` already models
    this as ``_n_kv_heads_by_layer``; the layout never learned about it.

    A list shorter than the layer count is padded with its own last value, which
    is what a trailing uniform tail means; 0 signals "unusable", so the caller
    abstains rather than pricing a cache of zero.
    """
    if n_attention <= 0:
        return 0
    if isinstance(n_kv_head, (list, tuple)):
        heads = [int(h) for h in n_kv_head]
        if not heads:
            return 0
        return sum(heads[i] if i < len(heads) else heads[-1]
                   for i in range(n_attention))
    try:
        return n_attention * int(n_kv_head)
    except (TypeError, ValueError):
        return 0


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

    kv_heads_total = _kv_heads_total(n_kv_head, int(n_attention))
    if not kv_heads_total:
        return ModelLayout()

    kv_per_token = kv_heads_total * (int(key_len) + int(val_len)) * 2

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
    per_class: dict[int, dict[SpillClass, int]] = {}
    lm_head = 0
    token_embd = 0
    per_layer_embd = 0
    other_resident = 0

    per_class_type: dict = {}
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
            cls = classify_tail(tail, is_moe)
            if cls is not None:
                bucket = per_class.setdefault(index, {})
                bucket[cls] = bucket.get(cls, 0) + nbytes
                # The rung's quant TYPE, for the CPU cost of running it there.
                # Recorded from the largest tensor in the class so a stray F32
                # bias cannot outvote the weight matrix that dominates the work.
                tb = per_class_type.setdefault(index, {})
                if nbytes > tb.get(cls, (0, ""))[0]:
                    tname = getattr(tensor, "tensor_type", None)
                    tb[cls] = (nbytes, getattr(tname, "name", "") or "")
            continue
        if name.startswith("per_layer_token_embd"):
            # gemma3/gemma4 per-layer embeddings. Host-resident like token_embd,
            # and kept in a SEPARATE total because the tied-embedding branch
            # below duplicates the VOCABULARY matrix and must not be handed this
            # as well: on gemma-4-E2B it is 1540 MiB against a 264 MiB vocabulary,
            # so folding it in charged VRAM 1540 MiB for a tensor llama.cpp never
            # puts there. See the comment on that branch for what it cost.
            per_layer_embd += nbytes
        elif "token_embd" in name:
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
        # ``token_embd`` ONLY, never the per-layer embeddings. What llama.cpp
        # duplicates is the vocabulary matrix -- ggml_dup_tensor on the tensor
        # routed through the OUTPUT buffer list -- and the per-layer input
        # embeddings are neither an output nor duplicated.
        #
        # MEASURED cost of getting this wrong, on gemma-4-E2B-it UD-Q4_K_XL with
        # 4.15 GiB free: the layout charged VRAM 1804 MiB for the duplicate
        # instead of 264 MiB, so ``all_resident_bytes`` came to 3.57 GiB against
        # the 1.45 GiB llama.cpp actually placed, and the planner spilled the FFN
        # of 32 of 35 blocks to cover a deficit that did not exist. ``--fit on``
        # left the model wholly resident and measured 447.8 t/s; the planner
        # measured 187.7. That is 0.42x, on a model that FIT -- the same failure
        # #9861 reported and the same shape as its two worst cells.
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
            ffn_down_bytes = per_class.get(i, {}).get(SpillClass.FFN_DOWN, 0),
            ffn_up_bytes = per_class.get(i, {}).get(SpillClass.FFN_UP, 0),
            ffn_gate_bytes = per_class.get(i, {}).get(SpillClass.FFN_GATE, 0),
            dense_ffn_bytes = per_class.get(i, {}).get(SpillClass.DENSE_FFN, 0),
            attn_bytes = per_class.get(i, {}).get(SpillClass.ATTENTION, 0),
            ffn_down_type = per_class_type.get(i, {}).get(SpillClass.FFN_DOWN, (0, ""))[1],
            ffn_up_type = per_class_type.get(i, {}).get(SpillClass.FFN_UP, (0, ""))[1],
            ffn_gate_type = per_class_type.get(i, {}).get(SpillClass.FFN_GATE, (0, ""))[1],
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
        token_embd_bytes = token_embd + per_layer_embd,
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
    # Same set _MOE_EXPERT_RE selected, or the plan credits itself bytes the
    # emitted pattern never moves.
    body = "ffn_(up|gate|down|gate_up)_(exps|chexps)" if layout.is_moe else "ffn_(up|gate|down)"
    if indices is None:
        block = r"\d+"
    else:
        block = "|".join(str(i) for i in sorted(indices))
        block = f"({block})"
    return rf"^blk\.{block}\.{body}\.weight$"


def spill_pattern_for_class(
    layout: ModelLayout, cls: SpillClass, indices: Optional[list[int]] = None
) -> str:
    """The anchored ``-ot`` pattern for one rung over ``indices``.

    Anchored for the same reason :func:`spill_pattern_for` is: llama.cpp matches
    with ``std::regex_search``, so an unanchored ``ffn_down`` would also match
    ``ffn_down_exps`` and move the whole expert rung when only the dense one was
    asked for -- which on a MoE model is most of the file.
    """
    body = _CLASS_BODIES[cls][0 if layout.is_moe else 1]
    if not body:
        raise ValueError(f"{cls.value} has no tensors on a {'MoE' if layout.is_moe else 'dense'} model")
    if indices is None:
        block = r"\d+"
    else:
        block = "(" + "|".join(str(i) for i in sorted(indices)) + ")"
    return rf"^blk\.{block}\.{body}\.weight$"


LM_HEAD_PATTERN = r"^output\.weight$"
