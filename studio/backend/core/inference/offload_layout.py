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
import os

logger = logging.getLogger(__name__)

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

# Sparse MoE experts: only expert_used_count of expert_count read per token, so host traffic is a small fraction of
# their size. The cheap thing to spill. Fused (ffn_gate_up_exps, cohere2moe/deepseek2/dots3note) and chunked
# (ffn_*_chexps, grovemoe) spellings are experts too: created per expert and dispatched with GGML_OP_MUL_MAT_ID, so read
# just as sparsely as the split form. NOT ffn_routed_up/down: kimi-k3 creates it {n_embd, n_embd_latent} with no expert
# axis and plain GGML_OP_MUL_MAT, so every token crosses it -- spilling it would send a hot tensor to the host at the
# rate reserved for cold ones.
_MOE_EXPERT_RE = re.compile(r"^ffn_(up|gate|down|gate_up)_(exps|chexps)\.weight$")
# Dense FFN. Fully activated: every byte crosses the bus every token.
_DENSE_FFN_RE = re.compile(r"^ffn_(up|gate|down)\.weight$")


class SpillClass(Enum):
    """One rung's worth of tensors, in the order the ladder gives them up. The order is
    llama.cpp's, from ``common/fit.cpp:407-440``, where the per-layer fractions are named for
    what STAYS resident and so read backwards: ``LAYER_FRACTION_GATE`` moves ``ffn_down`` only,
    ``_UP`` moves down plus gate, ``_ATTN`` moves the whole FFN.
    """

    FFN_DOWN = "ffn_down"
    # Fused ``gate_up`` counts here: it IS the up projection, with the gate
    # welded on, so there is no separate gate rung left for such a model.
    FFN_UP = "ffn_up"
    FFN_GATE = "ffn_gate"
    # Shared experts and any non-expert FFN on a MoE model: fully activated, so
    # dense-FFN bandwidth. Empty on a dense model, whose FFN is already rungs 1-3.
    DENSE_FFN = "dense_ffn"
    # The four big attention projections.
    ATTENTION = "attention"


# Ordered by what the ladder gives up first.
FFN_SPILL_CLASSES = (SpillClass.FFN_DOWN, SpillClass.FFN_UP, SpillClass.FFN_GATE)
BLOCK_SPILL_CLASSES = FFN_SPILL_CLASSES + (SpillClass.DENSE_FFN, SpillClass.ATTENTION)

# Which tail names fall in which class, for a MoE and for a dense model.
_CLASS_BODIES: dict[SpillClass, tuple[str, str]] = {
    #                     MoE                              dense
    SpillClass.FFN_DOWN: ("ffn_down_(exps|chexps)", "ffn_down"),
    SpillClass.FFN_UP: ("ffn_(up|gate_up)_(exps|chexps)", "ffn_up"),
    SpillClass.FFN_GATE: ("ffn_gate_(exps|chexps)", "ffn_gate"),
    # Shared experts, and the plain dense FFN MoE architectures put in their first k
    # layers: read in full every token, unlike the routed experts above, so a later rung.
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
    """Which rung ``blk.N.<tail>`` belongs to, or None for never-spillable."""
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
    # attention, norms, routers, shared experts, ssm: on the critical path every token, or the KV cache hangs off them.
    resident_bytes: int
    # The same bytes again, split by rung.
    ffn_down_bytes: int = 0
    ffn_up_bytes: int = 0
    ffn_gate_bytes: int = 0
    dense_ffn_bytes: int = 0
    attn_bytes: int = 0
    # The GGUF quant type of each rung's tensors ("" when unknown): what a rung costs
    # on the CPU is not a function of its BYTES alone, and the ladder wins exactly
    # where the tensor it keeps RESIDENT is an IQ type.
    ffn_down_type: str = ""
    ffn_up_type: str = ""
    ffn_gate_type: str = ""

    @property
    def graded(self) -> bool:
        """Whether the three FFN rungs account for the whole spillable FFN."""
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
    # Rides the layer list at index n_layer_all, so it is GPU-resident for any -ngl >= 1 and can only be moved with an
    # explicit override.
    lm_head_bytes: int = 0
    # llama-model.cpp pins dev_input to the CPU unconditionally, so this is never charged to VRAM. Tracked because it IS
    # charged to host RAM.
    token_embd_bytes: int = 0
    # output_norm and friends: GPU-resident, too small to be worth spilling.
    other_resident_bytes: int = 0
    # Attention cache for ONE token at f16, across the attention layers only.
    kv_bytes_per_token_f16: int = 0
    # Mamba conv/SSM or KDA conv/recurrent state; context independent, and follows the layer, which -ot never moves
    recurrent_bytes: int = 0
    n_ctx_train: int = 0
    is_moe: bool = False
    # offloaded experts move only expert_used/expert_count per token, a dense FFN all of it
    # Sparse-MoE routing: experts read per token is expert_used/expert_count. Offloaded experts move only that fraction
    # per token, a dense FFN all of it.
    n_expert: int = 0
    n_expert_used: int = 0
    # ``blocks`` drops the trailing nextn/MTP blk.<N> tensors: block_count counts them (llama-model.cpp reads it into
    # n_layer_all) but the target does not use them. They are real blk.<N>.ffn_* weights (models/qwen35moe.cpp,
    # load_block_mtp), so an unbounded ^blk\.\d+\. spill pattern WOULD match them once a draft is loaded. The planner
    # uses this to stay bounded.
    has_excluded_blocks: bool = False
    # Total bytes of those dropped blocks, so a caller that knows a draft WILL engage can charge them back. Dropping
    # them suits the ordinary load: every trailing block gets TENSOR_SKIP unless load_mtp is set
    # (models/glm4-moe.cpp:42-44, the same gate in every embedded-MTP arch) and TENSOR_SKIP returns before the tensor
    # exists (llama-model-loader.cpp:1123-1131). But ``--spec-type draft-mtp`` sets load_mtp on the TARGET's own params
    # (common/common.cpp:1713), so the block is materialised at its layer's buffer type, and i_gpu_start counting back
    # from n_layer_all (llama-model.cpp:1449) puts those blocks on a GPU FIRST. llama.cpp's own fitter widens its
    # offloadable-layer count the same way (common/fit.cpp:139-142). Zero when nothing was dropped.
    excluded_block_bytes: int = 0
    # sliding-window attention interleaves window-sized and full-context caches per layer
    # Sliding-window attention: some layers keep a window-sized cache, some the full context
    # (llama-kv-cache-iswa.cpp:69-104 builds two caches and filters each by hparams.is_swa(il)), interleaved per layer.
    # Every layer is still an attention layer, so n_attention_layers does NOT reveal this. A multi-device split has to
    # know WHERE the big caches land, so the planner abstains.
    has_swa: bool = False
    # Multi-head latent attention: the cache is one compressed K-only latent per token, not a K+V pair per head, so the
    # per-head product above over-counts it by up to two orders of magnitude. Keyed on attention.key_length_mla AND
    # attention.value_length_mla, as llama-hparams.cpp:llama_hparams::is_mla is: a GGUF that carries only
    # attention.kv_lora_rank (unsloth/DeepSeek-R1-GGUF, unsloth/DeepSeek-V3-0324-GGUF) gets the full per-head K+V cache
    # from llama.cpp, and the product above is exact for it -- claiming MLA there makes the planner discard the exact
    # number for a floor that is 40% short.
    has_mla: bool = False
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


_SPLIT_SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def split_shard_paths(path: str) -> Optional[list[str]]:
    """Every shard of the split *path* belongs to, in order, or None when the name
    is not llama.cpp's ``<prefix>-NNNNN-of-MMMMM.gguf`` (llama_split_path)."""
    directory, name = os.path.split(path)
    match = _SPLIT_SHARD_RE.match(name)
    if not match:
        return None
    prefix, _index, total = match.groups()
    return [
        os.path.join(directory, f"{prefix}-{i:05d}-of-{int(total):05d}.gguf")
        for i in range(1, int(total) + 1)
    ]


def layout_from_gguf(path: str, *, all_shards: bool = False) -> ModelLayout:
    """Read ``path`` into a :class:`ModelLayout`.

    Returns an incomplete layout (``complete = False``) rather than raising when
    anything required is missing, so a surprising GGUF makes the planner abstain
    instead of failing a load that llama.cpp would have handled.
    ``all_shards`` reads every sibling shard; all of them must be present.
    """
    try:
        from gguf import GGUFReader
        readers = [GGUFReader(path)]
        if all_shards and int(_field(readers[0], "split.count") or 0) > 1:
            shards = split_shard_paths(path)
            if not shards or not all(os.path.isfile(p) for p in shards):
                logger.debug("offload layout: split %s is missing a shard", path)
                return ModelLayout()
            readers = [GGUFReader(p) for p in shards]
    except Exception as exc:
        logger.debug("offload layout: cannot read %s (%s)", path, exc)
        return ModelLayout()

    try:
        return _layout_from_readers(readers)
    except Exception as exc:
        logger.debug("offload layout: cannot interpret %s (%s)", path, exc)
        return ModelLayout()


def _kv_heads_total(n_kv_head, n_attention: int) -> int:
    """Total KV heads across `n_attention` layers, for a scalar OR a per-layer list."""
    if n_attention <= 0:
        return 0
    if isinstance(n_kv_head, (list, tuple)):
        heads = [int(h) for h in n_kv_head]
        if not heads:
            return 0
        return sum(heads[i] if i < len(heads) else heads[-1] for i in range(n_attention))
    try:
        return n_attention * int(n_kv_head)
    except (TypeError, ValueError):
        return 0


# The default llama.cpp uses when a hybrid's GGUF omits full_attention_interval, per architecture. Read straight off
# the `uint32_t full_attn_interval = N;` that precedes each optional get_key: src/models/qwen3next.cpp:24,
# qwen35.cpp:23, qwen35moe.cpp:26, minimax-01.cpp:13. An architecture not listed abstains below rather than guessing.
_FULL_ATTENTION_INTERVAL_DEFAULT: dict[str, int] = {
    "qwen3next": 4,
    "qwen35": 4,
    "qwen35moe": 4,
    "minimax-01": 8,
}


# Architectures whose recurrent rows hold a Kimi-Delta-Attention state rather than a Mamba one, so llama.cpp sizes
# them from kda.head_dim and the head count (llama-hparams.cpp:n_embd_r, n_embd_s). Named rather than derived from the
# key's presence: an unlisted KDA family (kimi-linear, bailingmoe3) has no measured figure to check the shape against,
# and abstaining is the safe answer for it.
_KDA_STATE_ARCHS: frozenset[str] = frozenset({"kimi-k3", "glm5next"})


# Architectures whose zero-KV-head rows are recurrent only when their FFN width is 0 as well
# (models/nemotron-h.cpp:17, inherited by nemotron_h_moe at models/models.h:1516).
_RECURRENT_NEEDS_ZERO_FFN: frozenset[str] = frozenset({"nemotron_h", "nemotron_h_moe"})


def hybrid_layer_split(
    arch: str,
    n_layers: int,
    *,
    recurrent_layers = None,
    n_kv_head = None,
    full_attention_interval: int = 0,
    feed_forward_length = None,
) -> tuple[int, int, bool]:
    """``(n_attention, n_recurrent, known)`` over a hybrid's ``n_layers`` target rows.

    The one derivation both readers use, so the layout's cache product and the
    estimator's path 2 cannot disagree about the same file. llama.cpp resolves it in
    three steps (models/qwen3next.cpp:load_arch_hparams, and the identical block in
    qwen35, qwen35moe and minimax-01): an explicit per-layer mask first, then
    ``full_attention_interval``, then the ARCHITECTURE's built-in default for it.

    A per-layer ``attention.head_count_kv`` list with zeros beats all three: those rows
    hold no attention cache at all, and llama.cpp reads is_recr straight off them
    (models/nemotron-h.cpp:load_arch_hparams, models/falcon-h1.cpp). ``known`` is False
    only when nothing above said anything, which is the caller's cue to abstain rather
    than call every row attention.
    """
    if n_layers <= 0:
        return 0, 0, False
    known = False
    n_recurrent = 0
    if isinstance(recurrent_layers, (list, tuple)) and len(recurrent_layers) >= n_layers:
        n_recurrent = sum(1 for flag in list(recurrent_layers)[:n_layers] if flag)
        known = True
    else:
        fai = int(full_attention_interval or 0)
        if fai <= 0:
            fai = _FULL_ATTENTION_INTERVAL_DEFAULT.get(arch, 0)
        if fai > 0:
            n_recurrent = max(0, n_layers - -(-n_layers // fai))
            known = True
    n_attention = n_layers - n_recurrent

    if isinstance(n_kv_head, (list, tuple)):
        heads = [int(h) for h in n_kv_head]
        if heads:
            padded = [heads[i] if i < len(heads) else heads[-1] for i in range(n_layers)]
            attention_rows = sum(1 for h in padded if h > 0)
            if 0 < attention_rows < n_layers:
                n_attention = attention_rows
                n_recurrent = n_layers - n_attention
                known = True
                # ...except on nemotron_h, where a zero-head row is recurrent only if its FFN is 0 too. The MLP-only
                # rows are neither attention nor recurrent, so the two counts stop summing to n_layers here.
                if arch in _RECURRENT_NEEDS_ZERO_FFN and isinstance(
                    feed_forward_length, (list, tuple)
                ):
                    ffs = [int(f) for f in feed_forward_length]
                    if ffs:
                        n_recurrent = sum(
                            1
                            for i in range(n_layers)
                            if padded[i] <= 0 and (ffs[i] if i < len(ffs) else ffs[-1]) <= 0
                        )
    return n_attention, n_recurrent, known


def _layout_from_reader(reader) -> ModelLayout:
    return _layout_from_readers([reader])


def _layout_from_readers(readers) -> ModelLayout:
    """One reader per shard, the first carrying the metadata."""
    reader = readers[0]
    # Split GGUF: llama.cpp loads every sibling shard (llama-model-loader.cpp:590-618), but GGUFReader memmaps only the
    # ONE path it was given. Shard 1 still carries the metadata, so the layout would look complete while undercounting
    # resident and spillable by most of the model -- an overstated fit, too few -ot patterns, and a startup OOM with
    # --fit off. Abstain unless every shard was handed over; the seam then reproduces --fit on exactly.
    if (int(_field(reader, "split.count") or 0) or 1) != len(readers):
        return ModelLayout()

    arch = str(_field(reader, "general.architecture") or "")
    if not arch:
        return ModelLayout()

    blocks_total = _field(reader, f"{arch}.block_count")
    if not blocks_total:
        return ModelLayout()
    blocks_total = int(blocks_total)

    # llama.cpp keeps embedded MTP blocks out of the target context and prices their cache separately, so the attention
    # count must not include them.
    nextn = int(_field(reader, f"{arch}.nextn_predict_layers") or 0)
    n_layers = max(0, blocks_total - nextn)

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

    n_attention, n_recurrent, recurrent_known = hybrid_layer_split(
        arch,
        n_layers,
        recurrent_layers = _field(reader, f"{arch}.attention.recurrent_layers"),
        n_kv_head = n_kv_head,
        full_attention_interval = int(_field(reader, f"{arch}.full_attention_interval") or 0),
        feed_forward_length = _field(reader, f"{arch}.feed_forward_length"),
    )

    # A per-layer list with zeros names the rows that carry NO attention cache (a KDA /
    # linear-attention hybrid); summing them away would let a multi-device check spread the
    # cache over rows that hold none of it.
    if isinstance(n_kv_head, (list, tuple)) and n_attention < n_layers:
        _heads = [int(h) for h in n_kv_head]
        if _heads:
            _padded = [_heads[i] if i < len(_heads) else _heads[-1] for i in range(n_layers)]
            if 0 < sum(1 for h in _padded if h > 0) < n_layers:
                n_kv_head = [h for h in _padded if h > 0]

    kv_heads_total = _kv_heads_total(n_kv_head, int(n_attention))
    if not kv_heads_total:
        return ModelLayout()

    kv_per_token = kv_heads_total * (int(key_len) + int(val_len)) * 2

    # charging every layer the full context is the safe direction for the TOTAL
    # Charging every layer the full context above is the safe direction for the TOTAL; what it cannot say is which
    # layers hold the big caches.
    has_swa = bool(_field(reader, f"{arch}.attention.sliding_window") or 0)
    # Both MLA head lengths, never kv_lora_rank: llama-hparams.cpp:llama_hparams::is_mla.
    has_mla = bool(_field(reader, f"{arch}.attention.key_length_mla") or 0) and bool(
        _field(reader, f"{arch}.attention.value_length_mla") or 0
    )

    # Mamba conv + SSM state, one f32 copy per sequence. Mirrors llama.cpp's own sizing; zero when the model has no
    # recurrent layers.
    d_inner = int(_field(reader, f"{arch}.ssm.inner_size") or 0)
    d_state = int(_field(reader, f"{arch}.ssm.state_size") or 0)
    n_group = int(_field(reader, f"{arch}.ssm.group_count") or 0)
    d_conv = int(_field(reader, f"{arch}.ssm.conv_kernel") or 0)
    kda_head_dim = int(_field(reader, f"{arch}.kda.head_dim") or 0)
    if arch not in _KDA_STATE_ARCHS:
        kda_head_dim = 0
    recurrent = 0
    if n_recurrent and d_inner and d_state and d_conv:
        n_embd_r = max(0, d_conv - 1) * (d_inner + 2 * n_group * d_state)
        n_embd_s = d_state * d_inner
        recurrent = n_recurrent * (n_embd_r + n_embd_s) * 4
    elif n_recurrent and kda_head_dim and n_head:
        # A KDA row carries no ssm.inner_size, so the branch above sizes it at zero and every per-slot term the
        # planner adds separately from the cache (resident_floor_bytes, max_context_for's fixed term, the
        # multi-device recurrent guard) silently drops 443 MiB/slot on Kimi-K3. llama-hparams.cpp:n_embd_r/n_embd_s
        # size it from the head count and kda.head_dim instead; the conv kernel defaults to 4 there as well.
        d_inner_kda = int(n_head) * kda_head_dim
        n_embd_r = 3 * max(0, (d_conv or 4) - 1) * d_inner_kda
        n_embd_s = kda_head_dim * kda_head_dim * int(n_head)
        recurrent = n_recurrent * (n_embd_r + n_embd_s) * 4

    # ssm.*/kda.* keys say the model HAS recurrent layers; nothing above could say which.
    if not recurrent_known and ((d_inner and d_state and d_conv) or kda_head_dim):
        logger.debug("offload layout: %s has recurrent keys but no recurrent-layer map", arch)
        return ModelLayout()

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
    for tensor in (t for r in readers for t in r.tensors):
        name = str(tensor.name)
        nbytes = int(tensor.n_bytes)
        match = _BLOCK_RE.match(name)
        if match:
            index = int(match.group(1))
            tail = match.group(2)
            # Shared experts (ffn_*_shexp) and routers (ffn_gate_inp*) run on every token: dense-FFN bandwidth for a
            # rounding error of size. Not spillable.
            spillable = _MOE_EXPERT_RE.match(tail) or (not is_moe and _DENSE_FFN_RE.match(tail))
            if spillable:
                spill[index] = spill.get(index, 0) + nbytes
            else:
                resident[index] = resident.get(index, 0) + nbytes
            cls = classify_tail(tail, is_moe)
            if cls is not None:
                bucket = per_class.setdefault(index, {})
                bucket[cls] = bucket.get(cls, 0) + nbytes
                # The rung's quant TYPE, from the largest tensor in the class so a
                # stray F32 bias cannot outvote the weight matrix that dominates.
                tb = per_class_type.setdefault(index, {})
                if nbytes > tb.get(cls, (0, ""))[0]:
                    tname = getattr(tensor, "tensor_type", None)
                    tb[cls] = (nbytes, getattr(tname, "name", "") or "")
            continue
        if name.startswith("per_layer_token_embd"):
            # gemma3/gemma4 per-layer embeddings.
            per_layer_embd += nbytes
        elif "token_embd" in name:
            token_embd += nbytes
        elif name == "output.weight":
            lm_head += nbytes
        else:
            other_resident += nbytes

    if not spill and not resident:
        return ModelLayout()

    # Tied embeddings duplicate the vocabulary matrix, they do not SAVE it. With no output.weight llama.cpp re-creates
    # the output tensor from token_embd as TENSOR_DUPLICATED (models/llama.cpp:41-45, models/qwen3.cpp:22-25,
    # models/gemma3.cpp:43-47, and ~60 more) and routes a duplicated TOKEN_EMBD through the OUTPUT buffer list
    # (llama-model-loader.cpp:1113-1114). dev_input is CPU-pinned while dev_output follows the layer split
    # (llama-model.cpp:1465, 1474), so the buffer-type contexts differ, the same-context reuse check misses
    # (llama-model-loader.cpp:1309-1314), and ggml_dup_tensor allocates a second full matrix
    # (llama-model-loader.cpp:1318) that load_all_data fills by name with a real host to device copy (:1542,:1583).
    # Counting the one stored tensor as host-only understates VRAM by a whole vocabulary matrix -- the optimistic
    # direction. Resident, not lm_head: the duplicate keeps the name token_embd.weight, so LM_HEAD_PATTERN cannot match
    # and the lm_head rung would credit a spill that moves nothing.
    if not lm_head and token_embd:
        # ``token_embd`` ONLY, never the per-layer embeddings.
        other_resident += token_embd

    # trailing nextn/MTP blocks are not loaded unless a draft is engaged
    # Trailing nextn/MTP blocks are NOT part of the target model and are not loaded unless a draft is engaged, so an -ot
    # naming them moves nothing: measured, spilling only blk.<nextn> leaves the host buffer at exactly token_embd and
    # the device buffer unchanged. Counting them spillable would credit bytes that can never be freed. Unsloth prices
    # the drafter separately anyway.
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
        has_mla = has_mla,
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
    # same set _MOE_EXPERT_RE selected, or the plan credits itself bytes the emitted pattern never moves
    body = "ffn_(up|gate|down|gate_up)_(exps|chexps)" if layout.is_moe else "ffn_(up|gate|down)"
    if indices is None:
        block = r"\d+"
    else:
        block = "|".join(str(i) for i in sorted(indices))
        block = f"({block})"
    return rf"^blk\.{block}\.{body}\.weight$"


def spill_pattern_for_class(
    layout: ModelLayout,
    cls: SpillClass,
    indices: Optional[list[int]] = None,
) -> str:
    """The anchored ``-ot`` pattern for one rung over ``indices``."""
    body = _CLASS_BODIES[cls][0 if layout.is_moe else 1]
    if not body:
        raise ValueError(
            f"{cls.value} has no tensors on a {'MoE' if layout.is_moe else 'dense'} model"
        )
    if indices is None:
        block = r"\d+"
    else:
        block = "(" + "|".join(str(i) for i in sorted(indices)) + ")"
    return rf"^blk\.{block}\.{body}\.weight$"


LM_HEAD_PATTERN = r"^output\.weight$"
