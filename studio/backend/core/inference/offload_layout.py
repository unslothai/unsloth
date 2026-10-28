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

from core.inference.lookup_tables import is_per_layer_embedding

logger = logging.getLogger(__name__)

# Hybrid Mamba-2 archs storing head_count_kv (and on Bolt expert_used_count) per layer.
_NEMOTRON_H_ARCHS = frozenset({"nemotron_h", "nemotron_h_moe"})
_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

# Sparse experts incl. fused/chunked forms; NOT ffn_routed_*: kimi-k3 reads it every token
_MOE_EXPERT_RE = re.compile(r"^ffn_(up|gate|down|gate_up)_(exps|chexps)\.weight$")
# Dense FFN. Fully activated: every byte crosses the bus every token.
_DENSE_FFN_RE = re.compile(r"^ffn_(up|gate|down)\.weight$")


class SpillClass(Enum):
    """One rung's worth of tensors, in the order the ladder gives them up.

    The ladder's own order: ``ffn_down``, then ``ffn_up`` (or the fused ``gate_up``), then a
    separate ``ffn_gate``. It is NOT the fitter's: ``common/fit.cpp`` names its per-layer fractions
    for what STAYS resident, so ``LAYER_FRACTION_GATE`` moves ``ffn_down`` only and ``_UP`` moves
    down plus gate, and ``_fit_boundary_overflow`` models the fallback arm with that sequence.
    Measured, up and gate spill within 1.7% of each other, so the two rarely differ.
    """

    FFN_DOWN = "ffn_down"
    FFN_UP = "ffn_up"
    FFN_GATE = "ffn_gate"
    DENSE_FFN = "dense_ffn"
    ATTENTION = "attention"


FFN_SPILL_CLASSES = (SpillClass.FFN_DOWN, SpillClass.FFN_UP, SpillClass.FFN_GATE)
BLOCK_SPILL_CLASSES = FFN_SPILL_CLASSES + (SpillClass.DENSE_FFN, SpillClass.ATTENTION)

_CLASS_BODIES: dict[SpillClass, tuple[str, str]] = {
    SpillClass.FFN_DOWN: ("ffn_down_(exps|chexps)", "ffn_down"),
    SpillClass.FFN_UP: ("ffn_(up|gate_up)_(exps|chexps)", "ffn_up"),
    SpillClass.FFN_GATE: ("ffn_gate_(exps|chexps)", "ffn_gate"),
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
    resident_bytes: int
    ffn_down_bytes: int = 0
    ffn_up_bytes: int = 0
    ffn_gate_bytes: int = 0
    dense_ffn_bytes: int = 0
    attn_bytes: int = 0
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
    # llama.cpp pins dev_input to CPU: never VRAM, but charged to host RAM
    token_embd_bytes: int = 0
    per_layer_embd_bytes: int = 0
    # Every tensor's bytes, excluded blocks included; the file size less its metadata.
    tensor_bytes: int = 0
    # output_norm and friends: GPU-resident, too small to be worth spilling.
    other_resident_bytes: int = 0
    # Attention cache for ONE token at f16, across the attention layers only.
    kv_bytes_per_token_f16: int = 0
    # Recurrent state: context independent and follows the layer.
    recurrent_bytes: int = 0
    n_ctx_train: int = 0
    is_moe: bool = False
    # Offloaded MoE experts move only expert_used/expert_count per token; a dense FFN moves all of it.
    n_expert: int = 0
    n_expert_used: int = 0
    # Trailing nextn/MTP blocks dropped; an unbounded ^blk\.\d+\. pattern would match them
    has_excluded_blocks: bool = False
    # Dropped block bytes: --spec-type draft-mtp loads them on the target, GPU first
    excluded_block_bytes: int = 0
    # Sliding-window attention: some layers keep a window-sized cache, some the full context
    # (llama-kv-cache-iswa.cpp:69-104). Not visible in n_attention_layers; multi-device abstains.
    has_swa: bool = False
    # MLA: one compressed K-only latent per token. Keyed on key_length_mla AND value_length_mla
    # (llama_hparams::is_mla); kv_lora_rank alone (DeepSeek-R1/V3 GGUFs) still gets full K+V.
    has_mla: bool = False
    # Per-row KV head count (0 = no attention cache); empty when uniform or unknown. Fitter-only.
    kv_heads_by_layer: tuple[int, ...] = ()
    recurrent_rows: tuple[int, ...] = ()
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


def layout_from_gguf(
    path: str,
    *,
    all_shards: bool = False,
    build: Optional[int] = None,
) -> ModelLayout:
    """Read ``path`` into a :class:`ModelLayout`.

    Returns an incomplete layout (``complete = False``) rather than raising when
    anything required is missing, so a surprising GGUF makes the planner abstain
    instead of failing a load that llama.cpp would have handled.
    ``all_shards`` reads every sibling shard; all of them must be present. ``build`` is the
    llama.cpp build the layout is priced for (see :func:`indexer_cache_widths`).
    """
    try:
        import struct

        import numpy as np
        from gguf import GGUFEndian, GGUFReader, GGUFValueType

        class _LayoutReader(GGUFReader):
            # GGUFReader makes a numpy view per array element, seconds for a vocabulary; no string array is read here.
            def _get_field_parts(self, orig_offs, raw_type):
                if raw_type != GGUFValueType.ARRAY:
                    return super()._get_field_parts(orig_offs, raw_type)
                raw_itype = self._get(orig_offs, np.uint32)
                alen = self._get(orig_offs + 4, np.uint64)
                itype = GGUFValueType(raw_itype[0])
                types = [GGUFValueType.ARRAY, itype]
                nptype = self.gguf_scalar_to_np.get(itype)
                if nptype is not None:
                    items = self._get(orig_offs + 12, nptype, alen[0])
                    return 12 + int(items.nbytes), [raw_itype, alen, items], [2], types
                if itype != GGUFValueType.STRING:
                    return super()._get_field_parts(orig_offs, raw_type)
                length = struct.Struct("<Q" if self.endianess == GGUFEndian.LITTLE else ">Q")
                offs = orig_offs + 12
                for _ in range(int(alen[0])):
                    offs += 8 + length.unpack_from(self.data, offs)[0]
                return offs - orig_offs, [raw_itype, alen], [], types

        readers = [_LayoutReader(path)]
        if all_shards and int(_field(readers[0], "split.count") or 0) > 1:
            shards = split_shard_paths(path)
            if not shards or not all(os.path.isfile(p) for p in shards):
                logger.debug("offload layout: split %s is missing a shard", path)
                return ModelLayout()
            readers = [_LayoutReader(p) for p in shards]
    except Exception as exc:
        logger.debug("offload layout: cannot read %s (%s)", path, exc)
        return ModelLayout()

    try:
        return _layout_from_readers(readers, build = build)
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


# llama.cpp's full_attn_interval defaults (qwen3next.cpp:24, qwen35.cpp:23, qwen35moe.cpp:26,
# minimax-01.cpp:13). Unlisted archs abstain.
_FULL_ATTENTION_INTERVAL_DEFAULT: dict[str, int] = {
    "qwen3next": 4,
    "qwen35": 4,
    "qwen35moe": 4,
    "minimax-01": 8,
}


# Archs with a lightning-indexer MQA KV cache (llama_memory_hybrid_idx, indexer_head_size > 0).
INDEXER_CACHE_ARCHS: frozenset[str] = frozenset({"qwen4exp"})

# #28330 (b10889) allocates it K-only; #29751 (b11330) doubles a pooled qwen4exp row
# when any attention.compress_ratios entry is non-zero.
INDEXER_K_ONLY_BUILD = 10889
INDEXER_POOLED_ROW_BUILD = 11330


def indexer_cache_widths(
    key_len: int,
    val_len: int,
    compress_ratios = None,
    build: Optional[int] = None,
) -> tuple[int, int]:
    """``(k_width, v_width)`` elements per cell of the indexer cache for llama.cpp ``build``.

    Shared by the layout reader and the backend estimator so the two price one cache. An unknown
    build keeps the conservative K + V charge of the builds before #28330.
    """
    key_len = max(0, int(key_len or 0))
    if build is None or build < INDEXER_K_ONLY_BUILD:
        return key_len, max(0, int(val_len or 0))
    if build < INDEXER_POOLED_ROW_BUILD:
        return key_len, 0
    ratios = compress_ratios if isinstance(compress_ratios, (list, tuple)) else (compress_ratios,)
    pooled = any(int(r or 0) > 0 for r in ratios)
    return key_len * (2 if pooled else 1), 0


# KDA-state archs (llama-hparams.cpp n_embd_r/n_embd_s). Named, not derived: unlisted KDA
# families are unmeasured, and glm5next's indexer cache is unpriced, so they abstain.
_KDA_STATE_ARCHS: frozenset[str] = frozenset({"kimi-k3"})


# LLAMA_MAX_LAYERS (src/llama-hparams.h): the child refuses more, and readers allocate per layer.
LLAMA_MAX_LAYERS = 512


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

    The one derivation both readers use, so the layout's cache product and the estimator's path 2
    cannot disagree about the same file. llama.cpp resolves it in three steps
    (models/qwen3next.cpp:load_arch_hparams, and the identical block in qwen35, qwen35moe and
    minimax-01): an explicit per-layer mask, then ``full_attention_interval``, then the
    ARCHITECTURE's built-in default.

    A per-layer ``attention.head_count_kv`` list with zeros beats all three: those rows hold no
    attention cache at all, and llama.cpp reads is_recr straight off them
    (models/nemotron-h.cpp:load_arch_hparams, models/falcon-h1.cpp). ``known`` is False only when
    nothing above said anything, the caller's cue to abstain rather than call every row attention.
    """
    if n_layers <= 0 or n_layers > LLAMA_MAX_LAYERS:
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
            # Floor: row il is attention iff (il + 1) % fai == 0, so 30 layers at 4 gives 7.
            n_recurrent = max(0, n_layers - n_layers // fai)
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


def layer_kv_heads(
    arch: str,
    n_layers: int,
    *,
    recurrent_layers = None,
    n_kv_head = None,
    full_attention_interval: int = 0,
) -> tuple[int, ...]:
    """Each row's KV head count, 0 where no attention cache lives, in hybrid_layer_split's order.

    A per-layer list with zeros names the cache-free rows itself; otherwise the explicit recurrent
    mask, then ``full_attention_interval`` (row il is attention iff (il + 1) % fai == 0,
    models/qwen35.cpp), then the architecture's default. Empty when every row is the same.
    """
    if n_layers <= 0 or n_layers > LLAMA_MAX_LAYERS:
        return ()
    if isinstance(n_kv_head, (list, tuple)):
        heads = [int(h) for h in n_kv_head]
        if not heads:
            return ()
        rows = [max(0, heads[i] if i < len(heads) else heads[-1]) for i in range(n_layers)]
    else:
        try:
            rows = [max(0, int(n_kv_head))] * n_layers
        except (TypeError, ValueError):
            return ()
    if not 0 < sum(1 for h in rows if h > 0) < n_layers:
        if isinstance(recurrent_layers, (list, tuple)) and len(recurrent_layers) >= n_layers:
            rows = [0 if flag else h for flag, h in zip(list(recurrent_layers)[:n_layers], rows)]
        else:
            fai = int(full_attention_interval or 0)
            if fai <= 0:
                fai = _FULL_ATTENTION_INTERVAL_DEFAULT.get(arch, 0)
            if fai > 0:
                rows = [h if (i + 1) % fai == 0 else 0 for i, h in enumerate(rows)]
    return tuple(rows) if len(set(rows)) > 1 else ()


def _layout_from_reader(reader, *, build: Optional[int] = None) -> ModelLayout:
    return _layout_from_readers([reader], build = build)


def _layout_from_readers(readers, *, build: Optional[int] = None) -> ModelLayout:
    """One reader per shard, the first carrying the metadata."""
    reader = readers[0]
    # Split GGUF: GGUFReader maps one shard, so a partial read undercounts and can OOM under --fit off; abstain unless every shard is given
    if (int(_field(reader, "split.count") or 0) or 1) != len(readers):
        return ModelLayout()

    arch = str(_field(reader, "general.architecture") or "")
    if not arch:
        return ModelLayout()

    blocks_total = _field(reader, f"{arch}.block_count")
    if not blocks_total:
        return ModelLayout()
    blocks_total = int(blocks_total)
    if blocks_total > LLAMA_MAX_LAYERS:
        return ModelLayout()

    # llama.cpp keeps embedded MTP blocks out of the target context and prices their cache separately, so the attention
    # count must not include them.
    nextn = int(_field(reader, f"{arch}.nextn_predict_layers") or 0)
    n_layers = max(0, blocks_total - nextn)
    # A negative nextn would grow n_layers past the bound checked above.
    if n_layers > LLAMA_MAX_LAYERS:
        return ModelLayout()

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

    nemotron_h = arch in _NEMOTRON_H_ARCHS
    n_attention, n_recurrent, recurrent_known = hybrid_layer_split(
        arch,
        n_layers,
        recurrent_layers = _field(reader, f"{arch}.attention.recurrent_layers"),
        n_kv_head = n_kv_head,
        full_attention_interval = int(_field(reader, f"{arch}.full_attention_interval") or 0),
        feed_forward_length = _field(reader, f"{arch}.feed_forward_length"),
    )
    kv_heads_by_layer = layer_kv_heads(
        arch,
        n_layers,
        recurrent_layers = _field(reader, f"{arch}.attention.recurrent_layers"),
        n_kv_head = n_kv_head,
        full_attention_interval = int(_field(reader, f"{arch}.full_attention_interval") or 0),
    )
    if sum(1 for h in kv_heads_by_layer if h > 0) != int(n_attention):
        kv_heads_by_layer = ()
    recurrent_rows = tuple(i for i, h in enumerate(kv_heads_by_layer) if h <= 0)
    ffn_lengths = _field(reader, f"{arch}.feed_forward_length")
    if arch in _RECURRENT_NEEDS_ZERO_FFN and isinstance(ffn_lengths, (list, tuple)) and ffn_lengths:
        ffs = [int(f) for f in ffn_lengths]
        recurrent_rows = tuple(i for i in recurrent_rows if ffs[min(i, len(ffs) - 1)] <= 0)
    if len(recurrent_rows) != int(n_recurrent):
        recurrent_rows = ()

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

    indexer_key_len = (
        int(_field(reader, f"{arch}.attention.indexer.key_length") or 0)
        if arch in INDEXER_CACHE_ARCHS
        else 0
    )
    if indexer_key_len:
        k_width, v_width = indexer_cache_widths(
            indexer_key_len,
            int(val_len),
            _field(reader, f"{arch}.attention.compress_ratios"),
            build,
        )
        kv_per_token += int(n_attention) * (k_width + v_width) * 2

    # Full context on every layer is safe for TOTAL but cannot show which layers hold the big caches.
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
    # LFM2 short conv: (l_cache - 1) rows of n_embd, no ssm state (llama-hparams.cpp:n_embd_r).
    shortconv_l_cache = int(_field(reader, f"{arch}.shortconv.l_cache") or 0)
    recurrent = 0
    if n_recurrent and d_inner and d_state and d_conv:
        n_embd_r = max(0, d_conv - 1) * (d_inner + 2 * n_group * d_state)
        n_embd_s = d_state * d_inner
        recurrent = n_recurrent * (n_embd_r + n_embd_s) * 4
    elif n_recurrent and kda_head_dim and n_head:
        # KDA has no ssm.inner_size; size from head count and kda.head_dim (llama-hparams.cpp),
        # else Kimi-K3 drops 443 MiB/slot.
        d_inner_kda = int(n_head) * kda_head_dim
        n_embd_r = 3 * max(0, (d_conv or 4) - 1) * d_inner_kda
        n_embd_s = kda_head_dim * kda_head_dim * int(n_head)
        recurrent = n_recurrent * (n_embd_r + n_embd_s) * 4
    elif n_recurrent and shortconv_l_cache > 1 and n_embd:
        recurrent = n_recurrent * int(n_embd) * (shortconv_l_cache - 1) * 4

    # PLE conv history is its own recurrent-cache row (llama-hparams.cpp:ple_conv_state).
    ple_layers = _field(reader, f"{arch}.ple.layers")
    if recurrent and isinstance(ple_layers, (list, tuple)) and ple_layers:
        ple_conv_state = (
            max(0, int(_field(reader, f"{arch}.ple.conv_kernel") or 0) - 1)
            * int(_field(reader, f"{arch}.ple.ngram_size") or 0)
            * int(_field(reader, f"{arch}.hyper_connection.count") or 0)
            * int(n_embd or 0)
        )
        recurrent += len(ple_layers) * ple_conv_state * 4

    if not recurrent_known and (
        (d_inner and d_state and d_conv) or kda_head_dim or shortconv_l_cache > 1
    ):
        logger.debug("offload layout: %s has recurrent keys but no recurrent-layer map", arch)
        return ModelLayout()
    if n_recurrent and not recurrent:
        logger.debug(
            "offload layout: %s has %d recurrent rows of unknown state size", arch, n_recurrent
        )
        return ModelLayout()

    n_expert = int(_field(reader, f"{arch}.expert_count") or 0)
    n_expert_used = _field(reader, f"{arch}.expert_used_count") or 0
    if nemotron_h and isinstance(n_expert_used, (list, tuple)):
        n_expert_used = max(
            (int(n) for n in n_expert_used), default = 0
        )  # per layer on Nemotron 3.5 Bolt
    n_expert_used = int(n_expert_used)
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
                # Quant type from the largest tensor, so an F32 bias cannot outvote the weight.
                tb = per_class_type.setdefault(index, {})
                if nbytes > tb.get(cls, (0, ""))[0]:
                    tname = getattr(tensor, "tensor_type", None)
                    tb[cls] = (nbytes, getattr(tname, "name", "") or "")
            continue
        # Before the token_embd substring below, which would otherwise swallow the per-layer table.
        if is_per_layer_embedding(name):
            per_layer_embd += nbytes
        # token_embd_norm is a repeating-layer tensor, not a host-pinned input embedding.
        elif "token_embd" in name and not name.startswith("token_embd_norm"):
            token_embd += nbytes
        elif name == "output.weight":
            lm_head += nbytes
        else:
            other_resident += nbytes

    if not spill and not resident:
        return ModelLayout()

    # Tied embeddings: llama.cpp allocates a second full matrix on the output device; count it resident, else the lm_head rung credits a spill that moves nothing
    if not lm_head and token_embd:
        other_resident += token_embd

    # Trailing nextn/MTP blocks are not loaded without a draft: spilling them frees nothing
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
        kv_heads_by_layer = kv_heads_by_layer,
        recurrent_rows = recurrent_rows,
        blocks = blocks,
        lm_head_bytes = lm_head,
        token_embd_bytes = token_embd,
        per_layer_embd_bytes = per_layer_embd,
        tensor_bytes = sum(int(t.n_bytes) for r in readers for t in r.tensors),
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
    """Anchored: llama.cpp uses regex_search, so an unanchored pattern also moves attention tensors."""
    # Must match _MOE_EXPERT_RE, or the plan credits bytes the pattern never moves
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
