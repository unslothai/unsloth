# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF tensor layout by placement bucket; file IO and GGUF keys live here, not in offload_planner."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
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


@dataclass(frozen = True)
class BlockLayout:
    """One transformer block, split into what may and may not be spilled."""

    index: int
    # ffn_*_exps (MoE) or plain ffn_* (dense). Safe to push to host RAM.
    spillable_bytes: int
    # attention, norms, routers, shared experts, ssm: on the critical path every token, or the KV cache hangs off them.
    resident_bytes: int


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
    # Mamba conv/SSM state; context independent, and follows the layer, which -ot never moves
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
    # SWA: per-layer cache sizes differ, so a multi-device split abstains
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
    """Returns an incomplete layout rather than raising, so the planner abstains; all_shards reads every sibling shard, and all must be present"""
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
        return _layout_from_readers(readers)
    except Exception as exc:
        logger.debug("offload layout: cannot interpret %s (%s)", path, exc)
        return ModelLayout()


def _layout_from_reader(reader) -> ModelLayout:
    return _layout_from_readers([reader])


def _layout_from_readers(readers) -> ModelLayout:
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

    # llama.cpp keeps embedded MTP blocks out of the target context and prices their cache separately, so the attention
    # count must not include them.
    nextn = int(_field(reader, f"{arch}.nextn_predict_layers") or 0)
    n_layers = max(0, blocks_total - nextn)

    # Hybrid: only 1 in full_attention_interval layers carries a KV cache, the rest are recurrent. Absent (or 0) means
    # every layer is attention.
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

    nemotron_h = arch in _NEMOTRON_H_ARCHS
    if nemotron_h and isinstance(n_kv_head, (list, tuple)):
        # Per-layer head_count_kv: 0 marks a layer without an attention cache. The recurrent (Mamba) layers are the ones
        # with neither attention nor an FFN; the rest are dense or MoE FFN blocks.
        heads = [int(h) for h in list(n_kv_head)[:n_layers]]
        ffn = _field(reader, f"{arch}.feed_forward_length")
        ffn = (
            [int(f) for f in list(ffn)[:n_layers]]
            if isinstance(ffn, (list, tuple))
            else [0] * len(heads)
        )
        n_attention = sum(1 for h in heads if h > 0)
        n_recurrent = sum(1 for h, f in zip(heads, ffn) if h == 0 and f == 0)
        kv_per_token = sum(heads) * (int(key_len) + int(val_len)) * 2
    else:
        kv_per_token = int(n_attention) * int(n_kv_head) * (int(key_len) + int(val_len)) * 2

    # Full context on every layer is safe for TOTAL but cannot show which layers hold the big caches.
    has_swa = bool(_field(reader, f"{arch}.attention.sliding_window") or 0)

    # Mamba conv + SSM state, one f32 copy per sequence. Mirrors llama.cpp's own sizing; zero when the model has no
    # recurrent layers.
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
    n_expert_used = _field(reader, f"{arch}.expert_used_count") or 0
    if nemotron_h and isinstance(n_expert_used, (list, tuple)):
        n_expert_used = max(
            (int(n) for n in n_expert_used), default = 0
        )  # per layer on Nemotron 3.5 Bolt
    n_expert_used = int(n_expert_used)
    is_moe = bool(n_expert)

    spill: dict[int, int] = {}
    resident: dict[int, int] = {}
    lm_head = 0
    token_embd = 0
    per_layer_embd = 0
    other_resident = 0

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


LM_HEAD_PATTERN = r"^output\.weight$"
