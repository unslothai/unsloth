# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Serving one GGUF across N machines over llama.cpp RPC: any number of nodes, any mix of NVIDIA,
AMD and Apple Silicon, any OS that runs ``ggml-rpc-server``.

Opt-in only. With no cluster configured nothing here runs and ``spark_serving`` keeps its two-Spark
path exactly as it was. Configure with ``UNSLOTH_CLUSTER_NODES`` (comma list of
``[user@]host[:port]``) or ``<studio home>/cluster.json``::

    {"nodes": [{"host": "10.0.0.2", "rpc_port": 50052, "ssh_user": "me", "managed": true}]}

``managed`` nodes get their ``ggml-rpc-server`` started over ssh; the others are expected to run
``ggml-rpc-server -H <ip> -p <port>`` themselves (the way a Mac or a Windows box joins).

Placement reads free memory per remote device from ``llama-server --rpc ... --list-devices``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import struct
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ENV_NODES = "UNSLOTH_CLUSTER_NODES"
ENV_MANAGED = "UNSLOTH_CLUSTER_MANAGED"  # "1": start every node's rpc-server over ssh
ENV_TOPOLOGY = "UNSLOTH_CLUSTER_TOPOLOGY"  # single | layer_split
CONFIG_NAME = "cluster.json"
RPC_PORT_DEFAULT = 50052
_GIB = 1024**3
_MIB = 1024**2
# Compute buffers, the KV padding and the context llama.cpp keeps per device.
DEVICE_RESERVE_MIN_BYTES = 1 * _GIB
DEVICE_RESERVE_FRACTION = 0.08
# Whole layers: a device can be handed up to a layer more than its share of the weights.
LAYER_ROUNDING_MARGIN = 0.05
LIST_DEVICES_TIMEOUT_S = 60.0
WORKER_START_TIMEOUT_S = 60.0
REACH_TIMEOUT_S = 3.0
# KV share of the weights reserved for a Hub GGUF whose header is not downloaded yet.
_MMPROJ_VRAM_SAFETY = 1.4  # LlamaCppBackend._MMPROJ_VRAM_SAFETY

_MMPROJ_DEVICE_FLAGS = frozenset({"--mmproj-device", "-mmdev"})
_ADAPTER_FLAGS = frozenset(
    {"--lora", "--lora-scaled", "--control-vector", "--control-vector-scaled"}
)
_PLACEMENT_FLAGS = frozenset(
    {
        "--device",
        "-dev",
        "--split-mode",
        "-sm",
        "--tensor-split",
        "-ts",
        "--main-gpu",
        "-mg",
        "--gpu-layers",
        "--n-gpu-layers",
        "-ngl",
        "--override-tensor",
        "-ot",
        "--cpu-moe",
        "-cmoe",
        "--n-cpu-moe",
        "-ncmoe",
        "--n-cpu-ffn",
        "-ncffn",
        "--moe-cache-mib",  # an expert cache on the GPUs, beside the planned weights
        "-otd",
        "--override-tensor-draft",
        "--spec-draft-override-tensor",
    }
)
_PLACEMENT_ENV = (
    "LLAMA_ARG_DEVICE",
    "LLAMA_ARG_SPLIT_MODE",
    "LLAMA_ARG_TENSOR_SPLIT",
    "LLAMA_ARG_MAIN_GPU",
    "LLAMA_ARG_N_GPU_LAYERS",
    "LLAMA_ARG_OVERRIDE_TENSOR",
    "LLAMA_ARG_CPU_MOE",
    "LLAMA_ARG_N_CPU_MOE",
    "LLAMA_ARG_N_CPU_FFN",
    "LLAMA_ARG_MOE_CACHE_MIB",
    "LLAMA_ARG_OVERRIDE_TENSOR_DRAFT",
)


def _gguf_block_count(path: Optional[str], ss: Any) -> Optional[int]:
    if not path:
        return None
    try:
        meta = ss._cluster().gguf_metadata(path)
        arch = meta.get("general.architecture")
        count = meta.get(f"{arch}.block_count") if isinstance(arch, str) else None
        return int(count) if isinstance(count, int) and count > 0 else None
    except Exception:
        return None


def _remote_projector_bytes(model_path: str, hf_token: Optional[str]) -> Optional[int]:
    """The largest projector a not-yet-downloaded Hub repo ships: 0 when none, None when the
    listing could not be read."""
    if not model_path or "/" not in model_path or os.path.isabs(model_path):
        return 0
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_path, files_metadata = True, token = hf_token)
        sizes = [
            int(s.size or 0)
            for s in info.siblings or []
            if "mmproj" in s.rfilename.lower() and s.rfilename.lower().endswith(".gguf")
        ]
        return max(sizes, default = 0)
    except Exception:
        return None


def _quant_dir_projector(model_file: Optional[str], disable_vision: bool, ss: Any) -> Optional[str]:
    """The projector one level up from a weight in a quant subdirectory (snapshot/UD-Q4_K_XL/),
    which the loader finds and the adjacent-file scan does not."""
    if not model_file:
        return None
    try:
        from utils.models.model_config import (
            _local_gguf_companion_search_root,
            detect_mmproj_file,
        )

        root = _local_gguf_companion_search_root(model_file, model_file)
        if os.path.abspath(root) == os.path.dirname(os.path.abspath(model_file)):
            return None
        found = detect_mmproj_file(model_file, root)
    except Exception:
        return None
    if not found or (disable_vision and ss._projector_accepts_image(str(found))):
        return None
    return str(found)


_GGUF_INT_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    7: "<?",
    10: "<Q",
    11: "<q",
}
_GGUF_FIXED_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_KV_KEY_SUFFIXES = (
    ".block_count",
    ".attention.head_count",
    ".attention.head_count_kv",
    ".embedding_length",
    ".attention.key_length",
    ".attention.value_length",
    ".nextn_predict_layers",
    ".context_length",
    ".full_attention_interval",
    ".ssm.conv_kernel",
    ".ssm.inner_size",
    ".ssm.state_size",
    ".ssm.group_count",
)


def _read_gguf_header(
    f: Any,
    *,
    tensors: bool = False,
    kv_end: Optional[List[int]] = None,
) -> Tuple[Dict[str, Any], Dict[int, int]]:
    """From a GGUF header stream: general.architecture and the integer keys estimate_kv_bytes
    reads (a per-layer array keeps its largest value), and with ``tensors`` the weight bytes of
    each block in this file."""

    def read(n: int) -> bytes:
        data = f.read(n)
        if len(data) < n:
            raise EOFError
        return data

    magic, _version, n_tensors, count = struct.unpack("<IIQQ", read(24))
    if magic != 0x46554747:
        return {}, {}
    out: Dict[str, Any] = {}
    for _ in range(count):
        key = read(struct.unpack("<Q", read(8))[0]).decode("utf-8", "replace")
        vtype = struct.unpack("<I", read(4))[0]
        if not tensors and key.startswith("tokenizer.") and "general.architecture" in out:
            return out, {}  # the model keys precede the vocabulary
        wanted = key.endswith(_KV_KEY_SUFFIXES)
        if vtype == 8:
            text = read(struct.unpack("<Q", read(8))[0])
            if key == "general.architecture":
                out[key] = text.decode("utf-8", "replace")
        elif vtype == 9:
            atype, n = struct.unpack("<IQ", read(12))
            if atype == 8:
                for _ in range(n):
                    f.seek(struct.unpack("<Q", read(8))[0], 1)
            elif wanted and atype in _GGUF_INT_FORMATS:
                fmt = _GGUF_INT_FORMATS[atype]
                width = struct.calcsize(fmt)
                values = [int(struct.unpack(fmt, read(width))[0]) for _ in range(n)]
                out[key] = max(values, default = 0)
                if key.endswith(".head_count_kv") and 0 in values:
                    out[f"{key}.zeros"] = values.count(0)  # recurrent layers of a hybrid
            elif atype in _GGUF_FIXED_SIZES:
                f.seek(_GGUF_FIXED_SIZES[atype] * n, 1)
            else:
                return out, {}
        elif wanted and vtype in _GGUF_INT_FORMATS:
            fmt = _GGUF_INT_FORMATS[vtype]
            out[key] = int(struct.unpack(fmt, read(struct.calcsize(fmt)))[0])
        elif vtype in _GGUF_FIXED_SIZES:
            f.seek(_GGUF_FIXED_SIZES[vtype], 1)
        else:
            return out, {}
    if kv_end is not None:
        kv_end.append(f.tell())
    blocks: Dict[int, int] = {}
    if not tensors:
        return out, blocks
    try:
        from gguf.constants import GGML_QUANT_SIZES
    except ImportError:
        return out, {}
    for _ in range(n_tensors):
        name = read(struct.unpack("<Q", read(8))[0]).decode("utf-8", "replace")
        n_dims = struct.unpack("<I", read(4))[0]
        elements = 1
        for _ in range(n_dims):
            elements *= struct.unpack("<Q", read(8))[0]
        ttype = struct.unpack("<I", read(4))[0]
        read(8)  # data offset
        match = re.match(r"blk\.(\d+)\.", name)
        # The output matrix (or the token embedding tied embeddings copy there) is one more
        # indivisible allocation, kept on the last device; -2 / -1 below.
        index = int(match.group(1)) if match else _OUTPUT_KEYS.get(name.split(".", 1)[0])
        if index is None:
            continue
        if ttype not in GGML_QUANT_SIZES:
            return out, {}
        block_size, type_size = GGML_QUANT_SIZES[ttype]
        blocks[index] = blocks.get(index, 0) + elements // block_size * type_size
    return out, blocks


_OUTPUT_KEYS = {"output": -1, "token_embd": -2}
# The KV section of the last uncached header read, for the loader's own metadata reader.
_REMOTE_HEADER_KV: Dict[Tuple[str, str], bytes] = {}


def _ctx_compute_bytes(
    path: Optional[str],
    header_kv: Optional[bytes],
    ctx: int,
    ubatch: int,
    cache_type: Optional[str],
    flash_attn: bool,
    n_parallel: int,
    what: str = "",
    kv_unified: bool = True,
    swa_full: bool = False,
) -> Optional[int]:
    """The loader's per-device context-dependent compute scratch on a layer split (with
    ``what="checkpoint"`` one slot's context checkpoint, with ``"recurrent"`` the per-slot recurrent
    state of every slot), from its own estimator over this GGUF's
    header; None when the header cannot be read."""
    if ctx <= 0 and what != "recurrent":
        return 0
    import tempfile

    from core.inference.llama_cpp import LlamaCppBackend

    scratch = None
    try:
        if not path and header_kv:
            scratch = tempfile.NamedTemporaryFile(suffix = ".gguf", delete = False)
            scratch.write(header_kv)
            scratch.close()
            path = scratch.name
        if not path:
            return None
        est = LlamaCppBackend.__new__(LlamaCppBackend)
        est._model_identifier = None
        est._read_gguf_metadata(path)
        if not getattr(est, "_embedding_length", None):
            return None
        if what == "checkpoint":
            return int(
                est._ctx_checkpoint_bytes(cache_type, swa_full = swa_full, flash_attn = flash_attn)
            )
        if what == "recurrent":
            return int(est._rollback_state_bytes(n_parallel))
        if what == "kv_peak_ppm":
            # The largest layer's share of the KV, in millionths; 0 when the layers are uniform.
            weights = est._kv_layer_weights(
                ctx,
                swa_full = swa_full,
                n_parallel = n_parallel,
                kv_unified = kv_unified,
                n_ubatch = ubatch,
                flash_attn = flash_attn,
                cache_type_kv = cache_type,
            )
            total = sum(weights)
            return int(1_000_000 * max(weights) / total) if total else 0
        if what == "kv":
            # Architecture-aware: compact sliding-window layers, one MLA latent, attention-only
            # hybrids. 0 when the metadata does not allow it.
            return int(
                est._estimate_kv_cache_bytes(
                    ctx,
                    cache_type,
                    swa_full = swa_full,
                    n_parallel = n_parallel,
                    kv_unified = kv_unified,
                    n_ubatch = ubatch,
                    flash_attn = flash_attn,
                )
            )
        return int(
            est._compute_buffer_ctx_bytes(
                ctx,
                ubatch,
                cache_type,
                layer_split = True,
                flash_attn = flash_attn,
                n_parallel = n_parallel,
            )
        )
    except Exception:
        return None
    finally:
        if scratch is not None:
            try:
                os.unlink(scratch.name)
            except OSError:
                pass


def _shares_host_memory(device: Device) -> bool:
    """Whether a local accelerator's memory is the host RAM pool (Apple Silicon, APUs, DGX
    Spark): Metal always; otherwise a device holding at least half the host's RAM, which leans
    to charging host buffers on a large discrete card in a small host, the safe side."""
    if device.name.upper().startswith("MTL"):
        return True
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        host = (LlamaCppBackend._host_memory_capacity_mib() or 0) * 1024 * 1024
    except Exception:
        return True
    return not host or device.total_bytes >= host / 2


def _kv_unified_supported(binary: Optional[str]) -> bool:
    """The loader's probe for --kv-unified; a failed probe reads as absent there too."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return bool(LlamaCppBackend.probe_server_capabilities(binary).get("supports_kv_unified"))
    except Exception:
        return False


def _effective_checkpoints(
    binary: Optional[str],
    extras: Sequence[str],
    requested: Optional[int],
    per_checkpoint: int,
    n_parallel: int,
) -> int:
    """effective_ctx_checkpoints_for_caps, as the loader resolves it; llama.cpp's default of 32
    per slot when that cannot be asked."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend, effective_ctx_checkpoints_for_caps

        caps = LlamaCppBackend.probe_server_capabilities(binary)
        host = (LlamaCppBackend._host_memory_capacity_mib() or 0) * 1024 * 1024
        return max(
            0,
            int(
                effective_ctx_checkpoints_for_caps(
                    caps,
                    extras,
                    requested,
                    per_checkpoint_bytes = per_checkpoint,
                    n_parallel = n_parallel,
                    total_host_bytes = host or None,
                    env = os.environ,
                )
            ),
        )
    except Exception:
        from core.inference.llama_cpp import _env_ctx_checkpoints_override
        from core.inference.llama_server_args import parse_ctx_checkpoints_override

        for count in (
            parse_ctx_checkpoints_override(extras),
            requested,
            _env_ctx_checkpoints_override(os.environ),
        ):
            if count is not None:
                return max(0, int(count))
        return 32


def _ssm_state_bytes(vals: Dict[str, Any], n_parallel: int) -> int:
    """f32 recurrent state per slot, as llama.cpp allocates for Mamba / gated-delta layers, over
    the recurrent layers _mamba_recurrent_state_bytes counts (every block without a layout)."""
    arch = vals.get("general.architecture")
    inner = vals.get(f"{arch}.ssm.inner_size") or 0
    state = vals.get(f"{arch}.ssm.state_size") or 0
    if not (arch and inner and state):
        return 0
    conv = max(0, (vals.get(f"{arch}.ssm.conv_kernel") or 0) - 1)
    groups = vals.get(f"{arch}.ssm.group_count") or 0
    per_layer = conv * (inner + 2 * groups * state) + state * inner
    return int(per_layer * 4 * _recurrent_layers(vals) * max(1, int(n_parallel)))


def _recurrent_layers(vals: Dict[str, Any]) -> int:
    """Recurrent blocks as _mamba_recurrent_state_bytes counts them: all but every interval-th
    (attention) one, else the zero-KV-head ones, else every block; embedded NextN ones excluded."""
    arch = vals.get("general.architecture")
    n_layers = max(
        0,
        int(vals.get(f"{arch}.block_count") or 0)
        - int(vals.get(f"{arch}.nextn_predict_layers") or 0),
    )
    interval = vals.get(f"{arch}.full_attention_interval")
    zeros = vals.get(f"{arch}.attention.head_count_kv.zeros")
    if interval is not None:
        interval = int(interval or 0)
        return n_layers - (-(-n_layers // interval) if interval > 0 else n_layers)
    return min(n_layers, int(zeros)) if zeros else n_layers


def _gguf_kv_scalars(f: Any) -> Dict[str, Any]:
    return _read_gguf_header(f)[0]


def _kv_from_scalars(
    vals: Dict[str, Any], ctx: int, cache_type: Optional[str], cache_type_v: Optional[str], ss: Any
) -> Optional[int]:
    """estimate_kv_bytes's formula over header values read by _read_gguf_header."""
    arch = vals.get("general.architecture")
    if not arch or ctx <= 0:
        return None
    n_layer = vals.get(f"{arch}.block_count")
    n_head = vals.get(f"{arch}.attention.head_count")
    n_kv = vals.get(f"{arch}.attention.head_count_kv") or n_head
    n_embd = vals.get(f"{arch}.embedding_length")
    head_dim = vals.get(f"{arch}.attention.key_length")
    if head_dim is None and n_head and n_embd:
        head_dim = n_embd // n_head
    if not (n_layer and n_kv and head_dim):
        return None
    value_dim = vals.get(f"{arch}.attention.value_length") or head_dim
    per_cell = head_dim * ss.kv_bytes_per_elem(cache_type) + value_dim * ss.kv_bytes_per_elem(
        cache_type if cache_type_v is None else cache_type_v
    )
    return int(n_layer * int(ctx) * n_kv * per_cell)


def _local_gguf_header(path: str, ss: Any) -> Tuple[Dict[str, Any], Dict[int, int]]:
    """_read_gguf_header over every shard of a local GGUF, blocks summed across shards."""
    vals: Dict[str, Any] = {}
    blocks: Dict[int, int] = {}
    try:
        for index, shard in enumerate(ss.gguf_shard_paths(path) or [path]):
            with open(shard, "rb") as f:
                shard_vals, shard_blocks = _read_gguf_header(f, tensors = True)
            if index == 0:
                vals = shard_vals
            if not shard_blocks:
                return vals, {}
            for block, size in shard_blocks.items():
                blocks[block] = blocks.get(block, 0) + size
    except Exception:
        return vals, {}
    return vals, blocks


def _remote_gguf_header(
    model_path: str, variant: Optional[str], hf_token: Optional[str], ss: Any
) -> Tuple[Dict[str, Any], Dict[int, int]]:
    """_read_gguf_header for a Hub GGUF not downloaded yet, from its first shard over HTTP
    ranges; empty when it cannot be read."""
    try:
        from huggingface_hub import HfFileSystem
        from utils.models.model_config import list_gguf_variants

        variants, _vision = list_gguf_variants(model_path, hf_token)
        chosen = ss._pick_variant(variants, str(variant).strip().casefold() if variant else "")
        if chosen is None:
            return {}, {}
        fs = HfFileSystem(token = hf_token)
        first = str(chosen.filename)
        shards = [first]
        match = re.search(r"-00001-of-(\d{5})\.gguf$", first, re.IGNORECASE)
        if match:
            total = int(match.group(1))
            stem = first[: match.start()]
            shards = [f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
        vals: Dict[str, Any] = {}
        blocks: Dict[int, int] = {}
        for index, shard in enumerate(shards):
            end: List[int] = []
            with fs.open(f"{model_path}/{shard}", "rb", block_size = 1 << 20) as f:
                shard_vals, shard_blocks = _read_gguf_header(f, tensors = True, kv_end = end)
                if index == 0 and end:
                    f.seek(0)
                    _REMOTE_HEADER_KV[(model_path, str(variant))] = f.read(end[0])
            if index == 0:
                vals = shard_vals
            if not shard_blocks:
                return vals, {}  # a shard without a readable tensor table leaves the max unknown
            for block, size in shard_blocks.items():
                blocks[block] = blocks.get(block, 0) + size
        return vals, blocks
    except Exception:
        return {}, {}


_DRAFTER_FLAGS = frozenset({"--model-draft", "-md", "--spec-draft-model"})
_DRAFTER_ENV = ("LLAMA_ARG_SPEC_DRAFT_MODEL", "LLAMA_ARG_MODEL_DRAFT")  # new, pre-b8955
_HF_DRAFT_FLAGS = frozenset({"--spec-draft-hf", "-hfd", "-hfrd", "--hf-repo-draft"})
_HF_DRAFT_ENV = ("LLAMA_ARG_SPEC_DRAFT_HF_REPO", "LLAMA_ARG_HFD_REPO")


def _caller_speculation(args: Sequence[str]) -> Optional[str]:
    """The caller's drafter or MTP setting, if any: its weights, KV and rollback state are not
    in a plan, so such a load stays on this machine."""
    for a in args:
        name = str(a).split("=", 1)[0]
        if name in _DRAFTER_FLAGS or name in _HF_DRAFT_FLAGS:
            return name
    from core.inference.llama_cpp import _accumulated_spec_types, _child_spec_env

    # The launch scrubs every LLAMA_ARG_SPEC_* / drafter variable unless argv sets --spec-type.
    env = _child_spec_env(args, os.environ)
    for k in _DRAFTER_ENV + _HF_DRAFT_ENV:
        if (env.get(k) or "").strip():
            return k
    kinds = sorted(t for t in _accumulated_spec_types(args, env) if "draft" in t or "mtp" in t)
    return f"--spec-type {kinds[0]}" if kinds else None


def _normalized(args: Sequence[str]) -> List[str]:
    """Long option names with underscores spelled with hyphens, values untouched."""
    out = []
    for token in (str(a) for a in args):
        if token.startswith("--"):
            name, sep, value = token.partition("=")
            token = name.replace("_", "-") + sep + value
        out.append(token)
    return out


def _flag_args(args: Sequence[str], flags: frozenset) -> List[str]:
    tokens = [str(a) for a in args]
    out: List[str] = []
    for i, token in enumerate(tokens):
        name, sep, _ = token.partition("=")
        if name in flags:
            out += [token] if sep else tokens[i : i + 2]
    return out


def _locks_host_copy(request: Any, args: Sequence[str]) -> bool:
    """mlock in any spelling: a pinned full host copy, on unified memory the pool the layers use."""
    if "mlock" in str(getattr(request, "load_mode", None) or "").lower():
        return True
    from core.inference.llama_server_args import _LOAD_MODE_FLAGS, _MLOCK_FLAGS

    for i, a in enumerate(args):
        name, eq, inline = str(a).partition("=")
        if name in _MLOCK_FLAGS:
            return True
        if name in _LOAD_MODE_FLAGS:
            value = inline if eq else (str(args[i + 1]) if i + 1 < len(args) else "")
            if "mlock" in value.lower():
                return True
    from core.inference.llama_cpp import _LLAMA_ARG_FALSE_VALUES

    lock = (os.environ.get("LLAMA_ARG_MLOCK") or "").strip().lower()
    if lock and lock not in _LLAMA_ARG_FALSE_VALUES:
        return True
    return "mlock" in (os.environ.get("LLAMA_ARG_LOAD_MODE") or "").lower()


def _placement_owner(request: Any, flags: Sequence[str]) -> Optional[str]:
    """Why the request places the model itself, or None."""
    from core.inference import spark_serving as ss

    if getattr(request, "gpu_ids", None):
        return "an explicit GPU selection replaces placement"
    if str(getattr(request, "gpu_memory_mode", "auto") or "auto") == "manual":
        return "gpu_memory_mode=manual owns placement"
    if ss.argv_or_env_rpc(flags):
        return "the caller supplied --rpc; placement is theirs"
    placed = _caller_placement(flags)
    if placed:
        return f"the caller set {placed}; placement is theirs"
    if getattr(request, "tensor_parallel", False):
        # The toggle emits --split-mode tensor, which the block's layer split would override.
        return "tensor_parallel owns placement"
    return None


def _caller_placement(args: Sequence[str]) -> Optional[str]:
    """The first device / split setting the caller made, which a cluster block would override."""
    tokens = [str(a) for a in args]
    for i, token in enumerate(tokens):
        name, sep, inline = token.partition("=")
        if name not in _PLACEMENT_FLAGS:
            continue
        value = inline if sep else (tokens[i + 1] if i + 1 < len(tokens) else "")
        if name in _GPU_LAYER_FLAGS and value.strip().lower() in ("auto", "-1"):
            continue  # llama.cpp's own default, which leaves the fitter to place layers
        if name == "--moe-cache-mib" and value.strip() in ("", "0"):
            continue
        return name
    from core.inference.llama_cpp import _LLAMA_ARG_TRUE_VALUES, _is_positive_int

    for key in _PLACEMENT_ENV:
        value = (os.environ.get(key) or "").strip().lower()
        if not value or (key == "LLAMA_ARG_N_GPU_LAYERS" and value in ("auto", "-1")):
            continue
        if key == "LLAMA_ARG_CPU_MOE" and value not in _LLAMA_ARG_TRUE_VALUES:
            continue
        if key in ("LLAMA_ARG_N_CPU_MOE", "LLAMA_ARG_N_CPU_FFN") and not _is_positive_int(value):
            continue
        if key == "LLAMA_ARG_MOE_CACHE_MIB" and value == "0":
            continue
        return key
    return None


_GPU_LAYER_FLAGS = frozenset({"--gpu-layers", "--n-gpu-layers", "-ngl"})


# The Linux CUDA bundles link libcudart / libcublas from the Studio venv's nvidia wheels, which the
# backend adds for its own llama-server; a worker started over ssh gets the same dirs, found from
# the binary's place in a Studio install (<root>/llama.cpp/build/bin), else it serves CPU only.
_CUDA_RUNTIME = (
    'd=$(dirname "$0"); r="$d/../../.."; p="$d"; '
    'for x in "$r"/studio/unsloth_studio/lib/python3*/site-packages/nvidia/*/lib '
    '"$r"/unsloth_studio/lib/python3*/site-packages/nvidia/*/lib '
    '"$r"/studio/unsloth_studio/lib/python3*/site-packages/torch/lib '
    '"$r"/unsloth_studio/lib/python3*/site-packages/torch/lib /usr/local/cuda/lib64; do '
    '[ -d "$x" ] && p="$p:$x"; done; '
    'export LD_LIBRARY_PATH="$p${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; exec "$0" "$@"'
)


def _with_cuda_runtime(argv: List[str]) -> List[str]:
    return ["sh", "-c", _CUDA_RUNTIME] + list(argv)


_DEVICE_LINE = re.compile(r"^\s+(\S+): (.*) \((\d+) MiB, (\d+) MiB free\)\s*$")
_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]*$")


class ClusterConfigError(ValueError):
    """The cluster configuration cannot be used as written."""


@dataclass(frozen = True)
class Node:
    host: str
    rpc_port: int = RPC_PORT_DEFAULT
    ssh_user: Optional[str] = None
    managed: bool = False

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.rpc_port}"

    @property
    def ssh_target(self) -> str:
        return f"{self.ssh_user}@{self.host}" if self.ssh_user else self.host


@dataclass
class Device:
    name: str
    description: str
    total_bytes: int
    free_bytes: int
    endpoint: Optional[str] = None  # None: on this machine
    index: int = 0  # among the devices behind one endpoint; RPC<k> renumbers with the --rpc list

    @property
    def remote(self) -> bool:
        return self.endpoint is not None

    @property
    def key(self) -> Tuple[str, str]:
        return (self.endpoint, str(self.index)) if self.endpoint else ("", self.name)


@dataclass
class ClusterPlan:
    topology: str
    reason: str
    devices: List[Device] = field(default_factory = list)
    weights: List[float] = field(default_factory = list)
    need_bytes: Optional[float] = None
    usable_bytes: Optional[float] = None
    extra_bytes: float = 0.0
    # Charged but possibly never loaded (a Hub projector the vision switch may drop): not freed.
    uncertain_extra_bytes: float = 0.0
    # Where the projector / drafter was charged; kept even when it gets no layers.
    sidecar_device: Optional[Device] = None
    # Never counted as freed by a replacement: the KV estimate is not the cache's real layout.
    kv_bytes: float = 0.0
    layer_bytes: float = 0.0
    # The input embedding, counted in the model size but kept in host RAM: never freed VRAM.
    host_bytes: float = 0.0

    def resident(self) -> Dict[Tuple[str, str], float]:
        """What this plan puts on each device, so a replacement load can count it as freed."""
        held = max(0.0, float(self.need_bytes or 0) - self.kv_bytes - self.host_bytes)
        # Whole-layer rounding can leave a device a layer short of its share.
        out = {
            d.key: max(0.0, w * held - self.layer_bytes) for d, w in zip(self.devices, self.weights)
        }
        local = self.sidecar_device or next((d for d in self.devices if not d.remote), None)
        if local is not None:
            out[local.key] = out.get(local.key, 0.0) + max(
                0.0, self.extra_bytes - self.uncertain_extra_bytes
            )
        return out

    def snapshot(self) -> Dict[str, Any]:
        return {
            "topology": self.topology,
            "reason": self.reason,
            "need_gib": None if self.need_bytes is None else round(self.need_bytes / _GIB, 2),
            "usable_gib": None if self.usable_bytes is None else round(self.usable_bytes / _GIB, 2),
            "devices": [
                {
                    "name": d.name,
                    "endpoint": d.endpoint or "local",
                    "description": d.description,
                    "total_gib": round(d.total_bytes / _GIB, 2),
                    "free_gib": round(d.free_bytes / _GIB, 2),
                    "weight": round(w, 4),
                }
                for d, w in zip(self.devices, self.weights or [0.0] * len(self.devices))
            ],
        }


def _parse_node(text: str, *, managed: bool = False) -> Node:
    raw = text.strip()
    user = None
    if "@" in raw:
        user, raw = raw.split("@", 1)
        if not user:
            raise ClusterConfigError(f"empty ssh user in {text!r}")
    if raw.startswith("["):
        raise ClusterConfigError(f"IPv6 node addresses are not supported yet: {text!r}")
    host, sep, port_text = raw.partition(":")
    host = host.lower().rstrip(".")  # DNS is case-insensitive; "a.local." is "a.local"
    if not host or not _HOST_RE.match(host):
        raise ClusterConfigError(f"not a host name or IPv4 address: {text!r}")
    port = RPC_PORT_DEFAULT
    if sep:
        if not port_text.isdigit() or not (0 < int(port_text) < 65536):
            raise ClusterConfigError(f"not a port: {text!r}")
        port = int(port_text)
    return Node(host = host, rpc_port = port, ssh_user = user, managed = managed)


def parse_nodes_env(value: str, *, managed: bool = False) -> List[Node]:
    nodes = [_parse_node(part, managed = managed) for part in value.split(",") if part.strip()]
    return _dedupe(nodes)


def parse_nodes_json(data: Any) -> List[Node]:
    entries = data.get("nodes") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ClusterConfigError('cluster.json needs {"nodes": [...]}')
    nodes = []
    for entry in entries:
        if isinstance(entry, str):
            nodes.append(_parse_node(entry))
            continue
        if not isinstance(entry, dict) or not entry.get("host"):
            raise ClusterConfigError(f"a node needs a host: {entry!r}")
        managed = entry.get("managed", False)
        if not isinstance(managed, bool):
            raise ClusterConfigError(f"managed must be true or false: {entry!r}")
        node = _parse_node(str(entry["host"]), managed = managed)
        port = entry.get("rpc_port", node.rpc_port)
        if not isinstance(port, int) or not (0 < port < 65536):
            raise ClusterConfigError(f"not a port: {entry!r}")
        user = entry.get("ssh_user", node.ssh_user)
        nodes.append(
            Node(host = node.host, rpc_port = port, ssh_user = user or None, managed = node.managed)
        )
    return _dedupe(nodes)


def _dedupe(nodes: List[Node]) -> List[Node]:
    seen = set()
    for node in nodes:
        if node.endpoint in seen:
            raise ClusterConfigError(f"{node.endpoint} is listed twice")
        seen.add(node.endpoint)
    if not nodes:
        raise ClusterConfigError("the cluster lists no nodes")
    return nodes


def _config_path() -> Optional[Path]:
    try:
        from utils.paths.storage_roots import studio_root
        return studio_root() / CONFIG_NAME
    except Exception:
        return None


def configured() -> bool:
    """Whether the operator asked for a cluster at all; cheap, no file parsing beyond existence."""
    if (os.environ.get(ENV_NODES) or "").strip():
        return True
    path = _config_path()
    return bool(path is not None and path.is_file())


def load_nodes() -> List[Node]:
    value = (os.environ.get(ENV_NODES) or "").strip()
    if value:
        return _one_managed_worker_per_host(
            parse_nodes_env(value, managed = (os.environ.get(ENV_MANAGED) or "") == "1")
        )
    path = _config_path()
    if path is None or not path.is_file():
        raise ClusterConfigError("no cluster configured")
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError) as exc:
        raise ClusterConfigError(f"{path} could not be read: {exc}") from exc
    nodes = parse_nodes_json(data)
    if (os.environ.get(ENV_MANAGED) or "") == "1":
        nodes = [replace(n, managed = True) for n in nodes]
    return _one_managed_worker_per_host(nodes)


def _one_managed_worker_per_host(nodes: List[Node]) -> List[Node]:
    # A managed rpc-server exposes every accelerator on its host, so a second one would count the
    # same memory twice.
    hosts = [n.host for n in nodes]
    for host in {n.host for n in nodes if n.managed}:
        if hosts.count(host) > 1:
            raise ClusterConfigError(f"{host} is managed, so it can be listed only once")
    return nodes


def parse_device_list(text: str, endpoints: Sequence[str]) -> List[Device]:
    """``llama-server --list-devices``. In the unslothai fork an RPC device's description is its
    endpoint; upstream builds may say more, so a whole endpoint token inside it also counts."""
    devices = []
    seen: Dict[str, int] = {}
    for line in (text or "").splitlines():
        m = _DEVICE_LINE.match(line)
        if not m:
            continue
        name, desc, total, free = m.group(1), m.group(2).strip(), int(m.group(3)), int(m.group(4))
        endpoint = None
        if name.startswith("RPC"):
            endpoint = desc if desc in endpoints else _endpoint_in(desc, endpoints)
        elif name.startswith("BLAS"):
            continue  # a host-memory accelerator, not somewhere to put layers
        index = seen.get(endpoint or "", 0)
        seen[endpoint or ""] = index + 1
        devices.append(Device(name, desc, total * _MIB, free * _MIB, endpoint, index))
    return devices


def _endpoint_in(desc: str, endpoints: Sequence[str]) -> str:
    # host:5005 must not claim host:50052, so the port may not run on into more digits.
    for e in sorted(endpoints, key = len, reverse = True):
        if re.search(rf"(?<![\w.]){re.escape(e)}(?!\d)", desc):
            return e
    return desc


def list_devices(binary: str, endpoints: Sequence[str]) -> Tuple[List[Device], str]:
    """The devices the coordinator would see with these RPC servers, and any error text."""
    try:
        # The child's own environment: the managed CUDA / ROCm runtime and the bundle's libraries.
        from core.inference.llama_cpp import LlamaCppBackend
        env = LlamaCppBackend._llama_server_env_for_binary(binary)
    except Exception:
        env = None
    try:
        proc = subprocess.run(
            [binary, "--rpc", ",".join(endpoints), "--list-devices"],
            capture_output = True,
            text = True,
            timeout = LIST_DEVICES_TIMEOUT_S,
            env = env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    devices = parse_device_list(proc.stdout, endpoints)
    if devices:
        return devices, ""
    # An unreachable worker aborts the probe with a backtrace; the reason is the line naming it.
    lines = [l.strip() for l in (proc.stderr or "").splitlines() if l.strip()]
    reason = next((l for l in lines if "Failed to connect" in l or "error" in l.lower()), None)
    return [], (reason or (lines[-1] if lines else "no devices")).split(": ", 1)[-1][:300]


def usable_bytes(
    device: Device,
    margin_bytes: float = 0.0,
    scratch_bytes: float = 0.0,
) -> float:
    reserve = (
        max(DEVICE_RESERVE_MIN_BYTES, DEVICE_RESERVE_FRACTION * device.total_bytes, margin_bytes)
        + scratch_bytes
    )
    return max(0.0, float(device.free_bytes) - reserve)


def plan_cluster(
    model_bytes: Optional[float],
    kv_bytes: float,
    local_extra_bytes: float,
    devices: Sequence[Device],
    *,
    forced: Optional[str] = None,
    layer_bytes: Optional[float] = None,
    margin_bytes: float = 0.0,
    scratch_bytes: float = 0.0,
) -> ClusterPlan:
    """single when this machine holds the load, else a layer split weighted by usable memory.

    Projector and drafter are charged to the coordinator; callers add adapters to ``model_bytes``.
    """
    if model_bytes is None:
        return ClusterPlan("single", "model size unknown; not guessing, serving on this machine")
    local = [d for d in devices if not d.remote]
    remote = [d for d in devices if d.remote]
    need = float(model_bytes) + float(kv_bytes or 0)
    local_budgets = [usable_bytes(d, margin_bytes, scratch_bytes) for d in local]
    local_usable = sum(local_budgets)
    # The projector / drafter is pinned to the first local GPU, so it has to fit there itself.
    if local and local_extra_bytes > local_budgets[0]:
        return ClusterPlan(
            "refuse",
            f"the projector / drafter ({local_extra_bytes / _GIB:.1f} GiB) does not fit "
            f"{local[0].name} ({local_budgets[0] / _GIB:.1f} GiB usable), where it loads",
            need_bytes = need,
        )
    if forced != "layer_split" and local and need + local_extra_bytes <= local_usable:
        # The model's share per GPU, after the sidecar takes its room on the first.
        model_budgets = [local_budgets[0] - local_extra_bytes] + local_budgets[1:]
        return ClusterPlan(
            "single",
            f"fits this machine ({(need + local_extra_bytes) / _GIB:.1f} GiB of "
            f"{local_usable / _GIB:.1f} GiB usable)",
            list(local),
            [b / sum(model_budgets) for b in model_budgets],
            need_bytes = need,
            usable_bytes = local_usable,
            extra_bytes = float(local_extra_bytes),
        )
    if forced == "single":
        return ClusterPlan("single", f"forced by {ENV_TOPOLOGY}=single", need_bytes = need)
    if not remote:
        return ClusterPlan("single", "no remote device answered; serving on this machine")
    # Remote first, this machine last: llama.cpp puts the output layer on the last device, so the
    # logits stay local and the wire carries only the hidden state. Sidecars load on the first
    # local device.
    ordered = remote + local
    budgets = [usable_bytes(d, margin_bytes, scratch_bytes) for d in remote] + local_budgets
    if local:
        budgets[len(remote)] -= local_extra_bytes
    if layer_bytes:
        # Layers land whole, so any device can end up with one more than its share: keep that much
        # free on every device, not 5% of the total.
        budgets = [max(0.0, b - layer_bytes) for b in budgets]
        margin = 0.0
    else:
        margin = LAYER_ROUNDING_MARGIN
    total = sum(budgets)
    if not local and local_extra_bytes > 0:
        return ClusterPlan(
            "refuse",
            "this machine has no accelerator to hold the projector / drafter the load names",
            need_bytes = need,
        )
    if total <= 0 or need * (1 + margin) > total:
        slack = (
            f"one {layer_bytes / _GIB:.2f} GiB layer kept free per device"
            if layer_bytes
            else f"+{LAYER_ROUNDING_MARGIN:.0%} for whole layers"
        )
        return ClusterPlan(
            "refuse",
            f"the model with its KV needs {need / _GIB:.1f} GiB ({slack}) against "
            f"{total / _GIB:.1f} GiB usable over {len(ordered)} devices; "
            "add a node, use a smaller quant, fewer users or a shorter context",
            ordered,
            need_bytes = need,
            usable_bytes = total,
        )
    weights = [b / total for b in budgets]
    keep = [(d, w) for d, w in zip(ordered, weights) if w > 0]
    return ClusterPlan(
        "layer_split",
        f"{need / _GIB:.1f} GiB over {len(keep)} devices weighted by usable memory "
        f"({total / _GIB:.1f} GiB usable)",
        [d for d, _ in keep],
        [w for _, w in keep],
        need_bytes = need,
        usable_bytes = total,
        extra_bytes = float(local_extra_bytes),
        sidecar_device = local[0] if local else None,
    )


def split_extra_args(endpoints: Sequence[str], plan: ClusterPlan) -> List[str]:
    weights = ",".join(f"{w:.4f}".rstrip("0").rstrip(".") or "0" for w in plan.weights)
    return [
        "--rpc",
        ",".join(endpoints),
        "--device",
        ",".join(d.name for d in plan.devices),
        "-sm",
        "layer",
        "--tensor-split",
        weights,
        "--cache-ram",
        "0",
    ]


# Blocks this module appended to a launch, so a reload that inherits the stored extras drops them
# instead of reading them as the caller's own --rpc.
_MANAGED_BLOCKS: List[List[str]] = []


def strip_managed_block(extra: Sequence[str]) -> List[str]:
    tokens = [str(t) for t in extra]
    for block in reversed(_MANAGED_BLOCKS):
        n = len(block)
        if n and len(tokens) >= n and tokens[-n:] == block:
            return tokens[:-n]
    return tokens


def _remember_block(block: List[str]) -> None:
    if block in _MANAGED_BLOCKS:
        _MANAGED_BLOCKS.remove(block)
    _MANAGED_BLOCKS.append(list(block))
    del _MANAGED_BLOCKS[:-8]


class ClusterServing:
    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self.config_error: str = ""
        self.topology: str = "single"
        self.reason: str = "no load yet"
        self.plan: Optional[ClusterPlan] = None
        self.block: List[str] = []
        self.workers: Dict[str, Any] = {}  # endpoint -> spark_serving.PeerProcess
        self.last_error: str = ""
        self._passthrough: bool = False
        self._alongside: bool = False
        # A load this planned and has not completed: its after_load / load_failed come here even
        # if the config changed meanwhile.
        self.in_flight: bool = False
        self._has_projector: bool = False
        self._n_layers: Optional[int] = None
        # Set when the config went away while workers run: the next committed load stops them.
        self.retire_on_commit: bool = False
        self._pre_load: Optional[Tuple[str, str, Optional[ClusterPlan], List[str]]] = None
        self._cancel_event: Any = None
        # The backend serving the last committed plan; while it is loaded its buffers are still in
        # the probe's "used", and the load being planned replaces it.
        self._backend: Any = None
        self._resident: Dict[Tuple[str, str], float] = {}
        # Workers the current load's probe found unusable; a commit without them stops them.
        self._down: set = set()
        self._native_ctx: bool = False
        self._projector_url: bool = False
        self._unsized_kv: bool = False
        self._max_block: Optional[float] = None
        self._host_embd: float = 0.0
        self._untied = False
        self._ctx_compute: float = 0.0
        self._checkpoints: float = 0.0
        self._recurrent: float = 0.0
        self._recurrent_layer: float = 0.0
        self._kv_layer_frac: float = 0.0
        self._internal_pre: Optional[Tuple[Any, ...]] = None
        self._requested_model: Tuple[str, Any] = ("", None)
        self._served_model: Tuple[str, Any] = ("", None)
        self._attention_layers: int = 0
        self._nextn_bytes: float = 0.0
        self._projector_file: Optional[str] = None
        self._uncertain_extra: float = 0.0
        # Workers this load (re)started: they hold none of the served split's buffers.
        self._started: set = set()

    def _cancelled(self) -> bool:
        event = self._cancel_event
        try:
            return bool(event is not None and event.is_set())
        except Exception:
            return False

    def refresh_nodes(self) -> List[Node]:
        try:
            self.nodes = load_nodes()
            self.config_error = ""
        except ClusterConfigError as exc:
            # The nodes a served split runs on stay known, so a commit does not stop its workers.
            self.config_error = str(exc)
        return self.nodes

    async def _ensure_workers(self) -> Dict[str, str]:
        """Start the managed nodes' rpc-servers that are not running: ``{endpoint: why}`` for the
        ones that could not be.

        Live workers are never restarted, even on forced reloads: that drops the served model's buffers.
        """
        from core.inference import spark_serving as ss

        local_rpc = ss.rpc_server_binary()
        names = ([os.path.basename(local_rpc)] if local_rpc else []) + list(ss.rpc_server_names())
        failed: Dict[str, str] = {}
        for node in self.nodes:
            if not node.managed:
                continue
            if self._cancelled():
                break
            running = self.workers.get(node.endpoint)
            if running is not None and running.alive:
                continue
            if running is not None:
                await running.stop()
                self.workers.pop(node.endpoint, None)
            candidates: List[str] = []
            for name in dict.fromkeys(names):
                candidates += [
                    c for c in ss.peer_binary_candidates(local_rpc, name) if c not in candidates
                ]
            rc, out, _err = await ss.ssh_run(node.ssh_target, ss.find_binary_script(candidates))
            binary = out.strip().splitlines()[-1] if out.strip() else "MISSING"
            if rc != 0 or binary == "MISSING":
                failed[node.endpoint] = f"{node.host} has no ggml-rpc-server at the bundle path"
                continue
            argv = _with_cuda_runtime(
                ss.rpc_server_argv(binary, bind = node.host, port = node.rpc_port, cache = False)
            )
            log_dir = ss._log_dir()
            process = ss.PeerProcess(
                "ggml-rpc-server",
                node.ssh_target,
                argv,
                log_dir / f"rpc-server-{node.host}-{node.rpc_port}.log" if log_dir else None,
            )
            try:
                await process.start()
            except (OSError, ValueError) as exc:  # ssh missing, process limit, Windows: this node
                failed[node.endpoint] = f"could not start ssh to {node.host}: {exc}"
                continue
            if not await ss.wait_for_own_port(
                process,
                node.host,
                node.rpc_port,
                WORKER_START_TIMEOUT_S,
                cancelled = self._cancelled,
            ):
                tail = list(process.tail)[-3:]
                await process.stop()
                failed[node.endpoint] = (
                    f"ggml-rpc-server did not take {node.endpoint} (last output: {tail})"
                )
                continue
            self.workers[node.endpoint] = process
            self._started.add(node.endpoint)
        return failed

    async def before_load(
        self,
        request: Any,
        n_parallel: int,
        *,
        inherited_extra_args: Optional[List[str]] = None,
        cancel_event: Any = None,
    ) -> Any:
        from core.inference import spark_serving as ss

        out = await self._plan_load(
            request,
            n_parallel,
            inherited_extra_args = inherited_extra_args,
            cancel_event = cancel_event,
        )
        if not self._passthrough and self.topology != "layer_split" and self._serving_split():
            # Every path that drops a served split (forced single, caller placement, a plan that
            # now fits here): the route strips the old block, so equal arguments would keep the
            # primary on remote buffers this load no longer plans.
            out = ss.SparkServing._request_with(out, {"force_reload": True})
        return out

    async def _plan_load(
        self,
        request: Any,
        n_parallel: int,
        *,
        inherited_extra_args: Optional[List[str]] = None,
        cancel_event: Any = None,
    ) -> Any:
        from core.inference import spark_serving as ss

        self._cancel_event = cancel_event
        self.in_flight = True
        self._alongside = bool(getattr(request, "alongside", False)) or bool(
            getattr(request, "force_alongside", False)
        )
        self._passthrough = self._alongside or (
            str(getattr(request, "engine", "auto") or "auto") != "auto"
        )
        if self._passthrough:
            return request
        self.settle_internal_load()
        if n_parallel > 1 and not _kv_unified_supported(ss.llama_server_binary()):
            # The loader launches one slot then: --parallel N would split -c N ways.
            n_parallel = 1
        self._requested_model = (
            str(getattr(request, "model_path", "") or ""),
            getattr(request, "gguf_variant", None),
        )
        self._pre_load = (self.topology, self.reason, self.plan, list(self.block))
        # Every return below that does not plan leaves no plan behind for after_load to record.
        self.plan, self.block = None, []
        self._down = set()  # _started stays until a load commits: a refusal keeps those workers
        if not self.refresh_nodes() or self.config_error:
            if self.config_error and self._pre_load[3]:
                # The served split keeps its record, so a later retirement still sees it, and its
                # block (the route strips it from inherited args) so a forced reload keeps it.
                self.topology, _reason, self.plan, self.block = self._pre_load
                self.reason = f"cluster config: {self.config_error}; still serving the last split"
                if self._requested_model != self._served_model:
                    return request  # another model: no plan, so no stale block either
                explicit = getattr(request, "llama_extra_args", None)
                effective = list(explicit if explicit is not None else (inherited_extra_args or []))
                owner = _placement_owner(request, _normalized(effective))
                if owner:
                    # The block would override it (last wins); before_load forces the reload.
                    self.topology = "single"
                    self.reason = f"cluster config: {self.config_error}; {owner}"
                    return request
                if not _contains([str(a) for a in effective], self.block):
                    effective += self.block
                return ss.SparkServing._request_with(request, {"llama_extra_args": effective})
            self.topology, self.reason = "single", f"cluster config: {self.config_error}"
            return request
        explicit = getattr(request, "llama_extra_args", None)
        effective = list(explicit if explicit is not None else (inherited_extra_args or []))
        # llama.cpp reads --ctx_size as --ctx-size; every check below reads the normalised form.
        flags = _normalized(effective)
        owner = _placement_owner(request, flags)
        if owner:
            self.topology, self.reason = "single", owner
            return request
        if any(str(a).split("=", 1)[0] in ("-mv", "--model-vocoder") for a in flags):
            # A second model llama.cpp loads beside this one, on devices this does not plan.
            self.topology, self.reason = "single", "a vocoder model loads beside it"
            return request
        from core.inference.llama_server_args import fit_is_effectively_on

        if not fit_is_effectively_on(flags, os.environ):
            # Nothing then shrinks the context or its compute buffers, which this does not price.
            self.topology, self.reason = "single", "the fitter is off; placement is the caller's"
            return request
        drafter = _caller_speculation(flags)
        if drafter:
            self.topology, self.reason = (
                "single",
                f"the caller's speculative decoding ({drafter}) is placed by them",
            )
            return request
        from core.inference.llama_cpp import _kv_offload_from_args, _named_batch_sizes

        if not _kv_offload_from_args(flags):
            # Host KV shares the pool with the weights on unified-memory machines.
            self.topology, self.reason = (
                "single",
                "the caller keeps the KV cache in host memory; placement is theirs",
            )
            return request
        if _locks_host_copy(request, flags):
            self.topology, self.reason = (
                "single",
                "the caller locks a full host copy of the model (mlock); placement is theirs",
            )
            return request
        from core.inference.llama_cpp import _emitted_n_batch

        batch, ubatch, _named, _named_u = _named_batch_sizes(
            flags,
            os.environ,
            # The loader raises --batch-size to the slot count, and the micro-batch with it.
            _emitted_n_batch(getattr(request, "n_batch", None), n_parallel),
            getattr(request, "n_ubatch", None),
        )
        if min(batch, ubatch) > 512:
            # Every device's compute buffer grows with it; the plan prices llama.cpp's default.
            self.topology, self.reason = (
                "single",
                f"a micro-batch of {min(batch, ubatch)} is placed by the caller",
            )
            return request
        model_bytes, kv_bytes, local_extra = await self._size(request, n_parallel, flags, ss)
        if self._unsized_kv:
            self.topology, self.reason = (
                "single",
                "the KV cache and compute buffers could not be sized from the GGUF header",
            )
            return request
        if self._native_ctx:
            self.topology, self.reason = (
                "single",
                "a context size of 0 opens the model's whole context",
            )
            return request
        if self._has_projector and (
            any(str(a).split("=", 1)[0] in _MMPROJ_DEVICE_FLAGS for a in flags)
            or os.environ.get("MTMD_BACKEND_DEVICE")
        ):
            self.topology, self.reason = (
                "single",
                "the caller pins the projector; placement is theirs",
            )
            return request
        if self._projector_url:
            self.topology, self.reason = (
                "single",
                "a projector fetched during the load (a URL or an unread Hub listing) cannot be sized",
            )
            return request
        forced = (os.environ.get(ENV_TOPOLOGY) or "").strip().lower().replace("-", "_") or None
        if forced == "single":
            self.topology, self.reason = "single", f"forced by {ENV_TOPOLOGY}=single"
            return request
        if model_bytes is None:
            # Not a GGUF this can size (safetensors, diffusion, an unknown repo): nothing to split,
            # so no worker is started for it.
            self.topology, self.reason = "single", "model size unknown; serving on this machine"
            return request
        failed = await self._ensure_workers()
        if self._cancelled():
            self.topology, self.reason = "single", "the load was cancelled"
            return request
        binary = ss.llama_server_binary()
        if not binary:
            self.topology, self.reason = "single", "no local llama-server to plan with"
            return request
        # One unreachable worker aborts llama.cpp's whole probe, so plan with the ones that answer.
        live = [n for n in self.nodes if n.endpoint not in failed]
        reachable = await asyncio.gather(
            *(ss.wait_for_port(n.host, n.rpc_port, REACH_TIMEOUT_S) for n in live)
        )
        down = list(failed) + [n.endpoint for n, ok in zip(live, reachable) if not ok]
        endpoints = [n.endpoint for n, ok in zip(live, reachable) if ok]
        devices, error = (
            await asyncio.to_thread(list_devices, binary, endpoints) if endpoints else ([], "")
        )
        while endpoints and error and not any(d.remote for d in devices):
            # One worker that answers TCP but fails the handshake aborts the whole listing.
            bad = _endpoint_in(error, endpoints)
            if bad in endpoints:
                endpoints.remove(bad)
                down.append(bad)
            elif len(endpoints) > 1:
                # A hung handshake times out without naming anyone: ask each worker alone.
                alone = await asyncio.gather(
                    *(asyncio.to_thread(list_devices, binary, [e]) for e in endpoints)
                )
                answered = [
                    e for e, (devs, err) in zip(endpoints, alone) if any(d.remote for d in devs)
                ]
                if not answered or len(answered) == len(endpoints):
                    break
                down += [e for e in endpoints if e not in answered]
                endpoints = answered
            else:
                break
            devices, error = (
                await asyncio.to_thread(list_devices, binary, endpoints) if endpoints else ([], "")
            )
        if self._cancelled():
            self.topology, self.reason = "single", "the load was cancelled"
            return request
        if self._backend is not None and getattr(self._backend, "is_loaded", False):
            for d in devices:
                held = self._resident.get(d.key, 0.0)
                if d.remote and held > 512 * 2**20 and d.free_bytes > d.total_bytes - held / 2:
                    # Its memory is free again: the worker restarted (or was replaced by hand)
                    # and lost the served layers, so equal arguments must relaunch.
                    self._started.add(d.endpoint)
            # This load tears the current one down first, so what it holds counts as free.
            for d in devices:
                if d.endpoint in self._started:
                    continue
                d.free_bytes = int(min(d.total_bytes, d.free_bytes + self._resident.get(d.key, 0)))
        stray = [d.description for d in devices if d.remote and d.endpoint not in endpoints]
        if stray:
            # RPC<k> cannot be tied to an --rpc entry, so no --device list built from it is safe.
            self.topology, self.reason = "single", f"could not tell which worker {stray[0]!r} is"
            self.last_error = self.reason
            return request
        missing = down + [e for e in endpoints if not any(d.endpoint == e for d in devices)]
        self._down = set(missing)
        self.last_error = ""
        if missing:
            self.last_error = f"no device from {', '.join(missing)}" + (
                f": {error}" if error else ""
            )
            if failed:
                self.last_error += "; " + "; ".join(failed.values())
            logger.warning("cluster serving: %s", self.last_error)
        layer_bytes = None
        if self._n_layers and model_bytes:
            # The largest block a tensor split can round onto a device, not the average one.
            layer_weights = self._max_block or model_bytes / self._n_layers
            attention = max(0.0, kv_bytes - self._recurrent)
            # Spread over the attention layers only: a hybrid's recurrent blocks hold none.
            per_attention = attention / max(1, min(self._n_layers, self._attention_layers or 10**9))
            layer_kv = attention * self._kv_layer_frac or per_attention
            layer_bytes = layer_weights + max(layer_kv, per_attention)
            layer_bytes += self._recurrent_layer
        from core.inference.llama_server_args import fit_target_margin_in

        # llama.cpp's fitter keeps an explicit --fit-target free on every device.
        margin = (fit_target_margin_in(flags, os.environ) or 0.0) * 1024 * 1024
        # Context checkpoints live in host RAM, which a local accelerator may share (unified
        # memory); with none there they take nothing a plan places.
        local = [d for d in devices if not d.remote]
        unified = bool(local) and _shares_host_memory(local[0])
        checkpoints = self._checkpoints if unified else 0.0
        if not unified and self._host_projector:
            # A --no-mmproj-offload projector stays in host RAM, which a discrete GPU does not share.
            local_extra = max(0.0, local_extra - self._host_projector)
            self._uncertain_extra = max(0.0, self._uncertain_extra - self._host_projector)

        def plan_for(weights: float) -> ClusterPlan:
            return plan_cluster(
                weights,
                kv_bytes,
                local_extra + checkpoints,
                devices,
                forced = forced,
                layer_bytes = layer_bytes,
                margin_bytes = margin,
                # Every device of a layer split holds its own context-dependent compute scratch.
                scratch_bytes = self._ctx_compute,
            )

        # An untied input embedding stays in host RAM, which only a unified device draws on.
        off_device = self._host_embd if self._untied and not unified else 0.0
        plan = plan_for(model_bytes - off_device)
        skipped = self._nextn_bytes  # in the plan's weights, never loaded on a split
        if plan.topology == "refuse" and self._nextn_bytes:
            # A split runs without speculation, so trailing NextN / MTP blocks never load.
            trimmed = plan_for(model_bytes - off_device - self._nextn_bytes)
            if trimmed.topology == "layer_split":
                plan, skipped = trimmed, 0.0
        plan.kv_bytes = float(kv_bytes or 0)
        plan.uncertain_extra_bytes = self._uncertain_extra + checkpoints
        plan.layer_bytes = float(layer_bytes or 0.0)
        plan.host_bytes = (self._host_embd - off_device) + (
            skipped if plan.topology == "layer_split" else 0.0
        )
        self.plan = plan
        if plan.topology == "refuse":
            self.topology, self.reason, self.plan, self.block = self._pre_load
            # The route refuses before any load starts, so no after_load / load_failed follows.
            self._pre_load, self.in_flight = None, False
            self.last_error = plan.reason
            raise ss.SparkLoadDoesNotFit(plan.reason)
        self.topology, self.reason = plan.topology, plan.reason
        if plan.topology != "layer_split":
            self.block = []
            return request
        # RPC<k> counts devices over the probed --rpc list, so every endpoint that exposed one stays
        # in it even at zero weight; one that exposed none takes no number and is left out.
        used = [e for e in endpoints if any(d.endpoint == e for d in devices)]
        self.block = split_extra_args(used, plan)
        local = plan.sidecar_device
        from core.inference.llama_cpp import _resolved_mmproj_offload

        caller_pinned = (
            any(str(a).split("=", 1)[0] in _MMPROJ_DEVICE_FLAGS for a in flags)
            or bool(os.environ.get("MTMD_BACKEND_DEVICE"))
            # --no-mmproj-offload keeps it in host RAM; a device pin would undo that.
            or _resolved_mmproj_offload(flags, os.environ) is False
        )
        if local is not None and not caller_pinned:
            if ss.llama_server_supports("--mmproj-device", binary):
                # The projector otherwise follows the first --device, a worker; it was charged here.
                self.block += ["--mmproj-device", local.name]
            elif self._has_projector:
                self.plan, self.block = None, []
                self.topology = "single"
                self.reason = "this llama-server cannot pin the projector to this machine"
                return request
        _remember_block(self.block)
        logger.info("cluster serving: layer split %s", " ".join(self.block))
        updates: Dict[str, Any] = {"llama_extra_args": effective + self.block}
        from core.inference.llama_cpp import _canonicalize_spec_mode, _extra_args_set_spec_type

        mode = _canonicalize_spec_mode(getattr(request, "speculative_type", None)) or "auto"
        if mode not in ("off", "ngram", "ngram-simple") and not _extra_args_set_spec_type(flags):
            # MTP state and a drafter Studio picks are resolved after this plan, outside its budget.
            updates["speculative_type"] = "off"
            self.reason += "; speculative decoding off for the split"
        served = list(self._pre_load[3]) if self._pre_load else []
        served_rpc = served[served.index("--rpc") + 1].split(",") if "--rpc" in served[:-1] else []
        if (
            self._backend is not None
            and getattr(self._backend, "is_loaded", False)
            and self._started & set(served_rpc)
        ):
            # A restarted worker holds none of the served model: equal arguments must relaunch.
            updates["force_reload"] = True
        return ss.SparkServing._request_with(request, updates)

    async def _size(self, request, n_parallel, extras, ss) -> Tuple[Optional[float], float, float]:
        self._uncertain_extra = 0.0
        self._host_projector = 0.0
        model_path = str(getattr(request, "model_path", "") or "")
        variant = getattr(request, "gguf_variant", None)
        local_file = ss.cached_repo_file(model_path, variant)
        size = ss.gguf_size_bytes(local_file)
        self._n_layers = await asyncio.to_thread(_gguf_block_count, local_file, ss)
        if local_file:
            header, blocks = await asyncio.to_thread(_local_gguf_header, local_file, ss)
        elif size is None:
            header, blocks = await asyncio.to_thread(
                _remote_gguf_header, model_path, variant, getattr(request, "hf_token", None), ss
            )
        else:
            header, blocks = {}, {}
        arch = header.get("general.architecture")
        nextn = int(header.get(f"{arch}.nextn_predict_layers") or 0)
        n_blocks = int(header.get(f"{arch}.block_count") or 0)
        trailing = {k for k in blocks if nextn and n_blocks and k >= n_blocks - nextn}
        # Trailing NextN / MTP blocks are skipped with speculation off, as a split runs, so they
        # neither round onto a device nor stay resident.
        self._nextn_bytes = float(sum(blocks[k] for k in trailing))
        layers = {k: v for k, v in blocks.items() if k >= 0 and k not in trailing}
        # The input embedding stays on the host; the output matrix (or a tied model's copy of the
        # embedding) is one more lump on the last device.
        self._host_embd = float(blocks.get(-2, 0))
        self._untied = -1 in blocks
        output = blocks.get(-1, blocks.get(-2, 0))
        unread_adapters = 0
        if layers:
            for path in ss.sidecar_files(_flag_args(extras, _ADAPTER_FLAGS)):
                try:
                    with open(path, "rb") as f:
                        _vals, extra_blocks = _read_gguf_header(f, tensors = True)
                except Exception:
                    extra_blocks = {}
                lora = {k: v for k, v in extra_blocks.items() if k >= 0}
                if lora or -1 in extra_blocks or -2 in extra_blocks:
                    for k, v in lora.items():
                        layers[k] = layers.get(k, 0) + v  # beside the layer it modifies
                    output += extra_blocks.get(-1, 0)  # beside the output matrix
                    self._host_embd += extra_blocks.get(-2, 0)  # beside the host embedding
                else:
                    unread_adapters += int(ss.gguf_size_bytes(path) or 0)
        self._max_block = (
            float(max(max(layers.values()), output) + unread_adapters) if layers else None
        )
        if not self._n_layers:
            arch = header.get("general.architecture")
            self._n_layers = header.get(f"{arch}.block_count") or None
        sidecars, _unknown = ss.sidecar_bytes(extras)
        # Adapters and control vectors sit beside the layers they modify, on every device.
        adapters, _unknown = ss.sidecar_bytes(_flag_args(extras, _ADAPTER_FLAGS))
        sidecars -= adapters
        from core.inference.llama_server_args import extra_args_disable_mmproj

        named = {"--mmproj", "-mm", "--mmproj-url", "-mmu"}
        disable_vision = bool(getattr(request, "disable_vision", False))
        # Inherited projectors survive --no-mmproj (not the vision switch), as the loader reads them.
        inherited = any(
            (os.environ.get(k) or "").strip() for k in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL")
        )
        self._has_projector = inherited or (
            not extra_args_disable_mmproj(extras)
            and any(str(a).split("=", 1)[0] in named for a in extras)
        )
        from core.inference.llama_cpp import _resolved_mmproj_offload

        host_projector = _resolved_mmproj_offload(extras, os.environ) is False
        # llama.cpp applies LLAMA_ARG_MMPROJ before argv, so an argv --mmproj replaces it.
        env_projector = (os.environ.get("LLAMA_ARG_MMPROJ") or "").strip()
        if _flag_args(extras, frozenset({"--mmproj", "-mm"})):
            env_projector = ""
        self._projector_url = bool(
            self._has_projector
            and not disable_vision
            and (
                (os.environ.get("LLAMA_ARG_MMPROJ_URL") or "").strip()
                or (
                    not extra_args_disable_mmproj(extras)
                    and _flag_args(extras, frozenset({"--mmproj-url", "-mmu"}))
                )
            )
        )
        dropped: List[str] = []
        if self._has_projector and disable_vision:
            # The vision switch drops a named image projector (and an unread URL); an audio-only
            # one stays, as the backend does.
            files = ss.sidecar_files(_flag_args(extras, frozenset({"--mmproj", "-mm"})))
            env_file = env_projector
            if env_file:
                files.append(os.path.abspath(env_file))
            dropped = [p for p in dict.fromkeys(files) if ss._projector_accepts_image(p)]
            self._has_projector = any(p not in dropped and os.path.isfile(p) for p in files)
        auto_bytes = 0
        auto_projector = None
        if extra_args_disable_mmproj(extras):
            pass
        elif not any(str(a).split("=", 1)[0] in named for a in extras) and not any(
            os.environ.get(k) for k in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL")
        ):
            projector = ss.auto_projector_file(
                local_file, disable_vision = disable_vision
            ) or _quant_dir_projector(local_file, disable_vision, ss)
            auto_projector = projector
            if projector:
                auto_bytes = int(ss.gguf_size_bytes(projector) or 0)
                sidecars += auto_bytes
            self._has_projector = bool(projector)
            if not local_file:
                # A first load of a Hub repo fetches its projector during the load; with vision off
                # it may still be audio-only, which the backend keeps.
                remote = await asyncio.to_thread(
                    _remote_projector_bytes, model_path, getattr(request, "hf_token", None)
                )
                if remote is None:
                    # The loader retries its own lookup and may still fetch one: unsized.
                    self._projector_url = True
                    remote = 0
                sidecars += remote
                auto_bytes += remote
                if disable_vision:
                    self._uncertain_extra = float(remote) * _MMPROJ_VRAM_SAFETY
                self._has_projector = self._has_projector or remote > 0
        # Files named through the environment (LLAMA_ARG_MMPROJ) load all the same; a drafter there
        # either kept the load local or is scrubbed by the launch.
        named_files = ss.launch_files_from(extras)
        env_files = ss.env_launch_files(
            {k: v for k, v in os.environ.items() if k not in _DRAFTER_ENV + _HF_DRAFT_ENV}
        )
        replaced = (os.environ.get("LLAMA_ARG_MMPROJ") or "").strip()
        replaced = {os.path.abspath(replaced)} if replaced and not env_projector else set()
        for path in dict.fromkeys(env_files):
            if path in named_files or path in replaced:
                continue
            file_size = ss.gguf_size_bytes(path)
            if file_size is None:
                try:
                    file_size = os.path.getsize(path)
                except OSError:
                    file_size = 0
            sidecars += int(file_size)
        if extra_args_disable_mmproj(extras):
            # --no-mmproj suppresses a projector the args name.
            named_mm = ss.sidecar_files(_flag_args(extras, frozenset({"--mmproj", "-mm"})))
            dropped += [p for p in named_mm if p not in dropped]
        for path in dropped:
            sidecars -= int(ss.gguf_size_bytes(path) or 0)
        # A GPU projector's encoder allocates about 1.3x its file at runtime.
        projectors = (
            []
            if extra_args_disable_mmproj(extras)
            else ss.sidecar_files(_flag_args(extras, frozenset({"--mmproj", "-mm"})))
        )
        if env_projector:
            projectors.append(os.path.abspath(env_projector))
        loaded = [p for p in dict.fromkeys(projectors) if p not in dropped]
        self._projector_file = loaded[0] if loaded else auto_projector if auto_bytes else None
        named_bytes = sum(
            int(ss.gguf_size_bytes(p) or 0) for p in dict.fromkeys(projectors) if p not in dropped
        )
        sidecars += (_MMPROJ_VRAM_SAFETY - 1) * (named_bytes + auto_bytes)
        # The headroom above the file is an estimate, never proven freed by a replacement.
        self._uncertain_extra += (_MMPROJ_VRAM_SAFETY - 1) * (named_bytes + auto_bytes)
        if host_projector:
            # Charged for unified memory, but a discrete GPU never held it: never freed.
            self._uncertain_extra = _MMPROJ_VRAM_SAFETY * (named_bytes + auto_bytes)
            self._host_projector = _MMPROJ_VRAM_SAFETY * (named_bytes + auto_bytes)
        if size is None:
            size = await asyncio.to_thread(
                ss.remote_gguf_size_bytes, model_path, variant, getattr(request, "hf_token", None)
            )
        ctx, cache_type, cache_type_v = ss.effective_kv_settings(request, extras)
        from core.inference.llama_cpp import _VALID_CACHE_TYPES

        requested = str(getattr(request, "cache_type_kv", None) or "").strip()
        if requested.lower() not in _VALID_CACHE_TYPES:
            requested = ""  # the loader drops a type it cannot emit and keeps the inherited one
        if requested:
            # The loader emits the request's type over an inherited env one; only extras beat it.
            if not ss._last_operand(extras, ss._CACHE_TYPE_K_FLAGS):
                cache_type = requested
            if not ss._last_operand(extras, ss._CACHE_TYPE_V_FLAGS):
                cache_type_v = requested
        from core.inference.llama_server_args import fit_ctx_in

        argv_ctx = ss._last_operand(extras, ss._CTX_SIZE_FLAGS)
        env_ctx = (os.environ.get("LLAMA_ARG_CTX_SIZE") or "").strip()
        self._native_ctx = argv_ctx == "0" or (
            not argv_ctx and not getattr(request, "max_seq_length", None) and env_ctx == "0"
        )
        if not ctx:
            # llama-server takes the context from its environment when neither argv nor the
            # request sets one.
            try:
                ctx = max(0, int((os.environ.get("LLAMA_ARG_CTX_SIZE") or "0").strip()))
            except ValueError:
                ctx = 0
        floor = fit_ctx_in(extras, os.environ)
        if not ctx and floor is not None and floor < 0:
            self._native_ctx = True  # llama.cpp reads a negative floor as the whole context
        if not ctx and floor and floor > 0:
            ctx = floor  # the fitter never goes below --fit-ctx
        if not ctx and not self._native_ctx:
            # Auto context: the loader opens the native context and llama.cpp's fitter shrinks it
            # no further than its default --fit-ctx.
            native = header.get(f"{header.get('general.architecture')}.context_length") or 0
            ctx = min(4096, native) if native else 4096
        kv = None
        self._unsized_kv = False
        from core.inference.llama_cpp import (
            _kv_cache_cell_layout,
            _kv_unified_from_args,
            _named_batch_sizes,
            _planned_flash_attn_state,
        )

        # Studio passes --kv-unified for several slots; --no-kv-unified gives each its own
        # padded stream.
        unified = _kv_unified_from_args(extras, default = n_parallel > 1)
        _slots, streams, per_stream = _kv_cache_cell_layout(ctx, n_parallel, unified)
        cells = streams * per_stream if ctx else 0
        from core.inference.llama_cpp import (
            _batch_ubatch_for_mmproj,
            _emitted_n_batch,
            _mmproj_required_ubatch,
            _unknown_projector_ubatch,
        )

        n_batch, n_ubatch = getattr(request, "n_batch", None), getattr(request, "n_ubatch", None)
        if self._has_projector and not disable_vision:
            # The loader raises an unset micro-batch to what the projector's images need.
            arch = header.get("general.architecture")
            required = (
                _mmproj_required_ubatch(
                    self._projector_file, header.get(f"{arch}.embedding_length"), extras, os.environ
                )
                if self._projector_file
                else _unknown_projector_ubatch(extras, os.environ)
            )
            n_batch, n_ubatch = _batch_ubatch_for_mmproj(
                required, n_batch, n_ubatch, extras, os.environ
            )
        n_batch = _emitted_n_batch(n_batch, n_parallel)
        batch, ubatch, _nb, _nu = _named_batch_sizes(extras, os.environ, n_batch, n_ubatch)
        header_kv = _REMOTE_HEADER_KV.pop((model_path, str(variant)), None)
        k_type, v_type = cache_type or "f16", cache_type_v or cache_type or "f16"
        # Either quantized axis allocates dequantization scratch: price the lighter one.
        lighter = min((k_type, v_type), key = ss.kv_bytes_per_elem)
        binary = ss.llama_server_binary()
        try:
            supports_fa = not binary or ss.llama_server_supports("--flash-attn", binary)
        except Exception:
            supports_fa = True
        flash_attn = _planned_flash_attn_state(
            extras,
            planned_cache_types = (k_type, v_type),
            supports_flash_attn = supports_fa,
            architecture = header.get("general.architecture"),
            env = os.environ,
        )
        kv_v = cache_type_v
        if not flash_attn and ss.kv_bytes_per_elem(v_type) < 2.0:
            kv_v = "f16"  # without flash attention a quantized V cache starts as f16 (loader retry)
        loader_kv = 0
        self._kv_layer_frac = 0.0
        from core.inference.llama_cpp import _swa_full_from_args_or_env

        swa_full = bool(_swa_full_from_args_or_env(extras))
        if cells and size:
            # The loader's own estimator first; one cache type, so the heavier axis.
            loader_kv = await asyncio.to_thread(
                _ctx_compute_bytes,
                local_file,
                header_kv,
                ctx,
                min(batch, ubatch),
                max((k_type, kv_v or v_type), key = ss.kv_bytes_per_elem),
                flash_attn,
                n_parallel,
                "kv",
                kv_unified = unified,
                swa_full = swa_full,
            )
            kv = loader_kv or _kv_from_scalars(header, cells, cache_type, kv_v, ss)
            if loader_kv:
                # Whole layers land on a device: a full-context layer among sliding-window ones
                # carries far more than the average.
                peak = await asyncio.to_thread(
                    _ctx_compute_bytes,
                    local_file,
                    header_kv,
                    ctx,
                    min(batch, ubatch),
                    max((k_type, kv_v or v_type), key = ss.kv_bytes_per_elem),
                    flash_attn,
                    n_parallel,
                    "kv_peak_ppm",
                    kv_unified = unified,
                    swa_full = swa_full,
                )
                self._kv_layer_frac = (peak or 0) / 1_000_000
        if kv is None and local_file and cells:
            kv = await asyncio.to_thread(ss.estimate_kv_bytes, local_file, cells, cache_type, kv_v)
        compute = await asyncio.to_thread(
            _ctx_compute_bytes,
            local_file,
            header_kv,
            ctx,
            min(batch, ubatch),
            lighter,
            flash_attn,
            n_parallel,
        )
        if ctx and size and compute is None:
            self._unsized_kv = True
        self._ctx_compute = float(compute or 0)
        # Context checkpoints are host RAM, the same pool on unified memory: charged to this
        # machine, never counted as freed.
        per_checkpoint = 0
        if ctx:
            per_checkpoint = await asyncio.to_thread(
                _ctx_compute_bytes,
                local_file,
                header_kv,
                ctx,
                min(batch, ubatch),
                max((k_type, kv_v or v_type), key = ss.kv_bytes_per_elem),
                flash_attn,
                n_parallel,
                "checkpoint",
                swa_full = swa_full,
            )
        # A recurrent checkpoint is one slot's state; the loader's probe misses gated-delta layers.
        per_checkpoint = max(per_checkpoint or 0, _ssm_state_bytes(header, 1))
        # The count the loader will launch with: capability, arguments, and its host budget.
        count = await asyncio.to_thread(
            _effective_checkpoints,
            binary,
            extras,
            getattr(request, "ctx_checkpoints", None),
            int(per_checkpoint),
            n_parallel,
        )
        self._checkpoints = float(per_checkpoint * count * max(1, n_parallel))
        # The loader's probe covers Mamba and KDA layouts; the header formula, gated-delta ones.
        loader_recurrent = await asyncio.to_thread(
            _ctx_compute_bytes,
            local_file,
            header_kv,
            ctx,
            min(batch, ubatch),
            cache_type,
            flash_attn,
            n_parallel,
            "recurrent",
        )
        recurrent = max(loader_recurrent or 0, _ssm_state_bytes(header, n_parallel))
        arch = header.get("general.architecture")
        n_rec = _recurrent_layers(header)
        # A whole recurrent layer carries its share of the state, not the model-wide average.
        self._recurrent_layer = float(recurrent) / n_rec if recurrent and n_rec else 0.0
        self._recurrent = float(recurrent)
        zeros = header.get(f"{arch}.attention.head_count_kv.zeros") or 0
        interval = int(header.get(f"{arch}.full_attention_interval") or 0)
        if zeros and n_rec:
            self._attention_layers = max(1, int(header.get(f"{arch}.block_count") or 0) - zeros)
        elif interval > 1:
            self._attention_layers = max(1, int(header.get(f"{arch}.block_count") or 0) // interval)
        else:
            self._attention_layers = 0

        if kv is None and ctx and size and not recurrent:
            self._unsized_kv = True  # a pure state-space model has no attention KV
        # The loader's KV already holds the recurrent state it can see; add only what it missed.
        missed = max(0, recurrent - (loader_recurrent or 0)) if loader_kv else recurrent
        kv = float(kv or 0.0) + missed
        return (None if size is None else float(size) + adapters), kv, float(sidecars)

    async def after_load(self, llama_backend: Any) -> None:
        self.in_flight = False
        if self._passthrough:
            self._passthrough = False
            if not self._alongside:
                self.forget_served()  # an engine load replaced the primary; that was not ours
            return
        self._pre_load = None
        self._cancel_event = None
        self._backend = llama_backend
        self._served_model = self._requested_model
        self._started = set()
        plan = self.plan
        local_count = sum(1 for d in plan.devices if not d.remote) if plan is not None else 0
        # A local fit on several GPUs sends no device list, so which of them hold it is unknown.
        enforced = plan is not None and (plan.topology == "layer_split" or local_count <= 1)
        self._resident = plan.resident() if enforced else {}
        argv = [
            str(a) for a in (getattr(getattr(llama_backend, "_process", None), "args", None) or [])
        ]
        if (
            self.block
            and argv[-len(self.block) :] != self.block
            and not _contains(argv, self.block)
        ):
            self.topology = "single"
            self.reason = "llama-server launched without the planned split"
            self.plan, self.block, self._resident = None, [], {}
        managed = {n.endpoint for n in self.nodes if n.managed}
        in_use = set(self.block[self.block.index("--rpc") + 1].split(",")) if self.block else set()
        down, self._down = self._down, set()
        for endpoint in [
            e for e in self.workers if (e not in managed or e in down) and e not in in_use
        ]:
            # Dropped from the config (or made unmanaged), or hung while its ssh lives on; the load
            # that just committed does not use it, so it can go now and a later load restarts it.
            process = self.workers.pop(endpoint)
            try:
                await process.stop()
            except Exception:
                logger.warning("cluster serving: stopping %s failed", endpoint, exc_info = True)

    async def load_failed(self) -> None:
        self.in_flight = False
        if self._passthrough:
            self._passthrough = False
            backend_loaded = self._backend is not None and getattr(
                self._backend, "is_loaded", False
            )
            if not self._alongside and not backend_loaded:
                self.forget_served()  # the engine load unloaded our split before failing
            return
        self._cancel_event = None
        pre, self._pre_load = self._pre_load, None
        if self._backend is not None and getattr(self._backend, "is_loaded", False):
            if pre is not None:
                self.topology, self.reason, self.plan, self.block = pre
                if self.block:
                    _remember_block(self.block)  # failed replacements may have pushed it out
        else:
            # The failed load got as far as tearing the old one down.
            self.forget_served()

    def _serving_split(self) -> bool:
        return bool(
            self._pre_load
            and self._pre_load[3]
            and self._backend is not None
            and getattr(self._backend, "is_loaded", False)
        )

    def serving(self) -> bool:
        """Whether a split this planned is still what the primary serves, config or not."""
        return bool(self.workers) or (
            self._backend is not None and bool(getattr(self._backend, "is_loaded", False))
        )

    async def failed_while_retiring(self) -> None:
        """A replacement after the cluster was turned off failed; stop the workers if the split
        was already unloaded, keep it (and them) if it still serves."""
        if not (self._backend is not None and getattr(self._backend, "is_loaded", False)):
            self.forget_served()
            await self.shutdown()

    def note_internal_load(self) -> None:
        """A load that bypasses planning is about to start: forget the split, but keep what is
        needed to put the record back if that load is refused before it replaces the primary."""
        backend = self._backend
        if backend is not None and getattr(backend, "is_loaded", False) and self.block:
            self._internal_pre = (
                self.topology,
                self.reason,
                self.plan,
                list(self.block),
                backend,
                getattr(backend, "_process", None),
                dict(self._resident),
            )
        self.forget_served()

    def settle_internal_load(self) -> None:
        """Put the split record back when the bypassing load never replaced the primary."""
        pre, self._internal_pre = self._internal_pre, None
        if pre is None or self._backend is not None:
            return
        topology, reason, plan, block, backend, process, resident = pre
        if getattr(backend, "is_loaded", False) and getattr(backend, "_process", None) is process:
            self.topology, self.reason, self.plan, self.block = topology, reason, plan, block
            self._backend, self._resident = backend, resident
            _remember_block(block)

    def forget_served(self) -> None:
        """A load that bypassed planning replaced the served split; its workers can stay."""
        self.topology, self.reason = "single", "a load that bypassed the planner replaced the split"
        self.plan, self.block = None, []
        self._backend, self._resident = None, {}

    async def shutdown(self) -> None:
        for endpoint, process in list(self.workers.items()):
            try:
                await process.stop()
            except Exception:
                logger.warning("cluster serving: stopping %s failed", endpoint, exc_info = True)
        self.workers.clear()
        self._started = set()
        self.retire_on_commit = False
        self.topology, self.block = "single", []
        self._backend, self._resident = None, {}

    def status(self) -> Dict[str, Any]:
        if not self.nodes and not self.config_error:
            self.refresh_nodes()
        return {
            "enabled": True,
            "mode": "cluster",
            "topology": self.topology,
            "reason": self.reason,
            "config_error": self.config_error or None,
            "nodes": [
                {
                    "host": n.host,
                    "rpc_port": n.rpc_port,
                    "managed": n.managed,
                    "worker": (
                        self.workers[n.endpoint].snapshot() if n.endpoint in self.workers else None
                    ),
                }
                for n in self.nodes
            ],
            "plan": self.plan.snapshot() if self.plan is not None else None,
            "split_args": list(self.block) or None,
            "last_error": self.last_error,
        }


def _contains(argv: List[str], block: List[str]) -> bool:
    n = len(block)
    return any(argv[i : i + n] == block for i in range(0, max(0, len(argv) - n) + 1))


_STATE: Optional[ClusterServing] = None


def state() -> ClusterServing:
    global _STATE
    if _STATE is None:
        _STATE = ClusterServing()
    return _STATE


def reset_for_tests() -> None:
    global _STATE
    _STATE = None
    _MANAGED_BLOCKS.clear()
