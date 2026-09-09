# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two-Spark serving orchestrator: picks a topology and runs the peer half of it.

On a DGX Spark with a cabled, configured peer a GGUF load becomes one of ``single`` (one
llama-server here), ``replicas`` (a second llama-server on the peer over ssh with
``SparkRouter`` in front of both), or ``layer_split`` (``ggml-rpc-server`` on the peer and
``--rpc <peer>:<port> --device RPC0,CUDA0 -sm layer`` here). ``recommend_topology`` in
``spark_cluster`` decides; this module gathers its inputs and runs the processes.

The llama-server it probes is resolved by the backend's own ``_find_llama_server_binary``,
never by a search of this module's own, and the rpc-server is taken from beside it, so both
ends of the link are one build. ``enabled()`` is the gate every entry point calls first and
is False off a paired Spark; every ssh runs as an asyncio subprocess or in a worker thread,
so the Studio event loop never waits on the peer.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import getpass
import fnmatch
import glob
import importlib.util
import logging
import os
import os.path as osp
import re
import shlex
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from core.inference.spark_router import CONVERSATION_FIELD, Backend, SparkRouter

logger = logging.getLogger(__name__)

ENV_TOGGLE = "UNSLOTH_SPARK_SERVING"  # "0" disables
ENV_TOPOLOGY = "UNSLOTH_SPARK_TOPOLOGY"
ENV_PEER = "UNSLOTH_SPARK_PEER"  # peer address override (tests, unusual cabling)
ENV_RPC_BIND = "UNSLOTH_SPARK_RPC_BIND"  # ggml-rpc-server -H on the peer
ENV_PREFILL_HEAVY = "UNSLOTH_SPARK_PREFILL_HEAVY"  # "1": the work is long-prompt prefill
ENV_PIPELINE_GROUPS = "UNSLOTH_SPARK_PIPELINE_GROUPS"  # layer split only; 0 or unset adds nothing
# "0" launches without speculative decoding unless the caller owns it; see mtp_plan.
ENV_MTP = "UNSLOTH_SPARK_MTP"

TOPOLOGIES = ("single", "replicas", "layer_split")
RPC_PORT_DEFAULT = 50052
PROMPT_TOKENS_DEFAULT = 512  # the planner's measured table is keyed by prompt length
# Only added when the bundle's llama-server has the flag (unslothai/llama.cpp PR #187).
PIPELINE_GROUPS_DEFAULT = 2
PIPELINE_GROUPS_FLAG = "--pipeline-groups"
# LoadRequest.n_parallel's range, mirrored rather than imported from llama_server_args: this
# module is loaded by the CLI and by tests that never import the backend's request models.
PARALLEL_MIN = 1
PARALLEL_MAX = 64
# Never a second --spec-type: extras that own it switch the backend's whole speculative path
# off, with its memory budget, its sub-3B and MLA gates and its retry without speculation.
# Depth 3 is the mixed-traffic choice: 8 wins at one user and loses at eight. Draft models and
# n-gram stay off, both a loss here from 4 users.
MTP_SPEC_TYPE = "draft-mtp"
MTP_DRAFT_N_MAX = 3
# Depth 3 came from the one-Spark sweep and is the WORST of the three at every row count
# measured on a split; below 32 rows nothing was measured there.
MTP_DRAFT_N_MAX_BY_ROWS = {32: 2, 64: 1}  # spark_cluster.MTP_DRAFT_N_MAX_BY_ROWS
# LAYER SPLIT ONLY: single and replicas keep the one-Spark depths above.
SPLIT_MTP_OFF_ROWS = 64  # spark_cluster.SPLIT_MTP_OFF_ROWS
"""Concurrent rows at or above which a layer split launches with the drafter OFF.

33 to 63 rows is INTERPOLATED; nothing in between was measured. The bounding points are 32
rows, where the best depth (n-max 2) is worth +11.0 percent over the drafter off, 162.9
against 146.8 tok/s, and 64 rows, where the best depth (n-max 1) costs 7.9 percent, 171.7
against 186.3. The constant sits on 64, the lower of the two measured points at which
speculation loses. NOT ``GROUPS_X_MTP_MIN_ROWS`` (16), which is groups against one context.
"""
MTP_OFF_FOR_SPLIT_ROWS = "off for the split rows"
SPEC_TYPE_FLAG = "--spec-type"
SPEC_DRAFT_N_MAX_FLAG = "--spec-draft-n-max"
# What counts as the caller owning speculation: the backend's own rule plus any draft knob.
_SPEC_OWNER_FLAGS = frozenset(
    {
        "--spec-type",
        "--spec-default",
        "--model-draft",
        "-md",
        "--hf-repo-draft",
        "-hfd",
        "-hfrd",
        "--gpu-layers-draft",
        "--n-gpu-layers-draft",
        "-ngld",
        "--device-draft",
        "-devd",
        "--cache-type-k-draft",
        "--cache-type-v-draft",
        "-ctkd",
        "-ctvd",
    }
)
_SPEC_OWNER_PREFIXES = ("--spec-draft-", "--draft")
# Still refused with the groups after PR #187 (tools/server validate_pipeline_groups): one
# projector, one control vector set and one idle timer per server, none of them per group.
_GROUPS_REFUSED_FLAGS = frozenset(
    {
        "--mmproj",
        "-mm",
        "--mmproj-url",
        "-mmu",
        "--control-vector",
        "--control-vector-scaled",
        "--control-vector-layer-range",
        "--sleep-idle-seconds",
    }
)
# Mirrors of spark_cluster.GROUPS_X_MTP_*. The crossover is INTERPOLATED, the geometric
# midpoint of the two bracketing measured points, and every cell carries --kv-unified, which
# is worth 1.27x on a two-group split alone.
GROUPS_X_MTP_MIN_ROWS = 16  # spark_cluster.GROUPS_X_MTP_CROSSOVER_ROWS
GROUPS_X_MTP_OVER_MTP_ONLY = {8: 0.97, 32: 1.36}  # both over one context with MTP
GROUPS_X_MTP_OVER_GROUPS_ONLY = {8: 1.71, 32: 1.09}  # both over two groups alone
SPLIT_TENSOR_SPLIT_EVEN = "0.5,0.5"
# Slots kept, as a fraction of the slots asked for, below which the groups are not worth the
# concurrency the rounding costs. 1/1.4, the measured split speedup from two groups.
GROUPS_WORTH_SLOTS_RATIO = 1.0 / 1.4
# Capped because the throughput table alone says 128 and p90 TTFT says that is unshippable.
# Oversizing is not safe either: a 128-slot server driven at 32 is slower AND worse on TTFT.
SPLIT_ROWS_INTERACTIVE_MAX = 64
SPLIT_ROWS_THROUGHPUT_MAX = 128
SPLIT_CONFIG_BOTH = "groups + speculation"
SPLIT_CONFIG_SPEC = "one context + speculation"
SPLIT_CONFIG_GROUPS = "groups, no speculation"
SPLIT_CONFIG_PLAIN = "one context, no speculation"
HELP_PROBE_TIMEOUT_S = 20.0  # llama-server --help; a hung binary is a missing flag
GROUPS_DRAFTER_PROBE_TIMEOUT_S = 30.0
_GROUPS_REFUSAL_TEXT = "is not supported together with"
_PROBE_MODEL_NAME = "unsloth-spark-pipeline-groups-probe.gguf"
RELAUNCH_BACKOFF_S = (5.0, 15.0, 45.0)  # bounded: three attempts, then the peer stays down
# How long a relaunched peer has to become healthy before the attempt is counted as failed.
# Generous on purpose: this is a whole model load on the peer, the same wait the first launch
# is given (PEER_REPLICA_START_TIMEOUT_S), and being wrong here spends a relaunch attempt on a
# peer that was merely slow.
RELAUNCH_HEALTHY_TIMEOUT_S = 600.0
RELAUNCH_HEALTH_POLL_S = 2.0
PEER_START_TIMEOUT_S = 20.0  # for the rpc-server port to accept; the model load is separate
# A replica is a llama-server, not an rpc-server: it reads the whole model before it binds.
# PEER_START_TIMEOUT_S was being used for it too, which is the case its own comment above
# excludes -- a near-node-capacity GGUF exceeds 20 s from local storage, so the peer was
# killed mid-load and the deployment fell back to one node while the peer was perfectly
# healthy. Matched to the primary's own readiness budget (_wait_for_health, 600 s) so the two
# ends of a replica pair are given the same time to do the same work.
PEER_REPLICA_START_TIMEOUT_S = 600.0
PEER_REUSE_TIMEOUT_S = 3.0
SUPERVISOR_INTERVAL_S = 1.0
_LOG_TAIL = 60
_GIB = 2**30

# Mirrors of spark_cluster's numbers, used only when that module is unavailable.
_SPARK_USABLE_GIB = 121.69
_SERVE_OVERHEAD_GIB = 8.0


_CLUSTER: Any = None
_CLUSTER_LOOKED_UP = False


def _cluster():
    """``studio.spark_cluster``, by path when the package is not importable (the backend
    runs with ``studio/backend`` as its root)."""
    global _CLUSTER, _CLUSTER_LOOKED_UP
    if _CLUSTER_LOOKED_UP:
        return _CLUSTER
    _CLUSTER_LOOKED_UP = True
    try:
        from studio import spark_cluster as module  # type: ignore
        _CLUSTER = module
        return _CLUSTER
    except Exception:
        pass
    path = Path(__file__).resolve().parents[3] / "spark_cluster.py"
    try:
        spec = importlib.util.spec_from_file_location("unsloth_studio_spark_cluster", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLUSTER = module
    except Exception:
        _CLUSTER = None
    return _CLUSTER


def is_spark() -> bool:
    sc = _cluster()
    try:
        return bool(sc is not None and sc.is_dgx_spark())
    except Exception:
        return False


# Rail discovery walks sysfs and forks `ip`, measured at 16.2 ms on a paired Spark, and
# ``enabled()`` is on the status endpoint the UI polls: uncached, every poll of a route
# declared ``async`` spends that on the event loop, stalling whatever generation is
# streaming beside it. ``current_topology`` sidesteps it with a cheap check; the dedicated
# status endpoint cannot, because answering "not enabled" is the whole point of it off a
# Spark. Cached with a TTL rather than for the process lifetime so a cable plugged in after
# startup is still picked up, without a restart, within a minute.
_PEER_DISCOVERY_TTL_S = 60.0
_peer_discovery_cache: Optional[Tuple[float, Optional[str]]] = None


def reset_peer_discovery_cache() -> None:
    """Forget the discovered peer, so the next call re-walks the rails. For tests, and for
    anything that knows the cabling just changed."""
    global _peer_discovery_cache
    _peer_discovery_cache = None


def peer_address() -> Optional[str]:
    override = (os.environ.get(ENV_PEER) or "").strip()
    if override:
        return override
    if not is_spark():
        return None
    global _peer_discovery_cache
    cached = _peer_discovery_cache
    if cached is not None and (time.monotonic() - cached[0]) < _PEER_DISCOVERY_TTL_S:
        return cached[1]
    found = _discover_peer_address()
    _peer_discovery_cache = (time.monotonic(), found)
    return found


def _discover_peer_address() -> Optional[str]:
    sc = _cluster()
    try:
        peer = sc.peer_ip_for()
    except Exception:
        peer = None
    if peer:
        return peer
    try:
        for rail in sc.load_config().get("peer_rails") or []:
            if isinstance(rail, dict) and rail.get("address"):
                return str(rail["address"])
    except Exception:
        pass
    return None


def enabled() -> bool:
    if (os.environ.get(ENV_TOGGLE) or "").strip() == "0":
        return False
    if not is_spark():
        return False
    return peer_address() is not None


def forced_topology() -> Optional[str]:
    value = (os.environ.get(ENV_TOPOLOGY) or "").strip().lower().replace("-", "_")
    return value if value in TOPOLOGIES else None


def looks_like_a_gguf_load(
    model_path: str, variant: Optional[str], local_file: Optional[str]
) -> bool:
    """Whether this request resolves to a GGUF, as far as can be told before the loader says.

    Only a GGUF load can use a peer, and ``before_load`` runs ahead of model classification, so
    a forced topology would otherwise be applied to a Transformers or MLX load: an rpc-server
    started for weights that never touch it, and nothing to detach it afterwards."""
    if local_file:
        return True
    if str(variant or "").strip():
        return True
    text = str(model_path or "").strip().lower()
    if text.endswith(".gguf"):
        return True
    # Uncached repo: the sizing already asked the hub for the variants, so its answer settles
    # this too. None means the lister found no GGUF, not merely that it could not size one.
    return False


def node_budget_bytes() -> float:
    sc = _cluster()
    usable = getattr(sc, "SPARK_USABLE_GIB", _SPARK_USABLE_GIB)
    overhead = getattr(sc, "SERVE_OVERHEAD_GIB", _SERVE_OVERHEAD_GIB)
    return (float(usable) - float(overhead)) * _GIB


def plan_topology(
    model_bytes: Optional[float],
    *,
    users: int,
    kv_bytes_per_user: float = 0.0,
    prompt_tokens: int = PROMPT_TOKENS_DEFAULT,
    per_node_free_bytes: Optional[float] = None,
) -> Dict[str, Any]:
    """``spark_cluster.recommend_topology`` with this module's inputs filled in. An unknown
    model size answers ``single`` with a reason rather than a guess."""
    if model_bytes is None:
        return {
            "topology": "single",
            "reason": "model size unknown; not guessing, serving on this Spark only",
            "speedup": None,
            "users": users,
        }
    sc = _cluster()
    planner = getattr(sc, "recommend_topology", None)
    if not callable(planner):
        return {
            "topology": "single",
            "reason": "spark_cluster.recommend_topology unavailable; serving on this Spark only",
            "speedup": None,
            "users": users,
        }
    free = per_node_free_bytes if per_node_free_bytes is not None else node_budget_bytes()
    prefill_heavy = (os.environ.get(ENV_PREFILL_HEAVY) or "").strip() == "1"
    out = planner(
        float(model_bytes),
        float(kv_bytes_per_user or 0.0),
        int(users),
        int(prompt_tokens),
        float(free),
        prefill_heavy = prefill_heavy,
    )
    out = dict(out)
    out["topology"] = str(out.get("topology", "single")).replace("-", "_")
    if out["topology"] not in TOPOLOGIES:
        out["topology"] = "single"
    return out


def gguf_shard_paths(path: Optional[str]) -> List[str]:
    """Every file a load of ``path`` reads. llama.cpp takes only the first shard on the command
    line and opens the rest itself, so a caller that reasons about one name reasons about the
    whole set (ggml-org/llama.cpp tools/gguf-split/README.md)."""
    if not path:
        return []
    p = Path(path)
    match = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", p.name)
    if not match:
        return [str(p)]
    prefix, _first, count = match.groups()
    return [str(p.with_name(f"{prefix}-{i:05d}-of-{count}.gguf")) for i in range(1, int(count) + 1)]


def gguf_size_bytes(path: Optional[str]) -> Optional[int]:
    """The model's size on disk, or None when it cannot be known from what is here.

    A missing shard answers None rather than the sum of the ones present. The number is the
    input to the topology decision, and None is handled -- ``before_load`` falls through to
    ``remote_gguf_size_bytes`` and asks the hub -- while a short total is not: a half-downloaded
    120 GiB split model would price as 30 GiB, plan ``single``, and be found out only after the
    remaining shards arrive, as an out-of-memory on a node with 121.69 GiB shared between CPU
    and GPU. Undercounting is the one error this function must not make."""
    if not path:
        return None
    try:
        if not Path(path).is_file():
            return None
        total = 0
        for shard in gguf_shard_paths(path):
            try:
                total += Path(shard).stat().st_size
            except OSError:
                return None
        return total
    except OSError:
        return None


def cached_repo_file(model_path: str, variant: Optional[str]) -> Optional[str]:
    """The GGUF a not-yet-resolved load request will use, or None -- in which case the caller
    waits for the resolved ``gguf_path`` after the load rather than guessing."""
    if not model_path:
        return None
    expanded = osp.expanduser(model_path)
    if osp.isfile(expanded):
        return expanded
    if osp.isdir(expanded):
        # A local directory of GGUF variants is what an export produces, and the loader opens
        # one of them. Returning None here made the planner size nothing and answer `single`,
        # so a local export larger than one Spark could never reach the split it needs. Asked
        # of the loader's own lister and picker, so the file sized is the file opened.
        return _local_dir_gguf(expanded, variant)
    if "/" not in model_path or osp.isabs(model_path):
        return None
    cache = os.environ.get("HF_HUB_CACHE") or osp.join(
        os.environ.get("HF_HOME") or osp.expanduser("~/.cache/huggingface"), "hub"
    )
    resolved = _cached_repo_file_via_loader(model_path, variant)
    if resolved is not None:
        return resolved
    root = osp.join(cache, "models--" + model_path.replace("/", "--"), "snapshots")
    if not osp.isdir(root):
        return None
    pattern = f"*{variant}*.gguf" if variant else "*.gguf"
    # Newest snapshot first, like the loader. Lexicographic order is the hash, which says
    # nothing about age, so a stale copy could be sized instead of the one that will load.
    snapshots = sorted(
        (d for d in glob.glob(osp.join(root, "*")) if osp.isdir(d)),
        key = lambda d: _snapshot_mtime(d),
        reverse = True,
    )
    for snapshot in snapshots:
        candidates = sorted(glob.glob(osp.join(snapshot, "**", pattern), recursive = True))
        candidates = [c for c in candidates if not _is_companion_gguf(c)]
        if candidates:
            return candidates[0]
    return None


def _local_dir_gguf(directory: str, variant: Optional[str]) -> Optional[str]:
    """The GGUF the loader would open out of a local directory, or None.

    The loader's own lister and picker, not a glob: a directory holding several quants plus an
    mmproj has to resolve to the same one the load will open, or the plan prices a file nobody
    reads. Falls back to a companion-filtered glob when the loader cannot be imported, which is
    the case this module's tests run in."""
    try:
        from utils.models.model_config import list_local_gguf_variants

        # ``(variants, has_vision)``, the same shape as the remote lister, and the second
        # element is not ours to interpret here.
        variants, _has_vision = list_local_gguf_variants(directory)
        variants = list(variants or [])
    except Exception:
        variants = []
    if variants:
        chosen = _pick_variant(variants, str(variant).strip().casefold() if variant else "")
        if chosen is not None:
            name = str(getattr(chosen, "filename", "") or "")
            candidate = name if osp.isabs(name) else osp.join(directory, osp.basename(name))
            if osp.isfile(candidate):
                return candidate
    pattern = f"*{variant}*.gguf" if variant else "*.gguf"
    found = sorted(
        c
        for c in glob.glob(osp.join(directory, "**", pattern), recursive = True)
        if osp.isfile(c) and not _is_companion_gguf(c)
    )
    return found[0] if found else None


def _snapshot_mtime(directory: str) -> float:
    try:
        return osp.getmtime(directory)
    except OSError:
        return 0.0


# Names of files that live beside the weights and are not the weights: the vision projector,
# the importance matrix, and the MTP/draft companions a repo ships for speculative decoding.
# Sizing one of these plans against a few hundred megabytes instead of the model.
_COMPANION_GGUF_MARKERS = ("mmproj", "imatrix")
_COMPANION_GGUF_PARTS = ("mtp", "dspark", "draft")


def _is_companion_gguf(path: str) -> bool:
    name = osp.basename(path).lower()
    if any(marker in name for marker in _COMPANION_GGUF_MARKERS):
        return True
    # A companion often lives in its own directory rather than carrying the word in the file
    # name, so the path components count too.
    parts = [part.lower() for part in Path(path).parts[:-1]]
    return any(part in _COMPANION_GGUF_PARTS for part in parts)


def _pick_variant(variants: Any, wanted: str) -> Any:
    """The variant a load naming ``wanted`` resolves to, or the one the LOADER would default to.

    Both listers sort largest first, so ``variants[0]`` is a repo's BF16/F16 rather than its
    default. The load does not open that file: it calls ``_pick_best_gguf``, whose preference
    list puts UD-Q4 first. Sizing one file and opening another is not a rounding error in the
    same direction -- a repo whose full precision copy exceeds the pair while its Q4 needs two
    nodes and fits reads as "no topology fits", and the Q4 is then left to a single node launch
    that OOMs. Asked of the loader's own picker rather than reimplemented here, for the same
    reason ``_cached_repo_file_via_loader`` asks the resolver instead of guessing."""
    if not wanted:
        if not variants:
            return None
        try:
            from utils.models.model_config import _pick_best_gguf
            best = _pick_best_gguf([str(getattr(v, "filename", "")) for v in variants])
        except Exception:
            best = None
        if best:
            for info in variants:
                if str(getattr(info, "filename", "")) == best:
                    return info
        return variants[0]
    for info in variants:
        if wanted in (str(info.quant).casefold(), str(info.filename).casefold()):
            return info
    return None


def remote_gguf_size_bytes(
    model_path: str,
    variant: Optional[str],
    hf_token: Optional[str] = None,
) -> Optional[int]:
    """What a not-yet-downloaded GGUF will weigh, from the hub's own file metadata.

    Without a size the planner answers ``single``, which is correct for a guess but is also
    the one topology that cannot hold a model larger than one Spark, so a first load of an
    uncached repo could never start the split it needs: nothing re-plans after the download,
    and the single-node launch it would have to survive first is exactly what does not fit.
    The download happens either way, and this asks for the sizes before it starts, so the
    split is planned in the same request.

    Network, and optional in every sense: offline, gated, rate-limited or simply unavailable
    all fall back to the sizeless answer that was there before.

    ``hf_token`` is the request's own token, and it is not optional in the way the rest of this
    is. A private or gated repo answers the anonymous query with an authorization failure, which
    lands in the ``except`` below as ``None`` and plans ``single`` -- and then the authenticated
    loader downloads the model anyway, so a repo larger than one Spark reaches the one topology
    that cannot hold it. The failure is silent and looks exactly like being offline."""
    if not model_path or "/" not in model_path or osp.isabs(model_path):
        return None
    try:
        from utils.models.model_config import list_gguf_variants
    except Exception:
        return None
    try:
        variants, _has_vision = list_gguf_variants(model_path, hf_token)
        chosen = _pick_variant(variants, str(variant).strip().casefold() if variant else "")
        if chosen is None:
            return None
        size = int(getattr(chosen, "size_bytes", 0) or 0)
        return size or None
    except Exception:
        return None


def _cached_repo_file_via_loader(model_path: str, variant: Optional[str]) -> Optional[str]:
    """The GGUF the backend's own resolver would pick, or None when it cannot be asked.

    Guessing separately from the loader is what goes wrong here: a different snapshot, or a
    companion instead of the weights, is sized and planned against, and then a different file
    is loaded. This asks the resolver that the load itself uses. Imported lazily and inside a
    try, so this module stays importable without the backend, which its tests rely on."""
    try:
        from utils.models.model_config import (
            _iter_hf_cache_snapshots,
            list_local_gguf_variants,
        )
    except Exception:
        return None
    try:
        wanted = str(variant).strip().casefold() if variant else ""
        for snapshot in _iter_hf_cache_snapshots(model_path):
            variants, _has_vision = list_local_gguf_variants(str(snapshot))
            if not variants:
                continue
            chosen = _pick_variant(variants, wanted)
            if chosen is None:
                continue
            path = osp.join(str(snapshot), chosen.filename)
            if osp.isfile(path):
                return path
    except Exception:
        return None
    return None


# f16 when unknown. Kept local so this module stays importable without the backend.
_KV_BYTES_PER_ELEM = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q4_0": 18 / 32,
    "q4_1": 20 / 32,
    "q5_0": 22 / 32,
    "q5_1": 24 / 32,
    "iq4_nl": 18 / 32,
}


def kv_bytes_per_elem(cache_type: Optional[str]) -> float:
    return _KV_BYTES_PER_ELEM.get(str(cache_type or "f16").strip().lower(), 2.0)


def estimate_kv_bytes(
    gguf_path: Optional[str],
    n_ctx: int,
    cache_type: Optional[str] = None,
    cache_type_v: Optional[str] = None,
) -> Optional[int]:
    """KV cache for ``n_ctx`` tokens at ``cache_type``, and ``cache_type_v`` when K and V are
    configured apart, from the GGUF header, or None when the file or the keys cannot be read.
    SWA layers are charged in full, so the estimate errs high for models that have them, which
    is the direction that costs a needless split rather than a node."""
    if not gguf_path or n_ctx <= 0:
        return None
    try:
        from gguf import GGUFReader  # type: ignore

        reader = GGUFReader(gguf_path)
        fields = reader.fields

        def _scalar(key: str) -> Optional[int]:
            field = fields.get(key)
            if field is None:
                return None
            try:
                return int(field.parts[field.data[0]][0])
            except Exception:
                return None

        arch_field = fields.get("general.architecture")
        if arch_field is None:
            return None
        arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8", "replace")
        n_layer = _scalar(f"{arch}.block_count")
        n_head = _scalar(f"{arch}.attention.head_count")
        n_kv = _scalar(f"{arch}.attention.head_count_kv") or n_head
        n_embd = _scalar(f"{arch}.embedding_length")
        head_dim = _scalar(f"{arch}.attention.key_length")
        if head_dim is None and n_head and n_embd:
            head_dim = n_embd // n_head
        if not (n_layer and n_kv and head_dim):
            return None
        per_elem = kv_bytes_per_elem(cache_type) + kv_bytes_per_elem(
            cache_type if cache_type_v is None else cache_type_v
        )
        return int(n_layer * int(n_ctx) * n_kv * head_dim * per_elem)
    except Exception:
        return None


def _ssh_user() -> str:
    """This session's login, which is the peer's too (`provision` mirrors the install as one
    account). ``spark_cluster._ssh_user`` owns the rule; this copy serves a backend that cannot
    load that module, and falls back to the login database when USER is unset."""
    shared = getattr(_cluster(), "_ssh_user", None)
    if callable(shared):
        try:
            return str(shared())
        except Exception:
            pass
    for var in ("USER", "USERNAME", "LOGNAME"):
        value = os.environ.get(var)
        if value:
            return value
    try:
        return getpass.getuser()
    except Exception:
        return "nvidia"


def ssh_argv(
    peer: str,
    remote: str,
    *,
    connect_timeout: int = 8,
    keepalive: bool = False,
) -> List[str]:
    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"ConnectTimeout={connect_timeout}",
    ]
    if keepalive:
        argv += ["-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3"]
    argv += [f"{_ssh_user()}@{peer}", remote]
    return argv


async def ssh_run(
    peer: str,
    remote: str,
    *,
    timeout: float = 20.0,
) -> Tuple[int, str, str]:
    """Run one command on the peer without blocking the loop. rc 255 on transport failure."""
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh_argv(peer, remote),
            stdout = asyncio.subprocess.PIPE,
            stderr = asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout = timeout)
        return (
            proc.returncode if proc.returncode is not None else 255,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
        )
    except (asyncio.TimeoutError, OSError) as exc:
        # Timing out abandons the child rather than ending it, and an unreachable peer is
        # exactly when this fires, on a supervisor loop: without the kill and the wait it
        # leaks one live ssh per probe and then one zombie per probe.
        if proc is not None:
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass
            try:
                await proc.wait()
            except (OSError, asyncio.CancelledError):
                pass
        return 255, "", str(exc)


def peer_path(path: Path) -> str:
    """``path`` as the peer's shell should expand it. ``$HOME/`` and not ``~/``, because the
    remote checks quote their paths and a quoted tilde does not expand."""
    sc = _cluster()
    relative = getattr(sc, "_peer_relative_path", None)
    if callable(relative):
        try:
            text = str(relative(path))
            return "$HOME/" + text[2:] if text.startswith("~/") else text
        except Exception:
            pass
    try:
        return "$HOME/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return path.as_posix()


def peer_binary_candidates(local_binary: Optional[str], name: str) -> List[str]:
    """Where ``name`` should be on the peer, most likely first: the local binary's own
    directory, since the pair is provisioned by rsync and layouts match.

    The peer is launched over ssh with no ``LD_LIBRARY_PATH``, so the bundle there has to
    resolve its own shared objects. The shipped prebuilt does, through an ``$ORIGIN`` RUNPATH,
    and a bundle rsynced by ``spark provision`` keeps that property. A hand-built llama.cpp
    whose RUNPATH is an absolute path into its build tree does not, and fails on the peer with
    "error while loading shared libraries" while working locally, where that path exists.

    Forwarding this process's ``LD_LIBRARY_PATH`` to the peer would NOT be a fix. The peer's
    layout is not this node's, and an absolute directory that happens to exist on both with a
    different build in it would shadow the peer's correct libraries and put the two ends of the
    RPC link on mismatched builds -- the defect `d5d384c` exists to prevent, and one that fails
    silently as a malformed HELLO rather than loudly. A missing .so is the better failure: it
    names itself in the peer log. So the limitation is documented and not papered over; build a
    bundle with a relocatable RUNPATH, or provision it, and the peer resolves it."""
    out: List[str] = []
    if local_binary:
        out.append(peer_path(Path(local_binary).parent) + "/" + name)
    sc = _cluster()
    bundle = None
    try:
        bundle = sc.llama_bundle_dir() if sc is not None else None
    except Exception:
        bundle = None
    if bundle is not None:
        for parts in (("build", "bin"), ("bin",), ()):
            out.append(peer_path(bundle.joinpath(*parts) if parts else bundle) + "/" + name)
    out.append("$HOME/.unsloth/llama.cpp/build/bin/" + name)
    out.append("$HOME/llamacpp-rpc/bin/" + name)
    out.append(name)
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def find_binary_script(candidates: List[str]) -> str:
    """A shell snippet that prints the first executable candidate, or MISSING."""
    checks = " ".join(
        f'if [ -x "{c}" ]; then echo "{c}"; exit 0; fi;'
        if "/" in c
        else f"if command -v {c} >/dev/null 2>&1; then command -v {c}; exit 0; fi;"
        for c in candidates
    )
    return checks + " echo MISSING; exit 1"


# Sidecar weights: resident alongside the model, and every one of them is a file the replica
# needs at the same path. The scaled forms take a comma-separated list with an optional
# trailing :SCALE per entry (llama.cpp common_arg), which a raw token test sees as one string
# that is neither a path nor a file, so it checks nothing.
_SIDECAR_FLAGS = frozenset(
    {
        "--mmproj",
        "-mm",
        "--model-draft",
        "-md",
        "--lora",
        "--lora-scaled",
        "--control-vector",
        "--control-vector-scaled",
    }
)


# A file operand that is NOT a sidecar weight: one plain path, no comma list and no :SCALE, so
# it is absolutised and preflighted but never taken apart or charged as resident memory. It has
# to be here because llama.cpp resolves it relative to the server's working directory, and ssh
# starts the replica in the peer's login directory: a relative template either is not there,
# and the peer takes the whole startup window before falling back to one node, or a file of the
# same name IS there and the replica comes up healthy formatting prompts differently from the
# primary. The env twin, LLAMA_ARG_CHAT_TEMPLATE_FILE, is already absolutised in
# _REPLICA_ENV_PATHS -- one setting, two routes, and only one of them was covered.
_TEMPLATE_FILE_FLAGS = frozenset({"--chat-template-file"})


def template_files(args: Sequence[str], *, cwd: Optional[str] = None) -> List[str]:
    """Every template file ``args`` names, resolved the way llama-server will resolve it."""
    base = cwd or os.getcwd()
    out: List[str] = []
    tokens = [str(a) for a in args]
    for index, token in enumerate(tokens):
        name, sep, inline = token.partition("=")
        if name not in _TEMPLATE_FILE_FLAGS:
            continue
        value = (inline if sep else (tokens[index + 1] if index + 1 < len(tokens) else "")).strip()
        if value:
            out.append(value if osp.isabs(value) else osp.join(base, value))
    return out


def _looks_like_a_scale(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def sidecar_operand_paths(value: str) -> List[str]:
    """The files one sidecar operand names, with the list and ``:SCALE`` forms taken apart."""
    out: List[str] = []
    for piece in str(value).split(","):
        piece = piece.strip()
        if not piece:
            continue
        head, sep, tail = piece.rpartition(":")
        if sep and head and _looks_like_a_scale(tail):
            piece = head
        out.append(piece)
    return out


def sidecar_files(args: Sequence[str], *, cwd: Optional[str] = None) -> List[str]:
    """Every sidecar weight ``args`` names, resolved the way the launch will resolve it.

    Relative against the primary's working directory, because that is what llama-server does
    with them and a replica preflight that skipped them reported success and then watched the
    peer fail to launch for a missing file."""
    base = cwd or os.getcwd()
    out: List[str] = []
    tokens = [str(a) for a in args]
    for index, token in enumerate(tokens):
        name, sep, inline = token.partition("=")
        if name not in _SIDECAR_FLAGS:
            continue
        value = inline if sep else (tokens[index + 1] if index + 1 < len(tokens) else "")
        for path in sidecar_operand_paths(value):
            out.append(path if osp.isabs(path) else osp.join(base, path))
    return out


def sidecar_bytes(args: Sequence[str], *, cwd: Optional[str] = None) -> Tuple[int, bool]:
    """``(bytes, unknown)`` for the sidecars ``args`` names.

    They are resident for the whole load, so charging only the base GGUF understates the
    node's memory, and understating is the direction that plans ``single`` for something that
    then does not fit. ``unknown`` is True when a named sidecar cannot be sized, which the
    caller must not read as zero."""
    total = 0
    unknown = False
    for path in sidecar_files(args, cwd = cwd):
        size = gguf_size_bytes(path)
        if size is None:
            try:
                size = osp.getsize(path)
            except OSError:
                size = None
        if size is None:
            unknown = True
            continue
        total += int(size)
    return total, unknown


_CTX_SIZE_FLAGS = ("--ctx-size", "-c")
_CACHE_TYPE_K_FLAGS = ("--cache-type-k", "-ctk")
_CACHE_TYPE_V_FLAGS = ("--cache-type-v", "-ctv")


def _last_operand(args: Sequence[str], flags: Sequence[str]) -> Optional[str]:
    """The last operand any of *flags* carries in *args*, or None. Last wins, as llama.cpp does."""
    tokens = [str(a) for a in args or ()]
    found: Optional[str] = None
    for index, token in enumerate(tokens):
        name, sep, inline = token.partition("=")
        if name not in flags:
            continue
        value = inline if sep else (tokens[index + 1] if index + 1 < len(tokens) else "")
        if str(value).strip():
            found = str(value).strip()
    return found


def effective_kv_settings(
    request: Any, extras: Optional[Sequence[str]] = None
) -> Tuple[int, Optional[str], Optional[str]]:
    """``(context, cache_type_k, cache_type_v)`` this load will ACTUALLY run with.

    Pricing the request's first-class fields alone is pricing a configuration the load does not
    use. The pass-through block is appended after the managed flags and llama.cpp is last-wins,
    so a `--ctx-size 131072` or a `-ctk q4_0` in the extras is what the server allocates against,
    and the KV is where the difference is large enough to change the topology: a model whose
    weights fit one Spark but whose real cache does not was planned `single` and then spilled.

    K and V are read apart. Studio leaves the cache types in the ENVIRONMENT rather than
    materialising them into argv -- the same fact ``replica_env`` exists for -- so the env is
    consulted for those two and the extras override it. The context is not read from the
    environment: the managed argv sets `-c` from ``max_seq_length`` and beats
    ``LLAMA_ARG_CTX_SIZE``, which llama.cpp applies before argv. Pricing V as K understates an
    asymmetric cache by up to 4x, which is the direction that OOMs a node."""
    ctx = int(getattr(request, "max_seq_length", None) or 0)
    from_extras = _last_operand(extras or (), _CTX_SIZE_FLAGS)
    if from_extras:
        try:
            ctx = int(from_extras)
        except ValueError:
            pass  # llama-server will refuse it; nothing to price here

    def _cache(flags: Sequence[str], env_name: str) -> Optional[str]:
        value = _last_operand(extras or (), flags)
        if value:
            return value
        value = str(os.environ.get(env_name) or "").strip()
        if value:
            return value
        return getattr(request, "cache_type_kv", None)

    return (
        max(0, ctx),
        _cache(_CACHE_TYPE_K_FLAGS, "LLAMA_ARG_CACHE_TYPE_K"),
        _cache(_CACHE_TYPE_V_FLAGS, "LLAMA_ARG_CACHE_TYPE_V"),
    )


def launch_files(argv: List[str], gguf_path: str) -> List[str]:
    """Every file the launch reads; the replica needs all of them at the same path. argv names
    only the first shard, so expand it: a peer holding just that one passes preflight and then
    fails the load."""
    files = gguf_shard_paths(gguf_path)
    seen = set(files)
    for arg in argv[1:]:
        if arg not in seen and osp.isabs(arg) and osp.isfile(arg):
            files.append(arg)
            seen.add(arg)
    # The operand forms above never survive that test, so they are taken apart separately.
    for path in list(sidecar_files(argv[1:])) + list(template_files(argv[1:])):
        if path not in seen and osp.isfile(path):
            files.append(path)
            seen.add(path)
    return files


# Node-local: the slot KV save path is this node's disk, and llama-server refuses a path
# that does not exist.
_REPLICA_DROPPED_FLAGS = ("--port", "--host", "--slot-save-path")


def absolute_sidecar_operand(value: str, *, cwd: Optional[str] = None) -> str:
    """One sidecar operand with every relative path in it made absolute, forms preserved.

    Keeps the comma-separated list and the trailing ``:SCALE`` exactly as llama.cpp's
    ``common_arg`` parses them, so only the path part moves."""
    base = cwd or os.getcwd()
    pieces: List[str] = []
    for piece in str(value).split(","):
        stripped = piece.strip()
        if not stripped:
            pieces.append(piece)
            continue
        head, sep, tail = stripped.rpartition(":")
        if sep and head and _looks_like_a_scale(tail):
            path, scale = head, ":" + tail
        else:
            path, scale = stripped, ""
        pieces.append((path if osp.isabs(path) else osp.join(base, path)) + scale)
    return ",".join(pieces)


def replica_argv(
    local_argv: List[str],
    *,
    binary: str,
    host: str,
    port: int,
    cwd: Optional[str] = None,
) -> List[str]:
    """The local launch with only the binary, host, port and sidecar paths changed: a replica
    differing in any other flag would answer the same request differently.

    Sidecar operands are absolutised against this process's working directory, the same base
    ``sidecar_files`` resolves them against for the preflight. Without it the two disagreed: the
    preflight checked ``/cwd/adapter.gguf`` and passed, then the peer was handed the bare
    ``adapter.gguf`` over ssh, which resolves against the login directory there, and the launch
    died looking for a file the preflight had just confirmed. The failure surfaced as a
    fall-back to ``single`` reporting that the peer "did not take host:port", which names
    neither the file nor the reason."""
    base = cwd or os.getcwd()

    def _absolute_plain(value: str) -> str:
        text = str(value).strip()
        return text if (not text or osp.isabs(text)) else osp.join(base, text)

    out: List[str] = [binary]
    skip = 0
    pending_sidecar = False
    pending_template = False
    for arg in local_argv[1:]:
        if skip:
            skip -= 1
            continue
        if pending_sidecar:
            pending_sidecar = False
            out.append(absolute_sidecar_operand(arg, cwd = cwd))
            continue
        if pending_template:
            pending_template = False
            out.append(_absolute_plain(arg))
            continue
        if arg in _REPLICA_DROPPED_FLAGS:
            skip = 1
            continue
        if arg.startswith(tuple(f"{flag}=" for flag in _REPLICA_DROPPED_FLAGS)):
            continue
        name, sep, inline = arg.partition("=")
        if name in _SIDECAR_FLAGS:
            if sep:
                out.append(f"{name}={absolute_sidecar_operand(inline, cwd = cwd)}")
            else:
                out.append(arg)
                pending_sidecar = True
            continue
        if name in _TEMPLATE_FILE_FLAGS:
            # One plain path: no comma list and no :SCALE, so the sidecar splitter is not used
            # on it. A template path that happens to end in ":<number>" would otherwise be cut.
            if sep:
                out.append(f"{name}={_absolute_plain(inline)}")
            else:
                out.append(arg)
                pending_template = True
            continue
        out.append(arg)
    out += ["--host", host, "--port", str(port)]
    return out


# llama.cpp's common_arg reads every LLAMA_ARG_* itself, so a setting can reach the primary
# without ever appearing in its argv, and copying only the argv leaves the peer on defaults.
_REPLICA_ENV_PREFIX = "LLAMA_ARG_"
# The two the replica must NOT inherit: it is deliberately given a different endpoint.
_REPLICA_ENV_DENY = frozenset({"LLAMA_ARG_HOST", "LLAMA_ARG_PORT"})
# Normalised the way the backend normalises them before spawning, because llama.cpp compares
# the raw string against ggml_type_name and throws on anything else, whitespace included.
_REPLICA_ENV_LOWERED = frozenset({"LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V"})
# The env twins of _SIDECAR_FLAGS: llama.cpp reads the same file from either route, so a path
# that has to be absolutised in the argv has to be absolutised here too, or the fix only covers
# whichever route this particular caller happened to use. Taken from the server's own --help,
# which prints the env name beside each flag, rather than guessed from the flag spelling.
# Deliberately not here: LLAMA_ARG_MMPROJ_URL and LLAMA_ARG_SPEC_DRAFT_HF_REPO name a URL and a
# repo id, not a local file, and LLAMA_ARG_MODEL* / SSL / API_KEY_FILE are already refused
# outright by DENIED_ENV_VARS below.
_REPLICA_ENV_PATHS = frozenset(
    {
        "LLAMA_ARG_MMPROJ",
        "LLAMA_ARG_SPEC_DRAFT_MODEL",
        "LLAMA_ARG_CHAT_TEMPLATE_FILE",
    }
)


def replica_env(
    source: Optional[Dict[str, str]] = None, *, cwd: Optional[str] = None
) -> Dict[str, str]:
    """The llama.cpp settings the primary takes from its environment rather than its argv.

    ``LLAMA_ARG_CACHE_TYPE_K`` and ``_V`` are the ones that matter here: Studio leaves them in
    the environment instead of materialising them, so a peer launched from the argv alone
    falls back to f16 KV. That is a different cache from the primary's, more memory than the
    topology was priced against, and a different answer to the same request depending on which
    replica served it. ssh carries no environment, so they go on the remote command line.

    Only the ``LLAMA_ARG_`` namespace, so nothing else in this process's environment crosses to
    the peer, and never the endpoint, which the replica is given deliberately.

    ``DENIED_ENV_VARS`` is imported rather than mirrored, and imported here rather than at module
    scope to keep this module free of an import cycle through the backend. The primary never runs
    with those settings -- ``scrub_denied_env`` drops them from the environment it spawns with --
    but it scrubs a COPY, so they are still in ``os.environ`` when this reads it. Mirroring the
    list would let the peer drift into a configuration the primary refuses: ``LLAMA_ARG_API_KEY``
    or ``LLAMA_ARG_SSL_*`` would leave the replica demanding auth or speaking TLS while the
    router health-probes it over plain HTTP, and ``LLAMA_ARG_MODEL`` would point it at a
    different model that answers perfectly well."""
    from core.inference.llama_server_args import DENIED_ENV_VARS

    denied = _REPLICA_ENV_DENY.union(DENIED_ENV_VARS)
    env = os.environ if source is None else source
    out: Dict[str, str] = {}
    for name, value in env.items():
        if not name.startswith(_REPLICA_ENV_PREFIX) or name in denied:
            continue
        text = str(value).strip()
        if not text:
            continue
        if name in _REPLICA_ENV_PATHS:
            text = absolute_sidecar_operand(text, cwd = cwd)
        out[name] = text.lower() if name in _REPLICA_ENV_LOWERED else text
    return out


def with_replica_env(argv: List[str], env: Dict[str, str]) -> List[str]:
    """``argv`` prefixed with ``env NAME=VALUE`` so the remote shell exports them for it."""
    if not env:
        return list(argv)
    return ["env"] + [f"{name}={value}" for name, value in sorted(env.items())] + list(argv)


PEER_BUSY_PROBE_TIMEOUT_S = 8


async def peer_gpu_conflict(peer: str, *, own_pids: Sequence[int] = ()) -> Optional[str]:
    """Why the peer's GPU is not free to take, or None.

    A remote topology puts a second process on the peer's GPU, and it is priced against the
    whole node budget, so a resident training run or somebody else's llama-server means one of
    the two hits an out-of-memory or spends the load in unified-memory contention. The planner
    has no way to know that; ``spark_cluster.peer_gpu_busy`` does.

    KNOWN busy only, which is a deliberate departure from that probe's fail-closed contract.
    Failing closed is right for the ``rsync --delete`` it was written for, where being wrong
    destroys someone's work. Here being wrong the same way turns every slow ssh into silently
    serving on one node, so an unanswered probe leaves the behaviour exactly as it was before
    this check existed. Our own peer process does not count against us."""
    sc = _cluster()
    probe = getattr(sc, "peer_gpu_busy", None)
    if not callable(probe):
        return None
    try:
        result = await asyncio.to_thread(probe, peer, PEER_BUSY_PROBE_TIMEOUT_S)
    except Exception as exc:
        logger.info("spark serving: could not check the peer GPU: %s", exc)
        return None
    if not result.get("known") or not result.get("busy"):
        return None
    mine = {int(pid) for pid in own_pids if pid}
    theirs = [p for p in result.get("processes", []) if int(p.get("pid", 0)) not in mine]
    if not theirs:
        return None
    used = ", ".join(f"pid {p.get('pid')} holding {p.get('used_mib')} MiB" for p in theirs)
    return f"the peer GPU is already in use ({used})"


def argv_or_env_rpc(argv: Sequence[str], env: Optional[Dict[str, str]] = None) -> bool:
    """Whether this llama-server is already an RPC split, however it was told to be.

    ``LLAMA_ARG_RPC`` is a supported way to say it and llama.cpp's common_arg reads it
    directly, so it never appears in argv. Reading argv alone calls such a server unsplit, and
    at replica concurrency that means launching a full peer replica onto a GPU already serving
    the split: contention or an out-of-memory, and routing between two incompatible layouts."""
    for arg in argv or ():
        name = str(arg).partition("=")[0]
        if name == "--rpc":
            return True
    source = os.environ if env is None else env
    return bool(str(source.get("LLAMA_ARG_RPC") or "").strip())


_VERSION_LINE = re.compile(r"version:\s*(\S+)\s*\(([0-9a-f]+)\)", re.IGNORECASE)


def parse_llama_server_version(text: str) -> Optional[str]:
    """``<build> (<commit>)`` out of ``llama-server --version``, or None if it is not there."""
    match = _VERSION_LINE.search(str(text or ""))
    return f"{match.group(1)} ({match.group(2)})" if match else None


def local_llama_server_version(binary: Optional[str] = None) -> Optional[str]:
    path = binary or llama_server_binary()
    if not path:
        return None
    try:
        done = subprocess.run(
            [str(path), "--version"],
            stdin = subprocess.DEVNULL,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            timeout = HELP_PROBE_TIMEOUT_S,
        )
    except Exception:
        return None
    return parse_llama_server_version((done.stdout or b"").decode("utf-8", "replace"))


async def replica_build_mismatch(peer: str, peer_binary: str) -> Optional[str]:
    """The two builds, when they are KNOWN to differ, or None.

    The replica is launched from the primary's complete argv, so an older peer build rejects a
    flag the primary was given and the advertised replica never comes up, and two builds that
    both start can answer the same request differently. The layer split compares the RPC
    protocol before it commits; replicas compared nothing and took any executable with the
    right name, including one off the PATH.

    Known mismatch only, for the same reason as the GPU probe: a version that cannot be read
    on either end is not evidence, and refusing on it would drop the pair to one node whenever
    a probe is slow."""
    local = await asyncio.to_thread(local_llama_server_version)
    if not local:
        return None
    rc, out, _err = await ssh_run(peer, f"{shlex.quote(peer_binary)} --version 2>&1", timeout = 25.0)
    remote = parse_llama_server_version(out) if rc == 0 else None
    if not remote or remote == local:
        return None
    return f"this node runs llama-server {local} and {peer} runs {remote}"


# The names this process writes, not "anything under the temp directory". The directory alone
# was the whole test, so a model or a sidecar living directly under /tmp -- an ordinary place to
# put one -- was classified as generated and handed to ``replicate_generated_files``, which
# reads the file whole, base64-encodes it and embeds it in an ssh command line. For a
# multi-gigabyte GGUF that is Studio's memory and the OS argument limit, instead of the
# ordinary peer-file preflight that would have reported it missing in a sentence.
_GENERATED_NAME_PATTERNS = ("unsloth_chat_template_*.jinja",)
# A generated template is a few KiB. This is a backstop against a name that matches the pattern
# by accident, since being wrong here is measured in gigabytes over ssh.
_GENERATED_MAX_BYTES = 8 * 1024 * 1024


def _looks_generated(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in _GENERATED_NAME_PATTERNS)


def generated_launch_files(files: Sequence[str]) -> List[str]:
    """The launch files this process WROTE rather than found: the ones in the temp directory.

    A chat-template override, or a repaired GGUF template, is written to a uniquely named
    ``unsloth_chat_template_*.jinja`` per load and named in argv. The peer cannot already have
    a file this process just created, so a preflight that demands every launch file be present
    always failed and those loads never got replicas at all."""
    try:
        temp_root = osp.realpath(tempfile.gettempdir())
    except OSError:
        return []
    out: List[str] = []
    for path in files:
        try:
            if not _looks_generated(osp.basename(str(path))):
                continue
            if osp.realpath(osp.dirname(str(path))) != temp_root or not osp.isfile(str(path)):
                continue
            if osp.getsize(str(path)) <= _GENERATED_MAX_BYTES:
                out.append(str(path))
        except OSError:
            continue
    return out


async def replicate_generated_files(peer: str, files: Sequence[str]) -> Optional[str]:
    """Write each file to the peer at the SAME path, or name the first that would not go.

    The same path, because the replica is launched from the primary's argv unchanged and that
    identity is what the whole replica path rests on. The name is unique per load, so there is
    nothing on the peer to collide with. Base64 rather than a heredoc: a Jinja template is full
    of quotes and braces, and the transfer must not depend on any of them."""
    for path in files:
        try:
            payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        except OSError as exc:
            return f"{path} could not be read ({exc})"
        rc, _out, err = await ssh_run(
            peer,
            f"printf %s {shlex.quote(payload)} | base64 -d > {shlex.quote(path)}",
            timeout = 25.0,
        )
        if rc != 0:
            return f"{path} could not be written on {peer} ({(err or '').strip()[:120]})"
    return None


def redacted_argv(argv: List[str]) -> List[str]:
    out = list(argv)
    for index, arg in enumerate(out):
        if arg == "--api-key" and index + 1 < len(out):
            out[index + 1] = "<redacted>"
        elif arg.startswith("--api-key="):
            out[index] = "--api-key=<redacted>"
    return out


def rpc_server_argv(binary: str, *, bind: str, port: int, cache: bool) -> List[str]:
    """``ggml-rpc-server`` on the peer, caching tensors only when the model file is there
    to cache from."""
    argv = [binary, "-H", bind, "-p", str(port)]
    if cache:
        argv.append("-c")
    return argv


# The rpc-server under every name the bundles have used, in ``spark_cluster``'s order.
_RPC_SERVER_STEMS = ("ggml-rpc-server", "rpc-server")


def executable_suffixes() -> Tuple[str, ...]:
    """The suffixes a runnable file can carry on this platform, most likely first.

    Windows has no execute bit, so ``os.access(path, os.X_OK)`` is true for ANY existing file
    there and guards nothing: the name is what decides. POSIX keeps the empty suffix, where
    the extensionless name is the real one."""
    if os.name != "nt":
        return ("",)
    raw = os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
    found = tuple(part.strip().lower() for part in raw.split(";") if part.strip().startswith("."))
    return found or (".exe",)


def rpc_server_names() -> Tuple[str, ...]:
    """Candidate file names for the rpc-server, in the order this platform should try them.

    Built rather than listed, because a fixed list put the extensionless names first for every
    platform: on Windows a stray extensionless file then beat the real ``.exe`` and was
    returned as the binary, which is worse than returning nothing."""
    return tuple(stem + suffix for stem in _RPC_SERVER_STEMS for suffix in executable_suffixes())


def is_executable_file(path: Any) -> bool:
    """Whether ``path`` is a file this platform would run. See ``executable_suffixes``.

    os.path rather than pathlib throughout: ``Path()`` picks its flavour from ``os.name`` at
    construction, so a test that simulates Windows would build a WindowsPath and this would
    answer about a path the running system cannot even stat."""
    name = str(path)
    try:
        if not osp.isfile(name):
            return False
    except OSError:
        return False
    if os.name != "nt":
        return os.access(name, os.X_OK)
    return osp.splitext(name)[1].lower() in executable_suffixes()


def llama_server_binary() -> Optional[str]:
    """The llama-server this node will actually LAUNCH, asked of the backend's own
    ``_find_llama_server_binary`` and never searched for here: a second search order is a
    defect, and it was one -- ``spark_cluster``'s layouts include ``<root>/bin`` and the
    backend's do not, so the two ends of the link ran different builds. Hence no fallback."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend
    except Exception as exc:  # pragma: no cover - the backend package is always there
        logger.info("spark serving: cannot import the llama.cpp backend resolver: %s", exc)
        return None
    try:
        found = LlamaCppBackend._find_llama_server_binary()
    except Exception as exc:
        logger.info("spark serving: the backend's llama-server resolver failed: %s", exc)
        return None
    return str(found) if found else None


def rpc_server_binary() -> Optional[str]:
    """``ggml-rpc-server`` from beside the llama-server that launches, since the peer's copy
    is looked up from this path's directory. The bundle search is the fallback for a tree that
    ships llama-server alone."""
    launched = llama_server_binary()
    if launched:
        # os.path, like is_executable_file: this resolution is about names on disk and must
        # not depend on which pathlib flavour os.name selects.
        directory = osp.dirname(osp.abspath(launched))
        for name in rpc_server_names():
            candidate = osp.join(directory, name)
            if is_executable_file(candidate):
                return candidate
    sc = _cluster()
    try:
        found = sc.rpc_server_binary() if sc is not None else None
    except Exception:
        return None
    return str(found) if found else None


# Keyed by (path, mtime) so a reinstall at the same path probes again; a failure is cached,
# so a broken binary costs one timeout and not one per load.
_HELP_TEXT: Dict[Tuple[str, float], str] = {}


def llama_server_help(binary: Optional[str] = None) -> str:
    """The ``--help`` of the llama-server that will LAUNCH, empty on any failure. Never raises.

    Not the bundle's: ``llama_server_binary`` is the backend's own
    ``_find_llama_server_binary``, so LLAMA_SERVER_PATH, UNSLOTH_LLAMA_CPP_PATH and a custom
    llama.cpp folder all select the same executable here that the load will run, and a flag is
    never decided against a different build from the one it is passed to."""
    path = binary or llama_server_binary()
    if not path:
        return ""
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return ""
    key = (str(path), mtime)
    cached = _HELP_TEXT.get(key)
    if cached is not None:
        return cached
    text = ""
    try:
        done = subprocess.run(
            [str(path), "--help"],
            stdin = subprocess.DEVNULL,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            timeout = HELP_PROBE_TIMEOUT_S,
        )
        text = (done.stdout or b"").decode("utf-8", "replace") + (done.stderr or b"").decode(
            "utf-8", "replace"
        )
    except Exception as exc:
        logger.info("spark serving: llama-server --help probe failed for %s: %s", path, exc)
        text = ""
    _HELP_TEXT[key] = text
    return text


def llama_server_supports(flag: str, binary: Optional[str] = None) -> bool:
    """From the ``--help`` text; False on every failure, so a flag the build may lack is
    never passed."""
    try:
        text = llama_server_help(binary)
    except Exception:
        return False
    if not text or not flag:
        return False
    return re.search(re.escape(flag) + r"(?![\w-])", text) is not None


_ACCEPTS: Dict[Tuple[str, float, Tuple[str, ...]], bool] = {}


def llama_server_accepts(
    flag: str,
    value: str = "1",
    binary: Optional[str] = None,
    *,
    extra: Sequence[str] = (),
) -> bool:
    """For a flag a build hides from its usage text: the fork strips ``--pipeline-groups`` from
    argv before the common parser and prints the usage, while every other build stops at
    "invalid argument". False on every failure."""
    path = binary or llama_server_binary()
    if not path or not flag:
        return False
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return False
    probe = [str(flag), str(value), *(str(a) for a in extra)]
    key = (str(path), mtime, tuple(probe))
    cached = _ACCEPTS.get(key)
    if cached is not None:
        return cached
    accepted = False
    try:
        done = subprocess.run(
            [str(path), *probe, "--help"],
            stdin = subprocess.DEVNULL,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            timeout = HELP_PROBE_TIMEOUT_S,
        )
        accepted = done.returncode == 0 and bool((done.stdout or b"").strip())
    except Exception as exc:
        logger.info(
            "spark serving: llama-server %s probe failed for %s: %s",
            " ".join(probe),
            path,
            exc,
        )
        accepted = False
    _ACCEPTS[key] = accepted
    return accepted


def _free_local_port() -> int:
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except OSError:
        return 0


def llama_server_accepts_groups_with_drafter(
    groups: int = PIPELINE_GROUPS_DEFAULT, binary: Optional[str] = None
) -> bool:
    """Whether the bundle's llama-server runs ``--pipeline-groups N > 1`` TOGETHER with a
    drafter: the per-group speculative state of unslothai/llama.cpp PR #187 (a1dd7c5e8).

    A ``--help`` probe can only ever say yes, the refusal being inside ``load_model``, which
    ``--help`` exits long before; ``load_model`` validates the pair before it reads any weights,
    so the server runs for real against a path that cannot exist and the OUTPUT, not the exit
    status, is the discriminator. False on any doubt."""
    groups = int(groups or 0)
    if groups <= 1:
        return False
    # Only ask the expensive question of a build that has the flag at all.
    if not (
        llama_server_supports(PIPELINE_GROUPS_FLAG, binary)
        or llama_server_accepts(PIPELINE_GROUPS_FLAG, str(groups), binary)
    ):
        return False
    if not llama_server_supports(SPEC_TYPE_FLAG, binary):
        return False
    path = binary or llama_server_binary()
    if not path:
        return False
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return False
    probe = [
        PIPELINE_GROUPS_FLAG,
        str(groups),
        SPEC_TYPE_FLAG,
        MTP_SPEC_TYPE,
        "--parallel",
        str(groups),
        "-c",
        str(groups * 512),
        "-m",
        _PROBE_MODEL_NAME,
    ]
    key = (str(path), mtime, tuple(probe))
    cached = _ACCEPTS.get(key)
    if cached is not None:
        return cached
    accepted = False
    port = _free_local_port()
    argv = [str(path), *probe, "--host", "127.0.0.1", "--port", str(port or 1)]
    try:
        # An empty cwd, so a file of that name in the caller's cwd cannot fool the probe.
        with tempfile.TemporaryDirectory(prefix = "unsloth-spark-probe-") as workdir:
            done = subprocess.run(
                argv,
                cwd = workdir,
                stdin = subprocess.DEVNULL,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                timeout = GROUPS_DRAFTER_PROBE_TIMEOUT_S,
            )
        text = (done.stdout or b"").decode("utf-8", "replace")
        # Positive evidence, not merely the absence of the refusal.
        accepted = _GROUPS_REFUSAL_TEXT not in text and _PROBE_MODEL_NAME in text
    except Exception as exc:
        logger.info(
            "spark serving: %s %d with a drafter probe failed for %s: %s",
            PIPELINE_GROUPS_FLAG,
            groups,
            path,
            exc,
        )
        accepted = False
    _ACCEPTS[key] = accepted
    logger.info(
        "spark serving: %s %d together with %s %s: %s",
        PIPELINE_GROUPS_FLAG,
        groups,
        SPEC_TYPE_FLAG,
        MTP_SPEC_TYPE,
        "accepted" if accepted else "refused",
    )
    return accepted


def _first_shard(path: str) -> str:
    """The file that carries the header: shard 00001 of a split GGUF, else ``path``."""
    match = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", osp.basename(path))
    if not match:
        return path
    prefix, _index, count = match.groups()
    return osp.join(osp.dirname(path), f"{prefix}-00001-of-{count}.gguf")


def gguf_nextn_predict_layers(path: Optional[str]) -> Optional[int]:
    """``<arch>.nextn_predict_layers`` from the GGUF header, None on anything missing. The
    backend's own reader answers when importable, so this and the launch agree."""
    if not path:
        return None
    try:
        shard = _first_shard(str(path))
        if not osp.isfile(shard):
            return None
    except Exception:
        return None
    try:
        from utils.models.gguf_metadata import read_gguf_nextn_predict_layers  # type: ignore
        value = read_gguf_nextn_predict_layers(shard)
        return int(value) if value is not None else None
    except Exception:
        pass
    try:
        from gguf import GGUFReader  # type: ignore

        fields = GGUFReader(shard).fields
        arch_field = fields.get("general.architecture")
        if arch_field is None:
            return None
        arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8", "replace")
        field = fields.get(f"{arch}.nextn_predict_layers")
        if field is None:
            return None
        return int(field.parts[field.data[0]][0])
    except Exception:
        return None


def gguf_has_mtp_head(path: Optional[str]) -> bool:
    try:
        return (gguf_nextn_predict_layers(path) or 0) > 0
    except Exception:
        return False


def _arg_name(arg: Any) -> str:
    """The option name in ``arg`` the way llama-server itself reads it.

    common/arg.cpp folds ``_`` to ``-`` in every ``--`` token before it looks the option up, so
    ``--spec_type`` and ``--control_vector`` are spellings the server accepts and a raw
    comparison against the hyphenated name silently misses them. Shorts are left alone, which
    is also what arg.cpp does: the fold is guarded on the ``--`` prefix there too."""
    name = str(arg).strip().partition("=")[0]
    return name.replace("_", "-") if name.startswith("--") else name


def extra_args_own_speculation(extra_args: Optional[List[str]]) -> Optional[str]:
    """The first pass-through flag that makes speculative decoding the caller's, or None."""
    for arg in extra_args or []:
        name = _arg_name(arg)
        if name in _SPEC_OWNER_FLAGS or name.startswith(_SPEC_OWNER_PREFIXES):
            return name
    return None


def mtp_draft_n_max(users: Optional[int] = None) -> int:
    """The depth measured best at this many rows, and ``MTP_DRAFT_N_MAX`` below the lowest
    measured row count or when ``users`` is unknown."""
    if users is None:
        return MTP_DRAFT_N_MAX
    rows = max(1, int(users or 1))
    depth = MTP_DRAFT_N_MAX
    for key in sorted(MTP_DRAFT_N_MAX_BY_ROWS):
        if rows >= key:
            depth = MTP_DRAFT_N_MAX_BY_ROWS[key]
    return depth


def split_mtp_wins(users: Optional[int] = None) -> bool:
    """Whether a LAYER SPLIT at this many rows should run a drafter at all; ``users`` of None
    keeps it. Only ``reconcile_split_speculation`` calls this, so it cannot reach a single-node
    or replicas launch, where MTP is a large win at every measured depth."""
    if users is None:
        return True
    return max(1, int(users or 1)) < SPLIT_MTP_OFF_ROWS


def mtp_plan(
    gguf_path: Optional[str],
    extra_args: Optional[List[str]] = None,
    *,
    speculative_type: Optional[str] = None,
    spec_draft_n_max: Optional[int] = None,
    users: Optional[int] = None,
) -> Dict[str, Any]:
    """Whether a Spark load should ask for MTP self speculation, and at what depth. ``unknown``
    means the GGUF is not on disk yet, so the backend decides alone and ``after_load`` reports
    what launched."""
    out: Dict[str, Any] = {"mtp": "unknown", "reason": None, "request": {}}
    owner = extra_args_own_speculation(extra_args)
    if owner:
        out.update(mtp = "user override", reason = f"{owner} in the pass-through arguments")
        return out
    mode = str(speculative_type or "").strip().lower()
    if mode and mode not in ("auto", "default"):
        out.update(mtp = "user override", reason = f"speculative_type={speculative_type}")
        return out
    if spec_draft_n_max is not None:
        out.update(mtp = "user override", reason = f"spec_draft_n_max={spec_draft_n_max}")
        return out
    if (os.environ.get(ENV_MTP) or "").strip() == "0":
        out.update(
            mtp = "disabled by env",
            reason = f"{ENV_MTP}=0",
            request = {"speculative_type": "off"},
        )
        return out
    try:
        on_disk = bool(gguf_path) and osp.isfile(_first_shard(str(gguf_path)))
    except Exception:
        on_disk = False
    if not on_disk:
        out["reason"] = "GGUF not on disk before the load; the backend decides on its own"
        return out
    layers = gguf_nextn_predict_layers(gguf_path)
    if not layers:
        out.update(mtp = "no head", reason = "no <arch>.nextn_predict_layers in the GGUF header")
        return out
    if not llama_server_supports(SPEC_TYPE_FLAG):
        out.update(mtp = "server too old", reason = f"bundle llama-server lacks {SPEC_TYPE_FLAG}")
        return out
    depth = mtp_draft_n_max(users)
    out.update(
        mtp = "enabled",
        reason = (
            f"{layers} MTP layer(s) in the header; {SPEC_TYPE_FLAG} {MTP_SPEC_TYPE} "
            f"{SPEC_DRAFT_N_MAX_FLAG} {depth}"
            + ("" if depth == MTP_DRAFT_N_MAX else f" (measured best at {users} rows)")
        ),
        request = {"spec_draft_n_max": depth},
    )
    return out


def caller_speculation_off(
    speculative_type: Optional[str] = None, extra_args: Optional[List[str]] = None
) -> bool:
    """True only when what the caller set says no speculative decoding at all; anything else
    of theirs may launch a drafter."""
    mode = str(speculative_type or "").strip().lower()
    if mode in ("off", "none", "disable", "disabled"):
        return True
    if mode:
        return False
    owner = extra_args_own_speculation(extra_args)
    if owner is None:
        return False
    spec, _depth = launched_spec_flags(list(extra_args or []))
    others = [
        str(a).partition("=")[0]
        for a in (extra_args or [])
        if str(a).partition("=")[0] != SPEC_TYPE_FLAG
        and (
            str(a).partition("=")[0] in _SPEC_OWNER_FLAGS
            or str(a).partition("=")[0].startswith(_SPEC_OWNER_PREFIXES)
        )
    ]
    return spec == "none" and not others


def reconcile_split_speculation(
    groups: Dict[str, Any],
    mtp: Dict[str, Any],
    *,
    speculative_type: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> None:
    """Which of the three layer-split configurations to launch, resolved in place before the
    launch so the server never refuses a start.

    Decided on ``requested_slots``, the concurrency asked for and not the count rounded up to a
    multiple of the groups. ``SPLIT_MTP_OFF_ROWS`` applies FIRST, dropping this module's own
    speculation and saying off so the backend's auto mode cannot put a drafter back;
    ``GROUPS_X_MTP_MIN_ROWS`` then keeps both on a build with PR #187's per-group speculative
    state and otherwise drops the groups. A drafter the CALLER asked for is never taken away.
    """
    planned = int(groups.get("pipeline_groups") or 0)
    verdict = mtp.get("mtp")
    callers = verdict == "user override"
    speculating = verdict == "enabled" or (
        callers and not caller_speculation_off(speculative_type, extra_args)
    )
    rows = int(groups.get("requested_slots") or groups.get("slots") or 1)
    # "unknown" counts here as well as "enabled". On the FIRST load of an uncached MTP-capable
    # GGUF the header is not on disk yet, so mtp_plan cannot say, and leaving the decision open
    # let the backend switch its own MTP on after the download -- at a width where this module's
    # measured rule says every split is faster with no drafter at all. Turning it off is safe in
    # both directions: on a model with no head it is a no-op, and on one with a head it is what
    # the measurement asks for. A drafter the CALLER asked for is "user override" and untouched.
    if verdict in ("enabled", "unknown") and not split_mtp_wins(rows):
        # The depth field goes with it, so no draft flag of any kind is emitted.
        mtp["mtp"] = MTP_OFF_FOR_SPLIT_ROWS
        mtp["reason"] = (
            f"{rows} rows is at or above {SPLIT_MTP_OFF_ROWS}, where a split measured faster "
            f"with NO drafter at every depth swept (best depth 171.7 against 186.3 tok/s at "
            f"64 rows, -7.9 percent, and 164.1 against 212.7 at 128, -22.9 percent); below "
            f"{SPLIT_MTP_OFF_ROWS} it speculates at the measured depth, worth +11.0 percent "
            f"at 32 rows"
            + (
                " (the header was not readable yet, so this is applied without waiting for the "
                "download to say whether there is a head)"
                if verdict == "unknown"
                else ""
            )
            + f". Previously: {mtp.get('reason')}"
        )
        request = mtp.setdefault("request", {})
        request.pop("spec_draft_n_max", None)
        request["speculative_type"] = "off"
        groups["split_config"] = SPLIT_CONFIG_GROUPS if planned > 1 else SPLIT_CONFIG_PLAIN
        groups["split_config_reason"] = mtp["reason"]
        return
    if planned <= 1:
        groups["split_config"] = SPLIT_CONFIG_SPEC if speculating else SPLIT_CONFIG_PLAIN
        groups["split_config_reason"] = str(
            groups.get("reason") or f"{PIPELINE_GROUPS_FLAG} not added"
        )
        return
    if speculating:
        combined = llama_server_accepts_groups_with_drafter(planned)
        if rows >= GROUPS_X_MTP_MIN_ROWS and combined:
            hi = max(GROUPS_X_MTP_OVER_MTP_ONLY)
            groups["split_config"] = SPLIT_CONFIG_BOTH
            groups["split_config_reason"] = (
                f"{rows} rows is at or above the measured crossover of "
                f"{GROUPS_X_MTP_MIN_ROWS}, and this llama-server takes "
                f"{PIPELINE_GROUPS_FLAG} {planned} together with a drafter: both, measured "
                f"{GROUPS_X_MTP_OVER_MTP_ONLY[hi]:.2f}x of one context with speculation and "
                f"{GROUPS_X_MTP_OVER_GROUPS_ONLY[hi]:.2f}x of {planned} groups alone at "
                f"{hi} rows"
            )
            mtp["reason"] = (
                f"{mtp.get('reason')}; kept together with {PIPELINE_GROUPS_FLAG} {planned}"
            )
            return
        lo = min(GROUPS_X_MTP_OVER_MTP_ONLY)
        why = (
            f"{rows} rows is below the measured crossover of {GROUPS_X_MTP_MIN_ROWS}, where "
            f"{planned} groups halve the rows per group and measured "
            f"{GROUPS_X_MTP_OVER_MTP_ONLY[lo]:.2f}x of one context with speculation at "
            f"{lo} rows"
            if rows < GROUPS_X_MTP_MIN_ROWS
            else (
                "this llama-server refuses it together with a drafter (no per-group "
                "speculative state; see unslothai/llama.cpp PR #187)"
            )
        )
        if callers:
            why = f"{why}, and the speculation is the caller's"
        groups["pipeline_groups"] = 0
        groups["slots"] = groups.get("requested_slots", groups.get("slots"))
        groups["reason"] = f"{PIPELINE_GROUPS_FLAG} not added: {why}"
        groups["split_config"] = SPLIT_CONFIG_SPEC
        groups["split_config_reason"] = groups["reason"]
        return
    groups["split_config"] = SPLIT_CONFIG_GROUPS
    groups["split_config_reason"] = (
        f"{PIPELINE_GROUPS_FLAG} {planned} and no speculation to keep: "
        f"{mtp.get('reason') or verdict}"
    )
    if callers:
        # Their "off" is left exactly as they wrote it.
        return
    if (
        verdict == "unknown"
        and split_mtp_wins(rows)
        and llama_server_accepts_groups_with_drafter(planned)
    ):
        # "unknown" is not "no head": the GGUF is not on disk yet, so mtp_plan deliberately
        # declines to answer and the backend reads the header itself after the download.
        # Writing "off" here would take automatic MTP away from a model that does have a head,
        # which is what an uncached repo forced to a layer split always is. Left undecided only
        # where the answer could still be yes: this llama-server takes the groups together with
        # a drafter, and the rows are below the count where a split measured faster with none.
        groups["split_config_reason"] = (
            f"{PIPELINE_GROUPS_FLAG} {planned}, speculation left to the backend: "
            f"{mtp.get('reason') or verdict}"
        )
        return
    request = mtp.setdefault("request", {})
    if request.get("speculative_type") != "off":
        request["speculative_type"] = "off"
        no_head = "which this GGUF has no head for and " if verdict == "no head" else ""
        mtp["reason"] = (
            f"{mtp.get('reason')}; speculation off for {PIPELINE_GROUPS_FLAG} "
            f"{groups['pipeline_groups']}, {no_head}which a "
            f"sidecar drafter loses on from 4 users on this pair"
        )


def launched_spec_flags(argv: List[str]) -> Tuple[Optional[str], Optional[int]]:
    """The last ``--spec-type`` and ``--spec-draft-n-max`` an argv carries; last wins, as in
    llama.cpp."""
    spec: Optional[str] = None
    depth: Optional[int] = None
    args = [str(a) for a in argv]
    for index, arg in enumerate(args):
        _, _, inline = arg.partition("=")
        name = _arg_name(arg)
        if name not in (SPEC_TYPE_FLAG, SPEC_DRAFT_N_MAX_FLAG):
            continue
        value = inline if inline else (args[index + 1] if index + 1 < len(args) else "")
        if name == SPEC_TYPE_FLAG:
            spec = value.strip() or None
        else:
            try:
                depth = int(value.strip())
            except ValueError:
                continue
    return spec, depth


def _extra_args_slots(extra_args: Optional[List[str]]) -> Optional[int]:
    """The slot count a pass-through already sets, last wins, as llama.cpp does."""
    found: Optional[int] = None
    args = [str(a) for a in (extra_args or [])]
    for index, arg in enumerate(args):
        _, _, inline = arg.partition("=")
        name = _arg_name(arg)
        if name not in ("-np", "--parallel"):
            continue
        value = inline if inline else (args[index + 1] if index + 1 < len(args) else "")
        try:
            found = int(value.strip())
        except (TypeError, ValueError):
            continue
    return found


# The env twins of the flags the groups are refused with. llama.cpp's common_arg reads these
# itself, so a projector or control vector can reach the server without ever appearing in argv,
# and a check that only reads argv sees a clean launch and enables the groups anyway. Same
# mechanism as the sidecar paths above: one setting, two routes, and a guard on one route only
# is a guard that holds until somebody uses the other.
_GROUPS_REFUSED_ENV = frozenset({"LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"})


def extra_args_refuse_pipeline_groups(
    extra_args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """The first pass-through flag or environment setting the server still refuses together
    with the groups."""
    for arg in extra_args or []:
        name = _arg_name(arg)
        if name in _GROUPS_REFUSED_FLAGS:
            return name
    source = os.environ if env is None else env
    for name in _GROUPS_REFUSED_ENV:
        if str(source.get(name, "")).strip():
            return name
    return None


def _from_hub_repo(model_file: Optional[str]) -> bool:
    """Whether the load can still fetch a companion the directory does not have yet."""
    parts = Path(str(model_file or "")).parts
    return "snapshots" in parts and any(part.startswith("models--") for part in parts)


def projector_blocks_pipeline_groups(
    model_file: Optional[str], *, disable_vision: bool = False
) -> Optional[str]:
    """Why this load cannot have pipeline groups because of a multimodal projector, or None.

    ``--mmproj`` is emitted AFTER ``before_load`` and the backend DOWNLOADS the projector during
    the load, so a directory scan beforehand cannot clear a hub repo and only the Vision switch
    can. The server refuses the pair inside load_model, so the whole load fails rather than
    losing a flag. ``disable_vision`` clears the block unless the projector on disk is
    audio-only, which the switch does not drop."""
    on_disk = None
    if model_file:
        try:
            # NOT realpath: an HF cache snapshot is a directory of symlinks into blobs/,
            # where the neighbours are content hashes and nothing ends in .gguf.
            directory = osp.dirname(osp.abspath(str(model_file)))
            found = sorted(
                path
                for path in glob.glob(osp.join(directory, "*.gguf"))
                if "mmproj" in osp.basename(path).lower()
                and not osp.basename(path).startswith("._")
            )
        except OSError:
            found = []
        if found:
            on_disk = next((f for f in found if f.lower().endswith("-f16.gguf")), found[0])
    if not disable_vision:
        if on_disk:
            return (
                f"Studio opens this model's multimodal projector ({osp.basename(on_disk)}); "
                f"load with vision off to get {PIPELINE_GROUPS_FLAG}"
            )
        if _from_hub_repo(model_file):
            return (
                "Studio fetches this repo's multimodal projector during the load when it has "
                f"one, so it cannot be ruled out beforehand; load with vision off to get "
                f"{PIPELINE_GROUPS_FLAG}"
            )
        return None
    if not on_disk:
        return None
    try:
        from utils.models.gguf_metadata import mmproj_accepts_image  # type: ignore
        if mmproj_accepts_image(on_disk):
            return None  # the Vision switch really does suppress this one
    except Exception as exc:
        logger.info("spark serving: could not read the projector %s: %s", on_disk, exc)
    return (
        f"the projector {osp.basename(on_disk)} is not an image projector, so the load's "
        f"vision switch does not drop it"
    )


def pipeline_groups_plan(
    slots: int,
    extra_args: Optional[List[str]] = None,
    *,
    projector: Optional[str] = None,
) -> Dict[str, Any]:
    """How many pipeline groups a layer split should run, and with how many slots: ``slots``
    is rounded UP to a multiple of the group count and ``requested_slots`` is the count
    before rounding."""
    requested = _extra_args_slots(extra_args)
    base = max(1, int(requested if requested is not None else (slots or 1)))
    out: Dict[str, Any] = {
        "pipeline_groups": 0,
        "reason": None,
        "slots": base,
        "requested_slots": base,
    }
    raw = (os.environ.get(ENV_PIPELINE_GROUPS) or "").strip()
    groups = PIPELINE_GROUPS_DEFAULT
    if raw:
        try:
            groups = int(raw)
        except ValueError:
            out["reason"] = (
                f"{ENV_PIPELINE_GROUPS}={raw!r} is not a number; {PIPELINE_GROUPS_FLAG} not added"
            )
            return out
    if groups <= 1:
        out["reason"] = (
            f"disabled by {ENV_PIPELINE_GROUPS}={raw}"
            if raw
            else f"{PIPELINE_GROUPS_FLAG} not added"
        )
        return out
    if projector:
        out["reason"] = f"{PIPELINE_GROUPS_FLAG} not added: {projector}"
        return out
    refused = extra_args_refuse_pipeline_groups(extra_args)
    if refused:
        out["reason"] = (
            f"{PIPELINE_GROUPS_FLAG} not added: the server refuses it together with {refused}"
        )
        return out
    if not (
        llama_server_supports(PIPELINE_GROUPS_FLAG) or llama_server_accepts(PIPELINE_GROUPS_FLAG)
    ):
        out["reason"] = f"bundle llama-server lacks {PIPELINE_GROUPS_FLAG}"
        return out
    slots = max(groups, -(-base // groups) * groups)
    if slots > PARALLEL_MAX:
        # A LoadRequest field with a range, not free argv: rounding UP would be refused.
        slots = (PARALLEL_MAX // groups) * groups
        if slots < groups:
            out["reason"] = (
                f"{PIPELINE_GROUPS_FLAG} not added: {groups} groups do not fit in the "
                f"{PARALLEL_MAX}-slot maximum"
            )
            return out
        if base <= PARALLEL_MAX and slots < base * GROUPS_WORTH_SLOTS_RATIO:
            # _start_layer_split writes this straight into request.n_parallel, so rounding down
            # to fit the cap silently serves fewer concurrent users than were asked for. That
            # is a trade, not a bug, and it is only worth making while the groups are worth
            # more than the concurrency given up for them: they measured about 1.4x on this
            # pair, so keep them down to 1/1.4 of the request and drop them below that. A
            # ``base`` already over the maximum is not this trade at all, since that request
            # was going to be clamped with or without the groups.
            out["reason"] = (
                f"{PIPELINE_GROUPS_FLAG} not added: {groups} groups need {base} slots rounded "
                f"up to a multiple, which is over the {PARALLEL_MAX}-slot maximum, and "
                f"rounding down would drop {base - slots} of the {base} slots asked for"
            )
            return out
    out["pipeline_groups"] = groups
    out["slots"] = slots
    return out


def layer_split_extra_args(
    peer: str,
    port: int,
    *,
    pipeline_groups: int = 0,
) -> List[str]:
    """What the local llama-server needs to use the peer's rpc-server.

    The slot count the groups need is NOT emitted here: ``-np`` / ``--parallel`` is denied in a
    pass-through (llama_server_args._DENYLIST_GROUPS), so a ``--parallel`` in these extras
    failed the load with HTTP 400 before llama-server started."""
    # RPC device FIRST, local CUDA LAST: llama.cpp puts the output layer on the last device, so
    # this keeps the logits local and the wire carries only the hidden state. With CUDA0,RPC0
    # two groups are SLOWER than one context, the returning logits and the CPU sampling
    # serialising the groups.
    # --cache-ram 0: the host-RAM prompt cache save/restore of a whole slot state runs on the
    # single task thread at every handover, and with two groups the other group starves. KV
    # prefix reuse inside a slot is untouched.
    # --tensor-split explicitly, because llama.cpp divides the layers by each device's FREE
    # MEMORY at load time (llama-model.cpp, "default split, by free memory"), so the boundary
    # is not reproducible between two loads.
    out = [
        "--rpc",
        f"{peer}:{port}",
        "--device",
        "RPC0,CUDA0",
        "-sm",
        "layer",
        "--tensor-split",
        SPLIT_TENSOR_SPLIT_EVEN,
        "--cache-ram",
        "0",
    ]
    if pipeline_groups and int(pipeline_groups) > 1:
        out += [PIPELINE_GROUPS_FLAG, str(int(pipeline_groups))]
    return out


# Resolved here rather than in the child. preexec_fn runs between fork and exec, where any lock a
# non-surviving thread held is held forever, so an import or a library load there can hang the
# child outright; the docs' rule is that the callable touch as few libraries as possible. None on
# any platform without glibc, which is also where there is no PDEATHSIG to set.
try:
    import ctypes as _ctypes
    _LIBC = _ctypes.CDLL("libc.so.6", use_errno = True)
except Exception:
    _LIBC = None


def _die_with_parent() -> None:  # pragma: no cover - runs in the forked child
    """``PR_SET_PDEATHSIG(SIGKILL)``, so the ssh client cannot outlive this process. Runs between
    fork and exec: it must not raise, must not import, and must not load a library."""
    if _LIBC is None:
        return
    try:
        _LIBC.prctl(1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG, SIGKILL
    except Exception:
        pass


# A poll and not a read on the ssh channel: a half-open TCP connection never delivers EOF.
PEER_REAP_POLL_S = 5


class PeerProcess:
    """A long-lived process on the peer, driven through one ssh session.

    The remote command prints the server's pid, so teardown kills that pid rather than matching
    a name on a machine that may be serving something else, and it reaps itself: a Studio
    killed without its shutdown path used to leave the peer's rpc-server holding the peer's
    GPU. The ssh client gets ``PR_SET_PDEATHSIG`` (without it a SIGKILL reparents it to init
    and the session stays open) and the remote shell kills the ONE pid it started.
    """

    def __init__(
        self,
        name: str,
        peer: str,
        argv: List[str],
        log_path: Optional[Path] = None,
    ):
        self.name = name
        self.peer = peer
        self.argv = argv
        self.log_path = log_path
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.remote_pid: Optional[int] = None
        self.started_at: Optional[float] = None
        self.exited_at: Optional[float] = None
        self.returncode: Optional[int] = None
        self.tail: Deque[str] = collections.deque(maxlen = _LOG_TAIL)
        self._drain: Optional[asyncio.Task] = None

    @property
    def remote_command(self) -> str:
        """Start the server, report ITS pid, then watch the ssh session that started us.

        ``$PPID`` is the sshd session serving this command; when it exits the loop kills
        ``$srv`` and no other pid. If the server exits first the shell waits for it.
        """
        launch = " ".join(shlex.quote(a) for a in self.argv)
        return (
            f"{launch} & srv=$!; echo UNSLOTH_SPARK_PID=$srv; watch=$PPID; "
            f'while kill -0 "$srv" 2>/dev/null; do '
            f'if ! kill -0 "$watch" 2>/dev/null; then '
            f'kill "$srv" 2>/dev/null; sleep {PEER_REAP_POLL_S}; '
            f'kill -9 "$srv" 2>/dev/null; exit 143; fi; '
            f"sleep {PEER_REAP_POLL_S}; done; "
            f'wait "$srv"'
        )

    @property
    def redacted_command(self) -> str:
        return " ".join(redacted_argv(self.argv))

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self) -> None:
        # A relaunch must not inherit the previous pid: stop() would kill whatever the peer
        # has since reused that number for.
        self.remote_pid = None
        self.proc = await asyncio.create_subprocess_exec(
            *ssh_argv(self.peer, self.remote_command, keepalive = True),
            stdout = asyncio.subprocess.PIPE,
            stderr = asyncio.subprocess.STDOUT,
            stdin = asyncio.subprocess.DEVNULL,
            preexec_fn = _die_with_parent,
        )
        self.started_at = time.time()
        self.exited_at = None
        self.returncode = None
        self._drain = asyncio.create_task(self._drain_output())

    async def _drain_output(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        handle = None
        if self.log_path is not None:
            try:
                self.log_path.parent.mkdir(parents = True, exist_ok = True)
                handle = open(self.log_path, "a", encoding = "utf-8", errors = "replace")
                handle.write(
                    f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} {self.name} on {self.peer}: "
                    f"{self.redacted_command}\n"
                )
            except OSError:
                handle = None
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").rstrip()
                if text.startswith("UNSLOTH_SPARK_PID="):
                    try:
                        self.remote_pid = int(text.split("=", 1)[1].strip())
                    except ValueError:
                        pass
                    continue
                self.tail.append(text)
                if handle is not None:
                    handle.write(text + "\n")
                    handle.flush()
        except (asyncio.CancelledError, Exception):
            pass
        finally:
            if handle is not None:
                handle.close()
            if self.proc is not None:
                try:
                    await self.proc.wait()
                except Exception:
                    pass
                self.returncode = self.proc.returncode
                self.exited_at = time.time()

    async def stop(self, *, timeout: float = 10.0) -> None:
        """Kill the remote process by pid -- only a pid this run printed -- then the ssh
        session carrying it."""
        if self.remote_pid:
            await ssh_run(
                self.peer,
                f"kill {self.remote_pid} 2>/dev/null; sleep 1; kill -9 {self.remote_pid} 2>/dev/null; true",
                timeout = timeout,
            )
        proc = self.proc
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout = timeout)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        if self._drain is not None:
            try:
                await asyncio.wait_for(self._drain, timeout = 2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                self._drain.cancel()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "peer": self.peer,
            "alive": self.alive,
            "remote_pid": self.remote_pid,
            "started_at": self.started_at,
            "exited_at": self.exited_at,
            "returncode": self.returncode,
            "command": self.redacted_command,
            "log": str(self.log_path) if self.log_path else None,
            "tail": list(self.tail)[-5:],
        }


async def wait_for_port(
    host: str,
    port: int,
    timeout: float,
    *,
    cancelled: Optional[Callable[[], bool]] = None,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cancelled is not None and cancelled():
            return False
        try:
            _r, w = await asyncio.wait_for(asyncio.open_connection(host, port), timeout = 2.0)
            w.close()
            return True
        except (OSError, asyncio.TimeoutError):
            await asyncio.sleep(0.25)
    return False


# How long a freshly launched peer process has to fall over on a bind error before its port
# answering is believed. A port that was ALREADY occupied answers on the first probe, well
# inside the ssh round trip the child needs to reach its own bind() and exit.
PEER_OWNERSHIP_SETTLE_S = 1.5


async def wait_for_own_port(
    process: "PeerProcess",
    host: str,
    port: int,
    timeout: float,
    *,
    cancelled: Optional[Callable[[], bool]] = None,
) -> bool:
    """Wait until ``process`` is answering on ``port``, not merely until something is.

    An occupied port answers on the first probe while the child that could not bind it exits,
    and adopting that listener attaches the load to a server nothing here manages. For an
    rpc-server that means the supervisor relaunching a dead child until it gives up, and an
    unrecoverable split if the stranger later leaves; for a replica it is worse, because the
    stranger is admitted as a backend and generation traffic goes to whatever model it holds.

    Requiring the child to still be running is what tells the two apart, after a settle so a
    bind error has time to become an exit."""
    if not process.alive:
        return False
    if not await wait_for_port(host, port, timeout, cancelled = cancelled):
        return False
    if PEER_OWNERSHIP_SETTLE_S > 0:
        await asyncio.sleep(PEER_OWNERSHIP_SETTLE_S)
    return process.alive


def _log_dir() -> Optional[Path]:
    try:
        from utils.paths.storage_roots import studio_root
        return studio_root() / "logs" / "spark"
    except Exception:
        return None


class SparkServing:
    def __init__(self):
        self.topology: str = "single"
        self.reason: str = ""
        self.plan: Dict[str, Any] = {}
        self.preflight: Optional[Dict[str, Any]] = None
        self.router: Optional[SparkRouter] = None
        self.peer_process: Optional[PeerProcess] = None
        self.peer: Optional[str] = None
        self.attached_backend: Any = None
        self.attached_port: Optional[int] = None
        self.relaunch_attempts: int = 0
        self.relaunch_gave_up: bool = False
        self._cancel_event: Any = None
        self.relaunch_log: List[Dict[str, Any]] = []
        self.peer_model_present: Optional[bool] = None
        self._pre_load_state: Optional[tuple] = None
        self.pipeline_groups: int = 0
        self.pipeline_groups_reason: Optional[str] = None
        self.split_config: Optional[str] = None
        self.split_config_reason: Optional[str] = None
        self.mtp: str = "unknown"
        # Held from before_load to after_load. The loader clears its _process during a
        # load, and the supervisor must not read that as an unload.
        self.load_in_progress: bool = False
        self.mtp_reason: Optional[str] = "no load yet"
        self._supervisor: Optional[asyncio.Task] = None
        self._relaunch_task: Optional[asyncio.Task] = None
        self._lock: Optional[asyncio.Lock] = None
        self.last_error: str = ""

    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def decide(
        self,
        *,
        model_bytes: Optional[float],
        users: int,
        kv_bytes_per_user: Optional[float],
        gguf: bool = True,
    ) -> Dict[str, Any]:
        # before_load runs ahead of model classification, so a forced topology applied
        # unconditionally started an rpc-server and added llama arguments for a Transformers
        # or MLX load. Those loads succeed, after_load returns early because no llama backend
        # is loaded, and the peer process is never detached: the RPC port stays occupied and
        # the status stays on layer_split for a model that never used it.
        forced = forced_topology() if gguf else None
        plan = plan_topology(model_bytes, users = users, kv_bytes_per_user = kv_bytes_per_user or 0.0)
        if forced and forced != plan.get("topology"):
            plan["recommended"] = plan.get("topology")
            plan["topology"] = forced
            plan["reason"] = (
                f"forced by {ENV_TOPOLOGY}={forced}; the planner said {plan['recommended']}: "
                f"{plan.get('reason', '')}"
            )
        return plan

    @staticmethod
    def _request_with(request: Any, updates: Dict[str, Any]) -> Any:
        """``request`` with ``updates``: a copy for a pydantic model, in place otherwise."""
        if not updates:
            return request
        try:
            return request.model_copy(update = dict(updates))
        except AttributeError:
            for key, value in updates.items():
                setattr(request, key, value)
            return request

    async def before_load(
        self,
        request: Any,
        n_parallel: int,
        *,
        inherited_extra_args: Optional[List[str]] = None,
    ) -> Any:
        """Start the peer's rpc-server for a layer split and point llama-server at it, and in
        every topology ask for MTP at the Spark depth. The request comes back unchanged on
        every other path, including any failure. A ``--spec-type`` in
        ``inherited_extra_args`` counts as the caller's, not as room for this module."""
        if not enabled():
            return request
        # The route validates the pass-through args AFTER this runs, "up front so a managed-flag
        # collision returns 400 before any model work" -- but starting or restarting a peer is
        # model work, and it happens here, before that. Restoring the topology METADATA on a
        # rejected replacement is not enough for it: a forced reload of a live split stops the
        # peer's rpc-server on the way through, the outgoing llama-server's RPC allocations die
        # with it, and no later step can put that back, so a request refused with a 400 takes a
        # working split offline. Asked with the route's own validator, so the two cannot
        # disagree about what is refusable, and a refusal leaves the request and the live split
        # exactly as they were for the route to reject.
        # Scoped to a live peer on purpose. With nothing running there is nothing this could
        # destroy, and refusing early would only duplicate the route's own 400 while changing
        # the behaviour of every request that carries a managed flag.
        running_peer = self.peer_process
        if running_peer is not None and running_peer.alive:
            try:
                from core.inference.llama_server_args import validate_extra_args
                validate_extra_args(getattr(request, "llama_extra_args", None))
            except ValueError:
                logger.info(
                    "spark serving: the pass-through args will be refused by the route; "
                    "leaving the running peer and the current topology alone"
                )
                return request
            except Exception:
                pass  # not importable, or not a request shape this knows: behave as before
        # What the status surface says right now, kept so a REJECTED replacement can put it
        # back. The route validates after this runs and can raise 400/409 without unloading, so
        # a fall-back here would otherwise overwrite the live split's topology and group
        # metadata for a request that never replaced it: status then reports `single` while the
        # resident llama-server is still driving the peer, and the next load treats a perfectly
        # reusable rpc-server as dead. `load_failed` restores it on exactly that path.
        self._pre_load_state = (
            self.topology,
            self.reason,
            dict(self.plan) if isinstance(self.plan, dict) else self.plan,
            self.pipeline_groups,
            self.pipeline_groups_reason,
            self.split_config,
            self.split_config_reason,
        )
        self.load_in_progress = True
        try:
            # Nothing is torn down here: the load may be a no-op whose llama-server still
            # depends on the running peer.
            model_path = str(getattr(request, "model_path", "") or "")
            variant = getattr(request, "gguf_variant", None)
            local_file = cached_repo_file(model_path, variant)
            size = gguf_size_bytes(local_file)
            # A projector, a drafter, a LoRA or a control vector is resident for the whole
            # load, so pricing the base GGUF alone understates the node. That is the direction
            # that plans single for something which then does not fit, so a sidecar nobody can
            # size abstains rather than counting as zero.
            extras_for_sizing = list(
                getattr(request, "llama_extra_args", None)
                if getattr(request, "llama_extra_args", None) is not None
                else (inherited_extra_args or [])
            )
            sidecars, sidecars_unknown = sidecar_bytes(extras_for_sizing)
            if size is not None:
                size += sidecars
            if sidecars_unknown:
                # NOT abstaining. A sidecar the backend has yet to download is the ordinary
                # case, and answering "size unknown" to it means single, which is the topology
                # that cannot hold a large model: the cure would be worse than the omission.
                # What can be seen is charged, and the base term dominates either way.
                logger.info(
                    "spark serving: a sidecar could not be sized; the plan charges "
                    "%.1f GiB of sidecars it could see",
                    sidecars / _GIB,
                )
            remote_size = None
            if size is None:
                # Not cached yet. The hub knows what it weighs, and knowing that here is what
                # lets a model larger than one Spark be split on its FIRST load rather than
                # after a single-node launch that cannot fit. Asked once: its answer also
                # settles whether this request is a GGUF load at all.
                remote_size = await asyncio.to_thread(
                    remote_gguf_size_bytes,
                    model_path,
                    variant,
                    getattr(request, "hf_token", None),
                )
                size = remote_size
            # max_seq_length 0 means "let the backend size it", so after_load re-plans with
            # the context actually allocated.
            requested_ctx, cache_type, cache_type_v = effective_kv_settings(
                request, extras_for_sizing
            )
            kv_total = None
            if local_file and requested_ctx:
                kv_total = await asyncio.to_thread(
                    estimate_kv_bytes, local_file, requested_ctx, cache_type, cache_type_v
                )
            users = max(1, int(n_parallel))
            kv_per_user = (kv_total / users) if kv_total else 0.0
            plan = self.decide(
                model_bytes = size,
                users = users,
                kv_bytes_per_user = kv_per_user,
                gguf = (
                    looks_like_a_gguf_load(model_path, variant, local_file)
                    or remote_size is not None
                ),
            )
            # The header read and the --help probe are file and process work: off the loop.
            extra = getattr(request, "llama_extra_args", None)
            mtp = await asyncio.to_thread(
                mtp_plan,
                local_file,
                list(extra if extra is not None else (inherited_extra_args or [])),
                speculative_type = getattr(request, "speculative_type", None),
                spec_draft_n_max = getattr(request, "spec_draft_n_max", None),
                users = users,
            )
            if plan.get("topology") != "layer_split":
                self.plan = plan
                out = request
            else:
                peer = peer_address()
                if peer:
                    out = await self._start_layer_split(
                        request,
                        peer,
                        plan,
                        local_file,
                        users,
                        mtp = mtp,
                        inherited_extra_args = inherited_extra_args,
                    )
                else:
                    out = request
            # After the topology step, which may detach a replica: the verdict must survive it.
            self.mtp, self.mtp_reason = str(mtp["mtp"]), mtp.get("reason")
            if mtp["request"]:
                logger.info("spark serving: mtp %s (%s)", self.mtp, self.mtp_reason)
            return self._request_with(out, mtp["request"])
        except Exception as exc:
            self.last_error = f"before_load: {exc}"[:300]
            logger.warning(
                "spark serving: pre-load step failed, serving on this node only: %s", exc
            )
            return request

    async def _start_layer_split(
        self,
        request: Any,
        peer: str,
        plan: Dict[str, Any],
        local_file: Optional[str],
        slots: int = 1,
        mtp: Optional[Dict[str, Any]] = None,
        inherited_extra_args: Optional[List[str]] = None,
    ) -> Any:
        sc = _cluster()
        port = int(getattr(sc, "RPC_DEFAULT_PORT", RPC_PORT_DEFAULT))

        def _effective(req: Any) -> List[str]:
            """What llama-server will actually be launched with, which on a settings-Apply
            reload is not the request field. That path does not round-trip the extras, so the
            field is None and ``_resolve_inherited_extra_args`` puts the previous same-model
            load's extras back afterwards. Reading the bare field here would plan against an
            empty pass-through and, worse, writing a non-None list back would make that
            resolver skip inheritance and drop them for real."""
            extra = getattr(req, "llama_extra_args", None)
            return list(extra if extra is not None else (inherited_extra_args or []))

        groups = await asyncio.to_thread(
            pipeline_groups_plan,
            slots,
            _effective(request),
            projector = projector_blocks_pipeline_groups(
                local_file,
                disable_vision = bool(getattr(request, "disable_vision", False)),
            ),
        )
        if mtp is not None:
            # to_thread like pipeline_groups_plan above it: this reaches
            # llama_server_accepts_groups_with_drafter, whose first call for a binary is a
            # subprocess.run with a 30 s timeout, and a wedged binary would hold the whole
            # event loop, every active stream with it, for that long.
            await asyncio.to_thread(
                reconcile_split_speculation,
                groups,
                mtp,
                speculative_type = getattr(request, "speculative_type", None),
                extra_args = _effective(request),
            )

        def _with_rpc_args(req: Any) -> Any:
            extra = _effective(req)
            extra += layer_split_extra_args(
                peer, port, pipeline_groups = int(groups["pipeline_groups"])
            )
            updates: Dict[str, Any] = {"llama_extra_args": extra}
            if int(groups["pipeline_groups"]) > 1:
                # --parallel has to be a positive multiple of N, and it is a request field.
                updates["n_parallel"] = max(PARALLEL_MIN, int(groups["slots"]))
            self.pipeline_groups = int(groups["pipeline_groups"])
            self.pipeline_groups_reason = groups.get("reason")
            self.split_config = groups.get("split_config")
            self.split_config_reason = groups.get("split_config_reason")
            if self.split_config:
                logger.info(
                    "spark serving: layer split as %s: %s",
                    self.split_config,
                    self.split_config_reason,
                )
            if self.pipeline_groups:
                logger.info(
                    "spark serving: %s %d with %d slots (%d asked for)",
                    PIPELINE_GROUPS_FLAG,
                    self.pipeline_groups,
                    int(groups["slots"]),
                    int(groups["requested_slots"]),
                )
            else:
                logger.info("spark serving: no pipeline groups: %s", self.pipeline_groups_reason)
            return self._request_with(req, updates)

        def _fall_back(reason: str) -> Any:
            self.topology, self.reason = "single", reason
            self.plan = dict(plan, topology = "single", reason = reason)
            self.pipeline_groups, self.pipeline_groups_reason = 0, None
            self.split_config, self.split_config_reason = None, None
            logger.warning("spark serving: %s", reason)
            return request

        running = self.peer_process
        reusable = (
            self.topology == "layer_split"
            and running is not None
            and running.name == "ggml-rpc-server"
            and running.peer == peer
            and running.alive
        )
        if reusable and bool(getattr(request, "force_reload", False)):
            # The peer's rpc-server serves ONE client at a time and the outgoing llama-server
            # still holds that connection, so every forced reload of a split failed to connect.
            logger.info(
                "spark serving: forced reload of a layer split; restarting the peer "
                "rpc-server so the new llama-server can take the connection"
            )
            reusable = False
        if reusable and not await wait_for_port(peer, port, PEER_REUSE_TIMEOUT_S):
            # ``alive`` is the ssh session, which can outlive the server it carries: ask the
            # port before promising it to a launch.
            logger.warning(
                "spark serving: the peer rpc-server on %s:%s stopped answering; restarting it",
                peer,
                port,
            )
            reusable = False
        # These two are properties of the REQUEST, not of the peer, so they are decided before
        # the reuse shortcut and not after it. They used to sit below `if reusable:`, which meant
        # a second load reusing a live rpc-server returned past both of them: the same request
        # that correctly fell back to `single` on a cold start got `--rpc ... --device RPC0,CUDA0`
        # appended on a warm one. Order-dependent placement is the worst version of this bug,
        # because the first load looks like proof that the guard works.
        # Truthiness, not ``is not None``: the documented automatic-placement form is
        # ``gpu_ids: []``, and every other placement site in the backend reads it that way
        # (``if not gpu_ids``, ``automatic = not placement.requested_gpu_ids``). Reading an
        # empty list as an explicit pin refused the split for a GGUF that needs it and sent the
        # request to the single node that cannot hold it.
        if getattr(request, "gpu_ids", None):
            # The backend strips every --device pass-through when gpu_ids is set, because the
            # pin owns placement. A split needs --device RPC0,CUDA0 to keep the output layer
            # and the logits local, and without it llama.cpp's default CUDA-first enumeration
            # puts them on the peer: the measured slow path, silently. Better to say so than
            # to launch a split that is not the one this module priced.
            return _fall_back(
                "a layer split needs its own device order and an explicit GPU selection "
                "replaces it; clear the GPU pin to serve this model across both Sparks"
            )
        if argv_or_env_rpc(_effective(request)):
            # Their placement, not ours. Appending a second --rpc plus a managed device order,
            # split mode and tensor split either overrides a working manual split or reaches
            # llama-server as two conflicting placements, and starting a peer rpc-server for it
            # takes memory nothing will use. after_load records it, but that is after the fact.
            return _fall_back(
                "llama-server is being launched with a caller-supplied --rpc; leaving the "
                "placement alone"
            )
        if reusable:
            # The rpc-server is model-agnostic, so the next load can reuse it.
            self.plan = plan
            self.reason = str(plan.get("reason", ""))
            return _with_rpc_args(request)
        # Before anything is torn down or started: an rpc-server on a GPU that is already
        # holding somebody's work puts one of the two into an out-of-memory, and the plan was
        # priced against the whole node budget. Our own is excluded, since a reuse that got
        # this far has already been refused above.
        busy = await peer_gpu_conflict(
            peer, own_pids = [running.remote_pid] if running is not None else []
        )
        if busy:
            return _fall_back(f"layer split needs the peer GPU but {busy}")
        if running is not None:
            await self.detach()

        # Both bundles must speak the same RPC protocol and nothing stale may sit on the port.
        preflight = getattr(sc, "rpc_protocol_preflight", None)
        if callable(preflight):
            try:
                self.preflight = await asyncio.to_thread(preflight, peer, port)
            except Exception as exc:
                self.preflight = {"ok": None, "problems": [], "notes": [f"preflight failed: {exc}"]}
            if self.preflight.get("ok") is False:
                return _fall_back(
                    "layer split refused by the RPC preflight: "
                    + " ".join(str(p) for p in self.preflight.get("problems", []))
                )
            for note in self.preflight.get("notes", []):
                logger.info("spark serving: preflight: %s", note)
        local_rpc = rpc_server_binary()
        # Every name this platform would accept, the one the LOCAL bundle actually resolved
        # first. Hardcoding "ggml-rpc-server" discarded that basename, so a symmetrically
        # provisioned peer holding only the supported legacy `rpc-server` was reported missing
        # and every layer split that needed it fell back to one node. The llama-server lookup a
        # few lines down already passes `Path(argv[0]).name`; this was the one that did not.
        rpc_names: List[str] = []
        if local_rpc:
            rpc_names.append(osp.basename(local_rpc))
        rpc_names += [n for n in rpc_server_names() if n not in rpc_names]
        candidates: List[str] = []
        for rpc_name in rpc_names:
            candidates += [
                c for c in peer_binary_candidates(local_rpc, rpc_name) if c not in candidates
            ]
        rc, out, _err = await ssh_run(peer, find_binary_script(candidates))
        binary = out.strip().splitlines()[-1] if out.strip() else "MISSING"
        if rc != 0 or binary == "MISSING":
            return _fall_back(
                "layer split needed but the peer has no ggml-rpc-server (bundle "
                "b10796-mix-659e406 or newer ships it; run `unsloth spark provision`)"
            )
        present = False
        if local_file:
            rc2, out2, _ = await ssh_run(
                peer, f"test -f {shlex.quote(local_file)} && echo YES || echo NO"
            )
            present = rc2 == 0 and out2.strip().endswith("YES")
        self.peer_model_present = present
        bind = (os.environ.get(ENV_RPC_BIND) or "").strip() or peer
        argv = rpc_server_argv(binary, bind = bind, port = port, cache = present)
        log_dir = _log_dir()
        process = PeerProcess(
            "ggml-rpc-server", peer, argv, log_dir / "rpc-server.log" if log_dir else None
        )
        await process.start()
        try:
            # Not wait_for_port: an occupied port answers on the first probe while the child
            # that could not bind it exits, and adopting that listener leaves the split
            # attached to a server nothing here manages.
            ready = await wait_for_own_port(
                process, peer, port, PEER_START_TIMEOUT_S, cancelled = self._cancelled
            )
        except BaseException:
            # Readiness is the long wait here, so it is where a cancelled load lands. Without
            # this the rpc-server survives the cancellation holding the port, and the next
            # attempt finds it occupied.
            await process.stop()
            raise
        if not ready:
            tail = list(process.tail)[-3:]
            died = not process.alive
            await process.stop()
            return _fall_back(
                f"peer rpc-server did not accept on {peer}:{port} within "
                f"{PEER_START_TIMEOUT_S:.0f}s"
                + (" (it exited; the port may already be in use)" if died else "")
                + f" (last output: {tail})"
            )
        self.peer = peer
        self.peer_process = process
        self.topology = "layer_split"
        self.reason = str(plan.get("reason", ""))
        self.plan = plan
        self.relaunch_attempts = 0
        self.relaunch_gave_up = False
        logger.info("spark serving: layer split over %s:%s (%s)", peer, port, self.reason)
        return _with_rpc_args(request)

    async def load_failed(self) -> None:
        self.load_in_progress = False
        # Not every failed load leaves nothing running: the route validates and can raise 400/409
        # before it unloads, so a rejected replacement leaves the previous model loaded and still
        # being served. Tearing its topology down there would kill a working split or router over
        # a request that never touched it.
        backend = self.attached_backend
        if backend is not None and getattr(backend, "is_loaded", False):
            # The previous model is still resident and still being served, so the topology that
            # describes it must be the topology reported. before_load snapshots it precisely
            # because a rejected replacement can fall back and overwrite it without ever
            # replacing anything.
            if self._pre_load_state is not None:
                (
                    self.topology,
                    self.reason,
                    self.plan,
                    self.pipeline_groups,
                    self.pipeline_groups_reason,
                    self.split_config,
                    self.split_config_reason,
                ) = self._pre_load_state
            return
        self.mtp, self.mtp_reason = "unknown", "the load failed; nothing is running"
        if self.peer_process is not None or self.router is not None:
            await self.detach()

    async def after_load(self, llama_backend: Any, n_parallel: int) -> None:
        """Reconcile with what actually launched. Runs after every load, no-op reloads too."""
        self.load_in_progress = False
        if not enabled():
            return
        try:
            if not getattr(llama_backend, "is_loaded", False):
                await self.load_failed()
                return
            process = getattr(llama_backend, "_process", None)
            argv = list(getattr(process, "args", None) or [])
            port = getattr(llama_backend, "_port", None)
            self._record_launched_mtp(argv)
            if argv_or_env_rpc(argv):
                # A user-supplied --rpc is recorded, not managed.
                # ``router`` alone was not enough: a managed layer split has a peer
                # ggml-rpc-server and NO router, so a load bringing its OWN --rpc left the
                # managed rpc-server running and still tracked. It holds its share of the peer
                # GPU, so the caller's placement contends with it or does not fit, and the
                # status route goes on reporting the managed split's metadata for a split
                # nothing here manages any more.
                #
                # Which one it is cannot be read off "is there an --rpc": this branch is
                # reached by our OWN managed split too, whose argv this module put the --rpc
                # into. The endpoint tells them apart. Detaching on the flag alone tears down
                # the live split on every reconcile, which is how the reuse path
                # (test_before_load_reuses_a_live_rpc_server_and_after_load_reconciles)
                # catches it.
                if self.router is not None or (
                    self.peer_process is not None and not self._argv_names_our_peer(argv)
                ):
                    await self.detach()
                self.attached_backend = llama_backend
                self.attached_port = port
                self.topology = "layer_split"
                if self.peer_process is None:
                    self.reason = "llama-server launched with a user-supplied --rpc"
                    self.pipeline_groups = 0
                    self.pipeline_groups_reason = (
                        "llama-server launched with a user-supplied --rpc; nothing added"
                    )
                    self.split_config, self.split_config_reason = None, None
                self._ensure_supervisor()
                return
            if self.topology == "layer_split":
                await self.detach()
            if (
                self.topology == "replicas"
                and self.router is not None
                and self.attached_backend is llama_backend
                and self.attached_port == port
            ):
                return
            if self.router is not None or self.peer_process is not None:
                await self.detach()
            gguf_path = getattr(llama_backend, "gguf_path", None)
            size = gguf_size_bytes(gguf_path)
            # Prefer the aggregate. Once --parallel splits the cache, _effective_context_length is
            # the PER-SLOT window, so pricing that and then dividing by slots below reconstructs
            # one slot's cache instead of the whole one, short by up to the slot count. Falls back
            # to the per-slot value when the backend does not publish the aggregate.
            n_ctx = int(
                getattr(llama_backend, "_kv_cache_context_total", None)
                or getattr(llama_backend, "_effective_context_length", None)
                or getattr(llama_backend, "requested_n_ctx", 0)
                or 0
            )
            slots = max(
                1, int(getattr(llama_backend, "effective_parallel_slots", n_parallel) or n_parallel)
            )
            cache_types = getattr(llama_backend, "_effective_cache_types", None) or ()
            cache_type = cache_types[0] if cache_types else None
            # K and V are configurable apart. Pricing V as K understates an asymmetric cache by
            # up to 4x (q4_0 with f32), and understating is the direction that OOMs a node.
            cache_type_v = cache_types[1] if len(cache_types) > 1 else cache_type
            kv_total = (
                await asyncio.to_thread(
                    estimate_kv_bytes, gguf_path, n_ctx, cache_type, cache_type_v
                )
                if gguf_path
                else None
            )
            kv_per_user = (kv_total / slots) if kv_total else 0.0
            plan = self.decide(model_bytes = size, users = slots, kv_bytes_per_user = kv_per_user)
            self.plan = plan
            if plan.get("topology") != "replicas":
                self.topology = "single"
                self.reason = str(plan.get("reason", ""))
                return
            peer = peer_address()
            if not peer:
                self.topology, self.reason = "single", "no peer address"
                return
            await self._start_replicas(llama_backend, peer, plan, slots)
            if self._cancelled():
                # The cancel may have landed after the peer came up. Nothing is going to use
                # this topology -- the unload it belongs to has already reported back -- and a
                # peer llama-server left running holds its share of a 121.69 GiB node.
                await self.detach()
                self.topology, self.reason = "single", "the load was cancelled"
        except Exception as exc:
            self.last_error = f"after_load: {exc}"[:300]
            logger.warning(
                "spark serving: post-load step failed, serving on this node only: %s", exc
            )
            await self.detach()

    def _record_launched_mtp(self, argv: List[str]) -> None:
        """What launched, over the pre-load verdict: the backend may have declined the head,
        or the caller's own --spec-type may have won."""
        if not argv:
            return
        spec, depth = launched_spec_flags(argv)
        if spec is None:
            if self.mtp == "enabled":
                self.mtp = "not launched"
                self.mtp_reason = (
                    f"planned, but llama-server launched without {SPEC_TYPE_FLAG} "
                    f"(the backend declined it; see its log)"
                )
            return
        launched = f"launched with {SPEC_TYPE_FLAG} {spec}"
        if depth is not None:
            launched += f" {SPEC_DRAFT_N_MAX_FLAG} {depth}"
        types = {piece.strip() for piece in spec.split(",")}
        if types & {MTP_SPEC_TYPE, "mtp"}:
            if self.mtp not in ("user override",):
                self.mtp = "enabled"
            self.mtp_reason = launched
        elif self.mtp in ("enabled", "unknown"):
            self.mtp, self.mtp_reason = "other speculation", launched
        else:
            self.mtp_reason = f"{self.mtp_reason}; {launched}"

    async def _start_replicas(
        self, llama_backend: Any, peer: str, plan: Dict[str, Any], slots: int
    ) -> None:
        process = getattr(llama_backend, "_process", None)
        argv = list(getattr(process, "args", None) or [])
        port = getattr(llama_backend, "_port", None)
        gguf_path = getattr(llama_backend, "gguf_path", None)
        if not argv or not port or not gguf_path:
            self.topology, self.reason = "single", "cannot read this node's llama-server launch"
            return
        # Two round trips on purpose: the lookup script exits as soon as it finds the
        # binary, so a file check appended to it would never run.
        rc, out, _ = await ssh_run(
            peer,
            find_binary_script(peer_binary_candidates(argv[0], Path(argv[0]).name)),
            timeout = 25.0,
        )
        binary = out.strip().splitlines()[-1].strip() if out.strip() else "MISSING"
        if rc != 0 or binary == "MISSING":
            self.topology, self.reason = (
                "single",
                (
                    f"peer {peer} has no llama-server at the bundle path; run `unsloth spark provision`"
                ),
            )
            logger.warning("spark serving: %s", self.reason)
            return
        mismatch = await replica_build_mismatch(peer, binary)
        if mismatch:
            self.topology, self.reason = (
                "single",
                (
                    f"peer {peer} has a different llama-server build ({mismatch}); the replica "
                    f"is launched from this node's argv, so run `unsloth spark provision` to "
                    f"put the same build on both"
                ),
            )
            logger.warning("spark serving: %s", self.reason)
            return
        # Same argv, so every file it names has to exist at the same path on the peer.
        needed = launch_files(argv, str(gguf_path))
        generated = generated_launch_files(needed)
        if generated:
            failed = await replicate_generated_files(peer, generated)
            if failed:
                self.topology, self.reason = (
                    "single",
                    f"a generated launch file could not be put on {peer}: {failed}",
                )
                logger.warning("spark serving: %s", self.reason)
                return
            logger.info(
                "spark serving: copied %d generated launch file(s) to %s", len(generated), peer
            )
        needed = [p for p in needed if p not in set(generated)]
        # Size and mtime, not just existence. A replica is only interchangeable with the primary
        # if it is serving the SAME weights, and a peer holding a stale file at the same path --
        # an older quant of the same repo, a sidecar replaced locally but not there -- passes an
        # existence test and then answers from different weights. The router alternates between
        # them, so the same conversation gets model-dependent answers while every other parity
        # check (binary identity, argv, env) still reports a matched pair. Hashing 16 GiB over
        # ssh is not worth it; size plus mtime catches a replaced file, which is the case that
        # actually occurs, and disagreeing is treated exactly like the file being absent.
        mismatch: List[str] = []
        if needed:
            fmt = "; ".join(
                f"stat -c '%s %Y' {shlex.quote(p)} 2>/dev/null || echo NOSTAT" for p in needed
            )
            rc, out, _ = await ssh_run(peer, fmt, timeout = 25.0)
            remote = (out or "").strip().splitlines()
            self.peer_model_present = rc == 0 and len(remote) == len(needed)
            if self.peer_model_present:
                for path, line in zip(needed, remote):
                    try:
                        st = os.stat(path)
                        local_sig = f"{st.st_size} {int(st.st_mtime)}"
                    except OSError:
                        continue  # not ours to judge; the launch will report it
                    if line.strip() != local_sig:
                        mismatch.append(f"{path} (peer {line.strip()}, here {local_sig})")
                if mismatch:
                    self.peer_model_present = False
        else:
            self.peer_model_present = True
        if not self.peer_model_present:
            self.topology, self.reason = (
                "single",
                (
                    f"peer {peer} does not have the same {gguf_path} (or a sidecar the launch "
                    f"names): {'; '.join(mismatch) if mismatch else 'file missing'}. Copy it "
                    f"over the cluster link (rsync -a <file> {peer}:<same path>) to enable "
                    f"replicas"
                ),
            )
            logger.warning("spark serving: %s", self.reason)
            return
        busy = await peer_gpu_conflict(peer)
        if busy:
            self.topology, self.reason = (
                "single",
                f"replicas need the peer GPU but {busy}",
            )
            logger.warning("spark serving: %s", self.reason)
            return
        peer_port = int(port)
        # The environment the PRIMARY was actually spawned with, not this process's. The two are
        # not the same: the launch scrubs a copy conditionally, well beyond DENIED_ENV_VARS --
        # LLAMA_ARG_MMPROJ and _MMPROJ_URL under disable_vision, LLAMA_ARG_OVERRIDE_TENSOR,
        # _TENSOR_SPLIT and _SPLIT_MODE on a CPU-forced replay, _CTX_SIZE, _THREADS,
        # _KV_UNIFIED, _N_PARALLEL and _FLASH_ATTN under the memory fit. Rebuilding from
        # os.environ puts every one of them back on the peer, so the replica loads a projector
        # the primary refused or sizes a cache the plan never priced, and the router alternates
        # between two servers that are no longer the same server. Falls back to os.environ when
        # the backend cannot say, which is what this did before.
        peer_argv = with_replica_env(
            replica_argv(argv, binary = binary, host = peer, port = peer_port),
            replica_env(getattr(llama_backend, "launched_env", None) or None),
        )
        log_dir = _log_dir()
        self.peer = peer
        self.peer_process = PeerProcess(
            "llama-server", peer, peer_argv, log_dir / "peer-llama-server.log" if log_dir else None
        )
        await self.peer_process.start()
        router: Optional[SparkRouter] = None
        try:
            # The port has to belong to THIS launch before it is routed to. A stranger already
            # listening there is admitted as a healthy backend otherwise, and generation
            # traffic goes to whatever model it is holding.
            if not await wait_for_own_port(
                self.peer_process,
                peer,
                peer_port,
                PEER_REPLICA_START_TIMEOUT_S,
                cancelled = self._cancelled,
            ):
                tail = list(self.peer_process.tail)[-3:]
                await self.peer_process.stop()
                self.peer_process = None
                self.topology, self.reason = (
                    "single",
                    f"peer llama-server did not take {peer}:{peer_port} (last output: {tail})",
                )
                logger.warning("spark serving: %s", self.reason)
                return
            router = SparkRouter(on_backend_down = self._on_backend_down)
            router.add_backend("main", "127.0.0.1", int(port), slots, primary = True)
            router.add_backend("peer", peer, peer_port, slots)
            await router.start()
        except BaseException:
            # Cancellation lands here, and it does NOT reach after_load's except Exception.
            # Without this the peer llama-server keeps running, holding its memory and its
            # port, with nothing left that knows about it.
            if router is not None:
                try:
                    await router.stop()
                except Exception:
                    pass
            process, self.peer_process = self.peer_process, None
            if process is not None:
                try:
                    await process.stop()
                except Exception:
                    logger.warning("spark serving: peer teardown failed", exc_info = True)
            raise
        self.router = router
        self.attached_backend = llama_backend
        self.attached_port = int(port)
        self.topology = "replicas"
        self.reason = str(plan.get("reason", ""))
        self.relaunch_attempts = 0
        self.relaunch_gave_up = False
        logger.info(
            "spark serving: replicas on 127.0.0.1:%s and %s:%s behind %s (%s)",
            port,
            peer,
            peer_port,
            router.base_url,
            self.reason,
        )
        self._ensure_supervisor()

    def route_base_url(self, llama_backend: Any) -> Optional[str]:
        """The router's URL for this backend's requests, or None to go direct."""
        router = self.router
        if router is None or not router.running or router.listen_port is None:
            return None
        if self.attached_backend is not llama_backend:
            return None
        if getattr(llama_backend, "_port", None) != self.attached_port:
            return None  # respawned on a new port; direct until the supervisor re-points
        return router.base_url

    def tag_conversation(self, payload: Dict[str, Any], thread_id: Optional[str]) -> None:
        if thread_id and self.router is not None and self.router.running:
            payload[CONVERSATION_FIELD] = str(thread_id)

    def _ensure_supervisor(self) -> None:
        if self._supervisor is None or self._supervisor.done():
            self._supervisor = asyncio.create_task(self._supervise())

    async def _supervise(self) -> None:
        try:
            while True:
                await asyncio.sleep(SUPERVISOR_INTERVAL_S)
                backend = self.attached_backend
                if backend is None:
                    return
                if getattr(backend, "_process", None) is None:
                    if self.load_in_progress:
                        # NOT an unload. The loader clears _process before its download and
                        # preparation phase, so a replacement for an already-split model shows
                        # this transient state for the whole of it. Tearing down there kills
                        # the rpc-server the replacement has ALREADY been configured to use,
                        # and its llama-server then launches into nothing.
                        continue
                    logger.info(
                        "spark serving: this node's llama-server was unloaded; tearing the peer down"
                    )
                    await self.detach()
                    return
                await self._repoint_primary(backend)
                process = self.peer_process
                if process is not None and not process.alive and process.started_at is not None:
                    if self._relaunch_task is None or self._relaunch_task.done():
                        self._relaunch_task = asyncio.create_task(self._relaunch_peer())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("spark serving: supervisor stopped", exc_info = True)

    async def _on_backend_down(self, backend: Backend) -> None:
        if backend.primary:
            # Not ours to restart: LlamaCppBackend respawns its own child on the next request.
            return
        process = self.peer_process
        if process is not None and process.alive:
            # Health failed but the ssh session lives, so the server may still be loading.
            return
        if self._relaunch_task is None or self._relaunch_task.done():
            self._relaunch_task = asyncio.create_task(self._relaunch_peer())

    async def _relaunch_peer(self) -> None:
        process = self.peer_process
        if process is None or self.relaunch_gave_up:
            return
        while True:
            if self.attached_backend is None:
                return
            if self.relaunch_attempts >= len(RELAUNCH_BACKOFF_S):
                self.relaunch_gave_up = True
                self.relaunch_log.append(
                    {"at": time.time(), "event": "gave up", "attempts": self.relaunch_attempts}
                )
                logger.error(
                    "spark serving: peer %s on %s stayed down after %d relaunch attempts; "
                    "serving on this node only",
                    process.name,
                    process.peer,
                    self.relaunch_attempts,
                )
                return
            delay = RELAUNCH_BACKOFF_S[self.relaunch_attempts]
            self.relaunch_attempts += 1
            logger.warning(
                "spark serving: peer %s on %s exited (rc=%s); relaunch attempt %d/%d in %.0fs. "
                "Last output: %s",
                process.name,
                process.peer,
                process.returncode,
                self.relaunch_attempts,
                len(RELAUNCH_BACKOFF_S),
                delay,
                list(process.tail)[-3:],
            )
            self.relaunch_log.append(
                {
                    "at": time.time(),
                    "event": "relaunch",
                    "attempt": self.relaunch_attempts,
                    "delay_s": delay,
                    "returncode": process.returncode,
                    "tail": list(process.tail)[-3:],
                }
            )
            await asyncio.sleep(delay)
            if self.attached_backend is None:
                return
            try:
                await process.stop(timeout = 5.0)
                await process.start()
            except Exception as exc:
                logger.warning("spark serving: relaunch failed to spawn ssh: %s", exc)
                continue
            # Settle: an immediate exit means the peer refuses (port busy, OOM).
            await asyncio.sleep(3.0)
            if not process.alive:
                continue
            logger.info(
                "spark serving: peer %s relaunched (remote pid %s)",
                process.name,
                process.remote_pid,
            )
            # The budget bounds one crash loop, not the process lifetime: without this a peer
            # that restarts cleanly three times over days is then never recovered again. But
            # "alive after three seconds" is not recovery. A large model spends minutes in
            # load_model before it can OOM, so a peer that dies there is alive at the three
            # second mark every single time, the budget resets every single time, and the
            # three attempt bound is never reached: the peer reloads and fails forever, each
            # cycle paying for the whole load. Recovery is the router seeing it answer, which
            # is the same condition the traffic is gated on, so a peer that never becomes
            # healthy spends the budget and stops.
            if not await self._await_peer_healthy(process, RELAUNCH_HEALTHY_TIMEOUT_S):
                self.relaunch_log.append(
                    {"at": time.time(), "event": "relaunched but never healthy"}
                )
                continue
            self.relaunch_attempts = 0
            self.relaunch_log.append({"at": time.time(), "event": "recovered"})
            return

    async def _repoint_primary(self, backend: Any) -> None:
        """Follow this node's llama-server to a new port, once its load has finished.

        ``load_in_progress`` for the same reason the unload branch above it has it, and it was
        only on that one. A replacement GGUF gets its new port BEFORE the load finishes, so this
        could see the new healthy process mid-load, repoint the primary and move
        ``attached_port`` -- and ``after_load`` then reads its own matching backend and port as a
        no-op reload and returns without replacing the peer. The router is left alternating
        between the new primary model and the stale peer model, which is the exact failure the
        whole replica path exists to prevent. ``after_load`` rebuilds both sides; this must not
        race it."""
        if self.router is None or self.load_in_progress:
            return
        port = getattr(backend, "_port", None)
        if port and port != self.attached_port and getattr(backend, "_healthy", False):
            await self.router.set_backend_address("main", "127.0.0.1", int(port))
            self.attached_port = int(port)

    def _argv_names_our_peer(self, argv: Sequence[str]) -> bool:
        """Whether an ``--rpc`` in *argv* points at the rpc-server this module launched.

        Host AND port: a caller may legitimately run their own rpc-server on the same peer,
        and host alone would read that as ours and leave both running."""
        process = self.peer_process
        if process is None or not self.peer:
            return False
        remote = [str(a) for a in (getattr(process, "argv", None) or [])]
        port = ""
        for index, token in enumerate(remote):
            if token == "-p" and index + 1 < len(remote):
                port = remote[index + 1].strip()
        if not port:
            return False
        ours = f"{self.peer}:{port}"
        tokens = [str(a) for a in (argv or [])]
        for index, token in enumerate(tokens):
            name, sep, inline = token.partition("=")
            if name != "--rpc":
                continue
            value = inline if sep else (tokens[index + 1] if index + 1 < len(tokens) else "")
            if any(part.strip() == ours for part in str(value).split(",")):
                return True
        return False

    def _cancelled(self) -> bool:
        """Whether the load this orchestration belongs to has been cancelled.

        A scoped unload arriving mid-load only sets the event; nothing here used to read it, so
        a cancel during replica startup was ignored for up to PEER_REPLICA_START_TIMEOUT_S while
        the unload it triggered had already reported ``unloaded`` -- and then this attached a
        peer to a backend the caller had been told was gone."""
        event = self._cancel_event
        try:
            return bool(event is not None and event.is_set())
        except Exception:
            return False

    async def _await_peer_healthy(self, process: Any, timeout: float) -> bool:
        """Whether the router sees this peer answering within *timeout*.

        The router health-probes every backend it holds; this only reads the verdict rather
        than probing separately, so recovery is judged by the same signal that decides whether
        traffic is sent there. A peer that dies while waiting, or a router that goes away under
        us, is not healthy and says so at once instead of burning the timeout."""
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if self.attached_backend is None or process is not self.peer_process:
                return False  # detached, or this relaunch has been superseded
            if not process.alive:
                return False
            router = self.router
            if router is None:
                return False
            if any(b.healthy and not b.primary for b in getattr(router, "backends", ())):
                return True
            await asyncio.sleep(RELAUNCH_HEALTH_POLL_S)
        return False

    async def detach(self) -> None:
        """Stop the router, the supervisor and the peer process. Idempotent."""
        async with self.lock():
            supervisor, self._supervisor = self._supervisor, None
            if supervisor is not None and supervisor is not asyncio.current_task():
                supervisor.cancel()
                try:
                    await supervisor
                except (asyncio.CancelledError, Exception):
                    pass
            relaunch, self._relaunch_task = self._relaunch_task, None
            if relaunch is not None and relaunch is not asyncio.current_task():
                relaunch.cancel()
                try:
                    await relaunch
                except (asyncio.CancelledError, Exception):
                    pass
            router, self.router = self.router, None
            if router is not None:
                await router.stop()
            process, self.peer_process = self.peer_process, None
            if process is not None:
                try:
                    await process.stop()
                except Exception:
                    logger.warning("spark serving: peer teardown failed", exc_info = True)
            self.attached_backend = None
            self.attached_port = None
            self.pipeline_groups, self.pipeline_groups_reason = 0, None
            self.split_config, self.split_config_reason = None, None
            if self.topology != "single":
                self.topology = "single"
                self.reason = "detached"

    def status(self) -> Dict[str, Any]:
        if self.topology == "layer_split":
            groups, groups_reason = self.pipeline_groups, self.pipeline_groups_reason
            config, config_reason = self.split_config, self.split_config_reason
        else:
            groups, groups_reason = 0, f"not a layer split (topology {self.topology})"
            config, config_reason = None, f"not a layer split (topology {self.topology})"
        return {
            "enabled": enabled(),
            "topology": self.topology,
            "reason": self.reason,
            "peer": self.peer,
            "plan": self.plan,
            "preflight": self.preflight,
            "peer_model_present": self.peer_model_present,
            "pipeline_groups": groups,
            "pipeline_groups_reason": groups_reason,
            "split_config": config,
            "split_config_reason": config_reason,
            "mtp": self.mtp,
            "mtp_reason": self.mtp_reason,
            "router": self.router.status() if self.router is not None else None,
            "peer_process": self.peer_process.snapshot() if self.peer_process is not None else None,
            "relaunch_attempts": self.relaunch_attempts,
            "relaunch_gave_up": self.relaunch_gave_up,
            "relaunch_log": self.relaunch_log[-10:],
            "last_error": self.last_error,
        }


_STATE: Optional[SparkServing] = None


def state() -> SparkServing:
    global _STATE
    if _STATE is None:
        _STATE = SparkServing()
    return _STATE


def reset_for_tests() -> None:
    global _STATE, _CLUSTER, _CLUSTER_LOOKED_UP
    _STATE = None
    _CLUSTER = None
    _CLUSTER_LOOKED_UP = False
    _HELP_TEXT.clear()
    _ACCEPTS.clear()


def route_base_url(llama_backend: Any) -> Optional[str]:
    if _STATE is None:
        return None
    return _STATE.route_base_url(llama_backend)


def tag_conversation(payload: Dict[str, Any], thread_id: Optional[str]) -> None:
    if _STATE is None:
        return
    _STATE.tag_conversation(payload, thread_id)


def current_topology() -> Optional[str]:
    # Status polls run this constantly and it used to go through enabled(), which does rail
    # discovery: a sysfs walk and an `ip` fork, measured at 16.2 ms per call on a paired Spark,
    # synchronously on the event loop. It is not needed to answer this. Nothing is attached
    # unless enabled() was already true when the load ran, and with nothing attached the honest
    # answer is None on any machine, so the cheap check gives the same result off a Spark too.
    st = state()
    if st.attached_backend is None and st.peer_process is None and st.router is None:
        return None
    return st.topology


async def before_load(
    request: Any,
    n_parallel: int,
    *,
    inherited_extra_args: Optional[List[str]] = None,
    cancel_event: Any = None,
) -> Any:
    if not enabled():
        return request
    st = state()
    st._cancel_event = cancel_event
    return await st.before_load(request, n_parallel, inherited_extra_args = inherited_extra_args)


async def after_load(
    llama_backend: Any,
    n_parallel: int,
    *,
    cancel_event: Any = None,
) -> None:
    if not enabled():
        return
    st = state()
    if cancel_event is not None:
        st._cancel_event = cancel_event
    try:
        await st.after_load(llama_backend, n_parallel)
    finally:
        st._cancel_event = None


async def load_failed() -> None:
    if _STATE is None or not enabled():
        return
    await _STATE.load_failed()


async def reconcile_internal_load() -> None:
    """Drop a two-node topology before a load that did not come through ``before_load``.

    The auto-switch and preview paths call the loader directly, so nothing here plans or
    re-attaches for them. With replicas up that is not a missed optimisation but wrong answers:
    the peer keeps serving the model it was given, the supervisor repoints only the local
    backend, and the router then alternates requests between two sets of weights. A layer split
    is worse still, since the local server's RPC device is about to be pulled out from under it.

    Falling back to this node alone is the correct answer for both, and the next load through
    ``/load`` re-attaches. A no-op with nothing attached, so it costs an untopologied Spark and
    every other machine nothing."""
    if _STATE is None or _STATE.attached_backend is None and _STATE.peer_process is None:
        return
    logger.info("spark serving: serving on this node only for a load that bypassed the planner")
    await _STATE.detach()


async def shutdown() -> None:
    if _STATE is None:
        return
    await _STATE.detach()


def status() -> Dict[str, Any]:
    if not enabled():
        return {"enabled": False, "topology": None, "reason": "not a paired DGX Spark"}
    return state().status()
