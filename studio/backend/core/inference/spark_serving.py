# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two-Spark serving orchestrator: picks a topology and runs the peer half of it.

When Studio or Desktop loads a GGUF on a DGX Spark that has a cabled, configured peer,
this module decides between three layouts and runs whichever one the workload allows:

  single       the model fits on one node and expected concurrency is low: nothing
               changes, one llama-server on this node.
  replicas     the model plus KV fits on one node and the load asked for 8 or more
               parallel slots: a second llama-server on the peer (launched over ssh
               from the same bundle) and ``SparkRouter`` on this node spreading requests
               across both. Measured on this pair with Qwen3.8-27B: 1.30x aggregate
               decode at 8 users, 1.75x at 16, 1.91x at 32, with better per-user latency.
  layer_split  the model does not fit on one node: ``ggml-rpc-server`` on the peer and
               this node's llama-server launched with ``--rpc <peer>:<port> --device
               RPC0,CUDA0 -sm layer`` (RPC first so the output layer and the logits stay
               on the local GPU; see ``layer_split_extra_args``). Pipeline parallelism is
               enabled by llama.cpp
               itself when the RPC backend advertises async and events (b10796 does),
               so no flag is added for it. When the bundle's llama-server has
               ``--pipeline-groups`` (the unslothai/llama.cpp fork: N contexts from one
               model, the slots partitioned across them, N interleaved decode loops so
               one group's batch runs on the peer's layers while another's runs here)
               the launch adds ``--pipeline-groups 2`` and an even slot count. Measured
               on Qwen3.8-27B: two groups 1.31x to 1.37x of one context on the same split
               at 32 to 128 concurrent rows (130 to 170 tok/s, 1.12x to 1.13x of a single
               Spark) with both GPUs at 76 to 79 percent. The flag is probed for with
               ``llama-server --help`` once per binary; a bundle without it launches
               exactly as before, and ``UNSLOTH_SPARK_PIPELINE_GROUPS=0`` turns it off.
               The fork keeps the flag out of its usage text, so a build whose ``--help``
               does not name it is also tried for real (``llama_server_accepts``).
               Groups and speculative decoding are no longer either/or: from
               ``GROUPS_X_MTP_MIN_ROWS`` (16) concurrent rows up, a split asks for both on
               a server that takes them together (the per-group speculative state of
               unslothai/llama.cpp PR #187, probed for by running the server against a
               model that does not exist, since the refusal is a load-time check), and
               below that crossover it keeps the speculation and drops the groups. See
               ``reconcile_split_speculation``; the status says which of the three ran.

In every topology a GGUF that ships its own MTP head (``<arch>.nextn_predict_layers`` in
the header: Qwen3.5-4B-MTP, Qwen3.8-27B) self-speculates. The backend's own speculative
path emits ``--spec-type draft-mtp`` for such a file; this module reads the header, checks
the bundle's llama-server for ``--spec-type`` and asks for the Spark-measured draft depth
(``--spec-draft-n-max 3``: 2.6x / 1.9x / 1.6x aggregate decode at 1 / 4 / 8 users on the
27B, 2.0x / 1.7x / 1.5x on the 4B, against the backend's GPU default of 2). A caller's
``speculative_type``, ``spec_draft_n_max`` or ``--spec-type`` / draft flags are left
alone, ``UNSLOTH_SPARK_MTP=0`` launches without speculation, and the status reports
``mtp`` (enabled / no head / server too old / user override / disabled by env) from the
argv that actually launched, beside ``split_config`` and ``split_config_reason``.

The decision is ``studio/spark_cluster.recommend_topology`` (pure, measured); this
module only gathers its inputs (weights on disk, KV per slot from the GGUF header, the
slot count the load asked for) and runs the processes the answer needs. The
llama-server it probes is the one the backend will launch, resolved by the backend's
own ``_find_llama_server_binary`` and never by a search of this module's own, and the
rpc-server is taken from beside it, so the peer -- which ``unsloth spark provision``
mirrors from this node -- runs the same build as this node does.

Nothing here runs on the normal single-machine path. ``enabled()`` is the gate every
entry point calls first: it answers False from ``spark_cluster.is_dgx_spark()`` (two
string compares) on Windows, macOS, WSL and every non-Spark Linux box, and False on a
Spark with no configured peer. The module imports nothing POSIX-only; ssh is only
spawned once the gate has passed, and every ssh runs as an asyncio subprocess or in a
worker thread so the Studio event loop never waits on the peer.
"""

from __future__ import annotations

import asyncio
import collections
import getpass
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
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from core.inference.spark_router import CONVERSATION_FIELD, Backend, SparkRouter

logger = logging.getLogger(__name__)

ENV_TOGGLE = "UNSLOTH_SPARK_SERVING"  # "0" disables; anything else leaves detection alone
ENV_TOPOLOGY = "UNSLOTH_SPARK_TOPOLOGY"  # auto | single | replicas | layer_split
ENV_PEER = "UNSLOTH_SPARK_PEER"  # peer address override (tests, unusual cabling)
ENV_RPC_BIND = "UNSLOTH_SPARK_RPC_BIND"  # ggml-rpc-server -H on the peer; default is the peer's cluster address
ENV_PREFILL_HEAVY = (
    "UNSLOTH_SPARK_PREFILL_HEAVY"  # "1": tell the planner the work is long-prompt prefill
)
# Layer split only: N adds ``--pipeline-groups N`` when the bundle has the flag; unset
# or 0 adds nothing. Opt-in for now: see PIPELINE_GROUPS_DEFAULT.
ENV_PIPELINE_GROUPS = "UNSLOTH_SPARK_PIPELINE_GROUPS"
# Every topology: "0" launches a paired Spark's llama-server without speculative decoding
# (speculative_type "off" unless the caller set the field or owns --spec-type in the extras);
# unset or anything else lets a GGUF with an MTP head draft at MTP_DRAFT_N_MAX.
ENV_MTP = "UNSLOTH_SPARK_MTP"

TOPOLOGIES = ("single", "replicas", "layer_split")
RPC_PORT_DEFAULT = 50052
PROMPT_TOKENS_DEFAULT = 512  # the planner's measured table is keyed by prompt length
# Two groups by default on a layer split. Only added when the bundle's llama-server has the flag
# (unslothai/llama.cpp PR #187, ready for review; in no prebuilt yet), so a bundle without it
# launches exactly as before. Measured on Qwen3.8-27B split across two Sparks with the device
# order below, 32 / 64 / 128 concurrent, two repeats: one context 99.7 / 117.2 / 124.2 tok/s,
# two groups 130.4 / 157.2 / 170.1 (1.31x to 1.37x), both GPUs 76 to 79 percent against 42 to
# 47. With the old CUDA0,RPC0 order two groups LOST (75.5 against 94.9), so the order is a
# precondition. UNSLOTH_SPARK_PIPELINE_GROUPS=0 turns it off, =N sets the count.
PIPELINE_GROUPS_DEFAULT = 2
PIPELINE_GROUPS_FLAG = "--pipeline-groups"
# The slot count travels as LoadRequest.n_parallel, whose range is llama_server_args
# PARALLEL_MIN..PARALLEL_MAX. Mirrored here rather than imported: this module is loaded
# by the CLI and by tests that never import the backend's request models.
PARALLEL_MIN = 1
PARALLEL_MAX = 64
# MTP self speculation. A GGUF whose header has <arch>.nextn_predict_layers > 0 ships its own
# draft head (Unsloth's Qwen3.5-4B-MTP and Qwen3.8-27B do; the Qwen3.6-35B-A3B does not). The
# backend's speculative-decoding path (LlamaCppBackend._build_speculative_flags in "auto" mode)
# already emits --spec-type draft-mtp for such a file when its llama-server has --spec-type, and
# falls back cleanly when the head or the flag is missing; this module never emits a second
# --spec-type, because extras that own --spec-type switch that path off together with its memory
# budget, its sub-3B and MLA gates and its retry without speculation. What a Spark changes is
# the draft depth: the backend's GPU default is 2 (a B200 bench). Measured on this pair with
# b10796, real-text prompts, npp 128 / ntg 256, --parallel 8, one clock state (1690 MHz):
#
#   Qwen3.8-27B UD-Q4_K_XL     baseline 11.2 / 35.7 / 52.1 tok/s at 1 / 4 / 8 users
#     draft-mtp n-max 3         28.6 / 63.4 / 81.0  = 2.61x / 1.87x / 1.59x  accept 0.88 -> 0.74
#     draft-mtp n-max 8         32.0 / 63.4 / 70.6  = 2.93x / 1.87x / 1.38x  accept 0.62 -> 0.46
#   Qwen3.5-4B-MTP UD-Q4_K_XL  baseline 57.6 / 162.0 / 188.1
#     draft-mtp n-max 3        117.6 / 270.8 / 274.3 = 2.04x / 1.67x / 1.46x
#     draft-mtp n-max 2        106.9 / 238.5 / 259.8 = 1.86x / 1.47x / 1.38x
#
# n-max 3 is the mixed-traffic choice (n-max 8 wins only at one user or on long outputs and
# loses at 8), so a Spark load of a GGUF with a head asks for --spec-draft-n-max 3 unless the
# caller set a depth, a speculative_type, or owns --spec-type / a draft model in the extras.
# A replica copies this node's argv and a layer split's extras add only the RPC arguments, so
# the same flags reach every topology. Draft models and n-gram speculation are not turned on
# by default: on this pair both are a single-user trick and a 0.4x to 0.8x loss from 4 users.
MTP_SPEC_TYPE = "draft-mtp"
MTP_DRAFT_N_MAX = 3
SPEC_TYPE_FLAG = "--spec-type"
SPEC_DRAFT_N_MAX_FLAG = "--spec-draft-n-max"
# Pass-through flags that make the launch's speculative decoding the caller's: the backend's
# own rule (--spec-type / --spec-default, see _extra_args_set_spec_type) plus a draft model or
# any draft knob, which nobody passes without meaning their own speculation setup.
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
# Flags the fork still refuses together with --pipeline-groups N > 1, unchanged by PR #187
# (tools/server validate_pipeline_groups): one projector, one control vector set and one
# idle timer per server, none of them per group. Studio's own projector for a vision model
# is emitted by the backend rather than through these extras, so a caller's --mmproj is what
# is visible here; a split of a vision model is not a supported combination either way.
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
# ── Pipeline groups AND speculation on the same split: the crossover ──────────────────
# Mirrors of spark_cluster.GROUPS_X_MTP_* (spark_cluster carries the full table and
# groups_x_mtp_note; these are here so this module needs nothing loaded to decide).
# Measured on the pair, Qwen3.8-27B UD-Q4_K_XL split across the two Sparks with the fork of
# PR #187, --kv-unified --parallel 32 --cache-ram 0, npp 128 / ntg 256, two repeats in
# opposite arm order, both nodes pinned at 1700 MHz, decode tok/s:
#
#   rows | 1 context  | 1 context + MTP | 2 groups   | 2 groups + MTP | one Spark + MTP
#      8 | 51.2  51.6 |   91.3   84.3   | 50.6  48.9 |   88.1   81.6  |  85.6 (unpinned)
#     32 | 96.4  97.9 |  111.0  113.2   |145.7 133.3 |  154.9  150.2  |  83.8 (unpinned)
#
# --kv-unified is on every one of these, because Studio puts it on every GGUF load and the
# first version of this table did not: on a two-group split at 32 rows it is worth 1.27x on
# its own (137.1 -> 173.7 tok/s, everything else equal), and nothing at all on a one-context
# split, since one shared KV buffer only pays once the rows are split over two groups.
#
# At 32 rows the two together win: 152.5 tok/s on the means, 1.36x of one context with MTP
# and 1.09x of the groups alone. At 8 rows one context with MTP wins, 87.8 against 84.8, by
# 3 percent, so under the crossover the choice barely matters; above it the prize is large.
# Acceptance is unchanged by the second group (0.75 / 0.70 with one context, 0.74 / 0.71
# with two). Nothing was measured in between, so the crossover is the geometric midpoint of
# the two bracketing points and both sides of it are measured.
GROUPS_X_MTP_MIN_ROWS = 16  # spark_cluster.GROUPS_X_MTP_CROSSOVER_ROWS
GROUPS_X_MTP_OVER_MTP_ONLY = {8: 0.97, 32: 1.36}  # both over one context with MTP
GROUPS_X_MTP_OVER_GROUPS_ONLY = {8: 1.71, 32: 1.09}  # both over two groups alone
# Mirror of spark_cluster.SPLIT_TENSOR_SPLIT_EVEN. An EXPLICIT even --tensor-split, because
# llama.cpp's default divides the layers by each device's free memory at load time and so does
# not put the boundary in the same place twice. Measured 2026-09-06 at 128 rows with two groups,
# both nodes pinned at 1700 MHz, peer blocks -> decode tok/s: 27 -> 192.1, 30 -> 199.7,
# 33 -> 206.4, 34 (what the default chose that day) -> 201.2, 36 -> 192.5. The even split lands
# on 33 and is the best of the five, worth about 2.6 percent, and it is reproducible.
SPLIT_TENSOR_SPLIT_EVEN = "0.5,0.5"
# What a layer split was launched as, for the status surface beside ``mtp``.
SPLIT_CONFIG_BOTH = "groups + speculation"
SPLIT_CONFIG_SPEC = "one context + speculation"
SPLIT_CONFIG_GROUPS = "groups, no speculation"
SPLIT_CONFIG_PLAIN = "one context, no speculation"
HELP_PROBE_TIMEOUT_S = 20.0  # llama-server --help; a hung binary is a missing flag
# The load-time probe for --pipeline-groups TOGETHER with a drafter (below). The server
# validates that pair inside load_model, before it reads the weights, so a run against a
# model path that cannot exist reaches the verdict without touching a GPU or a real file.
GROUPS_DRAFTER_PROBE_TIMEOUT_S = 30.0
_GROUPS_REFUSAL_TEXT = "is not supported together with"
_PROBE_MODEL_NAME = "unsloth-spark-pipeline-groups-probe.gguf"
RELAUNCH_BACKOFF_S = (5.0, 15.0, 45.0)  # bounded: three attempts, then the peer stays down
PEER_START_TIMEOUT_S = 20.0  # for the rpc-server port to accept; the model load is separate
PEER_REUSE_TIMEOUT_S = 3.0  # a running peer has to answer this fast to be reused
SUPERVISOR_INTERVAL_S = 1.0
_LOG_TAIL = 60
_GIB = 2**30

# Mirrors of spark_cluster's numbers, used only when that module is unavailable.
_SPARK_USABLE_GIB = 121.69
_SERVE_OVERHEAD_GIB = 8.0


# ── Cluster module access ────────────────────────────────────────────────────

_CLUSTER: Any = None
_CLUSTER_LOOKED_UP = False


def _cluster():
    """``studio.spark_cluster``, loaded lazily and by path when the package is not
    importable (the backend runs with ``studio/backend`` as its root)."""
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


def peer_address() -> Optional[str]:
    override = (os.environ.get(ENV_PEER) or "").strip()
    if override:
        return override
    if not is_spark():
        return None
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
    """Whether two-Spark serving may run at all. Off everywhere but a paired Spark."""
    if (os.environ.get(ENV_TOGGLE) or "").strip() == "0":
        return False
    if not is_spark():
        return False
    return peer_address() is not None


def forced_topology() -> Optional[str]:
    value = (os.environ.get(ENV_TOPOLOGY) or "").strip().lower().replace("-", "_")
    return value if value in TOPOLOGIES else None


# ── Topology decision inputs ─────────────────────────────────────────────────


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
    """``spark_cluster.recommend_topology`` with this module's inputs filled in.

    An unknown model size answers ``single`` with a reason rather than a guess, the
    same refusal the cluster planner makes everywhere else.
    """
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


def gguf_size_bytes(path: Optional[str]) -> Optional[int]:
    """Size of a GGUF plus its sibling shards, or None when it cannot be told."""
    if not path:
        return None
    try:
        p = Path(path)
        if not p.is_file():
            return None
        total = p.stat().st_size
        match = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", p.name)
        if match:
            prefix, _first, count = match.groups()
            total = 0
            for index in range(1, int(count) + 1):
                shard = p.with_name(f"{prefix}-{index:05d}-of-{count}.gguf")
                try:
                    total += shard.stat().st_size
                except OSError:
                    pass
        return total
    except OSError:
        return None


def cached_repo_file(model_path: str, variant: Optional[str]) -> Optional[str]:
    """The GGUF a not-yet-resolved load request will use, before the load runs.

    A local file answers directly. A repo id is looked up in the HF cache for the
    first shard that names the variant; anything else answers None, and the caller
    waits for the resolved ``gguf_path`` after the load instead of guessing.
    """
    if not model_path:
        return None
    expanded = osp.expanduser(model_path)
    if osp.isfile(expanded):
        return expanded
    if "/" not in model_path or osp.isabs(model_path):
        return None
    cache = os.environ.get("HF_HUB_CACHE") or osp.join(
        os.environ.get("HF_HOME") or osp.expanduser("~/.cache/huggingface"), "hub"
    )
    root = osp.join(cache, "models--" + model_path.replace("/", "--"), "snapshots")
    if not osp.isdir(root):
        return None
    pattern = f"*{variant}*.gguf" if variant else "*.gguf"
    candidates = sorted(glob.glob(osp.join(root, "*", "**", pattern), recursive = True))
    candidates = [c for c in candidates if "mmproj" not in osp.basename(c).lower()]
    return candidates[0] if candidates else None


# Bytes per KV element per llama.cpp cache type; f16 when unknown. Same table as
# LlamaCppBackend's _kv_bytes_per_elem, kept local so this module stays importable
# without the 32k-line backend.
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
) -> Optional[int]:
    """KV cache for ``n_ctx`` tokens at ``cache_type`` (f16 default), from the GGUF header.

    Uses the ``gguf`` reader the backend already depends on; answers None when the
    file or the keys cannot be read, and the caller charges zero plus the planner's
    fixed overhead rather than refusing to plan. SWA layers are charged in full, so
    the estimate errs high for models that have them.
    """
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
        return int(2 * n_layer * int(n_ctx) * n_kv * head_dim * kv_bytes_per_elem(cache_type))
    except Exception:
        return None


# ── Remote process management ────────────────────────────────────────────────


def _ssh_user() -> str:
    """This session's login, which is the peer's too (`unsloth spark provision` mirrors
    the install as one account). ``spark_cluster._ssh_user`` owns the rule and every
    ssh site in the cluster module uses it; the copy below is only for a backend that
    cannot load that module: the environment first, then the login database, so a
    service context that sets no USER still names the right account."""
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
        return 255, "", str(exc)


def peer_path(path: Path) -> str:
    """``path`` as the peer's shell should expand it: ``$HOME/...`` under our home.

    ``spark_cluster._peer_relative_path`` owns the rule (provision copies to the same
    place on the peer, whose home may differ, so a path under our home is sent home
    relative and expanded there); its ``~/`` form is turned into ``$HOME/`` because
    the remote checks quote their paths and a quoted tilde does not expand.
    """
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
    """Where ``name`` should be on the peer, most likely first.

    The directory the local binary runs from (the pair is provisioned by rsync, so
    layouts match), then the managed bundle as ``llama_bundle_dir`` resolves it, then
    the source-build fallback ``unsloth spark serve`` documents, then PATH.
    """
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


def launch_files(argv: List[str], gguf_path: str) -> List[str]:
    """Every file the launch names: the weights plus any absolute path that is a file
    here (mmproj, drafter, adapters). The replica needs all of them at the same path."""
    files = [gguf_path]
    for arg in argv[1:]:
        if arg != gguf_path and osp.isabs(arg) and osp.isfile(arg):
            files.append(arg)
    return files


# Node-local state the replica must not share or need: the slot KV save directory is a
# cache on this node's disk and llama-server refuses a path that does not exist.
_REPLICA_DROPPED_FLAGS = ("--port", "--host", "--slot-save-path")


def replica_argv(local_argv: List[str], *, binary: str, host: str, port: int) -> List[str]:
    """The peer's llama-server argv: the local launch, re-pointed.

    Same flags, same model path, same slots, same context and cache types, because a
    replica that differs from the primary would answer the same request differently.
    Only the binary, ``--host`` and ``--port`` change, and node-local paths are dropped.
    """
    out: List[str] = [binary]
    skip = 0
    for arg in local_argv[1:]:
        if skip:
            skip -= 1
            continue
        if arg in _REPLICA_DROPPED_FLAGS:
            skip = 1
            continue
        if arg.startswith(tuple(f"{flag}=" for flag in _REPLICA_DROPPED_FLAGS)):
            continue
        out.append(arg)
    out += ["--host", host, "--port", str(port)]
    return out


def redacted_argv(argv: List[str]) -> List[str]:
    """``argv`` with the value after ``--api-key`` replaced, for logs and status."""
    out = list(argv)
    for index, arg in enumerate(out):
        if arg == "--api-key" and index + 1 < len(out):
            out[index + 1] = "<redacted>"
        elif arg.startswith("--api-key="):
            out[index] = "--api-key=<redacted>"
    return out


def rpc_server_argv(binary: str, *, bind: str, port: int, cache: bool) -> List[str]:
    """``ggml-rpc-server`` on the peer, bound to the cluster interface and caching
    tensors (``-c``) only when the model file is there to cache from."""
    argv = [binary, "-H", bind, "-p", str(port)]
    if cache:
        argv.append("-c")
    return argv


# The rpc-server under every name the bundles have used, in ``spark_cluster``'s order.
_RPC_SERVER_NAMES = ("ggml-rpc-server", "rpc-server", "ggml-rpc-server.exe", "rpc-server.exe")


def llama_server_binary() -> Optional[str]:
    """The llama-server this node will actually LAUNCH, or None.

    There is exactly one resolver for that binary and it is the llama.cpp backend's
    own ``LlamaCppBackend._find_llama_server_binary``: the load path launches
    ``_exec_path_for_launch(_find_llama_server_binary())`` and the backend's
    capability probe probes the same unwrapped path. This module must ask it rather
    than search for itself, because everything it decides -- pipeline groups,
    speculation, the draft depth, and the build the peer is handed -- is a statement
    about the binary that launches. A second search order here is not a duplicate, it
    is a defect.

    It was one. This function used to walk ``spark_cluster``'s bundle layouts, which
    include ``<root>/bin``; the backend's layouts (``utils.llama_cpp_path_settings``)
    do not. With ``UNSLOTH_LLAMA_CPP_PATH`` pointing at a ``<root>/bin`` build the
    orchestrator probed the fork under ``bin/`` and provisioned the peer from beside
    it, while the backend launched ``~/.unsloth/llama.cpp/llama-server``. The two ends
    of the RPC link then ran different builds and the preflight refused the split with
    a malformed HELLO; every capability verdict was about a binary that never ran.

    None when the backend cannot be imported or finds nothing. There is deliberately
    no fallback to another layout: falling back is how the two came apart.
    """
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
    """``ggml-rpc-server`` for THIS node, taken from beside the llama-server that
    launches whenever it is there, and only then from ``spark_cluster``'s bundle
    search.

    Both ends of the RPC link have to be one build: the peer's copy is looked up from
    the directory of this path (``peer_binary_candidates`` leads with it), so resolving
    the rpc-server out of the managed bundle while the backend launched a different
    llama-server is the same defect as above, seen from the other side. The bundle
    search stays as the fallback for a build tree that ships llama-server alone.
    """
    launched = llama_server_binary()
    if launched:
        directory = Path(launched).parent
        for name in _RPC_SERVER_NAMES:
            candidate = directory / name
            try:
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return str(candidate)
            except OSError:
                continue
    sc = _cluster()
    try:
        found = sc.rpc_server_binary() if sc is not None else None
    except Exception:
        return None
    return str(found) if found else None


# ``llama-server --help`` text keyed by (binary path, mtime): one run per build, and a
# reinstall at the same path (new mtime) probes again. A failed or hung run is cached
# as empty so a broken binary costs one timeout, not one per load.
_HELP_TEXT: Dict[Tuple[str, float], str] = {}


def llama_server_help(binary: Optional[str] = None) -> str:
    """The bundle llama-server's ``--help`` output, empty when the binary is missing,
    cannot run, exits without printing, or hangs past ``HELP_PROBE_TIMEOUT_S``.
    Never raises; the answer is cached per binary path and mtime."""
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
    """Whether the bundle's llama-server accepts ``flag``, from its ``--help`` text.
    False on every failure, so a flag the build may lack is never passed."""
    try:
        text = llama_server_help(binary)
    except Exception:
        return False
    if not text or not flag:
        return False
    return re.search(re.escape(flag) + r"(?![\w-])", text) is not None


# Probe verdicts keyed by (binary path, mtime, the probed arguments). Two probes share this
# cache. The first is ``llama-server <flag> <value> --help``: the fork's --pipeline-groups is
# taken out of argv by tools/server before the common parser runs and is not in the --help
# text, so a flag the text does not name is tried for real -- a build that has it strips it
# and prints the usage (exit 0), every other build stops at "invalid argument" (exit 1). The
# second, ``llama_server_accepts_groups_with_drafter``, answers whether a build runs the flag
# TOGETHER with a drafter; that one CANNOT use --help, because the refusal is a check inside
# load_model, so it runs the server against a model that does not exist and reads the output.
# One run per build and argument list; a failure or a hang is cached as rejected.
_ACCEPTS: Dict[Tuple[str, float, Tuple[str, ...]], bool] = {}


def llama_server_accepts(
    flag: str,
    value: str = "1",
    binary: Optional[str] = None,
    *,
    extra: Sequence[str] = (),
) -> bool:
    """Whether the bundle's llama-server takes ``flag`` (with ``value``, and with ``extra``
    after it) ahead of ``--help`` without rejecting it. For flags a build hides from its
    usage text. Never raises;
    False on every failure, so a flag the build may lack is never passed."""
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
    """A port nothing is listening on right now, for a probe that may bind one."""
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

    This CANNOT be answered from ``--help``. ``--pipeline-groups`` is taken out of argv by
    tools/server before the common parser and never printed, so its mere presence is the
    ``llama_server_accepts`` probe below; but the refusal of the pair is
    ``validate_pipeline_groups`` inside ``server_context::load_model``, which ``--help``
    exits long before. A ``--pipeline-groups N --spec-type draft-mtp --help`` therefore
    exits 0 on the fork that refuses the pair as well as on the fork that runs it, and a
    probe built on it can only ever answer yes.

    What does answer it: ``load_model`` validates the pair BEFORE it reads any weights, so
    the server is run for real against a model path that cannot exist. A build that refuses
    prints "--pipeline-groups > 1 is not supported together with speculative decoding" and
    stops there; a build that takes the pair gets past the check and stops at the missing
    file instead, naming it. Both exit non-zero, so the discriminator is the output, not the
    status. Nothing is loaded, no GPU is touched and no socket is left bound (the check runs
    before the listener); it costs a fraction of a second, once per build.

    Cached with every other probe of this build; False on any doubt, which falls back to
    keeping the speculation and dropping the groups."""
    groups = int(groups or 0)
    if groups <= 1:
        return False
    # A build without the flag at all is rejected at argv parse time, and that probe is
    # cheap and already cached; only ask the expensive question of a build that has it.
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
        # An empty directory as the working directory, so the relative model name above
        # names a file that certainly does not exist and the probe cannot be fooled by
        # something of that name in the caller's cwd.
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
        # Positive evidence, not merely the absence of the refusal: the run has to have
        # reached the model, which is the step straight after the check.
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
    """``<arch>.nextn_predict_layers`` from the GGUF header: the depth of the MTP head the
    file ships, 0 when the key says so, None when the file, the key or a reader is missing.
    Never raises. The backend's streaming header reader (about 30 ms, cached by path and
    mtime) answers when it is importable, so this and the launch agree; else the ``gguf``
    package's reader, which the backend already depends on."""
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
    """True only when the header says the model ships MTP layers; False on any doubt."""
    try:
        return (gguf_nextn_predict_layers(path) or 0) > 0
    except Exception:
        return False


def extra_args_own_speculation(extra_args: Optional[List[str]]) -> Optional[str]:
    """The first pass-through flag that makes speculative decoding the caller's, or None."""
    for arg in extra_args or []:
        name = str(arg).partition("=")[0]
        if name in _SPEC_OWNER_FLAGS or name.startswith(_SPEC_OWNER_PREFIXES):
            return name
    return None


def mtp_plan(
    gguf_path: Optional[str],
    extra_args: Optional[List[str]] = None,
    *,
    speculative_type: Optional[str] = None,
    spec_draft_n_max: Optional[int] = None,
) -> Dict[str, Any]:
    """Whether a Spark load should ask for MTP self speculation at the Spark depth.

    ``mtp`` is ``enabled`` (the header has the head, the bundle's llama-server has
    ``--spec-type``, nothing of the caller's stands in the way), ``user override`` (a
    ``speculative_type`` or ``spec_draft_n_max`` on the request, or a ``--spec-type`` /
    draft flag in the pass-through arguments: left exactly alone), ``disabled by env``
    (``UNSLOTH_SPARK_MTP=0``: the launch asks for ``speculative_type`` off), ``no head``,
    ``server too old`` or ``unknown`` (the GGUF is not on disk before the load, so the
    backend decides alone and ``after_load`` reports what launched). ``request`` holds
    the LoadRequest fields to set, ``reason`` says why in one line. The probe of the
    binary runs only once the header has said there is a head, and the header is read
    from the first shard of a split file.
    """
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
    out.update(
        mtp = "enabled",
        reason = (
            f"{layers} MTP layer(s) in the header; {SPEC_TYPE_FLAG} {MTP_SPEC_TYPE} "
            f"{SPEC_DRAFT_N_MAX_FLAG} {MTP_DRAFT_N_MAX}"
        ),
        request = {"spec_draft_n_max": MTP_DRAFT_N_MAX},
    )
    return out


def caller_speculation_off(
    speculative_type: Optional[str] = None, extra_args: Optional[List[str]] = None
) -> bool:
    """True when what the caller set says no speculative decoding at all: a
    ``speculative_type`` of off/none, or extras whose only speculation flag is
    ``--spec-type none``. Anything else of theirs may launch a drafter."""
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

    The choice is made on the concurrency the load asked for, which is
    ``groups["requested_slots"]``: the pass-through ``--parallel`` / ``-np`` when the caller
    set one, else the slot count the load carries (``pipeline_groups_plan`` rounds the
    launched ``--parallel`` up from it to a multiple of the group count). The fork of
    unslothai/llama.cpp PR #187 (a1dd7c5e8) gives every pipeline group its own speculative
    state, so ``--pipeline-groups N > 1`` and a drafter are no longer either/or. Measured on
    the pair with Qwen3.8-27B (``spark_cluster.GROUPS_X_MTP_DECODE_TOKS``), decode tok/s:

        rows | 1 context + MTP | 2 groups | 2 groups + MTP
           8 |      87.8       |   49.7   |      84.8
          32 |     112.1       |  139.5   |     152.5

    so from ``GROUPS_X_MTP_MIN_ROWS`` (16) rows up the two together win, 1.36x of one context
    with MTP and 1.09x of the groups alone on the repeat means, and below it one context with
    MTP wins by 3 percent, because two groups halve the rows per group. Both sides of the
    crossover carry --kv-unified, which is what the load actually launches with. The outcomes, recorded in ``split_config`` and
    ``split_config_reason`` for the status surface:

    * at or above the crossover on a build that takes both: keep both;
    * below it, or on a build that refuses the combination (the old fork's
      ``validate_pipeline_groups``: one ``common_speculative`` bound to one target context):
      keep the speculation and drop the groups, which is what this always did;
    * nothing to keep speculation for (no MTP head, a server without ``--spec-type``,
      ``UNSLOTH_SPARK_MTP=0``): keep the groups and launch with ``speculative_type`` off, so a
      sidecar drafter the backend would otherwise pick on its own is never in the launch. A
      caller who says off keeps the groups too, and nothing of theirs is touched.

    The caller's own speculation follows the same crossover: PR #187 takes ``--model-draft``
    per group as well. ``--mmproj``, control vectors and ``--sleep-idle-seconds`` are still
    refused with the groups whatever the speculation, and ``pipeline_groups_plan`` drops the
    groups outright for those.
    """
    planned = int(groups.get("pipeline_groups") or 0)
    verdict = mtp.get("mtp")
    callers = verdict == "user override"
    speculating = verdict == "enabled" or (
        callers and not caller_speculation_off(speculative_type, extra_args)
    )
    if planned <= 1:
        groups["split_config"] = SPLIT_CONFIG_SPEC if speculating else SPLIT_CONFIG_PLAIN
        groups["split_config_reason"] = str(
            groups.get("reason") or f"{PIPELINE_GROUPS_FLAG} not added"
        )
        return
    rows = int(groups.get("requested_slots") or groups.get("slots") or 1)
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
    request = mtp.setdefault("request", {})
    if request.get("speculative_type") != "off":
        request["speculative_type"] = "off"
        mtp["reason"] = (
            f"{mtp.get('reason')}; speculation off for {PIPELINE_GROUPS_FLAG} "
            f"{groups['pipeline_groups']}, which this GGUF has no head for and which a "
            f"sidecar drafter loses on from 4 users on this pair"
        )


def launched_spec_flags(argv: List[str]) -> Tuple[Optional[str], Optional[int]]:
    """The last ``--spec-type`` value and ``--spec-draft-n-max`` a llama-server argv
    carries (last wins, as in llama.cpp), or None for each that is absent."""
    spec: Optional[str] = None
    depth: Optional[int] = None
    args = [str(a) for a in argv]
    for index, arg in enumerate(args):
        name, _, inline = arg.partition("=")
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
    """The slot count a pass-through already sets (``-np`` / ``--parallel``, last wins,
    the same rule as llama.cpp and LlamaCppBackend._extra_args_n_parallel)."""
    found: Optional[int] = None
    args = [str(a) for a in (extra_args or [])]
    for index, arg in enumerate(args):
        name, _, inline = arg.partition("=")
        if name not in ("-np", "--parallel"):
            continue
        value = inline if inline else (args[index + 1] if index + 1 < len(args) else "")
        try:
            found = int(value.strip())
        except (TypeError, ValueError):
            continue
    return found


def extra_args_refuse_pipeline_groups(extra_args: Optional[List[str]] = None) -> Optional[str]:
    """The first pass-through flag the server still refuses together with
    ``--pipeline-groups N > 1``, or None. A projector, a control vector and an idle timer
    are one per server and not per group, and PR #187 did not change that."""
    for arg in extra_args or []:
        name = str(arg).partition("=")[0]
        if name in _GROUPS_REFUSED_FLAGS:
            return name
    return None


def _from_hub_repo(model_file: Optional[str]) -> bool:
    """Whether this file came out of a hub cache snapshot, where the load can still fetch a
    companion the directory does not have yet."""
    parts = Path(str(model_file or "")).parts
    return "snapshots" in parts and any(part.startswith("models--") for part in parts)


def projector_blocks_pipeline_groups(
    model_file: Optional[str], *, disable_vision: bool = False
) -> Optional[str]:
    """Why this load cannot have pipeline groups because of a multimodal projector, or None.

    ``--mmproj`` is emitted by ``LlamaCppBackend`` from the model's own repo, AFTER
    ``before_load`` has run, and the backend DOWNLOADS the projector as part of the load when
    the repo has one and the cache does not (measured: Qwen3.6-35B-A3B-GGUF had no
    ``mmproj-F16.gguf`` on disk before the load and llama-server was launched with one). The
    server refuses ``--pipeline-groups N > 1`` together with a projector, and it refuses it
    inside load_model, so the whole load fails rather than losing a flag. A directory scan
    before the load therefore cannot clear a repo: only the load's own Vision switch can.

    So the rule is the safe one. With ``disable_vision`` the backend launches with
    ``--no-mmproj-auto`` and no projector, and the groups may be asked for -- unless the
    projector that is already on disk is audio-only, which the switch does not drop (the same
    rule as ``_load_keeps_a_projector`` on the route). Without it, the groups are dropped,
    naming the projector when one is already there.

    Measured on the pair: Unsloth's Qwen3.8-27B-GGUF and Qwen3.6-35B-A3B-GGUF both ship one,
    so before this check the DEFAULT 27B split failed with
    "--pipeline-groups > 1 is not supported together with multimodal (--mmproj)" after four
    launch attempts, and the 35B failed the same way with NOTHING on disk beforehand.

    A model resolved out of a hub repo is therefore blocked whether or not the projector is
    cached yet; a plain local GGUF with no projector beside it is not, because nothing will
    fetch one for it. Losing the groups costs 1.17x at 32 rows; losing the load costs
    everything. Follow-up: the route resolves ``is_vision`` and the projector path a step
    later, and could hand that verdict to ``before_load`` the way it already hands over the
    inherited extras, which would let a text-only repo keep its groups with vision on."""
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
    """How many pipeline groups a layer-split llama-server should run, and with how
    many slots. Only ever consulted for a layer split.

    ``pipeline_groups`` is 0 with a ``reason`` when the env says so, the value is not
    a number, or the bundle's llama-server has no ``--pipeline-groups`` (the fork's
    flag is not in every prebuilt yet; a build without it launches as today). The
    flag is looked for in the ``--help`` text first and, failing that, tried for real
    (``llama_server_accepts``): the fork takes it out of argv ahead of the common
    parser and does not print it in its usage.
    It is also 0 when the pass-through carries a flag the server refuses with the groups
    (``--mmproj``, a control vector, ``--sleep-idle-seconds``: still refused after
    unslothai/llama.cpp PR #187, which only made a drafter per group work).
    Otherwise it is the env value or ``PIPELINE_GROUPS_DEFAULT`` (2), and ``slots`` is the
    launch's slot count rounded up to a multiple of it, at least one per group, so no
    group is left without a slot, which is what the server means by "--parallel must be a
    multiple of the group count". ``requested_slots`` is the count before rounding.
    """
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
        # The slot count is a LoadRequest field with a range, not free-form argv: rounding
        # UP past the maximum would be refused by the request model, so round down to the
        # last multiple that fits and drop the groups when even one slot each does not.
        slots = (PARALLEL_MAX // groups) * groups
        if slots < groups:
            out["reason"] = (
                f"{PIPELINE_GROUPS_FLAG} not added: {groups} groups do not fit in the "
                f"{PARALLEL_MAX}-slot maximum"
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
    """What the local llama-server needs to use the peer's rpc-server. No pipeline
    flag by default: llama.cpp turns pipelining on by itself once the RPC backend
    advertises async and events, which b10796 does. ``pipeline_groups`` above 1 adds
    the fork's ``--pipeline-groups N``.

    The slot count the groups need is NOT emitted here. ``-np`` / ``--parallel`` is one
    of the flags Studio denies in a pass-through (llama_server_args._DENYLIST_GROUPS:
    the launch owns it through LoadRequest.n_parallel, and a second one would desync the
    slot bookkeeping), so a ``--parallel`` in these extras made the load fail with
    HTTP 400 before llama-server was ever started. The rounded count travels as the
    request's ``n_parallel`` instead; see ``_start_layer_split``."""
    # RPC device FIRST, local CUDA device LAST. llama.cpp assigns the output layer to the last
    # device in the list, so this order keeps the last layers, the output projection and the
    # logits on the local GPU: the wire carries only the hidden state (20 KB per row each way)
    # instead of returning 1 MB per row of F32 logits every decode step, and the sampler reads
    # logits from local memory. Measured on Qwen3.8-27B, 32 concurrent, one context: 99.7 tok/s
    # against 94.9 with CUDA0,RPC0; with two pipeline groups 130.4 against 75.5, because with
    # the output on the peer the logits return and the CPU sampling under the other group's GPU
    # load serialise the groups. Memory per node is about the same either way.
    # --cache-ram 0 on a split: the host-RAM prompt cache saves and loads a whole slot state
    # (323 MiB for the 27B at this context) on every slot handover, most of it living on the
    # peer, on the single task thread. Measured at 32 concurrent: one context 75.9 tok/s and a
    # 33 s median TTFT with the cache on against 99.7 and 7.5 s with it off; with two groups the
    # other group starves (6.9 tok/s). This disables only that save/restore cache; the KV prefix
    # reuse inside a slot is untouched, and the sticky router keeps a conversation on its slot.
    # --tensor-split 0.5,0.5, explicitly, because the DEFAULT is not a half. With no -ts
    # llama.cpp divides the layers by each device's FREE MEMORY at load time
    # (llama-model.cpp, "default split, by free memory"), so the boundary depends on whatever
    # else the two nodes happened to be holding, and it is not reproducible between two loads
    # of the same model. Two earlier benchmark arms of the same 27B landed on different
    # boundaries for exactly this reason, and a load-time probe on 2026-09-06 caught the
    # default putting 34 of the 64 blocks on the peer rather than 32.
    #
    # Measured, 2026-09-06, Qwen3.8-27B UD-Q4_K_XL, two groups, 128 concurrent rows, both
    # nodes pinned at 1700 MHz, peer blocks -> decode tok/s:
    #     27 -> 192.1     30 -> 200.2 / 199.1     33 -> 203.7 / 209.2
    #     34 (the free-memory default) -> 200.9 / 201.5     36 -> 192.5
    # 33 wins in both passes, and it is where the two GPUs' busy fractions come out equal
    # (84 and 87 percent, against 90/66 at 27 blocks and 71/88 at 36). Explicit 0.5,0.5 is
    # worth about 2.6 percent over the wandering default and, more importantly, makes the
    # boundary the same on every load.
    #
    # 0.5,0.5 lands on 33 rather than 32 because llama.cpp indexes the split over
    # n_layer + 1 assignment slots, the last of which is the output block: half of 66 is 33,
    # so blocks 0..32 go to the first device. The output block stays on the LAST device
    # either way, which is why the device order above is RPC first.
    out = ["--rpc", f"{peer}:{port}", "--device", "RPC0,CUDA0", "-sm", "layer",
           "--tensor-split", SPLIT_TENSOR_SPLIT_EVEN, "--cache-ram", "0"]
    if pipeline_groups and int(pipeline_groups) > 1:
        out += [PIPELINE_GROUPS_FLAG, str(int(pipeline_groups))]
    return out


def _die_with_parent() -> None:  # pragma: no cover - runs in the forked child
    """``PR_SET_PDEATHSIG(SIGKILL)``, so the ssh client cannot outlive this process.

    Linux only, and best effort: on anything else, and on any failure, the child is
    simply left as it was. Runs between fork and exec, so it must not raise and must
    not allocate anything interesting.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno = True).prctl(1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG, SIGKILL
    except Exception:
        pass


# How often the peer-side reaper looks at the ssh session that started it. A poll of
# ``kill -0`` and ``sleep``: neither can block, which is why this is a poll and not a
# read on the ssh channel (a half-open TCP connection never delivers the EOF).
PEER_REAP_POLL_S = 5


class PeerProcess:
    """A long-lived process on the peer, driven through one ssh session.

    The remote command prints the server's pid, so teardown can ``kill`` exactly that
    pid rather than matching a name on a machine that may be serving something else.

    It also reaps itself. A Studio that dies without running its shutdown path used to
    leave the peer's ggml-rpc-server running and holding the peer's GPU, with nothing
    on either machine to reap it; the next Studio did not notice it either, since the
    preflight only sees a stale server when it speaks a different protocol. Two links
    now make the peer process die with the Studio that started it:

    * locally, the ssh client is given ``PR_SET_PDEATHSIG`` so it cannot outlive this
      process. Without it the client is simply reparented to init on a SIGKILL and
      keeps the remote session open, which is exactly how the orphan survived.
    * on the peer, the launched shell watches the sshd session it was started by and
      kills the ONE pid it started when that session goes away.

    Both are polls of ``kill -0`` or a signal disposition; neither waits on anything
    that can wedge, so the reaper cannot itself become the stuck process. And the
    remote half kills nothing but its own child, so another Studio's rpc-server, or a
    hand-run experiment on the same peer, is never touched.
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

        ``$PPID`` here is the sshd session serving this command. When this Studio goes,
        its ssh client goes with it (PR_SET_PDEATHSIG below), that sshd session exits,
        and the loop kills ``$srv`` -- the pid this shell started and no other. If the
        server exits first the loop ends and the shell waits for it, so a normal exit
        still reports a normal status through the channel.
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
        """For logs and status: the launch with the --api-key value hidden, the same
        rule as ``LlamaCppBackend._redacted_cmd_for_log``."""
        return " ".join(redacted_argv(self.argv))

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self) -> None:
        # A relaunch must not inherit the previous run's pid: a session that dies
        # before printing its own would otherwise have stop() kill whatever the
        # peer has since reused that number for.
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
        """Kill the remote process by pid (only a pid this run printed), then the ssh
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


async def wait_for_port(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _r, w = await asyncio.wait_for(asyncio.open_connection(host, port), timeout = 2.0)
            w.close()
            return True
        except (OSError, asyncio.TimeoutError):
            await asyncio.sleep(0.25)
    return False


def _log_dir() -> Optional[Path]:
    try:
        from utils.paths.storage_roots import studio_root
        return studio_root() / "logs" / "spark"
    except Exception:
        return None


# ── Orchestrator state ───────────────────────────────────────────────────────


class SparkServing:
    """Per-process state: the active topology, the router, and the peer process."""

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
        self.relaunch_log: List[Dict[str, Any]] = []
        self.peer_model_present: Optional[bool] = None
        self.pipeline_groups: int = 0
        self.pipeline_groups_reason: Optional[str] = None
        # Which of the three layer-split configurations the last split asked for, and why
        # (reconcile_split_speculation): both, one context with speculation, or the groups
        # alone. None until a split is planned.
        self.split_config: Optional[str] = None
        self.split_config_reason: Optional[str] = None
        # MTP self speculation for the most recent load on this node: mtp_plan's verdict
        # before the launch, then what the launched argv actually carries.
        self.mtp: str = "unknown"
        self.mtp_reason: Optional[str] = "no load yet"
        self._supervisor: Optional[asyncio.Task] = None
        self._relaunch_task: Optional[asyncio.Task] = None
        self._lock: Optional[asyncio.Lock] = None
        self.last_error: str = ""

    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ── Decision ─────────────────────────────────────────────────────────

    def decide(
        self, *, model_bytes: Optional[float], users: int, kv_bytes_per_user: Optional[float]
    ) -> Dict[str, Any]:
        forced = forced_topology()
        plan = plan_topology(model_bytes, users = users, kv_bytes_per_user = kv_bytes_per_user or 0.0)
        if forced and forced != plan.get("topology"):
            plan["recommended"] = plan.get("topology")
            plan["topology"] = forced
            plan["reason"] = (
                f"forced by {ENV_TOPOLOGY}={forced}; the planner said {plan['recommended']}: "
                f"{plan.get('reason', '')}"
            )
        return plan

    # ── Hooks called by the load path ───────────────────────────────────

    @staticmethod
    def _request_with(request: Any, updates: Dict[str, Any]) -> Any:
        """``request`` with ``updates`` applied: a copy for a pydantic model, in place
        for anything else. Unchanged (same object) when there is nothing to apply."""
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
        """Decide whether this load must be a layer split, and if so start the peer's
        rpc-server and return a request whose extra args point llama-server at it.
        In every topology, ask for MTP self speculation at the Spark depth when the GGUF
        has the head and nothing of the caller's says otherwise (``mtp_plan``).
        Returns the request unchanged on every other path, including any failure.

        ``inherited_extra_args`` are the previous same-model load's pass-through extras
        that the backend carries into a request which omits the field; a ``--spec-type``
        the caller owns there counts as theirs, not as room for this module to set.
        """
        if not enabled():
            return request
        try:
            # Nothing is torn down here. The load may turn out to be a no-op (the
            # model is already resident and _load_model_impl skips the reload), and
            # the running llama-server still depends on whatever peer process serves
            # it. after_load reconciles against what actually launched.
            model_path = str(getattr(request, "model_path", "") or "")
            variant = getattr(request, "gguf_variant", None)
            local_file = cached_repo_file(model_path, variant)
            size = gguf_size_bytes(local_file)
            # LoadRequest.max_seq_length: 0 means "let the backend size it", in which
            # case no KV is charged before the load and after_load re-plans with the
            # context that was actually allocated.
            requested_ctx = int(getattr(request, "max_seq_length", None) or 0)
            cache_type = getattr(request, "cache_type_kv", None)
            kv_total = None
            if local_file and requested_ctx:
                kv_total = await asyncio.to_thread(
                    estimate_kv_bytes, local_file, requested_ctx, cache_type
                )
            users = max(1, int(n_parallel))
            kv_per_user = (kv_total / users) if kv_total else 0.0
            plan = self.decide(model_bytes = size, users = users, kv_bytes_per_user = kv_per_user)
            # The header read and the --help probe are file and process work: off the loop.
            extra = getattr(request, "llama_extra_args", None)
            mtp = await asyncio.to_thread(
                mtp_plan,
                local_file,
                list(extra if extra is not None else (inherited_extra_args or [])),
                speculative_type = getattr(request, "speculative_type", None),
                spec_draft_n_max = getattr(request, "spec_draft_n_max", None),
            )
            if plan.get("topology") != "layer_split":
                # single or replicas: both are decided again after the load, when the
                # resolved file is known. Nothing else to do before it.
                self.plan = plan
                out = request
            else:
                peer = peer_address()
                if peer:
                    out = await self._start_layer_split(
                        request, peer, plan, local_file, users, mtp = mtp
                    )
                else:
                    out = request
            # After the topology step: a split may detach a previous replica first, and
            # the verdict must survive that.
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
    ) -> Any:
        sc = _cluster()
        port = int(getattr(sc, "RPC_DEFAULT_PORT", RPC_PORT_DEFAULT))
        # The --help probe forks the bundle's llama-server (once per build; cached), so
        # it runs off the loop like every other blocking step here.
        groups = await asyncio.to_thread(
            pipeline_groups_plan,
            slots,
            list(getattr(request, "llama_extra_args", None) or []),
            projector = projector_blocks_pipeline_groups(
                local_file,
                disable_vision = bool(getattr(request, "disable_vision", False)),
            ),
        )
        if mtp is not None:
            reconcile_split_speculation(
                groups,
                mtp,
                speculative_type = getattr(request, "speculative_type", None),
                extra_args = list(getattr(request, "llama_extra_args", None) or []),
            )

        def _with_rpc_args(req: Any) -> Any:
            extra = list(getattr(req, "llama_extra_args", None) or [])
            extra += layer_split_extra_args(
                peer, port, pipeline_groups = int(groups["pipeline_groups"])
            )
            updates: Dict[str, Any] = {"llama_extra_args": extra}
            if int(groups["pipeline_groups"]) > 1:
                # The server refuses --pipeline-groups N unless --parallel is a positive
                # multiple of N, and the slot count is the request's field, not argv: the
                # route resolved it into ``slots`` and the launch reads it back from here.
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
            # The peer's ggml-rpc-server serves ONE client at a time (unslothai/llama.cpp
            # serialises the client connection), and a reload starts the replacement
            # llama-server while the outgoing one still holds that connection: measured on
            # the pair, every forced reload of a split failed with "Failed to connect to
            # <peer>:<port>" and then "invalid device: RPC0", four launch attempts each, and
            # only recovered because the failure detached the peer. A forced reload always
            # starts a new server, so the peer is retired with the old one.
            logger.info(
                "spark serving: forced reload of a layer split; restarting the peer "
                "rpc-server so the new llama-server can take the connection"
            )
            reusable = False
        if reusable and not await wait_for_port(peer, port, PEER_REUSE_TIMEOUT_S):
            # ``alive`` is the ssh session, not the server behind it: the session can
            # outlive the process it carries. Reusing a peer that no longer answers made
            # the reload of a split fail with "Failed to connect to <peer>:<port>" and
            # then "invalid device: RPC0" (measured on the pair: every reload after a
            # successful split), so the port is asked before the launch is told to use it.
            logger.warning(
                "spark serving: the peer rpc-server on %s:%s stopped answering; restarting it",
                peer,
                port,
            )
            reusable = False
        if reusable:
            # The rpc-server is model-agnostic: the one already serving this node's
            # llama-server can serve the next load too, and a no-op reload keeps it.
            self.plan = plan
            self.reason = str(plan.get("reason", ""))
            return _with_rpc_args(request)
        if running is not None:
            # A replica from the previous model: the new one needs a split instead.
            await self.detach()

        # Both bundles must speak the same RPC protocol, and nothing stale may sit on
        # the port. The preflight does ssh and socket work, so it runs in a thread.
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
        # Beside the llama-server the backend launches first, so both ends of the link
        # are one build; spark_cluster's bundle search is the fallback inside this.
        local_rpc = rpc_server_binary()
        rc, out, _err = await ssh_run(
            peer, find_binary_script(peer_binary_candidates(local_rpc, "ggml-rpc-server"))
        )
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
        if not await wait_for_port(peer, port, PEER_START_TIMEOUT_S):
            tail = list(process.tail)[-3:]
            await process.stop()
            return _fall_back(
                f"peer rpc-server did not accept on {peer}:{port} within "
                f"{PEER_START_TIMEOUT_S:.0f}s (last output: {tail})"
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
        """The load raised or came back with nothing resident: stop whatever the
        pre-load step started for it."""
        self.mtp, self.mtp_reason = "unknown", "the load failed; nothing is running"
        if self.peer_process is not None or self.router is not None:
            await self.detach()

    async def after_load(self, llama_backend: Any, n_parallel: int) -> None:
        """Reconcile with what actually launched: attach replicas, keep or drop a
        layer split, or record single. Runs after every load, no-op reloads included."""
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
            if "--rpc" in argv or any(str(a).startswith("--rpc=") for a in argv):
                # The server that launched is a layer split. Its rpc-server is ours if
                # before_load started it; a user-supplied --rpc is recorded, not managed.
                if self.router is not None:
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
                # The split was planned but the launch dropped it (or a reload changed
                # the model): the rpc-server is idle now.
                await self.detach()
            if (
                self.topology == "replicas"
                and self.router is not None
                and self.attached_backend is llama_backend
                and self.attached_port == port
            ):
                # Same server, same port: a no-op reload. Keep the replica and router.
                return
            if self.router is not None or self.peer_process is not None:
                await self.detach()
            gguf_path = getattr(llama_backend, "gguf_path", None)
            size = gguf_size_bytes(gguf_path)
            n_ctx = int(
                getattr(llama_backend, "_effective_context_length", None)
                or getattr(llama_backend, "requested_n_ctx", 0)
                or 0
            )
            slots = max(
                1, int(getattr(llama_backend, "effective_parallel_slots", n_parallel) or n_parallel)
            )
            cache_types = getattr(llama_backend, "_effective_cache_types", None) or ()
            cache_type = cache_types[0] if cache_types else None
            kv_total = (
                await asyncio.to_thread(estimate_kv_bytes, gguf_path, n_ctx, cache_type)
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
        except Exception as exc:
            self.last_error = f"after_load: {exc}"[:300]
            logger.warning(
                "spark serving: post-load step failed, serving on this node only: %s", exc
            )
            await self.detach()

    def _record_launched_mtp(self, argv: List[str]) -> None:
        """What this node's llama-server was actually launched with, over the pre-load
        verdict: the backend may have declined the head (too small a model, a binary
        whose capability probe failed) or the caller's own --spec-type may have won."""
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
        # Every file the launch names (weights, mmproj, drafter, adapters) must exist at
        # the same path on the peer, since the replica is launched with the same argv.
        needed = launch_files(argv, str(gguf_path))
        checks = " && ".join(f"test -f {shlex.quote(p)}" for p in needed)
        rc, out, _ = await ssh_run(peer, f"{checks} && echo YES || echo NO", timeout = 25.0)
        self.peer_model_present = rc == 0 and out.strip().endswith("YES")
        if not self.peer_model_present:
            self.topology, self.reason = (
                "single",
                (
                    f"peer {peer} does not have {gguf_path} (or a sidecar the launch names); "
                    f"copy it over the cluster link (rsync -a <file> {peer}:<same path>) to "
                    f"enable replicas"
                ),
            )
            logger.warning("spark serving: %s", self.reason)
            return
        peer_port = int(port)
        peer_argv = replica_argv(argv, binary = binary, host = peer, port = peer_port)
        log_dir = _log_dir()
        self.peer = peer
        self.peer_process = PeerProcess(
            "llama-server", peer, peer_argv, log_dir / "peer-llama-server.log" if log_dir else None
        )
        await self.peer_process.start()
        router = SparkRouter(on_backend_down = self._on_backend_down)
        router.add_backend("main", "127.0.0.1", int(port), slots, primary = True)
        router.add_backend("peer", peer, peer_port, slots)
        await router.start()
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

    # ── Runtime ──────────────────────────────────────────────────────────

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
        """Follow the primary's lifecycle and the peer's process without blocking anything."""
        try:
            while True:
                await asyncio.sleep(SUPERVISOR_INTERVAL_S)
                backend = self.attached_backend
                if backend is None:
                    return
                if getattr(backend, "_process", None) is None:
                    logger.info(
                        "spark serving: this node's llama-server was unloaded; tearing the peer down"
                    )
                    await self.detach()
                    return
                if self.router is not None:
                    port = getattr(backend, "_port", None)
                    if port and port != self.attached_port and getattr(backend, "_healthy", False):
                        await self.router.set_backend_address("main", "127.0.0.1", int(port))
                        self.attached_port = int(port)
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
            # Not ours to restart: LlamaCppBackend respawns its own child on the next
            # failed request, and the router routes around it meanwhile.
            return
        process = self.peer_process
        if process is not None and process.alive:
            # Health failed but the ssh session lives: the server may be loading. Give
            # the health loop time; only a dead process triggers a relaunch.
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
            # Settle: an immediate exit means the peer refuses (port busy, OOM), retry
            # with the next backoff; a live session means the server is loading.
            await asyncio.sleep(3.0)
            if process.alive:
                logger.info(
                    "spark serving: peer %s relaunched (remote pid %s)",
                    process.name,
                    process.remote_pid,
                )
                return

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
        # pipeline_groups: the --pipeline-groups value this node's llama-server was
        # launched with, or 0 and the reason (no flag in the bundle, disabled by env,
        # or not a layer split at all). split_config: which of the three layer-split
        # configurations was picked and why ("groups + speculation" above the measured
        # crossover on a server that takes both, "one context + speculation" below it or
        # on a server that refuses the pair, "groups, no speculation" when there is no
        # head to speculate with). mtp: "enabled" when this node's llama-server
        # (and so every replica) runs the GGUF's own MTP head, else "no head", "server
        # too old", "user override", "disabled by env", "not launched" or "unknown".
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


# ── Thin module-level entry points used by the rest of the backend ───────────


def route_base_url(llama_backend: Any) -> Optional[str]:
    """Cheap on the hot path: one global read on a non-Spark host."""
    if _STATE is None:
        return None
    return _STATE.route_base_url(llama_backend)


def tag_conversation(payload: Dict[str, Any], thread_id: Optional[str]) -> None:
    if _STATE is None:
        return
    _STATE.tag_conversation(payload, thread_id)


def current_topology() -> Optional[str]:
    """For the status response: None off a paired Spark, else the active topology."""
    if not enabled():
        return None
    return state().topology


async def before_load(
    request: Any,
    n_parallel: int,
    *,
    inherited_extra_args: Optional[List[str]] = None,
) -> Any:
    if not enabled():
        return request
    return await state().before_load(request, n_parallel, inherited_extra_args = inherited_extra_args)


async def after_load(llama_backend: Any, n_parallel: int) -> None:
    if not enabled():
        return
    await state().after_load(llama_backend, n_parallel)


async def load_failed() -> None:
    if _STATE is None or not enabled():
        return
    await _STATE.load_failed()


async def shutdown() -> None:
    if _STATE is None:
        return
    await _STATE.detach()


def status() -> Dict[str, Any]:
    if not enabled():
        return {"enabled": False, "topology": None, "reason": "not a paired DGX Spark"}
    return state().status()
