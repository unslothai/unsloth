# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two-Spark serving orchestrator: picks a topology and runs the peer half of it."""

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
import platform
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
ENV_MTP = "UNSLOTH_SPARK_MTP"

TOPOLOGIES = ("single", "replicas", "layer_split")
RPC_PORT_DEFAULT = 50052
PROMPT_TOKENS_DEFAULT = 512  # the planner's measured table is keyed by prompt length
# Only added when the bundle's llama-server has the flag (unslothai/llama.cpp PR #187).
PIPELINE_GROUPS_DEFAULT = 2
PIPELINE_GROUPS_FLAG = "--pipeline-groups"
# LoadRequest.n_parallel's range, mirrored rather than imported from llama_server_args: this module
# is loaded by the CLI and by tests that never import the backend's request models.
PARALLEL_MIN = 1
PARALLEL_MAX = 64
# Never a second --spec-type: extras that own it switch the backend's whole speculative path off,
# with its memory budget, its sub-3B and MLA gates and its retry without speculation.
MTP_SPEC_TYPE = "draft-mtp"
MTP_DRAFT_N_MAX = 3
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
_SPEC_OWNER_ENV = frozenset(
    {
        "LLAMA_ARG_SPEC_TYPE",
        "LLAMA_ARG_SPEC_DRAFT_MODEL",
        "LLAMA_ARG_SPEC_DRAFT_HF_REPO",
        "LLAMA_ARG_SPEC_DRAFT_N_MAX",
        "LLAMA_ARG_SPEC_DRAFT_N_MIN",
        "LLAMA_ARG_SPEC_DRAFT_P_MIN",
        "LLAMA_ARG_N_GPU_LAYERS_DRAFT",
        "LLAMA_ARG_DEVICE_DRAFT",
        "LLAMA_ARG_CACHE_TYPE_K_DRAFT",
        "LLAMA_ARG_CACHE_TYPE_V_DRAFT",
    }
)
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
# Mirrors of spark_cluster.GROUPS_X_MTP_*: the crossover is interpolated between the measured 8
# and 32 rows, and every cell carries --kv-unified (1.27x on a two-group split alone).
GROUPS_X_MTP_MIN_ROWS = 16  # spark_cluster.GROUPS_X_MTP_CROSSOVER_ROWS
GROUPS_X_MTP_OVER_MTP_ONLY = {8: 0.97, 32: 1.36}  # both over one context with MTP
GROUPS_X_MTP_OVER_GROUPS_ONLY = {8: 1.71, 32: 1.09}  # both over two groups alone
SPLIT_TENSOR_SPLIT_EVEN = "0.5,0.5"
# Slots kept, as a fraction of the slots asked for, below which the groups are not worth the
# concurrency the rounding costs. 1/1.4, the measured split speedup from two groups.
GROUPS_WORTH_SLOTS_RATIO = 1.0 / 1.4
# Capped because the throughput table alone says 128 and p90 TTFT says that is unshippable.
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
RELAUNCH_HEALTHY_TIMEOUT_S = 600.0
RELAUNCH_HEALTH_POLL_S = 2.0
PEER_START_TIMEOUT_S = 20.0  # for the rpc-server port to accept; the model load is separate
# A replica is a llama-server, not an rpc-server: it reads the whole model before it binds.
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
    """``studio.spark_cluster``, by path when the package is not importable (the backend runs with
    ``studio/backend`` as its root)."""
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
    if not _CLUSTER_LOOKED_UP and (
        platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64")
    ):
        return False
    sc = _cluster()
    try:
        return bool(sc is not None and sc.is_dgx_spark())
    except Exception:
        return False


# Rail discovery walks sysfs and forks `ip`, measured at 16.2 ms on a paired Spark, and
# ``enabled()`` is on the status endpoint the UI polls: uncached, every poll of a route declared
# ``async`` spends that on the event loop, stalling whatever generation is streaming beside it.
_PEER_DISCOVERY_TTL_S = 60.0
_peer_discovery_cache: Optional[Tuple[float, Optional[str]]] = None


def reset_peer_discovery_cache() -> None:
    """Forget the discovered peer, so the next call re-walks the rails."""
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
    from core.inference import cluster_serving

    if cluster_serving.configured():
        return True
    if not is_spark():
        return False
    return peer_address() is not None


def forced_topology() -> Optional[str]:
    value = (os.environ.get(ENV_TOPOLOGY) or "").strip().lower().replace("-", "_")
    return value if value in TOPOLOGIES else None


def looks_like_a_gguf_load(
    model_path: str, variant: Optional[str], local_file: Optional[str]
) -> bool:
    """Whether this request resolves to a GGUF, as far as can be told before the loader says."""
    if local_file:
        return True
    if str(variant or "").strip():
        return True
    text = str(model_path or "").strip().lower()
    if text.endswith(".gguf"):
        return True
    # Uncached repo: the sizing already asked the hub, so its answer settles this too.
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
    """``spark_cluster.recommend_topology`` with this module's inputs filled in."""
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
    """Every file a load of ``path`` reads. llama.cpp takes only the first shard on the command line
    and opens the rest itself, so a caller that reasons about one name reasons about the whole set
    (ggml-org/llama.cpp tools/gguf-split/README.md)."""
    if not path:
        return []
    p = Path(path)
    match = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", p.name)
    if not match:
        return [str(p)]
    prefix, _first, count = match.groups()
    return [str(p.with_name(f"{prefix}-{i:05d}-of-{count}.gguf")) for i in range(1, int(count) + 1)]


def gguf_size_bytes(path: Optional[str]) -> Optional[int]:
    """The model's size on disk, or None when it cannot be known from what is here."""
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
    """The GGUF a not-yet-resolved load request will use, or None -- in which case the caller waits for
    the resolved ``gguf_path`` after the load rather than guessing."""
    if not model_path:
        return None
    expanded = osp.expanduser(model_path)
    if osp.isfile(expanded):
        return expanded
    if osp.isdir(expanded):
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
    """The GGUF the loader would open out of a local directory, or None."""
    try:
        from utils.models.model_config import list_local_gguf_variants
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


# Names of files that live beside the weights and are not the weights: the vision projector, the
# importance matrix, and the MTP/draft companions a repo ships for speculative decoding.
_COMPANION_GGUF_MARKERS = ("mmproj", "imatrix")
_COMPANION_GGUF_PARTS = ("mtp", "dspark", "draft")


def _is_companion_gguf(path: str) -> bool:
    name = osp.basename(path).lower()
    if any(marker in name for marker in _COMPANION_GGUF_MARKERS):
        return True
    parts = [part.lower() for part in Path(path).parts[:-1]]
    return any(part in _COMPANION_GGUF_PARTS for part in parts)


def _pick_variant(variants: Any, wanted: str) -> Any:
    """The variant a load naming ``wanted`` resolves to, or the one the LOADER would default to."""
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
    """What a not-yet-downloaded GGUF will weigh, from the hub's own file metadata."""
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
    """The GGUF the backend's own resolver would pick, or None when it cannot be asked."""
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
    configured apart, from the GGUF header, or None when the file or the keys cannot be read."""
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
    """This session's login, which is the peer's too (`provision` mirrors the install as one account)."""
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
    # A cluster node can name its own login ("me@10.0.0.3"); a Spark peer is a bare address.
    argv += [peer if "@" in peer else f"{_ssh_user()}@{peer}", remote]
    return argv


async def ssh_run(
    peer: str,
    remote: str,
    *,
    timeout: float = 20.0,
    stdin: Optional[bytes] = None,
) -> Tuple[int, str, str]:
    """Run one command on the peer without blocking the loop. rc 255 on transport failure."""
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh_argv(peer, remote),
            stdin = asyncio.subprocess.PIPE if stdin is not None else None,
            stdout = asyncio.subprocess.PIPE,
            stderr = asyncio.subprocess.PIPE,
        )
        exchange = proc.communicate(stdin) if stdin is not None else proc.communicate()
        out, err = await asyncio.wait_for(exchange, timeout = timeout)
        return (
            proc.returncode if proc.returncode is not None else 255,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
        )
    except (asyncio.TimeoutError, OSError) as exc:
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
    """``path`` as the peer's shell should expand it."""
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
    """Where ``name`` should be on the peer, most likely first: the local binary's own directory, since
    the pair is provisioned by rsync and layouts match."""
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


_SIDECAR_FLAGS = frozenset(
    {
        "--mmproj",
        "-mm",
        "--model-draft",
        "-md",
        "--spec-draft-model",
        "--lora",
        "--lora-scaled",
        "--control-vector",
        "--control-vector-scaled",
    }
)


# A file operand that is NOT a sidecar weight: one plain path, no comma list and no :SCALE, so it is
# absolutised and preflighted but never taken apart or charged as resident memory.
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
    """Every sidecar weight ``args`` names, resolved the way the launch will resolve it."""
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
    """``(bytes, unknown)`` for the sidecars ``args`` names."""
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
    """``(context, cache_type_k, cache_type_v)`` this load will ACTUALLY run with."""
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


def launch_files_from(args: Sequence[str]) -> List[str]:
    """The sidecar paths an argv names, absolutised."""
    return list(sidecar_files(args))


def env_launch_files(env: Optional[Dict[str, str]]) -> List[str]:
    """The existing local files an environment names through ``_REPLICA_ENV_PATHS``."""
    out: List[str] = []
    for name in sorted(_REPLICA_ENV_PATHS):
        raw = str((env or {}).get(name) or "").strip()
        if not raw:
            continue
        for piece in (
            sidecar_operand_paths(raw) if name != "LLAMA_ARG_CHAT_TEMPLATE_FILE" else [raw]
        ):
            path = piece if osp.isabs(piece) else osp.join(os.getcwd(), piece)
            if osp.isfile(path) and path not in out:
                out.append(path)
    return out


def launch_files(argv: List[str], gguf_path: str) -> List[str]:
    """Every file the launch reads; the replica needs all of them at the same path. argv names only the
    first shard, so expand it: a peer holding just that one passes preflight and then fails the
    load."""
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


# Node-local: the slot KV save path is this node's disk, and llama-server refuses a path that does
# not exist.
_REPLICA_DROPPED_FLAGS = ("--port", "--host", "--slot-save-path")


def absolute_sidecar_operand(value: str, *, cwd: Optional[str] = None) -> str:
    """One sidecar operand with every relative path in it made absolute, forms preserved."""
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
    """The local launch with only the binary, host, port and sidecar paths changed: a replica differing
    in any other flag would answer the same request differently."""
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
            if sep:
                out.append(f"{name}={_absolute_plain(inline)}")
            else:
                out.append(arg)
                pending_template = True
            continue
        out.append(arg)
    out += ["--host", host, "--port", str(port)]
    return out


# llama.cpp's common_arg reads every LLAMA_ARG_* itself, so a setting can reach the primary without
# ever appearing in its argv, and copying only the argv leaves the peer on defaults.
_REPLICA_ENV_PREFIX = "LLAMA_ARG_"
# The two the replica must NOT inherit: it is deliberately given a different endpoint.
_REPLICA_ENV_DENY = frozenset({"LLAMA_ARG_HOST", "LLAMA_ARG_PORT"})
# Normalised the way the backend normalises them before spawning, because llama.cpp compares the raw
# string against ggml_type_name and throws on anything else, whitespace included.
_REPLICA_ENV_LOWERED = frozenset({"LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V"})
# The env twins of _SIDECAR_FLAGS: llama.cpp reads the same file from either route, so a path that
# has to be absolutised in the argv has to be absolutised here too, or the fix only covers whichever
# route this particular caller happened to use.
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
    """The llama.cpp settings the primary takes from its environment rather than its argv."""
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
    """Why the peer's GPU is not free to take, or None."""
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
    """Whether this llama-server is already an RPC split, however it was told to be."""
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
    """The two builds, when they are KNOWN to differ, or None."""
    local = await asyncio.to_thread(local_llama_server_version)
    if not local:
        return None
    rc, out, _err = await ssh_run(peer, f"{shlex.quote(peer_binary)} --version 2>&1", timeout = 25.0)
    remote = parse_llama_server_version(out) if rc == 0 else None
    if not remote or remote == local:
        return None
    return f"this node runs llama-server {local} and {peer} runs {remote}"


# The names this process writes, not "anything under the temp directory".
_GENERATED_NAME_PATTERNS = ("unsloth_chat_template_*.jinja",)
_GENERATED_MAX_BYTES = 8 * 1024 * 1024


def _looks_generated(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in _GENERATED_NAME_PATTERNS)


def generated_launch_files(files: Sequence[str]) -> List[str]:
    """The launch files this process WROTE rather than found: the ones in the temp directory."""
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


# The per-launch llama-server key (llama_cpp._write_direct_stream_key): the replica runs the same
# argv, so its --api-key-file must exist on the peer too, or every replica load falls back to single.
_SECRET_NAME_PATTERNS = ("llama_api_key_*",)


def secret_launch_files(files: Sequence[str]) -> List[str]:
    return [
        str(path)
        for path in files
        if any(fnmatch.fnmatch(osp.basename(str(path)), p) for p in _SECRET_NAME_PATTERNS)
        and osp.isfile(str(path))
    ]


async def replicate_secret_files(peer: str, files: Sequence[str]) -> Optional[str]:
    """Copy each key file to the same path on the peer, 0600, over ssh stdin (never in an argv)."""
    for path in files:
        try:
            payload = Path(path).read_bytes()
        except OSError as exc:
            return f"{path} could not be read ({exc})"
        target = shlex.quote(path)
        parent = shlex.quote(osp.dirname(path))
        rc, _out, err = await ssh_run(
            peer,
            f"umask 077 && mkdir -p {parent} && cat > {target} && chmod 600 {target}",
            timeout = 25.0,
            stdin = payload,
        )
        if rc != 0:
            return f"{path} ({(err or '').strip() or f'ssh exit {rc}'})"
    return None


def _read_keys(files: Sequence[str]) -> Dict[str, bytes]:
    out: Dict[str, bytes] = {}
    for path in files:
        try:
            out[path] = Path(path).read_bytes()
        except OSError:
            continue
    return out


async def replicate_generated_files(peer: str, files: Sequence[str]) -> Optional[str]:
    """Write each file to the peer at the SAME path, or name the first that would not go."""
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


def _argv_value(argv: Sequence[str], flag: str) -> Optional[str]:
    tokens = [str(a) for a in argv]
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1].strip()
    return None


def rpc_server_argv(binary: str, *, bind: str, port: int, cache: bool) -> List[str]:
    """``ggml-rpc-server`` on the peer. `-c` writes every received tensor to the peer's disk, a copy
    of the remote half of the model, so it is only enabled where the user already keeps the GGUF."""
    argv = [binary, "-H", bind, "-p", str(port)]
    if cache:
        argv.append("-c")
    return argv


_RPC_SERVER_STEMS = ("ggml-rpc-server", "rpc-server")


def executable_suffixes() -> Tuple[str, ...]:
    """The suffixes a runnable file can carry on this platform, most likely first."""
    if os.name != "nt":
        return ("",)
    raw = os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
    found = tuple(part.strip().lower() for part in raw.split(";") if part.strip().startswith("."))
    return found or (".exe",)


def rpc_server_names() -> Tuple[str, ...]:
    """Candidate file names for the rpc-server, in the order this platform should try them."""
    return tuple(stem + suffix for stem in _RPC_SERVER_STEMS for suffix in executable_suffixes())


def is_executable_file(path: Any) -> bool:
    """Whether ``path`` is a file this platform would run."""
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
    ``_find_llama_server_binary`` and never searched for here: a second search order is a defect,
    and it was one -- ``spark_cluster``'s layouts include ``<root>/bin`` and the backend's do not,
    so the two ends of the link ran different builds."""
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
    """``ggml-rpc-server`` from beside the llama-server that launches, since the peer's copy is looked
    up from this path's directory."""
    launched = llama_server_binary()
    if launched:
        # os.path, like is_executable_file: this resolution is about names on disk and must not
        # depend on which pathlib flavour os.name selects.
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


# Keyed by (path, mtime) so a reinstall at the same path probes again; a failure is cached, so a
# broken binary costs one timeout and not one per load.
_HELP_TEXT: Dict[Tuple[str, float], str] = {}


def llama_server_help(binary: Optional[str] = None) -> str:
    """The ``--help`` of the llama-server that will LAUNCH, empty on any failure."""
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
    """From the ``--help`` text; False on every failure, so a flag the build may lack is never passed."""
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
    """For a flag a build hides from its usage text: the fork strips ``--pipeline-groups`` from argv
    before the common parser and prints the usage, while every other build stops at "invalid
    argument"."""
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
    """Whether the bundle's llama-server runs ``--pipeline-groups N > 1`` TOGETHER with a drafter: the
    per-group speculative state of unslothai/llama.cpp PR #187 (a1dd7c5e8)."""
    groups = int(groups or 0)
    if groups <= 1:
        return False
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
    """``<arch>.nextn_predict_layers`` from the GGUF header, None on anything missing."""
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
    """The option name in ``arg`` the way llama-server itself reads it."""
    name = str(arg).strip().partition("=")[0]
    return name.replace("_", "-") if name.startswith("--") else name


def _speculation_owner_reason(owner: str) -> str:
    """Where the caller's speculation setting came from."""
    if owner in _SPEC_OWNER_ENV:
        return f"{owner} in the environment"
    return f"{owner} in the pass-through arguments"


def extra_args_own_speculation(
    extra_args: Optional[List[str]], env: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """The first setting that makes speculative decoding the caller's, or None."""
    for arg in extra_args or []:
        name = _arg_name(arg)
        if name in _SPEC_OWNER_FLAGS or name.startswith(_SPEC_OWNER_PREFIXES):
            return name
    source = os.environ if env is None else env
    for name in sorted(_SPEC_OWNER_ENV):
        if str(source.get(name) or "").strip():
            return name
    return None


def mtp_draft_n_max(users: Optional[int] = None) -> int:
    """The depth measured best at this many rows, and ``MTP_DRAFT_N_MAX`` below the lowest measured row
    count or when ``users`` is unknown."""
    if users is None:
        return MTP_DRAFT_N_MAX
    rows = max(1, int(users or 1))
    depth = MTP_DRAFT_N_MAX
    for key in sorted(MTP_DRAFT_N_MAX_BY_ROWS):
        if rows >= key:
            depth = MTP_DRAFT_N_MAX_BY_ROWS[key]
    return depth


def split_mtp_wins(users: Optional[int] = None) -> bool:
    """Whether a LAYER SPLIT at this many rows should run a drafter at all; ``users`` of None keeps it."""
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
    """Whether a Spark load should ask for MTP self speculation, and at what depth."""
    out: Dict[str, Any] = {"mtp": "unknown", "reason": None, "request": {}}
    owner = extra_args_own_speculation(extra_args)
    if owner:
        out.update(mtp = "user override", reason = _speculation_owner_reason(owner))
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
    """True only when what the caller set says no speculative decoding at all; anything else of theirs
    may launch a drafter."""
    mode = str(speculative_type or "").strip().lower()
    if mode in ("off", "none", "disable", "disabled"):
        return True
    if mode:
        return False
    owner = extra_args_own_speculation(extra_args)
    if owner is None:
        return False
    if owner in _SPEC_OWNER_ENV:
        # Set through the environment only: its LLAMA_ARG_SPEC_TYPE is the value, and any other
        # draft knob there may still launch a drafter.
        env_set = {name for name in _SPEC_OWNER_ENV if str(os.environ.get(name) or "").strip()}
        env_type = str(os.environ.get("LLAMA_ARG_SPEC_TYPE") or "").strip().lower()
        return env_set == {"LLAMA_ARG_SPEC_TYPE"} and env_type in (
            "off",
            "none",
            "disable",
            "disabled",
        )
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
    """Which of the three layer-split configurations to launch, resolved in place before the launch so
    the server never refuses a start."""
    planned = int(groups.get("pipeline_groups") or 0)
    verdict = mtp.get("mtp")
    callers = verdict == "user override"
    speculating = verdict == "enabled" or (
        callers and not caller_speculation_off(speculative_type, extra_args)
    )
    rows = int(groups.get("requested_slots") or groups.get("slots") or 1)
    # "unknown" counts here as well as "enabled".
    if verdict in ("enabled", "unknown") and not split_mtp_wins(rows):
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
        return
    if (
        verdict == "unknown"
        and split_mtp_wins(rows)
        and llama_server_accepts_groups_with_drafter(planned)
    ):
        # "unknown" is not "no head": the GGUF is not on disk yet, so mtp_plan deliberately declines
        # to answer and the backend reads the header itself after the download.
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
    """The last ``--spec-type`` and ``--spec-draft-n-max`` an argv carries; last wins, as in llama.cpp."""
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


# The env twins of the flags the groups are refused with. llama.cpp's common_arg reads these itself,
# so a projector or control vector can reach the server without ever appearing in argv, and a check
# that only reads argv sees a clean launch and enables the groups anyway.
_GROUPS_REFUSED_ENV = frozenset({"LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"})


def extra_args_refuse_pipeline_groups(
    extra_args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """The first pass-through flag or environment setting the server still refuses together with the
    groups."""
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


def _snapshot_projector(model_file: Optional[str]) -> Optional[str]:
    """The multimodal projector Studio would pick up beside ``model_file``, or None."""
    if not model_file:
        return None
    try:
        # NOT realpath: an HF cache snapshot is a directory of symlinks into blobs/, where the
        # neighbours are content hashes and nothing ends in .gguf.
        directory = osp.dirname(osp.abspath(str(model_file)))
        found = sorted(
            path
            for path in glob.glob(osp.join(directory, "*.gguf"))
            if "mmproj" in osp.basename(path).lower() and not osp.basename(path).startswith("._")
        )
    except OSError:
        found = []
    if not found:
        return None
    return next((f for f in found if f.lower().endswith("-f16.gguf")), found[0])


def _projector_accepts_image(path: str) -> bool:
    try:
        from utils.models.gguf_metadata import mmproj_accepts_image  # type: ignore
        return bool(mmproj_accepts_image(path))
    except Exception as exc:
        logger.info("spark serving: could not read the projector %s: %s", path, exc)
        return False


def auto_projector_file(
    model_file: Optional[str], *, disable_vision: bool = False
) -> Optional[str]:
    """The projector the load will open without being named in the args, for sizing."""
    on_disk = _snapshot_projector(model_file)
    if on_disk and disable_vision and _projector_accepts_image(on_disk):
        return None
    return on_disk


def projector_blocks_pipeline_groups(
    model_file: Optional[str], *, disable_vision: bool = False
) -> Optional[str]:
    """Why this load cannot have pipeline groups because of a multimodal projector, or None."""
    on_disk = _snapshot_projector(model_file)
    if not disable_vision:
        if on_disk:
            return (
                f"Studio opens this model's multimodal projector ({osp.basename(on_disk)}); "
                f"load with vision off to get {PIPELINE_GROUPS_FLAG}"
            )
        # No file yet is a first load of an uncached repo, which is a Hub repo by definition.
        if not model_file or _from_hub_repo(model_file):
            return (
                "Studio fetches this repo's multimodal projector during the load when it has "
                f"one, so it cannot be ruled out beforehand; load with vision off to get "
                f"{PIPELINE_GROUPS_FLAG}"
            )
        return None
    if not on_disk:
        return None
    if _projector_accepts_image(on_disk):
        return None  # the Vision switch really does suppress this one
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
    """How many pipeline groups a layer split should run, and with how many slots: ``slots`` is rounded
    UP to a multiple of the group count and ``requested_slots`` is the count before rounding."""
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
            # _start_layer_split writes this straight into request.n_parallel, so rounding down to
            # fit the cap silently serves fewer concurrent users than were asked for.
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
    """What the local llama-server needs to use the peer's rpc-server."""
    # RPC device FIRST, local CUDA LAST: llama.cpp puts the output layer on the last device, so this
    # keeps the logits local and the wire carries only the hidden state.
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


def strip_layer_split_extra_args(extra: Sequence[str]) -> List[str]:
    """``extra`` without the block ``layer_split_extra_args`` appended to it, so a reload that
    inherits a split's extras is planned afresh instead of reading our own --rpc as the caller's."""
    from core.inference import cluster_serving

    # An N-node cluster's block is recognised by what that module recorded; empty off a cluster.
    tokens = cluster_serving.strip_managed_block([str(t) for t in extra])
    fixed = ["--device", "RPC0,CUDA0", "-sm", "layer", "--tensor-split", SPLIT_TENSOR_SPLIT_EVEN]
    fixed += ["--cache-ram", "0"]
    for start in range(len(tokens) - 1, -1, -1):
        if tokens[start] != "--rpc":
            continue
        end = start + 2 + len(fixed)
        if tokens[start + 2 : end] != fixed:
            return tokens
        tail = tokens[end:]
        if not tail or (len(tail) == 2 and tail[0] == PIPELINE_GROUPS_FLAG):
            return tokens[:start]
        return tokens
    return tokens


try:
    import ctypes as _ctypes
    _LIBC = _ctypes.CDLL("libc.so.6", use_errno = True)
except Exception:
    _LIBC = None


def _die_with_parent() -> None:  # pragma: no cover - runs in the forked child
    """``PR_SET_PDEATHSIG(SIGKILL)``, so the ssh client cannot outlive this process."""
    if _LIBC is None:
        return
    try:
        _LIBC.prctl(1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG, SIGKILL
    except Exception:
        pass


# A poll and not a read on the ssh channel: a half-open TCP connection never delivers EOF.
PEER_REAP_POLL_S = 5


class PeerProcess:
    """A long-lived process on the peer, driven through one ssh session."""

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
        """Start the server, report ITS pid, then watch the ssh session that started us."""
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
        # A relaunch must not inherit the previous pid: stop() would kill whatever the peer has
        # since reused that number for.
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
        """Kill the remote process by pid -- only a pid this run printed -- then the ssh session
        carrying it."""
        # Only while the ssh session that printed the pid is still up.
        if self.remote_pid and self.alive:
            await ssh_run(
                self.peer,
                f"kill {self.remote_pid} 2>/dev/null; sleep 1; kill -9 {self.remote_pid} 2>/dev/null; true",
                timeout = timeout,
            )
        elif self.remote_pid:
            logger.info(
                "spark serving: peer %s already exited; leaving remote pid %s to the remote "
                "wrapper rather than signalling a number we no longer own",
                self.name,
                self.remote_pid,
            )
            self.remote_pid = None
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


PEER_OWNERSHIP_SETTLE_S = 1.5


async def wait_for_own_port(
    process: "PeerProcess",
    host: str,
    port: int,
    timeout: float,
    *,
    cancelled: Optional[Callable[[], bool]] = None,
) -> bool:
    """Wait until ``process`` is answering on ``port``, not merely until something is."""
    if not process.alive:
        return False
    if not await wait_for_port(host, port, timeout, cancelled = cancelled):
        return False
    if PEER_OWNERSHIP_SETTLE_S > 0:
        await asyncio.sleep(PEER_OWNERSHIP_SETTLE_S)
    if not process.alive:
        return False
    # process.alive is the LOCAL ssh session, not a timely answer about the remote child.
    if not await peer_pid_alive(process):
        return False
    # Alive is not bound: a child still starting while a stale server holds the port looks the same.
    deadline = time.monotonic() + timeout
    while True:
        owner = await peer_port_owner(process, port)
        if owner is None or owner:
            return True  # None: the peer cannot say (no ss), so alive is the best evidence
        if time.monotonic() >= deadline or (cancelled is not None and cancelled()):
            return False
        if not process.alive or not await peer_pid_alive(process):
            return False
        await asyncio.sleep(1.0)


async def peer_port_owner(process: "PeerProcess", port: int) -> Optional[bool]:
    """Whether the listener on ``port`` is ``process``'s remote pid; None when it cannot be told."""
    pid = process.remote_pid
    if not pid:
        return None
    try:
        rc, out, _err = await ssh_run(
            process.peer, f"ss -ltnpH 'sport = :{int(port)}' 2>/dev/null", timeout = 10.0
        )
    except Exception:
        return None
    lines = [line for line in (out or "").splitlines() if line.strip()]
    if rc != 0 or not lines:
        return None
    return any(f"pid={int(pid)}," in line for line in lines)


async def peer_pid_alive(process: "PeerProcess", *, timeout: float = 10.0) -> bool:
    """Whether the remote pid this run printed is still running. False when it cannot be asked."""
    pid = process.remote_pid
    if not pid:
        return False
    try:
        rc, out, _err = await ssh_run(
            process.peer,
            f"kill -0 {int(pid)} 2>/dev/null && echo PIDLIVE || echo PIDGONE",
            timeout = timeout,
        )
    except Exception:
        return False
    return rc == 0 and "PIDLIVE" in (out or "")


def _log_dir() -> Optional[Path]:
    try:
        from utils.paths.storage_roots import studio_root
        return studio_root() / "logs" / "spark"
    except Exception:
        return None


class SparkLoadDoesNotFit(Exception):
    """No topology on this pair can hold the requested load."""


class _TeardownRefused(Exception):
    """The route refused to let this load tear down the split serving the current model."""


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
        # The llama-server process the replicas mirror: a reload can land on the same port.
        self.attached_process: Any = None
        # The key files the peer replica was launched with and their bytes, and the remote binary
        # and port it runs on, see _sync_replica.
        self.peer_keys: Dict[str, bytes] = {}
        self.replica_binary: Optional[str] = None
        self.replica_port: Optional[int] = None
        # (backend, process) of a split this load tore down before the route accepted it, see
        # load_failed.
        self._retired_split: Optional[tuple] = None
        # The current load is a model kept alongside, which this state does not describe.
        self._passthrough: bool = False
        # The route's refusal checks for replacing the served model, run before the split goes.
        self._confirm_teardown: Any = None
        # Awaited once a load passes the fit check, before anything is started or replaced.
        self._on_plan_accepted: Any = None
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
        # before_load runs ahead of model classification: never start an rpc-server for a Transformers/MLX load.
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
        """Start the peer's rpc-server for a layer split and point llama-server at it, and in every
        topology ask for MTP at the Spark depth."""
        if not enabled():
            return request
        # The route validates pass-through args AFTER this, but starting a peer is model work.
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
        # What the status surface says right now, kept so a REJECTED replacement can put it back.
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
            model_path = str(getattr(request, "model_path", "") or "")
            variant = getattr(request, "gguf_variant", None)
            local_file = cached_repo_file(model_path, variant)
            size = gguf_size_bytes(local_file)
            # A projector, a drafter, a LoRA or a control vector is resident for the whole load, so
            # pricing the base GGUF alone understates the node.
            extras_for_sizing = list(
                getattr(request, "llama_extra_args", None)
                if getattr(request, "llama_extra_args", None) is not None
                else (inherited_extra_args or [])
            )
            sidecars, sidecars_unknown = sidecar_bytes(extras_for_sizing)
            # Studio's own projector pick is resident too, though no arg names it.
            _named = {"--mmproj", "-mm", "--mmproj-url", "-mmu"}
            if not any(str(a).split("=", 1)[0] in _named for a in extras_for_sizing) and not any(
                os.environ.get(k) for k in _GROUPS_REFUSED_ENV
            ):
                projector = auto_projector_file(
                    local_file, disable_vision = bool(getattr(request, "disable_vision", False))
                )
                projector_size = gguf_size_bytes(projector) if projector else None
                if projector_size:
                    sidecars += int(projector_size)
            for path in env_launch_files(dict(os.environ)):
                if path in launch_files_from(extras_for_sizing):
                    continue  # already charged from argv; do not count it twice
                file_size = gguf_size_bytes(path)
                if file_size is None:
                    try:
                        file_size = osp.getsize(path)
                    except OSError:
                        file_size = None
                if file_size is None:
                    sidecars_unknown = True
                else:
                    sidecars += int(file_size)
            if size is not None:
                size += sidecars
            if sidecars_unknown:
                logger.info(
                    "spark serving: a sidecar could not be sized; the plan charges "
                    "%.1f GiB of sidecars it could see",
                    sidecars / _GIB,
                )
            remote_size = None
            if size is None:
                remote_size = await asyncio.to_thread(
                    remote_gguf_size_bytes,
                    model_path,
                    variant,
                    getattr(request, "hf_token", None),
                )
                size = None if remote_size is None else remote_size + sidecars
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
            # The planner does not only rank topologies, it can say that NONE of them holds this
            # load: model plus KV over the two nodes' budget together.
            if plan.get("fits_any_topology") is False:
                why = str(plan.get("reason", "")) or "no topology holds this load"
                logger.warning("spark serving: refusing the load: %s", why)
                # Nothing started or replaced: leave the module exactly as found.
                self.load_in_progress = False
                self._restore_pre_load_state()
                raise SparkLoadDoesNotFit(why)
            hook, self._on_plan_accepted = self._on_plan_accepted, None
            if hook is not None:
                try:
                    await hook()
                except BaseException as exc:
                    self.load_in_progress = False
                    self._restore_pre_load_state()
                    raise _TeardownRefused() from exc
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
        except (SparkLoadDoesNotFit, _TeardownRefused):
            raise
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
            """What llama-server will actually be launched with, which on a settings-Apply reload is
            not the request field."""
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
            logger.info(
                "spark serving: forced reload of a layer split; restarting the peer "
                "rpc-server so the new llama-server can take the connection"
            )
            reusable = False
        if reusable and not await wait_for_port(peer, port, PEER_REUSE_TIMEOUT_S):
            logger.warning(
                "spark serving: the peer rpc-server on %s:%s stopped answering; restarting it",
                peer,
                port,
            )
            reusable = False
        # Properties of the REQUEST, not the peer: decided before the reuse shortcut.
        if getattr(request, "gpu_ids", None):
            # The backend strips --device pass-through when gpu_ids is set.
            return _fall_back(
                "a layer split needs its own device order and an explicit GPU selection "
                "replaces it; clear the GPU pin to serve this model across both Sparks"
            )
        if argv_or_env_rpc(_effective(request)):
            return _fall_back(
                "llama-server is being launched with a caller-supplied --rpc; leaving the "
                "placement alone"
            )
        if reusable:
            self.plan = plan
            self.reason = str(plan.get("reason", ""))
            return _with_rpc_args(request)
        busy = await peer_gpu_conflict(
            peer, own_pids = [running.remote_pid] if running is not None else []
        )
        if busy:
            return _fall_back(f"layer split needs the peer GPU but {busy}")
        if running is not None:
            if running.alive and self._confirm_teardown is not None:
                # The model being served computes through this rpc-server: a load the route will
                # refuse (409 on active chats) must be refused before it goes, not after.
                try:
                    self._confirm_teardown()
                except BaseException as exc:
                    self.load_in_progress = False
                    self._restore_pre_load_state()
                    raise _TeardownRefused() from exc
            self._note_retired_split(running)
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
        # Every name this platform would accept, the one the LOCAL bundle actually resolved first.
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
            # Not wait_for_port: an occupied port answers on the first probe while the child that
            # could not bind it exits, and adopting that listener leaves the split attached to a
            # server nothing here manages.
            ready = await wait_for_own_port(
                process, peer, port, PEER_START_TIMEOUT_S, cancelled = self._cancelled
            )
        except BaseException:
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

    def _restore_pre_load_state(self) -> bool:
        """Put the reported topology back the way ``before_load`` found it."""
        state, self._pre_load_state = self._pre_load_state, None
        if state is None:
            return False
        (
            self.topology,
            self.reason,
            self.plan,
            self.pipeline_groups,
            self.pipeline_groups_reason,
            self.split_config,
            self.split_config_reason,
        ) = state
        return True

    def _note_retired_split(self, running: Any) -> None:
        """Remember the primary whose remote layers go with ``running``, for load_failed; only a
        split's primary has any, a replica's primary is a whole model."""
        if getattr(running, "name", "") == "ggml-rpc-server":
            backend = self.attached_backend
            self._retired_split = (backend, getattr(backend, "_process", None))

    async def load_failed(self) -> None:
        self.load_in_progress = False
        retired, self._retired_split = self._retired_split, None
        if retired is not None:
            backend, process = retired
            if process is not None and getattr(backend, "_process", None) is process:
                # Its remote half went with the rpc-server this load replaced, so it is loaded but
                # can no longer compute: unload it rather than report a model that cannot answer.
                logger.warning(
                    "spark serving: the replacement load failed after the layer split was torn "
                    "down; unloading the previous model, whose remote layers are gone"
                )
                try:
                    await asyncio.to_thread(backend.unload_model)
                except Exception:
                    logger.warning(
                        "spark serving: unloading the orphaned model failed", exc_info = True
                    )
        # Not every failed load leaves nothing running: the route validates and can raise 400/409
        # before it unloads, so a rejected replacement leaves the previous model loaded and still
        # being served.
        backend = self.attached_backend
        if backend is not None and getattr(backend, "is_loaded", False):
            self._restore_pre_load_state()
            return
        self.mtp, self.mtp_reason = "unknown", "the load failed; nothing is running"
        if self.peer_process is not None or self.router is not None:
            await self.detach()

    async def after_load(self, llama_backend: Any, n_parallel: int) -> Optional[bool]:
        """Reconcile with what actually launched. Runs after every load, no-op reloads too. True when
        the load was cancelled while the replica was starting, so the caller must undo it."""
        self.load_in_progress = False
        self._pre_load_state = None
        self._retired_split = None
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
                if self.router is not None or (
                    self.peer_process is not None and not self._argv_names_our_peer(argv)
                ):
                    await self.detach()
                self.attached_backend = llama_backend
                self.attached_port = port
                self.attached_process = getattr(llama_backend, "_process", None)
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
                and self.attached_process is getattr(llama_backend, "_process", None)
            ):
                return
            if self.router is not None or self.peer_process is not None:
                await self.detach()
            gguf_path = getattr(llama_backend, "gguf_path", None)
            size = gguf_size_bytes(gguf_path)
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
                await self.detach()
                self.topology, self.reason = "single", "the load was cancelled"
                return True
        except Exception as exc:
            self.last_error = f"after_load: {exc}"[:300]
            logger.warning(
                "spark serving: post-load step failed, serving on this node only: %s", exc
            )
            await self.detach()

    def _record_launched_mtp(self, argv: List[str]) -> None:
        """What launched, over the pre-load verdict: the backend may have declined the head, or the
        caller's own --spec-type may have won."""
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
        # Two round trips on purpose: the lookup script exits as soon as it finds the binary, so a
        # file check appended to it would never run.
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
        for extra in env_launch_files(getattr(llama_backend, "launched_env", None)):
            if extra not in needed:
                needed.append(extra)
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
        secrets_needed = secret_launch_files(needed)
        if secrets_needed:
            failed = await replicate_secret_files(peer, secrets_needed)
            if failed:
                self.topology, self.reason = (
                    "single",
                    f"the llama-server key could not be put on {peer}: {failed}",
                )
                logger.warning("spark serving: %s", self.reason)
                return
        self.peer_keys = _read_keys(secrets_needed)
        needed = [p for p in needed if p not in set(generated) | set(secrets_needed)]
        # Size and mtime, not just existence; -L as os.stat does, since HF snapshot entries are links.
        mismatch: List[str] = []
        if needed:
            fmt = "; ".join(
                f"stat -L -c '%s %Y' {shlex.quote(p)} 2>/dev/null || echo NOSTAT" for p in needed
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
        # The environment the PRIMARY was actually spawned with, not this process's.
        peer_argv = with_replica_env(
            replica_argv(argv, binary = binary, host = peer, port = peer_port),
            replica_env(getattr(llama_backend, "launched_env", None) or None),
        )
        log_dir = _log_dir()
        self.peer = peer
        self.replica_binary, self.replica_port = binary, peer_port
        self.peer_process = PeerProcess(
            "llama-server", peer, peer_argv, log_dir / "peer-llama-server.log" if log_dir else None
        )
        await self.peer_process.start()
        router: Optional[SparkRouter] = None
        try:
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
        self.attached_process = getattr(llama_backend, "_process", None)
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
        if getattr(llama_backend, "_process", None) is not self.attached_process:
            return None  # respawned on the same port, maybe with a new key the replica lacks
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
            await asyncio.sleep(3.0)
            if not process.alive:
                continue
            logger.info(
                "spark serving: peer %s relaunched (remote pid %s)",
                process.name,
                process.remote_pid,
            )
            # The budget bounds one crash loop, not the process lifetime: without this a peer that
            # restarts cleanly three times over days is then never recovered again.
            if not await self._await_peer_healthy(process, RELAUNCH_HEALTHY_TIMEOUT_S):
                self.relaunch_log.append(
                    {"at": time.time(), "event": "relaunched but never healthy"}
                )
                continue
            self.relaunch_attempts = 0
            self.relaunch_log.append({"at": time.time(), "event": "recovered"})
            return

    async def _repoint_primary(self, backend: Any) -> None:
        """Follow this node's llama-server to a new port, once its load has finished."""
        if self.router is None or self.load_in_progress:
            return
        if self.peer_keys and _read_keys(list(self.peer_keys)) != self.peer_keys:
            # A respawn rotated the key before it is healthy again; the replica would answer 401.
            await self.router.hold("peer", "the primary's API key changed")
        port = getattr(backend, "_port", None)
        replaced = port != self.attached_port or (
            getattr(backend, "_process", None) is not self.attached_process
        )
        if port and replaced and getattr(backend, "_healthy", False):
            await self.router.set_backend_address("main", "127.0.0.1", int(port))
            # Committed only once the replica is in step, so a failed key copy is retried next pass.
            if await self._sync_replica(backend):
                self.attached_port = int(port)
                self.attached_process = getattr(backend, "_process", None)

    async def _sync_replica(self, backend: Any) -> bool:
        """Bring the replica in line with a primary that was relaunched without a load request: a
        respawn rewrites the key file with a new key (the replica would answer 401), and the MTP
        crash fallback drops the speculative flags (the replica would keep crashing on them). False
        when it could not be brought in step and should be retried."""
        process, peer = self.peer_process, self.peer
        if process is None or not peer or self.topology != "replicas" or not self.replica_binary:
            return True
        argv = list(getattr(getattr(backend, "_process", None), "args", None) or [])
        rebuilt = process.argv
        if argv:
            rebuilt = with_replica_env(
                replica_argv(
                    argv, binary = self.replica_binary, host = peer, port = int(self.replica_port)
                ),
                replica_env(getattr(backend, "launched_env", None) or None),
            )
        keys = _read_keys(list(self.peer_keys))
        if rebuilt == process.argv and keys == self.peer_keys:
            return True
        # The primary already answers with the new key: a request sent to the old replica gets 401.
        if self.router is not None:
            await self.router.hold("peer", "replica out of step with a relaunched primary")
        if keys != self.peer_keys:
            failed = await replicate_secret_files(peer, list(keys))
            if failed:
                logger.warning(
                    "spark serving: the new llama-server key could not reach %s: %s", peer, failed
                )
                return False
            self.peer_keys = keys
        process.argv = rebuilt
        try:
            await process.stop(timeout = 5.0)
            await process.start()
        except Exception as exc:
            logger.warning("spark serving: peer restart for the new key failed: %s", exc)
            return False
        if self.router is not None:
            self.router.release("peer")
        return True

    def _argv_names_our_peer(self, argv: Sequence[str]) -> bool:
        """Whether an ``--rpc`` in *argv* points at the rpc-server this module launched."""
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
        """Whether the load this orchestration belongs to has been cancelled."""
        event = self._cancel_event
        try:
            return bool(event is not None and event.is_set())
        except Exception:
            return False

    async def _await_peer_healthy(self, process: Any, timeout: float) -> bool:
        """Whether the router sees this peer answering within *timeout*."""
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if self.attached_backend is None or process is not self.peer_process:
                return False  # detached, or this relaunch has been superseded
            if not process.alive:
                return False
            if getattr(process, "name", "") == "ggml-rpc-server":
                # A layer split has no router; its peer is healthy once it owns the RPC port.
                port = _argv_value(getattr(process, "argv", None) or [], "-p")
                if not port or not port.isdigit():
                    return False
                return await wait_for_own_port(
                    process,
                    process.peer,
                    int(port),
                    max(0.0, deadline - time.monotonic()),
                    cancelled = lambda: (
                        self.attached_backend is None or process is not self.peer_process
                    ),
                )
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
            self.attached_process = None
            self.peer_keys = {}
            self.replica_binary = self.replica_port = None
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
    global _STATE, _CLUSTER, _CLUSTER_LOOKED_UP, _DETACH_LEGACY_ON_COMMIT
    _STATE = None
    _DETACH_LEGACY_ON_COMMIT = False
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


# A two-Spark split an explicit-engine load replaces, detached once that load commits.
_DETACH_LEGACY_ON_COMMIT = False


def _cluster_active() -> bool:
    from core.inference import cluster_serving
    return (os.environ.get(ENV_TOGGLE) or "").strip() != "0" and cluster_serving.configured()


def current_topology() -> Optional[str]:
    from core.inference import cluster_serving

    if cluster_serving._STATE is not None and (
        _cluster_active() or cluster_serving._STATE.serving()
    ):
        return cluster_serving._STATE.topology
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
    confirm_teardown: Any = None,
) -> Any:
    from core.inference import cluster_serving

    global _DETACH_LEGACY_ON_COMMIT
    alongside = bool(
        getattr(request, "alongside", False) or getattr(request, "force_alongside", False)
    )
    replacing = not alongside and str(getattr(request, "engine", "auto") or "auto") == "auto"
    if _cluster_active():
        out = await cluster_serving.state().before_load(
            request,
            n_parallel,
            inherited_extra_args = inherited_extra_args,
            cancel_event = cancel_event,
        )
        legacy = _STATE is not None and (
            _STATE.peer_process is not None or _STATE.router is not None
        )
        _DETACH_LEGACY_ON_COMMIT = legacy and not alongside and not replacing
        if replacing and legacy:
            # A two-Spark split or replica set from before the cluster was configured; only once
            # the cluster has a plan, so a refused one leaves it serving.
            try:
                if confirm_teardown is not None:
                    confirm_teardown()
            except BaseException:
                await cluster_serving._STATE.load_failed()  # no load follows a refused teardown
                raise
            await _STATE.detach()
        return out
    if cluster_serving._STATE is not None and (
        cluster_serving._STATE.workers or cluster_serving._STATE._backend is not None
    ):
        # The cluster is off now; its workers and its record of the served split go once a load
        # that replaces the model commits.
        cluster_serving._STATE.retire_on_commit = not (
            getattr(request, "alongside", False) or getattr(request, "force_alongside", False)
        )
        if replacing and cluster_serving._STATE.retire_on_commit and enabled():
            # The two-Spark path may start its own rpc-server on a port a cluster worker holds, so
            # the workers go once its planner accepts the load; a refused one keeps them serving.
            async def retire_cluster() -> None:
                if confirm_teardown is not None:
                    confirm_teardown()
                cluster = cluster_serving._STATE
                if cluster._backend is not None and cluster.block and cluster.workers:
                    # Its remote layers go with the workers: a replacement the route then rejects
                    # unloads it, as for a torn-down two-Spark split.
                    state()._retired_split = (
                        cluster._backend,
                        getattr(cluster._backend, "_process", None),
                    )
                await cluster.shutdown()

            state()._on_plan_accepted = retire_cluster
    if not enabled():
        return request
    st = state()
    # A model kept alongside lands in an extra slot of its own, while this state describes the
    # primary's topology: planning it would detach the split the primary is still serving through.
    # An explicit engine (vLLM, SGLang) is not a llama.cpp launch, so there is no split to plan.
    st._passthrough = (
        bool(getattr(request, "alongside", False))
        or bool(getattr(request, "force_alongside", False))
        or str(getattr(request, "engine", "auto") or "auto") != "auto"
    )
    if st._passthrough:
        return request
    st._cancel_event = cancel_event
    st._confirm_teardown = confirm_teardown
    try:
        return await st.before_load(request, n_parallel, inherited_extra_args = inherited_extra_args)
    except _TeardownRefused as exc:
        raise exc.__cause__
    finally:
        st._confirm_teardown = None
        st._on_plan_accepted = None


async def after_load(
    llama_backend: Any,
    n_parallel: int,
    *,
    cancel_event: Any = None,
) -> bool:
    """True when the load was cancelled during the replica startup and must be undone."""
    from core.inference import cluster_serving

    global _DETACH_LEGACY_ON_COMMIT
    if cluster_serving._STATE is not None and cluster_serving._STATE.in_flight:
        await cluster_serving._STATE.after_load(llama_backend)
        if _DETACH_LEGACY_ON_COMMIT and _STATE is not None:
            await _STATE.detach()
        _DETACH_LEGACY_ON_COMMIT = False
        return False
    if cluster_serving._STATE is not None and cluster_serving._STATE.retire_on_commit:
        await cluster_serving._STATE.shutdown()
    if not enabled():
        return False
    st = state()
    if st._passthrough:
        st._passthrough = False
        return False
    if cancel_event is not None:
        st._cancel_event = cancel_event
    try:
        return bool(await st.after_load(llama_backend, n_parallel))
    finally:
        st._cancel_event = None


async def load_failed() -> None:
    from core.inference import cluster_serving

    global _DETACH_LEGACY_ON_COMMIT
    _DETACH_LEGACY_ON_COMMIT = False
    if cluster_serving._STATE is not None and cluster_serving._STATE.in_flight:
        await cluster_serving._STATE.load_failed()
        legacy = _STATE.attached_backend if _STATE is not None else None
        if legacy is not None and not getattr(legacy, "is_loaded", False):
            await _STATE.detach()  # an engine load unloaded the two-Spark primary, then failed
        return
    if cluster_serving._STATE is not None and cluster_serving._STATE.retire_on_commit:
        cluster_serving._STATE.failed_while_retiring()
    if _STATE is None or not enabled():
        return
    if _STATE._passthrough:
        _STATE._passthrough = False
        return
    await _STATE.load_failed()


async def reconcile_internal_load() -> None:
    """Drop a two-node topology before a load that did not come through ``before_load``."""
    from core.inference import cluster_serving

    if cluster_serving._STATE is not None:
        cluster_serving._STATE.forget_served()
    if _STATE is None or _STATE.attached_backend is None and _STATE.peer_process is None:
        return
    logger.info("spark serving: serving on this node only for a load that bypassed the planner")
    await _STATE.detach()


async def shutdown() -> None:
    from core.inference import cluster_serving

    if cluster_serving._STATE is not None:
        await cluster_serving._STATE.shutdown()
    if _STATE is None:
        return
    await _STATE.detach()


def status() -> Dict[str, Any]:
    from core.inference import cluster_serving

    if _cluster_active():
        return cluster_serving.state().status()
    if cluster_serving._STATE is not None and cluster_serving._STATE.serving():
        # Off for new loads, but the loaded split still runs on its workers until replaced.
        return dict(cluster_serving._STATE.status(), retiring = True)
    if not enabled():
        return {"enabled": False, "topology": None, "reason": "not a paired DGX Spark"}
    return state().status()
