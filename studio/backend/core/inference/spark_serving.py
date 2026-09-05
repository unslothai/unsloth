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
               CUDA0,RPC0 -sm layer``. Pipeline parallelism is enabled by llama.cpp
               itself when the RPC backend advertises async and events (b10796 does),
               so no flag is added for it. When the bundle's llama-server has
               ``--pipeline-groups`` (the unslothai/llama.cpp fork: N contexts from one
               model, the slots partitioned across them, N interleaved decode loops so
               one group's batch runs on the peer's layers while another's runs here)
               the launch adds ``--pipeline-groups 2`` and an even slot count. Measured
               with Qwen3.8-27B: 1.27x to 1.50x of one Spark at 32 to 128 concurrent
               rows with both GPUs near 80 percent, against 0.85x to 1.01x for the
               one-context split. The flag is probed for with ``llama-server --help``
               once per binary; a bundle without it launches exactly as before.

The decision is ``studio/spark_cluster.recommend_topology`` (pure, measured); this
module only gathers its inputs (weights on disk, KV per slot from the GGUF header, the
slot count the load asked for) and runs the processes the answer needs. Paths come
from ``spark_cluster.llama_bundle_dir`` and ``rpc_server_binary`` so the peer, which
``unsloth spark provision`` mirrors from this node, is launched from the same bundle.

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
import subprocess
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from core.inference.spark_router import CONVERSATION_FIELD, Backend, SparkRouter

logger = logging.getLogger(__name__)

ENV_TOGGLE = "UNSLOTH_SPARK_SERVING"  # "0" disables; anything else leaves detection alone
ENV_TOPOLOGY = "UNSLOTH_SPARK_TOPOLOGY"  # auto | single | replicas | layer_split
ENV_PEER = "UNSLOTH_SPARK_PEER"  # peer address override (tests, unusual cabling)
ENV_RPC_BIND = "UNSLOTH_SPARK_RPC_BIND"  # ggml-rpc-server -H on the peer; default is the peer's cluster address
ENV_PREFILL_HEAVY = (
    "UNSLOTH_SPARK_PREFILL_HEAVY"  # "1": tell the planner the work is long-prompt prefill
)
# Layer split only: 0 disables, N sets it; default 2 when the bundle has the flag.
ENV_PIPELINE_GROUPS = "UNSLOTH_SPARK_PIPELINE_GROUPS"

TOPOLOGIES = ("single", "replicas", "layer_split")
RPC_PORT_DEFAULT = 50052
PROMPT_TOKENS_DEFAULT = 512  # the planner's measured table is keyed by prompt length
PIPELINE_GROUPS_DEFAULT = 2  # two Sparks, two interleaved decode loops (see spark_cluster)
PIPELINE_GROUPS_FLAG = "--pipeline-groups"
HELP_PROBE_TIMEOUT_S = 20.0  # llama-server --help; a hung binary is a missing flag
RELAUNCH_BACKOFF_S = (5.0, 15.0, 45.0)  # bounded: three attempts, then the peer stays down
PEER_START_TIMEOUT_S = 20.0  # for the rpc-server port to accept; the model load is separate
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


# Where a bundle keeps its executables: the Linux/macOS layout, the Windows layout, the
# raw tarball layout, a flat directory. Same order as spark_cluster._BUNDLE_SUBDIRS.
_BUNDLE_SUBDIRS = (("build", "bin"), ("build", "bin", "Release"), ("bin",), ())


def llama_server_binary() -> Optional[str]:
    """The managed bundle's llama-server, or None. Resolved through
    ``spark_cluster.llama_bundle_dir`` so it is the binary the peer was provisioned
    with and the one this node launches on a default install."""
    sc = _cluster()
    try:
        bundle = Path(sc.llama_bundle_dir()) if sc is not None else None
    except Exception:
        return None
    if bundle is None:
        return None
    for parts in _BUNDLE_SUBDIRS:
        candidate = bundle.joinpath(*parts, "llama-server")
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        except OSError:
            continue
    return None


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


def pipeline_groups_plan(slots: int, extra_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """How many pipeline groups a layer-split llama-server should run, and with how
    many slots. Only ever consulted for a layer split.

    ``pipeline_groups`` is 0 with a ``reason`` when the env says so, the value is not
    a number, or the bundle's llama-server has no ``--pipeline-groups`` (the fork's
    flag is not in every prebuilt yet; a build without it launches as today).
    Otherwise it is the env value or ``PIPELINE_GROUPS_DEFAULT``, and ``slots`` is the
    launch's slot count rounded up to a multiple of it, at least one per group, so no
    group is left without a slot. ``requested_slots`` is the count before rounding.
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
    if not llama_server_supports(PIPELINE_GROUPS_FLAG):
        out["reason"] = f"bundle llama-server lacks {PIPELINE_GROUPS_FLAG}"
        return out
    out["pipeline_groups"] = groups
    out["slots"] = max(groups, -(-base // groups) * groups)
    return out


def layer_split_extra_args(
    peer: str,
    port: int,
    *,
    pipeline_groups: int = 0,
    slots: Optional[int] = None,
) -> List[str]:
    """What the local llama-server needs to use the peer's rpc-server. No pipeline
    flag by default: llama.cpp turns pipelining on by itself once the RPC backend
    advertises async and events, which b10796 does. ``pipeline_groups`` above 1 adds
    the fork's ``--pipeline-groups N`` and, with ``slots``, a ``--parallel`` that
    overrides the emitted one (extras come last and llama.cpp is last-wins, and the
    backend reads the override back for its own slot accounting)."""
    out = ["--rpc", f"{peer}:{port}", "--device", "CUDA0,RPC0", "-sm", "layer"]
    if pipeline_groups and int(pipeline_groups) > 1:
        out += [PIPELINE_GROUPS_FLAG, str(int(pipeline_groups))]
        if slots is not None:
            out += ["--parallel", str(max(1, int(slots)))]
    return out


class PeerProcess:
    """A long-lived process on the peer, driven through one ssh session.

    The remote command prints its own pid and ``exec``s the server, so the ssh channel
    is the server's lifetime and teardown can ``kill`` exactly that pid rather than
    matching a name on a machine that may be serving something else.
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
        return "echo UNSLOTH_SPARK_PID=$$; exec " + " ".join(shlex.quote(a) for a in self.argv)

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

    async def before_load(self, request: Any, n_parallel: int) -> Any:
        """Decide whether this load must be a layer split, and if so start the peer's
        rpc-server and return a request whose extra args point llama-server at it.
        Returns the request unchanged on every other path, including any failure."""
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
            if plan.get("topology") != "layer_split":
                # single or replicas: both are decided again after the load, when the
                # resolved file is known. Nothing to do before it.
                self.plan = plan
                return request
            peer = peer_address()
            if not peer:
                return request
            return await self._start_layer_split(request, peer, plan, local_file, users)
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
    ) -> Any:
        sc = _cluster()
        port = int(getattr(sc, "RPC_DEFAULT_PORT", RPC_PORT_DEFAULT))
        # The --help probe forks the bundle's llama-server (once per build; cached), so
        # it runs off the loop like every other blocking step here.
        groups = await asyncio.to_thread(
            pipeline_groups_plan, slots, list(getattr(request, "llama_extra_args", None) or [])
        )

        def _with_rpc_args(req: Any) -> Any:
            extra = list(getattr(req, "llama_extra_args", None) or [])
            extra += layer_split_extra_args(
                peer,
                port,
                pipeline_groups = int(groups["pipeline_groups"]),
                slots = int(groups["slots"]),
            )
            self.pipeline_groups = int(groups["pipeline_groups"])
            self.pipeline_groups_reason = groups.get("reason")
            if self.pipeline_groups:
                logger.info(
                    "spark serving: %s %d with %d slots (%d asked for)",
                    PIPELINE_GROUPS_FLAG,
                    self.pipeline_groups,
                    int(groups["slots"]),
                    int(groups["requested_slots"]),
                )
            else:
                logger.info(
                    "spark serving: no pipeline groups: %s", self.pipeline_groups_reason
                )
            try:
                return req.model_copy(update = {"llama_extra_args": extra})
            except AttributeError:
                req.llama_extra_args = extra
                return req

        def _fall_back(reason: str) -> Any:
            self.topology, self.reason = "single", reason
            self.plan = dict(plan, topology = "single", reason = reason)
            self.pipeline_groups, self.pipeline_groups_reason = 0, None
            logger.warning("spark serving: %s", reason)
            return request

        running = self.peer_process
        if (
            self.topology == "layer_split"
            and running is not None
            and running.name == "ggml-rpc-server"
            and running.peer == peer
            and running.alive
        ):
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
        local_rpc = None
        try:
            local_rpc = sc.rpc_server_binary() if sc is not None else None
        except Exception:
            local_rpc = None
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
            if self.topology != "single":
                self.topology = "single"
                self.reason = "detached"

    def status(self) -> Dict[str, Any]:
        # pipeline_groups: the --pipeline-groups value this node's llama-server was
        # launched with, or 0 and the reason (no flag in the bundle, disabled by env,
        # or not a layer split at all).
        if self.topology == "layer_split":
            groups, groups_reason = self.pipeline_groups, self.pipeline_groups_reason
        else:
            groups, groups_reason = 0, f"not a layer split (topology {self.topology})"
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


async def before_load(request: Any, n_parallel: int) -> Any:
    if not enabled():
        return request
    return await state().before_load(request, n_parallel)


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
