# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer-split serving over ggml RPC: the load hooks Studio calls, and the helpers they share."""

from __future__ import annotations

import asyncio
import collections
import getpass
import glob
import logging
import os
import os.path as osp
import re
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ENV_TOGGLE = "UNSLOTH_SPARK_SERVING"  # "0" disables layer-split serving

PIPELINE_GROUPS_FLAG = "--pipeline-groups"
# Still refused with groups after PR #187: projector, control vectors and idle timer are per server.
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
SPLIT_TENSOR_SPLIT_EVEN = "0.5,0.5"
HELP_PROBE_TIMEOUT_S = 20.0  # llama-server --help; a hung binary is a missing flag
_LOG_TAIL = 60


def gguf_shard_paths(path: Optional[str]) -> List[str]:
    """Every file a load of ``path`` reads; llama.cpp opens the shards after the first itself."""
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
    """The GGUF an unresolved load request will use, or None (the caller then waits for the resolved path)."""
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


def _loader_kv_bytes(
    gguf_path: str, n_ctx: int, cache_type: Optional[str], cache_type_v: Optional[str]
) -> Optional[int]:
    """``LlamaCppBackend._estimate_kv_cache_bytes`` on a probe instance; None when it cannot say."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend

        probe = object.__new__(LlamaCppBackend)
        probe._model_identifier = "spark-kv-probe"
        probe._read_gguf_metadata(str(_first_shard(gguf_path) or gguf_path))
        k_type = str(cache_type or "f16").strip().lower()
        v_type = k_type if cache_type_v is None else str(cache_type_v).strip().lower()
        k_bytes = int(probe._estimate_kv_cache_bytes(int(n_ctx), k_type))
        if k_bytes <= 0:
            return None
        if v_type == k_type:
            return k_bytes
        v_bytes = int(probe._estimate_kv_cache_bytes(int(n_ctx), v_type))
        if v_bytes <= 0:
            return None
        # The estimator takes one type for both; K and V each carry half of either pricing.
        return int((k_bytes + v_bytes) / 2)
    except Exception:
        return None


def estimate_kv_bytes(
    gguf_path: Optional[str],
    n_ctx: int,
    cache_type: Optional[str] = None,
    cache_type_v: Optional[str] = None,
) -> Optional[int]:
    """KV cache bytes for ``n_ctx`` tokens, priced by the loader's own estimator; None when unreadable."""
    if not gguf_path or n_ctx <= 0:
        return None
    loader = _loader_kv_bytes(gguf_path, int(n_ctx), cache_type, cache_type_v)
    if loader is not None:
        return loader
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
    """This session's login; a node that logs in as someone else names it (``me@10.0.0.3``)."""
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
    # A cluster node can name its own login ("me@10.0.0.3"); a bare address uses this one.
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
    """``path`` as a node with the same layout sees it: under its own ``$HOME`` when ours is."""
    try:
        return "$HOME/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return path.as_posix()


def peer_binary_candidates(local_binary: Optional[str], name: str) -> List[str]:
    """Where ``name`` should be on the peer, most likely first (nodes share the bundle layout)."""
    out: List[str] = []
    if local_binary:
        out.append(peer_path(Path(local_binary).parent) + "/" + name)
    sc = _llama_rpc()
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
    from core.inference.llama_server_args import _flag_name

    tokens = [str(a) for a in args or ()]
    found: Optional[str] = None
    for index, token in enumerate(tokens):
        _, sep, inline = token.partition("=")
        # llama.cpp folds long-option underscores (--ctx_size is --ctx-size), so the sizing does too.
        if _flag_name(token) not in flags:
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
    extras_ask_for_native = False
    if from_extras:
        try:
            ctx = int(from_extras)
            extras_ask_for_native = ctx == 0
        except ValueError:
            pass  # llama-server will refuse it; nothing to price here
    if ctx <= 0 and not extras_ask_for_native:
        # Auto-layers omits -c, and the launch keeps a positive inherited LLAMA_ARG_CTX_SIZE
        # (routes.inference._inherited_ctx_size): the child runs at that length.
        raw_env_ctx = str(os.environ.get("LLAMA_ARG_CTX_SIZE") or "").strip()
        try:
            ctx = max(0, int(raw_env_ctx)) if raw_env_ctx else ctx
        except ValueError:
            pass

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


# Env twins of _SIDECAR_FLAGS: llama.cpp reads either route, so both are absolutised.
_REPLICA_ENV_PATHS = frozenset(
    {
        "LLAMA_ARG_MMPROJ",
        "LLAMA_ARG_SPEC_DRAFT_MODEL",
        "LLAMA_ARG_CHAT_TEMPLATE_FILE",
    }
)


def argv_or_env_rpc(argv: Sequence[str], env: Optional[Dict[str, str]] = None) -> bool:
    """Whether this llama-server is already an RPC split, however it was told to be."""
    for arg in argv or ():
        name = str(arg).partition("=")[0]
        if name == "--rpc":
            return True
    source = os.environ if env is None else env
    return bool(str(source.get("LLAMA_ARG_RPC") or "").strip())


def redacted_argv(argv: List[str]) -> List[str]:
    out = list(argv)
    for index, arg in enumerate(out):
        if arg == "--api-key" and index + 1 < len(out):
            out[index + 1] = "<redacted>"
        elif arg.startswith("--api-key="):
            out[index] = "--api-key=<redacted>"
    return out


def rpc_server_argv(binary: str, *, bind: str, port: int, cache: bool) -> List[str]:
    """``ggml-rpc-server`` on the peer; `-c` caches tensors to disk, so only where the user keeps the GGUF."""
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
    """The llama-server this node will LAUNCH, from the backend's own resolver: a second search order once ran mismatched builds."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend
    except Exception as exc:  # pragma: no cover - the backend package is always there
        logger.info("layer-split serving: cannot import the llama.cpp backend resolver: %s", exc)
        return None
    try:
        found = LlamaCppBackend._find_llama_server_binary()
    except Exception as exc:
        logger.info("layer-split serving: the backend's llama-server resolver failed: %s", exc)
        return None
    return str(found) if found else None


def rpc_server_binary() -> Optional[str]:
    """``ggml-rpc-server`` from beside the llama-server that launches."""
    launched = llama_server_binary()
    if launched:
        directory = osp.dirname(osp.abspath(launched))
        for name in rpc_server_names():
            candidate = osp.join(directory, name)
            if is_executable_file(candidate):
                return candidate
    sc = _llama_rpc()
    try:
        found = sc.rpc_server_binary() if sc is not None else None
    except Exception:
        return None
    return str(found) if found else None


# Keyed by (path, mtime); failures are cached so a broken binary costs one timeout.
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
        logger.info("layer-split serving: llama-server --help probe failed for %s: %s", path, exc)
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
    """For a flag hidden from the usage text: the fork prints usage for it, other builds say "invalid argument"."""
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
            "layer-split serving: llama-server %s probe failed for %s: %s",
            " ".join(probe),
            path,
            exc,
        )
        accepted = False
    _ACCEPTS[key] = accepted
    return accepted


def _first_shard(path: str) -> str:
    """The file that carries the header: shard 00001 of a split GGUF, else ``path``."""
    match = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", osp.basename(path))
    if not match:
        return path
    prefix, _index, count = match.groups()
    return osp.join(osp.dirname(path), f"{prefix}-00001-of-{count}.gguf")


def _arg_name(arg: Any) -> str:
    """The option name in ``arg`` the way llama-server itself reads it."""
    name = str(arg).strip().partition("=")[0]
    return name.replace("_", "-") if name.startswith("--") else name


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


# Env twins of the group-refused flags: llama.cpp reads them itself, so an argv-only check misses them.
_GROUPS_REFUSED_ENV = frozenset({"LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"})


def extra_args_refuse_pipeline_groups(
    extra_args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """The first pass-through flag or env setting the server refuses together with the groups."""
    for arg in extra_args or []:
        name = _arg_name(arg)
        if name in _GROUPS_REFUSED_FLAGS:
            return name
    source = os.environ if env is None else env
    for name in _GROUPS_REFUSED_ENV:
        if str(source.get(name, "")).strip():
            return name
    return None


def _snapshot_projector(model_file: Optional[str]) -> Optional[str]:
    """The multimodal projector Studio would pick up beside ``model_file``, or None."""
    if not model_file:
        return None
    try:
        # NOT realpath: HF snapshot links resolve into blobs/, where nothing ends in .gguf.
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
        logger.info("layer-split serving: could not read the projector %s: %s", path, exc)
        return False


def auto_projector_file(
    model_file: Optional[str], *, disable_vision: bool = False
) -> Optional[str]:
    """The projector the load will open without being named in the args, for sizing."""
    on_disk = _snapshot_projector(model_file)
    if on_disk and disable_vision and _projector_accepts_image(on_disk):
        return None
    return on_disk


def strip_layer_split_extra_args(extra: Sequence[str]) -> List[str]:
    """``extra`` without our appended split block, so an inheriting reload re-plans the split."""
    from core.inference import cluster_serving

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
        # A stale pid would make stop() kill whatever the peer reused that number for.
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
        """Kill the remote process by the pid this run printed, then its ssh session."""
        # Only while the ssh session that printed the pid is still up.
        if self.remote_pid and self.alive:
            await ssh_run(
                self.peer,
                f"kill {self.remote_pid} 2>/dev/null; sleep 1; kill -9 {self.remote_pid} 2>/dev/null; true",
                timeout = timeout,
            )
        elif self.remote_pid:
            logger.info(
                "layer-split serving: peer %s already exited; leaving remote pid %s to the remote "
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


_GGUF_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


_GGUF_UNPACK = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def _gguf_read_value(
    handle,
    value_type: int,
    depth: int = 0,
):
    """One GGUF value."""
    if value_type in _GGUF_FIXED:
        raw = handle.read(_GGUF_FIXED[value_type])
        if len(raw) < _GGUF_FIXED[value_type]:
            raise ValueError("truncated GGUF header")
        return struct.unpack(_GGUF_UNPACK[value_type], raw)[0]
    if value_type == 8:
        (length,) = struct.unpack("<Q", handle.read(8))
        return handle.read(length).decode("utf-8", "replace")
    if value_type == 9:
        if depth:
            raise ValueError("nested GGUF array")
        element, count = struct.unpack("<IQ", handle.read(12))
        # Skips vocab-sized tables; per-layer arrays (head_count_kv on hybrids) must survive.
        if element in _GGUF_FIXED and count > 4096:
            handle.seek(_GGUF_FIXED[element] * count, 1)
            return []
        return [_gguf_read_value(handle, element, depth + 1) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {value_type}")


def gguf_metadata(path: str) -> Dict[str, Any]:
    """The GGUF key/value header, or `{}` when `path` is not a readable GGUF."""
    try:
        with open(osp.expanduser(path), "rb") as handle:
            magic, version = struct.unpack("<4sI", handle.read(8))
            if magic != b"GGUF" or version not in (2, 3):
                return {}
            _, kv_count = struct.unpack("<QQ", handle.read(16))
            out: Dict[str, Any] = {}
            for _ in range(min(kv_count, 4096)):
                (key_len,) = struct.unpack("<Q", handle.read(8))
                key = handle.read(key_len).decode("utf-8", "replace")
                (value_type,) = struct.unpack("<I", handle.read(4))
                out[key] = _gguf_read_value(handle, value_type)
            return out
    except (OSError, ValueError, struct.error, UnicodeDecodeError):
        return {}


_LLAMA_RPC: Any = None
_LLAMA_RPC_LOOKED_UP = False


def _llama_rpc():
    """``studio.llama_rpc``; the ``studio`` package is put on the path when only the backend is."""
    global _LLAMA_RPC, _LLAMA_RPC_LOOKED_UP
    if _LLAMA_RPC_LOOKED_UP:
        return _LLAMA_RPC
    _LLAMA_RPC_LOOKED_UP = True
    try:
        from studio import llama_rpc as module  # type: ignore
        _LLAMA_RPC = module
        return _LLAMA_RPC
    except Exception:
        pass
    here = Path(__file__).resolve().parents[3]
    if not (here / "llama_rpc.py").is_file():
        return None
    try:
        import studio  # type: ignore
        if str(here) not in list(studio.__path__):
            studio.__path__.append(str(here))
    except ImportError:
        if str(here.parent) not in sys.path:
            sys.path.append(str(here.parent))
    try:
        from studio import llama_rpc as module  # type: ignore
        _LLAMA_RPC = module
    except Exception:
        _LLAMA_RPC = None
    return _LLAMA_RPC


class LoadDoesNotFit(Exception):
    """The requested load cannot be placed; the route answers it as a refusal, not a crash."""


def request_with(request: Any, updates: Dict[str, Any]) -> Any:
    """``request`` with ``updates``: a copy for a pydantic model, in place otherwise."""
    if not updates:
        return request
    try:
        return request.model_copy(update = dict(updates))
    except AttributeError:
        for key, value in updates.items():
            setattr(request, key, value)
        return request


# The load hooks. An optional serving provider (two-Spark serving) may take them over whole; without
# one, they drive the N-node cluster planner only.
_PROVIDER: Any = None
_PROVIDER_LOOKED_UP = False


def _provider() -> Any:
    """Two-Spark serving when it is installed; it keeps the cluster path inside its own hooks."""
    global _PROVIDER, _PROVIDER_LOOKED_UP
    if _PROVIDER is None and not _PROVIDER_LOOKED_UP:
        _PROVIDER_LOOKED_UP = True
        try:
            from core.inference import spark_serving
        except ImportError as exc:
            if exc.name != "core.inference.spark_serving":
                logger.warning("layer-split serving: two-Spark serving failed to import: %s", exc)
            return None
        _PROVIDER = spark_serving
    return _PROVIDER


def register_provider(provider: Any) -> None:
    """``provider`` answers every hook below instead of the cluster-only path (None restores it)."""
    global _PROVIDER, _PROVIDER_LOOKED_UP
    _PROVIDER = provider
    _PROVIDER_LOOKED_UP = True


def _cluster_active() -> bool:
    from core.inference import cluster_serving
    return (os.environ.get(ENV_TOGGLE) or "").strip() != "0" and cluster_serving.configured()


def enabled() -> bool:
    provider = _provider()
    if provider is not None:
        return provider.enabled()
    return _cluster_active()


def route_base_url(llama_backend: Any) -> Optional[str]:
    """Where requests for ``llama_backend`` go when a provider routes them, else None."""
    provider = _provider()
    return provider.route_base_url(llama_backend) if provider is not None else None


def tag_conversation(payload: Dict[str, Any], thread_id: Optional[str]) -> None:
    provider = _provider()
    if provider is not None:
        provider.tag_conversation(payload, thread_id)


def replica_count(llama_backend: Any) -> int:
    """llama-server replicas serving ``llama_backend``'s requests; 1 without a routing provider."""
    provider = _provider()
    return provider.replica_count(llama_backend) if provider is not None else 1


def current_topology() -> Optional[str]:
    provider = _provider()
    if provider is not None:
        return provider.current_topology()
    from core.inference import cluster_serving

    if cluster_serving._STATE is not None and (
        _cluster_active() or cluster_serving._STATE.serving()
    ):
        return cluster_serving._STATE.topology
    return None


async def before_load(
    request: Any,
    n_parallel: int,
    *,
    inherited_extra_args: Optional[List[str]] = None,
    cancel_event: Any = None,
    confirm_teardown: Any = None,
) -> Any:
    provider = _provider()
    if provider is not None:
        return await provider.before_load(
            request,
            n_parallel,
            inherited_extra_args = inherited_extra_args,
            cancel_event = cancel_event,
            confirm_teardown = confirm_teardown,
        )
    from core.inference import cluster_serving

    alongside = bool(
        getattr(request, "alongside", False) or getattr(request, "force_alongside", False)
    )
    replacing = not alongside and str(getattr(request, "engine", "auto") or "auto") == "auto"
    if cluster_serving._STATE is not None:
        cluster_serving._STATE.settle_internal_load()
    if _cluster_active():
        return await cluster_serving.state().before_load(
            request,
            n_parallel,
            inherited_extra_args = inherited_extra_args,
            cancel_event = cancel_event,
        )
    cluster = cluster_serving._STATE
    if cluster is not None and (cluster.workers or cluster._backend is not None):
        # The cluster config went away: its workers retire once this load commits.
        cluster.retire_on_commit = not alongside
        if (
            replacing
            and cluster.retire_on_commit
            and cluster.block
            and getattr(cluster._backend, "is_loaded", False)
        ):
            # after_load retires the workers, so the primary using them must not be kept as is.
            request = request_with(request, {"force_reload": True})
    return request


async def after_load(
    llama_backend: Any,
    n_parallel: int,
    *,
    cancel_event: Any = None,
) -> bool:
    """True when the load was cancelled during the replica startup and must be undone."""
    provider = _provider()
    if provider is not None:
        return await provider.after_load(llama_backend, n_parallel, cancel_event = cancel_event)
    from core.inference import cluster_serving

    if cluster_serving._STATE is not None and cluster_serving._STATE.in_flight:
        await cluster_serving._STATE.after_load(llama_backend)
        return False
    if cluster_serving._STATE is not None and cluster_serving._STATE.retire_on_commit:
        await cluster_serving._STATE.shutdown()
    return False


async def load_failed() -> None:
    provider = _provider()
    if provider is not None:
        return await provider.load_failed()
    from core.inference import cluster_serving

    if cluster_serving._STATE is not None and cluster_serving._STATE.in_flight:
        await cluster_serving._STATE.load_failed()
        return
    if cluster_serving._STATE is not None and cluster_serving._STATE.retire_on_commit:
        await cluster_serving._STATE.failed_while_retiring()


async def reconcile_internal_load() -> None:
    """Tell the planner about a load that did not come through ``before_load``."""
    provider = _provider()
    if provider is not None:
        return await provider.reconcile_internal_load()
    from core.inference import cluster_serving

    if cluster_serving._STATE is not None:
        cluster_serving._STATE.note_internal_load()


async def shutdown() -> None:
    provider = _provider()
    if provider is not None:
        return await provider.shutdown()
    from core.inference import cluster_serving

    if cluster_serving._STATE is not None:
        await cluster_serving._STATE.shutdown()


def status() -> Dict[str, Any]:
    provider = _provider()
    if provider is not None:
        return provider.status()
    from core.inference import cluster_serving

    if _cluster_active():
        return cluster_serving.state().status()
    if cluster_serving._STATE is not None and cluster_serving._STATE.serving():
        return dict(cluster_serving._STATE.status(), retiring = True)
    return {"enabled": False, "topology": None, "reason": "not a paired DGX Spark"}


def reset_shared_for_tests() -> None:
    global _LLAMA_RPC, _LLAMA_RPC_LOOKED_UP
    _LLAMA_RPC = None
    _LLAMA_RPC_LOOKED_UP = False
    _HELP_TEXT.clear()
    _ACCEPTS.clear()


def reset_for_tests() -> None:
    reset_shared_for_tests()
    provider = _provider()
    if provider is not None and hasattr(provider, "reset_for_tests"):
        provider.reset_for_tests()
