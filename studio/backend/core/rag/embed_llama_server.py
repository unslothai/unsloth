# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in GGUF embedder (RAG_EMBED_BACKEND=llama-server, no torch): own llama-server --embedding
subprocess on /v1/embeddings and /tokenize; auto falls back to CPU if GPU fails, RAG_EMBED_DEVICE
forces; each request respawns it if dead; only llama_cpp static helpers, no LlamaCppBackend()."""

from __future__ import annotations

from core.training.account_jobs import account_path, managed_account
from utils.account_context import account_thread
from utils.gpu_memory_events import invalidates_gpu_memory as _invalidates_gpu_memory
import atexit
import logging
import os
import struct
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np

from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import windows_hidden_subprocess_kwargs
from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    forget_pid,
    is_process_shutting_down,
)

from . import config
from utils.paths.path_utils import is_appledouble_metadata

logger = logging.getLogger(__name__)

# httpx transport errors meaning "the server is gone" -> trigger a respawn.
_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.WriteError,
)


_GGUF_SCALAR_WIDTHS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

# Matches llama.cpp's -ub default; a longer embedding prompt is refused with a 500.
_UBATCH_SIZE = 512


def _skip_gguf_value(f, vtype: int) -> None:
    if vtype == 8:
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
    elif vtype == 9:
        elem_type, count = struct.unpack("<IQ", f.read(12))
        if elem_type in _GGUF_SCALAR_WIDTHS:
            f.seek(_GGUF_SCALAR_WIDTHS[elem_type] * count, 1)
        else:
            for _ in range(count):
                _skip_gguf_value(f, elem_type)
    else:
        f.seek(_GGUF_SCALAR_WIDTHS[vtype], 1)


def _gguf_arch_uint(path: str, field: str) -> tuple[str | None, int | None]:
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return None, None
            f.read(4)
            _, kv_count = struct.unpack("<QQ", f.read(16))
            arch = None
            values: dict[str, int] = {}
            for _ in range(kv_count):
                key = f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8")
                vtype = struct.unpack("<I", f.read(4))[0]
                if key == "general.architecture" and vtype == 8:
                    arch = f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8")
                elif key.endswith(f".{field}") and vtype in (4, 10):
                    values[key] = int.from_bytes(f.read(_GGUF_SCALAR_WIDTHS[vtype]), "little")
                else:
                    _skip_gguf_value(f, vtype)
                if arch is not None and f"{arch}.{field}" in values:
                    return arch, values[f"{arch}.{field}"]
    except (OSError, struct.error, UnicodeDecodeError, KeyError, ValueError):
        return None, None
    return arch, None


def _gguf_context_length(path: str) -> int | None:
    return _gguf_arch_uint(path, "context_length")[1] or None


_GGUF_POOLING = {1: "mean", 2: "cls", 3: "last"}
_GGUF_POOLING_DEFAULT = {"nomic-bert": "mean", "nomic-bert-moe": "mean"}


@lru_cache(maxsize = 32)
def _gguf_pooling_at(path: str, mtime_ns: int, size: int) -> str:
    arch, value = _gguf_arch_uint(path, "pooling_type")
    if value in _GGUF_POOLING:
        return _GGUF_POOLING[value]
    if value is None:
        # Older Nomic embedders omit pooling metadata: mean-pool that family; unknown archs keep legacy CLS.
        return _GGUF_POOLING_DEFAULT.get((arch or "").lower(), "cls")
    return "cls"


def _gguf_pooling(path: str) -> str:
    try:
        st = os.stat(path)
    except OSError:
        return "cls"
    return _gguf_pooling_at(path, st.st_mtime_ns, st.st_size)


def _resolve_entrypoint(binary: str) -> str:
    """Resolves past a wrapper on macOS, since SIP purges DYLD_* when the wrapper's /bin/sh starts."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._exec_path_for_launch(binary) or binary
    except Exception:  # noqa: BLE001 - launch what we were given
        return binary


def _binary_lib_dir(binary: str) -> str:
    """Directory holding ``binary``'s sibling libraries, through an entrypoint."""
    try:
        from core.inference.llama_cpp import _llama_lib_dir
        return str(_llama_lib_dir(binary))
    except Exception:  # noqa: BLE001 - fall back to the plain parent
        return str(Path(binary).parent)


def _with_dyld_path(env: dict[str, str], lib_dir: str) -> dict[str, str]:
    """Returns a copy with lib_dir first on DYLD_LIBRARY_PATH, sharing the chat backend's prepend."""
    out = dict(env)
    try:
        from core.inference.llama_cpp import _prepend_loader_dir
        out["DYLD_LIBRARY_PATH"] = _prepend_loader_dir(out.get("DYLD_LIBRARY_PATH", ""), lib_dir)
    except Exception:  # noqa: BLE001 - a search path is better than none
        existing = [p for p in out.get("DYLD_LIBRARY_PATH", "").split(os.pathsep) if p]
        out["DYLD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys([lib_dir, *existing]))
    return out


class LlamaServerBackend:
    """Manages a llama.cpp embedding subprocess and talks to it over HTTP."""

    def __init__(self) -> None:
        # Lifecycle (spawn/restart/kill) is serialized; HTTP requests are not.
        self._lifecycle_lock = threading.Lock()
        # Unload waits for in-flight encode/tokenize/probe calls and bars restarts after unpublish.
        self._operation_condition = threading.Condition()
        self._active_operations = 0
        self._operation_local = threading.local()
        # Repo/pooling from this thread's last encode, since the shared backend may switch models meanwhile.
        self._served_identity_local = threading.local()
        self._closed = False
        self._process: subprocess.Popen | None = None
        self._port: int | None = None
        self._stdout_lines: list[str] = []
        self._stdout_thread: threading.Thread | None = None
        # dim() takes _serve_lock like requests; reentrant because dim() calls encode() on the same thread.
        self._dim: int | None = None
        self._max_tokens: int | None = None
        # Readiness and request must be atomic: a model swap between them misroutes the POST.
        self._serve_lock = threading.RLock()
        self._model_path: str | None = None
        # Read when the path is adopted: the running server keeps this pooling even if the file later vanishes.
        self._model_pooling: str | None = None
        # A Settings change makes the cached path/dim stale, forcing a re-resolve and respawn (see _ensure_ready).
        self._model_repo: str | None = None
        self._binary: str | None = None
        self._binary_path_revision: int | None = None
        # Sticky after an auto GPU start fails or a GPU server crashes twice on one request.
        self._force_cpu = False
        self._on_gpu = False
        # trust_env=False skips HTTP(S)_PROXY; full URLs per request survive a respawn.
        self._client = httpx.Client(timeout = config.EMBED_REQUEST_TIMEOUT_S, trust_env = False)
        self._api_key: str | None = None
        atexit.register(self._shutdown)

    @contextmanager
    def _operation(self):
        depth = getattr(self._operation_local, "depth", 0)
        with self._operation_condition:
            if self._closed and depth == 0:
                raise RuntimeError("llama-server embedding backend was unloaded")
            if depth == 0:
                self._active_operations += 1
            self._operation_local.depth = depth + 1
        try:
            yield
        finally:
            with self._operation_condition:
                next_depth = self._operation_local.depth - 1
                self._operation_local.depth = next_depth
                if next_depth == 0:
                    self._active_operations -= 1
                    if self._active_operations == 0:
                        self._operation_condition.notify_all()

    @property
    def _base_url(self) -> str:
        return f"http://{config.EMBED_HOST}:{self._port}"

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _resolve_binary(self) -> str:
        """Find llama-server, verify embeddings support, cache it. Raises if
        missing/unsupported."""
        if self._binary is not None:
            return self._binary
        from core.inference.llama_cpp import LlamaCppBackend
        from utils.llama_cpp_path_settings import custom_llama_cpp_path_revision

        path_revision = custom_llama_cpp_path_revision()
        binary = LlamaCppBackend._find_llama_server_binary()
        if not binary:
            raise RuntimeError(
                "llama-server binary not found; cannot use RAG_EMBED_BACKEND="
                "llama-server. Install llama.cpp or set LLAMA_SERVER_PATH / "
                "UNSLOTH_LLAMA_CPP_PATH."
            )
        # Resolve the real executable before the probe: an entrypoint loses DYLD_* to SIP.
        binary = _resolve_entrypoint(binary)
        self._assert_embedding_support(binary)
        self._binary = binary
        self._binary_path_revision = path_revision
        return binary

    @staticmethod
    @lru_cache(maxsize = 8)
    def _help_text(binary: str) -> str:
        """Runs under the real launch's loader environment on macOS, else a loader error reads as help."""
        probe_env = None
        if sys.platform == "darwin":
            try:
                from core.inference.llama_cpp import LlamaCppBackend
                probe_env = LlamaCppBackend._llama_server_env_for_binary(binary)
            except Exception:  # noqa: BLE001 - probe with the inherited env
                probe_env = None
        try:
            proc = subprocess.run(
                [binary, "--help"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 30,
                env = probe_env,
                **windows_hidden_subprocess_kwargs(),
            )
            return (proc.stdout or "") + (proc.stderr or "")
        except Exception as e:  # noqa: BLE001
            logger.warning("could not run `llama-server --help`: %s", e)
            return ""

    def _assert_embedding_support(self, binary: str) -> None:
        help_text = self._help_text(binary)
        # Empty help -> assume capable (a missing flag still fails at spawn).
        if help_text and "--embedding" not in help_text:
            raise RuntimeError(
                "the bundled llama-server build lacks --embedding support; "
                "RAG_EMBED_BACKEND=llama-server requires an embeddings-capable build"
            )

    @staticmethod
    def _resolve_local_gguf(model: str) -> str | None:
        """A custom model may be a local .gguf file or a directory holding one;
        resolve it without the hub. None when the value is not a local path."""
        from utils.models.model_config import _local_gguf_load_path, colocated_split_shards
        from utils.paths import normalize_path

        # Normalize like the other probes, else a WSL user's C: path GGUF goes to the Hub as a bad filename.
        p = Path(normalize_path(model)).expanduser()

        if p.is_file() and p.suffix.lower() == ".gguf":
            return None if is_appledouble_metadata(p) else str(p)
        if p.is_dir():
            files = [f for f in p.iterdir() if not is_appledouble_metadata(f)]
            # Same pick as a hub listing gets, and its name tiebreak resolves a directory of split shards to shard 1.
            remaining = list(files)
            while remaining:
                picked = LlamaServerBackend._pick_gguf(remaining, key = lambda f: f.name)
                if picked is None:
                    break
                # llama-server opens split siblings implicitly, so an incomplete family is unusable.
                if colocated_split_shards(picked)[1]:
                    # Enter a split family at shard 1 with its full symlink set, as the chat loader does for local GGUF.
                    return str(_local_gguf_load_path(picked))
                remaining = [f for f in remaining if f != picked]
            raise RuntimeError(f"no .gguf file found in local model dir {model!r}")
        return None

    @classmethod
    def _pick_complete_gguf(cls, names):
        """Skips a torn GGUF family, so the loop tries the next repo instead of one it cannot finish."""
        remaining = list(names)
        while remaining:
            picked = cls._pick_gguf(remaining)
            if picked is None:
                return None, []
            family = cls._split_family(names, picked)
            if family:
                return picked, family
            remaining = [name for name in remaining if name != picked]
        return None, []

    @staticmethod
    def _split_family(names, picked: str):
        """A lone shard is not a plan: llama-server opens split siblings implicitly, so all must be
        fetched."""
        from pathlib import PurePosixPath
        from utils.models.model_config import _GGUF_SPLIT_FILE_RE

        path = PurePosixPath(picked)
        match = _GGUF_SPLIT_FILE_RE.match(path.name)
        if match is None:
            return [picked]
        total = int(match.group("total"))
        if total < 1:
            return None
        found: dict[int, str] = {}
        for name in names:
            other = PurePosixPath(name)
            other_match = _GGUF_SPLIT_FILE_RE.match(other.name)
            if (
                other.parent != path.parent
                or other_match is None
                or other_match.group("prefix").casefold() != match.group("prefix").casefold()
                or other_match.group("total") != match.group("total")
            ):
                continue
            index = int(other_match.group("index"))
            if 1 <= index <= total:
                found[index] = name
        if len(found) != total:
            return None
        return [found[index] for index in range(1, total + 1)]

    @staticmethod
    def _pick_gguf(
        names,
        *,
        key = None,
        require_variant = False,
    ):
        """Refuses a variant fallback: on a hub listing it means unpublished, in cache it may be stale."""
        from utils.models.model_config import _is_mtp_drafter

        name_of = key or (lambda n: n)
        usable = [
            n
            for n in names
            if name_of(n).lower().endswith(".gguf")
            and "mmproj" not in name_of(n).lower()
            # Drafters are companions like mmproj: a cache holding only one would read as holding the embedder.
            and not _is_mtp_drafter(name_of(n))
        ]
        if not usable:
            return None
        variant = config.EMBED_GGUF_VARIANT.lower()
        match = [n for n in usable if variant in name_of(n).lower()]
        if not match:
            if require_variant:
                return None
            match = usable
        # Name breaks a length tie: a directory scan yields no guaranteed order.
        return sorted(match, key = lambda n: (len(name_of(n)), name_of(n)))[0]

    @staticmethod
    def _cached_snapshot_dir(repo_id: str) -> Path | None:
        """Pins the embedder to refs/main, since the stored identity records no revision to compare
        against."""
        from utils.hf_cache_settings import active_hf_hub_cache
        from utils.paths import resolve_cached_repo_id_case

        # Casing-differing repo ids still match the cache folder; a miss re-downloads or fails offline.
        cased = resolve_cached_repo_id_case(repo_id)
        repo_dir = Path(active_hf_hub_cache()) / f"models--{cased.replace('/', '--')}"
        try:
            rev = (repo_dir / "refs" / "main").read_text(encoding = "utf-8").strip()
        except (OSError, ValueError):
            return None
        # A ref file holds a bare commit hash; a separator would join out of the cache.
        if not rev or rev in (".", "..") or "/" in rev or "\\" in rev:
            return None
        snapshot = repo_dir / "snapshots" / rev
        return snapshot if snapshot.is_dir() else None

    @staticmethod
    def _resolve_cached_gguf(repo_id: str, *, require_variant: bool = True) -> str | None:
        """Finds the GGUF on disk without the network, so restarts do not re-list a repo that may
        blackhole."""
        from utils.models.model_config import colocated_split_shards

        snapshot = LlamaServerBackend._cached_snapshot_dir(repo_id)
        if snapshot is None:
            return None
        try:
            # Not a *.gguf glob: it is case-sensitive and misses quant-per-directory layouts.

            files = [
                p for p in snapshot.rglob("*") if p.is_file() and not is_appledouble_metadata(p)
            ]
        except OSError:
            return None
        # Split sets serve only when whole; drop an unservable pick and retry, or it shadows a complete one.
        while files:
            pick = LlamaServerBackend._pick_gguf(
                files,
                key = lambda p: p.relative_to(snapshot).as_posix(),
                require_variant = require_variant,
            )
            # exists() guards only the window since the scan; is_file() dropped evicted blobs.
            if pick is None or not pick.exists():
                return None
            if colocated_split_shards(pick)[1]:
                return str(pick)
            files = [p for p in files if p != pick]
        return None

    @staticmethod
    def cached_pooling(model: str) -> str | None:
        """Pooling of ``model``'s GGUF already on disk, found without the network; None when
        the loader cannot yet know which file it will serve. Match its local, planned and configured-
        variant order so identity prediction never reads metadata from a different cached family."""
        try:
            desired = config.effective_gguf_repo_for_embedding_model(model)
            path = LlamaServerBackend._resolve_local_gguf(model)
            path = path or LlamaServerBackend._planned_family_path(model, desired)
            path = path or LlamaServerBackend._resolve_cached_gguf(desired)
            if path is None:
                try:
                    from utils.embedding_model_settings import get_stored_download_pending
                    download_pending = get_stored_download_pending(model)
                except Exception:  # noqa: BLE001 - old/unavailable settings store
                    download_pending = False
                if download_pending:
                    path = LlamaServerBackend._resolve_cached_gguf(desired, require_variant = False)
        except Exception:  # noqa: BLE001 - identity prediction must not block ingestion
            return None
        return None if path is None else _gguf_pooling(path)

    def _resolve_model_path(self, model_name: str | None = None) -> str:
        """Uses the caller's pinned model, not the live setting, which would resolve other weights."""
        account_path(model_name, reference = True)
        model = model_name or config.effective_embedding_model()
        # Captured once; tagged with its repo, so a mid-download setting change reads as stale, respawning.
        desired = config.effective_gguf_repo_for_embedding_model(model)
        if self._model_path is not None and self._model_repo == desired:
            if Path(self._model_path).is_file():
                return self._model_path
            # Process stopped here; a remembered cache path may be evicted, so re-resolve via cache/Hub.
            self._model_path = None
            self._model_pooling = None
            self._dim = None
            self._max_tokens = None
        local = self._resolve_local_gguf(model)
        if local is not None:
            return self._adopt_model_path(local, desired)
        # Identity carries the model and repo but not the file, so a preferred variant arriving later would
        # change the weights under an existing index without changing its tag.
        planned = self._planned_family_path(model, desired)
        if planned is not None:
            try:
                from utils.embedding_model_settings import clear_stored_download_pending
                clear_stored_download_pending(model)
            except Exception:  # noqa: BLE001 - a settings write must not fail a load
                pass
            return self._adopt_model_path(planned, desired)
        # Companion repo, then the naming variants, then the model repo itself.
        repo = desired
        candidates = list(dict.fromkeys([repo, *config.gguf_repo_candidates(model)]))
        # Only the preferred repo: a file cached under the fallback must not pre-empt a companion repo the
        # hub can still resolve.
        cached = self._resolve_cached_gguf(desired)
        if cached is not None:
            # Retire the pending marker here, or the model stays cache-only for the life of the install and a
            # later eviction reads as "never downloaded".
            try:
                from utils.embedding_model_settings import clear_stored_download_pending
                clear_stored_download_pending(model)
            except Exception:  # noqa: BLE001 - a settings write must not fail a load
                pass
            return self._adopt_model_path(cached, desired)
        try:
            from utils.embedding_model_settings import get_stored_download_pending
            download_pending = get_stored_download_pending(model)
        except Exception:  # noqa: BLE001 - old/unavailable settings store
            download_pending = False
        if download_pending:
            # A published repo may not carry the configured quant, so accept any complete cached quant; never
            # fall through to the online loader while the transfer is pending or cancelled.
            cached = self._resolve_cached_gguf(desired, require_variant = False)
            if cached is not None:
                # Reaching here means the configured variant is NOT cached (the strict pass returned if it
                # were), which happens two ways that need opposite answers.
                if self._is_planned_family(model, desired, cached):
                    # Serve a stand-in instead of failing the index; keep the marker or it pins the wrong quant.
                    try:
                        from utils.embedding_model_settings import clear_stored_download_pending
                        clear_stored_download_pending(model)
                    except Exception:  # noqa: BLE001 - a settings write must not fail a load
                        pass
                # Serve a stand-in rather than fail the index, but leave the marker: retiring on it would pin this
                # model to the wrong quant.
                return self._adopt_model_path(cached, desired)
            # Typed: the same condition the ST path raises, and /v1/embeddings answers it
            # with a 409. Bare, it fell into the catch-all as a 502 naming nothing.
            from .embeddings import EmbeddingModelDownloadRequiredError

            raise EmbeddingModelDownloadRequiredError(
                f"Embedding model {model!r} is not downloaded yet. "
                "Finish its Settings download before indexing documents."
            )
        return self._resolve_uncached_model_path(desired, candidates)

    @staticmethod
    def _planned_family_path(model: str, repo_id: str) -> str | None:
        """Whole family or nothing: a partial family is a torn transfer llama-server cannot open."""
        try:
            from pathlib import PurePosixPath
            from utils.embedding_model_settings import get_stored_gguf_files

            planned = get_stored_gguf_files(model)
            if not planned:
                return None
            snapshot = LlamaServerBackend._cached_snapshot_dir(repo_id)
            if snapshot is None:
                return None
            paths = []
            for name in planned:
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts:
                    return None
                path = snapshot.joinpath(*relative.parts)
                if not path.is_file():
                    return None
                paths.append(path)
            # Ordered shard 1..N by the planner, so the first is where llama-server enters the family.
            return str(paths[0]) if paths else None
        except Exception:  # noqa: BLE001 - an unreadable record answers nothing
            return None

    @staticmethod
    def _is_planned_family(model: str, repo_id: str, path: str) -> bool:
        """Compares the path relative to the snapshot: base names collide when each quant has its
        own folder."""
        try:
            from pathlib import PurePosixPath
            from utils.embedding_model_settings import get_stored_gguf_files

            planned = get_stored_gguf_files(model)
            if not planned:
                return False
            snapshot = LlamaServerBackend._cached_snapshot_dir(repo_id)
            if snapshot is None:
                return False
            relative = Path(path).relative_to(snapshot)
            names = {PurePosixPath(name).as_posix().casefold() for name in planned}
            return PurePosixPath(relative.as_posix()).as_posix().casefold() in names
        except Exception:  # noqa: BLE001 - an unreadable record or an outside path answers nothing
            return False

    def _adopt_model_path(self, path: str, desired: str) -> str:
        """Serve `path` for `desired`; the width is re-probed against whatever is adopted."""
        self._model_path = path
        self._model_repo = desired
        self._model_pooling = _gguf_pooling(path) if path else None
        self._dim = None
        self._max_tokens = None
        return self._model_path

    # A listing, not a transfer: a working hub answers well inside a second.
    _LIST_DEADLINE_S = 20.0

    def _resolve_uncached_model_path(self, desired: str, candidates: list[str]) -> str:
        """list_repo_files has no timeout, so the listing needs a deadline; transfers are bounded
        per read."""
        from huggingface_hub import hf_hub_download, list_repo_files
        from core.inference.llama_cpp import _hf_offline_if_unreachable
        from utils.utils import call_with_deadline

        token = False if managed_account() else os.environ.get("HF_TOKEN") or None
        repo = desired
        filename: str | None = None
        family: list[str] = []
        errors: list[str] = []
        with _hf_offline_if_unreachable():
            for candidate in candidates:
                try:
                    names = call_with_deadline(
                        lambda c = candidate: list_repo_files(c, token = token),
                        self._LIST_DEADLINE_S,
                        name = "embed-gguf-listing",
                    )
                    filename, family = self._pick_complete_gguf(names)
                except Exception as e:  # noqa: BLE001 - missing/gated/stalled -> next candidate
                    errors.append(f"{candidate!r}: {e}")
                    continue
                if filename is not None:
                    repo = candidate
                    break
                errors.append(f"{candidate!r}: no .gguf files")
            if filename is None:
                # Only where the hub failed to NAME a file: a transfer failing on its own (no disk, bad checksum)
                # must surface.
                for candidate in candidates:
                    cached = self._resolve_cached_gguf(candidate, require_variant = False)
                    if cached is not None:
                        logger.warning(
                            "using cached GGUF embedder %s: the %s variant could not be "
                            "resolved (%s)",
                            cached,
                            config.EMBED_GGUF_VARIANT,
                            "; ".join(errors),
                        )
                        return self._adopt_model_path(cached, desired)
                raise RuntimeError("no .gguf embedder found; tried " + "; ".join(errors))
            logger.info("resolving GGUF embedder %s/%s", repo, filename)
            from utils.hf_cache_settings import active_hf_hub_cache

            cache_dir = active_hf_hub_cache()
            # Fetch the whole split family, not just the picked shard: -m shard1 makes llama-server open the
            # siblings itself.
            downloaded = None
            for name in family:
                fetched = hf_hub_download(
                    repo_id = repo,
                    filename = name,
                    token = token,
                    cache_dir = cache_dir,
                )
                if name == filename:
                    downloaded = fetched
        return self._adopt_model_path(downloaded, desired)

    # Min free VRAM (MiB) for the embedder; below this, auto stays on CPU.
    _MIN_GPU_FREE_MIB = 1024

    def _use_gpu(self) -> bool:
        """A sticky CPU fallback overrides auto, so a restart does not pay the GPU startup timeout again."""
        dev = config.embed_device_preference()
        if dev == "gpu" and not self._force_cpu:
            return True
        if dev == "cpu" or self._force_cpu:
            return False
        return self._gpu_available()

    @staticmethod
    def _gpu_available() -> bool:
        """Smi-first GPU probe leaves no CUDA/HIP context behind; llama-server opts into the ROCm
        arch gate."""
        from utils.hardware import is_apple_silicon

        if is_apple_silicon():
            return True
        from core.inference.llama_cpp import LlamaCppBackend

        # [(idx, free_mib)], honors CVD
        gpus = LlamaCppBackend._get_gpu_free_memory(for_llama_server = True)
        return any(free >= LlamaServerBackend._MIN_GPU_FREE_MIB for _, free in gpus)

    @staticmethod
    def _arch_gated_gpu_ids(binary: str) -> list[int]:
        """Pins survivors, not just a supported device: an uncovered ROCm agent still crashes the child."""
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._arch_gate_survivors(binary)

    def _build_cmd(
        self,
        binary: str,
        model_path: str,
        port: int,
        *,
        use_gpu: bool,
        pooling: str | None = None,
    ) -> list[str]:
        # No --embd-normalize (missing in some builds; normalized in Python); --fit off keeps ctx fixed.
        cmd = [
            binary,
            "-m",
            model_path,
            "--host",
            config.EMBED_HOST,
            "--port",
            str(port),
            "--embedding",
            "--pooling",
            pooling or _gguf_pooling(model_path),
            "--fit",
            "off",
        ]
        # Raising -ub preallocates n_vocab * n_ubatch * 4 bytes; /props does not report the batch.
        cmd += ["-ub", str(_UBATCH_SIZE)]
        # -1 offloads every layer (matches the chat server); 0 keeps it on CPU.
        cmd += ["-ngl", "-1" if use_gpu else "0"]
        return cmd

    def _build_env(self, binary: str, *, use_gpu: bool) -> dict[str, str]:
        env = child_env_without_native_path_secret()
        # Strip inherited GGML_CUDA_P2P=0 here too: llama.cpp reads it as ON (#10613).
        from core.inference.llama_cpp import LlamaCppBackend

        LlamaCppBackend._sanitize_p2p_env(env)
        env["LLAMA_SET_ROWS"] = "1"
        if sys.platform == "darwin":
            # Managed installs put an entrypoint in front; dylibs sit next to the real target (#8566).
            env = _with_dyld_path(env, _binary_lib_dir(binary))
        elif use_gpu:
            if sys.platform == "win32":
                # Chat's DLL search path: without it a venv-hosted cudart is never found and the CUDA build runs on the CPU.
                from core.inference.llama_cpp import _llama_lib_dir
                path_dirs = LlamaCppBackend._build_windows_path_dirs(
                    str(_llama_lib_dir(binary)), sys.prefix, os.environ.get("CUDA_PATH", "")
                )
                env["PATH"] = ";".join(path_dirs) + ";" + env.get("PATH", "")
            else:
                # Path(binary).parent, unresolved: resolving would move the first LD_LIBRARY_PATH entry of every symlinked install.
                self._add_linux_cuda_libs(env, str(Path(binary).parent))
            _pinned = self._arch_gated_gpu_ids(binary)
            if _pinned:
                from core.inference.llama_cpp import LlamaCppBackend

                # prefer_rocr: a HIP-only mask still lets HSA enumerate, and die on, the unsupported agent.
                LlamaCppBackend._emit_child_gpu_visibility(
                    env, ",".join(str(i) for i in _pinned), prefer_rocr = True
                )
                logger.info("pinning the embed server to arch-supported GPU(s) %s", _pinned)
        # Not else: a macOS CPU start still has to blank the devices.
        if not use_gpu:
            # Blank devices so a CUDA build stays on CPU and reserves no VRAM.
            env["CUDA_VISIBLE_DEVICES"] = ""
            # HIP ignores CUDA_VISIBLE_DEVICES when HIP_VISIBLE_DEVICES is set; leave ROCR mask alone (#7624).
            env["HIP_VISIBLE_DEVICES"] = "-1"
            # A stale LLAMA_ARG_DEVICE/MAIN_GPU naming a hidden device makes llama.cpp exit, not fall back.
            from core.inference.llama_cpp import LlamaCppBackend

            LlamaCppBackend._clear_device_placement_env(env)
        return env

    @staticmethod
    def _add_linux_cuda_libs(env: dict[str, str], binary_dir: str) -> None:
        """Best-effort LD_LIBRARY_PATH so the prebuilt binary finds CUDA libs."""
        import glob
        import platform

        if sys.platform == "win32":
            return
        arch = platform.machine()
        lib_dirs = [binary_dir]
        # glob.escape: a prefix with [brackets] is otherwise read as a pattern.
        site = os.path.join(glob.escape(sys.prefix), "lib", "python*", "site-packages")
        pip_dirs = []
        for pattern in (
            os.path.join(site, "nvidia", "cu*", "lib"),
            os.path.join(site, "nvidia", "cudnn", "lib"),
            os.path.join(site, "torch", "lib"),
        ):
            pip_dirs.extend(d for d in glob.glob(pattern) if os.path.isdir(d))
        system_dirs = [
            d
            for d in (
                "/usr/local/cuda/lib64",
                f"/usr/local/cuda/targets/{arch}-linux/lib",
                "/usr/local/cuda-12/lib64",
            )
            if os.path.isdir(d)
        ]
        from utils.tegra import TEGRA_LIB_DIR, is_tegra

        if is_tegra():
            tegra = [TEGRA_LIB_DIR] if os.path.isdir(TEGRA_LIB_DIR) else []
            lib_dirs.extend(tegra + system_dirs + pip_dirs)
        else:
            lib_dirs.extend(pip_dirs + system_dirs)
        existing = env.get("LD_LIBRARY_PATH", "")
        joined = ":".join(lib_dirs)
        env["LD_LIBRARY_PATH"] = f"{joined}:{existing}" if existing else joined

    def _drain_stdout(self, proc: subprocess.Popen) -> None:
        """Drain the child's stdout so its pipe buffer never deadlocks; keep the
        tail for crash diagnostics."""
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.rstrip()
                if line:
                    self._stdout_lines.append(line)
                    if len(self._stdout_lines) > 200:
                        del self._stdout_lines[:-200]
                    logger.debug("[llama-embed] %s", line)
        except Exception:  # noqa: BLE001 - drain thread must never raise
            pass

    @_invalidates_gpu_memory("embedding server start")
    def _spawn(self, model_name: str | None = None) -> None:
        """Start the embed server (caller holds the lock). A failed GPU start falls
        back to CPU once, unless ``RAG_EMBED_DEVICE`` is the literal ``gpu``."""
        use_gpu = self._use_gpu()
        try:
            self._spawn_once(use_gpu, model_name)
        except RuntimeError:
            # A shutdown refusal is terminal: the CPU fallback would just spawn into the
            # same latch and be refused again.
            if is_process_shutting_down():
                raise
            if use_gpu and not config.embed_device_requires_gpu():
                logger.warning("embed server GPU start failed; falling back to CPU")
                self._force_cpu = True
                self._spawn_once(False, model_name)
            else:
                raise

    def _spawn_once(
        self,
        use_gpu: bool,
        model_name: str | None = None,
    ) -> None:
        binary = self._resolve_binary()
        model_path = self._resolve_model_path(model_name)
        # The cached path can be replaced in place between server lifetimes. Capture pooling once
        # per spawn so the explicit llama-server flag and the vectors' recorded identity agree.
        pooling = _gguf_pooling(model_path)
        self._model_pooling = pooling
        port = config.EMBED_PORT or self._find_free_port()
        env = self._build_env(binary, use_gpu = use_gpu)
        cmd = self._build_cmd(binary, model_path, port, use_gpu = use_gpu, pooling = pooling)
        logger.info(
            "starting llama-server embedder (%s): %s",
            "gpu" if use_gpu else "cpu",
            " ".join(cmd),
        )
        from core.inference.llama_cpp import (
            LlamaCppBackend,
            _llama_server_api_key_enabled,
            _llama_server_key_launch,
        )

        self._api_key = key_file = None
        if _llama_server_api_key_enabled() and LlamaCppBackend.probe_server_capabilities(
            binary
        ).get("supports_api_key_file", True):
            # Per-spawn key against permissive CORS; read once at startup, so deleted once healthy.
            import secrets

            self._api_key = secrets.token_urlsafe(32)
            key_argv, key_env, key_file = _llama_server_key_launch(self._api_key)
            cmd[1:1] = key_argv
            env.update(key_env)
        self._stdout_lines = []
        # No graceful-shutdown step covers this backend, so gate every spawn on the shutdown latch.
        if is_process_shutting_down():
            raise RuntimeError("Unsloth is shutting down; not starting the embed server")
        proc = subprocess.Popen(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            env = env,
            **windows_hidden_subprocess_kwargs(),
            **child_popen_kwargs(),
        )
        self._process = proc
        self._on_gpu = use_gpu
        # child_popen_kwargs() is empty on macOS, so the crash record is the only thing that can reap it
        # after a force quit.
        adopt_pid(proc.pid)
        # Recheck after recording the pid: the latch may have been set after the gate above.
        if is_process_shutting_down():
            logger.info("shutdown began during the spawn; killing the new embed server")
            self._kill_process()
            raise RuntimeError("Unsloth is shutting down; not starting the embed server")
        self._port = port
        self._stdout_thread = account_thread(
            target = self._drain_stdout,
            args = (proc,),
            daemon = True,
            name = "llama-embed-stdout",
        )
        self._stdout_thread.start()
        try:
            healthy = self._wait_for_health(config.EMBED_STARTUP_TIMEOUT_S)
        finally:
            if key_file is not None:
                try:
                    key_file.unlink()
                except OSError:
                    pass
        if not healthy:
            tail = "\n".join(self._stdout_lines[-30:])
            self._kill_process()
            raise RuntimeError(
                f"llama-server embedder failed to become healthy. Last output:\n{tail[:2000]}"
            )

    @staticmethod
    def _find_free_port() -> int:
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._find_free_port()

    def _wait_for_health(
        self,
        timeout: float,
        interval: float = 0.5,
    ) -> bool:
        """Poll /health until 200; bail early if the process exits."""
        deadline = time.monotonic() + timeout
        url = f"{self._base_url}/health"
        while time.monotonic() < deadline:
            if not self._process_alive():
                code = None if self._process is None else self._process.returncode
                logger.error("llama-server embedder exited early (code %s)", code)
                return False
            try:
                # trust_env=False: a proxy that 503s 127.0.0.1 must not block this probe.
                if httpx.get(url, timeout = 2.0, trust_env = False).status_code == 200:
                    return True
            except (*_TRANSPORT_ERRORS, httpx.TimeoutException):
                pass
            time.sleep(interval)
        logger.error("llama-server embedder health check timed out after %ss", timeout)
        return False

    def _process_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _current(self, model_name: str | None = None) -> bool:
        """Alive AND serving the repo for ``model_name`` (a Settings model change,
        or a request pinned to a different model, makes a live server stale)."""
        from utils.llama_cpp_path_settings import custom_llama_cpp_path_revision

        desired = config.effective_gguf_repo_for_embedding_model(
            model_name or config.effective_embedding_model()
        )
        return (
            self._process_alive()
            and self._model_repo == desired
            and self._binary_path_revision == custom_llama_cpp_path_revision()
        )

    def pooling_identity_snapshot(self) -> tuple[str | None, str | None, str | None, bool]:
        """One coherent path/repo/pooling/process snapshot for pre-encode identity prediction."""
        with self._serve_lock:
            return self._model_path, self._model_repo, self._model_pooling, self._process_alive()

    def _ensure_ready(self, model_name: str | None = None) -> None:
        """One subprocess serves one GGUF, so a request for another model respawns the server onto it."""
        if self._current(model_name):
            return
        with self._lifecycle_lock:
            if self._current(model_name):
                return
            self._kill_process()
            from utils.llama_cpp_path_settings import custom_llama_cpp_path_revision

            if self._binary_path_revision != custom_llama_cpp_path_revision():
                self._binary = None
                self._binary_path_revision = None
                # Must be cleared here: _resolve_model_path fast-paths on an unchanged repo.
                self._max_tokens = None
            self._spawn(model_name)

    def _restart(self, model_name: str | None = None) -> None:
        with self._lifecycle_lock:
            self._kill_process()
            self._spawn(model_name)

    @_invalidates_gpu_memory("embedding server stop")
    def _kill_process(self) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server embedder did not exit on SIGTERM; killing")
            proc.kill()
            try:
                proc.wait(timeout = 5)
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            logger.warning("error killing llama-server embedder: %s", e)
        finally:
            # A survivor must stay recorded so the next startup sweep can reap it.
            if proc.poll() is not None:
                forget_pid(proc.pid)
            self._process = None
            if self._stdout_thread is not None:
                self._stdout_thread.join(timeout = 2)
                self._stdout_thread = None

    def _shutdown(self) -> None:
        with self._operation_condition:
            self._closed = True
            while self._active_operations:
                self._operation_condition.wait()
        try:
            with self._lifecycle_lock:
                self._kill_process()
        finally:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            # Embedder may swap mid-call, so report the backend actually used (see encode_with_identity).
            try:
                atexit.unregister(self._shutdown)
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass

    def _post(
        self,
        path: str,
        payload: dict,
        model_name: str | None = None,
    ) -> dict:
        """POST to the server, restarting once and retrying on a dropped connection
        (the reaper may have killed us) or a timeout (the bundled build sometimes
        wedges a request); a fresh server unsticks both."""
        with self._operation():
            return self._post_active(path, payload, model_name)

    def _post_active(
        self,
        path: str,
        payload: dict,
        model_name: str | None = None,
    ) -> dict:
        last_exc: Exception | None = None
        for attempt in range(2):
            with self._serve_lock:
                # Held across readiness AND the request, so a swap can only land between whole requests.
                self._ensure_ready(model_name)
                try:
                    resp = self._client.post(
                        f"{self._base_url}{path}", json = payload, headers = self._auth_headers()
                    )
                    resp.raise_for_status()
                    return resp.json()
                except (*_TRANSPORT_ERRORS, httpx.TimeoutException) as e:
                    last_exc = e
                    crashed = not isinstance(e, httpx.TimeoutException) and self._server_exited()
                    if attempt == 0:
                        self._restart(model_name)
                        continue
                    # A second crash is the GPU build failing on this input, not the reaper (#13663).
                    if crashed and self._fall_back_to_cpu():
                        self._restart(model_name)
                        return self._post_active(path, payload, model_name)
                except httpx.HTTPStatusError as e:
                    body = e.response.text[:500] if e.response is not None else ""
                    raise RuntimeError(
                        f"llama-server embedder POST {path} -> {e.response.status_code}: {body}"
                    ) from e
        raise RuntimeError(f"llama-server embedder POST {path} failed after retry") from last_exc

    def _server_exited(self) -> bool:
        """Whether the server died under the request; logs its exit code and output tail."""
        proc = self._process
        if proc is None:
            return False
        try:
            code = proc.wait(timeout = 2)
        except subprocess.TimeoutExpired:
            return False
        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout = 2)
        tail = "\n".join(self._stdout_lines[-30:])
        logger.warning(
            "llama-server embedder exited during a request (code %s). Last output:\n%s",
            code,
            tail[:2000],
        )
        return True

    def _fall_back_to_cpu(self) -> bool:
        if not self._on_gpu or self._force_cpu or config.embed_device_requires_gpu():
            return False
        logger.warning("llama-server embedder crashed twice on the GPU; continuing on CPU")
        self._force_cpu = True
        return True

    def encode(
        self,
        texts,
        *,
        model_name = None,
        normalize = True,
    ):
        """Embed texts -> (N, dim) float32. ``model_name`` pins which GGUF serves
        the request, so a Settings change cannot answer it from another model.
        Normalizes in Python to match the ST backend."""
        self._served_identity_local.value = None
        with self._operation(), self._serve_lock:
            vectors = self._encode_active(texts, normalize = normalize, model_name = model_name)
            self._served_identity_local.value = (self._model_repo, self._model_pooling)
            return vectors

    def served_embedding_identity(self) -> tuple[str | None, str | None] | None:
        """Repo and pooling captured by this thread's last completed encode."""
        return getattr(self._served_identity_local, "value", None)

    def _encode_active(
        self,
        texts,
        *,
        normalize = True,
        model_name = None,
    ):
        n = len(texts)
        if n == 0:
            return np.zeros((0, self.dim(model_name = model_name)), dtype = np.float32)
        rows: list[list[float]] = []
        batch = max(1, config.EMBED_BATCH)
        for start in range(0, n, batch):
            chunk = list(texts[start : start + batch])
            data = self._post(
                "/v1/embeddings",
                {"input": chunk, "model": "embedding", "encoding_format": "float"},
                model_name = model_name,
            )
            items = data.get("data", [])
            if len(items) != len(chunk):
                raise RuntimeError(
                    f"embedder returned {len(items)} vectors for {len(chunk)} inputs"
                )
            # OpenAI spec lets the server reorder; sort back by index.
            items = sorted(items, key = lambda d: d.get("index", 0))
            rows.extend(d["embedding"] for d in items)
        arr = np.asarray(rows, dtype = np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f"embedder returned ragged vectors: shape {arr.shape}")
        if normalize:
            norms = np.linalg.norm(arr, axis = 1, keepdims = True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr

    def dim(self, *, model_name = None) -> int:
        """Under _serve_lock: _dim is per subprocess, so a model switch must not cache another
        model's width."""
        with self._operation(), self._serve_lock:
            self._ensure_ready(model_name)
            cached = self._dim
            if cached is not None:
                return cached
            vec = self.encode(["x"], normalize = False, model_name = model_name)
            width = int(vec.shape[1])
            self._dim = width
            return width

    def _server_props(self) -> dict | None:
        """Advisory only: any failure, including a malformed base URL on an unstarted server,
        returns None."""
        try:
            data = httpx.get(
                f"{self._base_url}/props",
                headers = self._auth_headers(),
                timeout = 2.0,
                trust_env = False,
            ).json()
        except Exception:  # noqa: BLE001 - see docstring
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _positive(data: dict | None, key: str) -> int | None:
        """*key* from /props, whether the build reports it at the top level or under
        ``default_generation_settings``."""
        for scope in (data, (data or {}).get("default_generation_settings")):
            if isinstance(scope, dict):
                value = scope.get(key)
                if isinstance(value, int) and value > 0:
                    return value
        return None

    def _server_context(self) -> int | None:
        return self._positive(self._server_props(), "n_ctx")

    def _server_batch(self) -> int | None:
        """Caps advertised input: a non-causal embedding prompt longer than the physical batch is
        refused."""
        # Effective physical batch is min(n_batch, n_ubatch), so only n_ubatch can limit us.
        found = self._positive(self._server_props(), "n_ubatch")
        return min(found, _UBATCH_SIZE) if found else _UBATCH_SIZE

    def max_tokens(self, *, model_name = None) -> int | None:
        with self._operation(), self._serve_lock:
            self._ensure_ready(model_name)
            if self._max_tokens is None and self._model_path:
                limit = _gguf_context_length(self._model_path)
                running = self._server_context()
                if limit and running:
                    limit = min(limit, running)
                elif running:
                    limit = running
                # Past one batch llama.cpp returns a 500, not the 400 this limit exists to give.
                batch = self._server_batch()
                if batch:
                    limit = min(limit, batch) if limit else batch
                if limit:
                    data = self._post(
                        "/tokenize",
                        {"content": "", "add_special": True},
                        model_name = model_name,
                    )
                    answer = max(1, limit - len(data.get("tokens", [])))
                    if running is None or batch is None:
                        # Do not cache the guess, or a too-large limit outlives a later /props readback.
                        return answer
                    self._max_tokens = answer
            return self._max_tokens

    def warm(self, *, model_name = None) -> None:
        """Readiness is left to dim(), which takes _serve_lock; warming here must not respawn outside it."""
        with self._operation():
            self.dim(model_name = model_name)

    def token_counter(self, *, model_name = None):
        """Count tokens via the GGUF's /tokenize so chunk sizing matches the
        embedder. Cached per text."""

        @lru_cache(maxsize = 4096)
        def _count(text: str) -> int:
            with self._operation():
                data = self._post(
                    "/tokenize",
                    {"content": text, "add_special": False},
                    model_name = model_name,
                )
                return len(data.get("tokens", []))

        return _count
