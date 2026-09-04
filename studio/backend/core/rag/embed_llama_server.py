# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF embedder over the bundled llama.cpp, served via HTTP (no torch).

Opt-in (``RAG_EMBED_BACKEND=llama-server``). Runs a dedicated
``llama-server --embedding`` subprocess on its own port and calls its OpenAI-style
``/v1/embeddings`` + ``/tokenize``, fully isolated from the chat backend.

Device is ``auto`` (GPU when present, else CPU, falling back to CPU if a GPU start
fails); ``RAG_EMBED_DEVICE`` forces it. We call only llama_cpp's *static* helpers
(no torch), copying the instance-coupled bits locally, since constructing a
``LlamaCppBackend`` runs an ``__init__`` reaper that kills any Unsloth llama-server
-- so each request re-spawns ours if it died (self-heal).
"""

from __future__ import annotations

import atexit
import logging
import os
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
from utils.process_lifetime import adopt_pid, child_popen_kwargs, forget_pid

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


def _resolve_entrypoint(binary: str) -> str:
    """The real executable behind a managed shell/symlink entrypoint.

    macOS only matters here: SIP purges DYLD_* while starting the protected
    /bin/sh a wrapper runs under, so probing or launching the wrapper loses the
    loader path however carefully the environment was built (#8566). Shared
    with the chat backend, which restricts this to OUR entrypoint so a user's
    own wrapper keeps whatever setup it does before its exec.
    """
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
    """``env`` with ``lib_dir`` first on the macOS loader search path.

    Returns a new dict rather than editing in place: the caller's environment is
    its own, and the chat backend's equivalent is a pure function too. Shares
    that one's prepend so both dedupe the same way.
    """
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
        # Unload waits for every encode/tokenize/probe already in flight and blocks late callers from
        # restarting after unpublish.
        self._operation_condition = threading.Condition()
        self._active_operations = 0
        self._operation_local = threading.local()
        self._closed = False
        self._process: subprocess.Popen | None = None
        self._port: int | None = None
        self._stdout_lines: list[str] = []
        self._stdout_thread: threading.Thread | None = None
        # dim() takes the same _serve_lock the request path does; reentrant because dim() -> encode() ->
        # _ensure_ready() re-enters on one thread.
        self._dim: int | None = None
        # One subprocess serves one GGUF, so readiness and the request must be indivisible: a swap between
        # them lands A's POST on B's server.
        self._serve_lock = threading.RLock()
        self._model_path: str | None = None
        # A Settings change makes the cached path/dim stale, forcing a re-resolve and respawn (see _ensure_ready).
        self._model_repo: str | None = None
        self._binary: str | None = None
        self._binary_path_revision: int | None = None
        # Sticky after an auto GPU start fails: later spawns stay on CPU.
        self._force_cpu = False
        # trust_env=False skips HTTP(S)_PROXY; full URLs per request survive a respawn.
        self._client = httpx.Client(timeout = config.EMBED_REQUEST_TIMEOUT_S, trust_env = False)
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
        """`llama-server --help`, cached. Ignore exit code (some builds exit
        non-zero on --help).

        On macOS, runs under the same loader environment as the real launch.
        Without it, a bundle that needs the search path dies in the loader
        here, and its error text reads as help output with no ``--embedding``
        in it, so the caller reports a build that lacks embeddings instead of a
        load failure. Elsewhere the probe keeps inheriting this process's
        environment exactly as it always has: the loader hole is macOS-only,
        and a probe that answers "does this build do embeddings" has no reason
        to run under a different environment than it did before.
        """
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

        # Normalize like the sentence-transformers and settings probes, else a WSL user's C:\models\... GGUF
        # resolves as an impossible filename and goes to the Hub.
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
                # llama-server opens the siblings implicitly, so a torn family is not a usable answer even with
                # shard 1 present.
                if colocated_split_shards(picked)[1]:
                    # Enters the family at shard 1 and keeps a complete symlink set intact, the same way the
                    # chat loader resolves a local GGUF.
                    return str(_local_gguf_load_path(picked))
                remaining = [f for f in remaining if f != picked]
            raise RuntimeError(f"no .gguf file found in local model dir {model!r}")
        return None

    @classmethod
    def _pick_complete_gguf(cls, names):
        """``(picked, family)`` for the preferred GGUF whose family is whole.

        A torn family is skipped rather than returned, so the candidate loop moves
        on to the next repo instead of adopting a repo it cannot finish.
        """
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
        """Every file `picked` needs, or None when the published family is torn.

        llama-server opens split siblings implicitly, so one selected shard is not a
        downloadable plan. Shared with the settings resolver so the plan it offers and
        the transfer this loader performs name the same set of files.
        """
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
        """GGUF pick from `names`, by variant then shortest name; `key` reads the comparable
        name off each entry.

        `require_variant` refuses the fallback to another variant. Falling back on a hub
        listing means the variant is not published; falling back on a cache would serve
        whatever happened to be fetched under an earlier setting."""
        from utils.models.model_config import _is_mtp_drafter

        name_of = key or (lambda n: n)
        usable = [
            n
            for n in names
            if name_of(n).lower().endswith(".gguf")
            and "mmproj" not in name_of(n).lower()
            # A drafter is a companion, so exclude it everywhere mmproj is: a cache holding only the companion
            # would read as holding the embedder.
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
        """The snapshot ``refs/main`` names in the active hub cache, or None.

        Only that revision, because it is the one hf_hub_download serves; the remaining
        snapshot directories are commit hashes, which order by nothing, so choosing among
        them could serve a superseded model.

        A hit therefore pins the embedder to the cached revision until the cache itself
        changes. Deliberate: a stored embedding identity records no revision, so adopting
        republished weights would leave an index answering one model's queries with another
        model's documents."""
        from utils.hf_cache_settings import active_hf_hub_cache
        from utils.paths import resolve_cached_repo_id_case

        # A repo id typed with different casing than the cache folder is still that repo; missing it re-
        # downloads or fails outright offline.
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
        """The GGUF this repo already has on disk, found without the network.

        ``_model_path`` is per-process, so without this every restart re-lists the repo to
        name a file it already holds -- unbounded on a host whose route to the hub
        blackholes."""
        from utils.models.model_config import colocated_split_shards

        snapshot = LlamaServerBackend._cached_snapshot_dir(repo_id)
        if snapshot is None:
            return None
        try:
            # Not a "*.gguf" glob: that is case-sensitive and misses quant-per-directory layouts, both of which
            # the hub listing handles.

            files = [
                p for p in snapshot.rglob("*") if p.is_file() and not is_appledouble_metadata(p)
            ]
        except OSError:
            return None
        # llama-server opens sibling shards implicitly, so a split set is servable only when whole; drop
        # an unservable winner and retry, else an incomplete family shadows a complete one.
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

    def _resolve_model_path(self, model_name: str | None = None) -> str:
        """Download (or cache-hit) the variant-matching, non-mmproj GGUF embedder,
        returning its local path. Re-resolves when the effective repo changed (a
        custom model was saved in Settings).

        ``model_name`` is the model the caller pinned; the live setting would
        resolve B's weights for a job still tagging its vectors as A."""
        model = model_name or config.effective_embedding_model()
        # Captured once: the path must stay tagged with the repo it was resolved FOR, so a mid-download
        # setting change reads as stale and respawns.
        desired = config.effective_gguf_repo_for_embedding_model(model)
        if self._model_path is not None and self._model_repo == desired:
            return self._model_path
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
                    # The repo publishes no such variant, so this quant is what the picker fetched: retire the marker
                    # here too.
                    try:
                        from utils.embedding_model_settings import clear_stored_download_pending
                        clear_stored_download_pending(model)
                    except Exception:  # noqa: BLE001 - a settings write must not fail a load
                        pass
                # Serve a stand-in rather than fail the index, but leave the marker: retiring on it would pin this
                # model to the wrong quant.
                return self._adopt_model_path(cached, desired)
            raise RuntimeError(
                f"Embedding model {model!r} is not downloaded yet. "
                "Finish its Settings download before indexing documents."
            )
        return self._resolve_uncached_model_path(desired, candidates)

    @staticmethod
    def _planned_family_path(model: str, repo_id: str) -> str | None:
        """The entry file of ``model``'s planned family, when all of it is cached.

        Whole family or nothing: a partially present one is a torn transfer, and
        answering with its first shard would serve llama-server a model it cannot
        finish opening. None when no family was recorded, which is every
        resolution written before it was stored."""
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
        """Whether ``path`` is one of the files the picker planned for ``model``.

        Compared by the path relative to the snapshot, which is the layout the
        record's repo-relative names already describe. Base names are not enough:
        a repo that files each quant in its own directory publishes
        ``Q8_0/model.gguf`` beside ``Q4_K_M/model.gguf``, and matching on the name
        alone would let a stale quant pass for the planned one.

        False when no family was recorded, which is every resolution written before
        it was stored, so the caller keeps its conservative answer on those."""
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
        self._dim = None
        return self._model_path

    # A listing, not a transfer: a working hub answers well inside a second.
    _LIST_DEADLINE_S = 20.0

    def _resolve_uncached_model_path(self, desired: str, candidates: list[str]) -> str:
        """Resolve the GGUF the local cache could not answer for.

        The listing needs the deadline because nothing else caps it: `list_repo_files` takes
        no timeout, and the pagination layer passes an explicit `timeout=None` that overrides
        any client-level default. The guard covers the transfer only by refusing to start one
        against an endpoint already known to be unreachable; once begun it is bounded per
        read, so a blob host that stalls mid-download still holds the lock."""
        from huggingface_hub import hf_hub_download, list_repo_files
        from core.inference.llama_cpp import _hf_offline_if_unreachable
        from utils.utils import call_with_deadline

        token = os.environ.get("HF_TOKEN") or None
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
        """``RAG_EMBED_DEVICE``: ``gpu``/``cpu`` force it; ``auto`` uses a GPU when
        present. A sticky CPU fallback (after a GPU start we were allowed to give up
        on) wins, so the reaper's next restart does not pay the startup timeout again
        on a path already known not to start. Only the literal ``gpu`` outranks it,
        and that spelling never sets the flag."""
        dev = config.embed_device_preference()
        if dev == "gpu" and not self._force_cpu:
            return True
        if dev == "cpu" or self._force_cpu:
            return False
        return self._gpu_available()

    @staticmethod
    def _gpu_available() -> bool:
        """Apple Metal, or an NVIDIA/ROCm GPU with enough free VRAM. Reuses
        llama_cpp's static probe, which asks an smi tool before torch, so the
        common path leaves no CUDA/HIP context behind. This backend IS
        llama-server, so it opts into the ROCm arch gate: an uncovered device
        crashes the embedding server as it does a chat load (#7624)."""
        from utils.hardware import is_apple_silicon

        if is_apple_silicon():
            return True
        from core.inference.llama_cpp import LlamaCppBackend

        # [(idx, free_mib)], honors CVD
        gpus = LlamaCppBackend._get_gpu_free_memory(for_llama_server = True)
        return any(free >= LlamaServerBackend._MIN_GPU_FREE_MIB for _, free in gpus)

    @staticmethod
    def _arch_gated_gpu_ids(binary: str) -> list[int]:
        """GPU ids to pin the embedding child to, or [] when it needs no mask.

        Knowing a supported device exists is not enough: the child enumerates every
        ROCm agent, and that HSA enumeration is what dies on an uncovered GPU (#7624),
        so on a mixed host the gate passes on the dGPU and the server still crashes on
        the iGPU. Pin the survivors instead. Empty unless the gate is both known and
        actually narrowing: NVIDIA, CPU, Vulkan and macOS have no mapped_targets
        marker, and a build covering every card needs no pin."""
        from core.inference.llama_cpp import LlamaCppBackend
        return LlamaCppBackend._arch_gate_survivors(binary)

    def _build_cmd(self, binary: str, model_path: str, port: int, *, use_gpu: bool) -> list[str]:
        # No --embd-normalize (absent in some builds; we normalize in Python to match the ST path), and
        # --fit off so ctx/offload are not auto-resized.
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
            "cls",
            "--fit",
            "off",
        ]
        # -1 offloads every layer (matches the chat server); 0 keeps it on CPU.
        cmd += ["-ngl", "-1" if use_gpu else "0"]
        return cmd

    def _build_env(self, binary: str, *, use_gpu: bool) -> dict[str, str]:
        env = child_env_without_native_path_secret()
        env["LLAMA_SET_ROWS"] = "1"
        if sys.platform == "darwin":
            # _llama_lib_dir, not Path(binary).parent: the managed install puts an entrypoint in front of the
            # real server and the dylibs sit next to the target (#8566).
            env = _with_dyld_path(env, _binary_lib_dir(binary))
        elif use_gpu:
            # Left as Path(binary).parent: resolving the entrypoint here would move the first LD_LIBRARY_PATH
            # entry for every existing Linux GPU install whose llama-server is a symlink.
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
            # HIP reads CUDA_VISIBLE_DEVICES only when HIP_VISIBLE_DEVICES is unset, so an inherited HIP mask
            # keeps a device (and the ~0.5 GB context it costs) visible to a child we just put on the CPU; an
            # inherited ROCR mask is left alone, since clearing it exposes MORE agents to the HSA enumeration
            # that dies on an uncovered arch (#7624).
            env["HIP_VISIBLE_DEVICES"] = "-1"
            # LLAMA_ARG_DEVICE / LLAMA_ARG_MAIN_GPU are the env spelling of --device / --main-gpu: hiding
            # every device while leaving that pick in place makes llama.cpp reject the name that no longer
            # enumerates and exit instead of running on the CPU.
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
        for pattern in (
            os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "cu*", "lib"),
            os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "cudnn", "lib"),
        ):
            lib_dirs.extend(d for d in glob.glob(pattern) if os.path.isdir(d))
        for cuda_lib in (
            "/usr/local/cuda/lib64",
            f"/usr/local/cuda/targets/{arch}-linux/lib",
            "/usr/local/cuda-12/lib64",
        ):
            if os.path.isdir(cuda_lib):
                lib_dirs.append(cuda_lib)
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

    def _spawn(self, model_name: str | None = None) -> None:
        """Start the embed server (caller holds the lock). A failed GPU start falls
        back to CPU once, unless ``RAG_EMBED_DEVICE`` is the literal ``gpu``."""
        use_gpu = self._use_gpu()
        try:
            self._spawn_once(use_gpu, model_name)
        except RuntimeError:
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
        port = config.EMBED_PORT or self._find_free_port()
        env = self._build_env(binary, use_gpu = use_gpu)
        cmd = self._build_cmd(binary, model_path, port, use_gpu = use_gpu)
        logger.info(
            "starting llama-server embedder (%s): %s",
            "gpu" if use_gpu else "cpu",
            " ".join(cmd),
        )
        self._stdout_lines = []
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
        # child_popen_kwargs() is empty on macOS, so the crash record is the only thing that can reap it
        # after a force quit.
        adopt_pid(proc.pid)
        self._port = port
        self._stdout_thread = threading.Thread(
            target = self._drain_stdout,
            args = (proc,),
            daemon = True,
            name = "llama-embed-stdout",
        )
        self._stdout_thread.start()
        if not self._wait_for_health(config.EMBED_STARTUP_TIMEOUT_S):
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

    def _ensure_ready(self, model_name: str | None = None) -> None:
        """Guarantee a live server on ``model_name``, (re)spawning if needed.
        Double-checked so the current path takes no lock; self-heals after the
        chat reaper kills us and re-resolves after a Settings model change.

        One subprocess serves one GGUF, so a request pinned to a model the server
        is not serving respawns onto it rather than answering from the wrong
        weights."""
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
            self._spawn(model_name)

    def _restart(self, model_name: str | None = None) -> None:
        with self._lifecycle_lock:
            self._kill_process()
            self._spawn(model_name)

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
            # Unload/rebuild is an ordinary cycle here, and a registry that only grows pins every retired
            # backend, and its captured stdout tail, for the life of the process.
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
                    resp = self._client.post(f"{self._base_url}{path}", json = payload)
                    resp.raise_for_status()
                    return resp.json()
                except (*_TRANSPORT_ERRORS, httpx.TimeoutException) as e:
                    last_exc = e
                    if attempt == 0:
                        self._restart(model_name)
                        continue
                except httpx.HTTPStatusError as e:
                    body = e.response.text[:500] if e.response is not None else ""
                    raise RuntimeError(
                        f"llama-server embedder POST {path} -> {e.response.status_code}: {body}"
                    ) from e
        raise RuntimeError(f"llama-server embedder POST {path} failed after retry") from last_exc

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
        with self._operation():
            return self._encode_active(texts, normalize = normalize, model_name = model_name)

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
        """Embedding width, probed via a 1-text encode and cached per model
        (_resolve_model_path clears it when the effective repo changes).

        Under the same lock the request path uses, for the same reason: ``_dim``
        belongs to the one subprocess, so with two jobs pinned to models of
        different width, one could ready A, have the other switch to B and cache
        B's width, and then answer A with it. Reentrant, so the probe's own encode
        re-enters rather than deadlocking."""
        with self._operation(), self._serve_lock:
            self._ensure_ready(model_name)
            cached = self._dim
            if cached is not None:
                return cached
            vec = self.encode(["x"], normalize = False, model_name = model_name)
            width = int(vec.shape[1])
            self._dim = width
            return width

    def warm(self, *, model_name = None) -> None:
        """Start the server and probe dim off the request path.

        Readiness is left to ``dim``, which takes ``_serve_lock`` for it. Calling it
        here as well put a kill-and-respawn outside that lock, so warming B could
        move the port out from under an encode pinned to A that had already passed
        its own readiness check, and store B's vectors under A's identity."""
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
