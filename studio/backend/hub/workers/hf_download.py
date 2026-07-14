# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HuggingFace Hub download worker, spawned as a subprocess so SIGKILL stops all chunk threads.

Resume safety
-------------
Downloads here MUST be single-stream sequential writers so the parent's
SIGKILL → restart loop can rely on ``os.path.getsize(.incomplete)`` to
compute the correct resume offset.

Enforced by:
- Setting ``HF_HUB_DISABLE_XET=1`` and ``HF_HUB_ENABLE_HF_TRANSFER=0`` on
  the spawning side (see :mod:`hub.utils.download_registry`) for transport=http.
- Passing ``max_workers=1`` to ``snapshot_download`` so files download
  serially, making the at-most-one-active-`.incomplete` invariant hold
  globally and simplifying reasoning about partial state during a SIGKILL.
- Letting ``prepare_cache_for_transport`` purge any pre-existing
  ``.incomplete`` blobs not provably from the same sequential writer.

If the final byte count doesn't match what HF declared, huggingface_hub
raises ``EnvironmentError`` ("Consistency check failed: …"); we surface
that on stderr so the watcher can show the exact message to the user.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from hub.utils.snapshot_filters import (
    SNAPSHOT_IGNORE_PATTERNS,
)
from hub.utils.gguf_plan import (
    GgufVariantPlan,
    build_gguf_variant_plans,
    plan_from_expected_files,
    sibling_sha256,
)
from hub.utils.state_dir import RepoType

HfTokenArg = str | bool | None


# Bound the metadata fetch so a stalled connection fails the worker (exit 1)
# instead of hanging at 0%. The file download itself is governed separately by
# huggingface_hub's own timeout.
_METADATA_REQUEST_TIMEOUT = 10.0
_METADATA_RETRY_TIMEOUT = 30.0
_METADATA_RETRY_DELAY = 1.0


def _on_signal(signum, frame):
    # 130 is what `classify_exit` maps to the "cancelled" job state.
    sys.exit(130)


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    sigpipe = getattr(signal, "SIGPIPE", None)
    if sigpipe is not None:
        signal.signal(sigpipe, _on_signal)


def _parent_poll_seconds() -> float:
    raw = os.environ.get("UNSLOTH_HF_WORKER_PARENT_POLL_SECONDS")
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return 2.0


def _protected_blob_hashes() -> frozenset[str]:
    """Blob hashes a concurrent same-repo peer is writing (passed by the backend
    as a plain env list). Excluded from this worker's purge so a shared
    ``.incomplete`` (e.g. a bundled mmproj) is never deleted under the peer."""
    raw = os.environ.get("UNSLOTH_PROTECTED_BLOB_HASHES", "")
    return frozenset(h for h in raw.split(",") if h)


def _parent_is_alive(parent_pid: int) -> bool:
    """Whether the recorded parent (the backend) is still running.

    Liveness ONLY: ``os.kill(pid, 0)`` on POSIX, an ``OpenProcess`` handle on
    Windows, against the *recorded* PID (never os.getppid(), so POSIX
    reparenting to init after the backend dies still resolves as dead). Probe
    ambiguity is treated as alive so a transient error never kills a healthy
    download.

    We deliberately do NOT compare psutil ``create_time()`` for PID-reuse
    detection: it isn't stable across reads on some platforms, so an exact match
    can spuriously kill a live download. PID-reuse after parent death is covered
    by the boot-time orphan reaper.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        SYNCHRONIZE = 0x00100000
        WAIT_OBJECT_0 = 0x0
        ERROR_INVALID_PARAMETER = 87
        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        ctypes.set_last_error(0)
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
        if not handle:
            return ctypes.get_last_error() != ERROR_INVALID_PARAMETER
        try:
            return kernel32.WaitForSingleObject(handle, 0) != WAIT_OBJECT_0
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _terminate_orphaned_self() -> None:
    # Hard exit from the watchdog thread: a self-SIGTERM would be deferred while
    # the main thread is GIL-blocked in a C socket read. The partial .incomplete
    # resumes byte-exact and marker/manifest writes are atomic, so cancelled code
    # 130 is safe. The diagnostic is best-effort: a dead parent's closed stderr
    # pipe can raise BrokenPipeError, which must never preempt the exit.
    try:
        print(
            "Parent process exited; stopping orphaned download worker.",
            file = sys.stderr,
        )
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(130)


def _install_parent_death_watchdog(parent_pid: int | None) -> None:
    if not parent_pid or parent_pid <= 0:
        return
    interval = _parent_poll_seconds()

    def _watch() -> None:
        while True:
            try:
                alive = _parent_is_alive(parent_pid)
            except Exception:
                alive = True
            if not alive:
                _terminate_orphaned_self()
                return
            time.sleep(interval)

    threading.Thread(
        target = _watch,
        name = "parent-death-watchdog",
        daemon = True,
    ).start()


def _hf_token_arg(hf_token: str | None) -> HfTokenArg:
    return hf_token if hf_token else False


def _retry_metadata_fetch(repo_id: str, fetch, *, label: str):
    for attempt, timeout in enumerate((_METADATA_REQUEST_TIMEOUT, _METADATA_RETRY_TIMEOUT)):
        try:
            return fetch(timeout)
        except Exception as e:
            if attempt == 1:
                raise
            print(
                f"{label} request failed for {repo_id} " f"({type(e).__name__}: {e}); retrying.",
                file = sys.stderr,
            )
            time.sleep(_METADATA_RETRY_DELAY)
    raise RuntimeError(f"{label} unavailable for {repo_id}")


def _model_info_with_retry(repo_id: str, hf_token: str | None):
    from huggingface_hub import model_info as hf_model_info
    return _retry_metadata_fetch(
        repo_id,
        lambda timeout: hf_model_info(
            repo_id,
            token = _hf_token_arg(hf_token),
            timeout = timeout,
            files_metadata = True,
        ),
        label = "Metadata",
    )


def _dataset_info_with_retry(repo_id: str, hf_token: str | None):
    from huggingface_hub import HfApi
    api = HfApi(token = _hf_token_arg(hf_token))
    return _retry_metadata_fetch(
        repo_id,
        lambda timeout: api.dataset_info(
            repo_id,
            timeout = timeout,
            files_metadata = True,
        ),
        label = "Dataset metadata",
    )


# Tied to drain_stderr_excerpt's 500-byte head/tail window in the parent (see
# hub/utils/download_registry.py): listing every expected file would blow past
# it and lose the diagnostic. Cap the preview so the summary line survives.
_VERIFY_PATH_LIST_CAP = 10


def _format_path_list(paths: tuple[str, ...], cap: int = _VERIFY_PATH_LIST_CAP) -> str:
    if len(paths) <= cap:
        return ", ".join(paths)
    head = ", ".join(paths[:cap])
    return f"{head}, ... and {len(paths) - cap} more"


def _verify_completed_download(
    repo_type: RepoType,
    repo_id: str,
    variant: str | None,
    snapshot_path: str,
    *,
    metadata_unavailable: bool = False,
) -> None:
    """Verify every manifest file is on disk at its declared size; exit nonzero
    with a diagnostic if not.

    No-op when no manifest exists: the manifest write is best-effort, so absence
    means "verification unavailable, trust snapshot_download's exit code".
    """
    from hub.utils import download_manifest

    manifest = download_manifest.read_manifest(repo_type, repo_id, variant)
    if manifest is None:
        return
    result = download_manifest.verify_against_disk(
        manifest,
        Path(snapshot_path),
    )
    if result.ok:
        return
    label = f"{repo_id}{f' [{variant}]' if variant else ''}"
    if metadata_unavailable:
        print(
            f"Could not reach Hugging Face for {label} and the copy on disk is "
            f"incomplete ({len(result.missing)} file(s) missing, "
            f"{len(result.size_mismatched)} the wrong size). Access to a private "
            "or restricted repo may have been lost (HF token removed or "
            "changed), the connection dropped, or Hugging Face is temporarily "
            "unavailable. Set a valid HF token or reconnect, then resume the "
            "download.",
            file = sys.stderr,
        )
    else:
        print(
            f"Verification failed for {label}: snapshot_download completed but "
            f"{len(result.missing)} expected file(s) are missing and "
            f"{len(result.size_mismatched)} have incorrect size on disk.",
            file = sys.stderr,
        )
    if result.missing:
        print(
            f"Missing: {_format_path_list(result.missing)}",
            file = sys.stderr,
        )
    if result.size_mismatched:
        print(
            f"Size mismatched: {_format_path_list(result.size_mismatched)}",
            file = sys.stderr,
        )
    sys.exit(1)


def _preflight_disk_space(repo_type: str, repo_id: str, expected_files: list) -> None:
    """Fail fast when the active HF cache filesystem can't hold what's left to
    download. Fail-open: any inability to size the work or read free space skips
    the check, so a real download is never blocked by an estimation gap."""
    import shutil

    from hub.utils.download_registry import existing_blob_bytes
    from hub.utils.hf_cache_state import hf_cache_root

    try:
        size_by_hash: dict[str, int] = {}
        unhashed_bytes = 0
        for expected in expected_files:
            size = int(getattr(expected, "size", 0) or 0)
            if size <= 0:
                continue
            blob_hash = getattr(expected, "sha256", None)
            if blob_hash:
                # Dedup by content hash: a blob listed under two filenames is
                # written once, so it must be counted once.
                size_by_hash[blob_hash] = size
            else:
                unhashed_bytes += size
        total_expected = sum(size_by_hash.values()) + unhashed_bytes
        if total_expected <= 0:
            return
        already_have = existing_blob_bytes(
            repo_type,
            repo_id,
            frozenset(size_by_hash),
        )
        remaining = max(0, total_expected - already_have)
        if remaining <= 0:
            return
        root = hf_cache_root(create = True)
        if root is None:
            return
        free = shutil.disk_usage(root).free
    except Exception:
        return

    if free < remaining:
        print(
            f"Not enough disk space to download {repo_id}: need about "
            f"{remaining / (1024 ** 3):.1f} GB free in {root}, but only "
            f"{free / (1024 ** 3):.1f} GB is available. Free up space and "
            "try again.",
            file = sys.stderr,
        )
        sys.exit(1)


def _snapshot_download_plan(info) -> tuple[list[str], list]:
    from hub.utils.download_manifest import ExpectedFile
    from hub.utils.snapshot_filters import (
        resolve_snapshot_ignore_patterns_for_files,
        snapshot_download_siblings,
    )

    filenames = [s.rfilename for s in info.siblings if isinstance(s.rfilename, str)]
    filtered = snapshot_download_siblings(info.siblings)
    expected_files = [
        ExpectedFile(
            path = s.rfilename,
            size = int(getattr(s, "size", 0) or 0),
            sha256 = sibling_sha256(s),
        )
        for s in filtered
        if isinstance(s.rfilename, str)
    ]
    return resolve_snapshot_ignore_patterns_for_files(filenames), expected_files


def _dataset_expected_files(info) -> list:
    from hub.utils.download_manifest import ExpectedFile
    return [
        ExpectedFile(
            path = s.rfilename,
            size = int(getattr(s, "size", 0) or 0),
            sha256 = sibling_sha256(s),
        )
        for s in info.siblings
        if isinstance(s.rfilename, str)
    ]


def _recover_manifest_after_download(
    repo_type: RepoType,
    repo_id: str,
    snapshot_path: str,
    mode: str,
    *,
    fetch_info,
    expected_files_from_info,
    label: str = "",
) -> None:
    """Best-effort manifest write for a download whose metadata was unavailable
    at start: re-fetch and record the expected files, else fall back to the
    on-disk file list. Shared by the model and dataset workers.

    A pre-existing manifest is authoritative and is preserved untouched. This is
    load-bearing: when access is lost on resume (token revoked/changed on a
    gated/private repo), snapshot_download returns the cached partial snapshot
    WITHOUT downloading, so rebuilding the manifest from on-disk files would record
    the partial set as expected and let _verify_completed_download certify a
    half-finished download as complete.

    The same hazard exists with NO prior manifest. When metadata is still
    unavailable here, leftover ``.incomplete`` blobs prove a cached partial was
    returned without downloading, so we fail (exit 1) instead of deriving a
    self-certifying manifest, leaving the partial intact for a later resume. That
    signal misses a file that never started (no ``.incomplete``), so a kill
    between files is accepted optimistically from the on-disk subset; a later
    metadata-bearing attempt writes the true manifest and catches any shortfall."""
    from hub.utils import download_manifest
    from hub.utils.hf_cache_state import has_active_incomplete_blobs

    if download_manifest.read_manifest(repo_type, repo_id, None) is not None:
        return

    try:
        if download_manifest.write_manifest(
            repo_type,
            repo_id,
            None,
            expected_files_from_info(fetch_info()),
            mode,
        ):
            return
        reason = "manifest write failed"
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"

    if has_active_incomplete_blobs(repo_type, repo_id):
        print(
            f"{label}could not reach Hugging Face for {repo_id} and the copy on "
            "disk is still incomplete. Access to a private or restricted repo may "
            "have been lost (HF token removed or changed), the connection dropped, "
            "or Hugging Face is temporarily unavailable. Set a valid HF token or "
            "reconnect, then resume the download.",
            file = sys.stderr,
        )
        sys.exit(1)

    fallback_files = download_manifest.expected_files_from_snapshot_dir(Path(snapshot_path))
    if fallback_files and download_manifest.write_manifest(
        repo_type,
        repo_id,
        None,
        fallback_files,
        mode,
    ):
        print(
            f"{label}could not record the metadata manifest for {repo_id}, "
            "recorded one from the downloaded files so completion "
            f"is tracked ({reason})",
            file = sys.stderr,
        )
    else:
        print(
            f"{label}could not record the metadata manifest for {repo_id}, "
            f"{download_manifest.MANIFEST_DEGRADED_MARKER} ({reason})",
            file = sys.stderr,
        )


def _download_snapshot(repo_id: str, hf_token: str | None, mode: str) -> None:
    from huggingface_hub import snapshot_download
    from hub.utils.download_registry import prepare_cache_for_transport
    from hub.utils import download_manifest

    # One metadata fetch powers both the ignore-pattern decision (drop
    # consolidated.* when transformers weights exist) and the manifest's
    # expected_files. A failure is non-fatal: fall back to the legacy
    # ignore-pattern set (keeping consolidated) and skip the manifest, so
    # download proceeds without the verification + partial detection it enables.
    try:
        info = _model_info_with_retry(repo_id, hf_token)
    except Exception as e:
        print(
            f"metadata unavailable, downloading full snapshot for {repo_id} "
            f"({type(e).__name__}: {e})",
            file = sys.stderr,
        )
        info = None

    download_manifest.clear_cancel_marker("model", repo_id, None)
    if info is not None:
        ignore_patterns, expected_files = _snapshot_download_plan(info)
        # Written for every transport. The manifest verifies the finalized
        # files under snapshots/, which both transports produce identically
        # (XET also renames a full, correctly-sized blob into place). XET's
        # block-level dedup lives only in the chunk-cache the manifest never
        # inspects, so per-file size verification is valid regardless of transport.
        download_manifest.write_manifest("model", repo_id, None, expected_files, mode)
    else:
        ignore_patterns = list(SNAPSHOT_IGNORE_PATTERNS)
        expected_files = []

    purged = prepare_cache_for_transport("model", repo_id, mode)
    if purged:
        print(
            f"Purged {purged} untrusted partial blob(s) for {repo_id} "
            f"before starting {mode} download.",
            file = sys.stderr,
        )
    _preflight_disk_space("model", repo_id, expected_files)
    snapshot_path = snapshot_download(
        repo_id = repo_id,
        token = _hf_token_arg(hf_token),
        ignore_patterns = ignore_patterns,
        max_workers = 1,
    )
    if info is None:
        _recover_manifest_after_download(
            "model",
            repo_id,
            snapshot_path,
            mode,
            fetch_info = lambda: _model_info_with_retry(repo_id, hf_token),
            expected_files_from_info = lambda recovered: _snapshot_download_plan(recovered)[1],
        )
    _verify_completed_download(
        "model",
        repo_id,
        None,
        snapshot_path,
        metadata_unavailable = info is None,
    )


def _gguf_variant_target_plan(
    repo_id: str, variant: str, hf_token: str | None
) -> GgufVariantPlan | None:
    try:
        info = _model_info_with_retry(repo_id, hf_token)
    except Exception as e:
        print(
            f"metadata unavailable, cannot resolve GGUF variant '{variant}' "
            f"for {repo_id} ({type(e).__name__}: {e})",
            file = sys.stderr,
        )
        raise RuntimeError(
            f"Metadata unavailable while resolving GGUF variant '{variant}' " f"for {repo_id}"
        ) from e
    return build_gguf_variant_plans(list(info.siblings)).get(variant.lower())


def _download_gguf_variant(repo_id: str, variant: str, hf_token: str | None, mode: str) -> None:
    from huggingface_hub import snapshot_download
    from hub.utils.download_registry import prepare_cache_for_transport
    from hub.utils.hf_cache_state import has_active_incomplete_blobs
    from hub.utils import download_manifest

    metadata_unavailable = False
    try:
        plan = _gguf_variant_target_plan(repo_id, variant, hf_token)
    except RuntimeError:
        plan = None
        metadata_unavailable = True

    if not metadata_unavailable:
        if plan is None:
            print(
                f"No GGUF shards matching variant '{variant}' in {repo_id}",
                file = sys.stderr,
            )
            sys.exit(1)
        targets = list(plan.target_filenames)
        expected_files = list(plan.expected_files)
        main_blob_hashes = plan.main_hashes
        companion_blob_hashes = plan.companion_hashes
        download_manifest.write_manifest(
            "model",
            repo_id,
            variant,
            expected_files,
            mode,
        )
    else:
        # Metadata unreachable (offline / gated / private). Resume the exact
        # shards the original attempt recorded so snapshot_download can range
        # over the surviving .incomplete blobs without a model_info call.
        manifest = download_manifest.read_manifest("model", repo_id, variant)
        if manifest is None or not manifest.expected_files:
            print(
                f"Metadata unavailable and no manifest to resume GGUF "
                f"variant '{variant}' for {repo_id}",
                file = sys.stderr,
            )
            sys.exit(1)
        plan = plan_from_expected_files(variant, manifest.expected_files)
        targets = list(plan.target_filenames)
        expected_files = list(plan.expected_files)
        download_manifest.write_manifest(
            "model",
            repo_id,
            variant,
            expected_files,
            mode,
        )
        main_blob_hashes = plan.main_hashes
        companion_blob_hashes = plan.companion_hashes
        print(
            f"Metadata unavailable; resuming GGUF variant '{variant}' for "
            f"{repo_id} from the existing manifest.",
            file = sys.stderr,
        )

    download_manifest.clear_cancel_marker("model", repo_id, variant)
    purge_blob_hashes = main_blob_hashes
    if not main_blob_hashes:
        if has_active_incomplete_blobs("model", repo_id):
            print(
                f"GGUF variant '{variant}' for {repo_id} has partial cache state "
                "but no resolvable blob hashes; delete the partial download or "
                "retry when metadata is available.",
                file = sys.stderr,
            )
            sys.exit(1)
        purge_blob_hashes = frozenset()
        print(
            f"GGUF variant '{variant}' for {repo_id} has no resolvable blob "
            "hashes; starting without partial cache reuse.",
            file = sys.stderr,
        )
    # Main quant blobs are owned by this variant (variant-scoped marker). The
    # shared vision companion (mmproj) is judged by a separate companion marker
    # and never purged while a concurrent peer is writing it.
    purged = prepare_cache_for_transport(
        "model",
        repo_id,
        mode,
        variant,
        only_blob_hashes = purge_blob_hashes,
        companion_blob_hashes = companion_blob_hashes,
        protected_blob_hashes = _protected_blob_hashes(),
    )
    if purged:
        print(
            f"Purged {purged} untrusted partial blob(s) for {repo_id} "
            f"before starting {mode} download.",
            file = sys.stderr,
        )
    _preflight_disk_space("model", repo_id, expected_files)
    snapshot_path = snapshot_download(
        repo_id = repo_id,
        token = _hf_token_arg(hf_token),
        allow_patterns = targets,
        max_workers = 1,
    )
    _verify_completed_download(
        "model",
        repo_id,
        variant,
        snapshot_path,
        metadata_unavailable = metadata_unavailable,
    )
    if plan is not None:
        try:
            from hub.services.models.deletion import reclaim_replaced_gguf_variant
            reclaim_replaced_gguf_variant(
                repo_id,
                variant,
                plan.main_hashes,
                hf_token,
            )
        except Exception as e:
            print(
                f"Verified GGUF update for {repo_id} [{variant}], but stale-cache "
                f"reclaim failed ({type(e).__name__}: {e})",
                file = sys.stderr,
            )


def _download_dataset(repo_id: str, hf_token: str | None, mode: str) -> None:
    from huggingface_hub import snapshot_download
    from hub.utils.download_registry import prepare_cache_for_transport
    from hub.utils import download_manifest

    try:
        info = _dataset_info_with_retry(repo_id, hf_token)
    except Exception as e:
        print(
            f"dataset metadata unavailable, downloading full dataset for {repo_id} "
            f"({type(e).__name__}: {e})",
            file = sys.stderr,
        )
        info = None
    # Cancel-marker clear and manifest write run on every transport. See
    # _download_snapshot for why per-file size verification is valid under XET.
    download_manifest.clear_cancel_marker("dataset", repo_id, None)
    if info is not None:
        expected_files = _dataset_expected_files(info)
        download_manifest.write_manifest(
            "dataset",
            repo_id,
            None,
            expected_files,
            mode,
        )
    else:
        expected_files = []
    purged = prepare_cache_for_transport("dataset", repo_id, mode)
    if purged:
        print(
            f"Purged {purged} untrusted partial blob(s) for {repo_id} "
            f"before starting {mode} download.",
            file = sys.stderr,
        )
    _preflight_disk_space("dataset", repo_id, expected_files)
    snapshot_path = snapshot_download(
        repo_id = repo_id,
        token = _hf_token_arg(hf_token),
        repo_type = "dataset",
        max_workers = 1,
    )
    if info is None:
        _recover_manifest_after_download(
            "dataset",
            repo_id,
            snapshot_path,
            mode,
            fetch_info = lambda: _dataset_info_with_retry(repo_id, hf_token),
            expected_files_from_info = _dataset_expected_files,
            label = "dataset ",
        )
    _verify_completed_download(
        "dataset",
        repo_id,
        None,
        snapshot_path,
        metadata_unavailable = info is None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description = "HuggingFace Hub download worker")
    parser.add_argument("--repo-id", required = True)
    parser.add_argument("--variant", default = None)
    parser.add_argument("--dataset", action = "store_true")
    parser.add_argument("--transport", choices = ("http", "xet"), default = "http")
    parser.add_argument("--parent-pid", type = int, default = None)
    args = parser.parse_args()

    _install_signal_handlers()
    _install_parent_death_watchdog(args.parent_pid)

    hf_token = os.environ.get("HF_TOKEN") or None

    try:
        if args.dataset:
            _download_dataset(args.repo_id, hf_token, args.transport)
        elif args.variant:
            _download_gguf_variant(args.repo_id, args.variant, hf_token, args.transport)
        else:
            _download_snapshot(args.repo_id, hf_token, args.transport)
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        # Surface a precise message so the UI doesn't show a generic "worker
        # exited with code 1". huggingface_hub's consistency check recommends
        # force_download=True to recover, which our "Restart" UI maps to a fresh
        # start by purging the partial via prepare_cache_for_transport.
        print(f"{type(e).__name__}: {e}", file = sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
