# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HuggingFace Hub download worker, spawned as a subprocess so SIGKILL stops all chunk threads.

Resume safety
-------------
The downloads done here MUST be single-stream sequential writers so the
parent process's SIGKILL → restart loop can rely on
``os.path.getsize(.incomplete)`` to compute the correct resume offset.

We enforce that by:
- Setting ``HF_HUB_DISABLE_XET=1`` and ``HF_HUB_ENABLE_HF_TRANSFER=0`` on
  the spawning side (see :mod:`hub.utils.download_registry`) when transport=http.
- Passing ``max_workers=1`` to ``snapshot_download`` so multiple files
  download serially. (Per-file safety is sequential regardless; setting
  ``max_workers=1`` makes the at-most-one-active-`.incomplete` invariant
  hold globally, which simplifies reasoning about partial state during a
  SIGKILL.)
- Letting ``prepare_cache_for_transport`` purge any pre-existing
  ``.incomplete`` blobs that can't be proven to come from the same
  sequential writer.

If the download finishes but the final byte count doesn't match what HF
declared, huggingface_hub raises ``EnvironmentError`` ("Consistency check
failed: …"). We surface that on stderr so the watcher thread can show
the exact message back to the user.
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


# Bound the variant-resolution metadata fetch so a stalled connection fails the
# worker (exit 1) instead of hanging at 0% until the user cancels. The file
# download itself is governed separately by huggingface_hub's own timeout.
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
    """Blob hashes a concurrent same-repo peer is writing (passed by the
    backend). Excluded from this worker's cache-preparation purge so a shared
    ``.incomplete`` (e.g. a bundled mmproj) is never deleted out from under the
    peer. Platform-agnostic: a plain env list, no filesystem locks."""
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
    detection: it is not stable across reads on some platforms, so an
    exact-equality check can spuriously kill a live download. PID-reuse after
    parent death is covered by the boot-time orphan reaper.
    """
    if sys.platform == "win32":
        import ctypes

        SYNCHRONIZE = 0x00100000
        WAIT_OBJECT_0 = 0x0
        ERROR_INVALID_PARAMETER = 87
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
        if not handle:
            return kernel32.GetLastError() != ERROR_INVALID_PARAMETER
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
    # Hard exit from the watchdog thread. A self-SIGTERM would be deferred while
    # the main thread is GIL-blocked in a C socket read (the reason the parent
    # cancel path uses SIGKILL); the partial .incomplete resumes byte-exact and
    # marker/manifest writes are atomic, so the cancelled code 130 is safe here.
    print(
        "Parent process exited; stopping orphaned download worker.",
        file = sys.stderr,
    )
    sys.stderr.flush()
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


def _model_info_with_retry(
    repo_id: str,
    hf_token: str | None,
    *,
    files_metadata: bool = False,
):
    from huggingface_hub import model_info as hf_model_info

    for attempt, timeout in enumerate(
        (_METADATA_REQUEST_TIMEOUT, _METADATA_RETRY_TIMEOUT)
    ):
        try:
            kwargs = {"token": _hf_token_arg(hf_token), "timeout": timeout}
            if files_metadata:
                kwargs["files_metadata"] = True
            return hf_model_info(repo_id, **kwargs)
        except Exception as e:
            if attempt == 1:
                raise
            print(
                f"Metadata request failed for {repo_id} "
                f"({type(e).__name__}: {e}); retrying.",
                file = sys.stderr,
            )
            time.sleep(_METADATA_RETRY_DELAY)
    raise RuntimeError(f"Metadata unavailable for {repo_id}")


def _dataset_info_with_retry(
    repo_id: str,
    hf_token: str | None,
    *,
    files_metadata: bool = False,
):
    from huggingface_hub import HfApi

    api = HfApi(token = _hf_token_arg(hf_token))
    for attempt, timeout in enumerate(
        (_METADATA_REQUEST_TIMEOUT, _METADATA_RETRY_TIMEOUT)
    ):
        try:
            kwargs = {"timeout": timeout}
            if files_metadata:
                kwargs["files_metadata"] = True
            return api.dataset_info(repo_id, **kwargs)
        except Exception as e:
            if attempt == 1:
                raise
            print(
                f"Dataset metadata request failed for {repo_id} "
                f"({type(e).__name__}: {e}); retrying.",
                file = sys.stderr,
            )
            time.sleep(_METADATA_RETRY_DELAY)
    raise RuntimeError(f"Dataset metadata unavailable for {repo_id}")


# Tied to drain_stderr_excerpt's 500-byte head/tail window in the parent
# (see hub/utils/download_registry.py): a verification failure that names
# every expected file would blow past that window and lose the diagnostic.
# Cap the per-list preview so the summary line survives the drain.
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
) -> None:
    """Walk the manifest and verify every expected file is on disk at the
    declared size. Exit nonzero with a diagnostic stderr message if not.

    No-op when no manifest exists for this triple: step 2's manifest
    write is best-effort (either the metadata fetch failed and we fell
    back to legacy ignore-patterns, or the disk write itself failed),
    so absence here means "verification unavailable, trust
    snapshot_download's exit code" — the pre-fix behavior.
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


def _preflight_disk_space(
    repo_type: str,
    repo_id: str,
    expected_files: list,
) -> None:
    """Fail fast with a clear message when the active HF cache filesystem can't
    hold what's left to download. Best-effort and fail-open: any inability to
    size the work or read free space skips the check, so a real download is
    never blocked by an estimation gap (the mid-write ENOSPC path still
    applies, and its partial blobs resume on retry)."""
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
            sha256 = _sibling_sha256(s),
        )
        for s in filtered
        if isinstance(s.rfilename, str)
    ]
    return resolve_snapshot_ignore_patterns_for_files(filenames), expected_files


def _sibling_sha256(sibling) -> str | None:
    return sibling_sha256(sibling)


def _dataset_expected_files(info) -> list:
    from hub.utils.download_manifest import ExpectedFile

    return [
        ExpectedFile(
            path = s.rfilename,
            size = int(getattr(s, "size", 0) or 0),
            sha256 = _sibling_sha256(s),
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
    """Best-effort manifest write for a download whose metadata was
    unavailable at start: re-fetch and record the expected files, else fall
    back to the on-disk file list so completion/partial detection still works.
    Shared by the model and dataset workers (they differ only in the metadata
    fetch, the expected-files extraction, and the log label)."""
    from hub.utils import download_manifest

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

    fallback_files = download_manifest.expected_files_from_snapshot_dir(
        Path(snapshot_path)
    )
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

    # One metadata fetch with files_metadata=True; powers both the
    # ignore-pattern decision (drop consolidated.* when transformers weights
    # exist) and the manifest's expected_files. A failure here is non-fatal:
    # we fall back to the legacy ignore-pattern set (keeping consolidated, the
    # safer default) and skip the manifest, so download proceeds without the
    # post-completion verification + scanner partial detection that the
    # manifest enables.
    try:
        info = _model_info_with_retry(repo_id, hf_token, files_metadata = True)
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
        # files materialized under snapshots/, which both transports produce
        # identically (XET's xet_get downloads into <blob>.incomplete and the
        # shared _chmod_and_move renames it to a full, correctly-sized blob,
        # exactly like the HTTP path). XET's block-level dedup lives only in
        # the separate chunk-cache, which the manifest never inspects, so
        # per-file size verification is valid regardless of transport.
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
            fetch_info = lambda: _model_info_with_retry(
                repo_id,
                hf_token,
                files_metadata = True,
            ),
            expected_files_from_info = lambda recovered: _snapshot_download_plan(
                recovered
            )[1],
        )
    _verify_completed_download("model", repo_id, None, snapshot_path)


def _gguf_variant_target_plan(
    repo_id: str,
    variant: str,
    hf_token: str | None,
) -> GgufVariantPlan | None:
    try:
        info = _model_info_with_retry(repo_id, hf_token, files_metadata = True)
    except Exception as e:
        print(
            f"metadata unavailable, cannot resolve GGUF variant '{variant}' "
            f"for {repo_id} ({type(e).__name__}: {e})",
            file = sys.stderr,
        )
        raise RuntimeError(
            f"Metadata unavailable while resolving GGUF variant '{variant}' "
            f"for {repo_id}"
        ) from e
    return build_gguf_variant_plans(list(info.siblings)).get(variant.lower())


def _download_gguf_variant(
    repo_id: str,
    variant: str,
    hf_token: str | None,
    mode: str,
) -> None:
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
    # The main quant blobs are owned by this variant and judged by the
    # variant-scoped transport marker. The vision companion (mmproj) is shared
    # across variants, so it is judged by a separate companion marker and never
    # purged while a concurrent peer is writing it.
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
    _verify_completed_download("model", repo_id, variant, snapshot_path)


def _download_dataset(repo_id: str, hf_token: str | None, mode: str) -> None:
    from huggingface_hub import snapshot_download
    from hub.utils.download_registry import prepare_cache_for_transport
    from hub.utils import download_manifest

    try:
        info = _dataset_info_with_retry(repo_id, hf_token, files_metadata = True)
    except Exception as e:
        print(
            f"dataset metadata unavailable, downloading full dataset for {repo_id} "
            f"({type(e).__name__}: {e})",
            file = sys.stderr,
        )
        info = None
    # Both the cancel-marker clear and the manifest write run on every
    # transport. See _download_snapshot for why per-file size verification
    # is valid under XET as well as HTTP.
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
            fetch_info = lambda: _dataset_info_with_retry(
                repo_id,
                hf_token,
                files_metadata = True,
            ),
            expected_files_from_info = _dataset_expected_files,
            label = "dataset ",
        )
    _verify_completed_download("dataset", repo_id, None, snapshot_path)


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
        # Surface a precise message to the parent so the UI doesn't show
        # a generic "worker exited with code 1". huggingface_hub's
        # consistency check explicitly recommends `force_download=True`
        # to recover, which our "Restart" UI maps to a fresh start by
        # purging the partial via prepare_cache_for_transport.
        print(f"{type(e).__name__}: {e}", file = sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
