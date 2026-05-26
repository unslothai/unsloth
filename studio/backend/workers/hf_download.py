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
  the spawning side (see :mod:`utils.hf_cache_scan`) when transport=http.
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
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils.hf_snapshot_filters import (
    SNAPSHOT_IGNORE_PATTERNS,
    CONSOLIDATED_PATTERN,
    repo_ships_transformers_weights,
)


# Bound the variant-resolution metadata fetch so a stalled connection fails the
# worker (exit 1, surfaced as a job error) instead of hanging at 0% until the
# user manually cancels. 10s is well above the typical sub-second hf_model_info
# response on healthy networks while keeping the worst-case "stuck at 0%"
# perception short on flaky connections. The file download itself is governed
# separately by huggingface_hub's own download timeout.
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
            kwargs = {"token": hf_token, "timeout": timeout}
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


def _dataset_info_with_retry(repo_id: str, hf_token: str | None):
    from huggingface_hub import HfApi

    api = HfApi(token = hf_token)
    for attempt, timeout in enumerate(
        (_METADATA_REQUEST_TIMEOUT, _METADATA_RETRY_TIMEOUT)
    ):
        try:
            return api.dataset_info(repo_id, timeout = timeout)
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


def _repo_ships_transformers_weights(repo_id: str, hf_token: str | None) -> bool:
    """True when the repo provides standard transformers-format weights
    (``model*.safetensors``/``pytorch_model*.bin`` or their sharded variants),
    which make a sibling ``consolidated.*`` copy redundant."""
    info = _model_info_with_retry(repo_id, hf_token)
    return repo_ships_transformers_weights(
        sibling.rfilename for sibling in info.siblings
    )


def _resolve_snapshot_ignore_patterns(
    repo_id: str, hf_token: str | None,
) -> list[str]:
    ignore = list(SNAPSHOT_IGNORE_PATTERNS)
    # Drop consolidated.* only when transformers-format weights exist. If the
    # layout can't be inspected, keep it (never silently strip the sole weights).
    try:
        redundant = _repo_ships_transformers_weights(repo_id, hf_token)
    except Exception as e:
        print(
            f"metadata unavailable, downloading full snapshot for {repo_id} "
            f"({type(e).__name__}: {e})",
            file = sys.stderr,
        )
        redundant = False
    if redundant:
        ignore.append(CONSOLIDATED_PATTERN)
    return ignore


def _download_snapshot(repo_id: str, hf_token: str | None, mode: str) -> None:
    from huggingface_hub import snapshot_download
    from utils.hf_cache_scan import prepare_cache_for_transport

    ignore_patterns = _resolve_snapshot_ignore_patterns(repo_id, hf_token)
    purged = prepare_cache_for_transport("model", repo_id, mode)
    if purged:
        print(
            f"Purged {purged} untrusted partial blob(s) for {repo_id} "
            f"before starting {mode} download.",
            file = sys.stderr,
        )
    snapshot_download(
        repo_id = repo_id,
        token = hf_token,
        ignore_patterns = ignore_patterns,
        max_workers = 1,
    )


def _gguf_variant_patterns(
    repo_id: str,
    variant: str,
    hf_token: str | None,
) -> list[str]:
    from utils.models.model_config import _extract_quant_label

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
    targets: list[str] = []
    mmproj_candidates: list[str] = []
    for sibling in info.siblings:
        fname = sibling.rfilename
        if not fname.lower().endswith(".gguf"):
            continue
        if "mmproj" in fname.lower():
            mmproj_candidates.append(fname)
            continue
        if _extract_quant_label(fname) != variant:
            continue
        targets.append(fname)
    # Bundle the vision projector with the variant so offline vision use works
    # and the Hub "downloaded" status reflects the full capability of the repo.
    # Prefer F16 (canonical), then any mmproj. Use the parsed quant label so
    # "F16" matches only mmproj-F16 — a substring check matches "bf16" too
    # and silently bundled the wrong precision whenever mmproj-BF16 appeared
    # first in the sibling order.
    if targets and mmproj_candidates:
        preferred_f16 = next(
            (
                f
                for f in mmproj_candidates
                if _extract_quant_label(f).upper() == "F16"
            ),
            None,
        )
        preferred = preferred_f16 or mmproj_candidates[0]
        if preferred_f16 is None:
            print(
                f"No F16 mmproj found for {repo_id}; bundling {preferred} "
                f"with GGUF variant '{variant}'.",
                file = sys.stderr,
            )
        targets.append(preferred)
    return targets


def _download_gguf_variant(repo_id: str, variant: str, hf_token: str | None, mode: str) -> None:
    from huggingface_hub import snapshot_download
    from utils.hf_cache_scan import prepare_cache_for_transport

    targets = _gguf_variant_patterns(repo_id, variant, hf_token)
    if not targets:
        print(
            f"No GGUF shards matching variant '{variant}' in {repo_id}",
            file = sys.stderr,
        )
        sys.exit(1)
    purged = prepare_cache_for_transport("model", repo_id, mode)
    if purged:
        print(
            f"Purged {purged} untrusted partial blob(s) for {repo_id} "
            f"before starting {mode} download.",
            file = sys.stderr,
        )
    snapshot_download(
        repo_id = repo_id,
        token = hf_token,
        allow_patterns = targets,
        max_workers = 1,
    )


def _download_dataset(repo_id: str, hf_token: str | None, mode: str) -> None:
    from huggingface_hub import snapshot_download
    from utils.hf_cache_scan import prepare_cache_for_transport

    try:
        _dataset_info_with_retry(repo_id, hf_token)
    except Exception as e:
        print(
            f"dataset metadata unavailable for {repo_id} "
            f"({type(e).__name__}: {e})",
            file = sys.stderr,
        )
        raise RuntimeError(f"Dataset metadata unavailable for {repo_id}") from e
    purged = prepare_cache_for_transport("dataset", repo_id, mode)
    if purged:
        print(
            f"Purged {purged} untrusted partial blob(s) for {repo_id} "
            f"before starting {mode} download.",
            file = sys.stderr,
        )
    snapshot_download(
        repo_id = repo_id,
        token = hf_token,
        repo_type = "dataset",
        max_workers = 1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description = "HuggingFace Hub download worker")
    parser.add_argument("--repo-id", required = True)
    parser.add_argument("--variant", default = None)
    parser.add_argument("--dataset", action = "store_true")
    parser.add_argument("--transport", choices = ("http", "xet"), default = "http")
    args = parser.parse_args()

    _install_signal_handlers()

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
