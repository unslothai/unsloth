# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Out-of-process byte transfer for one dictation model download.

The sidecars plan the download in-process and run only the transfer here. A
thread cannot be interrupted, so cancelling means terminating a process: like
``hub.workers.hf_download``, cancel is a SIGTERM and the negative returncode is
what the caller reads as cancelled rather than failed. HF partials resume
byte-exact, so a SIGTERM costs only the chunk in flight.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


def backend_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def spawn_download(args: Sequence[str], hf_token: Optional[str] = None) -> subprocess.Popen:
    """Run this module as a child process performing ``args``' download.

    The token travels in the environment, never argv, so it stays out of ``ps``.
    """
    cwd = backend_dir()
    from utils.hf_cache_settings import get_hf_cache_paths

    env = get_hf_cache_paths().child_env()
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # Parallel Range chunks leave sparse partials a resumed sequential writer
    # cannot reuse, which defeats the point of cancelling.
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    # Replace the inherited credentials only when the caller supplied one. With
    # no token, leave the ambient login alone: the parent plans the download
    # with it, so scrubbing here would fail gated repos that used to work.
    if hf_token:
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "0"
        for token_key in (
            "HF_TOKEN",
            "HF_HUB_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "HUGGINGFACEHUB_API_TOKEN",
        ):
            env.pop(token_key, None)
        env["HF_TOKEN"] = hf_token
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{cwd}{os.pathsep}{existing_path}" if existing_path else str(cwd)
    from utils.process_lifetime import adopt_pid, child_popen_kwargs

    process = subprocess.Popen(
        [sys.executable, "-m", "core.inference.stt_download_worker", *args],
        env = env,
        cwd = str(cwd),
        stdout = subprocess.DEVNULL,
        stderr = subprocess.PIPE,
        # Die with Studio: a detached worker would keep pulling gigabytes after
        # the app closed, with nothing left able to stop it.
        **child_popen_kwargs(),
    )
    adopt_pid(process.pid)  # terminate_all backstop for graceful exits
    return process


def reap_download(process: subprocess.Popen) -> bytes:
    """Wait for a worker and drop its PID. Returns its stderr.

    Callers pair every spawn with this: an adopted PID that outlives the process
    can be reused by something unrelated, which terminate_all would then signal
    (macOS and Windows cannot pin a PID to an identity the way /proc does).
    """
    from utils.process_lifetime import forget_pid

    try:
        _, stderr = process.communicate()
    finally:
        forget_pid(process.pid)
    return stderr or b""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description = "Download one dictation model.")
    parser.add_argument("--repo-id", required = True)
    # GGML is a single file; Transformers is a snapshot pinned to the revision
    # the sidecar already validated.
    parser.add_argument("--filename")
    parser.add_argument("--revision")
    parser.add_argument("--allow-pattern", action = "append", default = [])
    args = parser.parse_args(argv)

    token = os.environ.get("HF_TOKEN") or None
    if args.filename:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id = args.repo_id,
            filename = args.filename,
            revision = args.revision,
            token = token,
        )
        return 0

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id = args.repo_id,
        revision = args.revision,
        allow_patterns = list(args.allow_pattern) or None,
        token = token,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
