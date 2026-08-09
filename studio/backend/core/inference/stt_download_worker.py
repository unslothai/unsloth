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
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

# Fresh spawned interpreter: re-apply the OS-trust-store injection.
from utils.native_tls import activate_native_tls

activate_native_tls()


# How long a cancelled worker gets to exit on SIGTERM before it is killed.
TERMINATE_GRACE_SECONDS = 15.0


def backend_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def spawn_download(
    args: Sequence[str],
    hf_token: Optional[str] = None,
    *,
    hub_cache: Optional[Path] = None,
) -> subprocess.Popen:
    """Run this module as a child process performing ``args``' download.

    The token travels in the environment, never argv, so it stays out of ``ps``.
    """
    cwd = backend_dir()
    from utils.hf_cache_settings import get_hf_cache_paths

    env = get_hf_cache_paths().child_env()
    if hub_cache is not None:
        env["HF_HUB_CACHE"] = str(hub_cache)
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    # Xet's out-of-order chunks do not provide steady partial-file progress.
    env["HF_HUB_DISABLE_XET"] = "1"
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


def terminate_download(process: subprocess.Popen) -> None:
    """SIGTERM now, SIGKILL after a grace, so cancel() still returns at once.

    The canceller holds the repository reservation until the reap returns, so a
    worker that ignores SIGTERM would lock every Model Hub write on that repo
    until Studio restarts.
    """
    try:
        process.terminate()
    except Exception:  # noqa: BLE001
        return

    def escalate() -> None:
        time.sleep(TERMINATE_GRACE_SECONDS)
        try:
            if process.poll() is None:
                process.kill()
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target = escalate, daemon = True).start()


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
        pid = getattr(process, "pid", None)
        if isinstance(pid, int):
            forget_pid(pid)
    return stderr or b""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description = "Download one dictation model.")
    parser.add_argument("--repo-id", required = True)
    # Explicit filenames prevent repository patterns from widening the download.
    parser.add_argument("--filename", action = "append", required = True)
    parser.add_argument("--revision")
    args = parser.parse_args(argv)

    token = os.environ.get("HF_TOKEN") or None
    from huggingface_hub import hf_hub_download

    for filename in args.filename:
        hf_hub_download(
            repo_id = args.repo_id,
            filename = filename,
            revision = args.revision,
            token = token,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
