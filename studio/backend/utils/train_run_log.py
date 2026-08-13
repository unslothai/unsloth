# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-run disk log for the training worker.

The worker is a spawn child whose stdout and stderr are the ones it inherited, so its output
lands wherever the parent's happened to point: interleaved into a shared multi-hundred-KB
server session log on the desktop, and only in the journal on a headless host. A run that
died left no file of its own to ask for, which is why "my training failed" has no artifact
attached to it.

Give each run ``logs/train/<run_id>.log``, teed so the console keeps everything it printed
before, with faulthandler aimed at the same file so a native crash in a GPU runtime leaves a
stack trace behind. Path is by convention, so anything listing the directory finds runs
without a lookup.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Keep more than the server's 20: runs are the thing people come back to ask about, and a
# training log is small next to the session log it used to hide in.
KEEP_RUN_LOGS = 50

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_run_id(run_id: str) -> str:
    cleaned = _UNSAFE.sub("_", (run_id or "").strip())[:128]
    return cleaned or "unknown-run"


class _Tee:
    """Mirror writes to the inherited stream and the run log.

    Console behaviour is unchanged: writes and their return values delegate to the original
    stream, so the parent's line-oriented event parsing sees exactly what it saw before. The
    file copy is best effort in every direction; a full disk must not fail a training run.
    """

    def __init__(self, stream, log_fh):
        self._stream = stream
        self._log_fh = log_fh

    def write(self, data):
        try:
            self._log_fh.write(data)
        except Exception:
            pass
        if self._stream is None:
            return len(data)
        return self._stream.write(data)

    def flush(self):
        try:
            self._log_fh.flush()
        except Exception:
            pass
        if self._stream is not None:
            try:
                self._stream.flush()
            except Exception:
                pass

    def isatty(self):
        # tqdm and rich ask before choosing a renderer; answer for the console, not the file,
        # or an interactive run silently loses its progress bars.
        try:
            return bool(self._stream is not None and self._stream.isatty())
        except Exception:
            return False

    def fileno(self):
        if self._stream is None:
            raise OSError("no underlying stream")
        return self._stream.fileno()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _prune(log_dir: Path, keep: int = KEEP_RUN_LOGS) -> None:
    try:
        entries = sorted(log_dir.glob("*.log"), key = lambda p: p.stat().st_mtime)
    except OSError:
        return
    for old in entries[:-keep] if keep else entries:
        try:
            old.unlink(missing_ok = True)
        except OSError:
            pass


def setup_run_log(run_id: str, studio_home: str | os.PathLike[str] | None = None):
    """Tee this worker's stdout/stderr into ``logs/train/<run_id>.log``.

    Returns the log path, or None when disabled or unavailable. Opt out with
    ``UNSLOTH_STUDIO_NO_FILE_LOG=1``, the same switch the server session log honours.
    """
    if os.environ.get("UNSLOTH_STUDIO_NO_FILE_LOG") == "1":
        return None

    if studio_home is not None:
        home = Path(studio_home)
    else:
        try:
            from utils.paths import studio_root
            home = Path(studio_root())
        except Exception:
            home = Path(
                os.environ.get("UNSLOTH_STUDIO_HOME")
                or os.environ.get("STUDIO_HOME")
                or os.path.join(os.path.expanduser("~"), ".unsloth", "studio")
            )

    log_dir = home / "logs" / "train"
    try:
        log_dir.mkdir(parents = True, exist_ok = True)
        _prune(log_dir)
        log_path = log_dir / f"{_safe_run_id(run_id)}.log"
        # Append, not truncate: run ids are unique per run, so this creates the file, but a
        # respawn of the same run (the Xet -> HTTP retry) reuses the id and must add to what
        # the first attempt recorded instead of erasing the reason it was retried.
        # Line buffered so the tail survives a hard kill, which is the case this exists for.
        log_fh = open(log_path, "a", encoding = "utf-8", errors = "replace", buffering = 1)
    except Exception:
        return None

    try:
        import faulthandler
        faulthandler.enable(file = log_fh, all_threads = True)
    except Exception:
        pass

    sys.stdout = _Tee(sys.stdout, log_fh)
    sys.stderr = _Tee(sys.stderr, log_fh)
    return log_path
