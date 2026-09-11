# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Explicit local Windows setup, separate from tool execution and consent."""

from pathlib import Path
import subprocess
import sys
import threading
import tempfile

from . import srt_probe

_setup_lock = threading.Lock()
_setup_uncertain = False


def install_windows_sandbox(*, repair_existing = False):
    global _setup_uncertain
    if _setup_uncertain:
        return {
            "status": "timeout",
            "message": "The previous administrator operation timed out. Verify that it has finished, then restart Studio before retrying setup.",
        }
    if sys.platform != "win32":
        return {"status": "unavailable", "message": "This setup action requires Windows."}
    if not _setup_lock.acquire(blocking = False):
        return {"status": "busy", "message": "Windows isolation setup is already running."}
    conflict = False
    try:
        srt_probe.setup_in_progress.set()
        srt_probe.invalidate_cache()
        installer = Path(__file__).resolve().parents[3] / "install_srt_runtime.py"
        command = [sys.executable, "-I", str(installer), "--windows-install"]
        if repair_existing:
            command.append("--windows-force")
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(
                command,
                cwd = installer.parent,
                stdin = subprocess.DEVNULL,
                stdout = output,
                stderr = subprocess.STDOUT,
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout = 540,
                check = False,
            )
            output.seek(0, 2)
            output.seek(max(0, output.tell() - 8192))
            tail = output.read().decode("utf-8", errors = "replace")
        if result.returncode != 0:
            if (
                "filters already exist under this sublayer with a different port range or sandbox-user name"
                in tail
            ):
                conflict = True
                return {
                    "status": "conflict",
                    "message": "Administrator approval succeeded, but an existing SRT setup uses different network settings. Repairing replaces its sandbox network filters; other SRT sessions may need restarting.",
                }
            if "Install cancelled at the UAC prompt" in tail:
                return {
                    "status": "cancelled",
                    "message": "Windows administrator approval was cancelled. Setup was not completed.",
                }
            return {
                "status": "failed",
                "message": "Windows isolation setup failed after installation started. Check Node/npm and the SRT runtime installation, then retry.",
                "diagnostic": {
                    "code": "setup_failed",
                    "stage": "installation",
                    "exit_code": result.returncode,
                },
            }
        return {
            "status": "installed",
            "message": "Setup finished. Checking tool isolation; no command will be retried.",
        }
    except subprocess.TimeoutExpired:
        _setup_uncertain = True
        return {
            "status": "timeout",
            "message": "Windows setup timed out. An administrator operation may still be finishing. Check its status before retrying; tool isolation has not been enabled.",
        }
    except OSError:
        return {
            "status": "failed",
            "message": "Studio could not start Windows isolation setup with its selected Python.",
        }
    finally:
        srt_probe.invalidate_cache(setup_conflict = conflict)
        if not _setup_uncertain:
            srt_probe.setup_in_progress.clear()
        _setup_lock.release()
