# SPDX-License-Identifier: AGPL-3.0-only
"""Owned Windows runtime helper with bounded startup and shutdown."""

import json
import os
import queue
import secrets
import subprocess
import threading
import time


class _Broker:
    def __init__(
        self,
        identity,
        deadline,
        *,
        owner_script = "windows-read-owner.mjs",
        bootstrap = None,
    ):
        from . import srt_adapter
        from .tools import _windows_job_capture

        self.identity = identity
        self.token = secrets.token_hex(32)
        self.sid = None
        self.helper_env = {
            key: value
            for key, value in os.environ.items()
            if key.upper()
            in {
                "SYSTEMROOT",
                "WINDIR",
                "PATH",
                "TEMP",
                "TMP",
                "USERPROFILE",
                "LOCALAPPDATA",
                "APPDATA",
                "PROGRAMDATA",
                "PATHEXT",
            }
        }
        self.proc = subprocess.Popen(
            [srt_adapter.node_executable(), str(srt_adapter.RUNTIME / owner_script)],
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            creationflags = subprocess.CREATE_NO_WINDOW,
            env = self.helper_env,
        )
        self.job = None
        try:
            self.job = _windows_job_capture(self.proc, allow_breakaway = True)
            if self.job is None:
                raise srt_adapter.SrtError("Cannot establish sandbox session ownership")
            self.proc.stdin.write(
                json.dumps(
                    {**(bootstrap or {}), "token": self.token, "identity": identity}
                ).encode()
                + b"\n"
            )
            self.proc.stdin.flush()
            ready = queue.Queue(maxsize = 1)
            threading.Thread(
                target = lambda: ready.put(self.proc.stdout.readline(4097)), daemon = True
            ).start()
            try:
                line = ready.get(timeout = max(0.01, deadline - time.monotonic()))
            except queue.Empty as exc:
                raise srt_adapter.SrtError(
                    "Runtime read permission setup timed out",
                    code = "probe_timeout",
                    stage = "probe",
                ) from exc
            value = json.loads(line)
            if isinstance(value, dict) and isinstance(value.get("error"), dict):
                detail = value["error"]
                raise srt_adapter.SrtError(
                    "Runtime read permission setup failed",
                    code = detail.get("code"),
                    stage = detail.get("stage"),
                    dependency = detail.get("dependency"),
                    details = detail,
                )
            self.port = value["port"]
            self.sid = value.get("sid")
            if type(self.port) is not int or not 0 < self.port <= 65535:
                raise ValueError("Invalid session endpoint")
        except Exception:
            try:
                self.close()
            except Exception:
                pass  # Preserve the setup diagnostic; the owner was never admitted.
            raise

    def close(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout = 65)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout = 5)
        finally:
            if self.job is not None:
                self.job.terminate()
            if self.proc.stdout:
                self.proc.stdout.close()
        if self.proc.returncode != 0:
            from . import srt_adapter
            if self.sid is not None:
                result = subprocess.run(
                    [
                        srt_adapter.node_executable(),
                        str(srt_adapter.RUNTIME / "windows-read-recover.mjs"),
                    ],
                    input = json.dumps({"pid": self.proc.pid, "sid": self.sid}),
                    text = True,
                    capture_output = True,
                    timeout = 65,
                    creationflags = subprocess.CREATE_NO_WINDOW,
                    env = self.helper_env,
                )
                if result.returncode == 0 and json.loads(result.stdout) == {"released": True}:
                    return
            raise srt_adapter.SrtError("Sandbox session shutdown could not be verified")
