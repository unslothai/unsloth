# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Bounded, single-flight live capability probe for the Windows MXC path."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

from . import mxc_adapter, mxc_policy, mxc_runtime

_lock = threading.Lock()
_cache: dict[tuple[str, str, str, str], tuple[float, bool, str]] = {}
POSITIVE_TTL = 300.0
NEGATIVE_TTL = 30.0


def invalidate_cache() -> None:
    with _lock:
        _cache.clear()


def _terminal_probe(selected_executable: str, workdir: Path, canary: Path, outside_write: Path):
    name = Path(selected_executable).name.casefold()
    read_capture = workdir / "outside-read.txt"
    if name in {"cmd", "cmd.exe"}:
        command = (
            "echo ok>inside.txt"
            f' & type "{canary}" >"{read_capture}" 2>nul'
            f' & (echo bad>"{outside_write}") 2>nul'
            " & echo UNSLOTH_MXC_TERMINAL_PROBE_OK"
        )
        return (selected_executable, "/d", "/s", "/c", command)
    if name in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        script = workdir / "probe.ps1"
        quote = lambda value: str(value).replace("'", "''")
        script.write_text(
            "Set-Content -LiteralPath 'inside.txt' -Value 'ok' -Encoding utf8\n"
            f"try {{ Get-Content -LiteralPath '{quote(canary)}' -ErrorAction Stop | Set-Content -LiteralPath '{quote(read_capture)}' }} catch {{}}\n"
            f"try {{ Set-Content -LiteralPath '{quote(outside_write)}' -Value 'bad' -ErrorAction Stop }} catch {{}}\n"
            "Write-Output 'UNSLOTH_MXC_TERMINAL_PROBE_OK'\n",
            encoding="utf-8",
        )
        return (
            selected_executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(script),
        )
    if name in {"bash", "bash.exe"}:
        script = workdir / "probe.sh"
        canary_posix = str(canary).replace("\\", "/")
        outside_posix = str(outside_write).replace("\\", "/")
        script.write_text(
            "printf ok > inside.txt || exit 11\n"
            f"cat {shlex.quote(canary_posix)} > outside-read.txt 2>/dev/null || :\n"
            f"printf bad > {shlex.quote(outside_posix)} 2>/dev/null || :\n"
            "printf 'UNSLOTH_MXC_TERMINAL_PROBE_OK\\n'\n",
            encoding="utf-8",
        )
        return (selected_executable, str(script))
    raise ValueError(f"unsupported Windows terminal executable: {selected_executable}")


def _probe(selected_executable: str, execution_kind: str, cancel_event=None) -> tuple[bool, str]:
    if sys.platform != "win32":
        return False, "MXC is enabled only for native Windows Studio"
    if sys.getwindowsversion().build < 26100:
        return False, "Windows 11 build 26100 or newer is required for the MXC Preview"
    try:
        mxc_runtime.runner_path()
    except mxc_runtime.MxcRuntimeUnavailable as exc:
        return False, str(exc)
    if cancel_event is not None and cancel_event.is_set():
        return False, "MXC capability probe was cancelled"

    with tempfile.TemporaryDirectory(prefix="unsloth-mxc-probe-") as root:
        root_path = Path(root)
        workdir = root_path / "workdir"
        other = root_path / "other-chat"
        workdir.mkdir()
        other.mkdir()
        canary = other / "secret.txt"
        canary.write_text("outside-secret", encoding="utf-8")
        outside_write = other / "outside-write.txt"
        outside_write.write_text("host-positive", encoding="utf-8")
        outside_write.unlink()
        if execution_kind == "python":
            script = workdir / "probe.py"
            script.write_text(
                "import json, os, subprocess, sys\n"
                f"canary={str(canary)!r}\n"
                f"outside={str(outside_write)!r}\n"
                "result={}\n"
                "open('inside.txt','w',encoding='utf-8').write('ok')\n"
                "result['workdir']=open('inside.txt',encoding='utf-8').read()=='ok'\n"
                "result['import']=json.loads('{\"ok\":true}')['ok']\n"
                "result['subprocess']=subprocess.run([sys.executable,'-c','print(7)'],capture_output=True,text=True).stdout.strip()=='7'\n"
                "try:\n open(canary,encoding='utf-8').read(); result['outside_read_denied']=False\n"
                "except OSError: result['outside_read_denied']=True\n"
                "try:\n open(outside,'w',encoding='utf-8').write('bad'); result['outside_write_denied']=False\n"
                "except OSError: result['outside_write_denied']=True\n"
                "print('UNSLOTH_MXC_PROBE='+json.dumps(result,sort_keys=True))\n",
                encoding="utf-8",
            )
            probe_argv = (selected_executable, "-u", str(script))
        elif execution_kind == "terminal":
            try:
                probe_argv = _terminal_probe(selected_executable, workdir, canary, outside_write)
            except ValueError as exc:
                return False, str(exc)
        else:
            return False, f"unsupported MXC execution kind: {execution_kind}"
        env = {
            key: value
            for key, value in os.environ.items()
            if key.upper()
            in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PATHEXT", "PYTHONIOENCODING"}
        }
        env["PYTHONIOENCODING"] = "utf-8"

        probe_plan = SimpleNamespace(
            argv=probe_argv,
            execution_kind=execution_kind,
            timeout_seconds=20,
            workdir=str(workdir),
            env=env,
        )

        try:
            request = mxc_policy.build_launch_request(probe_plan)
            proc = mxc_adapter.spawn(
                request,
                cancel_event=cancel_event,
                popen_kwargs={
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "encoding": "utf-8",
                    "errors": "replace",
                    "creationflags": subprocess.CREATE_NO_WINDOW,
                },
            )
            deadline = time.monotonic() + 25
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    mxc_adapter.abort(proc)
                    return False, "MXC capability probe was cancelled"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    mxc_adapter.abort(proc)
                    return False, "the live MXC probe exceeded its deadline"
                try:
                    output, _ = proc.communicate(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            receipt = mxc_adapter.completion_receipt(proc)
        except Exception as exc:  # capability result, never a workload fallback decision
            return False, f"the live MXC probe failed: {type(exc).__name__}: {exc}"
        finally:
            if "proc" in locals():
                mxc_adapter.release_control(proc)
        if (
            proc.returncode != 0
            or receipt.get("exitCode") != 0
            or receipt.get("cleanup") != "complete"
        ):
            return False, "the live MXC probe did not complete cleanly"
        if execution_kind == "terminal":
            captured = workdir / "outside-read.txt"
            outside_read = (
                captured.read_text(encoding="utf-8", errors="replace") if captured.exists() else ""
            )
            if (
                "UNSLOTH_MXC_TERMINAL_PROBE_OK" not in output
                or outside_write.exists()
                or "outside-secret" in outside_read
                or not (workdir / "inside.txt").is_file()
            ):
                return False, "the live MXC Terminal positive/negative controls failed"
            return True, "the selected Terminal passed the live MXC positive and negative controls"
        marker = next(
            (line for line in output.splitlines() if line.startswith("UNSLOTH_MXC_PROBE=")), None
        )
        if marker is None:
            return False, "the live MXC probe returned no result marker"
        try:
            result = json.loads(marker.split("=", 1)[1])
        except ValueError:
            return False, "the live MXC probe returned malformed results"
        expected = {
            "workdir",
            "import",
            "subprocess",
            "outside_read_denied",
            "outside_write_denied",
        }
        if set(result) != expected or not all(result.values()):
            return (
                False,
                f"the live MXC positive/negative controls failed: {sorted(k for k, v in result.items() if not v)}",
            )
        if outside_write.exists():
            return False, "the MXC outside-write negative control failed"
        return True, "the selected Python passed the live MXC positive and negative controls"


def probe(
    selected_executable: str,
    *,
    execution_kind: str = "python",
    force: bool = False,
    cancel_event=None,
) -> tuple[bool, str]:
    try:
        identity = mxc_runtime.installation_identity()
    except Exception as exc:
        return False, str(exc)
    key = (
        identity,
        mxc_policy.PROFILE_ID,
        execution_kind,
        os.path.abspath(selected_executable),
    )
    with _lock:
        cached = _cache.get(key)
        if cached is not None and not force and time.monotonic() < cached[0]:
            return cached[1], cached[2]
        result = _probe(selected_executable, execution_kind, cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            return result
        ttl = POSITIVE_TTL if result[0] else NEGATIVE_TTL
        _cache[key] = (time.monotonic() + ttl, *result)
        return result
