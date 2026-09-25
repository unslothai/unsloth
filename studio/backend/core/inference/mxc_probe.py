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
_cache: dict[tuple, tuple[float, bool, str]] = {}
_inflight: dict[tuple, threading.Event] = {}
POSITIVE_TTL = 300.0
NEGATIVE_TTL = 30.0


_host_prep_cache: dict[str, tuple[float, str | None]] = {}


def invalidate_cache() -> None:
    with _lock:
        _cache.clear()
        _host_prep_cache.clear()


def host_prep_remediation() -> str | None:
    """Name the elevated command when MXC reports Tier 3 host preparation missing."""
    try:
        identity = mxc_runtime.installation_identity()
    except Exception:  # noqa: BLE001 - no runtime, nothing to prepare
        return None
    with _lock:
        cached = _host_prep_cache.get(identity)
        if cached is not None and time.monotonic() < cached[0]:
            return cached[1]
    steps = mxc_runtime.probe_host_prep_steps(env = mxc_adapter._control_environment())
    advice = None
    if steps:
        command = subprocess.list2cmdline(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[3] / "install_mxc_prebuilt.py"),
                "--prepare-host",
                "--install-dir",
                str(mxc_runtime._installed_package_root()),
            ]
        )
        reboot = " (prepare-null-device is undone by every reboot)" if (
            "prepare-null-device" in steps
        ) else ""
        advice = (
            f"MXC reports missing host preparation: {', '.join(steps)}{reboot}. "
            f"Run {command} and approve the administrator prompt."
        )
    with _lock:
        _host_prep_cache[identity] = (time.monotonic() + NEGATIVE_TTL, advice)
    return advice


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
        ps_inside = str(workdir / "inside.txt").replace("'", "''")
        ps_canary = str(canary).replace("'", "''")
        ps_capture = str(read_capture).replace("'", "''")
        ps_outside = str(outside_write).replace("'", "''")
        command = (
            f"Set-Content -LiteralPath '{ps_inside}' -Value 'ok'; "
            f"try {{ Get-Content -LiteralPath '{ps_canary}' -ErrorAction Stop | "
            f"Set-Content -LiteralPath '{ps_capture}' }} catch {{}}; "
            f"try {{ Set-Content -LiteralPath '{ps_outside}' -Value 'bad' "
            "-ErrorAction Stop } catch {}; "
            "Write-Output 'UNSLOTH_MXC_TERMINAL_PROBE_OK'"
        )
        return (
            selected_executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        )
    if name in {"bash", "bash.exe"}:
        command = (
            "printf ok > inside.txt; "
            f"cat {shlex.quote(str(canary))} > {shlex.quote(str(read_capture))} 2>/dev/null; "
            f"printf bad > {shlex.quote(str(outside_write))} 2>/dev/null; "
            "printf 'UNSLOTH_MXC_TERMINAL_PROBE_OK\\n'"
        )
        return (selected_executable, "-c", command)
    raise ValueError(f"the selected Windows Terminal shell is not qualified for MXC: {name}")


def _probe(
    selected_executable: str,
    execution_kind: str,
    cancel_event = None,
) -> tuple[bool, str]:
    if sys.platform != "win32":
        return False, "MXC is enabled only for native Windows Studio"
    if sys.getwindowsversion().build < 26100:
        return False, "Windows 11 build 26100 or newer is required for the MXC Preview"
    try:
        mxc_runtime.wxc_path()
    except mxc_runtime.MxcRuntimeUnavailable as exc:
        return False, str(exc)
    if cancel_event is not None and cancel_event.is_set():
        return False, "MXC capability probe was cancelled"

    with tempfile.TemporaryDirectory(prefix = "unsloth-mxc-probe-") as root:
        root_path = Path(root)
        workdir = root_path / "workdir"
        other = root_path / "other-chat"
        workdir.mkdir()
        other.mkdir()
        canary = other / "secret.txt"
        canary.write_text("outside-secret", encoding = "utf-8")
        outside_write = other / "outside-write.txt"
        outside_write.write_text("host-positive", encoding = "utf-8")
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
                encoding = "utf-8",
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
        env["HOME"] = str(workdir)
        env["TEMP"] = str(workdir)
        env["TMP"] = str(workdir)

        probe_plan = SimpleNamespace(
            argv = probe_argv,
            execution_kind = execution_kind,
            timeout_seconds = 20,
            workdir = str(workdir),
            env = env,
        )

        try:
            request = mxc_policy.build_launch_request(probe_plan)
            proc = mxc_adapter.spawn(
                request,
                cancel_event = cancel_event,
                popen_kwargs = {
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
                    output, _ = proc.communicate(timeout = min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            proc._unsloth_completion_reason = "finished"
            result = mxc_adapter.completion_result(proc)
        except Exception as exc:  # capability result, never a workload fallback decision
            return False, f"the live MXC probe failed: {type(exc).__name__}: {exc}"
        finally:
            if "proc" in locals():
                mxc_adapter.release_runtime(proc)
        if (
            proc.returncode != 0
            or result.get("exitCode") != 0
            or result.get("cleanup") != "complete"
        ):
            return False, "the live MXC probe did not complete cleanly"
        if execution_kind == "terminal":
            captured = workdir / "outside-read.txt"
            outside_read = (
                captured.read_text(encoding = "utf-8", errors = "replace") if captured.exists() else ""
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
    cancel_event = None,
) -> tuple[bool, str]:
    try:
        identity = mxc_runtime.installation_identity()
    except Exception as exc:
        return False, str(exc)
    key = (
        identity,
        mxc_runtime.PROFILE_ID,
        execution_kind,
        os.path.abspath(selected_executable),
        mxc_policy.dacl_fallback_enabled(),
    )
    while True:
        with _lock:
            cached = _cache.get(key)
            if cached is not None and not force and time.monotonic() < cached[0]:
                return cached[1], cached[2]
            flight = _inflight.get(key)
            if flight is None:
                flight = threading.Event()
                _inflight[key] = flight
                break
        while not flight.wait(0.05):
            if cancel_event is not None and cancel_event.is_set():
                return False, "MXC capability probe was cancelled"

    try:
        result = _probe(selected_executable, execution_kind, cancel_event)
        with _lock:
            if cancel_event is None or not cancel_event.is_set():
                ttl = POSITIVE_TTL if result[0] else NEGATIVE_TTL
                _cache[key] = (time.monotonic() + ttl, *result)
        return result
    finally:
        with _lock:
            _inflight.pop(key, None)
            flight.set()
