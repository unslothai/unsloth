# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Native checks for the selected Python and SRT no-network launch profile."""

import json
import logging
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from . import srt_adapter
from .srt_diagnostics import ProbeReason, boundary_identity

_lock = threading.Lock()
_cache = {}
_flights = {}
_cache_epoch = 0
_setup_conflict_identity = None
PROBE_CACHE_SECONDS = 60
PROBE_DEADLINE_SECONDS = 60
setup_in_progress = threading.Event()
_network_cache = None


def _executable_identity(path):
    if not path:
        return None
    resolved = os.path.realpath(path)
    try:
        info = os.stat(resolved)
        return resolved, info.st_size, info.st_mtime_ns
    except OSError:
        return resolved, "missing"


def _windows_probe_shell():
    from .tools import _windows_bash
    return _windows_bash() or shutil.which("cmd")


def runtime_inputs():
    return (
        _executable_identity(sys.executable),
        _executable_identity(shutil.which("node")),
        _executable_identity(
            _windows_probe_shell() if sys.platform == "win32" else shutil.which("bash")
        ),
        tuple(sys.path),
        os.environ.get("PATH", ""),
        os.environ.get("VIRTUAL_ENV", ""),
    )


def _setup_identity():
    return srt_adapter.installation_identity(), runtime_inputs(), boundary_identity()


def probe(
    *,
    force = False,
    execution_kind = None,
    selected_executable = None,
):
    global _setup_conflict_identity
    if setup_in_progress.is_set():
        return False, ProbeReason("operation_unsupported", "installation")
    if sys.platform != "win32":
        return False, ProbeReason("policy_invalid", "policy")
    if execution_kind == "python" and selected_executable == sys.executable:
        execution_kind, selected_executable = None, None
    if sys.platform == "win32" and execution_kind == "terminal" and selected_executable:
        default_shell = _windows_probe_shell()
        if default_shell and os.path.normcase(
            os.path.realpath(selected_executable)
        ) == os.path.normcase(os.path.realpath(default_shell)):
            # The default probe already executes this exact shell as a child.
            execution_kind, selected_executable = None, None
    identity = (
        srt_adapter.installation_identity(),
        os.path.abspath(sys.executable),
        os.path.realpath(sys.executable),
        sys.prefix,
        execution_kind,
        selected_executable,
        _executable_identity(selected_executable),
        runtime_inputs(),
        boundary_identity(),
    )
    with _lock:
        if _setup_conflict_identity is not None:
            if not force and _setup_conflict_identity == _setup_identity():
                return False, ProbeReason("setup_conflict", "installation")
            _setup_conflict_identity = None
        epoch = _cache_epoch
        key = (epoch, identity)
        flight = _flights.get(key)
        cached = _cache.get(identity)
        if (
            flight is None
            and not force
            and cached
            and (
                (sys.platform == "win32" and cached[1][0])
                or time.monotonic() - cached[0] < PROBE_CACHE_SECONDS
            )
        ):
            return cached[1]
        owner = flight is None
        if owner:
            flight = {"done": threading.Event(), "result": None}
            _flights[key] = flight
    if not owner:
        if not flight["done"].wait(PROBE_DEADLINE_SECONDS + 5):
            return False, ProbeReason("probe_timeout", "probe")
        return flight["result"]
    try:
        try:
            result = _native_probe(
                execution_kind = execution_kind,
                selected_executable = selected_executable,
            )
        except subprocess.TimeoutExpired:
            result = (False, ProbeReason("probe_timeout", "probe"))
        except srt_adapter.SrtError as exc:
            result = (False, exc.diagnostic)
        except Exception as exc:
            # Log only exception type and source locations, never command/env data.
            frames = []
            tb = exc.__traceback__
            while tb is not None:
                frames.append(f"{os.path.basename(tb.tb_frame.f_code.co_filename)}:{tb.tb_lineno}")
                tb = tb.tb_next
            logging.getLogger(__name__).warning(
                "SRT probe failed: %s at %s", type(exc).__name__, " -> ".join(frames[-8:])
            )
            result = (False, ProbeReason("probe_failed", "probe"))
        with _lock:
            if epoch != _cache_epoch:
                result = (False, ProbeReason("probe_failed", "probe"))
            else:
                if len(_cache) >= 8:
                    _cache.clear()
                _cache[identity] = (time.monotonic(), result)
            flight["result"] = result
        return result
    finally:
        with _lock:
            if flight["result"] is None:
                flight["result"] = (False, ProbeReason("probe_failed", "probe"))
            _flights.pop(key, None)
            flight["done"].set()


def invalidate_cache(*, setup_conflict = False):
    """Retire pre-setup results, including probes still running."""
    global _cache_epoch, _network_cache, _setup_conflict_identity
    with _lock:
        _cache_epoch += 1
        _cache.clear()
        _network_cache = None
        # A confirmed installer failure needs repair, not another routine probe.
        # This negative result never establishes isolation and is not persisted.
        _setup_conflict_identity = _setup_identity() if setup_conflict else None


def _native_probe(*, execution_kind = None, selected_executable = None):
    if sys.platform != "win32":
        return False, ProbeReason("operation_unsupported", "probe")
    return _supported_platform_probe(
        execution_kind = execution_kind, selected_executable = selected_executable
    )


def _windows_proxy_port_available(request):
    # Check only availability here; the native probe still establishes isolation.
    # Never choose a port outside the range provisioned in SRT's WFP rules.
    low, high = request.get("windowsProxyPortRange", (60080, 60089))
    for port in range(low, high + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            try:
                listener.bind(("127.0.0.1", port))
                return True
            except OSError:
                continue
    return False


def _supported_platform_probe(*, execution_kind = None, selected_executable = None):
    """Check the upstream platform contract without requiring Linux-only isolation."""
    from .tools import _build_safe_env

    budget = PROBE_DEADLINE_SECONDS
    deadline = time.monotonic() + budget
    with tempfile.TemporaryDirectory(prefix = "unsloth-srt-probe-") as directory:
        root = Path(directory)
        work = root / "work"
        work.mkdir()
        sentinel = root / "write-denied.txt"
        sentinel.write_text("unchanged", encoding = "utf-8")
        env = _build_safe_env(str(work))
        shell = (selected_executable if execution_kind == "terminal" else None) or (
            _windows_probe_shell()
            if sys.platform == "win32"
            else shutil.which("bash", path = env.get("PATH"))
        )
        if not shell:
            return False, ProbeReason("dependency_missing", "dependency", "selected_shell")
        python = (selected_executable if execution_kind == "python" else None) or sys.executable
        code = """
import json, pathlib, subprocess, sys
pathlib.Path('private.txt').write_text('workdir write', encoding='utf-8')
assert pathlib.Path('private.txt').read_text(encoding='utf-8') == 'workdir write'
try:
    pathlib.Path(sys.argv[1]).write_text('unexpected write', encoding='utf-8')
except OSError:
    pass
else:
    raise RuntimeError('SRT denied path remained writable')
assert subprocess.check_output(json.loads(sys.argv[2]), text=True).strip() == 'shell-ok'
print('UNSLOTH_SRT_SUPPORTED_PROBE_OK')
"""
        shell_args = (
            [shell, "/d", "/c", "echo shell-ok"]
            if sys.platform == "win32" and os.path.basename(shell).lower() in ("cmd", "cmd.exe")
            else [shell, "--noprofile", "--norc", "-c", "printf shell-ok"]
        )
        request = srt_adapter.request_for(
            [python, "-I", "-S", "-c", code, str(sentinel), json.dumps(shell_args)],
            str(work),
            env,
            min(30, budget),
            operation = "probe",
        )
        request["denyWriteRoots"] = [str(sentinel)]
        if sys.platform == "win32" and not _windows_proxy_port_available(request):
            return False, ProbeReason("proxy_port_unavailable", "launch")
        proc = srt_adapter.spawn(
            request,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            cwd = str(work),
            launch_deadline = deadline,
        )
        try:
            output, _ = proc.communicate(timeout = max(0.001, deadline - time.monotonic()))
            if proc.returncode == 0:
                srt_adapter.verify_success(proc)
            elif srt_adapter.completion_receipt(proc).get("reason") == "timeout":
                raise srt_adapter.SrtError(
                    "SRT probe timed out", code = "probe_timeout", stage = "enforcement"
                )
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.wait(timeout = 5)
            output = exc.output or b""
            raise srt_adapter.SrtError(
                "SRT probe timed out after launch", code = "probe_timeout", stage = "enforcement"
            ) from exc
        except Exception:
            proc.kill()
            proc.wait(timeout = 5)
            raise
        finally:
            srt_adapter.release_control(proc)
        if (
            proc.returncode != 0
            or b"UNSLOTH_SRT_SUPPORTED_PROBE_OK" not in output
            or sentinel.read_text(encoding = "utf-8") != "unchanged"
        ):
            return False, ProbeReason("enforcement_failed", "enforcement")
        return (
            True,
            "Selected Python, Terminal child, workdir write and filesystem deny checks passed.",
        )
