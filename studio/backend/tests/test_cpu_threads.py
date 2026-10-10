# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for Unsloth's early CPU thread-pool configuration."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import utils.cpu_threads as cpu_threads
from utils.cpu_threads import (
    _THREAD_POOL_ENV_VARS,
    configure_cpu_threads,
    default_openblas_threads,
)


_BACKEND_DIR = Path(__file__).resolve().parent.parent
_RUN_PY = _BACKEND_DIR / "run.py"
_MAIN_PY = _BACKEND_DIR / "main.py"


# Explicit positive integers seed all four native pool env vars.
def test_cpu_thread_cap_seeds_native_pool_limits():
    env = {"UNSLOTH_CPU_THREADS": " 6 "}

    configure_cpu_threads(env)

    assert {variable: env[variable] for variable in _THREAD_POOL_ENV_VARS} == {
        variable: "6" for variable in _THREAD_POOL_ENV_VARS
    }


# Explicit per-library values win over the Unsloth knob via setdefault.
def test_cpu_thread_cap_preserves_runtime_specific_override():
    env = {"UNSLOTH_CPU_THREADS": "4", "OMP_NUM_THREADS": "2"}

    configure_cpu_threads(env)

    assert env["OMP_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "4"


# Whitespace / plus-prefix / leading zero all normalise via int().
@pytest.mark.parametrize("raw", ["+4", "007", "  4  "])
def test_cpu_thread_cap_normalises_valid_inputs(raw):
    env = {"UNSLOTH_CPU_THREADS": raw}

    configure_cpu_threads(env)

    assert env["OMP_NUM_THREADS"] == str(int(raw.strip()))


@pytest.mark.parametrize("raw", [None, "", "   ", "\t"])
def test_cpu_thread_cap_unset_limits_only_openblas(raw, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.setattr(cpu_threads, "_openblas_memory_headroom", lambda: None)
    env = {} if raw is None else {"UNSLOTH_CPU_THREADS": raw}
    snapshot = dict(env)

    configure_cpu_threads(env)

    assert env == {**snapshot, "OPENBLAS_NUM_THREADS": "8"}


# About one thread per physical core, at most 8: numpy stays fast and OpenBLAS's per-thread buffers stay bounded.
@pytest.mark.parametrize(
    "cpus, expected", [(None, 1), (1, 1), (2, 1), (4, 2), (12, 6), (16, 8), (24, 8), (192, 8)]
)
def test_openblas_default_scales_with_cores_and_is_capped(cpus, expected, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(cpu_threads, "_openblas_memory_headroom", lambda: None)

    assert default_openblas_threads() == expected


# A tenth of the memory OpenBLAS's buffers draw on, about 32 MB per thread: a host short of it starts on fewer threads.
@pytest.mark.parametrize(
    "headroom_mb, expected",
    [(None, 8), (64 << 10, 8), (2600, 8), (1280, 4), (640, 2), (320, 1), (100, 1), (0, 1)],
)
def test_openblas_default_shrinks_when_memory_is_short(headroom_mb, expected, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.setattr(
        cpu_threads,
        "_openblas_memory_headroom",
        lambda: None if headroom_mb is None else headroom_mb << 20,
    )

    assert default_openblas_threads() == expected


def test_an_unreadable_memory_reading_keeps_the_core_count_default(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    def boom():
        raise OSError("no reading")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cpu_threads, "_address_space_headroom", boom)

    assert cpu_threads._openblas_memory_headroom() is None
    assert default_openblas_threads() == 8


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason = "address-space rlimit is read on Linux"
)
def test_linux_reads_the_address_space_rlimit(monkeypatch):
    import resource

    monkeypatch.setattr(
        resource, "getrlimit", lambda which: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )
    assert cpu_threads._address_space_headroom() is None
    monkeypatch.setattr(resource, "getrlimit", lambda which: (1 << 40, 1 << 40))
    assert 0 < cpu_threads._address_space_headroom() < 1 << 40


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows commit and job limits")
def test_windows_reads_the_free_commit():
    headroom = cpu_threads._windows_commit_headroom()

    assert headroom is not None and headroom > 0


def test_openblas_default_keeps_user_value():
    env = {"OPENBLAS_NUM_THREADS": "8"}

    configure_cpu_threads(env)

    assert env == {"OPENBLAS_NUM_THREADS": "8"}


@pytest.mark.parametrize("raw", ["", "  "])
def test_openblas_default_replaces_blank_value(raw, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(cpu_threads, "_openblas_memory_headroom", lambda: None)
    env = {"OPENBLAS_NUM_THREADS": raw}

    configure_cpu_threads(env)

    assert env == {"OPENBLAS_NUM_THREADS": "4"}


# Anything that is not a positive integer raises a clear ValueError.
@pytest.mark.parametrize("raw", ["zero", "0", "-3", "1.5", "abc", "8a", "0x4", "1e3", "4 0"])
def test_cpu_thread_cap_requires_positive_integer(raw):
    with pytest.raises(ValueError, match = "must be a positive integer"):
        configure_cpu_threads({"UNSLOTH_CPU_THREADS": raw})


# env=None path uses real os.environ (production call from run.py / main.py).
def test_cpu_thread_cap_uses_os_environ_when_env_is_none(monkeypatch):
    for variable in (*_THREAD_POOL_ENV_VARS, "UNSLOTH_CPU_THREADS"):
        monkeypatch.delenv(variable, raising = False)
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "3")

    configure_cpu_threads()

    for variable in _THREAD_POOL_ENV_VARS:
        assert os.environ[variable] == "3"


# Calling twice must not flip any seeded value.
def test_cpu_thread_cap_idempotent(monkeypatch):
    for variable in (*_THREAD_POOL_ENV_VARS, "UNSLOTH_CPU_THREADS"):
        monkeypatch.delenv(variable, raising = False)
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "5")

    configure_cpu_threads()
    snapshot = {v: os.environ.get(v) for v in _THREAD_POOL_ENV_VARS}
    configure_cpu_threads()

    assert {v: os.environ.get(v) for v in _THREAD_POOL_ENV_VARS} == snapshot


def _ast_line_of_configure_call(source: str) -> int:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "configure_cpu_threads"
        ):
            return node.lineno
    raise AssertionError("configure_cpu_threads() call not found")


def _ast_line_of_platform_compat_import(source: str) -> int:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "_platform_compat":
                    return node.lineno
    raise AssertionError("_platform_compat import not found")


# AST ordering: configure_cpu_threads() must precede _platform_compat in both
# run.py and main.py. Robust to formatting / line shifts.
@pytest.mark.parametrize("entry_point", [_RUN_PY, _MAIN_PY])
def test_cpu_thread_configuration_runs_before_backend_imports(entry_point):
    source = entry_point.read_text(encoding = "utf-8")
    call_line = _ast_line_of_configure_call(source)
    compat_line = _ast_line_of_platform_compat_import(source)
    assert call_line < compat_line, (
        f"{entry_point.name}: configure_cpu_threads() (line {call_line}) "
        f"must precede import _platform_compat (line {compat_line})"
    )


# Invalid env -> exit 1, one-line stderr, no traceback, gated before any
# heavy import. Parametrised over both entry points.
@pytest.mark.parametrize("entry_point", [_RUN_PY, _MAIN_PY])
def test_invalid_cpu_thread_cap_exits_without_traceback(entry_point):
    env = os.environ.copy()
    env["UNSLOTH_CPU_THREADS"] = "not-a-count"

    result = subprocess.run(
        [sys.executable, str(entry_point)],
        env = env,
        capture_output = True,
        text = True,
    )

    assert result.returncode == 1
    assert (
        "Error: Invalid UNSLOTH_CPU_THREADS value 'not-a-count': "
        "UNSLOTH_CPU_THREADS must be a positive integer"
    ) in result.stderr
    assert "Traceback" not in result.stderr
    assert "_platform_compat" not in result.stderr
