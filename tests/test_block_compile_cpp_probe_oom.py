# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""An OOM wrapped by the compiler while the C++ wrapper toolchain probe builds is memory pressure, not a
broken toolchain: the block class retries the probe later instead of keeping the Python wrapper for the
rest of the process. A probe that keeps running out of memory, or a real toolchain failure, still turns the
C++ wrapper off. CPU only: torch.compile is faked and the module is loaded by file path."""

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_PATH = os.path.join(os.path.dirname(__file__), "..", "unsloth", "models", "_block_compile.py")


@pytest.fixture(autouse = True)
def _block_env(monkeypatch):
    for key in (
        "UNSLOTH_BLOCK_COMPILE",
        "UNSLOTH_COMPILE_DISABLE",
        "UNSLOTH_FORCE_FLOAT32",
        "UNSLOTH_BLOCK_CPP_WRAPPER",
        "UNSLOTH_BLOCK_CPP_WRAPPER_AFTER",
    ):
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", "0")


def _load():
    spec = importlib.util.spec_from_file_location("_u_block_compile_probe_oom", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod._install_cpp_wrapper_param_fix = lambda: True
    mod._cpp_wrapper_steps_left = lambda: 10**6
    mod._CPP_WRAPPER_REASON = None
    mod._CPP_WRAPPER_AFTER = 0
    return mod


class BackendCompilerFailed(Exception):
    """Named like torch's: _backend_compile_failed matches the class name in the MRO."""


def _wrapped_oom():
    e = BackendCompilerFailed("probe compile ran out of memory")
    e.inner_exception = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    return e


def _fake_compile(calls, probe_failures):
    """probe_failures: exceptions the toolchain probe raises on its first runs, in order."""
    pending = list(probe_failures)

    def fake_compile(fn, **kwargs):
        cpp = bool((kwargs.get("options") or {}).get("cpp_wrapper"))
        probe = getattr(fn, "__name__", "") == "_cpp_wrapper_probe_fn"

        def compiled(*args, **kw):
            if probe:
                calls.append("probe")
                if pending:
                    raise pending.pop(0)
            else:
                calls.append("cpp" if cpp else "python")
            return fn(*args, **kw)

        return compiled

    return fake_compile


def _run(M, monkeypatch, failures, steps):
    calls = []
    monkeypatch.setattr(torch, "compile", _fake_compile(calls, failures))
    cls = type("ToyDecoderLayer", (torch.nn.Module,), {})

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.proj = torch.nn.Linear(8, 8)

    cls.__init__ = __init__
    cls.forward = lambda self, hidden_states, attention_mask = None: (
        self.proj(hidden_states) + hidden_states
    )
    M._wrap(cls, dict(M._COMPILE_OPTIONS))
    layer = cls()
    with torch.no_grad():
        for _ in range(steps):
            layer(torch.randn(2, 4, 8))
    return calls


def test_one_wrapped_oom_in_the_probe_does_not_latch_the_cpp_wrapper_off(monkeypatch):
    M = _load()
    calls = _run(M, monkeypatch, [_wrapped_oom()], steps = 4)
    assert calls.count("probe") >= 2, calls  # probed again once memory allowed
    assert "cpp" in calls, calls  # the C++ region is built and used
    assert M.CPP_WRAPPER_STATS["regions"] == 1
    assert not M._CPP_WRAPPER_BROKEN


def test_a_probe_that_keeps_running_out_of_memory_stops_retrying(monkeypatch):
    M = _load()
    calls = _run(M, monkeypatch, [_wrapped_oom() for _ in range(50)], steps = 30)
    assert 2 <= calls.count("probe") < 30, calls  # bounded retries, then the Python wrapper
    assert "cpp" not in calls and calls[-1] == "python", calls
    assert M.CPP_WRAPPER_STATS["regions"] == 0


def test_a_broken_toolchain_still_keeps_the_python_wrapper(monkeypatch):
    M = _load()
    calls = _run(
        M, monkeypatch, [BackendCompilerFailed("C++ compile error: g++ not found")], steps = 4
    )
    assert calls.count("probe") == 1, calls
    assert "cpp" not in calls, calls
    assert M._CPP_WRAPPER_BROKEN and M.CPP_WRAPPER_STATS["fallbacks"] == 1
