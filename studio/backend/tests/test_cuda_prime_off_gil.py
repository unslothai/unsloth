# SPDX-License-Identifier: AGPL-3.0-only
"""Hardware detection must not hold the GIL across the CUDA driver init.

torch.cuda.is_available() reaches cuInit inside cudaGetDeviceCount with the GIL held, for seconds on a
multi-GPU host, so every other thread (the event loop answering /images/load-progress included) froze
for that long during the first load. Detection now runs cuInit through ctypes first, which releases
the GIL, and torch's own call then returns at once."""

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import cuda_driver_prime as prime
from utils.hardware import hardware as hw

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse = True)
def isolated(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for name in ("PYTORCH_NVML_BASED_CUDA_CHECK", "UNSLOTH_STUDIO_DISABLE_CUDA_PRIME"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    prime._reset_for_tests()
    yield
    prime._reset_for_tests()


class _FakeCuInit:
    """Stands in for a ctypes foreign function (argtypes / restype are set on it)."""

    def __init__(
        self,
        calls,
        gate = None,
    ):
        self.calls = calls
        self.gate = gate

    def __call__(self, flags):
        if self.gate is not None:
            self.gate.wait(5)
        self.calls.append(("cuInit", flags))
        return 0


class _FakeLib:
    def __init__(
        self,
        calls,
        gate = None,
    ):
        self.cuInit = _FakeCuInit(calls, gate)


def _fake_torch(
    calls,
    *,
    cuda = "13.0",
    hip = None,
    initialized = False,
):
    def is_available():
        calls.append(("is_available",))
        return True

    return SimpleNamespace(
        version = SimpleNamespace(cuda = cuda, hip = hip),
        cuda = SimpleNamespace(is_available = is_available, is_initialized = lambda: initialized),
    )


@pytest.fixture
def fake_cdll(monkeypatch):
    import ctypes

    calls, opened, gate = [], [], []

    def cdll(name):
        opened.append(name)
        return _FakeLib(calls, gate[0] if gate else None)

    monkeypatch.setattr(ctypes, "CDLL", cdll)
    monkeypatch.setattr(prime.sys, "platform", "linux")
    return calls, opened, gate


def test_prime_runs_cuinit_once(fake_cdll):
    calls, opened, _gate = fake_cdll
    torch = _fake_torch(calls)
    assert prime.prime_cuda_driver(torch) is True
    assert calls == [("cuInit", 0)]
    assert opened == ["libcuda.so.1"]
    # Once per process.
    assert prime.prime_cuda_driver(torch) is False
    assert calls == [("cuInit", 0)]


@pytest.mark.parametrize(
    "torch_kw, env",
    [
        ({"cuda": None}, {}),  # CPU build
        ({"hip": "7.1"}, {}),  # ROCm build
        ({"initialized": True}, {}),  # torch already initialised CUDA
        ({}, {"PYTORCH_NVML_BASED_CUDA_CHECK": "1"}),  # user opted out of cuInit here
        ({}, {"UNSLOTH_STUDIO_DISABLE_CUDA_PRIME": "1"}),  # kill switch
        ({}, {"CUDA_VISIBLE_DEVICES": ""}),  # CUDA hidden
        ({}, {"CUDA_VISIBLE_DEVICES": "-1"}),
    ],
)
def test_prime_skips(fake_cdll, monkeypatch, torch_kw, env):
    calls, opened, _gate = fake_cdll
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert prime.prime_cuda_driver(_fake_torch(calls, **torch_kw)) is False
    assert calls == [] and opened == []


def test_prime_skips_macos(fake_cdll, monkeypatch):
    calls, opened, _gate = fake_cdll
    monkeypatch.setattr(prime.sys, "platform", "darwin")
    assert prime.prime_cuda_driver(_fake_torch(calls)) is False
    assert opened == []


def test_prime_never_imports_torch(fake_cdll, monkeypatch):
    calls, opened, _gate = fake_cdll
    monkeypatch.delitem(sys.modules, "torch", raising = False)
    assert prime.prime_cuda_driver() is False
    assert opened == [] and "torch" not in sys.modules


def test_prime_never_raises_without_a_driver(monkeypatch):
    import ctypes

    def cdll(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", cdll)
    monkeypatch.setattr(prime.sys, "platform", "linux")
    assert prime.prime_cuda_driver(_fake_torch([])) is False


def test_concurrent_caller_waits_for_the_init_in_flight(fake_cdll):
    """A second thread must not fall through to torch's GIL-holding probe while the init runs."""
    calls, _opened, gate = fake_cdll
    release = threading.Event()
    gate.append(release)
    torch = _fake_torch(calls)
    first = threading.Thread(target = prime.prime_cuda_driver, args = (torch,))
    first.start()
    while not prime._claimed:
        time.sleep(0.001)
    second_done = threading.Event()

    def second():
        prime.prime_cuda_driver(torch)
        torch.cuda.is_available()
        second_done.set()

    t2 = threading.Thread(target = second)
    t2.start()
    time.sleep(0.1)
    assert not second_done.is_set() and calls == []
    release.set()
    first.join(5)
    t2.join(5)
    assert calls == [("cuInit", 0), ("is_available",)]


@pytest.mark.skipif(not hasattr(os, "fork"), reason = "needs os.fork")
def test_a_child_forked_during_the_init_does_not_wait_for_it(fake_cdll):
    calls, _opened, gate = fake_cdll
    release = threading.Event()
    gate.append(release)
    torch = _fake_torch(calls)
    first = threading.Thread(target = prime.prime_cuda_driver, args = (torch,))
    first.start()
    while not prime._claimed:
        time.sleep(0.001)
    try:
        pid = os.fork()
        if pid == 0:  # child: the parent's init thread is gone; this must return at once
            started = time.monotonic()
            ran = prime.prime_cuda_driver(torch)
            os._exit(0 if (not ran and time.monotonic() - started < 2.0) else 1)
        deadline = time.monotonic() + 10
        status = None
        while time.monotonic() < deadline:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            time.sleep(0.05)
        else:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            pytest.fail("the forked child waited on the parent's cuInit")
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        release.set()
        first.join(5)


def test_detection_primes_before_torch_probe(fake_cdll, monkeypatch):
    calls, _opened, _gate = fake_cdll
    torch = _fake_torch(calls)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(hw, "_has_torch", lambda: True)
    monkeypatch.setattr(hw, "_ze_affinity_mask", lambda: None)
    seen = []

    def is_available():
        seen.append(list(calls))
        raise RuntimeError("stop")  # only the order up to the probe matters

    torch.cuda.is_available = is_available
    with hw._DETECT_LOCK:
        try:
            hw._detect_hardware_locked()
        except Exception:
            pass
    assert seen and seen[0] == [("cuInit", 0)]


def test_diffusers_import_window_primes(fake_cdll, monkeypatch):
    from utils import torch_warmup

    calls, _opened, _gate = fake_cdll
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls))
    monkeypatch.setattr(torch_warmup, "ensure_dynamo_imported", lambda **_k: True)
    monkeypatch.setattr(torch_warmup, "claim_media_import_window", lambda _log: None)
    assert torch_warmup.close_dynamo_import_window(torch_warmup.logger) is True
    assert calls == [("cuInit", 0)]


def test_torch_stack_gate_primes(fake_cdll, monkeypatch):
    from utils import torch_warmup

    calls, _opened, _gate = fake_cdll
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls))
    monkeypatch.setattr(torch_warmup, "_dynamo_done", True)
    assert torch_warmup.gate_torch_stack_import("test") is True
    assert calls == [("cuInit", 0)]


def test_torch_stack_gate_kill_switch_still_primes(fake_cdll, monkeypatch):
    from utils import torch_warmup

    calls, _opened, _gate = fake_cdll
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls))
    monkeypatch.setattr(torch_warmup, "_dynamo_done", False)
    monkeypatch.setenv(torch_warmup.DYNAMO_GATE_DISABLE_ENV_VAR, "1")
    assert torch_warmup.gate_torch_stack_import("test") is False
    assert calls == [("cuInit", 0)]


_GIL_PROBE = textwrap.dedent(
    """
    import sys, threading, time
    sys.path.insert(0, sys.argv[1])
    import torch
    from utils.hardware import hardware as hw
    gaps, stop = [], threading.Event()
    def tick():
        last = time.perf_counter()
        while not stop.is_set():
            time.sleep(0.005)
            now = time.perf_counter(); gaps.append(now - last); last = now
    t = threading.Thread(target = tick, daemon = True); t.start(); time.sleep(0.2)
    with hw._DETECT_LOCK:
        hw._detect_hardware_locked()
    torch.cuda.is_available()
    stop.set(); t.join()
    print(f"MAXGAP {max(gaps):.3f} {torch.cuda.is_initialized()}")
    """
)


def _real_cuda_host() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "x").strip() in ("", "-1"):
        return False
    try:
        import ctypes

        import torch

        if not torch.version.cuda or getattr(torch.version, "hip", None):
            return False
        ctypes.CDLL("libcuda.so.1")
    except Exception:
        return False
    return sys.platform.startswith("linux")


@pytest.mark.skipif(not _real_cuda_host(), reason = "needs an NVIDIA driver and a CUDA torch")
def test_real_detection_keeps_other_threads_running():
    """A fresh process (cold driver) runs detection while a 5 ms ticker thread measures its largest gap.
    Before: the gap is the whole cuInit (0.4-6.8 s measured on B200 hosts). After: under 20 ms."""
    out = subprocess.run(
        [sys.executable, "-c", _GIL_PROBE, str(BACKEND)],
        capture_output = True,
        text = True,
        timeout = 300,
        env = {**os.environ, "UNSLOTH_STUDIO_DISABLE_CUDA_PRIME": ""},
    )
    line = [l for l in out.stdout.splitlines() if l.startswith("MAXGAP")]
    assert line, out.stdout[-2000:] + out.stderr[-2000:]
    gap = float(line[-1].split()[1])
    assert gap < 0.25, f"another thread was frozen {gap:.2f}s during hardware detection"
