# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Idle-only background compile: a render never waits for the per-block compile. The pause stops the compile thread
at once (mid-trace through sys.monitoring for a resident denoiser), back-to-back renders owe it nothing, and it resumes
only after the last render ended. CPU only: real dynamo traces (backend "eager") stand in for a U-Net block graph."""

from __future__ import annotations

import logging
import sys
import threading
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_bg_compile as bg  # noqa: E402
from core.inference import diffusion_speed as ds  # noqa: E402

LOG = logging.getLogger("test_bg_compile_idle")

needs_monitoring = pytest.mark.skipif(
    not hasattr(sys, "monitoring"), reason = "mid-trace freeze needs sys.monitoring (3.12+)"
)


@pytest.fixture(autouse = True)
def _idle_env(monkeypatch):
    for env in (bg.IDLE_ENV, bg.IDLE_GRACE_ENV, bg.FREEZE_ENV, bg.YIELD_ENV):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setenv(bg.IDLE_GRACE_ENV, "0.3")
    yield
    torch._dynamo.reset()


def _wait_for(pred, timeout = 30.0):
    t0 = time.time()
    while not pred():
        if time.time() - t0 > timeout:
            raise AssertionError("timed out")
        time.sleep(0.005)


class _LongBlock(torch.nn.Module):
    """A block whose dynamo trace takes about a second on CPU (many small ops), like a U-Net block's graph."""

    def __init__(self, ops: int) -> None:
        super().__init__()
        self.ops = ops
        self.lin = torch.nn.Linear(8, 8)
        self.traced = 0

    def forward(self, x):
        for i in range(self.ops):
            x = torch.tanh(self.lin(x) * (1.0 + i * 1e-4)) + x * 0.5
        return x


class _Net(torch.nn.Module):
    def __init__(
        self,
        n: int = 3,
        ops: int = 400,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(_LongBlock(ops) for _ in range(n))
        for blk in self.blocks:
            blk.compile(backend = "eager", fullgraph = True, dynamic = False)
        ds.guard_compiled_blocks(self, LOG)
        self._unsloth_bg_yield = True

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def _compiled_frames() -> int:
    from torch._dynamo.utils import counters
    return int(counters["stats"]["unique_graphs"])


def _armed(net):
    job = bg.arm(net, logger = LOG)
    assert job is not None and job.idle
    with torch.no_grad(), bg.force_eager():
        net(torch.randn(2, 8))  # records the sample; the guard routes the blocks eager
    assert len(job.samples) == 1
    return job


def _render(net):
    with torch.no_grad(), bg.force_eager():
        return net(torch.randn(2, 8))


def _in_dynamo(job) -> bool:
    frame = sys._current_frames().get(job._thread.ident)
    while frame is not None:
        if (frame.f_globals.get("__name__") or "").startswith("torch._dynamo."):
            return True
        frame = frame.f_back
    return False


def test_switches(monkeypatch):
    assert bg.idle_only_enabled() is True
    assert bg.idle_grace_seconds() == pytest.approx(0.3)
    monkeypatch.setenv(bg.IDLE_GRACE_ENV, "junk")
    assert bg.idle_grace_seconds() == pytest.approx(1.0)
    monkeypatch.setenv(bg.IDLE_GRACE_ENV, "1e9")
    assert bg.idle_grace_seconds() == pytest.approx(60.0)
    monkeypatch.setenv(bg.IDLE_ENV, "0")
    assert bg.idle_only_enabled() is False
    assert bg.freeze_enabled() is False
    monkeypatch.delenv(bg.IDLE_ENV)
    monkeypatch.setenv(bg.FREEZE_ENV, "0")
    assert bg.freeze_enabled() is False
    assert bg.BackgroundCompile(torch.nn.Linear(2, 2)).idle is True


@needs_monitoring
def test_a_render_freezes_the_compile_mid_trace_without_waiting():
    net = _Net(n = 3, ops = 500)
    job = _armed(net)
    assert job.kick()
    # past the kick's grace, inside the first block graph's trace
    _wait_for(lambda: _in_dynamo(job))
    time.sleep(0.05)
    graphs = _compiled_frames()
    waited, paused = job.pause(budget_s = 5.0, freeze = True)
    assert paused
    assert waited < 0.2, waited
    assert job._frozen and job.freezes == 1
    assert _in_dynamo(job)  # parked inside the trace, not at a block boundary
    assert sys.monitoring.get_events(bg._FREEZE_TOOL[0]) == 0
    t0 = time.perf_counter()
    _render(net)
    eager_s = time.perf_counter() - t0
    time.sleep(0.5)
    assert (
        _compiled_frames() == graphs and job.compiling()
    )  # no progress while the render holds the pause
    job.resume()
    _wait_for(lambda: job.state == "done")
    assert job.error is None
    assert _compiled_frames() >= graphs + 1
    assert eager_s < 5.0
    x = torch.randn(2, 8)
    with torch.no_grad():
        compiled = net(x)
        with bg.force_eager():
            eager = net(x)
    torch.testing.assert_close(compiled, eager)


@needs_monitoring
def test_resume_waits_for_the_idle_grace(monkeypatch):
    monkeypatch.setenv(bg.IDLE_GRACE_ENV, "0.8")
    net = _Net(n = 2, ops = 500)
    job = _armed(net)
    job.kick()
    _wait_for(lambda: _in_dynamo(job))
    _, paused = job.pause(freeze = True)
    assert paused
    _wait_for(lambda: job._frozen, timeout = 2.0)
    job.resume()
    t0 = time.perf_counter()
    time.sleep(0.4)
    assert job._frozen  # still inside the grace: a back-to-back client's next request arrives here
    _, paused = job.pause(freeze = True)  # the next render: nothing to wait for
    assert paused and time.perf_counter() - t0 < 0.6
    job.resume()
    _wait_for(lambda: job.state == "done")


def test_back_to_back_renders_never_wait_and_owe_nothing():
    """main: each resume owed one block graph, so a render waited for it (0.3-5 s on L4). Idle-only: never."""
    net = _FakeBlocks(n = 5, compile_s = 0.3)
    job = bg.arm(net, logger = LOG)
    with torch.no_grad(), bg.force_eager():
        net(torch.randn(2, 4))
    job.kick()
    time.sleep(0.4)  # past the kick's grace: first block graph in flight
    waits = []
    for _ in range(12):
        waited, paused = job.pause(freeze = False)
        assert paused
        waits.append(waited)
        time.sleep(0.05)
        job.resume()
        time.sleep(0.02)  # the client's turnaround, inside the grace
    # without freeze the first pause waits out the block graph in flight; after that nothing is owed
    assert waits[0] < 0.4 and max(waits[1:]) < 0.05, waits
    done_during_renders = len(net.compiled_blocks)
    assert done_during_renders <= 2  # the graph in flight at the first pause, then nothing
    _wait_for(lambda: job.state == "done")
    assert len(net.compiled_blocks) == 5


def test_without_freeze_the_compile_parks_at_the_next_block_boundary():
    net = _FakeBlocks(n = 6, compile_s = 0.2)
    job = bg.arm(net, logger = LOG)
    with torch.no_grad(), bg.force_eager():
        net(torch.randn(2, 4))
    job.kick()
    _wait_for(lambda: len(net.compiled_blocks) >= 1)
    waited, paused = job.pause(freeze = False)
    # unfrozen: pause returns only once parked, an offloaded block must not run beside the render
    assert paused and job._parked and waited < 1.0
    held = len(net.compiled_blocks)
    time.sleep(0.5)
    assert len(net.compiled_blocks) == held and not job._frozen
    job.resume()
    _wait_for(lambda: job.state == "done")


def test_close_ends_an_idle_parked_compile():
    net = _FakeBlocks(n = 6, compile_s = 0.15)
    job = bg.arm(net, logger = LOG)
    with torch.no_grad(), bg.force_eager():
        net(torch.randn(2, 4))
    job.kick()
    job.pause(freeze = False)
    _wait_for(lambda: job._parked, timeout = 2.0)
    thread = job._thread
    job.close(timeout = 5.0)
    assert not thread.is_alive()
    assert job.state == "failed" and "closed" in (job.error or "")


@needs_monitoring
def test_close_ends_a_frozen_compile():
    net = _Net(n = 3, ops = 500)
    job = _armed(net)
    job.kick()
    _wait_for(lambda: _in_dynamo(job))
    job.pause(freeze = True)
    _wait_for(lambda: job._frozen, timeout = 2.0)
    thread = job._thread
    job.close(timeout = 30.0)
    assert not thread.is_alive()
    assert job.state == "failed"


@needs_monitoring
def test_a_compile_elsewhere_thaws_a_frozen_compile_instead_of_deadlocking():
    """The frozen trace holds dynamo's compile_lock; a render that compiles something must not wait forever."""
    net = _Net(n = 2, ops = 500)
    job = _armed(net)
    job.kick()
    _wait_for(lambda: _in_dynamo(job))
    job.pause(freeze = True)
    _wait_for(lambda: job._frozen, timeout = 2.0)

    other = torch.compile(lambda t: t * 2 + 1, backend = "eager")
    out: list = []
    worker = threading.Thread(target = lambda: out.append(other(torch.ones(3))), daemon = True)
    worker.start()
    worker.join(30.0)
    assert not worker.is_alive(), "deadlock: the render's compile waited on the frozen trace"
    torch.testing.assert_close(out[0], torch.full((3,), 3.0))
    assert job.thaws == 1
    # it ran on only to the end of the frame that held the lock, then froze again while the render still runs
    _wait_for(lambda: job._frozen, timeout = 30.0)
    assert job.freezes >= 2
    job.resume()
    _wait_for(lambda: job.state == "done")
    assert job.error is None


def test_freeze_safe_refuses_a_lock_taking_frame():
    def inner():
        return sys._getframe()

    # no compile-thread entry below: never safe
    assert bg._freeze_safe(inner()) is False

    code = bg.BackgroundCompile._run.__code__
    assert code.co_name == "_run"

    class _F:
        def __init__(
            self,
            name,
            back,
            code = None,
        ):
            self.f_globals = {"__name__": name}
            self.f_back = back
            self.f_code = code if code is not None else inner.__code__

    run = _F(bg.__name__, None, code)
    unet = _F("diffusers.models.unets.unet_2d_condition", run)
    trace = _F("torch._dynamo.symbolic_convert", _F("torch._dynamo.convert_frame", unet))
    assert bg._freeze_safe(trace) is True
    assert bg._freeze_safe(unet) is False  # outside the compile: Studio / diffusers eager code
    assert bg._freeze_safe(_F("logging", trace)) is False
    assert bg._freeze_safe(_F("logging.handlers", trace)) is False
    assert bg._freeze_safe(_F("torch._inductor.compile_worker.subproc_pool", trace)) is False
    # handing kernels to the pool holds no lock: a refusal there kept the compile beside the render for 0.7-1.6 s (L4)
    assert bg._freeze_safe(_F("torch._inductor.async_compile", trace)) is True
    assert bg._freeze_safe(_F("core.inference.diffusion_compile_config", trace)) is False


_HELD_LOCK = threading.Lock()


def _studio_like(record):
    record.append(sys._getframe().f_lineno)
    with _HELD_LOCK:
        record.append(sys._getframe().f_lineno)
    record.append(sys._getframe().f_lineno)


def test_a_studio_frame_is_refused_only_inside_its_lock_block():
    """Refusing every Studio frame kept the compile beside the render (A100: 30-290 s); only a frame in a lock is."""
    lines: list = []
    _studio_like(lines)
    before, held, after = lines
    code = _studio_like.__code__

    class _F:
        def __init__(self, line):
            self.f_globals = {"__name__": "core.inference.some_module"}
            self.f_code = code
            self.f_lineno = line
            self.f_back = None

    assert bg._frame_refusal(_F(before)) is None
    assert bg._frame_refusal(_F(after)) is None
    assert bg._frame_refusal(_F(held)) == "core.inference.some_module:_studio_like"
    # the real wrapper on the replay path: eager call outside its capture locks
    from core.inference import diffusion_cuda_graph as cg

    assert bg._lock_spans(cg.GraphedForward._eager.__code__) == ()
    assert bg._lock_spans(cg.run_if_capture_idle.__code__) is True


def test_freeze_refusal_memo_matches_a_fresh_walk():
    def leaf(memo):
        return bg._freeze_refusal(sys._getframe(), memo), bg._freeze_refusal(sys._getframe())

    memo: dict = {}
    first = leaf(memo)
    again = leaf(memo)
    assert first[0] == first[1] == again[0] == again[1] == "not the compile thread"
    assert memo


def test_compile_lock_proxy_is_the_same_lock():
    from torch._dynamo import convert_frame

    lock = convert_frame.compile_lock
    if not getattr(lock, "_unsloth_bg_proxy", False):
        assert bg._install_compile_lock_proxy()
    proxy = convert_frame.compile_lock
    assert proxy._unsloth_bg_proxy and bg._install_compile_lock_proxy()
    with proxy:
        with proxy:  # reentrant, like the RLock it wraps
            assert proxy._is_owned()
    got = []
    with proxy:
        t = threading.Thread(target = lambda: got.append(proxy.acquire(False)))
        t.start()
        t.join()
    assert got == [False]


class _FakeBg:
    def __init__(self) -> None:
        self.idle = True
        self.calls: list = []
        self._freeze_ok = False
        self._frozen = False
        self._parked = False

    def compiling(self):
        return True

    def yields(self):
        return True

    def pause(
        self,
        cancel = None,
        budget_s = 0.0,
        freeze = False,
    ):
        self.calls.append(("pause", budget_s, freeze))
        return 0.0, True

    def wait(self, cancel = None):
        self.calls.append(("wait",))
        return 9.0


@pytest.mark.parametrize("policy, freeze", [("none", True), ("group", False), ("model", False)])
def test_render_stops_an_idle_compile_without_a_slice(policy, freeze):
    from core.inference import diffusion as d

    fake = _FakeBg()
    pipe = SimpleNamespace(_unsloth_loras = (), _unsloth_eager_render_s = 9.9)
    state = SimpleNamespace(pipe = pipe, speed_optims = (), offload_policy = policy)
    assert d.OFFLOAD_NONE == "none"
    assert d._wait_or_pause_bg_compile(fake, state, None, threading.Event()) is fake
    assert fake.calls == [("pause", 0.0, freeze)]


def test_lora_render_still_waits_for_an_idle_compile():
    from core.inference import diffusion as d

    fake = _FakeBg()
    pipe = SimpleNamespace(_unsloth_loras = (), _unsloth_eager_render_s = 9.9)
    state = SimpleNamespace(pipe = pipe, speed_optims = (), offload_policy = "none")
    assert d._wait_or_pause_bg_compile(fake, state, [("a/lora", 1.0)], threading.Event()) is None
    assert fake.calls == [("wait",)]


class _FakeBlocks(torch.nn.Module):
    """Guarded blocks whose first compiled call sleeps ``compile_s`` (a block graph's compile)."""

    def __init__(self, n: int, compile_s: float) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(n))
        self.compiled_blocks: set = set()
        for i, blk in enumerate(self.blocks):
            blk._compiled_call_impl = self._fake(i, blk, compile_s)
        ds.guard_compiled_blocks(self, LOG)
        self._unsloth_bg_yield = True

    def _fake(self, index, blk, compile_s):
        def compiled(*args, **kwargs):
            if index not in self.compiled_blocks:
                time.sleep(compile_s)
                self.compiled_blocks.add(index)
            return blk.forward(*args, **kwargs)

        return compiled

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
