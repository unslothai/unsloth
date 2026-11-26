# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The background compile must not run inductor's first fx trace while a render calls modules on another thread.

``InterpreterShim._dummy_gm()`` is a cached ``torch.fx.symbolic_trace``: while it traces, ``nn.Module.__call__`` is
patched process-wide, so a module called on the render thread at that moment goes through the tracer and fails.
The race is made deterministic by holding the trace open (a blocking ``identity``) until the render ran. CPU only."""

from __future__ import annotations

import sys
import threading

import pytest

torch = pytest.importorskip("torch")
loop_body = pytest.importorskip("torch._inductor.loop_body")

from core.inference import diffusion_bg_compile as bg  # noqa: E402

needs_shim = pytest.mark.skipif(
    not callable(getattr(getattr(loop_body, "InterpreterShim", None), "_dummy_gm", None)),
    reason = "this torch has no InterpreterShim._dummy_gm",
)


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.setenv(bg.IDLE_ENV, "0")
    shim = getattr(loop_body, "InterpreterShim", None)
    fn = getattr(shim, "_dummy_gm", None)
    clear = getattr(fn, "cache_clear", None)
    if clear is not None:
        clear()
    yield
    if clear is not None:
        clear()


def _job_whose_compile_builds_the_shim():
    job = bg.BackgroundCompile(torch.nn.Linear(2, 2))
    job.samples = [((), {})]
    done = threading.Event()

    def run() -> None:
        try:
            loop_body.InterpreterShim._dummy_gm()  # what inductor's lowering does on the compile thread
        finally:
            done.set()

    job._run = run
    return job, done


@needs_shim
def test_render_during_first_fx_trace_is_not_hijacked(monkeypatch):
    main = threading.current_thread()
    inside = threading.Event()
    rendered = threading.Event()
    real_identity = loop_body.identity

    def blocking_identity(x):
        # Traced on the compile thread: hold the nn.Module patch open until the render ran.
        if threading.current_thread() is not main:
            inside.set()
            rendered.wait(10)
        return real_identity(x)

    monkeypatch.setattr(loop_body, "identity", blocking_identity)
    job, done = _job_whose_compile_builds_the_shim()
    assert job.kick()
    while not (inside.is_set() or done.is_set()):
        inside.wait(0.01)
    try:
        out = torch.nn.Linear(2, 2)(torch.ones(1, 2))  # the render, on the main thread
    finally:
        rendered.set()
        job._thread.join(10)
    assert isinstance(out, torch.Tensor)
    assert not inside.is_set(), "the compile thread ran the first fx trace"
    assert loop_body.InterpreterShim._dummy_gm.cache_info().currsize == 1


@needs_shim
def test_deferred_prewarm_fills_the_shim_before_its_thread(monkeypatch):
    calls = []
    monkeypatch.setenv(bg.PREWARM_ENV, "1")
    monkeypatch.setattr(bg, "_PREWARM_THREAD", None)
    for name in (
        "warm_compile_pool",
        "warm_triton_key",
        "warm_inductor_key",
        "warm_engage_imports",
    ):
        monkeypatch.setattr(bg, name, lambda *a, **k: 0)
    real_start = threading.Thread.start

    def start(self):
        calls.append(loop_body.InterpreterShim._dummy_gm.cache_info().currsize)
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    assert bg.start_deferred_prewarm()
    assert bg.wait_prewarm(10)
    assert calls == [1]


def test_warmup_is_a_noop_without_the_attribute(monkeypatch):
    job, done = _job_whose_compile_builds_the_shim()
    monkeypatch.setattr(loop_body, "InterpreterShim", type("InterpreterShim", (), {}))
    job._run = done.set
    assert bg.warm_fx_interpreter_shim() is False
    assert job.kick()
    assert done.wait(10)
    monkeypatch.setattr(loop_body, "InterpreterShim", None)
    assert bg.warm_fx_interpreter_shim() is False


def test_prewarm_site_never_imports_loop_body(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch._inductor.loop_body")
    assert bg.warm_fx_interpreter_shim(import_missing = False) is False
    assert "torch._inductor.loop_body" not in sys.modules
