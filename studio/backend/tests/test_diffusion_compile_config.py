# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compile knobs set at load must reach the thread that compiles. torch 2.12+ scopes every dynamo / inductor config
write to the writing thread's context, and Studio loads on one thread and renders (compiles) on another. Hermetic: no
GPU, no compile; torch is imported only for its config modules."""

from __future__ import annotations

import sys
import threading
import types

import pytest

torch = pytest.importorskip("torch")
import torch._dynamo.config  # noqa: E402
import torch._inductor.config  # noqa: E402

from core.inference import diffusion_compile_config as cc  # noqa: E402
from core.inference import diffusion_render_thread as rt  # noqa: E402
from core.inference import diffusion_speed as ds  # noqa: E402

_DYNAMO = torch._dynamo.config
_INDUCTOR = torch._inductor.config
_LIMIT = "recompile_limit" if hasattr(_DYNAMO, "recompile_limit") else "cache_size_limit"


@pytest.fixture(autouse = True)
def _clean_knobs():
    before = (getattr(_DYNAMO, _LIMIT), _INDUCTOR.emulate_precision_casts)
    cc._reset_for_tests()
    yield
    cc._reset_for_tests()
    setattr(_DYNAMO, _LIMIT, before[0])
    _INDUCTOR.emulate_precision_casts = before[1]


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setattr(rt, "enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: None)


def _on_thread(fn):
    out = {}

    def body():
        try:
            out["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the test
            out["error"] = exc

    t = threading.Thread(target = body)
    t.start()
    t.join()
    if "error" in out:
        raise out["error"]
    return out["value"]


def _read():
    return getattr(_DYNAMO, _LIMIT), bool(_INDUCTOR.emulate_precision_casts)


class _Transformer:
    _repeated_blocks = ()

    def compile_repeated_blocks(self, **kwargs):
        self.kwargs = kwargs

    def named_modules(self):
        return iter(())

    def modules(self):
        return iter(())

    def parameters(self):
        return iter(())


def _load_compile():
    pipe = types.SimpleNamespace(transformer = _Transformer())
    assert ds._compile_repeated_blocks(pipe, None) is True


def _baseline():
    """What a thread that never saw the load reads: the torch defaults, unless this torch has no per-thread config."""
    return _on_thread(_read)


def test_load_thread_knobs_reach_the_render_thread(armed):
    default_limit, _ = _baseline()
    _on_thread(_load_compile)
    # The caller of generate is yet another thread; rt.run copies ITS context onto the render thread.
    seen = _on_thread(lambda: rt.run("cfg1", _read))
    assert seen == (max(default_limit, 64), True)


def test_render_inline_path_applies_too(monkeypatch):
    # Render thread disabled (ROCm, env opt-out): fn runs on the caller, which must still see the knobs.
    monkeypatch.setattr(rt, "enabled", lambda: False)
    default_limit, _ = _baseline()
    _on_thread(_load_compile)
    assert _on_thread(lambda: rt.run("cfg2", _read)) == (max(default_limit, 64), True)


def test_guarded_compiled_block_applies_the_knobs():
    # Any thread that reaches a compiled block (not just rt.run) compiles with the load-time knobs.
    default_limit, _ = _baseline()
    _on_thread(_load_compile)
    owner = types.SimpleNamespace()
    guard = ds._CompileGuard(None)
    guarded = guard.wrap(lambda: _read(), lambda: ("eager",), owner)
    assert _on_thread(guarded) == (max(default_limit, 64), True)


def test_unload_restore_reaches_the_render_thread(armed):
    snap = _on_thread(ds.snapshot_backend_flags)
    assert snap["inductor_emulate_precision_casts"] is False
    _on_thread(_load_compile)
    assert _on_thread(lambda: rt.run("cfg3", _read))[1] is True
    _on_thread(lambda: ds.restore_backend_flags(snap))
    limit, emulate = _on_thread(lambda: rt.run("cfg3", _read))
    assert emulate is False
    # The recompile limit is not part of the unload snapshot (unchanged behaviour): it stays raised.
    assert limit >= 64


def test_snapshot_reads_the_process_value_not_this_threads_view():
    # A second backend loading on a fresh thread must snapshot what the first load set, else its unload would undo it.
    _on_thread(_load_compile)
    assert _on_thread(ds.snapshot_backend_flags)["inductor_emulate_precision_casts"] is True


def test_apply_is_idempotent_per_context(monkeypatch):
    cc.set_knob("torch._inductor.config", "emulate_precision_casts", True)
    writes = []
    real_write = cc._write
    monkeypatch.setattr(cc, "_write", lambda *a: (writes.append(a), real_write(*a)))

    def twice():
        cc.apply()
        cc.apply()
        return bool(_INDUCTOR.emulate_precision_casts)

    assert _on_thread(twice) is True
    assert len(writes) == 1


def test_missing_knob_is_not_recorded():
    assert cc.set_knob("torch._inductor.config", "no_such_knob_anywhere", True) is False
    assert not cc.is_recorded("torch._inductor.config", "no_such_knob_anywhere")
    assert cc.set_knob("torch._no_such_module.config", "x", 1) is False
    cc.apply()


def test_process_global_config_without_contextvar(monkeypatch):
    # torch before 2.12: config is a plain process-wide object; set_knob writes it, apply() then has nothing to do.
    fake_cfg = types.SimpleNamespace(emulate_precision_casts = False)
    fake_torch = types.SimpleNamespace(_inductor = types.SimpleNamespace(config = fake_cfg))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert cc.set_knob("torch._inductor.config", "emulate_precision_casts", True) is True
    assert fake_cfg.emulate_precision_casts is True
    assert _on_thread(lambda: (cc.apply(), fake_cfg.emulate_precision_casts)[1]) is True
    # An outside writer after the record: apply() re-asserts once per fresh context.
    fake_cfg.emulate_precision_casts = False
    assert _on_thread(lambda: (cc.apply(), fake_cfg.emulate_precision_casts)[1]) is True
