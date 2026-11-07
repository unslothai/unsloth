# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An image load returns while its encoder pins finish; streamed encoders pin beside the transformer's pin; the first
render waits for them. ``UNSLOTH_DIFFUSION_ENCODER_PINS_AFTER_LOAD=0`` keeps the old order and the load-time wait."""

import ast
import logging
import pathlib
import threading
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_memory as dm

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args if args else msg)

    warning = info


@pytest.fixture
def fake_pin(monkeypatch):
    """CPU stand-in for the pinned copy: a clone, held until ``release`` is set."""
    release = threading.Event()
    release.set()

    def pin_memory(self):
        release.wait(10)
        out = self.clone()
        out._fake_pin = True
        return out

    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: getattr(self, "_fake_pin", False))
    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
    return release


def _groups(module):
    out = []
    for p in module.parameters():
        out.append(types.SimpleNamespace(cpu_param_dict = {p: p.data}))
    return out


def _pipe_with_pinner(module = None):
    pipe = types.SimpleNamespace()
    module = module if module is not None else torch.nn.Linear(64, 64)
    pinner = dm._GroupPinner(module, _groups(module), torch.device("cpu"))
    setattr(pipe, dm._PENDING_PINS_ATTR, [pinner])
    return pipe, pinner


@pytest.mark.parametrize("env", [None, "1", "0"])
def test_the_load_waits_for_the_encoder_pins_only_under_the_kill_switch(monkeypatch, env):
    if env is None:
        monkeypatch.delenv(dm.ENCODER_PINS_AFTER_LOAD_ENV, raising = False)
    else:
        monkeypatch.setenv(dm.ENCODER_PINS_AFTER_LOAD_ENV, env)
    calls = []
    monkeypatch.setattr(dm, "start_background_pins", lambda pipe: calls.append("start") or 1)
    monkeypatch.setattr(dm, "finish_background_pins", lambda pipe: calls.append("finish") or 3.0)
    pipe = types.SimpleNamespace()
    waited = dm.finish_encoder_pins_at_load(pipe, _Log())
    if env == "0":
        assert calls == ["start", "finish"] and waited == 3.0
        assert getattr(pipe, dm._PINS_SETTLED_ATTR)
    else:
        assert calls == ["start"] and waited == 0.0
        assert not getattr(pipe, dm._PINS_SETTLED_ATTR, False)
    assert dm.finish_encoder_pins_at_load(None) == 0.0


def test_the_first_render_waits_for_a_running_pin_then_trims_once(monkeypatch, fake_pin):
    trims = []
    monkeypatch.setattr(
        dm, "trim_host_heap_after_load", lambda logger = None: trims.append(1) or True
    )
    fake_pin.clear()
    pipe, pinner = _pipe_with_pinner()
    pinner.start()
    done = []
    waiter = threading.Thread(
        target = lambda: done.append(dm.finish_encoder_pins_before_render(pipe, threading.Event()))
    )
    waiter.start()
    waiter.join(0.5)
    assert waiter.is_alive() and not done  # still pinning: the render is held
    fake_pin.set()
    waiter.join(10)
    assert done and pinner.join(0) and pinner.pinned > 0
    assert trims == [1] and getattr(pipe, dm._PINS_SETTLED_ATTR)
    # settled: later renders neither wait nor trim again
    assert dm.finish_encoder_pins_before_render(pipe, threading.Event()) == 0.0
    assert trims == [1]


def test_a_cancelled_render_leaves_the_pin_running_and_unsettled(monkeypatch, fake_pin):
    monkeypatch.setattr(dm, "trim_host_heap_after_load", lambda logger = None: True)
    fake_pin.clear()
    pipe, pinner = _pipe_with_pinner()
    cancel = threading.Event()
    cancel.set()
    try:
        dm.finish_encoder_pins_before_render(pipe, cancel)
        assert not getattr(pipe, dm._PINS_SETTLED_ATTR, False)
        assert not pinner.join(0)
    finally:
        fake_pin.set()
        pinner.join(10)
    assert dm.finish_encoder_pins_before_render(pipe, threading.Event()) >= 0.0
    assert getattr(pipe, dm._PINS_SETTLED_ATTR)


def test_no_pinners_settles_without_a_trim(monkeypatch):
    trims = []
    monkeypatch.setattr(dm, "trim_host_heap_after_load", lambda logger = None: trims.append(1))
    pipe = types.SimpleNamespace()
    assert dm.finish_encoder_pins_before_render(pipe) == 0.0
    assert getattr(pipe, dm._PINS_SETTLED_ATTR) and trims == []
    assert dm.finish_encoder_pins_before_render(None) == 0.0


@pytest.fixture
def streaming(monkeypatch, fake_pin):
    """``_apply_streaming_offload`` on a CPU host: diffusers' apply is a recorder, groups are one per parameter."""
    diffusers_hooks = pytest.importorskip("diffusers.hooks")
    monkeypatch.delenv(dm.ENCODER_PINS_AFTER_LOAD_ENV, raising = False)
    monkeypatch.delenv(dm.GROUP_OFFLOAD_PIN_ENV, raising = False)
    pipe = types.SimpleNamespace()
    pipe.transformer = torch.nn.Linear(64, 64)
    pipe.text_encoder = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 64))
    pipe.components = {"transformer": pipe.transformer, "text_encoder": pipe.text_encoder}
    groups = {id(m): _groups(m) for m in (pipe.transformer, pipe.text_encoder)}
    applied = []

    def apply_group_offloading(
        module,
        onload_device = None,
        offload_device = None,
        offload_type = None,
        num_blocks_per_group = None,
        use_stream = False,
        non_blocking = False,
        record_stream = False,
        low_cpu_mem_usage = False,
    ):
        raise AssertionError("routed through _apply_group_offloading_packed")

    def packed(_apply, module, kwargs, logger):
        encoder = next(
            (
                p
                for p in getattr(pipe, dm._PENDING_PINS_ATTR, None) or ()
                if p.module is pipe.text_encoder
            ),
            None,
        )
        name = "transformer" if module is pipe.transformer else "text_encoder"
        applied.append((name, bool(encoder is not None and encoder._started)))

    monkeypatch.setattr(diffusers_hooks, "apply_group_offloading", apply_group_offloading)
    monkeypatch.setattr(dm, "_apply_group_offloading_packed", packed)
    monkeypatch.setattr(dm, "_offload_groups", lambda module: groups.get(id(module), []))
    for name in (
        "install_group_offload_buffer_restore",
        "install_group_offload_hooks_eager",
        "install_group_pin_wait",
    ):
        monkeypatch.setattr(dm, name, lambda *a, **k: True)
    monkeypatch.setattr(dm, "install_group_prefetch", lambda *a, **k: None)
    monkeypatch.setattr(dm, "_pin_top_level_group", lambda *a, **k: True)
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda *a, **k: True)
    monkeypatch.setattr(dm, "_streaming_prefetch_enabled", lambda: True)
    monkeypatch.setattr(dm, "_encoder_background_pin_wanted", lambda *a, **k: True)
    monkeypatch.setattr(dm, "_streamed_pin_plan", lambda *a, **k: (True, True))
    monkeypatch.setattr(dm, "_streamed_pin_leftover_mib", lambda *a, **k: 0)
    return types.SimpleNamespace(pipe = pipe, applied = applied, monkeypatch = monkeypatch)


def _encoder_pinner(pipe):
    return next(p for p in getattr(pipe, dm._PENDING_PINS_ATTR) if p.module is pipe.text_encoder)


def test_a_whole_pin_encoder_starts_pinning_before_the_transformer_pins(streaming):
    log = _Log()
    dm._apply_streaming_offload(streaming.pipe, "cuda", log)
    assert streaming.applied == [("text_encoder", False), ("transformer", True)]
    pinner = _encoder_pinner(streaming.pipe)
    assert pinner.join(10) and pinner.pinned == sum(
        p.nbytes for p in streaming.pipe.text_encoder.parameters()
    )
    assert any("beside the transformer's pin" in line for line in log.lines)
    # the load's start is then a no-op restart of a running (here finished) pinner
    assert dm.start_background_pins(streaming.pipe) == 1


@pytest.mark.parametrize("case", ["kill_switch", "prefix_pin"])
def test_the_old_order_under_the_kill_switch_or_a_prefix_pin(streaming, case):
    if case == "kill_switch":
        streaming.monkeypatch.setenv(dm.ENCODER_PINS_AFTER_LOAD_ENV, "0")
    else:
        # the encoders do not fit whole: a capped prefix pin, priced against what the transformer leaves
        streaming.monkeypatch.setattr(dm, "_streamed_pin_plan", lambda *a, **k: (True, False))
        streaming.monkeypatch.setattr(dm, "_streamed_pin_leftover_mib", lambda *a, **k: 1)
    log = _Log()
    dm._apply_streaming_offload(streaming.pipe, "cuda", log)
    assert streaming.applied == [("transformer", False), ("text_encoder", False)]
    pinner = _encoder_pinner(streaming.pipe)
    assert not pinner._started
    assert not any("beside the transformer's pin" in line for line in log.lines)
    dm.start_background_pins(streaming.pipe)
    assert pinner.join(10)


def _calls(tree, name):
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == name
    ]


def test_generate_waits_for_the_pins_before_loras_and_checks_cancel():
    tree = ast.parse((_BACKEND / "core" / "inference" / "diffusion.py").read_text(encoding = "utf-8"))
    generate = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "generate"
    )
    waits = _calls(generate, "finish_encoder_pins_before_render")
    loras = _calls(generate, "_apply_loras")
    assert len(waits) == 1 and loras and waits[0].lineno < min(c.lineno for c in loras)
    assert [
        a.attr if isinstance(a, ast.Attribute) else getattr(a, "id", None) for a in waits[0].args
    ] == [
        "pipe",
        "cancel",
        "logger",
    ]
    # the wait is cancellable: a cancel check follows it before any other work
    cancel_checks = [
        n
        for n in ast.walk(generate)
        if isinstance(n, ast.If)
        and n.lineno > waits[0].lineno
        and "cancel.is_set" in ast.unparse(n.test)
    ]
    assert cancel_checks and min(n.lineno for n in cancel_checks) <= waits[0].lineno + 2
