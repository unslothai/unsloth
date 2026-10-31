# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A two-expert denoiser's second expert pins after the other weights, and the render does not wait for it."""

import threading
import time

import pytest

import core.inference.diffusion_memory as mem


class _Pipe:
    pass


class _Group:
    cpu_param_dict: dict = {}


class _Module:
    def __init__(self, name):
        self.name = name


class _Recorder(mem._GroupPinner):
    """Records when its pin ran instead of pinning; ``hold`` keeps it busy until set."""

    def __init__(
        self,
        name,
        log,
        hold = None,
        fast = False,
    ):
        super().__init__(_Module(name), [_Group()], None, None)
        self.name = name
        self.log = log
        self.hold = hold
        self.fast = fast

    def _run(self):
        self.log.append(("start", self.name))
        if self.fast:
            self.pinned = 64 << 30
        if self.hold is not None:
            self.hold.wait(10)
        self.log.append(("end", self.name))
        for event in self._done.values():
            event.set()


def _late(by, *names):
    return tuple(by[n].module if n in by else _Module(n) for n in names)


def _pipe(
    log,
    holds,
    fast = False,
):
    pipe = _Pipe()
    pinners = [
        _Recorder(name, log, holds.get(name), fast)
        for name in ("transformer", "transformer_2", "text_encoder")
    ]
    setattr(pipe, mem._PENDING_PINS_ATTR, pinners)
    return pipe, {p.name: p for p in pinners}


def test_late_expert_pins_after_the_others_and_the_render_skips_it(monkeypatch):
    monkeypatch.delenv(mem.LATE_EXPERT_PIN_ENV, raising = False)
    log: list = []
    hold = threading.Event()
    pipe, by = _pipe(log, {"text_encoder": hold})
    assert mem.defer_late_expert_pins(pipe, _late(by, "transformer_2")) == 1
    assert by["transformer_2"].late and not by["transformer"].late
    mem.start_background_pins(pipe)
    time.sleep(0.3)
    assert ("start", "transformer_2") not in log
    hold.set()
    mem.finish_background_pins(pipe, include_late = False)
    by["transformer_2"].join(10)
    order = [m for kind, m in log if kind == "start"]
    assert order[-1] == "transformer_2"
    assert log.index(("start", "transformer_2")) > log.index(("end", "text_encoder"))
    assert log.index(("start", "transformer_2")) > log.index(("end", "transformer"))


def test_the_render_wait_returns_while_the_late_expert_is_still_pinning(monkeypatch):
    monkeypatch.delenv(mem.LATE_EXPERT_PIN_ENV, raising = False)
    log: list = []
    hold = threading.Event()
    pipe, by = _pipe(log, {"transformer_2": hold})
    mem.defer_late_expert_pins(pipe, _late(by, "transformer_2"))
    mem.start_background_pins(pipe)
    began = time.perf_counter()
    mem.finish_background_pins(pipe, include_late = False)
    assert time.perf_counter() - began < 5
    assert ("end", "transformer_2") not in log
    hold.set()
    mem.finish_background_pins(pipe)
    assert ("end", "transformer_2") in log


def test_an_early_onload_of_the_late_expert_waits_for_its_pin(monkeypatch):
    monkeypatch.delenv(mem.LATE_EXPERT_PIN_ENV, raising = False)
    log: list = []
    pipe, by = _pipe(log, {})
    mem.defer_late_expert_pins(pipe, _late(by, "transformer_2"))
    late = by["transformer_2"]
    late.wait(late.groups[0])
    assert ("end", "transformer_2") in log
    assert log.index(("start", "transformer_2")) > log.index(("end", "transformer"))


def test_kill_switch_keeps_every_pinner_in_the_render_wait(monkeypatch):
    monkeypatch.setenv(mem.LATE_EXPERT_PIN_ENV, "0")
    log: list = []
    pipe, by = _pipe(log, {})
    assert mem.defer_late_expert_pins(pipe, _late(by, "transformer_2")) == 0
    assert not any(p.late or p.after for p in by.values())
    mem.start_background_pins(pipe)
    mem.finish_background_pins(pipe, include_late = False)
    assert all(("end", name) in log for name in by)


@pytest.mark.parametrize(
    "late", [(), ("missing",), ("transformer", "transformer_2", "text_encoder")]
)
def test_nothing_is_deferred_without_both_a_late_and_an_early_pinner(monkeypatch, late):
    monkeypatch.delenv(mem.LATE_EXPERT_PIN_ENV, raising = False)
    pipe, by = _pipe([], {})
    assert mem.defer_late_expert_pins(pipe, _late(by, *late)) == 0
    assert not any(p.late for p in by.values())


def test_a_stop_while_the_late_expert_waits_releases_its_groups(monkeypatch):
    monkeypatch.delenv(mem.LATE_EXPERT_PIN_ENV, raising = False)
    log: list = []
    hold = threading.Event()
    pipe, by = _pipe(log, {"transformer": hold})
    mem.defer_late_expert_pins(pipe, _late(by, "transformer_2"))
    mem.start_background_pins(pipe)
    late = by["transformer_2"]
    late.stop(5)
    assert not late._thread.is_alive()
    assert all(e.is_set() for e in late._done.values())
    assert ("start", "transformer_2") not in log
    hold.set()
    mem.stop_background_pins(pipe)


def test_fast_pins_run_the_late_expert_beside_the_others_and_the_render_waits_for_it(monkeypatch):
    monkeypatch.delenv(mem.LATE_EXPERT_PIN_ENV, raising = False)
    monkeypatch.setattr(mem, "_LATE_PIN_PROBE_S", 0.05)
    log: list = []
    hold = threading.Event()
    pipe, by = _pipe(log, {"transformer": hold, "text_encoder": hold}, fast = True)
    mem.defer_late_expert_pins(pipe, _late(by, "transformer_2"))
    mem.start_background_pins(pipe)
    deadline = time.time() + 5
    while ("end", "transformer_2") not in log and time.time() < deadline:
        time.sleep(0.02)
    assert ("end", "transformer_2") in log and ("end", "transformer") not in log
    assert not by["transformer_2"].late
    hold.set()
    mem.finish_background_pins(pipe, include_late = False)
    assert not any(p._thread.is_alive() for p in by.values())


def test_slow_pins_keep_the_late_expert_behind_the_others(monkeypatch):
    monkeypatch.delenv(mem.LATE_EXPERT_PIN_ENV, raising = False)
    monkeypatch.setattr(mem, "_LATE_PIN_PROBE_S", 0.05)
    log: list = []
    hold = threading.Event()
    pipe, by = _pipe(log, {"transformer": hold})
    mem.defer_late_expert_pins(pipe, _late(by, "transformer_2"))
    mem.start_background_pins(pipe)
    time.sleep(0.4)
    assert ("start", "transformer_2") not in log and by["transformer_2"].late
    hold.set()
    mem.finish_background_pins(pipe)
    assert log.index(("start", "transformer_2")) > log.index(("end", "transformer"))


def test_copied_bytes_count_per_tensor_before_the_group_is_placed(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(mem, "_registered_pin", lambda src, device = None: src)
    pinner = _Recorder("transformer", [])
    with ThreadPoolExecutor(2) as pool:
        futures = [pinner._submit(pool, torch.zeros(1024, dtype = torch.uint8)) for _ in range(3)]
        for f in futures:
            f.result()
    assert pinner.copied == 3 * 1024 and pinner.pinned == 0
