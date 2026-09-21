# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The keystroke metric must not drop the keystroke it exists to catch.

`collect()` used to be called after a FIXED 200 ms wait, so a keystroke whose paint had not
resolved by that moment was simply absent from the samples. The keystroke that has not painted yet
is the SLOWEST one, so the reading systematically omitted its own worst case: measured against the
real `input.js`, a 500 ms keystroke vanished from a reading whose max was 20 ms. The action still
reported success, because the only thing it checked was that the composer's value had grown by the
number of characters typed.

That inverts under the condition the metric exists to detect -- a build that makes typing worse has
more unpainted keystrokes at the drain, so it drops more of its slowest samples and reads FASTER --
and `keystroke_p95_ms` is the highest-weight metric in the scoring table. It also feeds the
null-treatment control, so an artificially tight reading there tightens the noise floor that every
later comparison on that machine is judged against.

The remedy is not a bigger constant, which has the same defect on a slower machine or a heavier
rung. The wait ends when the WORK ends: the driver polls `settled()` until nothing is in flight,
bounded only so a wedged renderer cannot eat the slot, and the reading then has to account for
every keystroke the instrument saw before it may be quoted.

Two levels. The JS half runs the shipped `input.js` under node over a controlled clock, so the
paint that is still in flight really is in flight. The Python half drives the shipped `keystroke`
action and asserts what a reader ends up with.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.runtime.types import ActionContext, Cell, Window  # noqa: E402
from studiobench.scene.actions import keystroke  # noqa: E402

INPUT_JS = Path(__file__).resolve().parents[1] / "input.js"

#: A DOM small enough to run the instrument and explicit enough to hold a paint open. Paints resolve
#: only when this driver says so, which makes "still in flight at the drain" a state the test can
#: create rather than race for.
DRIVER = """
import fs from "node:fs";
let now = 0;
const listeners = { window: {}, target: {} };
class Ev {
  constructor(type, target, timeStamp) {
    this.type = type;
    this.target = target;
    this.timeStamp = timeStamp;
  }
}
const drop = (bag, t, fn) => { bag[t] = (bag[t] || []).filter((f) => f !== fn); };
const target = {
  value: "",
  addEventListener: (t, fn) => ((listeners.target[t] ||= []).push(fn)),
  removeEventListener: (t, fn) => drop(listeners.target, t, fn),
};
const paints = [];
globalThis.window = {
  addEventListener: (t, fn) => ((listeners.window[t] ||= []).push(fn)),
  removeEventListener: (t, fn) => drop(listeners.window, t, fn),
  __sbNextPaint: () => new Promise((resolve) => paints.push(resolve)),
};
globalThis.document = { querySelector: () => target };
globalThis.performance = { now: () => now };
globalThis.requestAnimationFrame = (fn) => setTimeout(() => fn(now), 0);
eval(fs.readFileSync(process.argv[2], "utf8"));
const input = window.__sb.input;
const fire = (bucket, ev) => (listeners[bucket][ev.type] || []).forEach((fn) => fn(ev));
const tick = () => new Promise((r) => setImmediate(r));

const COUNT = 12, FAST = 20, SLOW = 500;
const MODE = process.argv[3];

async function main() {
  input.arm("textarea");
  for (let i = 0; i < COUNT; i += 1) {
    now += 60;
    fire("window", new Ev("keydown", target, now));
    target.value += "a";
    fire("target", new Ev("input", target, now));
    await tick();
    if (i !== COUNT - 1) { now += FAST; paints.shift()?.(now); await tick(); }
  }
  if (MODE === "fixed") {
    // The old drain: a flat 200 ms, which the 500 ms paint outlives.
    now += 200;
    await tick();
  } else {
    let waited = 0;
    while (input.settled().pending && waited < 3000) {
      now += 25; waited += 25;
      if (waited >= SLOW - FAST) paints.shift()?.(now);
      await tick();
    }
  }
  console.log(JSON.stringify(input.collect(COUNT)));
}
main();
"""


def _node():
    return shutil.which("node")


def _run(tmp_path: Path, mode: str) -> dict:
    driver = tmp_path / "driver.mjs"
    driver.write_text(DRIVER, encoding = "utf-8")
    out = subprocess.run(
        [_node(), str(driver), str(INPUT_JS), mode],
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(_node() is None, reason = "node is not installed")
def test_a_fixed_wait_loses_the_slowest_keystroke(tmp_path):
    """THE DEFECT, against the shipped instrument. Eleven keystrokes paint in 20 ms and the
    twelfth takes 500 ms; a flat 200 ms drain collects only the eleven fast ones."""

    got = _run(tmp_path, "fixed")

    assert got["samples"] == 11
    assert got["max_ms"] == 20
    # The instrument knows it is losing one, which is what makes the reading refusable.
    assert got["pending_at_collect"] is True
    assert got["inputs_seen"] == 12


@pytest.mark.skipif(_node() is None, reason = "node is not installed")
def test_settling_on_the_work_keeps_it(tmp_path):
    got = _run(tmp_path, "settle")

    assert got["samples"] == 12
    assert got["max_ms"] == 500
    assert got["p95_ms"] == 500
    assert got["pending_at_collect"] is False
    assert got["inputs_seen"] == 12


# ── what the action does with it ─────────────────────────────────────────────────────────────


class _Page:
    def __init__(self) -> None:
        self.value = ""

    def query_selector(self, selector):
        return object() if "Message input" in selector else None

    def click(self, *a, **k) -> None:
        pass

    def wait_for_timeout(self, ms) -> None:
        pass

    @property
    def keyboard(self):
        return types.SimpleNamespace(type = self._type)

    def _type(
        self,
        text,
        delay = 0,
    ) -> None:
        self.value += text


class _Instrument:
    """The input instrument, with the reading it will return dictated by the test."""

    def __init__(
        self,
        reading: dict,
        *,
        pending_forever: bool = False,
    ) -> None:
        self.reading = reading
        self.pending_forever = pending_forever
        self.polls = 0

    def arm(self, selector: str) -> dict:
        return {"armed": True, "baseline_length": 0}

    def settled(self) -> dict:
        self.polls += 1
        return {"pending": self.pending_forever, "samples": 0, "seen": 0}

    def collect(self, expected: int) -> dict:
        return self.reading


def _run_keystroke(inst: _Instrument, count: int = 12):
    page = _Page()
    ctx = ActionContext(
        page = page,
        cdp = None,
        cell = Cell(cell_id = "r10K.A0.rep0", rung = "10K", rung_tokens = 10_000),
        window = Window(name = "action:keystroke", kind = "action", cell = None, t_open_ms = 0.0),
        args = {"count": count, "_input_instrument": inst},
        budget_ms = 6_000,
        dom = None,
        log = lambda msg: None,
    )
    return keystroke(ctx)


def _reading(**overrides) -> dict:
    base = {
        "samples": 12,
        "samples_attempted": True,
        "expected": 12,
        "inputs_seen": 12,
        "pending_at_collect": False,
        "coalesced": 0,
        "p50_ms": 18.0,
        "p95_ms": 20.0,
        "max_ms": 20.0,
        "first_ms": 40.0,
        "text_length": 12,
        "grew_by": 12,
    }
    base.update(overrides)
    return base


def test_a_healthy_reading_still_passes():
    result = _run_keystroke(_Instrument(_reading()))

    assert result.ran is True
    assert result.expect_ok is True
    assert result.timings["p95_ms"] == 20.0


def test_a_jammed_page_that_coalesced_most_of_its_keystrokes_still_passes():
    """A LOW SAMPLE COUNT IS NOT A FAULT, it is the finding. On a jammed page most keystrokes
    arrive behind a paint that has not finished, and the slow paint that swallowed them IS
    measured. What matters is that every input is accounted for."""

    result = _run_keystroke(_Instrument(_reading(samples = 3, coalesced = 9, p95_ms = 480.0)))

    assert result.expect_ok is True
    assert result.expect["coalesced"] == 9
    assert result.timings["p95_ms"] == 480.0


def test_a_reading_with_a_keystroke_still_unpainted_is_not_quotable():
    result = _run_keystroke(_Instrument(_reading(samples = 11, pending_at_collect = True, max_ms = 20.0)))

    assert result.ran is True
    assert result.expect_ok is False
    assert "still unpainted" in result.reason
    assert result.expect["pending_at_collect"] is True


def test_a_keystroke_that_never_reached_the_instrument_is_not_quotable():
    """`grew_by` is satisfied by a textarea whose value is right; it says nothing about which
    keystrokes were measured on the way in."""

    result = _run_keystroke(_Instrument(_reading(samples = 8, inputs_seen = 8, coalesced = 0)))

    assert result.expect_ok is False
    assert "8 of 8" in result.reason or "covers" in result.reason
    assert result.expect["inputs_seen"] == 8


def test_the_composer_check_still_reports_its_own_failure():
    """The control: the pre-existing reason is unchanged when it is the one that failed."""

    result = _run_keystroke(_Instrument(_reading(grew_by = 4, text_length = 4)))

    assert result.expect_ok is False
    assert "composer value grew by 4" in result.reason


def test_the_drain_polls_until_the_page_settles_rather_than_waiting_a_constant():
    inst = _Instrument(_reading(), pending_forever = True)

    _run_keystroke(inst)

    # It kept asking, and stopped at its bound rather than at a fixed interval.
    assert inst.polls > 1


def test_no_samples_at_all_is_still_not_run():
    result = _run_keystroke(_Instrument(_reading(samples = 0, inputs_seen = 0)))

    assert result.ran is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
