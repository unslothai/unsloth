# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The screenshot taken beside the parity digest, and the things it must never do.

It must not fire when nobody asked, because it costs an encode per action per arm on every run of
a gate whose whole argument is that it is cheap. It must not take the page with it when it fails:
a shot is evidence about a measurement, so a broken camera has to leave the measurement standing
rather than turn a real reading into `parity_attempted: false`. And it must not be charged to the
measured window, because the film runs on a wall clock and an encode inside the window eats the
gap before the next slot, which is exactly how an action comes to report a MISSED SLOT.

Driven against a stub page rather than a browser. What is under test is the hook's contract --
when it fires, what it names the file, what it records, and what it does with an exception -- and
none of that needs Chromium. That the screenshot itself is a picture of the right thing is the
live job's business.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.scene.schedule import SceneRunner  # noqa: E402


class StubPage:
    def __init__(
        self,
        scroll = 512,
        shot_raises = False,
        capture_raises = False,
    ):
        self.scroll = scroll
        self.shot_raises = shot_raises
        self.capture_raises = capture_raises
        self.shots: list[str] = []

    def evaluate(self, script, *args):
        if "parity.capture" in script:
            if self.capture_raises:
                raise RuntimeError("threadRoot is not a function")
            return {"parity_attempted": True, "digest": "abcd1234", "chars": 10, "messages": []}
        return self.scroll

    def screenshot(self, path):
        if self.shot_raises:
            raise RuntimeError("Target closed")
        self.shots.append(path)
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n")


class StubCell:
    cell_id = "r100K.treatment.rep1"


def runner(page, **base_args) -> SceneRunner:
    return SceneRunner(
        cell = StubCell(),
        page = page,
        cdp = None,
        dom = None,
        recorder = None,
        open_window = None,
        log = lambda _m: None,
        base_args = base_args,
    )


def test_no_shot_is_taken_when_none_was_asked_for():
    page = StubPage()
    assert runner(page)._parity_shot("settings") == {}
    assert page.shots == []


def test_the_digest_call_never_takes_a_picture():
    # The digest is taken INSIDE the measured window and the shot outside it, so `_parity` must
    # not be a camera. If it becomes one again, the encode is charged to the film's own clock.
    page = StubPage()
    got = runner(page, parity_shots = "/nonexistent", arm_label = "base")._parity()
    assert page.shots == []
    assert got["digest"] == "abcd1234"


def test_the_shot_is_named_for_its_cell_action_and_arm(tmp_path):
    # The arm has to be IN the name. Both arms share a fixture, a film and a password, so a file
    # that says only "settings" cannot be attributed to a side once it leaves this directory.
    page = StubPage(scroll = 940)
    got = runner(page, parity_shots = str(tmp_path), arm_label = "treatment")._parity_shot("settings")
    assert got["shot"] == "r100K.treatment.rep1__settings__treatment.png"
    assert got["shot_scroll_top"] == 940
    assert (tmp_path / got["shot"]).exists()


def test_the_two_arms_do_not_collide_on_one_filename(tmp_path):
    base = runner(StubPage(), parity_shots = str(tmp_path), arm_label = "base")._parity_shot("settings")
    treat = runner(StubPage(), parity_shots = str(tmp_path), arm_label = "treatment")._parity_shot(
        "settings"
    )
    assert base["shot"] != treat["shot"]


def test_a_camera_failure_does_not_cost_the_measurement(tmp_path):
    # The digest is the reading; the picture is evidence about it. Losing the second must not
    # discard the first, or a flaky screenshot turns a green run red for no reason.
    page = StubPage(shot_raises = True)
    got = runner(page, parity_shots = str(tmp_path), arm_label = "base")._parity_shot("settings")
    assert "shot" not in got
    assert "Target closed" in got["shot_error"]
    # And the row it merges into keeps its reading.
    row = {"parity_attempted": True, "digest": "abcd1234"}
    row.update(got)
    assert row["parity_attempted"] is True and row["digest"] == "abcd1234"


def test_a_capture_failure_is_still_reported_as_a_failure(tmp_path):
    # And the other direction: the shot hook must not paper over a capture that did not happen.
    page = StubPage(capture_raises = True)
    got = runner(page, parity_shots = str(tmp_path), arm_label = "base")._parity()
    assert got["parity_attempted"] is False
    assert "threadRoot" in got["reason"]
    assert page.shots == []
