# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The shot fires only when asked, never fails the measurement, and stays outside the timed window."""

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
    # The digest is taken inside the measured window and the shot outside, so `_parity` is no camera.
    page = StubPage()
    got = runner(page, parity_shots = "/nonexistent", arm_label = "base")._parity()
    assert page.shots == []
    assert got["digest"] == "abcd1234"


def test_the_shot_is_named_for_its_cell_action_and_arm(tmp_path):
    # Both arms share fixture, film and password, so the arm must be in the filename.
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
    # A failed screenshot must not discard the digest.
    page = StubPage(shot_raises = True)
    got = runner(page, parity_shots = str(tmp_path), arm_label = "base")._parity_shot("settings")
    assert "shot" not in got
    assert "Target closed" in got["shot_error"]
    row = {"parity_attempted": True, "digest": "abcd1234"}
    row.update(got)
    assert row["parity_attempted"] is True and row["digest"] == "abcd1234"


def test_a_capture_failure_is_still_reported_as_a_failure(tmp_path):
    page = StubPage(capture_raises = True)
    got = runner(page, parity_shots = str(tmp_path), arm_label = "base")._parity()
    assert got["parity_attempted"] is False
    assert "threadRoot" in got["reason"]
    assert page.shots == []
