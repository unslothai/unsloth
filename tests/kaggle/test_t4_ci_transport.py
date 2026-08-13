# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""What the kernel builder generates and what the launcher can read back.

These cover the transport, not the payloads: the generated driver and payload
cells are EXECUTED here with Kaggle replaced by a stub, so a control-flow
mistake in a cell that only ever runs on a Kaggle T4 is caught on a runner.

No network call, no credential and no GPU: `subprocess` is swapped out
wholesale for the driver cells, and the payload cells that are executed stop
before they reach torch.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = REPO_ROOT / "tests" / "kaggle" / "t4_smoke"
CI_DIR = REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci"

sys.path.insert(0, str(SMOKE_DIR))
sys.path.insert(0, str(CI_DIR))

import launch  # noqa: E402


# ------------------------------------------------------------------ launcher


@pytest.mark.parametrize("plain", [False, True])
def test_a_report_reaches_the_launcher_through_kaggles_structured_log(tmp_path, plain):
    """The log fallback exists for the run whose notebook never came back.

    Kaggle hands the log over as a JSON array of stream records, so scanning
    the file as text finds no line that starts with the report prefix and a
    real failure is filed as `infra`.
    """
    payload = {"label": "control", "model": "unsloth/Qwen2.5-0.5B", "passed": False,
               "failures": ["reference band: out of band at step 3"]}
    line = launch.RESULT_PREFIX + json.dumps(payload) + "\n"
    body = (
        line
        if plain
        else json.dumps(
            [
                {"stream_name": "stdout", "time": 12.0, "data": "install done\n"},
                {"stream_name": "stdout", "time": 13.0, "data": line},
            ]
        )
    )
    kernel_dir = tmp_path / "unsloth-t4-ci-deadbeef"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.log").write_text(body, encoding = "utf-8")

    reports = launch.extract_reports(tmp_path)
    assert [r["label"] for r in reports] == ["control"]
    assert reports[0]["passed"] is False


def test_a_log_record_that_splits_the_report_is_still_read(tmp_path):
    """Record boundaries are not line boundaries; join before scanning."""
    payload = {"label": "canary", "model": "unsloth/Qwen2.5-0.5B", "passed": True}
    line = launch.RESULT_PREFIX + json.dumps(payload) + "\n"
    half = len(line) // 2
    kernel_dir = tmp_path / "unsloth-t4-ci-cafe"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.log").write_text(
        json.dumps(
            [
                {"stream_name": "stdout", "data": line[:half]},
                {"stream_name": "stdout", "data": line[half:]},
            ]
        ),
        encoding = "utf-8",
    )
    assert [r["label"] for r in launch.extract_reports(tmp_path)] == ["canary"]


def test_every_push_attempt_gets_its_own_slug(monkeypatch):
    """Retrying onto one slug pushes a SECOND session and hides the first.

    A push to an id that already exists creates a new VERSION and starts
    another batch session; it does not supersede the running one. And
    `kernels/output` and `kernels status` never pass a version label, so they
    answer for the latest session only. A retry after a lost response
    therefore reads the wrong execution's evidence while the first keeps
    billing unseen.
    """
    attempts: list[list[str]] = []
    deleted: list[str] = []

    def fake_run(cmd, **kw):
        cmd = [str(c) for c in cmd]
        attempts.append(cmd)
        if cmd[1:3] == ["kernels", "delete"]:
            deleted.append(cmd[3])
            return types.SimpleNamespace(returncode = 0, stdout = "", stderr = "")
        metadata = json.loads(
            (Path(cmd[cmd.index("-p") + 1]) / "kernel-metadata.json").read_text()
        )
        attempts[-1] = ["push", metadata["id"]]
        if len(deleted) + 1 < 3:
            return types.SimpleNamespace(returncode = 1, stdout = "", stderr = "Connection reset")
        return types.SimpleNamespace(returncode = 0, stdout = "Successfully pushed", stderr = "")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)
    pushed = launch.push(Path(__file__), "someuser", 3600)

    slugs = [a[1] for a in attempts if a[0] == "push"]
    assert len(slugs) == 3, slugs
    assert len(set(slugs)) == 3, f"every retry reused one slug: {slugs}"
    assert pushed["ok"] and pushed["slug"] == slugs[-1]
    # Each earlier attempt may have landed despite the error it reported, so
    # it is deleted before the next one adds a second concurrent session.
    assert deleted == [s for s in slugs[:-1]]
    assert pushed["attempts"] == slugs


def test_the_temp_dir_is_left_alone_when_the_log_is_not_json(tmp_path):
    """A plain-text log, and a JSON object that is not a record array."""
    kernel_dir = tmp_path / "unsloth-t4-ci-beef"
    kernel_dir.mkdir()
    (kernel_dir / "kernel.log").write_text(json.dumps({"log": "nothing here"}), encoding = "utf-8")
    assert launch.extract_reports(tmp_path) == []
