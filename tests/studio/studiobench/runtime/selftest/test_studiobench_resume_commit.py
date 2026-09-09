# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A ref is a pointer, so a resume is judged on the COMMIT the ref resolved to.

`prepare_payload` refuses a `--resume` whose configuration differs from the payload's, and every
axis it compares is a string the caller typed. `--branch main --ab fix` is the same string today
that it was yesterday; the build behind it is not. `checkout_ref` fetches and resolves the ref
afresh on every install and RETURNS the commit it landed on, and that return value was thrown
away, so nothing in the payload ever recorded which build produced its cells.

The result passed every check and was invisible: the completed cells were skipped, the rungs the
payload still owed were measured on today's build, and `report.assemble_rows` printed the mixture
under one header naming one ref. `unslothai/main` moves several times a day and a topic branch
under review moves whenever it is pushed to, so this is the ordinary shape of an interrupted run
resumed the next morning, not an unusual one.

The commit cannot be known where the other axes are checked -- `prepare_payload` runs BEFORE
anything is installed, deliberately, so a refusal costs a millisecond rather than two clones and
two builds. So it is checked at the first moment it exists: after the sides are up, before the
browser, the pacer and every cell.

`run()` itself is driven, with the seams that leave this process stubbed at the boundary they
cross. The identity, the payload and the refusal are the shipped ones.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import studiobench.__main__ as sb  # noqa: E402
from studiobench import pacer as pacer_mod  # noqa: E402
from studiobench.runtime import browser as browser_mod  # noqa: E402
from studiobench.runtime import bundle_guard, lifecycle  # noqa: E402
from studiobench.runtime import session as session_mod  # noqa: E402
from studiobench.runtime.lifecycle import StudioAuth, StudioInstall  # noqa: E402
from studiobench.runtime.types import Paths  # noqa: E402


class _Verdict:
    production = True
    bundle_type = 0
    reason = "production build"

    def as_dict(self) -> dict:
        return {"production": True, "reason": self.reason}


class _Bundle:
    # WHAT `browser.launch` WOULD HAVE RESOLVED on the machine running this test. `run_meta` records the
    # engine and `requested_identity` resolves the same way, so a stub naming a fixed one would make a
    # legitimate resume look like an engine change off Linux and macOS.
    engine = browser_mod.default_engine()[0]
    engine_note = "stubbed for this test"
    browser = context = page = cdp = None

    def close(self) -> None:
        pass


class _Pacer:
    base_url = "http://127.0.0.1:65535"

    def __init__(self) -> None:
        self.state = types.SimpleNamespace(model_ids = [])

    def start(self):
        return self

    def stop(self) -> None:
        pass


class _Runner:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def run(self, cell, plan) -> dict:
        return {
            "cell_id": cell.cell_id,
            "completed": True,
            "actions": [{"action": "keystroke", "ran": True}],
        }


@pytest.fixture
def studio(monkeypatch, tmp_path):
    """Every seam `run()` reaches outside this process. `commits` is the knob: which commit each
    ref resolves to on THIS invocation, which is what moving a branch means here."""

    state = {"commits": {}, "stopped": [], "out": tmp_path / "out"}

    def fake_install(ref, home, *args, **kwargs):
        install = StudioInstall(home = Path(home), repo = Path(home).parent / "repo", branch = ref)
        # `setattr` rather than a constructor argument, so this fixture also builds against a
        # `StudioInstall` that has no commit field and the tests fail on the subject rather than the way
        # in.
        install.commit = state["commits"].get(ref, f"c-{ref}-1")
        return install

    def fake_launch(install, port, log_path, *args, **kwargs):
        install.port = port
        install.pid = 90_000 + port
        install.bootstrap_password = "secret"
        return install

    monkeypatch.setattr(lifecycle, "install_studio", fake_install)
    monkeypatch.setattr(lifecycle, "launch_studio", fake_launch)
    monkeypatch.setattr(lifecycle, "stop_studio", lambda install: state["stopped"].append(install))
    monkeypatch.setattr(lifecycle, "wait_for_healthz", lambda *a, **k: True)
    monkeypatch.setattr(
        lifecycle,
        "authenticate",
        lambda base_url, username, password: StudioAuth(
            access_token = "t",
            refresh_token = "r",
            base_url = base_url,
            username = username,
            password = password,
        ),
    )
    monkeypatch.setattr(lifecycle, "register_provider", lambda *a, **k: "provider-1")
    monkeypatch.setattr(bundle_guard, "check_bundle", lambda url: _Verdict())
    monkeypatch.setattr(browser_mod, "launch", lambda *a, **k: _Bundle())
    monkeypatch.setattr(
        browser_mod,
        "install_wall_clock_watchdog",
        lambda *a, **k: types.SimpleNamespace(cancel = lambda: None),
    )
    monkeypatch.setattr(pacer_mod, "Pacer", _Pacer)
    monkeypatch.setattr(session_mod, "CellRunner", _Runner)
    return state


def _args(state, *extra):
    return sb.parse_args(
        ["--tier", "quick", "--rungs", "1K", "--out", str(state["out"])] + list(extra)
    )


def _rows(state):
    path = Paths.under(state["out"]).payload_jsonl
    return [json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines() if line]


# ── what a run records ───────────────────────────────────────────────────────────────────────


def test_a_run_records_the_commit_its_ref_resolved_to(studio):
    assert sb.run(_args(studio, "--branch", "main")) == 0

    meta = [r for r in _rows(studio) if r.get("row_type") == "run_meta"][0]
    assert meta["studio_ref"] == "main"
    assert meta["studio_commit"] == "c-main-1"


def test_an_attached_studio_records_no_commit(studio):
    """The build behind a URL is not visible from here, and a blank is the honest answer."""

    assert sb.run(_args(studio, "--attach", "http://127.0.0.1:5310")) == 0

    meta = [r for r in _rows(studio) if r.get("row_type") == "run_meta"][0]
    assert meta["studio_commit"] == ""


# ── the refusal ──────────────────────────────────────────────────────────────────────────────


def test_a_resume_after_the_branch_moved_is_refused(studio):
    assert sb.run(_args(studio, "--branch", "main")) == 0

    studio["commits"]["main"] = "c-main-2"  # `main` advanced overnight
    with pytest.raises(SystemExit) as excinfo:
        sb.run(_args(studio, "--branch", "main", "--resume"))

    message = str(excinfo.value)
    assert "studio_commit" in message
    assert "c-main-1" in message and "c-main-2" in message


def test_the_studios_that_refusal_launched_are_released(studio):
    """The refusal happens inside the setup guard, so it may not leak the servers it needed."""

    assert sb.run(_args(studio, "--branch", "main")) == 0
    studio["stopped"].clear()

    studio["commits"]["main"] = "c-main-2"
    with pytest.raises(SystemExit):
        sb.run(_args(studio, "--branch", "main", "--resume"))

    assert [i.branch for i in studio["stopped"]] == ["main"]


def test_a_resume_after_the_treatment_moved_is_refused(studio):
    """The second side gets the same rule, out of `ab_plan` rather than `run_meta`."""

    assert sb.run(_args(studio, "--branch", "main", "--ab", "fix"), ab_ref = "fix") == 0

    studio["commits"]["fix"] = "c-fix-2"  # the pull request was pushed to
    with pytest.raises(SystemExit) as excinfo:
        sb.run(_args(studio, "--branch", "main", "--ab", "fix", "--resume"), ab_ref = "fix")

    message = str(excinfo.value)
    assert "treatment_commit" in message
    assert "c-fix-1" in message and "c-fix-2" in message


# ── the controls ─────────────────────────────────────────────────────────────────────────────


def test_the_same_commit_still_resumes(studio):
    """The control that matters: a resume of the build the payload was recorded on is the whole
    point of `--resume` and must still skip its cells and exit 0."""

    assert sb.run(_args(studio, "--branch", "main")) == 0
    assert sb.run(_args(studio, "--branch", "main", "--resume")) == 0


def test_a_payload_recorded_before_commits_were_written_still_resumes(studio):
    """The back-compatibility control, and the same rule `recorded_identities` already applies:
    an axis a payload never declared cannot be a difference."""

    assert sb.run(_args(studio, "--branch", "main")) == 0

    path = Paths.under(studio["out"]).payload_jsonl
    kept = []
    for row in _rows(studio):
        row.pop("studio_commit", None)
        kept.append(json.dumps(row))
    path.write_text("\n".join(kept) + "\n", encoding = "utf-8")

    studio["commits"]["main"] = "c-main-2"
    assert sb.run(_args(studio, "--branch", "main", "--resume")) == 0


def test_an_attached_resume_is_not_refused_for_having_no_commit(studio):
    """The other control: attaching cannot start failing against its own payload just because
    this run has no commit to offer either."""

    assert sb.run(_args(studio, "--attach", "http://127.0.0.1:5310")) == 0
    assert sb.run(_args(studio, "--attach", "http://127.0.0.1:5310", "--resume")) == 0


def test_a_fresh_run_onto_a_moved_branch_is_not_refused(studio):
    """And the last one: this is a `--resume` rule. Without it the payload is archived and a new
    one is started, which was always the right answer for a different build."""

    assert sb.run(_args(studio, "--branch", "main")) == 0

    studio["commits"]["main"] = "c-main-2"
    assert sb.run(_args(studio, "--branch", "main")) == 0

    meta = [r for r in _rows(studio) if r.get("row_type") == "run_meta"]
    assert [m["studio_commit"] for m in meta] == ["c-main-2"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
