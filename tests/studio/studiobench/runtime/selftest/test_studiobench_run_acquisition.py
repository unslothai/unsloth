# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A run releases the Studios it acquired, and records which ones they were.

THE SIDES ARE ACQUIRED ONE AFTER THE OTHER. In a self-managed A/B the base is installed, launched
and already serving before the treatment's clone begins, so a failure in the treatment's setup used
to unwind past a base that nothing ever stopped: the cleanup lives in the `finally` under the cells
and a failure during setup never reaches it. The same hole swallowed the two setup steps that leave
by RETURNING -- the health check and the development-build gate -- with both Studios up.

An abandoned Studio is not idle. `launch_studio` detaches the server with `setsid -f`, so it keeps
the port; Studio's own launcher then ABORTS rather than binding when it finds one of its own servers
there (`studio/backend/run.py`, `_resolve_port` with `avoid_own_studio`), the retry's server exits,
and `wait_for_healthz` takes its 200 from the STALE process. That run measures the build the
previous attempt installed while `run_meta` records the ref this one asked for.

WHICH SERVER THE TREATMENT WAS is the second thing here, and it is asserted through the same drive
because it is a property of the row a real run writes. An attached base is identified by its URL;
the treatment carried only the label typed after `--ab`, so `--attach A --attach-b B --ab fix`
resumed happily against `--attach-b C` and reported B's measurements as C's result.

`run()` itself is driven, with the seams that leave this process stubbed at the boundary they cross:
the installer, the launcher, the health check, the login, the browser and the cell runner. Nothing
below them is re-implemented -- the acquisition loop, the gate, the identity and the payload are the
shipped ones.
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
    def __init__(self, production: bool) -> None:
        self.production = production
        self.bundle_type = 0
        self.reason = "production build" if production else "development server"

    def as_dict(self) -> dict:
        return {"production": self.production, "reason": self.reason}


class _Bundle:
    engine = "webkit"
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


@pytest.fixture
def studio(monkeypatch, tmp_path):
    """Every seam `run()` reaches outside this process, stubbed where it leaves it.

    The returned dict is both the knobs (which ref fails to install, whether /healthz answers,
    whether the bundle is a production build) and the record (what was installed, launched and
    stopped, and what the Studios looked like at the moment the first cell ran).
    """

    state = {
        "installed": [],
        "launched": [],
        "stopped": [],
        "stopped_when_the_cells_ran": None,
        "install_fails_for": None,
        "healthy": True,
        "production": True,
        "out": tmp_path / "out",
    }

    def fake_install(ref, home, *args, **kwargs):
        if ref == state["install_fails_for"]:
            raise RuntimeError(f"install.sh for {ref} exited 1")
        state["installed"].append(ref)
        return StudioInstall(home = Path(home), repo = Path(home).parent / "repo", branch = ref)

    def fake_launch(install, port, log_path, *args, **kwargs):
        install.port = port
        install.pid = 90_000 + port
        install.bootstrap_password = "secret"
        state["launched"].append(install)
        return install

    class _Runner:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def run(self, cell, plan):
            if state["stopped_when_the_cells_ran"] is None:
                state["stopped_when_the_cells_ran"] = list(state["stopped"])
            return {
                "cell_id": cell.cell_id,
                "completed": True,
                "actions": [{"action": "keystroke", "ran": True}],
            }

    monkeypatch.setattr(lifecycle, "install_studio", fake_install)
    monkeypatch.setattr(lifecycle, "launch_studio", fake_launch)
    monkeypatch.setattr(lifecycle, "stop_studio", lambda install: state["stopped"].append(install))
    monkeypatch.setattr(lifecycle, "wait_for_healthz", lambda *a, **k: state["healthy"])
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
    monkeypatch.setattr(bundle_guard, "check_bundle", lambda url: _Verdict(state["production"]))
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


# ── the sides a failed setup leaves behind ──────────────────────────────────────────────────


def test_the_base_studio_is_stopped_when_the_treatment_install_fails(studio):
    """The reported leak: the base is up and serving while the treatment's clone and build run."""

    studio["install_fails_for"] = "pr-9296"

    with pytest.raises(RuntimeError):
        sb.run(_args(studio, "--branch", "main", "--ab", "pr-9296"), ab_ref = "pr-9296")

    assert [i.branch for i in studio["launched"]] == ["main"]
    assert [i.branch for i in studio["stopped"]] == ["main"]


def test_every_studio_is_stopped_when_one_of_them_never_answers_healthz(studio):
    """A setup step that fails by RETURNING leaves the same abandoned server behind."""

    studio["healthy"] = False

    assert sb.run(_args(studio, "--branch", "main", "--ab", "pr-9296"), ab_ref = "pr-9296") == 2
    assert [i.branch for i in studio["stopped"]] == ["main", "pr-9296"]


def test_every_studio_is_stopped_when_the_gate_refuses_a_development_build(studio):
    studio["production"] = False

    assert sb.run(_args(studio, "--branch", "main", "--ab", "pr-9296"), ab_ref = "pr-9296") == 3
    assert [i.branch for i in studio["stopped"]] == ["main", "pr-9296"]


def test_keep_studio_still_leaves_a_failed_setup_running(studio):
    """The control on the guard itself: `--keep-studio` asks for exactly this leak."""

    studio["install_fails_for"] = "pr-9296"

    with pytest.raises(RuntimeError):
        sb.run(
            _args(studio, "--branch", "main", "--ab", "pr-9296", "--keep-studio"),
            ab_ref = "pr-9296",
        )

    assert [i.branch for i in studio["launched"]] == ["main"]
    assert studio["stopped"] == []


def test_a_studio_the_caller_attached_is_never_stopped(studio):
    """The other control: this harness only stops what it launched itself."""

    studio["healthy"] = False
    args = _args(
        studio,
        "--attach",
        "http://127.0.0.1:5310",
        "--attach-b",
        "http://127.0.0.1:5311",
        "--ab",
        "fix",
    )

    assert sb.run(args, ab_ref = "fix") == 2
    assert studio["stopped"] == []


def test_a_run_that_reaches_its_cells_stops_the_studios_once_at_the_end(studio):
    """The control that matters: an ordinary run still gets two live Studios and still cleans up.

    The guard is a `finally` over the whole of setup, so the failure it must not have is stopping
    the Studios on the way IN. `stopped_when_the_cells_ran` is read inside the cell runner.
    """

    args = _args(studio, "--branch", "main", "--ab", "pr-9296", "--reps", "2")

    assert sb.run(args, ab_ref = "pr-9296") == 0
    assert studio["stopped_when_the_cells_ran"] == []
    assert [i.branch for i in studio["stopped"]] == ["main", "pr-9296"]


# ── which server the treatment was ──────────────────────────────────────────────────────────


def test_an_attached_ab_records_the_treatment_url_it_measured(studio):
    args = _args(
        studio,
        "--attach",
        "http://127.0.0.1:5310/",
        "--attach-b",
        "http://127.0.0.1:5311/",
        "--ab",
        "fix",
    )

    assert sb.run(args, ab_ref = "fix") == 0

    plan = [r for r in _rows(studio) if r.get("row_type") == "ab_plan"]
    assert len(plan) == 1
    assert plan[0]["treatment_url"] == "http://127.0.0.1:5311"


def test_a_self_managed_ab_records_no_treatment_url(studio):
    """A treatment this run installed is identified by its ref; the port it landed on is not it."""

    args = _args(studio, "--branch", "main", "--ab", "pr-9296")

    assert sb.run(args, ab_ref = "pr-9296") == 0

    plan = [r for r in _rows(studio) if r.get("row_type") == "ab_plan"]
    assert plan[0]["treatment_url"] == ""


def test_a_resume_pointed_at_another_treatment_studio_is_refused(studio):
    """End to end, over the payload the run above actually wrote: same `--ab` label, second URL."""

    paths = Paths.under(studio["out"])
    first = _args(
        studio,
        "--attach",
        "http://127.0.0.1:5310",
        "--attach-b",
        "http://127.0.0.1:5311",
        "--ab",
        "fix",
    )
    assert sb.run(first, ab_ref = "fix") == 0

    again = _args(
        studio,
        "--attach",
        "http://127.0.0.1:5310",
        "--attach-b",
        "http://127.0.0.1:5312",
        "--ab",
        "fix",
        "--resume",
    )
    corpus_hash = _rows(studio)[0]["corpus_hash"]

    with pytest.raises(SystemExit) as excinfo:
        sb.prepare_payload(
            paths,
            sb.requested_identity(again, "fix", corpus_hash),
            resume = True,
            log = lambda *_a: None,
        )

    message = str(excinfo.value)
    assert "treatment_url" in message
    assert "5311" in message and "5312" in message


def test_the_same_treatment_studio_still_resumes(studio):
    """The control: the identity this payload was recorded under has to keep resuming."""

    paths = Paths.under(studio["out"])
    args = _args(
        studio,
        "--attach",
        "http://127.0.0.1:5310",
        "--attach-b",
        "http://127.0.0.1:5311",
        "--ab",
        "fix",
    )
    assert sb.run(args, ab_ref = "fix") == 0
    corpus_hash = _rows(studio)[0]["corpus_hash"]

    resumed = _args(
        studio,
        "--attach",
        "http://127.0.0.1:5310",
        "--attach-b",
        "http://127.0.0.1:5311/",
        "--ab",
        "fix",
        "--resume",
    )

    assert (
        sb.prepare_payload(
            paths,
            sb.requested_identity(resumed, "fix", corpus_hash),
            resume = True,
            log = lambda *_a: None,
        )
        is None
    )


# ── whether the probe ran before the film ───────────────────────────────────────────────────


def test_a_run_records_whether_the_click_probe_ran(studio):
    """REGRESSION, and the recording half of the identity axis.

    `--click-probe` runs a full `page.click`, a real mouse click, a dispatch, a focus and a hover
    over the thread before the film starts, and its own help text says it "makes the cell's
    timings incomparable with a cell that did not run it". A cell id carries the rung, the arm and
    the repetition and none of that, so unless `run_meta` says which way the run was measured, a
    later `--resume` has nothing to compare and cannot refuse a toggle.
    """

    assert sb.run(_args(studio, "--branch", "main", "--click-probe")) == 0

    meta = [r for r in _rows(studio) if r.get("row_type") == "run_meta"]
    assert len(meta) == 1
    assert meta[0]["click_probe"] is True


def test_a_resume_that_drops_the_click_probe_is_refused(studio):
    """End to end, over the payload the run above actually wrote."""

    paths = Paths.under(studio["out"])
    assert sb.run(_args(studio, "--branch", "main", "--click-probe")) == 0
    corpus_hash = _rows(studio)[0]["corpus_hash"]

    with pytest.raises(SystemExit) as excinfo:
        sb.prepare_payload(
            paths,
            sb.requested_identity(_args(studio, "--branch", "main", "--resume"), None, corpus_hash),
            resume = True,
            log = lambda *_a: None,
        )

    assert "click_probe" in str(excinfo.value)


def test_a_resume_that_keeps_the_click_probe_still_resumes(studio):
    """The control: the identity this payload was recorded under has to keep resuming."""

    paths = Paths.under(studio["out"])
    assert sb.run(_args(studio, "--branch", "main", "--click-probe")) == 0
    corpus_hash = _rows(studio)[0]["corpus_hash"]

    resumed = _args(studio, "--branch", "main", "--click-probe", "--resume")

    assert (
        sb.prepare_payload(
            paths,
            sb.requested_identity(resumed, None, corpus_hash),
            resume = True,
            log = lambda *_a: None,
        )
        is None
    )


def test_a_plain_run_and_a_plain_resume_are_unaffected(studio):
    """The other control. A run that never asks for the probe records it as false and resumes."""

    paths = Paths.under(studio["out"])
    assert sb.run(_args(studio, "--branch", "main")) == 0

    meta = [r for r in _rows(studio) if r.get("row_type") == "run_meta"]
    assert meta[0]["click_probe"] is False
    corpus_hash = _rows(studio)[0]["corpus_hash"]

    assert (
        sb.prepare_payload(
            paths,
            sb.requested_identity(_args(studio, "--branch", "main", "--resume"), None, corpus_hash),
            resume = True,
            log = lambda *_a: None,
        )
        is None
    )


# ── the external probe on the record ────────────────────────────────────────────────────────
#
# `SBENCH_EXTRA_INIT_SCRIPT` is an ENVIRONMENT variable, not a flag, so it outlives the command
# that wanted it. A resume under a shell that still has it set runs the rungs still owed with the
# probe in the page and appends a probed `run_meta`, and `refuse_if_probed` reads every `run_meta`
# in a file: the cells recorded cleanly before it stop being scorable too, permanently, because a
# payload is append-only. So `run_meta` has to carry the probe and `--resume` has to compare it.


class _ProbedBundle(_Bundle):
    """A probe run attaches console and `pageerror` listeners, which a `None` page cannot take."""

    page = types.SimpleNamespace(on = lambda *_a, **_k: None)


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """A probe installed the way a caller installs one, with a file that really is readable."""

    path = tmp_path / "paint_counter.js"
    path.write_text("window.__probe_ticks = 0;\n", encoding = "utf-8")
    monkeypatch.setattr(browser_mod, "launch", lambda *a, **k: _ProbedBundle())
    monkeypatch.setenv("SBENCH_EXTRA_INIT_SCRIPT", str(path))
    return path


def test_a_run_records_which_probe_was_in_the_page(studio, probe):
    """The recording half of the identity axis, over a `run_meta` a real run wrote."""

    assert sb.run(_args(studio, "--branch", "main")) == 0

    meta = [r for r in _rows(studio) if r.get("row_type") == "run_meta"]
    assert len(meta) == 1
    assert meta[0]["probe_init_script"] == str(probe)


def test_a_resume_that_drops_the_external_probe_is_refused(studio, probe, monkeypatch):
    """End to end, over the payload the run above actually wrote."""

    paths = Paths.under(studio["out"])
    assert sb.run(_args(studio, "--branch", "main")) == 0
    corpus_hash = _rows(studio)[0]["corpus_hash"]
    monkeypatch.delenv("SBENCH_EXTRA_INIT_SCRIPT")

    with pytest.raises(SystemExit) as excinfo:
        sb.prepare_payload(
            paths,
            sb.requested_identity(_args(studio, "--branch", "main", "--resume"), None, corpus_hash),
            resume = True,
            log = lambda *_a: None,
        )

    assert "probe_init_script" in str(excinfo.value)


def test_a_resume_under_the_same_probe_still_resumes(studio, probe):
    """The control. A potency ladder that died is meant to be resumable as the ladder it was."""

    paths = Paths.under(studio["out"])
    assert sb.run(_args(studio, "--branch", "main")) == 0
    corpus_hash = _rows(studio)[0]["corpus_hash"]

    assert (
        sb.prepare_payload(
            paths,
            sb.requested_identity(_args(studio, "--branch", "main", "--resume"), None, corpus_hash),
            resume = True,
            log = lambda *_a: None,
        )
        is None
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
