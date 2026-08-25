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
from studiobench.runtime.types import OutDirLock, Paths  # noqa: E402


class _Verdict:
    def __init__(self, production: bool) -> None:
        self.production = production
        self.bundle_type = 0
        self.reason = "production build" if production else "development server"

    def as_dict(self) -> dict:
        return {"production": self.production, "reason": self.reason}


class _Bundle:
    # WHAT `browser.launch` WOULD HAVE RESOLVED on the machine running this test. `run_meta`
    # records the engine and `requested_identity` resolves the same way, so a stub that names a
    # fixed one would make a legitimate resume look like an engine change off Linux and macOS.
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


def test_an_unreadable_probe_is_refused_before_the_payload_is_archived(studio, monkeypatch):
    """REGRESSION. A refusal must not cost the previous run's payload its standard path.

    Reusing an `--out` without `--resume` archives the payload already there, and that archive used
    to run before `SBENCH_EXTRA_INIT_SCRIPT` was read. A path typo therefore exited 2 having
    installed nothing, launched nothing and recorded nothing, while `payload.jsonl` was gone from
    the one name every reader opens: `--report`, `--assert-liveness` and the next `--resume`.
    """

    monkeypatch.delenv("SBENCH_EXTRA_INIT_SCRIPT", raising = False)
    paths = Paths.under(studio["out"])
    assert sb.run(_args(studio, "--branch", "main")) == 0
    recorded = paths.payload_jsonl.read_text(encoding = "utf-8")

    monkeypatch.setenv("SBENCH_EXTRA_INIT_SCRIPT", str(studio["out"] / "typo_not_a_file.js"))
    assert sb.run(_args(studio, "--branch", "main")) == 2

    assert paths.payload_jsonl.read_text(encoding = "utf-8") == recorded
    assert sorted(p.name for p in paths.out.glob("payload-*.jsonl")) == []
    # The refusal is still ahead of everything it was ahead of before.
    assert studio["installed"] == ["main"]


def test_a_duplicate_run_is_refused_before_it_archives_or_installs_anything(studio):
    """REGRESSION. A run refused for the directory must not have moved the live payload first.

    The guard against two runs in one `--out` used to be taken where the `Recorder` opens
    `payload.jsonl`, which is after `prepare_payload` has archived what was already there and after
    every clone, build and launch. A second launcher pointed at a busy directory therefore renamed
    the FIRST run's live payload out from under it -- the writer keeps its inode through a rename,
    so that run went on recording into `payload-<stamp>.jsonl` while `payload.jsonl`, the one name
    `--report`, `--assert-liveness` and the next `--resume` open, was gone -- and put a clone and a
    build on the machine the first run was measuring, before saying the word it could have said in
    the first millisecond.

    The holder here stands for that first run: it is the same lock a live run holds, so the second
    invocation meets exactly what it meets in the field.
    """

    paths = Paths.under(studio["out"])
    assert sb.run(_args(studio, "--branch", "main")) == 0
    recorded = paths.payload_jsonl.read_text(encoding = "utf-8")

    studio["installed"].clear()
    holder = OutDirLock.take(paths.out)
    try:
        with pytest.raises(SystemExit) as excinfo:
            sb.run(_args(studio, "--branch", "main"))
    finally:
        holder.release()

    assert "still running" in str(excinfo.value)
    assert (
        paths.payload_jsonl.read_text(encoding = "utf-8") == recorded
    ), "the refused duplicate archived the live payload of the run it was refused in favour of"
    assert sorted(p.name for p in paths.out.glob("payload-*.jsonl")) == []
    assert studio["installed"] == [], (
        "the duplicate installed a Studio on a machine that was already measuring, which is the "
        "contention the guard exists to prevent, paid as the cost of refusing it"
    )
    # And the directory is free again the moment the holder lets go.
    assert sb.run(_args(studio, "--branch", "main", "--resume")) == 0


def test_a_duplicate_is_still_refused_while_the_report_is_being_rendered(studio, monkeypatch):
    """REGRESSION. The directory stays held until `run()` has finished READING the payload back.

    `rec.close()` runs in the `finally` under the cells; `_render_ab` and `_summarise` then reopen
    `payload.jsonl` after it and before `run()`'s own outer `finally` lets the directory go. While
    `Recorder.close` released the lock it had ADOPTED from `run()`, the directory was free for the
    whole of that window, and a duplicate arriving in it was admitted. It then did what the guard
    exists to stop, to a run whose cells had all completed:

      * `prepare_payload` renames `payload.jsonl` to `payload-<stamp>.jsonl` BEFORE it clones
        anything, so for the minutes it spends installing there is no `payload.jsonl` at all and
        the first run's reporting step dies with `FileNotFoundError` -- after every cell passed.
      * once it has opened a payload of its own, that empty file is what `_render_ab` reads, so
        `ab.md` is written out of another run's rows and the first run still exits 0.

    The duplicate is driven through the real `run()`, from inside the reporting window, so it meets
    the guard exactly where a second launcher meets it in the field. `flock` treats two descriptors
    on one file independently even within a process, so the in-process contender is refused by the
    same kernel lock a separate launcher is.
    """

    paths = Paths.under(studio["out"])
    real_render = sb._render_ab
    seen: dict = {}

    def render_with_a_duplicate_arriving(*args, **kwargs):
        if "duplicate" not in seen:
            seen["installed_before"] = list(studio["installed"])
            seen["payload_before"] = paths.payload_jsonl.read_text(encoding = "utf-8")
            try:
                seen["duplicate"] = sb.run(_args(studio, "--branch", "main"))
            except SystemExit as exc:
                seen["duplicate"] = exc
        return real_render(*args, **kwargs)

    monkeypatch.setattr(sb, "_render_ab", render_with_a_duplicate_arriving)
    assert sb.run(_args(studio, "--branch", "main", "--ab", "pr-9296"), ab_ref = "pr-9296") == 0

    assert isinstance(
        seen["duplicate"], SystemExit
    ), "a second run was admitted to the output directory while the first was still reporting"
    assert "still running" in str(seen["duplicate"])
    # Nothing of the first run's was moved, and nothing was installed on top of it.
    assert paths.payload_jsonl.exists(), "the duplicate archived the payload being reported on"
    assert paths.payload_jsonl.read_text(encoding = "utf-8") == seen["payload_before"]
    assert sorted(p.name for p in paths.out.glob("payload-*.jsonl")) == []
    assert studio["installed"] == seen["installed_before"]
    # The report is the first run's own, over its own rows.
    table = (paths.out / "ab.md").read_text(encoding = "utf-8")
    assert "main -> pr-9296" in table
    # And the control: the directory is released once `run()` has actually finished with it.
    assert sb.run(_args(studio, "--branch", "main")) == 0


def _clean_summary(studio) -> Path:
    """A clean run, then the `--report` the README quickstart runs on it, into the same `--out`."""

    paths = Paths.under(studio["out"])
    assert sb.run(_args(studio, "--branch", "main")) == 0
    assert sb.main(["--report", str(paths.payload_jsonl), "--tier", "quick", "--rungs", "1K"]) == 0
    summary = paths.out / "summary.md"
    assert summary.exists()
    assert "studiobench summary" in summary.read_text(encoding = "utf-8")
    return summary


def test_a_fresh_probe_run_replaces_the_summary_it_inherited(studio, monkeypatch, tmp_path):
    """REGRESSION. A clean summary may not sit at the standard path over a probed payload.

    `archive_payload` moves `payload.jsonl` and nothing else, so the `summary.md` an earlier
    `--report` of this directory wrote stayed where every reader opens it while the payload it
    described was moved aside and a probed one took its place. Nothing later corrected it: a probe
    run is read through the probe's own console output, so `--report`, whose `SystemExit` clause
    does replace the file, is the one command nobody has a reason to run on that payload. Without
    `--ab` there is no `ab.md` either, so the stale summary was the only report-shaped file in the
    directory.
    """

    monkeypatch.delenv("SBENCH_EXTRA_INIT_SCRIPT", raising = False)
    summary = _clean_summary(studio)
    clean = summary.read_text(encoding = "utf-8")

    script = tmp_path / "paint_counter.js"
    script.write_text("window.__probe_ticks = 0;\n", encoding = "utf-8")
    monkeypatch.setattr(browser_mod, "launch", lambda *a, **k: _ProbedBundle())
    monkeypatch.setenv("SBENCH_EXTRA_INIT_SCRIPT", str(script))
    assert sb.run(_args(studio, "--branch", "main")) == 0

    text = summary.read_text(encoding = "utf-8")
    assert text != clean
    assert "NO SUMMARY" in text
    assert script.name in text
    assert "studiobench summary" not in text
    # The evidence itself is untouched: the refusal replaces the report, never the payload the
    # summary described, which is still on disk under its archived name.
    assert len(list(Paths.under(studio["out"]).out.glob("payload-*.jsonl"))) == 1


def test_a_fresh_single_arm_probe_run_replaces_the_ab_table_it_inherited(
    studio, monkeypatch, tmp_path
):
    """REGRESSION. `_render_ab`'s own probe refusal cannot reach this case.

    That function runs only under `if ab_ref`, so a fresh SINGLE-ARM probe run into a directory
    an earlier `--ab` run left behind never calls it, and the clean table survives beside the new
    unscorable payload. `archive_payload` moves only `payload.jsonl`, so nothing else touches it
    either. Distinct from the resumed-A/B hole fixed in 52fc3e848, where `_render_ab` did run and
    an early return jumped over its refusal.
    """

    monkeypatch.delenv("SBENCH_EXTRA_INIT_SCRIPT", raising = False)
    table = Paths.under(studio["out"]).out / "ab.md"
    table.write_text(
        "studiobench A/B\n===============\n\n  headline_ratio 0.923 (7.7% faster)\n",
        encoding = "utf-8",
    )
    clean = table.read_text(encoding = "utf-8")

    script = tmp_path / "paint_counter.js"
    script.write_text("window.__probe_ticks = 0;\n", encoding = "utf-8")
    monkeypatch.setattr(browser_mod, "launch", lambda *a, **k: _ProbedBundle())
    monkeypatch.setenv("SBENCH_EXTRA_INIT_SCRIPT", str(script))
    # No --ab, so _render_ab is never called and only this refusal can reach the file.
    assert sb.run(_args(studio, "--branch", "main")) == 0

    text = table.read_text(encoding = "utf-8")
    assert text != clean
    assert "NO TABLE" in text
    assert script.name in text
    assert "headline_ratio" not in text, "the clean A/B table survived a probe run"


def test_a_probe_run_invents_no_ab_table_where_there_was_none(studio, monkeypatch, tmp_path):
    """The control, matching the summary one: replace a stale table, never invent one."""

    script = tmp_path / "paint_counter.js"
    script.write_text("window.__probe_ticks = 0;\n", encoding = "utf-8")
    monkeypatch.setattr(browser_mod, "launch", lambda *a, **k: _ProbedBundle())
    monkeypatch.setenv("SBENCH_EXTRA_INIT_SCRIPT", str(script))
    assert sb.run(_args(studio, "--branch", "main")) == 0
    assert not (Paths.under(studio["out"]).out / "ab.md").exists()


def test_a_probe_run_invents_no_summary_where_there_was_none(studio, monkeypatch, tmp_path):
    """The control. The refusal replaces a stale report; it does not create a report-shaped file
    in a directory whose reader was never given one to misread."""

    script = tmp_path / "paint_counter.js"
    script.write_text("window.__probe_ticks = 0;\n", encoding = "utf-8")
    monkeypatch.setattr(browser_mod, "launch", lambda *a, **k: _ProbedBundle())
    monkeypatch.setenv("SBENCH_EXTRA_INIT_SCRIPT", str(script))
    assert sb.run(_args(studio, "--branch", "main")) == 0
    assert not (Paths.under(studio["out"]).out / "summary.md").exists()


def test_a_clean_rerun_also_invalidates_the_summary_it_inherited(studio, monkeypatch):
    """REGRESSION, and this test previously asserted the opposite.

    It was written as a control reading "only a PROBE run invalidates", on the reasoning that
    `--report` rewrites the summary properly afterwards. That reasoning holds only for a reader
    who runs `--report`, and it is the same asymmetry in reverse that made the probe case a bug:
    `archive_payload` moves `payload.jsonl` and nothing else, so after ANY rerun of this directory
    the standing `summary.md` describes a payload that is no longer at the path it names. A plain
    run writes `summary.md` never and `ab.md` only under `--ab`, so a single-arm rerun produces no
    report-shaped file to displace it.

    The sharper version of the same hole is probed-then-clean: the probe refusal says in so many
    words that the payload beside it is not scorable, and that claim survives into a directory
    whose payload is now perfectly scorable. A refusal that outlives its reason is read as a
    finding about the run that is actually there.

    `--report` still writes a real summary over it, which is the half of the original control
    that was correct and is kept below.
    """

    monkeypatch.delenv("SBENCH_EXTRA_INIT_SCRIPT", raising = False)
    summary = _clean_summary(studio)
    paths = Paths.under(studio["out"])

    assert sb.run(_args(studio, "--branch", "main")) == 0
    text = summary.read_text(encoding = "utf-8")
    assert "NO SUMMARY" in text
    assert "studiobench summary" not in text, "a summary of the archived payload survived a rerun"
    # It names where the payload it described went, so the reader can still reach it.
    assert "payload-" in text

    # And the legitimate `--report` path still scores the new payload rather than refusing it.
    assert sb.main(["--report", str(paths.payload_jsonl), "--tier", "quick", "--rungs", "1K"]) == 0
    assert "studiobench summary" in summary.read_text(encoding = "utf-8")


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


# ── one Studio, two sides: the attached null control ────────────────────────────────────────


class _ProviderBackend:
    """`lifecycle.register_provider`'s contract, per origin: idempotent by DISPLAY NAME.

    The real one deletes every existing provider whose `display_name` matches before creating the
    replacement, so registering the pacer twice against ONE Studio destroys the id the first
    registration handed out. Modelled rather than stubbed to a constant, because that deletion is
    the whole failure.
    """

    def __init__(self) -> None:
        self.live: dict = {}
        self.registrations: list = []
        self.deleted: list = []
        self._issued = 0

    def register(self, base_url, auth, provider):
        origin = base_url.rstrip("/")
        rows = self.live.setdefault(origin, {})
        for pid, name in list(rows.items()):
            if name == provider.name:
                del rows[pid]
                self.deleted.append((origin, pid))
        self._issued += 1
        pid = f"provider-{self._issued}"
        rows[pid] = provider.name
        provider.id = pid
        self.registrations.append((origin, pid))
        return pid


def _capture_init_scripts(monkeypatch) -> list:
    captured: list = []

    def fake_launch(
        engine,
        *args,
        init_scripts = None,
        **kwargs,
    ):
        captured.extend(init_scripts or [])
        return _Bundle()

    monkeypatch.setattr(browser_mod, "launch", fake_launch)
    return captured


def _selected_provider_ids(scripts: list, origin: str) -> set:
    """The provider ids the seed scripts for `origin` actually SELECT, out of their checkpoints."""

    import re

    ids = set()
    for script in scripts:
        if json.dumps(origin) not in script:
            continue
        for match in re.finditer(r"external::([^:]+)::", script):
            ids.add(match.group(1))
    return ids


def test_an_attached_null_control_registers_one_provider_for_the_one_studio(studio, monkeypatch):
    """`--attach U --attach-b U` is TWO SIDES ON ONE STUDIO, which `is_null_control` accepts.

    Registering per side registered the pacer twice against that single backend, and the second
    registration deleted the id the base side's seed script had already captured. Both scripts are
    scoped to the same origin, Playwright does not define the order init scripts run in, and
    `StudioAuth.rotate` re-adds them mid-run -- so the base could boot every cell with a DELETED
    provider selected, which renders as "No longer offered" and throws `Connection not found`
    without ever asking for a completion.
    """

    backend = _ProviderBackend()
    monkeypatch.setattr(lifecycle, "register_provider", backend.register)
    scripts = _capture_init_scripts(monkeypatch)
    url = "http://127.0.0.1:5310"

    args = _args(studio, "--attach", url, "--attach-b", url, "--branch", "main", "--ab", "main")
    assert sb.run(args, ab_ref = "main") == 0

    # THE SYMPTOM FIRST. Whatever the bookkeeping below says, the failure a cell actually meets is
    # a seed script that SELECTS a provider the backend no longer has, so that is what this pins:
    # every id named by a script scoped to this origin is still live there, and no live id is
    # unnamed. Asserting only "nothing was deleted" would pass a future fix that deleted and then
    # re-seeded both scripts with the survivor, and would fail a fix that never selects at all.
    selected = _selected_provider_ids(scripts, url)
    assert selected == set(backend.live[url])
    assert selected, "the one Studio must still have a provider selected"
    # And the registration itself: one backend, one registration, nothing destroyed.
    assert backend.deleted == []
    assert backend.registrations == [(url, "provider-1")]


def test_two_attached_studios_still_get_a_provider_each(studio, monkeypatch):
    """The control: two DIFFERENT origins are two backends and must each be registered."""

    backend = _ProviderBackend()
    monkeypatch.setattr(lifecycle, "register_provider", backend.register)
    scripts = _capture_init_scripts(monkeypatch)
    base_url, treatment_url = "http://127.0.0.1:5310", "http://127.0.0.1:5311"

    args = _args(studio, "--attach", base_url, "--attach-b", treatment_url, "--ab", "fix")
    assert sb.run(args, ab_ref = "fix") == 0

    assert backend.registrations == [(base_url, "provider-1"), (treatment_url, "provider-2")]
    assert _selected_provider_ids(scripts, base_url) == {"provider-1"}
    assert _selected_provider_ids(scripts, treatment_url) == {"provider-2"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
