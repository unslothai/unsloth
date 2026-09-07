# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Dispatch and collect: the ways this trades forty minutes for nothing.

The GPU jobs no longer wait for their Kaggle kernel, buying back 41.5 and 18.7
minutes of runner per commit (runs 33479481067 and 33486360729). Every test
here aims at the one new class of bug: **the result stops arriving and nothing
says so.** Four green shapes of that:

1. the dispatching job reports success and branch protection still requires IT,
   so the check now means "Kaggle accepted a push";
2. the kernel is never collected, so it bills quota to its ceiling and the
   commit keeps a status that never resolves;
3. the collector deletes a kernel before reading it, or one never ours;
4. the slug loses the commit, so a finished result cannot be attributed.

Everything runs on CPU against stubs; no Kaggle session is spent.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_DIR = REPO_ROOT / ".github" / "scripts" / "kaggle_t4_ci"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
NOTEBOOK_WF = WORKFLOWS / "kaggle-t4-notebook-ci.yml"
STUDIO_WF = WORKFLOWS / "kaggle-t4-studio-gpu-ci.yml"
COLLECT_WF = WORKFLOWS / "kaggle-collect.yml"

sys.path.insert(0, str(CI_DIR))

import collect  # noqa: E402
import launch  # noqa: E402
import post_statuses  # noqa: E402


def _wf(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding = "utf-8"))


def _steps(workflow: dict) -> list[tuple[str, str, dict]]:
    out = []
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", []) or []:
            out.append((job_name, step.get("name", ""), step))
    return out


# ------------------------------------------------------- the slug is the record


def test_the_slug_carries_the_commit_and_the_workflow():
    """Without this the collector has a finished kernel and nowhere to report it.

    The slug is the ONLY thing surviving the dispatching runner, so it must say
    which commit and which workflow, and be readable back.
    """
    name = launch.slug_name("notebook", "1a2b3c4d5e6f7890")
    parsed = launch.parse_slug(name)
    assert parsed is not None, f"{name} does not parse as one of ours"
    assert parsed["sha"] == "1a2b3c4d5e6f", parsed
    assert parsed["kind"] == "notebook", parsed


def test_the_slug_carries_twelve_hex_characters_and_still_reads_eight():
    """Eight characters is what two reachable commits can share, and a shared
    prefix answers the commits API with a 422 on every pass, so the status
    stays pending forever. Twelve is written; eight is still read, because
    kernels pushed with the old form can outlive the change."""
    assert launch.SLUG_SHA_LEN == 12
    name = launch.slug_name("studio", "a" * 40)
    assert launch.parse_slug(name)["sha"] == "a" * 12
    old = launch.parse_slug("me/unsloth-t4-ci-nabcdef01-1111")
    assert old is not None and old["sha"] == "abcdef01" and old["kind"] == "notebook"
    # Too short to be unambiguous is not written into a slug at all: it falls
    # back to the unattributable form rather than inventing a prefix.
    assert launch.parse_slug(launch.slug_name("notebook", "abcdef01"))["legacy"] is True


def test_in_flight_matches_the_old_and_the_new_slug_forms(tmp_path, monkeypatch):
    """A kernel pushed with the eight-character form must still count as in
    flight for its full commit, and a twelve-character one for the same."""
    full = "abcdef0123456789" + "0" * 24
    for slug_sha in ("abcdef01", "abcdef012345"):
        slug = f"danielhanchen/unsloth-t4-ci-n{slug_sha}-1111"
        api = _StubApi([_StubKernel(slug)], {slug: "RUNNING"})
        monkeypatch.setattr(launch, "_api", lambda api = api: api)
        monkeypatch.setattr(
            sys,
            "argv",
            ["collect", "--outdir", str(tmp_path / slug_sha), "--sha", full, "--kind", "notebook"],
        )
        assert collect.main() == 0
        result = json.loads((tmp_path / slug_sha / "collect_result.json").read_text())
        assert result["in_flight_for_sha"] is True, (slug_sha, result)


def test_two_dispatches_of_one_commit_do_not_collide():
    """A re-run, or notebook slot 1 and 2, must not push to the same slug.

    Pushing to an existing id files a new version and starts a SECOND session
    while the first runs on, and status/output answer for the newest only, so
    one session burns a slot unseen. See push()'s own docstring.
    """
    names = {launch.slug_name("notebook", "deadbeefcafe") for _ in range(200)}
    assert len(names) > 190, f"only {len(names)} distinct slugs in 200 draws"


def test_the_slug_still_round_trips_through_slugify():
    """Kaggle files the kernel under the slugified TITLE, not the metadata id.

    A name that does not round-trip files the kernel at an unexpected address
    where every later status and output call 403s.
    """
    for kind in ("notebook", "studio"):
        name = launch.slug_name(kind, "0123456789ab")
        assert launch._slugify(name.replace("-", " ")) == name, name


def test_the_gate_still_recognises_a_dispatched_kernel_as_ours():
    """THE ONE THAT KEEPS THE CONCURRENCY CONTROL WORKING.

    The GitHub concurrency group no longer bounds Kaggle sessions: the
    dispatching job exits and releases it while the kernel runs on. What bounds
    them is the gate's survey of kernels whose slug carries OWN_KERNEL_PREFIX.
    Drop that prefix and three pushes in ten minutes each believe the account
    idle, and Kaggle refuses the third at its 2-session cap.
    """
    import gate
    for kind in ("notebook", "studio"):
        name = launch.slug_name(kind, "abcdef01")
        assert name.startswith(gate.OWN_KERNEL_PREFIX), (
            f"{name} does not start with the gate's {gate.OWN_KERNEL_PREFIX!r}, so the "
            "gate would count a dispatched kernel as somebody else's or not at all"
        )


@pytest.mark.parametrize(
    "foreign",
    (
        "danielhanchen/unsloth-probe-vision-train-r3-6dd742",
        "someone/my-important-notebook",
        "danielhanchen/unsloth-t4-cixyz",
        "danielhanchen/unsloth-studio-ci-12345678",
        "",
    ),
)
def test_a_kernel_that_is_not_ours_is_invisible(foreign):
    """The collector enumerates a whole ACCOUNT and DELETES what it collects.

    The account is shared with a human, so there is deliberately no heuristic
    here: a false positive deletes somebody's work and looks exactly like a
    kernel that finished.
    """
    assert launch.parse_slug(foreign) is None, f"{foreign!r} was accepted as one of ours"


def test_a_legacy_slug_is_still_reapable_but_reports_nothing():
    """Kernels pushed before this change can outlive it.

    Still ours and still reapable, or they bill forever, but they carry no
    commit to attribute a verdict to and inventing one is worse than silence.
    """
    parsed = launch.parse_slug("danielhanchen/unsloth-t4-ci-d7faf2b8")
    assert parsed is not None and parsed["legacy"] is True
    assert parsed["sha"] is None and parsed["kind"] is None
    assert collect.statuses_from([{**parsed, "slug": "x", "verdict": "pass", "reason": "ok"}]) == []


# ------------------------------------------------------------ dispatch refuses


def test_dispatch_without_a_commit_is_a_usage_error(monkeypatch, capsys):
    """A dispatch whose slug carries no commit runs, costs quota, and reports to
    nobody. Refused at the only moment it is still free."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["launch.py", "--notebook", "k.ipynb", "--outdir", "out", "--dispatch"],
    )
    with pytest.raises(SystemExit) as excinfo:
        launch.main()
    assert excinfo.value.code != 0
    assert "commit-sha" in capsys.readouterr().err


def test_dispatch_does_not_delete_the_kernel_it_pushed():
    """The whole point: the kernel is LEFT RUNNING.

    Through the existing --keep-kernel flag rather than a second condition in
    release(); two guards on one deletion is how one ends up wrong, and this one
    bills GPU quota when it does.
    """
    src = (CI_DIR / "launch.py").read_text(encoding = "utf-8")
    assert re.search(r"if args\.dispatch:\s*\n\s*args\.keep_kernel = True", src), (
        "dispatch mode must set keep_kernel, or release() deletes the kernel it "
        "was told to leave running and the collector finds nothing"
    )


def test_dispatch_does_not_report_pass():
    """`dispatched` is not `pass`, and the difference is the whole change.

    Nothing has run when the dispatch returns, so `pass` would be worse than the
    silent skip this repo has been caught by twice: it looks like a result.
    """
    src = (CI_DIR / "launch.py").read_text(encoding = "utf-8")
    block = src[src.index("if args.dispatch:", src.index('result["slug"] = live[0]')) :]
    block = block[: block.index("return finish()")]
    assert '"dispatched"' in block, block[:400]
    assert 'result["verdict"] = "pass"' not in block


def test_the_dispatch_worst_case_excludes_the_phases_it_never_runs():
    """Otherwise the pre-push guard demands two hours of job deadline for a job
    that exits in five minutes, and stands runs down for a window they do not
    need -- the opposite of what dispatch mode is for."""
    full = launch.worst_case_seconds(5400, 1)
    quick = launch.worst_case_seconds(5400, 1, dispatch = True)
    assert quick < full, (quick, full)


# ---------------------------------------------------------------- the collector


class _StubKernel:
    def __init__(
        self,
        ref,
        last_run_time = None,
    ):
        self.ref = ref
        self.last_run_time = last_run_time


class _StubApi:
    CONFIG_NAME_USER = "username"

    def __init__(self, kernels, statuses):
        self.config_values = {"username": "danielhanchen"}
        self._kernels = kernels
        self._statuses = statuses
        self.status_calls: list[str] = []

    def kernels_list(
        self,
        mine = True,
        page = 1,
        page_size = 100,
        sort_by = "dateRun",
    ):
        return self._kernels if page == 1 else []

    def kernels_status(self, ref):
        self.status_calls.append(ref)
        state = self._statuses[ref]
        if isinstance(state, Exception):
            raise state
        return type("S", (), {"status": f"KernelWorkerStatus.{state}"})()


def test_a_running_kernel_within_its_ceiling_is_left_completely_alone(tmp_path):
    """No delete, no status, not even a warning. A kernel doing its job is not
    a problem, and reporting one would put a red on a commit whose result is
    still coming."""
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": "RUNNING"})
    deleted: list[str] = []
    entry = {
        "slug": "me/unsloth-t4-ci-nabcdef01-1111",
        "sha": "abcdef01",
        "kind": "notebook",
        "legacy": False,
        "age_hours": 0.5,
    }
    record = collect.collect_one(api, entry, tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["verdict"] == "pending"
    assert record["deleted"] is False
    assert collect.statuses_from([record]) == [], "a running kernel must post no status"
    assert deleted == []


def test_a_kernel_past_its_ceiling_is_reaped_and_reported(tmp_path, monkeypatch):
    """THE REASON THE SCHEDULED COLLECTOR EXISTS.

    The old design deleted every kernel it pushed on every path out. Without
    that a wedged kernel bills to its ceiling unwatched; one here was measured
    ignoring Kaggle's own `-t` timeout for over two hours. Reported as a failure
    rather than dropped, or the commit stays pending forever.
    """
    deleted: list[str] = []
    monkeypatch.setattr(
        launch, "delete_kernel", lambda slug, deadline = None: deleted.append(slug) or True
    )
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": "RUNNING"})
    entry = {
        "slug": "me/unsloth-t4-ci-nabcdef01-1111",
        "sha": "abcdef01",
        "kind": "notebook",
        "legacy": False,
        "age_hours": 9.0,
    }
    record = collect.collect_one(api, entry, tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["verdict"] == "reaped"
    assert deleted == ["me/unsloth-t4-ci-nabcdef01-1111"]
    status = collect.statuses_from([record])[0]
    assert status["state"] == "failure", status


def test_a_kernel_with_no_timestamp_is_never_reaped(tmp_path, monkeypatch):
    """A missing timestamp is not evidence that a kernel is old, and guessing in
    that direction DELETES A RUNNING SESSION."""
    deleted: list[str] = []
    monkeypatch.setattr(
        launch, "delete_kernel", lambda slug, deadline = None: deleted.append(slug) or True
    )
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": "RUNNING"})
    entry = {
        "slug": "me/unsloth-t4-ci-nabcdef01-1111",
        "sha": "abcdef01",
        "kind": "notebook",
        "legacy": False,
        "age_hours": None,
    }
    record = collect.collect_one(api, entry, tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["verdict"] == "pending"
    assert deleted == []


def test_evidence_is_downloaded_before_the_kernel_is_deleted(tmp_path, monkeypatch):
    """A delete that lands first turns a finished run into a result nobody can
    ever read. Asserted by ORDER of the real calls, not by reading the source."""
    order: list[str] = []
    monkeypatch.setattr(
        launch, "fetch_evidence", lambda slug, dest, deadline = None: order.append("fetch") or {}
    )
    monkeypatch.setattr(launch, "extract_reports", lambda dest: [{"passed": True}])
    monkeypatch.setattr(
        launch, "delete_kernel", lambda slug, deadline = None: order.append("delete") or True
    )
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": "COMPLETE"})
    entry = {
        "slug": "me/unsloth-t4-ci-nabcdef01-1111",
        "sha": "abcdef01",
        "kind": "notebook",
        "legacy": False,
        "age_hours": 0.4,
    }
    collect.collect_one(api, entry, tmp_path, expect = 1, max_age_hours = 3.0)
    assert order == ["fetch", "delete"], order


def test_a_kernel_whose_evidence_will_not_download_is_NOT_deleted(tmp_path, monkeypatch):
    """Deleting here destroys the only copy of a finished run's result, to
    reclaim a session slot the kernel is no longer using. The next pass retries."""

    def _boom(
        slug,
        dest,
        deadline = None,
    ):
        raise TimeoutError("slow")

    deleted: list[str] = []
    monkeypatch.setattr(launch, "fetch_evidence", _boom)
    monkeypatch.setattr(
        launch, "delete_kernel", lambda slug, deadline = None: deleted.append(slug) or True
    )
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": "COMPLETE"})
    entry = {
        "slug": "me/unsloth-t4-ci-nabcdef01-1111",
        "sha": "abcdef01",
        "kind": "notebook",
        "legacy": False,
        "age_hours": 0.4,
    }
    record = collect.collect_one(api, entry, tmp_path, expect = 1, max_age_hours = 3.0)
    # `pending`, not `infra`: an infra verdict posts green and is released by
    # --delete-collected, breaking the "next pass retries" promise.
    assert record["verdict"] == "pending"
    assert record["verdict"] not in collect.DELETABLE
    assert collect.statuses_from([record]) == []
    assert deleted == [], "evidence that would not download must not be deleted"


def test_an_unreadable_status_does_nothing_at_all(tmp_path, monkeypatch):
    """Both available actions are destructive: delete and we may kill a running
    session, report and we may fail a run that was fine. Ask again next pass."""
    deleted: list[str] = []
    monkeypatch.setattr(
        launch, "delete_kernel", lambda slug, deadline = None: deleted.append(slug) or True
    )
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": RuntimeError("503 upstream")})
    entry = {
        "slug": "me/unsloth-t4-ci-nabcdef01-1111",
        "sha": "abcdef01",
        "kind": "notebook",
        "legacy": False,
        "age_hours": 0.4,
    }
    record = collect.collect_one(api, entry, tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["verdict"] == "pending"
    assert deleted == []
    assert collect.statuses_from([record]) == []


def test_find_ours_skips_everything_it_does_not_recognise():
    """Driven against a listing that mixes a human's kernels with ours, because
    a rule fed a hand-written list of slugs proves nothing about the filter."""
    listing = [
        _StubKernel("danielhanchen/unsloth-probe-vision-train-r3-6dd742"),
        _StubKernel("danielhanchen/unsloth-t4-ci-nabcdef01-1111"),
        _StubKernel("danielhanchen/my-actual-work"),
        _StubKernel("danielhanchen/unsloth-t4-ci-d7faf2b8"),
    ]
    api = _StubApi(listing, {})
    ours = [k["slug"] for k in collect.find_ours(api)]
    assert ours == [
        "danielhanchen/unsloth-t4-ci-nabcdef01-1111",
        "danielhanchen/unsloth-t4-ci-d7faf2b8",
    ], ours


def test_infra_and_partial_do_not_go_red():
    """A red the author cannot act on is how a required check gets REMOVED from
    branch protection. `infra` and `partial` learned nothing about the code, so
    they stay green and say so in the description."""
    assert collect.VERDICT_STATE["infra"] == "success"
    assert collect.VERDICT_STATE["partial"] == "success"
    assert collect.VERDICT_STATE["fail"] == "failure"
    assert collect.VERDICT_STATE["reaped"] == "failure"


def test_a_failed_payload_reaches_github_as_a_failure():
    """Driven end to end through the real verdict and status functions, since
    every rule above this one is fed a dict written by hand."""
    record = {
        "slug": "me/unsloth-t4-ci-nabcdef01-1111",
        "sha": "abcdef01",
        "kind": "notebook",
        "verdict": None,
        "reason": "",
    }
    verdict, reason = collect.verdict_of([{"passed": False, "payload": "gptoss"}], 1)
    record["verdict"], record["reason"] = verdict, reason
    status = collect.statuses_from([record])[0]
    assert status["state"] == "failure"
    assert status["context"] == "kaggle-t4-notebook"
    assert "gptoss" in status["description"]


def test_the_collector_never_sees_a_github_token():
    """A script holding both a Kaggle and a GitHub credential is one bug away
    from sending one to the other. It emits statuses as DATA; the workflow
    posts them."""
    src = (CI_DIR / "collect.py").read_text(encoding = "utf-8")
    for forbidden in ("GH_TOKEN", "GITHUB_TOKEN", "api.github.com"):
        assert forbidden not in src, f"collect.py references {forbidden}"


# ------------------------------------------------------------- the workflows


@pytest.mark.parametrize(
    "path,kind,context",
    ((NOTEBOOK_WF, "notebook", "kaggle-t4-notebook"), (STUDIO_WF, "studio", "kaggle-studio-gpu")),
    ids = ("notebook", "studio"),
)
def test_the_gpu_job_dispatches_rather_than_waits(path, kind, context):
    """The measured win, asserted from the workflow rather than from a comment."""
    body = path.read_text(encoding = "utf-8")
    launch_calls = re.findall(r"kaggle_t4_ci/launch\.py \\\n(?:.*\n)*?(?=\n)", body)
    assert launch_calls, f"{path.name} does not invoke launch.py"
    for call in launch_calls:
        assert "--dispatch" in call, f"{path.name} still waits for its kernel:\n{call}"
        assert "--commit-sha" in call, f"{path.name} dispatches without a commit:\n{call}"
        assert f"--kind {kind}" in call, f"{path.name} dispatches without its kind:\n{call}"


@pytest.mark.parametrize(
    "path,context",
    ((NOTEBOOK_WF, "kaggle-t4-notebook"), (STUDIO_WF, "kaggle-studio-gpu")),
    ids = ("notebook", "studio"),
)
def test_the_dispatching_job_can_post_the_status_that_replaces_it(path, context):
    """The job succeeds by dispatching, so the verdict has to travel some other
    way. Without `statuses: write` the collection runs and reaches nobody."""
    wf = _wf(path)
    gpu_jobs = [j for name, j in wf["jobs"].items() if name != "gate"]
    assert gpu_jobs, path.name
    for job in gpu_jobs:
        perms = job.get("permissions") or {}
        assert perms.get("statuses") == "write", (
            f"{path.name}'s GPU job cannot post commit statuses, so the verdict "
            "produced by its collection step is discarded"
        )


@pytest.mark.parametrize("path", (NOTEBOOK_WF, STUDIO_WF), ids = ("notebook", "studio"))
def test_the_gpu_job_collects_before_it_dispatches(path):
    """Order matters twice: it frees the session slot this job is about to want,
    and it answers whether this commit is already running, so a re-run does not
    pay for a second session."""
    body = path.read_text(encoding = "utf-8")
    assert "kaggle_t4_ci/collect.py" in body, f"{path.name} never collects"
    assert body.index("kaggle_t4_ci/collect.py") < body.index(
        "--dispatch"
    ), f"{path.name} dispatches before it collects"


@pytest.mark.parametrize("path", (NOTEBOOK_WF, STUDIO_WF), ids = ("notebook", "studio"))
def test_a_commit_already_in_flight_is_not_dispatched_again(path):
    """Two sessions for one result is quota spent twice for an answer already
    coming, and Kaggle's 2-session cap means the second may take the slot the
    first needs."""
    for _job, name, step in _steps(_wf(path)):
        if "--dispatch" in (step.get("run") or ""):
            assert "in_flight != 'true'" in (step.get("if") or ""), (
                f"{path.name}'s dispatch step is not gated on the in-flight check: "
                f"{step.get('if')!r}"
            )
            return
    pytest.fail(f"{path.name} has no dispatch step")


def test_a_scheduled_collector_exists_and_reaps():
    """The other route only fires when somebody pushes, so a quiet weekend with
    a wedged kernel is a weekend of unwatched billing. The old design could not
    have that bug because the pushing job also deleted."""
    wf = _wf(COLLECT_WF)
    on = wf.get(True) or wf.get("on")
    assert "schedule" in on, "the collector does not run on a schedule, so a quiet repo never reaps"
    assert on["schedule"], on


def test_the_collector_covers_both_accounts():
    """A kernel dispatched on either account has to be collected. The gate's
    weighted draw decides where a session is SPENT; it must not decide where
    results are read from."""
    wf = _wf(COLLECT_WF)
    job = wf["jobs"]["collect"]
    names = {e["secret_name"] for e in job["strategy"]["matrix"]["include"]}
    import gate

    assert names == set(gate.DEFAULT_ACCOUNT_ENVS), names


def test_the_collector_holds_one_token_per_job():
    """Same rule as the GPU workflows. A step holding both could authenticate as
    either, and would then delete kernels belonging to an account it is not
    reporting for."""
    for _job, name, step in _steps(_wf(COLLECT_WF)):
        env = step.get("env") or {}
        tokens = sorted(k for k in env if k.startswith("KAGGLE_API_TOKEN"))
        if tokens:
            assert tokens == ["KAGGLE_API_TOKEN"], f"{name} sees {tokens}"


def test_the_collector_serialises_per_account():
    """Two collectors on one account race: the loser deletes what the winner
    already deleted and reports the 404 as a failure."""
    wf = _wf(COLLECT_WF)
    conc = wf.get("concurrency")
    assert conc, "the collector has no concurrency group"
    assert conc.get("cancel-in-progress") is False, (
        "a pass midway through downloading evidence must finish; cancelling it "
        "throws away work the next pass has to redo"
    )


def test_the_status_contexts_are_stable_strings():
    """Branch protection is configured against these names, and renaming one
    silently stops requiring it: a context that never reports is not a failing
    context."""
    assert collect.STATUS_CONTEXTS == {
        "notebook": "kaggle-t4-notebook",
        "studio": "kaggle-studio-gpu",
    }
    for path, context in ((NOTEBOOK_WF, "kaggle-t4-notebook"), (STUDIO_WF, "kaggle-studio-gpu")):
        assert context in path.read_text(encoding = "utf-8"), (
            f"{path.name} never mentions {context}, so its pending status is posted "
            "under a name nothing else uses"
        )


def test_a_dispatch_posts_a_pending_status():
    """Without it a required context that has never reported blocks the pull
    request with no explanation, indistinguishable from an unconfigured check."""
    for path, context in ((NOTEBOOK_WF, "kaggle-t4-notebook"), (STUDIO_WF, "kaggle-studio-gpu")):
        body = path.read_text(encoding = "utf-8")
        assert re.search(
            r"state=pending -f context=" + re.escape(context), body
        ), f"{path.name} never posts a pending {context} status"


def test_the_reporters_wait_for_an_EXECUTED_NOTEBOOK_not_just_a_directory():
    """Measured on run 33628507954, which reported `Kaggle T4 smoke: PARTIAL`
    for a dispatch where nothing had run.

    `hashFiles('kaggle_evidence/**')` is true on a dispatching run, since
    launch.py writes launch_result.json there before exiting, so the reporters
    print "half a comparison" for a queued kernel. The condition has to name
    what a report is MADE of: an executed notebook, which only a collected run
    has.
    """
    for path in (NOTEBOOK_WF, STUDIO_WF):
        for _job, name, step in _steps(_wf(path)):
            body = step.get("run") or ""
            if "report.py" not in body:
                continue
            condition = step.get("if") or ""
            assert "_output.ipynb" in condition, (
                f"{path.name}: step {name!r} reports on the presence of the evidence "
                f"DIRECTORY, which a dispatching run creates and does not fill: "
                f"if={condition!r}"
            )


def _fake_gh(
    monkeypatch,
    resolve_to = None,
    post_ok = True,
    lookup_error = "",
):
    """Stand in for `gh`, recording every call. `resolve_to` is what
    `repos/../commits/<sha>` answers; None means gone, in GitHub's own words (a
    422 "No commit found"); `lookup_error` is what a failed lookup says."""
    calls: list[list[str]] = []

    def _gh(args):
        calls.append(list(args))
        if "/commits/" in args[1]:
            if lookup_error:
                return (1, "", lookup_error)
            if resolve_to:
                return (0, resolve_to, "")
            return (1, "", "gh: No commit found for SHA: 2ecb19df (HTTP 422)")
        return (0 if post_ok else 1, "", "")

    monkeypatch.setattr(post_statuses, "_gh", _gh)
    return calls


_STATUS = {
    "sha": "2ecb19df",
    "state": "success",
    "context": "kaggle-t4-notebook",
    "description": "pass: all 5 payload(s) passed",
    "target_url": "https://example/run/1",
    "slug": "me/unsloth-t4-ci-n2ecb19df-1111",
    "slugs": ["me/unsloth-t4-ci-n2ecb19df-1111"],
}


def test_an_abbreviated_sha_is_EXPANDED_before_a_status_is_posted(monkeypatch):
    """MEASURED AGAINST THE REAL API, and it fails closed in the worst way.

        POST /repos/{o}/{r}/statuses/2ecb19df
        422 "Sha must be a valid hex object ID"

    The slug carries only 8 hex characters, so the poster must resolve them
    first; without that every verdict 422s while the collection quietly
    succeeds. Driven through the real poster against a recording `gh`.
    """
    full = "2ecb19df" + "a" * 32
    calls = _fake_gh(monkeypatch, resolve_to = full)
    outcome = post_statuses.post_all([dict(_STATUS)], "unslothai/unsloth")
    assert outcome["ok"] == [_STATUS["slug"]], outcome
    posts = [c for c in calls if "/statuses/" in c[1]]
    assert posts == [posts[0]] and posts[0][1] == f"repos/unslothai/unsloth/statuses/{full}"
    assert not any(c[1].endswith("/statuses/2ecb19df") for c in calls)
    # Every field is its own argument: nothing is assembled into a shell line.
    assert "-f" in posts[0] and f"description={_STATUS['description']}" in posts[0]


def test_a_commit_that_no_longer_exists_is_reported_not_silently_dropped(monkeypatch, capsys):
    """A force-push can remove the commit a running kernel was dispatched for.
    Posting is then impossible, and that is fine, but it must SAY so, and the
    kernel must be released rather than retried forever."""
    calls = _fake_gh(monkeypatch, resolve_to = None)
    outcome = post_statuses.post_all([dict(_STATUS)], "unslothai/unsloth")
    assert outcome["unresolved"] == [_STATUS["slug"]] and outcome["failed"] == []
    assert not any("/statuses/" in c[1] for c in calls), "posted to a commit that is gone"
    assert "Could not resolve a collected commit" in capsys.readouterr().out


def test_every_workflow_posts_through_the_shared_poster_and_none_keeps_a_shell_loop():
    for path in (NOTEBOOK_WF, STUDIO_WF, COLLECT_WF):
        body = path.read_text(encoding = "utf-8")
        assert "kaggle_t4_ci/post_statuses.py" in body, f"{path.name} does not use the poster"
        assert "statuses.txt" not in body, f"{path.name} still carries the tab-delimited loop"


# ---------------------------------------------- post, then delete; never the reverse


@pytest.mark.parametrize(
    "path", (NOTEBOOK_WF, STUDIO_WF, COLLECT_WF), ids = ("notebook", "studio", "collect")
)
def test_collection_never_deletes_and_the_release_step_comes_after_posting(path):
    """A verdict that does not reach GitHub must leave its kernel up for the
    next pass: collect --no-delete, post, then --delete-collected releases only
    what was delivered."""
    body = path.read_text(encoding = "utf-8")
    steps = [(name, step) for _job, name, step in _steps(_wf(path))]
    collect_steps = [
        s
        for _n, s in steps
        if "kaggle_t4_ci/collect.py" in (s.get("run") or "")
        and "--delete-collected" not in s["run"]
    ]
    assert collect_steps, f"{path.name} never collects"
    for step in collect_steps:
        assert (
            "--no-delete" in step["run"]
        ), f"{path.name} collects with deletion on:\n{step['run']}"
    names = [n for n, _s in steps]
    post = next(i for i, (n, s) in enumerate(steps) if "post_statuses.py" in (s.get("run") or ""))
    release = next(
        i for i, (n, s) in enumerate(steps) if "--delete-collected" in (s.get("run") or "")
    )
    assert (
        release > post
    ), f"{path.name} releases kernels before posting: {names[release]!r} before {names[post]!r}"
    assert "--posted kaggle_collected/posted.json" in steps[release][1]["run"]


def test_a_kernel_whose_status_did_not_post_is_KEPT_for_the_next_pass(tmp_path, monkeypatch):
    deleted: list[str] = []
    monkeypatch.setattr(
        launch, "delete_kernel", lambda slug, deadline = None: deleted.append(slug) or True
    )
    result = tmp_path / "collect_result.json"
    result.write_text(
        json.dumps(
            {
                "kernels": [
                    {"slug": "me/ok", "verdict": "pass", "deleted": False},
                    {"slug": "me/refused", "verdict": "fail", "deleted": False},
                    {"slug": "me/legacy", "verdict": "reaped", "deleted": False},
                    {"slug": "me/running", "verdict": "pending", "deleted": False},
                ],
                "statuses": [],
            }
        )
    )
    posted = tmp_path / "posted.json"
    posted.write_text(json.dumps({"ok": ["me/ok"], "failed": ["me/refused"], "unresolved": []}))
    assert collect.delete_collected(result, posted) == 0
    assert deleted == ["me/ok", "me/legacy"], deleted
    outcome = json.loads((tmp_path / "delete_result.json").read_text())
    assert outcome["kept"] == ["me/refused"]


def test_no_delivery_record_at_all_keeps_every_kernel_that_had_something_to_post(
    tmp_path, monkeypatch
):
    """The poster never ran (the job died between the two steps). Deleting
    then would lose every verdict of the pass."""
    deleted: list[str] = []
    monkeypatch.setattr(
        launch, "delete_kernel", lambda slug, deadline = None: deleted.append(slug) or True
    )
    result = tmp_path / "collect_result.json"
    result.write_text(
        json.dumps(
            {
                "kernels": [
                    {"slug": "me/a", "verdict": "pass", "deleted": False},
                    {"slug": "me/nothing-to-post", "verdict": "infra", "deleted": False},
                ],
                "statuses": [{"slug": "me/a", "slugs": ["me/a"]}],
            }
        )
    )
    collect.delete_collected(result, tmp_path / "posted.json")
    assert deleted == ["me/nothing-to-post"], deleted


def test_a_refused_post_is_red_and_recorded(monkeypatch, tmp_path):
    _fake_gh(monkeypatch, resolve_to = "2ecb19df" + "b" * 32, post_ok = False)
    outcome = post_statuses.post_all([dict(_STATUS)], "unslothai/unsloth")
    assert outcome["failed"] == [_STATUS["slug"]]
    result = tmp_path / "r.json"
    result.write_text(json.dumps({"statuses": [dict(_STATUS)]}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "post_statuses",
            "--result",
            str(result),
            "--out",
            str(tmp_path / "p.json"),
            "--repo",
            "o/r",
        ],
    )
    assert post_statuses.main() == 1
    assert json.loads((tmp_path / "p.json").read_text())["failed"] == [_STATUS["slug"]]


def test_the_poster_signs_only_records_it_recognises(monkeypatch, capsys):
    calls = _fake_gh(monkeypatch, resolve_to = "2ecb19df" + "c" * 32)
    bad_state = dict(_STATUS, state = "green")
    bad_context = dict(_STATUS, context = "ci/somebody-else")
    outcome = post_statuses.post_all([bad_state, bad_context], "o/r")
    assert outcome["invalid"] == [_STATUS["slug"], _STATUS["slug"]]
    assert calls == [], "a malformed record reached gh"


# ------------------------------------------------- what a status is built from


def test_two_kernels_for_one_commit_and_context_post_ONE_status_and_a_failure_wins():
    """Notebook slot 1 and slot 2 on one sha: two kernels, one context. Two
    statuses would race, and the last posted would be the visible verdict."""
    records = [
        {
            "slug": "me/one",
            "sha": "abcdef01",
            "kind": "notebook",
            "verdict": "pass",
            "reason": "all passed",
        },
        {
            "slug": "me/two",
            "sha": "abcdef01",
            "kind": "notebook",
            "verdict": "fail",
            "reason": "1 of 5 failed",
        },
        {
            "slug": "me/other",
            "sha": "abcdef02",
            "kind": "notebook",
            "verdict": "pass",
            "reason": "all passed",
        },
    ]
    statuses = collect.statuses_from(records)
    assert len(statuses) == 2
    joint = next(s for s in statuses if s["sha"] == "abcdef01")
    assert joint["state"] == "failure" and sorted(joint["slugs"]) == ["me/one", "me/two"]
    # Order must not matter: the failure wins whichever kernel was listed first.
    statuses = collect.statuses_from(list(reversed(records)))
    assert next(s for s in statuses if s["sha"] == "abcdef01")["state"] == "failure"


def test_a_description_is_one_line():
    record = {
        "slug": "me/k",
        "sha": "abcdef01",
        "kind": "notebook",
        "verdict": "fail",
        "reason": "a\nb\tc",
    }
    assert collect.statuses_from([record])[0]["description"] == "fail: a b c"


# ------------------------------------ the evidence decides, and only complete evidence


def _terminal_entry():
    return {
        "slug": "me/unsloth-t4-ci-nabcdef01-1111",
        "sha": "abcdef01",
        "kind": "notebook",
        "legacy": False,
        "age_hours": 0.4,
    }


def test_an_incomplete_download_judges_nothing_and_keeps_the_kernel(tmp_path, monkeypatch):
    """fetch_evidence flags a spent budget or a lost notebook instead of
    raising. Judging the short set reads a run that lost half its notebooks as
    whatever the surviving half says, and deleting makes that permanent."""
    deleted: list[str] = []
    monkeypatch.setattr(
        launch,
        "fetch_evidence",
        lambda slug, dest, deadline = None: {"notebooks": ["a"], "truncated": True},
    )
    monkeypatch.setattr(launch, "extract_reports", lambda dest: [{"passed": True}])
    monkeypatch.setattr(
        launch, "delete_kernel", lambda slug, deadline = None: deleted.append(slug) or True
    )
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": "COMPLETE"})
    record = collect.collect_one(api, _terminal_entry(), tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["verdict"] == "pending", record
    assert deleted == []
    assert collect.statuses_from([record]) == []


def test_a_kernel_another_collector_finished_first_posts_nothing(tmp_path, monkeypatch):
    """Three workflows collect on one account and nothing serialises them.
    The loser of that race must not post an `infra` on top of the winner's
    real verdict."""

    def _gone(
        slug,
        dest,
        deadline = None,
    ):
        raise RuntimeError(
            "404 Client Error: Not Found for url: https://www.kaggle.com/api/v1/kernels/output"
        )

    deleted: list[str] = []
    monkeypatch.setattr(launch, "fetch_evidence", _gone)
    monkeypatch.setattr(
        launch, "delete_kernel", lambda slug, deadline = None: deleted.append(slug) or True
    )
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": "COMPLETE"})
    record = collect.collect_one(api, _terminal_entry(), tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["verdict"] == "gone", record
    assert deleted == [] and collect.statuses_from([record]) == []


def test_an_unreadable_report_is_infra_not_a_crash(tmp_path, monkeypatch):
    """Raising here writes no result and wedges every later pass on the same
    kernel. Nothing was learned, so `infra`, and the kernel is released."""

    def _boom(dest):
        raise AttributeError("'list' object has no attribute 'get'")

    monkeypatch.setattr(
        launch,
        "fetch_evidence",
        lambda slug, dest, deadline = None: {"notebooks": ["a"], "truncated": False},
    )
    monkeypatch.setattr(launch, "extract_reports", _boom)
    monkeypatch.setattr(launch, "delete_kernel", lambda slug, deadline = None: True)
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": "COMPLETE"})
    record = collect.collect_one(api, _terminal_entry(), tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["verdict"] == "infra" and "could not be read" in record["reason"]
    assert record["deleted"] is True


def test_a_malformed_report_line_is_skipped_by_the_extractor(tmp_path):
    (tmp_path / "kernel.log").write_text(
        'T4_SMOKE_REPORT []\nT4_SMOKE_REPORT "text"\nT4_SMOKE_REPORT {"label": "ok", "passed": true}\n',
        encoding = "utf-8",
    )
    assert launch.extract_reports(tmp_path) == [{"label": "ok", "passed": True}]


def test_the_expected_report_count_travels_inside_the_kernel(tmp_path, monkeypatch):
    """The scheduled collector has no workflow output to read a payload count
    from, and judging a five-payload kernel against `--expect 1` turns a run
    that lost four into a pass. The driver's own sentinel is the record."""

    def _fetch(
        slug,
        dest,
        deadline = None,
    ):
        dest.mkdir(parents = True, exist_ok = True)
        (dest / "kernel.log").write_text(
            'KAGGLE_T4_CI_DRIVER_EXPECT {"reports": 5}\nT4_SMOKE_REPORT {"label": "a", "passed": true}\n',
            encoding = "utf-8",
        )
        return {"notebooks": [], "log": "kernel.log", "truncated": False}

    monkeypatch.setattr(launch, "fetch_evidence", _fetch)
    monkeypatch.setattr(launch, "delete_kernel", lambda slug, deadline = None: True)
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": "COMPLETE"})
    record = collect.collect_one(api, _terminal_entry(), tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["expected"] == 5
    assert record["verdict"] == "partial", record
    # And with no sentinel the caller's number stands.
    (tmp_path / "plain").mkdir()
    assert collect.expected_reports(tmp_path / "plain", 3) == 3


def test_the_built_driver_records_its_own_expected_report_count():
    sys.path.insert(0, str(CI_DIR))
    import build_kernel

    payloads = {
        "t4_a.ipynb": {"cells": []},
        "t4_b.ipynb": {"cells": []},
        "studio_install.ipynb": {"cells": []},
    }

    def _source(nb):
        return "".join("".join(cell.get("source", [])) for cell in nb["cells"])

    driver = _source(build_kernel.build_driver(payloads, 60, cpu_lane = "studio_install.ipynb"))
    assert 'KAGGLE_T4_CI_DRIVER_EXPECT " + json.dumps({"reports": 2})' in driver
    plain = _source(build_kernel.build_driver({"t4_a.ipynb": {"cells": []}}, 60))
    assert 'json.dumps({"reports": 1})' in plain


def test_the_scheduled_collector_no_longer_guesses_the_payload_count():
    body = COLLECT_WF.read_text(encoding = "utf-8")
    assert (
        "--expect" not in body
    ), "the scheduled collector still applies one flat --expect to every kernel"


# ------------------------------------------------------------- the reaper's reach


def test_a_kernel_far_past_the_ceiling_is_still_seen(monkeypatch):
    """A pass delayed six hours by an outage still finds the wedge it exists
    for. Ageing it out of the listing leaves it billing to Kaggle's 12-hour kill
    with its commit pending forever."""
    from datetime import datetime, timedelta

    now = datetime(2026, 9, 6, 12, 0, 0)
    old = _StubKernel("me/unsloth-t4-ci-nabcdef01-1111", last_run_time = now - timedelta(hours = 30))
    api = _StubApi([old], {})
    found = collect.find_ours(api, now = now, max_age_hours = 3.0)
    assert [f["slug"] for f in found] == ["me/unsloth-t4-ci-nabcdef01-1111"]
    assert found[0]["age_hours"] == pytest.approx(30.0)


# ------------------------------------------------------- who is allowed to be quiet


def test_the_scheduled_collector_fails_loudly_when_it_cannot_authenticate(
    tmp_path, monkeypatch, capsys
):
    """Its token is the repository's own, so a green pass that collected
    nothing would hide an expired credential while kernels bill to their
    ceiling and commits stay pending."""

    def _no(*a, **k):
        raise OSError("Could not find kaggle.json")

    monkeypatch.setattr(launch, "_api", _no)
    monkeypatch.setattr(sys, "argv", ["collect", "--outdir", str(tmp_path), "--require-auth"])
    assert collect.main() == 1
    assert "Kaggle authentication failed" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["collect", "--outdir", str(tmp_path)])
    assert collect.main() == 0, "a fork job with the secret withheld is still a skip"


def test_the_scheduled_workflow_asks_for_that():
    body = COLLECT_WF.read_text(encoding = "utf-8")
    assert "--require-auth" in body


def test_the_collector_installs_the_client_the_gpu_workflows_install():
    """1.7.4.5 cannot read KAGGLE_API_TOKEN at all (#9535), so a scheduled pass
    on it authenticated nothing and, before --require-auth, said so to nobody."""
    pins = {}
    for path in (NOTEBOOK_WF, STUDIO_WF, COLLECT_WF):
        found = re.findall(
            r"pip install [^\n]*'kaggle==([0-9][^']*)'", path.read_text(encoding = "utf-8")
        )
        assert found, f"{path.name} pins no kaggle client"
        pins[path.name] = set(found)
    assert (
        len(set().union(*pins.values())) == 1
    ), f"the workflows disagree on the kaggle client: {pins}"


# -------------------------------------------------- what a dispatch is allowed to be


def test_dispatch_refuses_a_ref_that_is_not_a_commit(monkeypatch, capsys):
    """`slug_name` falls back to the legacy unattributable form for anything
    not hex, so a branch here pushes a kernel that runs, bills, reports to
    nobody."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch",
            "--notebook",
            "k.ipynb",
            "--outdir",
            "out",
            "--dispatch",
            "--kind",
            "studio",
            "--commit-sha",
            "main",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        launch.main()
    assert exc.value.code == 2
    assert "hex commit id" in capsys.readouterr().err


def test_the_studio_workflow_resolves_its_ref_to_a_commit_before_dispatching():
    """workflow_dispatch accepts any `unsloth_ref`; the notebook workflow
    already resolved it, the Studio one passed it straight through."""
    for path in (NOTEBOOK_WF, STUDIO_WF):
        step = next(s for _j, n, s in _steps(_wf(path)) if n == "Resolve the ref under test")
        run = step["run"]
        assert (
            "git ls-remote" in run and "[0-9a-f]{40}" in run
        ), f"{path.name} does not resolve refs"
        assert "exit 1" in run, f"{path.name} dispatches on an unresolved ref"
        # A full SHA passes the shape test whether or not the repository has
        # it; only fetching the object says it is there. Without this the
        # Studio leg spent a session on a commit no status could be posted to.
        assert (
            "git fetch --quiet --depth=1 https://github.com/unslothai/unsloth" in run
        ), f"{path.name} dispatches a full SHA without checking the repository serves it"


@pytest.mark.parametrize("path", (NOTEBOOK_WF, STUDIO_WF), ids = ("notebook", "studio"))
def test_pending_is_posted_only_for_a_kernel_that_was_actually_dispatched(path):
    """launch.py exits 0 on every infrastructure stand-down. A pending status
    for a kernel that does not exist is one no collector can ever replace."""
    step = next(
        s for _j, n, s in _steps(_wf(path)) if n == "Mark the dispatch pending on this commit"
    )
    assert step.get("if") == "steps.launch.outputs.verdict == 'dispatched'", step.get("if")


def test_a_lookup_that_could_not_be_made_keeps_the_kernel(monkeypatch, capsys):
    """Only GitHub saying the commit is not there releases a kernel. A 5xx, a
    rate limit or a dropped connection says nothing, and reading it as "gone"
    deletes the only copy of the result."""
    calls = _fake_gh(monkeypatch, lookup_error = "gh: HTTP 503 Service Unavailable")
    outcome = post_statuses.post_all([dict(_STATUS)], "unslothai/unsloth")
    assert outcome["failed"] == [_STATUS["slug"]]
    assert outcome["unresolved"] == []
    assert not any("/statuses/" in c[1] for c in calls)
    assert "Commit lookup failed" in capsys.readouterr().out


def test_resolve_sha_separates_missing_from_unknown(monkeypatch):
    """Measured: a commit that is not there answers HTTP 422 "No commit found
    for SHA", not a 404. Anything else is an unknown, never a missing."""
    answers = {}

    def _gh(args):
        return answers[args[1]]

    monkeypatch.setattr(post_statuses, "_gh", _gh)
    answers["repos/o/r/commits/aa"] = (0, "a" * 40, "")
    answers["repos/o/r/commits/bb"] = (1, "", "gh: No commit found for SHA: bb (HTTP 422)")
    answers["repos/o/r/commits/cc"] = (1, "", "gh: HTTP 502 Bad Gateway")
    answers["repos/o/r/commits/dd"] = (1, "", "")
    # The status code alone is not the answer: 404 is also an unreadable
    # repository, 422 also an ambiguous abbreviation.
    answers["repos/o/r/commits/ee"] = (1, "", "gh: Not Found (HTTP 404)")
    answers["repos/o/r/commits/ff"] = (1, "", "gh: Validation Failed (HTTP 422)")
    assert post_statuses.resolve_sha("o/r", "aa") == ("ok", "a" * 40)
    assert post_statuses.resolve_sha("o/r", "bb") == ("missing", None)
    for sha in ("cc", "dd", "ee", "ff"):
        assert post_statuses.resolve_sha("o/r", sha) == ("error", None), sha


# ------------------------------------------ the terminal path is for known states only


_ENTRY = {
    "slug": "me/unsloth-t4-ci-nabcdef01-1111",
    "sha": "abcdef01",
    "kind": "notebook",
    "legacy": False,
    "age_hours": 0.4,
}


def test_an_unrecognised_kernel_state_is_kept_and_not_judged(tmp_path, monkeypatch):
    """Kaggle's enum has states this collector never sees (NEW_SCRIPT) and can
    grow more. A non-terminal state has no evidence, and judging it posts a
    green `infra` for a run still to come and then deletes the kernel."""
    touched: list[str] = []
    monkeypatch.setattr(
        launch, "fetch_evidence", lambda slug, dest, deadline = None: touched.append(slug) or {}
    )
    monkeypatch.setattr(
        launch, "delete_kernel", lambda slug, deadline = None: touched.append("delete") or True
    )
    api = _StubApi([], {_ENTRY["slug"]: "NEW_SCRIPT"})
    record = collect.collect_one(api, dict(_ENTRY), tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["verdict"] == "pending"
    assert record["verdict"] not in collect.DELETABLE
    assert collect.statuses_from([record]) == []
    assert touched == [], "an unjudged kernel must be neither downloaded nor deleted"


def test_a_downloaded_file_that_is_not_a_notebook_does_not_wedge_the_collector(
    tmp_path, monkeypatch
):
    """`[]` is valid JSON and not a notebook. The report reader's raise is
    caught; the expected-count reader's was not, and a collector that raises
    writes no result and stalls on the same kernel every pass."""
    slug = _ENTRY["slug"]
    dest = tmp_path / slug.rsplit("/", 1)[-1]
    dest.mkdir()
    (dest / f"x{launch.OUTPUT_SUFFIX}").write_text("[]", encoding = "utf-8")
    (dest / f"y{launch.OUTPUT_SUFFIX}").write_text(
        json.dumps({"cells": ["not a cell", {"outputs": ["not an output", {"text": 7}]}]}),
        encoding = "utf-8",
    )
    monkeypatch.setattr(launch, "fetch_evidence", lambda slug, dest, deadline = None: {})
    monkeypatch.setattr(launch, "delete_kernel", lambda slug, deadline = None: True)
    api = _StubApi([], {slug: "COMPLETE"})
    assert collect.expected_reports(dest, 4) == 4
    record = collect.collect_one(api, dict(_ENTRY), tmp_path, expect = 4, max_age_hours = 3.0)
    assert record["verdict"] == "infra", record
    # A notebook kernel with no expected-count record has an UNKNOWN plan.
    assert record["expected"] is None


def test_the_evidence_download_is_clamped_to_the_pass_deadline(tmp_path, monkeypatch):
    """A kernel started just inside the pass budget must not get a fresh
    five-minute download budget, or a fifteen-minute collector runs for twenty
    inside a twenty-five minute job with the release step still to come."""
    seen: list[float] = []

    def _fetch(
        slug,
        dest,
        deadline = None,
    ):
        seen.append(deadline)
        return {}

    monkeypatch.setattr(launch, "fetch_evidence", _fetch)
    monkeypatch.setattr(launch, "extract_reports", lambda dest: [{"passed": True}])
    monkeypatch.setattr(launch, "delete_kernel", lambda slug, deadline = None: True)
    api = _StubApi([], {_ENTRY["slug"]: "COMPLETE"})
    pass_deadline = time.time() + 5.0
    collect.collect_one(
        api, dict(_ENTRY), tmp_path, expect = 1, max_age_hours = 3.0, deadline = pass_deadline
    )
    assert seen and seen[0] <= pass_deadline
    # And the collector's loop hands its deadline down at the one call site.
    source = (CI_DIR / "collect.py").read_text(encoding = "utf-8")
    body = source[source.index("deadline = time.time() + BUDGET_SEC") :]
    assert "deadline = deadline," in body[: body.index("statuses_from")]


# ---------------------------------------- collect BEFORE the recheck, never between


@pytest.mark.parametrize("path", (NOTEBOOK_WF, STUDIO_WF), ids = ("notebook", "studio"))
def test_collection_runs_before_the_recheck_so_the_recheck_is_last_before_the_push(path):
    """Collection can spend up to BUDGET_SEC downloading. Between the recheck
    and the push that gap is the stale window the recheck exists to close: the
    other GPU workflow, on its own group, can take the last session in it."""
    steps = [(n, s) for _j, n, s in _steps(_wf(path))]
    names = [n for n, _s in steps]
    collect_i = names.index("Collect finished Kaggle runs")
    recheck_i = names.index("Recheck the Kaggle account")
    launch_i = names.index("Dispatch to Kaggle")
    assert collect_i < recheck_i < launch_i, names[collect_i : launch_i + 1]
    assert "steps.recheck" not in (steps[collect_i][1].get("if") or "")
    # Nothing that takes BUDGET_SEC sits between the recheck and the push.
    between = [n for n in names[recheck_i + 1 : launch_i]]
    assert not any("collect.py" in (s.get("run") or "") for n, s in steps if n in between), between
    assert "steps.recheck.outputs.should_run == 'true'" in steps[launch_i][1]["if"]


# ------------------------------------- collected Studio evidence is unpacked too


@pytest.mark.parametrize("path", (NOTEBOOK_WF, STUDIO_WF), ids = ("notebook", "studio"))
def test_the_unpack_step_reads_the_tree_a_collected_kernel_lands_in(path):
    """Dispatch mode never writes kaggle_evidence; a kernel collected later
    lands in kaggle_collected/<slug>, so an unpack step reading only the first
    tree never fires on the run that retrieved the screenshots."""
    step = next(s for _j, n, s in _steps(_wf(path)) if n == "Unpack the Playwright evidence")
    assert "kaggle_collected/**/*_output.ipynb" in step["if"]
    assert "kaggle_evidence/**/*_output.ipynb" in step["if"]
    run = step["run"]
    assert "kaggle_collected/*/" in run and "kaggle_evidence/*/" in run
    # One kernel per call: chunks are numbered per bundle, so a walk over two
    # would splice them together.
    assert '--evidence "$dir"' in run and 'studio_evidence/$(basename "$dir")' in run


def test_the_scheduled_collector_unpacks_and_uploads_what_it_collected():
    """It is the pass that collects most Studio kernels, so it is the pass
    that has to put the screenshots back and keep them somewhere."""
    steps = [(n, s) for _j, n, s in _steps(_wf(COLLECT_WF))]
    names = [n for n, _s in steps]
    unpack = steps[names.index("Unpack the Playwright evidence")][1]
    assert "kaggle_collected/*/" in unpack["run"] and "collect_evidence.py" in unpack["run"]
    upload = steps[names.index("Upload evidence")][1]
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert "kaggle_collected/**" in upload["with"]["path"]
    assert "studio_evidence/**" in upload["with"]["path"]
    assert upload.get("continue-on-error") is True, "an artifact outage is not a collection failure"
    assert names.index("Upload evidence") > names.index("Post the commit statuses")


def test_a_listed_notebook_without_a_download_url_marks_the_evidence_incomplete(
    tmp_path, monkeypatch
):
    """Skipping it silently left `truncated` false, so the short set was judged,
    posted and its kernel deleted: a failing payload lost for good."""
    listing = {
        "files": [{"fileName": f"a{launch.OUTPUT_SUFFIX}"}],
        "log": "",
        "truncated": False,
    }
    monkeypatch.setattr(launch, "list_outputs", lambda slug, timeout = None, deadline = None: listing)
    evidence = launch.fetch_evidence("me/k", tmp_path / "k", deadline = time.time() + 60)
    assert evidence["notebooks"] == []
    assert evidence["truncated"] is True


def test_old_and_new_slug_forms_for_one_commit_post_one_status(monkeypatch):
    """A kernel pushed with the eight-character slug and one with the twelve
    can name the same commit in one pass. Only the commits API can say so, so
    the collector keeps them apart and the POSTER merges what resolves to one
    full sha: one post, failure winning, both slugs named."""
    full = "abcdef012345" + "0" * 28
    calls: list[list[str]] = []

    def _gh(args):
        calls.append(list(args))
        if "/commits/" in args[1]:
            return (0, full, "")
        return (0, "", "")

    monkeypatch.setattr(post_statuses, "_gh", _gh)
    records = [
        {
            "slug": "me/unsloth-t4-ci-nabcdef012345-2222",
            "sha": "abcdef012345",
            "kind": "notebook",
            "verdict": "fail",
            "reason": "leg red",
        },
        {
            "slug": "me/unsloth-t4-ci-nabcdef01-1111",
            "sha": "abcdef01",
            "kind": "notebook",
            "verdict": "pass",
            "reason": "ok",
        },
    ]
    statuses = collect.statuses_from(records)
    assert len(statuses) == 2, "the collector cannot know these are one commit"
    for order in (statuses, statuses[::-1]):
        calls.clear()
        outcome = post_statuses.post_all([dict(s) for s in order], "unslothai/unsloth")
        posts = [c for c in calls if "/statuses/" in c[1]]
        assert len(posts) == 1, posts
        assert "state=failure" in " ".join(posts[0])
        assert sorted(outcome["ok"]) == sorted(r["slug"] for r in records)


def test_two_commits_that_merely_share_eight_characters_post_two_statuses():
    """The reason the slug grew to twelve characters. Merged, one commit would
    lose its verdict and have its kernel deleted as delivered."""
    records = [
        {
            "slug": "me/unsloth-t4-ci-nabcdef012345-1111",
            "sha": "abcdef012345",
            "kind": "notebook",
            "verdict": "pass",
            "reason": "ok",
        },
        {
            "slug": "me/unsloth-t4-ci-nabcdef01ffff-2222",
            "sha": "abcdef01ffff",
            "kind": "notebook",
            "verdict": "fail",
            "reason": "leg red",
        },
    ]
    statuses = collect.statuses_from(records)
    assert len(statuses) == 2, statuses
    assert {s["sha"]: s["state"] for s in statuses} == {
        "abcdef012345": "success",
        "abcdef01ffff": "failure",
    }
    assert all(len(s["slugs"]) == 1 for s in statuses)


def test_the_release_phase_stops_at_its_budget_and_keeps_the_rest(tmp_path, monkeypatch):
    """delete_kernel allows three 180-second attempts per kernel. Unbounded,
    the release after a full collection could outlive the job and be killed
    mid-delete; bounded, what is left is released by the next pass."""
    clock = [1000.0]
    monkeypatch.setattr(collect.time, "time", lambda: clock[0])

    def _slow_delete(slug, deadline = None):
        clock[0] += collect.RELEASE_BUDGET_SEC  # one delete eats the whole budget
        return True

    monkeypatch.setattr(launch, "delete_kernel", _slow_delete)
    kernels = [
        {"slug": f"me/unsloth-t4-ci-nabcdef01234{i}-1111", "verdict": "pass"} for i in range(3)
    ]
    result = tmp_path / "collect_result.json"
    result.write_text(json.dumps({"kernels": kernels, "statuses": []}), encoding = "utf-8")
    posted = tmp_path / "posted.json"
    posted.write_text(json.dumps({"ok": [], "failed": []}), encoding = "utf-8")
    assert collect.delete_collected(result, posted) == 0
    outcome = json.loads((tmp_path / "delete_result.json").read_text())
    assert len(outcome["deleted"]) == 1 and len(outcome["kept"]) == 2, outcome


def test_the_scheduled_collector_timeout_covers_collection_and_release():
    job = _wf(COLLECT_WF)["jobs"]["collect"]
    floor = (collect.BUDGET_SEC + collect.RELEASE_BUDGET_SEC) / 60 + 3
    assert job["timeout-minutes"] >= floor, (job["timeout-minutes"], floor)


def test_nothing_tells_the_reader_to_require_a_sampled_context():
    """The gate samples and the workflows path-filter, so a commit the gate
    skips never gets a status; a required context that never arrives blocks
    the merge. The text that said to require it was wrong."""
    for path in (NOTEBOOK_WF, STUDIO_WF, COLLECT_WF, CI_DIR / "launch.py", CI_DIR / "collect.py"):
        assert "branch protection must require" not in path.read_text(encoding = "utf-8"), path


def test_a_delete_is_clamped_to_the_deadline_it_is_handed(monkeypatch):
    """Checked only before the call, the deadline did not bound the release:
    one delete can run three 180-second attempts, so two back to back still
    outlive a 600-second budget. Every attempt is now clamped to what is left
    and none starts past it."""
    clock = [1000.0]
    timeouts: list[float] = []

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "503"

    def _run(
        cmd,
        capture_output = True,
        text = True,
        timeout = None,
    ):
        timeouts.append(timeout)
        clock[0] += timeout  # the call uses its whole allowance
        return _Proc()

    monkeypatch.setattr(launch.subprocess, "run", _run)
    monkeypatch.setattr(launch.time, "time", lambda: clock[0])
    monkeypatch.setattr(launch.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    assert launch.delete_kernel("me/k", deadline = 1000.0 + 200.0) is False
    assert timeouts and timeouts[0] <= 200.0
    assert sum(timeouts) <= 200.0, timeouts
    assert len(timeouts) < launch.DELETE_ATTEMPTS, "an attempt started past the deadline"


def test_the_release_hands_every_delete_its_deadline(tmp_path, monkeypatch):
    seen: list = []
    monkeypatch.setattr(
        launch, "delete_kernel", lambda slug, deadline = None: seen.append(deadline) or True
    )
    result = tmp_path / "collect_result.json"
    result.write_text(
        json.dumps(
            {
                "kernels": [{"slug": "me/unsloth-t4-ci-nabcdef012345-1111", "verdict": "pass"}],
                "statuses": [],
            }
        ),
        encoding = "utf-8",
    )
    posted = tmp_path / "posted.json"
    posted.write_text(json.dumps({"ok": [], "failed": []}), encoding = "utf-8")
    collect.delete_collected(result, posted)
    assert seen and seen[0] is not None and seen[0] <= time.time() + collect.RELEASE_BUDGET_SEC


def test_the_scheduled_collector_uploads_once_and_reports_real_deletions():
    steps = [(n, s) for _j, n, s in _steps(_wf(COLLECT_WF))]
    uploads = [
        s for _n, s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    assert len(uploads) == 1, "two uploads: the second one fails the job on an artifact outage"
    assert uploads[0].get("continue-on-error") is True
    assert "kaggle_collected/**" in uploads[0]["with"]["path"]
    summary = next(s for n, s in steps if n == "Summarise")
    assert (
        "delete_result.json" in summary["run"]
    ), "the summary reports the collect step's always-false deleted flag"


def test_a_kernel_whose_status_record_was_rejected_is_kept(tmp_path, monkeypatch):
    """The poster refuses a malformed record and posts nothing for it. Kept
    only the refused posts, the release deleted that kernel with no status
    delivered and its evidence gone."""
    deleted: list[str] = []
    monkeypatch.setattr(
        launch, "delete_kernel", lambda slug, deadline = None: deleted.append(slug) or True
    )
    slug = "me/unsloth-t4-ci-nabcdef012345-1111"
    result = tmp_path / "collect_result.json"
    result.write_text(
        json.dumps({"kernels": [{"slug": slug, "verdict": "pass"}], "statuses": []}),
        encoding = "utf-8",
    )
    posted = tmp_path / "posted.json"
    posted.write_text(json.dumps({"ok": [], "failed": [], "invalid": [slug]}), encoding = "utf-8")
    collect.delete_collected(result, posted)
    assert deleted == []
    outcome = json.loads((tmp_path / "delete_result.json").read_text())
    assert outcome["kept"] == [slug]


def test_the_pass_budget_starts_before_the_kernel_listing():
    """Five slow listing pages under the socket timeout are minutes. A budget
    started after them is that much later than the job timeout was sized for."""
    source = (CI_DIR / "collect.py").read_text(encoding = "utf-8")
    body = source[source.index("def main(") :]
    assert body.index("deadline = time.time() + BUDGET_SEC") < body.index("ours = find_ours(")


def test_two_commits_that_share_eight_characters_resolve_apart_and_post_twice(monkeypatch):
    """The transition case the collector must not guess at: an old
    eight-character slug for commit A and a twelve-character one for commit B
    that starts with the same eight. Resolved, they are two commits."""
    a_full = "abcdef01" + "a" * 32
    b_full = "abcdef01ffff" + "b" * 28
    calls: list[list[str]] = []

    def _gh(args):
        calls.append(list(args))
        if "/commits/" in args[1]:
            sha = args[1].rsplit("/", 1)[-1]
            return (0, a_full if sha == "abcdef01" else b_full, "")
        return (0, "", "")

    monkeypatch.setattr(post_statuses, "_gh", _gh)
    statuses = collect.statuses_from(
        [
            {"slug": "me/unsloth-t4-ci-nabcdef01-1111", "sha": "abcdef01", "kind": "notebook", "verdict": "pass", "reason": "ok"},
            {"slug": "me/unsloth-t4-ci-nabcdef01ffff-2222", "sha": "abcdef01ffff", "kind": "notebook", "verdict": "fail", "reason": "red"},
        ]
    )
    outcome = post_statuses.post_all(statuses, "unslothai/unsloth")
    posts = [c for c in calls if "/statuses/" in c[1]]
    assert len(posts) == 2
    assert {c[1].rsplit("/", 1)[-1] for c in posts} == {a_full, b_full}
    assert len(outcome["ok"]) == 2


def test_a_notebook_kernel_without_its_expected_count_is_never_a_pass(tmp_path, monkeypatch):
    """A notebook kernel that predates the expected-count record could have
    lost four legs and reported one; judged against a caller default of one
    that read as a pass, was posted green and deleted. Unknown plan: a failure
    is still a failure, and a clean set is infra, not pass."""
    monkeypatch.setattr(launch, "fetch_evidence", lambda slug, dest, deadline = None: {})
    monkeypatch.setattr(launch, "delete_kernel", lambda slug, deadline = None: True)
    api = _StubApi([], {_ENTRY["slug"]: "COMPLETE"})
    monkeypatch.setattr(launch, "extract_reports", lambda dest: [{"passed": True}])
    record = collect.collect_one(api, dict(_ENTRY), tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["verdict"] == "infra" and record["expected"] is None, record
    monkeypatch.setattr(launch, "extract_reports", lambda dest: [{"passed": False, "payload": "x"}])
    record = collect.collect_one(api, dict(_ENTRY), tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["verdict"] == "fail"
    # Studio kernels carry exactly one report and keep the default.
    monkeypatch.setattr(launch, "extract_reports", lambda dest: [{"passed": True}])
    studio = dict(_ENTRY, kind = "studio", slug = "me/unsloth-t4-ci-sabcdef01-1111")
    api = _StubApi([], {studio["slug"]: "COMPLETE"})
    assert collect.collect_one(api, studio, tmp_path, expect = 1, max_age_hours = 3.0)["verdict"] == "pass"


def test_every_gate_budget_covers_the_reaper_window():
    """A dispatching job does not delete its kernel; a kernel that ignores its
    own timeout runs until the scheduled collector reaps it at the age ceiling,
    and that pass can be a whole schedule interval plus its job timeout late.
    The gate admits a run when `remaining >= budget + reserve`, so a budget
    below that window lets one wedged kernel bill into the human reserve."""
    collect_wf = _wf(COLLECT_WF)
    job = collect_wf["jobs"]["collect"]
    on = collect_wf.get("on") or collect_wf.get(True) or {}  # PyYAML reads a bare `on:` as True
    crons = [s["cron"] for s in on.get("schedule") or []]
    assert crons, "the collector is not scheduled"
    minutes = sorted(int(m) for m in crons[0].split()[0].split(","))
    interval_min = max(b - a for a, b in zip(minutes, minutes[1:])) if len(minutes) > 1 else 60
    source = COLLECT_WF.read_text(encoding = "utf-8")
    m = re.search(r"--max-age-hours '\$\{\{ inputs\.max_age_hours \|\| '(\d+)' \}\}'", source)
    assert m, "the collector's default age ceiling is not readable from the workflow"
    max_age = float(m.group(1))
    assert max_age == collect.DEFAULT_MAX_AGE_HOURS
    window = max_age + (interval_min + job["timeout-minutes"]) / 60
    for path in (NOTEBOOK_WF, STUDIO_WF):
        budgets = {
            int(b)
            for b in re.findall(r"--budget-hours (\d+)", path.read_text(encoding = "utf-8"))
        }
        assert len(budgets) == 1, (path.name, budgets)
        assert budgets.pop() >= window, (
            f"{path.name}: a wedged kernel can bill {window:.2f}h before it is reaped"
        )
