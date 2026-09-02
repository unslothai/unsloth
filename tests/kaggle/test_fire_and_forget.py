# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Dispatch and collect: the ways this trades forty minutes for nothing.

The GPU jobs no longer wait for their Kaggle kernel. That buys back 41.5 and
18.7 minutes of GitHub runner per commit (measured on runs 33479481067 and
33486360729), and it introduces exactly one new class of bug, which every test
here is aimed at: **the result stops arriving and nothing says so.**

Four shapes of that, all green:

1. the dispatching job reports success, branch protection still requires IT,
   and the check now means "Kaggle accepted a push";
2. the kernel is never collected, so it bills quota to its own ceiling and the
   commit keeps a status that never resolves;
3. the collector deletes a kernel before reading it, or deletes one that was
   never ours;
4. the slug loses the commit, so a finished result cannot be attributed to
   anything and is silently dropped.

Everything runs on CPU against stubs; no Kaggle session is spent.
"""

from __future__ import annotations

import json
import re
import sys
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

    The slug is the ONLY thing that survives the dispatching runner: no
    artifact, no branch, no database. So it has to say which commit the result
    is about and which workflow asked, and it has to be readable back.
    """
    name = launch.slug_name("notebook", "1a2b3c4d5e6f7890")
    parsed = launch.parse_slug(name)
    assert parsed is not None, f"{name} does not parse as one of ours"
    assert parsed["sha"] == "1a2b3c4d", parsed
    assert parsed["kind"] == "notebook", parsed


def test_two_dispatches_of_one_commit_do_not_collide():
    """A re-run, or notebook slot 1 and 2, must not push to the same slug.

    Pushing to an id that already exists does not replace it, it files a new
    version and starts a SECOND session while the first keeps running -- and
    status/output then answer for the newest only, so one session burns a slot
    and its quota entirely unseen. That is in push()'s own docstring; the
    commit-derived slug must not reintroduce it.
    """
    names = {launch.slug_name("notebook", "deadbeefcafe") for _ in range(200)}
    assert len(names) > 190, f"only {len(names)} distinct slugs in 200 draws"


def test_the_slug_still_round_trips_through_slugify():
    """Kaggle files the kernel under the slugified TITLE, not the metadata id.

    A name that does not survive that round trip files the kernel at an
    unexpected address, where every later status and output call 403s -- and
    now, worse, where the commit in the name is no longer the commit of the
    kernel anyone can find.
    """
    for kind in ("notebook", "studio"):
        name = launch.slug_name(kind, "0123456789ab")
        assert launch._slugify(name.replace("-", " ")) == name, name


def test_the_gate_still_recognises_a_dispatched_kernel_as_ours():
    """THE ONE THAT KEEPS THE CONCURRENCY CONTROL WORKING.

    With fire-and-forget the GitHub concurrency group stops bounding Kaggle
    sessions -- the dispatching job exits immediately, so its group is released
    while the kernel runs on. What still bounds them is the gate's own survey,
    which counts in-flight kernels whose slug carries OWN_KERNEL_PREFIX.

    Change the slug format without keeping that prefix and the gate stops
    counting our own sessions: three pushes in ten minutes would each believe
    the account idle, and Kaggle would refuse the third at its 2-session cap
    after the quota for the first two was already committed.
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

    That account is shared with a human and holds every probe anyone has ever
    pushed. There is deliberately no heuristic here -- no prefix-ish match, no
    "looks like CI" -- because the cost of a false positive is deleting
    somebody's work, and it would look exactly like a kernel that finished.
    """
    assert launch.parse_slug(foreign) is None, f"{foreign!r} was accepted as one of ours"


def test_a_legacy_slug_is_still_reapable_but_reports_nothing():
    """Kernels pushed before this change can outlive it.

    They are ours and must still be reaped, or they bill forever. They carry no
    commit, so there is nothing to attribute a verdict to, and inventing one is
    worse than silence.
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
        sys, "argv",
        ["launch.py", "--notebook", "k.ipynb", "--outdir", "out", "--dispatch"],
    )
    with pytest.raises(SystemExit) as excinfo:
        launch.main()
    assert excinfo.value.code != 0
    assert "commit-sha" in capsys.readouterr().err


def test_dispatch_does_not_delete_the_kernel_it_pushed():
    """The whole point: the kernel is LEFT RUNNING.

    Expressed through the existing --keep-kernel flag rather than a second
    condition in release(), because two independent guards on one deletion is
    how one of them ends up wrong -- and this one bills GPU quota when it is.
    """
    src = (CI_DIR / "launch.py").read_text(encoding = "utf-8")
    assert re.search(r"if args\.dispatch:\s*\n\s*args\.keep_kernel = True", src), (
        "dispatch mode must set keep_kernel, or release() deletes the kernel it "
        "was told to leave running and the collector finds nothing"
    )


def test_dispatch_does_not_report_pass():
    """`dispatched` is not `pass`, and the difference is the whole change.

    Nothing has run when the dispatch returns. Reporting `pass` would be worse
    than the silent skip this repo has already been caught by twice, because it
    would look like a result rather than like an absence.
    """
    src = (CI_DIR / "launch.py").read_text(encoding = "utf-8")
    block = src[src.index("if args.dispatch:", src.index("result[\"slug\"] = live[0]")):]
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
    def __init__(self, ref, last_run_time = None):
        self.ref = ref
        self.last_run_time = last_run_time


class _StubApi:
    CONFIG_NAME_USER = "username"

    def __init__(self, kernels, statuses):
        self.config_values = {"username": "danielhanchen"}
        self._kernels = kernels
        self._statuses = statuses
        self.status_calls: list[str] = []

    def kernels_list(self, mine = True, page = 1, page_size = 100, sort_by = "dateRun"):
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
    entry = {"slug": "me/unsloth-t4-ci-nabcdef01-1111", "sha": "abcdef01",
             "kind": "notebook", "legacy": False, "age_hours": 0.5}
    record = collect.collect_one(api, entry, tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["verdict"] == "pending"
    assert record["deleted"] is False
    assert collect.statuses_from([record]) == [], "a running kernel must post no status"
    assert deleted == []


def test_a_kernel_past_its_ceiling_is_reaped_and_reported(tmp_path, monkeypatch):
    """THE REASON THE SCHEDULED COLLECTOR EXISTS.

    The old design deleted every kernel it pushed on every path out. Dispatch
    mode removes that, so a wedged kernel bills to its own ceiling with nobody
    watching -- this directory has already measured one ignoring Kaggle's own
    `-t` timeout for over two hours.

    It is reported as a failure rather than dropped: the run it belonged to will
    never produce a result, and saying nothing leaves the commit pending forever.
    """
    deleted: list[str] = []
    monkeypatch.setattr(launch, "delete_kernel", lambda slug: deleted.append(slug) or True)
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": "RUNNING"})
    entry = {"slug": "me/unsloth-t4-ci-nabcdef01-1111", "sha": "abcdef01",
             "kind": "notebook", "legacy": False, "age_hours": 9.0}
    record = collect.collect_one(api, entry, tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["verdict"] == "reaped"
    assert deleted == ["me/unsloth-t4-ci-nabcdef01-1111"]
    status = collect.statuses_from([record])[0]
    assert status["state"] == "failure", status


def test_a_kernel_with_no_timestamp_is_never_reaped(tmp_path, monkeypatch):
    """A missing timestamp is not evidence that a kernel is old, and guessing in
    that direction DELETES A RUNNING SESSION."""
    deleted: list[str] = []
    monkeypatch.setattr(launch, "delete_kernel", lambda slug: deleted.append(slug) or True)
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": "RUNNING"})
    entry = {"slug": "me/unsloth-t4-ci-nabcdef01-1111", "sha": "abcdef01",
             "kind": "notebook", "legacy": False, "age_hours": None}
    record = collect.collect_one(api, entry, tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["verdict"] == "pending"
    assert deleted == []


def test_evidence_is_downloaded_before_the_kernel_is_deleted(tmp_path, monkeypatch):
    """A delete that lands first turns a finished run into a result nobody can
    ever read. Asserted by ORDER of the real calls, not by reading the source."""
    order: list[str] = []
    monkeypatch.setattr(launch, "fetch_evidence",
                        lambda slug, dest, deadline = None: order.append("fetch") or {})
    monkeypatch.setattr(launch, "extract_reports", lambda dest: [{"passed": True}])
    monkeypatch.setattr(launch, "delete_kernel", lambda slug: order.append("delete") or True)
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": "COMPLETE"})
    entry = {"slug": "me/unsloth-t4-ci-nabcdef01-1111", "sha": "abcdef01",
             "kind": "notebook", "legacy": False, "age_hours": 0.4}
    collect.collect_one(api, entry, tmp_path, expect = 1, max_age_hours = 3.0)
    assert order == ["fetch", "delete"], order


def test_a_kernel_whose_evidence_will_not_download_is_NOT_deleted(tmp_path, monkeypatch):
    """Deleting here destroys the only copy of a finished run's result, to
    reclaim a session slot the kernel is no longer using. The next pass retries."""
    def _boom(slug, dest, deadline = None):
        raise TimeoutError("slow")

    deleted: list[str] = []
    monkeypatch.setattr(launch, "fetch_evidence", _boom)
    monkeypatch.setattr(launch, "delete_kernel", lambda slug: deleted.append(slug) or True)
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": "COMPLETE"})
    entry = {"slug": "me/unsloth-t4-ci-nabcdef01-1111", "sha": "abcdef01",
             "kind": "notebook", "legacy": False, "age_hours": 0.4}
    record = collect.collect_one(api, entry, tmp_path, expect = 1, max_age_hours = 3.0)
    assert record["verdict"] == "infra"
    assert deleted == [], "evidence that would not download must not be deleted"


def test_an_unreadable_status_does_nothing_at_all(tmp_path, monkeypatch):
    """Both available actions are destructive: delete and we may kill a running
    session, report and we may fail a run that was fine. Ask again next pass."""
    deleted: list[str] = []
    monkeypatch.setattr(launch, "delete_kernel", lambda slug: deleted.append(slug) or True)
    api = _StubApi([], {"me/unsloth-t4-ci-nabcdef01-1111": RuntimeError("503 upstream")})
    entry = {"slug": "me/unsloth-t4-ci-nabcdef01-1111", "sha": "abcdef01",
             "kind": "notebook", "legacy": False, "age_hours": 0.4}
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
    branch protection. `infra` and `partial` mean nothing was learned about the
    code, so they stay green and say so in the description."""
    assert collect.VERDICT_STATE["infra"] == "success"
    assert collect.VERDICT_STATE["partial"] == "success"
    assert collect.VERDICT_STATE["fail"] == "failure"
    assert collect.VERDICT_STATE["reaped"] == "failure"


def test_a_failed_payload_reaches_github_as_a_failure():
    """Driven end to end through the real verdict and status functions, since
    every rule above this one is fed a dict written by hand."""
    record = {"slug": "me/unsloth-t4-ci-nabcdef01-1111", "sha": "abcdef01",
              "kind": "notebook", "verdict": None, "reason": ""}
    verdict, reason = collect.verdict_of([{"passed": False, "payload": "gptoss"}], 1)
    record["verdict"], record["reason"] = verdict, reason
    status = collect.statuses_from([record])[0]
    assert status["state"] == "failure"
    assert status["context"] == "kaggle-t4-notebook"
    assert "gptoss" in status["description"]


def test_the_collector_never_sees_a_github_token():
    """A script holding a Kaggle credential and a GitHub credential is one bug
    away from sending one to the other. It emits statuses as DATA; the workflow
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
    """Order matters twice over: it frees the Kaggle session slot this job is
    about to want, and it is what answers whether this commit is already
    running so a re-run does not pay for a second session."""
    body = path.read_text(encoding = "utf-8")
    assert "kaggle_t4_ci/collect.py" in body, f"{path.name} never collects"
    assert body.index("kaggle_t4_ci/collect.py") < body.index("--dispatch"), (
        f"{path.name} dispatches before it collects"
    )


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
    """Route 1 (the next run collects) only fires when somebody pushes. A quiet
    weekend with a wedged kernel is a weekend of billing nobody is watching, and
    the old design could not have that bug because the pushing job also deleted."""
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
    """Two collectors on one account race: both enumerate the same kernels, and
    the loser tries to delete what the winner already deleted and reports the
    404 as a failure."""
    wf = _wf(COLLECT_WF)
    conc = wf.get("concurrency")
    assert conc, "the collector has no concurrency group"
    assert conc.get("cancel-in-progress") is False, (
        "a pass midway through downloading evidence must finish; cancelling it "
        "throws away work the next pass has to redo"
    )


def test_the_status_contexts_are_stable_strings():
    """Branch protection is configured against these names. Renaming one
    silently stops requiring the check it names -- the rule keeps passing
    because a context that never reports is not a failing context."""
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
    request with no explanation on the page -- indistinguishable from a check
    that was never configured."""
    for path, context in ((NOTEBOOK_WF, "kaggle-t4-notebook"), (STUDIO_WF, "kaggle-studio-gpu")):
        body = path.read_text(encoding = "utf-8")
        assert re.search(r"state=pending -f context=" + re.escape(context), body), (
            f"{path.name} never posts a pending {context} status"
        )


def test_the_reporters_do_not_run_over_an_empty_evidence_directory():
    """A reporter over a directory that does not exist yet prints "0 of 5
    payloads", which reads as a failure rather than as a result that has not
    arrived. This is the exact shape of the silent-red this change could
    introduce."""
    for path in (NOTEBOOK_WF, STUDIO_WF):
        for _job, name, step in _steps(_wf(path)):
            body = step.get("run") or ""
            if "report.py" not in body:
                continue
            assert "hashFiles('kaggle_evidence/**')" in (step.get("if") or ""), (
                f"{path.name}: step {name!r} reports over evidence that a dispatching "
                f"run never produces: if={step.get('if')!r}"
            )
