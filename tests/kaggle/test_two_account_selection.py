# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Two Kaggle accounts: the failure is a green run that spent the wrong account or nothing at all."""

from __future__ import annotations

import collections
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

sys.path.insert(0, str(CI_DIR))

import gate  # noqa: E402
import launch  # noqa: E402


def _wf(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding = "utf-8"))


def _steps(workflow: dict) -> list[tuple[str, str, dict]]:
    out = []
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", []) or []:
            out.append((job_name, step.get("name", ""), step))
    return out


@pytest.mark.parametrize("path", (NOTEBOOK_WF, STUDIO_WF), ids = ("notebook", "studio"))
def test_no_workflow_names_a_secret_that_does_not_exist(path):
    """A workflow may name only the closed set of Kaggle secrets; a deleted one made both no-op."""
    referenced = set(re.findall(r"secrets\.([A-Z0-9_]+)", path.read_text(encoding = "utf-8")))
    kaggle = {s for s in referenced if "KAGGLE" in s}
    assert kaggle == set(gate.DEFAULT_ACCOUNT_ENVS), (
        f"{path.name} names Kaggle secrets {sorted(kaggle)}, but the accounts that "
        f"exist are {sorted(gate.DEFAULT_ACCOUNT_ENVS)}. A name that is not a real "
        "secret resolves to an empty string and the gate skips green."
    )


@pytest.mark.parametrize("path", (NOTEBOOK_WF, STUDIO_WF), ids = ("notebook", "studio"))
def test_every_step_that_runs_a_kaggle_script_is_given_a_token(path):
    """Derived from the workflow, not from a list, so a step added later cannot
    quietly run credential-less and report the skip as a normal outcome."""
    missing = []
    for job_name, step_name, step in _steps(_wf(path)):
        body = step.get("run") or ""
        if "kaggle_t4_ci/gate.py" not in body and "kaggle_t4_ci/launch.py" not in body:
            continue
        env = step.get("env") or {}
        if "KAGGLE_API_TOKEN" not in env:
            missing.append(f"{job_name}/{step_name}")
    assert missing == [], f"these steps run a Kaggle script with no token: {missing}"


@pytest.mark.parametrize("path", (NOTEBOOK_WF, STUDIO_WF), ids = ("notebook", "studio"))
def test_only_the_gate_sees_both_accounts(path):
    """Only the gate step may see both account tokens; a later step holding both could act as either."""
    for job_name, step_name, step in _steps(_wf(path)):
        env = step.get("env") or {}
        tokens = sorted(k for k in env if k.startswith("KAGGLE_API_TOKEN"))
        if not tokens:
            continue
        if step.get("id") == "decide":
            assert tokens == sorted(
                gate.DEFAULT_ACCOUNT_ENVS
            ), f"the gate must weigh every account, got {tokens}"
        else:
            assert tokens == [
                "KAGGLE_API_TOKEN"
            ], f"{job_name}/{step_name} sees {tokens}; only the gate may see more than one"


@pytest.mark.parametrize("path", (NOTEBOOK_WF, STUDIO_WF), ids = ("notebook", "studio"))
def test_the_chosen_token_is_INDEXED_and_never_a_ternary(path):
    """Index secrets by name, never a && || ternary: an empty first secret hands over the other token."""
    for job_name, step_name, step in _steps(_wf(path)):
        env = step.get("env") or {}
        expr = env.get("KAGGLE_API_TOKEN", "")
        if step.get("id") == "decide":
            continue
        if not expr:
            continue
        assert "&&" not in expr and "||" not in expr, (
            f"{job_name}/{step_name} selects its token with a ternary ({expr}), which "
            "silently falls through to the other account when the first secret is empty"
        )
        assert (
            "secrets[matrix." in expr
        ), f"{job_name}/{step_name} does not index the secrets context: {expr}"


@pytest.mark.parametrize("path", (NOTEBOOK_WF, STUDIO_WF), ids = ("notebook", "studio"))
def test_no_token_is_ever_a_job_output(path):
    """The matrix carries a secret NAME. A token in an output is a credential in
    a place GitHub redacts by pattern rather than by promise."""
    for job_name, job in _wf(path)["jobs"].items():
        for key, value in (job.get("outputs") or {}).items():
            assert "secrets." not in str(value) and "secrets[" not in str(
                value
            ), f"{job_name}.outputs.{key} publishes a secret: {value}"


@pytest.mark.parametrize(
    "path,suffix",
    ((NOTEBOOK_WF, "notebook"), (STUDIO_WF, "studio")),
    ids = ("notebook", "studio"),
)
def test_the_concurrency_group_is_keyed_on_the_account(path, suffix):
    """Kaggle's 2-session cap is per account, so the concurrency group must be keyed per account."""
    groups = [
        (job.get("concurrency") or {}).get("group")
        for job in _wf(path)["jobs"].values()
        if isinstance(job.get("concurrency"), dict)
    ]
    account_groups = [g for g in groups if g and suffix in g]
    assert account_groups, f"no per-account {suffix} group found in {path.name}"
    for group in account_groups:
        assert (
            "needs.gate.outputs.account" in group
        ), f"{group!r} does not vary by account, so two accounts share one lock"


@pytest.mark.parametrize("path", (NOTEBOOK_WF, STUDIO_WF), ids = ("notebook", "studio"))
def test_the_gpu_job_takes_the_account_through_a_one_element_matrix(path):
    for job_name, job in _wf(path)["jobs"].items():
        if job_name == "gate":
            continue
        matrix = (job.get("strategy") or {}).get("matrix")
        assert matrix and "needs.gate.outputs.matrix" in str(
            matrix
        ), f"{job_name} does not receive the gate's account matrix: {matrix!r}"


@pytest.mark.parametrize("path", (NOTEBOOK_WF, STUDIO_WF), ids = ("notebook", "studio"))
def test_no_kaggle_username_is_hardcoded_on_the_launch_path(path):
    """A literal kernel owner belongs to one account, so the other cannot push or delete under it."""
    for job_name, step_name, step in _steps(_wf(path)):
        body = step.get("run") or ""
        if "launch.py" not in body:
            continue
        # To end of line: the value is quoted, so `\S+` would capture only `'${{`.
        for match in re.findall(r"--user\s+(.+)", body):
            assert (
                "matrix.kaggle_user" in match
            ), f"{job_name}/{step_name} pushes under a hardcoded owner {match!r}"


@pytest.mark.parametrize("path", (NOTEBOOK_WF, STUDIO_WF), ids = ("notebook", "studio"))
def test_the_recheck_can_actually_stop_the_push(path):
    """Both workflows recheck with the account slot held, and the push is gated on that answer."""
    steps = _steps(_wf(path))
    recheck = [s for _, _, s in steps if s.get("id") == "recheck"]
    assert recheck, f"{path.name} never re-asks the gate with the slot in hand"
    assert "--account-env KAGGLE_API_TOKEN" in (recheck[0].get("run") or ""), (
        "the recheck must be narrowed to the ONE account the gate chose, or it can "
        "clear an account this job is not holding a slot for"
    )
    launched = [s for _, _, s in steps if "launch.py" in (s.get("run") or "")]
    assert launched, f"{path.name} never launches"
    for step in launched:
        assert "steps.recheck.outputs.should_run == 'true'" in (
            step.get("if") or ""
        ), "the push does not depend on the recheck, so the recheck decides nothing"


def test_the_split_follows_the_weekly_hours():
    """60 and 30 must come out 2:1, and the weights are the accounts' own
    totals rather than a number written here."""
    counts = collections.Counter(
        gate.weighted_pick(str(i), {"1": 60.0, "2": 30.0})[0] for i in range(60000)
    )
    share = counts["1"] / sum(counts.values())
    assert 0.65 < share < 0.685, f"account 1 took {share:.4f} of the traffic, wanted ~0.667"


def test_a_rerun_returns_to_the_same_account(monkeypatch):
    """Account choice keys on the run id alone, so a rerun returns to the account holding its kernels."""
    weights = {"1": 60.0, "2": 30.0}
    for run_id in ("1", "17", "912837", "40000000001"):
        picks = set()
        for attempt in ("1", "2", "3", "17"):
            monkeypatch.setenv("GITHUB_RUN_ATTEMPT", attempt)
            picks.add(gate.weighted_pick(run_id, weights)[0])
        assert len(picks) == 1, (
            f"run {run_id} lands on {picks} across attempts; a re-run would push to an "
            "account that is not holding the previous attempt's kernels"
        )


def test_the_account_draw_is_salted_apart_from_the_sampling_draw():
    """Salt asserted in source: a statistical test of the salt was tried and could not fail on it."""
    source = (CI_DIR / "gate.py").read_text(encoding = "utf-8")
    picked = source.split("def weighted_pick", 1)[1].split("\ndef ", 1)[0]
    assert (
        'sha256(("account:" + key)' in picked
    ), "the account draw hashes the bare key, which is what sampled_in hashes"


def test_every_run_of_one_commit_lands_on_the_same_account():
    """A label or a forced dispatch starts a second run of the SAME commit with
    a new run id. Keyed on the run id the draw could pick the other account,
    which cannot see the first's kernel, and dispatch a duplicate."""
    weights = {"1": 60.0, "2": 30.0}
    sha = "0123456789abcdef0123456789abcdef01234567"
    assert len({gate.weighted_pick(sha, weights)[0] for _ in range(5)}) == 1
    source = (CI_DIR / "gate.py").read_text(encoding = "utf-8")
    assert (
        "account_key = (args.head_sha or" in source
    ), "the gate does not key the draw on the commit"
    for path in (NOTEBOOK_WF, STUDIO_WF):
        gate_steps = [
            s
            for _j, _n, s in _steps(_wf(path))
            if "gate.py" in (s.get("run") or "") and "--percent" in s["run"]
        ]
        assert gate_steps, f"{path.name} has no gate step"
        for step in gate_steps:
            assert "--head-sha" in step["run"], f"{path.name}: the gate is not told the commit"


def test_an_account_with_no_readable_quota_gets_no_weight_but_keeps_its_turn():
    """Unknown is not zero-sized and it is not disqualified either: the account
    loses its share of the traffic, because a share is what its plan says and we
    did not get to hear it, and keeps its place as a fallback."""
    chosen, _ = gate.weighted_pick("5", {"1": 60.0})
    assert chosen == "1"
    chosen, _ = gate.weighted_pick("5", {})
    assert chosen == ""


def test_the_reserve_is_a_fraction_of_the_plan_not_a_flat_number():
    """20h held out of a 30h account is two thirds of it against one third of a
    60h one, which silently makes the SMALLER account the stricter one -- on top
    of it already taking less traffic by weight."""
    assert gate.scaled_reserve(20.0, 60.0, 60.0) == 20.0
    assert gate.scaled_reserve(20.0, 30.0, 60.0) == 10.0
    assert gate.scaled_reserve(10.0, 30.0, 60.0) == 5.0


def test_an_unknown_plan_size_leaves_the_reserve_alone():
    """Scaling by a total nobody could read would invent a number. The flat
    value is the conservative answer and is what is kept."""
    assert gate.scaled_reserve(20.0, 0.0, 60.0) == 20.0
    assert gate.scaled_reserve(20.0, 30.0, 0.0) == 20.0


def test_a_sweep_leaves_the_other_account_s_kernels_filed(tmp_path, monkeypatch):
    """This token cannot delete that kernel. Trying turns a real leak into a
    log line that reads exactly like a kernel already gone -- and DROPS the
    record, which is the only thing that still knows the kernel exists."""
    registry = tmp_path / "inflight.json"
    registry.write_text(
        json.dumps(
            [
                {"slug": "alice/unsloth-t4-ci-aaaa", "pid": 999999, "at": 0, "owner": "alice"},
                {"slug": "bob/unsloth-t4-ci-bbbb", "pid": 999999, "at": 0, "owner": "bob"},
            ]
        ),
        encoding = "utf-8",
    )
    monkeypatch.setattr(launch, "INFLIGHT", registry)
    monkeypatch.setattr(launch, "_pid_alive", lambda pid: False)

    attempted: list[str] = []

    def _delete(slug, *_a, **_k):
        attempted.append(slug)
        return True

    monkeypatch.setattr(launch, "delete_kernel", _delete, raising = False)
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *a, **k: _delete(a[0][3]) and types_simple(),
        raising = False,
    )

    launch.sweep_orphans("alice")

    assert all(
        "bob/" not in slug for slug in attempted
    ), f"the sweep tried to delete another account's kernel: {attempted}"
    left = {e["slug"] for e in json.loads(registry.read_text(encoding = "utf-8"))}
    assert (
        "bob/unsloth-t4-ci-bbbb" in left
    ), "the other account's kernel was dropped from the registry, so nothing knows it exists"


class types_simple:  # noqa: N801 - a stand-in CompletedProcess
    returncode = 0
    stdout = ""
    stderr = ""


def _run_launcher(monkeypatch, tmp_path, username, *, user_arg):
    """`launch.main()` with Kaggle stubbed, up to the point it decides to push."""
    outdir = tmp_path / "out"
    pushed: list = []

    class _Api:
        CONFIG_NAME_USER = "username"

        def __init__(self):
            self.config_values = {"username": username} if username else {}

    monkeypatch.setattr(launch, "_api", lambda *a, **k: _Api())
    monkeypatch.setattr(launch, "sweep_orphans", lambda *a, **k: [])
    monkeypatch.setattr(
        launch,
        "push",
        lambda *a, **k: pushed.append(a) or {"ok": False, "reason": "stub", "attempts": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch.py",
            "--notebook",
            str(tmp_path / "k.ipynb"),
            "--user",
            user_arg,
            "--outdir",
            str(outdir),
        ],
    )
    code = launch.main()
    result = json.loads((outdir / "launch_result.json").read_text(encoding = "utf-8"))
    return code, result, pushed


def test_the_launcher_refuses_a_username_the_token_does_not_own(tmp_path, monkeypatch):
    """Launcher must refuse a username its token does not own, since cleanup could not delete the
    kernels."""
    code, result, pushed = _run_launcher(monkeypatch, tmp_path, "alice", user_arg = "bob")
    assert pushed == [], "a kernel was pushed under a name the token does not own"
    assert result["verdict"] == "infra"
    assert "alice" in result["reason"] and "bob" in result["reason"], result["reason"]


def test_the_launcher_refuses_a_token_that_cannot_name_its_account(tmp_path, monkeypatch):
    """No owner, no push. The owner is not optional: it is half the kernel id."""
    code, result, pushed = _run_launcher(monkeypatch, tmp_path, None, user_arg = "bob")
    assert pushed == []
    assert "could not determine which Kaggle account" in result["reason"]


def test_the_stand_down_still_writes_the_result_the_report_step_reads(tmp_path, monkeypatch):
    """A bare `return 1` here would leave no launch_result.json, and the report
    step reads that file: a configuration error would arrive looking exactly
    like a runner that died mid-run."""
    _, result, _ = _run_launcher(monkeypatch, tmp_path, "alice", user_arg = "bob")
    assert result["verdict"] == "infra" and result["reason"]


def test_a_filed_slug_records_the_account_that_owns_it(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "INFLIGHT", tmp_path / "inflight.json")
    launch._inflight_add("carol/unsloth-t4-ci-cccc")
    entry = json.loads((tmp_path / "inflight.json").read_text(encoding = "utf-8"))[0]
    assert entry["owner"] == "carol", entry


def test_neither_token_name_can_reach_the_kernel():
    """The built notebook is what Kaggle receives. Both account env vars belong
    in the forbidden list, not just the first one."""
    source = (REPO_ROOT / "tests" / "kaggle" / "test_t4_smoke_harness.py").read_text(
        encoding = "utf-8"
    )
    for name in gate.DEFAULT_ACCOUNT_ENVS:
        assert (
            f'"{name}"' in source
        ), f"{name} is not in the credential-leak guard, so a kernel could carry it"


def test_a_kernel_already_running_this_commit_on_any_account_stands_the_run_down():
    """A handover lands a retry on the other account, whose GPU job collects
    with only its own token and sees nothing in flight. So the gate asks every
    account it surveys before it picks one."""
    sha = "abcdef0123456789" + "0" * 24
    own = [
        "danielhanchen/unsloth-t4-ci-sabcdef012345-1111 (RUNNING)",  # studio, this commit
        "danielhanchen/unsloth-t4-ci-nffffffffffff-2222 (RUNNING)",  # notebook, other commit
        "danielhanchen/unsloth-t4-ci-nabcdef01-3333 (QUEUED)",  # notebook, old slug form
    ]
    assert gate.in_flight_for_commit(own, sha, "notebook") == (
        "danielhanchen/unsloth-t4-ci-nabcdef01-3333"
    )
    assert gate.in_flight_for_commit(own, sha, "studio") == (
        "danielhanchen/unsloth-t4-ci-sabcdef012345-1111"
    )
    assert gate.in_flight_for_commit(own, "1234567890ab", "notebook") is None
    assert gate.in_flight_for_commit(own, sha, "") is None
    assert gate.in_flight_for_commit(["someone/unsloth-probe-x (RUNNING)"], sha, "notebook") is None
    source = (CI_DIR / "gate.py").read_text(encoding = "utf-8")
    main = source[source.index("def main(") :]
    # Every candidate is surveyed before any is chosen.
    sweep, selection = main.split("for account_id in order:", 2)[1:]
    assert "in_flight_for_commit(survey" in sweep and "concurrency_verdict(" not in sweep
    assert "concurrency_verdict(" in selection and "in_flight_for_commit(" not in selection
    assert (
        "_survey(account_id)" in sweep and "_survey(account_id)" in selection
    ), "surveys are not shared between the sweep and the selection"
    for path, kind in ((NOTEBOOK_WF, "notebook"), (STUDIO_WF, "studio")):
        gate_steps = [
            s
            for _j, _n, s in _steps(_wf(path))
            if "gate.py" in (s.get("run") or "") and "--percent" in s["run"]
        ]
        assert gate_steps, path.name
        for step in gate_steps:
            assert f"--kind {kind}" in step["run"], f"{path.name}: the gate is not told its kind"


def _drive_gate(
    monkeypatch,
    tmp_path,
    *,
    holder,
    outcomes = None,
    extra = (),
    clock = None,
    holder_slot = "1",
):
    """Run gate.main() with two stub accounts. `holder` is the account whose
    survey shows a notebook kernel of the commit under test; `outcomes` maps an
    account to a probe outcome other than ok, the client still handed back."""
    sha = "abcdef0123456789" + "0" * 24
    outcomes = outcomes or {}
    apis = {"1": object(), "2": object()}

    def probe(account_id, env_name, **_kw):
        quota = {"total_hours": 60.0 if account_id == "1" else 30.0, "remaining_hours": 50.0}
        record = {
            "account": account_id,
            "env": env_name,
            "outcome": outcomes.get(account_id, "ok"),
            "user": f"user{account_id}",
            "quota": quota,
            "total_hours": quota["total_hours"],
            "remaining_hours": 50.0,
            "reserve_hours": 1.0,
        }
        return record, apis[account_id]

    surveys_asked: list[tuple[str, float | None]] = []

    def survey(api, *a, **k):
        account_id = next(i for i, obj in apis.items() if obj is api)
        surveys_asked.append((account_id, k.get("budget_sec")))
        if clock is not None:
            clock[0] += 170.0  # a slow survey
        mark = "" if holder_slot == "1" else holder_slot
        own = (
            [f"user{holder}/unsloth-t4-ci-n{sha[:12]}-{mark}1111 (RUNNING)"]
            if account_id == holder
            else []
        )
        return {
            "busy": list(own),
            "own": own,
            "foreign": [],
            "complete": True,
            "out_of_budget": False,
            "surveyed": len(own),
            "unreadable": 0,
            "gone": 0,
            "window_hours": 12,
        }

    monkeypatch.setattr(gate, "probe_account", probe)
    monkeypatch.setattr(gate, "survey_kernels", survey)
    if clock is not None:
        monkeypatch.setattr(gate.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("KAGGLE_API_TOKEN", "x")
    monkeypatch.setenv("KAGGLE_API_TOKEN_2", "y")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out.txt"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gate.py",
            "--budget-hours",
            "1",
            "--reserve-hours",
            "1",
            "--force",
            "true",
            "--head-sha",
            sha,
            "--kind",
            "notebook",
            "--run-id",
            "7",
            *extra,
        ],
    )
    code = gate.main()
    outputs = dict(
        line.split("=", 1)
        for line in (tmp_path / "out.txt").read_text().splitlines()
        if "=" in line
    )
    return code, outputs, surveys_asked, sha


def _other(sha: str) -> tuple[str, str]:
    sampled = gate.weighted_pick(sha, {"1": 60.0, "2": 30.0})[0]
    return sampled, ("2" if sampled == "1" else "1")


def test_the_gate_stands_down_when_the_other_account_already_runs_this_commit(
    monkeypatch, tmp_path
):
    """The sampled account is FREE and the other holds a kernel for this commit,
    which is what a retry after a handover looks like. A gate that only asks the
    account it is about to pick finds it clear and dispatches a duplicate."""
    _sampled, other = _other("abcdef0123456789" + "0" * 24)
    code, outputs, asked, _sha = _drive_gate(monkeypatch, tmp_path, holder = other)
    assert code == 0
    assert outputs["should_run"] == "false", outputs["reason"]
    assert f"account {other}" in outputs["reason"] and "already running" in outputs["reason"]
    assert {a for a, _b in asked} == {"1", "2"}, "the account holding the kernel was never asked"
    assert len(asked) == 2, asked


def test_an_account_that_cannot_launch_is_still_asked_whether_it_runs_this_commit(
    monkeypatch, tmp_path
):
    """The account that dispatched this commit is the one likely to be short of
    quota now. Skipping it in the sweep hands the retry to the other account,
    which dispatches the same commit again."""
    _sampled, other = _other("abcdef0123456789" + "0" * 24)
    for outcome in ("insufficient_quota", "quota_unreadable"):
        code, outputs, asked, _sha = _drive_gate(
            monkeypatch, tmp_path, holder = other, outcomes = {other: outcome}
        )
        assert outputs["should_run"] == "false", (outcome, outputs["reason"])
        assert "already running" in outputs["reason"], (outcome, outputs["reason"])
        assert other in {a for a, _b in asked}, (outcome, asked)


def test_the_second_slot_is_not_a_duplicate_but_its_own_retry_is(monkeypatch, tmp_path):
    """`slot: 2` is a second session on the same ref beside slot 1, so a slot-1
    kernel must not stand it down, while a retry of the slot-2 run itself must.
    The slot in the slug tells the two apart."""
    sampled, _unused = _other("abcdef0123456789" + "0" * 24)
    # Slot 1 kernel up, slot 2 asked for: runs.
    _c, outputs, _a, _s = _drive_gate(
        monkeypatch, tmp_path, holder = sampled, extra = ("--slot", "2"), holder_slot = "1"
    )
    assert outputs["should_run"] == "true", outputs["reason"]
    # Slot 2 kernel up, slot 2 asked for again: stands down.
    _c, outputs, _a, _s = _drive_gate(
        monkeypatch, tmp_path, holder = sampled, extra = ("--slot", "2"), holder_slot = "2"
    )
    assert outputs["should_run"] == "false", outputs["reason"]
    assert "slot 2" in outputs["reason"]
    # Slot 2 kernel up, slot 1 asked for: runs, it is the other seat.
    _c, outputs, _a, _s = _drive_gate(
        monkeypatch, tmp_path, holder = sampled, extra = ("--slot", "1"), holder_slot = "2"
    )
    assert outputs["should_run"] == "true", outputs["reason"]
    _c, outputs, _a, _s = _drive_gate(monkeypatch, tmp_path, holder = sampled, holder_slot = "1")
    assert outputs["should_run"] == "false", outputs["reason"]
    steps = {s.get("id"): s for _j, _n, s in _steps(_wf(NOTEBOOK_WF)) if s.get("id")}
    for step_id, script in (
        ("decide", "gate.py"),
        ("collect", "collect.py"),
        ("launch", "launch.py"),
    ):
        run = steps[step_id]["run"]
        assert script in run
        assert "--slot '${{ inputs.slot || '1' }}'" in run, f"{step_id} is not told the slot"
    assert "--sha '${{ steps.ref.outputs.ref }}'" in steps["collect"]["run"]


def test_the_surveys_share_one_budget(monkeypatch, tmp_path):
    """Two surveys each bounded by SURVEY_BUDGET_SEC outlive the gate job back
    to back. The second gets whatever the first left of ONE budget."""
    clock = [1000.0]
    _sampled, other = _other("abcdef0123456789" + "0" * 24)
    _code, _outputs, asked, _sha = _drive_gate(monkeypatch, tmp_path, holder = other, clock = clock)
    assert len(asked) == 2, asked
    first, second = asked[0][1], asked[1][1]
    assert first == pytest.approx(gate.SURVEY_BUDGET_SEC)
    assert second == pytest.approx(gate.SURVEY_BUDGET_SEC - 170.0), (first, second)


def test_the_gate_is_keyed_on_the_commit_the_gpu_job_will_test():
    """A dispatch naming `unsloth_ref` tests that ref, not github.sha. Keyed on
    github.sha the gate looks for the wrong kernel, and once the default branch
    moves two runs of one ref can land on different accounts."""
    for path in (NOTEBOOK_WF, STUDIO_WF):
        gate_job = _wf(path)["jobs"]["gate"]
        steps = {s.get("id"): s for s in gate_job["steps"] if s.get("id")}
        ref = steps.get("ref")
        assert ref is not None, f"{path.name}: the gate job does not name the commit under test"
        assert ref["env"]["UNSLOTH_REF"] == "${{ inputs.unsloth_ref }}"
        assert "head_sha=" in ref["run"] and "git ls-remote" in ref["run"]
        decide = steps["decide"]
        assert (
            "--head-sha '${{ steps.ref.outputs.head_sha }}'" in decide["run"]
        ), f"{path.name}: the gate is keyed on a commit the GPU job may not test"
        assert "github.sha" not in decide["run"]
        # The GPU job reuses the gate's SHA instead of resolving a moving ref again.
        assert gate_job["outputs"]["head_sha"] == "${{ steps.ref.outputs.head_sha }}"
        gpu = next(
            s
            for _j, _n, s in _steps(_wf(path))
            if s.get("id") == "ref" and "GATE_SHA" in (s.get("env") or {})
        )
        assert gpu["env"]["GATE_SHA"] == "${{ needs.gate.outputs.head_sha }}"
        assert "$(git ls-remote" not in gpu["run"]
