# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A VirusTotal delta must report regressions and treat unmeasurable results as VOID, not clean."""

from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "virustotal_delta.py"

sys.path.insert(0, str(REPO / "scripts"))
import virustotal_delta as vtd  # noqa: E402


def _snap(
    payload: dict,
    label: str = "candidate",
    sha: str = "b" * 64,
):
    return vtd.snapshot_from_payload(label, sha, payload)


def _baseline():
    return _snap(vtd._BASELINE_FIXTURE, "baseline", "a" * 64)


def test_the_baseline_hash_is_the_file_the_reporter_ran() -> None:
    """The baseline SHA-256 is recomputed from install.ps1 in git history, not copied from an issue."""
    # Shape check is unconditional: the git half below skips on the shallow clones CI uses.
    assert len(vtd.BASELINE_SHA256) == 64, "the baseline is not a SHA-256"
    assert all(
        c in "0123456789abcdef" for c in vtd.BASELINE_SHA256
    ), "the baseline is not lowercase hex, so it can never match a VirusTotal lookup"

    result = subprocess.run(
        ["git", "show", "1ad44677d:install.ps1"],
        cwd = REPO,
        capture_output = True,
        timeout = 120,
    )
    if result.returncode != 0:
        pytest.skip("that commit is not present in this clone (shallow checkout)")
    blob = result.stdout
    assert hashlib.sha256(blob).hexdigest() == vtd.BASELINE_SHA256, (
        "BASELINE_SHA256 is not the hash of install.ps1 at 1ad44677d. Every delta this tool has "
        "ever reported was against the wrong file."
    )
    assert len(blob) == 427113, (
        f"install.ps1 at 1ad44677d is {len(blob)} bytes, not the 427,113 the reported sample is "
        f"recorded as, so this is not the revision the user in #10805 ran"
    )


def test_the_recorded_baseline_note_matches_the_fixture() -> None:
    """The prose and the fixture are two statements of the same fact, and they drift apart in
    exactly the situation where someone is reading one and trusting the other."""
    baseline = _baseline()
    assert baseline.sigma_total == 17
    assert baseline.sigma == {"high": 1, "medium": 11, "low": 5}
    assert baseline.engines == ["Skyhigh (BehavesLike.PS.Suspicious.gr)"]
    assert len(baseline.yara) == 2
    for fragment in ("1 high, 11 medium, 5 low", "Skyhigh", "2 YARA"):
        assert (
            fragment in vtd.BASELINE_NOTE
        ), f"the recorded note no longer says {fragment!r}, but the fixture still does"


def test_a_new_high_severity_sigma_rule_is_worse() -> None:
    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    payload["data"]["attributes"]["sigma_analysis_stats"] = {"high": 2, "medium": 11, "low": 5}
    delta = vtd.compare(_baseline(), _snap(payload))
    assert delta.worse and delta.exit_code() == 2


def test_a_different_engine_flagging_is_worse_even_at_the_same_count() -> None:
    """A count-only comparison misses an engine swap; a new engine flagging must count as worse."""
    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    payload["data"]["attributes"]["last_analysis_results"] = {
        "Microsoft": {"category": "malicious", "result": "Trojan:Script/Wacatac.B!ml"},
        "Skyhigh": {"category": "undetected", "result": None},
    }
    delta = vtd.compare(_baseline(), _snap(payload))
    assert delta.worse, "an engine swap at the same count was reported as no difference"
    assert any("Microsoft" in row for row in delta.worse)


def test_a_new_yara_hit_is_worse() -> None:
    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    payload["data"]["attributes"]["crowdsourced_yara_results"] = [
        {"rule_name": "SUSP_PS1_Shape_A"},
        {"rule_name": "SUSP_PS1_Shape_B"},
        {"rule_name": "SOMETHING_NEW"},
    ]
    delta = vtd.compare(_baseline(), _snap(payload))
    assert delta.worse and delta.exit_code() == 2


def test_severity_is_compared_per_bucket_and_not_in_total() -> None:
    """Trading one high for three lows is an improvement that a total calls a regression, and the
    reverse is a regression a total calls an improvement."""
    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    # 17 rules before, 17 after, but one medium became a high.
    payload["data"]["attributes"]["sigma_analysis_stats"] = {"high": 2, "medium": 10, "low": 5}
    delta = vtd.compare(_baseline(), _snap(payload))
    assert delta.exit_code() == 2
    assert any("high" in row for row in delta.worse)


def test_trading_a_high_for_several_lows_is_the_improvement_the_comment_claims() -> None:
    """Trading a high for several lows must pass; per-bucket comparison cannot express a trade."""
    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    payload["data"]["attributes"]["sigma_analysis_stats"] = {"high": 0, "medium": 11, "low": 8}
    delta = vtd.compare(_baseline(), _snap(payload))
    assert delta.exit_code() == 0, f"a high rule traded for lows was rejected: {delta.worse}"
    assert any("high" in row for row in delta.better)
    # The lower buckets are still reported; they just do not decide.
    assert any("low" in row for row in delta.better)


def test_a_low_traded_for_a_high_is_still_a_regression() -> None:
    """The reverse of the trade above, which a total would have called an improvement."""
    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    payload["data"]["attributes"]["sigma_analysis_stats"] = {"high": 2, "medium": 11, "low": 0}
    delta = vtd.compare(_baseline(), _snap(payload))
    assert delta.exit_code() == 2
    assert any("high" in row for row in delta.worse)


def test_a_baseline_with_no_engine_verdicts_is_void_like_the_candidate() -> None:
    """A baseline with no engine verdicts must be VOID, or every finding reads as newly introduced."""
    empty = {
        "data": {"attributes": {"size": 1, "last_analysis_stats": {}, "last_analysis_results": {}}}
    }
    baseline = _snap(empty, "baseline", "a" * 64)
    delta = vtd.compare(baseline, _snap(copy.deepcopy(vtd._BASELINE_FIXTURE)))
    assert delta.exit_code() == 3, "an unanalysed baseline was compared against instead of voiding"
    assert not delta.worse, f"it reported regressions against an empty baseline: {delta.worse}"


def test_a_clear_improvement_passes_and_is_named() -> None:
    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    payload["data"]["attributes"]["sigma_analysis_stats"] = {"medium": 4, "low": 5}
    payload["data"]["attributes"]["crowdsourced_yara_results"] = []
    delta = vtd.compare(_baseline(), _snap(payload))
    assert delta.exit_code() == 0
    assert delta.better and not delta.worse


def test_an_unmoved_engine_verdict_is_reported_as_unchanged() -> None:
    """A cloud behavioural verdict is not recomputed because we deleted some code. Sigma moving
    while the engine does not is the expected shape of a win here, and reporting it as a clean bill
    of health would be the overclaim this tool exists to avoid."""
    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    payload["data"]["attributes"]["sigma_analysis_stats"] = {"medium": 4}
    delta = vtd.compare(_baseline(), _snap(payload))
    assert any("unchanged" in row for row in delta.same)
    assert any("Skyhigh" in row for row in delta.same)


def test_an_unknown_candidate_hash_is_void() -> None:
    missing = vtd.Snapshot(label = "candidate", sha256 = "f" * 64, note = "not present on VirusTotal")
    delta = vtd.compare(_baseline(), missing)
    assert delta.exit_code() == 3
    assert (
        not delta.same and not delta.worse and not delta.better
    ), "an unknown hash produced comparison rows it cannot support"


def test_a_known_but_never_analysed_file_is_void() -> None:
    """The subtle one. Zero engine verdicts is not sixty engines clearing the file."""
    payload = {
        "data": {"attributes": {"size": 1, "last_analysis_stats": {}, "last_analysis_results": {}}}
    }
    delta = vtd.compare(_baseline(), _snap(payload))
    assert delta.exit_code() == 3


def test_a_missing_api_key_exits_three_and_never_says_clean() -> None:
    env = {k: v for k, v in os.environ.items() if k != vtd.API_KEY_ENV}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--candidate-sha256", "a" * 64],
        capture_output = True,
        text = True,
        timeout = 120,
        env = env,
    )
    assert result.returncode == 3, (
        "a missing key must not exit zero. It is the most likely reason this ever produces no "
        f"comparison, and it must not be spelled the same as 'nothing got worse'.\n{result.stdout}"
    )
    assert "COULD NOT MEASURE" in result.stdout
    assert "clean" not in result.stdout.lower().replace("not a clean result", "")


def test_the_three_outcomes_have_distinct_exit_codes() -> None:
    void, worse, fine = vtd.Delta(), vtd.Delta(), vtd.Delta()
    void.void.append("x")
    worse.worse.append("y")
    assert (void.exit_code(), worse.exit_code(), fine.exit_code()) == (3, 2, 0)


def test_the_self_test_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-test"], capture_output = True, text = True, timeout = 120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_self_test_can_fail() -> None:
    """A control that cannot fail is decoration."""
    original = vtd.compare
    try:
        vtd.compare = lambda base, cand: vtd.Delta()
        failures = vtd.self_test()
        assert failures, "a comparison that reports nothing at all still passed the controls"
    finally:
        vtd.compare = original


@pytest.mark.parametrize("raw", [None, [], "high", {"high": "two"}, {"high": True}])
def test_the_sigma_parser_survives_a_schema_change(raw) -> None:
    """VirusTotal has renamed and added buckets over time. A crash here would break the measurement
    on precisely the day the schema moved, which is when it is most worth having."""
    assert vtd.parse_sigma(raw) == {}


@pytest.mark.parametrize("raw", [None, {}, "x", [1, 2], [{"no_name": 1}]])
def test_the_yara_parser_survives_a_schema_change(raw) -> None:
    assert vtd.parse_yara(raw) == []


def test_the_tool_has_no_upload_path_at_all() -> None:
    """Not an oversight. A freshly uploaded file is a prevalence-zero first-seen sample, which is
    what the strict cloud settings punish -- so uploading a candidate can create the detection it
    was meant to measure, under a hash no user will ever have."""
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding = "utf-8"))
    # Parsed, not grepped: the module docstring itself explains there is no upload path.
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            calls.append(node.value)
    methods = {c.upper() for c in calls if c.upper() in {"POST", "PUT", "PATCH", "DELETE"}}
    assert not methods, (
        f"the delta tool now issues {sorted(methods)}. It must only ever GET: a freshly uploaded "
        f"file is a prevalence-zero first-seen sample, so uploading a candidate can create the "
        f"detection it was meant to measure, under a hash no user will ever have."
    )
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    assert (
        "scan_file" not in imported and "upload_file" not in imported
    ), "the delta tool imported an upload helper from virustotal_scan"


def test_the_workflow_reads_the_secret_this_repository_actually_has() -> None:
    """The workflow must read the VIRUS_TOTAL_API_TOKEN secret, since a misnamed one voids every run."""
    import yaml as _yaml

    workflow = REPO / ".github" / "workflows" / "virustotal-installer-delta.yml"
    body = workflow.read_text(encoding = "utf-8")
    assert "secrets.VIRUS_TOTAL_API_TOKEN" in body, (
        "the delta lane no longer reads secrets.VIRUS_TOTAL_API_TOKEN, which is the only VirusTotal "
        "secret this repository defines"
    )
    assert "secrets.VT_API_KEY" not in body, (
        "the workflow reads secrets.VT_API_KEY, which does not exist. VT_API_KEY is the ENV VAR "
        "name; the secret is VIRUS_TOTAL_API_TOKEN."
    )
    data = _yaml.safe_load(body)
    checkout = next(
        s for s in data["jobs"]["delta"]["steps"] if "checkout" in str(s.get("uses", ""))
    )
    assert (
        checkout.get("with", {}).get("fetch-depth") == 0
    ), "the baseline is reverified against 1ad44677d, which a shallow clone does not contain"


def test_two_rulesets_sharing_a_rule_name_stay_distinct() -> None:
    """YARA hits are keyed by ruleset and rule name, as rule names repeat across rulesets."""
    parsed = vtd.parse_yara(
        [
            {"ruleset_name": "set_a", "rule_name": "SUSP_Script"},
            {"ruleset_name": "set_b", "rule_name": "SUSP_Script"},
        ]
    )
    assert len(parsed) == 2, f"the two rulesets collapsed to one hit: {parsed}"

    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    payload["data"]["attributes"]["crowdsourced_yara_results"] = [
        {"ruleset_name": "set_a", "rule_name": "SUSP_Script"},
        {"ruleset_name": "set_b", "rule_name": "SUSP_Script"},
    ]
    baseline_payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    baseline_payload["data"]["attributes"]["crowdsourced_yara_results"] = [
        {"ruleset_name": "set_a", "rule_name": "SUSP_Script"},
    ]
    delta = vtd.compare(_snap(baseline_payload, "baseline", "a" * 64), _snap(payload))
    assert delta.exit_code() == 2, "a hit gained from a second ruleset was reported as unchanged"
    assert any("YARA" in row for row in delta.worse), delta.worse


def test_third_party_text_cannot_break_the_job_summary() -> None:
    """Engine and rule names must go through virustotal_scan._md_text before reaching the job summary."""
    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    payload["data"]["attributes"]["last_analysis_results"] = {
        "Evil|Engine": {"category": "malicious", "result": "x | y\n| broken | row |<img src=x>"},
    }
    delta = vtd.compare(_baseline(), _snap(payload))
    report = vtd.render(_baseline(), _snap(payload), delta)
    assert "<img" not in report, "raw HTML from a detection label reached the summary"
    for line in report.splitlines():
        if line.startswith("- ") or (line.startswith("|") and "---" not in line):
            assert "\n" not in line
    assert (
        "Evil\\|Engine" in report or "Evil|Engine" not in report
    ), "the engine name's pipe was not escaped, so it opens a new table cell"


def test_an_overridden_baseline_is_not_labelled_as_the_recorded_one() -> None:
    """Overridden baselines must not carry the recorded install.ps1 label, or scores are misattributed."""
    # The real recorded hash: the label is chosen by comparing against it.
    real = _snap(copy.deepcopy(vtd._BASELINE_FIXTURE), "baseline", vtd.BASELINE_SHA256)
    recorded = vtd.render(real, _snap(copy.deepcopy(vtd._BASELINE_FIXTURE)), vtd.Delta())
    assert vtd.BASELINE_NOTE in recorded, "the recorded baseline lost its provenance note"

    other = _snap(copy.deepcopy(vtd._BASELINE_FIXTURE), "baseline", "c" * 64)
    overridden = vtd.render(other, _snap(copy.deepcopy(vtd._BASELINE_FIXTURE)), vtd.Delta())
    assert (
        vtd.BASELINE_NOTE not in overridden
    ), "an overridden baseline is still labelled as install.ps1 at the recorded commit"
    assert "OVERRIDDEN" in overridden, overridden.splitlines()[:4]


def test_a_spent_deadline_becomes_a_void_row_and_not_a_crash() -> None:
    """A spent deadline raises TimeoutError, not RuntimeError, so fetch must catch it to void the row."""

    class _Expired:
        def request(self, *args, **kwargs):
            raise TimeoutError("deadline reached before GET /files/x")

    snap = vtd.fetch(_Expired(), "d" * 64, "candidate", deadline = 0.0)
    assert snap.total_engines == 0, "a timed-out lookup invented engine verdicts"
    assert "budget" in snap.note, snap.note


def test_a_renamed_sigma_bucket_is_not_silently_dropped() -> None:
    """parse_sigma must keep unrecognised buckets, since a rule in a renamed bucket would vanish."""
    assert vtd.parse_sigma({"informational": 3}) == {
        "informational": 3
    }, "an unrecognised severity bucket is still dropped by the parser"

    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    payload["data"]["attributes"]["sigma_analysis_stats"] = {"high": 1, "informational": 4}
    base_payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    base_payload["data"]["attributes"]["sigma_analysis_stats"] = {"high": 1}
    delta = vtd.compare(_snap(base_payload, "baseline", "a" * 64), _snap(payload))
    assert delta.worse, "rules gained in an unranked bucket were reported as no change"
    assert any("informational" in row for row in delta.worse), delta.worse
    assert delta.exit_code() == 2, delta


def test_an_unranked_bucket_never_overrides_the_ordered_tradeoff() -> None:
    """An unranked bucket that did not move must not turn a high-for-lows trade into a regression."""
    base_payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    base_payload["data"]["attributes"]["sigma_analysis_stats"] = {
        "high": 1,
        "low": 5,
        "informational": 2,
    }
    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    payload["data"]["attributes"]["sigma_analysis_stats"] = {
        "high": 0,
        "low": 8,
        "informational": 2,
    }
    delta = vtd.compare(_snap(base_payload, "baseline", "a" * 64), _snap(payload))
    assert delta.exit_code() == 0, (delta.worse, delta.better)


def test_engines_that_answered_in_a_newer_bucket_still_count() -> None:
    """ScanStats.total must count the type-unsupported and failure buckets, or engines go uncounted."""
    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    stats = payload["data"]["attributes"]["last_analysis_stats"]
    before = vtd.snapshot_from_payload("candidate", "b" * 64, payload).total_engines
    stats["type-unsupported"] = 5
    stats["failure"] = 2
    after = vtd.snapshot_from_payload("candidate", "b" * 64, payload).total_engines
    assert (
        after == before + 7
    ), f"engines that answered in a newer bucket were not counted: {before} -> {after}"

    only_new = copy.deepcopy(vtd._BASELINE_FIXTURE)
    only_new["data"]["attributes"]["last_analysis_stats"] = {"type-unsupported": 3}
    snap = vtd.snapshot_from_payload("candidate", "b" * 64, only_new)
    assert snap.total_engines == 3, snap.total_engines
    assert "no engine verdicts" not in snap.note, snap.note


def test_a_non_numeric_bucket_cannot_inflate_the_engine_count() -> None:
    """VirusTotal sends counts; a bool or a string in that position must not add one each."""
    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    clean = vtd.snapshot_from_payload("candidate", "b" * 64, payload).total_engines
    payload["data"]["attributes"]["last_analysis_stats"]["weird"] = True
    payload["data"]["attributes"]["last_analysis_stats"]["odd"] = "12"
    assert vtd.snapshot_from_payload("candidate", "b" * 64, payload).total_engines == clean


def test_an_engine_that_did_not_answer_has_not_cleared_us() -> None:
    """An engine absent from the results has not cleared the file, so absence must not read as clean."""
    base_payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    base_payload["data"]["attributes"]["last_analysis_results"] = {
        "Skyhigh": {"category": "malicious", "result": "BehavesLike.PS.Suspicious.gr"},
        "Microsoft": {"category": "undetected", "result": None},
    }
    baseline = _snap(base_payload, "baseline", "a" * 64)

    # Skyhigh did not run on the candidate at all.
    silent = copy.deepcopy(vtd._BASELINE_FIXTURE)
    silent["data"]["attributes"]["last_analysis_results"] = {
        "Microsoft": {"category": "undetected", "result": None},
    }
    delta = vtd.compare(baseline, _snap(silent))
    assert not any(
        "no longer flag" in row for row in delta.better
    ), f"an engine that never answered was reported as having cleared the candidate: {delta.better}"
    assert any("NOT cleared" in row for row in delta.same), delta.same

    cleared = copy.deepcopy(vtd._BASELINE_FIXTURE)
    cleared["data"]["attributes"]["last_analysis_results"] = {
        "Skyhigh": {"category": "undetected", "result": None},
        "Microsoft": {"category": "undetected", "result": None},
    }
    delta = vtd.compare(baseline, _snap(cleared))
    assert any("no longer flag" in row for row in delta.better), delta.better
    assert any("Skyhigh" in row for row in delta.better), delta.better


@pytest.mark.parametrize(
    "category", ["timeout", "confirmed-timeout", "failure", "type-unsupported"]
)
def test_an_inconclusive_result_does_not_clear_a_prior_detection(category: str) -> None:
    """An engine that timed out has not cleared us any more than one that never ran.

    The responder set was built on the truthiness of `category`, so these four entries counted as
    answers while `parse_detections` correctly excluded them from the flagging list. The engine then
    appeared in neither set and was reported as having stopped flagging the candidate.
    """
    base_payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    base_payload["data"]["attributes"]["last_analysis_results"] = {
        "Skyhigh": {"category": "malicious", "result": "BehavesLike.PS.Suspicious.gr"},
        "Microsoft": {"category": "undetected", "result": None},
    }
    baseline = _snap(base_payload, "baseline", "a" * 64)

    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    payload["data"]["attributes"]["last_analysis_results"] = {
        "Skyhigh": {"category": category, "result": None},
        "Microsoft": {"category": "undetected", "result": None},
    }
    delta = vtd.compare(baseline, _snap(payload))
    assert not any(
        "no longer flag" in row for row in delta.better
    ), f"a {category!r} result was treated as Skyhigh clearing the candidate: {delta.better}"
    assert any("NOT cleared" in row for row in delta.same), delta.same
