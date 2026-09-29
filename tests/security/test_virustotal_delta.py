# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The VirusTotal delta has to be able to report a regression, and has to refuse to guess.

Six hardening passes shipped with no before-and-after number. This tool produces one, so the way it
fails matters more than the way it succeeds: a comparison that silently never reports a regression is
indistinguishable from one that keeps passing, and a network call that breaks is loud while a
comparison that quietly stops comparing is not.

The other property tested here is that "could not measure" never reads as "clean". A missing API key,
an unknown hash and a file VirusTotal knows but has never analysed are all VOID. That last one is the
subtle case: zero engine verdicts is not sixty engines clearing the file.
"""

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


# ---------------------------------------------------------------------------
# The baseline is a real file, not a number someone typed
# ---------------------------------------------------------------------------


def test_the_baseline_hash_is_the_file_the_reporter_ran() -> None:
    """Recomputed from this repository's history rather than trusted.

    A baseline hash copied from an issue is a number nobody can check. This one is
    `install.ps1` at `1ad44677d`, and if the constant and the history ever disagree, the whole
    comparison is against the wrong file while still looking perfectly healthy.
    """
    # Shape first, and unconditionally. The git half below skips on a shallow clone, which is every
    # CI lane that collects tests/security -- so without this, the single assertion tying the
    # baseline to real history would be green locally and silently absent everywhere that matters.
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


# ---------------------------------------------------------------------------
# It must report a regression
# ---------------------------------------------------------------------------


def test_a_new_high_severity_sigma_rule_is_worse() -> None:
    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    payload["data"]["attributes"]["sigma_analysis_stats"] = {"high": 2, "medium": 11, "low": 5}
    delta = vtd.compare(_baseline(), _snap(payload))
    assert delta.worse and delta.exit_code() == 2


def test_a_different_engine_flagging_is_worse_even_at_the_same_count() -> None:
    """The failure mode a count-only comparison has by construction.

    One engine before, one engine after, so the count says nothing changed. But Skyhigh being
    replaced by Microsoft is the most consequential change this file could undergo: Defender is on
    every Windows machine and Skyhigh is enterprise-deployed.
    """
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
    """The stated intent, finally enforced.

    Reporting each bucket independently cannot express a trade: it moves two buckets in opposite
    directions, so it landed in `worse` and in `better` at once and exit_code answered `worse`.
    A run that swapped the single high rule for three extra low ones was therefore rejected while
    the comment beside it called that exact swap an improvement.
    """
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
    """The guard existed on one side only.

    A baseline hash VirusTotal knows but has never analysed carries no engines, no Sigma and no
    YARA. Compared against a real candidate, every finding reads as newly introduced, so the run
    exited 2 and named a list of regressions while having compared against nothing at all.
    """
    empty = {
        "data": {"attributes": {"size": 1, "last_analysis_stats": {}, "last_analysis_results": {}}}
    }
    baseline = _snap(empty, "baseline", "a" * 64)
    delta = vtd.compare(baseline, _snap(copy.deepcopy(vtd._BASELINE_FIXTURE)))
    assert delta.exit_code() == 3, "an unanalysed baseline was compared against instead of voiding"
    assert not delta.worse, f"it reported regressions against an empty baseline: {delta.worse}"


# ---------------------------------------------------------------------------
# It must recognise an improvement without overstating it
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# VOID is not clean
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# The tool's own controls, and the parsers
# ---------------------------------------------------------------------------


def test_the_self_test_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-test"], capture_output = True, text = True, timeout = 120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_self_test_can_fail() -> None:
    """A control that cannot fail is decoration."""
    original = vtd.compare
    try:
        vtd.compare = lambda base, cand: vtd.Delta()  # never reports anything
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
    # Parsed, not grepped. The module docstring explains at length why there is no upload path, so a
    # substring search over the file text matches its own explanation -- which is how the first
    # version of this test failed. What matters is what the code does.
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
    """`VT_API_KEY` is the env var the Python reads; the repository secret is
    `VIRUS_TOTAL_API_TOKEN`.

    Naming the secret `VT_API_KEY` in the workflow expands to empty, and the lane then exits 3 and
    reports VOID on every run. That fails loudly rather than reporting a false clean, which is the
    right shape -- but a lane that can never measure anything is worth catching here rather than
    after someone dispatches it and waits.
    """
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
    # And the history check the lane performs needs full depth, or it reverifies nothing.
    data = _yaml.safe_load(body)
    checkout = next(
        s for s in data["jobs"]["delta"]["steps"] if "checkout" in str(s.get("uses", ""))
    )
    assert (
        checkout.get("with", {}).get("fetch-depth") == 0
    ), "the baseline is reverified against 1ad44677d, which a shallow clone does not contain"


def test_two_rulesets_sharing_a_rule_name_stay_distinct() -> None:
    """A hit gained from another ruleset must not compare equal to the baseline's.

    Crowdsourced rulesets are independent, so the same rule identifier can appear in two of them.
    Keying a hit on the rule name alone collapsed them, and a candidate that picked up a hit from a
    second ruleset still parsed to the baseline's list, so the comparison reported YARA unchanged
    and the run exited 0 on a genuine regression.
    """
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
    """Engine names and rule names are third-party data rendered as Markdown.

    The summary is appended to `$GITHUB_STEP_SUMMARY`, where a newline ends the row or bullet, `|`
    opens a new cell and `<` begins HTML that GitHub renders. `virustotal_scan._md_text` exists for
    exactly this and the report was not using it.
    """
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
    """The report identified every baseline as install.ps1 at 1ad44677d, override or not.

    A dispatch can supply `baseline_sha256`, and the lookup and comparison then use that hash while
    the header still claimed the recorded identity and printed that file's historical scores. The
    resulting artifact combines one file's live table with another's name, which is a delta built
    to be misread.
    """
    # The real recorded hash, not the placeholder the other helpers use: the label is chosen by
    # comparing against it, so a stand-in would test nothing.
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
    """The deadline this tool added raised past its own handler.

    `VirusTotalClient.request` signals a spent budget with `TimeoutError`, which inherits from
    `OSError` and not from `RuntimeError`, so the `except RuntimeError` in `fetch` let it through.
    The workflow then died with a traceback before `render` wrote anything, which is the lost run
    the deadline was added to prevent.
    """

    class _Expired:
        def request(self, *args, **kwargs):
            raise TimeoutError("deadline reached before GET /files/x")

    snap = vtd.fetch(_Expired(), "d" * 64, "candidate", deadline = 0.0)
    assert snap.total_engines == 0, "a timed-out lookup invented engine verdicts"
    assert "budget" in snap.note, snap.note


def test_a_renamed_sigma_bucket_is_not_silently_dropped() -> None:
    """The parser warned about renamed buckets in its docstring and then read a fixed key list.

    A candidate that gains rules only in a bucket outside critical/high/medium/low produced a Sigma
    dictionary identical to the baseline's, so the comparison reported Sigma unchanged and the run
    exited 0 on a real regression. The day VirusTotal renames or adds a bucket is exactly the day
    this measurement matters.
    """
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
    """The ordered comparison depends on knowing which bucket is more severe, so it keeps its own.

    Trading a high for several lows is still the improvement the comment claims, and an unranked
    bucket that did not move must not turn it into a regression.
    """
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
    """`ScanStats.total` sums five buckets, and `parse_stats` documents more than five.

    `type-unsupported` and `failure` are named in that parser's own docstring as categories
    VirusTotal has added, but the dataclass has no field for them, so the engine count this report
    prints was short by however many engines answered that way. Worse, a response made up entirely
    of those buckets summed to zero, which this tool treats as a file nothing has scanned and voids.
    """
    payload = copy.deepcopy(vtd._BASELINE_FIXTURE)
    stats = payload["data"]["attributes"]["last_analysis_stats"]
    before = vtd.snapshot_from_payload("candidate", "b" * 64, payload).total_engines
    stats["type-unsupported"] = 5
    stats["failure"] = 2
    after = vtd.snapshot_from_payload("candidate", "b" * 64, payload).total_engines
    assert (
        after == before + 7
    ), f"engines that answered in a newer bucket were not counted: {before} -> {after}"

    # The bucket-only case, which used to read as unanalysed and void the run.
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
    """Absence from the candidate's results is not a clean verdict from that engine.

    Skyhigh is the one engine that actually flags this file, so "Skyhigh no longer flags it" is the
    single most consequential sentence this report can print. Deriving it from a set difference
    made a sparse or older candidate analysis -- one where Skyhigh simply had not run -- say
    exactly that, and exit 0.
    """
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

    # And the real improvement still reads as one: Skyhigh answered, and answered undetected.
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
