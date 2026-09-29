#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compare a candidate installer against a known-flagged baseline on VirusTotal.

Six antivirus-hardening passes shipped without a before-and-after number, because nobody had one.
There is one now: `install.ps1` at commit `1ad44677d` -- the revision reported in
unslothai/unsloth#10805 -- is `ec29980b...`, and VirusTotal holds its verdict. That gives this work a
baseline for the first time, and this tool is the diff against it.

Three numbers, because they move independently and mean different things:

  engine verdicts   Which of ~60 engines flagged the file, and with what label. Exactly one flags the
                    baseline: Skyhigh (Trellix / McAfee Enterprise) with `BehavesLike.PS.Suspicious.gr`,
                    a generic behavioural label. This is the number a user feels and the slowest to
                    move, since a cloud behavioural verdict is not recomputed because we deleted some
                    code.
  Sigma rules       17 on the baseline (1 high, 11 medium, 5 low). Static, deterministic, and what
                    drives the alarming scores third-party analysis sites report. This is the number
                    shape work moves, and it should move immediately.
  YARA rules        2 crowdsourced hits on the baseline. Also static.

Reporting only the engine count would make every shape improvement look like it achieved nothing, and
reporting only Sigma would overclaim. Both, separately, or the number is not worth having.

## Hash lookup only. Never upload.

This tool only ever issues `GET /files/{sha256}`. It has no upload path at all, and that is a
deliberate omission rather than an oversight: a freshly uploaded file is a prevalence-zero, first-seen
sample, which is precisely what `CloudBlockLevel HighPlus` and ASR rule `01443614` punish. Uploading a
candidate can therefore *create* the detection it was meant to measure, under a hash no user will ever
have. Released artefacts get uploaded by `virustotal_scan.py` at release time; branches do not.

The consequence is that an unknown hash is a **VOID** outcome, not a clean one. Current `main`'s
`install.ps1` is unknown to VirusTotal today, and "0 engines flagged it" for a file nothing has ever
scanned is the single most misleading sentence this tool could print.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from virustotal_scan import (  # noqa: E402
    API_KEY_ENV,
    API_ROOT,
    VirusTotalClient,
    _md_text,
    parse_detections,
    parse_stats,
)


# The revision reported in #10805, hashed from this repository's own history rather than copied from
# the report: `git show 1ad44677d:install.ps1 | sha256sum` gives exactly this, at exactly the 427,113
# bytes the sample is recorded as. That is what makes it the file that user actually ran, and a test
# recomputes it rather than trusting this constant.
BASELINE_SHA256 = "ec29980bffff30740f876f4955d890e0f1a1c9fd681521be6fbfccd24ffae296"

# What the baseline scored, recorded so a run can say "unchanged" or "moved" without a second network
# call, and so a reader can sanity-check the live answer against what we believed.
BASELINE_NOTE = (
    "install.ps1 at 1ad44677d: 1 malicious / 58 undetected, Skyhigh "
    "BehavesLike.PS.Suspicious.gr; 17 Sigma rules (1 high, 11 medium, 5 low); 2 YARA hits"
)

SEVERITIES = ("critical", "high", "medium", "low")


@dataclass
class Snapshot:
    """What VirusTotal holds about one file."""

    label: str
    sha256: str = ""
    found: bool = False
    size: int = 0
    first_seen: str = ""
    engines: list[str] = field(default_factory = list)
    # Every engine that returned a verdict of ANY kind. Needed because an engine missing from the
    # candidate's results has not cleared it -- it did not look -- and a set difference against the
    # flagging engines alone cannot tell those two apart.
    responders: set[str] = field(default_factory = set)
    malicious: int = 0
    suspicious: int = 0
    total_engines: int = 0
    sigma: dict[str, int] = field(default_factory = dict)
    yara: list[str] = field(default_factory = list)
    note: str = ""

    @property
    def sigma_total(self) -> int:
        return sum(self.sigma.values())


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# The only categories that mean "this engine looked and did not object". `timeout`,
# `confirmed-timeout`, `failure` and `type-unsupported` are all entries an engine can return without
# having reached a verdict, and treating them as answers is the same mistake as treating absence as
# one: it lets a detection be reported as cleared by an engine that never decided.
CONCLUSIVE_CLEAN = ("undetected", "harmless")


def responding_engines(raw: object) -> set[str]:
    """Names of every engine that reached a conclusive verdict, flagging or clean."""
    if not isinstance(raw, dict):
        return set()
    conclusive = set(CONCLUSIVE_CLEAN) | {"malicious", "suspicious"}
    return {
        str(engine)
        for engine, result in raw.items()
        if isinstance(result, dict) and result.get("category") in conclusive
    }


def count_all_verdicts(raw: object, stats) -> int:
    """How many engines answered, counting buckets `ScanStats` has no field for.

    `parse_stats` names `type-unsupported` and `failure` as categories VirusTotal has added, but
    `ScanStats` carries fields only for malicious, suspicious, undetected, harmless and timeout, so
    its `total` undercounts the denominator this report prints, and a response made up entirely of
    the omitted buckets looks like a file nothing has scanned. Summing every numeric bucket in the
    raw dict keeps the shared dataclass untouched, which matters because the release scanner uses
    it too, while still counting what actually came back.
    """
    if not isinstance(raw, dict):
        return stats.total
    total = 0
    for value in raw.values():
        # Booleans are ints in Python and would each add one; VirusTotal sends counts, not flags.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        total += int(value)
    return total


def parse_sigma(raw: object) -> dict[str, int]:
    """Sigma counts by severity.

    VirusTotal has renamed and added buckets over time, so every key is read defensively rather than
    indexed. A KeyError here would turn a measurement into a crash on a day VirusTotal shipped a
    schema change, which is exactly when the measurement matters.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    # Every bucket VirusTotal reports, not only the four this tool knows how to rank. Iterating a
    # fixed key list is what the docstring above warns about and then did anyway: a candidate that
    # gained rules only in a renamed or newly added bucket produced a dictionary identical to the
    # baseline's, so compare() called Sigma unchanged and the run exited 0 on a regression.
    # Unrankable buckets are kept here and handled separately in compare().
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if int(value):
            out[key] = int(value)
    return out


def parse_yara(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    names = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        # Ruleset AND rule. Two crowdsourced rulesets can carry the same rule identifier, and
        # keying on the rule alone let a candidate gain a hit from a different ruleset while the
        # parsed list stayed equal to the baseline's, so the comparison reported YARA unchanged and
        # the run exited 0.
        rule = entry.get("rule_name")
        ruleset = entry.get("ruleset_name")
        parts = [p for p in (ruleset, rule) if isinstance(p, str) and p]
        if parts:
            names.append("/".join(parts))
    return sorted(set(names))


def snapshot_from_payload(label: str, sha256: str, payload: object) -> Snapshot:
    """Build a snapshot from a `GET /files/{sha256}` body."""
    snap = Snapshot(label = label, sha256 = sha256)
    attributes = {}
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("attributes"), dict):
            attributes = data["attributes"]
    if not attributes:
        snap.note = "VirusTotal returned no attributes for this hash"
        return snap

    snap.found = True
    stats = parse_stats(attributes.get("last_analysis_stats"))
    snap.malicious = stats.malicious
    snap.suspicious = stats.suspicious
    snap.total_engines = count_all_verdicts(attributes.get("last_analysis_stats"), stats)
    snap.engines = parse_detections(attributes.get("last_analysis_results"))
    snap.responders = responding_engines(attributes.get("last_analysis_results"))
    snap.sigma = parse_sigma(attributes.get("sigma_analysis_stats"))
    snap.yara = parse_yara(attributes.get("crowdsourced_yara_results"))
    size = attributes.get("size")
    snap.size = int(size) if isinstance(size, int) else 0
    first = attributes.get("first_submission_date")
    snap.first_seen = str(first) if first else ""

    if snap.total_engines == 0:
        # Zero verdicts of any kind means no engine has run, which is not the same as every engine
        # having cleared it. Recorded so the verdict below can refuse to call it clean.
        snap.note = "no engine verdicts at all: the file is known but has not been analysed"
    return snap


# Below the 20 minutes the workflow gives the whole job. Without a deadline one lookup can spend
# four 300-second socket attempts plus 20, 40 and 80 second backoffs, so the baseline alone can eat
# the budget and the runner kills the step before it fetches the candidate or writes its summary --
# which is a lost run rather than a reported one. `request` checks this BEFORE each attempt, because
# a single attempt can block for the full socket timeout.
LOOKUP_BUDGET_SECONDS = 420.0


def fetch(
    client: VirusTotalClient,
    sha256: str,
    label: str,
    deadline: float | None = None,
) -> Snapshot:
    try:
        status, payload = client.request(
            "GET",
            f"{API_ROOT}/files/{sha256}",
            allow_status = (404,),
            deadline = deadline,
        )
    except (RuntimeError, TimeoutError) as exc:
        # The retry budget or the deadline is spent. VOID, not clean and not a crash: we did not
        # find out, and the summary has to say so while there is still time to write it.
        # Both are needed: a spent retry budget raises RuntimeError, but a spent deadline raises
        # TimeoutError, which is an OSError and would otherwise walk straight past this handler.
        snap = Snapshot(label = label, sha256 = sha256)
        snap.note = f"the lookup did not complete within its budget: {exc}"
        return snap
    if status == 404:
        snap = Snapshot(label = label, sha256 = sha256)
        snap.note = "not present on VirusTotal"
        return snap
    if status != 200:
        snap = Snapshot(label = label, sha256 = sha256)
        snap.note = f"VirusTotal answered {status}"
        return snap
    return snapshot_from_payload(label, sha256, payload)


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


@dataclass
class Delta:
    void: list[str] = field(default_factory = list)
    worse: list[str] = field(default_factory = list)
    better: list[str] = field(default_factory = list)
    same: list[str] = field(default_factory = list)

    def exit_code(self) -> int:
        # 3 = could not measure, 2 = worse than baseline, 0 = same or better. Distinct, because a
        # tool whose "we could not look" is spelled the same as "it got worse" is unreadable, and one
        # whose "we could not look" is spelled the same as "it is fine" is dangerous.
        if self.void:
            return 3
        if self.worse:
            return 2
        return 0


def compare(baseline: Snapshot, candidate: Snapshot) -> Delta:
    delta = Delta()

    if not candidate.found:
        delta.void.append(
            f"the candidate ({candidate.sha256[:16]}...) is not on VirusTotal: {candidate.note}. "
            f"This is the expected answer for an unreleased build, and it is NOT a clean result -- "
            f"'0 engines flagged it' for a file nothing has scanned is the most misleading thing "
            f"this tool could say. Upload released artefacts only; a fresh upload is a "
            f"prevalence-zero first-seen sample, which is what the strict cloud settings punish."
        )
    if not baseline.found:
        delta.void.append(
            f"the baseline ({baseline.sha256[:16]}...) is not on VirusTotal: {baseline.note}. "
            f"Without it there is nothing to compare against, and an absolute verdict on a 443 KB "
            f"script is noise."
        )
    if candidate.found and candidate.total_engines == 0:
        delta.void.append(f"the candidate has no engine verdicts: {candidate.note}")
    if baseline.found and baseline.total_engines == 0:
        # Same guard, same reason, other side. A baseline VirusTotal knows but has never analysed
        # carries no engines, no Sigma and no YARA, so every finding on the candidate reads as newly
        # introduced and the run exits 2 while having compared against nothing at all.
        delta.void.append(f"the baseline has no engine verdicts: {baseline.note}")
    if delta.void:
        return delta

    # Engines, by name. A count alone hides the case that matters most: the same number of
    # detections, but a different and more widely deployed engine.
    base_engines = {e.split(" (")[0] for e in baseline.engines}
    cand_engines = {e.split(" (")[0] for e in candidate.engines}
    new_engines = sorted(cand_engines - base_engines)
    # An engine that flagged the baseline and is simply ABSENT from the candidate's results has not
    # cleared it: it did not evaluate it. Older or sparser analyses do this routinely, and counting
    # it as an improvement is how a vendor that never looked turns into a vendor that passed us.
    # Only an engine that answered on the candidate, and answered without flagging, has cleared it.
    cleared = base_engines & candidate.responders - cand_engines
    silent = base_engines - candidate.responders
    gone_engines = sorted(cleared)
    if new_engines:
        delta.worse.append(
            f"engines that did not flag the baseline and now flag the candidate: "
            f"{', '.join(new_engines)}"
        )
    if gone_engines:
        delta.better.append(
            f"engines that flagged the baseline and no longer flag the candidate: "
            f"{', '.join(gone_engines)}"
        )
    if silent:
        # Not an improvement and not a regression: an unanswered question. Reported so the summary
        # cannot read as though the vendor that matters most had cleared us.
        delta.same.append(
            f"engines that flagged the baseline and returned no verdict at all on the candidate, "
            f"so they have NOT cleared it: {', '.join(sorted(silent))}"
        )
    if not new_engines and not gone_engines:
        delta.same.append(
            f"engine verdicts unchanged ({len(cand_engines) or 'none'}"
            f"{': ' + ', '.join(sorted(cand_engines)) if cand_engines else ''})"
        )

    # Sigma, per severity and in severity ORDER. Trading one high for three lows is an improvement
    # and a total would call it a regression; the reverse is a regression a total would call an
    # improvement. Reporting each bucket independently does not express that either: a trade moves
    # two buckets in opposite directions, so it lands in `worse` and in `better` at once, and
    # exit_code answers `worse`. SEVERITIES is ordered most severe first, so the most severe bucket
    # that moved is the one that decides; the rest are reported alongside it and do not flip it.
    movements = [
        (severity, baseline.sigma.get(severity, 0), candidate.sigma.get(severity, 0))
        for severity in SEVERITIES
        if baseline.sigma.get(severity, 0) != candidate.sigma.get(severity, 0)
    ]
    if movements:
        detail = ", ".join(f"{sev} {before} -> {after}" for sev, before, after in movements)
        decisive, before, after = movements[0]
        if after > before:
            delta.worse.append(
                f"Sigma {decisive} rose ({before} -> {after}), the most severe bucket that moved: "
                f"{detail}"
            )
        else:
            delta.better.append(
                f"Sigma {decisive} fell ({before} -> {after}), the most severe bucket that moved: "
                f"{detail}"
            )

    # Buckets this tool cannot place in the severity order, which is what a VirusTotal rename or
    # addition looks like on the day it happens. They are deliberately kept out of the ordered
    # trade-off above, because that logic depends on knowing which of two buckets is more severe.
    # A rise in one is reported as worse rather than assumed minor: an unrankable bucket could sit
    # anywhere, including above critical, and calling it an improvement is the one answer that
    # cannot be defended.
    unknown = sorted(
        (key, baseline.sigma.get(key, 0), candidate.sigma.get(key, 0))
        for key in set(baseline.sigma) | set(candidate.sigma)
        if key not in SEVERITIES and baseline.sigma.get(key, 0) != candidate.sigma.get(key, 0)
    )
    for key, before, after in unknown:
        if after > before:
            delta.worse.append(
                f"Sigma {key} rose ({before} -> {after}). This tool does not know where {key!r} "
                f"ranks against {', '.join(SEVERITIES)}, so it is not being called minor."
            )
        else:
            delta.better.append(f"Sigma {key} fell ({before} -> {after}), an unranked bucket")

    if baseline.sigma == candidate.sigma:
        delta.same.append(
            f"Sigma unchanged ({baseline.sigma_total} rules: "
            f"{', '.join(f'{k}={v}' for k, v in sorted(baseline.sigma.items())) or 'none'})"
        )

    base_yara, cand_yara = set(baseline.yara), set(candidate.yara)
    if cand_yara - base_yara:
        delta.worse.append(f"new YARA hits: {', '.join(sorted(cand_yara - base_yara))}")
    if base_yara - cand_yara:
        delta.better.append(f"YARA hits gone: {', '.join(sorted(base_yara - cand_yara))}")
    if base_yara == cand_yara:
        delta.same.append(f"YARA unchanged ({len(cand_yara)} hit(s))")

    return delta


def render(baseline: Snapshot, candidate: Snapshot, delta: Delta) -> str:
    lines = [
        "### VirusTotal delta against the reported baseline",
        "",
        # Only when the baseline really IS the recorded one. A dispatch can override the hash, and
        # printing the recorded identity beside an override's live numbers produced an artifact that
        # combined someone else's file with install.ps1's historical scores, which is a delta built
        # to be misattributed.
        (
            f"Baseline recorded as: {BASELINE_NOTE}"
            if baseline.sha256.lower() == BASELINE_SHA256.lower()
            else f"Baseline OVERRIDDEN to `{baseline.sha256[:16]}...`, so the recorded "
            f"install.ps1 scores do not apply and no historical comparison is implied."
        ),
        "",
        "| | baseline | candidate |",
        "|---|---|---|",
        f"| sha256 | `{baseline.sha256[:16]}...` | `{candidate.sha256[:16]}...` |",
        f"| on VirusTotal | {'yes' if baseline.found else 'NO'} | {'yes' if candidate.found else 'NO'} |",
        f"| engines flagging | {baseline.malicious + baseline.suspicious} / {baseline.total_engines} "
        f"| {candidate.malicious + candidate.suspicious} / {candidate.total_engines} |",
        f"| which | {_md_text(', '.join(baseline.engines)) or 'none'} "
        f"| {_md_text(', '.join(candidate.engines)) or 'none'} |",
        f"| Sigma | {baseline.sigma_total} ({', '.join(f'{k}={v}' for k, v in sorted(baseline.sigma.items())) or 'none'}) "
        f"| {candidate.sigma_total} ({', '.join(f'{k}={v}' for k, v in sorted(candidate.sigma.items())) or 'none'}) |",
        f"| YARA | {len(baseline.yara)} | {len(candidate.yara)} |",
        "",
    ]
    for heading, rows in (
        ("Worse", delta.worse),
        ("Better", delta.better),
        ("Unchanged", delta.same),
        ("Could not measure", delta.void),
    ):
        if rows:
            lines.append(f"**{heading}**")
            lines.append("")
            # Escaped here rather than at every append site. Engine names, detection labels and
            # YARA rule names all reach these bullets, all are third-party text, and this is
            # appended to $GITHUB_STEP_SUMMARY where a newline ends the bullet, `|` opens a cell and
            # `<` starts HTML that GitHub renders.
            lines.extend(f"- {_md_text(row)}" for row in rows)
            lines.append("")
    if not delta.void and not delta.worse and not delta.better:
        lines.append("Nothing moved. Reported as unchanged rather than dressed up.")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Offline self-test
# ---------------------------------------------------------------------------

# Recorded shapes, not live answers, so the comparison logic is testable with no key and no network.
# This is the half that can be wrong in a way nobody notices: a network call that fails is loud, and
# a comparison that silently never reports a regression is not.
_BASELINE_FIXTURE = {
    "data": {
        "attributes": {
            "size": 427113,
            "last_analysis_stats": {"malicious": 1, "undetected": 58},
            "last_analysis_results": {
                "Skyhigh": {"category": "malicious", "result": "BehavesLike.PS.Suspicious.gr"},
                "Microsoft": {"category": "undetected", "result": None},
            },
            "sigma_analysis_stats": {"high": 1, "medium": 11, "low": 5},
            "crowdsourced_yara_results": [
                {"rule_name": "SUSP_PS1_Shape_A"},
                {"rule_name": "SUSP_PS1_Shape_B"},
            ],
        }
    },
}


def self_test() -> list[str]:
    failures: list[str] = []
    baseline = snapshot_from_payload("baseline", "a" * 64, _BASELINE_FIXTURE)

    if baseline.sigma_total != 17 or baseline.sigma.get("high") != 1:
        failures.append(f"the Sigma parser no longer reads the recorded baseline: {baseline.sigma}")
    if baseline.engines != ["Skyhigh (BehavesLike.PS.Suspicious.gr)"]:
        failures.append(f"the engine parser no longer reads the baseline: {baseline.engines}")
    if len(baseline.yara) != 2:
        failures.append(f"the YARA parser no longer reads the baseline: {baseline.yara}")

    identical = snapshot_from_payload("candidate", "b" * 64, _BASELINE_FIXTURE)
    delta = compare(baseline, identical)
    if delta.worse or delta.better or delta.exit_code() != 0:
        failures.append(f"an identical candidate was not reported as unchanged: {delta}")

    import copy

    improved_payload = copy.deepcopy(_BASELINE_FIXTURE)
    improved_payload["data"]["attributes"]["sigma_analysis_stats"] = {"medium": 4, "low": 5}
    improved_payload["data"]["attributes"]["crowdsourced_yara_results"] = []
    improved = snapshot_from_payload("candidate", "c" * 64, improved_payload)
    delta = compare(baseline, improved)
    if not delta.better or delta.worse:
        failures.append(f"a clear improvement was not reported as better: {delta}")
    if delta.exit_code() != 0:
        failures.append("an improvement exited non-zero")

    worse_payload = copy.deepcopy(_BASELINE_FIXTURE)
    worse_payload["data"]["attributes"]["sigma_analysis_stats"] = {
        "high": 2,
        "medium": 11,
        "low": 5,
    }
    worse = snapshot_from_payload("candidate", "d" * 64, worse_payload)
    delta = compare(baseline, worse)
    if not delta.worse or delta.exit_code() != 2:
        failures.append(f"a new high-severity Sigma rule was not reported as worse: {delta}")

    swapped = copy.deepcopy(_BASELINE_FIXTURE)
    swapped["data"]["attributes"]["last_analysis_results"] = {
        "Microsoft": {"category": "malicious", "result": "Trojan:Script/Wacatac.B!ml"},
        "Skyhigh": {"category": "undetected", "result": None},
    }
    swap = snapshot_from_payload("candidate", "e" * 64, swapped)
    delta = compare(baseline, swap)
    # Same count, different engine, and vastly more consequential. A count-only comparison calls this
    # unchanged, which is the failure mode this check exists for.
    if not delta.worse:
        failures.append(
            "one engine's detection being replaced by another's was not reported as worse. The "
            "count is identical, so a comparison on counts alone would call the most consequential "
            "possible change 'no difference'."
        )

    missing = Snapshot(label = "candidate", sha256 = "f" * 64, note = "not present on VirusTotal")
    delta = compare(baseline, missing)
    if delta.exit_code() != 3 or not delta.void:
        failures.append("an unknown candidate hash was not reported as VOID")
    if delta.worse or delta.same:
        failures.append("an unknown candidate hash produced comparison rows it cannot support")

    unanalysed = snapshot_from_payload(
        "candidate",
        "0" * 64,
        {
            "data": {
                "attributes": {
                    "size": 1,
                    "last_analysis_stats": {},
                    "last_analysis_results": {},
                }
            }
        },
    )
    delta = compare(baseline, unanalysed)
    if delta.exit_code() != 3:
        failures.append(
            "a file that is known to VirusTotal but has never been analysed was not VOID. Zero "
            "verdicts of any kind is not every engine clearing it."
        )

    return failures


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--candidate", type = Path, help = "a file to hash and look up")
    parser.add_argument(
        "--candidate-sha256", default = "", help = "look up this hash instead of hashing a file"
    )
    parser.add_argument("--baseline-sha256", default = BASELINE_SHA256)
    parser.add_argument(
        "--summary",
        type = Path,
        default = None,
        help = "write the markdown report here as well as to stdout",
    )
    parser.add_argument(
        "--self-test", action = "store_true", help = "check the comparison logic offline and exit"
    )
    parser.add_argument(
        "--request-interval",
        type = float,
        default = 20.0,
        help = "seconds between API calls; the public tier allows 4 per minute",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        failures = self_test()
        for failure in failures:
            print(f"::error::self-test: {failure}")
        if failures:
            print(
                "::error::the delta tool's own checks failed, so no comparison it reports can be "
                "trusted. Refusing to compare."
            )
            return 1
        print("self-test: the comparison reports regressions, improvements and VOID correctly")
        return 0

    if not args.candidate and not args.candidate_sha256:
        print("::error::give --candidate or --candidate-sha256")
        return 1

    # Checked before the key, so a misconfigured run fails on the thing it can see.
    candidate_sha = args.candidate_sha256
    if args.candidate:
        if not args.candidate.is_file():
            print(f"::error::{args.candidate} does not exist")
            return 1
        candidate_sha = sha256_of(args.candidate)
        print(f"{args.candidate} -> {candidate_sha}")

    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        # VOID, and exit 3. A missing key is the most likely reason this ever produces no comparison,
        # and it must not be spelled the same as "nothing got worse".
        print(f"::warning::{API_KEY_ENV} is not set, so nothing was compared.")
        print("::warning::COULD NOT MEASURE. This is a missing key, not a clean result.")
        return 3

    client = VirusTotalClient(api_key, request_interval = args.request_interval)
    # One budget shared across both lookups, so a slow baseline cannot leave the candidate with the
    # whole remaining job timeout and still overrun it.
    deadline = time.monotonic() + LOOKUP_BUDGET_SECONDS
    baseline = fetch(client, args.baseline_sha256, "baseline", deadline = deadline)
    candidate = fetch(client, candidate_sha, "candidate", deadline = deadline)
    delta = compare(baseline, candidate)

    report = render(baseline, candidate, delta)
    print(report)
    if args.summary:
        args.summary.parent.mkdir(parents = True, exist_ok = True)
        args.summary.write_text(report, encoding = "utf-8")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding = "utf-8") as handle:
            handle.write(report + "\n")

    code = delta.exit_code()
    if code == 3:
        print("::error::VOID. Nothing was compared, so nothing was shown to have improved.")
    elif code == 2:
        print("::error::the candidate scores WORSE than the baseline on at least one axis.")
    elif delta.better:
        print(
            "the candidate scores better than the baseline. Sigma and YARA move immediately; a "
            "cloud engine's behavioural verdict may not, and that is reported as unchanged rather "
            "than dressed up."
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
