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
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from virustotal_scan import (  # noqa: E402
    API_KEY_ENV,
    API_ROOT,
    VirusTotalClient,
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


def parse_sigma(raw: object) -> dict[str, int]:
    """Sigma counts by severity.

    VirusTotal has renamed and added buckets over time, so every key is read defensively rather than
    indexed. A KeyError here would turn a measurement into a crash on a day VirusTotal shipped a
    schema change, which is exactly when the measurement matters.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for severity in SEVERITIES:
        value = raw.get(severity, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if int(value):
            out[severity] = int(value)
    return out


def parse_yara(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    names = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("rule_name") or entry.get("ruleset_name")
        if isinstance(name, str) and name:
            names.append(name)
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
    snap.total_engines = stats.total
    snap.engines = parse_detections(attributes.get("last_analysis_results"))
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


def fetch(client: VirusTotalClient, sha256: str, label: str) -> Snapshot:
    status, payload = client.request(
        "GET", f"{API_ROOT}/files/{sha256}", allow_status = (404,),
    )
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
    if delta.void:
        return delta

    # Engines, by name. A count alone hides the case that matters most: the same number of
    # detections, but a different and more widely deployed engine.
    base_engines = {e.split(" (")[0] for e in baseline.engines}
    cand_engines = {e.split(" (")[0] for e in candidate.engines}
    new_engines = sorted(cand_engines - base_engines)
    gone_engines = sorted(base_engines - cand_engines)
    if new_engines:
        delta.worse.append(f"engines that did not flag the baseline and now flag the candidate: "
                           f"{', '.join(new_engines)}")
    if gone_engines:
        delta.better.append(f"engines that flagged the baseline and no longer flag the candidate: "
                            f"{', '.join(gone_engines)}")
    if not new_engines and not gone_engines:
        delta.same.append(
            f"engine verdicts unchanged ({len(cand_engines) or 'none'}"
            f"{': ' + ', '.join(sorted(cand_engines)) if cand_engines else ''})"
        )

    # Sigma, per severity rather than in total. Trading one high for three lows is an improvement and
    # a total would call it a regression; the reverse is a regression a total would call an
    # improvement.
    for severity in SEVERITIES:
        before = baseline.sigma.get(severity, 0)
        after = candidate.sigma.get(severity, 0)
        if after > before:
            delta.worse.append(f"Sigma {severity}: {before} -> {after}")
        elif after < before:
            delta.better.append(f"Sigma {severity}: {before} -> {after}")
    if baseline.sigma == candidate.sigma:
        delta.same.append(f"Sigma unchanged ({baseline.sigma_total} rules: "
                          f"{', '.join(f'{k}={v}' for k, v in sorted(baseline.sigma.items())) or 'none'})")

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
        f"Baseline recorded as: {BASELINE_NOTE}",
        "",
        "| | baseline | candidate |",
        "|---|---|---|",
        f"| sha256 | `{baseline.sha256[:16]}...` | `{candidate.sha256[:16]}...` |",
        f"| on VirusTotal | {'yes' if baseline.found else 'NO'} | {'yes' if candidate.found else 'NO'} |",
        f"| engines flagging | {baseline.malicious + baseline.suspicious} / {baseline.total_engines} "
        f"| {candidate.malicious + candidate.suspicious} / {candidate.total_engines} |",
        f"| which | {', '.join(baseline.engines) or 'none'} | {', '.join(candidate.engines) or 'none'} |",
        f"| Sigma | {baseline.sigma_total} ({', '.join(f'{k}={v}' for k, v in sorted(baseline.sigma.items())) or 'none'}) "
        f"| {candidate.sigma_total} ({', '.join(f'{k}={v}' for k, v in sorted(candidate.sigma.items())) or 'none'}) |",
        f"| YARA | {len(baseline.yara)} | {len(candidate.yara)} |",
        "",
    ]
    for heading, rows in (("Worse", delta.worse), ("Better", delta.better),
                          ("Unchanged", delta.same), ("Could not measure", delta.void)):
        if rows:
            lines.append(f"**{heading}**")
            lines.append("")
            lines.extend(f"- {row}" for row in rows)
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
    "data": {"attributes": {
        "size": 427113,
        "last_analysis_stats": {"malicious": 1, "undetected": 58},
        "last_analysis_results": {
            "Skyhigh": {"category": "malicious", "result": "BehavesLike.PS.Suspicious.gr"},
            "Microsoft": {"category": "undetected", "result": None},
        },
        "sigma_analysis_stats": {"high": 1, "medium": 11, "low": 5},
        "crowdsourced_yara_results": [
            {"rule_name": "SUSP_PS1_Shape_A"}, {"rule_name": "SUSP_PS1_Shape_B"},
        ],
    }},
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
    worse_payload["data"]["attributes"]["sigma_analysis_stats"] = {"high": 2, "medium": 11, "low": 5}
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

    unanalysed = snapshot_from_payload("candidate", "0" * 64, {"data": {"attributes": {
        "size": 1, "last_analysis_stats": {}, "last_analysis_results": {},
    }}})
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
    parser.add_argument("--candidate", type = Path,
                        help = "a file to hash and look up")
    parser.add_argument("--candidate-sha256", default = "",
                        help = "look up this hash instead of hashing a file")
    parser.add_argument("--baseline-sha256", default = BASELINE_SHA256)
    parser.add_argument("--summary", type = Path, default = None,
                        help = "write the markdown report here as well as to stdout")
    parser.add_argument("--self-test", action = "store_true",
                        help = "check the comparison logic offline and exit")
    parser.add_argument("--request-interval", type = float, default = 20.0,
                        help = "seconds between API calls; the public tier allows 4 per minute")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        failures = self_test()
        for failure in failures:
            print(f"::error::self-test: {failure}")
        if failures:
            print("::error::the delta tool's own checks failed, so no comparison it reports can be "
                  "trusted. Refusing to compare.")
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
    baseline = fetch(client, args.baseline_sha256, "baseline")
    candidate = fetch(client, candidate_sha, "candidate")
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
        print("the candidate scores better than the baseline. Sigma and YARA move immediately; a "
              "cloud engine's behavioural verdict may not, and that is reported as unchanged rather "
              "than dressed up.")
    return code


if __name__ == "__main__":
    sys.exit(main())
