# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Turn collected Kaggle evidence into a job summary and an exit code.

This is the only place that decides whether the workflow goes red, and it
holds a deliberately narrow line: red means the payload RAN on a T4 and
disagreed with its assertions. Everything else is a warning.

That distinction matters more than it looks. Kaggle is a free external
service with a hard concurrency cap, a weekly quota and its own queue, and
every one of those can stop the test from producing a result. If any of them
turned a pull request red, the check would be noise within a week and would
be ignored the one time it was right.

Exit codes:
    0  passed, partially reported, or never ran
    1  a payload ran and failed its assertions
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _summary(text: str) -> None:
    print(text, flush=True)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def _notice(level: str, title: str, message: str) -> None:
    flat = message.replace("\n", " ").replace("::", ":")
    print(f"::{level} title={title}::{flat}", flush=True)


def _fmt_metric(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        # NaN is a real, reproducible fp16 gradient-scaler outcome, not a
        # missing value, so name it rather than blanking it.
        if value != value:
            return "NaN (step skipped)"
        return f"{value:.6g}"
    return str(value)


def render(report: dict) -> list[str]:
    lines = [f"#### payload `{report.get('label', '?')}`  "
             f"model `{report.get('model', '?')}`", ""]
    env = report.get("environment", {})
    if env:
        lines.append(
            f"GPU `{env.get('gpu_name', '?')}` ({env.get('gpu_capability', '?')}, "
            f"{env.get('gpu_total_gb', '?')} GB) - torch `{env.get('torch', '?')}` "
            f"- transformers `{env.get('transformers', '?')}` "
            f"- trl `{env.get('trl', '?')}` - unsloth `{env.get('unsloth', '?')}`")
        lines.append("")

    lines += ["| step | loss | grad_norm |", "| --- | --- | --- |"]
    for entry in report.get("metrics", []):
        lines.append(f"| {entry.get('step')} | {_fmt_metric(entry.get('loss'))} "
                     f"| {_fmt_metric(entry.get('grad_norm'))} |")
    lines.append("")

    repro = report.get("reproducibility")
    if repro:
        if repro.get("identical"):
            lines.append("Reproducibility: two fresh processes agreed "
                         "**bitwise** on every step.")
        else:
            lines.append(
                f"Reproducibility: **DIFFERED** from step "
                f"{repro.get('first_diff_step')} "
                f"(max abs {repro.get('max_abs_diff')}).")
        lines.append("")

    for run in report.get("runs", []):
        lines.append(f"Cycle {run.get('run_index')} generated: "
                     f"`{run.get('generated', '')}` - canary "
                     f"{'found' if run.get('canary_found') else '**MISSING**'}")
    lines.append("")

    ref = report.get("reference_check")
    if ref:
        status = ref.get("status")
        if status == "ok":
            lines.append(f"Reference band: within tolerance "
                         f"(worst relative deviation {ref.get('worst_rel')}).")
        elif status == "absent":
            lines.append("Reference band: no committed reference for this "
                         "configuration, so nothing was compared.")
        else:
            lines.append(f"Reference band: **{status}** - {ref.get('deviations')}")
        lines.append("")

    if report.get("failures"):
        lines.append("Failures:")
        lines += [f"- {f}" for f in report["failures"]]
        lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--expect", type=int, default=2)
    args = ap.parse_args()

    evidence = Path(args.evidence)
    result_file = evidence / "launch_result.json"
    if not result_file.exists():
        _summary("### Kaggle T4 smoke\n\nNo launch result was written. The "
                 "launcher did not get far enough to record anything, so "
                 "nothing is known about the code under test.")
        _notice("warning", "Kaggle T4 smoke did not run",
                "no launch_result.json was produced")
        return 0

    result = json.loads(result_file.read_text(encoding="utf-8"))
    verdict = result.get("verdict", "infra")
    reason = result.get("reason", "")
    reports = result.get("reports", [])

    header = {
        "pass": "### Kaggle T4 smoke: PASS",
        "fail": "### Kaggle T4 smoke: FAIL",
        "partial": "### Kaggle T4 smoke: PARTIAL",
        "infra": "### Kaggle T4 smoke: NOT RUN",
    }.get(verdict, "### Kaggle T4 smoke")

    lines = [header, "", reason, ""]
    if result.get("slug"):
        lines.append(f"Kernel: `{result['slug']}` (private), terminal state "
                     f"`{result.get('kernel_state')}`.")
        lines.append("")
    for report in reports:
        lines += render(report)

    if verdict == "infra":
        lines += [
            "This is not a code failure. The test never produced a result, "
            "so there is nothing to conclude about this change. Common "
            "causes: the Kaggle account was at its 2-kernel concurrency cap, "
            "the weekly GPU quota was exhausted, or the push was throttled.",
            "",
            "Re-run with the `kaggle-t4-ci` label or a manual dispatch to "
            "force another attempt.",
        ]

    _summary("\n".join(lines))

    if verdict == "fail":
        _notice("error", "Kaggle T4 smoke failed", reason)
        return 1
    if verdict == "partial":
        _notice("warning", "Kaggle T4 smoke partially reported", reason)
        return 0
    if verdict == "infra":
        _notice("warning", "Kaggle T4 smoke did not run", reason)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
