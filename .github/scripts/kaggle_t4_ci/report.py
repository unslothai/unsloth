# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Turn collected Kaggle evidence into a job summary and an exit code.

The only place that decides whether the workflow goes red, on a deliberately
narrow line: red means the payload RAN on a T4 and disagreed with its
assertions. Everything else is a warning, because Kaggle is a free service with
a hard concurrency cap, a weekly quota and its own queue, any of which can stop
the test from producing a result; if those turned a PR red the check would be
noise within a week and ignored the one time it was right.

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
    print(text, flush = True)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding = "utf-8") as fh:
            fh.write(text + "\n")


def _notice(level: str, title: str, message: str) -> None:
    flat = message.replace("\n", " ").replace("::", ":")
    print(f"::{level} title={title}::{flat}", flush = True)


def _fmt_metric(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        # NaN is a real fp16 gradient-scaler outcome, not a missing value.
        if value != value:
            return "NaN (step skipped)"
        return f"{value:.6g}"
    return str(value)


def resolved_versions(report: dict) -> dict:
    """The installed version of every watched package, whichever leg wrote it.

    The SFT payload nests it under ``environment.resolved`` because its
    environment block predates this and the committed reference carries that
    shape; gpt-oss and GRPO write ``versions_flat`` at the top level.
    """
    flat = report.get("versions_flat")
    if isinstance(flat, dict):
        return flat
    nested = (report.get("environment") or {}).get("resolved")
    return nested if isinstance(nested, dict) else {}


def version_table(reports: list) -> list[str]:
    """Every leg's library set side by side, with what differs called out.

    The payoff of a pinned control beside a canary: when the canary is the only
    red leg, "which bump did it" is answered on the summary page, without
    downloading an artifact or reconstructing an install log.
    """
    columns = [(r.get("label", "?"), resolved_versions(r)) for r in reports]
    columns = [(label, versions) for label, versions in columns if versions]
    if len(columns) < 2:
        return []
    packages = sorted({p for _, versions in columns for p in versions})
    differing = [p for p in packages if len({str(versions.get(p)) for _, versions in columns}) > 1]
    lines = [
        "<details><summary>Resolved library versions per leg"
        + (f" ({len(differing)} differ)" if differing else " (identical across legs)")
        + "</summary>",
        "",
    ]
    lines.append("| package | " + " | ".join(l for l, _ in columns) + " |")
    lines.append("| --- |" + " --- |" * len(columns))
    for package in packages:
        cells = [str(versions.get(package) or "-") for _, versions in columns]
        name = f"**{package}**" if package in differing else package
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "</details>", ""]
    if differing:
        lines += [f"Legs differ in: {', '.join(differing)}.", ""]
    return lines


# The label the Studio payload reports under. Duplicated in
# kaggle_studio_ci/report.py rather than shared: these two packages have no
# import relationship and both already ship a module called `report`, so a
# shared helper would mean putting one of them on the other's sys.path -- which
# is how `import report` starts resolving to the wrong file.
STUDIO_LABEL = "studio-gpu"


def own_verdict(kernel_verdict: str, kernel_reason: str, reports: list, expect: int):
    """This reporter's verdict over ITS OWN payloads, not the kernel's.

    The launcher writes one verdict for the whole kernel, and since
    --with-studio that kernel holds two unrelated experiments. Reading the
    kernel verdict here means a failing training leg prints "Kaggle T4 smoke: FAIL"
    above a section listing zero failures, and a failing Studio payload prints
    the same over four green legs. Both are the misleading-red twin of the
    green tick that tested nothing, and both would send someone to read the
    wrong payload.

    So the verdict is recomputed from the filtered reports. The kernel reason
    is kept only when the two agree; otherwise it describes the other half.

    `infra` is deliberately not synthesised: with nothing of ours back, the
    kernel-level reason (quota, concurrency cap, a push that was throttled) is
    the only account of why, and it applies to every payload equally.
    """
    if not reports:
        return (kernel_verdict if kernel_verdict == "infra" else "partial"), kernel_reason
    failing = [r for r in reports if not r.get("passed")]
    if failing:
        return "fail", f"{len(failing)} of {len(reports)} payload(s) failed their assertions"
    if len(reports) < expect:
        return "partial", f"only {len(reports)} of {expect} payload(s) reported back"
    return "pass", f"all {len(reports)} payload(s) passed"


def render(report: dict) -> list[str]:
    lines = [
        f"#### payload `{report.get('label', '?')}`  model `{report.get('model', '?')}`",
        "",
    ]
    if report.get("probe"):
        lines += [
            "This payload ran in **probe mode**: everything is "
            "recorded and nothing is asserted, so `passed` says only "
            "that it reported back. Read `observed_failures`.",
            "",
        ]
    env = report.get("environment", {})
    if env:
        lines.append(
            f"GPU `{env.get('gpu_name', '?')}` ({env.get('gpu_capability', '?')}, "
            f"{env.get('gpu_total_gb', '?')} GB) - torch `{env.get('torch', '?')}` "
            f"- transformers `{env.get('transformers', '?')}` "
            f"- trl `{env.get('trl', '?')}` - unsloth `{env.get('unsloth', '?')}`"
        )
        lines.append("")

    config = report.get("config", {})
    if config:
        # max_steps is up front: it decides whether the committed reference applies to this run at all.
        lines.append(
            f"Config: max_steps `{config.get('max_steps')}` - lr "
            f"`{config.get('learning_rate')}` - batch "
            f"`{config.get('batch_size')}` - init_loss_scale "
            f"`{config.get('init_loss_scale')}`"
        )
        scales = [r.get("loss_scale") for r in report.get("runs", []) if r.get("loss_scale")]
        if scales and not all(s.get("applied") for s in scales):
            lines.append(
                f"fp16 loss-scale pin did NOT apply: "
                f"`{scales[0].get('reason', 'unknown')}`. The run used the "
                f"framework default, so its first steps were spent on scaler "
                f"overflows."
            )
        lines.append("")

    lines += ["| step | loss | grad_norm |", "| --- | --- | --- |"]
    for entry in report.get("metrics", []):
        lines.append(
            f"| {entry.get('step')} | {_fmt_metric(entry.get('loss'))} "
            f"| {_fmt_metric(entry.get('grad_norm'))} |"
        )
    lines.append("")

    repro = report.get("reproducibility")
    if repro:
        if repro.get("identical"):
            lines.append("Reproducibility: two fresh processes agreed **bitwise** on every step.")
        else:
            lines.append(
                f"Reproducibility: **DIFFERED** from step "
                f"{repro.get('first_diff_step')} "
                f"(max abs {repro.get('max_abs_diff')})."
            )
        lines.append("")

    for run in report.get("runs", []):
        lines.append(
            f"Cycle {run.get('run_index')} generated: "
            f"`{run.get('generated', '')}` - canary "
            f"{'found' if run.get('canary_found') else '**MISSING**'}"
        )
    lines.append("")

    ref = report.get("reference_check")
    if ref:
        status = ref.get("status")
        if status == "ok":
            lines.append(
                f"Reference band: within tolerance (worst relative deviation "
                f"{ref.get('worst_rel')}), against a reference captured at "
                f"max_steps={ref.get('reference_max_steps')}."
            )
        elif status == "absent":
            lines.append(
                "Reference band: no committed reference for this "
                "configuration, so nothing was compared."
            )
        elif ref.get("note"):
            # A refusal, not a deviation: the deviations list is empty for
            # these, so printing it alone would read like a clean result.
            lines.append(f"Reference band: **{status}** - {ref['note']}")
        else:
            lines.append(f"Reference band: **{status}** - {ref.get('deviations')}")
        # What the band did NOT compare, up front rather than buried in the evidence.
        unchecked = ref.get("config_unchecked")
        if unchecked:
            lines.append(
                "Not compared, absent from one side: "
                + ", ".join(f"`{key}`" for key in unchecked)
                + ". Recapture the reference (references/README.md) to bring "
                "them into the check."
            )
        lines.append("")

    pins = report.get("pins")
    if pins:
        lines.append(
            "Pins: "
            + (
                "held"
                if not pins.get("failures")
                else "**DID NOT HOLD** - " + "; ".join(pins["failures"])
            )
            + f" ({len(pins.get('requested', {}))} pinned)."
        )
        lines.append("")

    compiled = report.get("compile")
    if compiled:
        if not compiled.get("available"):
            lines.append(f"torch.compile: **unreadable** - {compiled.get('error')}")
        else:
            lines.append(
                f"torch.compile: {compiled.get('unique_graphs')} unique "
                f"graph(s), {compiled.get('calls_captured')} calls captured, "
                f"{compiled.get('graph_breaks_total')} graph break(s)."
                + (
                    ""
                    if compiled.get("unique_graphs")
                    else " **Zero graphs means the run was entirely eager.**"
                )
            )
        lines.append("")

    # The whole point of the instrumentation: a reader answers "is the Hub
    # download worth optimising" from the job summary, without downloading the
    # evidence artifact. `fetch_seconds` is None when the timer never attached,
    # and that is rendered as its own sentence rather than as a zero -- "no
    # download happened" and "nothing was measured" are different findings.
    phases = report.get("load_phases")
    if phases:
        if phases.get("fetch_seconds") is None:
            lines.append(
                f"Load {phases.get('total_seconds')}s, split unknown: the fetch "
                f"timer never attached, so this run says nothing about it."
            )
        else:
            rate = phases.get("fetch_mb_s")
            lines.append(
                f"Load {phases.get('total_seconds')}s = fetch "
                f"{phases.get('fetch_seconds')}s "
                f"({phases.get('fetch_mb')} MB"
                + (f" at {rate} MB/s" if rate else "")
                + f") + weight load {phases.get('weight_load_seconds')}s."
            )
        lines.append("")

    memory = report.get("memory_peak") or report.get("memory_after_train")
    if memory:
        lines.append(
            f"Peak VRAM: {memory.get('peak_reserved_gb')} GB reserved / "
            f"{memory.get('peak_allocated_gb')} GB allocated of "
            f"{memory.get('total_gb')} GB."
        )
        lines.append("")

    history = report.get("log_history")
    if history:
        # GRPO: loss is ~0 by construction at num_iterations=1 and beta=0, so reward and reward_std are what is worth
        # showing.
        lines += ["| step | reward | reward_std |", "| --- | --- | --- |"]
        for entry in history:
            if entry.get("reward") is None:
                continue
            lines.append(
                f"| {entry.get('step')} | "
                f"{_fmt_metric(entry.get('reward'))} | "
                f"{_fmt_metric(entry.get('reward_std'))} |"
            )
        lines.append("")
        sample = [t for group in (report.get("completions") or []) for t in group if t.strip()]
        if sample:
            lines.append(f"First completion: `{sample[0][:200]}`")
            lines.append("")

    if report.get("observed_failures") and not report.get("failures"):
        lines.append("Observed (not asserted, this is a probe):")
        lines += [f"- {f}" for f in report["observed_failures"]]
        lines.append("")

    if report.get("failures"):
        lines.append("Failures:")
        lines += [f"- {f}" for f in report["failures"]]
        lines.append("")
    return lines


SENTINELS = (
    "KAGGLE_T4_CI_DRIVER",
    "KAGGLE_T4_CI_PAYLOAD",
    "Error",
    "error:",
    "Traceback",
    "SystemExit",
    "papermill.exceptions",
)


def kernel_log_text(evidence: Path) -> str:
    """The kernel log as flat text, whichever shape Kaggle returned it in.

    Kaggle's `kernels/output` returns the log as a JSON array of
    ``{stream_name, time, data}`` records, not as text, so reading the file
    directly shows a wall of JSON with one word of message per line.
    """
    chunks = []
    # rglob: a run is several kernels, each collecting into its own directory, so there is no single kernel.log any
    # more.
    for path in sorted(evidence.rglob("kernel.log")):
        raw = path.read_text(encoding = "utf-8", errors = "replace")
        try:
            records = json.loads(raw)
        except json.JSONDecodeError:
            chunks.append(raw)
            continue
        if not isinstance(records, list):
            chunks.append(raw)
            continue
        chunks.append("".join(r.get("data", "") for r in records if isinstance(r, dict)))
    return "".join(chunks)


PREFETCH_SENTINEL = "KAGGLE_CI_PREFETCH"


def prefetch_table(evidence: Path) -> list[str]:
    """What the prefetch lane actually achieved, from the kernel log.

    This is the instrument, not decoration. The gpt-oss download time was never
    measured -- an earlier estimate of "~282s" was subtraction, not measurement
    -- and the whole leg order is arranged around it. Putting the number in the
    job summary is what lets the next person confirm or reject the reorder
    without downloading an artifact, including the case where it did not pay
    for itself.

    Absent on a kernel built without the lane, which reads as no section at
    all rather than as a table of zeroes.
    """
    text = kernel_log_text(evidence)
    if not text:
        return []
    records = []
    for line in text.splitlines():
        marker = PREFETCH_SENTINEL + " "
        if marker in line:
            try:
                records.append(json.loads(line.split(marker, 1)[1]))
            except (ValueError, IndexError):
                continue
    if not records:
        return []
    lines = [
        "#### model prefetch",
        "",
        "| repo | ok | download s | MB/s | GB | transport | attempts |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        gb = round((r.get("bytes") or 0) / 1e9, 2)
        lines.append(
            f"| `{r.get('repo', '?')}` | {'yes' if r.get('ok') else '**NO**'} | "
            f"{r.get('download_seconds') if r.get('download_seconds') is not None else '-'} | "
            f"{r.get('mb_per_s') if r.get('mb_per_s') is not None else '-'} | {gb} | "
            f"{r.get('transport', '?')} | {r.get('attempts', '?')} |"
        )
    failed = [r for r in records if not r.get("ok")]
    lines.append("")
    if failed:
        # Not a failure of the run. Said out loud anyway, because the schedule
        # assumes this lane worked: legs.KERNELS starts gptoss third to give it
        # a window, and if the window went unused the makespan is the ~568s
        # fallback rather than the ~500s the order was chosen for.
        lines.append(
            f"{len(failed)} of {len(records)} prefetch(es) failed. This does not fail "
            "the run -- the leg downloads the model itself -- but the leg order in "
            "`legs.KERNELS` is arranged around this lane working, so the makespan "
            "above is the fallback rather than the intended one."
        )
        lines.append("")
    return lines


def diagnostic_lines(evidence: Path, limit: int = 40) -> list[str]:
    """The lines of the kernel log worth putting in front of a human.

    A kernel that finished but reported nothing is the hardest outcome to read:
    no metrics to show, cause buried in an artifact nobody downloads. Both real
    instances so far, a dependency probe that mis-ordered its imports and a
    generated cell with a syntax error, were one grep away in this log.
    """
    text = kernel_log_text(evidence)
    if not text:
        return []
    hits = [line.rstrip() for line in text.splitlines() if any(s in line for s in SENTINELS)]
    return hits[-limit:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required = True)
    ap.add_argument("--expect", type = int, default = 2)
    args = ap.parse_args()

    evidence = Path(args.evidence)
    result_file = evidence / "launch_result.json"
    if not result_file.exists():
        _summary(
            "### Kaggle T4 smoke\n\nNo launch result was written. The "
            "launcher did not get far enough to record anything, so "
            "nothing is known about the code under test."
        )
        _notice("warning", "Kaggle T4 smoke did not run", "no launch_result.json was produced")
        return 0

    result = json.loads(result_file.read_text(encoding = "utf-8"))
    verdict = result.get("verdict", "infra")
    reason = result.get("reason", "")
    reports = result.get("reports", [])

    # The Studio payload can share this kernel (see kaggle_t4_ci/build_kernel.py
    # --with-studio), and it emits its report through the same prefix, so it
    # arrives in this list. It is a different SHAPE -- assertions rather than a
    # per-step metric trace, no `config`, no `model` -- so rendering it here
    # produces a training leg made of question marks. kaggle_studio_ci/report.py
    # renders it properly; each reporter owns its own labels.
    reports = [r for r in reports if r.get("label") != STUDIO_LABEL]
    verdict, reason = own_verdict(verdict, reason, reports, args.expect)

    header = {
        "pass": "### Kaggle T4 smoke: PASS",
        "fail": "### Kaggle T4 smoke: FAIL",
        "partial": "### Kaggle T4 smoke: PARTIAL",
        "infra": "### Kaggle T4 smoke: NOT RUN",
    }.get(verdict, "### Kaggle T4 smoke")

    lines = [header, "", reason, ""]
    for kernel in result.get("kernels") or []:
        if kernel.get("slug"):
            lines.append(
                f"Kernel: `{kernel['slug']}` (private), terminal state `{kernel.get('state')}`."
            )
        else:
            lines.append(
                f"Kernel from `{kernel.get('notebook')}` was never "
                f"pushed: {kernel.get('push_error')}"
            )
    if not result.get("kernels") and result.get("slug"):
        lines.append(
            f"Kernel: `{result['slug']}` (private), terminal state `{result.get('kernel_state')}`."
        )
    lines.append("")

    lines += prefetch_table(evidence)
    lines += version_table(reports)
    for report in reports:
        lines += render(report)

    if verdict in ("infra", "partial") and len(reports) < args.expect:
        hits = diagnostic_lines(evidence)
        if hits:
            lines += (
                ["<details><summary>Kernel log, filtered</summary>", "", "```"]
                + hits
                + ["```", "", "</details>", ""]
            )

    if verdict == "infra":
        lines += [
            "This is not a code failure. The test never produced a result, "
            "so there is nothing to conclude about this change. Common "
            "causes: the Kaggle account was at its 2-kernel concurrency cap, "
            "the weekly GPU quota was exhausted, or the push was throttled.",
            "",
            "Re-run with the `kaggle-t4-ci` label or a manual dispatch to force another attempt.",
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
