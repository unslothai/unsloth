# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Turn the Unsloth payload's report into a job summary and an exit code.

Sibling of ``.github/scripts/kaggle_t4_ci/report.py``, which holds the same
line and is reused wholesale for everything that is not rendering: the
launcher's verdict vocabulary, the kernel-log flattening and the
"infra is not a failure" policy all come from there and are imported, not
copied. What is local is the rendering, because that file renders a training
trace -- a loss table, a canary, a reference band -- and this payload
produces a list of assertions about a server.

The line itself is unchanged and is worth restating: **red means the payload
ran on a GPU and disagreed with its assertions.** Kaggle being busy, out of
quota, or unreachable teaches nothing about the code and must never colour a
pull request.

Exit codes:
    0  passed, partially reported, or never ran
    1  the payload ran and failed an assertion
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[1] / "kaggle_t4_ci" / "report.py"


def _load_shared():
    """The notebook leg's reporter, imported by path rather than duplicated.

    Degrades instead of exploding: this file is owned elsewhere and is under
    active change, and the only thing borrowed from it is a log-flattening
    helper. Losing that costs a diagnostic section, not the verdict.
    """
    try:
        spec = importlib.util.spec_from_file_location("kaggle_t4_ci_report", _SHARED)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001
        return None


def _summary(text: str) -> None:
    print(text, flush = True)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding = "utf-8") as fh:
            fh.write(text + "\n")


def _notice(level: str, title: str, message: str) -> None:
    flat = message.replace("\n", " ").replace("::", ":")
    print(f"::{level} title={title}::{flat}", flush = True)


# Order the assertions are presented in, and the one-line reminder of what
# each is actually worth. A reader who has never seen this job before should
# not have to open the payload to know whether a tick means anything.
ASSERTION_BLURB = {
    "preflight": "a GPU is present and there is disk to use it",
    "studio_ready": "Unsloth answered /api/health as healthy, hardware detection settled",
    "authenticate": "the bootstrap credential worked",
    "gpu_inference": "the GGUF was on the GPU, not on a CPU fallback that returns text anyway",
    "tool_calling": "the model emitted a real tool call, not prose",
    "lora_training": "a training run completed AND left an adapter on disk",
    "gguf_export": "export ran against a CUDA llama.cpp and the file it wrote loads",
    "chat_ui_driver": "tests/studio/playwright_chat_ui.py passed against this server",
}


def render(report: dict) -> list[str]:
    env = report.get("environment", {})
    config = report.get("config", {})

    lines = [
        f"#### payload `{report.get('label', '?')}` - {report.get('seconds', '?')}s",
        "",
    ]
    gpus = env.get("gpus") or []
    lines.append(
        f"GPU `{env.get('gpu_name') or (gpus[0] if gpus else '?')}` "
        f"(capability `{env.get('gpu_capability', '?')}`, {env.get('gpu_count', '?')} visible) "
        f"- torch `{env.get('torch', '?')}` (cuda `{env.get('cuda', '?')}`) "
        f"- llama.cpp install kind `{env.get('llama_cpp_install_kind')}`"
    )
    lines.append("")
    lines.append(
        f"Chat model `{config.get('chat_model')}` `{config.get('chat_variant')}` "
        f"- train model `{config.get('train_model')}` at `{config.get('max_steps')}` steps "
        f"- export `{config.get('quantization')}` - gpu_layers pin `{config.get('gpu_layers')}`"
    )
    lines.append("")

    lines += ["| assertion | verdict | what it is worth |", "| --- | --- | --- |"]
    for entry in report.get("assertions", []):
        name = entry.get("name", "?")
        verdict = "pass" if entry.get("passed") else "**FAIL**"
        lines.append(f"| `{name}` | {verdict} | {ASSERTION_BLURB.get(name, '')} |")
    lines.append("")

    for entry in report.get("assertions", []):
        if entry.get("name") != "gpu_inference":
            continue
        evidence = entry.get("evidence") or []
        if evidence:
            lines.append("GPU offload evidence:")
            lines += [f"- {item}" for item in evidence]
            lines.append("")

    for entry in report.get("assertions", []):
        if entry.get("name") == "lora_training" and entry.get("output_dir"):
            lines.append(
                f"Training: phase `{entry.get('phase')}`, "
                f"{entry.get('steps_with_loss', '?')} step(s) with a logged loss, adapter "
                f"`{entry.get('adapter_weights', 'missing')}` "
                f"({entry.get('adapter_bytes', 0)} bytes)."
            )
            lines.append("")
        if entry.get("name") == "gguf_export" and entry.get("gguf"):
            lines.append(
                f"Export: `{Path(entry['gguf']).name}` ({entry.get('gguf_bytes', 0)} bytes), "
                f"reloaded on the GPU and generated "
                f"{'the canary' if entry.get('canary_found') else 'text'}."
            )
            lines.append("")

    if report.get("failures"):
        lines.append("Failures:")
        lines += [f"- {item}" for item in report["failures"]]
        lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required = True)
    args = ap.parse_args()

    evidence = Path(args.evidence)
    result_file = evidence / "launch_result.json"
    if not result_file.exists():
        _summary(
            "### Unsloth GPU smoke\n\nNo launch result was written. The launcher did not "
            "get far enough to record anything, so nothing is known about the code "
            "under test."
        )
        _notice("warning", "Unsloth GPU smoke did not run", "no launch_result.json was produced")
        return 0

    result = json.loads(result_file.read_text(encoding = "utf-8"))
    verdict = result.get("verdict", "infra")
    reason = result.get("reason", "")
    reports = result.get("reports", [])

    header = {
        "pass": "### Unsloth GPU smoke: PASS",
        "fail": "### Unsloth GPU smoke: FAIL",
        "partial": "### Unsloth GPU smoke: PARTIAL",
        "infra": "### Unsloth GPU smoke: NOT RUN",
    }.get(verdict, "### Unsloth GPU smoke")

    lines = [header, "", reason, ""]
    if result.get("slug"):
        lines.append(
            f"Kernel: `{result['slug']}` (private), terminal state `{result.get('kernel_state')}`."
        )
        lines.append("")
    for report in reports:
        lines += render(report)

    if verdict in ("infra", "partial"):
        shared = _load_shared()
        hits = []
        if shared is not None and hasattr(shared, "diagnostic_lines"):
            try:
                hits = shared.diagnostic_lines(evidence)
            except Exception:  # noqa: BLE001
                hits = []
        if hits:
            lines += (
                [
                    "<details><summary>Kernel log, filtered</summary>",
                    "",
                    "```",
                ]
                + hits
                + ["```", "", "</details>", ""]
            )

    if verdict == "infra":
        lines += [
            "This is not a code failure. The payload never produced a result, so there "
            "is nothing to conclude about this change. Common causes: the Kaggle "
            "account was at its 2-kernel concurrency cap, the weekly GPU quota was "
            "exhausted, or the push was throttled.",
            "",
            "Re-run with the `kaggle-studio-gpu-ci` label or a manual dispatch to force "
            "another attempt.",
        ]

    _summary("\n".join(lines))

    if verdict == "fail":
        _notice("error", "Unsloth GPU smoke failed", reason)
        return 1
    if verdict == "partial":
        _notice("warning", "Unsloth GPU smoke partially reported", reason)
        return 0
    if verdict == "infra":
        _notice("warning", "Unsloth GPU smoke did not run", reason)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
