# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compare local pre-contract and working-tree TestClient p50/p95; fail above 5%.

Run from the repo root with the Studio test environment active:
python tests/studio/multi_account/perf/compare.py --output artifacts/perf.json
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def regressions(results: dict, *, tolerance: float = 0.05) -> list[str]:
    return [
        f"{endpoint} {percentile}: {results['head'][endpoint][percentile]:.4f} ms > "
        f"{results['base'][endpoint][percentile] * (1 + tolerance):.4f} ms"
        for endpoint in ("status", "history")
        for percentile in ("p50_ms", "p95_ms")
        if results["head"][endpoint][percentile]
        > results["base"][endpoint][percentile] * (1 + tolerance)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--base-ref", default = None, help = "defaults to the merge base with origin/main"
    )
    parser.add_argument("--rounds", type = int, default = 3)
    parser.add_argument("--output", type = Path, default = REPO / "artifacts/perf.json")
    args = parser.parse_args()
    assert args.rounds >= 1
    assert args.output.resolve().is_relative_to(REPO), "Benchmark artifacts must stay in this clone"
    sys.path.insert(0, str(REPO / "studio/backend/tests/multi_account"))
    from perf_utils import baseline_ref, materialize_revision, run_probe

    if args.base_ref is None:
        args.base_ref = baseline_ref()
        if args.base_ref is None:
            raise SystemExit("no baseline commit reachable; pass --base-ref")

    scratch_parent = REPO / ".tmp"
    scratch_parent.mkdir(exist_ok = True)
    with tempfile.TemporaryDirectory(prefix = "account-perf-", dir = scratch_parent) as directory:
        scratch = Path(directory)
        base = materialize_revision(args.base_ref, scratch / "baseline")
        series = {"base": [], "head": []}
        for iteration in range(args.rounds):
            # Alternate order to reduce a monotonic machine-load or temperature bias.
            for label in ("base", "head") if iteration % 2 == 0 else ("head", "base"):
                print(f"Round {iteration + 1}/{args.rounds}: {label}", flush = True)
                series[label].append(
                    run_probe(
                        base if label == "base" else REPO / "studio/backend",
                        scratch / f"{label}-{iteration}",
                        mode = "timing",
                    )
                )
        results = {
            label: {
                endpoint: {
                    metric: statistics.median(run[endpoint][metric] for run in runs)
                    for metric in ("samples", "p50_ms", "p95_ms")
                }
                for endpoint in ("status", "history")
            }
            for label, runs in series.items()
        }
    results["rounds"] = series
    results["base_ref"] = args.base_ref
    results["base_commit"] = subprocess.check_output(
        ["git", "rev-parse", f"{args.base_ref}^{{commit}}"], cwd = REPO, text = True
    ).strip()
    results["head_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd = REPO, text = True
    ).strip()
    results["python"] = platform.python_version()
    results["platform"] = platform.platform()
    failures = regressions(results)
    results["passed"] = not failures
    results["failures"] = failures
    args.output.parent.mkdir(parents = True, exist_ok = True)
    args.output.write_text(json.dumps(results, indent = 2) + "\n", encoding = "utf-8")
    for endpoint in ("status", "history"):
        for metric in ("p50_ms", "p95_ms"):
            base_value, head_value = (
                results["base"][endpoint][metric],
                results["head"][endpoint][metric],
            )
            print(
                f"{endpoint:7} {metric}: base={base_value:.4f} ms head={head_value:.4f} ms "
                f"delta={(head_value / base_value - 1) * 100:+.2f}%"
            )
    assert not failures, "; ".join(failures)


if __name__ == "__main__":
    main()
