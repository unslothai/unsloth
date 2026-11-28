"""Pairwise equivalence diff for GRPO backend runs.

Runs `torch_debugging_utils.compare_training_runs` over the StatisticsCallback
JSONs produced by `qwen3_grpo_unified.py`, plus reward / KL diffs which the
base util doesn't track (it's loss/grad-focused).

Usage:
    python scripts/benchmarks/compare_grpo_runs.py \
        --ref logs/grpo_vllm_30.json \
        --candidate logs/grpo_unsloth_fi_false_30.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path("/mnt/disks/unslothai/ubuntu/workspace_31")
for p in (HERE, WORKSPACE_ROOT):
    sys.path.insert(0, str(p))


def _arrays(path: str):
    with open(path) as f:
        logs = json.load(f)
    return {
        "loss": [l.get("loss") for l in logs if "loss" in l],
        "reward": [l.get("reward") for l in logs if "reward" in l],
        "kl": [l.get("kl") for l in logs if "kl" in l],
        "grad_norm": [l.get("grad_norm") for l in logs if "grad_norm" in l],
        "time_ms": [l.get("time_ms") for l in logs if "time_ms" in l],
    }


def _diff(a, b):
    if not a or not b:
        return None
    n = min(len(a), len(b))
    diffs = [
        abs(a[i] - b[i]) for i in range(n) if a[i] is not None and b[i] is not None
    ]
    if not diffs:
        return None
    return {
        "n_compared": len(diffs),
        "max_abs": max(diffs),
        "mean_abs": sum(diffs) / len(diffs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required = True)
    p.add_argument("--candidate", required = True)
    args = p.parse_args()

    from torch_debugging_utils import compare_training_runs

    base = compare_training_runs(args.ref, args.candidate, loss_tol = 1e-3, grad_tol = 1e-3)

    ref_a = _arrays(args.ref)
    cand_a = _arrays(args.candidate)
    extras = {k: _diff(ref_a[k], cand_a[k]) for k in ("reward", "kl", "time_ms")}

    out = {
        "ref": args.ref,
        "candidate": args.candidate,
        "compare_training_runs": base,
        "reward_diff": extras["reward"],
        "kl_diff": extras["kl"],
        "time_diff_ms": extras["time_ms"],
    }
    print(json.dumps(out, indent = 2))


if __name__ == "__main__":
    main()
