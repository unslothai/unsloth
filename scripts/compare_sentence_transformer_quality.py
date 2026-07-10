#!/usr/bin/env python3
"""Compare paired baseline/optimized SentenceTransformer quality runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--optimized", type=Path, action="append", required=True)
    parser.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=1 / 1024,
        help="Allowed mean absolute accuracy drop (default: one 1024-row example).",
    )
    parser.add_argument(
        "--max-margin-drop",
        type=float,
        default=1e-3,
        help="Allowed mean cosine-margin drop.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.baseline) != len(args.optimized):
        parser.error("--baseline and --optimized must be provided the same number of times")
    if args.max_accuracy_drop < 0 or args.max_margin_drop < 0:
        parser.error("non-inferiority margins must be non-negative")
    return args


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(
    baseline: dict[str, Any],
    optimized: dict[str, Any],
    path: tuple[str, ...],
) -> None:
    left: Any = baseline
    right: Any = optimized
    for key in path:
        left = left[key]
        right = right[key]
    if left != right:
        dotted = ".".join(path)
        raise ValueError(f"Paired contract mismatch at {dotted}: {left!r} != {right!r}")


def require_equal_if_present(
    baseline: dict[str, Any],
    optimized: dict[str, Any],
    path: tuple[str, ...],
) -> None:
    left: Any = baseline
    right: Any = optimized
    for key in path:
        left_present = isinstance(left, dict) and key in left
        right_present = isinstance(right, dict) and key in right
        if left_present != right_present:
            dotted = ".".join(path)
            raise ValueError(f"Paired contract field presence mismatch at {dotted}")
        if not left_present:
            return
        left = left[key]
        right = right[key]
    if left != right:
        dotted = ".".join(path)
        raise ValueError(f"Paired contract mismatch at {dotted}: {left!r} != {right!r}")


def require_optimized_fast_path(optimized: dict[str, Any], path: Path) -> None:
    model_contract = optimized["model_contract"]
    if (
        model_contract["fused_lora_layers"] <= 0
        and model_contract["fused_lora_linears"] <= 0
    ):
        raise ValueError(
            f"Optimized run did not activate an eager LoRA fast path: {path}"
        )


def main() -> None:
    args = parse_args()
    pairs = list(zip(args.baseline, args.optimized, strict=True))
    rows: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    campaign_contract: dict[str, Any] | None = None
    campaign_model: str | None = None

    for baseline_path, optimized_path in pairs:
        baseline = load(baseline_path)
        optimized = load(optimized_path)
        if baseline.get("mode") != "baseline" or optimized.get("mode") != "optimized":
            raise ValueError("Each pair must contain baseline then optimized results")
        for contract_path in (
            ("model",),
            ("revision",),
            ("seed",),
            ("steps",),
            ("batch_size",),
            ("eval_batch_size",),
            ("max_length",),
            ("dtype",),
            ("torch",),
            ("device",),
            ("learning_rate",),
            ("weight_decay",),
            ("fused_optimizer",),
            ("prompt_contract",),
            ("dataset", "revision"),
            ("dataset", "selected"),
            ("inputs", "training_schedule_sha256"),
            ("inputs", "all_token_tensors_sha256"),
            ("model_contract", "target_modules"),
            ("model_contract", "lora_config"),
            ("model_contract", "initial_trainable_sha256"),
            ("model_contract", "initial_frozen_sha256"),
            ("quality", "final", "examples"),
        ):
            require_equal(baseline, optimized, contract_path)
        for optional_contract_path in (
            ("model_contract", "resolved_model_commit"),
            ("model_contract", "initial_frozen_values_sha256"),
        ):
            require_equal_if_present(baseline, optimized, optional_contract_path)

        seed = int(baseline["seed"])
        if seed in seen_seeds:
            raise ValueError(f"Duplicate seed in comparison campaign: {seed}")
        seen_seeds.add(seed)
        current_campaign_contract = {
            "model": baseline["model"],
            "revision": baseline["revision"],
            "steps": baseline["steps"],
            "batch_size": baseline["batch_size"],
            "eval_batch_size": baseline["eval_batch_size"],
            "max_length": baseline["max_length"],
            "dtype": baseline["dtype"],
            "torch": baseline["torch"],
            "device": baseline["device"],
            "learning_rate": baseline["learning_rate"],
            "weight_decay": baseline["weight_decay"],
            "fused_optimizer": baseline["fused_optimizer"],
            "prompt_contract": baseline["prompt_contract"],
            "dataset_revision": baseline["dataset"]["revision"],
            "dataset_selected": baseline["dataset"]["selected"],
            "target_modules": baseline["model_contract"]["target_modules"],
            "lora_config": baseline["model_contract"]["lora_config"],
            "initial_frozen_sha256": baseline["model_contract"][
                "initial_frozen_sha256"
            ],
            "initial_frozen_values_sha256": baseline["model_contract"].get(
                "initial_frozen_values_sha256"
            ),
            "resolved_model_commit": baseline["model_contract"].get(
                "resolved_model_commit"
            ),
            "eval_examples": baseline["quality"]["final"]["examples"],
        }
        if campaign_contract is None:
            campaign_contract = current_campaign_contract
            campaign_model = baseline["model"]
        elif current_campaign_contract != campaign_contract:
            raise ValueError(
                "Every paired run must share one model/training/evaluation campaign contract"
            )
        if not baseline["model_contract"]["frozen_parameters_unchanged"]:
            raise ValueError(f"Baseline frozen parameters changed: {baseline_path}")
        if not optimized["model_contract"]["frozen_parameters_unchanged"]:
            raise ValueError(f"Optimized frozen parameters changed: {optimized_path}")
        require_optimized_fast_path(optimized, optimized_path)

        baseline_quality = baseline["quality"]["final"]
        optimized_quality = optimized["quality"]["final"]
        rows.append(
            {
                "seed": seed,
                "baseline_accuracy": baseline_quality["triplet_accuracy"],
                "optimized_accuracy": optimized_quality["triplet_accuracy"],
                "accuracy_delta": (
                    optimized_quality["triplet_accuracy"]
                    - baseline_quality["triplet_accuracy"]
                ),
                "baseline_mean_margin": baseline_quality["mean_margin"],
                "optimized_mean_margin": optimized_quality["mean_margin"],
                "mean_margin_delta": (
                    optimized_quality["mean_margin"] - baseline_quality["mean_margin"]
                ),
            }
        )

    accuracy_delta = statistics.mean(row["accuracy_delta"] for row in rows)
    margin_delta = statistics.mean(row["mean_margin_delta"] for row in rows)
    checks = {
        "accuracy_non_inferior": accuracy_delta >= -args.max_accuracy_drop,
        "mean_margin_non_inferior": margin_delta >= -args.max_margin_drop,
        "at_least_three_distinct_seeds": len(seen_seeds) >= 3,
    }
    result = {
        "model": campaign_model,
        "campaign_contract": campaign_contract,
        "paired_runs": rows,
        "mean_baseline_accuracy": statistics.mean(
            row["baseline_accuracy"] for row in rows
        ),
        "mean_optimized_accuracy": statistics.mean(
            row["optimized_accuracy"] for row in rows
        ),
        "mean_accuracy_delta": accuracy_delta,
        "mean_margin_delta": margin_delta,
        "thresholds": {
            "max_accuracy_drop": args.max_accuracy_drop,
            "max_margin_drop": args.max_margin_drop,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
