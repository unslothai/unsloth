# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import re
import time
from typing import Optional


# lm_eval mixes real metrics with bookkeeping keys in the per-task results dict.
# These are not scores and must never surface in the metrics list.
_NON_METRIC_KEYS = frozenset(
    {
        "alias",
        "name",
        "sample_len",  # token/sample count, not a score
        "num_samples",
        "n_samples",
        "effective_samples",
        "bootstrap_iters",
    }
)
_NON_METRIC_KEYS_LOWER = frozenset(k.lower() for k in _NON_METRIC_KEYS)

# Headline metric, first match wins. The DB derives it at read time instead of storing it.
_DEFAULT_METRIC_PRIORITY = (
    "acc_norm,none",
    "acc_norm",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match",
    "pass@1",
    "mc2",
    "f1,none",
    "f1",
    "prompt_level_strict_acc",
    "acc,none",
    "acc",
)


def pick_default_metric(metrics: list) -> str:
    """The headline metric name from a parsed metrics list, or "" if none is known."""
    names = [m.get("name") for m in metrics if isinstance(m, dict) and m.get("name")]
    for candidate in _DEFAULT_METRIC_PRIORITY:
        if candidate in names:
            return candidate
    # Task-specific filters: "exact_match,custom-extract" (MMLU-Pro), "exact_match,none" (MATH).
    for candidate in _DEFAULT_METRIC_PRIORITY:
        base = candidate.partition(",")[0]
        for name in names:
            if name.partition(",")[0] == base:
                return name
    return ""


def _leaf_tasks(data: dict, task_name: str) -> list[str]:
    """The scored subtasks under a group (mmlu -> mmlu_stem -> mmlu_anatomy ...), or the task itself."""
    group_subtasks = data.get("group_subtasks") or {}
    if task_name not in group_subtasks:
        return [task_name]
    leaves: list[str] = []
    for sub in group_subtasks[task_name]:
        leaves.extend(_leaf_tasks(data, sub))
    return leaves


def parse_run_summary(
    dir_name: str,
    data: dict,
    task: Optional[str] = None,
) -> Optional[dict]:
    """Extract a summary dict from a parsed results.json.

    ``task`` is the task that was requested. For a group (mmlu, bbh, ...) its
    aggregate lives under ``groups`` while ``results`` lists every subtask, so
    the first ``results`` entry would be one subtask, not the score asked for.
    """
    results = data.get("results", {})
    groups = data.get("groups") or {}
    configs = data.get("configs", {})
    n_shot_map = data.get("n-shot", {})
    n_samples_map = data.get("n-samples", {})

    if not results:
        return None

    task_name = task if task in groups or task in results else next(iter(results))
    task_results = groups.get(task_name) or results[task_name]
    leaves = _leaf_tasks(data, task_name)
    task_config = configs.get(task_name) or configs.get(leaves[0], {})
    metadata = task_config.get("metadata", {}) if isinstance(task_config, dict) else {}

    # The run's own model_args are the record of what ran; lm_eval copies them into task
    # metadata only when it builds the TaskManager itself.
    run_args = (data.get("config") or {}).get("model_args")
    model = (
        (run_args.get("model") if isinstance(run_args, dict) else None)
        or (metadata.get("model") if isinstance(metadata, dict) else None)
        or "unknown"
    )

    created_at = dir_name
    ts_match = re.search(r"_(\d{8}_\d{6})$", dir_name)
    if ts_match:
        try:
            dt = time.strptime(ts_match.group(1), "%Y%m%d_%H%M%S")
            created_at = time.strftime("%Y-%m-%dT%H:%M:%S", dt)
        except ValueError:
            created_at = dir_name

    metrics = []
    for key, val in task_results.items():
        # lm_eval keys are "<metric>,<filter>" ("acc,none",
        # "exact_match,strict-match"); the stderr of each is "<metric>_stderr,<filter>".
        metric, sep, filt = key.partition(",")
        if metric.endswith("_stderr") or key.lower() in _NON_METRIC_KEYS_LOWER:
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            stderr_key = f"{metric}_stderr{sep}{filt}"
            stderr_val = (
                str(task_results.get(stderr_key, "")) if stderr_key in task_results else None
            )
            metrics.append(
                {
                    "name": key,
                    "score": round(float(val), 4),
                    "stderr": stderr_val,
                }
            )
        elif isinstance(val, str):
            try:
                fv = float(val)
                metrics.append(
                    {
                        "name": key,
                        "score": round(fv, 4),
                        "stderr": None,
                    }
                )
            except (ValueError, TypeError):
                pass

    n_shot = None
    n_shot_raw = None
    if isinstance(n_shot_map, dict):
        n_shot_raw = n_shot_map.get(task_name, n_shot_map.get(leaves[0]))
    if n_shot_raw is not None:
        n_shot = int(n_shot_raw) if isinstance(n_shot_raw, (int, float)) else None

    n_samples = 0
    for leaf in leaves:
        ns_raw = n_samples_map.get(leaf) if isinstance(n_samples_map, dict) else None
        if isinstance(ns_raw, dict):
            n_samples += int(ns_raw.get("effective", ns_raw.get("original", 0)))
        elif isinstance(ns_raw, (int, float)):
            n_samples += int(ns_raw)

    return {
        "id": dir_name,
        "task": task_name,
        "model": model,
        "metrics": metrics,
        "n_samples": n_samples,
        "num_fewshot": n_shot,
        "created_at": created_at,
    }


def _sample_score(s: dict):
    """The sample's value for the headline metric (0 to 1, 1.0 is full credit), or None."""
    for candidate in _DEFAULT_METRIC_PRIORITY:
        val = s.get(candidate)
        if isinstance(val, (int, float)):
            return float(val)
    # Fallback: any exact_match/acc-style key present on the sample.
    for key, val in s.items():
        if key.endswith("_stderr,none") or key.endswith("_stderr"):
            continue
        if key.startswith(("exact_match", "acc", "mc2", "pass")) and isinstance(val, (int, float)):
            return float(val)
    return None


def _sample_correct(s: dict) -> bool:
    score = _sample_score(s)
    return score is not None and score >= 1.0


def _as_text(value):
    """eval_samples columns are text; loglikelihood responses are (logprob, greedy) tuples."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, default = str)


def _headline_filter(data: dict, task_name: str) -> Optional[str]:
    results = data.get("results") or {}
    task_results = (data.get("groups") or {}).get(task_name) or results.get(task_name) or {}
    names = [{"name": k} for k in task_results if isinstance(k, str)]
    _, sep, filt = pick_default_metric(names).partition(",")
    return filt if sep else None


def extract_samples(data: dict, task_name: str) -> list[dict]:
    """Extract per-sample results from lm_eval output."""
    samples = []
    raw_samples = data.get("samples", {})
    task_samples = []
    leaves = _leaf_tasks(data, task_name)
    if isinstance(raw_samples, dict):
        for leaf in leaves:
            task_samples.extend(raw_samples.get(leaf, []))
    # lm_eval logs each doc once per filter (gsm8k: strict-match and flexible-extract).
    filters = {s.get("filter") for s in task_samples if s.get("filter")}
    if len(filters) > 1:
        keep = _headline_filter(data, task_name)
        if keep not in filters:
            keep = next(s.get("filter") for s in task_samples if s.get("filter"))
        task_samples = [s for s in task_samples if s.get("filter") == keep]
    # doc_ids restart in every subtask of a group, and eval_samples is keyed
    # on (run_id, doc_id): number a group's samples in order instead.
    renumber = len(leaves) > 1

    for index, s in enumerate(task_samples):
        doc = s.get("doc", {})
        question = doc.get("question", "") if isinstance(doc, dict) else ""
        if not question:
            # Tasks like hellaswag have no "question" field; the rendered
            # prompt is the context of the first request.
            args = s.get("arguments") or []
            first = args[0] if args else None
            if isinstance(first, (list, tuple)) and first and isinstance(first[0], str):
                question = first[0]
        target = s.get("target", "")
        filtered = s.get("filtered_resps", [])
        # Generation tasks have one text response; loglikelihood tasks have one
        # (logprob, is_greedy) pair per choice, which only make sense together.
        if len(filtered) == 1 or (filtered and isinstance(filtered[0], str)):
            response = filtered[0]
        else:
            response = filtered or None
        raw_resp = s.get("resps", [])
        raw_response = raw_resp[0][0] if raw_resp and raw_resp[0] else None

        score = _sample_score(s)

        samples.append(
            {
                "doc_id": index if renumber else s.get("doc_id", 0),
                "question": _as_text(question),
                "target": _as_text(target),
                "response": _as_text(response),
                "raw_response": _as_text(raw_response),
                "correct": score is not None and score >= 1.0,
                # f1, mc2 (truthfulqa_mc2's "acc") and the like give partial credit.
                "score": score,
            }
        )
    return samples
