# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import re
import time
from typing import Optional


# lm_eval mixes real metrics with bookkeeping keys in the per-task results dict.
# These are not scores and must never surface in the metrics list.
_NON_METRIC_KEYS = frozenset({
    "alias",
    "name",
    "sample_len",       # token/sample count, not a score
    "num_samples",
    "n_samples",
    "effective_samples",
    "bootstrap_iters",
})
_NON_METRIC_KEYS_LOWER = frozenset(k.lower() for k in _NON_METRIC_KEYS)

# Priority list for selecting the default metric from lm_eval results.
# First match wins. Shared by the summary parser and the DB read path, which
# derives the headline metric for historical runs instead of storing it.
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
    "acc,none",
    "acc",
)


def pick_default_metric(metrics: list) -> str:
    """Pick the headline metric from an already-filtered metrics list.

    Walks :data:`_DEFAULT_METRIC_PRIORITY` over the metric names. Because the
    priority ordering is deterministic given the names, runs derive their
    headline metric at read time (see the DB layer) instead of persisting it.
    """
    names = {m.get("name") for m in metrics if isinstance(m, dict)}
    for candidate in _DEFAULT_METRIC_PRIORITY:
        if candidate in names:
            return candidate
    return ""


def parse_run_summary(dir_name: str, data: dict) -> Optional[dict]:
    """Extract a summary dict from a parsed results.json."""
    results = data.get("results", {})
    configs = data.get("configs", {})
    n_shot_map = data.get("n-shot", {})
    n_samples_map = data.get("n-samples", {})

    if not results:
        return None

    task_name = next(iter(results))
    task_results = results[task_name]
    task_config = configs.get(task_name, {})
    metadata = task_config.get("metadata", {}) if isinstance(task_config, dict) else {}

    model = metadata.get("model", "unknown") if isinstance(metadata, dict) else "unknown"

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
            stderr_val = str(task_results.get(stderr_key, "")) if stderr_key in task_results else None
            metrics.append({
                "name": key,
                "score": round(float(val), 4),
                "stderr": stderr_val,
            })
        elif isinstance(val, str):
            try:
                fv = float(val)
                metrics.append({
                    "name": key,
                    "score": round(fv, 4),
                    "stderr": None,
                })
            except (ValueError, TypeError):
                pass

    n_shot = None
    n_shot_raw = n_shot_map.get(task_name) if isinstance(n_shot_map, dict) else None
    if n_shot_raw is not None:
        n_shot = int(n_shot_raw) if isinstance(n_shot_raw, (int, float)) else None

    n_samples = 0
    ns_raw = n_samples_map.get(task_name) if isinstance(n_samples_map, dict) else None
    if isinstance(ns_raw, dict):
        n_samples = int(ns_raw.get("effective", ns_raw.get("original", 0)))
    elif isinstance(ns_raw, (int, float)):
        n_samples = int(ns_raw)

    return {
        "id": dir_name,
        "task": task_name,
        "model": model,
        "metrics": metrics,
        "n_samples": n_samples,
        "num_fewshot": n_shot,
        "created_at": created_at,
    }


# lm_eval writes per-sample metric values under version-dependent keys:
# older releases use the bare metric name ("exact_match"), newer ones append
# the filter group ("exact_match,strict-match"). Prefer the strict-match EM so
# per-sample correctness stays consistent with the headline metric.


def _sample_correct(s: dict) -> bool:
    """Derive per-sample correctness from the task's primary scoring metric.

    Reuses the same priority ordering used to pick the headline metric in
    ``parse_run_summary`` so per-sample correctness stays consistent with the
    score shown in the UI. Every metric in ``_DEFAULT_METRIC_PRIORITY`` is
    normalized to 0–1, where 1.0 means fully correct (this covers both
    log_likelihood tasks that score with ``acc``/``acc_norm`` and generation
    tasks that score with ``exact_match``/``pass@1``/``f1``).
    """
    for candidate in _DEFAULT_METRIC_PRIORITY:
        val = s.get(candidate)
        if isinstance(val, (int, float)):
            return val >= 1.0
    # Fallback: any exact_match/acc-style key present on the sample.
    for key, val in s.items():
        if key.endswith("_stderr,none") or key.endswith("_stderr"):
            continue
        if key.startswith(("exact_match", "acc", "mc2", "pass")) and isinstance(val, (int, float)):
            return val >= 1.0
    return False


def _as_text(value):
    """eval_samples columns are text; loglikelihood responses are (logprob, greedy) tuples."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, default = str)


def extract_samples(data: dict, task_name: str) -> list[dict]:
    """Extract per-sample results from lm_eval output."""
    samples = []
    raw_samples = data.get("samples", {})
    task_samples = raw_samples.get(task_name, []) if isinstance(raw_samples, dict) else []

    for s in task_samples:
        doc = s.get("doc", {})
        question = doc.get("question", "") if isinstance(doc, dict) else ""
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

        correct = _sample_correct(s)

        samples.append({
            "doc_id": s.get("doc_id", 0),
            "question": _as_text(question),
            "target": _as_text(target),
            "response": _as_text(response),
            "raw_response": _as_text(raw_response),
            "correct": correct,
        })
    return samples
