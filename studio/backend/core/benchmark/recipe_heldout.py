# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The "Recipe held-out" benchmark task: answer held-out rows of a recipe run with the model under
test (and optionally a baseline), and score each answer with a check or an LLM judge.

Runs through the same orchestrator slot and run storage as the lm-eval tasks, but talks to any
OpenAI-compatible chat endpoint, so it also scores a LoRA adapter that lm-eval can't load yet."""

from __future__ import annotations

import ast
import math
import os
import re
from typing import Any, Callable, Optional

from core.data_recipe.text_format_validators import _validate_text_format

RECIPE_HELDOUT_TASK_ID = "recipe_heldout"
VALIDATOR_SCORERS = ("reference", "json", "markdown", "python")
SCORERS = (*VALIDATOR_SCORERS, "judge")
MAX_BENCHMARK_ROWS = 200
MAX_FAILED_EXAMPLES = 50

DEFAULT_JUDGE_RUBRIC = (
    "The answer is correct, complete and follows the instruction. "
    "If a reference answer is given, the answer must agree with it."
)

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)
_VERDICT_RE = re.compile(r"\b(PASS|FAIL)\b", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

ChatFn = Callable[[str, list[dict[str, str]]], str]


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _unfence(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text


def score_with_validator(scorer: str, output: str, reference: str | None) -> tuple[bool, str]:
    """(passed, reason) for one answer under a validator scorer."""
    if scorer == "reference":
        expected = _normalize(reference or "")
        if not expected:
            return False, "Row has no reference answer."
        got = _normalize(output)
        if got == expected or f" {expected} " in f" {got} ":
            return True, ""
        return False, "Answer does not contain the reference."
    if scorer == "python":
        code = _unfence(output)
        if not code.strip():
            return False, "Answer is empty."
        try:
            ast.parse(code)
        except (SyntaxError, ValueError) as exc:
            return False, f"Python syntax error: {exc}"
        return True, ""
    if scorer in ("json", "markdown"):
        value = _unfence(output) if scorer == "json" else output
        result = _validate_text_format(value = value, format_kind = scorer)
        return bool(result["is_valid"]), str(result.get("error_message") or "")
    raise ValueError(f"unknown scorer: {scorer}")


def build_judge_messages(
    rubric: str, prompt: str, output: str, reference: str | None
) -> list[dict[str, str]]:
    parts = [
        f"Criteria: {rubric.strip() or DEFAULT_JUDGE_RUBRIC}",
        f"Instruction:\n{prompt}",
    ]
    if reference:
        parts.append(f"Reference answer:\n{reference}")
    parts.append(f"Answer to grade:\n{output}")
    parts.append("Reply with PASS or FAIL on the first line, then one short sentence why.")
    return [
        {
            "role": "system",
            "content": "You grade answers strictly against the criteria. You only reply PASS or FAIL plus one sentence.",
        },
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_judge_verdict(text: str) -> Optional[bool]:
    match = _VERDICT_RE.search(strip_thinking(text))
    if match is None:
        return None
    return match.group(1).upper() == "PASS"


def _cell(row: dict[str, Any], column: str | None) -> str:
    if not column:
        return ""
    value = row.get(column)
    return "" if value is None else str(value)


def _score_answer(
    spec: dict[str, Any], chat: ChatFn, prompt: str, output: str, reference: str
) -> tuple[bool, str]:
    scorer = spec["scorer"]
    if scorer != "judge":
        return score_with_validator(scorer, output, reference)
    verdict_text = chat(
        "judge", build_judge_messages(spec.get("rubric") or "", prompt, output, reference or None)
    )
    verdict = parse_judge_verdict(verdict_text)
    if verdict is None:
        return False, "Judge reply had no PASS or FAIL."
    reason = strip_thinking(verdict_text)
    return verdict, "" if verdict else reason[:500]


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in results if r.get("error") is None]
    passed = sum(1 for r in scored if r["passed"])
    return {
        "total": len(results),
        "scored": len(scored),
        "passed": passed,
        "errors": len(results) - len(scored),
        "score": (passed / len(scored)) if scored else None,
    }


def run_benchmark(
    spec: dict[str, Any],
    *,
    chat: ChatFn,
    on_progress: Callable[[int, int], None] = lambda done, total: None,
    is_cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Score ``spec["rows"]``. ``chat(target, messages)`` answers for target in
    {"candidate", "baseline", "judge"}; a raised exception marks only that row as errored."""
    rows: list[dict[str, Any]] = spec["rows"]
    prompt_column: str = spec["prompt_column"]
    reference_column: str | None = spec.get("reference_column") or None
    system_prompt: str = spec.get("system_prompt") or ""
    targets = ["candidate"] + (["baseline"] if spec.get("with_baseline") else [])
    per_target: dict[str, list[dict[str, Any]]] = {t: [] for t in targets}
    total = len(rows) * len(targets)
    done = 0
    on_progress(done, total)
    cancelled = False
    for index, row in enumerate(rows):
        prompt = _cell(row, prompt_column)
        reference = _cell(row, reference_column)
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": prompt}
        ]
        for target in targets:
            if is_cancelled():
                cancelled = True
                break
            result: dict[str, Any] = {"index": index, "passed": False, "error": None, "output": None}
            try:
                output = strip_thinking(chat(target, messages))
                passed, reason = _score_answer(spec, chat, prompt, output, reference)
                result.update(output = output, passed = passed, reason = reason)
            except Exception as exc:
                result["error"] = str(exc)[:500] or type(exc).__name__
            per_target[target].append(result)
            done += 1
            on_progress(done, total)
        if cancelled:
            break

    candidate = per_target["candidate"]
    samples = {
        target: [
            {
                "doc_id": r["index"],
                "question": _cell(rows[r["index"]], prompt_column),
                "target": _cell(rows[r["index"]], reference_column),
                "response": r.get("output"),
                "raw_response": r.get("error") or r.get("reason") or None,
                "correct": bool(r["passed"]),
            }
            for r in results
        ]
        for target, results in per_target.items()
    }
    failed = []
    for result in candidate:
        if result["passed"]:
            continue
        row = rows[result["index"]]
        failed.append(
            {
                "prompt": _cell(row, prompt_column),
                "reference": _cell(row, reference_column) or None,
                "output": result.get("output"),
                "reason": result.get("error") or result.get("reason") or "",
                "row": row,
            }
        )
    summary = {"candidate": _summary(candidate)}
    if "baseline" in per_target:
        summary["baseline"] = _summary(per_target["baseline"])
        a, b = summary["candidate"]["score"], summary["baseline"]["score"]
        summary["delta"] = (a - b) if a is not None and b is not None else None
    return {
        **summary,
        "cancelled": cancelled,
        "failed_examples": failed[:MAX_FAILED_EXAMPLES],
        "failed_total": len(failed),
        "samples": samples,
    }


def eval_metrics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """The pass rate as an ``acc,none`` metric, so the Benchmarks history reads it as the headline."""
    score, n = summary.get("score"), summary.get("scored") or 0
    if score is None:
        return []
    stderr = math.sqrt(score * (1 - score) / (n - 1)) if n > 1 else None
    return [{"name": "acc,none", "score": round(score, 4), "stderr": None if stderr is None else f"{stderr:.4f}"}]


DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT_S = 120


def resolve_chat_target(
    alias: str, configs: dict[str, dict[str, Any]], providers: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """URL, headers and body template for one model block of the recipe."""
    mc = configs.get(alias)
    if mc is None:
        raise ValueError(f"Model {alias} is not in the recipe.")
    provider = providers.get(str(mc.get("provider") or ""))
    if provider is None:
        raise ValueError(f"Model {alias} has no provider.")
    params = dict(mc.get("inference_parameters") or {})
    api_key = provider.get("api_key") or ""
    env_name = provider.get("api_key_env")
    if not api_key and isinstance(env_name, str) and env_name:
        api_key = os.environ.get(env_name, "")
    body: dict[str, Any] = {"model": mc.get("model") or "local"}
    body.update(provider.get("extra_body") or {})
    body.update(params.pop("extra_body", None) or {})
    timeout = params.pop("timeout", None) or DEFAULT_TIMEOUT_S
    params.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
    params.setdefault("temperature", 0)
    body.update(params)
    headers = dict(provider.get("extra_headers") or {})
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return {
        "url": f"{str(provider.get('endpoint') or '').rstrip('/')}/chat/completions",
        "headers": headers,
        "body": body,
        "timeout": float(timeout),
    }


def post_chat(client: Any, target: dict[str, Any], messages: list[dict[str, str]]) -> str:
    response = client.post(
        target["url"],
        json = {**target["body"], "messages": messages, "stream": False},
        headers = target["headers"],
        timeout = target["timeout"],
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail") or response.text
        except ValueError:
            detail = response.text
        raise RuntimeError(f"HTTP {response.status_code}: {str(detail)[:300]}")
    message = response.json()["choices"][0]["message"]
    return str(message.get("content") or "")
