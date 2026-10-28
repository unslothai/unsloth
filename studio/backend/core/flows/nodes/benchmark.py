# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Benchmark node: a front end to the Benchmarks Evals tab, so every run it starts is stored there too."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional
from urllib.parse import quote

from core.flows.nodes._common import note_loaded, resident_name
from core.flows.registry import FlowNodeError, NodeContext, handler

EVALS = "/api/benchmarks/evals"
RECIPE_TASK = "recipe_heldout"
TASKS = {
    "Recipe held-out": RECIPE_TASK,
    "HellaSwag": "hellaswag",
    "ARC-Challenge": "arc_challenge",
    "WinoGrande": "winogrande",
    "GSM8K": "gsm8k",
    "MMLU": "mmlu",
    "TruthfulQA MC1": "truthfulqa_mc1",
}
SCORERS = {"Reference match": "reference", "LLM judge": "judge"}
MAX_ROWS = 200
PREVIEW_CHARS = 400
PROMPT_GUESSES = ("prompt", "instruction", "question", "input", "query", "user")
REFERENCE_GUESSES = ("response", "output", "answer", "completion", "reference", "target", "assistant")


def guess_column(columns: list[str], wanted: str, guesses: tuple[str, ...]) -> Optional[str]:
    if wanted:
        if wanted not in columns:
            raise FlowNodeError(f"The rows have no {wanted!r} column (they have {', '.join(columns)})")
        return wanted
    lower = {c.lower(): c for c in columns}
    return next((lower[g] for g in guesses if g in lower), None)


def headline(metrics: list[dict]) -> tuple[Optional[float], str]:
    """The metric the Evals tab leads with: accuracy or exact match before anything else."""
    scored = [m for m in metrics if isinstance(m.get("score"), (int, float))]
    for prefix in ("acc,", "acc_norm,", "exact_match,", "prompt_level_strict_acc,"):
        for m in scored:
            if str(m.get("name", "")).startswith(prefix):
                return float(m["score"]), str(m["name"])
    return (float(scored[0]["score"]), str(scored[0]["name"])) if scored else (None, "")


def _short(value: Any) -> Any:
    if isinstance(value, str) and len(value) > PREVIEW_CHARS:
        return value[:PREVIEW_CHARS] + "…"
    return value


async def _load_target(ctx: NodeContext) -> tuple[str, bool]:
    """Load what is being scored; (label, is_lora)."""
    checkpoint = ctx.inputs.get("checkpoint")
    name = checkpoint.get("name") if isinstance(checkpoint, dict) else None
    if name:
        from studio_mcp import loading, trained_models

        from core.flows.nodes._common import FLOW_CHAT_CONTEXT

        deadline = time.monotonic() + 3600
        ctx.set_status("loading_model", f"Loading {name}")
        try:
            trained = await trained_models.resolve_by(
                deadline, ctx.caller, model = None, checkpoint = name, export = None, kind = "llm", variant = None
            )
            await loading.load_llm(
                ctx.caller,
                None,
                model = trained.ref,
                variant = None,
                max_seq_length = FLOW_CHAT_CONTEXT,
                load_in_4bit = True,
                hf_token = None,
                kind = "llm",
                is_lora = trained.is_lora,
                deadline = deadline,
            )
            note_loaded(ctx, "llm", trained.ref)
        except FlowNodeError:
            raise
        except Exception as exc:
            raise FlowNodeError(f"Could not load {name}: {exc}") from None
        ctx.resident.clear()
        ctx.set_status("running")
        return name, bool(trained.is_lora)
    model = ctx.params.get("model")
    await ctx.ensure_model("chat", model if isinstance(model, dict) and model.get("repo") else None)
    label = await resident_name(ctx, "chat")
    if not label:
        raise FlowNodeError("Wire a Fine-tune checkpoint in, pick a model, or load one in Chat first")
    return label, False


async def _read_rows(ctx: NodeContext, dataset: Any, limit: int) -> list[dict]:
    from core.flows.nodes.data import recipe_artifact, recipe_rows
    from core.flows.nodes.training import _load_rows
    from core.training.account_jobs import account_hf_token

    if not isinstance(dataset, dict) or dataset.get("kind") != "dataset":
        raise FlowNodeError("Recipe held-out scores a dataset: wire Fine-tune's Held-out rows or a recipe in")
    if dataset.get("source") == "recipe" and not recipe_artifact(dataset.get("ref")):
        rows = (await recipe_rows(ctx, str(dataset["ref"]), limit = limit))[:limit]
    else:
        token = account_hf_token(getattr(ctx.caller, "hf_token", None))
        try:
            rows = await asyncio.to_thread(_load_rows, dataset, token, limit - 1)
        except FlowNodeError:
            raise
        except Exception as exc:
            raise FlowNodeError(f"Could not read the rows to score: {exc}") from None
    from core.data_recipe.jsonable import to_jsonable

    rows = [to_jsonable(r) for r in rows if isinstance(r, dict)]
    if not rows:
        raise FlowNodeError("There are no rows to score")
    return rows


async def _samples(ctx: NodeContext, run_id: Optional[str]) -> list[dict]:
    if not run_id:
        return []
    detail = await ctx.call("GET", f"{EVALS}/runs/{quote(run_id, safe = '')}")
    samples = detail.get("samples") if isinstance(detail, dict) else None
    return sorted((s for s in samples or [] if isinstance(s, dict)), key = lambda s: s.get("doc_id", 0))


async def _recipe_heldout(ctx: NodeContext, label: str, is_lora: bool, max_rows: int) -> dict:
    rows = await _read_rows(ctx, ctx.inputs.get("dataset"), max_rows)
    columns = sorted({k for r in rows for k in r})
    prompt_col = guess_column(columns, str(ctx.params.get("prompt_column") or "").strip(), PROMPT_GUESSES)
    if not prompt_col:
        raise FlowNodeError("Set the prompt column under Advanced: no column looks like a prompt")
    reference_col = guess_column(
        [c for c in columns if c != prompt_col],
        str(ctx.params.get("reference_column") or "").strip(),
        REFERENCE_GUESSES,
    )
    scorer = SCORERS.get(str(ctx.params.get("scorer") or ""), "reference")
    if scorer == "reference" and not reference_col:
        raise FlowNodeError("Reference match needs a reference column; set it under Advanced or use LLM judge")
    with_base = is_lora and ctx.params.get("compare_base") is not False
    spec = {
        "rows": rows,
        "max_rows": max_rows,
        "prompt_column": prompt_col,
        "reference_column": reference_col,
        "system_prompt": str(ctx.params.get("system_prompt") or ""),
        "scorer": scorer,
        "rubric": str(ctx.params.get("rubric") or ""),
        "model_alias": "candidate",
        "baseline": {"mode": "adapter_off" if with_base else "none"},
        # The judge is the loaded model; with an adapter loaded the route turns it off for judging.
        "judge_alias": "candidate" if scorer == "judge" else None,
        "model_providers": [{"name": "local", "is_local": True}],
        "model_configs": [{"alias": "candidate", "provider": "local", "model": ""}],
    }
    ctx.progress(phase = f"Scoring {len(rows)} rows" + (" with the adapter on and off" if with_base else ""))
    done = await ctx.call(
        "POST",
        f"{EVALS}/run",
        json = {"checkpoint_path": label, "model_source": "local", "task": RECIPE_TASK, "recipe": spec},
    )
    done = done if isinstance(done, dict) else {}
    details = done.get("details") or {}
    if not done.get("success"):
        raise FlowNodeError(str(done.get("message") or "The benchmark did not finish"))
    found = details.get("recipe") or {}
    run_ids = found.get("run_ids") or {}
    candidate = await _samples(ctx, run_ids.get("candidate"))
    base = {s.get("doc_id"): s for s in await _samples(ctx, run_ids.get("baseline"))}
    results, failures = [], []
    for s in candidate:
        index = s.get("doc_id")
        row = rows[index] if isinstance(index, int) and 0 <= index < len(rows) else {}
        item = {
            "prompt": s.get("question"),
            "reference": s.get("target") or None,
            "output": s.get("response"),
            "passed": bool(s.get("correct")),
            "reason": s.get("raw_response") or "",
        }
        if index in base:
            item["base_output"] = base[index].get("response")
            item["base_passed"] = bool(base[index].get("correct"))
        results.append(item)
        if not item["passed"]:
            failures.append({**row, "model_output": item["output"], "reason": item["reason"]})
    summary = found.get("candidate") or {}
    baseline = found.get("baseline") or {}
    return {
        "score": summary.get("score"),
        "passed": summary.get("passed", 0),
        "scored": summary.get("scored", 0),
        "base_score": baseline.get("score") if baseline else None,
        "metric": "reference match" if scorer == "reference" else "LLM judge",
        "run_id": run_ids.get("candidate"),
        "results": results,
        "failures": failures,
    }


async def _lm_eval(ctx: NodeContext, task: str, label: str, is_lora: bool, max_rows: int) -> dict:
    from studio_mcp import loading

    if is_lora:
        raise FlowNodeError(
            "lm-eval tasks score a GGUF. Export the checkpoint to GGUF and pick it as the model, or use Recipe held-out"
        )
    try:
        _slots, status = await loading._resident(ctx.caller)
    except Exception:
        status = None
    if not isinstance(status, dict) or status.get("is_gguf") is not True:
        raise FlowNodeError("lm-eval tasks score a GGUF loaded in llama.cpp; pick a GGUF model on this node")
    ctx.progress(phase = f"Running {task} on {max_rows} samples")
    done = await ctx.call(
        "POST",
        f"{EVALS}/run",
        json = {"checkpoint_path": label, "model_source": "local", "task": task, "limit": max_rows},
    )
    done = done if isinstance(done, dict) else {}
    details = done.get("details") or {}
    if not done.get("success"):
        raise FlowNodeError(str(done.get("message") or "The benchmark did not finish"))
    run_id = details.get("run_id")
    detail = await ctx.call("GET", f"{EVALS}/runs/{quote(run_id, safe = '')}") if run_id else {}
    detail = detail if isinstance(detail, dict) else {}
    score, metric = headline(detail.get("metrics") or [])
    samples = sorted(
        (s for s in (detail.get("samples") or []) if isinstance(s, dict)), key = lambda s: s.get("doc_id", 0)
    )
    results = [
        {
            "prompt": s.get("question"),
            "reference": s.get("target"),
            "output": s.get("response"),
            "passed": bool(s.get("correct")),
            "reason": "",
        }
        for s in samples
    ]
    return {
        "score": score,
        "passed": sum(1 for r in results if r["passed"]),
        "scored": len(results),
        "base_score": None,
        "metric": metric,
        "run_id": run_id,
        "results": results,
        "failures": [
            {"question": r["prompt"], "target": r["reference"], "model_output": r["output"]}
            for r in results
            if not r["passed"]
        ],
    }


async def _as_dataset(ctx: NodeContext, rows: list[dict], name: str) -> dict:
    from core.flows.nodes.data import upload_rows

    if not rows:
        return {"kind": "dataset", "source": "local", "ref": "", "rows": 0, "columns": []}
    return await upload_rows(ctx, rows, name)


@handler("train.benchmark")
async def benchmark(ctx: NodeContext) -> dict:
    task = TASKS.get(str(ctx.params.get("task") or ""), RECIPE_TASK)
    max_rows = max(1, min(MAX_ROWS, int(ctx.params.get("max_rows") or 20)))
    label, is_lora = await _load_target(ctx)

    async def cancel() -> None:
        await ctx.call("POST", f"{EVALS}/cancel")

    ctx.on_cancel(cancel)
    if task == RECIPE_TASK:
        found = await _recipe_heldout(ctx, label, is_lora, max_rows)
    else:
        found = await _lm_eval(ctx, task, label, is_lora, max_rows)
    tag = f"benchmark-{task}-{found.get('run_id') or int(time.time())}"
    results = await _as_dataset(ctx, found["results"], f"{tag}-results")
    results.update(
        {
            "task": task,
            "model": label,
            "metric": found["metric"],
            "score": found["score"],
            "base_score": found["base_score"],
            "passed": found["passed"],
            "scored": found["scored"],
            "run_id": found["run_id"],
            "preview": [{k: _short(v) for k, v in r.items()} for r in found["results"][:MAX_ROWS]],
        }
    )
    failures = await _as_dataset(ctx, found["failures"], f"{tag}-failures")
    failures["preview"] = [{k: _short(v) for k, v in r.items()} for r in found["failures"][:MAX_ROWS]]
    return {"score": found["score"], "results": results, "failures": failures}
