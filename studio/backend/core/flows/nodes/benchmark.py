# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Benchmark node: a front end to the Benchmarks Evals tab, so every run it starts is stored there too."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
import time
import uuid
from types import SimpleNamespace
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
REFERENCE_GUESSES = (
    "response",
    "output",
    "answer",
    "completion",
    "reference",
    "target",
    "assistant",
)


def guess_column(columns: list[str], wanted: str, guesses: tuple[str, ...]) -> Optional[str]:
    if wanted:
        if wanted not in columns:
            raise FlowNodeError(
                f"The rows have no {wanted!r} column (they have {', '.join(columns)})"
            )
        return wanted
    lower = {c.lower(): c for c in columns}
    return next((lower[g] for g in guesses if g in lower), None)


def headline(metrics: list[dict]) -> tuple[Optional[float], str]:
    """The metric the Evals tab leads with, so the node and the tab show the same number."""
    from core.benchmark.parse import pick_default_metric

    name = pick_default_metric(metrics)
    found = next((m for m in metrics if m.get("name") == name), None)
    if found is None or not isinstance(found.get("score"), (int, float)):
        return None, name
    return float(found["score"]), name


def _short(value: Any) -> Any:
    if isinstance(value, str) and len(value) > PREVIEW_CHARS:
        return value[:PREVIEW_CHARS] + "…"
    return value


async def _load_export(ctx: NodeContext, model: dict) -> tuple[str, bool]:
    """Load an Export Model output by its export name (a GGUF loads as its folder plus quant)."""
    from studio_mcp import loading, trained_models

    from core.flows.nodes._common import FLOW_CHAT_CONTEXT

    name = str(model["repo"])
    deadline = time.monotonic() + 3600
    ctx.set_status("loading_model", f"Loading {name}")
    try:
        try:
            trained = await trained_models.resolve_by(
                deadline,
                ctx.caller,
                model = None,
                checkpoint = None,
                export = name,
                kind = "llm",
                variant = model.get("variant") or None,
            )
        except Exception as exc:
            trained, missing = None, exc
        else:
            missing = None
        if trained is None:
            flat = await asyncio.to_thread(_flat_export, name)
            if flat is None:
                raise missing or FlowNodeError(f"No export named {name}")
            trained = SimpleNamespace(ref = flat[0], is_lora = flat[1], gguf_variant = None)
        await loading.load_llm(
            ctx.caller,
            None,
            model = trained.ref,
            variant = trained.gguf_variant,
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
    await _require_primary(ctx, trained.ref, name)
    ctx.set_status("running")
    return name, bool(trained.is_lora)


def _flat_export(name: str) -> Optional[tuple[str, bool]]:
    """(folder, is_lora) of a merged or LoRA export Export Model wrote as one folder under exports,
    a layout the export listing reads only for GGUF."""
    from utils.paths.storage_roots import resolve_export_dir

    try:
        folder = resolve_export_dir(name)
    except (OSError, ValueError):
        return None
    if (folder / "adapter_config.json").is_file():
        return str(folder), True
    if (folder / "config.json").is_file():
        return str(folder), False
    return None


async def _require_primary(ctx: NodeContext, ref: str, label: str) -> None:
    """The scoring requests reach Chat's main model. A target already kept in another slot stays
    there when loaded again, so it would not be the model scored: refuse instead."""
    from hub.utils.host_paths import resolve_host_path_reference
    from studio_mcp import loading

    try:
        _slots, status = await loading._resident(ctx.caller)
    except Exception:
        return
    serving = [s for s in status.get("serving") or [] if isinstance(s, str)]
    if len(serving) < 2:
        return
    main = resolve_host_path_reference(serving[0]) or serving[0]
    same = loading._same_id(main, ref) or (
        os.path.isabs(main)
        and os.path.isabs(ref)
        and os.path.realpath(main) == os.path.realpath(ref)
    )
    if not same:
        raise FlowNodeError(
            f"{label} is kept loaded beside Chat's main model, and Benchmark scores the main model. "
            "Unload it, or set Keep multiple models loaded to Off, then run again."
        )


async def _load_target(ctx: NodeContext) -> tuple[str, bool]:
    """Load what is being scored; (label, is_lora)."""
    wired = ctx.inputs.get("model")
    if isinstance(wired, dict) and wired.get("repo"):
        return await _load_export(ctx, wired)
    checkpoint = ctx.inputs.get("checkpoint")
    name = checkpoint.get("name") if isinstance(checkpoint, dict) else None
    if name:
        from studio_mcp import loading, trained_models

        from core.flows.nodes._common import FLOW_CHAT_CONTEXT

        deadline = time.monotonic() + 3600
        ctx.set_status("loading_model", f"Loading {name}")
        try:
            trained = await trained_models.resolve_by(
                deadline,
                ctx.caller,
                model = None,
                checkpoint = name,
                export = None,
                kind = "llm",
                variant = None,
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
        await _require_primary(ctx, trained.ref, name)
        ctx.set_status("running")
        return name, bool(trained.is_lora)
    model = ctx.params.get("model")
    await ctx.ensure_model("chat", model if isinstance(model, dict) and model.get("repo") else None)
    kept = getattr(ctx, "chat_model", None)
    if kept:
        # The scoring requests reach Chat's main model, so a model kept beside it would not be the one scored.
        raise FlowNodeError(
            f"Benchmark scores Chat's main model, and {kept} is kept loaded beside it. Unload the main "
            "model, or set Keep multiple models loaded to Off, then run again."
        )
    label = await resident_name(ctx, "chat")
    if not label:
        raise FlowNodeError(
            "Wire a Fine-tune checkpoint in, pick a model, or load one in Chat first"
        )
    return label, await asyncio.to_thread(_is_adapter, label)


def _is_adapter(name: str) -> bool:
    """A model Chat loaded from a LoRA folder, so the adapter-off comparison and judging apply to it too.
    The status a flow reads names a local folder by its `ref:` handle; the folder is what is checked."""
    from hub.utils.host_paths import resolve_host_path_reference

    path = resolve_host_path_reference(name) or name
    try:
        return (Path(path) / "adapter_config.json").is_file()
    except (OSError, ValueError):
        return False


def _head(dataset: dict, token: Any, limit: int) -> list[dict]:
    """The first rows of a dataset, streamed so a big file or split is never read whole."""
    from datasets import load_dataset

    if dataset.get("source") == "hf":
        ref, subset = str(dataset["ref"]), dataset.get("subset") or None
        split = str(dataset.get("split") or "train").replace(" ", "")
        # Streaming takes only plain split names: `test[10:100]` becomes a bounded read of test.
        sliced = re.fullmatch(r"([\w.-]+)\[(\d*):(\d*)\]", split)
        if sliced:
            lo = int(sliced.group(2) or 0)
            if sliced.group(3):
                limit = max(0, min(limit, int(sliced.group(3)) - lo))
            stream = load_dataset(ref, subset, split = sliced.group(1), token = token, streaming = True)
            return list((stream.skip(lo) if lo else stream).take(limit))
        if "[" in split or "+" in split:
            # Percent slices and joined splits: read the way training does.
            loaded = load_dataset(ref, subset, split = split, token = token)
            return loaded.select(range(min(limit, len(loaded)))).to_list()
        stream = load_dataset(ref, subset, split = split, token = token, streaming = True)
    else:
        from core.flows.nodes.data import local_seed_path
        from core.flows.nodes.training import LOADERS

        path = local_seed_path(dataset.get("ref"))
        builder = next((b for ext, b in LOADERS.items() if path.endswith(ext)), None)
        if builder is None:
            raise FlowNodeError("Benchmark reads Parquet, CSV or JSON rows")
        from utils.datasets.cells import csv_as_text_kwargs

        # CSV cells stay text as Fine-tune reads them ("00501" keeps its zeros, "NA" stays a value).
        stream = load_dataset(
            builder,
            data_files = path,
            split = "train",
            streaming = True,
            **csv_as_text_kwargs([path]),
        )
    return list(stream.take(limit))


async def _read_rows(ctx: NodeContext, dataset: Any, limit: int) -> list[dict]:
    from core.flows.nodes.data import recipe_artifact, recipe_rows
    from core.training.account_jobs import account_hf_token

    if not isinstance(dataset, dict) or dataset.get("kind") != "dataset":
        raise FlowNodeError(
            "Recipe held-out scores a dataset: wire Fine-tune's Held-out rows or a recipe in"
        )
    if dataset.get("source") == "recipe" and not recipe_artifact(dataset.get("ref")):
        rows = (await recipe_rows(ctx, str(dataset["ref"]), limit = limit))[:limit]
    else:
        token = account_hf_token(getattr(ctx.caller, "hf_token", None))
        try:
            rows = await asyncio.to_thread(_head, dataset, token, limit)
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
    return sorted(
        (s for s in samples or [] if isinstance(s, dict)), key = lambda s: s.get("doc_id", 0)
    )


ALPACA_PROMPT = "__instruction_and_input"


def _alpaca_question(row: dict) -> Any:
    extra = str(row.get("input") or "").strip()
    return f"{row.get('instruction') or ''}\n\n{extra}" if extra else row.get("instruction")


def _system_prompt(ctx: NodeContext) -> str:
    """The Advanced override, else the system prompt the checkpoint was trained with."""
    own = str(ctx.params.get("system_prompt") or "")
    if own.strip():
        return own
    checkpoint = ctx.inputs.get("checkpoint")
    trained = checkpoint.get("system_prompt") if isinstance(checkpoint, dict) else None
    return trained if isinstance(trained, str) else ""


async def _recipe_heldout(
    ctx: NodeContext, label: str, is_lora: bool, max_rows: int, token: str
) -> dict:
    rows = await _read_rows(ctx, ctx.inputs.get("dataset"), max_rows)
    columns = sorted({k for r in rows for k in r})
    prompt_col = guess_column(
        columns, str(ctx.params.get("prompt_column") or "").strip(), PROMPT_GUESSES
    )
    if not prompt_col:
        raise FlowNodeError("Set the prompt column under Advanced: no column looks like a prompt")
    if (
        prompt_col == "instruction"
        and "input" in columns
        and not str(ctx.params.get("prompt_column") or "").strip()
    ):
        # An Alpaca row's question is its instruction plus its input, as training formats it.
        asked = [{**r, ALPACA_PROMPT: _alpaca_question(r)} for r in rows]
        prompt_col = ALPACA_PROMPT
        columns = [c for c in columns if c != "input"]
    else:
        asked = rows
    reference_col = guess_column(
        [c for c in columns if c != prompt_col],
        str(ctx.params.get("reference_column") or "").strip(),
        REFERENCE_GUESSES,
    )
    scorer = SCORERS.get(str(ctx.params.get("scorer") or ""), "reference")
    if scorer == "reference" and not reference_col:
        raise FlowNodeError(
            "Reference match needs a reference column; set it under Advanced or use LLM judge"
        )
    with_base = is_lora and ctx.params.get("compare_base") is not False
    spec = {
        "rows": asked,
        "max_rows": max_rows,
        "prompt_column": prompt_col,
        "reference_column": reference_col,
        "system_prompt": _system_prompt(ctx),
        "scorer": scorer,
        "rubric": str(ctx.params.get("rubric") or ""),
        "model_alias": "candidate",
        "baseline": {"mode": "adapter_off" if with_base else "none"},
        # The judge is the loaded model; with an adapter loaded the route turns it off for judging.
        "judge_alias": "candidate" if scorer == "judge" else None,
        "local_adapter": is_lora,
        "model_providers": [{"name": "local", "is_local": True}],
        "model_configs": [{"alias": "candidate", "provider": "local", "model": ""}],
    }
    ctx.progress(
        phase = f"Scoring {len(rows)} rows" + (" with the adapter on and off" if with_base else "")
    )
    done = await ctx.call(
        "POST",
        f"{EVALS}/run",
        json = {
            "checkpoint_path": label,
            "model_source": "local",
            "task": RECIPE_TASK,
            "recipe": spec,
            "run_token": token,
        },
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
    errored = {
        e.get("index"): e.get("error") for e in found.get("errored") or [] if isinstance(e, dict)
    }
    candidate = sorted(
        candidate
        + [
            {
                "doc_id": i,
                "question": asked[i].get(prompt_col),
                "target": rows[i].get(reference_col) if reference_col else None,
                "raw_response": err,
            }
            for i, err in errored.items()
            if isinstance(i, int) and 0 <= i < len(rows)
        ],
        key = lambda s: s.get("doc_id", 0),
    )
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


async def _lm_eval(
    ctx: NodeContext, task: str, label: str, is_lora: bool, max_rows: int, token: str
) -> dict:
    from studio_mcp import loading

    if is_lora:
        raise FlowNodeError(
            "lm-eval tasks score a GGUF. Wire Fine-tune into Export Model (GGUF) and Export Model into this "
            "Benchmark's Model, or use Recipe held-out"
        )
    try:
        _slots, status = await loading._resident(ctx.caller)
    except Exception:
        status = None
    if not isinstance(status, dict) or status.get("is_gguf") is not True:
        raise FlowNodeError(
            "lm-eval tasks score a GGUF loaded in llama.cpp; pick a GGUF model on this node"
        )
    ctx.progress(phase = f"Running {task} on {max_rows} samples")
    done = await ctx.call(
        "POST",
        f"{EVALS}/run",
        json = {
            "checkpoint_path": label,
            "model_source": "local",
            "task": task,
            "limit": max_rows,
            "run_token": token,
        },
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
        (s for s in (detail.get("samples") or []) if isinstance(s, dict)),
        key = lambda s: s.get("doc_id", 0),
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


def _percent(score: Any) -> Optional[float]:
    """The 0..1 pass rate as a percent with two decimals, the unit the card and a Loop's target use."""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    return round(score * 100, 2)


@handler("train.benchmark")
async def benchmark(ctx: NodeContext) -> dict:
    task = TASKS.get(str(ctx.params.get("task") or ""), RECIPE_TASK)
    max_rows = max(1, min(MAX_ROWS, int(ctx.params.get("max_rows") or 20)))
    label, is_lora = await _load_target(ctx)

    # Stop cancels only the run this node starts, never another one of the account's.
    token = uuid.uuid4().hex

    async def cancel() -> None:
        await ctx.call("POST", f"{EVALS}/cancel", params = {"token": token})

    ctx.on_cancel(cancel)
    if task == RECIPE_TASK:
        found = await _recipe_heldout(ctx, label, is_lora, max_rows, token)
    else:
        found = await _lm_eval(ctx, task, label, is_lora, max_rows, token)
    tag = f"benchmark-{task}-{found.get('run_id') or int(time.time())}"
    results = await _as_dataset(ctx, found["results"], f"{tag}-results")
    results.update(
        {
            "task": task,
            "model": label,
            "metric": found["metric"],
            "score": _percent(found["score"]),
            "base_score": _percent(found["base_score"]),
            "passed": found["passed"],
            "scored": found["scored"],
            "run_id": found["run_id"],
            "preview": [{k: _short(v) for k, v in r.items()} for r in found["results"][:MAX_ROWS]],
        }
    )
    failures = await _as_dataset(ctx, found["failures"], f"{tag}-failures")
    failures["preview"] = [
        {k: _short(v) for k, v in r.items()} for r in found["failures"][:MAX_ROWS]
    ]
    return {"score": _percent(found["score"]), "results": results, "failures": failures}
