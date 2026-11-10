# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Benchmark node: a front end to the Benchmarks Evals tab, so every run it starts is stored there too."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from pathlib import Path
import time
import uuid
from types import SimpleNamespace
from typing import Any, Optional
from urllib.parse import quote

from core.benchmark.recipe_heldout import (
    cell_messages,
    prompt_text,
    row_reference,
    split_answer,
)
from core.flows.nodes._common import note_loaded, resident_name
from utils.datasets.cells import cell_text
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
        ctx.benchmark_target = trained.ref
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


def _same_model(loaded: str, ref: str) -> bool:
    """Whether the loaded model is ``ref``: both sides as paths, since a checkpoint or export also
    resolves to an opaque `ref:` handle."""
    from hub.utils.host_paths import resolve_host_path_reference
    from studio_mcp import loading

    loaded = resolve_host_path_reference(loaded) or loaded
    ref = resolve_host_path_reference(ref) or ref
    return loading._same_id(loaded, ref) or (
        os.path.isabs(loaded)
        and os.path.isabs(ref)
        and os.path.realpath(loaded) == os.path.realpath(ref)
    )


async def _require_primary(ctx: NodeContext, ref: str, label: str) -> None:
    """The scoring requests reach Chat's main model. A target already kept in another slot stays
    there when loaded again, so it would not be the model scored: refuse instead."""
    from studio_mcp import loading

    try:
        _slots, status = await loading._resident(ctx.caller)
    except Exception:
        return
    serving = [s for s in status.get("serving") or [] if isinstance(s, str)]
    if len(serving) < 2:
        return
    if not _same_model(serving[0], ref):
        raise FlowNodeError(
            f"{label} is kept loaded beside Chat's main model, and Benchmark scores the main model. "
            "Unload it, or set Keep multiple models loaded to Off, then run again."
        )


async def _loaded_identity(
    ctx: Optional[NodeContext] = None,
    target: Optional[str] = None,
    label: str = "",
) -> Optional[str]:
    """What this node loaded, as the eval checks it once scoring holds the model; None if unknown.
    With ``target`` (what the node asked to load), another model in Chat's main slot fails the run:
    it was swapped in between the load and this read, and would be scored under ``label``."""
    from core.benchmark.hold import loaded_identity

    if ctx is not None and target:
        from studio_mcp import loading

        try:
            _slots, status = await loading._resident(ctx.caller)
        except Exception:
            status = {}
        serving = [s for s in status.get("serving") or [] if isinstance(s, str)]
        main = serving[0] if serving else status.get("model_identifier")
        variant = str(getattr(ctx, "benchmark_variant", "") or "").lower()
        quant = status.get("gguf_variant")
        # A picked quant reloaded at another quant of the same repo is another model too.
        swapped_quant = bool(variant) and isinstance(quant, str) and quant.lower() != variant
        if isinstance(main, str) and main and (swapped_quant or not _same_model(main, target)):
            raise FlowNodeError(
                f"Another model ({main}) was loaded after Benchmark loaded {label or target}. Run again."
            )
    try:
        return await asyncio.to_thread(loaded_identity) or None
    except Exception:
        return None


async def _load_target(ctx: NodeContext) -> tuple[str, bool]:
    """Returns (label, is_lora)."""
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
            ctx.benchmark_target = trained.ref
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
    picked = model.get("repo") if isinstance(model, dict) else None
    if picked and not _same_model(label, str(picked)):
        raise FlowNodeError(
            f"Another model ({label}) was loaded after Benchmark loaded {picked}. Run again."
        )
    # What the identity read later must still find: the model and, when one was picked, its quant.
    ctx.benchmark_target = label
    ctx.benchmark_variant = str(model.get("variant") or "") if picked else ""
    return label, await asyncio.to_thread(_is_adapter, label)


def _is_adapter(name: str) -> bool:
    """A model Chat loaded from a LoRA folder or Hub repo, so the adapter-off comparison and judging apply
    to it too. The status a flow reads names a local folder by its `ref:` handle; the folder is what is
    checked. A Hub repo Chat loaded is already in the Hub cache, which is read without a network call."""
    from hub.utils.host_paths import resolve_host_path_reference

    path = resolve_host_path_reference(name) or name
    try:
        if (Path(path) / "adapter_config.json").is_file():
            return True
    except (OSError, ValueError):
        return False
    if path != name or not re.fullmatch(r"[\w.-]+/[\w.-]+", name):
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(name, "adapter_config.json"), str)
    except Exception:
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
    detail = await ctx.call("GET", f"{EVALS}/runs/{quote(run_id, safe='')}")
    samples = detail.get("samples") if isinstance(detail, dict) else None
    return sorted(
        (s for s in samples or [] if isinstance(s, dict)), key = lambda s: s.get("doc_id", 0)
    )


ALPACA_PROMPT = "__instruction_and_input"
# The question and answer of a checkpoint whose mapping joined several columns into each.
MAPPED_PROMPT = "__mapped_question"
MAPPED_REFERENCE = "__mapped_answer"
MAPPED_SYSTEM = "__mapped_system"


def _alpaca_question(row: dict) -> Any:
    # As training reads the cells: an input of 0 or False is part of the question, not a blank.
    extra = cell_text(row.get("input")).strip()
    return f"{cell_text(row.get('instruction'))}\n\n{extra}" if extra else row.get("instruction")


def _system_prompt(ctx: NodeContext) -> str:
    """The Advanced override, else the system prompt the checkpoint was trained with."""
    own = str(ctx.params.get("system_prompt") or "")
    if own.strip():
        return own
    checkpoint = ctx.inputs.get("checkpoint")
    trained = checkpoint.get("system_prompt") if isinstance(checkpoint, dict) else None
    return trained if isinstance(trained, str) else ""


SYSTEM_GUESSES = ("system", "system_prompt")


def _conversation_column(rows: list[dict], column: Optional[str]) -> bool:
    """Whether ``column`` holds whole conversations that close on an assistant answer."""
    cell = next((r[column] for r in rows if column and r.get(column) is not None), None)
    turns = cell_messages(cell) if cell is not None else None
    return bool(turns) and bool(split_answer(turns)[1].strip())


def _system_column(ctx: NodeContext, columns: list[str]) -> Optional[str]:
    """The column whose value was each training row's system message: the one the checkpoint's
    mapping named, else one named like a system prompt (training's formatter finds it by name).
    None under the Advanced override, which is every row's system message."""
    if str(ctx.params.get("system_prompt") or "").strip():
        return None
    checkpoint = ctx.inputs.get("checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    if "system_column" in checkpoint:
        # Training's mapping said which column, if any, was the system message.
        named = checkpoint["system_column"]
        return named if isinstance(named, str) and named in columns else None
    if checkpoint.get("system_prompt"):
        # Training sent one system prompt to every row, whatever columns the rows have.
        return None
    return next((c for c in columns if c.strip().lower() in SYSTEM_GUESSES), None)


def _mapped_cells(
    row: dict,
    columns: list[str],
    labels: Any,
    join: str,
    role: str = "user",
) -> Any:
    """Several mapped columns as training joined them: one message per column (a message list's
    own turns), or their lines in one message (template mapping, Alpaca format) with label names
    for numeric labels."""
    from utils.datasets.cells import cell_turns
    from utils.datasets.dataset_utils import _extract_column_value

    if join == "turns":
        return [turn for c in columns for turn in cell_turns(row.get(c), role)]
    labels = labels if isinstance(labels, dict) else {}
    return "\n".join(_extract_column_value(row.get(c), c, labels) for c in columns)


def system_message(row: dict, columns: Any) -> str:
    """A row's own system message from one column or several (training sends each, in order)."""
    names = (
        [columns] if isinstance(columns, str) else [c for c in columns or [] if isinstance(c, str)]
    )
    return "\n\n".join(t for t in (cell_text(row.get(c)).strip() for c in names) if t)


def _asked_spec(
    prompt_col: str, reference_col: Optional[str], mapped: Optional[dict], asked: list[dict]
) -> Any:
    """How a Loop rebuilds each row's question as Benchmark asked it: from several mapped columns,
    as one column (a conversation whole, or a prompt with its assistant prefill when a reference
    column holds the answer), or "" for the Loop's own Alpaca reading."""
    if prompt_col == MAPPED_PROMPT:
        return mapped
    if prompt_col == ALPACA_PROMPT:
        return ""
    if reference_col and any(cell_messages(r.get(prompt_col)) for r in asked[:1]):
        return {"kind": "column", "column": prompt_col, "whole": False}
    return prompt_col


def mapped_question(row: dict, spec: dict) -> Any:
    """The question of a row whose checkpoint mapped several columns into it (``spec`` as the
    Benchmark node reports it to a Loop): user turns, lines, or an Alpaca instruction and input."""
    users = [c for c in spec.get("columns") or [] if isinstance(c, str)]
    inputs = [c for c in spec.get("inputs") or [] if isinstance(c, str)]
    labels, join = spec.get("labels"), spec.get("join")
    if join != "alpaca":
        return _mapped_cells(row, users, labels, "turns" if join == "turns" else "lines")
    question = _mapped_cells(row, users, labels, "lines")
    extra = _mapped_cells(row, inputs, labels, "lines").strip() if inputs else ""
    return f"{question}\n\n{extra}" if extra else question


def _labelled(rows: list[dict], columns: tuple, ctx: NodeContext) -> list[dict]:
    """The rows with the label names training gave numeric labels (Fine-tune's label mapping)."""
    checkpoint = ctx.inputs.get("checkpoint")
    labels = checkpoint.get("label_mapping") if isinstance(checkpoint, dict) else None
    named = [
        c for c in columns if c and isinstance(labels, dict) and isinstance(labels.get(c), dict)
    ]
    if not named:
        return rows
    from utils.datasets.dataset_utils import _extract_column_value

    return [
        {**r, **{c: _extract_column_value(r[c], c, labels) for c in named if r.get(c) is not None}}
        for r in rows
    ]


async def _recipe_heldout(
    ctx: NodeContext,
    label: str,
    is_lora: bool,
    max_rows: int,
    token: str,
    expected: Optional[str] = None,
) -> dict:
    rows = await _read_rows(ctx, ctx.inputs.get("dataset"), max_rows)
    columns = sorted({k for r in rows for k in r})
    # Read against every column first: training may have mapped `input` as the system message.
    system_col = _system_column(ctx, columns)
    checkpoint = ctx.inputs.get("checkpoint")
    trained_system = checkpoint.get("system_column") if isinstance(checkpoint, dict) else None
    # A column training read as the system message is never guessed as the question or the answer.
    unasked = {system_col, trained_system} - {None, ""}
    # Several system columns: training sent each, so the row's system message is all of them.
    systems = [
        c
        for c in (checkpoint.get("system_columns") if isinstance(checkpoint, dict) else None) or []
        if isinstance(c, str)
    ]
    systems = systems if len(systems) > 1 and all(c in columns for c in systems) else []
    unasked |= set(systems)
    guessable = [c for c in columns if c not in unasked]
    named_prompt = str(ctx.params.get("prompt_column") or "").strip()
    facts = checkpoint if isinstance(checkpoint, dict) else {}
    # The checkpoint's mapping names the question and answer columns, unless one is missing here.
    users = [c for c in facts.get("user_columns") or [] if isinstance(c, str)]
    inputs = [c for c in facts.get("input_columns") or [] if isinstance(c, str)]
    users = users if users and all(c in columns for c in [*users, *inputs]) else []
    # The template mapping and the Alpaca format read each cell through the label and SQuAD
    # conversions; a plain role mapping reads it as its text.
    converted = facts.get("user_join") in ("lines", "alpaca")
    # Several columns make the question (or one read through those conversions): the Loop rebuilds
    # it from this to know a row by it.
    mapped = (
        {
            "columns": users,
            "join": facts.get("user_join") or "turns",
            "inputs": inputs,
            "labels": facts.get("label_mapping"),
        }
        # One column under the template mapping or Alpaca format still reads its cell as training
        # does there (a SQuAD answer's text, a label's name).
        if len(users) > 1 or (users and inputs) or (users and converted)
        else None
    )
    answers = [c for c in facts.get("answer_columns") or [] if isinstance(c, str)]
    answers = answers if answers and all(c in columns for c in answers) else []
    asked = rows
    conversation = facts.get("conversation_column")
    # The whole-conversation column training read: its turns are the question and its closing
    # answer the reference, whatever seed columns the rows also keep.
    conversation = (
        conversation if isinstance(conversation, str) and conversation in columns else None
    )
    if not named_prompt and not users and conversation:
        prompt_col = conversation
    elif named_prompt or not users:
        prompt_col = guess_column(
            columns if named_prompt else guessable, named_prompt, PROMPT_GUESSES
        )
    elif mapped is None:
        prompt_col = users[0]
    else:
        asked = [{**r, MAPPED_PROMPT: mapped_question(r, mapped)} for r in rows]
        prompt_col = MAPPED_PROMPT
    if not prompt_col and not named_prompt:
        # A whole-conversation SFT row (`messages`, `conversations`): its turns are the question
        # and its closing assistant answer the reference.
        prompt_col = next((c for c in guessable if _conversation_column(rows, c)), None)
    if not prompt_col:
        raise FlowNodeError("Set the prompt column under Advanced: no column looks like a prompt")
    if (
        prompt_col == "instruction"
        and "input" in columns
        and system_col != "input"
        and not named_prompt
        and not users
    ):
        # An Alpaca row's question is its instruction plus its input, as training formats it.
        asked = [{**r, ALPACA_PROMPT: _alpaca_question(r)} for r in rows]
        prompt_col = ALPACA_PROMPT
        columns = [c for c in columns if c != "input"]
    named_reference = str(ctx.params.get("reference_column") or "").strip()
    if not named_reference and conversation and prompt_col == conversation:
        reference_col = None
    elif not named_reference and len(answers) == 1 and not converted:
        reference_col = answers[0]
    elif not named_reference and answers:
        labels = facts.get("label_mapping")
        # As training read them: a plain role mapping's assistant turns, else their lines.
        join = "lines" if converted else "turns"
        asked = [
            {**r, MAPPED_REFERENCE: _mapped_cells(r, answers, labels, join, "assistant")}
            for r in asked
        ]
        reference_col = MAPPED_REFERENCE
    else:
        reference_col = guess_column(
            [c for c in columns if c != prompt_col and (named_reference or c not in unasked)],
            named_reference,
            REFERENCE_GUESSES,
        )
    if systems and system_col:
        asked = [{**r, MAPPED_SYSTEM: system_message(r, systems)} for r in asked]
        system_col = MAPPED_SYSTEM
    asked = _labelled(asked, (prompt_col, reference_col), ctx)
    scorer = SCORERS.get(str(ctx.params.get("scorer") or ""), "reference")
    if scorer == "reference" and not reference_col and not _conversation_column(asked, prompt_col):
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
        "system_override": bool(str(ctx.params.get("system_prompt") or "").strip()),
        "system_column": system_col,
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
            **({"expected_model": expected} if expected else {}),
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
                "question": prompt_text(asked[i].get(prompt_col)),
                "target": row_reference(asked[i], prompt_col, reference_col) or None,
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
            failures.append(
                {
                    **row,
                    "model_output": item["output"],
                    # The question as asked, when no one column of the row holds it.
                    **({"scored_prompt": item["prompt"]} if prompt_col == MAPPED_PROMPT else {}),
                    # What the answer was scored against: the label name, a structured answer's text.
                    "scored_reference": item["reference"],
                    "reason": item["reason"],
                }
            )
    summary = found.get("candidate") or {}
    baseline = found.get("baseline") or {}
    return {
        "score": summary.get("score"),
        "passed": summary.get("passed", 0),
        "scored": summary.get("scored", 0),
        "errored": len(errored),
        "base_score": baseline.get("score") if baseline else None,
        "metric": "reference match" if scorer == "reference" else "LLM judge",
        "run_id": run_ids.get("candidate"),
        "results": results,
        "failures": failures,
        # The column asked, so a Loop reads failures and test questions as they were scored; an
        # Alpaca row's question is its instruction plus input, which the Loop rebuilds itself.
        "prompt_column": _asked_spec(prompt_col, reference_col, mapped, asked),
        # Rows sharing a question can ask different tasks through their own system message.
        "system_column": systems if system_col == MAPPED_SYSTEM else (system_col or ""),
    }


async def _lm_eval(
    ctx: NodeContext,
    task: str,
    label: str,
    is_lora: bool,
    max_rows: int,
    token: str,
    expected: Optional[str] = None,
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
            **({"expected_model": expected} if expected else {}),
        },
    )
    done = done if isinstance(done, dict) else {}
    details = done.get("details") or {}
    if not done.get("success"):
        raise FlowNodeError(str(done.get("message") or "The benchmark did not finish"))
    run_id = details.get("run_id")
    detail = await ctx.call("GET", f"{EVALS}/runs/{quote(run_id, safe='')}") if run_id else {}
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


async def _settle_loads(ctx: NodeContext, timeout: float = 1800.0) -> None:
    """Wait for this account's model loads still running (a stopped Benchmark's own among them), so the
    eval slot it held is not freed while one could still swap Chat's model."""
    from studio_mcp import loading

    account = getattr(ctx.caller, "account_id", None)
    tasks = {
        job.task for job in loading.LOADS.running(account) if getattr(job, "task", None) is not None
    }
    if tasks:
        with contextlib.suppress(Exception):
            await asyncio.wait(tasks, timeout = timeout)


@handler("train.benchmark")
async def benchmark(ctx: NodeContext) -> dict:
    task = TASKS.get(str(ctx.params.get("task") or ""), RECIPE_TASK)
    max_rows = max(1, min(MAX_ROWS, int(ctx.params.get("max_rows") or 20)))
    # Stop cancels only the run this node starts, never another one of the account's.
    token = uuid.uuid4().hex
    loaded = False

    async def cancel() -> None:
        # While the model loads, Stop leaves the slot held: the load runs on as its own job and would
        # swap the model under the next eval. The finally below frees it once that load has ended.
        if loaded:
            await ctx.call("POST", f"{EVALS}/cancel", params = {"token": token})

    # The eval slot is held before the model loads: loading while another eval runs would swap
    # the model that run is scoring.
    await ctx.call("POST", f"{EVALS}/reserve", params = {"token": token})
    ctx.on_cancel(cancel)
    try:
        label, is_lora = await _load_target(ctx)
        loaded = True
        # Read now: the eval fails if, by the time scoring holds the model, another one is loaded.
        expected = await _loaded_identity(ctx, getattr(ctx, "benchmark_target", None), label)
        if task == RECIPE_TASK:
            found = await _recipe_heldout(ctx, label, is_lora, max_rows, token, expected)
        else:
            found = await _lm_eval(ctx, task, label, is_lora, max_rows, token, expected)
    finally:
        if not loaded:
            await _settle_loads(ctx)
        # Frees a reservation no run took over; a finished run's slot is already free. Sent as a cancel
        # hook's call, which a stopped run still lets through.
        from core.flows.engine import _run_cancel_hook
        await _run_cancel_hook(lambda: ctx.call("POST", f"{EVALS}/cancel", params = {"token": token}))
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
            "errored": found.get("errored", 0),
            "run_id": found["run_id"],
            "preview": [{k: _short(v) for k, v in r.items()} for r in found["results"][:MAX_ROWS]],
        }
    )
    failures = await _as_dataset(ctx, found["failures"], f"{tag}-failures")
    failures["preview"] = [
        {k: _short(v) for k, v in r.items()} for r in found["failures"][:MAX_ROWS]
    ]
    if "prompt_column" in found:
        failures["prompt_column"] = found["prompt_column"]
    if found.get("system_column"):
        failures["system_column"] = found["system_column"]
    return {"score": _percent(found["score"]), "results": results, "failures": failures}
