# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training nodes: LoRA fine-tuning, export and chatting with a checkpoint, through the routes the MCP tools use."""

from __future__ import annotations

import asyncio
import json
import random
import secrets
import time
from typing import Any, Optional

from core.flows.nodes._common import FLOW_CHAT_CONTEXT, as_text, note_loaded, poll_job, strip_think
from core.flows.nodes.data import recipe_artifact
from core.flows.registry import FlowNodeError, NodeContext, handler

POLL_S = 5.0
BUSY_WAIT_S = 180.0
TRAINING_TYPES = {"lora": "LoRA/QLoRA", "full": "Full Finetuning"}
TASK_FLAGS = {
    "vision": {"is_dataset_image": True},
    "audio": {"is_dataset_audio": True},
    "embedding": {"is_embedding": True},
}
OPTIONAL_INTS = (
    "lora_alpha",
    "batch_size",
    "gradient_accumulation_steps",
    "warmup_steps",
    "vision_image_size",
    "dataset_slice_end",
)
HOLDOUT_MAX_ROWS = 200_000
HISTORY_POINTS = 80
LOADERS = {".parquet": "parquet", ".csv": "csv", ".json": "json", ".jsonl": "json"}
OBJECTIVES = ("sft", "grpo", "dpo", "orpo")
GRPO_SETTINGS = {
    "num_generations": ("grpo_num_generations", int),
    "max_completion_length": ("grpo_max_completion_length", int),
    "temperature": ("grpo_temperature", float),
    "grpo_variant": ("grpo_variant", str),
}
EXPORT_QUANTS = {"q4_k_m": "Q4_K_M", "q5_k_m": "Q5_K_M", "q8_0": "Q8_0", "f16": "F16"}


@handler("train.finetune")
async def finetune(ctx: NodeContext) -> dict:
    from studio_mcp.tools.jobs import TRAINING_STATUS, _run_folder
    from studio_mcp.tools.status import _training

    dataset = ctx.inputs["dataset"]
    if not isinstance(dataset, dict) or dataset.get("kind") != "dataset":
        raise FlowNodeError("Fine-tune needs a dataset")
    config: dict[str, Any] = {
        "model_name": str(ctx.params.get("base_model") or "").strip(),
        "training_type": TRAINING_TYPES.get(str(ctx.params.get("training_type")), "LoRA/QLoRA"),
        "format_type": str(ctx.params.get("format_type") or "auto"),
        "max_steps": int(ctx.params.get("max_steps") or 60),
        "learning_rate": str(ctx.params.get("learning_rate") or "2e-4"),
        "lora_r": int(ctx.params.get("lora_r") or 16),
        "max_seq_length": int(ctx.params.get("max_seq_length") or 2048),
        "random_seed": ctx.seed % (2**31),
        "start_request_id": f"flow-{secrets.token_hex(8)}",
    }
    if not config["model_name"]:
        raise FlowNodeError("Name the base model to fine-tune")
    config.update(TASK_FLAGS.get(str(ctx.params.get("task") or "text"), {}))
    if ctx.params.get("num_epochs") not in (None, ""):
        config["num_epochs"] = int(ctx.params["num_epochs"])
        config["max_steps"] = None
    for key in OPTIONAL_INTS:
        if ctx.params.get(key) not in (None, ""):
            config[key] = int(ctx.params[key])
    if ctx.params.get("dataset_streaming"):
        config["dataset_streaming"] = True
    if ctx.params.get("load_in_4bit") is False:
        config["load_in_4bit"] = False
    if ctx.params.get("train_on_completions"):
        config["train_on_completions"] = True
    if ctx.params.get("weight_decay") not in (None, ""):
        config["weight_decay"] = float(ctx.params["weight_decay"])
    mapping = ctx.params.get("custom_format_mapping")
    prompt_col = str(ctx.params.get("prompt_column") or "").strip()
    response_col = str(ctx.params.get("response_column") or "").strip()
    objective = str(ctx.params.get("objective") or "sft")
    if objective not in OBJECTIVES:
        raise FlowNodeError(f"Unknown objective {objective!r}")
    rewards: list[dict] = []
    if isinstance(mapping, dict) and mapping:
        config["custom_format_mapping"] = mapping
    elif objective != "sft":
        roles = rl_roles(ctx.params, objective)
        if roles:
            config["custom_format_mapping"] = roles
    elif prompt_col and prompt_col == response_col:
        raise FlowNodeError("The prompt and the response need different columns")
    elif prompt_col and response_col:
        config["custom_format_mapping"] = {prompt_col: "user", response_col: "assistant"}
    elif prompt_col or response_col:
        raise FlowNodeError("Set both the prompt and the response column, or leave both blank")
    if objective != "sft":
        rewards = rl_config(ctx.params, objective, config)
    holdout = None
    percent = int(ctx.params.get("holdout_percent") or 0)
    if percent > 0:
        dataset, holdout = await holdout_split(
            ctx, dataset, percent, int(ctx.params.get("holdout_seed") or 0)
        )
        # The split already covers only the sliced rows; slicing the train half again would shrink it.
        config.pop("dataset_slice_end", None)
    if dataset.get("source") == "hf":
        config["hf_dataset"] = dataset["ref"]
        if dataset.get("subset"):
            config["subset"] = dataset["subset"]
        if dataset.get("split"):
            config["train_split"] = dataset["split"]
    elif dataset.get("source") == "local":
        config["local_datasets"] = [dataset["ref"]]
    elif dataset.get("source") == "recipe" and recipe_artifact(dataset.get("ref")):
        config["local_datasets"] = [dataset["ref"]]
    elif dataset.get("source") == "recipe":
        from core.flows.nodes.data import recipe_rows, upload_rows

        rows = await recipe_rows(ctx, str(dataset["ref"]))
        if not rows:
            raise FlowNodeError("The Data Recipe produced no rows to train on")
        saved = await upload_rows(ctx, rows, f"recipe-{dataset['ref']}")
        config["local_datasets"] = [saved["ref"]]
    else:
        raise FlowNodeError("Fine-tune takes a Hub dataset, a Data Recipe or Rows to Dataset")
    waited = 0.0
    while True:
        started = await ctx.call("POST", "/api/train/start", json = config)
        if not isinstance(started, dict) or started.get("status") != "error":
            break
        # A refused start answers 200 with the running job's id (never poll it as ours); a just-finished run stays busy briefly.
        if "already active" not in str(started.get("error") or "") or waited >= BUSY_WAIT_S:
            raise FlowNodeError(
                str(started.get("message") or started.get("error") or "Training did not start")
            )
        ctx.progress(phase = "Waiting for the previous training run to finish")
        await asyncio.sleep(POLL_S)
        waited += POLL_S
        ctx.check_cancelled()
    job_id = started.get("job_id") if isinstance(started, dict) else None
    if not job_id:
        raise FlowNodeError("Unsloth Studio did not start training")

    async def cancel() -> None:
        await ctx.call("POST", "/api/train/stop", json = {"save": True, "expected_job_id": job_id})

    history: dict[str, list] = {"loss": [], "reward": []}
    reward_max = reward_ceiling(rewards) if rewards else None

    async def status() -> Any:
        payload = await ctx.call("GET", "/api/train/status")
        slot = _training(payload if isinstance(payload, dict) else {})
        if slot.step and (not slot.job_id or slot.job_id == job_id):
            history["loss"] = (
                status_losses(payload) or await loss_history(ctx, job_id) or history["loss"]
            )
            history["reward"] = reward_history(payload) or history["reward"]
            ctx.partial(
                "metrics",
                {
                    "loss": slot.loss,
                    "steps": slot.step,
                    "total": slot.total_steps,
                    "loss_history": history["loss"],
                    **reward_metrics(history["reward"], reward_max),
                },
                step = slot.step,
                total = slot.total_steps,
            )
        return payload

    def parse(payload: Any) -> tuple[str, Optional[dict], Optional[str]]:
        slot = _training(payload if isinstance(payload, dict) else {})
        if slot.job_id and slot.job_id != job_id:
            raise FlowNodeError("Another training run replaced this one")
        status = TRAINING_STATUS.get(slot.phase, "running")
        progress = None
        if slot.progress_percent is not None:
            progress = {
                "fraction": slot.progress_percent / 100.0,
                "phase": slot.phase,
                "step": slot.step,
                "total": slot.total_steps,
            }
        return status, progress, slot.error or f"Training {status}"

    payload = await poll_job(
        ctx,
        status = status,
        parse = parse,
        cancel = cancel,
        interval = POLL_S,
    )
    slot = _training(payload if isinstance(payload, dict) else {})
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    run, _ = await _run_folder(ctx.caller, details.get("output_dir"))
    if not run:
        raise FlowNodeError("Training finished but its run folder was not found")
    metrics = {
        "loss": slot.loss,
        "steps": slot.step,
        "total": slot.total_steps,
        "job_id": job_id,
        "loss_history": status_losses(payload)
        or await loss_history(ctx, job_id)
        or history["loss"],
        **reward_metrics(reward_history(payload) or history["reward"], reward_max),
    }
    if objective != "sft":
        metrics["objective"] = objective
    if holdout:
        metrics["train_rows"] = dataset.get("rows")
        metrics["holdout_rows"] = holdout["rows"]
    return {
        "checkpoint": {
            "kind": "checkpoint",
            "name": run,
            "base_model": config["model_name"],
        },
        "metrics": metrics,
        "holdout": holdout,
    }


def rl_roles(params: dict, objective: str) -> dict:
    """Column -> RL role for the columns the node names; Studio detects the rest by name."""
    named = {"prompt": params.get("prompt_column")}
    if objective == "grpo":
        named["answer"] = params.get("answer_column")
    else:
        named["chosen"] = params.get("chosen_column")
        named["rejected"] = params.get("rejected_column")
    return {str(col).strip(): role for role, col in named.items() if str(col or "").strip()}


def rl_config(params: dict, objective: str, config: dict) -> list[dict]:
    """The /api/train/start RL fields; returns the GRPO rewards it picked."""
    from core.flows.specs import SPECS_BY_TYPE

    # A node saved before these params existed (or built by notebook import or the API) has no
    # rewards or system prompt; without the format prompt GRPO's rewards never fire.
    defaults = {p["id"]: p["default"] for p in SPECS_BY_TYPE["train.finetune"]["params"]}
    params = {**defaults, **params}
    config["objective"] = objective
    if params.get("beta") not in (None, ""):
        config["rl_beta"] = float(params["beta"])
    if objective != "grpo":
        return []
    rewards = [
        {"name": str(r.get("name") or "").strip(), "weight": float(r.get("weight", 1.0))}
        for r in params.get("rewards") or []
        if isinstance(r, dict) and str(r.get("name") or "").strip()
    ]
    if not rewards:
        raise FlowNodeError("GRPO needs at least one reward")
    config["grpo_rewards"] = rewards
    for param, (field, cast) in GRPO_SETTINGS.items():
        if params.get(param) not in (None, ""):
            config[field] = cast(params[param])
    if str(params.get("system_prompt") or "").strip():
        config["rl_system_prompt"] = str(params["system_prompt"])
    return rewards


def _rule_best(rule: dict) -> tuple[float, float]:
    scores = [v for v in (rule.get("score") or {}).values()]
    scores += [b["score"] for b in rule.get("bands") or []]
    scores += [rule[k] for k in ("missing", "else") if k in rule]
    return min(scores), max(scores)


def reward_ceiling(rewards: list[dict]) -> Optional[float]:
    """Best summed reward a completion can earn with these weights; None if a reward is unknown."""
    from core.training.rewards import RewardError, get_reward

    total = 0.0
    for r in rewards:
        try:
            lo, hi = _rule_best(get_reward(r["name"])["rule"])
        except (RewardError, KeyError, ValueError):
            return None
        total += r["weight"] * (hi if r["weight"] >= 0 else lo)
    return round(total, 4)


def reward_history(payload: Any) -> list:
    rl = ((payload or {}).get("metric_history") or {}).get("rl") if isinstance(payload, dict) else None
    if not isinstance(rl, list):
        return []
    return downsample(
        [round(float(e["reward"]), 4) for e in rl if isinstance(e, dict) and isinstance(e.get("reward"), (int, float))]
    )


def reward_metrics(history: list, ceiling: Optional[float]) -> dict:
    if not history:
        return {}
    return {
        "reward": history[-1],
        "reward_start": history[0],
        "reward_history": history,
        "reward_max": ceiling,
    }


def downsample(values: list, points: int = HISTORY_POINTS) -> list:
    if len(values) <= points:
        return list(values)
    stride = len(values) / points
    return [values[min(len(values) - 1, round(i * stride))] for i in range(points - 1)] + [
        values[-1]
    ]


def status_losses(payload: Any) -> list:
    """The loss history /api/train/status already carries, so a poll needs no second request."""
    found = payload.get("metric_history") if isinstance(payload, dict) else None
    losses = found.get("loss") if isinstance(found, dict) else None
    if not isinstance(losses, list):
        return []
    return downsample([round(float(v), 4) for v in losses if isinstance(v, (int, float))])


async def loss_history(ctx: NodeContext, job_id: str) -> list:
    try:
        found = await ctx.call("GET", "/api/train/metrics", params = {"expected_job_id": job_id})
    except FlowNodeError:
        return []
    losses = found.get("loss_history") if isinstance(found, dict) else None
    if not isinstance(losses, list):
        return []
    return downsample([round(float(v), 4) for v in losses if isinstance(v, (int, float))])


def holdout_row_ids(total: int, percent: int, seed: int) -> list[int]:
    """Same seeded split as the Data Recipes Train card: rounded, at least one row, never all of them."""
    if percent <= 0 or total <= 1:
        return []
    count = min(total - 1, max(1, round(total * percent / 100)))
    order = list(range(total))
    random.Random(seed).shuffle(order)
    return sorted(order[:count])


def _too_many(rows: int) -> FlowNodeError:
    return FlowNodeError(
        f"{rows} rows is too many to hold out from here; set Hold out % to 0 or use a smaller slice"
    )


def _hub_rows(dataset: dict, token: Any, end: Optional[int]) -> Any:
    """A Hub split, without downloading more than the hold-out can use: a slice streams just its
    rows, and a split the Hub says is over the cap is refused before anything is fetched."""
    from datasets import Dataset, load_dataset, load_dataset_builder

    ref, subset = str(dataset["ref"]), dataset.get("subset") or None
    split = dataset.get("split") or "train"
    if end is not None:
        if end + 1 > HOLDOUT_MAX_ROWS:
            raise _too_many(end + 1)
        streamed = load_dataset(ref, subset, split = split, token = token, streaming = True)
        return Dataset.from_list(list(streamed.take(end + 1)))
    try:
        known = load_dataset_builder(ref, subset, token = token).info.splits[split].num_examples
    except Exception:
        known = None
    if known and known > HOLDOUT_MAX_ROWS:
        raise _too_many(known)
    return load_dataset(ref, subset, split = split, token = token)


def _load_rows(dataset: dict, token: Any, end: Optional[int]) -> list[dict]:
    from datasets import load_dataset

    from core.flows.nodes.data import local_seed_path

    if dataset.get("source") == "hf":
        loaded = _hub_rows(dataset, token, end)
    else:
        path = local_seed_path(dataset.get("ref"))
        builder = next((b for ext, b in LOADERS.items() if path.endswith(ext)), None)
        if builder is None:
            raise FlowNodeError(
                "Hold-out works on Parquet, CSV or JSON datasets; set Hold out % to 0 for this one"
            )
        loaded = load_dataset(builder, data_files = path, split = "train")
    if end is not None:
        loaded = loaded.select(range(min(len(loaded), end + 1)))
    if len(loaded) > HOLDOUT_MAX_ROWS:
        raise _too_many(len(loaded))
    # Timestamps and decimals in unused columns would stop the JSONL upload; NaN and inf become null.
    return [
        json.loads(json.dumps(row, default = str), parse_constant = lambda _: None)
        for row in loaded.to_list()
    ]


async def holdout_split(
    ctx: NodeContext, dataset: dict, percent: int, seed: int
) -> tuple[dict, dict]:
    """Train on the rest of the rows and hand the held-out ones on as their own dataset."""
    from core.flows.nodes.data import recipe_artifact, recipe_rows, upload_rows
    from core.training.account_jobs import account_hf_token

    if str(ctx.params.get("task") or "text") != "text" or ctx.params.get("dataset_streaming"):
        raise FlowNodeError(
            "Hold-out works on text datasets that are not streamed; set Hold out % to 0"
        )
    ctx.progress(phase = "Holding out rows")
    # Same "use rows up to" slice the trainer would apply, so the split is of the configured range.
    end = ctx.params.get("dataset_slice_end")
    end = None if end in (None, "") else int(end)
    if dataset.get("source") == "recipe" and not recipe_artifact(dataset.get("ref")):
        rows = await recipe_rows(ctx, str(dataset["ref"]))
        if end is not None:
            rows = rows[: end + 1]
    else:
        # The request's own Hub token, as /api/train/start gets it; managed accounts never borrow the server's.
        token = account_hf_token(getattr(ctx.caller, "hf_token", None))
        try:
            rows = await asyncio.to_thread(_load_rows, dataset, token, end)
        except FlowNodeError:
            raise
        except Exception as exc:
            raise FlowNodeError(f"Could not read the dataset to hold rows out: {exc}") from None
    held_ids = holdout_row_ids(len(rows), percent, seed)
    if not held_ids:
        raise FlowNodeError(f"{len(rows)} row(s) is too few to hold any out; set Hold out % to 0")
    held = set(held_ids)
    name = str(dataset.get("ref") or "dataset").replace("/", "-")
    tag = f"{name}-holdout{percent}-seed{seed}"
    train = await upload_rows(ctx, [r for i, r in enumerate(rows) if i not in held], f"{tag}-train")
    holdout = await upload_rows(ctx, [rows[i] for i in held_ids], f"{tag}-holdout")
    holdout.update({"percent": percent, "seed": seed, "row_ids": held_ids, "of_rows": len(rows)})
    return train, holdout


@handler("train.export")
async def export(ctx: NodeContext) -> dict:
    from studio_mcp import checkpoints

    checkpoint = ctx.inputs["checkpoint"]
    name = checkpoint.get("name") if isinstance(checkpoint, dict) else None
    if not name:
        raise FlowNodeError("Export needs a checkpoint")
    status = await ctx.call("GET", "/api/export/status")
    if isinstance(status, dict) and status.get("is_export_active"):
        raise FlowNodeError("Another export is running; try again when it ends", status = 409)
    try:
        found = await checkpoints.resolve(ctx.caller, name)
    except Exception as exc:
        raise FlowNodeError(str(exc)) from None
    fmt = str(ctx.params.get("format") or "gguf")
    save_directory = f"flows-{name.replace('/', '-')}-{fmt}-{int(time.time())}"
    op = f"flow-{secrets.token_hex(6)}"

    async def cancel() -> None:
        for step in ("load", "export"):
            try:
                await ctx.call(
                    "POST", "/api/export/cancel", json = {"expected_op_id": f"{op}-{step}"}
                )
            except FlowNodeError:
                pass

    ctx.on_cancel(cancel)
    loaded = await ctx.call(
        "POST",
        "/api/export/load-checkpoint",
        json = {"checkpoint_path": found.path, "max_seq_length": 2048, "op_id": f"{op}-load"},
    )
    if isinstance(loaded, dict) and loaded.get("success") is False:
        raise FlowNodeError(loaded.get("message") or "Could not load the checkpoint for export")
    body: dict[str, Any] = {
        "save_directory": save_directory,
        "push_to_hub": False,
        "expected_checkpoint": found.path,
        "op_id": f"{op}-export",
    }
    if fmt == "gguf":
        body["quantization_method"] = EXPORT_QUANTS.get(str(ctx.params.get("quant")), "Q4_K_M")
    result = await ctx.call("POST", f"/api/export/export/{fmt}", json = body)
    if isinstance(result, dict) and result.get("success") is False:
        raise FlowNodeError(result.get("message") or "The export failed")
    return {
        "model": {
            "kind": "model",
            "model_kind": "chat",
            "repo": save_directory,
            "variant": body.get("quantization_method"),
        }
    }


@handler("train.chat_checkpoint")
async def chat_checkpoint(ctx: NodeContext) -> dict:
    from studio_mcp import loading, trained_models

    checkpoint = ctx.inputs["checkpoint"]
    name = checkpoint.get("name") if isinstance(checkpoint, dict) else None
    if not name:
        raise FlowNodeError("Chat With Checkpoint needs a checkpoint")
    deadline = time.monotonic() + 3600
    previous = await active_chat_model(ctx)
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
    try:
        prompt = as_text(ctx.inputs.get("prompt")) or str(ctx.params.get("prompt") or "")
        if not prompt.strip():
            raise FlowNodeError("The prompt is empty")
        text = await ctx.llm_text(
            model = None,
            system = "",
            prompt = prompt,
            max_tokens = int(ctx.params.get("max_tokens") or 512),
            seed = ctx.seed,
        )
    finally:
        # Put back the chat model the adapter replaced: a Data Recipe step on the next run needs it loaded.
        await restore_chat_model(ctx, previous, trained.ref)
    return {"text": strip_think(text)}


async def active_chat_model(ctx: NodeContext) -> Optional[dict]:
    try:
        status = await ctx.call("GET", "/api/inference/status")
    except FlowNodeError:
        return None
    model = status.get("model_identifier") if isinstance(status, dict) else None
    if not isinstance(model, str) or not model.strip():
        return None
    return {"repo": model, "variant": status.get("gguf_variant") or None}


async def restore_chat_model(ctx: NodeContext, previous: Optional[dict], replaced_by: str) -> None:
    from studio_mcp import loading

    if not previous or loading._same_id(previous["repo"], replaced_by):
        return
    label = previous["repo"] + (f" ({previous['variant']})" if previous["variant"] else "")
    ctx.set_status("loading_model", f"Loading {label} again")
    try:
        await loading.load_llm(
            ctx.caller,
            None,
            model = previous["repo"],
            variant = previous["variant"],
            max_seq_length = FLOW_CHAT_CONTEXT,
            load_in_4bit = True,
            hf_token = None,
            kind = "llm",
            deadline = time.monotonic() + 1800,
        )
        note_loaded(ctx, "llm", previous["repo"])
        ctx.resident.clear()
    except Exception as exc:
        ctx.log(f"Could not load {label} again; load it before the next run ({exc})", "warning")
