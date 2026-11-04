# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training nodes: LoRA fine-tuning, export and chatting with a checkpoint, through the routes the MCP tools use."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import secrets
import time
from typing import Any, Optional

from core.flows.nodes._common import FLOW_CHAT_CONTEXT, as_text, note_loaded, poll_job, strip_think
from core.flows.nodes.data import recipe_artifact
from core.flows.registry import FlowCancelled, FlowNodeError, NodeContext, cache_hook, handler

POLL_S = 5.0
BUSY_WAIT_S = 180.0
TRAINING_TYPES = {"lora": "LoRA/QLoRA", "full": "Full Finetuning", "cpt": "Continued Pretraining"}
# What the Train page sends when the node leaves batch size and accumulation blank (GRPO sets its own).
TRAIN_PAGE_BATCH = {"batch_size": 4, "gradient_accumulation_steps": 8}
CPT_EMBEDDING_MODULES = ("embed_tokens", "lm_head")
LORA_VARIANTS = ("rslora", "loftq", "dora")
VISION_LORA = (
    "finetune_vision_layers",
    "finetune_language_layers",
    "finetune_attention_modules",
    "finetune_mlp_modules",
)
TASK_FLAGS = {
    "vision": {"is_dataset_image": True},
    "audio": {"is_dataset_audio": True},
    "embedding": {"is_embedding": True},
}
OPTIONAL_INTS = (
    "batch_size",
    "gradient_accumulation_steps",
    "vision_image_size",
    "dataset_slice_start",
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
    "epsilon_high": ("grpo_epsilon_high", float),
    "enable_thinking": ("grpo_enable_thinking", bool),
    "mask_truncated_completions": ("grpo_mask_truncated_completions", bool),
}
EXPORT_QUANTS = {"q4_k_m": "Q4_K_M", "q5_k_m": "Q5_K_M", "q8_0": "Q8_0", "f16": "F16"}


def _picked_cache(cache: Any) -> dict:
    """The On device copy the base model was picked from: train on those weights, as the Train page
    does. A path missing here (a flow from another machine) falls back to the plain repo."""
    if not isinstance(cache, dict) or cache.get("known_cached") is not True:
        return {}
    path = cache.get("local_path")
    if isinstance(path, str) and path:
        return (
            {"model_known_cached": True, "model_local_path": path} if os.path.exists(path) else {}
        )
    return {"model_known_cached": True}


@handler("train.finetune")
async def finetune(ctx: NodeContext) -> dict:
    from studio_mcp.tools.jobs import TRAINING_STATUS, _run_folder
    from studio_mcp.tools.status import _training

    dataset = ctx.inputs["dataset"]
    if not isinstance(dataset, dict) or dataset.get("kind") != "dataset":
        raise FlowNodeError("Fine-tune needs a dataset")
    params = ctx.params
    objective = str(params.get("objective") or "sft")
    if objective not in OBJECTIVES:
        raise FlowNodeError(f"Unknown objective {objective!r}")
    config: dict[str, Any] = {
        "model_name": str(params.get("base_model") or "").strip(),
        "format_type": str(params.get("format_type") or "auto"),
        "max_steps": int(params.get("max_steps") or 60),
        "start_request_id": f"flow-{secrets.token_hex(8)}",
    }
    if not config["model_name"]:
        raise FlowNodeError("Name the base model to fine-tune")
    config.update(_picked_cache(params.get("base_model_cache")))
    config.update(TASK_FLAGS.get(str(params.get("task") or "text"), {}))
    if params.get("num_epochs") not in (None, ""):
        config["num_epochs"] = int(params["num_epochs"])
        config["max_steps"] = None
    for key in OPTIONAL_INTS:
        if params.get(key) not in (None, ""):
            config[key] = int(params[key])
    if str(params.get("project_name") or "").strip():
        config["project_name"] = str(params["project_name"]).strip()
    method_fields(params, config)
    lora_fields(params, config)
    schedule_fields(params, config, ctx.seed)
    memory_fields(params, config, objective)
    if params.get("dataset_streaming") and objective == "sft":
        config["dataset_streaming"] = True
    if (
        params.get("drafter_enabled")
        and objective == "sft"
        and config["training_type"] != TRAINING_TYPES["cpt"]
    ):
        config["drafter"] = drafter_request(params)
    mapping = ctx.params.get("custom_format_mapping")
    prompt_col = str(ctx.params.get("prompt_column") or "").strip()
    response_col = str(ctx.params.get("response_column") or "").strip()
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
    for key, value in TRAIN_PAGE_BATCH.items():
        config.setdefault(key, value)
    holdout = None
    source = dataset
    percent = int(ctx.params.get("holdout_percent") or 0)
    if percent > 0:
        dataset, holdout = await holdout_split(
            ctx, dataset, percent, int(ctx.params.get("holdout_seed") or 0)
        )
        # The split already covers only the sliced rows; slicing the train half again would shrink it.
        config.pop("dataset_slice_start", None)
        config.pop("dataset_slice_end", None)
        eval_split = str(ctx.params.get("eval_split") or "").strip()
        train_split = source.get("split") or "train"
        if source.get("source") == "hf" and not eval_split and config["eval_steps"] > 0:
            # Blank means the trainer picks the Hub's validation/test split, which it can no longer see.
            from core.training.account_jobs import account_hf_token
            token = account_hf_token(getattr(ctx.caller, "hf_token", None))
            eval_split = await _off_thread(ctx, _auto_eval_split, source, train_split, token) or ""
        if (
            source.get("source") == "hf"
            and eval_split not in ("", train_split)
            and config["eval_steps"] > 0
        ):
            # The train half is now a local upload, so the Hub eval split has to travel the same way.
            # The training split itself means "carve eval from the training rows", which the trainer still does.
            config["local_eval_datasets"] = [await hub_eval_upload(ctx, source, eval_split)]
    if dataset.get("source") == "hf":
        config["hf_dataset"] = dataset["ref"]
        if dataset.get("subset"):
            config["subset"] = dataset["subset"]
        if dataset.get("split"):
            config["train_split"] = dataset["split"]
        if str(ctx.params.get("eval_split") or "").strip():
            config["eval_split"] = str(ctx.params["eval_split"]).strip()
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
    checkpoint = {"kind": "checkpoint", "name": run, "base_model": config["model_name"]}
    row = None
    if objective == "grpo":
        from core.training.account_jobs import account_hf_token

        # Whether a prompt carried its own system message is in the rows: read the first, as the
        # formatter saw it (an Arrow column holds one type, so one row tells text from conversation).
        token = account_hf_token(getattr(ctx.caller, "hf_token", None))
        row = await asyncio.to_thread(_first_row, dataset, token)
    system = trained_system_prompt(config, row)
    if system:
        checkpoint["system_prompt"] = system
    return {
        "checkpoint": checkpoint,
        "metrics": metrics,
        "holdout": holdout,
    }


def method_fields(params: dict, config: dict) -> None:
    """Method, precision, sequence length and learning rates, as the Train page sends them."""
    method = str(params.get("training_type") or "lora")
    config["training_type"] = TRAINING_TYPES.get(method, TRAINING_TYPES["lora"])
    if method == "lora":
        config["load_in_4bit"] = params.get("load_in_4bit") is not False
    else:
        # Full trains in 16-bit; CPT loads 4-bit only for a pre-quantized checkpoint, like the Train page.
        config["load_in_4bit"] = method == "cpt" and "4bit" in config["model_name"].lower()
    config["max_seq_length"] = int(params.get("max_seq_length") or 2048)
    lr = params.get("learning_rate")
    config["learning_rate"] = "2e-4" if lr in (None, "") else str(lr)
    if method == "cpt" and params.get("embedding_learning_rate") not in (None, ""):
        config["embedding_learning_rate"] = float(params["embedding_learning_rate"])


def lora_fields(params: dict, config: dict) -> None:
    adapter = config["training_type"] != TRAINING_TYPES["full"]
    config["use_lora"] = adapter
    config["lora_r"] = int(params.get("lora_r") or 16)
    config["lora_alpha"] = int(params.get("lora_alpha") or 32)
    if params.get("lora_dropout") not in (None, ""):
        config["lora_dropout"] = float(params["lora_dropout"])
    variant = str(params.get("lora_variant") or "lora")
    for name in LORA_VARIANTS:
        config[f"use_{name}"] = adapter and variant == name
    modules = [str(m).strip() for m in params.get("target_modules") or [] if str(m).strip()]
    if config["training_type"] == TRAINING_TYPES["cpt"] and not set(CPT_EMBEDDING_MODULES) & set(
        modules
    ):
        modules += CPT_EMBEDDING_MODULES
    config["target_modules"] = modules if adapter else []
    for key in VISION_LORA:
        config[key] = params.get(key) is not False


def schedule_fields(params: dict, config: dict, flow_seed: int) -> None:
    config["optim"] = str(params.get("optim") or "adamw_8bit")
    config["lr_scheduler_type"] = str(params.get("lr_scheduler_type") or "linear")
    if params.get("weight_decay") not in (None, ""):
        config["weight_decay"] = float(params["weight_decay"])
    if params.get("warmup_steps") not in (None, ""):
        config["warmup_steps"] = int(params["warmup_steps"])
    config["save_steps"] = int(params.get("save_steps") or 0)
    eval_steps = float(params.get("eval_steps") or 0)
    config["eval_steps"] = eval_steps
    # Patience needs evaluation, and the Train page sends 0 without it.
    config["early_stopping_patience"] = (
        int(params.get("early_stopping_patience") or 0) if eval_steps > 0 else 0
    )
    seed = params.get("random_seed")
    config["random_seed"] = int(seed) if seed not in (None, "") else flow_seed % (2**31)


def memory_fields(params: dict, config: dict, objective: str) -> None:
    sft = objective == "sft"
    task = str(params.get("task") or "text")
    cpt = config["training_type"] == TRAINING_TYPES["cpt"]
    config["packing"] = bool(params.get("packing")) and sft and task != "embedding"
    config["train_on_completions"] = (
        bool(params.get("train_on_completions"))
        and sft
        and not cpt
        and task != "embedding"
        and config["format_type"] != "raw"
    )
    checkpointing = str(params.get("gradient_checkpointing") or "unsloth")
    config["gradient_checkpointing"] = checkpointing
    mode = str(params.get("offload_layers") or "off")
    offloads = (
        config["training_type"] != TRAINING_TYPES["full"]
        and checkpointing != "none"
        and task in ("text", "vision")
    )
    if not offloads or mode == "off":
        config["offload_layers"] = 0
    elif mode == "auto":
        config["offload_layers"] = "auto"
        if params.get("offload_vram_gb") not in (None, ""):
            config["offload_vram_gb"] = float(params["offload_vram_gb"])
    else:
        config["offload_layers"] = min(1024, max(1, int(params.get("offload_count") or 8)))
    depth = str(params.get("prefetch_depth") or "2")
    config["prefetch_depth"] = "auto" if depth == "auto" else min(8, max(1, int(depth)))


def drafter_request(params: dict) -> dict:
    lr = params.get("drafter_learning_rate")
    return {
        "enabled": True,
        "data_source": str(params.get("drafter_data_source") or "dataset"),
        "max_examples": int(params.get("drafter_max_examples") or 2000),
        "epochs": float(params.get("drafter_epochs") or 1.0),
        "learning_rate": float(lr) if lr not in (None, "") and float(lr) > 0 else None,
        "draft_steps": int(params.get("drafter_draft_steps") or 3),
    }


def trained_system_prompt(config: dict, row: Optional[dict] = None) -> str:
    """The system message a new prompt would have got in training: the SFT mapping's system prompt,
    or GRPO's, which fills in for every prompt with no system message of its own. GRPO claims it
    only when ``row`` (the dataset's first) shows every prompt got it: plain-text prompts and no
    system column. A conversation can carry its own system message, and rows it cannot read
    claim nothing."""
    mapping = config.get("custom_format_mapping")
    if config.get("objective") == "grpo":
        from core.training.rl import resolve_role_columns

        if not isinstance(row, dict):
            return ""
        try:
            roles = resolve_role_columns(
                list(row), "grpo", mapping if isinstance(mapping, dict) else None
            )
        except ValueError:
            return ""
        if "system" in roles or not isinstance(row.get(roles["prompt"]), str):
            return ""
        return str(config.get("rl_system_prompt") or "").strip()
    if isinstance(mapping, dict) and isinstance(mapping.get("__system_prompt"), str):
        return mapping["__system_prompt"].strip()
    return ""


def _hub_revision(dataset: dict, token: Any) -> Optional[str]:
    from huggingface_hub import HfApi
    return HfApi().dataset_info(str(dataset["ref"]), token = token).sha


def _hub_model_revision(repo: str, token: Any) -> Optional[str]:
    from huggingface_hub import HfApi
    return HfApi().model_info(repo, token = token).sha


# A cell naming an image, clip or sound file: training reads that file, which the rows' hash never sees.
_MEDIA_PATH = re.compile(
    rb"\.(?:png|jpe?g|webp|gif|bmp|tiff?|wav|mp3|flac|ogg|m4a|mp4|webm|mov)(?=[\"',\s\]})]|$)",
    re.IGNORECASE,
)


def _names_media_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_MEDIA_PATH.search(value.encode()))
    if isinstance(value, dict):
        return any(_names_media_value(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_names_media_value(v) for v in value)
    return False


def _names_media(file: str) -> bool:
    if file.lower().endswith(".parquet"):
        # Compressed pages hide the text: read the string columns.
        import pyarrow as pa
        import pyarrow.parquet as pq

        try:
            table = pq.read_table(file)
        except Exception:
            table = None
        if table is not None:
            # An Image or Audio column keeps its file under a struct ({"bytes", "path"}), or a list of them.
            return any(
                _names_media_value(v)
                for column in table.columns
                if not pa.types.is_binary(column.type)
                for v in column.to_pylist()
            )
    with open(file, "rb") as fh:
        # The overlap keeps a name split across two reads.
        tail = b""
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            if _MEDIA_PATH.search(tail + chunk):
                return True
            tail = chunk[-16:]
    return False


def _content_hash(path: str) -> Optional[str]:
    """The dataset files' digest; None when the rows name media files, whose content it cannot see."""
    import glob
    import hashlib
    import os

    digest = hashlib.sha256()
    files = sorted(glob.glob(path)) if "*" in path else [path]
    if not files:
        # A folder of Arrow, CSV or JSON files (a saved dataset) matches no `*.parquet`: the digest
        # would be of nothing and never change, so train again rather than reuse.
        return None
    for file in files:
        if _names_media(file):
            return None
        digest.update(os.path.basename(file).encode() + b"\0")
        with open(file, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _adapter_base(path: str) -> Optional[str]:
    """The base a saved LoRA folder trains on top of (its adapter_config.json), else None."""
    import json

    try:
        with open(os.path.join(path, "adapter_config.json"), encoding = "utf-8") as fh:
            named = json.load(fh).get("base_model_name_or_path")
    except (OSError, ValueError, AttributeError):
        return None
    return named.strip() if isinstance(named, str) and named.strip() else None


async def _model_material(ctx: NodeContext, base: str) -> Optional[dict]:
    """What identifies the weights a run starts from: a local folder's files, a Hub repo's revision,
    and for a saved LoRA folder the same for the base it loads under it. None = cannot tell."""
    found: dict = {}
    if base.startswith("ref:"):
        # A model picked from Studio's inventory names its folder by a handle, as training resolves it.
        from hub.utils.host_paths import resolve_host_path_reference
        base = resolve_host_path_reference(base) or ""
        if not os.path.isabs(base):
            return None
    from core.flows.nodes.data import _REPO

    if not os.path.isabs(base) and (os.path.exists(base) or not _REPO.match(base)):
        # A relative local folder (./my-model, or owner/model on disk) loads from where Studio runs,
        # ahead of a Hub repo of that name, so it is fingerprinted there.
        if not os.path.exists(base):
            return None
        base = os.path.abspath(base)
    if os.path.isabs(base):
        from core.flows.engine import _local_fingerprint

        # Weights replaced at the same path must train again, as a local model's cache key does.
        found["base_model_files"] = await asyncio.to_thread(_local_fingerprint, base)
        if found["base_model_files"] is None:
            return None
        under = await asyncio.to_thread(_adapter_base, base)
        if under and under != base:
            # Continuing an adapter also loads its base: a changed base must train again too.
            inner = await _model_material(ctx, under)
            if inner is None:
                return None
            found["adapter_base"] = inner
        return found
    from core.training.account_jobs import account_hf_token

    if _REPO.match(base):
        # New weights pushed to the same Hub repo must train again too.
        token = account_hf_token(getattr(ctx.caller, "hf_token", None))
        try:
            found["base_model_revision"] = await asyncio.to_thread(_hub_model_revision, base, token)
            loaded = {}
            for repo in _loader_repos(base, _loads_4bit(ctx.params, base)):
                # The loader may fetch a 4-bit or 16-bit mirror instead: its changes count too.
                loaded[repo] = await asyncio.to_thread(_hub_model_revision, repo, token)
                if not loaded[repo]:
                    return None
            if loaded:
                found["loader_revisions"] = loaded
            if not found["base_model_revision"]:
                return None
            under = await asyncio.to_thread(
                _hub_adapter_base, base, found["base_model_revision"], token
            )
        except Exception:
            return None
        if under and under != base:
            # A Hub LoRA loads its base too, unpinned: a changed base must train again.
            inner = await _model_material(ctx, under)
            if inner is None:
                return None
            found["adapter_base"] = inner
    return found


def _hub_adapter_base(repo: str, revision: str, token: Any) -> Optional[str]:
    """The base a Hub LoRA repo names in its adapter_config.json at ``revision``; None when the repo is
    not an adapter. Any other failure raises, so the caller trains again rather than guess."""
    import json

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        path = hf_hub_download(repo, "adapter_config.json", revision = revision, token = token)
    except LocalEntryNotFoundError:
        raise  # offline and not cached: cannot tell
    except EntryNotFoundError:
        return None
    with open(path, encoding = "utf-8") as fh:
        named = json.load(fh).get("base_model_name_or_path")
    return named.strip() if isinstance(named, str) and named.strip() else None


def _loads_4bit(params: dict, base: str) -> bool:
    """Whether the run asks for a 4-bit load, as method_fields sets it."""
    method = str(params.get("training_type") or "lora")
    if method == "lora":
        return params.get("load_in_4bit") is not False
    return method == "cpt" and "4bit" in base.lower()


def _loader_repos(base: str, load_in_4bit: bool) -> list[str]:
    """The repos Unsloth's loader can fetch in place of ``base``: its mirror for each load mode the run
    can end in (a 4-bit request falls back to 16-bit, never the reverse) and that mirror without its
    4-bit suffix (what a GPU without pre-quantized support downloads). The worker scans the same set."""
    from utils.models.unsloth_mirror import unsloth_public_mirror

    repos: list[str] = []
    for mode in (True, False) if load_in_4bit else (False,):
        mirrored = unsloth_public_mirror(base, mode)
        if not mirrored or mirrored == base:
            continue
        repos.append(mirrored)
        suffix = next((x for x in ("-unsloth-bnb-4bit", "-bnb-4bit") if mirrored.endswith(x)), "")
        if suffix and mirrored[: -len(suffix)] != base:
            repos.append(mirrored[: -len(suffix)])
    return list(dict.fromkeys(repos))


async def finetune_cache_material(ctx: NodeContext) -> Optional[dict]:
    """What the params can't show: the dataset's content and, for GRPO, the reward rules.
    None (an unreachable Hub, a missing file) means train again rather than guess."""
    dataset = ctx.inputs.get("dataset")
    if str(ctx.params.get("task") or "text") in ("vision", "audio") and not (
        isinstance(dataset, dict) and dataset.get("source") == "hf"
    ):
        # A local vision or audio set names image and sound files training reads (scan.jfif, a path
        # with no extension): their content is not in the rows' hash, so train again.
        return None
    found = await dataset_material(ctx)
    if found is None:
        return None
    base = str(ctx.params.get("base_model") or "").strip()
    if base:
        model = await _model_material(ctx, base)
        if model is None:
            return None
        found.update(model)
        picked = _picked_cache(ctx.params.get("base_model_cache"))
        if picked:
            from core.training.training import resolve_training_model_load_target

            # Trained from an On device copy: the snapshot it resolves to (its commit) is the weights, and
            # a copy gone from disk falls back to the Hub, which must train again too.
            target = await asyncio.to_thread(
                resolve_training_model_load_target, {"model_name": base, **picked}
            )
            found["base_model_snapshot"] = os.path.basename(target) if target != base else None
    if str(ctx.params.get("objective") or "sft") == "grpo":
        # A reward can be saved again under the same name, so the key holds what each rule checks.
        found["rewards"] = reward_rules(ctx.params.get("rewards"))
    return found


def reward_rules(rewards: Any) -> list:
    from core.training.rewards import get_reward

    rules = []
    for r in rewards or []:
        name = str((r or {}).get("name") or "").strip() if isinstance(r, dict) else ""
        try:
            record = get_reward(name)
            rules.append([name, record.get("kind"), record.get("rule")])
        except Exception:
            rules.append([name, None])
    return rules


async def dataset_material(ctx: NodeContext) -> Optional[dict]:
    from core.flows.nodes.data import local_seed_path, recipe_artifact
    from core.training.account_jobs import account_hf_token

    dataset = ctx.inputs.get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("ref"):
        return None
    if dataset.get("source") == "hf":
        token = account_hf_token(getattr(ctx.caller, "hf_token", None))
        sha = await asyncio.to_thread(_hub_revision, dataset, token)
        return {"hub": sha} if sha else None
    if dataset.get("source") == "recipe" and not recipe_artifact(dataset["ref"]):
        # A recipe job id names one run's rows and is never reused.
        return {"job": str(dataset["ref"])}
    found = await asyncio.to_thread(_content_hash, local_seed_path(dataset["ref"]))
    return {"sha256": found} if found else None


async def finetune_cache_valid(ctx: NodeContext, outputs: dict) -> bool:
    """A cached run is only reusable while its adapter folder (and held-out file) still exist."""
    from studio_mcp import checkpoints

    from core.flows.nodes.data import local_seed_path

    name = (outputs.get("checkpoint") or {}).get("name") if isinstance(outputs, dict) else None
    if not name:
        return False
    try:
        await checkpoints.resolve(ctx.caller, name)
        holdout = outputs.get("holdout")
        if isinstance(holdout, dict) and holdout.get("ref"):
            local_seed_path(holdout["ref"])
    except Exception:
        return False
    return True


# Fine-tune never reads the flow's parameters (a question asked downstream must not retrain it).
cache_hook(
    "train.finetune",
    material = finetune_cache_material,
    valid = finetune_cache_valid,
    flow_params = False,
)


def rl_roles(params: dict, objective: str) -> dict:
    """Column -> RL role for the columns the node names; Studio detects the rest by name."""
    named = {"prompt": params.get("prompt_column")}
    if objective == "grpo":
        named["answer"] = params.get("answer_column")
    else:
        named["chosen"] = params.get("chosen_column")
        named["rejected"] = params.get("rejected_column")
    roles = {role: str(col).strip() for role, col in named.items() if str(col or "").strip()}
    if len(set(roles.values())) < len(roles):
        raise FlowNodeError("Each role needs its own column")
    return {col: role for role, col in roles.items()}


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
    if params.get("max_prompt_length") not in (None, ""):
        config["rl_max_prompt_length"] = int(params["max_prompt_length"])
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
    if params.get("batch_size") in (None, ""):
        # One prompt and all its completions per step, as Unsloth's GRPO notebooks run. The start
        # route's batch of 1 has TRL spread each prompt over num_generations steps instead.
        config["batch_size"] = int(params.get("num_generations") or 4)
        if params.get("gradient_accumulation_steps") in (None, ""):
            config["gradient_accumulation_steps"] = 1
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
    rl = (
        ((payload or {}).get("metric_history") or {}).get("rl")
        if isinstance(payload, dict)
        else None
    )
    if not isinstance(rl, list):
        return []
    raw = [
        float(e["reward"])
        for e in rl
        if isinstance(e, dict) and isinstance(e.get("reward"), (int, float))
    ]
    if not raw:
        return []
    # One step is one prompt's group, so the per-step reward swings between 0 and the max; a trailing
    # mean over about a twentieth of the run is what shows whether it is learning.
    window = max(1, min(25, len(raw) // 20))
    head = sum(raw[:window]) / window
    smooth = [
        head if i < window else sum(raw[i - window + 1 : i + 1]) / window for i in range(len(raw))
    ]
    return downsample([round(v, 4) for v in smooth])


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


def _too_many(rows: Any) -> FlowNodeError:
    return FlowNodeError(
        f"{rows} rows is too many to hold out from here; set Hold out % to 0 or use a smaller slice"
    )


def _stream_rows(streamed: Any, start: int, end: Optional[int]) -> Any:
    """Rows start..end of a streamed dataset, reading no further than the slice end or the cap plus one."""
    from datasets import Dataset

    # The end may be past the last row (the trainer clamps it): only rows actually read count.
    count = (
        HOLDOUT_MAX_ROWS + 1 if end is None else min(max(end + 1 - start, 0), HOLDOUT_MAX_ROWS + 1)
    )
    rows = list((streamed.skip(start) if start else streamed).take(count))
    if len(rows) > HOLDOUT_MAX_ROWS:
        raise _too_many(f"More than {HOLDOUT_MAX_ROWS}")
    return Dataset.from_list(rows)


def _compound_rows(
    ref: str, subset: Optional[str], split: str, token: Any, start: int, end: Optional[int]
) -> Any:
    """Percent slices and joined splits (`train[:50%]`, `train+validation`), resolved against the
    Hub's split sizes and streamed one bounded range at a time, so nothing past the hold-out's rows
    or the cap is downloaded. Percents need those sizes; without them the split is refused."""
    from datasets import Dataset, ReadInstruction, load_dataset, load_dataset_builder

    try:
        info = load_dataset_builder(ref, subset, token = token).info.splits or {}
        sizes = {name: s.num_examples for name, s in info.items() if s.num_examples}
    except Exception:
        sizes = {}
    try:
        parts = [
            (p.splitname, p.from_, p.to)
            for p in ReadInstruction.from_spec(split).to_absolute(sizes)
        ]
    except Exception:
        if "[" in split:
            raise FlowNodeError(
                f"The Hub does not list the split sizes {split!r} needs; use a row slice such as "
                "train[:1000], or set Hold out % to 0"
            )
        parts = [(name, 0, None) for name in split.split("+")]
    if all(hi is not None for _, _, hi in parts):
        total = sum(hi - lo for _, lo, hi in parts)
        stop = total if end is None else min(total, end + 1)
        if stop - start > HOLDOUT_MAX_ROWS:
            raise _too_many(stop - start)
    rows: list = []
    seen = 0  # rows of the joined split before this part
    for name, lo, hi in parts:
        if end is not None and seen > end:
            break
        streamed = load_dataset(ref, subset, split = name, token = token, streaming = True)
        if hi is None:
            # A part of unknown length: walk it, counting the rows it really has.
            for row in streamed:
                if end is not None and seen > end:
                    break
                if seen >= start:
                    rows.append(row)
                    if len(rows) > HOLDOUT_MAX_ROWS:
                        raise _too_many(f"More than {HOLDOUT_MAX_ROWS}")
                seen += 1
            continue
        size = hi - lo
        skip = max(start - seen, 0)
        want = size if end is None else min(size, end + 1 - seen)
        if want > skip:
            rows += list(streamed.skip(lo + skip).take(want - skip))
            if len(rows) > HOLDOUT_MAX_ROWS:
                raise _too_many(f"More than {HOLDOUT_MAX_ROWS}")
        seen += size
    return Dataset.from_list(rows)


def _first_row(dataset: dict, token: Any) -> Optional[dict]:
    """The dataset's first row as training reads it, or None when it cannot be read here."""
    from datasets import load_dataset

    from core.flows.nodes.data import local_seed_path

    try:
        if dataset.get("source") == "hf":
            split = re.split(r"[\[+]", str(dataset.get("split") or "train").replace(" ", ""))[0]
            streamed = load_dataset(
                str(dataset["ref"]),
                dataset.get("subset") or None,
                split = split,
                token = token,
                streaming = True,
            )
        else:
            path = local_seed_path(dataset.get("ref"))
            builder = next((b for ext, b in LOADERS.items() if path.endswith(ext)), None)
            if builder is None:
                return None
            from utils.datasets.cells import csv_as_text_kwargs

            # CSV cells stay text as training reads them: "00123" is a string prompt, not 123.
            streamed = load_dataset(
                builder,
                data_files = path,
                split = "train",
                streaming = True,
                **csv_as_text_kwargs([path]),
            )
        return next(iter(streamed), None)
    except Exception:
        return None


def _hub_rows(dataset: dict, token: Any, start: int, end: Optional[int]) -> Any:
    """A Hub split, without downloading more than the hold-out can use: a slice streams just its
    rows, a split the Hub says is over the cap is refused before anything is fetched, and one of
    unknown size streams at most the cap plus one row."""
    from datasets import load_dataset, load_dataset_builder

    ref, subset = str(dataset["ref"]), dataset.get("subset") or None
    split = str(dataset.get("split") or "train").replace(" ", "")
    # Streaming takes only plain split names: `train[10:100]` becomes a bounded read of train.
    sliced = re.fullmatch(r"([\w.-]+)\[(\d*):(\d*)\]", split)
    if sliced:
        lo = int(sliced.group(2) or 0)
        split, start = sliced.group(1), lo + start
        end = None if end is None else lo + end
        if sliced.group(3):
            end = int(sliced.group(3)) - 1 if end is None else min(end, int(sliced.group(3)) - 1)
    elif "[" in split or "+" in split:
        return _compound_rows(ref, subset, split, token, start, end)

    def stream() -> Any:
        streamed = load_dataset(ref, subset, split = split, token = token, streaming = True)
        return _stream_rows(streamed, start, end)

    if end is not None:
        return stream()
    try:
        known = load_dataset_builder(ref, subset, token = token).info.splits[split].num_examples
    except Exception:
        known = None
    if not known or start:
        if known and known - start > HOLDOUT_MAX_ROWS:
            raise _too_many(known - start)
        return stream()
    if known > HOLDOUT_MAX_ROWS:
        raise _too_many(known)
    # Streamed even when small: a whole-split load prepares every split of the dataset first.
    return stream()


def _load_rows(
    dataset: dict,
    token: Any,
    end: Optional[int],
    start: int = 0,
) -> list[dict]:
    """Rows start..end (inclusive) of the dataset, already sliced."""
    return _jsonable(_load_table(dataset, token, end, start).to_list())


def _load_table(
    dataset: dict,
    token: Any,
    end: Optional[int],
    start: int = 0,
):
    """The same rows as a ``datasets.Dataset``, keeping the source's column types."""
    from datasets import load_dataset

    from core.flows.nodes.data import local_seed_path

    if dataset.get("source") == "hf":
        loaded = _hub_rows(dataset, token, start, end)
    else:
        path = local_seed_path(dataset.get("ref"))
        builder = next((b for ext, b in LOADERS.items() if path.endswith(ext)), None)
        if builder is None:
            raise FlowNodeError(
                "Hold-out works on Parquet, CSV or JSON datasets; set Hold out % to 0 for this one"
            )
        from utils.datasets.cells import csv_as_text_kwargs

        # Streamed, so a big local file is never read past the slice end or the cap. CSV cells stay
        # text as training reads them ("00501" keeps its zeros, "NA" stays a value).
        streamed = load_dataset(
            builder,
            data_files = path,
            split = "train",
            streaming = True,
            **csv_as_text_kwargs([path]),
        )
        loaded = _stream_rows(streamed, start, end)
    if len(loaded) > HOLDOUT_MAX_ROWS:
        raise _too_many(len(loaded))
    return loaded


def _jsonable(rows: list[dict]) -> list[dict]:
    # Timestamps and decimals in unused columns would stop the JSONL upload; NaN and inf become null.
    return [json.loads(json.dumps(row, default = str), parse_constant = lambda _: None) for row in rows]


async def _off_thread(ctx: NodeContext, fn: Any, *args: Any) -> Any:
    """``fn`` on a worker thread, given up on Stop: a running read cannot be interrupted, and waiting it
    out would keep the run (and the Flows GPU lane) busy until it ends."""
    task = asyncio.ensure_future(asyncio.to_thread(fn, *args))
    while True:
        done, _ = await asyncio.wait({task}, timeout = 0.25)
        if done:
            return task.result()
        if ctx.cancelled:
            task.add_done_callback(lambda t: t.cancelled() or t.exception())
            ctx.check_cancelled()


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
    # Same train split start/end slice the trainer would apply, so the split is of the configured range.
    end = ctx.params.get("dataset_slice_end")
    end = None if end in (None, "") else int(end)
    start = ctx.params.get("dataset_slice_start")
    start = 0 if start in (None, "") else int(start)
    table = None
    if dataset.get("source") == "recipe" and not recipe_artifact(dataset.get("ref")):
        rows = await recipe_rows(ctx, str(dataset["ref"]))
        rows = rows[start : None if end is None else end + 1]
    else:
        # The request's own Hub token, as /api/train/start gets it; managed accounts never borrow the server's.
        token = account_hf_token(getattr(ctx.caller, "hf_token", None))
        try:
            table = await _off_thread(ctx, _load_table, dataset, token, end, start)
        except (FlowNodeError, FlowCancelled):
            raise
        except Exception as exc:
            raise FlowNodeError(f"Could not read the dataset to hold rows out: {exc}") from None
    count = len(rows) if table is None else len(table)
    held_ids = holdout_row_ids(count, percent, seed)
    if not held_ids:
        raise FlowNodeError(f"{count} row(s) is too few to hold any out; set Hold out % to 0")
    held = set(held_ids)
    kept = [i for i in range(count) if i not in held]
    name = str(dataset.get("ref") or "dataset").replace("/", "-")
    tag = f"{name}-holdout{percent}-seed{seed}"
    if table is None:
        train = await upload_rows(ctx, [rows[i] for i in kept], f"{tag}-train")
        holdout = await upload_rows(ctx, [rows[i] for i in held_ids], f"{tag}-holdout")
    elif dataset.get("source") != "hf" and _is_csv(dataset):
        # Saved as CSV again, so training reads it as it reads the source: a numeric id column stays
        # out of a raw-text run's text, which the JSONL's all-string rows would lose.
        rows = _jsonable(table.to_list())
        train = await _upload_csv(ctx, [rows[i] for i in kept], f"{tag}-train")
        holdout = await _upload_csv(ctx, [rows[i] for i in held_ids], f"{tag}-holdout")
    else:
        # Saved as Parquet with the source's column types: a decimal or date column stringified into
        # JSONL would become the text column a raw-text run picks.
        train = await _upload_parquet(ctx, table.select(kept), f"{tag}-train")
        holdout = await _upload_parquet(ctx, table.select(held_ids), f"{tag}-holdout")
    holdout.update({"percent": percent, "seed": seed, "row_ids": held_ids, "of_rows": count})
    return train, holdout


def _is_csv(dataset: dict) -> bool:
    from core.flows.nodes.data import local_seed_path
    try:
        return local_seed_path(dataset.get("ref")).lower().endswith(".csv")
    except FlowNodeError:
        return False


async def _upload_parquet(ctx: NodeContext, table: Any, name: str) -> dict:
    import io

    from core.flows.nodes.data import upload_ref

    out = io.BytesIO()
    table.to_parquet(out)
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "flow-dataset"
    saved = await ctx.call(
        "POST",
        "/api/hub/datasets/upload",
        files = {"file": (f"{name}.parquet", out.getvalue(), "application/octet-stream")},
    )
    stored = saved.get("stored_path") if isinstance(saved, dict) else None
    if not isinstance(stored, str) or not stored:
        raise FlowNodeError("Unsloth Studio did not say where it saved the dataset")
    return {
        "kind": "dataset",
        "source": "local",
        "ref": upload_ref(stored),
        "rows": len(table),
        "columns": sorted(table.column_names),
    }


async def _upload_csv(ctx: NodeContext, rows: list[dict], name: str) -> dict:
    import csv
    import io

    from core.flows.nodes.data import upload_ref

    columns = list(dict.fromkeys(k for r in rows for k in r))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames = columns)
    writer.writeheader()
    writer.writerows({k: "" if v is None else v for k, v in r.items()} for r in rows)
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "flow-dataset"
    saved = await ctx.call(
        "POST",
        "/api/hub/datasets/upload",
        files = {"file": (f"{name}.csv", out.getvalue().encode(), "application/octet-stream")},
    )
    stored = saved.get("stored_path") if isinstance(saved, dict) else None
    if not isinstance(stored, str) or not stored:
        raise FlowNodeError("Unsloth Studio did not say where it saved the dataset")
    return {
        "kind": "dataset",
        "source": "local",
        "ref": upload_ref(stored),
        "rows": len(rows),
        "columns": sorted(columns),
    }


def _eval_rows(dataset: dict, token: Any, split: str) -> list[dict]:
    """The whole evaluation split: those rows are scored, not held out, so the hold-out cap doesn't apply."""
    from datasets import load_dataset

    loaded = load_dataset(
        str(dataset["ref"]), dataset.get("subset") or None, split = split, token = token
    )
    return _jsonable(loaded.to_list())


def _auto_eval_split(dataset: dict, train_split: str, token: Any) -> Optional[str]:
    """The split the trainer would pick for a blank eval split: the first candidate the Hub has with
    enough rows."""
    from datasets import get_dataset_split_names, load_dataset, load_dataset_builder

    from core.training.eval_dataset import EVAL_SPLIT_CANDIDATES, MIN_EVAL_ROWS
    from utils.hf_dataset_options import hf_dataset_split_instruction_names

    ref, subset = str(dataset["ref"]), dataset.get("subset") or None
    try:
        names = get_dataset_split_names(ref, subset, token = token)
    except Exception:
        return None
    try:
        sizes = load_dataset_builder(ref, subset, token = token).info.splits or {}
    except Exception:
        sizes = {}
    # Every split the training instruction reads is excluded, as the trainer excludes them.
    trained = set(hf_dataset_split_instruction_names(train_split)) or {
        m.group(0) for m in [re.match(r"[\w.-]+", train_split)] if m
    }
    for candidate in EVAL_SPLIT_CANDIDATES:
        if candidate not in names or candidate in trained:
            continue
        known = getattr(sizes.get(candidate), "num_examples", None)
        try:
            rows = known or sum(
                1
                for _ in load_dataset(
                    ref, subset, split = candidate, token = token, streaming = True
                ).take(MIN_EVAL_ROWS)
            )
        except Exception:
            # Uncounted: the eval upload reads it anyway and reports a failure there.
            return candidate
        # The trainer skips a split too small to evaluate on and tries the next.
        if rows >= MIN_EVAL_ROWS:
            return candidate
    return None


async def hub_eval_upload(ctx: NodeContext, dataset: dict, split: str) -> str:
    from core.flows.nodes.data import upload_rows
    from core.training.account_jobs import account_hf_token

    token = account_hf_token(getattr(ctx.caller, "hf_token", None))
    try:
        rows = await _off_thread(ctx, _eval_rows, dataset, token, split)
    except (FlowNodeError, FlowCancelled):
        raise
    except Exception as exc:
        raise FlowNodeError(f"Could not read the evaluation split {split!r}: {exc}") from None
    name = str(dataset.get("ref") or "dataset").replace("/", "-")
    return (await upload_rows(ctx, rows, f"{name}-{split}-eval"))["ref"]


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
        system = str(ctx.params.get("system_prompt") or "").strip() or str(
            checkpoint.get("system_prompt") or ""
        )
        text = await ctx.llm_text(
            model = None,
            system = system,
            prompt = prompt,
            max_tokens = int(ctx.params.get("max_tokens") or 512),
            seed = ctx.seed,
            use_adapter = True if trained.is_lora else None,
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
    found = {"repo": model, "variant": status.get("gguf_variant") or None}
    # How it was loaded, so putting it back keeps its context and precision.
    context = status.get("requested_context_length")
    # 0 is a request too: automatic sizing.
    if isinstance(context, int) and not isinstance(context, bool) and context >= 0:
        found["max_seq_length"] = context
    if isinstance(status.get("load_in_4bit"), bool):
        found["load_in_4bit"] = status["load_in_4bit"]
    extra = {}
    # A model served by vLLM or SGLang goes back on that engine, with its launch settings.
    if status.get("engine") in ("vllm", "sglang"):
        extra.update({k: status[k] for k in ENGINE_FIELDS if status.get(k) is not None})
    # A GGUF placed by hand (all in RAM at gpu_layers 0, say) goes back where it was.
    if status.get("gpu_memory_mode") == "manual":
        extra.update({k: status[k] for k in PLACEMENT_FIELDS if status.get(k) is not None})
    # The GPUs it was asked to use (tensor parallel over two, or pinned to another card); a status
    # from before requested_gpu_ids reports only the effective ones. None means automatic.
    gpus = status.get("requested_gpu_ids", status.get("gpu_ids"))
    if isinstance(gpus, list) and gpus:
        extra["gpu_ids"] = gpus
    template = status.get("chat_template_override")
    if isinstance(template, str) and template.strip():
        # A template the user set to fix the model's prompt format goes back with it.
        extra["chat_template_override"] = template
    if extra:
        found["extra"] = extra
    return found


ENGINE_FIELDS = (
    "engine",
    "engine_parallelism",
    "engine_precision",
    "engine_kv_cache_dtype",
    "engine_seed",
    "engine_gpu_memory_utilization",
)
PLACEMENT_FIELDS = ("gpu_memory_mode", "gpu_layers", "n_cpu_moe")


async def restore_chat_model(ctx: NodeContext, previous: Optional[dict], replaced_by: str) -> None:
    from studio_mcp import loading

    if not previous or loading._same_id(previous["repo"], replaced_by):
        return
    label = previous["repo"] + (f" ({previous['variant']})" if previous["variant"] else "")
    if getattr(ctx, "cancelled", False):
        # A stopped run must not hold the GPU lane for a reload that can take minutes.
        ctx.log(f"Stopped before {label} was loaded again; load it before the next run", "warning")
        return
    ctx.set_status("loading_model", f"Loading {label} again")
    try:
        await loading.load_llm(
            ctx.caller,
            None,
            model = previous["repo"],
            variant = previous["variant"],
            max_seq_length = previous.get("max_seq_length", FLOW_CHAT_CONTEXT),
            load_in_4bit = previous.get("load_in_4bit", True),
            hf_token = None,
            kind = "llm",
            deadline = time.monotonic() + 1800,
            extra_body = previous.get("extra") or None,
        )
        note_loaded(ctx, "llm", previous["repo"])
        ctx.resident.clear()
    except Exception as exc:
        ctx.log(f"Could not load {label} again; load it before the next run ({exc})", "warning")
