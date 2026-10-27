# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Several queued LoRA jobs trained together on one base model with the fused multi-LoRA kernels.

Each member keeps its own dataset, rank, alpha, learning rate, epochs, seed, optimizer and output, and
reports its own events (``member_run_id``) with the same shapes a lone job sends."""

from __future__ import annotations

import copy
import math
import os
import queue
import time
import traceback
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from loggers import get_logger

logger = get_logger(__name__)

_DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_OPTIMIZERS = ("adamw_8bit", "adamw_torch", "adamw_torch_fused")


@dataclass
class _Member:
    slot: int
    run_id: str
    config: dict
    tokenizer: Any = None
    rows: list = field(default_factory = list)
    output_dir: str = ""
    total_steps: int = 0
    steps_per_epoch: int = 1
    step: int = 0
    num_tokens: int = 0
    windows: Optional[Iterator] = None
    optimizer: Any = None
    scheduler: Any = None
    params: list = field(default_factory = list)
    done: bool = False


def _send(
    event_queue,
    kind: str,
    member: Optional[_Member] = None,
    **payload,
) -> None:
    event = {"type": kind, **payload, "ts": time.time()}
    if member is not None:
        event["member_run_id"] = member.run_id
    event_queue.put(event)


def _new_trainer():
    from core.training.trainer import UnslothTrainer
    return UnslothTrainer()


def _detect_model(trainer, config: dict, hf_token) -> None:
    from core.training import worker

    model_load_name = worker._resolve_cached_model_load_name(config)
    local_only = worker._model_local_files_only(config)
    worker._pre_detect_training_model(
        trainer,
        config,
        config["model_name"],
        hf_token,
        model_load_name,
        local_only,
        None if local_only else config.get("model_revision"),
    )


def _format_dataset(trainer, config: dict, hf_token):
    from core.training import worker

    hf_dataset = config.get("hf_dataset", "")
    result = trainer.load_and_format_dataset(
        dataset_source = hf_dataset if hf_dataset and hf_dataset.strip() else None,
        format_type = config.get("format_type", ""),
        local_datasets = config.get("local_datasets") or None,
        custom_format_mapping = config.get("custom_format_mapping"),
        subset = config.get("subset"),
        train_split = config.get("train_split", "train"),
        dataset_slice_start = config.get("dataset_slice_start"),
        dataset_slice_end = config.get("dataset_slice_end"),
        s3_config = config.get("s3_config"),
        dataset_local_files_only = worker._dataset_local_files_only(config),
        dataset_local_path = config.get("dataset_snapshot_path"),
        dataset_revision = config.get("dataset_revision"),
        hf_token = hf_token,
        max_train_rows_seed = config.get("random_seed", 3407),
    )
    if result is None:
        raise RuntimeError(trainer.training_progress.error or "Failed to load dataset")
    return result[0] if isinstance(result, tuple) else result


def _load_base(trainer, config: dict, hf_token, event_queue) -> Optional[bool]:
    """The single job's LoRA load; returns the effective 4-bit flag, or None when stopped."""
    from core.training import worker
    from utils.hf_xet_fallback import start_watchdog

    model_name = config["model_name"]
    model_load_name = worker._resolve_cached_model_load_name(config)
    local_only = worker._model_local_files_only(config)
    base_error = worker._adapter_base_changed_error(config) or worker._escaping_weights_error(
        config
    )
    if base_error:
        raise RuntimeError(base_error)
    event_queue.put({"type": "model_load_started", "ts": time.time()})
    watchdog = start_watchdog(
        repo_ids = [model_name],
        on_stall = lambda msg: event_queue.put({"type": "stall", "message": msg, "ts": time.time()}),
        xet_disabled = os.environ.get("HF_HUB_DISABLE_XET") == "1",
    )

    def report_repo(repo_id):
        if not os.path.isdir(os.path.expanduser(repo_id)):
            event_queue.put({"type": "model_load_resolved", "repo_id": repo_id, "ts": time.time()})

    try:
        load_in_4bit = worker._effective_training_load_in_4bit(config, model_load_name, hf_token)
        success = trainer.load_model(
            on_model_resolved = report_repo,
            model_name = model_name,
            max_seq_length = config["max_seq_length"],
            load_in_4bit = load_in_4bit,
            full_finetuning = False,
            hf_token = hf_token,
            trust_remote_code = config.get("trust_remote_code", False),
            gpu_ids = config.get("resolved_gpu_ids"),
            model_load_name = model_load_name,
            local_files_only = local_only,
            actual_model_repo_id = config.get("actual_model_repo_id"),
            model_revision = None if local_only else config.get("model_revision"),
            use_gradient_checkpointing = config.get("gradient_checkpointing", "unsloth"),
            offload_layers = config.get("offload_layers", 0),
            prefetch_depth = config.get("prefetch_depth", 2),
            offload_plan_shape = worker._offload_plan_shape(config),
        )
    finally:
        watchdog.set()
        event_queue.put({"type": "model_load_completed", "ts": time.time()})
    if trainer.should_stop:
        return None
    if not success:
        raise RuntimeError(
            worker._with_vram_budget_hint(
                config, trainer.training_progress.error or "Failed to load model"
            )
        )
    worker._emit_resource_provenance(
        event_queue,
        config,
        trainer.model,
        model_load_target = model_load_name,
        model_load_in_4bit = load_in_4bit,
        dataset_loaded_from_exact_snapshot = bool(
            getattr(trainer, "dataset_loaded_from_exact_snapshot", False)
        ),
    )
    return load_in_4bit


def _is_moe(model) -> bool:
    """unsloth.models._utils.is_moe_model, without importing unsloth's package root."""
    config = getattr(model, "config", None)
    for cfg in (config, getattr(config, "text_config", None)):
        count = next(
            (
                getattr(cfg, k)
                for k in ("num_experts", "n_routed_experts", "num_local_experts")
                if getattr(cfg, k, None) is not None
            ),
            None,
        )
        if count is not None:
            return isinstance(count, int) and count > 0
    return False


def _resolve_targets(model, target_modules) -> list[str]:
    """Leaf names as PEFT resolves them: an explicit list, the Studio default, or every linear layer
    except the output head for ``all-linear``."""
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    target_modules = list(target_modules or [])
    # As the single job: explicit names beside "all-linear" win; "all-linear" alone means every linear layer.
    explicit = [m for m in target_modules if m != "all-linear"]
    if explicit or not target_modules:
        return explicit or list(_DEFAULT_TARGETS)
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    names = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if name
        and module is not head
        and hasattr(module, "in_features")
        and hasattr(module, "weight")
        and not list(module.children())
    }
    return sorted(names)


def _multi_lora_layers(model):
    from unsloth.kernels.multi_lora import MultiLoraLinear
    return [(n, m) for n, m in model.named_modules() if isinstance(m, MultiLoraLinear)]


def _seed_adapter(model, slot: int, seed: int) -> None:
    """Kaiming A from the member's own seed, B zero: the same init whatever the batch order."""
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for _, layer in _multi_lora_layers(model):
            A = torch.empty(layer.lora_A[slot].shape, dtype = torch.float32)
            torch.nn.init.kaiming_uniform_(A, a = math.sqrt(5), generator = generator)
            layer.lora_A[slot].copy_(A)
            layer.lora_B[slot].zero_()


def _enable_training(model, gradient_checkpointing, max_seq_length: int) -> None:
    """What get_peft_model does for checkpointing on a LoRA model, minus the PEFT wrapper."""
    from unsloth.models._utils import (
        apply_unsloth_gradient_checkpointing,
        prepare_model_for_kbit_training,
    )

    dtype = model.get_input_embeddings().weight.dtype
    gradient_checkpointing = apply_unsloth_gradient_checkpointing(
        gradient_checkpointing, max_seq_length, dtype
    )
    model._unsloth_gradient_checkpointing = gradient_checkpointing
    prepare_model_for_kbit_training(
        model, use_gradient_checkpointing = gradient_checkpointing, use_reentrant = True
    )
    for_training = getattr(model, "for_training", None)
    if callable(for_training):
        for_training(use_gradient_checkpointing = gradient_checkpointing)
    else:
        model.train()


def _check_routed(model, device) -> None:
    """One tiny forward must reach every injected layer, else the patched forwards bypass them."""
    import torch
    from unsloth.kernels.multi_lora import set_multi_lora_batch

    layers = _multi_lora_layers(model)
    hits = set()
    handles = [
        layer.register_forward_hook(lambda m, i, o, name = name: hits.add(name))
        for name, layer in layers
    ]
    try:
        ids = torch.zeros(1, 2, dtype = torch.long, device = device)
        set_multi_lora_batch(model, torch.zeros(1, dtype = torch.long, device = device), seq_len = 2)
        # No grad here, so checkpointed layers warn that their inputs need none.
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model(input_ids = ids, attention_mask = torch.ones_like(ids), use_cache = False)
    finally:
        for handle in handles:
            handle.remove()
        for _, layer in layers:
            layer.batch = layer.seen = layer.trained = None
    missed = [name for name, _ in layers if name not in hits]
    if missed:
        raise RuntimeError(
            f"Batched training cannot run on this model: its forward skips {len(missed)} adapter "
            f"layers (e.g. {missed[0]})."
        )


def _member_tokenizer(tokenizer, model_name: str, final_format: str):
    from utils.datasets.chat_templates import get_training_chat_template
    if getattr(tokenizer, "chat_template", None) or final_format not in (
        "alpaca",
        "chatml_messages",
        "chatml_conversations",
    ):
        return tokenizer
    return get_training_chat_template(copy.deepcopy(tokenizer), model_name, final_format)


def _tokenize_member(
    dataset_info,
    tokenizer,
    config: dict,
    model_name: str,
    max_length: int,
    vocab_size = None,
):
    """(tokenizer to save, [(input_ids, labels)]) with the single SFT path's tokenization and masking."""
    from datasets import Dataset

    from utils.datasets.completion_masking import apply_completion_masking
    from utils.datasets.online_tokenization import (
        build_tokenizing_transform,
        first_sample_text,
        resolve_add_special_tokens,
    )
    from utils.datasets.pretokenized import (
        PRETOKENIZED_FORMAT,
        _collator_masks_user_labels,
        prepare_pretokenized_dataset,
    )

    dataset = dataset_info["dataset"] if isinstance(dataset_info, dict) else dataset_info
    final_format = (
        str(dataset_info.get("final_format", "")).lower() if isinstance(dataset_info, dict) else ""
    )
    member_tok = _member_tokenizer(tokenizer, model_name, final_format)
    labels = None
    if final_format == PRETOKENIZED_FORMAT:
        # The lone job's checks (ids inside the vocabulary, mask and label lengths) and truncation: a bad
        # dataset then fails only its own member, never the shared forward.
        dataset = prepare_pretokenized_dataset(
            dataset, max_seq_length = max_length, vocab_size = vocab_size
        )
        masks = dataset["assistant_masks"] if "assistant_masks" in dataset.column_names else None
        input_ids = [list(x)[:max_length] for x in dataset["input_ids"]]
        if "labels" in dataset.column_names:
            labels = [list(x)[:max_length] for x in dataset["labels"]]
            if not _collator_masks_user_labels():
                masks = None
        if masks is not None:
            # TRL's collator trains only the assistant tokens, on top of user labels before TRL 1.7.
            labels = [
                [t if m else -100 for t, m in zip(row, mask)]
                for row, mask in zip(labels if labels is not None else input_ids, masks)
            ]
    else:
        add_special_tokens = resolve_add_special_tokens(
            member_tok, first_sample_text(dataset, "text")
        )
        encode = build_tokenizing_transform(member_tok, "text", max_length, add_special_tokens)
        input_ids = [list(x) for x in encode({"text": list(dataset["text"])})["input_ids"]]
    raw = final_format == "raw_text" or config.get("format_type") == "raw"
    if labels is None and config.get("train_on_completions") and not raw:

        def build_masker(holder, **kwargs):
            from unsloth.chat_templates import train_on_responses_only

            kwargs.pop("num_proc", None)
            masker = train_on_responses_only(
                None, tokenizer = holder.processing_class, return_function = True, **kwargs
            )
            holder.train_dataset = holder.train_dataset.map(masker, batched = True)
            return holder

        holder, applied = apply_completion_masking(
            types.SimpleNamespace(
                processing_class = member_tok,
                train_dataset = Dataset.from_dict({"input_ids": input_ids}),
                eval_dataset = None,
            ),
            model_name,
            build_masker,
            notify = lambda level, message: logger.info(message),
            dataset_template = "alpaca" if final_format == "alpaca" else None,
        )
        if applied:
            labels = [list(x) for x in holder.train_dataset["labels"]]
            kept = [i for i, row in enumerate(labels) if any(t != -100 for t in row[1:])]
            dropped = len(labels) - len(kept)
            drop_pct = round(100 * dropped / len(labels), 1) if labels else 0
            if not kept or drop_pct > 30:
                raise ValueError(
                    f"{dropped}/{len(labels)} samples ({drop_pct}%) were dropped after applying "
                    f"'Train on completions': after masking, those rows had no trainable response "
                    f"tokens left. The usual cause is that this model's response template was not "
                    f"found in the formatted samples, so 'Train on completions' should be turned "
                    f"off for this dataset."
                )
            input_ids = [input_ids[i] for i in kept]
            labels = [labels[i] for i in kept]
    if labels is None:
        labels = [list(x) for x in input_ids]
    rows = [(ids, lab) for ids, lab in zip(input_ids, labels) if len(ids) > 1]
    if not rows:
        raise ValueError("The dataset has no rows to train on after tokenization.")
    return member_tok, rows


def _windows(n_rows: int, batch_size: int, grad_accum: int, seed: int):
    """Endless (epoch, [microbatch row indices] * <= grad_accum); windows never span an epoch."""
    import torch

    epoch = 0
    while True:
        order = torch.randperm(n_rows, generator = torch.Generator().manual_seed(seed + epoch))
        order = order.tolist()
        micro = [order[i : i + batch_size] for i in range(0, n_rows, batch_size)]
        for i in range(0, len(micro), grad_accum):
            yield epoch, micro[i : i + grad_accum]
        epoch += 1


def _member_scheduler(config: dict, lr: float, total_steps: int):
    """The single job's schedule for this member, on a stand-in optimizer: it only yields learning rates."""
    import torch
    from transformers import get_scheduler

    warmup_ratio, warmup_steps = config.get("warmup_ratio"), config.get("warmup_steps")
    if warmup_ratio is not None:
        warmup = math.ceil(total_steps * float(warmup_ratio))
    elif warmup_steps is not None:
        warmup = int(warmup_steps)
    else:
        warmup = 5
    stand_in = torch.optim.SGD([torch.zeros(1, requires_grad = True)], lr = lr)
    return get_scheduler(
        config.get("lr_scheduler_type", "linear") or "linear",
        stand_in,
        num_warmup_steps = warmup,
        num_training_steps = total_steps,
    )


class _PerMemberLR:
    """Steps every member's own schedule and writes its rate into that member's parameter group."""

    def __init__(self, optimizer, schedulers: list):
        self.optimizer = optimizer
        self.schedulers = schedulers
        self._apply()

    def _apply(self) -> None:
        for group, scheduler in zip(self.optimizer.param_groups, self.schedulers):
            group["lr"] = scheduler.get_last_lr()[0]

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()
        self._apply()

    def get_last_lr(self) -> list:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


def _collate(rows: list, slots: list, pad_id: int) -> dict:
    import torch

    T = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), T), pad_id, dtype = torch.long)
    labels = torch.full((len(rows), T), -100, dtype = torch.long)
    attention = torch.zeros((len(rows), T), dtype = torch.long)
    for i, (ids, lab) in enumerate(rows):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        attention[i, : len(ids)] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": labels,
        "adapter_ids": torch.tensor(slots, dtype = torch.long),
    }


def _unique_output_dir(config: dict, model_name: str, taken: set) -> str:
    from utils.paths import resolve_output_dir
    from utils.training_runs import build_default_output_dir_name

    explicit = config.get("output_dir")
    stamp = int(time.time())
    for attempt in range(1000):
        if explicit:
            name = explicit if attempt == 0 else f"{explicit}-{attempt}"
        else:
            # A later timestamp, not a suffix: the folder name is parsed back by its trailing timestamp.
            name = build_default_output_dir_name(
                model_name, config.get("project_name"), timestamp = stamp + attempt
            )
        path = str(resolve_output_dir(name))
        if path not in taken and not os.path.exists(path):
            taken.add(path)
            return path
    raise RuntimeError("Could not reserve an output folder for a batched job.")


def _export(
    model,
    member: _Member,
    trainer,
    targets: list,
    use_rslora: bool,
    path: str = "",
) -> None:
    from peft import LoraConfig
    from safetensors.torch import save_file

    from core.training.trainer import UnslothTrainer
    from unsloth.kernels.multi_lora import adapter_state_dict
    from utils.paths import ensure_dir

    path = path or member.output_dir
    ensure_dir(Path(path))
    state = {
        k: v.detach().to("cpu").contiguous()
        for k, v in adapter_state_dict(model, member.slot).items()
    }
    save_file(
        state,
        os.path.join(path, "adapter_model.safetensors"),
        metadata = {"format": "pt"},
    )
    LoraConfig(
        r = int(member.config.get("lora_r", 16)),
        lora_alpha = member.config.get("lora_alpha", 16),
        lora_dropout = 0.0,
        bias = "none",
        target_modules = targets,
        use_rslora = use_rslora,
        task_type = "CAUSAL_LM",
        base_model_name_or_path = getattr(model.config, "_name_or_path", None),
    ).save_pretrained(path)
    member.tokenizer.save_pretrained(path)
    UnslothTrainer._patch_adapter_config(trainer, path)


def _status_forwarder(event_queue):
    from core.training.worker import _send_status

    sent: set = set()

    def forward(progress) -> None:
        if progress.status_message:
            _send_status(event_queue, progress.status_message)
        for message in progress.warnings:
            if message not in sent:
                sent.add(message)
                event_queue.put({"type": "warning", "message": message, "ts": time.time()})

    return forward


def _start_stop_reader(stop_queue, inbox: "queue.Queue", trainer) -> None:
    from utils.account_context import account_thread
    def read():
        while True:
            try:
                message = stop_queue.get(timeout = 1.0)
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError):
                return
            if not isinstance(message, dict):
                continue
            if message.get("type") == "stop":
                trainer.should_stop = True
                trainer.save_on_stop = bool(message.get("save", True))
            if message.get("type") in ("stop", "stop_member"):
                inbox.put(message)

    account_thread(target = read, daemon = True).start()


def _oom_message(exc: BaseException) -> Optional[str]:
    text = str(exc).lower()
    if "out of memory" in text or type(exc).__name__ == "OutOfMemoryError":
        return (
            "GPU ran out of VRAM while training the batched jobs.\n"
            "To fix: reduce max_seq_length or the batch size of the queued jobs."
        )
    return None


def run_multi_lora_training(event_queue: Any, stop_queue: Any, config: dict) -> None:
    """Worker entry for ``config["batch_members"]`` (same event and stop contract as one job)."""
    members = [
        _Member(slot = -1, run_id = m["run_id"], config = m["config"]) for m in config["batch_members"]
    ]
    try:
        _run(event_queue, stop_queue, config, members)
    except Exception as exc:
        message = _oom_message(exc) or str(exc)
        stack = traceback.format_exc(limit = 20)
        logger.error("Batched training failed: %s", exc)
        for member in members:
            if not member.done:
                member.done = True
                _send(event_queue, "error", member, error = message, stack = stack)
        event_queue.put({"type": "error", "error": message, "stack": stack, "ts": time.time()})


def _finish_stopped(event_queue, members, save: bool, export) -> None:
    for member in members:
        if member.done:
            continue
        member.done = True
        saved = bool(save and member.step > 0 and member.output_dir)
        if saved:
            export(member)
        _send(
            event_queue,
            "stopped",
            member,
            output_dir = member.output_dir if saved else None,
            status_message = (
                f"Training stopped. Model saved to {member.output_dir}"
                if saved
                else "Training cancelled."
            ),
            saved = saved,
        )


def _run(event_queue, stop_queue, config: dict, members: list) -> None:
    from core.import_guards import ensure_real_packages

    ensure_real_packages("unsloth_zoo", "unsloth")
    import torch

    from core.training.trainer import normalize_gradient_checkpointing
    from core.training.worker import _send_status, _worker_hf_token
    from unsloth.kernels.multi_lora import inject_multi_lora

    trainer = _new_trainer()
    if hasattr(trainer, "add_progress_callback"):
        trainer.add_progress_callback(_status_forwarder(event_queue))
    inbox: queue.Queue = queue.Queue()
    _start_stop_reader(stop_queue, inbox, trainer)

    def stopped_early() -> bool:
        if not trainer.should_stop:
            return False
        _finish_stopped(event_queue, members, False, None)
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return True

    hf_token = _worker_hf_token(config)
    model_name = config["model_name"]
    _send_status(event_queue, "Detecting model type...")
    _detect_model(trainer, config, hf_token)
    if stopped_early():
        return

    datasets = {}
    for i, member in enumerate(members):
        _send_status(event_queue, f"Loading and formatting dataset {i + 1}/{len(members)}...")
        try:
            datasets[member.run_id] = _format_dataset(trainer, member.config, hf_token)
        except Exception as exc:
            member.done = True
            _send(
                event_queue, "error", member, error = str(exc), stack = traceback.format_exc(limit = 20)
            )
        if stopped_early():
            return
    if not datasets:
        raise RuntimeError("None of the batched jobs could load its dataset.")

    _send_status(event_queue, "Loading model...")
    load_in_4bit = _load_base(trainer, config, hf_token, event_queue)
    if stopped_early():
        return
    if getattr(trainer, "diffusion_lm_profile", None):
        raise RuntimeError(
            "Batched training does not support text-diffusion models; train these jobs one at a time."
        )
    if _is_moe(trainer.model):
        # The single-job path trains fused experts through PEFT target_parameters; these layers cannot.
        raise RuntimeError(
            "Batched training does not support MoE models; train these jobs one at a time."
        )
    if getattr(trainer.model, "peft_config", None):
        # A saved LoRA loads as a PeftModel; the multi-LoRA layers would train fresh adapters and drop its weights.
        raise RuntimeError(
            "Batched training cannot continue an existing LoRA adapter; train these jobs one at a time."
        )
    trainer.load_in_4bit = load_in_4bit
    trainer.is_cpt = False
    model, tokenizer = trainer.model, trainer.tokenizer
    max_length = int(config.get("max_seq_length") or 2048)
    model_cap = getattr(model, "max_seq_length", None)
    if isinstance(model_cap, int) and 0 < model_cap < max_length:
        max_length = model_cap

    _send_status(event_queue, "Tokenizing datasets...")
    for member in members:
        if member.done:
            continue
        try:
            member.tokenizer, member.rows = _tokenize_member(
                datasets.pop(member.run_id),
                tokenizer,
                member.config,
                model_name,
                max_length,
                vocab_size = int(model.get_input_embeddings().num_embeddings),
            )
        except Exception as exc:
            member.done = True
            _send(
                event_queue, "error", member, error = str(exc), stack = traceback.format_exc(limit = 20)
            )
    live = [m for m in members if not m.done]
    if not live:
        raise RuntimeError("None of the batched jobs has rows to train on.")
    for slot, member in enumerate(live):
        member.slot = slot
    U = len(live)

    _send_status(event_queue, f"Configuring {U} LoRA adapters...")
    targets = _resolve_targets(model, config.get("target_modules"))
    use_rslora = bool(config.get("use_rslora", False))
    injected = inject_multi_lora(
        model,
        targets,
        U,
        r = [int(m.config.get("lora_r", 16)) for m in live],
        lora_alpha = [float(m.config.get("lora_alpha", 16)) for m in live],
        use_rslora = use_rslora,
    )
    targets = sorted({name.rsplit(".", 1)[-1] for name in injected})
    for member in live:
        _seed_adapter(model, member.slot, member.config.get("random_seed", 3407))
    _enable_training(
        model,
        normalize_gradient_checkpointing(config.get("gradient_checkpointing", "unsloth")),
        max_length,
    )
    device = model.get_input_embeddings().weight.device
    _check_routed(model, device)

    layers = _multi_lora_layers(model)
    grad_accum = max(int(config.get("gradient_accumulation_steps", 4) or 1), 1)
    taken: set = set()
    for member in live:
        cfg = member.config
        batch_size = max(int(cfg.get("batch_size", 2) or 1), 1)
        n_micro = math.ceil(len(member.rows) / batch_size)
        member.steps_per_epoch = max(math.ceil(n_micro / grad_accum), 1)
        max_steps = int(cfg.get("max_steps") or 0)
        member.total_steps = (
            max_steps
            if max_steps > 0
            else member.steps_per_epoch * int(cfg.get("num_epochs", 3) or 1)
        )
        member.windows = _windows(
            len(member.rows), batch_size, grad_accum, int(cfg.get("random_seed", 3407))
        )
        member.params = [
            p for _, layer in layers for p in (layer.lora_A[member.slot], layer.lora_B[member.slot])
        ]
        member.output_dir = _unique_output_dir(cfg, model_name, taken)
        _send(event_queue, "output_dir", member, output_dir = member.output_dir)

    def export(member: _Member, path: str = "") -> None:
        _export(model, member, trainer, targets, use_rslora, path)

    def take_messages() -> Optional[bool]:
        """Applies queued member stops; returns the save flag of a whole-batch stop, if one arrived."""
        save_all = None
        while not inbox.empty():
            message = inbox.get_nowait()
            if message.get("type") == "stop":
                save_all = bool(message.get("save", True))
                continue
            target = next(
                (m for m in live if m.run_id == message.get("run_id") and not m.done), None
            )
            if target is not None:
                _finish_stopped(event_queue, [target], bool(message.get("save", True)), export)
        return save_all

    def stop_all(save: bool) -> None:
        _finish_stopped(event_queue, live, save, export)
        event_queue.put(
            {
                "type": "complete",
                "output_dir": None,
                "status_message": "Training stopped",
                "ts": time.time(),
            }
        )

    save_all = take_messages()
    if save_all is not None:
        stop_all(save_all)
        return
    if all(m.done for m in live):
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    sft = _build_sft_trainer(
        model,
        tokenizer,
        config,
        live,
        grad_accum,
        device,
        event_queue,
        export,
        take_messages,
        stop_all,
    )
    for member in live:
        if not member.done:
            sft.progress(member, None, None, 0.0, sft.step_lrs[member.slot])
    _send_status(event_queue, f'Training {U} jobs together on "{model_name}"')
    sft.train()
    if not sft.stopped:
        event_queue.put(
            {
                "type": "complete",
                "output_dir": None,
                "status_message": "Training completed",
                "ts": time.time(),
            }
        )


def _build_sft_trainer(
    model, tokenizer, config, live, grad_accum, device, event_queue, export, take_messages, stop_all
):
    """TRL's SFTTrainer over the batch: each optimizer step gives every running member its own next
    window, losses and gradient clipping stay per adapter, and each member keeps its own learning rate
    schedule (one parameter group each)."""
    import torch
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from core.training.resume import session_eta_seconds
    from core.training.training import apply_save_strategy
    from unsloth.kernels.multi_lora import (
        adapter_token_counts,
        clip_grad_norm_per_adapter_,
        drop_absent_adapter_grads_,
        per_adapter_loss,
        set_multi_lora_batch,
    )

    U = len(live)
    total = max(m.total_steps for m in live if not m.done)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    pad_id = 0 if pad_id is None else int(pad_id)
    bf16 = fp16 = False
    if device.type == "cuda":
        from unsloth import is_bfloat16_supported
        bf16 = is_bfloat16_supported()
        fp16 = not bf16
    args = {
        "output_dir": live[0].output_dir,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": grad_accum,
        "max_steps": total,
        "learning_rate": float(live[0].config.get("learning_rate", 2e-4)),
        "max_grad_norm": float(config.get("max_grad_norm", 1.0) or 1.0),
        "bf16": bf16,
        "fp16": fp16,
        "logging_strategy": "no",
        "report_to": "none",
        "disable_tqdm": True,
        "seed": int(live[0].config.get("random_seed", 3407)),
        "remove_unused_columns": False,
        "gradient_checkpointing": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "dataloader_num_workers": 0,
    }
    apply_save_strategy(args, config.get("save_steps", 0))
    micro_total = total * grad_accum

    class _Steps(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            sft.after_step(control)

    class MultiLoraSFTTrainer(SFTTrainer):
        def __init__(self, **kwargs):
            # Trainer refuses a quantized base without a PEFT wrapper; the adapters here are injected.
            loaded = getattr(model, "_hf_peft_config_loaded", None)
            model._hf_peft_config_loaded = True
            try:
                super().__init__(**kwargs)
            finally:
                if loaded is None:
                    del model._hf_peft_config_loaded
                else:
                    model._hf_peft_config_loaded = loaded
            # compute_loss already normalizes each adapter by its own window's tokens.
            self.model_accepts_loss_kwargs = True
            # Every Trainer clips through this one call (4.x inline, 5.x via _clip_grad_norm): clip per member.
            self.accelerator.clip_grad_norm_ = self._clip_members
            self.stopped = False
            self.round = []
            self.round_losses = torch.zeros(U)
            self.member_norms = torch.zeros(U)
            self.step_lrs = [float(m.config.get("learning_rate", 2e-4)) for m in live]
            self.start = time.time()

        def get_train_dataloader(self):
            return DataLoader(range(micro_total), batch_size = 1)

        def create_optimizer(self, model = None):
            if self.optimizer is None:
                groups = [
                    {
                        "params": m.params,
                        "lr": float(m.config.get("learning_rate", 2e-4)),
                        "weight_decay": float(m.config.get("weight_decay", 0.001) or 0.0),
                    }
                    for m in live
                ]
                optim = config.get("optim", "adamw_8bit") or "adamw_8bit"
                if optim == "adamw_8bit":
                    import bitsandbytes as bnb
                    self.optimizer = bnb.optim.AdamW8bit(groups)
                elif optim in ("adamw_torch", "adamw_torch_fused"):
                    fused = optim == "adamw_torch_fused" and device.type == "cuda"
                    self.optimizer = torch.optim.AdamW(groups, fused = fused)
                else:
                    raise ValueError(f"Batched training does not support the optimizer {optim!r}.")
            return self.optimizer

        def create_scheduler(
            self,
            num_training_steps,
            optimizer = None,
        ):
            if self.lr_scheduler is None:
                self.lr_scheduler = _PerMemberLR(
                    optimizer or self.optimizer,
                    [
                        _member_scheduler(
                            m.config, float(m.config.get("learning_rate", 2e-4)), m.total_steps
                        )
                        for m in live
                    ],
                )
            return self.lr_scheduler

        def get_batch_samples(self, epoch_iterator, num_batches, device):
            """One optimizer step: micro-batch k holds every running member's k-th micro-batch."""
            self.round = [m for m in live if not m.done]
            if not self.round:
                return [], None
            windows = {m.slot: next(m.windows)[1] for m in self.round}
            samples = []
            counts = None
            for k in range(num_batches):
                rows, slots = [], []
                for member in self.round:
                    window = windows[member.slot]
                    if k < len(window):
                        picked = [member.rows[r] for r in window[k]]
                        member.num_tokens += sum(len(ids) for ids, _ in picked)
                        rows.extend(picked)
                        slots.extend([member.slot] * len(picked))
                if not rows:
                    # Every member's window is shorter here (an epoch end); the step keeps its size.
                    samples.append({"multi_lora_empty": torch.zeros(1)})
                    continue
                batch = _collate(rows, slots, pad_id)
                found = adapter_token_counts(batch["labels"], batch["adapter_ids"], U).cpu()
                counts = found if counts is None else counts + found
                samples.append(batch)
            self.round_losses = torch.zeros(U)
            return samples, counts.to(device)

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs = False,
            num_items_in_batch = None,
        ):
            if "multi_lora_empty" in inputs:
                return torch.zeros((), device = self.args.device, requires_grad = True)
            ids = inputs["adapter_ids"]
            set_multi_lora_batch(self.model, ids, seq_len = inputs["input_ids"].shape[1])
            logits = model(
                input_ids = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                use_cache = False,
            ).logits
            loss, per = per_adapter_loss(
                logits,
                inputs["labels"],
                ids,
                U,
                model = self.model,
                num_items_in_batch = num_items_in_batch,
            )
            self.round_losses += per.detach().float().cpu()
            return loss

        def _clip_members(
            self,
            parameters,
            max_norm,
            norm_type = 2,
            **kwargs,
        ):
            # The rate this step uses, as a lone job reports it (read before the scheduler moves).
            self.step_lrs = [group["lr"] for group in self.optimizer.param_groups]
            drop_absent_adapter_grads_(self.model)
            self.accelerator.unscale_gradients()
            self.member_norms = clip_grad_norm_per_adapter_(self.model, max_norm).float().cpu()
            return self.member_norms.norm()

        def _save_checkpoint(self, model, trial):
            name = f"checkpoint-{self.state.global_step}"
            for member in self.round:
                export(member, os.path.join(member.output_dir, name))

        def progress(self, member, loss, grad_norm, elapsed, lr) -> None:
            _send(
                event_queue,
                "progress",
                member,
                step = member.step,
                epoch = round(member.step / member.steps_per_epoch, 2),
                loss = loss,
                learning_rate = lr,
                total_steps = member.total_steps,
                elapsed_seconds = elapsed,
                eta_seconds = session_eta_seconds(elapsed, member.step, 0, member.total_steps),
                session_start_step = 0,
                grad_norm = grad_norm,
                num_tokens = member.num_tokens,
                eval_loss = None,
                status_message = "",
            )

        def after_step(self, control) -> None:
            elapsed = time.time() - self.start
            for member in self.round:
                member.step += 1
                self.progress(
                    member,
                    float(self.round_losses[member.slot]),
                    float(self.member_norms[member.slot]),
                    elapsed,
                    self.step_lrs[member.slot],
                )
                if member.step >= member.total_steps and not member.done:
                    export(member)
                    member.done = True
                    _send(
                        event_queue,
                        "complete",
                        member,
                        output_dir = member.output_dir,
                        status_message = f"Training completed! Model saved to {member.output_dir}",
                    )
            save_all = take_messages()
            if save_all is not None:
                stop_all(save_all)
                self.stopped = True
            if self.stopped or all(m.done for m in live):
                control.should_training_stop = True

    sft = MultiLoraSFTTrainer(
        model = model,
        args = SFTConfig(**args),
        train_dataset = Dataset.from_dict({"input_ids": [[pad_id]] * micro_total}),
        processing_class = tokenizer,
        data_collator = lambda rows: rows,
        callbacks = [_Steps()],
    )
    return sft
