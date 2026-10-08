# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# FastDecisionModel and DecisionTrainer on Apple Silicon, over unsloth-zoo's MLX decision models.
# Loaded by path as unsloth._decision_mlx, because importing unsloth.models needs torch.

__all__ = [
    "FastDecisionModel",
    "DecisionTrainer",
]

import copy
import dataclasses
import functools
import importlib.util
import json
import random
import sys
import tempfile
import types
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

_common_spec = importlib.util.spec_from_file_location(
    "unsloth._decision_common", Path(__file__).with_name("_decision_common.py")
)
sys.modules[_common_spec.name] = importlib.util.module_from_spec(_common_spec)
_common_spec.loader.exec_module(sys.modules[_common_spec.name])
from unsloth._decision_common import (
    CLEF_MAX_LEN,
    CLEF_SERVE_MAX_LEN,
    DecisionDataError,
    HOLDOUT_MAX,
    QUESTION_TYPES,
    TRAIN_MAX_LEN,
    _ADAPTER_CONFIG,
    _calibrate_clef,
    _checkpoint_folder,
    _clef_question,
    _fit_temperatures,
    _internal,
    _is_plain_lm,
    _laya,
    _lm_subfolder,
    _metrics,
    _option_keys,
    _parsed,
    _predicted,
    _served_lengths,
    _served_temperatures,
    _target,
    _target_for,
    is_clef_checkpoint,
)


def _decision_clef_items(rows, options: Callable, encode: Callable, validate, report, skip) -> list:
    # One item per row. `options(question)` names a question's options in the head's order and
    # `encode(state, questions)` returns the backend's tokenized fields, raising when they do not fit.
    items = []
    for index, row in enumerate(rows):
        row = row if isinstance(row, dict) else {}
        state = _parsed(row.get("state"))
        questions = _parsed(row.get("questions"))
        gold = _parsed(row["gold"] if row.get("gold") is not None else row.get("answers"))
        if state is None or not isinstance(questions, dict) or not isinstance(gold, dict):
            report["total"] += 1
            skip(index, "needs state, questions and gold")
            continue
        kept, targets, labels = {}, [], []
        for name, question in questions.items():
            report["total"] += 1
            if name not in gold:
                skip(index, "has no gold", name)
                continue
            try:
                if validate is not None:
                    validate(name, question)
                clef_question = _clef_question(question)
                keys = options(clef_question)
                target, label = _target_for(clef_question["type"], keys, _parsed(gold[name]))
            except (TypeError, ValueError, KeyError) as exc:
                skip(index, str(exc) or "is not a valid question", name)
                continue
            kept[str(name)], targets, labels = (
                clef_question,
                targets + [target],
                labels + [label],
            )
        if not kept:
            continue
        try:
            encoded = encode(state, kept)
        except (TypeError, ValueError) as exc:
            for name in kept:
                skip(index, f"does not fit: {exc}", name)
            continue
        items.append(
            {
                **encoded,
                "targets": targets,
                "labels": labels,
                "row": index,
                "source": {"state": state, "questions": kept},
            }
        )
    return items


def _decision_dataset(
    rows,
    tokenizer,
    config: dict,
    validate: Optional[Callable[[str, dict], None]] = None,
    clef: Optional[Callable] = None,
) -> tuple:
    # `clef` builds a Clef model's items: the backend's (rows, tokenizer, max_len, validate, report, skip).
    max_len = int(config.get("max_len", 512))
    head_max_len = int(config.get("head_max_len", 192))
    items, report, skips = [], {"total": 0, "skipped": 0, "reason": None, "truncated": 0}, {}

    def skip(
        index,
        reason,
        name = None,
    ):
        report["skipped"] += 1
        where = f"row {index + 1}" if name is None else f'row {index + 1}: "{name}"'
        skips.setdefault(reason, [0, f"{where} {reason}"])[0] += 1

    if clef is not None:
        items = clef(rows, tokenizer, max_len, validate, report, skip)
        rows = ()
    else:
        common = _laya().common
    for index, row in enumerate(rows):
        row = row if isinstance(row, dict) else {}
        state = _parsed(row.get("state"))
        questions = _parsed(row.get("questions"))
        gold = _parsed(row["gold"] if row.get("gold") is not None else row.get("answers"))
        if state is None or not isinstance(questions, dict) or not isinstance(gold, dict):
            report["total"] += 1
            skip(index, "needs state, questions and gold")
            continue
        for name, question in questions.items():
            report["total"] += 1
            if name not in gold:
                skip(index, "has no gold", name)
                continue
            try:
                if validate is not None:
                    validate(name, question)
                internal = _internal(question)
                target, label = _target(internal, _parsed(gold[name]))
                ids, markers = common.build_sequence(
                    tokenizer, state, internal, max_len, head_max_len
                )
            except (TypeError, ValueError) as exc:
                skip(index, str(exc), name)
                continue
            if len(markers) != len(target):
                skip(index, f"options exceed the {max_len}-token context", name)
                continue
            items.append(
                {
                    "input_ids": ids,
                    "markers": markers,
                    "qtype": common.QTYPES[internal["t"]],
                    "target": target,
                    "label": label,
                    "row": index,
                }
            )
    if skips:
        # The most common reason, shown with the first decision it applied to.
        count, example = max(skips.values(), key = lambda skipped: skipped[0])
        report["reason"] = example if count == 1 else f"{example} (and {count - 1:,} more like it)"
    # Over max_seq_length the end of the state is cut, never questions or options.
    report["truncated"] = sum(len(item["input_ids"]) >= max_len for item in items)
    if report["truncated"]:
        print(
            f"Unsloth: {report['truncated']:,} of {len(items):,} training inputs are longer than "
            f"max_seq_length = {max_len}, so the end of their state is cut. Raise "
            "max_seq_length to train on all of it."
        )
    return items, report


def _decision_holdout(
    items: list,
    seed: int = 3407,
    fraction: float = 0.1,
    max_items: int = HOLDOUT_MAX,
):
    # Counted in decisions: a Clef item holds every question of its row.
    sizes = Counter()
    for item in items:
        sizes[item["row"]] += len(item.get("labels", (None,)))
    target = min(max_items, int(sum(sizes.values()) * fraction))
    rows = sorted(sizes)
    random.Random(seed).shuffle(rows)
    held, count = set(), 0
    # Whole rows that still fit under the target; the last row always stays in training.
    for row in rows[:-1]:
        if count + sizes[row] <= target:
            held.add(row)
            count += sizes[row]
    return (
        [item for item in items if item["row"] not in held],
        [item for item in items if item["row"] in held],
    )


def _decision_evaluation(config: dict, logits, items: list) -> dict:
    return _metrics(logits, items, _served_temperatures(config, logits, items))


def _decision_calibration(
    config: dict,
    logits,
    items: list,
    clef: bool = False,
) -> dict:
    common = _laya().common
    fallback = [common.clamp_temperature(t) for t in config.get("temperature", [1.0] * 3)]
    if clef:
        return _calibrate_clef(config, logits, items)
    everything = range(len(items))
    temperature, fitted = _fit_temperatures(logits, items, everything, fallback)
    # Reported numbers score each half of the rows with temperatures fitted on the other half.
    half = {row: i % 2 for i, row in enumerate(sorted({item["row"] for item in items}))}
    per_item = [1.0] * len(items)
    for side in (0, 1):
        other = [i for i in everything if half[items[i]["row"]] != side]
        side_temperature, _ = _fit_temperatures(logits, items, other, fallback)
        for i in everything:
            if half[items[i]["row"]] == side:
                per_item[i] = side_temperature[items[i]["qtype"]]
    config["temperature"] = temperature
    buckets = {
        key: value
        for key, value in (config.pop("temperature_by_options", None) or {}).items()
        if common.QTYPES.get(key.split(":")[0]) not in fitted
    }
    if buckets:
        config["temperature_by_options"] = buckets
    return {**_metrics(logits, items, per_item), "fitted_types": sorted(fitted)}


# PEFT options the MLX adapters have no counterpart for, with the value that leaves each one off.
_DECISION_UNSUPPORTED_LORA = {
    "bias": "none",
    "layers_to_transform": None,
    "layers_pattern": None,
    "use_dora": False,
    "modules_to_save": None,
    "init_lora_weights": True,
}

# Training arguments that change what is trained and that the MLX trainer does not implement.
_DECISION_UNSUPPORTED_ARGUMENTS = ("optim_args", "auto_find_batch_size")

# The MLX trainer for language models takes these; this trainer has no counterpart for them.
_DECISION_UNUSED = {"neftune_noise_alpha", "push_to_hub"}

# Read by the MLX decision trainer without a field in the MLX config, or set by transformers itself.
_DECISION_QUIET_ARGUMENTS = (
    "do_eval",
    "do_train",
    "logging_dir",
    "run_name",
    "label_names",
    "label_smoothing_factor",
    "eval_delay",
    "logging_first_step",
    "dataloader_drop_last",
)

# What the torch DecisionTrainer takes beyond transformers' own arguments, and the MLX trainer too.
_DECISION_TRAINER_OPTIONS = (
    "brier_weight",
    "ordinal_weight",
    "kl_weight",
    "permute_fields",
    "compute_metrics",
    "preprocess_logits_for_metrics",
)

# Defaults for training a Clef, as the torch DecisionTrainer's.
CLEF_RECIPE = {}


@functools.lru_cache(maxsize = None)
def _decision_zoo():
    # The vendored Laya and Clef code scores with torch, which unsloth-zoo leaves out on Apple Silicon.
    if importlib.util.find_spec("torch") is None:
        raise ImportError(
            "Unsloth: decision models on MLX need PyTorch. Install it with `pip install torch`."
        )
    try:
        from unsloth_zoo.mlx import decision
        from unsloth_zoo.mlx.trainer import MLXDecisionTrainer
        decision.load_language_model_as_clef
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "Unsloth: training decision models on MLX needs a newer unsloth-zoo. "
            "Upgrade with `pip install -U unsloth-zoo`."
        ) from error
    return types.SimpleNamespace(MLXDecisionTrainer = MLXDecisionTrainer, **vars(decision))


def _is_clef(model) -> bool:
    return getattr(model, "is_clef", False)


def _decision_annotate(model, **attributes):
    # Plain attributes: an mlx module would otherwise hold them as part of its state.
    for name, value in attributes.items():
        object.__setattr__(model, name, value)
    model.save_pretrained_merged = types.MethodType(_decision_save_merged, model)
    model.push_to_hub_merged = types.MethodType(_decision_push_merged, model)
    if _is_clef(model):
        model.save_pretrained = types.MethodType(_clef_save, model)
        model.push_to_hub = types.MethodType(_clef_push, model)
    model.save_pretrained_gguf = types.MethodType(_decision_save_gguf, model)
    model.push_to_hub_gguf = types.MethodType(_decision_gguf().push_to_hub_gguf, model)
    return model


def _clef_save(
    self,
    save_directory,
    tokenizer = None,
    **kwargs,
) -> None:
    # Adapters plus the head, as Unsloth's other models; a full finetune has none, so it saves merged.
    if not getattr(self, "_unsloth_lora", False):
        return self.save_pretrained_merged(save_directory, tokenizer)
    config = {**self.decision_config, "fine_tuned": True}
    _decision_zoo().save_clef_adapter(
        self._unsloth_pipeline,
        save_directory,
        self._unsloth_source,
        config["base_model"],
        config.get("base_revision"),
        config,
    )


def _clef_push(
    self,
    repo_id,
    tokenizer = None,
    token = None,
    private = None,
    **kwargs,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token = token)
    repo_id = api.create_repo(repo_id, private = private, exist_ok = True).repo_id
    with tempfile.TemporaryDirectory() as folder:
        self.save_pretrained(folder, tokenizer)
        api.upload_folder(folder_path = folder, repo_id = repo_id)
    print(f"Unsloth: Saved the decision model to https://huggingface.co/{repo_id}")


@functools.lru_cache(maxsize = None)
def _decision_gguf():
    # By path: the exporter runs llama.cpp's converter in a subprocess, and unsloth.models needs torch.
    spec = importlib.util.spec_from_file_location(
        "unsloth._decision_gguf", Path(__file__).with_name("decision_gguf.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decision_save_gguf(
    self,
    save_directory,
    tokenizer = None,
    quantization_method = "q8_0",
    source_folder = None,
    print_output = False,
    **kwargs,
) -> dict:
    """GGUF for llama.cpp's decision server in <save_directory>/gguf, from a temporary merged save."""
    gguf = _decision_gguf()
    # Fail before the merge when llama.cpp cannot convert.
    gguf._converter_dir(print_output)
    output = Path(save_directory)
    output.mkdir(parents = True, exist_ok = True)
    gguf._remove_abandoned_temp(output)
    prefix = gguf._temp_prefix(".unsloth-merged-")
    with (
        gguf._exit_on_sigterm(),
        tempfile.TemporaryDirectory(prefix = prefix, dir = output) as merged,
    ):
        self.save_pretrained_merged(merged, tokenizer)
        return gguf.export_decision_gguf(
            merged,
            quantization_method,
            output_dir = output / gguf._contract().EXPORT_DIR,
            source_folder = source_folder,
            print_output = print_output,
        )


def _decision_pad_token_id(tokenizer) -> int:
    return getattr(tokenizer, "tokenizer", tokenizer).pad_token_id


def _clef_items(pipeline, rows, tokenizer, max_len, validate, report, skip) -> list:
    zoo = _decision_zoo()
    return _decision_clef_items(
        rows,
        functools.partial(zoo.clef_option_keys, pipeline),
        lambda state, questions: zoo.clef_training_item(pipeline, state, questions, max_len),
        validate,
        report,
        skip,
    )


def _decision_logits(model, tokenizer, items: list) -> tuple:
    import torch

    # Torch rows, so both backends score and calibrate with the same code.
    if not _is_clef(model):
        rows = _decision_zoo().decision_logits(model, items, _decision_pad_token_id(tokenizer))
        return [torch.from_numpy(row) for row in rows], items
    # One Laya-shaped item per question for the shared metrics.
    logits, questions = [], []
    for item, rows in zip(items, _decision_zoo().clef_logits(model, items)):
        kinds = [question["type"] for question in item["source"]["questions"].values()]
        for row, target, label, kind in zip(rows, item["targets"], item["labels"], kinds):
            logits.append(torch.from_numpy(row))
            questions.append(
                {
                    "target": target,
                    "label": label,
                    "qtype": QUESTION_TYPES.index(kind),
                    "row": item["row"],
                }
            )
    return logits, questions


def _decision_save_merged(
    self,
    save_directory,
    tokenizer = None,
    save_method = "merged_16bit",
    **kwargs,
) -> None:
    if save_method != "merged_16bit":
        raise NotImplementedError(
            f"Unsloth: decision models are saved merged in 16-bit, not as {save_method!r}."
        )
    # The tokenizer is the checkpoint's own, which the savers copy from the source.
    config = {**self.decision_config, "fine_tuned": True}
    # The Clef saver writes the weights that train, so a frozen backbone is saved as trained before it was frozen.
    frozen = getattr(self, "_unsloth_frozen_backbone", ())
    FastDecisionModel.unfreeze_backbone(self)
    try:
        if _is_clef(self):
            _decision_zoo().save_clef_model(
                self._unsloth_pipeline, save_directory, self._unsloth_source, config
            )
        else:
            _decision_zoo().save_decision_model(self, save_directory, self._unsloth_source, config)
    finally:
        if frozen:
            FastDecisionModel.freeze_backbone(self)


def _decision_push_merged(
    self,
    repo_id,
    tokenizer = None,
    save_method = "merged_16bit",
    token = None,
    private = None,
    **kwargs,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token = token)
    repo_id = api.create_repo(repo_id, private = private, exist_ok = True).repo_id
    with tempfile.TemporaryDirectory() as folder:
        self.save_pretrained_merged(folder, tokenizer, save_method)
        api.upload_folder(folder_path = folder, repo_id = repo_id)
    print(f"Unsloth: Saved the decision model to https://huggingface.co/{repo_id}")


_CLEF_ATTRIBUTES = (
    "decision_config",
    "is_clef",
    "_unsloth_pipeline",
    "_unsloth_source",
    "_unsloth_full_finetuning",
    "_saved_temp_tokenizer",
)


def _clef_network(pipeline, folder, config, full_finetuning, gradient_checkpointing):
    zoo = _decision_zoo()
    # A checkpoint saved as adapters comes with them, and they go on training.
    adapters = hasattr(pipeline, "base_folder")
    if full_finetuning or adapters:
        network = zoo.clef_training_network(
            pipeline,
            full_finetuning = bool(full_finetuning),
            gradient_checkpointing = gradient_checkpointing,
        )
    else:
        # Until get_peft_model adds adapters, only the joint head trains.
        pipeline.model.freeze()
        pipeline.head.unfreeze()
        network = zoo.ClefNetwork(pipeline, gradient_checkpointing)
        network.train()
    _decision_annotate(
        network,
        decision_config = config,
        is_clef = True,
        _unsloth_pipeline = pipeline,
        _unsloth_source = getattr(pipeline, "base_folder", folder),
        _unsloth_full_finetuning = bool(full_finetuning),
        _saved_temp_tokenizer = pipeline.tokenizer,
        _unsloth_lora = adapters,
    )
    return network, pipeline.tokenizer


def _load_clef(
    folder, max_seq_length, load_in_4bit, full_finetuning, token, gradient_checkpointing, name
):
    # A 4-bit decoder trains through LoRA adapters only.
    pipeline = _decision_zoo().load_decision_model(
        folder, token = token, load_in_4bit = bool(load_in_4bit) and not full_finetuning
    )
    max_len = int(max_seq_length or CLEF_MAX_LEN)
    config = {"layout": "clef", "max_len": max_len, "temperature": [1.0] * 3}
    saved = folder / "unsloth_decision_config.json"
    if saved.is_file():
        config.update(json.loads(saved.read_text(encoding = "utf-8")), max_len = max_len)
        # The parent run's training record does not describe the next fine-tune, as for Laya.
        config.pop("training", None)
    if hasattr(pipeline, "base_folder"):
        # Adapters stay over the base they were trained on.
        adapter = json.loads((folder / _ADAPTER_CONFIG).read_text(encoding = "utf-8"))
        config.setdefault("base_model", adapter["base_model_name_or_path"])
        if adapter.get("revision"):
            config.setdefault("base_revision", adapter["revision"])
    else:
        # A merged checkpoint is its own base, whatever model it once started from.
        config.pop("base_revision", None)
        config.update(name)
    return _clef_network(pipeline, folder, config, full_finetuning, gradient_checkpointing)


def _load_lm_as_clef(
    model_name,
    max_seq_length,
    load_in_4bit,
    full_finetuning,
    token,
    revision,
    local_files_only,
    gradient_checkpointing,
    random_state,
    decision_head = "clef",
    head_width = None,
    head_config = None,
    **kwargs,
):
    # A plain language model plus a new joint schema head, as unsloth/models/decision_from_lm.py builds one.
    if decision_head != "clef":
        raise ValueError(f"Unsloth: decision_head must be one of ('clef',), not {decision_head!r}.")
    if kwargs:
        raise NotImplementedError(
            f"Unsloth: decision models on MLX do not support {', '.join(sorted(kwargs))}."
        )
    folder = Path(str(model_name)).expanduser()
    if not folder.is_dir():
        from huggingface_hub import snapshot_download
        folder = Path(
            snapshot_download(
                str(model_name),
                token = token,
                revision = revision,
                local_files_only = local_files_only,
            )
        )
    if not (folder / "config.json").is_file():
        raise NotImplementedError(
            f"Unsloth: {folder} holds LoRA adapters over a base model, which MLX does not load. "
            "Merge them into the base model first."
        )
    load_in_4bit = bool(load_in_4bit) and not full_finetuning
    pipeline = _decision_zoo().load_language_model_as_clef(
        folder,
        head_width = head_width,
        head_config = head_config,
        seed = random_state,
        token = token,
        load_in_4bit = load_in_4bit,
    )
    config = {
        "layout": "clef",
        "max_len": int(max_seq_length or CLEF_MAX_LEN),
        "temperature": [1.0] * 3,
        "base_model": str(model_name),
        **({"base_revision": revision} if revision else {}),
        "load_in_4bit": load_in_4bit,
    }
    return _clef_network(pipeline, folder, config, full_finetuning, gradient_checkpointing)


class FastDecisionModel:
    @staticmethod
    def from_pretrained(
        model_name: str,
        subfolder: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        dtype = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        full_finetuning: bool = False,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        use_gradient_checkpointing = "unsloth",
        random_state: int = 3407,
        **kwargs,
    ):
        if load_in_8bit:
            raise NotImplementedError("Unsloth: decision models do not support load_in_8bit.")
        checkpointing = bool(use_gradient_checkpointing)
        if kwargs.get("decision_head") is None and _is_plain_lm(
            model_name, subfolder, token, revision, local_files_only
        ):
            kwargs["decision_head"] = "clef"
        lm = kwargs.get("decision_head") is not None
        if not lm:
            folder = _checkpoint_folder(model_name, subfolder, token, revision, local_files_only)
        if (lm or is_clef_checkpoint(folder)) and dtype is not None:
            raise NotImplementedError(
                f"Unsloth: Clef on MLX trains at its checkpoint's precision, not {dtype}."
            )
        if lm:
            if subfolder:
                model_name = _lm_subfolder(model_name, subfolder, token, revision, local_files_only)
            return _load_lm_as_clef(
                model_name,
                max_seq_length,
                load_in_4bit,
                full_finetuning,
                token,
                revision,
                local_files_only,
                checkpointing,
                random_state,
                **kwargs,
            )
        if is_clef_checkpoint(folder):
            name = {
                "base_model": str(model_name),
                **({"base_revision": revision} if revision else {}),
            }
            return _load_clef(
                folder,
                max_seq_length,
                load_in_4bit,
                full_finetuning,
                token,
                checkpointing,
                name,
            )
        if load_in_4bit:
            raise NotImplementedError(
                "Unsloth: Laya decision models train in 16-bit, so load_in_4bit is not supported."
            )
        if not (full_finetuning or dtype is None or str(dtype).rsplit(".", 1)[-1] == "float16"):
            raise NotImplementedError(
                f"Unsloth: decision models on MLX keep the frozen encoder in float16, not {dtype}."
            )
        from transformers import AutoTokenizer

        config = json.loads((folder / "rl_agent_config.json").read_text(encoding = "utf-8"))
        # The base model's training record does not describe the fine-tune.
        config.pop("training", None)
        encoder = json.loads((folder / "encoder" / "config.json").read_text(encoding = "utf-8"))
        config["max_len"], config["head_max_len"] = _served_lengths(
            config, int(encoder.get("max_position_embeddings", TRAIN_MAX_LEN)), max_seq_length
        )
        tokenizer = AutoTokenizer.from_pretrained(str(folder / "tokenizer"))
        model = _decision_zoo().load_trainable_decision_model(
            folder, full_finetuning, gradient_checkpointing = checkpointing
        )
        _decision_annotate(
            model,
            decision_config = config,
            _unsloth_source = folder,
            _unsloth_full_finetuning = bool(full_finetuning),
            _saved_temp_tokenizer = tokenizer,
        )
        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r = 64,
        target_modules = "all-linear",
        lora_alpha = 64,
        lora_dropout = 0.0,
        bias = "none",
        layers_to_transform = None,
        layers_pattern = None,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        max_seq_length = None,
        use_rslora = False,
        use_dora = False,
        modules_to_save = None,
        init_lora_weights = True,
        loftq_config = {},
        **kwargs,
    ):
        if getattr(model, "_unsloth_full_finetuning", False):
            print("Unsloth: Full finetuning is enabled, so .get_peft_model has no effect")
            return model
        given = {**locals(), **kwargs}
        unsupported = [
            name for name, off in _DECISION_UNSUPPORTED_LORA.items() if given[name] != off
        ]
        unsupported += ["loftq_config"] if loftq_config else []
        unsupported += sorted(kwargs)
        if unsupported:
            raise NotImplementedError(
                f"Unsloth: decision models on MLX do not support {', '.join(unsupported)}."
            )
        if _is_clef(model):
            if getattr(model, "_unsloth_lora", False):
                raise RuntimeError("Unsloth: You already added LoRA adapters to your model!")
            network = _decision_zoo().clef_training_network(
                model._unsloth_pipeline,
                r = r,
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                use_rslora = use_rslora,
                target_modules = target_modules,
                random_state = random_state,
                gradient_checkpointing = bool(use_gradient_checkpointing),
            )
            attributes = {name: getattr(model, name) for name in _CLEF_ATTRIBUTES}
            return _decision_annotate(network, **attributes, _unsloth_lora = True)
        _decision_zoo().add_lora_adapters(
            model,
            r = r,
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout,
            use_rslora = use_rslora,
            target_modules = target_modules,
            random_state = random_state,
        )
        model.gradient_checkpointing = bool(use_gradient_checkpointing)
        return model

    @staticmethod
    def freeze_backbone(model):
        # Head-only warm-up for a fresh decision head; undo with unfreeze_backbone.
        from mlx.utils import tree_flatten

        frozen = [name for name, _ in tree_flatten(model.encoder.trainable_parameters())]
        model.encoder.freeze()
        object.__setattr__(model, "_unsloth_frozen_backbone", frozen)
        return model

    @staticmethod
    def unfreeze_backbone(model):
        names = set(getattr(model, "_unsloth_frozen_backbone", ()))
        for path, module in model.encoder.named_modules():
            keys = [key for key in module if f"{path}.{key}".lstrip(".") in names]
            if keys:
                module.unfreeze(recurse = False, keys = keys)
        object.__setattr__(model, "_unsloth_frozen_backbone", ())
        return model

    @staticmethod
    def for_inference(model):
        model.eval()
        return model

    @staticmethod
    def for_training(model, use_gradient_checkpointing = True):
        model.train()
        if not _is_clef(model):
            model.gradient_checkpointing = bool(use_gradient_checkpointing)
        return model

    @staticmethod
    def build_dataset(
        rows,
        tokenizer,
        model,
        validate: Optional[Callable[[str, dict], None]] = None,
    ) -> tuple:
        clef = functools.partial(_clef_items, model._unsloth_pipeline) if _is_clef(model) else None
        return _decision_dataset(rows, tokenizer, model.decision_config, validate, clef)

    split_holdout = staticmethod(_decision_holdout)

    @staticmethod
    def predict(model, tokenizer, state, questions: dict) -> dict:
        """Answers one Decision API request: {name: answer} at the calibrated temperatures.
        Each answer is the Decision API's (choice / confidence, score / legend, or noul) plus
        "answer" (the option, True / False for noul, the level number for score) and
        "probabilities" over every option."""
        import torch

        from unsloth._vendor.clef.joint_schema_model import systemone_answer

        if not isinstance(questions, dict) or not questions:
            raise DecisionDataError("questions must be a non-empty dict of name to question")
        state, config, zoo = _parsed(state), model.decision_config, _decision_zoo()
        if _is_clef(model):
            pipeline = model._unsloth_pipeline
            questions = {str(name): _clef_question(q) for name, q in questions.items()}
            # Read up to CLEF_SERVE_MAX_LEN tokens, like serving, even past the training cut.
            max_length = max(int(config.get("max_len", CLEF_MAX_LEN)), CLEF_SERVE_MAX_LEN)
            item = zoo.clef_training_item(pipeline, state, questions, max_length)
            logits = zoo.clef_logits(model, [item])[0]
            keys = [zoo.clef_option_keys(pipeline, q) for q in questions.values()]
            kinds = [QUESTION_TYPES.index(q["type"]) for q in questions.values()]
        else:
            common, max_len = _laya().common, int(config.get("max_len", 512))
            tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
            items, keys = [], []
            for name, question in questions.items():
                internal = _internal(question)
                ids, markers = common.build_sequence(
                    tokenizer, state, internal, max_len, int(config.get("head_max_len", 192))
                )
                keys.append(_option_keys(internal))
                if len(markers) != len(keys[-1]):
                    raise DecisionDataError(
                        f'"{name}" has more options than fit in {max_len} tokens'
                    )
                items.append(
                    {
                        "input_ids": ids,
                        "markers": markers,
                        "qtype": common.QTYPES[internal["t"]],
                        "target": [0.0] * len(markers),
                    }
                )
            logits = zoo.decision_logits(model, items, tokenizer.pad_token_id)
            kinds = [item["qtype"] for item in items]
        logits = [torch.from_numpy(row) for row in logits]
        scales = _served_temperatures(config, logits, [{"qtype": kind} for kind in kinds])
        answers = {}
        for (name, question), row, scale, options in zip(questions.items(), logits, scales, keys):
            probabilities = dict(zip(options, torch.softmax(row / scale, -1).tolist()))
            answers[name] = _predicted(
                question, systemone_answer(question, probabilities), probabilities
            )
        return answers

    # batch_size is the torch call shape; MLX scores Clef a record at a time and Laya in its own batches.
    @staticmethod
    def evaluate(
        model,
        tokenizer,
        items: list,
        batch_size = None,
    ) -> dict:
        logits, items = _decision_logits(model, tokenizer, items)
        return _decision_evaluation(model.decision_config, logits, items)

    @staticmethod
    def calibrate(
        model,
        tokenizer,
        items: list,
        batch_size = None,
    ) -> dict:
        logits, items = _decision_logits(model, tokenizer, items)
        return _decision_calibration(model.decision_config, logits, items, _is_clef(model))


def _decision_arguments(args):
    from unsloth import _MLX_ALLOWED_EXTRA_ARGUMENTS, _mlx_training_argument_values
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    if args is None:
        # The torch trainer's default: three epochs, no evaluation schedule.
        from transformers import TrainingArguments
        args = TrainingArguments(output_dir = "tmp_trainer")
    given, checkpointing = args, None
    if isinstance(args, MLXTrainingConfig):
        args = copy.copy(args)
    else:
        # transformers.TrainingArguments, as the torch DecisionTrainer takes.
        checkpointing = bool(getattr(args, "gradient_checkpointing", False)) or None
        fields = {field.name for field in dataclasses.fields(MLXTrainingConfig)}
        values = _mlx_training_argument_values(args)
        args = MLXTrainingConfig(**{k: v for k, v in values.items() if k in fields})
        if dataclasses.is_dataclass(given):
            # Whatever else differs from the defaults has no MLX counterpart.
            default, quiet = (
                type(given)(output_dir = given.output_dir),
                {*fields, *_DECISION_QUIET_ARGUMENTS, *_DECISION_UNSUPPORTED_ARGUMENTS},
            )
            ignored = [
                field.name
                for field in dataclasses.fields(given)
                if field.name not in (quiet | _MLX_ALLOWED_EXTRA_ARGUMENTS) - _DECISION_UNUSED
                and getattr(given, field.name) != getattr(default, field.name)
            ]
            if ignored:
                warnings.warn(
                    f"Unsloth: DecisionTrainer on MLX ignores {', '.join(sorted(ignored))}."
                )
    # The MLX trainer and transformers callbacks read arguments the MLX config has no field for.
    for name, value in vars(given).items():
        if not (name.startswith("_") or hasattr(args, name)):
            setattr(args, name, value)
    unsupported = [name for name in _DECISION_UNSUPPORTED_ARGUMENTS if getattr(given, name, None)]
    saving = getattr(given, "save_strategy", None)
    if str(getattr(saving, "value", saving)).lower() == "best":
        unsupported.append('save_strategy = "best"')
    if unsupported:
        raise NotImplementedError(
            f"Unsloth: DecisionTrainer on MLX does not support {', '.join(unsupported)}."
        )
    return args, checkpointing


class DecisionTrainer:
    """The MLX decision trainer behind the torch DecisionTrainer's arguments.

    Everything else, such as `train(resume_from_checkpoint = ...)`, `evaluate` and `predict`,
    is the MLX trainer's own.
    """

    def __init__(
        self,
        model = None,
        args = None,
        train_dataset = None,
        eval_dataset = None,
        *,
        head_learning_rate: Optional[float] = None,
        tokenizer = None,
        label_smoothing: Optional[float] = None,
        callbacks = None,
        processing_class = None,
        **kwargs,
    ):
        unknown = sorted(set(kwargs) - set(_DECISION_TRAINER_OPTIONS))
        if unknown:
            raise NotImplementedError(
                f"Unsloth: DecisionTrainer on MLX does not support {', '.join(unknown)}."
            )
        args, gradient_checkpointing = _decision_arguments(args)
        clef = _is_clef(model)
        recipe = CLEF_RECIPE if clef else {}
        if label_smoothing is None:
            label_smoothing = getattr(args, "label_smoothing_factor", 0.0) or recipe.get(
                "label_smoothing", 0.0
            )
        options = {name: recipe[name] for name in _DECISION_TRAINER_OPTIONS if name in recipe}
        options.update({name: value for name, value in kwargs.items() if value is not None})
        preprocess = options.get("preprocess_logits_for_metrics")
        if preprocess is not None:
            import torch

            # The hook is written for the torch trainer, which hands it tensors.
            options["preprocess_logits_for_metrics"] = lambda logits, labels: preprocess(
                torch.from_numpy(logits), torch.from_numpy(labels)
            )
        if gradient_checkpointing and not clef:
            model.gradient_checkpointing = True
        if clef and getattr(model, "_unsloth_lora", False):
            # Lets a checkpoint be written as the adapters save_pretrained writes.
            config = model.decision_config
            origin = (
                model._unsloth_source,
                config["base_model"],
                config.get("base_revision"),
                config,
            )
            object.__setattr__(model, "_origin", origin)
        if clef and options.get("permute_fields") and train_dataset is not None:
            # A record is encoded again in its new order, to the length it was built for.
            length = model.decision_config["max_len"]
            train_dataset = [
                {**item, "source": {**item["source"], "max_length": length}}
                if "source" in item
                else item
                for item in train_dataset
            ]
        processing_class = tokenizer if processing_class is None else processing_class
        if processing_class is None:
            processing_class = model._saved_temp_tokenizer
        self._trainer = _decision_zoo().MLXDecisionTrainer(
            model,
            args,
            train_dataset,
            eval_dataset,
            pad_token_id = _decision_pad_token_id(processing_class),
            head_learning_rate = head_learning_rate,
            callbacks = callbacks,
            processing_class = processing_class,
            label_smoothing = float(label_smoothing or 0.0),
            **options,
        )

    def __getattr__(self, name):
        if name == "_trainer":
            raise AttributeError(name)
        return getattr(self._trainer, name)
