# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

__all__ = [
    "FastDecisionModel",
    "DecisionTrainer",
]

import copy
import functools
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers import Trainer, TrainingArguments
from transformers.training_args import ParallelMode

from ._utils import (
    _mark_full_finetuning,
    apply_unsloth_gradient_checkpointing,
    is_bfloat16_supported,
)
from .loader_utils import is_distributed

TRAIN_MAX_LEN, TRAIN_HEAD_MAX_LEN = 1024, 256
HOLDOUT_MAX = 400
MIN_CALIBRATION_ITEMS = 10
HEAD_LEARNING_RATE = 1e-4
QUESTION_TYPES = ("choice", "score", "noul")
_FILES = ("rl_agent_config.json", "model.safetensors")
_DIRS = ("encoder", "tokenizer")
_CLEF_HEAD_FILES = ("joint_head.safetensors", "joint_head_config.json")
# Copied next to a saved Clef fine-tune so Cloudflare's own loader and chat template keep working.
_CLEF_EXTRA_FILES = ("joint_schema_model.py", "chat_template.jinja", "LICENSE")
CLEF_MAX_LEN = 4096
# laya 0.3.5 ships inside Unsloth for Studio's Decision API (studio/backend/vendor/README.md).
_VENDORED_LAYA = (
    Path(__file__).resolve().parents[2] / "studio" / "backend" / "vendor" / "laya" / "__init__.py"
).resolve()


class DecisionDataError(ValueError):
    pass


@functools.lru_cache(maxsize = None)
def _laya():
    # By path, so a pip installed "laya" is never used or replaced; Studio registers this copy.
    for name in ("laya", "unsloth._laya"):
        module = sys.modules.get(name)
        if getattr(module, "__file__", None) and Path(module.__file__).resolve() == _VENDORED_LAYA:
            return module
    if not _VENDORED_LAYA.is_file():
        raise ImportError(
            f"Unsloth: decision models need laya, which ships with Unsloth at {_VENDORED_LAYA.parent}. "
            "Please reinstall Unsloth."
        )
    spec = importlib.util.spec_from_file_location(
        "unsloth._laya", _VENDORED_LAYA, submodule_search_locations = [str(_VENDORED_LAYA.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # laya's JSON reads and writes use a bare open(), which is the locale encoding on Windows.
    module.agent.open = functools.partial(open, encoding = "utf-8")
    return module


def is_decision_checkpoint(folder) -> bool:
    folder = Path(folder)
    return is_clef_checkpoint(folder) or (
        all((folder / name).is_file() for name in _FILES)
        and all((folder / name).is_dir() for name in _DIRS)
    )


def is_clef_checkpoint(folder) -> bool:
    # Cloudflare's Clef layout: a Qwen3.5 backbone plus a joint schema head.
    folder = Path(folder)
    return all((folder / name).is_file() for name in ("config.json", *_CLEF_HEAD_FILES))


def _checkpoint_folder(model_name, subfolder, token, revision, local_files_only) -> Path:
    root = Path(model_name).expanduser()
    if not root.is_dir():
        try:
            from unsloth_zoo.hf_xet_fallback import (
                snapshot_download_with_xet_fallback as snapshot_download,
            )
        except ImportError:
            from huggingface_hub import snapshot_download
        prefix = f"{subfolder}/" if subfolder else ""
        download = functools.partial(
            snapshot_download,
            model_name,
            token = token,
            revision = revision,
            local_files_only = local_files_only,
        )
        # Laya repos hold several checkpoints in subfolders, so only the asked one is fetched.
        root = Path(
            download(
                allow_patterns = [prefix + name for name in (*_FILES, _CLEF_HEAD_FILES[1])]
                + [f"{prefix}{name}/*" for name in _DIRS]
            )
        )
        if (root / prefix / _CLEF_HEAD_FILES[1]).is_file():
            root = Path(download(allow_patterns = [prefix + "*"]))
    folder = root / subfolder if subfolder else root
    if not is_decision_checkpoint(folder):
        raise ValueError(
            f"Unsloth: {folder} is not a decision model checkpoint "
            "(rl_agent_config.json, model.safetensors, encoder/ and tokenizer/, "
            "or a Clef backbone with joint_head.safetensors and joint_head_config.json)."
        )
    return folder


def _encoder_config(folder):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(folder))
    # transformers 4 ignores transformers 5's rope_parameters; copy the trained thetas over.
    rope = getattr(config, "rope_parameters", None)
    if isinstance(rope, dict):
        for layer_type, name in (
            ("full_attention", "global_rope_theta"),
            ("sliding_attention", "local_rope_theta"),
        ):
            theta = (rope.get(layer_type) or {}).get("rope_theta")
            if theta is not None and hasattr(config, name):
                setattr(config, name, float(theta))
    return config


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _amp_dtype(device):
    if device.type == "cuda":
        return torch.bfloat16 if is_bfloat16_supported() else torch.float16
    return torch.bfloat16 if device.type == "xpu" else None


def _gradient_checkpointing(model, use_gradient_checkpointing) -> None:
    device = next(model.parameters()).device
    # Unsloth's offloaded checkpointing needs an accelerator, and its re-entrant backward breaks DDP (#3713).
    if use_gradient_checkpointing == "unsloth" and (device.type == "cpu" or is_distributed()):
        use_gradient_checkpointing = True
    mode = apply_unsloth_gradient_checkpointing(
        use_gradient_checkpointing, model.decision_config["max_len"], _amp_dtype(device)
    )
    if mode == "unsloth" or mode is True:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs = {"use_reentrant": mode == "unsloth"}
        )
        if mode == "unsloth":
            model.encoder.enable_input_require_grads()
    else:
        model.encoder.gradient_checkpointing_disable()


def _parsed(value):
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _internal(question) -> dict:
    if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
        raise DecisionDataError("is not a valid question")
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice" and not (
        isinstance(criteria, (dict, list))
        and criteria
        and all(isinstance(option, str) for option in criteria)
    ):
        raise DecisionDataError("needs criteria naming its options")
    if kind == "score" and not (isinstance(criteria, list) and criteria):
        raise DecisionDataError("needs a list of criteria levels")
    if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
        raise DecisionDataError('criteria may only have "true" and "false"')
    laya_question = {"type": kind, "instructions": question.get("instructions") or ""}
    if criteria is not None:
        laya_question["criteria"] = criteria
    return _laya().agent.Agent._to_internal(laya_question)


def _option_keys(internal: dict) -> list:
    if internal["t"] == "choice":
        return [str(key) for key in internal["crit"]]
    if internal["t"] == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(internal["crit"]))]


def _label(kind: str, label):
    if kind == "score" and isinstance(label, (int, float, str)) and not isinstance(label, bool):
        try:
            level = float(label)
        except ValueError:
            return None
        return str(int(level)) if level.is_integer() else None
    if label is None:
        return None
    return str(label).strip().lower() if kind == "noul" else str(label)


def _target(internal: dict, gold) -> tuple:
    return _target_for(internal["t"], _option_keys(internal), gold)


def _target_for(kind: str, keys: list, gold) -> tuple:
    if not isinstance(gold, dict):
        gold = {"label": gold}
    label = _label(kind, gold.get("label"))
    probabilities = gold.get("probabilities")
    if kind == "noul" and not isinstance(probabilities, dict):
        noul = gold.get("noul")
        probabilities = {"true": noul} if isinstance(noul, (int, float)) else None
    if isinstance(probabilities, dict):
        try:
            values = {str(key): float(value) for key, value in probabilities.items()}
        except (TypeError, ValueError):
            values = {}
        if not all(math.isfinite(value) for value in values.values()):
            raise DecisionDataError("gold has probabilities that are not finite numbers")
        if kind == "noul" and len(values.keys() & {"false", "true"}) == 1:
            known = "true" if "true" in values else "false"
            values[{"true": "false", "false": "true"}[known]] = 1.0 - values[known]
        target = [max(0.0, values.get(key, 0.0)) for key in keys]
        total = sum(target)
        if 0 < total < math.inf:
            target = [value / total for value in target]
            return target, keys.index(label) if label in keys else target.index(max(target))
    if label in keys:
        return [1.0 if key == label else 0.0 for key in keys], keys.index(label)
    raise DecisionDataError("gold has no usable label or probabilities")


def _soft_cross_entropy(
    logits,
    target,
    mask,
    label_smoothing = 0.0,
    brier_weight = 0.0,
):
    logits = logits.float().masked_fill(~mask, -1e4)
    log_p = torch.log_softmax(logits, -1)
    smoothed = target
    if label_smoothing:
        uniform = mask.float() / mask.sum(-1, keepdim = True).clamp(min = 1)
        smoothed = (1.0 - label_smoothing) * target + label_smoothing * uniform
    loss = -(smoothed * log_p).sum(-1).mean()
    if brier_weight:
        # Cloudflare trains Clef with label smoothed cross entropy plus a Brier term for calibration.
        loss = loss + brier_weight * ((log_p.exp() - target) ** 2 * mask).sum(-1).mean()
    return loss


class DecisionDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, items: list) -> dict:
        rows, length = len(items), max(len(item["input_ids"]) for item in items)
        options = max(len(item["markers"]) for item in items)
        batch = {
            "input_ids": torch.full((rows, length), self.pad_token_id, dtype = torch.long),
            "attention_mask": torch.zeros((rows, length), dtype = torch.long),
            "marker_pos": torch.zeros((rows, options), dtype = torch.long),
            "marker_mask": torch.zeros((rows, options), dtype = torch.bool),
            "qtype": torch.tensor([item["qtype"] for item in items]),
            "target": torch.zeros((rows, options), dtype = torch.float32),
        }
        for i, item in enumerate(items):
            ids, markers = item["input_ids"], item["markers"]
            batch["input_ids"][i, : len(ids)] = torch.tensor(ids)
            batch["attention_mask"][i, : len(ids)] = 1
            batch["marker_pos"][i, : len(markers)] = torch.tensor(markers)
            batch["marker_mask"][i, : len(markers)] = True
            batch["target"][i, : len(item["target"])] = torch.tensor(item["target"])
        return batch


class ClefDataCollator:
    # One row per record: every question of the record is scored jointly, as Clef serves them.
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, items: list) -> dict:
        rows, length = len(items), max(len(item["input_ids"]) for item in items)
        targets = [target for item in items for target in item["targets"]]
        options = max(len(target) for target in targets)
        batch = {
            "input_ids": torch.full((rows, length), self.pad_token_id, dtype = torch.long),
            "attention_mask": torch.zeros((rows, length), dtype = torch.long),
            "records": [item["record"] for item in items],
            "marker_mask": torch.zeros((len(targets), options), dtype = torch.bool),
            "target": torch.zeros((len(targets), options), dtype = torch.float32),
        }
        for i, item in enumerate(items):
            batch["input_ids"][i, : len(item["input_ids"])] = torch.tensor(item["input_ids"])
            batch["attention_mask"][i, : len(item["input_ids"])] = 1
        for i, target in enumerate(targets):
            batch["marker_mask"][i, : len(target)] = True
            batch["target"][i, : len(target)] = torch.tensor(target)
        return batch


class ClefDecisionModel(torch.nn.Module):
    # `encoder` is the Qwen3.5 backbone, named like Laya's so the trainer treats both alike.
    is_clef = True

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def _backbone(self):
        return (
            self.encoder.get_base_model()
            if hasattr(self.encoder, "get_base_model")
            else self.encoder
        )

    def forward(self, input_ids, attention_mask, records, **kwargs):
        backbone = self._backbone()
        text_model = backbone.model
        text_model = getattr(text_model, "language_model", text_model)
        hidden = text_model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            use_cache = False,
            return_dict = True,
        ).last_hidden_state
        logits = self.head(
            hidden,
            input_ids,
            attention_mask,
            records,
            backbone.get_output_embeddings().weight.detach(),
        )
        flat = [question for record in logits for question in record]
        padded = torch.full(
            (len(flat), max(len(z) for z in flat)), -1e4, dtype = flat[0].dtype, device = flat[0].device
        )
        for i, z in enumerate(flat):
            padded[i, : len(z)] = z
        return padded, None


def _clef_question(question) -> dict:
    if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
        raise DecisionDataError("is not a valid question")
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list) and criteria and all(isinstance(c, str) for c in criteria):
            criteria = dict.fromkeys(criteria)
        if not (isinstance(criteria, dict) and criteria):
            raise DecisionDataError("needs criteria naming its options")
        if len({str(key) for key in criteria}) != len(criteria):
            raise DecisionDataError("has repeated options")
    if kind == "score" and not (isinstance(criteria, list) and criteria):
        raise DecisionDataError("needs a list of criteria levels")
    if kind == "noul" and criteria is not None:
        if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
            raise DecisionDataError('criteria may only have "true" and "false"')
    clef_question = {"type": kind, "instructions": question.get("instructions")}
    if criteria is not None:
        clef_question["criteria"] = criteria
    return clef_question


def _clef_items(rows, tokenizer, max_len, validate, report, skip) -> list:
    from .clef import QUESTION_TYPES as CLEF_TYPES, encode_record, question_options

    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
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
                keys = [key for key, _ in question_options(clef_question)]
                target, label = _target_for(clef_question["type"], keys, _parsed(gold[name]))
            except (TypeError, ValueError, KeyError) as exc:
                skip(index, str(exc) or "is not a valid question", name)
                continue
            kept[str(name)], targets, labels = clef_question, targets + [target], labels + [label]
        if not kept:
            continue
        try:
            record = encode_record(
                tokenizer, {"state": state, "questions": kept}, max_length = max_len
            )
        except (TypeError, ValueError) as exc:
            for name in kept:
                skip(index, f"does not fit: {exc}", name)
            continue
        items.append(
            {
                "input_ids": list(record.input_ids),
                "record": record,
                "qtypes": [CLEF_TYPES[q["type"]] for q in kept.values()],
                "targets": targets,
                "labels": labels,
                "row": index,
            }
        )
    return items


@torch.no_grad()
def _clef_logits(
    model,
    items: list,
    pad_token_id: int,
    batch_size: int = 4,
) -> tuple:
    # Per question logits plus one Laya-shaped item per question for the shared metrics.
    from .clef import QUESTION_TYPES as CLEF_TYPES

    device = next(model.parameters()).device
    # Never fp16 autocast: the gated delta net overflows in pure fp16, so off bf16 GPUs the
    # backbone runs in the dtype Unsloth loaded it with.
    amp_dtype = _amp_dtype(device)
    amp_dtype = amp_dtype if amp_dtype == torch.bfloat16 else None
    collate = ClefDataCollator(pad_token_id)
    # Clef numbers question types noul, choice, score; Laya's metrics use choice, score, noul.
    laya_type = {CLEF_TYPES[kind]: QUESTION_TYPES.index(kind) for kind in QUESTION_TYPES}
    was_training = model.training
    model.eval()
    logits, questions = [], []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        batch = collate(chunk)
        batch.pop("target")
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast(device.type, dtype = amp_dtype, enabled = amp_dtype is not None):
            z, _ = model(**inputs)
        z = z.float().cpu()
        flat = [
            (target, label, qtype, item["row"])
            for item in chunk
            for target, label, qtype in zip(item["targets"], item["labels"], item["qtypes"])
        ]
        for row, (target, label, qtype, record_row) in enumerate(flat):
            logits.append(z[row, : len(target)])
            questions.append(
                {"target": target, "label": label, "qtype": laya_type[qtype], "row": record_row}
            )
    model.train(was_training)
    return logits, questions


def _decision_logits(model, tokenizer, items: list) -> tuple:
    pad_token_id = getattr(tokenizer, "tokenizer", tokenizer).pad_token_id
    if getattr(model, "is_clef", False):
        return _clef_logits(model, items, pad_token_id)
    return _logits(model, items, pad_token_id), items


def _load_clef(
    folder,
    max_seq_length,
    dtype,
    load_in_4bit,
    full_finetuning,
    token,
    use_gradient_checkpointing,
    kwargs,
):
    from safetensors.torch import load_file

    from .clef import JointSchemaHead

    max_len = int(max_seq_length or CLEF_MAX_LEN)
    if dtype == torch.float16:
        # Qwen3.5's gated delta net NaNs in pure fp16; Unsloth picks the dtype and keeps it in fp32 autocast.
        print("Unsloth: Clef ignores dtype = torch.float16 and lets Unsloth pick the dtype.")
        dtype = None
    fast = _device().type != "cpu"
    if fast:
        from .loader import FastModel
        backbone, processor = FastModel.from_pretrained(
            str(folder),
            max_seq_length = max_len,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            full_finetuning = full_finetuning,
            token = token,
            use_gradient_checkpointing = use_gradient_checkpointing,
            **kwargs,
        )
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        backbone = AutoModelForImageTextToText.from_pretrained(
            str(folder), dtype = dtype or torch.float32
        )
        processor = AutoProcessor.from_pretrained(str(folder))
        if not full_finetuning:
            backbone.requires_grad_(False)
    backbone.config.use_cache = False
    head_config = json.loads((folder / _CLEF_HEAD_FILES[1]).read_text(encoding = "utf-8"))
    head = JointSchemaHead(**head_config)
    head.load_state_dict(load_file(str(folder / _CLEF_HEAD_FILES[0])), strict = True)
    device = next(backbone.parameters()).device
    # The head trains in fp32 over the 16-bit backbone, as the Laya head does.
    model = ClefDecisionModel(backbone, head.to(device = device, dtype = torch.float32))
    config = {"layout": "clef", "max_len": max_len, "temperature": [1.0] * 3}
    saved = folder / "unsloth_decision_config.json"
    if saved.is_file():
        config.update(json.loads(saved.read_text(encoding = "utf-8")), max_len = max_len)
    model.decision_config = config
    _mark_full_finetuning(model, full_finetuning)
    model._unsloth_fast_backbone = fast
    model._saved_temp_tokenizer = processor
    model._unsloth_source_folder = str(folder)
    model.save_pretrained_merged = types.MethodType(save_pretrained_merged, model)
    model.push_to_hub_merged = types.MethodType(push_to_hub_merged, model)
    return model, processor


def _clef_peft_model(model, target_modules, use_gradient_checkpointing, random_state, **kwargs):
    # LoRA on the language model only; the vision tower and the lm_head the head reads stay frozen.
    if target_modules == "all-linear":
        target_modules = None
    backbone = model.encoder
    if getattr(model, "_unsloth_fast_backbone", False):
        from .loader import FastModel
        model.encoder = FastModel.get_peft_model(
            backbone,
            target_modules = target_modules,
            finetune_vision_layers = False,
            finetune_language_layers = True,
            use_gradient_checkpointing = use_gradient_checkpointing,
            random_state = random_state,
            **kwargs,
        )
    else:
        from peft import LoraConfig, get_peft_model
        from transformers import set_seed

        set_seed(random_state)
        kwargs.pop("max_seq_length", None)
        model.encoder = get_peft_model(
            backbone,
            LoraConfig(
                target_modules = target_modules
                or r"model\.language_model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)",
                **kwargs,
            ),
        )
    return model


def _fold_temperature(state: dict, temperature: float):
    # logits / T = prior / T + gate * (joint_scale / T * cosine + residual / T), exactly, as long
    # as both exp(scale) stay under the head's clamp at log(100).
    limit, shift = math.log(100.0), math.log(temperature)
    folded = dict(state)
    for name in ("prior_logit_scale", "joint_logit_scale"):
        scale = state[name].float().clamp(max = limit) - shift
        if scale > limit:
            return None
        folded[name] = scale.to(state[name].dtype)
    last = max(
        int(k.split(".")[1])
        for k in state
        if k.startswith("residual_scorer.") and k.endswith(".weight")
    )
    for kind in ("weight", "bias"):
        key = f"residual_scorer.{last}.{kind}"
        folded[key] = (state[key].float() / temperature).to(state[key].dtype)
    return folded


def _save_clef(self, save_directory, tokenizer) -> None:
    import shutil

    from safetensors.torch import save_file

    output = Path(save_directory)
    output.mkdir(parents = True, exist_ok = True)
    for name in _CLEF_HEAD_FILES:
        (output / name).unlink(missing_ok = True)
    encoder = self.encoder
    if hasattr(encoder, "save_pretrained_merged"):
        # Unsloth's merge dequantizes a 4-bit base and writes the processor files too.
        encoder.save_pretrained_merged(str(output), tokenizer, save_method = "merged_16bit")
    else:
        if hasattr(encoder, "merge_and_unload"):
            encoder = copy.deepcopy(encoder).merge_and_unload()
        encoder.save_pretrained(str(output))
        tokenizer.save_pretrained(str(output))
    source = Path(getattr(self, "_unsloth_source_folder", "") or output)
    for name in _CLEF_EXTRA_FILES:
        if (source / name).is_file() and not (output / name).exists():
            shutil.copyfile(source / name, output / name)
    config = {**self.decision_config, "fine_tuned": True}
    state = self.head.state_dict()
    temperature = config.pop("global_temperature", None)
    folded = _fold_temperature(state, temperature) if temperature else None
    if folded is not None:
        # Cloudflare's loader reads only the head, so the calibration lives in its weights;
        # the per type temperatures stay relative to the folded logits.
        state, config["folded_temperature"] = folded, temperature
        config["temperature"] = [t / temperature for t in config.get("temperature", [1.0] * 3)]
        if config.get("temperature_by_options"):
            config["temperature_by_options"] = {
                k: v / temperature for k, v in config["temperature_by_options"].items()
            }
    elif temperature:
        # Folding would push a logit scale past the head's clamp, so only Unsloth applies it.
        config["global_temperature"] = temperature
    weights = {}
    for name, value in state.items():
        value = value.detach().to("cpu", torch.bfloat16).contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(
                f"Unsloth: head weight {name} is not finite, so the model cannot be saved."
            )
        weights[name] = value
    (output / "unsloth_decision_config.json").write_text(
        json.dumps(config, indent = 2), encoding = "utf-8"
    )
    (output / _CLEF_HEAD_FILES[1]).write_text(
        json.dumps(self.head.config, indent = 2), encoding = "utf-8"
    )
    # Written last: the head marks a complete Clef checkpoint.
    save_file(weights, str(output / "joint_head.safetensors.tmp"))
    os.replace(output / "joint_head.safetensors.tmp", output / _CLEF_HEAD_FILES[0])


class _LengthGroupedBatches(torch.utils.data.Sampler):
    # Similar-length micro-batches, shuffled so a step mixes lengths; longest first to show OOM.
    def __init__(self, lengths: list, batch_size: int, seed: int):
        self.lengths, self.batch_size, self.seed, self.epoch = lengths, batch_size, seed, 0

    def __len__(self):
        return len(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        from transformers.trainer_pt_utils import get_length_grouped_indices

        seed = self.seed + self.epoch
        order = get_length_grouped_indices(
            self.lengths, self.batch_size, generator = torch.Generator().manual_seed(seed)
        )
        batches = [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        short = batches.pop() if len(batches) > 1 and len(batches[-1]) < self.batch_size else []
        rest = batches[1:]
        random.Random(seed).shuffle(rest)
        return iter([index for batch in batches[:1] + rest + [short] for index in batch])


class DecisionTrainer(Trainer):
    def __init__(
        self,
        model = None,
        args = None,
        *,
        head_learning_rate: Optional[float] = None,
        label_smoothing: float = 0.0,
        brier_weight: float = 0.0,
        tokenizer = None,
        **kwargs,
    ):
        args = copy.copy(args) if args is not None else TrainingArguments(output_dir = "tmp_trainer")
        args.remove_unused_columns = False
        # The model trains on one GPU: no DataParallel, and the batch stays per_device_train_batch_size.
        if args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
            args._n_gpu = 1
        if args.label_names is None:
            args.label_names = ["target"]
        clef = getattr(model, "is_clef", False)
        if args.gradient_checkpointing:
            # Trainer would call model.gradient_checkpointing_enable, which only the encoder has.
            # Clef's backbone already got Unsloth's checkpointing when it loaded.
            args.gradient_checkpointing = False
            if not clef and not model.encoder.is_gradient_checkpointing:
                _gradient_checkpointing(model, True)
        if kwargs.get("processing_class") is None:
            kwargs["processing_class"] = tokenizer
        if kwargs.get("data_collator") is None:
            processing_class = kwargs["processing_class"]
            if processing_class is None:
                pad_token_id = getattr(model, "_saved_temp_tokenizer", None)
                pad_token_id = (
                    getattr(pad_token_id, "tokenizer", pad_token_id).pad_token_id
                    if clef
                    else model.encoder.config.pad_token_id
                )
            else:
                pad_token_id = getattr(processing_class, "tokenizer", processing_class).pad_token_id
            kwargs["data_collator"] = (ClefDataCollator if clef else DecisionDataCollator)(
                pad_token_id
            )
        self.head_learning_rate = head_learning_rate
        self.label_smoothing, self.brier_weight = label_smoothing, brier_weight
        super().__init__(model = model, args = args, **kwargs)

    def _get_train_sampler(self, train_dataset = None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        # With one micro-batch per step, length grouping would make each step one question type.
        if dataset is None or self.args.gradient_accumulation_steps == 1:
            return super()._get_train_sampler(train_dataset)
        return _LengthGroupedBatches(
            [len(item["input_ids"]) for item in dataset], self.args.train_batch_size, self.args.seed
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        target = inputs.pop("target")
        logits, _ = model(**inputs)
        loss = _soft_cross_entropy(
            logits, target, inputs["marker_mask"], self.label_smoothing, self.brier_weight
        )
        return (loss, {"loss": loss, "logits": logits}) if return_outputs else loss

    def create_optimizer(self, model = None):
        if self.optimizer is not None:
            return self.optimizer
        model = self.model if model is None else model
        decay = set(self.get_decay_parameter_names(model))
        head_lr = HEAD_LEARNING_RATE if self.head_learning_rate is None else self.head_learning_rate
        groups = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                groups.setdefault((name.startswith("encoder."), name in decay), []).append(param)
        optimizer_cls, optimizer_kwargs = (
            self.optimizer_cls_and_kwargs or self.get_optimizer_cls_and_kwargs(self.args, model)
        )
        self.optimizer = optimizer_cls(
            [
                {
                    "params": params,
                    "weight_decay": self.args.weight_decay if decayed else 0.0,
                    **({} if encoder else {"lr": head_lr}),
                }
                for (encoder, decayed), params in groups.items()
            ],
            **optimizer_kwargs,
        )
        return self.optimizer


@torch.no_grad()
def _logits(
    model,
    items: list,
    pad_token_id: int,
    batch_size: int = 16,
) -> list:
    device = next(model.parameters()).device
    amp_dtype = _amp_dtype(device)
    collate = DecisionDataCollator(pad_token_id)
    was_training = model.training
    model.eval()
    out = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        batch = collate(chunk)
        batch.pop("target")
        with torch.autocast(device.type, dtype = amp_dtype, enabled = amp_dtype is not None):
            logits, _ = model(**{k: v.to(device) for k, v in batch.items()})
        logits = logits.float().cpu()
        out.extend(logits[row, : len(item["markers"])] for row, item in enumerate(chunk))
    model.train(was_training)
    return out


def _metrics(logits, items, temperatures) -> dict:
    import numpy as np

    conf, correct, loss = [], [], []
    for z, item, temperature in zip(logits, items, temperatures):
        log_p = torch.log_softmax(z / temperature, -1)
        conf.append(float(log_p.exp().max()))
        correct.append(float(int(log_p.argmax()) == item["label"]))
        loss.append(float(-(torch.tensor(item["target"]) * log_p).sum()))
    return {
        "accuracy": float(np.mean(correct)),
        "ece": _laya().common.ece_score(np.array(conf), np.array(correct)),
        "loss": float(np.mean(loss)),
    }


def _fit_temperature(logits, items) -> float:
    options = max(len(z) for z in logits)
    z = torch.full((len(logits), options), -1e4)
    target = torch.zeros((len(logits), options))
    for i, (row, item) in enumerate(zip(logits, items)):
        z[i, : len(row)] = row
        target[i, : len(row)] = torch.tensor(item["target"])
    log_t = torch.zeros(1, requires_grad = True)
    optimizer = torch.optim.LBFGS([log_t], lr = 0.1, max_iter = 100)

    def closure():
        optimizer.zero_grad()
        loss = -(target * torch.log_softmax(z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def _fit_temperatures(logits, items, indices, fallback: list) -> tuple:
    clamp = _laya().common.clamp_temperature
    temperature, fitted = list(fallback), set()
    for qtype in range(3):
        chosen = [i for i in indices if items[i]["qtype"] == qtype]
        if len(chosen) >= MIN_CALIBRATION_ITEMS:
            temperature[qtype] = clamp(
                _fit_temperature([logits[i] for i in chosen], [items[i] for i in chosen])
            )
            fitted.add(qtype)
    return temperature, fitted


def _served_temperatures(config: dict, logits, items) -> list:
    common = _laya().common
    per_type = [common.clamp_temperature(t) for t in config.get("temperature", [1.0] * 3)]
    buckets = {
        key: common.clamp_temperature(value)
        for key, value in (config.get("temperature_by_options") or {}).items()
    }
    return [
        buckets.get(common.temp_bucket(item["qtype"], len(z)), per_type[item["qtype"]])
        for z, item in zip(logits, items)
    ]


def save_pretrained_merged(
    self,
    save_directory,
    tokenizer = None,
    save_method = "merged_16bit",
    **kwargs,
) -> None:
    from safetensors.torch import save_file

    if save_method != "merged_16bit":
        raise NotImplementedError(
            f"Unsloth: decision models are saved merged in 16-bit, not as {save_method!r}."
        )
    tokenizer = self._saved_temp_tokenizer if tokenizer is None else tokenizer
    if getattr(self, "is_clef", False):
        return _save_clef(self, save_directory, tokenizer)
    encoder = self.encoder
    if hasattr(encoder, "merge_and_unload"):
        # Merged on a CPU copy: the model keeps its adapters and the GPU never holds a second encoder.
        device = next(encoder.parameters()).device
        encoder.to("cpu")
        try:
            encoder = copy.deepcopy(encoder).merge_and_unload()
        finally:
            self.encoder.to(device)
    state = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}
    state.update((k, v) for k, v in self.state_dict().items() if not k.startswith("encoder."))
    weights = {}
    for name, value in state.items():
        value = value.detach().to("cpu", torch.float16).contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(
                f"Unsloth: {name} has NaN or values too large for float16, so the model cannot be saved."
            )
        weights[name] = value
    del encoder, state
    config = json.dumps({**self.decision_config, "fine_tuned": True}, indent = 2)

    output = Path(save_directory)
    output.mkdir(parents = True, exist_ok = True)
    (output / "rl_agent_config.json").unlink(missing_ok = True)
    save_file(weights, str(output / "model.safetensors"))
    (output / "encoder").mkdir(exist_ok = True)
    (output / "encoder" / "config.json").write_text(self._unsloth_encoder_config, encoding = "utf-8")
    tokenizer.save_pretrained(str(output / "tokenizer"))
    _laya().agent._fix_tokenizer_config(str(output))
    # Written last: a folder with rl_agent_config.json is a complete checkpoint.
    partial = output / "rl_agent_config.json.tmp"
    partial.write_text(config, encoding = "utf-8")
    os.replace(partial, output / "rl_agent_config.json")


def push_to_hub_merged(
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


# Decision models in Laya's rl_agent_config.json layout: any encoder plus a typed decision head.
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
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        folder = _checkpoint_folder(model_name, subfolder, token, revision, local_files_only)
        if load_in_4bit and not is_clef_checkpoint(folder):
            raise NotImplementedError(
                "Unsloth: Laya decision models train in 16-bit, so load_in_4bit is not supported."
            )
        if is_clef_checkpoint(folder):
            return _load_clef(
                folder,
                max_seq_length,
                dtype,
                load_in_4bit,
                full_finetuning,
                token,
                use_gradient_checkpointing,
                kwargs,
            )
        config = json.loads((folder / "rl_agent_config.json").read_text(encoding = "utf-8"))
        # The base model's training record does not describe the fine-tune.
        config.pop("training", None)
        tokenizer = AutoTokenizer.from_pretrained(str(folder / "tokenizer"))
        encoder = AutoModel.from_config(
            _encoder_config(folder / "encoder"), attn_implementation = "sdpa"
        )
        model = _laya().common.DecisionModel(
            encoder, config.get("head_layers", 2), len(config.get("act_costs", {})) + 1
        )
        model.load_state_dict(load_file(str(folder / "model.safetensors")), strict = True)
        # ModernBERT would otherwise torch.compile parts of the encoder.
        model.encoder.config.reference_compile = False

        positions = int(getattr(model.encoder.config, "max_position_embeddings", TRAIN_MAX_LEN))
        wanted = max_seq_length or max(int(config.get("max_len", 512)), TRAIN_MAX_LEN)
        config["max_len"] = min(positions, int(wanted))
        config["head_max_len"] = min(
            config["max_len"] // 2, max(int(config.get("head_max_len", 192)), TRAIN_HEAD_MAX_LEN)
        )
        model.decision_config = config

        device = _device()
        model.float()
        if not full_finetuning:
            dtype = dtype or (torch.float32 if device.type == "cpu" else torch.float16)
            # Weights only: the rotary buffers stay fp32, as they are when the model is served.
            for param in model.encoder.parameters():
                param.requires_grad_(False)
                param.data = param.data.to(dtype)
        model.to(device)
        _mark_full_finetuning(model, full_finetuning)
        _gradient_checkpointing(model, use_gradient_checkpointing)

        model._saved_temp_tokenizer = tokenizer
        model._unsloth_encoder_config = (folder / "encoder" / "config.json").read_text(
            encoding = "utf-8"
        )
        model.save_pretrained_merged = types.MethodType(save_pretrained_merged, model)
        model.push_to_hub_merged = types.MethodType(push_to_hub_merged, model)
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
        if hasattr(model.encoder, "peft_config"):
            raise RuntimeError("Unsloth: You already added LoRA adapters to your model!")
        if getattr(model, "is_clef", False):
            return _clef_peft_model(
                model,
                r = r,
                target_modules = target_modules,
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                bias = bias,
                use_gradient_checkpointing = use_gradient_checkpointing,
                random_state = random_state,
                use_rslora = use_rslora,
                modules_to_save = modules_to_save,
                **kwargs,
            )
        from peft import LoraConfig, get_peft_model
        from transformers import set_seed

        set_seed(random_state)
        model.encoder = get_peft_model(
            model.encoder,
            LoraConfig(
                r = r,
                target_modules = target_modules,
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                bias = bias,
                layers_to_transform = layers_to_transform,
                layers_pattern = layers_pattern,
                use_rslora = use_rslora,
                use_dora = use_dora,
                modules_to_save = modules_to_save,
                init_lora_weights = init_lora_weights,
                loftq_config = {} if loftq_config is None else loftq_config,
                **kwargs,
            ),
        )
        _gradient_checkpointing(model, use_gradient_checkpointing)
        return model

    @staticmethod
    def for_inference(model):
        model.eval()
        return model

    @staticmethod
    def for_training(model, use_gradient_checkpointing = True):
        model.train()
        if getattr(model, "is_clef", False):
            return model
        if not use_gradient_checkpointing:
            model.encoder.gradient_checkpointing_disable()
        elif not model.encoder.is_gradient_checkpointing:
            _gradient_checkpointing(model, use_gradient_checkpointing)
        return model

    @staticmethod
    def build_dataset(
        rows,
        tokenizer,
        model,
        validate: Optional[Callable[[str, dict], None]] = None,
    ) -> tuple:
        max_len = int(model.decision_config.get("max_len", 512))
        head_max_len = int(model.decision_config.get("head_max_len", 192))
        items, report, skips = [], {"total": 0, "skipped": 0, "reason": None}, {}

        def skip(
            index,
            reason,
            name = None,
        ):
            report["skipped"] += 1
            where = f"row {index + 1}" if name is None else f'row {index + 1}: "{name}"'
            skips.setdefault(reason, [0, f"{where} {reason}"])[0] += 1

        if getattr(model, "is_clef", False):
            items = _clef_items(rows, tokenizer, max_len, validate, report, skip)
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
            report["reason"] = (
                example if count == 1 else f"{example} (and {count - 1:,} more like it)"
            )
        return items, report

    @staticmethod
    def split_holdout(
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

    @staticmethod
    def evaluate(model, tokenizer, items: list) -> dict:
        logits, items = _decision_logits(model, tokenizer, items)
        return _metrics(logits, items, _served_temperatures(model.decision_config, logits, items))

    @staticmethod
    def calibrate(model, tokenizer, items: list) -> dict:
        common = _laya().common
        config = model.decision_config
        fallback = [common.clamp_temperature(t) for t in config.get("temperature", [1.0] * 3)]
        logits, items = _decision_logits(model, tokenizer, items)
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
        if getattr(model, "is_clef", False):
            # The released head has one set of logit scales, so one temperature for every type.
            config["global_temperature"] = common.clamp_temperature(_fit_temperature(logits, items))
        buckets = {
            key: value
            for key, value in (config.pop("temperature_by_options", None) or {}).items()
            if common.QTYPES.get(key.split(":")[0]) not in fitted
        }
        if buckets:
            config["temperature_by_options"] = buckets
        return {**_metrics(logits, items, per_item), "fitted_types": sorted(fitted)}
