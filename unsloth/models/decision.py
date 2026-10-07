# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

__all__ = [
    "FastDecisionModel",
    "DecisionTrainer",
]

import contextlib
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
from ._decision_fast import compiled_encoder, pad_length

TRAIN_MAX_LEN, TRAIN_HEAD_MAX_LEN = 1024, 256
HOLDOUT_MAX = 400
MIN_CALIBRATION_ITEMS = 10
HEAD_LEARNING_RATE = 1e-4
QUESTION_TYPES = ("choice", "score", "noul")
# DecisionTrainer defaults for Clef; every key is also a DecisionTrainer argument.
CLEF_RECIPE = {}
_FILES = ("rl_agent_config.json", "model.safetensors")
_DIRS = ("encoder", "tokenizer")
_CLEF_HEAD_FILES = ("joint_head.safetensors", "joint_head_config.json")
_ADAPTER_CONFIG = "adapter_config.json"
_DECISION_CONFIG = "unsloth_decision_config.json"
# Copied byte for byte from the checkpoint a Clef fine-tune started from: Cloudflare's loader and
# license, and the tokenizer / processor files, which transformers would otherwise rewrite (a
# re-saved tokenizer_config.json trips transformers' "incorrect regex pattern" warning).
_CLEF_EXTRA_FILES = (
    "joint_schema_model.py",
    "LICENSE",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "generation_config.json",
)
CLEF_MAX_LEN = 4096
# predict() and serving read this many tokens whatever the training length: cutting a state drops evidence.
CLEF_SERVE_MAX_LEN = 16384
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
    # Cloudflare's Clef layout: a backbone (merged, or LoRA adapters over a base) plus a joint schema head.
    folder = Path(folder)
    return all((folder / name).is_file() for name in _CLEF_HEAD_FILES) and (
        (folder / "config.json").is_file() or (folder / _ADAPTER_CONFIG).is_file()
    )


def _is_clef_adapter(folder) -> bool:
    folder = Path(folder)
    return not (folder / "config.json").is_file() and (folder / _ADAPTER_CONFIG).is_file()


def _is_clef_repo(model_name, prefix, token, revision) -> Optional[bool]:
    # Asked up front: Unsloth's download wrapper rejects a snapshot that lacks an exact file it was
    # asked for, so a Laya pattern on a Clef repo (or the reverse) would fail as "incomplete".
    from huggingface_hub import HfApi, constants

    if constants.HF_HUB_OFFLINE:
        return None
    try:
        files = HfApi(token = token).list_repo_files(model_name, revision = revision)
    except Exception:
        return None
    return prefix + _CLEF_HEAD_FILES[1] in files


def _is_plain_lm(model_name, subfolder, token, revision, local_files_only) -> bool:
    # Unknown (offline, no access) answers False, so the checkpoint loader names what is missing.
    markers = {_FILES[0], _CLEF_HEAD_FILES[1]}
    if subfolder:
        return False
    root = Path(str(model_name)).expanduser()
    if root.is_dir():
        has = lambda name: (root / name).is_file()
        return (has("config.json") or has(_ADAPTER_CONFIG)) and not any(map(has, markers))
    from huggingface_hub import constants

    if not (local_files_only or constants.HF_HUB_OFFLINE):
        try:
            from huggingface_hub import HfApi
            files = HfApi(token = token).list_repo_files(str(model_name), revision = revision)
        except Exception:
            files = None
        if files is not None:
            names = {name.rsplit("/", 1)[-1] for name in files}
            return bool({"config.json", _ADAPTER_CONFIG} & set(files)) and not (names & markers)
    from huggingface_hub import try_to_load_from_cache

    def cached(name) -> bool:
        try:
            return isinstance(try_to_load_from_cache(str(model_name), name, revision = revision), str)
        except Exception:
            return False

    return (cached("config.json") or cached(_ADAPTER_CONFIG)) and not any(map(cached, markers))


def _checkpoint_folder(model_name, subfolder, token, revision, local_files_only) -> Path:
    root = Path(model_name).expanduser()
    if not root.is_dir():
        from huggingface_hub import snapshot_download as cached_snapshot

        try:
            from unsloth_zoo.hf_xet_fallback import (
                snapshot_download_with_xet_fallback as snapshot_download,
            )
        except ImportError:
            snapshot_download = cached_snapshot
        prefix = f"{subfolder}/" if subfolder else ""
        laya = [prefix + name for name in _FILES] + [f"{prefix}{name}/*" for name in _DIRS]
        clef = None if local_files_only else _is_clef_repo(model_name, prefix, token, revision)
        if clef is None:
            # Offline: the cache already holds one layout or the other.
            root = Path(
                cached_snapshot(
                    model_name,
                    token = token,
                    revision = revision,
                    local_files_only = True,
                    allow_patterns = laya + [prefix + "*.json"],
                )
            )
            clef = (root / prefix / _CLEF_HEAD_FILES[1]).is_file()
        # Laya repos hold several checkpoints in subfolders, so only the asked one is fetched.
        root = Path(
            snapshot_download(
                model_name,
                token = token,
                revision = revision,
                local_files_only = local_files_only,
                allow_patterns = [prefix + "*"] if clef else laya,
            )
        )
    folder = root / subfolder if subfolder else root
    if not is_decision_checkpoint(folder):
        raise ValueError(
            f"Unsloth: {folder} is not a decision model checkpoint "
            "(rl_agent_config.json, model.safetensors, encoder/ and tokenizer/, "
            "or a Clef backbone with joint_head.safetensors and joint_head_config.json). "
            'To turn a plain language model into a decision model, pass decision_head = "clef".'
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


def _pin_device_map(kwargs) -> None:
    """Without a caller's device_map, FastModel's planner may split the backbone over every visible GPU, but the decision head reads its embedding rows on one device. Pin the load to this process's device, its LOCAL_RANK under torchrun."""
    if kwargs.get("device_map") is not None:
        return
    from .loader_utils import prepare_device_map

    device_map, _ = prepare_device_map()
    if device_map is None:
        device = _device()
        backend = getattr(torch, device.type, None)
        index = backend.current_device() if backend is not None and backend.is_available() else 0
        device_map = {"": f"{device.type}:{index}"}
    kwargs["device_map"] = device_map


def _amp_dtype(device):
    if device.type == "cuda":
        return torch.bfloat16 if is_bfloat16_supported() else torch.float16
    return torch.bfloat16 if device.type == "xpu" else None


def _no_cudnn_attention():
    # cuDNN SDPA rebuilds its bf16 plan for every new sequence length, about 100x slower steps on a B200.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    )


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


def _lean_lora_forward(self, x, *args, **kwargs):
    adapter = self._unsloth_adapter
    if (
        self.disable_adapters
        or self.merged
        or args
        or kwargs
        or (
            not torch.is_autocast_enabled(x.device.type)
            and x.dtype != self.lora_A[adapter].weight.dtype
        )
    ):
        return self._unsloth_peft_forward(x, *args, **kwargs)
    # PEFT's maths without its per-call checks, its round trip of x through the fp32 adapter
    # dtype, and the multiply when scaling is 1.
    lora = self.lora_B[adapter](self.lora_A[adapter](x))
    scaling = self.scaling[adapter]
    return self.base_layer(x) + (lora if scaling == 1 else lora * scaling)


def _lean_lora(encoder) -> None:
    # Plain LoRA (one adapter, no dropout, DoRA or other variant) skips PEFT's per-call checks and casts.
    from peft.tuners.lora.layer import Linear

    # Unsloth's compiler (a FastModel load earlier in this process) already gave PEFT a compiled forward.
    if Linear.forward.__name__ == "unsloth_forward":
        return
    for module in encoder.modules():
        if type(module) is not Linear or len(module.lora_A) != 1 or module.lora_variant:
            continue
        adapter = next(iter(module.lora_A))
        if not isinstance(module.lora_dropout[adapter], torch.nn.Identity):
            continue
        # Bound methods, so the deepcopy that merges for saving rebinds them to the copy.
        module._unsloth_adapter = adapter
        module._unsloth_peft_forward = module.forward
        module.forward = types.MethodType(_lean_lora_forward, module)


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


def _soft_cross_entropy(logits, target, mask):
    logits = logits.float().masked_fill(~mask, -1e4)
    return -(target * torch.log_softmax(logits, -1)).sum(-1).mean()


def _decision_loss(
    logits,
    target,
    mask,
    ordinal = None,
    label_smoothing = 0.0,
    brier_weight = 0.0,
    ordinal_weight = 0.0,
):
    # Cloudflare's Clef recipe: label-smoothed cross entropy, a Brier term for calibration, and
    # partial credit on score questions (expected distance from the gold level).
    if not (label_smoothing or brier_weight or ordinal_weight):
        return _soft_cross_entropy(logits, target, mask)
    log_p = torch.log_softmax(logits.float().masked_fill(~mask, -1e4), -1)
    smoothed = target
    if label_smoothing:
        uniform = mask.float() / mask.sum(-1, keepdim = True).clamp(min = 1)
        smoothed = (1.0 - label_smoothing) * target + label_smoothing * uniform
    loss = -(smoothed * log_p).sum(-1).mean()
    p = log_p.exp()
    if brier_weight:
        loss = loss + brier_weight * ((p - target) ** 2 * mask).sum(-1).mean()
    if ordinal_weight and ordinal is not None and ordinal.any():
        levels = torch.arange(p.shape[-1], device = p.device, dtype = p.dtype)
        distance = (levels[:, None] - levels[None, :]).abs()
        span = (mask.sum(-1) - 1).clamp(min = 1).to(p.dtype)
        expected = torch.einsum("ri,ij,rj->r", p, distance, target) / span
        loss = loss + ordinal_weight * (expected * ordinal).sum() / ordinal.sum()
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
    # permute_fields re-encodes each training record with its fields shuffled, as Cloudflare's
    # training data permutes field order; evaluation always keeps the dataset's order.
    def __init__(
        self,
        pad_token_id: int,
        tokenizer = None,
        max_len = None,
        permute_fields = False,
        seed = 3407,
    ):
        self.pad_token_id = pad_token_id
        self.tokenizer, self.max_len = getattr(tokenizer, "tokenizer", tokenizer), max_len
        self.permute_fields = permute_fields and tokenizer is not None
        self.random = random.Random(seed)

    def _permuted(self, item) -> dict:
        from .clef import encode_record

        order = list(range(len(item["targets"])))
        self.random.shuffle(order)
        names = list(item["source"]["questions"])
        questions = {names[i]: item["source"]["questions"][names[i]] for i in order}
        try:
            record = encode_record(
                self.tokenizer,
                {"state": item["source"]["state"], "questions": questions},
                max_length = self.max_len,
            )
        except ValueError:
            return item
        return {
            **item,
            "input_ids": list(record.input_ids),
            "record": record,
            "qtypes": [item["qtypes"][i] for i in order],
            "targets": [item["targets"][i] for i in order],
        }

    def __call__(self, items: list) -> dict:
        from .clef import QUESTION_TYPES as CLEF_TYPES

        if self.permute_fields:
            items = [self._permuted(item) if "source" in item else item for item in items]
        rows, length = len(items), max(len(item["input_ids"]) for item in items)
        targets = [target for item in items for target in item["targets"]]
        options = max(len(target) for target in targets)
        batch = {
            "input_ids": torch.full((rows, length), self.pad_token_id, dtype = torch.long),
            "attention_mask": torch.zeros((rows, length), dtype = torch.long),
            "records": [item["record"] for item in items],
            "marker_mask": torch.zeros((len(targets), options), dtype = torch.bool),
            "target": torch.zeros((len(targets), options), dtype = torch.float32),
            "ordinal": torch.tensor(
                [qtype == CLEF_TYPES["score"] for item in items for qtype in item["qtypes"]],
                dtype = torch.bool,
            ),
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

    def forward(
        self,
        input_ids,
        attention_mask,
        records,
        head = None,
        **kwargs,
    ):
        head = self.head if head is None else head
        backbone = self._backbone()
        text_model = backbone.model
        text_model = getattr(text_model, "language_model", text_model)
        hidden = text_model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            use_cache = False,
            return_dict = True,
        ).last_hidden_state
        logits = head(
            hidden,
            input_ids,
            attention_mask,
            records,
            backbone.get_output_embeddings().weight.detach(),
        )
        if hasattr(logits, "flat_padded"):
            return logits.flat_padded, None
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
                "source": {"state": state, "questions": kept},
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
    amp_dtype = _clef_amp_dtype(model, device)
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


def _decision_logits(
    model,
    tokenizer,
    items: list,
    batch_size = None,
) -> tuple:
    pad_token_id = getattr(tokenizer, "tokenizer", tokenizer).pad_token_id
    if getattr(model, "is_clef", False):
        # A run that lowered its batch to fit a 9B / 27B backbone scores in that batch too.
        return _clef_logits(model, items, pad_token_id, batch_size or 4)
    return _logits(model, items, pad_token_id), items


def _predicted(question: dict, answer: dict, probabilities: dict) -> dict:
    # The Decision API answer plus "answer": the option (choice), True / False (noul) or level number (score).
    kind = question["type"]
    best = max(probabilities, key = probabilities.__getitem__)
    return {
        **answer,
        "answer": int(best) if kind == "score" else best == "true" if kind == "noul" else best,
        "probabilities": answer.get("probabilities")
        or {key: round(float(value), 4) for key, value in probabilities.items()},
    }


@torch.no_grad()
def _clef_decide(
    model,
    tokenizer,
    state,
    questions: dict,
    max_length = None,
    predicted = False,
) -> dict:
    # One Decision API request over every question in one prefill, at served temperatures.
    from .clef import encode_record, systemone_answer

    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    max_length = int(max_length or model.decision_config.get("max_len", CLEF_MAX_LEN))
    encoded = encode_record(
        tokenizer, {"state": state, "questions": questions}, max_length = max_length
    )
    device = next(model.parameters()).device
    amp_dtype = _clef_amp_dtype(model, device)
    ids = torch.tensor([encoded.input_ids], device = device)
    was_training = model.training
    model.eval()
    try:
        with torch.autocast(device.type, dtype = amp_dtype, enabled = amp_dtype is not None):
            logits, _ = model(input_ids = ids, attention_mask = torch.ones_like(ids), records = [encoded])
    finally:
        model.train(was_training)
    rows = [
        row[: len(question.option_ids)]
        for question, row in zip(encoded.questions, logits.float().cpu())
    ]
    scales = _served_temperatures(
        model.decision_config,
        rows,
        [
            {"qtype": QUESTION_TYPES.index(questions[q.question_id]["type"])}
            for q in encoded.questions
        ],
    )
    answers = {}
    for question, row, scale in zip(encoded.questions, rows, scales):
        probabilities = dict(zip(question.option_ids, (row / scale).softmax(-1).tolist()))
        asked = questions[question.question_id]
        answer = systemone_answer(asked, probabilities)
        answers[question.question_id] = (
            _predicted(asked, answer, probabilities) if predicted else answer
        )
    return {
        "answers": answers,
        "input_tokens": len(encoded.input_ids),
        "truncated": _clef_truncated(tokenizer, state, questions, encoded, max_length),
    }


def _clef_truncated(tokenizer, state, questions, encoded, max_length) -> bool:
    # One spare token tells a cut state from one that fits exactly.
    if len(encoded.input_ids) < max_length:
        return False
    from .clef import encode_record

    record = {"state": state, "questions": questions}
    return len(encode_record(tokenizer, record, max_length = max_length + 1).input_ids) > max_length


def _laya_decide(
    model,
    tokenizer,
    state,
    questions: dict,
    predicted = False,
) -> dict:
    from .clef import systemone_answer

    common = _laya().common
    config = model.decision_config
    max_len = int(config.get("max_len", 512))
    items, keys = [], []
    for name, question in questions.items():
        internal = _internal(question)
        ids, markers = common.build_sequence(
            tokenizer, _parsed(state), internal, max_len, int(config.get("head_max_len", 192))
        )
        if len(markers) != len(_option_keys(internal)):
            raise DecisionDataError(f'"{name}" has more options than fit in {max_len} tokens')
        items.append(
            {
                "input_ids": ids,
                "markers": markers,
                "qtype": common.QTYPES[internal["t"]],
                "target": [0.0] * len(markers),
            }
        )
        keys.append(_option_keys(internal))
    logits = _logits(model, items, tokenizer.pad_token_id)
    temperatures = _served_temperatures(config, logits, items)
    answers = {}
    for (name, question), z, t, options in zip(questions.items(), logits, temperatures, keys):
        probabilities = dict(zip(options, torch.softmax(z / t, -1).tolist()))
        answer = systemone_answer(question, probabilities)
        answers[name] = _predicted(question, answer, probabilities) if predicted else answer
    return {
        "answers": answers,
        "input_tokens": max(len(item["input_ids"]) for item in items),
        "truncated": any(len(item["input_ids"]) >= max_len for item in items),
    }


# Kept in 16-bit like unsloth/Qwen3.8-27B-unsloth-bnb-4bit: Clef's backbone is a merged
# fine-tune, not stock Qwen, so it is quantized on load with the same dynamic list.
CLEF_4BIT_SKIP_MODULES = (
    "model.visual",
    r".*\.visual\..*",
    "in_proj_a",
    "in_proj_b",
    "in_proj_qkv",
)


def _clef_bnb_config(dtype):
    from transformers import BitsAndBytesConfig
    from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES

    float16 = dtype == torch.float16 or not is_bfloat16_supported()
    return BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16 if float16 else torch.bfloat16,
        llm_int8_skip_modules = list(SKIP_QUANTIZATION_MODULES) + list(CLEF_4BIT_SKIP_MODULES),
    )


def _clef_forced_float32(model) -> bool:
    return bool(getattr(model, "_unsloth_forced_float32", False))


def _clef_amp_dtype(model, device):
    # As _clef_mixed_precision trains (serving uses it too): the float32 path runs as loaded (Qwen3.5's
    # gated delta net overflows in fp16); anything else autocasts, fp16 on a T4.
    return None if _clef_forced_float32(model) else _amp_dtype(device)


def _load_clef(
    folder,
    max_seq_length,
    dtype,
    load_in_4bit,
    full_finetuning,
    token,
    use_gradient_checkpointing,
    kwargs,
    model_name = None,
):
    from safetensors.torch import load_file

    from .clef import JointSchemaHead

    max_len = int(max_seq_length or CLEF_MAX_LEN)
    config = {"layout": "clef", "max_len": max_len, "temperature": [1.0] * 3}
    saved = folder / _DECISION_CONFIG
    if saved.is_file():
        config.update(json.loads(saved.read_text(encoding = "utf-8")), max_len = max_len)
        # The parent run's training record does not describe the next fine-tune, as for Laya.
        config.pop("training", None)
    adapter = _is_clef_adapter(folder)
    if adapter:
        # LoRA adapters over the base LLM: the base comes from where it was trained from.
        base = config.get("base_model") or json.loads(
            (folder / _ADAPTER_CONFIG).read_text(encoding = "utf-8")
        ).get("base_model_name_or_path")
        if not base:
            raise ValueError(f"Unsloth: {folder} has adapters but does not name their base model.")
        config["base_model"] = str(base)
    else:
        # A later adapter save sits on these merged weights, not on what they were trained from.
        config["base_model"] = str(model_name or folder)
    fast = _device().type != "cpu"
    if fast:
        from .loader import FastModel

        # FastModel drops load_in_4bit for full finetuning, but not an explicit quantization config.
        if load_in_4bit and not full_finetuning and kwargs.get("quantization_config") is None:
            kwargs["quantization_config"] = _clef_bnb_config(dtype)
        _pin_device_map(kwargs)
        # A float16 request (or a GPU without bfloat16) puts Qwen3.5 on Unsloth's float32 path,
        # which stores bfloat16 weights: the gated delta net NaNs in pure float16.
        # An adapter folder loads its base and the adapters through Unsloth's own PEFT path.
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
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        source = config["base_model"] if adapter else str(folder)
        # Clef's own backbones are vision models; ones converted from a text-only LM are not.
        if (
            getattr(AutoConfig.from_pretrained(source, token = token), "vision_config", None)
            is not None
        ):
            from transformers import AutoModelForImageTextToText as AutoClass
        else:
            from transformers import AutoModelForCausalLM as AutoClass
            AutoProcessor = AutoTokenizer
        backbone = AutoClass.from_pretrained(source, dtype = dtype or torch.float32, token = token)
        if adapter:
            from peft import PeftModel
            backbone = PeftModel.from_pretrained(
                backbone, str(folder), is_trainable = not full_finetuning
            )
        processor = AutoProcessor.from_pretrained(str(folder))
        if not full_finetuning and not adapter:
            backbone.requires_grad_(False)
    backbone.config.use_cache = False
    head_config = json.loads((folder / _CLEF_HEAD_FILES[1]).read_text(encoding = "utf-8"))
    head = JointSchemaHead(**head_config)
    head.load_state_dict(load_file(str(folder / _CLEF_HEAD_FILES[0])), strict = True)
    device = next(backbone.parameters()).device
    # The head trains in fp32 over the 16-bit backbone, as the Laya head does.
    model = ClefDecisionModel(backbone, head.to(device = device, dtype = torch.float32))
    config["load_in_4bit"] = bool(load_in_4bit)
    model.decision_config = config
    _mark_full_finetuning(model, full_finetuning)
    model._unsloth_forced_float32 = bool(getattr(backbone, "_unsloth_forced_float32", False))
    model._unsloth_fast_backbone = fast
    model._saved_temp_tokenizer = processor
    source = folder
    if adapter:
        from .decision_from_lm import _source_folder
        source = _source_folder(config["base_model"], token, None, False) or folder
    model._unsloth_source_folder = str(source)
    model._unsloth_source_vocab = len(getattr(processor, "tokenizer", processor))
    _attach_clef_saving(model)
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
                or r"model\.(?:language_model\.)?layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)",
                **kwargs,
            ),
        )
    return model


def _commit_staged(
    staging: Path,
    output: Path,
    marker: str,
    stale = (),
) -> None:
    # Old files stay until the new ones are complete; the marker that makes a folder a
    # checkpoint moves in last, so a failed save leaves the previous checkpoint loadable.
    import shutil

    staged = sorted(staging.iterdir(), key = lambda path: path.name == marker)
    names = {path.name for path in staged}
    for path in staged:
        target = output / path.name
        if path.is_dir() and target.is_dir():
            old = output / f".{path.name}.unsloth-old"
            shutil.rmtree(old, ignore_errors = True)
            os.replace(target, old)
            os.replace(path, target)
            shutil.rmtree(old, ignore_errors = True)
        else:
            os.replace(path, target)
    for path in output.iterdir():
        if path.name not in names and any(path.match(pattern) for pattern in stale):
            path.unlink()


@contextlib.contextmanager
def _staging(output: Path):
    import shutil

    output.mkdir(parents = True, exist_ok = True)
    staging = Path(tempfile.mkdtemp(prefix = ".unsloth-save-", dir = output))
    try:
        yield staging
    finally:
        shutil.rmtree(staging, ignore_errors = True)


def _fold_temperature(state: dict, temperature: float) -> bool:
    # logits / T == prior / T + gate * (joint_scale / T * cosine + residual / T): exact while the
    # rescaled log-scales stay under the head's clamp(max = log 100).
    cap = math.log(100.0)
    scales = {}
    for name in ("prior_logit_scale", "joint_logit_scale"):
        value = min(float(state[name]), cap) - math.log(temperature)
        if value > cap:
            return False
        scales[name] = value
    for name, value in scales.items():
        state[name] = torch.tensor(value, dtype = state[name].dtype)
    last = max(int(key.split(".")[1]) for key in state if key.startswith("residual_scorer."))
    for kind in ("weight", "bias"):
        state[f"residual_scorer.{last}.{kind}"] = (
            state[f"residual_scorer.{last}.{kind}"] / temperature
        )
    return True


def _stamp_transformers_version(config_file: Path) -> None:
    # Unsloth's merged save writes config.json without transformers_version; transformers then
    # cannot rule out an old Mistral tokenizer and warns of an "incorrect regex pattern" on load.
    import transformers

    if not config_file.is_file():
        return
    config = json.loads(config_file.read_text(encoding = "utf-8"))
    if "transformers_version" not in config:
        config["transformers_version"] = transformers.__version__
        config_file.write_text(json.dumps(config, indent = 2) + "\n", encoding = "utf-8")


def _clef_head_weights(self, exact = False) -> tuple:
    # exact: a resumable trainer checkpoint, so float32 and no temperature folded in.
    config = {**self.decision_config, "fine_tuned": True}
    state = {k: v.detach().to("cpu", torch.float32) for k, v in self.head.state_dict().items()}
    if exact:
        return config, {k: v.contiguous() for k, v in state.items()}
    # Folded in, so Cloudflare's loader serves calibrated confidences too; the per-type
    # temperatures Unsloth applies are already relative to it.
    temperature = config.pop("head_temperature", None)
    if temperature and temperature != 1.0:
        if _fold_temperature(state, temperature):
            config["folded_temperature"] = config.get("folded_temperature", 1.0) * temperature
        else:
            config["head_temperature"] = temperature
    weights = {}
    for name, value in state.items():
        # The logit scales stay float32: a folded temperature moves them, and bfloat16's 8 bit
        # mantissa would turn that into a few percent on every logit.
        value = value.to(torch.float32 if value.dim() == 0 else torch.bfloat16).contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(
                f"Unsloth: head weight {name} is not finite, so the model cannot be saved."
            )
        weights[name] = value
    return config, weights


def _write_clef_head(self, staging: Path, config: dict, weights: dict) -> None:
    from safetensors.torch import save_file

    (staging / _DECISION_CONFIG).write_text(json.dumps(config, indent = 2), encoding = "utf-8")
    (staging / _CLEF_HEAD_FILES[1]).write_text(
        json.dumps(self.head.config, indent = 2), encoding = "utf-8"
    )
    save_file(weights, str(staging / _CLEF_HEAD_FILES[0]))


# Cloudflare's joint_schema_model.py, vendored unmodified.
_REFERENCE_CODE = Path(__file__).resolve().parents[1] / "_vendor" / "clef" / "joint_schema_model.py"
_MERGED_FILES = ("config.json", "model*.safetensors", "model.safetensors.index.json")
_ADAPTER_FILES = (_ADAPTER_CONFIG, "adapter_model.safetensors", "adapter_model.bin")


def _save_clef(
    self,
    save_directory,
    tokenizer,
    token = None,
) -> None:
    import shutil

    output = Path(save_directory)
    config, weights = _clef_head_weights(self)
    source = Path(getattr(self, "_unsloth_source_folder", "") or output)
    with _staging(output) as staging:
        encoder = self.encoder
        if hasattr(encoder, "save_pretrained_merged"):
            # Unsloth's merge dequantizes a 4-bit base and writes the processor files too.
            encoder.save_pretrained_merged(
                str(staging),
                tokenizer,
                save_method = "merged_16bit",
                **({} if token is None else {"token": token}),
            )
            # A merge with local_dir = the save folder leaves huggingface_hub's .cache behind.
            shutil.rmtree(staging / ".cache", ignore_errors = True)
        else:
            if hasattr(encoder, "merge_and_unload"):
                encoder = copy.deepcopy(encoder).merge_and_unload()
            encoder.save_pretrained(str(staging))
            tokenizer.save_pretrained(str(staging))
        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        # A tokenizer the caller grew (added tokens) is saved as it is.
        same_tokenizer = len(text_tokenizer) == getattr(self, "_unsloth_source_vocab", -1)
        for name in _CLEF_EXTRA_FILES:
            if not same_tokenizer and name.startswith("tokenizer"):
                continue
            if (source / name).is_file():
                shutil.copyfile(source / name, staging / name)
        # Cloudflare's loader is for Qwen3.5 vision backbones; ship it so the folder loads there too.
        architectures = getattr(self._backbone().config, "architectures", None) or []
        if (
            "Qwen3_5ForConditionalGeneration" in architectures
            and _REFERENCE_CODE.is_file()
            and not (staging / "joint_schema_model.py").exists()
        ):
            shutil.copyfile(_REFERENCE_CODE, staging / "joint_schema_model.py")
        _stamp_transformers_version(staging / "config.json")
        _write_clef_head(self, staging, config, weights)
        _commit_staged(
            staging, output, _CLEF_HEAD_FILES[0], stale = _MERGED_FILES[1:] + _ADAPTER_FILES
        )


def _save_clef_adapter(
    self,
    save_directory,
    tokenizer,
    exact = False,
) -> None:
    # Only the LoRA adapters, the head and the configs; from_pretrained puts them back on the base.
    output = Path(save_directory)
    config, weights = _clef_head_weights(self, exact)
    with _staging(output) as staging:
        self.encoder.save_pretrained(str(staging))
        adapter_config = staging / _ADAPTER_CONFIG
        if not adapter_config.is_file():
            raise RuntimeError(f"Unsloth: the adapters were not saved to {output}.")
        # Named as the user loaded it, not a local snapshot path or a pre-quantized mirror.
        if config.get("base_model"):
            adapter = json.loads(adapter_config.read_text(encoding = "utf-8"))
            adapter["base_model_name_or_path"] = config["base_model"]
            adapter_config.write_text(json.dumps(adapter, indent = 2), encoding = "utf-8")
        tokenizer.save_pretrained(str(staging))
        # A processor save can write a config.json of its own; this folder holds no merged weights.
        (staging / "config.json").unlink(missing_ok = True)
        _write_clef_head(self, staging, config, weights)
        _commit_staged(staging, output, _CLEF_HEAD_FILES[0], stale = _MERGED_FILES)


def _load_clef_checkpoint(model, folder: Path) -> None:
    # Resumes a DecisionTrainer checkpoint: the adapters and the float32 head saved by _save.
    from safetensors.torch import load_file

    if (
        not hasattr(model.encoder, "peft_config")
        or not (folder / "adapter_model.safetensors").is_file()
    ):
        raise NotImplementedError(
            "Unsloth: only LoRA Clef decision runs resume from a checkpoint; start a new run."
        )
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model.encoder, load_file(str(folder / "adapter_model.safetensors")))
    device = next(model.head.parameters()).device
    head = load_file(str(folder / _CLEF_HEAD_FILES[0]), device = str(device))
    model.head.load_state_dict({k: v.float() for k, v in head.items()}, strict = True)


def save_pretrained_clef(
    self,
    save_directory,
    tokenizer = None,
    **kwargs,
) -> None:
    # Adapters plus the head, as Unsloth's other models; a full finetune has none, so it saves merged.
    tokenizer = self._saved_temp_tokenizer if tokenizer is None else tokenizer
    if not hasattr(self.encoder, "peft_config"):
        return _save_clef(self, save_directory, tokenizer)
    return _save_clef_adapter(self, save_directory, tokenizer)


def push_to_hub_clef(
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


def _attach_clef_saving(model) -> None:
    model.save_pretrained = types.MethodType(save_pretrained_clef, model)
    model.save_pretrained_merged = types.MethodType(save_pretrained_merged, model)
    model.push_to_hub = types.MethodType(push_to_hub_clef, model)
    model.push_to_hub_merged = types.MethodType(push_to_hub_merged, model)
    _attach_gguf_saving(model)


def _attach_gguf_saving(model) -> None:
    from .decision_gguf import push_to_hub_gguf, save_pretrained_gguf
    model.save_pretrained_gguf = types.MethodType(save_pretrained_gguf, model)
    model.push_to_hub_gguf = types.MethodType(push_to_hub_gguf, model)


def _clef_mixed_precision(model, args) -> None:
    # Unsloth's rule for Qwen3.5 (rl.py): on its float32 path a model never autocasts, since
    # float16 NaNs the gated delta net; bfloat16 weights pair with bf16 only; an fp16 load (T4) keeps fp16.
    if _clef_forced_float32(model):
        if args.fp16 or args.bf16:
            print("Unsloth: Clef trains in float32 here, since Qwen3.5 cannot train in float16.")
        args.fp16 = args.bf16 = False
    elif args.fp16 and getattr(model._backbone(), "dtype", None) == torch.bfloat16:
        print("Unsloth: Clef is in bfloat16, so fp16 = True is switched to bf16 = True.")
        args.fp16, args.bf16 = False, True
    elif not args.fp16 and not args.bf16:
        # float32 norms (UNSLOTH_HIGH_PRECISION_LAYERNORM) beside 16-bit projections only run under autocast.
        amp = _clef_amp_dtype(model, next(model.parameters()).device)
        args.bf16 = amp == torch.bfloat16
        args.fp16 = amp == torch.float16
    precision = "bf16" if args.bf16 else "fp16" if args.fp16 else "no"
    # transformers 5 reads args.mixed_precision; 4.x reads this variable, set before the switches above.
    os.environ["ACCELERATE_MIXED_PRECISION"] = precision
    if hasattr(args, "mixed_precision"):
        args.mixed_precision = precision


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
        tokenizer = None,
        label_smoothing: Optional[float] = None,
        brier_weight: Optional[float] = None,
        ordinal_weight: Optional[float] = None,
        kl_weight: float = 0.0,
        permute_fields: Optional[bool] = None,
        **kwargs,
    ):
        # Laya never loads through FastLanguageModel / FastModel, which apply this patch: without it
        # transformers 5.0 - 5.5 divides accumulated gradients by gradient_accumulation_steps twice.
        from ._utils import patch_gradient_accumulation_fix

        patch_gradient_accumulation_fix(Trainer)
        args = copy.copy(args) if args is not None else TrainingArguments(output_dir = "tmp_trainer")
        recipe = CLEF_RECIPE if getattr(model, "is_clef", False) else {}
        if label_smoothing is None:
            label_smoothing = args.label_smoothing_factor or recipe.get("label_smoothing", 0.0)
        self.label_smoothing = float(label_smoothing)
        self.brier_weight = float(
            recipe.get("brier_weight", 0.0) if brier_weight is None else brier_weight
        )
        self.ordinal_weight = float(
            recipe.get("ordinal_weight", 0.0) if ordinal_weight is None else ordinal_weight
        )
        self.kl_weight = float(kl_weight)
        if permute_fields is None:
            permute_fields = recipe.get("permute_fields", False)
        # Opt in: a KL penalty to the starting model (adapters off, the head as loaded), against
        # forgetting what the base model knew outside the fine-tuning data.
        self._reference_head = None
        if self.kl_weight:
            if not getattr(model, "is_clef", False):
                raise NotImplementedError("Unsloth: kl_weight needs a Clef decision model.")
            # The reference is the backbone with its adapters off, which full finetuning does not have.
            if not hasattr(model.encoder, "disable_adapter"):
                raise NotImplementedError(
                    "Unsloth: kl_weight needs a LoRA Clef model, not full finetuning."
                )
            self._reference_head = copy.deepcopy(model.head).requires_grad_(False)
        args.remove_unused_columns = False
        # The model trains on one GPU: no DataParallel, and the batch stays per_device_train_batch_size.
        if args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:
            args._n_gpu = 1
        if args.label_names is None:
            args.label_names = ["target"]
        clef = getattr(model, "is_clef", False)
        if clef:
            _clef_mixed_precision(model, args)
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
            if clef:
                kwargs["data_collator"] = ClefDataCollator(
                    pad_token_id,
                    tokenizer = kwargs["processing_class"]
                    or getattr(model, "_saved_temp_tokenizer", None),
                    max_len = model.decision_config["max_len"],
                    permute_fields = permute_fields,
                    seed = args.seed,
                )
            else:
                kwargs["data_collator"] = DecisionDataCollator(pad_token_id)
        self.head_learning_rate = head_learning_rate
        super().__init__(model = model, args = args, **kwargs)
        backward = self.accelerator.backward

        def _backward(loss, **backward_kwargs):
            # Checkpointed layers rerun their forward here, so they need the forward's attention.
            with _no_cudnn_attention():
                return backward(loss, **backward_kwargs)

        self.accelerator.backward = _backward

    @contextlib.contextmanager
    def _dataset_field_order(self):
        collator = self.data_collator
        permute = getattr(collator, "permute_fields", False)
        if permute:
            collator.permute_fields = False
        try:
            yield
        finally:
            if permute:
                collator.permute_fields = True

    def evaluate(self, *args, **kwargs):
        with self._dataset_field_order():
            return super().evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        with self._dataset_field_order():
            return super().predict(*args, **kwargs)

    def train(self, *args, **kwargs):
        forwards = self.args.max_steps * self.args.gradient_accumulation_steps
        if forwards <= 0 and self.train_dataset is not None:
            batches = math.ceil(len(self.train_dataset) / self.args.train_batch_size)
            forwards = int(batches * self.args.num_train_epochs)
        amp_dtype = torch.bfloat16 if self.args.bf16 else torch.float16 if self.args.fp16 else None
        try:
            max_length = max(len(item["input_ids"]) for item in self.train_dataset)
        except (TypeError, KeyError, ValueError):
            max_length = None
        with compiled_encoder(self.model, forwards, amp_dtype, max_length):
            return super().train(*args, **kwargs)

    def _save(
        self,
        output_dir = None,
        state_dict = None,
    ):
        if not getattr(self.model, "is_clef", False):
            return super()._save(output_dir, state_dict)
        # Adapters plus the head, not one state dict holding the (shared-weight) backbone.
        output = Path(self.args.output_dir if output_dir is None else output_dir)
        tokenizer = self.processing_class or self.model._saved_temp_tokenizer
        if hasattr(self.model.encoder, "peft_config"):
            _save_clef_adapter(self.model, output, tokenizer, exact = True)
        else:
            _save_clef(self.model, output, tokenizer)
        torch.save(self.args, str(output / "training_args.bin"))

    def _load_from_checkpoint(
        self,
        resume_from_checkpoint,
        model = None,
    ):
        model = self.model if model is None else model
        if not getattr(model, "is_clef", False):
            return super()._load_from_checkpoint(resume_from_checkpoint, model)
        _load_clef_checkpoint(model, Path(resume_from_checkpoint))

    def _load_best_model(self):
        if not getattr(self.model, "is_clef", False):
            return super()._load_best_model()
        _load_clef_checkpoint(self.model, Path(self.state.best_model_checkpoint))

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
        ordinal = inputs.pop("ordinal", None)
        inputs = pad_length(model, inputs)
        with _no_cudnn_attention():
            logits, _ = model(**inputs)
        mask = inputs["marker_mask"]
        loss = _decision_loss(
            logits,
            target,
            mask,
            ordinal,
            self.label_smoothing,
            self.brier_weight,
            self.ordinal_weight,
        )
        if self._reference_head is not None:
            unwrapped = self.accelerator.unwrap_model(model)
            with torch.no_grad(), unwrapped.encoder.disable_adapter(), _no_cudnn_attention():
                reference, _ = unwrapped(**inputs, head = self._reference_head)
            log_ref = torch.log_softmax(reference.float().masked_fill(~mask, -1e4), -1)
            log_p = torch.log_softmax(logits.float().masked_fill(~mask, -1e4), -1)
            kl = (log_ref.exp() * (log_ref - log_p) * mask).sum(-1).mean()
            loss = loss + self.kl_weight * kl
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
    # Similar lengths share a batch, so little is padded; logits go back in the callers' order.
    order = sorted(range(len(items)), key = lambda i: len(items[i]["input_ids"]))
    out = [None] * len(items)
    for start in range(0, len(items), batch_size):
        indices = order[start : start + batch_size]
        chunk = [items[i] for i in indices]
        batch = collate(chunk)
        batch.pop("target")
        autocast = torch.autocast(device.type, dtype = amp_dtype, enabled = amp_dtype is not None)
        with autocast, _no_cudnn_attention():
            logits, _ = model(**{k: v.to(device) for k, v in batch.items()})
        logits = logits.float().cpu()
        for row, i in enumerate(indices):
            out[i] = logits[row, : len(items[i]["markers"])]
    model.train(was_training)
    return out


def _metrics(logits, items, temperatures) -> dict:
    import numpy as np

    conf, correct, loss, records = [], [], [], {}
    for z, item, temperature in zip(logits, items, temperatures):
        log_p = torch.log_softmax(z / temperature, -1)
        conf.append(float(log_p.exp().max()))
        correct.append(float(int(log_p.argmax()) == item["label"]))
        loss.append(float(-(torch.tensor(item["target"]) * log_p).sum()))
        row = item.get("row", ("item", len(correct)))
        records[row] = records.get(row, True) and bool(correct[-1])
    return {
        "accuracy": float(np.mean(correct)),
        "ece": _laya().common.ece_score(np.array(conf), np.array(correct)),
        "loss": float(np.mean(loss)),
        # Every question of a row right, the record-level precision Cloudflare rewards.
        "record_accuracy": float(np.mean(list(records.values()))),
    }


def _fit_temperature(
    logits,
    items,
    line_search = None,
) -> float:
    options = max(len(z) for z in logits)
    z = torch.full((len(logits), options), -1e4)
    target = torch.zeros((len(logits), options))
    for i, (row, item) in enumerate(zip(logits, items)):
        z[i, : len(row)] = row
        target[i, : len(row)] = torch.tensor(item["target"])
    log_t = torch.zeros(1, requires_grad = True)
    # Clef's hard-label fit passes "strong_wolfe": plain LBFGS can overshoot on peaked targets.
    optimizer = torch.optim.LBFGS([log_t], lr = 0.1, max_iter = 100, line_search_fn = line_search)

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
    # A Clef temperature not yet folded into the head (_save_clef folds it on save).
    head = config.get("head_temperature", 1.0)
    return [
        head * buckets.get(common.temp_bucket(item["qtype"], len(z)), per_type[item["qtype"]])
        for z, item in zip(logits, items)
    ]


HEAD_TEMPERATURE_RANGE = (0.05, 20.0)


def _calibrate_clef(config: dict, logits, items) -> dict:
    # Calibrated against being right (the gold label), not the soft gold distribution: Clef's
    # confidence is read as the chance the answer is correct, and soft gold targets left a tuned
    # model underconfident (confidence 0.61 at accuracy 0.78 on typed-decisions).
    common = _laya().common
    hard = [
        {**item, "target": [float(j == item["label"]) for j in range(len(z))]}
        for z, item in zip(logits, items)
    ]

    def fit(indices) -> tuple:
        chosen = list(indices)
        head = _fit_temperature(
            [logits[i] for i in chosen], [hard[i] for i in chosen], line_search = "strong_wolfe"
        )
        head = min(max(head, HEAD_TEMPERATURE_RANGE[0]), HEAD_TEMPERATURE_RANGE[1])
        scaled = [z / head for z in logits]
        relative, fitted = _fit_temperatures(scaled, hard, chosen, [1.0] * 3)
        return head, relative, fitted

    everything = range(len(items))
    if len(items) < MIN_CALIBRATION_ITEMS:
        return {
            **_metrics(logits, items, _served_temperatures(config, logits, items)),
            "fitted_types": [],
        }
    head, relative, fitted = fit(everything)
    half = {row: i % 2 for i, row in enumerate(sorted({item["row"] for item in items}))}
    per_item = [head * common.clamp_temperature(relative[item["qtype"]]) for item in items]
    for side in (0, 1) if len(half) > 1 else ():
        side_head, side_relative, _ = fit(i for i in everything if half[items[i]["row"]] != side)
        for i in everything:
            if half[items[i]["row"]] == side:
                per_item[i] = side_head * common.clamp_temperature(side_relative[items[i]["qtype"]])
    config["head_temperature"] = head
    config["temperature"] = relative
    config.pop("temperature_by_options", None)
    return {**_metrics(logits, items, per_item), "fitted_types": sorted(fitted)}


def _served_lengths(
    config: dict,
    positions: int,
    max_seq_length = None,
) -> tuple:
    """(max_len, head_max_len) a Laya checkpoint serves with once loaded."""
    wanted = max_seq_length or max(int(config.get("max_len", 512)), TRAIN_MAX_LEN)
    max_len = min(int(positions), int(wanted))
    return max_len, min(max_len // 2, max(int(config.get("head_max_len", 192)), TRAIN_HEAD_MAX_LEN))


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
        return _save_clef(self, save_directory, tokenizer, kwargs.get("token"))
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
    encoder_config = self._unsloth_encoder_config
    vocab = encoder.get_input_embeddings().num_embeddings
    if json.loads(encoder_config).get("vocab_size") != vocab:
        # A caller-grown tokenizer resized the embedding; the source config keeps its bytes otherwise.
        encoder_config = json.dumps({**json.loads(encoder_config), "vocab_size": vocab}, indent = 2)
    del encoder, state
    config = json.dumps({**self.decision_config, "fine_tuned": True}, indent = 2)

    with _staging(Path(save_directory)) as staging:
        save_file(weights, str(staging / "model.safetensors"))
        (staging / "encoder").mkdir()
        (staging / "encoder" / "config.json").write_text(encoder_config, encoding = "utf-8")
        tokenizer.save_pretrained(str(staging / "tokenizer"))
        _laya().agent._fix_tokenizer_config(str(staging))
        (staging / "rl_agent_config.json").write_text(config, encoding = "utf-8")
        # A folder with rl_agent_config.json is a complete checkpoint, so it moves in last.
        _commit_staged(staging, Path(save_directory), "rl_agent_config.json")


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
        if kwargs.get("decision_head") is None and _is_plain_lm(
            model_name, subfolder, token, revision, local_files_only
        ):
            kwargs["decision_head"] = "clef"
        if kwargs.get("decision_head") is not None:
            # A plain language model plus a fresh (or given) decision head: see decision_from_lm.py.
            from .decision_from_lm import load_lm_as_decision_model
            return load_lm_as_decision_model(
                model_name,
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
                full_finetuning = full_finetuning,
                token = token,
                revision = revision,
                local_files_only = local_files_only,
                use_gradient_checkpointing = use_gradient_checkpointing,
                random_state = random_state,
                **kwargs,
            )
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
                model_name = model_name,
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
        config["max_len"], config["head_max_len"] = _served_lengths(
            config, positions, max_seq_length
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
        _attach_gguf_saving(model)
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
            unsupported = {
                "use_dora": use_dora,
                "layers_to_transform": layers_to_transform,
                "layers_pattern": layers_pattern,
                "loftq_config": loftq_config,
                "init_lora_weights": init_lora_weights is not True,
            }
            unsupported = [name for name, value in unsupported.items() if value]
            if unsupported:
                raise NotImplementedError(
                    f"Unsloth: Clef LoRA does not support {', '.join(unsupported)} yet."
                )
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
        _lean_lora(model.encoder)
        _gradient_checkpointing(model, use_gradient_checkpointing)
        return model

    @staticmethod
    def freeze_backbone(model):
        # Head-only warm-up for a fresh decision head; undo with unfreeze_backbone.
        from .decision_from_lm import freeze_backbone
        return freeze_backbone(model)

    @staticmethod
    def unfreeze_backbone(model):
        from .decision_from_lm import unfreeze_backbone
        return unfreeze_backbone(model)

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
        items, report, skips = [], {"total": 0, "skipped": 0, "reason": None, "truncated": 0}, {}

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
        # Over max_seq_length the end of the state is cut, never questions or options.
        report["truncated"] = sum(len(item["input_ids"]) >= max_len for item in items)
        if report["truncated"]:
            print(
                f"Unsloth: {report['truncated']:,} of {len(items):,} training inputs are longer than "
                f"max_seq_length = {max_len}, so the end of their state is cut. Raise "
                "max_seq_length to train on all of it."
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
    def predict(model, tokenizer, state, questions: dict) -> dict:
        """Answers one Decision API request: {name: answer} at the calibrated temperatures.
        Each answer is the Decision API's (choice / confidence, score / legend, or noul) plus
        "answer" (the option, True / False for noul, the level number for score) and
        "probabilities" over every option. Works for Laya, Clef and converted language models."""
        if not isinstance(questions, dict) or not questions:
            raise DecisionDataError("questions must be a non-empty dict of name to question")
        state = _parsed(state)
        if getattr(model, "is_clef", False):
            questions = {str(name): _clef_question(q) for name, q in questions.items()}
            # Read up to CLEF_SERVE_MAX_LEN tokens, like serving, even past the training cut.
            max_length = max(
                int(model.decision_config.get("max_len", CLEF_MAX_LEN)), CLEF_SERVE_MAX_LEN
            )
            return _clef_decide(
                model, tokenizer, state, questions, max_length = max_length, predicted = True
            )["answers"]
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        return _laya_decide(model, tokenizer, state, questions, predicted = True)["answers"]

    @staticmethod
    def evaluate(
        model,
        tokenizer,
        items: list,
        batch_size = None,
    ) -> dict:
        logits, items = _decision_logits(model, tokenizer, items, batch_size)
        return _metrics(logits, items, _served_temperatures(model.decision_config, logits, items))

    @staticmethod
    def calibrate(
        model,
        tokenizer,
        items: list,
        batch_size = None,
    ) -> dict:
        common = _laya().common
        config = model.decision_config
        fallback = [common.clamp_temperature(t) for t in config.get("temperature", [1.0] * 3)]
        logits, items = _decision_logits(model, tokenizer, items, batch_size)
        clef = getattr(model, "is_clef", False)
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
