# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TrainBackend protocol, fake trainer, and pack → shadow adapter."""

from __future__ import annotations

import importlib.util
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from unforgettable.sidecar.adapters import STATUS_SHADOW, insert_adapter
from unforgettable.sidecar.format import preference_pairs
from unforgettable.sidecar.pack import (
    PACK_MIN_TRAIN,
    ROLE_TRAIN,
    get_pack,
    list_pack_items,
)
from unforgettable.store.db import default_db_path

FAKE_BASE_MODEL = "fake"
FULL_FINETUNE_REFUSED = "sidecar refuses full fine-tune; unset UNSLOTH_ENABLE_FULL_FINETUNING"
PREFERENCE_NEEDS_DPO = "preference recipe needs trl.DPOTrainer"
NO_PREFERENCE_PAIRS = "no preference pairs (need a world pass and an admitted error_fix)"
RECIPE_PREFERENCE = "preference"
DPO_BETA = 0.1


@dataclass(frozen = True)
class TrainResult:
    adapter_id: str
    path: str
    backend: str
    recipe: str
    n_examples: int
    gguf_path: Optional[str] = None


class TrainBackend(Protocol):
    def train(
        self,
        examples: list[dict],
        *,
        output_dir: Path,
        base_model: str,
        recipe: str = "sft",
    ) -> None: ...

    def complete(
        self,
        messages: list[dict],
        *,
        adapter_path: Optional[str],
        max_tokens: int = 80,
    ) -> str: ...


def adapters_root(db_path = None) -> Path:
    path = Path(db_path) if db_path is not None else default_db_path()
    return path.resolve().parent / "adapters"


def _example_messages(example: Any) -> list[dict[str, Any]]:
    if isinstance(example, dict):
        messages = example.get("messages")
        if isinstance(messages, str):
            messages = json.loads(messages)
        return list(messages or [])
    if isinstance(example, list):
        return list(example)
    return []


def _user_assistant(example: Any) -> tuple[str, str]:
    user = ""
    assistant = ""
    for msg in _example_messages(example):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "user":
            user = content
        elif role == "assistant":
            assistant = content
    return user, assistant


def _refuse_full_finetune() -> None:
    if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING") == "1":
        raise RuntimeError(FULL_FINETUNE_REFUSED)


def _vocab_token(tokenizer: Any, attr: str) -> Optional[str]:
    token = getattr(tokenizer, attr, None)
    if not isinstance(token, str) or not token:
        return None
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert) and convert(token) is None:
        return None
    return token


def _sft_text(example: Any, tokenizer: Any) -> str:
    messages = _example_messages(example)
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply) and messages:
        try:
            return apply(messages, tokenize = False, add_generation_prompt = False)
        except (TypeError, ValueError):
            pass
    user, assistant = _user_assistant(example)
    return f"user\n{user}\nassistant\n{assistant}"


def _prompt_text(messages: list[dict], tokenizer: Any) -> str:
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply) and messages:
        try:
            return apply(messages, tokenize = False, add_generation_prompt = True)
        except (TypeError, ValueError):
            pass
    user, assistant = _user_assistant(messages)
    if assistant:
        return f"user\n{user}\nassistant\n{assistant}"
    return f"user\n{user}\nassistant\n"


def _preference_gold(examples: list[dict]) -> dict[str, str]:
    gold: dict[str, str] = {}
    for example in examples:
        if not isinstance(example, dict):
            continue
        user = _pair_prompt_text(example)
        if user:
            gold[user] = _pair_completion_text(example.get("chosen"))
    return gold


def _message_text(value: Any, *, role: Optional[str] = None) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    text = ""
    for msg in value:
        if not isinstance(msg, dict):
            continue
        if role is not None and msg.get("role") != role:
            continue
        content = msg.get("content") or ""
        if content:
            text = content
    return text


def _pair_prompt_text(example: dict) -> str:
    prompt = example.get("prompt")
    if isinstance(prompt, str):
        return prompt
    return _message_text(prompt, role = "user")


def _pair_completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _message_text(value)


def _dpo_rows(examples: list[dict]) -> list[dict[str, str]]:
    """TRL DPO wants string prompt / chosen / rejected columns."""
    rows: list[dict[str, str]] = []
    for example in examples:
        if not isinstance(example, dict):
            continue
        prompt = _pair_prompt_text(example)
        chosen = _pair_completion_text(example.get("chosen"))
        rejected = _pair_completion_text(example.get("rejected"))
        if prompt and chosen and rejected:
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return rows


def _write_pairs_jsonl(dest: Path, examples: list[dict]) -> None:
    dest.mkdir(parents = True, exist_ok = True)
    (dest / "pairs.jsonl").write_text(
        "".join(json.dumps(example) + "\n" for example in examples),
        encoding = "utf-8",
    )


def _require_preference_trl() -> None:
    if importlib.util.find_spec("trl") is None:
        raise RuntimeError(PREFERENCE_NEEDS_DPO)


def _new_peft_model(base_model: str):
    loader = _unsloth_loader()
    model, tokenizer = _from_pretrained(loader, base_model)
    model = loader.get_peft_model(
        model,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0.0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    return model, tokenizer


def _import_dpo():
    # Unsloth first so DPOTrainer / DPOConfig are the patched classes.
    _unsloth_loader()
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError(PREFERENCE_NEEDS_DPO) from exc
    return DPOTrainer, DPOConfig


def _dataset_from_list(rows: list[dict]) -> Any:
    from datasets import Dataset
    return Dataset.from_list(rows)


def _dpo_config(DPOConfig, dest: Path, n_rows: int):
    batch = 2 if n_rows >= 2 else 1
    shared = {
        "output_dir": str(dest),
        "per_device_train_batch_size": batch,
        "num_train_epochs": 1,
        "logging_steps": 1,
        "seed": 3407,
    }
    attempts = (
        {**shared, "report_to": [], "beta": DPO_BETA},
        {**shared, "report_to": "none", "beta": DPO_BETA},
        {**shared, "report_to": "none"},
        shared,
    )
    last_error: Optional[TypeError] = None
    for kwargs in attempts:
        try:
            return DPOConfig(**kwargs)
        except TypeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _dpo_trainer(
    DPOTrainer,
    model,
    tokenizer,
    args,
    dataset,
    extra = None,
):
    extras = (dict(extra or {}), {})
    attempts = (
        {"ref_model": None, "processing_class": tokenizer},
        {"ref_model": None, "tokenizer": tokenizer},
        {"processing_class": tokenizer},
        {"tokenizer": tokenizer},
    )
    last_error: Optional[TypeError] = None
    for payload in extras:
        for kw in attempts:
            try:
                return DPOTrainer(model = model, args = args, train_dataset = dataset, **kw, **payload)
            except TypeError as exc:
                last_error = exc
    assert last_error is not None
    raise last_error


class FakeTrainBackend:
    name = "fake"

    def train(
        self,
        examples: list[dict],
        *,
        output_dir: Path,
        base_model: str,
        recipe: str = "sft",
    ) -> None:
        del base_model
        dest = Path(output_dir)
        dest.mkdir(parents = True, exist_ok = True)
        if recipe == RECIPE_PREFERENCE:
            _write_pairs_jsonl(dest, examples)
            gold = _preference_gold(examples)
        else:
            gold = {}
            for example in examples:
                user, assistant = _user_assistant(example)
                if user:
                    gold[user] = assistant
        (dest / "adapter_config.json").write_text(
            json.dumps({"fake": True, "recipe": recipe, "n": len(examples)}),
            encoding = "utf-8",
        )
        (dest / "fake_gold.json").write_text(json.dumps(gold), encoding = "utf-8")

    def complete(
        self,
        messages: list[dict],
        *,
        adapter_path: Optional[str],
        max_tokens: int = 80,
    ) -> str:
        del max_tokens
        if not adapter_path:
            return ""
        gold_path = Path(adapter_path) / "fake_gold.json"
        if not gold_path.is_file():
            return ""
        gold = json.loads(gold_path.read_text(encoding = "utf-8"))
        user = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user = msg.get("content") or ""
        return gold.get(user, "")


def _unsloth_loader():
    try:
        from unsloth import FastModel
        return FastModel
    except ImportError:
        from unsloth import FastLanguageModel
        return FastLanguageModel


def _from_pretrained(loader, base_model: str):
    kwargs = {
        "model_name": base_model,
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "full_finetuning": False,
    }
    try:
        return loader.from_pretrained(text_only = True, **kwargs)
    except TypeError:
        return loader.from_pretrained(**kwargs)


def _model_device(model: Any):
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return None


class UnslothTrainBackend:
    name = "unsloth"

    def __init__(self, base_model: Optional[str] = None) -> None:
        self._base_model = base_model

    def train(
        self,
        examples: list[dict],
        *,
        output_dir: Path,
        base_model: str,
        recipe: str = "sft",
    ) -> None:
        _refuse_full_finetune()
        self._base_model = base_model
        dest = Path(output_dir)
        if recipe == RECIPE_PREFERENCE:
            self._train_preference(examples, dest)
            return
        self._train_sft(examples, dest, base_model)

    def _train_preference(self, examples: list[dict], dest: Path) -> None:
        _require_preference_trl()
        rows = _dpo_rows(examples)
        if not rows:
            raise ValueError(NO_PREFERENCE_PAIRS)
        dest.mkdir(parents = True, exist_ok = True)
        _write_pairs_jsonl(dest, examples)
        DPOTrainer, DPOConfig = _import_dpo()
        model, tokenizer = _new_peft_model(self._base_model or "")
        ds = _dataset_from_list(rows)
        dpo_args = _dpo_config(DPOConfig, dest, len(rows))
        extra = {} if hasattr(dpo_args, "beta") else {"beta": DPO_BETA}
        trainer = _dpo_trainer(DPOTrainer, model, tokenizer, dpo_args, ds, extra)
        trainer.train()
        model.save_pretrained(dest)
        tokenizer.save_pretrained(dest)

    def _train_sft(self, examples: list[dict], dest: Path, base_model: str) -> None:
        from datasets import Dataset

        dest.mkdir(parents = True, exist_ok = True)
        # Import TRL after Unsloth so SFTConfig is the patched class TRL's
        # isinstance check expects. A pre-import instance is rebuilt and
        # picks up Unsloth's '<EOS_TOKEN>' sentinel.
        model, tokenizer = _new_peft_model(base_model)
        from trl import SFTConfig, SFTTrainer

        eos = _vocab_token(tokenizer, "eos_token")
        pad = _vocab_token(tokenizer, "pad_token") or eos
        ds = Dataset.from_dict({"text": [_sft_text(example, tokenizer) for example in examples]})
        sft_kwargs = {
            "output_dir": str(dest),
            "per_device_train_batch_size": 2,
            "num_train_epochs": 1,
            "logging_steps": 1,
            "seed": 3407,
            "report_to": [],
        }
        if eos:
            sft_kwargs["eos_token"] = eos
        if pad:
            sft_kwargs["pad_token"] = pad
        sft_args = SFTConfig(**sft_kwargs)
        try:
            trainer = SFTTrainer(
                model = model,
                processing_class = tokenizer,
                train_dataset = ds,
                args = sft_args,
            )
        except TypeError:
            trainer = SFTTrainer(
                model = model,
                tokenizer = tokenizer,
                train_dataset = ds,
                args = sft_args,
            )
        trainer.train()
        model.save_pretrained(dest)
        tokenizer.save_pretrained(dest)

    def complete(
        self,
        messages: list[dict],
        *,
        adapter_path: Optional[str],
        max_tokens: int = 80,
    ) -> str:
        _refuse_full_finetune()
        loader = _unsloth_loader()

        base = getattr(self, "_base_model", None)
        if not base:
            return ""
        model, tokenizer = _from_pretrained(loader, base)
        if adapter_path:
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_path)
            except Exception:
                load_adapter = getattr(model, "load_adapter", None)
                if not callable(load_adapter):
                    raise
                load_adapter(adapter_path)
        loader.for_inference(model)
        inputs = tokenizer(_prompt_text(messages, tokenizer), return_tensors = "pt")
        device = _model_device(model)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)
        outputs = model.generate(**inputs, max_new_tokens = max_tokens)
        prompt_len = int(inputs["input_ids"].shape[-1])
        return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens = True)


def _backend_name(backend: TrainBackend) -> str:
    name = getattr(backend, "name", None)
    if isinstance(name, str) and name:
        return name
    return type(backend).__name__


def train_pack(
    pack_id: str,
    *,
    backend: TrainBackend,
    base_model: str,
    recipe: str = "sft",
    db_path = None,
    export_gguf: bool = True,
) -> TrainResult:
    items = list_pack_items(pack_id, db_path = db_path)
    if not items and get_pack(pack_id, db_path = db_path) is None:
        raise KeyError(pack_id)
    train_items = [item for item in items if item.get("role") == ROLE_TRAIN]
    if len(train_items) < PACK_MIN_TRAIN:
        raise ValueError(f"need at least {PACK_MIN_TRAIN} train items, got {len(train_items)}")
    if recipe == RECIPE_PREFERENCE:
        train_eps = {item.get("episode_id") for item in train_items if item.get("episode_id")}
        examples = preference_pairs(db_path = db_path, train_episode_ids = train_eps)
        if not examples:
            raise ValueError(NO_PREFERENCE_PAIRS)
    else:
        examples = []
        for item in train_items:
            messages = item.get("messages")
            if isinstance(messages, str):
                messages = json.loads(messages)
            examples.append({"messages": messages or []})
    adapter_id = str(uuid.uuid4())
    output_dir = adapters_root(db_path) / adapter_id
    backend.train(examples, output_dir = output_dir, base_model = base_model, recipe = recipe)
    path = str(output_dir)
    gguf_path = None
    if export_gguf and _backend_name(backend) == "unsloth":
        try:
            from unforgettable.sidecar.export_gguf import export_adapter_gguf
            gguf_path = export_adapter_gguf(output_dir, base_model = base_model)
        except Exception:
            gguf_path = None
    insert_adapter(
        adapter_id = adapter_id,
        pack_id = pack_id,
        status = STATUS_SHADOW,
        backend = _backend_name(backend),
        base_model = base_model,
        recipe = recipe,
        path = path,
        gguf_path = gguf_path,
        db_path = db_path,
    )
    return TrainResult(
        adapter_id = adapter_id,
        path = path,
        backend = _backend_name(backend),
        recipe = recipe,
        n_examples = len(examples),
        gguf_path = gguf_path,
    )
