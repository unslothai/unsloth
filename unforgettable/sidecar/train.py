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

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from unforgettable.sidecar.adapters import STATUS_SHADOW, insert_adapter
from unforgettable.sidecar.pack import (
    PACK_MIN_TRAIN,
    ROLE_TRAIN,
    get_pack,
    list_pack_items,
)
from unforgettable.store.db import default_db_path

FAKE_BASE_MODEL = "fake"
FULL_FINETUNE_REFUSED = (
    "sidecar refuses full fine-tune; unset UNSLOTH_ENABLE_FULL_FINETUNING"
)
PREFERENCE_NEEDS_DPO = "preference recipe needs trl.DPOTrainer"


@dataclass(frozen=True)
class TrainResult:
    adapter_id: str
    path: str
    backend: str
    recipe: str
    n_examples: int


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


def adapters_root(db_path=None) -> Path:
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


def _sft_text(example: Any, tokenizer: Any) -> str:
    messages = _example_messages(example)
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply) and messages:
        try:
            return apply(messages, tokenize=False, add_generation_prompt=False)
        except (TypeError, ValueError):
            pass
    user, assistant = _user_assistant(example)
    return f"user\n{user}\nassistant\n{assistant}"


def _prompt_text(messages: list[dict], tokenizer: Any) -> str:
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply) and messages:
        try:
            return apply(messages, tokenize=False, add_generation_prompt=True)
        except (TypeError, ValueError):
            pass
    user, assistant = _user_assistant(messages)
    if assistant:
        return f"user\n{user}\nassistant\n{assistant}"
    return f"user\n{user}\nassistant\n"


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
        dest.mkdir(parents=True, exist_ok=True)
        gold: dict[str, str] = {}
        for example in examples:
            user, assistant = _user_assistant(example)
            if user:
                gold[user] = assistant
        (dest / "adapter_config.json").write_text(
            json.dumps({"fake": True, "recipe": recipe, "n": len(examples)}),
            encoding="utf-8",
        )
        (dest / "fake_gold.json").write_text(json.dumps(gold), encoding="utf-8")

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
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        user = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user = msg.get("content") or ""
        return gold.get(user, "")


class UnslothTrainBackend:
    name = "unsloth"

    def train(
        self,
        examples: list[dict],
        *,
        output_dir: Path,
        base_model: str,
        recipe: str = "sft",
    ) -> None:
        _refuse_full_finetune()
        if recipe == "preference":
            raise RuntimeError(PREFERENCE_NEEDS_DPO)
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset

        self._base_model = base_model
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            base_model,
            max_seq_length=2048,
            load_in_4bit=True,
            full_finetuning=False,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        ds = Dataset.from_dict(
            {"text": [_sft_text(example, tokenizer) for example in examples]}
        )
        sft_args = SFTConfig(
            output_dir=str(dest),
            per_device_train_batch_size=2,
            num_train_epochs=1,
            logging_steps=1,
            seed=3407,
            report_to=[],
        )
        try:
            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=ds,
                args=sft_args,
            )
        except TypeError:
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=ds,
                args=sft_args,
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
        from unsloth import FastLanguageModel

        load_name = adapter_path or getattr(self, "_base_model", None)
        if not load_name:
            return ""
        model, tokenizer = FastLanguageModel.from_pretrained(
            load_name,
            max_seq_length=2048,
            load_in_4bit=True,
            full_finetuning=False,
        )
        FastLanguageModel.for_inference(model)
        inputs = tokenizer(_prompt_text(messages, tokenizer), return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        prompt_len = int(inputs["input_ids"].shape[-1])
        return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)


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
    db_path=None,
) -> TrainResult:
    items = list_pack_items(pack_id, db_path=db_path)
    if not items and get_pack(pack_id, db_path=db_path) is None:
        raise KeyError(pack_id)
    train_items = [item for item in items if item.get("role") == ROLE_TRAIN]
    if len(train_items) < PACK_MIN_TRAIN:
        raise ValueError(
            f"need at least {PACK_MIN_TRAIN} train items, got {len(train_items)}"
        )
    examples = []
    for item in train_items:
        messages = item.get("messages")
        if isinstance(messages, str):
            messages = json.loads(messages)
        examples.append({"messages": messages or []})
    adapter_id = str(uuid.uuid4())
    output_dir = adapters_root(db_path) / adapter_id
    backend.train(
        examples, output_dir=output_dir, base_model=base_model, recipe=recipe
    )
    path = str(output_dir)
    insert_adapter(
        adapter_id=adapter_id,
        pack_id=pack_id,
        status=STATUS_SHADOW,
        backend=_backend_name(backend),
        base_model=base_model,
        recipe=recipe,
        path=path,
        db_path=db_path,
    )
    return TrainResult(
        adapter_id=adapter_id,
        path=path,
        backend=_backend_name(backend),
        recipe=recipe,
        n_examples=len(examples),
    )
