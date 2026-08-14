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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from unforgettable.sidecar.adapters import STATUS_SHADOW, insert_adapter
from unforgettable.sidecar.pack import PACK_MIN_TRAIN, ROLE_TRAIN, list_pack_items
from unforgettable.store.db import default_db_path

FAKE_BASE_MODEL = "fake"
UNSLOTH_NOT_IMPLEMENTED = "UnslothTrainBackend is not implemented"


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
        raise NotImplementedError(UNSLOTH_NOT_IMPLEMENTED)

    def complete(
        self,
        messages: list[dict],
        *,
        adapter_path: Optional[str],
        max_tokens: int = 80,
    ) -> str:
        raise NotImplementedError(UNSLOTH_NOT_IMPLEMENTED)


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
