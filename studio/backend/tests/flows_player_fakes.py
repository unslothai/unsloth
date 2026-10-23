# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A scripted NodeContext for the story/world handler tests: no Studio, no models."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from core.flows.registry import FlowCancelled
from core.flows.specs import SPECS_BY_TYPE


class FakeCtx:
    def __init__(
        self,
        node_type: str,
        *,
        params: Optional[dict] = None,
        inputs: Optional[dict] = None,
        seed: int = 7,
        llm: Optional[Callable[..., Any]] = None,
    ):
        self.run_id = "run"
        self.flow_id = "flow"
        self.node_id = "n1"
        self.spec = SPECS_BY_TYPE[node_type]
        self.params = {p["id"]: p["default"] for p in self.spec["params"]}
        self.params.update(params or {})
        self.inputs = inputs or {}
        self.seed = seed
        self.flow_params: dict = {}
        self.llm = llm
        self.llm_calls: list[dict] = []
        self.images: list[dict] = []
        self.cutouts: list[dict] = []
        self.spoken: list[dict] = []
        self.progress_events: list[tuple] = []
        self.logs: list[tuple[str, str]] = []
        self._cancelled = False

    async def llm_json(self, **kw: Any) -> Any:
        self.llm_calls.append(kw)
        return self.llm(**kw) if self.llm else {}

    async def llm_text(self, **kw: Any) -> str:
        self.llm_calls.append(kw)
        return str(self.llm(**kw)) if self.llm else ""

    async def generate_image(self, **kw: Any) -> list[dict]:
        self.images.append(kw)
        n = len(self.images)
        return [
            {
                "kind": "image",
                "ref": f"image:img{n}",
                "url": f"/img/{n}.png",
                "width": kw.get("width"),
                "height": kw.get("height"),
                "seed": kw.get("seed"),
                "prompt": kw.get("prompt"),
            }
        ]

    async def remove_background(
        self,
        image: dict,
        subject: str,
        trim: bool = True,
    ) -> tuple[dict, dict]:
        self.cutouts.append({"image": image, "subject": subject})
        return (
            {**image, "ref": image["ref"] + "-cut", "mime": "image/png"},
            {**image, "ref": image["ref"] + "-mask"},
        )

    async def speak(self, **kw: Any) -> dict:
        self.spoken.append(kw)
        return {"kind": "audio", "ref": f"audio:a{len(self.spoken)}"}

    def progress(
        self,
        fraction = None,
        **kw: Any,
    ) -> None:
        self.progress_events.append((fraction, kw))

    def log(
        self,
        message: str,
        level: str = "info",
    ) -> None:
        self.logs.append((level, message))

    def text_delta(self, port: str, delta: str) -> None:
        pass

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def check_cancelled(self) -> None:
        if self._cancelled:
            raise FlowCancelled()

    def on_cancel(self, fn) -> None:
        pass


def run(coro):
    return asyncio.run(coro)
