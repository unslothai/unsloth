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

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

from unforgettable.host import GenerateRequest, GenerateResult
from unforgettable.loop.context import EpisodeRequest
from unforgettable.loop.episode import run


class _ForwardHost:
    """Minimal host that fires ``on_chunk`` when episode.run passes it through."""

    def __init__(
        self,
        root: Path,
        fires: int = 1,
    ):
        self.db = root / "memory.db"
        self.world = root / "world"
        self.world.mkdir()
        self.fires = fires
        self.seen_on_chunk = None
        self.generate_calls = 0

    def memory_db_path(self) -> Path:
        return self.db

    def world_session_id(self, request) -> str:
        return "world"

    def create_sim_session(self, episode_id: str) -> str:
        return f"sim-{episode_id}"

    def sandbox_path(self, session_id: str) -> Path:
        return self.world

    def remove_sim_session(self, session_id: str) -> None:
        return None

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        self.generate_calls += 1
        self.seen_on_chunk = req.on_chunk
        pieces = []
        if req.on_chunk is not None:
            for i in range(self.fires):
                raw = f'data: {{"choices":[{{"delta":{{"content":"tok{i}"}}}}]}}\n\n'.encode(
                    "utf-8"
                )
                pieces.append(f"tok{i}")
                maybe = req.on_chunk(raw)
                if inspect.isawaitable(maybe):
                    await maybe
        text = "".join(pieces) or "ok"
        return GenerateResult(text = text)


def test_episode_run_passes_on_chunk_through(tmp_path: Path):
    seen: list[bytes] = []

    async def on_chunk(data: bytes) -> None:
        seen.append(data)

    host = _ForwardHost(tmp_path, fires = 2)
    outcome = asyncio.run(
        run(
            host,
            EpisodeRequest(
                messages = [{"role": "user", "content": "hi"}],
                on_chunk = on_chunk,
            ),
        )
    )
    assert host.generate_calls == 1
    assert host.seen_on_chunk is on_chunk
    assert len(seen) == 2
    assert seen[0].startswith(b"data: ")
    assert b"tok0" in seen[0]
    assert b"tok1" in seen[1]
    assert outcome.text == "tok0tok1"


def test_episode_run_without_on_chunk_still_generates(tmp_path: Path):
    host = _ForwardHost(tmp_path, fires = 3)
    outcome = asyncio.run(
        run(
            host,
            EpisodeRequest(messages = [{"role": "user", "content": "hi"}]),
        )
    )
    assert host.seen_on_chunk is None
    assert outcome.text == "ok"
