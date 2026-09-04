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

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen = True)
class RecognizedFailure:
    summary: str
    source: str  # world | sim | tool


class WorldEyes(Protocol):
    def grade(self, name: str, result: str) -> Optional[RecognizedFailure]: ...


class SimEyes(Protocol):
    def grade(self, name: str, result: str) -> Optional[RecognizedFailure]: ...


@dataclass(frozen = True)
class Contradiction:
    title_key: str
    record_ids: tuple[str, ...]
    reason: str


class GateEyes(Protocol):
    def note(self, message: str) -> None: ...

    def contradictions(self, db_path = None) -> list[Contradiction]: ...

    def review_write(
        self,
        *,
        kind: str,
        title: str,
        body: str,
        provenance: str,
        db_path = None,
        speaker: str | None = None,
        warrant: str | None = None,
    ) -> str:
        """Return '' or a reason to force proposed."""
        ...
