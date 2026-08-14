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

import re
from typing import Optional

from .protocols import RecognizedFailure

_TRACEBACK = "Traceback (most recent call last)"
_EXIT = re.compile(r"(?:exit(?:ed)?(?: with)? code|returncode|exit_code)\s*[:=]?\s*(-?\d+)", re.I)


def inspect_tool_result(name: str, result: str, *, contact: str = "world") -> Optional[RecognizedFailure]:
    text = result or ""
    if _TRACEBACK in text:
        return RecognizedFailure(summary=f"{name} raised", source=contact)
    if text.startswith("Error:") or "\nError:" in text:
        first = text.strip().splitlines()[0][:200]
        return RecognizedFailure(summary=first, source=contact)
    match = _EXIT.search(text)
    if match and match.group(1) not in {"0"}:
        return RecognizedFailure(
            summary=f"{name} exited {match.group(1)}",
            source=contact,
        )
    lowered = text.lower()
    if "command failed" in lowered or "returned non-zero" in lowered:
        return RecognizedFailure(summary=f"{name} failed", source=contact)
    return None
