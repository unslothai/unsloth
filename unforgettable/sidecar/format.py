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

"""SFT message JSON for pack items. Gold is the admitted record body only."""

from __future__ import annotations

from typing import Any

PACK_BODY_CHARS = 1200


def _clip(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit]


def format_sft_item(rec: dict[str, Any]) -> list[dict[str, str]]:
    title = (rec.get("title") or "").strip()
    body = _clip((rec.get("body") or "").strip(), PACK_BODY_CHARS)
    return [
        {"role": "user", "content": title},
        {"role": "assistant", "content": body},
    ]
