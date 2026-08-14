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

from datetime import datetime, timezone
from typing import Any, Optional

from unforgettable.store.search import search_records


def retrieve(query: str, *, top_k: int = 6, db_path=None) -> list[dict[str, Any]]:
    if not (query or "").strip():
        return []
    return search_records(query, top_k=top_k, db_path=db_path)


def _age_note(updated_at: Optional[str]) -> str:
    if not updated_at:
        return ""
    try:
        when = datetime.fromisoformat(updated_at)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        days = (datetime.now(timezone.utc) - when).days
    except ValueError:
        return ""
    if days >= 30:
        return f" (last updated {days}d ago — verify)"
    return ""


def format_inject(records: list[dict[str, Any]]) -> str:
    if not records:
        return ""
    lines = ["Durable memories relevant to this task:"]
    for rec in records:
        age = _age_note(rec.get("updated_at"))
        lines.append(
            f"- [{rec['id'][:8]}] ({rec['kind']}, {rec['provenance']}) {rec['title']}{age}"
        )
        body = (rec.get("body") or "").strip()
        if body:
            snippet = body if len(body) <= 280 else body[:277] + "..."
            lines.append(f"  {snippet}")
    return "\n".join(lines)
