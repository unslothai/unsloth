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

"""SFT / distill message JSON and world fail/pass preference pairs."""

from __future__ import annotations

from typing import Any

from unforgettable.store.records import list_records, list_rollouts

PACK_BODY_CHARS = 1200
PREFERENCE_MAX_PAIRS = 32
_TRUSTED_CHOSEN = frozenset({"world", "mixed", "human"})


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


def _twin_note_episodes(*, db_path = None) -> set[str]:
    eps: set[str] = set()
    for rec in list_records(kinds = ["twin_note"], statuses = ["active"], db_path = db_path):
        eid = rec.get("source_episode_id")
        if eid:
            eps.add(eid)
    return eps


def _fail_text_from_error_fix(rec: dict[str, Any]) -> str:
    """Rejected text: first Tried:/failed line, else the error_fix title."""
    for line in (rec.get("body") or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("tried:"):
            rest = stripped.split(":", 1)[1].strip()
            return rest or stripped
        if "failed" in lower:
            return stripped
    return (rec.get("title") or "").strip()


def preference_pairs(*, db_path = None, train_episode_ids: set[str] | None = None) -> list[dict]:
    """World-pass + admitted error_fix pairs. Chosen is never sim-only."""
    # run() stores last fail and last pass per contact. Chosen is the admitted body.
    world_pass = {
        row["episode_id"]
        for row in list_rollouts(contact = "world", outcome = "pass", db_path = db_path)
        if row.get("episode_id")
    }
    twin_eps = _twin_note_episodes(db_path = db_path)
    pairs: list[dict] = []
    seen: set[str] = set()
    for rec in list_records(kinds = ["error_fix"], statuses = ["active"], db_path = db_path):
        eid = rec.get("source_episode_id")
        if not eid or eid in seen or eid in twin_eps:
            continue
        if train_episode_ids is not None and eid not in train_episode_ids:
            continue
        if eid not in world_pass:
            continue
        if rec.get("provenance") not in _TRUSTED_CHOSEN:
            continue
        chosen = _clip((rec.get("body") or "").strip(), PACK_BODY_CHARS)
        rejected = _clip(_fail_text_from_error_fix(rec), PACK_BODY_CHARS)
        if not chosen or not rejected:
            continue
        seen.add(eid)
        pairs.append(
            {
                "prompt": [{"role": "user", "content": rejected}],
                "chosen": chosen,
                "rejected": rejected,
                "episode_id": eid,
            }
        )
        if len(pairs) >= PREFERENCE_MAX_PAIRS:
            break
    return pairs
