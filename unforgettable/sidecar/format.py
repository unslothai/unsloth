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

from unforgettable.store.records import (
    get_record,
    list_records,
    list_retrieve_uses,
    list_rollouts,
)

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


def _twin_note_episodes(*, db_path=None) -> set[str]:
    eps: set[str] = set()
    for rec in list_records(kinds=["twin_note"], db_path=db_path):
        eid = rec.get("source_episode_id")
        if eid:
            eps.add(eid)
    return eps


def _linked_procedure_title(
    episode_id: str, fail: dict[str, Any], *, db_path=None
) -> str:
    candidates: list[str] = []
    sid = fail.get("source_record_id")
    if sid:
        candidates.append(sid)
    for use in list_retrieve_uses(episode_id=episode_id, db_path=db_path):
        rid = use.get("record_id")
        if rid:
            candidates.append(rid)
    for rid in candidates:
        rec = get_record(rid, db_path=db_path)
        if rec is None or rec.get("kind") != "procedure":
            continue
        title = (rec.get("title") or "").strip()
        if title:
            return title
    return ""


def _admitted_error_fix_body(episode_id: str, *, db_path=None) -> str:
    for rec in list_records(kinds=["error_fix"], statuses=["active"], db_path=db_path):
        if rec.get("source_episode_id") != episode_id:
            continue
        if rec.get("provenance") not in _TRUSTED_CHOSEN:
            continue
        body = (rec.get("body") or "").strip()
        if body:
            return _clip(body, PACK_BODY_CHARS)
    return ""


def preference_pairs(*, db_path=None) -> list[dict]:
    """World fail/pass pairs. Chosen is never sim-only."""
    has_world_fail: set[str] = set()
    has_world_pass: set[str] = set()
    for row in list_rollouts(db_path=db_path):
        eid = row.get("episode_id")
        if not eid or row.get("contact") != "world":
            continue
        if row.get("outcome") == "fail":
            has_world_fail.add(eid)
        elif row.get("outcome") == "pass":
            has_world_pass.add(eid)
    twin_eps = _twin_note_episodes(db_path=db_path)
    pairs: list[dict] = []
    for eid in sorted(has_world_fail & has_world_pass):
        if eid in twin_eps:
            continue
        fail = None
        later_pass = None
        # episode_id filter is created_at ASC: later pass is after the fail.
        for row in list_rollouts(episode_id=eid, db_path=db_path):
            if (
                fail is None
                and row.get("contact") == "world"
                and row.get("outcome") == "fail"
            ):
                fail = row
                continue
            if (
                fail is not None
                and later_pass is None
                and row.get("contact") == "world"
                and row.get("outcome") == "pass"
            ):
                later_pass = row
                break
        if fail is None or later_pass is None:
            continue
        fail_summary = _clip((fail.get("summary") or "").strip(), PACK_BODY_CHARS)
        pass_summary = _clip((later_pass.get("summary") or "").strip(), PACK_BODY_CHARS)
        prompt_content = fail_summary or _linked_procedure_title(
            eid, fail, db_path=db_path
        )
        if not prompt_content or not fail_summary:
            continue
        chosen = _admitted_error_fix_body(eid, db_path=db_path) or pass_summary
        if not chosen:
            continue
        pairs.append(
            {
                "prompt": [{"role": "user", "content": prompt_content}],
                "chosen": chosen,
                "rejected": fail_summary,
                "episode_id": eid,
            }
        )
        if len(pairs) >= PREFERENCE_MAX_PAIRS:
            break
    return pairs
