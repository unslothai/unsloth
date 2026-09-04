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

from collections import defaultdict

from unforgettable.constants import (
    WHAT_GATE_KINDS,
    WHO_CANDIDATE_KINDS,
    is_what,
    is_who,
    speaker_of,
)
from unforgettable.store.records import list_records, log_admission
from unforgettable.store.titles import normalize_title

from .protocols import Contradiction

CONTRADICTION_KIND = "claim"
PROCEDURE_KIND = "procedure"
CONTRADICTION_STATUS = "active"
MIN_DISTINCT_BODIES = 2
CONTRADICTION_REASON = "same title, distinct bodies"
DISSONANCE_REASON = "who collides with what"
CONTRADICTS_PREFIX = "contradicts "
DISSONANCE_PREFIX = "dissonance: contradicts "
WHO_UNBACKED_USER = "who: unbacked user assertion"
WHO_UNBACKED_OTHER = "who: unbacked other"
NOTE_DECISION = "note"


def review_write(
    *,
    kind: str,
    title: str,
    body: str,
    provenance: str,
    db_path = None,
    speaker: str | None = None,
    warrant: str | None = None,
) -> str:
    incoming = {
        "kind": kind,
        "title": title,
        "body": body,
        "provenance": provenance,
        "speaker": speaker,
        "warrant": warrant,
    }
    title_key = normalize_title(title)
    body_key = normalize_title(body)
    if kind == CONTRADICTION_KIND:
        for rec in list_records(
            kinds = [CONTRADICTION_KIND],
            statuses = [CONTRADICTION_STATUS],
            db_path = db_path,
        ):
            if normalize_title(rec["title"]) != title_key:
                continue
            if normalize_title(rec["body"]) != body_key:
                return f"{CONTRADICTS_PREFIX}{rec['id']}"
    if kind == PROCEDURE_KIND:
        for rec in list_records(
            kinds = [PROCEDURE_KIND],
            statuses = [CONTRADICTION_STATUS],
            db_path = db_path,
        ):
            if normalize_title(rec["title"]) != title_key:
                continue
            if normalize_title(rec["body"]) != body_key:
                return f"{CONTRADICTS_PREFIX}{rec['id']}"
    if is_who(incoming):
        peer = colliding_what(incoming, db_path = db_path)
        if peer is not None:
            return f"{DISSONANCE_PREFIX}{peer['id']}"
        if kind in WHO_CANDIDATE_KINDS:
            who = speaker_of(incoming)
            if who == "user":
                return WHO_UNBACKED_USER
            if who == "other":
                return WHO_UNBACKED_OTHER
    return ""


def colliding_what(rec: dict, *, db_path = None) -> dict | None:
    if not is_who(rec):
        return None
    title_key = normalize_title(rec.get("title") or "")
    skip_id = rec.get("id")
    for other in list_records(
        kinds = list(WHAT_GATE_KINDS),
        statuses = [CONTRADICTION_STATUS],
        db_path = db_path,
    ):
        if skip_id and other["id"] == skip_id:
            continue
        if not is_what(other):
            continue
        if normalize_title(other.get("title") or "") == title_key:
            return other
    return None


def contradictions(db_path = None) -> list[Contradiction]:
    groups: dict[str, list] = defaultdict(list)
    active = list_records(statuses = [CONTRADICTION_STATUS], db_path = db_path)
    proposed = list_records(statuses = ["proposed"], db_path = db_path)
    for rec in active:
        if rec.get("kind") in {CONTRADICTION_KIND, PROCEDURE_KIND}:
            groups[normalize_title(rec["title"])].append(rec)
    found: list[Contradiction] = []
    seen_titles: set[str] = set()
    for title_key in sorted(groups):
        recs = groups[title_key]
        bodies = {normalize_title(rec["body"]) for rec in recs}
        if len(bodies) < MIN_DISTINCT_BODIES:
            continue
        found.append(
            Contradiction(
                title_key = title_key,
                record_ids = tuple(rec["id"] for rec in recs),
                reason = CONTRADICTION_REASON,
            )
        )
        seen_titles.add(title_key)
    by_title: dict[str, list] = defaultdict(list)
    for rec in [*active, *proposed]:
        by_title[normalize_title(rec.get("title") or "")].append(rec)
    for title_key in sorted(by_title):
        if title_key in seen_titles:
            continue
        recs = by_title[title_key]
        who_ids = tuple(rec["id"] for rec in recs if is_who(rec))
        what_ids = tuple(
            rec["id"]
            for rec in recs
            if rec.get("status") == CONTRADICTION_STATUS
            and is_what(rec)
            and rec.get("kind") in WHAT_GATE_KINDS
        )
        if not who_ids or not what_ids:
            continue
        found.append(
            Contradiction(
                title_key = title_key,
                record_ids = what_ids + who_ids,
                reason = DISSONANCE_REASON,
            )
        )
    return found


class LogGateEyes:
    def note(
        self,
        message: str,
        *,
        db_path = None,
    ) -> None:
        log_admission(
            record_id = None,
            decision = NOTE_DECISION,
            reason = message,
            db_path = db_path,
        )

    def contradictions(self, db_path = None) -> list[Contradiction]:
        return contradictions(db_path = db_path)

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
        return review_write(
            kind = kind,
            title = title,
            body = body,
            provenance = provenance,
            db_path = db_path,
            speaker = speaker,
            warrant = warrant,
        )
