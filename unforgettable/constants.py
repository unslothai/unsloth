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

"""Record kinds, statuses, and provenance. Phase 1 implements the full set."""

from __future__ import annotations

KINDS = frozenset(
    {"claim", "procedure", "error_fix", "entity", "episode", "directive", "twin_note"}
)
# Tools may not mint episode rows; the runner owns those.
TOOL_WRITE_KINDS = frozenset(
    {"claim", "procedure", "error_fix", "entity", "directive", "twin_note"}
)
STATUSES = frozenset({"active", "superseded", "deprecated", "proposed", "rejected"})
PROVENANCES = frozenset({"world", "sim", "mixed", "human", "infer"})
SPEAKER_WORLD = "world"
SPEAKER_SIM = "sim"
SPEAKER_USER = "user"
SPEAKER_MODEL = "model"
SPEAKER_OTHER = "other"
SPEAKERS = frozenset({SPEAKER_WORLD, SPEAKER_SIM, SPEAKER_USER, SPEAKER_MODEL, SPEAKER_OTHER})
ADMISSION_MODES = frozenset({"auto", "propose", "deny"})
# Operator CLI admit without --force.
ADMIT_FROM_STATUSES = frozenset({"proposed", "deprecated"})
WHAT_GATE_KINDS = frozenset({"claim", "procedure", "error_fix", "entity"})
WHO_CANDIDATE_KINDS = frozenset({"claim", "entity"})

# Hard caps so a tool dump cannot blow up FTS or become uncapped gold.
RECORD_TITLE_CHARS = 200
RECORD_BODY_CHARS = 4000
EVENT_SUMMARY_CHARS = 240
ROLLOUT_SUMMARY_CHARS = 400
SEARCH_TOP_K_MAX = 32
SEARCH_FTS_SCAN_CAP = 256

DEFAULT_NAMESPACE_ID = "default"
DEFAULT_NAMESPACE_NAME = "default"

# Lower is preferred at retrieve time.
PROVENANCE_WEIGHT = {
    "world": 0,
    "mixed": 1,
    "human": 2,
    "sim": 3,
    "infer": 4,
}

# Speaker default when the writer omitted it. Mixed has a world leg.
DEFAULT_SPEAKER_BY_PROVENANCE = {
    "world": SPEAKER_WORLD,
    "sim": SPEAKER_SIM,
    "mixed": SPEAKER_WORLD,
    "human": SPEAKER_USER,
    "infer": SPEAKER_MODEL,
}

# Lower is preferred at retrieve. WHAT outranks WHO.
TYPOLOGY_WHAT_WORLD = 0
TYPOLOGY_WHAT_MIXED = 1
TYPOLOGY_WHAT_SIM = 2
TYPOLOGY_WHAT_BACKED = 3
TYPOLOGY_WHO_USER = 4
TYPOLOGY_WHO_OTHER = 5
TYPOLOGY_WHO_MODEL = 6

ACTIVE_RETRIEVE_STATUSES = frozenset({"active"})
WARRANT_CHARS = 800
SPEAKER_LABEL_CHARS = 80


def speaker_of(rec: dict) -> str:
    speaker = rec.get("speaker")
    if speaker in SPEAKERS:
        return speaker
    kind = rec.get("kind")
    if kind == "directive":
        return SPEAKER_USER
    contact = rec.get("contact_tag")
    if contact in {SPEAKER_WORLD, SPEAKER_SIM}:
        return contact
    return DEFAULT_SPEAKER_BY_PROVENANCE.get(rec.get("provenance"), SPEAKER_MODEL)


def warrant_of(rec: dict) -> str:
    return (rec.get("warrant") or "").strip()


def resolve_speaker(
    *,
    speaker: str | None,
    provenance: str,
    kind: str,
    contact_tag: str | None = None,
) -> str:
    if speaker in SPEAKERS:
        return speaker
    if kind == "directive":
        return SPEAKER_USER
    if contact_tag in {SPEAKER_WORLD, SPEAKER_SIM}:
        return contact_tag
    return DEFAULT_SPEAKER_BY_PROVENANCE.get(provenance, SPEAKER_MODEL)


def coerce_unbacked_user_provenance(provenance: str, *, speaker: str, warrant: str) -> str:
    """Unbacked user assertions cannot mint world provenance."""
    if speaker == SPEAKER_USER and not (warrant or "").strip() and provenance == "world":
        return "infer"
    return provenance


def is_who(rec: dict) -> bool:
    if rec.get("kind") == "directive":
        return True
    speaker = speaker_of(rec)
    if speaker in {SPEAKER_WORLD, SPEAKER_SIM}:
        return False
    if warrant_of(rec):
        return False
    return True


def is_what(rec: dict) -> bool:
    return not is_who(rec)


def typology_class(rec: dict) -> int:
    """Retrieve rank: WHAT world … WHAT backed text, then WHO user/other/model."""
    if rec.get("kind") == "directive":
        return TYPOLOGY_WHO_USER
    speaker = speaker_of(rec)
    warranted = bool(warrant_of(rec))
    provenance = rec.get("provenance") or ""
    if speaker == SPEAKER_WORLD:
        if provenance == "mixed":
            return TYPOLOGY_WHAT_MIXED
        return TYPOLOGY_WHAT_WORLD
    if speaker == SPEAKER_SIM:
        if provenance == "mixed":
            return TYPOLOGY_WHAT_MIXED
        return TYPOLOGY_WHAT_SIM
    if provenance == "mixed" and warranted:
        return TYPOLOGY_WHAT_MIXED
    if warranted:
        return TYPOLOGY_WHAT_BACKED
    if speaker == SPEAKER_USER:
        return TYPOLOGY_WHO_USER
    if speaker == SPEAKER_OTHER:
        return TYPOLOGY_WHO_OTHER
    return TYPOLOGY_WHO_MODEL
