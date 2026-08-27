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

from unforgettable.agents.admissions import admit
from unforgettable.store.db import get_connection
from unforgettable.store.records import ensure_default_namespace


def test_explicit_human_admitted(db_path):
    ensure_default_namespace(db_path = db_path)
    decision = admit(kind = "claim", provenance = "human", explicit = True, db_path = db_path)
    assert decision.status == "active"


def test_infer_stays_proposed(db_path):
    ensure_default_namespace(db_path = db_path)
    decision = admit(kind = "claim", provenance = "infer", explicit = True, db_path = db_path)
    assert decision.status == "proposed"


def test_sim_only_claim_not_auto_promoted(db_path):
    ensure_default_namespace(db_path = db_path)
    decision = admit(kind = "claim", provenance = "sim", explicit = True, db_path = db_path)
    assert decision.status == "proposed"


def test_auto_extract_is_proposed(db_path):
    ensure_default_namespace(db_path = db_path)
    decision = admit(kind = "error_fix", provenance = "mixed", explicit = False, db_path = db_path)
    assert decision.status == "proposed"
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT decision, reason FROM admissions_log").fetchall()
    finally:
        conn.close()
    assert rows
    assert rows[-1]["decision"] == "proposed"


def test_bookkeeping_twin_note_is_active(db_path):
    ensure_default_namespace(db_path = db_path)
    decision = admit(
        kind = "twin_note",
        provenance = "mixed",
        explicit = False,
        bookkeeping = True,
        db_path = db_path,
    )
    assert decision.status == "active"


def test_bookkeeping_episode_is_active(db_path):
    ensure_default_namespace(db_path = db_path)
    decision = admit(
        kind = "episode",
        provenance = "mixed",
        explicit = False,
        bookkeeping = True,
        db_path = db_path,
    )
    assert decision.status == "active"


def test_force_proposed_beats_bookkeeping(db_path):
    ensure_default_namespace(db_path = db_path)
    decision = admit(
        kind = "twin_note",
        provenance = "mixed",
        explicit = False,
        bookkeeping = True,
        force_proposed_reason = "contradicts other",
        db_path = db_path,
    )
    assert decision.status == "proposed"
    assert decision.reason == "contradicts other"


def test_sim_procedure_not_auto_promoted(db_path):
    ensure_default_namespace(db_path = db_path)
    decision = admit(kind = "procedure", provenance = "sim", explicit = True, db_path = db_path)
    assert decision.status == "proposed"


def test_infer_directive_stays_proposed(db_path):
    ensure_default_namespace(db_path = db_path)
    decision = admit(kind = "directive", provenance = "infer", explicit = True, db_path = db_path)
    assert decision.status == "proposed"


def test_world_directive_stays_proposed(db_path):
    ensure_default_namespace(db_path = db_path)
    decision = admit(kind = "directive", provenance = "world", explicit = True, db_path = db_path)
    assert decision.status == "proposed"


def test_human_directive_is_admitted(db_path):
    ensure_default_namespace(db_path = db_path)
    decision = admit(kind = "directive", provenance = "human", explicit = True, db_path = db_path)
    assert decision.status == "active"
