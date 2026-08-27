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

from unforgettable.operators import (
    ERROR_BLOCKED,
    ERROR_REFUSED,
    ERROR_UNKNOWN,
    ERROR_VOTER_OFF,
    admit_record,
    reject_record,
    review_proposed,
    summarize_store,
)
from unforgettable.store.records import get_record, insert_record
from unforgettable.supervisor import SupervisorConfig, VOTER_ADVISORY, VOTER_BINDING


class _ScriptedHost:
    def __init__(self, text: str):
        self.text = text
        self.calls = []

    async def supervise(
        self,
        purpose,
        messages,
        *,
        model = None,
        max_tokens = 400,
    ):
        self.calls.append(purpose)
        return self.text


def test_admit_record_promotes_proposed(db_path):
    rec = insert_record(
        kind = "claim",
        title = "Draft",
        body = "maybe",
        provenance = "infer",
        status = "proposed",
        db_path = db_path,
    )
    outcome = admit_record(rec["id"], db_path = db_path)
    assert outcome.ok
    assert get_record(rec["id"], db_path = db_path)["status"] == "active"


def test_admit_record_refuses_active_without_force(db_path):
    rec = insert_record(
        kind = "claim",
        title = "Live",
        body = "stays",
        provenance = "world",
        db_path = db_path,
    )
    outcome = admit_record(rec["id"], db_path = db_path)
    assert not outcome.ok
    assert outcome.error_kind == ERROR_REFUSED
    assert outcome.error_detail == "active"


def test_admit_colliding_who_needs_force(db_path):
    what = insert_record(
        kind = "procedure",
        title = "Deploy window",
        body = "deploy on Tuesday after tests",
        provenance = "world",
        db_path = db_path,
    )
    who = insert_record(
        kind = "directive",
        title = "Deploy window",
        body = "ship whenever the exec says",
        provenance = "human",
        speaker = "user",
        status = "proposed",
        db_path = db_path,
    )
    outcome = admit_record(who["id"], db_path = db_path)
    assert not outcome.ok
    assert outcome.error_kind == ERROR_REFUSED
    assert what["id"] in (outcome.error_detail or "")
    forced = admit_record(who["id"], force = True, db_path = db_path)
    assert forced.ok
    assert get_record(who["id"], db_path = db_path)["status"] == "active"


def test_admit_record_unknown_id(db_path):
    outcome = admit_record("missing", db_path = db_path)
    assert not outcome.ok
    assert outcome.error_kind == ERROR_UNKNOWN


def test_admit_binding_deny_blocks(db_path):
    rec = insert_record(
        kind = "claim",
        title = "Draft",
        body = "maybe",
        provenance = "infer",
        status = "proposed",
        db_path = db_path,
    )
    host = _ScriptedHost('{"decision":"deny","reason":"weak"}')
    cfg = SupervisorConfig(voter = VOTER_BINDING)
    outcome = admit_record(rec["id"], db_path = db_path, host = host, config = cfg)
    assert not outcome.ok
    assert outcome.error_kind == ERROR_BLOCKED
    assert get_record(rec["id"], db_path = db_path)["status"] == "proposed"
    forced = admit_record(rec["id"], force = True, db_path = db_path, host = host, config = cfg)
    assert forced.ok
    assert get_record(rec["id"], db_path = db_path)["status"] == "active"


def test_reject_record(db_path):
    rec = insert_record(
        kind = "claim",
        title = "Draft",
        body = "no",
        provenance = "infer",
        status = "proposed",
        db_path = db_path,
    )
    outcome = reject_record(rec["id"], reason = "nope", db_path = db_path)
    assert outcome.ok
    assert get_record(rec["id"], db_path = db_path)["status"] == "rejected"


def test_review_requires_voter(db_path):
    outcome = review_proposed(db_path = db_path)
    assert not outcome.ok
    assert outcome.error_kind == ERROR_VOTER_OFF


def test_review_advisory_apply(db_path):
    keep = insert_record(
        kind = "error_fix",
        title = "Keep",
        body = "use pytest",
        provenance = "world",
        status = "proposed",
        db_path = db_path,
    )
    drop = insert_record(
        kind = "claim",
        title = "Drop",
        body = "noise",
        provenance = "infer",
        status = "proposed",
        db_path = db_path,
    )

    class _ById:
        async def supervise(
            self,
            purpose,
            messages,
            *,
            model = None,
            max_tokens = 400,
        ):
            import json

            payload = json.loads(messages[-1]["content"])
            if payload["id"] == keep["id"]:
                return '{"decision":"allow","reason":"solid"}'
            return '{"decision":"deny","reason":"junk"}'

    cfg = SupervisorConfig(voter = VOTER_ADVISORY)
    outcome = review_proposed(apply = True, db_path = db_path, host = _ById(), config = cfg)
    assert outcome.ok
    assert get_record(keep["id"], db_path = db_path)["status"] == "active"
    assert get_record(drop["id"], db_path = db_path)["status"] == "rejected"


def test_summarize_store_counts(db_path):
    insert_record(
        kind = "claim",
        title = "A",
        body = "a",
        provenance = "world",
        db_path = db_path,
    )
    insert_record(
        kind = "procedure",
        title = "B",
        body = "b",
        provenance = "infer",
        status = "proposed",
        db_path = db_path,
    )
    summary = summarize_store(db_path = db_path)
    assert summary["records"]["total"] == 2
    assert summary["records"]["by_status"]["proposed"] == 1
    assert summary["compiled_count"] == 0
    assert summary["adapters"]["promoted"] == 0
    assert summary["adapters"]["promoted_id"] is None
    assert summary["last_inject"] is None
    assert str(db_path.resolve()) == summary["db_path"]
