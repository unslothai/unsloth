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

"""Operator sequences shared by the CLI and the Studio face.

Does not change ``admit()`` predicates. Vote + status / compile / promote live
here so HTTP and argv cannot fork policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from unforgettable.constants import ADMIT_FROM_STATUSES
from unforgettable.eyes.gate import colliding_what, contradictions
from unforgettable.sidecar.adapters import get_adapter, list_adapters, promote_adapter
from unforgettable.store.compile import count_compiled, pin_compiled
from unforgettable.store.db import default_db_path
from unforgettable.store.records import (
    get_record,
    insert_record,
    list_admissions,
    list_inject_stats,
    list_records,
    list_rollouts,
    set_record_status,
    summarize_records,
)
from unforgettable.supervisor import (
    SKIP_VOTE_KINDS,
    VOTER_OFF,
    SupervisorConfig,
    Vote,
    config_from_env,
    request_mine_sync,
    request_vote_sync,
    voter_blocks,
)

CLI_ADMIT_REASON = "cli admit"
CLI_REJECT_REASON = "cli reject"
STUDIO_ADMIT_REASON = "studio admit"
STUDIO_REJECT_REASON = "studio reject"

ERROR_UNKNOWN = "unknown"
ERROR_REFUSED = "refused"
ERROR_BLOCKED = "blocked"
ERROR_VOTER_OFF = "voter_off"
ERROR_NO_HOST = "no_host"
ERROR_INVALID = "invalid"

UNKNOWN_ID_EXIT = 2


@dataclass
class OperatorOutcome:
    ok: bool
    code: int = 0
    record: Optional[dict[str, Any]] = None
    vote: Optional[Vote] = None
    error_kind: Optional[str] = None
    error_detail: Optional[str] = None
    items: Any = None


def _cfg(config: SupervisorConfig | None) -> SupervisorConfig:
    return config if config is not None else config_from_env()


def maybe_vote(
    candidate: dict[str, Any],
    *,
    db_path,
    force: bool,
    host: Any = None,
    config: SupervisorConfig | None = None,
) -> OperatorOutcome:
    cfg = _cfg(config)
    if cfg.voter == VOTER_OFF:
        return OperatorOutcome(ok = True)
    vote = request_vote_sync(
        candidate,
        host = host,
        config = cfg,
        db_path = db_path,
    )
    if voter_blocks(vote, force = force, config = cfg):
        return OperatorOutcome(
            ok = False,
            code = UNKNOWN_ID_EXIT,
            vote = vote,
            error_kind = ERROR_BLOCKED,
            error_detail = vote.reason,
        )
    return OperatorOutcome(ok = True, vote = vote)


def admit_record(
    record_id: str,
    *,
    force: bool = False,
    db_path = None,
    host: Any = None,
    config: SupervisorConfig | None = None,
    reason: str = CLI_ADMIT_REASON,
) -> OperatorOutcome:
    existing = get_record(record_id, db_path = db_path)
    if existing is None:
        return OperatorOutcome(
            ok = False,
            code = UNKNOWN_ID_EXIT,
            error_kind = ERROR_UNKNOWN,
            error_detail = record_id,
        )
    if not force and existing["status"] not in ADMIT_FROM_STATUSES:
        return OperatorOutcome(
            ok = False,
            code = UNKNOWN_ID_EXIT,
            record = existing,
            error_kind = ERROR_REFUSED,
            error_detail = existing["status"],
        )
    if not force and existing["status"] in ADMIT_FROM_STATUSES:
        peer = colliding_what(existing, db_path = db_path)
        if peer is not None:
            return OperatorOutcome(
                ok = False,
                code = UNKNOWN_ID_EXIT,
                record = existing,
                error_kind = ERROR_REFUSED,
                error_detail = f"dissonance: contradicts {peer['id']}",
            )
    voted = maybe_vote(existing, db_path = db_path, force = force, host = host, config = config)
    if not voted.ok:
        return voted
    try:
        rec = set_record_status(record_id, "active", reason = reason, db_path = db_path)
    except KeyError:
        return OperatorOutcome(
            ok = False,
            code = UNKNOWN_ID_EXIT,
            vote = voted.vote,
            error_kind = ERROR_UNKNOWN,
            error_detail = record_id,
        )
    return OperatorOutcome(ok = True, record = rec, vote = voted.vote)


def reject_record(
    record_id: str,
    *,
    reason: str = CLI_REJECT_REASON,
    db_path = None,
) -> OperatorOutcome:
    try:
        rec = set_record_status(record_id, "rejected", reason = reason, db_path = db_path)
    except KeyError:
        return OperatorOutcome(
            ok = False,
            code = UNKNOWN_ID_EXIT,
            error_kind = ERROR_UNKNOWN,
            error_detail = record_id,
        )
    return OperatorOutcome(ok = True, record = rec)


def compile_record(
    record_id: str,
    *,
    db_path = None,
    host: Any = None,
    config: SupervisorConfig | None = None,
) -> OperatorOutcome:
    existing = get_record(record_id, db_path = db_path)
    if existing is None:
        return OperatorOutcome(
            ok = False,
            code = UNKNOWN_ID_EXIT,
            error_kind = ERROR_UNKNOWN,
            error_detail = record_id,
        )
    voted = maybe_vote(existing, db_path = db_path, force = False, host = host, config = config)
    if not voted.ok:
        return voted
    try:
        row = pin_compiled(record_id, explicit = True, db_path = db_path)
    except ValueError as exc:
        return OperatorOutcome(
            ok = False,
            code = UNKNOWN_ID_EXIT,
            vote = voted.vote,
            error_kind = ERROR_INVALID,
            error_detail = str(exc),
        )
    return OperatorOutcome(ok = True, record = row, vote = voted.vote)


def promote_adapter_record(
    adapter_id: str,
    *,
    force: bool = False,
    db_path = None,
    host: Any = None,
    config: SupervisorConfig | None = None,
) -> OperatorOutcome:
    adapter = get_adapter(adapter_id, db_path = db_path)
    if adapter is None:
        return OperatorOutcome(
            ok = False,
            code = UNKNOWN_ID_EXIT,
            error_kind = ERROR_UNKNOWN,
            error_detail = adapter_id,
        )
    candidate = {
        "id": adapter.get("id"),
        "kind": "adapter",
        "status": adapter.get("status"),
        "title": f"adapter {adapter.get('id')}",
        "body": adapter.get("metrics") or "",
        "provenance": "mixed",
        "extra": {
            "pack_id": adapter.get("pack_id"),
            "backend": adapter.get("backend"),
            "recipe": adapter.get("recipe"),
        },
    }
    voted = maybe_vote(candidate, db_path = db_path, force = force, host = host, config = config)
    if not voted.ok:
        return voted
    try:
        row = promote_adapter(adapter_id, force = force, db_path = db_path)
    except KeyError:
        return OperatorOutcome(
            ok = False,
            code = UNKNOWN_ID_EXIT,
            vote = voted.vote,
            error_kind = ERROR_UNKNOWN,
            error_detail = adapter_id,
        )
    except ValueError as exc:
        return OperatorOutcome(
            ok = False,
            code = UNKNOWN_ID_EXIT,
            vote = voted.vote,
            error_kind = ERROR_INVALID,
            error_detail = str(exc),
        )
    return OperatorOutcome(ok = True, record = row, vote = voted.vote)


def proposed_for_review(*, db_path = None) -> list[dict[str, Any]]:
    rows = list_records(statuses = ["proposed"], db_path = db_path)
    return [row for row in rows if row.get("kind") not in SKIP_VOTE_KINDS]


def review_proposed(
    *,
    apply: bool = False,
    limit: int = 20,
    db_path = None,
    host: Any = None,
    config: SupervisorConfig | None = None,
) -> OperatorOutcome:
    cfg = _cfg(config)
    if cfg.voter == VOTER_OFF:
        return OperatorOutcome(
            ok = False,
            code = UNKNOWN_ID_EXIT,
            error_kind = ERROR_VOTER_OFF,
        )
    rows = proposed_for_review(db_path = db_path)[:limit]
    report = []
    for rec in rows:
        vote = request_vote_sync(rec, host = host, config = cfg, db_path = db_path)
        applied = None
        if apply and vote.decision == "allow":
            set_record_status(rec["id"], "active", reason = "review allow", db_path = db_path)
            applied = "active"
        elif apply and vote.decision == "deny":
            set_record_status(rec["id"], "rejected", reason = "review deny", db_path = db_path)
            applied = "rejected"
        report.append(
            {
                "id": rec["id"],
                "kind": rec["kind"],
                "title": rec["title"],
                "decision": vote.decision,
                "reason": vote.reason,
                "applied": applied,
            }
        )
    return OperatorOutcome(ok = True, items = report)


def mine_store(
    *,
    apply: bool = False,
    limit: int = 20,
    db_path = None,
    host: Any = None,
    config: SupervisorConfig | None = None,
) -> OperatorOutcome:
    cfg = _cfg(config)
    if cfg.voter == VOTER_OFF:
        return OperatorOutcome(
            ok = False,
            code = UNKNOWN_ID_EXIT,
            error_kind = ERROR_VOTER_OFF,
        )
    if host is None:
        return OperatorOutcome(
            ok = False,
            code = UNKNOWN_ID_EXIT,
            error_kind = ERROR_NO_HOST,
        )
    proposed = proposed_for_review(db_path = db_path)[:limit]
    rollouts = list_rollouts(limit = limit, db_path = db_path)
    admissions = list_admissions(limit = limit, db_path = db_path)
    items = request_mine_sync(
        host,
        proposed = proposed,
        rollouts = rollouts,
        admissions = admissions,
        config = cfg,
    )
    report = []
    for item in items:
        rec_id = item.get("id")
        applied = None
        if rec_id:
            existing = get_record(rec_id, db_path = db_path)
            if existing is None:
                report.append({**item, "applied": None, "error": "unknown id"})
                continue
            if apply and item.get("decision") == "allow":
                set_record_status(rec_id, "active", reason = "mine allow", db_path = db_path)
                applied = "active"
            elif apply and item.get("decision") == "deny":
                set_record_status(rec_id, "rejected", reason = "mine deny", db_path = db_path)
                applied = "rejected"
            report.append({**item, "applied": applied})
            continue
        inserted = None
        if apply and item.get("title") and item.get("kind"):
            inserted = insert_record(
                kind = item["kind"],
                title = item["title"],
                body = item.get("body") or "",
                provenance = "infer",
                status = "proposed",
                db_path = db_path,
            )
            applied = "proposed"
        report.append({**item, "id": (inserted or {}).get("id"), "applied": applied})
    return OperatorOutcome(ok = True, items = report)


def summarize_store(*, db_path = None) -> dict[str, Any]:
    path = Path(db_path) if db_path is not None else default_db_path()
    records = summarize_records(db_path = path)
    adapters = list_adapters(db_path = path)
    by_adapter = {"shadow": 0, "promoted": 0, "discarded": 0}
    promoted_id = None
    for row in adapters:
        status = row.get("status")
        if status in by_adapter:
            by_adapter[status] += 1
        if status == "promoted":
            promoted_id = row.get("id")
    inject_rows = list_inject_stats(limit = 1, db_path = path)
    by_status = records["by_status"]
    archive = (
        int(by_status.get("deprecated") or 0)
        + int(by_status.get("superseded") or 0)
        + int(by_status.get("rejected") or 0)
    )
    return {
        "db_path": str(path.expanduser().resolve()),
        "records": records,
        "archive_count": archive,
        "compiled_count": count_compiled(db_path = path),
        "adapters": {**by_adapter, "promoted_id": promoted_id},
        "contradiction_count": len(contradictions(db_path = path)),
        "last_inject": inject_rows[0] if inject_rows else None,
    }
