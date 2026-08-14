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

import json
import uuid
from typing import Any, Optional

from unforgettable.agents.admissions import admit
from unforgettable.agents.retriever import DEFAULT_MAX_RECORDS, DEFAULT_RETRIEVE_KINDS
from unforgettable.constants import DEFAULT_NAMESPACE_ID
from unforgettable.eyes.gate import review_write
from unforgettable.loop.runtime import current_db_path, current_episode_id, current_namespace
from unforgettable.store.compact import CompactReport, run_compact
from unforgettable.store.compile import (
    get_compiled,
    is_compile_candidate,
    maybe_compile,
    pin_compiled,
    procedure_hits,
)
from unforgettable.store.records import (
    deprecate_record,
    get_record,
    insert_record,
    list_records,
    supersede_record,
)
from unforgettable.store.search import search_records


def dispatch(name: str, arguments: dict[str, Any] | None, *, db_path=None) -> str:
    args = arguments or {}
    path = db_path if db_path is not None else current_db_path()
    name = name.replace(".", "_")
    if name == "memory_write":
        return _write(args, db_path=path)
    if name == "memory_search":
        return _search(args, db_path=path)
    if name == "memory_get":
        return _get(args, db_path=path)
    if name == "memory_supersede":
        return _supersede(args, db_path=path)
    if name == "memory_deprecate":
        return _deprecate(args, db_path=path)
    if name == "memory_compact":
        return _compact(args, db_path=path)
    if name == "memory_compile":
        return _compile(args, db_path=path)
    if name == "rims_enter_sim":
        return "enter_sim requested"
    return f"Error: unknown memory tool '{name}'"


def _write(args: dict[str, Any], *, db_path) -> str:
    try:
        kind = str(args["kind"])
        title = str(args["title"])
        body = str(args["body"])
        provenance = str(args["provenance"])
    except KeyError as exc:
        return f"Error: missing field {exc}"
    namespace = str(args.get("namespace") or current_namespace() or DEFAULT_NAMESPACE_ID)
    review_reason = review_write(
        kind=kind,
        title=title,
        body=body,
        provenance=provenance,
        db_path=db_path,
    )
    rid = str(uuid.uuid4())
    decision = admit(
        kind=kind,
        provenance=provenance,
        explicit=True,
        namespace_id=namespace,
        record_id=rid,
        db_path=db_path,
        force_proposed_reason=review_reason or None,
    )
    try:
        rec = insert_record(
            kind=kind,
            title=title,
            body=body,
            provenance=provenance,
            status=decision.status,
            namespace_id=namespace,
            source_episode_id=current_episode_id(),
            contact_tag=provenance,
            record_id=rid,
            db_path=db_path,
        )
    except ValueError as exc:
        return f"Error: {exc}"
    return json.dumps(
        {"id": rec["id"], "status": rec["status"], "admission": decision.reason},
        indent=2,
    )


def _search(args: dict[str, Any], *, db_path) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "Error: query is empty."
    top_k = int(args.get("top_k") or DEFAULT_MAX_RECORDS)
    kinds = DEFAULT_RETRIEVE_KINDS
    raw_kinds = args.get("kinds")
    if isinstance(raw_kinds, (list, tuple)):
        kinds = [str(part).strip() for part in raw_kinds if str(part).strip()]
    elif raw_kinds:
        kinds = [part.strip() for part in str(raw_kinds).split(",") if part.strip()]
    provenances = None
    if args.get("provenance"):
        provenances = [str(args["provenance"])]
    namespace = current_namespace() or None
    hits = search_records(
        query,
        top_k=top_k,
        kinds=kinds,
        provenances=provenances,
        namespace_id=namespace if namespace != DEFAULT_NAMESPACE_ID else None,
        db_path=db_path,
    )
    if not hits:
        return "No matching active memories."
    return json.dumps(
        [
            {
                "id": h["id"],
                "kind": h["kind"],
                "title": h["title"],
                "provenance": h["provenance"],
                "status": h["status"],
                "body": h["body"],
            }
            for h in hits
        ],
        indent=2,
    )


def _get(args: dict[str, Any], *, db_path) -> str:
    rid = str(args.get("id") or "")
    rec = get_record(rid, db_path=db_path)
    if rec is None:
        return f"Error: no record {rid}"
    return json.dumps(rec, indent=2, default=str)


def _supersede(args: dict[str, Any], *, db_path) -> str:
    rid = str(args.get("id") or "")
    body = args.get("body")
    if not rid or body is None:
        return "Error: id and body are required."
    old = get_record(rid, db_path=db_path)
    if old is None:
        return f"Error: no record {rid}"
    new_title = args.get("title")
    if new_title is None:
        new_title = old["title"]
    else:
        new_title = str(new_title)
    new_prov = str(args.get("provenance") or old["provenance"])
    review_reason = review_write(
        kind=old["kind"],
        title=new_title,
        body=str(body),
        provenance=new_prov,
        db_path=db_path,
    )
    new_id = str(uuid.uuid4())
    decision = admit(
        kind=old["kind"],
        provenance=new_prov,
        explicit=True,
        namespace_id=old["namespace_id"],
        record_id=new_id,
        db_path=db_path,
        force_proposed_reason=review_reason or None,
    )
    try:
        rec = supersede_record(
            rid,
            body=str(body),
            title=new_title,
            provenance=new_prov,
            source_episode_id=current_episode_id(),
            status=decision.status,
            new_id=new_id,
            db_path=db_path,
        )
    except (KeyError, ValueError) as exc:
        return f"Error: {exc}"
    return json.dumps(
        {
            "id": rec["id"],
            "supersedes": rid,
            "status": rec["status"],
            "admission": decision.reason,
        },
        indent=2,
    )


def _deprecate(args: dict[str, Any], *, db_path) -> str:
    rid = str(args.get("id") or "")
    if not rid:
        return "Error: id is required."
    try:
        rec = deprecate_record(rid, reason=args.get("reason"), db_path=db_path)
    except KeyError:
        return f"Error: no record {rid}"
    return json.dumps({"id": rec["id"], "status": rec["status"]}, indent=2)


def _compact(args: dict[str, Any], *, db_path) -> str:
    dry_run = args.get("dry_run", True)
    if dry_run is None:
        dry_run = True
    report = run_compact(db_path=db_path, dry_run=bool(dry_run))
    return json.dumps(_report_payload(report), indent=2)


def _dry_run(args: dict[str, Any]) -> bool:
    dry_run = args.get("dry_run", True)
    if dry_run is None:
        dry_run = True
    return bool(dry_run)


def _compile(args: dict[str, Any], *, db_path) -> str:
    dry_run = _dry_run(args)
    rid = str(args.get("id") or "").strip()
    if rid:
        return _compile_one(rid, dry_run=dry_run, db_path=db_path)
    return _compile_maybe(dry_run=dry_run, db_path=db_path)


def _maybe_compile_candidates(db_path) -> list[str]:
    would_pin: list[str] = []
    for rec in list_records(kinds=["procedure"], statuses=["active"], db_path=db_path):
        rid = rec["id"]
        if get_compiled(rid, db_path=db_path) is not None:
            continue
        hits = procedure_hits(rid, db_path=db_path)
        if is_compile_candidate(rec, hits=hits, explicit=False):
            would_pin.append(rid)
    return would_pin


def _compile_maybe(*, dry_run: bool, db_path) -> str:
    if dry_run:
        return json.dumps(
            {"dry_run": True, "would_pin": _maybe_compile_candidates(db_path)},
            indent=2,
        )
    return json.dumps(
        {"dry_run": False, "pinned": maybe_compile(db_path=db_path)},
        indent=2,
    )


def _compile_one(rid: str, *, dry_run: bool, db_path) -> str:
    rec = get_record(rid, db_path=db_path)
    if rec is None:
        return f"Error: no record {rid}"
    hits = procedure_hits(rid, db_path=db_path)
    existing = get_compiled(rid, db_path=db_path)
    eligible = is_compile_candidate(rec, hits=hits, explicit=True)
    if dry_run:
        return json.dumps(
            {
                "dry_run": True,
                "id": rid,
                "hits": hits,
                "eligible": eligible,
                "compiled": existing is not None,
                "explicit": bool(existing["explicit"]) if existing else False,
            },
            indent=2,
        )
    try:
        row = pin_compiled(rid, explicit=True, db_path=db_path)
    except ValueError as exc:
        return f"Error: {exc}"
    return json.dumps(
        {
            "dry_run": False,
            "id": rid,
            "hits": hits,
            "pinned": row["source_record_id"],
            "explicit": bool(row["explicit"]),
        },
        indent=2,
    )


def _report_payload(report: CompactReport) -> dict[str, Any]:
    return {
        "emptied": report.emptied,
        "deduped": [list(pair) for pair in report.deduped],
        "folded": report.folded,
        "dry_run": report.dry_run,
    }
