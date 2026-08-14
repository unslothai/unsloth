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
from typing import Any, Optional

from unforgettable.agents.admissions import admit
from unforgettable.constants import DEFAULT_NAMESPACE_ID
from unforgettable.loop.runtime import current_db_path, current_episode_id, current_namespace
from unforgettable.store.records import (
    deprecate_record,
    get_record,
    insert_record,
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
    decision = admit(
        kind=kind,
        provenance=provenance,
        explicit=True,
        namespace_id=namespace,
        db_path=db_path,
    )
    rec = insert_record(
        kind=kind,
        title=title,
        body=body,
        provenance=provenance,
        status=decision.status,
        namespace_id=namespace,
        source_episode_id=current_episode_id(),
        contact_tag=provenance,
        db_path=db_path,
    )
    return json.dumps(
        {"id": rec["id"], "status": rec["status"], "admission": decision.reason},
        indent=2,
    )


def _search(args: dict[str, Any], *, db_path) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "Error: query is empty."
    top_k = int(args.get("top_k") or 6)
    kinds = None
    raw_kinds = args.get("kinds")
    if raw_kinds:
        kinds = [part.strip() for part in str(raw_kinds).split(",") if part.strip()]
    provenances = None
    if args.get("provenance"):
        provenances = [str(args["provenance"])]
    hits = search_records(
        query,
        top_k=top_k,
        kinds=kinds,
        provenances=provenances,
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
    try:
        rec = supersede_record(
            rid,
            body=str(body),
            title=args.get("title"),
            provenance=args.get("provenance"),
            source_episode_id=current_episode_id(),
            db_path=db_path,
        )
    except KeyError:
        return f"Error: no record {rid}"
    return json.dumps({"id": rec["id"], "supersedes": rid, "status": rec["status"]}, indent=2)


def _deprecate(args: dict[str, Any], *, db_path) -> str:
    rid = str(args.get("id") or "")
    if not rid:
        return "Error: id is required."
    try:
        rec = deprecate_record(rid, reason=args.get("reason"), db_path=db_path)
    except KeyError:
        return f"Error: no record {rid}"
    return json.dumps({"id": rec["id"], "status": rec["status"]}, indent=2)
