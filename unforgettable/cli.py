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

"""Inspect a local memory.db. argparse / stdlib only. No Studio imports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from unforgettable.constants import KINDS, STATUSES
from unforgettable.eyes.gate import contradictions
from unforgettable.eyes.probes import list_probes, run_probes
from unforgettable.store.compact import run_compact
from unforgettable.store.db import default_db_path
from unforgettable.store.records import (
    get_record,
    list_admissions,
    list_records,
    list_rollouts,
    set_record_status,
)
from unforgettable.store.search import search_records

DEFAULT_SEARCH_TOP = 20
DEFAULT_LIST_LIMIT = 20
DEFAULT_SEARCH_STATUS = "active"
DEFAULT_LIST_STATUS = "all"
DB_ENV_NAME = "UNFORGETTABLE_DB"
TABLE_ID_CHARS = 8
CLI_ADMIT_REASON = "cli admit"
CLI_REJECT_REASON = "cli reject"
UNKNOWN_ID_EXIT = 2
STATUS_ALL = "all"

STUDIO_DB_HELP = (
    'Studio operators can pass --db "$STUDIO_HOME/memory/memory.db" '
    f"(or set {DB_ENV_NAME})."
)
COMPACT_FIRST_DRY_RUN_HELP = (
    "First compact on an existing $STUDIO_HOME/memory/memory.db should be "
    "compact --dry-run."
)


def resolve_db_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = (os.environ.get(DB_ENV_NAME) or "").strip()
    if env:
        return Path(env).expanduser()
    return default_db_path()


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def _print_aligned(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    if not rows:
        return
    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt(parts: Iterable[str]) -> str:
        return "  ".join(part.ljust(widths[i]) for i, part in enumerate(parts))

    print(fmt(headers))
    print(fmt("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def _print_table(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    _print_aligned(
        ("id", "kind", "status", "provenance", "title"),
        [
            (
                rec["id"][:TABLE_ID_CHARS],
                rec["kind"],
                rec["status"],
                rec["provenance"],
                rec["title"],
            )
            for rec in records
        ],
    )


def _print_probe_table(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    _print_aligned(
        ("id", "title", "command"),
        [
            (
                rec["id"][:TABLE_ID_CHARS],
                rec["title"],
                rec.get("command") or "",
            )
            for rec in records
        ],
    )


def _unknown_id(record_id: str) -> int:
    print(f"unknown id: {record_id}", file=sys.stderr)
    return UNKNOWN_ID_EXIT


def _kind_filter(kind: str | None) -> Optional[list[str]]:
    if not kind:
        return None
    return [kind]


def _list_statuses(status: str) -> Optional[list[str]]:
    if status == STATUS_ALL:
        return None
    return [status]


def _search_statuses(status: str) -> Optional[list[str]]:
    if status == STATUS_ALL:
        return list(STATUSES)
    return [status]


def _cmd_path(_args: argparse.Namespace, db_path: Path) -> int:
    print(str(db_path.expanduser().resolve()))
    return 0


def _cmd_search(args: argparse.Namespace, db_path: Path) -> int:
    hits = search_records(
        args.query,
        top_k=args.top,
        kinds=_kind_filter(args.kind),
        statuses=_search_statuses(args.status),
        db_path=db_path,
    )
    _print_table(hits)
    return 0


def _cmd_get(args: argparse.Namespace, db_path: Path) -> int:
    rec = get_record(args.id, db_path=db_path)
    if rec is None:
        return _unknown_id(args.id)
    payload: dict[str, Any] = dict(rec)
    if rec.get("kind") == "episode":
        episode_id = rec.get("source_episode_id") or rec["id"]
        payload["rollouts"] = list_rollouts(episode_id=episode_id, db_path=db_path)
    _print_json(payload)
    return 0


def _cmd_list(args: argparse.Namespace, db_path: Path) -> int:
    rows = list_records(
        kinds=_kind_filter(args.kind),
        statuses=_list_statuses(args.status),
        limit=args.limit,
        db_path=db_path,
    )
    _print_table(rows)
    return 0


def _cmd_admissions(args: argparse.Namespace, db_path: Path) -> int:
    rows = list_admissions(
        limit=args.limit,
        decision=args.decision,
        db_path=db_path,
    )
    _print_json(rows)
    return 0


def _cmd_contradictions(_args: argparse.Namespace, db_path: Path) -> int:
    rows = contradictions(db_path=db_path)
    _print_json(
        [
            {
                "title_key": item.title_key,
                "record_ids": list(item.record_ids),
                "reason": item.reason,
            }
            for item in rows
        ]
    )
    return 0


def _cmd_admit(args: argparse.Namespace, db_path: Path) -> int:
    try:
        rec = set_record_status(
            args.id, "active", reason=CLI_ADMIT_REASON, db_path=db_path
        )
    except KeyError:
        return _unknown_id(args.id)
    _print_json(rec)
    return 0


def _cmd_reject(args: argparse.Namespace, db_path: Path) -> int:
    reason = args.reason if args.reason else CLI_REJECT_REASON
    try:
        rec = set_record_status(args.id, "rejected", reason=reason, db_path=db_path)
    except KeyError:
        return _unknown_id(args.id)
    _print_json(rec)
    return 0


def _cmd_compact(args: argparse.Namespace, db_path: Path) -> int:
    report = run_compact(db_path, dry_run=args.dry_run)
    _print_json(asdict(report))
    return 0


def _cmd_probes(args: argparse.Namespace, db_path: Path) -> int:
    if not args.run:
        _print_probe_table(list_probes(db_path=db_path))
        return 0
    world = Path(args.world).expanduser() if args.world else Path.cwd()
    results = run_probes(world=world, host=None, db_path=db_path, on_chunk=None)
    if any(row.get("outcome") != "pass" for row in results):
        return 1
    return 0


def _add_db_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        default=None,
        help=(
            f"Path to memory.db (default: ${DB_ENV_NAME}, else "
            "$UNFORGETTABLE_HOME/memory.db or ~/.unforgettable/memory.db). "
            f"{STUDIO_DB_HELP}"
        ),
    )


def _status_choices() -> list[str]:
    return sorted(STATUSES | {STATUS_ALL})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m unforgettable",
        description="Inspect a local Unforgettable memory.db.",
        epilog=STUDIO_DB_HELP,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    path_p = sub.add_parser("path", help="Print the resolved memory.db path.")
    _add_db_flag(path_p)
    path_p.set_defaults(func=_cmd_path)

    search_p = sub.add_parser(
        "search",
        help="FTS search. Every kind unless --kind is set (includes episode).",
    )
    _add_db_flag(search_p)
    search_p.add_argument("query")
    search_p.add_argument("--kind", choices=sorted(KINDS), default=None)
    search_p.add_argument("--top", type=int, default=DEFAULT_SEARCH_TOP)
    search_p.add_argument(
        "--status",
        choices=_status_choices(),
        default=DEFAULT_SEARCH_STATUS,
    )
    search_p.set_defaults(func=_cmd_search)

    get_p = sub.add_parser("get", help="Print one record as JSON.")
    _add_db_flag(get_p)
    get_p.add_argument("id")
    get_p.set_defaults(func=_cmd_get)

    list_p = sub.add_parser("list", help="List records as a compact table.")
    _add_db_flag(list_p)
    list_p.add_argument("--kind", choices=sorted(KINDS), default=None)
    list_p.add_argument(
        "--status",
        choices=_status_choices(),
        default=DEFAULT_LIST_STATUS,
    )
    list_p.add_argument("--limit", type=int, default=DEFAULT_LIST_LIMIT)
    list_p.set_defaults(func=_cmd_list)

    adm_p = sub.add_parser("admissions", help="Print the admissions log as JSON.")
    _add_db_flag(adm_p)
    adm_p.add_argument("--limit", type=int, default=DEFAULT_LIST_LIMIT)
    adm_p.add_argument("--decision", default=None)
    adm_p.set_defaults(func=_cmd_admissions)

    contra_p = sub.add_parser(
        "contradictions",
        help="List same-title active claims with distinct bodies.",
    )
    _add_db_flag(contra_p)
    contra_p.set_defaults(func=_cmd_contradictions)

    admit_p = sub.add_parser("admit", help="Promote a record to active.")
    _add_db_flag(admit_p)
    admit_p.add_argument("id")
    admit_p.set_defaults(func=_cmd_admit)

    reject_p = sub.add_parser("reject", help="Reject a record.")
    _add_db_flag(reject_p)
    reject_p.add_argument("id")
    reject_p.add_argument("--reason", default=None)
    reject_p.set_defaults(func=_cmd_reject)

    compact_p = sub.add_parser(
        "compact",
        help="Hygiene pass (wet). " + COMPACT_FIRST_DRY_RUN_HELP,
        description=(
            "Drop old empty proposed rows, deprecate duplicate notebook titles, "
            "and fold long superseded chains. Mutates unless --dry-run. "
            + COMPACT_FIRST_DRY_RUN_HELP
        ),
        epilog=COMPACT_FIRST_DRY_RUN_HELP,
    )
    _add_db_flag(compact_p)
    compact_p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Preview without mutating. "
            + COMPACT_FIRST_DRY_RUN_HELP
        ),
    )
    compact_p.set_defaults(func=_cmd_compact)

    probes_p = sub.add_parser(
        "probes",
        help="List or run active Probe: procedures.",
    )
    _add_db_flag(probes_p)
    probes_p.add_argument(
        "--run",
        action="store_true",
        help="Execute every listed probe in a temp clone of --world.",
    )
    probes_p.add_argument(
        "--world",
        default=None,
        help="World tree to clone when running probes (default: cwd).",
    )
    probes_p.set_defaults(func=_cmd_probes)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = resolve_db_path(args.db)
    return args.func(args, db_path)
