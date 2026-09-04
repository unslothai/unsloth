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
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from unforgettable.agents.retriever import DEFAULT_RETRIEVE_KINDS
from unforgettable.constants import KINDS, STATUSES
from unforgettable.eyes.gate import contradictions
from unforgettable.eyes.probes import list_probes, run_probes
from unforgettable.store.compact import run_compact
from unforgettable.store.compile import (
    get_compiled,
    list_compiled,
    unpin_compiled,
)
from unforgettable.store.db import default_db_path
from unforgettable.store.records import (
    ROLLOUT_CONTACTS,
    ROLLOUT_OUTCOMES,
    get_record,
    list_admissions,
    list_inject_stats,
    list_records,
    list_rollouts,
)
from unforgettable.sidecar.adapters import (
    ADAPTER_STATUSES,
    get_adapter,
    list_adapters,
    rollback_adapter,
    set_adapter_gguf_path,
)
from unforgettable.sidecar.eval import eval_adapter
from unforgettable.sidecar.pack import (
    list_packs,
    pack_from_admitted_b,
    pack_is_retrieval_heavy,
)
from unforgettable.sidecar.train import FAKE_BASE_MODEL, FakeTrainBackend, train_pack
from unforgettable.store.search import search_records
from unforgettable.operators import (
    CLI_ADMIT_REASON,
    CLI_REJECT_REASON,
    ERROR_BLOCKED,
    ERROR_INVALID,
    ERROR_NO_HOST,
    ERROR_REFUSED,
    ERROR_UNKNOWN,
    ERROR_VOTER_OFF,
    admit_record,
    compile_record,
    mine_store,
    promote_adapter_record,
    reject_record,
    review_proposed,
)
from unforgettable.supervisor import config_from_env, resolve_supervisor_host

DEFAULT_SEARCH_TOP = 20
DEFAULT_LIST_LIMIT = 20
DEFAULT_SEARCH_STATUS = "active"
DEFAULT_LIST_STATUS = "all"
DB_ENV_NAME = "UNFORGETTABLE_DB"
TABLE_ID_CHARS = 8
CLI_ROLLOUT_SUMMARY_CHARS = 60
CLI_ADAPTER_PATH_CHARS = 56
TRAIN_RECIPES = ("sft", "distill", "preference")
TRAIN_BACKENDS = ("fake", "unsloth")
UNSLOTH_BASE_REQUIRED = "--base is required when --backend is unsloth"
UNKNOWN_ID_EXIT = 2
APPLY_CONFLICT_EXIT = 2
MISSING_DB_EXIT = 2
ADMIT_STATUS_REFUSED = "admit refused: status is {status} (use --force)"
VOTER_DENIED = "refused: voter deny: {reason}"
MINE_NEEDS_VOTER = "mine requires UNFORGETTABLE_VOTER=advisory|binding"
APPLY_CONFLICT = "cannot combine --apply and --dry-run"
MISSING_DB = "memory.db not found: {path}"
STATUS_ALL = "all"

STUDIO_DB_HELP = (
    'Studio operators can pass --db "$STUDIO_HOME/memory/memory.db" ' f"(or set {DB_ENV_NAME})."
)
COMPACT_FIRST_DRY_RUN_HELP = (
    "compact previews by default; compact --apply mutates $STUDIO_HOME/memory/memory.db."
)
PACK_FIRST_DRY_RUN_HELP = "pack previews by default; pack --apply inserts."


def resolve_db_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = (os.environ.get(DB_ENV_NAME) or "").strip()
    if env:
        return Path(env).expanduser()
    studio = (os.environ.get("STUDIO_HOME") or "").strip()
    if studio:
        candidate = Path(studio).expanduser() / "memory" / "memory.db"
        if candidate.is_file() or candidate.parent.is_dir():
            return candidate
    return default_db_path()


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent = 2, default = str))


def _print_aligned(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    if not rows:
        return
    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]

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
    print(f"unknown id: {record_id}", file = sys.stderr)
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
    kinds = _kind_filter(args.kind)
    if kinds is None:
        kinds = list(DEFAULT_RETRIEVE_KINDS)
    hits = search_records(
        args.query,
        top_k = args.top,
        kinds = kinds,
        statuses = _search_statuses(args.status),
        db_path = db_path,
    )
    _print_table(hits)
    return 0


def _cmd_get(args: argparse.Namespace, db_path: Path) -> int:
    rec = get_record(args.id, db_path = db_path)
    if rec is None:
        return _unknown_id(args.id)
    payload: dict[str, Any] = dict(rec)
    if rec.get("kind") == "episode":
        episode_id = rec.get("source_episode_id") or rec["id"]
        payload["rollouts"] = list_rollouts(episode_id = episode_id, db_path = db_path)
    _print_json(payload)
    return 0


def _cmd_list(args: argparse.Namespace, db_path: Path) -> int:
    rows = list_records(
        kinds = _kind_filter(args.kind),
        statuses = _list_statuses(args.status),
        limit = args.limit,
        db_path = db_path,
    )
    _print_table(rows)
    return 0


def _cmd_admissions(args: argparse.Namespace, db_path: Path) -> int:
    rows = list_admissions(
        limit = args.limit,
        decision = args.decision,
        db_path = db_path,
    )
    _print_json(rows)
    return 0


def _cmd_contradictions(_args: argparse.Namespace, db_path: Path) -> int:
    rows = contradictions(db_path = db_path)
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


def _supervisor_host():
    return resolve_supervisor_host()


def _print_vote(outcome) -> None:
    if outcome.vote is None:
        return
    print(
        f"voter {outcome.vote.decision}: {outcome.vote.reason}",
        file = sys.stderr,
    )


def _cmd_admit(args: argparse.Namespace, db_path: Path) -> int:
    outcome = admit_record(
        args.id,
        force = args.force,
        db_path = db_path,
        host = _supervisor_host(),
        reason = CLI_ADMIT_REASON,
    )
    _print_vote(outcome)
    if outcome.error_kind == ERROR_UNKNOWN:
        return _unknown_id(args.id)
    if outcome.error_kind == ERROR_REFUSED:
        detail = outcome.error_detail or ""
        if detail.startswith("dissonance:"):
            print(f"admit refused: {detail} (use --force)", file = sys.stderr)
        else:
            print(
                ADMIT_STATUS_REFUSED.format(status = detail),
                file = sys.stderr,
            )
        return UNKNOWN_ID_EXIT
    if outcome.error_kind == ERROR_BLOCKED:
        print(VOTER_DENIED.format(reason = outcome.error_detail), file = sys.stderr)
        return UNKNOWN_ID_EXIT
    if not outcome.ok or outcome.record is None:
        return UNKNOWN_ID_EXIT
    _print_json(outcome.record)
    return 0


def _cmd_reject(args: argparse.Namespace, db_path: Path) -> int:
    reason = args.reason if args.reason else CLI_REJECT_REASON
    outcome = reject_record(args.id, reason = reason, db_path = db_path)
    if outcome.error_kind == ERROR_UNKNOWN:
        return _unknown_id(args.id)
    if not outcome.ok or outcome.record is None:
        return UNKNOWN_ID_EXIT
    _print_json(outcome.record)
    return 0


def _dry_run_from_apply(args: argparse.Namespace) -> bool | None:
    apply = bool(getattr(args, "apply", False))
    dry_run = bool(getattr(args, "dry_run", False))
    if apply and dry_run:
        return None
    return not apply


def _cmd_compact(args: argparse.Namespace, db_path: Path) -> int:
    dry_run = _dry_run_from_apply(args)
    if dry_run is None:
        print(APPLY_CONFLICT, file = sys.stderr)
        return APPLY_CONFLICT_EXIT
    older = getattr(args, "older_than", None)
    report = run_compact(db_path, dry_run = dry_run, older_than_days = older)
    _print_json(asdict(report))
    return 0


def _cmd_compiled(_args: argparse.Namespace, db_path: Path) -> int:
    rows = list_compiled(db_path = db_path)
    _print_aligned(
        ("id", "hits", "explicit", "title"),
        [
            (
                rec["id"][:TABLE_ID_CHARS],
                str(rec.get("hits") or 0),
                "yes" if rec.get("explicit") else "no",
                rec["title"],
            )
            for rec in rows
        ],
    )
    return 0


def _cmd_compile(args: argparse.Namespace, db_path: Path) -> int:
    outcome = compile_record(
        args.id,
        db_path = db_path,
        host = _supervisor_host(),
    )
    _print_vote(outcome)
    if outcome.error_kind == ERROR_UNKNOWN:
        return _unknown_id(args.id)
    if outcome.error_kind == ERROR_BLOCKED:
        print(VOTER_DENIED.format(reason = outcome.error_detail), file = sys.stderr)
        return UNKNOWN_ID_EXIT
    if outcome.error_kind == ERROR_INVALID:
        print(outcome.error_detail, file = sys.stderr)
        return UNKNOWN_ID_EXIT
    if not outcome.ok or outcome.record is None:
        return UNKNOWN_ID_EXIT
    _print_json(outcome.record)
    return 0


def _cmd_uncompile(args: argparse.Namespace, db_path: Path) -> int:
    row = get_compiled(args.id, db_path = db_path)
    if row is None:
        return _unknown_id(args.id)
    unpin_compiled(args.id, db_path = db_path)
    _print_json(row)
    return 0


def _clip_rollout_summary(text: str) -> str:
    line = (text or "").replace("\n", " ").strip()
    if len(line) <= CLI_ROLLOUT_SUMMARY_CHARS:
        return line
    return line[: CLI_ROLLOUT_SUMMARY_CHARS - 3] + "..."


def _cmd_rollouts(args: argparse.Namespace, db_path: Path) -> int:
    rows = list_rollouts(
        contact = args.contact,
        outcome = args.outcome,
        limit = args.limit,
        db_path = db_path,
    )
    _print_aligned(
        ("episode", "contact", "outcome", "summary"),
        [
            (
                (row.get("episode_id") or "")[:TABLE_ID_CHARS],
                row["contact"],
                row["outcome"],
                _clip_rollout_summary(row.get("summary") or ""),
            )
            for row in rows
        ],
    )
    return 0


def _csv_count(value: str | None) -> int:
    if not value:
        return 0
    return len([part for part in str(value).split(",") if part])


def _cmd_pack(args: argparse.Namespace, db_path: Path) -> int:
    dry_run = _dry_run_from_apply(args)
    if dry_run is None:
        print(APPLY_CONFLICT, file = sys.stderr)
        return APPLY_CONFLICT_EXIT
    if not dry_run:
        print(f"using {db_path}", file = sys.stderr)
    report = pack_from_admitted_b(
        include_sim = args.include_sim,
        dry_run = dry_run,
        db_path = db_path,
    )
    _print_json(asdict(report))
    return 0


def _cmd_packs(args: argparse.Namespace, db_path: Path) -> int:
    rows = list_packs(limit = args.limit, db_path = db_path)
    _print_aligned(
        ("id", "n_train", "n_holdout", "include_sim", "created"),
        [
            (
                rec["id"][:TABLE_ID_CHARS],
                str(rec.get("n_train") or 0),
                str(rec.get("n_holdout") or 0),
                "yes" if rec.get("include_sim") else "no",
                rec.get("created_at") or "",
            )
            for rec in rows
        ],
    )
    return 0


def _default_train_backend() -> str:
    return "unsloth" if importlib.util.find_spec("unsloth") else "fake"


def _clip_adapter_path(path: str) -> str:
    text = path or ""
    if len(text) <= CLI_ADAPTER_PATH_CHARS:
        return text
    return "..." + text[-(CLI_ADAPTER_PATH_CHARS - 3) :]


def _cmd_train(args: argparse.Namespace, db_path: Path) -> int:
    print(f"using {db_path}", file = sys.stderr)
    backend_name = args.backend or _default_train_backend()
    if backend_name == "unsloth":
        if not (args.base or "").strip():
            print(UNSLOTH_BASE_REQUIRED, file = sys.stderr)
            return UNKNOWN_ID_EXIT
        from unforgettable.sidecar.train import UnslothTrainBackend

        backend = UnslothTrainBackend(base_model = args.base)
        base_model = args.base
    else:
        backend = FakeTrainBackend()
        base_model = args.base or FAKE_BASE_MODEL
    if args.pack:
        pack_id = args.pack
    else:
        packs = list_packs(limit = 1, db_path = db_path)
        if not packs:
            print("no packs; run pack first", file = sys.stderr)
            return UNKNOWN_ID_EXIT
        pack_id = packs[0]["id"]
    recipe = args.recipe
    if recipe is None:
        recipe = "distill" if pack_is_retrieval_heavy(db_path) else "sft"
    try:
        result = train_pack(
            pack_id,
            backend = backend,
            base_model = base_model,
            recipe = recipe,
            db_path = db_path,
            export_gguf = not bool(getattr(args, "no_gguf", False)),
        )
    except KeyError:
        return _unknown_id(pack_id)
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        print(str(exc), file = sys.stderr)
        return UNKNOWN_ID_EXIT
    _print_json(asdict(result))
    return 0


def _cmd_adapters(args: argparse.Namespace, db_path: Path) -> int:
    rows = list_adapters(status = args.status, db_path = db_path)
    _print_aligned(
        ("id", "status", "recipe", "backend", "pack", "path"),
        [
            (
                rec["id"][:TABLE_ID_CHARS],
                rec["status"],
                rec.get("recipe") or "",
                rec.get("backend") or "",
                (rec.get("pack_id") or "")[:TABLE_ID_CHARS],
                _clip_adapter_path(rec.get("path") or ""),
            )
            for rec in rows
        ],
    )
    return 0


def _cmd_export_gguf(args: argparse.Namespace, db_path: Path) -> int:
    adapter = get_adapter(args.id, db_path = db_path)
    if adapter is None:
        return _unknown_id(args.id)
    from unforgettable.sidecar.export_gguf import export_adapter_gguf

    try:
        path = export_adapter_gguf(adapter.get("path"), base_model = adapter.get("base_model"))
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file = sys.stderr)
        return UNKNOWN_ID_EXIT
    set_adapter_gguf_path(args.id, path, db_path = db_path)
    _print_json({"adapter_id": args.id, "gguf_path": path})
    return 0


def _cmd_eval(args: argparse.Namespace, db_path: Path) -> int:
    adapter = get_adapter(args.id, db_path = db_path)
    if adapter is None:
        return _unknown_id(args.id)
    world = Path(args.world).expanduser() if args.world else None
    if adapter.get("backend") == "unsloth":
        from unforgettable.sidecar.train import UnslothTrainBackend
        backend = UnslothTrainBackend(base_model = adapter.get("base_model"))
    else:
        backend = FakeTrainBackend()
    try:
        report = eval_adapter(
            args.id,
            backend = backend,
            world = world,
            db_path = db_path,
            host = _supervisor_host(),
            config = config_from_env(),
        )
    except KeyError:
        return _unknown_id(args.id)
    _print_json(asdict(report))
    return 0 if report.passed else 1


def _cmd_promote(args: argparse.Namespace, db_path: Path) -> int:
    outcome = promote_adapter_record(
        args.id,
        force = args.force,
        db_path = db_path,
        host = _supervisor_host(),
    )
    _print_vote(outcome)
    if outcome.error_kind == ERROR_UNKNOWN:
        return _unknown_id(args.id)
    if outcome.error_kind == ERROR_BLOCKED:
        print(VOTER_DENIED.format(reason = outcome.error_detail), file = sys.stderr)
        return UNKNOWN_ID_EXIT
    if outcome.error_kind == ERROR_INVALID:
        print(outcome.error_detail, file = sys.stderr)
        return UNKNOWN_ID_EXIT
    if not outcome.ok or outcome.record is None:
        return UNKNOWN_ID_EXIT
    _print_json(outcome.record)
    return 0


def _cmd_review(args: argparse.Namespace, db_path: Path) -> int:
    outcome = review_proposed(
        apply = args.apply,
        limit = args.limit,
        db_path = db_path,
        host = _supervisor_host(),
    )
    if outcome.error_kind == ERROR_VOTER_OFF:
        print("voter off; set UNFORGETTABLE_VOTER=advisory|binding", file = sys.stderr)
        return UNKNOWN_ID_EXIT
    _print_json(outcome.items or [])
    return 0


def _cmd_mine(args: argparse.Namespace, db_path: Path) -> int:
    outcome = mine_store(
        apply = args.apply,
        limit = args.limit,
        db_path = db_path,
        host = _supervisor_host(),
    )
    if outcome.error_kind == ERROR_VOTER_OFF:
        print(MINE_NEEDS_VOTER, file = sys.stderr)
        return UNKNOWN_ID_EXIT
    if outcome.error_kind == ERROR_NO_HOST:
        print("mine needs UNFORGETTABLE_SUPERVISOR_URL", file = sys.stderr)
        return UNKNOWN_ID_EXIT
    _print_json(outcome.items or [])
    return 0


def _cmd_rollback(_args: argparse.Namespace, db_path: Path) -> int:
    row = rollback_adapter(db_path = db_path)
    if row is None:
        _print_json({"promoted": None})
        return 0
    _print_json(row)
    return 0


def _cmd_load(args: argparse.Namespace, db_path: Path) -> int:
    rows = list_inject_stats(limit = args.limit, db_path = db_path)
    _print_aligned(
        ("episode", "contact", "standing", "retrieve", "traj", "total", "n_compiled"),
        [
            (
                rec["episode_id"][:TABLE_ID_CHARS],
                rec["contact"],
                str(rec.get("standing_chars") or 0),
                str(rec.get("retrieve_chars") or 0),
                str(rec.get("trajectory_chars") or 0),
                str(rec.get("total_chars") or 0),
                str(_csv_count(rec.get("compiled_ids"))),
            )
            for rec in rows
        ],
    )
    return 0


def _cmd_probes(args: argparse.Namespace, db_path: Path) -> int:
    if not args.run:
        _print_probe_table(list_probes(db_path = db_path))
        return 0
    world = Path(args.world).expanduser() if args.world else Path.cwd()
    results = run_probes(world = world, host = None, db_path = db_path, on_chunk = None)
    if any(row.get("outcome") != "pass" for row in results):
        return 1
    return 0


def _add_db_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        default = None,
        help = (
            f"Path to memory.db (default: ${DB_ENV_NAME}, else "
            "$UNFORGETTABLE_HOME/memory.db or ~/.unforgettable/memory.db). "
            f"{STUDIO_DB_HELP}"
        ),
    )


def _status_choices() -> list[str]:
    return sorted(STATUSES | {STATUS_ALL})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog = "python -m unforgettable",
        description = "Inspect a local Unforgettable memory.db.",
        epilog = STUDIO_DB_HELP,
    )
    sub = parser.add_subparsers(dest = "command", required = True)

    path_p = sub.add_parser("path", help = "Print the resolved memory.db path.")
    _add_db_flag(path_p)
    path_p.set_defaults(func = _cmd_path)

    search_p = sub.add_parser(
        "search",
        help = "FTS search. Default kinds exclude episode (pass --kind episode).",
    )
    _add_db_flag(search_p)
    search_p.add_argument("query")
    search_p.add_argument("--kind", choices = sorted(KINDS), default = None)
    search_p.add_argument("--top", type = int, default = DEFAULT_SEARCH_TOP)
    search_p.add_argument(
        "--status",
        choices = _status_choices(),
        default = DEFAULT_SEARCH_STATUS,
    )
    search_p.set_defaults(func = _cmd_search)

    get_p = sub.add_parser("get", help = "Print one record as JSON.")
    _add_db_flag(get_p)
    get_p.add_argument("id")
    get_p.set_defaults(func = _cmd_get)

    list_p = sub.add_parser("list", help = "List records as a compact table.")
    _add_db_flag(list_p)
    list_p.add_argument("--kind", choices = sorted(KINDS), default = None)
    list_p.add_argument(
        "--status",
        choices = _status_choices(),
        default = DEFAULT_LIST_STATUS,
    )
    list_p.add_argument("--limit", type = int, default = DEFAULT_LIST_LIMIT)
    list_p.set_defaults(func = _cmd_list)

    adm_p = sub.add_parser("admissions", help = "Print the admissions log as JSON.")
    _add_db_flag(adm_p)
    adm_p.add_argument("--limit", type = int, default = DEFAULT_LIST_LIMIT)
    adm_p.add_argument("--decision", default = None)
    adm_p.set_defaults(func = _cmd_admissions)

    contra_p = sub.add_parser(
        "contradictions",
        help = "List same-title active claims with distinct bodies.",
    )
    _add_db_flag(contra_p)
    contra_p.set_defaults(func = _cmd_contradictions)

    admit_p = sub.add_parser("admit", help = "Promote a proposed or deprecated record to active.")
    _add_db_flag(admit_p)
    admit_p.add_argument("id")
    admit_p.add_argument(
        "--force",
        action = "store_true",
        help = "Allow admit from rejected, superseded, or already-active.",
    )
    admit_p.set_defaults(func = _cmd_admit)

    reject_p = sub.add_parser("reject", help = "Reject a record.")
    _add_db_flag(reject_p)
    reject_p.add_argument("id")
    reject_p.add_argument("--reason", default = None)
    reject_p.set_defaults(func = _cmd_reject)

    review_p = sub.add_parser(
        "review",
        help = "Ask the approval voter about proposed records (preview). Pass --apply to mutate.",
    )
    _add_db_flag(review_p)
    review_p.add_argument("--limit", type = int, default = DEFAULT_LIST_LIMIT)
    review_p.add_argument(
        "--apply",
        action = "store_true",
        help = "Admit voter-allow rows and reject voter-deny rows.",
    )
    review_p.set_defaults(func = _cmd_review)

    mine_p = sub.add_parser(
        "mine",
        help = "Batch voter over proposed rows, rollouts, and the admissions log.",
    )
    _add_db_flag(mine_p)
    mine_p.add_argument("--limit", type = int, default = DEFAULT_LIST_LIMIT)
    mine_p.add_argument(
        "--apply",
        action = "store_true",
        help = "Apply allow/deny on existing ids; insert new drafts as proposed infer.",
    )
    mine_p.set_defaults(func = _cmd_mine)

    compact_p = sub.add_parser(
        "compact",
        help = "Hygiene pass (preview). Pass --apply to mutate. " + COMPACT_FIRST_DRY_RUN_HELP,
        description = (
            "Drop old empty proposed rows, deprecate duplicate notebook titles, "
            "and fold long superseded chains. Preview unless --apply. " + COMPACT_FIRST_DRY_RUN_HELP
        ),
        epilog = COMPACT_FIRST_DRY_RUN_HELP,
    )
    _add_db_flag(compact_p)
    compact_p.add_argument(
        "--dry-run",
        action = "store_true",
        help = ("Preview without mutating (default). " + COMPACT_FIRST_DRY_RUN_HELP),
    )
    compact_p.add_argument(
        "--apply",
        action = "store_true",
        help = "Mutate memory.db. Refused when combined with --dry-run.",
    )
    compact_p.add_argument(
        "--older-than",
        type = int,
        default = None,
        metavar = "DAYS",
        help = "Reject stale proposed WHO/infer rows older than DAYS (default 30).",
    )
    compact_p.set_defaults(func = _cmd_compact)

    probes_p = sub.add_parser(
        "probes",
        help = "List or run active Probe: procedures.",
    )
    _add_db_flag(probes_p)
    probes_p.add_argument(
        "--run",
        action = "store_true",
        help = "Execute every listed probe in a temp clone of --world.",
    )
    probes_p.add_argument(
        "--world",
        default = None,
        help = "World tree to clone when running probes (default: cwd).",
    )
    probes_p.set_defaults(func = _cmd_probes)

    compiled_p = sub.add_parser(
        "compiled",
        help = "List procedures in the standing compile cache.",
    )
    _add_db_flag(compiled_p)
    compiled_p.set_defaults(func = _cmd_compiled)

    compile_p = sub.add_parser(
        "compile",
        help = "Pin an admitted procedure into the standing prompt cache.",
    )
    _add_db_flag(compile_p)
    compile_p.add_argument("id")
    compile_p.set_defaults(func = _cmd_compile)

    uncompile_p = sub.add_parser(
        "uncompile",
        help = "Drop a procedure from the standing compile cache.",
    )
    _add_db_flag(uncompile_p)
    uncompile_p.add_argument("id")
    uncompile_p.set_defaults(func = _cmd_uncompile)

    rollouts_p = sub.add_parser(
        "rollouts",
        help = "List graded world and sim rollouts.",
    )
    _add_db_flag(rollouts_p)
    rollouts_p.add_argument(
        "--contact",
        choices = sorted(ROLLOUT_CONTACTS),
        default = None,
    )
    rollouts_p.add_argument(
        "--outcome",
        choices = sorted(ROLLOUT_OUTCOMES),
        default = None,
    )
    rollouts_p.add_argument("--limit", type = int, default = DEFAULT_LIST_LIMIT)
    rollouts_p.set_defaults(func = _cmd_rollouts)

    load_p = sub.add_parser(
        "load",
        help = "Print inject char splits (standing / retrieve / traj / total).",
    )
    _add_db_flag(load_p)
    load_p.add_argument("--limit", type = int, default = DEFAULT_LIST_LIMIT)
    load_p.set_defaults(func = _cmd_load)

    pack_p = sub.add_parser(
        "pack",
        help = "Build a PEFT pack from admitted B (preview). Pass --apply to insert. "
        + PACK_FIRST_DRY_RUN_HELP,
        description = (
            "Pack admitted procedure/error_fix bodies. World-pass traces vote; "
            "they are not text sources. Preview unless --apply. " + PACK_FIRST_DRY_RUN_HELP
        ),
        epilog = PACK_FIRST_DRY_RUN_HELP,
    )
    _add_db_flag(pack_p)
    pack_p.add_argument(
        "--dry-run",
        action = "store_true",
        help = "Preview without inserting (default). " + PACK_FIRST_DRY_RUN_HELP,
    )
    pack_p.add_argument(
        "--apply",
        action = "store_true",
        help = "Insert a pack. Refused when combined with --dry-run.",
    )
    pack_p.add_argument(
        "--include-sim",
        action = "store_true",
        help = "Allow sim/pass votes only when the same episode also has world/pass and no twin_note.",
    )
    pack_p.set_defaults(func = _cmd_pack)

    packs_p = sub.add_parser("packs", help = "List built packs as a compact table.")
    _add_db_flag(packs_p)
    packs_p.add_argument("--limit", type = int, default = DEFAULT_LIST_LIMIT)
    packs_p.set_defaults(func = _cmd_packs)

    train_p = sub.add_parser(
        "train",
        help = (
            "Train a shadow adapter from a pack. Unsloth uses "
            "FastModel.from_pretrained (falls back to FastLanguageModel), "
            "get_peft_model, and SFTTrainer or DPOTrainer (--recipe preference)."
        ),
    )
    _add_db_flag(train_p)
    train_p.add_argument("--pack", default = None, help = "Pack id (default: latest pack).")
    train_p.add_argument(
        "--backend",
        choices = TRAIN_BACKENDS,
        default = None,
        help = (
            "Training backend (default: unsloth if importable, else fake). "
            "unsloth calls FastModel.from_pretrained (falls back to "
            "FastLanguageModel), get_peft_model, and SFTTrainer or DPOTrainer."
        ),
    )
    train_p.add_argument(
        "--base",
        default = None,
        help = "Base model id. Required for --backend unsloth; defaults to fake for fake.",
    )
    train_p.add_argument(
        "--recipe",
        choices = TRAIN_RECIPES,
        default = None,
        help = "Train recipe (default: distill if retrieval-heavy, else sft).",
    )
    train_p.add_argument(
        "--no-gguf",
        action = "store_true",
        help = "Skip GGUF LoRA export after Unsloth train. PEFT dir is still written.",
    )
    train_p.set_defaults(func = _cmd_train)

    export_gguf_p = sub.add_parser(
        "export-gguf",
        help = (
            "Convert a PEFT adapter directory to a GGUF LoRA "
            "(llama.cpp --lora). Does not load the adapter onto a running server."
        ),
    )
    _add_db_flag(export_gguf_p)
    export_gguf_p.add_argument("id")
    export_gguf_p.set_defaults(func = _cmd_export_gguf)

    adapters_p = sub.add_parser(
        "adapters",
        help = "List adapters as a compact table.",
    )
    _add_db_flag(adapters_p)
    adapters_p.add_argument(
        "--status",
        choices = sorted(ADAPTER_STATUSES),
        default = None,
    )
    adapters_p.set_defaults(func = _cmd_adapters)

    eval_p = sub.add_parser(
        "eval",
        help = "Score a shadow adapter on holdout lean vs base.",
    )
    _add_db_flag(eval_p)
    eval_p.add_argument("id")
    eval_p.add_argument(
        "--world",
        default = None,
        help = "World tree to clone when running Probe: procedures.",
    )
    eval_p.set_defaults(func = _cmd_eval)

    promote_p = sub.add_parser(
        "promote",
        help = "Promote a shadow adapter. Refuses without eval metrics unless --force.",
    )
    _add_db_flag(promote_p)
    promote_p.add_argument("id")
    promote_p.add_argument(
        "--force",
        action = "store_true",
        help = "Skip the eval gate (promote without adapter_lean metrics).",
    )
    promote_p.set_defaults(func = _cmd_promote)

    rollback_p = sub.add_parser(
        "rollback",
        help = "Discard the current promoted adapter. Does not delete files.",
    )
    _add_db_flag(rollback_p)
    rollback_p.set_defaults(func = _cmd_rollback)

    return parser


_NEED_EXISTING_DB = frozenset({"pack", "train", "eval", "promote", "rollback", "export-gguf"})


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = resolve_db_path(args.db)
    needs_file = args.command in _NEED_EXISTING_DB or (
        args.command == "compact" and bool(getattr(args, "apply", False))
    )
    if needs_file and not db_path.expanduser().is_file():
        print(MISSING_DB.format(path = db_path), file = sys.stderr)
        return MISSING_DB_EXIT
    return args.func(args, db_path)
