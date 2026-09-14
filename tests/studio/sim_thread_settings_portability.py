# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Portability checks for the per-chat settings column.

`settings_json` is added to an existing `chat_threads` by an idempotent ALTER, and
every Unsloth install that upgrades runs it exactly once against a database it has
been writing to for months. The interesting differences between platforms are the
bundled SQLite, the filesystem and the path handling, none of which CI exercises
today: the chat settings tests only ever run on Linux.

Everything here is stdlib plus `storage.studio_db`, so it runs unchanged on
Windows, macOS and Linux.

    python tests/studio/sim_thread_settings_portability.py
"""

import json
import os
import platform
import sqlite3
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "studio" / "backend"))

FAILURES = []
CHECKS = 0


def check(
    name,
    ok,
    detail = "",
):
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush = True)


def fresh_home():
    """An Unsloth home under a real temp dir, so path handling is the platform's own."""
    home = Path(tempfile.mkdtemp(prefix = "sim8686_"))
    os.environ["UNSLOTH_STUDIO_HOME"] = str(home)
    return home


def thread_row(title, **extra):
    now = int(time.time() * 1000)
    row = {
        "id": str(uuid.uuid4()),
        "title": title,
        "modelType": "base",
        "modelId": "",
        "archived": False,
        "createdAt": now,
        "updatedAt": now,
    }
    row.update(extra)
    return row


def main():
    # The Windows console is cp1252 by default, so printing a kbId containing emoji raises UnicodeEncodeError and fails
    # the run for a reason that has nothing to do with what is being tested.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding = "utf-8", errors = "replace")
        except (AttributeError, ValueError):  # pragma: no cover - older stream types
            pass
    print(f"platform : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"python   : {sys.version.split()[0]}")
    print(f"sqlite3  : {sqlite3.sqlite_version}")
    print()

    # ALTER TABLE ADD COLUMN is SQLite 3.2.0 (2005) and COALESCE predates it, so the floor is far below anything
    # shipping today. Assert it rather than assume it.
    major, minor, _ = (int(p) for p in sqlite3.sqlite_version.split("."))
    check(
        "sqlite supports ALTER TABLE ADD COLUMN (>= 3.2)",
        (major, minor) >= (3, 2),
        f"found {sqlite3.sqlite_version}",
    )
    # UPSERT (ON CONFLICT DO UPDATE), which the thread writer uses, needs 3.24.
    check(
        "sqlite supports UPSERT (>= 3.24)",
        (major, minor) >= (3, 24),
        f"found {sqlite3.sqlite_version}",
    )

    print("\n--- upgrade: a database created before the column ---")
    home = fresh_home()
    import storage.studio_db as db  # noqa: E402 - after UNSLOTH_STUDIO_HOME is set

    # Build the database with the real schema, populate it the way a months-old install would be, then DROP the new
    # column. Hand-writing the old CREATE TABLE drifts from the real one (it is missing pair_id and everything else the
    # schema step indexes), so this is both more faithful and self-maintaining.
    legacy_ids = []
    for i in range(200):
        row = thread_row(f"legacy {i}")
        db.upsert_chat_thread(row)
        legacy_ids.append(row["id"])
    db_path = Path(db.get_db_path()) if hasattr(db, "get_db_path") else home / "studio.db"

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("ALTER TABLE chat_threads DROP COLUMN settings_json")
        conn.commit()
        dropped = True
    except sqlite3.OperationalError as exc:
        # DROP COLUMN is 3.35+. Older SQLite needs the copy-and-rename dance.
        print(f"  (DROP COLUMN unavailable: {exc}; rebuilding the table instead)")
        cols = [
            r[1] for r in conn.execute("PRAGMA table_info(chat_threads)") if r[1] != "settings_json"
        ]
        joined = ", ".join(cols)
        conn.executescript(
            f"""CREATE TABLE chat_threads_old AS SELECT {joined} FROM chat_threads;
                DROP TABLE chat_threads;
                ALTER TABLE chat_threads_old RENAME TO chat_threads;"""
        )
        conn.commit()
        dropped = True
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chat_threads)")}
    conn.close()
    check(
        "the fixture really predates the column",
        dropped and "settings_json" not in cols,
        f"columns={len(cols)}",
    )

    db._schema_ready = False
    got = db.get_chat_thread(legacy_ids[0])
    check(
        "a pre-existing thread still reads after the migration",
        got is not None,
        f"title={got.get('title') if got else None!r}",
    )
    check(
        "its settings are absent, not invented",
        got is not None and got.get("settings") in (None, {}),
        f"settings={got.get('settings') if got else None!r}",
    )
    conn = sqlite3.connect(str(db_path))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chat_threads)")}
    count = conn.execute("SELECT COUNT(*) FROM chat_threads").fetchone()[0]
    conn.close()
    check("the column was added", "settings_json" in cols)
    check("all 200 pre-existing rows survived", count == 200, f"count={count}")

    print("\n--- the migration is idempotent ---")
    for _ in range(3):
        db._schema_ready = False
        db.get_chat_thread(legacy_ids[1])
    check("running the schema step repeatedly is safe", True)

    print("\n--- round trip, including content platforms disagree about ---")
    payloads = {
        "plain": {"toolsEnabled": True, "permissionMode": "ask"},
        "unicode": {"ragSource": {"type": "kb", "kbId": "文書 kb"}},
        "emoji": {"ragSource": {"type": "kb", "kbId": "notes 🧠 v2"}},
        "windows path shaped": {"ragSource": {"type": "kb", "kbId": r"C:\Users\a\kb"}},
        "newlines": {"ragSource": {"type": "kb", "kbId": "a\r\nb"}},
        "float": {"ragAutoInjectMinScore": 0.7},
        "bounds": {"ragTopK": 50},
        "empty": {},
    }
    for name, payload in payloads.items():
        row = thread_row(f"rt {name}")
        db.upsert_chat_thread(row)
        db.update_chat_thread(row["id"], {"settings": payload})
        back = db.get_chat_thread(row["id"]).get("settings")
        if payload == {}:
            check(f"round trip: {name}", back in ({}, None), f"got {back!r}")
        else:
            check(f"round trip: {name}", back == payload, f"got {back!r}")

    print("\n--- a large blob ---")
    row = thread_row("big")
    db.upsert_chat_thread(row)
    big = {"ragSource": {"type": "kb", "kbId": "x" * 200}}
    db.update_chat_thread(row["id"], {"settings": big})
    check("a 200 char kbId round trips", db.get_chat_thread(row["id"]).get("settings") == big)

    print("\n--- the writers that rebuild a row must not clear the snapshot ---")
    row = thread_row("coalesce")
    db.upsert_chat_thread(row)
    db.update_chat_thread(row["id"], {"settings": {"toolsEnabled": True}})
    renamed = dict(row)
    renamed["title"] = "renamed by an autosave"
    db.upsert_chat_thread(renamed)
    after = db.get_chat_thread(row["id"])
    check(
        "a title rewrite leaves the snapshot alone",
        after.get("settings") == {"toolsEnabled": True},
        f"got {after.get('settings')!r}",
    )
    check("and the rewrite did land", after.get("title") == "renamed by an autosave")

    print("\n--- explicit clears ---")
    db.update_chat_thread(row["id"], {"settings": None})
    check(
        "PATCH settings=null clears the column",
        db.get_chat_thread(row["id"]).get("settings") in (None, {}),
    )

    print("\n--- the listing stays free of snapshots ---")
    row = thread_row("listed")
    db.upsert_chat_thread(row)
    db.update_chat_thread(row["id"], {"settings": {"toolsEnabled": True}})
    listed = db.list_chat_threads()
    entries = [t for t in listed if t.get("id") == row["id"]]
    check("the thread is listed", len(entries) == 1)
    check(
        "but carries no snapshot in the listing",
        bool(entries) and not entries[0].get("settings"),
        f"got {entries[0].get('settings') if entries else None!r}",
    )

    print("\n--- corrupt content degrades instead of exploding ---")
    row = thread_row("corrupt")
    db.upsert_chat_thread(row)
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE chat_threads SET settings_json = ? WHERE id = ?", ('{"nope', row["id"]))
    conn.commit()
    conn.close()
    try:
        got = db.get_chat_thread(row["id"])
        check(
            "unparseable JSON reads as no snapshot",
            got.get("settings") in (None, {}),
            f"got {got.get('settings')!r}",
        )
    except Exception as exc:  # noqa: BLE001 - that would be the finding
        check("unparseable JSON reads as no snapshot", False, f"{type(exc).__name__}: {exc}")

    print("\n--- WAL, which is what a running Unsloth uses ---")
    conn = sqlite3.connect(str(db_path))
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    row = thread_row("wal")
    db.upsert_chat_thread(row)
    db.update_chat_thread(row["id"], {"settings": {"codeToolsEnabled": True}})
    check(
        f"writes work under journal_mode={mode}",
        db.get_chat_thread(row["id"]).get("settings") == {"codeToolsEnabled": True},
    )

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} passed")
    if FAILURES:
        print("FAILED: " + ", ".join(FAILURES))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
