# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Mint (or reuse) an internal API key for the voice benchmark.

The Studio API is auth-gated (Bearer JWT or ``sk-unsloth-`` API key). The
benchmark is a headless client with no login, so this bootstrap mints a hidden
*internal* API key straight against the local auth DB using the same hashing the
server uses, then caches the raw key in ``.bench_token`` next to this file.

Run once (uses the Studio venv so ``auth.storage`` imports cleanly):

    /c/Users/<you>/.unsloth/studio/unsloth_studio/Scripts/python.exe mint_token.py

``voice_bench.py`` calls this automatically if ``.bench_token`` is missing, so
you normally never run it by hand.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN_FILE = HERE / ".bench_token"
KEY_NAME = "voice-benchmark"

# studio/backend, so `import auth.storage` and the path helpers resolve.
BACKEND = HERE.parent.parent / "backend"


def _load_backend():
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    from auth import storage  # noqa: E402
    from utils.paths.storage_roots import auth_db_path  # noqa: E402

    return storage, auth_db_path()


def _first_username(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT username FROM auth_user ORDER BY rowid LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise SystemExit(
            "No user in the auth DB yet. Open Studio once and sign in, then re-run."
        )
    return row[0]


def get_token(force: bool = False) -> str:
    """Return a usable raw API key, minting and caching one if needed."""
    if TOKEN_FILE.exists() and not force:
        cached = TOKEN_FILE.read_text(encoding = "utf-8").strip()
        if cached:
            return cached

    storage, db_path = _load_backend()
    if not Path(db_path).exists():
        raise SystemExit(f"Auth DB not found at {db_path}. Start Studio at least once.")

    username = _first_username(Path(db_path))
    raw_key, _row = storage.create_api_key(
        username = username, name = KEY_NAME, internal = True
    )
    TOKEN_FILE.write_text(raw_key + "\n", encoding = "utf-8")
    try:
        TOKEN_FILE.chmod(0o600)
    except OSError:
        pass
    return raw_key


if __name__ == "__main__":
    force = "--force" in sys.argv
    token = get_token(force = force)
    print(f"user token ready -> {TOKEN_FILE}")
    print(f"{token[:16]}... ({len(token)} chars)")
