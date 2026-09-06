# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lexical Windows checks; these do not claim to run Windows ACL or filesystem semantics."""

import ntpath
import re
import unicodedata
import uuid
from pathlib import PureWindowsPath

import pytest

from utils.account_context import AccountContext, OWNER, run_as
from utils.paths import storage_roots

ACCOUNT_ID = "0123456789abcdef0123456789abcdef"
ROOT = PureWindowsPath(r"C:\Users\Research Engineer\AppData\Local\UnslothStudio")
RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
# Leaves have model, checkpoint, document and tool identifiers of realistic lengths.
LEAVES = (
    r"studio.db",
    r"outputs\Llama-3.1-8B-Instruct\checkpoint-10000\adapter_model.safetensors",
    r"exports\Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    r"rag\documents\0123456789abcdef0123456789abcdef\quarterly-research-report.pdf",
    r"sandbox\0123456789abcdef0123456789abcdef\workspace\analysis\results.csv",
    r"assets\datasets\0123456789abcdef0123456789abcdef\training-data.jsonl",
)


def utf16_units(path: str) -> int:
    return len(path.encode("utf-16-le")) // 2


@pytest.mark.parametrize(
    "username",
    sorted(RESERVED)
    + [
        "CON.txt",
        "aux.",
        "..",
        "alice/bob",
        "alice\\bob",
        "Élodie",
        "E\u0301lodie",
        "Ａｌｉｃｅ",
        "ALICE",
        "alice",
    ],
)
def test_username_never_becomes_a_windows_storage_component(monkeypatch, username):
    monkeypatch.setattr(storage_roots, "studio_root", lambda: ROOT)
    account = AccountContext(ACCOUNT_ID, username)
    actual = str(run_as(account, storage_roots.workspace_root))
    expected = ntpath.join(str(ROOT), "accounts", ACCOUNT_ID)
    assert actual == expected
    assert ntpath.commonpath([str(ROOT), actual]) == str(ROOT)
    assert re.fullmatch(r"[0-9a-f]{32}", ntpath.basename(actual))
    assert ntpath.basename(actual).upper() not in RESERVED
    for form in ("NFC", "NFD"):
        assert unicodedata.normalize(form, actual) == expected


def test_owner_windows_layout_is_unchanged(monkeypatch):
    monkeypatch.setattr(storage_roots, "studio_root", lambda: ROOT)
    assert run_as(OWNER, storage_roots.workspace_root) == ROOT
    assert run_as(OWNER, storage_roots.studio_db_path) == ROOT / "studio.db"


def test_distinct_generated_account_ids_do_not_collide_under_case_folding(isolated_auth):
    ids = []
    for username in ("alice", "ALICE", "café", "cafe\u0301", "CON"):
        isolated_auth.create_initial_user(username, "account-password", uuid.uuid4().hex)
        account_id = isolated_auth.get_account(username).account_id
        assert uuid.UUID(hex = account_id).version == 4
        ids.append(account_id)
    assert len({ntpath.normcase(ntpath.join("accounts", value)) for value in ids}) == len(ids)


@pytest.mark.parametrize("leaf", LEAVES)
def test_max_path_budget_for_realistic_leaves(leaf):
    owner_path = ntpath.join(str(ROOT), leaf)
    account_path = ntpath.join(str(ROOT), "accounts", ACCOUNT_ID, leaf)
    # MAX_PATH includes the trailing NUL. The fixed accounts/hex component adds 42 UTF-16 units.
    assert utf16_units(account_path) - utf16_units(owner_path) == 42
    assert utf16_units(account_path) + 1 <= 260
    budget = 259 - utf16_units(ntpath.join("accounts", ACCOUNT_ID, leaf)) - 1
    assert utf16_units(str(ROOT)) <= budget
    at_limit = ntpath.join("C:\\" + "x" * (budget - 3), "accounts", ACCOUNT_ID, leaf)
    assert utf16_units(at_limit) == 259
    assert utf16_units(at_limit + "x") + 1 > 260


def test_path_budget_counts_utf16_units_not_unicode_code_points():
    assert utf16_units("📁") == 2
    assert utf16_units("é") == 1
    assert utf16_units("e\u0301") == 2
    assert utf16_units(ACCOUNT_ID) == 32
