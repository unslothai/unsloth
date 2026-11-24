# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""conftest's WAL keeper cleanup runs before a test's monkeypatch is undone, so it must not depend on a faked
os.name. Many tests fake Windows with monkeypatch.setattr(os, "name", "nt"); when a WAL keeper was opened
meanwhile, the cleanup's Path(...).resolve() built a WindowsPath and errored at teardown
("cannot instantiate 'WindowsPath' on your system", seen on test_reset_password_command.py in m-r)."""

import os


def test_a_keeper_left_while_faking_windows_is_still_closed(monkeypatch, tmp_path):
    from storage import rag_db
    monkeypatch.setattr(os, "name", "nt")
    # What a keeper opened during the test leaves behind; the conftest teardown closes it.
    rag_db._wal_keepers[(tmp_path / "rag.db").resolve()] = None
