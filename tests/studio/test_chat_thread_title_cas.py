# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A title patch can carry guards, so a rename or a deleted opening message
beats a background rewrite.

studio_db imports its siblings by bare name (utils.paths), so the functional
checks run in a subprocess with studio/backend on PYTHONPATH rather than
putting those names on this session's sys.path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "studio" / "backend"
ROUTE = BACKEND / "routes" / "chat_history.py"

PROBE = r"""
import json, sys
from storage import studio_db as db

def message(thread_id, message_id, role, text, created_at):
    db.upsert_chat_message({
        "id": message_id,
        "threadId": thread_id,
        "role": role,
        "content": [{"type": "text", "text": text}],
        "createdAt": created_at,
    })

def thread(thread_id, title):
    db.upsert_chat_thread({
        "id": thread_id,
        "title": title,
        "modelType": "base",
        "createdAt": 1,
        "updatedAt": 1,
    })

out = {}

# No guard: the write lands, as every other caller expects.
thread("t1", "old")
out["unguarded"] = db.update_chat_thread("t1", {"title": "new"})["title"]

# Guard matches: the rewrite lands.
thread("t2", "legacy title...")
out["guard_match"] = db.update_chat_thread(
    "t2", {"title": "whole first line"}, expected_title = "legacy title..."
)["title"]

# A rename landed after the rewrite read the row: the rename has to win.
thread("t3", "legacy title...")
db.update_chat_thread("t3", {"title": "what the user typed"})
try:
    db.update_chat_thread(
        "t3", {"title": "whole first line"}, expected_title = "legacy title..."
    )
    out["guard_stale"] = "no error"
except db.ChatThreadPreconditionFailed:
    out["guard_stale"] = "mismatch"
out["after_stale"] = db.get_chat_thread("t3")["title"]

# The opening message guard, while that message is still the opening one.
thread("t4", "legacy title...")
message("t4", "m2", "user", "second", 20)
message("t4", "m1", "user", "first", 10)
message("t4", "a1", "assistant", "reply", 15)
out["opening_match"] = db.update_chat_thread(
    "t4",
    {"title": "whole first line"},
    expected_title = "legacy title...",
    expected_opening_message_id = "m1",
)["title"]

# The opening message was deleted after the rewrite read it. Its text must not
# be expanded into the title.
thread("t5", "legacy title...")
message("t5", "t5m1", "user", "first", 10)
message("t5", "t5m2", "user", "second", 20)
db.sync_chat_messages("t5", [
    {"id": "t5m2", "threadId": "t5", "role": "user",
     "content": [{"type": "text", "text": "second"}], "createdAt": 20},
], prune_missing = True)
try:
    db.update_chat_thread(
        "t5",
        {"title": "whole first line"},
        expected_title = "legacy title...",
        expected_opening_message_id = "t5m1",
    )
    out["opening_deleted"] = "no error"
except db.ChatThreadPreconditionFailed:
    out["opening_deleted"] = "mismatch"
out["after_opening_deleted"] = db.get_chat_thread("t5")["title"]

# A thread whose messages are all gone: the subquery is NULL, still no match.
thread("t6", "legacy title...")
try:
    db.update_chat_thread(
        "t6",
        {"title": "whole first line"},
        expected_opening_message_id = "t6m1",
    )
    out["opening_empty"] = "no error"
except db.ChatThreadPreconditionFailed:
    out["opening_empty"] = "mismatch"

# A thread that is gone reads as missing, not as a mismatch.
out["missing"] = db.update_chat_thread(
    "gone", {"title": "anything"}, expected_title = "legacy title..."
)

# updatedAt is untouched by a title patch, so Recents keeps its order.
out["updated_at"] = db.get_chat_thread("t2")["updatedAt"]

print(json.dumps(out))
"""


@pytest.fixture(scope = "module")
def probe(tmp_path_factory) -> dict:
    env = dict(os.environ)
    env["UNSLOTH_STUDIO_HOME"] = str(tmp_path_factory.mktemp("studio_home"))
    env["PYTHONPATH"] = str(BACKEND)
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output = True,
        text = True,
        env = env,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_a_patch_without_a_guard_still_applies(probe):
    assert probe["unguarded"] == "new"


def test_the_guard_lets_the_write_through_while_the_title_is_unchanged(probe):
    assert probe["guard_match"] == "whole first line"


def test_a_rename_between_the_read_and_the_write_wins(probe):
    assert probe["guard_stale"] == "mismatch"
    assert probe["after_stale"] == "what the user typed"


def test_a_missing_thread_reads_as_missing_not_as_a_mismatch(probe):
    assert probe["missing"] is None


def test_a_title_patch_leaves_updated_at_alone(probe):
    assert probe["updated_at"] == 1


def test_the_opening_guard_matches_the_earliest_user_message(probe):
    """Earliest by createdAt, not by insertion order, and not the assistant's."""
    assert probe["opening_match"] == "whole first line"


def test_a_prompt_deleted_between_the_read_and_the_write_wins(probe):
    assert probe["opening_deleted"] == "mismatch"
    assert probe["after_opening_deleted"] == "legacy title..."


def test_a_thread_with_no_messages_left_reads_as_a_mismatch(probe):
    assert probe["opening_empty"] == "mismatch"


def test_the_route_turns_a_failed_precondition_into_409():
    source = ROUTE.read_text(encoding = "utf-8")
    assert 'expected_title = patch.pop("expectedTitle", None)' in source
    assert 'expected_opening_message_id = patch.pop("expectedOpeningMessageId", None)' in source
    assert "except ChatThreadPreconditionFailed:" in source
    assert "status_code = 409," in source
