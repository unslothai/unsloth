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

from pathlib import Path

import pytest

from unforgettable.eyes.basic import inspect_tool_result
from unforgettable.loop.context import EpisodeRequest, EpisodeState
from unforgettable.rims.clone import clone_tree
from unforgettable.throne.policy import Action, decide, policy_from_request, require_confirm_retry


def test_clone_tree_copies_and_skips_markers(tmp_path: Path):
    src = tmp_path / "world"
    dst = tmp_path / "sim"
    src.mkdir()
    (src / "main.py").write_text("print(1)\n")
    (src / ".unsloth_sandbox").write_text("world\n")
    (src / ".unsloth_sandbox_remap.json").write_text("{}\n")
    (src / "keep.txt").write_text("ok\n")
    stale = src / "old.deleting-abcdef12"
    stale.mkdir()
    (stale / "gone.txt").write_text("nope\n")
    clone_tree(src, dst)
    assert (dst / "main.py").read_text() == "print(1)\n"
    assert (dst / "keep.txt").exists()
    assert not (dst / ".unsloth_sandbox").exists()
    assert not (dst / ".unsloth_sandbox_remap.json").exists()
    assert not (dst / "old.deleting-abcdef12").exists()
    (src / "main.py").write_text("print(2)\n")
    assert (dst / "main.py").read_text() == "print(1)\n"


def test_clone_tree_same_path_raises(tmp_path: Path):
    src = tmp_path / "world"
    src.mkdir()
    (src / "main.py").write_text("print(1)\n")
    with pytest.raises(ValueError, match = "clone_tree refuses to copy a tree onto itself"):
        clone_tree(src, src)
    with pytest.raises(ValueError, match = "clone_tree refuses to copy a tree onto itself"):
        clone_tree(src, src / ".")


def test_clone_tree_refuses_dest_inside_source(tmp_path: Path):
    src = tmp_path / "world"
    src.mkdir()
    (src / "main.py").write_text("print(1)\n")
    with pytest.raises(ValueError, match = "into itself"):
        clone_tree(src, src / "nested-sim")


def test_clone_tree_copies_symlinks_without_following(tmp_path: Path):
    src = tmp_path / "world"
    dst = tmp_path / "sim"
    outside = tmp_path / "secret"
    outside.write_text("do-not-copy-bytes\n")
    src.mkdir()
    (src / "link").symlink_to(outside)
    clone_tree(src, dst)
    copied = dst / "link"
    assert copied.is_symlink()
    assert copied.readlink() == outside
    outside.write_text("changed\n")
    assert copied.read_text() == "changed\n"


def test_eyes_detect_traceback_and_exit_code():
    assert inspect_tool_result("python", "Traceback (most recent call last):\nValueError")
    assert inspect_tool_result("terminal", "exit code 1\n")
    assert inspect_tool_result("python", "Error: boom")
    assert inspect_tool_result("terminal", "ok\n") is None


def test_throne_world_failure_enters_sim_then_retry():
    state = EpisodeState(episode_id = "e", world_session = "world")
    assert decide("failure", state) == Action.ENTER_SIM
    state.enter_sim("sim-1")
    state.had_world_failure = True
    assert decide("success", state) == Action.RETRY_WORLD
    state.enter_world()
    assert decide("success", state) == Action.FINISH
    state.clone_count = 1
    assert decide("failure", state) == Action.ESCALATE


def test_require_confirm_retry_matrix():
    assert require_confirm_retry(stakes = "high", permission_mode = None, confirm_retry = None)
    assert require_confirm_retry(stakes = None, permission_mode = "ask", confirm_retry = None)
    assert not require_confirm_retry(stakes = "high", permission_mode = "ask", confirm_retry = False)
    for mode in ("full", "off", "auto", None):
        assert not require_confirm_retry(stakes = None, permission_mode = mode, confirm_retry = None)
    assert require_confirm_retry(stakes = None, permission_mode = None, confirm_retry = True)

    def wired(**kwargs) -> bool:
        return policy_from_request(EpisodeRequest(messages = [], **kwargs)).require_confirm_retry

    assert wired(stakes = "high") is True
    assert wired(permission_mode = "ask") is True
    assert wired(stakes = "high", confirm_retry = False) is False
    for mode in ("full", "off", "auto", None):
        assert wired(permission_mode = mode) is False
    assert wired(confirm_retry = True) is True
