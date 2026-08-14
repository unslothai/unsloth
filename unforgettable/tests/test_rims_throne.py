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

from unforgettable.eyes.basic import inspect_tool_result
from unforgettable.loop.context import EpisodeState
from unforgettable.rims.clone import clone_tree
from unforgettable.throne.policy import Action, decide


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


def test_eyes_detect_traceback_and_exit_code():
    assert inspect_tool_result("python", "Traceback (most recent call last):\nValueError")
    assert inspect_tool_result("terminal", "exit code 1\n")
    assert inspect_tool_result("python", "Error: boom")
    assert inspect_tool_result("terminal", "ok\n") is None


def test_throne_world_failure_enters_sim_then_retry():
    state = EpisodeState(episode_id="e", world_session="world")
    assert decide("failure", state) == Action.ENTER_SIM
    state.enter_sim("sim-1")
    state.had_world_failure = True
    assert decide("success", state) == Action.RETRY_WORLD
    state.enter_world()
    assert decide("success", state) == Action.FINISH
    state.clone_count = 1
    assert decide("failure", state) == Action.ESCALATE
