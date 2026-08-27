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

import asyncio
from pathlib import Path

import pytest

from unforgettable.loop.context import EpisodeRequest
from unforgettable.loop.episode import run
from unforgettable.rims.fs_copy import FsCopyPlugin
from unforgettable.rims.plugin import (
    DEFAULT_TWIN_PLUGIN,
    FS_COPY_ID,
    NONE_ID,
    coerce_twin_plugin,
    get_twin_plugin,
)
from unforgettable.tests.test_episode import FakeHost, _fail_world, _ok
from unforgettable.throne.policy import Action


def test_coerce_twin_plugin_aliases():
    assert coerce_twin_plugin(None) == DEFAULT_TWIN_PLUGIN
    assert coerce_twin_plugin("fs.copy") == FS_COPY_ID
    assert coerce_twin_plugin("filesystem") == FS_COPY_ID
    assert coerce_twin_plugin("none") == NONE_ID
    assert coerce_twin_plugin("verbal") == NONE_ID
    with pytest.raises(ValueError, match = "unknown twin plugin"):
        coerce_twin_plugin("docker")


def test_fs_copy_two_spawns_are_independent(tmp_path: Path):
    host = FakeHost(tmp_path, [])
    (host.world / "app.py").write_text("print('world')\n")
    plugin = FsCopyPlugin()
    world = plugin.world(host, EpisodeRequest(messages = []))
    first = plugin.spawn_sim(host, world, "episode-a", clone_index = 1)
    second = plugin.spawn_sim(host, world, "episode-a", clone_index = 2)
    assert first.location.handle != second.location.handle
    assert first.location.handle != world.location.handle
    copied = Path(host.sandbox_path(first.location.handle)) / "app.py"
    assert copied.read_text() == "print('world')\n"
    (host.world / "app.py").write_text("print('changed')\n")
    assert copied.read_text() == "print('world')\n"
    plugin.cleanup(host, first)
    plugin.cleanup(host, second)
    assert first.location.handle in host.removed
    assert second.location.handle in host.removed


def test_none_spawn_does_not_copy(tmp_path: Path):
    host = FakeHost(tmp_path, [])
    plugin = get_twin_plugin("none")
    world = plugin.world(host, EpisodeRequest(messages = []))
    sim = plugin.spawn_sim(host, world, "abcdef12-episode", clone_index = 1)
    assert sim.location.plugin == NONE_ID
    assert sim.location.uri == "sim://none"
    assert sim.location.handle.startswith("sim-none-")
    assert host.sims == {}
    plugin.cleanup(host, sim)
    assert host.removed == []


def test_episode_none_twin_skips_clone(tmp_path: Path):
    host = FakeHost(
        tmp_path,
        [_fail_world(), _ok("verbal sim", "sim"), _ok("works in world", "world")],
    )
    before = set(host.sims)
    outcome = asyncio.run(
        run(
            host,
            EpisodeRequest(
                messages = [{"role": "user", "content": "run the tests"}],
                twin_plugin = "none",
            ),
        )
    )
    assert Action.ENTER_SIM in outcome.actions
    assert any(call.startswith("sim-none-") for call in host.calls)
    assert set(host.sims) == before
    assert host.removed == []
    system = " ".join(
        str(m.get("content")) for m in (host.last_messages or []) if m.get("role") == "system"
    )
    assert "Retry in the world with the repaired plan." in system


def test_episode_still_clones_on_fs_copy(tmp_path: Path):
    host = FakeHost(
        tmp_path,
        [_fail_world(), _ok("fixed in sim", "sim"), _ok("works in world", "world")],
    )
    asyncio.run(
        run(
            host,
            EpisodeRequest(messages = [{"role": "user", "content": "run the tests"}]),
        )
    )
    sim_id = host.calls[1]
    assert sim_id.startswith("sim-")
    assert (host.sims[sim_id] / "app.py").read_text() == "print('world')\n"
