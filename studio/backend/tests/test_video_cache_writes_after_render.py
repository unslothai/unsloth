# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A first video load's cache writes wait for its first render (or an idle window), and never begin after teardown."""

import ast
import inspect
import threading
import time

import pytest

from core.inference import video as V


class _W:
    def __init__(self, log, name):
        self.log = log
        self.name = name

    def start(self):
        self.log.append(("start", self.name))
        return True

    def discard(self):
        self.log.append(("discard", self.name))


class _Native:
    def __init__(self, log):
        self.log = log

    def start(self):
        self.log.append(("start", "native"))

    def finish(self, timeout = None):
        self.log.append(("finish", "native"))


class _Pipe:
    pass


def _holder(
    log,
    busy = lambda: False,
    idle_s = 0.0,
):
    pipe = _Pipe()
    writes = (_W(log, "cast"), _Native(log), _W(log, "resident"))
    holder = V._HeldCacheWrites(pipe, writes, busy = busy, idle_s = idle_s)
    setattr(pipe, V._HELD_CACHE_WRITES_ATTR, holder)
    return pipe, holder, writes


def test_the_first_render_end_starts_them_once():
    log = []
    pipe, holder, writes = _holder(log)
    holder.arm()
    assert log == []
    V._release_held_cache_writes(pipe, "the first render ended")
    assert log == [("start", "cast"), ("start", "native"), ("start", "resident")]
    assert pipe._unsloth_cast_writes is writes[0]
    assert getattr(pipe, V._NATIVE_WRITES_ATTR) is writes[1]
    assert getattr(pipe, V._RESIDENT_QUANT_WRITE_ATTR) is writes[2]
    V._release_held_cache_writes(pipe, "a second render ended")
    assert len(log) == 3


def test_teardown_before_a_render_discards_them():
    log = []
    pipe, holder, _writes = _holder(log, idle_s = 30.0)
    holder.arm()
    V._discard_held_cache_writes(pipe)
    assert ("start", "cast") not in log
    assert sorted(log) == [("discard", "cast"), ("discard", "resident"), ("finish", "native")]
    assert getattr(pipe, V._HELD_CACHE_WRITES_ATTR) is None
    V._release_held_cache_writes(pipe, "late")
    assert not any(kind == "start" for kind, _n in log)


def test_an_idle_backend_starts_them_and_a_busy_one_waits_for_the_render():
    log = []
    busy = threading.Event()
    busy.set()
    pipe, holder, _writes = _holder(log, busy = busy.is_set, idle_s = 0.05)
    holder.arm()
    time.sleep(0.3)
    assert log == []  # the render in flight releases them
    V._release_held_cache_writes(pipe, "the first render ended")
    assert [kind for kind, _n in log] == ["start"] * 3

    log2 = []
    pipe2, holder2, _w = _holder(log2, idle_s = 0.05)
    holder2.arm()
    deadline = time.time() + 5
    while len(log2) < 3 and time.time() < deadline:
        time.sleep(0.02)
    assert [kind for kind, _n in log2] == ["start"] * 3


def test_kill_switch_and_idle_env(monkeypatch):
    monkeypatch.setenv(V.VIDEO_CACHE_WRITES_AFTER_RENDER_ENV, "0")
    assert V._cache_writes_after_render() is False
    monkeypatch.delenv(V.VIDEO_CACHE_WRITES_AFTER_RENDER_ENV)
    assert V._cache_writes_after_render() is True
    monkeypatch.setenv(V.VIDEO_CACHE_WRITE_IDLE_ENV, "bad")
    assert V._cache_write_idle_s() == V._VIDEO_CACHE_WRITE_IDLE_DEFAULT_S
    monkeypatch.setenv(V.VIDEO_CACHE_WRITE_IDLE_ENV, "-3")
    assert V._cache_write_idle_s() == 0.0


def test_generate_releases_and_unload_discards_before_finishing():
    gen = inspect.getsource(V.VideoBackend.generate)
    assert (
        '_release_held_cache_writes(getattr(state, "pipe", None), "the first render ended")' in gen
    )
    # inside the finally, so a failed or cancelled first render releases them too
    tree = ast.parse("class _X:\n" + gen if gen.startswith("    ") else gen)
    finals = [
        ast.unparse(stmt)
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for stmt in node.finalbody
    ]
    assert any("_release_held_cache_writes" in text for text in finals)
    src = inspect.getsource(V.VideoBackend)
    discard = src.index('_discard_held_cache_writes(getattr(state, "pipe", None))')
    assert discard < src.index('_finish_cast_writes(getattr(state, "pipe", None))')
