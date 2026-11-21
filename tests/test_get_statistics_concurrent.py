# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""get_statistics() fetches its stats repos concurrently under ONE 120 s limit: one child per
repo, one after another, blocked from_pretrained ~76 s behind a slow proxy (#14366).
Helpers loaded by AST; huggingface_hub, psutil and unsloth_zoo are stubs: CPU only, no network, no deps."""

import ast
import functools
import logging
import os
import pathlib
import re
import socket
import sys
import threading
import time
import types

import pytest

_UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"
_DELAY = 0.4


class _Props:
    total_memory = 32 * 1024**3


def _load(monkeypatch, limiter, snapshot_download):
    tree = ast.parse(_UTILS.read_text(encoding = "utf-8"))
    keep = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and re.search(r"statistic|has_internet", node.name)
    ]
    ns = {
        "os": os,
        "functools": functools,
        "socket": socket,
        "psutil": types.SimpleNamespace(cpu_count = lambda logical = True: 4),
        "logger": logging.getLogger("unsloth"),
        "torch": types.SimpleNamespace(
            cuda = types.SimpleNamespace(get_device_properties = lambda i: _Props())
        ),
        "DEVICE_TYPE": "cuda",
        "DEVICE_COUNT": 1,
        "USE_MODELSCOPE": False,
    }
    exec(compile(ast.Module(body = keep, type_ignores = []), str(_UTILS), "exec"), ns)
    ns["has_internet"] = lambda: True
    zoo = types.ModuleType("unsloth_zoo")
    zoo.__path__ = []
    rl = types.ModuleType("unsloth_zoo.rl_environments")
    rl.execute_with_time_limit = limiter
    zoo.rl_environments = rl
    hub = types.ModuleType("huggingface_hub")
    hub.__path__ = []
    hub.snapshot_download = snapshot_download
    hub_utils = types.ModuleType("huggingface_hub.utils")
    bars = {"disabled": False}
    hub_utils.are_progress_bars_disabled = lambda: bars["disabled"]
    hub_utils.disable_progress_bars = lambda: bars.update(disabled = True)
    hub_utils.enable_progress_bars = lambda: bars.update(disabled = False)
    hub.utils = hub_utils
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", hub_utils)
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.rl_environments", rl)
    for k in (
        "UNSLOTH_DISABLE_STATISTICS",
        "UNSLOTH_USE_MODELSCOPE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_OFFLINE",
    ):
        monkeypatch.delenv(k, raising = False)
    return ns


class _Recorder:
    def __init__(self, fail = ()):
        self.calls = []
        self.limits = []
        self.fail = set(fail)
        self.lock = threading.Lock()

    def snapshot_download(
        self,
        repo_id,
        force_download = False,
        **kwargs,
    ):
        with self.lock:
            self.calls.append((repo_id, force_download))
        time.sleep(_DELAY)
        if repo_id.split("/", 1)[1] in self.fail:
            raise OSError("simulated HTTP error")

    def limiter(self, seconds, **kwargs):
        self.limits.append((seconds, kwargs))
        return lambda f: f


@pytest.fixture
def rec():
    return _Recorder()


def test_repos_fetched_concurrently(monkeypatch, rec):
    ns = _load(monkeypatch, rec.limiter, rec.snapshot_download)
    t0 = time.perf_counter()
    ns["get_statistics"](False)
    elapsed = time.perf_counter() - t0
    names = sorted(r.split("/", 1)[1] for r, _ in rec.calls)
    assert len(names) == 4 and "repeat" in names and "vram-40" in names and "1" in names
    assert all(force for _, force in rec.calls)
    # Sequential would be 4 * _DELAY.
    assert elapsed < 2.5 * _DELAY, elapsed


def test_one_time_limited_child_with_process_backend(monkeypatch, rec):
    ns = _load(monkeypatch, rec.limiter, rec.snapshot_download)
    ns["get_statistics"](False)
    # SIGALRM cannot interrupt worker threads; only a killed child bounds a hung download.
    assert rec.limits == [(120, {"backend": "process"})]


def test_hf_down_message_kept(monkeypatch, rec):
    def limiter(seconds, **kwargs):
        def wrap(f):
            def run():
                raise TimeoutError("Timed out after 120s")

            return run

        return wrap

    ns = _load(monkeypatch, limiter, rec.snapshot_download)
    with pytest.raises(TimeoutError, match = "HuggingFace seems to be down"):
        ns["get_statistics"](False)


def test_failed_repo_is_swallowed_and_others_still_fetched(monkeypatch):
    rec = _Recorder(fail = {"repeat"})
    ns = _load(monkeypatch, rec.limiter, rec.snapshot_download)
    assert ns["get_statistics"](False) is None
    assert len(rec.calls) == 4


@pytest.mark.parametrize(
    "env, local_files_only",
    [
        ({"UNSLOTH_DISABLE_STATISTICS": "1"}, False),
        ({"HF_HUB_OFFLINE": "1"}, False),
        ({}, True),
    ],
)
def test_opt_outs_make_no_request(monkeypatch, rec, env, local_files_only):
    ns = _load(monkeypatch, rec.limiter, rec.snapshot_download)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ns["get_statistics"](local_files_only)
    assert rec.calls == [] and rec.limits == []
