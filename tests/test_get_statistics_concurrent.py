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
        if (isinstance(node, ast.FunctionDef) and re.search(r"statistic|has_internet", node.name))
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id in ("_STATISTICS_LIMIT_S", "_HF_DOWN_MESSAGE")
                for t in node.targets
            )
        )
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
        "threading": threading,
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
    # These cases pin the blocking path; the default background path is covered at the end of this file.
    monkeypatch.setenv("UNSLOTH_STATISTICS_BLOCKING", "1")
    # The blocking path's child needs fork; pin it present so the cases also run on Windows.
    import multiprocessing

    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["fork", "spawn"])
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


def test_default_path_does_not_block_the_load(monkeypatch, rec):
    """The default path forks the child and returns at once; the child still makes all four requests."""
    if sys.platform != "linux":
        pytest.skip("background statistics need the fork start method")
    ns = _load(monkeypatch, rec.limiter, rec.snapshot_download)
    monkeypatch.delenv("UNSLOTH_STATISTICS_BLOCKING")
    marker = pathlib.Path(os.environ.get("PYTEST_TMPDIR", "/tmp")) / f"unsloth_stats_{os.getpid()}"
    marker.unlink(missing_ok = True)

    def slow_download(
        repo_id,
        force_download = False,
        **kwargs,
    ):
        time.sleep(1.5)
        with open(marker, "a") as f:
            f.write(repo_id + "\n")

    sys.modules["huggingface_hub"].snapshot_download = slow_download
    started = []
    real = ns["_download_statistics_background"]
    ns["_download_statistics_background"] = lambda names: started.append(real(names)) or started[-1]
    t0 = time.perf_counter()
    ns["get_statistics"](False)
    elapsed = time.perf_counter() - t0
    assert (
        elapsed < 1.0
    ), elapsed  # the blocking path takes >= 1.5 s here (slow_download); margin for loaded CI hosts
    assert rec.limits == []  # the blocking time-limited wrapper was not used
    # The reaper thread owns the join; poll the child's side effect instead of racing it on waitpid.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and (
        not marker.exists() or len(marker.read_text().split()) < 4
    ):
        time.sleep(0.05)
    names = sorted(line.split("/", 1)[1] for line in marker.read_text().split())
    marker.unlink(missing_ok = True)
    assert len(names) == 4 and "repeat" in names and "vram-40" in names


def test_background_hang_is_killed_and_warned(monkeypatch, caplog):
    if sys.platform != "linux":
        pytest.skip("background statistics need the fork start method")
    import multiprocessing as mp

    ns = _load(monkeypatch, lambda *a, **k: (lambda f: f), lambda *a, **k: None)
    proc = mp.get_context("fork").Process(target = time.sleep, args = (60,), daemon = True)
    proc.start()
    with caplog.at_level(logging.WARNING, logger = "unsloth"):
        ns["_reap_statistics"](proc, 0.3)
    assert not proc.is_alive()
    assert "HuggingFace seems to be down" in caplog.text


def test_background_start_failure_falls_back(monkeypatch):
    # Inside a daemonic multiprocessing worker Process.start raises AssertionError; the load must not fail.
    if sys.platform != "linux":
        pytest.skip("background statistics need the fork start method")
    import multiprocessing.process as mp_process

    ns = _load(monkeypatch, lambda *a, **k: (lambda f: f), lambda *a, **k: None)

    def refuse(self):
        raise AssertionError("daemonic processes are not allowed to have children")

    monkeypatch.setattr(mp_process.BaseProcess, "start", refuse)
    assert ns["_download_statistics_background"](["repeat"]) is None


def test_no_fork_skips_the_probe_and_the_requests(monkeypatch, rec):
    """Without fork (Windows) the time-limited child can never start, so the 8.8.8.8 probe is skipped as well."""
    import multiprocessing

    ns = _load(monkeypatch, rec.limiter, rec.snapshot_download)
    probes = []
    ns["has_internet"] = lambda: probes.append(1) or True
    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    assert ns["get_statistics"](False) is None
    assert probes == [] and rec.calls == [] and rec.limits == []


def test_real_platform_routing(monkeypatch, rec):
    import multiprocessing

    real = multiprocessing.get_all_start_methods
    ns = _load(monkeypatch, rec.limiter, rec.snapshot_download)
    monkeypatch.setattr(multiprocessing, "get_all_start_methods", real)
    probes = []
    ns["has_internet"] = lambda: probes.append(1) or False
    ns["get_statistics"](False)
    assert probes == ([1] if "fork" in real() else [])
