# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The /proc walk finds what psutil finds, and psutil answers whenever the walk cannot."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import pytest

from utils import process_tree

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "Linux /proc walk")


def _sleeper():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


@pytest.fixture
def tree():
    nested = (
        "import subprocess, sys, time; "
        "[subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']) for _ in range(2)]; "
        "time.sleep(60)"
    )
    psutil = pytest.importorskip("psutil")
    me = psutil.Process()
    # Earlier tests in this session can leave children of their own; count only ours.
    before = {child.pid for child in me.children(recursive = True)}
    procs = [_sleeper(), subprocess.Popen([sys.executable, "-c", nested])]
    # A live thread lists its own children (they move to the main task once it exits): keep it alive.
    forked = threading.Event()
    release = threading.Event()

    def fork_and_wait():
        procs.append(_sleeper())
        forked.set()
        release.wait(60)

    worker = threading.Thread(target = fork_and_wait, daemon = True)
    worker.start()
    forked.wait(30)
    deadline = time.monotonic() + 30

    def ours():
        return {child.pid for child in me.children(recursive = True)} - before

    while len(ours()) < 5 and time.monotonic() < deadline:
        time.sleep(0.1)
    yield me
    for child in me.children(recursive = True):
        if child.pid in before:
            continue
        try:
            child.kill()
        except psutil.Error:
            pass
    for proc in procs:
        proc.wait()
    release.set()
    worker.join()


@linux_only
def test_the_walk_finds_what_psutil_finds(tree):
    # Another test's leftover child can exit between the two reads; a real difference persists.
    for _ in range(3):
        expected = sorted(child.pid for child in tree.children(recursive = True))
        walked = sorted(process_tree.linux_descendant_pids(os.getpid()))
        if walked == expected:
            break
    assert len(expected) >= 5
    assert walked == expected


def test_other_platforms_defer_to_psutil(monkeypatch):
    monkeypatch.setattr(process_tree.sys, "platform", "darwin")
    assert process_tree.linux_descendant_pids(os.getpid()) is None


@linux_only
def test_a_kernel_without_children_files_defers_to_psutil(tmp_path, monkeypatch):
    (tmp_path / "123" / "task" / "123").mkdir(parents = True)
    monkeypatch.setattr(process_tree, "_PROC", str(tmp_path))
    assert process_tree.linux_descendant_pids(123) is None


@linux_only
def test_a_vanished_root_defers_to_psutil(tmp_path, monkeypatch):
    monkeypatch.setattr(process_tree, "_PROC", str(tmp_path))
    assert process_tree.linux_descendant_pids(123) is None


@linux_only
def test_a_descendant_exiting_mid_walk_is_skipped(tmp_path, monkeypatch):
    root = tmp_path / "1" / "task" / "1"
    root.mkdir(parents = True)
    (root / "children").write_text("2 3 ", encoding = "utf-8")
    # 2 has exited (no /proc entry); 3 is alive with a child of its own.
    alive = tmp_path / "3" / "task" / "3"
    alive.mkdir(parents = True)
    (alive / "children").write_text("4", encoding = "utf-8")
    monkeypatch.setattr(process_tree, "_PROC", str(tmp_path))
    assert sorted(process_tree.linux_descendant_pids(1)) == [2, 3, 4]


@linux_only
def test_a_thread_vanishing_mid_walk_defers_to_psutil(tmp_path, monkeypatch):
    # Thread 7 exited mid-walk; its children may have moved to thread 1, already read.
    root = tmp_path / "1" / "task"
    (root / "1").mkdir(parents = True)
    (root / "1" / "children").write_text("2", encoding = "utf-8")
    (root / "7").mkdir()
    monkeypatch.setattr(process_tree, "_PROC", str(tmp_path))
    assert process_tree.linux_descendant_pids(1) is None


@linux_only
def test_tree_memory_uses_the_walk_and_falls_back_to_psutil(tree, monkeypatch):
    import main

    walked = main._process_tree_rss_mb(tree)
    monkeypatch.setattr(process_tree, "linux_descendant_pids", lambda pid: None)
    scanned = main._process_tree_rss_mb(tree)
    assert abs(walked - scanned) <= max(8, scanned // 20)
