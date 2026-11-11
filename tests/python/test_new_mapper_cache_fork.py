# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import threading
import time

import pytest

from unsloth.models import loader_utils


@pytest.mark.skipif(not hasattr(os, "fork"), reason = "needs os.fork")
def test_a_fork_during_a_mapper_fetch_does_not_deadlock_the_child(monkeypatch):
    """A worker forked while another thread is fetching the upstream mapper must still resolve names."""
    started, release = threading.Event(), threading.Event()

    def _slow_fetch():
        started.set()
        release.wait(30)
        return ({}, {}, {}, {}, {})

    monkeypatch.setattr(loader_utils, "_get_new_mapper", _slow_fetch)
    monkeypatch.setattr(loader_utils, "_env_says_offline", lambda: False)
    fetcher = threading.Thread(
        target = loader_utils.get_model_name, args = ("org/parent-model",), daemon = True
    )
    fetcher.start()
    assert started.wait(10)
    pid = os.fork()
    if pid == 0:
        try:
            loader_utils._get_new_mapper = lambda: ({}, {}, {}, {}, {})
            loader_utils.get_model_name("org/child-model", load_in_4bit = False)
        finally:
            os._exit(0)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                assert os.waitstatus_to_exitcode(status) == 0
                break
            time.sleep(0.05)
        else:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            pytest.fail("the forked child hung on the mapper lock")
    finally:
        release.set()
        fetcher.join(10)
