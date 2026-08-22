# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Studio this harness launched and could not reach is terminated, not abandoned.

THE PROCESS WE LAUNCH IS NOT THE PROCESS WE SPAWN. `launch_studio` runs the server under
`setsid -f`, which always forks and whose parent exits without waiting, so `Popen.pid` belongs to a
`setsid` that is already gone and the server sits in a session of its own that our process group
cannot reach. `pgrep` is the only handle on it -- and it used to be taken AFTER the health check,
so a server that started and stayed unhealthy raised with `install.pid` still None and
`stop_studio` had nothing to kill.

That leak is not idle. It holds the requested port, and Studio's own launcher aborts rather than
binding when it finds one of its own servers there (`studio/backend/run.py`, `_resolve_port` with
`avoid_own_studio`), so the next attempt's server exits and `wait_for_healthz` takes its 200 from
the STALE one -- which by then has finished starting. The run then measures the build the previous
attempt installed and records the ref this one asked for.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.runtime import lifecycle  # noqa: E402
from studiobench.runtime.lifecycle import StudioInstall, launch_studio  # noqa: E402

STUDIO_PID = 4242


@pytest.fixture
def launched(monkeypatch, tmp_path):
    """Everything `launch_studio` reaches outside this process, stubbed at the seam it uses.

    Returns a dict the test reads back: which pids were signalled, and whether the server was
    running at all when `pgrep` was asked.
    """

    state = {"signalled": [], "pgrep_finds": True, "healthy": False}

    # `raising = False` so this fixture also builds against a lifecycle without the constant, which
    # is what makes the test below fail on the unfixed code for the reason it is about rather than
    # on the way in.
    monkeypatch.setattr(lifecycle, "PID_DISCOVERY_TIMEOUT_S", 0.0, raising = False)
    monkeypatch.setattr(lifecycle, "_find_unsloth_bin", lambda install: "/bin/true")
    monkeypatch.setattr(lifecycle, "_read_bootstrap_password", lambda *a, **k: "secret")
    monkeypatch.setattr(lifecycle, "wait_for_healthz", lambda *a, **k: state["healthy"])
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: None)

    def fake_run(cmd, *a, **k):
        assert cmd[0] == "pgrep", cmd
        out = f"{STUDIO_PID}\n" if state["pgrep_finds"] else ""
        return subprocess.CompletedProcess(cmd, 0, stdout = out, stderr = "")

    monkeypatch.setattr(lifecycle, "_run", fake_run)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: state["signalled"].append((pgid, sig)))

    state["install"] = StudioInstall(home = tmp_path / "home", repo = tmp_path / "repo", branch = "main")
    state["log"] = tmp_path / "studio.log"
    return state


def test_a_studio_that_never_answers_healthz_is_terminated(launched):
    launched["healthy"] = False

    with pytest.raises(TimeoutError):
        launch_studio(launched["install"], 5399, launched["log"], healthz_timeout_s = 1)

    assert launched["install"].pid == STUDIO_PID
    assert [pgid for pgid, _sig in launched["signalled"]] == [STUDIO_PID]


def test_a_studio_that_never_started_at_all_still_raises(launched):
    """The control for the discovery itself: nothing to find is not a reason to crash on the way
    to reporting the timeout."""

    launched["healthy"] = False
    launched["pgrep_finds"] = False

    with pytest.raises(TimeoutError):
        launch_studio(launched["install"], 5399, launched["log"], healthz_timeout_s = 1)

    assert launched["install"].pid is None
    assert launched["signalled"] == []


def test_a_healthy_studio_is_returned_with_its_pid_and_is_not_signalled(launched):
    """The control that matters: the ordinary launch must still hand back a running Studio."""

    launched["healthy"] = True

    install = launch_studio(launched["install"], 5399, launched["log"], healthz_timeout_s = 1)

    assert install.pid == STUDIO_PID
    assert install.port == 5399
    assert install.base_url == "http://127.0.0.1:5399"
    assert install.bootstrap_password == "secret"
    assert launched["signalled"] == []


def test_a_healthy_studio_whose_pid_cannot_be_found_is_still_returned(launched):
    """`pgrep` is a best effort and always has been; losing it may not fail a healthy launch."""

    launched["healthy"] = True
    launched["pgrep_finds"] = False

    install = launch_studio(launched["install"], 5399, launched["log"], healthz_timeout_s = 1)

    assert install.pid is None
    assert launched["signalled"] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
