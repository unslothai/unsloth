# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""setsid -f detaches the server from Popen.pid, so pgrep is the only handle left to terminate it."""

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
    """Stubs launch_studio's outside seams; the returned dict records signalled pids and pgrep state."""

    state = {
        "signalled": [],
        "spawned": [],
        "pgrep_finds": True,
        "healthy": False,
        "port_busy": False,
    }

    # `raising = False` so the fixture builds against code lacking the constant and fails on the subject.
    monkeypatch.setattr(lifecycle, "PID_DISCOVERY_TIMEOUT_S", 0.0, raising = False)
    # Stubbed so whatever this machine has on :5399 cannot decide the answer.
    monkeypatch.setattr(
        lifecycle, "port_is_busy", lambda *a, **k: state["port_busy"], raising = False
    )
    monkeypatch.setattr(lifecycle, "_find_unsloth_bin", lambda install: "/bin/true")
    monkeypatch.setattr(lifecycle, "_read_bootstrap_password", lambda *a, **k: "secret")
    monkeypatch.setattr(lifecycle, "wait_for_healthz", lambda *a, **k: state["healthy"])
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: state["spawned"].append(a))

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
    """The control that matters: the ordinary launch must still hand back a running Unsloth."""

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


def test_a_port_that_is_already_serving_is_refused_before_anything_is_launched(launched):
    """A busy port is refused before spawning: healthz and login would both succeed on the older server."""

    launched["port_busy"] = True
    launched["healthy"] = True

    with pytest.raises(RuntimeError) as excinfo:
        launch_studio(launched["install"], 5399, launched["log"], healthz_timeout_s = 1)

    assert "5399" in str(excinfo.value)
    assert launched["spawned"] == []


def test_the_occupied_port_does_not_come_back_as_a_healthy_studio(launched):
    """The consequence, stated as the caller sees it: no `StudioInstall` is returned at all, so
    nothing records a ref against a build it never installed."""

    launched["port_busy"] = True
    launched["healthy"] = True

    with pytest.raises(RuntimeError):
        launch_studio(launched["install"], 5399, launched["log"], healthz_timeout_s = 1)

    assert launched["install"].port is None
    assert launched["signalled"] == []


def test_a_free_port_still_launches(launched):
    """The control: the guard may not refuse the ordinary launch."""

    launched["healthy"] = True

    install = launch_studio(launched["install"], 5399, launched["log"], healthz_timeout_s = 1)

    assert install.pid == STUDIO_PID
    assert install.port == 5399
    assert len(launched["spawned"]) == 1


def test_the_probe_itself_gives_both_answers_against_a_real_socket():
    """The port probe is tested unstubbed on a real socket, so it cannot answer busy for everything."""

    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert lifecycle.port_is_busy(port) is True

    # A just-released port can linger in TIME_WAIT, so the negative case uses a never-bound port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    assert lifecycle.port_is_busy(free_port) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
