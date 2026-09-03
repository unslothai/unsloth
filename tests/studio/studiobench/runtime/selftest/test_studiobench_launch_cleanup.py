# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An Unsloth this harness launched and could not reach is terminated, not abandoned.

THE PROCESS WE LAUNCH IS NOT THE PROCESS WE SPAWN. `launch_studio` runs the server under
`setsid -f`, which always forks and whose parent exits without waiting, so `Popen.pid` belongs to a
`setsid` that is already gone and the server sits in a session of its own that our process group
cannot reach. `pgrep` is the only handle on it -- and it used to be taken AFTER the health check,
so a server that started and stayed unhealthy raised with `install.pid` still None and
`stop_studio` had nothing to kill.

That leak is not idle. It holds the requested port, and Unsloth's own launcher aborts rather than
binding when it finds one of its own servers there (`studio/backend/run.py`, `_resolve_port` with
`avoid_own_studio`), so the next attempt's server exits and `wait_for_healthz` takes its 200 from
the STALE one -- which by then has finished starting. The run then measures the build the previous
attempt installed and records the ref this one asked for.

AND THE PORT CAN BE OCCUPIED WITHOUT ANYTHING HAVING FAILED. `--keep-studio` asks for an Unsloth to
be LEFT RUNNING, so no cleanup reaches it by design and the next run walks into exactly the same
launch: `_discover_pid` pgreps `unsloth studio.*-p <port>` and finds the older process, `/healthz`
answers 200 from it, and `authenticate` retries with `BENCH_PASSWORD` -- which a previous
studiobench run has already rotated that Unsloth to -- so the login succeeds as well. Nothing
downstream can tell which build answered, so an occupied port is refused before anything is
launched rather than reported afterwards.
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

    state = {
        "signalled": [],
        "spawned": [],
        "pgrep_finds": True,
        "healthy": False,
        "port_busy": False,
    }

    # `raising = False` so this fixture also builds against a lifecycle without the constant, which
    # is what makes the test below fail on the unfixed code for the reason it is about rather than
    # on the way in.
    monkeypatch.setattr(lifecycle, "PID_DISCOVERY_TIMEOUT_S", 0.0, raising = False)
    # Stubbed for the same reason and, for every test but the two about it, so that whatever this
    # machine happens to have on :5399 cannot decide the answer.
    monkeypatch.setattr(
        lifecycle, "port_is_busy", lambda *a, **k: state["port_busy"], raising = False
    )
    monkeypatch.setattr(lifecycle, "_find_unsloth_bin", lambda install: "/bin/true")
    monkeypatch.setattr(lifecycle, "_read_bootstrap_password", lambda *a, **k: "secret")
    monkeypatch.setattr(lifecycle, "wait_for_healthz", lambda *a, **k: state["healthy"])
    # Recorded rather than dropped: a launch refused before the spawn has to be shown not to have
    # spawned anything.
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


# ── the port somebody else is already on ────────────────────────────────────────────────────


def test_a_port_that_is_already_serving_is_refused_before_anything_is_launched(launched):
    """The `--keep-studio` case, which no cleanup covers because retention is what was asked for.

    Everything downstream would have agreed the launch worked: `pgrep` finds the older server on
    the same port, `/healthz` answers 200 from it, and `authenticate` reaches it with the password
    a previous run rotated it to. The refusal has to arrive before the spawn.
    """

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
    """The probe, unstubbed, against a listener this test owns.

    A guard that answered "busy" for everything would pass both tests above and refuse every real
    launch, so the two answers are taken from a real socket rather than from the stub.
    """

    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert lifecycle.port_is_busy(port) is True

    # Closed, so the same port is now the negative case. A port the kernel has just released can
    # linger in TIME_WAIT for a connect, which is why the assertion below is on a port that was
    # never bound at all rather than on this one.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    assert lifecycle.port_is_busy(free_port) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
