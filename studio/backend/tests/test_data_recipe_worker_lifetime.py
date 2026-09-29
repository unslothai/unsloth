# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A recipe outlives the request thread that starts it, but remains cancellable."""

import signal
import sys
import threading
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.data_recipe.jobs import manager as manager_mod
from utils import process_lifetime


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "Linux parent-death signal")
@pytest.mark.parametrize("cancel", [False, True], ids = ["complete", "cancel"])
def test_recipe_survives_its_starting_thread(tmp_path, monkeypatch, cancel):
    if not process_lifetime._pdeathsig_available():
        pytest.skip("PR_SET_PDEATHSIG unavailable")

    # Keep the real manager, spawn context and secret-scrubbing/lifetime wrapper.
    # Only the expensive Data Designer target is replaced with a cooperative worker.
    module_name = "recipe_lifetime_probe"
    (tmp_path / f"{module_name}.py").write_text(
        "def run(event_queue, recipe, run):\n"
        "    event_queue.put('ready')\n"
        "    import time\n"
        "    from pathlib import Path\n"
        "    deadline = time.monotonic() + 30\n"
        "    while not Path(run['release']).exists() and time.monotonic() < deadline:\n"
        "        time.sleep(0.01)\n",
        encoding = "utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(
        manager_mod,
        "account_process_spec",
        lambda module, target, env, kwargs: ((module_name, "run", env), kwargs),
    )
    # The test consumes the worker's ready event itself. No progress pump is needed.
    monkeypatch.setattr(manager_mod.JobManager, "_pump_loop", lambda self: None)
    monkeypatch.setattr(process_lifetime, "adopt_pid", lambda pid: None)
    manager = manager_mod.JobManager()
    release = tmp_path / "release"
    errors = []

    def request_thread():
        try:
            manager.start(recipe = {}, run = {"release": str(release)})
            # Do not end the spawning thread until the child has armed PDEATHSIG.
            assert manager._mp_q.get(timeout = 10) == "ready"
        except BaseException as exc:
            errors.append(exc)

    caller = threading.Thread(target = request_thread)
    try:
        caller.start()
        caller.join(timeout = 15)
        assert not caller.is_alive(), "job start did not return"
        assert not errors, errors
        worker = manager._proc
        assert worker is not None
        worker.join(timeout = 0.5)
        assert worker.is_alive(), f"request thread exit killed the worker: {worker.exitcode}"
        if cancel:
            assert manager.cancel(manager.get_current_job_id())
        else:
            release.touch()
        worker.join(timeout = 5)
        assert not worker.is_alive()
        assert worker.exitcode == (-signal.SIGTERM if cancel else 0)
    finally:
        release.touch()
        caller.join(timeout = 15)
        if manager._proc is not None:
            if manager._proc.is_alive():
                manager._proc.kill()
            manager._proc.join(timeout = 5)
        if manager._mp_q is not None:
            manager._mp_q.close()
            manager._mp_q.join_thread()
