# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio run` has to stop on a signal, and stop cleanly.

Measured on a live two-Spark run: SIGINT did not end the process within 60 to 90 s,
five attempts out of five, and only SIGTERM did -- which, with no handler installed,
is a hard kill: the lifespan shutdown never ran and the peer's ggml-rpc-server was
left holding the peer's GPU. These pin the two halves of the fix: both signals are
handled and run the same graceful shutdown, and the command leaves through os._exit
so nothing in interpreter shutdown can hold the process open after the cleanup.
"""

from __future__ import annotations

import inspect
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def studio_mod():
    from unsloth_cli.commands import studio as _studio

    saved = {}
    for sig in _studio._RUN_SHUTDOWN_SIGNALS:
        saved[sig] = signal.getsignal(sig)
    yield _studio
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _fake_run_mod():
    calls = []
    return SimpleNamespace(
        _server = object(),
        _shutdown_event = threading.Event(),
        _graceful_shutdown = lambda server: calls.append(server),
        calls = calls,
    )


def test_both_signals_are_handled_not_only_the_keyboard_interrupt(studio_mod):
    """SIGTERM with no handler is a hard kill, and that is what orphaned the peer."""
    run_mod = _fake_run_mod()
    studio_mod._install_run_shutdown_handlers(run_mod)
    for sig in (signal.SIGINT, signal.SIGTERM):
        handler = signal.getsignal(sig)
        assert callable(handler), f"{sig!r} must be handled"
        assert handler not in (signal.SIG_DFL, signal.SIG_IGN)


def test_the_handler_cleans_up_once_and_lets_a_second_signal_force_quit(studio_mod):
    run_mod = _fake_run_mod()
    request_shutdown = studio_mod._install_run_shutdown_handlers(run_mod)

    request_shutdown(signal.SIGTERM, None)
    assert run_mod.calls == [run_mod._server], "the graceful shutdown runs, with the server"
    assert run_mod._shutdown_event.is_set(), "the wait loop is woken"
    for sig in (signal.SIGINT, signal.SIGTERM):
        assert signal.getsignal(sig) is signal.SIG_DFL, (
            "the default disposition is restored first, so an impatient second signal "
            "force-quits a shutdown that is taking too long"
        )

    request_shutdown(signal.SIGINT, None)
    assert run_mod.calls == [run_mod._server], "cleanup runs once, however many signals arrive"


def test_the_handler_survives_a_run_module_without_a_server_or_an_event(studio_mod):
    """Nothing here may raise inside a signal handler."""
    run_mod = SimpleNamespace(_graceful_shutdown = lambda server: None)
    studio_mod._install_run_shutdown_handlers(run_mod)(signal.SIGINT, None)


def test_the_run_command_installs_the_handlers_and_exits_hard(studio_mod):
    """The wait-for-Ctrl+C block must use the handlers and must not fall into the
    interpreter's own shutdown, where an atexit join on a worker thread still inside
    an ssh or a subprocess wait holds the process for as long as that call takes."""
    source = inspect.getsource(studio_mod.run)
    assert "_install_run_shutdown_handlers(run_mod)" in source
    tail = source[source.index("_install_run_shutdown_handlers(run_mod)") :]
    assert "_wait_for_server_shutdown" in tail
    assert tail.index("_wait_for_server_shutdown") < tail.index(
        "os._exit(0)"
    ), "cleanup first, then the hard exit"
