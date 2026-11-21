# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Clear every Kaggle credential env var before each test, so an exported token cannot make it live."""

from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest

CI_DIR = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "kaggle_t4_ci"
sys.path.insert(0, str(CI_DIR))

# Read from the gate so newly added accounts are covered.
try:
    from gate import DEFAULT_ACCOUNT_ENVS
except Exception:  # noqa: BLE001 - the suite has its own import guards
    DEFAULT_ACCOUNT_ENVS = ("KAGGLE_API_TOKEN", "KAGGLE_API_TOKEN_2")

# The client reads these too, and a stray one authenticates like the token.
_OTHER_CREDENTIAL_ENVS = ("KAGGLE_KEY", "KAGGLE_USERNAME", "KAGGLE_ACCESS_TOKEN_GH")


@pytest.fixture(autouse = True)
def _no_ambient_kaggle_credentials(monkeypatch):
    for name in (*DEFAULT_ACCOUNT_ENVS, *_OTHER_CREDENTIAL_ENVS):
        monkeypatch.delenv(name, raising = False)


_RELEASE_SIGNALS = tuple(
    sig
    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None))
    if sig is not None
)


@pytest.fixture(autouse = True)
def _no_process_wide_release_handlers(monkeypatch):
    """Stop launch's process-wide signal handlers outliving a test; any signal change left behind fails."""
    launch = sys.modules.get("launch")
    if launch is not None and hasattr(launch, "_install_release_handlers"):
        monkeypatch.setattr(launch, "_install_release_handlers", lambda release: None)
    before = {sig: signal.getsignal(sig) for sig in _RELEASE_SIGNALS}
    yield
    changed = [sig.name for sig in _RELEASE_SIGNALS if signal.getsignal(sig) is not before[sig]]
    for sig in _RELEASE_SIGNALS:
        # None: a handler not set from Python, which signal.signal cannot reinstall.
        if before[sig] is not None:
            signal.signal(sig, before[sig])
    assert not changed, f"the test left process-wide handlers installed for {changed}"
