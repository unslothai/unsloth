# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which httpx client a managed account gets, and that the choice follows the switch live.

The pin is not merely an optimisation: with the switch on it would refuse the very
connection the owner allowed, so the selection has to move with the setting rather
than being decided once at import.
"""

from pathlib import Path
import sys

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import external_provider
from utils.account_context import OWNER, AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def fresh_singleton(monkeypatch):
    monkeypatch.setattr(external_provider, "_managed_http_client", None, raising = False)
    yield
    external_provider._managed_http_client = None


@pytest.fixture
def switch(monkeypatch):
    state = {"allowed": False}
    from utils import managed_provider_url_settings

    # Patched on the settings module rather than on a caller: `_client` imports the helper per call,
    # which is what makes a flip take effect without a restart, so a caller-side patch would miss.
    monkeypatch.setattr(
        managed_provider_url_settings,
        "get_managed_private_provider_urls_allowed",
        lambda: state["allowed"],
    )
    return state


def client_as(account):
    token = bind_account(account)
    try:
        return external_provider._client()
    finally:
        reset_account(token)


def is_pinned(client) -> bool:
    return isinstance(getattr(client, "_transport", None), external_provider._PinnedPublicTransport)


def test_owner_always_gets_the_shared_client(switch):
    for allowed in (False, True):
        switch["allowed"] = allowed
        assert client_as(OWNER) is external_provider._http_client


def test_managed_account_is_pinned_by_default(switch):
    assert is_pinned(client_as(ALICE))


def test_managed_account_is_unpinned_when_allowed(switch):
    switch["allowed"] = True
    assert client_as(ALICE) is external_provider._http_client


def test_the_choice_follows_a_live_flip(switch):
    """A stale module-level singleton must not outlive the setting that selected it."""
    assert is_pinned(client_as(ALICE))
    switch["allowed"] = True
    assert client_as(ALICE) is external_provider._http_client
    switch["allowed"] = False
    assert is_pinned(client_as(ALICE))


def test_the_pinning_client_is_reused_not_rebuilt(switch):
    """Rebuilding per call would drop every pooled connection."""
    assert client_as(ALICE) is client_as(ALICE)


def test_an_unbound_thread_defaults_to_owner(switch):
    """Documents the fail-open default of the account ContextVar: a worker thread that
    forgot run_as is treated as the owner, which the switch does not change but which
    any caller of _client() off the request path must keep in mind."""
    import threading

    seen = []

    def worker():
        seen.append(external_provider._client())

    token = bind_account(ALICE)
    try:
        thread = threading.Thread(target = worker)
        thread.start()
        thread.join()
    finally:
        reset_account(token)
    assert seen[0] is external_provider._http_client
