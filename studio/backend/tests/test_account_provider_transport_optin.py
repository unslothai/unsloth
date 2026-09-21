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
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture(autouse = True)
def fresh_singleton(monkeypatch):
    monkeypatch.setattr(external_provider, "_managed_clients", {}, raising = False)
    yield
    external_provider._managed_clients = {}


@pytest.fixture
def switch(monkeypatch):
    state = {"allowed": False}
    from utils import managed_provider_url_settings

    # Patched on the settings module: `_client` imports the helper per call, so a caller-side
    # patch would miss.
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
    """Public-only. Exact type: the allowed-private screen subclasses this, so isinstance would
    call that pinned too and every assertion below would stop meaning anything."""
    return type(getattr(client, "_transport", None)) is external_provider._PinnedPublicTransport


def test_owner_always_gets_the_shared_client(switch):
    for allowed in (False, True):
        switch["allowed"] = allowed
        assert client_as(OWNER) is external_provider._http_client


def test_managed_account_is_pinned_by_default(switch):
    assert is_pinned(client_as(ALICE))


def test_managed_account_stops_being_held_to_public_when_allowed(switch):
    """The public pin comes off; a screen that still refuses metadata replaces it."""
    switch["allowed"] = True
    client = client_as(ALICE)
    assert not is_pinned(client)
    assert isinstance(client._transport, external_provider._PinnedNonMetadataTransport)
    assert client is not external_provider._http_client


def test_the_choice_follows_a_live_flip(switch):
    """A stale module-level singleton must not outlive the setting that selected it."""
    assert is_pinned(client_as(ALICE))
    switch["allowed"] = True
    assert isinstance(client_as(ALICE)._transport, external_provider._PinnedNonMetadataTransport)
    switch["allowed"] = False
    assert is_pinned(client_as(ALICE))


def test_the_pinning_client_is_reused_not_rebuilt(switch):
    """Rebuilding per call would drop every pooled connection."""
    assert client_as(ALICE) is client_as(ALICE)


def test_two_managed_accounts_never_share_a_client(switch):
    """httpx persists cookies per client, so one shared by two accounts carries a gateway
    session from one to the other. https://www.python-httpx.org/advanced/clients/"""
    for allowed in (False, True):
        switch["allowed"] = allowed
        alice, bob = client_as(ALICE), client_as(BOB)
        assert alice is not bob

        alice.cookies.set("gateway_session", "ALICE", domain = "gw.example")
        try:
            assert bob.cookies.get("gateway_session", domain = "gw.example") is None
        finally:
            alice.cookies.clear()


def test_an_unbound_thread_defaults_to_owner(switch):
    """The account ContextVar fails open: a worker thread that forgot run_as reads as the owner.
    Unchanged by the switch, and load-bearing for any _client() caller off the request path."""
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


def test_the_allowed_client_is_never_the_owners_object(switch):
    """An httpx client carries a cookie jar, so sharing the object shares Set-Cookie across accounts."""
    switch["allowed"] = True
    managed = client_as(ALICE)
    assert managed is not external_provider._http_client
    assert managed.cookies is not external_provider._http_client.cookies

    external_provider._http_client.cookies.set("owner_session", "SECRET", domain = "gw.example")
    try:
        assert managed.cookies.get("owner_session", domain = "gw.example") is None
    finally:
        external_provider._http_client.cookies.clear()


def test_the_allowed_client_still_screens_every_connection(switch):
    """The switch opens private addresses, not the host's credentials endpoint."""
    switch["allowed"] = True
    managed = client_as(ALICE)
    transport = managed._transport
    assert isinstance(transport, external_provider._PinnedNonMetadataTransport)

    from core.inference.providers import provider_address_excluding_metadata

    # Private is now fine, metadata never is, whichever spelling it arrives in.
    assert provider_address_excluding_metadata("http://127.0.0.1:11434/v1") == "127.0.0.1"
    assert provider_address_excluding_metadata("http://192.168.1.50:8000/v1") == "192.168.1.50"
    for metadata in (
        "http://169.254.169.254/v1",
        "http://2852039166/v1",
        "http://[fd00:ec2::254]/v1",
    ):
        with pytest.raises(ValueError) as refusal:
            provider_address_excluding_metadata(metadata)
        assert "metadata" in str(refusal.value).lower()
