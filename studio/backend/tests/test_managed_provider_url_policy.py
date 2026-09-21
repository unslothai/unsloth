# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The owner's switch for managed-account provider base URLs on private addresses (#11382).

Default off, so every installation keeps the refusal that shipped. Turned on, the three places that
enforce it -- the validator, the pinned transport and the recipe egress guard -- all stand down
together, because a connection that saves and then fails at send time is worse than one that never
saved. Cloud metadata endpoints stay refused either way, and the shared-host environment opt-in
outranks the switch.
"""

import threading

import pytest

from core.inference import external_provider, providers
from storage import studio_db
from utils import managed_provider_url_settings as setting
from utils.account_context import OWNER, AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")

PRIVATE_URLS = [
    "http://127.0.0.1:11434/v1",
    "http://192.168.1.50:8000/v1",
    "http://10.1.2.3:8000/v1",
]
METADATA_URLS = [
    "http://169.254.169.254/v1",
    "http://metadata.google.internal/v1",
]


@pytest.fixture(autouse = True)
def _clean_resolver_state(monkeypatch, tmp_path):
    # An isolated store per test. Without it the tests here that exercise the real setter write
    # into whatever installation happens to be on the machine running them.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.delenv(setting.BLOCK_PRIVATE_ENV, raising = False)
    setting.forget_cached_setting()
    providers._dns_cache.clear()
    providers._dns_in_flight = threading.BoundedSemaphore(providers._DNS_MAX_IN_FLIGHT)
    yield
    setting.forget_cached_setting()
    providers._dns_cache.clear()


@pytest.fixture
def as_alice():
    token = bind_account(ALICE)
    try:
        yield
    finally:
        reset_account(token)


def _allow(monkeypatch, allowed: bool) -> None:
    """Answer the setting without a DB; the round trip through one is tested separately."""
    monkeypatch.setattr(
        setting, "get_managed_private_provider_urls_allowed", lambda: allowed, raising = True
    )


@pytest.mark.parametrize("url", PRIVATE_URLS)
def test_a_managed_account_is_refused_by_default(monkeypatch, as_alice, url):
    _allow(monkeypatch, False)
    with pytest.raises(ValueError) as refusal:
        providers.validate_provider_base_url(url)
    assert "public-network provider base URLs" in str(refusal.value)
    # The person reading this cannot lift it themselves, so it says who can.
    assert "Settings > General" in str(refusal.value)


@pytest.mark.parametrize("url", PRIVATE_URLS)
def test_a_managed_account_may_use_a_private_url_once_the_owner_allows_it(
    monkeypatch, as_alice, url
):
    _allow(monkeypatch, True)
    assert providers.validate_provider_base_url(url) == url


@pytest.mark.parametrize("url", PRIVATE_URLS)
def test_the_owner_is_unaffected_either_way(monkeypatch, url):
    for allowed in (False, True):
        _allow(monkeypatch, allowed)
        assert providers.validate_provider_base_url(url) == url


@pytest.mark.parametrize("url", METADATA_URLS)
def test_cloud_metadata_stays_refused_with_the_setting_on(monkeypatch, as_alice, url):
    """The switch is about the owner's LAN, never about the host's credentials endpoint."""
    _allow(monkeypatch, True)
    with pytest.raises(ValueError) as refusal:
        providers.validate_provider_base_url(url)
    assert "metadata" in str(refusal.value).lower()


@pytest.mark.parametrize("url", PRIVATE_URLS)
def test_the_environment_opt_in_outranks_the_setting(monkeypatch, as_alice, url):
    """A shared host says no for everyone, and a settings page does not undo the environment."""
    monkeypatch.setenv(setting.BLOCK_PRIVATE_ENV, "1")
    setting.set_managed_private_provider_urls_allowed(True)
    assert setting.get_managed_private_provider_urls_allowed() is False
    with pytest.raises(ValueError):
        providers.validate_provider_base_url(url)


def test_the_owner_never_pays_for_the_setting_read(monkeypatch):
    """The owner path answers before the store is touched, so a single-user install reads nothing."""

    def _explode():
        raise AssertionError("the owner path must not read the setting")

    monkeypatch.setattr(setting, "get_managed_private_provider_urls_allowed", _explode)
    assert external_provider._client() is external_provider._http_client


@pytest.fixture
def settings_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import routes.settings as settings_routes

    app = FastAPI()
    app.include_router(settings_routes.router)
    app.dependency_overrides[settings_routes.get_current_subject] = lambda: "unsloth"
    # An interactive owner at the console, which is what the write requires.
    app.dependency_overrides[settings_routes.authenticated_via_api_key] = lambda: False
    with TestClient(app, raise_server_exceptions = False) as client:
        yield client


def test_the_owner_route_reads_and_writes_the_real_store(settings_client):
    """The route module is unit-tested against a stub elsewhere; this is the round trip."""
    body = settings_client.get("/managed-provider-urls").json()
    assert body["allowed"] is False
    assert body["default_allowed"] is False
    assert body["locked_by_environment"] is False

    updated = settings_client.put("/managed-provider-urls", json = {"allowed": True})
    assert updated.status_code == 200, updated.text
    assert updated.json()["allowed"] is True
    assert settings_client.get("/managed-provider-urls").json()["allowed"] is True
    assert setting.get_managed_private_provider_urls_allowed() is True


def test_the_route_reports_the_environment_lock(monkeypatch, settings_client):
    """Stored on, held off by the environment: the UI needs to say why, not show a switch that lies."""
    settings_client.put("/managed-provider-urls", json = {"allowed": True})
    monkeypatch.setenv(setting.BLOCK_PRIVATE_ENV, "1")
    body = settings_client.get("/managed-provider-urls").json()
    assert body["allowed"] is False
    assert body["locked_by_environment"] is True


def test_the_recipe_endpoint_check_stands_down_with_the_setting_on(monkeypatch, as_alice):
    from core.data_recipe import service

    _allow(monkeypatch, False)
    with pytest.raises(Exception) as refusal:
        service._require_public_provider_endpoint("http://192.168.1.50:8000/v1")
    assert "Settings > General" in str(getattr(refusal.value, "detail", refusal.value))

    _allow(monkeypatch, True)
    service._require_public_provider_endpoint("http://192.168.1.50:8000/v1")


@pytest.mark.parametrize(
    "url", ["http://169.254.169.254/latest/meta-data/", "http://2852039166/v1"]
)
def test_the_recipe_endpoint_check_still_refuses_metadata_with_the_setting_on(
    monkeypatch, as_alice, url
):
    """The recipe path has no validator behind it, so what it waves through is dialled."""
    from core.data_recipe import service

    _allow(monkeypatch, True)
    with pytest.raises(Exception) as refusal:
        service._require_public_provider_endpoint(url)
    assert "metadata" in str(getattr(refusal.value, "detail", refusal.value)).lower()


def test_the_recipe_egress_guard_narrows_rather_than_standing_down(monkeypatch, as_alice):
    """With the setting on the guard stays installed and keeps refusing the metadata service."""
    import socket

    from core.data_recipe import service

    original = socket.getaddrinfo
    monkeypatch.setattr(socket, "getaddrinfo", original)
    _allow(monkeypatch, True)
    try:
        service.install_public_egress_guard()
        guarded = socket.getaddrinfo
        assert guarded is not original

        # A LAN answer is what the switch bought; the metadata answer is not.
        monkeypatch.setattr(
            service.socket if hasattr(service, "socket") else socket,
            "getaddrinfo",
            guarded,
            raising = False,
        )
        assert guarded("192.168.1.50", 8000, type = socket.SOCK_STREAM)
        with pytest.raises(socket.gaierror) as refusal:
            guarded("169.254.169.254", 80, type = socket.SOCK_STREAM)
        assert "metadata" in str(refusal.value).lower()
    finally:
        socket.getaddrinfo = original


def test_the_recipe_guard_follows_a_later_flip(monkeypatch, as_alice):
    """A worker outlives the switch it started under, so the guard asks per lookup."""
    import socket

    from core.data_recipe import service

    original = socket.getaddrinfo
    monkeypatch.setattr(socket, "getaddrinfo", original)
    # The real store, not a stubbed helper: the guard closes over the function it imported at
    # install time, so a test that rebinds that name proves nothing about a running worker.
    setting.set_managed_private_provider_urls_allowed(True)
    try:
        service.install_public_egress_guard()
        guarded = socket.getaddrinfo
        assert guarded("192.168.1.50", 8000, type = socket.SOCK_STREAM)

        # The owner turns it off while this worker is still running.
        setting.set_managed_private_provider_urls_allowed(False)
        with pytest.raises(socket.gaierror) as refusal:
            guarded("192.168.1.50", 8000, type = socket.SOCK_STREAM)
        assert "public-network" in str(refusal.value)
    finally:
        socket.getaddrinfo = original


@pytest.mark.parametrize("url", PRIVATE_URLS)
def test_the_refusal_stops_naming_the_switch_when_the_environment_holds_it(
    monkeypatch, as_alice, url
):
    """Sending someone to a control their owner cannot use either is worse than saying nothing."""
    monkeypatch.setenv(setting.BLOCK_PRIVATE_ENV, "1")
    with pytest.raises(ValueError) as refusal:
        providers.validate_provider_base_url(url)
    assert "Settings > General" not in str(refusal.value)

    monkeypatch.delenv(setting.BLOCK_PRIVATE_ENV)
    _allow(monkeypatch, False)
    with pytest.raises(ValueError) as refusal:
        providers.validate_provider_base_url(url)
    assert "Settings > General" in str(refusal.value)
