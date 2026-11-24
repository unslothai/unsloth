# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's provider URL policy sees through NAT64, and the IPv6 metadata services stay
refused with the owner's private-URL switch on."""

import threading

import pytest

from core.inference import providers
from storage import studio_db
from utils import managed_provider_url_settings as setting
from utils.account_context import AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.delenv(setting.BLOCK_PRIVATE_ENV, raising = False)
    setting.forget_cached_setting()
    providers._dns_cache.clear()
    providers._dns_in_flight = threading.BoundedSemaphore(providers._DNS_MAX_IN_FLIGHT)
    token = bind_account(ALICE)
    yield
    reset_account(token)
    setting.forget_cached_setting()


def _allow(monkeypatch, allowed: bool) -> None:
    monkeypatch.setattr(setting, "get_managed_private_provider_urls_allowed", lambda: allowed)


@pytest.mark.parametrize("url", ["http://[64:ff9b::a4d:5]:8000/v1", "http://[64:ff9b:1::a4d:5]/v1"])
def test_nat64_spelling_of_a_private_address_is_refused(monkeypatch, url):
    _allow(monkeypatch, False)
    with pytest.raises(ValueError):
        providers.validate_provider_base_url(url)
    with pytest.raises(ValueError):
        providers.public_provider_address(url)


def test_nat64_of_a_public_address_still_works(monkeypatch):
    _allow(monkeypatch, False)
    url = "http://[64:ff9b::808:808]/v1"
    assert providers.validate_provider_base_url(url) == url


@pytest.mark.parametrize(
    "url",
    ["http://[fd00:c1::a9fe:a9fe]/opc/v1", "http://[fd00:ec2::23]/v1", "http://168.63.129.16/v1"],
)
def test_ipv6_metadata_stays_refused_with_the_switch_on(monkeypatch, url):
    _allow(monkeypatch, True)
    with pytest.raises(ValueError) as refusal:
        providers.validate_provider_base_url(url)
    assert "metadata" in str(refusal.value).lower()
