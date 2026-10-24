# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ddgs engines get a short connect timeout and keep the whole budget to answer (unslothai/unsloth#15143)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.inference import tools


@pytest.fixture
def fresh_install(monkeypatch):
    """Undo the install after each test, so no test sees another's wrapped clients."""
    from ddgs import http_client, http_client2

    monkeypatch.setattr(http_client, "primp", http_client.primp)
    cls = http_client2.HttpClient2
    monkeypatch.setattr(cls, "__init__", cls.__init__)
    monkeypatch.setattr(cls, "_unsloth_connect_timeout", False, raising = False)
    return http_client, http_client2


@pytest.mark.parametrize(("timeout", "connect"), [(300, 10), (3, 3), (None, 10)])
def test_primp_clients_connect_fast_and_keep_the_read_budget(fresh_install, timeout, connect):
    http_client, _ = fresh_install
    tools._install_ddgs_connect_timeout()

    client = http_client.HttpClient(timeout = timeout).client

    if not hasattr(client, "connect_timeout"):
        pytest.skip("installed primp has no connect_timeout")
    assert client.connect_timeout == connect
    assert client.timeout == timeout


@pytest.mark.parametrize(("timeout", "connect"), [(300, 10), (3, 3), (None, 10)])
def test_httpx_clients_connect_fast_and_keep_the_read_budget(fresh_install, timeout, connect):
    _, http_client2 = fresh_install
    tools._install_ddgs_connect_timeout()

    budget = http_client2.HttpClient2(timeout = timeout).client.timeout

    assert budget.connect == connect
    assert (budget.read, budget.write, budget.pool) == (timeout, timeout, timeout)


def test_installing_twice_wraps_once(fresh_install):
    http_client, http_client2 = fresh_install
    tools._install_ddgs_connect_timeout()
    primp, init = http_client.primp, http_client2.HttpClient2.__init__

    tools._install_ddgs_connect_timeout()

    assert http_client.primp is primp
    assert http_client2.HttpClient2.__init__ is init


def test_a_primp_without_connect_timeout_still_builds_its_client():
    """primp 0.15 (the ddgs 9.8.0 pin) takes no connect_timeout; the search must still run."""
    built = []

    def Client(**kwargs):
        if "connect_timeout" in kwargs:
            raise TypeError("'connect_timeout' is an invalid keyword argument")
        built.append(kwargs)
        return "client"

    primp = tools._PrimpWithConnectTimeout(SimpleNamespace(Client = Client, TimeoutError = OSError))

    assert primp.Client(timeout = 300) == "client"
    assert built == [{"timeout": 300}]
    assert primp.TimeoutError is OSError


def test_an_unreachable_engine_says_it_may_be_blocked():
    from ddgs.exceptions import TimeoutException

    exc = TimeoutException(
        "error sending request for url (https://html.duckduckgo.com/html/) > connection timed out"
    )

    result = tools._search_failure_message(exc, 300)

    assert result.startswith("Search failed: the search engines could not be reached within 10s")
    assert "HTTPS_PROXY" in result
    assert '{"url": "<URL>"}' in result


def test_a_connected_engine_that_answers_too_late_keeps_the_budget_message():
    from ddgs.exceptions import TimeoutException
    exc = TimeoutException("error sending request for url (https://x/) > operation timed out")

    assert (
        tools._search_failure_message(exc, 300)
        == "Search failed: the search engines did not respond within 300s."
    )
