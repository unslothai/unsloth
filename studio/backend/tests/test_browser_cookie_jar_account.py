# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The browser panel's site cookies belong to the account, not to its (reusable) username."""

import asyncio

import pytest

import routes.browser as browser_mod
import routes.browser_cookies as cookies
from utils.account_context import AccountContext, bind_account, reset_account

PAGE = "https://mail.example.com/"


class _Client:
    async def is_disconnected(self):
        return False


@pytest.fixture(autouse = True)
def _site(monkeypatch):
    cookies._jars.clear()
    sent = []

    def fake_fetch(url, **kwargs):
        sent.append(kwargs["host_headers"](url.split("/")[2]).get("Cookie"))
        kwargs["meta_out"].update(url = url, status = 200, headers = [], allow_origin = None)
        return None, b"<html></html>", "text/html"

    monkeypatch.setattr(browser_mod, "_fetch_url_raw", fake_fetch)
    yield sent
    cookies._jars.clear()


def _as(account_id, username, make_call):
    async def run():
        token = bind_account(AccountContext(account_id, username))
        try:
            return await make_call()
        finally:
            reset_account(token)

    return asyncio.run(run())


def _sign_in(account_id, username, value):
    _as(
        account_id,
        username,
        lambda: browser_mod.browser_page_cookie(
            browser_mod.BrowserPageCookie(page = PAGE, key = "sid=/", value = f"|{value}"),
            current_subject = username,
        ),
    )


def _cookie_sent(sent, account_id, username):
    _as(
        account_id,
        username,
        lambda: browser_mod.browser_fetch(
            browser_mod.BrowserFetchRequest(url = PAGE), _Client(), current_subject = username
        ),
    )
    return sent[-1]


def test_a_recreated_username_does_not_inherit_the_deleted_accounts_cookies(_site):
    _sign_in("a" * 32, "alice", "old-account-session")
    assert _cookie_sent(_site, "a" * 32, "alice") == "sid=old-account-session"
    # The owner deletes alice, then creates a new account that reuses the name.
    assert _cookie_sent(_site, "b" * 32, "alice") is None


def test_a_rename_keeps_the_accounts_cookies_and_frees_the_old_name(_site):
    _sign_in("a" * 32, "alice", "s1")
    assert _cookie_sent(_site, "a" * 32, "alice2") == "sid=s1"
    assert _cookie_sent(_site, "c" * 32, "alice") is None


def test_clear_browsing_data_clears_the_accounts_jar(_site):
    _sign_in("a" * 32, "alice", "s1")
    _as("a" * 32, "alice", lambda: browser_mod.browser_clear_cookies(current_subject = "alice"))
    assert _cookie_sent(_site, "a" * 32, "alice") is None
