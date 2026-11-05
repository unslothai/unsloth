# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""open_login against a scripted page: an ERR_ABORTED from the app's own logout redirect must not use up
the retry a stalled /login load needs, and a redirect that already loaded /login is used as is."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

DRIVER_PATH = Path(__file__).resolve().parent / "playwright_multi_account.py"
ABORTED = "Page.goto: net::ERR_ABORTED at http://127.0.0.1:8000/login"


def _driver():
    spec = importlib.util.spec_from_file_location(
        "playwright_multi_account_under_test", DRIVER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScriptedPage:
    """Each goto takes the next outcome: None loads /login, a URL is where the app's redirect lands
    before the abort, and an exception class is raised."""

    def __init__(self, base_url: str, outcomes: list):
        self.base_url = base_url
        self.outcomes = list(outcomes)
        self.url = f"{base_url}/chat"
        self.gotos = 0

    def on(self, *_):
        pass

    def remove_listener(self, *_):
        pass

    def wait_for_load_state(self, *_):
        pass

    def evaluate(self, *_):
        return "loading"

    def goto(self, url: str):
        self.gotos += 1
        outcome = self.outcomes.pop(0)
        if outcome is None:
            self.url = url
            return
        if isinstance(outcome, str):
            self.url = f"{self.base_url}{outcome}"
            raise sync_api.Error(ABORTED)
        raise outcome("Page.goto: Timeout 30000ms exceeded.")


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.delenv("PW_ART_DIR", raising = False)
    return _driver()


def test_a_redirect_that_already_loaded_login_is_used(driver):
    page = ScriptedPage(driver.BASE_URL, ["/login"])
    driver.open_login(page)
    assert page.gotos == 1 and page.url.endswith("/login")


def test_a_timeout_after_an_abort_still_gets_its_retry(driver):
    # The CI failure: the redirect aborts the first goto without reaching /login, the next load stalls.
    page = ScriptedPage(driver.BASE_URL, ["/chat", sync_api.TimeoutError, None])
    driver.open_login(page)
    assert page.gotos == 3 and page.url.endswith("/login")


def test_a_redirect_to_change_password_navigates_to_login(driver):
    page = ScriptedPage(driver.BASE_URL, ["/change-password", None])
    driver.open_login(page)
    assert page.gotos == 2 and page.url.endswith("/login")


@pytest.mark.parametrize(
    "outcomes",
    [
        ["/chat", "/chat"],
        [sync_api.TimeoutError, sync_api.TimeoutError],
        ["/chat", sync_api.TimeoutError, sync_api.TimeoutError],
    ],
    ids = ["two aborts", "two timeouts", "abort then two timeouts"],
)
def test_a_second_failure_of_the_same_kind_is_real(driver, outcomes):
    page = ScriptedPage(driver.BASE_URL, outcomes)
    with pytest.raises(sync_api.Error):
        driver.open_login(page)
    assert page.gotos == len(outcomes)


def test_an_error_other_than_an_abort_is_not_retried(driver):
    page = ScriptedPage(driver.BASE_URL, [sync_api.Error])
    with pytest.raises(sync_api.Error):
        driver.open_login(page)
    assert page.gotos == 1
