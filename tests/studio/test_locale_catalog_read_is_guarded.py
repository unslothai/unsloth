# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A locale catalog that fails to load must fail the leg with something to act on.

`tests/studio/playwright_data_settings.py` walks twelve locales and reads each one's
catalog out of the i18n store. Every catalog but `en` is a lazy import of its own, so a
hiccup fetching one leaves `setLocale` reporting the failure and `messages[locale]`
unset. Reaching straight into it threw

    Page.evaluate: TypeError: Cannot read properties of undefined (reading 'settings')

which names neither the locale nor the cause, out of an eval, on a Windows runner
(Frontend CI 35478582784, Chromium leg, on `main`). The driver now retries the load once
-- the loader keeps a retry URL for exactly this -- and then raises an error that says
which locale, whether the store thinks the catalog failed, and what `setLocale` returned.

A Playwright run needs a browser and a dev server, so what is checked here is the shape
of the driver: that the guard is still in front of every read of a catalog by locale.
"""

from __future__ import annotations

import re
from pathlib import Path

DRIVER = Path(__file__).resolve().parent / "playwright_data_settings.py"

# `api.messages[locale]`, however it is spaced, and whatever is read off it.
_CATALOG_READ = re.compile(r"api\.messages\[\s*locale\s*\]")
_GUARD = re.compile(r"api\.messages\[\s*locale\s*\]\s*===\s*undefined")


def _source() -> str:
    return DRIVER.read_text(encoding = "utf-8")


def test_the_driver_still_reads_a_catalog_by_locale():
    """If this stops being true the test below passes for the wrong reason."""
    source = _source()
    assert _CATALOG_READ.search(
        source
    ), f"{DRIVER.name} no longer reads api.messages[locale]; this guard is stale"


def test_every_catalog_read_sits_behind_the_check_that_gives_up():
    """Measured against the LAST guard, the one that throws, not the first.

    The first `=== undefined` check only decides whether to retry: a read placed
    between it and the give-up still dereferences a catalog that failed twice, and
    still produces the original unhelpful TypeError. What every read has to be after
    is the point where a still-missing catalog stops the driver.
    """
    source = _source()
    guards = [match.start() for match in _GUARD.finditer(source)]
    assert guards, (
        f"{DRIVER.name} reads api.messages[locale] without ever checking it is defined, "
        f"so a catalog that fails to load reports a TypeError from inside an eval "
        f"instead of naming the locale"
    )
    gives_up = source.index("catalog never loaded")
    assert max(guards) < gives_up, (
        f"{DRIVER.name}: the last `=== undefined` check is at {max(guards)}, after the "
        f"give-up at {gives_up}; the throw must be what a failed check reaches"
    )
    reads = [
        match.start()
        for match in _CATALOG_READ.finditer(source)
        if not _GUARD.match(source, match.start())
    ]
    assert reads, f"{DRIVER.name}: no unguarded-looking read found; this guard is stale"
    early = [offset for offset in reads if offset < gives_up]
    assert not early, (
        f"{DRIVER.name} reads api.messages[locale] at {early}, before the give-up at "
        f"{gives_up}: a catalog that failed both attempts is read as undefined there, "
        f"which is the TypeError this guard exists to keep out"
    )


def test_the_failure_names_the_locale_and_what_the_store_reported():
    """The whole point of the change: an error a reader can act on."""
    source = _source()
    index = source.index("catalog never loaded")
    window = source[index - 200 : index + 400]
    for needed in ("' + locale + '", "getLocaleCatalogFailed", "setLocale="):
        assert needed in window, (
            f"the catalog-load failure no longer reports {needed!r}, which is what made "
            f"the original TypeError impossible to act on"
        )


def test_the_load_is_retried_before_it_is_called_a_failure():
    """One transient fetch must not fail a leg; the loader keeps a retry URL for it."""
    source = _source()
    index = source.index("catalog never loaded")
    before = source[:index]
    assert (
        before.count("await api.setLocale(locale)") >= 2
    ), "the driver gives up on the first failed catalog load; it must ask once more"
