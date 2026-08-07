# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The macOS tab-capability smoke has to be able to fail.

tests/studio/playwright_mac_tab_capabilities.py needs a live Studio and a browser, so
CI is the only place it runs and nothing else checks that a red case comes out red.
Twice now it has gone green having observed nothing: first by authenticating with
nobody, then by computing `seen_spinner` and only logging it, so a backend that
settled before the browser arrived skipped every assertion.

This drives the same functions with the page and the backend stubbed, over the exact
shapes that used to pass: the warm window already shut, the row absent, the row greyed
out. It is a plain pytest file so it runs in the Backend CI walk over tests/, where
neither playwright nor a Studio is installed.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tests/studio/playwright_mac_tab_capabilities.py"
APPEARANCE_STORE = REPO / "studio/frontend/src/features/settings/stores/appearance-custom-store.ts"

BASE = "http://127.0.0.1:18893"
# A settled reply: no hardware_detecting at all. This is the state the runner is in by
# the time the browser is authenticated, and the one the old code passed vacuously on.
SETTLED = {"status": "healthy", "service": "Unsloth UI Backend", "device_type": "mac"}
UNMEASURED = {"status": "healthy", "service": "Unsloth UI Backend", "hardware_detecting": True}

# Spelled out rather than read off the script, so these cases run unchanged against a
# build of it that does not define the constant yet. test_inline_row_ids_match_the_
# frontends_default_pinned_set is what keeps the spelling honest.
TRAIN = "train"

GREYED = {"disabled": True, "spinner": False}
SPINNING = {"disabled": False, "spinner": True}
SETTLED_ENABLED = {"disabled": False, "spinner": False}


def _load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Import the script with playwright and its env contract stubbed out.

    Fresh per test: the script keeps its failures and its row sightings in module
    globals, so a shared instance would carry one test's verdict into the next.
    """
    if "playwright.sync_api" not in sys.modules:
        pkg = types.ModuleType("playwright")
        api = types.ModuleType("playwright.sync_api")
        api.sync_playwright = lambda: None
        pkg.sync_api = api
        monkeypatch.setitem(sys.modules, "playwright", pkg)
        monkeypatch.setitem(sys.modules, "playwright.sync_api", api)
    monkeypatch.setenv("BASE_URL", BASE)
    monkeypatch.setenv("STUDIO_OLD_PW", "stub-password")
    monkeypatch.setenv("PW_ART_DIR", str(tmp_path / "art"))
    # The real default gives the row 15s to settle; the stub answers instantly, so the
    # only thing the wait would buy here is 15s of a red test.
    monkeypatch.setenv("STUDIO_MAC_FORCED_PENDING_S", "0.2")
    spec = importlib.util.spec_from_file_location("mac_tab_capabilities_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeLocator:
    def __init__(self, present: bool) -> None:
        self._present = present
        self.clicked = False

    def count(self) -> int:
        return 1 if self._present else 0

    @property
    def first(self):
        return self

    def is_enabled(self) -> bool:
        return True

    def click(self, timeout = None) -> None:
        self.clicked = True


class FakePage:
    """Enough of a Playwright page for the pending-state checks.

    `rows` maps a nav row id to the DOM state the stub reports, or None for a row that
    is not in the document. `row_missing` makes wait_for_selector time out the way a
    sidebar that never rendered does.
    """

    def __init__(
        self,
        rows: dict,
        *,
        row_missing: bool = False,
    ) -> None:
        self.rows = rows
        self.row_missing = row_missing
        self.url = f"{BASE}/chat"
        self.routed: list[str] = []
        self.unrouted: list[str] = []
        self.gotos: list[str] = []
        self.screenshots: list[str] = []

    def evaluate(
        self,
        script: str,
        arg = None,
    ):
        return {rid: self.rows.get(rid) for rid in (arg or [])}

    def route(self, pattern, handler) -> None:
        self.routed.append(pattern)
        # Prove the stub body is valid JSON and reaches the browser, rather than only
        # that route() was called: a body the frontend cannot parse would leave the row
        # in its pre-fetch state and the check would read the wrong thing.
        handler(_RecordingRoute(self))

    def unroute(
        self,
        pattern,
        handler = None,
    ) -> None:
        self.unrouted.append(pattern)

    def goto(
        self,
        url,
        wait_until = None,
        timeout = None,
    ) -> None:
        self.gotos.append(url)
        self.url = url

    def wait_for_selector(
        self,
        selector,
        timeout = None,
    ) -> None:
        if self.row_missing:
            raise TimeoutError(f"waiting for {selector}")

    def wait_for_timeout(self, ms) -> None:
        pass

    def screenshot(
        self,
        path = None,
        full_page = None,
    ) -> None:
        self.screenshots.append(str(path))

    def locator(self, selector: str):
        rid = selector.split("nav-row-")[1].rstrip('"]')
        return FakeLocator(self.rows.get(rid) is not None)


class _RecordingRoute:
    def __init__(self, page: FakePage) -> None:
        self.page = page

    def fulfill(
        self,
        status = None,
        content_type = None,
        body = None,
    ) -> None:
        import json
        self.page.fulfilled = json.loads(body)
        self.page.fulfilled_status = status


def _health(mod, bodies):
    """Point the script's backend reads at a scripted sequence of /api/health bodies."""
    queue = list(bodies)

    def fake(path, timeout = 10.0):
        body = queue.pop(0) if len(queue) > 1 else queue[0]
        return 200, dict(body)

    mod._get_json = fake


# --------------------------------------------------------------------------------
# The regression Codex found: the window shut before the browser got there.
# --------------------------------------------------------------------------------


def test_greyed_row_fails_even_though_the_warm_window_already_shut(tmp_path, monkeypatch):
    """The vacuous-pass case. Health has settled, so the real-warm sampler observes
    nothing and breaks on its first iteration; the row is blacked out exactly as it was
    in the field. Before the forced-verdict check this run went green."""
    mod = _load(tmp_path, monkeypatch)
    _health(mod, [SETTLED])
    page = FakePage({TRAIN: GREYED})

    mod.assert_row_never_greyed_while_unmeasured(page)

    assert mod._failed, (
        "the run passed with the warm window already shut and the Train row greyed "
        "out; this is the state the whole script exists to catch"
    )
    assert any("disabled" in m for m in mod._failed), mod._failed


def test_row_with_no_spinner_fails(tmp_path, monkeypatch):
    """Enabled but not spinning is still wrong: an unmeasured capability has to read as
    'still checking', not as a settled verdict that happens to allow the click."""
    mod = _load(tmp_path, monkeypatch)
    _health(mod, [SETTLED])
    page = FakePage({TRAIN: SETTLED_ENABLED})

    mod.assert_row_never_greyed_while_unmeasured(page)

    assert any("spinner" in m for m in mod._failed), mod._failed


def test_absent_row_fails_instead_of_skipping(tmp_path, monkeypatch):
    """A row that never renders is the signed-out / unrendered shape. It must not be
    read as 'nothing to check here'."""
    mod = _load(tmp_path, monkeypatch)
    _health(mod, [SETTLED])
    page = FakePage({TRAIN: None}, row_missing = True)

    mod.assert_row_never_greyed_while_unmeasured(page)

    assert any("never rendered" in m for m in mod._failed), mod._failed


def test_spinning_row_passes_and_the_route_is_lifted(tmp_path, monkeypatch):
    """The green case, and the only one there should be: the row spins on a forced
    unmeasured verdict, and the interception is taken back off so the tab walk that
    follows sees the real backend."""
    mod = _load(tmp_path, monkeypatch)
    _health(mod, [SETTLED])
    page = FakePage({TRAIN: SPINNING})

    mod.assert_row_never_greyed_while_unmeasured(page)

    assert mod._failed == []
    assert page.routed == ["**/api/health"]
    assert page.unrouted == ["**/api/health"]
    assert page.gotos == [f"{BASE}/chat"]


def test_forced_body_is_a_real_reply_with_the_measurement_removed(tmp_path, monkeypatch):
    """What the browser is served has to be provisional by env.ts's rules: hardware
    detecting, no device_type, and not the deferred marker (which env.ts reads as
    settled and would grey the row out legitimately)."""
    mod = _load(tmp_path, monkeypatch)
    _health(mod, [{**SETTLED, "hardware_detection_deferred": True, "studio_root_id": "abc"}])
    page = FakePage({TRAIN: SPINNING})

    mod.assert_row_never_greyed_while_unmeasured(page)

    assert page.fulfilled_status == 200
    assert page.fulfilled["hardware_detecting"] is True
    assert page.fulfilled["chat_only"] is True
    assert "device_type" not in page.fulfilled
    assert "hardware_detection_deferred" not in page.fulfilled
    # Untouched fields survive, so the reply differs from a real one only where it must.
    assert page.fulfilled["studio_root_id"] == "abc"


def test_unreadable_health_fails_rather_than_returning_early(tmp_path, monkeypatch):
    """No body to build the provisional reply from means the check did not run. Say so."""
    mod = _load(tmp_path, monkeypatch)
    mod._get_json = lambda path, timeout = 10.0: (0, None)
    page = FakePage({TRAIN: SPINNING})

    mod.assert_row_never_greyed_while_unmeasured(page)

    assert any("provisional" in m for m in mod._failed), mod._failed


# --------------------------------------------------------------------------------
# The real-warm sampler still has to fail when it does catch a grey-out.
# --------------------------------------------------------------------------------


def test_real_warm_grey_out_still_fails(tmp_path, monkeypatch):
    mod = _load(tmp_path, monkeypatch)
    _health(mod, [UNMEASURED, SETTLED])
    page = FakePage({TRAIN: GREYED})

    mod.sample_natural_warm_window(page)

    assert any("hardware_detecting=true" in m for m in mod._failed), mod._failed


def test_sampler_that_cannot_read_the_page_at_all_fails(tmp_path, monkeypatch):
    """The bare `except Exception: break` this replaced turned a dead page into a
    silent zero-observation pass."""
    mod = _load(tmp_path, monkeypatch)
    _health(mod, [UNMEASURED])
    page = FakePage({TRAIN: SPINNING})
    page.evaluate = lambda script, arg = None: (_ for _ in ()).throw(RuntimeError("page closed"))

    mod.sample_natural_warm_window(page)

    assert any("could not read the sidebar" in m for m in mod._failed), mod._failed


def test_missed_warm_window_alone_is_not_a_failure(tmp_path, monkeypatch):
    """Missing the real window is normal and must stay quiet, or the macOS job goes red
    on every run. The forced check above is what carries the guarantee instead."""
    mod = _load(tmp_path, monkeypatch)
    _health(mod, [SETTLED])
    page = FakePage({TRAIN: SPINNING})

    mod.sample_natural_warm_window(page)

    assert mod._failed == []


# --------------------------------------------------------------------------------
# The tab walk: a pinned row that is not there means the tab checked nothing.
# --------------------------------------------------------------------------------


def test_drive_tabs_fails_when_the_pinned_rows_never_render(tmp_path, monkeypatch):
    mod = _load(tmp_path, monkeypatch)
    _health(mod, [SETTLED])
    page = FakePage({})

    mod.drive_tabs(page)

    assert mod._rows_seen == set()
    for row_id in mod.INLINE_ROW_IDS:
        assert any(f"nav row {row_id} is pinned inline" in m for m in mod._failed), mod._failed


def test_drive_tabs_does_not_fail_on_the_rows_that_live_under_more(tmp_path, monkeypatch):
    """Video and Export render inside the More dropdown, which mounts no data-testid at
    all. Their absence is the documented shape, not a miss -- asserting on it would be
    asserting on something that can never be true."""
    mod = _load(tmp_path, monkeypatch)
    _health(mod, [SETTLED])
    page = FakePage({rid: SETTLED_ENABLED for rid in mod.INLINE_ROW_IDS})

    mod.drive_tabs(page)

    assert mod._failed == []
    assert mod._rows_seen == set(mod.INLINE_ROW_IDS)


# --------------------------------------------------------------------------------
# Drift guard: the row asserted on has to be one the sidebar actually pins.
# --------------------------------------------------------------------------------


def test_inline_row_ids_match_the_frontends_default_pinned_set():
    """If a row is unpinned in the store, it stops rendering a data-testid and every
    assertion pinned to it silently becomes unobservable. That is how the Video half of
    this script came to check nothing, and it must not happen again unnoticed."""
    src = APPEARANCE_STORE.read_text(encoding = "utf-8")
    block = re.search(
        r"SIDEBAR_NAV_DEFAULT_PINNED[^{]*\{(.*?)\n\};",
        src,
        re.S,
    )
    assert block, "SIDEBAR_NAV_DEFAULT_PINNED is gone or its shape changed"
    entries = re.findall(r"^\s*(\w+):\s*(true|false),", block.group(1), re.M)
    assert entries, "no id: boolean entries parsed out of SIDEBAR_NAV_DEFAULT_PINNED"
    pinned = {name for name, value in entries if value == "true"}

    mod_ids = _module_constant("INLINE_ROW_IDS")
    assert set(mod_ids) == pinned, (
        f"the script treats {sorted(set(mod_ids))} as pinned inline but the store pins "
        f"{sorted(pinned)}; a row that is not pinned renders no data-testid, so any "
        "assertion on it can only ever observe nothing"
    )
    assert _module_constant("GATED_ROW_ID") in pinned


def _module_constant(name: str):
    """Read a literal constant out of the script without importing it (no playwright,
    no env contract, so this stays usable from a bare collection)."""
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not defined in {SCRIPT.name}")


def test_the_forced_verdict_check_is_wired_into_the_public_entry_point():
    """main() calls assert_row_never_greyed_while_unmeasured, and that has to be what
    reaches the forced check. Splitting them apart without calling both from main is the
    one edit that would restore the vacuous pass while every test above still passes."""
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding = "utf-8"))
    called = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            called[node.name] = {
                sub.func.id
                for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
            }
    assert "assert_pending_state_on_forced_verdict" in called.get(
        "assert_row_never_greyed_while_unmeasured", set()
    )
    assert "assert_row_never_greyed_while_unmeasured" in called.get("main", set())
