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
        return 200, dict(body), "ok"

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
    mod._get_json = lambda path, timeout = 10.0: (0, None, "refused")
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



# --------------------------------------------------------------------------------
# What the survival poller fails on, now that it no longer replays the watchdog.
# --------------------------------------------------------------------------------

POLL_S = 5.0
BUDGET_S = 10.0


def _timeline(duration_s, stalls = ()):
    """Simulate the poller against a backend that answers nothing during *stalls*.

    Each stall is a (start, end) window in seconds. A probe issued inside one answers
    when the stall lifts if that falls inside its 10s budget, and times out otherwise.
    Probes are sequential and a timeout costs the whole budget before the next route is
    tried, which is what the real poller does.
    """
    def lifts_at(t):
        for start, end in stalls:
            if start <= t < end:
                return end
        return None

    samples, now = [], 0.0
    while now < duration_s:
        for path in ("/api/liveness", "/api/health"):
            end = lifts_at(now)
            if end is None:
                took, kind = 0.005, "ok"
            elif end - now <= BUDGET_S:
                took, kind = end - now, "ok"
            else:
                took, kind = BUDGET_S, "timeout"
            now += took
            samples.append({
                "t": round(now, 3), "path": path, "kind": kind,
                "status": 200 if kind == "ok" else 0,
                "ms": round(took * 1000, 1), "inference_active": None,
                "hardware_detecting": None, "torch_warm_in_progress": None,
            })
        now += POLL_S
    return samples


def _verdict(mod, samples, final_kind = "ok", final_status = 200):
    poller = mod.BackendSurvivalPoller()
    poller.samples = samples
    poller.report(final_kind = final_kind, final_status = final_status)
    return list(mod._failed)


def test_a_stall_the_backend_recovers_from_is_not_a_failure(tmp_path, monkeypatch):
    """The case run 32862298967 went red on. A backend that stops answering and then
    answers again survived, on any reading of any budget, so it warns and passes."""
    mod = _load(tmp_path, monkeypatch)
    samples = _timeline(150, stalls = [(30, 63)])
    assert any(s["kind"] == "timeout" for s in samples), "fixture stopped producing a stall"
    assert _verdict(mod, samples) == []


def test_even_a_long_stall_passes_if_the_backend_comes_back(tmp_path, monkeypatch):
    """No threshold shorter than the launcher's most generous path is decidable in the
    120s this phase gets, so there is deliberately no threshold at all. A stall far past
    anything the old replay would have killed on still passes once the backend answers."""
    mod = _load(tmp_path, monkeypatch)
    assert _verdict(mod, _timeline(220, stalls = [(30, 150)])) == []


def test_a_backend_that_never_answers_again_fails(tmp_path, monkeypatch):
    """The terminal stall. This is what the poller exists to catch and it needs no
    watchdog arithmetic: the backend stopped answering and was still not answering when
    the run ended, confirmed by the final probe."""
    mod = _load(tmp_path, monkeypatch)
    failed = _verdict(mod, _timeline(150, stalls = [(60, 9999)]), final_kind = "timeout", final_status = 0)
    assert len(failed) == 1, failed
    assert "never answered again" in failed[0], failed


def test_a_trailing_stall_the_final_probe_clears_is_not_a_failure(tmp_path, monkeypatch):
    """A stall still in progress when sampling stopped is not a terminal one if the
    backend answers the final probe. Without that probe this would be indistinguishable
    from death, which is why report() takes it rather than guessing from the samples."""
    mod = _load(tmp_path, monkeypatch)
    samples = _timeline(150, stalls = [(60, 9999)])
    assert samples[-1]["kind"] == "timeout", "fixture no longer ends mid-stall"
    assert _verdict(mod, samples, final_kind = "ok") == []


def test_a_dead_backend_with_no_samples_still_fails(tmp_path, monkeypatch):
    """The post-run watch is load-bearing on its own, not only as a tie-breaker."""
    mod = _load(tmp_path, monkeypatch)
    failed = _verdict(mod, _timeline(60), final_kind = "refused", final_status = 0)
    assert len(failed) == 1, failed
    assert "port is gone" in failed[0], failed


def test_a_refused_connection_fails_on_the_first_one(tmp_path, monkeypatch):
    """A refused port is death, not a stall, and nothing transient produces it against a
    backend that is meant to be up. It is fatal even though the backend recovers."""
    mod = _load(tmp_path, monkeypatch)
    samples = _timeline(120)
    hit = [s for s in samples if s["t"] > 50][0]
    hit["kind"], hit["status"], hit["ms"] = "refused", 0, 1.0
    failed = _verdict(mod, samples)
    assert any("refused" in m for m in failed), failed


def test_a_non_200_answer_fails_on_the_first_one(tmp_path, monkeypatch):
    """An answered non-200 is the backend itself saying it is unhealthy. Also not a stall,
    and also fatal even though every other probe in the run succeeded."""
    mod = _load(tmp_path, monkeypatch)
    samples = _timeline(120)
    hit = [s for s in samples if s["t"] > 50][0]
    hit["kind"], hit["status"], hit["ms"] = "http", 503, 4.0
    failed = _verdict(mod, samples)
    assert any("unhealthy" in m for m in failed), failed


def test_only_a_refused_connection_counts_as_a_dead_port(tmp_path, monkeypatch):
    """ECONNREFUSED is the kernel saying nothing is bound. A reset or a truncated read is
    a listener that accepted and then failed to finish, which is a stall."""
    mod = _load(tmp_path, monkeypatch)
    assert mod._transport_kind(ConnectionRefusedError()) == "refused"
    assert mod._transport_kind(ConnectionResetError()) == "timeout"
    assert mod._transport_kind(TimeoutError()) == "timeout"
    assert mod._transport_kind(OSError("something else entirely")) == "timeout"


def test_an_answer_from_either_route_closes_a_stall(tmp_path, monkeypatch):
    """check_health_inner falls back from /api/liveness to /api/health, so one answer
    from either is proof the backend was serving and ends the span."""
    mod = _load(tmp_path, monkeypatch)
    samples = [
        {"t": 10.0, "path": "/api/liveness", "kind": "timeout", "status": 0, "ms": 10000.0,
         "inference_active": None, "hardware_detecting": None, "torch_warm_in_progress": None},
        {"t": 10.1, "path": "/api/health", "kind": "ok", "status": 200, "ms": 5.0,
         "inference_active": None, "hardware_detecting": None, "torch_warm_in_progress": None},
    ]
    spans = mod._stall_windows(samples)
    assert len(spans) == 1, spans
    assert spans[0][2] is False, spans


def test_the_probe_budget_is_the_launchers_own_number():
    """The one watchdog constant this file still mirrors. If HEALTH_PROBE_TIMEOUT moves
    and this does not, a probe here calls a miss at a different point than the product."""
    rust = (REPO / "studio/src-tauri/src/commands.rs").read_text(encoding = "utf-8")
    match = re.search(r"const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs\((\d+)\);", rust)
    assert match, "HEALTH_PROBE_TIMEOUT is not in commands.rs any more"
    assert _module_constant("PROBE_TIMEOUT_S") == float(match.group(1))


def test_the_watchdog_replay_is_gone():
    """It was removed on purpose: mirroring the launcher's state machine made every line
    of commands.rs a correctness requirement here, for a rule this phase never runs and
    could not decide inside its 120s window. Reintroducing it should be a deliberate act,
    not a quiet one."""
    source = SCRIPT.read_text(encoding = "utf-8")
    for gone in ("watchdog_replay", "WATCHDOG_MAX_FAILURES", "WATCHDOG_INTERVAL_S"):
        assert f"def {gone}" not in source and f"\n{gone} =" not in source, gone


# --------------------------------------------------------------------------------
# The post-run watch, which is what stops the window boundary deciding the verdict.
# --------------------------------------------------------------------------------


def _scripted_probes(mod, kinds):
    """Point _get_json at a scripted sequence of outcomes; the last one repeats."""
    seq = list(kinds)
    calls = []

    def fake(path, timeout = 10.0):
        kind = seq.pop(0) if len(seq) > 1 else seq[0]
        calls.append(path)
        return (200 if kind == "ok" else (503 if kind == "http" else 0)), None, kind

    mod._get_json = fake
    return calls


def test_the_watch_keeps_probing_until_the_backend_answers(tmp_path, monkeypatch):
    """A stall straddling the end of sampling is not a dead backend. One probe cannot
    tell the difference, so the watch keeps asking until something comes back."""
    mod = _load(tmp_path, monkeypatch)
    calls = _scripted_probes(mod, ["timeout", "timeout", "ok"])
    kind, status, _ = mod.await_recovery(window_s = 30.0)
    assert (kind, status) == ("ok", 200)
    assert len(calls) == 3, calls


def test_the_watch_gives_up_and_reports_a_timeout(tmp_path, monkeypatch):
    """Bounded, so a genuinely dead backend still fails. It answers none of these
    probes either, which is why extending the observation cannot rescue a real death."""
    mod = _load(tmp_path, monkeypatch)
    calls = _scripted_probes(mod, ["timeout"])
    kind, _, _ = mod.await_recovery(window_s = 0.05)
    assert kind == "timeout"
    assert len(calls) >= 1, calls


def test_a_refused_port_does_not_get_the_recovery_window(tmp_path, monkeypatch):
    """Only a stall is worth waiting on. A refused port is already decided, so it fails
    at once rather than costing the run 90 seconds to reach the same answer."""
    mod = _load(tmp_path, monkeypatch)
    calls = _scripted_probes(mod, ["refused"])
    # Small but non-zero on purpose. A build that stopped returning early would spend it
    # and rack up probes, so this fails on the call count in a moment rather than hanging
    # the suite for as long as the real window lasts.
    kind, _, _ = mod.await_recovery(window_s = 2.0)
    assert kind == "refused"
    assert len(calls) == 1, f"spent the window instead of returning at once: {len(calls)} probes"


def test_an_answered_non_200_does_not_get_the_recovery_window(tmp_path, monkeypatch):
    """Also already decided: the backend answered, so waiting adds nothing."""
    mod = _load(tmp_path, monkeypatch)
    calls = _scripted_probes(mod, ["http"])
    kind, status, _ = mod.await_recovery(window_s = 2.0)
    assert (kind, status) == ("http", 503)
    assert len(calls) == 1, f"spent the window instead of returning at once: {len(calls)} probes"


def test_the_recovery_window_clears_the_stalls_actually_seen():
    """Sized from evidence, not from a round number. Run 32862298967 produced stalls of
    10.03s, 25.1s, 27.75s and 33.2s on one runner, so the window has to clear 33.2s with
    room to spare or it decides "never ended" about stalls this job has already seen."""
    longest_observed_s = 33.2
    assert _module_constant("RECOVERY_WINDOW_S") > 2 * longest_observed_s


def test_the_post_run_watch_is_wired_into_the_public_entry_point():
    """report() is driven with an injected verdict everywhere above, so nothing there
    would notice main() dropping the watch and going back to a single probe."""
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding = "utf-8"))
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {
        sub.func.id
        for sub in ast.walk(main)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
    }
    assert "await_recovery" in called
    reports = [
        sub for sub in ast.walk(main)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "report" and sub.keywords
    ]
    assert reports, "main() no longer hands a post-run verdict to report()"
    passed = {kw.arg for call in reports for kw in call.keywords}
    assert {"final_kind", "final_wait_s"} <= passed, passed
