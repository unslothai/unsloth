# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Nothing that matters may ever be suppressed, and the budget may not be met by deleting.

``test_log_budget.py`` caps how much gets written. On its own that is a dangerous test: an
upper bound is satisfied just as well by a middleware that was never mounted, a logger
replaced with a no-op, a scenario that stopped issuing requests, or a developer who deleted
an error log to get under the number. Every one of those is a worse outcome than the
regression the cap exists to catch.

So this file asserts the floor. Failures and mutations log every time at any interval, and
the replay is proved to have actually exercised the real middleware before any of it counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from loggers import handlers as hmod  # noqa: E402
from log_budget import policy, replay, session  # noqa: E402

# Every way a request can fail that a user or a support engineer would go looking for.
FAILURE_STATUSES = (400, 401, 403, 404, 409, 422, 429, 500, 502, 503)
MUTATING_METHODS = ("POST", "PUT", "PATCH", "DELETE")


def _drive(
    monkeypatch,
    requests,
    gap_s = 0.0,
):
    """Send requests through one middleware instance, advancing the clock by `gap_s`."""
    from loggers.handlers import LoggingMiddleware

    clock = replay.install(hmod, monkeypatch)
    capture = replay.LogCapture()
    monkeypatch.setattr(hmod, "logger", capture)
    middleware = LoggingMiddleware(replay._app_returning(200))

    for request in requests:
        middleware.app = replay._app_returning(request.status, request.duration_ms, clock)
        scope = {
            "type": "http",
            "path": request.path,
            "method": request.method,
            "query_string": request.query,
        }
        import asyncio

        asyncio.run(middleware(scope, replay._noop_receive, replay._noop_send))
        clock.advance(gap_s)
    return capture


class TestFailuresAreNeverSuppressed:
    """A failing poll must log every time, however fast it repeats."""

    @pytest.mark.parametrize("status", FAILURE_STATUSES)
    def test_repeated_failures_all_log_on_every_classified_path(self, status, monkeypatch):
        # Zero gap: the worst case for any window-based suppressor.
        #
        # The `excluded` class is left out because it is the one suppressor that is NOT
        # gated on a 2xx: `__call__` drops the path before the status is considered, so a
        # 500 on /api/train/status is invisible here. That is existing behaviour, not
        # something this guard can assert away; it is pinned instead by
        # test_the_excluded_set_is_exactly_what_was_reviewed below.
        paths = sorted(
            p
            for p in session.ALL_POLLS
            if policy.classify(hmod, p) != policy.EXCLUDED
            # Chat-list 401s have one narrow, deliberate exemption during the bootstrap
            # token race. Its exact boundaries are pinned by
            # test_the_chat_list_401_exemption_is_only_pre_auth rather than waved through.
            and not (status == 401 and p in hmod._CHAT_LIST_PATHS)
        )
        requests = [
            replay.Request(method = "GET", path = path, status = status)
            for path in paths
            for _ in range(3)
        ]
        capture = _drive(monkeypatch, requests, gap_s = 0.0)

        logged = [kw.get("path") for _lvl, _ev, kw in capture.events]
        missing = sorted({p for p in paths if logged.count(p) != 3})
        assert not missing, (
            f"a {status} response was de-duplicated on these paths, so a user hitting a "
            "repeated failure would see one line instead of every occurrence:\n  "
            + "\n  ".join(missing)
            + "\n\nSuppression must be gated on a 2xx status. See _is_redundant_repeat and "
            "_is_quiet_success in loggers/handlers.py."
        )

    @pytest.mark.parametrize("method", MUTATING_METHODS)
    def test_repeated_mutations_all_log(self, method, monkeypatch):
        requests = [
            replay.Request(method = method, path = "/api/chat/threads", status = 200) for _ in range(3)
        ]
        capture = _drive(monkeypatch, requests, gap_s = 0.0)
        assert len(capture.events) == 3, (
            f"three identical {method} requests produced {len(capture.events)} lines. "
            "Mutations change state and must never be collapsed, however fast they repeat."
        )

    def test_a_failure_inside_a_quiet_window_still_logs(self, monkeypatch):
        """The case that matters most: a poll that was quiet and starts failing."""
        quiet = [
            p
            for p in session.ALL_POLLS
            if policy.classify(hmod, p) in (policy.QUIET, policy.LIVENESS)
        ]
        assert quiet, "no quiet-poll paths configured; this guard would be vacuous"
        path = sorted(quiet)[0]

        capture = _drive(
            monkeypatch,
            [
                replay.Request("GET", path, 200),
                replay.Request("GET", path, 200),  # collapsed, correctly
                replay.Request("GET", path, 503),  # must not be
                replay.Request("GET", path, 503),
            ],
            gap_s = 0.0,
        )

        failures = [kw for _l, _e, kw in capture.events if kw.get("status_code") == 503]
        assert len(failures) == 2, (
            f"{path} went from healthy to failing inside its heartbeat window and only "
            f"{len(failures)} of 2 failures were logged. A watchdog going red is exactly "
            "what these logs are read for."
        )

    def test_the_chat_list_401_exemption_is_only_pre_auth(self, monkeypatch):
        """The one status-specific exemption, held to its stated scope.

        A chat list poll racing the first token refresh answers 401 for reasons that are
        not a problem, so it is suppressed. Once a refresh has succeeded a 401 means
        something real and must log. An exemption that quietly widened past the bootstrap
        window would hide genuine auth failures for the rest of the session.
        """
        from loggers.handlers import LoggingMiddleware
        import asyncio

        path = sorted(hmod._CHAT_LIST_PATHS)[0]
        replay.install(hmod, monkeypatch)
        capture = replay.LogCapture()
        monkeypatch.setattr(hmod, "logger", capture)
        middleware = LoggingMiddleware(replay._app_returning(401))

        def send(status):
            middleware.app = replay._app_returning(status)
            asyncio.run(
                middleware(
                    {"type": "http", "path": path, "method": "GET", "query_string": b""},
                    replay._noop_receive,
                    replay._noop_send,
                )
            )

        send(401)
        assert not capture.events, (
            "the bootstrap 401 on a chat list poll should be suppressed before the first "
            "successful refresh"
        )

        # A 500 is not covered by the exemption even during bootstrap.
        send(500)
        assert (
            len(capture.events) == 1
        ), "only 401 is exempt during bootstrap; a 500 on the same path must log"

        # After a refresh succeeds, a 401 is real.
        middleware._auth_refreshed = True
        send(401)
        assert len(capture.events) == 2, (
            "a 401 after the first successful token refresh is a real auth failure and "
            "must be logged; the bootstrap exemption has widened past its window"
        )

    def test_the_excluded_set_is_exactly_what_was_reviewed(self):
        """The one class where a failure genuinely does disappear.

        Every other suppressor checks the status first, so a 4xx or 5xx always logs. The
        ``excluded`` check in ``LoggingMiddleware.__call__`` runs before the status is
        known, so these paths log nothing at all, including a 500. That may be the right
        trade for a metrics endpoint polled twice a second, but it should never grow by
        accident: adding a path here means accepting that its failures are invisible in the
        access log.
        """
        reviewed = {
            "/api/system",
            "/api/train/hardware",
            "/api/train/metrics",
            "/api/train/status",
        }
        actual = set(hmod._EXCLUDED_PATHS)
        assert actual == reviewed, (
            "_EXCLUDED_PATHS changed. Unlike the heartbeat classes this one drops errors "
            "too, so a path added here will never report a failure in the access log.\n"
            f"  added:   {sorted(actual - reviewed)}\n"
            f"  removed: {sorted(reviewed - actual)}\n"
            "If the addition is intended, update this list and say why the path's failures "
            "do not need to be visible."
        )


class TestTheGuardIsNotVacuous:
    """Prove the replay exercised the real thing before trusting any count from it."""

    def test_the_replay_actually_reaches_the_middleware(self, monkeypatch):
        result = replay.replay(hmod, monkeypatch, session.IDLE_POLLS, 60.0, session.BOOT_REQUESTS)
        assert result.sent, "the scenario issued no requests at all"
        assert result.emitted > 0, (
            "the replay produced zero log lines. Every budget assertion would pass "
            "trivially. Either the middleware is not mounted or the capture is not the "
            "logger it calls."
        )

    def test_the_capture_is_the_logger_the_middleware_calls(self, monkeypatch):
        """A no-op logger would satisfy every ceiling in the budget file."""
        capture = _drive(monkeypatch, [replay.Request("GET", "/api/nope", 404)])
        assert hmod.logger is capture, (
            "the middleware is not logging through the captured object, so the budget "
            "tests are measuring nothing"
        )
        assert capture.events, "a 404 produced no record through the real middleware"

    def test_boot_emits_one_mutation_and_one_failure_sentinel(self, monkeypatch):
        """Lower bounds on the known-good scenario, so silent deletion fails here."""
        result = replay.replay(hmod, monkeypatch, {}, 0.0, session.BOOT_REQUESTS)
        records = [kw for _l, _e, kw in result.capture.events]

        mutations = [r for r in records if r.get("method") in MUTATING_METHODS]
        failures = [
            r
            for r in records
            if isinstance(r.get("status_code"), int) and not 200 <= r["status_code"] < 300
        ]
        successes = [
            r
            for r in records
            if isinstance(r.get("status_code"), int) and 200 <= r["status_code"] < 300
        ]

        assert len(mutations) == 1, (
            f"boot should log exactly one mutation, saw {len(mutations)}. "
            "If the login POST stopped being logged, mutation logging has regressed."
        )
        assert len(failures) == 1, (
            f"boot should log exactly one failure, saw {len(failures)}. "
            "The pre-auth 401 is the sentinel that proves failures survive."
        )
        assert successes, (
            "boot logged no successful request at all. Suppression has gone too far, or "
            "the scenario stopped issuing requests."
        )

    def test_every_scenario_path_is_actually_requested(self, monkeypatch):
        """A path that quietly stops being polled would lower every count for free."""
        result = replay.replay(hmod, monkeypatch, session.ALL_POLLS, 60.0)
        requested = {r.path for r in result.sent}
        missing = sorted(set(session.ALL_POLLS) - requested)
        assert not missing, (
            "the replay never issued these registered polls, so their budget is "
            "meaningless:\n  " + "\n  ".join(missing)
        )


class TestSlowSuccessIsNotYetSignal:
    """A 200 that took a minute is treated exactly like a 200 that took a millisecond.

    Both suppressors key on the STATUS CODE. Nothing anywhere reads how long the request
    took, so a degrading endpoint stays invisible for as long as it keeps returning 2xx.

    These tests assert the CURRENT behaviour rather than the desired one, on purpose. The
    gap is real and worth closing, but a guard that silently tolerates either answer would
    let the exemption be added and then removed again without anyone noticing. Closing it
    should flip these deliberately, with the new volume budgeted the same way as every
    other line here.
    """

    SLOW_MS = 30_000.0

    def _slow_lines(
        self,
        monkeypatch,
        path,
        count = 6,
    ):
        capture = _drive(
            monkeypatch,
            [replay.Request("GET", path, 200, duration_ms = self.SLOW_MS) for _ in range(count)],
            gap_s = 0.0,
        )
        return capture.records_for(path)

    def test_a_slow_success_on_a_silent_path_writes_nothing(self, monkeypatch):
        silent = sorted(
            p for p in session.ALL_POLLS if policy.classify(hmod, p) == policy.QUIET_SUCCESS
        )
        assert silent, "no quiet-success paths configured; this guard would be vacuous"
        path = silent[0]

        records = self._slow_lines(monkeypatch, path)
        assert records == [], (
            f"{path} now logs a slow success ({len(records)} line(s)). If that is the "
            "intended change, budget it: a sustained degradation on a 5s poll emits one "
            "line per request unless the slow line gets a heartbeat of its own."
        )

    def test_the_harness_can_tell_a_slow_request_from_a_fast_one(self, monkeypatch):
        """Guards the guard: without this, the two tests around it are vacuous.

        ``duration_ms`` has to actually reach the middleware's clock. If it silently did
        nothing, every 'slow' case above would really be a fast one and would pass for the
        wrong reason.
        """
        path = "/api/models/list"
        capture = _drive(
            monkeypatch,
            [replay.Request("GET", path, 200, duration_ms = self.SLOW_MS)],
            gap_s = 0.0,
        )
        records = capture.records_for(path)
        assert records, f"{path} is in the normal class and should log on the first hit"
        assert records[0]["process_time_ms"] >= self.SLOW_MS, (
            "duration_ms did not reach the middleware: it recorded "
            f"{records[0]['process_time_ms']}ms for a {self.SLOW_MS}ms request, so every "
            "slow-path assertion here is really testing a fast request."
        )
