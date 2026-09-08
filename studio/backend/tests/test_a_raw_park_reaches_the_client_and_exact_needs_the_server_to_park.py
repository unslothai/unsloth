# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Four gaps between what a park promises and what reached the client or the ledger.

1. The raw passthrough loops relayed ``data:`` lines only, so a parked request's ``: preempted``
   and two-second ``: preempt-keepalive`` reached nobody and a long park sent the client no bytes
   at all. ``_server_park_sse`` maps those notices onto the comments every surface forwards.

2. Exact concurrency was reported ``on`` under ``UNSLOTH_LLAMA_PREEMPT_MODE=studio``, though a
   chat Studio pauses resumes with fresh sampler and draft state where one the server parks is
   restored cell for cell. ``_exact_state_after_launch`` now takes ``server_parks``.

3. A user-named ``--preempt-ram`` below the KV pool was reported exact too, though a park that
   outgrows it is re-prefilled; ``_exact_parking_shortfall_mib`` names the shortfall.

4. A swap build predating the stream notices parks in silence, and the read wrapper forwarded
   nothing, so a durable run's lease had nothing to renew on. The backend stamps the excuse.
"""

import asyncio
import inspect
import time
from types import SimpleNamespace

import httpx
import pytest

import core.inference.chat_generation_runs as runs
import core.inference.llama_cpp as llama_mod
import routes.inference as inference
from core.inference import llama_exact as exact
from core.inference import llama_preemption as preemption_mod
from core.inference.llama_cpp import LlamaCppBackend


class TestARawStreamForwardsTheServersPark:
    def test_the_three_notices_map_onto_the_studio_comments(self):
        assert inference._server_park_sse(llama_mod._SERVER_PARKED_COMMENT) == (
            inference._OPENAI_PREEMPT_SSE_PAUSED
        )
        assert inference._server_park_sse(llama_mod._SERVER_RESUMED_COMMENT) == (
            inference._OPENAI_PREEMPT_SSE_RESUMED
        )
        assert inference._server_park_sse(llama_mod._SERVER_KEEPALIVE_COMMENT) == (
            inference._OPENAI_PREEMPT_SSE_KEEPALIVE
        )
        # The routes spell the notices out; the backend's constants are the source of truth.
        assert set(inference._SERVER_PARK_SSE_BY_COMMENT) == set(llama_mod._SERVER_PARK_COMMENTS)

    def test_everything_else_is_left_to_the_loop(self):
        assert inference._server_park_sse('data: {"choices": []}') is None
        assert inference._server_park_sse("") is None
        assert inference._server_park_sse(None) is None
        assert inference._server_park_sse(": keep-alive") is None

    def test_bytes_and_trailing_whitespace_are_read_the_same(self):
        assert inference._server_park_sse(b": preempt-keepalive") == (
            inference._OPENAI_PREEMPT_SSE_KEEPALIVE
        )
        assert inference._server_park_sse(": preempted\r") == inference._OPENAI_PREEMPT_SSE_PAUSED

    def test_every_raw_loop_asks(self):
        """One relay per raw loop: Responses, Anthropic passthrough, chat passthrough and the
        completions byte loop. The count is the guard against a loop that forgets."""
        source = inspect.getsource(inference)
        calls = source.count("_park = _server_park_sse(")
        assert calls >= 4, f"{calls} raw loops relay a park; four did when this was written"
        # Each relay yields the comment and moves on, before the `data:` filter.
        assert source.count("if _park is not None:\n") >= 4


class TestExactNeedsTheServerToPark:
    _ARGV = ["llama-server", "--kv-unified", "--flash-attn", "on"]

    def _state(self, **kw):
        kw.setdefault("supports_exact", True)
        kw.setdefault("setting", "auto")
        kw.setdefault("env", {exact.CHILD_ENV: "1"})
        kw.setdefault("args", self._ARGV)
        return LlamaCppBackend._exact_state_after_launch(**kw)

    def test_on_when_the_server_parks_and_the_budget_holds(self):
        assert self._state() == exact.EXACT_STATE_ON
        assert self._state(server_parks = True, parking_holds = True) == exact.EXACT_STATE_ON

    @pytest.mark.parametrize("setting", ["auto", "on"])
    def test_studio_side_pausing_is_not_exact(self, setting):
        assert self._state(setting = setting, server_parks = False) == (exact.EXACT_STATE_UNAVAILABLE)

    @pytest.mark.parametrize("setting", ["auto", "on"])
    def test_a_budget_below_the_pool_is_not_exact(self, setting):
        assert self._state(setting = setting, parking_holds = False) == (exact.EXACT_STATE_UNAVAILABLE)

    def test_off_stays_off_whatever_the_server_does(self):
        assert self._state(setting = "off", server_parks = False) == exact.EXACT_STATE_OFF

    def test_the_launch_hands_both_answers_over(self):
        source = inspect.getsource(LlamaCppBackend.load_model)
        assert "server_parks = self.server_preempts_kv" in source
        assert "parking_holds = _exact_short is None" in source


_GIB = 1024 * 1024 * 1024


class TestTheNamedBudgetIsJudged:
    def test_a_budget_below_the_pool_is_a_shortfall(self):
        short = llama_mod._exact_parking_shortfall_mib(
            12 * _GIB, args = ["--preempt-ram", "1024"], env = {}
        )
        assert short == (1024, 12 * 1024, 12 * 1024 + 64)

    def test_a_budget_that_holds_the_pool_is_fine(self):
        assert (
            llama_mod._exact_parking_shortfall_mib(
                12 * _GIB, args = ["--preempt-ram", str(12 * 1024 + 64)], env = {}
            )
            is None
        )

    def test_unlimited_and_off_are_not_shortfalls(self):
        # -1 is no limit; 0 is parking off, which server_preempts_kv already reports.
        for value in ("-1", "0"):
            assert (
                llama_mod._exact_parking_shortfall_mib(
                    12 * _GIB, args = ["--preempt-ram", value], env = {}
                )
                is None
            )

    def test_nothing_named_is_nothing_to_judge(self):
        assert llama_mod._exact_parking_shortfall_mib(12 * _GIB, args = [], env = {}) is None
        assert (
            llama_mod._exact_parking_shortfall_mib(0, args = ["--preempt-ram", "1"], env = {}) is None
        )

    def test_the_servers_default_is_judged_for_a_pool_sized_after_launch(self):
        # An auto-fit context leaves the pool unknown at launch; after it the default budget
        # the child ran with is judged against the context the server chose.
        short = llama_mod._exact_parking_shortfall_mib(
            12 * _GIB, args = [], env = {}, default_mib = llama_mod._PREEMPT_RAM_DEFAULT_MIB
        )
        assert short == (8192, 12 * 1024, 12 * 1024 + 64)
        assert (
            llama_mod._exact_parking_shortfall_mib(
                4 * _GIB, args = [], env = {}, default_mib = llama_mod._PREEMPT_RAM_DEFAULT_MIB
            )
            is None
        )
        # A named budget still wins over the default.
        assert (
            llama_mod._exact_parking_shortfall_mib(
                12 * _GIB, args = ["--preempt-ram", "-1"], env = {}, default_mib = 8192
            )
            is None
        )

    def test_an_unknown_pool_is_sized_off_the_servers_context_after_launch(self):
        source = inspect.getsource(LlamaCppBackend.load_model)
        assert "self._exact_pool_unknown = _exact_kv_bytes <= 0" in source
        judged = source.index('getattr(self, "_exact_pool_unknown", False)')
        window = source[judged : judged + 1600]
        assert "self._query_server_n_ctx()" in window
        assert "default_mib = _PREEMPT_RAM_DEFAULT_MIB" in window
        assert (
            "_exact_short = (_PREEMPT_RAM_DEFAULT_MIB, 0, 0)" in window
        ), "a pool that cannot be sized must not be certified"
        assert window.index("self._exact_parking_short = _exact_short") < window.index(
            "self._exact_state_after_launch("
        )

    def test_the_environment_and_a_later_flag_are_read_in_llama_cpps_order(self):
        # The variable first, argv last-wins over it, as the child applies them.
        assert llama_mod._named_preempt_ram_mib([], {"LLAMA_ARG_PREEMPT_RAM": "512"}) == 512
        assert (
            llama_mod._named_preempt_ram_mib(
                ["--preempt-ram", "512", "--preempt-ram=2048"], {"LLAMA_ARG_PREEMPT_RAM": "1"}
            )
            == 2048
        )
        assert llama_mod._named_preempt_ram_mib(["--preempt-ram", "lots"], {}) is None
        short = llama_mod._exact_parking_shortfall_mib(
            12 * _GIB, args = [], env = {"LLAMA_ARG_PREEMPT_RAM": "2048"}
        )
        assert short is not None and short[0] == 2048


class TestASilentParkStillRenewsTheLease:
    def _backend(self, monkeypatch, preempted):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._port = 65010
        backend._api_key = None
        monkeypatch.setattr(
            "core.inference.llama_stats.scrape_llama_metrics",
            lambda base_url, timeout_s = 3.0, headers = None: {"requests_preempted": preempted},
        )
        return backend

    def test_an_excused_silence_is_stamped(self, monkeypatch):
        backend = self._backend(monkeypatch, 1)
        assert backend.server_park_grace_recent(1.0) is False
        assert backend._server_park_grace() is True
        assert backend.server_park_grace_recent(1.0) is True

    def test_a_pool_with_nothing_parked_stamps_nothing(self, monkeypatch):
        backend = self._backend(monkeypatch, 0)
        assert backend._server_park_grace() is False
        assert backend.server_park_grace_recent(60.0) is False

    def test_the_stamp_ages_out(self, monkeypatch):
        backend = self._backend(monkeypatch, 1)
        assert backend._server_park_grace() is True
        backend._server_park_grace_at = time.monotonic() - 100.0
        assert backend.server_park_grace_recent(10.0) is False

    def test_the_run_loop_consults_the_stamp(self, monkeypatch):
        class _Backend:
            def server_park_grace_recent(self, within_s):
                return within_s > 0

        monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: _Backend())
        assert runs._server_park_excused_recently() is True
        monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: object())
        assert runs._server_park_excused_recently() is False

    def test_the_silent_branch_renews_and_is_bounded(self):
        """The wait is bounded even with nothing to flush, and the silent branch renews
        from the stamp; the source is the only place this loop can be read."""
        source = inspect.getsource(runs.ChatGenerationSupervisor)
        assert "else _renew_interval_seconds()" in source
        assert "elif await asyncio.to_thread(_server_park_excused_recently):" in source
        branch = source.index("elif await asyncio.to_thread(_server_park_excused_recently):")
        assert "await self._try_touch_progress(run_id)" in source[branch : branch + 400]


class TestTheRunLoopProbesParkingRatherThanWaitForTheStamp:
    """The read wrapper only asks `/metrics` at its read deadline, and before the first token that
    deadline IS the 20 minute first-token budget, so a run parked during prefill had no stamp to
    renew from until the sweeper had had its chance to cancel it. The run loop asks for itself."""

    @pytest.fixture(autouse = True)
    def _reset_probe_rate_limit(self):
        runs._park_probe_at[0] = None
        yield
        runs._park_probe_at[0] = None

    class _Backend:
        """A swap build that parks in silence and has never been asked, so it has no stamp."""

        server_preempts_kv = True

        def __init__(self):
            self.asked = 0

        def server_park_grace_recent(self, within_s):
            return False

        def _server_park_grace(self):
            self.asked += 1
            return True

    def test_a_park_with_no_stamp_yet_still_renews(self, monkeypatch):
        backend = self._Backend()
        monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backend)
        assert runs._server_park_excused_recently() is True
        assert backend.asked == 1

    def test_a_server_that_does_not_park_is_never_scraped(self, monkeypatch):
        backend = self._Backend()
        backend.server_preempts_kv = False
        monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backend)
        assert runs._server_park_excused_recently() is False
        assert backend.asked == 0

    def test_nothing_parked_is_not_excused(self, monkeypatch):
        backend = self._Backend()
        backend._server_park_grace = lambda: False
        monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backend)
        assert runs._server_park_excused_recently() is False

    def test_the_stamp_still_wins_and_costs_no_scrape(self, monkeypatch):
        backend = self._Backend()
        backend.server_park_grace_recent = lambda within_s: True
        monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backend)
        assert runs._server_park_excused_recently() is True
        assert backend.asked == 0

    def test_the_live_probe_is_rate_limited_across_runs(self, monkeypatch):
        backend = self._Backend()
        monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backend)
        assert runs._server_park_excused_recently() is True
        # A short lease drives the renewal cadence to 0.25s; the floor keeps that off /metrics.
        assert runs._server_park_excused_recently() is False
        assert backend.asked == 1
        runs._park_probe_at[0] = time.monotonic() - runs._PARK_PROBE_MIN_INTERVAL_S - 0.1
        assert runs._server_park_excused_recently() is True
        assert backend.asked == 2

    def test_the_probe_stays_off_the_event_loop(self):
        """It blocks on one HTTP GET, so the silent branch must keep reaching it via to_thread."""
        source = inspect.getsource(runs.ChatGenerationSupervisor)
        assert "elif await asyncio.to_thread(_server_park_excused_recently):" in source


class TestAutoDoesNotStartAModeItWillReportUnavailable:
    _ARGV = ["llama-server", "--kv-unified"]

    def test_studio_side_pausing_blocks_an_auto_launch(self, monkeypatch):
        monkeypatch.setenv(preemption_mod.PREEMPT_MODE_ENV, "studio")
        why = llama_mod._exact_auto_blocker(exact.EXACT_AUTO, self._ARGV, {})
        assert why and "UNSLOTH_LLAMA_PREEMPT_MODE=studio" in why

    def test_parking_switched_off_blocks_an_auto_launch(self, monkeypatch):
        monkeypatch.delenv(preemption_mod.PREEMPT_MODE_ENV, raising = False)
        assert llama_mod._exact_auto_blocker(
            exact.EXACT_AUTO, self._ARGV + ["--preempt-ram", "0"], {}
        )
        assert llama_mod._exact_auto_blocker(
            exact.EXACT_AUTO, self._ARGV, {"LLAMA_ARG_PREEMPT_RAM": "0"}
        )
        # Studio's own budget after an inherited zero wins, as the child applies argv last.
        assert (
            llama_mod._exact_auto_blocker(
                exact.EXACT_AUTO,
                self._ARGV + ["--preempt-ram", "4096"],
                {"LLAMA_ARG_PREEMPT_RAM": "0"},
            )
            is None
        )

    def test_a_clean_auto_launch_and_every_on_launch_go_ahead(self, monkeypatch):
        monkeypatch.delenv(preemption_mod.PREEMPT_MODE_ENV, raising = False)
        assert llama_mod._exact_auto_blocker(exact.EXACT_AUTO, self._ARGV, {}) is None
        monkeypatch.setenv(preemption_mod.PREEMPT_MODE_ENV, "studio")
        # `on` is the post-launch check's to fail, naming the reason.
        assert llama_mod._exact_auto_blocker(exact.EXACT_ON, self._ARGV, {}) is None

    def test_the_launch_consults_it_before_the_child_env_is_written(self):
        source = inspect.getsource(LlamaCppBackend.load_model)
        blocker = source.index("_exact_blocker = _exact_auto_blocker(")
        short_drop = source.index("if _exact_setting == _exact.EXACT_AUTO:\n")
        child_env = source.index("_exact.apply_child_env(env, on = _exact_wanted)")
        assert blocker < child_env and short_drop < child_env
        # And an unknown pool is never handed a budget on speculation: an unlimited one
        # generated before the server has confirmed the mode survives every fallback.
        assert '"--preempt-ram", "-1"' not in source


class _Request:
    async def is_disconnected(self):
        return False


class TestARawRelayWaitsThroughAServerPark:
    @staticmethod
    async def _stream(timeouts_before_second: int):
        yield "data: a"
        for _ in range(timeouts_before_second):
            await asyncio.sleep(0.03)
            raise httpx.ReadTimeout("read timed out")
        yield "data: b"

    @staticmethod
    async def _resumable(timeouts: int):
        """An iterator whose ReadTimeouts do not end it, like a live httpx line stream."""
        state = {"left": timeouts, "sent_a": False}

        class _It:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if not state["sent_a"]:
                    state["sent_a"] = True
                    return "data: a"
                if state["left"] > 0:
                    state["left"] -= 1
                    await asyncio.sleep(0.03)
                    raise httpx.ReadTimeout("read timed out")
                if state.get("done"):
                    raise StopAsyncIteration
                state["done"] = True
                return "data: b"

        return _It()

    def test_a_parked_relay_keeps_waiting(self):
        async def run():
            it = await self._resumable(3)
            seen = []
            async for item in inference._aiter_llama_stream_items(
                it,
                request = _Request(),
                post_first_item_read_timeout_s = 0.01,
                stall_grace = lambda: True,
            ):
                seen.append(item)
            return seen

        assert asyncio.run(run()) == ["data: a", "data: b"]

    def test_without_a_park_the_stall_still_ends_the_relay(self):
        async def run():
            it = await self._resumable(3)
            seen = []
            async for item in inference._aiter_llama_stream_items(
                it,
                request = _Request(),
                post_first_item_read_timeout_s = 0.01,
                stall_grace = lambda: False,
            ):
                seen.append(item)
            return seen

        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(run())

    def test_the_probe_is_the_backends_and_only_when_the_server_parks(self):
        class _Parks:
            server_preempts_kv = True

            def _server_park_grace(self):
                return True

        class _DoesNot:
            server_preempts_kv = False

            def _server_park_grace(self):
                return True

        assert inference._raw_park_grace(_Parks())() is True
        assert inference._raw_park_grace(_DoesNot()) is None
        assert inference._raw_park_grace(object()) is None

    def test_every_raw_relay_hands_it_over(self):
        source = inspect.getsource(inference)
        assert source.count("stall_grace = _raw_park_grace(llama_backend),") == 4
        assert inference._RAW_PARK_STALL_CAP_S == llama_mod._SERVER_PARK_STALL_CAP_S


class TestARawStreamIsParkableWhenTheServerParks:
    """`pausable=False` is about Studio's preemptor. With the server parking slots itself a raw
    relay is parked and restored like any other, so it is priced like any other."""

    class _Backend:
        _kv_cache_unified = True
        context_length = 16384
        effective_parallel_slots = 4
        server_preempts_kv = True

    class _StudioOnly(_Backend):
        server_preempts_kv = False

    def test_the_predicate_reads_the_backend(self):
        assert inference._server_parks_raw_streams(self._Backend()) is True
        assert inference._server_parks_raw_streams(self._StudioOnly()) is False
        assert inference._server_parks_raw_streams(object()) is False

    def test_both_entry_points_lift_the_share_for_a_parking_server(self):
        source = inspect.getsource(inference._openai_llama_admission_enforced_max_tokens)
        assert "pausable = pausable or _server_parks_raw_streams(llama_backend)" in source
        source = inspect.getsource(inference._openai_llama_admission_reserve)
        assert "pausable = pausable or _server_parks_raw_streams(llama_backend)" in source
        assert source.index("pausable = pausable or") < source.index("preemption_active = pausable")

    def test_the_wire_cap_is_the_window_not_a_share(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "1")
        monkeypatch.setattr(inference, "_openai_llama_admission_budget", lambda b: 16384)
        monkeypatch.setattr(inference, "_openai_llama_admission_context_window", lambda b: 16384)
        monkeypatch.setattr(inference, "_openai_llama_admission_capacity", lambda r, b: 4)
        monkeypatch.setattr(inference, "_openai_llama_admission_raw_total", lambda b: 16384)
        monkeypatch.setattr(inference, "_openai_llama_admission_image_tokens", lambda b: 0)
        monkeypatch.setattr(
            inference, "_openai_llama_admission_prompt_tokens", lambda *a, **k: 1000
        )
        monkeypatch.setattr(
            inference, "_openai_llama_preemption_will_apply", lambda b, budget: True
        )
        payload = {"messages": [{"role": "user", "content": "x"}]}
        parked = inference._openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = self._Backend(), pausable = False
        )
        studio_only = inference._openai_llama_admission_enforced_max_tokens(
            payload, request = None, llama_backend = self._StudioOnly(), pausable = False
        )
        assert studio_only == 16384 // 4 - 1000, "Studio-only: the honest share"
        assert parked == 16384 - 1000, "a parking server: the window, like every other stream"


class TestAParkDuringPrefillIsExcusedToo:
    def test_the_first_item_deadline_takes_the_grace(self, monkeypatch):
        # Each read times out the way httpx's own read timeout does, after the deadline has
        # passed, and the excuse extends the deadline by a short first-token window.
        monkeypatch.setattr(inference, "_DEFAULT_FIRST_TOKEN_TIMEOUT_S", 0.002)

        class _It:
            def __init__(self):
                self.left = 3

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.left > 0:
                    self.left -= 1
                    await asyncio.sleep(0.005)
                    raise httpx.ReadTimeout("read timed out")
                if getattr(self, "done", False):
                    raise StopAsyncIteration
                self.done = True
                return "data: first"

        async def run(grace):
            seen = []
            async for item in inference._aiter_llama_stream_items(
                _It(),
                request = _Request(),
                first_token_deadline = time.monotonic() + 0.001,
                stall_grace = grace,
            ):
                seen.append(item)
            return seen

        assert asyncio.run(run(lambda: True)) == ["data: first"]
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(run(None))

    def test_the_first_item_grace_is_bounded(self):
        source = inspect.getsource(inference._aiter_llama_stream_items)
        excuse = source.index("def _first_item_excused(")
        window = source[excuse : excuse + 700]
        assert "first_deadline_crossed_at" in window and "_RAW_PARK_STALL_CAP_S" in window
        # Both ways the first read can time out ask it.
        assert source.count("_first_item_excused(") == 3


class TestTheGlobalOptOutBlocksAnAutoLaunch:
    def test_preemption_off_with_nothing_named_is_parking_off(self, monkeypatch):
        monkeypatch.delenv(preemption_mod.PREEMPT_MODE_ENV, raising = False)
        monkeypatch.setattr(llama_mod._preemption, "preemption_enabled", lambda: False)
        why = llama_mod._exact_auto_blocker(exact.EXACT_AUTO, ["llama-server"], {})
        assert why and "UNSLOTH_LLAMA_ADMISSION_PREEMPT=0" in why
        # A named budget keeps its say, as `_stand_down_child_parking` leaves it alone.
        assert (
            llama_mod._exact_auto_blocker(
                exact.EXACT_AUTO, ["llama-server", "--preempt-ram", "4096"], {}
            )
            is None
        )
        assert (
            llama_mod._exact_auto_blocker(
                exact.EXACT_AUTO, ["llama-server"], {"LLAMA_ARG_PREEMPT_RAM": "4096"}
            )
            is None
        )
        monkeypatch.setattr(llama_mod._preemption, "preemption_enabled", lambda: True)
        assert llama_mod._exact_auto_blocker(exact.EXACT_AUTO, ["llama-server"], {}) is None


class TestTheParkGraceLivesBelowTheHttpxIterators:
    """An httpx async generator that raised is closed, so a retry above it returned
    StopAsyncIteration and the relay ended as if the parked answer were complete. The grace is
    applied to the network stream's read; this runs the relay against a server that goes silent."""

    @staticmethod
    async def _serve(first_delay: float, gap: float):
        async def handle(reader, writer):
            try:
                await reader.readuntil(b"\r\n\r\n")
            except Exception:
                return
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
            )
            await writer.drain()
            for delay, chunk in ((first_delay, b"data: a\n\n"), (gap, b"data: b\n\n")):
                await asyncio.sleep(delay)
                writer.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
            writer.close()

        return await asyncio.start_server(handle, "127.0.0.1", 0)

    async def _relay(
        self,
        first_delay,
        gap,
        *,
        grace,
        stall_s = 0.05,
        first_s = 0.05,
    ):
        server = await self._serve(first_delay, gap)
        port = server.sockets[0].getsockname()[1]
        seen = []
        try:
            async with httpx.AsyncClient(timeout = httpx.Timeout(5.0, read = 0.05)) as client:
                async with client.stream("GET", f"http://127.0.0.1:{port}/") as resp:
                    async for line in inference._aiter_llama_stream_items(
                        resp.aiter_lines(),
                        request = _Request(),
                        response = resp,
                        first_token_deadline = time.monotonic() + first_s,
                        post_first_item_read_timeout_s = stall_s,
                        stall_grace = grace,
                    ):
                        if line:
                            seen.append(line)
        finally:
            server.close()
            await server.wait_closed()
        return seen

    def test_a_park_after_the_first_item_is_waited_out(self):
        asked = []

        def grace():
            asked.append(time.monotonic())
            return True

        assert asyncio.run(self._relay(0.0, 0.4, grace = grace)) == ["data: a", "data: b"]
        assert asked, "the probe was never consulted"

    def test_a_park_during_prefill_is_waited_out(self):
        assert asyncio.run(self._relay(0.4, 0.0, grace = lambda: True)) == ["data: a", "data: b"]

    def test_without_a_park_the_stall_is_still_an_error_not_a_short_answer(self):
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(self._relay(0.0, 0.4, grace = lambda: False))

    def test_a_closed_iterator_is_never_retried(self, monkeypatch):
        # Grace above the iterator only: the raised generator is done, and the relay reports
        # the stall rather than returning the one line it had as the whole answer.
        monkeypatch.setattr(inference, "_install_park_aware_read", lambda *a, **k: False)
        with pytest.raises(httpx.ReadTimeout):
            asyncio.run(self._relay(0.0, 0.4, grace = lambda: True))

    def test_the_wrapper_follows_the_request_a_kept_alive_connection_serves_next(self):
        class _Stream:
            async def read(
                self,
                max_bytes,
                timeout = None,
            ):
                return b""

        stream = _Stream()
        first = SimpleNamespace(extensions = {"network_stream": stream})
        second = SimpleNamespace(extensions = {"network_stream": stream})
        one, two = (lambda: True), (lambda: False)
        assert inference._install_park_aware_read(first, one) is True
        assert inference._install_park_aware_read(second, two) is True
        assert stream._unsloth_park_state["stall_grace"] is two
        assert inference._install_park_aware_read(SimpleNamespace(extensions = {}), one) is False

    def test_every_raw_relay_reads_through_the_response(self):
        source = inspect.getsource(inference)
        assert source.count("stall_grace = _raw_park_grace(llama_backend),") == 4
        assert (
            source.count("                    response = resp,\n")
            + source.count("                response = resp,\n")
            >= 4
        )


class TestTheGraceStartsAtTheDeadlineAndTheProbeLeavesTheLoopAlone:
    @staticmethod
    def _wrapped(
        monkeypatch,
        read,
        grace,
        *,
        read_timeout = 0.03,
    ):
        stream = SimpleNamespace(read = read)
        response = SimpleNamespace(
            extensions = {"network_stream": stream},
            request = SimpleNamespace(extensions = {"timeout": {"read": read_timeout}}),
        )
        assert inference._install_park_aware_read(response, grace) is True
        return stream

    def test_the_cap_is_measured_from_the_deadline_it_first_crossed(self, monkeypatch):
        import httpcore

        # A 30ms window and a 50ms grace: the retries after the first deadline add up to the
        # grace, and not to the grace less the window it took to reach the deadline. The clock
        # is faked and advanced by exactly each window: real sleeps overshoot on a loaded
        # runner and read as a short grace.
        monkeypatch.setattr(inference, "_RAW_PARK_STALL_CAP_S", 0.05)
        windows = []
        clock = [1000.0]

        class _Clock:
            monotonic = staticmethod(lambda: clock[0])

            def __getattr__(self, name):
                return getattr(time, name)

        monkeypatch.setattr(inference, "time", _Clock())

        async def silent(max_bytes, timeout = None):
            windows.append(timeout)
            clock[0] += timeout
            await asyncio.sleep(0)
            raise httpcore.ReadTimeout("silence")

        stream = self._wrapped(monkeypatch, silent, lambda: True)
        with pytest.raises(httpcore.ReadTimeout):
            asyncio.run(stream.read(65536, timeout = 1200.0))
        assert windows[0] == pytest.approx(0.03)
        assert sum(windows[1:]) == pytest.approx(0.05, abs = 1e-9), windows
        assert len(windows) >= 3

    def test_the_probe_runs_off_the_event_loop(self, monkeypatch):
        import httpcore

        calls = {"n": 0}

        async def read(max_bytes, timeout = None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpcore.ReadTimeout("silence")
            return b"data: a\n\n"

        def slow_probe():
            time.sleep(0.2)  # `/metrics` over urllib, blocking
            return True

        stream = self._wrapped(monkeypatch, read, slow_probe)

        async def run():
            ticks = []

            async def ticker():
                while True:
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.005)

            task = asyncio.create_task(ticker())
            try:
                got = await stream.read(65536, timeout = 1200.0)
            finally:
                task.cancel()
            return got, len(ticks)

        got, ticks = asyncio.run(run())
        assert got == b"data: a\n\n"
        assert ticks > 10, f"the loop was held while the probe ran ({ticks} ticks)"

    def test_the_grace_above_the_iterator_asks_off_the_loop_too(self):
        source = inspect.getsource(inference._aiter_llama_stream_items)
        assert "grace_above()" not in source
        assert source.count("await _probe_off_the_loop(grace_above)") == 2


class TestAnExplicitOptOutOfTheUnifiedCacheIsKept:
    _CAPS = {"supports_kv_unified": True}

    def test_the_launch_line_does_not_reverse_it(self):
        add = LlamaCppBackend._exact_missing_launch_flags
        assert add(["llama-server", "--parallel", "1"], self._CAPS) == ["--kv-unified"]
        assert add(["llama-server", "--parallel", "1", "--no-kv-unified"], self._CAPS) == []
        assert add(["llama-server", "-no-kvu"], self._CAPS) == []
        assert add(["llama-server", "--no-kv-unified", "--kv-unified"], self._CAPS) == []

    def test_it_is_the_contradiction_it_is(self):
        assert exact.contradicting_args(["--no-kv-unified"]) == ["--no-kv-unified"]
        assert exact.contradicting_args(["-no-kvu"]) == ["-no-kvu"]
        # A later spelling of the same option decides for it, as llama-server applies argv.
        assert exact.contradicting_args(["--no-kv-unified", "--kv-unified"]) == []
        assert exact.contradicting_args(["--kv-unified", "-no-kvu"]) == ["-no-kvu"]

    def test_auto_does_not_start_a_mode_the_extras_contradict(self, monkeypatch):
        monkeypatch.delenv(preemption_mod.PREEMPT_MODE_ENV, raising = False)
        monkeypatch.delenv(preemption_mod.PREEMPT_ENV, raising = False)
        reason = llama_mod._exact_auto_blocker(
            exact.EXACT_AUTO, ["llama-server", "--kv-unified", "--no-kv-unified"], {}
        )
        assert reason is not None and "--no-kv-unified" in reason
        assert (
            llama_mod._exact_auto_blocker(exact.EXACT_AUTO, ["llama-server", "--kv-unified"], {})
            is None
        )


class TestTheGlobalOptOutBlocksAnExactOnLaunchToo:
    """`auto` was preflighted for the opt-out, `on` was not: the exact launch sized a parking budget
    of its own, and `_stand_down_child_parking` reads any `--preempt-ram` as one somebody named, so
    the child parked with UNSLOTH_LLAMA_ADMISSION_PREEMPT=0 set. The budget is generated only when
    the child is going to be allowed to park at all."""

    _POOL = 12 * _GIB

    @staticmethod
    def _opt_out(monkeypatch):
        monkeypatch.delenv(preemption_mod.PREEMPT_MODE_ENV, raising = False)
        monkeypatch.setattr(llama_mod._preemption, "preemption_enabled", lambda: False)

    def test_the_launch_generates_no_budget_under_the_opt_out(self, monkeypatch):
        self._opt_out(monkeypatch)
        argv = ["llama-server", "--kv-unified"]
        # What the launch would have appended, for a sized pool and for an auto-fit one.
        assert llama_mod._exact_parking_budget_mib(self._POOL, args = argv, env = {}) is not None
        assert llama_mod._named_preempt_ram_mib(argv, {}) is None
        # ... and the guard that now stands in front of both.
        assert llama_mod._child_parking_stands_down(argv, {}) is True
        # So nothing names a budget, and the child is handed parking off.
        env: dict = {}
        assert llama_mod._stand_down_child_parking(env, argv) is True
        assert env["LLAMA_ARG_PREEMPT_RAM"] == "0"

    def test_a_budget_somebody_named_still_keeps_its_say(self, monkeypatch):
        self._opt_out(monkeypatch)
        for argv, env in (
            (["llama-server", "--preempt-ram", "4096"], {}),
            (["llama-server", "--preempt-ram=4096"], {}),
            (["llama-server"], {"LLAMA_ARG_PREEMPT_RAM": "4096"}),
        ):
            assert llama_mod._child_parking_stands_down(argv, env) is False

    def test_studio_side_pausing_stands_the_child_down_the_same_way(self, monkeypatch):
        monkeypatch.setattr(llama_mod._preemption, "preemption_enabled", lambda: True)
        monkeypatch.setenv(preemption_mod.PREEMPT_MODE_ENV, "studio")
        assert llama_mod._child_parking_stands_down(["llama-server"], {}) is True

    def test_a_launch_with_preemption_on_still_gets_its_budget(self, monkeypatch):
        monkeypatch.delenv(preemption_mod.PREEMPT_MODE_ENV, raising = False)
        monkeypatch.setattr(llama_mod._preemption, "preemption_enabled", lambda: True)
        argv = ["llama-server", "--kv-unified"]
        assert llama_mod._child_parking_stands_down(argv, {}) is False
        assert llama_mod._exact_parking_budget_mib(self._POOL, args = argv, env = {}) is not None
        env: dict = {}
        assert llama_mod._stand_down_child_parking(env, argv) is False
        assert env == {}

    def test_the_stand_down_and_the_launch_read_one_predicate(self):
        stand_down = inspect.getsource(llama_mod._stand_down_child_parking)
        assert "_child_parking_stands_down(args, env)" in stand_down
        source = inspect.getsource(LlamaCppBackend.load_model)
        guard = source.index('server_caps.get("supports_preempt_ram")')
        window = source[guard : source.index("self._exact_pool_unknown = _exact_kv_bytes <= 0")]
        assert "not _child_parking_stands_down(" in window
        assert window.index("not _child_parking_stands_down(") < window.index("_exact_budget = ")


class TestAnAbandonedExactAttemptLeavesNoUnlimitedParkingBudget:
    """The exact launch used to append ``--preempt-ram -1`` for an auto-fit context, before the
    running server had confirmed the mode, and every way the attempt is abandoned keeps that argv
    (the refusal rung relaunches the same command; a build ignoring the variable never relaunches).
    A server with no exact concurrency then parked into unbounded host RAM instead of llama.cpp's
    8192 MiB default. So the launch names no budget it cannot size, judging the default later."""

    def test_the_launch_never_generates_an_unlimited_budget(self):
        source = inspect.getsource(LlamaCppBackend.load_model)
        assert '"--preempt-ram", "-1"' not in source
        assert "--preempt-ram=-1" not in source
        # The only budget the launch may name is the one it sized for a pool it could measure.
        assert 'cmd.extend(["--preempt-ram", str(_exact_budget)])' in source

    def test_an_unknown_pool_stays_unknown_so_the_default_is_judged_after_launch(self):
        source = inspect.getsource(LlamaCppBackend.load_model)
        marked = source.index("self._exact_pool_unknown = _exact_kv_bytes <= 0")
        # Nothing clears the flag between marking it and the post-launch judging that reads it.
        judged = source.index('getattr(self, "_exact_pool_unknown", False)')
        assert marked < judged
        assert "self._exact_pool_unknown = False" not in source[marked:judged]

    def test_the_refusal_rung_relaunches_the_same_argv_with_only_the_env_changed(self):
        # Why a generated flag had to go: the fallback rebuilds nothing.
        drop = inspect.getsource(LlamaCppBackend._drop_exact_after_refusal)
        assert "apply_child_env(env, on = False)" in drop
        spawn = inspect.getsource(LlamaCppBackend.load_model)
        rung = spawn.index("if not _did_exact_retry and self._drop_exact_after_refusal(")
        window = spawn[rung : rung + 900]
        assert "continue" in window
        assert "run_cmd" not in window.split("continue")[0]

    def test_a_short_budget_is_reported_as_the_mode_running_short_not_as_absent(self):
        # The child keeps the mode after a late shortfall, so the warning, and the `on`
        # refusal, say it runs but cannot hold every park, not that it came up without it.
        spawn = inspect.getsource(LlamaCppBackend.load_model)
        judged = spawn.index("_exact_running = self._server_reports_exact_concurrency()")
        window = spawn[judged : judged + 6000]
        assert "if _exact_running:" in window
        assert "runs the mode, but it cannot hold every " in window
        assert "park, and a chat re-prefilled after a park it could not hold" in window
        assert "came up without it" in window.split("if _exact_running:")[1]
        assert window.count("_exact_what") >= 4

    def test_the_default_budget_the_child_keeps_is_the_one_that_gets_judged(self):
        # An auto-fit pool the server's default cannot hold is reported, not papered over
        # with an unlimited budget; one it can hold certifies as before.
        assert llama_mod._exact_parking_shortfall_mib(
            12 * _GIB,
            args = ["llama-server", "--kv-unified"],
            env = {},
            default_mib = llama_mod._PREEMPT_RAM_DEFAULT_MIB,
        ) == (8192, 12 * 1024, 12 * 1024 + 64)
        assert (
            llama_mod._exact_parking_shortfall_mib(
                4 * _GIB,
                args = ["llama-server", "--kv-unified"],
                env = {},
                default_mib = llama_mod._PREEMPT_RAM_DEFAULT_MIB,
            )
            is None
        )
        # A server left on its default names nothing, so nothing reads back as unlimited.
        assert llama_mod._named_preempt_ram_mib(["llama-server", "--kv-unified"], {}) is None
        assert llama_mod._preempt_ram_disabled_in(["llama-server", "--kv-unified"], env = {}) is False
