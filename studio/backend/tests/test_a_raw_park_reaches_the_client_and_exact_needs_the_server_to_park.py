# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Four gaps between what a park promises and what reached the client or the ledger.

1. The raw passthrough loops (Responses, chat and completions passthrough, Anthropic
   passthrough) relayed ``data:`` lines only. A request the server parked sent
   ``: preempted`` and a ``: preempt-keepalive`` every two seconds, each of which reset
   the stall clock and reached nobody, so a long park sent the client no bytes at all.
   ``_server_park_sse`` maps those notices onto the comments every other surface forwards.

2. Exact concurrency was reported ``on`` under ``UNSLOTH_LLAMA_PREEMPT_MODE=studio``. A chat
   the server parks is restored cell for cell; one Studio pauses is resumed as a
   continuation with fresh sampler and draft state, so the promise needs the server to be
   the one pausing. ``_exact_state_after_launch`` now takes ``server_parks``.

3. A user-named ``--preempt-ram`` below the KV pool was reported exact too, though a park
   that outgrows it is re-prefilled. ``_exact_parking_shortfall_mib`` names the shortfall
   and the state reads ``unavailable`` for it.

4. A swap build predating the stream notices parks in silence. The read wrapper excused the
   silence from `/metrics` and forwarded nothing, so a durable run's lease had nothing to
   renew on and the sweeper could cancel a legitimate park. The backend stamps the excuse
   and the run loop renews from it.
"""

import asyncio
import inspect
import time

import pytest

import core.inference.chat_generation_runs as runs
import core.inference.llama_cpp as llama_mod
import routes.inference as inference
from core.inference import llama_exact as exact
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
        assert self._state(setting = setting, server_parks = False) == (
            exact.EXACT_STATE_UNAVAILABLE
        )

    @pytest.mark.parametrize("setting", ["auto", "on"])
    def test_a_budget_below_the_pool_is_not_exact(self, setting):
        assert self._state(setting = setting, parking_holds = False) == (
            exact.EXACT_STATE_UNAVAILABLE
        )

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
        assert llama_mod._exact_parking_shortfall_mib(0, args = ["--preempt-ram", "1"], env = {}) is None

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
        assert "await self._try_touch_progress(run_id)" in source[branch:branch + 400]
