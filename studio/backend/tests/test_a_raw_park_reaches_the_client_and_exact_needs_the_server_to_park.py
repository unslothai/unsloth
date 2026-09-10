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
import json
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
from routes.inference import _OPENAI_LLAMA_ADMISSION_WIRE_RESERVE_TOKENS as _RESERVE
from .preempt_fakes import (
    PreemptRecorder,
    RecordingPolicy,
    delta,
    tool_call_chunk,
    done,
    finish,
    run_tool_loop,
    web_search_tool,
)


class TestARawStreamForwardsTheServersPark:
    def test_the_notices_map_onto_the_studio_comments(self):
        assert inference._server_park_sse(llama_mod._SERVER_PARKED_COMMENT) == (
            inference._OPENAI_PREEMPT_SSE_PAUSED
        )
        assert inference._server_park_sse(llama_mod._SERVER_RESUMED_COMMENT) == (
            inference._OPENAI_PREEMPT_SSE_RESUMED
        )
        assert inference._server_park_sse(llama_mod._SERVER_KEEPALIVE_COMMENT) == (
            inference._OPENAI_PREEMPT_SSE_KEEPALIVE
        )
        assert inference._server_park_sse(llama_mod._SERVER_RECOMPUTED_COMMENT) == (
            inference._OPENAI_PREEMPT_SSE_RECOMPUTED
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


class TestARecomputeReachesTheClient:
    """A park the host budget could not hold is restored by re-prefilling, so the answer is no
    longer byte-identical. The server says so three ways (unslothai/llama.cpp#197) and Studio has
    to be tolerant of a build that sends none of them."""

    def test_the_notice_becomes_a_preempt_event_without_ending_an_epoch(self):
        seen = []

        class _Policy:
            def on_server_parked(self):
                seen.append("parked")

            def on_server_resumed(self):
                seen.append("resumed")

        event = LlamaCppBackend._server_park_event(llama_mod._SERVER_RECOMPUTED_COMMENT, _Policy())
        assert event == {"type": "preempt", "state": "recomputed", "source": "server"}
        # It follows the resume it qualifies, so the policy has already been told.
        assert seen == []

    def test_every_state_the_backend_emits_has_a_comment_to_relay_it(self):
        for state in ("paused", "resumed", "recomputed", "keepalive"):
            assert state in inference._OPENAI_PREEMPT_SSE_BY_STATE

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ({"preempt": {"parks": 2, "recomputes": 1}}, {"parks": 2, "recomputes": 1}),
            ({"preempt": {"parks": 2}}, {"parks": 2, "recomputes": 0}),
            ({"preempt": {"parks": "x", "recomputes": None}}, {"parks": 0, "recomputes": 0}),
            ({"preempt": {"recomputes": -3}}, {"parks": 0, "recomputes": 0}),
            # A server that does not report parks says nothing, which is not zero.
            ({}, None),
            ({"preempt": None}, None),
            ({"preempt": 1}, None),
            ("not a chunk", None),
        ],
    )
    def test_the_final_objects_counters_are_read_tolerantly(self, body, expected):
        assert LlamaCppBackend._server_preempt_counts(body) == expected

    def test_the_stream_relays_the_notice_and_carries_the_counters(self):
        source = inspect.getsource(LlamaCppBackend.generate_chat_completion)
        assert "_metadata_preempt = _chunk_preempt" in source
        assert '"preempt": _metadata_preempt' in source
        # A build that counts without writing the notice still reaches the client.
        synth = source.index('yield {"type": "preempt", "state": "recomputed", "source": "server"}')
        assert "not _saw_recompute" in source[synth - 400 : synth]

    def test_the_tool_loop_carries_the_counters_too(self):
        source = inspect.getsource(LlamaCppBackend.generate_chat_completion_with_tools)
        assert "_turn_preempt[_k] = _turn_preempt.get(_k, 0) + _v" in source
        assert '"preempt": dict(_turn_preempt) or None' in source

    def test_the_tool_loop_relays_a_recompute_the_server_only_counted(self, monkeypatch):
        # The final object counts a recompute no notice announced: the client is told once,
        # from the count, the way the plain generator already does.
        signal = preemption_mod.PreemptSignal()
        stream = [delta("x"), delta("y", preempt = {"parks": 1, "recomputes": 1}), finish(), done()]
        recorder = PreemptRecorder(monkeypatch, [stream], signal = signal)
        events = run_tool_loop(
            recorder.backend, signal = signal, policy = RecordingPolicy(), tools = [web_search_tool()]
        )
        dicts = [e for e in events if isinstance(e, dict)]
        recomputed = [
            e for e in dicts if e.get("type") == "preempt" and e.get("state") == "recomputed"
        ]
        assert recomputed == [{"type": "preempt", "state": "recomputed", "source": "server"}]
        metadata = [e for e in dicts if e.get("type") == "metadata"][-1]
        assert metadata["preempt"] == {"parks": 1, "recomputes": 1}

    def test_a_turn_of_several_requests_sums_their_counters(self, monkeypatch):
        # Each final object counts its own request; the turn's metadata is every request.
        signal = preemption_mod.PreemptSignal()
        first = [
            tool_call_chunk(),
            "data: "
            + json.dumps(
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    "preempt": {"parks": 1, "recomputes": 1},
                }
            )
            + "\n",
            done(),
        ]
        second = [delta("x", preempt = {"parks": 0, "recomputes": 0}), finish(), done()]
        recorder = PreemptRecorder(monkeypatch, [first, second], signal = signal, execute_tool = True)
        events = run_tool_loop(
            recorder.backend, signal = signal, policy = RecordingPolicy(), tools = [web_search_tool()]
        )
        assert len(recorder.payloads) == 2
        metadata = [e for e in events if isinstance(e, dict) and e.get("type") == "metadata"][-1]
        assert metadata["preempt"] == {"parks": 1, "recomputes": 1}

    def test_a_stream_of_a_server_that_does_not_park_scans_no_notices(self):
        # The tracker's tail scan runs on every read; with no park grace no notice can come.
        source = " ".join(inspect.getsource(LlamaCppBackend._install_cancel_aware_read).split())
        assert "ServerParkNotices(stall_grace) if stall_grace is not None else None" in source
        assert "if notices is not None: notices.feed(data)" in source
        assert "parked = notices is not None and notices.excuses_silence()" in source

    def test_a_build_that_writes_the_notices_is_never_asked_the_aggregate(self):
        # A stream that heard nothing on such a build is not parked; the aggregate reading
        # excused an unrelated stall for as long as somebody else stayed parked.
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._server_park_notices = True
        assert backend._server_park_grace() is False
        source = " ".join(inspect.getsource(LlamaCppBackend.load_model).split())
        assert 'self._server_park_notices = "exact_concurrency" in _server_props' in source
        assert "self._server_park_notices = False" in source

    def test_a_notice_the_server_wrote_is_not_relayed_twice(self, monkeypatch):
        signal = preemption_mod.PreemptSignal()
        stream = [
            delta("x"),
            llama_mod._SERVER_RECOMPUTED_COMMENT + "\n",
            delta("y", preempt = {"parks": 1, "recomputes": 1}),
            finish(),
            done(),
        ]
        recorder = PreemptRecorder(monkeypatch, [stream], signal = signal)
        events = run_tool_loop(
            recorder.backend, signal = signal, policy = RecordingPolicy(), tools = [web_search_tool()]
        )
        recomputed = [
            e
            for e in events
            if isinstance(e, dict) and e.get("type") == "preempt" and e.get("state") == "recomputed"
        ]
        assert len(recomputed) == 1


_GIB = 1024 * 1024 * 1024


class TestTheNamedBudgetIsJudged:
    def test_the_shortfall_follows_the_server_that_came_up(self):
        # A drafter priced before launch and dropped by the retry: the budget is judged again
        # with no draft state, before a healthy exact server is failed on the stale answer.
        # Whitespace folded: the formatter wraps these expressions.
        source = " ".join(inspect.getsource(LlamaCppBackend.load_model).split())
        at = source.index("_mtp_will_engage and not _mtp_active_for_launched_server")
        again = source.index("_exact_parking_shortfall_mib(", at)
        assert "draft_bytes = 0" in source[again : again + 400]
        assert "parking_holds = _exact_short is None" in source[again:]
        # The auto-fit pricing drops the draft state on the same server.
        assert (
            "_draft_kv_state_bytes(_fitted_ctx) if _mtp_active_for_launched_server else 0" in source
        )

    def test_a_budget_below_the_pool_is_a_shortfall(self):
        short = llama_mod._exact_parking_shortfall_mib(
            12 * _GIB, args = ["--preempt-ram", "1024"], env = {}
        )
        assert short == (1024, 12 * 1024, 12 * 1024 + 64)

    def test_the_draft_state_and_the_parked_slots_are_budgeted_too(self):
        # A 1024 MiB pool beside a draft cache of its own size, four slots: three sequences
        # can be parked at once, each having grown to the whole pool before its park, and the
        # server holds one more snapshot while it rotates another in.
        pool = 1024 * 1024 * 1024
        need = llama_mod._exact_parking_need_mib(pool, draft_bytes = pool, parallel = 4)
        assert need == 4 * 2048 + llama_mod._PARKING_MARGIN_MIB
        # A 768 MiB history parked twice is 1536 MiB; the per-slot share this replaces
        # accepted 1088 MiB for a 1024 MiB pool at four slots.
        assert llama_mod._exact_parking_need_mib(pool, parallel = 4) == 4 * 1024 + 64
        assert llama_mod._exact_parking_shortfall_mib(
            pool, args = ["--preempt-ram", "1088"], env = {}, parallel = 4
        ) == (1088, 1024, 4 * 1024 + 64)
        assert llama_mod._exact_parking_need_mib(pool) == 1088
        short = llama_mod._exact_parking_shortfall_mib(
            pool,
            args = ["--preempt-ram", "1088"],
            env = {},
            draft_bytes = pool,
            parallel = 4,
        )
        assert short is not None
        named, saved, reported = short
        assert (named, saved, reported) == (1088, 2048, need)
        # A budget that does hold it is no shortfall.
        assert (
            llama_mod._exact_parking_shortfall_mib(
                pool,
                args = ["--preempt-ram", str(need)],
                env = {},
                draft_bytes = pool,
                parallel = 4,
            )
            is None
        )

    def test_a_budget_past_the_hosts_free_memory_is_a_shortfall(self, monkeypatch):
        # The budget is a cap the server parks up to, not an allocation: a park the host
        # cannot hold fails its allocation and is re-prefilled, so it is judged like a
        # budget too small: `auto` runs without the mode and `on` fails the load.
        source = " ".join(inspect.getsource(LlamaCppBackend.load_model).split())
        site = source.index('cmd.extend(["--preempt-ram", str(_exact_budget)])')
        window = source[site : site + 2600]
        # Judged for every budget in force, not only the one sized here: a named budget
        # that holds every park can still be more than the host has.
        assert "if _exact_kv_bytes > 0: _exact_cap = _exact_budget" in window
        assert "_exact_cap = _named_preempt_ram_mib(" in window
        assert "_exact_cap = _PREEMPT_RAM_DEFAULT_MIB" in window
        assert "_exact_writes = min(_exact_cap, _exact_writes)" in window
        assert "_available_host_memory_mib()" in window
        assert "_exact_writes > _host_free_mib" in window
        assert "self._exact_host_short = (_exact_writes, _host_free_mib)" in window
        assert "if _exact_setting == _exact.EXACT_AUTO: _exact_wanted = False" in window
        assert "parking_holds = _exact_short is None and _exact_host_short is None" in source
        assert "self._exact_host_short = None" in source

    def test_the_host_is_read_again_once_the_weights_are_resident(self):
        # The reading before launch predates the model: a load that keeps the weights in
        # anonymous host memory takes what the parks were told they could have.
        source = " ".join(inspect.getsource(LlamaCppBackend.load_model).split())
        launch = source.index('cmd.extend(["--preempt-ram", str(_exact_budget)])')
        assert "self._exact_parking_writes = _exact_writes" in source[launch : launch + 1800]
        after = source.index("_server_props = self._query_server_props() or {}")
        again = source.index("_exact_host_shortfall_after_load(")
        assert again < after, "judged after the props read, which is after the launch"
        assert "if _exact_host_short is None: " in source[again - 400 : again]
        assert (
            "parking_holds = _exact_short is None and _exact_host_short is None" in source[again:]
        )
        assert llama_mod._exact_host_shortfall_after_load(4096, 8192) is None
        assert llama_mod._exact_host_shortfall_after_load(4096, 4096) is None
        assert llama_mod._exact_host_shortfall_after_load(4096, 1000) == (4096, 1000)
        assert llama_mod._exact_host_shortfall_after_load(None, 1000) is None
        assert llama_mod._exact_host_shortfall_after_load(4096, None) is None

    def test_a_single_slot_still_budgets_the_one_snapshot_it_writes(self):
        pool = 1024 * 1024 * 1024
        assert llama_mod._exact_parking_need_mib(pool, draft_bytes = pool, parallel = 1) == (
            2048 + llama_mod._PARKING_MARGIN_MIB
        )

    def test_the_launch_prices_the_draft_state_and_the_slot_count(self):
        source = inspect.getsource(LlamaCppBackend.load_model)
        assert "_exact_draft_bytes = _draft_kv_state_bytes(effective_ctx)" in source
        for call in ("_server_owned_parking_budget_mib(", "_exact_parking_shortfall_mib("):
            site = source.index(call)
            window = source[site : site + 500]
            assert "draft_bytes = _exact_draft_bytes" in window
            assert "parallel = n_parallel" in window
        # The drafter's weights stay resident over a park, so the reserve is not the measure.
        draft = inspect.getsource(LlamaCppBackend.load_model)
        assert "self._mtp_draft_kv_bytes(" in draft

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

    def test_the_pool_sized_after_launch_prices_its_draft_state_too(self):
        source = " ".join(inspect.getsource(LlamaCppBackend.load_model).split())
        site = source.index("_fitted_bytes = _kv_bytes(_fitted_ctx)")
        window = source[site : site + 700]
        assert (
            "_draft_kv_state_bytes(_fitted_ctx) if _mtp_active_for_launched_server else 0" in window
        )
        assert "draft_bytes = _fitted_draft" in window
        assert "parallel = n_parallel" in window
        # A draft cache with no dimensions is a park nobody can size, so it is not certified.
        assert "if _fitted_draft is None:" in window

    def test_the_budget_named_for_an_unknown_pool_is_judged_after_launch(self):
        # An auto-fit context leaves the pool unknown at launch, so the launch names the unsized
        # budget and that figure is judged against the context the server chose. The server has
        # no default of its own left to judge: naming nothing parks nothing.
        assert llama_mod._PREEMPT_RAM_DEFAULT_MIB == 0
        assert (
            llama_mod._exact_parking_shortfall_mib(
                12 * _GIB, args = [], env = {}, default_mib = llama_mod._PREEMPT_RAM_DEFAULT_MIB
            )
            is None
        )
        unsized = ["--preempt-ram", str(llama_mod._PREEMPT_RAM_UNSIZED_MIB)]
        short = llama_mod._exact_parking_shortfall_mib(12 * _GIB, args = unsized, env = {})
        assert short == (8192, 12 * 1024, 12 * 1024 + 64)
        assert llama_mod._exact_parking_shortfall_mib(4 * _GIB, args = unsized, env = {}) is None
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
        window = source[judged : judged + 2700]
        assert "self._query_server_n_ctx()" in window
        assert "default_mib = _PREEMPT_RAM_DEFAULT_MIB" in window
        assert "_named_now = _named_preempt_ram_mib(" in window
        assert (
            "_exact_short = (_named_now or 0, 0, 0)" in window
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
    # Opted in: exact concurrency needs the SERVER to park, and the server only parks where
    # preemption was asked for, so the switch is what these cases hold still while they vary
    # the launch line. `TestTheGlobalOptOutBlocksAnAutoLaunch` is the other half.
    pytestmark = pytest.mark.usefixtures("preemption_opted_in")

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

    def test_an_inherited_cpu_placement_blocks_an_auto_launch(self, monkeypatch):
        # The env twins reach the child whatever the argv says, and they append.
        monkeypatch.delenv(preemption_mod.PREEMPT_MODE_ENV, raising = False)
        why = llama_mod._exact_auto_blocker(
            exact.EXACT_AUTO, self._ARGV, {"LLAMA_ARG_N_CPU_MOE": "8"}
        )
        assert why and "LLAMA_ARG_N_CPU_MOE=8" in why

    def test_manual_mode_drops_the_placement_twins_before_the_preflight(self, monkeypatch):
        # Manual mode scrubs LLAMA_ARG_CPU_MOE / LLAMA_ARG_N_CPU_MOE from the child, so the
        # preflight must not block on a value the child never gets; an -ot stays.
        monkeypatch.delenv(preemption_mod.PREEMPT_MODE_ENV, raising = False)
        inherited = {"LLAMA_ARG_N_CPU_MOE": "8", "LLAMA_ARG_OVERRIDE_TENSOR": "attn=CUDA0"}
        env = llama_mod._exact_preflight_env(inherited, "manual")
        assert "LLAMA_ARG_N_CPU_MOE" not in env and env["LLAMA_ARG_OVERRIDE_TENSOR"] == "attn=CUDA0"
        assert llama_mod._exact_auto_blocker(exact.EXACT_AUTO, self._ARGV, env) is None
        # Any other memory mode hands the variable on, so it blocks.
        assert llama_mod._exact_auto_blocker(
            exact.EXACT_AUTO, self._ARGV, llama_mod._exact_preflight_env(inherited, "auto")
        )
        cpu = llama_mod._exact_preflight_env({"LLAMA_ARG_OVERRIDE_TENSOR": "exps=CPU"}, "manual")
        assert llama_mod._exact_auto_blocker(exact.EXACT_AUTO, self._ARGV, cpu)
        # Both launch sites judge the child's environment, not the parent's.
        source = " ".join(inspect.getsource(LlamaCppBackend.load_model).split())
        assert source.count("_exact_preflight_env(os.environ, gpu_memory_mode)") == 2
        assert "contradicting_env(os.environ)" not in source

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
        # Both less the wire reserve, which comes out of every bound.
        assert studio_only == 16384 // 4 - 1000 - _RESERVE, "Studio-only: the honest share"
        assert (
            parked == 16384 - 1000 - _RESERVE
        ), "a parking server: the window, like every other stream"


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
        # Spelled as the operator set it. Unset is the default, and "=0" there would name a
        # variable nobody wrote.
        assert why and f"{preemption_mod.PREEMPT_ENV} is not set" in why
        monkeypatch.setenv(preemption_mod.PREEMPT_ENV, "0")
        assert f"{preemption_mod.PREEMPT_ENV}=0" in llama_mod._exact_auto_blocker(
            exact.EXACT_AUTO, ["llama-server"], {}
        )
        monkeypatch.delenv(preemption_mod.PREEMPT_ENV, raising = False)
        # A named budget does not buy the child a park either: `_stand_down_child_parking`
        # zeroes it, so the mode would be started for a server that never parks.
        for argv, env in (
            (["llama-server", "--preempt-ram", "4096"], {}),
            (["llama-server"], {"LLAMA_ARG_PREEMPT_RAM": "4096"}),
        ):
            blocked = llama_mod._exact_auto_blocker(exact.EXACT_AUTO, argv, env)
            assert blocked and preemption_mod.PREEMPT_ENV in blocked
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
        first_state = stream._unsloth_park_state
        assert inference._install_park_aware_read(second, two) is True
        # A second request gets its own notice reader, so the first one's park is not this
        # stream's excuse, and the aggregate probe behind it is the second's.
        assert stream._unsloth_park_state is not first_state
        assert stream._unsloth_park_state["park"].excuses_silence() is False
        assert first_state["park"].excuses_silence() is True
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
        assert "park_above.excuses_silence()" not in source
        assert source.count("await _probe_off_the_loop(park_above.excuses_silence)") == 2

    def test_this_streams_own_park_excuses_it_without_asking_the_aggregate(self, monkeypatch):
        import httpcore

        asked = {"n": 0}

        def probe():
            asked["n"] += 1
            return False

        reads = [b": preempted\n\n", httpcore.ReadTimeout("parked"), b"data: a\n\n"]

        async def read(max_bytes, timeout = None):
            item = reads.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        stream = self._wrapped(monkeypatch, read, probe)

        async def run():
            first = await stream.read(65536, timeout = 1200.0)
            return first, await stream.read(65536, timeout = 1200.0)

        first, second = asyncio.run(run())
        assert first == b": preempted\n\n" and second == b"data: a\n\n"
        # The aggregate reading is never consulted: this stream said it was parked.
        assert asked["n"] == 0

    def test_a_stream_that_resumed_is_no_longer_excused_by_a_neighbours_park(self, monkeypatch):
        import httpcore

        # `/metrics` still counts the neighbour as parked; this stream has said it resumed, so
        # its silence is its own stall and the relay must end rather than wait out the cap.
        reads = [b": preempted\n\n: resumed\n\n", httpcore.ReadTimeout("stalled")]

        async def read(max_bytes, timeout = None):
            item = reads.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        stream = self._wrapped(monkeypatch, read, lambda: True)

        async def run():
            await stream.read(65536, timeout = 1200.0)
            await stream.read(65536, timeout = 1200.0)

        with pytest.raises(httpcore.ReadTimeout):
            asyncio.run(run())


class TestAStreamsOwnParkIsTheExcuse:
    """`/metrics` counts every request, so the aggregate reading excused a stream stalled for its
    own reason while an unrelated chat sat parked, and kept excusing one that had resumed."""

    def test_the_notices_are_the_ones_the_backend_relays(self):
        assert preemption_mod._PARK_NOTICE_PARKED == (
            b"\n" + llama_mod._SERVER_PARKED_COMMENT.encode()
        )
        assert preemption_mod._PARK_NOTICE_RESUMED == (
            b"\n" + llama_mod._SERVER_RESUMED_COMMENT.encode()
        )

    def test_a_park_this_stream_was_told_of_needs_no_probe(self):
        notices = preemption_mod.ServerParkNotices(lambda: False)
        notices.feed(b"data: a\n\n: preempted\n\n")
        assert notices.parked is True
        assert notices.excuses_silence() is True

    def test_a_resume_retires_the_aggregate_for_this_stream(self):
        notices = preemption_mod.ServerParkNotices(lambda: True)
        notices.feed(b": preempted\n\n")
        assert notices.excuses_silence() is True
        notices.feed(b": resumed\n\n")
        assert notices.parked is False
        assert notices.heard_a_notice is True
        assert notices.excuses_silence() is False

    def test_one_read_carrying_both_takes_the_last(self):
        notices = preemption_mod.ServerParkNotices(lambda: True)
        notices.feed(b": preempted\n\n: resumed\n\n")
        assert notices.excuses_silence() is False
        notices.feed(b": resumed\n\n: preempted\n\n")
        assert notices.excuses_silence() is True

    def test_a_notice_split_across_two_reads_is_still_read(self):
        notices = preemption_mod.ServerParkNotices(lambda: False)
        notices.feed(b"data: a\n\n: preem")
        assert notices.excuses_silence() is False
        notices.feed(b"pted\n\n")
        assert notices.excuses_silence() is True

    def test_a_payload_quoting_a_notice_is_not_one(self):
        # The notices are SSE comments, so they start a line; a model writing about one does not.
        notices = preemption_mod.ServerParkNotices(lambda: False)
        notices.feed(b'data: {"content":"the log said : preempted"}\n\n')
        assert notices.heard_a_notice is False
        assert notices.excuses_silence() is False

    def test_a_build_that_sends_nothing_still_gets_the_aggregate(self):
        assert preemption_mod.ServerParkNotices(lambda: True).excuses_silence() is True
        assert preemption_mod.ServerParkNotices(lambda: False).excuses_silence() is False
        assert preemption_mod.ServerParkNotices(None).excuses_silence() is False

        def raises():
            raise RuntimeError("/metrics is down")

        assert preemption_mod.ServerParkNotices(raises).excuses_silence() is False

    def test_a_line_oriented_source_is_read_the_same(self):
        notices = preemption_mod.ServerParkNotices(lambda: False)
        notices.feed_line("data: a")
        notices.feed_line(": preempted")
        assert notices.excuses_silence() is True
        notices.feed_line(": resumed")
        assert notices.excuses_silence() is False
        # A non-text item from a parsed iterator is not a notice and must not raise.
        notices.feed_line({"type": "preempt"})
        assert notices.excuses_silence() is False

    def test_the_backends_own_stream_reads_its_notices_too(self):
        source = inspect.getsource(LlamaCppBackend._install_cancel_aware_read)
        assert "notices = _preemption.ServerParkNotices(stall_grace)" in source
        assert "notices.feed(data)" in source
        assert "parked = notices is not None and notices.excuses_silence()" in source
        assert "bool(stall_grace())" not in source


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
        # On, so the launch line is the only thing left to block on: unset is the opt-out, and
        # that blocks every auto launch on its own.
        monkeypatch.setenv(preemption_mod.PREEMPT_ENV, "1")
        reason = llama_mod._exact_auto_blocker(
            exact.EXACT_AUTO, ["llama-server", "--kv-unified", "--no-kv-unified"], {}
        )
        assert reason is not None and "--no-kv-unified" in reason
        assert (
            llama_mod._exact_auto_blocker(exact.EXACT_AUTO, ["llama-server", "--kv-unified"], {})
            is None
        )

    def test_cpu_expert_placement_is_read_off_the_whole_launch_line(self, monkeypatch):
        monkeypatch.delenv(preemption_mod.PREEMPT_MODE_ENV, raising = False)
        # As above: the launch line is what is under test, so the switch is held on.
        monkeypatch.setenv(preemption_mod.PREEMPT_ENV, "1")
        # Studio emits --n-cpu-moe itself, so the preflight reads the line, not just the extras,
        # and llama-server does not refuse this one: exact concurrency does.
        reason = llama_mod._exact_auto_blocker(
            exact.EXACT_AUTO, ["llama-server", "--kv-unified", "--n-cpu-moe", "12"], {}
        )
        assert reason is not None and "--n-cpu-moe" in reason
        assert "llama-server cannot combine" not in reason
        assert (
            llama_mod._exact_auto_blocker(
                exact.EXACT_AUTO, ["llama-server", "--kv-unified", "--n-cpu-moe", "0"], {}
            )
            is None
        )


class TestTheGlobalOptOutBlocksAnExactOnLaunchToo:
    """`auto` was preflighted for the opt-out, `on` was not: the exact launch sized a parking budget
    of its own, and `_stand_down_child_parking` read any `--preempt-ram` as one somebody named, so
    the child parked with UNSLOTH_LLAMA_ADMISSION_PREEMPT=0 set. The budget is generated only when
    the child is going to be allowed to park at all, and one owner pauses chats: under the opt-out
    or `studio` mode a named budget is overridden rather than left to run beside Studio."""

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
        assert llama_mod._child_parking_stands_down() is True
        # So nothing names a budget, and with the server parking only when it is told to there
        # is nothing to write to the child either: it comes up exactly as upstream ships it.
        env: dict = {}
        assert llama_mod._stand_down_child_parking(env, argv) == []
        assert env == {}
        assert llama_mod._preempt_ram_disabled_in(argv, env = env) is True

    def test_a_budget_somebody_named_is_overridden_and_named(self, monkeypatch):
        self._opt_out(monkeypatch)
        for argv, env, overridden in (
            (["llama-server", "--preempt-ram", "4096"], {}, ["--preempt-ram 4096"]),
            (["llama-server", "--preempt-ram=4096"], {}, ["--preempt-ram=4096"]),
            (["llama-server"], {"LLAMA_ARG_PREEMPT_RAM": "4096"}, ["LLAMA_ARG_PREEMPT_RAM=4096"]),
        ):
            assert llama_mod._child_parking_stands_down() is True
            assert llama_mod._stand_down_child_parking(env, argv) == overridden
            assert llama_mod._preempt_ram_disabled_in(argv, env = env)

    def test_studio_side_pausing_stands_the_child_down_the_same_way(self, monkeypatch):
        monkeypatch.setattr(llama_mod._preemption, "preemption_enabled", lambda: True)
        monkeypatch.setenv(preemption_mod.PREEMPT_MODE_ENV, "studio")
        assert llama_mod._child_parking_stands_down() is True
        argv = ["llama-server", "--preempt-ram", "4096"]
        env: dict = {}
        assert llama_mod._stand_down_child_parking(env, argv) == ["--preempt-ram 4096"]
        assert llama_mod._preempt_ram_disabled_in(argv, env = env)
        assert llama_mod._child_parking_stand_down_reason() == "UNSLOTH_LLAMA_PREEMPT_MODE=studio"

    def test_a_launch_with_preemption_on_still_gets_its_budget(self, monkeypatch):
        monkeypatch.delenv(preemption_mod.PREEMPT_MODE_ENV, raising = False)
        monkeypatch.setattr(llama_mod._preemption, "preemption_enabled", lambda: True)
        argv = ["llama-server", "--kv-unified"]
        assert llama_mod._child_parking_stands_down() is False
        assert llama_mod._exact_parking_budget_mib(self._POOL, args = argv, env = {}) is not None
        env: dict = {}
        assert llama_mod._stand_down_child_parking(env, argv) is None
        assert env == {}
        assert argv == ["llama-server", "--kv-unified"]

    def test_the_stand_down_and_the_launch_read_one_predicate(self):
        stand_down = inspect.getsource(llama_mod._stand_down_child_parking)
        assert "_child_parking_stands_down(server_supports)" in stand_down
        source = inspect.getsource(LlamaCppBackend.load_model)
        guard = source.index('server_caps.get("supports_preempt_ram")')
        window = source[guard : source.index("self._exact_pool_unknown = _exact_kv_bytes <= 0")]
        assert "not _child_parking_stands_down(" in window
        assert window.index("not _child_parking_stands_down(") < window.index("_exact_budget = ")


class TestParkingIsOffUntilTheLaunchAsksForIt:
    """A llama-server launched without ``--preempt-ram`` parks nothing (unslothai/llama.cpp#197),
    so server-side preemption is something Studio asks for by name. A default install passes the
    flag no more than it passes any other and its child is a stock llama-server; with the switch
    on, the launch names the budget itself rather than relying on a default that is now zero."""

    _POOL = 12 * _GIB
    _ARGV = ["llama-server", "--kv-unified"]

    @staticmethod
    def _plan(
        argv,
        env,
        *,
        supported = True,
        unified = True,
        pool = 0,
        parallel = 1,
    ):
        """The launch's decision, taken with the helpers and in the order load_model takes it.

        ``test_the_launch_takes_the_decision_this_way`` pins that this is that order.
        Returns the argv, the child environment, the budget named, whether the child parks, and
        what the stand-down overrode."""
        cmd = list(argv)
        supported = bool(supported)
        owned = supported and not llama_mod._child_parking_stands_down(supported)
        budget = None
        if owned:
            budget = llama_mod._server_owned_parking_budget_mib(
                pool,
                args = cmd,
                env = env,
                server_supports = supported,
                kv_unified = unified,
                parallel = parallel,
            )
            if budget is not None:
                cmd.extend(["--preempt-ram", str(budget)])
        parks = owned and not llama_mod._preempt_ram_disabled_in(cmd, env = env)
        child_env = dict(env)
        overridden = llama_mod._stand_down_child_parking(child_env, cmd, server_supports = supported)
        return cmd, child_env, budget, parks, overridden

    def test_a_default_install_launches_a_stock_llama_server(self, monkeypatch):
        monkeypatch.delenv(preemption_mod.PREEMPT_MODE_ENV, raising = False)
        monkeypatch.delenv(preemption_mod.PREEMPT_ENV, raising = False)
        # An unset switch is off; pinned here so this reads the default install rather than
        # whichever way the constant happens to be set.
        monkeypatch.setattr(preemption_mod, "DEFAULT_PREEMPT_ENABLED", False)
        cmd, env, budget, parks, overridden = self._plan(self._ARGV, {}, pool = self._POOL)
        assert budget is None
        assert cmd == self._ARGV, "no --preempt-ram on the launch line"
        assert env == {}, "and no LLAMA_ARG_PREEMPT_RAM in the child environment either"
        assert parks is False
        # Nothing was overridden, so the load has nothing to warn about.
        assert overridden == []

    def test_the_switch_on_names_the_budget_and_the_server_parks(self, monkeypatch):
        monkeypatch.delenv(preemption_mod.PREEMPT_MODE_ENV, raising = False)
        monkeypatch.setenv(preemption_mod.PREEMPT_ENV, "1")
        cmd, env, budget, parks, overridden = self._plan(
            self._ARGV, {}, pool = self._POOL, parallel = 4
        )
        assert budget == llama_mod._exact_parking_need_mib(self._POOL, parallel = 4)
        assert cmd[-2:] == ["--preempt-ram", str(budget)]
        assert parks is True
        assert overridden is None, "the child keeps the parking Studio just asked it for"
        assert env == {}
        # The property the rest of Studio reads follows the launch, not the build.
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._server_preempts_kv = parks
        backend._kv_cache_unified = True
        assert backend.server_preempts_kv is True

    @pytest.mark.parametrize(("supported", "unified"), [(False, True), (True, False)])
    def test_a_child_that_could_not_park_anyway_is_named_nothing(
        self, monkeypatch, supported, unified
    ):
        monkeypatch.delenv(preemption_mod.PREEMPT_MODE_ENV, raising = False)
        monkeypatch.setenv(preemption_mod.PREEMPT_ENV, "1")
        argv = ["llama-server"] + (["--kv-unified"] if unified else [])
        cmd, _, budget, parks, _ = self._plan(
            argv, {}, supported = supported, unified = unified, pool = self._POOL
        )
        assert budget is None and parks is False
        assert cmd == argv

    def test_a_budget_the_user_named_still_wins_and_their_zero_still_stands_it_down(
        self, monkeypatch
    ):
        monkeypatch.delenv(preemption_mod.PREEMPT_MODE_ENV, raising = False)
        monkeypatch.setenv(preemption_mod.PREEMPT_ENV, "1")
        for argv, env in (
            (self._ARGV + ["--preempt-ram", "4096"], {}),
            (self._ARGV, {"LLAMA_ARG_PREEMPT_RAM": "4096"}),
        ):
            cmd, _, budget, parks, _ = self._plan(argv, env, pool = self._POOL)
            assert budget is None, "Studio sizes no budget over one somebody named"
            assert cmd == argv and parks is True
        for argv, env in (
            (self._ARGV + ["--preempt-ram", "0"], {}),
            (self._ARGV, {"LLAMA_ARG_PREEMPT_RAM": "0"}),
        ):
            cmd, _, budget, parks, _ = self._plan(argv, env, pool = self._POOL)
            assert budget is None and parks is False
            assert cmd == argv

    def test_studio_mode_names_no_flag_and_hands_the_child_a_zero(self, monkeypatch):
        monkeypatch.setenv(preemption_mod.PREEMPT_ENV, "1")
        monkeypatch.setenv(preemption_mod.PREEMPT_MODE_ENV, "studio")
        cmd, env, budget, parks, overridden = self._plan(self._ARGV, {}, pool = self._POOL)
        assert budget is None and parks is False
        assert cmd == self._ARGV
        # Studio's own preemptor is armed here, so a build that parks after all is told not to.
        assert env["LLAMA_ARG_PREEMPT_RAM"] == "0"
        assert overridden == []

    def test_the_launch_takes_the_decision_this_way(self):
        source = " ".join(inspect.getsource(LlamaCppBackend.load_model).split())
        site = source.index('_park_supported = bool(server_caps.get("supports_preempt_ram"))')
        window = source[site : site + 1500]
        assert (
            "_park_owned = _park_supported and not _child_parking_stands_down(_park_supported)"
            in window
        )
        assert "_exact_budget = _server_owned_parking_budget_mib(" in window
        assert "kv_unified = _kv_unified_from_args(cmd)," in window
        assert 'cmd.extend(["--preempt-ram", str(_exact_budget)])' in window
        # Read off what this launch turned on, not off what the build could do.
        assert "self._server_preempts_kv = _park_owned and not _preempt_ram_disabled_in(" in source
        assert 'self._server_preempts_kv = bool(server_caps.get("supports_preempt_ram"))' not in (
            source
        )
        # Named before the child environment is written, so the stand-down can still override it.
        assert source.index("_exact_budget = _server_owned_parking_budget_mib(") < source.index(
            "_parking_overridden = _stand_down_child_parking("
        )


class TestAnAbandonedExactAttemptLeavesNoUnlimitedParkingBudget:
    """The exact launch used to append ``--preempt-ram -1`` for an auto-fit context, before the
    running server had confirmed the mode, and every way the attempt is abandoned keeps that argv
    (the refusal rung relaunches the same command; a build ignoring the variable never relaunches).
    A server with no exact concurrency then parked into unbounded host RAM. So the launch names the
    unsized budget for a pool it cannot measure, and judges that figure after launch."""

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
        judged = spawn.index('_exact_running = _server_props.get("exact_concurrency") is True')
        window = spawn[judged : judged + 6000]
        assert "if _exact_running:" in window
        assert "runs the mode, but it cannot hold every " in window
        assert "park, and a chat re-prefilled after a park it could not hold" in window
        assert "came up without it" in window.split("if _exact_running:")[1]
        assert window.count("_exact_what") >= 4

    def test_the_budget_the_launch_named_is_the_one_that_gets_judged(self):
        # An auto-fit pool the named budget cannot hold is reported, not papered over with an
        # unlimited one; one it can hold certifies as before.
        named = ["llama-server", "--kv-unified", "--preempt-ram", "8192"]
        assert llama_mod._exact_parking_shortfall_mib(
            12 * _GIB,
            args = named,
            env = {},
            default_mib = llama_mod._PREEMPT_RAM_DEFAULT_MIB,
        ) == (8192, 12 * 1024, 12 * 1024 + 64)
        assert (
            llama_mod._exact_parking_shortfall_mib(
                4 * _GIB,
                args = named,
                env = {},
                default_mib = llama_mod._PREEMPT_RAM_DEFAULT_MIB,
            )
            is None
        )
        # A launch that names nothing gets a server that parks nothing, so there is no budget to
        # judge and nothing reads back as unlimited.
        assert llama_mod._named_preempt_ram_mib(["llama-server", "--kv-unified"], {}) is None
        assert llama_mod._preempt_ram_disabled_in(["llama-server", "--kv-unified"], env = {}) is True
        assert (
            llama_mod._exact_parking_shortfall_mib(
                12 * _GIB,
                args = ["llama-server", "--kv-unified"],
                env = {},
                default_mib = llama_mod._PREEMPT_RAM_DEFAULT_MIB,
            )
            is None
        )
