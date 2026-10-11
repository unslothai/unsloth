# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Preemption is on by default only where the running llama-server proves it parks slots.

Anything else (a stock or older server, an unreadable report, a host with no RAM to spare, or
``UNSLOTH_LLAMA_ADMISSION_PREEMPT=0``) keeps the fair share admission every install ran before."""

import pytest

from core.inference import llama_cpp as llama_mod
from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend

_GIB = 1024 * 1024 * 1024


class _Launch:
    """The launch's parking decision, in load_model's order (pinned in the sibling test module)."""

    _POOL = 12 * 1024 * 1024 * 1024
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
        cmd = list(argv)
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
                bound_to_host = True,
            )
            if budget is not None:
                cmd.extend(["--preempt-ram", str(budget)])
        parks = owned and not llama_mod._preempt_ram_disabled_in(cmd, env = env)
        child_env = dict(env)
        overridden = llama_mod._stand_down_child_parking(child_env, cmd, server_supports = supported)
        return cmd, child_env, budget, parks, overridden


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(preemption.PREEMPT_ENV, raising = False)
    monkeypatch.delenv(preemption.PREEMPT_MODE_ENV, raising = False)
    preemption.note_server_parking(False)
    yield
    preemption.note_server_parking(False)


class TestTheSwitch:
    @pytest.mark.parametrize(
        ("raw", "setting"),
        [
            (None, "auto"),
            ("", "auto"),
            ("auto", "auto"),
            ("1", "on"),
            ("true", "on"),
            ("0", "off"),
            ("off", "off"),
        ],
    )
    def test_spellings(self, monkeypatch, raw, setting):
        if raw is not None:
            monkeypatch.setenv(preemption.PREEMPT_ENV, raw)
        assert preemption.preemption_setting() == setting
        assert preemption.preemption_wanted() is (setting != "off")

    def test_auto_follows_the_server(self):
        assert preemption.preemption_enabled() is False
        preemption.note_server_parking(True)
        assert preemption.preemption_enabled() is True
        preemption.note_server_parking(False)
        assert preemption.preemption_enabled() is False

    def test_off_wins_over_a_parking_server(self, monkeypatch):
        monkeypatch.setenv(preemption.PREEMPT_ENV, "0")
        preemption.note_server_parking(True)
        assert preemption.preemption_enabled() is False
        assert preemption.preemption_wanted() is False

    def test_on_and_studio_mode_are_armed_without_the_server(self, monkeypatch):
        monkeypatch.setenv(preemption.PREEMPT_ENV, "1")
        assert preemption.preemption_enabled() is True
        monkeypatch.delenv(preemption.PREEMPT_ENV)
        monkeypatch.setenv(preemption.PREEMPT_MODE_ENV, "studio")
        assert preemption.preemption_enabled() is True


class TestTheServerHasToSayItParks:
    @staticmethod
    def _backend(props):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._query_server_props = lambda: props
        return backend

    @pytest.mark.parametrize(
        ("props", "auto", "on"),
        [
            ({"preempt": {"enabled": True}}, True, True),
            ({"preempt": {"enabled": False, "recurrent": True}}, False, False),
            ({}, False, True),
            (None, False, True),
        ],
    )
    def test_the_report_decides_auto(self, monkeypatch, props, auto, on):
        assert self._backend(props)._server_confirms_parking() is auto
        monkeypatch.setenv(preemption.PREEMPT_ENV, "1")
        assert self._backend(props)._server_confirms_parking() is on


class TestTheDefaultBudgetFitsTheHost:
    @staticmethod
    def _host(monkeypatch, available, total):
        monkeypatch.setattr(llama_mod, "_available_host_memory_mib", lambda: available)
        monkeypatch.setattr(llama_mod, "_total_host_memory_mib", lambda: total)

    def test_a_large_host_gets_what_the_parks_need(self, monkeypatch):
        self._host(monkeypatch, available = 512 * 1024, total = 1024 * 1024)
        assert llama_mod._host_bounded_parking_mib(10_000) == 10_000

    def test_a_small_host_is_capped_and_a_tiny_one_parks_nothing(self, monkeypatch):
        self._host(monkeypatch, available = 16 * 1024, total = 16 * 1024)
        assert llama_mod._host_bounded_parking_mib(10_000) == 4096
        self._host(monkeypatch, available = 3 * 1024, total = 8 * 1024)
        assert llama_mod._host_bounded_parking_mib(10_000) == 0
        self._host(monkeypatch, available = None, total = None)
        assert llama_mod._host_bounded_parking_mib(10_000) == 0

    def test_no_room_means_no_flag(self, monkeypatch):
        self._host(monkeypatch, available = 3 * 1024, total = 8 * 1024)
        budget = llama_mod._server_owned_parking_budget_mib(
            12 * _GIB,
            args = ["llama-server", "--kv-unified"],
            env = {},
            server_supports = True,
            kv_unified = True,
            parallel = 4,
            bound_to_host = True,
        )
        assert budget is None

    def test_exact_keeps_the_full_need(self, monkeypatch):
        self._host(monkeypatch, available = 16 * 1024, total = 16 * 1024)
        budget = llama_mod._server_owned_parking_budget_mib(
            12 * _GIB,
            args = ["llama-server", "--kv-unified"],
            env = {},
            server_supports = True,
            kv_unified = True,
            parallel = 4,
        )
        assert budget == llama_mod._exact_parking_need_mib(12 * _GIB, parallel = 4)

    def test_a_named_budget_is_kept(self, monkeypatch):
        self._host(monkeypatch, available = 3 * 1024, total = 8 * 1024)
        assert (
            llama_mod._server_owned_parking_budget_mib(
                12 * _GIB,
                args = ["llama-server", "--kv-unified", "--preempt-ram", "0"],
                env = {},
                server_supports = True,
                kv_unified = True,
                bound_to_host = True,
            )
            is None
        )


class TestADefaultLaunch:
    def test_a_stock_server_is_launched_as_before(self):
        cmd, env, budget, parks, overridden = _Launch._plan(
            _Launch._ARGV, {}, supported = False, pool = _Launch._POOL, parallel = 4
        )
        assert budget is None and parks is False
        assert cmd == _Launch._ARGV
        assert env == {}, "no LLAMA_ARG_PREEMPT_RAM written for a server that cannot park"

    def test_an_unread_probe_still_tells_the_child_not_to_park(self):
        # A --help that timed out may be a swap build from before the zero default, which parks
        # on an implicit budget Studio never launched and so never excuses.
        env = {}
        cmd = list(_Launch._ARGV)
        assert (
            llama_mod._stand_down_child_parking(
                env, cmd, server_supports = False, probe_inconclusive = True
            )
            == []
        )
        assert env == {"LLAMA_ARG_PREEMPT_RAM": "0"} and cmd == _Launch._ARGV

    def test_a_parking_server_is_asked_to_park(self, monkeypatch):
        monkeypatch.setattr(llama_mod, "_available_host_memory_mib", lambda: 512 * 1024)
        monkeypatch.setattr(llama_mod, "_total_host_memory_mib", lambda: 1024 * 1024)
        cmd, env, budget, parks, overridden = _Launch._plan(
            _Launch._ARGV, {}, pool = _Launch._POOL, parallel = 4
        )
        assert budget is not None and cmd[-2:] == ["--preempt-ram", str(budget)]
        assert parks is True and env == {}

    def test_the_launch_confirms_before_auto_arms(self):
        import inspect

        source = inspect.getsource(LlamaCppBackend.load_model)
        confirm = source.index("self._server_confirms_parking()")
        assert source.index("_preemption.note_server_parking(self._server_preempts_kv") > confirm
        assert "bound_to_host = not _exact_wanted" in source


class TestAnExactFallbackKeepsTheHostBound:
    """Exact auto that falls back for a short host must not leave its full budget on the line."""

    def test_the_budget_is_bounded_in_place(self, monkeypatch):
        monkeypatch.setattr(llama_mod, "_available_host_memory_mib", lambda: 16 * 1024)
        monkeypatch.setattr(llama_mod, "_total_host_memory_mib", lambda: 16 * 1024)
        cmd = ["llama-server", "--kv-unified", "--preempt-ram", "20000"]
        assert llama_mod._rebound_named_parking_budget(cmd, 20000) == 4096
        assert cmd[-2:] == ["--preempt-ram", "4096"]

    def test_no_room_drops_the_flag(self, monkeypatch):
        monkeypatch.setattr(llama_mod, "_available_host_memory_mib", lambda: 3 * 1024)
        monkeypatch.setattr(llama_mod, "_total_host_memory_mib", lambda: 8 * 1024)
        cmd = ["llama-server", "--preempt-ram", "20000", "--kv-unified"]
        assert llama_mod._rebound_named_parking_budget(cmd, 20000) is None
        assert cmd == ["llama-server", "--kv-unified"]

    def test_the_load_rebounds_on_the_auto_fallback_and_reprices_a_dropped_drafter(self):
        import inspect

        source = inspect.getsource(LlamaCppBackend.load_model)
        fallback = source.index(
            "if _exact_setting == _exact.EXACT_AUTO:\n                                    _exact_wanted = False"
        )
        assert "_rebound_named_parking_budget(" in source[fallback : fallback + 600]
        reprice = source.index(
            "_writes = _exact_parking_need_mib(_exact_kv_bytes, parallel = n_parallel)"
        )
        assert reprice < source.index(
            '_exact_host_short = getattr(self, "_exact_host_short", None)'
        )


class TestAnUnconfirmedParkIsStillReadAsAPark:
    """A swap build without the ``/props`` report still parks when the launch asked it to."""

    @staticmethod
    def _backend(
        *,
        launched,
        confirmed,
        notices = False,
    ):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._server_launched_parking = launched
        backend._server_preempts_kv = confirmed
        backend._kv_cache_unified = True
        backend._server_park_notices = notices
        return backend

    def test_auto_keeps_fair_share_but_reads_the_silence_as_a_park(self):
        backend = self._backend(launched = True, confirmed = False)
        assert backend.server_preempts_kv is False, "admission stays fair share"
        assert backend.server_may_park is True, "but the child does park"
        from routes.inference import _raw_park_grace

        assert _raw_park_grace(backend) is not None

    def test_nothing_launched_means_nothing_parks(self):
        backend = self._backend(launched = False, confirmed = False)
        assert backend.server_may_park is False
        from routes.inference import _raw_park_grace

        assert _raw_park_grace(backend) is None
        assert _raw_park_grace(backend, streaming = False) is None

    def test_a_non_streaming_response_reads_the_aggregate_even_beside_notices(self, monkeypatch):
        import core.inference.llama_stats as llama_stats
        from routes.inference import _raw_park_grace

        backend = self._backend(launched = True, confirmed = True, notices = True)
        backend._port = 1
        backend._api_key = None
        monkeypatch.setattr(
            llama_stats, "scrape_llama_metrics", lambda *a, **k: {"requests_preempted": 1.0}
        )
        assert _raw_park_grace(backend)() is False, "a stream hears its own notice instead"
        assert _raw_park_grace(backend, streaming = False)() is True


class TestEachServerAnswersForItself:
    def test_the_routed_backend_decides_and_a_helper_cannot_flip_it(self):
        main = LlamaCppBackend.__new__(LlamaCppBackend)
        helper = LlamaCppBackend.__new__(LlamaCppBackend)
        routed = [main]
        preemption.set_routed_backend(lambda: routed[0])
        try:
            preemption.note_server_parking(True, main)
            preemption.note_server_parking(False, helper)
            assert preemption.preemption_enabled() is True, "the helper's load is not the main's"
            routed[0] = helper
            assert preemption.preemption_enabled() is False
            routed[0] = object()
            preemption.note_server_parking(True)
            assert preemption.preemption_enabled() is True, "an unknown route takes the fallback"
        finally:
            import routes.inference as routes_inference
            preemption.set_routed_backend(routes_inference.get_llama_cpp_backend)

    def test_every_launch_writes_its_own_entry(self):
        import inspect

        source = inspect.getsource(LlamaCppBackend.load_model)
        assert "note_server_parking(False)" not in source
        assert "note_server_parking(self._server_preempts_kv and kv_cache_unified, self)" in source


class TestExactParksAreCountedAcrossModels:
    """A park budget is a cap allocated only on a park, so each exact model has to be measured
    against the RAM the others did not already certify, or two pass on the same free reading."""

    @staticmethod
    def _backend(state, writes):
        backend = object.__new__(LlamaCppBackend)
        backend._exact_concurrency = state
        backend._exact_parking_writes = writes
        return backend

    def test_an_exact_neighbour_holds_its_parks(self, monkeypatch):
        from core.inference import llama_exact as exact

        monkeypatch.setattr(llama_mod, "_available_host_memory_mib", lambda: 10_000)
        me = self._backend(exact.EXACT_STATE_OFF, 8_000)
        neighbour = self._backend(exact.EXACT_STATE_ON, 8_000)
        stock = self._backend(exact.EXACT_STATE_OFF, 4_000)
        monkeypatch.setattr(llama_mod, "_serving_backends_snapshot", lambda: [me, neighbour, stock])
        monkeypatch.setattr(llama_mod, "_parked_host_mib", lambda backend: 0)
        assert llama_mod._exact_host_free_mib(me) == 2_000
        assert llama_mod._exact_host_shortfall_after_load(8_000, 2_000) == (8_000, 2_000)
        # Alone, the same model fits.
        monkeypatch.setattr(llama_mod, "_serving_backends_snapshot", lambda: [me, stock])
        assert llama_mod._exact_host_free_mib(me) == 10_000

    def test_what_a_neighbour_already_parked_is_not_counted_twice(self, monkeypatch):
        # 6 GiB of the neighbour's 8 GiB is parked, so the free reading already lacks it: only
        # the 2 GiB still to come is held back from this model.
        from core.inference import llama_exact as exact

        monkeypatch.setattr(llama_mod, "_available_host_memory_mib", lambda: 10_000)
        me = self._backend(exact.EXACT_STATE_OFF, 8_000)
        neighbour = self._backend(exact.EXACT_STATE_ON, 8_000)
        monkeypatch.setattr(llama_mod, "_serving_backends_snapshot", lambda: [me, neighbour])
        monkeypatch.setattr(
            llama_mod, "_parked_host_mib", lambda backend: 6_000 if backend is neighbour else 0
        )
        assert llama_mod._exact_host_free_mib(me) == 8_000

    def test_the_parked_reading_comes_off_the_servers_metrics(self, monkeypatch):
        import core.inference.llama_stats as stats

        backend = self._backend("on", 8_000)
        backend._port = 1
        monkeypatch.setattr(
            stats,
            "scrape_llama_metrics",
            lambda *a, **k: {"preempt_ram_bytes": 3 * 1024 * 1024 * 1024},
        )
        assert llama_mod._parked_host_mib(backend) == 3 * 1024
        monkeypatch.setattr(stats, "scrape_llama_metrics", lambda *a, **k: None)
        assert llama_mod._parked_host_mib(backend) == 0

    def test_a_plain_parking_neighbour_holds_its_launched_budget(self, monkeypatch):
        # Default-on parking hands an ordinary load a host-bounded budget it may still fill.
        from core.inference import llama_exact as exact

        monkeypatch.setattr(llama_mod, "_available_host_memory_mib", lambda: 10_000)
        me = self._backend(exact.EXACT_STATE_OFF, 8_000)
        plain = self._backend(exact.EXACT_STATE_OFF, None)
        plain._server_launched_parking = True
        plain._launched_parking_mib = 3_000
        idle = self._backend(exact.EXACT_STATE_OFF, None)
        idle._server_launched_parking = False
        idle._launched_parking_mib = 5_000
        monkeypatch.setattr(llama_mod, "_serving_backends_snapshot", lambda: [me, plain, idle])
        monkeypatch.setattr(llama_mod, "_parked_host_mib", lambda backend: 0)
        assert llama_mod._exact_host_free_mib(me) == 7_000

    def test_the_launch_records_the_budget_it_spawned_with(self):
        import inspect
        source = " ".join(inspect.getsource(LlamaCppBackend.load_model).split())
        assert (
            "self._launched_parking_mib = ( _named_preempt_ram_mib(_last_spawn_cmd or cmd, env) "
            "if self._server_launched_parking else None )"
        ) in source

    def test_an_unlimited_neighbour_leaves_nothing_certifiable(self, monkeypatch):
        from core.inference import llama_exact as exact

        monkeypatch.setattr(llama_mod, "_available_host_memory_mib", lambda: 10_000)
        me = self._backend(exact.EXACT_STATE_OFF, 8_000)
        unlimited = self._backend(exact.EXACT_STATE_OFF, None)
        unlimited._server_launched_parking = True
        unlimited._launched_parking_mib = -1
        monkeypatch.setattr(llama_mod, "_serving_backends_snapshot", lambda: [me, unlimited])
        monkeypatch.setattr(llama_mod, "_parked_host_mib", lambda backend: 0)
        assert llama_mod._exact_host_free_mib(me) == 0

    def test_the_host_readings_are_the_containers(self, monkeypatch):
        # A cgroup limit below the host's RAM: parking may only plan against what it can charge.
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 3_000)
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_host_memory_capacity_mib", staticmethod(lambda: 8_000)
        )
        assert llama_mod._available_host_memory_mib() == 3_000
        assert llama_mod._total_host_memory_mib() == 8_000

    def test_a_helper_backend_that_parks_is_counted_though_it_does_not_serve(self, monkeypatch):
        # The dataset and Hub assistants build their own backend and never register as serving.
        from core.inference import llama_exact as exact

        monkeypatch.setattr(llama_mod, "_available_host_memory_mib", lambda: 10_000)
        me = self._backend(exact.EXACT_STATE_OFF, 8_000)
        helper = self._backend(exact.EXACT_STATE_OFF, None)
        helper._server_launched_parking = True
        helper._launched_parking_mib = 4_000
        monkeypatch.setattr(llama_mod, "_serving_backends_snapshot", lambda: [me])
        monkeypatch.setattr(llama_mod, "_parked_host_mib", lambda backend: 0)
        monkeypatch.setattr(llama_mod, "_parking_backends", __import__("weakref").WeakSet())
        llama_mod._note_parking_backend(helper)
        assert llama_mod._exact_host_free_mib(me) == 6_000

    def test_a_launch_that_parks_is_noted(self):
        import inspect

        source = " ".join(inspect.getsource(LlamaCppBackend.load_model).split())
        assert "if self._server_launched_parking: _note_parking_backend(self)" in source

    def test_every_exact_host_check_asks_for_this_backends_share(self):
        import inspect
        source = inspect.getsource(LlamaCppBackend.load_model)
        assert source.count("_exact_host_free_mib(self)") == 4
