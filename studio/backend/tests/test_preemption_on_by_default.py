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
