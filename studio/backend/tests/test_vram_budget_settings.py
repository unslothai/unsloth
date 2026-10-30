# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unset VRAM budget equals the old 0.97; a malformed value such as NaN must never reach the fit."""

from __future__ import annotations

import pytest

import utils.vram_budget_settings as vb


def _shared_setup_1():
    import inspect

    import core.inference.llama_cpp as lc

    compact = "".join(inspect.getsource(lc.LlamaCppBackend.load_model).split())
    return compact


def _shared_setup_2():
    import inspect

    import core.inference.llama_cpp as lc

    source = inspect.getsource(lc.LlamaCppBackend.load_model)
    return source


@pytest.fixture(autouse = True)
def _isolate(monkeypatch):
    """No stored value and no environment, so each test states its own inputs."""
    monkeypatch.delenv(vb.VRAM_FRACTION_ENV_VAR, raising = False)
    monkeypatch.setattr(vb, "_cached_setting", lambda _key: None)


class TestCoerceFraction:
    @pytest.mark.parametrize(
        "value", [0.10, 0.30, 0.50, 0.80, 0.9, 0.97, 1.0, "0.25", "0.85", "1.00"]
    )
    def test_accepts_in_range(self, value):
        assert vb.coerce_fraction(value) == pytest.approx(float(value))

    @pytest.mark.parametrize(
        "value",
        [
            0.0,
            0.05,
            0.099,
            1.01,
            -1.0,
            2.0,
            "nan",
            "NaN",
            float("nan"),
            "inf",
            float("inf"),
            float("-inf"),
            "",
            "   ",
            "abc",
            "0.9x",
            None,
            [],
            {},
        ],
    )
    def test_rejects_unusable(self, value):
        assert vb.coerce_fraction(value) is None

    def test_rejects_bool(self):
        # bool subclasses int, so True would coerce to a legitimate-looking 1.0.
        assert vb.coerce_fraction(True) is None
        assert vb.coerce_fraction(False) is None

    def test_boundaries_are_inclusive(self):
        assert vb.coerce_fraction(vb.VRAM_FRACTION_MIN) == vb.VRAM_FRACTION_MIN
        assert vb.coerce_fraction(vb.VRAM_FRACTION_MAX) == vb.VRAM_FRACTION_MAX


class TestPrecedence:
    def test_unset_is_the_historical_default(self):
        assert vb.get_vram_budget_fraction() == vb.VRAM_FRACTION_DEFAULT

    def test_default_matches_the_constant_it_replaced(self):
        # A no-op when no budget is set, so this must track _CTX_FIT_VRAM_FRACTION.
        from core.inference.llama_cpp import _CTX_FIT_VRAM_FRACTION
        assert vb.VRAM_FRACTION_DEFAULT == _CTX_FIT_VRAM_FRACTION

    def test_env_beats_default(self, monkeypatch):
        monkeypatch.setenv(vb.VRAM_FRACTION_ENV_VAR, "0.93")
        assert vb.get_vram_budget_fraction() == pytest.approx(0.93)

    def test_bad_env_falls_back_without_raising(self, monkeypatch):
        for bad in ("nan", "12", "-0.5", "abc", ""):
            monkeypatch.setenv(vb.VRAM_FRACTION_ENV_VAR, bad)
            assert vb.get_vram_budget_fraction() == vb.VRAM_FRACTION_DEFAULT

    def test_stored_beats_env(self, monkeypatch):
        monkeypatch.setenv(vb.VRAM_FRACTION_ENV_VAR, "0.90")
        monkeypatch.setattr(vb, "_cached_setting", lambda _key: 0.99)
        assert vb.get_vram_budget_fraction() == pytest.approx(0.99)

    def test_corrupt_stored_value_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv(vb.VRAM_FRACTION_ENV_VAR, "0.90")
        monkeypatch.setattr(vb, "_cached_setting", lambda _key: "garbage")
        assert vb.get_vram_budget_fraction() == pytest.approx(0.90)

    def test_out_of_range_stored_value_is_ignored(self, monkeypatch):
        monkeypatch.setattr(vb, "_cached_setting", lambda _key: 4.0)
        assert vb.get_vram_budget_fraction() == vb.VRAM_FRACTION_DEFAULT

    def test_unreadable_db_does_not_fail_a_load(self, monkeypatch):
        def _boom(_key):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(vb, "_cached_setting", _boom)
        with pytest.raises(RuntimeError):
            vb.get_vram_budget_fraction()
        from core.inference.llama_cpp import _active_vram_fraction, _CTX_FIT_VRAM_FRACTION

        assert _active_vram_fraction() == _CTX_FIT_VRAM_FRACTION


class TestGrid:
    def test_a_tenth_of_a_percent_is_a_legal_budget(self):
        # The slider steps in tenths, so the fraction must round-trip on that grid.
        assert vb.coerce_fraction(0.975) == 0.975
        assert vb.coerce_fraction("0.975") == 0.975

    def test_off_grid_values_are_quantised_rather_than_refused(self):
        assert vb.coerce_fraction(0.9749999999) == 0.975
        assert vb.coerce_fraction(0.9754) == 0.975
        assert vb.coerce_fraction(0.8) == 0.8
        assert vb.coerce_fraction(1.0) == 1.0


class TestState:
    def test_reports_inherited_values_as_not_stored(self, monkeypatch):
        monkeypatch.setenv(vb.VRAM_FRACTION_ENV_VAR, "0.91")
        fraction, is_stored = vb.get_vram_budget_state()
        assert fraction == pytest.approx(0.91)
        assert is_stored is False

    def test_reports_a_saved_value_as_stored(self, monkeypatch):
        monkeypatch.setattr(vb, "_cached_setting", lambda _key: 0.99)
        fraction, is_stored = vb.get_vram_budget_state()
        assert fraction == pytest.approx(0.99)
        assert is_stored is True


class TestWrite:
    def test_rejects_out_of_range(self, monkeypatch):
        monkeypatch.setattr(vb, "_invalidate", lambda _key: None)
        for bad in (0.0, 0.05, -0.25, 1.5, float("nan"), "abc"):
            with pytest.raises(ValueError):
                vb.set_vram_budget_fraction(bad)

    def test_stores_a_valid_value(self, monkeypatch):
        written: dict = {}
        monkeypatch.setattr(vb, "_invalidate", lambda _key: None)
        monkeypatch.setitem(
            __import__("sys").modules,
            "storage.studio_db",
            type("_M", (), {"upsert_app_settings": staticmethod(written.update)}),
        )
        monkeypatch.setattr(
            vb, "_cached_setting", lambda _key: written.get(vb.VRAM_BUDGET_SETTING_KEY)
        )
        assert vb.set_vram_budget_fraction(0.99) == pytest.approx(0.99)
        assert written == {vb.VRAM_BUDGET_SETTING_KEY: 0.99}

    def test_none_clears_back_to_the_default(self, monkeypatch):
        written: dict = {}
        monkeypatch.setattr(vb, "_invalidate", lambda _key: None)
        monkeypatch.setitem(
            __import__("sys").modules,
            "storage.studio_db",
            type("_M", (), {"upsert_app_settings": staticmethod(written.update)}),
        )
        assert vb.set_vram_budget_fraction(None) == vb.VRAM_FRACTION_DEFAULT
        assert written == {vb.VRAM_BUDGET_SETTING_KEY: None}


class TestActiveFractionWiring:
    def test_llama_cpp_uses_the_setting(self, monkeypatch):
        import core.inference.llama_cpp as lc
        monkeypatch.setenv(vb.VRAM_FRACTION_ENV_VAR, "0.88")
        assert lc._active_vram_fraction() == pytest.approx(0.88)

    def test_llama_cpp_defaults_to_the_old_constant(self):
        import core.inference.llama_cpp as lc
        assert lc._active_vram_fraction() == lc._CTX_FIT_VRAM_FRACTION


class TestLaunchedMarker:
    """_vram_fraction_launched must track the running child, so the reload prompt stays accurate."""

    @staticmethod
    def _resident_backend(monkeypatch, *, launched: float, active: float):
        import core.inference.llama_cpp as lc

        backend = lc.LlamaCppBackend()
        backend._process = object()
        backend._healthy = True
        backend._vram_fraction_launched = launched
        monkeypatch.setattr(lc, "_active_vram_fraction", lambda: active)
        monkeypatch.setattr(backend, "adopt_load_intent_if_matched", lambda _intent, **_kw: True)
        return backend, lc

    def test_duplicate_load_leaves_the_running_child_marker(self, monkeypatch):
        backend, lc = self._resident_backend(monkeypatch, launched = 0.97, active = 0.85)
        backend._audio_probed = True

        assert backend.load_model(lc.GgufLoadIntent(model_identifier = "owner/repo"))
        assert backend._vram_fraction_launched == pytest.approx(0.97)

    @staticmethod
    def _adoptable(monkeypatch, *, launched):
        """A backend whose every other adopt predicate already matches."""
        import core.inference.llama_cpp as lc

        backend = lc.LlamaCppBackend()
        backend._process = object()
        backend._healthy = True
        backend._vram_fraction_launched = launched
        monkeypatch.setattr(backend, "matches_load_source", lambda _i: True)
        monkeypatch.setattr(backend, "_runtime_matches_intent", lambda _i, _e: True)
        monkeypatch.setattr(backend, "_record_matching_gpu_request", lambda *_a, **_k: None)
        return backend, lc

    def test_adopt_is_refused_when_the_budget_changed(self, monkeypatch):
        backend, lc = self._adoptable(monkeypatch, launched = 0.97)
        monkeypatch.setattr(lc, "_active_vram_fraction", lambda: 0.85)

        assert not backend.adopt_load_intent_if_matched(
            lc.GgufLoadIntent(model_identifier = "owner/repo")
        )

    def test_adopt_is_allowed_when_the_budget_is_unchanged(self, monkeypatch):
        backend, lc = self._adoptable(monkeypatch, launched = 0.97)
        monkeypatch.setattr(lc, "_active_vram_fraction", lambda: 0.97)

        assert backend.adopt_load_intent_if_matched(
            lc.GgufLoadIntent(model_identifier = "owner/repo")
        )

    def test_adopt_is_allowed_when_placement_never_used_the_budget(self, monkeypatch):
        backend, lc = self._adoptable(monkeypatch, launched = None)
        monkeypatch.setattr(lc, "_active_vram_fraction", lambda: 0.85)

        assert backend.adopt_load_intent_if_matched(
            lc.GgufLoadIntent(model_identifier = "owner/repo")
        )

    def test_marker_is_committed_with_the_rest_of_the_launch_state(self):
        compact = _shared_setup_1()
        assert (
            "self._vram_fraction_launched=_budget_priced_placement()"
            "self._vram_fraction_pending=Noneself._requested_n_batch" in compact
        )

    def test_marker_is_none_when_placement_had_no_devices(self):
        compact = _shared_setup_1()
        assert "self._vram_fraction_pending=_budget_priced_placement()" in compact
        assert "self._vram_fraction_launched=_budget_priced_placement()" in compact


class TestRouteContract:
    """The two places the HTTP layer can answer wrongly on well-formed input."""

    @staticmethod
    def _settings_module():
        import routes.settings as rs
        return rs

    def test_payload_rejects_a_boolean_fraction(self):
        # bool subclasses int; pydantic would coerce True to 1.0 before the util's guard.
        import pydantic
        import pytest as _pytest

        rs = self._settings_module()
        with _pytest.raises(pydantic.ValidationError):
            rs.VramBudgetPayload.model_validate({"fraction": True})
        with _pytest.raises(pydantic.ValidationError):
            rs.VramBudgetPayload.model_validate_json('{"fraction": true}')
        assert rs.VramBudgetPayload.model_validate({"fraction": 0.9}).fraction == 0.9
        assert rs.VramBudgetPayload.model_validate({"fraction": None}).fraction is None

    def test_reload_required_answers_from_a_load_that_has_not_spawned(self, monkeypatch):
        rs = self._settings_module()

        class _Backend:
            is_active = False
            _vram_fraction_pending = 0.97
            _vram_fraction_launched = None

        monkeypatch.setattr(rs, "get_llama_cpp_backend", lambda: _Backend(), raising = False)
        import routes.inference as ri

        monkeypatch.setattr(ri, "get_llama_cpp_backend", lambda: _Backend(), raising = False)

        assert rs._vram_budget_reload_required(0.85)
        assert not rs._vram_budget_reload_required(0.97)

    def test_reload_not_required_when_no_load_is_in_flight(self, monkeypatch):
        rs = self._settings_module()

        class _Backend:
            is_active = False
            _vram_fraction_pending = None
            _vram_fraction_launched = 0.97

        import routes.inference as ri

        monkeypatch.setattr(ri, "get_llama_cpp_backend", lambda: _Backend(), raising = False)

        assert not rs._vram_budget_reload_required(0.85)


class TestDiffusionPath:
    def test_the_diffusion_launch_clears_the_marker(self):
        source = _shared_setup_2()
        diffusion = source[source.index("if self._is_diffusion:") :]
        diffusion = diffusion[: diffusion.index("_start_diffusion_server")]
        compact = "".join(diffusion.split())
        assert "self._vram_fraction_launched=None" in compact


class TestLaunchFinalization:
    """The marker and the pending value must describe the child that is running."""

    @staticmethod
    def _load_model_source():
        import inspect

        import core.inference.llama_cpp as lc
        return "".join(inspect.getsource(lc.LlamaCppBackend.load_model).split())

    def test_the_fraction_is_resolved_under_the_load_lock(self):
        compact = self._load_model_source()
        lock_at = compact.index("withself._serial_load_scope():")
        resolve_at = compact.index("_vram_frac=_active_vram_fraction()")
        dedupe_at = compact.index("ifself.adopt_load_intent_if_matched(intent)")
        assert lock_at < resolve_at < dedupe_at

    def test_a_healthy_spawn_keeps_the_pending_value_until_the_commit(self):
        compact = self._load_model_source()
        assert "ifnothealthy:self._vram_fraction_pending=None" not in compact
        assert (
            "self._vram_fraction_launched=_budget_priced_placement()self._vram_fraction_pending=None"
            in compact
        )

    def test_a_cpu_fallback_child_is_not_stamped_with_a_budget(self):
        source = _shared_setup_2()
        helper = source[source.index("def _budget_priced_placement()") :]
        # Bounded by a statement, not a comment, since comments here get rewritten.
        helper = helper[: helper.index("self._vram_fraction_pending")]
        compact = "".join(helper.split())
        assert "ifintent.cpu_fallback:returnNone" in compact
        assert compact.index("ifintent.cpu_fallback:returnNone") < compact.index("ifgpus:")


class TestPreLaunchWindow:
    @staticmethod
    def _compact():
        import inspect

        import core.inference.llama_cpp as lc
        return "".join(inspect.getsource(lc.LlamaCppBackend.load_model).split())

    def test_the_pending_value_is_armed_before_the_duplicate_check(self):
        compact = self._compact()
        armed = compact.index("self._vram_fraction_pending=_vram_frac")
        dedupe = compact.index("ifself.adopt_load_intent_if_matched(intent)")
        assert armed < dedupe

    def test_the_pending_value_is_armed_before_the_download(self):
        compact = self._compact()
        armed = compact.index("self._vram_fraction_pending=_vram_frac\n".strip())
        cancel = compact.index("self._cancel_event.clear()")
        spawn = compact.index("self._vram_fraction_pending=_budget_priced_placement()")
        assert armed < cancel < spawn

    def test_a_terminal_failure_releases_the_pending_value(self):
        source = _shared_setup_2()
        funnel = source[source.index("def _raise_terminal_load_failure") :]
        funnel = funnel[: funnel.index("def _try_auto_vulkan_cpu_fallback")]
        compact = "".join(funnel.split())
        assert "self._vram_fraction_pending=None" in compact
        assert compact.index("self._vram_fraction_pending=None") < compact.index(
            "raiseRuntimeError(detail)"
        )


class TestPendingOwnership:
    def test_the_pending_value_is_released_with_the_load_lock(self):
        import ast
        import inspect
        import textwrap

        import core.inference.llama_cpp as lc

        scope = ast.parse(
            textwrap.dedent(inspect.getsource(lc.LlamaCppBackend._serial_load_scope))
        ).body[0]
        held = next((n for n in scope.body if isinstance(n, (ast.With, ast.AsyncWith))), None)
        assert held is not None, "the scope no longer takes the load lock in a with"
        assert ast.unparse(held.items[0].context_expr) == "self._serial_load_lock"
        guarded = next((n for n in held.body if isinstance(n, ast.Try)), None)
        assert guarded is not None, "the yield is no longer wrapped in try/finally"
        assert any(
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Yield)
            for node in guarded.body
        )
        cleared = {
            ast.unparse(target)
            for node in guarded.finalbody
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
            for target in node.targets
        }
        assert "self._vram_fraction_pending" in cleared
        load = "".join(inspect.getsource(lc.LlamaCppBackend.load_model).split())
        assert "withself._serial_load_scope():" in load

    def test_a_pre_launch_exit_leaves_no_pending_value(self, monkeypatch):
        import core.inference.llama_cpp as lc

        backend = lc.LlamaCppBackend()
        backend._audio_probed = True
        monkeypatch.setattr(lc, "_active_vram_fraction", lambda: 0.9)
        monkeypatch.setattr(backend, "adopt_load_intent_if_matched", lambda _intent: False)
        monkeypatch.setattr(
            backend,
            "_find_llama_server_binary",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no binary")),
        )
        with pytest.raises(Exception):
            backend.load_model(lc.GgufLoadIntent(model_identifier = "owner/repo"))
        assert backend._vram_fraction_pending is None


class TestFloorReserve:
    """100% still leaves a card the margin llama.cpp keeps for its own fitter."""

    @staticmethod
    def _usable(free, total, frac):
        import core.inference.llama_cpp as lc
        return lc._vram_usable_mib(free, total, frac)

    @pytest.mark.parametrize("total", [4_096, 8_192, 16_384, 24_576, 81_920])
    def test_raising_the_budget_never_hands_back_less(self, total):
        usable = [self._usable(total, total, frac) for frac in (0.80, 0.90, 0.97, 0.971, 0.99, 1.0)]
        assert usable == sorted(usable), usable

    def test_the_floor_never_exceeds_the_default_reserve(self):
        import core.inference.llama_cpp as lc
        for total in (4_096, 8_192, 16_384, 24_576, 81_920):
            floor = lc._vram_reserve_floor_mib(total)
            assert floor <= (1.0 - lc._CTX_FIT_VRAM_FRACTION) * total
            assert floor <= lc._VRAM_FLOOR_RESERVE_MIB

    def test_a_card_with_no_reported_total_still_keeps_a_margin(self):
        import core.inference.llama_cpp as lc
        assert self._usable(24_576, 0, 1.0) == pytest.approx(24_576 - lc._VRAM_FLOOR_RESERVE_MIB)
        assert self._usable(24_576, 0, lc._CTX_FIT_VRAM_FRACTION) == pytest.approx(
            24_576 * lc._CTX_FIT_VRAM_FRACTION
        )

    def test_full_budget_still_leaves_the_floor(self):
        import core.inference.llama_cpp as lc
        assert self._usable(24_576, 24_576, 1.0) == pytest.approx(
            24_576 - lc._VRAM_FLOOR_RESERVE_MIB
        )
        assert self._usable(8_192, 8_192, 1.0) == pytest.approx(8_192 * lc._CTX_FIT_VRAM_FRACTION)

    @pytest.mark.parametrize("total", [4_096, 8_192, 16_384, 24_576, 81_920])
    def test_the_default_reserve_is_unchanged_on_every_card_size(self, total):
        import core.inference.llama_cpp as lc
        assert self._usable(total, total, lc._CTX_FIT_VRAM_FRACTION) == pytest.approx(
            total - (1.0 - lc._CTX_FIT_VRAM_FRACTION) * total
        )

    def test_the_floor_only_binds_where_the_percentage_reserves_less(self):
        import core.inference.llama_cpp as lc
        assert self._usable(81_920, 81_920, 0.99) == pytest.approx(81_920 * 0.99)
        assert self._usable(24_576, 24_576, 0.99) == pytest.approx(
            24_576 - lc._VRAM_FLOOR_RESERVE_MIB
        )

    def test_an_absolute_pool_budget_is_not_charged_twice(self):
        # Tensor-parallel paths pass a pool budget that already paid its reserve: no flooring.
        import core.inference.llama_cpp as lc
        assert lc._vram_usable_mib(12_000, 0, 1.0, pooled = True) == pytest.approx(12_000)
        assert lc._vram_usable_mib(12_000, None, 1.0, pooled = True) == pytest.approx(12_000)

    def test_every_pooled_caller_says_so(self):
        # Six call sites hand _fit_context_to_vram an absolute pool budget (the sixth is
        # the drafter probe's one-slot fit, #8872); each has to be marked, since the flag
        # is what keeps them from double-paying.
        compact = _shared_setup_1()
        assert compact.count("budget_frac=1.0,pooled=True,total_mib=None,") == 6


class TestRetriesAndDedup:
    def test_a_nonterminal_retry_keeps_the_pending_value(self):
        compact = _shared_setup_1()
        assert "ifnothealthy:self._vram_fraction_pending=None" not in compact

    def test_the_duplicate_check_uses_the_captured_fraction(self):
        compact = _shared_setup_1()
        assert "self._vram_fraction_pending=_vram_frac" in compact
        assert "adopt_load_intent_if_matched(intent)" in compact

    def test_the_route_fast_path_still_resolves_its_own(self, monkeypatch):
        import core.inference.llama_cpp as lc

        backend = lc.LlamaCppBackend()
        backend._process = object()
        backend._healthy = True
        backend._vram_fraction_launched = 0.97
        monkeypatch.setattr(backend, "matches_load_source", lambda _i: True)
        monkeypatch.setattr(backend, "_runtime_matches_intent", lambda _i, _e: True)
        monkeypatch.setattr(backend, "_record_matching_gpu_request", lambda *_a, **_k: None)
        monkeypatch.setattr(lc, "_active_vram_fraction", lambda: 0.85)
        intent = lc.GgufLoadIntent(model_identifier = "owner/repo")

        assert not backend.adopt_load_intent_if_matched(intent)
        backend._vram_fraction_pending = 0.97
        assert backend.adopt_load_intent_if_matched(intent)


class TestFitTarget:
    """The budget must reach llama.cpp's own fitter via --fit-target, not stop at Unsloth's planner."""

    _CAPS = {"supports_fit_ctx": True, "supports_fit_target": True, "supports_kv_unified": True}

    def _flags(self, *, auto_fit, delta):
        import core.inference.llama_cpp as lc
        return lc.LlamaCppBackend._ctx_integrity_flags(
            1,
            True,
            auto_fit,
            0,
            0,
            self._CAPS,
            fit_target_delta_mib = delta,
        )

    def test_the_default_budget_emits_exactly_what_it_did_before(self):
        assert self._flags(auto_fit = True, delta = 0.0)[-2:] == ["--fit-target", "512"]
        assert "--fit-target" not in self._flags(auto_fit = False, delta = 0.0)

    def test_a_lowered_budget_reaches_the_fitter_on_both_paths(self):
        # Raised from each path's own starting margin, not from zero (legacy path keeps 1024).
        assert self._flags(auto_fit = True, delta = 4096.0)[-2:] == ["--fit-target", "4608"]
        assert self._flags(auto_fit = False, delta = 4096.0)[-2:] == ["--fit-target", "5120"]

    def test_the_margin_grows_as_the_budget_falls(self):
        seen = [
            int(self._flags(auto_fit = auto, delta = delta)[-1])
            for auto in (True, False)
            for delta in (512.0, 1024.0, 2048.0)
        ]
        assert seen == sorted(seen[:3]) + sorted(seen[3:])

    def test_a_raised_budget_reaches_the_fallback_too(self):
        assert self._flags(auto_fit = False, delta = -369.0)[-2:] == ["--fit-target", "655"]

    def test_a_raised_budget_stops_at_the_floor(self):
        assert self._flags(auto_fit = False, delta = -4096.0)[-2:] == ["--fit-target", "512"]
        assert self._flags(auto_fit = True, delta = -4096.0)[-2:] == ["--fit-target", "512"]

    def test_nothing_is_emitted_without_the_capability(self):
        # An older llama-server rejects unknown flags outright.
        caps = {"supports_fit_ctx": True, "supports_fit_target": False}
        import core.inference.llama_cpp as lc

        flags = lc.LlamaCppBackend._ctx_integrity_flags(
            1,
            True,
            True,
            0,
            0,
            caps,
            fit_target_delta_mib = 4096.0,
        )
        assert "--fit-target" not in flags

    def test_the_move_is_measured_from_the_card_that_makes_it_safe(self):
        # The broadcast value, kept wherever llama.cpp's device order is not provable; TestPerDeviceFitTarget covers the per-device list.
        compact = _shared_setup_1()
        assert "if_vram_frac!=_CTX_FIT_VRAM_FRACTIONandgpus:" in compact
        assert "_fit_target_delta_mib=(_CTX_FIT_VRAM_FRACTION-_vram_frac)*_scale" in compact
        assert "_scale=(max(_scales)if_vram_frac<_CTX_FIT_VRAM_FRACTIONelsemin(_scales))" in compact
        assert "fit_target_delta_mib=_fit_target_delta_mib," in compact
        assert "total_by_idx.get(_idx)or_freefor_idx,_freeingpus" in compact


class TestPerDeviceFitTarget:
    """On mixed cards the margin must be per device: a broadcast one made the 24 GB card keep 33 GiB
    free."""

    _ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    # 48 GB and 24 GB cards at a 30% budget, legacy path (1024 MiB base).
    _MARGINS = {0: 1024 + 0.67 * 49140, 1: 1024 + 0.67 * 24576}
    _CMD = ["llama-server", "--fit", "on", "--fit-target", "33947", "--jinja"]

    def _spell(
        self,
        cmd = None,
        env = None,
        margins = None,
    ):
        import core.inference.llama_cpp as lc
        return lc.LlamaCppBackend._per_device_fit_target_argv(
            list(cmd or self._CMD),
            self._ENV if env is None else env,
            self._MARGINS if margins is None else margins,
        )

    def test_each_card_keeps_its_own_margin(self):
        out = self._spell()
        assert out[out.index("--fit-target") + 1] == "33947,17489"
        # Only the value moves.
        assert [t for t in out if t != "33947,17489"] == [t for t in self._CMD if t != "33947"]

    def test_the_list_follows_the_child_mask_order(self):
        # CUDA enumerates a numeric mask in the order it lists the ids.
        out = self._spell(env = {**self._ENV, "CUDA_VISIBLE_DEVICES": "1,0"})
        assert out[out.index("--fit-target") + 1] == "17489,33947"

    def test_a_pinned_single_card_gets_its_own_value(self):
        out = self._spell(env = {**self._ENV, "CUDA_VISIBLE_DEVICES": "1"})
        assert out[out.index("--fit-target") + 1] == "17489"

    def test_identical_cards_keep_the_single_value(self):
        # The same meaning, and an unchanged command on homogeneous hosts.
        margins = {0: 9378.0, 1: 9378.0}
        assert self._spell(cmd = ["--fit-target", "9378"], margins = margins) == [
            "--fit-target",
            "9378",
        ]

    @pytest.mark.parametrize(
        "env",
        [
            {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"},
            {},
            {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "GPU-1234"},
            {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,2"},
            {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "-1"},
            {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "LLAMA_ARG_DEVICE": "CUDA1"},
            {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "LLAMA_ARG_RPC": "host:50052"},
        ],
        ids = [
            "fastest-first",
            "no-order",
            "uuid-mask",
            "unknown-id",
            "cpu",
            "env-device",
            "env-rpc",
        ],
    )
    def test_an_unprovable_order_keeps_the_broadcast_value(self, env):
        assert self._spell(env = env) == self._CMD

    @pytest.mark.parametrize("flag", ["--device", "-dev", "--rpc"])
    def test_device_flags_in_the_command_keep_the_broadcast_value(self, flag):
        cmd = [*self._CMD, flag, "CUDA0"]
        assert self._spell(cmd = cmd) == cmd

    def test_nothing_to_spell_without_margins_or_the_flag(self):
        assert self._spell(margins = {}) == self._CMD
        cmd = ["llama-server", "--fit", "on"]
        assert self._spell(cmd = cmd) == cmd

    def test_a_raised_budget_gives_each_card_its_own_smaller_margin(self):
        # Above the default, delta is negative: a larger card may pack closer than the smaller.
        import core.inference.llama_cpp as lc

        margins = lc.LlamaCppBackend._fit_target_margins_mib(
            auto_fit = False,
            fit_target_delta_by_idx = {0: -0.01 * 49140, 1: -0.01 * 24576},
        )
        out = self._spell(margins = margins)
        assert out[out.index("--fit-target") + 1] == "532,778"
        # At 100% both reach the 512 MiB floor, one value again.
        floored = lc.LlamaCppBackend._fit_target_margins_mib(
            auto_fit = False,
            fit_target_delta_by_idx = {0: -0.03 * 49140, 1: -0.03 * 24576},
        )
        assert floored == {0: 512.0, 1: 512.0}
        out = self._spell(margins = floored)
        assert out[out.index("--fit-target") + 1] == "512"

    def test_per_card_margins_use_the_scalar_formula(self):
        import core.inference.llama_cpp as lc

        margins = lc.LlamaCppBackend._fit_target_margins_mib(
            auto_fit = False, fit_target_delta_by_idx = {0: 32924.0, 1: -4096.0}
        )
        for idx, delta in ((0, 32924.0), (1, -4096.0)):
            assert margins[idx] == lc.LlamaCppBackend._fit_target_margin_mib(
                auto_fit = False, fit_target_delta_mib = delta
            )
        assert (
            lc.LlamaCppBackend._fit_target_margins_mib(auto_fit = False, fit_target_delta_by_idx = None)
            is None
        )
        assert (
            lc.LlamaCppBackend._fit_target_margins_mib(
                auto_fit = False,
                fit_target_delta_by_idx = {0: 1.0},
                supports_fit_target = False,
            )
            is None
        )

    def test_the_paging_check_credits_each_card_its_own_margin(self, monkeypatch):
        # The load-mode fit must spend what the child is told to keep, card by card.
        import core.inference.llama_cpp as lc

        backend = lc.LlamaCppBackend()
        rows = [(0, 47000, 49140), (1, 24000, 24576)]
        gib = 1024**3
        # Broadcast spills 7.3 GiB to RAM; per-card margins spill only 0.9 GiB.
        kwargs = dict(avail_mib = 4 * 1024, headroom_mib = 0)
        broadcast = backend._fits_without_paging(20 * gib, rows, vram_margin_mib = 33947.0, **kwargs)
        per_card = backend._fits_without_paging(
            20 * gib, rows, vram_margin_mib = {0: 33947.0, 1: 17489.0}, **kwargs
        )
        assert broadcast is False
        assert per_card is True

    def _ok(
        self,
        monkeypatch,
        *,
        extra_args = None,
        env = None,
        **attrs,
    ):
        import core.inference.llama_cpp as lc

        B = lc.LlamaCppBackend
        # The eligibility check declines on macOS, so pin the host OS the case means.
        monkeypatch.setattr(lc.sys, "platform", attrs.get("platform", "linux"))
        monkeypatch.setattr(B, "_GPU_IDS_ARE_PCI_INDICES", attrs.get("pci", True))
        monkeypatch.setattr(B, "_SMI_MIG_INDICES", attrs.get("mig", frozenset()))
        monkeypatch.setattr(
            B,
            "_visibility_mask_is_unmappable",
            staticmethod(lambda: attrs.get("unmappable", False)),
        )
        monkeypatch.setattr(
            B, "_host_torch_is_rocm", staticmethod(lambda: attrs.get("rocm", False))
        )
        monkeypatch.setenv("CUDA_DEVICE_ORDER", attrs.get("order", "PCI_BUS_ID"))
        if attrs.get("mask") is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", attrs["mask"])
        return B._fit_target_per_device_ok(
            is_vulkan_backend = attrs.get("vulkan", False),
            extra_args = extra_args,
            env = env or {},
        )

    def test_eligible_on_a_plain_cuda_host(self, monkeypatch):
        assert self._ok(monkeypatch)
        assert self._ok(monkeypatch, mask = "1,0")

    def test_a_resolvable_uuid_mask_is_ineligible(self, monkeypatch):
        # The child inherits the uuids, which _per_device_fit_target_argv cannot order.
        assert not self._ok(monkeypatch, mask = "GPU-aaaa,GPU-bbbb")

    @pytest.mark.parametrize(
        "attrs",
        [
            {"vulkan": True},
            {"rocm": True},
            {"pci": None},
            {"pci": False},
            {"order": "FASTEST_FIRST"},
            {"unmappable": True},
            {"mig": frozenset({0})},
        ],
        ids = [
            "vulkan",
            "rocm",
            "ids-unknown",
            "torch-ordinals",
            "fastest-first",
            "uuid-mask",
            "mig",
        ],
    )
    def test_an_unprovable_device_order_is_ineligible(self, monkeypatch, attrs):
        assert not self._ok(monkeypatch, **attrs)

    @pytest.mark.parametrize(
        "extra_args, env",
        [
            (["--fit-target", "2048"], None),
            (None, {"LLAMA_ARG_FIT_TARGET": "2048"}),
            (["--device", "CUDA1"], None),
            (["--rpc", "host:50052"], None),
            (None, {"LLAMA_ARG_DEVICE": "CUDA1"}),
            (["--spec-draft-device", "CUDA1"], None),
            (["--mmproj-device", "CUDA1"], None),
        ],
        ids = [
            "argv-fit-target",
            "env-fit-target",
            "device",
            "rpc",
            "env-device",
            "draft-device",
            "mmproj-device",
        ],
    )
    def test_pass_through_placement_is_ineligible(self, monkeypatch, extra_args, env):
        assert not self._ok(monkeypatch, extra_args = extra_args, env = env)

    def test_the_load_path_wires_the_per_card_margins(self):
        compact = _shared_setup_1()
        # Planned over every probed row, which an unpinned child sees.
        assert "for_idx,_free,_totalin_gpu_mem" in compact
        # Spelled at both spawn sites, against the final env.
        assert compact.count("_per_device_fit_target_argv(") == 2
        # Priced with the same margins it emits.
        assert "fit_margin_mib=_fit_margins_by_idxor_fit_margin_mib" in compact


class TestManualAutoIsPriced:
    """Manual + Auto empties ``gpus``, but --fit-target still spends the budget."""

    def _helper(self):
        import inspect

        import core.inference.llama_cpp as lc

        source = inspect.getsource(lc.LlamaCppBackend.load_model)
        helper = source[source.index("def _budget_priced_placement()") :]
        helper = helper[: helper.index("self._vram_fraction_pending")]
        return "".join(helper.split())

    def test_an_unplanned_but_fitted_child_is_still_stamped(self):
        assert "return_vram_fracif_fit_target_pricedelseNone" in self._helper()

    def test_priced_is_read_off_the_emitted_flags(self):
        compact = _shared_setup_1()
        assert '_fit_target_priced="--fit-target"in_integrity_flags' in compact

    def test_a_cpu_fallback_child_is_still_refused(self):
        helper = self._helper()
        assert helper.index("ifintent.cpu_fallback:returnNone") < helper.index("_fit_target_priced")
