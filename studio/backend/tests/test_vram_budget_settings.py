# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the VRAM budget setting.

The budget decides how much of each card a load may claim, so the bar is that an
unset budget behaves exactly as the hard-coded 0.97 did, and that no malformed
value can ever reach the fit. A NaN in particular would turn every per-GPU budget
into NaN and silently fit nothing.
"""

from __future__ import annotations

import pytest

import utils.vram_budget_settings as vb


@pytest.fixture(autouse = True)
def _isolate(monkeypatch):
    """No stored value and no environment, so each test states its own inputs."""
    monkeypatch.delenv(vb.VRAM_FRACTION_ENV_VAR, raising = False)
    monkeypatch.setattr(vb, "_cached_setting", lambda _key: None)


class TestCoerceFraction:
    @pytest.mark.parametrize("value", [0.80, 0.9, 0.97, 1.0, "0.85", "1.00"])
    def test_accepts_in_range(self, value):
        assert vb.coerce_fraction(value) == pytest.approx(float(value))

    @pytest.mark.parametrize(
        "value",
        [
            0.79,
            1.01,
            0.0,
            -1.0,
            2.0,  # out of range
            "nan",
            "NaN",
            float("nan"),  # NaN loses every comparison
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
        # A no-op when nobody sets a budget, so this must track
        # _CTX_FIT_VRAM_FRACTION. Imported lazily: the inference package is heavy.
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
        # A future build's wider range must not widen the budget on this one.
        monkeypatch.setattr(vb, "_cached_setting", lambda _key: 4.0)
        assert vb.get_vram_budget_fraction() == vb.VRAM_FRACTION_DEFAULT

    def test_unreadable_db_does_not_fail_a_load(self, monkeypatch):
        def _boom(_key):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(vb, "_cached_setting", _boom)
        with pytest.raises(RuntimeError):
            vb.get_vram_budget_fraction()
        # ...but the llama_cpp caller swallows it, which is what protects the load.
        from core.inference.llama_cpp import _active_vram_fraction, _CTX_FIT_VRAM_FRACTION

        assert _active_vram_fraction() == _CTX_FIT_VRAM_FRACTION


class TestGrid:
    def test_a_tenth_of_a_percent_is_a_legal_budget(self):
        # The slider steps in tenths, so the fraction has to survive storage on
        # that grid or the control would show a value the backend never held.
        assert vb.coerce_fraction(0.975) == 0.975
        assert vb.coerce_fraction("0.975") == 0.975

    def test_off_grid_values_are_quantised_rather_than_refused(self):
        # An environment variable or a hand-written row need not land on a stop.
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
        for bad in (0.5, 1.5, float("nan"), "abc"):
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
        # A null row reads back as "no value", so env/default applies again.
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
    """``_vram_fraction_launched`` must describe the child that is actually running.

    The settings route reports "reload required" by comparing the saved budget
    against this marker, so a path that returns without launching must leave it
    alone. The duplicate-load fast path is the reachable one: the route declines
    to reuse a resident model while its audio probe is unfinished, so the request
    reaches ``load_model``, which adopts the live server and returns without
    replacing it.
    """

    @staticmethod
    def _resident_backend(monkeypatch, *, launched: float, active: float):
        import core.inference.llama_cpp as lc

        backend = lc.LlamaCppBackend()
        # is_loaded / is_active only test "is not None"; nothing here talks to it.
        backend._process = object()
        backend._healthy = True
        backend._vram_fraction_launched = launched
        # The saved budget the next load would use, different from the running one.
        monkeypatch.setattr(lc, "_active_vram_fraction", lambda: active)
        monkeypatch.setattr(backend, "adopt_load_intent_if_matched", lambda _intent: True)
        return backend, lc

    def test_duplicate_load_leaves_the_running_child_marker(self, monkeypatch):
        backend, lc = self._resident_backend(monkeypatch, launched = 0.97, active = 0.85)
        backend._audio_probed = True

        assert backend.load_model(lc.GgufLoadIntent(model_identifier = "owner/repo"))
        # Nothing relaunched, so the child is still on 0.97 and needs a reload.
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
        # The budget is server-wide and on no request field, so the intent is
        # identical; without this check the slider silently does nothing.
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
        # Manual mode and GPU-less hosts plan with no devices, so a reload changes
        # nothing.
        backend, lc = self._adoptable(monkeypatch, launched = None)
        monkeypatch.setattr(lc, "_active_vram_fraction", lambda: 0.85)

        assert backend.adopt_load_intent_if_matched(
            lc.GgufLoadIntent(model_identifier = "owner/repo")
        )

    def test_marker_is_committed_with_the_rest_of_the_launch_state(self):
        # Guards the placement: next to _requested_n_batch, inside the block only a
        # _healthy=True launch runs, not at the top of load_model with no child yet.
        import inspect

        import core.inference.llama_cpp as lc

        compact = "".join(inspect.getsource(lc.LlamaCppBackend.load_model).split())
        assert (
            "self._vram_fraction_launched=_budget_priced_placement()"
            "self._vram_fraction_pending=Noneself._requested_n_batch" in compact
        )

    def test_marker_is_none_when_placement_had_no_devices(self):
        # gpus is empty in manual mode and on GPU-less hosts, and every consumer of
        # the fraction is gated on it, so a value there is a budget the child never
        # applied. Pending value and committed marker share the one predicate.
        import inspect

        import core.inference.llama_cpp as lc

        compact = "".join(inspect.getsource(lc.LlamaCppBackend.load_model).split())
        assert "self._vram_fraction_pending=_budget_priced_placement()" in compact
        assert "self._vram_fraction_launched=_budget_priced_placement()" in compact


class TestRouteContract:
    """The two places the HTTP layer can answer wrongly on well-formed input."""

    @staticmethod
    def _settings_module():
        import routes.settings as rs
        return rs

    def test_payload_rejects_a_boolean_fraction(self):
        # bool subclasses int, so non-strict parsing turns True into 1.0 and stores
        # the max budget instead of 422; pydantic coerces before the util's guard.
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
        # The window: load_model captured its fraction but _process is still None,
        # so is_active would report no reload while the child is already committed.
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
        # The diffusion branch returns before the launch block that commits the
        # marker, so a previous llama-server's fraction would survive and, since the
        # dedupe compares it, relaunch a healthy diffusion runner on every Apply.
        import inspect

        import core.inference.llama_cpp as lc

        source = inspect.getsource(lc.LlamaCppBackend.load_model)
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
        # A queued request would otherwise plan with the fraction as it stood on
        # arrival while the duplicate check reads the live one, evicting the
        # resident child for a budget the queued load then fails to apply.
        compact = self._load_model_source()
        lock_at = compact.index("withself._serial_load_scope():")
        resolve_at = compact.index("_vram_frac=_active_vram_fraction()")
        dedupe_at = compact.index("ifself.adopt_load_intent_if_matched(intent):")
        assert lock_at < resolve_at < dedupe_at

    def test_a_healthy_spawn_keeps_the_pending_value_until_the_commit(self):
        # The decode probe and the no-flash, drafter and projector retries run after
        # the first spawn and the marker is committed after them, so releasing the
        # pending value at the spawn would answer from the PREVIOUS child's marker.
        compact = self._load_model_source()
        assert "ifnothealthy:self._vram_fraction_pending=None" in compact
        assert (
            "self._vram_fraction_launched=_budget_priced_placement()self._vram_fraction_pending=None"
            in compact
        )

    def test_a_cpu_fallback_child_is_not_stamped_with_a_budget(self):
        # An auto Vulkan crash that recovers on CPU rewrites the intent but leaves
        # gpus populated from the failed attempt, so gpus alone would stamp a
        # CPU-only child.
        import inspect

        import core.inference.llama_cpp as lc

        source = inspect.getsource(lc.LlamaCppBackend.load_model)
        helper = source[source.index("def _budget_priced_placement()") :]
        helper = helper[: helper.index("# Nothing is committed yet")]
        compact = "".join(helper.split())
        assert "ifnotgpusorintent.cpu_fallback:returnNone" in compact


class TestPreLaunchWindow:
    @staticmethod
    def _compact():
        import inspect

        import core.inference.llama_cpp as lc
        return "".join(inspect.getsource(lc.LlamaCppBackend.load_model).split())

    def test_the_pending_value_is_armed_before_the_download(self):
        # The download and planning before the spawn take minutes with the old child
        # gone, so a save there would be told no reload is needed while the eventual
        # child carries the old fraction.
        compact = self._compact()
        armed = compact.index("self._vram_fraction_pending=_vram_frac\n".strip())
        cancel = compact.index("self._cancel_event.clear()")
        spawn = compact.index("self._vram_fraction_pending=_budget_priced_placement()")
        assert armed < cancel < spawn

    def test_a_terminal_failure_releases_the_pending_value(self):
        # The route reads the pending value before it checks is_active, so a value
        # left behind by a failed load would ask for a reload with nothing loaded.
        import inspect

        import core.inference.llama_cpp as lc

        source = inspect.getsource(lc.LlamaCppBackend.load_model)
        funnel = source[source.index("def _raise_terminal_load_failure") :]
        funnel = funnel[: funnel.index("def _try_auto_vulkan_cpu_fallback")]
        compact = "".join(funnel.split())
        assert "self._vram_fraction_pending=None" in compact
        assert compact.index("self._vram_fraction_pending=None") < compact.index(
            "raiseRuntimeError(detail)"
        )


class TestPendingOwnership:
    def test_the_pending_value_is_released_with_the_load_lock(self):
        # Armed before the download, so every exit ahead of the spawn must give it
        # back and the route reads it before is_active. The release belongs to the
        # lock, not the call: overlapping /load calls hand the lock over before the
        # first returns, so clearing on the way out would discard the queued marker.
        import inspect

        import core.inference.llama_cpp as lc

        code = [
            line
            for line in inspect.getsource(lc.LlamaCppBackend._serial_load_scope).splitlines()
            if not line.strip().startswith("#")
        ]
        compact = "".join("".join(code).split())
        assert "withself._serial_load_lock:try:yieldfinally:" in compact
        assert compact.endswith("self._vram_fraction_pending=None")
        # And the load has to go through it, or the scope guards nothing.
        load = "".join(inspect.getsource(lc.LlamaCppBackend.load_model).split())
        assert "withself._serial_load_scope():" in load

    def test_a_pre_launch_exit_leaves_no_pending_value(self, monkeypatch):
        # Exercised rather than read: the diffusion path returns before the spawn.
        import core.inference.llama_cpp as lc

        backend = lc.LlamaCppBackend()
        backend._audio_probed = True
        monkeypatch.setattr(lc, "_active_vram_fraction", lambda: 0.9)
        monkeypatch.setattr(backend, "adopt_load_intent_if_matched", lambda _intent: False)
        # Any failure ahead of the spawn will do; the wrapper must still clean up.
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

    def test_full_budget_still_leaves_the_floor(self):
        import core.inference.llama_cpp as lc

        # 24 GB card, nothing else resident.
        assert self._usable(24_576, 24_576, 1.0) == pytest.approx(
            24_576 - lc._VRAM_FLOOR_RESERVE_MIB
        )
        # And on a small card, where a pure percentage would leave nothing.
        assert self._usable(8_192, 8_192, 1.0) == pytest.approx(8_192 - lc._VRAM_FLOOR_RESERVE_MIB)

    @pytest.mark.parametrize("total", [4_096, 8_192, 16_384, 24_576, 81_920])
    def test_the_default_reserve_is_unchanged_on_every_card_size(self, total):
        # The acceptance bar for the whole setting: unset behaves exactly as the
        # hard-coded 0.97 did. The floor must not reach below the default, which
        # it otherwise would on any card under about 17 GB.
        import core.inference.llama_cpp as lc
        assert self._usable(total, total, lc._CTX_FIT_VRAM_FRACTION) == pytest.approx(
            total - (1.0 - lc._CTX_FIT_VRAM_FRACTION) * total
        )

    def test_the_floor_only_binds_where_the_percentage_reserves_less(self):
        import core.inference.llama_cpp as lc

        # 1% of 80 GB is 819 MiB, above the floor, so 99% keeps its percentage.
        assert self._usable(81_920, 81_920, 0.99) == pytest.approx(81_920 * 0.99)
        # 1% of 24 GB is 245 MiB, under the floor, so the floor takes over.
        assert self._usable(24_576, 24_576, 0.99) == pytest.approx(
            24_576 - lc._VRAM_FLOOR_RESERVE_MIB
        )

    def test_an_absolute_pool_budget_is_not_charged_twice(self):
        # The tensor-parallel paths pass an already computed pool budget with
        # budget_frac = 1.0 and total_mib = None. Flooring there would subtract a
        # reserve the pool budget has already paid for on each of its cards.
        assert self._usable(12_000, 0, 1.0) == pytest.approx(12_000)
        assert self._usable(12_000, None, 1.0) == pytest.approx(12_000)
