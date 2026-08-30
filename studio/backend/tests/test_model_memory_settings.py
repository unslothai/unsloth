# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the Model Memory residency settings.

Pins what the toggles promise: which llama-server flags reach the subprocess,
and that residency vetoes the idle-unload TTL without destroying the stored one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent

# Load directly, like test_llama_server_args.py: importing the package would
# drag in the whole inference chain.
_spec = importlib.util.spec_from_file_location(
    "_lsa_model_memory_test_only", _BACKEND / "core" / "inference" / "llama_server_args.py"
)
_lsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lsa)
apply_model_memory_policy = _lsa.apply_model_memory_policy
apply_load_mode_policy = _lsa.apply_load_mode_policy
memory_state_satisfies_settings = _lsa.memory_state_satisfies_settings
model_memory_suppresses_load_mode = _lsa.model_memory_suppresses_load_mode
resolve_effective_memory_state = _lsa.resolve_effective_memory_state
scrub_memory_env = _lsa.scrub_memory_env

import utils.model_memory_settings as mm_settings  # noqa: E402

strip_shadowing_flags = _lsa.strip_shadowing_flags


@pytest.fixture
def policy(monkeypatch):
    """Run the policy under a given toggle pair.

    The policy imports the settings module lazily, so patch that module.
    """
    import utils.model_memory_settings as mm

    def run(
        keep_resident: bool,
        no_ram_reserve: bool,
        extras,
        supports_load_mode = False,
        weights_in_host_memory = True,
    ):
        monkeypatch.setattr(mm, "get_keep_resident", lambda: keep_resident)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: no_ram_reserve)
        monkeypatch.setattr(mm, "should_mlock", lambda: keep_resident and not no_ram_reserve)
        return apply_model_memory_policy(
            extras,
            supports_load_mode = supports_load_mode,
            weights_in_host_memory = weights_in_host_memory,
        )

    return run


class TestFlagPolicy:
    def test_both_off_is_a_pure_pass_through(self, policy):
        # Pre-feature contract: hand-typed flags survive, nothing is added.
        extras = ["--mlock", "--no-mmap", "--temp", "0.7"]
        managed, out = policy(False, False, extras)
        assert managed == []
        assert out == extras

    def test_keep_resident_emits_mlock(self, policy):
        # Legacy build: --mlock is deprecated upstream but still accepted.
        managed, out = policy(True, False, ["--temp", "0.7"])
        assert managed == ["--mlock"]
        assert out == ["--temp", "0.7"]

    def test_keep_resident_prefers_load_mode_when_supported(self, policy):
        managed, out = policy(True, False, ["--temp", "0.7"], supports_load_mode = True)
        assert managed == ["--load-mode", "mmap+mlock"]
        assert out == ["--temp", "0.7"]

    def test_load_mode_is_stripped_from_extras(self, policy):
        # A user --load-mode would last-wins-override the managed one, and
        # "--load-mode mlock" is a RAM reservation no-reserve must veto.
        _, out = policy(
            True, False, ["--load-mode", "none", "--temp", "0.7"], supports_load_mode = True
        )
        assert out == ["--temp", "0.7"]
        _, out = policy(False, True, ["-lm", "mlock", "--temp", "0.7"])
        assert out == ["--temp", "0.7"]

    def test_load_mode_strip_is_not_boolean(self, policy):
        # It takes a value, so the value must go with the flag, not survive.
        _, out = policy(False, True, ["--load-mode", "mmap+mlock"])
        assert out == []

    def test_keep_resident_does_not_double_emit_mlock(self, policy):
        # A user --mlock folds into the managed one, not a second copy.
        managed, out = policy(True, False, ["--mlock", "--temp", "0.7"])
        assert managed == ["--mlock"]
        assert "--mlock" not in out

    def test_no_ram_reserve_strips_both_reservation_flags(self, policy):
        managed, out = policy(False, True, ["--mlock", "--no-mmap", "-ngl", "99"])
        assert managed == []
        # Unrelated flags (and their values) survive untouched.
        assert out == ["-ngl", "99"]

    def test_no_ram_reserve_wins_over_keep_resident(self, policy):
        # --mlock is itself a RAM reservation, so no-reserve vetoes it.
        managed, out = policy(True, True, ["--mlock", "--no-mmap", "--temp", "0.7"])
        assert managed == []
        assert out == ["--temp", "0.7"]

    def test_handles_absent_extras(self, policy):
        assert policy(True, False, None) == (["--mlock"], [])
        assert policy(False, False, None) == ([], [])

    def test_caller_extras_are_not_mutated(self, policy):
        # load_model reads extra_args after this: None must stay None (inherit
        # previous) rather than becoming [] (clear), and the user's saved flags
        # must survive, so the policy only ever returns a launch-only copy.
        original = ["--mlock", "--no-mmap", "--temp", "0.7"]
        passed = list(original)
        _, out = policy(True, True, passed)
        assert passed == original
        assert out is not passed

    @pytest.mark.parametrize("flag", ["--mlock", "-mlock", "--no-mmap", "-no-mmap"])
    def test_aliases_are_stripped(self, policy, flag):
        _, out = policy(False, True, [flag, "--temp", "0.7"])
        assert out == ["--temp", "0.7"]

    def test_strip_is_boolean_and_keeps_the_next_token(self):
        # --mlock takes no value, so stripping it must not swallow "0.7".
        assert strip_shadowing_flags(
            ["--mlock", "0.7"],
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = False,
            strip_mlock = True,
        ) == ["0.7"]


class TestIdleUnloadVeto:
    def test_keep_resident_zeroes_the_effective_ttl(self, monkeypatch):
        import utils.model_memory_settings as mm
        import utils.openai_auto_switch_settings as aus

        monkeypatch.setattr(aus, "_stored_idle_seconds", lambda: 300)
        monkeypatch.setattr(aus, "get_openai_auto_switch_enabled", lambda: True)

        monkeypatch.setattr(mm, "get_keep_resident", lambda: False)
        assert aus.get_auto_unload_idle_seconds() == 300

        monkeypatch.setattr(mm, "get_keep_resident", lambda: True)
        assert aus.get_auto_unload_idle_seconds() == 0
        # The stored value survives, so turning residency off restores it.
        assert aus.get_stored_auto_unload_idle_seconds() == 300


class TestPersistence:
    def test_partial_update_leaves_the_other_key_alone(self, monkeypatch):
        import utils.model_memory_settings as mm

        store: dict = {}
        monkeypatch.setattr(mm, "_cached_setting", lambda key: store.get(key))
        monkeypatch.setattr(
            "storage.studio_db.upsert_app_settings", lambda updates: store.update(updates)
        )

        assert mm.set_model_memory_settings(keep_resident = True) == (True, False)
        assert mm.set_model_memory_settings(no_ram_reserve = True) == (True, True)
        # Only keep_resident is sent; no_ram_reserve must not reset.
        assert mm.set_model_memory_settings(keep_resident = False) == (False, True)

    @pytest.mark.parametrize("value", ["banana", 2.5, object()])
    def test_rejects_non_boolean(self, value):
        import utils.model_memory_settings as mm
        with pytest.raises(ValueError):
            mm.set_model_memory_settings(keep_resident = value)

    @pytest.mark.parametrize(
        ("stored", "expected"),
        [
            (True, True),
            ("true", True),
            ("on", True),
            (False, False),
            ("off", False),
            ("", False),
            (None, False),
            ("nonsense", False),
        ],
    )
    def test_coercion_defaults_to_off(self, monkeypatch, stored, expected):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "_cached_setting", lambda key: stored)
        assert mm.get_keep_resident() is expected
        assert mm.get_no_ram_reserve() is expected


@pytest.mark.skipif(
    importlib.util.find_spec("resource") is None,
    reason = "resource is unavailable on Windows",
)
class TestMemlockLimit:
    """mlock cannot exceed RLIMIT_MEMLOCK. Linux commonly defaults to 8 MB,
    where llama.cpp warns and carries on, so residency silently does nothing.
    The settings response reports the cap so the UI can say so."""

    def test_unlimited_reports_none(self, monkeypatch):
        import resource

        import utils.model_memory_settings as mm

        monkeypatch.setattr(
            resource,
            "getrlimit",
            lambda _w: (resource.RLIM_INFINITY, resource.RLIM_INFINITY),
        )
        assert mm.memlock_limit_bytes() is None

    @pytest.mark.parametrize("soft", [0, 64 * 1024, 8 * 1024 * 1024])
    def test_finite_limits_are_reported(self, monkeypatch, soft):
        import resource

        import utils.model_memory_settings as mm

        monkeypatch.setattr(resource, "getrlimit", lambda _w: (soft, soft))
        assert mm.memlock_limit_bytes() == soft

    def test_negative_is_treated_as_unlimited(self, monkeypatch):
        import resource

        import utils.model_memory_settings as mm

        monkeypatch.setattr(resource, "getrlimit", lambda _w: (-1, -1))
        assert mm.memlock_limit_bytes() is None

    @pytest.mark.parametrize("exc", [ValueError, OSError, AttributeError])
    def test_probe_failure_never_raises(self, monkeypatch, exc):
        import resource

        import utils.model_memory_settings as mm

        def boom(_w):
            raise exc("nope")

        monkeypatch.setattr(resource, "getrlimit", boom)
        assert mm.memlock_limit_bytes() is None


class TestMemoryEnv:
    """llama.cpp reads LLAMA_ARG_MLOCK / _MMAP / _LOAD_MODE before argv, so
    stripping the tokens alone leaves an inherited value in force."""

    @pytest.fixture
    def toggles(self, monkeypatch):
        import utils.model_memory_settings as mm
        def set(keep, no_res):
            monkeypatch.setattr(mm, "get_keep_resident", lambda: keep)
            monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: no_res)

        return set

    @pytest.mark.parametrize(
        ("var", "value"),
        [
            ("LLAMA_ARG_MLOCK", "1"),
            ("LLAMA_ARG_MMAP", "off"),  # mmap disabled -> mode none
            ("LLAMA_ARG_NO_MMAP", "0"),  # presence alone -> mode none
            ("LLAMA_ARG_DIO", "off"),  # DirectIO disabled -> mode none
            ("LLAMA_ARG_NO_DIO", "0"),
            ("LLAMA_ARG_LOAD_MODE", "none"),
            ("LLAMA_ARG_LOAD_MODE", "mlock"),
            ("LLAMA_ARG_LOAD_MODE", "mmap+mlock"),
        ],
    )
    def test_a_toggle_scrubs_inherited_reservations(self, toggles, var, value):
        toggles(False, True)
        env = {var: value, "PATH": "/usr/bin"}
        assert var in scrub_memory_env(env)
        assert var not in env
        assert env["PATH"] == "/usr/bin"

    @pytest.mark.parametrize(
        ("var", "value"),
        [
            ("LLAMA_ARG_MLOCK", "0"),  # measured: falsy does not lock
            ("LLAMA_ARG_MMAP", "1"),  # mmap: maps, holds no full copy
            ("LLAMA_ARG_DIO", "1"),  # DirectIO: streams
            ("LLAMA_ARG_LOAD_MODE", "mmap"),
            ("LLAMA_ARG_LOAD_MODE", "dio"),
            ("LLAMA_ARG_LOAD_MODE", "future-mode"),  # unknown: leave it alone
        ],
    )
    def test_a_non_reserving_loader_choice_survives(self, toggles, var, value):
        """The settings own the reservation, not the loader, exactly as on the
        argv side where --load-mode dio is kept."""
        toggles(False, True)
        env = {var: value, "PATH": "/usr/bin"}
        assert scrub_memory_env(env) == []
        assert env == {var: value, "PATH": "/usr/bin"}

    def test_a_kept_choice_really_does_satisfy_no_reserve(self, toggles):
        """Otherwise keeping it would just make the reload hint fire forever."""
        toggles(False, True)
        for env in (
            {"LLAMA_ARG_DIO": "1"},
            {"LLAMA_ARG_LOAD_MODE": "dio"},
            {"LLAMA_ARG_MMAP": "1"},
        ):
            scrub_memory_env(dict(env))
            assert resolve_effective_memory_state([], env) == (False, False)

    def test_residency_still_clears_a_conflicting_inherited_lock_setting(self, toggles):
        toggles(True, False)
        env = {"LLAMA_ARG_LOAD_MODE": "none"}
        assert scrub_memory_env(env) == ["LLAMA_ARG_LOAD_MODE"]
        assert env == {}

    def test_both_off_leaves_the_env_untouched(self, toggles):
        toggles(False, False)
        env = {"LLAMA_ARG_MLOCK": "1"}
        assert scrub_memory_env(env) == []
        assert env == {"LLAMA_ARG_MLOCK": "1"}


class TestEffectiveMemoryState:
    """What the child really runs with: env defaults, argv last-wins on top."""

    @pytest.mark.parametrize(
        ("argv", "env", "expected"),
        [
            ([], {}, (False, False)),
            (["--mlock"], {}, (True, False)),
            (["--no-mmap"], {}, (False, True)),
            (["--load-mode", "mmap+mlock"], {}, (True, False)),
            (["--load-mode=mmap+mlock"], {}, (True, False)),
            (["-lm", "mlock"], {}, (True, True)),  # mlock without mmap
            (["--load-mode", "dio"], {}, (False, False)),  # DirectIO streams
            (["--load-mode", "none"], {}, (False, True)),
            ([], {"LLAMA_ARG_MLOCK": "1"}, (True, False)),
            ([], {"LLAMA_ARG_MMAP": "off"}, (False, True)),
            ([], {"LLAMA_ARG_LOAD_MODE": "mmap+mlock"}, (True, False)),
            # argv beats env, matching llama.cpp.
            (["--load-mode", "mmap"], {"LLAMA_ARG_MLOCK": "1"}, (False, False)),
            (["--mlock", "--load-mode", "mmap"], {}, (False, False)),
            # --no-mmap is the deprecated selector for the whole "none" mode, so
            # it clears the mlock. Both orderings measured against the binary.
            (["--mlock", "--no-mmap"], {}, (False, True)),
            (["--no-mmap", "--mlock"], {}, (True, True)),
            (["--mlock", "--mmap"], {}, (False, False)),
            (["--mlock", "--direct-io"], {}, (False, False)),
        ],
    )
    def test_precedence(self, argv, env, expected):
        assert resolve_effective_memory_state(argv, env) == expected

    def test_degenerate_inputs(self):
        assert resolve_effective_memory_state(None, None) == (False, False)
        assert resolve_effective_memory_state(["--load-mode"], {}) == (False, False)
        # A following flag is not swallowed as the value.
        assert resolve_effective_memory_state(["--load-mode", "--mlock"], {}) == (True, False)


class TestReloadRequired:
    """The reload hint must reflect the launched state, not only what Unsloth
    emitted, so a user-supplied --mlock / --no-mmap counts too."""

    @staticmethod
    def _required(
        keep,
        no_res,
        state,
        monkeypatch,
        *,
        is_loaded = True,
        is_active = True,
    ):
        import routes.inference
        import routes.settings as rs
        import utils.model_memory_settings as mm

        backend = type(
            "_B",
            (),
            {
                "is_loaded": is_loaded,
                "is_active": is_active,
                "_memory_state": state,
                "_memory_policy_active": True,
                "_memory_mlock_applicable": True,
            },
        )()
        monkeypatch.setattr(routes.inference, "get_llama_cpp_backend", lambda: backend)
        monkeypatch.setattr(mm, "get_keep_resident", lambda: keep)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: no_res)
        return rs._model_memory_reload_required()

    @pytest.mark.parametrize(
        ("keep", "no_res", "state", "expected"),
        [
            # no_ram_reserve: neither reservation may survive, whoever asked.
            (False, True, (True, False), True),  # user --mlock still live
            (False, True, (False, True), True),  # user --no-mmap still live
            (False, True, (False, False), False),
            # keep_resident: satisfied by any mlock, including a user one.
            (True, False, (True, False), False),
            (True, False, (False, False), True),
            # Both off after the policy changed the launch: it has to be undone.
            (False, False, (True, True), True),
            (False, False, (False, False), True),
        ],
    )
    def test_matrix(self, monkeypatch, keep, no_res, state, expected):
        assert self._required(keep, no_res, state, monkeypatch) is expected

    def test_no_model_loaded_never_asks_for_a_reload(self, monkeypatch):
        assert (
            self._required(
                False, True, (False, True), monkeypatch, is_loaded = False, is_active = False
            )
            is False
        )

    def test_a_save_during_startup_still_asks_for_a_reload(self, monkeypatch):
        """The child is spawned and committed to its flags well before the
        health check flips is_loaded; keying on that reported no reload while
        the process was already coming up on the pre-save placement."""
        assert (
            self._required(False, True, (True, False), monkeypatch, is_loaded = False, is_active = True)
            is True
        )

    def test_a_skipped_mlock_is_not_a_permanent_reload_prompt(self, monkeypatch):
        import routes.inference
        import routes.settings as rs
        import utils.model_memory_settings as mm

        backend = type(
            "_B",
            (),
            {
                "is_loaded": True,
                "is_active": True,
                "_memory_state": (False, False),
                "_memory_policy_active": True,
                "_memory_mlock_applicable": False,
            },
        )()
        monkeypatch.setattr(routes.inference, "get_llama_cpp_backend", lambda: backend)
        monkeypatch.setattr(mm, "get_keep_resident", lambda: True)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
        assert rs._model_memory_reload_required() is False


class TestManagedFlagIsNotReset:
    """Measured against llama.cpp: ANY trailing mmap-family or load-mode flag
    resets the whole load mode, so a user preset after the managed flag would
    silently drop the mlock."""

    @pytest.mark.parametrize("load_mode", [True, False])
    @pytest.mark.parametrize(
        "preset",
        [
            ["--no-mmap"],
            ["--mmap"],
            ["-no-mmap"],
            ["-lm", "none"],
            ["--load-mode", "mmap"],
            ["--load-mode=none"],
            ["--mlock"],
            # Deprecated load-mode selectors: measured to reset the mode in BOTH
            # polarities, so all four spellings must go.
            ["--direct-io"],
            ["-dio"],
            ["--no-direct-io"],
            ["-ndio"],
        ],
    )
    def test_nothing_after_the_managed_flag_can_reset_it(self, policy, load_mode, preset):
        managed, extras = policy(
            True, False, preset + ["--temp", "0.7"], supports_load_mode = load_mode
        )
        assert managed  # residency emitted something
        # The resolved state of the full argv must still be mlock.
        mlock, _ = resolve_effective_memory_state(managed + extras, {})
        assert mlock is True, (managed, extras)
        assert extras == ["--temp", "0.7"], extras

    def test_affirmative_mmap_survives_when_nothing_is_managed(self, policy):
        """--mmap is not a reservation, so no-reserve leaves it alone."""
        _, extras = policy(False, True, ["--mmap", "--temp", "0.7"])
        assert extras == ["--mmap", "--temp", "0.7"]


class TestDuplicateLoadComparator:
    """Toggling a setting changes only the launch flags, so the load intent is
    unchanged and the already-loaded fast path would otherwise reuse the
    process and never apply the setting."""

    @pytest.mark.parametrize(
        ("launched", "policy_active", "keep", "no_res", "satisfied"),
        [
            ((False, False), False, True, False, False),  # turn residency on
            ((True, False), True, False, True, False),  # no-reserve on, mlocked
            ((False, True), False, False, True, False),  # no-reserve on, no-mmap
            ((True, False), True, True, False, True),  # already pinned
            ((False, False), False, False, True, True),  # already clean
            # Both off: anything the policy did must be undone, but a launch it
            # never touched is left alone.
            ((True, False), True, False, False, False),  # our flag still live
            ((True, False), False, False, False, True),  # user's own flag
            ((False, False), True, False, False, False),  # it suppressed theirs
            ((False, False), False, False, False, True),
            # DirectIO is not a RAM reservation, so no-reserve is satisfied.
            ((False, False), False, False, True, True),
            # A process this policy does not govern (diffusion) always matches,
            # else every identical /load would tear it down and reload.
            (None, False, True, False, True),
            (None, False, False, True, True),
            (None, True, True, True, True),
        ],
    )
    def test_forces_a_reload_only_when_the_policy_changed(
        self, monkeypatch, launched, policy_active, keep, no_res, satisfied
    ):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: keep)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: no_res)
        assert memory_state_satisfies_settings(launched, policy_active) is satisfied


class TestCapabilityProbeFallback:
    def test_load_mode_flag_survives_a_failed_probe(self, monkeypatch):
        """A timed-out or broken --help probe must fall back conservatively,
        not raise UnboundLocalError and block the load."""
        import subprocess

        from core.inference.llama_cpp import LlamaCppBackend

        LlamaCppBackend._capability_cache.clear()

        def boom(*_a, **_k):
            raise subprocess.TimeoutExpired("llama-server", 10)

        monkeypatch.setattr(subprocess, "run", boom)
        caps = LlamaCppBackend.probe_server_capabilities("/nonexistent/llama-server")
        assert caps.get("supports_load_mode") is False

    def test_an_ungoverned_process_never_asks_for_a_reload(self, monkeypatch):
        """A diffusion GGUF has no llama-server load-mode, so nothing about it
        can contradict the settings."""
        import routes.inference
        import routes.settings as rs
        import utils.model_memory_settings as mm

        backend = type(
            "_B",
            (),
            {"is_loaded": True, "_memory_state": None, "_memory_policy_active": False},
        )()
        monkeypatch.setattr(routes.inference, "get_llama_cpp_backend", lambda: backend)
        monkeypatch.setattr(mm, "get_keep_resident", lambda: True)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
        assert rs._model_memory_reload_required() is False


class TestCacheInvalidationRace:
    """A read that began before a write must not refill the memo cache with the
    value it already fetched: the new setting would appear to revert for the
    rest of the TTL, and a load could launch flags contradicting it."""

    def test_stale_fill_is_dropped(self, monkeypatch):
        import threading

        import utils.model_memory_settings as mm

        mm._cache.clear()
        store = {mm.KEEP_RESIDENT_SETTING_KEY: False}
        gate = threading.Event()
        reading = threading.Event()
        slow = {}

        def get_app_setting(key, fallback = None):
            # SELECT first, then stall, so the reader holds the OLD value.
            value = store.get(key, fallback)
            if slow.get("ident") == threading.get_ident():
                reading.set()
                gate.wait(2)
            return value

        monkeypatch.setattr("storage.studio_db.get_app_setting", get_app_setting)
        monkeypatch.setattr(
            "storage.studio_db.upsert_app_settings", lambda updates: store.update(updates)
        )

        def reader():
            slow["ident"] = threading.get_ident()
            mm.get_keep_resident()

        thread = threading.Thread(target = reader)
        thread.start()
        assert reading.wait(5), "reader never reached the DB"
        mm.set_model_memory_settings(keep_resident = True)
        gate.set()
        thread.join(5)
        assert not thread.is_alive()

        assert mm.get_keep_resident() is True

    def test_an_uncontended_read_still_caches(self, monkeypatch):
        import utils.model_memory_settings as mm

        mm._cache.clear()
        calls = []

        def get_app_setting(key, fallback = None):
            calls.append(key)
            return True

        monkeypatch.setattr("storage.studio_db.get_app_setting", get_app_setting)
        assert mm.get_keep_resident() is True
        assert mm.get_keep_resident() is True
        assert len(calls) == 1, "the guard must not disable caching"


class TestHostResidencyGate:
    """mlock pins host RAM. When the weights are fully offloaded to a discrete
    GPU there is nothing in host RAM worth pinning, so asking for it would
    reserve RAM for a copy that is not there."""

    def test_discrete_full_offload_does_not_page_lock(self, policy):
        managed, out = policy(
            True,
            False,
            ["--temp", "0.7"],
            supports_load_mode = True,
            weights_in_host_memory = False,
        )
        assert managed == []
        assert out == ["--temp", "0.7"]

    def test_unified_memory_or_partial_offload_still_page_locks(self, policy):
        managed, out = policy(
            True,
            False,
            ["--temp", "0.7"],
            supports_load_mode = True,
            weights_in_host_memory = True,
        )
        assert managed == ["--load-mode", "mmap+mlock"]
        assert out == ["--temp", "0.7"]

    def test_the_gate_does_not_leak_into_the_legacy_flag_path(self, policy):
        managed, _ = policy(True, False, [], supports_load_mode = False, weights_in_host_memory = False)
        assert managed == []
        managed, _ = policy(True, False, [], supports_load_mode = False, weights_in_host_memory = True)
        assert managed == ["--mlock"]

    def test_no_ram_reserve_still_strips_a_user_flag_off_a_discrete_gpu(self, policy):
        # The gate only suppresses what the policy ADDS. Removal is unchanged.
        managed, out = policy(
            False, True, ["--mlock", "--temp", "0.7"], weights_in_host_memory = False
        )
        assert managed == []
        assert out == ["--temp", "0.7"]

    def test_both_off_is_still_a_pass_through_either_way(self, policy):
        for host in (True, False):
            extras = ["--mlock", "--no-mmap", "--temp", "0.7"]
            managed, out = policy(False, False, extras, weights_in_host_memory = host)
            assert managed == []
            assert out == extras


class TestRacingReadReturnsTheNewValue:
    """The load path uses the returned value directly, so a read that raced a
    write must not hand back the pre-write setting."""

    def test_a_read_invalidated_mid_flight_is_retried(self, monkeypatch):
        import threading

        import utils.model_memory_settings as mm

        mm._cache.clear()
        store = {mm.KEEP_RESIDENT_SETTING_KEY: False}
        gate = threading.Event()
        reading = threading.Event()
        slow = {}
        stalled = {"done": False}

        def get_app_setting(key, fallback = None):
            value = store.get(key, fallback)
            # Stall the first read only; the retry must see the committed write.
            if slow.get("ident") == threading.get_ident() and not stalled["done"]:
                stalled["done"] = True
                reading.set()
                gate.wait(2)
                return value
            return value

        monkeypatch.setattr("storage.studio_db.get_app_setting", get_app_setting)
        monkeypatch.setattr(
            "storage.studio_db.upsert_app_settings", lambda updates: store.update(updates)
        )

        seen = {}

        def reader():
            slow["ident"] = threading.get_ident()
            seen["value"] = mm.get_keep_resident()

        thread = threading.Thread(target = reader)
        thread.start()
        assert reading.wait(5), "reader never reached the DB"
        mm.set_model_memory_settings(keep_resident = True)
        gate.set()
        thread.join(5)
        assert not thread.is_alive()

        assert seen["value"] is True, "the racing reader served the pre-write value"

    def test_a_write_storm_cannot_spin_forever(self, monkeypatch):
        import utils.model_memory_settings as mm

        mm._cache.clear()
        reads = []

        def get_app_setting(key, fallback = None):
            reads.append(key)
            mm._invalidate(key)  # a write lands during every read
            return True

        monkeypatch.setattr("storage.studio_db.get_app_setting", get_app_setting)
        assert mm.get_keep_resident() is True
        assert len(reads) == mm._MAX_REREADS

    def test_an_unreadable_db_falls_back_to_the_default(self, monkeypatch):
        import utils.model_memory_settings as mm

        mm._cache.clear()

        def boom(key, fallback = None):
            raise RuntimeError("database is locked")

        monkeypatch.setattr("storage.studio_db.get_app_setting", boom)
        assert mm.get_keep_resident() is mm.DEFAULT_KEEP_RESIDENT
        assert mm.get_no_ram_reserve() is mm.DEFAULT_NO_RAM_RESERVE
        assert mm.should_mlock() is False


class TestMlockApplicability:
    """A launch that deliberately skips mlock (full offload to a discrete GPU)
    still satisfies residency. Demanding the flag would ask for a reload no
    relaunch could satisfy, and would reject every duplicate load forever."""

    def test_a_skipped_mlock_satisfies_keep_resident(self, monkeypatch):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: True)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
        state = resolve_effective_memory_state([], {})
        assert state == (False, False)
        # The regression: without the applicability flag this reads as unsatisfied.
        assert memory_state_satisfies_settings(state, True, True) is False
        assert memory_state_satisfies_settings(state, True, False) is True

    def test_page_lockable_launches_are_unchanged(self, monkeypatch):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: True)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
        unpinned = resolve_effective_memory_state([], {})
        pinned = resolve_effective_memory_state(["--load-mode", "mmap+mlock"], {})
        assert memory_state_satisfies_settings(unpinned, True) is False
        assert memory_state_satisfies_settings(pinned, True) is True

    def test_applicability_does_not_override_no_ram_reserve(self, monkeypatch):
        """no-reserve still wins: a pinned process must be relaunched even where
        mlock would not have been applicable."""
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: True)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: True)
        state = resolve_effective_memory_state(["--mlock"], {})
        assert memory_state_satisfies_settings(state, True, False) is False

    def test_applicability_is_ignored_with_both_toggles_off(self, monkeypatch):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: False)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
        state = resolve_effective_memory_state([], {})
        assert memory_state_satisfies_settings(state, False, False) is True
        assert memory_state_satisfies_settings(state, True, False) is False

    def test_an_ungoverned_process_still_always_matches(self, monkeypatch):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: True)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
        assert memory_state_satisfies_settings(None, True, False) is True

    def test_the_default_keeps_the_old_meaning(self, monkeypatch):
        """Callers that never pass the flag behave exactly as before."""
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: True)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
        assert memory_state_satisfies_settings((False, False), True) is False
        assert memory_state_satisfies_settings((True, False), True) is True


class TestFullOffloadDetection:
    """``fully_gpu_offloaded`` is set only by the auto branch, so the mlock gate
    has to derive manual mode and a user -ngl for itself."""

    @staticmethod
    def _backend(n_layers, n_cpu_moe = 0):
        from core.inference.llama_cpp import LlamaCppBackend
        return type(
            "_B",
            (),
            {
                "n_layers": n_layers,
                "_n_cpu_moe": n_cpu_moe,
                "_offloads_every_layer": LlamaCppBackend._offloads_every_layer,
            },
        )()

    def _check(
        self,
        n_layers,
        mode,
        layers,
        extras = None,
        n_cpu_moe = 0,
    ):
        backend = self._backend(n_layers, n_cpu_moe)
        return backend._offloads_every_layer(
            gpu_memory_mode = mode, gpu_layers = layers, extra_args = extras
        )

    def test_manual_at_the_pickers_maximum_is_a_full_offload(self):
        # The slider's maximum is block_count + 1.
        assert self._check(32, "manual", 33) is True

    def test_manual_short_of_the_maximum_leaves_layers_on_the_host(self):
        assert self._check(32, "manual", 32) is False
        assert self._check(32, "manual", 31) is False
        assert self._check(32, "manual", 0) is False

    def test_manual_with_cpu_experts_is_not_a_full_offload(self):
        assert self._check(32, "manual", 33, n_cpu_moe = 4) is False

    def test_a_user_ngl_override_counts_in_auto_mode(self):
        assert self._check(32, "auto", None, ["-ngl", "99"]) is True
        assert self._check(32, "auto", None, ["--gpu-layers", "33"]) is True
        assert self._check(32, "auto", None, ["-ngl", "-1"]) is True
        assert self._check(32, "auto", None, ["-ngl", "16"]) is False

    def test_a_tensor_override_keeps_weights_on_the_host(self):
        for flag in ("-ot", "--override-tensor", "-cmoe", "--cpu-moe"):
            assert self._check(32, "auto", None, ["-ngl", "99", flag, "x"]) is False

    def test_every_unknown_answers_no(self):
        # No block count, no extras, unparseable -ngl: all keep the old behaviour
        # of page-locking rather than guessing a full offload.
        assert self._check(None, "manual", 33) is False
        assert self._check(0, "manual", 33) is False
        assert self._check(32, "auto", None, None) is False
        assert self._check(32, "auto", None, []) is False
        assert self._check(32, "auto", None, ["-ngl", "abc"]) is False

    def test_manual_auto_layers_falls_back_to_the_extras(self):
        # gpu_layers < 0 means manual did not pin a count.
        assert self._check(32, "manual", -1, ["-ngl", "99"]) is True
        assert self._check(32, "manual", -1, []) is False

    def test_the_manual_branch_really_does_not_set_fully_gpu_offloaded(self):
        """Pins the premise. If a later change starts setting it there, this
        test fails and the derived check can be simplified away."""
        import ast
        import inspect
        import textwrap

        from core.inference.llama_cpp import LlamaCppBackend

        source = textwrap.dedent(inspect.getsource(LlamaCppBackend.load_model))
        tree = ast.parse(source)
        manual_branches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and 'gpu_memory_mode == "manual" and gpu_layers >= 0'
            in ast.unparse(node.test).replace("'", '"')
        ]
        assert manual_branches, "the manual offload branch moved"
        # Only the branch body: its orelse holds the auto branch, which is the
        # one place that does set the flag.
        assigned = {
            target.id
            for branch in manual_branches
            for statement in branch.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert "fully_gpu_offloaded" not in assigned


class TestNegativeDirectIoIsARamReservation:
    """Upstream maps -ndio / --no-direct-io to mode `none`, like --no-mmap, so
    they hold a full host buffer. Calling them non-reserving let no-reserve
    leave one in the argv and still report the process compliant."""

    @pytest.mark.parametrize("flag", ["--no-direct-io", "-ndio", "--no_direct_io"])
    def test_the_negative_spellings_reserve_ram(self, flag):
        assert resolve_effective_memory_state([flag], {}) == (False, True)

    @pytest.mark.parametrize("flag", ["--direct-io", "-dio"])
    def test_the_affirmative_spellings_do_not(self, flag):
        assert resolve_effective_memory_state([flag], {}) == (False, False)

    def test_it_matches_no_mmap_in_both_orderings(self):
        for negative in ("--no-direct-io", "-ndio"):
            assert resolve_effective_memory_state(["--mlock", negative], {}) == (False, True)
            assert resolve_effective_memory_state([negative, "--mlock"], {}) == (True, True)

    @pytest.mark.parametrize("flag", ["--no-direct-io", "-ndio"])
    def test_no_ram_reserve_strips_them(self, policy, flag):
        managed, out = policy(False, True, [flag, "--temp", "0.7"])
        assert managed == []
        assert out == ["--temp", "0.7"]

    @pytest.mark.parametrize("flag", ["--direct-io", "-dio"])
    def test_no_ram_reserve_leaves_the_affirmative_ones(self, policy, flag):
        # DirectIO streams, so it is not a reservation and nothing has to go.
        _, out = policy(False, True, [flag, "--temp", "0.7"])
        assert out == [flag, "--temp", "0.7"]

    def test_the_comparator_now_sees_the_reservation(self, monkeypatch):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: False)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: True)
        state = resolve_effective_memory_state(["--no-direct-io"], {})
        assert memory_state_satisfies_settings(state, True) is False


class TestEnvVarsAssignTheWholeMode:
    """Each var runs its flag's handler, so it assigns the whole mode and a
    later one wins. Treating LLAMA_ARG_MMAP as a reserves_ram bit left mlock
    standing, so residency read an unlocked child as already satisfied."""

    @pytest.mark.parametrize(
        ("env", "expected"),
        [
            # Measured against the shipped binary: VmLck is 0 for both of these.
            ({"LLAMA_ARG_MLOCK": "1", "LLAMA_ARG_MMAP": "on"}, (False, False)),
            ({"LLAMA_ARG_MLOCK": "1", "LLAMA_ARG_DIO": "0"}, (False, True)),
            ({"LLAMA_ARG_MLOCK": "1", "LLAMA_ARG_DIO": "1"}, (False, False)),
            ({"LLAMA_ARG_MLOCK": "1", "LLAMA_ARG_MMAP": "off"}, (False, True)),
            # Registration order: load-mode is read last and wins outright.
            (
                {"LLAMA_ARG_MMAP": "off", "LLAMA_ARG_LOAD_MODE": "mmap+mlock"},
                (True, False),
            ),
            # Each still works alone.
            ({"LLAMA_ARG_MLOCK": "1"}, (True, False)),
            ({"LLAMA_ARG_DIO": "1"}, (False, False)),
            ({"LLAMA_ARG_DIO": "0"}, (False, True)),
            # An unset or unparseable value assigns nothing.
            ({"LLAMA_ARG_DIO": ""}, (False, False)),
            ({"LLAMA_ARG_MLOCK": "1", "LLAMA_ARG_MMAP": "banana"}, (True, False)),
        ],
    )
    def test_env_precedence(self, env, expected):
        assert resolve_effective_memory_state([], env) == expected

    def test_argv_still_beats_every_env_var(self):
        env = {"LLAMA_ARG_MLOCK": "1", "LLAMA_ARG_DIO": "0"}
        assert resolve_effective_memory_state(["--load-mode", "mmap+mlock"], env) == (True, False)

    def test_residency_is_not_reported_satisfied_against_an_unlocked_child(self, monkeypatch):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: True)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
        state = resolve_effective_memory_state([], {"LLAMA_ARG_MLOCK": "1", "LLAMA_ARG_MMAP": "on"})
        assert memory_state_satisfies_settings(state, False) is False


class TestMlockActiveReflectsWhatWillActuallyBePassed:
    """mlock_active drives the ulimit -l warning. Taking it from the toggles
    alone tells a discrete-GPU user to raise a limit nothing consults, since
    the gate suppresses the lock there."""

    @staticmethod
    def _response(keep, no_res, backend, monkeypatch):
        import routes.inference
        import routes.settings as rs
        import utils.model_memory_settings as mm

        monkeypatch.setattr(routes.inference, "get_llama_cpp_backend", lambda: backend)
        monkeypatch.setattr(mm, "get_keep_resident", lambda: keep)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: no_res)
        monkeypatch.setattr(rs, "get_model_memory_settings", lambda: (keep, no_res))
        monkeypatch.setattr(rs, "memlock_limit_bytes", lambda: 8 * 1024 * 1024)
        return rs._model_memory_response()

    @staticmethod
    def _backend(
        loaded,
        mlock_applicable,
        state = None,
    ):
        # A host-resident load with residency on carries the lock, so its state
        # says so; a gated one carries nothing. Keeping the two in step matters,
        # because the response reads the launched state rather than the flag.
        if state is None:
            state = (True, False) if mlock_applicable else (False, False)
        return type(
            "_B",
            (),
            {
                "is_loaded": loaded,
                "is_active": loaded,
                "_memory_state": state,
                "_memory_policy_active": False,
                "_memory_mlock_applicable": mlock_applicable,
                "_memory_launch_pending": False,
            },
        )()

    def test_a_gated_load_reports_no_active_lock_and_no_limit(self, monkeypatch):
        resp = self._response(True, False, self._backend(True, False), monkeypatch)
        assert resp.mlock_active is False
        assert resp.memlock_limit_bytes is None

    def test_a_host_resident_load_still_reports_the_limit(self, monkeypatch):
        resp = self._response(True, False, self._backend(True, True), monkeypatch)
        assert resp.mlock_active is True
        assert resp.memlock_limit_bytes == 8 * 1024 * 1024

    def test_with_nothing_loaded_the_intent_is_reported(self, monkeypatch):
        # No launch to read, so fall back to what the toggles ask for.
        resp = self._response(True, False, self._backend(False, False), monkeypatch)
        assert resp.mlock_active is True

    def test_no_ram_reserve_launch_reports_no_lock(self, monkeypatch):
        backend = self._backend(True, True, state = (False, False))
        resp = self._response(True, True, backend, monkeypatch)
        assert resp.mlock_active is False
        assert resp.memlock_limit_bytes is None


class TestHostMemoryGate:
    """The full gate, not just the layer count: CPU-placement extras and
    unified-memory devices both keep weights in pageable host RAM."""

    @staticmethod
    def _gate(
        monkeypatch,
        *,
        apple = False,
        amd = False,
        vulkan_igpu = False,
        **kwargs,
    ):
        import utils.hardware
        from core.inference.llama_cpp import LlamaCppBackend

        monkeypatch.setattr(utils.hardware, "is_apple_silicon", lambda: apple)
        monkeypatch.setattr(
            LlamaCppBackend, "_amd_apu_wants_unified_memory", staticmethod(lambda idx = None: amd)
        )
        monkeypatch.setattr(
            LlamaCppBackend,
            "_vulkan_targets_are_igpus",
            staticmethod(lambda binary, idx = None: vulkan_igpu),
        )
        backend = type(
            "_B",
            (),
            {
                "n_layers": 32,
                "_n_cpu_moe": 0,
                "_weights_in_host_memory": LlamaCppBackend._weights_in_host_memory,
                "_offloads_every_layer": LlamaCppBackend._offloads_every_layer,
                "_amd_apu_wants_unified_memory": staticmethod(lambda idx = None: amd),
                "_vulkan_targets_are_igpus": staticmethod(lambda binary, idx = None: vulkan_igpu),
            },
        )()
        params = {
            "fully_gpu_offloaded": False,
            "gpu_memory_mode": "auto",
            "gpu_layers": None,
            "extra_args": None,
        }
        params.update(kwargs)
        return backend._weights_in_host_memory(**params)

    def test_a_discrete_full_offload_is_not_host_resident(self, monkeypatch):
        assert self._gate(monkeypatch, fully_gpu_offloaded = True) is False

    def test_a_reserving_mode_promotes_full_offload_to_host_resident(self):
        settings = (True, False)
        managed, extras = apply_model_memory_policy(
            [],
            supports_load_mode = True,
            weights_in_host_memory = False,
            model_memory_settings = settings,
        )
        preview_mode, preview_extras = apply_load_mode_policy(
            extras,
            supports_load_mode = True,
            weights_in_host_memory = False,
            requested_load_mode = "none",
            model_memory_settings = settings,
        )
        assert resolve_effective_memory_state([*managed, *preview_mode, *preview_extras], {}) == (
            False,
            True,
        )

        managed, extras = apply_model_memory_policy(
            [],
            supports_load_mode = True,
            weights_in_host_memory = True,
            model_memory_settings = settings,
        )
        final_mode, final_extras = apply_load_mode_policy(
            extras,
            supports_load_mode = True,
            weights_in_host_memory = True,
            requested_load_mode = "none",
            model_memory_settings = settings,
        )
        assert [*managed, *final_mode, *final_extras] == ["--load-mode", "mmap+mlock"]

        import inspect
        from core.inference.llama_cpp import LlamaCppBackend

        source = inspect.getsource(LlamaCppBackend.load_model)
        start = source.index("_resolved_load_mode = load_mode or _fit_load_mode")
        end = source.index("_load_mode_managed, _mem_extras", start)
        block = source[start:end]
        assert "resolve_effective_memory_state(" in block
        assert "_mem_host_resident = True" in block
        assert "apply_model_memory_policy(" in block

    def test_a_partial_offload_is_host_resident(self, monkeypatch):
        assert self._gate(monkeypatch, fully_gpu_offloaded = False) is True

    @pytest.mark.parametrize("flag", ["-ncmoe", "--n-cpu-moe", "-cmoe", "--cpu-moe"])
    def test_cpu_moe_extras_survive_an_auto_full_offload(self, monkeypatch, flag):
        """The auto branch sets fully_gpu_offloaded, but an extra that pins
        experts on the CPU still leaves weights in RAM, so it must not be
        short-circuited past."""
        assert self._gate(monkeypatch, fully_gpu_offloaded = True, extra_args = [flag, "4"]) is True

    def test_a_tensor_override_survives_an_auto_full_offload(self, monkeypatch):
        assert (
            self._gate(
                monkeypatch,
                fully_gpu_offloaded = True,
                extra_args = ["--override-tensor", "blk.*=CPU"],
            )
            is True
        )

    @pytest.mark.parametrize("flag", ["-ngl", "--gpu-layers", "--n-gpu-layers"])
    @pytest.mark.parametrize("count", ["0", "8"])
    def test_a_pass_through_ngl_beats_the_auto_full_offload_prediction(
        self, monkeypatch, flag, count
    ):
        """Auto keeps the user's -ngl and appends it after ours, and llama.cpp is
        last-wins, so the prediction is void and the weights are in RAM."""
        assert self._gate(monkeypatch, fully_gpu_offloaded = True, extra_args = [flag, count]) is True

    @pytest.mark.parametrize("extras", [["--fit", "on"], ["--fit=on"], ["-fit", "1"]])
    def test_a_pass_through_fit_on_beats_the_auto_full_offload_prediction(
        self, monkeypatch, extras
    ):
        """--fit on re-enables the fitter, which may leave a prefix on the CPU."""
        assert self._gate(monkeypatch, fully_gpu_offloaded = True, extra_args = extras) is True

    @pytest.mark.parametrize("extras", [["--fit", "off"], ["-fit", "off"], ["--fit=off"]])
    def test_a_pass_through_fit_off_leaves_the_prediction_alone(self, monkeypatch, extras):
        """It restates what we already pass, and a disabled fitter cannot move
        anything to the CPU, so pinning here is the redundant host copy."""
        assert self._gate(monkeypatch, fully_gpu_offloaded = True, extra_args = extras) is False

    def test_the_fit_value_is_last_wins(self, monkeypatch):
        on_last = ["--fit", "off", "--fit", "on"]
        off_last = ["--fit", "on", "--fit", "off"]
        assert self._gate(monkeypatch, fully_gpu_offloaded = True, extra_args = on_last) is True
        assert self._gate(monkeypatch, fully_gpu_offloaded = True, extra_args = off_last) is False

    def test_a_pass_through_full_offload_still_skips_the_lock(self, monkeypatch):
        """The guard must not over-fire: -ngl above the block count is a real
        full offload, so page-locking a redundant host copy stays skipped."""
        assert self._gate(monkeypatch, fully_gpu_offloaded = True, extra_args = ["-ngl", "33"]) is False

    def test_apple_silicon_is_always_host_resident(self, monkeypatch):
        assert self._gate(monkeypatch, fully_gpu_offloaded = True, apple = True) is True

    def test_an_amd_unified_apu_is_always_host_resident(self, monkeypatch):
        assert self._gate(monkeypatch, fully_gpu_offloaded = True, amd = True) is True

    def test_a_vulkan_igpu_is_host_resident(self, monkeypatch):
        """An iGPU's reported VRAM is shared system RAM, which the repo already
        accounts for in the fit, so a full offload there is still pageable."""
        assert (
            self._gate(
                monkeypatch,
                fully_gpu_offloaded = True,
                is_vulkan_backend = True,
                vulkan_igpu = True,
            )
            is True
        )

    def test_a_discrete_vulkan_card_is_not(self, monkeypatch):
        assert (
            self._gate(
                monkeypatch,
                fully_gpu_offloaded = True,
                is_vulkan_backend = True,
                vulkan_igpu = False,
            )
            is False
        )

    def test_the_igpu_probe_is_not_consulted_off_vulkan(self, monkeypatch):
        """A CUDA install must not pay for the probe subprocess."""
        assert (
            self._gate(
                monkeypatch,
                fully_gpu_offloaded = True,
                is_vulkan_backend = False,
                vulkan_igpu = True,
            )
            is False
        )


class TestVulkanIgpuDetection:
    @staticmethod
    def _probe(monkeypatch, rows):
        from core.inference.llama_cpp import LlamaCppBackend
        monkeypatch.setattr(
            LlamaCppBackend, "_run_vulkan_probe", staticmethod(lambda binary = None: rows)
        )
        return LlamaCppBackend._vulkan_targets_are_igpus

    def test_all_igpus(self, monkeypatch):
        rows = [{"index": 0, "is_igpu": True}]
        assert self._probe(monkeypatch, rows)("bin", None) is True

    def test_a_mixed_set_still_has_host_weights(self, monkeypatch):
        """A split puts part of the model on the iGPU, whose VRAM is system RAM,
        so those pages are as evictable as if it were the only device."""
        rows = [{"index": 0, "is_igpu": True}, {"index": 1, "is_igpu": False}]
        assert self._probe(monkeypatch, rows)("bin", None) is True

    def test_only_the_selected_devices_count(self, monkeypatch):
        rows = [{"index": 0, "is_igpu": True}, {"index": 1, "is_igpu": False}]
        assert self._probe(monkeypatch, rows)("bin", [0]) is True
        assert self._probe(monkeypatch, rows)("bin", [1]) is False
        assert self._probe(monkeypatch, rows)("bin", [0, 1]) is True

    def test_discrete_only_stays_no(self, monkeypatch):
        rows = [{"index": 0, "is_igpu": False}, {"index": 1, "is_igpu": False}]
        assert self._probe(monkeypatch, rows)("bin", None) is False

    def test_an_unreadable_probe_answers_conservatively(self, monkeypatch):
        assert self._probe(monkeypatch, [])("bin", None) is True

    def test_an_unknown_device_type_answers_conservatively(self, monkeypatch):
        rows = [{"index": 0, "is_igpu": False, "type_known": False}]
        assert self._probe(monkeypatch, rows)("bin", None) is True

    def test_a_raising_probe_never_fails_the_load(self, monkeypatch):
        from core.inference.llama_cpp import LlamaCppBackend

        def boom(binary = None):
            raise OSError("no vulkan loader")

        monkeypatch.setattr(LlamaCppBackend, "_run_vulkan_probe", staticmethod(boom))
        assert LlamaCppBackend._vulkan_targets_are_igpus("bin", None) is True


class TestLegacyNegativeEnvAliases:
    """Upstream rewrites LLAMA_ARG_<NAME> to LLAMA_ARG_NO_<NAME> for any option
    with a negative form and, if that var EXISTS, forces the value falsey before
    reading the affirmative one (common/arg.cpp get_value_from_env). Confirmed
    against the shipped binary: NO_MMAP and NO_DIO fire their handler's
    deprecation warning even at "0", and NO_MLOCK does nothing."""

    @pytest.mark.parametrize("name", ["LLAMA_ARG_NO_MMAP", "LLAMA_ARG_NO_DIO"])
    @pytest.mark.parametrize("value", ["1", "0", "", "false", "anything"])
    def test_presence_alone_reserves_ram(self, name, value):
        assert resolve_effective_memory_state([], {name: value}) == (False, True)

    @pytest.mark.parametrize("name", ["LLAMA_ARG_NO_MMAP", "LLAMA_ARG_NO_DIO"])
    def test_the_negative_beats_its_own_affirmative(self, name):
        affirmative = name.replace("_NO_", "_")
        assert resolve_effective_memory_state([], {name: "0", affirmative: "on"}) == (False, True)

    def test_mlock_has_no_negative_form_so_the_alias_is_inert(self):
        assert resolve_effective_memory_state([], {"LLAMA_ARG_NO_MLOCK": "1"}) == (False, False)
        # And it cannot cancel the affirmative one.
        assert resolve_effective_memory_state(
            [], {"LLAMA_ARG_MLOCK": "1", "LLAMA_ARG_NO_MLOCK": "1"}
        ) == (True, False)

    def test_absence_does_not(self):
        assert resolve_effective_memory_state([], {}) == (False, False)

    @pytest.mark.parametrize("name", ["LLAMA_ARG_NO_MMAP", "LLAMA_ARG_NO_DIO"])
    def test_it_is_scrubbed_when_a_toggle_owns_placement(self, monkeypatch, name):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: False)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: True)
        env = {name: "1", "PATH": "/usr/bin"}
        assert name in scrub_memory_env(env)
        assert env == {"PATH": "/usr/bin"}

    def test_both_off_leaves_it_alone(self, monkeypatch):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: False)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
        env = {"LLAMA_ARG_NO_MMAP": "1", "LLAMA_ARG_NO_DIO": "1"}
        assert scrub_memory_env(env) == []
        assert env == {"LLAMA_ARG_NO_MMAP": "1", "LLAMA_ARG_NO_DIO": "1"}

    def test_argv_still_overrides_it(self):
        assert resolve_effective_memory_state(["--mmap"], {"LLAMA_ARG_NO_MMAP": "1"}) == (
            False,
            False,
        )


class TestNonReservingLoadModesSurvive:
    """No-reserve vetoes the reservation, not the loader: dio and mmap hold no
    full host copy, so a DirectIO preset must not silently become mmap."""

    @pytest.mark.parametrize("value", ["dio", "mmap"])
    def test_a_non_reserving_mode_is_kept(self, policy, value):
        _managed, out = policy(False, True, ["--load-mode", value, "--temp", "0.7"])
        assert out == ["--load-mode", value, "--temp", "0.7"]

    @pytest.mark.parametrize("value", ["none", "mlock", "mmap+mlock"])
    def test_a_reserving_or_locking_mode_is_dropped(self, policy, value):
        _managed, out = policy(False, True, ["--load-mode", value, "--temp", "0.7"])
        assert out == ["--temp", "0.7"]

    def test_the_attached_spelling_is_handled_too(self, policy):
        _managed, out = policy(False, True, ["--load-mode=mlock", "-c", "4096"])
        assert out == ["-c", "4096"]
        _managed, out = policy(False, True, ["--load-mode=dio", "-c", "4096"])
        assert out == ["--load-mode=dio", "-c", "4096"]

    def test_the_short_alias_is_handled_too(self, policy):
        _managed, out = policy(False, True, ["-lm", "mlock"])
        assert out == []
        _managed, out = policy(False, True, ["-lm", "dio"])
        assert out == ["-lm", "dio"]

    def test_an_unknown_value_is_left_alone(self, policy):
        _managed, out = policy(False, True, ["--load-mode", "future-mode"])
        assert out == ["--load-mode", "future-mode"]

    def test_a_trailing_flag_does_not_crash(self, policy):
        _managed, out = policy(False, True, ["--load-mode"])
        assert out == ["--load-mode"]

    def test_the_kept_mode_actually_satisfies_no_reserve(self, policy):
        """The point of keeping it: the resolver must agree it reserves nothing,
        or the reload hint would fire forever."""
        _managed, out = policy(False, True, ["--load-mode", "dio"])
        assert resolve_effective_memory_state(out, {}) == (False, False)

    def test_keep_resident_still_strips_every_mode(self, policy):
        """The mlock branch is unchanged: a trailing mode of ANY value resets
        the whole thing and would drop the managed lock."""
        managed, out = policy(True, False, ["--load-mode", "dio"], supports_load_mode = True)
        assert managed == ["--load-mode", "mmap+mlock"]
        assert out == []

    def test_a_suppressed_per_model_mode_marks_the_policy_active(self, monkeypatch):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_model_memory_settings", lambda: (False, True))
        assert model_memory_suppresses_load_mode(
            "none", supports_load_mode = True, weights_in_host_memory = True
        )
        assert not model_memory_suppresses_load_mode(
            "dio", supports_load_mode = True, weights_in_host_memory = True
        )
        assert not memory_state_satisfies_settings(
            (False, False), True, True, settings = (False, False)
        )

    def test_an_unsupported_mode_does_not_mark_the_policy_active(self, monkeypatch):
        import utils.model_memory_settings as mm
        monkeypatch.setattr(mm, "get_model_memory_settings", lambda: (True, False))
        assert not model_memory_suppresses_load_mode(
            "dio", supports_load_mode = False, weights_in_host_memory = True
        )

    def test_one_snapshot_drives_application_and_suppression(self, monkeypatch):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_model_memory_settings", lambda: (False, False))
        snapshot = (False, True)
        managed, extras = apply_load_mode_policy(
            [],
            supports_load_mode = True,
            weights_in_host_memory = True,
            requested_load_mode = "none",
            model_memory_settings = snapshot,
        )
        assert (managed, extras) == ([], [])
        assert model_memory_suppresses_load_mode(
            "none",
            supports_load_mode = True,
            weights_in_host_memory = True,
            model_memory_settings = snapshot,
        )


def _fake_backend(**attrs):
    base = {
        "is_loaded": False,
        "is_active": False,
        "_memory_state": None,
        "_memory_policy_active": False,
        "_memory_mlock_applicable": True,
        "_memory_launch_pending": False,
    }
    base.update(attrs)
    return type("_B", (), base)()


def _install_backend(monkeypatch, backend, *, keep, no_res):
    import routes.inference
    import utils.model_memory_settings as mm

    monkeypatch.setattr(routes.inference, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(mm, "get_keep_resident", lambda: keep)
    monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: no_res)
    monkeypatch.setattr(mm, "should_mlock", lambda: keep and not no_res)


class TestPreSpawnWindow:
    """The placement is fixed before Popen assigns _process, so is_active alone
    still leaves a window where a save reports no reload."""

    def test_a_save_before_the_child_spawns_still_asks_for_a_reload(self, monkeypatch):
        import routes.settings as rs

        backend = _fake_backend(
            is_active = False,
            _memory_launch_pending = True,
            _memory_state = (True, False),
            _memory_policy_active = True,
        )
        _install_backend(monkeypatch, backend, keep = False, no_res = True)
        assert rs._model_memory_reload_required() is True

    def test_nothing_running_still_never_asks(self, monkeypatch):
        import routes.settings as rs

        backend = _fake_backend(_memory_state = (True, False), _memory_policy_active = True)
        _install_backend(monkeypatch, backend, keep = False, no_res = True)
        assert rs._model_memory_reload_required() is False

    def test_a_pending_launch_that_matches_needs_no_reload(self, monkeypatch):
        import routes.settings as rs

        backend = _fake_backend(
            _memory_launch_pending = True,
            _memory_state = (True, False),
            _memory_policy_active = True,
        )
        _install_backend(monkeypatch, backend, keep = True, no_res = False)
        assert rs._model_memory_reload_required() is False


class TestMlockActiveReporting:
    """mlock_active drives the ulimit -l warning, so it has to describe the lock
    that was actually taken once something is running."""

    def test_with_nothing_loaded_it_reports_the_intent(self, monkeypatch):
        import routes.settings as rs

        _install_backend(monkeypatch, _fake_backend(), keep = True, no_res = False)
        body = rs._model_memory_response()
        assert body.mlock_active is True
        assert body.memlock_limit_bytes == mm_settings.memlock_limit_bytes()

    def test_a_diffusion_runner_reports_no_lock(self, monkeypatch):
        """It has no load-mode, so it never received a lock flag and warning
        about ulimit -l would be noise."""
        import routes.settings as rs

        backend = _fake_backend(is_active = True, is_loaded = True, _memory_state = None)
        _install_backend(monkeypatch, backend, keep = True, no_res = False)
        body = rs._model_memory_response()
        assert body.mlock_active is False
        assert body.memlock_limit_bytes is None
        assert body.keep_resident is True, "the toggle itself still reads back on"

    def test_a_skipped_lock_on_a_discrete_gpu_reports_no_lock(self, monkeypatch):
        import routes.settings as rs

        backend = _fake_backend(
            is_active = True,
            is_loaded = True,
            _memory_state = (False, False),
            _memory_mlock_applicable = False,
        )
        _install_backend(monkeypatch, backend, keep = True, no_res = False)
        assert rs._model_memory_response().mlock_active is False

    def test_a_lock_that_was_taken_reports_active(self, monkeypatch):
        import routes.settings as rs

        backend = _fake_backend(is_active = True, is_loaded = True, _memory_state = (True, False))
        _install_backend(monkeypatch, backend, keep = True, no_res = False)
        body = rs._model_memory_response()
        assert body.mlock_active is True
        assert body.memlock_limit_bytes == mm_settings.memlock_limit_bytes()

    def test_a_users_own_mlock_counts_as_a_real_lock(self, monkeypatch):
        """Keep resident on, full discrete offload so Unsloth emits nothing, but
        the user typed --mlock: the child IS locked, so say so."""
        import routes.settings as rs

        backend = _fake_backend(
            is_active = True,
            is_loaded = True,
            _memory_state = resolve_effective_memory_state(["--mlock"], {}),
            _memory_mlock_applicable = False,
        )
        _install_backend(monkeypatch, backend, keep = True, no_res = False)
        assert rs._model_memory_response().mlock_active is True

    def test_a_users_own_mlock_is_reported_with_the_toggle_off(self, monkeypatch):
        import routes.settings as rs

        backend = _fake_backend(
            is_active = True,
            is_loaded = True,
            _memory_state = resolve_effective_memory_state(["--mlock"], {}),
            _memory_policy_active = False,
        )
        _install_backend(monkeypatch, backend, keep = False, no_res = False)
        body = rs._model_memory_response()
        assert body.mlock_active is True
        assert body.memlock_limit_bytes == mm_settings.memlock_limit_bytes()

    def test_a_saved_veto_does_not_rewrite_the_running_child(self, monkeypatch):
        import routes.settings as rs

        backend = _fake_backend(
            is_active = True,
            is_loaded = True,
            _memory_state = (True, False),
            _memory_policy_active = True,
        )
        _install_backend(monkeypatch, backend, keep = True, no_res = True)
        body = rs._model_memory_response()
        assert body.mlock_active is True
        assert body.reload_required is True

    def test_no_reserve_launch_reports_no_lock(self, monkeypatch):
        import routes.settings as rs

        backend = _fake_backend(is_active = True, is_loaded = True, _memory_state = (False, False))
        _install_backend(monkeypatch, backend, keep = True, no_res = True)
        assert rs._model_memory_response().mlock_active is False


class TestMlockApplicable:
    """mlock_active alone cannot say WHY the lock is off, so the panel could not
    tell "you vetoed it" from "there was nothing to lock" and said nothing at all
    (issue #9549). This is the second bit that separates them."""

    def test_a_discrete_full_offload_reports_not_applicable(self, monkeypatch):
        import routes.settings as rs

        backend = _fake_backend(
            is_active = True,
            is_loaded = True,
            _memory_state = (False, False),
            _memory_mlock_applicable = False,
        )
        _install_backend(monkeypatch, backend, keep = True, no_res = False)
        body = rs._model_memory_response()
        assert body.mlock_applicable is False
        # The three fields the UI had to work from before, all unremarkable.
        assert body.keep_resident is True
        assert body.mlock_active is False
        assert body.reload_required is False

    def test_a_host_resident_launch_reports_applicable(self, monkeypatch):
        """A unified-memory APU or a partial offload: the lock does apply, and
        on a real gfx1151 host it is emitted (measured, issue #9549)."""
        import routes.settings as rs

        backend = _fake_backend(
            is_active = True,
            is_loaded = True,
            _memory_state = (True, False),
            _memory_mlock_applicable = True,
        )
        _install_backend(monkeypatch, backend, keep = True, no_res = False)
        assert rs._model_memory_response().mlock_applicable is True

    def test_with_nothing_loaded_it_is_applicable(self, monkeypatch):
        """No launch to describe, so the panel says nothing rather than claiming
        a placement it cannot know until something loads."""
        import routes.settings as rs

        _install_backend(monkeypatch, _fake_backend(), keep = True, no_res = False)
        assert rs._model_memory_response().mlock_applicable is True

    def test_a_diffusion_runner_reports_ungoverned(self, monkeypatch):
        import routes.settings as rs

        backend = _fake_backend(is_active = True, is_loaded = True, _memory_state = None)
        _install_backend(monkeypatch, backend, keep = True, no_res = False)
        body = rs._model_memory_response()
        assert body.mlock_applicable is False
        assert body.mlock_skip_reason == "ungoverned"

    def test_a_discrete_offload_reports_its_reason(self, monkeypatch):
        import routes.settings as rs

        backend = _fake_backend(
            is_active = True,
            is_loaded = True,
            _memory_state = (False, False),
            _memory_mlock_applicable = False,
        )
        _install_backend(monkeypatch, backend, keep = True, no_res = False)
        assert rs._model_memory_response().mlock_skip_reason == "full_gpu_offload"

    def test_a_reserving_load_mode_keeps_full_offload_applicable(self, monkeypatch):
        import routes.settings as rs

        backend = _fake_backend(
            is_active = True,
            is_loaded = True,
            _memory_state = (False, True),
            _memory_mlock_applicable = False,
        )
        _install_backend(monkeypatch, backend, keep = True, no_res = False)
        body = rs._model_memory_response()
        assert body.mlock_applicable is True
        assert body.mlock_skip_reason is None
        assert body.reload_required is True

    def test_an_inactive_llama_backend_defers_to_the_gpu_owner(self, monkeypatch):
        import core.inference.gpu_arbiter as arbiter
        import core.inference.orchestrator as orchestrator
        import routes.settings as rs

        _install_backend(monkeypatch, _fake_backend(), keep = True, no_res = False)
        monkeypatch.setattr(arbiter, "current_owner", lambda: arbiter.DIFFUSION)
        monkeypatch.setattr(orchestrator, "peek_inference_backend", lambda: None)
        body = rs._model_memory_response()
        assert body.mlock_applicable is False
        assert body.mlock_skip_reason == "ungoverned"
        assert body.mlock_active is False

    def test_an_active_transformers_backend_is_ungoverned(self, monkeypatch):
        import core.inference.gpu_arbiter as arbiter
        import core.inference.orchestrator as orchestrator
        import routes.settings as rs

        _install_backend(monkeypatch, _fake_backend(), keep = True, no_res = False)
        monkeypatch.setattr(arbiter, "current_owner", lambda: None)
        monkeypatch.setattr(
            orchestrator,
            "peek_inference_backend",
            lambda: type(
                "_Orchestrator", (), {"active_model_name": "model", "loading_models": []}
            )(),
        )
        body = rs._model_memory_response()
        assert body.mlock_applicable is False
        assert body.mlock_skip_reason == "ungoverned"

    def test_a_resident_stt_backend_is_ungoverned(self, monkeypatch):
        import core.inference.gpu_arbiter as arbiter
        import core.inference.orchestrator as orchestrator
        import routes.settings as rs

        _install_backend(monkeypatch, _fake_backend(), keep = True, no_res = False)
        monkeypatch.setattr(arbiter, "current_owner", lambda: None)
        monkeypatch.setattr(
            orchestrator,
            "peek_inference_backend",
            lambda: type(
                "_Orchestrator",
                (),
                {
                    "active_model_name": None,
                    "loading_models": [],
                    "resident_stt_model": lambda self: {"engine": "whisper.cpp"},
                },
            )(),
        )
        body = rs._model_memory_response()
        assert body.mlock_applicable is False
        assert body.mlock_skip_reason == "ungoverned"


class TestModelMemoryResponseSnapshot:
    def test_the_launch_placement_is_read_once(self, monkeypatch):
        import routes.settings as rs

        reads = 0

        def placement():
            nonlocal reads
            reads += 1
            return (False, False), False, False

        monkeypatch.setattr(rs, "_active_launch_placement", placement)
        monkeypatch.setattr(rs, "get_model_memory_settings", lambda: (True, False))
        body = rs._model_memory_response()
        assert reads == 1
        assert body.mlock_active is False
        assert body.mlock_applicable is False
        assert body.reload_required is False

    def test_lock_intent_comes_from_the_same_settings_pair(self, monkeypatch):
        import routes.settings as rs

        monkeypatch.setattr(rs, "get_model_memory_settings", lambda: (True, False))
        monkeypatch.setattr(
            rs,
            "_active_launch_placement",
            lambda: (rs._NO_LAUNCH, False, True),
        )
        body = rs._model_memory_response()
        assert body.keep_resident is True
        assert body.no_ram_reserve is False
        assert body.mlock_active is True

    def test_reload_uses_the_same_settings_pair(self, monkeypatch):
        import routes.settings as rs
        import utils.model_memory_settings as mm

        monkeypatch.setattr(rs, "get_model_memory_settings", lambda: (True, False))
        monkeypatch.setattr(
            rs,
            "_active_launch_placement",
            lambda: ((False, False), False, True),
        )

        def stale_getter():
            raise AssertionError("the response must not re-read settings")

        monkeypatch.setattr(mm, "get_model_memory_settings", stale_getter)
        body = rs._model_memory_response()
        assert body.keep_resident is True
        assert body.no_ram_reserve is False
        assert body.reload_required is True


class TestRecordedMemoryState:
    def test_every_snapshot_resolves_the_command_that_runs(self):
        import ast
        import inspect
        import textwrap

        from core.inference.llama_cpp import LlamaCppBackend

        tree = ast.parse(textwrap.dedent(inspect.getsource(LlamaCppBackend.load_model)))
        recorded_from = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Attribute) and target.attr == "_memory_state"
                for target in node.targets
            ):
                continue
            call = node.value
            assert isinstance(call, ast.Call)
            assert isinstance(call.func, ast.Name)
            assert call.func.id == "resolve_effective_memory_state"
            assert isinstance(call.args[0], ast.Name)
            recorded_from.append(call.args[0].id)

        assert {"_run", "cmd", "replay", "run_cmd", "_last_spawn_cmd"}.issubset(recorded_from)


class TestFitOnRetryReArmsResidency:
    """The --fit on fallback fires exactly when the full-offload prediction that
    suppressed the lock turns out to be wrong, so the retry must re-apply it."""

    @staticmethod
    def _retry_argv(original, *, supports_load_mode = True):
        """What the fallback builds: flip --fit, then append the managed flag.

        Appending (rather than inserting before the extras) is measured against
        the binary in the simulation suite: the last --load-mode wins.
        """
        run = list(original)
        if "--fit" in run:
            run[run.index("--fit") + 1] = "on"
        run.extend(["--load-mode", "mmap+mlock"] if supports_load_mode else ["--mlock"])
        return run

    def test_the_retry_is_page_locked(self):
        original = ["-ngl", "-1", "--fit", "off", "--temp", "0.7"]
        retry = self._retry_argv(original)
        assert resolve_effective_memory_state(original, {}) == (False, False)
        assert resolve_effective_memory_state(retry, {}) == (True, False)
        assert retry[:4] == ["-ngl", "-1", "--fit", "on"]

    def test_it_wins_over_a_user_load_mode_in_the_extras(self):
        original = ["--fit", "off", "--load-mode", "dio"]
        assert resolve_effective_memory_state(self._retry_argv(original), {}) == (True, False)

    def test_it_wins_over_a_user_no_mmap(self):
        original = ["--fit", "off", "--no-mmap"]
        assert resolve_effective_memory_state(self._retry_argv(original), {}) == (True, False)

    def test_the_legacy_flag_path_re_arms_too(self):
        original = ["--fit", "off"]
        retry = self._retry_argv(original, supports_load_mode = False)
        assert resolve_effective_memory_state(retry, {}) == (True, False)

    def test_the_re_armed_launch_satisfies_residency(self, monkeypatch):
        """Without this the retry would nag for a reload that cannot help."""
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: True)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
        state = resolve_effective_memory_state(self._retry_argv(["--fit", "off"]), {})
        assert memory_state_satisfies_settings(state, True, True) is True

    def test_a_missing_fit_flag_does_not_crash(self):
        assert resolve_effective_memory_state(self._retry_argv(["-ngl", "-1"]), {}) == (
            True,
            False,
        )


class TestTheRetryCanReadTheGate:
    """The re-arm above runs inside _spawn_and_wait but assigns
    _mem_host_resident, which is a load_model local. Without a nonlocal that
    assignment makes it local to _spawn_and_wait, so reading it first is an
    UnboundLocalError and the --fit on retry raises instead of retrying."""

    @staticmethod
    def _load_model_ast():
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"
        for node in ast.walk(ast.parse(src.read_text(encoding = "utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name == "load_model":
                return node
        raise AssertionError("load_model not found")

    def test_every_writer_of_the_gate_can_also_read_it(self):
        import ast
        outer = self._load_model_ast()
        for inner in ast.walk(outer):
            if not isinstance(inner, ast.FunctionDef) or inner is outer:
                continue
            writes = any(
                isinstance(n, ast.Name)
                and n.id == "_mem_host_resident"
                and isinstance(n.ctx, ast.Store)
                for n in ast.walk(inner)
            )
            if not writes:
                continue
            declared = any(
                isinstance(n, ast.Nonlocal) and "_mem_host_resident" in n.names
                for n in ast.walk(inner)
            )
            assert declared, (
                f"{inner.name} assigns _mem_host_resident without a nonlocal, so "
                f"reading it there raises UnboundLocalError"
            )


class TestInheritedCpuPlacement:
    """The child inherits these, so they outlive stripping the equivalent
    flags and keep weights in host RAM whatever the layer count says."""

    @pytest.mark.parametrize(
        "env,expected",
        [
            ({}, False),
            ({"LLAMA_ARG_OVERRIDE_TENSOR": "blk.*=CPU"}, True),
            ({"LLAMA_ARG_OVERRIDE_TENSOR": ""}, False),
            ({"LLAMA_ARG_CPU_MOE": "1"}, True),
            ({"LLAMA_ARG_CPU_MOE": "on"}, True),
            ({"LLAMA_ARG_CPU_MOE": "0"}, False),
            ({"LLAMA_ARG_N_CPU_MOE": "4"}, True),
            ({"LLAMA_ARG_N_CPU_MOE": "0"}, False),
            ({"LLAMA_ARG_N_CPU_MOE": "-1"}, False),
            ({"LLAMA_ARG_N_CPU_MOE": "nonsense"}, False),
        ],
    )
    def test_the_predicate(self, env, expected):
        from core.inference.llama_cpp import _env_places_tensors_on_cpu
        assert _env_places_tensors_on_cpu(env) is expected

    def test_it_is_the_same_predicate_the_pipeline_check_uses(self):
        """Shared, so the two cannot drift apart."""
        import inspect

        from core.inference.llama_cpp import _pipeline_parallel_disabled_by_args

        source = inspect.getsource(_pipeline_parallel_disabled_by_args)
        assert "_env_places_tensors_on_cpu" in source
        for name in ("LLAMA_ARG_OVERRIDE_TENSOR", "LLAMA_ARG_CPU_MOE", "LLAMA_ARG_N_CPU_MOE"):
            assert name not in source, f"{name} re-implemented instead of shared"

    @pytest.mark.parametrize(
        "env",
        [
            {"LLAMA_ARG_OVERRIDE_TENSOR": "blk.*=CPU"},
            {"LLAMA_ARG_CPU_MOE": "1"},
            {"LLAMA_ARG_N_CPU_MOE": "4"},
        ],
    )
    def test_an_inherited_placement_keeps_a_full_offload_host_resident(self, monkeypatch, env):
        assert (
            TestHostMemoryGate._gate(
                monkeypatch, fully_gpu_offloaded = True, extra_args = None, env = env
            )
            is True
        )

    def test_an_unset_environment_leaves_the_gate_alone(self, monkeypatch):
        assert (
            TestHostMemoryGate._gate(monkeypatch, fully_gpu_offloaded = True, extra_args = None, env = {})
            is False
        )


class TestCpuMoeCountIsParsed:
    """A CPU-MoE count places nothing at zero, which the env side already knew.
    Presence alone would page-lock an all-GPU load for a no-op flag."""

    @pytest.mark.parametrize(
        ("extras", "expected"),
        [
            (None, False),
            ([], False),
            (["--n-cpu-moe", "0"], False),
            (["-ncmoe", "0"], False),
            (["--n-cpu-moe=0"], False),
            (["--n-cpu-moe", "4"], True),
            (["-ncmoe", "4"], True),
            (["--n-cpu-moe=4"], True),
            (["--n-cpu-moe", "-1"], False),
            (["--n-cpu-moe", "nonsense"], False),
            (["--n-cpu-moe"], False),  # trailing, no value
            # No value to parse: presence is the whole signal.
            (["--cpu-moe"], True),
            (["-cmoe"], True),
            (["-ot", "blk.*=CPU"], True),
            (["--override-tensor", "x"], True),
        ],
    )
    def test_the_predicate(self, extras, expected):
        from core.inference.llama_cpp import _args_place_tensors_on_cpu
        assert _args_place_tensors_on_cpu(extras) is expected

    def test_it_is_the_same_predicate_the_pipeline_check_uses(self):
        import inspect

        from core.inference.llama_cpp import _pipeline_parallel_disabled_by_args

        import ast
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(_pipeline_parallel_disabled_by_args)))
        fn = tree.body[0]
        if ast.get_docstring(fn) is not None:
            fn.body = fn.body[1:]
        # unparse drops comments and the docstring, so only real code is left;
        # either would otherwise name the flags without being a second copy.
        code = ast.unparse(fn)
        assert "_args_place_tensors_on_cpu" in code
        for flag in ("-ncmoe", "--n-cpu-moe", "--override-tensor"):
            assert flag not in code, f"{flag} re-implemented instead of shared"

    def test_a_zero_count_no_longer_forces_a_page_lock(self, monkeypatch):
        assert (
            TestHostMemoryGate._gate(
                monkeypatch, fully_gpu_offloaded = True, extra_args = ["--n-cpu-moe", "0"]
            )
            is False
        )

    def test_a_real_count_still_does(self, monkeypatch):
        assert (
            TestHostMemoryGate._gate(
                monkeypatch, fully_gpu_offloaded = True, extra_args = ["--n-cpu-moe", "4"]
            )
            is True
        )


class TestManualModeIgnoresClearedEnv:
    """Manual mode strips its placement vars from the child env, so the gate
    must not pin for a setting the child is never going to see. It reads the
    reconciled env, built with the same helper the launch uses."""

    @staticmethod
    def _child_env(parent, gpu_memory_mode):
        from core.inference.llama_cpp import LlamaCppBackend

        env = dict(parent)
        if gpu_memory_mode == "manual":
            LlamaCppBackend._clear_manual_placement_env(env)
        return env

    @pytest.mark.parametrize("var", ["LLAMA_ARG_CPU_MOE", "LLAMA_ARG_N_CPU_MOE"])
    def test_manual_mode_drops_them_so_the_gate_ignores_them(self, monkeypatch, var):
        parent = {var: "4" if var.endswith("N_CPU_MOE") else "1"}
        env = self._child_env(parent, "manual")
        assert env == {}, "the launch clears these, so the gate must not see them"
        assert (
            TestHostMemoryGate._gate(
                monkeypatch, fully_gpu_offloaded = True, extra_args = None, env = env
            )
            is False
        )

    @pytest.mark.parametrize("var", ["LLAMA_ARG_CPU_MOE", "LLAMA_ARG_N_CPU_MOE"])
    def test_auto_mode_still_honours_them(self, monkeypatch, var):
        parent = {var: "4" if var.endswith("N_CPU_MOE") else "1"}
        env = self._child_env(parent, "auto")
        assert env == parent
        assert (
            TestHostMemoryGate._gate(
                monkeypatch, fully_gpu_offloaded = True, extra_args = None, env = env
            )
            is True
        )

    def test_override_tensor_is_not_cleared_by_manual_mode(self, monkeypatch):
        """It is absent from _MANUAL_PLACEMENT_ENV_VARS, so it DOES reach the
        child and must keep counting even in manual mode."""
        from core.inference.llama_cpp import LlamaCppBackend

        assert "LLAMA_ARG_OVERRIDE_TENSOR" not in LlamaCppBackend._MANUAL_PLACEMENT_ENV_VARS
        env = self._child_env({"LLAMA_ARG_OVERRIDE_TENSOR": "blk.*=CPU"}, "manual")
        assert env == {"LLAMA_ARG_OVERRIDE_TENSOR": "blk.*=CPU"}
        assert (
            TestHostMemoryGate._gate(
                monkeypatch, fully_gpu_offloaded = True, extra_args = None, env = env
            )
            is True
        )


class TestADefaultLaunchRecordsItsApplicability:
    """_memory_mlock_applicable is recorded from the gate, and a later "keep
    resident" save is compared against it. Skipping the gate when no lock was on
    the table recorded the placeholder True, so enabling the toggle on a discrete
    full offload demanded a reload and relaunched identical argv."""

    def test_the_gate_is_not_skipped_when_should_mlock_is_false(self):
        import ast

        outer = TestTheRetryCanReadTheGate._load_model_ast()
        parents = {}
        for node in ast.walk(outer):
            for child in ast.iter_child_nodes(node):
                parents[id(child)] = node
        calls = [
            n
            for n in ast.walk(outer)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_weights_in_host_memory"
        ]
        assert calls, "the gate call vanished"
        for call in calls:
            cur = parents.get(id(call))
            while cur is not None and cur is not outer:
                if isinstance(cur, ast.If):
                    named = {
                        f.func.id
                        for f in ast.walk(cur.test)
                        if isinstance(f, ast.Call) and isinstance(f.func, ast.Name)
                    }
                    assert "should_mlock" not in named, (
                        "the gate is behind should_mlock again, so a default "
                        "launch records a placeholder applicability"
                    )
                cur = parents.get(id(cur))

    def test_enabling_residency_after_a_default_full_offload_needs_no_reload(self, monkeypatch):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: True)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
        # What the gate records for a discrete full offload.
        assert memory_state_satisfies_settings((False, False), False, False) is True

    def test_a_partial_offload_still_demands_the_reload(self, monkeypatch):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: True)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
        assert memory_state_satisfies_settings((False, False), False, True) is False

    def test_the_bookkeeping_call_skips_the_vulkan_probe(self, monkeypatch):
        from core.inference.llama_cpp import LlamaCppBackend

        def boom(binary = None):
            raise AssertionError("the probe must not run for a bookkeeping call")

        monkeypatch.setattr(LlamaCppBackend, "_run_vulkan_probe", staticmethod(boom))
        assert (
            TestHostMemoryGate._gate(
                monkeypatch,
                fully_gpu_offloaded = True,
                is_vulkan_backend = True,
                probe_vulkan = False,
            )
            is True
        )

    def test_skipping_the_probe_leaves_non_vulkan_alone(self, monkeypatch):
        assert (
            TestHostMemoryGate._gate(
                monkeypatch,
                fully_gpu_offloaded = True,
                is_vulkan_backend = False,
                probe_vulkan = False,
            )
            is False
        )

    def test_the_fit_probe_snapshot_classifies_default_vulkan_launches(self, monkeypatch):
        assert (
            TestHostMemoryGate._gate(
                monkeypatch,
                fully_gpu_offloaded = True,
                is_vulkan_backend = True,
                probe_vulkan = False,
                known_vulkan_igpus = set(),
            )
            is False
        )
        assert (
            TestHostMemoryGate._gate(
                monkeypatch,
                fully_gpu_offloaded = True,
                is_vulkan_backend = True,
                probe_vulkan = False,
                known_vulkan_igpus = {0},
            )
            is True
        )

    def test_the_fit_probe_snapshot_only_classifies_targeted_vulkan_igpus(self, monkeypatch):
        assert (
            TestHostMemoryGate._gate(
                monkeypatch,
                fully_gpu_offloaded = True,
                is_vulkan_backend = True,
                gpu_indices = [1],
                probe_vulkan = False,
                known_vulkan_igpus = {0},
            )
            is False
        )
        assert (
            TestHostMemoryGate._gate(
                monkeypatch,
                fully_gpu_offloaded = True,
                is_vulkan_backend = True,
                gpu_indices = [0, 1],
                probe_vulkan = False,
                known_vulkan_igpus = {0},
            )
            is True
        )

    def test_an_unknown_vulkan_snapshot_reprobes_before_skipping_mlock(self, monkeypatch):
        from core.inference.llama_cpp import LlamaCppBackend

        assert (
            TestHostMemoryGate._gate(
                monkeypatch,
                fully_gpu_offloaded = True,
                is_vulkan_backend = True,
                probe_vulkan = True,
                known_vulkan_igpus = None,
                vulkan_igpu = True,
            )
            is True
        )

        import inspect

        source = inspect.getsource(LlamaCppBackend.load_model)
        assert "_shared_gpu_ids: Optional[set[int]] = None" in source

    def test_an_empty_vulkan_inventory_keeps_the_snapshot_unknown(self):
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        source = inspect.getsource(LlamaCppBackend.load_model)
        start = source.index("if is_vulkan_backend:", source.index("_detected_gpus = list(gpus)"))
        block = source[start : source.index("# The --fit fallback", start)]
        assert "if _gpu_mem" in block
        assert "else None" in block


class TestPairedWritesAreInvalidatedTogether:
    """The write commits both keys in one transaction, so a reader must never
    observe a combination that was never stored. Invalidating key by key let a
    load read the new keep_resident against a cached old no_ram_reserve and emit
    --mlock for a launch the user had just told not to reserve RAM."""

    def test_a_paired_write_invalidates_once_for_both_keys(self, monkeypatch):
        """Deterministic: the timing window itself is only a few instructions, so
        pin the contract instead of racing for it."""
        import utils.model_memory_settings as mm

        store: dict = {}
        monkeypatch.setattr(
            "storage.studio_db.get_app_setting", lambda key, default = None: store.get(key, default)
        )
        monkeypatch.setattr(
            "storage.studio_db.upsert_app_settings", lambda updates: store.update(updates)
        )
        calls: list[tuple] = []
        real = mm._invalidate
        monkeypatch.setattr(mm, "_invalidate", lambda *keys: (calls.append(keys), real(*keys))[1])

        mm.set_model_memory_settings(keep_resident = True, no_ram_reserve = True)

        assert len(calls) == 1, f"the pair must be dropped in one call, got {calls}"
        assert set(calls[0]) == {mm.KEEP_RESIDENT_SETTING_KEY, mm.NO_RAM_RESERVE_SETTING_KEY}

    def test_invalidating_a_pair_bumps_both_generations(self):
        import utils.model_memory_settings as mm

        mm._generation.clear()
        mm._invalidate(mm.KEEP_RESIDENT_SETTING_KEY, mm.NO_RAM_RESERVE_SETTING_KEY)
        assert mm._generation[mm.KEEP_RESIDENT_SETTING_KEY] == 1
        assert mm._generation[mm.NO_RAM_RESERVE_SETTING_KEY] == 1

    def test_one_acquisition_covers_every_key(self):
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "utils" / "model_memory_settings.py"
        tree = ast.parse(src.read_text(encoding = "utf-8"))
        setter = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "set_model_memory_settings"
        )
        for node in ast.walk(setter):
            if isinstance(node, ast.For):
                called = {
                    c.func.id
                    for c in ast.walk(node)
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                }
                assert (
                    "_invalidate" not in called
                ), "_invalidate is back in a loop, so the pair is not atomic"


class TestThePolicyReadsOneSnapshot:
    """apply_model_memory_policy decided stripping and locking from separate
    reads, so a save landing between them produced a launch for a pair that was
    never stored: no strip for the new no-reserve, and no lock either, leaving a
    saved --mlock in the extras."""

    def test_a_save_between_the_two_reads_is_not_observable(self, monkeypatch):
        import utils.model_memory_settings as mm

        # Start at (keep_resident=True, no_ram_reserve=False).
        store = {
            mm.KEEP_RESIDENT_SETTING_KEY: True,
            mm.NO_RAM_RESERVE_SETTING_KEY: False,
        }
        monkeypatch.setattr(
            "storage.studio_db.get_app_setting", lambda key, default = None: store.get(key, default)
        )
        monkeypatch.setattr(
            "storage.studio_db.upsert_app_settings", lambda updates: store.update(updates)
        )
        mm._cache.clear()
        mm._generation.clear()

        # Flip to (False, True) the moment keep_resident has been read.
        real = mm.get_keep_resident
        fired: list[bool] = []

        def flip_after_first_read():
            value = real()
            if not fired:
                fired.append(True)
                store[mm.KEEP_RESIDENT_SETTING_KEY] = False
                store[mm.NO_RAM_RESERVE_SETTING_KEY] = True
                mm._invalidate(mm.KEEP_RESIDENT_SETTING_KEY, mm.NO_RAM_RESERVE_SETTING_KEY)
            return value

        monkeypatch.setattr(mm, "get_keep_resident", flip_after_first_read)
        keep_resident, no_ram_reserve = mm.get_model_memory_settings()
        assert fired, "the write never landed, so this proves nothing"
        # (True, True) was never stored: the old pair or the new one, not a mix.
        assert (keep_resident, no_ram_reserve) in {(True, False), (False, True)}

    def test_the_policy_derives_both_from_one_call(self):
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_server_args.py"
        tree = ast.parse(src.read_text(encoding = "utf-8"))
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "apply_model_memory_policy"
        )
        named = [
            c.func.id
            for c in ast.walk(fn)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        ]
        assert named.count("get_model_memory_settings") == 1
        for separate in ("should_mlock", "get_no_ram_reserve", "get_keep_resident"):
            assert (
                separate not in named
            ), f"{separate} is read separately again, so the pair can tear"


class TestTheGateDoesNotOverFire:
    """Pinning where the weights are all on a discrete GPU is the redundant host
    copy this gate exists to avoid, so the two ways it could over-fire are
    pinned here: a zero CPU-MoE count, and the ROCm APU probe under Vulkan."""

    @pytest.mark.parametrize("flag", ["--n-cpu-moe", "-ncmoe"])
    def test_a_zero_cpu_moe_count_places_nothing(self, monkeypatch, flag):
        """_args_place_tensors_on_cpu already treats 0 as a no-op; the offload
        proof used flag presence, so the count flipped an all-GPU launch."""
        assert TestHostMemoryGate._gate(monkeypatch, extra_args = ["-ngl", "-1", flag, "0"]) is False

    @pytest.mark.parametrize("extras", [["-cmoe"], ["--n-cpu-moe", "4"], ["-ot", "exps=CPU"]])
    def test_real_cpu_placement_still_counts(self, monkeypatch, extras):
        assert TestHostMemoryGate._gate(monkeypatch, extra_args = ["-ngl", "-1", *extras]) is True

    def test_the_rocm_apu_probe_is_not_consulted_under_vulkan(self, monkeypatch):
        """gpu_indices are Vulkan ordinals there, which the ROCm helper would
        read as physical ids and answer for a different device."""
        assert (
            TestHostMemoryGate._gate(
                monkeypatch,
                extra_args = ["-ngl", "-1"],
                is_vulkan_backend = True,
                amd = True,
                vulkan_igpu = False,
            )
            is False
        )

    def test_the_rocm_apu_probe_still_decides_off_vulkan(self, monkeypatch):
        assert TestHostMemoryGate._gate(monkeypatch, extra_args = ["-ngl", "-1"], amd = True) is True


class TestResidencyDoesNotBlockReload:
    """Residency stops UNLOADS, not reloads. A model the idle loop already freed
    must still come back on the next request, and then stay resident."""

    @pytest.fixture
    def idle_env(self, monkeypatch):
        """Standalone UNSLOTH_MODEL_IDLE_TTL with auto-switch off."""
        import utils.openai_auto_switch_settings as aus

        monkeypatch.setattr(aus, "_stored_idle_seconds", lambda: None)
        monkeypatch.setattr(aus, "_env_idle_seconds", lambda: 300)
        monkeypatch.setattr(aus, "get_openai_auto_switch_enabled", lambda: False)

        def residency(on):
            import utils.model_memory_settings as mm
            monkeypatch.setattr(mm, "get_keep_resident", lambda: on)

        return residency

    def test_the_effective_ttl_is_still_vetoed(self, idle_env):
        import utils.openai_auto_switch_settings as aus

        idle_env(False)
        assert aus.get_auto_unload_idle_seconds() == 300
        idle_env(True)
        assert aus.get_auto_unload_idle_seconds() == 0, "the veto must still apply"

    def test_but_idle_unload_is_still_configured(self, idle_env):
        import utils.openai_auto_switch_settings as aus
        idle_env(True)
        assert aus.idle_unload_is_configured() is True

    def test_an_automatic_load_may_still_run_under_residency(self, idle_env):
        """The regression: this gated on the effective TTL, so enabling residency
        made the next request 400 instead of reloading the freed model."""
        import routes.inference as ri

        idle_env(False)
        assert ri._automatic_model_load_may_run() is True
        idle_env(True)
        assert ri._automatic_model_load_may_run() is True

    def test_turning_idle_unload_off_still_disables_the_reload_path(self, monkeypatch):
        """The converse: no TTL and no auto-switch means no automatic load, with
        or without residency, so this is not just always-true."""
        import routes.inference as ri
        import utils.model_memory_settings as mm
        import utils.openai_auto_switch_settings as aus

        monkeypatch.setattr(aus, "_stored_idle_seconds", lambda: None)
        monkeypatch.setattr(aus, "_env_idle_seconds", lambda: None)
        monkeypatch.setattr(aus, "get_openai_auto_switch_enabled", lambda: False)
        for resident in (False, True):
            monkeypatch.setattr(mm, "get_keep_resident", lambda: resident)
            assert ri._automatic_model_load_may_run() is False

    def test_a_stored_ttl_still_needs_auto_switch_on(self, monkeypatch):
        """The configured reader keeps the same auto-switch gating as the
        effective one, so swapping it in changes nothing but the veto."""
        import utils.model_memory_settings as mm
        import utils.openai_auto_switch_settings as aus

        monkeypatch.setattr(aus, "_stored_idle_seconds", lambda: 300)
        monkeypatch.setattr(aus, "_env_idle_seconds", lambda: None)
        monkeypatch.setattr(mm, "get_keep_resident", lambda: False)
        monkeypatch.setattr(aus, "get_openai_auto_switch_enabled", lambda: False)
        assert aus.idle_unload_is_configured() is False
        assert aus.get_auto_unload_idle_seconds() == 0
        monkeypatch.setattr(aus, "get_openai_auto_switch_enabled", lambda: True)
        assert aus.idle_unload_is_configured() is True
        assert aus.get_auto_unload_idle_seconds() == 300

    def test_the_two_readers_agree_except_on_residency(self, monkeypatch):
        """Pins the substitution itself across the whole input space."""
        import utils.model_memory_settings as mm
        import utils.openai_auto_switch_settings as aus

        monkeypatch.setattr(mm, "get_keep_resident", lambda: False)
        for stored in (None, 0, 300):
            for env in (None, 0, 300):
                for switch in (False, True):
                    monkeypatch.setattr(aus, "_stored_idle_seconds", lambda s = stored: s)
                    monkeypatch.setattr(aus, "_env_idle_seconds", lambda e = env: e)
                    monkeypatch.setattr(aus, "get_openai_auto_switch_enabled", lambda v = switch: v)
                    assert aus.idle_unload_is_configured() == (
                        aus.get_auto_unload_idle_seconds() > 0
                    ), (stored, env, switch)

    def test_the_idle_loop_itself_still_reads_the_vetoed_value(self):
        """Scheduling keeps the veto; only the reload-capability checks moved."""
        import inspect

        from core.inference import llama_keepwarm

        source = inspect.getsource(llama_keepwarm)
        assert "get_auto_unload_idle_seconds" in source
        assert "idle_unload_is_configured" not in source


class TestACpuDevicePinIsHostResident:
    """--device cpu/none leaves llama.cpp nowhere to offload to, so the model
    stays in host RAM whatever the layer count predicted. Unlike a silent
    fallback this is knowable before launch, so the gate can just read it."""

    @pytest.mark.parametrize(
        "extras",
        [
            ["--device", "cpu"],
            ["--device", "none"],
            ["-dev", "cpu"],
            ["--device=none"],
            ["--device", "cpu,none"],
        ],
    )
    def test_a_cpu_pin_beats_the_offload_prediction(self, monkeypatch, extras):
        assert (
            TestHostMemoryGate._gate(monkeypatch, fully_gpu_offloaded = True, extra_args = extras)
            is True
        )

    @pytest.mark.parametrize("extras", [["--device", "CUDA0"], ["--device", "CUDA0,cpu"]])
    def test_a_pin_naming_a_gpu_still_skips_the_lock(self, monkeypatch, extras):
        assert (
            TestHostMemoryGate._gate(monkeypatch, fully_gpu_offloaded = True, extra_args = extras)
            is False
        )

    def test_an_unreadable_pin_answers_conservatively(self, monkeypatch):
        assert (
            TestHostMemoryGate._gate(
                monkeypatch, fully_gpu_offloaded = True, extra_args = ["--device", ""]
            )
            is True
        )

    @pytest.mark.parametrize(
        ("value", "expected"), [("none", True), ("cpu", True), ("CUDA0", False)]
    )
    def test_the_inherited_env_pin_counts_too(self, monkeypatch, value, expected):
        assert (
            TestHostMemoryGate._gate(
                monkeypatch, fully_gpu_offloaded = True, env = {"LLAMA_ARG_DEVICE": value}
            )
            is expected
        )

    def test_argv_wins_over_the_env_pin(self, monkeypatch):
        """llama.cpp applies the env first and argv after, so a GPU pin in the
        extras overrides an inherited LLAMA_ARG_DEVICE=none."""
        assert (
            TestHostMemoryGate._gate(
                monkeypatch,
                fully_gpu_offloaded = True,
                extra_args = ["--device", "CUDA0"],
                env = {"LLAMA_ARG_DEVICE": "none"},
            )
            is False
        )


class TestAGpuIdsPinOverridesADeviceFlag:
    """gpu_ids owns placement: the launch drops the device flags from argv and
    the env twin, so the child really is on the GPU. Classifying on the raw
    extras would pin a redundant host copy for a --device the child never sees.
    Built with the same helpers the launch uses, so the two cannot drift."""

    @staticmethod
    def _child(extra_args, env, gpu_ids):
        from core.inference.llama_cpp import LlamaCppBackend

        env = dict(env or {})
        if gpu_ids is not None:
            extra_args = LlamaCppBackend._strip_device_extra_args(extra_args)
            LlamaCppBackend._clear_device_placement_env(env)
        return extra_args, env

    @pytest.mark.parametrize(
        "extras", [["--device", "cpu"], ["--device", "none"], ["-dev", "cpu"], ["--device=none"]]
    )
    def test_a_pin_makes_the_gate_ignore_the_device_flag(self, monkeypatch, extras):
        extra_args, env = self._child(extras, None, gpu_ids = [0])
        assert extra_args == [], "the launch drops these, so the gate must not see them"
        assert (
            TestHostMemoryGate._gate(
                monkeypatch, fully_gpu_offloaded = True, extra_args = extra_args, env = env
            )
            is False
        )

    def test_a_pin_drops_the_env_twin_too(self, monkeypatch):
        extra_args, env = self._child(None, {"LLAMA_ARG_DEVICE": "none"}, gpu_ids = [0])
        assert env == {}
        assert (
            TestHostMemoryGate._gate(
                monkeypatch, fully_gpu_offloaded = True, extra_args = extra_args, env = env
            )
            is False
        )

    def test_without_a_pin_the_device_flag_still_counts(self, monkeypatch):
        extra_args, env = self._child(["--device", "cpu"], None, gpu_ids = None)
        assert extra_args == ["--device", "cpu"]
        assert (
            TestHostMemoryGate._gate(
                monkeypatch, fully_gpu_offloaded = True, extra_args = extra_args, env = env
            )
            is True
        )

    def test_a_pin_leaves_the_other_placement_extras_alone(self, monkeypatch):
        """Only the device family is stripped, so -ot still forces host residency."""
        extra_args, _env = self._child(
            ["--device", "cpu", "-ot", r"\.ffn_.*=CPU"], None, gpu_ids = [0]
        )
        assert extra_args == ["-ot", r"\.ffn_.*=CPU"]
        assert (
            TestHostMemoryGate._gate(monkeypatch, fully_gpu_offloaded = True, extra_args = extra_args)
            is True
        )

    def test_the_launch_really_sanitizes_before_classifying(self):
        """Source check: the gate call must receive the stripped extras and env,
        so this cannot regress into reading the raw ones again."""
        import inspect
        import re

        from core.inference.llama_cpp import LlamaCppBackend

        src = inspect.getsource(LlamaCppBackend.load_model)
        strip = src.find("_mem_extra_args = self._strip_device_extra_args(extra_args)")
        assert strip != -1, "the launch no longer strips device flags for the gate"
        assert re.search(r"self\._clear_device_placement_env\(_mem_env\)", src)
        call = src.find("self._weights_in_host_memory(", strip)
        assert call != -1 and "extra_args = _mem_extra_args" in src[call : call + 400]


class TestVulkanCpuFallbackMemoryPolicy:
    def test_cpu_replay_reapplies_the_host_lock(self, monkeypatch):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_model_memory_settings", lambda: (True, False))
        replay = ["--gpu-layers", "0", "--fit", "off", "--device", "none"]
        managed, extras = apply_model_memory_policy(
            replay, supports_load_mode = True, weights_in_host_memory = True
        )
        load_mode, extras = apply_load_mode_policy(
            extras,
            supports_load_mode = True,
            weights_in_host_memory = True,
            requested_load_mode = None,
        )
        argv = ["llama-server", *managed, *load_mode, *extras]
        assert resolve_effective_memory_state(argv, {}) == (True, False)
        assert argv[-6:] == ["--gpu-layers", "0", "--fit", "off", "--device", "none"]

    def test_the_fallback_records_policy_before_spawning(self):
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        src = inspect.getsource(LlamaCppBackend.load_model)
        branch = src.index("fallback_managed, fallback_args")
        spawn = src.index('if not _spawn_and_wait(replay, label = "-cpu")', branch)
        tail = src[branch:spawn]
        assert "resolve_effective_memory_state(replay, env)" in tail
        assert "self._memory_mlock_applicable = True" in tail


class TestFitOffRetryDropsTheLock:
    """The mirror of TestFitOnRetryReArmsResidency. --fit off leaves -ngl at its
    default, which llama.cpp resolves to every layer (llama-model.cpp:
    n_gpu_layers < 0 -> n_layer_all + 1), so a lock taken for the fitted attempt
    would reserve a full host copy of a fully offloaded model."""

    @staticmethod
    def _retry_argv(original, managed, *, drops):
        """What the fallback builds: append --fit off, then drop the managed run."""
        from core.inference.llama_cpp import _without_subsequence

        run = [*original, "--fit", "off"]
        return _without_subsequence(run, managed) if drops else run

    def test_the_retry_is_not_page_locked(self):
        managed = ["--load-mode", "mmap+mlock"]
        original = [*managed, "--fit", "on", "--temp", "0.7"]
        assert resolve_effective_memory_state(original, {}) == (True, False)
        retry = self._retry_argv(original, managed, drops = True)
        assert resolve_effective_memory_state(retry, {}) == (False, False)
        assert retry == ["--fit", "on", "--temp", "0.7", "--fit", "off"]

    def test_the_legacy_flag_path_drops_too(self):
        managed = ["--mlock"]
        original = [*managed, "--fit", "on"]
        retry = self._retry_argv(original, managed, drops = True)
        assert resolve_effective_memory_state(retry, {}) == (False, False)

    def test_a_user_mlock_in_the_extras_survives(self):
        """Managed flags go in before the user's, so only the first run is ours."""
        managed = ["--mlock"]
        original = [*managed, "--fit", "on", "--mlock"]
        retry = self._retry_argv(original, managed, drops = True)
        assert retry == ["--fit", "on", "--mlock", "--fit", "off"]
        assert resolve_effective_memory_state(retry, {}) == (True, False)

    def test_staying_host_resident_keeps_the_lock(self):
        managed = ["--load-mode", "mmap+mlock"]
        original = [*managed, "--fit", "on"]
        retry = self._retry_argv(original, managed, drops = False)
        assert resolve_effective_memory_state(retry, {}) == (True, False)

    def test_the_retry_restores_a_suppressed_per_model_mode(self):
        settings = (True, False)
        managed, extras = apply_model_memory_policy(
            [],
            supports_load_mode = True,
            weights_in_host_memory = True,
            model_memory_settings = settings,
        )
        initial_mode, extras = apply_load_mode_policy(
            extras,
            supports_load_mode = True,
            weights_in_host_memory = True,
            requested_load_mode = "dio",
            model_memory_settings = settings,
        )
        assert managed == ["--load-mode", "mmap+mlock"]
        assert initial_mode == []

        retry_mode, _ = apply_load_mode_policy(
            [],
            supports_load_mode = True,
            weights_in_host_memory = False,
            requested_load_mode = "dio",
            model_memory_settings = settings,
        )
        retry = self._retry_argv([*managed, *extras, "--fit", "on"], managed, drops = True)
        retry = [*retry[:-2], *retry_mode, *retry[-2:]]
        assert retry == ["--fit", "on", "--load-mode", "dio", "--fit", "off"]

    def test_a_restored_reserving_mode_remains_applicable(self):
        settings = (True, False)
        managed, extras = apply_model_memory_policy(
            [],
            supports_load_mode = True,
            weights_in_host_memory = True,
            model_memory_settings = settings,
        )
        retry_mode, retry_extras = apply_load_mode_policy(
            extras,
            supports_load_mode = True,
            weights_in_host_memory = False,
            requested_load_mode = "none",
            model_memory_settings = settings,
        )
        retry = self._retry_argv([*managed, "--fit", "on"], managed, drops = True)
        retry = [*retry[:-2], *retry_mode, *retry_extras, *retry[-2:]]
        state = resolve_effective_memory_state(retry, {})
        assert state == (False, True)
        assert state[1] is True

    def test_the_retry_restores_stripped_hand_typed_extras(self):
        from core.inference.llama_cpp import _replace_subsequence

        settings = (True, False)
        original_extras = ["--load-mode", "dio", "--temp", "0.7"]
        managed, initial_extras = apply_model_memory_policy(
            original_extras,
            supports_load_mode = True,
            weights_in_host_memory = True,
            model_memory_settings = settings,
        )
        initial_mode, initial_extras = apply_load_mode_policy(
            initial_extras,
            supports_load_mode = True,
            weights_in_host_memory = True,
            requested_load_mode = None,
            model_memory_settings = settings,
        )
        initial_policy = [*managed, *initial_mode, *initial_extras]

        retry_managed, retry_extras = apply_model_memory_policy(
            original_extras,
            supports_load_mode = True,
            weights_in_host_memory = False,
            model_memory_settings = settings,
        )
        retry_mode, retry_extras = apply_load_mode_policy(
            retry_extras,
            supports_load_mode = True,
            weights_in_host_memory = False,
            requested_load_mode = None,
            model_memory_settings = settings,
        )
        retry_policy = [*retry_managed, *retry_mode, *retry_extras]
        run = ["llama-server", *initial_policy, "--device", "Vulkan0", "--fit", "off"]
        assert _replace_subsequence(run, initial_policy, retry_policy) == [
            "llama-server",
            "--load-mode",
            "dio",
            "--temp",
            "0.7",
            "--device",
            "Vulkan0",
            "--fit",
            "off",
        ]

    def test_the_dropped_launch_does_not_demand_a_reload(self, monkeypatch):
        """mlock_applicable goes False with the lock, so a later residency save
        is not compared against a lock this launch deliberately dropped."""
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: True)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
        managed = ["--load-mode", "mmap+mlock"]
        state = resolve_effective_memory_state(
            self._retry_argv([*managed, "--fit", "on"], managed, drops = True), {}
        )
        assert memory_state_satisfies_settings(state, True, False) is True

    def test_the_launch_really_reclassifies_the_fit_off_retry(self):
        """Source check: the branch must re-ask the gate and clear the
        bookkeeping, so it cannot drift back to reusing the fitted verdict."""
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        src = inspect.getsource(LlamaCppBackend.load_model)
        branch = src.find('run_cmd = [*run_cmd, "--fit", "off"]')
        assert branch != -1, "the --fit off retry moved"
        # To the end of the retry block, not a fixed width: the branch grows.
        end = src.find("return False", branch)
        assert end != -1
        tail = src[branch:end]
        for needle in (
            "self._weights_in_host_memory(",
            "fully_gpu_offloaded = True",
            "_replace_subsequence(",
            "_mem_policy_argv",
            "_retry_policy_argv",
            "requested_load_mode = load_mode",
            "_mem_host_resident = False",
            "self._memory_mlock_applicable = (",
            "_mem_host_resident or self._memory_state[1]",
            "resolve_effective_memory_state(run_cmd, env)",
        ):
            assert needle in tail, needle


class TestFitOffRetryClearsPolicyActivity:
    """Dropping the lock can leave the child identical to an unmanaged launch.
    Keeping policy_active set from the first attempt then makes turning the
    toggles off demand a reload that would relaunch the very same argv."""

    @staticmethod
    def _satisfied(monkeypatch, *, policy_active):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: False)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
        # The retry child: lock dropped, so no lock and no reservation.
        return memory_state_satisfies_settings((False, False), policy_active, False)

    def test_an_untouched_child_is_left_alone(self, monkeypatch):
        assert self._satisfied(monkeypatch, policy_active = False) is True

    def test_a_still_touched_child_is_relaunched(self, monkeypatch):
        """A scrub or a strip survives the drop, so that one must still reload."""
        assert self._satisfied(monkeypatch, policy_active = True) is False

    def test_the_launch_recomputes_activity_without_the_managed_flag(self):
        """Source check: the retry must reuse the non-managed half of the launch
        expression, not leave the first attempt's verdict standing."""
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        src = inspect.getsource(LlamaCppBackend.load_model)
        assert "_mem_managed or _load_mode_policy_suppressed" in src
        assert ") or _mem_policy_touched_extras" in src
        branch = src.find('run_cmd = [*run_cmd, "--fit", "off"]')
        assert branch != -1
        end = src.find("return False", branch)
        assert end != -1
        assert "self._memory_policy_active = _mem_policy_touched_extras" in src[branch:end]


class TestNoDeadMemoryBookkeeping:
    """Every _memory_* marker the launch records is read by something. A
    write-only one silently goes stale on the retry paths, which is how the
    fit-off retry's activity marker was missed."""

    def test_the_launch_records_nothing_unread(self):
        import ast
        from pathlib import Path

        backend = Path(__file__).resolve().parent.parent
        target = backend / "core" / "inference" / "llama_cpp.py"

        def attrs(tree, ctx):
            return {
                n.attr
                for n in ast.walk(tree)
                if isinstance(n, ast.Attribute) and isinstance(n.ctx, ctx)
            }

        written = {
            a
            for a in attrs(ast.parse(target.read_text(encoding = "utf-8")), ast.Store)
            if a.startswith("_memory_") or a.endswith("_mlock_enabled")
        }
        assert written, "no markers found; the scan is looking in the wrong place"

        read: set[str] = set()
        for path in backend.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding = "utf-8"))
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue
            read |= attrs(tree, ast.Load)
            # routes read some of these dynamically:
            # getattr(backend, "_memory_policy_active", False).
            read |= {
                n.args[1].value
                for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "getattr"
                and len(n.args) >= 2
                and isinstance(n.args[1], ast.Constant)
                and isinstance(n.args[1].value, str)
            }
        unread = sorted(written - read)
        assert not unread, f"written but never read, so they go stale on retries: {unread}"


class TestAnActiveFitterVoidsTheAllLayersVerdict:
    """-ngl -1 IS llama.cpp's default, so common/fit.cpp does not abort on it
    (it aborts only on a count the user really set) and the fitter is free to
    move layers back to the CPU. A concrete count stands, so only -1 is gated."""

    @staticmethod
    def _all_on_gpu(monkeypatch, extras, *, fit_active):
        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        monkeypatch.setattr(type(backend), "n_layers", property(lambda self: 32), raising = False)
        backend._n_cpu_moe = 0
        return backend._offloads_every_layer(
            gpu_memory_mode = "auto",
            gpu_layers = None,
            extra_args = extras,
            fit_active = fit_active,
        )

    def test_minus_one_under_an_active_fitter_is_not_full_offload(self, monkeypatch):
        assert self._all_on_gpu(monkeypatch, ["-ngl", "-1"], fit_active = True) is False

    def test_minus_one_with_the_fitter_off_still_is(self, monkeypatch):
        assert self._all_on_gpu(monkeypatch, ["-ngl", "-1"], fit_active = False) is True

    @pytest.mark.parametrize("count", ["33", "99"])
    def test_a_concrete_count_stands_under_the_fitter(self, monkeypatch, count):
        """fit.cpp aborts on a user-set n_gpu_layers, so it cannot be lowered."""
        assert self._all_on_gpu(monkeypatch, ["-ngl", count], fit_active = True) is True

    def test_a_count_at_or_below_the_blocks_is_never_full(self, monkeypatch):
        assert self._all_on_gpu(monkeypatch, ["-ngl", "32"], fit_active = True) is False


class TestTheEffectiveFitterState:
    """fit_is_enabled_in answers for the extras; this answers for the child, so
    Unsloth's own --fit counts and llama.cpp's ON default is respected."""

    @pytest.mark.parametrize(
        ("args", "env", "expected"),
        [
            ([], None, True),
            (["--fit", "on"], None, True),
            (["--fit", "off"], None, False),
            (["--fit", "on", "--fit", "off"], None, False),
            (["--fit", "off", "--fit", "on"], None, True),
            (["--fit=off"], None, False),
            (["-fit", "off"], None, False),
            ([], {"LLAMA_ARG_FIT": "off"}, False),
            ([], {"LLAMA_ARG_FIT": "on"}, True),
            (["--fit", "off"], {"LLAMA_ARG_FIT": "on"}, False),
            (["--fit", "on"], {"LLAMA_ARG_FIT": "off"}, True),
            (["--fit", "banana"], None, True),
        ],
    )
    def test_only_an_explicit_off_disables_it(self, args, env, expected):
        assert _lsa.fit_is_effectively_on(args, env) is expected

    def test_the_launch_asks_over_the_whole_command(self):
        """Source check: Unsloth emits its own --fit into cmd, so reading the
        extras alone would miss it."""
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        src = inspect.getsource(LlamaCppBackend.load_model)
        assert "fit_active = fit_is_effectively_on(" in src
        assert "[*cmd, *(_mem_extra_args or [])], _mem_env" in src
