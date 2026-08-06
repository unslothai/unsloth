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
memory_state_satisfies_settings = _lsa.memory_state_satisfies_settings
resolve_effective_memory_state = _lsa.resolve_effective_memory_state
scrub_memory_env = _lsa.scrub_memory_env
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
    ):
        monkeypatch.setattr(mm, "get_keep_resident", lambda: keep_resident)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: no_ram_reserve)
        monkeypatch.setattr(mm, "should_mlock", lambda: keep_resident and not no_ram_reserve)
        return apply_model_memory_policy(extras, supports_load_mode = supports_load_mode)

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
        "var", ["LLAMA_ARG_MLOCK", "LLAMA_ARG_MMAP", "LLAMA_ARG_LOAD_MODE", "LLAMA_ARG_DIO"]
    )
    def test_a_toggle_scrubs_inherited_memory_env(self, toggles, var):
        toggles(False, True)
        env = {var: "1", "PATH": "/usr/bin"}
        assert var in scrub_memory_env(env)
        assert var not in env
        assert env["PATH"] == "/usr/bin"

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
            (["-lm", "mlock"], {}, (True, True)),
            (["--load-mode", "none"], {}, (False, True)),
            ([], {"LLAMA_ARG_MLOCK": "1"}, (True, False)),
            ([], {"LLAMA_ARG_MMAP": "off"}, (False, True)),
            ([], {"LLAMA_ARG_LOAD_MODE": "mmap+mlock"}, (True, False)),
            # argv beats env, matching llama.cpp.
            (["--load-mode", "mmap"], {"LLAMA_ARG_MLOCK": "1"}, (False, False)),
            (["--mlock", "--load-mode", "mmap"], {}, (False, False)),
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
    def _required(keep, no_res, state, monkeypatch):
        import routes.inference
        import routes.settings as rs
        import utils.model_memory_settings as mm

        backend = type("_B", (), {"is_loaded": True, "_memory_state": state})()
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
            # Both off: Unsloth does not manage placement.
            (False, False, (True, True), False),
            (False, False, (False, False), False),
        ],
    )
    def test_matrix(self, monkeypatch, keep, no_res, state, expected):
        assert self._required(keep, no_res, state, monkeypatch) is expected

    def test_no_model_loaded_never_asks_for_a_reload(self, monkeypatch):
        import routes.inference
        import routes.settings as rs

        backend = type("_B", (), {"is_loaded": False, "_memory_state": (False, True)})()
        monkeypatch.setattr(routes.inference, "get_llama_cpp_backend", lambda: backend)
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
        ("launched", "keep", "no_res", "satisfied"),
        [
            ((False, False), True, False, False),  # turn residency on
            ((True, False), False, True, False),  # turn no-reserve on, mlocked
            ((False, True), False, True, False),  # turn no-reserve on, no-mmap
            ((True, False), True, False, True),  # already pinned
            ((False, False), False, True, True),  # already clean
            ((True, True), False, False, True),  # unmanaged
        ],
    )
    def test_forces_a_reload_only_when_the_policy_changed(
        self, monkeypatch, launched, keep, no_res, satisfied
    ):
        import utils.model_memory_settings as mm

        monkeypatch.setattr(mm, "get_keep_resident", lambda: keep)
        monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: no_res)
        assert memory_state_satisfies_settings(launched) is satisfied


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
