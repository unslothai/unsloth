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
strip_shadowing_flags = _lsa.strip_shadowing_flags


@pytest.fixture
def policy(monkeypatch):
    """Run the policy under a given toggle pair.

    The policy imports the settings module lazily, so patch that module.
    """
    import utils.model_memory_settings as mm

    def run(keep_resident: bool, no_ram_reserve: bool, extras, supports_load_mode = False):
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
        _, out = policy(True, False, ["--load-mode", "none", "--temp", "0.7"],
                        supports_load_mode = True)
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
