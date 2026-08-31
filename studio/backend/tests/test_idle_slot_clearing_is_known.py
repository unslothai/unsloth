# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether a released slot's KV cells actually come back.

Re-costing a growing tool loop yields its old commitment before asking for a bigger one,
which is only capacity if llama-server clears the idle slot. Under ``--kv-unified`` it
clears only with ``--cache-idle-slots``, which requires cache-ram and which
``--cache-ram 0`` force-disables. Studio emits ``--cache-ram 0`` on Windows under full
GPU offload (#5692) alongside ``--kv-unified``, so this is a live configuration.

Read off the argv actually spawned, so every emitter is covered by construction.
"""

from __future__ import annotations

import pytest

from core.inference.llama_cpp import _idle_slot_clearing_active as active
from routes.inference import _openai_llama_admission_can_yield as can_yield


BASE = ["llama-server", "-m", "model.gguf", "--parallel", "4", "--kv-unified"]


class TestTheArgvIsWhatDecides:
    def test_a_modern_binary_clears_by_default(self):
        """--cache-ram defaults to 8192 MiB and Studio rarely emits the flag, so an
        absent flag is the common case and means ON."""
        assert active(BASE, supports_cache_ram = True) is True

    def test_a_binary_without_cache_ram_never_clears(self):
        """It predates the feature, and Studio skips the flag rather than failing the
        launch, so the argv looks identical to the default case."""
        assert active(BASE, supports_cache_ram = False) is False

    def test_windows_full_offload_does_not_clear(self):
        """The #5692 path: --cache-ram 0 force-disables --cache-idle-slots."""
        assert active(BASE + ["--cache-ram", "0"], supports_cache_ram = True) is False

    def test_an_explicit_cache_ram_clears(self):
        assert active(BASE + ["--cache-ram", "4096"], supports_cache_ram = True) is True

    def test_the_flag_can_be_turned_off_by_name(self):
        assert (
            active(
                BASE + ["--cache-ram", "4096", "--no-cache-idle-slots"],
                supports_cache_ram = True,
            )
            is False
        )

    def test_it_can_be_turned_back_on_by_name(self):
        assert (
            active(BASE + ["--cache-ram", "0", "--cache-idle-slots"], supports_cache_ram = True)
            is True
        )

    def test_last_wins_like_llama_cpp(self):
        assert (
            active(BASE + ["--cache-ram", "4096", "--cache-ram", "0"], supports_cache_ram = True)
            is False
        )
        assert (
            active(BASE + ["--cache-ram", "0", "--cache-ram", "4096"], supports_cache_ram = True)
            is True
        )

    @pytest.mark.parametrize("bad", ["", "abc", "-1", None])
    def test_an_unreadable_value_is_treated_as_no_clearing(self, bad):
        cmd = BASE + ["--cache-ram"] + ([] if bad is None else [bad])
        assert active(cmd, supports_cache_ram = True) is False

    def test_a_trailing_flag_does_not_raise(self):
        assert active(BASE + ["--cache-ram"], supports_cache_ram = True) is False


class _Backend:
    def __init__(self, value):
        if value is not None:
            self.idle_slot_clearing_active = value


class TestTheRouteAsksTheBackend:
    def test_it_yields_when_the_server_clears(self):
        assert can_yield(_Backend(True)) is True

    def test_it_does_not_yield_when_the_server_does_not(self):
        assert can_yield(_Backend(False)) is False

    def test_a_backend_that_cannot_say_does_not_yield(self):
        """No load yet, a stub, or an older backend object. Declining is safe;
        promising the same cells twice is not."""
        assert can_yield(_Backend(None)) is False
        assert can_yield(None) is False
