# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio asks llama-server for byte-identical output, instead of the user asking
llama.cpp: the setting, the child environment, the launch flags, the fallback and the state."""

from __future__ import annotations

import pytest

from core.inference import llama_exact as exact
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_preemption import PreemptionController, reset_preemption_controllers


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(exact.EXACT_ENV, raising = False)
    monkeypatch.delenv(exact.CHILD_ENV, raising = False)
    reset_preemption_controllers()
    yield
    reset_preemption_controllers()


# A launch line of the shape Studio actually emits: four slots, a unified cache, flash
# attention on, no context shift.
_STUDIO_ARGV = [
    "llama-server",
    "-m",
    "/models/Qwen3.5-4B-UD-Q4_K_XL.gguf",
    "--port",
    "9705",
    "--parallel",
    "4",
    "--flash-attn",
    "on",
    "--no-context-shift",
    "-c",
    "8192",
    "--metrics",
    "-ngl",
    "-1",
    "--fit",
    "off",
    "--kv-unified",
    "--jinja",
]

# What the server prints on its way out when it will not run the mode: all three
# spellings from unslothai/llama.cpp#194.
_REFUSAL = (
    "llama_kv_cache: LLAMA_EXACT_CONCURRENCY is set but it needs a unified KV cache "
    "(pass --kv-unified)\n"
    "terminate called after throwing an instance of 'std::runtime_error'\n"
    "  what():  exact concurrency: unsupported KV cache configuration\n"
)
_REFUSAL_LAYER = (
    "llama_kv_cache: LLAMA_EXACT_CONCURRENCY is set but layer 3 keeps its KV cache on CPU, "
    "which has no paged attention\n"
    "what():  exact concurrency: KV cache layer is not on the CUDA backend\n"
)
_REFUSAL_BOUND = (
    "GGML_CUDA_BATCH_INVARIANT_MAX_COLS is 4 but LLAMA_EXACT_CONCURRENCY needs at least 8 "
    "to cover a decode step of 4 slots\n"
)
_UNRELATED_CRASH = (
    "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 12000.00 MiB on device 0: "
    "cudaMalloc failed: out of memory"
)


class TestTheSetting:
    def test_the_default_is_off_because_the_mode_costs_throughput(self):
        assert exact.resolve_exact_setting(None, stored = None, environ = {}) == exact.EXACT_OFF

    @pytest.mark.parametrize("value", ["auto", "off", "on"])
    def test_all_three_come_back_from_the_request_the_store_and_the_environment(self, value):
        assert exact.resolve_exact_setting(value, stored = None, environ = {}) == value
        assert exact.resolve_exact_setting(None, stored = value, environ = {}) == value
        assert (
            exact.resolve_exact_setting("on", stored = "off", environ = {exact.EXACT_ENV: value})
            == value
        ), "the environment overrides both"

    def test_the_request_overrides_the_store(self):
        assert exact.resolve_exact_setting("on", stored = "off", environ = {}) == exact.EXACT_ON
        assert exact.resolve_exact_setting("off", stored = "on", environ = {}) == exact.EXACT_OFF

    @pytest.mark.parametrize("spelling", ["yes", "true", "1", "", "exact", None])
    def test_an_unknown_spelling_is_not_a_setting(self, spelling):
        assert exact.normalize_setting(spelling) is None
        assert exact.resolve_exact_setting(spelling, stored = "auto", environ = {}) == exact.EXACT_AUTO

    def test_an_inherited_llama_variable_is_the_default_rather_than_ignored(self):
        environ = {exact.CHILD_ENV: "1"}
        assert exact.resolve_exact_setting(None, stored = None, environ = environ) == exact.EXACT_ON
        # And an explicit off, from anywhere, still beats it.
        assert exact.resolve_exact_setting(None, stored = "off", environ = environ) == exact.EXACT_OFF
        assert exact.resolve_exact_setting("off", stored = None, environ = environ) == exact.EXACT_OFF
        assert (
            exact.resolve_exact_setting(
                None, stored = None, environ = {**environ, exact.EXACT_ENV: "off"}
            )
            == exact.EXACT_OFF
        )

    @pytest.mark.parametrize(
        ("raw", "set_"),
        [
            ("0", False),
            ("", False),
            ("no", False),
            ("off", False),
            ("false", False),
            ("1", True),
            ("2", True),
            ("true", True),
            ("on", True),
            ("yes", True),
        ],
    )
    def test_what_counts_as_an_inherited_variable_being_set(self, raw, set_):
        assert exact.child_flag_set({exact.CHILD_ENV: raw}) is set_

    def test_wants_exact_is_auto_and_on(self):
        assert [exact.wants_exact(v) for v in ("auto", "on", "off", None)] == [
            True,
            True,
            False,
            False,
        ]


class TestTheChildEnvironment:
    def test_the_variable_is_set_exactly_when_the_answer_is_yes(self):
        for setting in ("auto", "on"):
            env = {"PATH": "/usr/bin"}
            assert exact.apply_child_env(env, on = exact.wants_exact(setting)) is True
            assert env[exact.CHILD_ENV] == "1"
            assert env["PATH"] == "/usr/bin", "nothing else in the environment is touched"
        env = {"PATH": "/usr/bin"}
        assert exact.apply_child_env(env, on = exact.wants_exact("off")) is False
        assert exact.CHILD_ENV not in env

    def test_an_explicit_off_takes_an_inherited_variable_back_out(self):
        env = {exact.CHILD_ENV: "1"}
        assert exact.apply_child_env(env, on = False) is True
        assert exact.CHILD_ENV not in env

    def test_applying_the_same_answer_twice_changes_nothing(self):
        env: dict = {}
        assert exact.apply_child_env(env, on = True) is True
        assert exact.apply_child_env(env, on = True) is False
        assert exact.apply_child_env(env, on = False) is True
        assert exact.apply_child_env(env, on = False) is False


class TestTheLaunchArgs:
    @pytest.mark.parametrize(
        "args, expected",
        [
            (_STUDIO_ARGV, []),
            (["--cache-reuse", "256"], ["--cache-reuse"]),
            (["--cache-reuse=256"], ["--cache-reuse"]),
            # The flag spelled as its default is the default, not a contradiction.
            (["--cache-reuse", "0"], []),
            (["--context-shift"], ["--context-shift"]),
            (["--no-kv-offload"], ["--no-kv-offload"]),
            (["-nkvo"], ["-nkvo"]),
            (["--no-flash-attn"], ["--no-flash-attn"]),
            (["--flash-attn", "off"], ["--flash-attn"]),
            (["-fa", "0"], ["-fa"]),
            (["--flash-attn", "on"], []),
            (["--cache-type-k", "q8_0"], ["--cache-type-k"]),
            (["-ctv", "q4_0"], ["-ctv"]),
            (["--cache-type-k", "f16", "--cache-type-v", "f16"], []),
            (["--no-context-shift"], []),
            (["--cache-reuse", "512", "--ctk", "q8_0", "-nkvo"], ["--cache-reuse", "-nkvo"]),
            ([], []),
            (None, []),
            # An option's last occurrence decides, as llama-server applies argv.
            (["--flash-attn", "off", "--flash-attn", "on"], []),
            (["-fa", "off", "--flash-attn=on"], []),
            (["--no-flash-attn", "-fa", "on"], []),
            (["--flash-attn", "on", "-fa", "off"], ["-fa"]),
            (["-ctk", "q8_0", "--cache-type-k", "f16"], []),
            (["--cache-type-v", "f16", "-ctv", "q4_0"], ["-ctv"]),
            (["--cache-reuse", "512", "--cache-reuse", "0"], []),
            (["--context-shift", "--no-context-shift"], []),
            (["-nkvo", "--kv-offload"], []),
            (["--kv-offload", "-nkvo"], ["-nkvo"]),
        ],
    )
    def test_what_the_mode_cannot_run_beside(self, args, expected):
        assert exact.contradicting_args(args) == expected

    @pytest.mark.parametrize(
        ("args", "caps", "env", "missing"),
        [
            (["llama-server", "-m", "x.gguf", "--parallel", "1"], True, {}, ["--kv-unified"]),
            (_STUDIO_ARGV, True, {}, []),
            (["llama-server"], False, {}, []),
            (["llama-server"], True, {"LLAMA_ARG_KV_UNIFIED": "1"}, []),
        ],
    )
    def test_a_load_is_handed_the_unified_cache_the_mode_needs_and_nothing_else(
        self, args, caps, env, missing
    ):
        assert (
            LlamaCppBackend._exact_missing_launch_flags(
                args, {"supports_kv_unified": caps}, env = env
            )
            == missing
        )


class TestRefusalDetection:
    @pytest.mark.parametrize("output", [_REFUSAL, _REFUSAL_LAYER, _REFUSAL_BOUND])
    def test_every_way_the_server_names_the_mode_is_recognised(self, output):
        assert exact.is_exact_refusal(output) is True

    @pytest.mark.parametrize("output", [_UNRELATED_CRASH, "", None, "exact match not found"])
    def test_nothing_else_is(self, output):
        assert exact.is_exact_refusal(output) is False


class TestTheAutoFallback:
    def test_a_named_refusal_under_auto_drops_the_variable_and_asks_for_a_relaunch(self):
        env = {exact.CHILD_ENV: "1", "PATH": "/usr/bin"}
        assert (
            LlamaCppBackend._drop_exact_after_refusal(
                env, setting = "auto", crashed = True, output = _REFUSAL
            )
            is True
        )
        assert exact.CHILD_ENV not in env and env["PATH"] == "/usr/bin"
        assert (
            LlamaCppBackend._drop_exact_after_refusal(
                env, setting = "auto", crashed = True, output = _REFUSAL
            )
            is False
        ), "the fallback fires once"

    @pytest.mark.parametrize(
        ("setting", "crashed", "output"),
        [
            # Under `on` a refusal is the answer rather than a prompt to retry.
            ("on", True, _REFUSAL),
            ("auto", True, _UNRELATED_CRASH),
            ("auto", False, _REFUSAL),
        ],
    )
    def test_nothing_else_spends_the_fallback(self, setting, crashed, output):
        env = {exact.CHILD_ENV: "1"}
        assert (
            LlamaCppBackend._drop_exact_after_refusal(
                env, setting = setting, crashed = crashed, output = output
            )
            is False
        )
        assert env[exact.CHILD_ENV] == "1", "the load fails carrying the mode it asked for"

    def test_the_refusal_message_names_the_setting_and_what_the_mode_needs(self):
        message = LlamaCppBackend._classify_start_failure_text(
            _REFUSAL, "/models/x.gguf", "unsloth/x", returncode = 1
        )
        assert "exact concurrency" in message.lower()
        assert "'on'" in message and "'auto'" in message
        assert "unified KV cache" in message and "pass --kv-unified" in message
        ordinary = LlamaCppBackend._classify_start_failure_text(
            _UNRELATED_CRASH, "/models/x.gguf", "unsloth/x", returncode = 1
        )
        assert "exact concurrency is set to" not in ordinary.lower()


class TestTheReportedState:
    def _state(self, **kw):
        # The build's own answer, read off the `--preempt-ram` capability: the same fork
        # carries both, and a build without the mode starts fine while ignoring the variable.
        kw.setdefault("supports_exact", True)
        return LlamaCppBackend._exact_state_after_launch(**kw)

    @pytest.mark.parametrize("setting", ["auto", "on"])
    def test_a_build_that_ignores_the_variable_is_not_reported_as_on(self, setting):
        assert (
            self._state(
                setting = setting,
                env = {exact.CHILD_ENV: "1"},
                args = _STUDIO_ARGV,
                supports_exact = False,
            )
            == exact.EXACT_STATE_UNAVAILABLE
        )

    def test_off_when_the_load_never_asked(self):
        assert self._state(setting = "off", env = {}, args = _STUDIO_ARGV) == exact.EXACT_STATE_OFF
        # Even with the variable still on the environment: the load resolved to off, so
        # apply_child_env removed it.
        assert (
            self._state(setting = "off", env = {exact.CHILD_ENV: "1"}, args = _STUDIO_ARGV)
            == exact.EXACT_STATE_OFF
        )

    @pytest.mark.parametrize("setting", ["auto", "on"])
    def test_on_when_the_variable_and_the_launch_both_hold(self, setting):
        assert (
            self._state(setting = setting, env = {exact.CHILD_ENV: "1"}, args = _STUDIO_ARGV)
            == exact.EXACT_STATE_ON
        )

    def test_unavailable_once_the_fallback_or_a_respawn_took_away_what_it_needs(self):
        assert (
            self._state(setting = "auto", env = {}, args = _STUDIO_ARGV) == exact.EXACT_STATE_UNAVAILABLE
        )
        no_flash = [a for a in _STUDIO_ARGV if a not in ("--flash-attn", "on")]
        no_unified = [a for a in _STUDIO_ARGV if a != "--kv-unified"]
        for setting, args in (
            ("auto", no_flash + ["--flash-attn", "off"]),
            ("on", no_unified),
            ("auto", _STUDIO_ARGV + ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]),
        ):
            assert (
                self._state(setting = setting, env = {exact.CHILD_ENV: "1"}, args = args)
                == exact.EXACT_STATE_UNAVAILABLE
            )


class TestThePreemptionSnapshotReportsItAndNeverActsOnIt:
    def test_the_snapshot_carries_the_mode_and_defaults_to_off(self):
        controller = PreemptionController("exact")
        assert controller.snapshot().exact == exact.EXACT_STATE_OFF
        controller.configure(budget = 8192, kv_unified = True, slots = 4, exact = "on")
        assert controller.snapshot().exact == "on"
        controller.configure(budget = 4096)
        assert controller.snapshot().exact == "on", "configure without the argument keeps it"


class TestTheParkingBudgetHoldsTheWholePool:
    """A park that outgrows --preempt-ram is re-prefilled, and a re-prefill is not byte-identical
    on CUDA, so an exact launch sizes the budget to the pool when the default would not hold it."""

    def test_a_pool_past_the_default_gets_a_budget_that_holds_it(self):
        from core.inference.llama_cpp import _PREEMPT_RAM_DEFAULT_MIB, _exact_parking_budget_mib

        pool = 12 * 1024 * 1024 * 1024
        budget = _exact_parking_budget_mib(pool, args = ["llama-server"], env = {})
        assert budget is not None
        assert budget * 1024 * 1024 >= pool
        assert budget > _PREEMPT_RAM_DEFAULT_MIB

    def test_a_pool_the_default_holds_needs_no_flag(self):
        from core.inference.llama_cpp import _exact_parking_budget_mib
        assert _exact_parking_budget_mib(2 * 1024 * 1024 * 1024, args = [], env = {}) is None

    def test_an_unknown_pool_needs_no_flag(self):
        from core.inference.llama_cpp import _exact_parking_budget_mib
        assert _exact_parking_budget_mib(0, args = [], env = {}) is None

    @pytest.mark.parametrize(
        ("args", "env"),
        [
            (["llama-server", "--preempt-ram", "1024"], {}),
            (["llama-server", "--preempt-ram=1024"], {}),
            (["llama-server"], {"LLAMA_ARG_PREEMPT_RAM": "0"}),
        ],
    )
    def test_a_budget_someone_named_keeps_its_say(self, args, env):
        from core.inference.llama_cpp import _exact_parking_budget_mib
        assert _exact_parking_budget_mib(64 * 1024 * 1024 * 1024, args = args, env = env) is None
