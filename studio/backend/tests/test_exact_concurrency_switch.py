# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio asks llama-server for byte-identical output, instead of the user asking llama.cpp.

Exact concurrency (unslothai/llama.cpp#194) makes a chat's generated tokens the same whether
it decodes alone or beside three others in one KV cache. It has no command-line flag, no
``--help`` entry and nothing in ``/props``: the whole interface is ``LLAMA_EXACT_CONCURRENCY``
on the server's environment. Today that means setting a llama.cpp variable on the STUDIO
process, which is neither per-load nor discoverable, and which every child then inherits
whether or not this load wanted it.

These tests pin the switch that replaces it: three values, resolved from the environment, the
load request and a stored setting in that order; a child environment that carries the
variable exactly when the answer is yes; a launch line that never contradicts the mode; one
relaunch without it when the server names it as the reason it would not start; and a load
that says which of the three answers it ended up with.

The one thing none of this can do is ask a server whether it HAS the mode. A build that
predates #194 ignores the variable and starts perfectly, so it reports ``on``. That is
recorded here as the behaviour it is, not fixed, because there is nothing to read it from.
"""

from __future__ import annotations

import pytest

from core.inference import llama_exact as exact
from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend
from core.inference.llama_preemption import (
    PreemptionController,
    reset_preemption_controllers,
)
from models.inference import InferenceStatusResponse, LoadRequest


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    """No inherited answer from the machine the tests run on: both variables are read at
    load time, and a developer with either one set would otherwise see a different suite."""
    monkeypatch.delenv(exact.EXACT_ENV, raising = False)
    monkeypatch.delenv(exact.CHILD_ENV, raising = False)
    reset_preemption_controllers()
    yield
    reset_preemption_controllers()


# A launch line of the shape Studio actually emits, taken from the swap-C runs: four slots,
# a unified cache, flash attention on, no context shift.
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

# What the server prints on its way out when it will not run the mode. Both spellings from
# unslothai/llama.cpp#194: the thrown message and the log line that precedes it.
_REFUSAL_THROWN = (
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


# ------------------------------------------------------------------ resolving the setting


class TestSetting:
    def test_the_default_is_off_because_the_mode_costs_throughput(self):
        assert exact.resolve_exact_setting(None, stored = None, environ = {}) == exact.EXACT_OFF

    @pytest.mark.parametrize("value", ["auto", "off", "on"])
    def test_all_three_come_back_from_the_request(self, value):
        assert exact.resolve_exact_setting(value, stored = None, environ = {}) == value

    @pytest.mark.parametrize("value", ["auto", "off", "on"])
    def test_all_three_come_back_from_the_store(self, value):
        assert exact.resolve_exact_setting(None, stored = value, environ = {}) == value

    @pytest.mark.parametrize("value", ["auto", "off", "on"])
    def test_the_environment_overrides_both(self, value):
        environ = {exact.EXACT_ENV: value}
        assert exact.resolve_exact_setting("on", stored = "off", environ = environ) == value

    def test_the_request_overrides_the_store(self):
        assert exact.resolve_exact_setting("on", stored = "off", environ = {}) == exact.EXACT_ON
        assert exact.resolve_exact_setting("off", stored = "on", environ = {}) == exact.EXACT_OFF

    @pytest.mark.parametrize("spelling", ["ON", " Auto ", "off"])
    def test_case_and_padding_are_accepted(self, spelling):
        assert (
            exact.resolve_exact_setting(spelling, stored = None, environ = {})
            == spelling.strip().lower()
        )

    @pytest.mark.parametrize("spelling", ["yes", "true", "1", "", "exact", None])
    def test_an_unknown_spelling_is_not_a_setting(self, spelling):
        """It falls through rather than becoming a fourth value: a typo in an environment
        variable must not be the difference between a load that is byte-identical and one
        that quietly is not."""
        assert exact.normalize_setting(spelling) is None
        assert exact.resolve_exact_setting(spelling, stored = "auto", environ = {}) == exact.EXACT_AUTO

    def test_an_inherited_llama_variable_is_the_default_rather_than_ignored(self):
        """The workaround this switch replaces. Somebody running today with
        LLAMA_EXACT_CONCURRENCY=1 on Studio is in exact mode; shipping a switch that
        defaults to off would silently take it away from exactly the people who wanted it."""
        environ = {exact.CHILD_ENV: "1"}
        assert exact.resolve_exact_setting(None, stored = None, environ = environ) == exact.EXACT_ON

    def test_an_explicit_off_still_beats_the_inherited_variable(self):
        environ = {exact.CHILD_ENV: "1"}
        assert exact.resolve_exact_setting(None, stored = "off", environ = environ) == exact.EXACT_OFF
        assert exact.resolve_exact_setting("off", stored = None, environ = environ) == exact.EXACT_OFF
        assert (
            exact.resolve_exact_setting(
                None, stored = None, environ = {**environ, exact.EXACT_ENV: "off"}
            )
            == exact.EXACT_OFF
        )

    @pytest.mark.parametrize("raw", ["0", "", "no", "off", "false"])
    def test_a_zero_inherited_variable_is_not_exact_mode(self, raw):
        assert exact.child_flag_set({exact.CHILD_ENV: raw}) is False

    @pytest.mark.parametrize("raw", ["1", "2", "true", "on", "yes"])
    def test_a_set_inherited_variable_is(self, raw):
        assert exact.child_flag_set({exact.CHILD_ENV: raw}) is True

    def test_wants_exact_is_auto_and_on(self):
        assert exact.wants_exact("auto") is True
        assert exact.wants_exact("on") is True
        assert exact.wants_exact("off") is False
        assert exact.wants_exact(None) is False

    def test_the_request_field_carries_the_three_values(self):
        assert LoadRequest(model_path = "m.gguf").exact_concurrency is None
        for value in ("auto", "off", "on"):
            assert (
                LoadRequest(model_path = "m.gguf", exact_concurrency = value).exact_concurrency == value
            )
        with pytest.raises(Exception):
            LoadRequest(model_path = "m.gguf", exact_concurrency = "sometimes")

    def test_the_intent_carries_it_to_the_backend(self):
        assert GgufLoadIntent(model_identifier = "m").exact_concurrency is None
        assert (
            GgufLoadIntent(model_identifier = "m", exact_concurrency = "on").exact_concurrency == "on"
        )


# ------------------------------------------------------------------- the child environment


class TestChildEnvironment:
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
        """The child environment starts as a copy of Studio's, so leaving it alone would
        let the inherited variable outvote a load that explicitly resolved to off, which is
        the one answer that has to be obeyed exactly: it is what a user picks when the 9 per
        cent is the thing they are trying to get rid of."""
        env = {exact.CHILD_ENV: "1"}
        assert exact.apply_child_env(env, on = False) is True
        assert exact.CHILD_ENV not in env

    def test_applying_the_same_answer_twice_changes_nothing(self):
        env = {}
        assert exact.apply_child_env(env, on = True) is True
        assert exact.apply_child_env(env, on = True) is False
        assert exact.apply_child_env(env, on = False) is True
        assert exact.apply_child_env(env, on = False) is False


# ------------------------------------------------------------------------- the launch line


class TestLaunchArgs:
    def test_studios_own_launch_line_contradicts_nothing(self):
        assert exact.contradicting_args(_STUDIO_ARGV) == []

    @pytest.mark.parametrize(
        "args, expected",
        [
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
            ([], []),
            (None, []),
        ],
    )
    def test_what_the_mode_cannot_run_beside(self, args, expected):
        assert exact.contradicting_args(args) == expected

    def test_every_contradiction_in_the_users_extras_is_named(self):
        extras = ["--cache-reuse", "512", "--ctk", "q8_0", "-nkvo"]
        assert exact.contradicting_args(extras) == ["--cache-reuse", "-nkvo"]

    def test_studio_never_emits_cache_reuse_itself(self):
        """The mode cannot live with it and Studio has no reason to pass it, so the
        guarantee is that no emitter can produce the token, rather than that something
        strips it later. Over string literals, not the file text: a comment saying Studio
        does not pass it is the opposite of a violation. A future launch flag that adds it
        should fail here and be thought about."""
        import ast
        from pathlib import Path

        backend_dir = Path(__file__).resolve().parent.parent
        for name in ("core/inference/llama_cpp.py", "core/inference/llama_server_args.py"):
            tree = ast.parse((backend_dir / name).read_text(encoding = "utf-8"))
            literals = [
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            ]
            assert not [text for text in literals if text.strip() == "--cache-reuse"], name

    def test_a_single_slot_load_gets_the_unified_cache_the_mode_needs(self):
        """--parallel 1 skips --kv-unified, and the paged pool needs it whether or not
        anything else is decoding."""
        assert LlamaCppBackend._exact_missing_launch_flags(
            ["llama-server", "-m", "x.gguf", "--parallel", "1"],
            {"supports_kv_unified": True},
            env = {},
        ) == ["--kv-unified"]

    def test_a_launch_that_already_has_it_is_left_alone(self):
        assert (
            LlamaCppBackend._exact_missing_launch_flags(
                _STUDIO_ARGV, {"supports_kv_unified": True}, env = {}
            )
            == []
        )

    def test_a_build_without_the_flag_is_not_handed_one(self):
        assert (
            LlamaCppBackend._exact_missing_launch_flags(
                ["llama-server"], {"supports_kv_unified": False}, env = {}
            )
            == []
        )

    def test_the_inherited_env_twin_counts_as_the_flag(self):
        assert (
            LlamaCppBackend._exact_missing_launch_flags(
                ["llama-server"],
                {"supports_kv_unified": True},
                env = {"LLAMA_ARG_KV_UNIFIED": "1"},
            )
            == []
        )


# --------------------------------------------------------------- recognising the refusal


class TestRefusalDetection:
    @pytest.mark.parametrize("output", [_REFUSAL_THROWN, _REFUSAL_LAYER, _REFUSAL_BOUND])
    def test_every_way_the_server_names_the_mode_is_recognised(self, output):
        assert exact.is_exact_refusal(output) is True

    @pytest.mark.parametrize("output", [_UNRELATED_CRASH, "", None, "exact match not found"])
    def test_nothing_else_is(self, output):
        assert exact.is_exact_refusal(output) is False


class TestAutoFallback:
    def test_a_named_refusal_under_auto_drops_the_variable_and_asks_for_a_relaunch(self):
        env = {exact.CHILD_ENV: "1", "PATH": "/usr/bin"}
        assert (
            LlamaCppBackend._drop_exact_after_refusal(
                env, setting = "auto", crashed = True, output = _REFUSAL_THROWN
            )
            is True
        )
        assert exact.CHILD_ENV not in env
        assert env["PATH"] == "/usr/bin"

    def test_under_on_a_refusal_is_the_answer_rather_than_a_prompt_to_retry(self):
        env = {exact.CHILD_ENV: "1"}
        assert (
            LlamaCppBackend._drop_exact_after_refusal(
                env, setting = "on", crashed = True, output = _REFUSAL_THROWN
            )
            is False
        )
        assert env[exact.CHILD_ENV] == "1", "the load fails carrying the mode it asked for"

    def test_an_unrelated_crash_never_spends_the_fallback(self):
        env = {exact.CHILD_ENV: "1"}
        assert (
            LlamaCppBackend._drop_exact_after_refusal(
                env, setting = "auto", crashed = True, output = _UNRELATED_CRASH
            )
            is False
        )
        assert env[exact.CHILD_ENV] == "1"

    def test_a_child_that_did_not_crash_is_not_a_refusal(self):
        env = {exact.CHILD_ENV: "1"}
        assert (
            LlamaCppBackend._drop_exact_after_refusal(
                env, setting = "auto", crashed = False, output = _REFUSAL_THROWN
            )
            is False
        )

    def test_the_fallback_fires_once(self):
        """Second call is a no-op without a second flag: the variable is already gone, and
        a relaunch loop that kept answering yes would burn the attempt budget the ROCm and
        fit recoveries share."""
        env = {exact.CHILD_ENV: "1"}
        assert (
            LlamaCppBackend._drop_exact_after_refusal(
                env, setting = "auto", crashed = True, output = _REFUSAL_THROWN
            )
            is True
        )
        assert (
            LlamaCppBackend._drop_exact_after_refusal(
                env, setting = "auto", crashed = True, output = _REFUSAL_THROWN
            )
            is False
        )

    def test_the_refusal_message_names_the_setting_and_what_the_mode_needs(self):
        message = LlamaCppBackend._classify_start_failure_text(
            _REFUSAL_THROWN, "/models/x.gguf", "unsloth/x", returncode = 1
        )
        assert "exact concurrency" in message.lower()
        assert "'on'" in message and "'auto'" in message
        assert "unified KV cache" in message
        # The server's own line survives into what the user is shown.
        assert "pass --kv-unified" in message

    def test_an_ordinary_crash_keeps_its_own_message(self):
        message = LlamaCppBackend._classify_start_failure_text(
            _UNRELATED_CRASH, "/models/x.gguf", "unsloth/x", returncode = 1
        )
        assert "exact concurrency is set to" not in message.lower()


# --------------------------------------------------------------------- what the load reports


class TestReportedState:
    def test_off_when_the_load_never_asked(self):
        assert (
            LlamaCppBackend._exact_state_after_launch(setting = "off", env = {}, args = _STUDIO_ARGV)
            == exact.EXACT_STATE_OFF
        )
        # Even with the variable still on the environment: the load resolved to off, so
        # apply_child_env removed it, and a state derived from anything else would lie.
        assert (
            LlamaCppBackend._exact_state_after_launch(
                setting = "off", env = {exact.CHILD_ENV: "1"}, args = _STUDIO_ARGV
            )
            == exact.EXACT_STATE_OFF
        )

    @pytest.mark.parametrize("setting", ["auto", "on"])
    def test_on_when_the_variable_and_the_launch_both_hold(self, setting):
        assert (
            LlamaCppBackend._exact_state_after_launch(
                setting = setting, env = {exact.CHILD_ENV: "1"}, args = _STUDIO_ARGV
            )
            == exact.EXACT_STATE_ON
        )

    def test_unavailable_once_the_fallback_has_taken_the_variable_away(self):
        assert (
            LlamaCppBackend._exact_state_after_launch(setting = "auto", env = {}, args = _STUDIO_ARGV)
            == exact.EXACT_STATE_UNAVAILABLE
        )

    def test_unavailable_when_a_respawn_took_away_what_the_mode_needs(self):
        """The no-flash retry is the live case: the variable is still there and the child
        is healthy, but V is transposed again and the guarantee stopped holding."""
        no_flash = [a for a in _STUDIO_ARGV if a not in ("--flash-attn", "on")]
        assert (
            LlamaCppBackend._exact_state_after_launch(
                setting = "auto", env = {exact.CHILD_ENV: "1"}, args = no_flash + ["--flash-attn", "off"]
            )
            == exact.EXACT_STATE_UNAVAILABLE
        )
        no_unified = [a for a in _STUDIO_ARGV if a != "--kv-unified"]
        assert (
            LlamaCppBackend._exact_state_after_launch(
                setting = "on", env = {exact.CHILD_ENV: "1"}, args = no_unified
            )
            == exact.EXACT_STATE_UNAVAILABLE
        )

    def test_a_quantized_kv_cache_is_not_exact_mode(self):
        assert (
            LlamaCppBackend._exact_state_after_launch(
                setting = "auto",
                env = {exact.CHILD_ENV: "1"},
                args = _STUDIO_ARGV + ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
            )
            == exact.EXACT_STATE_UNAVAILABLE
        )

    def test_a_build_that_ignores_the_variable_is_reported_as_on(self):
        """Recorded, not endorsed. A llama-server from before unslothai/llama.cpp#194 reads
        no such variable and starts normally, and there is no flag, no --help entry and
        nothing in /props to ask instead. Trying it IS the probe, so a build that neither
        implements nor refuses the mode is indistinguishable from one that granted it."""
        assert (
            LlamaCppBackend._exact_state_after_launch(
                setting = "on", env = {exact.CHILD_ENV: "1"}, args = _STUDIO_ARGV
            )
            == exact.EXACT_STATE_ON
        )


class TestBackendProperties:
    def test_a_backend_that_never_launched_reports_off(self):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        assert backend.exact_concurrency == exact.EXACT_STATE_OFF
        assert backend.requested_exact_concurrency == exact.EXACT_OFF

    def test_the_properties_report_what_the_load_recorded(self):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._exact_concurrency = exact.EXACT_STATE_UNAVAILABLE
        backend._requested_exact_concurrency = exact.EXACT_AUTO
        assert backend.exact_concurrency == "unavailable"
        assert backend.requested_exact_concurrency == "auto"

    def test_the_status_schema_carries_both(self):
        """`_llama_runtime_fields` reads every field of the runtime schema off the backend
        by name and raises naming any it cannot resolve, so the schema and the properties
        have to be added together."""
        for name in ("exact_concurrency", "requested_exact_concurrency"):
            assert name in InferenceStatusResponse.model_fields, name
            assert hasattr(LlamaCppBackend, name), name
        assert InferenceStatusResponse.model_fields["exact_concurrency"].default == "off"

    def test_the_status_field_resolves_the_way_the_route_resolves_it(self):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._exact_concurrency = "on"
        backend._requested_exact_concurrency = "on"
        resolved = {
            name: getattr(backend, name, getattr(backend, f"_{name}", None))
            for name in ("exact_concurrency", "requested_exact_concurrency")
        }
        assert resolved == {"exact_concurrency": "on", "requested_exact_concurrency": "on"}


class TestDuplicateLoad:
    """The mode is an environment variable on the child, so changing it needs a new child."""

    def _backend(self, requested):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._is_diffusion = False
        backend._requested_exact_concurrency = requested
        return backend

    def test_the_same_answer_is_not_a_reload(self, monkeypatch):
        monkeypatch.setattr(exact, "stored_exact_setting", lambda: None)
        backend = self._backend("on")
        assert backend.requested_exact_concurrency == exact.resolve_exact_setting(
            GgufLoadIntent(model_identifier = "m", exact_concurrency = "on").exact_concurrency
        )

    def test_a_different_answer_is(self, monkeypatch):
        monkeypatch.setattr(exact, "stored_exact_setting", lambda: None)
        backend = self._backend("off")
        assert backend.requested_exact_concurrency != exact.resolve_exact_setting("on")


# ------------------------------------------------------------------------- what it reports


class TestPreemptionSnapshot:
    def test_the_snapshot_carries_the_mode_and_defaults_to_off(self):
        controller = PreemptionController("exact")
        assert controller.snapshot().exact == exact.EXACT_STATE_OFF
        controller.configure(budget = 8192, kv_unified = True, slots = 4, exact = "on")
        assert controller.snapshot().exact == "on"

    def test_configure_without_the_argument_keeps_it(self):
        controller = PreemptionController("keep-exact")
        controller.configure(budget = 8192, exact = "unavailable")
        controller.configure(budget = 4096)
        assert controller.snapshot().exact == "unavailable"

    def test_an_unknown_state_reads_as_off_rather_than_being_reported_verbatim(self):
        controller = PreemptionController("bogus-exact")
        controller.configure(exact = "probably")
        assert controller.snapshot().exact == exact.EXACT_STATE_OFF

    def test_it_is_reported_and_never_acted_on(self):
        """Exact mode changes what a pause RETURNS to, not who pauses. Turning it on must
        not move a single victim, or the two knobs would silently interact."""
        controller = PreemptionController("exact-policy")
        controller.configure(budget = 8192, kv_unified = True, slots = 4, exact = "on")
        for index in range(4):
            controller.register(f"g{index}", tokens = 2400)
        with_exact = [v.gen_id for v in controller.plan_preemptions()]
        reset_preemption_controllers()
        plain = PreemptionController("exact-policy")
        plain.configure(budget = 8192, kv_unified = True, slots = 4, exact = "off")
        for index in range(4):
            plain.register(f"g{index}", tokens = 2400)
        assert [v.gen_id for v in plain.plan_preemptions()] == with_exact


class _LogRecorder:
    """The module logger is structlog, which `caplog` does not see, so record the call."""

    def __init__(self):
        self.lines: list[str] = []

    def info(self, template, *args):
        self.lines.append(template % args if args else template)

    debug = warning = error = info


class TestArmedLine:
    def _armed(self, monkeypatch, **fields) -> str:
        import routes.inference as inference_routes

        recorder = _LogRecorder()
        monkeypatch.setattr(inference_routes, "logger", recorder)
        inference_routes._llama_preemption_log("armed", **fields)
        return "\n".join(recorder.lines)

    def test_the_armed_line_reports_the_mode(self, monkeypatch):
        line = self._armed(
            monkeypatch,
            gen_id = "chatcmpl-1",
            mode = preemption.PREEMPT_MODE_SERVER,
            exact = "on",
            charged = 512,
        )
        assert "llama preemption armed:" in line
        assert "mode=server" in line
        assert "exact=on" in line

    @pytest.mark.parametrize("state", ["on", "off", "unavailable"])
    def test_every_state_reaches_the_line(self, monkeypatch, state):
        assert f"exact={state}" in self._armed(monkeypatch, gen_id = "g", exact = state)


# ------------------------------------------------------------------------ the stored setting


class TestStoredSetting:
    def _store(
        self,
        monkeypatch,
        initial = None,
    ):
        """The app_settings row, without a database."""
        import storage.studio_db as studio_db
        import utils.exact_concurrency_settings as settings

        rows = {} if initial is None else dict(initial)
        monkeypatch.setattr(
            studio_db, "get_app_setting", lambda key, default = None: rows.get(key, default)
        )
        monkeypatch.setattr(studio_db, "upsert_app_settings", lambda updates: rows.update(updates))
        settings._cache.clear()
        settings._generation.clear()
        return settings, rows

    def test_nothing_stored_reads_as_none_rather_than_off(self, monkeypatch):
        """Not the same answer: nothing stored falls through to an inherited
        LLAMA_EXACT_CONCURRENCY, which is the workaround this switch replaces and which has
        to keep working; a stored off is a user saying no even there."""
        settings, _rows = self._store(monkeypatch)
        assert settings.get_exact_concurrency() is None

    @pytest.mark.parametrize("value", ["auto", "off", "on"])
    def test_a_round_trip(self, monkeypatch, value):
        settings, rows = self._store(monkeypatch)
        assert settings.set_exact_concurrency(value) == value
        assert rows[settings.EXACT_CONCURRENCY_SETTING_KEY] == value
        assert settings.get_exact_concurrency() == value

    def test_a_value_that_is_not_one_of_the_three_is_refused(self, monkeypatch):
        settings, rows = self._store(monkeypatch)
        with pytest.raises(ValueError):
            settings.set_exact_concurrency("sometimes")
        assert rows == {}

    def test_a_stored_value_that_is_no_longer_valid_reads_as_unset(self, monkeypatch):
        settings, _rows = self._store(monkeypatch, {"llama_exact_concurrency": "maybe"})
        assert settings.get_exact_concurrency() is None

    def test_an_unreadable_store_never_fails_a_load(self, monkeypatch):
        import storage.studio_db as studio_db
        import utils.exact_concurrency_settings as settings

        def _boom(key, default = None):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(studio_db, "get_app_setting", _boom)
        settings._cache.clear()
        settings._generation.clear()
        assert settings.get_exact_concurrency() is None
        assert exact.stored_exact_setting() is None
        assert exact.resolve_exact_setting(None, environ = {}) == exact.EXACT_OFF
