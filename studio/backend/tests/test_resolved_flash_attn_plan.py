# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The estimate and the launch price the same flash-attention state.

``load_model`` used to pin ``planned_flash_attn = False`` while emitting ``--flash-attn on``,
so every placement figure described a load that was not going to happen (#9697, #10489). The
cushion that pin gave is not lost: tensor mode cannot take the FA-off recovery at all, and
elsewhere the respawn re-enters ``_spawn_and_wait``, which re-places it with ``--fit on``.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = str(_TESTS_DIR.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _TESTS_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_placement = _load("_placement_for_flash_attn_plan", "test_llama_cpp_placement.py")

from core.inference.llama_cpp import (  # noqa: E402
    LlamaCppBackend,
    _planned_flash_attn_state,
)

GB = 1024**3
NATIVE = 262144


class TestTheResolver:
    def test_the_managed_default_is_on(self):
        assert _planned_flash_attn_state() is True

    def test_a_build_without_the_flag_is_off(self):
        """No flag emitted, so the padded, f16-floored arm is the right price."""
        assert _planned_flash_attn_state(supports_flash_attn = False) is False

    @pytest.mark.parametrize(
        "extras",
        [
            ["--flash-attn", "off"],
            ["--flash-attn=off"],
            ["-fa", "off"],
            ["-fa=0"],
        ],
    )
    def test_an_explicit_off_in_the_extras_wins(self, extras):
        assert _planned_flash_attn_state(extras) is False

    def test_the_last_flag_wins_like_llama_cpp(self):
        assert _planned_flash_attn_state(["-fa", "off", "--flash-attn", "on"]) is True

    def test_the_environment_loses_to_the_managed_flag(self):
        """The env is read before argv (arg.cpp set_env), so the managed on wins."""
        assert _planned_flash_attn_state(env = {"LLAMA_ARG_FLASH_ATTN": "0"}) is True

    def test_a_user_off_still_beats_the_managed_flag(self):
        """Order of authority: environment, managed flag, then the user's extras."""
        assert _planned_flash_attn_state(["-fa", "off"], env = {"LLAMA_ARG_FLASH_ATTN": "0"}) is False
        assert _planned_flash_attn_state(["-fa", "off"], env = {}) is False

    def test_the_extras_beat_the_environment(self):
        assert (
            _planned_flash_attn_state(["--flash-attn", "on"], env = {"LLAMA_ARG_FLASH_ATTN": "0"})
            is True
        )

    @pytest.mark.parametrize("v_type", ["q8_0", "q4_0", "q5_1", "iq4_nl"])
    def test_a_quantized_v_cache_forces_it_on(self, v_type):
        """llama.cpp turns it on itself, so an explicit off does not survive."""
        assert (
            _planned_flash_attn_state(["--flash-attn", "off"], planned_cache_types = ("f16", v_type))
            is True
        )

    @pytest.mark.parametrize("v_type", ["f16", "bf16", "f32"])
    def test_an_unquantized_v_cache_forces_nothing(self, v_type):
        assert (
            _planned_flash_attn_state(["--flash-attn", "off"], planned_cache_types = ("f16", v_type))
            is False
        )

    def test_a_quantized_v_cannot_force_a_build_that_has_no_flag(self):
        """The launch rewrites V to f16 instead, so forcing "on" would under-reserve."""
        assert (
            _planned_flash_attn_state(
                planned_cache_types = ("q8_0", "q8_0"), supports_flash_attn = False
            )
            is False
        )


def _tensor_backend(tmp_path, *, free_mib: int):
    backend, gguf = _placement._backend(
        tmp_path,
        vulkan = False,
        memory = [(0, free_mib, free_mib), (1, free_mib, free_mib)],
    )
    # The harness turns KV estimation off; seed a real shape (Qwen3-0.6B) for arithmetic.
    backend._can_estimate_kv = lambda: True
    backend._n_layers = 28
    backend._embedding_length = 1024
    backend._n_heads = 16
    backend._n_kv_heads = 8
    backend._kv_key_length = 128
    backend._kv_value_length = 128
    backend._context_length = NATIVE
    backend._get_gguf_size_bytes = lambda _path: 4 * GB
    return backend, gguf


def _ctx_of(cmd) -> int:
    values = [cmd[i + 1] for i, token in enumerate(cmd) if token in ("-c", "--ctx-size")]
    assert values, f"no context in argv: {cmd}"
    return int(values[-1])


def _flash_attn_in(cmd) -> bool:
    for i, token in enumerate(cmd):
        if token in ("--flash-attn", "-fa"):
            return i + 1 >= len(cmd) or cmd[i + 1] != "off"
    return False


class TestTheTensorPlanPricesTheLaunch:
    """A pool too small at f16-priced V and large enough once V is priced as launched."""

    def test_the_planned_context_matches_the_emitted_flash_attention(self, tmp_path):
        backend, gguf = _tensor_backend(tmp_path, free_mib = 12000)
        captured = _placement._launch(
            backend, gguf, n_ctx = NATIVE, cache_type_kv = "q8_0", tensor_parallel = True
        )
        cmd = captured["cmd"]
        assert _flash_attn_in(cmd), "the launch emits flash attention"
        planned = _ctx_of(cmd)

        # The gap is the defect, so assert the side of it rather than a magic number.
        pessimistic = backend._plan_tensor_parallel(
            [(0, 12000), (1, 12000)],
            4 * GB,
            NATIVE,
            cache_type_kv = "q8_0",
            max_target_ctx = NATIVE,
            flash_attn = False,
        )[0]
        optimistic = backend._plan_tensor_parallel(
            [(0, 12000), (1, 12000)],
            4 * GB,
            NATIVE,
            cache_type_kv = "q8_0",
            max_target_ctx = NATIVE,
            flash_attn = True,
        )[0]
        assert pessimistic < optimistic, "harness no longer straddles the difference"
        assert planned > pessimistic, (
            f"the tensor plan emitted {planned} tokens, at or below the {pessimistic} a "
            f"flash-attention-off cache prices, while the launch runs with flash "
            f"attention on and fits {optimistic}"
        )

    def test_the_published_ceiling_is_the_same_number(self, tmp_path):
        """max_context_length is what the context warning compares against."""
        backend, gguf = _tensor_backend(tmp_path, free_mib = 12000)
        captured = _placement._launch(
            backend, gguf, n_ctx = NATIVE, cache_type_kv = "q8_0", tensor_parallel = True
        )
        assert backend.max_context_length == _ctx_of(captured["cmd"])

    def test_an_unquantized_cache_is_unaffected(self, tmp_path):
        """On the arithmetic, not a hand-built plan: the loader charges extra overheads."""
        backend, gguf = _tensor_backend(tmp_path, free_mib = 12000)
        captured = _placement._launch(
            backend, gguf, n_ctx = NATIVE, cache_type_kv = "f16", tensor_parallel = True
        )
        planned = _ctx_of(captured["cmd"])
        assert backend._estimate_kv_cache_bytes(
            planned, "f16", flash_attn = True
        ) == backend._estimate_kv_cache_bytes(planned, "f16", flash_attn = False)


class TestTheSizingCallsSeeTheResolvedState:
    def _seen_flash_attn(self, tmp_path, **load_kwargs):
        backend, gguf = _tensor_backend(tmp_path, free_mib = 12000)
        seen: list[object] = []
        real = backend._estimate_kv_cache_bytes

        def spy(
            n_ctx,
            cache_type_kv = None,
            **kwargs,
        ):
            seen.append(kwargs.get("flash_attn"))
            return real(n_ctx, cache_type_kv, **kwargs)

        backend._estimate_kv_cache_bytes = spy
        _placement._launch(backend, gguf, **load_kwargs)
        assert seen, "no KV estimate was taken during the load"
        return seen

    def test_a_quantized_cache_is_priced_with_flash_attention_on(self, tmp_path):
        seen = self._seen_flash_attn(
            tmp_path, n_ctx = NATIVE, cache_type_kv = "q8_0", tensor_parallel = True
        )
        assert all(state is True for state in seen), (
            f"the load priced its KV cache with flash_attn states {sorted(set(map(str, seen)))} "
            f"while emitting --flash-attn on"
        )

    def test_an_explicit_off_in_the_extras_is_priced_off(self, tmp_path):
        seen = self._seen_flash_attn(
            tmp_path,
            n_ctx = NATIVE,
            cache_type_kv = "f16",
            extra_args = ["--flash-attn", "off"],
        )
        assert all(state is False for state in seen)


def test_the_state_is_still_named_planned_flash_attn():
    """The offload-planner seam asserts on this identifier by AST."""
    import inspect

    source = inspect.getsource(inspect.unwrap(LlamaCppBackend.load_model))
    # Through the reserve wrapper now, which only holds a reading DOWN.
    assert "planned_flash_attn = _reserved_flash_attn_state(" in source
    assert "_planned_flash_attn_state(" in source
    assert "planned_flash_attn = False" not in source


class TestAutoIsNotAnAnswer:
    """``auto`` is decided at load time and silently decided against whenever the backend,
    model or cache pair cannot take it, so reading it as "on" under-reserves."""

    def test_an_explicit_auto_prices_the_padded_cache(self):
        assert _planned_flash_attn_state(["--flash-attn", "auto"]) is False
        assert _planned_flash_attn_state(["-fa", "auto"]) is False
        assert _planned_flash_attn_state(["-fa=auto"]) is False
        # llama.cpp's numeric spelling of the same value.
        assert _planned_flash_attn_state(["-fa", "-1"]) is False

    def test_an_inherited_auto_is_overridden_by_the_managed_flag(self):
        """The managed flag is appended after the env is read, so only a USER auto survives."""
        assert _planned_flash_attn_state(None, env = {"LLAMA_ARG_FLASH_ATTN": "auto"}) is True
        assert (
            _planned_flash_attn_state(["-fa", "auto"], env = {"LLAMA_ARG_FLASH_ATTN": "1"}) is False
        )
        assert (
            _planned_flash_attn_state(["-fa", "on"], env = {"LLAMA_ARG_FLASH_ATTN": "auto"}) is True
        )
        assert (
            _planned_flash_attn_state(["-fa", "auto"], env = {"LLAMA_ARG_FLASH_ATTN": "1"}) is False
        )

    def test_the_last_flag_still_wins(self):
        assert _planned_flash_attn_state(["-fa", "auto", "-fa", "on"]) is True
        assert _planned_flash_attn_state(["-fa", "on", "-fa", "auto"]) is False

    def test_a_quantized_v_cache_still_forces_it_on(self):
        assert (
            _planned_flash_attn_state(["-fa", "auto"], planned_cache_types = ("q8_0", "q4_0")) is True
        )

    def test_tensor_split_decides_auto_rather_than_leaving_it_open(self):
        """llama.cpp upgrades AUTO to ENABLED under SPLIT_MODE_TENSOR, so pricing the
        padded V there charges a cache the child never allocates."""
        # The three ways the launch decides the mode: toggle, extras, inherited env.
        assert _planned_flash_attn_state(["-fa", "auto"], tensor_parallel = True) is True
        assert _planned_flash_attn_state(["-fa", "auto", "--split-mode", "tensor"], env = {}) is True
        assert (
            _planned_flash_attn_state(["-fa", "auto"], env = {"LLAMA_ARG_SPLIT_MODE": "tensor"})
            is True
        )
        # An extras split mode last-wins over the toggle, so layer stays undecided.
        assert (
            _planned_flash_attn_state(
                ["-fa", "auto", "--split-mode", "layer"], tensor_parallel = True, env = {}
            )
            is False
        )
        assert _planned_flash_attn_state(["-fa", "auto"], env = {}) is False
        # A user OFF is not silently flipped on: refusing the pair is the launch's job.
        assert _planned_flash_attn_state(["-fa", "off"], tensor_parallel = True) is False

    def test_the_managed_launch_is_unaffected(self):
        assert _planned_flash_attn_state(None) is True
        assert _planned_flash_attn_state([]) is True


class TestTheDowngradesRePlanTheAttention:
    """The plan is made once near the top of the load and the split is decided later, so a
    stale True budgets an unpadded V the layer split will not get."""

    def _load_model_body(self):
        import ast
        import inspect

        from core.inference import llama_cpp as module

        source = inspect.getsource(module.LlamaCppBackend.load_model)
        return ast.parse(textwrap.dedent(source)).body[0]

    def test_every_tensor_downgrade_re_plans_the_attention(self):
        """Not only the ones that strip the extras: the three manual guards set the toggle
        to False on their own, and Manual placement runs with --fit off."""
        import ast

        func = self._load_model_body()
        # Only downgrades AFTER the plan is made; virtualised Metal drops it before.
        planned_at = min(
            node.lineno
            for node in ast.walk(func)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_planned_flash_attn_state"
        )
        checked = 0
        for node in ast.walk(func):
            for attr in ("body", "orelse", "finalbody"):
                block = getattr(node, attr, None)
                if not isinstance(block, list):
                    continue
                for index, statement in enumerate(block):
                    if not (
                        isinstance(statement, ast.Assign)
                        and getattr(statement.targets[0], "id", None) == "tensor_parallel"
                        and isinstance(statement.value, ast.Constant)
                        and statement.value.value is False
                        and statement.lineno > planned_at
                    ):
                        continue
                    checked += 1
                    following = [
                        getattr(getattr(later, "targets", [None])[0], "id", None)
                        for later in block[index + 1 : index + 4]
                        if isinstance(later, ast.Assign)
                    ]
                    assert "planned_flash_attn" in following, (
                        "a tensor downgrade at line "
                        f"{statement.lineno} does not re-plan the attention"
                    )
        assert checked >= 6, checked

    def test_every_split_mode_strip_re_plans_the_attention(self):
        import ast

        func = self._load_model_body()
        strips = 0
        for node in ast.walk(func):
            for attr in ("body", "orelse", "finalbody"):
                block = getattr(node, attr, None)
                if not isinstance(block, list):
                    continue
                for index, statement in enumerate(block):
                    if not isinstance(statement, ast.Assign):
                        continue
                    call = statement.value
                    if not (
                        isinstance(call, ast.Call)
                        and getattr(call.func, "id", None) == "strip_split_mode_only"
                    ):
                        continue
                    strips += 1
                    following = block[index + 1] if index + 1 < len(block) else None
                    assert isinstance(following, ast.Assign), ast.dump(statement)
                    assert (
                        getattr(following.targets[0], "id", None) == "planned_flash_attn"
                    ), ast.dump(following)
        assert strips >= 6, strips

    def test_the_re_plan_reads_the_current_split(self):
        import ast

        func = self._load_model_body()
        helper = next(
            node
            for node in ast.walk(func)
            if isinstance(node, ast.FunctionDef) and node.name == "_replanned_flash_attn"
        )
        call = next(
            node
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_planned_flash_attn_state"
        )
        passed = {kw.arg: kw.value for kw in call.keywords}
        assert getattr(passed["tensor_parallel"], "id", None) == "_current_tp"
        assert getattr(call.args[0], "id", None) == "extra_args"


class TestTheReserveWhenTheFitterIsOff:
    """The reserve has to survive the respawn too, and a user's own `--fit off` is what
    takes the re-placement away."""

    def test_a_user_fit_off_keeps_the_conservative_reserve(self):
        from core.inference.llama_cpp import _reserved_flash_attn_state

        # The ordinary managed launch is unchanged: fitting is on, so the respawn re-places.
        assert _reserved_flash_attn_state(True, None, env = {}) is True
        assert _reserved_flash_attn_state(True, ["--ctx-size", "4096"], env = {}) is True
        # With the fitter off, the respawn lands on the placement this reserve chose.
        assert _reserved_flash_attn_state(True, ["--fit", "off"], env = {}) is False
        assert _reserved_flash_attn_state(True, ["--fit=off"], env = {}) is False
        assert _reserved_flash_attn_state(True, None, env = {"LLAMA_ARG_FIT": "off"}) is False
        # Last-wins, like every other flag: a later --fit on is the state that runs.
        assert _reserved_flash_attn_state(True, ["--fit", "off", "--fit", "on"], env = {}) is True
        assert (
            _reserved_flash_attn_state(True, ["--fit", "on"], env = {"LLAMA_ARG_FIT": "off"}) is True
        )

    def test_tensor_mode_keeps_its_answer(self):
        """No no-flash respawn to reserve for under SPLIT_MODE_TENSOR, and pricing the
        padded layout would refuse loads that fit."""
        from core.inference.llama_cpp import _reserved_flash_attn_state

        assert (
            _reserved_flash_attn_state(True, ["--fit", "off"], tensor_parallel = True, env = {}) is True
        )
        assert (
            _reserved_flash_attn_state(True, ["--fit", "off", "--split-mode", "tensor"], env = {})
            is True
        )

    def test_a_false_plan_is_never_raised(self):
        """This only holds a reading DOWN; an off is already the conservative one."""
        from core.inference.llama_cpp import _reserved_flash_attn_state

        assert _reserved_flash_attn_state(False, ["--fit", "off"], env = {}) is False
        assert _reserved_flash_attn_state(False, None, env = {}) is False

    def test_the_load_reserves_through_it(self):
        import inspect

        from core.inference import llama_cpp as module

        body = inspect.getsource(module.LlamaCppBackend.load_model)
        assert "_reserved_flash_attn_state(" in body
        # Both the first plan and every re-plan, or a downgrade would put the raw reading back.
        assert body.count("_reserved_flash_attn_state(") == 2, body.count(
            "_reserved_flash_attn_state("
        )


class TestTheReplanIsAuthoritative:
    """A downgrade strips the extras, never the env, but the helper also resolves an
    inherited LLAMA_ARG_SPLIT_MODE, so a re-plan carrying the downgraded False still met a
    tensor answer inside it. The launch clears that variable for exactly this reason."""

    def _load_model_body(self):
        import ast
        import inspect
        import textwrap

        from core.inference import llama_cpp as module

        source = inspect.getsource(module.LlamaCppBackend.load_model)
        return ast.parse(textwrap.dedent(source)).body[0]

    def test_the_replan_scrubs_the_inherited_split_mode(self):
        import ast

        func = self._load_model_body()
        helper = next(
            node
            for node in ast.walk(func)
            if isinstance(node, ast.FunctionDef) and node.name == "_replan_env"
        )
        text = ast.dump(helper)
        assert "LLAMA_ARG_SPLIT_MODE" in text
        assert "LLAMA_ARG_TENSOR_SPLIT" in text
        replan = next(
            node
            for node in ast.walk(func)
            if isinstance(node, ast.FunctionDef) and node.name == "_replanned_flash_attn"
        )
        calls = [
            node
            for node in ast.walk(replan)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None)
            in {"_planned_flash_attn_state", "_reserved_flash_attn_state"}
        ]
        assert len(calls) == 2, [getattr(c.func, "id", None) for c in calls]
        for call in calls:
            passed = {kw.arg for kw in call.keywords}
            assert "env" in passed, ast.dump(call)

    def test_the_helper_answers_layer_once_the_variable_is_gone(self):
        from core.inference.llama_cpp import _planned_flash_attn_state

        inherited = {"LLAMA_ARG_SPLIT_MODE": "tensor"}
        # The state the re-plan used to resolve: a downgraded toggle, a tensor environment.
        assert (
            _planned_flash_attn_state(["-fa", "auto"], tensor_parallel = False, env = inherited) is True
        )
        assert _planned_flash_attn_state(["-fa", "auto"], tensor_parallel = False, env = {}) is False
