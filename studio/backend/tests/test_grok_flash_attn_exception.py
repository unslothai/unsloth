# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Grok is priced without flash attention, whatever else the launch resolves.

llama.cpp decides this before it reads anything else, at the top of
``llama_init_from_model`` (llama-context.cpp)::

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED
            && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("flash_attn is not compatible with Grok - forcing off");
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

Both later upgrades sit BELOW it -- ``enabling flash_attn since it is required for
SPLIT_MODE_TENSOR`` and ``enabling flash_attn since it is required for quantized V
cache`` -- so on Grok neither can put it back. The reason is in llama-graph.cpp's
``LLM_ARCH_GROK`` branch: the attention logits are softcapped (``30 * tanh(kq / 30)``)
outside the fused kernel.

Pricing it any other way is not conservative, it is a different server: with flash
attention off llama.cpp pads every layer's V to ``n_embd_v_gqa_max()`` model-wide and
floors it at f16 (``_max_kv_value_width``), and the KQ mask becomes f32 rather than f16.
A resolution that answered "on" for Grok would publish a context the child cannot hold.

This file exists because the rule is easy to lose in a merge: it arrived on one side of
the tree (#11043, as a separate ``_planned_flash_attn`` helper) while the single resolved
state arrived on the other (#9697, #10489, ``_planned_flash_attn_state``). Taking either
side alone drops something, so these tests hold BOTH ends: the rule itself, and the
architecture actually reaching the resolver at every seam that prices a load.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
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


_placement = _load("_placement_for_grok_flash_attn", "test_llama_cpp_placement.py")
_plan = _load("_plan_for_grok_flash_attn", "test_resolved_flash_attn_plan.py")

from core.inference.llama_cpp import (  # noqa: E402
    LlamaCppBackend,
    _architecture_forces_flash_attn_off,
    _planned_flash_attn_state,
    _reserved_flash_attn_state,
)

NATIVE = _plan.NATIVE


# ----------------------------------------------------------------- the rule itself


class TestTheRule:
    def test_grok_is_forced_off(self):
        assert _planned_flash_attn_state(architecture = "grok") is False

    def test_the_architecture_is_read_as_the_gguf_stores_it(self):
        """general.architecture is copied out of the header verbatim, so normalise here
        rather than trusting every writer to have lower-cased it."""
        assert _architecture_forces_flash_attn_off("grok") is True
        assert _architecture_forces_flash_attn_off("Grok") is True
        assert _architecture_forces_flash_attn_off(" grok ") is True

    @pytest.mark.parametrize("architecture", [None, "", "qwen3", "llama", "grok2", "deepseek2"])
    def test_every_other_architecture_keeps_the_managed_default(self, architecture):
        """The exception is one architecture, not a new default. An unread header
        (None) keeps the managed ``--flash-attn on``, the same way an unreadable
        capability probe does."""
        assert _architecture_forces_flash_attn_off(architecture) is False
        assert _planned_flash_attn_state(architecture = architecture) is True


class TestNothingPutsItBack:
    """Every other branch in the resolver sits below llama.cpp's own Grok branch."""

    @pytest.mark.parametrize("v_type", ["q8_0", "q4_0", "q5_1", "iq4_nl"])
    def test_a_quantized_v_cache_does_not(self, v_type):
        """ "enabling flash_attn since it is required for quantized V cache" is the
        branch below the Grok one, so it never runs on Grok. (llama.cpp then refuses
        that load outright -- "quantized V cache requires flash_attn to be enabled" --
        which is a launch that cannot happen either way, so the padded arm is the only
        honest price.)"""
        assert (
            _planned_flash_attn_state(planned_cache_types = ("f16", v_type), architecture = "grok")
            is False
        )
        # And the same pair on any other architecture still forces it on, so this test
        # is straddling the rule rather than asserting a constant.
        assert (
            _planned_flash_attn_state(planned_cache_types = ("f16", v_type), architecture = "qwen3")
            is True
        )

    def test_a_tensor_split_does_not(self):
        """SPLIT_MODE_TENSOR upgrades AUTO to ENABLED, below the Grok branch."""
        assert (
            _planned_flash_attn_state(["-fa", "auto"], tensor_parallel = True, architecture = "grok")
            is False
        )
        assert (
            _planned_flash_attn_state(["-fa", "auto"], tensor_parallel = True, architecture = "qwen3")
            is True
        )

    def test_an_explicit_on_in_the_extras_does_not(self):
        assert _planned_flash_attn_state(["--flash-attn", "on"], architecture = "grok") is False
        assert _planned_flash_attn_state(["-fa", "1"], architecture = "grok") is False

    def test_the_inherited_environment_does_not(self):
        assert (
            _planned_flash_attn_state(architecture = "grok", env = {"LLAMA_ARG_FLASH_ATTN": "1"})
            is False
        )

    def test_the_reserve_wrapper_keeps_it_off(self):
        """The wrapper only ever holds the planned reading DOWN, so an off stays off --
        including in tensor mode, which is the one case it returns ``planned`` unread."""
        for tensor_parallel in (False, True):
            planned = _planned_flash_attn_state(
                ["-fa", "on"],
                planned_cache_types = ("q8_0", "q8_0"),
                tensor_parallel = tensor_parallel,
                architecture = "grok",
            )
            assert (
                _reserved_flash_attn_state(planned, ["-fa", "on"], tensor_parallel = tensor_parallel)
                is False
            )


# ------------------------------------------------- the rule reaching a real load


def _grok_backend(tmp_path, architecture: str):
    backend, gguf = _plan._tensor_backend(tmp_path, free_mib = 12000)
    # _backend() stubs _read_gguf_metadata out, so this survives the load.
    backend._architecture = architecture
    return backend, gguf


def _states_seen(tmp_path, architecture: str, method: str, **load_kwargs):
    """Every ``flash_attn`` the load hands to ``method`` while it sizes itself."""
    backend, gguf = _grok_backend(tmp_path, architecture)
    seen: list[object] = []
    real = getattr(backend, method)

    def spy(*args, **kwargs):
        seen.append(kwargs.get("flash_attn"))
        return real(*args, **kwargs)

    setattr(backend, method, spy)
    _placement._launch(backend, gguf, **load_kwargs)
    assert seen, f"the load took no {method} reading"
    return seen


class TestTheLoadPricesGrokWithoutIt:
    """The load, end to end, not the helper. A quantized V cache under a tensor split is
    the hardest case: both of llama.cpp's upgrades are asking for flash attention at once,
    and on Grok neither of them gets it."""

    def test_the_kv_cache_is_priced_without_it(self, tmp_path):
        seen = _states_seen(
            tmp_path,
            "grok",
            "_estimate_kv_cache_bytes",
            n_ctx = NATIVE,
            cache_type_kv = "q8_0",
            tensor_parallel = True,
        )
        assert all(state is False for state in seen), (
            f"the load priced its KV cache with flash_attn states "
            f"{sorted(set(map(str, seen)))} on an architecture llama.cpp will not run "
            f"with flash attention"
        )

    def test_the_same_load_on_another_architecture_is_priced_with_it(self, tmp_path):
        """The control arm. Without this the test above passes on a resolver that
        answers False for everything."""
        seen = _states_seen(
            tmp_path,
            "qwen3",
            "_estimate_kv_cache_bytes",
            n_ctx = NATIVE,
            cache_type_kv = "q8_0",
            tensor_parallel = True,
        )
        assert all(state is True for state in seen)

    def test_the_compute_buffer_is_priced_without_it(self, tmp_path):
        """The KQ mask is f16 under flash attention and f32 without it, and the no-flash
        arm adds f32 scores per token, so the compute reserve moves with the same state
        the KV cache does."""
        seen = _states_seen(
            tmp_path,
            "grok",
            "_compute_buffer_ctx_bytes",
            n_ctx = NATIVE,
            cache_type_kv = "q8_0",
            tensor_parallel = True,
        )
        assert all(state is False for state in seen), (
            f"the load priced its compute buffers with flash_attn states "
            f"{sorted(set(map(str, seen)))} on Grok"
        )

    def test_the_compute_buffer_control_arm_is_priced_with_it(self, tmp_path):
        seen = _states_seen(
            tmp_path,
            "qwen3",
            "_compute_buffer_ctx_bytes",
            n_ctx = NATIVE,
            cache_type_kv = "q8_0",
            tensor_parallel = True,
        )
        assert all(state is True for state in seen)

    def test_the_padded_cache_actually_costs_more(self, tmp_path):
        """Reachability: the two states have to price differently, or the assertions
        above are asserting nothing."""
        backend, _ = _grok_backend(tmp_path, "grok")
        on = backend._estimate_kv_cache_bytes(NATIVE, "q8_0", flash_attn = True)
        off = backend._estimate_kv_cache_bytes(NATIVE, "q8_0", flash_attn = False)
        assert off > on
        assert backend._compute_buffer_ctx_bytes(
            NATIVE, 512, "q8_0", flash_attn = False
        ) > backend._compute_buffer_ctx_bytes(NATIVE, 512, "q8_0", flash_attn = True)


# ------------------------------------- the architecture reaching every other seam


def _calls_to(source: str, name: str) -> list[ast.Call]:
    tree = ast.parse(textwrap.dedent(source))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    ]


class TestEverySeamHandsTheArchitectureOver:
    """Read by AST rather than by text: a comment mentioning the architecture would pass a
    string match, and a reflow would fail one. Neither says what the call passes.
    """

    def test_the_loader_does(self):
        source = inspect.getsource(inspect.unwrap(LlamaCppBackend.load_model))
        calls = _calls_to(source, "_planned_flash_attn_state")
        assert calls, "the launch path no longer resolves an attention plan"
        for call in calls:
            passed = {kw.arg for kw in call.keywords if kw.arg}
            assert "architecture" in passed, (
                "load_model resolves its attention plan without telling the resolver "
                "which architecture it is, so the Grok exception cannot fire"
            )

    def test_the_memory_estimate_route_does(self):
        import routes.inference as ri

        source = inspect.getsource(ri._gguf_runtime_bytes)
        calls = _calls_to(source, "_planned_flash_attn_state")
        assert calls, "the estimate no longer resolves an attention plan"
        for call in calls:
            assert "architecture" in {kw.arg for kw in call.keywords if kw.arg}

    def test_the_kv_cache_estimate_route_does(self):
        """routes/models.get_kv_cache_estimate, read from the file: the route is an async
        closure over a large body and importing it drags the whole app in."""
        source = (Path(_BACKEND_DIR) / "routes" / "models.py").read_text(encoding = "utf-8")
        calls = _calls_to(source, "_planned_flash_attn_state")
        assert calls, "the kv-cache-estimate route no longer resolves an attention plan"
        for call in calls:
            assert "architecture" in {kw.arg for kw in call.keywords if kw.arg}


class TestOneStateNotTwo:
    """The other half of the collision this file guards.

    The Grok rule can also be lost by RE-DERIVING the state somewhere else: a second
    helper call beside the resolved one drifts from it the moment either is changed, and
    that is how the compute buffers came to be priced from their own reading. Both
    sizing seams in ``load_model`` read the one resolved name, which is also what lets a
    tensor-mode downgrade re-price compute and not just KV.
    """

    def test_both_sizing_seams_price_the_one_resolved_state(self):
        source = inspect.getsource(inspect.unwrap(LlamaCppBackend.load_model))
        for method in ("_estimate_kv_cache_bytes", "_compute_buffer_ctx_bytes"):
            calls = _calls_to(source, method)
            assert calls, f"the launch path no longer calls {method}"
            for call in calls:
                passed = {
                    kw.arg: kw.value
                    for kw in call.keywords
                    if kw.arg and isinstance(kw.value, ast.Name)
                }
                assert getattr(passed.get("flash_attn"), "id", None) == "planned_flash_attn", (
                    f"{method} is priced from something other than the resolved "
                    f"planned_flash_attn, so the two can disagree about the same load"
                )

    def test_there_is_only_one_resolver(self):
        """A second module-level resolver is exactly how the rule got duplicated, and a
        duplicate carrying only some of the rules is worse than none."""
        import core.inference.llama_cpp as llama_cpp

        resolvers = sorted(
            name
            for name in dir(llama_cpp)
            if name.startswith("_planned_flash_attn") or name.startswith("_reserved_flash_attn")
        )
        assert resolvers == ["_planned_flash_attn_state", "_reserved_flash_attn_state"]
