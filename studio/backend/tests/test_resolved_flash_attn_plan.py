# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The estimate and the launch price the same flash-attention state.

``load_model`` used to pin ``planned_flash_attn = False`` unconditionally, as a cushion for
the crash recovery that may relaunch a plan with flash attention off, and then emit
``--flash-attn on`` on every launch whose build has the flag. So every placement figure
described a load that was not going to happen.

The estimator is not indifferent to that. With flash attention off it floors the V axis at
f16 and pads variable-width V tensors to the model-wide maximum, which on Qwen3's shape at
262,144 is

    q8_0   15,971,909,632 on   vs   23,018,340,352 off   (1.44x)
    q4_0    8,455,716,864 on   vs   19,260,243,968 off   (2.28x)

In tensor mode ``--fit`` is a no-op, so that inflated cache is a hard cap and becomes the
published ``max_context_length``: the report behind #9697 is a 5090 + 3070 pair where q8_0
was selectable but the context came out sized as though the cache were fp16. The same
pessimistic ceiling is what the context warning compares against while the memory panel
prices the optimistic one, which is #10489.

The cushion is not lost. Tensor mode cannot take the FA-off recovery at all -- llama.cpp
requires flash attention for ``SPLIT_MODE_TENSOR``, which
``test_tensor_quant_kv_platform_matrix.py`` already pins -- and elsewhere the no-flash
respawn re-enters ``_spawn_and_wait``, whose own rung hands placement back to llama.cpp
with ``--fit on`` when a forced ``--fit off`` crashes at startup.
"""

from __future__ import annotations

import importlib.util
import sys
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


# The placement harness owns the fake GPU probe, the stub GGUF and the captured Popen.
_placement = _load("_placement_for_flash_attn_plan", "test_llama_cpp_placement.py")

from core.inference.llama_cpp import (  # noqa: E402
    LlamaCppBackend,
    _planned_flash_attn_state,
)

GB = 1024**3
NATIVE = 262144


# ── the resolver, in isolation ────────────────────────────────────────────────


class TestTheResolver:
    def test_the_managed_default_is_on(self):
        """What the launch emits when nothing says otherwise: --flash-attn on."""
        assert _planned_flash_attn_state() is True

    def test_a_build_without_the_flag_is_off(self):
        """Nothing is emitted, so the child runs without it and the padded, f16-floored
        arm of the estimator is the right price."""
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

    def test_the_environment_is_read(self):
        assert _planned_flash_attn_state(env = {"LLAMA_ARG_FLASH_ATTN": "0"}) is False

    def test_the_extras_beat_the_environment(self):
        """llama.cpp applies LLAMA_ARG_* before parsing argv, so the CLI still wins."""
        assert (
            _planned_flash_attn_state(["--flash-attn", "on"], env = {"LLAMA_ARG_FLASH_ATTN": "0"})
            is True
        )

    @pytest.mark.parametrize("v_type", ["q8_0", "q4_0", "q5_1", "iq4_nl"])
    def test_a_quantized_v_cache_forces_it_on(self, v_type):
        """Not a choice: llama.cpp logs "enabling flash_attn since it is required for
        quantized V cache" and turns it on itself, so an explicit off does not survive."""
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
        """The one case the route's own two-step version got wrong. With no --flash-attn
        to emit, the launch rewrites the V cache to f16 instead (_reset_quantized_v_cache),
        so the padded price is the honest one and forcing "on" would under-reserve."""
        assert (
            _planned_flash_attn_state(
                planned_cache_types = ("q8_0", "q8_0"), supports_flash_attn = False
            )
            is False
        )


# ── through the real load_model ───────────────────────────────────────────────


def _tensor_backend(tmp_path, *, free_mib: int):
    backend, gguf = _placement._backend(
        tmp_path,
        vulkan = False,
        memory = [(0, free_mib, free_mib), (1, free_mib, free_mib)],
    )
    # The harness turns KV estimation off (it is about argv, not arithmetic); this file is
    # about the arithmetic, so seed a real shape: Qwen3-0.6B's, whose ratios are measured.
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
    """A two-card pool too small for the full context at f16-priced V, and large enough
    for it once the quantized V is priced the way the launch will run it."""

    def test_the_planned_context_matches_the_emitted_flash_attention(self, tmp_path):
        backend, gguf = _tensor_backend(tmp_path, free_mib = 12000)
        captured = _placement._launch(
            backend, gguf, n_ctx = NATIVE, cache_type_kv = "q8_0", tensor_parallel = True
        )
        cmd = captured["cmd"]
        assert _flash_attn_in(cmd), "the launch emits flash attention"
        planned = _ctx_of(cmd)

        # What the same pool prices with the V axis at f16, i.e. the plan main used to
        # publish for this launch. The gap is the defect, so assert the plan is on the
        # right side of it rather than on a magic number.
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
        """max_context_length is what the context warning compares against, so a ceiling
        priced off a different plan than the launch is #10489's third number."""
        backend, gguf = _tensor_backend(tmp_path, free_mib = 12000)
        captured = _placement._launch(
            backend, gguf, n_ctx = NATIVE, cache_type_kv = "q8_0", tensor_parallel = True
        )
        assert backend.max_context_length == _ctx_of(captured["cmd"])

    def test_an_unquantized_cache_is_unaffected(self, tmp_path):
        """The V axis only moves for a quantized cache on this shape, so resolving the
        state cannot have changed what an f16 load plans. Asserted on the arithmetic at
        the context this load actually chose, rather than against a second hand-built
        plan: the loader charges overheads (the CUDA context reserve, the flat MTP
        cushion, the compute buffers, the VRAM fraction) that a bare planner call does
        not, and comparing the two would fail on those instead."""
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
        """The resolution is not "always on": a user who turns flash attention off in the
        extra arguments must be priced for the launch they will get."""
        seen = self._seen_flash_attn(
            tmp_path,
            n_ctx = NATIVE,
            cache_type_kv = "f16",
            extra_args = ["--flash-attn", "off"],
        )
        assert all(state is False for state in seen)


def test_the_state_is_still_named_planned_flash_attn():
    """The offload-planner seam asserts on this identifier by AST. Keep the name; only
    what it is assigned changed."""
    import inspect

    source = inspect.getsource(inspect.unwrap(LlamaCppBackend.load_model))
    assert "planned_flash_attn = _planned_flash_attn_state(" in source
    assert "planned_flash_attn = False" not in source
