# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Smoke tests for the deferred FlexEngine batch-size sizing.

Unit-level: covers the four cases described in the implementation plan
by monkey-patching :class:`FlexEngine` with a cheap stand-in so the
tests run on any box (no CUDA / no model download). The dispatch logic
lives entirely in :func:`build_flex_engine`,
:func:`install_flex_sentinel`, and :func:`_build_flex_from_args`, which
are the units under test.

Run as:
    python tests/flex_lazy_batch_smoke.py
"""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class _StubFlexEngine:
    """Records the ``max_batch_size`` construction arg and nothing else.

    ``_cudagraph_primed`` flips True after the first ``generate`` so the
    post-warmup refuse path can be exercised without touching CUDA.
    """

    instances: list = []

    def __init__(
        self,
        hf_model,
        tokenizer,
        *,
        dtype = None,
        max_seq_length: int = 2048,
        max_lora_rank: int = 64,
        max_batch_size: int = 32,
        page_size: int = 128,
        gpu_memory_utilization: float = 0.5,
        max_new_tokens: int = 512,
        prefill_kernel_options = None,
        decode_kernel_options = None,
        fa4_prefill = None,
        capture_cudagraph: bool = True,
        base_model = None,
        peft_model = None,
        inference_model = None,
    ):
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.compute_dtype = dtype
        self._cudagraph_primed = False
        self.generate_calls = 0
        _StubFlexEngine.instances.append(self)

    def generate(self, prompts = None, *args, **kwargs):
        self.generate_calls += 1
        self._cudagraph_primed = True
        return [("stub", prompts)]


def _make_stub_model():
    """An object that quacks like an HF model for ``install_flex_sentinel``."""

    model = types.SimpleNamespace()
    model._unsloth_needs_flex_engine = dict(
        dtype = "bf16",
        max_seq_length = 2048,
        max_lora_rank = 64,
        max_batch_size = 32,
        gpu_memory_utilization = 0.5,
    )
    model._unsloth_flex_inference_copy = object()  # never dereferenced
    return model


def _install_stub():
    """Patch FlexEngine with :class:`_StubFlexEngine` for the duration of the
    test process. Imports happen lazily inside ``build_flex_engine``, so we
    patch the module attribute before those calls fire."""

    import unsloth.inference.flex_engine as fe

    _StubFlexEngine.instances.clear()
    fe.FlexEngine = _StubFlexEngine


def _case1_default_floor():
    """No trainer, no kwargs: fast_generate builds at floor=32."""

    from unsloth.inference.flex_engine import install_flex_sentinel

    _install_stub()
    model = _make_stub_model()
    install_flex_sentinel(model, tokenizer = object())

    assert hasattr(model, "vllm_engine"), "sentinel not installed"
    assert not hasattr(model, "_flex_engine_instance"), (
        "engine should NOT exist before first use"
    )

    out = model.fast_generate(["hello"])
    assert out == [("stub", ["hello"])]

    engine = model._flex_engine_instance
    assert engine.max_batch_size == 32, engine.max_batch_size
    # Sentinel was replaced with the real engine after build.
    assert model.vllm_engine is engine
    print("  [1/4] default path: floor=32 build on first fast_generate OK")


def _case2_grpo_bump():
    """User kwarg=16 + GRPO target=64 → engine built at 64, warning logged."""

    from unsloth.inference.flex_engine import (
        _build_flex_from_args,
        install_flex_sentinel,
    )

    _install_stub()
    model = _make_stub_model()
    model._unsloth_needs_flex_engine["max_batch_size"] = 16  # user floor
    install_flex_sentinel(model, tokenizer = object())

    args = types.SimpleNamespace(
        per_device_train_batch_size = 2,
        steps_per_generation = 4,
        num_generations = 8,
        gradient_accumulation_steps = 1,
    )
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _build_flex_from_args(model, args)

    engine = model._flex_engine_instance
    assert engine.max_batch_size == 64, engine.max_batch_size
    assert any("16 -> 64" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]
    print("  [2/4] GRPO bump: 16 -> 64 with warning OK")


def _case3_user_floor_wins():
    """User kwarg=128 + GRPO target=8 → engine stays at 128, no warning."""

    from unsloth.inference.flex_engine import (
        _build_flex_from_args,
        install_flex_sentinel,
    )

    _install_stub()
    model = _make_stub_model()
    model._unsloth_needs_flex_engine["max_batch_size"] = 128
    install_flex_sentinel(model, tokenizer = object())

    args = types.SimpleNamespace(
        per_device_train_batch_size = 1,
        steps_per_generation = 2,
        num_generations = 4,
        gradient_accumulation_steps = 1,
    )
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _build_flex_from_args(model, args)

    engine = model._flex_engine_instance
    assert engine.max_batch_size == 128, engine.max_batch_size
    assert not any("FlexEngine" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]
    print("  [3/4] user floor wins: engine.max_batch_size=128 OK")


def _case4_post_warmup_refused():
    """fast_generate primes the engine; later GRPO target=64 must raise."""

    from unsloth.inference.flex_engine import (
        _build_flex_from_args,
        install_flex_sentinel,
    )

    _install_stub()
    model = _make_stub_model()
    install_flex_sentinel(model, tokenizer = object())

    model.fast_generate(["hi"])  # builds at floor=32, sets _cudagraph_primed
    assert model._flex_engine_instance.max_batch_size == 32

    args = types.SimpleNamespace(
        per_device_train_batch_size = 2,
        steps_per_generation = 4,
        num_generations = 8,
        gradient_accumulation_steps = 1,
    )
    try:
        _build_flex_from_args(model, args)
    except RuntimeError as exc:
        msg = str(exc)
        assert "32" in msg and "64" in msg, msg
        assert "max_batch_size=64" in msg, msg
        print("  [4/4] post-warmup rebuild refused with actionable msg OK")
        return
    raise AssertionError("expected RuntimeError when growing a built engine")


def main():
    print("flex_lazy_batch_smoke:")
    _case1_default_floor()
    _case2_grpo_bump()
    _case3_user_floor_wins()
    _case4_post_warmup_refused()
    print("ALL CASES PASSED")


if __name__ == "__main__":
    main()
