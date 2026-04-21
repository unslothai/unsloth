# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""Live smoke-test for deferred FlexEngine construction.

Loads a small flex-supported model with ``UNSLOTH_FAST_INFERENCE=1`` and
checks:

  1. After ``from_pretrained``, ``model.vllm_engine`` is the lazy
     sentinel -- no ``_flex_engine_instance`` yet.
  2. ``build_flex_engine(model)`` constructs the engine at the stashed
     ``max_batch_size`` floor (default 32) and wires
     ``model.vllm_engine`` / ``fast_generate`` onto it.
  3. ``build_flex_engine(model, max_batch_size=X)`` with ``X`` larger
     than the built size raises :class:`RuntimeError` with an actionable
     hint pointing the user back to ``max_batch_size=`` in
     ``from_pretrained``.

This intentionally does NOT call ``engine.generate`` -- that path is
covered by the existing ``tests/flex_fastlm_smoke.py`` and would
duplicate its warm-up cost. The goal here is to confirm the lazy
dispatch behavior end-to-end.

Run as:
    CUDA_VISIBLE_DEVICES=0 UNSLOTH_FAST_INFERENCE=1 \
    python tests/flex_lazy_live_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main():
    assert os.environ.get("UNSLOTH_FAST_INFERENCE", "0") == "1", (
        "export UNSLOTH_FAST_INFERENCE=1 before running this smoke"
    )
    import unsloth  # noqa: F401  (must import before transformers)
    from unsloth import FastLanguageModel
    from unsloth.inference.flex_engine import (
        FlexEngine,
        _LazyFlexEngineSentinel,
        build_flex_engine,
    )

    model_name = os.environ.get(
        "FLEX_LAZY_SMOKE_MODEL", "unsloth/Qwen3-0.6B-Base"
    )
    print(f"loading {model_name} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 1024,
        fast_inference = True,
        load_in_4bit = False,
    )

    assert hasattr(model, "vllm_engine"), "vllm_engine attr missing"
    sentinel = model.vllm_engine
    assert isinstance(sentinel, _LazyFlexEngineSentinel), (
        f"expected sentinel, got {type(sentinel)}"
    )
    assert not hasattr(model, "_flex_engine_instance"), (
        "engine should NOT be built before first use"
    )
    print("  [1/3] sentinel installed, no engine yet")

    engine = build_flex_engine(model)
    assert isinstance(engine, FlexEngine), type(engine)
    assert engine.max_batch_size == 32, engine.max_batch_size
    assert model._flex_engine_instance is engine
    assert model.vllm_engine is engine
    print(f"  [2/3] build_flex_engine built engine at max_batch_size={engine.max_batch_size}")

    try:
        build_flex_engine(model, max_batch_size = 64)
    except RuntimeError as exc:
        msg = str(exc)
        assert "32" in msg and "64" in msg, msg
        assert "max_batch_size=64" in msg, msg
        print("  [3/3] post-build resize refused with actionable msg")
    else:
        raise AssertionError("expected RuntimeError when growing a built engine")

    print("LIVE SMOKE PASSED")


if __name__ == "__main__":
    main()
