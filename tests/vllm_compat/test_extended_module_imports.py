# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Extended import-smoke + API surface checks for unsloth + unsloth-zoo
modules under the CUDA spoof harness.

Walks the full set of modules the public surface depends on (vs the 5 in
test_unsloth_zoo_imports.py), catching import-time symbol drift, spoof-
flipped gates (e.g. _IS_MLX on a non-Mac box), and FastModel API drift.
CPU-only; inherits the _zoo_aggressive_cuda_spoof harness.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path

import pytest


# Apply the spoof BEFORE any unsloth-touching import.
_SPOOF_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SPOOF_DIR))
import _zoo_aggressive_cuda_spoof as _spoof  # noqa: E402

_spoof.apply()


# Stub optional deps absent on a CPU-only runner (mirrors test_unsloth_zoo_imports.py).
def _stub_module(name: str, attrs: dict | None = None) -> None:
    """Stub a missing optional dep, with __spec__ set so find_spec() doesn't
    raise `ValueError: __spec__ is None` for torch/transformers/torchcodec callers."""
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(
        name = name, loader = None, origin = "<test stub>"
    )
    for k, v in (attrs or {}).items():
        setattr(m, k, v)
    sys.modules[name] = m


_stub_module(
    "pynvml",
    {
        "nvmlInit": lambda: None,
        "nvmlShutdown": lambda: None,
        "nvmlDeviceGetCount": lambda: 1,
        "nvmlDeviceGetHandleByIndex": lambda i: object(),
        "nvmlDeviceGetMemoryInfo": lambda h: type(
            "_M",
            (),
            {"total": 80 * 1024**3, "free": 70 * 1024**3, "used": 10 * 1024**3},
        )(),
    },
)
_stub_module("torchcodec")


@pytest.fixture(autouse = True)
def _torch_distributed_safe(monkeypatch):
    """unsloth_zoo modules occasionally probe torch.distributed."""
    try:
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_available", lambda: True, raising = False)
        monkeypatch.setattr(dist, "is_initialized", lambda: False, raising = False)
        monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 1, raising = False)
        monkeypatch.setattr(dist, "get_rank", lambda *a, **k: 0, raising = False)
    except Exception:
        pass


def _has_unsloth_zoo() -> bool:
    return importlib.util.find_spec("unsloth_zoo") is not None


def _has_unsloth() -> bool:
    return importlib.util.find_spec("unsloth") is not None


# unsloth-zoo modules with no top-level vllm/CUDA import: must load cleanly under spoof.


_ZOO_VLLM_FREE_MODULES = [
    "unsloth_zoo.compiler",
    "unsloth_zoo.compiler_replacements",
    "unsloth_zoo.dataset_utils",
    "unsloth_zoo.device_type",
    "unsloth_zoo.empty_model",
    "unsloth_zoo.gradient_checkpointing",
    "unsloth_zoo.hf_utils",
    "unsloth_zoo.llama_cpp",
    "unsloth_zoo.logging_utils",
    "unsloth_zoo.loss_utils",
    "unsloth_zoo.patching_utils",
    "unsloth_zoo.patch_torch_functions",
    "unsloth_zoo.peft_utils",
    "unsloth_zoo.rl_replacements",
    "unsloth_zoo.saving_utils",
    "unsloth_zoo.tiled_mlp",
    "unsloth_zoo.tokenizer_utils",
    "unsloth_zoo.training_utils",
    "unsloth_zoo.utils",
    "unsloth_zoo.vision_utils",
]


@pytest.mark.skipif(not _has_unsloth_zoo(), reason = "unsloth_zoo not installed")
@pytest.mark.parametrize("modname", _ZOO_VLLM_FREE_MODULES)
def test_unsloth_zoo_module_imports_under_spoof(modname: str):
    """Each unsloth_zoo module imports cleanly under spoof (catches import-time symbol drift)."""
    # Drop stale partial-import state from a prior failure.
    sys.modules.pop(modname, None)
    try:
        importlib.import_module(modname)
    except Exception as e:
        pytest.fail(
            f"{modname} failed to import under CUDA spoof: "
            f"{type(e).__name__}: {str(e)[:300]}"
        )


# Spoof correctness: _IS_MLX stays False on a non-Mac runner.


@pytest.mark.skipif(not _has_unsloth(), reason = "unsloth not installed")
def test_unsloth_is_mlx_false_under_spoof():
    """CUDA spoof must not flip _IS_MLX on non-Apple-Silicon hosts."""
    sys.modules.pop("unsloth", None)
    import unsloth

    assert unsloth._IS_MLX is False, (
        f"_IS_MLX activated on a non-Apple-Silicon runner under CUDA spoof; "
        f"the MLX gate logic in unsloth/__init__.py is too lax"
    )


# unsloth.models.* — core surfaces loaded transitively by `from unsloth import FastLanguageModel`.


_UNSLOTH_CORE_MODULES = [
    "unsloth.models.rl",
    "unsloth.models.rl_replacements",
    "unsloth.models.sentence_transformer",
    "unsloth.models._utils",
    "unsloth.models.loader",
    "unsloth.models.loader_utils",
    "unsloth.models.mapper",
]


@pytest.mark.skipif(not _has_unsloth(), reason = "unsloth not installed")
@pytest.mark.parametrize("modname", _UNSLOTH_CORE_MODULES)
def test_unsloth_core_module_imports_under_spoof(modname: str):
    """Core unsloth modules must import under spoof (module-top symbol drift
    crashes here). Bootstraps `import unsloth` first for its _gpu_init side effects."""
    try:
        import unsloth  # noqa: F401  -- triggers _gpu_init side effects
    except Exception as e:
        pytest.skip(f"`import unsloth` failed under spoof: {e}")
    sys.modules.pop(modname, None)
    try:
        importlib.import_module(modname)
    except OSError as e:
        # "could not get source code": editable-install + frozen sub-import
        # quirk, not symbol drift. Skip rather than false-fail.
        pytest.skip(f"{modname} env issue: {e!s}")
    except Exception as e:
        pytest.fail(
            f"{modname} failed to import under CUDA spoof: "
            f"{type(e).__name__}: {str(e)[:300]}"
        )


# FastLanguageModel/FastVisionModel/FastModel surface must be non-empty
# with the notebook-relied methods present.


@pytest.mark.skipif(not _has_unsloth(), reason = "unsloth not installed")
def test_fast_model_class_surface_under_spoof():
    sys.modules.pop("unsloth", None)
    import unsloth

    found_at_least_one = False
    for cls_name in ("FastLanguageModel", "FastVisionModel", "FastModel"):
        cls = getattr(unsloth, cls_name, None)
        if cls is None:
            continue
        found_at_least_one = True
        public = sorted(n for n in dir(cls) if not n.startswith("_"))
        # Notebooks rely on these methods.
        for method in ("from_pretrained", "get_peft_model"):
            assert method in public, (
                f"unsloth.{cls_name}.{method} missing under spoof; "
                f"every Colab notebook calling it breaks"
            )
    assert found_at_least_one, (
        f"none of FastLanguageModel/FastVisionModel/FastModel reachable "
        f"on `unsloth` package root"
    )


# RL surface: GRPO/SFT/DPO dispatch table must be populated, not silently
# empty while rl_replacements imports cleanly.


@pytest.mark.skipif(not _has_unsloth(), reason = "unsloth not installed")
def test_unsloth_rl_replacements_dispatch_populated():
    try:
        import unsloth  # noqa: F401  -- _gpu_init bootstrap
    except Exception as e:
        pytest.skip(f"`import unsloth` failed under spoof: {e}")
    sys.modules.pop("unsloth.models.rl_replacements", None)
    try:
        rl = importlib.import_module("unsloth.models.rl_replacements")
    except OSError as e:
        pytest.skip(f"env issue importing rl_replacements: {e!s}")
    funcs = getattr(rl, "RL_FUNCTIONS", None)
    if funcs is None:
        pytest.skip("RL_FUNCTIONS attribute not present (architecture changed; check)")
    assert isinstance(
        funcs, dict
    ), f"RL_FUNCTIONS expected dict, got {type(funcs).__name__}"
    # Trainer types unsloth-zoo dispatches against must be keys.
    for key in ("grpo_trainer", "sft_trainer", "dpo_trainer"):
        assert key in funcs, (
            f"RL_FUNCTIONS missing dispatch key '{key}'; "
            f"unsloth_zoo source rewrites silently no-op"
        )
        assert (
            isinstance(funcs[key], list) and len(funcs[key]) > 0
        ), f"RL_FUNCTIONS[{key!r}] is empty list; rewrites no-op"


# unsloth-zoo compiler test_apply_fused_lm_head (compiler.py:1983) must be callable.


@pytest.mark.skipif(not _has_unsloth_zoo(), reason = "unsloth_zoo not installed")
def test_zoo_compiler_apply_fused_lm_head_callable():
    sys.modules.pop("unsloth_zoo.compiler", None)
    compiler = importlib.import_module("unsloth_zoo.compiler")
    fn = getattr(compiler, "test_apply_fused_lm_head", None)
    assert fn is not None and callable(fn), (
        f"unsloth_zoo.compiler.test_apply_fused_lm_head missing or non-callable; "
        f"the in-file CPU regression test is the only fused-LM-head coverage"
    )


# FastModel.from_pretrained kwarg stability: removal becomes silent positional drift.


@pytest.mark.skipif(not _has_unsloth(), reason = "unsloth not installed")
def test_fast_model_from_pretrained_kwargs_under_spoof():
    sys.modules.pop("unsloth", None)
    import unsloth

    cls = getattr(unsloth, "FastLanguageModel", None) or getattr(
        unsloth, "FastModel", None
    )
    if cls is None:
        pytest.skip("FastLanguageModel/FastModel not exported")
    fn = getattr(cls, "from_pretrained", None)
    if fn is None:
        pytest.skip("from_pretrained not on class (might be classmethod stub)")
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        pytest.skip("from_pretrained signature not introspectable")
    # Notebooks use these kwargs by name everywhere.
    for kwarg in ("model_name", "max_seq_length", "load_in_4bit"):
        assert kwarg in params, (
            f"FastLanguageModel.from_pretrained missing kwarg `{kwarg}`; "
            f"every Colab notebook breaks at the install cell"
        )
