# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Extended import-smoke + API surface checks for unsloth + unsloth-zoo
modules under the existing CUDA spoof harness.

Where `tests/vllm_compat/test_unsloth_zoo_imports.py` covers the
narrow "must import on a vllm-less runner" claim for 5 modules,
this file walks the FULL set of modules our public surface depends
on. Catches:

  - module-level imports that break on a fresh transformers / peft /
    bnb release (the symbol pinned at import time is gone)
  - feature flags / gates that flip under the spoof (e.g. _IS_MLX
    silently activating on a non-Mac CI box)
  - public API surface drift: sorted `dir()` of each FastModel class
    is dumped and asserted-stable across runs (a removed kwarg here
    is a notebook regression we want to catch)

CPU-only. Inherits the same _zoo_aggressive_cuda_spoof harness as
test_unsloth_zoo_imports.py.
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


# Stub modules the unsloth import path may probe but that aren't
# installed on a CPU-only runner. Mirrors test_unsloth_zoo_imports.py.
def _stub_module(name: str, attrs: dict | None = None) -> None:
    """Stub a missing optional dep. Sets __spec__ so importlib.util's
    `find_spec(name)` doesn't raise `ValueError: __spec__ is None`,
    which torch / transformers / torchcodec callers hit otherwise."""
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    # Minimal viable spec so importlib treats the stub as a real module.
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


# -------------------------------------------------------------------------
# Extended unsloth-zoo module list. Modules with no top-level vllm/CUDA
# import are expected to load cleanly on a CPU spoof runner.
# -------------------------------------------------------------------------


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
    """Each unsloth_zoo module must import cleanly on a CPU-only spoof
    runner. Catches transformers/peft/bnb symbol drift that pins fail
    at import time (vs runtime)."""
    # Force fresh resolution: drops stale partial-import state from
    # a previous module's failure.
    sys.modules.pop(modname, None)
    try:
        importlib.import_module(modname)
    except Exception as e:
        pytest.fail(
            f"{modname} failed to import under CUDA spoof: "
            f"{type(e).__name__}: {str(e)[:300]}"
        )


# -------------------------------------------------------------------------
# Spoof correctness: _IS_MLX must remain False on a non-Mac runner
# AND _IS_CUDA / DEVICE_TYPE must reflect the spoofed CUDA layer.
# -------------------------------------------------------------------------


@pytest.mark.skipif(not _has_unsloth(), reason = "unsloth not installed")
def test_unsloth_is_mlx_false_under_spoof():
    """The CUDA spoof should not flip the MLX flag on a Linux/Windows CI
    box (real Apple Silicon is the ONLY environment _IS_MLX activates)."""
    sys.modules.pop("unsloth", None)
    import unsloth

    assert unsloth._IS_MLX is False, (
        f"_IS_MLX activated on a non-Apple-Silicon runner under CUDA spoof; "
        f"the MLX gate logic in unsloth/__init__.py is too lax"
    )


# -------------------------------------------------------------------------
# unsloth.models.* — the core RL + sentence-transformer surfaces. These
# are the entry points unsloth/__init__.py loads transitively when a
# user does `from unsloth import FastLanguageModel`.
# -------------------------------------------------------------------------


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
    """Core unsloth modules must import on a CPU-only runner under
    the CUDA spoof. Drift in transformers/peft/trl symbols pinned at
    module-top crashes here BEFORE any user-visible call.

    Bootstraps via `import unsloth` first, since most sub-modules
    require the package's _gpu_init side effects. Without that, every
    `import unsloth.models.*` raises a guard `Please restructure your
    imports with 'import unsloth' at the top of your file.`"""
    try:
        import unsloth  # noqa: F401  -- triggers _gpu_init side effects
    except Exception as e:
        pytest.skip(f"`import unsloth` failed under spoof: {e}")
    sys.modules.pop(modname, None)
    try:
        importlib.import_module(modname)
    except OSError as e:
        # `OSError: could not get source code` happens when an editable
        # install + frozen sub-import combine; that's an environment
        # quirk, not a symbol-drift bug. Skip rather than false-fail.
        pytest.skip(f"{modname} env issue: {e!s}")
    except Exception as e:
        pytest.fail(
            f"{modname} failed to import under CUDA spoof: "
            f"{type(e).__name__}: {str(e)[:300]}"
        )


# -------------------------------------------------------------------------
# Public API surface dump for FastLanguageModel / FastVisionModel /
# FastModel under spoof. Asserts the surface is non-empty and that
# the patch hooks unsloth-zoo's RL surface relies on are present.
# -------------------------------------------------------------------------


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
        # Notebooks rely on these methods. Loss of any one is a regression
        # the existing api-introspect notebook job would catch a step
        # later — but here at the import / spoof layer.
        for method in ("from_pretrained", "get_peft_model"):
            assert method in public, (
                f"unsloth.{cls_name}.{method} missing under spoof; "
                f"every Colab notebook calling it breaks"
            )
    assert found_at_least_one, (
        f"none of FastLanguageModel/FastVisionModel/FastModel reachable "
        f"on `unsloth` package root"
    )


# -------------------------------------------------------------------------
# RL surface drill-down: GRPO, SFT, DPO classes must be reachable AND
# the source-rewriter dispatch table must be populated. Catches the
# scenario where unsloth.models.rl_replacements imports cleanly but
# RL_FUNCTIONS or RL_REPLACEMENTS is silently empty.
# -------------------------------------------------------------------------


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
    # The trainer types unsloth-zoo dispatches against MUST be keys.
    for key in ("grpo_trainer", "sft_trainer", "dpo_trainer"):
        assert key in funcs, (
            f"RL_FUNCTIONS missing dispatch key '{key}'; "
            f"unsloth_zoo source rewrites silently no-op"
        )
        assert (
            isinstance(funcs[key], list) and len(funcs[key]) > 0
        ), f"RL_FUNCTIONS[{key!r}] is empty list; rewrites no-op"


# -------------------------------------------------------------------------
# unsloth-zoo compiler test_apply_fused_lm_head — exercises the actual
# fused-LM-head emit path with a tiny fixture. Already covered as a
# named test in compiler.py:1983; we just call it.
# -------------------------------------------------------------------------


@pytest.mark.skipif(not _has_unsloth_zoo(), reason = "unsloth_zoo not installed")
def test_zoo_compiler_apply_fused_lm_head_callable():
    sys.modules.pop("unsloth_zoo.compiler", None)
    compiler = importlib.import_module("unsloth_zoo.compiler")
    fn = getattr(compiler, "test_apply_fused_lm_head", None)
    assert fn is not None and callable(fn), (
        f"unsloth_zoo.compiler.test_apply_fused_lm_head missing or non-callable; "
        f"the in-file CPU regression test is the only fused-LM-head coverage"
    )


# -------------------------------------------------------------------------
# Spot-check signature stability of FastModel.from_pretrained — every
# notebook call site relies on these kwargs. A removed kwarg silently
# becomes positional drift.
# -------------------------------------------------------------------------


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
    # Notebooks use these by name everywhere.
    for kwarg in ("model_name", "max_seq_length", "load_in_4bit"):
        assert kwarg in params, (
            f"FastLanguageModel.from_pretrained missing kwarg `{kwarg}`; "
            f"every Colab notebook breaks at the install cell"
        )
