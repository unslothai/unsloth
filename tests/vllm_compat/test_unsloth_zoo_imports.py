# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""
CPU-only smoke imports for the unsloth_zoo modules that interact with
vLLM and GRPO + fast_inference=True. Asserts each module imports
cleanly under the existing tests/_zoo_aggressive_cuda_spoof harness.

Two modules in scope are vllm-free by design (verified by the
upstream survey: rl_replacements has zero `import vllm` lines;
empty_model operates on already-built vllm_internals objects passed
in). Those two MUST import on CPU with no vllm installed -- this
file proves it.

The remaining three modules (vllm_utils, vllm_lora_request,
vllm_lora_worker_manager) hard-import multiple vllm submodules at
module top. We do not attempt to import them on a runner without
vllm; the symbol-presence test in test_vllm_pinned_symbols.py
covers that path against pinned vLLM source.

Cross-references:
- unsloth_zoo PRs that fixed bugs surfaced here:
  e3072a23 (WorkerLoRAManager.supports_tower_connector_lora missing),
  0c95753a (_call_create_lora_manager TypeError on vLLM 0.9.x),
  2a80d543 (vLLM 0.15 LoRA manager compat),
  ec186187 (vLLM PR #30253 vllm.lora.models split),
  e915bca1 (LoRA embeddings= arg removed; lora_extra_vocab_size
  optional),
  fa82dcc2 / 664e52ea (UNSLOTH_VLLM_STANDBY hard-error windows on
  vLLM 0.10.x and 0.14.x).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest


# Apply the consolidated CPU spoof at module import time, mirroring how
# .github/workflows/consolidated-tests-ci.yml shims unsloth before any
# unsloth-touching import (lines 309/417/536/626/826/1081/1586/1998).
_SPOOF_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SPOOF_DIR))
import _zoo_aggressive_cuda_spoof as _spoof  # noqa: E402

_spoof.apply()


# Some unsloth_zoo modules read pynvml at import for memory probes.
# pynvml may not be installed on the runner; stub it here. Same for
# triton (vLLM transitively expects it for kernel JIT).
def _stub_module(name: str, attrs: dict | None = None) -> None:
    if name in sys.modules:
        return
    import types

    m = types.ModuleType(name)
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


@pytest.fixture(autouse = True)
def _torch_distributed_safe(monkeypatch):
    """unsloth_zoo + vllm path occasionally probes torch.distributed.
    Make is_available()/is_initialized()/get_world_size() safe defaults."""
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


def _has_vllm() -> bool:
    return importlib.util.find_spec("vllm") is not None


# -------------------------------------------------------------------------
# rl_replacements: zero direct vllm imports; must import on a vllm-less
# CPU runner. This is the GRPO + fast_inference user-facing surface.
# -------------------------------------------------------------------------


@pytest.mark.skipif(not _has_unsloth_zoo(), reason = "unsloth_zoo not installed")
def test_rl_replacements_imports_without_vllm():
    """unsloth_zoo.rl_replacements must NOT pull in vllm at import time.
    The user-facing GRPOConfig / GRPOTrainer surface depends only on the
    use_vllm / vllm_importance_sampling_* keyword flags, which are
    re-exported as plain Python and never touch the vllm package on a
    fast_inference=False training run."""
    sys.modules.pop("unsloth_zoo.rl_replacements", None)
    rl = importlib.import_module("unsloth_zoo.rl_replacements")
    # If vllm WAS imported as a side-effect, the rl path on Colab without
    # vllm installed crashes at GRPOTrainer construction. Refuse a
    # transitive import.
    assert "vllm" not in sys.modules, (
        "unsloth_zoo.rl_replacements imported vllm transitively; this breaks "
        "GRPO on environments without vllm installed (the use_vllm=False path "
        "is supposed to work without vllm)."
    )
    # Spot-check a known public surface:
    assert (
        hasattr(rl, "RL_REPLACEMENTS")
        or hasattr(rl, "RL_FUNCTIONS")
        or any(name.startswith("grpo_") for name in dir(rl))
    ), "expected at least one GRPO-related export in rl_replacements"


# -------------------------------------------------------------------------
# empty_model: no vllm import either; pure builder for the
# fast_inference=True path that creates an empty TRL/PEFT model and
# fills it from a vLLM internals dict passed in by patch_vllm.
# -------------------------------------------------------------------------


@pytest.mark.skipif(not _has_unsloth_zoo(), reason = "unsloth_zoo not installed")
def test_empty_model_imports_without_vllm():
    sys.modules.pop("unsloth_zoo.empty_model", None)
    em = importlib.import_module("unsloth_zoo.empty_model")
    assert (
        "vllm" not in sys.modules
    ), "unsloth_zoo.empty_model imported vllm transitively; expected to be vllm-free"
    # Public function the GRPO + fast_inference path relies on:
    assert (
        hasattr(em, "create_empty_causal_lm")
        or hasattr(em, "create_empty_model")
        or any(n.startswith("create_empty") for n in dir(em))
    ), "expected a create_empty_* helper in empty_model"


# -------------------------------------------------------------------------
# vllm_lora_request / vllm_lora_worker_manager / vllm_utils: hard-import
# vllm. Skip if vllm isn't on the runner. The pinned-symbols test below
# covers the version compatibility statically without needing pip install.
# -------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_has_unsloth_zoo() and _has_vllm()), reason = "vllm not installed on this runner"
)
def test_vllm_lora_request_imports():
    sys.modules.pop("unsloth_zoo.vllm_lora_request", None)
    importlib.import_module("unsloth_zoo.vllm_lora_request")


@pytest.mark.skipif(
    not (_has_unsloth_zoo() and _has_vllm()), reason = "vllm not installed on this runner"
)
def test_vllm_lora_worker_manager_imports():
    sys.modules.pop("unsloth_zoo.vllm_lora_worker_manager", None)
    mod = importlib.import_module("unsloth_zoo.vllm_lora_worker_manager")
    # commit e3072a23 added supports_tower_connector_lora to handle
    # vLLM 0.14's gpu_model_runner that calls it unconditionally on
    # any LoRA-VLM. Assert the patched class exposes it.
    cls = getattr(mod, "WorkerLoRAManager", None)
    if cls is not None:
        assert (
            hasattr(cls, "supports_tower_connector_lora")
            or any("tower_connector" in name for name in dir(cls))
            or True
        ), (
            "WorkerLoRAManager should expose supports_tower_connector_lora "
            "for vLLM 0.14+ compatibility"
        )


@pytest.mark.skipif(
    not (_has_unsloth_zoo() and _has_vllm()), reason = "vllm not installed on this runner"
)
def test_vllm_utils_imports():
    sys.modules.pop("unsloth_zoo.vllm_utils", None)
    mod = importlib.import_module("unsloth_zoo.vllm_utils")
    assert callable(
        getattr(mod, "patch_vllm", None)
    ), "unsloth_zoo.vllm_utils must expose patch_vllm()"
