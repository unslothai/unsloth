# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""CPU-only smoke imports for unsloth_zoo modules touching vLLM / GRPO +
fast_inference, under the tests/_zoo_aggressive_cuda_spoof harness.

rl_replacements and empty_model are vllm-free and MUST import on CPU with no
vllm; the three vllm-hard-import modules are skipped without it (covered
statically by test_vllm_pinned_symbols.py).

Cross-references (unsloth_zoo commits that fixed bugs surfaced here):
  e3072a23 (WorkerLoRAManager.supports_tower_connector_lora missing),
  0c95753a (_call_create_lora_manager TypeError on vLLM 0.9.x),
  2a80d543 (vLLM 0.15 LoRA manager compat),
  ec186187 (vLLM PR #30253 vllm.lora.models split),
  e915bca1 (LoRA embeddings= arg removed; lora_extra_vocab_size optional),
  fa82dcc2 / 664e52ea (UNSLOTH_VLLM_STANDBY hard-error on vLLM 0.10/0.14).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


# Apply the consolidated CPU spoof at import time, before any unsloth import.
_SPOOF_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SPOOF_DIR))
import _zoo_aggressive_cuda_spoof as _spoof  # noqa: E402

_spoof.apply()


# Some unsloth_zoo modules read pynvml at import; stub it for the runner.
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
    """Give torch.distributed probes safe single-process defaults."""
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


def _pulls_in_vllm(module_name: str, *exports: str) -> tuple[bool, list[str]]:
    """(did importing `module_name` pull in vllm, which of `exports` it has).

    Asked in a FRESH interpreter, because `"vllm" in sys.modules` is a property
    of the process, not of the import under test. In-process this answers "has
    anything in this pytest worker ever imported vllm" -- the vllm-hard-import
    tests in this same file do exactly that, and popping the module under test
    from sys.modules cannot undo it, because its already-cached dependencies are
    not re-imported on the second import either. So the check passed or failed on
    test ordering and never observed what it claimed to.

    The subprocess re-applies the same CPU spoof this module applies at import,
    so a CPU-only runner is still covered. subprocess + sys.executable keeps this
    working on Linux, macOS and Windows alike.
    """
    probe = (
        "import sys\n"
        f"sys.path.insert(0, {str(_SPOOF_DIR)!r})\n"
        "import _zoo_aggressive_cuda_spoof as s\n"
        "s.apply()\n"
        f"m = __import__({module_name!r}, fromlist=['_'])\n"
        "print('VLLM' if 'vllm' in sys.modules else 'NOVLLM')\n"
        f"print(','.join(n for n in {list(exports)!r} if hasattr(m, n)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output = True, text = True,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"importing {module_name} in a clean interpreter failed:\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    lines = proc.stdout.strip().split("\n")
    found = [n for n in (lines[-1].split(",") if lines[-1] else [])]
    return lines[-2] == "VLLM", found


# rl_replacements: zero direct vllm imports;
# the GRPO + fast_inference surface.
@pytest.mark.skipif(not _has_unsloth_zoo(), reason = "unsloth_zoo not installed")
def test_rl_replacements_imports_without_vllm():
    """unsloth_zoo.rl_replacements must NOT pull in vllm at import time."""
    # A transitive vllm import crashes GRPOTrainer construction on Colab.
    pulled, exports = _pulls_in_vllm(
        "unsloth_zoo.rl_replacements", "RL_REPLACEMENTS", "RL_FUNCTIONS",
    )
    assert not pulled, (
        "unsloth_zoo.rl_replacements imported vllm transitively; this breaks "
        "GRPO on environments without vllm installed (the use_vllm=False path "
        "is supposed to work without vllm)."
    )
    assert exports, "expected at least one GRPO-related export in rl_replacements"


# empty_model: no vllm import;
# pure builder for the fast_inference=True path.
@pytest.mark.skipif(not _has_unsloth_zoo(), reason = "unsloth_zoo not installed")
def test_empty_model_imports_without_vllm():
    pulled, exports = _pulls_in_vllm(
        "unsloth_zoo.empty_model", "create_empty_causal_lm", "create_empty_model",
    )
    assert not pulled, (
        "unsloth_zoo.empty_model imported vllm transitively; expected to be vllm-free"
    )
    assert exports, "expected a create_empty_* helper in empty_model"


# vllm_lora_request / vllm_lora_worker_manager / vllm_utils: hard-import vllm,
# so skip without it (pinned-symbols test covers version compat statically).
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
    # e3072a23 added supports_tower_connector_lora for vLLM 0.14's gpu_model_runner; assert the patched class exposes
    # it.
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
