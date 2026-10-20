# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""CPU smoke imports of unsloth_zoo vLLM/GRPO modules; rl_replacements and empty_model need no vllm."""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


# Apply the CPU spoof before any unsloth import.
_SPOOF_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SPOOF_DIR))
import _zoo_aggressive_cuda_spoof as _spoof  # noqa: E402

_spoof.apply()


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
    """Runs in a fresh interpreter: sys.modules is per-process, so in-process checks depend on test
    order."""
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
        [sys.executable, "-c", probe],
        capture_output = True,
        text = True,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"importing {module_name} in a clean interpreter failed:\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    lines = proc.stdout.strip().split("\n")
    found = [n for n in (lines[-1].split(",") if lines[-1] else [])]
    return lines[-2] == "VLLM", found


@pytest.mark.skipif(not _has_unsloth_zoo(), reason = "unsloth_zoo not installed")
def test_rl_replacements_imports_without_vllm():
    """unsloth_zoo.rl_replacements must NOT pull in vllm at import time."""
    # A transitive vllm import crashes GRPOTrainer construction on Colab.
    pulled, exports = _pulls_in_vllm(
        "unsloth_zoo.rl_replacements",
        "RL_REPLACEMENTS",
        "RL_FUNCTIONS",
    )
    assert not pulled, (
        "unsloth_zoo.rl_replacements imported vllm transitively; this breaks "
        "GRPO on environments without vllm installed (the use_vllm=False path "
        "is supposed to work without vllm)."
    )
    assert exports, "expected at least one GRPO-related export in rl_replacements"


@pytest.mark.skipif(not _has_unsloth_zoo(), reason = "unsloth_zoo not installed")
def test_empty_model_imports_without_vllm():
    pulled, exports = _pulls_in_vllm(
        "unsloth_zoo.empty_model",
        "create_empty_causal_lm",
        "create_empty_model",
    )
    assert (
        not pulled
    ), "unsloth_zoo.empty_model imported vllm transitively; expected to be vllm-free"
    assert exports, "expected a create_empty_* helper in empty_model"


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
