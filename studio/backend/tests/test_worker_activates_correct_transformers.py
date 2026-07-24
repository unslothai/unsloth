# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: after the training worker runs its preflight and then activates the transformers
sidecar, the in-process ``transformers`` must be the sidecar version the model requires -- not the
default 4.57.x that the base environment ships.

The CPU-only "does it choose the correct transformers version" guard, stronger than the pure
import-order check in ``test_training_worker_import_discipline.py``: it runs the REAL tier detection
(``get_transformers_tier``) and REAL activation (``activate_transformers_for_subprocess``) for a
transformers-5.x model (Qwen3.5, tier 530) and asserts the version actually switched. It catches the
whole failure family at once:

  * a stale pre-activation ``transformers`` import (the #6951 / ``TokenizersBackend`` regression: an
    already-cached 4.57.x defeats the sidecar's ``sys.path`` prepend),
  * a wrong tier selected for a 5.x model, and
  * activation not actually swapping the resident module.

Why the CUDA spoof matters (verified): ``unsloth_zoo``'s eager ``import transformers`` only happens on
its full, GPU-present init path. On a GPU-less runner it silently degrades and never preloads
transformers -- which would MASK the stale-import bug (the check would falsely pass). Spoofing
``torch.cuda`` so ``unsloth_zoo`` believes a GPU is present forces the real init path, exposing the
regression on CPU CI. The spoof mirrors ``tests/_zoo_aggressive_cuda_spoof.py`` but is inlined so the
test is self-contained in the ``studio-backend-ci`` matrix (whose conftest does not apply the shared
spoof). No GPU/network/weights/real sidecar needed: a one-line stub sidecar stands in for the 5.x venv,
so we only assert activation lands on it.

Proven: passes on the fixed tree (active == 5.3.0) and fails on the buggy tree (active == 4.57.x) on
a simulated GPU-less runner.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent  # studio/backend
# Canonical CUDA spoof at the repo root (studio/backend -> studio -> repo root). Loaded by the
# subprocess when present (matches the consolidated CI); absent in a standalone studio checkout, where
# the subprocess falls back to a minimal inline spoof.
_SPOOF_PATH = _BACKEND_DIR.parent.parent / "tests" / "_zoo_aggressive_cuda_spoof.py"

# Runs in a fresh interpreter with cwd == studio/backend so ``utils.*`` resolves like the worker.
# STUB_HOME (a pytest tmp dir) holds a throwaway ``.venv_t5_530`` sidecar exporting transformers 5.3.0.
_SNIPPET = r"""
import os, sys
sys.path.insert(0, os.getcwd())

# CUDA spoof so unsloth_zoo takes its full, transformers-importing init path on a GPU-less runner.
# Without it unsloth_zoo degrades and never preloads transformers, which would MASK the stale-import
# regression under test (verified). Prefer the repo's canonical spoof (single source of truth, and the
# one the consolidated CI already relies on); fall back to a minimal inline spoof so this also works in
# a standalone studio checkout. If torch is absent the fixed tree still passes below; the bug just
# would not be exposable in that shard.
try:
    import torch  # noqa: F401
    _sp = os.environ.get("SPOOF_PATH")
    if _sp and os.path.exists(_sp):
        import importlib.util
        _spec = importlib.util.spec_from_file_location("_zoo_aggressive_cuda_spoof", _sp)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.apply()
    else:
        torch.cuda.is_available = lambda: True
        torch.cuda.device_count = lambda: 1
        torch.cuda.current_device = lambda: 0
        torch.cuda.get_device_capability = lambda *a, **k: (8, 0)
        torch.cuda.get_device_name = lambda *a, **k: "NVIDIA A100-SPOOFED"
        torch.cuda.is_bf16_supported = lambda *a, **k: True
        class _Props:
            name = "NVIDIA A100-SPOOFED"
            major = 8
            minor = 0
            total_memory = 80 * 1024**3
            multi_processor_count = 108
        torch.cuda.get_device_properties = lambda *a, **k: _Props()
        torch.cuda.mem_get_info = lambda *a, **k: (0, 80 * 1024**3)
except Exception:
    pass
os.environ["UNSLOTH_IS_PRESENT"] = "1"

# Stub 5.x sidecar: activation only edits sys.path, so a package that merely exports __version__ is
# enough to prove the resident transformers switched to it.
home = os.environ["STUB_HOME"]
pkg = os.path.join(home, ".venv_t5_530", "transformers")
os.makedirs(pkg, exist_ok = True)
with open(os.path.join(pkg, "__init__.py"), "w") as f:
    f.write('__version__ = "5.3.0"\n')
os.environ["UNSLOTH_STUDIO_HOME"] = home

# Faithful worker preflight (worker.py: from utils.hf_xet_fallback import child_should_disable_xet).
# This is the exact stale-import trigger: on the buggy tree it pulls unsloth_zoo -> transformers 4.57.x
# into sys.modules BEFORE activation.
from utils.hf_xet_fallback import child_should_disable_xet
child_should_disable_xet({})
_tf = sys.modules.get("transformers")
preload = _tf.__version__ if _tf is not None else None

# Real tier detection + real activation, with the 530 sidecar pointed at the stub above.
import utils.transformers_version as tv
tv._VENV_T5_530_DIR = os.path.join(home, ".venv_t5_530")
tv._ensure_venv_t5_530_exists = lambda: True
tier = tv.get_transformers_tier("Qwen/Qwen3.5-9B", None)
tv.activate_transformers_for_subprocess("Qwen/Qwen3.5-9B", None)

import transformers
print(f"RESULT tier={tier} preload={preload} active={transformers.__version__}")
"""

# Astral (#7353): custom-code checkpoints must activate the 5.10 sidecar, not default 4.57.x.
# Uses a local stub checkpoint (no Hub fetch) with the same worker preflight + activation path.
_SNIPPET_ASTRAL_510 = r"""
import json, os, sys
sys.path.insert(0, os.getcwd())

try:
    import torch  # noqa: F401
    _sp = os.environ.get("SPOOF_PATH")
    if _sp and os.path.exists(_sp):
        import importlib.util
        _spec = importlib.util.spec_from_file_location("_zoo_aggressive_cuda_spoof", _sp)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.apply()
    else:
        torch.cuda.is_available = lambda: True
        torch.cuda.device_count = lambda: 1
        torch.cuda.current_device = lambda: 0
        torch.cuda.get_device_capability = lambda *a, **k: (8, 0)
        torch.cuda.get_device_name = lambda *a, **k: "NVIDIA A100-SPOOFED"
        torch.cuda.is_bf16_supported = lambda *a, **k: True
        class _Props:
            name = "NVIDIA A100-SPOOFED"
            major = 8
            minor = 0
            total_memory = 80 * 1024**3
            multi_processor_count = 108
        torch.cuda.get_device_properties = lambda *a, **k: _Props()
        torch.cuda.mem_get_info = lambda *a, **k: (0, 80 * 1024**3)
except Exception:
    pass
os.environ["UNSLOTH_IS_PRESENT"] = "1"

home = os.environ["STUB_HOME"]
model_dir = os.path.join(home, "astral-checkpoint")
os.makedirs(model_dir, exist_ok = True)
cfg = {
    "architectures": ["AstralForCausalLM"],
    "model_type": "astral",
    "auto_map": {"AutoModelForCausalLM": "modeling_astral.AstralForCausalLM"},
}
with open(os.path.join(model_dir, "config.json"), "w") as f:
    json.dump(cfg, f)
with open(os.path.join(model_dir, "modeling_astral.py"), "w") as f:
    f.write("from transformers.modeling_layers import GradientCheckpointingLayer\n")

pkg = os.path.join(home, ".venv_t5_510", "transformers")
os.makedirs(pkg, exist_ok = True)
with open(os.path.join(pkg, "__init__.py"), "w") as f:
    f.write('__version__ = "5.10.2"\n')
os.environ["UNSLOTH_STUDIO_HOME"] = home

from utils.hf_xet_fallback import child_should_disable_xet
child_should_disable_xet({})
_tf = sys.modules.get("transformers")
preload = _tf.__version__ if _tf is not None else None

import utils.transformers_version as tv
tv._VENV_T5_510_DIR = os.path.join(home, ".venv_t5_510")
tv._ensure_venv_t5_510_exists = lambda: True
tier = tv.get_transformers_tier(model_dir, None)
tv.activate_transformers_for_subprocess(model_dir, None)

import transformers
print(f"RESULT tier={tier} preload={preload} active={transformers.__version__}")
"""


def _parse(stdout: str) -> dict[str, str]:
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            return dict(kv.split("=", 1) for kv in line.split()[1:])
    return {}


def test_worker_activates_correct_transformers_version(tmp_path):
    """The worker's real preflight + activation for a transformers-5.x model (Qwen3.5, tier 530) must
    leave the in-process ``transformers`` on the 5.x sidecar. A stale pre-activation import leaves the
    default 4.57.x pinned and fails this assertion -- exactly the #6951 ``TokenizersBackend`` regression."""
    result = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        cwd = str(_BACKEND_DIR),
        env = {
            **__import__("os").environ,
            "STUB_HOME": str(tmp_path),
            **({"SPOOF_PATH": str(_SPOOF_PATH)} if _SPOOF_PATH.exists() else {}),
        },
        capture_output = True,
        text = True,
    )
    assert result.returncode == 0, (
        "Worker preflight + activation harness crashed.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    parsed = _parse(result.stdout)
    assert parsed, f"No RESULT line.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Correct tier chosen for a transformers-5.x model (pure, deterministic; no network/GPU).
    assert parsed["tier"] == "530", (
        f"Wrong transformers tier for Qwen3.5 (expected 530, got {parsed['tier']}). "
        "Tier detection regressed."
    )

    # Activation must actually swap the resident transformers to the sidecar version. If a preflight
    # import cached 4.57.x first, the sidecar prepend is a no-op and this stays 4.57.x -- the bug.
    assert parsed["active"] == "5.3.0", (
        "Sidecar activation did NOT switch the in-process transformers to the model's 5.x version "
        f"(active={parsed['active']}, preloaded-before-activation={parsed['preload']}). A pre-activation "
        "transformers import (directly or via unsloth_zoo) defeated the sidecar; 5.x models (Qwen3.5, "
        "GLM-4.7, gemma-4) then fail with 'Tokenizer class TokenizersBackend does not exist'. See #6951."
    )


def test_worker_activates_transformers_510_for_astral(tmp_path):
    """Astral custom-code checkpoints (#7353) must route to tier 510 and swap in the 5.10 sidecar.

    Without this, training fails before model load with
    ``No module named 'transformers.tokenization_utils_tokenizers'`` because the default
    4.57.x sidecar is still active after worker preflight.
    """
    result = subprocess.run(
        [sys.executable, "-c", _SNIPPET_ASTRAL_510],
        cwd = str(_BACKEND_DIR),
        env = {
            **__import__("os").environ,
            "STUB_HOME": str(tmp_path),
            **({"SPOOF_PATH": str(_SPOOF_PATH)} if _SPOOF_PATH.exists() else {}),
        },
        capture_output = True,
        text = True,
    )
    assert result.returncode == 0, (
        "Astral worker preflight + activation harness crashed.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    parsed = _parse(result.stdout)
    assert parsed, f"No RESULT line.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    assert parsed["tier"] == "510", (
        f"Wrong transformers tier for Astral (expected 510, got {parsed['tier']}). "
        "Custom-code Astral checkpoints must use the 5.10 sidecar. See #7353."
    )
    assert parsed["active"] == "5.10.2", (
        "Sidecar activation did NOT switch the in-process transformers to Astral's 5.10.x version "
        f"(active={parsed['active']}, preloaded-before-activation={parsed['preload']}). "
        "A stale pre-activation transformers import defeats the sidecar; Astral then crashes loading "
        "its remote tokenizer. See #7353."
    )
