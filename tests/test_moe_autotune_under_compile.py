# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""MoE kernel tuning called from a compiled block must run eagerly, not fall back to defaults."""

import importlib
import sys
import types
from pathlib import Path

import pytest
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("pandas")

MOE_DIR = ROOT / "unsloth" / "kernels" / "moe"


@pytest.fixture
def ac(monkeypatch):
    # Import autotune_cache without running unsloth/__init__.py (needs a GPU).
    for name, path in (
        ("unsloth", ROOT / "unsloth"),
        ("unsloth.kernels", ROOT / "unsloth" / "kernels"),
        ("unsloth.kernels.moe", MOE_DIR),
    ):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, pkg)
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            monkeypatch.delitem(sys.modules, name)
    module = importlib.import_module("unsloth.kernels.moe.autotune_cache")
    module.clear_cache()
    yield module
    module.clear_cache()
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            sys.modules.pop(name, None)


def _call_under_compile(ac, x):
    def block(x):
        fwd, dx, dw = ac.get_or_autotune_moe_kernels(
            num_experts = 8, hidden_dim = 128, intermediate_dim = 64, top_k = 2, dtype = x.dtype
        )
        return x + fwd.BLOCK_SIZE_M

    torch._dynamo.reset()
    try:
        return torch.compile(block, backend = "eager")(x)
    finally:
        torch._dynamo.reset()


def test_tuning_runs_eagerly_inside_a_compiled_caller(ac, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    monkeypatch.setattr(ac, "load_cached_config", lambda key: None)
    monkeypatch.setattr(ac, "save_cached_config", lambda *a, **k: None)
    monkeypatch.delenv("UNSLOTH_MOE_DISABLE_AUTOTUNE", raising = False)
    tuned = ac._get_default_configs(n_dim = 128, k_dim = 128)
    tuned[0].BLOCK_SIZE_M = 32  # differs from the default 128, so a fallback is visible
    seen = []

    def fake_tuning(*args, **kwargs):
        # A traced tuning never launches a kernel, so the real one falls back to the defaults.
        seen.append(torch.compiler.is_compiling())
        if torch.compiler.is_compiling():
            return ac._get_default_configs(n_dim = 128, k_dim = 128)
        return tuned

    monkeypatch.setattr(ac, "_run_moe_autotuning", fake_tuning)
    out = _call_under_compile(ac, torch.zeros(4))
    assert seen == [False]
    assert torch.equal(out, torch.full((4,), 32.0))
    # Cached for the next layer: no second tuning, same tuned config.
    out = _call_under_compile(ac, torch.zeros(4))
    assert seen == [False] and torch.equal(out, torch.full((4,), 32.0))


_REAL_TUNING = r"""
import logging, sys, torch
import unsloth  # noqa: F401
from unsloth.kernels.moe import autotune_cache as ac

ac.load_cached_config = lambda key: None
ac.save_cached_config = lambda *a, **k: None
records = []
handler = logging.Handler()
handler.emit = lambda record: records.append(record.getMessage())
ac.logger.addHandler(handler)
ac.logger.setLevel(logging.INFO)


def block(x):
    fwd, dx, dw = ac.get_or_autotune_moe_kernels(
        num_experts = 8, hidden_dim = 128, intermediate_dim = 64, top_k = 2, dtype = x.dtype
    )
    return x + fwd.BLOCK_SIZE_M


torch.compile(block)(torch.zeros(4, device = "cuda"))
text = "\n".join(records)
print(text)
sys.exit(0 if "auto-tuning completed" in text and "auto-tuning failed" not in text else 1)
"""


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU")
def test_real_tuning_under_compile_keeps_the_tuned_configs(tmp_path):
    # Real Triton autotuning launched from a compiled caller: no fallback to the default configs.
    import os
    import subprocess

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    env.pop("UNSLOTH_MOE_DISABLE_AUTOTUNE", None)
    run = subprocess.run(
        [sys.executable, "-c", _REAL_TUNING],
        env = env,
        cwd = tmp_path,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    if run.returncode != 0 and "auto-tuning" not in run.stdout:
        pytest.skip(f"unsloth did not import here: {run.stderr[-2000:]}")
    assert run.returncode == 0, run.stdout + run.stderr[-4000:]
