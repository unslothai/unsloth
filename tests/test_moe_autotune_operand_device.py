# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""On a device_map split across GPU models, the MoE autotune lookup read the current device:
a layer on another card took tiles keyed, tuned and SMEM-fitted for the current card (and #14578
persists them across processes). With ``device=`` the key, the tuning run and the SMEM fit follow
the experts' card; ``device=None`` and the current card keep today's path."""

import contextlib
import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

MOE_DIR = ROOT / "unsloth" / "kernels" / "moe"
CAPS = {0: (9, 0), 1: (8, 0)}
SMEM = {0: 232448, 1: 101376}


@pytest.fixture
def ac(monkeypatch, tmp_path):
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
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_MOE_DISABLE_AUTOTUNE", raising = False)
    current = {"index": 0}

    @contextlib.contextmanager
    def device(index):
        previous, current["index"] = current["index"], index
        try:
            yield
        finally:
            current["index"] = previous

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current["index"])
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda index = None: CAPS[current["index"] if index is None else index]
    )
    module = importlib.import_module("unsloth.kernels.moe.autotune_cache")
    monkeypatch.setattr(module, "_device_smem_per_block", lambda: SMEM[current["index"]])
    seen = []

    def fake_tuning(*args, **kwargs):
        seen.append((current["index"], torch.cuda.get_device_capability()))
        return module._get_default_configs(n_dim = 2 * 768, k_dim = 2048, dtype = torch.bfloat16)

    monkeypatch.setattr(module, "_run_moe_autotuning", fake_tuning)
    monkeypatch.setattr(module, "save_cached_config", lambda *a, **k: None)
    module.clear_cache()
    yield module, seen
    module.clear_cache()
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            sys.modules.pop(name, None)


DIMS = dict(num_experts = 128, hidden_dim = 2048, intermediate_dim = 768, top_k = 8, dtype = torch.bfloat16)


def test_a_layer_on_another_card_tunes_and_keys_for_that_card(ac):
    module, seen = ac
    module.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cuda", 1))
    assert seen == [(1, (8, 0))]
    # The current card's model is a different key, so it tunes for itself.
    module.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cuda", 0))
    assert seen == [(1, (8, 0)), (0, (9, 0))]
    assert torch.cuda.current_device() == 0


def test_no_device_or_the_current_card_keeps_the_current_path(ac):
    module, seen = ac
    first = module.get_or_autotune_moe_kernels(**DIMS)
    second = module.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cuda", 0))
    assert seen == [(0, (9, 0))] and first is second


def test_cpu_or_unindexed_devices_keep_the_current_path(ac):
    module, seen = ac
    module.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cpu"))
    module.get_or_autotune_moe_kernels(**DIMS, device = "cuda")
    assert seen == [(0, (9, 0))]
