# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""MoE kernel configs follow the operands' GPU, not the current one (H100 + L4 in one process).

The launcher enters X.device, so configs picked from the current card's capability, shared memory
or disk cache entry could exceed the operand card's limit. The cache key also carries the GPU model
(name, capability, SM count), so two models sharing a capability (L40S + RTX 4090) tune separately,
and a pre-model-key disk entry is never reused (nothing records which model wrote it). CPU only: device properties
are mocked."""

import importlib
import json
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

MOE_DIR = ROOT / "unsloth" / "kernels" / "moe"

# index -> (capability, shared_memory_per_block_optin[, name, SM count])
H100, L4 = ((9, 0), 232448, "NVIDIA H100", 132), ((8, 9), 101376, "NVIDIA L4", 58)
F32, BF16, F16 = torch.float32, torch.bfloat16, torch.float16
DTYPES = [(BF16, None), (F16, None), (F32, BF16), (F32, F16)]
SHAPES = [(2048, 768), (2048, 1408), (1024, 352), (768, 2048)]
KINDS = ("fwd", "dX", "dW")


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
    monkeypatch.delenv("UNSLOTH_MOE_DISABLE_AUTOTUNE", raising = False)
    module = importlib.import_module("unsloth.kernels.moe.autotune_cache")
    module.clear_cache()
    yield module
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            sys.modules.pop(name, None)


def _gpus(
    monkeypatch,
    cards,
    current = 0,
):
    def index(device = None):
        if device is None:
            return current
        if isinstance(device, int):
            return device
        device = torch.device(device)
        return current if device.index is None else device.index

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(cards))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d = None: cards[index(d)][0])
    reads = []

    def properties(d = None):
        card = cards[index(d)]
        reads.append(index(d))
        return types.SimpleNamespace(
            shared_memory_per_block_optin = card[1],
            name = card[2] if len(card) > 2 else f"GPU sm{card[0][0]}{card[0][1]}",
            major = card[0][0],
            minor = card[0][1],
            multi_processor_count = card[3] if len(card) > 3 else 80,
        )

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    return reads


def _key(ac, card, index):
    # The key get_or_autotune_moe_kernels writes for 8 x 2048 x 768 top-2 bf16 on card ``index``.
    return ac._get_cache_key(
        8, 2048, 768, 2, BF16, card[0], 8192, None, device_model = ac._device_model(index)
    )


def _smem(ac, configs, act, weight, card):
    a, w = act.itemsize, (weight or act).itemsize
    return [ac._estimate_smem(k, c, a, w, card[0]) for k, c in zip(KINDS, configs)]


def _dicts(configs):
    return [c.__dict__.copy() for c in configs]


@pytest.mark.parametrize("which", ["heuristic", "default"])
@pytest.mark.parametrize("act,weight", DTYPES)
@pytest.mark.parametrize("hidden,inter", SHAPES)
def test_fallback_configs_fit_the_operand_card(ac, monkeypatch, which, act, weight, hidden, inter):
    _gpus(monkeypatch, [H100, L4], current = 0)
    fn = ac._get_heuristic_configs if which == "heuristic" else ac._get_default_configs
    for n, k in ((2 * inter, hidden), (hidden, inter)):
        configs = fn(
            n_dim = n, k_dim = k, dtype = act, weight_dtype = weight, device = torch.device("cuda", 1)
        )
        assert max(_smem(ac, configs, act, weight, L4)) <= L4[1]


def test_heuristic_env_path_uses_the_operand_card(ac, monkeypatch):
    # Current H100, operand L4: the H100 fit left the bf16 heuristic dX at 122,880 B (> 101,376 B).
    _gpus(monkeypatch, [H100, L4], current = 0)
    monkeypatch.setenv("UNSLOTH_MOE_DISABLE_AUTOTUNE", "1")
    dims = dict(num_experts = 8, hidden_dim = 2048, intermediate_dim = 768, top_k = 2)
    for act, weight in ((BF16, None), (F32, BF16)):
        configs = ac.get_or_autotune_moe_kernels(
            dtype = act, weight_dtype = weight, device = torch.device("cuda", 1), **dims
        )
        assert max(_smem(ac, configs, act, weight, L4)) <= L4[1]


def _big_configs():
    from unsloth.kernels.moe.grouped_gemm.kernels.tuning import (
        KernelConfigBackward_dW,
        KernelConfigBackward_dX,
        KernelConfigForward,
    )
    return (
        KernelConfigForward(
            BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 128, BLOCK_SIZE_K = 128, num_warps = 8, num_stages = 3
        ),
        KernelConfigBackward_dX(
            BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 64, BLOCK_SIZE_K = 256, num_warps = 8, num_stages = 4
        ),
        KernelConfigBackward_dW(
            BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 256, BLOCK_SIZE_K = 256, num_warps = 8, num_stages = 4
        ),
    )


def test_cache_lookup_and_tuning_follow_the_operand_card(ac, monkeypatch):
    _gpus(monkeypatch, [H100, L4], current = 0)
    dims = dict(num_experts = 8, hidden_dim = 2048, intermediate_dim = 768, top_k = 2)
    h100_key = ac._get_cache_key(8, 2048, 768, 2, BF16, H100[0], 8192, None)
    ac.save_cached_config(h100_key, *_big_configs())

    tuned_on = []

    def tune(
        *args,
        device = None,
        **kwargs,
    ):
        tuned_on.append(device)
        return ac._get_heuristic_configs(n_dim = 1536, k_dim = 2048, dtype = BF16, device = device)

    monkeypatch.setattr(ac, "_run_moe_autotuning", tune)
    operand = torch.device("cuda", 1)
    configs = ac.get_or_autotune_moe_kernels(dtype = BF16, device = operand, **dims)
    # The H100 entry is not reused for the L4 operand; tuning runs on (and is saved for) the L4.
    assert tuned_on == [operand]
    assert max(_smem(ac, configs, BF16, None, L4)) <= L4[1]
    l4_key = _key(ac, L4, 1)
    saved = json.loads(Path(ac._get_cache_file_path(l4_key)).read_text())
    assert tuple(saved["device_capability"]) == L4[0]
    # Both entries survive: the capability check reads the operand card, not the current one.
    assert ac.load_cached_config(l4_key, device = operand) is not None
    assert ac.load_cached_config(h100_key, device = torch.device("cuda", 0)) is not None


def test_disk_hit_refits_to_the_operand_card(ac, monkeypatch):
    _gpus(monkeypatch, [H100, L4], current = 0)
    operand = torch.device("cuda", 1)
    l4_key = _key(ac, L4, 1)
    ac.save_cached_config(l4_key, *_big_configs(), device = operand)
    monkeypatch.setattr(ac, "_run_moe_autotuning", lambda *a, **k: pytest.fail("cache miss"))
    configs = ac.get_or_autotune_moe_kernels(
        num_experts = 8, hidden_dim = 2048, intermediate_dim = 768, top_k = 2, dtype = BF16, device = operand
    )
    assert max(_smem(ac, configs, BF16, None, L4)) <= L4[1]


def test_smem_limit_reads_the_operand_index(ac, monkeypatch):
    _gpus(monkeypatch, [H100, L4], current = 0)
    assert ac._device_smem_per_block() == H100[1]
    assert ac._device_smem_per_block(torch.device("cuda", 1)) == L4[1]
    assert ac._device_smem_per_block(1) == L4[1]
    assert ac._device_smem_per_block(torch.device("cuda")) == H100[1]


@pytest.mark.parametrize("card", [((7, 5), 65536), L4, ((8, 0), 166912), H100, ((12, 0), 101376)])
def test_single_gpu_configs_unchanged(ac, monkeypatch, card):
    # One card: passing its device must pick exactly what the device-less call picked.
    _gpus(monkeypatch, [card], current = 0)
    for fn in (ac._get_heuristic_configs, ac._get_default_configs):
        for act, weight in DTYPES:
            for hidden, inter in SHAPES:
                kw = dict(n_dim = 2 * inter, k_dim = hidden, dtype = act, weight_dtype = weight)
                base = _dicts(fn(**kw))
                assert _dicts(fn(device = torch.device("cuda", 0), **kw)) == base
                assert _dicts(fn(device = torch.device("cuda"), **kw)) == base
                assert _dicts(fn(device = 0, **kw)) == base


DIMS = dict(num_experts = 8, hidden_dim = 2048, intermediate_dim = 768, top_k = 2, dtype = BF16)


def _record_tuning(ac, monkeypatch):
    seen = []

    def tune(
        *args,
        device = None,
        **kwargs,
    ):
        seen.append(None if device is None else ac._cuda_index(device))
        return ac._get_heuristic_configs(n_dim = 1536, k_dim = 2048, dtype = BF16, device = device)

    monkeypatch.setattr(ac, "_run_moe_autotuning", tune)
    monkeypatch.setattr(ac, "save_cached_config", lambda *a, **k: None)
    return seen


def test_each_card_tunes_and_keys_for_itself(ac, monkeypatch):
    _gpus(monkeypatch, [H100, L4], current = 0)
    seen = _record_tuning(ac, monkeypatch)
    ac.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cuda", 1))
    ac.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cuda", 0))
    assert seen == [1, 0]


def test_no_device_or_the_current_card_shares_the_current_entry(ac, monkeypatch):
    _gpus(monkeypatch, [H100, L4], current = 0)
    seen = _record_tuning(ac, monkeypatch)
    first = ac.get_or_autotune_moe_kernels(**DIMS)
    second = ac.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cuda", 0))
    assert seen == [None] and first is second


def test_cpu_or_unindexed_devices_keep_the_current_path(ac, monkeypatch):
    _gpus(monkeypatch, [H100, L4], current = 0)
    seen = _record_tuning(ac, monkeypatch)
    first = ac.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cpu"))
    assert ac.get_or_autotune_moe_kernels(**DIMS, device = "cuda") is first
    assert seen == [None]


L40S = ((8, 9), 101376, "NVIDIA L40S", 142)
RTX_4090 = ((8, 9), 101376, "NVIDIA GeForce RTX 4090", 128)


def test_two_models_with_one_capability_tune_separately(ac, monkeypatch):
    _gpus(monkeypatch, [L40S, RTX_4090], current = 0)
    seen = _record_tuning(ac, monkeypatch)
    ac.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cuda", 0))
    ac.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cuda", 1))
    assert seen == [0, 1]


def test_identical_cards_share_one_entry_without_repeated_property_reads(ac, monkeypatch):
    reads = _gpus(monkeypatch, [L40S, L40S], current = 0)
    seen = _record_tuning(ac, monkeypatch)
    first = ac.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cuda", 0))
    second = ac.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cuda", 1))
    assert seen == [0] and first is second
    before = list(reads)
    for index in (0, 1, 0, 1):
        ac.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cuda", index))
    assert reads == before and seen == [0]


def _write_legacy_entry(ac, card):
    fwd, dx, dw = ac._get_default_configs(n_dim = 1536, k_dim = 2048, dtype = BF16)
    legacy = ac._get_cache_key(8, 2048, 768, 2, BF16, card[0], 8192, None)
    Path(ac._get_cache_file_path(legacy)).write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "device_capability": list(card[0]),
                "schema_version": ac._CACHE_SCHEMA_VERSION,
                "triton_version": getattr(ac.triton, "__version__", None),
                "config_fwd": fwd.__dict__,
                "config_bwd_dx": dx.__dict__,
                "config_bwd_dw": dw.__dict__,
            }
        ),
        encoding = "utf-8",
    )


def test_a_pre_model_key_entry_is_retuned_once_and_saved_under_the_model_key(ac, monkeypatch):
    # Every visible card matching cannot prove which model wrote the entry (shared ~/.cache, swapped card).
    _gpus(monkeypatch, [L40S, L40S], current = 0)
    real_save = ac.save_cached_config
    seen = _record_tuning(ac, monkeypatch)
    monkeypatch.setattr(ac, "save_cached_config", real_save)
    _write_legacy_entry(ac, L40S)
    ac.get_or_autotune_moe_kernels(**DIMS)
    ac.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cuda", 1))
    assert seen == [None]
    assert Path(ac._get_cache_file_path(_key(ac, L40S, 0))).exists()
    # A fresh process loads the model-keyed entry instead of retuning or reading the legacy file.
    ac._kernel_config_cache.clear()
    ac._autotune_completed.clear()
    ac.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cuda", 1))
    assert seen == [None]


def test_a_pre_model_key_entry_is_not_trusted_on_a_mixed_model_host(ac, monkeypatch):
    # Same capability, different models: the old key cannot say which card wrote it.
    _gpus(monkeypatch, [L40S, RTX_4090], current = 0)
    seen = _record_tuning(ac, monkeypatch)
    _write_legacy_entry(ac, L40S)
    ac.get_or_autotune_moe_kernels(**DIMS, device = torch.device("cuda", 1))
    assert seen == [1]
