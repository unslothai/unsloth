# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""PrefixGrouper flex tiles on 64 KB shared memory GPUs (T4 sm75, AMD 64 KB LDS).

The 16-bit 128 / 64 tile needs 98304 B at head_dim 128, so the kernel never compiled there and
PrefixGrouper silently fell back. Small-smem devices now get pinned forward and backward tiles,
chosen from the queried limit; larger GPUs keep the exact same options as before.
"""

import importlib.util
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch.nn.attention.flex_attention")

KERNEL = (
    pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "utils" / "prefix_grouper_kernel.py"
)
MAIN_16BIT = {"BLOCK_M": 128, "BLOCK_N": 64}
MAIN_FP32 = {"BLOCK_M": 32, "BLOCK_N": 32}

T4 = 65536  # Triton max_shared_mem on sm75; gfx1151 / MI300 LDS is the same
V100 = 98304
L4 = 101376  # sm86 / sm89 / sm120
A100 = 166912
H100 = 232448


@pytest.fixture
def kernel(monkeypatch):
    for name in (
        "PG_FLEX_BLOCK_M",
        "PG_FLEX_BLOCK_N",
        "PG_FLEX_BF16_BLOCK_M",
        "PG_FLEX_BF16_BLOCK_N",
    ):
        monkeypatch.delenv(name, raising = False)
    name = "_pg_small_smem_kernel_under_test"
    spec = importlib.util.spec_from_file_location(name, KERNEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _limits(monkeypatch, kernel, limits):
    monkeypatch.setattr(kernel, "_SMEM_LIMIT", dict(limits), raising = False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("limit", [L4, A100, H100])
def test_large_smem_options_unchanged(kernel, monkeypatch, dtype, limit):
    _limits(monkeypatch, kernel, {"cuda:0": limit})
    dev = torch.device("cuda:0")
    assert kernel._kernel_options_for_dtype(dtype, False, dev) == MAIN_16BIT
    assert kernel._kernel_options_for_dtype(dtype, True, dev) == {**MAIN_16BIT, "num_stages": 1}
    assert kernel._kernel_options_for_dtype(torch.float32, False, dev) == MAIN_FP32
    # Callers without a device (older call sites) also keep the old options.
    assert kernel._kernel_options_for_dtype(dtype) == MAIN_16BIT


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("limit", [T4, V100])
def test_small_smem_pins_forward_and_backward(kernel, monkeypatch, dtype, limit):
    _limits(monkeypatch, kernel, {"cuda:0": limit})
    dev = torch.device("cuda:0")
    opts = kernel._kernel_options_for_dtype(dtype, False, dev)
    # Every option is pass-prefixed: nothing leaks from the forward tile into the backward.
    assert set(opts) == {
        "fwd_BLOCK_M",
        "fwd_BLOCK_N",
        "fwd_num_stages",
        "fwd_num_warps",
        "bwd_BLOCK_M1",
        "bwd_BLOCK_N1",
        "bwd_BLOCK_M2",
        "bwd_BLOCK_N2",
        "bwd_num_stages",
        "bwd_num_warps",
    }
    # The flex templates' static asserts, against the 128 sparse block of create_block_mask.
    assert 128 % opts["fwd_BLOCK_M"] == 0 and 128 % opts["fwd_BLOCK_N"] == 0
    assert opts["bwd_BLOCK_N1"] % opts["bwd_BLOCK_M1"] == 0
    assert opts["bwd_BLOCK_M2"] % opts["bwd_BLOCK_N2"] == 0
    for key in ("bwd_BLOCK_M1", "bwd_BLOCK_N1", "bwd_BLOCK_M2", "bwd_BLOCK_N2"):
        assert 128 % opts[key] == 0
    # 16-bit forward tile estimate at head_dim 128 (Q + pipelined K / V buffers) fits 64 KB.
    stages = max(opts["fwd_num_stages"] - 1, 1)
    smem = 2 * 128 * (opts["fwd_BLOCK_M"] + 2 * opts["fwd_BLOCK_N"] * stages)
    assert smem < T4
    # One variant only: the one-stage retry does not create a second config, callers can't mutate it.
    assert kernel._kernel_options_for_dtype(dtype, True, dev) == opts
    opts["fwd_BLOCK_M"] = 1
    assert kernel._kernel_options_for_dtype(dtype, False, dev)["fwd_BLOCK_M"] == 32
    # fp32 keeps its 32 / 32 tile and the existing one-stage retry.
    assert kernel._kernel_options_for_dtype(torch.float32, False, dev) == MAIN_FP32


def test_limit_queried_once_per_device(kernel, monkeypatch):
    calls = []

    class _Utils:
        @staticmethod
        def get_device_properties(index):
            calls.append(index)
            return {"max_shared_mem": T4 if index == 0 else A100}

    fake = types.ModuleType("triton")
    fake.runtime = types.SimpleNamespace(
        driver = types.SimpleNamespace(active = types.SimpleNamespace(utils = _Utils))
    )
    monkeypatch.setitem(sys.modules, "triton", fake)
    _limits(monkeypatch, kernel, {})
    for _ in range(3):
        assert kernel._small_smem(torch.device("cuda:0"))
        assert not kernel._small_smem(torch.device("cuda:1"))
    assert calls == [0, 1]
    assert not kernel._small_smem(torch.device("cpu"))
    assert kernel._SMEM_LIMIT["cpu"] is None


def test_falls_back_to_torch_properties(kernel, monkeypatch):
    fake = types.ModuleType("triton")  # no runtime attribute: the Triton query raises
    monkeypatch.setitem(sys.modules, "triton", fake)
    props = types.SimpleNamespace(shared_memory_per_block_optin = T4, shared_memory_per_block = 49152)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: props)
    _limits(monkeypatch, kernel, {})
    assert kernel._small_smem(torch.device("cuda:0"))
    assert kernel._SMEM_LIMIT["cuda:0"] == T4


@pytest.mark.parametrize("limit,small", [(T4, True), (A100, False)])
def test_run_flex_passes_the_device_options(kernel, monkeypatch, limit, small):
    seen = []

    def fake_flex(
        q,
        k,
        v,
        block_mask = None,
        enable_gqa = False,
        scale = None,
        kernel_options = None,
    ):
        seen.append(kernel_options)
        return q

    monkeypatch.setattr(kernel, "_flex_attention_compiled", fake_flex)
    _limits(monkeypatch, kernel, {"cpu": limit})
    x = torch.zeros(1, 2, 200, 8, dtype = torch.float16)
    for T in (200, 130):
        out = kernel._run_flex(
            x[:, :, :T], x[:, :, :T], x[:, :, :T], None, False, None, True, T, 256
        )
        assert out.shape[2] == T
    assert len(seen) == 2 and seen[0] == seen[1]
    if small:
        assert seen[0] == kernel._SMALL_SMEM_16BIT_OPTIONS
    else:
        assert seen[0] == MAIN_16BIT
