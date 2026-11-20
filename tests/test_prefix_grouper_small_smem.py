# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""PrefixGrouper flex tiles on 64 KB shared memory GPUs (T4 sm75, AMD 64 KB LDS).

The 16-bit 128 / 64 tile needs 98304 B at head_dim 128, so on T4 the compile check spent about 3
minutes failing before PrefixGrouper fell back. Now that case skips the compile up front, and
UNSLOTH_GRPO_PREFIX_GROUPER_SMALL_SMEM=1 picks smaller forward and backward tiles from the queried
limit. Larger GPUs keep the exact same options.
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
        "UNSLOTH_GRPO_PREFIX_GROUPER_SMALL_SMEM",
        "UNSLOTH_GRPO_PREFIX_GROUPER_COMPILE_CHECK",
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


def _opt_in(
    monkeypatch,
    kernel,
    on = True,
):
    monkeypatch.setattr(kernel, "_SMALL_SMEM_OPT_IN", on, raising = False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("limit", [V100, L4, A100, H100])
@pytest.mark.parametrize("opt_in", [False, True])
def test_large_smem_options_unchanged(kernel, monkeypatch, dtype, limit, opt_in):
    _limits(monkeypatch, kernel, {"cuda:0": limit})
    _opt_in(monkeypatch, kernel, opt_in)
    dev = torch.device("cuda:0")
    assert kernel._kernel_options_for_dtype(dtype, False, dev) == MAIN_16BIT
    assert kernel._kernel_options_for_dtype(dtype, True, dev) == {**MAIN_16BIT, "num_stages": 1}
    assert kernel._kernel_options_for_dtype(torch.float32, False, dev) == MAIN_FP32
    assert kernel._kernel_options_for_dtype(dtype) == MAIN_16BIT
    _limits(monkeypatch, kernel, {"meta": limit})
    q = torch.empty(1, 2, 4, 128, dtype = dtype, device = "meta")
    assert kernel._small_smem_skip_reason(q) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_small_smem_default_keeps_main_options(kernel, monkeypatch, dtype):
    _limits(monkeypatch, kernel, {"cuda:0": T4})
    assert kernel._SMALL_SMEM_OPT_IN is False
    dev = torch.device("cuda:0")
    assert kernel._kernel_options_for_dtype(dtype, False, dev) == MAIN_16BIT


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_small_smem_opt_in_pins_forward_and_backward(kernel, monkeypatch, dtype):
    _limits(monkeypatch, kernel, {"cuda:0": T4, "meta": T4})
    _opt_in(monkeypatch, kernel)
    dev = torch.device("cuda:0")
    opts = kernel._kernel_options_for_dtype(dtype, False, dev)
    # Every option is pass-prefixed: nothing leaks from the forward tile into the backward.
    assert set(opts) == {
        "fwd_BLOCK_M", "fwd_BLOCK_N", "fwd_num_stages", "fwd_num_warps",
        "bwd_BLOCK_M1", "bwd_BLOCK_N1", "bwd_BLOCK_M2", "bwd_BLOCK_N2",
        "bwd_num_stages", "bwd_num_warps",
    }  # fmt: skip
    # The flex templates' static asserts, against the 128 sparse block of create_block_mask.
    assert 128 % opts["fwd_BLOCK_M"] == 0 and 128 % opts["fwd_BLOCK_N"] == 0
    assert opts["bwd_BLOCK_N1"] % opts["bwd_BLOCK_M1"] == 0
    assert opts["bwd_BLOCK_M2"] % opts["bwd_BLOCK_N2"] == 0
    for key in ("bwd_BLOCK_M1", "bwd_BLOCK_N1", "bwd_BLOCK_M2", "bwd_BLOCK_N2"):
        assert 128 % opts[key] == 0
    # Measured on T4: forward 64 / 64 needs 65536 B at any stage count, so the forward tile stays
    # below that, and the backward stays below the 32 / 64 tile that needed 81920 B.
    assert opts["fwd_BLOCK_M"] * opts["fwd_BLOCK_N"] < 64 * 64
    assert opts["bwd_BLOCK_M1"] * opts["bwd_BLOCK_N1"] < 32 * 64
    # One variant only: the one-stage retry does not create a second config, callers can't mutate it.
    assert kernel._kernel_options_for_dtype(dtype, True, dev) == opts
    opts["fwd_BLOCK_M"] = 1
    assert kernel._kernel_options_for_dtype(dtype, False, dev)["fwd_BLOCK_M"] == 32
    # fp32 keeps its 32 / 32 tile and the existing one-stage retry.
    assert kernel._kernel_options_for_dtype(torch.float32, False, dev) == MAIN_FP32
    q = torch.empty(1, 2, 4, 128, dtype = dtype, device = "meta")
    assert kernel._small_smem_skip_reason(q) is None


def test_skip_reason_only_where_main_tiles_cannot_fit(kernel, monkeypatch):
    _limits(monkeypatch, kernel, {"meta": T4})

    def q(dtype = torch.float16, d = 128):
        return torch.empty(1, 2, 4, d, dtype = dtype, device = "meta")

    assert "65536" in kernel._small_smem_skip_reason(q())
    assert kernel._small_smem_skip_reason(q(torch.bfloat16, 256)) is not None
    assert kernel._small_smem_skip_reason(q(d = 64)) is None  # 128 / 64 fits at head_dim 64
    assert kernel._small_smem_skip_reason(q(torch.float32)) is None  # fp32 has its own retry
    monkeypatch.setattr(torch.version, "hip", "7.1", raising = False)
    assert kernel._small_smem_skip_reason(q()) is None  # ROCm keeps today's behaviour
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    _opt_in(monkeypatch, kernel)
    assert kernel._small_smem_skip_reason(q()) is None


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


@pytest.mark.parametrize(
    "limit,opt_in,small", [(T4, True, True), (T4, False, False), (A100, True, False)]
)
def test_run_flex_passes_the_device_options(kernel, monkeypatch, limit, opt_in, small):
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
    _opt_in(monkeypatch, kernel, opt_in)
    x = torch.zeros(1, 2, 200, 8, dtype = torch.float16)
    for T in (200, 130):
        out = kernel._run_flex(
            x[:, :, :T], x[:, :, :T], x[:, :, :T], None, False, None, True, T, 256
        )
        assert out.shape[2] == T
    assert len(seen) == 2 and seen[0] == seen[1]
    assert seen[0] == (kernel._SMALL_SMEM_16BIT_OPTIONS if small else MAIN_16BIT)


def test_compile_check_skips_the_failing_compile(kernel, monkeypatch, capsys):
    """T4 + head_dim 128 + fp16, no opt-in: no strict compile at all, PrefixGrouper reports off."""
    runs = []
    monkeypatch.setattr(kernel, "_run_flex", lambda *a, **k: runs.append(1))
    _limits(monkeypatch, kernel, {"cpu": T4})
    seg, _ = kernel.build_seg_info_multigroup([(130, [20, 30])], torch.device("cpu"))
    Q = torch.zeros(1, seg.T, 4, 128, dtype = torch.float16)
    K = torch.zeros(1, seg.T, 2, 128, dtype = torch.float16)
    with pytest.raises(kernel.PrefixGrouperCompileError):
        kernel.flex_shared_prefix_attention(Q, K, K, seg, block_mask = object())
    assert runs == []
    assert "UNSLOTH_GRPO_PREFIX_GROUPER_SMALL_SMEM" in kernel.compile_check_failed()
    assert "PrefixGrouper is off" in capsys.readouterr().out
    with pytest.raises(kernel.PrefixGrouperCompileError):  # later calls fail fast, same reason
        kernel.flex_shared_prefix_attention(Q, K, K, seg, block_mask = object())
    assert runs == []


def test_user_tile_overrides_go_through_the_compile_check(kernel, monkeypatch):
    # PG_FLEX_BF16_BLOCK_M / _N set by the user: try their tiles instead of skipping up front.
    _limits(monkeypatch, kernel, {"meta": T4})
    q = torch.empty(1, 2, 4, 128, dtype = torch.float16, device = "meta")
    assert kernel._small_smem_skip_reason(q) is not None
    monkeypatch.setattr(kernel, "_BF16_BLOCK_M", 32)
    monkeypatch.setattr(kernel, "_BF16_BLOCK_N", 32)
    assert kernel._small_smem_skip_reason(q) is None
