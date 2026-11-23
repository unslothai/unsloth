# SPDX-License-Identifier: AGPL-3.0-only
# CPU only: the per-GPU NVFP4 fused row gate scales with the weight size (capability monkeypatched).
import os
import types

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def N(monkeypatch):
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import nvfp4

    monkeypatch.setattr(nvfp4, "_IS_HIP", False)
    return nvfp4


def _weight(n, k):
    packed = torch.zeros((n, k // 2), dtype = torch.uint8)
    scale = torch.ones((n, k // 16), dtype = torch.float32)
    return packed, scale, torch.tensor([1.0])


def _rows(rows, width):
    # A CPU tensor dressed as CUDA: the gate reads only shape, dtype, device and is_cuda.
    t = torch.zeros((rows, width), dtype = torch.bfloat16)
    return types.SimpleNamespace(
        is_cuda = True,
        get_device = lambda: 0,
        numel = t.numel,
        dtype = t.dtype,
        shape = t.shape,
        device = t.device,
    )


def _decide(
    N,
    monkeypatch,
    cap,
    rows,
    n,
    k,
    dx = False,
    mode = "auto",
    max_rows = None,
):
    monkeypatch.setattr(N, "_fused_env", lambda: (mode, max_rows))
    monkeypatch.setattr(N, "_device_capability", lambda index: cap)
    packed, scale, gs = _weight(n, k)
    inner = n if dx else k
    return N._fused_ok(_rows(rows, inner), packed, scale, gs, inner, dx = dx)


# (capability, direction, N, K, rows fused up to): the measured crossovers of the bf16 sweep.
CASES = [
    ((8, 0), "fwd", 4096, 4096, 42),  # was 64 rows: slower than dense at M=64
    ((8, 0), "fwd", 3072, 768, 64),  # narrow reduction: c // K
    ((8, 0), "fwd", 4096, 14336, 64),
    ((8, 0), "dx", 2048, 2048, 16),
    ((8, 0), "dx", 8192, 8192, 64),
    ((8, 9), "fwd", 4096, 4096, 85),
    ((8, 9), "dx", 4096, 4096, 170),  # dX turns over later than forward
    ((8, 9), "dx", 3072, 768, 32),
    ((8, 9), "fwd", 4096, 14336, 256),
    ((12, 0), "fwd", 4096, 4096, 42),  # was 128 rows: +12% slower than dense at M=64
    ((12, 0), "dx", 4096, 4096, 73),  # fused dX still wins at M=64 here
    ((12, 0), "dx", 1024, 4096, 32),  # wide dX output: K // c
    ((12, 0), "dx", 3072, 768, 16),
    ((12, 0), "fwd", 3072, 768, 85),
    ((12, 0), "fwd", 8192, 8192, 128),
    ((12, 0), "dx", 8192, 8192, 256),
    ((12, 0), "dx", 2880, 2880, 16),  # N, K not multiples of 128: the floor
    ((8, 9), "fwd", 2880, 2880, 32),
]


@pytest.mark.parametrize("cap, direction, n, k, limit", CASES)
def test_row_limit_scales_with_weight(N, monkeypatch, cap, direction, n, k, limit):
    dx = direction == "dx"
    assert N._fused_max_rows(cap, n, k, dx) == limit
    assert _decide(N, monkeypatch, cap, limit, n, k, dx)
    assert not _decide(N, monkeypatch, cap, limit + 1, n, k, dx)


@pytest.mark.parametrize("cap", [(7, 5), (8, 6), (9, 0), (10, 0), (12, 1)])
def test_unmeasured_gpus_stay_dense(N, monkeypatch, cap):
    assert N._fused_max_rows(cap, 8192, 8192) == 0
    assert not _decide(N, monkeypatch, cap, 1, 8192, 8192)
    assert not _decide(N, monkeypatch, cap, 1, 8192, 8192, dx = True)


def test_env_overrides(N, monkeypatch):
    # UNSLOTH_NVFP4_FUSED=1 fuses past the gate (even on unmeasured GPUs), 0 never fuses, _MAX_ROWS is a flat limit.
    assert _decide(N, monkeypatch, (9, 0), 4096, 4096, 4096, mode = "1")
    assert _decide(N, monkeypatch, (12, 0), 4096, 4096, 4096, dx = True, mode = "1")
    assert not _decide(N, monkeypatch, (12, 0), 1, 8192, 8192, mode = "0")
    assert _decide(N, monkeypatch, (8, 0), 200, 3072, 768, max_rows = 200)
    assert not _decide(N, monkeypatch, (8, 0), 201, 3072, 768, max_rows = 200)
    assert _decide(N, monkeypatch, (9, 0), 200, 3072, 768, dx = True, max_rows = 200)


def test_env_parsing(N, monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_FUSED_MAX_ROWS", "96")
    monkeypatch.delenv("UNSLOTH_NVFP4_FUSED", raising = False)
    assert N._read_fused_env() == ("auto", 96)
    monkeypatch.setenv("UNSLOTH_NVFP4_FUSED", " Off ")
    assert N._read_fused_env()[0] in N._FUSED_OFF


def test_gate_is_monotone_in_weight_size(N):
    # At a fixed K the row limit never shrinks as N grows, and tops out at max_rows.
    for cap, gates in N._FUSED_ROW_GATE.items():
        for dx in (False, True):
            for k in (768, 4096, 16384):
                limits = [N._fused_max_rows(cap, 128 * i, k, dx) for i in range(1, 4097)]
                assert limits == sorted(limits) and limits[0] > 0 and limits[-1] == gates[dx][3]


def test_degenerate_weights_stay_dense(N, monkeypatch):
    assert (
        N._fused_max_rows((12, 0), 0, 4096) == 0 and N._fused_max_rows((12, 0), 4096, 0, True) == 0
    )
    monkeypatch.setattr(N, "_fused_env", lambda: ("auto", None))
    monkeypatch.setattr(N, "_device_capability", lambda index: (12, 0))
    flat = torch.zeros(4096 * 2048, dtype = torch.uint8)
    assert not N._fused_ok(_rows(1, 4096), flat, torch.ones(1, 1), torch.tensor([1.0]), 4096)


# (capability, SMs, dx, M, n_out, k_red, split): sm89 dX up to 32 rows skips split K once the output fills every SM.
SPLIT_CASES = [
    ((8, 9), 58, True, 1, 14336, 4096, 1),  # Llama-3-8B down_proj dX: was 4
    ((8, 9), 58, True, 32, 12288, 4096, 1),  # Qwen3-8B down_proj dX: was 4
    ((8, 9), 58, True, 64, 14336, 4096, 2),  # past 32 rows: unchanged
    ((8, 9), 58, True, 16, 4096, 14336, 8),  # 32 output tiles: still split
    ((8, 9), 58, False, 16, 4096, 14336, 4),  # forward: unchanged
    ((8, 0), 108, True, 16, 14336, 4096, 4),  # A100: split 4 stays fastest
    ((12, 0), 188, True, 16, 14336, 4096, 8),
]


@pytest.mark.parametrize("cap, sms, dx, M, n_out, k_red, split", SPLIT_CASES)
def test_split_k_per_capability(N, monkeypatch, cap, sms, dx, M, n_out, k_red, split):
    monkeypatch.setattr(N, "_device_capability", lambda index: cap)
    monkeypatch.setattr(N, "_sm_count", lambda index: sms)
    assert N._fused_config(M, n_out, k_red, torch.device("cuda", 0), dx)[-1] == split


@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (12, 0)])
@pytest.mark.parametrize("dx", [False, True])
def test_unaligned_k_stays_dense_in_both_directions(N, monkeypatch, cap, dx):
    # dX reduces over N, so K must be checked apart from inner.
    assert _decide(N, monkeypatch, cap, 16, 4096, 4096, dx)
    assert not _decide(N, monkeypatch, cap, 16, 4096, 4112, dx)
    assert not _decide(N, monkeypatch, cap, 16, 4112, 4096, dx)
    assert _decide(N, monkeypatch, cap, 16, 4096, 4112, dx, mode = "1")
