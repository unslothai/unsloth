# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A quantized V cache is priced with flash attention on, because llama.cpp will
not run it any other way.

The estimator used to pass ``flash_attn = False`` to ``_estimate_kv_cache_bytes``
unconditionally, as the conservative arm. It is conservative for a *choice*; for a
quantized V cache it is not a choice. llama.cpp turns flash attention on itself --

    enabling flash_attn since it is required for quantized V cache

-- and aborts the load without it, which ``_tensor_quant_kv_unsupported_binary``
already documents as "V cache quantization requires flash_attn". The False arm sets
``bpe_v = max(bpe_k, f16)``, so the entire quantized saving on the V axis was charged
straight back and the panel reported a launch that cannot happen.

Measured against llama-server b10632 (``version: 0.3.0-dev (build 10632, commit
11cd98842)``), Qwen3-0.6B-Q4_K_M, identical on the CPU and Vulkan builds:

    ctx    -ctk/-ctv   reserved before   allocated     after
    4096   q8_0            343 MiB        238 MiB     238 MiB
    4096   q4_0            287 MiB        126 MiB     126 MiB
    32768  q8_0           2744 MiB       1904 MiB    1904 MiB
    32768  q4_0           2296 MiB       1008 MiB    1008 MiB

The tests below do not need a binary: they assert the arithmetic identity that
produced those numbers, which is that BOTH axes shrink with the cache type, not just
K. A regression puts the f16 V back and the ratio jumps.
"""

import sys
import types as _types
from pathlib import Path

# Stub heavy / unavailable deps before importing the module under test.
# Same block, same reasons, as tests/test_memory_estimate.py.

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

import importlib.util as _ilu  # noqa: E402

import pytest  # noqa: E402

import routes.inference as ri  # noqa: E402
from core.inference.llama_cpp import _kv_bytes_per_elem  # noqa: E402

# Reuse the GGUF blob builder rather than copying it, for the reason its first
# borrower gives: a second copy of the writer would drift from the parser.
_kv_spec = _ilu.spec_from_file_location(
    "_kv_cache_estimation_for_flash_attn_parity",
    Path(__file__).resolve().parent / "test_kv_cache_estimation.py",
)
_kv_mod = _ilu.module_from_spec(_kv_spec)
_kv_spec.loader.exec_module(_kv_mod)
_make_gguf_bytes = _kv_mod._make_gguf_bytes

# Qwen3-0.6B's real shape, so the ratios below are the ones that were measured.
_QWEN3_FIELDS = {
    "context_length": 40960,
    "block_count": 28,
    "embedding_length": 1024,
    "attention.head_count": 16,
    "attention.head_count_kv": 8,
    "attention.key_length": 128,
    "attention.value_length": 128,
}


@pytest.fixture(autouse = True)
def _clear_flash_attn_parity_caches():
    ri._estimate_files_cache.clear()
    ri._estimate_config_cache.clear()
    yield
    ri._estimate_files_cache.clear()
    ri._estimate_config_cache.clear()


@pytest.fixture
def qwen3_shaped_gguf(tmp_path):
    # general.architecture goes in FIRST. The reader takes it before it knows which
    # prefix the rest of the keys carry, so a blob that writes it last parses as a
    # header with no dimensions and every figure below silently becomes zero.
    fields = {"general.architecture": "qwen3"}
    fields.update({f"qwen3.{k}": v for k, v in _QWEN3_FIELDS.items()})
    path = tmp_path / "qwen3-shaped.gguf"
    path.write_bytes(_make_gguf_bytes("qwen3", fields))
    return str(path)


def _kv_bytes(path, ctx, cache_type, slots = 1):
    return ri._gguf_runtime_bytes(path, ctx, None, slots, cache_type, False).kv_bytes


@pytest.mark.parametrize("cache_type", ["q8_0", "q4_0", "q5_1", "iq4_nl"])
@pytest.mark.parametrize("n_ctx", [4096, 32768])
def test_a_quantized_cache_shrinks_both_axes_not_only_k(qwen3_shaped_gguf, cache_type, n_ctx):
    """The load-bearing one. K and V are the same width in this model, so a correct
    price is exactly ``bpe(type) / bpe(f16)`` of the f16 price. The old behaviour
    shrank K and pinned V at f16, landing halfway."""
    f16 = _kv_bytes(qwen3_shaped_gguf, n_ctx, "f16")
    quantized = _kv_bytes(qwen3_shaped_gguf, n_ctx, cache_type)
    expected_ratio = _kv_bytes_per_elem(cache_type) / _kv_bytes_per_elem("f16")

    assert quantized == pytest.approx(f16 * expected_ratio, rel = 1e-6), (
        f"{cache_type} at {n_ctx} priced {quantized} bytes against an f16 price of "
        f"{f16}: ratio {quantized / f16:.4f}, expected {expected_ratio:.4f}. A ratio "
        f"near the midpoint means V is still being charged at f16, which is the "
        f"flash_attn = False arm on a cache llama.cpp will not start without it."
    )


def test_the_measured_llama_cpp_reservations_are_reproduced(qwen3_shaped_gguf):
    """The four cells from the docstring, as figures rather than as a ratio. These
    are what llama-server b10632 actually allocated for this model shape."""
    mib = 1024 * 1024
    for n_ctx, cache_type, allocated_mib in (
        (4096, "q8_0", 238.0),
        (4096, "q4_0", 126.0),
        (32768, "q8_0", 1904.0),
        (32768, "q4_0", 1008.0),
    ):
        priced = _kv_bytes(qwen3_shaped_gguf, n_ctx, cache_type) / mib
        assert priced == pytest.approx(allocated_mib, rel = 1e-6), (
            f"{cache_type} at {n_ctx}: priced {priced:.1f} MiB, llama-server b10632 "
            f"allocated {allocated_mib} MiB"
        )


def test_an_f16_cache_is_unchanged_by_the_flash_attn_rule(qwen3_shaped_gguf):
    """The fix must not move the unquantized path. f16 leaves flash attention to
    llama.cpp's own auto-detection, so the padded-V arm still applies and these
    figures are the ones the guard has always used."""
    mib = 1024 * 1024
    assert _kv_bytes(qwen3_shaped_gguf, 4096, "f16") / mib == pytest.approx(448.0)
    assert _kv_bytes(qwen3_shaped_gguf, 32768, "f16") / mib == pytest.approx(3584.0)
    assert _kv_bytes(qwen3_shaped_gguf, 4096, None) == _kv_bytes(qwen3_shaped_gguf, 4096, "f16")


@pytest.mark.parametrize("wide_type", ["bf16", "f32"])
def test_a_cache_wider_than_f16_does_not_force_flash_attention(qwen3_shaped_gguf, wide_type):
    """The rule keys on "quantized", not on "not f16". bf16 and f32 are neither
    quantized nor eligible for the forced-FA path, so they keep the conservative
    padding."""
    f16 = _kv_bytes(qwen3_shaped_gguf, 4096, "f16")
    wide = _kv_bytes(qwen3_shaped_gguf, 4096, wide_type)
    assert wide >= f16, f"{wide_type} priced {wide} below the f16 price {f16}"


def test_a_quantized_v_in_the_extra_arguments_is_read_the_same_way(qwen3_shaped_gguf):
    """The panel field is not the only way to ask for one. An extras-supplied
    --cache-type-v must reach the same rule, because it reaches the same child."""
    field = ri._gguf_runtime_bytes(qwen3_shaped_gguf, 4096, None, 1, "q4_0", False).kv_bytes
    extras = ri._gguf_runtime_bytes(
        qwen3_shaped_gguf,
        4096,
        ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"],
        1,
        None,
        False,
    ).kv_bytes
    assert extras == field, (
        f"extras priced {extras} where the field priced {field}; the launch cannot "
        f"tell the two apart, so neither should the estimate"
    )
