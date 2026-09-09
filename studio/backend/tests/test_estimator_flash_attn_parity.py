# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A quantized V cache is priced with flash attention on, because llama.cpp will
not run it any other way.

``flash_attn = False`` used to be passed unconditionally as the conservative arm. It is
conservative for a *choice*; for a quantized V cache it is not one. llama.cpp turns
flash attention on itself --

    enabling flash_attn since it is required for quantized V cache

-- and aborts without it, which ``_tensor_quant_kv_unsupported_binary`` documents as
"V cache quantization requires flash_attn". The False arm sets
``bpe_v = max(bpe_k, f16)``, charging the entire quantized saving on the V axis back
and reporting a launch that cannot happen.

Measured against llama-server b10632 (commit ``11cd98842``), Qwen3-0.6B-Q4_K_M,
identical on the CPU and Vulkan builds:

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

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import routes.inference as ri  # noqa: E402
from models.inference import EstimateMemoryRequest  # noqa: E402
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


def _estimate_through_route(**kwargs):
    """Drive the real route far enough to capture what it hands the sizing call."""
    try:
        asyncio.run(
            ri.estimate_memory(
                EstimateMemoryRequest(**kwargs),
                fastapi_request = None,
                current_subject = "test",
            )
        )
    except RuntimeError:
        pass


def _kv_bytes(
    path,
    ctx,
    cache_type,
    slots = 1,
):
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


# The round that followed: four placement inputs the estimate was reading wrong.
# Each of these fails on the parent commit.


def test_a_cpu_only_manual_launch_is_not_charged_for_a_pinned_card(monkeypatch, qwen3_shaped_gguf):
    """Manual with zero layers is a CPU-only launch, and the loader drops the split
    flags for it. Charging the pinned card count added per-device pipeline overhead
    and replicated the context-linear compute term for buffers no card allocates:
    measured on a two-card pin, 1039 -> 2105 MiB at 4k and 1417 -> 5129 MiB at 262k.

    Driven through the route, because the defect was in what the route HANDS the
    device count, not in the device count itself. Asserting the two helpers in
    isolation passes on the unfixed tree."""
    one = ri._gguf_runtime_bytes(
        qwen3_shaped_gguf, 32768, None, 4, "f16", False, None, None, n_devices = 1
    )
    two = ri._gguf_runtime_bytes(
        qwen3_shaped_gguf, 32768, None, 4, "f16", False, None, None, n_devices = 2
    )
    assert (
        two.compute_bytes > one.compute_bytes
    ), "a second device has to cost something, or this test proves nothing"

    seen = {}

    def _spy(*args, **kwargs):
        seen["n_devices"] = kwargs.get("n_devices")
        raise RuntimeError("stop here; the argument is the whole assertion")

    monkeypatch.setattr(ri, "_gguf_memory_breakdown", _spy)
    monkeypatch.setattr(
        ri,
        "_cached_estimate_config",
        lambda *a, **kw: SimpleNamespace(
            identifier = "local",
            gguf_file = qwen3_shaped_gguf,
            is_gguf = True,
            gguf_variant = None,
            gguf_mmproj_file = None,
            gguf_mtp_file = None,
            gguf_dspark_file = None,
            gguf_dflash_file = None,
        ),
    )
    for layers, expected in ((0, 1), (40, 2)):
        seen.clear()
        _estimate_through_route(
            model_path = qwen3_shaped_gguf,
            gpu_memory_mode = "manual",
            gpu_layers = layers,
            selected_gpu_ids = [0, 1],
        )
        assert seen.get("n_devices") == expected, (
            f"--gpu-layers {layers} with a two-card pin priced "
            f"{seen.get('n_devices')} devices, expected {expected}"
        )


def test_an_inherited_gpu_layer_count_is_read_in_auto(monkeypatch, qwen3_shaped_gguf):
    """Auto emits no -ngl on the fitting path, so LLAMA_ARG_N_GPU_LAYERS is the only
    layer policy the child sees and llama.cpp's fitter will not overrule it
    (common/fit.cpp:463, "n_gpu_layers already set by user")."""
    monkeypatch.delenv("LLAMA_ARG_N_GPU_LAYERS", raising = False)
    assert ri._gguf_offloaded_layer_fraction("auto", None, 27, None) == 1.0

    monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", "0")
    assert ri._gguf_offloaded_layer_fraction("auto", None, 27, None) == 0.0

    monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", "14")
    half = ri._gguf_offloaded_layer_fraction("auto", None, 27, None)
    assert 0.0 < half < 1.0, f"a partial inherited count priced {half}"

    # -1 and auto are llama.cpp's own default, so they leave the fitter free to choose
    # and are not an override.
    for auto_value in ("-1", "auto", ""):
        monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", auto_value)
        assert ri._gguf_offloaded_layer_fraction("auto", None, 27, None) == 1.0


def test_an_explicit_ngl_in_the_extras_still_beats_the_inherited_count(
    monkeypatch, qwen3_shaped_gguf
):
    """argv is parsed after the environment and wins, so an extras -ngl is the answer
    whatever the env says."""
    monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", "0")
    assert ri._gguf_offloaded_layer_fraction("auto", None, 27, ["-ngl", "999"]) == 1.0


def test_pass_through_adapters_are_charged_and_follow_the_base_placement(
    tmp_path, qwen3_shaped_gguf
):
    """llama.cpp loads every --lora / --control-vector into resident tensors on top of
    the base model, on the base tensor's buffer type. The files term prices only the
    target and its companions, so without this an adapter load was a fit on the base
    model's size alone."""
    from core.inference.llama_cpp import _sidecar_adapter_bytes

    lora = tmp_path / "adapter.gguf"
    size = 7 * 1024 * 1024
    lora.write_bytes(b"\0" * size)
    assert _sidecar_adapter_bytes(["--lora", str(lora)]) == size
    # A named file that cannot be stat'd is the "engaged but unsized" case.
    assert _sidecar_adapter_bytes(["--lora", str(tmp_path / "missing.gguf")]) is None
    # And no adapters is zero, not None, so the marker stays off for every ordinary load.
    assert _sidecar_adapter_bytes([]) == 0

    # The helper pre-dates this; what is new is that the panel asks it. Driven through
    # the breakdown so the test fails on a tree where the term is computed and dropped.
    config = SimpleNamespace(
        identifier = "local",
        gguf_file = qwen3_shaped_gguf,
        is_gguf = True,
        gguf_variant = None,
        gguf_mmproj_file = None,
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
    )
    bare = ri._gguf_memory_breakdown(config, qwen3_shaped_gguf, n_ctx = 4096)
    with_lora = ri._gguf_memory_breakdown(
        config, qwen3_shaped_gguf, n_ctx = 4096, llama_extra_args = ["--lora", str(lora)]
    )
    assert bare is not None and with_lora is not None
    assert with_lora.weights_bytes - bare.weights_bytes == size, (
        f"a {size}-byte adapter moved weights by " f"{with_lora.weights_bytes - bare.weights_bytes}"
    )
    assert with_lora.total_bytes - bare.total_bytes == size
    assert not with_lora.adapters_unsized

    missing = ri._gguf_memory_breakdown(
        config,
        qwen3_shaped_gguf,
        n_ctx = 4096,
        llama_extra_args = ["--lora", str(tmp_path / "missing.gguf")],
    )
    assert (
        missing is not None and missing.adapters_unsized
    ), "an unsizable adapter has to mark the total a floor rather than vanish"
