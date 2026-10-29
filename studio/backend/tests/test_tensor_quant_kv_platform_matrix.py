# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tensor-split quantized KV matrix over OS, accelerator and cache type; most cells are simulated."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _TESTS_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Both harnesses already exist; by path, because the tests dir is not a package.
_placement = _load("_placement_harness_tp_quant_kv", "test_llama_cpp_placement.py")
_platforms = _load("_platform_harness_tp_quant_kv", "test_llama_extra_args_platforms.py")

_backend = _placement._backend
_launch = _placement._launch
_apply_platform = _platforms._apply_platform
PLATFORMS = _platforms.PLATFORMS

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from utils.hardware import hardware as _hw  # noqa: E402


# (label, vulkan, is_rocm, apple_budget_bytes, memory, tensor_is_viable)
# tensor_is_viable is the expectation: tensor needs >= 2 devices, else layer-split.
ACCELERATORS = [
    ("nvidia-multi", False, False, 0, [(0, 20_000, 24_000), (1, 20_000, 24_000)], True),
    ("nvidia-single", False, False, 0, [(0, 20_000, 24_000)], False),
    ("amd-rocm-multi", False, True, 0, [(0, 20_000, 24_000), (1, 20_000, 24_000)], True),
    ("amd-vulkan-multi", True, False, 0, [(0, 12_000, 16_000), (1, 12_000, 16_000)], True),
    ("apple-unified", False, False, 32 * 1024**3, [], False),
    ("cpu-only", False, False, 0, [], False),
]

CACHE_CELLS = [
    ("f16", "f16", "f16"),
    ("q8_0", "q8_0", "q8_0"),
    ("q4_0", "q4_0", "q4_0"),
    ("q5_1", "q5_1", "q5_1"),
    # llama.cpp#27116: iq4_nl still asserts under tensor split; that abort is latched.
    ("iq4_nl", "iq4_nl", "iq4_nl"),
]

MATRIX = [
    pytest.param(p, a, c, id = f"{p[0]}-{a[0]}-{c[0]}")
    for p in PLATFORMS
    for a in ACCELERATORS
    for c in CACHE_CELLS
]


def _cell_backend(tmp_path, monkeypatch, platform, accelerator):
    _label, vulkan, is_rocm, apple_budget, memory, _viable = accelerator
    _apply_platform(monkeypatch, platform)
    # An inherited visibility mask would make "placement pinned nothing" read as a
    # pin on any box that exports CUDA_VISIBLE_DEVICES.
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising = False)
    for name in ("LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V", "LLAMA_ARG_SPLIT_MODE"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(_hw, "IS_ROCM", is_rocm, raising = False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_apple_metal_memory_budget_bytes",
        staticmethod(lambda: apple_budget),
    )
    backend, gguf = _backend(tmp_path, vulkan = vulkan, memory = list(memory))
    # A session latch from another cell would silently turn a tensor cell into a
    # layer cell and the assertion below would still pass for the wrong reason.
    backend._tensor_split_aborts = lambda *args, **kwargs: False
    return backend, gguf


def _axes(cmd: list[str]) -> tuple[list[str], list[str]]:
    """Every --cache-type-k and --cache-type-v value, in emission order."""
    return (
        [cmd[i + 1] for i, a in enumerate(cmd) if a == "--cache-type-k"],
        [cmd[i + 1] for i, a in enumerate(cmd) if a == "--cache-type-v"],
    )


@pytest.mark.parametrize("platform,accelerator,cache", MATRIX)
def test_the_requested_cache_reaches_every_platform_unchanged(
    tmp_path, monkeypatch, platform, accelerator, cache
):
    """The requested cache type must reach the child unchanged on every platform, tensor or not."""
    tensor_viable = accelerator[5]
    kv_type, expect_k, expect_v = cache
    backend, gguf = _cell_backend(tmp_path, monkeypatch, platform, accelerator)

    cmd = _launch(backend, gguf, tensor_parallel = True, cache_type_kv = kv_type)["cmd"]
    ks, vs = _axes(cmd)

    assert ks[-1:] == [expect_k], f"K axis rewritten: {ks}"
    assert vs[-1:] == [expect_v], f"V axis rewritten: {vs}"
    assert backend.cache_type_kv == kv_type
    # Below 2 devices the split-mode group must be gone, so extras cannot re-engage it.
    if tensor_viable:
        assert cmd[cmd.index("--split-mode") + 1] == "tensor"
    else:
        assert "--split-mode" not in cmd
        assert "--tensor-split" not in cmd


@pytest.mark.parametrize("platform,accelerator,cache", MATRIX)
def test_asymmetric_axes_reach_every_platform_unchanged(
    tmp_path, monkeypatch, platform, accelerator, cache
):
    """An asymmetric per-axis cache request must survive on every cell, with the quantized axis on K."""
    tensor_viable = accelerator[5]
    kv_type, _k, _v = cache
    if kv_type == "f16":
        pytest.skip("f16/f16 is not an asymmetric pair")
    backend, gguf = _cell_backend(tmp_path, monkeypatch, platform, accelerator)

    cmd = _launch(
        backend,
        gguf,
        tensor_parallel = True,
        extra_args = ["--cache-type-k", kv_type, "--cache-type-v", "f16", "--top-k", "5"],
    )["cmd"]
    ks, vs = _axes(cmd)

    # Extras are appended last and win per axis.
    assert ks[-1] == kv_type, f"K axis rewritten: {ks}"
    assert vs[-1] == "f16", f"V axis rewritten: {vs}"
    assert cmd[cmd.index("--top-k") + 1] == "5", "unrelated user extras dropped"
    if tensor_viable:
        assert cmd[cmd.index("--split-mode") + 1] == "tensor"
    else:
        assert "--split-mode" not in cmd


@pytest.mark.parametrize(
    "platform,accelerator",
    [pytest.param(p, a, id = f"{p[0]}-{a[0]}") for p in PLATFORMS for a in ACCELERATORS],
)
def test_an_inherited_quantized_kv_env_survives_on_every_platform(
    tmp_path, monkeypatch, platform, accelerator
):
    """The tensor env scrub keeps LLAMA_ARG_CACHE_TYPE_K/_V, clearing only Unsloth's generated split."""
    tensor_viable = accelerator[5]
    backend, gguf = _cell_backend(tmp_path, monkeypatch, platform, accelerator)
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", "q8_0")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "q4_0")
    monkeypatch.setenv("LLAMA_ARG_TENSOR_SPLIT", "9,1")

    out = _launch(backend, gguf, tensor_parallel = True)
    env = out["env"]

    assert env["LLAMA_ARG_CACHE_TYPE_K"] == "q8_0"
    assert env["LLAMA_ARG_CACHE_TYPE_V"] == "q4_0"
    if tensor_viable:
        assert "LLAMA_ARG_TENSOR_SPLIT" not in env
    else:
        # A bare inherited ratio is a valid layer ratio and is left to the child.
        assert env["LLAMA_ARG_TENSOR_SPLIT"] == "9,1"
    # Env-only: Unsloth emits no flag, so it records no type.
    assert backend.cache_type_kv is None


@pytest.mark.parametrize(
    "platform,accelerator",
    [pytest.param(p, a, id = f"{p[0]}-{a[0]}") for p in PLATFORMS for a in ACCELERATORS],
)
def test_a_quantized_cache_survives_the_tensor_to_layer_downgrade(
    tmp_path, monkeypatch, platform, accelerator
):
    """A quantized cache must survive the tensor-to-layer downgrade, which drops only split-mode."""
    backend, gguf = _cell_backend(tmp_path, monkeypatch, platform, accelerator)
    backend._tensor_split_aborts = lambda *args, **kwargs: True

    cmd = _launch(
        backend,
        gguf,
        tensor_parallel = True,
        extra_args = ["--cache-type-k", "q4_0", "--cache-type-v", "f16", "--top-k", "5"],
    )["cmd"]
    ks, vs = _axes(cmd)

    assert "--split-mode" not in cmd
    assert "--tensor-split" not in cmd
    assert ks[-1] == "q4_0", f"downgrade rewrote the K axis: {ks}"
    assert vs[-1] == "f16", f"downgrade rewrote the V axis: {vs}"
    assert cmd[cmd.index("--top-k") + 1] == "5"


@pytest.mark.parametrize("platform,accelerator,cache", MATRIX)
def test_a_tensor_launch_never_pairs_a_disabled_flash_attn(
    tmp_path, monkeypatch, platform, accelerator, cache
):
    """llama.cpp hard-errors on --flash-attn off with --split-mode tensor, so keep FA on."""
    tensor_viable = accelerator[5]
    kv_type, _k, _v = cache
    backend, gguf = _cell_backend(tmp_path, monkeypatch, platform, accelerator)

    cmd = _launch(backend, gguf, tensor_parallel = True, cache_type_kv = kv_type)["cmd"]

    if not tensor_viable:
        return
    assert cmd[cmd.index("--split-mode") + 1] == "tensor"
    if "--flash-attn" in cmd:
        assert cmd[cmd.index("--flash-attn") + 1] != "off", _stable_join(cmd)
    assert "-fa" not in cmd or cmd[cmd.index("-fa") + 1] != "off"


def _stable_join(cmd: list[str]) -> str:
    return " ".join(cmd)


def test_a_config_saved_by_a_pre_23792_studio_still_loads(tmp_path, monkeypatch):
    """A config saved before this change must still load: no field added, removed or renamed."""
    backend, gguf = _cell_backend(tmp_path, monkeypatch, PLATFORMS[0], ACCELERATORS[0])

    cmd = _launch(backend, gguf, tensor_parallel = True, cache_type_kv = "q8_0")["cmd"]

    assert cmd[cmd.index("--split-mode") + 1] == "tensor"
    assert cmd[cmd.index("--cache-type-k") + 1] == "q8_0"


def test_the_load_intent_gained_no_field(tmp_path):
    """The load intent must gain no field, so an older Unsloth can still read a newer config."""
    from dataclasses import fields

    from core.inference.llama_cpp import GgufLoadIntent

    names = {f.name for f in fields(GgufLoadIntent)}

    assert "scratch_cache_type_kv" not in names
    assert "cache_type_kv" in names
    assert "tensor_parallel" in names


@pytest.mark.parametrize(
    "platform,accelerator",
    [pytest.param(p, a, id = f"{p[0]}-{a[0]}") for p in PLATFORMS for a in ACCELERATORS],
)
@pytest.mark.parametrize("bad", ["q3_K", "typo", ""])
def test_an_unparseable_inherited_cache_type_is_dropped(
    tmp_path, monkeypatch, platform, accelerator, bad
):
    """An unparsable inherited LLAMA_ARG_CACHE_TYPE env value must be dropped, or both attempts fail."""
    backend, gguf = _cell_backend(tmp_path, monkeypatch, platform, accelerator)
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", bad)
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "q8_0")

    env = _launch(backend, gguf, tensor_parallel = True)["env"]

    assert env.get("LLAMA_ARG_CACHE_TYPE_K") in (None, "")
    assert env["LLAMA_ARG_CACHE_TYPE_V"] == "q8_0"


@pytest.mark.parametrize(
    "platform,accelerator",
    [pytest.param(p, a, id = f"{p[0]}-{a[0]}") for p in PLATFORMS for a in ACCELERATORS],
)
def test_a_miscased_inherited_cache_type_is_normalised_not_dropped(
    tmp_path, monkeypatch, platform, accelerator
):
    """kv_cache_type_from_str is case-sensitive, so a miscased inherited type must be lowercased first."""
    backend, gguf = _cell_backend(tmp_path, monkeypatch, platform, accelerator)
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", "Q8_0")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "IQ4_NL")

    env = _launch(backend, gguf, tensor_parallel = True)["env"]

    assert env["LLAMA_ARG_CACHE_TYPE_K"] == "q8_0"
    assert env["LLAMA_ARG_CACHE_TYPE_V"] == "iq4_nl"


@pytest.mark.parametrize(
    "platform,accelerator",
    [pytest.param(p, a, id = f"{p[0]}-{a[0]}") for p in PLATFORMS for a in ACCELERATORS],
)
@pytest.mark.parametrize(
    "raw,expected",
    [
        (" q8_0 ", "q8_0"),
        ("\tq4_0\n", "q4_0"),
        (" Q8_0 ", "q8_0"),
    ],
)
def test_a_whitespace_padded_inherited_cache_type_is_rewritten(
    tmp_path, monkeypatch, platform, accelerator, raw, expected
):
    """The raw getenv value is compared exactly, so padding aborts the child; rewrite the stripped value."""
    backend, gguf = _cell_backend(tmp_path, monkeypatch, platform, accelerator)
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", raw)
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "q8_0")

    env = _launch(backend, gguf, tensor_parallel = True)["env"]

    assert env["LLAMA_ARG_CACHE_TYPE_K"] == expected
    assert env["LLAMA_ARG_CACHE_TYPE_V"] == "q8_0"
