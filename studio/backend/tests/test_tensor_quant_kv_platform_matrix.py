# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OS x accelerator x cache-type matrix for the tensor-split quantized KV cache.

llama.cpp runs a quantized KV cache under ``--split-mode tensor`` since
ggml-org/llama.cpp#23792 (build b9455), so Unsloth stopped rewriting the requested
type for the tensor attempt. That removal touches no platform-conditional code, but
it changes the argv emitted on every platform, and the cells where tensor mode is
dropped BEFORE the cache is ever considered (CPU-only, single GPU, Apple) are the
ones most likely to regress silently: they emit the same command either way, so only
a per-cell record shows the drop still happens for the right reason.

Each cell records the emitted split mode and BOTH cache axes, because the whole
class of bug this file guards against is an axis being rewritten -- and the two
axes only differ when the user sets them separately.

Simulation notice: this suite runs on one host. Only Linux/NVIDIA is native. Windows,
WSL2 and macOS are ``sys.platform`` / ``platform.release`` monkeypatches via the
shared ``_apply_platform`` seam, Apple unified memory is a non-zero
``_apple_metal_memory_budget_bytes`` with an empty GPU probe, and every AMD cell is a
memory shape plus ``utils.hardware.IS_ROCM`` or the Vulkan flag. No ROCm runtime, no
Metal device, no Windows kernel and no llama-server process is exercised -- the child
is a captured ``Popen``. The authoritative signal for those remains the per-OS CI
matrix on real runners; this is the branch coverage one host can give.
"""

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
# tensor_is_viable is the expectation, not an input: tensor mode needs >= 2 devices,
# so every single-device and device-less cell must come out layer-split whatever the
# cache type is.
ACCELERATORS = [
    ("nvidia-multi", False, False, 0, [(0, 20_000, 24_000), (1, 20_000, 24_000)], True),
    ("nvidia-single", False, False, 0, [(0, 20_000, 24_000)], False),
    ("amd-rocm-multi", False, True, 0, [(0, 20_000, 24_000), (1, 20_000, 24_000)], True),
    ("amd-vulkan-multi", True, False, 0, [(0, 12_000, 16_000), (1, 12_000, 16_000)], True),
    ("apple-unified", False, False, 32 * 1024**3, [], False),
    ("cpu-only", False, False, 0, [], False),
]

# The types the launcher will emit (_VALID_CACHE_TYPES), plus the two shapes that
# only became reachable once the tensor gate was removed.
CACHE_CELLS = [
    ("f16", "f16", "f16"),
    ("q8_0", "q8_0", "q8_0"),
    ("q4_0", "q4_0", "q4_0"),
    ("q5_1", "q5_1", "q5_1"),
    # iq4_nl is in _VALID_CACHE_TYPES, so the gate removal admits it. ggml-org/
    # llama.cpp#27116 reports it still asserting under a tensor split on b10441;
    # that abort carries split_axis, so _should_record_tensor_split_abort latches
    # it and the route falls back. Studio's job is only to emit what was asked.
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
    """No OS and no accelerator rewrites the requested KV cache type.

    Studio emits one type on both axes for a managed request, so both must equal
    what was asked, on every cell, whether or not tensor mode survives.
    """
    tensor_viable = accelerator[5]
    kv_type, expect_k, expect_v = cache
    backend, gguf = _cell_backend(tmp_path, monkeypatch, platform, accelerator)

    cmd = _launch(backend, gguf, tensor_parallel = True, cache_type_kv = kv_type)["cmd"]
    ks, vs = _axes(cmd)

    assert ks[-1:] == [expect_k], f"K axis rewritten: {ks}"
    assert vs[-1:] == [expect_v], f"V axis rewritten: {vs}"
    assert backend.cache_type_kv == kv_type
    # Tensor mode is a >= 2-device feature; below that the split-mode group must be
    # gone entirely, not just set to layer, so an extras --tensor-split cannot
    # re-engage it.
    if tensor_viable:
        assert cmd[cmd.index("--split-mode") + 1] == "tensor"
    else:
        assert "--split-mode" not in cmd
        assert "--tensor-split" not in cmd


@pytest.mark.parametrize("platform,accelerator,cache", MATRIX)
def test_asymmetric_axes_reach_every_platform_unchanged(
    tmp_path, monkeypatch, platform, accelerator, cache
):
    """A per-axis request survives on every cell, with the quantized axis on K.

    The pre-#23792 gate tested EVERY axis and rewrote both, so an asymmetric pair
    is the shape that regresses first if a whitelist is ever reintroduced -- under
    any name, which is why this asserts the axes rather than the absence of an
    attribute.
    """
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
    """The tensor-branch env scrub owns the split, not the cache type.

    LLAMA_ARG_CACHE_TYPE_K/_V must reach the child on every cell, while the tensor
    split Unsloth generates itself is still cleared so a stale one cannot override
    the ratio this launch computed.
    """
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
        # Tensor mode owns the ratio it computed, so a stale inherited one goes.
        assert "LLAMA_ARG_TENSOR_SPLIT" not in env
    else:
        # A layer load only clears inherited placement when it also inherited a
        # non-layer LLAMA_ARG_SPLIT_MODE; a bare ratio is a valid layer ratio and
        # is left to the child. Pinned so the cache assertions above cannot be
        # read as a claim about placement.
        assert env["LLAMA_ARG_TENSOR_SPLIT"] == "9,1"
    # Env-only: Studio emits no flag of its own, so it records no type. Pinned
    # because /status and the reload matcher both read this, and the matcher
    # compares it against an intent field that is also None for this shape.
    assert backend.cache_type_kv is None


@pytest.mark.parametrize(
    "platform,accelerator",
    [pytest.param(p, a, id = f"{p[0]}-{a[0]}") for p in PLATFORMS for a in ACCELERATORS],
)
def test_a_quantized_cache_survives_the_tensor_to_layer_downgrade(
    tmp_path, monkeypatch, platform, accelerator
):
    """A downgrade removes the split-mode group and nothing else.

    Layer split has always supported a quantized cache. Before #23792 the tensor
    attempt dropped it and a downgrade restored it; now it is never dropped. Both
    routes have to end at the same command, so this pins the outcome rather than
    the mechanism -- the assertion the restore-path test used to carry.
    """
    backend, gguf = _cell_backend(tmp_path, monkeypatch, platform, accelerator)
    # The session latch is the cheapest downgrade to force and needs no VRAM shape.
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
    """llama.cpp hard-errors on --flash-attn off under --split-mode tensor.

    Verified against current master (src/llama-context.cpp): the quantized-KV guard
    #23792 deleted is gone, but the one directly above it is not --
    "SPLIT_MODE_TENSOR requires flash_attn to be enabled" still returns nullptr, and
    AUTO is upgraded to ENABLED. A quantized V axis independently needs FA too, so
    this pair became reachable in tensor mode only once the cache stopped being
    rewritten. Pinned on every cell because the remedy differs per platform and the
    failure is a startup abort, not a downgrade.
    """
    tensor_viable = accelerator[5]
    kv_type, _k, _v = cache
    backend, gguf = _cell_backend(tmp_path, monkeypatch, platform, accelerator)

    cmd = _launch(backend, gguf, tensor_parallel = True, cache_type_kv = kv_type)["cmd"]

    if not tensor_viable:
        return  # layer split; llama.cpp imposes nothing here
    assert cmd[cmd.index("--split-mode") + 1] == "tensor"
    if "--flash-attn" in cmd:
        assert cmd[cmd.index("--flash-attn") + 1] != "off", _stable_join(cmd)
    assert "-fa" not in cmd or cmd[cmd.index("-fa") + 1] != "off"


def _stable_join(cmd: list[str]) -> str:
    return " ".join(cmd)


# ── Old installs ────────────────────────────────────────────────────────────


def test_a_config_saved_by_a_pre_23792_studio_still_loads(tmp_path, monkeypatch):
    """Upgrading Studio must not invalidate a saved model config.

    A user who set "Tensor Parallelism + q8_0" before this change had it silently
    coerced to f16. The same saved config now gets what it asked for -- which is
    the fix -- and nothing about reading it changes: no field is added, removed or
    renamed, so an old config loads here and a new one loads on an old Studio.
    """
    backend, gguf = _cell_backend(tmp_path, monkeypatch, PLATFORMS[0], ACCELERATORS[0])

    # Exactly the fields a pre-#23792 Studio wrote for this combination.
    cmd = _launch(backend, gguf, tensor_parallel = True, cache_type_kv = "q8_0")["cmd"]

    assert cmd[cmd.index("--split-mode") + 1] == "tensor"
    assert cmd[cmd.index("--cache-type-k") + 1] == "q8_0"


def test_the_load_intent_gained_no_field(tmp_path):
    """Schema tripwire for both directions of the upgrade.

    The fix threads a second cache type through the planner, and the cheap way to
    do that would have been a new GgufLoadIntent field -- which an older Studio
    reading a newer config would reject. It is a function parameter instead; this
    fails if that ever changes.
    """
    from dataclasses import fields

    from core.inference.llama_cpp import GgufLoadIntent

    names = {f.name for f in fields(GgufLoadIntent)}

    assert "scratch_cache_type_kv" not in names
    # The one the UI does own is still there and still spelled the same.
    assert "cache_type_kv" in names
    assert "tensor_parallel" in names


# ── An inherited cache type llama.cpp cannot parse must not kill both attempts ──


@pytest.mark.parametrize(
    "platform,accelerator",
    [pytest.param(p, a, id = f"{p[0]}-{a[0]}") for p in PLATFORMS for a in ACCELERATORS],
)
@pytest.mark.parametrize("bad", ["q3_K", "typo", ""])
def test_an_unparseable_inherited_cache_type_is_dropped(
    tmp_path, monkeypatch, platform, accelerator, bad
):
    """kv_cache_type_from_str aborts the child on an unknown type, at argument
    parsing -- and the layer retry inherits the same env, so BOTH attempts would
    fail and the user would be left with no server at all.

    The managed path has always been allow-listed at emission; the env path was
    only scrubbed as a side effect of the tensor-mode gate, so it needs the check
    on its own now. Split-mode independent: llama.cpp parses it the same way.
    """
    backend, gguf = _cell_backend(tmp_path, monkeypatch, platform, accelerator)
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", bad)
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "q8_0")

    env = _launch(backend, gguf, tensor_parallel = True)["env"]

    assert env.get("LLAMA_ARG_CACHE_TYPE_K") in (None, "")
    # The valid axis is untouched -- this drops what llama.cpp rejects, nothing more.
    assert env["LLAMA_ARG_CACHE_TYPE_V"] == "q8_0"


@pytest.mark.parametrize(
    "platform,accelerator",
    [pytest.param(p, a, id = f"{p[0]}-{a[0]}") for p in PLATFORMS for a in ACCELERATORS],
)
def test_a_miscased_inherited_cache_type_is_normalised_not_dropped(
    tmp_path, monkeypatch, platform, accelerator
):
    """kv_cache_type_from_str is case-sensitive, so "Q8_0" aborts the child just as
    hard as a typo -- but the user clearly meant a type that exists, and the managed
    path already lowercases for exactly this reason. Match it rather than dropping
    the request on a capitalisation."""
    backend, gguf = _cell_backend(tmp_path, monkeypatch, platform, accelerator)
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", "Q8_0")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "IQ4_NL")

    env = _launch(backend, gguf, tensor_parallel = True)["env"]

    assert env["LLAMA_ARG_CACHE_TYPE_K"] == "q8_0"
    assert env["LLAMA_ARG_CACHE_TYPE_V"] == "iq4_nl"


# ── An inherited value llama.cpp cannot parse verbatim ──────────────────────


@pytest.mark.parametrize(
    "platform,accelerator",
    [pytest.param(p, a, id = f"{p[0]}-{a[0]}") for p in PLATFORMS for a in ACCELERATORS],
)
@pytest.mark.parametrize(
    "raw,expected",
    [
        (" q8_0 ", "q8_0"),  # surrounding whitespace
        ("\tq4_0\n", "q4_0"),  # any whitespace, not just spaces
        (" Q8_0 ", "q8_0"),  # whitespace AND case together
    ],
)
def test_a_whitespace_padded_inherited_cache_type_is_rewritten(
    tmp_path, monkeypatch, platform, accelerator, raw, expected
):
    """common_arg::get_value_from_env hands the raw getenv string straight to
    kv_cache_type_from_str, which compares it to ggml_type_name(t) exactly and
    throws otherwise -- neither side trims. So " q8_0 " aborts the child just as
    a typo does, and normalising against an already-stripped copy would compare
    equal and leave the padding on the child.
    """
    backend, gguf = _cell_backend(tmp_path, monkeypatch, platform, accelerator)
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", raw)
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "q8_0")

    env = _launch(backend, gguf, tensor_parallel = True)["env"]

    assert env["LLAMA_ARG_CACHE_TYPE_K"] == expected
    assert env["LLAMA_ARG_CACHE_TYPE_V"] == "q8_0"
