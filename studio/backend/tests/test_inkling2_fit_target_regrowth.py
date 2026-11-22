# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inkling 2 under --fit on: llama.cpp's fitter sizes only the load-time compute reserve, so
--fit-target also keeps free the growth the banded buffer takes after the first prompt."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_auto_fit_target_trim as trim  # noqa: E402
import test_inkling2_banded_quant_kv as ink  # noqa: E402
import test_llama_cpp_placement as placement  # noqa: E402
from core.inference import llama_cpp as lc  # noqa: E402

B = lc.LlamaCppBackend
GIB = 1024**3
_host = trim._host

_INKLING2_FIELDS = (
    "_architecture",
    "_context_length",
    "_n_layers",
    "_n_weight_layers",
    "_n_experts",
    "_n_kv_heads",
    "_n_kv_heads_by_layer",
    "_n_heads_by_layer",
    "_attn_res_block_size",
    "_embedding_length",
    "_feed_forward_length",
    "_expert_used_count",
    "_expert_feed_forward_length",
    "_vocab_size",
    "_kv_key_length",
    "_kv_value_length",
    "_kv_key_length_swa",
    "_kv_value_length_swa",
    "_sliding_window",
    "_sliding_window_pattern",
)


def _launch(
    tmp_path,
    monkeypatch,
    memory,
    support,
    *,
    arch = "inkling2",
    **kw,
):
    tmp_path.mkdir(parents = True, exist_ok = True)
    backend, gguf = placement._backend(tmp_path, vulkan = False, memory = list(memory))
    backend.probe_server_capabilities = lambda _binary = None: dict(trim.CAPS)
    backend._get_gguf_size_bytes = lambda _path: 80 * GIB
    ref = ink._inkling2_mini(support)
    fields = {f: getattr(ref, f) for f in _INKLING2_FIELDS}
    fields["_architecture"] = arch

    def _metadata(_path):
        for name, value in fields.items():
            setattr(backend, name, value)

    backend._read_gguf_metadata = _metadata
    monkeypatch.setattr(B, "inkling2_banded_support", classmethod(lambda cls, binary = None: support))
    cmd = placement._launch(backend, gguf, **kw)["cmd"]
    return backend, cmd


def _flag(cmd, name):
    vals = [cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a == name]
    return vals[-1] if vals else None


def _norm(cmd, root):
    """argv with the per-run port and temp paths masked."""
    out = [str(t).replace(str(root), "X") for t in cmd]
    for flag in ("--port", "--api-key-file"):
        if flag in out:
            out[out.index(flag) + 1] = flag.upper()
    return out


def _ub(cmd):
    v = _flag(cmd, "--ubatch-size")
    return int(v) if v else None


# Inkling-2-Mini UD-IQ1_S, CUDA, -np 4 --kv-unified, slot filled to 90%: compute buffer at load
# and after the prompt (MiB). #399 = llama.cpp #399 build b12628, old = shipped b12602.
MEASURED = [
    ("new", "f16", 512, 32768, 258, 747),
    ("new", "q8_0", 512, 32768, 258, 747),
    ("new", "f16", 512, 131072, 382, 837),
    ("new", "q8_0", 512, 131072, 541, 837),
    ("new", "f16", 512, 524288, 1078, 1077),
    ("new", "f16", 512, 1048576, 2102, 2101),
    ("new", "f16", 1024, 32768, 514, 1594),
    ("new", "f16", 1024, 1048576, 4206, 4205),
    ("new", "f16", 2048, 32768, 1035, 3582),
    ("new", "q8_0", 2048, 32768, 1035, 3582),
    ("new", "f16", 2048, 131072, 1443, 3925),
    ("new", "q8_0", 2048, 131072, 1443, 3925),
    ("new", "f16", 2048, 262144, 2275, 4492),
    ("new", "f16", 2048, 524288, 4323, 5836),
    ("old", "f16", 512, 32768, 541, 811),
    ("old", "f16", 512, 131072, 1357, 1497),
]
PAD_MIB = 100


@pytest.mark.parametrize("build, kv, ub, ctx, load, runtime", MEASURED)
def test_margin_covers_every_measured_growth(build, kv, ub, ctx, load, runtime):
    b = ink._inkling2_mini(ink.NEW_CUDA if build == "new" else ink.OLD_CUDA)
    assert b._inkling2_fit_regrowth_mib(ctx, ub, kv, n_parallel = 4) >= runtime - load + PAD_MIB


@pytest.mark.parametrize("build, kv, ub, ctx, load, runtime", MEASURED)
def test_reserve_bound_stays_under_every_measured_reserve(build, kv, ub, ctx, load, runtime):
    b = ink._inkling2_mini(ink.NEW_CUDA if build == "new" else ink.OLD_CUDA)
    act = b._INKLING2_FIT_ACT_MIB_PER_UB * ub * 1024 * 1024
    if build == "new":
        bound = max(
            b._INKLING2_FIT_RESERVE_ACT_FLOOR * act,
            b._INKLING2_FIT_RESERVE_ACT_BASE * act + b._INKLING2_FIT_RESERVE_MASK_BYTES * ub * ctx,
        )
    else:
        bound = (
            b._INKLING2_FIT_LEGACY_RESERVE_ACT * act
            + b._INKLING2_FIT_LEGACY_RESERVE_BYTES * ub * ctx
        )
    assert bound / (1024 * 1024) <= load


def test_runtime_estimate_is_left_as_it_is():
    # #18919's estimate is unchanged; the margin's runtime floor only engages past ~262K cells.
    b = ink._inkling2_mini(ink.NEW_CUDA)
    assert round(ink._compute(b, 131072, "f16", ub = 2048, slots = 4) / (1024 * 1024)) == 4036
    assert round(ink._compute(b, 524288, "f16", ub = 2048, slots = 4) / (1024 * 1024)) == 5572


def test_fixed_scale_matches_the_mini_activation_estimate():
    b = ink._inkling2_mini(ink.NEW_CUDA)
    for ub in (512, 1024, 2048):
        act = b._estimate_compute_buffer_bytes(n_ubatch = ub, n_parallel = 4) / (1024 * 1024)
        assert act == pytest.approx(b._INKLING2_FIT_ACT_MIB_PER_UB * ub, rel = 0.01)


def test_growth_does_not_scale_with_a_larger_model():
    small, mini = ink._inkling2_mini(ink.NEW_CUDA), ink._inkling2_mini(ink.NEW_CUDA)
    small._embedding_length *= 2
    small._expert_feed_forward_length *= 2
    assert small._inkling2_fit_regrowth_mib(131072, 2048, "f16") == pytest.approx(
        mini._inkling2_fit_regrowth_mib(131072, 2048, "f16"), abs = 1
    )


def test_unknown_context_takes_the_largest_growth():
    b = ink._inkling2_mini(ink.NEW_CUDA)
    fixed = [b._inkling2_fit_regrowth_mib(c, 2048, "f16") for c in (8192, 65536, 131072, 1048576)]
    assert b._inkling2_fit_regrowth_mib(8192, 2048, "f16", n_ctx_max = 1048576) >= max(fixed)


@pytest.mark.parametrize("ub", [512, 2048])
def test_older_build_growth_does_not_rise_past_the_measured_trend(ub):
    # Older builds measured 270 then 140 MiB of growth at 32K and 128K: the reserve outgrows the
    # runtime, so long contexts must not keep a planner bound's surplus free as growth.
    b = ink._inkling2_mini(ink.OLD_CUDA)
    by_ctx = [b._inkling2_fit_regrowth_mib(c, ub, "f16") for c in (32768, 131072, 524288, 1048576)]
    assert by_ctx == sorted(by_ctx, reverse = True)
    assert b._inkling2_fit_regrowth_mib(
        8192, ub, "f16", n_ctx_max = 1048576
    ) == b._inkling2_fit_regrowth_mib(8192, ub, "f16")


@pytest.mark.parametrize(
    "support, kv",
    [
        # Pre-#399 q8_0 runs dense, and Metal has no banded kernel: nothing modelled.
        (ink.OLD_CUDA, "q8_0"),
        (lc._Inkling2BandedSupport("metal", False), "f16"),
    ],
)
def test_no_banded_path_no_growth(support, kv):
    b = ink._inkling2_mini(support)
    assert b._inkling2_fit_regrowth_mib(32768, 2048, kv) == 0
    assert b._inkling2_fit_regrowth_mib(32768, 2048, kv, flash_attn = False) == 0


def test_regrowth_is_zero_off_inkling2():
    b = ink._inkling2_mini(ink.NEW_CUDA)
    b._architecture = "qwen3moe"
    assert b._inkling2_fit_regrowth_mib(32768, 2048, "f16") == 0
    b._architecture = "inkling"
    assert b._inkling2_fit_regrowth_mib(32768, 2048, "f16") == 0


@pytest.mark.parametrize("kv", ["f16", "q8_0"])
def test_fit_on_ub_2048_launch_spells_the_margin(tmp_path, monkeypatch, kv):
    # Experts in RAM raise -ub to 2048: 1024 default + 2690 MiB growth (measured 2547; #18919: unspelled).
    _b, cmd = _launch(
        tmp_path, monkeypatch, [(0, 48000, 49140)], ink.NEW_CUDA, n_ctx = 32768, cache_type_kv = kv
    )
    assert (_flag(cmd, "--fit"), _ub(cmd), _flag(cmd, "--fit-target")) == ("on", 2048, "3714")


@pytest.mark.parametrize("kv", ["f16", "q8_0"])
def test_fit_on_launch_keeps_the_regrowth_free(tmp_path, monkeypatch, kv):
    b, cmd = _launch(
        tmp_path,
        monkeypatch,
        [(0, 48000, 49140)],
        ink.NEW_CUDA,
        n_ctx = 32768,
        cache_type_kv = kv,
    )
    assert _flag(cmd, "--fit") == "on"
    regrowth = b._inkling2_fit_regrowth_mib(32768, _ub(cmd), kv)
    assert regrowth > 1000
    # llama.cpp's 1024 MiB default plus the regrowth, where main left the default unspelled.
    assert _flag(cmd, "--fit-target") == str(int(lc._LLAMA_FIT_TARGET_DEFAULT_MIB) + regrowth)


def test_fit_on_regrowth_stacks_on_the_small_card_trim(tmp_path, monkeypatch):
    b, cmd = _launch(
        tmp_path,
        monkeypatch,
        [(0, 24000, 24564)],
        ink.NEW_CUDA,
        n_ctx = 32768,
        cache_type_kv = "f16",
    )
    assert _flag(cmd, "--fit") == "on"
    regrowth = b._inkling2_fit_regrowth_mib(32768, _ub(cmd), "f16")
    assert regrowth > 0
    assert _flag(cmd, "--fit-target") == str(int(0.03 * 24564 + regrowth))


def test_fit_on_regrowth_reaches_every_per_device_margin(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "_fit_target_per_device_ok", lambda self, **_k: True)
    b, cmd = _launch(
        tmp_path,
        monkeypatch,
        [(0, 24000, 24564), (1, 48000, 49140)],
        ink.NEW_CUDA,
        n_ctx = 32768,
        cache_type_kv = "f16",
    )
    assert _flag(cmd, "--fit") == "on"
    regrowth = b._inkling2_fit_regrowth_mib(32768, _ub(cmd), "f16")
    target = _flag(cmd, "--fit-target")
    margins = [int(v) for v in target.split(",")]
    assert margins == [int(0.03 * 24564 + regrowth), 1024 + regrowth]


def test_a_metal_build_and_a_legacy_dense_q8_keep_todays_margin(tmp_path, monkeypatch):
    _b, cmd = _launch(
        tmp_path,
        monkeypatch,
        [(0, 48000, 49140)],
        ink.OLD_CUDA,
        n_ctx = 32768,
        cache_type_kv = "q8_0",
    )
    assert _flag(cmd, "--fit") == "on"
    assert _flag(cmd, "--fit-target") is None


@pytest.mark.parametrize("arch", ["qwen3moe", "inkling", "llama"])
def test_other_models_launch_byte_identical(tmp_path, monkeypatch, arch):
    # The regrowth helper forced to a large value must not reach a non-Inkling 2 argv.
    _b, before = _launch(
        tmp_path / "a", monkeypatch, [(0, 48000, 49140)], ink.NEW_CUDA, arch = arch, n_ctx = 32768
    )
    monkeypatch.setattr(B, "_inkling2_fit_regrowth_mib", lambda self, *a, **k: 4096, raising = False)
    _b, after = _launch(
        tmp_path / "b", monkeypatch, [(0, 48000, 49140)], ink.NEW_CUDA, arch = arch, n_ctx = 32768
    )
    assert _norm(before, tmp_path / "a") == _norm(after, tmp_path / "b")
    assert _flag(after, "--fit-target") is None
