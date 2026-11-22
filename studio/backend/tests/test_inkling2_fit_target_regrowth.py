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


@pytest.mark.parametrize(
    "support, kv, ub, ctx, want",
    [
        # #399: 1200 KiB per ub token less the 2 B/cell the reserve leads by (measured +1081 at ub 1024).
        (ink.NEW_CUDA, "f16", 1024, 32768, 1136),
        (ink.NEW_CUDA, "q8_0", 1024, 32768, 1136),
        (ink.NEW_CUDA, "f16", 2048, 32768, 2272),
        (ink.NEW_CUDA, "f16", 512, 131072, 472),
        # The reserve has overtaken the regrowth at 1M (measured: no growth).
        (ink.NEW_CUDA, "f16", 1024, 1048576, 0),
        # Pre-#399 f16: 800 KiB per ub token less 3 B/cell.
        (ink.OLD_CUDA, "f16", 512, 32768, 352),
        # Pre-#399 q8_0 runs dense, and Metal has no banded kernel: nothing modelled.
        (ink.OLD_CUDA, "q8_0", 2048, 32768, 0),
        (lc._Inkling2BandedSupport("metal", False), "f16", 2048, 32768, 0),
    ],
)
def test_regrowth_is_the_runtime_minus_the_reserve(support, kv, ub, ctx, want):
    b = ink._inkling2_mini(support)
    assert b._inkling2_fit_regrowth_mib(ctx, ub, kv) == want


def test_regrowth_matches_the_estimate_it_comes_from():
    # Same terms as _compute_buffer_ctx_bytes: estimate - reserve-only estimate == regrowth.
    b = ink._inkling2_mini(ink.NEW_CUDA)
    ctx, ub = 32768, 2048
    total = b._compute_buffer_ctx_bytes(ctx, ub, "f16")
    reserve, regrown, fixed = b._inkling2_banded_terms(ctx, ub, "f16")
    assert total == max(reserve, regrown) + fixed
    assert (
        0 <= b._inkling2_fit_regrowth_mib(ctx, ub, "f16") - (regrown - reserve) / (1024 * 1024) < 1
    )


def test_regrowth_is_zero_off_inkling2():
    b = ink._inkling2_mini(ink.NEW_CUDA)
    b._architecture = "qwen3moe"
    assert b._inkling2_fit_regrowth_mib(32768, 2048, "f16") == 0
    b._architecture = "inkling"
    assert b._inkling2_fit_regrowth_mib(32768, 2048, "f16") == 0


@pytest.mark.parametrize("kv", ["f16", "q8_0"])
def test_fit_on_ub_2048_launch_spells_the_margin(tmp_path, monkeypatch, kv):
    # Experts in RAM raise -ub to 2048: 1024 default + 2272 MiB regrowth (main: unspelled 1024).
    _b, cmd = _launch(
        tmp_path, monkeypatch, [(0, 48000, 49140)], ink.NEW_CUDA, n_ctx = 32768, cache_type_kv = kv
    )
    assert (_flag(cmd, "--fit"), _ub(cmd), _flag(cmd, "--fit-target")) == ("on", 2048, "3296")


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
