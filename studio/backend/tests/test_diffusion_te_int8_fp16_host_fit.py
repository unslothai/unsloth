# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""float16 cards: an automatic pick takes Qwen-Image-2.1's hosted int8 ConvRot encoder only when the dense float16
load cannot be held in host RAM, priced by the small-host route's own resolver; explicit picks and the switch win."""

from __future__ import annotations

import json
import struct
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_small_host as sh  # noqa: E402
import core.inference.diffusion_te_prequant as tpq  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402
from core.inference.diffusion_precision import AutoTeQuant  # noqa: E402

BASE = "Qwen/Qwen-Image-2.1"
INT8_NAME = "Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors"
CUDA_FP16 = types.SimpleNamespace(device = "cuda", dtype = torch.float16)
SWITCH = "UNSLOTH_TE_INT8_FP16"
GIB = 1024
# host readings (total, available) MiB: a 51 GB Colab high-RAM T4 (as measured: 49760 MiB available before the load)
# and a 31 GB Kaggle T4 host
HOST_51 = (52 * GIB, 49760)
HOST_31 = (31 * GIB, 28 * GIB)


def _fam():
    return detect_family("", override = "qwen-image-2.1")


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(SWITCH, raising = False)
    monkeypatch.delenv(sh.HOST_RAM_LIMIT_ENV, raising = False)
    monkeypatch.delenv(sh.SMALL_HOST_ENV, raising = False)
    monkeypatch.setattr(sh, "host_commit_mib", lambda: None)
    # no local snapshot: the family's bf16 table prices the load
    monkeypatch.setattr(sh, "resolve_snapshot_dir", lambda repo, cache_dir = None: None)
    tpq._FP16_TE_LOGGED.clear()


def _host(monkeypatch, reading):
    monkeypatch.setattr(sh, "host_ram_mib", lambda: reading)


def _sources(mode):
    src = tpq.te_prequant_sources_for_base(_fam(), BASE, te_quant_mode = mode, target = CUDA_FP16)
    return src.get("text_encoder")


def test_auto_on_a_51_gb_host_keeps_the_dense_encoder(monkeypatch):
    _host(monkeypatch, HOST_51)
    assert _sources(AutoTeQuant("int8")) is None
    fits, why = tpq.dense_fp16_encoder_fits_host(_fam(), BASE)
    assert fits is True and "MiB" in why


def test_auto_on_a_31_gb_host_takes_the_int8_encoder(monkeypatch):
    _host(monkeypatch, HOST_31)
    src = _sources(AutoTeQuant("int8"))
    assert src is not None and src.filename == INT8_NAME
    assert tpq.te_candidate_filenames(src) == (INT8_NAME,)


def test_auto_inside_an_unset_request_flips_the_same_way(monkeypatch):
    for reading, int8 in ((HOST_51, False), (HOST_31, True)):
        _host(monkeypatch, reading)
        with tpq.auto_te_request(True):
            assert (_sources("int8") is not None) is int8


def test_an_explicit_int8_request_is_honoured_on_a_large_host(monkeypatch):
    _host(monkeypatch, HOST_51)
    src = _sources("int8")
    assert src is not None and src.filename == INT8_NAME


def test_the_switch_wins_both_ways(monkeypatch):
    _host(monkeypatch, HOST_51)
    monkeypatch.setenv(SWITCH, "1")
    assert _sources(AutoTeQuant("int8")) is not None
    _host(monkeypatch, HOST_31)
    monkeypatch.setenv(SWITCH, "0")
    assert _sources(AutoTeQuant("int8")) is None
    assert _sources("int8") is None


def test_unreadable_host_ram_keeps_the_int8_encoder(monkeypatch):
    _host(monkeypatch, (None, None))
    assert _sources(AutoTeQuant("int8")) is not None


def test_bf16_cards_are_unchanged(monkeypatch):
    _host(monkeypatch, HOST_51)
    bf16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)
    assert not tpq.int8_convrot_te_runs_fp16(_fam(), bf16, auto = True, base = BASE)


def _write_component(
    root,
    name,
    mib,
    dtype = "BF16",
    library = "transformers",
):
    sub = root / name
    sub.mkdir()
    numel = mib * 1024 * 1024 // 2
    header = json.dumps(
        {"w": {"dtype": dtype, "shape": [numel], "data_offsets": [0, numel * 2]}}
    ).encode()
    with open(sub / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header)
        f.truncate(8 + len(header) + numel * 2)
    return library


@pytest.mark.parametrize("reading", [HOST_51, HOST_31, (24 * GIB, 20 * GIB), (64 * GIB, 60 * GIB)])
def test_the_pick_agrees_with_the_small_host_decision_on_the_same_snapshot(
    tmp_path, monkeypatch, reading
):
    """Same components, same host reading: int8 exactly when the dense load would take the low-RAM route."""
    index = {"_class_name": "QwenImage21Pipeline"}
    for name, mib in (("transformer", 13540), ("text_encoder", 16700), ("vae", 650)):
        index[name] = [_write_component(tmp_path, name, mib), "X"]
    (tmp_path / "model_index.json").write_text(json.dumps(index))
    monkeypatch.setattr(sh, "resolve_snapshot_dir", lambda repo, cache_dir = None: tmp_path)
    _host(monkeypatch, reading)
    decision = sh.decide_small_host(
        sh.stored_components(tmp_path),
        torch.float16,
        device = "cuda",
        host_total_mib = reading[0],
        host_available_mib = reading[1],
        host_commit_mib = None,
        pread_libraries = ("transformers",),
    )
    int8 = _sources(AutoTeQuant("int8")) is not None
    assert int8 is decision.engaged
    fits, dense = sh.dense_load_fits(
        sh.stored_components(tmp_path),
        torch.float16,
        host_total_mib = reading[0],
        host_available_mib = reading[1],
    )
    assert dense == decision.dense_host_mib and fits is (not decision.engaged)
