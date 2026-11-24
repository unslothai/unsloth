# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""device_config's bit-exact probe verdict persists per card (diffusion_probe_cache): a returning process skips the
probe launches and their prewarm compiles; any key change, an out-of-memory probe or a failed probe re-probes.
CPU only (card, kernels and probe faked)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_int8_gemm as gemm
from core.inference import diffusion_probe_cache as pc


class _Ctx:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def card(monkeypatch, tmp_path):
    for name in (
        "UNSLOTH_DIFFUSION_INT8_GEMM",
        "UNSLOTH_DIFFUSION_INT8_GEMM_TILES",
        "UNSLOTH_DIFFUSION_PROBE_CACHE",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (8, 9))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "device", _Ctx)
    monkeypatch.setattr(gemm, "is_rocm_torch", lambda t: False)
    monkeypatch.setattr(gemm, "_triton_version_ok", lambda version = None: True)
    monkeypatch.setattr(gemm, "_kernels", lambda: object())
    monkeypatch.setattr(gemm, "_op", lambda: object())
    monkeypatch.setattr(gemm, "_triton38_sm8x", lambda cap: False)
    monkeypatch.setattr(gemm, "_tma_probe", lambda index, cap: None)
    monkeypatch.setattr(pc, "_cache_file", lambda: tmp_path / "diffusion_quant_probe.json")
    monkeypatch.setattr(pc, "fingerprint", lambda c: {"card": c, "uuid": "GPU-test", "torch": "x"})
    extra = {"libdevice": "toolkit", "ptxas": "13.0", "prefetch_fix": False}
    monkeypatch.setattr(gemm, "_cfg_extra", lambda: dict(extra))
    monkeypatch.setattr(gemm, "_PROBE_OOMS", [0])
    state = {"probes": [], "fail": set(), "oom": False}

    def probe(
        index,
        cfg,
        tma = False,
    ):
        state["probes"].append(cfg)
        if state["oom"]:
            gemm._PROBE_OOMS[0] += 1  # what _probe's except counts on an out-of-memory failure
            return False
        return cfg not in state["fail"]

    monkeypatch.setattr(gemm, "_probe", probe)
    want = gemm.arch_config((8, 9))
    if want is None:
        pytest.skip("no int8 GEMM tile listed for sm89")
    state["want"] = want
    state["extra"] = extra
    return state


def _new_process(monkeypatch):
    monkeypatch.setattr(gemm, "_DEVICE_CFG", {})
    monkeypatch.setattr(gemm, "_DEVICE_TILES", {})
    monkeypatch.setattr(gemm, "_DEVICE_TMA", {})


def test_returning_process_reads_the_verdict_back(monkeypatch, card):
    _new_process(monkeypatch)
    cfg = gemm.device_config(0)
    assert cfg == card["want"] and card["probes"]
    tiles = gemm._DEVICE_TILES[0]
    first = list(card["probes"])
    _new_process(monkeypatch)
    card["probes"].clear()
    assert gemm._gemm_probe_jobs(0) == []  # nothing for the prewarm to compile
    assert gemm.device_config(0) == cfg
    assert gemm._DEVICE_TILES[0] == tiles
    assert card["probes"] == []
    assert first[0] == card["want"]


def test_failed_shape_tile_stays_failed(monkeypatch, card):
    # sm120 is the arch with per-shape tiles
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (12, 0))
    card["want"] = gemm.arch_config((12, 0))
    rules = gemm.shape_tiles((12, 0))
    bad = next((r[4] for r in rules if r[4] != card["want"]), None)
    if card["want"] is None or bad is None:
        pytest.skip("no extra shape tile listed")
    card["fail"].add(bad)
    _new_process(monkeypatch)
    gemm.device_config(0)
    tiles = gemm._DEVICE_TILES[0]
    assert all(r[4] != bad for r in tiles)
    _new_process(monkeypatch)
    card["probes"].clear()
    gemm.device_config(0)
    assert card["probes"] == [] and gemm._DEVICE_TILES[0] == tiles


@pytest.mark.parametrize("field", ["libdevice", "ptxas", "prefetch_fix"])
def test_toolchain_change_reprobes(monkeypatch, card, field):
    _new_process(monkeypatch)
    gemm.device_config(0)
    card["extra"][field] = "changed"
    _new_process(monkeypatch)
    card["probes"].clear()
    assert gemm._gemm_probe_jobs(0) != []
    gemm.device_config(0)
    assert card["probes"]


def test_out_of_memory_and_failure_are_not_persisted(monkeypatch, card):
    card["oom"] = True
    _new_process(monkeypatch)
    assert gemm.device_config(0) is None
    card["oom"] = False
    _new_process(monkeypatch)
    card["probes"].clear()
    assert (
        gemm.device_config(0) == card["want"] and card["probes"]
    )  # probed again, not read back as failed
    # a plain failure of both candidates: the stock path, re-probed next time
    tmp = gemm._cfg_store()._path()
    tmp.unlink()
    card["fail"].update({card["want"], gemm._FALLBACK_CONFIG})
    _new_process(monkeypatch)
    assert gemm.device_config(0) is None
    card["fail"].clear()
    _new_process(monkeypatch)
    card["probes"].clear()
    assert gemm.device_config(0) == card["want"] and card["probes"]


def test_kill_switch_probes_every_process(monkeypatch, card):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PROBE_CACHE", "0")
    _new_process(monkeypatch)
    gemm.device_config(0)
    _new_process(monkeypatch)
    card["probes"].clear()
    gemm.device_config(0)
    assert card["probes"]
