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
    monkeypatch.setattr(gemm, "_cfg_extra", lambda index = None: dict(extra))
    monkeypatch.setattr(gemm, "_PROBE_OOMS", [0])
    monkeypatch.setattr(gemm, "_PROBE_ERRORS", [0], raising = False)
    # raise: {cfg: how many of its next probes raise (a compile / launch error, not a mismatch)}
    state = {"probes": [], "fail": set(), "oom": False, "raise": {}}

    def probe(
        index,
        cfg,
        tma = False,
    ):
        state["probes"].append(cfg)
        if state["oom"]:
            gemm._PROBE_OOMS[0] += 1  # what _probe's except counts on an out-of-memory failure
            return False
        if state["raise"].get(cfg, 0) > 0:
            state["raise"][cfg] -= 1
            gemm._PROBE_ERRORS[0] += 1  # what _probe's except counts on any other exception
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
    tmp = gemm._cfg_store(0)._path()
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


def test_one_probe_error_never_pins_the_fallback(monkeypatch, card):
    """A one-off compile / launch error of the preferred tile is retried: it neither picks nor persists the fallback."""
    if card["want"] == gemm._FALLBACK_CONFIG:
        pytest.skip("sm89 prefers the fallback tile")
    card["raise"][card["want"]] = 1
    _new_process(monkeypatch)
    assert gemm.device_config(0) == card["want"]
    assert card["probes"][:2] == [card["want"], card["want"]]
    _new_process(monkeypatch)
    card["probes"].clear()
    assert gemm.device_config(0) == card["want"] and card["probes"] == []


def test_the_same_error_twice_is_a_stable_rejection(monkeypatch, card):
    """A tile that raises on every launch (past the card's shared memory) is rejected and that verdict persists."""
    if card["want"] == gemm._FALLBACK_CONFIG:
        pytest.skip("sm89 prefers the fallback tile")
    card["raise"][card["want"]] = 2
    _new_process(monkeypatch)
    assert gemm.device_config(0) == gemm._FALLBACK_CONFIG
    _new_process(monkeypatch)
    card["probes"].clear()
    assert gemm.device_config(0) == gemm._FALLBACK_CONFIG and card["probes"] == []


def test_one_shape_tile_error_does_not_drop_the_tile(monkeypatch, card):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (12, 0))
    card["want"] = gemm.arch_config((12, 0))
    rules = gemm.shape_tiles((12, 0))
    flaky = next((r[4] for r in rules if r[4] != card["want"]), None)
    if card["want"] is None or flaky is None:
        pytest.skip("no extra shape tile listed")
    card["raise"][flaky] = 1
    _new_process(monkeypatch)
    gemm.device_config(0)
    assert any(r[4] == flaky for r in gemm._DEVICE_TILES[0])
    _new_process(monkeypatch)
    card["probes"].clear()
    gemm.device_config(0)
    assert card["probes"] == [] and any(r[4] == flaky for r in gemm._DEVICE_TILES[0])


@pytest.fixture
def toolchain(monkeypatch, tmp_path):
    pytest.importorskip("triton.backends.nvidia.compiler")
    from triton import knobs
    from triton.backends.nvidia import compiler

    lib = tmp_path / "libdevice.10.bc"
    lib.write_bytes(b"libdevice")
    monkeypatch.setattr(knobs.nvidia, "libdevice_path", str(lib))
    monkeypatch.setattr(gemm, "_prefetch_fix_key", lambda: False)
    caps = {0: (8, 9), 1: (12, 0)}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: caps[index or 0])
    asked = []

    def ptxas(arch):
        asked.append(arch)
        return f"ptxas for sm{arch}"

    monkeypatch.setattr(compiler, "get_ptxas_version", ptxas)
    return {"asked": asked, "compiler": compiler}


def test_ptxas_is_the_probed_cards_not_the_current_ones(monkeypatch, toolchain):
    assert gemm._cfg_extra(1)["ptxas"] == "ptxas for sm120"
    assert gemm._cfg_extra(0)["ptxas"] == "ptxas for sm89"
    assert toolchain["asked"] == [120, 89]


def test_ptxas_of_a_triton_that_takes_no_arch(monkeypatch, toolchain):
    # Triton 3.2: get_ptxas_version() reads the one bundled ptxas
    monkeypatch.setattr(toolchain["compiler"], "get_ptxas_version", lambda: "ptxas 12.4")
    extra = gemm._cfg_extra(1)
    assert extra["ptxas"] == "ptxas 12.4" and extra["triton"]


def test_a_triton_without_knobs_still_keys_libdevice(monkeypatch, toolchain, tmp_path):
    """Triton 3.2 / 3.3 (torch 2.6 / 2.7) have no ``triton.knobs``: libdevice comes from TRITON_LIBDEVICE_PATH, else
    the bundled file, as their CUDAOptions read it, so the verdict still persists there (real 3.2.0 / 3.3.1 wheels:
    scripts/r19308_old_triton_check.py)."""
    import sys

    import triton

    monkeypatch.delattr(triton, "knobs", raising = False)
    monkeypatch.setitem(sys.modules, "triton.knobs", None)
    alt = tmp_path / "toolkit_libdevice.10.bc"
    alt.write_bytes(b"toolkit libdevice")
    monkeypatch.setenv("TRITON_LIBDEVICE_PATH", str(alt))
    assert gemm._libdevice_path() == str(alt)
    pinned = gemm._cfg_extra(0)["libdevice"]
    monkeypatch.delenv("TRITON_LIBDEVICE_PATH")
    bundled = gemm._libdevice_path()
    assert bundled.endswith("libdevice.10.bc") and bundled != str(alt)
    if __import__("os").path.isfile(bundled):
        assert gemm._cfg_extra(0)["libdevice"] != pinned


def test_fingerprint_reads_the_triton_windows_distribution(monkeypatch):
    import types

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: types.SimpleNamespace(uuid = "GPU-x", name = "card", major = 8, minor = 9),
    )
    monkeypatch.setattr(pc, "_driver_identity", lambda hip: "drv")
    versions = {"triton-windows": "3.2.0.post11", "torchao": None}
    monkeypatch.setattr(pc, "_version", lambda name: versions.get(name))
    # the distribution name rides along, as diffusion_probe_cache._triton_version spells it (#19319)
    assert pc.fingerprint("cuda:0")["triton"] == "triton-windows==3.2.0.post11"
    versions["triton"] = "3.3.0"
    assert pc.fingerprint("cuda:0")["triton"] == "3.3.0"
