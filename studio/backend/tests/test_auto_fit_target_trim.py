# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto hands an offloaded load to llama.cpp's fitter with Studio's own budget reserve as the
margin on a small card, instead of llama.cpp's 1024 MiB default, and never more than that
default anywhere."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement as placement  # noqa: E402
from core.inference import llama_cpp as lc  # noqa: E402

B = lc.LlamaCppBackend
GIB = 1024**3
CAPS = {
    "supports_fit_target": True,
    "supports_fit_ctx": True,
    "supports_kv_unified": True,
}


@pytest.fixture(autouse = True)
def _host(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    for k in (
        "CUDA_VISIBLE_DEVICES",
        "LLAMA_ARG_DEVICE",
        "LLAMA_ARG_SPLIT_MODE",
        "LLAMA_ARG_FIT_TARGET",
    ):
        monkeypatch.delenv(k, raising = False)
    monkeypatch.setattr(lc, "_metal_capable_host", lambda: False)
    monkeypatch.setattr(B, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0))
    monkeypatch.setattr(B, "_sysmem_fallback_risk", staticmethod(lambda binary = None: False))
    monkeypatch.setattr(lc, "_active_vram_fraction", lambda: lc._CTX_FIT_VRAM_FRACTION)
    # Discrete cards, answered without a CUDA device query.
    monkeypatch.setattr(B, "_integrated_cuda_gpu_ids", staticmethod(lambda: set()))
    monkeypatch.setattr(B, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: set()))
    monkeypatch.setattr(B, "_cuda_device_info", staticmethod(lambda: []))
    from utils.hardware import nvidia

    monkeypatch.setattr(nvidia, "cuda_device_info_or_torch", lambda *_a, **_k: [], raising = False)


def _fit_target(cmd):
    vals = [cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a == "--fit-target"]
    return vals[-1] if vals else None


def _launch(tmp_path, memory, model_bytes, **kw):
    backend, gguf = placement._backend(tmp_path, vulkan = False, memory = list(memory))
    backend.probe_server_capabilities = lambda _binary = None: dict(CAPS)
    backend._get_gguf_size_bytes = lambda _path: model_bytes
    return backend, placement._launch(backend, gguf, **kw)["cmd"]


@pytest.mark.parametrize(
    "total, margin",
    [
        (8188, 512),  # 8 GB: the 512 MiB floor binds
        (12282, 512),
        (16380, 512),  # 3% is 491 MiB, under the floor
        (24564, 0.03 * 24564),  # 24 GB: the 3% budget reserve
        (34133, 1024),  # where 3% reaches llama.cpp's default
        (49140, 1024),  # larger cards keep the default
        (183359, 1024),
    ],
)
def test_the_default_budget_keeps_its_own_reserve_capped_at_the_default(total, margin):
    trim = lc._auto_fit_target_trim_mib(total, lc._CTX_FIT_VRAM_FRACTION)
    assert trim <= 0.0
    assert lc._LLAMA_FIT_TARGET_DEFAULT_MIB + trim == pytest.approx(min(1024, margin), abs = 1)


@pytest.mark.parametrize("frac", [0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 1.0])
@pytest.mark.parametrize("total", [4096, 8188, 12282, 24564, 49140, 81920])
def test_the_trim_never_raises_a_margin_and_stops_at_the_budget_reserve(total, frac):
    today = lc._LLAMA_FIT_TARGET_DEFAULT_MIB + (lc._CTX_FIT_VRAM_FRACTION - frac) * total
    reserve = max((1.0 - frac) * total, lc._VRAM_FLOOR_RESERVE_MIB)
    trimmed = today + lc._auto_fit_target_trim_mib(total, frac)
    # The reserve Studio's own placement keeps on that card, or today's margin where smaller.
    assert trimmed == pytest.approx(min(today, reserve))


@pytest.mark.parametrize("total", [None, 0, -1])
def test_an_unknown_total_leaves_the_margin_alone(total):
    assert lc._auto_fit_target_trim_mib(total, lc._CTX_FIT_VRAM_FRACTION) == 0.0


def test_an_offloaded_load_on_an_8gb_card_keeps_512_mib(tmp_path):
    _b, cmd = _launch(tmp_path, [(0, 7788, 8188)], 12 * GIB)
    assert "--fit" in cmd and cmd[cmd.index("--fit") + 1] == "on"
    assert _fit_target(cmd) == "512"


def test_a_24gb_card_keeps_its_three_percent(tmp_path):
    _b, cmd = _launch(tmp_path, [(0, 24000, 24564)], 40 * GIB)
    assert _fit_target(cmd) == str(int(0.03 * 24564))


def test_a_large_card_launch_is_unchanged(tmp_path):
    _b, cmd = _launch(tmp_path, [(0, 48000, 49140)], 80 * GIB)
    assert cmd[cmd.index("--fit") + 1] == "on"
    # llama.cpp's own default, left unspelled as before.
    assert _fit_target(cmd) is None


def test_windows_sysmem_fallback_keeps_the_default_margin(tmp_path, monkeypatch):
    # WDDM spills an overrun to host RAM instead of failing, so the margin is not trimmed there.
    monkeypatch.setattr(B, "_sysmem_fallback_risk", staticmethod(lambda binary = None: True))
    _b, cmd = _launch(tmp_path, [(0, 7788, 8188)], 12 * GIB)
    assert _fit_target(cmd) is None


def test_a_lowered_budget_keeps_the_budget_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "_active_vram_fraction", lambda: 0.90)
    _b, cmd = _launch(tmp_path, [(0, 7788, 8188)], 12 * GIB)
    # 10% of the card, not 1024 + 7% of it.
    assert _fit_target(cmd) == str(int(0.10 * 8188))


def test_manual_auto_layers_keep_their_margin(tmp_path):
    _b, cmd = _launch(
        tmp_path, [(0, 7788, 8188)], 12 * GIB, gpu_memory_mode = "manual", gpu_layers = -1
    )
    assert _fit_target(cmd) == "512"


def test_a_resident_load_is_unchanged(tmp_path):
    _b, cmd = _launch(tmp_path, [(0, 7788, 8188)], 1 * GIB)
    assert "--fit-target" not in cmd
    assert not ("--fit" in cmd and cmd[cmd.index("--fit") + 1] == "on")


def test_a_broadcast_margin_holds_every_card(tmp_path):
    _b, cmd = _launch(tmp_path, [(0, 7788, 8188), (1, 24000, 24564)], 60 * GIB)
    values = [int(v) for v in str(_fit_target(cmd)).split(",")]
    # One value for both cards must keep the larger card's reserve.
    assert max(values) == int(0.03 * 24564)
    assert min(values) >= 512


@pytest.mark.parametrize("probe", ["_integrated_cuda_gpu_ids", "_rocm_unified_memory_gpu_ids"])
def test_a_lone_integrated_gpu_keeps_the_default_margin(tmp_path, monkeypatch, probe):
    # A lone Jetson or AMD APU skips multi-GPU classification; VRAM is host RAM, so no trim.
    monkeypatch.setattr(B, probe, staticmethod(lambda: {0}))
    _b, cmd = _launch(tmp_path, [(0, 15000, 16384)], 24 * GIB)
    assert cmd[cmd.index("--fit") + 1] == "on"
    assert _fit_target(cmd) is None


def test_a_picked_integrated_gpu_keeps_the_default_margin(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "_integrated_cuda_gpu_ids", staticmethod(lambda: {1}))
    _b, cmd = _launch(tmp_path, [(0, 7788, 8188), (1, 15000, 16384)], 24 * GIB, gpu_ids = [1])
    assert _fit_target(cmd) is None


def test_a_small_card_beside_a_large_one_gets_its_own_margin(tmp_path, monkeypatch):
    # A per-device list is needed to trim the small card; a broadcast gives the large default.
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setattr(B, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.setattr(B, "_SMI_MIG_INDICES", None, raising = False)
    monkeypatch.setattr(B, "_host_torch_is_rocm", staticmethod(lambda: False))
    _b, cmd = _launch(tmp_path, [(0, 7788, 8188), (1, 48000, 49140)], 70 * GIB)
    assert _fit_target(cmd) == "512,1024"


def test_equal_per_card_deltas_keep_the_default_unspelled():
    flags = B._ctx_integrity_flags(
        1, True, False, 0, 0, dict(CAPS), fit_target_delta_by_idx = {0: 0.0, 1: 0.0}
    )
    assert "--fit-target" not in flags
    flags = B._ctx_integrity_flags(
        1, True, False, 0, 0, dict(CAPS), fit_target_delta_by_idx = {0: -512.0, 1: 0.0}
    )
    assert flags[flags.index("--fit-target") + 1] == "1024"
