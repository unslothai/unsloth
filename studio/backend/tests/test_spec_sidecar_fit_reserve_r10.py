# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tenth-round cells for the DSpark/DFlash sidecar reserve.

The fit-sized report reads the launch's own argv and env, as the spawn does: device extras
a gpu_ids pin strips do not name the devices. A single-card reserve is logged on a server that takes --fit-target even when the
budget slider's own pair is left at llama.cpp's default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
import test_spec_sidecar_fit_reserve_r4 as _r4  # noqa: E402
import test_spec_sidecar_fit_reserve_r7 as _r7  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402

_GIB = 1024 * 1024 * 1024
_FLOOR = 1024 + 8 * 1024
_TWO = ((0, 24_000, 24_000), (1, 24_000, 24_000))


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.delenv("LLAMA_ARG_DEVICE", raising = False)
    monkeypatch.delenv("LLAMA_ARG_SPLIT_MODE", raising = False)
    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)


_spy_fit_sized = _r2._spy_fit_sized  # planning plus the spawn's report


def test_device_extras_a_gpu_ids_pin_strips_do_not_block_the_report(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    backend._arch_gate_survivors = lambda binary = None: []
    # Past one card, so the fitter runs across both.
    backend._get_gguf_size_bytes = lambda path: (8 if "dspark" in str(path) else 30) * _GIB
    seen = _spy_fit_sized(backend)
    cmd = _r7._dspark(
        backend, gguf, tmp_path, gpu_ids = [0, 1], extra_args = ["--device", "CUDA0,CUDA1"]
    )
    assert "--device" not in cmd
    assert len(_r7._margins(cmd)) == 2
    assert seen["fit_sized"] is True


def test_a_default_slider_single_card_reserve_is_logged(tmp_path, monkeypatch):
    backend, gguf = _r2._cuda_backend(tmp_path)
    messages = []
    real_info = _lc.logger.info

    def info(msg, *args, **kwargs):
        messages.append(str(msg))
        return real_info(msg, *args, **kwargs)

    monkeypatch.setattr(_lc.logger, "info", info)
    cmd = _r7._dspark(backend, gguf, tmp_path)
    assert _r4._margin(cmd) >= _FLOOR
    assert any("llama.cpp cannot size the sidecar, so" in m for m in messages)
