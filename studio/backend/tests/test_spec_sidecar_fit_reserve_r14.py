# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fourteenth-round cells for the DSpark/DFlash sidecar reserve.

Host-resident draft KV leaves a card's reserve only on positive evidence that the drafter's
card is discrete: an override the build did not list, or a card Vulkan Auto filtered out,
keeps the charge, and a one-card draft pin or --split-mode none narrows to that card. The
fit warning is one verdict for DSpark and DFlash and skips drafters llama.cpp sizes itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement as _placement  # noqa: E402
import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
import test_spec_sidecar_fit_reserve_r7 as _r7  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402

_KV = 768 * 1024 * 1024


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("LLAMA_ARG_DEVICE", raising = False)
    monkeypatch.delenv("LLAMA_ARG_SPLIT_MODE", raising = False)
    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)
    monkeypatch.setattr(
        _lc.LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0)
    )


def _host_kv_left_out(
    tmp_path,
    monkeypatch,
    *,
    memory,
    vulkan,
    listed,
    extras,
    apu = None,
    **kw,
):
    """MiB the -nkvo launch's largest margin is below the KV-on-card launch's."""
    monkeypatch.setattr(
        _lc.LlamaCppBackend,
        "_enumerated_gpu_devices",
        classmethod(lambda cls, binary = None, env = None: None if listed is None else list(listed)),
    )
    out = []
    for name, more in (("on", []), ("off", ["-nkvo"])):
        d = tmp_path / name
        d.mkdir()
        backend, gguf = _r2._cuda_backend(d, memory = memory, vulkan = vulkan)
        backend._arch_gate_survivors = lambda binary = None: []
        backend._mtp_draft_kv_bytes = lambda *a, **k: _KV
        backend._amd_apu_wants_unified_memory = lambda gpu_indices = None: apu is not None and (
            gpu_indices is None or apu in gpu_indices
        )
        backend._integrated_cuda_unified_memory = lambda gpu_indices = None: False
        cmd = _r7._dspark(backend, gguf, d, extra_args = [*extras, *more], **kw)
        out.append(max(_r7._margins(cmd)))
    return out[0] - out[1]


_VK_IGPU = ((0, 24_000, 0), (1, 24_000, 24_000))
_APU_DGPU = ((0, 24_000, 24_000), (1, 24_000, 24_000))


def test_a_vulkan_igpu_auto_filtered_out_keeps_host_kv(tmp_path, monkeypatch):
    """Vulkan Auto drops iGPU 0 before the shared-card set is built, so it is unknown,
    not discrete, when --device Vulkan0 points the child back at it."""
    assert (
        _host_kv_left_out(
            tmp_path,
            monkeypatch,
            memory = _VK_IGPU,
            vulkan = True,
            listed = ["Vulkan0", "Vulkan1"],
            extras = ["--device", "Vulkan0"],
        )
        == 0
    )


def test_an_override_the_build_does_not_list_keeps_host_kv(tmp_path, monkeypatch):
    """No --list-devices answer: the selection falls back to the planned discrete card,
    which is not evidence about the device the child really uses."""
    assert (
        _host_kv_left_out(
            tmp_path,
            monkeypatch,
            memory = _VK_IGPU,
            vulkan = True,
            listed = None,
            extras = ["--device", "Vulkan0"],
        )
        == 0
    )


def test_a_one_card_draft_pin_on_the_discrete_card_leaves_host_kv_out(tmp_path, monkeypatch):
    """APU 0 plus discrete card 1, the drafter pinned to card 1: its KV under -nkvo is host
    memory that card does not hold, whatever else the target spans."""
    assert (
        _host_kv_left_out(
            tmp_path,
            monkeypatch,
            memory = _APU_DGPU,
            vulkan = False,
            listed = ["CUDA0", "CUDA1"],
            extras = ["--device-draft", "CUDA1"],
            apu = 0,
        )
        >= 760
    )


@pytest.mark.parametrize("devices, left_out", [("CUDA1,CUDA0", True), ("CUDA0,CUDA1", False)])
def test_split_mode_none_narrows_to_the_main_card(tmp_path, monkeypatch, devices, left_out):
    """--split-mode none keeps devices[main_gpu], 0 by default: the first listed card."""
    delta = _host_kv_left_out(
        tmp_path,
        monkeypatch,
        memory = _APU_DGPU,
        vulkan = False,
        listed = ["CUDA0", "CUDA1"],
        extras = ["--device", devices, "--split-mode", "none"],
        apu = 0,
    )
    assert (delta >= 760) is left_out
    if not left_out:
        assert delta == 0


# One fit warning verdict for both kinds.


def _warned(
    monkeypatch,
    tmp_path,
    kind,
    extras = (),
):
    messages = []
    real_info = _lc.logger.info

    def info(msg, *args, **kwargs):
        messages.append(str(msg))
        return real_info(msg, *args, **kwargs)

    monkeypatch.setattr(_lc.logger, "info", info)
    backend, gguf = _r2._cuda_backend(tmp_path)
    backend.probe_server_capabilities = lambda _binary = None: dict(_r2._CAPS, supports_dflash = True)
    sidecar = tmp_path / f"{kind}-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    cmd = _placement._launch(
        backend,
        gguf,
        speculative_type = kind,
        extra_args = list(extras),
        **{f"{kind}_draft_path": str(sidecar)},
    )["cmd"]
    assert f"draft-{kind}" in cmd
    return any("is not reserved" in m for m in messages)


@pytest.mark.parametrize("kind", ["dspark", "dflash"])
def test_a_self_contained_drafter_is_not_reported_unreserved(tmp_path, monkeypatch, kind):
    """A drafter carrying its own token_embd and output is measured by llama.cpp's fitter."""
    monkeypatch.setattr(_lc, "_sidecar_borrows_target_tensors", lambda path: False)
    assert _warned(monkeypatch, tmp_path, kind) is False


@pytest.mark.parametrize("kind", ["dspark", "dflash"])
def test_a_borrowing_drafter_without_a_reserve_is_reported(tmp_path, monkeypatch, kind):
    """Control: a borrowing drafter on a server without --fit-target still gets the warning."""
    monkeypatch.setattr(_r2, "_CAPS", dict(_r2._CAPS, supports_fit_target = False))
    assert _warned(monkeypatch, tmp_path, kind) is True
