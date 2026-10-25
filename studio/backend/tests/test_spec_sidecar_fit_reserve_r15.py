# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fifteenth-round cells for the DSpark/DFlash sidecar reserve and Metal retirement.

Host-resident draft KV leaves the reserve with no ordinal mapping on a host whose cards are
all discrete; on a mixed host only through an order the child provably shares and a numeric
mask, and only for this backend's own device names. A CPU child gets no reserve and no
report, and a retirement thread that cannot start does not latch the poison flag.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
import test_spec_sidecar_fit_reserve_r7 as _r7  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402

_KV = 768 * 1024 * 1024
_TWO = ((0, 24_000, 24_000), (1, 24_000, 24_000))


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("LLAMA_ARG_DEVICE", raising = False)
    monkeypatch.delenv("LLAMA_ARG_SPLIT_MODE", raising = False)
    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)
    monkeypatch.setattr(
        _lc.LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0)
    )
    monkeypatch.setattr(_lc.LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.setattr(_lc.LlamaCppBackend, "_smi_uuid_mask_order", staticmethod(lambda: None))
    monkeypatch.setattr(
        _lc.LlamaCppBackend, "_visibility_mask_is_unmappable", staticmethod(lambda: False)
    )


def _host_kv_left_out(
    tmp_path,
    monkeypatch,
    *,
    extras,
    apu,
    listed = ("CUDA0", "CUDA1"),
):
    monkeypatch.setattr(
        _lc.LlamaCppBackend,
        "_enumerated_gpu_devices",
        classmethod(lambda cls, binary = None, env = None: list(listed)),
    )
    out = []
    for name, more in (("on", []), ("off", ["-nkvo"])):
        d = tmp_path / name
        d.mkdir()
        backend, gguf = _r2._cuda_backend(d, memory = _TWO)
        backend._arch_gate_survivors = lambda binary = None: []
        backend._mtp_draft_kv_bytes = lambda *a, **k: _KV
        backend._amd_apu_wants_unified_memory = lambda gpu_indices = None: apu is not None and (
            gpu_indices is None or apu in gpu_indices
        )
        backend._integrated_cuda_unified_memory = lambda gpu_indices = None: False
        cmd = _r7._dspark(backend, gguf, d, extra_args = [*extras, *more])
        out.append(max(_r7._margins(cmd)))
    return out[0] - out[1]


@pytest.mark.parametrize(
    "order, left_out",
    [
        # CUDA's default order: the child's CUDA1 need not be nvidia-smi's card 1.
        ("FASTEST_FIRST", False),
        ("PCI_BUS_ID", True),
    ],
)
def test_a_mixed_host_narrows_only_through_a_proven_child_order(
    tmp_path, monkeypatch, order, left_out
):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    delta = _host_kv_left_out(tmp_path, monkeypatch, extras = ["--device", "CUDA1"], apu = 0)
    assert (delta >= 760) is left_out
    if not left_out:
        assert delta == 0


def test_a_uuid_mask_is_not_read_as_physical_ids(tmp_path, monkeypatch):
    """CUDA_VISIBLE_DEVICES=GPU-<dgpu>,GPU-<apu>: the child's CUDA1 is the APU, which
    _physical_ordinals (numeric masks only) would leave as card 1, the discrete one."""
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-11111111,GPU-00000000")
    assert _host_kv_left_out(tmp_path, monkeypatch, extras = ["--device", "CUDA1"], apu = 0) == 0


def test_another_backends_device_name_is_not_read_as_cuda(tmp_path, monkeypatch):
    """A CUDA + Vulkan build lists Vulkan1 too, but it is not CUDA card 1."""
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    assert (
        _host_kv_left_out(
            tmp_path,
            monkeypatch,
            extras = ["--device", "Vulkan1"],
            apu = 0,
            listed = ("CUDA0", "CUDA1", "Vulkan0", "Vulkan1"),
        )
        == 0
    )


def test_an_all_discrete_host_needs_no_ordinal_mapping(tmp_path, monkeypatch):
    """Control: no shared card anywhere, so any card the drafter lands on is discrete,
    even under an order the child does not share."""
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    assert _host_kv_left_out(tmp_path, monkeypatch, extras = ["--device", "CUDA1"], apu = None) >= 760


@pytest.mark.parametrize("device", ["none", "cpu"])
def test_a_cpu_child_gets_no_reserve_and_no_report(tmp_path, monkeypatch, device):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    messages = []
    real_info = _lc.logger.info

    def info(msg, *args, **kwargs):
        messages.append(str(msg))
        return real_info(msg, *args, **kwargs)

    monkeypatch.setattr(_lc.logger, "info", info)
    backend, gguf = _r2._cuda_backend(tmp_path)
    cmd = _r7._dspark(backend, gguf, tmp_path, extra_args = ["--device", device])
    assert max(_r7._margins(cmd)) < 1024 + 4 * 1024
    assert not any("cannot size the sidecar" in m for m in messages)


def test_a_retirement_thread_that_cannot_start_releases_the_poison_flag(monkeypatch):
    def refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    backend = _lc.LlamaCppBackend()
    poison = {"pending": True}
    backend._retire_device_lost_server_soon(None, metal_poison = poison)
    assert poison["pending"] is False
