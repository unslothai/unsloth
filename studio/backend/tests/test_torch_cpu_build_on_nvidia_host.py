# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A host whose GPUs PyTorch cannot use must not be reported as having no GPUs.

Every GPU field in utils/hardware is gated on ``torch.cuda.is_available()``, so a
managed environment that ends up with a CPU-only wheel goes silent about hardware
that is plainly still there. Two Windows users hit exactly that after an in-app
update resolved torch from PyPI (whose Windows default is ``2.11.0+cpu``) over a
``cu124`` wheel: Settings > System showed ``VRAM --`` and "No visible GPU" while
``nvidia-smi`` listed both RTX A4000s, ``UNSLOTH_PREBUILT_INFO.json`` still said
``backend cuda``, and models ran on CPU (#8473, HF discussion 87).

What these pin:
  * the nvidia-smi inventory is read even though ``DEVICE`` is CPU;
  * ``devices`` stays EMPTY regardless. That list is what the model-fit estimate
    budgets against and what the training device picker pins from, so a card torch
    cannot open must never be merged into it. The inventory travels beside it, in
    ``physical_devices`` / ``mismatch``;
  * ``CHAT_ONLY_REASON`` stops being ``"no_gpu"``, because on this host that is
    false and the advice it implies ("get a GPU") is not the fix;
  * a CPU-only wheel and an accelerator wheel whose runtime will not start stay
    distinct reasons -- one is repaired by reinstalling torch, the other by the
    driver;
  * a host that genuinely has no GPU is unaffected, and a probe that cannot answer
    (no nvidia-smi, malformed CSV, a comma inside a device name) degrades to a
    structured unavailable result rather than raising out of the endpoint.

No AMD/Windows hardware exists here, so torch and nvidia-smi are faked in the shapes
the reports describe; the Windows adapter half is exercised through its own map.
"""

from __future__ import annotations

import subprocess
import types
from types import SimpleNamespace

import pytest

import main
import utils.hardware as hardware_pkg
import utils.hardware.hardware as hw
from utils.hardware import nvidia

# nvidia-smi rows for User A's box: two A4000s, the second carrying a comma in its
# name so the rejoin the parser does is covered by a real assertion.
_TWO_A4000_ROWS = "\n".join(
    [
        "0, NVIDIA RTX A4000, 16376",
        "1, NVIDIA RTX A4000, Founders Edition, 16376",
    ]
)


def _fake_torch(vendor: str):
    """A fake ``torch``, in the vendor shapes the reports describe.

    "cpu"       -- the PyPI Windows default an unconstrained update resolves to.
    "cuda_dead" -- a cu124 wheel whose runtime refuses to initialise.
    "cuda"      -- a healthy CUDA wheel.
    """
    torch = types.ModuleType("torch")
    if vendor == "cpu":
        torch.version = SimpleNamespace(hip = None, cuda = None)
        torch.__version__ = "2.11.0+cpu"
        available = False
    elif vendor == "cuda_dead":
        torch.version = SimpleNamespace(hip = None, cuda = "12.4")
        torch.__version__ = "2.6.0+cu124"
        available = False
    else:
        torch.version = SimpleNamespace(hip = None, cuda = "12.8")
        torch.__version__ = "2.9.1+cu128"
        available = True
    torch.cuda = SimpleNamespace(
        is_available = lambda: available,
        device_count = lambda: 2 if available else 0,
    )
    return torch


def _smi(monkeypatch, stdout: str, *, returncode: int = 0):
    """Pin what nvidia-smi answers, at the subprocess boundary."""
    monkeypatch.setattr(
        nvidia.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode = returncode, stdout = stdout),
    )


@pytest.fixture(autouse = True)
def _no_inherited_visibility_mask(monkeypatch):
    """An emptied mask is a deliberate CPU pin and suppresses the whole report, so a
    runner that exports one (a GPU-partitioning CI job) would silently void these."""
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)


@pytest.fixture
def cpu_torch_on_an_nvidia_host(monkeypatch):
    """torch reports no accelerator; nvidia-smi lists two A4000s."""
    monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch("cpu"))
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    # Cached for 60s so a poll cannot spawn nvidia-smi per request; a test that did
    # not clear it would read whatever an earlier one measured on the real host.
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    _smi(monkeypatch, _TWO_A4000_ROWS)


# ========== The inventory itself ==========


def test_the_physical_probe_runs_without_a_cuda_device(cpu_torch_on_an_nvidia_host):
    inventory = hw.get_physical_gpu_inventory()

    assert inventory["available"] is True
    assert inventory["sources"] == ["nvidia-smi"]
    assert [device["name"] for device in inventory["devices"]] == [
        "NVIDIA RTX A4000",
        # Rejoined: nvidia-smi's CSV has no quoting, so a name holding a comma
        # arrives split across columns.
        "NVIDIA RTX A4000, Founders Edition",
    ]
    assert [device["memory_total_gb"] for device in inventory["devices"]] == [15.99, 15.99]
    assert {device["vendor"] for device in inventory["devices"]} == {"nvidia"}


@pytest.mark.parametrize(
    ("stdout", "returncode", "failure"),
    [
        ("", 0, None),
        ("", 9, None),
        ("nonsense\n0, only two columns\n", 0, None),
        ("", 0, FileNotFoundError("nvidia-smi")),
        ("", 0, subprocess.TimeoutExpired("nvidia-smi", 10)),
    ],
)
def test_a_probe_that_cannot_answer_returns_a_result_rather_than_raising(
    monkeypatch, stdout, returncode, failure
):
    # This runs inside GET /api/system. A probe that raised would 500 the whole
    # System tab over a diagnostic, on the host least able to afford it.
    if failure is None:
        _smi(monkeypatch, stdout, returncode = returncode)
    else:
        def _raise(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr(nvidia.subprocess, "run", _raise)
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")

    result = nvidia.get_physical_gpu_inventory()
    assert result["available"] is False
    assert result["devices"] == []

    inventory = hw.get_physical_gpu_inventory()
    assert inventory == {"available": False, "devices": [], "sources": []}


def test_the_windows_amd_adapters_are_inventoried_too(monkeypatch):
    # No vendor CLI is guaranteed on Windows AMD (amd-smi elevates a child without a
    # HIP SDK), so the DirectX registry map _rocm_windows_per_device_vram already
    # trusts is the source here as well.
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    _smi(monkeypatch, "", returncode = 9)
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda: {
            0x24CF5: {"name": "AMD Radeon RX 7900 XT", "dedicated_memory_bytes": 20 * 1024**3},
            # No dedicated-memory value: unknown capacity, which is not an empty card.
            0x14CF5: {"name": "AMD Radeon(TM) Graphics"},
        },
    )

    inventory = hw.get_physical_gpu_inventory()

    assert inventory["sources"] == ["directx-registry"]
    assert [(d["name"], d["memory_total_gb"]) for d in inventory["devices"]] == [
        ("AMD Radeon(TM) Graphics", None),
        ("AMD Radeon RX 7900 XT", 20.0),
    ]


# ========== Classifying the build ==========


def test_a_cpu_wheel_and_a_dead_cuda_wheel_are_different_reasons(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    assert hw.classify_torch_build() == "torch_cpu_build"

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda_dead"))
    assert hw.classify_torch_build() == "torch_cuda_unavailable"

    # A healthy accelerator has nothing to explain.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda"))
    assert hw.classify_torch_build() is None


def test_an_untagged_wheel_is_only_a_cpu_build_when_it_names_no_runtime(monkeypatch):
    import sys

    # The PyPI macOS/CPU wheel shape: no local version tag at all.
    untagged = _fake_torch("cpu")
    untagged.__version__ = "2.9.0"
    monkeypatch.setitem(sys.modules, "torch", untagged)
    assert hw.classify_torch_build() == "torch_cpu_build"

    # Conda builds are untagged but DO set version.cuda, so they are CUDA wheels and
    # telling that host to reinstall torch would send it the wrong way.
    conda = _fake_torch("cuda_dead")
    conda.__version__ = "2.6.0"
    monkeypatch.setitem(sys.modules, "torch", conda)
    assert hw.classify_torch_build() == "torch_cuda_unavailable"


@pytest.mark.parametrize(
    "var", ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]
)
@pytest.mark.parametrize("mask", ["", " ", "-1"])
def test_a_deliberately_emptied_mask_is_not_a_broken_install(monkeypatch, var, mask):
    # Hiding the GPUs produces exactly the shape this whole feature keys on -- torch
    # sees none, nvidia-smi sees them all -- and offering that host a repair would be
    # telling it to undo its own configuration.
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda_dead"))
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    _smi(monkeypatch, _TWO_A4000_ROWS)
    monkeypatch.setenv(var, mask)

    assert hw.classify_torch_build() is None
    assert hw._torch_gpu_mismatch_report() == {}

    # A mask that NAMES a device is a different thing: that host expects it to work.
    monkeypatch.setenv(var, "0")
    assert hw.classify_torch_build() == "torch_cuda_unavailable"


def test_a_runtime_that_raises_on_its_own_probe_counts_as_unavailable(monkeypatch):
    import sys

    hostile = _fake_torch("cuda_dead")

    def _boom():
        raise RuntimeError("CUDA driver version is insufficient for CUDA runtime version")

    hostile.cuda.is_available = _boom
    monkeypatch.setitem(sys.modules, "torch", hostile)
    assert hw.classify_torch_build() == "torch_cuda_unavailable"


# ========== The detection verdict ==========


def _detect(monkeypatch):
    """Run one detection pass with the globals restored afterwards."""
    for name, value in (
        ("DEVICE", None),
        ("CHAT_ONLY", True),
        ("CHAT_ONLY_REASON", None),
        ("CHAT_ONLY_DETAIL", None),
    ):
        monkeypatch.setattr(hw, name, value)
    with hw._DETECT_LOCK:
        return hw._detect_hardware_locked()


def test_chat_only_stops_claiming_this_host_has_no_gpu(
    monkeypatch, cpu_torch_on_an_nvidia_host
):
    device = _detect(monkeypatch)

    assert device == hw.DeviceType.CPU
    assert hw.CHAT_ONLY is True
    # The whole point: "no_gpu" here is false, and the advice it carries is useless.
    assert hw.CHAT_ONLY_REASON == "torch_cpu_build"
    assert hw.CHAT_ONLY_REASON != "no_gpu"
    assert hw.CHAT_ONLY_DETAIL == "2.11.0+cpu"


def test_a_host_that_really_has_no_gpu_still_reads_no_gpu(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    _smi(monkeypatch, "", returncode = 9)

    _detect(monkeypatch)

    assert hw.CHAT_ONLY_REASON == "no_gpu"
    assert hw.CHAT_ONLY_DETAIL is None
    # And nothing is claimed about hardware that was never found.
    assert hw._torch_gpu_mismatch_report() == {}


def test_a_healthy_cuda_host_reports_no_mismatch_at_all(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda"))
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    _smi(monkeypatch, _TWO_A4000_ROWS)

    # Absence is the signal a reader keys on, so it has to be absence, not a null.
    assert hw._torch_gpu_mismatch_report() == {}


# ========== What /api/system publishes ==========


def _system_gpu_info(monkeypatch):
    """(gpu, inference_gpu) from main, with the real visibility probe in place."""
    monkeypatch.setattr(
        hardware_pkg,
        "get_visible_gpu_utilization",
        lambda: {"available": False, "backend": "cpu", "devices": []},
    )
    monkeypatch.setattr(hardware_pkg, "get_vulkan_inference_gpu_info", lambda: None)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CPU)
    # 10s TTL on the endpoint's own cache, so a sibling test's reading would answer.
    monkeypatch.setattr(main, "_system_gpu_cache", None)
    return main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))


def test_the_system_endpoint_names_the_cards_without_offering_them(
    monkeypatch, cpu_torch_on_an_nvidia_host
):
    gpu, _inference_gpu = _system_gpu_info(monkeypatch)

    # The hard constraint. `devices` is the runtime-usable list: model fit budgets
    # against it and the training device picker pins from it, so an unusable card
    # appearing here would let a user select a GPU that cannot run anything.
    assert gpu["devices"] == []
    assert gpu["available"] is False

    assert gpu["mismatch"]["reason"] == "torch_cpu_build"
    assert gpu["mismatch"]["torch_version"] == "2.11.0+cpu"
    assert gpu["mismatch"]["physical_count"] == 2
    assert gpu["mismatch"]["sources"] == ["nvidia-smi"]
    assert [device["name"] for device in gpu["physical_devices"]] == [
        "NVIDIA RTX A4000",
        "NVIDIA RTX A4000, Founders Edition",
    ]
    # Same list, reachable through the visibility endpoint on its own.
    assert hw.get_backend_visible_gpu_info()["physical_devices"] == gpu["physical_devices"]


def test_a_cpu_host_with_no_cards_publishes_neither_field(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    _smi(monkeypatch, "", returncode = 9)

    gpu, _inference_gpu = _system_gpu_info(monkeypatch)

    assert gpu["devices"] == []
    assert "mismatch" not in gpu
    assert "physical_devices" not in gpu
