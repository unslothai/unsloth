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

import inspect
import json
import pathlib
import subprocess
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import main
import utils.hardware as hardware_pkg
import utils.hardware.hardware as hw
from utils.hardware import nvidia

# nvidia-smi rows for User A's box: two A4000s, the second carrying a comma in its name.
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


def _smi(
    monkeypatch,
    stdout: str,
    *,
    returncode: int = 0,
    raises: "type[BaseException] | None" = None,
):
    """Pin what nvidia-smi answers, at the subprocess boundary.

    The procfs count goes with it. A CLI that cannot answer falls back to
    /proc/driver/nvidia/gpus, and this suite runs on a real NVIDIA host, so a test
    simulating a machine with no cards has to simulate the kernel driver's absence too.
    The fallback's own tests set it back.
    """

    def _run(*_args, **_kwargs):
        # `raises` is how an ABSENT binary is spelled: FileNotFoundError is an answer
        # (every AMD, Intel and CPU host), while a nonzero exit is a probe that failed.
        if raises is not None:
            raise raises("nvidia-smi")
        return SimpleNamespace(returncode = returncode, stdout = stdout)

    monkeypatch.setattr(nvidia.subprocess, "run", _run)
    monkeypatch.setattr(nvidia, "_linux_nvidia_procfs_gpu_count", lambda: 0)


@pytest.fixture(autouse = True)
def _no_inherited_visibility_mask(monkeypatch):
    """An emptied mask is a deliberate CPU pin and suppresses the whole report, so a
    runner that exports one (a GPU-partitioning CI job) would silently void these."""
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)


@pytest.fixture(autouse = True)
def _no_carried_over_torch_measurement(monkeypatch):
    """The torch snapshot is cached with a TTL, so one test's fake host would answer
    for the next one. Both caches start empty here, as they do in a fresh process."""
    monkeypatch.setattr(hw, "_torch_build_snapshot_cache", None)
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)


@pytest.fixture(autouse = True)
def _no_background_inventory_refresh(monkeypatch):
    """Keep the non-blocking refresh from probing the REAL host mid-test.

    get_physical_gpu_inventory(block=False) hands a stale or cold cache to a daemon
    thread, which is the whole point on a request path, but in a test it races the
    assertions and can drop this machine's actual nvidia-smi output into the cache.
    The tests that assert on the scheduling patch Thread themselves.
    """

    class _NoThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(hw.threading, "Thread", _NoThread)
    monkeypatch.setattr(hw, "_physical_gpu_inventory_refreshing", False)


@pytest.fixture
def cpu_torch_on_an_nvidia_host(monkeypatch):
    """torch reports no accelerator; nvidia-smi lists two A4000s."""
    monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch("cpu"))
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    # Cached for 60s, so a test that did not clear it would read the real host's answer.
    _smi(monkeypatch, _TWO_A4000_ROWS)


def test_a_card_whose_capacity_is_unreported_is_still_a_card(monkeypatch):
    """ "[N/A]" for memory.total is a missing metric, not a missing GPU.

    Dropping the row took the card out of the whole inventory, and the Linux procfs
    fallback does not cover it either: that answers for a query that FAILED, not one that
    came back short. The host lost its mismatch and its repair guidance over a size.
    """
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    _smi(monkeypatch, "0, NVIDIA RTX A4000, [N/A]\n1, NVIDIA RTX A4000, 16376\n")

    inventory = hw.get_physical_gpu_inventory()

    assert [d["name"] for d in inventory["devices"]] == ["NVIDIA RTX A4000"] * 2
    assert [d["memory_total_gb"] for d in inventory["devices"]] == [None, 15.99]
    assert inventory["unknown"] is False
    assert hw._devices_that_can_establish_a_mismatch(inventory["devices"]) == inventory["devices"]


def test_the_physical_probe_runs_without_a_cuda_device(cpu_torch_on_an_nvidia_host):
    inventory = hw.get_physical_gpu_inventory()

    assert inventory["available"] is True
    assert inventory["sources"] == ["nvidia-smi"]
    assert [device["name"] for device in inventory["devices"]] == [
        "NVIDIA RTX A4000",
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
    if failure is None:
        _smi(monkeypatch, stdout, returncode = returncode)
    else:

        def _raise(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr(nvidia.subprocess, "run", _raise)
    # This suite runs on a real NVIDIA host, so pin procfs empty: the case under test is a
    # machine with no NVIDIA driver at all.
    monkeypatch.setattr(nvidia, "_linux_nvidia_procfs_gpu_count", lambda: 0)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")

    result = nvidia.get_physical_gpu_inventory()
    assert result["available"] is False
    assert result["devices"] == []

    inventory = hw.get_physical_gpu_inventory()
    assert inventory["available"] is False
    assert inventory["devices"] == []
    assert inventory["sources"] == []
    # "The driver answered and there are no cards" is not "no probe could answer". An absent
    # nvidia-smi is the exception: it is the normal state of every AMD, Intel and CPU host.
    _could_not_answer = returncode != 0 or (
        failure is not None and not isinstance(failure, FileNotFoundError)
    )
    assert inventory["unknown"] is _could_not_answer


def test_the_windows_amd_adapters_are_inventoried_too(monkeypatch):
    # No vendor CLI is guaranteed on Windows AMD, so the DirectX registry map is the source.
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    _smi(monkeypatch, "", returncode = 9)
    monkeypatch.setattr(
        hw,
        "_windows_live_adapter_names",
        lambda: ["AMD Radeon RX 7900 XT", "AMD Radeon(TM) Graphics"],
    )
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda vendor_id = hw._AMD_PCI_VENDOR_ID, **_kw: (
            {
                0x24CF5: {
                    "name": "AMD Radeon RX 7900 XT",
                    "dedicated_memory_bytes": 20 * 1024**3,
                },
                # No dedicated-memory value: unknown capacity, not an empty card.
                0x14CF5: {"name": "AMD Radeon(TM) Graphics"},
            }
            if vendor_id == hw._AMD_PCI_VENDOR_ID
            else {}
        ),
    )

    inventory = hw.get_physical_gpu_inventory()

    assert inventory["sources"] == ["directx-registry"]
    assert [(d["name"], d["memory_total_gb"]) for d in inventory["devices"]] == [
        ("AMD Radeon(TM) Graphics", None),
        ("AMD Radeon RX 7900 XT", 20.0),
    ]


def test_a_cpu_wheel_and_a_dead_cuda_wheel_are_different_reasons(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    assert hw.classify_torch_build() == "torch_cpu_build"

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda_dead"))
    assert hw.classify_torch_build() == "torch_cuda_unavailable"

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda"))
    assert hw.classify_torch_build() is None


def test_an_untagged_wheel_is_only_a_cpu_build_when_it_names_no_runtime(monkeypatch):
    import sys

    untagged = _fake_torch("cpu")
    untagged.__version__ = "2.9.0"
    monkeypatch.setitem(sys.modules, "torch", untagged)
    assert hw.classify_torch_build() == "torch_cpu_build"

    conda = _fake_torch("cuda_dead")
    conda.__version__ = "2.6.0"
    monkeypatch.setitem(sys.modules, "torch", conda)
    assert hw.classify_torch_build() == "torch_cuda_unavailable"


@pytest.mark.parametrize("var", ["CUDA_VISIBLE_DEVICES"])
@pytest.mark.parametrize("mask", ["", " ", "-1"])
def test_a_deliberately_emptied_mask_is_not_a_broken_install(monkeypatch, var, mask):
    # Hiding the GPUs produces exactly the shape this feature keys on (torch sees none,
    # nvidia-smi sees them all) without anything being broken.
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda_dead"))
    _smi(monkeypatch, _TWO_A4000_ROWS)
    monkeypatch.setenv(var, mask)

    assert hw.classify_torch_build() is None
    assert hw._torch_gpu_mismatch_report() == {}

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


def test_chat_only_stops_claiming_this_host_has_no_gpu(monkeypatch, cpu_torch_on_an_nvidia_host):
    device = _detect(monkeypatch)

    assert device == hw.DeviceType.CPU
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "torch_cpu_build"
    assert hw.CHAT_ONLY_REASON != "no_gpu"
    assert hw.CHAT_ONLY_DETAIL == "2.11.0+cpu"


def test_a_host_that_really_has_no_gpu_still_reads_no_gpu(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _smi(monkeypatch, "", returncode = 9)

    _detect(monkeypatch)

    assert hw.CHAT_ONLY_REASON == "no_gpu"
    assert hw.CHAT_ONLY_DETAIL is None
    assert hw._torch_gpu_mismatch_report() == {}


def test_a_healthy_cuda_host_reports_no_mismatch_at_all(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda"))
    _smi(monkeypatch, _TWO_A4000_ROWS)

    assert hw._torch_gpu_mismatch_report() == {}


def _system_gpu_info(monkeypatch):
    """(gpu, inference_gpu) from main, with the real visibility probe in place."""
    monkeypatch.setattr(
        hardware_pkg,
        "get_visible_gpu_utilization",
        lambda: {"available": False, "backend": "cpu", "devices": []},
    )
    monkeypatch.setattr(hardware_pkg, "get_vulkan_inference_gpu_info", lambda: None)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CPU)
    monkeypatch.setattr(main, "_system_gpu_cache", None)
    # Warm the inventory the way startup does: the mismatch report reads it WITHOUT
    # blocking, because /api/system is polled every three seconds, so a cold cache answers
    # unknown. In a running backend _detect_hardware_locked has already blocked once.
    hw.get_physical_gpu_inventory()
    hw.torch_build_snapshot()
    return main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))


def test_the_system_endpoint_names_the_cards_without_offering_them(
    monkeypatch, cpu_torch_on_an_nvidia_host
):
    gpu, _inference_gpu = _system_gpu_info(monkeypatch)

    # The hard constraint: `devices` is the runtime-usable list that model fit budgets
    # against and the training picker pins from.
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
    assert hw.get_backend_visible_gpu_info()["physical_devices"] == gpu["physical_devices"]


def test_a_cpu_host_with_no_cards_publishes_neither_field(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _smi(monkeypatch, "", returncode = 9)

    gpu, _inference_gpu = _system_gpu_info(monkeypatch)

    assert gpu["devices"] == []
    assert "mismatch" not in gpu
    assert "physical_devices" not in gpu


@pytest.mark.parametrize("local", ["cpu", "cpu.cxx11.abi", "cpu.cxx11abi", "CPU"])
def test_extended_cpu_local_tags_are_still_cpu_builds(monkeypatch, local):
    """PyTorch publishes CPU wheels whose local tag carries a suffix.

    An exact "cpu" match called those CUDA wheels whose runtime failed to start, and
    the UI then pointed the user at a driver rather than at the reinstall that is the
    actual fix.
    """
    import sys

    wheel = _fake_torch("cpu")
    wheel.__version__ = f"2.8.0+{local}"
    monkeypatch.setitem(sys.modules, "torch", wheel)
    assert hw.classify_torch_build() == "torch_cpu_build"


def test_a_cuda_local_tag_is_not_mistaken_for_a_cpu_one(monkeypatch):
    import sys

    wheel = _fake_torch("cuda_dead")
    wheel.__version__ = "2.6.0+cu124"
    monkeypatch.setitem(sys.modules, "torch", wheel)
    assert hw.classify_torch_build() == "torch_cuda_unavailable"


def test_export_and_video_stop_saying_no_accelerator_was_found(monkeypatch):
    """Those two pages render the message verbatim, so it has to match the System tab.

    Telling a two-A4000 host that no supported accelerator was found contradicts the
    inventory the same server just published, and points at hardware instead of at the
    repair.
    """
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cpu_build")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.11.0+cpu")
    monkeypatch.setattr(hw, "_has_torch", lambda: True)
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    # The verdict is re-derived against the live inventory, so the cards have to be in
    # it. Without them the honest answer really is that this host has no GPU.
    monkeypatch.setattr(
        hw, "current_chat_only_verdict", lambda: (hw.CHAT_ONLY_REASON, hw.CHAT_ONLY_DETAIL)
    )

    export = hw.export_capability()
    assert export["export_supported"] is False
    assert export["export_unsupported_reason"] == "torch_cpu_build"
    assert "2.11.0+cpu" in export["export_unsupported_message"]
    assert "No supported" not in export["export_unsupported_message"]

    video = hw.video_capability()
    assert video["video_supported"] is False
    assert video["video_unsupported_reason"] == "torch_cpu_build"
    assert "No supported accelerator" not in video["video_unsupported_message"]

    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cuda_unavailable")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.6.0+cu124")
    assert hw.export_capability()["export_unsupported_reason"] == "torch_cuda_unavailable"
    assert "driver" in hw.video_capability()["video_unsupported_message"]


def test_a_genuinely_gpu_less_host_keeps_the_old_wording(monkeypatch):
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU)
    # This suite runs on a real NVIDIA box, so without these the verdict refresh correctly
    # notices the accelerator this test is pretending not to have.
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CPU)
    monkeypatch.setattr(hw, "_torch_reports_a_usable_accelerator", lambda: False)
    monkeypatch.setattr(hw, "classify_torch_build", lambda **_kw: None)
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {"available": False, "devices": [], "unknown": False},
    )
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "no_gpu")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", None)
    monkeypatch.setattr(hw, "_has_torch", lambda: True)
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")

    assert hw.export_capability()["export_unsupported_reason"] == "no_accelerator"
    assert hw.video_capability()["video_unsupported_reason"] == "no_accelerator"


def test_nvidia_smi_is_resolved_from_the_standard_windows_locations(monkeypatch):
    """A driver install can leave nvidia-smi.exe off PATH entirely.

    The bare name then raises FileNotFoundError and the inventory comes back empty on
    exactly the host this feature exists for. setup.ps1 already falls back to these two
    paths, so the backend has to as well.
    """
    monkeypatch.setattr(nvidia.platform, "system", lambda: "Windows")
    monkeypatch.setattr(nvidia.shutil, "which", lambda _name: None)
    monkeypatch.setenv("ProgramFiles", r"C:\Program Files")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")

    nvsmi = nvidia.os.path.join(r"C:\Program Files", r"NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    monkeypatch.setattr(nvidia.os.path, "isfile", lambda p: p == nvsmi)
    assert nvidia._nvidia_smi_executable() == nvsmi

    system32 = nvidia.os.path.join(r"C:\Windows", r"System32\nvidia-smi.exe")
    monkeypatch.setattr(nvidia.os.path, "isfile", lambda p: p == system32)
    assert nvidia._nvidia_smi_executable() == system32

    monkeypatch.setattr(nvidia.os.path, "isfile", lambda _p: False)
    assert nvidia._nvidia_smi_executable() == "nvidia-smi"


def test_path_resolution_is_a_no_op_off_windows_and_when_path_has_it(monkeypatch):
    monkeypatch.setattr(nvidia.platform, "system", lambda: "Linux")
    monkeypatch.setattr(nvidia.shutil, "which", lambda _name: None)
    assert nvidia._nvidia_smi_executable() == "nvidia-smi"

    monkeypatch.setattr(nvidia.platform, "system", lambda: "Windows")
    monkeypatch.setattr(nvidia.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    assert nvidia._nvidia_smi_executable() == "/usr/bin/nvidia-smi"


@pytest.mark.parametrize("mask", ["", " ", "-1"])
@pytest.mark.parametrize("var", ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"])
def test_a_hip_mask_only_counts_where_it_can_hide_something(monkeypatch, var, mask):
    """A mask that cannot take effect must not silence the mismatch.

    Windows HIP has no ROCr layer, and this module's own visibility resolver already
    ignores ROCR_VISIBLE_DEVICES there. A stray empty one on a Windows NVIDIA host
    would otherwise restore exactly the "no GPU" verdict this feature exists to
    correct. The HIP variables likewise address AMD devices, so on an NVIDIA-only
    inventory they mask nothing.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda_dead"))
    _smi(monkeypatch, _TWO_A4000_ROWS)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    # sys.platform too, not just platform.system(): the ROCR row is gated on sys.platform
    # (hardware.py's own resolver reads it), so on a Windows runner ROCR is correctly
    # ignored and this row asserted the opposite of what that host should do.
    monkeypatch.setattr(hw.sys, "platform", "linux")
    monkeypatch.setenv(var, mask)
    # Prime the cache: the mask set reads the inventory WITHOUT blocking (it is reached
    # from the verdict /api/liveness reads), and a cold cache keeps every mask.
    hw.get_physical_gpu_inventory()

    assert hw.classify_torch_build() == "torch_cuda_unavailable"

    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {"available": True, "devices": [{"vendor": "amd"}], "sources": ["x"]},
    )
    assert hw.classify_torch_build() is None


def test_rocr_is_ignored_on_windows_even_on_an_amd_host(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda_dead"))
    monkeypatch.setattr(hw.sys, "platform", "win32")
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {"available": True, "devices": [{"vendor": "amd"}], "sources": ["x"]},
    )
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
    assert hw.classify_torch_build() == "torch_cuda_unavailable"

    monkeypatch.delenv("ROCR_VISIBLE_DEVICES")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    assert hw.classify_torch_build() is None


def test_an_inventory_that_answers_nothing_keeps_every_mask(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda_dead"))
    monkeypatch.setattr(hw.sys, "platform", "linux")
    monkeypatch.setattr(
        hw, "get_physical_gpu_inventory", lambda **_kw: {"available": False, "devices": []}
    )
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
    assert hw.classify_torch_build() is None


def test_a_deliberate_cpu_install_is_not_reported_as_broken(monkeypatch, tmp_path):
    """Pinning /cpu on a machine that has a GPU is a supported thing to do.

    No mask is empty in that case, so the classifier called the wheel the user asked
    for broken, and the UI offered a repair whose only effect would be to replace it.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.setattr(hw.sys, "prefix", str(tmp_path))

    assert hw.classify_torch_build() == "torch_cpu_build"

    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cpu")
    assert hw.classify_torch_build() is None
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY")

    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cpu/")
    assert hw.classify_torch_build() is None
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL")

    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cu124")
    assert hw.classify_torch_build() == "torch_cpu_build"
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY")

    # A recorded cpu counts only when the record says someone NAMED it: setup.ps1 selects
    # /cpu automatically on a GPU-less host and records it identically.
    manifest = tmp_path / "unsloth_install_manifest.json"
    manifest.write_text('{"schema": 1, "expected_torch_tag": "cpu"}', encoding = "utf-8")
    assert hw.classify_torch_build() == "torch_cpu_build"

    manifest.write_text(
        '{"schema": 1, "expected_torch_tag": "cpu", "expected_torch_tag_pinned": true}',
        encoding = "utf-8",
    )
    assert hw.classify_torch_build() is None

    manifest.write_text(
        '{"schema": 1, "expected_torch_tag": "cpu", "expected_torch_tag_pinned": false}',
        encoding = "utf-8",
    )
    assert (
        hw.classify_torch_build() == "torch_cpu_build"
    ), "an automatic CPU selection is not a choice to protect"

    manifest.write_text('{"schema": 1, "expected_torch_tag": "cu124"}', encoding = "utf-8")
    assert hw.classify_torch_build() == "torch_cpu_build"

    manifest.write_text('{"schema": 1}', encoding = "utf-8")
    assert hw.classify_torch_build() == "torch_cpu_build"
    manifest.write_text("{not json", encoding = "utf-8")
    assert hw.classify_torch_build() == "torch_cpu_build"


def test_a_dead_accelerator_wheel_is_unaffected_by_a_cpu_record(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda_dead"))
    monkeypatch.setattr(hw.sys, "prefix", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cpu")
    assert hw.classify_torch_build() is None  # the pin is honoured first

    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY")
    assert hw.classify_torch_build() == "torch_cuda_unavailable"


def test_linux_amd_and_intel_cards_are_inventoried_from_sysfs(monkeypatch, tmp_path):
    """The AMD/Linux shape of #8473: nvidia-smi contributes nothing.

    sysfs rather than amd-smi, because amd-smi is separate ROCm userspace and this is
    the host whose ROCm install is in question. vendor 0x1002 and a byte-valued
    mem_info_vram_total are both documented amdgpu interfaces.
    """
    drm = tmp_path / "drm"
    for name, vendor, vram in (
        ("card0", "0x1002\n", str(16 * 1024**3)),
        ("card1", "0x1002\n", "0"),  # an APU with no dedicated VRAM
        ("card2", "0x10de\n", str(1024**3)),  # NVIDIA: nvidia-smi's business, not this
        ("card4", "0x8086\n", ""),  # an Arc card: no vram total published
        ("card3", "0x1002\n", "not a number"),
    ):
        device = drm / name / "device"
        device.mkdir(parents = True)
        (device / "vendor").write_text(vendor, encoding = "utf-8")
        (device / "mem_info_vram_total").write_text(vram, encoding = "utf-8")
    (drm / "card0-DP-1").mkdir()
    (drm / "card9").mkdir()

    real_listdir = hw.os.listdir
    monkeypatch.setattr(
        hw.os,
        "listdir",
        lambda p: real_listdir(str(drm)) if p == "/sys/class/drm" else real_listdir(p),
    )
    real_join = hw.os.path.join
    monkeypatch.setattr(
        hw.os.path,
        "join",
        lambda *parts: (
            real_join(str(drm), *parts[1:]) if parts[0] == "/sys/class/drm" else real_join(*parts)
        ),
    )

    records = hw._linux_drm_sysfs_records()
    assert [r["index"] for r in records] == [0, 1, 2, 3]
    assert [r["vendor"] for r in records] == ["amd", "amd", "amd", "intel"]
    assert all(r["source"] == "sysfs-drm" for r in records)
    assert records[0]["memory_total_gb"] == 16.0
    assert records[1]["memory_total_gb"] is None
    assert records[2]["memory_total_gb"] is None
    assert records[3]["memory_total_gb"] is None


def test_the_sysfs_probe_is_silent_where_there_is_no_sysfs(monkeypatch):
    monkeypatch.setattr(hw.os, "listdir", lambda _p: (_ for _ in ()).throw(OSError("no such path")))
    assert hw._linux_drm_sysfs_records() == []


def test_a_token_authenticated_cpu_pin_is_still_a_cpu_pin(monkeypatch, tmp_path):
    """A pinned index may carry its credential in the query, which is supported.

    A raw final-segment split sees "cpu?token=..." there, so the deliberate CPU build on
    a GPU host was reported as broken and offered a repair that would replace it.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.setattr(hw.sys, "prefix", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)

    for pinned in (
        "https://download.pytorch.org/whl/cpu?token=abc/",
        "https://mirror.corp.example/whl/cpu#sha256=deadbeef",
        "https://mirror.corp.example/whl/cpu//",
        "https://user:pw@mirror.corp.example/whl/CPU",
    ):
        monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", pinned)
        assert hw.classify_torch_build() is None, pinned

    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", "https://mirror.corp.example/whl/cu128?token=abc")
    assert hw.classify_torch_build() == "torch_cpu_build"


@pytest.mark.parametrize(
    "url,leaf",
    [
        ("https://download.pytorch.org/whl/cpu?token=x/", "cpu"),
        ("https://download.pytorch.org/whl/cu128//", "cu128"),
        ("https://download.pytorch.org/whl/cpu#f", "cpu"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_the_leaf_reader_matches_the_installers(url, leaf):
    assert hw._torch_index_leaf(url) == leaf


def test_the_chat_only_verdict_follows_the_inventory(monkeypatch):
    """The inventory refreshes on a 60s TTL; the startup verdict never did.

    An eGPU attached after launch, or a driver that finished restarting, left the
    sidebar and the Export and Video pages saying no accelerator exists while
    /api/system listed the card and published a mismatch.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "no_gpu")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", None)
    monkeypatch.setattr(
        hw, "get_physical_gpu_inventory", lambda **_kw: {"devices": [{"vendor": "nvidia"}]}
    )
    # The verdict never probes torch inline; measure this fake host as detection would.
    hw.torch_build_snapshot()
    assert hw.current_chat_only_verdict() == ("torch_cpu_build", "2.11.0+cpu")

    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cpu_build")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.11.0+cpu")
    monkeypatch.setattr(hw, "get_physical_gpu_inventory", lambda **_kw: {"devices": []})
    assert hw.current_chat_only_verdict() == ("no_gpu", None)


@pytest.mark.parametrize("frozen", ["mlx_unavailable", "detection_failed", "intel_mac", None])
def test_the_other_verdicts_are_left_exactly_as_detection_set_them(monkeypatch, frozen):
    # A 60 second probe cannot change any of these, and re-deriving them would fight
    # detect_hardware() rather than follow it.
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", frozen)
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "whatever detection recorded")
    monkeypatch.setattr(
        hw, "get_physical_gpu_inventory", lambda **_kw: {"devices": [{"vendor": "nvidia"}]}
    )
    assert hw.current_chat_only_verdict() == (frozen, "whatever detection recorded")


def test_a_probe_that_raises_keeps_the_frozen_verdict(monkeypatch):
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cpu_build")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.11.0+cpu")
    monkeypatch.setattr(
        hw, "classify_torch_build", lambda **_kw: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert hw.current_chat_only_verdict() == ("torch_cpu_build", "2.11.0+cpu")


def test_export_and_video_read_the_refreshed_verdict(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "no_gpu")  # the STALE verdict
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", None)
    monkeypatch.setattr(hw, "_has_torch", lambda: True)
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hw, "get_physical_gpu_inventory", lambda **_kw: {"devices": [{"vendor": "nvidia"}]}
    )

    hw.torch_build_snapshot()
    assert hw.export_capability()["export_unsupported_reason"] == "torch_cpu_build"
    assert hw.video_capability()["video_unsupported_reason"] == "torch_cpu_build"


def test_windows_intel_adapters_are_inventoried_too(monkeypatch):
    """An Arc host whose XPU wheel was replaced has exactly this shape.

    nvidia-smi contributes nothing there either, and the registry scan was filtered to
    AMD, so the inventory came back empty and the mismatch was discarded.
    """
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    _smi(monkeypatch, "", returncode = 9)
    monkeypatch.setattr(
        hw, "_windows_live_adapter_names", lambda: ["Intel(R) Arc(TM) A770 Graphics"]
    )
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda vendor_id = hw._AMD_PCI_VENDOR_ID, **_kw: (
            {0x1: {"name": "Intel(R) Arc(TM) A770", "dedicated_memory_bytes": 16 * 1024**3}}
            if vendor_id == hw._INTEL_PCI_VENDOR_ID
            else {}
        ),
    )

    inventory = hw.get_physical_gpu_inventory()

    assert [d["vendor"] for d in inventory["devices"]] == ["intel"]
    assert inventory["devices"][0]["memory_total_gb"] == 16.0
    assert inventory["available"] is True
    assert inventory["unknown"] is True

    _smi(monkeypatch, "", returncode = 0)
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    inventory = hw.get_physical_gpu_inventory()
    assert [d["vendor"] for d in inventory["devices"]] == ["intel"]
    assert inventory["unknown"] is False


def test_a_transient_probe_failure_does_not_retire_a_settled_mismatch(monkeypatch):
    """nvidia-smi timing out is not the GPU going away.

    The aggregate probe returns a structured empty result rather than raising, so the
    refreshed verdict read it as "no cards" and handed the user the opposite advice for
    a whole cache interval.
    """
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cpu_build")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.11.0+cpu")
    monkeypatch.setattr(hw, "classify_torch_build", lambda **_kw: "torch_cpu_build")
    monkeypatch.setattr(hw, "_torch_reports_a_usable_accelerator", lambda: False)
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {"available": False, "devices": [], "unknown": True},
    )
    hw.torch_build_snapshot()
    assert hw.current_chat_only_verdict() == ("torch_cpu_build", "2.11.0+cpu")

    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {"available": False, "devices": [], "unknown": False},
    )
    assert hw.current_chat_only_verdict() == ("no_gpu", None)


def test_only_uncertainty_about_the_mismatched_vendor_holds_the_verdict(monkeypatch):
    """A broken probe for a vendor that never had a card cannot pin the old mismatch.

    Detach an AMD eGPU: sysfs conclusively reports no AMD card, while nvidia-smi is broken
    on a host that never had an NVIDIA one. The carry-forward has nothing to re-add, so
    holding "your GPU is unusable" here asserts a card is present with nothing to point at,
    and it holds for as long as that unrelated probe stays broken.
    """
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cpu_build")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.11.0+cpu")
    monkeypatch.setattr(hw, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"amd"}))
    monkeypatch.setattr(hw, "classify_torch_build", lambda **_kw: "torch_cpu_build")
    monkeypatch.setattr(hw, "_torch_reports_a_usable_accelerator", lambda: False)
    hw.torch_build_snapshot()

    def _inventory(unanswered):
        monkeypatch.setattr(
            hw,
            "get_physical_gpu_inventory",
            lambda **_kw: {
                "available": False,
                "devices": [],
                "unknown": True,
                "unanswered": unanswered,
            },
        )

    _inventory(["nvidia"])
    assert hw.current_chat_only_verdict() == ("no_gpu", None)

    # The vendor the mismatch DID come from, so the card may still be there and unread.
    _inventory(["amd"])
    assert hw.current_chat_only_verdict() == ("torch_cpu_build", "2.11.0+cpu")

    # An unknown that names nobody is a cold cache, which is silent about who could not
    # answer rather than saying everyone did.
    _inventory([])
    assert hw.current_chat_only_verdict() == ("torch_cpu_build", "2.11.0+cpu")


def test_the_recorded_vendors_follow_the_mismatch_when_it_moves(monkeypatch):
    """The mismatch can change vendor inside one process: swap an AMD eGPU for an NVIDIA one.

    Holding the startup answer would later freeze the verdict on a vendor a refresh has
    already watched disappear, which is the same stale-scope bug one level up.
    """
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cpu_build")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.11.0+cpu")
    monkeypatch.setattr(hw, "CHAT_ONLY_MISMATCH_VENDORS", frozenset({"amd"}))
    monkeypatch.setattr(hw, "classify_torch_build", lambda **_kw: "torch_cpu_build")
    monkeypatch.setattr(hw, "_torch_reports_a_usable_accelerator", lambda: False)
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {
            "available": True,
            "devices": [{"vendor": "nvidia", "name": "NVIDIA RTX A4000"}],
            "unknown": False,
        },
    )
    hw.torch_build_snapshot()

    # Only the reason: the detail is re-read from the live torch, which is this host's.
    assert hw.current_chat_only_verdict()[0] == "torch_cpu_build"
    assert hw.CHAT_ONLY_MISMATCH_VENDORS == frozenset({"nvidia"})

    # And the AMD probe going unreadable now proves nothing about the card that is here.
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {
            "available": False,
            "devices": [],
            "unknown": True,
            "unanswered": ["amd"],
        },
    )
    assert hw.current_chat_only_verdict() == ("no_gpu", None)


def test_the_health_path_never_waits_on_the_gpu_probe(monkeypatch):
    """/api/health and /api/liveness both read the chat-only verdict.

    The NVIDIA half shells out with a 10 second timeout and the lock makes concurrent
    callers queue, so a hung driver -- the exact host this feature is for -- would stall
    the event loop every time the TTL expired, against a 2 second desktop timeout.
    """
    import sys

    calls = {"blocking": 0, "threads": 0}

    def _probe():
        calls["blocking"] += 1
        return {
            "available": True,
            "devices": [{"vendor": "nvidia"}],
            "sources": ["x"],
            "unknown": False,
        }

    class _Thread:
        def __init__(self, *a, **k):
            calls["threads"] += 1

        def start(self):
            pass

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.setattr(hw, "_probe_physical_gpu_inventory", _probe)
    monkeypatch.setattr(hw.threading, "Thread", _Thread)
    monkeypatch.setattr(hw, "_physical_gpu_inventory_refreshing", False)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cpu_build")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.11.0+cpu")

    hw.torch_build_snapshot()
    # That warm-up IS the blocking pass, standing in for detection. What the request path
    # may not do is shell out again, so count from here.
    calls["blocking"] = 0
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    monkeypatch.setattr(hw, "_physical_gpu_inventory_refreshing", False)

    # Cold inventory cache: no probe inline, the refresh goes to a thread, and the explicit
    # unknown keeps the frozen verdict.
    assert hw.current_chat_only_verdict() == ("torch_cpu_build", "2.11.0+cpu")
    assert calls["blocking"] == 0, "nothing may shell out on the request path"
    assert calls["threads"] >= 1

    monkeypatch.setattr(hw, "_physical_gpu_inventory_refreshing", True)
    before = calls["threads"]
    hw.get_physical_gpu_inventory(block = False)
    assert calls["threads"] == before


def test_a_stale_cache_is_served_rather_than_re_probed_off_the_request_path(monkeypatch):
    warm = {
        "available": True,
        "devices": [{"vendor": "nvidia"}],
        "sources": ["x"],
        "unknown": False,
    }
    calls = {"n": 0}

    def _probe():
        calls["n"] += 1
        return warm

    class _Thread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

    monkeypatch.setattr(hw, "_probe_physical_gpu_inventory", _probe)
    monkeypatch.setattr(hw.threading, "Thread", _Thread)
    monkeypatch.setattr(hw, "_physical_gpu_inventory_refreshing", False)
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", (hw.time.monotonic() - 3600, warm))

    assert hw.get_physical_gpu_inventory(block = False) is warm
    assert calls["n"] == 0, "a stale answer beats a subprocess on the request path"

    assert hw.get_physical_gpu_inventory() is warm
    assert calls["n"] == 1


def test_a_process_that_cannot_start_a_thread_keeps_the_stale_answer(monkeypatch):
    warm = {
        "available": True,
        "devices": [{"vendor": "nvidia"}],
        "sources": ["x"],
        "unknown": False,
    }
    monkeypatch.setattr(hw, "_physical_gpu_inventory_refreshing", False)
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", (hw.time.monotonic() - 3600, warm))

    def _boom(*_a, **_k):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(hw.threading, "Thread", _boom)
    assert hw.get_physical_gpu_inventory(block = False) is warm
    assert hw._physical_gpu_inventory_refreshing is False


def test_a_stale_registry_record_is_not_reported_as_a_gpu(monkeypatch):
    """The DirectX registry outlives the hardware.

    setup.ps1 says so and uses these records only to RE-LABEL an adapter its live WMI
    scan also returned, never to add one. A CPU-only machine with a driver record left
    behind would otherwise be told it has an unusable GPU and offered a repair that
    cannot restore absent hardware.
    """
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    _smi(monkeypatch, "", returncode = 9)
    monkeypatch.setattr(hw, "_windows_live_adapter_names", lambda: ["Microsoft Basic Display"])
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda vendor_id = hw._AMD_PCI_VENDOR_ID, **_kw: (
            {0x1: {"name": "AMD Radeon RX 6800", "dedicated_memory_bytes": 16 * 1024**3}}
            if vendor_id == hw._AMD_PCI_VENDOR_ID
            else {}
        ),
    )

    inventory = hw.get_physical_gpu_inventory()
    assert inventory["devices"] == []
    assert inventory["sources"] == []


def test_a_live_scan_that_cannot_answer_reports_unknown_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    _smi(monkeypatch, "", returncode = 9)
    monkeypatch.setattr(hw, "_windows_live_adapter_names", lambda: None)
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda vendor_id = hw._AMD_PCI_VENDOR_ID, **_kw: (
            {0x1: {"name": "AMD Radeon RX 6800"}} if vendor_id == hw._AMD_PCI_VENDOR_ID else {}
        ),
    )

    inventory = hw.get_physical_gpu_inventory()
    assert inventory["devices"] == []
    assert inventory["unknown"] is True


@pytest.mark.parametrize(
    "registry,live,matched",
    [
        ("AMD Radeon RX 7900 XT", "AMD Radeon RX 7900 XT", True),
        ("AMD Radeon RX 7900 XT", "AMD Radeon RX 7900 XT Graphics", True),
        ("Intel(R) Arc(TM) A770 Graphics", "Intel(R) Arc(TM) A770", True),
        ("AMD Radeon RX 6800", "NVIDIA GeForce RTX 4090", False),
        ("", "AMD Radeon RX 6800", False),
        ("AMD Radeon RX 6800", "", False),
    ],
)
def test_the_registry_to_live_join_matches_setup_ps1(registry, live, matched):
    assert hw._adapter_name_is_live(registry, [live]) is matched


def test_an_ordinary_intel_igpu_does_not_establish_a_mismatch(monkeypatch, tmp_path):
    """setup.sh does not autodetect Linux XPU, and setup.ps1 limits it to Arc.

    An Intel UHD iGPU beside a CPU wheel is the correct state of that machine, so
    counting it would report a broken install and offer a repair that reinstalls the
    very CPU build it just replaced.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.setattr(hw.sys, "prefix", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)

    igpu = [{"vendor": "intel", "name": None, "index": 0}]
    assert hw._devices_that_can_establish_a_mismatch(igpu) == []

    arc = [{"vendor": "intel", "name": "Intel(R) Arc(TM) A770 Graphics", "index": 0}]
    assert hw._devices_that_can_establish_a_mismatch(arc) == arc
    others = [
        {"vendor": "nvidia", "index": 0},
        {"vendor": "amd", "index": 0, "gfx_candidates": ["gfx1100"]},
    ]
    assert hw._devices_that_can_establish_a_mismatch(others) == others


def test_a_nameless_intel_card_counts_once_xpu_was_actually_chosen(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.setattr(hw.sys, "prefix", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    nameless = [{"vendor": "intel", "name": None, "index": 0}]

    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "xpu")
    assert hw._devices_that_can_establish_a_mismatch(nameless) == nameless
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY")

    manifest = tmp_path / "unsloth_install_manifest.json"
    manifest.write_text('{"schema": 1, "expected_torch_tag": "xpu"}', encoding = "utf-8")
    assert hw._devices_that_can_establish_a_mismatch(nameless) == nameless

    manifest.write_text('{"schema": 1, "expected_torch_tag": "cpu"}', encoding = "utf-8")
    assert hw._devices_that_can_establish_a_mismatch(nameless) == []

    xpu_torch = _fake_torch("cpu")
    xpu_torch.__version__ = "2.9.0+xpu"
    monkeypatch.setitem(sys.modules, "torch", xpu_torch)
    assert hw._devices_that_can_establish_a_mismatch(nameless) == nameless


def test_an_accelerator_that_came_back_retires_the_cached_verdict(monkeypatch):
    """Only reason and detail are refreshed here; DEVICE and CHAT_ONLY are not.

    So a driver that finished restarting left Train and Export disabled AND the UI
    saying there is no GPU, which is worse than the stale mismatch it replaced.
    """
    import sys

    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cuda_unavailable")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.6.0+cu124")
    monkeypatch.setattr(hw, "_REDETECTION_REQUESTED", False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda"))
    calls = {"n": 0}
    monkeypatch.setattr(hw, "invalidate_detection", lambda: calls.__setitem__("n", calls["n"] + 1))
    # The recovery is measured, not probed inline. The recovery path drops the snapshot
    # again, so the fresh pass re-reads this fake host rather than the pre-recovery answer.
    hw.torch_build_snapshot()

    assert hw.current_chat_only_verdict() == ("torch_cuda_unavailable", "2.6.0+cu124")
    assert calls["n"] == 1

    hw.current_chat_only_verdict()
    hw.current_chat_only_verdict()
    assert calls["n"] == 1


def test_a_host_whose_accelerator_never_came_back_is_not_re_detected(monkeypatch):
    import sys

    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cpu_build")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.11.0+cpu")
    monkeypatch.setattr(hw, "_REDETECTION_REQUESTED", False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.setattr(
        hw, "get_physical_gpu_inventory", lambda **_kw: {"devices": [{"vendor": "nvidia"}]}
    )
    calls = {"n": 0}
    monkeypatch.setattr(hw, "invalidate_detection", lambda: calls.__setitem__("n", calls["n"] + 1))

    assert hw.current_chat_only_verdict() == ("torch_cpu_build", "2.11.0+cpu")
    assert calls["n"] == 0


def test_a_healthy_xpu_wheel_is_not_called_unavailable(monkeypatch):
    """An Intel host that started while its runtime was down and later recovered.

    Asking only about CUDA classified a perfectly healthy +xpu wheel as
    torch_cuda_unavailable forever, so the recovery branch was never reached and the
    process stayed chat-only until restart.
    """
    import sys

    xpu = _fake_torch("cuda_dead")
    xpu.__version__ = "2.9.0+xpu"
    xpu.xpu = types.SimpleNamespace(is_available = lambda: True)
    monkeypatch.setitem(sys.modules, "torch", xpu)
    assert hw.classify_torch_build() is None

    xpu.xpu = types.SimpleNamespace(is_available = lambda: False)
    assert hw.classify_torch_build() == "torch_cuda_unavailable"

    def _boom():
        raise RuntimeError("Level Zero not initialised")

    xpu.xpu = types.SimpleNamespace(is_available = _boom)
    assert hw.classify_torch_build() == "torch_cuda_unavailable"


@pytest.mark.parametrize("mask", ["", " ", "-1"])
def test_an_emptied_xpu_mask_is_a_deliberate_hide(monkeypatch, mask):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda_dead"))
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {"available": True, "devices": [{"vendor": "intel"}]},
    )
    monkeypatch.setenv("ZE_AFFINITY_MASK", mask)
    assert hw.classify_torch_build() is None

    monkeypatch.setenv("ZE_AFFINITY_MASK", "0")
    assert hw.classify_torch_build() == "torch_cuda_unavailable"


def test_the_xpu_mask_is_ignored_on_an_nvidia_only_host(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda_dead"))
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {"available": True, "devices": [{"vendor": "nvidia"}]},
    )
    monkeypatch.setenv("ZE_AFFINITY_MASK", "")
    assert hw.classify_torch_build() == "torch_cuda_unavailable"


def test_duplicate_registry_records_are_claimed_one_to_one(monkeypatch):
    """Two identical cards leave two identical registry records.

    Remove one and WMI reports a single live adapter, which a reusable predicate matched
    to BOTH: a GPU reported that is gone, and its VRAM counted twice.
    """
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    # nvidia-smi ANSWERS (absent), so only the registry vendors go unanswered here.
    _smi(monkeypatch, "", raises = FileNotFoundError)
    monkeypatch.setattr(hw, "_windows_live_adapter_names", lambda: ["AMD Radeon RX 7900 XT"])
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda vendor_id = hw._AMD_PCI_VENDOR_ID, **_kw: (
            {
                0x1: {"name": "AMD Radeon RX 7900 XT", "dedicated_memory_bytes": 20 * 1024**3},
                0x2: {"name": "AMD Radeon RX 7900 XT", "dedicated_memory_bytes": 20 * 1024**3},
            }
            if vendor_id == hw._AMD_PCI_VENDOR_ID
            else {}
        ),
    )

    inventory = hw.get_physical_gpu_inventory()
    assert len(inventory["devices"]) == 1, "one live adapter corroborates one record"
    assert inventory["devices"][0]["memory_total_gb"] == 20.0


def test_both_records_survive_when_both_cards_are_live(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    _smi(monkeypatch, "", returncode = 9)
    monkeypatch.setattr(
        hw,
        "_windows_live_adapter_names",
        lambda: ["AMD Radeon RX 7900 XT", "AMD Radeon RX 7900 XT"],
    )
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda vendor_id = hw._AMD_PCI_VENDOR_ID, **_kw: (
            {
                0x1: {"name": "AMD Radeon RX 7900 XT"},
                0x2: {"name": "AMD Radeon RX 7900 XT"},
            }
            if vendor_id == hw._AMD_PCI_VENDOR_ID
            else {}
        ),
    )

    assert len(hw.get_physical_gpu_inventory()["devices"]) == 2


def test_the_live_scan_stops_on_a_cim_failure_rather_than_reporting_none(monkeypatch):
    """-ErrorAction SilentlyContinue exits 0 with empty stdout.

    That is indistinguishable from "no adapters", so the corroboration would have
    dropped every real card on a host with damaged WMI, recreating the false no-GPU
    result this whole change exists to remove.
    """
    source = inspect.getsource(hw._windows_live_adapter_names)
    command = "".join(line for line in source.splitlines() if not line.strip().startswith("#"))
    assert "SilentlyContinue" not in command
    assert "$ErrorActionPreference='Stop'" in command


def test_a_missing_nvidia_smi_does_not_warn_every_refresh(monkeypatch, capsys):
    """The normal state of every CPU-only, AMD and Intel host.

    The inventory calls this on a 60 second refresh reached from the health and system
    polls, so a warning here is a line a minute on a machine that is working correctly.
    """

    def _missing(*_a, **_k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(nvidia.subprocess, "run", _missing)
    capsys.readouterr()
    assert (
        nvidia._query_gpu_inventory("test") is nvidia.NVIDIA_SMI_ABSENT
    ), "an absent CLI is its own answer, not the None that means a probe failed"
    assert '"level": "warning"' not in capsys.readouterr().out

    def _hang(*_a, **_k):
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(nvidia.subprocess, "run", _hang)
    capsys.readouterr()
    assert nvidia._query_gpu_inventory("test") is None
    assert '"level": "warning"' in capsys.readouterr().out


def test_the_recovery_actually_starts_a_detection_pass(monkeypatch):
    """Retiring the epoch alone did nothing, so the round-six fix never took effect.

    invalidate_detection leaves DEVICE set and DETECTION_COMPLETE raised, so /api/health
    kept reading the settled snapshot and start_background_detection returned at once
    because DEVICE was not None. The process stayed chat-only until restart, which is
    exactly what the fix claimed to solve.
    """
    import sys

    calls = []
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cuda_unavailable")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.6.0+cu124")
    monkeypatch.setattr(hw, "_REDETECTION_REQUESTED", False)
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda"))
    monkeypatch.setattr(hw, "invalidate_detection", lambda: calls.append("epoch") or 1)
    monkeypatch.setattr(hw, "_discard_detection_locked", lambda: calls.append("discard"))
    monkeypatch.setattr(hw, "start_background_detection", lambda: calls.append("start"))

    # detect_hardware() warms the snapshot on the real path; the verdict only ever reads it
    # non-blocking, so measure this fake host the same way first.
    hw.torch_build_snapshot()
    hw.current_chat_only_verdict()

    # Order matters: discarding after the pass started would race it, and starting before
    # the discard is the no-op this fixes.
    assert calls == ["epoch", "discard", "start"]


def test_the_system_mismatch_report_does_not_block(monkeypatch):
    # GET /api/system is polled every three seconds with _system_gpu_cache_lock held for the
    # whole call, so a hung nvidia-smi would queue every concurrent read behind it.
    source = inspect.getsource(hw._torch_gpu_mismatch_report)
    assert "get_physical_gpu_inventory(block=False)" in source.replace(" ", "")


def test_an_unimportable_torch_still_reports_the_cards(monkeypatch, tmp_path):
    """A CUDA wheel whose native runtime will not load.

    Returning None meant the mismatch report never ran, so a GPU host published no
    physical_devices and the System tab said no visible GPU while nvidia-smi could
    enumerate the card: the central failure, for the runtime-failure case.
    """
    import importlib.abc
    import importlib.util
    import sys

    # A real import that raises, not an object in sys.modules whose attribute access does.
    # Reading __spec__ off an existing sys.modules entry is import-machinery behaviour that
    # changed in 3.13: on 3.10-3.12 `import torch` hands back the stub untouched, so the
    # premise below silently inverted and this test passed for the wrong reason.
    class _WillNotLoad(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            raise OSError("[WinError 126] cudart64_12.dll could not be loaded")

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(
            self,
            fullname,
            path = None,
            target = None,
        ):
            if fullname == "torch":
                return importlib.util.spec_from_loader("torch", _WillNotLoad())
            return None

    monkeypatch.delitem(sys.modules, "torch", raising = False)
    monkeypatch.setattr(sys, "meta_path", [_Finder(), *sys.meta_path])
    # _has_torch() is NOT forced here: it reports False for a wheel that will not import,
    # and the early return on it used to keep this host from the on-disk fallback below.
    assert hw._has_torch() is False, "the premise: an unimportable torch reads as absent"

    monkeypatch.setattr(hw, "_installed_torch_label_on_disk", lambda: "2.6.0+cu124")
    assert hw.classify_torch_build() == "torch_cuda_unavailable"

    for label in ("2.11.0+rocm7.2", "2.9.1+xpu"):
        monkeypatch.setattr(hw, "_installed_torch_label_on_disk", lambda label = label: label)
        assert hw.classify_torch_build() == "torch_cuda_unavailable"

    monkeypatch.setattr(hw, "_installed_torch_label_on_disk", lambda: "2.11.0+cpu")
    assert hw.classify_torch_build() == "torch_cpu_build"

    monkeypatch.setattr(hw, "_installed_torch_label_on_disk", lambda: "")
    assert hw.classify_torch_build() is None


def test_the_disk_label_reader_needs_no_interpreter(tmp_path):
    pkg = tmp_path / "torch"
    pkg.mkdir()
    (pkg / "version.py").write_text(
        '__version__ = "2.6.0+cu124"\ncuda = "12.4"\n', encoding = "utf-8"
    )
    with patch.object(
        hw.importlib.util,
        "find_spec",
        return_value = SimpleNamespace(submodule_search_locations = [str(pkg)]),
    ):
        assert hw._installed_torch_label_on_disk() == "2.6.0+cu124"

    with patch.object(hw.importlib.util, "find_spec", return_value = None):
        assert hw._installed_torch_label_on_disk() == ""

    with patch.object(hw.importlib.util, "find_spec", side_effect = ValueError("boom")):
        assert hw._installed_torch_label_on_disk() == ""


def test_an_untagged_xpu_build_is_not_called_a_cpu_wheel(monkeypatch):
    """torch.version.xpu is the marker, and an untagged wheel can carry it.

    An Arc host whose Level Zero runtime is down has no local version tag and neither
    a cuda nor a hip one, so it read as a CPU wheel and the UI offered a reinstall,
    which is the one remedy that cannot help. _torch_reports_an_xpu_runtime() already
    knew better, so the card established a mismatch and then got the wrong advice.
    """
    import sys

    torch = types.ModuleType("torch")
    torch.__version__ = "2.9.1"  # untagged, as a source or vendor build is
    torch.version = SimpleNamespace(cuda = None, hip = None, xpu = "20250101")
    torch.cuda = SimpleNamespace(is_available = lambda: False)
    torch.xpu = SimpleNamespace(is_available = lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert hw.classify_torch_build() == "torch_cuda_unavailable"

    torch.version = SimpleNamespace(cuda = None, hip = None, xpu = None)
    assert hw.classify_torch_build() == "torch_cpu_build"


def test_the_health_path_never_imports_torch_inline(monkeypatch):
    """classify_torch_build() imports torch and asks CUDA and XPU if they are available.

    Both block while a driver is wedged or restarting, which is the state
    torch_cuda_unavailable names, and /api/health and /api/liveness reach them through
    the chat-only verdict against a two second desktop timeout. Making only the
    inventory lookup non-blocking still left this on the request thread.
    """
    calls = {"probes": 0, "threads": 0}

    def _classify(**_kw):
        calls["probes"] += 1
        return "torch_cpu_build"

    class _Thread:
        def __init__(self, *a, **k):
            calls["threads"] += 1

        def start(self):
            pass

    monkeypatch.setattr(hw, "classify_torch_build", _classify)
    monkeypatch.setattr(hw, "_torch_reports_a_usable_accelerator", lambda: False)
    monkeypatch.setattr(hw.threading, "Thread", _Thread)
    monkeypatch.setattr(hw, "_torch_build_snapshot_refreshing", False)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cpu_build")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.11.0+cpu")
    monkeypatch.setattr(
        hw, "get_physical_gpu_inventory", lambda **_kw: {"devices": [], "unknown": True}
    )

    assert hw.current_chat_only_verdict() == ("torch_cpu_build", "2.11.0+cpu")
    assert calls["probes"] == 0, "no torch probe may run on the health path"
    assert calls["threads"] == 1

    hw.torch_build_snapshot()
    assert calls["probes"] == 1
    assert hw.current_chat_only_verdict() == ("torch_cpu_build", "2.11.0+cpu")
    assert calls["probes"] == 1


def test_a_recovery_re_measures_torch_rather_than_reusing_the_old_answer(monkeypatch):
    """The snapshot that said the accelerator was missing must not outlive it.

    Left in place, the fresh detection pass the recovery starts would read the same
    stale measurement for the rest of its TTL and settle on the verdict it just retired.
    """
    import sys

    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cuda_unavailable")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.6.0+cu124")
    monkeypatch.setattr(hw, "_REDETECTION_REQUESTED", False)
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda"))
    monkeypatch.setattr(hw, "invalidate_detection", lambda: 1)
    monkeypatch.setattr(hw, "_discard_detection_locked", lambda: None)
    monkeypatch.setattr(hw, "start_background_detection", lambda: None)

    hw.torch_build_snapshot()
    assert hw._torch_build_snapshot_cache is not None
    hw.current_chat_only_verdict()
    assert (
        hw._torch_build_snapshot_cache is None
    ), "the recovery must drop the measurement it was taken before"


def test_a_vendor_mask_does_not_hide_another_vendors_card(monkeypatch):
    """A hybrid NVIDIA + Arc host with ZE_AFFINITY_MASK="".

    The mask hides the Arc and nothing else, but it was cancelling the whole
    classification, so a CPU-only wheel went unreported for the NVIDIA card the user
    never masked. The Arc still has to drop out of the mismatch inventory: it is
    hidden on purpose, and counting it would offer a repair for a chosen configuration.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.setenv("ZE_AFFINITY_MASK", "")
    hybrid = [
        {"vendor": "nvidia", "name": "NVIDIA RTX A4000"},
        {"vendor": "intel", "name": "Intel(R) Arc(TM) A770"},
    ]
    monkeypatch.setattr(
        hw, "get_physical_gpu_inventory", lambda **_kw: {"devices": hybrid, "unknown": False}
    )

    assert hw.classify_torch_build() == "torch_cpu_build"
    kept = hw._devices_that_can_establish_a_mismatch(hybrid)
    assert [d["vendor"] for d in kept] == ["nvidia"]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert hw.classify_torch_build() is None
    assert hw._devices_that_can_establish_a_mismatch(hybrid) == []


def test_an_intel_only_host_still_reads_an_emptied_ze_mask_as_deliberate(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.setenv("ZE_AFFINITY_MASK", "-1")
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {"devices": [{"vendor": "intel", "name": "Arc A770"}], "unknown": False},
    )
    assert hw.classify_torch_build() is None


def test_an_untagged_conda_cuda_build_that_will_not_import_is_not_a_cpu_wheel(monkeypatch):
    """torch/version.py records the runtime even when the wheel carries no local tag.

    The importable path already reads exactly these attributes, so reading only
    __version__ on the failure path gave the same installation the opposite diagnosis
    and sent the user to reinstall a GPU wheel it already has.
    """
    monkeypatch.setattr(hw, "_installed_torch_label_on_disk", lambda: "2.9.1")
    monkeypatch.setattr(
        hw, "_installed_torch_markers_on_disk", lambda: {"cuda": "12.8", "hip": None, "xpu": None}
    )
    assert hw._classification_from_disk_label() == "torch_cuda_unavailable"

    for marker in ("hip", "xpu"):
        monkeypatch.setattr(
            hw,
            "_installed_torch_markers_on_disk",
            lambda marker = marker: {"cuda": None, "hip": None, "xpu": None} | {marker: "1.0"},
        )
        assert hw._classification_from_disk_label() == "torch_cuda_unavailable"

    monkeypatch.setattr(
        hw, "_installed_torch_markers_on_disk", lambda: {"cuda": None, "hip": None, "xpu": None}
    )
    assert hw._classification_from_disk_label() == "torch_cpu_build"

    monkeypatch.setattr(hw, "_installed_torch_label_on_disk", lambda: "")
    assert hw._classification_from_disk_label() is None


def test_the_marker_reader_parses_both_shapes_torch_has_shipped(tmp_path):
    """Parsed, not executed: this runs when importing torch is the thing that fails."""
    package = tmp_path / "torch"
    package.mkdir()
    (package / "version.py").write_text(
        "from typing import Optional\n"
        "__version__ = '2.9.1'\n"
        "debug = False\n"
        "cuda: Optional[str] = '12.8'\n"
        "hip = None\n"
        "xpu: Optional[str] = None\n",
        encoding = "utf-8",
    )

    class _Spec:
        submodule_search_locations = [str(package)]

    import types as _types

    original = hw.importlib.util.find_spec
    hw.importlib.util.find_spec = lambda name: _Spec() if name == "torch" else original(name)
    try:
        assert hw._installed_torch_markers_on_disk() == {
            "cuda": "12.8",
            "hip": None,
            "xpu": None,
        }
    finally:
        hw.importlib.util.find_spec = original
    assert isinstance(_types, _types.ModuleType)


def test_an_unimportable_gpu_wheel_is_a_mismatch_rather_than_a_detection_failure():
    """detection_failed sends the user to the server log; the mismatch offers the repair.

    _has_torch() is False for a wheel whose runtime will not load, so detection took
    the TORCH_IMPORT_ERROR arm and never reached the classification -- and the verdict
    refresh deliberately freezes detection_failed, so nothing revisited it later.
    """
    import pathlib

    source = pathlib.Path(hw.__file__).read_text(encoding = "utf-8")
    branch = source[source.index("elif TORCH_IMPORT_ERROR is not None:") :]
    branch = branch[: branch.index('elif platform.system() == "Darwin":')]
    assert (
        "_classification_from_disk_label()" in branch
    ), "the broken-runtime host must still be classified from the wheel on disk"
    assert "torch_build_snapshot()" not in branch, (
        "and classified WITHOUT probing: importing torch is what failed here, it takes "
        "seconds to fail on a real broken wheel, and a retry re-runs torch/__init__ "
        "against the partial module tree the first attempt left behind"
    )
    assert branch.index("detection_failed") < branch.index("_mismatch_verdict_for_this_host"), (
        "detection_failed stays the default; the mismatch only replaces it when the "
        "inventory actually found a card"
    )


def test_the_mismatch_verdict_names_the_wheel_it_could_not_import(monkeypatch):
    monkeypatch.setattr(
        hw,
        "torch_build_snapshot",
        lambda **_kw: {
            "reason": "torch_cuda_unavailable",
            "usable": False,
            "unknown": False,
        },
    )
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {"devices": [{"vendor": "nvidia", "name": "A4000"}], "unknown": False},
    )
    monkeypatch.setattr(hw, "_torch_version_label", lambda: None)
    monkeypatch.setattr(hw, "_installed_torch_label_on_disk", lambda: "2.6.0+cu124")

    assert hw._mismatch_verdict_for_this_host() == ("torch_cuda_unavailable", "2.6.0+cu124")

    monkeypatch.setattr(hw, "get_physical_gpu_inventory", lambda **_kw: {"devices": []})
    assert hw._mismatch_verdict_for_this_host() == (None, None)


def test_cuda_visible_devices_masks_an_amd_host_too(monkeypatch):
    """HIP honours CUDA_VISIBLE_DEVICES alongside its own variables.

    This module's own visibility resolver reads all three together on an AMD host, so
    mapping the variable to NVIDIA alone had an AMD-only box launched with
    CUDA_VISIBLE_DEVICES="" reported as broken and offered a repair for a mask its
    owner set on purpose.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    amd = [{"vendor": "amd", "name": "Radeon RX 7900 XT", "gfx_candidates": ["gfx1100"]}]
    monkeypatch.setattr(
        hw, "get_physical_gpu_inventory", lambda **_kw: {"devices": amd, "unknown": False}
    )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert hw.classify_torch_build() is None
    assert hw._devices_that_can_establish_a_mismatch(amd) == []

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert hw.classify_torch_build() == "torch_cpu_build"
    assert hw._devices_that_can_establish_a_mismatch(amd) == amd


def test_the_health_path_never_retries_a_failed_torch_import(monkeypatch):
    """The label was read with _torch_version_label(), which imports torch.

    On the broken-runtime host that import is what fails: it can take seconds, and
    _has_torch() purges the partial module afterwards, so each call genuinely re-runs
    the native load. /api/health, /api/liveness and GET /api/system all reach it.
    """
    calls = {"n": 0}

    def _label():
        calls["n"] += 1
        return "should not be reached"

    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", "OSError('libcudart.so.12')")
    monkeypatch.setattr(hw, "_torch_version_label", _label)
    monkeypatch.setattr(hw, "_installed_torch_label_on_disk", lambda: "2.6.0+cu124")
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cuda_unavailable")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.6.0+cu124")
    monkeypatch.setattr(hw, "classify_torch_build", lambda **_kw: "torch_cuda_unavailable")
    monkeypatch.setattr(hw, "_torch_reports_a_usable_accelerator", lambda: False)
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {
            "devices": [{"vendor": "nvidia", "name": "A4000"}],
            "sources": ["nvidia-smi"],
            "unknown": False,
        },
    )
    hw.torch_build_snapshot()

    assert hw.current_chat_only_verdict() == ("torch_cuda_unavailable", "2.6.0+cu124")
    report = hw._torch_gpu_mismatch_report()
    assert report["mismatch"]["torch_version"] == "2.6.0+cu124"
    assert calls["n"] == 0, "the failed import must not be retried on a request path"

    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(hw, "_torch_version_label", lambda: "2.11.0+cpu")
    assert hw._reported_torch_label() == "2.11.0+cpu"


def test_an_amd_card_the_installers_decline_is_not_a_broken_install(monkeypatch):
    """setup.sh ships a ROCm wheel only for RDNA 2 and newer.

    A Polaris or RDNA 1 card is left on CPU torch on purpose, and the DRM sysfs walk
    reports it as vendor amd like any other, so it was being counted as evidence of a
    broken install and offered a repair that cannot make that GPU usable.
    """
    monkeypatch.setattr(hw, "_expected_rocm_flavor_was_chosen", lambda: False)
    monkeypatch.setattr(hw, "_torch_reports_a_hip_runtime", lambda: False)

    polaris = [{"vendor": "amd", "gfx_candidates": ["gfx803"]}]
    assert hw._devices_that_can_establish_a_mismatch(polaris) == []

    supported = [{"vendor": "amd", "gfx_candidates": ["gfx1100"]}]
    assert hw._devices_that_can_establish_a_mismatch(supported) == supported

    # A host with no ROCm userspace names no arch at all, and setup.sh detects AMD only
    # through rocminfo and amd-smi, so that machine was never going to get a ROCm wheel.
    unnamed = [{"vendor": "amd"}]
    assert hw._devices_that_can_establish_a_mismatch(unnamed) == []

    assert hw._devices_that_can_establish_a_mismatch([{"vendor": "amd", "gfx": "gfx1201"}])
    assert hw._devices_that_can_establish_a_mismatch([{"vendor": "amd", "gfx": "gfx900"}]) == []


def test_a_rocm_expectation_outranks_the_arch_table(monkeypatch):
    """If this install asked for a ROCm wheel, its absence is a fault either way."""
    monkeypatch.setattr(hw, "_torch_reports_a_hip_runtime", lambda: False)
    monkeypatch.setattr(hw, "_expected_rocm_flavor_was_chosen", lambda: True)
    old = [{"vendor": "amd", "gfx_candidates": ["gfx803"]}]
    assert hw._devices_that_can_establish_a_mismatch(old) == old

    monkeypatch.setattr(hw, "_expected_rocm_flavor_was_chosen", lambda: False)
    monkeypatch.setattr(hw, "_torch_reports_a_hip_runtime", lambda: True)
    assert hw._devices_that_can_establish_a_mismatch(old) == old


def test_the_supported_arch_set_matches_the_installer(monkeypatch):
    """One table in two places drifts, and it drifts BOTH ways.

    install.sh's _amd_arch_index_family_for_gfx is the map that decides whether this
    stack ships a ROCm wheel for a card (it mirrors install.ps1's $archFamilyMap). A gfx
    it lists but the backend does not is a supported card reported as no_gpu with no
    repair; one the backend lists but it does not is a repair that cannot help.
    """
    import pathlib
    import re

    root = pathlib.Path(hw.__file__).resolve().parents[4]
    install_sh = (root / "install.sh").read_text(encoding = "utf-8")
    block = install_sh[install_sh.index("_amd_arch_index_family_for_gfx()") :]
    block = block[: block.index("esac")]
    # Only the case LABELS: the values on the right are index families (gfx103X-all), whose
    # prefixes a bare regex over the block would collect.
    shipped = {
        gfx
        for line in block.splitlines()
        for gfx in re.findall(r"gfx[0-9a-f]+", line.split(")")[0])
    }
    assert len(shipped) > 10, "the installer's arch map moved"

    extra = {"gfx906"}
    assert set(hw._ROCM_SUPPORTED_GFX) == shipped | extra, (
        "the backend must accept exactly the architectures the installers ship a wheel "
        f"for; installer has {sorted(shipped | extra)}"
    )
    stack = (root / "studio" / "install_python_stack.py").read_text(encoding = "utf-8")
    assert "gfx906" in stack, "the gfx906 path this set carries has gone"


def test_the_gfx_probe_answers_nothing_without_a_rocm_userspace(monkeypatch):
    """No rocminfo and no amd-smi is the common case on the host in question."""

    def _missing(*_a, **_k):
        raise FileNotFoundError("rocminfo")

    monkeypatch.setattr(hw.subprocess, "run", _missing)
    assert hw._linux_amd_gfx_candidates() == []

    class _Result:
        stdout = "  Name:                    gfx1030\n  Name:                    gfx1030\n"

    monkeypatch.setattr(hw.subprocess, "run", lambda *_a, **_k: _Result())
    assert hw._linux_amd_gfx_candidates() == ["gfx1030"]


def test_a_cold_start_measures_the_inventory_before_honouring_a_mask(monkeypatch):
    """An irrelevant empty mask must not decide the verdict from a cold cache.

    HIP_VISIBLE_DEVICES="" on an NVIDIA-only host is not a statement about the NVIDIA
    card, but on the first pass the non-blocking read answers unknown, every mask counts
    as relevant, and the classification was suppressed and cached as "torch is fine" for
    a whole TTL -- withholding the repair from a host nvidia-smi describes a moment
    later.
    """
    import sys

    probes = {"n": 0}

    def _probe():
        probes["n"] += 1
        return {
            "available": True,
            "devices": [{"vendor": "nvidia", "name": "A4000"}],
            "sources": ["nvidia-smi"],
            "unknown": False,
        }

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    monkeypatch.setattr(hw, "_probe_physical_gpu_inventory", _probe)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")

    snapshot = hw.torch_build_snapshot()

    assert probes["n"] >= 1, "the cached pass must measure rather than read a cold cache"
    assert (
        snapshot["reason"] == "torch_cpu_build"
    ), "a HIP mask says nothing about an NVIDIA card and must not suppress the repair"


def test_an_absent_nvidia_smi_is_an_answer_not_a_failed_probe(monkeypatch):
    """An AMD-only host has no nvidia-smi by design.

    Marking its inventory unknown kept a settled mismatch alive for good once the AMD
    card that established it was detached: the sysfs walk correctly found nothing, and
    the verdict refresh read the unknown as "the probe declined" and preserved the old
    reason, while /api/system had already dropped the device rows.
    """

    def _missing(*_a, **_k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(nvidia.subprocess, "run", _missing)
    monkeypatch.setattr(nvidia, "_linux_nvidia_procfs_gpu_count", lambda: 0)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_linux_drm_sysfs_records", lambda **_kw: [])

    inventory = hw.get_physical_gpu_inventory()
    assert inventory["devices"] == []
    assert inventory["unknown"] is False, (
        "no nvidia-smi is the normal state of an AMD or CPU-only host, not a probe "
        "that could not answer"
    )

    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cpu_build")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.11.0+cpu")
    monkeypatch.setattr(hw, "classify_torch_build", lambda **_kw: "torch_cpu_build")
    monkeypatch.setattr(hw, "_torch_reports_a_usable_accelerator", lambda: False)
    hw.torch_build_snapshot()
    assert hw.current_chat_only_verdict() == ("no_gpu", None)

    def _hang(*_a, **_k):
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(nvidia.subprocess, "run", _hang)
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    assert hw.get_physical_gpu_inventory()["unknown"] is True


def test_a_second_recovery_can_still_ask_for_a_pass(monkeypatch):
    """The guard is one request per recovery, not one per process.

    It was cleared only by detect_hardware(), while the recovery starts its pass through
    start_background_detection(), which runs ensure_hardware_detected(). So the first
    recovery raised the guard for good: a driver that flapped, or any later lifecycle
    that published an inventory-sensitive CPU verdict, could never get another pass.
    """
    import sys

    monkeypatch.setattr(hw, "_REDETECTION_REQUESTED", False)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cuda_unavailable")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.6.0+cu124")
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda"))
    starts = {"n": 0}
    monkeypatch.setattr(hw, "invalidate_detection", lambda: 1)
    monkeypatch.setattr(hw, "_discard_detection_locked", lambda: None)
    monkeypatch.setattr(
        hw, "start_background_detection", lambda: starts.__setitem__("n", starts["n"] + 1)
    )

    hw.torch_build_snapshot()
    hw.current_chat_only_verdict()
    assert starts["n"] == 1
    assert hw._REDETECTION_REQUESTED is True, "one request per recovery, while it runs"

    source = _hardware_source()
    ensure = source[source.index("def ensure_hardware_detected(") :]
    ensure = ensure[: ensure.index("def _detect_hardware_locked(")]
    assert (
        "_REDETECTION_REQUESTED = False" in ensure
    ), "a settled background pass must release the guard, or there is never a second one"
    assert ensure.index("DETECTION_COMPLETE.set()") < ensure.index(
        "_REDETECTION_REQUESTED = False"
    ), "released only once the pass has actually published"


def test_one_capability_response_describes_one_host(monkeypatch):
    """Three reads of a verdict that refreshes on a TTL can disagree.

    An eGPU attached or removed mid-response could give the reason from one host and the
    message from another, or a reason with no message at all.
    """
    verdicts = [
        ("torch_cpu_build", "2.11.0+cpu"),
        ("no_gpu", None),
        (None, None),
    ]
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CPU)
    monkeypatch.setattr(hw, "_has_torch", lambda: True)
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hw, "current_chat_only_verdict", lambda: verdicts.pop(0) if verdicts else (None, None)
    )

    export = hw.export_capability()
    assert export["export_unsupported_reason"] == "torch_cpu_build"
    assert "CPU-only build" in export["export_unsupported_message"]
    assert len(verdicts) == 2, "the response must be built from ONE reading of the verdict"


def _hardware_source() -> str:
    import pathlib
    return pathlib.Path(hw.__file__).resolve().read_text(encoding = "utf-8")


def test_the_hardware_module_loads_without_the_rest_of_the_package(tmp_path):
    """tests/python/test_e2e_no_torch_sandbox.py executes this module on its own.

    It builds a minimal stub tree -- loggers, structlog, utils/hardware -- and nothing
    else, so a top-level import of anything further inside utils makes the module
    unloadable there, which is how a Windows-only console-hiding helper broke hardware
    detection on a host with no torch at all.
    """
    import pathlib
    import re as _re

    source = pathlib.Path(hw.__file__).resolve().read_text(encoding = "utf-8")
    header = source[: source.index("logger = get_logger(__name__)")]
    imports = _re.findall(r"^\s*(?:from|import)\s+([\w.]+)", header, _re.MULTILINE)
    for module in imports:
        if module == "utils.hardware" or module.startswith("utils.hardware."):
            continue
        assert not module.startswith("utils"), (
            f"{module} is imported at module scope; the no-torch sandbox stubs only "
            "utils.hardware, so import it inside the function that needs it"
        )


def test_a_driver_without_the_cli_still_reports_its_cards(monkeypatch):
    """/proc/driver/nvidia/gpus is published whatever nvidia-smi's state is.

    install_python_stack._has_usable_nvidia_gpu() falls back to it for the same reason,
    so without this the installer could select or repair a CUDA wheel on a host where
    the backend insisted there was no card, and the user got no_gpu with no repair.
    """

    def _missing(*_a, **_k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(nvidia.subprocess, "run", _missing)
    monkeypatch.setattr(nvidia, "_linux_nvidia_procfs_gpu_count", lambda: 2)

    result = nvidia.get_physical_gpu_inventory()
    assert result["available"] is True
    assert result["absent"] is False
    assert [d["vendor"] for d in result["devices"]] == ["nvidia", "nvidia"]
    assert [d["name"] for d in result["devices"]] == [None, None]
    assert [d["memory_total_gb"] for d in result["devices"]] == [None, None]
    assert result["source"] == "proc-driver-nvidia"


def test_a_kfd_confirmed_amd_card_stays_eligible(monkeypatch):
    """A minimal AMD host with /dev/kfd but no rocminfo or amd-smi.

    install_python_stack._has_rocm_gpu() uses the KFD topology as its fallback exactly
    so `studio update` can repair CPU-only torch there, so a card the installer would
    repair cannot be dropped from the evidence here.
    """
    monkeypatch.setattr(hw, "_expected_rocm_flavor_was_chosen", lambda: False)
    monkeypatch.setattr(hw, "_torch_reports_a_hip_runtime", lambda: False)
    unnamed = [{"vendor": "amd"}]

    monkeypatch.setattr(hw, "_linux_kfd_reports_an_amd_gpu", lambda: True)
    assert hw._devices_that_can_establish_a_mismatch(unnamed) == unnamed

    monkeypatch.setattr(hw, "_linux_kfd_reports_an_amd_gpu", lambda: False)
    assert hw._devices_that_can_establish_a_mismatch(unnamed) == []

    monkeypatch.setattr(hw, "_linux_kfd_reports_an_amd_gpu", lambda: True)
    assert (
        hw._devices_that_can_establish_a_mismatch([{"vendor": "amd", "gfx_candidates": ["gfx803"]}])
        == []
    )


def test_the_kfd_probe_rejects_a_non_amd_node(monkeypatch, tmp_path):
    """The NVIDIA open kernel module registers KFD nodes of its own.

    Their vendor_id is 4318, not AMD's 4098, and the installer guards on exactly that,
    so an NVIDIA-only host must not read as AMD here either.
    """
    nodes = tmp_path / "nodes"
    for name, gpu_id, vendor in (("0", "0", "4098"), ("1", "5555", "4318")):
        node = nodes / name
        node.mkdir(parents = True)
        (node / "gpu_id").write_text(gpu_id, encoding = "utf-8")
        (node / "properties").write_text(f"vendor_id {vendor}\n", encoding = "utf-8")

    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    real_listdir, real_open = hw.os.listdir, open
    monkeypatch.setattr(
        hw.os,
        "listdir",
        lambda path: real_listdir(nodes) if "kfd" in str(path) else real_listdir(path),
    )
    monkeypatch.setattr(
        hw.os.path,
        "join",
        lambda *parts: str(nodes.joinpath(*parts[1:])) if "kfd" in parts[0] else "/".join(parts),
    )
    assert (
        hw._linux_kfd_reports_an_amd_gpu() is False
    ), "a CPU node and an NVIDIA-owned node are not an AMD GPU"

    (nodes / "1" / "properties").write_text("vendor_id 4098\n", encoding = "utf-8")
    assert hw._linux_kfd_reports_an_amd_gpu() is True
    assert real_open is open


def test_the_installer_records_who_named_the_flavor(tmp_path, monkeypatch):
    """The manifest has to carry the provenance, or the backend cannot ask.

    setup.ps1 publishes UNSLOTH_EXPECTED_TORCH_TAG for an automatic /cpu choice exactly
    as it does for a pinned one, so the handover variable alone is not evidence.
    """
    import importlib.util
    import pathlib

    root = pathlib.Path(hw.__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "_install_manifest_probe", root / "install_manifest.py"
    )
    manifest_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manifest_mod)

    monkeypatch.setattr(manifest_mod, "venv_root", lambda: tmp_path)
    monkeypatch.setattr(manifest_mod, "manifest_path", lambda root = None: tmp_path / "m.json")
    monkeypatch.setattr(manifest_mod, "requirement_digests", lambda *_a, **_k: {})
    monkeypatch.setattr(manifest_mod, "installed_requirements_root", lambda *_a, **_k: None)
    monkeypatch.setattr(manifest_mod, "_installed_version", lambda *_a, **_k: "0")

    import json as _json

    manifest_mod.write_manifest(expected_torch_tag = "cpu", expected_torch_tag_pinned = False)
    written = _json.loads((tmp_path / "m.json").read_text(encoding = "utf-8"))
    assert written["expected_torch_tag"] == "cpu"
    assert written["expected_torch_tag_pinned"] is False

    manifest_mod.write_manifest(expected_torch_tag = "cpu", expected_torch_tag_pinned = True)
    written = _json.loads((tmp_path / "m.json").read_text(encoding = "utf-8"))
    assert written["expected_torch_tag_pinned"] is True

    manifest_mod.write_manifest(expected_torch_tag = "cpu")
    written = _json.loads((tmp_path / "m.json").read_text(encoding = "utf-8"))
    assert "expected_torch_tag_pinned" not in written
    monkeypatch.setattr(manifest_mod, "read_manifest", lambda root = None: written)
    assert manifest_mod.recorded_torch_flavor_was_pinned() is False


def test_a_vendor_that_did_not_answer_keeps_the_inventory_unknown(monkeypatch):
    """A hybrid Intel iGPU plus NVIDIA dGPU host with nvidia-smi timing out.

    The Intel row cancelled the unknown, and _devices_that_can_establish_a_mismatch then
    discarded the iGPU as ineligible, so the verdict saw neither an eligible card nor an
    unanswered probe and downgraded a settled mismatch to no_gpu for a cache interval --
    hiding the repair while the NVIDIA probe was merely unavailable.
    """
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hw,
        "_linux_drm_sysfs_records",
        lambda **_kw: [{"vendor": "intel", "name": None, "index": 0}],
    )

    def _timeout(*_a, **_k):
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(nvidia.subprocess, "run", _timeout)
    monkeypatch.setattr(nvidia, "_linux_nvidia_procfs_gpu_count", lambda: 0)

    inventory = hw.get_physical_gpu_inventory()
    assert [d["vendor"] for d in inventory["devices"]] == ["intel"]
    assert inventory["unknown"] is True

    _smi(monkeypatch, "0, NVIDIA RTX A4000, 16376\n")
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    assert hw.get_physical_gpu_inventory()["unknown"] is False


def test_a_stale_registry_record_cannot_claim_a_longer_named_live_card(monkeypatch):
    """ "RX 7900 XT" is a prefix of a live "RX 7900 XTX".

    Records are walked in LUID order and consume the first live name they match, so the
    stale XT could claim the XTX that is really installed and the inventory would
    publish the removed card's name and its VRAM.
    """
    live = ["AMD Radeon RX 7900 XTX"]
    assert hw._claim_live_adapter("AMD Radeon RX 7900 XTX", live) == 0

    both = ["AMD Radeon RX 7900 XTX", "AMD Radeon RX 7900 XT"]
    assert both[hw._claim_live_adapter("AMD Radeon RX 7900 XT", both)] == "AMD Radeon RX 7900 XT"
    assert both[hw._claim_live_adapter("AMD Radeon RX 7900 XTX", both)] == "AMD Radeon RX 7900 XTX"

    # The prefix rule still has to work: the registry description and the WMI display name
    # spell the same card differently, and one is routinely a prefix of the other.
    assert hw._claim_live_adapter("AMD Radeon RX 7900 XTX", ["AMD Radeon RX 7900 XTX 24GB"]) == 0
    assert hw._claim_live_adapter("Something Else", live) is None


def test_a_broken_nvidia_smi_still_reports_the_kernel_driver_cards(monkeypatch):
    """Absent is not the only way the CLI fails to answer.

    A nvidia-smi that hangs past its timeout, or exits non-zero, leaves the kernel
    driver enumerating cards regardless. On a cold start there is no settled verdict for
    the resulting unknown to protect, so the host was reported as having no GPU at all,
    with no repair, for as long as the CLI stayed broken.
    """
    monkeypatch.setattr(nvidia, "_linux_nvidia_procfs_gpu_count", lambda: 1)

    def _hang(*_a, **_k):
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(nvidia.subprocess, "run", _hang)
    result = nvidia.get_physical_gpu_inventory()
    assert [d["vendor"] for d in result["devices"]] == ["nvidia"]
    assert result["source"] == "proc-driver-nvidia"

    monkeypatch.setattr(
        nvidia.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode = 9, stdout = ""),
    )
    assert nvidia.get_physical_gpu_inventory()["available"] is True

    monkeypatch.setattr(nvidia, "_linux_nvidia_procfs_gpu_count", lambda: 0)
    assert nvidia.get_physical_gpu_inventory()["error"] is not None


def test_hip_visible_devices_outranks_the_cuda_alias(monkeypatch):
    """HIP reads its own variables first; the alias is only a fallback.

    An AMD host that NAMES its devices in HIP_VISIBLE_DEVICES while inheriting an empty
    CUDA_VISIBLE_DEVICES has not hidden anything, and _get_parent_visible_gpu_spec in
    this same module already applies that precedence.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    amd = [{"vendor": "amd", "gfx_candidates": ["gfx1100"]}]
    monkeypatch.setattr(
        hw, "get_physical_gpu_inventory", lambda **_kw: {"devices": amd, "unknown": False}
    )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    assert hw.classify_torch_build() == "torch_cpu_build"
    assert hw._devices_that_can_establish_a_mismatch(amd) == amd

    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    assert hw.classify_torch_build() is None
    assert hw._devices_that_can_establish_a_mismatch(amd) == []

    monkeypatch.delenv("HIP_VISIBLE_DEVICES")
    assert hw.classify_torch_build() is None


def test_only_the_highest_priority_mask_that_is_set_decides(monkeypatch):
    """A lower-priority variable naming devices does not un-hide them.

    HIP stops at the first of the three that is SET, so HIP_VISIBLE_DEVICES=-1 hides
    every AMD card however ROCR_VISIBLE_DEVICES is spelled. Reading the ROCR value there
    would report a mismatch, and offer a repair, for a host masked exactly as asked.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cpu"))
    amd = [{"vendor": "amd", "gfx_candidates": ["gfx1100"]}]
    monkeypatch.setattr(
        hw, "get_physical_gpu_inventory", lambda **_kw: {"devices": amd, "unknown": False}
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)

    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    assert hw.classify_torch_build() is None
    assert hw._devices_that_can_establish_a_mismatch(amd) == []

    # The other direction, so this is precedence rather than "an emptied one always wins":
    # ROCR only decides when HIP is unset, and then a named device is a real expectation.
    monkeypatch.delenv("HIP_VISIBLE_DEVICES")
    assert hw.classify_torch_build() == "torch_cpu_build"
    assert hw._devices_that_can_establish_a_mismatch(amd) == amd


# =============================================== a refresh that cannot answer is not news


def test_an_unanswerable_refresh_keeps_the_cards_it_already_found(monkeypatch):
    """nvidia-smi timing out must not read as "the GPUs were removed".

    Overwriting the cache with the empty failure holds for a whole TTL, and the two
    consumers then disagree inside one response: current_chat_only_verdict() keeps its
    frozen mismatch on ``unknown`` while _torch_gpu_mismatch_report() reads the devices,
    finds none, and drops physical_devices and mismatch from /api/system.
    """
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    _smi(monkeypatch, _TWO_A4000_ROWS)
    good = hw.get_physical_gpu_inventory()
    assert len(good["devices"]) == 2 and good["unknown"] is False

    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", (0.0, good))
    _smi(monkeypatch, "", returncode = 9)
    after = hw.get_physical_gpu_inventory()

    assert [d["name"] for d in after["devices"]] == [d["name"] for d in good["devices"]]
    assert after["available"] is True
    # Still unknown: these rows describe the host as it was, not as this pass measured it.
    assert after["unknown"] is True
    assert after["unanswered"] == ["nvidia"]


def test_a_vendor_that_answered_none_is_allowed_to_lose_its_cards(monkeypatch):
    """The carry-forward is per unanswered vendor only, so a real removal still lands."""
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _smi(monkeypatch, _TWO_A4000_ROWS)
    good = hw.get_physical_gpu_inventory()

    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", (0.0, good))
    _smi(monkeypatch, "", raises = FileNotFoundError)
    after = hw.get_physical_gpu_inventory()

    assert after["devices"] == []
    assert after["unknown"] is False


def test_a_registry_that_cannot_be_read_is_not_a_host_without_adapters(monkeypatch):
    """`{}` from the DirectX helper covers both, so the inventory has to ask which."""
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    # nvidia-smi ANSWERS (absent), so only the registry vendors go unanswered here.
    _smi(monkeypatch, "", raises = FileNotFoundError)
    monkeypatch.setattr(hw, "_windows_live_adapter_names", lambda: ["AMD Radeon RX 7900 XT"])
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda vendor_id = hw._AMD_PCI_VENDOR_ID, **kw: (
            None if kw.get("distinguish_failure") else {}
        ),
    )

    inventory = hw.get_physical_gpu_inventory()

    assert inventory["devices"] == []
    assert inventory["unknown"] is True
    assert inventory["unanswered"] == ["amd", "intel"]


def test_the_ranking_callers_still_see_an_empty_map(monkeypatch):
    """Only the inventory asked for the distinction; nothing else changed shape."""
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_windows_amd_adapter_records_or_none", lambda *a, **k: None)
    assert hw._windows_amd_adapter_records_by_luid() == {}
    assert hw._windows_amd_adapter_records_by_luid(distinguish_failure = True) is None


# ============================================ a Windows AMD card the registry did not name


@pytest.mark.parametrize(
    "name,eligible",
    [
        ("AMD Radeon RX 7900 XT", True),
        ("AMD Radeon RX 9070 XT", True),
        ("AMD Radeon 780M Graphics", True),
        # Polaris and RDNA 1: no wheel family covers them, so a repair could not change
        # anything and this host is on CPU torch on purpose.
        ("AMD Radeon RX 580", False),
        ("AMD Radeon RX 5700 XT", False),
        ("", False),
    ],
)
def test_a_windows_adapter_with_no_adapter_family_is_read_from_its_name(
    monkeypatch, name, eligible
):
    """AdapterFamily is written by the driver, so the reported RX 7900 XT arrives bare.

    Falling through to the KFD probe discards it: that probe is Linux-only and returns
    False on Windows, which made Windows accidentally stricter than Linux for the exact
    card this feature was written for.
    """
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_linux_kfd_reports_an_amd_gpu", lambda: False)
    device = {"vendor": "amd", "name": name, "source": "directx-registry"}
    assert hw._amd_device_can_establish_a_mismatch(device) is eligible


def test_a_named_arch_still_wins_over_the_marketing_name(monkeypatch):
    """The name table only fills a gap; it never overrides what the driver reported."""
    monkeypatch.setattr(hw, "_linux_kfd_reports_an_amd_gpu", lambda: True)
    assert (
        hw._amd_device_can_establish_a_mismatch(
            {"vendor": "amd", "name": "AMD Radeon RX 7900 XT", "gfx": "gfx803"}
        )
        is False
    )


# ================================================== only a real ROCm family is a ROCm pin


@pytest.mark.parametrize(
    "leaf,chosen",
    [
        ("rocm6.4", True),
        ("rocm7", True),
        ("gfx1151", True),
        ("gfx120X-all", True),
        # Suffixed leaves are custom pins the installer routes verbatim. Reading one as ROCm
        # waives the supported-architecture filter and calls a deliberate install broken.
        ("rocm-rel-7.2.1", False),
        ("rocm7.2-private", False),
        ("gfx-mirror", False),
        ("cpu", False),
    ],
)
def test_only_the_installers_rocm_families_count_as_a_rocm_choice(monkeypatch, leaf, chosen):
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", f"https://example.invalid/whl/{leaf}")
    monkeypatch.setattr(hw.sys, "prefix", "/nonexistent-prefix-for-this-test")
    assert hw._expected_rocm_flavor_was_chosen() is chosen


def test_the_backend_and_the_installer_agree_on_the_family_predicate():
    """One vocabulary, two files: drift here is a silent behaviour split."""
    for leaf in (
        "rocm6.4",
        "rocm7",
        "gfx1151",
        "gfx120X-all",
        "rocm-rel-7.2.1",
        "rocm7.2-private",
        "gfx-mirror",
        "cpu",
        "cu124",
        "",
    ):
        assert hw._is_pip_rocm_family_leaf(leaf) == _installer_rocm_family(leaf), leaf


def _installer_rocm_family(leaf: str) -> bool:
    """install_python_stack._is_pip_rocm_family_leaf, loaded without importing the module."""
    import ast
    import pathlib
    import re as _re

    source = (pathlib.Path(__file__).resolve().parents[2] / "install_python_stack.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(source)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_is_pip_rocm_family_leaf"
    )
    namespace: dict = {"re": _re}
    exec(compile(ast.Module(body = [fn], type_ignores = []), "<installer>", "exec"), namespace)
    return namespace["_is_pip_rocm_family_leaf"](leaf)


# ======================================= a detection failure the disk already explained


def test_a_disk_classified_detection_failure_transitions_when_the_probe_recovers(monkeypatch):
    """torch will not import AND the OS probe timed out: only the inventory was missing.

    detect_hardware() classifies the wheel from disk for exactly this host, then publishes
    detection_failed when the inventory cannot corroborate it. Freezing that forever leaves
    /api/system reporting the recovered mismatch while the sidebar, Export and Video keep
    saying detection failed and never offer the repair.
    """
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "detection_failed")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", None)
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", OSError("[WinError 126] cudart64_12.dll"))
    monkeypatch.setattr(hw, "_installed_torch_label_on_disk", lambda: "2.11.0+cpu")
    monkeypatch.setattr(
        hw,
        "torch_build_snapshot",
        lambda **_kw: {"reason": "torch_cpu_build", "usable": False, "unknown": False},
    )
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {
            "devices": [{"vendor": "nvidia", "name": "NVIDIA RTX A4000"}],
            "unknown": False,
        },
    )

    assert hw.current_chat_only_verdict() == ("torch_cpu_build", "2.11.0+cpu")


def test_a_detection_failure_never_degrades_into_no_gpu(monkeypatch):
    """A host that never measured cannot be told it has no GPU; that IS the claim
    detect_hardware() refused to make for it."""
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "detection_failed")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", None)
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", OSError("boom"))
    monkeypatch.setattr(
        hw,
        "torch_build_snapshot",
        lambda **_kw: {"reason": "torch_cpu_build", "usable": False, "unknown": False},
    )
    monkeypatch.setattr(
        hw, "get_physical_gpu_inventory", lambda **_kw: {"devices": [], "unknown": False}
    )

    assert hw.current_chat_only_verdict() == ("detection_failed", None)


@pytest.mark.parametrize("reason", ["mlx_unavailable", "intel_mac"])
def test_the_other_frozen_verdicts_stay_frozen(monkeypatch, reason):
    """Only the disk-classified failure moved; a 60 second probe still cannot speak to these."""
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", reason)
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "detail")
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", OSError("boom"))
    assert hw.current_chat_only_verdict() == (reason, "detail")


def test_a_detection_failure_with_an_importable_torch_stays_frozen(monkeypatch):
    """Nothing classified this host, so there is nothing for the inventory to confirm."""
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "detection_failed")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", None)
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", None)
    assert hw.current_chat_only_verdict() == ("detection_failed", None)


def test_an_unreadable_drm_walk_is_not_a_host_without_amd_cards(monkeypatch, tmp_path):
    """/sys/class/drm going unreadable is the driver restart the TTL exists to tolerate.

    Swallowing that into an empty list made the failure look like a successful "no
    cards", so nothing marked the vendor unanswered and the carry-forward had nothing
    to act on: the previously detected cards were dropped for a whole TTL.
    """

    def denied(_root):
        raise PermissionError("EACCES during driver restart")

    monkeypatch.setattr(hw.os, "listdir", denied)
    assert hw._linux_drm_sysfs_records() == [], "the ranking callers keep the old shape"
    assert hw._linux_drm_sysfs_records(distinguish_failure = True) is None


def test_a_host_with_no_drm_subsystem_has_answered(monkeypatch):
    """Missing is not unreadable: a container or a machine with no DRM said "no cards"."""

    def absent(_root):
        raise FileNotFoundError("/sys/class/drm")

    monkeypatch.setattr(hw.os, "listdir", absent)
    assert hw._linux_drm_sysfs_records(distinguish_failure = True) == []


def test_a_card_whose_vendor_cannot_be_read_makes_the_walk_partial(monkeypatch):
    """A partial answer published as a complete one drops that card for a TTL."""
    monkeypatch.setattr(hw.os, "listdir", lambda _root: ["card0"])
    real_open = open

    def blocked(path, *a, **k):
        if str(path).endswith("vendor"):
            raise PermissionError("EACCES")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", blocked)
    assert hw._linux_drm_sysfs_records(distinguish_failure = True) is None
    assert hw._linux_drm_sysfs_records() == []


def test_a_cardn_with_no_pci_vendor_is_not_a_lost_card(monkeypatch):
    """A virtual or platform cardN publishes no vendor file and never was a GPU."""
    monkeypatch.setattr(hw.os, "listdir", lambda _root: ["card0"])
    real_open = open

    def missing(path, *a, **k):
        if str(path).endswith("vendor"):
            raise FileNotFoundError(path)
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", missing)
    assert hw._linux_drm_sysfs_records(distinguish_failure = True) == []


def test_the_inventory_marks_both_sysfs_vendors_unanswered(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _smi(monkeypatch, "", raises = FileNotFoundError)
    monkeypatch.setattr(hw, "_linux_drm_sysfs_records", lambda **_kw: None)

    inventory = hw.get_physical_gpu_inventory()

    assert inventory["unanswered"] == ["amd", "intel"]
    assert inventory["unknown"] is True


def test_a_previously_seen_amd_card_survives_an_unreadable_walk(monkeypatch):
    """End to end: the distinction is only worth having if the carry-forward uses it."""
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _smi(monkeypatch, "", raises = FileNotFoundError)
    card = {
        "vendor": "amd",
        "index": 0,
        "name": None,
        "memory_total_gb": 20.0,
        "source": "sysfs-drm",
    }
    monkeypatch.setattr(hw, "_linux_drm_sysfs_records", lambda **_kw: [dict(card)])
    good = hw.get_physical_gpu_inventory()
    assert [d["vendor"] for d in good["devices"]] == ["amd"]

    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", (0.0, good))
    monkeypatch.setattr(hw, "_linux_drm_sysfs_records", lambda **_kw: None)
    after = hw.get_physical_gpu_inventory()

    assert [d["vendor"] for d in after["devices"]] == ["amd"]
    assert after["unknown"] is True


# =========================== the URL outranks the family, as install.sh's resolver does


@pytest.mark.parametrize(
    "chosen,url,family",
    [
        # install.sh returns on UNSLOTH_TORCH_INDEX_URL without ever reading the family, so
        # when the two disagree the family is not a second opinion, it is dead. A stale
        # ..._FAMILY=cpu beside a new ..._URL=.../cu128 suppressed a real CPU-wheel mismatch.
        (False, "https://download.pytorch.org/whl/cu128", "cpu"),
        (True, "https://download.pytorch.org/whl/cpu", "cu128"),
        (True, "", "cpu"),
        (True, "https://download.pytorch.org/whl/cpu", ""),
        (False, "", "cu128"),
        (False, "", ""),
        # Whitespace-only is unset, the way install.sh trims it.
        (True, "   ", "cpu"),
    ],
)
def test_a_cpu_choice_reads_the_url_before_the_family(monkeypatch, chosen, url, family):
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", url)
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", family)
    monkeypatch.setattr(hw.sys, "prefix", "/nonexistent-prefix-for-this-test")
    assert hw._expected_cpu_flavor_was_chosen() is chosen


@pytest.mark.parametrize(
    "helper,url,family,chosen",
    [
        (
            "_expected_rocm_flavor_was_chosen",
            "https://download.pytorch.org/whl/cu128",
            "rocm6.4",
            False,
        ),
        (
            "_expected_rocm_flavor_was_chosen",
            "https://download.pytorch.org/whl/rocm6.4",
            "cpu",
            True,
        ),
        ("_expected_xpu_flavor_was_chosen", "https://download.pytorch.org/whl/cu128", "xpu", False),
        ("_expected_xpu_flavor_was_chosen", "https://download.pytorch.org/whl/xpu", "cpu", True),
    ],
)
def test_the_rocm_and_xpu_helpers_use_the_same_precedence(monkeypatch, helper, url, family, chosen):
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", url)
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", family)
    monkeypatch.setattr(hw.sys, "prefix", "/nonexistent-prefix-for-this-test")
    assert getattr(hw, helper)() is chosen


def test_the_backend_precedence_matches_install_sh():
    """One rule, two languages: install.sh returns on the URL and never reads the family."""
    source = (pathlib.Path(hw.__file__).resolve().parents[4] / "install.sh").read_text(
        encoding = "utf-8"
    )
    block = source[source.index('_url="${UNSLOTH_TORCH_INDEX_URL:-}"') :]
    block = block[: block.index('_family="${UNSLOTH_TORCH_INDEX_FAMILY:-}"')]
    assert (
        'echo "$_url"; return' in block
    ), "install.sh no longer short-circuits on the URL; the backend mirrors that rule"


# ================== a deliberate CPU install whose torch will not import is not broken


@pytest.fixture
def _broken_torch_on_a_gpu_host(monkeypatch):
    """detect_hardware()'s disk-only branch: torch is installed and will not import."""
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", OSError("[WinError 126]"))
    monkeypatch.setattr(hw, "_installed_torch_label_on_disk", lambda: "2.11.0+cpu")
    # The real venv's torch DOES carry CUDA markers, and the classifier reads those too.
    monkeypatch.setattr(
        hw, "_installed_torch_markers_on_disk", lambda: {"hip": "", "cuda": "", "xpu": ""}
    )
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {
            "devices": [{"vendor": "nvidia", "name": "NVIDIA RTX A4000"}],
            "unknown": False,
        },
    )
    for var in (
        "UNSLOTH_TORCH_INDEX_URL",
        "UNSLOTH_TORCH_INDEX_FAMILY",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "ZE_AFFINITY_MASK",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(hw.sys, "prefix", "/nonexistent-prefix-for-this-test")


def test_a_broken_torch_on_an_unpinned_host_still_reports_the_mismatch(
    monkeypatch, _broken_torch_on_a_gpu_host
):
    assert hw._classification_from_disk_label() == "torch_cpu_build"
    assert hw._expected_cpu_flavor_was_chosen() is False


def test_a_deliberate_cpu_install_is_not_offered_a_gpu_repair(
    monkeypatch, _broken_torch_on_a_gpu_host
):
    """The repair the UI would offer reinstalls the very wheel that was asked for.

    classify_torch_build() suppresses this before its own disk fallback; the branch in
    detect_hardware() reached _classification_from_disk_label() directly and did not.
    """
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cpu")
    assert hw._expected_cpu_flavor_was_chosen() is True

    source = inspect.getsource(hw._detect_hardware_locked).replace(" ", "")
    assert (
        "_expected_cpu_flavor_was_chosen()" in source
    ), "the disk-only branch has to apply the same suppression classify_torch_build does"
    assert (
        "_masks_hide_every_accelerator" in source
    ), "and the mask suppression beside it: an emptied mask is a deliberate CPU pin too"


# ============================ the recovery the message names has to exist where it is read


@pytest.mark.parametrize("reason", ["torch_cpu_build", "torch_cuda_unavailable"])
def test_the_repair_advice_names_a_route_every_deployment_has(monkeypatch, reason):
    """Settings carries the repair row only inside the desktop app, and only for a backend
    it manages: DesktopRepairControl returns null without a Tauri repair context and null
    for an externally started server. A browser-hosted Studio, or a desktop attached to a
    server started from a terminal, was told to use a control that is not on its page."""
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", reason)
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.11.0+cpu")
    message = hw._gpu_present_but_unusable_message("export", (reason, "2.11.0+cpu"))

    assert message is not None
    assert (
        "installer" in message.lower()
    ), "the only route a browser-hosted or externally-attached Studio has is the installer"
    assert (
        "desktop app" in message
    ), "and the Settings route has to say where it lives, or it reads as universal"


def test_the_control_that_advice_points_at_still_has_both_gates():
    """If either gate is dropped the message could go back to naming Settings alone."""
    source = (
        pathlib.Path(hw.__file__).resolve().parents[3]
        / "frontend"
        / "src"
        / "features"
        / "settings"
        / "components"
        / "desktop-repair-control.tsx"
    ).read_text(encoding = "utf-8")
    assert "if (!repair || repair.isExternalServer) return null;" in source


@pytest.mark.parametrize(
    "pinned,chosen",
    [
        (True, True),
        (False, False),
        # bool("false") is True, so a migrated or hand-edited manifest carrying the string
        # would read as a deliberate pin and suppress the repair on a host that never chose
        # one. Unknown provenance gets the same answer an absent key gets.
        ("false", False),
        ("true", False),
        (1, False),
        (None, False),
    ],
)
def test_only_a_real_boolean_records_a_deliberate_cpu_choice(monkeypatch, tmp_path, pinned, chosen):
    manifest = {"schema": 1, "expected_torch_tag": "cpu"}
    if pinned is not None:
        manifest["expected_torch_tag_pinned"] = pinned
    (tmp_path / "unsloth_install_manifest.json").write_text(json.dumps(manifest), encoding = "utf-8")
    monkeypatch.setattr(hw.sys, "prefix", str(tmp_path))
    for var in ("UNSLOTH_TORCH_INDEX_URL", "UNSLOTH_TORCH_INDEX_FAMILY"):
        monkeypatch.delenv(var, raising = False)

    assert hw._recorded_install_flavor() == ("cpu", chosen)
    assert hw._expected_cpu_flavor_was_chosen() is chosen
