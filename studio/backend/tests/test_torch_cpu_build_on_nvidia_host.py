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


def _smi(
    monkeypatch,
    stdout: str,
    *,
    returncode: int = 0,
):
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
    assert inventory["available"] is False
    assert inventory["devices"] == []
    assert inventory["sources"] == []
    # "The driver answered and there are no cards" is not the same fact as "no probe
    # could answer", and collapsing them let a transient nvidia-smi timeout read as a
    # GPU disappearing and flip a settled mismatch verdict back to no_gpu.
    assert inventory["unknown"] is (returncode != 0 or failure is not None)


def test_the_windows_amd_adapters_are_inventoried_too(monkeypatch):
    # No vendor CLI is guaranteed on Windows AMD (amd-smi elevates a child without a
    # HIP SDK), so the DirectX registry map _rocm_windows_per_device_vram already
    # trusts is the source here as well.
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    _smi(monkeypatch, "", returncode = 9)
    # A registry record only counts when the live scan sees the card too.
    monkeypatch.setattr(
        hw,
        "_windows_live_adapter_names",
        lambda: ["AMD Radeon RX 7900 XT", "AMD Radeon(TM) Graphics"],
    )
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        # Called once per vendor id now; only AMD answers on this host.
        lambda vendor_id = hw._AMD_PCI_VENDOR_ID: (
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


@pytest.mark.parametrize("var", ["CUDA_VISIBLE_DEVICES"])
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


def test_chat_only_stops_claiming_this_host_has_no_gpu(monkeypatch, cpu_torch_on_an_nvidia_host):
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


# ========== What the review round after the first one added ==========


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

    # The driver case reads differently, because the remedy differs.
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cuda_unavailable")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.6.0+cu124")
    assert hw.export_capability()["export_unsupported_reason"] == "torch_cuda_unavailable"
    assert "driver" in hw.video_capability()["video_unsupported_message"]


def test_a_genuinely_gpu_less_host_keeps_the_old_wording(monkeypatch):
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU)
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

    # os.path.join, not a literal: this test runs on Linux, where the separator differs.
    nvsmi = nvidia.os.path.join(r"C:\Program Files", r"NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    monkeypatch.setattr(nvidia.os.path, "isfile", lambda p: p == nvsmi)
    assert nvidia._nvidia_smi_executable() == nvsmi

    # The driver store copy is the second chance.
    system32 = nvidia.os.path.join(r"C:\Windows", r"System32\nvidia-smi.exe")
    monkeypatch.setattr(nvidia.os.path, "isfile", lambda p: p == system32)
    assert nvidia._nvidia_smi_executable() == system32

    # Nothing anywhere: hand back the bare name so the caller's OSError path still runs.
    monkeypatch.setattr(nvidia.os.path, "isfile", lambda _p: False)
    assert nvidia._nvidia_smi_executable() == "nvidia-smi"


def test_path_resolution_is_a_no_op_off_windows_and_when_path_has_it(monkeypatch):
    monkeypatch.setattr(nvidia.platform, "system", lambda: "Linux")
    monkeypatch.setattr(nvidia.shutil, "which", lambda _name: None)
    assert nvidia._nvidia_smi_executable() == "nvidia-smi"

    # PATH wins whenever it answers, on every platform.
    monkeypatch.setattr(nvidia.platform, "system", lambda: "Windows")
    monkeypatch.setattr(nvidia.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    assert nvidia._nvidia_smi_executable() == "/usr/bin/nvidia-smi"


# ========== Round three ==========


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
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    _smi(monkeypatch, _TWO_A4000_ROWS)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setenv(var, mask)
    # Prime the cache: the mask set is read off the inventory WITHOUT blocking,
    # because it is reached from the chat-only verdict that /api/liveness reads. A
    # cold cache answers "unknown", which deliberately keeps every mask, so the
    # vendor rule only has something to say once a blocking read has happened -- as
    # one has by then, in _detect_hardware_locked at startup.
    hw.get_physical_gpu_inventory()

    assert hw.classify_torch_build() == "torch_cuda_unavailable"

    # Put an AMD card in the inventory and the same variable becomes meaningful.
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

    # HIP does work on Windows, so that one still counts.
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    assert hw.classify_torch_build() is None


def test_an_inventory_that_answers_nothing_keeps_every_mask(monkeypatch):
    # Unknown stays conservative: it must not start ignoring masks that may be real.
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

    # Nothing recorded and nothing pinned: unknown is not a choice.
    assert hw.classify_torch_build() == "torch_cpu_build"

    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cpu")
    assert hw.classify_torch_build() is None
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY")

    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cpu/")
    assert hw.classify_torch_build() is None
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL")

    # A GPU pin is not this, and neither is a GPU flavor in the manifest.
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cu124")
    assert hw.classify_torch_build() == "torch_cpu_build"
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY")

    manifest = tmp_path / "unsloth_install_manifest.json"
    manifest.write_text('{"schema": 1, "expected_torch_tag": "cpu"}', encoding = "utf-8")
    assert hw.classify_torch_build() is None

    manifest.write_text('{"schema": 1, "expected_torch_tag": "cu124"}', encoding = "utf-8")
    assert hw.classify_torch_build() == "torch_cpu_build"

    # A manifest with no flavor key, and a corrupt one, both mean unknown.
    manifest.write_text('{"schema": 1}', encoding = "utf-8")
    assert hw.classify_torch_build() == "torch_cpu_build"
    manifest.write_text("{not json", encoding = "utf-8")
    assert hw.classify_torch_build() == "torch_cpu_build"


def test_a_dead_accelerator_wheel_is_unaffected_by_a_cpu_record(monkeypatch, tmp_path):
    # The suppression is about a CPU wheel that was ASKED for. A cu124 wheel whose
    # runtime will not start is a real problem whatever the manifest says.
    import sys

    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda_dead"))
    monkeypatch.setattr(hw.sys, "prefix", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cpu")
    assert hw.classify_torch_build() is None  # the pin is honoured first

    # ... but with no CPU choice recorded at all it stays a real fault.
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
    # A connector entry pointing at the same card, which must not be double-counted.
    (drm / "card0-DP-1").mkdir()
    # And a card whose device directory is unreadable.
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
    # 0 and an unparseable value are both unknown, never a zero-capacity claim.
    assert records[1]["memory_total_gb"] is None
    assert records[2]["memory_total_gb"] is None
    # Intel publishes no equivalent total on the discrete path. Reporting the card with
    # an unknown capacity is the point; leaving it out would be the bug.
    assert records[3]["memory_total_gb"] is None


def test_the_sysfs_probe_is_silent_where_there_is_no_sysfs(monkeypatch):
    monkeypatch.setattr(hw.os, "listdir", lambda _p: (_ for _ in ()).throw(OSError("no such path")))
    assert hw._linux_drm_sysfs_records() == []


# ========== Round four ==========


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

    # A cu128 pin carrying the same token shape is not a CPU pin.
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
    assert hw.current_chat_only_verdict() == ("torch_cpu_build", "2.11.0+cpu")

    # And the reverse: a card that went away must not leave a mismatch behind.
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
        hw, "classify_torch_build", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
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

    assert hw.export_capability()["export_unsupported_reason"] == "torch_cpu_build"
    assert hw.video_capability()["video_unsupported_reason"] == "torch_cpu_build"


# ========== Round five ==========


def test_windows_intel_adapters_are_inventoried_too(monkeypatch):
    """An Arc host whose XPU wheel was replaced has exactly this shape.

    nvidia-smi contributes nothing there either, and the registry scan was filtered to
    AMD, so the inventory came back empty and the mismatch was discarded.
    """
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    _smi(monkeypatch, "", returncode = 9)
    monkeypatch.setattr(
        hw, "_windows_live_adapter_names", lambda: ["Intel(R) Arc(TM) A770 Graphics"]
    )
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda vendor_id = hw._AMD_PCI_VENDOR_ID: (
            {0x1: {"name": "Intel(R) Arc(TM) A770", "dedicated_memory_bytes": 16 * 1024**3}}
            if vendor_id == hw._INTEL_PCI_VENDOR_ID
            else {}
        ),
    )

    inventory = hw.get_physical_gpu_inventory()

    assert [d["vendor"] for d in inventory["devices"]] == ["intel"]
    assert inventory["devices"][0]["memory_total_gb"] == 16.0
    assert inventory["available"] is True
    # Devices found is an answer, whatever the nvidia probe did.
    assert inventory["unknown"] is False


def test_a_transient_probe_failure_does_not_retire_a_settled_mismatch(monkeypatch):
    """nvidia-smi timing out is not the GPU going away.

    The aggregate probe returns a structured empty result rather than raising, so the
    refreshed verdict read it as "no cards" and handed the user the opposite advice for
    a whole cache interval.
    """
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cpu_build")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.11.0+cpu")
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {"available": False, "devices": [], "unknown": True},
    )
    assert hw.current_chat_only_verdict() == ("torch_cpu_build", "2.11.0+cpu")

    # A probe that DID answer, and found nothing, still retires it.
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {"available": False, "devices": [], "unknown": False},
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
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    monkeypatch.setattr(hw, "_physical_gpu_inventory_refreshing", False)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "torch_cpu_build")
    monkeypatch.setattr(hw, "CHAT_ONLY_DETAIL", "2.11.0+cpu")

    # Cold cache: no probe runs inline, the refresh is handed to a thread, and the
    # explicit unknown keeps the frozen verdict rather than retiring it.
    assert hw.current_chat_only_verdict() == ("torch_cpu_build", "2.11.0+cpu")
    assert calls["blocking"] == 0, "nothing may shell out on the request path"
    assert calls["threads"] >= 1

    # A second caller while that refresh is still in flight does not queue another.
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
    # Stale by an hour.
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", (hw.time.monotonic() - 3600, warm))

    assert hw.get_physical_gpu_inventory(block = False) is warm
    assert calls["n"] == 0, "a stale answer beats a subprocess on the request path"

    # The blocking caller, which is not on a request path, does re-probe.
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
    # And the single-flight flag is released, or every later refresh is suppressed.
    assert hw._physical_gpu_inventory_refreshing is False


# ========== Round six ==========


def test_a_stale_registry_record_is_not_reported_as_a_gpu(monkeypatch):
    """The DirectX registry outlives the hardware.

    setup.ps1 says so and uses these records only to RE-LABEL an adapter its live WMI
    scan also returned, never to add one. A CPU-only machine with a driver record left
    behind would otherwise be told it has an unusable GPU and offered a repair that
    cannot restore absent hardware.
    """
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    _smi(monkeypatch, "", returncode = 9)
    monkeypatch.setattr(hw, "_windows_live_adapter_names", lambda: ["Microsoft Basic Display"])
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda vendor_id = hw._AMD_PCI_VENDOR_ID: (
            {0x1: {"name": "AMD Radeon RX 6800", "dedicated_memory_bytes": 16 * 1024**3}}
            if vendor_id == hw._AMD_PCI_VENDOR_ID
            else {}
        ),
    )

    inventory = hw.get_physical_gpu_inventory()
    assert inventory["devices"] == []
    assert inventory["sources"] == []


def test_a_live_scan_that_cannot_answer_reports_unknown_rather_than_guessing(monkeypatch):
    # Neither "this card is real" nor "this card is gone" is knowable then, and unknown
    # is what keeps a settled verdict instead of inventing one.
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    _smi(monkeypatch, "", returncode = 9)
    monkeypatch.setattr(hw, "_windows_live_adapter_names", lambda: None)
    monkeypatch.setattr(
        hw,
        "_windows_amd_adapter_records_by_luid",
        lambda vendor_id = hw._AMD_PCI_VENDOR_ID: (
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
        # The two sources spell the same card differently; either may be the prefix.
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

    # An Arc card by name does, and so do NVIDIA and AMD unconditionally.
    arc = [{"vendor": "intel", "name": "Intel(R) Arc(TM) A770 Graphics", "index": 0}]
    assert hw._devices_that_can_establish_a_mismatch(arc) == arc
    others = [{"vendor": "nvidia", "index": 0}, {"vendor": "amd", "index": 0}]
    assert hw._devices_that_can_establish_a_mismatch(others) == others


def test_a_nameless_intel_card_counts_once_xpu_was_actually_chosen(monkeypatch, tmp_path):
    # The Linux sysfs walk publishes no name, which is why the expectation and the
    # installed runtime are consulted as well.
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

    # An installed XPU wheel counts too: that venv plainly expected an Intel GPU.
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

    # The current answer is kept for now; the next detection pass publishes the real one.
    assert hw.current_chat_only_verdict() == ("torch_cuda_unavailable", "2.6.0+cu124")
    assert calls["n"] == 1

    # At most one request per recovery: a poll every few seconds must not retire the
    # epoch on every call and starve detection of a chance to settle.
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
