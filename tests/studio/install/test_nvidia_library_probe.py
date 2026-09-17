# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The installers read the NVIDIA inventory from the driver's libraries when nvidia-smi cannot answer.

Measured on a 8x B200 host (driver 590.48.01, CUDA 13.1) with a stub nvidia-smi that exits 1:
main resolved no prebuilt at all ("runnable by this driver=none", a source build or the CPU)
and picked the cu126 torch index; with the probe it resolves the cuda13 bundle and cu130.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
STUDIO = PACKAGE_ROOT / "studio"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, STUDIO / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROBE = _load("studio_nvidia_probe", "nvidia_probe.py")
ILP = _load("studio_install_llama_prebuilt", "install_llama_prebuilt.py")
IPS = _load("studio_install_python_stack", "install_python_stack.py")


def _inventory(
    source = "nvml",
    cuda = (12, 8),
    caps = ("8.9",),
    uuids = None,
):
    devices = [
        {
            "index": str(i),
            "uuid": (uuids or [f"GPU-{i:04d}" for i in range(len(caps))])[i],
            "name": "NVIDIA test",
            "compute_cap": cap,
        }
        for i, cap in enumerate(caps)
    ]
    return PROBE.NvidiaLibraryInventory(
        source = source, cuda_driver_version = cuda, driver_version = "570.1", devices = devices
    )


# ── the probe module ──


class TestProbeModule:
    def test_mig_rows_are_not_devices_to_the_installers(self):
        payload = {
            "source": "nvml",
            "cuda_driver_version": [13, 0],
            "devices": [
                {"index": "0", "uuid": "GPU-a", "name": "H100", "compute_cap": "9.0"},
                {
                    "index": "0",
                    "uuid": "MIG-b",
                    "name": "H100 MIG",
                    "compute_cap": "9.0",
                    "mig": "1",
                },
            ],
        }
        inv = PROBE._from_payload(payload)
        assert [d["uuid"] for d in inv.devices] == ["GPU-a"]

    def test_the_payload_round_trips(self):
        payload = {
            "source": "nvml",
            "cuda_driver_version": [13, 1],
            "driver_version": "590.48.01",
            "devices": [
                {
                    "index": "0",
                    "uuid": "GPU-x",
                    "name": "B200",
                    "compute_cap": "10.0",
                    "memory_total_mib": "183359",
                    "memory_free_mib": "182630",
                }
            ],
        }
        inv = PROBE._from_payload(payload)
        assert inv.cuda_driver_version == (13, 1)
        assert inv.devices == payload["devices"]
        # An older payload without the memory fields still reads; they come back empty.
        old = {
            **payload,
            "devices": [{"index": "0", "uuid": "GPU-x", "name": "B200", "compute_cap": "10.0"}],
        }
        assert PROBE._from_payload(old).devices[0]["memory_free_mib"] == ""
        assert PROBE._from_payload(None) is None
        assert PROBE._from_payload({"source": "other"}) is None

    def test_cuda_packs_as_major_thousand_minor_ten(self):
        assert PROBE._split_cuda_version(13010) == (13, 1)
        assert PROBE._split_cuda_version(12080) == (12, 8)
        assert PROBE._split_cuda_version(0) is None

    @pytest.mark.parametrize(
        ("driver", "cuda"),
        [
            ("590.48.01", (13, 0)),
            ("580.65.06", (13, 0)),
            ("575.57.08", (12, 8)),
            ("570.26", (12, 8)),
            ("560.35.03", (12, 6)),
            ("550.54.14", (12, 4)),
            ("535.104.05", (12, 2)),
            ("525.60.13", (12, 0)),
            ("470.182.03", (11, 0)),
            ("440.33", None),
            ("", None),
            ("garbage", None),
        ],
    )
    def test_the_driver_release_bounds_the_cuda_version(self, driver, cuda):
        # NVIDIA's minor-version-compatibility table: CUDA 13.x needs R580+, 12.x R525+, 11.x R450+.
        assert PROBE.cuda_version_for_driver(driver) == cuda

    def test_the_proc_version_line_is_read(self, tmp_path):
        path = tmp_path / "version"
        path.write_text(
            "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  590.48.01  Release Build\n"
            "GCC version:  gcc version 13.3.0\n",
            encoding = "utf-8",
        )
        assert PROBE.proc_driver_version(str(path)) == "590.48.01"
        assert PROBE.proc_driver_version(str(tmp_path / "missing")) == ""

    def test_probe_runs_the_module_in_a_child_with_a_deadline(self, monkeypatch):
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["timeout"] = kwargs.get("timeout")
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout = json.dumps(
                    {
                        "source": "cuda",
                        "cuda_driver_version": [12, 4],
                        "driver_version": "",
                        "devices": [],
                    }
                ),
                stderr = "",
            )

        monkeypatch.setattr(PROBE.subprocess, "run", fake_run)
        monkeypatch.setattr(PROBE.sys, "platform", "linux")
        monkeypatch.delenv("UNSLOTH_NVIDIA_LIBRARY_PROBE", raising = False)
        inv = PROBE.probe(timeout = 7)
        assert seen["cmd"][0] == sys.executable and seen["cmd"][-1] == "--json"
        assert "-I" in seen["cmd"]
        assert seen["timeout"] == 7
        assert inv.source == "cuda" and inv.cuda_driver_version == (12, 4)

    def test_a_hung_or_broken_child_is_none(self, monkeypatch):
        def hang(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

        monkeypatch.setattr(PROBE.subprocess, "run", hang)
        monkeypatch.setattr(PROBE.sys, "platform", "linux")
        monkeypatch.delenv("UNSLOTH_NVIDIA_LIBRARY_PROBE", raising = False)
        assert PROBE.probe() is None
        monkeypatch.setattr(
            PROBE.subprocess,
            "run",
            lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout = "not json", stderr = ""),
        )
        assert PROBE.probe() is None

    def test_macos_never_probes(self, monkeypatch):
        monkeypatch.setattr(PROBE.sys, "platform", "darwin")
        monkeypatch.setattr(PROBE.subprocess, "run", lambda *a, **k: pytest.fail("ran"))
        assert PROBE.probe() is None

    def test_the_switch_turns_it_off(self, monkeypatch):
        # The suites set it, so a test faking a CPU host cannot find the real GPUs.
        monkeypatch.setattr(PROBE.sys, "platform", "linux")
        monkeypatch.setattr(PROBE.subprocess, "run", lambda *a, **k: pytest.fail("ran"))
        monkeypatch.setenv("UNSLOTH_NVIDIA_LIBRARY_PROBE", "0")
        assert PROBE.probe() is None

    def test_the_cli_exits_by_gpu_presence(self, monkeypatch, capsys):
        monkeypatch.setattr(PROBE, "probe_in_process", lambda: None)
        assert PROBE.main(["--json"]) == 1
        assert capsys.readouterr().out.strip() == "null"
        monkeypatch.setattr(
            PROBE,
            "probe_in_process",
            lambda: {
                "source": "nvml",
                "cuda_driver_version": [12, 8],
                "driver_version": "570",
                "devices": [{"index": "0", "uuid": "u", "name": "RTX", "compute_cap": "8.9"}],
            },
        )
        assert PROBE.main([]) == 0
        assert "GPU 0: RTX (compute 8.9)" in capsys.readouterr().out

    @pytest.mark.skipif(
        sys.platform == "darwin" or not shutil.which("nvidia-smi"),
        reason = "needs a real NVIDIA host",
    )
    def test_on_a_real_host_the_library_agrees_with_nvidia_smi(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_NVIDIA_LIBRARY_PROBE", raising = False)
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output = True, text = True, timeout = 60)
        if listing.returncode != 0 or "GPU " not in listing.stdout:
            pytest.skip("nvidia-smi lists no GPU here")
        inv = PROBE.probe()
        assert inv is not None and inv.source == "nvml"
        assert len(inv.devices) == sum(
            1 for line in listing.stdout.splitlines() if line.startswith("GPU ")
        )
        assert inv.cuda_driver_version is not None and inv.cuda_driver_version[0] >= 11
        # The memory reading the runtime probe needs; a busy card may legitimately have none free.
        assert all(int(d["memory_total_mib"]) > 0 for d in inv.devices)
        assert all(
            0 <= int(d["memory_free_mib"]) <= int(d["memory_total_mib"]) for d in inv.devices
        )


# ── detect_host ──


_HOST_TOOLS = {"nvidia-smi", "rocminfo", "amd-smi", "rocm-smi", "hipinfo", "vulkaninfo", "nvcc"}


def _hide_the_real_host(monkeypatch, *, nvidia_smi):
    """A Linux x86_64 host with no ROCm and no DRM vendors; nvidia-smi is `nvidia_smi`."""
    monkeypatch.setattr(ILP.platform, "system", lambda: "Linux")
    monkeypatch.setattr(ILP.platform, "machine", lambda: "x86_64")
    which = ILP.shutil.which
    monkeypatch.setattr(
        ILP.shutil,
        "which",
        lambda name, *a, **k: (nvidia_smi if name == "nvidia-smi" else None)
        if name in _HOST_TOOLS
        else which(name, *a, **k),
    )
    isdir = ILP.os.path.isdir
    monkeypatch.setattr(ILP.os.path, "isdir", lambda p: False if "nvidia" in str(p) else isdir(p))
    exists = ILP.os.path.exists
    monkeypatch.setattr(
        ILP.os.path,
        "exists",
        lambda p: False if ("rocm" in str(p) or "kfd" in str(p)) else exists(p),
    )
    monkeypatch.setattr(ILP.os, "access", lambda p, mode, *a, **k: False)
    monkeypatch.setattr(ILP.glob, "glob", lambda *a, **k: [])


def _failing_smi(monkeypatch, *, exit_code = 1):
    def run_capture(command, **kwargs):
        if command and "nvidia-smi" in str(command[0]):
            return subprocess.CompletedProcess(command, exit_code, stdout = "", stderr = "failed")
        return subprocess.CompletedProcess(command, 1, stdout = "", stderr = "")

    monkeypatch.setattr(ILP, "run_capture", run_capture)


class TestDetectHostFallsBackToTheLibraries:
    def test_a_failing_nvidia_smi_no_longer_reads_as_no_gpu(self, monkeypatch):
        _hide_the_real_host(monkeypatch, nvidia_smi = "/usr/bin/nvidia-smi")
        _failing_smi(monkeypatch)
        monkeypatch.setattr(
            ILP, "nvidia_library_inventory", lambda: _inventory(caps = ("8.9", "8.6"))
        )
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        host = ILP.detect_host()
        assert host.has_physical_nvidia and host.has_usable_nvidia
        assert host.driver_cuda_version == (12, 8)
        assert host.compute_caps == ["89", "86"]
        assert host.physical_compute_caps == ["89", "86"]

    def test_an_absent_nvidia_smi_is_the_same(self, monkeypatch):
        _hide_the_real_host(monkeypatch, nvidia_smi = None)
        monkeypatch.setattr(ILP, "nvidia_library_inventory", lambda: _inventory())
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        host = ILP.detect_host()
        assert host.has_usable_nvidia and host.driver_cuda_version == (12, 8)

    def test_nvml_rows_are_masked_like_nvidia_smi_rows(self, monkeypatch):
        # NVML is the physical inventory; an emptied mask leaves the GPU physical, not usable.
        _hide_the_real_host(monkeypatch, nvidia_smi = None)
        monkeypatch.setattr(
            ILP, "nvidia_library_inventory", lambda: _inventory(caps = ("8.9", "6.1"))
        )
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        host = ILP.detect_host()
        assert host.has_physical_nvidia and not host.has_usable_nvidia
        assert host.compute_caps == [] and host.physical_compute_caps == ["89", "61"]
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        host = ILP.detect_host()
        assert host.has_usable_nvidia and host.compute_caps == ["61"]

    def test_a_mask_nvml_rows_cannot_name_keeps_the_gpu_usable(self, monkeypatch):
        # A MIG UUID names a slice, not the parent GPU NVML lists: usable, as with nvidia-smi.
        _hide_the_real_host(monkeypatch, nvidia_smi = None)
        monkeypatch.setattr(ILP, "nvidia_library_inventory", lambda: _inventory(caps = ("9.0",)))
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-4b3c2a1d-0000-1111-2222-333344445555")
        host = ILP.detect_host()
        assert host.has_physical_nvidia and host.has_usable_nvidia
        assert host.compute_caps == [] and host.physical_compute_caps == ["90"]
        # An explicit UUID that matches nothing stays unusable.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-00000000-0000-0000-0000-000000000000")
        assert not ILP.detect_host().has_usable_nvidia

    def test_the_cuda_driver_api_rows_are_already_masked(self, monkeypatch):
        _hide_the_real_host(monkeypatch, nvidia_smi = None)
        monkeypatch.setattr(
            ILP, "nvidia_library_inventory", lambda: _inventory(source = "cuda", caps = ("7.5",))
        )
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        host = ILP.detect_host()
        assert host.has_usable_nvidia and host.compute_caps == ["75"]

    def test_a_working_nvidia_smi_is_not_second_guessed(self, monkeypatch):
        _hide_the_real_host(monkeypatch, nvidia_smi = "/usr/bin/nvidia-smi")

        def run_capture(command, **kwargs):
            args = command[1:]
            if args == ["-L"]:
                out = "GPU 0: NVIDIA RTX (UUID: GPU-abc)\n"
            elif not args:
                out = "NVIDIA-SMI 570  Driver Version: 570.1  CUDA Version: 12.8\n"
            else:
                out = "0, GPU-abc, 8.9\n"
            return subprocess.CompletedProcess(command, 0, stdout = out, stderr = "")

        monkeypatch.setattr(ILP, "run_capture", run_capture)
        monkeypatch.setattr(ILP, "nvidia_library_inventory", lambda: pytest.fail("probed"))
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        host = ILP.detect_host()
        assert host.driver_cuda_version == (12, 8) and host.compute_caps == ["89"]

    def test_the_kernel_module_release_bounds_the_driver_version(self, monkeypatch):
        # nvidia-smi and the libraries both failed; /proc/driver/nvidia/gpus says NVIDIA.
        _hide_the_real_host(monkeypatch, nvidia_smi = None)
        isdir = ILP.os.path.isdir
        monkeypatch.setattr(
            ILP.os.path,
            "isdir",
            lambda p: True
            if str(p) == "/proc/driver/nvidia/gpus"
            else (False if "nvidia" in str(p) else isdir(p)),
        )
        monkeypatch.setattr(
            ILP.os, "listdir", lambda p: ["0000:01:00.0"] if "nvidia" in str(p) else []
        )
        monkeypatch.setattr(ILP, "nvidia_library_inventory", lambda: None)
        monkeypatch.setenv("UNSLOTH_NVIDIA_LIBRARY_PROBE", "1")
        monkeypatch.setattr(ILP._nvidia_probe, "proc_driver_version", lambda: "580.65.06")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        host = ILP.detect_host()
        assert host.has_physical_nvidia and host.driver_cuda_version == (13, 0)

    def test_a_cpu_host_stays_cpu(self, monkeypatch):
        _hide_the_real_host(monkeypatch, nvidia_smi = None)
        monkeypatch.setattr(ILP, "nvidia_library_inventory", lambda: None)
        monkeypatch.setattr(ILP, "proc_driver_version", lambda: "")
        host = ILP.detect_host()
        assert not host.has_physical_nvidia and host.driver_cuda_version is None


# ── install_python_stack ──


class TestTorchIndexFallsBackToTheLibraries:
    def _no_smi(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
        monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
        monkeypatch.setattr(IPS, "_nvidia_smi_usable_candidates", lambda: [])
        monkeypatch.setattr(IPS.platform, "machine", lambda: "x86_64")

    def test_the_library_version_picks_the_family(self, monkeypatch):
        self._no_smi(monkeypatch)
        monkeypatch.setattr(
            IPS, "_nvidia_library_inventory", lambda: _inventory(cuda = (13, 1), caps = ("10.0",))
        )
        assert IPS._detect_cuda_torch_index_url().endswith("/cu130")
        monkeypatch.setattr(
            IPS, "_nvidia_library_inventory", lambda: _inventory(cuda = (12, 6), caps = ("8.6",))
        )
        assert IPS._detect_cuda_torch_index_url().endswith("/cu126")

    def test_the_pre_turing_cap_reads_the_library_sms(self, monkeypatch):
        self._no_smi(monkeypatch)
        monkeypatch.setattr(
            IPS, "_nvidia_library_inventory", lambda: _inventory(cuda = (13, 0), caps = ("6.1",))
        )
        assert IPS._detect_cuda_torch_index_url().endswith("/cu126")

    def test_without_compute_caps_the_default_stays(self, monkeypatch):
        # A driver version alone cannot rule out Maxwell, Pascal or Volta, which cu128+ drop.
        self._no_smi(monkeypatch)
        monkeypatch.setattr(IPS.sys, "platform", "linux")
        monkeypatch.setenv("UNSLOTH_NVIDIA_LIBRARY_PROBE", "1")
        monkeypatch.setattr(IPS._nvidia_probe, "proc_driver_version", lambda: "580.65.06")
        capless = PROBE.NvidiaLibraryInventory("cuda", (13, 0), "", [{"compute_cap": ""}])
        for inventory in (None, capless):
            monkeypatch.setattr(IPS, "_nvidia_library_inventory", lambda inv = inventory: inv)
            assert IPS._detect_cuda_torch_index_url().endswith("/cu126")

    def test_nothing_at_all_still_defaults_to_cu126(self, monkeypatch):
        self._no_smi(monkeypatch)
        monkeypatch.setattr(IPS, "_nvidia_library_inventory", lambda: None)
        monkeypatch.setattr(IPS._nvidia_probe, "proc_driver_version", lambda: "")
        assert IPS._detect_cuda_torch_index_url().endswith("/cu126")

    def test_presence_falls_back_to_the_libraries(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        monkeypatch.setattr(IPS, "_nvidia_smi_candidates", lambda: [])
        monkeypatch.setattr(IPS.os.path, "isdir", lambda p: False)
        monkeypatch.setattr(IPS, "_nvidia_library_inventory", lambda: _inventory())
        assert IPS._has_usable_nvidia_gpu() is True
        monkeypatch.setattr(IPS, "_nvidia_library_inventory", lambda: None)
        assert IPS._has_usable_nvidia_gpu() is False

    def test_presence_applies_an_explicit_mask_to_the_nvml_rows(self, monkeypatch):
        monkeypatch.setattr(IPS, "_nvidia_smi_candidates", lambda: [])
        monkeypatch.setattr(IPS.os.path, "isdir", lambda p: False)
        monkeypatch.setattr(
            IPS, "_nvidia_library_inventory", lambda: _inventory(caps = ("8.9", "6.1"))
        )
        for mask, usable in (
            ("1", True),
            ("gpu-0001", True),
            ("5", False),
            ("GPU-ffff", False),
            ("MIG-4b3c2a1d-0000-1111-2222-333344445555", True),
        ):
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
            assert IPS._has_usable_nvidia_gpu() is usable, mask
        # The CUDA driver rows are already masked, so a miss there is not second-guessed.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
        monkeypatch.setattr(IPS, "_nvidia_library_inventory", lambda: _inventory(source = "cuda"))
        assert IPS._has_usable_nvidia_gpu() is True


# ── setup.sh ──


SETUP_TEXT = (STUDIO / "setup.sh").read_text(encoding = "utf-8")


def test_setup_sh_asks_the_library_last():
    start = SETUP_TEXT.index("_setup_has_physical_nvidia_gpu() {")
    body = SETUP_TEXT[start : SETUP_TEXT.index("\n}\n", start)]
    assert body.index("/proc/driver/nvidia/gpus") < body.index("nvidia_probe.py")
    assert '_setup_run_smi python3 -I "$SCRIPT_DIR/nvidia_probe.py"' in body


@pytest.mark.skipif(
    shutil.which("bash") is None or sys.platform == "win32",
    reason = "a POSIX bash is required (Windows resolves bash to WSL)",
)
def test_setup_sh_probe_hook_runs(tmp_path):
    start = SETUP_TEXT.index("_setup_run_smi() {")
    runner = SETUP_TEXT[start : SETUP_TEXT.index("\n}\n", start) + 3]
    start = SETUP_TEXT.index("_setup_has_physical_nvidia_gpu() {")
    # The real /proc would answer for this host; point the fallback at an empty dir.
    fn = SETUP_TEXT[start : SETUP_TEXT.index("\n}\n", start) + 3].replace(
        "/proc/driver/nvidia/gpus", str(tmp_path / "nogpus")
    )
    script_dir = tmp_path / "studio"
    script_dir.mkdir()
    (script_dir / "nvidia_probe.py").write_text("import sys; sys.exit(int(sys.argv[-1] != 'x'))\n")
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    (stub_bin / "python3").write_text('#!/bin/sh\nexit "$PROBE_RC"\n')
    (stub_bin / "python3").chmod(0o755)
    (stub_bin / "nvidia-smi").write_text("#!/bin/sh\nexit 1\n")
    (stub_bin / "nvidia-smi").chmod(0o755)
    harness = (
        runner
        + fn
        + '\nSCRIPT_DIR="$1"\nif _setup_has_physical_nvidia_gpu; then echo GPU; else echo NONE; fi\n'
    )
    for rc, expect in (("0", "GPU"), ("1", "NONE")):
        result = subprocess.run(
            ["bash", "-c", harness, "h", str(script_dir)],
            env = {
                **os.environ,
                "PATH": f"{stub_bin}:{os.environ['PATH']}",
                "PROBE_RC": rc,
                "UNSLOTH_NVIDIA_LIBRARY_PROBE": "1",
            },
            capture_output = True,
            text = True,
            timeout = 60,
        )
        assert result.stdout.strip() == expect, result.stderr
