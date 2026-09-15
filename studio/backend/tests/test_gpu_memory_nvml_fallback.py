# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The runtime GPU memory probe reads NVML when nvidia-smi cannot answer.

An absent, stale or hung nvidia-smi left _get_gpu_memory empty, and the embedding server
read that as "no GPU": -ngl 0 and a blanked CUDA_VISIBLE_DEVICES around a working CUDA
build. The installers stopped trusting that misread in #10985; this is the runtime half.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

from core.inference import llama_cpp as mod
from core.inference.llama_cpp import LlamaCppBackend
from core.rag import embed_llama_server as embed_mod

# Captured before the autouse fixture stubs it, for the one test that exercises the real finder.
_REAL_FINDER = LlamaCppBackend.__dict__["_find_llama_server_binary"]


def _payload(rows, source = "nvml"):
    return {
        "source": source,
        "cuda_driver_version": [13, 0],
        "driver_version": "580.65.06",
        "devices": rows,
    }


def _row(
    index,
    free,
    total = 24576,
    uuid = None,
):
    return {
        "index": str(index),
        "uuid": uuid or f"GPU-{index:04d}",
        "name": "NVIDIA test",
        "compute_cap": "8.9",
        "memory_total_mib": str(total),
        "memory_free_mib": str(free),
    }


@pytest.fixture
def probe_script(tmp_path, monkeypatch):
    """A stand-in for studio/nvidia_probe.py that prints whatever payload the test wrote."""
    payload_path = tmp_path / "payload.json"
    script = tmp_path / "nvidia_probe.py"
    script.write_text(
        f"import sys; sys.stdout.write(open({str(payload_path)!r}).read())\n", encoding = "utf-8"
    )
    monkeypatch.setenv("UNSLOTH_NVIDIA_PROBE", str(script))
    monkeypatch.delenv("UNSLOTH_NVIDIA_LIBRARY_PROBE", raising = False)
    monkeypatch.setattr(mod.sys, "platform", "linux")

    def write(payload):
        payload_path.write_text(json.dumps(payload), encoding = "utf-8")

    return write


@pytest.fixture(autouse = True)
def _no_other_probes(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: False))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/opt/llama-server")
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_get_gpu_memory_amd_smi", staticmethod(lambda *a, **k: [])
    )
    monkeypatch.setitem(sys.modules, "torch", None)
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)


def _failing_smi(monkeypatch):
    real_run = mod.subprocess.run

    def run(cmd, *args, **kwargs):
        if cmd and os.path.basename(str(cmd[0])) == "nvidia-smi":
            return types.SimpleNamespace(returncode = 1, stdout = "", stderr = "")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(mod.subprocess, "run", run)


class TestTheMemoryProbeFallsBackToNvml:
    def test_a_failing_nvidia_smi_no_longer_reads_as_no_gpu(self, monkeypatch, probe_script):
        _failing_smi(monkeypatch)
        probe_script(_payload([_row(1, 20000), _row(0, 8000, 12288)]))
        assert LlamaCppBackend._get_gpu_memory() == [(0, 8000, 12288), (1, 20000, 24576)]
        assert LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES is True

    def test_the_mask_applies_as_it_does_to_nvidia_smi_rows(self, monkeypatch, probe_script):
        _failing_smi(monkeypatch)
        probe_script(_payload([_row(0, 8000), _row(1, 20000)]))
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        assert LlamaCppBackend._get_gpu_memory() == [(1, 20000, 24576)]
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert LlamaCppBackend._get_gpu_memory() == []

    def test_a_uuid_mask_selects_by_the_rows_own_uuid(self, monkeypatch, probe_script):
        _failing_smi(monkeypatch)
        probe_script(
            _payload([_row(0, 8000, uuid = "GPU-aaaa1111-0"), _row(1, 20000, uuid = "GPU-bbbb2222-1")])
        )
        # A full uuid, a prefix, mask order, and a mixed index + uuid mask, as the CUDA runtime reads them.
        for mask, expected in (
            ("GPU-bbbb2222-1", [(1, 20000, 24576)]),
            ("GPU-aaaa", [(0, 8000, 24576)]),
            ("GPU-bbbb,GPU-aaaa", [(0, 8000, 24576), (1, 20000, 24576)]),
            ("1,GPU-aaaa", [(0, 8000, 24576), (1, 20000, 24576)]),
            # An entry naming no single device ends the mask there, as CUDA documents
            # for "0,2,-1,1": the devices before it stay visible, nothing after it does.
            ("nope", []),
            ("GPU-", []),
            ("1,-1,0", [(1, 20000, 24576)]),
            ("0,GPU-,1", [(0, 8000, 24576)]),
            ("0,7,1", [(0, 8000, 24576)]),
        ):
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
            assert LlamaCppBackend._get_gpu_memory() == expected, mask

    def test_a_mig_assignment_names_the_slice_row_the_probe_lists(self, monkeypatch, probe_script):
        _failing_smi(monkeypatch)
        slice_row = dict(_row(0, 9000, 20480, uuid = "MIG-cccc3333-0"), mig = "1")
        probe_script(_payload([_row(0, 60000, 81920, uuid = "GPU-aaaa1111-0"), slice_row]))
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-cccc")
        assert LlamaCppBackend._get_gpu_memory() == [(0, 9000, 20480)]
        # An index, a GPU- entry and no mask at all name the parent, and CUDA exposes its
        # first slice then, not the whole card: the slice's memory is what the model gets.
        for mask in ("0", "GPU-aaaa"):
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
            assert LlamaCppBackend._get_gpu_memory() == [(0, 9000, 20480)], mask
            assert LlamaCppBackend._child_visibility_for([0]) == "MIG-cccc3333-0"
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        assert LlamaCppBackend._get_gpu_memory() == [(0, 9000, 20480)]
        assert LlamaCppBackend._child_visibility_for([0]) == "MIG-cccc3333-0"
        # A slice the probe does not list hides every GPU rather than exposing the parent.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-dddd")
        assert LlamaCppBackend._get_gpu_memory() == []
        # A second slice named by the mask is that slice, not the parent's first.
        second = dict(_row(0, 4000, 10240, uuid = "MIG-dddd4444-0"), mig = "1")
        probe_script(_payload([_row(0, 60000, 81920, uuid = "GPU-aaaa1111-0"), slice_row, second]))
        assert LlamaCppBackend._get_gpu_memory() == [(0, 4000, 10240)]
        assert LlamaCppBackend._child_visibility_for([0]) == "MIG-dddd4444-0"
        # Two slices of one card: one row per physical index, the first named, and the
        # child is pinned to that one so the budget and the launch agree.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-dddd,MIG-cccc")
        assert LlamaCppBackend._get_gpu_memory() == [(0, 4000, 10240)]
        assert LlamaCppBackend._child_visibility_for([0]) == "MIG-dddd4444-0"

    def test_the_child_pin_belongs_to_no_probe(self, monkeypatch, probe_script):
        """The uuid each index stands for is derived from the last NVML inventory and the
        mask in force when the launch asks, so another query answering in between (nvidia-smi
        recovering, a concurrent preflight) neither blanks it nor leaves a stale one."""
        _failing_smi(monkeypatch)
        slice_row = dict(_row(0, 9000, 20480, uuid = "MIG-cccc3333-0"), mig = "1")
        probe_script(
            _payload(
                [
                    _row(0, 60000, 81920, uuid = "GPU-aaaa1111-0"),
                    _row(1, 20000, uuid = "GPU-bbbb2222-1"),
                    slice_row,
                ]
            )
        )
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb,GPU-aaaa")
        assert LlamaCppBackend._get_gpu_memory() == [(0, 9000, 20480), (1, 20000, 24576)]

        def smi_ok(cmd, *args, **kwargs):
            if cmd and os.path.basename(str(cmd[0])) == "nvidia-smi":
                return types.SimpleNamespace(
                    returncode = 0, stdout = "0, 60000, 81920\n1, 20000, 24576\n", stderr = ""
                )
            raise AssertionError("no other probe should run")

        monkeypatch.setattr(mod.subprocess, "run", smi_ok)
        assert LlamaCppBackend._get_gpu_memory() == [(0, 60000, 81920), (1, 20000, 24576)]
        # The inherited uuid mask is still what the child gets, a MIG parent still its slice.
        assert LlamaCppBackend._child_visibility_for([1, 0]) == "GPU-bbbb2222-1,MIG-cccc3333-0"
        # The mask changing underneath is read as it is now, not as it was at the probe.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        assert LlamaCppBackend._child_visibility_for([1]) == "1"
        assert LlamaCppBackend._child_visibility_for([0]) == "MIG-cccc3333-0"
        # No NVML inventory at all: indices as before.
        monkeypatch.setattr(LlamaCppBackend, "_NVML_ROWS", [])
        assert LlamaCppBackend._child_visibility_for([0, 1]) == "0,1"

    def test_a_uuid_mask_is_handed_to_the_child_as_uuids(self, monkeypatch, probe_script):
        _failing_smi(monkeypatch)
        slice_row = dict(_row(0, 9000, 20480, uuid = "MIG-cccc3333-0"), mig = "1")
        probe_script(
            _payload(
                [
                    _row(0, 60000, 81920, uuid = "GPU-aaaa1111-0"),
                    _row(1, 20000, uuid = "GPU-bbbb2222-1"),
                    slice_row,
                ]
            )
        )
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-cccc")
        assert LlamaCppBackend._get_gpu_memory() == [(0, 9000, 20480)]
        # The launch must not turn the slice into its parent's index.
        assert LlamaCppBackend._child_visibility_for([0]) == "MIG-cccc3333-0"
        # The GPU- entry names the MIG parent, which CUDA exposes as its first slice.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb,GPU-aaaa")
        assert LlamaCppBackend._get_gpu_memory() == [(0, 9000, 20480), (1, 20000, 24576)]
        assert LlamaCppBackend._child_visibility_for([1, 0]) == "GPU-bbbb2222-1,MIG-cccc3333-0"
        # A numeric or absent mask re-emits indices as before; only a slice standing in
        # for its MIG parent is named by uuid, and a selection mixing both stays numeric.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        assert LlamaCppBackend._get_gpu_memory() == [(0, 9000, 20480), (1, 20000, 24576)]
        assert LlamaCppBackend._child_visibility_for([1]) == "1"
        assert LlamaCppBackend._child_visibility_for([0]) == "MIG-cccc3333-0"
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        LlamaCppBackend._get_gpu_memory()
        assert LlamaCppBackend._child_visibility_for([0, 1]) == "0,1"

    def test_rows_without_a_memory_reading_are_not_evidence(self, monkeypatch, probe_script):
        _failing_smi(monkeypatch)
        probe_script(_payload([_row(0, 0, 0)], source = "cuda"))
        assert LlamaCppBackend._get_gpu_memory() == []
        probe_script(_payload([_row(0, 0, 0)]))
        assert LlamaCppBackend._get_gpu_memory() == []
        # One visible GPU without a reading voids the answer rather than shrinking the host;
        # a full card (free 0, a total) is a reading and stays listed.
        probe_script(_payload([_row(0, 8000), _row(1, 0, 0)]))
        assert LlamaCppBackend._get_gpu_memory() == []
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        assert LlamaCppBackend._get_gpu_memory() == [(0, 8000, 24576)]
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        probe_script(_payload([_row(0, 8000), _row(1, 0)]))
        assert LlamaCppBackend._get_gpu_memory() == [(0, 8000, 24576), (1, 0, 24576)]

    def test_a_working_nvidia_smi_is_not_second_guessed(self, monkeypatch, probe_script):
        probe_script(_payload([_row(5, 1)]))
        real_run = mod.subprocess.run

        def run(cmd, *args, **kwargs):
            if cmd and os.path.basename(str(cmd[0])) == "nvidia-smi":
                return types.SimpleNamespace(returncode = 0, stdout = "0, 4096, 8192\n", stderr = "")
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(mod.subprocess, "run", run)
        assert LlamaCppBackend._get_gpu_memory() == [(0, 4096, 8192)]

    def test_the_switch_and_a_missing_script_turn_it_off(self, monkeypatch, probe_script):
        _failing_smi(monkeypatch)
        probe_script(_payload([_row(0, 8000)]))
        monkeypatch.setenv("UNSLOTH_NVIDIA_LIBRARY_PROBE", "0")
        assert LlamaCppBackend._get_gpu_memory() == []
        monkeypatch.delenv("UNSLOTH_NVIDIA_LIBRARY_PROBE")
        monkeypatch.setenv("UNSLOTH_NVIDIA_PROBE", "/nonexistent/nvidia_probe.py")
        monkeypatch.setattr(LlamaCppBackend, "_nvidia_probe_script", staticmethod(lambda: None))
        assert LlamaCppBackend._get_gpu_memory() == []

    def test_the_real_script_is_found_next_to_the_installers(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_NVIDIA_PROBE", raising = False)
        script = LlamaCppBackend._nvidia_probe_script()
        assert script is not None and script.name == "nvidia_probe.py"
        assert (script.parent / "install_llama_prebuilt.py").is_file()


class TestTheEmbeddingServerKeepsTheGpu:
    def test_embeddings_offload_when_only_nvml_answers(self, monkeypatch, probe_script):
        _failing_smi(monkeypatch)
        probe_script(_payload([_row(0, 8000)]))
        import utils.hardware as hardware

        monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)  # Metal answers on a Mac
        monkeypatch.setattr(embed_mod.config, "embed_device_preference", lambda: "auto")
        monkeypatch.setattr(
            LlamaCppBackend, "_arch_gate_survivors", staticmethod(lambda binary: []), raising = False
        )
        server = embed_mod.LlamaServerBackend.__new__(embed_mod.LlamaServerBackend)
        server._force_cpu = False
        assert server._use_gpu() is True
        cmd = server._build_cmd("/opt/llama-server", "m.gguf", 9999, use_gpu = True)
        assert cmd[-2:] == ["-ngl", "-1"]
        # And the misread this closes: nothing answering pins the server to the CPU.
        probe_script(_payload([]))
        assert server._use_gpu() is False

    def test_windows_embeddings_get_the_same_dll_search_path_as_chat(self, monkeypatch, tmp_path):
        monkeypatch.setattr(embed_mod.sys, "platform", "win32")
        monkeypatch.setattr(mod.sys, "platform", "win32")
        monkeypatch.setattr(
            embed_mod, "child_env_without_native_path_secret", lambda: {"PATH": "inherited"}
        )
        monkeypatch.setattr(LlamaCppBackend, "_sanitize_p2p_env", staticmethod(lambda env: None))
        monkeypatch.setattr(
            LlamaCppBackend, "_arch_gate_survivors", staticmethod(lambda b: []), raising = False
        )
        monkeypatch.setattr(
            LlamaCppBackend,
            "_build_windows_path_dirs",
            staticmethod(
                lambda binary_dir, prefix, cuda_path: [binary_dir, "C:\\venv\\nvidia\\cu13\\bin"]
            ),
        )
        monkeypatch.setattr(mod, "_llama_lib_dir", lambda binary: Path("C:/llama/build/bin"))
        server = embed_mod.LlamaServerBackend.__new__(embed_mod.LlamaServerBackend)
        env = server._build_env("C:/llama/llama-server.exe", use_gpu = True)
        assert env["PATH"].split(";")[:2] == [
            str(Path("C:/llama/build/bin")),
            "C:\\venv\\nvidia\\cu13\\bin",
        ]
        assert env["PATH"].endswith(";inherited")


class TestAGpuCapableBuildIsPreferred:
    """#5941: a CPU build/ beside a CUDA build-cuda/ ran on the CPU forever."""

    @staticmethod
    def _tree(root, platform, layouts):
        from utils.llama_cpp_path_settings import llama_server_binary_name

        name = llama_server_binary_name(platform)
        made = {}
        for build, libs in layouts.items():
            bindir = root / build / "bin" / ("Release" if platform == "win32" else "")
            bindir.mkdir(parents = True, exist_ok = True)
            exe = bindir / name
            exe.write_text("#!/bin/sh\n", encoding = "utf-8")
            exe.chmod(0o755)
            for lib in libs:
                (bindir / lib).write_bytes(b"")
            made[build] = exe
        return made

    @pytest.mark.parametrize("platform", ["linux", "win32"])
    def test_the_cuda_sibling_build_wins_over_a_cpu_only_first_hit(
        self, tmp_path, platform, monkeypatch
    ):
        from utils import llama_cpp_path_settings as ps

        monkeypatch.setattr(ps, "host_gpu_vendors", lambda: None)  # not this host's vendors
        so = ".dll" if platform == "win32" else ".so"
        pre = "" if platform == "win32" else "lib"
        made = self._tree(
            tmp_path,
            platform,
            {
                "build": [f"{pre}ggml-cpu{so}", f"{pre}ggml-base{so}"],
                "build-cuda": [f"{pre}ggml-cuda{so}", f"{pre}ggml-cpu{so}", f"{pre}ggml-base{so}"],
            },
        )
        assert ps.resolve_llama_server_binary(tmp_path, platform = platform) == made["build-cuda"]
        assert ps.binary_gpu_verdict(made["build"]) == "cpu"
        assert ps.binary_gpu_verdict(made["build-cuda"]) == "gpu"

    def test_a_stale_cuda_build_does_not_shadow_the_build_this_host_can_run(
        self, tmp_path, monkeypatch
    ):
        from utils import llama_cpp_path_settings as ps

        made = self._tree(
            tmp_path,
            "linux",
            {
                "build": ["libggml-cpu.so", "libggml-base.so"],
                "build-cuda": ["libggml-cuda.so", "libggml-cpu.so"],
                "build-hip": ["libggml-hip.so", "libggml-cpu.so"],
                "build-vulkan": ["libggml-vulkan.so", "libggml-cpu.so"],
            },
        )
        for vendors, winner in (
            ({"amd"}, "build-hip"),
            ({"nvidia"}, "build-cuda"),
            (None, "build-cuda"),
        ):
            monkeypatch.setattr(ps, "host_gpu_vendors", lambda v = vendors: v)
            assert (
                ps.resolve_llama_server_binary(tmp_path, platform = "linux") == made[winner]
            ), vendors
        # A vendor no build targets still gets the vendor-agnostic Vulkan build.
        monkeypatch.setattr(ps, "host_gpu_vendors", lambda: {"intel"})
        assert ps.resolve_llama_server_binary(tmp_path, platform = "linux") == made["build-vulkan"]

    def test_host_vendors_read_the_drm_sysfs(self, tmp_path, monkeypatch):
        from utils import llama_cpp_path_settings as ps

        monkeypatch.setattr(ps.sys, "platform", "linux")
        monkeypatch.setattr(ps.os.path, "isdir", lambda p: False)
        nodes = {"/dev/nvidiactl", "/dev/kfd"}
        monkeypatch.setattr(ps.os.path, "exists", lambda p: p in nodes)
        monkeypatch.setattr(ps, "_DRM_ROOT", str(tmp_path))
        for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            monkeypatch.delenv(var, raising = False)
        assert ps.host_gpu_vendors() is None
        for card, vendor in (("card0", "0x1002"), ("card1", "0x10de"), ("card1-DP-1", "0x10de")):
            (tmp_path / card / "device").mkdir(parents = True)
            (tmp_path / card / "device" / "vendor").write_text(vendor + "\n", encoding = "utf-8")
        assert ps.host_gpu_vendors() == {"amd", "nvidia"}
        # A container exposing only the AMD device node, or a mask hiding NVIDIA: not runnable.
        nodes.discard("/dev/nvidiactl")
        assert ps.host_gpu_vendors() == {"amd"}
        nodes.add("/dev/nvidiactl")
        # HIP reads CUDA_VISIBLE_DEVICES when neither HIP_ nor ROCR_ is set, so an empty one
        # hides both vendors: detected but nothing reachable is an empty set, not unknown,
        # and only Vulkan fits then. A set HIP_ mask outranks it.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert ps.host_gpu_vendors() == set()
        assert ps._fits_host({"cuda"}, set()) is False and ps._fits_host({"vulkan"}, set()) is True
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        assert ps.host_gpu_vendors() == {"amd"}
        # The two ROCm masks stack: an empty ROCR_ under a set HIP_ still hides every agent.
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
        assert ps.host_gpu_vendors() == set()
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
        assert ps.host_gpu_vendors() == {"amd"}
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "-1")
        assert ps.host_gpu_vendors() == set()
        assert ps._fits_host({"cuda"}, set()) is False and ps._fits_host({"vulkan"}, set()) is True

    def test_host_vendors_read_the_wsl_runtimes(self, tmp_path, monkeypatch):
        """WSL2 has no DRM card and no /dev/kfd or /dev/nvidiactl: the GPU is behind /dev/dxg
        and the vendor is the runtime that drives it."""
        from utils import llama_cpp_path_settings as ps

        monkeypatch.setattr(ps.sys, "platform", "linux")
        monkeypatch.setattr(ps.os.path, "isdir", lambda p: False)
        monkeypatch.setattr(ps, "_DRM_ROOT", str(tmp_path / "drm"))
        rocm = tmp_path / "rocm"
        wsl_lib = tmp_path / "wsl"
        monkeypatch.setattr(ps, "_WSL_ROCM_LIB_DIRS", (str(rocm),))
        monkeypatch.setattr(ps, "_WSL_CUDA_LIB_DIR", str(wsl_lib))
        nodes = {"/dev/dxg"}
        monkeypatch.setattr(ps.os.path, "exists", lambda p: p in nodes or os.path.lexists(p))
        for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            monkeypatch.delenv(var, raising = False)
        assert ps.host_gpu_vendors() is None
        rocm.mkdir()
        (rocm / "librocdxg.so.1").write_bytes(b"")
        assert ps.host_gpu_vendors() == {"amd"}
        wsl_lib.mkdir()
        (wsl_lib / "libcuda.so.1.1").write_bytes(b"")
        assert ps.host_gpu_vendors() == {"amd", "nvidia"}
        # The masks still apply, and without /dev/dxg the runtimes prove nothing.
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
        assert ps.host_gpu_vendors() == {"nvidia"}
        nodes.discard("/dev/dxg")
        assert ps.host_gpu_vendors() is None

    def test_a_gpu_first_hit_for_another_vendor_yields_too(self, tmp_path, monkeypatch):
        from utils import llama_cpp_path_settings as ps

        # No build/ at all: a stale build-cuda/ is the first hit on an AMD box.
        made = self._tree(
            tmp_path, "linux", {"build-cuda": ["libggml-cuda.so"], "build-hip": ["libggml-hip.so"]}
        )
        monkeypatch.setattr(ps, "host_gpu_vendors", lambda: {"amd"})
        assert ps.resolve_llama_server_binary(tmp_path, platform = "linux") == made["build-hip"]
        # On the NVIDIA box, or an unknown host, the first hit stands.
        for vendors in ({"nvidia"}, None):
            monkeypatch.setattr(ps, "host_gpu_vendors", lambda v = vendors: v)
            assert ps.resolve_llama_server_binary(tmp_path, platform = "linux") == made["build-cuda"]

    def test_a_first_hit_of_unknown_layout_keeps_its_place(self, tmp_path):
        from utils import llama_cpp_path_settings as ps
        made = self._tree(tmp_path, "linux", {"build": [], "build-cuda": ["libggml-cuda.so"]})
        # A static build or the installer's wrapper: not proven CPU-only, so search order holds.
        assert ps.resolve_llama_server_binary(tmp_path, platform = "linux") == made["build"]

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0, reason = "needs a real permission denial"
    )
    def test_a_denied_candidate_ahead_of_the_gpu_build_still_stops_discovery(
        self, tmp_path, monkeypatch
    ):
        made = self._tree(
            tmp_path, "linux", {"build": ["libggml-cpu.so"], "build-cuda": ["libggml-cuda.so"]}
        )
        monkeypatch.setattr(LlamaCppBackend, "_find_llama_server_binary", _REAL_FINDER)
        monkeypatch.setattr(mod.sys, "platform", "linux")
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
        monkeypatch.delenv("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH", raising = False)
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path))
        # An in-flight replace or an ACL: the pinned layout's build/ cannot be read at all.
        (tmp_path / "build" / "bin").chmod(0)
        try:
            assert LlamaCppBackend._find_llama_server_binary() is None
        finally:
            (tmp_path / "build" / "bin").chmod(0o755)
        assert LlamaCppBackend._find_llama_server_binary() == str(made["build-cuda"])

    def test_the_runtime_finder_agrees(self, tmp_path, monkeypatch):
        made = self._tree(
            tmp_path,
            "linux",
            {"build": ["libggml-cpu.so", "libggml-base.so"], "build-cuda": ["libggml-cuda.so"]},
        )
        monkeypatch.setattr(LlamaCppBackend, "_find_llama_server_binary", _REAL_FINDER)
        monkeypatch.setattr(mod.sys, "platform", "linux")
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
        monkeypatch.delenv("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH", raising = False)
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path))
        assert LlamaCppBackend._find_llama_server_binary() == str(made["build-cuda"])

    def test_the_legacy_in_tree_build_prefers_the_gpu_build_too(self, tmp_path, monkeypatch):
        """No managed or custom runtime configured: the project_root/llama.cpp fallback."""
        from utils import llama_cpp_path_settings as ps

        made = self._tree(
            tmp_path / "llama.cpp",
            "linux",
            {"build": ["libggml-cpu.so", "libggml-base.so"], "build-cuda": ["libggml-cuda.so"]},
        )
        monkeypatch.setattr(LlamaCppBackend, "_find_llama_server_binary", _REAL_FINDER)
        monkeypatch.setattr(mod.sys, "platform", "linux")
        monkeypatch.setattr(mod, "__file__", str(tmp_path / "s" / "b" / "c" / "i" / "llama_cpp.py"))
        for var in (
            "LLAMA_SERVER_PATH",
            "UNSLOTH_LLAMA_CPP_PATH",
            "UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH",
        ):
            monkeypatch.delenv(var, raising = False)
        monkeypatch.setattr(ps, "get_stored_custom_llama_cpp_path", lambda: None)
        monkeypatch.setattr(
            LlamaCppBackend,
            "_resolved_studio_root_and_is_legacy",
            staticmethod(lambda: (tmp_path / "no-such-studio", False)),
        )
        monkeypatch.setattr(mod.shutil, "which", lambda name: None)
        assert LlamaCppBackend._find_llama_server_binary() == str(made["build-cuda"])


@pytest.mark.skipif(
    sys.platform == "win32", reason = "the Linux library path is built from posix paths"
)
class TestTheLinuxLibrarySearchPath:
    @pytest.mark.parametrize("prefix_name", ["Studio", "Studio[CUDA]"])
    def test_a_bracketed_prefix_and_torch_lib_are_found(self, monkeypatch, tmp_path, prefix_name):
        prefix = tmp_path / prefix_name
        site = prefix / "lib" / "python3.12" / "site-packages"
        cu = site / "nvidia" / "cu13" / "lib"
        torch_lib = site / "torch" / "lib"
        cu.mkdir(parents = True)
        torch_lib.mkdir(parents = True)
        monkeypatch.setattr(embed_mod.sys, "platform", "linux")
        monkeypatch.setattr(embed_mod.sys, "prefix", str(prefix))
        env: dict[str, str] = {}
        embed_mod.LlamaServerBackend._add_linux_cuda_libs(env, "/opt/llama")
        parts = env["LD_LIBRARY_PATH"].split(":")
        assert str(cu) in parts and str(torch_lib) in parts
        # The chat server's builder reads the same prefix the same way.
        monkeypatch.setattr(mod.sys, "platform", "linux")
        monkeypatch.setattr(mod.sys, "prefix", str(prefix))
        monkeypatch.setattr(mod, "_llama_lib_dir", lambda binary: Path("/opt/llama"))
        monkeypatch.setattr(mod, "_wsl_system_rocm_lib_dirs", lambda: [])
        monkeypatch.setattr(mod, "_native_linux_system_rocm_lib_dirs", lambda binary_dir: [])
        monkeypatch.setattr(
            LlamaCppBackend, "_prefers_bundle_only_rocm", staticmethod(lambda b: True)
        )
        chat_env = LlamaCppBackend._llama_server_env_for_binary("/opt/llama/llama-server")
        chat_parts = chat_env["LD_LIBRARY_PATH"].split(":")
        assert str(cu) in chat_parts and str(torch_lib) in chat_parts
