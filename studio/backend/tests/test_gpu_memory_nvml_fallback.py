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
    return {"source": source, "cuda_driver_version": [13, 0], "driver_version": "580.65.06", "devices": rows}


def _row(index, free, total = 24576, uuid = None):
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
    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory_amd_smi", staticmethod(lambda *a, **k: []))
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

    def test_rows_without_a_memory_reading_are_not_evidence(self, monkeypatch, probe_script):
        _failing_smi(monkeypatch)
        probe_script(_payload([_row(0, 0, 0)], source = "cuda"))
        assert LlamaCppBackend._get_gpu_memory() == []
        probe_script(_payload([_row(0, 0, 0)]))
        assert LlamaCppBackend._get_gpu_memory() == []

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
        monkeypatch.setattr(embed_mod, "child_env_without_native_path_secret", lambda: {"PATH": "inherited"})
        monkeypatch.setattr(LlamaCppBackend, "_sanitize_p2p_env", staticmethod(lambda env: None))
        monkeypatch.setattr(LlamaCppBackend, "_arch_gate_survivors", staticmethod(lambda b: []), raising = False)
        monkeypatch.setattr(
            LlamaCppBackend,
            "_build_windows_path_dirs",
            staticmethod(lambda binary_dir, prefix, cuda_path: [binary_dir, "C:\\venv\\nvidia\\cu13\\bin"]),
        )
        monkeypatch.setattr(mod, "_llama_lib_dir", lambda binary: Path("C:/llama/build/bin"))
        server = embed_mod.LlamaServerBackend.__new__(embed_mod.LlamaServerBackend)
        env = server._build_env("C:/llama/llama-server.exe", use_gpu = True)
        assert env["PATH"].split(";")[:2] == [str(Path("C:/llama/build/bin")), "C:\\venv\\nvidia\\cu13\\bin"]
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
    def test_the_cuda_sibling_build_wins_over_a_cpu_only_first_hit(self, tmp_path, platform):
        from utils import llama_cpp_path_settings as ps

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

    def test_a_first_hit_of_unknown_layout_keeps_its_place(self, tmp_path):
        from utils import llama_cpp_path_settings as ps

        made = self._tree(tmp_path, "linux", {"build": [], "build-cuda": ["libggml-cuda.so"]})
        # A static build or the installer's wrapper: not proven CPU-only, so search order holds.
        assert ps.resolve_llama_server_binary(tmp_path, platform = "linux") == made["build"]

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
        monkeypatch.setattr(LlamaCppBackend, "_prefers_bundle_only_rocm", staticmethod(lambda b: True))
        chat_env = LlamaCppBackend._llama_server_env_for_binary("/opt/llama/llama-server")
        chat_parts = chat_env["LD_LIBRARY_PATH"].split(":")
        assert str(cu) in chat_parts and str(torch_lib) in chat_parts
