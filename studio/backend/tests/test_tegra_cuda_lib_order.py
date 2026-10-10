# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""On a Jetson, JetPack's CUDA goes ahead of the generic pip CUDA builds in every loader path a
llama-server child or the backend itself gets (#4862); every other host keeps today's order."""

import os
import sys

import pytest

import run
from core.inference import llama_cpp
from core.rag import embed_llama_server
from utils import tegra

SYSTEM = ("/usr/local/cuda/lib64", "/usr/local/cuda/targets/aarch64-linux/lib", tegra.TEGRA_LIB_DIR)


@pytest.fixture
def host(monkeypatch, tmp_path):
    """A Linux aarch64 host whose venv carries pip CUDA and torch libs; returns a Tegra switch."""
    site = tmp_path / "lib" / "python3.12" / "site-packages"
    pip_dirs = [site / "nvidia" / "cu13" / "lib", site / "nvidia" / "cudnn" / "lib", site / "torch" / "lib"]
    for d in pip_dirs:
        d.mkdir(parents = True)
    (site / "torch" / "__init__.py").write_text("")
    real_isdir = os.path.isdir
    monkeypatch.setattr(os.path, "isdir", lambda p: str(p) in SYSTEM or real_isdir(p))
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising = False)
    state = {"tegra": False, "site": site}

    def set_tegra(value):
        state["tegra"] = value
        monkeypatch.setattr(tegra, "is_tegra", lambda: state["tegra"])
        monkeypatch.setattr(run, "_is_tegra", lambda: state["tegra"], raising = False)

    set_tegra(False)
    return set_tegra, state


def _first_system_and_pip(paths):
    sys_pos = min(i for i, p in enumerate(paths) if p.startswith("/usr/local/cuda"))
    pip_pos = min(i for i, p in enumerate(paths) if "site-packages" in p)
    return sys_pos, pip_pos


def _llama_paths(monkeypatch, tmp_path):
    binary = tmp_path / "bin" / "llama-server"
    binary.parent.mkdir()
    binary.write_text("")
    monkeypatch.setattr(llama_cpp, "_wsl_system_rocm_lib_dirs", lambda: [])
    monkeypatch.setattr(llama_cpp, "_native_linux_system_rocm_lib_dirs", lambda d: [])
    env = llama_cpp.LlamaCppBackend._llama_server_env_for_binary(str(binary), use_system_rocm = False)
    return env["LD_LIBRARY_PATH"].split(":")


def _embed_paths(tmp_path):
    env = {}
    embed_llama_server.LlamaServerBackend._add_linux_cuda_libs(env, str(tmp_path / "bin"))
    return env["LD_LIBRARY_PATH"].split(":")


def _run_paths(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")
    run._fix_torch_cuda_ld_path()
    return os.environ["LD_LIBRARY_PATH"].split(":")


def test_jetson_puts_jetpack_cuda_ahead_of_pip_cuda_for_llama_server(host, monkeypatch, tmp_path):
    set_tegra, _ = host
    set_tegra(True)
    sys_pos, pip_pos = _first_system_and_pip(_llama_paths(monkeypatch, tmp_path))
    assert sys_pos < pip_pos


def test_jetson_puts_jetpack_cuda_ahead_of_pip_cuda_for_the_embedding_server(host, tmp_path):
    set_tegra, _ = host
    set_tegra(True)
    paths = _embed_paths(tmp_path)
    sys_pos, pip_pos = _first_system_and_pip(paths)
    assert sys_pos < pip_pos
    assert paths.index(tegra.TEGRA_LIB_DIR) < sys_pos


def test_jetson_backend_does_not_hoist_pip_cuda_over_jetpack(host, monkeypatch):
    set_tegra, state = host
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: type("S", (), {"origin": str(state["site"] / "torch" / "__init__.py")})(),
    )
    set_tegra(True)
    paths = _run_paths(monkeypatch)
    assert not any("nvidia" in p for p in paths[: paths.index("/usr/local/cuda/lib64")])


@pytest.mark.parametrize("machine", ["aarch64", "x86_64"])
def test_other_hosts_keep_pip_cuda_first(host, monkeypatch, tmp_path, machine):
    monkeypatch.setattr("platform.machine", lambda: machine)
    sys_pos, pip_pos = _first_system_and_pip(_llama_paths(monkeypatch, tmp_path))
    assert pip_pos < sys_pos
    sys_pos, pip_pos = _first_system_and_pip(_embed_paths(tmp_path))
    assert pip_pos < sys_pos
    assert tegra.TEGRA_LIB_DIR not in _embed_paths(tmp_path)


def test_other_hosts_still_hoist_torch_cuda_in_the_backend(host, monkeypatch):
    _, state = host
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: type("S", (), {"origin": str(state["site"] / "torch" / "__init__.py")})(),
    )
    paths = _run_paths(monkeypatch)
    assert paths.index(str(state["site"] / "nvidia" / "cu13" / "lib")) < paths.index("/usr/local/cuda/lib64")


def test_is_tegra_reads_the_markers_and_never_raises(monkeypatch, tmp_path):
    release = tmp_path / "nv_tegra_release"
    compatible = tmp_path / "compatible"
    monkeypatch.setattr(tegra, "_TEGRA_RELEASE", str(release))
    monkeypatch.setattr(tegra, "_DEVICE_TREE_COMPATIBLE", str(compatible))
    tegra.is_tegra.cache_clear()
    assert tegra.is_tegra() is False  # neither marker present
    compatible.write_bytes(b"nvidia,p3737-0000+p3701-0005\x00nvidia,tegra234\x00")
    tegra.is_tegra.cache_clear()
    assert tegra.is_tegra() is True
    compatible.write_bytes(b"raspberrypi,5-model-b\x00brcm,bcm2712\x00")
    tegra.is_tegra.cache_clear()
    assert tegra.is_tegra() is False
    release.write_text("# R36 (release), REVISION: 4.0\n")
    tegra.is_tegra.cache_clear()
    assert tegra.is_tegra() is True
    tegra.is_tegra.cache_clear()
