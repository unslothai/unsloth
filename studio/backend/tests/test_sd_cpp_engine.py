# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the sd-cli engine + routing (``sd_cpp_engine.py``).

Hermetic: the binary finder is driven against a tmp filesystem, and ``generate``
runs a fake ``subprocess.Popen`` that emits canned lines and writes the output
PNG -- no real ``sd-cli``, no GPU.
"""

from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import pytest

from core.inference import sd_cpp_engine as eng
from core.inference.sd_cpp_engine import (
    ENGINE_DIFFUSERS,
    ENGINE_SD_CPP,
    SdCppEngine,
    find_sd_cpp_binary,
    find_sd_server_binary,
    runtime_env,
    select_diffusion_engine,
)
from core.inference.sd_cpp_args import SdCppGenParams, SdCppModelFiles, SdCppUpscaleParams


# ── binary discovery ────────────────────────────────────────────────────────


def _clear_env(monkeypatch):
    monkeypatch.delenv("SD_CLI_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_SD_CPP_PATH", raising = False)


def test_find_prefers_sd_cli_path_env(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    binary = tmp_path / "sd-cli"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv("SD_CLI_PATH", str(binary))
    # even with PATH empty, the direct env wins
    monkeypatch.setattr(eng.shutil, "which", lambda *_a: None)
    assert find_sd_cpp_binary() == str(binary)


def test_find_custom_install_dir_build_layout(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    root = tmp_path / "sdcpp"
    built = root / "build" / "bin" / "sd-cli"
    built.parent.mkdir(parents = True)
    built.write_text("x")
    monkeypatch.setenv("UNSLOTH_SD_CPP_PATH", str(root))
    monkeypatch.setattr(eng.shutil, "which", lambda *_a: None)
    assert find_sd_cpp_binary() == str(built)


def test_find_falls_back_to_path(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(eng.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setattr(
        eng.shutil, "which", lambda stem: "/usr/bin/sd-cli" if stem == "sd-cli" else None
    )
    assert find_sd_cpp_binary() == "/usr/bin/sd-cli"


def test_find_returns_none_when_absent(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(eng.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setattr(eng.shutil, "which", lambda *_a: None)
    assert find_sd_cpp_binary() is None


# ── sd-server discovery ──────────────────────────────────────────────────────


def _clear_server_env(monkeypatch):
    monkeypatch.delenv("SD_SERVER_PATH", raising = False)
    monkeypatch.delenv("SD_CLI_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_SD_CPP_PATH", raising = False)


def test_find_server_prefers_sd_server_path_env(tmp_path, monkeypatch):
    _clear_server_env(monkeypatch)
    binary = tmp_path / "sd-server"
    binary.write_text("x")
    monkeypatch.setenv("SD_SERVER_PATH", str(binary))
    monkeypatch.setattr(eng.shutil, "which", lambda *_a: None)
    assert find_sd_server_binary() == str(binary)


def test_find_server_build_layout(tmp_path, monkeypatch):
    _clear_server_env(monkeypatch)
    root = tmp_path / "sdcpp"
    built = root / "build" / "bin" / "sd-server"
    built.parent.mkdir(parents = True)
    built.write_text("x")
    monkeypatch.setenv("UNSLOTH_SD_CPP_PATH", str(root))
    monkeypatch.setattr(eng.shutil, "which", lambda *_a: None)
    assert find_sd_server_binary() == str(built)


def test_find_server_path_fallback(tmp_path, monkeypatch):
    _clear_server_env(monkeypatch)
    monkeypatch.setattr(eng.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setattr(
        eng.shutil, "which", lambda stem: "/usr/bin/sd-server" if stem == "sd-server" else None
    )
    assert find_sd_server_binary() == "/usr/bin/sd-server"


def test_find_server_not_confused_with_sd_cli(tmp_path, monkeypatch):
    # A tree that has only sd-cli must NOT be reported as an sd-server (and vice versa),
    # so the backend correctly falls back to one-shot when only the CLI is present.
    _clear_server_env(monkeypatch)
    root = tmp_path / "sdcpp"
    (root / "build" / "bin").mkdir(parents = True)
    (root / "build" / "bin" / "sd-cli").write_text("x")
    monkeypatch.setenv("UNSLOTH_SD_CPP_PATH", str(root))
    monkeypatch.setattr(eng.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setattr(eng.shutil, "which", lambda *_a: None)
    assert find_sd_server_binary() is None
    assert find_sd_cpp_binary() == str(root / "build" / "bin" / "sd-cli")


# ── availability / version ──────────────────────────────────────────────────


def test_engine_unavailable_when_no_binary(monkeypatch):
    # Force the "no binary anywhere" condition so the test is hermetic even on a host
    # that happens to have sd-cli installed.
    monkeypatch.setattr(eng, "find_sd_cpp_binary", lambda: None)
    e = SdCppEngine(binary = None)
    assert e.is_available() is False
    assert e.version() is None


def test_engine_version_parsed_and_cached(tmp_path, monkeypatch):
    binary = tmp_path / "sd-cli"
    binary.write_text("x")
    e = SdCppEngine(binary = str(binary))
    calls = {"n": 0}

    def _fake_run(*_a, **_k):
        calls["n"] += 1
        return types.SimpleNamespace(
            stdout = "stable-diffusion.cpp version master-721\n", stderr = "", returncode = 0
        )

    monkeypatch.setattr(eng.subprocess, "run", _fake_run)
    assert e.version() == "stable-diffusion.cpp version master-721"
    assert e.version() == "stable-diffusion.cpp version master-721"
    assert calls["n"] == 1  # cached after the first probe


# ── runtime env (bundled shared libs) ───────────────────────────────────────


def test_runtime_env_prepends_binary_dir_to_lib_path():
    var = eng._lib_path_var()
    env = runtime_env("/opt/sdcpp/bin/sd-cli", {var: "/existing"})
    first = env[var].split(os.pathsep)[0]
    assert first == "/opt/sdcpp/bin"
    assert "/existing" in env[var]


def test_runtime_env_handles_missing_lib_path():
    var = eng._lib_path_var()
    env = runtime_env("/opt/sdcpp/bin/sd-cli", {})
    assert env[var] == "/opt/sdcpp/bin"


# ── generate (fake subprocess) ──────────────────────────────────────────────


class _FakePopen:
    """Stand-in for subprocess.Popen: streams ``lines`` then writes ``out_file``
    (unless ``write`` is False) and exits with ``returncode``."""

    captured_cmd: list[str] = []
    captured_env: dict = {}

    def __init__(
        self,
        cmd,
        *,
        lines,
        returncode,
        out_file,
        write,
        env = None,
    ):
        type(self).captured_cmd = list(cmd)
        type(self).captured_env = dict(env or {})
        self._lines = list(lines)
        self.returncode = returncode
        self._out_file = out_file
        self._write = write

    @property
    def stdout(self):
        return iter(self._lines)

    def wait(self, timeout = None):
        if self._write:
            Path(self._out_file).write_bytes(b"\x89PNG\r\n")
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        pass


def _patch_popen(
    monkeypatch,
    *,
    lines,
    returncode,
    out_file,
    write = True,
):
    def _factory(cmd, **kw):
        return _FakePopen(
            cmd,
            lines = lines,
            returncode = returncode,
            out_file = out_file,
            write = write,
            env = kw.get("env"),
        )

    monkeypatch.setattr(eng.subprocess, "Popen", _factory)


def _engine(tmp_path):
    binary = tmp_path / "sd-cli"
    binary.write_text("x")
    return SdCppEngine(binary = str(binary))


def test_generate_success_returns_path_and_collects_logs(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    _patch_popen(
        monkeypatch, lines = ["loading model", "step 1/8", "done"], returncode = 0, out_file = out
    )
    seen: list[str] = []
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf", vae = "/m/ae.sft", llm = "/m/q.gguf")
    params = SdCppGenParams(prompt = "a cat", steps = 8, seed = 1)

    result = e.generate(files, params, output_path = str(out), on_log = seen.append)

    assert result == out and out.is_file()
    assert seen == ["loading model", "step 1/8", "done"]
    # the real argv was built and handed to Popen
    assert "--diffusion-model" in _FakePopen.captured_cmd
    assert str(out) == _FakePopen.captured_cmd[_FakePopen.captured_cmd.index("--output") + 1]
    # the subprocess env carries the binary's dir on the library path
    var = eng._lib_path_var()
    assert str(Path(e.binary).resolve().parent) in _FakePopen.captured_env.get(var, "")


def test_generate_raises_on_nonzero_exit(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    _patch_popen(monkeypatch, lines = ["boom: bad gguf"], returncode = 1, out_file = out, write = False)
    with pytest.raises(RuntimeError, match = "exited 1"):
        e.generate(
            SdCppModelFiles(diffusion_model = "/m/z.gguf"),
            SdCppGenParams(prompt = "x"),
            output_path = str(out),
        )


def test_generate_raises_when_no_output_despite_success(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    _patch_popen(monkeypatch, lines = ["ok"], returncode = 0, out_file = out, write = False)
    with pytest.raises(RuntimeError, match = "no image"):
        e.generate(
            SdCppModelFiles(diffusion_model = "/m/z.gguf"),
            SdCppGenParams(prompt = "x"),
            output_path = str(out),
        )


def test_generate_raises_when_binary_missing():
    e = SdCppEngine(binary = None)
    with pytest.raises(RuntimeError, match = "not found"):
        e.generate(
            SdCppModelFiles(diffusion_model = "/m/z.gguf"),
            SdCppGenParams(prompt = "x"),
            output_path = "/tmp/x.png",
        )


class _HangingPopen:
    """A child that runs but never prints and never exits -- the case a plain
    `for line in stdout` would block on forever, ignoring the timeout."""

    def __init__(self, cmd, **_kw):
        self._alive = True

    class _Blocking:
        def __init__(self, owner):
            self.owner = owner

        def __iter__(self):
            return self

        def __next__(self):
            while self.owner._alive:
                time.sleep(0.01)
            raise StopIteration

    @property
    def stdout(self):
        return self._Blocking(self)

    def poll(self):
        return None if self._alive else -9

    def wait(self, timeout = None):
        self._alive = False
        return -9

    def kill(self):
        self._alive = False


def test_generate_times_out_on_silent_hang(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    monkeypatch.setattr(eng.subprocess, "Popen", lambda cmd, **kw: _HangingPopen(cmd, **kw))
    t0 = time.time()
    with pytest.raises(RuntimeError, match = "timed out"):
        e.generate(
            SdCppModelFiles(diffusion_model = "/m/z.gguf"),
            SdCppGenParams(prompt = "x"),
            output_path = str(tmp_path / "x.png"),
            timeout = 0.3,
        )
    # The timeout is enforced promptly (not blocked until stdout EOF).
    assert time.time() - t0 < 5.0


def test_img2img_generate_passes_init_image(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n")
    _patch_popen(monkeypatch, lines = ["img2img"], returncode = 0, out_file = out)
    e.generate(
        SdCppModelFiles(diffusion_model = "/m/z.gguf"),
        SdCppGenParams(prompt = "x", init_img = str(src), strength = 0.5),
        output_path = str(out),
    )
    assert "--init-img" in _FakePopen.captured_cmd
    assert str(src) == _FakePopen.captured_cmd[_FakePopen.captured_cmd.index("--init-img") + 1]


def test_generate_native_speed_dedupes_against_offload(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    _patch_popen(monkeypatch, lines = ["ok"], returncode = 0, out_file = out)
    # offload already adds --diffusion-fa; native_speed="default" would add it again.
    e.generate(
        SdCppModelFiles(diffusion_model = "/m/z.gguf"),
        SdCppGenParams(prompt = "x"),
        output_path = str(out),
        offload = ["--offload-to-cpu", "--diffusion-fa"],
        native_speed = "default",
    )
    # --diffusion-fa appears exactly once (de-duped), not twice.
    assert _FakePopen.captured_cmd.count("--diffusion-fa") == 1


def test_generate_native_speed_adds_flag_when_not_offloaded(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    out = tmp_path / "img.png"
    _patch_popen(monkeypatch, lines = ["ok"], returncode = 0, out_file = out)
    e.generate(
        SdCppModelFiles(diffusion_model = "/m/z.gguf"),
        SdCppGenParams(prompt = "x"),
        output_path = str(out),
        offload = [],  # fast/resident tier: no offload, but speed flag still applies
        native_speed = "default",
    )
    assert _FakePopen.captured_cmd.count("--diffusion-fa") == 1


def test_upscale_runs_and_returns_path(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    out = tmp_path / "big.png"
    _patch_popen(monkeypatch, lines = ["upscaling", "done"], returncode = 0, out_file = out)
    result = e.upscale(
        SdCppUpscaleParams(input_image = "/in/small.png", upscale_model = "/m/esrgan.pth", repeats = 2),
        output_path = str(out),
    )
    assert result == out and out.is_file()
    assert _FakePopen.captured_cmd[_FakePopen.captured_cmd.index("--mode") + 1] == "upscale"
    assert "--upscale-model" in _FakePopen.captured_cmd


def test_upscale_raises_when_binary_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(eng, "find_sd_cpp_binary", lambda: None)
    e = SdCppEngine(binary = None)
    with pytest.raises(RuntimeError, match = "not found"):
        e.upscale(
            SdCppUpscaleParams(input_image = "/i.png", upscale_model = "/m/e.pth"),
            output_path = str(tmp_path / "x.png"),
        )


# ── engine routing ──────────────────────────────────────────────────────────


def test_routing_gpu_backends_use_diffusers():
    for backend in ("cuda", "rocm", "xpu"):
        assert select_diffusion_engine(backend, native_available = True) == ENGINE_DIFFUSERS


def test_routing_cpu_and_mps_use_native_when_available():
    assert select_diffusion_engine("cpu", native_available = True) == ENGINE_SD_CPP
    assert select_diffusion_engine("mps", native_available = True) == ENGINE_SD_CPP


def test_routing_cpu_falls_back_to_diffusers_without_binary():
    assert select_diffusion_engine("cpu", native_available = False) == ENGINE_DIFFUSERS


def test_routing_prefer_native_overrides_gpu():
    assert (
        select_diffusion_engine("cuda", native_available = True, prefer_native = True) == ENGINE_SD_CPP
    )
    # but only if a binary is actually available
    assert (
        select_diffusion_engine("cuda", native_available = False, prefer_native = True)
        == ENGINE_DIFFUSERS
    )
