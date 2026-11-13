# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest import mock

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_llama_cpp_placement as placement  # noqa: E402
import utils.hardware as hardware  # noqa: E402
from _sidecar_fit_helpers import last  # noqa: E402
from core.inference import llama_cpp as lc  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from test_llama_cpp_placement import _discrete_linux_host, _moe_cache_host  # noqa: E402,F401

MIB = 1024 * 1024
KV_BYTES_PER_TOKEN = MIB // 4
CTX = 65536
FULL_CONTEXT_MIB = CTX * KV_BYTES_PER_TOKEN // MIB


@pytest.fixture
def launch(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    monkeypatch.delenv("LLAMA_ARG_CACHE_RAM", raising = False)

    def run(
        avail_mib = 64 * 1024,
        caps = None,
        platform = "linux",
        memory = ((0, 48 * 1024, 48 * 1024),),
        keep_estimator = False,
        **load_kwargs,
    ):
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: avail_mib)
        )
        backend, gguf = placement._backend(tmp_path, vulkan = False, memory = list(memory))
        backend.probe_server_capabilities = lambda _binary = None: {
            "supports_cache_ram": True,
            "supports_kv_unified": True,
            **(caps or {}),
        }
        backend._apple_metal_memory_budget_bytes = lambda: 0
        backend._get_gguf_size_bytes = lambda _path: 4 * 1024 * MIB
        backend._can_estimate_kv = lambda: True
        if not keep_estimator:
            backend._estimate_kv_cache_bytes = lambda n_ctx, *args, **kwargs: (
                max(0, int(n_ctx)) * KV_BYTES_PER_TOKEN
            )
        load_kwargs.setdefault("n_ctx", CTX)
        return placement._launch(backend, gguf, **load_kwargs)["cmd"]

    return run


def cache_ram_flags(cmd):
    return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--cache-ram"]


def test_prompt_cache_holds_a_full_context_conversation(launch):
    assert FULL_CONTEXT_MIB > 8192
    assert last(launch(), "--cache-ram") == str(FULL_CONTEXT_MIB)


def test_the_cache_holds_the_checkpoints_saved_with_the_conversation(launch, monkeypatch):
    ckpt_mib = 512

    def estimate(
        self,
        n_ctx,
        *args,
        ctx_checkpoints = 0,
        **kwargs,
    ):
        return max(0, int(n_ctx)) * KV_BYTES_PER_TOKEN + (ckpt_mib * MIB if ctx_checkpoints else 0)

    monkeypatch.setattr(LlamaCppBackend, "_estimate_kv_cache_bytes", estimate)
    sized = launch(keep_estimator = True, ctx_checkpoints = 8)
    assert last(sized, "--cache-ram") == str(FULL_CONTEXT_MIB + ckpt_mib)


def test_a_context_the_default_already_holds_adds_no_flag(launch):
    assert "--cache-ram" not in launch(n_ctx = 16384)


def test_prompt_cache_never_outgrows_free_host_ram(launch):
    avail_mib = 14 * 1024
    sized = int(last(launch(avail_mib = avail_mib), "--cache-ram"))
    assert 8192 < sized < FULL_CONTEXT_MIB
    assert sized == avail_mib - 2048 - 1024


def test_a_host_without_room_keeps_the_default(launch):
    assert "--cache-ram" not in launch(avail_mib = 9 * 1024)


def test_a_full_cache_leaves_the_server_its_own_memory(launch):
    assert "--cache-ram" not in launch(avail_mib = 8192 + 2048 + 1024)
    assert last(launch(avail_mib = 8192 + 2048 + 1024 + 1), "--cache-ram") == "8193"


def test_a_chosen_cache_size_is_launched_as_given(launch):
    assert last(launch(cache_ram = 4096), "--cache-ram") == "4096"
    assert last(launch(cache_ram = 0), "--cache-ram") == "0"


def test_a_typed_cache_flag_is_the_only_one_launched(launch):
    assert cache_ram_flags(launch(extra_args = ["--cache-ram", "2048"])) == ["2048"]
    cmd = launch(extra_args = ["-cram", "2048"])
    assert "--cache-ram" not in cmd


def test_an_inherited_cache_size_is_left_alone(launch, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_CACHE_RAM", "2048")
    assert "--cache-ram" not in launch()


def test_a_build_without_the_flag_gets_none(launch):
    assert "--cache-ram" not in launch(caps = {"supports_cache_ram": False})


def test_windows_off_the_gpu_is_sized(launch):
    assert cache_ram_flags(launch(platform = "win32", memory = ())) == [str(FULL_CONTEXT_MIB)]


def test_windows_partial_offload_is_sized(launch):
    cmd = launch(platform = "win32", memory = ((0, 6 * 1024, 6 * 1024),))
    assert cache_ram_flags(cmd) == [str(FULL_CONTEXT_MIB)]


def test_windows_full_offload_keeps_only_its_own_zero(launch):
    assert cache_ram_flags(launch(platform = "win32")) == ["0"]


def test_windows_full_offload_launches_a_chosen_size_alone(launch):
    assert cache_ram_flags(launch(platform = "win32", cache_ram = 4096)) == ["4096"]


def test_apple_silicon_keeps_the_default(launch, monkeypatch):
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: True)
    assert "--cache-ram" not in launch(platform = "darwin")


@pytest.mark.parametrize("ram_gib, sized", [(26, None), (28, "10240")])
def test_the_moe_expert_cache_is_priced_before_the_prompt_cache(
    tmp_path, _moe_cache_host, ram_gib, sized
):
    backend, gguf = placement._moe_backend(
        tmp_path, size_gib = 20, memory = [(0, 32000, 48000)], moe = True
    )
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda *a, **k: 12 * 1024 * MIB
    backend._available_system_memory_mib = lambda: ram_gib * 1024
    backend._gguf_tensor_scan = lambda _path: ("qwen3moe", {}, 15 * 1024 * MIB)
    caps = dict(LlamaCppBackend.probe_server_capabilities.__func__(LlamaCppBackend, None))
    caps.update(placement._CACHE_CAPS_ON)
    backend.probe_server_capabilities = lambda _binary = None: caps
    cmd = placement._launch(backend, gguf)["cmd"]
    assert placement._has_moe_cache(cmd)
    assert (last(cmd, "--cache-ram") if "--cache-ram" in cmd else None) == sized


def test_the_cpu_fallback_drops_the_sized_prompt_cache():
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._memory_dio_flags = []
    backend._fit_load_mode_flags = []
    backend._sized_cache_ram_flags = ["--cache-ram", "16384"]
    replay = ["llama-server", "-m", "model.gguf", "--cache-ram", "16384", "--device", "none"]
    with (
        mock.patch.object(lc.LlamaCppBackend, "_is_vulkan_backend", return_value = True),
        mock.patch.object(lc.LlamaCppBackend, "_cpu_isolated_replay", return_value = list(replay)),
        mock.patch.object(lc.LlamaCppBackend, "_cpu_isolated_binary", return_value = "cpu-server"),
        mock.patch.object(
            lc.LlamaCppBackend,
            "_llama_server_env_for_binary",
            return_value = {lc._loader_path_var(): "/staged"},
        ),
    ):
        out, _reason, _note = backend._prepare_cpu_fallback_launch("llama-server", replay, {}, {})
    assert "--cache-ram" not in out


def test_every_retry_that_drops_the_fits_load_mode_drops_the_sized_cache():
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert src.count("_without_subsequence(fa_cmd, self._sized_cache_ram_flags)") == 1
    assert src.count("_without_subsequence(_run, self._sized_cache_ram_flags)") == 1
    assert src.count("_without_subsequence(cmd, self._sized_cache_ram_flags)") == 1
    assert src.count("_cpu_projector_cmd, self._sized_cache_ram_flags") == 1
