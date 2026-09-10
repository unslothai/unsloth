# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio emits --lazy-mode for the two architectures whose per-layer embedding table
llama.cpp creates TENSOR_READ_LAZY, instead of leaving it to ``auto``.

``auto`` resolves inside the child from two facts Studio already holds: the table has to
clear 4 GiB (src/llama-model-loader.cpp:llama_model_loader::lazy_read::add), and every
selected device has to report mmap support or the mode silently becomes OFF
(src/llama-model.cpp:llama_model_base::load_tensors). A planner that priced the table as
paged and got a resident one over-commits by the whole table, so the mode is spelled out
and the planner is told which one was spelled.
"""

import inspect
from types import SimpleNamespace

import pytest

from core.inference.llama_cpp import (
    LlamaCppBackend,
    _lazy_mode_explicitly_set,
    _per_layer_embd_read_lazily,
)

GIB = 1024**3


class _Backend:
    """The real methods on a bare stub, the way the seam tests borrow _planned_tensor_spill."""

    _architecture = "gemma4"
    ple_bytes = 8 * GIB
    layout_readable = True
    vulkan_igpu = False
    integrated_cuda = False
    amd_apu = False

    def _tensor_spill_layout(
        self,
        model_path,
        *,
        all_shards = False,
    ):
        if not self.layout_readable:
            return None
        return SimpleNamespace(per_layer_embd_bytes = self.ple_bytes, arch = self._architecture)

    def _vulkan_targets_are_igpus(
        self,
        binary,
        gpu_indices = None,
    ):
        return self.vulkan_igpu

    def _integrated_cuda_unified_memory(self, gpu_indices = None):
        return self.integrated_cuda

    def _amd_apu_wants_unified_memory(self, gpu_indices = None):
        return self.amd_apu

    _selected_devices_can_mmap = LlamaCppBackend._selected_devices_can_mmap
    _studio_lazy_mode = LlamaCppBackend._studio_lazy_mode


def _mode(**attrs):
    stub = _Backend()
    call = {
        "model_path": "/models/m.gguf",
        "server_caps": {"supports_lazy_mode": True},
        "gpu_indices": [0],
        "binary": "/bin/llama-server",
        "env": {},
    }
    for key, value in attrs.items():
        if key in ("extra_args", "env", "server_caps", "is_vulkan_backend", "gpu_indices"):
            call[key] = value
        else:
            setattr(stub, key, value)
    return stub._studio_lazy_mode(**call)


@pytest.mark.parametrize("arch", ["gemma4", "qwen4exp"])
def test_emitted_on_for_the_two_lazy_archs(arch):
    """models/gemma4.cpp:llama_model_gemma4::load_arch_tensors and
    models/qwen4exp.cpp:llama_model_qwen4exp::load_arch_tensors are the only two that pass
    TENSOR_READ_LAZY for per_layer_token_embd."""
    assert _mode(_architecture = arch) == "on"


@pytest.mark.parametrize("arch", ["gemma3n", "llama", "qwen3moe", ""])
def test_not_emitted_for_any_other_arch(arch):
    """gemma3n has a per-layer table too and llama.cpp passes it flag 0, so it is resident
    whatever the mode says. Emitting there would move nothing and change the argv."""
    assert _mode(_architecture = arch) is None


def test_off_on_a_vulkan_igpu():
    """ggml-vulkan.cpp:ggml_backend_vk_device_get_props sets mmap_support =
    !ctx->is_integrated_gpu, and llama-model-loader's AUTO turns OFF on the first device
    without it, so ``on`` here would be a mode the child refuses and ``auto`` a price the
    planner gets wrong."""
    assert _mode(is_vulkan_backend = True, vulkan_igpu = True) == "off"
    assert _mode(is_vulkan_backend = True, vulkan_igpu = False) == "on"


def test_off_on_an_integrated_cuda_or_hip_device():
    """ggml-cuda.cu:ggml_backend_cuda_device_get_props reports mmap_support only for a
    device that is not GGML_BACKEND_DEVICE_TYPE_IGPU, and that file serves HIP too."""
    assert _mode(integrated_cuda = True) == "off"
    assert _mode(amd_apu = True) == "off"


@pytest.mark.parametrize(
    "extra_args, env",
    [
        (["-lzm", "off"], {}),
        (["--lazy-mode", "on"], {}),
        (["-lzm=auto"], {}),
        (["--lazy-mode=off"], {}),
        (None, {"LLAMA_ARG_LAZY_MODE": "off"}),
        (None, {"LLAMA_ARG_LAZY_MODE": "auto"}),
    ],
)
def test_a_user_who_chose_a_mode_keeps_it(extra_args, env):
    """Including ``auto``: someone who typed it asked for llama.cpp's own size test."""
    assert _mode(extra_args = extra_args, env = env) is None
    assert _lazy_mode_explicitly_set(extra_args, env) is True


def test_a_build_without_the_flag_gets_nothing():
    """--lazy-mode first appears at b10700; an unknown argument is an immediate exit."""
    assert _mode(server_caps = {"supports_lazy_mode": False}) is None
    assert _mode(server_caps = {}) is None


def test_a_model_with_no_per_layer_table_gets_nothing():
    assert _mode(ple_bytes = 0) is None
    assert _mode(layout_readable = False) is None


def test_the_planner_prices_the_emitted_mode_not_the_auto_default():
    """Both directions, and both are cases ``auto`` gets wrong: a 1540 MiB table under the
    4 GiB threshold that ``-lzm on`` really does page, and a 26 GiB table on an iGPU that
    ``auto`` would have declined to page anyway."""
    small = 1540 * 1024 * 1024
    assert _per_layer_embd_read_lazily("gemma4", small, supports_lazy_mode = True, env = {}) is False
    assert (
        _per_layer_embd_read_lazily(
            "gemma4", small, supports_lazy_mode = True, env = {}, emitted_mode = "on"
        )
        is True
    )
    big = 26 * GIB
    assert _per_layer_embd_read_lazily("gemma4", big, supports_lazy_mode = True, env = {}) is True
    assert (
        _per_layer_embd_read_lazily(
            "gemma4", big, supports_lazy_mode = True, env = {}, emitted_mode = "off"
        )
        is False
    )


def test_load_model_emits_the_mode_it_priced():
    """One emission site, and the same value reaches the planner's inputs. A second site,
    or an emission the inputs never see, is the disagreement this change removes."""
    source = inspect.getsource(LlamaCppBackend.load_model)
    assert source.count("_studio_lazy_mode(") == 1
    assert 'cmd.extend(["-lzm", _emitted_lazy_mode])' in source
    assert '_spill_inputs["emitted_lazy_mode"] = _emitted_lazy_mode' in source
    seam = inspect.getsource(LlamaCppBackend._planned_tensor_spill)
    assert 'emitted_mode = inputs.get("emitted_lazy_mode")' in seam
