# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prompt-cache-dropping offload tuning is skipped on iGPU/APU, which have one memory pool and no bus."""

from __future__ import annotations

import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend

_VISIBLE_DEVICE_MASKS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


@pytest.fixture(autouse = True)
def _no_inherited_gpu_mask(monkeypatch):
    """The ROCm arm asks about physical ids on a faked torch host. A mask inherited
    from the shell remaps them onto ids the fake host does not have."""
    for _mask in _VISIBLE_DEVICE_MASKS:
        monkeypatch.delenv(_mask, raising = False)


def _shares(**kwargs) -> bool:
    return LlamaCppBackend._offload_target_shares_system_memory(**kwargs)


def _vulkan(
    shared_gpu_ids,
    detected,
    gpu_indices = None,
) -> bool:
    return _shares(
        is_vulkan_backend = True,
        shared_gpu_ids = shared_gpu_ids,
        detected_gpus = detected,
        gpu_indices = gpu_indices,
    )


def _fake_torch(archs, *, hip = "6.4.0"):
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip = hip)
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: len(archs),
        get_device_properties = lambda i: types.SimpleNamespace(gcnArchName = archs[i]),
    )
    return torch


def test_an_igpu_only_target_shares_system_memory():
    assert _vulkan({0}, [(0, 170079)], gpu_indices = [0]) is True


def test_an_unpinned_load_reads_the_detected_list():
    """The auto-Vulkan arm pins nothing at this point, which is the shape the
    Strix Halo session ran in."""
    assert _vulkan({0}, [(0, 170079)]) is True


def test_a_discrete_card_beside_the_igpu_keeps_the_tuning():
    assert _vulkan({0}, [(0, 170079), (1, 24000)], gpu_indices = [0, 1]) is False
    assert _vulkan({0}, [(0, 170079), (1, 24000)]) is False


def test_pinning_only_the_igpu_of_a_mixed_host_still_shares():
    assert _vulkan({0}, [(0, 170079), (1, 24000)], gpu_indices = [0]) is True


def test_a_discrete_only_host_keeps_the_tuning():
    assert _vulkan(set(), [(0, 24000)], gpu_indices = [0]) is False


def test_a_load_with_no_device_at_all_fails_closed():
    assert _vulkan({0}, []) is False
    assert _vulkan(set(), []) is False


def test_an_amd_apu_shares_system_memory(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch(["gfx1151"]))
    assert (
        _shares(
            is_vulkan_backend = False,
            shared_gpu_ids = set(),
            detected_gpus = [(0, 60000)],
            gpu_indices = [0],
        )
        is True
    )


def test_a_discrete_amd_card_keeps_the_tuning(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch(["gfx1100"]))
    assert (
        _shares(
            is_vulkan_backend = False,
            shared_gpu_ids = set(),
            detected_gpus = [(0, 24000)],
            gpu_indices = [0],
        )
        is False
    )


def test_an_apu_next_to_a_discrete_card_keeps_the_tuning(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch(["gfx1151", "gfx1100"]))
    assert (
        _shares(
            is_vulkan_backend = False,
            shared_gpu_ids = set(),
            detected_gpus = [(0, 60000), (1, 24000)],
            gpu_indices = [0, 1],
        )
        is False
    )


def test_an_unreadable_inventory_fails_closed(monkeypatch):
    """No torch, so neither unified-memory set can be built. The tuning stays as it
    is rather than being dropped on a guess."""
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert (
        _shares(
            is_vulkan_backend = False,
            shared_gpu_ids = set(),
            detected_gpus = [(0, 60000)],
            gpu_indices = [0],
        )
        is False
    )


def test_the_vulkan_shared_set_is_not_read_as_physical_ids(monkeypatch):
    """Vulkan ordinals and HIP ids are different spaces. A ROCm launch must not
    inherit a shared set that was built from Vulkan's numbering."""
    monkeypatch.setitem(__import__("sys").modules, "torch", _fake_torch(["gfx1100"]))
    assert (
        _shares(
            is_vulkan_backend = False,
            shared_gpu_ids = {0},
            detected_gpus = [(0, 24000)],
            gpu_indices = [0],
        )
        is False
    )


def test_a_cuda_host_is_answered_without_touching_the_device(monkeypatch):
    """Answer from torch.version.hip alone: a device probe initialises CUDA and holds ~700 MiB of VRAM."""

    class _DeviceProbed(BaseException):
        """BaseException on purpose: every helper in this family swallows
        ``Exception`` per device, so an ordinary error would be caught and the
        test would pass against the very code it exists to catch."""

    def _explode(_ordinal):
        raise _DeviceProbed("probed the CUDA device to answer a Windows-only question")

    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip = None)
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: 1,
        get_device_properties = _explode,
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    assert (
        _shares(
            is_vulkan_backend = False,
            shared_gpu_ids = set(),
            detected_gpus = [(0, 24000)],
            gpu_indices = [0],
        )
        is False
    )


def test_only_the_pairs_this_policy_emitted_are_stripped():
    """Strip only exact flag-value pairs, since a valueless flag would swallow the next user argument."""
    cmd = [
        "llama-server",
        "--cache-ram",
        "0",
        "--ctx-checkpoints",
        "0",
        "--cache-ram",
        "4096",
        "--verbose",
    ]
    out = LlamaCppBackend._without_flag_pairs(cmd, ["--cache-ram", "0", "--ctx-checkpoints", "0"])
    assert out == ["llama-server", "--cache-ram", "4096", "--verbose"]


def test_stripping_a_pair_that_is_gone_is_a_no_op():
    cmd = ["llama-server", "--verbose"]
    assert LlamaCppBackend._without_flag_pairs(cmd, ["--cache-ram", "0"]) == cmd


def test_a_trailing_flag_without_its_value_is_left_alone():
    cmd = ["llama-server", "--cache-ram"]
    assert LlamaCppBackend._without_flag_pairs(cmd, ["--cache-ram"]) == cmd


def test_a_user_device_flag_makes_the_target_unknown():
    assert LlamaCppBackend._cache_tuning_target_unknown(["--device", "ROCm1"], False, {})


def test_an_inherited_device_env_makes_the_target_unknown():
    # llama.cpp reads LLAMA_ARG_DEVICE before argv and only gpu_ids clears it.
    assert LlamaCppBackend._cache_tuning_target_unknown(None, False, {"LLAMA_ARG_DEVICE": "ROCm0"})
    assert not LlamaCppBackend._cache_tuning_target_unknown(None, False, {"LLAMA_ARG_DEVICE": "  "})


def test_an_explicit_pin_owns_the_placement_so_the_target_is_known():
    assert not LlamaCppBackend._cache_tuning_target_unknown(
        ["--device", "ROCm1"], True, {"LLAMA_ARG_DEVICE": "ROCm0"}
    )
    assert not LlamaCppBackend._cache_tuning_target_unknown(None, False, {})


_CAPS = {"supports_cache_ram": True, "ctx_checkpoints_flag": "--ctx-checkpoints"}


def test_the_retry_applies_the_tuning_when_nothing_states_it():
    assert LlamaCppBackend._retry_cache_tuning_flags(
        ["llama-server", "-m", "x.gguf"],
        cache_ram = None,
        ctx_checkpoints = None,
        server_caps = _CAPS,
    ) == ["--cache-ram", "0", "--ctx-checkpoints", "0"]


def test_the_retry_does_not_overrule_a_cache_flag_the_command_already_states():
    # The extras are already in cmd, so appending here would win last-wins, the reverse of launch.
    flags = LlamaCppBackend._retry_cache_tuning_flags(
        ["llama-server", "-m", "x.gguf", "--cache-ram", "8192"],
        cache_ram = None,
        ctx_checkpoints = None,
        server_caps = _CAPS,
    )
    assert flags == ["--ctx-checkpoints", "0"]

    assert (
        LlamaCppBackend._retry_cache_tuning_flags(
            ["llama-server", "--cache-ram=8192", "--ctx-checkpoints=4"],
            cache_ram = None,
            ctx_checkpoints = None,
            server_caps = _CAPS,
        )
        == []
    )


def test_the_retry_reads_the_short_spellings_as_the_same_settings():
    # -cram and -ctxcp are accepted aliases; appending the long form would zero the user's value.
    assert (
        LlamaCppBackend._retry_cache_tuning_flags(
            ["llama-server", "-cram", "8192", "-ctxcp", "4"],
            cache_ram = None,
            ctx_checkpoints = None,
            server_caps = _CAPS,
        )
        == []
    )

    assert LlamaCppBackend._retry_cache_tuning_flags(
        ["llama-server", "-cram", "8192"],
        cache_ram = None,
        ctx_checkpoints = None,
        server_caps = _CAPS,
    ) == ["--ctx-checkpoints", "0"]
    assert LlamaCppBackend._retry_cache_tuning_flags(
        ["llama-server", "-ctxcp=4"],
        cache_ram = None,
        ctx_checkpoints = None,
        server_caps = _CAPS,
    ) == ["--cache-ram", "0"]


def test_the_retry_reads_the_other_checkpoint_alias_as_the_same_setting():
    assert LlamaCppBackend._retry_cache_tuning_flags(
        ["llama-server", "--swa-checkpoints", "4"],
        cache_ram = None,
        ctx_checkpoints = None,
        server_caps = _CAPS,
    ) == ["--cache-ram", "0"]


def test_the_retry_skips_what_the_build_and_the_fields_already_own():
    assert (
        LlamaCppBackend._retry_cache_tuning_flags(
            ["llama-server"], cache_ram = 4096, ctx_checkpoints = 8, server_caps = _CAPS
        )
        == []
    )
    assert (
        LlamaCppBackend._retry_cache_tuning_flags(
            ["llama-server"],
            cache_ram = None,
            ctx_checkpoints = None,
            server_caps = {"supports_cache_ram": False, "ctx_checkpoints_flag": None},
        )
        == []
    )
