# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows full-offload tuning (#5692) drops the llama-server prompt cache to avoid
WDDM/PCI-E traffic. An iGPU or APU has one memory pool and no bus, so the same flags
save nothing and remove prefix reuse instead.

Measured on a Strix Halo running the Vulkan build with -ngl -1: one 48.8 h session
spent 44.3 h re-ingesting prompts, 3.38 M prompt tokens against 72 k generated, with
`--cache-ram 0 --ctx-checkpoints 0` in its own argv and
`--cache-idle-slots requires --cache-ram, disabling` in llama-server's first lines.

The predicate fails closed everywhere it cannot prove the whole target is shared, so
a discrete card keeps the tuning it was written for.
"""

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


# ── Vulkan, where the shared set is ggml's compact ordinals ──


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


# ── ROCm and CUDA, where the ids are physical ──


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
    """The predicate must not create a CUDA primary context to answer.

    ``torch.cuda.get_device_properties`` initialises CUDA in the backend process
    and never gives the memory back (~700 MiB), which is VRAM the child
    llama-server then cannot use. On a CUDA host the answer is False either way,
    so it has to come from ``torch.version.hip`` alone. The fake raises on any
    device probe, so a reintroduced ``_integrated_cuda_gpu_ids()`` call fails
    this test rather than merely costing memory in production."""

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


# ── _without_flag_pairs: the strip the arch-crash respawn uses ──


def test_only_the_pairs_this_policy_emitted_are_stripped():
    """The respawn reuses the argv it already built, so taking the tuning back off has
    to be an exact-token removal. Every pair is a flag with its value, which is what
    makes the two-token step safe: a valueless flag would swallow the user extra that
    follows it, which is why the mlock path refuses to strip at all."""
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
    # The user's own --cache-ram 4096 survives; only the emitted zeros go.
    assert out == ["llama-server", "--cache-ram", "4096", "--verbose"]


def test_stripping_a_pair_that_is_gone_is_a_no_op():
    cmd = ["llama-server", "--verbose"]
    assert LlamaCppBackend._without_flag_pairs(cmd, ["--cache-ram", "0"]) == cmd


def test_a_trailing_flag_without_its_value_is_left_alone():
    # Defensive: an odd-length pair list would otherwise index past the end.
    cmd = ["llama-server", "--cache-ram"]
    assert LlamaCppBackend._without_flag_pairs(cmd, ["--cache-ram"]) == cmd


# ── the target the tuning is chosen against ──


def test_a_user_device_flag_makes_the_target_unknown():
    assert LlamaCppBackend._cache_tuning_target_unknown(
        ["--device", "ROCm1"], None, {}
    )


def test_an_inherited_device_env_makes_the_target_unknown():
    # The env twin survives an automatic load verbatim -- only an explicit gpu_ids
    # clears it -- and llama.cpp reads it before argv, so the generated pin is not
    # what the child places against. Reading argv alone emitted --cache-ram 0 at an
    # APU the picker had paired with a discrete card.
    assert LlamaCppBackend._cache_tuning_target_unknown(
        None, None, {"LLAMA_ARG_DEVICE": "ROCm0"}
    )
    # Set but empty is not a selection.
    assert not LlamaCppBackend._cache_tuning_target_unknown(
        None, None, {"LLAMA_ARG_DEVICE": "  "}
    )


def test_an_explicit_pin_owns_the_placement_so_the_target_is_known():
    # The control: gpu_ids clears both spellings, so neither can name another device.
    assert not LlamaCppBackend._cache_tuning_target_unknown(
        ["--device", "ROCm1"], [0], {"LLAMA_ARG_DEVICE": "ROCm0"}
    )
    assert not LlamaCppBackend._cache_tuning_target_unknown(None, None, {})


# ── the arch-crash retry keeps the launch's precedence ──

_CAPS = {"supports_cache_ram": True, "ctx_checkpoints_flag": "--ctx-checkpoints"}


def test_the_retry_applies_the_tuning_when_nothing_states_it():
    assert LlamaCppBackend._retry_cache_tuning_flags(
        ["llama-server", "-m", "x.gguf"],
        cache_ram = None,
        ctx_checkpoints = None,
        server_caps = _CAPS,
    ) == ["--cache-ram", "0", "--ctx-checkpoints", "0"]


def test_the_retry_does_not_overrule_a_cache_flag_the_command_already_states():
    # The extras sit in cmd already, so appending here wins last-wins and would zero a
    # value the panel still shows -- the reverse of the launch, where extras win.
    flags = LlamaCppBackend._retry_cache_tuning_flags(
        ["llama-server", "-m", "x.gguf", "--cache-ram", "8192"],
        cache_ram = None,
        ctx_checkpoints = None,
        server_caps = _CAPS,
    )
    assert flags == ["--ctx-checkpoints", "0"]

    # The attached spelling is the same setting.
    assert LlamaCppBackend._retry_cache_tuning_flags(
        ["llama-server", "--cache-ram=8192", "--ctx-checkpoints=4"],
        cache_ram = None,
        ctx_checkpoints = None,
        server_caps = _CAPS,
    ) == []


def test_the_retry_reads_the_other_checkpoint_alias_as_the_same_setting():
    # A build advertising --ctx-checkpoints can still be handed --swa-checkpoints.
    assert LlamaCppBackend._retry_cache_tuning_flags(
        ["llama-server", "--swa-checkpoints", "4"],
        cache_ram = None,
        ctx_checkpoints = None,
        server_caps = _CAPS,
    ) == ["--cache-ram", "0"]


def test_the_retry_skips_what_the_build_and_the_fields_already_own():
    # An explicit field, as at launch, and a build without the capability.
    assert LlamaCppBackend._retry_cache_tuning_flags(
        ["llama-server"], cache_ram = 4096, ctx_checkpoints = 8, server_caps = _CAPS
    ) == []
    assert LlamaCppBackend._retry_cache_tuning_flags(
        ["llama-server"],
        cache_ram = None,
        ctx_checkpoints = None,
        server_caps = {"supports_cache_ram": False, "ctx_checkpoints_flag": None},
    ) == []
