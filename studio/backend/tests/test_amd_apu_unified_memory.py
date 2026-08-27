# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGML_CUDA_ENABLE_UNIFIED_MEMORY must be set only for AMD unified-memory APUs
(gfx1150/gfx1151/gfx1152), never for discrete AMD, NVIDIA, CPU or macOS."""

from __future__ import annotations

import sys
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend

_VISIBLE_DEVICE_MASKS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


@pytest.fixture(autouse = True)
def _no_inherited_gpu_mask(monkeypatch):
    """These tests fake a torch host and then ask about GPU ordinal 0. A mask
    inherited from the shell remaps that ordinal onto a physical id the fake
    host does not have, so the answer flips to False and nine tests fail. CI
    runners carry no mask, so it only ever bites locally. The tests that are
    about a mask still set their own, after this."""
    for _m in _VISIBLE_DEVICE_MASKS:
        monkeypatch.delenv(_m, raising = False)


def _fake_torch(
    hip,
    archs,
    *,
    cuda_ok = True,
):
    t = types.ModuleType("torch")
    t.version = types.SimpleNamespace(hip = hip)
    t.cuda = types.SimpleNamespace(
        is_available = lambda: cuda_ok,
        device_count = lambda: len(archs),
        get_device_properties = lambda i: types.SimpleNamespace(gcnArchName = archs[i]),
    )
    return t


@pytest.mark.parametrize(
    "hip,archs,expected",
    [
        ("6.2.0", ["gfx1151:xnack-"], True),  # Strix Halo APU (suffix stripped)
        ("6.2.0", ["gfx1150"], True),  # Strix Point APU
        ("6.2.0", ["gfx1152"], True),  # Krackan Point APU (Radeon 860M/840M)
        ("6.2.0", ["gfx1152:sramecc-:xnack-"], True),  # same, feature flags stripped
        ("6.2.0", ["gfx1100"], False),  # discrete RDNA3
        ("6.2.0", ["gfx1201"], False),  # discrete RDNA4
        ("6.2.0", ["gfx942"], False),  # MI300X (data center)
        (None, ["sm_90"], False),  # NVIDIA (no torch.version.hip)
        ("6.2.0", ["gfx1100", "gfx1151"], True),  # mixed dGPU + APU
    ],
)
def test_apu_unified_memory_gating(monkeypatch, hip, archs, expected):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(hip, archs))
    assert LlamaCppBackend._amd_apu_wants_unified_memory() is expected


def test_apu_guard_scopes_to_selected_gpu(monkeypatch):
    # Mixed host: physical id 0 = discrete gfx1100, 1 = gfx1151 APU.
    for _m in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(_m, raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("6.2.0", ["gfx1100", "gfx1151"]))
    # Selecting only the dGPU, or an empty selection, must not be unified-memory.
    assert LlamaCppBackend._amd_apu_wants_unified_memory([0]) is False
    assert LlamaCppBackend._amd_apu_wants_unified_memory([]) is False
    # Selecting the APU, or no selection, does.
    assert LlamaCppBackend._amd_apu_wants_unified_memory([1]) is True
    assert LlamaCppBackend._amd_apu_wants_unified_memory() is True


def test_apu_guard_honors_hip_visible_devices_mask(monkeypatch):
    # ROCm resolves ids via HIP first: the mask exposes only the APU as ordinal 0
    # but physical id 1, so the selection [1] must still match.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("6.2.0", ["gfx1151"]))
    assert LlamaCppBackend._amd_apu_wants_unified_memory([1]) is True
    assert LlamaCppBackend._amd_apu_wants_unified_memory([0]) is False


def test_cpu_no_cuda_returns_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("6.2.0", [], cuda_ok = False))
    assert LlamaCppBackend._amd_apu_wants_unified_memory() is False


def test_missing_torch_returns_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert LlamaCppBackend._amd_apu_wants_unified_memory() is False


_GB = 1024**3
_MIB_PER_GB = 1024
# Module-level (not a class attr) so it stays a plain function, not a bound method.
_shortfall = LlamaCppBackend._apu_ram_shortfall_message


class TestApuRamShortfall:
    """On a unified-memory APU the weights load into system RAM, so a model
    larger than available RAM (the field case: a 64.6 GB GGUF on a WSL VM capped
    well below the ROCm-reported APU budget) must be refused before spawning,
    not left to OOM-kill the Unsloth process."""

    def test_field_case_wsl_cap_refuses(self):
        # 64.6 GB weights, ~46 GB available (WSL VM): refuse with guidance.
        msg = _shortfall(int(64.6 * _GB), 46 * _MIB_PER_GB)
        assert msg is not None
        assert "65 GB" in msg and "46 GB" in msg
        assert ".wslconfig" in msg

    def test_bare_metal_fits_allows(self):
        # Same model, ~92 GB available (no WSL cap): allow.
        assert _shortfall(int(64.6 * _GB), 92 * _MIB_PER_GB) is None

    def test_unknown_available_never_refuses(self):
        assert _shortfall(int(64.6 * _GB), None) is None

    def test_boundary_at_headroom(self):
        # 20 GB weights, headroom 2 GB. avail 23 GB -> fits; 21 GB -> refuse.
        assert _shortfall(20 * _GB, 23 * _MIB_PER_GB) is None
        assert _shortfall(20 * _GB, 21 * _MIB_PER_GB) is not None

    def test_available_system_memory_is_int_or_none(self):
        v = LlamaCppBackend._available_system_memory_mib()
        assert v is None or (isinstance(v, int) and v > 0)


# The local B200's real profile, so the spoofed APU is sized like a machine we
# actually have rather than an invented one. A large shared pool is exactly where
# the missing host reserve mattered: 3% of 179 GiB is 5.4 GiB, but the reserve is
# an absolute 1 GiB off free, and the "total" must not become a VRAM budget.
_B200_TOTAL_MIB = 183359
_B200_FREE_MIB = 181928
_MIB = 1024 * 1024


def _fake_torch_with_memory(
    hip,
    archs,
    free_mib,
    total_mib,
    *,
    cuda_ok = True,
):
    """_fake_torch plus mem_get_info, which the memory probe needs."""
    t = _fake_torch(hip, archs, cuda_ok = cuda_ok)
    t.cuda.mem_get_info = lambda i: (free_mib * _MIB, total_mib * _MIB)
    return t


def _probe(
    monkeypatch,
    hip,
    archs,
    free_mib = _B200_FREE_MIB,
    total_mib = _B200_TOTAL_MIB,
):
    for _m in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(_m, raising = False)
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch_with_memory(hip, archs, free_mib, total_mib)
    )
    # Force the torch branch: no Vulkan build, and nvidia-smi must not answer.
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b: False))
    monkeypatch.setattr(LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "x"))
    # Pin host availability: the shared path caps by it, so a runner with less RAM
    # than the spoofed profile would otherwise change every expectation below.
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 1 << 30)
    )
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no nvidia-smi")),
    )
    return LlamaCppBackend._get_gpu_memory()


class TestTheRocmProbeReservesHostRamOnAnApu:
    """ROCm reports no iGPU flag, so before this an APU was sized as a discrete
    card: no host margin, and an absolute reserve taken off what is really
    system RAM. The Vulkan probe already did both; this matches it."""

    def test_an_apu_loses_the_host_reserve_and_reports_no_total(self, monkeypatch):
        assert _probe(monkeypatch, "6.2.0", ["gfx1151:xnack-"]) == [(0, _B200_FREE_MIB - 1024, 0)]

    def test_a_discrete_amd_card_is_untouched(self, monkeypatch):
        assert _probe(monkeypatch, "6.2.0", ["gfx1100"]) == [(0, _B200_FREE_MIB, _B200_TOTAL_MIB)]

    def test_nvidia_through_the_torch_fallback_is_untouched(self, monkeypatch):
        """The real local GPU: no HIP, so nothing here may apply."""
        assert _probe(monkeypatch, None, ["sm_100"]) == [(0, _B200_FREE_MIB, _B200_TOTAL_MIB)]

    def test_a_mixed_host_only_reserves_on_the_apu(self, monkeypatch):
        assert _probe(monkeypatch, "6.2.0", ["gfx1100", "gfx1151"]) == [
            (0, _B200_FREE_MIB, _B200_TOTAL_MIB),
            (1, _B200_FREE_MIB - 1024, 0),
        ]

    def test_the_reserve_cannot_go_negative(self, monkeypatch):
        assert _probe(monkeypatch, "6.2.0", ["gfx1151"], free_mib = 512, total_mib = 512) == [(0, 0, 0)]

    @pytest.mark.parametrize("arch", ["gfx1150", "gfx1151", "gfx1152"])
    def test_every_unified_arch_is_covered(self, monkeypatch, arch):
        assert _probe(monkeypatch, "6.2.0", [arch])[0][1] == _B200_FREE_MIB - 1024

    def test_the_probe_and_the_mlock_gate_agree(self, monkeypatch):
        """Both read one arch map, so they cannot disagree about a device."""
        for arch, shared in (("gfx1151", True), ("gfx1100", False)):
            rows = _probe(monkeypatch, "6.2.0", [arch])
            assert (rows[0][2] == 0) is shared
            assert LlamaCppBackend._amd_apu_wants_unified_memory([0]) is shared


class TestTheGateStillFailsOpen:
    """Every helper in this family answers False rather than raising: they are
    consulted on the load path, so a bad argument must skip the optimisation,
    not fail the load."""

    @pytest.mark.parametrize("gpu_indices", [5, [[0]], "0", [None], object()])
    def test_a_bad_gpu_indices_answers_false(self, monkeypatch, gpu_indices):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch("6.2.0", ["gfx1151"]))
        assert LlamaCppBackend._amd_apu_wants_unified_memory(gpu_indices) is False

    def test_a_good_one_still_works(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch("6.2.0", ["gfx1151"]))
        assert LlamaCppBackend._amd_apu_wants_unified_memory([0]) is True


class TestAmdSdkWheelsCountAsRocm:
    """AMD SDK / Radeon wheels leave torch.version.hip unset and only encode
    "rocm" in __version__, which _resolve_visible_physical_ids already handles.
    The arch map must use the same predicate or an APU goes unrecognised there."""

    @staticmethod
    def _torch(
        hip,
        version,
        archs = ("gfx1151",),
    ):
        t = _fake_torch(hip, list(archs))
        t.__version__ = version
        return t

    @pytest.mark.parametrize(
        ("hip", "version", "expected"),
        [
            ("6.2.0", "2.5.0+rocm6.2", True),
            (None, "2.11.0+rocm7.13", True),  # AMD SDK wheel
            (None, "2.5.0+ROCm7.0", True),  # case-insensitive
            (None, "2.5.0+cu124", False),
            (None, "2.5.0", False),
        ],
    )
    def test_the_arch_map_matches_the_id_resolver(self, monkeypatch, hip, version, expected):
        monkeypatch.setitem(sys.modules, "torch", self._torch(hip, version))
        assert bool(LlamaCppBackend._rocm_unified_memory_gpu_ids()) is expected
        assert LlamaCppBackend._amd_apu_wants_unified_memory([0]) is expected


class TestTheApuBudgetIsCappedByHostRam:
    """Windows HIP without the SDK reports free==total (#7072), so the ROCm free
    figure cannot be trusted on a shared pool. System RAM is the real ceiling."""

    @staticmethod
    def _probe(monkeypatch, arch, free_mib, avail_mib):
        t = _fake_torch("6.2.0", [arch])
        t.cuda.mem_get_info = lambda i: (free_mib * 1024 * 1024, free_mib * 1024 * 1024)
        monkeypatch.setitem(sys.modules, "torch", t)
        for _m in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(_m, raising = False)
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: avail_mib)
        )
        monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b: False))
        monkeypatch.setattr(
            LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "llama-server")
        )
        monkeypatch.setattr(
            "core.inference.llama_cpp.subprocess.run",
            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no nvidia-smi")),
        )
        return LlamaCppBackend._get_gpu_memory()

    def test_the_sentinel_is_capped(self, monkeypatch):
        assert self._probe(monkeypatch, "gfx1151", 100_000, 12_000) == [(0, 12_000 - 1024, 0)]

    def test_an_honest_smaller_free_wins(self, monkeypatch):
        """The cap is a ceiling, never a floor."""
        assert self._probe(monkeypatch, "gfx1151", 8_000, 64_000) == [(0, 8_000 - 1024, 0)]

    def test_unreadable_system_ram_keeps_the_old_answer(self, monkeypatch):
        assert self._probe(monkeypatch, "gfx1151", 100_000, None) == [(0, 100_000 - 1024, 0)]

    def test_a_discrete_card_is_never_capped(self, monkeypatch):
        assert self._probe(monkeypatch, "gfx1100", 100_000, 12_000) == [(0, 100_000, 100_000)]


class TestRadeonWheelsWithoutAnArchName:
    """AMD SDK / Radeon wheels may populate none of the arch attributes. The
    training worker's classifier already handles that (is_integrated, then the
    arch spellings, then the Radeon name table); this path shares it so the two
    cannot disagree about a device."""

    @staticmethod
    def _torch(**props):
        t = _fake_torch("6.2.0", ["unused"])
        t.cuda.get_device_properties = lambda i: types.SimpleNamespace(**props)
        return t

    @pytest.mark.parametrize(
        ("props", "expected"),
        [
            ({"gcnArchName": "gfx1151"}, True),
            ({"gcn_arch_name": "gfx1150"}, True),  # variant spelling
            ({"name": "AMD Radeon 8060S Graphics"}, True),  # Strix Halo by name
            ({"name": "AMD Radeon 860M"}, True),  # Krackan by name
            ({"is_integrated": True, "name": "AMD Radeon Graphics"}, True),
            ({"gcnArchName": "gfx1100"}, False),
            ({"name": "AMD Radeon RX 7900 XTX"}, False),
        ],
    )
    def test_the_probe_uses_every_fallback(self, monkeypatch, props, expected):
        monkeypatch.setitem(sys.modules, "torch", self._torch(**props))
        assert LlamaCppBackend._amd_apu_wants_unified_memory([0]) is expected
        assert bool(LlamaCppBackend._rocm_unified_memory_gpu_ids()) is expected


class TestTheProbeTestsDoNotDependOnHostRam:
    """The shared path caps by available system RAM, so the spoofed profile must
    pin it or every expectation moves with the runner's memory."""

    def test_the_helper_pins_availability(self, monkeypatch):
        """_probe must survive a runner smaller than the spoofed free figure."""
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 14_000)
        )
        assert _probe(monkeypatch, "6.2.0", ["gfx1151"]) == [(0, _B200_FREE_MIB - 1024, 0)]

    def test_without_the_pin_the_cap_really_would_bite(self, monkeypatch):
        """Proves the pin is load-bearing rather than decorative."""
        rows = _probe(monkeypatch, "6.2.0", ["gfx1151"])
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 14_000)
        )
        capped = LlamaCppBackend._get_gpu_memory()
        assert rows == [(0, _B200_FREE_MIB - 1024, 0)]
        assert capped == [(0, 14_000 - 1024, 0)]


class TestTheOptOutHelper:
    """_unified_memory_opted_out, the #8651 escape hatch. ggml tests presence, not
    value, so only ABSENCE is off and this decides when to make it absent."""

    @pytest.mark.parametrize(
        "env,expected",
        [
            ({}, False),  # nothing set: the default decides
            ({"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}, False),
            ({"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "2"}, False),  # any value is ON to ggml
            ({"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "true"}, False),
            ({"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "0"}, True),  # the reported trap
            ({"GGML_CUDA_ENABLE_UNIFIED_MEMORY": ""}, True),
            ({"GGML_CUDA_ENABLE_UNIFIED_MEMORY": " Off "}, True),  # trimmed, folded
            ({"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "no"}, True),
            ({"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "false"}, True),
            ({"UNSLOTH_DISABLE_UNIFIED_MEMORY": "1"}, True),
            ({"UNSLOTH_DISABLE_UNIFIED_MEMORY": "0"}, False),  # exact "1", like the DC switch
            ({"UNSLOTH_DISABLE_UNIFIED_MEMORY": "yes"}, False),
            # The switch has to beat a truthy value.
            (
                {
                    "UNSLOTH_DISABLE_UNIFIED_MEMORY": "1",
                    "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1",
                },
                True,
            ),
        ],
    )
    def test_opt_out_decisions(self, env, expected):
        assert LlamaCppBackend._unified_memory_opted_out(env) is expected

    def test_none_reads_the_process_env(self, monkeypatch):
        """The default arg is the process env, so a shell export is honoured."""
        monkeypatch.delenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY", raising = False)
        monkeypatch.setenv("UNSLOTH_DISABLE_UNIFIED_MEMORY", "1")
        assert LlamaCppBackend._unified_memory_opted_out() is True

    def test_a_hostile_env_fails_open(self):
        """Fails open: a bad env must not block a load. False is pre-#8651."""

        class _Exploding(dict):
            def get(self, *_args, **_kwargs):
                raise RuntimeError("no")

        assert LlamaCppBackend._unified_memory_opted_out(_Exploding()) is False
