# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OS x GPU-vendor matrix for the CUDA SM gate, the sibling of
``test_gpu_arch_gate_os_matrix_7624.py`` for the ROCm arch gate.

The gate refuses a launch only when every visible GPU is OLDER than the oldest
arch the bundle was compiled for, and must be inert everywhere else. Output
alone cannot show that: a gate that ran and passed returns None exactly like one
that never ran, so the marker reader and the nvidia-smi probe are spied and
asserted un-called on the cells that must not reach them.

Matrix: [Windows, Linux, WSL, macOS] x [NVIDIA, AMD, CPU-only]. No NVIDIA GPU is
needed; nvidia-smi, the marker and the masks are faked in the shapes the
installer writes (digit-string ``supported_sms``, ``compute_cap`` as "8.6").
"""

from __future__ import annotations

import subprocess

import pytest

from core.inference.llama_cpp import LlamaCppBackend

# sys.platform / platform.system() pair per simulated host.
_OS_CELLS = {
    "windows": ("win32", "Windows"),
    "linux": ("linux", "Linux"),
    "wsl": ("linux", "Linux"),
    "macos": ("darwin", "Darwin"),
}
OS_KEYS = list(_OS_CELLS)
VENDORS = ["nvidia", "amd", "cpu"]

# As the installer writes them.
CUDA12_OLDER = ["75", "80", "86", "89"]
CUDA13_NEWER = ["100", "120"]


def _smi(rows: str, *, returncode: int = 0):
    """A fake nvidia-smi returning ``rows`` for the compute_cap query."""

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout = rows, stderr = "")

    return _run


def _no_smi(cmd, **kwargs):
    raise FileNotFoundError("nvidia-smi")


@pytest.fixture
def marker_spy(monkeypatch):
    """Records marker reads so a cell can assert it never happened."""
    calls: list = []
    from utils import llama_cpp_freshness as freshness

    def _spy(binary_path):
        calls.append(binary_path)
        return None

    monkeypatch.setattr(freshness, "read_install_marker", _spy)
    return calls


@pytest.fixture
def smi_spy(monkeypatch):
    """Records every nvidia-smi invocation the gate makes."""
    calls: list = []

    def _spy(cmd, **kwargs):
        calls.append(cmd)
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _spy)
    return calls


@pytest.fixture(autouse = True)
def _no_inherited_visibility(monkeypatch):
    """A box pinning CUDA_VISIBLE_DEVICES would otherwise mask rows out of
    every fake nvidia-smi table below."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising = False)


@pytest.fixture
def host(monkeypatch):
    """Applies one OS cell."""

    def _apply(os_key: str):
        sys_platform, system = _OS_CELLS[os_key]
        monkeypatch.setattr("sys.platform", sys_platform)
        monkeypatch.setattr("platform.system", lambda: system)

    return _apply


def _gate(monkeypatch, *, sms, caps_rows):
    """Run the gate with a given marker coverage and nvidia-smi output."""
    monkeypatch.setattr(
        LlamaCppBackend, "_installed_llama_cuda_sms", staticmethod(lambda binary = None: sms)
    )
    if caps_rows is None:
        monkeypatch.setattr(subprocess, "run", _no_smi)
    else:
        monkeypatch.setattr(subprocess, "run", _smi(caps_rows))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/x/llama-server")
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_runtime_remedy",
        classmethod(lambda cls, b: "run `unsloth studio update`"),
    )
    return LlamaCppBackend._cuda_sm_gate_error()


class TestTheGateAcrossTheOsAndVendorMatrix:
    @pytest.mark.parametrize("os_key", OS_KEYS)
    @pytest.mark.parametrize("vendor", VENDORS)
    def test_only_an_nvidia_host_can_be_refused(self, os_key, vendor, monkeypatch, host):
        """No nvidia-smi means unknowable coverage, so the gate opens whatever
        the marker claims."""
        host(os_key)
        rows = "0, 7.5\n" if vendor == "nvidia" else None
        error = _gate(monkeypatch, sms = frozenset({86, 89}), caps_rows = rows)
        if vendor == "nvidia":
            assert error and "sm_75" in error, error
        else:
            assert error is None, f"{os_key}/{vendor} was refused: {error}"

    @pytest.mark.parametrize("os_key", OS_KEYS)
    @pytest.mark.parametrize("vendor", VENDORS)
    def test_an_unmarked_install_never_probes_the_gpu(
        self, os_key, vendor, monkeypatch, host, marker_spy, smi_spy
    ):
        """Unknown coverage must short-circuit before shelling out to
        nvidia-smi on every load."""
        host(os_key)
        monkeypatch.setattr(
            LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/x/llama-server")
        )
        assert LlamaCppBackend._cuda_sm_gate_error() is None
        assert marker_spy, "the gate must consult the install marker"
        assert smi_spy == [], f"{os_key}/{vendor} probed the GPU on unknown coverage"

    @pytest.mark.parametrize("os_key", OS_KEYS)
    def test_a_non_cuda_bundle_is_inert(self, os_key, monkeypatch, host):
        """Vulkan/CPU/ROCm bundles declare no supported_sms. That invariant is
        why the call site needs no explicit is_vulkan_backend guard."""
        host(os_key)
        assert _gate(monkeypatch, sms = None, caps_rows = "0, 7.5\n") is None


class TestTheSmFloorDecision:
    @pytest.mark.parametrize(
        "sms, cap, refused",
        [
            (CUDA12_OLDER, "7.5", False),  # exactly the floor
            (CUDA12_OLDER, "7.0", True),  # below every compiled arch
            (CUDA12_OLDER, "9.0", False),  # newer card JITs the PTX forward
            (CUDA13_NEWER, "8.9", True),  # sm_89 below an sm_100 floor
            (CUDA13_NEWER, "12.0", False),
            (["120"], "12.1", False),  # GB10 on a 5090 bundle, same major
            (["121"], "12.0", True),  # sm_121 PTX cannot JIT down to sm_120
        ],
    )
    def test_floor_not_exact_membership(self, sms, cap, refused, monkeypatch):
        """Only the too-old direction is broken: an exact-SM test would refuse
        the legacy sm_50-61 PTX bundle that drives an sm_86 host fine."""
        error = _gate(monkeypatch, sms = frozenset(int(s) for s in sms), caps_rows = f"0, {cap}\n")
        assert bool(error) is refused, error

    def test_a_mixed_host_passes_on_its_newest_card(self, monkeypatch):
        assert _gate(monkeypatch, sms = frozenset({86}), caps_rows = "0, 7.5\n1, 8.6\n") is None

    def test_every_card_too_old_names_them_all(self, monkeypatch):
        error = _gate(monkeypatch, sms = frozenset({86}), caps_rows = "0, 7.5\n1, 7.0\n")
        assert error and "GPU 0 is sm_75" in error and "GPU 1 is sm_70" in error

    @pytest.mark.parametrize("rows", ["", "0, N/A\n", "garbage\n", "0\n"])
    def test_unreadable_caps_fail_open(self, rows, monkeypatch):
        assert _gate(monkeypatch, sms = frozenset({86}), caps_rows = rows) is None

    def test_a_failing_probe_fails_open(self, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend,
            "_installed_llama_cuda_sms",
            staticmethod(lambda binary = None: frozenset({86})),
        )
        monkeypatch.setattr(subprocess, "run", _smi("0, 7.5\n", returncode = 9))
        monkeypatch.setattr(
            LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/x/llama-server")
        )
        assert LlamaCppBackend._cuda_sm_gate_error() is None


class TestVisibilityMasks:
    """Mask entries are ordinals in CUDA's order while nvidia-smi reports
    physical PCI indices, so the mask is trusted only when both agree."""

    def test_a_numeric_mask_hiding_the_old_card_opens_the_gate(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        assert _gate(monkeypatch, sms = frozenset({86}), caps_rows = "0, 7.5\n1, 8.6\n") is None

    def test_a_numeric_mask_hiding_the_new_card_still_refuses(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        error = _gate(monkeypatch, sms = frozenset({86}), caps_rows = "0, 7.5\n1, 8.6\n")
        assert error and "GPU 0 is sm_75" in error

    def test_an_unordered_numeric_mask_fails_open(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising = False)
        assert _gate(monkeypatch, sms = frozenset({86}), caps_rows = "0, 7.5\n") is None

    def test_an_empty_mask_hides_everything_and_opens_the_gate(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        assert _gate(monkeypatch, sms = frozenset({86}), caps_rows = "0, 7.5\n") is None

    @pytest.mark.parametrize(
        "mask",
        [
            "GPU-c0ffee00-1111-2222-3333-444455556666",
            "MIG-GPU-c0ffee00-1111-2222-3333-444455556666/1/0",
        ],
    )
    def test_a_uuid_or_mig_mask_is_ignored_and_the_whole_host_is_judged(self, mask, monkeypatch):
        """Neither names a physical index we can map back, so the mask is
        dropped and every card is weighed. That errs toward launching and only
        refuses when nothing on the host could run, whichever instance wins."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        assert _gate(monkeypatch, sms = frozenset({86}), caps_rows = "0, 7.5\n1, 8.6\n") is None
        error = _gate(monkeypatch, sms = frozenset({86}), caps_rows = "0, 7.5\n")
        assert error and "sm_75" in error


class TestMarkerParsing:
    @pytest.mark.parametrize("raw", [None, [], "86", ["gfx1100"], ["86", "abc"], ["8.6"], [""]])
    def test_unusable_coverage_reads_as_unknown(self, raw, monkeypatch, tmp_path):
        from utils import llama_cpp_freshness as freshness
        monkeypatch.setattr(
            freshness,
            "read_install_marker",
            lambda b: {"supported_sms": raw} if raw is not None else {},
        )
        assert LlamaCppBackend._installed_llama_cuda_sms("/x/llama-server") is None

    @pytest.mark.parametrize("raw", [["86"], [86], [" 86 "], ["75", "86"]])
    def test_the_installer_string_format_round_trips(self, raw, monkeypatch):
        """The installer writes sorted digit strings; the reader must take
        those plus the ints an older marker may hold."""
        from utils import llama_cpp_freshness as freshness

        monkeypatch.setattr(freshness, "read_install_marker", lambda b: {"supported_sms": raw})
        got = LlamaCppBackend._installed_llama_cuda_sms("/x/llama-server")
        assert got == frozenset(int(str(s).strip()) for s in raw)
