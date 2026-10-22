# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""detect_hardware() records why a host is chat-only; Apple Silicon without MLX gives mlx_unavailable."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.hardware.hardware as hw  # noqa: E402


@pytest.fixture(autouse = True)
def _no_torch(monkeypatch):
    # Force the non-CUDA/XPU path regardless of the test host's real GPUs.
    monkeypatch.setattr(hw, "_has_torch", lambda: False)
    # The no_torch verdict reads the venv's install manifest; pin it so the answer does
    # not depend on which venv runs these tests.
    monkeypatch.setattr(hw, "_installed_without_torch", lambda: False)
    monkeypatch.setattr(hw, "_NO_TORCH_SETTLED_EPOCH", None)
    # detect_hardware sets these globals directly; restore them so they do not leak.
    saved = (hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM)
    try:
        yield
    finally:
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM = saved


def test_apple_silicon_without_mlx_is_chat_only_with_reason(monkeypatch):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: False)
    hw.detect_hardware()
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"


def test_apple_silicon_with_mlx_enables_training(monkeypatch):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: True)
    hw.detect_hardware()
    assert hw.CHAT_ONLY is False
    assert hw.CHAT_ONLY_REASON is None


def test_apple_silicon_with_incomplete_mlx_stack_stays_chat_only(monkeypatch):
    # mlx imports but the full mlx/mlx-lm/mlx-vlm stack does not; stay chat-only.
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "_has_mlx", lambda: True)
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: False)
    assert hw.detect_hardware() == hw.DeviceType.CPU
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"


def _no_torch_apple_silicon(monkeypatch, *, mlx_on_disk: bool):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "_installed_without_torch", lambda: True)
    monkeypatch.setattr(hw, "_mlx_distribution_installed", lambda: mlx_on_disk)
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: False)


def test_a_no_torch_install_off_mac_is_off_by_request_not_missing_a_gpu(monkeypatch):
    # A chat-only Windows / Linux install on a GPU host: "needs an NVIDIA or AMD GPU" is wrong.
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_installed_without_torch", lambda: True)
    assert hw.detect_hardware() == hw.DeviceType.CPU
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "no_torch"
    assert hw.current_chat_only_verdict() == ("no_torch", None)


def test_a_torch_install_off_mac_without_an_accelerator_stays_no_gpu(monkeypatch):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    hw.detect_hardware()
    assert hw.CHAT_ONLY_REASON == "no_gpu"


def test_apple_silicon_no_torch_install_without_mlx_is_off_by_request(monkeypatch):
    _no_torch_apple_silicon(monkeypatch, mlx_on_disk = False)
    assert hw.detect_hardware() == hw.DeviceType.CPU
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "no_torch"
    assert hw.CHAT_ONLY_DETAIL is None


def test_apple_silicon_no_torch_install_with_mlx_on_disk_waits_for_the_probe(monkeypatch):
    # A hand-installed stack can lose the warm's import race and recover; keep polling.
    _no_torch_apple_silicon(monkeypatch, mlx_on_disk = True)
    hw.detect_hardware()
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"
    assert hw.verdict_blames_the_mlx_stack() is True


def test_the_probe_settles_a_no_torch_host_once_the_stack_measures_unusable(monkeypatch):
    _no_torch_apple_silicon(monkeypatch, mlx_on_disk = True)
    hw.detect_hardware()
    assert hw.settle_the_no_torch_verdict(hw.current_detection_epoch()) is True
    assert hw.CHAT_ONLY_REASON == "no_torch"
    assert hw.CHAT_ONLY_DETAIL is None
    assert hw.verdict_blames_the_mlx_stack() is False
    hw.detect_hardware()
    assert hw.CHAT_ONLY_REASON == "no_torch"


def test_a_probe_retired_by_a_shutdown_settles_nothing(monkeypatch):
    _no_torch_apple_silicon(monkeypatch, mlx_on_disk = True)
    hw.detect_hardware()
    assert hw.settle_the_no_torch_verdict(hw.current_detection_epoch() - 1) is False
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"
    assert hw._NO_TORCH_SETTLED_EPOCH is None


def test_a_settled_flag_does_not_outlive_its_lifespan(monkeypatch):
    _no_torch_apple_silicon(monkeypatch, mlx_on_disk = True)
    monkeypatch.setattr(hw, "_NO_TORCH_SETTLED_EPOCH", hw.current_detection_epoch() - 1)
    hw.detect_hardware()
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"


def test_settling_leaves_any_other_verdict_alone(monkeypatch):
    for chat_only, reason in ((False, None), (True, "intel_mac"), (True, "no_torch")):
        hw.CHAT_ONLY, hw.CHAT_ONLY_REASON = chat_only, reason
        assert hw.settle_the_no_torch_verdict(hw.current_detection_epoch()) is False
        assert (hw.CHAT_ONLY, hw.CHAT_ONLY_REASON) == (chat_only, reason)


def test_a_declined_settle_does_not_mark_the_epoch(monkeypatch):
    # A declined settle must not record the epoch, or a later failure reads as settled.
    _no_torch_apple_silicon(monkeypatch, mlx_on_disk = True)
    hw.CHAT_ONLY, hw.CHAT_ONLY_REASON = False, None
    assert hw.settle_the_no_torch_verdict(hw.current_detection_epoch()) is False
    assert hw._NO_TORCH_SETTLED_EPOCH != hw.current_detection_epoch()

    hw.detect_hardware()
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"


def test_apple_silicon_no_torch_install_with_usable_mlx_enables_training(monkeypatch):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "_installed_without_torch", lambda: True)
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: True)
    hw.detect_hardware()
    assert hw.CHAT_ONLY is False
    assert hw.CHAT_ONLY_REASON is None


def test_intel_mac_reason(monkeypatch):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    # Pinned: a translated (Rosetta) test Python would otherwise answer "rosetta".
    monkeypatch.setattr(hw, "_running_under_rosetta", lambda: False)
    monkeypatch.setattr(hw, "_has_mlx", lambda: False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    hw.detect_hardware()
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "intel_mac"


def _fake_sysctl(arm64: str, calls: list):
    def run(cmd, *args, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode = 0, stdout = arm64 + "\n", stderr = "")

    return run


@pytest.mark.parametrize(
    "machine, arm64, expected",
    [
        ("x86_64", "1", "rosetta"),
        ("x86_64", "0", "intel_mac"),
    ],
)
def test_rosetta_reason(monkeypatch, machine, arm64, expected):
    calls = []
    monkeypatch.setattr(hw, "_ROSETTA", None, raising = False)
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw, "_has_mlx", lambda: False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hw.platform, "machine", lambda: machine)
    monkeypatch.setattr(hw.subprocess, "run", _fake_sysctl(arm64, calls))
    hw.detect_hardware()
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == expected
    assert calls == [["sysctl", "-n", "hw.optional.arm64"]]


def test_rosetta_probe_fails_open_to_intel_mac(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no sysctl")

    monkeypatch.setattr(hw, "_ROSETTA", None, raising = False)
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw, "_has_mlx", lambda: False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hw.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(hw.subprocess, "run", boom)
    hw.detect_hardware()
    assert hw.CHAT_ONLY_REASON == "intel_mac"


def test_native_arm64_mac_never_probes_rosetta(monkeypatch):
    calls = []
    monkeypatch.setattr(hw, "_ROSETTA", None, raising = False)
    monkeypatch.setattr(hw, "_has_mlx", lambda: False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hw.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hw.subprocess, "run", _fake_sysctl("1", calls))
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: False, raising = False)
    hw.detect_hardware()
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"
    assert ["sysctl", "-n", "hw.optional.arm64"] not in calls


def test_cpu_only_non_mac_reason(monkeypatch):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw, "_has_mlx", lambda: False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    hw.detect_hardware()
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "no_gpu"


@pytest.mark.parametrize("tegra, chosen", [(True, True), (False, False)])
def test_an_automatic_cpu_record_counts_as_chosen_only_on_a_jetson(monkeypatch, tegra, chosen):
    # A pre-CUDA-13 Jetson always resolves to cpu, so offering a GPU repair there loops back to cpu (#4856).
    from utils import tegra as tegra_mod

    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    monkeypatch.setattr(hw, "_recorded_install_flavor", lambda: ("cpu", False))
    monkeypatch.setattr(tegra_mod, "tegra_before_cuda13", lambda: tegra)
    assert hw._expected_cpu_flavor_was_chosen() is chosen
    # A pinned cpu record stays a choice everywhere.
    monkeypatch.setattr(hw, "_recorded_install_flavor", lambda: ("cpu", True))
    assert hw._expected_cpu_flavor_was_chosen() is True


@pytest.mark.parametrize(
    "var, value",
    [
        ("UNSLOTH_TORCH_INDEX_FAMILY", "cu126"),
        ("UNSLOTH_TORCH_INDEX_URL", "https://example.test/whl/cu126"),
    ],
)
def test_a_stated_gpu_index_beats_the_jetson_cpu_record(monkeypatch, var, value):
    # An explicit GPU pin wins over the automatic cpu history: the CPU build is reported as before.
    from utils import tegra as tegra_mod

    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    monkeypatch.setenv(var, value)
    monkeypatch.setattr(hw, "_recorded_install_flavor", lambda: ("cpu", False))
    monkeypatch.setattr(tegra_mod, "tegra_before_cuda13", lambda: True)
    assert hw._expected_cpu_flavor_was_chosen() is False


@pytest.mark.parametrize(
    "release, compatible, expected",
    [
        ("# R36 (release), REVISION: 4.3\n", b"nvidia,tegra234\0", True),  # JetPack 6 Orin
        ("# R35 (release), REVISION: 5.0\n", b"nvidia,tegra194\0", True),  # JetPack 5 Xavier
        ("# R38 (release), REVISION: 2.0\n", b"nvidia,tegra264\0", False),  # JetPack 7 Thor
        (None, b"nvidia,tegra264\0", False),
        (None, b"nvidia,tegra234\0", True),
        (None, b"nvidia,grace-hopper\0", False),  # not a Jetson
    ],
)
def test_tegra_before_cuda13_reads_the_l4t_release(
    tmp_path, monkeypatch, release, compatible, expected
):
    from utils import tegra as tegra_mod

    rel = tmp_path / "nv_tegra_release"
    if release is not None:
        rel.write_text(release)
    dt = tmp_path / "compatible"
    dt.write_bytes(compatible)
    monkeypatch.setattr(tegra_mod, "_TEGRA_RELEASE", str(rel))
    monkeypatch.setattr(tegra_mod, "_DEVICE_TREE_COMPATIBLE", str(dt))
    tegra_mod.is_tegra.cache_clear()
    tegra_mod.tegra_before_cuda13.cache_clear()
    try:
        assert tegra_mod.tegra_before_cuda13() is expected
    finally:
        tegra_mod.is_tegra.cache_clear()
        tegra_mod.tegra_before_cuda13.cache_clear()
