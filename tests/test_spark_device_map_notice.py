# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The single-Spark device_map notice in loader_utils: silent off a Spark, and never breaks a load.

Under tests/ so the suite's conftest lets `unsloth` import on a CPU-only runner."""

import pytest


_PLATFORMS = [
    ("Linux", "x86_64"),
    ("Linux", "aarch64"),  # not a Spark: GH200, Jetson
    ("Windows", "AMD64"),
    ("Windows", "ARM64"),
    ("Darwin", "arm64"),
    ("Darwin", "x86_64"),
]

_DEVICE_MAPS = [
    "balanced",
    "balanced_low_0",
    "auto",
    "unsloth",
    "unsloth_balanced",
    "sequential",
    "cuda:0",
    "cpu",
    "",
    None,
    0,
    {"": "cuda:0"},
    ["cuda:0"],
]


def _spark_notice(
    monkeypatch,
    system,
    machine,
    *,
    opener = None,
    device_count = 1,
):
    pytest.importorskip("torch")
    import builtins
    import platform as _platform
    import torch
    from unsloth.models import loader_utils as LU

    monkeypatch.setattr(_platform, "system", lambda: system)
    monkeypatch.setattr(_platform, "machine", lambda: machine)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)
    if opener is not None:
        real_open = builtins.open

        def guarded(path, *a, **k):
            p = str(path)
            if p.startswith("/etc/dgx") or p.startswith("/sys/"):
                raise opener("simulated")
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", guarded)
    LU._SPARK_NOTICE_SHOWN[0] = False
    return LU


@pytest.mark.parametrize("system,machine", _PLATFORMS)
def test_spark_notice_is_silent_off_a_spark(monkeypatch, capsys, system, machine):
    """No user on another platform should ever see this."""
    LU = _spark_notice(monkeypatch, system, machine, opener = FileNotFoundError)
    capsys.readouterr()  # drop unsloth's import banner, which is not ours to assert on
    for device_map in _DEVICE_MAPS:
        LU.notify_device_map_cannot_span_sparks(device_map)
    assert "DGX Spark" not in capsys.readouterr().out


@pytest.mark.parametrize("system,machine", _PLATFORMS)
@pytest.mark.parametrize(
    "opener", [OSError, PermissionError, IsADirectoryError, ValueError, RuntimeError]
)
def test_spark_notice_never_breaks_a_load(monkeypatch, system, machine, opener):
    """A probe that raises must not propagate: the caller is loading a model."""
    LU = _spark_notice(monkeypatch, system, machine, opener = opener)
    for device_map in _DEVICE_MAPS:
        LU.notify_device_map_cannot_span_sparks(device_map)


def test_spark_notice_survives_a_broken_cuda_probe(monkeypatch):
    pytest.importorskip("torch")
    import torch
    from unsloth.models import loader_utils as LU

    def boom():
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(torch.cuda, "device_count", boom)
    LU._SPARK_NOTICE_SHOWN[0] = False
    LU.notify_device_map_cannot_span_sparks("balanced")
