# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression guard for https://github.com/unslothai/unsloth/issues/10951.

A standalone API process reaches the local-model catalog before the FastAPI
lifespan hardware detection has run (or without it ever running).
``_host_serves_mlx()`` read ``DEVICE`` without ever triggering detection, so an
undecided host read as "not MLX" forever and every MLX weights entry was
withheld from the catalog — an installed model requested by its exact id got a
404 ``model_not_found`` that also misreported the model as not downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[2]
BACKEND = WORKDIR / "studio" / "backend"


def _backend_on_path() -> None:
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))


def test_undecided_device_kicks_background_detection(monkeypatch):
    """DEVICE unset must kick the single background detector, not read as a
    permanent non-MLX host."""
    _backend_on_path()
    from utils.hardware import hardware as hw

    kicked = []
    monkeypatch.setattr(hw, "DEVICE", None)
    monkeypatch.setattr(hw, "start_background_detection", lambda: kicked.append(True))

    from core.inference.local_model_resolver import _host_serves_mlx

    assert _host_serves_mlx() is False  # detection has not settled yet
    assert len(kicked) == 1, "an undecided DEVICE must start background detection"


def test_decided_mlx_host_serves_without_kick(monkeypatch):
    _backend_on_path()
    from utils.hardware import hardware as hw

    kicked = []
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.MLX)
    monkeypatch.setattr(hw, "start_background_detection", lambda: kicked.append(True))

    from core.inference.local_model_resolver import _host_serves_mlx

    assert _host_serves_mlx() is True
    assert kicked == []


def test_decided_cuda_host_does_not_serve_or_kick(monkeypatch):
    _backend_on_path()
    from utils.hardware import hardware as hw

    kicked = []
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "start_background_detection", lambda: kicked.append(True))

    from core.inference.local_model_resolver import _host_serves_mlx

    assert _host_serves_mlx() is False
    assert kicked == []
