# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/system's ml_packages feed the Environment rows in Settings."""

from importlib.metadata import PackageNotFoundError


def test_reports_installed_mlx_libraries_and_omits_missing_ones(monkeypatch):
    import importlib.metadata

    import main

    installed = {"torch": "2.11.0", "mlx_lm": "0.32.2", "mlx_vlm": "0.7.8"}

    def fake_version(name):
        if name not in installed:
            raise PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    monkeypatch.setattr(
        main,
        "_get_cached_system_gpu_info",
        lambda logger, refresh_memory = False: ({"available": False}, {"available": False}),
    )
    packages = main.get_system_info(current_subject = "test")["ml_packages"]

    assert packages == installed
