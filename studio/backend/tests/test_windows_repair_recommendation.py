# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An existing Windows home on an RTX 50-series card gets a Repair recommendation when its torch or
xFormers is older than the managed pair."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from utils.hardware import windows_repair as wr

RTX50, RTX40 = (12, 0), (8, 9)


def _rec(
    torch = "2.10.0+cu128",
    xformers = None,
    caps = (RTX50,),
    platform = "win32",
    route_host = True,
    driver = (13, 0),
):
    versions = {"torch": torch, "xformers": xformers}
    return wr.windows_repair_recommendation(
        platform = platform,
        dist_version = versions.get,
        capabilities = lambda: dict(enumerate(caps)),
        managed_route_host = lambda: route_host,
        driver_cuda_version = lambda: driver,
    )


def test_a_driver_too_old_for_the_managed_route_asks_for_a_driver_update_first():
    # Repair alone would keep the cu128 torch, so the recommendation names the driver instead.
    assert _rec(driver = (12, 8))["reason"] == "driver_older_than_managed"
    assert _rec(driver = None)["reason"] == "torch_older_than_managed"  # unknown driver: Repair


def test_a_host_off_the_managed_route_gets_no_recommendation():
    # Windows on ARM, or a venv Python other than 3.13: Repair would not move torch.
    assert _rec(route_host = False) is None


def test_the_driver_header_is_parsed_as_install_ps1_reads_it(monkeypatch):
    from types import SimpleNamespace

    from utils.hardware import gpu_query

    header = "| NVIDIA-SMI 576.88   Driver Version: 576.88   CUDA Version: 12.9     |"
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", lambda *a, **k: SimpleNamespace(stdout = header))
    wr._driver_cuda_version.cache_clear()
    try:
        assert wr._driver_cuda_version() == (12, 9)
    finally:
        wr._driver_cuda_version.cache_clear()


def test_the_route_gate_matches_install_ps1():
    install = Path(__file__).resolve().parents[3] / "install.ps1"
    body = re.search(
        r"function Test-Cu130Torch213Platform \{.*?\n    \}",
        install.read_text(encoding = "utf-8"),
        re.S,
    ).group(0)
    assert '-ne "cu130"' in body and '-ne "x86_64"' in body and '"3.13 win-amd64"' in body


def test_an_old_torch_on_rtx50_recommends_repair():
    rec = _rec()
    assert rec["reason"] == "torch_older_than_managed"
    assert rec["torch"] == "2.10.0+cu128"
    assert (rec["managed_torch"], rec["managed_xformers"]) == ("2.13", "0.0.35")


def test_an_old_xformers_on_the_managed_torch_recommends_repair():
    assert _rec(torch = "2.13.0+cu130", xformers = "0.0.33")["reason"] == "xformers_older_than_managed"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(torch = "2.13.0+cu130", xformers = "0.0.35"),  # the managed pair
        dict(torch = "2.14.0+cu134"),  # newer, no xFormers
        dict(caps = (RTX40,)),  # older cards train fine on this torch
        dict(caps = ()),  # no readable card
        dict(torch = "2.10.0+cpu"),  # a CPU wheel is a different repair
        dict(torch = None),  # no torch (Chat only)
        dict(platform = "linux"),
        dict(platform = "darwin"),
    ],
)
def test_nothing_is_recommended_otherwise(kwargs):
    assert _rec(**kwargs) is None


def test_a_mixed_host_answers_for_its_rtx50_card():
    assert _rec(caps = (RTX40, RTX50))["reason"] == "torch_older_than_managed"


def test_the_floor_matches_install_ps1():
    """The managed torch floor is install.ps1's new-install CUDA spec; bumping one must bump the other."""
    install = Path(__file__).resolve().parents[3] / "install.ps1"
    spec = re.search(
        r'\$script:Cu130NewInstallTorchSpec = "torch>=(\d+)\.(\d+)',
        install.read_text(encoding = "utf-8"),
    )
    assert spec, "Cu130NewInstallTorchSpec not found in install.ps1"
    assert (int(spec.group(1)), int(spec.group(2))) == wr.WINDOWS_MANAGED_TORCH_FLOOR


def test_system_route_ships_the_recommendation(monkeypatch):
    import main

    monkeypatch.setattr(
        wr, "windows_repair_recommendation", lambda: {"reason": "torch_older_than_managed"}
    )
    assert main._repair_recommendation() == {"reason": "torch_older_than_managed"}

    def boom():
        raise RuntimeError("probe failed")

    monkeypatch.setattr(wr, "windows_repair_recommendation", boom)
    assert main._repair_recommendation() is None
    src = Path(main.__file__).read_text(encoding = "utf-8")
    assert '"repair_recommendation": _repair_recommendation(),' in src


def test_the_module_stays_torch_free():
    src = Path(wr.__file__).read_text(encoding = "utf-8")
    assert "import torch" not in src
