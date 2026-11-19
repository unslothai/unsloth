# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The Apple Silicon torch 2.13 route in install.sh agrees with the pins Studio installs beside it."""

from __future__ import annotations
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = (REPO_ROOT / "install.sh").read_text(encoding = "utf-8")


def _assignment(name: str) -> str:
    match = re.search(rf'^{name}="([^"]*)"$', INSTALL_SH, re.MULTILINE)
    assert match, name
    return match.group(1)


def _ips():
    sys.path.insert(0, str(REPO_ROOT / "studio"))
    try:
        import install_python_stack
    finally:
        sys.path.pop(0)
    return install_python_stack


def _route_body() -> str:
    match = re.search(
        r'^if \[ -z "\$_PREV_TORCH_PIN" \] && \[ "\$_MACOS_NEW_INSTALL_ROUTE" = true \]; then\n(.*?)^fi$',
        INSTALL_SH,
        re.MULTILINE | re.DOTALL,
    )
    assert match
    return match.group(1)


def test_macos_route_is_one_minor_with_its_torchvision():
    assert _assignment("_MACOS_NEW_INSTALL_TORCH") == "torch>=2.13.0,<2.14.0"
    # torchvision 0.N pairs with torch 2.(N - 15).
    assert 'TORCHVISION_CONSTRAINT="torchvision>=0.28.0,<0.29.0"' in _route_body()
    # torchaudio keeps the shared window: 2.11 is the last release and pairs with every newer torch.
    assert "TORCHAUDIO_CONSTRAINT" not in _route_body()
    assert _assignment("_TORCHAUDIO_CEILING") == "2.12.0"


def test_macos_preservation_ceiling_matches_cu130():
    # Both routes keep a resident 2.4 - 2.14 release; install_python_stack mirrors the cu130 one.
    assert _assignment("_MACOS_TORCH_CEILING") == _assignment("_CU130_TORCH_CEILING") == "2.15.0"
    assert _ips()._CU130_PRESERVE_TORCH_CEILING_MINOR == 15


def test_generic_ceiling_is_unchanged():
    # Linux / Windows CUDA, ROCm and CPU defaults stay below 2.12.
    assert _assignment("_TORCH_CEILING") == "2.12.0"
    assert _ips()._CUDA_TORCH_PKG_SPEC[0] == "torch>=2.4,<2.12.0"


def test_torchcodec_on_the_macos_route_is_abi_stable_and_macos_14():
    ips = _ips()
    for version in ("2.13.0", "2.14.1"):
        assert ips._select_torchcodec_spec(version) == ips._TORCHCODEC_ABI_STABLE_SPEC
    # 0.12+ wheels are macosx_14_0, which is the route's own macOS floor.
    assert ips._TORCHCODEC_MACOS_14_ONLY_FROM == (0, 12, 0)
    assert "-ge 14" in INSTALL_SH.split("_macos_torch213_host_ok() {", 1)[1].split("\n}", 1)[0]


def test_mflux_pin_floor_is_met_by_the_route():
    # mflux 0.20.0+ declares torch>=2.13.0 (installed --no-deps); the route is what satisfies it.
    pins = _ips()._MLX_DIFFUSION_PINS
    assert pins and pins[0].startswith("mflux==")
    version = tuple(int(p) for p in pins[0].split("==", 1)[1].split(".")[:2])
    if version >= (0, 20):
        floor = _assignment("_MACOS_NEW_INSTALL_TORCH").split(",", 1)[0]
        assert floor == "torch>=2.13.0"
