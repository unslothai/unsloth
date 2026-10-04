# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`studio update` torch repairs on the cu130 torch 2.13 route keep the resident release.

install.sh gives new Linux x86_64 cu130 Python 3.13 installs torch 2.13; a repair on that
route must not move an existing install to another torch, and every other route must get
exactly the specs it got before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_INSTALL_SCRIPT = Path(__file__).resolve().parents[2] / "install_python_stack.py"
CU130 = "https://download.pytorch.org/whl/cu130"


@pytest.fixture
def mod(monkeypatch):
    sys.modules.pop("install_python_stack", None)
    monkeypatch.syspath_prepend(str(_INSTALL_SCRIPT.parent))
    import install_python_stack

    return install_python_stack


def _route(
    monkeypatch,
    mod,
    *,
    platform = "linux",
    machine = "x86_64",
    py = (3, 13),
    torch = None,
):
    monkeypatch.setattr(mod.sys, "platform", platform)
    monkeypatch.setattr(mod.platform, "machine", lambda: machine)
    monkeypatch.setattr(mod.sys, "version_info", (*py, 0, "final", 0))
    monkeypatch.setattr(mod, "_resident_torch_release", lambda: torch)


@pytest.mark.parametrize(
    "resident, expected",
    [
        ("2.11.0", ("torch==2.11.0", "torchvision==0.26.*", "torchaudio==2.11.*")),
        ("2.10.0", ("torch==2.10.0", "torchvision==0.25.*", "torchaudio==2.10.*")),
        ("2.12.1", ("torch==2.12.1", "torchvision==0.27.*", "torchaudio==2.11.*")),
        ("2.13.0", ("torch==2.13.0", "torchvision==0.28.*", "torchaudio==2.11.*")),
        ("2.14.0", ("torch==2.14.0", "torchvision==0.29.*", "torchaudio==2.11.*")),
    ],
)
def test_a_resident_release_is_kept(monkeypatch, mod, resident, expected):
    _route(monkeypatch, mod, torch = resident)
    assert mod._cuda_repair_torch_specs(CU130, mod._CUDA_TORCH_PKG_SPEC) == expected
    assert mod._cuda_repair_torch_specs(CU130, mod._TORCH_FLAVOR_REPAIR_PKG_SPEC) == expected


# 2.4-2.8 are older than anything the cu130 index serves, so pinning them would fail the repair.
@pytest.mark.parametrize("resident", [None, "2.15.0", "2.3.1", "2.6.0", "2.8.0"])
def test_no_keepable_release_repairs_to_the_default_range(monkeypatch, mod, resident):
    # 2.13 without a kept release would bypass install.sh's PyPI gate and any uv upload cutoff.
    _route(monkeypatch, mod, torch = resident)
    for default in (mod._CUDA_TORCH_PKG_SPEC, mod._TORCH_FLAVOR_REPAIR_PKG_SPEC):
        assert mod._cuda_repair_torch_specs(CU130, default) is default


@pytest.mark.parametrize(
    "index_url, kwargs",
    [
        ("https://download.pytorch.org/whl/cu128", {}),
        ("https://download.pytorch.org/whl/cu126", {}),
        ("https://download.pytorch.org/whl/cpu", {}),
        ("https://download.pytorch.org/whl/rocm7.2", {}),
        (CU130, {"platform": "win32"}),
        (CU130, {"platform": "darwin"}),
        (CU130, {"machine": "aarch64"}),
        (CU130, {"py": (3, 12)}),
        (CU130, {"py": (3, 11)}),
        (None, {}),
    ],
)
def test_every_other_route_is_unchanged(monkeypatch, mod, index_url, kwargs):
    _route(monkeypatch, mod, torch = "2.13.0", **kwargs)
    for default in (mod._CUDA_TORCH_PKG_SPEC, mod._TORCH_FLAVOR_REPAIR_PKG_SPEC):
        assert mod._cuda_repair_torch_specs(index_url, default) is default


def test_resident_release_reads_metadata_and_rejects_non_releases(monkeypatch, mod):
    import importlib.metadata as md

    for raw, want in (
        ("2.13.0+cu130", "2.13.0"),
        ("2.11.0", "2.11.0"),
        ("2.14.0.dev20260801+cu130", None),
        ("2.13.0a0+git1234", None),
    ):
        monkeypatch.setattr(md, "version", lambda _name, raw = raw: raw)
        assert mod._resident_torch_release() == want, raw

    def missing(_name):
        raise md.PackageNotFoundError("torch")

    monkeypatch.setattr(md, "version", missing)
    assert mod._resident_torch_release() is None


def test_both_cuda_repairs_use_the_route_aware_specs(mod):
    import re

    source = _INSTALL_SCRIPT.read_text(encoding = "utf-8")
    calls = re.findall(r"_cuda_repair_torch_specs\(\s*index_url,\s*(\w+)\s*\)", source)
    assert sorted(calls) == [
        "_CUDA_TORCH_PKG_SPEC",
        "_CUDA_TORCH_PKG_SPEC",
        "_TORCH_FLAVOR_REPAIR_PKG_SPEC",
    ]


@pytest.mark.parametrize(
    "platform, resident, pinned",
    [
        ("linux", "2.13.0", True),
        ("linux", "2.14.0", True),
        ("linux", "2.12.1", False),
        ("linux", "2.11.0", False),
        ("linux", None, False),
        ("win32", "2.13.0", False),
        ("darwin", "2.13.0", False),
    ],
)
def test_core_update_freezes_only_torch_past_the_released_ceiling(
    monkeypatch, tmp_path, mod, platform, resident, pinned
):
    monkeypatch.setattr(mod.sys, "platform", platform)
    monkeypatch.setattr(mod, "_resident_torch_release", lambda: resident)
    monkeypatch.setattr(
        mod,
        "_resident_torch_trio_pins",
        lambda: ["torch==2.13.0+cu130", "torchvision==0.28.0+cu130", "torchaudio==2.11.0+cu130"],
    )
    inherited = tmp_path / "overrides.txt"
    inherited.write_text("Torch<2.13\ntorchvision>=0.1\nnumpy<3\n", encoding = "utf-8")
    monkeypatch.setenv("UV_OVERRIDE", str(inherited))
    with mod._FreezeNewTorchForCoreUpdate() as freeze:
        value = mod.os.environ["UV_OVERRIDE"]
        assert mod._TORCH_FREEZE_ACTIVE is pinned
        if pinned:
            # One file: the inherited trio lines would make uv's resolution unsatisfiable.
            assert Path(value).read_text(encoding = "utf-8").split() == [
                "torch==2.13.0+cu130",
                "torchvision==0.28.0+cu130",
                "torchaudio==2.11.0+cu130",
                "numpy<3",
            ]
        else:
            assert value == str(inherited)
    assert mod.os.environ["UV_OVERRIDE"] == str(inherited)
    assert mod._TORCH_FREEZE_ACTIVE is False
    if pinned:
        assert not freeze._path.exists()


def test_core_update_freeze_restores_an_unset_override(monkeypatch, mod):
    monkeypatch.setattr(mod.sys, "platform", "linux")
    monkeypatch.setattr(mod, "_resident_torch_release", lambda: "2.13.0")
    monkeypatch.setattr(mod, "_resident_torch_trio_pins", lambda: ["torch==2.13.0+cu130"])
    monkeypatch.delenv("UV_OVERRIDE", raising = False)
    with mod._FreezeNewTorchForCoreUpdate():
        assert mod.os.environ["UV_OVERRIDE"].endswith(".txt")
    assert "UV_OVERRIDE" not in mod.os.environ


def test_both_core_updates_run_under_the_freeze():
    source = _INSTALL_SCRIPT.read_text(encoding = "utf-8")
    assert (
        source.count(
            'with _FreezeNewTorchForCoreUpdate():\n            pip_install(\n                "Updating core packages"'
        )
        == 2
    )


def test_a_uv_failure_under_the_freeze_never_falls_back_to_pip(monkeypatch, mod):
    import subprocess

    monkeypatch.setattr(mod, "USE_UV", True)
    monkeypatch.setattr(mod, "_TORCH_FREEZE_ACTIVE", True)
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, b"no solution")
    )
    ran = []
    monkeypatch.setattr(mod, "run", lambda *a, **k: ran.append(a))
    with pytest.raises(SystemExit):
        mod._pip_install_once("Updating core packages", "unsloth")
    assert ran == []


def test_inherited_relative_includes_still_resolve(monkeypatch, tmp_path, mod):
    monkeypatch.setattr(mod.sys, "platform", "linux")
    monkeypatch.setattr(mod, "_resident_torch_release", lambda: "2.13.0")
    monkeypatch.setattr(mod, "_resident_torch_trio_pins", lambda: ["torch==2.13.0+cu130"])
    (tmp_path / "nested.txt").write_text("numpy<3\n", encoding = "utf-8")
    inherited = tmp_path / "overrides.txt"
    inherited.write_text(
        "-r nested.txt\n--constraint=/abs/c.txt\n-r https://example.com/r.txt\n", encoding = "utf-8"
    )
    monkeypatch.setenv("UV_OVERRIDE", str(inherited))
    with mod._FreezeNewTorchForCoreUpdate():
        merged = Path(mod.os.environ["UV_OVERRIDE"]).read_text(encoding = "utf-8").splitlines()
    assert merged == [
        "torch==2.13.0+cu130",
        f"-r {(tmp_path / 'nested.txt').resolve()}",
        "--constraint=/abs/c.txt",
        "-r https://example.com/r.txt",
    ]
