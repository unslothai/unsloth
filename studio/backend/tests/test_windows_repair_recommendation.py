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


@pytest.mark.parametrize(
    "header, expected",
    [
        ("| NVIDIA-SMI 576.88   Driver Version: 576.88   CUDA Version: 12.9     |", (12, 9)),
        # Newer drivers (616.56 on an RTX 5090) print the user-mode driver's version under another label.
        (
            "| NVIDIA-SMI 616.56                 KMD Version: 616.56        CUDA UMD Version: 13.4     |",
            (13, 4),
        ),
        ("| NVIDIA-SMI 616.56   no version here |", None),
    ],
)
def test_both_driver_header_spellings_are_read(monkeypatch, header, expected):
    from types import SimpleNamespace

    from utils.hardware import gpu_query

    monkeypatch.setattr(gpu_query, "run_nvidia_smi", lambda *a, **k: SimpleNamespace(stdout = header))
    wr._driver_cuda_version.cache_clear()
    try:
        assert wr._driver_cuda_version() == expected
    finally:
        wr._driver_cuda_version.cache_clear()


def test_the_header_regex_matches_install_ps1():
    install = Path(__file__).resolve().parents[3] / "install.ps1"
    assert "CUDA(?: UMD)? Version:" in install.read_text(encoding = "utf-8")
    assert "CUDA(?: UMD)? Version:" in Path(wr.__file__).read_text(encoding = "utf-8")


def test_the_driver_header_is_parsed_as_install_ps1_reads_it(monkeypatch):
    from types import SimpleNamespace

    from utils.hardware import gpu_query

    header = "| NVIDIA-SMI 576.88   Driver Version: 576.88   CUDA Version: 12.9     |"
    seen = {}

    def fake(argv, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(stdout = header)

    monkeypatch.setattr(gpu_query, "run_nvidia_smi", fake)
    from utils import subprocess_compat

    monkeypatch.setattr(
        subprocess_compat, "windows_hidden_subprocess_kwargs", lambda: {"creationflags": 0x08000000}
    )
    wr._driver_cuda_version.cache_clear()
    try:
        assert wr._driver_cuda_version() == (12, 9)
        # Launched like nvidia.py's queries: no console window on Windows.
        assert seen.get("creationflags") == 0x08000000 and "env" in seen
    finally:
        wr._driver_cuda_version.cache_clear()


@pytest.mark.parametrize(
    "env, minor",
    [
        ({}, "3.13"),
        ({"UNSLOTH_PYTHON": "3.12"}, "3.12"),
        ({"UNSLOTH_PYTHON": "3.13.9"}, "3.13"),
        ({"UNSLOTH_PYTHON": ""}, "3.13"),
    ],
)
def test_repair_builds_its_venv_with_the_python_install_ps1_selects(env, minor):
    # Repair (UNSLOTH_TORCH_UPGRADE=1) does not keep the running venv's minor: a 3.12 home moves to 3.13.
    assert wr._repair_python_minor(env) == minor


@pytest.mark.parametrize(
    "env, eligible",
    [
        ({}, True),
        ({"UNSLOTH_TORCH_INDEX_FAMILY": "cu128"}, False),
        ({"UNSLOTH_TORCH_INDEX_FAMILY": "cu130"}, True),
        ({"UNSLOTH_TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu128/"}, False),
        ({"UNSLOTH_TORCH_INDEX_URL": "https://mirror.example/whl/cu130?x=1"}, True),
        ({"UNSLOTH_PYTHON": "3.12"}, False),
    ],
)
def test_an_index_override_off_cu130_closes_the_route(monkeypatch, env, eligible):
    # install.ps1 honors these overrides before the driver, and the 2.13 route needs the cu130 leaf.
    monkeypatch.setattr(wr._platform, "machine", lambda: "AMD64")
    assert wr._managed_route_host(env) is eligible


@pytest.mark.parametrize(
    "env",
    [
        {"UV_INDEX_URL": "https://mirror.example/simple"},
        {"UV_DEFAULT_INDEX": "https://mirror.example/simple"},
        {"UV_EXTRA_INDEX_URL": "https://mirror.example/simple"},
        {"UV_OFFLINE": "1"},
        {"UV_EXCLUDE_NEWER": "2026-01-01"},
    ],
)
def test_a_package_mirror_or_offline_uv_closes_the_route(monkeypatch, env):
    # install.ps1's Test-PypiUnslothAdmitsTorch returns false here, so Repair keeps the old torch window.
    monkeypatch.setattr(wr._platform, "machine", lambda: "AMD64")
    assert wr._managed_route_host(env) is False


def test_an_index_in_the_uv_config_file_closes_the_route(monkeypatch, tmp_path):
    monkeypatch.setattr(wr._platform, "machine", lambda: "AMD64")
    (tmp_path / "uv").mkdir()
    config = tmp_path / "uv" / "uv.toml"
    config.write_text(
        "[[index]]\nurl = 'https://mirror.example/simple'\ndefault = true\n", encoding = "utf-8"
    )
    assert wr._managed_route_host({"APPDATA": str(tmp_path)}) is False
    config.write_text("cache-dir = 'D:/uv'\n", encoding = "utf-8")
    assert wr._managed_route_host({"APPDATA": str(tmp_path)}) is True


def test_uv_config_precedence_follows_install_ps1(monkeypatch, tmp_path):
    monkeypatch.setattr(wr._platform, "machine", lambda: "AMD64")
    (tmp_path / "uv").mkdir()
    (tmp_path / "uv" / "uv.toml").write_text(
        "index-url = 'https://mirror.example/simple'\n", encoding = "utf-8"
    )
    clean = tmp_path / "clean.toml"
    clean.write_text("cache-dir = 'D:/uv'\n", encoding = "utf-8")
    user = {"APPDATA": str(tmp_path)}
    assert wr._managed_route_host(user) is False  # the user file names a mirror
    # UV_NO_CONFIG skips discovery, and UV_CONFIG_FILE is exclusive: the mirror-free file wins.
    assert wr._managed_route_host({**user, "UV_NO_CONFIG": "1"}) is True
    assert wr._managed_route_host({**user, "UV_CONFIG_FILE": str(clean)}) is True
    assert (
        wr._managed_route_host({**user, "UV_CONFIG_FILE": str(clean), "UV_NO_CONFIG": "1"}) is True
    )
    # A system file is read unless UV_NO_SYSTEM_CONFIG.
    system = tmp_path / "sys"
    (system / "uv").mkdir(parents = True)
    (system / "uv" / "uv.toml").write_text("[[index]]\nurl = 'x'\n", encoding = "utf-8")
    assert wr._managed_route_host({"ProgramData": str(system)}) is False
    assert wr._managed_route_host({"ProgramData": str(system), "UV_NO_SYSTEM_CONFIG": "1"}) is True


@pytest.mark.parametrize("project_file", ["uv.toml", "pyproject.toml"])
def test_an_explicit_uv_project_config_closes_the_route(monkeypatch, tmp_path, project_file):
    monkeypatch.setattr(wr._platform, "machine", lambda: "AMD64")
    project = tmp_path / "proj"
    (project / "sub").mkdir(parents = True)
    body = "index-url = 'https://mirror.example/simple'\n"
    if project_file == "pyproject.toml":
        body = "[tool.uv]\n" + body
    (project / project_file).write_text(body, encoding = "utf-8")
    # Found walking up from UV_PROJECT, as install.ps1's discovery does.
    assert wr._managed_route_host({"UV_PROJECT": str(project / "sub")}) is False
    # UV_NO_CONFIG skips it, as it skips every discovered file.
    assert wr._managed_route_host({"UV_PROJECT": str(project / "sub"), "UV_NO_CONFIG": "1"}) is True


@pytest.mark.parametrize("name", ["UV_WORKING_DIR", "UV_WORKING_DIRECTORY"])
def test_a_working_directory_override_starts_discovery(monkeypatch, tmp_path, name):
    # install.ps1's Test-MirrorConfigured walks up from the working-directory override even without
    # UV_PROJECT, and resolves UV_PROJECT and UV_CONFIG_FILE against it.
    monkeypatch.setattr(wr._platform, "machine", lambda: "AMD64")
    work = tmp_path / "work"
    (work / "proj").mkdir(parents = True)
    (work / "uv.toml").write_text("index-url = 'https://mirror.example/simple'\n", encoding = "utf-8")
    assert wr._managed_route_host({name: str(work)}) is False
    assert wr._managed_route_host({name: str(work), "UV_PROJECT": "proj"}) is False
    assert wr._managed_route_host({name: str(work), "UV_NO_CONFIG": "1"}) is True
    (work / "clean.toml").write_text("cache-dir = 'D:/uv'\n", encoding = "utf-8")
    assert wr._managed_route_host({name: str(work), "UV_CONFIG_FILE": "clean.toml"}) is True
    (work / "mirror.toml").write_text("[[index]]\nurl = 'x'\n", encoding = "utf-8")
    assert wr._managed_route_host({name: str(work), "UV_CONFIG_FILE": "mirror.toml"}) is False


def test_discovery_walks_from_the_desktop_installer_directory(monkeypatch, tmp_path):
    # Desktop starts install.ps1 in %USERPROFILE%\\.unsloth, and its discovery walks up from there
    # with no override set.
    monkeypatch.setattr(wr._platform, "machine", lambda: "AMD64")
    profile = {"USERPROFILE": str(tmp_path)}
    (tmp_path / ".unsloth" / "sub").mkdir(parents = True)
    assert wr._managed_route_host(profile) is True
    mirror = "index-url = 'https://mirror.example/simple'\n"
    (tmp_path / ".unsloth" / "uv.toml").write_text(mirror, encoding = "utf-8")
    assert wr._managed_route_host(profile) is False
    assert wr._managed_route_host({**profile, "UV_NO_CONFIG": "1"}) is True
    # A relative working directory resolves against it too.
    assert wr._managed_route_host({**profile, "UV_WORKING_DIR": "sub"}) is False
    (tmp_path / ".unsloth" / "uv.toml").unlink()
    # Found walking up, in the profile itself.
    (tmp_path / "uv.toml").write_text(mirror, encoding = "utf-8")
    assert wr._managed_route_host(profile) is False


def test_the_mirror_gate_matches_install_ps1():
    install = (Path(__file__).resolve().parents[3] / "install.ps1").read_text(encoding = "utf-8")
    body = re.search(r"function Test-PypiUnslothAdmitsTorch \{.*?\n    \}", install, re.S).group(0)
    assert (
        "Test-MirrorConfigured -Tool uv" in body
        and "UV_OFFLINE" in body
        and "UV_EXCLUDE_NEWER" in body
    )
    assert '"$env:UV_DEFAULT_INDEX$env:UV_INDEX_URL$env:UV_INDEX$env:UV_EXTRA_INDEX_URL"' in install


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
    assert rec["managed_torch"] == "2.13"
    # The managed 0.0.35 xFormers is never offered: on sm_120 it has no FlashAttention 2 of its own.
    assert not any("xformers" in key for key in rec)
    assert "0.0.35" not in repr(rec)


@pytest.mark.parametrize("xformers", ["0.0.34", "0.0.33.post2", None])
def test_an_xformers_older_than_the_managed_pin_is_never_a_reason(xformers):
    # RTX 5090 with the managed torch and xFormers 0.0.34 (FA2 through flash_attn): working, and 0.0.35 from the
    # cu130 index would not be an upgrade there, so nothing is recommended.
    assert _rec(torch = "2.13.0+cu130", xformers = xformers) is None


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
