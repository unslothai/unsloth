# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))

import install_python_stack as ips  # noqa: E402
from unsloth_pwsh_runner import run_pwsh  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason = "needs uv")


def _wheel(
    index: Path,
    name: str,
    version: str,
    requires = (),
) -> None:
    dist = f"{name.replace('-', '_')}-{version}.dist-info"
    meta = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    meta += "".join(f"Requires-Dist: {r}\n" for r in requires)
    path = index / f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(dist + "/METADATA", meta)
        z.writestr(
            dist + "/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        z.writestr(dist + "/RECORD", "")


@pytest.fixture
def lagging_index(tmp_path, monkeypatch):
    index = tmp_path / "index"
    index.mkdir()
    _wheel(index, "unsloth", "2026.10.6", ["unsloth-zoo>=2026.10.6"])
    _wheel(index, "unsloth", "2026.10.7", ["unsloth-zoo>=2026.10.7"])
    _wheel(index, "unsloth-zoo", "2026.10.6")
    for key in ("UV_NO_DEPS", "UV_CONSTRAINT", "UV_OVERRIDE", "UV_RESOLUTION", "UV_EXCLUDE_NEWER"):
        monkeypatch.delenv(key, raising = False)
    config = tmp_path / "uv.toml"
    config.write_text(
        "[pip]\nno-index = true\nfind-links = [" + json.dumps(str(index)) + "]\n", encoding = "utf-8"
    )
    monkeypatch.delenv("UV_NO_CONFIG", raising = False)
    monkeypatch.setenv("UV_CONFIG_FILE", str(config))
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_FIND_LINKS", str(index))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("UNSLOTH_DESKTOP_BACKEND_VERSION", "2026.10.6")
    monkeypatch.setattr(ips, "USE_UV", True)

    monkeypatch.setattr(ips.sys, "prefix", str(tmp_path))
    return index


def _no_deps_selection(specs, index) -> dict:
    out = subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            "--no-index",
            "--find-links",
            str(index),
            "--no-deps",
            "--resolution",
            "highest",
            "--no-header",
            "--no-annotate",
            "-",
        ],
        input = "\n".join(specs) + "\n",
        capture_output = True,
        text = True,
        check = True,
    ).stdout
    return dict(re.findall(r"^(unsloth(?:-zoo)?)==(\S+)", out, re.MULTILINE))


def test_no_torch_install_gets_the_resolved_unsloth_not_a_lower_bound(lagging_index, monkeypatch):
    monkeypatch.setattr(ips, "NO_TORCH", True)
    specs = ips._desktop_unsloth_spec("unsloth")
    assert _no_deps_selection(specs, lagging_index) == {
        "unsloth": "2026.10.6",
        "unsloth-zoo": "2026.10.6",
    }


def test_with_deps_install_keeps_the_floor(lagging_index, monkeypatch):
    monkeypatch.setattr(ips, "NO_TORCH", False)
    assert ips._desktop_unsloth_spec("unsloth") == ("unsloth>=2026.10.6", "unsloth-zoo>=2026.10.6")


@pytest.fixture(params = ["python", "sh", "powershell"])
def installer_pair(request, tmp_path, monkeypatch):
    monkeypatch.setattr(ips, "NO_TORCH", True)

    def resolve(*, selected_uv = False):
        if request.param == "python":
            return ips._desktop_unsloth_spec("unsloth")
        floor = os.environ.get("UNSLOTH_DESKTOP_BACKEND_VERSION", "")
        release = f"unsloth>={floor or '2026.10.6'}"
        marker = tmp_path / ips.FEED_CEILING_MARKER
        if request.param == "sh":
            shell = shutil.which("sh")
            if not shell:
                pytest.skip("POSIX shell unavailable")
            source = (STUDIO_DIR.parent / "install.sh").read_text(encoding = "utf-8")
            functions = "\n".join(
                m.group(0)
                for name in ("_rfi_newest", "_rfi_no_torch_pair", "_relax_unsloth_floors_to_index")
                if (m := re.search(rf"(?ms)^{name}\(\) \{{.*?^\}}", source))
            )
            values = {
                "VENV_DIR": tmp_path.as_posix(),
                "_VENV_PY": Path(sys.executable).as_posix(),
                "SKIP_TORCH": "true",
                "PACKAGE_NAME": "unsloth",
                "C_WARN": "",
                "_unsloth_release_install_spec": release,
                "_unsloth_desktop_install_spec": f"unsloth>={floor}" if floor else "",
                "_unsloth_zoo_install_spec": "unsloth-zoo>=2026.10.6",
                "_UNSLOTH_FEED_CEILING_MARKER": marker.as_posix(),
            }
            script = tmp_path / "pair.sh"
            script.write_text(
                "set -e\n"
                + functions
                + "\n"
                + "\n".join(f"{k}={shlex.quote(v)}" for k, v in values.items())
                + '\nsubstep() { printf "%s\\n" "$1" >&2; }\n_relax_unsloth_floors_to_index\n'
                + 'printf "%s\\n" "$_unsloth_release_install_spec" "$_unsloth_zoo_install_spec"\n',
                encoding = "utf-8",
            )
            argv = [shell, str(script)]
        else:
            shell = shutil.which("powershell") or shutil.which("pwsh")
            if not shell:
                pytest.skip("PowerShell unavailable")
            source = (STUDIO_DIR.parent / "install.ps1").read_text(encoding = "utf-8")
            functions = "\n".join(
                m.group(0)
                for name in ("Resolve-NoTorchPair", "Resolve-UnslothFloorsForIndex")
                if (m := re.search(rf"(?ms)^    function {name} \{{.*?^    \}}", source))
            )
            quote = lambda value: "'" + str(value).replace("'", "''") + "'"
            script = tmp_path / "pair.ps1"
            script.write_text(
                '$ErrorActionPreference = "Stop"\n'
                + functions
                + f"\n$VenvDir = {quote(tmp_path)}\n$VenvPython = {quote(sys.executable)}\n$script:UvExe = {quote(shutil.which('uv') if selected_uv else 'uv')}\n"
                + "function substep { param($Message, $Color); [Console]::Error.WriteLine($Message) }\n"
                + f"$r = Resolve-UnslothFloorsForIndex -ReleaseSpec {quote(release)} -ZooSpec 'unsloth-zoo>=2026.10.6' -DesktopVersion {quote(floor)} -Marker {quote(marker)}"
                + (" -NoTorch $true" if "[bool]$NoTorch" in functions else "")
                + "\nWrite-Output $r.Release\nWrite-Output $r.Zoo\n",
                encoding = "utf-8-sig",
            )
            argv = [shell, "-NoProfile", "-File", str(script)]
        env = dict(os.environ, PATH = str(tmp_path / "empty-path")) if selected_uv else None
        run = run_pwsh if request.param == "powershell" else subprocess.run
        result = run(argv, capture_output = True, text = True, timeout = 180, env = env)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return tuple(result.stdout.strip().splitlines()[-2:])

    return resolve


@pytest.mark.parametrize("desktop", [True, False])
def test_every_no_torch_entrypoint_installs_a_compatible_pair(
    lagging_index, installer_pair, monkeypatch, desktop
):
    if not desktop:
        monkeypatch.delenv("UNSLOTH_DESKTOP_BACKEND_VERSION")
    assert _no_deps_selection(installer_pair(), lagging_index) == {
        "unsloth": "2026.10.6",
        "unsloth-zoo": "2026.10.6",
    }


def test_no_torch_does_not_backtrack_for_missing_torch(lagging_index, installer_pair):
    _wheel(lagging_index, "unsloth", "2026.10.7", ["unsloth-zoo>=2026.10.6", "torch>=999"])
    assert _no_deps_selection(installer_pair(), lagging_index) == {
        "unsloth": "2026.10.7",
        "unsloth-zoo": "2026.10.6",
    }


def test_feed_ceiling_is_cleared_when_the_pair_becomes_available(
    lagging_index, installer_pair, monkeypatch, tmp_path
):
    monkeypatch.setenv("UNSLOTH_DESKTOP_BACKEND_VERSION", "2026.10.7")
    assert _no_deps_selection(installer_pair(), lagging_index)["unsloth"] == "2026.10.6"
    marker = tmp_path / ips.FEED_CEILING_MARKER
    assert json.loads(marker.read_text(encoding = "utf-8")) == {
        "requested": "2026.10.7",
        "available": "2026.10.6",
    }
    _wheel(lagging_index, "unsloth-zoo", "2026.10.7")
    assert _no_deps_selection(installer_pair(), lagging_index) == {
        "unsloth": "2026.10.7",
        "unsloth-zoo": "2026.10.7",
    }
    assert not marker.exists()


def test_an_index_with_no_compatible_pair_is_refused(lagging_index, installer_pair):
    _wheel(lagging_index, "unsloth", "2026.10.6", ["unsloth-zoo>=9999"])
    _wheel(lagging_index, "unsloth", "2026.10.7", ["unsloth-zoo>=9999"])
    with pytest.raises(RuntimeError):
        installer_pair()


def test_pip_fallback_resolves_the_same_pair(lagging_index, monkeypatch):
    monkeypatch.setattr(ips, "NO_TORCH", True)
    monkeypatch.setattr(ips, "USE_UV", False)
    monkeypatch.setenv("PIP_CONFIG_FILE", os.devnull)
    assert _no_deps_selection(ips._desktop_unsloth_spec("unsloth"), lagging_index) == {
        "unsloth": "2026.10.6",
        "unsloth-zoo": "2026.10.6",
    }


@pytest.mark.parametrize("policy", ["env", "config"])
def test_pip_metadata_uses_the_install_hash_policy(lagging_index, monkeypatch, tmp_path, policy):
    monkeypatch.setattr(ips, "NO_TORCH", True)
    monkeypatch.setattr(ips, "USE_UV", False)
    if policy == "env":
        monkeypatch.setenv("PIP_REQUIRE_HASHES", "1")
    else:
        config = tmp_path / "pip.conf"
        config.write_text("[install]\nrequire-hashes = true\n", encoding = "utf-8")
        monkeypatch.delenv("PIP_REQUIRE_HASHES", raising = False)
        monkeypatch.setenv("PIP_CONFIG_FILE", str(config))
    assert ips._desktop_unsloth_spec("unsloth") == ("unsloth==2026.10.6", "unsloth-zoo==2026.10.6")


def test_powershell_metadata_uses_the_selected_uv(lagging_index, installer_pair, request):
    if request.node.callspec.params["installer_pair"] != "powershell":
        pytest.skip("PowerShell executable selection")
    assert installer_pair(selected_uv = True) == ("unsloth==2026.10.6", "unsloth-zoo==2026.10.6")


def test_bootstrap_pair_resolvers_match_the_python_installer():
    source = (STUDIO_DIR / "install_python_stack.py").read_text(encoding = "utf-8")
    reference = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_no_torch_pair"
    )
    shell = (STUDIO_DIR.parent / "install.sh").read_text(encoding = "utf-8")
    powershell = (STUDIO_DIR.parent / "install.ps1").read_text(encoding = "utf-8")
    snippets = [
        shell.split("<<'UNSLOTH_PAIR_PY'\n", 1)[1].split("\nUNSLOTH_PAIR_PY", 1)[0],
        powershell.split("$pairCode = @'\n", 1)[1].split("\n'@", 1)[0],
    ]
    for snippet in snippets:
        actual = next(
            n
            for n in ast.parse(snippet).body
            if isinstance(n, ast.FunctionDef) and n.name == "_no_torch_pair"
        )
        assert ast.dump(actual) == ast.dump(reference)
