# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The no-torch install is --no-deps, so it must get the unsloth the pair probe resolved.

A lower bound lets --no-deps pick a newer unsloth whose unsloth-zoo floor the index cannot meet.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
sys.path.insert(0, str(STUDIO_DIR))

import install_python_stack as ips  # noqa: E402

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
    # unsloth .7 is published but its zoo is not yet: only .6 + .6 is a valid pair.
    index = tmp_path / "index"
    index.mkdir()
    _wheel(index, "unsloth", "2026.10.6", ["unsloth-zoo>=2026.10.6"])
    _wheel(index, "unsloth", "2026.10.7", ["unsloth-zoo>=2026.10.7"])
    _wheel(index, "unsloth-zoo", "2026.10.6")
    for key in ("UV_NO_DEPS", "UV_CONSTRAINT", "UV_OVERRIDE", "UV_RESOLUTION", "UV_EXCLUDE_NEWER"):
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setenv("UV_NO_CONFIG", "1")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("UNSLOTH_DESKTOP_BACKEND_VERSION", "2026.10.6")
    monkeypatch.setattr(ips, "USE_UV", True)

    def offline_run(argv, **kwargs):
        # The real resolver, offline: only this index is visible.
        argv = [*argv[:3], "--no-index", "--find-links", str(index), *argv[3:]]
        return subprocess.run(argv, **kwargs)

    offline = types.SimpleNamespace(
        run = offline_run, DEVNULL = subprocess.DEVNULL, SubprocessError = subprocess.SubprocessError
    )
    monkeypatch.setattr(ips, "subprocess", offline)
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
    # The with-deps path lets the installer read unsloth's zoo floor, so a floor is enough.
    assert ips._desktop_unsloth_spec("unsloth") == ("unsloth>=2026.10.6", "unsloth-zoo>=2026.10.6")
