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

"""The four installers pin one uv, gate reuse on one floor, and the WSL guest is not left behind.

uv below 0.13.4 still trusts the compromised emSign CAs (GHSA-hq85-5cpj-mr62), so a script that
drifts back to an older pin or floor reopens that exposure.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLOOR = (0, 13, 4)

SCRIPTS = {
    "install.sh": (r'^UV_PINNED_VERSION="([^"]+)"', r'^UV_MIN_VERSION="([^"]+)"'),
    "studio/setup.sh": (
        r'^_SETUP_UV_PINNED_VERSION="([^"]+)"',
        r'^_SETUP_UV_MIN_VERSION="([^"]+)"',
    ),
    "install.ps1": (r'^\s*\$UvPinnedVersion = "([^"]+)"', r'^\s*\$UvMinVersion = "([^"]+)"'),
    "studio/setup.ps1": (r'^\$UvPinnedVersion = "([^"]+)"', r'^\$SetupUvMinVersion = "([^"]+)"'),
}


def _one(pattern: str, text: str, name: str) -> str:
    found = re.findall(pattern, text, re.M)
    assert len(found) == 1, (name, pattern, found)
    return found[0]


def _version(text: str) -> tuple:
    return tuple(int(part) for part in text.split("."))


def test_installers_agree_on_pin_floor_and_hashes():
    pins, floors, tables = {}, {}, {}
    for name, (pin_re, floor_re) in SCRIPTS.items():
        text = (ROOT / name).read_text(encoding = "utf-8")
        pins[name] = _one(pin_re, text, name)
        floors[name] = _one(floor_re, text, name)
        pin = pins[name]
        # Every uv artifact the script names is the pinned release: archives, wheels, mirror probes.
        for version in re.findall(r"uv/releases/download/(\d+\.\d+\.\d+)/", text):
            assert version == pin, (name, version)
        wheels = re.findall(r"/uv-(\d+\.\d+\.\d+)-py3-none-\S+?\.whl", text)
        assert wheels and set(wheels) == {pin}, (name, wheels)
        if name.endswith(".sh"):
            assets = re.findall(r'echo "(uv-[a-z0-9_-]+\.tar\.gz) ([0-9a-f]+)"', text)
            assert len(assets) == 4, (name, assets)
            wheel_rows = re.findall(r'echo "packages/\S+\.whl ([0-9a-f]+)"', text)
            assert len(wheel_rows) == 4, (name, wheel_rows)
            installer = re.findall(r'INSTALLER_SH_SHA256="([0-9a-f]+)"', text)
            assert len(installer) == 1, (name, installer)
            digests = [d for _, d in assets] + wheel_rows + installer
        else:
            assets = re.findall(r'Asset = "(uv-[a-z0-9_-]+\.zip)";\s+Sha256 = "([0-9A-F]+)"', text)
            assert len(assets) == 3, (name, assets)
            wheel_rows = re.findall(r'WheelSha256 = "([0-9A-F]+)"', text)
            assert len(wheel_rows) == 3, (name, wheel_rows)
            digests = [d for _, d in assets] + wheel_rows
        assert all(len(d) == 64 for d in digests), (name, digests)
        assert len(set(d.lower() for d in digests)) == len(digests), name
        tables[name] = set(d.lower() for d in digests)

    assert len(set(pins.values())) == 1, pins
    assert set(floors.values()) == {".".join(map(str, FLOOR))}, floors
    assert _version(next(iter(pins.values()))) >= FLOOR

    # The two shell scripts and the two PowerShell scripts carry identical tables.
    assert tables["install.sh"] == tables["studio/setup.sh"]
    assert tables["install.ps1"] == tables["studio/setup.ps1"]


def test_wsl_guest_uv_is_at_least_the_floor():
    source = (ROOT / "studio/backend/core/inference/wsl_host.py").read_text(encoding = "utf-8")
    uv = None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "UV" for t in node.targets
        ):
            uv = ast.literal_eval(node.value)
    assert uv is not None
    assert _version(uv["version"]) >= FLOOR, uv
    assert f"/download/{uv['version']}/" in uv["url"], uv
    assert re.fullmatch(r"[0-9a-f]{64}", uv["sha256"]), uv
    # An existing guest is re-checked against the pinned version, not merely for a uv binary.
    assert 'UV["version"]' in source
