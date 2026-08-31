# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A healthy install must never be condemned by the payload probe.

`__DAMAGED__` and `__MISSING__` reach `setup_fail 1` on the post-update path, so a
false verdict here does not degrade an update, it fails one that worked. Every case
below is a tree that a correct probe has to accept, driven through the REAL probe
extracted from the installer rather than through assertions about its source.

The bash copy is expanded through bash rather than read off disk: it lives in a
double-quoted shell string, so the file text is not what the interpreter receives.
"""

import csv
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "expands the bash installer string")

REPO = Path(__file__).resolve().parents[2]
SETUP_SH = REPO / "studio" / "setup.sh"
SETUP_PS1 = REPO / "studio" / "setup.ps1"

SH_ASSIGNMENT = re.compile(r'^(_PKG_PROBE_PY="\n.*?\n")$', re.M | re.S)

PKG = "demo"
VER = "1.0"

# Runs the probe with a throwaway directory standing in for the venv's own
# purelib/platlib, which is the only place the probe is allowed to look.
RUNNER = """
import sys, sysconfig
_sp = sys.argv[2]
_orig = sysconfig.get_path
sysconfig.get_path = lambda name, *a, **k: _sp if name in ('purelib', 'platlib') else _orig(name, *a, **k)
sys.path.insert(0, _sp)
sys.argv = [sys.argv[0], sys.argv[1]]
exec(compile(open(sys.argv[0] + '.probe').read(), 'probe', 'exec'))
"""


@pytest.fixture(scope = "module")
def probe_source() -> str:
    m = SH_ASSIGNMENT.search(SETUP_SH.read_text(encoding = "utf-8"))
    assert m, "could not find _PKG_PROBE_PY in setup.sh"
    return subprocess.run(
        ["bash"],
        input = (m.group(1) + '\nprintf "%s" "$_PKG_PROBE_PY"\n').encode(),
        stdout = subprocess.PIPE,
        check = True,
    ).stdout.decode()


def _record(dist_info: Path, rows) -> None:
    buf = io.StringIO(newline = "")
    writer = csv.writer(buf, lineterminator = "\n")
    for row in rows:
        writer.writerow(row)
    (dist_info / "RECORD").write_text(buf.getvalue(), encoding = "utf-8", newline = "")


def _dist(site_packages: Path, name = PKG, version = VER) -> Path:
    dist_info = site_packages / f"{name}-{version}.dist-info"
    dist_info.mkdir(parents = True, exist_ok = True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding = "utf-8")
    (dist_info / "WHEEL").write_text("Wheel-Version: 1.0\n", encoding = "utf-8")
    return dist_info


def _write(path: Path, text: str) -> int:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")
    return len(text.encode())


def _verdict(probe_source: str, tmp_path: Path, site_packages: Path, name = PKG) -> str:
    runner = tmp_path / "run.py"
    runner.write_text(RUNNER, encoding = "utf-8")
    (tmp_path / "run.py.probe").write_text(probe_source, encoding = "utf-8")
    done = subprocess.run(
        [sys.executable, "-I", str(runner), name, str(site_packages)],
        stdout = subprocess.PIPE, stderr = subprocess.PIPE, timeout = 120,
    )
    assert done.returncode == 0, done.stderr.decode(errors = "replace")
    sentinels = [
        line[len("POSTVER="):]
        for line in done.stdout.decode(errors = "replace").splitlines()
        if line.startswith("POSTVER=")
    ]
    assert sentinels, f"probe printed no sentinel: {done.stdout!r}"
    return sentinels[-1]


def test_a_regenerated_frontend_dist_is_not_damage(probe_source, tmp_path):
    """setup.sh runs `vite build` inside the installed tree.

    vite empties dist/ and rewrites every asset under a fresh content hash, so the
    RECORD rows for the shipped bundle name files that no longer exist. That is the
    installer's own doing, not damage, and it must not fail the update -- nor can a
    size waiver reach it, because the recorded files are gone rather than shorter.
    """
    site_packages = tmp_path / "site-packages"
    dist_info = _dist(site_packages)
    payload = _write(site_packages / PKG / "__init__.py", "x = 1\n")
    old = _write(site_packages / "studio/frontend/dist/assets/index-OLDHASH1.js", "console.log(1)\n")
    index = _write(site_packages / "studio/frontend/dist/index.html", "<html>old</html>\n")
    _record(dist_info, [
        [f"{PKG}/__init__.py", "sha256=x", payload],
        ["studio/frontend/dist/assets/index-OLDHASH1.js", "sha256=x", old],
        ["studio/frontend/dist/index.html", "sha256=x", index],
        [f"{dist_info.name}/RECORD", "", ""],
    ])

    # What a rebuild leaves behind: new hashes, old names gone, index.html shorter.
    (site_packages / "studio/frontend/dist/assets/index-OLDHASH1.js").unlink()
    _write(site_packages / "studio/frontend/dist/assets/index-NEWHASH2.js", "console.log(2)\n")
    _write(site_packages / "studio/frontend/dist/index.html", "<html>\n")

    assert _verdict(probe_source, tmp_path, site_packages) == VER


def test_real_damage_is_still_caught_after_a_frontend_rebuild(probe_source, tmp_path):
    """The carve-out is a subtree, not an amnesty: the rest of the tree still counts."""
    site_packages = tmp_path / "site-packages"
    dist_info = _dist(site_packages)
    payload = _write(site_packages / PKG / "__init__.py", "x = 1\n")
    old = _write(site_packages / "studio/frontend/dist/assets/index-OLDHASH1.js", "console.log(1)\n")
    _record(dist_info, [
        [f"{PKG}/__init__.py", "sha256=x", payload],
        ["studio/frontend/dist/assets/index-OLDHASH1.js", "sha256=x", old],
        [f"{dist_info.name}/RECORD", "", ""],
    ])
    (site_packages / "studio/frontend/dist/assets/index-OLDHASH1.js").unlink()
    (site_packages / PKG / "__init__.py").unlink()

    assert _verdict(probe_source, tmp_path, site_packages) == "__DAMAGED__"


def test_a_pth_only_dispatch_wheel_is_not_damage(probe_source, tmp_path):
    """A wheel whose whole payload is one sys.path-extending .pth is a real shape.

    nvidia-cutlass-dsl ships exactly that. Only an editable, whose shims ARE its
    payload and whose checkout cannot be validated from here, is damage.
    """
    site_packages = tmp_path / "site-packages"
    dist_info = _dist(site_packages)
    _write(site_packages / PKG / "__init__.py", "x = 1\n")
    pth = _write(site_packages / f"{PKG}_packages.pth", f"import sys, os, {PKG}\n")
    (dist_info / "top_level.txt").write_text("\n", encoding = "utf-8")
    _record(dist_info, [
        [f"{PKG}_packages.pth", "sha256=x", pth],
        [f"{dist_info.name}/RECORD", "", ""],
    ])

    assert _verdict(probe_source, tmp_path, site_packages) == VER


def test_an_editable_whose_checkout_is_gone_is_still_damage(probe_source, tmp_path):
    """The case the shim-only rule exists for has to survive the fix above."""
    site_packages = tmp_path / "site-packages"
    dist_info = _dist(site_packages)
    checkout = tmp_path / "deleted-checkout"
    (dist_info / "direct_url.json").write_text(
        json.dumps({"url": checkout.as_uri(), "dir_info": {"editable": True}}), encoding = "utf-8")
    pth = _write(site_packages / f"__editable__.{PKG}.pth", str(checkout) + "\n")
    _record(dist_info, [
        [f"__editable__.{PKG}.pth", "sha256=x", pth],
        [f"{dist_info.name}/RECORD", "", ""],
    ])

    assert _verdict(probe_source, tmp_path, site_packages) == "__DAMAGED__"


@pytest.mark.parametrize("bad", ["we\nird.py", "we\x0bird.py"])
def test_a_quoted_line_break_in_a_recorded_name_is_not_damage(probe_source, tmp_path, bad):
    """RECORD is CSV: a quoted field may hold a newline, and str.splitlines also
    breaks on \\v, \\f and the Unicode separators, which csv does not. Splitting the
    text by lines turns one legal row into two broken ones and condemns the file."""
    site_packages = tmp_path / "site-packages"
    dist_info = _dist(site_packages)
    size = _write(site_packages / PKG / bad, "x = 1\n")
    _record(dist_info, [
        [f"{PKG}/{bad}", "sha256=x", size],
        [f"{dist_info.name}/RECORD", "", ""],
    ])

    assert _verdict(probe_source, tmp_path, site_packages) == VER


def test_both_installers_carry_the_same_carve_outs():
    """The two probe copies are hand-mirrored, so the fixes have to land in both."""
    sh = SETUP_SH.read_text(encoding = "utf-8")
    ps1 = SETUP_PS1.read_text(encoding = "utf-8")
    for needle in (
        "_regen = (('studio', 'frontend', 'dist'),)",
        "if any(f.parts[:len(_p)] == _p for _p in _regen):",
        "for r in csv.reader(io.StringIO(record or '', newline='')):",
        "for r in csv.reader(io.StringIO(d_record or '', newline='')):",
        "if _shim and not _real and (_edmark or _durl):",
    ):
        assert needle in sh, f"setup.sh lost: {needle}"
        assert needle in ps1, f"setup.ps1 lost: {needle}"
    # splitlines() on RECORD is the defect the io.StringIO form replaces.
    assert "csv.reader((record or '').splitlines())" not in sh
    assert "csv.reader((d_record or '').splitlines())" not in sh
