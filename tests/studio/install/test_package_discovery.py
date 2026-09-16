# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import fnmatch
import os
import re
import subprocess
import zipfile
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Directories that never hold packaged sources but are expensive to walk.
_PRUNED = {".git", "__pycache__", "node_modules", "dist", "build", "venv", ".venv"}


def _finder_patterns(field):
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8")
    finder = text.split("[tool.setuptools.packages.find]", 1)[1].split("\n[", 1)[0]
    match = re.search(rf"^{field}\s*=\s*\[(.*?)\]", finder, re.MULTILINE | re.DOTALL)
    assert match, f"no packages.find {field} list in pyproject.toml"
    return re.findall(r'["\']([^"\']+)["\']', match.group(1))


def _exclude_package_data():
    """The [tool.setuptools.exclude-package-data] table, as {package: patterns}."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8")
    section = text.split("[tool.setuptools.exclude-package-data]", 1)
    assert len(section) == 2, "no exclude-package-data table in pyproject.toml"
    body = section[1].split("\n[", 1)[0]
    table = {}
    # DOTALL so a value spread over several lines is read whole. Matching only
    # single-line arrays would silently return {} for a multi-line entry, and every
    # caller reads an empty veto list as "nothing is excluded", so the simulation
    # would disagree with the wheel setuptools actually builds.
    for entry in re.finditer(
        r'^\s*"?([\w.*]+)"?\s*=\s*\[(.*?)\]',
        body,
        re.MULTILINE | re.DOTALL,
    ):
        # "*" is setuptools' pyproject spelling of the all-packages key.
        key = "" if entry.group(1) == "*" else entry.group(1)
        table[key] = re.findall(r'["\']([^"\']+)["\']', entry.group(2))
    return table


def _discovered_packages():
    """Packages under studio/, resolved the way setuptools' PackageFinder does."""
    include = _finder_patterns("include")
    exclude = _finder_patterns("exclude")
    packages = set()
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT / "studio"):
        dirnames[:] = [d for d in dirnames if d not in _PRUNED]
        if "__init__.py" not in filenames:
            continue
        name = str(Path(dirpath).relative_to(REPO_ROOT)).replace(os.sep, ".")
        if not any(fnmatchcase(name, pat) for pat in include):
            continue
        if any(fnmatchcase(name, pat) for pat in exclude):
            continue
        packages.add(name)
    assert "studio.backend" in packages, "studio.backend should always be packaged"
    return packages


def _tracked_files():
    try:
        out = subprocess.run(
            ["git", "ls-files", "studio"],
            cwd = REPO_ROOT,
            capture_output = True,
            text = True,
            check = True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - no git checkout
        pytest.skip("not a git checkout, cannot resolve the setuptools-scm file list")
    return [line for line in out.splitlines() if line]


def _wheel_payload():
    """Files that build_py would stage, i.e. what the wheel actually ships.

    include-package-data hands every tracked file to the nearest ANCESTOR package
    that survived discovery, so a directory dropped from packages.find comes back
    as data of its parent. exclude-package-data is the veto that stops it.
    """
    package_dirs = {name.replace(".", "/"): name for name in _discovered_packages()}
    excluded = _exclude_package_data()
    shipped = []
    for path in _tracked_files():
        parent = os.path.dirname(path)
        while parent and parent not in package_dirs:
            parent = os.path.dirname(parent)
        if parent not in package_dirs:
            continue
        patterns = excluded.get("", []) + excluded.get(package_dirs[parent], [])
        if any(fnmatch.filter([path], f"{parent}/{pat}") for pat in patterns):
            continue
        shipped.append(path)
    return shipped


def test_generated_compiled_caches_are_excluded():
    patterns = _finder_patterns("exclude")

    for package in ("unsloth_compiled_cache", "studio.backend.unsloth_compiled_cache"):
        assert any(fnmatchcase(package, pattern) for pattern in patterns)


def test_backend_test_suites_stay_out_of_the_wheel():
    # Dropping them from packages.find is not enough on its own: with include-package-data they return as package data
    # of studio.backend.
    leaked = [
        path
        for path in _wheel_payload()
        if path.startswith("studio/backend/") and "/tests/" in path
    ]
    assert not leaked, f"{len(leaked)} backend test files would ship, e.g. {leaked[:3]}"


def test_frontend_source_tree_stays_out_of_the_wheel():
    # public/ is copied into frontend/dist by Vite, so without the veto every one of
    # its files shipped twice, and src/ is .tsx that is already compiled into
    # dist/assets. Together they were 35MB of an 82MB wheel in 2026.9.2.
    shipped = _wheel_payload()
    leaked = [
        path
        for path in shipped
        if path.startswith(("studio/frontend/public/", "studio/frontend/src/"))
    ]
    assert not leaked, f"{len(leaked)} frontend source files would ship, e.g. {leaked[:3]}"


def test_desktop_crate_sources_stay_out_of_the_wheel():
    # `tauri build` reads these from a git checkout in release-desktop.yml. Only
    # src-tauri/icons is needed from an installed Unsloth, by install.sh.
    leaked = [
        path
        for path in _wheel_payload()
        if path.startswith("studio/src-tauri/") and not path.startswith("studio/src-tauri/icons/")
    ]
    assert not leaked, f"{len(leaked)} desktop crate files would ship, e.g. {leaked[:3]}"


def test_installer_icons_survive_the_src_tauri_exclusion():
    # install.sh copies both out of site-packages/studio/src-tauri/icons to build the
    # macOS .icns and the Linux .desktop launcher, so the exclusion above must not
    # take the whole directory with it.
    shipped = set(_wheel_payload())
    for path in (
        "studio/src-tauri/icons/icon.icns",
        "studio/src-tauri/icons/icon.png",
    ):
        assert path in shipped, f"{path} must stay in the wheel for install.sh"


def test_built_wheel_has_no_frontend_source():
    """Artifact-level check, run when a wheel has already been built."""
    wheels = sorted((REPO_ROOT / "dist").glob("unsloth-*.whl"))
    if not wheels:
        pytest.skip("no built wheel in dist/, run `python -m build --wheel` first")
    names = zipfile.ZipFile(wheels[-1]).namelist()
    leaked = [n for n in names if n.startswith(("studio/frontend/public/", "studio/frontend/src/"))]
    assert not leaked, f"{wheels[-1].name} ships {len(leaked)} frontend source files"
    # The served frontend must still be there. Guarding the exclusion alone would
    # pass just as happily on a wheel with no UI in it at all.
    assert any(
        n.startswith("studio/frontend/dist/") for n in names
    ), f"{wheels[-1].name} ships no frontend/dist; the UI would 404"


def test_manifest_prunes_match_exclude_package_data():
    """The two vetoes have to name the same paths, for different artifacts.

    exclude-package-data is what keeps these out of the WHEEL, and it keeps
    working in the sdist -> wheel rebuild `python -m build` performs: restoring
    the excluded files into an extracted sdist and listing them in its MANIFEST
    still produces a wheel without them, which is setuptools' documented
    exclusion precedence. These prunes are what keep them out of the SDIST
    itself, so neither artifact carries a tree nothing installs.
    """
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding = "utf-8")
    for path in (
        "studio/frontend/public",
        "studio/frontend/src",
        "studio/src-tauri/src",
    ):
        assert f"prune {path}\n" in manifest, f"MANIFEST.in must prune {path}"
    assert (
        "prune studio/src-tauri/icons" not in manifest
    ), "src-tauri/icons is read from site-packages by install.sh; do not prune it"


def test_backend_runtime_still_ships():
    shipped = set(_wheel_payload())
    for path in ("studio/backend/main.py", "studio/backend/hub/__init__.py"):
        assert path in shipped, f"{path} must stay in the wheel"


def test_built_wheel_has_no_backend_tests():
    """Artifact-level check, run when a wheel has already been built."""
    wheels = sorted((REPO_ROOT / "dist").glob("unsloth-*.whl"))
    if not wheels:
        pytest.skip("no built wheel in dist/, run `python -m build --wheel` first")
    names = zipfile.ZipFile(wheels[-1]).namelist()
    leaked = [n for n in names if n.startswith("studio/backend/") and "/tests/" in n]
    assert not leaked, f"{wheels[-1].name} ships {len(leaked)} backend test files"
