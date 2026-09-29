# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The vendored truststore and laya stay byte-identical to the releases they came from.

They are static copies: nothing refreshes them, so any change to these bytes is
either a deliberate version bump that must update the manifest with it, or an
accident. For truststore the accident is the dangerous one, since it means
Unsloth verifies certificates with code no upstream release ever shipped.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_VENDOR = _BACKEND / "vendor"
_MANIFESTS = {
    path.name.removesuffix("_manifest.json"): json.loads(path.read_text(encoding = "utf-8"))
    for path in sorted(_VENDOR.glob("*_manifest.json"))
}
_MANIFEST = _MANIFESTS["truststore"]
_PACKAGES = {"truststore", "laya"}

# Everything the vendor directory is allowed to hold, beyond the packages themselves.
_SIDECARS = {
    "LICENSE",
    "LICENSE.laya",
    "README.md",
    *(f"{name}_manifest.json" for name in _PACKAGES),
}
# Ours, not upstream's: prose we may reword, and the manifests cannot hash themselves.
_UNPINNED = {"README.md", *(f"{name}_manifest.json" for name in _PACKAGES)}


def _tracked_files() -> dict[str, Path]:
    """The real tree, enumerated, so an *added* file is caught and not just an edit."""
    return {
        path.relative_to(_VENDOR).as_posix(): path
        for path in sorted(_VENDOR.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.name not in _UNPINNED
    }


def test_every_vendored_package_has_a_manifest():
    assert set(_MANIFESTS) == _PACKAGES
    assert all(manifest["package"] == name for name, manifest in _MANIFESTS.items())


def test_vendored_tree_matches_the_manifests():
    found = _tracked_files()
    recorded = {name: digest for m in _MANIFESTS.values() for name, digest in m["files"].items()}
    assert sum(len(m["files"]) for m in _MANIFESTS.values()) == len(recorded)
    assert set(found) == set(recorded), (
        "the vendored tree gained or lost a file; each package is a static copy of an upstream "
        "release, so update its <package>_manifest.json in the same commit"
    )
    drifted = [
        name
        for name, path in found.items()
        if hashlib.sha256(path.read_bytes()).hexdigest() != recorded[name]
    ]
    assert not drifted, (
        f"vendored files no longer match upstream: {', '.join(drifted)}. "
        "A formatter most likely rewrote them; check the vendor excludes in pyproject.toml "
        "and .pre-commit-config.yaml"
    )


def test_no_symlinks_or_special_files():
    """A symlink would let the digest check pass while the imported bytes differ."""
    offenders = [
        str(path.relative_to(_VENDOR))
        for path in _VENDOR.rglob("*")
        if path.is_symlink() or (path.exists() and not path.is_file() and not path.is_dir())
    ]
    assert not offenders, f"vendor tree must be plain files: {offenders}"


def test_vendor_holds_nothing_but_the_vendored_packages():
    """The gate appends this directory to sys.path, so anything else here is importable."""
    top_level = {path.name for path in _VENDOR.iterdir()} - _SIDECARS
    assert top_level == _PACKAGES, (
        f"unexpected entries in the vendor directory: {sorted(top_level - _PACKAGES)}. "
        "Appending it to sys.path would make them importable as top-level modules"
    )


def test_vendor_is_not_a_package():
    """No __init__.py: a dotted import would load these files under a second name.

    `import truststore` and `import studio.backend.vendor.truststore` are two
    sys.modules entries, each with its own _original_SSLContext, so injecting
    from both wraps ssl twice.
    """
    assert not (_VENDOR / "__init__.py").exists(), (
        "studio/backend/vendor must not be a package; it ships via the "
        "backend/vendor/**/* package-data glob instead"
    )


def test_nothing_imports_the_vendor_path_directly():
    studio = _BACKEND.parent
    offenders = []
    for path in studio.rglob("*.py"):
        if "vendor" in path.parts or "node_modules" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8", errors = "ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and "vendor" in (node.module or ""):
                offenders.append(f"{path.relative_to(studio)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                if any("vendor" in alias.name.split(".") for alias in node.names):
                    offenders.append(f"{path.relative_to(studio)}:{node.lineno}")
    assert not offenders, (
        "import truststore as top-level `truststore` after appending the vendor directory to "
        "sys.path, and laya through core.systemone.laya_runtime._laya(), never by a dotted "
        f"vendor path: {offenders}"
    )


def test_vendored_version_is_the_one_recorded():
    from utils.native_tls import vendor_dir

    spec = Path(vendor_dir()) / "truststore" / "__init__.py"
    version = next(
        line.split("=")[1].strip().strip('"')
        for line in spec.read_text(encoding = "utf-8").splitlines()
        if line.startswith("__version__")
    )
    assert (
        version == _MANIFEST["version"]
    ), f"vendored truststore is {version} but the manifest records {_MANIFEST['version']}"


def test_vendored_laya_version_is_the_one_recorded():
    from core.systemone.laya_runtime import _VENDORED_LAYA

    assert _VENDORED_LAYA == _VENDOR / "laya"
    init = (_VENDORED_LAYA / "__init__.py").read_text(encoding = "utf-8")
    assert f'__version__ = "{_MANIFESTS["laya"]["version"]}"' in init


@pytest.mark.parametrize("relative", ["LICENSE", "LICENSE.laya", "README.md"])
def test_provenance_files_are_present(relative):
    """MIT and Apache-2.0 both require the licence to travel with the copy."""
    assert (_VENDOR / relative).read_text(encoding = "utf-8").strip()
