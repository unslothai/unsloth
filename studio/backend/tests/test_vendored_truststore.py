# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The vendored truststore stays byte-identical to the release it came from.

Drift here is silent and dangerous in both directions: a formatter reflowing the
upstream files makes the next sync a merge conflict, and an edit to them means
Studio verifies certificates with code no upstream release ever shipped.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_VENDOR = _BACKEND / "vendor"
_MANIFEST = json.loads((_VENDOR / "truststore_manifest.json").read_text(encoding = "utf-8"))

# Everything the vendor directory is allowed to hold, beyond the package itself.
_SIDECARS = {"LICENSE", "README.md", "truststore_manifest.json"}
# Ours, not upstream's: prose we may reword, and the manifest cannot hash itself.
_UNPINNED = {"README.md", "truststore_manifest.json"}


def _tracked_files() -> dict[str, Path]:
    """The real tree, enumerated, so an *added* file is caught and not just an edit."""
    return {
        path.relative_to(_VENDOR).as_posix(): path
        for path in sorted(_VENDOR.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.name not in _UNPINNED
    }


def test_vendored_tree_matches_the_manifest():
    found = _tracked_files()
    recorded = _MANIFEST["files"]
    assert set(found) == set(recorded), (
        "the vendored tree gained or lost a file; re-run "
        "scripts/sync_vendored_truststore.py rather than editing it by hand"
    )
    drifted = [
        name
        for name, path in found.items()
        if hashlib.sha256(path.read_bytes()).hexdigest() != recorded[name]
    ]
    assert not drifted, (
        f"vendored files no longer match upstream {_MANIFEST['version']}: {', '.join(drifted)}. "
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


def test_vendor_holds_nothing_but_truststore():
    """The gate appends this directory to sys.path, so anything else here is importable."""
    top_level = {path.name for path in _VENDOR.iterdir()} - _SIDECARS
    assert top_level == {"truststore"}, (
        f"unexpected entries in the vendor directory: {sorted(top_level - {'truststore'})}. "
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
        "import the vendored package as top-level `truststore` after appending the vendor "
        f"directory to sys.path, never by its dotted path: {offenders}"
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


@pytest.mark.parametrize("relative", ["LICENSE", "README.md"])
def test_provenance_files_are_present(relative):
    """MIT requires the licence to travel with the copy."""
    assert (_VENDOR / relative).read_text(encoding = "utf-8").strip()
