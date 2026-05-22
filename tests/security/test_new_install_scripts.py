"""Regression tests for `scripts/check_new_install_scripts.py`.

The fixture lockfiles are tiny dicts written to `tmp_path` so the
tests stay self-contained. The session-wide `network_blocker` fixture
in conftest.py refuses any real-world socket connect; the scanner
treats that block as "registry unreachable, emit finding anyway",
which is the offline-safe path under test.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_new_install_scripts.py"


def _run(base: Path, head: Path, *, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base",
            str(base),
            "--head",
            str(head),
        ],
        capture_output = True,
        text = True,
        timeout = timeout,
    )


def _write(path: Path, content: dict) -> Path:
    path.write_text(json.dumps(content), encoding = "utf-8")
    return path


# ---------------------------------------------------------------------------
# Lockfile fixtures.
# ---------------------------------------------------------------------------


def _v3_lockfile(packages: dict) -> dict:
    return {
        "name": "unsloth-theme",
        "version": "0.0.0",
        "lockfileVersion": 3,
        "requires": True,
        "packages": packages,
    }


def _v2_lockfile(packages: dict, dependencies: dict) -> dict:
    return {
        "name": "unsloth-theme",
        "version": "0.0.0",
        "lockfileVersion": 2,
        "requires": True,
        "packages": packages,
        "dependencies": dependencies,
    }


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_no_new_install_scripts_exit_0(tmp_path: Path):
    """If base == head, nothing new can have been added."""
    same = _v3_lockfile(
        {
            "": {"name": "unsloth-theme", "version": "0.0.0"},
            "node_modules/node-gyp": {
                "version": "10.0.1",
                "resolved": "https://registry.npmjs.org/node-gyp/-/node-gyp-10.0.1.tgz",
                "integrity": "sha512-fake",
                "hasInstallScript": True,
            },
        }
    )
    base = _write(tmp_path / "base.json", same)
    head = _write(tmp_path / "head.json", same)
    result = _run(base, head)
    assert result.returncode == 0, result.stderr
    assert "no newly-added install-script" in result.stdout.lower()


def test_new_dep_with_postinstall_exits_1(tmp_path: Path):
    """A NEW dep in head with `hasInstallScript: true` must exit 1."""
    base_pkgs = {
        "": {"name": "unsloth-theme", "version": "0.0.0"},
        "node_modules/react": {
            "version": "19.2.4",
            "resolved": "https://registry.npmjs.org/react/-/react-19.2.4.tgz",
            "integrity": "sha512-fake",
        },
    }
    head_pkgs = dict(base_pkgs)
    head_pkgs["node_modules/evil-postinstall"] = {
        "version": "1.0.0",
        "resolved": (
            "https://registry.npmjs.org/evil-postinstall/-/evil-postinstall-1.0.0.tgz"
        ),
        "integrity": "sha512-fake",
        "hasInstallScript": True,
    }
    base = _write(tmp_path / "base.json", _v3_lockfile(base_pkgs))
    head = _write(tmp_path / "head.json", _v3_lockfile(head_pkgs))
    result = _run(base, head)
    assert (
        result.returncode == 1
    ), f"expected exit 1, got {result.returncode}; stderr:\n{result.stderr}"
    assert "evil-postinstall" in result.stderr
    assert "1.0.0" in result.stderr


def test_existing_dep_with_postinstall_ignored(tmp_path: Path):
    """An install-script dep present in BOTH base and head is not new."""
    base_pkgs = {
        "": {"name": "unsloth-theme", "version": "0.0.0"},
        "node_modules/node-gyp": {
            "version": "10.0.1",
            "resolved": "https://registry.npmjs.org/node-gyp/-/node-gyp-10.0.1.tgz",
            "integrity": "sha512-fake",
            "hasInstallScript": True,
        },
        # Transitive install-script copy, nested under another dep.
        "node_modules/some-build-pkg/node_modules/node-gyp": {
            "version": "10.0.1",
            "resolved": "https://registry.npmjs.org/node-gyp/-/node-gyp-10.0.1.tgz",
            "integrity": "sha512-fake",
            "hasInstallScript": True,
        },
    }
    head_pkgs = dict(base_pkgs)
    # An ENTIRELY UNRELATED non-install-script dep is added in head.
    head_pkgs["node_modules/lodash"] = {
        "version": "4.17.21",
        "resolved": "https://registry.npmjs.org/lodash/-/lodash-4.17.21.tgz",
        "integrity": "sha512-fake",
    }
    base = _write(tmp_path / "base.json", _v3_lockfile(base_pkgs))
    head = _write(tmp_path / "head.json", _v3_lockfile(head_pkgs))
    result = _run(base, head)
    assert result.returncode == 0, (
        f"expected exit 0, got {result.returncode}; stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )
    # Sanity: the existing node-gyp must NOT be reported.
    assert "node-gyp" not in result.stderr


def test_v2_v3_lockfile_format_support(tmp_path: Path):
    """A lockfileVersion 2 lockfile with the same shape parses the same."""
    base_pkgs = {
        "": {"name": "unsloth-theme", "version": "0.0.0"},
    }
    base_deps = {}  # v2 carries both; empty deps OK
    head_pkgs = {
        "": {"name": "unsloth-theme", "version": "0.0.0"},
        "node_modules/v2-postinstall-dep": {
            "version": "2.0.0",
            "resolved": (
                "https://registry.npmjs.org/v2-postinstall-dep/-/"
                "v2-postinstall-dep-2.0.0.tgz"
            ),
            "integrity": "sha512-fake",
            "hasInstallScript": True,
        },
    }
    head_deps = {
        "v2-postinstall-dep": {
            "version": "2.0.0",
            "resolved": (
                "https://registry.npmjs.org/v2-postinstall-dep/-/"
                "v2-postinstall-dep-2.0.0.tgz"
            ),
            "integrity": "sha512-fake",
        },
    }
    base = _write(tmp_path / "base.json", _v2_lockfile(base_pkgs, base_deps))
    head = _write(tmp_path / "head.json", _v2_lockfile(head_pkgs, head_deps))
    result = _run(base, head)
    assert result.returncode == 1, (
        f"expected exit 1 for v2 lockfile, got {result.returncode}; "
        f"stderr:\n{result.stderr}"
    )
    assert "v2-postinstall-dep" in result.stderr

    # And again: same packages dict but lockfileVersion 3 -- should
    # produce the same finding shape.
    base_v3 = _write(tmp_path / "base_v3.json", _v3_lockfile(base_pkgs))
    head_v3 = _write(tmp_path / "head_v3.json", _v3_lockfile(head_pkgs))
    result_v3 = _run(base_v3, head_v3)
    assert result_v3.returncode == 1, (
        f"expected exit 1 for v3 lockfile, got {result_v3.returncode}; "
        f"stderr:\n{result_v3.stderr}"
    )
    assert "v2-postinstall-dep" in result_v3.stderr
