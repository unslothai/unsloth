# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Coverage for studio/install_manifest.py.

The manifest separates "the install finished" from "the installer was killed
part-way and the venv only looks fine". The CLI, setup.sh's fast path and the
Tauri preflight all read it, so a wrong answer either crashes the backend on
launch or forces needless reinstalls for everyone.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "studio" / "install_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("studio_install_manifest_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


im = _load_module()


@pytest.fixture
def req_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """A requirements tree whose studio.txt names one installed and one absent dist."""
    root = tmp_path / "requirements"
    root.mkdir()
    (root / "studio.txt").write_text(
        "# comment line\n\npytest\nunsloth-definitely-not-a-real-package\n",
        encoding = "utf-8",
    )
    return root


@pytest.fixture
def install_root(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "venv"
    root.mkdir()
    return root


def test_parse_requirement_line_handles_the_shapes_studio_txt_uses():
    assert im._parse_requirement_line("structlog>=24.1.0") == ("structlog", "", ">=24.1.0")
    assert im._parse_requirement_line("matplotlib==3.10.9") == ("matplotlib", "", "==3.10.9")
    assert im._parse_requirement_line("boto3>=1.34.0  # optional: S3") == (
        "boto3",
        "",
        ">=1.34.0",
    )
    assert im._parse_requirement_line("uvicorn[standard]") == ("uvicorn", "", "")
    assert im._parse_requirement_line("# just a comment") is None
    assert im._parse_requirement_line("") is None
    assert im._parse_requirement_line("--index-url https://example.invalid") is None
    name, marker, specifier = im._parse_requirement_line("pywin32 ; sys_platform == 'win32'")
    assert name == "pywin32"
    assert "sys_platform" in marker
    assert specifier == ""


def test_missing_requirements_rejects_an_incompatible_installed_version(tmp_path):
    req = tmp_path / "studio.txt"
    req.write_text(
        "matplotlib==3.10.9\nstructlog>=24.1.0\n",
        encoding = "utf-8",
    )
    installed = {
        "matplotlib": "3.9.0",
        "structlog": "24.1.0",
    }
    assert im.missing_requirements(req, installed = installed) == ["matplotlib"]


def test_platform_gated_lines_are_skipped_when_the_marker_does_not_apply(tmp_path):
    req = tmp_path / "studio.txt"
    req.write_text(
        "unsloth-not-real-a ; sys_platform == 'definitely-not-this-platform'\n"
        "unsloth-not-real-b\n",
        encoding = "utf-8",
    )
    missing = im.missing_requirements(req)
    assert missing == ["unsloth-not-real-b"], (
        "a requirement gated to another OS must not be reported missing, or every "
        "install would look broken on the platforms that legitimately skip it"
    )


def test_missing_requirements_matches_on_distribution_not_import_name(tmp_path):
    # studio.txt lists PyJWT / python-docx / pymupdf, whose import names are
    # jwt / docx / fitz, so matching on imports would look missing.
    req = tmp_path / "studio.txt"
    req.write_text("pytest\n", encoding = "utf-8")
    assert im.missing_requirements(req) == []


def test_complete_install_verifies_ok(install_root, req_root):
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    state = im.verify_install(root = install_root, req_root = req_root, package_name = "pytest")
    assert state["manifest_ok"] is True
    assert state["deps_ok"] is False  # the fake dist is intentionally absent
    assert state["reason"] == "studio_deps_missing"
    assert "unsloth-definitely-not-a-real-package" in state["missing"]


def test_missing_manifest_reports_incomplete(install_root, req_root):
    state = im.verify_install(root = install_root, req_root = req_root, package_name = "pytest")
    assert state["ok"] is False
    assert state["manifest_ok"] is False
    assert state["reason"] == "studio_install_incomplete"


def test_interrupted_install_leaves_no_manifest(install_root, req_root):
    # remove_manifest() runs before the dependency pass, so a later kill cannot
    # leave a stale-but-valid manifest behind.
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    assert im.manifest_path(install_root).is_file()
    assert im.remove_manifest(install_root) is True
    assert not im.manifest_path(install_root).is_file()
    state = im.verify_install(root = install_root, req_root = req_root, package_name = "pytest")
    assert state["reason"] == "studio_install_incomplete"


def test_remove_manifest_reports_whether_the_marker_is_really_gone(
    install_root, req_root, monkeypatch
):
    # Nothing to remove is success: a first install has no manifest yet.
    assert im.remove_manifest(install_root) is True

    # A surviving marker must be reported, not swallowed: the dependency pass
    # would then run behind a manifest that still verifies, so a part-way kill
    # looks complete.
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    path = im.manifest_path(install_root)

    def _refuse(*_args, **_kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(pathlib.Path, "unlink", _refuse)
    assert im.remove_manifest(install_root) is False
    monkeypatch.undo()

    # The stale marker still verifies, which is why the installer has to stop.
    assert path.is_file()
    state = im.verify_install(root = install_root, req_root = req_root, package_name = "pytest")
    assert state["manifest_ok"] is True


def test_schema_bump_invalidates_an_old_manifest(install_root, req_root):
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    path = im.manifest_path(install_root)
    data = json.loads(path.read_text(encoding = "utf-8"))
    data["schema"] = im.MANIFEST_SCHEMA + 1
    path.write_text(json.dumps(data), encoding = "utf-8")
    state = im.verify_install(root = install_root, req_root = req_root, package_name = "pytest")
    assert state["reason"] == "studio_install_manifest_schema"


def test_package_upgrade_invalidates_the_manifest(install_root, req_root):
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    path = im.manifest_path(install_root)
    data = json.loads(path.read_text(encoding = "utf-8"))
    data["package_version"] = "0.0.0-not-the-installed-version"
    path.write_text(json.dumps(data), encoding = "utf-8")
    state = im.verify_install(root = install_root, req_root = req_root, package_name = "pytest")
    assert state["reason"] == "studio_install_version_changed"


def test_verify_follows_the_package_the_manifest_names(install_root, req_root):
    # `studio update --package X` records X. Checking unsloth's version instead
    # would report a change on every probe and repair for ever.
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    state = im.verify_install(
        root = install_root,
        req_root = req_root,
        package_name = "unsloth-definitely-not-a-real-package",
    )
    assert state["manifest_ok"] is True


def test_edited_requirements_invalidate_the_manifest(install_root, req_root):
    # The --local dev path: an edited studio.txt must re-run the dependency
    # pass, not sit behind setup.sh's "up to date" fast path.
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    (req_root / "studio.txt").write_text("pytest\nrich\n", encoding = "utf-8")
    state = im.verify_install(root = install_root, req_root = req_root, package_name = "pytest")
    assert state["reason"] == "studio_install_requirements_changed"


def test_unwritable_root_degrades_to_incomplete(tmp_path, req_root):
    missing_root = tmp_path / "does" / "not" / "exist"
    assert im.write_manifest(root = missing_root, req_root = req_root) is None
    state = im.verify_install(root = missing_root, req_root = req_root, package_name = "pytest")
    assert state["ok"] is False
