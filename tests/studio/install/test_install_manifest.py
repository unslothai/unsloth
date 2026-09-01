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
import sysconfig

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


def _write_dist_metadata(site: pathlib.Path, name: str, version: str) -> None:
    stem = name.replace("-", "_")
    info = site / f"{stem}-{version}.dist-info"
    info.mkdir(parents = True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding = "utf-8",
    )


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


def test_installed_versions_reports_every_canonical_metadata_record(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    site.mkdir()
    _write_dist_metadata(site, "demo_pkg", "1.0")
    _write_dist_metadata(site, "demo-pkg", "2.0")
    monkeypatch.setattr(im, "_metadata_scan_paths", lambda: [str(site)])

    assert im.installed_versions("demo.pkg") == ["1.0", "2.0"]
    assert im._installed_version("demo-pkg") is None


def test_installed_versions_ignores_malformed_unrelated_metadata(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    site.mkdir()
    _write_dist_metadata(site, "demo", "1.0")
    _write_dist_metadata(site, "demo", "2.0")
    malformed = site / "unrelated-1.0.dist-info"
    malformed.mkdir()
    (malformed / "METADATA").write_bytes(b"\xff\xfe")
    monkeypatch.setattr(im, "_metadata_scan_paths", lambda: [str(site)])

    assert im.installed_versions("demo") == ["1.0", "2.0"]


def test_installed_versions_marks_malformed_matching_metadata_as_a_conflict(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    site.mkdir()
    _write_dist_metadata(site, "demo", "1.0")
    malformed = site / "demo-2.0.dist-info"
    malformed.mkdir()
    (malformed / "METADATA").write_bytes(b"\xff\xfe")
    monkeypatch.setattr(im, "_metadata_scan_paths", lambda: [str(site)])

    versions = im.installed_versions("demo")
    assert versions == ["", "1.0"]
    assert im.invalid_metadata_paths("demo") == [malformed]
    assert im.metadata_conflict(versions) is True
    assert im._installed_version("demo") is None


def test_a_pip_tilde_backup_is_named_as_a_backup(tmp_path, monkeypatch):
    """pip's AdjacentTempDirectory leftover counts as a duplicate but pip will
    not uninstall it, so the repair has to find it by directory name."""
    site = tmp_path / "site-packages"
    site.mkdir()
    _write_dist_metadata(site, "demo", "2.0")
    backup = site / "~emo-1.0.dist-info"
    backup.mkdir()
    (backup / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n", encoding = "utf-8"
    )
    monkeypatch.setattr(im, "_metadata_scan_paths", lambda: [str(site)])

    assert im.installed_versions("demo") == ["1.0", "2.0"]
    assert im.pip_backup_metadata_paths("demo") == [backup]
    # A healthy record must never be mistaken for one.
    assert im.pip_backup_metadata_paths("demo") != [site / "demo-2.0.dist-info"]


def test_a_sole_pip_backup_is_reported_as_a_conflict(tmp_path, monkeypatch):
    """pip killed after moving the old record aside leaves one readable version,
    so a version count sees nothing wrong while the package is really gone."""
    site = tmp_path / "site-packages"
    site.mkdir()
    backup = site / "~emo-1.0.dist-info"
    backup.mkdir()
    (backup / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n", encoding = "utf-8"
    )
    monkeypatch.setattr(im, "_metadata_scan_paths", lambda: [str(site)])

    assert im.metadata_conflict(im.installed_versions("demo")) is False
    assert im.installed_version_probe("demo") == ("1.0", True)
    assert im.installed_version_probe("other", ("demo",))[1] is True


def test_a_healthy_install_has_no_pip_backups(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    site.mkdir()
    _write_dist_metadata(site, "demo", "2.0")
    monkeypatch.setattr(im, "_metadata_scan_paths", lambda: [str(site)])

    assert im.pip_backup_metadata_paths("demo") == []


def test_single_malformed_matching_metadata_is_a_conflict(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    site.mkdir()
    malformed = site / "demo_pkg-2.0.dist-info"
    malformed.mkdir()
    (malformed / "METADATA").write_bytes(b"\xff\xfe")
    monkeypatch.setattr(im, "_metadata_scan_paths", lambda: [str(site)])

    versions = im.installed_versions("demo-pkg")
    assert versions == [""]
    assert im.invalid_metadata_paths("demo-pkg") == [malformed]
    assert im.metadata_conflict(versions) is True


def test_nameless_matching_metadata_is_a_conflict(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    site.mkdir()
    _write_dist_metadata(site, "demo", "1.0")
    nameless = site / "demo-2.0.dist-info"
    nameless.mkdir()
    (nameless / "METADATA").write_text("Metadata-Version: 2.1\nVersion: 2.0\n", encoding = "utf-8")
    monkeypatch.setattr(im, "_metadata_scan_paths", lambda: [str(site)])

    versions = im.installed_versions("demo")
    assert versions == ["", "1.0"]
    assert im.invalid_metadata_paths("demo") == [nameless]
    assert im.metadata_conflict(versions) is True


def test_malformed_matching_metadata_invalidates_the_manifest(
    tmp_path, monkeypatch, install_root, req_root
):
    site = tmp_path / "site-packages"
    site.mkdir()
    _write_dist_metadata(site, "demo", "1.0")
    monkeypatch.setattr(im, "_metadata_scan_paths", lambda: [str(site)])
    im.write_manifest(root = install_root, req_root = req_root, package_name = "demo")

    malformed = site / "demo-2.0.dist-info"
    malformed.mkdir()
    (malformed / "METADATA").write_bytes(b"\xff\xfe")

    state = im.verify_install(root = install_root, req_root = req_root, package_name = "demo")
    assert state["manifest_ok"] is False
    assert state["reason"] == "studio_install_metadata_conflict"


def test_duplicate_package_metadata_invalidates_the_manifest(
    tmp_path, monkeypatch, install_root, req_root
):
    site = tmp_path / "site-packages"
    site.mkdir()
    _write_dist_metadata(site, "demo", "1.0")
    _write_dist_metadata(site, "demo", "2.0")
    monkeypatch.setattr(im, "_metadata_scan_paths", lambda: [str(site)])

    im.write_manifest(root = install_root, req_root = req_root, package_name = "demo")
    state = im.verify_install(root = install_root, req_root = req_root, package_name = "demo")
    assert state["manifest_ok"] is False
    assert state["reason"] == "studio_install_metadata_conflict"


def test_duplicate_zoo_metadata_invalidates_a_core_package_manifest(
    tmp_path, monkeypatch, install_root, req_root
):
    site = tmp_path / "site-packages"
    site.mkdir()
    _write_dist_metadata(site, "demo", "1.0")
    _write_dist_metadata(site, "unsloth-zoo", "1.0")
    _write_dist_metadata(site, "unsloth-zoo", "2.0")
    monkeypatch.setattr(im, "_metadata_scan_paths", lambda: [str(site)])

    im.write_manifest(root = install_root, req_root = req_root, package_name = "demo")
    state = im.verify_install(root = install_root, req_root = req_root, package_name = "demo")
    assert state["reason"] == "studio_install_metadata_conflict"


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
    # studio.txt lists PyJWT / python-docx / pymupdf, whose import names are jwt / docx / fitz, so matching on imports
    req = tmp_path / "studio.txt"
    req.write_text("pytest\n", encoding = "utf-8")
    assert im.missing_requirements(req) == []


def test_complete_install_verifies_ok(install_root, req_root):
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    state = im.verify_install(root = install_root, req_root = req_root, package_name = "pytest")
    assert state["manifest_ok"] is True
    assert state["deps_ok"] is False
    assert state["reason"] == "studio_deps_missing"
    assert "unsloth-definitely-not-a-real-package" in state["missing"]


def test_missing_manifest_reports_incomplete(install_root, req_root):
    state = im.verify_install(root = install_root, req_root = req_root, package_name = "pytest")
    assert state["ok"] is False
    assert state["manifest_ok"] is False
    assert state["reason"] == "studio_install_incomplete"


def test_interrupted_install_leaves_no_manifest(install_root, req_root):
    # remove_manifest() runs before the dependency pass, so a later kill cannot leave a stale-but-valid manifest behind.
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    assert im.manifest_path(install_root).is_file()
    assert im.remove_manifest(install_root) is True
    assert not im.manifest_path(install_root).is_file()
    state = im.verify_install(root = install_root, req_root = req_root, package_name = "pytest")
    assert state["reason"] == "studio_install_incomplete"


def test_remove_manifest_reports_whether_the_marker_is_really_gone(
    install_root, req_root, monkeypatch
):
    assert im.remove_manifest(install_root) is True

    # A surviving marker must be reported, not swallowed:
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
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    state = im.verify_install(
        root = install_root,
        req_root = req_root,
        package_name = "unsloth-definitely-not-a-real-package",
    )
    assert state["manifest_ok"] is True


@pytest.mark.parametrize("conflict", ["pytest", "unsloth-zoo"])
def test_foreign_metadata_conflicts_invalidate_the_manifest(install_root, req_root, conflict):
    # `studio update --package X` records X.
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    state = im.verify_install(
        root = install_root,
        req_root = req_root,
        package_name = "pytest",
        installed = {"pytest": pytest.__version__},
        installed_conflicts = {conflict},
    )

    assert state["manifest_ok"] is False
    assert state["reason"] == "studio_install_metadata_conflict"


def test_edited_requirements_invalidate_the_manifest(install_root, req_root):
    # The --local dev path: an edited studio.txt must re-run the dependency pass, not sit behind setup.sh's "up to
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    (req_root / "studio.txt").write_text("pytest\nrich\n", encoding = "utf-8")
    state = im.verify_install(root = install_root, req_root = req_root, package_name = "pytest")
    assert state["reason"] == "studio_install_requirements_changed"


def test_unwritable_root_degrades_to_incomplete(tmp_path, req_root):
    missing_root = tmp_path / "does" / "not" / "exist"
    assert im.write_manifest(root = missing_root, req_root = req_root) is None
    state = im.verify_install(root = missing_root, req_root = req_root, package_name = "pytest")
    assert state["ok"] is False


def test_no_torch_mode_round_trips_through_the_manifest(install_root, req_root):
    # `unsloth studio update` injects no UNSLOTH_NO_TORCH, so the venv has to remember how it was built or the update
    # reinstalls torch into a GGUF-only environment (and on Windows deletes the venv it is running out of).
    for recorded in (True, False):
        im.write_manifest(
            root = install_root,
            req_root = req_root,
            package_name = "pytest",
            no_torch = recorded,
        )
        assert im.recorded_no_torch(root = install_root) is recorded
        assert (
            json.loads((install_root / im.MANIFEST_NAME).read_text(encoding = "utf-8"))["no_torch"]
            is recorded
        )


def test_manifest_without_the_no_torch_key_reads_as_unknown(install_root, req_root):
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    payload = json.loads((install_root / im.MANIFEST_NAME).read_text(encoding = "utf-8"))
    assert "no_torch" not in payload

    assert im.recorded_no_torch(root = install_root) is None
    state = im.verify_install(root = install_root, req_root = req_root, package_name = "pytest")
    assert state["manifest_ok"] is True


def test_recorded_no_torch_tolerates_a_hand_edited_manifest(install_root, req_root):
    # Manifests written before the key existed must keep verifying, and must report None rather than False so callers
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    path = install_root / im.MANIFEST_NAME
    payload = json.loads(path.read_text(encoding = "utf-8"))

    for value, expected in (("true", True), ("ON", True), ("0", False), (123, None)):
        payload["no_torch"] = value
        path.write_text(json.dumps(payload), encoding = "utf-8")
        assert im.recorded_no_torch(root = install_root) is expected


def test_recorded_no_torch_reports_unknown_without_a_manifest(install_root):
    assert im.recorded_no_torch(root = install_root) is None


def test_marker_preserves_no_torch_across_the_manifest_drop(install_root, req_root):
    im.set_no_torch_marker(True, root = install_root)
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest", no_torch = True)
    assert im.recorded_no_torch(root = install_root) is True

    im.remove_manifest(root = install_root)
    assert im.recorded_no_torch(root = install_root) is True


def test_manifest_key_overrides_a_stale_marker(install_root, req_root):
    im.set_no_torch_marker(True, root = install_root)
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest", no_torch = False)
    assert im.recorded_no_torch(root = install_root) is False


def test_set_no_torch_marker_clears_itself_and_never_raises(install_root):
    # Migrating out of no-torch must not be blocked by a marker left behind.
    # remove_manifest() runs before every dependency pass, so a run killed during it leaves no manifest.
    im.set_no_torch_marker(True, root = install_root)
    assert im.no_torch_marker_path(root = install_root).exists()

    im.set_no_torch_marker(False, root = install_root)
    assert not im.no_torch_marker_path(root = install_root).exists()
    assert im.recorded_no_torch(root = install_root) is None

    # Absent directory: must degrade quietly, it runs mid-install.
    im.set_no_torch_marker(True, root = install_root / "does" / "not" / "exist")


def test_scan_paths_dedupes_a_lib64_symlink(tmp_path, monkeypatch):
    """purelib hardcodes `lib`, platlib follows sys.platlibdir.

    On a lib64 build (Fedora, SuSE) venv creates lib64 as a symlink to lib, so
    the two schemes name ONE directory by two paths. Scanning both reported
    every installed package twice, which made metadata_conflict() true for a
    perfectly healthy environment and sent the installer into a repair it could
    never finish.
    """
    real = tmp_path / "lib" / "python3.13" / "site-packages"
    real.mkdir(parents = True)
    (tmp_path / "lib64").symlink_to("lib")
    alias = tmp_path / "lib64" / "python3.13" / "site-packages"

    monkeypatch.setattr(
        sysconfig, "get_paths", lambda *a, **k: {"purelib": str(real), "platlib": str(alias)}
    )

    assert im._metadata_scan_paths() == [str(real)]


def test_scan_paths_keeps_genuinely_separate_roots(tmp_path, monkeypatch):
    pure = tmp_path / "purelib"
    plat = tmp_path / "platlib"
    pure.mkdir()
    plat.mkdir()

    # install_manifest imports sysconfig inside the function, so patch the module.
    monkeypatch.setattr(
        sysconfig, "get_paths", lambda *a, **k: {"purelib": str(pure), "platlib": str(plat)}
    )

    assert im._metadata_scan_paths() == [str(pure), str(plat)]


def test_a_record_outside_this_interpreters_scheme_is_not_installed(tmp_path, monkeypatch):
    """The scan is purelib/platlib only, so a user site or an inherited
    PYTHONPATH entry is invisible. _installed_version used to answer from all of
    sys.path, so this narrows it deliberately: every caller runs against the
    managed venv, where those trees are either disabled or someone else's. A
    duplicate the venv does not own must not make the venv look damaged.
    """
    scheme = tmp_path / "site-packages"
    scheme.mkdir()
    elsewhere = tmp_path / "user-site"
    elsewhere.mkdir()
    record = elsewhere / "demo-1.0.dist-info"
    record.mkdir()
    (record / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n", encoding = "utf-8"
    )

    monkeypatch.setattr(
        sysconfig, "get_paths", lambda *a, **k: {"purelib": str(scheme), "platlib": str(scheme)}
    )
    monkeypatch.syspath_prepend(str(elsewhere))

    assert im.installed_versions("demo") == []
    assert im.installed_version_probe("demo") == ("", False)


def _fake_venv(root, files):
    """A venv-shaped tree whose installed package ships its own requirements."""
    site = root / "lib" / "python3.12" / "site-packages"
    reqs = site / "studio" / "backend" / "requirements"
    (reqs / "single-env").mkdir(parents = True, exist_ok = True)
    for name, body in files.items():
        (reqs / name).write_text(body, encoding = "utf-8")
    return reqs


def test_the_manifest_records_the_venvs_own_requirements_not_the_installers(install_root, req_root):
    """
    The regression that broke every fresh desktop install on 2026-08-19.

    A desktop bundle carries its own `studio/install_python_stack.py`, so the
    digests used to come from whatever that bundle shipped. Verification reads
    the *installed* package's copy. v0.1.800-beta (2026-08-14) installed unsloth
    2026.8.18, #9148 had pinned openai in extras.txt in between, and the two
    trees disagreed: every install came up `studio_install_requirements_changed`
    and repaired itself before it would run.

    So the two roots are made deliberately different here, and the manifest must
    describe the one the verifier will read.
    """
    installed = _fake_venv(
        install_root, {"studio.txt": "pytest\n", "extras.txt": "openai==3.2.0\n"}
    )
    (req_root / "studio.txt").write_text("pytest\n", encoding = "utf-8")
    (req_root / "extras.txt").write_text("openai>=2.7.2\n", encoding = "utf-8")

    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")

    recorded = json.loads((install_root / im.MANIFEST_NAME).read_text(encoding = "utf-8"))
    assert recorded["requirement_files"] == im.requirement_digests(installed), (
        "the manifest recorded the installer's requirement digests; verification "
        "reads the installed package's, so a bundle even one commit behind marks "
        "every install it performs as stale"
    )

    # And the whole point: the install it just described reads as finished.
    state = im.verify_install(root = install_root, req_root = installed, package_name = "pytest")
    assert state["manifest_ok"] is True, state["reason"]


def test_a_source_install_still_uses_the_root_it_was_given(install_root, req_root):
    """
    An editable / `--local` install has no copy under site-packages, and there the
    caller's root is already the tree both sides read. Falling back to it is what
    keeps test_edited_requirements_invalidate_the_manifest meaningful.
    """
    assert im.installed_requirements_root(install_root) is None
    im.write_manifest(root = install_root, req_root = req_root, package_name = "pytest")
    recorded = json.loads((install_root / im.MANIFEST_NAME).read_text(encoding = "utf-8"))
    assert recorded["requirement_files"] == im.requirement_digests(req_root)


def test_the_venv_resolver_finds_both_layouts(tmp_path):
    """posix `lib/python3.x/site-packages` and Windows `Lib/site-packages`."""
    for layout in ("lib/python3.12/site-packages", "Lib/site-packages"):
        root = tmp_path / layout.replace("/", "_")
        reqs = root / layout / "studio" / "backend" / "requirements"
        reqs.mkdir(parents = True)
        assert im.installed_requirements_root(root) == reqs
