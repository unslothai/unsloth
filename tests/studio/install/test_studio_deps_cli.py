# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Coverage for unsloth_cli/_studio_deps.py.

Two things have to be right for the CLI half of the install check.

It must describe the venv it was *asked* about. The wheel ships studio/, so a
CLI installed outside the managed venv always finds its own copy of the manifest
helper, and would otherwise report on its own prefix: a healthy managed install
comes back "incomplete", a broken one comes back with the wrong missing list.

And it must name the *distribution* to install rather than the import that
failed. `pip install jwt` / `docx` / `fitz` all succeed and install unrelated
PyPI projects, leaving the backend just as broken as before.
"""

from __future__ import annotations

import importlib.util
import io
import json
import contextlib
import os
import pathlib
import shutil
import sys
import sysconfig

import pytest
import typer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEPS_PATH = REPO_ROOT / "unsloth_cli" / "_studio_deps.py"
MANIFEST_PATH = REPO_ROOT / "studio" / "install_manifest.py"
REQUIREMENTS = REPO_ROOT / "studio" / "backend" / "requirements"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MANIFEST = _load(MANIFEST_PATH, "install_manifest_for_deps_test")


def _studio_distributions() -> list:
    lines = (REQUIREMENTS / "studio.txt").read_text(encoding = "utf-8").splitlines()
    parsed = [_MANIFEST._parse_requirement_line(line) for line in lines]
    return [
        name
        for name, marker, _ in (p for p in parsed if p is not None)
        if _MANIFEST._marker_applies(marker)
    ]


def _studio_distribution_versions() -> dict:
    versions = {}
    lines = (REQUIREMENTS / "studio.txt").read_text(encoding = "utf-8").splitlines()
    for parsed in (_MANIFEST._parse_requirement_line(line) for line in lines):
        if parsed is None:
            continue
        name, marker, specifier = parsed
        if not _MANIFEST._marker_applies(marker):
            continue
        version = "1.0.0"
        for part in specifier.split(","):
            if part.startswith("=="):
                version = part[2:]
                break
            if part.startswith(">="):
                version = part[2:]
        versions[name] = version
    return versions


def _venv_executable(root: pathlib.Path) -> pathlib.Path:
    """Where _venv_site_packages looks for this venv's interpreter.

    Writing bin/python on Windows leaves the probe with nothing to run, so the
    fixture falls through to the glob fallback and the case under test never
    happens.
    """
    return root / "Scripts" / "python.exe" if os.name == "nt" else root / "bin" / "python"


def _make_venv(
    root: pathlib.Path,
    *,
    unsloth_version: str,
    distributions,
    extra_requirement = "",
):
    """A venv tree: pyvenv.cfg, the shipped studio/ package and .dist-info dirs."""
    site_packages = root / "lib" / "python3.11" / "site-packages"
    (site_packages / "studio" / "backend").mkdir(parents = True)
    shutil.copy(MANIFEST_PATH, site_packages / "studio" / "install_manifest.py")
    shutil.copytree(REQUIREMENTS, site_packages / "studio" / "backend" / "requirements")
    if extra_requirement:
        studio_txt = site_packages / "studio" / "backend" / "requirements" / "studio.txt"
        studio_txt.write_text(
            studio_txt.read_text(encoding = "utf-8") + extra_requirement, encoding = "utf-8"
        )
    (root / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding = "utf-8")
    studio_versions = _studio_distribution_versions()
    for name in [*distributions, "unsloth"]:
        version = unsloth_version if name == "unsloth" else studio_versions.get(name, "1.0.0")
        dist_info = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            encoding = "utf-8",
        )
    return site_packages


def _write_manifest(root: pathlib.Path, site_packages: pathlib.Path, version: str):
    (root / _MANIFEST.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema": _MANIFEST.MANIFEST_SCHEMA,
                "package": "unsloth",
                "package_version": version,
                "requirement_files": _MANIFEST.requirement_digests(
                    site_packages / "studio" / "backend" / "requirements",
                ),
            }
        ),
        encoding = "utf-8",
    )


@pytest.fixture
def cross_venv(tmp_path, monkeypatch):
    """`unsloth studio verify-install` run from a CLI outside the managed venv.

    Returns a callable: build the managed venv, then ask about it.
    """

    def build(
        *,
        managed_version = "2026.6.1",
        caller_version = "2026.7.9",
        managed_distributions = None,
        extra_requirement = "",
        with_manifest = True,
        duplicate_versions = (),
        inactive_duplicate_version = "",
        unresolved_active_paths = False,
        malformed_unrelated_metadata = False,
        malformed_core_metadata = False,
        nameless_core_metadata = False,
        versionless_core_metadata = False,
        editable_requirements = False,
    ):
        caller = tmp_path / "caller_venv"
        caller_site = _make_venv(caller, unsloth_version = caller_version, distributions = [])
        managed = tmp_path / "studio_home" / "unsloth_studio"
        managed_site = _make_venv(
            managed,
            unsloth_version = managed_version,
            distributions = _studio_distributions()
            if managed_distributions is None
            else managed_distributions,
            extra_requirement = extra_requirement,
        )
        if with_manifest:
            _write_manifest(managed, managed_site, managed_version)
        for index, version in enumerate(duplicate_versions):
            dist_info = managed_site / f"unsloth_duplicate_{index}-{version}.dist-info"
            dist_info.mkdir()
            (dist_info / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: unsloth\nVersion: {version}\n",
                encoding = "utf-8",
            )
        if inactive_duplicate_version:
            inactive_site = managed / "lib" / "python3.10" / "site-packages"
            dist_info = inactive_site / f"unsloth-{inactive_duplicate_version}.dist-info"
            dist_info.mkdir(parents = True)
            (dist_info / "METADATA").write_text(
                "Metadata-Version: 2.1\n"
                "Name: unsloth\n"
                f"Version: {inactive_duplicate_version}\n",
                encoding = "utf-8",
            )
        if unresolved_active_paths:
            (managed / "lib" / "python3.10" / "site-packages").mkdir(parents = True)
        if malformed_unrelated_metadata:
            malformed = managed_site / "unrelated-1.0.dist-info"
            malformed.mkdir()
            (malformed / "METADATA").write_bytes(b"\xff\xfe")
        if malformed_core_metadata:
            malformed = managed_site / "unsloth-2.0.dist-info"
            malformed.mkdir()
            (malformed / "METADATA").write_bytes(b"\xff\xfe")
        if nameless_core_metadata:
            nameless = managed_site / "unsloth-2.0.dist-info"
            nameless.mkdir()
            (nameless / "METADATA").write_text(
                "Metadata-Version: 2.1\nVersion: 2.0\n", encoding = "utf-8"
            )
        if versionless_core_metadata:
            versionless = managed_site / "unsloth-2.0.dist-info"
            versionless.mkdir()
            (versionless / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: unsloth\n", encoding = "utf-8"
            )
        checkout_requirements = None
        if editable_requirements:
            checkout_studio = tmp_path / "checkout" / "studio"
            shutil.move(str(managed_site / "studio"), checkout_studio)
            checkout_requirements = checkout_studio / "backend" / "requirements"
            executable = _venv_executable(managed)
            executable.parent.mkdir(parents = True, exist_ok = True)
            executable.write_text("probe placeholder", encoding = "utf-8")

        (caller_site / "unsloth_cli").mkdir(parents = True)
        shutil.copy(DEPS_PATH, caller_site / "unsloth_cli" / "_studio_deps.py")
        monkeypatch.setattr(sys, "prefix", str(caller))
        deps = _load(caller_site / "unsloth_cli" / "_studio_deps.py", "studio_deps_cross_venv")
        if checkout_requirements is not None:

            def editable_paths(args, **_kwargs):
                if "sysconfig" in args[-1]:
                    return json.dumps([str(managed_site), str(managed_site)])
                return json.dumps(str(checkout_requirements))

            monkeypatch.setattr(deps.subprocess, "check_output", editable_paths)
        if inactive_duplicate_version:
            executable = _venv_executable(managed)
            executable.parent.mkdir(parents = True, exist_ok = True)
            executable.write_text("probe placeholder", encoding = "utf-8")

            def active_paths(*_args, **_kwargs):
                return json.dumps([str(managed_site), str(managed_site)])

            monkeypatch.setattr(deps.subprocess, "check_output", active_paths)
        return deps.install_state(extra_roots = (managed,))

    return build


def test_a_healthy_managed_venv_is_not_reported_incomplete(cross_venv):
    """The caller's own prefix has no manifest and none of studio.txt, so
    describing it instead sends a working install through a needless repair."""
    state = cross_venv()
    assert state["ok"] is True, state
    assert state["reason"] is None
    assert state["missing"] == []


def test_a_newer_caller_does_not_look_like_a_changed_managed_version(cross_venv):
    """The version and requirement digests must come from the managed venv too:
    reading them here compares two unrelated installs."""
    state = cross_venv(managed_version = "2026.1.1", caller_version = "2026.12.31")
    assert state["ok"] is True, state


def test_a_managed_venv_missing_a_boot_dep_names_that_dep(cross_venv):
    """The other direction: report what is actually absent over there."""
    state = cross_venv(
        managed_distributions = [d for d in _studio_distributions() if d != "fastmcp"],
    )
    assert state["ok"] is False
    assert state["reason"] == "studio_deps_missing"
    assert state["missing"] == ["fastmcp"], state


def test_an_unfinished_managed_install_is_still_reported_incomplete(cross_venv):
    state = cross_venv(with_manifest = False)
    assert state["ok"] is False
    assert state["reason"] == "studio_install_incomplete"


@pytest.mark.parametrize("duplicate_version", ["2026.6.1", "2026.5.9"])
def test_duplicate_metadata_in_a_foreign_managed_venv_is_not_collapsed(
    cross_venv, duplicate_version
):
    state = cross_venv(duplicate_versions = (duplicate_version,))

    assert state["ok"] is False
    assert state["manifest_ok"] is False
    assert state["reason"] == "studio_install_metadata_conflict"


def test_inactive_python_site_packages_do_not_create_a_foreign_conflict(cross_venv):
    state = cross_venv(inactive_duplicate_version = "2025.1.1")

    assert state["ok"] is True, state
    assert state["reason"] is None


def test_unresolved_foreign_site_packages_fail_closed(cross_venv):
    state = cross_venv(unresolved_active_paths = True)

    assert state["ok"] is False
    assert state["manifest_ok"] is False
    assert state["deps_ok"] is False
    assert state["reason"] == "studio_install_incomplete"


def test_malformed_unrelated_foreign_metadata_does_not_hide_valid_records(cross_venv):
    state = cross_venv(malformed_unrelated_metadata = True)

    assert state["ok"] is True, state
    assert state["reason"] is None


def test_malformed_core_metadata_is_a_foreign_conflict(cross_venv):
    state = cross_venv(malformed_core_metadata = True)

    assert state["ok"] is False
    assert state["manifest_ok"] is False
    assert state["reason"] == "studio_install_metadata_conflict"


def test_nameless_core_metadata_is_a_foreign_conflict(cross_venv):
    state = cross_venv(nameless_core_metadata = True)

    assert state["ok"] is False
    assert state["manifest_ok"] is False
    assert state["reason"] == "studio_install_metadata_conflict"


def test_versionless_core_metadata_is_a_foreign_conflict(cross_venv):
    state = cross_venv(versionless_core_metadata = True)

    assert state["ok"] is False
    assert state["manifest_ok"] is False
    assert state["reason"] == "studio_install_metadata_conflict"


def test_editable_foreign_install_follows_its_requirements_checkout(cross_venv):
    state = cross_venv(editable_requirements = True)

    assert state["ok"] is True, state
    assert state["reason"] is None


# ── import name vs distribution name ─────────────────────────────────


@pytest.fixture
def deps():
    return _load(DEPS_PATH, "studio_deps_under_test")


def _remediation(deps, trigger: str, studio_missing) -> str:
    deps._missing_studio_packages = lambda: list(studio_missing)
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), pytest.raises(typer.Exit):
        with deps.studio_backend_imports("unsloth studio"):
            raise ModuleNotFoundError(f"No module named '{trigger}'", name = trigger)
    return stderr.getvalue()


@pytest.mark.parametrize(
    "trigger, distribution",
    [("jwt", "pyjwt"), ("docx", "python-docx"), ("fitz", "pymupdf")],
)
def test_a_missing_studio_package_is_named_by_its_distribution(deps, trigger, distribution):
    """`pip install jwt` installs a different JWT library and repairs nothing."""
    output = _remediation(deps, trigger, [distribution])
    assert f"pip install {trigger}" not in output, output
    assert distribution in output
    assert "unsloth studio update" in output


def test_a_normalised_name_still_counts_as_a_studio_dependency(deps):
    """studio.txt writes huggingface-hub; the import is huggingface_hub."""
    output = _remediation(deps, "huggingface_hub", ["huggingface-hub"])
    assert "Install it:" not in output, output
    assert "also missing:" not in output, output


def test_a_missing_submodule_is_traced_to_its_installable_package(deps):
    """exc.name is dotted when the top level survived a partial install, and
    `pip install fastmcp.server` is not a package name at all."""
    output = _remediation(deps, "fastmcp.server", ["fastmcp"])
    assert "fastmcp.server" not in output.split("Install it:")[-1], output
    assert "Install it:" not in output, output
    assert "unsloth studio update" in output


def test_a_non_studio_dependency_keeps_its_own_install_line(deps):
    """train reaches torch through the same wrapped import and the studio extra
    does not carry it."""
    output = _remediation(deps, "torch", ["pyjwt"])
    assert "pip install torch" in output
    assert "also missing: pyjwt" in output


def test_studio_only_guard_preserves_non_studio_failures(deps):
    deps._missing_studio_packages = lambda: ["pyjwt"]
    with pytest.raises(ModuleNotFoundError):
        with deps.studio_backend_imports("unsloth inference", studio_only = True):
            raise ModuleNotFoundError("No module named 'mlx'", name = "mlx")


def test_the_import_map_only_names_studio_distributions():
    """Drift guard: an entry pointing at a dropped requirement is dead advice."""
    known = {deps_name.lower() for deps_name in _studio_distributions()}
    module = _load(DEPS_PATH, "studio_deps_map_check")
    for import_name, distribution in module._IMPORT_TO_DISTRIBUTION.items():
        assert distribution.lower() in known, (
            f"_IMPORT_TO_DISTRIBUTION maps {import_name} to {distribution}, "
            "which studio.txt no longer requires"
        )


def test_a_torn_tree_without_the_manifest_helper_is_incomplete(tmp_path, monkeypatch):
    """studio/install_manifest.py ships in the same wheel as _studio_deps.py, so
    only a torn install has one without the other. Answering yes here launches a
    backend whose own files may be just as absent."""
    caller = tmp_path / "caller_venv"
    site_packages = caller / "lib" / "python3.11" / "site-packages"
    (site_packages / "unsloth_cli").mkdir(parents = True)
    shutil.copy(DEPS_PATH, site_packages / "unsloth_cli" / "_studio_deps.py")
    (caller / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding = "utf-8")
    monkeypatch.setattr(sys, "prefix", str(caller))

    deps = _load(site_packages / "unsloth_cli" / "_studio_deps.py", "studio_deps_torn_tree")
    state = deps.install_state()

    assert state["ok"] is False, state
    assert state["reason"] == "studio_install_manifest_missing"


# ── which prefix owns the manifest module ────────────────────────────


def test_a_manifest_inside_a_venv_site_packages_is_owned_by_that_venv(tmp_path, deps):
    """The wheel case: studio/ really is installed into that prefix."""
    venv = tmp_path / "venv"
    site_packages = venv / "lib" / "python3.11" / "site-packages"
    (site_packages / "studio").mkdir(parents = True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding = "utf-8")
    module_path = site_packages / "studio" / "install_manifest.py"
    shutil.copy(MANIFEST_PATH, module_path)

    module = _load(module_path, "manifest_in_site_packages")

    assert deps._venv_root_for_module(module) == venv


def test_an_editable_checkout_is_not_owned_by_a_surrounding_venv(tmp_path, deps):
    """`./install.sh --local` leaves studio/install_manifest.py in the repo. When the
    clone happens to live inside some other virtualenv's directory, the first
    pyvenv.cfg above it belongs to a venv the managed install has nothing to do
    with. Claiming it there made verify-install walk that venv's site-packages and
    report every managed dependency as missing, so `unsloth studio verify-install`
    exited 1 on a healthy install and setup.sh could never take its fast path."""
    surrounding = tmp_path / "unrelated_venv"
    (surrounding / "lib" / "python3.11" / "site-packages").mkdir(parents = True)
    (surrounding / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding = "utf-8")
    repo = surrounding / "repo"
    (repo / "studio").mkdir(parents = True)
    module_path = repo / "studio" / "install_manifest.py"
    shutil.copy(MANIFEST_PATH, module_path)

    module = _load(module_path, "manifest_in_editable_checkout")

    assert deps._venv_root_for_module(module) is None


def test_scan_paths_dedupes_a_lib64_symlink(tmp_path, monkeypatch, deps):
    """A lib64 build names one site-packages twice.

    purelib hardcodes `lib` while platlib follows sys.platlibdir, and venv
    creates lib64 as a symlink to lib, so Fedora and SuSE would otherwise scan
    the same directory twice and report EVERY installed package as having
    duplicate metadata -- failing `unsloth studio update` on a healthy venv.
    """
    real = tmp_path / "lib" / "python3.13" / "site-packages"
    real.mkdir(parents = True)
    (tmp_path / "lib64").symlink_to("lib")
    alias = tmp_path / "lib64" / "python3.13" / "site-packages"

    monkeypatch.setattr(
        sysconfig, "get_paths", lambda *a, **k: {"purelib": str(real), "platlib": str(alias)}
    )

    assert deps._scan_paths() == {"path": [str(real)]}


def test_a_foreign_lib64_venv_reports_no_duplicates(tmp_path, deps):
    venv = tmp_path / "managed"
    real = venv / "lib" / "python3.13" / "site-packages"
    real.mkdir(parents = True)
    (venv / "lib64").symlink_to("lib")
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\nversion = 3.13.0\n", encoding = "utf-8")
    dist_info = real / "unsloth-2026.8.15.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: unsloth\nVersion: 2026.8.15\n", encoding = "utf-8"
    )

    found = deps._distributions_in(venv)

    assert found is not None
    installed, conflicts = found
    assert installed["unsloth"] == "2026.8.15"
    assert conflicts == set()


def test_a_nameless_local_record_is_reported_as_a_conflict(tmp_path, monkeypatch, deps):
    """install_manifest.installed_versions() calls this state a conflict, so the
    CLI's own check has to agree: pip cannot parse the record either."""
    site = tmp_path / "site-packages"
    site.mkdir()
    for name, metadata in (
        (
            "unsloth-2026.8.15.dist-info",
            "Metadata-Version: 2.1\nName: unsloth\nVersion: 2026.8.15\n",
        ),
        ("unsloth-2026.8.12.dist-info", "Metadata-Version: 2.1\nVersion: 2026.8.12\n"),
    ):
        entry = site / name
        entry.mkdir()
        (entry / "METADATA").write_text(metadata, encoding = "utf-8")

    monkeypatch.setattr(deps, "_scan_paths", lambda: {"path": [str(site)]})

    conflicts = deps.installed_metadata_conflicts(names = ("unsloth",))

    assert len(conflicts) == 1
    assert "unsloth: multiple metadata records" in conflicts[0]


def test_a_single_unreadable_record_is_reported_as_a_conflict(tmp_path, monkeypatch, deps):
    site = tmp_path / "site-packages"
    site.mkdir()
    entry = site / "unsloth-2026.8.15.dist-info"
    entry.mkdir()
    (entry / "METADATA").write_bytes(b"Metadata-Version: 2.1\nName: un\xffsloth\n")

    monkeypatch.setattr(deps, "_scan_paths", lambda: {"path": [str(site)]})

    assert deps.installed_metadata_conflicts(names = ("unsloth",))


def test_one_readable_record_is_not_a_conflict(tmp_path, monkeypatch, deps):
    site = tmp_path / "site-packages"
    site.mkdir()
    entry = site / "unsloth-2026.8.15.dist-info"
    entry.mkdir()
    (entry / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: unsloth\nVersion: 2026.8.15\n", encoding = "utf-8"
    )

    monkeypatch.setattr(deps, "_scan_paths", lambda: {"path": [str(site)]})

    assert deps.installed_metadata_conflicts(names = ("unsloth",)) == []


def test_a_versionless_local_record_is_reported_as_a_conflict(tmp_path, monkeypatch, deps):
    """install_manifest.metadata_conflict() counts an empty version as a
    conflict, so trusting the same record here would leave the two checks
    disagreeing about one directory, and would let the file-damage scan treat an
    unparseable record as authoritative."""
    site = tmp_path / "site-packages"
    site.mkdir()
    entry = site / "unsloth-2026.8.15.dist-info"
    entry.mkdir()
    (entry / "METADATA").write_text("Metadata-Version: 2.1\nName: unsloth\n", encoding = "utf-8")

    monkeypatch.setattr(deps, "_scan_paths", lambda: {"path": [str(site)]})

    conflicts = deps.installed_metadata_conflicts(names = ("unsloth",))

    assert len(conflicts) == 1
    assert "unreadable metadata" in conflicts[0]
