# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The payload scan behind `studio_install_damaged`.

Both directions: real damage has to be seen, and the trees our own setup
rewrites must not be mistaken for it, since a false positive repairs a healthy
venv on every run.
"""

import csv
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "studio"))

import install_manifest  # noqa: E402


PKG = "demo"
VER = "1.0"


@pytest.fixture
def site_packages(tmp_path, monkeypatch):
    """A throwaway site-packages that the scan treats as the venv's own."""
    root = tmp_path / "site-packages"
    root.mkdir()
    monkeypatch.setattr(install_manifest, "_metadata_scan_paths", lambda: [str(root)])
    return root


def _dist(
    site_packages: Path,
    name = PKG,
    version = VER,
) -> Path:
    dist_info = site_packages / f"{name}-{version}.dist-info"
    dist_info.mkdir(parents = True, exist_ok = True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding = "utf-8"
    )
    return dist_info


def _record(dist_info: Path, rows) -> None:
    buf = io.StringIO(newline = "")
    writer = csv.writer(buf, lineterminator = "\n")
    for row in rows:
        writer.writerow(row)
    (dist_info / "RECORD").write_text(buf.getvalue(), encoding = "utf-8", newline = "")


def _write(path: Path, text: str) -> int:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")
    return len(text.encode())


def _healthy(site_packages: Path):
    dist_info = _dist(site_packages)
    size = _write(site_packages / PKG / "__init__.py", "x = 1\n")
    return dist_info, [
        [f"{PKG}/__init__.py", "sha256=x", size],
        [f"{dist_info.name}/RECORD", "", ""],
    ]




def test_a_healthy_install_reports_nothing(site_packages):
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    assert install_manifest.damaged_payload_files(PKG) == []


def test_a_quarantined_file_is_damage(site_packages):
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    (site_packages / PKG / "__init__.py").unlink()
    assert install_manifest.damaged_payload_files(PKG) == [f"{PKG}/__init__.py is missing"]


def test_a_truncated_file_is_damage(site_packages):
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    (site_packages / PKG / "__init__.py").write_text("", encoding = "utf-8")
    assert install_manifest.damaged_payload_files(PKG) == [
        f"{PKG}/__init__.py is 0 bytes, expected 6"
    ]


def test_a_directory_where_a_file_was_recorded_is_damage(site_packages):
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    target = site_packages / PKG / "__init__.py"
    target.unlink()
    target.mkdir()
    assert install_manifest.damaged_payload_files(PKG) == [
        f"{PKG}/__init__.py is not a regular file"
    ]


def test_the_findings_respect_the_limit(site_packages):
    dist_info = _dist(site_packages)
    _record(dist_info, [[f"{PKG}/m{i}.py", "sha256=x", 10] for i in range(9)])
    assert len(install_manifest.damaged_payload_files(PKG, limit = 3)) == 3




def test_a_regenerated_frontend_dist_is_not_damage(site_packages):
    """The wheel ships the bundle, so its files are in RECORD, and setup's own
    `npm run build` rehashes every one of them."""
    dist_info, rows = _healthy(site_packages)
    old = _write(site_packages / "studio/frontend/dist/assets/index-OLDHASH1.js", "x\n")
    index = _write(site_packages / "studio/frontend/dist/index.html", "<html>old</html>\n")
    _record(
        dist_info,
        rows
        + [
            ["studio/frontend/dist/assets/index-OLDHASH1.js", "sha256=x", old],
            ["studio/frontend/dist/index.html", "sha256=x", index],
        ],
    )
    (site_packages / "studio/frontend/dist/assets/index-OLDHASH1.js").unlink()
    _write(site_packages / "studio/frontend/dist/assets/index-NEWHASH2.js", "y\n")
    _write(site_packages / "studio/frontend/dist/index.html", "<html>\n")

    assert install_manifest.damaged_payload_files(PKG) == []


def test_damage_outside_the_regenerated_tree_still_counts(site_packages):
    """The carve-out is a subtree, not an amnesty."""
    dist_info, rows = _healthy(site_packages)
    old = _write(site_packages / "studio/frontend/dist/assets/index-OLDHASH1.js", "x\n")
    _record(dist_info, rows + [["studio/frontend/dist/assets/index-OLDHASH1.js", "sha256=x", old]])
    (site_packages / "studio/frontend/dist/assets/index-OLDHASH1.js").unlink()
    (site_packages / PKG / "__init__.py").unlink()

    assert install_manifest.damaged_payload_files(PKG) == [f"{PKG}/__init__.py is missing"]


def test_a_shrunk_package_lock_is_not_damage(site_packages):
    """npm dedupes hoisted entries, shrinking the lockfile below its recorded size."""
    dist_info, rows = _healthy(site_packages)
    lock = _write(site_packages / "studio/frontend/package-lock.json", '{"a": 1, "b": 2}\n')
    _record(dist_info, rows + [["studio/frontend/package-lock.json", "sha256=x", lock]])
    _write(site_packages / "studio/frontend/package-lock.json", "{}\n")

    assert install_manifest.damaged_payload_files(PKG) == []


def test_a_deleted_package_lock_is_still_damage(site_packages):
    """Only the size claim is waived for a rewritten file, not its existence."""
    dist_info, rows = _healthy(site_packages)
    lock = _write(site_packages / "studio/frontend/package-lock.json", "{}\n")
    _record(dist_info, rows + [["studio/frontend/package-lock.json", "sha256=x", lock]])
    (site_packages / "studio/frontend/package-lock.json").unlink()

    assert install_manifest.damaged_payload_files(PKG) == [
        "studio/frontend/package-lock.json is missing"
    ]


def test_a_shared_non_runtime_root_is_not_damage(site_packages):
    """unsloth_zoo <= 2026.8.5 shipped tests/ into a squatted namespace."""
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows + [["tests/conftest.py", "sha256=x", 9999]])
    assert install_manifest.damaged_payload_files(PKG) == []


def test_a_larger_file_is_not_damage(site_packages):
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    _write(site_packages / PKG / "__init__.py", "x = 1\n" * 50)
    assert install_manifest.damaged_payload_files(PKG) == []


def test_an_absent_record_is_not_damage(site_packages):
    """RECORD is optional per the installed-projects spec."""
    _dist(site_packages)
    _write(site_packages / PKG / "__init__.py", "x = 1\n")
    assert install_manifest.damaged_payload_files(PKG) == []


def test_an_unreadable_file_is_not_damage(site_packages):
    """A reinstall cannot fix EACCES, and unreadable is not missing."""
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    target = site_packages / PKG / "__init__.py"
    os.chmod(target, 0o000)
    try:
        if os.access(target, os.R_OK):
            pytest.skip("running as a user that ignores the mode")
        assert install_manifest.damaged_payload_files(PKG) == []
    finally:
        os.chmod(target, 0o644)


def test_a_quoted_line_break_in_a_recorded_name_is_not_damage(site_packages):
    """RECORD is CSV: a quoted field may hold a newline that splitlines breaks."""
    dist_info = _dist(site_packages)
    size = _write(site_packages / PKG / "we\nird.py", "x = 1\n")
    _record(
        dist_info, [[f"{PKG}/we\nird.py", "sha256=x", size], [f"{dist_info.name}/RECORD", "", ""]]
    )
    assert install_manifest.damaged_payload_files(PKG) == []


def test_an_unrelated_distribution_is_not_scanned(site_packages):
    """It answers for the managed package alone."""
    other = _dist(site_packages, name = "other")
    _record(other, [["other/__init__.py", "sha256=x", 10]])
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    assert install_manifest.damaged_payload_files(PKG) == []


def test_an_unreadable_environment_reports_undamaged(monkeypatch):
    """A scan that cannot answer leaves the fast path as it found it."""
    monkeypatch.setattr(install_manifest, "_metadata_scan_paths", lambda: [])
    assert install_manifest.damaged_payload_files(PKG) == []




def _complete_install(tmp_path, monkeypatch, site_packages):
    """A manifest and requirements that make every metadata check pass."""
    req_root = tmp_path / "requirements"
    req_root.mkdir()
    (req_root / install_manifest.BOOT_REQUIREMENT_FILE).write_text("", encoding = "utf-8")
    install_manifest.write_manifest(root = tmp_path, req_root = req_root, package_name = PKG)
    return req_root


def test_a_damaged_payload_invalidates_an_otherwise_complete_install(
    tmp_path, monkeypatch, site_packages
):
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    req_root = _complete_install(tmp_path, monkeypatch, site_packages)

    ok_state = install_manifest.verify_install(root = tmp_path, req_root = req_root, deep = True)
    assert ok_state["ok"] is True and ok_state["reason"] is None

    (site_packages / PKG / "__init__.py").unlink()
    state = install_manifest.verify_install(root = tmp_path, req_root = req_root, deep = True)
    assert state["ok"] is False
    assert state["reason"] == "studio_install_damaged"
    # The deps walk still succeeded;
    assert state["deps_ok"] is True


def test_the_scan_is_off_unless_asked_for(tmp_path, monkeypatch, site_packages):
    """Default off: an external CLI can load a newer copy of this module out of
    the venv it drives, and cannot decline a scan it does not expect."""
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    req_root = _complete_install(tmp_path, monkeypatch, site_packages)
    (site_packages / PKG / "__init__.py").unlink()

    for kwargs in ({}, {"deep": False}):
        state = install_manifest.verify_install(root = tmp_path, req_root = req_root, **kwargs)
        assert state["ok"] is True and state["reason"] is None, kwargs


def test_describing_a_foreign_venv_never_scans_this_one(tmp_path, monkeypatch, site_packages):
    """`installed` means another venv, whose tree this interpreter cannot stat."""
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    req_root = _complete_install(tmp_path, monkeypatch, site_packages)
    (site_packages / PKG / "__init__.py").unlink()

    state = install_manifest.verify_install(
        root = tmp_path, req_root = req_root, installed = {PKG: VER}, deep = True
    )
    assert state["reason"] != "studio_install_damaged"


def test_the_scan_runs_last_and_diverts_no_existing_reason(tmp_path, monkeypatch, site_packages):
    """A missing manifest still reports incomplete, not damaged."""
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    (site_packages / PKG / "__init__.py").unlink()

    state = install_manifest.verify_install(root = tmp_path, deep = True)
    assert state["reason"] == "studio_install_incomplete"




def test_the_installers_ask_for_the_scan():
    """They import this module from their own directory, so unlike an external
    CLI they can never be skewed against it."""
    repo = Path(__file__).resolve().parents[3]
    for name in ("studio/setup.sh", "studio/setup.ps1"):
        text = (repo / name).read_text(encoding = "utf-8")
        assert "verify_install(deep = True)" in text, f"{name} stopped asking for the scan"
        # and it must survive an older module that has no such keyword
        assert "except TypeError:" in text, f"{name} lost its older-tree fallback"


def test_the_desktop_boot_path_does_not_ask_for_the_scan():
    """`desktop-capabilities` feeds the Tauri preflight under a 10 second
    timeout, and a probe that overruns it repairs a healthy install."""
    repo = Path(__file__).resolve().parents[3]
    deps = (repo / "unsloth_cli" / "_studio_deps.py").read_text(encoding = "utf-8")
    assert "def install_state(extra_roots: Sequence[Path] = (), deep: bool = False)" in deps

    cli = (repo / "unsloth_cli" / "commands" / "studio.py").read_text(encoding = "utf-8")
    assert "def _install_state(deep: bool = False)" in cli
    assert "_install_state(deep = True)" in cli




def test_a_package_directory_replaced_by_a_file_is_damage(site_packages):
    """NotADirectoryError is an OSError but not a FileNotFoundError, so the
    generic arm read a payload replaced by a quarantine stub as healthy."""
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    import shutil

    shutil.rmtree(site_packages / PKG)
    (site_packages / PKG).write_text("quarantined", encoding = "utf-8")
    assert install_manifest.damaged_payload_files(PKG) == [f"{PKG}/__init__.py is not reachable"]


def test_the_companion_distribution_is_scanned_too(site_packages):
    """unsloth-zoo is not in studio.txt, so nothing else would notice it gone."""
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    zoo_info = _dist(site_packages, name = "unsloth-zoo", version = "2.0")
    size = _write(site_packages / "unsloth_zoo" / "__init__.py", "y = 2\n")
    _record(zoo_info, [["unsloth_zoo/__init__.py", "sha256=y", size]])

    assert install_manifest.damaged_payload_files(PKG, companion_names = ("unsloth-zoo",)) == []
    (site_packages / "unsloth_zoo" / "__init__.py").unlink()
    assert install_manifest.damaged_payload_files(PKG) == []
    assert install_manifest.damaged_payload_files(PKG, companion_names = ("unsloth-zoo",)) == [
        "unsloth_zoo/__init__.py is missing"
    ]


def test_scan_paths_points_the_scan_at_another_venv(tmp_path, site_packages):
    """Without it a caller holding a foreign venv could only check metadata."""
    other = tmp_path / "other-site-packages"
    other.mkdir()
    dist_info = _dist(other)
    size = _write(other / PKG / "__init__.py", "x = 1\n")
    _record(dist_info, [[f"{PKG}/__init__.py", "sha256=x", size]])

    assert install_manifest.damaged_payload_files(PKG, scan_paths = [str(other)]) == []
    (other / PKG / "__init__.py").unlink()
    # This interpreter's own tree is healthy and must not be the one answered for.
    healthy_info, rows = _healthy(site_packages)
    _record(healthy_info, rows)
    assert install_manifest.damaged_payload_files(PKG) == []
    assert install_manifest.damaged_payload_files(PKG, scan_paths = [str(other)]) == [
        f"{PKG}/__init__.py is missing"
    ]


def test_a_foreign_venv_is_scanned_when_its_paths_are_given(tmp_path, monkeypatch, site_packages):
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    req_root = _complete_install(tmp_path, monkeypatch, site_packages)
    (site_packages / PKG / "__init__.py").unlink()

    state = install_manifest.verify_install(
        root = tmp_path,
        req_root = req_root,
        package_name = PKG,
        installed = {PKG: VER},
        deep = True,
        scan_paths = [str(site_packages)],
    )
    assert state["reason"] == "studio_install_damaged"


def test_an_uninstalled_managed_distribution_is_damage(tmp_path, monkeypatch, site_packages):
    """No dist-info at all: every version check compares against it and passes."""
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    req_root = _complete_install(tmp_path, monkeypatch, site_packages)

    import shutil

    shutil.rmtree(dist_info)
    shutil.rmtree(site_packages / PKG)
    shallow = install_manifest.verify_install(root = tmp_path, req_root = req_root)
    assert shallow["ok"] is True, "the metadata-only checks cannot see this"

    state = install_manifest.verify_install(root = tmp_path, req_root = req_root, deep = True)
    assert state["ok"] is False
    assert state["reason"] == "studio_install_damaged"


def test_the_budget_is_checked_before_every_stat(site_packages, monkeypatch):
    """Batched every 64 rows, one slow stat per row overran the budget."""
    dist_info = _dist(site_packages)
    rows = []
    for index in range(500):
        rel = f"{PKG}/mod{index}.py"
        _write(site_packages / PKG / f"mod{index}.py", "x = 1\n")
        rows.append([rel, "sha256=x", 6])
    _record(dist_info, rows)

    clock = {"now": 0.0}
    monkeypatch.setattr(install_manifest.time, "monotonic", lambda: clock["now"])
    real_stat = Path.stat

    def slow_stat(self, *args, **kwargs):
        clock["now"] += 1.0
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", slow_stat)
    install_manifest.damaged_payload_files(PKG, budget_seconds = 5.0)
    # The lower bound keeps the patched stat honest:
    assert 5.0 <= clock["now"] <= 6.0


def test_the_cli_hands_over_the_managed_venvs_own_paths():
    """RECORD rows resolve against the distribution that declared them, so
    without this the foreign branch scans the CLI's own tree, or nothing."""
    repo = Path(__file__).resolve().parents[3]
    deps = (repo / "unsloth_cli" / "_studio_deps.py").read_text(encoding = "utf-8")
    assert 'kwargs["scan_paths"] = paths' in deps
    # Guarded on its own: a module new enough for `deep` may still predate it.
    assert '_verify_install_supports(module, "scan_paths")' in deps


def test_a_quarantined_console_script_is_damage(tmp_path, site_packages):
    """Skipping every parent-relative row meant the `unsloth` command could be
    gone, tree intact, while the deep check called the install healthy."""
    venv = site_packages.parent
    (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding = "utf-8")
    dist_info = _dist(site_packages)
    size = _write(site_packages / PKG / "__init__.py", "x = 1\n")
    launcher = _write(venv / "bin" / PKG, "#!/usr/bin/env python\n")
    _record(
        dist_info,
        [[f"{PKG}/__init__.py", "sha256=x", size], [f"../bin/{PKG}", "sha256=b", launcher]],
    )
    assert install_manifest.damaged_payload_files(PKG) == []
    (venv / "bin" / PKG).unlink()
    assert install_manifest.damaged_payload_files(PKG) == [f"../bin/{PKG} is missing"]


def test_a_recorded_path_outside_the_venv_is_not_damage(tmp_path, site_packages):
    """It belongs to something else, and reinstalling ours would not restore it."""
    venv = site_packages.parent
    (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding = "utf-8")
    dist_info = _dist(site_packages)
    _record(dist_info, [["../../elsewhere/thing", "sha256=z", 4]])
    assert install_manifest.damaged_payload_files(PKG) == []


def test_a_parent_relative_row_outside_a_venv_is_skipped(site_packages):
    """No pyvenv.cfg, so nothing bounds the row and it stays out of scope."""
    dist_info = _dist(site_packages)
    _record(dist_info, [[f"../bin/{PKG}", "sha256=b", 4]])
    assert install_manifest.damaged_payload_files(PKG) == []


def test_a_vanished_companion_is_damage(tmp_path, monkeypatch, site_packages):
    """No dist-info leaves the RECORD scan nothing to walk, and no check above
    ever looked at the companion's version."""
    dist_info = _dist(site_packages, name = "unsloth", version = VER)
    size = _write(site_packages / "unsloth" / "__init__.py", "x = 1\n")
    _record(dist_info, [["unsloth/__init__.py", "sha256=x", size]])
    req_root = tmp_path / "requirements"
    req_root.mkdir()
    (req_root / install_manifest.BOOT_REQUIREMENT_FILE).write_text("", encoding = "utf-8")
    monkeypatch.setattr(install_manifest, "installed_versions", lambda name: [VER])
    install_manifest.write_manifest(root = tmp_path, req_root = req_root, package_name = "unsloth")

    state = install_manifest.verify_install(root = tmp_path, req_root = req_root, deep = True)
    assert state["ok"] is True

    monkeypatch.setattr(
        install_manifest,
        "installed_versions",
        lambda name: [] if _canonical_name(name) == "unsloth-zoo" else [VER],
    )
    state = install_manifest.verify_install(root = tmp_path, req_root = req_root, deep = True)
    assert state["reason"] == "studio_install_damaged"


def test_a_custom_package_does_not_have_to_ship_unsloth_zoo(tmp_path, monkeypatch, site_packages):
    """`--package X` installs X alone, so its neighbours are not ours to fix."""
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    req_root = _complete_install(tmp_path, monkeypatch, site_packages)
    zoo_info = _dist(site_packages, name = "unsloth-zoo", version = "2.0")
    _write(site_packages / "unsloth_zoo" / "__init__.py", "y = 2\n")
    _record(zoo_info, [["unsloth_zoo/__init__.py", "sha256=y", 999999]])

    state = install_manifest.verify_install(
        root = tmp_path, req_root = req_root, package_name = PKG, deep = True
    )
    assert state["ok"] is True


def _canonical_name(name):
    return install_manifest._canonical(name)


def test_a_custom_package_installs_no_companion(monkeypatch):
    """The installer's own view has to agree with the scan's."""
    import install_python_stack as ips

    assert ips._core_package_names("unsloth") == ("unsloth", "unsloth-zoo")
    assert ips._core_package_names("unsloth-nightly") == ("unsloth-nightly",)


def test_a_manifest_with_no_recorded_version_is_damage(tmp_path, monkeypatch, site_packages):
    """write_manifest stores whatever version it could read, so a package gone
    when it ran is recorded as a finished install with none. Absence is the only
    way to get here empty: ambiguity sets `local_conflict` first."""
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    req_root = _complete_install(tmp_path, monkeypatch, site_packages)
    manifest_path = install_manifest.manifest_path(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    manifest["package_version"] = ""
    manifest_path.write_text(json.dumps(manifest), encoding = "utf-8")

    monkeypatch.setattr(install_manifest, "installed_versions", lambda name: [])
    state = install_manifest.verify_install(root = tmp_path, req_root = req_root, deep = True)
    assert state["reason"] == "studio_install_damaged"


def test_ambiguous_metadata_still_reports_its_own_reason(tmp_path, monkeypatch, site_packages):
    """Also empty, but a conflict: these strings are a contract the desktop reads."""
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    req_root = _complete_install(tmp_path, monkeypatch, site_packages)
    monkeypatch.setattr(install_manifest, "installed_versions", lambda name: [VER, "2.0"])
    state = install_manifest.verify_install(root = tmp_path, req_root = req_root, deep = True)
    assert state["reason"] == "studio_install_metadata_conflict"


def test_a_stat_that_never_returns_does_not_wedge_setup(site_packages, monkeypatch):
    """Nothing in-process bounds a syscall that never returns, and no installer
    wraps its `verify_install(deep = True)` call in a timeout."""
    import threading

    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    released = threading.Event()
    monkeypatch.setattr(Path, "stat", lambda self, *a, **k: released.wait())
    try:
        started = time.monotonic()
        assert install_manifest.damaged_payload_files(PKG, budget_seconds = 0.5) == []
        assert time.monotonic() - started < 5.0
    finally:
        released.set()


def test_an_unbounded_scan_stays_on_this_thread(site_packages, monkeypatch):
    """`budget_seconds = 0` is the installer: it needs the real answer."""
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    (site_packages / PKG / "__init__.py").unlink()
    caller = threading.get_ident()
    seen = {}
    real = install_manifest._scan_payload_files

    def record(*args, **kwargs):
        seen["thread"] = threading.get_ident()
        return real(*args, **kwargs)

    monkeypatch.setattr(install_manifest, "_scan_payload_files", record)
    assert install_manifest.damaged_payload_files(PKG, budget_seconds = 0.0) == [
        f"{PKG}/__init__.py is missing"
    ]
    assert seen["thread"] == caller


def _launcher_case(site_packages: Path):
    """A recorded console script, the way a wheel records one."""
    venv = site_packages.parent
    (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding = "utf-8")
    dist_info = _dist(site_packages)
    size = _write(venv / "Scripts" / "unsloth.exe", "MZ binary\n")
    _record(dist_info, [["../Scripts/unsloth.exe", "sha256=b", size]])
    return venv / "Scripts" / "unsloth.exe"


@pytest.mark.parametrize("suffix", [".update-stale", ".update-backup", ".deleteme"])
def test_a_launcher_an_update_moved_aside_is_not_damage(site_packages, suffix):
    """setup.ps1's deep check runs inside the window `_move_launcher_aside`
    opens, so without this every healthy Windows update loses its fast path and
    force-reinstalls the core package over the network."""
    launcher = _launcher_case(site_packages)
    launcher.rename(launcher.with_name(launcher.name + suffix))
    assert install_manifest.damaged_payload_files(PKG) == []


def test_a_launcher_with_nothing_staged_beside_it_is_still_damage(site_packages):
    """The quarantine this exists to catch, with no update in flight."""
    launcher = _launcher_case(site_packages)
    launcher.unlink()
    assert install_manifest.damaged_payload_files(PKG) == ["../Scripts/unsloth.exe is missing"]


def test_an_unrelated_sibling_does_not_excuse_a_missing_file(site_packages):
    """Only the three names `_recover_missing_launcher` reads count."""
    launcher = _launcher_case(site_packages)
    launcher.rename(launcher.with_name(launcher.name + ".bak"))
    assert install_manifest.damaged_payload_files(PKG) == ["../Scripts/unsloth.exe is missing"]


def test_a_truncated_staged_copy_is_no_excuse(site_packages):
    """`_recover_missing_launcher` reads these through `_is_valid_pe`."""
    launcher = _launcher_case(site_packages)
    launcher.unlink()
    (launcher.parent / "unsloth.exe.update-stale").write_text("", encoding = "utf-8")
    assert install_manifest.damaged_payload_files(PKG) == ["../Scripts/unsloth.exe is missing"]


def test_a_staged_copy_that_is_not_a_pe_is_no_excuse(site_packages):
    launcher = _launcher_case(site_packages)
    launcher.unlink()
    (launcher.parent / "unsloth.exe.deleteme").write_text("not a binary", encoding = "utf-8")
    assert install_manifest.damaged_payload_files(PKG) == ["../Scripts/unsloth.exe is missing"]


def test_only_the_launcher_is_ever_excused(site_packages):
    """Only the launcher is staged, so a stray sibling elsewhere is no excuse."""
    dist_info, rows = _healthy(site_packages)
    _record(dist_info, rows)
    module = site_packages / PKG / "__init__.py"
    module.rename(module.with_name(module.name + ".update-stale"))
    assert install_manifest.damaged_payload_files(PKG) == [f"{PKG}/__init__.py is missing"]


def _installer_helper_probe() -> str:
    """The manifest-helper probe exactly as setup.sh runs it."""
    import re

    repo = Path(__file__).resolve().parents[3]
    setup = (repo / "studio" / "setup.sh").read_text(encoding = "utf-8")
    match = re.search(
        r'if ! "\$VENV_DIR/bin/python" -c "\n(import os, sys\n.*?)" "\$SCRIPT_DIR"', setup, re.S
    )
    assert match, "the manifest-helper probe moved; this test is reading the wrong block"
    return match.group(1)


@pytest.mark.parametrize(
    "contents,expected",
    [
        (None, 0),
        ("def broken(:\n", 1),
        ("raise RuntimeError('quarantined')\n", 1),
    ],
    ids = ["absent-keeps-the-fast-path", "truncated-forces-repair", "raises-forces-repair"],
)
def test_an_unimportable_helper_forces_the_dependency_pass(tmp_path, contents, expected):
    """The one file whose damage silences every check that follows it.

    Absent keeps the old escape: telling it from an old tree needs a RECORD walk
    here, and the CLI reports it as studio_install_manifest_missing anyway.
    """
    import subprocess

    script_dir = tmp_path / "studio"
    script_dir.mkdir()
    if contents is not None:
        (script_dir / "install_manifest.py").write_text(contents, encoding = "utf-8")
    # -I -S so `script_dir` is the ONLY place install_manifest can come from.
    # The probe imports it off sys.path, and an install_manifest reachable from site-packages or PYTHONPATH satisfies
    # that import even when the directory under test is empty: the "absent" case then falls through to a real
    # verify_install against whatever tree the runner happens to have, and answers about that instead.
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _installer_helper_probe(), str(script_dir)],
        capture_output = True,
    )
    assert result.returncode == expected


def test_both_installers_run_the_same_helper_probe():
    repo = Path(__file__).resolve().parents[3]
    guard = "sys.exit(1 if os.path.isfile(os.path.join(sys.argv[1], 'install_manifest.py')) else 0)"
    for name in ("studio/setup.sh", "studio/setup.ps1"):
        assert guard in (repo / name).read_text(encoding = "utf-8"), name
