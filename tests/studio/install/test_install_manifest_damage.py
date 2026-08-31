# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The payload scan behind `studio_install_damaged`.

verify_install's other checks read metadata only, so a payload an antivirus
quarantined after installation still reports the announced version and takes the
dependency fast path -- the repair pass that would restore it never runs. These
cover both directions: real damage has to be seen, and the trees our own setup
rewrites must not be mistaken for it, because a false positive makes setup repair
a healthy venv on every run.
"""

import csv
import io
import json
import os
import sys
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


# ----------------------------------------------------------------- true positives


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


# ---------------------------------------------------------------- false positives


def test_a_regenerated_frontend_dist_is_not_damage(site_packages):
    """setup.sh runs `npm run build` in the installed tree.

    vite empties dist/ and rewrites every asset under a fresh content hash, so
    the recorded bundle names are gone by the installer's own doing. The wheel
    ships that bundle, so those rows are in RECORD.
    """
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


# ------------------------------------------------------------- verify_install wiring


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
    # The deps walk still succeeded; only the payload is at fault.
    assert state["deps_ok"] is True


def test_the_scan_is_off_unless_asked_for(tmp_path, monkeypatch, site_packages):
    """Default off, so every caller that predates the kwarg is unaffected.

    The skew fails the safe way round this way: an external CLI resolves this
    module out of the managed venv, so an older caller can load a newer copy of
    it, and that caller has no way to decline a scan it does not expect.
    """
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


# ------------------------------------------------------- who asks for the scan


def test_the_installers_ask_for_the_scan():
    """setup.sh and setup.ps1 are the two callers that want it.

    They import this module from their own directory, so unlike an external CLI
    they can never be skewed against it.
    """
    repo = Path(__file__).resolve().parents[3]
    for name in ("studio/setup.sh", "studio/setup.ps1"):
        text = (repo / name).read_text(encoding = "utf-8")
        assert "verify_install(deep = True)" in text, f"{name} stopped asking for the scan"
        # and it must survive an older module that has no such keyword
        assert "except TypeError:" in text, f"{name} lost its older-tree fallback"


def test_the_desktop_boot_path_does_not_ask_for_the_scan():
    """`desktop-capabilities` feeds the Tauri preflight under a 10 second timeout.

    Pinned because nothing else would notice it regressing: a refactor that made
    install_state scan by default would put a stat of every recorded file inside
    that budget, and a timed-out probe reports a healthy install stale and
    repairs it.
    """
    repo = Path(__file__).resolve().parents[3]
    deps = (repo / "unsloth_cli" / "_studio_deps.py").read_text(encoding = "utf-8")
    assert "def install_state(extra_roots: Sequence[Path] = (), deep: bool = False)" in deps

    cli = (repo / "unsloth_cli" / "commands" / "studio.py").read_text(encoding = "utf-8")
    assert "def _install_state(deep: bool = False)" in cli
    # verify-install is the one command that opts in.
    assert "_install_state(deep = True)" in cli
