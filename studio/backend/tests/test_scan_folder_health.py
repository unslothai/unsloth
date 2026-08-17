# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A registered folder Unsloth cannot read must say so, not look empty.

Covers the add-time probe (os.access alone passes paths macOS TCC still
refuses), the status the scan records when a folder stops being readable, and
that reading the status back never touches the disk.
"""

import errno
import os
import stat
from pathlib import Path

import pytest

from utils.paths.scan_folder_health import (
    STATUS_MISSING,
    STATUS_OK,
    STATUS_PERMISSION_DENIED,
    STATUS_UNREADABLE,
    annotate_scan_folders,
    classify_scan_error,
    clear_scan_failure,
    is_readable_dir,
    note_scan_folder_scanned,
    record_scan_failure,
    scan_folder_status,
)


@pytest.fixture(autouse = True)
def _clean_registry():
    """Each test starts with no remembered failures."""
    import utils.paths.scan_folder_health as health

    health._failed.clear()
    yield
    health._failed.clear()


def test_a_readable_directory_passes_the_probe(tmp_path: Path):
    (tmp_path / "model.gguf").write_bytes(b"stub")
    assert is_readable_dir(str(tmp_path)) is True


def test_an_empty_directory_still_passes(tmp_path: Path):
    # Nothing to list is not the same as refusing to list.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert is_readable_dir(str(empty)) is True


@pytest.mark.skipif(os.geteuid() == 0, reason = "root ignores mode bits")
@pytest.mark.skipif(os.name == "nt", reason = "POSIX mode bits")
def test_a_chmod_000_directory_fails_the_probe(tmp_path: Path):
    denied = tmp_path / "denied"
    denied.mkdir()
    (denied / "model.gguf").write_bytes(b"stub")
    denied.chmod(0o000)
    try:
        assert is_readable_dir(str(denied)) is False
    finally:
        denied.chmod(stat.S_IRWXU)


def test_the_probe_catches_what_os_access_misses(tmp_path: Path, monkeypatch):
    """The macOS TCC shape: mode bits say yes, opening the directory says no.

    This is the case the old os.access-only check accepted, which then showed
    the user a folder with no models in it and no reason why.
    """
    (tmp_path / "model.gguf").write_bytes(b"stub")
    assert os.access(str(tmp_path), os.R_OK | os.X_OK) is True

    def _refuse(path, *args, **kwargs):
        raise PermissionError(errno.EPERM, "Operation not permitted", str(path))

    monkeypatch.setattr(os, "scandir", _refuse)
    assert is_readable_dir(str(tmp_path)) is False


def test_a_missing_directory_fails_the_probe(tmp_path: Path):
    assert is_readable_dir(str(tmp_path / "gone")) is False


@pytest.mark.parametrize(
    "error, expected",
    [
        (PermissionError(errno.EACCES, "Permission denied"), STATUS_PERMISSION_DENIED),
        (PermissionError(errno.EPERM, "Operation not permitted"), STATUS_PERMISSION_DENIED),
        (FileNotFoundError(errno.ENOENT, "No such file or directory"), STATUS_MISSING),
        (NotADirectoryError(errno.ENOTDIR, "Not a directory"), STATUS_MISSING),
        (OSError(errno.EIO, "Input/output error"), STATUS_UNREADABLE),
        (OSError("no errno at all"), STATUS_UNREADABLE),
    ],
)
def test_scan_errors_classify_into_statuses(error: OSError, expected: str):
    assert classify_scan_error(error) == expected


def test_an_unknown_folder_reports_ok():
    assert scan_folder_status("/models/never-scanned") == STATUS_OK


def test_a_recorded_failure_is_readable_back():
    record_scan_failure("/models/denied", PermissionError(errno.EACCES, "Permission denied"))
    assert scan_folder_status("/models/denied") == STATUS_PERMISSION_DENIED


def test_a_folder_that_reads_again_clears_its_failure():
    record_scan_failure("/models/denied", PermissionError(errno.EACCES, "Permission denied"))
    clear_scan_failure("/models/denied")
    assert scan_folder_status("/models/denied") == STATUS_OK


def test_clearing_an_unknown_folder_is_harmless():
    clear_scan_failure("/models/never-scanned")
    assert scan_folder_status("/models/never-scanned") == STATUS_OK


def test_the_registry_stays_bounded():
    import utils.paths.scan_folder_health as health
    for i in range(health._MAX_TRACKED + 5):
        record_scan_failure(f"/models/{i}", PermissionError(errno.EACCES, "Permission denied"))
    assert len(health._failed) <= health._MAX_TRACKED


def test_annotate_marks_healthy_folders_ok():
    rows = [{"id": 1, "path": "/models/a", "created_at": "2026-01-01"}]
    assert annotate_scan_folders(rows) == [
        {"id": 1, "path": "/models/a", "created_at": "2026-01-01", "status": STATUS_OK}
    ]


def test_annotate_marks_only_the_folder_that_failed():
    record_scan_failure("/models/b", PermissionError(errno.EACCES, "Permission denied"))
    rows = [
        {"id": 1, "path": "/models/a", "created_at": "2026-01-01"},
        {"id": 2, "path": "/models/b", "created_at": "2026-01-02"},
    ]
    statuses = {row["path"]: row["status"] for row in annotate_scan_folders(rows)}
    assert statuses == {"/models/a": STATUS_OK, "/models/b": STATUS_PERMISSION_DENIED}


def test_annotate_does_not_mutate_the_database_rows():
    rows = [{"id": 1, "path": "/models/a", "created_at": "2026-01-01"}]
    annotate_scan_folders(rows)
    assert "status" not in rows[0]


def test_a_folder_that_scanned_and_found_models_is_ok():
    note_scan_folder_scanned("/models/never-stat-me", found = True)
    assert scan_folder_status("/models/never-stat-me") == STATUS_OK


def test_a_folder_that_disappeared_reports_missing(tmp_path: Path):
    # Unmounted drive or renamed folder: the scanners return nothing, same as empty.
    note_scan_folder_scanned(str(tmp_path / "gone"), found = False)
    assert scan_folder_status(str(tmp_path / "gone")) == STATUS_MISSING


def test_an_empty_folder_that_is_still_there_stays_ok(tmp_path: Path):
    note_scan_folder_scanned(str(tmp_path), found = False)
    assert scan_folder_status(str(tmp_path)) == STATUS_OK


@pytest.mark.skipif(os.geteuid() == 0, reason = "root ignores mode bits")
@pytest.mark.skipif(os.name == "nt", reason = "POSIX mode bits")
def test_the_real_scan_records_a_folder_it_cannot_read(tmp_path: Path):
    """End to end through collect_local_models, the scan behind the model list."""
    from routes.models import collect_local_models

    denied = tmp_path / "denied"
    denied.mkdir()
    (denied / "model.gguf").write_bytes(b"stub")
    denied.chmod(0o000)
    rows = [{"id": 1, "path": str(denied), "created_at": "2026-01-01"}]
    try:
        collect_local_models(tmp_path, custom_folders = list(rows))
        assert annotate_scan_folders(rows)[0]["status"] == STATUS_PERMISSION_DENIED

        # And it recovers on its own once the folder is readable again.
        denied.chmod(stat.S_IRWXU)
        collect_local_models(tmp_path, custom_folders = list(rows))
        assert annotate_scan_folders(rows)[0]["status"] == STATUS_OK
    finally:
        denied.chmod(stat.S_IRWXU)


def test_reading_status_never_touches_the_filesystem(monkeypatch):
    """The folder list must stay a dict lookup: no stat, no listing, no cost.

    Every filesystem entry point raises, so any disk access fails the test.
    """
    record_scan_failure("/models/b", PermissionError(errno.EACCES, "Permission denied"))

    def _boom(*args, **kwargs):
        raise AssertionError("scan folder status touched the filesystem")

    for name in ("scandir", "listdir", "stat", "lstat", "access", "open"):
        monkeypatch.setattr(os, name, _boom)
    monkeypatch.setattr(os.path, "exists", _boom)
    monkeypatch.setattr(os.path, "isdir", _boom)

    rows = [
        {"id": 1, "path": "/models/a", "created_at": "2026-01-01"},
        {"id": 2, "path": "/models/b", "created_at": "2026-01-02"},
    ]
    statuses = {row["path"]: row["status"] for row in annotate_scan_folders(rows)}
    assert statuses == {"/models/a": STATUS_OK, "/models/b": STATUS_PERMISSION_DENIED}
    assert scan_folder_status("/models/b") == STATUS_PERMISSION_DENIED
