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
    STATUS_PARTIAL,
    STATUS_PERMISSION_DENIED,
    STATUS_UNKNOWN,
    STATUS_UNREADABLE,
    annotate_scan_folders,
    classify_scan_error,
    clear_scan_failure,
    is_readable_dir,
    note_scan_folder_scanned,
    probe_folder,
    probe_status,
    record_scan_failure,
    refresh_failed_scan_folders,
    scan_folder_status,
)


# os.geteuid is missing on Windows, and a skipif condition is evaluated at import,
# so the check has to be resolved before the decorator sees it.
requires_posix_permissions = pytest.mark.skipif(
    os.name == "nt" or getattr(os, "geteuid", lambda: 0)() == 0,
    reason = "needs POSIX mode bits and a non-root user",
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


@requires_posix_permissions
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


def test_a_healthy_folder_costs_a_bounded_number_of_opens(tmp_path: Path, monkeypatch):
    """The scan probes even when models were found, so the bound is the guarantee.

    A denied model sits next to readable ones and nothing raises, so the only way
    to know is to ask, and the only thing keeping that cheap is the open budget.
    """
    import utils.paths.scan_folder_health as health

    for i in range(health._PROBE_OPEN_LIMIT * 4):
        (tmp_path / f"model{i}").mkdir()

    opened: list[str] = []
    real_scandir = os.scandir

    def counting_scandir(path, *args, **kwargs):
        opened.append(str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", counting_scandir)
    note_scan_folder_scanned(str(tmp_path), found = True)
    assert scan_folder_status(str(tmp_path)) == STATUS_OK
    assert len(opened) <= health._PROBE_OPEN_LIMIT


@requires_posix_permissions
def test_a_folder_with_one_denied_model_says_so(tmp_path: Path):
    """Partial denial hides models without the list looking wrong.

    The readable model makes the scan succeed, so nothing raises and nothing is
    empty, yet the denied one is silently absent.
    """
    good = tmp_path / "good"
    good.mkdir()
    (good / "model.gguf").write_bytes(b"stub")
    bad = tmp_path / "bad"
    bad.mkdir()
    bad.chmod(0o000)
    try:
        note_scan_folder_scanned(str(tmp_path), found = True)
        assert scan_folder_status(str(tmp_path)) == STATUS_PARTIAL
    finally:
        bad.chmod(stat.S_IRWXU)


@requires_posix_permissions
def test_a_partial_folder_is_not_reported_as_unreadable(tmp_path: Path):
    """The folder works, so the copy must not claim it cannot be read."""
    bad = tmp_path / "bad"
    bad.mkdir()
    bad.chmod(0o000)
    try:
        note_scan_folder_scanned(str(tmp_path), found = True)
        assert scan_folder_status(str(tmp_path)) == STATUS_PARTIAL

        note_scan_folder_scanned(str(tmp_path), found = False)
        assert scan_folder_status(str(tmp_path)) == STATUS_PERMISSION_DENIED
    finally:
        bad.chmod(stat.S_IRWXU)


@requires_posix_permissions
def test_the_recheck_keeps_a_partial_folder_partial(tmp_path: Path):
    bad = tmp_path / "bad"
    bad.mkdir()
    bad.chmod(0o000)
    rows = [{"id": 1, "path": str(tmp_path), "created_at": "2026-01-01"}]
    try:
        note_scan_folder_scanned(str(tmp_path), found = True)
        refresh_failed_scan_folders(rows)
        assert annotate_scan_folders(rows)[0]["status"] == STATUS_PARTIAL
    finally:
        bad.chmod(stat.S_IRWXU)


@requires_posix_permissions
def test_the_real_scan_flags_a_folder_with_one_denied_model(tmp_path: Path):
    from routes.models import collect_local_models

    folder = tmp_path / "models"
    good = folder / "good"
    good.mkdir(parents = True)
    (good / "model.gguf").write_bytes(b"stub")
    (good / "config.json").write_text("{}", encoding = "utf-8")
    bad = folder / "bad"
    bad.mkdir()
    (bad / "model.gguf").write_bytes(b"stub")
    bad.chmod(0o000)
    rows = [{"id": 1, "path": str(folder), "created_at": "2026-01-01"}]
    try:
        found = collect_local_models(tmp_path / "root", custom_folders = list(rows))
        assert [m for m in found if m.source == "custom"], "the readable model still lists"
        assert annotate_scan_folders(rows)[0]["status"] == STATUS_PARTIAL
    finally:
        bad.chmod(stat.S_IRWXU)


def test_a_healthy_folder_stays_ok_when_models_were_found(tmp_path: Path):
    (tmp_path / "model.gguf").write_bytes(b"stub")
    note_scan_folder_scanned(str(tmp_path), found = True)
    assert scan_folder_status(str(tmp_path)) == STATUS_OK


def test_a_folder_that_disappeared_reports_missing(tmp_path: Path):
    # Unmounted drive or renamed folder: the scanners return nothing, same as empty.
    note_scan_folder_scanned(str(tmp_path / "gone"), found = False)
    assert scan_folder_status(str(tmp_path / "gone")) == STATUS_MISSING


def test_an_empty_folder_that_is_still_there_stays_ok(tmp_path: Path):
    note_scan_folder_scanned(str(tmp_path), found = False)
    assert scan_folder_status(str(tmp_path)) == STATUS_OK


@requires_posix_permissions
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


@pytest.mark.parametrize(
    "winerror, expected",
    [
        (5, STATUS_PERMISSION_DENIED),  # ERROR_ACCESS_DENIED
        (65, STATUS_PERMISSION_DENIED),  # ERROR_NETWORK_ACCESS_DENIED
        (21, STATUS_MISSING),  # ERROR_NOT_READY: nothing in the card reader
        (53, STATUS_MISSING),  # ERROR_BAD_NETPATH
        (267, STATUS_MISSING),  # ERROR_DIRECTORY
        (23, STATUS_UNREADABLE),  # ERROR_CRC: the drive is failing
        (31, STATUS_UNREADABLE),  # ERROR_GEN_FAILURE
    ],
)
def test_windows_errors_classify_from_the_native_code(winerror, expected):
    """Windows errno is a lossy translation, so read the native code first.

    CPython's PC/errmap.h folds 27 distinct winerrors onto EACCES, only two of
    which are access denials. Going by errno alone tells a user whose drive is
    unplugged to fix the folder's permissions.
    """
    error = OSError(errno.EACCES, "simulated")
    error.winerror = winerror
    assert classify_scan_error(error) == expected


def test_a_posix_error_is_unaffected_by_the_windows_branch():
    assert classify_scan_error(PermissionError(errno.EACCES, "denied")) == (
        STATUS_PERMISSION_DENIED
    )
    assert classify_scan_error(FileNotFoundError(errno.ENOENT, "gone")) == STATUS_MISSING


def test_a_model_deleted_mid_scan_does_not_condemn_the_folder(tmp_path: Path):
    """A child in the listing and gone by the time it is opened proves nothing.

    Downloads create and rename temp directories inside a scan folder constantly,
    so treating a vanished child as the folder's own status told a user who was
    merely downloading a model that some of their models could not be read.
    """
    folder = tmp_path / "models"
    keep = folder / "keep"
    keep.mkdir(parents = True)
    (keep / "model.gguf").write_bytes(b"stub")
    doomed = folder / "downloading.tmp"
    doomed.mkdir()

    real_scandir = os.scandir

    class _Listed:
        """A finished listing, so the entry survives the directory going away."""

        def __init__(self, entries):
            self._entries = iter(entries)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return self._entries

        def __next__(self):
            return next(self._entries)

    def _vanishing(path = "."):
        # Read the listing to completion, then delete the temp dir. The entry is
        # still in what we hand back, exactly as when a download renames it away
        # between the folder being listed and the child being opened.
        if str(path) == str(folder) and doomed.exists():
            with real_scandir(path) as entries:
                listed = list(entries)
            doomed.rmdir()
            return _Listed(listed)
        return real_scandir(path)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(os, "scandir", _vanishing)
    try:
        note_scan_folder_scanned(str(folder), found = True)
    finally:
        monkey.undo()

    assert scan_folder_status(str(folder)) == STATUS_OK


def test_the_folder_itself_disappearing_still_reports_missing(tmp_path: Path):
    """The guard above must not swallow the case the feature exists for."""
    note_scan_folder_scanned(str(tmp_path / "never-existed"), found = False)
    assert scan_folder_status(str(tmp_path / "never-existed")) == STATUS_MISSING


def test_the_hub_scan_probes_off_the_event_loop(tmp_path: Path):
    """The probe opens directories, so it cannot run on the loop.

    Every other filesystem step in ``_collect_models_from_default_sources`` is
    already wrapped in ``asyncio.to_thread``. This one opens up to 64 directories
    per registered folder, and on a stalled network mount ``scandir`` sits in the
    kernel with nothing to yield to, so the whole Unsloth server stops answering.
    """
    import asyncio as _asyncio

    import hub.services.models.local_inventory as inventory

    folder = tmp_path / "models"
    (folder / "some-model").mkdir(parents = True)
    (folder / "some-model" / "model.gguf").write_bytes(b"stub")
    rows = [{"id": 1, "path": str(folder), "created_at": "2026-01-01"}]

    on_loop: list[bool] = []

    def _spy(path, *, found):
        try:
            _asyncio.get_running_loop()
        except RuntimeError:
            on_loop.append(False)
        else:
            on_loop.append(True)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(inventory, "note_scan_folder_scanned", _spy)
    try:
        _asyncio.run(
            inventory._collect_models_from_default_sources(
                tmp_path,
                tmp_path / "hf",
                tmp_path / "legacy",
                tmp_path / "default",
                [],
                [],
                [],
                list(rows),
            )
        )
    finally:
        monkey.undo()

    assert on_loop == [False], "the folder probe ran on the event loop"


@requires_posix_permissions
def test_the_hub_scan_records_a_folder_it_cannot_read(tmp_path: Path):
    """The Hub inventory has its own custom-folder loop, and it feeds the dialog.

    /api/hub/models/local goes through here, not through collect_local_models,
    so a status the dialog can show has to be recorded on this path too.
    """
    import asyncio as _asyncio

    from hub.services.models.local_inventory import _collect_models_from_default_sources

    denied = tmp_path / "denied"
    denied.mkdir()
    (denied / "model.gguf").write_bytes(b"stub")
    denied.chmod(0o000)
    rows = [{"id": 1, "path": str(denied), "created_at": "2026-01-01"}]

    def scan():
        return _asyncio.run(
            _collect_models_from_default_sources(
                tmp_path,
                tmp_path / "hf",
                tmp_path / "legacy",
                tmp_path / "default",
                [],
                [],
                [],
                list(rows),
            )
        )

    try:
        scan()
        assert annotate_scan_folders(rows)[0]["status"] == STATUS_PERMISSION_DENIED

        denied.chmod(stat.S_IRWXU)
        scan()
        assert annotate_scan_folders(rows)[0]["status"] == STATUS_OK
    finally:
        denied.chmod(stat.S_IRWXU)


@requires_posix_permissions
def test_a_root_that_lists_but_hides_every_model_is_not_ok(tmp_path: Path):
    """Mixed permissions: the root reads fine, the model under it does not.

    The scanners skip an unreadable child silently, so this arrives as the same
    empty list as a genuinely empty folder.
    """
    denied_child = tmp_path / "modelA"
    denied_child.mkdir()
    (denied_child / "model.gguf").write_bytes(b"stub")
    denied_child.chmod(0o000)
    try:
        assert probe_status(str(tmp_path)) == STATUS_OK
        assert probe_status(str(tmp_path), children = True) == STATUS_PERMISSION_DENIED
    finally:
        denied_child.chmod(stat.S_IRWXU)


@requires_posix_permissions
def test_the_real_scan_flags_a_folder_whose_models_are_all_denied(tmp_path: Path):
    from routes.models import collect_local_models

    folder = tmp_path / "models"
    denied_child = folder / "modelA"
    denied_child.mkdir(parents = True)
    (denied_child / "model.gguf").write_bytes(b"stub")
    denied_child.chmod(0o000)
    rows = [{"id": 1, "path": str(folder), "created_at": "2026-01-01"}]
    try:
        collect_local_models(tmp_path / "root", custom_folders = list(rows))
        assert annotate_scan_folders(rows)[0]["status"] == STATUS_PERMISSION_DENIED
    finally:
        denied_child.chmod(stat.S_IRWXU)


@requires_posix_permissions
def test_a_denied_model_under_a_readable_publisher_is_not_ok(tmp_path: Path):
    """The LM Studio shape: <root>/<publisher>/<model>, denied at the model.

    Both levels above it list fine, so a probe that stops at the first level
    reports ok while the scan returns nothing.
    """
    model = tmp_path / "publisher" / "modelA"
    model.mkdir(parents = True)
    (model / "model.gguf").write_bytes(b"stub")
    model.chmod(0o000)
    try:
        assert probe_status(str(tmp_path), children = True) == STATUS_PERMISSION_DENIED
    finally:
        model.chmod(stat.S_IRWXU)


@requires_posix_permissions
def test_the_real_scan_flags_a_denied_model_two_levels_down(tmp_path: Path):
    from routes.models import collect_local_models

    folder = tmp_path / "models"
    model = folder / "publisher" / "modelA"
    model.mkdir(parents = True)
    (model / "model.gguf").write_bytes(b"stub")
    model.chmod(0o000)
    rows = [{"id": 1, "path": str(folder), "created_at": "2026-01-01"}]
    try:
        collect_local_models(tmp_path / "root", custom_folders = list(rows))
        assert annotate_scan_folders(rows)[0]["status"] == STATUS_PERMISSION_DENIED
    finally:
        model.chmod(stat.S_IRWXU)


@requires_posix_permissions
def test_the_deep_probe_finds_the_denial_in_a_few_opens(tmp_path: Path, monkeypatch):
    """Depth first, so the whole-mount-denied case costs three opens, not a walk."""
    denied = tmp_path / "publisher" / "modelA"
    denied.mkdir(parents = True)
    denied.chmod(0o000)

    opened: list[str] = []
    real_scandir = os.scandir

    def counting_scandir(path, *args, **kwargs):
        opened.append(str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", counting_scandir)
    try:
        assert probe_status(str(tmp_path), children = True) == STATUS_PERMISSION_DENIED
        # Root, publisher, model.
        assert len(opened) == 3
    finally:
        denied.chmod(stat.S_IRWXU)


def test_this_file_can_be_collected_without_geteuid(monkeypatch):
    """Windows has no os.geteuid, and a skipif condition runs at import time."""
    source = Path(__file__).read_text(encoding = "utf-8")
    monkeypatch.delattr(os, "geteuid", raising = False)
    monkeypatch.setattr(os, "name", "nt")
    namespace: dict = {"__name__": "windows_collection_probe"}
    # The module body is what pytest evaluates while collecting.
    exec(compile(source, __file__, "exec"), namespace)


def test_the_child_probe_is_bounded(tmp_path: Path, monkeypatch):
    """A folder with many readable children must not turn into a walk."""
    import utils.paths.scan_folder_health as health

    for i in range(health._PROBE_OPEN_LIMIT * 3):
        (tmp_path / f"child{i}").mkdir()

    opened: list[str] = []
    real_scandir = os.scandir

    def counting_scandir(path, *args, **kwargs):
        opened.append(str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", counting_scandir)
    # Too wide to finish inside the budget, so the answer is "did not see
    # everything", not "healthy". Only an exhaustive probe may report ok.
    assert probe_status(str(tmp_path), children = True) == STATUS_UNKNOWN
    # One shared budget across the whole walk, however deep or wide the tree.
    assert len(opened) <= health._PROBE_OPEN_LIMIT


def test_reopening_the_dialog_clears_a_folder_the_user_fixed(tmp_path: Path):
    """The row tells the user to fix it and come back, so coming back must work.

    Nothing rescans between inventory scans, so the folder list has to recheck.
    """
    (tmp_path / "model.gguf").write_bytes(b"stub")
    rows = [{"id": 1, "path": str(tmp_path), "created_at": "2026-01-01"}]
    record_scan_failure(str(tmp_path), PermissionError(errno.EACCES, "Permission denied"))
    assert annotate_scan_folders(rows)[0]["status"] == STATUS_PERMISSION_DENIED

    refresh_failed_scan_folders(rows)
    assert annotate_scan_folders(rows)[0]["status"] == STATUS_OK


@requires_posix_permissions
def test_reopening_the_dialog_keeps_a_folder_that_is_still_denied(tmp_path: Path):
    denied = tmp_path / "denied"
    denied.mkdir()
    denied.chmod(0o000)
    rows = [{"id": 1, "path": str(denied), "created_at": "2026-01-01"}]
    record_scan_failure(str(denied), PermissionError(errno.EACCES, "Permission denied"))
    try:
        refresh_failed_scan_folders(rows)
        assert annotate_scan_folders(rows)[0]["status"] == STATUS_PERMISSION_DENIED
    finally:
        denied.chmod(stat.S_IRWXU)


def test_the_recheck_updates_a_status_that_changed(tmp_path: Path):
    gone = tmp_path / "gone"
    rows = [{"id": 1, "path": str(gone), "created_at": "2026-01-01"}]
    record_scan_failure(str(gone), PermissionError(errno.EACCES, "Permission denied"))
    refresh_failed_scan_folders(rows)
    assert annotate_scan_folders(rows)[0]["status"] == STATUS_MISSING


def test_the_recheck_leaves_healthy_folders_alone(monkeypatch):
    """No folder is marked bad, so the folder list must not open anything."""

    def _boom(*args, **kwargs):
        raise AssertionError("the folder list touched the filesystem")

    for name in ("scandir", "listdir", "stat", "access"):
        monkeypatch.setattr(os, name, _boom)

    rows = [{"id": 1, "path": "/models/a", "created_at": "2026-01-01"}]
    refresh_failed_scan_folders(rows)
    assert annotate_scan_folders(rows)[0]["status"] == STATUS_OK


def test_the_recheck_only_opens_the_folder_that_failed(tmp_path: Path, monkeypatch):
    good = tmp_path / "good"
    good.mkdir()
    bad = tmp_path / "bad"
    bad.mkdir()
    record_scan_failure(str(bad), PermissionError(errno.EACCES, "Permission denied"))

    opened: list[str] = []
    real_scandir = os.scandir

    def counting_scandir(path, *args, **kwargs):
        opened.append(str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", counting_scandir)
    refresh_failed_scan_folders(
        [
            {"id": 1, "path": str(good), "created_at": "2026-01-01"},
            {"id": 2, "path": str(bad), "created_at": "2026-01-02"},
        ]
    )
    assert opened == [str(bad)]


@requires_posix_permissions
def test_an_exhausted_budget_does_not_clear_a_known_failure(tmp_path: Path):
    """Running out of budget proves nothing, so it must not report healthy.

    The folder is too wide to finish probing, and the denied directory sits past
    the cutoff, so a probe that treated exhaustion as ok would drop the warning.
    """
    import utils.paths.scan_folder_health as health

    for i in range(health._PROBE_OPEN_LIMIT * 3):
        (tmp_path / f"model{i:03d}").mkdir()
    # Deterministically past the budget: pick by real listing order, not by name.
    order = [entry.name for entry in os.scandir(tmp_path)]
    denied = tmp_path / order[-1]
    denied.chmod(0o000)
    try:
        record_scan_failure(str(tmp_path), PermissionError(errno.EACCES, "Permission denied"))
        health._failed[str(tmp_path)] = (STATUS_PARTIAL, str(denied))

        note_scan_folder_scanned(str(tmp_path), found = True)
        assert scan_folder_status(str(tmp_path)) == STATUS_PARTIAL
    finally:
        denied.chmod(stat.S_IRWXU)


@requires_posix_permissions
def test_a_wide_folder_still_clears_once_it_is_fixed(tmp_path: Path):
    """The flip side: recovery cannot depend on the probe reaching the tail."""
    import utils.paths.scan_folder_health as health

    for i in range(health._PROBE_OPEN_LIMIT * 3):
        (tmp_path / f"model{i:03d}").mkdir()
    order = [entry.name for entry in os.scandir(tmp_path)]
    fixed = tmp_path / order[-1]
    # Recorded as the cause, but readable again by the time the scan runs.
    health._failed[str(tmp_path)] = (STATUS_PARTIAL, str(fixed))

    note_scan_folder_scanned(str(tmp_path), found = True)
    assert scan_folder_status(str(tmp_path)) == STATUS_OK


@requires_posix_permissions
def test_a_partial_folder_that_disappears_reports_missing(tmp_path: Path):
    """Deleting the folder outranks "some models could not be read"."""
    import shutil

    folder = tmp_path / "models"
    bad = folder / "bad"
    bad.mkdir(parents = True)
    bad.chmod(0o000)
    rows = [{"id": 1, "path": str(folder), "created_at": "2026-01-01"}]

    note_scan_folder_scanned(str(folder), found = True)
    assert annotate_scan_folders(rows)[0]["status"] == STATUS_PARTIAL

    bad.chmod(stat.S_IRWXU)
    shutil.rmtree(folder)
    refresh_failed_scan_folders(rows)
    assert annotate_scan_folders(rows)[0]["status"] == STATUS_MISSING


def test_the_internal_unknown_status_never_reaches_the_api(tmp_path: Path):
    """STATUS_UNKNOWN is a probe result, not something the UI can render."""
    import utils.paths.scan_folder_health as health

    for i in range(health._PROBE_OPEN_LIMIT * 3):
        (tmp_path / f"model{i:03d}").mkdir()
    rows = [{"id": 1, "path": str(tmp_path), "created_at": "2026-01-01"}]

    note_scan_folder_scanned(str(tmp_path), found = True)
    refresh_failed_scan_folders(rows)
    assert annotate_scan_folders(rows)[0]["status"] != STATUS_UNKNOWN
    assert STATUS_UNKNOWN not in {status for status, _cause in health._failed.values()}


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


@requires_posix_permissions
def test_a_denied_hf_snapshot_commit_directory_is_not_ok(tmp_path: Path):
    """The HF cache keeps the files one level below <publisher>/<model>.

    <root>/models--org--name/snapshots/<commit>/ is where the weights are; refuse
    it and every directory above it still lists, so a depth that stops at
    ``snapshots`` calls the folder healthy while the model is gone from the list.
    """
    repo = tmp_path / "models--org--model"
    (repo / "blobs").mkdir(parents = True)
    (repo / "blobs" / "deadbeef").write_bytes(b"stub")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40, encoding = "utf-8")
    commit = repo / "snapshots" / ("a" * 40)
    commit.mkdir(parents = True)
    (commit / "config.json").write_text("{}", encoding = "utf-8")
    commit.chmod(0o000)
    try:
        status, cause = probe_folder(str(tmp_path), children = True)
        assert status == STATUS_PERMISSION_DENIED
        assert cause == str(commit)

        note_scan_folder_scanned(str(tmp_path), found = False)
        assert scan_folder_status(str(tmp_path)) == STATUS_PERMISSION_DENIED
    finally:
        commit.chmod(stat.S_IRWXU)


@requires_posix_permissions
def test_the_snapshot_level_costs_nothing_on_a_plain_folder(tmp_path: Path, monkeypatch):
    """Only a directory named ``snapshots`` buys the extra level.

    A diffusers pipeline keeps component directories under each model, so raising
    the depth for everything would spend the budget on <publisher>/<model>/<part>.
    """
    part = tmp_path / "publisher" / "model" / "transformer"
    part.mkdir(parents = True)
    (part / "denied").mkdir()
    (part / "denied").chmod(0o000)

    opened: list[str] = []
    real_scandir = os.scandir

    def counting_scandir(path, *args, **kwargs):
        opened.append(str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", counting_scandir)
    try:
        # Root, publisher, model. The component level is still out of reach.
        assert probe_status(str(tmp_path), children = True) == STATUS_OK
        assert len(opened) == 3
    finally:
        (part / "denied").chmod(stat.S_IRWXU)


@requires_posix_permissions
def test_a_partial_folder_whose_root_is_denied_stops_saying_partial(tmp_path: Path):
    """Once the registered root refuses, nothing in it can be scanned.

    Telling the user only some models could not be read sends them looking for
    the one bad model in a folder none of which is readable.
    """
    bad = tmp_path / "bad"
    bad.mkdir()
    bad.chmod(0o000)
    rows = [{"id": 1, "path": str(tmp_path), "created_at": "2026-01-01"}]
    try:
        note_scan_folder_scanned(str(tmp_path), found = True)
        assert scan_folder_status(str(tmp_path)) == STATUS_PARTIAL

        tmp_path.chmod(0o000)
        try:
            refresh_failed_scan_folders(rows)
            assert annotate_scan_folders(rows)[0]["status"] == STATUS_PERMISSION_DENIED
        finally:
            tmp_path.chmod(stat.S_IRWXU)
    finally:
        bad.chmod(stat.S_IRWXU)
