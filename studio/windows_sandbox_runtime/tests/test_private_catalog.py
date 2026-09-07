# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Real private-hive ownership tests; not full LPAC/Windows qualification."""

from dataclasses import replace
import ctypes
from ctypes import wintypes as W
import os
from pathlib import Path
import sys
import subprocess

import pytest
from test_launch import installed_runtime, runtime_wheel, run_harness, LAUNCH

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox import private_catalog as catalog
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

pytestmark = pytest.mark.skipif(os.name != "nt", reason = "Windows private registry hive")


def test_catalog_receipt_is_invocation_scoped_data_without_filesystem_reopens(
    tmp_path, monkeypatch
):
    temporary = tmp_path / "Temp"
    value = {
        "directory": str(temporary / ("private-winsock-" + "a" * 32)),
        "identity": "11" * 32,
        "binding_digest": "22" * 32,
    }
    monkeypatch.setattr(
        catalog, "checked_path", lambda *_: pytest.fail("Broker reopened catalog files")
    )
    receipt = catalog.adopt_catalog(value, temporary)
    assert receipt.hive_path == Path(value["directory"]) / "winsock.hiv"
    assert receipt.identity == bytes.fromhex(value["identity"])
    assert receipt.binding_digest == value["binding_digest"]
    for changes in (
        {"directory": str(tmp_path / ("private-winsock-" + "a" * 32))},
        {"directory": str(temporary / "unowned")},
        {"identity": "11"},
        {"binding_digest": "G" * 64},
        {"extra": "untrusted"},
    ):
        with pytest.raises(WindowsRuntimeError):
            catalog.adopt_catalog({**value, **changes}, temporary)


@pytest.fixture
def snapshot():
    return catalog._snapshot(catalog.CatalogBounds())


def test_real_hive_query_root_and_cleanup(tmp_path, snapshot):
    import winreg

    before, _ = snapshot
    owner = catalog.prepare_private_catalog(tmp_path)
    directory = owner.directory
    try:
        assert owner.metadata["qualified"] is False
        assert owner.metadata["private_subkeys_read_only"] is False
        assert owner.metadata["root_access"] == 1
        assert len(owner.metadata["digest"]) == 64
        assert owner.metadata["providers"]
        with pytest.raises(OSError):
            winreg.SetValueEx(owner.query_root_handle, "test", 0, winreg.REG_DWORD, 1)
        with winreg.OpenKey(owner.query_root_handle, catalog._ROOTS[1], 0, winreg.KEY_READ) as key:
            assert winreg.QueryInfoKey(key)[0] > 0
        # A same-process duplicate models the parent's explicit query-only handoff.
        kernel = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel.DuplicateHandle.argtypes = [
            W.HANDLE,
            W.HANDLE,
            W.HANDLE,
            ctypes.POINTER(W.HANDLE),
            W.DWORD,
            W.BOOL,
            W.DWORD,
        ]
        kernel.DuplicateHandle.restype = W.BOOL
        duplicate = W.HANDLE()
        assert kernel.DuplicateHandle(
            W.HANDLE(-1),
            W.HANDLE(owner.query_root_handle),
            W.HANDLE(-1),
            ctypes.byref(duplicate),
            1,
            False,
            0,
        )
        try:
            assert winreg.QueryInfoKey(duplicate.value)[0] == 1
            with pytest.raises(OSError):
                winreg.SetValueEx(duplicate.value, "test", 0, winreg.REG_DWORD, 1)
        finally:
            winreg.CloseKey(duplicate.value)
    finally:
        owner.close()
    owner.close()
    assert not directory.exists()
    with pytest.raises(WindowsRuntimeError):
        owner.query_root_handle
    after, _ = catalog._snapshot(catalog.CatalogBounds())
    assert before == after  # Host trees were only read.


@pytest.mark.parametrize(
    "mutation",
    ["external", "unknown_system", "packed_size", "missing", "selector", "callout", "transport"],
)
def test_provider_refusal_uses_snapshot_bytes(snapshot, mutation):
    records, _ = snapshot
    changed = list(records)
    for index, value in enumerate(changed):
        if (
            mutation in {"external", "unknown_system", "packed_size", "missing"}
            and value.name == "PackedCatalogItem"
        ):
            if mutation == "missing":
                del changed[index]
            elif mutation == "packed_size":
                changed[index] = replace(value, data = value.data[:-1])
            else:
                path = (
                    b"C:\\outside\\mswsock.dll"
                    if mutation == "external"
                    else b"%SystemRoot%\\system32\\unknown.dll"
                )
                changed[index] = replace(value, data = path.ljust(260, b"\0") + value.data[260:])
            break
        if mutation == "selector" and value.name == "Current_Protocol_Catalog":
            changed[index] = replace(value, data = "Protocol_Catalog10")
            break
        if mutation == "callout" and value.name == "AutodialDLL":
            changed[index] = replace(value, data = r"C:\outside\rasadhlp.dll")
            break
        if mutation == "transport" and value.name == "Transports":
            changed[index] = replace(value, data = ("unreviewed",))
            break
    else:
        pytest.fail("fixture did not contain the expected catalog record")
    from core.inference.windows_sandbox.native_plan import windows_loader_policy

    with pytest.raises(WindowsRuntimeError):
        catalog._provider_names(changed, windows_loader_policy().directory)


@pytest.mark.parametrize(
    "bounds",
    [
        catalog.CatalogBounds(keys = 1),
        catalog.CatalogBounds(values = 1),
        catalog.CatalogBounds(bytes = 1),
        catalog.CatalogBounds(depth = 0),
    ],
)
def test_budget_refusal_leaves_no_hive(tmp_path, bounds):
    with pytest.raises(WindowsRuntimeError):
        catalog.prepare_private_catalog(tmp_path, bounds = bounds)
    assert not list(tmp_path.iterdir())


def test_copy_failure_closes_hive_and_removes_files(tmp_path, monkeypatch):
    import winreg

    def fail(*_args):
        raise OSError("injected private write failure")

    monkeypatch.setattr(winreg, "SetValueEx", fail)
    with pytest.raises(OSError, match = "injected"):
        catalog.prepare_private_catalog(tmp_path)
    assert not list(tmp_path.iterdir())


def test_cleanup_preserves_unexpected_files_and_can_retry(tmp_path):
    owner = catalog.prepare_private_catalog(tmp_path)
    sentinel = owner.directory / "unrelated.txt"
    sentinel.write_text("preserve")
    with pytest.raises(WindowsRuntimeError, match = "Unexpected"):
        owner.close()
    assert sentinel.read_text() == "preserve"
    sentinel.unlink()
    owner.close()
    assert not owner.directory.exists()


@pytest.mark.parametrize("wrong_identity", [False, True])
def test_child_opens_exact_hive_with_public_api(tmp_path, wrong_identity):
    """Real process control; package-SID LPAC access is a separate launch test."""
    with catalog.prepare_private_catalog(tmp_path) as owner:
        code = """
import ctypes, sys, winreg
from ctypes import wintypes as W
api = ctypes.WinDLL('advapi32')
api.RegLoadAppKeyW.argtypes = [W.LPCWSTR, ctypes.POINTER(W.HKEY), W.DWORD, W.DWORD, W.DWORD]
api.RegLoadAppKeyW.restype = W.LONG
root = W.HKEY()
status = api.RegLoadAppKeyW(sys.argv[1], ctypes.byref(root), 1, 0, 0)
assert status == 0, status
try:
    marker, kind = winreg.QueryValueEx(root.value, 'UnslothCatalogIdentity')
    if kind != winreg.REG_BINARY or marker != bytes.fromhex(sys.argv[2]):
        sys.exit(23)
    assert winreg.QueryInfoKey(root.value)[0] == 1
    try:
        winreg.SetValueEx(root.value, 'test', 0, winreg.REG_DWORD, 1)
    except PermissionError:
        print('PRIVATE_QUERY_ROOT_OK')
    else:
        raise AssertionError('root writable')
finally:
    winreg.CloseKey(root.value)
"""
        marker = b"\0" * 32 if wrong_identity else owner.identity
        child = subprocess.run(
            [sys.executable, "-I", "-c", code, str(owner.hive_path), marker.hex()],
            capture_output = True,
            text = True,
            timeout = 15,
        )
        assert child.returncode == (23 if wrong_identity else 0), child.stderr
        assert child.stdout.strip() == ("" if wrong_identity else "PRIVATE_QUERY_ROOT_OK")


def test_package_sid_validation_does_not_create_hive(tmp_path):
    with pytest.raises(WindowsRuntimeError, match = "SID"):
        catalog.prepare_private_catalog(tmp_path, package_sid = "S-1-1-0")
    assert not list(tmp_path.iterdir())


def test_failed_cleanup_retains_owner_for_retry(tmp_path, monkeypatch):
    import winreg

    def fail(*_args):
        raise OSError("injected failure")

    with monkeypatch.context() as patch:
        patch.setattr(winreg, "SetValueEx", fail)
        patch.setattr(catalog.PrivateCatalog, "close", fail)
        with pytest.raises(OSError) as caught:
            catalog.prepare_private_catalog(tmp_path)
    owner = caught.value.catalog_owner
    assert owner.directory.exists()
    owner.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(
    not os.environ.get("UNSLOTH_TEST_RUNTIME_WHEEL"),
    reason = "Explicit built runtime wheel required for LPAC launch control",
)
@pytest.mark.parametrize("read_only", [False, True])
def test_lpac_hive_backing_file_access(installed_runtime, tmp_path, read_only):
    patch = ""
    if read_only:
        patch = """
from core.inference.windows_sandbox.content_files import native_files
files = native_files()
directory = owner.catalog.hive_path.parent
for path in (directory, *directory.iterdir()):
    is_directory = path == directory
    handle = files.open(path, directory=is_directory, write_dac=True)
    try:
        sddl = files.private_sddl
        if is_directory:
            sddl = sddl.replace('(A;;', '(A;OICI;')
        sddl += ('(A;OICI;FRFX;;;' if is_directory else '(A;;FRFX;;;') + owner.identity.sid_string + ')'
        files.set_owned_dacl(handle, sddl)
    finally:
        assert files.kernel.CloseHandle(handle)
"""
    body = (
        "script.write_text(\"print('PRIVATE_CATALOG_LPAC_OK',flush=True)\",encoding='utf-8')\n"
        + LAUNCH
        + patch
        + f"""
try:
    try:
        process = spawn_prepared_launch(prepared, **kwargs)
    except Exception as error:
        assert {read_only!r}, str(error)
        assert 'stage 1' in str(error) and 'WinError 5' in str(error), str(error)
        print('READ_ONLY_HIVE_REFUSED')
    else:
        assert not {read_only!r}, 'Read-only control unexpectedly launched'
        assert process.wait(timeout=15) == 0
        assert process.stdout.read().strip() == 'PRIVATE_CATALOG_LPAC_OK'
        print('LPAC_HIVE_OPEN_OK')
finally:
    prepared.cleanup()
assert not prepared.cleanup_diagnostics, prepared.cleanup_diagnostics
"""
    )
    output = run_harness(installed_runtime, tmp_path, body)
    assert ("READ_ONLY_HIVE_REFUSED" if read_only else "LPAC_HIVE_OPEN_OK") in output


def test_binding_survives_invocation_identity_and_cleanup(tmp_path):
    sid1 = "S-1-15-2-101-102-103-104-105-106-107"
    sid2 = "S-1-15-2-201-202-203-204-205-206-207"
    with catalog.prepare_private_catalog(tmp_path, package_sid = sid1) as first:
        expected = first.binding_digest
        identity = first.identity
        evidence = first.metadata["digest"]
    assert first.binding_digest == expected
    with catalog.prepare_private_catalog(
        tmp_path, package_sid = sid2, expected_binding_digest = expected
    ) as second:
        assert second.binding_digest == expected == second.metadata["binding_digest"]
        assert second.identity != identity
        assert second.metadata["digest"] != evidence
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mutation", ["catalog", "provider", "access"])
def test_binding_rejects_drift_before_hive_creation(tmp_path, monkeypatch, snapshot, mutation):
    import winreg

    with catalog.prepare_private_catalog(tmp_path) as qualified:
        expected = qualified.binding_digest
    kwargs = {}
    if mutation == "catalog":
        records, counts = snapshot
        records = list(records)
        for index, value in enumerate(records):
            if value.kind == winreg.REG_DWORD:
                records[index] = replace(value, data = value.data ^ 1)
                break
        else:
            pytest.fail("catalog fixture lacks a DWORD control")
        monkeypatch.setattr(catalog, "_snapshot", lambda _bounds: (tuple(records), counts))
    elif mutation == "provider":
        original = catalog._system_providers

        def changed(*args):
            providers = original(*args)
            providers[0] = dict(providers[0], sha256 = "0" * 64)
            return providers

        monkeypatch.setattr(catalog, "_system_providers", changed)
    else:
        kwargs["package_sid"] = "S-1-15-2-101-102-103-104-105-106-107"
    with pytest.raises(WindowsRuntimeError, match = "binding changed"):
        catalog.prepare_private_catalog(tmp_path, expected_binding_digest = expected, **kwargs)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("expected", ["", "A" * 64, "0" * 63, 123])
def test_invalid_binding_refused_before_snapshot(tmp_path, monkeypatch, expected):
    def forbidden(_bounds):
        pytest.fail("invalid qualification binding must fail before host snapshot")

    monkeypatch.setattr(catalog, "_snapshot", forbidden)
    with pytest.raises(WindowsRuntimeError, match = "qualification binding"):
        catalog.prepare_private_catalog(tmp_path, expected_binding_digest = expected)
    assert not list(tmp_path.iterdir())


def test_writer_handoff_close_failure_retains_both_roots(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import winreg

    api = catalog._registry_api()
    refusing = SimpleNamespace(RegLoadAppKeyW = api.RegLoadAppKeyW, RegCloseKey = lambda _key: 5)
    monkeypatch.setattr(catalog, "_registry_api", lambda: refusing)
    with pytest.raises(OSError) as failure:
        catalog.prepare_private_catalog(tmp_path)
    owner = failure.value.catalog_owner
    assert owner._root is not None and len(owner._extra_roots) == 1
    # Both owned handles remain valid and retryable after the injected close error.
    assert winreg.QueryInfoKey(owner._root)[0] == 1
    assert winreg.QueryInfoKey(owner._extra_roots[0])[0] == 1
    owner._api = api
    owner.close()
    assert owner._root is None and not owner._extra_roots
    assert not list(tmp_path.iterdir())
