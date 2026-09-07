# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox import package_snapshot as packages
from core.inference.windows_sandbox.profiles import WindowsRuntimeError


@pytest.fixture
def selected(tmp_path):
    root = tmp_path / "site-packages"
    package = root / "example"
    package.mkdir(parents = True)
    (package / "__init__.py").write_text("raise RuntimeError('must not import')", encoding = "utf-8")
    (package / "native.pyd").write_bytes(b"native payload bytes")
    (package / "table.dat").write_bytes(b"data payload bytes")
    return root


def test_preserves_package_data_and_native_files_without_imports_or_pth(selected, tmp_path):
    marker = tmp_path / "pth-executed"
    malicious = f"import pathlib; pathlib.Path({str(marker)!r}).write_text('executed')\n"
    (selected / "malicious.pth").write_text(malicious, encoding = "utf-8")
    result = packages.inventory_package_files((str(selected),))
    assert {item.relative_path for item in result} == {
        "packages/0/example/__init__.py",
        "packages/0/example/native.pyd",
        "packages/0/example/table.dat",
        "packages/0/malicious.pth",
    }
    assert not marker.exists()
    for item in result:
        assert item.source.sha256 == hashlib.sha256(Path(item.source.path).read_bytes()).hexdigest()
    assert not hasattr(result, "ordered_loads")


def test_keeps_selected_root_order_and_never_follows_pth_paths(selected, tmp_path):
    other = tmp_path / "other-packages"
    other.mkdir()
    (other / "data.bin").write_bytes(b"second root")
    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.txt").write_text("not selected")
    (selected / "external.pth").write_text(str(external))
    result = packages.inventory_package_files((str(other), str(selected)))
    assert any(item.relative_path == "packages/0/data.bin" for item in result)
    assert any(item.relative_path == "packages/1/external.pth" for item in result)
    assert not any("secret" in item.relative_path for item in result)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("entries", 1, "entry"),
        ("files", 1, "count"),
        ("bytes", 1, "byte"),
        ("file_bytes", 1, "byte"),
        ("depth", 1, "depth"),
    ],
)
def test_explicit_budgets_fail_closed(selected, field, value, match):
    (selected / "example" / "nested").mkdir()
    (selected / "example" / "nested" / "data").write_bytes(b"nested")
    with pytest.raises(WindowsRuntimeError, match = match):
        packages.inventory_package_files(
            (str(selected),), bounds = replace(packages.PackageSnapshotBounds(), **{field: value})
        )


def test_root_aliases_and_overlap_rejected(selected):
    for roots in ((str(selected), str(selected)), (str(selected), str(selected / "example"))):
        with pytest.raises(WindowsRuntimeError, match = "overlap"):
            packages.inventory_package_files(roots)


def test_hardlinked_package_file_rejected(selected, tmp_path):
    os.link(selected / "example" / "table.dat", tmp_path / "hardlink.dat")
    with pytest.raises(WindowsRuntimeError, match = "single-link"):
        packages.inventory_package_files((str(selected),))


@pytest.mark.parametrize("directory", [False, True])
def test_package_symlinks_never_traversed(selected, tmp_path, directory):
    target = tmp_path / "external"
    if directory:
        target.mkdir()
    else:
        target.write_bytes(b"outside")
    try:
        (selected / "linked").symlink_to(target, target_is_directory = directory)
    except OSError as error:
        pytest.skip(f"Symlink creation unavailable: {error}")
    with pytest.raises(WindowsRuntimeError, match = "reparse"):
        packages.inventory_package_files((str(selected),))


def test_streaming_hash_never_requests_whole_file(selected, monkeypatch):
    source = selected / "large.dat"
    source.write_bytes(b"x" * 4096)
    original = Path.open
    requests = []

    class Observed:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            requests.append(size)
            return self.stream.read(size)

    monkeypatch.setattr(packages, "CHUNK_BYTES", 127)
    monkeypatch.setattr(
        Path, "open", lambda path, *args, **kwargs: Observed(original(path, *args, **kwargs))
    )
    packages.inventory_package_files((str(selected),))
    assert len(requests) > 32
    assert all(0 < size <= 127 for size in requests)


def test_deadline_and_invalid_budget(selected, monkeypatch):
    clock = iter((0, 121))
    monkeypatch.setattr(packages.time, "monotonic", lambda: next(clock))
    with pytest.raises(WindowsRuntimeError, match = "deadline"):
        packages.inventory_package_files((str(selected),))
    with pytest.raises(WindowsRuntimeError, match = "bounds"):
        packages.inventory_package_files(
            (str(selected),), bounds = replace(packages.PackageSnapshotBounds(), files = True)
        )


def test_empty_selection_is_valid():
    assert packages.inventory_package_files(()) == ()


def test_derived_cache_changes_do_not_change_source_inventory(selected):
    cache = selected / "example" / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-312.pyc").write_bytes(b"first derived cache")
    (selected / "sourceless.pyc").write_bytes(b"keep ordinary sourceless module")
    first = packages.inventory_package_files((str(selected),))
    (cache / "module.cpython-312.pyc").write_bytes(b"different derived cache")
    (cache / "new-cache.pyc").write_bytes(b"new derived cache")
    second = packages.inventory_package_files((str(selected),))
    assert first == second
    assert any(item.relative_path.endswith("sourceless.pyc") for item in first)
    assert not any("__pycache__" in item.relative_path for item in first)


def test_file_changed_during_hash_is_rejected(selected):
    path = selected / "example" / "table.dat"
    mutated = False

    def mutate_once():
        nonlocal mutated
        if not mutated:
            mutated = True
            path.write_bytes(b"different-size replacement while read is open")

    with pytest.raises(WindowsRuntimeError, match = "changed"):
        packages._hash_regular_file(path, limit = 1000, check_deadline = mutate_once)
