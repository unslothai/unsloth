# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native parser and pin-handoff controls; no activation hook or LPAC claim."""

from dataclasses import replace
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox.activation_manifest import inspect_activation_image
from core.inference.windows_sandbox.activation_plan import ActivationPlan, measure_file_identity


@pytest.fixture
def case(tmp_path):
    driver = os.environ.get("UNSLOTH_ACTIVATION_PLAN_DRIVER")
    if sys.platform != "win32" or not driver:
        pytest.skip("Set UNSLOTH_ACTIVATION_PLAN_DRIVER to the native parser control driver")
    selected = next(
        (p for entry in sys.path if entry for p in (Path(entry) / "PIL").glob("_imaging.*.pyd")),
        None,
    )
    if selected is None:
        pytest.skip("Selected interpreter has no installed Pillow image")
    root = tmp_path / "files"
    root.mkdir()
    home = root / "runtime"
    home.mkdir()
    target = root / selected.name
    shutil.copyfile(selected, target)
    image = inspect_activation_image(target)
    plan = ActivationPlan(
        b"n" * 32, b"p" * 32, b"c" * 32, (image,), (measure_file_identity(image),)
    )
    return Path(driver), home, target, tmp_path / "plan.bin", plan


def run(case, data, *arguments):
    driver, home, _, wire, _ = case
    wire.write_bytes(data)
    return subprocess.run(
        [str(driver), str(wire), str(home), *arguments],
        capture_output = True,
        text = True,
        timeout = 20,
        check = False,
    )


def test_native_accepts_exact_plan_and_holds_pin_at_handoff(case):
    result = run(case, case[4].encode())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "count=1 prepares=1 pin_denials=1" in result.stdout
    assert "diagnostic=11" in result.stdout


def test_native_empty_plan_never_prepares(case):
    result = run(case, replace(case[4], images = (), file_identities = ()).encode())
    assert result.returncode == 0, result.stdout
    assert "count=0 prepares=0 pin_denials=0" in result.stdout


def test_native_opened_handle_query_survives_normalized_access_denial(case):
    result = run(case, case[4].encode(), "normalized-denied")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "normalized_denials=1 opened_queries=1" in result.stdout
    assert "dos_denials=1" in result.stdout
    assert "count=1 prepares=1 pin_denials=1" in result.stdout


@pytest.mark.parametrize("directory", [False, True])
def test_native_reparse_alias_rejected_even_with_matching_image_hash(case, directory):
    alias = case[2].parent / "alias"
    try:
        alias.symlink_to(case[2].parent if directory else case[2], target_is_directory = directory)
    except OSError as error:
        pytest.skip(f"Symlink creation unavailable: {error}")
    target = alias / case[2].name if directory else alias
    original = case[4].images[0]
    aliased = replace(original, file = replace(original.file, path = str(target)))
    result = run(case, replace(case[4], images = (aliased,)).encode())
    assert result.returncode == 1, result.stdout
    assert "prepares=0" in result.stdout
    assert f"diagnostic={8 if directory else 7}" in result.stdout


def test_native_directory_junction_rejected_with_matching_image_hash(case):
    alias = case[2].parent / "junction-alias"
    created = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(case[2].parent)],
        capture_output = True,
        text = True,
        timeout = 10,
        check = False,
    )
    assert created.returncode == 0, created.stdout + created.stderr
    try:
        original = case[4].images[0]
        aliased = replace(original, file = replace(original.file, path = str(alias / case[2].name)))
        result = run(case, replace(case[4], images = (aliased,)).encode())
        assert result.returncode == 1, result.stdout
        assert "prepares=0" in result.stdout
        assert "diagnostic=8" in result.stdout and "error=161 " in result.stdout
    finally:
        # Remove only the junction entry, without traversing its target.
        alias.rmdir()
    assert case[2].is_file()


@pytest.mark.parametrize("offset,value", [(8, 1), (12, 0), (16, 65), (116, 3), (120, 65537)])
def test_native_bad_schema_and_bounds_do_not_prepare(case, offset, value):
    wire = bytearray(case[4].encode())
    struct.pack_into("<I", wire, offset, value)
    result = run(case, wire)
    assert result.returncode == 1
    assert "prepares=0" in result.stdout


@pytest.mark.parametrize("offset", [20, 52, 84, 140, 172])
def test_native_bad_bindings_or_digests_do_not_prepare(case, offset):
    wire = bytearray(case[4].encode())
    wire[offset] ^= 1
    result = run(case, wire)
    assert result.returncode == 1
    assert "prepares=0" in result.stdout
    if offset in (140, 172):
        assert "error=23 " in result.stdout
        assert f"diagnostic={9 if offset == 140 else 5}" in result.stdout
    else:
        assert "diagnostic=1" in result.stdout


def test_native_changed_image_same_size_do_not_prepare(case):
    contents = bytearray(case[2].read_bytes())
    contents[-1] ^= 1
    case[2].write_bytes(contents)
    result = run(case, case[4].encode())
    assert result.returncode == 1
    assert "prepares=0" in result.stdout


@pytest.mark.parametrize("offset", [204, 212])
def test_native_wrong_volume_or_file_identity_do_not_prepare(case, offset):
    wire = bytearray(case[4].encode())
    wire[offset] ^= 1
    result = run(case, wire)
    assert result.returncode == 1, result.stdout
    assert "error=1006 " in result.stdout and "diagnostic=7" in result.stdout
    assert "prepares=0" in result.stdout


def test_native_replaced_image_with_identical_bytes_do_not_prepare(case):
    original = case[2].with_suffix(".original")
    case[2].rename(original)
    shutil.copyfile(original, case[2])
    result = run(case, case[4].encode())
    assert result.returncode == 1, result.stdout
    assert "error=1006 " in result.stdout and "diagnostic=7" in result.stdout
    assert "prepares=0" in result.stdout


def test_native_duplicate_paths_do_not_prepare(case):
    wire = bytearray(case[4].encode())
    wire.extend(wire[116:])
    struct.pack_into("<II", wire, 12, len(wire), 2)
    result = run(case, wire)
    assert result.returncode == 1
    assert "prepares=0" in result.stdout


def test_native_outside_generation_do_not_prepare(case):
    original = case[4]
    image = original.images[0]
    outside = replace(image.file, path = str(case[1].parent.parent / case[2].name))
    wire = replace(original, images = (replace(image, file = outside),)).encode()
    result = run(case, wire)
    assert result.returncode == 1
    assert "prepares=0" in result.stdout
    assert "error=161 " in result.stdout and "diagnostic=4" in result.stdout


@pytest.mark.parametrize("replacement", [b"\x00\x00", b"\x00\xd8", b"/\x00", b":\x00"])
def test_native_invalid_unicode_or_path_do_not_prepare(case, replacement):
    wire = bytearray(case[4].encode())
    wire[228 + 6 : 228 + 8] = replacement
    result = run(case, wire)
    assert result.returncode == 1
    assert "prepares=0" in result.stdout


def test_native_does_not_prepare_first_entry_when_second_is_invalid(case):
    wire = bytearray(case[4].encode())
    wire.extend(wire[116:])
    struct.pack_into("<II", wire, 12, len(wire), 2)
    # Truncate the second entry after updating the declared total size.
    wire = wire[:-1]
    struct.pack_into("<I", wire, 12, len(wire))
    result = run(case, wire)
    assert result.returncode == 1
    assert "prepares=0" in result.stdout
