# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Static resource and XML controls; these do not qualify native startup."""

from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox import activation_manifest as admission
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

EMPTY = b'<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0"/>'


def resource_image(payload = EMPTY):
    """A small real PE32+ DLL with one RT_MANIFEST/2/1033 resource."""
    data = bytearray(0x1200)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", data, 0x84, 0x8664, 1, 0, 0, 0, 240, 0x2022)
    opt = 0x98
    struct.pack_into("<H", data, opt, 0x20B)
    struct.pack_into("<I", data, opt + 16, 0x1000)
    struct.pack_into("<Q", data, opt + 24, 0x180000000)
    struct.pack_into("<II", data, opt + 32, 0x1000, 0x200)
    struct.pack_into("<II", data, opt + 56, 0x2000, 0x200)
    struct.pack_into("<H", data, opt + 68, 3)
    struct.pack_into("<I", data, opt + 108, 16)
    struct.pack_into("<II", data, opt + 112 + 16, 0x1000, 0x1000)
    section = opt + 240
    data[section : section + 8] = b".rsrc\0\0\0"
    struct.pack_into("<IIII", data, section + 8, 0x1000, 0x1000, 0x1000, 0x200)
    struct.pack_into("<I", data, section + 36, 0x40000040)
    for offset in (0, 24, 48):
        struct.pack_into("<IIHHHH", data, 0x200 + offset, 0, 0, 0, 0, 0, 1)
    struct.pack_into("<II", data, 0x210, 24, 0x80000018)
    struct.pack_into("<II", data, 0x228, 2, 0x80000030)
    struct.pack_into("<II", data, 0x240, 1033, 72)
    struct.pack_into("<IIII", data, 0x248, 0x1100, len(payload), 65001, 0)
    data[0x300 : 0x300 + len(payload)] = payload
    return data


def inspect_fixture(tmp_path, data):
    path = tmp_path / "selected-version.pyd"
    path.write_bytes(data)
    return admission.inspect_activation_image(path)


def test_single_read_exact_identity_and_immutable_inventory(tmp_path, monkeypatch):
    pytest.importorskip("pefile")
    data = resource_image()
    reader = admission.dependencies.read_regular_file
    calls = []

    def read(path, **kwargs):
        result = reader(path, **kwargs)
        calls.append(result)
        Path(path).write_bytes(b"changed after authorized snapshot")
        return result

    monkeypatch.setattr(admission.dependencies, "read_regular_file", read)
    result = inspect_fixture(tmp_path, data)
    manifest = admission.require_empty_activation_manifest(result)
    assert len(calls) == 1
    assert result.file.sha256 == hashlib.sha256(data).hexdigest()
    assert manifest.data == EMPTY
    assert manifest.sha256 == hashlib.sha256(EMPTY).hexdigest()
    assert (manifest.resource_id, manifest.language, manifest.codepage) == (2, 1033, 65001)
    with pytest.raises(FrozenInstanceError):
        manifest.language = 0


@pytest.mark.parametrize(
    "payload",
    [
        EMPTY,
        b'<?xml version="1.0" encoding="UTF-8"?>\n' + EMPTY,
        ('<?xml version="1.0" encoding="UTF-16"?>' + EMPTY.decode()).encode("utf-16"),
        EMPTY.replace(b"/>", b"> \r\n\t</assembly>"),
    ],
)
def test_only_empty_assembly_accepted(payload):
    admission._require_empty_xml(payload)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not XML",
        EMPTY + EMPTY,
        EMPTY.replace(b"/>", b"><dependency/></assembly>"),
        EMPTY.replace(b"/>", b">text</assembly>"),
        EMPTY.replace(b"/>", b"><!-- comment --></assembly>"),
        b"<!--before-->" + EMPTY,
        EMPTY + b"<!--after-->",
        b"<?instruction value?>" + EMPTY,
        b'<!DOCTYPE assembly [<!ENTITY test "value">]>' + EMPTY,
        b'<!DOCTYPE assembly SYSTEM "file:///host-secret">' + EMPTY,
        EMPTY.replace(b"1.0", b"&#49;.0"),
        EMPTY.replace(b"/>", b"><![CDATA[ ]]></assembly>"),
        EMPTY.replace(b"/>", b' extra="x"/>'),
        EMPTY.replace(b"/>", b' xmlns:x="urn:other"/>'),
        EMPTY.replace(b"asm.v1", b"asm.v2"),
        EMPTY.replace(b"manifestVersion", b"version"),
        EMPTY.replace(b"1.0", b"2.0"),
        b'<?xml version="1.0" encoding="ISO-8859-1"?>' + EMPTY,
        EMPTY.replace(b"/>", b"/>") + b"\0",
        b" " * (admission.MAX_MANIFEST_BYTES + 1),
    ],
    ids = lambda value: hashlib.sha256(value).hexdigest()[:12],
)
def test_unknown_xml_semantics_rejected(payload):
    with pytest.raises(WindowsRuntimeError):
        admission._require_empty_xml(payload)


@pytest.mark.parametrize(
    "offset,fmt,value,match",
    [
        (0x214, "<I", 0x80000000, "Cyclic"),
        (0x20E, "<H", 513, "limit"),
        (0x244, "<I", 0x80000048, "deep"),
        (0x244, "<I", 0xFFF, "Truncated"),
        (0x248, "<I", 0xFFFFFFF0, "Unmapped"),
        (0x24C, "<I", 0x1001, "Unmapped"),
        (0x254, "<I", 1, "Invalid"),
    ],
)
def test_malformed_resource_controls(tmp_path, offset, fmt, value, match):
    pytest.importorskip("pefile")
    data = resource_image()
    struct.pack_into(fmt, data, offset, value)
    with pytest.raises(WindowsRuntimeError, match = match):
        inspect_fixture(tmp_path, data)


def test_no_manifest_and_ambiguous_admission(tmp_path):
    pytest.importorskip("pefile")
    image = inspect_fixture(tmp_path, resource_image())
    assert admission.require_empty_activation_manifest(replace(image, manifests = ())) is None
    for manifests in (
        (image.manifests[0],) * 2,
        (replace(image.manifests[0], resource_id = "named"),),
        (replace(image.manifests[0], language = "named"),),
    ):
        with pytest.raises(WindowsRuntimeError):
            admission.require_empty_activation_manifest(replace(image, manifests = manifests))
    data = resource_image()
    struct.pack_into("<II", data, 0x98 + 112 + 16, 0, 0)
    assert inspect_fixture(tmp_path, data).manifests == ()


def test_named_resource_is_preserved_but_not_admitted(tmp_path):
    pytest.importorskip("pefile")
    data = resource_image()
    struct.pack_into("<HH", data, 0x224, 1, 0)
    struct.pack_into("<I", data, 0x228, 0x80000080)
    struct.pack_into("<H", data, 0x280, 5)
    data[0x282:0x28C] = "named".encode("utf-16-le")
    image = inspect_fixture(tmp_path, data)
    assert image.manifests[0].resource_id == "named"
    with pytest.raises(WindowsRuntimeError, match = "resource ID"):
        admission.require_empty_activation_manifest(image)


@pytest.mark.parametrize(
    "language,target,match",
    [(1033, 88, "Duplicate"), (1041, 96, "Shared"), (1041, 112, "Overlapping")],
)
def test_ambiguous_resource_entries_rejected(tmp_path, language, target, match):
    pytest.importorskip("pefile")
    data = resource_image()
    struct.pack_into("<H", data, 0x23E, 2)
    struct.pack_into("<II", data, 0x240, 1033, 96)
    struct.pack_into("<II", data, 0x248, language, target)
    struct.pack_into("<IIII", data, 0x260, 0x1100, len(EMPTY), 0, 0)
    struct.pack_into("<IIII", data, 0x270, 0x1100, len(EMPTY), 0, 0)
    with pytest.raises(WindowsRuntimeError, match = match):
        inspect_fixture(tmp_path, data)


def test_parser_version_is_pinned(tmp_path, monkeypatch):
    pefile = pytest.importorskip("pefile")
    monkeypatch.setattr(pefile, "__version__", "unreviewed")
    with pytest.raises(WindowsRuntimeError, match = "pinned"):
        inspect_fixture(tmp_path, resource_image())


def test_selected_installed_pillow_static_inventory():
    pytest.importorskip("pefile")
    # Search interpreter paths without importing Pillow or its native initializers.
    candidates = sorted(
        {
            path
            for entry in sys.path
            if entry
            for path in (Path(entry) / "PIL").glob("_imaging.*.pyd")
        }
    )
    if not candidates:
        pytest.skip("Selected interpreter has no installed Pillow native image")
    assert len(candidates) == 1, "Select an unambiguous interpreter environment"
    image = admission.inspect_activation_image(candidates[0])
    manifest = admission.require_empty_activation_manifest(image)
    assert manifest is not None
    assert manifest.resource_id == 2
    assert manifest.language == 1033
    assert Path(image.file.path) == candidates[0].resolve()
