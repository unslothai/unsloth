# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Static activation-resource admission; never loads or modifies the selected image."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from xml.parsers import expat

from . import dependencies
from .profiles import WindowsRuntimeError

MAX_IMAGE_BYTES = 128 * 1024 * 1024
MAX_RESOURCE_BYTES = 8 * 1024 * 1024
MAX_RESOURCE_ENTRIES = 512
MAX_MANIFEST_BYTES = 64 * 1024
MAX_MANIFESTS = 8
_NS = "urn:schemas-microsoft-com:asm.v1"


@dataclass(frozen = True)
class ActivationManifest:
    resource_id: int | str
    language: int | str
    codepage: int
    data: bytes
    sha256: str


@dataclass(frozen = True)
class ActivationImage:
    file: dependencies.FileIdentity
    manifests: tuple[ActivationManifest, ...]


def _reject(message: str) -> None:
    raise WindowsRuntimeError("WINDOWS_SANDBOX_ACTIVATION_UNSUPPORTED", message)


def _require_empty_xml(data: bytes) -> None:
    if not data or len(data) > MAX_MANIFEST_BYTES:
        _reject("Missing or oversized activation manifest.")
    try:
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            source = data.decode("utf-16")
        elif data.startswith(b"\x00<"):
            source = data.decode("utf-16-be")
        elif data.startswith(b"<\x00"):
            source = data.decode("utf-16-le")
        else:
            source = data.decode("utf-8-sig")
    except UnicodeError:
        _reject("Activation manifest must use valid UTF-8 or UTF-16.")
    if "&" in source:
        _reject("Activation manifest contains entity or character references.")
    parser = expat.ParserCreate(namespace_separator = "|")
    roots = 0
    depth = 0
    namespaces = 0

    def start_namespace(prefix, uri):
        nonlocal namespaces
        namespaces += 1
        if prefix is not None or uri != _NS or namespaces != 1:
            _reject("Activation manifest has extra or unsupported namespaces.")

    def start(name, attrs):
        nonlocal roots, depth
        if depth or roots or name != _NS + "|assembly" or attrs != {"manifestVersion": "1.0"}:
            _reject("Activation manifest must be a single empty assembly with manifestVersion=1.0.")
        roots += 1
        depth += 1

    def end(_name):
        nonlocal depth
        depth -= 1

    def text(value):
        if value.strip(" \t\r\n"):
            _reject("Activation manifest contains significant text.")

    def forbidden(*_args):
        _reject("Activation manifest contains comments, processing instructions, DTD or entities.")

    def declaration(version, encoding, standalone):
        if version != "1.0" or (encoding and encoding.upper() not in ("UTF-8", "UTF-16")):
            _reject("Unsupported activation manifest XML declaration.")

    parser.StartNamespaceDeclHandler = start_namespace
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = text
    parser.XmlDeclHandler = declaration
    parser.CommentHandler = forbidden
    parser.ProcessingInstructionHandler = forbidden
    parser.StartDoctypeDeclHandler = forbidden
    parser.EntityDeclHandler = forbidden
    parser.ExternalEntityRefHandler = forbidden
    parser.StartCdataSectionHandler = forbidden
    try:
        parser.Parse(data, True)
    except expat.ExpatError as exc:
        _reject(f"Malformed activation manifest XML: {exc}.")
    if roots != 1 or namespaces != 1 or depth:
        _reject("Activation manifest is not an empty assembly.")


def require_empty_activation_manifest(image: ActivationImage) -> ActivationManifest | None:
    """Admit only the measured DLL manifest slot; absence needs no adapter."""
    if not image.manifests:
        return None
    if len(image.manifests) != 1:
        _reject("Multiple activation manifests are ambiguous and unsupported.")
    manifest = image.manifests[0]
    if manifest.resource_id != 2 or not isinstance(manifest.language, int):
        _reject("Unsupported activation manifest resource ID or language.")
    _require_empty_xml(manifest.data)
    return manifest


def inspect_activation_image(path: str | Path) -> ActivationImage:
    identity, data = dependencies.read_regular_file(path, limit = MAX_IMAGE_BYTES)
    try:
        import pefile
    except ImportError as exc:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_RUNTIME_MISSING", f"Requires pefile=={dependencies.PEFILE_VERSION}."
        ) from exc
    if pefile.__version__ != dependencies.PEFILE_VERSION:
        _reject("The PE parser version does not match the pinned activation inventory parser.")
    try:
        with pefile.PE(data = data, fast_load = True) as image:
            if not 0 < image.FILE_HEADER.NumberOfSections <= 96:
                _reject("PE section limit exceeded.")
            if image.get_warnings():
                _reject("PE parser reported malformed image headers.")

            def mapped(rva, size):
                matches = []
                for section in image.sections:
                    delta = rva - section.VirtualAddress
                    if delta >= 0 and delta + size <= section.SizeOfRawData:
                        offset = section.PointerToRawData + delta
                        if offset + size <= len(data):
                            matches.append(offset)
                if len(matches) != 1 or size <= 0:
                    _reject("Unmapped, overlapping or truncated PE resource data.")
                offset = matches[0]
                return data[offset : offset + size]

            directory = image.OPTIONAL_HEADER.DATA_DIRECTORY[2]
            if not directory.VirtualAddress and not directory.Size:
                return ActivationImage(identity, ())
            if not directory.VirtualAddress or not 16 <= directory.Size <= MAX_RESOURCE_BYTES:
                _reject("Missing or oversized PE resource directory.")
            resource = mapped(directory.VirtualAddress, directory.Size)
            manifests = []
            visited = set()
            leaves = set()
            payload_ranges = []
            count = 0
            payload_bytes = 0

            def unpack(fmt, offset):
                size = struct.calcsize(fmt)
                if offset < 0 or offset + size > len(resource):
                    _reject("Truncated PE resource directory.")
                return struct.unpack_from(fmt, resource, offset)

            def key(raw):
                if not raw & 0x80000000:
                    if raw > 0xFFFF:
                        _reject("Invalid numeric PE resource identifier.")
                    return raw
                offset = raw & 0x7FFFFFFF
                (length,) = unpack("<H", offset)
                if not 0 < length <= 256 or offset + 2 + length * 2 > len(resource):
                    _reject("Invalid PE resource name length.")
                return resource[offset + 2 : offset + 2 + length * 2].decode(
                    "utf-16-le", errors = "strict"
                )

            def walk(offset, keys):
                nonlocal count, payload_bytes
                if offset in visited or len(keys) > 2:
                    _reject("Cyclic, shared or overly deep PE resource directories.")
                visited.add(offset)
                _, _, _, _, named, numbered = unpack("<IIHHHH", offset)
                count += named + numbered
                if count > MAX_RESOURCE_ENTRIES:
                    _reject("PE resource entry limit exceeded.")
                siblings = set()
                for index in range(named + numbered):
                    raw, target = unpack("<II", offset + 16 + index * 8)
                    if bool(raw & 0x80000000) != (index < named):
                        _reject("Ambiguous PE resource name table.")
                    name = key(raw)
                    if name in siblings:
                        _reject("Duplicate PE resource identifier.")
                    siblings.add(name)
                    current = (*keys, name)
                    if len(current) < 3:
                        if not target & 0x80000000:
                            _reject("Premature PE resource leaf.")
                        walk(target & 0x7FFFFFFF, current)
                    else:
                        if target & 0x80000000:
                            _reject("Overly deep PE resource tree.")
                        if target in leaves:
                            _reject("Shared PE resource data entries are ambiguous.")
                        leaves.add(target)
                        rva, size, codepage, reserved = unpack("<IIII", target)
                        if reserved or size > MAX_RESOURCE_BYTES:
                            _reject("Invalid PE resource data entry.")
                        payload_bytes += size
                        if payload_bytes > MAX_RESOURCE_BYTES:
                            _reject("Cumulative PE resource data limit exceeded.")
                        if any(rva < end and start < rva + size for start, end in payload_ranges):
                            _reject("Overlapping PE resource payloads are ambiguous.")
                        payload_ranges.append((rva, rva + size))
                        payload = mapped(rva, size)
                        if current[0] == 24:
                            if len(manifests) >= MAX_MANIFESTS or size > MAX_MANIFEST_BYTES:
                                _reject("Activation manifest inventory limit exceeded.")
                            manifests.append(
                                ActivationManifest(
                                    current[1],
                                    current[2],
                                    codepage,
                                    payload,
                                    hashlib.sha256(payload).hexdigest(),
                                )
                            )

            walk(0, ())
            if image.get_warnings():
                _reject("PE parser reported malformed image headers.")
            return ActivationImage(identity, tuple(manifests))
    except (
        pefile.PEFormatError,
        IndexError,
        AttributeError,
        ValueError,
        TypeError,
        struct.error,
    ) as exc:
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_PE_INVALID", "Malformed activation image resources."
        ) from exc
