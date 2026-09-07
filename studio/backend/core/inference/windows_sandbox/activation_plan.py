# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Bounded activation input bound to a final snapshot, not startup authorization.

The launch owner must hold the protected generation's directory/read lease while
building and using this record. The native consumer must compare launch bindings,
pin and hash each image, and compare the exact manifest before preparing contexts.
The plan belongs on a trusted launch channel, never a payload-writable path.
"""

from dataclasses import dataclass
import ctypes
from ctypes import wintypes as W
import hashlib
import ntpath
from pathlib import Path
import struct

from .activation_manifest import (
    ActivationImage,
    ActivationManifest,
    MAX_IMAGE_BYTES,
    MAX_MANIFEST_BYTES,
    inspect_activation_image,
    require_empty_activation_manifest,
)
from .content import ContentGeneration, SnapshotSpec, _relative
from .dependencies import FileIdentity
from .host_config import _path, _string
from .profiles import WindowsRuntimeError

MAGIC = b"USLPACT\0"
HEADER = struct.Struct("<8sIII32s32s32s")
ENTRY = struct.Struct("<IIHHIQ32s32s24s")
MAX_ENTRIES = 64
MAX_BYTES = 1024 * 1024


def _invalid(message):
    return WindowsRuntimeError("WINDOWS_SANDBOX_ACTIVATION_UNSUPPORTED", message)


def _digest(value):
    if type(value) is not bytes or len(value) != 32:
        raise _invalid("Invalid activation launch digest or nonce.")
    return value


def _hex_digest(value):
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise _invalid("Invalid activation image digest.")
    return bytes.fromhex(value)


def measure_file_identity(image: ActivationImage) -> bytes:
    """Bind the final image's Win32 volume/file ID while its generation is leased."""
    from .content_files import native_files

    class FileIdInfo(ctypes.Structure):
        _fields_ = [("volume", ctypes.c_ulonglong), ("identifier", ctypes.c_ubyte * 16)]

    api = native_files()
    query = api.kernel.GetFileInformationByHandleEx
    query.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD]
    query.restype = W.BOOL
    handle = api.open(Path(image.file.path))
    try:
        api.info(handle)
        identity = FileIdInfo()
        if not query(handle, 18, ctypes.byref(identity), ctypes.sizeof(identity)):
            raise _invalid(
                f"Final activation file identity query failed: {ctypes.get_last_error()}."
            )
        current = Path(image.file.path).stat()
        if (current.st_dev, current.st_ino, current.st_size) != (
            image.file.device,
            image.file.inode,
            image.file.size,
        ):
            raise _invalid("Final activation image changed before file identity binding.")
        return bytes(identity)
    finally:
        if not api.kernel.CloseHandle(handle):
            raise _invalid("Final activation identity handle could not be closed.")


@dataclass(frozen = True)
class ActivationPlan:
    nonce: bytes
    profile_digest: bytes
    content_digest: bytes
    images: tuple[ActivationImage, ...]
    file_identities: tuple[bytes, ...] = ()

    def encode(self) -> bytes:
        bindings = tuple(
            _digest(value) for value in (self.nonce, self.profile_digest, self.content_digest)
        )
        if type(self.images) is not tuple or len(self.images) > MAX_ENTRIES:
            raise _invalid("Activation plan entry limit exceeded.")
        if (
            type(self.file_identities) is not tuple
            or len(self.file_identities) != len(self.images)
            or any(type(value) is not bytes or len(value) != 24 for value in self.file_identities)
        ):
            raise _invalid("Activation plan requires exact volume and file identity bindings.")
        chunks = []
        size = HEADER.size
        paths = set()
        for image, file_identity in zip(self.images, self.file_identities):
            if type(image) is not ActivationImage or type(image.manifests) is not tuple:
                raise _invalid("Invalid activation image record.")
            path = _path(image.file.path)
            if path.casefold() in paths:
                raise _invalid("Duplicate activation image path.")
            paths.add(path.casefold())
            manifest = require_empty_activation_manifest(image)
            if manifest is None:
                raise _invalid("An activation plan entry must contain its approved manifest.")
            if type(manifest.data) is not bytes or not 0 < len(manifest.data) <= MAX_MANIFEST_BYTES:
                raise _invalid("Invalid activation manifest bytes.")
            if type(manifest.resource_id) is not int or manifest.resource_id != 2:
                raise _invalid("Unsupported activation resource ID.")
            if type(manifest.language) is not int or not 0 <= manifest.language <= 65535:
                raise _invalid("Unsupported activation resource language.")
            if type(manifest.codepage) is not int or not 0 <= manifest.codepage <= 0xFFFFFFFF:
                raise _invalid("Unsupported activation resource codepage.")
            if type(image.file.size) is not int or not 0 < image.file.size <= MAX_IMAGE_BYTES:
                raise _invalid("Invalid activation image size.")
            image_digest = _hex_digest(image.file.sha256)
            manifest_digest = _hex_digest(manifest.sha256)
            if hashlib.sha256(manifest.data).digest() != manifest_digest:
                raise _invalid("Activation manifest digest differs from its exact bytes.")
            encoded_path = _string(path)[4:]
            size += ENTRY.size + len(encoded_path) + len(manifest.data)
            if size > MAX_BYTES:
                raise _invalid("Activation plan byte limit exceeded.")
            chunks.extend(
                (
                    ENTRY.pack(
                        len(encoded_path),
                        len(manifest.data),
                        manifest.resource_id,
                        manifest.language,
                        manifest.codepage,
                        image.file.size,
                        image_digest,
                        manifest_digest,
                        file_identity,
                    ),
                    encoded_path,
                    manifest.data,
                )
            )
        return HEADER.pack(MAGIC, 2, size, len(self.images), *bindings) + b"".join(chunks)

    @classmethod
    def decode(
        cls,
        data: bytes,
        *,
        nonce: bytes,
        profile_digest: bytes,
        content_digest: bytes,
        runtime_home: str,
    ) -> "ActivationPlan":
        """Validate wire input, not the filesystem; native image pins remain required."""
        if type(data) is not bytes or not HEADER.size <= len(data) <= MAX_BYTES:
            raise _invalid("Invalid activation plan length.")
        root = _path(runtime_home)
        magic, version, size, count, wire_nonce, profile, content = HEADER.unpack_from(data)
        if (
            magic != MAGIC
            or version != 2
            or size != len(data)
            or count > MAX_ENTRIES
            or wire_nonce != _digest(nonce)
            or profile != _digest(profile_digest)
            or content != _digest(content_digest)
        ):
            raise _invalid("Activation plan version, bounds or launch binding mismatch.")
        cursor = HEADER.size
        images = []
        identities = []
        for _ in range(count):
            if cursor + ENTRY.size > len(data):
                raise _invalid("Truncated activation entry header.")
            (
                path_bytes,
                manifest_bytes,
                resource_id,
                language,
                codepage,
                image_size,
                image_hash,
                manifest_hash,
                file_identity,
            ) = ENTRY.unpack_from(data, cursor)
            cursor += ENTRY.size
            if (
                not 0 < path_bytes <= 32766
                or path_bytes % 2
                or not 0 < manifest_bytes <= MAX_MANIFEST_BYTES
                or cursor + path_bytes + manifest_bytes > len(data)
            ):
                raise _invalid("Invalid activation entry byte bounds.")
            try:
                path = data[cursor : cursor + path_bytes].decode("utf-16-le", errors = "strict")
            except UnicodeError as error:
                raise _invalid("Invalid activation entry Unicode.") from error
            cursor += path_bytes
            payload = data[cursor : cursor + manifest_bytes]
            cursor += manifest_bytes
            _path(path)
            if not path.casefold().startswith(root.casefold() + "\\"):
                raise _invalid("Activation image is outside the protected generation.")
            manifest = ActivationManifest(
                resource_id, language, codepage, payload, manifest_hash.hex()
            )
            images.append(
                ActivationImage(FileIdentity(path, image_hash.hex(), image_size, 0, 0), (manifest,))
            )
            identities.append(file_identity)
        result = cls(wire_nonce, profile, content, tuple(images), tuple(identities))
        if cursor != len(data) or result.encode() != data:
            raise _invalid("Noncanonical activation plan or trailing bytes.")
        return result


def build_activation_plan(
    generation: ContentGeneration,
    spec: SnapshotSpec,
    approved_relative_paths: tuple[str, ...],
    *,
    nonce: bytes,
    profile_digest: bytes,
) -> ActivationPlan:
    """Inspect explicitly approved final snapshot files while the caller holds leases.

    Selection must come from the reviewed startup profile. This function proves
    content/admission correspondence; supplying a path does not grant it trust.
    """
    if type(generation) is not ContentGeneration or type(spec) is not SnapshotSpec:
        raise _invalid("Activation planning requires a final content generation and specification.")
    manifest = spec.manifest()
    expected = hashlib.sha256(manifest).hexdigest()
    if generation.digest != expected or _hex_digest(spec.profile_digest) != _digest(profile_digest):
        raise _invalid("Activation plan snapshot or profile binding mismatch.")
    _digest(nonce)
    if type(approved_relative_paths) is not tuple or len(approved_relative_paths) > MAX_ENTRIES:
        raise _invalid("Activation selection entry limit exceeded.")
    root = _path(str(generation.directory / "files"))
    members = {str(path).casefold() for path in generation.files}
    expected_files = {item.relative_path.casefold(): item for item in spec.files}
    selected = set()
    images = []
    identities = []
    for relative in approved_relative_paths:
        relative = _relative(relative)
        if relative.casefold() in selected:
            raise _invalid("Duplicate activation selection.")
        selected.add(relative.casefold())
        expected_file = expected_files.get(relative.casefold())
        if expected_file is None:
            raise _invalid("Activation image is absent from the snapshot specification.")
        path = _path(ntpath.join(root, relative.replace("/", "\\")))
        if path.casefold() not in members:
            raise _invalid("Activation image is absent from the final generation.")
        image = inspect_activation_image(Path(path))
        if (
            image.file.path.casefold() != path.casefold()
            or image.file.sha256 != expected_file.source.sha256
            or image.file.size != expected_file.source.size
        ):
            raise _invalid("Final activation image differs from the approved snapshot content.")
        if require_empty_activation_manifest(image) is not None:
            images.append(image)
            identities.append(measure_file_identity(image))
    plan = ActivationPlan(
        nonce, profile_digest, _hex_digest(generation.digest), tuple(images), tuple(identities)
    )
    plan.encode()
    return plan
