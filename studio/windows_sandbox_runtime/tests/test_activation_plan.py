# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from dataclasses import replace
import hashlib
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox import activation_plan as plans
from core.inference.windows_sandbox.activation_manifest import ActivationImage, ActivationManifest
from core.inference.windows_sandbox.content import ContentGeneration, SnapshotFile, SnapshotSpec
from core.inference.windows_sandbox.dependencies import FileIdentity, read_regular_file
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

EMPTY = b'<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0"/>'
NONCE, PROFILE, CONTENT = b"n" * 32, b"p" * 32, b"c" * 32
ROOT = "C:\\protected\\generation"


def image(path = ROOT + "\\packages\\image.pyd"):
    manifest = ActivationManifest(2, 1033, 0, EMPTY, hashlib.sha256(EMPTY).hexdigest())
    return ActivationImage(FileIdentity(path, "ab" * 32, 1000, 4, 5), (manifest,))


def plan():
    return plans.ActivationPlan(NONCE, PROFILE, CONTENT, (image(),), (b"i" * 24,))


def decode(data, **overrides):
    kwargs = dict(nonce = NONCE, profile_digest = PROFILE, content_digest = CONTENT, runtime_home = ROOT)
    kwargs.update(overrides)
    return plans.ActivationPlan.decode(data, **kwargs)


def test_wire_roundtrip_retains_exact_bytes_and_bindings():
    encoded = plan().encode()
    assert encoded[:8] == b"USLPACT\0"
    assert decode(encoded).encode() == encoded
    parsed = decode(encoded)
    assert parsed.images[0].manifests == plan().images[0].manifests
    assert parsed.images[0].file.sha256 == "ab" * 32
    assert plans.HEADER.size == 116
    assert plans.ENTRY.size == 112
    assert parsed.file_identities == plan().file_identities


@pytest.mark.parametrize("binding", ["nonce", "profile_digest", "content_digest"])
def test_launch_binding_mismatch(binding):
    with pytest.raises(WindowsRuntimeError, match = "binding"):
        decode(plan().encode(), **{binding: b"x" * 32})


@pytest.mark.parametrize("offset,value", [(8, 1), (12, 0), (16, 65), (116, 3), (120, 65537)])
def test_wire_bounds_and_schema_rejected(offset, value):
    encoded = bytearray(plan().encode())
    struct.pack_into("<I", encoded, offset, value)
    with pytest.raises(WindowsRuntimeError):
        decode(bytes(encoded))


def test_trailing_and_truncated_input_rejected():
    encoded = plan().encode()
    for length in (0, 115, 116, 117, len(encoded) - 1):
        with pytest.raises(WindowsRuntimeError):
            decode(encoded[:length])
    with pytest.raises(WindowsRuntimeError):
        decode(encoded + b"extra")


def test_corrupted_manifest_digest_and_duplicate_paths_rejected():
    original = plan()
    manifest = replace(original.images[0].manifests[0], sha256 = "00" * 32)
    with pytest.raises(WindowsRuntimeError, match = "digest"):
        replace(original, images = (replace(original.images[0], manifests = (manifest,)),)).encode()
    with pytest.raises(WindowsRuntimeError, match = "Duplicate"):
        replace(
            original,
            images = (original.images[0], original.images[0]),
            file_identities = original.file_identities * 2,
        ).encode()


@pytest.mark.parametrize("path", ["C:\\elsewhere\\image.pyd", ROOT + "-other\\image.pyd"])
def test_outside_generation_rejected(path):
    with pytest.raises(WindowsRuntimeError, match = "outside"):
        decode(replace(plan(), images = (image(path),)).encode())


def test_manifestless_entry_and_invalid_numeric_metadata_rejected():
    original = image()
    for candidate in (
        replace(original, manifests = ()),
        replace(original, manifests = (replace(original.manifests[0], language = True),)),
        replace(original, manifests = (replace(original.manifests[0], language = 65536),)),
        replace(original, manifests = (replace(original.manifests[0], codepage = -1),)),
    ):
        with pytest.raises(WindowsRuntimeError):
            replace(plan(), images = (candidate,)).encode()


def test_cumulative_and_entry_limits(monkeypatch):
    with pytest.raises(WindowsRuntimeError, match = "entry limit"):
        replace(plan(), images = (image(),) * 65).encode()
    monkeypatch.setattr(plans, "MAX_BYTES", plans.HEADER.size + plans.ENTRY.size)
    with pytest.raises(WindowsRuntimeError, match = "byte limit"):
        plan().encode()


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    directory = tmp_path / "generation"
    directory.mkdir()
    files = directory / "files"
    native = files / "runtime" / "Lib" / "site-packages" / "PIL"
    native.mkdir(parents = True)
    selected = native / "selected-version.pyd"
    selected.write_bytes(b"selected final snapshot bytes")
    identity, _ = read_regular_file(selected, limit = 1000)
    relative = selected.relative_to(files).as_posix()
    spec = SnapshotSpec(
        (SnapshotFile(identity, relative),), "11" * 32, "22" * 32, PROFILE.hex(), "44" * 32
    )
    generation = ContentGeneration(
        hashlib.sha256(spec.manifest()).hexdigest(), directory, (selected,)
    )

    def inspect(path):
        identity, _ = read_regular_file(path, limit = 1000)
        return replace(image(str(path)), file = identity)

    monkeypatch.setattr(plans, "inspect_activation_image", inspect)
    return generation, spec, selected


def build(snapshot):
    generation, spec, selected = snapshot
    relative = selected.relative_to(generation.directory / "files").as_posix()
    return plans.build_activation_plan(
        generation, spec, (relative,), nonce = NONCE, profile_digest = PROFILE
    )


def test_builder_binds_final_snapshot(snapshot):
    built = build(snapshot)
    assert built.content_digest.hex() == snapshot[0].digest
    assert built.images[0].file.path == str(snapshot[2])
    assert built.images[0].file.sha256 == snapshot[1].files[0].source.sha256


def test_builder_rejects_changed_final_bytes(snapshot):
    snapshot[2].write_bytes(b"replaced image")
    with pytest.raises(WindowsRuntimeError, match = "differs"):
        build(snapshot)


def test_builder_rejects_wrong_generation_or_unlisted_file(snapshot):
    generation, spec, selected = snapshot
    for changed in (replace(generation, digest = "00" * 32), replace(generation, files = ())):
        with pytest.raises(WindowsRuntimeError):
            build((changed, spec, selected))
    for paths in (("../outside.pyd",), ("unlisted.pyd",), (selected.name, selected.name)):
        with pytest.raises(WindowsRuntimeError):
            plans.build_activation_plan(
                generation, spec, paths, nonce = NONCE, profile_digest = PROFILE
            )


def test_no_manifest_needs_no_plan_entry(snapshot, monkeypatch):
    inspector = plans.inspect_activation_image
    monkeypatch.setattr(
        plans, "inspect_activation_image", lambda path: replace(inspector(path), manifests = ())
    )
    assert build(snapshot).images == ()
