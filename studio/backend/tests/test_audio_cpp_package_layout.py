# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Spec packages in the HF cache: the runtime is handed the primary GGUF, and the files beside it
keep their places under the package root. A symlinked cache is served through the link farm."""

import hashlib
import os
import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
from test_audio_cpp_models import _gguf_bytes, _put, _snapshot, hub  # noqa: E402, F401

from core.inference import audio_cpp_files  # noqa: E402
from core.inference.audio_cpp_models import AudioCppModel, AudioCppVariant, RepoFile  # noqa: E402
from core.inference.audio_cpp_server import AudioCppUnavailableError  # noqa: E402

ZONOS_REPO = "Zyphra/ZONOS3-Mini"
PRIMARY = "gguf/backbone-ptq.gguf"
PTQ = {
    PRIMARY: _gguf_bytes(arch = "zonos3"),
    "gguf/zac-keep2.gguf": b"zac" * 40,
    "gguf/spk-encoder.gguf": b"spk" * 30,
}
# Symlinks need privileges on Windows, where a real-file cache is served through the link farm too.
WINDOWS = sys.platform == "win32"


def _zonos_model(extra = (), key = "zonos3_mini_ptq"):
    files = (*(RepoFile(p, len(d)) for p, d in PTQ.items()), *extra)
    variant = AudioCppVariant(key, tuple(files), PRIMARY, root = "gguf")
    return AudioCppModel(
        id = ZONOS_REPO,
        repo_id = ZONOS_REPO,
        folder = "",
        display_name = "ZONOS3 Mini",
        family = "zonos3",
        task = "tts",
        server_task = "tts",
        variant = variant,
        variants = (variant,),
        default_variant = key,
    )


def _cache_file(snap, rel, data, *, linked):
    """A snapshot file: plain, or a symlink to a blob the way the HF cache stores it."""
    if not linked:
        return _put(snap, rel, data)
    blob = snap.parent.parent / "blobs" / hashlib.sha256(data).hexdigest()
    blob.parent.mkdir(parents = True, exist_ok = True)
    blob.write_bytes(data)
    path = snap / rel
    path.parent.mkdir(parents = True, exist_ok = True)
    os.symlink(blob, path)
    return path


def _fill(snap, *, linked):
    for rel, data in PTQ.items():
        _cache_file(snap, rel, data, linked = linked)


def _farm(hub_root, model):
    return hub_root.parent / "unsloth-audiocpp-links" / model.key


@pytest.mark.skipif(WINDOWS, reason = "Windows always serves from the link farm")
def test_real_snapshot_files_are_served_in_place(hub):
    snap = _snapshot(hub, ZONOS_REPO)
    _fill(snap, linked = False)

    assert audio_cpp_files.materialize(_zonos_model()) == str(snap / PRIMARY)


def test_symlinked_snapshot_is_served_from_a_flat_link_farm(hub):
    snap = _snapshot(hub, ZONOS_REPO)
    _fill(snap, linked = not WINDOWS)
    manifest = b'{"voices": []}'
    _cache_file(snap, "gguf/voices/manifest.json", manifest, linked = not WINDOWS)
    model = _zonos_model(extra = (RepoFile("gguf/voices/manifest.json", len(manifest)),))
    farm = _farm(hub, model)

    assert audio_cpp_files.materialize(model) == str(farm / "backbone-ptq.gguf")
    for name in ("zac-keep2.gguf", "spk-encoder.gguf"):
        flat = farm / name
        assert flat.is_file() and not flat.is_symlink()
    assert not (farm / "gguf").exists()
    # A file below the root keeps its place there.
    assert (farm / "voices" / "manifest.json").read_bytes() == manifest


def test_file_outside_the_package_root_is_refused(hub):
    snap = _snapshot(hub, ZONOS_REPO)
    _fill(snap, linked = not WINDOWS)
    stray = b"stray" * 10
    _cache_file(snap, "other/x.gguf", stray, linked = not WINDOWS)
    model = _zonos_model(extra = (RepoFile("other/x.gguf", len(stray)),))

    with pytest.raises(AudioCppUnavailableError):
        audio_cpp_files.materialize(model)


def test_missing_files_lists_what_the_cache_lacks_with_sizes(hub):
    _put(_snapshot(hub, ZONOS_REPO), PRIMARY, PTQ[PRIMARY])

    assert audio_cpp_files.missing_files(_zonos_model()) == [
        ("gguf/zac-keep2.gguf", len(PTQ["gguf/zac-keep2.gguf"])),
        ("gguf/spk-encoder.gguf", len(PTQ["gguf/spk-encoder.gguf"])),
    ]
