# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A small .ppt whose slide list names one big slide many times must not multiply its text
in the shared backend's memory; a normal deck (each slide listed once) reads as before."""

import struct

import pytest

from core.rag import office_formats

from ._compound_file import compound_file
from .test_rag_office_formats import _ppt_atom, _ppt_container, build_saved_ppt


def _deck(path, repeats, text):
    chars = lambda t: _ppt_atom(0x0FA0, t.encode("utf-16-le"))
    persist = _ppt_atom(0x03F3, struct.pack("<5I", 2, 0, 0, 256, 0))
    slide = _ppt_container(
        0x03EE, [_ppt_container(0x040C, [_ppt_container(0xF00D, [chars(text)])])]
    )
    document = _ppt_container(0x03E8, [_ppt_container(0x0FF0, [persist] * repeats, inst = 0)])
    stream = bytearray()

    def put(record):
        stream.extend(record)
        return len(stream) - len(record)

    doc_at, slide_at = put(document), put(slide)
    body = b"".join(
        struct.pack("<II", (1 << 20) | pid, off) for pid, off in ((1, doc_at), (2, slide_at))
    )
    directory_at = put(_ppt_atom(0x1772, body))
    edit_at = put(_ppt_atom(0x0FF5, struct.pack("<IHBBIIII", 0, 0, 0, 3, 0, directory_at, 1, 4)))
    current_user = _ppt_atom(0x0FF6, struct.pack("<III", 20, 0xE391C05F, edit_at))
    path.write_bytes(
        compound_file({("PowerPoint Document",): bytes(stream), ("Current User",): current_user})
    )
    return path


def test_repeated_slide_references_are_budgeted(tmp_path):
    path = _deck(tmp_path / "bomb.ppt", repeats = 200, text = "Z" * 200_000)
    assert path.stat().st_size < 1_000_000
    try:
        sections = office_formats.ppt(str(path))
    except ValueError as exc:
        assert "repeats too much" in str(exc)
        return
    total = sum(len(text) for text, _ in sections)
    pytest.fail(f"a {path.stat().st_size} byte deck expanded to {total} chars")


def test_normal_decks_read_unchanged(tmp_path):
    assert office_formats.ppt(str(_deck(tmp_path / "one.ppt", repeats = 1, text = "Hello"))) == [
        ("Hello", 1)
    ]
    assert len(office_formats.ppt(str(_deck(tmp_path / "few.ppt", repeats = 5, text = "Hi")))) == 5
    sections = office_formats.ppt(str(build_saved_ppt(tmp_path / "saved.ppt")))
    assert sections == [("Title from list\nText box zebramarker\nNotes:\nSpeaker notes", 1)]
