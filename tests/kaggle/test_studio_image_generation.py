# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Image generation: the smallest run that can still be wrong in a visible way.

256x256 at 2 steps, because the claim is that the path executes rather than
that the picture is good.

"Nothing errored" is not the check, and that is the whole design. A diffusion
pipeline that fails part-way still writes a gallery record and still answers
200; a pipeline whose weights never loaded produces a FLAT frame, which is a
perfectly valid PNG. So the verdict is read off the downloaded file:

* the PNG magic, so the download endpoint is serving an image rather than a
  JSON error with a 200 on it;
* the size out of the IHDR chunk, not out of the gallery record -- the record
  repeats what was ASKED for and the file says what was MADE;
* not-one-flat-colour, on decoded extrema where PIL is available and on a
  compressed-size floor where it is not.
"""

from __future__ import annotations

import ast
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "studio_gpu" / "run_studio_gpu.py"
SRC = PAYLOAD.read_text(encoding = "utf-8")


def _func(name: str) -> ast.FunctionDef:
    for cls in ast.walk(ast.parse(SRC)):
        if not isinstance(cls, ast.ClassDef):
            continue
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
    raise AssertionError(f"no method named {name!r}")


def _body(name: str = "assert_image_generation") -> str:
    return ast.get_source_segment(SRC, _func(name)) or ""


def _png(width: int, height: int, payload: bytes) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", payload)


def test_the_assertion_exists_and_is_off_by_default():
    """Last priority, and it pulls a diffusion checkpoint the rest of the
    payload has no use for. A dispatch that wants it says so."""
    assert _body()
    assert "self.assert_image_generation()" in _body("execute")
    assert '"--image-generation",' in SRC
    assert "self.args.image_generation" in _body("execute")


def test_the_run_is_the_smallest_the_schema_allows():
    body = _body()
    assert "want = 256" in body, "256 is the schema's floor for width and height"
    assert '"steps": 2,' in body


def test_the_size_is_read_from_the_file_and_not_from_the_record():
    """The gallery record repeats what was asked for. Comparing the request
    against itself is the vacuity this rule exists against."""
    body = _body()
    assert 'int.from_bytes(png[16:20], "big")' in body
    assert 'int.from_bytes(png[20:24], "big")' in body


def test_the_ihdr_offsets_are_right():
    """Executed against a PNG built here, because an off-by-four in a
    fixed-offset parse reads a plausible number out of the wrong bytes and
    every other rule still passes."""
    blob = _png(256, 256, zlib.compress(b"\x00" * 16))
    assert int.from_bytes(blob[16:20], "big") == 256
    assert int.from_bytes(blob[20:24], "big") == 256
    wrong = _png(512, 64, zlib.compress(b"\x00" * 16))
    assert int.from_bytes(wrong[16:20], "big") == 512
    assert int.from_bytes(wrong[20:24], "big") == 64


def test_a_flat_image_is_a_failure():
    """A pipeline whose weights never loaded returns a uniform frame, and a
    uniform frame is a valid PNG of exactly the right size. Every other rule
    here passes on it."""
    func = _func("assert_image_generation")
    assert any(
        isinstance(n, ast.If) and ast.unparse(n.test) == "flat" for n in ast.walk(func)
    ), "nothing fails on a flat image"
    body = _body()
    assert "getextrema()" in body
    assert "flatness_source" in body, "the reader must be able to see which rule ruled"


def test_the_png_magic_is_checked():
    body = _body()
    assert 'png.startswith(b"\\x89PNG\\r\\n\\x1a\\n")' in body


def test_the_bytes_are_fetched_raw_rather_than_through_the_json_client():
    """`Studio.request` decodes to utf-8, which corrupts the bytes this whole
    assertion is about."""
    body = _body()
    assert "urllib.request.urlopen" in body
    assert "self.studio.get(" not in body.split("gallery/{image_id}/file")[0][-400:]


def test_the_pipeline_is_unloaded_in_a_finally():
    """A diffusion pipeline is the largest single thing this payload puts on a
    T4. Left resident, it takes the card from whatever runs next and that
    failure lands on the wrong assertion."""
    func = _func("assert_image_generation")
    finals = "\n".join(
        ast.unparse(n) for t in ast.walk(func) if isinstance(t, ast.Try) for n in t.finalbody
    )
    assert "images/unload" in finals


def test_a_load_or_generate_error_is_a_failure_rather_than_a_skip():
    body = _body()
    assert 'failures.append(f"images/load returned HTTP' in body
    assert "failures.append(" in body
    func = _func("assert_image_generation")
    assert any(
        isinstance(n, ast.If) and "code >= 400" in ast.unparse(n.test) for n in ast.walk(func)
    )
