# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Previews of chat attachments only the python tool can read: an image becomes a PNG, a document
its text, anything else a short outline. Every reader is bounded by what the file states it would
have to read, so a large file costs about as much as a small one."""

from __future__ import annotations

import ast
import base64
import bz2
import codecs
import gzip
import io
import json
import lzma
import mmap
import multiprocessing as mp
import re
import sqlite3
import struct
import tarfile
import zipfile
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import BinaryIO, Callable, Iterable
from xml.etree import ElementTree

from loggers import get_logger

logger = get_logger(__name__)

MAX_OUTLINE_CHARS = 2000
# The inline route's text cap for documents (open-document.ts).
MAX_TEXT_CHARS = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 1024
MAX_XML_BYTES = 10 * 1024 * 1024
LISTED_MEMBERS = 40
LISTED_VALUES = 12
CELL_CHARS = 80
SAMPLE_ROWS = 5
SCANNED_MEMBERS = 1000
SMALL_MEMBER_BYTES = 1500
MEMBER_TEXT_CHARS = 1200
PAST_OUTLINE = "runs past what an outline shows"
# Reaching a tar member means decompressing everything before it.
MAX_SCANNED_BYTES = 64 * 1024 * 1024
# What sampling materialises: a whole stripe, batch, array, row group or long-string table.
MAX_SAMPLED_BYTES = 16 * 1024 * 1024
MAX_SAMPLED_CELLS = 2_000_000
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
# Pages without text add nothing to the text cap, so the walk has its own.
MAX_DOCUMENT_PAGES = 1000
# MuPDF inflates a zip member whole, to the size the central directory states.
MAX_DOCUMENT_MEMBER_BYTES = 128 * 1024 * 1024
_STATA_HEADER = {b"117": ("H", "I", "B"), b"118": ("H", "Q", "H"), b"119": ("I", "Q", "H")}
# The releases before 117 open with their own number and have no long strings.
_STATA_PLAIN_RELEASES = frozenset(range(102, 116))
_SAS_MAGIC = (
    b"\x00" * 12
    + b"\xc2\xea\x81\x60\xb3\x14\x11\xcf\xbd\x92\x08\x00\x09\xc7\x31\x8c\x18\x1f\x10\x11"
)
# ZipFile parses the whole central directory; real archives use about 80-170 bytes an entry.
MAX_ZIP_DIRECTORY_BYTES = 4 * 1024 * 1024
# octet_length measures a cell from the record header; older builds have only length(), for blobs.
SQLITE_MEASURES_UNREAD = sqlite3.sqlite_version_info >= (3, 43)
# What SQLite may allocate for one preview, which is the only bound on parsing a schema.
MAX_SQLITE_BYTES = 64 * 1024 * 1024
# MuPDF lays out one long paragraph in quadratic time: 128k characters take 18 s.
PREVIEW_TIMEOUT_SECONDS = 10.0
IMAGE_EXTS = set(
    ".psd .ico .icns .cur .tga .dds .pcx .ppm .pgm .pbm .pnm .qoi .jp2 .j2k .xbm .xpm .sgi .fits".split()
)
# Pillow picks a decoder from the bytes; naming the format keeps that format's checks in force.
PILLOW_FORMATS = {
    ".pgm": "PPM",
    ".pbm": "PPM",
    ".pnm": "PPM",
    ".jp2": "JPEG2000",
    ".j2k": "JPEG2000",
}
# Stored attachments are named by hash, so MuPDF is told the format instead of guessing.
MUPDF_FORMATS = dict.fromkeys((".docm", ".dotx", ".dotm"), "docx") | dict.fromkeys(
    (".potx", ".potm", ".ppsm"), "pptx"
)
PAGED_EXTS = {".xps": "Page", ".oxps": "Page", ".potx": "Slide", ".potm": "Slide", ".ppsm": "Slide"}
FLOWING_EXTS = {".epub", ".mobi", ".fb2", ".cbz", ".docm", ".dotx", ".dotm"}
DRAWING_EXTS = (".odp", ".odg", ".vsdx")
TAR_EXTS = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tbz", ".tar.xz", ".txz")
STREAMS = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open, ".lzma": lzma.open}
# Frame sizes: PNG 16 bytes past the signature, a JP2 "ihdr" box + 4, a codestream's SIZ + 8.
_FRAME_HEADERS = ((b"\x89PNG\r\n\x1a\n", 16), (b"ihdr", 4), (b"\xff\x4f\xff\x51", 8))
_TAR_COMPRESSION = ((b"\x1f\x8b", gzip.open), (b"BZh", bz2.open), (b"\xfd7zXZ\x00", lzma.open))
_HANDLED = IMAGE_EXTS | FLOWING_EXTS | {*PAGED_EXTS, *DRAWING_EXTS, *STREAMS}
_CTX = mp.get_context("spawn")


def preview_attachment(path: Path, filename: str) -> dict | None:
    """``{"kind": "image", "description", "image"}``, ``{"kind": "text", "label", "text"}`` or
    ``{"kind": "outline", "text"}``; None when not one of these, or unreadable in time."""
    name = filename.lower()
    ext = Path(name).suffix
    if not (ext in _HANDLED or ext in _OUTLINES or name.endswith(TAR_EXTS)):
        return None
    receive, send = _CTX.Pipe(duplex = False)
    worker = _CTX.Process(target = _send_preview, args = (send, path, filename), daemon = True)
    try:
        worker.start()
        send.close()
        if receive.poll(PREVIEW_TIMEOUT_SECONDS):
            return receive.recv()
        logger.info("chat_attachment_preview.timeout", extension = ext)
    except (OSError, EOFError) as exc:  # EOFError: the child died without answering
        logger.info("chat_attachment_preview.failed", extension = ext, error = type(exc).__name__)
    finally:
        if worker.pid is not None:
            worker.kill()
            worker.join()
        send.close()
        receive.close()
    return None


def _send_preview(send, path: Path, filename: str) -> None:
    # The limit can be lowered but never lifted, so it belongs here, not to the reader.
    with closing(sqlite3.connect(":memory:")) as database:
        database.execute(f"pragma hard_heap_limit = {MAX_SQLITE_BYTES}")
    send.send(build_preview(path, filename))


def build_preview(path: Path, filename: str) -> dict | None:
    """``preview_attachment`` in this process, with no deadline and no limit on SQLite."""
    name = filename.lower()
    ext = Path(name).suffix
    try:
        if ext in IMAGE_EXTS:
            return _image(path, ext)
        if ext in PAGED_EXTS or ext in FLOWING_EXTS:
            return _document(path, ext)
        if ext in DRAWING_EXTS:
            return _text(ext, _drawing_text(path, ext))
        if name.endswith(TAR_EXTS):
            return _outline(_tar_members(path))
        if ext in STREAMS:
            return _stream(path, ext, filename)
        builder = _OUTLINES.get(ext)
        return _outline(builder(path)) if builder else None
    except Exception as exc:  # noqa: BLE001 - no preview keeps the tool-only note
        logger.info(
            "chat_attachment_preview.failed",
            extension = ext,
            error = f"{type(exc).__name__}: {exc}"[:300],
        )
        return None


def _clip(text: str, limit: int, marker: str) -> str:
    # Names and values out of a file can hold lone surrogates, which JSON cannot encode.
    text = text.encode("utf-8", "replace").decode("utf-8")
    return text if len(text) <= limit else text[:limit] + f"\n[Truncated: {marker}]"


def _outline(text: str) -> dict:
    return {"kind": "outline", "text": _clip(text, MAX_OUTLINE_CHARS, "the outline is longer")}


def _text(ext: str, text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    return {
        "kind": "text",
        "label": ext[1:].upper(),
        "text": _clip(
            text, MAX_TEXT_CHARS, "the document has more text than one attachment carries"
        ),
    }


def _as_text(data: bytes, complete: bool = False) -> str | None:
    # A read cut mid-character is not an error; complete, the same bytes are another encoding.
    try:
        text = codecs.getincrementaldecoder("utf-8")().decode(data, final = complete)
    except UnicodeDecodeError:
        if not complete:
            return None
        text = data.decode("latin-1")
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in text)
    return text if text and printable >= 0.95 * len(text) else None


def _frame_sizes(path: Path, ext: str) -> Iterable[tuple[int, int]]:
    # Icons decode frames at the size the frame's own header states, not the directory's.
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access = mmap.ACCESS_READ) as data:
        if ext == ".icns":
            for marker, offset in _FRAME_HEADERS:
                start = data.find(marker)
                while start >= 0:
                    yield struct.unpack(
                        ">II", data[start + offset : start + offset + 8].ljust(8, b"\0")
                    )
                    start = data.find(marker, start + 1)
            return
        # ICO/CUR directory entries point at a PNG (IHDR sizes) or a BMP info header (signed sizes).
        (count,) = struct.unpack("<H", data[4:6])
        for entry in range(6, 6 + 16 * count, 16):
            (offset,) = struct.unpack("<I", data[entry + 12 : entry + 16])
            frame = data[offset : offset + 24].ljust(24, b"\0")
            if frame.startswith(b"\x89PNG"):
                yield struct.unpack(">II", frame[16:24])
            else:
                yield tuple(abs(side) for side in struct.unpack("<ii", frame[4:12]))


def _image(path: Path, ext: str) -> dict:
    import numpy as np
    from PIL import Image

    if ext in (".ico", ".cur", ".icns") and any(
        width * height > MAX_IMAGE_PIXELS for width, height in _frame_sizes(path, ext)
    ):
        return _outline(f"{ext[1:].upper()} icon with a frame over {MAX_IMAGE_PIXELS} pixels")
    with Image.open(path, formats = [PILLOW_FORMATS.get(ext, ext[1:].upper())]) as image:
        description = f"{image.format} image, {image.width}x{image.height}, mode {image.mode}"
        frames = getattr(image, "n_frames", 1)
        if frames > 1:
            description += f", {frames} frames"
        if image.width * image.height > MAX_IMAGE_PIXELS:
            return _outline(description)
        if image.format == "ICO":
            image.size = max(image.info.get("sizes", {image.size}))
        image.load()
        if image.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
            pixels = np.asarray(image, dtype = np.float64)
            span = float(pixels.max() - pixels.min()) or 1.0
            frame = Image.fromarray(((pixels - pixels.min()) * (255 / span)).astype(np.uint8))
        else:
            frame = image.convert(
                "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
            )
    frame.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    out = io.BytesIO()
    frame.save(out, format = "PNG")
    return {
        "kind": "image",
        "description": description,
        "image": "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii"),
    }


def _document(path: Path, ext: str) -> dict | None:
    import fitz

    # EPUB, CBZ, XPS and Office files are zips, whose directory MuPDF loads on open.
    try:
        _check_zip_directory(path)
    except _DirectoryTooLarge as exc:
        return _outline(str(exc))
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            largest = max(archive.infolist(), key = lambda info: info.file_size, default = None)
        if largest is not None and largest.file_size > MAX_DOCUMENT_MEMBER_BYTES:
            return _outline(f"{largest.filename} is {largest.file_size} bytes, too large to read")
    with fitz.open(path, filetype = MUPDF_FORMATS.get(ext, ext[1:])) as doc:
        info = doc.metadata or {}
        meta = ", ".join(f"{key}: {info[key]}" for key in ("title", "author") if info.get(key))
        heading = PAGED_EXTS.get(ext)
        pages, size, read = [], 0, 0
        for page in doc.pages(0, min(doc.page_count, MAX_DOCUMENT_PAGES)):
            read += 1
            text = page.get_text().strip()
            if text:
                pages.append(f"[{heading} {read}]\n{text}" if heading else text)
                size += len(text)
            if size > MAX_TEXT_CHARS:
                break
        unread = doc.page_count - read
        if not pages:
            scope = f", no text in the first {read}" if unread else " with no text"
            return _outline(f"{doc.page_count} pages{scope} (images only)")
    if unread and size <= MAX_TEXT_CHARS:
        pages.append(f"[Truncated: {unread} more pages not read]")
    return _text(ext, "\n\n".join([meta, *pages] if meta else pages))


class _DirectoryTooLarge(Exception):
    pass


def _check_zip_directory(path: Path) -> None:
    # _EndRecData sizes exactly the central directory ZipFile would parse.
    with path.open("rb") as handle:
        end = zipfile._EndRecData(handle)
    if end and end[zipfile._ECD_SIZE] > MAX_ZIP_DIRECTORY_BYTES:
        size = end[zipfile._ECD_SIZE]
        raise _DirectoryTooLarge(f"central directory of {size} bytes, too large to list")


def _open_zip(path: Path) -> zipfile.ZipFile:
    _check_zip_directory(path)
    return zipfile.ZipFile(path)


def _read_member(archive: zipfile.ZipFile, member: str, limit: int) -> bytes:
    # ZipFile caps one read's inflation for deflate only; LZMA and bzip2 expand its whole input.
    info = archive.getinfo(member)
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ValueError(f"{member} uses zip compression method {info.compress_type}")
    with archive.open(info) as handle:
        return handle.read(limit)


def _zip_xml(
    archive: zipfile.ZipFile,
    member: str,
    limit: int = MAX_XML_BYTES,
) -> ElementTree.Element:
    data = _read_member(archive, member, limit + 1)
    if len(data) > limit:
        raise ValueError(f"{member} is larger than {limit} bytes")
    return ElementTree.fromstring(data)


def _xml_text(root: ElementTree.Element, tags: set[str], budget: int) -> str:
    # A matching element nested in another repeats its text, so the budget bounds what is built.
    texts, cut = [], False
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] in tags and (text := "".join(element.itertext()).strip()):
            if budget <= 0:
                cut = True
                break
            texts.append(text[:budget])
            cut = cut or len(text) > budget
            # The separator it will be joined with counts too.
            budget -= len(texts[-1]) + 1
    return "\n".join(texts + [f"[Truncated: the text {PAST_OUTLINE}]"] * cut)


def _drawing_text(path: Path, ext: str) -> str:
    with _open_zip(path) as archive:
        if ext == ".vsdx":
            names = [n for n in archive.namelist() if re.fullmatch(r"visio/pages/page\d+\.xml", n)]
            pages = sorted(names, key = lambda n: int(re.search(r"\d+", n).group()))
            texts, room, budget = [], MAX_XML_BYTES, MAX_TEXT_CHARS + 1
            for name in pages:
                size = archive.getinfo(name).file_size
                if size > room:
                    break
                texts.append(_xml_text(_zip_xml(archive, name, room), {"Text"}, budget))
                room, budget = room - size, budget - len(texts[-1]) - 2
            unread = len(pages) - len(texts)
        else:
            unread, texts, budget = 0, [], MAX_TEXT_CHARS + 1
            root = _zip_xml(archive, "content.xml")
            drawn = [element for element in root.iter() if element.tag.endswith("}page")] or [root]
            for page in drawn:
                texts.append(_xml_text(page, {"p", "h"}, budget))
                budget -= len(texts[-1]) + 2
    heading = "Slide" if ext == ".odp" else "Page"
    text = "\n\n".join(
        f"[{heading} {number}]\n{text}" for number, text in enumerate(texts, start = 1) if text
    )
    return text + (f"\n[Truncated: {unread} more pages over the read budget]" if unread else "")


def _members(
    entries: Iterable[tuple[str, int]],
    read: Callable,
    complete: bool = True,
) -> str:
    entries = list(entries)
    lines = [f"{len(entries)}{'' if complete else '+'} entries:"]
    lines += [f"  {name} ({size} bytes)" for name, size in entries[:LISTED_MEMBERS]]
    if len(entries) > LISTED_MEMBERS:
        lines.append(f"  ... {len(entries) - LISTED_MEMBERS} more")
    budget = MEMBER_TEXT_CHARS
    small = [name for name, size in entries if 0 < size <= SMALL_MEMBER_BYTES]
    for name in small[:LISTED_MEMBERS]:
        if budget <= 0:
            break
        if (data := read(name)) and (text := _as_text(data)) and text.strip():
            text = text.strip()[:budget]
            lines.append(f"--- {name} ---\n{text}")
            budget -= len(text)
    return "\n".join(lines)


def _zip_members(path: Path) -> str:
    def read(name: str) -> bytes | None:
        # The listed size is the archive's claim; a member that inflates past it is skipped.
        try:
            data = _read_member(archive, name, SMALL_MEMBER_BYTES + 1)
        except ValueError:
            return None
        return data if len(data) <= SMALL_MEMBER_BYTES else None

    try:
        archive = _open_zip(path)
    except _DirectoryTooLarge as exc:
        return str(exc)
    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        return _members(((info.filename, info.file_size) for info in infos), read)


class _Capped(io.RawIOBase):
    """Ends the stream after ``left`` bytes, whatever the reader on top asks for."""

    def __init__(self, raw: BinaryIO, left: int) -> None:
        self.raw, self.left = raw, left

    def readinto(self, buffer) -> int:
        data = self.raw.read(min(len(buffer), self.left))
        self.left -= len(data)
        buffer[: len(data)] = data
        return len(data)


def _tar_members(path: Path) -> str:
    # tarfile reads extended headers into memory and skips member data by reading it, so the cap
    # bounds both. A name that is not UTF-8 reads better replaced than surrogate-escaped.
    with path.open("rb") as handle:
        magic = handle.read(6)
    opener = next((opener for prefix, opener in _TAR_COMPRESSION if magic.startswith(prefix)), open)
    entries, texts, complete = [], {}, True
    with opener(path, "rb") as raw:
        capped = _Capped(raw, MAX_SCANNED_BYTES)
        try:
            with tarfile.open(fileobj = capped, mode = "r|", errors = "replace") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    entries.append((member.name, member.size))
                    if member.size <= SMALL_MEMBER_BYTES and len(texts) < LISTED_MEMBERS:
                        texts[member.name] = archive.extractfile(member).read()
                    if len(entries) >= SCANNED_MEMBERS:
                        complete = False
                        break
        except (tarfile.ReadError, EOFError):
            complete = False
    # tarfile ends quietly on a header cut short after the first member.
    return _members(entries, texts.get, complete and capped.left > 0)


def _stream(path: Path, ext: str, filename: str) -> dict:
    with STREAMS[ext](path, "rb") as handle:
        data = handle.read(MAX_TEXT_CHARS + 1)
    inner = filename[: -len(ext)]
    # The cap counts bytes, so multi-byte text is cut before it reaches MAX_TEXT_CHARS characters.
    cut = len(data) > MAX_TEXT_CHARS
    text = _as_text(data[:MAX_TEXT_CHARS])
    if text is None:
        size = f"over {MAX_TEXT_CHARS}" if cut else str(len(data))
        return _outline(f"compressed {inner}: binary content, {size} bytes")
    if cut:
        text += f"\n[Truncated: {inner} is longer than one attachment carries]"
    return {"kind": "text", "label": ext[1:].upper(), "text": text}


def _cell(value) -> str:
    # A SAS reader hands back bytes for character columns as well as for binary ones.
    if isinstance(value, bytes):
        # Reading a cut value as a complete one mangles the cell, so it is the last resort.
        prefix = value[:SMALL_MEMBER_BYTES]
        text = _as_text(prefix, complete = len(value) <= SMALL_MEMBER_BYTES) or _as_text(
            prefix, complete = True
        )
        if text is None:
            return f"<{len(value)}-byte blob>"
    else:
        text = str(value)
    return text if len(text) <= CELL_CHARS else text[:CELL_CHARS] + "..."


def _frame(
    frame,
    rows: int | None,
    note: str = "",
) -> str:
    lines = [f"{'unknown' if rows is None else rows} rows x {frame.shape[1]} columns"]
    lines.append(
        "columns: "
        + ", ".join(f"{column} ({dtype})" for column, dtype in frame.dtypes.astype(str).items())
    )
    if len(frame):
        lines.append(f"first {len(frame)} rows:")
        lines += ["  " + "\t".join(map(_cell, row)) for row in frame.itertuples(index = False)]
    return "\n".join(lines + ([note] if note else []))


def _over_budget(schema, rows: int) -> bool:
    """Whether a sample would materialise more than the budget. A nested column states no size: an
    8 KB Feather file of 500 lists took 200 MB to sample, an ORC one 492 MB."""
    import pyarrow as pa
    return rows * len(schema) > MAX_SAMPLED_CELLS or any(
        pa.types.is_nested(field.type) for field in schema
    )


def _parquet(path: Path) -> str:
    import pyarrow.parquet as pq
    with pq.ParquetFile(path) as file:
        meta = file.metadata
        # What costs is one large value, and every row group states its uncompressed size.
        wanted, largest = SAMPLE_ROWS, 0
        for index in range(meta.num_row_groups):
            if wanted <= 0:
                break
            group = meta.row_group(index)
            largest = max(largest, group.total_byte_size)
            wanted -= group.num_rows
        if largest > MAX_SAMPLED_BYTES:
            note = f"[First rows not read: a row group of {largest} bytes]"
            return _frame(file.schema_arrow.empty_table().to_pandas(), meta.num_rows, note)
        batch = next(file.iter_batches(batch_size = SAMPLE_ROWS), None)
        table = batch if batch is not None else file.schema_arrow.empty_table()
        return _frame(table.to_pandas(), meta.num_rows)


def _table(path: Path, layout: str) -> str:
    """An Arrow IPC or ORC file: both materialise a whole batch or stripe, and state no size."""
    import pyarrow.dataset as ds

    data = ds.dataset(str(path), format = layout)
    size = path.stat().st_size
    # Counting walks every batch's metadata, which on a large file is the file itself.
    if size > MAX_SAMPLED_BYTES:
        note = f"[First rows not read: {size} bytes]"
        return _frame(data.schema.empty_table().to_pandas(), None, note)
    rows = data.count_rows()
    if _over_budget(data.schema, rows):
        note = f"[First rows not read: {rows} rows]"
        return _frame(data.schema.empty_table().to_pandas(), rows, note)
    return _frame(data.head(SAMPLE_ROWS).to_pandas(), rows)


def _arrow(path: Path) -> str:
    import pyarrow as pa
    try:
        return _table(path, "arrow")
    except pa.ArrowInvalid:
        # Feather v1: uncompressed, and not openable by the IPC reader, so its size is its cost.
        import pyarrow.feather as feather

        if path.stat().st_size > MAX_SAMPLED_BYTES:
            raise
        table = feather.read_table(path)
        return _frame(table.slice(0, SAMPLE_ROWS).to_pandas(), table.num_rows)


def _stata_tables(path: Path) -> tuple[int, int | None] | None:
    """What the file's map states its long-string and value-label tables hold: zero where the
    release has none, None where it states no size, and None for the pair when the header is not
    one this can walk, which a dataset label holding the same tags is why it is walked at all."""
    with path.open("rb") as handle:
        window = handle.read(8192)
    # A release opening with its own number has no long-string table, and no map for its labels.
    if window[:1] and window[0] in _STATA_PLAIN_RELEASES:
        return 0, None
    if not window.startswith(b"<stata_dta><header><release>"):
        return None
    widths = _STATA_HEADER.get(window[28:31])
    if widths is None:
        return None
    variables, observations, label = widths
    order = ">" if window[52:55] == b"MSF" else "<"
    at = 55 + len("</byteorder><K>") + struct.calcsize(variables) + len("</K><N>")
    at += struct.calcsize(observations) + len("</N><label>")
    if len(window) - at < struct.calcsize(label):
        return None
    (stated,) = struct.unpack_from(order + label, window, at)
    at += struct.calcsize(label) + stated + len("</label><timestamp>")
    if at >= len(window):
        return None
    at += 1 + window[at] + len("</timestamp></header><map>")
    if window[at - 5 : at] != b"<map>" or len(window) - at < 112:
        return None
    offsets = struct.unpack_from(f"{order}14Q", window, at)
    return max(offsets[11] - offsets[10], 0), max(offsets[12] - offsets[11], 0)


def _stata(path: Path) -> str:
    import pandas as pd

    # pandas decodes the long-string table whole on the first row read, and every value label with
    # it. Labels too large only leave the values as their codes; a long-string table leaves no rows.
    size = path.stat().st_size
    tables = _stata_tables(path)
    long_strings, value_labels = (None, None) if tables is None else tables
    if long_strings is None:
        if size > MAX_SAMPLED_BYTES:
            return f"header in no known layout, and {size} bytes to read through"
        long_strings = 0
    if long_strings > MAX_SAMPLED_BYTES:
        return f"long-string table of {long_strings} bytes, too large to read"
    labels = size if value_labels is None else value_labels
    note = (
        ""
        if labels <= MAX_SAMPLED_BYTES
        else f"[Value labels not read: {labels} bytes to read through]"
    )
    # StataReader counts observations only privately, so the outline leaves the count out.
    with pd.read_stata(path, iterator = True, convert_categoricals = not note) as reader:
        return _frame(reader.read(SAMPLE_ROWS), None, note)


def _sas_claimed_bytes(path: Path) -> int | None:
    """The largest read a sas7bdat header asks for, or None when it states none. Python reserves the
    buffer before pandas rejects it: a 5,120-byte file claiming a 64 MiB page costs 70 MB."""
    with path.open("rb") as handle:
        window = handle.read(288)
    if len(window) < 288 or not window.startswith(_SAS_MAGIC):
        return None
    order = "<" if window[37:38] == b"\x01" else ">"
    at = 196 + (4 if window[35:36] == b"3" else 0)
    header, page = struct.unpack_from(f"{order}2I", window, at)
    # pandas reads the rest of the header as `header - 288`, which below 288 is a read to the end.
    return None if header < 288 else max(header, page)


def _sas(path: Path, layout: str) -> str:
    import pandas as pd

    if layout == "sas7bdat":
        claimed = _sas_claimed_bytes(path)
        claimed = path.stat().st_size if claimed is None else claimed
        if claimed > MAX_SAMPLED_BYTES:
            return f"{claimed} bytes to read through before the first row"
    with pd.read_sas(path, format = layout, iterator = True, chunksize = SAMPLE_ROWS) as reader:
        rows = reader.nobs if layout == "xport" else reader.row_count
        frame = reader.read(SAMPLE_ROWS)
        named = _codec(getattr(reader, "inferred_encoding", None))
    if named is not None:
        # Character columns come back as bytes; the file names the encoding they are in.
        frame = frame.map(
            lambda value: value.decode(named, "replace") if isinstance(value, bytes) else value
        )
    return _frame(frame, rows)


def _codec(name: str | None) -> str | None:
    try:
        return codecs.lookup(name).name if name else None
    except LookupError:  # a SAS file can name an encoding by a code Python has no name for
        return None


def _array_line(name: str, array) -> str:
    import numpy as np

    line = f"{name}: shape {tuple(array.shape)}, dtype {array.dtype}"
    if array.size and array.dtype.kind in "biuf":
        line += f", min {array.min():.6g}, max {array.max():.6g}, mean {array.mean():.6g}"
    if array.size <= LISTED_VALUES:
        line += f", values {np.asarray(array).tolist()}"
    return line


def _array_header(handle: BinaryIO) -> str:
    """What an array too large to load says about itself, from the front of its .npy stream."""
    import numpy as np

    # numpy reads headers for 1.0 and 2.0 only, yet writes 3.0 for field names outside Latin-1.
    version = np.lib.format.read_magic(handle)
    length_format = "<H" if version[0] == 1 else "<I"
    (length,) = struct.unpack(length_format, handle.read(struct.calcsize(length_format)))
    if length > MAX_HEADER_BYTES:
        return f"header of {length} bytes, too large to read"
    header = ast.literal_eval(handle.read(length).decode("utf-8" if version[0] > 2 else "latin-1"))
    return f"shape {tuple(header['shape'])}, dtype {np.dtype(header['descr'])}"


def _npy(path: Path) -> str:
    import numpy as np
    if path.stat().st_size > MAX_SAMPLED_BYTES:
        with path.open("rb") as handle:
            return f"array: {_array_header(handle)}"
    # A wide structured dtype passes numpy's 10,000-byte header refusal in an otherwise small file.
    return _array_line("array", np.load(path, allow_pickle = False, max_header_size = MAX_HEADER_BYTES))


def _npz(path: Path) -> str:
    import numpy as np
    with _open_zip(path) as archive:
        names = [info.filename for info in archive.infolist() if info.filename.endswith(".npy")]
        lines = [f"{len(names)} arrays"]
        for name in names[:LISTED_MEMBERS]:
            label = name[: -len(".npy")]
            try:
                data = _read_member(archive, name, MAX_SAMPLED_BYTES + 1)
            except ValueError as exc:
                lines.append(f"{label}: {exc}")
                continue
            if len(data) > MAX_SAMPLED_BYTES:
                lines.append(f"{label}: {_array_header(io.BytesIO(data))}")
            else:
                lines.append(
                    _array_line(
                        label,
                        np.load(
                            io.BytesIO(data), allow_pickle = False, max_header_size = MAX_HEADER_BYTES
                        ),
                    )
                )
    return "\n".join(lines)


def _safetensors(path: Path) -> str:
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if length > MAX_HEADER_BYTES:
            return f"header of {length} bytes, too large to read"
        header = json.loads(handle.read(length))
    metadata = header.pop("__metadata__", None)
    parameters = sum(_count(entry["shape"]) for entry in header.values())
    lines = [
        f"{len(header)} tensors, {parameters} parameters"
        + (f", metadata {metadata}" if metadata else "")
    ]
    lines += [
        f"{name}: shape {tuple(entry['shape'])}, dtype {entry['dtype']}"
        for name, entry in list(header.items())[:LISTED_MEMBERS]
    ]
    return "\n".join(lines)


def _count(shape: Iterable[int]) -> int:
    total = 1
    for dimension in shape:
        total *= dimension
    return total


def _sqlite(path: Path) -> str:
    # Immutable: a hash-named copy nothing else writes, and one given no journal or lock files.
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    size = path.stat().st_size
    with closing(sqlite3.connect(uri, uri = True)) as database:
        # Opening one table parses every CREATE first, at a cost nothing in the file states and only
        # the preview process's limit bounds. An older SQLite reports none, leaving the file's size.
        row = database.execute("pragma hard_heap_limit").fetchone()
        bounded = row[0] if row else 0
        if not bounded and size > MAX_SAMPLED_BYTES:
            return f"database of {size} bytes, not read: nothing here bounds what its schema costs"
        try:
            return _sqlite_outline(database)
        except MemoryError:
            return f"database of {size} bytes, more than {bounded} bytes to read"


def _sqlite_outline(database: sqlite3.Connection) -> str:
    # Views and virtual tables compute their rows at an unstated cost. rootpage says which a table
    # is, having no b-tree of its own, unlike the schema text beside it, which can declare anything.
    tables = database.execute(
        "select name, rootpage = 0 from sqlite_master where type = 'table' order by name"
    ).fetchall()
    names = [name for name, _ in tables]
    lines = [f"{len(names)} tables:"] + [f"  {name}" for name in names[:LISTED_MEMBERS]]
    if len(names) > LISTED_MEMBERS:
        lines.append(f"  ... {len(names) - LISTED_MEMBERS} more")
    # The undetailed line is part of the outline, so without room for it a detail can end exactly
    # on the outline cap, leaving the cap's own marker silent.
    undetailed = "... %d more tables, not detailed"
    room = sum(len(line) + 1 for line in lines) + len(undetailed % len(tables)) + 1
    budget = MAX_OUTLINE_CHARS - room
    for detailed, (name, computed) in enumerate(tables):
        if budget <= 0:
            lines.append(undetailed % (len(tables) - detailed))
            break
        if computed:
            lines.append(f"{name}: a virtual table, not read")
        else:
            lines.append(_sqlite_table(database, name))
        budget -= len(lines[-1]) + 1
    return "\n".join(lines)


def _sqlite_table(database: sqlite3.Connection, name: str) -> str:
    quoted = _quoted(name)
    columns = [(row[1], row[2]) for row in database.execute(f"pragma table_info({quoted})")]
    (rows,) = database.execute(f"select count(*) from {quoted}").fetchone()
    lines = [f"{name} ({rows} rows): " + ", ".join(f"{column} {kind}" for column, kind in columns)]
    if not SQLITE_MEASURES_UNREAD:
        version = sqlite3.sqlite_version
        return lines[0] + f"\n  [First rows not read: SQLite {version} reads a cell to size it]"
    # A select loads every value whole, so one longer than a cell shows is described instead.
    sampled = ", ".join(
        f"case when octet_length({column}) > {SMALL_MEMBER_BYTES} then "
        f"'<' || typeof({column}) || ' of ' || octet_length({column}) || ' bytes>' else {column} end"
        for column in (_quoted(column) for column, _ in columns)
    )
    if sampled:
        lines += [
            "  " + "\t".join(map(_cell, row))
            for row in database.execute(f"select {sampled} from {quoted} limit {SAMPLE_ROWS}")
        ]
    return "\n".join(lines)


def _listed(items: list[str], total: int) -> str:
    """What fits on one line, and how many it leaves out."""
    return ", ".join(items) + (f", ... {total - len(items)} more" if total > len(items) else "")


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _model_member(
    archive: zipfile.ZipFile, member: str, main: str, relation: str
) -> tuple[str | None, str]:
    """The part the package declares it starts at - a 3MF in its relationships, a KMZ, having none,
    by the conventional name - or anything carrying the extension, and what went unread deciding."""
    names, note = archive.namelist(), ""
    if relation and "_rels/.rels" in names:
        stated = archive.getinfo("_rels/.rels").file_size
        if stated > MAX_XML_BYTES:
            note = f"[_rels/.rels not read: {stated} bytes of XML]\n"
        else:
            for related in _zip_xml(archive, "_rels/.rels"):
                target = related.get("Target", "").lstrip("/")
                if related.get("Type", "").endswith(relation) and target in names:
                    return target, note
    named = next((n for n in names if n.lower().endswith(member)), None)
    return (main if main in names else named), note


def _zip_model(
    path: Path,
    member: str,
    main: str,
    relation: str = "",
) -> str:
    """What the model or map inside a 3MF or KMZ states, over the listing of the container."""
    listing = _zip_members(path)
    try:
        archive = _open_zip(path)
    except _DirectoryTooLarge:
        return listing
    note = ""
    with archive:
        try:
            found, note = _model_member(archive, member, main, relation)
            if found is None:
                return note + listing
            # The member is parsed whole, so what it costs is the size the directory states for it.
            stated = archive.getinfo(found).file_size
            if stated > MAX_XML_BYTES:
                return f"{note}[{found} not read: {stated} bytes of XML]\n{listing}"
            root = _zip_xml(archive, found)
        except (ValueError, ElementTree.ParseError):
            return note + listing
    counts = Counter(element.tag.rsplit("}", 1)[-1] for element in root.iter())
    tags = [f"{count} {tag}" for tag, count in counts.most_common(LISTED_VALUES)]
    listed = _listed(tags, len(counts))
    # A 3MF titles itself in metadata elements, a KML names its places.
    names = _xml_text(root, {"metadata", "name"}, MEMBER_TEXT_CHARS).replace("\n", ", ")
    return f"{note}{found}: {listed}\n" + (f"named: {names}\n" if names else "") + listing


def _stl(path: Path) -> str:
    size = path.stat().st_size
    with path.open("rb") as handle:
        window = handle.read(MAX_OUTLINE_CHARS)
    header = window[:84]
    triangles = struct.unpack_from("<I", header, 80)[0] if len(header) == 84 else 0
    # Both layouts can open with "solid"; the binary count's 50-byte triangles fill the file.
    if size == 84 + 50 * triangles:
        # The 80-byte header holds a name padded out with nulls or spaces, which are not its text.
        described = (_as_text(header[:80].strip(b"\0 \t\r\n"), complete = True) or "").strip()
        return f"binary STL, {triangles} triangles" + (
            f", header {described!r}" if described else ""
        )
    # A binary header is 80 bytes of anything; only a wholly textual first line proves ASCII.
    text = _as_text(window, complete = True)
    first = text.splitlines()[0].strip() if text else ""
    if window[:5].lower() == b"solid" and first and first.isprintable():
        # A name that strips down short leaves the outline under the cap that would report the cut.
        cut = (
            ""
            if b"\n" in window or size <= len(window)
            else f"\n[Truncated: the name {PAST_OUTLINE}]"
        )
        return f"ASCII STL, {first!r}" + cut
    return f"STL of {size} bytes, in neither layout"


def _ply(path: Path) -> str:
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(MAX_OUTLINE_CHARS)
    # A PLY states its elements in the text header. The terminator is a line of its own; the same
    # words inside a comment end nothing.
    end = re.search(rb"(?m)^end_header\b", header)
    text = _as_text(header[: end.end() if end else len(header)], complete = True)
    if text is None:
        return f"PLY of {size} bytes, header not text"
    # Decoding and stripping both shorten what is shown, so the cap may never report the cut.
    cut = "" if end or size <= len(header) else f"\n[Truncated: the header {PAST_OUTLINE}]"
    return "PLY header:\n" + text.strip() + cut


def _glb(path: Path) -> str:
    with path.open("rb") as handle:
        magic, version, _ = struct.unpack("<4sII", handle.read(12))
        length, kind = struct.unpack("<I4s", handle.read(8))
        # The spec puts the scene first, as JSON, and the buffers it describes in the chunk behind.
        if magic != b"glTF" or kind != b"JSON":
            return f"GLB of {path.stat().st_size} bytes, in no glTF layout"
        if length > MAX_SAMPLED_BYTES:
            return f"glTF {version} binary, scene of {length} bytes, too large to read"
        scene = json.loads(handle.read(length))
    if not isinstance(scene, dict):
        return f"glTF {version} binary, a scene that is no glTF object"
    # Only what the spec has as a list is counted, and only what it has as an object is named.
    counted = ("scenes", "nodes", "meshes", "materials", "textures", "images", "animations")
    listed = [key for key in counted if isinstance(scene.get(key), list) and scene[key]]
    counts = ", ".join(f"{len(scene[key])} {key}" for key in listed)
    lines = [f"glTF {version} binary" + (f", {counts}" if counts else "")]
    asset = scene.get("asset")
    if isinstance(asset, dict) and (generator := asset.get("generator")):
        lines.append(f"generator: {generator}")
    meshes = scene["meshes"] if "meshes" in listed else []
    named = [mesh["name"] for mesh in meshes if isinstance(mesh, dict) and mesh.get("name")]
    if named:
        lines.append("meshes: " + _listed(named[:LISTED_VALUES], len(named)))
    return "\n".join(lines)


def _font(path: Path) -> str:
    import fitz

    size = path.stat().st_size
    # FreeType reads the whole font in, so the file's own size is what loading it costs.
    if size > MAX_SAMPLED_BYTES:
        return f"font of {size} bytes, too large to read"
    font = fitz.Font(fontfile = str(path))
    # Fixed pitch is the one trait MuPDF reads from the font's tables; it guesses the rest.
    pitch = ", monospaced" if font.flags.get("mono") else ""
    with path.open("rb") as handle:
        # MuPDF loads one face, so a collection is reported as the font it opens.
        collected = ", first font of a collection" if handle.read(4) == b"ttcf" else ""
    return f"{font.name}, {font.glyph_count} glyphs, {size} bytes{pitch}{collected}"


_OUTLINES: dict[str, Callable[[Path], str]] = {
    ext: builder
    for exts, builder in (
        (".zip .jar .whl .apk", _zip_members),
        (".3mf", lambda path: _zip_model(path, ".model", "3D/3dmodel.model", "3dmodel")),
        (".kmz", lambda path: _zip_model(path, ".kml", "doc.kml")),
        (".parquet", _parquet),
        (".feather .arrow", _arrow),
        (".orc", lambda path: _table(path, "orc")),
        (".dta", _stata),
        (".sas7bdat", lambda path: _sas(path, "sas7bdat")),
        (".xpt", lambda path: _sas(path, "xport")),
        (".npy", _npy),
        (".npz", _npz),
        (".safetensors", _safetensors),
        (".sqlite .sqlite3 .db .gpkg .mbtiles", _sqlite),
        (".stl", _stl),
        (".ply", _ply),
        (".glb", _glb),
        (".ttf .otf .ttc", _font),
    )
    for ext in exts.split()
}
