# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Previews of chat attachments only the python tool can read, so the message already carries what
the file holds: images become a PNG, documents their text, and other files a short outline.

Every reader takes headers, schemas, the first rows, listings or small members, never the whole
payload, so a large file costs about as much as a small one.
"""

from __future__ import annotations

import base64
import io
import mmap
import multiprocessing as mp
import re
import struct
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

from loggers import get_logger

logger = get_logger(__name__)

MAX_OUTLINE_CHARS = 2000
# The inline route's text cap for documents (open-document.ts).
MAX_TEXT_CHARS = 10 * 1024 * 1024
MAX_IMAGE_SIDE = 1024
MAX_XML_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
# Pages without text add nothing to the text cap, so the walk has its own.
MAX_DOCUMENT_PAGES = 1000
# MuPDF inflates a zip member whole, to the size the central directory states.
MAX_DOCUMENT_MEMBER_BYTES = 128 * 1024 * 1024
# ZipFile parses the whole central directory; real archives use about 80-170 bytes an entry.
MAX_ZIP_DIRECTORY_BYTES = 4 * 1024 * 1024
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
# Frame sizes: PNG 16 bytes past the signature, a JP2 "ihdr" box + 4, a codestream's SIZ + 8.
_FRAME_HEADERS = ((b"\x89PNG\r\n\x1a\n", 16), (b"ihdr", 4), (b"\xff\x4f\xff\x51", 8))
_HANDLED = IMAGE_EXTS | FLOWING_EXTS | {*PAGED_EXTS, *DRAWING_EXTS}
_CTX = mp.get_context("spawn")


def preview_attachment(path: Path, filename: str) -> dict | None:
    """``{"kind": "image", "description", "image"}``, ``{"kind": "text", "label", "text"}`` or
    ``{"kind": "outline", "text"}``; None when not one of these, or unreadable in time."""
    name = filename.lower()
    ext = Path(name).suffix
    if ext not in _HANDLED:
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
    send.send(build_preview(path, filename))


def build_preview(path: Path, filename: str) -> dict | None:
    """``preview_attachment`` in this process, with no deadline."""
    name = filename.lower()
    ext = Path(name).suffix
    try:
        if ext in IMAGE_EXTS:
            return _image(path, ext)
        if ext in PAGED_EXTS or ext in FLOWING_EXTS:
            return _document(path, ext)
        if ext in DRAWING_EXTS:
            return _text(ext, _drawing_text(path, ext))
        return None
    except Exception as exc:  # noqa: BLE001 - no preview keeps the tool-only note
        logger.info(
            "chat_attachment_preview.failed",
            extension = ext,
            error = f"{type(exc).__name__}: {exc}"[:300],
        )
        return None


def _clip(text: str, limit: int, marker: str) -> str:
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
    texts = []
    for element in root.iter():
        if budget <= 0:
            break
        if element.tag.rsplit("}", 1)[-1] in tags and (text := "".join(element.itertext()).strip()):
            texts.append(text[:budget])
            # The separator it will be joined with counts too.
            budget -= len(texts[-1]) + 1
    return "\n".join(texts)


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
