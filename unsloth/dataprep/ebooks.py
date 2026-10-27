# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Load EPUB books and, with Calibre installed, MOBI books on CPU.

Pass a Hugging Face fast tokenizer, not model weights. Token counts exclude
special tokens added later by a trainer; reserve space for those in chunk_size.
"""

from bisect import bisect_right
from html.parser import HTMLParser
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import tempfile
import unicodedata
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

__all__ = ["load_ebook_dataset"]

_MAX_MEMBER_BYTES = 32 * 1024 * 1024
_MAX_BOOK_BYTES = 256 * 1024 * 1024
_BLOCKS = {
    "p",
    "div",
    "section",
    "article",
    "blockquote",
    "li",
    "tr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "pre",
    "ul",
    "ol",
    "dl",
    "dt",
    "dd",
    "table",
    "caption",
    "figure",
    "figcaption",
    "header",
    "footer",
    "main",
    "aside",
    "address",
    "details",
    "summary",
}
_VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_IGNORE = {"head", "script", "style", "nav", "svg"}


class _BookTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs = True)
        self.parts = []
        self.stack = []
        self.encoding = None
        # <pre> text is kept verbatim and spliced back after whitespace cleanup.
        self.preformatted = []

    def handle_starttag(self, tag, attrs):
        if tag == "body":
            self.handle_endtag("head")
        attrs = dict(attrs)
        if tag == "meta" and self.encoding is None:
            self.encoding = attrs.get("charset")
            if not self.encoding and attrs.get("http-equiv", "").lower() == "content-type":
                charset = re.search(r"charset\s*=\s*[\"']?([\w-]+)", attrs.get("content", ""), re.I)
                if charset:
                    self.encoding = charset[1]
        hidden = (
            bool(self.stack and self.stack[-1][1])
            or tag in _IGNORE
            or "hidden" in attrs
            or attrs.get("aria-hidden", "").lower() == "true"
        )
        if not hidden:
            if tag in _BLOCKS or tag == "hr":
                self.parts.append("\n\n")
            elif tag == "br":
                self.parts.append("\n")
            elif tag in {"td", "th"}:
                self.parts.append(" ")
        if tag not in _VOID:
            self.stack.append((tag, hidden))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        hidden = bool(self.stack and self.stack[-1][1])
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break
        if not hidden and tag in _BLOCKS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.stack or not self.stack[-1][1]:
            if any(tag == "pre" for tag, _ in self.stack):
                self.parts.append(f"\ue000{len(self.preformatted)}\ue001")
                self.preformatted.append(data)
            else:
                # Source newlines inside an HTML paragraph render as spaces.
                self.parts.append(re.sub(r"\s+", " ", data))

    def unknown_decl(self, data):
        if data.startswith("CDATA["):
            self.handle_data(data[6:])


def _html_text(data):
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    elif data.startswith(b"\x00<"):
        encoding = "utf-16-be"
    elif data.startswith(b"<\x00"):
        encoding = "utf-16-le"
    else:
        declaration = re.match(
            rb'(?:\xef\xbb\xbf)?\s*<\?xml\b[^?]*\bencoding\s*=\s*["\']([\w-]+)', data[:1024], re.I
        )
        if declaration:
            encoding = declaration[1].decode("ascii")
        else:
            header = _BookTextParser()
            header.feed(data[:1024].decode("latin-1"))
            encoding = header.encoding or "utf-8-sig"
    parser = _BookTextParser()
    # A declared UTF-8 encoding keeps a leading BOM, which str.strip() leaves in place.
    parser.feed(data.decode(encoding).lstrip("\ufeff"))
    parser.close()
    text = unicodedata.normalize("NFC", "".join(parser.parts))
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return re.sub(
        "\ue000(\\d+)\ue001",
        lambda m: unicodedata.normalize("NFC", parser.preformatted[int(m[1])]),
        text,
    )


def _archive_path(base, href):
    url = urlsplit(href)
    if url.scheme or url.netloc or url.path.startswith("/") or "\\" in url.path:
        raise ValueError(f"Invalid EPUB resource path: {href!r}")
    name = posixpath.normpath(posixpath.join(base, unquote(url.path)))
    if name == ".." or name.startswith("../") or name.startswith("/"):
        raise ValueError(f"Invalid EPUB resource path: {href!r}")
    return name


def _read_epub(path, chapter_markers):
    with ZipFile(path) as archive:
        total = 0

        def read(name):
            nonlocal total
            info = archive.getinfo(name)
            total += info.file_size
            if info.file_size > _MAX_MEMBER_BYTES or total > _MAX_BOOK_BYTES:
                raise ValueError("EPUB exceeds the uncompressed size limit")
            return archive.read(info)

        container = ET.fromstring(read("META-INF/container.xml"))
        rootfile = container.find(".//{*}rootfile")
        if rootfile is None or not rootfile.get("full-path"):
            raise ValueError("EPUB container has no package document")
        package_path = _archive_path("", rootfile.get("full-path"))
        package = ET.fromstring(read(package_path))
        base = posixpath.dirname(package_path)
        manifest = package.find("{*}manifest")
        spine = package.find("{*}spine")
        if manifest is None or spine is None:
            raise ValueError("EPUB package is missing its manifest or reading order")
        items = {item.get("id"): item for item in manifest}
        encrypted = set()
        if "META-INF/encryption.xml" in archive.namelist():
            encryption = ET.fromstring(read("META-INF/encryption.xml"))
            encrypted = {
                _archive_path("", ref.get("URI", ""))
                for ref in encryption.iter("{http://www.w3.org/2001/04/xmlenc#}CipherReference")
            }
        chapters = []
        for ref in spine:
            if ref.get("linear", "yes") == "no":
                continue
            item = items.get(ref.get("idref"))
            if item is None:
                raise ValueError("EPUB reading order references a missing manifest item")
            if "nav" in item.get("properties", "").split():
                continue
            if item.get("media-type") not in {"application/xhtml+xml", "text/html"}:
                continue
            name = _archive_path(base, item.get("href", ""))
            if name in encrypted:
                raise ValueError("Encrypted/DRM-protected EPUB text is not supported")
            text = _html_text(read(name))
            if text:
                if chapter_markers:
                    text = "<|chapter_start|>\n" + text + "\n<|chapter_end|>"
                chapters.append(text)
        metadata = package.find("{*}metadata")
        fields = {"title": "title", "author": "creator", "genre": "subject"}
        values = {}
        for key, field in fields.items():
            elements = (
                []
                if metadata is None
                else metadata.findall("{http://purl.org/dc/elements/1.1/}" + field)
            )
            values[key] = "; ".join(
                " ".join("".join(element.itertext()).split()) for element in elements
            )
        return "\n\n".join(chapters), values


def _read_mobi(path, chapter_markers):
    converter = shutil.which("ebook-convert")
    if converter is None:
        raise ImportError(
            "MOBI support requires Calibre's ebook-convert on PATH. Install Calibre or use EPUB files."
        )
    with tempfile.TemporaryDirectory(prefix = "unsloth-ebook-") as directory:
        output = Path(directory) / "book.epub"
        subprocess.run(
            [converter, str(path), str(output)], check = True, capture_output = True, timeout = 120
        )
        return _read_epub(output, chapter_markers)


# Add formats here. Readers return (clean_text, {"title": str, "author": str, "genre": str}).
_READERS = {".epub": _read_epub, ".mobi": _read_mobi}


def _read_book(path, chapter_markers):
    try:
        return _READERS[path.suffix.lower()](path, chapter_markers)
    except (
        BadZipFile,
        ET.ParseError,
        LookupError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        raise ValueError(f"Could not read ebook '{path}': {exc}") from exc


def _chunk_text(text, tokenizer, chunk_size, overlap):
    encoded = tokenizer(
        text,
        add_special_tokens = False,
        return_offsets_mapping = True,
        return_attention_mask = False,
        truncation = False,
        verbose = False,
    )
    offsets = [(a, b) for a, b in encoded["offset_mapping"] if b > a]
    if not offsets:
        raise ValueError("The tokenizer produced no tokens for the ebook text")
    ends = [b for _, b in offsets]
    starts = [a for a, _ in offsets]
    boundaries = [
        m.end() for m in re.finditer(r'(?<=[.!?。！？])["\u201d\u2019\)\]]*\s+|\n\s*\n', text)
    ]
    start = 0
    char_start = 0
    while start < len(offsets):
        while char_start < len(text) and text[char_start].isspace():
            char_start += 1
        start = max(start, bisect_right(ends, char_start))
        if start == len(offsets):
            break
        end = min(start + chunk_size, len(offsets))
        char_end = len(text) if end == len(offsets) else ends[end - 1]
        if end < len(offsets):
            boundary_idx = bisect_right(boundaries, char_end) - 1
            if boundary_idx >= 0:
                boundary = boundaries[boundary_idx]
                boundary_token = bisect_right(ends, boundary)
                if (
                    boundary_token - start >= max(overlap + 1, chunk_size // 2)
                    and text[char_start:boundary].strip()
                ):
                    end, char_end = boundary_token, boundary
        # BPE offsets may share a Unicode character; re-encoding also checks the new left boundary.
        while end > start:
            chunk = text[char_start:char_end].strip()
            if chunk and len(tokenizer.encode(chunk, add_special_tokens = False)) <= chunk_size:
                break
            end -= 1
            char_end = ends[end - 1] if end > start else char_start
        if end == start:
            raise ValueError(
                "chunk_size is too small to represent a complete character with this tokenizer"
            )
        yield chunk
        # Consume every byte token of the final character, even when it spans the token window.
        consumed = bisect_right(ends, char_end)
        if consumed == len(offsets):
            break
        if overlap:
            start = max(consumed - overlap, bisect_right(starts, char_start))
            char_start = starts[start]
        else:
            start, char_start = consumed, char_end


def load_ebook_dataset(
    path,
    file_types = None,
    chunk_size = 4096,
    overlap = 256,
    *,
    tokenizer = None,
    include_metadata = False,
    chapter_markers = False,
):
    """Return a Hugging Face Dataset with a ``text`` column from a book or library.

    ``tokenizer`` must be a Hugging Face fast tokenizer (``use_fast=True``).
    ``chunk_size`` and ``overlap`` count its tokens, excluding trainer-added
    special tokens. Chunks prefer sentence/paragraph boundaries in the latter
    half of a window; overlap uses source-token offsets and may expand to keep
    Unicode characters intact. No model is loaded and nothing is downloaded.

    Directories are searched recursively in sorted path order. Each book is
    chunked separately and any unreadable book raises an error naming its path.
    ``file_types=None`` selects all formats in the internal ``_READERS`` mapping.
    EPUB uses the package's primary reading order, skipping navigation and
    non-linear content. MOBI requires Calibre's ``ebook-convert`` on PATH.
    DRM-protected text is unsupported. Images, CSS and scripts are discarded;
    paragraphs and chapter breaks remain. ``chapter_markers=True`` inserts
    ``<|chapter_start|>`` and ``<|chapter_end|>`` around each content document.

    ``include_metadata=True`` adds ``source`` (absolute original filename),
    ``chunk_id`` (zero-based within each book), ``title``, ``author`` and
    ``genre``. Missing metadata is an empty string; multiple values use ``; ``.

    Example::

        from unsloth import load_ebook_dataset
        dataset = load_ebook_dataset(
            "./my_library/", tokenizer=tokenizer,
            chunk_size=4096, overlap=256, include_metadata=True,
        )

    Reserve room for EOS/BOS or chat-template tokens in ``chunk_size`` when
    passing the text dataset to a trainer.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be a non-negative integer smaller than chunk_size")
    if tokenizer is None or not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "Pass a Hugging Face fast tokenizer (use_fast=True) to measure token counts without loading a model"
        )
    if isinstance(file_types, str):
        file_types = [file_types]
    extensions = (
        set(_READERS)
        if file_types is None
        else {"." + extension.lower().lstrip(".") for extension in file_types}
    )
    if not extensions or not extensions <= _READERS.keys():
        raise ValueError(
            f"file_types must select supported ebook formats: {', '.join(sorted(_READERS))}"
        )
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
        if not files:
            raise ValueError(f"No matching ebook files found in '{path}'")
    else:
        if path.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported ebook file type: '{path.suffix}'")
        files = [path]
    columns = {"text": []}
    if include_metadata:
        columns.update({key: [] for key in ("source", "chunk_id", "title", "author", "genre")})
    for book in files:
        text, metadata = _read_book(book, chapter_markers)
        if not text:
            raise ValueError(f"Ebook '{book}' contains no readable text")
        for index, chunk in enumerate(_chunk_text(text, tokenizer, chunk_size, overlap)):
            columns["text"].append(chunk)
            if include_metadata:
                for key, value in {"source": str(book), "chunk_id": index, **metadata}.items():
                    columns[key].append(value)
    from datasets import Dataset

    return Dataset.from_dict(columns)
