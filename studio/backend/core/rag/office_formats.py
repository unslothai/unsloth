# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text from Office, OpenDocument, e-book, email and RTF files, using only the stdlib.

Each reader returns ``[(text, page_number), ...]``: one entry per slide (numbered), or
per sheet / chapter (unnumbered). Table rows are joined with " | ", as for .docx.
"""

from __future__ import annotations

import codecs
import datetime as dt
import email
import email.policy
import math
import posixpath
import re
import struct
import zipfile
from urllib.parse import unquote
from xml.etree import ElementTree as ET

from .cfb import CompoundFile, CompoundFileError

Section = tuple[str, "int | None"]

# Bounds on what a small archive may expand to.
_MAX_MEMBER_BYTES = 128 * 1024 * 1024
_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_COLUMNS = 1024
_MAX_REPEAT = 1024
# Text a document may add by repeating cells and rows.
_MAX_REPEATED_CHARS = 32 * 1024 * 1024

_NS = {
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "presentation": "urn:oasis:names:tc:opendocument:xmlns:presentation:1.0",
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
}
# ISO Strict OOXML names the same schemas with other URIs.
_STRICT_NS = tuple(
    (f"http://purl.oclc.org/ooxml/{strict}".encode(), _NS[prefix].encode())
    for strict, prefix in (
        ("spreadsheetml/main", "s"),
        ("presentationml/main", "p"),
        ("drawingml/main", "a"),
        ("officeDocument/relationships", "r"),
    )
)


def _q(prefix: str, tag: str) -> str:
    return f"{{{_NS[prefix]}}}{tag}"


def _row(cells: list[str]) -> str:
    while cells and not cells[-1].strip():
        cells.pop()
    return " | ".join(cells) if any(c.strip() for c in cells) else ""


def _number(value: float) -> str:
    if math.isfinite(value) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


# ---------------------------------------------------------------- zip archives


class _Archive:
    def __init__(self, path: str):
        try:
            self._zip = zipfile.ZipFile(path)
        except zipfile.BadZipFile as exc:
            raise ValueError("file is not a valid archive") from exc
        self._names = {name.lower(): name for name in self._zip.namelist()}
        self._budget = _MAX_TOTAL_BYTES

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self._zip.close()

    def has(self, name: str) -> bool:
        return name.lower() in self._names

    def read(self, name: str) -> bytes:
        real = self._names.get(name.lower())
        if real is None:
            raise ValueError(f"missing archive member: {name}")
        if self._zip.getinfo(real).flag_bits & 0x1:
            raise ValueError("file is password protected")
        limit = min(_MAX_MEMBER_BYTES, self._budget)
        with self._zip.open(real) as member:
            data = member.read(limit + 1)
        if len(data) > limit:
            raise ValueError("archive member is too large")
        self._budget -= len(data)
        return data

    def xml(self, name: str) -> ET.Element:
        data = self.read(name)
        # Office XML never declares entities; refusing them rules out expansion attacks.
        if b"<!ENTITY" in data:
            raise ValueError("XML entity declarations are not supported")
        if b"purl.oclc.org/ooxml/" in data:
            for strict, transitional in _STRICT_NS:
                data = data.replace(strict, transitional)
        try:
            return ET.fromstring(data)
        except ET.ParseError as exc:
            raise ValueError(f"malformed XML in {name}") from exc

    def rels(self, part: str) -> dict[str, str]:
        """Relationship id -> member name for an OOXML part."""
        folder, base = posixpath.split(part)
        rels_name = posixpath.join(folder, "_rels", base + ".rels")
        if not self.has(rels_name):
            return {}
        out = {}
        for rel in self.xml(rels_name).iter(_q("rel", "Relationship")):
            target = rel.get("Target", "")
            if rel.get("TargetMode") == "External" or not target:
                continue
            path = target.lstrip("/") if target.startswith("/") else posixpath.join(folder, target)
            out[rel.get("Id", "")] = posixpath.normpath(path)
        return out


# ---------------------------------------------------------------- .xlsx / .xlsm

# Built-in date and time format IDs, including the CJK (27-36, 50-58) and Thai (71-81) ones.
_BUILTIN_DATE_FORMATS = {
    *range(14, 23),
    *range(27, 37),
    *range(45, 48),
    *range(50, 59),
    *range(71, 82),
}


def _is_date_format(code: str) -> bool:
    code = re.sub(r'"[^"]*"|\[[^\]]*\]|\\.', "", code).lower()
    return bool(re.search(r"[dy]", code)) or ("m" in code and ("h" in code or "s" in code))


def _serial_date(serial: float, date1904: bool) -> str:
    base = dt.datetime(1904, 1, 1) if date1904 else dt.datetime(1899, 12, 30)
    try:
        moment = base + dt.timedelta(days = serial)
    except (OverflowError, ValueError):
        return _number(serial)
    return moment.date().isoformat() if moment.time() == dt.time() else moment.isoformat(" ")


def _column_index(ref: str) -> int:
    index = 0
    for ch in ref:
        if not ch.isalpha():
            break
        index = index * 26 + (ord(ch.upper()) - 64)
    return index - 1


def _xlsx_text(node: ET.Element) -> str:
    # Plain <t> or rich runs <r><t>; phonetic hints (<rPh>) are skipped.
    parts = []
    for child in node:
        if child.tag == _q("s", "t"):
            parts.append(child.text or "")
        elif child.tag == _q("s", "r"):
            t = child.find(_q("s", "t"))
            parts.append((t.text or "") if t is not None else "")
    return "".join(parts)


def xlsx(path: str) -> list[Section]:
    with _Archive(path) as zf:
        workbook_part = "xl/workbook.xml"
        workbook = zf.xml(workbook_part)
        pr = workbook.find(_q("s", "workbookPr"))
        date1904 = pr is not None and pr.get("date1904") in ("1", "true")
        rels = zf.rels(workbook_part)

        strings: list[str] = []
        if zf.has("xl/sharedStrings.xml"):
            strings = [_xlsx_text(si) for si in zf.xml("xl/sharedStrings.xml").iter(_q("s", "si"))]

        date_styles: list[bool] = []
        if zf.has("xl/styles.xml"):
            styles = zf.xml("xl/styles.xml")
            custom = {
                int(f.get("numFmtId", "-1")): f.get("formatCode", "")
                for f in styles.iter(_q("s", "numFmt"))
            }
            xfs = styles.find(_q("s", "cellXfs"))
            for xf in xfs.findall(_q("s", "xf")) if xfs is not None else []:
                fmt = int(xf.get("numFmtId", "0") or 0)
                date_styles.append(
                    fmt in _BUILTIN_DATE_FORMATS or _is_date_format(custom.get(fmt, ""))
                )

        sections: list[Section] = []
        for sheet in workbook.iter(_q("s", "sheet")):
            target = rels.get(sheet.get(_q("r", "id"), ""))
            if not target or not zf.has(target):
                continue
            rows = []
            for row in zf.xml(target).iter(_q("s", "row")):
                cells: list[str] = []
                for c in row.findall(_q("s", "c")):
                    col = _column_index(c.get("r", "")) if c.get("r") else len(cells)
                    if col < 0 or col >= _MAX_COLUMNS:
                        continue
                    kind, v = c.get("t", "n"), c.find(_q("s", "v"))
                    raw = v.text if v is not None and v.text is not None else ""
                    if kind == "s":
                        value = (
                            strings[int(raw)] if raw.isdigit() and int(raw) < len(strings) else ""
                        )
                    elif kind == "inlineStr":
                        inline = c.find(_q("s", "is"))
                        value = _xlsx_text(inline) if inline is not None else ""
                    elif kind == "b":
                        value = "TRUE" if raw == "1" else "FALSE"
                    elif kind in ("str", "e"):
                        value = raw
                    else:
                        try:
                            number = float(raw)
                        except ValueError:
                            value = raw
                        else:
                            style = int(c.get("s", "0") or 0)
                            is_date = style < len(date_styles) and date_styles[style]
                            value = _serial_date(number, date1904) if is_date else _number(number)
                    cells.extend([""] * (col - len(cells)))
                    if col == len(cells):
                        cells.append(value)
                line = _row(cells)
                if line:
                    rows.append(line)
            if rows:
                sections.append((f"Sheet: {sheet.get('name', '')}\n" + "\n".join(rows), None))
        return sections


# ---------------------------------------------------------------- .pptx


def _drawing_lines(root: ET.Element) -> list[str]:
    """Paragraphs of a DrawingML part in document order; table rows as one line each."""
    lines: list[str] = []
    in_table: set[ET.Element] = set()
    for node in root.iter():
        if node.tag == _q("a", "tbl"):
            for tr in node.iter(_q("a", "tr")):
                cells = []
                for tc in tr.findall(_q("a", "tc")):
                    texts = (_drawing_paragraph(p) for p in tc.iter(_q("a", "p")))
                    cells.append(" ".join(t for t in texts if t.strip()))
                line = _row(cells)
                if line:
                    lines.append(line)
            in_table.update(node.iter(_q("a", "p")))
        elif node.tag == _q("a", "p") and node not in in_table:
            text = _drawing_paragraph(node)
            if text.strip():
                lines.append(text)
    return lines


def _drawing_paragraph(p: ET.Element) -> str:
    parts = []
    for node in p.iter():
        if node.tag == _q("a", "t"):
            parts.append(node.text or "")
        elif node.tag == _q("a", "br"):
            parts.append("\n")
    return "".join(parts)


def _without_slide_number(root: ET.Element) -> ET.Element:
    """Drops the notes page's slide number placeholder."""
    ph_path = f"{_q('p', 'nvSpPr')}/{_q('p', 'nvPr')}/{_q('p', 'ph')}"
    doomed = [
        (parent, sp)
        for parent in root.iter()
        for sp in parent.findall(_q("p", "sp"))
        if (ph := sp.find(ph_path)) is not None and ph.get("type") == "sldNum"
    ]
    for parent, sp in doomed:
        parent.remove(sp)
    return root


def pptx(path: str) -> list[Section]:
    with _Archive(path) as zf:
        presentation_part = "ppt/presentation.xml"
        presentation = zf.xml(presentation_part)
        rels = zf.rels(presentation_part)
        slide_ids = presentation.find(_q("p", "sldIdLst"))
        sections: list[Section] = []
        for number, sld in enumerate(
            slide_ids.findall(_q("p", "sldId")) if slide_ids is not None else [], 1
        ):
            part = rels.get(sld.get(_q("r", "id"), ""))
            if not part or not zf.has(part):
                continue
            lines = _drawing_lines(zf.xml(part))
            notes = next(
                (t for t in zf.rels(part).values() if "notesslide" in t.lower() and zf.has(t)),
                None,
            )
            if notes:
                note_lines = _drawing_lines(_without_slide_number(zf.xml(notes)))
                if note_lines:
                    lines += ["Notes:"] + note_lines
            if lines:
                sections.append(("\n".join(lines), number))
        return sections


# ---------------------------------------------------------------- OpenDocument


class _Budget:
    """Caps the text repeated cells and rows add, which their source size does not bound."""

    def __init__(self):
        self.left = _MAX_REPEATED_CHARS

    def spend(self, chars: int) -> None:
        self.left -= chars
        if self.left < 0:
            raise ValueError("document repeats too much content")


def _odf_inline(node: ET.Element, budget: _Budget) -> str:
    parts = [node.text or ""]
    for child in node:
        tag = child.tag
        if tag == _q("text", "s"):
            parts.append(" " * min(int(child.get(_q("text", "c"), "1") or 1), 100))
        elif tag == _q("text", "tab"):
            parts.append("\t")
        elif tag == _q("text", "line-break"):
            parts.append("\n")
        elif tag == _q("text", "note"):
            body = child.find(_q("text", "note-body"))
            if body is not None:
                parts.append(" [" + " ".join(_odf_blocks(body, budget)) + "]")
        elif tag == _q("office", "annotation"):
            pass
        else:
            parts.append(_odf_inline(child, budget))
        parts.append(child.tail or "")
    return "".join(parts)


_ODF_ROW_GROUPS = frozenset(
    _q("table", tag) for tag in ("table-header-rows", "table-rows", "table-row-group")
)


def _odf_rows(node: ET.Element):
    """Rows of this table, through row groups but not into nested tables."""
    for child in node:
        if child.tag == _q("table", "table-row"):
            yield child
        elif child.tag in _ODF_ROW_GROUPS:
            yield from _odf_rows(child)


def _odf_repeat(node: ET.Element, attribute: str, text: str, room: int, budget: _Budget) -> int:
    repeat = int(node.get(_q("table", attribute), "1") or 1)
    # Trailing empty cells repeat to the sheet edge; only content is expanded.
    count = max(min(repeat, _MAX_REPEAT, room) if text else min(repeat, 1), 0)
    budget.spend(len(text) * max(count - 1, 0))
    return count


_ODF_VALUE_ATTRIBUTES = {
    "float": "value",
    "percentage": "value",
    "currency": "value",
    "date": "date-value",
    "time": "time-value",
    "boolean": "boolean-value",
    "string": "string-value",
}


def _odf_cell_value(cell: ET.Element) -> str:
    """A cell's typed value, for cells written without a text paragraph."""
    kind = cell.get(_q("office", "value-type"), "")
    attribute = _ODF_VALUE_ATTRIBUTES.get(kind)
    value = cell.get(_q("office", attribute), "") if attribute else ""
    if kind == "boolean" and value:
        return "TRUE" if value.lower() == "true" else "FALSE"
    if attribute == "value" and value:
        try:
            return _number(float(value))
        except ValueError:
            pass
    return value


def _odf_table(table: ET.Element, budget: _Budget) -> list[str]:
    rows = []
    for row in _odf_rows(table):
        cells = []
        for cell in row:
            if cell.tag not in (_q("table", "table-cell"), _q("table", "covered-table-cell")):
                continue
            text = " ".join(_odf_blocks(cell, budget)) or _odf_cell_value(cell)
            room = _MAX_COLUMNS - len(cells)
            cells += [text] * _odf_repeat(cell, "number-columns-repeated", text, room, budget)
            if len(cells) >= _MAX_COLUMNS:
                break
        line = _row(cells)
        if line:
            rows += [line] * _odf_repeat(row, "number-rows-repeated", line, _MAX_REPEAT, budget)
    return rows


# Annotations and notes are read separately; tracked changes hold deleted text.
_ODF_SKIP = frozenset(
    (_q("office", "annotation"), _q("presentation", "notes"), _q("text", "tracked-changes"))
)


def _odf_blocks(node: ET.Element, budget: _Budget) -> list[str]:
    lines: list[str] = []
    for child in node:
        if child.tag in (_q("text", "p"), _q("text", "h")):
            text = _odf_inline(child, budget)
            if text.strip():
                lines.append(text)
        elif child.tag == _q("table", "table"):
            lines += _odf_table(child, budget)
        elif child.tag in _ODF_SKIP:
            continue
        else:
            lines += _odf_blocks(child, budget)
    return lines


def opendocument(path: str) -> list[Section]:
    with _Archive(path) as zf:
        manifest = "META-INF/manifest.xml"
        if zf.has(manifest) and b"encryption-data" in zf.read(manifest):
            raise ValueError("file is password protected")
        body = zf.xml("content.xml").find(_q("office", "body"))
        if body is None:
            return []
        sections: list[Section] = []
        budget = _Budget()
        sheet_doc = body.find(_q("office", "spreadsheet"))
        slides_doc = body.find(_q("office", "presentation"))
        if sheet_doc is not None:
            for table in sheet_doc.findall(_q("table", "table")):
                rows = _odf_table(table, budget)
                if rows:
                    name = table.get(_q("table", "name"), "")
                    sections.append((f"Sheet: {name}\n" + "\n".join(rows), None))
        elif slides_doc is not None:
            for number, page in enumerate(slides_doc.findall(_q("draw", "page")), 1):
                lines = _odf_blocks(page, budget)
                notes = page.find(_q("presentation", "notes"))
                if notes is not None:
                    note_lines = _odf_blocks(notes, budget)
                    if note_lines:
                        lines += ["Notes:"] + note_lines
                if lines:
                    sections.append(("\n".join(lines), number))
        else:
            lines = _odf_blocks(body, budget)
            if lines:
                sections.append(("\n".join(lines), None))
        return sections


# ---------------------------------------------------------------- .epub


def epub(path: str, html_text) -> list[Section]:
    """Chapters in spine order; ``html_text`` turns one XHTML document into text."""
    with _Archive(path) as zf:
        container = zf.xml("META-INF/container.xml")
        rootfile = next(container.iter(_q("container", "rootfile")), None)
        if rootfile is None or not rootfile.get("full-path"):
            raise ValueError("epub has no package document")
        opf_path = rootfile.get("full-path")
        opf = zf.xml(opf_path)
        folder = posixpath.dirname(opf_path)
        manifest = {item.get("id"): item for item in opf.iter(_q("opf", "item")) if item.get("id")}
        sections: list[Section] = []
        for ref in opf.iter(_q("opf", "itemref")):
            item = manifest.get(ref.get("idref"))
            if item is None or "html" not in (item.get("media-type") or ""):
                continue
            # The navigation document only repeats the chapter titles.
            if "nav" in (item.get("properties") or "").split():
                continue
            href = unquote(item.get("href", "").split("#", 1)[0])
            member = posixpath.normpath(posixpath.join(folder, href))
            if not zf.has(member):
                continue
            text = html_text(zf.read(member)).strip()
            if text:
                sections.append((text, None))
        return sections


# ---------------------------------------------------------------- .eml


def eml(path: str, html_text) -> list[Section]:
    with open(path, "rb") as f:
        message = email.message_from_binary_file(f, policy = email.policy.default)
    return [(_email_text(message, html_text), None)]


def _email_text(message, html_text) -> str:
    lines = []
    for header in ("From", "To", "Cc", "Date", "Subject"):
        value = message.get(header)
        if value:
            lines.append(f"{header}: {value}")
    body = message.get_body(preferencelist = ("plain", "html"))
    text = ""
    if body is not None:
        try:
            content = body.get_content()
        except (LookupError, ValueError):
            content = body.get_payload(decode = True) or b""
            content = content.decode("utf-8", "replace") if isinstance(content, bytes) else content
        if body.get_content_subtype() == "html":
            text = html_text(content.encode("utf-8") if isinstance(content, str) else content)
        else:
            text = content
    names = [part.get_filename() for part in message.iter_attachments() if part.get_filename()]
    if names:
        lines.append("Attachments: " + ", ".join(names))
    return "\n".join(lines) + "\n\n" + text.strip()


# ---------------------------------------------------------------- .rtf

_RTF_TOKEN = re.compile(
    r"\\([a-zA-Z]{1,32})(-?\d{1,10})? ?|\\'([0-9a-fA-F]{2})|\\(.)|([{}])|[\r\n]+|([^\\{}\r\n]+)",
    re.S,
)
# Destinations whose content is never body text.
_RTF_SKIP = frozenset(
    """
    fonttbl colortbl stylesheet info pict object header headerl headerr headerf footer footerl
    footerr footerf listtable listoverridetable rsidtbl generator filetbl revtbl themedata
    colorschememapping latentstyles datastore xmlnstbl mmathpr pgdsctbl fldinst bkmkstart
    bkmkend picprop shppict nonshppict userprops docvar xform template
    """.split()
)
_RTF_BREAKS = {
    "par": "\n",
    "line": "\n",
    "sect": "\n\n",
    "page": "\n\n",
    "row": "\n",
    "tab": "\t",
    "cell": " | ",
    "emdash": "\u2014",
    "endash": "\u2013",
    "bullet": "\u2022",
    "lquote": "\u2018",
    "rquote": "\u2019",
    "ldblquote": "\u201c",
    "rdblquote": "\u201d",
    "enspace": " ",
    "emspace": " ",
    "qmspace": " ",
}


def rtf(path: str) -> list[Section]:
    with open(path, "rb") as f:
        data = f.read().decode("latin-1")
    if not data.lstrip().startswith("{\\rtf"):
        raise ValueError("file is not RTF")
    return [(_rtf_text(data), None)]


def _rtf_text(data: str) -> str:
    out: list[str] = []
    pending = bytearray()
    codepage = "cp1252"
    stack: list[tuple[bool, int, bool]] = []
    skip, uc, to_skip = False, 1, 0
    deleted = False  # tracked deletion
    ignorable = False

    def flush():
        if pending:
            out.append(pending.decode(codepage, "replace"))
            pending.clear()

    pos = 0
    while pos < len(data):
        m = _RTF_TOKEN.match(data, pos)
        if m is None:
            pos += 1
            continue
        pos = m.end()
        word, arg, hex_byte, symbol, brace, text = m.groups()
        if to_skip and (hex_byte or text or symbol):
            # Fallback characters after \uN.
            if text and len(text) > to_skip:
                text, to_skip = text[to_skip:], 0
            else:
                to_skip -= len(text) if text else 1
                continue
        if brace == "{":
            flush()
            stack.append((skip, uc, deleted))
            ignorable = False
        elif brace == "}":
            flush()
            skip, uc, deleted = stack.pop() if stack else (False, 1, False)
            to_skip = 0
        elif symbol is not None:
            if symbol == "*":
                ignorable = True
            elif not (skip or deleted):
                flush()
                out.append(
                    {"~": "\u00a0", "_": "-", "-": "", "\n": "\n", "\r": "\n"}.get(symbol, symbol)
                )
        elif word is not None:
            if word == "bin":
                # Raw bytes follow; a negative length would rewind and loop forever.
                pos = min(len(data), pos + max(int(arg or 0), 0))
                continue
            if word == "ansicpg" and arg:
                codepage = f"cp{arg}"
                try:
                    b"".decode(codepage)
                except LookupError:
                    codepage = "cp1252"
            elif word == "uc" and arg:
                uc = int(arg)
            elif word == "deleted":
                deleted = arg != "0"
            elif word == "plain":
                deleted = False
            if ignorable or word in _RTF_SKIP:
                skip, ignorable = True, False
                continue
            if skip:
                continue
            if word == "u" and arg:
                flush()
                code = int(arg)
                if not deleted:
                    out.append(chr(code + 65536 if code < 0 else code))
                to_skip = uc
            elif word in _RTF_BREAKS:
                flush()
                out.append(_RTF_BREAKS[word])
        elif hex_byte is not None:
            if not (skip or deleted):
                pending.append(int(hex_byte, 16))
        elif text is not None and not (skip or deleted):
            # Literal 8-bit text is in the document code page, like \'hh escapes.
            pending += text.encode("latin-1")
    flush()
    # \uN gives astral characters as two surrogates; pair them, and replace any left alone.
    text = "".join(out).encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    text = re.sub(r"( \| )+\n", "\n", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


# ---------------------------------------------------------------- compound files


def _compound(path: str) -> CompoundFile:
    with open(path, "rb") as f:
        data = f.read()
    try:
        return CompoundFile(data)
    except CompoundFileError as exc:
        raise ValueError(str(exc)) from exc


def _clean_control(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0e-\x1f]", "", text)


# ---------------------------------------------------------------- .doc


def doc(path: str) -> list[Section]:
    cf = _compound(path)
    try:
        word = cf.open("WordDocument")
    except CompoundFileError as exc:
        raise ValueError("not a Word 97-2003 document") from exc
    if len(word) < 0x200 or struct.unpack_from("<H", word, 0)[0] != 0xA5EC:
        raise ValueError("not a Word 97-2003 document")
    flags = struct.unpack_from("<H", word, 0x0A)[0]
    if flags & 0x0100:
        raise ValueError("file is password protected")
    table = cf.open("1Table" if flags & 0x0200 else "0Table")

    # FibRgLw97 story lengths and FibRgFcLcb97.fcClx/lcbClx, located by the FIB's own counts.
    csw = struct.unpack_from("<H", word, 32)[0]
    lw_start = 34 + csw * 2 + 2
    cslw = struct.unpack_from("<H", word, 34 + csw * 2)[0]
    # ccpText, ccpFtn, ccpHdd, ccpMcr, ccpAtn, ccpEdn, ccpTxbx
    count = max(min(cslw, 10) - 3, 1)
    ccps = [max(n, 0) for n in struct.unpack_from(f"<{count}i", word, lw_start + 12)]
    ccps += [0] * (7 - len(ccps))
    fc_start = lw_start + cslw * 4 + 2
    fc_clx, lcb_clx = struct.unpack_from("<II", word, fc_start + 33 * 8)
    clx = table[fc_clx : fc_clx + lcb_clx]

    pos, pieces = 0, None
    while pos < len(clx):
        if clx[pos] == 0x01:  # Prc: skip its property modifiers
            pos += 3 + struct.unpack_from("<H", clx, pos + 1)[0]
        elif clx[pos] == 0x02:  # Pcdt: the piece table
            size = struct.unpack_from("<I", clx, pos + 1)[0]
            plc = clx[pos + 5 : pos + 5 + size]
            n = (len(plc) - 4) // 12
            cps = struct.unpack_from(f"<{n + 1}I", plc, 0)
            pieces = [
                (cps[i], cps[i + 1], struct.unpack_from("<I", plc, (n + 1) * 4 + i * 8 + 2)[0])
                for i in range(n)
            ]
            break
        else:
            break
    if pieces is None:
        raise ValueError("Word document has no piece table")

    def story(first: int, length: int) -> str:
        chars: list[str] = []
        for start, end, fc in pieces:
            lo, hi = max(start, first), min(end, first + length)
            if hi <= lo:
                continue
            if fc & 0x40000000:
                offset = (fc & 0x3FFFFFFF) // 2 + (lo - start)
                chars.append(word[offset : offset + hi - lo].decode("cp1252", "replace"))
            else:
                offset = fc + 2 * (lo - start)
                chars.append(word[offset : offset + 2 * (hi - lo)].decode("utf-16-le", "replace"))
        return _word_text("".join(chars))

    # Stories follow the main text in this order; headers and comments are left out, as for .docx.
    starts = [sum(ccps[:i]) for i in range(len(ccps))]
    text, ftn, _hdd, _mcr, _atn, edn, txbx = range(7)
    parts = [story(0, ccps[text]), story(starts[txbx], ccps[txbx])]
    for label, kind in (("Footnotes", ftn), ("Endnotes", edn)):
        notes = story(starts[kind], ccps[kind])
        if notes:
            parts.append(f"{label}:\n{notes}")
    return [("\n\n".join(p for p in parts if p), None)]


def _word_text(raw: str) -> str:
    # Fields: keep the result (after 0x14), drop the instruction (before it).
    out, depth_in_code = [], []
    for ch in raw:
        if ch == "\x13":
            depth_in_code.append(True)
        elif ch == "\x14":
            if depth_in_code:
                depth_in_code[-1] = False
        elif ch == "\x15":
            if depth_in_code:
                depth_in_code.pop()
        elif not any(depth_in_code):
            out.append(ch)
    # A cell ends in 0x07; a row ends in one more.
    text = "".join(out).replace("\x07\x07", "\n").replace("\x07", " | ")
    text = text.translate({0x0D: "\n", 0x0B: "\n", 0x0C: "\n", 0x1E: "-", 0x1F: None})
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _clean_control(text).strip()


# ---------------------------------------------------------------- .xls


def _rk(value: int) -> float:
    if value & 2:
        number = float(value >> 2 if not value & 0x80000000 else (value >> 2) - (1 << 30))
    else:
        number = struct.unpack("<d", struct.pack("<Q", (value & 0xFFFFFFFC) << 32))[0]
    return number / 100 if value & 1 else number


class _Records:
    def __init__(
        self,
        data: bytes,
        pos: int = 0,
    ):
        self.data, self.pos = data, pos

    def __iter__(self):
        data = self.data
        while self.pos + 4 <= len(data):
            kind, size = struct.unpack_from("<HH", data, self.pos)
            body = data[self.pos + 4 : self.pos + 4 + size]
            self.pos += 4 + size
            yield kind, body


def _xls_string(segments: list[bytes], seg: int, pos: int) -> tuple[str, int, int]:
    """One XLUnicodeRichExtendedString, which may continue across CONTINUE records."""

    def take(n):
        nonlocal seg, pos
        out = b""
        while n > 0:
            if pos >= len(segments[seg]):
                seg, pos = seg + 1, 0
                if seg >= len(segments):
                    raise ValueError("truncated string table")
            chunk = segments[seg][pos : pos + n]
            out += chunk
            pos += len(chunk)
            n -= len(chunk)
        return out

    if pos >= len(segments[seg]):
        seg, pos = seg + 1, 0
    count = struct.unpack("<H", take(2))[0]
    flags = take(1)[0]
    runs = struct.unpack("<H", take(2))[0] if flags & 0x08 else 0
    ext = struct.unpack("<I", take(4))[0] if flags & 0x04 else 0
    parts, wide = [], flags & 0x01
    while count > 0:
        if pos >= len(segments[seg]):
            # A continued string restates its width in the CONTINUE record's first byte.
            seg, pos = seg + 1, 0
            if seg >= len(segments):
                raise ValueError("truncated string table")
            wide = segments[seg][0] & 0x01
            pos = 1
        width = 2 if wide else 1
        avail = (len(segments[seg]) - pos) // width
        n = min(count, max(avail, 1))
        raw = segments[seg][pos : pos + n * width]
        pos += n * width
        parts.append(raw.decode("utf-16-le" if wide else "latin-1", "replace"))
        count -= n
    take(4 * runs + ext)
    return "".join(parts), seg, pos


def _xls_short_string(body: bytes, pos: int, length_bytes: int) -> str:
    count = body[pos] if length_bytes == 1 else struct.unpack_from("<H", body, pos)[0]
    pos += length_bytes
    wide = body[pos] & 0x01
    pos += 1
    raw = body[pos : pos + count * (2 if wide else 1)]
    return raw.decode("utf-16-le" if wide else "latin-1", "replace")


def xls(path: str) -> list[Section]:
    cf = _compound(path)
    stream = next((name for name in ("Workbook", "Book") if cf.exists(name)), None)
    if stream is None:
        raise ValueError("not an Excel 97-2003 workbook")
    data = cf.open(stream)

    sheets: list[tuple[str, int]] = []
    strings: list[str] = []
    sst: list[bytes] | None = None  # SST body and its CONTINUE records
    sst_count = 0
    formats: dict[int, str] = {}
    xf_formats: list[int] = []
    date1904 = False
    for kind, body in _Records(data):
        if sst is not None and kind != 0x003C:
            seg = pos = 0
            for _ in range(sst_count):
                text, seg, pos = _xls_string(sst, seg, pos)
                strings.append(text)
            sst = None
        if kind == 0x002F:
            raise ValueError("file is password protected")
        if kind == 0x0085 and len(body) >= 8 and body[5] == 0:  # BOUNDSHEET, worksheet
            sheets.append((_xls_short_string(body, 6, 1), struct.unpack_from("<I", body, 0)[0]))
        elif kind == 0x00FC and len(body) >= 8:  # SST
            sst_count, sst = struct.unpack_from("<I", body, 4)[0], [body[8:]]
        elif kind == 0x003C and sst is not None:
            sst.append(body)
        elif kind == 0x041E and len(body) >= 5:  # FORMAT
            formats[struct.unpack_from("<H", body)[0]] = _xls_short_string(body, 2, 2)
        elif kind == 0x00E0 and len(body) >= 4:  # XF
            xf_formats.append(struct.unpack_from("<H", body, 2)[0])
        elif kind == 0x0022 and len(body) >= 2:  # DATEMODE
            date1904 = struct.unpack_from("<H", body)[0] == 1
        elif kind == 0x000A:  # end of the workbook globals
            break
    date_xfs = {
        i
        for i, fmt in enumerate(xf_formats)
        if fmt in _BUILTIN_DATE_FORMATS or _is_date_format(formats.get(fmt, ""))
    }

    def number(value: float, xf: int) -> str:
        return _serial_date(value, date1904) if xf in date_xfs else _number(value)

    sections: list[Section] = []
    for name, offset in sheets:
        cells: dict[int, dict[int, str]] = {}
        last_formula: tuple[int, int] | None = None

        def put(r, c, value):
            if c < _MAX_COLUMNS:
                cells.setdefault(r, {})[c] = value

        depth = 0
        for kind, body in _Records(data, offset):
            # Embedded charts nest their own BOF/EOF inside the sheet.
            if kind == 0x0809:
                depth += 1
            elif kind == 0x000A:
                depth -= 1
                if depth <= 0:
                    break
            if len(body) < 6 and kind not in (0x0207,):
                continue
            if kind == 0x00FD:  # LABELSST
                r, c, _xf, i = struct.unpack_from("<HHHI", body)
                put(r, c, strings[i] if i < len(strings) else "")
            elif kind == 0x0203:  # NUMBER
                r, c, xf, v = struct.unpack_from("<HHHd", body)
                put(r, c, number(v, xf))
            elif kind == 0x027E:  # RK
                r, c, xf, v = struct.unpack_from("<HHHI", body)
                put(r, c, number(_rk(v), xf))
            elif kind == 0x00BD:  # MULRK
                r, c = struct.unpack_from("<HH", body)
                for i in range((len(body) - 6) // 6):
                    xf, v = struct.unpack_from("<HI", body, 4 + i * 6)
                    put(r, c + i, number(_rk(v), xf))
            elif kind == 0x0204:  # LABEL
                r, c, _xf = struct.unpack_from("<HHH", body)
                put(r, c, _xls_short_string(body, 6, 2))
            elif kind == 0x0205:  # BOOLERR
                r, c, _xf, v, is_error = struct.unpack_from("<HHHBB", body)
                if not is_error:
                    put(r, c, "TRUE" if v else "FALSE")
            elif kind == 0x0006 and len(body) >= 14:  # FORMULA, cached result
                r, c, xf = struct.unpack_from("<HHH", body)
                result = body[6:14]
                last_formula = None
                if result[6:8] != b"\xff\xff":
                    put(r, c, number(struct.unpack("<d", result)[0], xf))
                elif result[0] == 0:
                    last_formula = (r, c)
                elif result[0] == 1:
                    put(r, c, "TRUE" if result[2] else "FALSE")
            elif kind == 0x0207 and last_formula and len(body) >= 3:  # STRING after FORMULA
                put(*last_formula, _xls_short_string(body, 0, 2))
                last_formula = None
        rows = []
        for r in sorted(cells):
            row = cells[r]
            line = _row([row.get(c, "") for c in range(max(row) + 1)])
            if line:
                rows.append(line)
        if rows:
            sections.append((f"Sheet: {name}\n" + "\n".join(rows), None))
    return sections


# ---------------------------------------------------------------- .ppt

_PPT_TEXT_CHARS, _PPT_TEXT_BYTES = 0x0FA0, 0x0FA8
_PPT_SLIDE_LIST, _PPT_SLIDE_PERSIST, _PPT_SLIDE = 0x0FF0, 0x03F3, 0x03EE


def ppt(path: str) -> list[Section]:
    cf = _compound(path)
    try:
        data = cf.open("PowerPoint Document")
    except CompoundFileError as exc:
        raise ValueError("not a PowerPoint 97-2003 presentation") from exc
    if cf.exists("EncryptedSummary"):
        raise ValueError("file is password protected")

    listed: list[list[str]] = []  # slide text from SlideListWithText (instance 0)
    drawn: list[list[str]] = []  # slide text from Slide containers, for files without it
    stack: list[tuple[int, int, int]] = []  # (end, type, instance)
    pos = 0
    while pos + 8 <= len(data):
        while stack and pos >= stack[-1][0]:
            stack.pop()
        ver_inst, kind, size = struct.unpack_from("<HHI", data, pos)
        if ver_inst & 0x000F == 0x000F:  # container: descend
            stack.append((pos + 8 + size, kind, ver_inst >> 4))
            if kind == _PPT_SLIDE:
                drawn.append([])
            pos += 8
            continue
        body = data[pos + 8 : pos + 8 + size]
        pos += 8 + size
        in_list = next((inst for _end, t, inst in reversed(stack) if t == _PPT_SLIDE_LIST), None)
        if kind == _PPT_SLIDE_PERSIST and in_list == 0:
            listed.append([])
        elif kind in (_PPT_TEXT_CHARS, _PPT_TEXT_BYTES):
            text = body.decode("utf-16-le" if kind == _PPT_TEXT_CHARS else "latin-1", "replace")
            text = _clean_control(text.replace("\r", "\n").replace("\x0b", "\n")).strip()
            if not text:
                continue
            if in_list == 0 and listed:
                listed[-1].append(text)
            elif in_list is None and any(t == _PPT_SLIDE for _e, t, _i in stack) and drawn:
                drawn[-1].append(text)
    slides = listed if any(listed) else drawn
    return [("\n".join(lines), n) for n, lines in enumerate(slides, 1) if lines]


# ---------------------------------------------------------------- .msg

_MSG_HEADERS = (("0C1A", "From"), ("0E04", "To"), ("0E03", "Cc"), ("0037", "Subject"))


# Windows code pages Python names differently from cpNNN.
_CODE_PAGES = {
    1200: "utf-16-le",
    1201: "utf-16-be",
    20127: "ascii",
    20866: "koi8_r",
    21866: "koi8_u",
    50220: "iso2022_jp",
    50221: "iso2022_jp",
    50222: "iso2022_jp",
    51932: "euc_jp",
    51949: "euc_kr",
    52936: "hz",
    54936: "gb18030",
    65000: "utf-7",
    65001: "utf-8",
    **{28590 + n: f"iso8859_{n}" for n in range(1, 16)},
}


def _code_page_codec(code_page: int | None) -> str:
    name = _CODE_PAGES.get(code_page, f"cp{code_page}") if code_page else "cp1252"
    try:
        codecs.lookup(name)
    except LookupError:
        return "cp1252"
    return name


def _msg_props(cf: CompoundFile) -> dict[int, bytes]:
    """Fixed-size top-level properties: tag -> 8-byte value."""
    if not cf.exists("__properties_version1.0"):
        return {}
    props = cf.open("__properties_version1.0")
    return {
        struct.unpack_from("<I", props, off)[0]: props[off + 8 : off + 16]
        for off in range(32, len(props) - 15, 16)
    }


def _msg_prop(cf: CompoundFile, prefix: tuple[str, ...], prop: str, codec: str) -> str | None:
    # String8 ("001E") values are in the message code page.
    for kind, kind_codec in (("001F", "utf-16-le"), ("001E", codec)):
        name = f"__substg1.0_{prop}{kind}"
        if cf.exists(*prefix, name):
            return cf.open(*prefix, name).decode(kind_codec, "replace").rstrip("\0")
    return None


def _msg_date(props: dict[int, bytes]) -> str | None:
    for prop in (0x0039, 0x0E06):  # submit / delivery time
        value = props.get((prop << 16) | 0x0040)
        if value is None:
            continue
        ticks = struct.unpack("<Q", value)[0]
        try:
            moment = dt.datetime(1601, 1, 1) + dt.timedelta(microseconds = ticks // 10)
        except OverflowError:
            return None
        return moment.isoformat(" ", "minutes") + " UTC"
    return None


# [MS-OXRTFCP] dictionary prefill for compressed RTF.
_RTF_PREFILL = (
    b"{\\rtf1\\ansi\\mac\\deff0\\deftab720{\\fonttbl;}{\\f0\\fnil \\froman \\fswiss "
    b"\\fmodern \\fscript \\fdecor MS Sans SerifSymbolArialTimes New RomanCourier"
    b"{\\colortbl\\red0\\green0\\blue0\r\n\\par \\pard\\plain\\f0\\fs20\\b\\i\\u\\tab\\tx"
)


def _decompress_rtf(data: bytes) -> bytes:
    """PidTagRtfCompressed ([MS-OXRTFCP]) to RTF bytes."""
    if len(data) < 16:
        raise ValueError("truncated compressed RTF")
    comp_size, raw_size, magic = struct.unpack_from("<III", data)
    if raw_size > _MAX_MEMBER_BYTES:
        raise ValueError("compressed RTF body is too large")
    if magic == 0x414C454D:  # "MELA": stored uncompressed
        return data[16 : 16 + raw_size]
    if magic != 0x75465A4C:  # "LZFu"
        raise ValueError("unknown compressed RTF format")
    window = bytearray(4096)
    window[: len(_RTF_PREFILL)] = _RTF_PREFILL
    write = len(_RTF_PREFILL)
    out = bytearray()
    pos, end = 16, min(len(data), comp_size + 4)
    while pos < end:
        control = data[pos]
        pos += 1
        for bit in range(8):
            # Stop at the declared size: back-references can expand far beyond it.
            if pos >= end or len(out) >= raw_size:
                return bytes(out[:raw_size])
            if control & (1 << bit):
                if pos + 2 > end:
                    break
                ref = (data[pos] << 8) | data[pos + 1]
                pos += 2
                offset, length = ref >> 4, (ref & 0xF) + 2
                if offset == write:  # end marker
                    return bytes(out)
                for k in range(length):
                    byte = window[(offset + k) % 4096]
                    out.append(byte)
                    window[write] = byte
                    write = (write + 1) % 4096
            else:
                out.append(data[pos])
                window[write] = data[pos]
                write = (write + 1) % 4096
                pos += 1
    return bytes(out[:raw_size])


def msg(path: str, html_text) -> list[Section]:
    cf = _compound(path)
    if not any(name.lower().startswith("__substg1.0_") for name in cf.listdir()):
        raise ValueError("not an Outlook message")
    props = _msg_props(cf)
    # PidTagMessageCodepage, else PidTagInternetCodepage.
    code_page = next(
        (
            struct.unpack_from("<I", props[tag])[0]
            for tag in (0x3FFD0003, 0x3FDE0003)
            if tag in props
        ),
        None,
    )
    codec = _code_page_codec(code_page)
    headers = {
        label: (_msg_prop(cf, (), prop, codec) or "").strip() for prop, label in _MSG_HEADERS
    }
    address = (_msg_prop(cf, (), "0C1F", codec) or "").strip()
    # Exchange senders carry an X.500 path, not an address.
    if address and not address.startswith("/") and address != headers["From"]:
        headers["From"] = f"{headers['From']} <{address}>".strip()
    headers["Date"] = _msg_date(props) or ""
    lines = [
        f"{label}: {headers[label]}"
        for label in ("From", "To", "Cc", "Date", "Subject")
        if headers[label]
    ]
    body = _msg_prop(cf, (), "1000", codec)
    if not body and cf.exists("__substg1.0_10130102"):
        body = html_text(cf.open("__substg1.0_10130102"))
    if not body and cf.exists("__substg1.0_10090102"):
        rtf_bytes = _decompress_rtf(cf.open("__substg1.0_10090102"))
        body = _rtf_text(rtf_bytes.decode("latin-1"))
    attachments = []
    for name in cf.listdir():
        if name.lower().startswith("__attach_version1.0_"):
            label = _msg_prop(cf, (name,), "3707", codec) or _msg_prop(cf, (name,), "3704", codec)
            if label:
                attachments.append(label)
    if attachments:
        lines.append("Attachments: " + ", ".join(sorted(attachments)))
    body = (body or "").replace("\r\n", "\n").strip()
    return [("\n".join(lines) + "\n\n" + body, None)]
