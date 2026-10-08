# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text from Office, OpenDocument, e-book, email and RTF files, using only the stdlib.

Each reader returns ``[(text, page_number), ...]``: one entry per slide (numbered), or
per sheet / chapter (unnumbered). Table rows are joined with " | ", as for .docx.
"""

from __future__ import annotations

import datetime as dt
import email
import email.policy
import math
import posixpath
import re
import struct
import zipfile
from xml.etree import ElementTree as ET

from .cfb import CompoundFile, CompoundFileError

Section = tuple[str, "int | None"]

# Bounds on what a small archive may expand to.
_MAX_MEMBER_BYTES = 128 * 1024 * 1024
_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_COLUMNS = 1024
_MAX_REPEAT = 1024

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

_BUILTIN_DATE_FORMATS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 45, 46, 47}


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
                # Skip the slide image placeholder's number field.
                note_lines = [l for l in _drawing_lines(zf.xml(notes)) if not l.strip().isdigit()]
                if note_lines:
                    lines += ["Notes:"] + note_lines
            if lines:
                sections.append(("\n".join(lines), number))
        return sections


# ---------------------------------------------------------------- OpenDocument


def _odf_inline(node: ET.Element) -> str:
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
                parts.append(" [" + " ".join(_odf_blocks(body)) + "]")
        elif tag in (_q("office", "annotation"), _q("text", "bookmark-ref")):
            pass
        else:
            parts.append(_odf_inline(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _odf_table(table: ET.Element) -> list[str]:
    rows = []
    for row in table.iter(_q("table", "table-row")):
        cells = []
        for cell in row:
            if cell.tag not in (_q("table", "table-cell"), _q("table", "covered-table-cell")):
                continue
            text = " ".join(_odf_blocks(cell))
            repeat = int(cell.get(_q("table", "number-columns-repeated"), "1") or 1)
            # Trailing empty cells repeat to the sheet edge; only content is expanded.
            cells += [text] * (min(repeat, _MAX_REPEAT) if text else min(repeat, 1))
            if len(cells) >= _MAX_COLUMNS:
                break
        line = _row(cells[:_MAX_COLUMNS])
        if line:
            repeat = int(row.get(_q("table", "number-rows-repeated"), "1") or 1)
            rows += [line] * min(repeat, _MAX_REPEAT)
    return rows


def _odf_blocks(node: ET.Element) -> list[str]:
    lines: list[str] = []
    for child in node:
        if child.tag in (_q("text", "p"), _q("text", "h")):
            text = _odf_inline(child)
            if text.strip():
                lines.append(text)
        elif child.tag == _q("table", "table"):
            lines += _odf_table(child)
        elif child.tag in (_q("office", "annotation"), _q("presentation", "notes")):
            continue
        else:
            lines += _odf_blocks(child)
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
        sheet_doc = body.find(_q("office", "spreadsheet"))
        slides_doc = body.find(_q("office", "presentation"))
        if sheet_doc is not None:
            for table in sheet_doc.findall(_q("table", "table")):
                rows = _odf_table(table)
                if rows:
                    name = table.get(_q("table", "name"), "")
                    sections.append((f"Sheet: {name}\n" + "\n".join(rows), None))
        elif slides_doc is not None:
            for number, page in enumerate(slides_doc.findall(_q("draw", "page")), 1):
                lines = _odf_blocks(page)
                notes = page.find(_q("presentation", "notes"))
                if notes is not None:
                    note_lines = _odf_blocks(notes)
                    if note_lines:
                        lines += ["Notes:"] + note_lines
                if lines:
                    sections.append(("\n".join(lines), number))
        else:
            lines = _odf_blocks(body)
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
            member = posixpath.normpath(posixpath.join(folder, item.get("href", "")))
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
    stack: list[tuple[bool, int]] = []
    skip, uc, to_skip = False, 1, 0
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
            stack.append((skip, uc))
            ignorable = False
        elif brace == "}":
            flush()
            skip, uc = stack.pop() if stack else (False, 1)
            to_skip = 0
        elif symbol is not None:
            if symbol == "*":
                ignorable = True
            elif not skip:
                flush()
                out.append(
                    {"~": "\u00a0", "_": "-", "-": "", "\n": "\n", "\r": "\n"}.get(symbol, symbol)
                )
        elif word is not None:
            if word == "bin":
                pos += int(arg or 0)
                continue
            if word == "ansicpg" and arg:
                codepage = f"cp{arg}"
                try:
                    b"".decode(codepage)
                except LookupError:
                    codepage = "cp1252"
            elif word == "uc" and arg:
                uc = int(arg)
            if ignorable or word in _RTF_SKIP:
                skip, ignorable = True, False
                continue
            if skip:
                continue
            if word == "u" and arg:
                flush()
                code = int(arg)
                out.append(chr(code + 65536 if code < 0 else code))
                to_skip = uc
            elif word in _RTF_BREAKS:
                flush()
                out.append(_RTF_BREAKS[word])
        elif hex_byte is not None:
            if not skip:
                pending.append(int(hex_byte, 16))
        elif text is not None and not skip:
            flush()
            out.append(text)
    flush()
    text = re.sub(r"( \| )+\n", "\n", "".join(out))
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

    # FibRgLw97.ccpText and FibRgFcLcb97.fcClx/lcbClx, located by the FIB's own counts.
    csw = struct.unpack_from("<H", word, 32)[0]
    lw_start = 34 + csw * 2 + 2
    ccp_text = struct.unpack_from("<i", word, lw_start + 3 * 4)[0]
    cslw = struct.unpack_from("<H", word, 34 + csw * 2)[0]
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

    chars: list[str] = []
    for start, end, fc in pieces:
        end = min(end, max(ccp_text, 0))
        if end <= start:
            continue
        count = end - start
        if fc & 0x40000000:
            offset = (fc & 0x3FFFFFFF) // 2
            chars.append(word[offset : offset + count].decode("cp1252", "replace"))
        else:
            chars.append(word[fc : fc + 2 * count].decode("utf-16-le", "replace"))
    return [(_word_text("".join(chars)), None)]


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
        elif kind == 0x000A:  # end of the workbook globals
            break

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
                r, c, _xf, v = struct.unpack_from("<HHHd", body)
                put(r, c, _number(v))
            elif kind == 0x027E:  # RK
                r, c, _xf, v = struct.unpack_from("<HHHI", body)
                put(r, c, _number(_rk(v)))
            elif kind == 0x00BD:  # MULRK
                r, c = struct.unpack_from("<HH", body)
                for i in range((len(body) - 6) // 6):
                    put(r, c + i, _number(_rk(struct.unpack_from("<I", body, 4 + i * 6 + 2)[0])))
            elif kind == 0x0204:  # LABEL
                r, c, _xf = struct.unpack_from("<HHH", body)
                put(r, c, _xls_short_string(body, 6, 2))
            elif kind == 0x0205:  # BOOLERR
                r, c, _xf, v, is_error = struct.unpack_from("<HHHBB", body)
                if not is_error:
                    put(r, c, "TRUE" if v else "FALSE")
            elif kind == 0x0006 and len(body) >= 14:  # FORMULA, cached result
                r, c, _xf = struct.unpack_from("<HHH", body)
                result = body[6:14]
                last_formula = None
                if result[6:8] != b"\xff\xff":
                    put(r, c, _number(struct.unpack("<d", result)[0]))
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


def _msg_prop(cf: CompoundFile, prefix: tuple[str, ...], prop: str) -> str | None:
    for kind, codec in (("001F", "utf-16-le"), ("001E", "cp1252")):
        name = f"__substg1.0_{prop}{kind}"
        if cf.exists(*prefix, name):
            return cf.open(*prefix, name).decode(codec, "replace").rstrip("\0")
    return None


def _msg_date(cf: CompoundFile) -> str | None:
    if not cf.exists("__properties_version1.0"):
        return None
    props = cf.open("__properties_version1.0")
    for off in range(32, len(props) - 15, 16):
        tag = struct.unpack_from("<I", props, off)[0]
        if tag >> 16 in (0x0039, 0x0E06) and tag & 0xFFFF == 0x0040:  # submit / delivery time
            ticks = struct.unpack_from("<Q", props, off + 8)[0]
            try:
                moment = dt.datetime(1601, 1, 1) + dt.timedelta(microseconds = ticks // 10)
            except OverflowError:
                return None
            return moment.isoformat(" ", "minutes") + " UTC"
    return None


def msg(path: str, html_text) -> list[Section]:
    cf = _compound(path)
    if not any(name.lower().startswith("__substg1.0_") for name in cf.listdir()):
        raise ValueError("not an Outlook message")
    headers = {label: (_msg_prop(cf, (), prop) or "").strip() for prop, label in _MSG_HEADERS}
    address = (_msg_prop(cf, (), "0C1F") or "").strip()
    # Exchange senders carry an X.500 path, not an address.
    if address and not address.startswith("/") and address != headers["From"]:
        headers["From"] = f"{headers['From']} <{address}>".strip()
    headers["Date"] = _msg_date(cf) or ""
    lines = [
        f"{label}: {headers[label]}"
        for label in ("From", "To", "Cc", "Date", "Subject")
        if headers[label]
    ]
    body = _msg_prop(cf, (), "1000")
    if not body and cf.exists("__substg1.0_10130102"):
        body = html_text(cf.open("__substg1.0_10130102"))
    attachments = []
    for name in cf.listdir():
        if name.lower().startswith("__attach_version1.0_"):
            label = _msg_prop(cf, (name,), "3707") or _msg_prop(cf, (name,), "3704")
            if label:
                attachments.append(label)
    if attachments:
        lines.append("Attachments: " + ", ".join(sorted(attachments)))
    body = (body or "").replace("\r\n", "\n").strip()
    return [("\n".join(lines) + "\n\n" + body, None)]
