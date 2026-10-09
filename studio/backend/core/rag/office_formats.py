# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text from Office, OpenDocument, e-book, email and RTF files, using only the stdlib.

Each reader returns ``[(text, page_number), ...]``: one entry per slide (numbered), or
per sheet / chapter (unnumbered). Table rows are joined with " | ", as for .docx.
"""

from __future__ import annotations

import codecs
import datetime as dt
import decimal
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
# Parsed XML costs about 100-300 bytes per element, far more than its compressed size.
_MAX_XML_ELEMENTS = 2_000_000
_MAX_UNIT_ELEMENTS = 200_000  # one streamed row or string
_MAX_XML_DEPTH = 512
# Text a document may add by repeating cells and rows.
_MAX_REPEATED_CHARS = 32 * 1024 * 1024

_NS = {
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
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
        ("drawingml/chart", "c"),
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

    def _xml_events(self, name: str, events: tuple[str, ...]):
        data = self.read(name)
        # Office XML never declares entities; refusing them rules out expansion attacks.
        if b"<!ENTITY" in data:
            raise ValueError("XML entity declarations are not supported")
        if b"purl.oclc.org/ooxml/" in data:
            for strict, transitional in _STRICT_NS:
                data = data.replace(strict, transitional)
        parser = ET.XMLPullParser(events)
        try:
            for start in range(0, len(data), 1 << 20):
                parser.feed(data[start : start + (1 << 20)])
                yield from parser.read_events()
            parser.close()
            yield from parser.read_events()
        except ET.ParseError as exc:
            raise ValueError(f"malformed XML in {name}") from exc

    def xml(self, name: str) -> ET.Element:
        root, count = None, 0
        for _event, element in self._xml_events(name, ("start",)):
            root = element if root is None else root
            count += 1
            if count > _MAX_XML_ELEMENTS:
                raise ValueError(f"{name} is too large to index")
        if root is None:
            raise ValueError(f"malformed XML in {name}")
        return root

    def stream(self, name: str, units: frozenset[str]):
        """("start", element) outside units and ("unit", element) once a unit is complete.

        Handled elements are dropped, so memory holds one unit rather than the whole part.
        """
        stack: list[ET.Element] = []
        unit, count = None, 0
        for event, element in self._xml_events(name, ("start", "end")):
            if event == "start":
                stack.append(element)
                if len(stack) > _MAX_XML_DEPTH:
                    raise ValueError(f"{name} is nested too deeply")
                if unit is not None:
                    count += 1
                    if count > _MAX_UNIT_ELEMENTS:
                        raise ValueError(f"{name} is too large to index")
                elif element.tag in units:
                    unit, count = element, 0
                else:
                    yield "start", element
                continue
            stack.pop()
            if unit is not None and element is not unit:
                continue
            if element is unit:
                unit = None
                yield "unit", element
            if stack:
                # Earlier siblings are gone already, so this is the parent's first child.
                stack[-1].remove(element)

    def relationships(self, part: str) -> list[tuple[str, str, str]]:
        """(id, type, member name) of an OOXML part's relationships; "" for the package."""
        folder, base = posixpath.split(part)
        rels_name = posixpath.join(folder, "_rels", base + ".rels")
        if not self.has(rels_name):
            return []
        out = []
        for rel in self.xml(rels_name).iter(_q("rel", "Relationship")):
            target = rel.get("Target", "")
            if rel.get("TargetMode") == "External" or not target:
                continue
            path = target.lstrip("/") if target.startswith("/") else posixpath.join(folder, target)
            out.append((rel.get("Id", ""), rel.get("Type", ""), posixpath.normpath(path)))
        return out

    def rels(self, part: str) -> dict[str, str]:
        """Relationship id -> member name for an OOXML part."""
        return {rel_id: target for rel_id, _type, target in self.relationships(part)}

    def related(self, part: str, kind: str, default: str) -> str | None:
        """The part's first ``kind`` relationship target, else ``default`` if present."""
        for _id, rel_type, target in self.relationships(part):
            if rel_type.endswith("/" + kind) and self.has(target):
                return target
        return default if self.has(default) else None


# ---------------------------------------------------------------- .xlsx / .xlsm

# Built-in date and time format IDs, including the CJK (27-36, 50-58) and Thai (71-81) ones.
_BUILTIN_DATE_FORMATS = {
    *range(14, 23),
    *range(27, 37),
    *range(45, 48),
    *range(50, 59),
    *range(71, 82),
}


_BUILTIN_DURATION_FORMATS = {46}  # [h]:mm:ss
# Classified as openpyxl does: quoted text and locale tags aside, any date or time token.
_FORMAT_STRIP = re.compile(r'".*?"|\[(?!hh?\]|mm?\]|ss?\])[^\]]*\]')
_FORMAT_DURATION = re.compile(
    r"\[hh?\](:mm(:ss(\.0*)?)?)?|\[mm?\](:ss(\.0*)?)?|\[ss?\](\.0*)?", re.I
)


def _format_kind(format_id: int, code: str) -> str | None:
    """ "date" for dates and times of day, "duration" for elapsed time, else None."""
    if not code:
        if format_id in _BUILTIN_DURATION_FORMATS:
            return "duration"
        return "date" if format_id in _BUILTIN_DATE_FORMATS else None
    code = code.split(";")[0]
    if _FORMAT_DURATION.search(code):
        return "duration"
    return "date" if re.search(r"(?<![_\\])[dmhysDMHYS]", _FORMAT_STRIP.sub("", code)) else None


def _serial_text(serial: float, kind: str, date1904: bool) -> str:
    if kind == "duration":
        seconds = round(serial * 86_400)
        hours, rest = divmod(abs(seconds), 3600)
        return f"{'-' if seconds < 0 else ''}{hours}:{rest // 60:02}:{rest % 60:02}"
    day, fraction = divmod(serial, 1)
    clock = dt.timedelta(milliseconds = round(fraction * 86_400_000))
    if 0 <= serial < 1 and clock.days == 0:
        return (dt.datetime.min + clock).time().isoformat()
    if not date1904 and 0 < serial < 60:
        day += 1  # Excel counts a 29 February 1900 that never was.
    base = dt.datetime(1904, 1, 1) if date1904 else dt.datetime(1899, 12, 30)
    try:
        moment = base + dt.timedelta(days = day) + clock
    except (OverflowError, ValueError):
        return _number(serial)
    return moment.date().isoformat() if moment.time() == dt.time() else moment.isoformat(" ")


# Built-in numeric formats ([ECMA-376] 18.8.30); fractions and text are left as stored.
_BUILTIN_NUMBER_FORMATS = {
    1: "0",
    2: "0.00",
    3: "#,##0",
    4: "#,##0.00",
    9: "0%",
    10: "0.00%",
    11: "0.00E+00",
    37: "#,##0 ;(#,##0)",
    38: "#,##0 ;[Red](#,##0)",
    39: "#,##0.00;(#,##0.00)",
    40: "#,##0.00;[Red](#,##0.00)",
    48: "##0.0E+0",
}
_FORMAT_LITERALS = frozenset(" $-+/():!^&'~{}<>=\u20ac\u00a3\u00a5")


def _format_tokens(code: str) -> list[tuple[str, str]] | None:
    """("lit" | "ph" | "exp" | "sep", text) tokens of a number format, or None if unsupported."""
    tokens, i = [], 0
    while i < len(code):
        ch = code[i]
        if ch == '"':
            end = code.find('"', i + 1)
            end = len(code) if end < 0 else end
            tokens.append(("lit", code[i + 1 : end]))
            i = end + 1
        elif ch == "\\" and i + 1 < len(code):
            tokens.append(("lit", code[i + 1]))
            i += 2
        elif ch == "[":
            end = code.find("]", i)
            inner = code[i + 1 : end] if end > 0 else ""
            if inner.startswith("$"):  # currency and locale, such as [$EUR-407]
                tokens.append(("lit", inner[1:].split("-")[0]))
            elif not inner.isalnum():  # conditions such as [>100]
                return None
            i = end + 1  # colours are dropped
        elif ch == "_":  # space the width of the next character
            tokens.append(("lit", " "))
            i += 2
        elif ch == "*":  # fill with the next character
            i += 2
        elif ch in "0#?.,%":
            tokens.append(("ph", ch))
            i += 1
        elif ch in "Ee" and code[i + 1 : i + 2] in ("+", "-"):
            tokens.append(("exp", code[i + 1]))
            i += 2
        elif ch == ";":
            tokens.append(("sep", ";"))
            i += 1
        elif ch in _FORMAT_LITERALS:
            tokens.append(("lit", ch))
            i += 1
        else:
            return None
    return tokens


def _round(value: float, places: int) -> str:
    # Excel rounds halves away from zero.
    quantum = decimal.Decimal(1).scaleb(-places)
    return format(decimal.Decimal(repr(value)).quantize(quantum, decimal.ROUND_HALF_UP), "f")


def _group(digits: str) -> str:
    head = len(digits) % 3 or 3
    return ",".join([digits[:head]] + [digits[i : i + 3] for i in range(head, len(digits), 3)])


def _format_section(value: float, tokens: list[tuple[str, str]]) -> str | None:
    marks = [i for i, (kind, _) in enumerate(tokens) if kind in ("ph", "exp")]
    if not any(kind == "ph" and ch in "0#?" for kind, ch in tokens):
        return "".join(text for kind, text in tokens if kind == "lit")
    first, last = marks[0], marks[-1]
    if any(kind == "lit" for kind, _ in tokens[first:last]):
        return None
    prefix = "".join(text for kind, text in tokens[:first] if kind == "lit")
    suffix = "".join(text for kind, text in tokens[last + 1 :] if kind == "lit")
    pattern = "".join(ch if kind == "ph" else "E" + ch for kind, ch in tokens[first : last + 1])
    percent = "%" * pattern.count("%")
    value *= 100 ** len(percent)
    pattern = pattern.replace("%", "")
    mantissa, _, exponent = pattern.partition("E")
    sign_style, exponent = (exponent[:1], exponent[1:]) if exponent else ("", "")
    whole, dot, fraction = mantissa.partition(".")
    while whole.endswith(","):  # trailing commas scale by thousands
        whole, value = whole[:-1], value / 1000
    grouped, whole = "," in whole, whole.replace(",", "")
    places, required = sum(fraction.count(c) for c in "0#?"), fraction.count("0")
    min_whole = whole.count("0")
    power = 0
    if sign_style:
        width = max(len(whole), 1)
        if value:
            power = math.floor(math.log10(value))
            power = power - power % width if "#" in whole else power - (max(min_whole, 1) - 1)
        if float(_round(value / 10**power, places)) >= 10 ** max(width, min_whole, 1):
            power += width if "#" in whole else 1
        value /= 10**power
    digits = _round(value, places)
    int_part, _, frac_part = digits.partition(".")
    while len(frac_part) > required and frac_part.endswith("0"):
        frac_part = frac_part[:-1]
    int_part = "" if int_part == "0" and min_whole == 0 else int_part.zfill(min_whole)
    if grouped and int_part:
        int_part = _group(int_part)
    text = int_part + (dot + frac_part if dot else "")
    if sign_style:
        exp_sign = "-" if power < 0 else ("+" if sign_style == "+" else "")
        text += f"E{exp_sign}{str(abs(power)).zfill(exponent.count('0'))}"
    return prefix + text + percent + suffix


def _format_number(value: float, code: str) -> str | None:
    """``value`` as a numeric format displays it, or None for formats left as stored."""
    tokens = _format_tokens(code) if code else None
    if not tokens:
        return None
    sections: list[list[tuple[str, str]]] = [[]]
    for token in tokens:
        if token[0] == "sep":
            sections.append([])
        else:
            sections[-1].append(token)
    if value < 0 and len(sections) > 1 and sections[1]:
        return _format_section(-value, sections[1])
    if value == 0 and len(sections) > 2 and sections[2]:
        return _format_section(0.0, sections[2])
    text = _format_section(abs(value), sections[0])
    return None if text is None else ("-" + text if value < 0 else text)


def _cell_number(value: float, format_id: int, code: str, date1904: bool) -> str:
    """A numeric cell as its format displays it, falling back to the stored number."""
    kind = _format_kind(format_id, code)
    if kind:
        return _serial_text(value, kind, date1904)
    try:
        text = _format_number(value, code or _BUILTIN_NUMBER_FORMATS.get(format_id, ""))
    except (ArithmeticError, ValueError, decimal.InvalidOperation):
        text = None
    return _number(value) if text is None else text


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
        workbook_part = zf.related("", "officeDocument", "xl/workbook.xml")
        if workbook_part is None:
            raise ValueError("missing archive member: xl/workbook.xml")
        workbook = zf.xml(workbook_part)
        pr = workbook.find(_q("s", "workbookPr"))
        date1904 = pr is not None and pr.get("date1904") in ("1", "true")
        rels = zf.rels(workbook_part)
        folder = posixpath.dirname(workbook_part)

        strings: list[str] = []
        strings_part = zf.related(workbook_part, "sharedStrings", f"{folder}/sharedStrings.xml")
        if strings_part:
            strings = [
                _xlsx_text(si)
                for event, si in zf.stream(strings_part, frozenset([_q("s", "si")]))
                if event == "unit"
            ]

        style_formats: list[tuple[int, str]] = []
        styles_part = zf.related(workbook_part, "styles", f"{folder}/styles.xml")
        if styles_part:
            styles = zf.xml(styles_part)
            custom = {
                int(f.get("numFmtId", "-1")): f.get("formatCode", "")
                for f in styles.iter(_q("s", "numFmt"))
            }
            xfs = styles.find(_q("s", "cellXfs"))
            for xf in xfs.findall(_q("s", "xf")) if xfs is not None else []:
                fmt = int(xf.get("numFmtId", "0") or 0)
                style_formats.append((fmt, custom.get(fmt, "")))

        sections: list[Section] = []
        for sheet in workbook.iter(_q("s", "sheet")):
            target = rels.get(sheet.get(_q("r", "id"), ""))
            if not target or not zf.has(target):
                continue
            rows = []
            for event, row in zf.stream(target, frozenset([_q("s", "row")])):
                if event != "unit":
                    continue
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
                            fmt, code = (
                                style_formats[style] if style < len(style_formats) else (0, "")
                            )
                            value = _cell_number(number, fmt, code, date1904)
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


def _chart_lines(root: ET.Element) -> list[str]:
    """Chart title and axis titles, then the cached data as rows of category and values."""
    lines = _drawing_lines(root)
    names, categories, values = [], {}, []
    for series in root.iter(_q("c", "ser")):
        name = series.find(f"{_q('c', 'tx')}//{_q('c', 'v')}")
        names.append(name.text or "" if name is not None else "")
        points = {}
        for axis, store in ((_q("c", "cat"), categories), (_q("c", "val"), points)):
            for point in series.iterfind(f"{axis}//{_q('c', 'pt')}"):
                v = point.find(_q("c", "v"))
                store[int(point.get("idx", "0") or 0)] = v.text or "" if v is not None else ""
        values.append(points)
    if any(names) and len(names) > 1:
        lines.append(_row(["", *names]).lstrip(" |"))
    for idx in sorted(set(categories) | {i for points in values for i in points}):
        line = _row([categories.get(idx, ""), *(points.get(idx, "") for points in values)])
        if line:
            lines.append(line)
    if any(names) and len(names) == 1:
        lines.insert(len(lines) - len(categories), names[0]) if categories else lines.append(
            names[0]
        )
    return lines


def pptx(path: str) -> list[Section]:
    with _Archive(path) as zf:
        presentation_part = zf.related("", "officeDocument", "ppt/presentation.xml")
        if presentation_part is None:
            raise ValueError("missing archive member: ppt/presentation.xml")
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
            slide = zf.xml(part)
            slide_rels = zf.rels(part)
            lines = _drawing_lines(slide)
            for chart in slide.iter(_q("c", "chart")):
                target = slide_rels.get(chart.get(_q("r", "id"), ""))
                if target and zf.has(target):
                    lines += _chart_lines(zf.xml(target))
            notes = next(
                (t for t in slide_rels.values() if "notesslide" in t.lower() and zf.has(t)),
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


def _odf_row(row: ET.Element, budget: _Budget) -> list[str]:
    """The row's lines: one, repeated as the row is."""
    cells, gap = [], 0
    for cell in row:
        if cell.tag not in (_q("table", "table-cell"), _q("table", "covered-table-cell")):
            continue
        text = " ".join(_odf_blocks(cell, budget)) or _odf_cell_value(cell)
        if not text:
            # Empty runs before content keep their width; trailing ones are dropped.
            gap += max(int(cell.get(_q("table", "number-columns-repeated"), "1") or 1), 0)
            continue
        cells += [""] * min(gap, _MAX_COLUMNS - len(cells))
        gap = 0
        room = _MAX_COLUMNS - len(cells)
        cells += [text] * _odf_repeat(cell, "number-columns-repeated", text, room, budget)
        if len(cells) >= _MAX_COLUMNS:
            break
    line = _row(cells)
    if not line:
        return []
    return [line] * _odf_repeat(row, "number-rows-repeated", line, _MAX_REPEAT, budget)


def _odf_table(table: ET.Element, budget: _Budget) -> list[str]:
    return [line for row in _odf_rows(table) for line in _odf_row(row, budget)]


def _ods_streamed(zf: "_Archive", budget: _Budget) -> list[Section]:
    """Spreadsheet sheets row by row, without holding content.xml as one tree."""
    sections: list[Section] = []
    name, rows = None, []

    def close_sheet():
        if name is not None and rows:
            sections.append((f"Sheet: {name}\n" + "\n".join(rows), None))

    for event, element in zf.stream("content.xml", frozenset([_q("table", "table-row")])):
        if event == "start" and element.tag == _q("table", "table"):
            close_sheet()
            name, rows = element.get(_q("table", "name"), ""), []
        elif event == "unit" and name is not None:
            rows += _odf_row(element, budget)
    close_sheet()
    return sections


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
        budget = _Budget()
        if zf.has("mimetype") and zf.read("mimetype").strip().endswith(b".spreadsheet"):
            return _ods_streamed(zf, budget)
        body = zf.xml("content.xml").find(_q("office", "body"))
        if body is None:
            return []
        sections: list[Section] = []
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


def _email_parts(part, html_text) -> list[str]:
    """Inline body text in order: every part of a mixed message, one choice per alternative."""
    if part.get_content_disposition() == "attachment":
        return []
    if part.get_content_maintype() == "multipart":
        children = list(part.iter_parts())
        if part.get_content_subtype() != "alternative":
            return [text for child in children for text in _email_parts(child, html_text)]
        # Plain text first; some mailers leave it empty beside a full HTML part.
        for child in sorted(children, key = lambda c: c.get_content_type() != "text/plain"):
            texts = _email_parts(child, html_text)
            if any(t.strip() for t in texts):
                return texts
        return []
    if part.get_content_type() not in ("text/plain", "text/html"):
        return []
    try:
        content = part.get_content()
    except (LookupError, ValueError):
        content = part.get_payload(decode = True) or b""
        content = content.decode("utf-8", "replace") if isinstance(content, bytes) else content
    if part.get_content_subtype() == "html":
        return [html_text(content.encode("utf-8") if isinstance(content, str) else content)]
    return [content]


def _email_text(message, html_text) -> str:
    lines = []
    for header in ("From", "To", "Cc", "Date", "Subject"):
        value = message.get(header)
        if value:
            lines.append(f"{header}: {value}")
    text = "\n\n".join(t.strip() for t in _email_parts(message, html_text) if t.strip())
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


_RTF_MAX_DEPTH = 1000
# Font \fcharset -> code page. 0 (ANSI) and 1 (default) follow the document's \ansicpg.
_RTF_CHARSETS = {
    77: "mac_roman",
    128: "cp932",
    129: "cp949",
    130: "johab",
    134: "cp936",
    136: "cp950",
    161: "cp1253",
    162: "cp1254",
    163: "cp1258",
    177: "cp1255",
    178: "cp1256",
    186: "cp1257",
    204: "cp1251",
    222: "cp874",
    238: "cp1250",
    254: "cp437",
    255: "cp850",
}


def _rtf_codec(code_page: str) -> str | None:
    try:
        b"".decode(code_page)
    except LookupError:
        return None
    return code_page


def _rtf_font_codecs(data: str) -> dict[int, str]:
    """Font number -> codec, from each font table entry's cpg or fcharset."""
    start = data.find("{\\fonttbl")
    if start < 0:
        return {}
    depth, end = 0, start
    for end in range(start, min(len(data), start + 1_000_000)):
        if data[end] in "{}" and data[end - 1] != "\\":
            depth += 1 if data[end] == "{" else -1
            if depth == 0:
                break
    table = data[start:end]
    fonts = {}
    entries = list(re.finditer(r"\\f(\d+)", table))
    for entry, following in zip(entries, entries[1:] + [None]):
        segment = table[entry.end() : following.start() if following else len(table)]
        cpg = re.search(r"\\cpg(\d+)", segment)
        charset = re.search(r"\\fcharset(\d+)", segment)
        codec = _rtf_codec(f"cp{cpg.group(1)}") if cpg else None
        if codec is None and charset:
            codec = _RTF_CHARSETS.get(int(charset.group(1)))
        if codec:
            fonts[int(entry.group(1))] = codec
    return fonts


def rtf(path: str) -> list[Section]:
    with open(path, "rb") as f:
        data = f.read().decode("latin-1")
    if not data.lstrip().startswith("{\\rtf"):
        raise ValueError("file is not RTF")
    return [(_rtf_text(data), None)]


def _rtf_text(data: str) -> str:
    out: list[str] = []
    pending = bytearray()
    # Bytes decode in the active font's code page, else the document's.
    fonts = _rtf_font_codecs(data)
    document_codepage, default_font = "cp1252", None
    codepage = document_codepage
    stack: list[tuple[bool, int, bool, str]] = []
    skip, uc, to_skip = False, 1, 0
    deleted = False  # tracked deletion
    ignorable = False
    upr = False  # the next group is \upr's ANSI branch

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
            if len(stack) >= _RTF_MAX_DEPTH:
                raise ValueError("RTF groups are nested too deeply")
            stack.append((skip, uc, deleted, codepage))
            ignorable = False
            if upr:
                # \upr{ANSI}{\*\ud{Unicode}}: the Unicode branch follows.
                skip, upr = True, False
        elif brace == "}":
            flush()
            skip, uc, deleted, codepage = stack.pop() if stack else (False, 1, False, codepage)
            to_skip, upr = 0, False
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
                document_codepage = _rtf_codec(f"cp{arg}") or "cp1252"
                codepage = fonts.get(default_font, document_codepage)
            elif word == "deff" and arg:
                default_font = int(arg)
                codepage = fonts.get(default_font, document_codepage)
            elif word == "uc" and arg:
                uc = int(arg)
            elif word == "deleted":
                deleted = arg != "0"
            elif word == "upr":
                upr = True
            elif word == "ud":
                ignorable = False
            elif word == "plain":
                flush()
                deleted = False
                codepage = fonts.get(default_font, document_codepage)
            if ignorable or word in _RTF_SKIP:
                skip, ignorable = True, False
                continue
            if skip:
                continue
            if word == "f" and arg:
                flush()
                codepage = fonts.get(int(arg), document_codepage)
            elif word == "u" and arg:
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


_XLS_ERRORS = {
    0x00: "#NULL!",
    0x07: "#DIV/0!",
    0x0F: "#VALUE!",
    0x17: "#REF!",
    0x1D: "#NAME?",
    0x24: "#NUM!",
    0x2A: "#N/A",
    0x2B: "#GETTING_DATA",
}


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

    def number(value: float, xf: int) -> str:
        fmt = xf_formats[xf] if xf < len(xf_formats) else 0
        return _cell_number(value, fmt, formats.get(fmt, ""), date1904)

    sections: list[Section] = []
    for name, offset in sheets:
        cells: dict[int, dict[int, str]] = {}
        last_formula: tuple[int, int] | None = None
        # A long STRING result goes on in CONTINUE records: (cell, segments).
        string: tuple[tuple[int, int], list[bytes]] | None = None

        def put(r, c, value):
            if c < _MAX_COLUMNS:
                cells.setdefault(r, {})[c] = value

        def put_string():
            nonlocal string
            if string is not None:
                try:
                    put(*string[0], _xls_string(string[1], 0, 0)[0])
                except (ValueError, struct.error, IndexError):
                    pass
                string = None

        depth = 0
        for kind, body in _Records(data, offset):
            if string is not None:
                if kind == 0x003C:
                    string[1].append(body)
                    continue
                put_string()
            # Embedded charts nest their own BOF/EOF inside the sheet.
            if kind == 0x0809:
                depth += 1
            elif kind == 0x000A:
                depth -= 1
                if depth <= 0:
                    break
            if len(body) < 6 and kind != 0x0207:
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
                elif v in _XLS_ERRORS:
                    put(r, c, _XLS_ERRORS[v])
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
                elif result[0] == 2 and result[2] in _XLS_ERRORS:
                    put(r, c, _XLS_ERRORS[result[2]])
            elif kind == 0x0207 and last_formula and len(body) >= 3:  # STRING after FORMULA
                string, last_formula = (last_formula, [body]), None
        put_string()
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
_PPT_TEXT_HEADER, _PPT_OUTLINE_REF = 0x0F9F, 0x0F9E
_PPT_DOCUMENT, _PPT_SLIDE, _PPT_NOTES, _PPT_NOTES_ATOM = 0x03E8, 0x03EE, 0x03F0, 0x03F1
_PPT_SLIDE_LIST, _PPT_SLIDE_PERSIST = 0x0FF0, 0x03F3
_PPT_USER_EDIT, _PPT_PERSIST_DIRECTORY = 0x0FF5, 0x1772


def _ppt_records(data: bytes, start: int, end: int):
    """(type, instance, body start, body end, is container) of the records directly in a range."""
    pos = start
    while pos + 8 <= end:
        ver_inst, kind, size = struct.unpack_from("<HHI", data, pos)
        body = pos + 8
        yield kind, ver_inst >> 4, body, min(body + size, end), ver_inst & 0x000F == 0x000F
        pos = body + size


def _ppt_walk(
    data: bytes,
    start: int,
    end: int,
    depth: int = 0,
):
    """Every record in a range, depth first, in stream order."""
    for record in _ppt_records(data, start, end):
        yield record
        if record[4] and depth < 32:
            yield from _ppt_walk(data, record[2], record[3], depth + 1)


def _ppt_text(data: bytes, kind: int, start: int, end: int) -> str:
    text = data[start:end].decode("utf-16-le" if kind == _PPT_TEXT_CHARS else "latin-1", "replace")
    return _clean_control(text.replace("\r", "\n").replace("\x0b", "\n")).strip()


def _ppt_record_at(data: bytes, offset: int | None, kind: int) -> tuple[int, int] | None:
    if offset is None or offset + 8 > len(data):
        return None
    _ver_inst, found, size = struct.unpack_from("<HHI", data, offset)
    return (offset + 8, min(offset + 8 + size, len(data))) if found == kind else None


def _ppt_current(cf: CompoundFile, data: bytes) -> tuple[dict[int, int], int] | None:
    """Persist id -> offset as of the latest save, and the document's persist id.

    Incremental saves append new versions of records, so the stream also holds stale ones;
    the edit chain from "Current User" says which are live.
    """
    if not cf.exists("Current User"):
        return None
    user = cf.open("Current User")
    if len(user) < 20:
        return None
    offset = struct.unpack_from("<I", user, 16)[0]
    persist: dict[int, int] = {}
    document, seen, floor, edits = None, set(), offset, None
    while offset:
        floor = min(floor, offset)
        edit = _ppt_record_at(data, offset, _PPT_USER_EDIT)
        if edit is None or edit[1] - edit[0] < 20 or offset in seen:
            if not seen:
                return None
            # A broken link (some writers point an edit at itself): go on with the next older edit.
            if edits is None:
                edits = [
                    b - 8
                    for k, _i, b, _e, _c in _ppt_records(data, 0, len(data))
                    if k == _PPT_USER_EDIT
                ]
            offset = max((o for o in edits if o < floor), default = 0)
            continue
        seen.add(offset)
        last_edit, directory_offset, document_ref = struct.unpack_from("<III", data, edit[0] + 8)
        document = document_ref if document is None else document
        directory = _ppt_record_at(data, directory_offset, _PPT_PERSIST_DIRECTORY)
        pos, end = directory if directory else (0, 0)
        while pos + 4 <= end:
            entry = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            for i in range(entry >> 20):
                if pos + 4 > end:
                    break
                # Older edits come later in the chain and never replace newer offsets.
                persist.setdefault((entry & 0xFFFFF) + i, struct.unpack_from("<I", data, pos)[0])
                pos += 4
        offset = last_edit
    return (persist, document) if document is not None else None


def _ppt_sheet_lines(data: bytes, start: int, end: int, listed: list[str]) -> list[str]:
    """Text of a slide or notes page in shape order; placeholders point into ``listed``."""
    lines, used = [], set()
    for kind, _inst, body, stop, _container in _ppt_walk(data, start, end):
        if kind == _PPT_OUTLINE_REF and stop - body >= 4:
            index = struct.unpack_from("<I", data, body)[0]
            if index < len(listed) and index not in used:
                used.add(index)
                lines.append(listed[index])
        elif kind in (_PPT_TEXT_CHARS, _PPT_TEXT_BYTES):
            lines.append(_ppt_text(data, kind, body, stop))
    lines += [text for i, text in enumerate(listed) if i not in used]
    # "*" is the slide number field.
    return [line for line in lines if line and line != "*"]


def ppt(path: str) -> list[Section]:
    cf = _compound(path)
    try:
        data = cf.open("PowerPoint Document")
    except CompoundFileError as exc:
        raise ValueError("not a PowerPoint 97-2003 presentation") from exc
    if cf.exists("EncryptedSummary"):
        raise ValueError("file is password protected")
    current = _ppt_current(cf, data)
    document = current and _ppt_record_at(data, current[0].get(current[1]), _PPT_DOCUMENT)
    if not document:
        return _ppt_scan(data)
    persist = current[0]

    # SlideListWithText instance 0 lists slides with their placeholder text; instance 2 lists notes.
    slides: list[tuple[int, int, list[str]]] = []  # (persist id, slide id, placeholder text)
    notes_refs: list[int] = []
    for kind, inst, body, stop, _container in _ppt_records(data, *document):
        if kind != _PPT_SLIDE_LIST or inst not in (0, 2):
            continue
        for child, _i, child_body, child_stop, _c in _ppt_records(data, body, stop):
            if child == _PPT_SLIDE_PERSIST and child_stop - child_body >= 16:
                ref, _flags, _texts, slide_id = struct.unpack_from("<IIII", data, child_body)
                if inst == 0:
                    slides.append((ref, slide_id, []))
                else:
                    notes_refs.append(ref)
            elif inst == 0 and slides and child == _PPT_TEXT_HEADER:
                slides[-1][2].append("")
            elif inst == 0 and slides and child in (_PPT_TEXT_CHARS, _PPT_TEXT_BYTES):
                if not slides[-1][2]:
                    slides[-1][2].append("")
                slides[-1][2][-1] = _ppt_text(data, child, child_body, child_stop)

    notes: dict[int, list[str]] = {}
    for ref in notes_refs:
        page = _ppt_record_at(data, persist.get(ref), _PPT_NOTES)
        if page is None:
            continue
        atom = next(
            (r for r in _ppt_records(data, *page) if r[0] == _PPT_NOTES_ATOM and r[3] - r[2] >= 4),
            None,
        )
        lines = _ppt_sheet_lines(data, *page, [])
        if atom is not None and lines:
            notes[struct.unpack_from("<I", data, atom[2])[0]] = lines

    sections: list[Section] = []
    for number, (ref, slide_id, listed) in enumerate(slides, 1):
        slide = _ppt_record_at(data, persist.get(ref), _PPT_SLIDE)
        lines = _ppt_sheet_lines(data, *slide, listed) if slide else [t for t in listed if t]
        if slide_id in notes:
            lines += ["Notes:"] + notes[slide_id]
        if lines:
            sections.append(("\n".join(lines), number))
    return sections


def _ppt_scan(data: bytes) -> list[Section]:
    """Slide text by a linear scan, for files whose edit history cannot be followed."""
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
    # Exchange senders carry an X.500 path; their SMTP address is stored separately.
    if not address or address.startswith("/"):
        address = (
            _msg_prop(cf, (), "5D01", codec) or _msg_prop(cf, (), "5D02", codec) or ""
        ).strip()
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
