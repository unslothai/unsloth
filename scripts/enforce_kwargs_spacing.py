#!/usr/bin/env python3
"""Ensure keyword arguments use spaces around '=', prune redundant pass statements,
drop the blank line after a short indented import block, merge adjacent same-line
string literals, normalize def-signature magic commas (pre-ruff) so a def with
>= 3 params and a default goes one-per-line while everything else stays
collapsible, and collapse a short multi-line assert onto one line (pre-ruff) by
stripping the magic trailing comma that holds it open.

The fast engine shares one AST per file stage and tokenizes only headers,
short assert spans, and f-strings. A sparse lexer handles the rest. Compatibility
routines preserve the reference's behavior on malformed or unusual source.
There is no dependency on ruff's canonical layout and no persistent cache.

The pre-commit hook formats with ruff-unsloth, which writes the spacing, the
magic-comma rules and the import blank lines itself, and then runs only
``--residual``: string merging and redundant-pass removal, on the files that
could need them. The full passes stay for running this script on its own.
"""

from __future__ import annotations

import ast
import argparse
import bisect
import functools
import gc
import io
import os
import re
import stat
import sys
import tempfile
import textwrap
import tokenize
from collections import defaultdict, deque
from pathlib import Path


def _atomic_write_text(path: Path, data: str, encoding: str) -> None:
    """Write ``data`` to ``path`` atomically via same-dir tmp + fsync + os.replace, so a crash mid-write leaves either the old or the full new content, never a truncation."""
    dirpath = str(path.parent) or "."
    # mkstemp creates 0600 and os.replace keeps it, so carry the original mode across
    # (else run_ruff_format.py loses its executable bit).
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        mode = None
    fd, tmp_path = tempfile.mkstemp(prefix = ".kwargs_fix.", dir = dirpath)
    try:
        # newline = "\n", not the default None. Default translates every "\n" to os.linesep,
        # which on Windows is "\r\n", and the read side (tokenize.open, below) has already
        # normalised CRLF to LF in memory -- so on Windows this rewrote EVERY file it touched to
        # CRLF, against .gitattributes' own `*.py text eol=lf`. A Windows contributor running the
        # pre-commit hook got a whole-file diff on everything they edited. "\n" rather than ""
        # because it states the intended ending rather than "do not translate", and it holds even
        # if the read side stops normalising.
        with os.fdopen(fd, "w", encoding = encoding, newline = "\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# The entry points below use a stage-local analysis, never a persistent file or
# string cache. The _reference_* paths at the end of the file cover malformed
# input, unusual line separators, and syntax the sparse lexer does not know.


def _without_gc(fn):
    """TokenInfo/AST objects have no cycles; do not scan them during this call.

    Restore the caller's GC setting, including on exceptions. No import-time
    global GC change (in particular, no effect on a differential reference).
    """

    @functools.wraps(fn)
    def run(*args, **kwargs):
        enabled = gc.isenabled()
        if enabled:
            gc.disable()
        try:
            return fn(*args, **kwargs)
        finally:
            if enabled:
                gc.enable()

    return run


_ODD_LINES = re.compile(r"[\r\v\f\x1c-\x1e\x85\u2028\u2029]")
# Anything that could open a 3.14 template string (t"", tr"", rt""), anywhere in the
# text, strings and comments included: the sparse lexer below knows no t-strings.
_TSTRING_PREFIX = re.compile(r"(?<![\w])(?:[tT][rR]?|[rR][tT])['\"]")


def _ordinary(text):
    # splitlines() and Python's physical-line numbering disagree on these.
    # Retain even the reference's erroneous indexing for that input.
    if _ODD_LINES.search(text):
        return False
    return sys.version_info < (3, 14) or not _TSTRING_PREFIX.search(text)


# match_case is 3.10+; on 3.9 there is no match statement to reach.
_SUITE_NODES = (ast.stmt, ast.ExceptHandler) + (
    (ast.match_case,) if hasattr(ast, "match_case") else ()
)


def _statements(tree):
    """Breadth-first statement/suite traversal, without visiting expressions.

    No expression can contain a statement. The order of statement ancestors and
    siblings agrees with ast.walk, including except handlers and match cases.
    """
    todo = deque([tree])
    while todo:
        node = todo.popleft()
        yield node
        for field in node._fields:
            value = getattr(node, field, None)
            if isinstance(value, list):
                todo.extend(x for x in value if isinstance(x, _SUITE_NODES))


def _tree(text):
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def _def_edits(lines, nodes):
    edits = []
    for node in nodes:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        count, has_default = _def_spec(node)
        force = count >= _DEF_MIN_PARAMS_FOR_MULTILINE and has_default
        # Read only the header; the generator is stopped at its closing paren.
        # Including the first body line supports a one-line suite.
        lo = node.lineno - 1
        source = "".join(lines[lo : node.body[0].lineno])
        depth = 0
        last = None
        seen_def = False
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if not seen_def:
                seen_def = tok.type == tokenize.NAME and tok.string == "def"
                if seen_def and tok.start[0] != 1:
                    # An explicit continuation between async and def gives
                    # different keyword rows; the reference misses that def.
                    break
                continue
            if not depth:
                if tok.type == tokenize.OP and tok.string == "[":
                    # PEP 695: the reference searches the first '(', even if it
                    # belongs to a type parameter bound. Keep its AST guard.
                    return None
                if tok.type != tokenize.OP or tok.string != "(":
                    continue
                depth = 1
                last = tok
                continue
            if tok.type == tokenize.OP:
                if tok.string == "(":
                    depth += 1
                elif tok.string == ")":
                    depth -= 1
                    if depth == 0:
                        comma = last.type == tokenize.OP and last.string == ","
                        empty = last.type == tokenize.OP and last.string == "("
                        if force and not comma and not empty:
                            row, col = last.end
                            edits.append((row + lo, col, col, ","))
                        elif not force and comma:
                            row, col = last.start
                            edits.append((row + lo, col, col + 1, ""))
                        break
            if tok.type not in _STRING_TRIVIA:
                last = tok
    return edits


def _fits_one_line(segment):
    """The over-estimate the reference uses: lines joined with one space, plus the indent."""
    indent = len(segment[0]) - len(segment[0].lstrip())
    return indent + len(" ".join(s.strip() for s in segment)) <= _LINE_LENGTH


def _apply(lines, edits):
    for row, c0, c1, replacement in sorted(edits, key = lambda e: (e[0], e[1]), reverse = True):
        line = lines[row - 1]
        lines[row - 1] = line[:c0] + replacement + line[c1:]


def _assert_edits(lines, nodes):
    edits = []
    occupied = set()
    for node in nodes:
        if not isinstance(node, ast.Assert) or node.end_lineno == node.lineno:
            continue
        lo, hi = node.lineno, node.end_lineno
        segment = lines[lo - 1 : hi]
        if not _fits_one_line(segment):
            continue
        first = segment[0]
        if node.col_offset != len(first) - len(first.lstrip()) or (
            lo > 1 and lines[lo - 2].rstrip("\r\n").endswith("\\")
        ):
            # Another statement shares the first row, or the row continues the line
            # above: the rows alone would parse as something else (`obj\` + `[1,]`
            # is a subscript in the file but a list here). The whole-file check decides.
            return None
        # Shared physical rows require the reference's sequential whole-file
        # checks. They are uncommon (semicolon suites spanning multiple lines).
        if occupied.intersection(range(lo, hi + 1)):
            return None
        occupied.update(range(lo, hi + 1))
        source = "".join(segment)
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
            if any(t.type == tokenize.COMMENT for t in tokens):
                continue
            if any(
                t.type in (tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", None))
                and t.start[0] != t.end[0]
                for t in tokens
            ):
                # Dedenting an isolated statement can alter a multiline string
                # value. Leave this unusual case to the whole-file guard.
                return None
            significant = [t for t in tokens if t.type not in _STRING_TRIVIA]
            positions = [
                t.start
                for t, nxt in zip(significant, significant[1:])
                if t.type == tokenize.OP
                and t.string == ","
                and nxt.type == tokenize.OP
                and nxt.string in (")", "]", "}")
            ]
            if not positions:
                continue
            trial = segment[:]
            _apply(trial, [(r, c, c + 1, "") for r, c in positions])
            before = ast.parse(textwrap.dedent(source))
            after = ast.parse(textwrap.dedent("".join(trial)))
        except (SyntaxError, tokenize.TokenError):
            # An assert on an elif/except/etc. header cannot be parsed alone.
            return None
        if ast.dump(before) == ast.dump(after):
            edits.extend((r + lo - 1, c, c + 1, "") for r, c in positions)
    return edits


def _pre(
    text,
    definitions = True,
    assertions = True,
):
    if not _ordinary(text):
        return _reference_pre(text, definitions, assertions)
    if not ((definitions and "def" in text) or (assertions and "assert" in text)):
        return text, False
    tree = _tree(text)
    if tree is None:
        return text, False
    nodes = list(_statements(tree))
    lines = text.splitlines(True)
    changed = False
    if definitions:
        edits = _def_edits(lines, nodes)
        if edits is None:
            return _reference_pre(text, definitions, assertions)
        _apply(lines, edits)
        changed = bool(edits)
    if assertions:
        edits = _assert_edits(lines, nodes)
        if edits is None:
            updated, collapsed = _reference_collapse_short_asserts("".join(lines))
            return updated, changed or collapsed
        _apply(lines, edits)
        changed |= bool(edits)
    return ("".join(lines), True) if changed else (text, False)


def _reference_pre(text, definitions, assertions):
    changed = False
    if definitions:
        text, changed = _reference_normalize_def_trailing_comma(text)
    if assertions:
        text, other = _reference_collapse_short_asserts(text)
        changed |= other
    return text, changed


# A sparse lexer for VALID Python. The AST parser has already validated the
# input. We only need comments, string literals, and standalone '=' operators.
# F-strings use the running interpreter's tokenizer, up to their matching END;
# this deliberately preserves the Python 3.10/3.11 versus 3.12+ distinction.
# Unrolled (`a*(?:b a*)*`, not `(?:a|b)*`): the alternation kept one backtracking
# frame per character, about 120 bytes each, so a 10 MB literal took 1.2 GB.
_QUOTE = (
    r'''(?:"""[^"\\]*(?:(?:\\[\s\S]|"(?!""))[^"\\]*)*"""'''
    r"""|\'\'\'[^\'\\]*(?:(?:\\[\s\S]|\'(?!\'\'))[^\'\\]*)*\'\'\'"""
    r'''|"[^"\\\n]*(?:\\[\s\S][^"\\\n]*)*"'''
    r"""|\'[^\'\\\n]*(?:\\[\s\S][^\'\\\n]*)*\')"""
)
_MARKER = re.compile("[#'\"=]")
_LITERAL = re.compile(_QUOTE)


def _literal_start(text, quote):
    def boundary(i):
        return i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")

    if (
        quote >= 2
        and text[quote - 2 : quote].lower() in ("br", "rb", "fr", "rf")
        and boundary(quote - 2)
    ):
        return quote - 2
    if quote and text[quote - 1].lower() in "rubf" and boundary(quote - 1):
        return quote - 1
    return quote


def _lex(text):
    """Return absolute '=' offsets and (kind, start, end) literal pieces."""
    equals, strings = [], []
    position = 0
    while True:
        match = _MARKER.search(text, position)
        if match is None:
            break
        start, end = match.span()
        marker = text[start]
        if marker == "#":
            end = text.find("\n", end)
            if end < 0:
                break
        elif marker == "=":
            if (start == 0 or text[start - 1] not in "!<>=:+-*/%@&|^") and (
                end == len(text) or text[end] != "="
            ):
                equals.append(start)
        else:
            quote = start
            start = _literal_start(text, quote)
            if "f" not in text[start:quote].lower():
                end = _LITERAL.match(text, quote).end()
                strings.append(("str", start, end))
                position = end
                continue
            piece = _fstring(text, start, equals)
            if piece is not None:
                end = piece[2]
                strings.append(piece)
        position = end
    return equals, strings


def _fstring(text, start, equals):
    """Tokenize one f-string from ``start`` with the running interpreter, so its '=' tokens
    and its extent are whatever ``tokenize`` says on this Python (one STRING before 3.12,
    FSTRING_START..END from 3.12). Appends those '=' offsets to ``equals``; returns the piece."""
    cursor = start
    starts = [start]

    def readline():
        nonlocal cursor
        if cursor == len(text):
            return ""
        stop = text.find("\n", cursor)
        stop = len(text) if stop < 0 else stop + 1
        line = text[cursor:stop]
        cursor = stop
        starts.append(stop)
        return line

    depth = 0
    for tok in tokenize.generate_tokens(readline):
        if tok.type == tokenize.OP and tok.string == "=":
            equals.append(starts[tok.start[0] - 1] + tok.start[1])
        if tok.type == _FSTRING_START:
            depth += 1
        elif tok.type == _FSTRING_END:
            depth -= 1
            if depth == 0:
                return ("f", start, starts[tok.end[0] - 1] + tok.end[1])
        elif tok.type == tokenize.STRING and _FSTRING_START is None:
            return ("str", start, starts[tok.end[0] - 1] + tok.end[1])
    return None


def _line_starts(lines):
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    return starts


def _space(lines, starts, equals):
    insertions = defaultdict(list)
    for offset in equals:
        row = bisect.bisect_right(starts, offset) - 1
        col = offset - starts[row]
        line = lines[row]
        if col and line[col - 1] not in " \t":
            insertions[row].append(col)
        if col + 1 < len(line) and line[col + 1] not in " \t\n\r":
            insertions[row].append(col + 1)
    for row, columns in insertions.items():
        line = lines[row]
        pieces = []
        previous = 0
        for col in columns:
            pieces.extend((line[previous:col], " "))
            previous = col
        pieces.append(line[previous:])
        lines[row] = "".join(pieces)
    return insertions


def _blank_rows(lines, nodes):
    drop = set()
    for node in nodes:
        if isinstance(node, ast.Module):
            continue
        for attr in ("body", "orelse", "finalbody"):
            suite = getattr(node, attr, None)
            if not isinstance(suite, list) or not 1 < len(suite) <= 3:
                continue
            for previous, following in zip(suite, suite[1:]):
                if isinstance(previous, (ast.Import, ast.ImportFrom)) and not isinstance(
                    following, (ast.Import, ast.ImportFrom)
                ):
                    rows = range(previous.end_lineno + 1, following.lineno)
                    if rows and all(not lines[r - 1].strip() for r in rows):
                        drop.update(rows)
    return drop


def _pass_nodes(tree):
    redundant = []
    todo = [tree]
    while todo:
        node = todo.pop()
        for attr in ("body", "orelse", "finalbody"):
            suite = getattr(node, attr, None)
            # Deliberately preserve the reference's single-statement-suite
            # recursion bug and its omission of match-case bodies.
            if not isinstance(suite, list) or len(suite) <= 1:
                continue
            redundant.extend(s for s in suite if isinstance(s, ast.Pass))
            todo.extend(child for child in suite if isinstance(child, ast.AST))
        todo.extend(getattr(node, "handlers", ()))
    return redundant


def _merge_edits(text, lines, starts, strings, insertions, nodes):
    runs = []
    current = []
    previous_end = None
    previous_row = None
    for kind, start, end in strings:
        row = bisect.bisect_right(starts, start) - 1
        lastrow = bisect.bisect_right(starts, end - 1) - 1
        adjacent = current and previous_row == row and not text[previous_end:start].strip(" \t")
        if not adjacent or row != lastrow:
            if len(current) > 1:
                runs.append(current)
            current = []
        if row == lastrow:
            c0, c1 = start - starts[row], end - starts[row]
            cols = insertions.get(row, ())
            c0 += bisect.bisect_right(cols, c0)
            c1 += bisect.bisect_left(cols, c1)
            current.append((kind, row + 1, c0, c1, lines[row][c0:c1]))
        previous_end, previous_row = end, lastrow
    if len(current) > 1:
        runs.append(current)
    # A u prefix anywhere in a concatenation decides the `kind` of the constants an
    # f-fold produces, and the rest of that concatenation can sit outside the run
    # (`(f"" u""` + newline + `"a")`), so no check on the run alone can see it.
    # `u` combines with no other prefix, so a piece's first character tells.
    has_u = any(text[start] in "uU" for _kind, start, _end in strings)
    edits = []
    for run in runs:
        merged = _merge_string_run([(p[0], p[4]) for p in run])
        if merged is None:
            continue
        row, c0, c1 = run[0][1], run[0][2], run[-1][3]
        if any(p[0] == "f" for p in run):
            if has_u:
                return None
            enclosing = [
                n for n in nodes if isinstance(n, ast.stmt) and n.lineno <= row <= n.end_lineno
            ]
            stmt = min(enclosing, key = lambda n: n.end_lineno - n.lineno, default = None)
            if isinstance(stmt, ast.Assert) and stmt.lineno != stmt.end_lineno:
                segment = lines[stmt.lineno - 1 : stmt.end_lineno]
                segment[row - stmt.lineno] = lines[row - 1][:c0] + merged + lines[row - 1][c1:]
                if not _fits_one_line(segment):
                    continue
        original = lines[row - 1][c0:c1]
        try:
            # Include parentheses to handle implicit concatenation and use AST
            # dumps only for these small expressions. With u prefixes ruled out
            # above, the pieces outside the run concatenate to the same values
            # either way, so local equality is a sufficient proof; local
            # inequality is NOT proof of file-level inequality (two changes in a
            # larger concatenation may cancel). Defer that case to the
            # reference's all-or-nothing global guard.
            before = ast.parse("(" + original + ")", mode = "eval")
            after = ast.parse("(" + merged + ")", mode = "eval")
            if ast.dump(before) != ast.dump(after):
                return None
        except SyntaxError:
            return None
        if merged[:1] != original[:1] and c0 and ("_" + lines[row - 1][c0 - 1]).isidentifier():
            # The fold adds a prefix, and the run is glued to a name or keyword
            # (`return"a" f"{x}"`): the result would read `returnf"..."`.
            return None
        edits.append((row, c0, c1, merged))
    return edits


def _post(
    text,
    spacing = True,
    blanks = True,
    merging = True,
    passes = True,
):
    if not _ordinary(text):
        return _reference_post(text, spacing, blanks, merging, passes)
    tree = _tree(text)
    if tree is None:
        # enforce_spacing historically tokenizes even invalid syntax and can
        # raise TokenError/IndentationError; the other passes return unchanged.
        return _reference_post(text, spacing, blanks, merging, passes)
    lines = text.splitlines(True)
    if not lines:
        return text, False
    starts = _line_starts(lines)
    nodes = list(_statements(tree)) if blanks or merging else ()
    try:
        equals, strings = _lex(text) if spacing or merging else ((), ())
    except (tokenize.TokenError, SyntaxError):
        # The f-string scan runs the real tokenizer from mid-file; should it ever
        # disagree with the parser, the reference path decides, as for bad syntax.
        return _reference_post(text, spacing, blanks, merging, passes)
    original_lines = lines[:]
    insertions = _space(lines, starts, equals) if spacing else {}
    drop = _blank_rows(lines, nodes) if blanks and "import" in text else set()
    edits = _merge_edits(text, lines, starts, strings, insertions, nodes) if merging else []
    if edits is None:
        return _reference_post(text, spacing, blanks, merging, passes)
    redundant = _pass_nodes(tree) if passes and "pass" in text else []
    # Convert insertions/replacements to UTF-8 byte coordinates ONLY for pass
    # nodes. Preserve the reference's byte-column-as-character-column behavior.
    shifts = defaultdict(list)
    for node in redundant:
        row = node.lineno - 1
        if row not in shifts:
            shifts[row] = [
                (len(original_lines[row][:c].encode("utf-8")), 1) for c in insertions.get(row, ())
            ]
    pass_positions = []
    for node in redundant:
        row = node.lineno - 1
        start, end = node.col_offset, node.end_col_offset
        start += sum(delta for pos, delta in shifts[row] if pos <= node.col_offset)
        end += sum(delta for pos, delta in shifts[row] if pos < node.end_col_offset)
        merge_shift = 0
        for erow, c0, c1, replacement in edits:
            if erow == row + 1:
                end_byte = len(lines[row][:c1].encode("utf-8"))
                delta = len(replacement.encode("utf-8")) - len(lines[row][c0:c1].encode("utf-8"))
                if end_byte <= start:
                    merge_shift += delta
        start += merge_shift
        end += merge_shift
        pass_positions.append((row, start, end))
    _apply(lines, edits)
    for row, start, end in sorted(pass_positions, reverse = True):
        line = lines[row]
        segment = line[:start] + line[end:]
        lines[row] = segment if segment.strip() else ""
    changed = bool(insertions or drop or edits or redundant)
    if not changed:
        return text, False
    kept = [line for row, line in enumerate(lines, 1) if row not in drop]
    if redundant:
        kept = [
            line if i == len(kept) - 1 or line.endswith("\n") else line + "\n"
            for i, line in enumerate(kept)
            if line
        ]
    return "".join(kept), True


def _reference_post(text, spacing, blanks, merging, passes):
    changed = False
    for enabled, fn in (
        (spacing, _reference_enforce_spacing),
        (blanks, _reference_remove_blank_after_short_import),
        (merging, _reference_merge_adjacent_string_literals),
        (passes, _reference_remove_redundant_passes),
    ):
        if enabled:
            text, other = fn(text)
            changed |= other
    return text, changed


@_without_gc
def normalize_def_trailing_comma(text: str) -> tuple[str, bool]:
    return _pre(text, assertions = False)


@_without_gc
def collapse_short_asserts(text: str) -> tuple[str, bool]:
    return _pre(text, definitions = False)


@_without_gc
def enforce_spacing(text: str) -> tuple[str, bool]:
    return _post(text, blanks = False, merging = False, passes = False)


@_without_gc
def remove_blank_after_short_import(text: str) -> tuple[str, bool]:
    if "import" not in text:
        return text, False
    return _post(text, spacing = False, merging = False, passes = False)


@_without_gc
def merge_adjacent_string_literals(text: str) -> tuple[str, bool]:
    return _post(text, spacing = False, blanks = False, passes = False)


@_without_gc
def remove_redundant_passes(text: str) -> tuple[str, bool]:
    if "pass" not in text:
        return text, False
    return _post(text, spacing = False, blanks = False, merging = False)


# A closing quote, blanks, then an opening one: every same-line implicit concatenation
# matches, and so does `" ".join`. The lexer below settles which it is.
_ADJACENT_QUOTES = re.compile(r"""['"][ \t]+(?:[rRuUbBfF]{1,2})?['"]""")
_PASS_ROW = re.compile(r"^([ \t]*)pass[ \t]*(?:#.*)?$", re.MULTILINE)


def _may_merge(text):
    """Whether a same-line run of adjacent literals is one _merge_string_run would fold."""
    if not _ADJACENT_QUOTES.search(text):
        return False
    try:
        _, strings = _lex(text)
    except Exception:
        # The lexer assumes valid source; anything it cannot read gets the full pass.
        return True
    # Runs built as _merge_edits builds them: single-row pieces, blanks between.
    run = []
    for piece in strings + [None]:
        if (
            piece is not None
            and run
            and "\n" not in text[run[-1][1] : piece[2]]
            and not text[run[-1][2] : piece[1]].strip(" \t")
        ):
            run.append(piece)
            continue
        if len(run) > 1 and _merge_string_run([(k, text[a:b]) for k, a, b in run]) is not None:
            return True
        run = [] if piece is None else [piece]
    return False


def _may_have_redundant_pass(text):
    """Whether some `pass` can share its suite with another statement.

    Over-approximates on formatted source. A sibling before the `pass` ends on a row
    indented at least as deep (its last line, a closing bracket, a compound body) or on
    the row that closes a triple-quoted string; a sibling after it starts at the same
    indent. Comment and blank rows are skipped, and a backslash continuation anywhere
    sends the file to the full pass.
    """
    if "pass" not in text:
        return False
    if "\\\n" in text:
        return True
    lines = text.split("\n")
    rows = [m for m in _PASS_ROW.finditer(text)]
    if not rows:
        return False
    starts = _line_starts([line + "\n" for line in lines])

    def significant(row):
        stripped = lines[row].strip()
        return stripped and not stripped.startswith("#")

    for match in rows:
        row = bisect.bisect_right(starts, match.start()) - 1
        own = len(match.group(1))
        before = next((r for r in range(row - 1, -1, -1) if significant(r)), None)
        if before is not None:
            line = lines[before]
            if len(line) - len(line.lstrip()) >= own or '"""' in line or "'''" in line:
                return True
        after = next((r for r in range(row + 1, len(lines)) if significant(r)), None)
        if after is not None and len(lines[after]) - len(lines[after].lstrip()) == own:
            return True
    return False


@_without_gc
def residual(text: str) -> tuple[str, bool]:
    """The two passes ruff-unsloth leaves to Python, run only where they could apply."""
    if not _ordinary(text):
        return _post(text, spacing = False, blanks = False)
    merging = _may_merge(text)
    passes = _may_have_redundant_pass(text)
    if not (merging or passes):
        return text, False
    return _post(text, spacing = False, blanks = False, merging = merging, passes = passes)


@_without_gc
def process_file(
    path: Path,
    pre: bool = False,
    only_residual: bool = False,
) -> bool:
    try:
        with tokenize.open(path) as handle:
            original = handle.read()
            encoding = handle.encoding
    except (OSError, SyntaxError) as exc:
        print(f"Failed to read {path}: {exc}", file = sys.stderr)
        return False
    if only_residual:
        updated, changed = residual(original)
    else:
        updated, changed = _pre(original) if pre else _post(original)
    if changed:
        _atomic_write_text(path, updated, encoding)
    return changed


# Shared literal policy and exact compatibility paths. These are deliberately
# retained so exceptional source keeps the reference behavior, including bugs.


def _reference_enforce_spacing(text: str) -> tuple[str, bool]:
    """Return updated text with keyword '=' padded by spaces, plus change flag."""
    lines = text.splitlines(keepends = True)
    if not lines:
        return text, False

    offsets: dict[int, int] = defaultdict(int)
    changed = False

    reader = io.StringIO(text).readline
    for token in tokenize.generate_tokens(reader):
        if token.type != tokenize.OP or token.string != "=":
            continue

        line_index = token.start[0] - 1
        col = token.start[1] + offsets[line_index]

        if line_index < 0 or line_index >= len(lines):
            continue

        line = lines[line_index]
        if col >= len(line) or line[col] != "=":
            continue

        line_changed = False

        if col > 0 and line[col - 1] not in {" ", "\t"}:
            line = f"{line[:col]} {line[col:]}"
            offsets[line_index] += 1
            col += 1
            line_changed = True
            changed = True

        next_index = col + 1
        if next_index < len(line) and line[next_index] not in {" ", "\t", "\n", "\r"}:
            line = f"{line[:next_index]} {line[next_index:]}"
            offsets[line_index] += 1
            line_changed = True
            changed = True

        if line_changed:
            lines[line_index] = line

    if not changed:
        return text, False

    return "".join(lines), True


def _reference_remove_redundant_passes(text: str) -> tuple[str, bool]:
    """Drop pass statements that share a block with other executable code."""

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, False

    redundant: list[ast.Pass] = []

    def visit(node: ast.AST) -> None:
        for attr in ("body", "orelse", "finalbody"):
            value = getattr(node, attr, None)
            if not isinstance(value, list) or len(value) <= 1:
                continue
            for stmt in value:
                if isinstance(stmt, ast.Pass):
                    redundant.append(stmt)
            for stmt in value:
                if isinstance(stmt, ast.AST):
                    visit(stmt)
        handlers = getattr(node, "handlers", None)
        if handlers:
            for handler in handlers:
                visit(handler)

    visit(tree)

    if not redundant:
        return text, False

    lines = text.splitlines(keepends = True)
    changed = False

    for node in sorted(redundant, key = lambda item: (item.lineno, item.col_offset), reverse = True):
        start = node.lineno - 1
        end = (node.end_lineno or node.lineno) - 1
        if start >= len(lines):
            continue
        changed = True
        if start == end:
            line = lines[start]
            col_start = node.col_offset
            col_end = node.end_col_offset or (col_start + 4)
            segment = line[:col_start] + line[col_end:]
            lines[start] = segment if segment.strip() else ""
            continue

        prefix = lines[start][: node.col_offset]
        lines[start] = prefix if prefix.strip() else ""
        for idx in range(start + 1, end):
            lines[idx] = ""
        suffix = lines[end][(node.end_col_offset or 0) :]
        lines[end] = suffix

    result_lines: list[str] = []
    for index, line in enumerate(lines):
        if not line:
            continue
        if index < len(lines) - 1 and not line.endswith("\n"):
            result_lines.append(f"{line}\n")
        else:
            result_lines.append(line)

    return "".join(result_lines), changed


def _reference_remove_blank_after_short_import(text: str) -> tuple[str, bool]:
    """Drop blank line(s) after an import block in a small nested suite. In an indented suite of <= 3 statements (never module level), when consecutive imports are followed across blank lines (nothing else) by another statement, remove those blanks. A comment in the gap blocks the rule. Removing blank lines never changes the AST."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, False

    lines = text.splitlines(keepends = True)
    import_types = (ast.Import, ast.ImportFrom)
    drop: set[int] = set()

    def suites_of(node: ast.AST) -> list[list[ast.stmt]]:
        if isinstance(node, ast.Module):
            return []  # module-level import spacing is left alone
        out: list[list[ast.stmt]] = []
        for attr in ("body", "orelse", "finalbody"):
            val = getattr(node, attr, None)
            if isinstance(val, list) and val and all(isinstance(s, ast.stmt) for s in val):
                out.append(val)
        return out

    for node in ast.walk(tree):
        for suite in suites_of(node):
            if len(suite) > 3:
                continue
            i = 0
            while i < len(suite):
                if not isinstance(suite[i], import_types):
                    i += 1
                    continue
                j = i
                while j + 1 < len(suite) and isinstance(suite[j + 1], import_types):
                    j += 1
                if j + 1 < len(suite):
                    last_imp, nxt = suite[j], suite[j + 1]
                    gap = range((last_imp.end_lineno or last_imp.lineno) + 1, nxt.lineno)
                    nums = [n for n in gap if 1 <= n <= len(lines)]
                    if nums and all(lines[n - 1].strip() == "" for n in nums):
                        drop.update(nums)
                i = j + 1

    if not drop:
        return text, False
    kept = [ln for idx, ln in enumerate(lines, start = 1) if idx not in drop]
    return "".join(kept), True


_STRING_TRIVIA = (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT)


_DEF_MIN_PARAMS_FOR_MULTILINE = 3


def _def_spec(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, bool]:
    """(param count, has-any-default). ``*`` and ``/`` markers are not counted. A default exists if any positional default is present or any keyword-only default is not ``None`` (``None`` in ``kw_defaults`` means a required keyword-only arg)."""
    a = node.args
    count = (
        len(a.posonlyargs)
        + len(a.args)
        + len(a.kwonlyargs)
        + (1 if a.vararg else 0)
        + (1 if a.kwarg else 0)
    )
    return count, bool(a.defaults) or any(d is not None for d in a.kw_defaults)


def _def_specs_by_line(tree: ast.AST) -> dict[int, tuple[int, bool]]:
    """Map each def keyword line to ``_def_spec`` of that def."""
    out: dict[int, tuple[int, bool]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.lineno] = _def_spec(node)
    return out


def _reference_normalize_def_trailing_comma(text: str) -> tuple[str, bool]:
    """Force a def signature one-per-line iff >= 3 params AND a default, else collapsible. A qualifying signature gets a magic trailing comma added (ruff wraps it one-per-line); every other signature has its trailing comma stripped so ruff collapses it when it fits. Def parameter lists only, never call sites or collection literals. Run BEFORE ruff format. Never changes the AST (re-checked)."""
    try:
        tree = ast.parse(text)
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text, False

    specs = _def_specs_by_line(tree)
    n = len(toks)
    edits: list[tuple[int, int, str]] = []  # (row, col, "del" | "ins")
    i = 0
    while i < n:
        t = toks[i]
        if t.type == tokenize.NAME and t.string == "def" and t.start[0] in specs:
            cnt, has_default = specs[t.start[0]]
            force_multiline = cnt >= _DEF_MIN_PARAMS_FOR_MULTILINE and has_default
            j = i + 1
            while j < n and not (toks[j].type == tokenize.OP and toks[j].string == "("):
                if toks[j].type == tokenize.NEWLINE:
                    break
                j += 1
            if j < n and toks[j].type == tokenize.OP and toks[j].string == "(":
                depth = 0
                k = j
                while k < n:
                    tk = toks[k]
                    if tk.type == tokenize.OP and tk.string == "(":
                        depth += 1
                    elif tk.type == tokenize.OP and tk.string == ")":
                        depth -= 1
                        if depth == 0:
                            m = k - 1
                            while m > j and toks[m].type in _STRING_TRIVIA:
                                m -= 1
                            last = toks[m]
                            has_comma = last.type == tokenize.OP and last.string == ","
                            empty = m == j
                            if force_multiline and not has_comma and not empty:
                                edits.append((last.end[0], last.end[1], "ins"))
                            elif not force_multiline and has_comma:
                                edits.append((last.start[0], last.start[1], "del"))
                            break
                    k += 1
                i = k + 1
                continue
        i += 1

    if not edits:
        return text, False

    lines = text.splitlines(keepends = True)
    for row, col, kind in sorted(edits, reverse = True):
        ln = lines[row - 1]
        if kind == "del":
            if col < len(ln) and ln[col] == ",":
                lines[row - 1] = ln[:col] + ln[col + 1 :]
        else:
            lines[row - 1] = ln[:col] + "," + ln[col:]
    out = "".join(lines)
    try:
        if ast.dump(ast.parse(out)) != ast.dump(ast.parse(text)):
            return text, False
    except SyntaxError:
        return text, False
    return out, True


def _split_string_token(s: str) -> tuple[str, str, str] | None:
    """Split a string literal source into (prefix, quote, body): the letters before the opening quote, the delimiter, and everything between. ``None`` if not a recognizable string literal."""
    i = 0
    while i < len(s) and s[i] not in ("'", '"'):
        i += 1
    if i >= len(s):
        return None
    prefix, rest = s[:i], s[i:]
    for q in ('"""', "'''", '"', "'"):
        if rest.startswith(q) and rest.endswith(q) and len(rest) >= 2 * len(q):
            return prefix, q, rest[len(q) : len(rest) - len(q)]
    return None


# FSTRING_* tokens exist only on 3.12+ (PEP 701); older versions emit one STRING token.
_FSTRING_START = getattr(tokenize, "FSTRING_START", None)
_FSTRING_END = getattr(tokenize, "FSTRING_END", None)


# A piece is a STRING token or a whole FSTRING_START..FSTRING_END run.
def _string_pieces(
    toks: list[tokenize.TokenInfo], lines: list[str]
) -> list[tuple[str, tuple[int, int], tuple[int, int], str | None]]:
    pieces: list[tuple[str, tuple[int, int], tuple[int, int], str | None]] = []
    n = len(toks)

    def raw_of(start: tuple[int, int], end: tuple[int, int]) -> str | None:
        if start[0] != end[0]:
            return None
        return lines[start[0] - 1][start[1] : end[1]]

    i = 0
    while i < n:
        t = toks[i]
        if t.type == tokenize.STRING:
            pieces.append(("str", t.start, t.end, raw_of(t.start, t.end)))
            i += 1
        elif t.type == _FSTRING_START:
            depth = 0
            j = i
            while j < n:
                if toks[j].type == _FSTRING_START:
                    depth += 1
                elif toks[j].type == _FSTRING_END:
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            end = toks[j].end
            pieces.append(("f", t.start, end, raw_of(t.start, end)))
            i = j + 1
        else:
            pieces.append(("other", t.start, t.end, None))
            i += 1
    return pieces


def _merge_string_run(pieces: list[tuple[str, str]]) -> str | None:
    """Merge a run of adjacent string pieces into one literal's source text. ``pieces`` is ``(kind, raw_source)`` with kind ``"str"`` or ``"f"``. Bytes are left side-by-side (``None``); a run with no f-string merges plain/raw/unicode sharing one prefix+quote by body concatenation; a run mixing an f-string with a plain string (no bytes, no raw) folds into one f-string with plain braces escaped. Runs of only f-strings are left alone. The caller re-checks the AST and drops a differing change."""
    parsed = []
    for kind, raw in pieces:
        pqb = _split_string_token(raw)
        if pqb is None:
            return None
        prefix, quote, body = pqb
        if "b" in prefix.lower():
            return None
        parsed.append((kind, prefix, quote, body))
    if len({p[2] for p in parsed}) != 1:
        return None
    quote = parsed[0][2]
    if not any(p[0] == "f" for p in parsed):
        if len({p[1].lower() for p in parsed}) != 1:
            return None
        return f"{parsed[0][1]}{quote}{''.join(p[3] for p in parsed)}{quote}"
    # Fold only a plain string glued onto an f-string; long f-string runs would force ruff to re-wrap.
    if all(p[0] == "f" for p in parsed):
        return None
    if any("r" in p[1].lower() for p in parsed):
        return None
    body = "".join(
        b if kind == "f" else b.replace("{", "{{").replace("}", "}}")
        for kind, _pfx, _q, b in parsed
    )
    return f"f{quote}{body}{quote}"


_LINE_LENGTH = 100  # ruff line-length; an f-fold must not push a statement past it


def _enclosing_stmt(tree: ast.AST, row: int) -> ast.stmt | None:
    """The innermost statement whose physical-line span contains ``row``."""
    best: tuple[ast.stmt, int] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            lo = node.lineno
            hi = node.end_lineno or lo
            if lo <= row <= hi and (best is None or hi - lo < best[1]):
                best = (node, hi - lo)
    return best[0] if best else None


def _fold_collapses(
    tree: ast.AST, lines: list[str], row: int, c0: int, c1: int, merged: str
) -> bool:
    """Whether an f-string fold at ``row[c0:c1]`` -> ``merged`` is safe to apply. Only ``assert`` wraps awkwardly when a message folds (ruff parenthesizes the condition once it no longer fits one line), so an ``assert`` fold is allowed only if already one line or its estimated folded length fits; every other construct wraps acceptably."""
    stmt = _enclosing_stmt(tree, row)
    if not isinstance(stmt, ast.Assert):
        return True
    lo, hi = stmt.lineno, stmt.end_lineno or stmt.lineno
    if lo == hi:
        return True
    seg = []
    for k in range(lo, hi + 1):
        ln = lines[k - 1].rstrip("\n")
        if k == row:
            ln = ln[:c0] + merged + ln[c1:]
        seg.append(ln)
    indent = len(seg[0]) - len(seg[0].lstrip())
    # Over-estimate by joining with a space so borderline cases skip the fold.
    joined = " ".join(s.strip() for s in seg)
    return indent + len(joined) <= _LINE_LENGTH


def _reference_merge_adjacent_string_literals(text: str) -> tuple[str, bool]:
    """Merge adjacent string literals on ONE physical line into a single literal. Plain/raw/unicode runs merge by concatenation; an f-string + plain string folds into one f-string (plain braces escaped) only while the statement still fits one line. Runs of only f-strings, and bytes, are left side-by-side. The file AST is re-checked and a differing change dropped."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
        tree = ast.parse(text)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text, False

    lines = text.splitlines(keepends = True)
    pieces = _string_pieces(toks, lines)

    runs: list[list[tuple[str, tuple[int, int], tuple[int, int], str]]] = []
    cur: list[tuple[str, tuple[int, int], tuple[int, int], str]] = []
    for kind, start, end, raw in pieces:
        if kind in ("str", "f") and raw is not None:
            if cur and cur[-1][2][0] != start[0]:
                if len(cur) >= 2:
                    runs.append(cur)
                cur = []
            cur.append((kind, start, end, raw))
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
    if len(cur) >= 2:
        runs.append(cur)
    if not runs:
        return text, False

    edits = []
    for run in runs:
        merged = _merge_string_run([(kind, raw) for kind, _s, _e, raw in run])
        if merged is None:
            continue
        row, c0, c1 = run[0][1][0], run[0][1][1], run[-1][2][1]
        if any(kind == "f" for kind, _s, _e, _r in run) and not _fold_collapses(
            tree, lines, row, c0, c1, merged
        ):
            continue
        edits.append((row, c0, c1, merged))
    if not edits:
        return text, False

    for row, c0, c1, repl in sorted(edits, key = lambda e: (e[0], e[1]), reverse = True):
        ln = lines[row - 1]
        lines[row - 1] = ln[:c0] + repl + ln[c1:]
    out = "".join(lines)
    try:
        if ast.dump(ast.parse(text)) != ast.dump(ast.parse(out)):
            return text, False
    except SyntaxError:
        return text, False
    return out, True


def _reference_collapse_short_asserts(text: str) -> tuple[str, bool]:
    """Collapse a multi-line ``assert`` onto one line when the estimated one-line length fits, by stripping the magic trailing commas (a comma before a closer) holding it open so ruff rejoins it. Run BEFORE ruff format. Skips asserts with a comment (would oscillate). Stripping is non-semantic except for a one-element tuple; the AST is re-checked and changing asserts left alone."""
    try:
        tree = ast.parse(text)
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text, False

    lines = text.splitlines(keepends = True)
    multiline = [
        (n.lineno, n.end_lineno)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assert) and (n.end_lineno or n.lineno) > n.lineno
    ]
    if not multiline:
        return text, False

    comment_rows = {t.start[0] for t in toks if t.type == tokenize.COMMENT}

    targets = []
    for lo, hi in multiline:
        if any(lo <= r <= hi for r in comment_rows):
            continue
        seg = [lines[k].rstrip("\n") for k in range(lo - 1, hi)]
        indent = len(seg[0]) - len(seg[0].lstrip())
        # Over-estimate so a fits verdict never fights ruff's real one-line output.
        if indent + len(" ".join(s.strip() for s in seg)) <= _LINE_LENGTH:
            targets.append((lo, hi))
    if not targets:
        return text, False

    sig = [t for t in toks if t.type not in _STRING_TRIVIA]
    by_target: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for i, t in enumerate(sig):
        if t.type == tokenize.OP and t.string == ",":
            nxt = sig[i + 1] if i + 1 < len(sig) else None
            if nxt and nxt.type == tokenize.OP and nxt.string in (")", "]", "}"):
                for lo, hi in targets:
                    if lo <= t.start[0] <= hi:
                        by_target[(lo, hi)].append(t.start)
                        break
    if not by_target:
        return text, False

    base_dump = ast.dump(tree)
    working = lines[:]
    changed = False
    for positions in by_target.values():
        trial = working[:]
        for row, col in sorted(positions, reverse = True):
            ln = trial[row - 1]
            if col < len(ln) and ln[col] == ",":
                trial[row - 1] = ln[:col] + ln[col + 1 :]
        try:
            if ast.dump(ast.parse("".join(trial))) == base_dump:
                working, changed = trial, True
        except SyntaxError:
            pass
    return ("".join(working), True) if changed else (text, False)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("files", nargs = "+", help = "Python files to fix")
    parser.add_argument(
        "--pre",
        action = "store_true",
        help = "pre-ruff pass: normalize def-signature commas + collapse short multi-line asserts",
    )
    parser.add_argument(
        "--residual",
        action = "store_true",
        help = "after ruff-unsloth: only string merging and redundant-pass removal",
    )
    args = parser.parse_args(argv)

    touched: list[Path] = []
    self_path = Path(__file__).resolve()

    for entry in args.files:
        path = Path(entry)
        if path.resolve() == self_path:
            continue
        if not path.exists() or path.is_dir():
            continue
        if process_file(path, pre = args.pre, only_residual = args.residual):
            touched.append(path)

    if touched:
        for path in touched:
            print(f"Adjusted kwarg spacing in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
