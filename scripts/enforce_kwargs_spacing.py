#!/usr/bin/env python3
"""Ensure keyword arguments use spaces around '=', prune redundant pass statements,
drop the blank line after a short indented import block, merge adjacent same-line
string literals, normalize def-signature magic commas (pre-ruff) so a def with
>= 3 params and a default goes one-per-line while everything else stays
collapsible, and collapse a short multi-line assert onto one line (pre-ruff) by
stripping the magic trailing comma that holds it open."""

from __future__ import annotations

import ast
import argparse
import io
import os
import sys
import tempfile
import tokenize
from collections import defaultdict
from pathlib import Path


def _atomic_write_text(path: Path, data: str, encoding: str) -> None:
    """Write ``data`` to ``path`` atomically via same-dir tmp + fsync + os.replace,
    so a crash mid-write leaves either the old or full new content, never a truncation."""
    dirpath = str(path.parent) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".kwargs_fix.", dir=dirpath)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def enforce_spacing(text: str) -> tuple[str, bool]:
    """Return updated text with keyword '=' padded by spaces, plus change flag."""
    lines = text.splitlines(keepends=True)
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

        # Insert a space before '=' when missing and not preceded by whitespace.
        if col > 0 and line[col - 1] not in {" ", "\t"}:
            line = f"{line[:col]} {line[col:]}"
            offsets[line_index] += 1
            col += 1
            line_changed = True
            changed = True

        # Insert a space after '=' when missing and not followed by whitespace or newline.
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


def remove_redundant_passes(text: str) -> tuple[str, bool]:
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

    lines = text.splitlines(keepends=True)
    changed = False

    for node in sorted(redundant, key=lambda item: (item.lineno, item.col_offset), reverse=True):
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

        # Fall-back for unexpected multi-line 'pass'.
        prefix = lines[start][: node.col_offset]
        lines[start] = prefix if prefix.strip() else ""
        for idx in range(start + 1, end):
            lines[idx] = ""
        suffix = lines[end][(node.end_col_offset or 0) :]
        lines[end] = suffix

    # Normalise to ensure lines end with newlines except at EOF.
    result_lines: list[str] = []
    for index, line in enumerate(lines):
        if not line:
            continue
        if index < len(lines) - 1 and not line.endswith("\n"):
            result_lines.append(f"{line}\n")
        else:
            result_lines.append(line)

    return "".join(result_lines), changed


def remove_blank_after_short_import(text: str) -> tuple[str, bool]:
    """Drop blank line(s) after an import block in a small nested suite.

    In an indented suite of <= 3 statements (never module level), when consecutive
    imports are followed across blank lines (nothing else) by another statement,
    remove those blanks. A comment in the gap blocks the rule. Removing blank lines
    never changes the AST.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, False

    lines = text.splitlines(keepends=True)
    import_types = (ast.Import, ast.ImportFrom)
    drop: set[int] = set()  # 1-based physical line numbers to delete

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
            if len(suite) > 3:  # only small blocks
                continue
            i = 0
            while i < len(suite):
                if not isinstance(suite[i], import_types):
                    i += 1
                    continue
                j = i
                while j + 1 < len(suite) and isinstance(suite[j + 1], import_types):
                    j += 1
                if j + 1 < len(suite):  # an import block followed by another statement
                    last_imp, nxt = suite[j], suite[j + 1]
                    gap = range((last_imp.end_lineno or last_imp.lineno) + 1, nxt.lineno)
                    nums = [n for n in gap if 1 <= n <= len(lines)]
                    if nums and all(lines[n - 1].strip() == "" for n in nums):
                        drop.update(nums)
                i = j + 1

    if not drop:
        return text, False
    kept = [ln for idx, ln in enumerate(lines, start=1) if idx not in drop]
    return "".join(kept), True


_STRING_TRIVIA = (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT)


_DEF_MIN_PARAMS_FOR_MULTILINE = 3  # signatures with < this many params stay one line


def _def_specs_by_line(tree: ast.AST) -> dict[int, tuple[int, bool]]:
    """Map each def keyword line to (param count, has-any-default).

    ``*`` / ``/`` markers aren't counted. A default exists if any positional default
    is present or any keyword-only default is not ``None`` (``None`` in ``kw_defaults``
    means a required keyword-only arg).
    """
    out: dict[int, tuple[int, bool]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            count = (
                len(a.posonlyargs)
                + len(a.args)
                + len(a.kwonlyargs)
                + (1 if a.vararg else 0)
                + (1 if a.kwarg else 0)
            )
            has_default = bool(a.defaults) or any(d is not None for d in a.kw_defaults)
            out[node.lineno] = (count, has_default)
    return out


def normalize_def_trailing_comma(text: str) -> tuple[str, bool]:
    """Force a def signature one-per-line iff >= 3 params AND a default; else collapsible.

    A qualifying signature gets a magic trailing comma added (ruff wraps it
    one-per-line); every other signature has its trailing comma stripped so ruff
    collapses it when it fits. Def parameter lists only, never call sites or
    collection literals. Run BEFORE ruff format. Never changes the AST (re-checked).
    """
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
                            empty = m == j  # nothing between ( and )
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

    lines = text.splitlines(keepends=True)
    for row, col, kind in sorted(edits, reverse=True):
        ln = lines[row - 1]
        if kind == "del":
            if col < len(ln) and ln[col] == ",":
                lines[row - 1] = ln[:col] + ln[col + 1 :]
        else:  # ins
            lines[row - 1] = ln[:col] + "," + ln[col:]
    out = "".join(lines)
    try:
        if ast.dump(ast.parse(out)) != ast.dump(ast.parse(text)):
            return text, False
    except SyntaxError:
        return text, False
    return out, True


def _split_string_token(s: str) -> tuple[str, str, str] | None:
    """Split a string literal source into (prefix, quote, body).

    ``prefix`` is the letters before the opening quote, ``quote`` the delimiter,
    ``body`` everything between. ``None`` if not a recognizable string literal.
    """
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


# A "piece" is one string literal in source: a plain STRING token, or a whole
# f-string spanning FSTRING_START..FSTRING_END. (kind, (row, col0), (row, col1), raw)
def _string_pieces(
    toks: list[tokenize.TokenInfo], lines: list[str]
) -> list[tuple[str, tuple[int, int], tuple[int, int], str | None]]:
    pieces: list[tuple[str, tuple[int, int], tuple[int, int], str | None]] = []
    n = len(toks)

    def raw_of(start: tuple[int, int], end: tuple[int, int]) -> str | None:
        if start[0] != end[0]:  # only single-physical-line pieces are mergeable
            return None
        return lines[start[0] - 1][start[1] : end[1]]

    i = 0
    while i < n:
        t = toks[i]
        if t.type == tokenize.STRING:
            pieces.append(("str", t.start, t.end, raw_of(t.start, t.end)))
            i += 1
        elif t.type == tokenize.FSTRING_START:
            depth = 0
            j = i
            while j < n:  # walk to the matching FSTRING_END (f-strings can nest)
                if toks[j].type == tokenize.FSTRING_START:
                    depth += 1
                elif toks[j].type == tokenize.FSTRING_END:
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
    """Merge a run of adjacent string pieces into one literal's source text.

    ``pieces`` is ``(kind, raw_source)`` with kind ``"str"`` or ``"f"``. Bytes are
    left side-by-side (``None``); a run with no f-string merges plain/raw/unicode
    sharing one prefix+quote by body concatenation; a run mixing an f-string with a
    plain string (no bytes, no raw) folds into one f-string with plain braces escaped.
    Runs of only f-strings are left alone. Caller re-checks the AST and drops a
    differing change, so subtle cases are caught.
    """
    parsed = []
    for kind, raw in pieces:
        pqb = _split_string_token(raw)
        if pqb is None:
            return None
        prefix, quote, body = pqb
        if "b" in prefix.lower():
            return None  # bytes: leave side-by-side
        parsed.append((kind, prefix, quote, body))
    if len({p[2] for p in parsed}) != 1:
        return None  # mixed quote style: not a safe textual merge
    quote = parsed[0][2]
    if not any(p[0] == "f" for p in parsed):
        # No f-string: merge plain/raw/unicode sharing one prefix by concatenation.
        if len({p[1].lower() for p in parsed}) != 1:
            return None
        return f"{parsed[0][1]}{quote}{''.join(p[3] for p in parsed)}{quote}"
    # f-string fold only when a plain string is glued onto an f-string; a run of
    # only f-strings is left side-by-side (folding long ones would force ruff to
    # re-wrap the surrounding statement).
    if all(p[0] == "f" for p in parsed):
        return None
    # raw mixed with f is too subtle (backslash + brace escaping) -> skip.
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
    """Whether an f-string fold at ``row[c0:c1]`` -> ``merged`` is safe to apply.

    Only ``assert`` wraps awkwardly when a message folds (ruff parenthesizes the
    condition once it no longer fits one line); every other construct wraps
    acceptably so is always allowed. An ``assert`` fold is allowed only if already
    one line, or its estimated folded one-line length fits the line length.
    """
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
    # Conservative over-estimate: join continuation lines with a single space
    # (ruff joins bracketed wraps with none), so borderline cases skip the fold.
    joined = " ".join(s.strip() for s in seg)
    return indent + len(joined) <= _LINE_LENGTH


def merge_adjacent_string_literals(text: str) -> tuple[str, bool]:
    """Merge adjacent string literals on ONE physical line into a single literal.

    Plain/raw/unicode runs merge by concatenation; an f-string + plain string folds
    into one f-string (plain braces escaped) only while the statement still fits one
    line. Runs of only f-strings, and bytes, are left side-by-side. The file AST is
    re-checked and a differing change dropped, so meaning never changes.
    """
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
        tree = ast.parse(text)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text, False

    lines = text.splitlines(keepends=True)
    pieces = _string_pieces(toks, lines)

    # Group consecutive mergeable pieces (str/f, single line, same physical line).
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
        # An f-string fold must not push its statement onto extra lines; a plain
        # concatenation always collapses cleanly so it skips this check.
        if any(kind == "f" for kind, _s, _e, _r in run) and not _fold_collapses(
            tree, lines, row, c0, c1, merged
        ):
            continue
        edits.append((row, c0, c1, merged))
    if not edits:
        return text, False

    for row, c0, c1, repl in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        ln = lines[row - 1]
        lines[row - 1] = ln[:c0] + repl + ln[c1:]
    out = "".join(lines)
    try:
        if ast.dump(ast.parse(text)) != ast.dump(ast.parse(out)):
            return text, False
    except SyntaxError:
        return text, False
    return out, True


def collapse_short_asserts(text: str) -> tuple[str, bool]:
    """Collapse a multi-line ``assert`` onto one line when it would fit.

    When the statement's estimated one-line length fits, strip the magic trailing
    commas (comma before a closer) holding it open so ruff rejoins it. Run BEFORE
    ruff format. Skips asserts with a comment (would oscillate). Stripping is
    non-semantic except for a one-element tuple; AST is re-checked and changing
    asserts left alone.
    """
    try:
        tree = ast.parse(text)
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text, False

    lines = text.splitlines(keepends=True)
    multiline = [
        (n.lineno, n.end_lineno)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assert) and (n.end_lineno or n.lineno) > n.lineno
    ]
    if not multiline:
        return text, False

    comment_rows = {t.start[0] for t in toks if t.type == tokenize.COMMENT}

    targets = []  # (lo, hi) spans whose one-line form fits and have no comment
    for lo, hi in multiline:
        if any(lo <= r <= hi for r in comment_rows):
            continue  # a comment would keep ruff multi-line -> never collapses
        seg = [lines[k].rstrip("\n") for k in range(lo - 1, hi)]
        indent = len(seg[0]) - len(seg[0].lstrip())
        # Over-estimate (join with a space; keep the comma) so a "fits" verdict
        # is always at least as long as ruff's real one-line output -> no fight.
        if indent + len(" ".join(s.strip() for s in seg)) <= _LINE_LENGTH:
            targets.append((lo, hi))
    if not targets:
        return text, False

    # Trailing commas (a ',' whose next significant token is a closer), grouped
    # by the target assert they belong to.
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
    for positions in by_target.values():  # apply per assert; skip any that break AST
        trial = working[:]
        for row, col in sorted(positions, reverse=True):
            ln = trial[row - 1]
            if col < len(ln) and ln[col] == ",":
                trial[row - 1] = ln[:col] + ln[col + 1 :]
        try:
            if ast.dump(ast.parse("".join(trial))) == base_dump:
                working, changed = trial, True
        except SyntaxError:
            pass
    return ("".join(working), True) if changed else (text, False)


def process_file(path: Path, pre: bool = False) -> bool:
    try:
        with tokenize.open(path) as handle:
            original = handle.read()
            encoding = handle.encoding
    except (OSError, SyntaxError) as exc:  # SyntaxError from tokenize on invalid python
        print(f"Failed to read {path}: {exc}", file=sys.stderr)
        return False

    if pre:
        # Pre-ruff: normalize def-signature magic commas (>=3 params + a default
        # add so ruff forces one-per-line; everything else strips so ruff
        # collapses), and strip the magic trailing comma from a short multi-line
        # assert so ruff joins it onto one line. Everything else runs post-ruff.
        updated, normalized = normalize_def_trailing_comma(original)
        updated, collapsed = collapse_short_asserts(updated)
        if normalized or collapsed:
            _atomic_write_text(path, updated, encoding)
            return True
        return False

    updated, changed = enforce_spacing(original)
    updated, blanked = remove_blank_after_short_import(updated)
    updated, merged = merge_adjacent_string_literals(updated)
    updated, removed = remove_redundant_passes(updated)
    if changed or blanked or merged or removed:
        _atomic_write_text(path, updated, encoding)
        return True
    return False


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Python files to fix")
    parser.add_argument(
        "--pre",
        action="store_true",
        help="pre-ruff pass: normalize def-signature commas + collapse short multi-line asserts",
    )
    args = parser.parse_args(argv)

    touched: list[Path] = []
    self_path = Path(__file__).resolve()

    for entry in args.files:
        path = Path(entry)
        # Skip modifying this script to avoid self-edit loops.
        if path.resolve() == self_path:
            continue
        if not path.exists() or path.is_dir():
            continue
        if process_file(path, pre=args.pre):
            touched.append(path)

    if touched:
        for path in touched:
            print(f"Adjusted kwarg spacing in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
