#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# .github/workflows/security-audit.yml's pip-scan-packages job depends
# on this file existing at scripts/scan_packages.py.
"""
scan_packages.py -- Standalone pre-install package scanner.

Downloads PyPI packages WITHOUT installing them and inspects archive
contents for malicious patterns: weaponized .pth files, credential
stealers, obfuscated payloads, install-time droppers.

Motivated by the litellm 1.82.7/1.82.8 supply chain attack (March 2026).
Single file, stdlib only, Python 3.10+.

Examples:
    # Scan specific packages
    python scan_packages.py requests==2.32.5
    python scan_packages.py fastapi uvicorn pydantic

    # Scan requirements files
    python scan_packages.py -r requirements.txt
    python scan_packages.py -r base.txt -r extras.txt

    # Auto-discover requirements files in a project
    python scan_packages.py -d ./my-project/

    # Scan with full transitive dependency tree
    python scan_packages.py --with-deps unsloth unsloth-zoo

    # Scan + auto-fix CRITICAL findings in requirements files
    python scan_packages.py --fix -r requirements.txt
    python scan_packages.py --fix --max-search 20 -r requirements.txt

    # Triage to a baseline once, then gate on anything NEW
    python scan_packages.py -r requirements.txt --write-baseline scripts/scan_packages_baseline.json
    python scan_packages.py -r requirements.txt   # auto-loads the baseline, exits 0 if only baselined findings remain

False positives:
    .py files are scanned code-only: comments and bare docstrings/doctests are
    blanked before pattern matching (line numbers preserved), so prose, usage
    examples and `>>>` doctests cannot trip a finding. Residual findings that
    are genuine library behavior (a HTTP client reading HF_TOKEN, a vendored
    test fixture) are suppressed via a reviewed baseline allowlist, matched on
    (package, package-relative file, check, evidence hash). A new check, or
    changed flagged code under the same check, reopens the finding; version
    bumps and line shifts do not. This mirrors the Hugging Face Hub approach
    (ClamAV/picklescan: low-FP, signature/structural, surface status).

Exit codes:
    0 -- no non-baselined CRITICAL or HIGH findings (or --write-baseline)
    1 -- non-baselined CRITICAL or HIGH findings detected
    2 -- no packages specified, or scan incomplete (pip download failure)
"""

import argparse
import ast
import atexit
import bisect
import functools
import hashlib
import io
import json
import keyword
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tokenize
import unicodedata
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


# Severity
CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"

SEVERITY_ORDER = {CRITICAL: 0, HIGH: 1, MEDIUM: 2}

# Hard pin-blocks for confirmed malicious PyPI versions (Socket.dev 2026-05-12
# Mini Shai-Hulud wave; earlier Semgrep/Endor reports for `lightning`).
BLOCKED_PYPI_VERSIONS: dict[str, set[str]] = {
    "guardrails-ai": {"0.10.1"},
    "mistralai": {"2.4.6"},
    "lightning": {"2.6.2", "2.6.3"},
}

# Pattern definitions

# Subprocess / OS exec patterns
RE_SUBPROCESS = re.compile(
    r"\bsubprocess\s*\.\s*(Popen|call|run|check_call|check_output)\b"
    r"|\bos\s*\.\s*(system|popen|exec[lv]p?e?)\b",
)

# Encoding / obfuscation
RE_BASE64 = re.compile(
    r"\bbase64\s*\.\s*(b64decode|decodebytes|b32decode|b16decode)\b|\bcodecs\s*\.\s*decode\b",
)

# exec / eval. The BUILTINS only: `\b` matches after a dot, so the bare form also
# caught `model.eval()`, which every torch package calls. Paired with a dynamic
# import that promoted ordinary inference code to a HIGH "obfuscation + exec/eval",
# and HIGH is what fails the scan. A payload calls the builtin; a method named eval
# on some object is not one, so require no attribute access in front.
#
# Routes to the builtin, all of which must stay detectable:
#   1. bare              `exec(payload)`
#   2. through builtins  `builtins.exec(payload)`, `(__builtins__).eval(payload)`
#   3. aliased module    `import builtins as b` ... `b.exec(payload)`
#   4. aliased function  `from builtins import exec as run` ... `run(payload)`
#   5. dynamic module    `__import__('builtins').exec(payload)`
#   6. parked module     `b = __import__('builtins')` ... `b.exec(payload)`
#   7. computed name     `__import__(name).exec(payload)`
# Telling those apart from `model.eval()` is a question about the *structure* of
# the call, and raw text cannot answer it. `model . eval()` and `model. eval()`
# are legal Python, so a fixed-width lookbehind for the dot reads them as the
# bare builtin - the exact false positive this rule exists to remove. `import
# os, builtins as b` hides the alias behind a comma. A string literal is not
# code, yet a text scan reads `decoy = """\nb = harmless\n"""` as a rebinding of
# `b` and drops the alias. And one regex branch per alias costs O(aliases) per
# character, so 15,000 aliases stall the scan on members allowed up to 64 MiB.
#
# `_ExecEvalPattern` therefore adjudicates on the token stream: strings are
# single tokens (so their contents are never code), whitespace between tokens is
# irrelevant by construction, imports are parsed rather than pattern-matched,
# and an alias is a set membership test rather than a regex branch. The regexes
# below stay as the fallback for source that will not tokenize - the scanner
# reads arbitrary third-party files, including Python 2 and truncated ones.
#
# Route 6 stops at the assignment whose value names the module: `b =
# __import__('builtins')`, `b = importlib.import_module('builtins')` (through an
# alias of `importlib` too), and `b = builtins`. Following `c = b` from there is
# general dataflow with no bound, so a chain of aliases is not tracked; the
# first link is what the pre-token scan happened to catch, because its bare
# `exec\\s*\\(` also matched the `b.exec(` at the end of the chain.
#
# Route 7 is not that dataflow wearing a different hat: the receiver *is* the
# loader call, so what it is handed never has to be traced, and a call on one is
# taken as the module whatever the argument says. `name = 'builtins'` first and
# then `b = __import__(name)` is the untracked chain again, and stays a limit.

_EXEC_NAMES = frozenset(("exec", "eval"))
_BUILTINS_NAMES = frozenset(("builtins", "__builtins__"))
_BUILTINS_LEN = len("builtins")
# The loader calls a receiver can be: `__import__(...)` is a builtin and
# `import_module(...)` needs only `from importlib import import_module`.
# `__import__` is a builtin, so it means the loader in any file without being
# imported. `import_module` is not: it is the standard loader only when the file
# actually imported it, and `_collect_import_bindings` adds it when it did.
# Trusting the bare name made `import_module(name).eval(x)` HIGH in any file that
# defines or imports an unrelated function of that name.
_DEFAULT_LOADER_FUNCS = frozenset(("__import__",))
_OPENERS = frozenset(("(", "[", "{"))
_CLOSERS = frozenset((")", "]", "}"))
_COMPARISON_OPS = frozenset(("==", "!=", "<=", ">="))
# Dropped from the token stream before statements are assembled: they carry no
# binding and no call, and skipping them is what makes whitespace and comments
# between a receiver and its `.exec(` irrelevant.
_TOK_IGNORED = frozenset(
    (
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
    )
)
# PEP 701 (Python 3.12) made `tokenize` split an f-string into its literal and
# expression parts. Before that the whole literal arrives as one STRING token,
# so `f'{exec(payload)}'` carries a call the statement scan cannot see even
# though it runs at import. requires-python is >=3.9, so the scanner has to
# handle both; on 3.12+ this flag is False and the extra work never happens.
_FSTRING_OPAQUE = not hasattr(tokenize, "FSTRING_START")
# The three token types 3.12 splits an f-string into. `-1` never matches a real
# token type, so the readers below can test against them unconditionally.
_FSTRING_START = getattr(tokenize, "FSTRING_START", -1)
_FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", -1)
_FSTRING_END = getattr(tokenize, "FSTRING_END", -1)
# Longest quote first, so `"""` is not read as an empty `""`. A pre-3.12
# f-string may not reuse the quote style that delimits it, which is what caps
# how deep one opaque STRING token can nest them.
_FSTRING_QUOTES = ('"""', "'''", '"', "'")
_MAX_FSTRING_NESTING = len(_FSTRING_QUOTES)
# Longest escape that can stand for one character is `\N{...}`, and no Unicode
# name runs to 500 characters, so a literal longer than this cannot spell an
# eight-character module name however it is escaped - and a member is allowed
# to hold a 64 MiB one.
_MAX_LITERAL_DECODE = 512
# An escape sequence that resolves to a character: `'buil\x74ins'` names the
# `builtins` module without spelling it, so a file that carries one has to be
# read even though the plain word never appears in it.
_RE_STRING_ESCAPE = re.compile(
    r"\\(?:x[0-9a-fA-F]{2}|[0-7]{1,3}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|N\{)"
)

# `__import__('built' 'ins')` names the module without the word ever appearing,
# because the compiler joins adjacent literals and folds `+` between two of
# them. Only a run whose first fragment is a non-empty proper prefix of the name
# can spell it, so the ordinary file that merely splits a long message across
# two lines still skips the pass.
_RE_SPLIT_BUILTINS = re.compile(
    r"""['"](?:b|bu|bui|buil|built|builti|builtin)['"]"""
    r"""(?:\s|\\|\+|\#[^\n]*\n)*[A-Za-z]{0,2}['"]"""
)


class _Span:
    """`re.Match`-shaped result. `_extract_evidence` needs `start()`/`end()`."""

    __slots__ = ("_start", "_end")

    def __init__(self, start: int, end: int):
        self._start = start
        self._end = end

    def start(self, group: int = 0) -> int:
        return self._start

    def end(self, group: int = 0) -> int:
        return self._end


class _Offsets:
    """Lazy (row, col) -> character offset table for one text.

    Built on the first hit, not on construction: most members match nothing,
    and a line table for every 64 MiB member scanned would cost more than the
    match it is there to locate.
    """

    __slots__ = ("text", "starts")

    def __init__(self, text: str):
        self.text = text
        self.starts: "list[int] | None" = None

    def of(self, row: int, col: int) -> int:
        starts = self.starts
        if starts is None:
            # `io.StringIO.readline` breaks on "\n" only, so tokenize's row
            # numbers count "\n"-separated lines - split the same way or the
            # offsets drift on a file with lone "\r"s.
            starts = [0]
            pos = 0
            for line in self.text.split("\n"):
                pos += len(line) + 1
                starts.append(pos)
            self.starts = starts
        if not 1 <= row <= len(starts):
            return 0
        return starts[row - 1] + col


# The Unicode categories a Python identifier keeps but `\w` throws away. `\w`
# matches what `str.isalnum` accepts, and a combining mark is not alphanumeric,
# so `e` + U+0301 - the NFKC-equivalent of `é`, and a legal identifier - splits
# into `e` and a character no pattern here matches. An alias spelled that way
# was then read as `e` and its call missed entirely.
_MARK_CATEGORIES = frozenset(("Mn", "Mc", "Me"))


def _mark_class(text: str) -> str:
    """The combining marks `text` holds, as the body of a regex character class.

    Built from the marks actually present rather than from every one Unicode
    defines, because there are thousands and `re` cannot name a category: a file
    that holds none - every ASCII file, and nearly every other - answers with one
    flag test and keeps the module-level patterns.
    """
    if text.isascii():
        return ""
    return "".join(sorted(c for c in set(text) if unicodedata.category(c) in _MARK_CATEGORIES))


def _ident(name: str) -> str:
    """`name` as the interpreter resolves it.

    PEP 3131 normalizes every identifier to NFKC before it is looked up, while
    `tokenize` hands back the source spelling. So `import builtins as b` followed
    by a call on `\U0001d553` is a call on `b`, and comparing the raw tokens
    misses it - as does the reverse, an alias declared in a decorated spelling
    and rebound in ASCII, which would leave the alias live and hand back the
    `model.eval()` false positive. ASCII cannot change under NFKC, and the test
    for that is a flag on the string object, so ordinary source pays nothing.
    """
    return name if name.isascii() else unicodedata.normalize("NFKC", name)


def _mentions_name(text: str, names) -> bool:
    """Whether any name in `names` occurs in `text` as a whole identifier.

    One pass over the text, not one substring search per name. `_extract_evidence`
    re-searches every line of the file, so testing each alias against each line
    is quadratic: a hostile member declaring N function aliases beside N other
    lines costs N*N searches, on members allowed up to 64 MiB.
    """
    if not names:
        return False
    for m in _ident_patterns(_mark_class(text)).word.finditer(text):
        if _ident(m.group()) in names:
            return True
    return False


def _continues_name(stmt: list, tok) -> bool:
    """Whether `tok` is a combining mark that belongs to the name before it.

    It has to be adjacent: a mark separated by whitespace starts nothing and
    ends nothing, and gluing it on would invent an identifier the source does
    not spell.
    """
    return bool(
        stmt
        and stmt[-1].type == tokenize.NAME
        and tok.start == stmt[-1].end
        and tok.string
        and all(unicodedata.category(c) in _MARK_CATEGORIES for c in tok.string)
    )


def _statements(text: str, failed: list):
    """Yield each logical statement of `text` as a list of significant tokens.

    A tokenizer error ends the stream and appends to `failed`; the statements
    produced before it are still valid, and the caller unions them with the
    regex fallback so a truncated parse can only add detections, never remove
    them. Statements are yielded and dropped one at a time, so peak memory is
    the largest statement rather than the whole token list.

    Identifiers arrive NFKC-normalized, the spelling the interpreter resolves.
    Doing it here rather than at each comparison keeps the binding side and the
    call side in one alphabet, which is what makes an alias declared in one
    spelling and used in another the same name to every pass below. Positions
    are untouched, so evidence still points at the source as written.
    """
    stmt: list = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            ttype = tok.type
            if ttype in _TOK_IGNORED:
                continue
            if ttype == tokenize.ERRORTOKEN and _continues_name(stmt, tok):
                # Below 3.12 `tokenize` splits identifiers on `\w`, which throws
                # away a combining mark: `e` + U+0301 - the NFKC-equivalent of
                # `é`, and a legal identifier - arrives as NAME `e` and an
                # ERRORTOKEN, and no error is raised, so the whole token pass ran
                # on a name the interpreter never sees. Gluing the mark back on
                # is what makes the alias and its call the same name again.
                stmt[-1] = stmt[-1]._replace(
                    string = unicodedata.normalize("NFKC", stmt[-1].string + tok.string),
                    end = tok.end,
                )
                continue
            if ttype == tokenize.NAME and not tok.string.isascii():
                tok = tok._replace(string = unicodedata.normalize("NFKC", tok.string))
            if ttype == tokenize.NEWLINE or ttype == tokenize.ENDMARKER:
                if stmt:
                    yield from _with_suite_tail(stmt)
                    stmt = []
                continue
            if ttype == tokenize.OP and tok.string == ";":
                if stmt:
                    yield from _with_suite_tail(stmt)
                    stmt = []
                continue
            stmt.append(tok)
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError, MemoryError):
        failed.append(True)
    if stmt:
        yield from _with_suite_tail(stmt)


_SUITE_HEADS = frozenset(
    (
        "if",
        "elif",
        "else",
        "for",
        "while",
        "with",
        "try",
        "except",
        "finally",
        "def",
        "class",
        "async",
        "match",
        "case",
    )
)


def _with_suite_tail(stmt: list):
    """`stmt`, plus the first simple statement of a one-line suite it heads.

    A semicolon splits the later statements of `def f(): import builtins as b;
    b.exec(BLOB)`, but the first one stays glued to the header, so every pass
    that keys off the head token sees `def` and the import binding is never
    recorded. Yielding the tail as its own statement is what puts that head
    back in view.

    The original is yielded too, unsplit. Every consumer either collects
    bindings or reports call spans, so an extra statement can only add to what
    is found - and the first depth-0 colon is the header's for real code, but
    not for every oddity Python's grammar admits (`if lambda: 1: pass`), and a
    split that lands in the wrong place must not be able to lose a detection.
    """
    yield stmt
    head = stmt[0]
    if head.type != tokenize.NAME or head.string not in _SUITE_HEADS:
        return
    if len(stmt) > 1 and stmt[1].type == tokenize.OP and stmt[1].string in (":", "="):
        return  # `match: int = 5` is an annotated assignment, not a suite
    depth = 0
    for i, tok in enumerate(stmt):
        if tok.type != tokenize.OP:
            continue
        if tok.string in _OPENERS:
            depth += 1
        elif tok.string in _CLOSERS:
            depth -= 1
        elif tok.string == ":" and depth == 0:
            if i + 1 < len(stmt):
                yield stmt[i + 1 :]
            return


def _header_end_index(stmt: list) -> int:
    """Where a compound header ends: the index of its first depth-0 colon.

    `def f(): run = model` keeps the tail glued to the header, so the last
    token is the tail's, not the header's. The body a suite governs starts
    after the colon.
    """
    depth = 0
    for i, tok in enumerate(stmt):
        if tok.type != tokenize.OP:
            continue
        if tok.string in _OPENERS:
            depth += 1
        elif tok.string in _CLOSERS:
            depth -= 1
        elif tok.string == ":" and depth == 0:
            return i
    return len(stmt) - 1


def _header_end(stmt: list):
    """The token the `def`/`class` header ends on."""
    return stmt[_header_end_index(stmt)]


def _split_top(toks: list, sep: str = ",") -> list:
    """Split a token list on `sep` at bracket depth 0."""
    parts: list = []
    cur: list = []
    depth = 0
    for tok in toks:
        if tok.type == tokenize.OP:
            text = tok.string
            if text in _OPENERS:
                depth += 1
            elif text in _CLOSERS:
                depth -= 1
            elif text == sep and depth == 0:
                parts.append(cur)
                cur = []
                continue
        cur.append(tok)
    parts.append(cur)
    return parts


def _name_index(toks: list, name: str) -> "int | None":
    for i, tok in enumerate(toks):
        if tok.type == tokenize.NAME and tok.string == name:
            return i
    return None


def _string_body(literal: str) -> "str | None":
    """The value of a string token, or None for bytes/computed/odd quoting.

    Escapes are decoded, because the interpreter compares the module name the
    literal evaluates to, not the way it was spelled: `__import__('buil\\x74ins')`
    loads exactly what `__import__('builtins')` loads, and returning the source
    text would read the two as different modules. Raw literals have no escapes
    to decode, and a literal far longer than any module name cannot be one, so
    both skip the parse.

    An f-string with no replacement field evaluates to the same constant as the
    plain literal - `__import__(f'builtins')` loads the module - so rejecting
    every `f` prefix let one spelling of the module name through unread. One
    holding a field is computed at runtime and stays unsupported.
    """
    i = 0
    while i < len(literal) and literal[i] not in "\"'":
        i += 1
    raw_prefix = literal[:i]
    prefix = raw_prefix.lower()
    if "b" in prefix:
        return None
    fstring = "f" in prefix
    body = literal[i:]
    for quote in ('"""', "'''", '"', "'"):
        if body.startswith(quote) and body.endswith(quote) and len(body) >= 2 * len(quote):
            body = body[len(quote) : -len(quote)]
            if fstring and ("{" in body or "}" in body):
                return None  # a replacement field, or the `{{` that escapes one
            if "\\" not in body or "r" in prefix or len(body) > _MAX_LITERAL_DECODE:
                return body
            # `literal_eval` refuses a JoinedStr however plain it is, so the `f`
            # comes off before the escapes are decoded.
            plain = literal if not fstring else _drop_f_prefix(raw_prefix) + literal[i:]
            try:
                value = ast.literal_eval(plain)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                return body
            return value if isinstance(value, str) else None
    return None


def _drop_f_prefix(prefix: str) -> str:
    return prefix.replace("f", "").replace("F", "")


def _fstring_const(toks: list, i: int) -> tuple:
    """`(value, index past the literal)` for the f-string run starting at `i`.

    3.12 splits an f-string into START / MIDDLE... / END instead of handing over
    one STRING token, so a constant one has to be rejoined here rather than in
    `_string_body`. Only MIDDLE text may sit in the run: a replacement field
    arrives as `{` ... `}` OP tokens, and that value is computed at runtime.
    `(None, i + 1)` for anything else, so the caller still makes progress.
    """
    parts = []
    j = i + 1
    while j < len(toks) and toks[j].type == _FSTRING_MIDDLE:
        parts.append(toks[j].string)
        j += 1
    if j >= len(toks) or toks[j].type != _FSTRING_END:
        return None, i + 1
    body = "".join(parts)
    opener = toks[i].string
    q = 0
    while q < len(opener) and opener[q] not in "\"'":
        q += 1
    prefix = opener[:q]
    if "\\" not in body or "r" in prefix.lower() or len(body) > _MAX_LITERAL_DECODE:
        # MIDDLE text already carries `{{` as the single brace it evaluates to;
        # only backslash escapes are still spelled the source way.
        return body, j + 1
    quote = opener[q:]
    try:
        value = ast.literal_eval(_drop_f_prefix(prefix) + quote + body + quote)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return body, j + 1
    return (value if isinstance(value, str) else None), j + 1


def _collect_import_bindings(
    stmt: list,
    modules: set,
    funcs: set,
    loaders: "_Loaders | None" = None,
) -> bool:
    """Record `builtins` aliases bound by `stmt`. True if `stmt` is an import.

    Parsing the statement rather than matching `import\\s+builtins` means the
    alias is found wherever it sits in the list: `import os, builtins as b`
    binds `b` just as `import builtins as b` does.

    `loaders`, when given, also collects the local names of the two module
    loaders `_assignment_bindings` understands, so `import importlib as il` ...
    `b = il.import_module('builtins')` is read the same as the unaliased call.
    """
    head = stmt[0]
    if head.type != tokenize.NAME:
        return False
    if head.string == "import":
        for part in _split_top(stmt[1:]):
            if not part:
                continue
            as_at = _name_index(part, "as")
            if as_at is None or as_at + 1 >= len(part):
                continue  # `import importlib` needs no record: it is a default
            dotted = "".join(t.string for t in part[:as_at])
            alias = part[as_at + 1]
            if alias.type != tokenize.NAME:
                continue
            if dotted == "builtins":
                modules.add(alias.string)
            elif loaders is not None and dotted == "importlib":
                loaders.modules.add(alias.string)
        return True
    if head.string != "from":
        return False
    import_at = _name_index(stmt, "import")
    if import_at is None:
        return True
    dotted = "".join(t.string for t in stmt[1:import_at])
    if dotted != "builtins":
        if loaders is not None and dotted == "importlib":
            items = stmt[import_at + 1 :]
            if items and items[0].type == tokenize.OP and items[0].string == "(":
                items = items[1:-1] if items[-1].string == ")" else items[1:]
            for part in _split_top(items):
                names = [t for t in part if t.type == tokenize.NAME]
                if not names or names[0].string != "import_module":
                    continue
                as_at = _name_index(part, "as")
                if as_at is not None and as_at + 1 < len(part):
                    loaders.funcs.add(part[as_at + 1].string)
                else:
                    loaders.funcs.add("import_module")
        return True
    items = stmt[import_at + 1 :]
    if items and items[0].type == tokenize.OP and items[0].string == "(":
        items = items[1:-1] if items[-1].string == ")" else items[1:]
    for part in _split_top(items):
        names = [t for t in part if t.type == tokenize.NAME]
        if not names:
            continue
        imported = names[0].string
        # `from builtins import __import__ as load` renames the loader itself, so
        # `load(name).exec(...)` is the same builtin call as `__import__(name).exec(...)`.
        loader = loaders is not None and imported == "__import__"
        if not loader and imported not in _EXEC_NAMES:
            continue
        as_at = _name_index(part, "as")
        if as_at is not None and as_at + 1 < len(part):
            local = part[as_at + 1].string
        else:
            local = imported
        if loader:
            loaders.funcs.add(local)
        else:
            funcs.add(local)
    return True


class _Loaders:
    """The local names one file binds to the module loaders, plus their defaults.

    `importlib` is the one spelling that needs no import of its own to be
    meaningful in the source, because a file cannot reach the module without
    naming it. `import_module` is an ordinary function name until the file
    imports it from `importlib`, so it starts empty and the import collector
    fills it in.
    """

    __slots__ = ("modules", "funcs")

    def __init__(self):
        self.modules = {"importlib"}
        self.funcs = set()


def _is_loader_name(owner: str, name: str, loaders, loader_modules) -> bool:
    """Whether `owner.name` (or bare `name`, with `owner` empty) is a module loader.

    `import_module` is the standard loader only when it is reached through
    `importlib` or was imported from it. Treating the bare name as a loader in
    any file made `import_module(x).eval(y)` a builtins call wherever an
    unrelated function happened to carry that name.
    """
    if owner:
        return name == "import_module" and owner in loader_modules
    return name in loaders


def _strip_parens(toks: list) -> list:
    while (
        len(toks) >= 2
        and toks[0].type == tokenize.OP
        and toks[0].string == "("
        and toks[-1].type == tokenize.OP
        and toks[-1].string == ")"
    ):
        depth = 0
        for j, tok in enumerate(toks):
            if tok.type == tokenize.OP:
                if tok.string in _OPENERS:
                    depth += 1
                elif tok.string in _CLOSERS:
                    depth -= 1
                    if depth == 0 and j != len(toks) - 1:
                        return toks  # `(a).b` - the parens are not the outermost
        toks = toks[1:-1]
    return toks


def _matching_opener(toks: list, close_at: int) -> "int | None":
    """The index of the bracket `toks[close_at]` closes, or None if unbalanced."""
    depth = 0
    for i in range(close_at, -1, -1):
        tok = toks[i]
        if tok.type != tokenize.OP:
            continue
        if tok.string in _CLOSERS:
            depth += 1
        elif tok.string in _OPENERS:
            depth -= 1
            if depth == 0:
                return i
    return None


def _const_string(toks: list) -> "str | None":
    """The value of a constant string expression, or None if it is not one.

    The compiler joins adjacent literals and folds `+` between two of them, so
    `__import__('built' 'ins')` loads exactly what `__import__('builtins')`
    loads; reading only a lone STRING token let those spellings pass as an
    unknown module and demoted a real builtins call to the non-blocking
    obfuscation finding. Parentheses are stripped for the same reason. Anything
    with a runtime part - a name, a call, a bytes or f-string literal - is not a
    constant and returns None.
    """
    toks = _strip_parens(toks)
    if not toks:
        return None
    out = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok.type == tokenize.OP and tok.string == "+" and out:
            i += 1
            continue
        if tok.type == _FSTRING_START:
            body, i = _fstring_const(toks, i)
        elif tok.type == tokenize.STRING:
            body = _string_body(tok.string)
            i += 1
        else:
            return None
        if body is None:
            return None
        out.append(body)
    return "".join(out)


def _loads_builtins(
    value: list,
    loaders: _Loaders,
    known: "set | None" = None,
) -> bool:
    """Whether `value` plainly evaluates to the `builtins` module.

    The forms that need no dataflow to read: the module named outright, the two
    loader calls that take its name as a literal, and a copy of an alias the
    file has already bound to it. `known` is that set of aliases; statements
    arrive in source order and each copy adds its target to it, so
    `import builtins as b` ... `c = b` ... `d = c` ... `d.exec(BLOB)` is read
    through in one pass rather than by chasing a chain per call site. Refusing
    the copy left that an enforced-scan bypass, since the alias was reported
    only under its original name.
    """
    value = _strip_parens(value)
    if not value:
        return False
    if len(value) == 1:
        return value[0].type == tokenize.NAME and (
            value[0].string in _BUILTINS_NAMES or (known is not None and value[0].string in known)
        )
    if not (value[-1].type == tokenize.OP and value[-1].string == ")"):
        return False
    # The module name is the loader's first argument, not necessarily its only
    # one: `__import__('builtins', fromlist=[])` and
    # `import_module('builtins', package=None)` load the same module.
    open_at = _matching_opener(value, len(value) - 1)
    if open_at is None:
        return False
    if _const_string(_module_argument(_split_top(value[open_at + 1 : -1]))) != "builtins":
        return False
    call = value[:open_at]
    if len(call) == 1:
        # `__import__('builtins')`, or `import_module('builtins')` for a name
        # `from importlib import import_module` bound.
        return call[0].type == tokenize.NAME and (
            call[0].string == "__import__" or call[0].string in loaders.funcs
        )
    return (
        len(call) == 3
        and call[0].type == tokenize.NAME
        and call[0].string in loaders.modules
        and call[1].type == tokenize.OP
        and call[1].string == "."
        and call[2].type == tokenize.NAME
        and call[2].string == "import_module"
    )


def _bare_singleton(toks: list) -> "list | None":
    """The sole element of an unbracketed one-tuple `x,`, or None.

    `b, = (builtins,)` binds exactly what `[b] = [builtins]` binds; without the
    brackets there is nothing for `_sequence_element` to strip, so the trailing
    comma is read here instead.
    """
    if len(toks) < 2 or toks[-1].type != tokenize.OP or toks[-1].string != ",":
        return None
    head = toks[:-1]
    return head if _name_index_op(head, ",") is None else None


def _module_argument(args: list) -> list:
    """The tokens naming the module a loader call loads.

    Both loaders call that parameter `name`, and both accept it by keyword:
    `__import__(name='builtins')` and `importlib.import_module(name='builtins')`
    return the module just as the positional spelling does. Handing the whole
    argument group to the constant reader made the leading `name` `=` tokens
    reject it, so parking either call in a name and calling `exec` through it
    read as an unknown module. An empty list for a call whose first argument is
    some other keyword and which names no `name=`, since then nothing here says
    what it loads.
    """
    for i, arg in enumerate(args):
        eq = _name_index_op(arg, "=")
        if eq == 1 and arg[0].type == tokenize.NAME:
            if arg[0].string == "name":
                return arg[2:]
        elif eq is None and i == 0:
            return arg
    return []


def _sequence_element(toks: list) -> "list | None":
    """The sole element of a one-element `[...]` or `(...)`, or None.

    Nothing is stripped unless the brackets really enclose the whole expression
    and hold exactly one item: `[builtins] + rest` is not a sequence display,
    and `[a, builtins]` does not put the module anywhere `[b] = ...` would find
    it. A single trailing comma still leaves one item - `(builtins,)` is the
    one-tuple `(b,) = ...` unpacks - so it is the one comma allowed through;
    rejecting it read a working spelling of the binding as a tuple target and
    dropped the alias.
    """
    if len(toks) < 3 or toks[0].type != tokenize.OP or toks[-1].type != tokenize.OP:
        return None
    if (toks[0].string, toks[-1].string) not in (("[", "]"), ("(", ")")):
        return None
    end = len(toks) - 1
    depth = 0
    for i, tok in enumerate(toks):
        if tok.type != tokenize.OP:
            continue
        if tok.string in _OPENERS:
            depth += 1
        elif tok.string in _CLOSERS:
            depth -= 1
            if depth == 0 and i != len(toks) - 1:
                return None  # the opener closed early: not the outermost bracket
        elif tok.string == "," and depth == 1:
            if i != len(toks) - 2:
                return None  # a second item, so this is not a one-element display
            end = i
    return toks[1:end] or None


def _unwrapped_target(group: list) -> tuple:
    """`(name, came from a one-element sequence)` for an assignment target.

    `(b) = builtins` and `[b] = [builtins]` bind `b` exactly as `b = builtins`
    does; requiring the target to be a single NAME token read both as a tuple or
    subscript target and dropped the alias, so the `b.exec(...)` below went
    unflagged. `(name, False)` for the plain form, `(None, False)` for a target
    this pass cannot reduce to one name.

    A trailing comma is what tells the two apart: `(b)` is `b` in parentheses,
    while `(b,)` and `b,` are one-tuples that unpack a one-element sequence.
    """
    unpacked = False
    if len(group) == 1 and group[0].type == tokenize.NAME:
        return group[0].string, unpacked
    bare = _bare_singleton(group)
    if bare is not None:
        group, unpacked = bare, True
        if len(group) == 1 and group[0].type == tokenize.NAME:
            return group[0].string, unpacked
    inner = _sequence_element(group)
    while inner is not None:
        if group[0].string == "[" or (group[-2].type == tokenize.OP and group[-2].string == ","):
            unpacked = True  # a real destructuring, so the value is a sequence too
        group = inner
        if len(group) == 1 and group[0].type == tokenize.NAME:
            return group[0].string, unpacked
        inner = _sequence_element(group)
    return None, False


def _assignment_bindings(
    stmt: list,
    loaders: _Loaders,
    known: "set | None" = None,
) -> list:
    """The names `stmt` assigns the `builtins` module to, in source order.

    `b = __import__('builtins')` reaches the builtin through `b.exec(...)` just
    as `import builtins as b` does, and the token pass only follows names it
    knows are the module. Returns an empty list for every other statement,
    including `b += ...` and `b = load_model()`, so the caller can also use a
    non-empty result to mean "this statement is a binding, not a rebinding".

    `known` are the aliases the file has already bound to the module, so a plain
    copy of one (`c = b`) binds it too.
    """
    # Every accepted form ends in the module - named outright, or as the closing
    # parenthesis of a loader call - and needs a plain `=`. That is one test on
    # the last token for most statements, and a single early-exiting pass for the
    # rest; nothing is allocated until a statement can actually be one.
    if len(stmt) < 3:
        return []
    last = stmt[-1]
    if last.type == tokenize.OP and last.string == "," and len(stmt) > 3:
        last = stmt[-2]  # `b, = builtins,`: the one-tuple ends in the module
    if last.type == tokenize.NAME:
        if last.string not in _BUILTINS_NAMES and not (known and last.string in known):
            return []
    elif not (last.type == tokenize.OP and last.string in (")", "]")):
        # A loader call may carry optional arguments after the module name
        # (`__import__('builtins', fromlist=[])`), so the closing parenthesis is
        # all that is fixed about it; the marker pass below is what rejects
        # `v = compute(0)`, and it allocates nothing either.
        return []
    marker = assign = False
    # The bodies of the adjacent literals seen so far, since the compiler joins
    # them: neither half of `'built' 'ins'` carries the module name, but the run
    # does. Dropped once it outgrows the name it could spell, so a statement
    # holding a long chain of literals still costs one comparison per token.
    joined = ""
    for tok in stmt:
        ttype = tok.type
        if ttype == tokenize.NAME:
            joined = ""
            if tok.string in _BUILTINS_NAMES or (known and tok.string in known):
                if assign:
                    break
                marker = True
        elif ttype == tokenize.OP:
            if tok.string != "+":
                joined = ""  # `+` between two literals folds; anything else ends the run
            if tok.string == "=":
                if marker:
                    break
                assign = True
        elif ttype == tokenize.STRING or ttype == _FSTRING_MIDDLE:
            # 3.12 hands the inside of an f-string over as MIDDLE text with no
            # quotes; wrapping it in a plain literal reuses the same decoder,
            # so `f'buil\\x74ins'` spells the module here exactly as `'buil\\x74ins'`
            # does. START and END carry no text and must not end the run, since
            # `'built' f'ins'` concatenates just like two plain literals.
            body = _string_body(
                tok.string if ttype == tokenize.STRING else "'''" + tok.string + "'''"
            )
            if joined is None or body is None or len(joined) + len(body) > _BUILTINS_LEN:
                joined = None
            else:
                joined += body
            if "builtins" in tok.string or joined == "builtins":
                if assign:
                    break
                marker = True
        elif ttype != _FSTRING_START and ttype != _FSTRING_END:
            joined = ""
    else:
        return []
    groups = _split_top(stmt, "=")
    if len(groups) < 2:
        return []
    value = groups[-1]
    names = []
    unpacked = False
    for group in groups[:-1]:
        colon_at = _name_index_op(group, ":")  # annotated target: `b: Any = value`
        if colon_at is not None:
            group = group[:colon_at]
        name, from_sequence = _unwrapped_target(group)
        if name is None:
            return []  # a tuple, attribute or subscript target: not a plain alias
        names.append(name)
        unpacked = unpacked or from_sequence
    if unpacked:
        # `[b] = [builtins]` binds exactly what `b = builtins` binds. It takes a
        # one-element sequence on BOTH sides to be that: `b = [builtins]` binds
        # the list itself, and `[b] = xs` takes an element out of something this
        # pass never read. `(builtins,)` and a bare `builtins,` are that
        # sequence too, which is what `(b,) = (builtins,)` unpacks.
        value = _sequence_element(value) or _bare_singleton(value)
        if value is None:
            return []
    if not _loads_builtins(value, loaders, known):
        return []
    return names


def _copied_alias(stmt: list) -> "str | None":
    """The name `stmt` copies into its target, for `c = b` and nothing else.

    The collecting pass reads the file's aliases before it knows where each one
    stops being the module, so a copy has to be re-checked once the rebindings
    are in hand: `import builtins as b` ... `b = model` ... `c = b` copies the
    model, and taking the copy on the first pass alone reported `c.exec(...)`.
    """
    groups = _split_top(stmt, "=")
    if len(groups) < 2:
        return None
    value = _strip_parens(groups[-1])
    if len(value) != 1 or value[0].type != tokenize.NAME:
        return None
    return value[0].string


def _walrus_bindings(
    stmt: list,
    loaders: _Loaders,
    known: "set | None" = None,
) -> list:
    """The names `stmt` binds to the `builtins` module with `:=`.

    `if (b := builtins):` parks the module exactly as `b = builtins` does, and an
    assignment expression is legal wherever an expression is, so the plain-`=`
    reader above never sees one.

    Each value runs to the bracket or separator that really ends it. Capping it
    at a fixed number of tokens instead was an evasion anyone could reach:
    padding a working binding with redundant parentheses or extra loader
    arguments pushed the module past the cutoff, `_loads_builtins` saw a
    truncated expression, and the `b.exec(...)` under it fell to the
    non-blocking obfuscation finding. The ends are found in ONE pass for the
    whole statement, so a right-nested chain of walruses costs its length rather
    than the sum of its values - which is what the cap was really guarding, on
    members allowed up to 64 MiB.
    """
    ends: dict = {}
    pending: list = []
    depth = 0
    for j, tok in enumerate(stmt):
        if tok.type != tokenize.OP:
            continue
        text = tok.string
        if text == ":=":
            if j and stmt[j - 1].type == tokenize.NAME:
                pending.append((j, depth))
            continue
        if text in _OPENERS:
            depth += 1
        elif text in _CLOSERS:
            depth -= 1
            # This bracket ends every value opened inside it, deepest first.
            while pending and pending[-1][1] > depth:
                ends[pending.pop()[0]] = j
        elif text in (",", ":") and pending and pending[-1][1] >= depth:
            while pending and pending[-1][1] >= depth:
                ends[pending.pop()[0]] = j
    names: list = []
    for i, _ in pending:
        ends[i] = len(stmt)
    for i in sorted(ends):
        if _loads_builtins(stmt[i + 1 : ends[i]], loaders, known):
            names.append(stmt[i - 1].string)
    return names


def _name_index_op(toks: list, text: str) -> "int | None":
    depth = 0
    for i, tok in enumerate(toks):
        if tok.type != tokenize.OP:
            continue
        if tok.string in _OPENERS:
            depth += 1
        elif tok.string in _CLOSERS:
            depth -= 1
        elif depth == 0 and tok.string == text:
            return i
    return None


def _add_assignment_targets(part: list, rebound: set) -> None:
    for item in _split_top(part):
        toks = item
        while toks and toks[0].type == tokenize.OP and toks[0].string in ("*", "**"):
            toks = toks[1:]
        if not toks:
            continue
        depth = 0
        for j, tok in enumerate(toks):  # annotated target: `name: T = value`
            if tok.type != tokenize.OP:
                continue
            if tok.string in _OPENERS:
                depth += 1
            elif tok.string in _CLOSERS:
                depth -= 1
            elif tok.string == ":" and depth == 0:
                toks = toks[:j]
                break
        if len(toks) == 1 and toks[0].type == tokenize.NAME:
            rebound.add(toks[0].string)
        elif len(toks) >= 2 and toks[0].type == tokenize.OP and toks[0].string in ("(", "["):
            _add_assignment_targets(toks[1:-1], rebound)


def _add_param_names(stmt: list, start: int, scoped: set, candidates: frozenset) -> None:
    """Record the parameters of a `def` header that shadow an alias.

    A parameter binds its name for the whole body and nothing outside it, so
    `from builtins import exec as run` ... `def process(run):` makes the
    `run(...)` written inside the function the callback rather than the builtin.
    Only a name that opens a parameter is one: an annotation or a default is an
    expression evaluated in the enclosing scope and binds nothing, and `*`, `**`
    and `/` are punctuation between parameters. Filtered to `candidates`, so a
    long parameter list of ordinary names costs a membership test each.
    """
    depth = 0
    opened = False  # the parameter list, not the type parameters of `def f[T]()`
    expect = False
    for tok in stmt[start:]:
        if tok.type == tokenize.OP:
            if tok.string in _OPENERS:
                if depth == 0 and tok.string == "(":
                    opened = True
                    expect = True
                depth += 1
                continue
            if tok.string in _CLOSERS:
                depth -= 1
                if depth <= 0:
                    if opened:
                        return
                    depth = 0
                continue
            if opened and depth == 1 and tok.string == ",":
                expect = True
            continue
        if expect and depth == 1 and tok.type == tokenize.NAME:
            if tok.string in candidates:
                scoped.add(tok.string)
            expect = False


def _add_capture_names(stmt: list, scoped: set, candidates: frozenset) -> None:
    """Record the capture patterns of a `case` header that shadow an alias.

    A bare name in a pattern binds the subject, so `case run:` makes the
    `run(...)` in the suite the subject rather than the imported builtin. The
    two shapes that name something without binding it are excluded: a value
    pattern (`case mod.run:`) and a class or keyword pattern (`case Run(x=1)`,
    where the `x` names an attribute). The guard is an ordinary expression, so
    the walk stops at the depth-0 `if` that opens it.
    """
    end = _header_end_index(stmt)
    depth = 0
    for j in range(1, end):
        tok = stmt[j]
        if tok.type == tokenize.OP:
            if tok.string in _OPENERS:
                depth += 1
            elif tok.string in _CLOSERS:
                depth -= 1
            continue
        if tok.type != tokenize.NAME:
            continue
        if depth == 0 and tok.string == "if":
            return  # the guard: what it names, it reads
        if tok.string not in candidates:
            continue
        prev = stmt[j - 1]
        if prev.type == tokenize.OP and prev.string == ".":
            continue  # `case mod.run:` reads a value, it does not bind one
        nxt = stmt[j + 1]
        if nxt.type == tokenize.OP and nxt.string in (".", "(", "="):
            continue  # a value pattern, a class pattern, or a keyword name
        scoped.add(tok.string)


def _collect_rebindings(
    stmt: list,
    rebound: set,
    candidates: frozenset,
    scoped: "set | None" = None,
    in_match: bool = False,
) -> None:
    """Record the names in `candidates` that `stmt` binds to something else.

    An alias that the file rebinds (`import builtins as m` ... `m = load()`) is
    not the module at the call site, and treating it as one is how `m.eval()`
    becomes a false positive again. Restricted to the handful of names that are
    actually aliases, so a statement that cannot change the answer costs one
    membership test per token instead of a full target analysis.

    `scoped` collects the bindings that reach the statement's own suite and
    nothing after it - a loop target, an exception target, a parameter. They are
    separated because the difference decides both directions: `for b in ():`
    never runs its body, so cancelling `b` past the loop would suppress a real
    `b.exec(...)` below it, while inside the suite the name really is the item,
    the exception or the argument.
    """
    body = rebound if scoped is None else scoped
    for tok in stmt:
        if tok.type == tokenize.NAME and tok.string in candidates:
            break
    else:
        return
    if in_match and stmt[0].type == tokenize.NAME and stmt[0].string == "case":
        # The only binding a `case` header does is its capture patterns, and
        # they reach the case suite alone: a case that does not match binds
        # nothing, so the alias is still the module below the `match`.
        if scoped is not None:
            _add_capture_names(stmt, scoped, candidates)
        return
    if stmt[0].type == tokenize.NAME and stmt[0].string == "del" and len(stmt) > 1:
        # `del b` unbinds the alias: the `b.eval(...)` below it raises
        # `NameError`, and inside a function the deletion makes `b` a local of
        # that function throughout, so a call above it raises
        # `UnboundLocalError` rather than reaching the module-level alias.
        # Only a plain name is unbound - `del holder.b` and `del cache[k]`
        # leave the alias exactly where it was.
        _add_assignment_targets(stmt[1:], rebound)
        return
    head = 1 if (stmt[0].type == tokenize.NAME and stmt[0].string == "async") else 0
    if (
        len(stmt) > head + 1
        and stmt[head].type == tokenize.NAME
        and stmt[head].string in ("def", "class")
        and stmt[head + 1].type == tokenize.NAME
    ):
        rebound.add(stmt[head + 1].string)
        if scoped is not None:
            _add_param_names(stmt, head + 2, scoped, candidates)
    # `except E as b` binds nothing that outlives the handler: with no exception
    # the name is never bound, and with one Python deletes it at the end of the
    # suite. Recording the header as an ordinary rebinding cancelled the alias
    # to the end of the file - and the header sits at the same indent as the
    # code below it, so nothing ever closed the span. Inside the suite the name
    # is the exception, which is what the scoped span says.
    handler = stmt[head].type == tokenize.NAME and stmt[head].string == "except"
    depth = 0
    for j, tok in enumerate(stmt):
        if tok.type == tokenize.OP:
            if tok.string in _OPENERS:
                depth += 1
            elif tok.string in _CLOSERS:
                depth -= 1
            elif tok.string == ":=":
                if j and stmt[j - 1].type == tokenize.NAME:
                    rebound.add(stmt[j - 1].string)
            continue
        if tok.type != tokenize.NAME:
            continue
        if tok.string == "for":
            if depth:
                continue  # a comprehension: its target is scoped to the comprehension
            k = j + 1
            while k < len(stmt) and not (stmt[k].type == tokenize.NAME and stmt[k].string == "in"):
                k += 1
            # The target binds what an ordinary assignment would: `for a, b
            # in ...` rebinds both, `for holder.b in ...` rebinds neither -
            # taking every name in between would let that no-op loop cancel
            # a live alias. Scoped to the suite, because an empty iterable
            # leaves the outer binding exactly as it was.
            _add_assignment_targets(stmt[j + 1 : k], body)
        elif tok.string == "as":  # `with ... as x`, `import x as y`, `except E as x`
            if j + 1 < len(stmt) and stmt[j + 1].type == tokenize.NAME:
                (body if handler else rebound).add(stmt[j + 1].string)
    groups: list = []
    cur: list = []
    depth = 0
    for tok in stmt:
        if tok.type == tokenize.OP:
            text = tok.string
            if text in _OPENERS:
                depth += 1
            elif text in _CLOSERS:
                depth -= 1
            elif depth == 0 and text == "=":
                groups.append(cur)  # `a = b = value` binds both a and b
                cur = []
                continue
            elif (
                depth == 0
                and len(text) > 1
                and text != ":="
                and text.endswith("=")
                and text not in _COMPARISON_OPS
            ):
                groups.append(cur)  # augmented: `b += value`
                cur = []
                break
        cur.append(tok)
    for group in groups:
        _add_assignment_targets(group, rebound)


def _comprehension_shadows(stmt: list, candidates: frozenset, offsets, local_shadows: dict) -> None:
    """Record where a comprehension target hides an alias of the same name.

    A comprehension runs in a scope of its own, so `[run(x) for run, x in cb]`
    calls an element of `cb` and not the `from builtins import exec as run`
    written above it. Skipping the target outright read that as the builtin,
    which is a false HIGH on an ordinary callback table.

    The one part NOT in that scope is the outermost iterable: it is evaluated
    where the comprehension is written and passed in, so `[y for run in
    run(marshal.loads(BLOB))]` really is the builtin and is left live. That is
    why the shadow is two spans around it rather than the whole bracket.

    Nothing is recorded for the ordinary statement: a bracket with no `for` in
    it never allocates, and the whole pass is skipped unless the statement
    mentions an alias.
    """
    for tok in stmt:
        if tok.type == tokenize.NAME and tok.string in candidates:
            break
    else:
        return
    # Per open bracket: [opener index, targets, first iterable start, its end].
    stack: list = []
    for j, tok in enumerate(stmt):
        if tok.type == tokenize.OP:
            if tok.string in _OPENERS:
                stack.append([j, None, None, None])
            elif tok.string in _CLOSERS and stack:
                open_at, targets, iter_at, iter_end = stack.pop()
                if not targets or iter_at is None or iter_at >= len(stmt):
                    continue
                if iter_end is None:
                    iter_end = j
                head = (offsets.of(*stmt[open_at].start), offsets.of(*stmt[iter_at].start))
                tail = (offsets.of(*stmt[iter_end].start), offsets.of(*stmt[j].end))
                for name in targets:
                    local_shadows.setdefault(name, []).extend((head, tail))
            continue
        if tok.type != tokenize.NAME or not stack:
            continue
        frame = stack[-1]
        if tok.string in ("for", "if", "async") and frame[2] is not None and frame[3] is None:
            frame[3] = j  # the clause after the outermost iterable is what ends it
        if tok.string == "for":
            k = j + 1
            depth = 0
            while k < len(stmt):
                nxt = stmt[k]
                if nxt.type == tokenize.OP:
                    if nxt.string in _OPENERS:
                        depth += 1
                    elif nxt.string in _CLOSERS:
                        depth -= 1
                elif not depth and nxt.type == tokenize.NAME and nxt.string == "in":
                    break
                k += 1
            if k >= len(stmt):
                continue
            names: set = set()
            _add_assignment_targets(stmt[j + 1 : k], names)
            names &= candidates
            if frame[1] is None:
                frame[1] = names
            else:
                frame[1] |= names
            if frame[2] is None:
                frame[2] = k + 1  # the outermost iterable of THIS comprehension
    return


def _lambda_shadows(stmt: list, candidates: frozenset, offsets, local_shadows: dict) -> None:
    """Record where a lambda parameter hides an alias of the same name.

    A lambda is a scope like any other function, so `(lambda run:
    run(marshal.loads(x)))(model.eval)` calls the argument and not the `from
    builtins import exec as run` written above it. Only `def` and `class`
    headers were read for parameters, which left that an unconditional false
    HIGH - and HIGH is what fails the scan.

    The span starts after the `:` that ends the parameter list, because a
    default is evaluated where the lambda is written and not in its body, and
    ends where the body does: the first comma or unmatched closing bracket at
    the depth the `lambda` sits at, or the end of the statement.
    """
    for tok in stmt:
        if tok.type == tokenize.NAME and tok.string in candidates:
            break
    else:
        return
    for j, tok in enumerate(stmt):
        if tok.type != tokenize.NAME or tok.string != "lambda":
            continue
        names, colon = _lambda_params(stmt, j, candidates)
        if not names:
            continue
        end = _lambda_body_end(stmt, colon)
        if end <= colon:
            continue
        span = (offsets.of(*stmt[colon].end), offsets.of(*stmt[end].end))
        for name in names:
            local_shadows.setdefault(name, []).append(span)


def _lambda_params(stmt: list, at: int, candidates: frozenset) -> "tuple[set, int]":
    """The parameters of the lambda at `at` that shadow an alias, and its colon.

    A parameter is a name that opens an item of the list: `*`, `**` and `/` are
    punctuation between them, and everything after a `=` is a default expression
    evaluated in the enclosing scope.

    A `lambda` written in a default owns the colons and commas that follow it
    until its own colon closes it - `lambda run=lambda x, y: x: run(BLOB)` gives
    both `x` and `y` to the inner lambda - so they pair off innermost-first.
    Abandoning the outer list at the nested `lambda` instead dropped the
    parameters already read, and the outer body was then reported as a call to
    the imported builtin.
    """
    names: set = set()
    depth = 0
    expect = True
    nested = 0
    k = at + 1
    while k < len(stmt):
        tok = stmt[k]
        if tok.type == tokenize.OP:
            if tok.string in _OPENERS:
                depth += 1
            elif tok.string in _CLOSERS:
                if not depth:
                    break
                depth -= 1
            elif not depth:
                if tok.string == ":":
                    if not nested:
                        return names, k
                    nested -= 1  # closes an inner lambda; its body starts here
                elif tok.string == ",":
                    if not nested:
                        expect = True  # the inner body ended: back in this list
                elif tok.string == "=":
                    if not nested:
                        expect = False
        elif tok.type == tokenize.NAME and not depth:
            if tok.string == "lambda":
                nested += 1
            elif expect and not nested:
                if tok.string in candidates:
                    names.add(tok.string)
                expect = False
        k += 1
    return set(), -1


def _lambda_body_end(stmt: list, colon: int) -> int:
    """The last token of the body of the lambda whose colon is at `colon`.

    A depth-0 `:` ends it too: the body of a lambda cannot hold one - a dict
    entry, a slice and an annotation are all bracketed or written before the
    `lambda` - so the next one belongs to an enclosing lambda whose parameter
    list this body sits in. Running past it let a nested `lambda run: run`
    written as a default shadow the outer body, where `run` is still the alias.
    """
    depth = 0
    end = colon
    k = colon + 1
    while k < len(stmt):
        tok = stmt[k]
        if tok.type == tokenize.OP:
            if tok.string in _OPENERS:
                depth += 1
            elif tok.string in _CLOSERS:
                if not depth:
                    break
                depth -= 1
            elif not depth and tok.string in (",", ":"):
                break
        end = k
        k += 1
    return end


def _self_assigned(stmt: list) -> "set | None":
    """The names `stmt` assigns to themselves, or None if it is not that shape.

    `b = b` cannot make `b` something other than what it already was: at module
    level it rebinds the name to its own value, and inside a function it raises
    rather than binding anything else. Yet the rebinding collector sees an
    assignment target and the binding reader sees a right-hand side that is not
    the module named outright, so between them a live alias was dropped and the
    `b.exec(...)` below went unflagged. Only the exact shape is accepted:
    `a = b = b` really does rebind `a`.
    """
    groups = _split_top(stmt, "=")
    if len(groups) < 2:
        return None
    value = _strip_parens(groups[-1])
    if len(value) != 1 or value[0].type != tokenize.NAME:
        return None
    names = set()
    for group in groups[:-1]:
        colon_at = _name_index_op(group, ":")  # annotated target: `b: Any = b`
        if colon_at is not None:
            group = group[:colon_at]
        if len(group) != 1 or group[0].type != tokenize.NAME:
            return None
        names.add(group[0].string)
    return names if names == {value[0].string} else None


def _receiver_start(
    stmt: list,
    dot_at: int,
    receivers: frozenset,
    loaders: frozenset = frozenset(),
    memo: "dict | None" = None,
    loader_modules: frozenset = frozenset(("importlib",)),
) -> "tuple":
    """Where the receiver of `stmt[dot_at]` starts, and the alias that made it one.

    Walks the primary expression left of the dot - names, subscripts, calls and
    parentheses - and accepts it when it mentions the `builtins` module by name
    or by string. That covers `(builtins).exec`, `b.exec` for an alias `b`, and
    `__import__('builtins').exec`, and rejects `model.eval` regardless of how
    much whitespace sits around the dot, because whitespace is not a token.

    A receiver that is itself a call to a module loader (`__import__(name)`,
    `import_module(name)`, `importlib.import_module(name)`) is accepted whatever
    its argument, so a computed module name does not hide the call: the module
    a loader returns is not something `model.eval()` can be, and the argument is
    exactly the part a payload can compute. The name it loads is not tracked, so
    this is a syntactic test on the receiver, not dataflow.

    Returns `(start_index, alias)`, or `(None, None)` when the receiver is not
    `builtins`. `alias` is the local name the receiver was reached through, or
    None when the module was named outright (`builtins`, `__builtins__`, or the
    string `'builtins'`) - those spellings cannot be rebound out from under the
    call, so only a real alias is a candidate for the rebinding cutoff.

    `memo`, when given, caches the walk per statement. Every `.exec` in
    `builtins.exec(x).exec(x)...` otherwise restarts the walk over the whole
    chain to its left, which is quadratic in the number of calls - a 25 KB file
    of them took 4.3 s, on archive members allowed up to 64 MiB. A walk that
    reaches a position an earlier one started from is the earlier walk from
    there on, so it stops and folds in that result instead.
    """
    k = dot_at - 1
    start = None
    found = False
    hard = False
    alias = None
    while k >= 0:
        if memo is not None:
            cached = memo.get(k)
            if cached is not None:
                seen_start, seen_found, seen_hard, seen_alias = cached
                if seen_start is not None:
                    start = seen_start
                found = found or seen_found
                hard = hard or seen_hard
                if seen_alias is not None:
                    alias = seen_alias  # the leftmost spelling is the receiver's
                break
        tok = stmt[k]
        if tok.type == tokenize.OP:
            if tok.string in _CLOSERS:
                close_at = k
                depth = 0
                while k >= 0:
                    inner = stmt[k]
                    if inner.type == tokenize.OP:
                        if inner.string in _CLOSERS:
                            depth += 1
                        elif inner.string in _OPENERS:
                            depth -= 1
                            if depth == 0:
                                break
                    elif inner.type == tokenize.NAME and inner.string in receivers:
                        found = True
                        if inner.string in _BUILTINS_NAMES:
                            hard = True
                        else:
                            alias = inner.string
                    elif (
                        inner.type == tokenize.STRING
                        and _string_body(inner.string) in _BUILTINS_NAMES
                    ):
                        found = True
                        hard = True
                    k -= 1
                if k < 0:
                    break
                if (
                    not found
                    and stmt[k].string == "("
                    and close_at + 1 < len(stmt)
                    and stmt[close_at + 1].type == tokenize.OP
                    and stmt[close_at + 1].string == "("
                ):
                    # `(__import__)(name).exec(...)`: parenthesizing the callable
                    # does not change what it returns, so the group is a loader
                    # call whenever its contents name one.
                    callee = _strip_parens(stmt[k : close_at + 1])
                    if (
                        callee
                        and callee[-1].type == tokenize.NAME
                        and _is_loader_name(
                            callee[-3].string
                            if len(callee) >= 3 and callee[-2].string == "."
                            else "",
                            callee[-1].string,
                            loaders,
                            loader_modules,
                        )
                    ):
                        found = True
                        hard = True
                start = k
                k -= 1
                continue
            if tok.string == ".":
                k -= 1
                continue
            break
        if tok.type in (tokenize.NAME, tokenize.STRING, tokenize.NUMBER):
            if tok.type == tokenize.NAME:
                if keyword.iskeyword(tok.string) and tok.string not in ("None", "True", "False"):
                    break  # `return`, `import`, ... - not part of the receiver
                if tok.string in receivers:
                    found = True
                    if tok.string in _BUILTINS_NAMES:
                        hard = True
                    else:
                        alias = tok.string
                elif (
                    k + 1 < len(stmt)
                    and stmt[k + 1].type == tokenize.OP
                    and stmt[k + 1].string == "("
                    and _is_loader_name(
                        stmt[k - 2].string
                        if (
                            k >= 2
                            and stmt[k - 1].type == tokenize.OP
                            and stmt[k - 1].string == "."
                            and stmt[k - 2].type == tokenize.NAME
                        )
                        else "",
                        tok.string,
                        loaders,
                        loader_modules,
                    )
                ):
                    found = True  # `__import__(name).exec(...)`: the receiver is the call
                    hard = True
            elif tok.type == tokenize.STRING and _string_body(tok.string) in _BUILTINS_NAMES:
                found = True
                hard = True
            start = k
            k -= 1
            if k >= 0 and stmt[k].type == tokenize.OP and stmt[k].string == ".":
                continue
            break
        break
    if memo is not None:
        memo[dot_at - 1] = (start, found, hard, alias)
    if not found:
        return None, None
    return start, (None if hard else alias)


class _Aliases:
    """The local names one file binds to `builtins`, and where each stops being one.

    `live_*` are every alias the file ever binds; `cancel` maps the ones the file
    also rebinds to where those rebindings are, and `safe_*` are the aliases
    with no rebinding at all. A call before its alias's rebinding still reaches
    the builtin - module-level code runs top to bottom - so the cutoff is
    per-call-site rather than a set difference over the whole file. Scans that
    have no file offset to compare against (`_extract_evidence` re-searching one
    line, or the regex fallback, which reports offsets but not the name behind
    them) use `safe_*` and so stay exactly as conservative as before.
    """

    __slots__ = (
        "live_receivers",
        "live_funcs",
        "cancel",
        "declared_global",
        "scoped_imports",
        "local_shadows",
        "loader_modules",
        "safe_receivers",
        "safe_funcs",
        "loaders",
    )

    def __init__(
        self,
        modules: set,
        funcs: set,
        cancel: dict,
        loader_funcs: "set | None" = None,
        declared_global: "dict | None" = None,
        loader_modules: "set | None" = None,
        scoped_imports: "dict | None" = None,
        local_shadows: "dict | None" = None,
    ):
        self.live_receivers = frozenset(_BUILTINS_NAMES | modules)
        self.live_funcs = frozenset(funcs)
        self.cancel = _index_cancellations(cancel)
        self.declared_global = declared_global or {}
        # An alias only a function-local import binds is not the module anywhere
        # else in the file: `def a(): from builtins import exec as run` leaves
        # the `run(...)` in a sibling function its own name, not the builtin.
        self.scoped_imports = _index_scopes(scoped_imports) if scoped_imports else {}
        # An alias a function assigns is that function's local over its whole
        # body: `def f(): run(BLOB); run = model` raises `UnboundLocalError`
        # rather than reaching the module-level `run`.
        self.local_shadows = _index_scopes(local_shadows) if local_shadows else {}
        self.safe_receivers = frozenset(self.live_receivers - set(cancel))
        self.safe_funcs = frozenset(self.live_funcs - set(cancel))
        # `__import__` is a builtin, so it means the loader in any file; the
        # names the file binds to a loader itself are added to it.
        self.loaders = frozenset(_DEFAULT_LOADER_FUNCS | (loader_funcs or set()))
        # `importlib` needs no import of its own to name the module, so it is a
        # default here where `import_module` is not; an alias adds to it.
        self.loader_modules = frozenset({"importlib"} | (loader_modules or set()))


def _opens_scope(stmt: list) -> "str | None":
    """`"class"`, `"def"`, or None for the scope `stmt` heads.

    Decorators are their own statements, so the header is the first token; an
    `async def` is read past the `async`. Every other compound statement - `if`,
    `for`, `with`, `try` - keeps the namespace it is written in, which is why
    only these two are worth tracking.
    """
    head = stmt[0]
    if head.type != tokenize.NAME:
        return None
    if head.string in ("class", "def"):
        return head.string
    if head.string == "async" and len(stmt) > 1 and stmt[1].string == "def":
        return "def"
    return None


# The compound statements whose header is spelled with a hard keyword. `match`
# and `case` are soft ones - a name, until the statement is shaped like a
# header - so they are recognised separately.
_BLOCK_KEYWORDS = frozenset(
    ("if", "elif", "else", "for", "while", "with", "try", "except", "finally", "def", "class")
)


def _opens_block(stmt: list) -> "tuple | None":
    """`(keyword, unconditional)` for the suite `stmt` heads, or None.

    `unconditional` marks the one block whose bindings outlive its own dedent:
    an `if True:` suite always runs, and control only reaches the statement
    below it by falling off the end of that suite, so `if True: run = cb`
    really does leave `run` bound to `cb` afterwards. Every other header is
    conditional - a loop can iterate zero times, a `try` body can raise
    partway, a `with` can have its exception swallowed by `__exit__`.

    `match` and `case` are soft keywords, so they are read as headers only when
    the statement is shaped like one: a depth-0 colon at least two tokens in.
    That is what separates `match x:` from the annotation `match: int = 5` and
    the call statement `match(x)`, either of which may name an ordinary
    variable `match`.
    """
    head = stmt[0]
    if head.type != tokenize.NAME:
        return None
    word = head.string
    if word == "async" and len(stmt) > 1:
        word = stmt[1].string
    if word in _BLOCK_KEYWORDS:
        if word != "if":
            return word, False
        return word, _is_truthy_constant(stmt[1 : _header_end_index(stmt)])
    if word in ("match", "case"):
        end = _header_end_index(stmt)
        colon = stmt[end]
        if end >= 2 and colon.type == tokenize.OP and colon.string == ":":
            return word, False
    return None


def _is_truthy_constant(toks: list) -> bool:
    """Whether `toks` is a condition that is always true.

    Only the two spellings a generated or vendored file writes to open an
    unconditional block. `True` cannot be reassigned in Python 3 - it is a
    keyword - so reading it as a constant is safe.
    """
    toks = _strip_parens(toks)
    if len(toks) != 1:
        return False
    tok = toks[0]
    return (tok.type == tokenize.NAME and tok.string == "True") or (
        tok.type == tokenize.NUMBER and tok.string == "1"
    )


def _survives_dedent(blocks: list, level: int) -> "int | None":
    """The indent a rebinding written at `level` stays live at after its dedent.

    None when it does not: the enclosing block may not have run, so the binding
    it made cannot be assumed. The chain is walked all the way out in one go,
    so a nested `if True:` inside another is re-homed once at the outermost
    indent rather than once per dedent as the block unwinds.
    """
    target = None
    for entry in reversed(blocks):
        if entry[0] >= level:
            continue
        if not entry[2]:
            break
        target = level = entry[0]
    return target


def _relevel(levels: list, col: int, entries: list) -> None:
    """Re-file `entries` under indent `col`, keeping `levels` sorted by indent."""
    for i in range(len(levels) - 1, -1, -1):
        if levels[i][0] == col:
            levels[i][1].extend(entries)
            return
        if levels[i][0] < col:
            levels.insert(i + 1, (col, list(entries)))
            return
    levels.insert(0, (col, list(entries)))


def _cancel_add(cancel: dict, opened: dict, levels: list, name: str, at: int, col: int) -> None:
    """Record that `name` stops being the module at offset `at`, indent `col`.

    A rebinding only reaches a call inside the block it is written in: `def f():
    b = model` cannot change what a module-level `b.exec(...)` below it calls,
    and neither can a sibling function or branch, which is why the record is a
    span - offset, indent, and the offset the block ends at - rather than a
    point. `opened` holds the spans whose block is still being read; one already
    open at this indent or shallower decides every call the new one would, so
    the record stays at one entry per indent level however many times a hostile
    file rebinds the name.

    `levels` holds those same spans grouped by indent, deepest last, so closing
    a block reaches only the spans that block actually ends. Every statement
    closes and only a rebinding opens, so the walk over every name ever opened
    is the one shape that must stay off the per-statement path.
    """
    stack = opened.setdefault(name, [])
    if stack and stack[-1][1] <= col:
        return
    entry = [at, col, None]
    cancel.setdefault(name, []).append(entry)
    stack.append(entry)
    # `_cancel_close` has already ended every span deeper than `col`, so the
    # deepest surviving group is at `col` or shallower and the list stays sorted.
    if levels and levels[-1][0] == col:
        levels[-1][1].append((name, entry))
    else:
        levels.append((col, [(name, entry)]))


def _cancel_rearm(
    opened: dict,
    name: str,
    at: int,
    floor: int = -1,
    outer: "list | None" = None,
) -> None:
    """End `name`'s open cancellation spans at `at`, keeping what they decided.

    A statement that puts the module back in the name - `import builtins as b`,
    `b = __import__('builtins')` - stops the earlier rebinding deciding anything
    BELOW it, and nothing above it. Dropping the record outright instead read
    `b = model` ... `b.eval(x)` ... `import builtins as b` as the builtin, a
    false HIGH on a file that re-imports an alias further down.

    The entries stay in their `levels` group, so the block that opened them
    closes them again later; `_cancel_close` keeps the earlier end because a
    span that has one is already ended.

    `outer` collects the `(name, indent)` of every span ended here that was
    opened at `floor` or shallower - a scope the re-arm is written inside cannot
    bind for, so the caller reopens those where that scope ends.
    """
    for entry in opened.pop(name, ()):
        if entry[2] is None:
            entry[2] = at
        if outer is not None and entry[1] <= floor:
            outer.append((name, entry[1]))


def _rearm_here(opened: dict, scopes: list, name: str, at: int) -> None:
    """Re-arm `name` at `at`, restoring an enclosing scope's cancellation after.

    A binding written in a `def` or `class` body is that scope's local or its
    attribute, so it cannot put the module back in a name the enclosing scope
    rebound: `b = model` then `def f(): b = __import__('builtins')` leaves the
    module-level `b` the model, and reading the `b.exec(...)` below `f` as the
    builtin was a false HIGH. The re-arm still governs the body it is written in
    - a call under it there really does reach the module - so the outer span is
    ended here and reopened at the offset the body ends, which `_close_scope`
    knows and this statement does not.

    `global b` and `nonlocal b` are the two statements that stop the binding
    being local, so a name under either re-arms file-wide exactly as before.
    """
    scope = scopes[-1] if scopes else None
    if scope is None or name in (scope[3] or ()) or name in (scope[7] or ()):
        _cancel_rearm(opened, name, at)
        return
    outer: list = []
    _cancel_rearm(opened, name, at, scope[0], outer)
    if outer:
        if scope[8] is None:
            scope[8] = []
        scope[8].extend(outer)


def _cancel_close(
    opened: dict,
    levels: list,
    at: int,
    col: int,
    blocks: "list | None" = None,
) -> None:
    """End the spans whose block a statement at `at`, indent `col`, has left.

    Only the groups deeper than `col` are touched and each one ends once, so a
    file holding N aliases rebound in a block and then N ordinary statements
    costs N closes in total rather than N per statement. Walking `opened`
    instead kept every name that had ever been rebound in the loop, empty stack
    or not, which on a member allowed up to 64 MiB is enough to stall the scan.

    `blocks` are the compound headers the closing statement sits under, most
    recent last. A span written under an `if True:` is not ended at the dedent -
    that suite always runs, so its rebinding is still the live one below the
    block; it is re-filed at the enclosing indent and goes on deciding calls
    from there.
    """
    kept: dict = {}
    while levels and levels[-1][0] > col:
        level, group = levels.pop()
        out = _survives_dedent(blocks, level) if blocks else None
        for name, entry in group:
            # A span a re-arm already ended keeps that end: the block closing
            # later must not stretch it back over the calls in between.
            if out is not None and out <= col and entry[2] is None:
                entry[1] = out
                kept.setdefault(out, []).append((name, entry))
                continue
            if entry[2] is None:
                entry[2] = at
            stack = opened.get(name)
            if stack:
                # The deepest span recorded for the name is the one in this
                # group; a name re-armed meanwhile has no stack left to pop.
                stack.pop()
                if not stack:
                    del opened[name]
    for out, entries in sorted(kept.items()):
        _relevel(levels, out, entries)


def _close_scope(
    scope: list,
    declared_global: dict,
    end: int,
    local_imports: "dict | None" = None,
    file_wide: "set | None" = None,
    local_shadows: "dict | None" = None,
    cancel: "dict | None" = None,
) -> None:
    """Record what a closing `def` or `class` body bound, as offset spans.

    The scope is what a declaration governs, not the statement: `global b`
    applies to the whole function, including the lines written above it. So the
    span runs from the header to wherever the body ends, and an alias imported
    in the body is visible over exactly the same range - the function and the
    scopes nested in it, which are written inside it.

    A name the body assigns is local to the whole function for the same reason,
    and Python decides that by scanning the block rather than by running it, so
    a call written *above* the assignment raises `UnboundLocalError` instead of
    reaching the module-level alias. That span starts after the header, because
    a default argument is evaluated in the enclosing scope where the alias does
    still resolve.
    """
    names = scope[3]
    imported = scope[4] if len(scope) > 4 else None
    shadowed = scope[6] if len(scope) > 6 else None
    rearmed = scope[8] if len(scope) > 8 else None
    if rearmed and cancel is not None:
        # The body re-armed an alias the enclosing scope had rebound, which it
        # can only do for itself. Past this offset the enclosing rebinding is
        # the live one again, at the indent it was written at - so a call
        # shallower than that is left alone exactly as the original span left it.
        for name, col in rearmed:
            cancel.setdefault(name, []).append([end, col, None])
    if shadowed and local_shadows is not None:
        for name in shadowed:
            local_shadows.setdefault(name, []).append((scope[5], end))
    if imported and local_imports is not None:
        for name in imported:
            if names and name in names:
                # `global run` then `from builtins import exec as run` binds the
                # module-level name, which every scope in the file can see.
                if file_wide is not None:
                    file_wide.add(name)
            else:
                local_imports.setdefault(name, []).append((scope[2], end))
    if not names:
        return
    for name in names:
        declared_global.setdefault(name, []).append((scope[2], end))


def _shadow_scope(scope: list, names) -> None:
    """Record `names` as locals of `scope`, minus the ones it declared away.

    `global b` and `nonlocal b` are the two statements that stop an assignment
    creating a local, so a name under either goes on resolving outward and its
    calls are not shadowed.
    """
    shadow = set(names).difference(scope[3] or (), scope[7] or ())
    if not shadow:
        return
    if scope[6] is None:
        scope[6] = set()
    scope[6] |= shadow


def _index_scopes(by_name: dict) -> dict:
    """`by_name`, with each span list turned into what `_in_scope` bisects.

    A member holding N sibling functions that each import and call the same
    alias leaves N spans on the name and asks N containment questions, and
    walking the list for each was quadratic - 16,000 such functions fit in
    under 1 MiB and took seconds, on members allowed up to 64 MiB.

    The spans nest, so the last one starting at or before an offset is not
    necessarily the one that contains it: an outer scope still can. Carrying the
    running maximum of the ends alongside the starts answers that in one
    comparison - if no span starting at or before the offset reaches past it,
    none contains it.
    """
    indexed: dict = {}
    for name, spans in by_name.items():
        starts: list = []
        reach: list = []
        furthest = 0
        for at, end in sorted(spans):
            if end > furthest:
                furthest = end
            starts.append(at)
            reach.append(furthest)
        indexed[name] = (starts, reach)
    return indexed


def _in_scope(indexed: tuple, start: int) -> bool:
    """Whether offset `start` sits inside one of the spans `_index_scopes` built."""
    starts, reach = indexed
    i = bisect.bisect_right(starts, start)
    return bool(i) and reach[i - 1] > start


def _index_globals(spans: list) -> tuple:
    """`spans`, sorted by start, with each one's enclosing span recorded.

    Scope ranges nest, so the span a bisection lands on either contains the
    offset or is a sibling written before it; from there the answer, if there is
    one, is up the parent chain and nowhere else. That makes a lookup cost the
    nesting depth instead of the number of declarations - N sibling functions
    each declaring `global b` and calling `b.eval(...)` re-read all N spans per
    call otherwise, which is quadratic on source a hostile archive member
    chooses: 16,000 such functions fit in 645 KiB and took over 15 seconds, on
    members allowed up to 64 MiB.
    """
    starts: list = []
    ends: list = []
    parents: list = []
    stack: list = []
    for at, end in spans:
        while stack and ends[stack[-1]] <= at:
            stack.pop()
        parents.append(stack[-1] if stack else -1)
        starts.append(at)
        ends.append(end)
        stack.append(len(starts) - 1)
    return starts, ends, parents


def _declares_global(indexed: tuple, start: int) -> "tuple | None":
    """The innermost scope containing `start` that declared the name global.

    The span itself, not a flag, because the declaring scope is also the one
    place whose rebindings still count: `global b` makes `b = model` write the
    module-level name, so that assignment silences the calls below it exactly
    as a module-level one would. Only an *enclosing* function's local is the
    thing a declaration outranks. Innermost wins, since a nested `global` is
    the declaration that governs the call.
    """
    starts, ends, parents = indexed
    i = bisect.bisect_right(starts, start) - 1
    while i >= 0:
        if ends[i] > start:
            return starts[i], ends[i]
        i = parents[i]
    return None


def _index_cancellations(cancel: dict) -> dict:
    """`cancel`, regrouped so a call site tests one span per indent, not all of them.

    A file with one rebinding inside each of N sibling functions leaves N closed
    spans on the name, and a linear walk re-read every one of them at every call
    below - quadratic in N, on source a hostile archive member chooses. At a
    fixed indent the spans for one name cannot overlap (`_cancel_add` refuses to
    open a second while one is open), so within an indent group they are
    disjoint and already in source order: the only span that can contain an
    offset is the last one starting at or before it, and a bisection finds it.
    Indent groups are bounded by nesting depth, not by file size.
    """
    indexed: dict = {}
    for name, frontier in cancel.items():
        groups: dict = {}
        for entry in frontier:
            ats, entries = groups.setdefault(entry[1], ([], []))
            ats.append(entry[0])
            entries.append(entry)
        indexed[name] = sorted(groups.items())
    return indexed


def _cancelled_at(cancel: dict, name: str, at: int, col: int) -> bool:
    """Whether `name` is already cancelled at `at`, mid-collection.

    The collecting pass builds `cancel` in source order, so everything written
    above this statement is in it; the one name is indexed on its own because a
    copy of an alias is rare enough that the allocation never adds up.
    """
    frontier = cancel.get(name)
    if not frontier:
        return False
    return _is_cancelled(_index_cancellations({name: frontier})[name], at, col)


def _is_cancelled(
    indexed: list,
    start: int,
    col: int,
    global_scope: "tuple | None" = None,
) -> bool:
    for at_col, (ats, entries) in indexed:
        if at_col > col:
            break  # sorted by indent, so nothing deeper can reach this call
        i = bisect.bisect_right(ats, start)
        if not i:
            continue
        entry = entries[i - 1]
        if (
            global_scope is not None
            and at_col > 0
            and not (global_scope[0] <= entry[0] < global_scope[1])
        ):
            # The call is in a scope that declared the name `global`, so it
            # resolves at module level whatever any *enclosing* function did to
            # its own local of that name. A rebinding written inside the
            # declaring scope is not that: `global b` makes it write the
            # module-level name, and the call below it reads the new value.
            continue
        end = entry[2]
        if end is None or start < end:
            return True
    return False


def _fstring_code(literal: str, nesting: int = _MAX_FSTRING_NESTING) -> "str | None":
    """`literal` with everything that is not interpolated expression source blanked.

    Only the replacement fields of an f-string are code; the literal text around
    them is not, and neither is a string nested inside a field. Blanking both -
    length-preserved, so offsets still point into the file - is what keeps
    `f'{v.replace("exec(", "")}'` from reading as a call while
    `f'{exec(payload)}'` still does. A nested f-string is masked in turn, so an
    inner interpolation is not lost with the literal that carries it.

    `nesting` bounds the recursion at the grammar's own ceiling rather than at
    an arbitrary depth: a pre-3.12 f-string cannot reuse the quote style that
    delimits it, so one literal can nest at most as deep as there are styles.
    Stopping any shallower blanks a real call - `f'''{f\"\"\"{f'{f\"{exec(p)}\"}'}\"\"\"}'''`
    tokenizes as one STRING and compiles fine on 3.11.
    """
    i = 0
    while i < len(literal) and literal[i] not in "\"'":
        i += 1
    if i >= len(literal):
        return None
    for quote in _FSTRING_QUOTES:
        if literal.startswith(quote, i) and literal.endswith(quote):
            if len(literal) >= i + 2 * len(quote):
                break
    else:
        return None
    start = i + len(quote)
    end = len(literal) - len(quote)
    out = [" "] * len(literal)
    depth = 0
    j = start
    while j < end:
        ch = literal[j]
        if depth == 0:  # literal text: `{{` and `}}` are escaped braces
            if ch == "{" and not literal.startswith("{{", j):
                depth = 1
                j += 1
            else:
                j += 2 if ch in "{}" else 1
            continue
        if ch in "\"'":
            # A string inside the expression. Its contents are data, not code -
            # unless it is itself an f-string, whose fields are code again.
            at = j
            while at > start and literal[at - 1].isalpha() and j - at < 3:
                at -= 1
            prefix = literal[at:j].lower()
            closer = ch * 3 if literal.startswith(ch * 3, j) else ch
            k = literal.find(closer, j + len(closer))
            if k == -1 or k >= end:
                break  # unterminated: nothing after it can be trusted as code
            k += len(closer)
            if nesting and "f" in prefix and "b" not in prefix:
                inner = _fstring_code(literal[at:k], nesting - 1)
                if inner is not None:
                    out[at:k] = inner
            j = k
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if not depth:
                j += 1
                continue
        out[j] = ch
        j += 1
    return "".join(out)


def _fstring_spans(
    tok,
    aliases: _Aliases,
    offsets: _Offsets,
    out: list,
    positional: bool = False,
    col: int = 0,
) -> None:
    """Scan the expressions inside an f-string the tokenizer left opaque.

    Below Python 3.12 the whole literal is one STRING token, so the statement
    scan sees no call inside `f'{exec(marshal.loads(BLOB))}'` even though the
    interpolation runs at import. Adjudicate the expression source of that one
    token with the regex fallback and shift the offsets back into the file, so
    evidence still points at the real line. The fallback refuses a name preceded
    by a dot, so `f'{model.eval()}'` stays clean - the false positive this rule
    exists to remove is not reintroduced inside an f-string.

    The literal sits at a known offset, so it gets the same per-call rebinding
    cutoff as the statement scan rather than the file-wide `safe_*` sets: a
    trailing `run = model` must not erase an `f'{run(BLOB)}'` written above it.
    """
    text = tok.string
    if "{" not in text:
        return  # no interpolation: an f-string without a replacement field
    if positional:
        receivers, funcs, cancel = aliases.live_receivers, aliases.live_funcs, aliases.cancel
    else:
        receivers, funcs, cancel = aliases.safe_receivers, aliases.safe_funcs, None
    if "exec" not in text and "eval" not in text and not _mentions_name(text, funcs):
        return
    code = _fstring_code(text)
    if code is None:
        return
    base = offsets.of(*tok.start)
    live = None
    scoped = aliases.scoped_imports if positional else None
    shadows = aliases.local_shadows if positional else None
    if cancel or scoped or shadows:

        def live(name: str, at: int) -> bool:
            start = base + at
            if scoped:
                where = scoped.get(name)
                if where is not None and not _in_scope(where, start):
                    return False  # imported in some other function
            spans = aliases.declared_global.get(name)
            declaring = _declares_global(spans, start) if spans else None
            if shadows and declaring is None:
                body = shadows.get(name)
                if body is not None and _in_scope(body, start):
                    return False  # the enclosing function assigns it: a local
            frontier = cancel.get(name) if cancel else None
            if frontier is None:
                return True
            return not _is_cancelled(frontier, start, col, declaring)

    for span in _regex_spans(code, receivers, funcs, live, aliases.loaders, aliases.loader_modules):
        out.append(_Span(base + span.start(), base + span.end()))


def _alias_live(
    alias: str, start: int, col: int, aliases: _Aliases, cancel: "dict | None", positional: bool
) -> bool:
    """Whether `alias` still names the builtin at offset `start`, indent `col`.

    The three ways a file takes the name back, in the order they are cheapest to
    test: an import that only one function ran, an assignment that makes the
    name that function's local over its whole body, and a rebinding written
    above this offset. `global` opts a scope back out of the middle one, which
    is why the declaration is read before it and handed to `_is_cancelled`.
    """
    if positional and aliases.scoped_imports:
        # Bound by an import written inside one function, so a use anywhere else
        # in the file resolves some other `run`.
        where = aliases.scoped_imports.get(alias)
        if where is not None and not _in_scope(where, start):
            return False
    declaring = None
    if cancel or (positional and aliases.local_shadows):
        spans = aliases.declared_global.get(alias)
        declaring = _declares_global(spans, start) if spans else None
    if positional and aliases.local_shadows and declaring is None:
        # The enclosing function assigns the alias, which makes it that
        # function's local everywhere in the body - so a use written above the
        # assignment raises `UnboundLocalError` instead of reaching the
        # module-level name. A `global` declaration in that scope opts back out.
        body = aliases.local_shadows.get(alias)
        if body is not None and _in_scope(body, start):
            return False
    if cancel:
        # `import builtins as m` ... `m = load_model()` ... `m.eval`: past a
        # rebinding that reaches here the name is the model, not the module. A
        # rebinding indented deeper does not reach it, so a local `m = model` in
        # some function above cannot silence a module-level use.
        frontier = cancel.get(alias)
        if frontier is not None and _is_cancelled(frontier, start, col, declaring):
            return False
    return True


def _names_builtin(
    value: list,
    receivers: frozenset,
    funcs: frozenset,
    aliases: _Aliases,
    offsets: _Offsets,
    col: int,
    cancel: "dict | None",
    positional: bool,
) -> bool:
    """Whether `value` is the exec/eval builtin itself rather than some object.

    The three spellings that hand the function over: the bare builtin, a file
    alias of it (`from builtins import eval as run`), and an attribute of the
    module (`builtins.eval`). `model.eval` is a method of an object and is not
    one, which is what keeps `holder.eval = model.eval` off the report.

    An alias is only the builtin while it still is one, so the same liveness the
    call site applies decides the right-hand side too: `run = safe` then
    `holder.exec = run` parks the safe local, and `b = model` then
    `holder.exec = b.exec` parks a method of the model. Reading either as the
    builtin was a false HIGH next to any marshal or dynamic-import marker. The
    bare builtin is not an alias and has no liveness to test - a file that
    shadows `eval` and then hands `eval` over is read as the builtin here
    exactly as it is at a call site.
    """
    if len(value) == 1 and value[0].type == tokenize.NAME:
        name = value[0].string
        if name in _EXEC_NAMES:
            return True
        if name not in funcs:
            return False
        return _alias_live(name, offsets.of(*value[0].start), col, aliases, cancel, positional)
    if (
        len(value) >= 3
        and value[-1].type == tokenize.NAME
        and value[-1].string in _EXEC_NAMES
        and value[-2].type == tokenize.OP
        and value[-2].string == "."
    ):
        at, alias = _receiver_start(
            value, len(value) - 2, receivers, aliases.loaders, None, aliases.loader_modules
        )
        if at is None:
            return False
        if alias is None:
            return True  # `builtins.exec`: a spelling no rebinding can take away
        return _alias_live(alias, offsets.of(*value[at].start), col, aliases, cancel, positional)
    return False


def _unpacked_value(stmt: list, at: int) -> "list | None":
    """The value an unpacking assignment binds to the target holding `stmt[at]`.

    Read positionally, so `holder.eval, spare = eval, None` is the same parked
    builtin as `holder.eval = eval`. Only a flat target list is read: a starred
    target shifts every position after it, and a single right-hand side
    (`holder.eval, spare = pair`) names no element at all, so both are left
    alone rather than guessed at.
    """
    eq = _name_index_op(stmt, "=")
    if eq is None or eq < at:
        return None  # the name is on the value side, or there is no assignment
    targets = _split_top(stmt[:eq])
    values = _split_top(stmt[eq + 1 :])
    if len(targets) < 2 or len(targets) != len(values):
        return None
    pos = depth = 0
    for tok in stmt[:at]:
        if tok.type != tokenize.OP:
            continue
        if tok.string in _OPENERS:
            depth += 1
        elif tok.string in _CLOSERS:
            depth -= 1
        elif depth == 0 and tok.string == ",":
            pos += 1
    for target in targets:
        if target and target[0].type == tokenize.OP and target[0].string in ("*", "**"):
            return None
    return _strip_parens(values[pos]) or None


def _statement_spans(
    stmt: list, aliases: _Aliases, offsets: _Offsets, out: list, positional: bool
) -> None:
    if positional:
        receivers, funcs, cancel = aliases.live_receivers, aliases.live_funcs, aliases.cancel
    else:
        receivers, funcs, cancel = aliases.safe_receivers, aliases.safe_funcs, None
    last = len(stmt) - 1
    walked: dict = {}
    col = stmt[0].start[1]
    # `_with_suite_tail` yields a one-line suite twice: once glued to its header
    # and once on its own. The glued copy carries the header's indent, which is
    # shallower than the suite spans a parameter or a `case` capture is recorded
    # at, so `def f(run): return run(BLOB)` was adjudicated as if it sat outside
    # the body the parameter shadows. Past the header colon the tokens are the
    # suite's, so that is the indent they are read at.
    header = last + 1
    if stmt[0].type == tokenize.NAME and stmt[0].string in _SUITE_HEADS:
        header = _header_end_index(stmt)
    for j, tok in enumerate(stmt):
        at_col = col if j <= header else col + 1
        if tok.type == tokenize.STRING:
            # A prefix is the only thing that can make this an f-string, so the
            # quote test rejects every ordinary literal for one character.
            if _FSTRING_OPAQUE and tok.string[:1] not in "\"'" and _is_fstring(tok.string):
                _fstring_spans(tok, aliases, offsets, out, positional, at_col)
            continue
        if tok.type != tokenize.NAME or j == last:
            continue
        direct = tok.string in _EXEC_NAMES
        if not direct and tok.string not in funcs:
            continue
        nxt = stmt[j + 1]
        prev = stmt[j - 1] if j else None
        attribute = prev is not None and prev.type == tokenize.OP and prev.string == "."
        if nxt.type != tokenize.OP or nxt.string != "(":
            if direct and attribute and nxt.type == tokenize.OP and nxt.string in ("=", ":", ","):
                if nxt.string == ":":
                    # `holder.eval: object = eval` parks the builtin exactly as
                    # the unannotated spelling does - the annotation is never
                    # evaluated into the attribute. Without a `=` after it there
                    # is no value at all (`holder.eval: object` binds nothing),
                    # and the colon of a dict entry or a slice is inside
                    # brackets, so it never reaches a depth-0 `=` either.
                    if _name_index_op(stmt[j + 1 :], "=") is None:
                        continue
                # `holder.eval = eval` parks the builtin on an attribute, and
                # the `holder.eval(payload)` below it is the same three tokens
                # as `model.eval()` - the call site cannot tell them apart, and
                # reading it as the builtin is the false positive this rule
                # exists to remove. The assignment is the one place the builtin
                # is named outright, so that is where the route is recorded: a
                # file that hands `exec` or `eval` to an attribute has used it,
                # whatever it does with it afterwards.
                #
                # `holder.eval, spare = eval, None` parks it exactly the same
                # way, and the target list is the only difference: read the
                # element the attribute is unpacked from, and fall back to the
                # whole right-hand side when there is only one target.
                value = _unpacked_value(stmt, j)
                if value is None:
                    if nxt.string == ",":
                        continue  # a tuple this pass cannot line up: not a park
                    value = _strip_parens(_split_top(stmt, "=")[-1])
                if value and _names_builtin(
                    value, receivers, funcs, aliases, offsets, at_col, cancel, positional
                ):
                    out.append(_Span(offsets.of(*tok.start), offsets.of(*value[-1].end)))
            continue
        alias = None
        if attribute:
            # An attribute. Only the builtins module reaches the builtin this
            # way; `run` aliases the function, so `obj.run(...)` is not it.
            if not direct:
                continue
            at, alias = _receiver_start(
                stmt, j - 1, receivers, aliases.loaders, walked, aliases.loader_modules
            )
            if at is None:
                continue
        elif prev is not None and prev.type == tokenize.NAME and prev.string in ("def", "class"):
            # `def eval(self, x):` declares a method; the parenthesis is its
            # parameter list, not a call. Only the token stream separates this
            # from `eval(x)` - to a text scan they are the same three characters
            # followed by an open paren.
            continue
        else:
            at = j
            if not direct:
                alias = tok.string
        start = offsets.of(*stmt[at].start)
        # A spelling that names the module outright reports no alias, and nothing
        # a file writes can take it away, so only a real alias is tested.
        if alias is not None and not _alias_live(alias, start, at_col, aliases, cancel, positional):
            continue
        out.append(_Span(start, offsets.of(*nxt.end)))


# Fallback, for text the tokenizer will not accept. Deliberately close to the
# pre-token behavior: on unparseable source detection matters more than the
# false positive, and the alternatives are checked in Python against a set, so
# the cost does not grow with the number of aliases.
# Text rendering of the rule, for anything that reports or hashes it rather
# than runs it; the matcher itself is the token pass above, and an alias
# receiver has no fixed spelling to put here.
_EXEC_EVAL_PATTERN_TEXT = (
    r"(?<![\w.])(?:__builtins__|builtins)\s*\)*\s*\.\s*(?:exec|eval)\s*\("
    r"|(?<![\w.])(?:exec|eval)\s*\("
)

RE_FALLBACK_STR_RECEIVER = re.compile(
    r"""(['"])(?:__builtins__|builtins)\1\s*\)\s*\.\s*(?:exec|eval)\s*\("""
)
# `__import__(name).exec(`: a loader call is the receiver whatever it is handed.
# The callee is captured and looked up rather than spelled out, so a file that
# renames the loader - `from builtins import __import__ as load` - is read the
# same way. Membership beats an alternation per alias, which would cost the
# alias count per character of the file.
RE_FALLBACK_BARE = re.compile(r"(?<![\w.])(exec|eval)\s*\(")
# `import os, builtins as b` binds `b` too, so take the whole import list and
# look for the aliased item anywhere in it.
RE_IMPORT_LIST = re.compile(r"(?<![\w.])import[ \t]+([^\n;]*)")

RE_BUILTINS_FUNC_ALIAS = re.compile(r"(?<![\w.])from\s+builtins\s+import\s+([^\n]*)")
# The same alias without an import statement: `b = __import__('builtins')`. The
# token pass decides this properly; this is only the fallback for source that
# will not tokenize, so it stays on one line and does not chase aliases of
# `importlib` - what it misses, the token pass already has.

# Any name this file assigns to. One pass for the whole file, then a set
# difference: an alias that is also an assignment target is not the builtin at
# the call site, and per-name searching would reintroduce a quadratic cost.
# Every pattern that captures an identifier, as a template over the character
# class a name is made of. `%(w)s` is `\w` for the ASCII spellings the module
# names below keep, and `\w` widened with a file's own combining marks when it
# holds any - so `e` + U+0301 is captured whole and `_ident` can normalize it to
# the `é` the alias was declared as, instead of being read as `e`.
_IDENT_PATTERN_SOURCES = (
    ("word", r"%(w)s+", 0),
    ("qualified", r"(?<![\w.])(%(w)s+)\s*\)*\s*\.\s*(?:exec|eval)\s*\(", 0),
    # `__import__(name).exec(`: a loader call is the receiver whatever it is
    # handed. The callee is captured and looked up rather than spelled out, so a
    # file that renames the loader - `from builtins import __import__ as load` -
    # is read the same way. Membership beats an alternation per alias, which
    # would cost the alias count per character of the file.
    (
        "loader_receiver",
        r"(?<![\w.])(%(w)s+)(?:\s*\.\s*(%(w)s+))?\s*\([^()]*\)\s*\.\s*(?:exec|eval)\s*\(",
        0,
    ),
    ("call", r"(?<![\w.])(%(w)s+)\s*\(", 0),
    ("module_alias_item", r"(?<![\w.])builtins[ \t]+as[ \t]+(%(w)s+)", 0),
    # The same alias without an import statement: `b = __import__('builtins')`.
    # The token pass decides this properly; this is only the fallback for source
    # that will not tokenize, so it stays on one line and does not chase aliases
    # of `importlib` - what it misses, the token pass already has.
    (
        "assign_builtins",
        r"^[ \t]*(%(w)s+)[ \t]*(?::[^=\n]+)?=[ \t]*\(*[ \t]*"
        r"(?:(?:__import__|(?:importlib[ \t]*\.[ \t]*)?import_module)[ \t]*\([ \t]*"
        r"""(['"])builtins\2|__builtins__|builtins)[ \t]*\)*[ \t]*$""",
        re.M,
    ),
    ("func_alias_item", r"(?<![\w.])(?:exec|eval)(?:\s+as\s+(%(w)s+))?", 0),
    # Any name this file assigns to. One pass for the whole file, then a set
    # difference: an alias that is also an assignment target is not the builtin
    # at the call site, and per-name searching would reintroduce a quadratic
    # cost.
    (
        "assigned_name",
        r"^[ \t]*(%(w)s+)[ \t]*(?::[^=\n]+)?=(?!=)"
        r"|(?<![\w.])for\s+(%(w)s+)\s+in\s"
        r"|(?<![\w.])(?:def|class)\s+(%(w)s+)"
        r"|(?<![\w.])as\s+(%(w)s+)\s*:",
        re.M,
    ),
)


class _Idents:
    """One compiled family of the templates above, for a given mark class."""

    __slots__ = tuple(name for name, _src, _flags in _IDENT_PATTERN_SOURCES)

    def __init__(self, marks: str):
        word = "[\\w%s]" % marks if marks else r"\w"
        for name, src, flags in _IDENT_PATTERN_SOURCES:
            setattr(self, name, re.compile(src % {"w": word}, flags))


# Cached, because a file is scanned line by line as well as whole and every one
# of those asks for the same family. The bound is what stops a hostile archive
# recompiling it per member; the ASCII family is the one nearly every file gets.
_ident_patterns = functools.lru_cache(maxsize = 8)(_Idents)
_ASCII_IDENTS = _ident_patterns("")
RE_FALLBACK_QUALIFIED = _ASCII_IDENTS.qualified
RE_FALLBACK_LOADER_RECEIVER = _ASCII_IDENTS.loader_receiver
RE_FALLBACK_CALL = _ASCII_IDENTS.call
RE_MODULE_ALIAS_ITEM = _ASCII_IDENTS.module_alias_item
RE_ASSIGN_BUILTINS = _ASCII_IDENTS.assign_builtins
RE_FUNC_ALIAS_ITEM = _ASCII_IDENTS.func_alias_item
RE_ASSIGNED_NAME = _ASCII_IDENTS.assigned_name


def _preceded_by_dot(text: str, at: int) -> bool:
    """Whether the name starting at `at` is an attribute, whitespace and all."""
    j = at - 1
    while j >= 0 and (text[j].isspace() or text[j] == "\\"):
        j -= 1
    return j >= 0 and text[j] == "."


def _regex_bindings(text: str) -> "tuple[set, set]":
    pat = _ident_patterns(_mark_class(text))
    modules: set = set()
    funcs: set = set()
    for m in RE_IMPORT_LIST.finditer(text):
        for item in pat.module_alias_item.finditer(m.group(1)):
            modules.add(_ident(item.group(1)))
    for m in RE_BUILTINS_FUNC_ALIAS.finditer(text):
        for item in pat.func_alias_item.finditer(m.group(1)):
            funcs.add(_ident(item.group(1) or item.group(0)))
    for m in pat.assign_builtins.finditer(text):
        modules.add(_ident(m.group(1)))
    return modules, funcs


def _regex_rebindings(text: str) -> set:
    """Names assigned somewhere in `text`, minus the assignments that bind the
    module itself. Both patterns anchor at the start of a line, so an alias
    assignment is recognised by the offset it matched at rather than by
    re-parsing it - without that, `b = __import__('builtins')` would cancel the
    very alias it creates.
    """
    pat = _ident_patterns(_mark_class(text))
    aliased = {m.start() for m in pat.assign_builtins.finditer(text)}
    return {
        _ident(g)
        for m in pat.assigned_name.finditer(text)
        if m.start() not in aliased
        for g in m.groups()
        if g
    }


def _regex_spans(
    text: str,
    receivers: frozenset,
    funcs: frozenset,
    live = None,
    loaders: frozenset = _DEFAULT_LOADER_FUNCS,
    loader_modules: frozenset = frozenset(("importlib",)),
) -> list:
    """Spans of every call this text reaches the builtin through.

    `live(name, at)`, when given, decides whether the alias `name` is still the
    builtin at offset `at`; without it every alias in `receivers` / `funcs` is
    taken as live, which is what the unparseable-source fallback wants.

    `loaders` is the file's local names for the module loaders. It has to be the
    file's rather than the defaults, because this is the only pass that reads a
    pre-3.12 f-string - one opaque STRING token the token walk skips - and
    `f"{load(name).exec(...)}"` runs the builtin exactly as `__import__` does.
    """
    pat = _ident_patterns(_mark_class(text))
    out = []
    for m in pat.qualified.finditer(text):
        name = _ident(m.group(1))
        if name in receivers and (live is None or live(name, m.start())):
            out.append(_Span(m.start(), m.end()))
    for m in RE_FALLBACK_STR_RECEIVER.finditer(text):
        out.append(_Span(m.start(), m.end()))
    for m in pat.loader_receiver.finditer(text):
        # `importlib.import_module(n)` names the loader in the second group;
        # a bare `load(n)` names it in the first.
        owner = _ident(m.group(1)) if m.group(2) else ""
        if _is_loader_name(owner, _ident(m.group(2) or m.group(1)), loaders, loader_modules):
            out.append(_Span(m.start(), m.end()))
    for m in RE_FALLBACK_BARE.finditer(text):
        if not _preceded_by_dot(text, m.start()):
            out.append(_Span(m.start(), m.end()))
    if funcs:
        # Only reachable through `from builtins import exec as ...`, so the one
        # scan over every call site in the file is confined to those files.
        for m in pat.call.finditer(text):
            name = _ident(m.group(1))
            if (
                name in funcs
                and not _preceded_by_dot(text, m.start())
                and (live is None or live(name, m.start()))
            ):
                out.append(_Span(m.start(), m.end()))
    out.sort(key = lambda s: s.start())
    return out


class _ExecEvalMatcher:
    """`re.Pattern`-shaped matcher bound to one file's `builtins` aliases.

    `_extract_evidence` re-searches line by line, and a line holding `b.exec(x)`
    says nothing on its own about whether `b` is the module. Binding the aliases
    to the whole file up front lets that per-line pass see them, so an
    alias-only payload gets its line recorded as evidence instead of an empty
    string. Per-line results are memoized: the answer for a line depends only on
    its text and the bound aliases, so a file of repeated lines costs one
    tokenize per distinct line.
    """

    __slots__ = (
        "aliases",
        "receivers",
        "funcs",
        "pattern",
        "bound",
        "_memo",
        "_whole",
        "_lines",
    )

    _MEMO_CAP = 4096

    def __init__(
        self,
        modules: set,
        funcs: set,
        cancel: "dict | None" = None,
        bound = None,
        loader_funcs: "set | None" = None,
        declared_global: "dict | None" = None,
        loader_modules: "set | None" = None,
        scoped_imports: "dict | None" = None,
        local_shadows: "dict | None" = None,
    ):
        self.aliases = _Aliases(
            modules,
            funcs,
            cancel or {},
            loader_funcs,
            declared_global,
            loader_modules,
            scoped_imports,
            local_shadows,
        )
        self.receivers = self.aliases.live_receivers
        self.funcs = self.aliases.live_funcs
        self.pattern = _EXEC_EVAL_PATTERN_TEXT
        # The text the aliases were derived from. Offsets only mean anything
        # against it, so it is what tells a whole-file search apart from
        # `_extract_evidence` re-searching one of its lines.
        self.bound = bound
        self._memo: dict = {}
        # The whole-file result, keyed by identity: check_py_file searches the
        # member, then _extract_evidence iterates the same string.
        self._whole: "tuple[str, list] | None" = None
        self._lines: "tuple[str, set] | None" = None

    def _spans(
        self,
        text: str,
        positional: bool = True,
    ) -> list:
        cached = self._whole
        if cached is not None and cached[0] is text:
            return cached[1]
        spans = self._scan(text, positional)
        if text is self.bound or "\n" in text:
            self._whole = (text, spans)
        return spans

    def _scan(
        self,
        text: str,
        positional: bool = True,
    ) -> list:
        # Neither route to the builtin can appear without one of these words,
        # and this test is a memchr - it keeps whole-archive scanning off the
        # tokenizer for the files that have nothing to adjudicate. A function
        # alias (`from builtins import exec as run`) spells neither at the call
        # site, so those names have to be part of the test or `run(payload)`
        # never reaches the tokenizer at all.
        if "exec" not in text and "eval" not in text and not _mentions_name(text, self.funcs):
            return []
        failed: list = []
        offsets = _Offsets(text)
        out: list = []
        for stmt in _statements(text, failed):
            _statement_spans(stmt, self.aliases, offsets, out, positional)
        if failed:
            # Tokenizing stopped early. Union with the regex so nothing the old
            # pass caught is lost, then dedupe on start offset. The regex
            # reports offsets but not the alias behind them, so it uses the
            # aliases with no rebinding at all rather than the per-call cutoff.
            seen = {span.start() for span in out}
            for span in _regex_spans(
                text,
                self.aliases.safe_receivers,
                self.aliases.safe_funcs,
                None,
                self.aliases.loaders,
                self.aliases.loader_modules,
            ):
                if span.start() not in seen:
                    out.append(span)
            out.sort(key = lambda s: s.start())
        return out

    def search(self, text: str):
        if text is self.bound or "\n" in text:
            spans = self._spans(text)
            return spans[0] if spans else None
        hit = self._memo.get(text, False)
        if hit is False:
            # One line carries no file offset, so a rebinding cutoff cannot be
            # applied to it; scan it against the never-rebound aliases only.
            # `hit_lines` supplies whatever the whole-file pass saw instead.
            spans = self._spans(text, positional = False)
            hit = spans[0] if spans else None
            if len(self._memo) < self._MEMO_CAP:
                self._memo[text] = hit
        return hit

    def hit_lines(self, text: str) -> set:
        """1-based line numbers of `text` holding a call, from the whole-file scan.

        The per-line pass in `_extract_evidence` re-tokenizes each line on its
        own, and a line can only be adjudicated there if it is self-contained.
        `from builtins import exec as run` then `run(payload)`, or a call that
        precedes the rebinding of its alias, are decided by the whole file, so
        without this their evidence comes back empty and the baseline key binds
        no payload line at all.
        """
        cached = self._lines
        if cached is not None and cached[0] is text:
            return cached[1]
        spans = self._spans(text)
        rows: set = set()
        if spans:
            # Numbered the way `_extract_evidence` indexes its `lines`, which is
            # `str.splitlines` - that also breaks on "\r" and "\x0c", so counting
            # "\n"s here would hand back a row that names a different line.
            starts = []
            pos = 0
            for line in text.splitlines(keepends = True):
                starts.append(pos)
                pos += len(line)
            for span in spans:
                rows.add(bisect.bisect_right(starts, span.start()))
        self._lines = (text, rows)
        return rows

    def finditer(self, text: str):
        return iter(self._spans(text))


class _ExecEvalPattern:
    """`re.Pattern`-shaped view over every route to the exec/eval builtins.

    Only `search` and `finditer` are used against it (`_extract_evidence` and
    the per-check tables), plus `for_text` where the caller has the whole file
    and then re-matches its lines.
    """

    def __init__(self):
        self.pattern = _EXEC_EVAL_PATTERN_TEXT
        # check_py_file searches this pattern, then hands the same string to
        # _hidden_payload_findings and _extract_evidence. Deriving the aliases
        # is a tokenize pass over the whole member, so keep the last result and
        # reuse it when the identical string object comes back. Identity, not
        # equality: it is the cheap test, and holding the reference is what
        # makes it safe.
        self._cached: "tuple[str, _ExecEvalMatcher] | None" = None

    def for_text(self, content: str) -> _ExecEvalMatcher:
        """A matcher carrying the aliases `content` binds.

        Collecting them needs a pass of its own, because Python resolves a name
        when the call runs: `def go(): b.exec(p)` above `import builtins as b`
        still runs the builtin, so the aliases cannot be accumulated as the
        calls are adjudicated. The pass is skipped outright unless the word
        appears at all, which is every ordinary file.
        """
        cached = self._cached
        if cached is not None and cached[0] is content:
            return cached[1]
        modules: set = set()
        funcs: set = set()
        cancel: dict = {}
        # Where each name is declared `global`, as the offset range of the scope
        # that declared it. A call in there resolves at module level, so no
        # enclosing function's local of that name can silence it.
        declared_global: dict = {}
        # Where an alias imported inside a function may be read, for the aliases
        # no module-level binding makes visible everywhere.
        scoped_imports: dict = {}
        # Where an alias a function assigns is that function's local, as the
        # offsets of the body that assigns it. Python decides this by scanning
        # the block, so the whole body is shadowed - not just the lines below
        # the assignment.
        local_shadows: dict = {}
        loaders = _Loaders()
        # Every route spells one of the two words somewhere: a module alias at
        # its call site (`b.exec(...)`), a function alias at its import (`from
        # builtins import exec as run` - the collector only records a name
        # imported from `_EXEC_NAMES`). So a member holding neither has no alias
        # worth deriving, and `_scan` refuses the same text a moment later
        # anyway. The test is what keeps the tokenize pass off it: 1.3 MiB of
        # ordinary `builtins` statements spent 2.0s here to be rejected, and a
        # member is allowed up to 64 MiB.
        #
        # An escaped literal spells the module without the word appearing, so
        # that file has to be read too - but only when it also holds a call to
        # reach through the alias, since `_scan` returns nothing without one.
        if ("exec" in content or "eval" in content) and (
            "builtins" in content
            # A file that imports `import_module` binds a loader, and
            # `import_module(n).exec(...)` reaches the builtin without the word
            # `builtins` appearing anywhere.
            or "import_module" in content
            or _RE_STRING_ESCAPE.search(content)
            or _RE_SPLIT_BUILTINS.search(content)
        ):
            failed: list = []
            # The aliases bound by something other than an import. An import is
            # the one route whose scope the second pass tracks, so a name also
            # bound this way is left visible everywhere rather than confined to
            # some function that happens to import it too.
            by_value: set = set()
            for stmt in _statements(content, failed):
                head = stmt[0]
                if head.type == tokenize.NAME and head.string in ("import", "from"):
                    _collect_import_bindings(stmt, modules, funcs, loaders)
                else:
                    # `b = __import__('builtins')` binds an alias without an
                    # import statement, and the call through it is route 5 with
                    # the module parked in a name first; `(b := builtins)` parks
                    # it the same way inside an expression.
                    # `modules` is the running set of aliases, so a copy of one
                    # bound above (`c = b`) is read as the module here and can
                    # itself be copied again further down.
                    parked = _assignment_bindings(stmt, loaders, modules)
                    parked += _walrus_bindings(stmt, loaders, modules)
                    if parked:
                        by_value.update(parked)
                        modules.update(parked)
            modules |= by_value
            if failed:
                re_modules, re_funcs = _regex_bindings(content)
                modules |= re_modules
                funcs |= re_funcs
                # The fallback reports no scope, so a name it binds is read
                # wherever it appears.
                by_value |= re_modules | re_funcs
            if modules or funcs:
                # Only now is a second pass worth its cost: without an alias
                # there is nothing for a rebinding to cancel, and every file
                # that merely mentions `builtins` lands here.
                candidates = frozenset(modules | funcs)
                # Where each alias stops being the module, not merely whether it
                # ever does: `import builtins as b` ... `b.exec(BLOB)` ...
                # `b = harmless` runs the builtin, and a file-wide set
                # difference would let the trailing line erase the call above it.
                offsets = _Offsets(content)
                rebound: set = set()
                # The bindings that reach only the statement's own suite: a loop
                # target, an exception target, a parameter.
                scoped: set = set()
                # The rebindings whose block is still open, so a sibling scope
                # does not inherit one: `def a(): b = model` stops deciding
                # anything at the next statement written no deeper than it.
                opened: dict = {}
                # The same spans grouped by indent, so a statement that closes
                # nothing costs one comparison however many names are open.
                levels: list = []
                # The `def` and `class` headers the statement sits under, as
                # [indent, is a class, header offset, names declared global,
                # aliases imported here, header end offset, names this body
                # assigns, names declared nonlocal, enclosing cancellations this
                # body re-armed]. Only those two open a
                # scope, so this is what says whether a binding is a class
                # attribute.
                scopes: list = []
                # Where an alias imported inside a function is visible, as the
                # offsets of the scope that imported it.
                local_imports: dict = {}
                # Every compound header the statement sits under, as
                # [indent, keyword, runs unconditionally]. `match` is what tells
                # a `case` pattern apart from an ordinary variable named `case`,
                # and the last field is what lets an `if True:` rebinding
                # outlive its own dedent.
                blocks: list = []
                failed = []
                for stmt in _statements(content, failed):
                    head = stmt[0]
                    col = head.start[1]
                    at = offsets.of(*head.start)
                    if levels:
                        _cancel_close(opened, levels, at, col, blocks)
                    while blocks and blocks[-1][0] >= col:
                        blocks.pop()
                    in_match = bool(blocks) and blocks[-1][1] == "match"
                    block = _opens_block(stmt)
                    if block is not None:
                        blocks.append((col, block[0], block[1]))
                    while scopes and scopes[-1][0] >= col:
                        _close_scope(
                            scopes.pop(),
                            declared_global,
                            at,
                            local_imports,
                            by_value,
                            local_shadows,
                            cancel,
                        )
                    opens = _opens_scope(stmt)
                    # A class body is the one scope a name does not cross: the
                    # methods written in it do not see its bindings. Read before
                    # this statement's own header is pushed, and true for the
                    # header itself so a one-line `class C: b = model` counts.
                    in_class = (scopes and scopes[-1][1]) or opens == "class"
                    # Whether the name this statement binds becomes a class
                    # attribute. A `def` or `class` header binds its own name in
                    # the ENCLOSING scope, so `import builtins as b` followed by
                    # a module-level `class b:` really does replace the alias -
                    # reading the header itself as "inside a class" discarded
                    # that rebinding and reported the `b.exec(...)` under it.
                    # Read before the header is pushed, so this is the scope
                    # around the statement either way.
                    binds_in_class = bool(scopes and scopes[-1][1])
                    if opens is not None:
                        scopes.append(
                            [
                                col,
                                opens == "class",
                                at,
                                None,
                                None,
                                offsets.of(*_header_end(stmt).end),
                                None,
                                None,
                                None,
                            ]
                        )
                    if head.type == tokenize.NAME and head.string in ("global", "nonlocal"):
                        # `global b` in a nested function makes its `b` the
                        # module-level one, whatever the enclosing function did
                        # to its own local of that name. Recorded against the
                        # whole scope, because the declaration governs the body
                        # written above it as well as below.
                        #
                        # `nonlocal b` is the same statement about the enclosing
                        # function's binding rather than the module's, so it
                        # equally stops the assignments below it creating a
                        # local: `nonlocal b` then `b.exec(BLOB)` then `b = x`
                        # reads the outer alias and runs the builtin. It is kept
                        # apart from `global` because it does NOT make the
                        # rebinding a module-level one, which is the other thing
                        # `declared_global` decides.
                        if scopes:
                            slot = 3 if head.string == "global" else 7
                            names = scopes[-1][slot]
                            if names is None:
                                names = scopes[-1][slot] = set()
                            for tok in stmt[1:]:
                                if tok.type == tokenize.NAME:
                                    names.add(tok.string)
                        continue
                    if head.type == tokenize.NAME and head.string in ("import", "from"):
                        # Only an import naming an alias can bind or re-arm one,
                        # so the ordinary file never parses its imports twice.
                        if any(t.type == tokenize.NAME and t.string in candidates for t in stmt):
                            bound: set = set()
                            _collect_import_bindings(stmt, bound, bound, None)
                            for name in bound:
                                _rearm_here(opened, scopes, name, at)
                                for scope in scopes:
                                    # The import re-arms the alias, so the body
                                    # is no longer shadowing a name it only
                                    # assigns; the calls under it read the
                                    # module again.
                                    if scope[6]:
                                        scope[6].discard(name)
                            if scopes and not scopes[-1][1]:
                                # An import inside a `def` binds a local: the
                                # sibling function below it does not see the
                                # alias, and reading its own `run(...)` as the
                                # builtin is a false positive. A class body is
                                # left file-wide instead of scoped - its
                                # bindings are attributes, and guessing which
                                # of them a method resolves is how a real call
                                # would go unread.
                                local = scopes[-1][4]
                                if local is None:
                                    local = scopes[-1][4] = set()
                                local |= bound
                            else:
                                by_value |= bound
                        continue
                    _comprehension_shadows(stmt, candidates, offsets, local_shadows)
                    _lambda_shadows(stmt, candidates, offsets, local_shadows)
                    _collect_rebindings(stmt, rebound, candidates, scoped, in_match)
                    if scoped:
                        # Recorded one indent deeper than the header, which is
                        # what makes the span end at the first statement written
                        # back at the header's own indent: the suite is
                        # everything below the header and indented past it. The
                        # binding does not survive the suite - an empty loop or
                        # an unraised exception never made it, and Python
                        # deletes an exception target at the end of its handler
                        # - so nothing after the block may be cancelled by it.
                        #
                        # Measured from the header's colon, not the last token:
                        # `def f(run): return run(BLOB)` and `case run: run(BLOB)`
                        # keep the suite glued to the header, so the last token
                        # is the call's and the shadow would start after the very
                        # call the parameter or the capture already governs.
                        ends = offsets.of(*_header_end(stmt).end)
                        for name in scoped:
                            _cancel_add(cancel, opened, levels, name, ends, col + 1)
                        if scopes and opens is None and not in_class:
                            # Python decides a name is local by scanning the
                            # block, so a `for b in ...` or `except E as b`
                            # written anywhere in a function makes `b` that
                            # function's local THROUGHOUT it - a call above the
                            # loop raises `UnboundLocalError` rather than
                            # reaching an outer builtins alias. The suite span
                            # above still stands on its own: it is what covers
                            # the module level, where there is no local to make.
                            #
                            # A `def` header is skipped for the reason the
                            # assignment path skips one: its parameters are the
                            # new scope's, and that scope is the one the suite
                            # span already covers.
                            _shadow_scope(scopes[-1], scoped)
                        scoped.clear()
                    if not rebound:
                        continue
                    same = _self_assigned(stmt)
                    if same:
                        # `b = b` leaves the alias exactly as it was - at module
                        # level. Inside a function it is still an assignment, so
                        # Python makes the name that function's local over the
                        # whole body and the statement itself raises
                        # `UnboundLocalError`; no call in there can reach the
                        # module-level alias, above the line or below it. So the
                        # name is dropped from the rebindings and recorded as a
                        # local, rather than left as neither.
                        if scopes and opens is None and not in_class:
                            _shadow_scope(scopes[-1], same)
                        rebound -= same
                        if not rebound:
                            continue
                    aliased = _assignment_bindings(stmt, loaders, modules) + _walrus_bindings(
                        stmt, loaders, modules
                    )
                    if aliased:
                        copied = _copied_alias(stmt)
                        if (
                            copied is not None
                            and copied not in _BUILTINS_NAMES
                            and _cancelled_at(cancel, copied, at, col)
                        ):
                            # The name being copied stopped being the module
                            # above this line, so this is an ordinary rebinding
                            # of the target rather than a re-arm of it.
                            aliased = []
                    if aliased:
                        # The statement that makes the name the module is not a
                        # rebinding of it, and it undoes any earlier one:
                        # `b = load()` ... `b = __import__('builtins')` ...
                        # `b.exec(BLOB)` runs the builtin. Checked only for a
                        # statement that does rebind an alias, which is rare.
                        rearmed = offsets.of(*stmt[-1].end)
                        for name in aliased:
                            _rearm_here(opened, scopes, name, rearmed)
                            for scope in scopes:
                                # The body puts the module back in the name, so
                                # it is no longer shadowing an alias it only
                                # assigns something else to.
                                if scope[6]:
                                    scope[6].discard(name)
                        rebound.clear()
                        continue
                    if binds_in_class:
                        # `class C: b = model` binds an attribute of C, and the
                        # methods under it still resolve `b` to the global. So
                        # this rebinding cancels nothing they do; recording it
                        # suppressed the real `b.exec(...)` in every one of them.
                        rebound.clear()
                        continue
                    # Measured from the END of the statement, not its start:
                    # Python evaluates the right-hand side before it rebinds the
                    # target, so `b = b.exec(marshal.loads(BLOB))` runs the
                    # builtin and only then stops `b` being the module. Starting
                    # the span at the first token suppressed that call.
                    ends = offsets.of(*stmt[-1].end)
                    if block is not None:
                        # A compound header binds before its own suite runs, and
                        # a one-line suite is re-yielded as its own statement
                        # AFTER this one - so measuring from the last token puts
                        # the binding past the body it governs, and
                        # `def run(): return run(BLOB)` was read as a call to the
                        # imported builtin rather than as recursion, and
                        # `with open(p) as b: b.exec(BLOB)` as the module. The
                        # header end is where a multiline header already ends,
                        # and everything the header evaluates - the context
                        # expression, a walrus in a condition - sits before it.
                        ends = offsets.of(*_header_end(stmt).end)
                    if scopes and opens is None:
                        # An assignment inside a `def` makes the name that
                        # function's local for the whole body, so the alias is
                        # unreachable above the assignment too - a call there is
                        # an `UnboundLocalError`, not the builtin. A name the
                        # scope declared `global` is exempt: that is what stops
                        # the assignment creating a local at all.
                        #
                        # A statement that opens a scope is skipped: `def f()`
                        # binds `f` in the enclosing scope, and the tail of a
                        # one-line suite is re-yielded as its own statement,
                        # where the body it really belongs to is the open one.
                        _shadow_scope(scopes[-1], rebound)
                    for name in rebound:
                        # Statements arrive in source order, so the first
                        # rebinding seen at a given indent is the earliest one.
                        _cancel_add(cancel, opened, levels, name, ends, col)
                    rebound.clear()
                for scope in reversed(scopes):
                    _close_scope(
                        scope,
                        declared_global,
                        len(content),
                        local_imports,
                        by_value,
                        local_shadows,
                        cancel,
                    )
                if failed:
                    # The tokenizer gave up, so there is no reliable order or
                    # indent to compare against; cancel from the top of the file
                    # at column 0, which is the whole-file subtraction this path
                    # always did.
                    for name in _regex_rebindings(content) & candidates:
                        cancel[name] = [[0, 0, None]]
                else:
                    # A name the file also binds at module level, by assignment
                    # or through the regex fallback is visible everywhere; only
                    # the ones bound by a function-local import alone are
                    # confined to the scope that imported them.
                    for name, spans in local_imports.items():
                        if name in by_value:
                            continue
                        spans.sort()
                        scoped_imports[name] = spans
                for name, spans in declared_global.items():
                    spans.sort()
                    declared_global[name] = _index_globals(spans)
                for spans in local_shadows.values():
                    spans.sort()
        matcher = _ExecEvalMatcher(
            modules,
            funcs,
            cancel,
            content,
            loaders.funcs,
            declared_global,
            loaders.modules,
            scoped_imports,
            local_shadows,
        )
        self._cached = (content, matcher)
        return matcher

    def search(self, text: str):
        return self.for_text(text).search(text)

    def finditer(self, text: str):
        return self.for_text(text).finditer(text)


RE_EXEC_EVAL = _ExecEvalPattern()

# Network APIs (excludes urllib.parse which is pure string manipulation)
# ``httpx2`` is the pydantic-maintained successor and a separate import name, so the older
# ``httpx``-only alternative did not see it. openai 3.0.0 requires httpx2 and routes every
# call through it, which made the SDK's own HTTP invisible to each combined check that needs
# a network half (secrets + network, IMDS + network, archive + network).
RE_NETWORK = re.compile(
    r"\burllib\.request\b"
    r"|\burlopen\s*\("
    r"|\brequests\s*\.\s*(get|post|put|patch|delete|head|Session)\b"
    r"|\b(?:httpx|httpx2)\s*\.\s*(get|post|put|patch|delete|Client|AsyncClient)\b"
    r"|\bsocket\s*\.\s*(socket|create_connection)\b"
    r"|\bhttp\.client\b"
    r"|\bhttp\.server\b",
)

# Large base64 blob (>200 chars of contiguous base64 alphabet)
RE_LARGE_BLOB = re.compile(r"[A-Za-z0-9+/=]{200,}")

# Credential path access (requires file-access context, not just string mentions)
RE_CRED_ACCESS = re.compile(
    r"(?:open|Path|read_text|read_bytes)\s*\([^)]*?"
    r"(?:\.ssh[/\\]|\.aws[/\\]|\.kube[/\\]|\.gnupg[/\\]|\.docker[/\\]"
    r"|\.azure[/\\]|\.gcp[/\\]"
    r"|credentials\.json|\.git-credentials|\.npmrc|\.pypirc|wallet\.dat"
    r"|/etc/shadow|/etc/passwd"
    r"|id_rsa|id_ed25519|id_ecdsa"
    r"|kubeconfig|service-account-token)"
    r"|os\.path\.(?:join|expanduser)\([^)]*?"
    r"(?:\.ssh|\.aws|\.kube|\.gnupg|\.docker|\.azure|\.gcp|credentials)"
    r"|(?:open|Path)\(\s*['\"]\.env['\"]\s*[,)]",
    re.DOTALL,
)

# Chained / advanced obfuscation (marshal, compile, zlib, nested decode)
RE_OBFUSCATION = re.compile(
    r"\bmarshal\s*\.\s*(loads|load)\b"
    r"|\bcompile\s*\([^)]*['\"]exec['\"]\s*\)"
    r"|\bzlib\s*\.\s*decompress\b"
    r"|\blzma\s*\.\s*decompress\b"
    r"|\bbz2\s*\.\s*decompress\b"
    r"|\bbytearray\s*\(\s*\[.*?\]\s*\)"  # bytearray([104,101,...])
    r"|\bchr\s*\(\s*\d+\s*\).*chr\s*\(\s*\d+\s*\)"  # chr() obfuscation chains
    r"|\b__import__\s*\("  # dynamic import
    r"|\bgetattr\s*\(\s*__builtins__"  # getattr(__builtins__, ...)
    r"|\brotate\s*=.*\blambda\b.*\bchr\b"  # rotation ciphers
    r"|\b(?:b64decode|decodebytes)\s*\(.*(?:b64decode|decodebytes)\s*\(",  # double base64
    re.DOTALL,
)

# Embedded cryptographic keys (PEM-encoded)
RE_EMBEDDED_KEYS = re.compile(
    r"-----BEGIN\s+(?:RSA\s+)?(?:PUBLIC|PRIVATE|ENCRYPTED|EC|DSA|OPENSSH)\s+KEY-----"
    r"|\bRSA\s+PUBLIC\s+KEY\b.*[A-Za-z0-9+/=]{64,}"
    r"|\bMII[A-Za-z0-9+/]{20,}",  # DER-encoded key prefix (base64)
    re.DOTALL,
)

# Full PEM block (BEGIN..END), used to pin a multiline key body in evidence.
RE_PEM_BLOCK = re.compile(r"-----BEGIN[^\n]*KEY-----.*?-----END[^\n]*KEY-----", re.DOTALL)

# Cloud metadata / IMDS endpoints
RE_CLOUD_METADATA = re.compile(
    r"169\.254\.169\.254"  # AWS/Azure/GCP IMDS
    r"|metadata\.google\.internal"  # GCP metadata
    r"|169\.254\.170\.2"  # AWS ECS task metadata
    r"|100\.100\.100\.200"  # Alibaba Cloud metadata
    r"|/latest/meta-data"  # AWS IMDS path
    r"|/metadata/instance"  # GCP metadata path
    r"|/metadata/identity"  # Azure managed identity
    r"|\bIMDSv[12]\b",
)

# Persistence mechanisms (systemd, cron, launchd, registry, startup dirs)
RE_PERSISTENCE = re.compile(
    r"/etc/systemd/"
    r"|systemctl\s+(enable|start|daemon-reload)"
    r"|\.service\b.*\[Service\]"  # systemd unit content
    r"|/etc/cron"
    r"|crontab\s"
    r"|/etc/init\.d/"
    r"|/Library/LaunchDaemons"
    r"|/Library/LaunchAgents"
    r"|~/\.config/autostart"
    r"|~/.local/share/systemd"
    r"|~/\.config/systemd/user/"  # user-level systemd
    r"|HKEY_LOCAL_MACHINE.*\\\\Run"  # Windows registry autorun
    r"|HKEY_CURRENT_USER.*\\\\Run"
    r"|\\\\Start Menu\\\\Programs\\\\Startup"
    r"|schtasks\s",  # Windows scheduled tasks
    re.IGNORECASE,
)

# Container / orchestration abuse
RE_CONTAINER_ABUSE = re.compile(
    r"/var/run/docker\.sock"
    r"|\bdocker\s+(run|exec|cp|build)\b"
    r"|\bkubectl\s+(apply|create|exec|run|cp)\b"
    r"|\bkubernetes\.client\b"
    r"|\bfrom_incluster_config\b"
    r"|\blist_namespaced_secret\b"
    r"|\bcreate_namespaced_pod\b"
    r"|\bcreate_namespaced_daemon_set\b"
    r"|\bcreate_namespaced_secret\b"
    r"|\bkube-system\b"
    r"|\bhostPID\s*:\s*true"
    r"|\bprivileged\s*:\s*true"
    r"|\bhostNetwork\s*:\s*true"
    r"|\bhostPath\b.*\bpath\s*:\s*/",  # k8s hostPath mounts
    re.IGNORECASE,
)

# Environment variable harvesting (bulk access or known secret vars)
RE_ENV_HARVEST = re.compile(
    r"\bos\.environ\s*\.\s*copy\s*\("  # full env copy
    r"|\bdict\s*\(\s*os\.environ\s*\)"
    r"|\bjson\.dumps\s*\(\s*(?:dict\s*\(\s*)?os\.environ"
    r"|\bfor\s+\w+\s*,\s*\w+\s+in\s+os\.environ\.items\(\)"  # iterating all env vars
    r"|\bos\.environ\b.*(?:SECRET|TOKEN|KEY|PASSWORD|CREDENTIAL|API_KEY|PRIVATE)"
    r"|\b(?:SECRET|TOKEN|PASSWORD|API_KEY|PRIVATE_KEY)\b.*os\.environ",
    re.IGNORECASE,
)

# Archive staging / exfiltration prep (create archive + network send)
RE_ARCHIVE_STAGING = re.compile(
    r"\btarfile\s*\.\s*open\s*\("
    r"|\bzipfile\s*\.\s*ZipFile\s*\([^)]*['\"]w['\"]\s*\)"
    r"|\bshutil\s*\.\s*make_archive\b"
    r"|\b\.add\s*\([^)]*(?:\.ssh|\.aws|\.env|\.kube|credentials|\.gnupg|\.docker)"
    r"|\b\.write\s*\([^)]*(?:\.ssh|\.aws|\.env|\.kube|credentials|\.gnupg|\.docker)",
    re.DOTALL,
)

# Anti-analysis / sandbox evasion / debugger detection
# NB: deliberately does NOT include a bare ``platform.system() ... Linux/Windows
# /Darwin`` branch. Under re.DOTALL that matched across the whole file -- any
# cross-platform library (typer, packaging, pandas, pymupdf, ...) trips it -- so
# it had ~zero precision and only generated false positives. OS detection alone
# is not an anti-analysis signal; the debugger/VM/long-sleep signals below are.
RE_ANTI_ANALYSIS = re.compile(
    r"\bptrace\b"
    r"|\bsys\s*\.\s*gettrace\s*\("
    r"|\bsys\s*\.\s*settrace\b"
    r"|\bTracerPid\b"
    # /proc/self/status is read to scrape TracerPid for anti-debug. A leading
    # \b here is unsatisfiable (\b never holds between a non-word boundary and
    # "/"), so the old pattern was dead; a lookbehind that only forbids a
    # preceding word char or path separator lets `open("/proc/self/status")`
    # and `cat /proc/self/status` match while avoiding mid-path partials.
    r"|(?<![\w/])/proc/self/status\b"
    r"|\bIsDebuggerPresent\b"
    r"|\bvirtualbox\b.*\bhardware\b"
    r"|\bvmware\b.*\bdetect\b"
    r"|\btime\.sleep\s*\(\s*(?:[3-9]\d{2,}|[1-9]\d{3,})\s*\)",  # long sleep (anti-sandbox)
    re.IGNORECASE | re.DOTALL,
)

# DNS exfiltration / tunneling
RE_DNS_EXFIL = re.compile(
    r"\bdns\.resolver\b"
    r"|\bsocket\.getaddrinfo\s*\([^)]*\+[^)]*\)"  # dynamic hostname construction
    r"|\bdnspython\b"
    r"|\bTXT\b.*\bresolver\b"
    r"|\bresolver\b.*\bTXT\b"
    r"|\bnslookup\b"
    r"|\bdig\s+",
)

# File system enumeration / bulk file theft
RE_FS_ENUM = re.compile(
    r"\bos\.walk\s*\(\s*['\"](?:/|~|/home|/root|/Users|C:\\\\)"
    r"|\bglob\s*\.\s*glob\s*\([^)]*(?:\*\*|\*\.pem|\*\.key|\*\.cer|\*\.pfx|\*\.p12)"
    r"|\bos\.listdir\s*\(\s*['\"](?:/home|/root|/Users|/etc)"
    r"|\bPath\s*\(\s*['\"]~['\"]\s*\)\s*\.\s*glob\b"
    r"|\bhistory\b.*\bread\b"  # reading shell history
    r"|\b\.bash_history\b"
    r"|\b\.zsh_history\b"
    r"|/etc/shadow"
    r"|/etc/passwd",
    re.DOTALL,
)

# Reverse shell / bind shell patterns
RE_REVERSE_SHELL = re.compile(
    r"\bsocket\b.*\bconnect\b.*\bsubprocess\b"
    r"|\bsocket\b.*\bconnect\b.*\b(?:sh|bash|cmd)\b"
    r"|\b/bin/(?:sh|bash)\b.*\bsocket\b"
    r"|\bpty\s*\.\s*spawn\b"
    r"|\bos\s*\.\s*dup2\s*\("
    r"|\bwebbrowser\s*\.\s*open\b.*\bdata:\b",  # data: URI abuse
    re.DOTALL,
)

# Process injection / code loading from remote
RE_REMOTE_CODE = re.compile(
    r"\bexec\s*\(\s*(?:urllib|requests|httpx|urlopen)"  # exec(requests.get(...))
    r"|\bexec\s*\([^)]*\.(?:text|content|read)\s*\("
    r"|\beval\s*\([^)]*\.(?:text|content|read)\s*\("
    r"|\bimportlib\s*\.\s*import_module\s*\([^)]*\+"  # dynamic import with concatenation
    r"|\b__import__\s*\([^)]*\+",  # __import__ with concatenation
    re.DOTALL,
)

# Crypto wallet / cryptocurrency theft
RE_CRYPTO_THEFT = re.compile(
    r"\bwallet\.dat\b"
    r"|\b\.bitcoin[/\\]"
    r"|\b\.ethereum[/\\]"
    r"|\b\.solana[/\\]"
    r"|\b\.monero[/\\]"
    r"|\b\.litecoin[/\\]"
    r"|\b\.config/solana[/\\]"
    r"|\bkeystore[/\\]UTC--"
    r"|\bseed\s*phrase\b"
    r"|\bmnemonic\b.*\b(?:word|phrase|recover|restore)\b"
    r"|\b(?:xprv|xpub|bc1|0x[a-fA-F0-9]{40})\b",
    re.IGNORECASE,
)

# Import line in .pth (Python site.py only exec()s lines starting with "import")
RE_PTH_IMPORT = re.compile(r"^\s*import\s+", re.MULTILINE)

# openssl CLI invocations via subprocess (encrypted exfiltration)
RE_OPENSSL_CLI = re.compile(r"\bopenssl\s+(enc|rand|rsautl|pkeyutl|genrsa|dgst|s_client)\b")

# Write to /tmp then execute (staged dropper)
RE_TEMP_EXEC = re.compile(
    r"/tmp/\S+.*(?:subprocess|os\.system|os\.popen|Popen|chmod.*\+x)",
    re.DOTALL,
)

# C2 polling / beaconing loop
RE_C2_POLLING = re.compile(
    r"while\s+True.*(?:time\.sleep|sleep)\s*\(.*(?:urlopen|requests\.|httpx\.)",
    re.DOTALL,
)

# Developer-tool persistence hooks. Lightning 2.6.x planted SessionStart hooks
# into Claude Code / VS Code / Cursor so the payload re-attached on editor open.
RE_DEV_TOOL_HIJACK = re.compile(
    r"\.claude/settings\.json"
    r"|\.cursor/.*hooks"
    r"|\.vscode/(?:tasks|settings|launch)\.json"
    r"|SessionStart|folderOpen|onCommand:.*runTask"
    r"|/etc/profile\.d/"
    r"|\b\.bashrc\b|\b\.zshrc\b|\b\.profile\b"
    r"|\bautomator\b.*\.workflow\b",
)

# Hard-coded credential / API-token regexes embedded in source. Packages that
# ship regexes for OTHER people's secrets are nearly always stealers.
RE_TOKEN_REGEX = re.compile(
    r"\bgh[psoru]_[A-Za-z0-9_]{20,}"  # GitHub PAT/OAuth/etc.
    r"|\bgithub_pat_[A-Za-z0-9_]{20,}"
    r"|\bnpm_[A-Za-z0-9]{30,}"  # npm token
    r"|\bsk-[A-Za-z0-9]{20,}"  # OpenAI / Anthropic
    r"|\bxox[bpaesr]-"  # Slack
    r"|\bAIza[0-9A-Za-z_-]{20,}"  # Google API key
    r"|\bAKIA[0-9A-Z]{16}"  # AWS access key id
    r"|\bASIA[0-9A-Z]{16}"  # AWS STS
    r"|\bgithub.com/login/oauth/access_token"
    r"|\bglpat-[0-9A-Za-z_-]{20,}",  # GitLab PAT
)

# Mini Shai-Hulud May-12 2026 wave indicators. `transformers.pyz` dropper name
# is high-confidence; the host + slogans are CRITICAL.
RE_MAY12_IOC = re.compile(
    r"(git-tanstack\.com|/tmp/transformers\.pyz|transformers\.pyz"
    r"|With Love TeamPCP|We've been online over 2 hours)",
    re.IGNORECASE,
)

# JavaScript-side obfuscation. A bundle full of `_0x1f2e3d` hex-var identifiers
# is a near-universal tell for a malicious npm payload, rare in legit wheels.
RE_JS_OBFUSCATION = re.compile(
    r"_0x[a-f0-9]{4,6}\s*=\s*function"
    r"|var\s+_0x[a-f0-9]{4,6}\b"
    r"|(?:\\x[0-9a-f]{2}){10,}"  # \x-escape strings
    r"|String\.fromCharCode\s*\(\s*\d+\s*(?:,\s*\d+\s*){10,}\)",
)

# Web3 / wallet-hijack pattern. The Qix npm phish overrode fetch/XMLHttpRequest
# and swapped recipient addresses via a `window.ethereum` listener.
RE_WEB3_HIJACK = re.compile(
    r"\bwindow\.ethereum\b"
    r"|\bweb3\.eth\.\w+\s*\("
    r"|XMLHttpRequest\.prototype\.(?:open|send)\s*="
    r"|(?:^|\s)fetch\s*=\s*\(?\s*async"
    r"|TronWeb|solanaWeb3",
)

# Self-propagating worms (Shai-Hulud, ForceMemo) plant their own GitHub workflow
# in every repo they reach and use trufflehog/gitleaks for credential discovery.
# Any of these strings in a package payload is strong repo-takeover evidence.
RE_WORKFLOW_INJECT = re.compile(
    r"\.github/workflows/[^\"\']*\.ya?ml"
    r"|\btrufflehog\b|\bgitleaks\b"
    r"|/user/repos\?affiliation=.*owner.*collaborator"
    r"|\bshai-hulud\b|EveryBoiWeBuildIsAWormyBoi"
    r"|\bgit\s+push\s+--force\b.*--no-verify",
    re.IGNORECASE | re.DOTALL,
)

# install.sh / postinstall scripts piping remote code into a shell.
# `curl ... | sh` is the canonical npm postinstall dropper.
RE_SHELL_DROPPER = re.compile(
    r"\bcurl\b[^\n|]*\|\s*(?:sh|bash|zsh)\b"
    r"|\bwget\b[^\n|]*-O-\s*\|\s*(?:sh|bash|zsh)\b"
    r"|\bnpx\b\s+-y\s+[^\s]+@latest\s*\|"
    r"|\beval\s+\$\(\s*curl\b"
    r"|\bbash\s+<\(\s*curl\b",
)


@dataclass
class Finding:
    severity: str
    package: str
    filename: str
    check: str
    evidence: str = ""
    # Whole-file digest; a baseline entry may pin it (see _load_baseline).
    file_sha256: str = ""


# Checkers


def check_pth_file(content: str, filename: str, package: str) -> list[Finding]:
    """Run all .pth-specific checks.

    Executable .pth files run on every Python startup, so any suspicious
    pattern in a .pth is treated as CRITICAL.
    """
    findings = []

    # Only .pth files with import lines are executable
    import_lines = [line for line in content.splitlines() if RE_PTH_IMPORT.match(line)]
    if not import_lines:
        return findings  # Pure path entries, inert

    # All patterns are CRITICAL inside executable .pth files
    _pth_checks = [
        (RE_SUBPROCESS, ".pth has subprocess/os exec calls"),
        (RE_BASE64, ".pth has base64/encoding obfuscation"),
        (RE_EXEC_EVAL, ".pth has exec()/eval()"),
        (RE_NETWORK, ".pth has network API calls"),
        (
            RE_OBFUSCATION,
            ".pth has advanced obfuscation (marshal/compile/zlib/__import__)",
        ),
        (RE_EMBEDDED_KEYS, ".pth has embedded cryptographic key material"),
        (RE_CLOUD_METADATA, ".pth accesses cloud metadata / IMDS endpoints"),
        (RE_PERSISTENCE, ".pth installs persistence (systemd/cron/launchd/registry)"),
        (RE_CONTAINER_ABUSE, ".pth interacts with container/orchestration runtime"),
        (RE_ENV_HARVEST, ".pth harvests environment variables / secrets"),
        (RE_ARCHIVE_STAGING, ".pth stages archive for exfiltration"),
        (RE_ANTI_ANALYSIS, ".pth has anti-analysis / sandbox evasion"),
        (RE_DNS_EXFIL, ".pth has DNS exfiltration / tunneling patterns"),
        (RE_FS_ENUM, ".pth enumerates filesystem / steals files"),
        (RE_REVERSE_SHELL, ".pth has reverse/bind shell patterns"),
        (RE_REMOTE_CODE, ".pth loads and executes remote code"),
        (RE_CRYPTO_THEFT, ".pth targets cryptocurrency wallets / keys"),
        (RE_CRED_ACCESS, ".pth accesses credential files"),
        (RE_OPENSSL_CLI, ".pth invokes openssl CLI (encrypted exfil pattern)"),
        (RE_TEMP_EXEC, ".pth writes to /tmp and executes (staged dropper)"),
        (RE_C2_POLLING, ".pth has C2 polling/beaconing loop"),
    ]

    for pattern, description in _pth_checks:
        if pattern.search(content):
            findings.append(
                Finding(
                    CRITICAL,
                    package,
                    filename,
                    description,
                    _extract_evidence(content, pattern),
                )
            )

    # Large base64 blob
    if RE_LARGE_BLOB.search(content):
        # Digest every blob (not just the first 120 chars, and not just the
        # first blob), so a later payload that keeps the prefix or appends a
        # second encoded blob reopens.
        blob, digest = _blob_digest(content)
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                f".pth has large base64-like blob ({len(blob)} chars)",
                f"{blob[:120]}... sha256:{digest}",
            )
        )

    # Catch-all: any import line in .pth if nothing else triggered. Bind every
    # line through a digest so an appended/swapped import reopens the key, but cap
    # the displayed text so a large .pth of benign-looking imports cannot dump up
    # to the archive member cap into the logs or baseline JSON.
    if not findings and import_lines:
        evidence = _cap_line("\n".join(import_lines))
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                f".pth has {len(import_lines)} executable import line(s)",
                evidence,
            )
        )

    # Unusually large executable .pth (litellm's was 34 KB; legit ones are <100 bytes)
    size = len(content)
    if size > 500 and import_lines:
        # Pin the content so a different payload of the same size/import count reopens.
        digest = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                f"Unusually large executable .pth ({size} bytes)",
                f"{len(import_lines)} import line(s) in {size}-byte .pth file sha256:{digest}",
            )
        )

    return findings


# A STRING after one of these tokens (and before a NEWLINE) is a bare
# docstring/doctest/prose statement -- the dominant FP source -- so we blank it.
# A string after `=` or `(` is real code and is never blanked.
_LINE_START_TOKENS = frozenset({tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT})


def _is_fstring(tok_string: str) -> bool:
    """True if a STRING token is an f-string (3.10/3.11 emit one STRING token).

    A bare f-string statement evaluates its expressions at import, so unlike an
    inert docstring it must never be blanked.
    """
    q = min((tok_string.find(c) for c in "'\"" if c in tok_string), default = -1)
    return q > 0 and "f" in tok_string[:q].lower()


def _strip_noncode(content: str, blank_comments: bool = True) -> str:
    """Blank comments and bare docstrings so IOC patterns see code only.

    Removed regions become spaces (newlines kept) so line numbers stay exact for
    _extract_evidence. Fails open on tokenizer errors (the raw text is still
    fully scanned, so a real detection is never lost). ``blank_comments=False``
    keeps comments (only strings/docstrings blanked) to isolate the span that
    exec() could actually run.
    """
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(content).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return content

    spans: list[tuple[int, int, int, int]] = []  # (srow, scol, erow, ecol)
    prev_significant = tokenize.NEWLINE  # start-of-file behaves like a new line
    n = len(toks)
    for i, tok in enumerate(toks):
        ttype = tok.type
        if ttype == tokenize.COMMENT:
            if blank_comments:
                spans.append((*tok.start, *tok.end))
            continue  # transparent; never advances prev_significant
        if (
            ttype == tokenize.STRING
            and prev_significant in _LINE_START_TOKENS
            and not _is_fstring(tok.string)  # f-strings execute; never blank them
        ):
            # Bare string only if it is the whole statement: next significant
            # token must close the logical line.
            j = i + 1
            while j < n and toks[j].type in (tokenize.COMMENT, tokenize.NL):
                j += 1
            if j < n and toks[j].type == tokenize.NEWLINE:
                spans.append((*tok.start, *tok.end))
                prev_significant = ttype
                continue
        if ttype in (
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENCODING,
        ):
            prev_significant = ttype
            continue
        prev_significant = ttype

    if not spans:
        return content

    buf = content.splitlines(keepends = True)
    for srow, scol, erow, ecol in spans:
        for row in range(srow, erow + 1):
            line = buf[row - 1]
            if line.endswith("\n"):
                body, nl = line[:-1], "\n"
            elif line.endswith("\r"):
                body, nl = line[:-1], "\r"
            else:
                body, nl = line, ""
            start = scol if row == srow else 0
            end = ecol if row == erow else len(body)
            end = min(end, len(body))
            if start < end:
                body = body[:start] + (" " * (end - start)) + body[end:]
            buf[row - 1] = body + nl
    return "".join(buf)


# Payload carriers that are suspicious when hidden in a blanked region (a
# docstring/string) of a file that can dynamically execute strings.
_HIDDEN_PAYLOAD_PATTERNS = (
    (RE_LARGE_BLOB, "large base64 blob"),
    (RE_EMBEDDED_KEYS, "embedded key material"),
    (RE_MAY12_IOC, "Shai-Hulud IOC string"),
    (RE_OBFUSCATION, "marshal/compile/obfuscation"),
)


def _hidden_payload_findings(
    original: str, stripped: str, filename: str, package: str
) -> list[Finding]:
    """Flag payloads that live only in the blanked (docstring/string) region of
    a file that contains exec/eval. Such a string is invisible to code-only
    scanning yet ``exec(__doc__)`` / ``exec(<str>)`` could still run it."""
    if not RE_EXEC_EVAL.search(stripped):
        return []
    # Only docstrings/strings run via exec(__doc__)/exec(<str>); comments cannot.
    # Isolate that span: keep comments as real code, take what string-blanking
    # removed (length-preserved, so offsets stay exact for _extract_evidence).
    code = _strip_noncode(original, blank_comments = False)
    removed = "".join(o if o != s else " " for o, s in zip(original, code))
    out = []

    # The visible exec/eval line is what makes the hidden string executable, so
    # bind it into every finding's evidence: otherwise a reviewed false positive
    # that keeps the same hidden text but flips a harmless `eval("1+1")` to
    # `exec(__doc__)` (now running the payload) keeps the same key and stays
    # suppressed. Taken from `stripped` (real code), where the exec/eval lives.
    trigger = _extract_evidence(stripped, RE_EXEC_EVAL)

    def _hidden(pat):
        # Carrier present in a blanked region but NOT in real code. A carrier in
        # real code is already caught by the normal check, so restricting to
        # blanked-only avoids re-flagging legitimate in-code constants.
        return bool(pat.search(removed)) and not pat.search(stripped)

    for pat, label in _HIDDEN_PAYLOAD_PATTERNS:
        if _hidden(pat):
            out.append(
                Finding(
                    HIGH,
                    package,
                    filename,
                    "exec/eval with payload hidden in a docstring/string",
                    f"exec: {trigger}\n{label}: {_extract_evidence(removed, pat)}",
                )
            )
    # Fetch-then-run dropper: a network call AND an os/subprocess exec that both
    # live in the blanked region. Search the removed span directly (not "absent
    # from real code") so a benign visible network/subprocess call cannot mask
    # the docstring payload.
    if RE_NETWORK.search(removed) and RE_SUBPROCESS.search(removed):
        out.append(
            Finding(
                HIGH,
                package,
                filename,
                "exec/eval with hidden network+exec payload",
                f"exec: {trigger}\n"
                f"network+exec: {_extract_evidence(removed, RE_NETWORK)} | "
                f"{_extract_evidence(removed, RE_SUBPROCESS)}",
            )
        )
    return out


def check_py_file(content: str, filename: str, package: str) -> list[Finding]:
    """Run all .py-specific checks."""
    # Code-only scanning: strip comments/docstrings up front so prose, doctests
    # and usage examples cannot manufacture false positives. Aligns with the
    # Hugging Face Hub model (ClamAV/picklescan: low-FP, signature/structural).
    original = content
    content = _strip_noncode(content)
    findings = _hidden_payload_findings(original, content, filename, package)
    basename = os.path.basename(filename)
    is_setup = basename in ("setup.py", "setup.cfg")
    is_init = basename == "__init__.py"

    # Pre-compute pattern matches
    has_network = bool(RE_NETWORK.search(content))
    has_subprocess = bool(RE_SUBPROCESS.search(content))
    has_base64 = bool(RE_BASE64.search(content))
    has_exec_eval = bool(RE_EXEC_EVAL.search(content))
    has_creds = bool(RE_CRED_ACCESS.search(content))
    has_blob = bool(RE_LARGE_BLOB.search(content))
    has_obfuscation = bool(RE_OBFUSCATION.search(content))
    has_keys = bool(RE_EMBEDDED_KEYS.search(content))
    has_cloud_meta = bool(RE_CLOUD_METADATA.search(content))
    has_persistence = bool(RE_PERSISTENCE.search(content))
    has_container = bool(RE_CONTAINER_ABUSE.search(content))
    has_env_harvest = bool(RE_ENV_HARVEST.search(content))
    has_archive = bool(RE_ARCHIVE_STAGING.search(content))
    has_anti = bool(RE_ANTI_ANALYSIS.search(content))
    has_dns_exfil = bool(RE_DNS_EXFIL.search(content))
    has_fs_enum = bool(RE_FS_ENUM.search(content))
    has_rev_shell = bool(RE_REVERSE_SHELL.search(content))
    has_remote_code = bool(RE_REMOTE_CODE.search(content))
    has_crypto_theft = bool(RE_CRYPTO_THEFT.search(content))
    has_openssl_cli = bool(RE_OPENSSL_CLI.search(content))
    has_temp_exec = bool(RE_TEMP_EXEC.search(content))
    has_c2_polling = bool(RE_C2_POLLING.search(content))
    has_may12_ioc = bool(RE_MAY12_IOC.search(content))

    # CRITICAL: combination patterns that strongly indicate malice

    # base64 decode + subprocess execution (staged payload)
    if has_base64 and has_subprocess:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "base64 decode + subprocess execution (staged payload)",
                f"Base64: {_extract_evidence(content, RE_BASE64)}\n"
                f"Subprocess: {_extract_evidence(content, RE_SUBPROCESS)}",
            )
        )

    # openssl encryption + network/key material (encrypted exfiltration)
    if has_openssl_cli and (has_network or has_keys):
        # Bind whichever side(s) co-occur so a changed endpoint or key reopens.
        evidence = [f"OpenSSL: {_extract_evidence(content, RE_OPENSSL_CLI)}"]
        if has_network:
            evidence.append(f"Network: {_extract_evidence(content, RE_NETWORK)}")
        if has_keys:
            evidence.append(f"Key: {_embedded_key_evidence(content)}")
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "openssl encryption + network/key material (encrypted exfiltration)",
                "\n".join(evidence),
            )
        )

    # Writes to /tmp and executes (staged dropper)
    if has_temp_exec:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Writes to /tmp and executes (staged dropper)",
                _extract_evidence(content, RE_TEMP_EXEC),
            )
        )

    # May-12 Shai-Hulud IOC string in Python source.
    if has_may12_ioc:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "May-12 Shai-Hulud IOC string present in Python file",
                _extract_evidence(content, RE_MAY12_IOC),
            )
        )

    # C2 polling/beaconing loop
    if has_c2_polling:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "C2 polling/beaconing loop detected",
                _extract_evidence(content, RE_C2_POLLING),
            )
        )

    # Credential stealer: reads cred paths AND phones home
    if has_creds and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Reads credential paths AND makes network calls",
                f"Creds: {_extract_evidence(content, RE_CRED_ACCESS)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Reverse / bind shell
    if has_rev_shell:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Reverse shell / bind shell pattern",
                _extract_evidence(content, RE_REVERSE_SHELL),
            )
        )

    # Remote code execution: exec/eval on HTTP response
    if has_remote_code:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Downloads and executes remote code",
                _extract_evidence(content, RE_REMOTE_CODE),
            )
        )

    # Env harvest + network exfil
    if has_env_harvest and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Harvests environment variables/secrets AND makes network calls",
                f"Env: {_extract_evidence(content, RE_ENV_HARVEST)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Filesystem enum + network exfil
    if has_fs_enum and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Enumerates filesystem AND makes network calls",
                f"FS: {_extract_evidence(content, RE_FS_ENUM)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Cloud metadata access + network (exfil IMDS tokens)
    if has_cloud_meta and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Accesses cloud metadata/IMDS AND makes network calls",
                f"IMDS: {_extract_evidence(content, RE_CLOUD_METADATA)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Crypto wallet theft + network
    if has_crypto_theft and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Targets cryptocurrency wallets AND makes network calls",
                f"Crypto: {_extract_evidence(content, RE_CRYPTO_THEFT)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Archive staging with credential content + network
    if has_archive and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Creates archive with sensitive data AND makes network calls",
                f"Archive: {_extract_evidence(content, RE_ARCHIVE_STAGING)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Persistence + network (dropper that persists)
    if has_persistence and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Installs persistence AND makes network calls (backdoor pattern)",
                f"Persist: {_extract_evidence(content, RE_PERSISTENCE)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Container/k8s abuse + network
    if has_container and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Container/orchestration abuse AND makes network calls",
                f"Container: {_extract_evidence(content, RE_CONTAINER_ABUSE)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # HIGH: single strong signals or weaker combinations

    # Obfuscated payload: base64 + exec/eval + large blob
    if has_base64 and has_exec_eval and has_blob:
        # Digest every blob too: a payload may sit on a separate line from the
        # decode call, and a second encoded blob may be appended later, so
        # binding only the base64/exec lines or the first blob would miss it.
        _, blob_digest = _blob_digest(content)
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "base64 decode + exec/eval + large encoded blob",
                f"Base64: {_extract_evidence(content, RE_BASE64)}\n"
                f"Exec: {_extract_evidence(content, RE_EXEC_EVAL)}\n"
                f"Blob: sha256:{blob_digest}",
            )
        )

    # Advanced obfuscation + exec/eval
    if has_obfuscation and has_exec_eval:
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Advanced obfuscation (marshal/compile/zlib) + exec/eval",
                f"Obfusc: {_extract_evidence(content, RE_OBFUSCATION)}\n"
                f"Exec: {_extract_evidence(content, RE_EXEC_EVAL)}",
            )
        )

    # Embedded crypto key + network (hardcoded key for encrypted exfil)
    if has_keys and has_network:
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Embedded cryptographic key + network calls (encrypted exfil pattern)",
                f"Key: {_embedded_key_evidence(content)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )

    # Anti-analysis + any other suspicious pattern
    if has_anti and (has_network or has_subprocess or has_exec_eval):
        # Bind the suspicious side too so a changed payload reopens.
        evidence = [f"Anti: {_extract_evidence(content, RE_ANTI_ANALYSIS)}"]
        if has_network:
            evidence.append(f"Network: {_extract_evidence(content, RE_NETWORK)}")
        if has_subprocess:
            evidence.append(f"Subprocess: {_extract_evidence(content, RE_SUBPROCESS)}")
        if has_exec_eval:
            evidence.append(f"Exec: {_extract_evidence(content, RE_EXEC_EVAL)}")
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Anti-analysis/sandbox evasion + suspicious behavior",
                "\n".join(evidence),
            )
        )

    # DNS exfiltration with dynamic hostnames
    if has_dns_exfil and (has_base64 or has_network or has_creds):
        # Bind the co-occurring side so a changed exfil channel reopens.
        evidence = [f"DNS: {_extract_evidence(content, RE_DNS_EXFIL)}"]
        if has_base64:
            evidence.append(f"Base64: {_extract_evidence(content, RE_BASE64)}")
        if has_network:
            evidence.append(f"Network: {_extract_evidence(content, RE_NETWORK)}")
        if has_creds:
            evidence.append(f"Creds: {_extract_evidence(content, RE_CRED_ACCESS)}")
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "DNS exfiltration / tunneling patterns",
                "\n".join(evidence),
            )
        )

    # Cloud metadata standalone (IMDS access in a PyPI package is suspicious)
    if has_cloud_meta and not findings:
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Accesses cloud metadata / IMDS endpoints",
                _extract_evidence(content, RE_CLOUD_METADATA),
            )
        )

    # Persistence standalone (a PyPI package installing systemd/cron is suspicious)
    if has_persistence and not has_network:
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Installs persistence mechanism (systemd/cron/launchd/registry)",
                _extract_evidence(content, RE_PERSISTENCE),
            )
        )

    # Container abuse standalone
    if has_container and not has_network:
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Interacts with container/orchestration runtime",
                _extract_evidence(content, RE_CONTAINER_ABUSE),
            )
        )

    # openssl CLI standalone (uncommon in PyPI packages)
    if has_openssl_cli and not (has_network or has_keys):
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Invokes openssl CLI (uncommon in PyPI packages)",
                _extract_evidence(content, RE_OPENSSL_CLI),
            )
        )

    # setup.py checks
    if is_setup:
        if has_network and has_subprocess:
            findings.append(
                Finding(
                    HIGH,
                    package,
                    filename,
                    "setup.py has network calls + subprocess (dropper pattern)",
                    f"Network: {_extract_evidence(content, RE_NETWORK)}\n"
                    f"Subprocess: {_extract_evidence(content, RE_SUBPROCESS)}",
                )
            )
        elif has_network:
            findings.append(
                Finding(
                    MEDIUM,
                    package,
                    filename,
                    "setup.py makes network calls at install time",
                    _extract_evidence(content, RE_NETWORK),
                )
            )

    # MEDIUM: standalone signals (informational, may be legitimate)

    # base64 + exec/eval without blob
    if has_base64 and has_exec_eval and not has_blob:
        findings.append(
            Finding(
                MEDIUM,
                package,
                filename,
                "base64 decode + exec/eval (no large blob)",
                f"Base64: {_extract_evidence(content, RE_BASE64)}\n"
                f"Exec: {_extract_evidence(content, RE_EXEC_EVAL)}",
            )
        )

    # Standalone obfuscation without exec
    if has_obfuscation and not has_exec_eval:
        findings.append(
            Finding(
                MEDIUM,
                package,
                filename,
                "Advanced obfuscation patterns (marshal/compile/zlib/__import__)",
                _extract_evidence(content, RE_OBFUSCATION),
            )
        )

    # Embedded crypto keys standalone
    if has_keys and not has_network:
        findings.append(
            Finding(
                MEDIUM,
                package,
                filename,
                "Embedded cryptographic key material",
                _embedded_key_evidence(content),
            )
        )

    # Env harvest standalone
    if has_env_harvest and not has_network:
        findings.append(
            Finding(
                MEDIUM,
                package,
                filename,
                "Harvests environment variables / secrets",
                _extract_evidence(content, RE_ENV_HARVEST),
            )
        )

    # Filesystem enum standalone
    if has_fs_enum and not has_network:
        findings.append(
            Finding(
                MEDIUM,
                package,
                filename,
                "Enumerates filesystem / reads sensitive file paths",
                _extract_evidence(content, RE_FS_ENUM),
            )
        )

    # Crypto wallet references standalone
    if has_crypto_theft and not has_network:
        findings.append(
            Finding(
                MEDIUM,
                package,
                filename,
                "References cryptocurrency wallets / keys",
                _extract_evidence(content, RE_CRYPTO_THEFT),
            )
        )

    digest = hashlib.sha256(original.encode("utf-8", "replace")).hexdigest()
    for f in findings:
        f.file_sha256 = digest
    return findings


_MAX_MULTILINE_LINES = 12
# How far a single matched call is followed over its bracket continuations. A call
# that genuinely closes is bound all the way to its real close, up to the hard
# limit, so a ``requests.post(`` with many option/header lines before ``data=``
# binds its whole argument list in the digest and a changed payload on a late
# continuation line reopens (a 40-line soft cap would hash only the first 40 lines
# and let a later ``data=``/headers change ride the baseline key). A bracket that
# never closes within the hard limit is a miscount (a multi-line string the
# single-line blanker cannot mask) or a stray opener, so it is bound only to the
# soft cap and cannot swallow unrelated code.
_MAX_CALL_LINES = 40  # soft cap: how far a NEVER-closing opener is followed
_MAX_CALL_HARD_LINES = 200  # hard cap: how far a closing call is followed to bind it

# Cap a single rendered line. A short line is shown verbatim; a long (e.g.
# minified one-liner) line is shown as a bounded prefix plus a sha256 of the full
# line, so a packed payload cannot dump unbounded content into the evidence and
# baseline while a change past the cutoff still changes the digest and reopens the
# finding. The npm scanner bounds its snippets the same way.
_MAX_LINE_CHARS = 200
# Cap on recorded spans in one evidence string; beyond it the remaining spans are
# folded into a digest so a file with thousands of matching lines cannot build a
# multi-megabyte evidence blob, while an added/removed span past the cap still
# changes the key. Comfortably above the largest real baseline entry.
_MAX_EVIDENCE_SPANS = 96


def _cap_line(code: str) -> str:
    """Bound a single line's displayed code: return it verbatim when short, else a
    ``_MAX_LINE_CHARS`` prefix plus a digest of the whole line so the tail is still
    pinned (fail-closed) without recording the entire line."""
    if len(code) <= _MAX_LINE_CHARS:
        return code
    digest = hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()
    return f"{code[:_MAX_LINE_CHARS]} sha256:{digest}"


_PY_TRIPLE = ("'''", '"""')


def _ends_with_odd_backslash(s: str) -> bool:
    """True if ``s`` ends with an odd run of backslashes, i.e. a trailing
    backslash that escapes the newline (a string/line continuation) rather than a
    literal ``\\\\`` pair."""
    return (len(s) - len(s.rstrip("\\"))) % 2 == 1


# Single-line quoted string literal; blanks complete one-line strings (the legacy
# view) so the single-line and multi-line blanked spans can be unioned below.
_RE_STR_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def _blank_code_strings(lines: list[str]) -> list[str]:
    """Replace string contents (single- and triple-quoted, escapes honoured) with
    spaces across ``lines``, keeping the line count and every bracket OUTSIDE a
    string intact. Bracket counting then never miscounts a ``)`` that lives inside
    a string -- including a triple-quoted string spanning several lines, which a
    per-line regex cannot blank."""
    out: list[str] = []
    in_triple: str | None = None  # active ''' or \"\"\" delimiter, or None
    in_string: str | None = None  # active ' or " continued via a trailing backslash
    for line in lines:
        buf: list[str] = []
        i, n = 0, len(line)
        while i < n:
            if in_triple is not None:
                end = line.find(in_triple, i)
                if end == -1:
                    buf.append(" " * (n - i))
                    i = n
                else:
                    buf.append(" " * (end - i + 3))
                    i = end + 3
                    in_triple = None
                continue
            if in_string is not None:
                # A single-/double-quoted string continued onto this line by a
                # backslash-escaped newline. Resume blanking until its closing quote;
                # if this line also ends on an odd trailing backslash the string
                # continues again, otherwise it closes (or is unterminated) here. A
                # per-line regex blanker cannot see this, so a `)` on the
                # continuation line would otherwise be counted as code and close the
                # call early -- dropping the URL/body lines that follow.
                j, closed = i, False
                while j < n:
                    if line[j] == "\\":
                        j += 2
                        continue
                    if line[j] == in_string:
                        j += 1
                        closed = True
                        break
                    j += 1
                buf.append(" " * (min(j, n) - i))
                if closed:
                    in_string = None
                    i = j
                else:
                    i = n
                    if not _ends_with_odd_backslash(line):
                        in_string = None  # unterminated without continuation; stop
                continue
            ch = line[i]
            if ch in "'\"":
                if line[i : i + 3] in _PY_TRIPLE:
                    delim = line[i : i + 3]
                    end = line.find(delim, i + 3)
                    if end == -1:  # opens a triple string that runs past this line
                        buf.append(" " * (n - i))
                        in_triple = delim
                        i = n
                    else:
                        buf.append(" " * (end - i + 3))
                        i = end + 3
                    continue
                j = i + 1  # single-line string; skip to its closing quote
                closed = False
                while j < n:
                    if line[j] == "\\":
                        j += 2
                        continue
                    if line[j] == ch:
                        j += 1
                        closed = True
                        break
                    j += 1
                buf.append(" " * (min(j, n) - i))
                if closed:
                    i = j
                else:
                    # Ran off the line without closing: an odd trailing backslash
                    # escapes the newline and continues the string onto the next
                    # line, so remember the quote; otherwise it is just unterminated.
                    i = n
                    if _ends_with_odd_backslash(line):
                        in_string = ch
                continue
            buf.append(ch)
            i += 1
        out.append("".join(buf))
    return out


_RE_BRACKETS = re.compile(r"[()\[\]{}]")
_OPENERS = frozenset("([{")


def _bracket_lr(line: str) -> tuple[int, int]:
    """Order-aware bracket reduction of one already-string-blanked line: ``(L, R)``
    where ``L`` is the count of closers with no opener earlier on the line (they
    need an opener to the LEFT / a prior line) and ``R`` is the count of openers
    with no closer later on the line (they need a closer to the RIGHT / a later
    line). A plain net count (opens minus closes) collapses order and so masks a
    trailing opener that follows leading closers on the same line, e.g.
    ``]; requests.post(`` nets to 0 and hides the ``(`` that opens the flagged
    call; tracking the running minimum keeps that opener visible so the call's
    argument lines still bind. Only bracket characters are walked (pulled out with
    one C-level regex pass) so a long minified line stays cheap."""
    depth = 0
    low = 0
    for ch in _RE_BRACKETS.findall(line):
        if ch in _OPENERS:
            depth += 1
        else:
            depth -= 1
            if depth < low:
                low = depth
    return -low, depth - low


def _scan_line_end(view: list[str], start: int) -> int:
    """1-based line where the statement at ``start`` closes its brackets in
    ``view`` (one blanked view of the file). A call that closes is followed to its
    real close up to ``_MAX_CALL_HARD_LINES`` so its whole argument list binds; a
    bracket that never closes within that hard limit (a stray/miscounted opener) is
    bound only to the ``_MAX_CALL_LINES`` soft cap so it cannot swallow the file.
    Brackets are applied in order via ``_bracket_lr`` (leading closers clamp at 0)
    so a closer that precedes the opener on the same line does not cancel it."""
    depth = 0
    hard = min(len(view), start + _MAX_CALL_HARD_LINES - 1)
    for j in range(start, hard + 1):
        ln = view[j - 1]
        left, right = _bracket_lr(ln)
        depth = max(0, depth - left) + right
        if ln.rstrip().endswith("\\"):
            continue  # explicit backslash continuation: the call (e.g. its `(` and
            # URL/body) is on the next physical line, so do not close here
        if depth <= 0:
            return j
    # Never closed within the hard limit: bind only the soft cap so a stray opener
    # cannot bind a giant unrelated span.
    return min(len(view), start + _MAX_CALL_LINES - 1)


def _logical_line_end(sl_blanked: list[str], ml_blanked: list[str], start: int) -> int:
    """1-based line where the statement opened at ``start`` closes, so a multi-line
    call binds its argument lines (a changed URL/body on a continuation line
    reopens, not just the API line). Returns the LARGER of the spans found in the
    single-line-blanked view (legacy: a payload embedded inside a string still
    counts, so its brackets bind the call) and the multi-line-blanked view (a
    bracket inside a triple-quoted string argument no longer closes the call
    early). Taking the union never shrinks the bound span below either view, so
    neither blanking strategy can drop a continuation line a malicious change
    relies on."""
    return max(_scan_line_end(sl_blanked, start), _scan_line_end(ml_blanked, start))


def _extract_evidence(
    content: str,
    pattern: re.Pattern,
    max_matches: int = 0,
) -> str:
    """Pull matching lines as evidence snippets (``max_matches=0`` means all).

    Records every matching line in full, not a truncated sample, so an extra
    match (or extra code on a long line) appended to an already-flagged file
    changes the evidence and the baseline key instead of riding the first few.
    Leading whitespace is kept so a flagged line moved out of a guarded block
    reads as changed. Each single-line match is extended over bracket
    continuations so a multi-line call binds its argument lines too. Cross-line
    matches the per-line scan cannot see (DOTALL IOC regexes, or a multi-line
    construct appended under a check that already had a one-line match) are
    recorded afterwards, so an added multiline payload reopens the finding. A
    pathological greedy span is bounded to its head line plus a digest of the
    rest.
    """
    # Patterns whose meaning depends on the rest of the file (exec/eval reached
    # through an alias of `builtins`) resolve against the whole content once,
    # before the line-by-line scan below can no longer see it.
    for_text = getattr(pattern, "for_text", None)
    if for_text is not None:
        pattern = for_text(content)
    # Some lines can only be adjudicated with the whole file in hand: a call
    # through a function alias (`run(payload)` for `from builtins import exec as
    # run`) and a call above the rebinding of its alias both read as ordinary
    # code on their own. Take the whole-file scan's line numbers as an extra
    # source alongside the per-line search rather than a replacement for it, so
    # the evidence an already-flagged file records only ever grows.
    hit_lines = getattr(pattern, "hit_lines", None)
    rows = hit_lines(content) if hit_lines is not None else ()
    lines = content.splitlines()
    sl_blanked = [_RE_STR_LITERAL.sub("", ln) for ln in lines]
    ml_blanked = _blank_code_strings(lines)
    out = []
    seen: set[tuple[int, int]] = set()
    # Overflow is streamed, not buffered: once `out` holds _MAX_EVIDENCE_SPANS
    # rendered spans, every further span is folded straight into a running digest
    # instead of being materialized and sliced off at the end. On a minified or
    # padded file with hundreds of thousands of matching lines that keeps memory
    # and work bounded to the display cap rather than the match count, while the
    # digest still covers every overflow span so an over-cap payload change
    # reopens. The fold reproduces _canon_evidence(" | ".join(overflow)) exactly
    # (strip each span to its non-empty L<NN>-less code lines, join with "\n"), so
    # the digest is identical to buffering the whole list and canonicalizing once.
    overflow_count = 0
    overflow_hash = hashlib.sha256()
    overflow_started = False

    def _emit(rendered: str) -> None:
        nonlocal overflow_count, overflow_started
        if len(out) < _MAX_EVIDENCE_SPANS:
            out.append(rendered)
            return
        overflow_count += 1
        for piece in _RE_EVIDENCE_SPLIT.split(rendered):
            piece = _RE_EVIDENCE_PREFIX.sub("", piece, count = 1).rstrip()
            if not piece:
                continue
            if overflow_started:
                overflow_hash.update(b"\n")
            overflow_hash.update(piece.encode("utf-8", "replace"))
            overflow_started = True

    def _render(start: int, end: int) -> str:
        span = lines[start - 1 : end] or ["<multiline match>"]
        if len(span) > _MAX_MULTILINE_LINES:
            # Digest the code without the L<NN>: markers so a pure line shift of
            # the same span stays stable while a code change still reopens. The
            # head is truncated for display only; the span digest already binds
            # its full content, so no per-line digest is needed here.
            code = "\n".join(ln.rstrip() for ln in span)
            digest = hashlib.sha256(code.encode("utf-8", "replace")).hexdigest()
            head = span[0].rstrip()
            if len(head) > _MAX_LINE_CHARS:
                head = head[:_MAX_LINE_CHARS] + "..."
            return f"L{start}: {head} sha256:{digest}"
        return "\n".join(f"L{start + i}: {_cap_line(ln.rstrip())}" for i, ln in enumerate(span))

    for i, line in enumerate(lines, 1):
        if i in rows or pattern.search(line):
            span = (i, _logical_line_end(sl_blanked, ml_blanked, i))
            if span in seen:
                continue
            # Only track spans while still filling the display list: past the cap
            # every span is folded into the overflow digest, so growing `seen` with
            # all of them would keep memory proportional to the match count (the
            # behavior this cap exists to bound) on a generated file with millions
            # of one-line matches. The per-line spans are unique by line number, so
            # dropping them from `seen` past the cap cannot cause a missed dedup
            # here; at worst the fallback re-folds an over-cap span into the same
            # digest, which stays deterministic and still reopens on a change.
            if len(out) < _MAX_EVIDENCE_SPANS:
                seen.add(span)
            _emit(_render(*span))
            if max_matches and len(out) >= max_matches:
                return " | ".join(out)

    # Precompute newline offsets once so mapping a match offset to its 1-based line
    # is O(log n) (bisect) rather than O(n) (content.count) per match; the latter
    # made this fallback quadratic on a minified file with thousands of matches.
    nl = [p for p, ch in enumerate(content) if ch == "\n"]
    for m in pattern.finditer(content):
        start = bisect.bisect_left(nl, m.start()) + 1
        end = bisect.bisect_left(nl, m.end()) + 1
        if end <= start or (start, end) in seen:
            continue  # single-line matches are already covered by the pass above
        # A giant greedy DOTALL span is bound by the full digest of its content
        # (via _render, which renders a >12-line span as a head line plus a sha256
        # of the whole span). Binding only the anchors leaves the bridged interior
        # unhashed, so an attacker could insert a new cross-line payload (a `/tmp`
        # line and a later `subprocess` line, sharing no single line so the
        # per-line pass never binds them) between unchanged outer anchors and keep
        # the same key. Digesting the interior reopens on any such change; a pure
        # line shift stays stable because the digest is over the markerless code.
        if len(out) < _MAX_EVIDENCE_SPANS:
            seen.add((start, end))
        _emit(_render(start, end))
        if max_matches and len(out) >= max_matches:
            break
    if overflow_count:
        # The overflow digest was accumulated from the canonicalized (L<NN>:-less)
        # spans as they were emitted, so a pure line shift above the overflow
        # region does not change it and reopen an otherwise-unchanged finding,
        # matching the per-span key's line-shift stability.
        out.append(f"(+{overflow_count} more) sha256:{overflow_hash.hexdigest()}")
    return " | ".join(out)


def _embedded_key_evidence(content: str) -> str:
    """Key evidence that also pins the full PEM block(s) via a digest, so a key
    body swapped under the same BEGIN marker reopens the finding (single-line and
    DER keys are already bound by their full matched line)."""
    ev = _extract_evidence(content, RE_EMBEDDED_KEYS)
    blocks = RE_PEM_BLOCK.findall(content)
    if blocks:
        digest = hashlib.sha256("\n".join(blocks).encode("utf-8", "replace")).hexdigest()
        ev = f"{ev} sha256:{digest}" if ev else f"sha256:{digest}"
    return ev


def _blob_digest(content: str) -> tuple[str, str]:
    """First large blob (for display) plus a digest binding EVERY large blob, so
    an appended or swapped encoded payload reopens the finding rather than riding
    an unchanged first blob. Assumes at least one blob is present (single-blob
    files keep the prior single-blob digest, so the baseline does not drift)."""
    blobs = RE_LARGE_BLOB.findall(content)
    digest = hashlib.sha256("\n".join(blobs).encode("utf-8", "replace")).hexdigest()
    return blobs[0], digest


# Non-Python checkers
# Recent PyPI compromises (Lightning 2.6.x, ForceMemo) carried the payload in a
# bundled .js / .sh / workflow yaml so the Python imports looked clean. These
# checkers scan those file types when they appear inside a wheel/sdist.


def check_js_file(content: str, filename: str, package: str) -> list[Finding]:
    """Run JS-side checks. Triggered by .js / .mjs / .cjs / .ts."""
    findings = []

    # A >100 KB JS file inside a Python wheel is anomalous: CRITICAL combined
    # with any other JS heuristic, HIGH standalone.
    is_large = len(content) > 100 * 1024
    has_obf = bool(RE_JS_OBFUSCATION.search(content))
    has_web3 = bool(RE_WEB3_HIJACK.search(content))
    has_token_regex = bool(RE_TOKEN_REGEX.search(content))
    has_workflow_inj = bool(RE_WORKFLOW_INJECT.search(content))
    has_network = bool(RE_NETWORK.search(content))

    if has_obf:
        sev = CRITICAL if (is_large or has_web3 or has_token_regex) else HIGH
        findings.append(
            Finding(
                sev,
                package,
                filename,
                "JS minifier-style hex-var obfuscation (npm-payload signature)",
                _extract_evidence(content, RE_JS_OBFUSCATION),
            )
        )
    if has_web3:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "JS Web3 / wallet hijack (window.ethereum or fetch override)",
                _extract_evidence(content, RE_WEB3_HIJACK),
            )
        )
    if has_token_regex and has_network:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "JS embeds credential regexes AND makes network calls (stealer)",
                f"Token: {_extract_evidence(content, RE_TOKEN_REGEX)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )
    if has_workflow_inj:
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "JS self-propagation: workflow injection / repo takeover signature",
                _extract_evidence(content, RE_WORKFLOW_INJECT),
            )
        )
    # Pin the whole file's content digest to EVERY JS finding (not just large
    # bundles). _extract_evidence blanks only Python string forms before counting
    # brackets, so a JS backtick template literal that contains `)` can close a
    # call's span early and omit the option/body lines that follow; binding the
    # full content means a change to those omitted lines still reopens instead of
    # riding the matched-line evidence. A large bundle with no other heuristic is a
    # standalone HIGH.
    if findings or is_large:
        digest = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
        if findings:
            for f in findings:
                f.evidence = f"{f.evidence} bundle-sha256:{digest}"
        else:
            findings.append(
                Finding(
                    HIGH,
                    package,
                    filename,
                    # Size stays out of the check label (from main) so the baseline
                    # key does not drift when a benign bundle grows; the full-content
                    # digest below still binds the bytes so a payload swap reopens.
                    "Python wheel ships large JS bundle (uncommon; manually review)",
                    f"sha256: {digest}",
                )
            )
    return findings


def check_shell_file(content: str, filename: str, package: str) -> list[Finding]:
    """Run shell-side checks. Triggered by .sh / .bash / install scripts."""
    findings = []
    if RE_SHELL_DROPPER.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Shell pipes remote code into an interpreter (curl|sh dropper)",
                _extract_evidence(content, RE_SHELL_DROPPER),
            )
        )
    if RE_DEV_TOOL_HIJACK.search(content) and (
        RE_NETWORK.search(content) or RE_SUBPROCESS.search(content)
    ):
        # Bind the hook AND the network/exec signal so a changed exfil reopens.
        evidence = [f"Hook: {_extract_evidence(content, RE_DEV_TOOL_HIJACK)}"]
        if RE_NETWORK.search(content):
            evidence.append(f"Network: {_extract_evidence(content, RE_NETWORK)}")
        if RE_SUBPROCESS.search(content):
            evidence.append(f"Exec: {_extract_evidence(content, RE_SUBPROCESS)}")
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Shell installs developer-tool persistence hook (.bashrc / "
                "profile.d / vscode tasks) AND has network or exec",
                "\n".join(evidence),
            )
        )
    if RE_TOKEN_REGEX.search(content) and RE_NETWORK.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Shell embeds credential regexes AND makes network calls",
                f"Token: {_extract_evidence(content, RE_TOKEN_REGEX)}\n"
                f"Network: {_extract_evidence(content, RE_NETWORK)}",
            )
        )
    if RE_WORKFLOW_INJECT.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Shell self-propagation: workflow injection / repo takeover signature",
                _extract_evidence(content, RE_WORKFLOW_INJECT),
            )
        )
    if RE_MAY12_IOC.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "May-12 Shai-Hulud IOC string present in shell script",
                _extract_evidence(content, RE_MAY12_IOC),
            )
        )
    return findings


def check_workflow_file(content: str, filename: str, package: str) -> list[Finding]:
    """Run GitHub-Actions workflow checks. Triggered by .github/workflows/*.yml."""
    findings = []
    # A workflow file inside a PyPI package is suspicious (Shai-Hulud plants
    # `shai-hulud.yml` everywhere); injection-signature matches are CRITICAL.
    if RE_WORKFLOW_INJECT.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Workflow file inside PyPI package matches self-propagation signature",
                _extract_evidence(content, RE_WORKFLOW_INJECT),
            )
        )
    if RE_TOKEN_REGEX.search(content):
        findings.append(
            Finding(
                HIGH,
                package,
                filename,
                "Workflow file embeds credential regexes (token harvesting?)",
                _extract_evidence(content, RE_TOKEN_REGEX),
            )
        )
    if RE_SHELL_DROPPER.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "Workflow pipes remote code into a shell (curl|sh dropper)",
                _extract_evidence(content, RE_SHELL_DROPPER),
            )
        )
    if RE_MAY12_IOC.search(content):
        findings.append(
            Finding(
                CRITICAL,
                package,
                filename,
                "May-12 Shai-Hulud IOC string present in workflow file",
                _extract_evidence(content, RE_MAY12_IOC),
            )
        )
    return findings


# Archive handling

# Tarbomb caps, mirrored from scripts/scan_npm_packages.py::safe_extract.
# Refuses zip/tar-of-death so a hostile archive cannot exhaust memory before
# scanning. Keep in sync with the npm side; duplicated to stay standalone.
HARD_MAX_FILE_BYTES = 64 * 1024 * 1024  # 64 MiB per member
HARD_MAX_TOTAL_BYTES = 512 * 1024 * 1024  # 512 MiB cumulative
HARD_MAX_MEMBERS = 50_000  # entries per archive


def _refuse_unsafe_member_name(name: str) -> str | None:
    """Return a refusal reason for a member name, or None if safe.

    Mirrors `safe_extract`: no absolute paths, no `..` traversal. We never write
    to disk, so the name-shape check plus the in-memory size cap is sufficient.
    """
    if name.startswith("/") or ".." in Path(name).parts:
        return f"unsafe member name {name!r}"
    return None


def iter_archive_files(archive_path: str):
    """Yield (filename, text_content) for every file in a wheel/sdist.

    Streams members with per-member size + count caps so a tarbomb/zipbomb can't
    blow the memory budget. On cap breach, emits a `[WARN]` and short-circuits.
    """
    path = Path(archive_path)

    if path.suffix == ".whl" or path.suffix == ".zip":
        total = 0
        count = 0
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                count += 1
                if count > HARD_MAX_MEMBERS:
                    print(
                        f"  [WARN] {path.name}: refused; member count "
                        f"{count} exceeds cap {HARD_MAX_MEMBERS}",
                        file = sys.stderr,
                    )
                    return
                reason = _refuse_unsafe_member_name(info.filename)
                if reason is not None:
                    print(
                        f"  [WARN] {path.name}: refused member ({reason})",
                        file = sys.stderr,
                    )
                    continue
                # Declared (uncompressed) size cap
                if info.file_size > HARD_MAX_FILE_BYTES:
                    print(
                        f"  [WARN] {path.name}: skipped {info.filename!r} "
                        f"(declared {info.file_size} > cap {HARD_MAX_FILE_BYTES})",
                        file = sys.stderr,
                    )
                    continue
                if total + info.file_size > HARD_MAX_TOTAL_BYTES:
                    print(
                        f"  [WARN] {path.name}: cumulative bytes cap "
                        f"{HARD_MAX_TOTAL_BYTES} hit at {info.filename!r}",
                        file = sys.stderr,
                    )
                    return
                try:
                    data = zf.read(info.filename)
                    total += len(data)
                    text = data.decode("utf-8", errors = "replace")
                    yield info.filename, text
                except Exception:
                    continue

    elif path.name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar")):
        total = 0
        count = 0
        # Streaming open so we never read the whole archive into memory.
        with tarfile.open(path, mode = "r|*") as tf:
            for member in tf:
                count += 1
                if count > HARD_MAX_MEMBERS:
                    print(
                        f"  [WARN] {path.name}: refused; member count "
                        f"{count} exceeds cap {HARD_MAX_MEMBERS}",
                        file = sys.stderr,
                    )
                    return
                # Refuse symlinks/hardlinks/devices: tar parsers have
                # historically dereferenced them on extract.
                if member.issym() or member.islnk():
                    print(
                        f"  [WARN] {path.name}: refused link member {member.name!r}",
                        file = sys.stderr,
                    )
                    continue
                if member.isdev() or member.isfifo():
                    print(
                        f"  [WARN] {path.name}: refused special member {member.name!r}",
                        file = sys.stderr,
                    )
                    continue
                if not member.isfile():
                    continue
                reason = _refuse_unsafe_member_name(member.name)
                if reason is not None:
                    print(
                        f"  [WARN] {path.name}: refused member ({reason})",
                        file = sys.stderr,
                    )
                    continue
                declared = max(member.size, 0)
                if declared > HARD_MAX_FILE_BYTES:
                    print(
                        f"  [WARN] {path.name}: skipped {member.name!r} "
                        f"(declared {declared} > cap {HARD_MAX_FILE_BYTES})",
                        file = sys.stderr,
                    )
                    continue
                if total + declared > HARD_MAX_TOTAL_BYTES:
                    print(
                        f"  [WARN] {path.name}: cumulative bytes cap "
                        f"{HARD_MAX_TOTAL_BYTES} hit at {member.name!r}",
                        file = sys.stderr,
                    )
                    return
                try:
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    # Bound the read: a tar header may lie about size
                    data = f.read(HARD_MAX_FILE_BYTES + 1)
                    if len(data) > HARD_MAX_FILE_BYTES:
                        print(
                            f"  [WARN] {path.name}: body of {member.name!r} exceeded declared cap",
                            file = sys.stderr,
                        )
                        continue
                    total += len(data)
                    text = data.decode("utf-8", errors = "replace")
                    yield member.name, text
                except Exception:
                    continue
    else:
        print(f"  [WARN] Unknown archive format: {path.name}", file = sys.stderr)


def scan_archive(archive_path: str, package: str) -> list[Finding]:
    """Scan all files in an archive for malicious patterns.

    A corrupted archive container (truncated wheel, bad gzip header, etc.) emits
    a CRITICAL ``archive_corrupted`` finding rather than being silently skipped
    and reported as "0 findings" (silent-failure hardening SF1).
    """
    findings: list[Finding] = []
    try:
        for filename, content in iter_archive_files(archive_path):
            lower = filename.lower()
            if lower.endswith(".pth"):
                findings.extend(check_pth_file(content, filename, package))
            elif lower.endswith(".py"):
                findings.extend(check_py_file(content, filename, package))
            elif lower.endswith((".js", ".mjs", ".cjs", ".ts")):
                # Lightning 2.6.x hid its payload in a 14.8 MB router_runtime.js;
                # without this branch we'd only see the small Python loader.
                findings.extend(check_js_file(content, filename, package))
            elif lower.endswith((".sh", ".bash")):
                findings.extend(check_shell_file(content, filename, package))
            elif "/.github/workflows/" in lower and lower.endswith((".yml", ".yaml")):
                # Shai-Hulud/ForceMemo plant their own GHA workflow
                findings.extend(check_workflow_file(content, filename, package))
    except (zipfile.BadZipFile, tarfile.TarError, EOFError, OSError) as exc:
        # Archive cannot be opened / is structurally broken: either transport
        # corruption or a deliberate attempt to bypass error-swallowing scanners.
        findings.append(
            Finding(
                CRITICAL,
                package,
                os.path.basename(archive_path),
                "archive_corrupted",
                f"{type(exc).__name__}: {exc}"[:240],
            )
        )
    return findings


# Download packages


_RE_PYPI_SPEC_VERSION = re.compile(r"==\s*([A-Za-z0-9_.\-+!]+)")


def _check_blocked_pypi_versions(specs: list[str]) -> tuple[list[str], list[Finding]]:
    """Filter ``specs`` against ``BLOCKED_PYPI_VERSIONS``.

    Returns ``(safe_specs, findings)``. Each blocked spec emits a CRITICAL
    ``Finding`` and is dropped so the malicious tarball is never fetched. Specs
    without an ``==X.Y.Z`` pin pass through; the IOC regexes catch them later.
    """
    safe: list[str] = []
    findings: list[Finding] = []
    for spec in specs:
        name = _extract_pkg_name(spec).lower()
        blocked = BLOCKED_PYPI_VERSIONS.get(name, set())
        if not blocked:
            safe.append(spec)
            continue
        m = _RE_PYPI_SPEC_VERSION.search(spec)
        version = m.group(1) if m else None
        if version is not None and version in blocked:
            findings.append(
                Finding(
                    CRITICAL,
                    f"{name}=={version}",
                    "<spec>",
                    "blocked-known-malicious",
                    f"{name}=={version} is on the BLOCKED_PYPI_VERSIONS list",
                )
            )
            # Drop the spec; do not download
            continue
        safe.append(spec)
    return safe, findings


def _pip_download_env() -> dict[str, str]:
    """Return a scrubbed environment for invoking `pip download`.

    Strips every PIP_* override and forces the resolver at PyPI; PIP_CONFIG_FILE
    is /dev/null so a stray pip.conf extra-index-url cannot bypass the pin.
    """
    env = {**os.environ}
    # Drop any user override
    for key in [k for k in env if k.startswith("PIP_")]:
        env.pop(key, None)
    env["PIP_INDEX_URL"] = "https://pypi.org/simple"
    env["PIP_EXTRA_INDEX_URL"] = ""
    env["PIP_CONFIG_FILE"] = "/dev/null"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


# Pip resolver flags shared by both download branches. CLI index-URL pin is
# belt + braces with the env scrub; `--only-binary :all:` avoids running setup.py.
_PIP_DOWNLOAD_PIN_FLAGS = [
    "--index-url",
    "https://pypi.org/simple",
    "--only-binary",
    ":all:",
]


# Strip characters that could escape `dest` via `os.path.join`, so a spec like
# `../../etc/foo==1.0` cannot land outside the temp tree.
_RE_PKG_NAME_SANITIZE = re.compile(r"[^A-Za-z0-9._-]")


# sdist fallback. `--only-binary :all:` never builds an sdist (no setup.py
# exec), but a wheel-less project then can't be fetched at all and one such
# package fails the whole --with-deps resolve (exit 2) -- a coverage hole. So on
# resolve failure we drop to per-spec and fetch any sdist-only package's raw
# tarball from the PyPI JSON API for scan_archive() to read statically: no pip,
# no build, same no-exec guarantee. Transport failures are still exit 2; only
# "no wheel" is downgraded to a direct fetch.

# How many levels of indirect-dep recovery to chase (a wheel dep whose own child
# is sdist-only, and so on). Bounded with dedup so recovery always terminates.
_MAX_DEP_FOLLOWUP_DEPTH = 2
_SDIST_DOWNLOAD_TIMEOUT = 180
# Never fetch an archive larger than we would be willing to scan (iter_archive_files cap).
_MAX_SDIST_BYTES = HARD_MAX_TOTAL_BYTES
# Direct sdist bytes only ever come from PyPI's own CDN; refuse anything else.
_TRUSTED_PYPI_HOSTS = frozenset({"files.pythonhosted.org", "pypi.org", "pypi.python.org"})


def _spec_pin_version(spec: str) -> str | None:
    """Return the ``==X.Y.Z`` pin from a spec, or None if unpinned."""
    m = _RE_PYPI_SPEC_VERSION.search(spec)
    return m.group(1) if m else None


def _pypi_json(name: str, version: str | None = None) -> dict | None:
    """Fetch PyPI metadata JSON (read-only HTTPS GET, no exec); None on error.
    With ``version`` it fetches that release's document, whose ``requires_dist``
    is accurate for the pin (the project-level doc describes only the latest)."""
    url = "https://pypi.org/pypi/" + urllib.parse.quote(name, safe = "")
    if version:
        url += "/" + urllib.parse.quote(version, safe = "")
    url += "/json"
    try:
        req = urllib.request.Request(url, headers = {"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout = 30) as resp:
            if getattr(resp, "status", 200) != 200:
                return None
            data = resp.read(16 * 1024 * 1024)  # metadata is small; cap regardless
        return json.loads(data.decode("utf-8", errors = "replace"))
    except Exception:
        return None


def _release_files(meta: dict, version: str | None) -> list[dict]:
    """Files for a pinned version, else the latest release's. A pin that is
    absent or empty returns [] (never the latest) so a yanked/bad pin fails
    closed instead of a different artifact being scanned in its place."""
    if version is not None:
        return meta.get("releases", {}).get(version) or []
    return meta.get("urls", []) or []


def _release_has_wheel(meta: dict, version: str | None) -> bool:
    """True if the (pinned or latest) release publishes any bdist_wheel."""
    return any(f.get("packagetype") == "bdist_wheel" for f in _release_files(meta, version))


def _is_trusted_pypi_url(url: str) -> bool:
    """Only download sdist bytes from PyPI's own hosts, over HTTPS."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    return parsed.scheme == "https" and parsed.hostname in _TRUSTED_PYPI_HOSTS


_MARKER_ENV_VARS = (
    "sys_platform",
    "platform_system",
    "platform_machine",
    "platform_release",
    "platform_version",
    "platform_python_implementation",
    "os_name",
    "python_version",
    "python_full_version",
    "implementation_name",
    "implementation_version",
)


def _marker_holds_by_default(marker: str) -> bool:
    """Keep (scan) a dep unless its marker is purely ``extra``-gated. The scanner
    runs on one OS/Python but a package may be installed on another, so a marker
    that can be true on a different target (``sys_platform == 'win32'``,
    ``python_version == '3.13'``) is always kept; only a marker depending solely
    on ``extra`` and false with no extra requested is dropped. Conservative: on
    any uncertainty, keep (over-scan, never silently skip)."""
    m = marker.strip()
    if not m or "extra" not in m:
        return True  # no extra gate: installed by default on some target -> scan
    if any(v in m for v in _MARKER_ENV_VARS):
        return True  # also platform/python gated: true on some target -> scan
    # Pure extra marker: decide by evaluating with no extra requested.
    try:
        from packaging.markers import Marker, default_environment

        env = default_environment()
        env["extra"] = ""
        return bool(Marker(m).evaluate(env))
    except Exception:
        # packaging missing/unparseable: drop only a pure positive extra-equality.
        return re.fullmatch(r"\s*extra\s*==\s*['\"][^'\"]+['\"]\s*", m) is None


def _requires_dist_names(meta: dict) -> list[str]:
    """Transitive dep specs (name + version specifier) from metadata, to recover
    a sdist-only package's tree. The specifier is kept so a pinned malicious
    version is fetched, not latest. Drops deps whose marker cannot hold for a
    default install."""
    info = meta.get("info", {}) or {}
    reqs = info.get("requires_dist") or []
    specs: list[str] = []
    for r in reqs:
        if not isinstance(r, str):
            continue
        head = r
        if ";" in r:
            head, marker = r.split(";", 1)
            if not _marker_holds_by_default(marker):
                continue
        if not _RE_NAME.match(head.strip()):
            continue
        # "torch (>=1.10)" / "torch >=1.10" -> "torch>=1.10" (pip-friendly).
        specs.append(re.sub(r"\s+", "", head).replace("(", "").replace(")", ""))
    return specs


def _requires_dist_for(
    name: str,
    version: str | None,
    project_meta: dict,
    errors: list[str] | None = None,
) -> list[str]:
    """Declared deps for the pinned version, read from that release's metadata
    (its ``requires_dist`` can differ from latest). Unpinned uses the
    project-level (latest) document. A pinned version whose own metadata cannot
    be fetched returns [] (never latest's deps) and, when ``errors`` is given,
    records an incomplete-scan error so a partial tree is not read as "no deps"."""
    if not version:
        return _requires_dist_names(project_meta)
    vmeta = _pypi_json(name, version)
    if vmeta is None:
        msg = f"metadata fetch failed for pinned {name}=={version}; dependency scan incomplete"
        if errors is None:
            print(f"  [WARN] {msg}", file = sys.stderr)
        else:
            errors.append(msg)
        return []
    return _requires_dist_names(vmeta)


def _download_sdist_direct(
    name: str,
    version: str | None,
    dest: str,
    *,
    meta: dict | None = None,
) -> tuple[str | None, str | None]:
    """Fetch a project's sdist tarball directly from PyPI (no pip, no build).

    Returns ``(filepath, error)``, one non-None. Suffix preserved for the archive
    reader; bounded by ``_MAX_SDIST_BYTES`` and restricted to PyPI's CDN.
    """
    if meta is None:
        meta = _pypi_json(name)
    if meta is None:
        return None, f"PyPI metadata fetch failed for {name}"
    picked: tuple[str, str] | None = None
    for f in _release_files(meta, version):
        if f.get("packagetype") == "sdist" and f.get("url") and f.get("filename"):
            picked = (f["filename"], f["url"])
            break
    if picked is None:
        return None, f"no sdist published for {name} (version={version or 'latest'})"
    fname, url = picked
    if not _is_trusted_pypi_url(url):
        return None, f"refusing non-PyPI sdist URL for {name}: {url[:80]}"
    # basename + sanitize keeps the path inside dest; the char class preserves
    # the real `.tar.gz` / `.zip` suffix so the archive reader picks the format.
    safe_fname = _RE_PKG_NAME_SANITIZE.sub("_", os.path.basename(fname)) or "sdist.tar.gz"
    out = os.path.join(dest, safe_fname)
    try:
        req = urllib.request.Request(url, headers = {"Accept": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout = _SDIST_DOWNLOAD_TIMEOUT) as resp:
            if getattr(resp, "status", 200) != 200:
                return None, f"sdist HTTP {getattr(resp, 'status', '?')} for {name}"
            data = resp.read(_MAX_SDIST_BYTES + 1)
        if len(data) > _MAX_SDIST_BYTES:
            return None, f"sdist for {name} exceeds {_MAX_SDIST_BYTES} byte cap"
        with open(out, "wb") as fh:
            fh.write(data)
        print(
            f"  [INFO] fetched sdist directly (no build) for {name}: {safe_fname}",
            file = sys.stderr,
        )
        return out, None
    except Exception as exc:
        return None, f"sdist download failed for {name}: {type(exc).__name__}: {str(exc)[:120]}"


def _pip_download_with_deps(
    specs: list[str],
    dest: str,
    env: dict,
    *,
    timeout: int = 600,
) -> tuple[int, str]:
    """One `pip download --with-deps --only-binary :all:` call. Returns (rc, stderr)."""
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        *_PIP_DOWNLOAD_PIN_FLAGS,
        "--dest",
        dest,
    ] + list(specs)
    try:
        proc = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout, env = env)
        return proc.returncode, proc.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "pip download (with deps) timed out"


def _collect_flat_dir(dest: str, results: list[tuple[str, str]]) -> None:
    """Append every archive in a flat dest dir as (pkg_name, path)."""
    for fname in sorted(os.listdir(dest)):
        fpath = os.path.join(dest, fname)
        if os.path.isfile(fpath):
            pkg_name = fname.split("-")[0].replace("_", "-").lower()
            results.append((pkg_name, fpath))


def _resolve_per_spec_with_deps(
    specs: list[str], dest: str, env: dict, download_errors: list[str]
) -> None:
    """Fallback when the bulk --with-deps resolve fails: resolve each spec alone.

    A still-failing spec is probed against PyPI: sdist-only -> direct fetch (deps
    recovered one level); wheel-present but tree-unresolvable -> a --no-deps fetch
    of just that package. Only a genuine fetch failure errors (caller exits 2);
    unfetchable indirect deps are warned, since the named package is still scanned.
    """
    sdist_dep_followups: list[str] = []
    for spec in specs:
        name = _extract_pkg_name(spec)
        version = _spec_pin_version(spec)
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            *_PIP_DOWNLOAD_PIN_FLAGS,
            "--dest",
            dest,
            spec,
        ]
        try:
            proc = subprocess.run(cmd, capture_output = True, text = True, timeout = 300, env = env)
        except subprocess.TimeoutExpired:
            download_errors.append(f"per-spec --with-deps timed out for {spec}")
            continue
        if proc.returncode == 0:
            continue  # archives landed in dest; collected by the caller
        meta = _pypi_json(name)
        if meta is not None and not _release_has_wheel(meta, version):
            fpath, serr = _download_sdist_direct(name, version, dest, meta = meta)
            if fpath is None:
                download_errors.append(serr or f"sdist fetch failed for {name}")
                continue
            sdist_dep_followups.extend(_requires_dist_for(name, version, meta, download_errors))
            continue
        # Has a wheel but the full transitive tree won't co-resolve
        # (ResolutionImpossible) -- typically a package the requirement file
        # installs with --no-deps by design (e.g. descript-audio-codec, whose
        # own pins conflict). Fetch just the package itself with --no-deps so it
        # is still scanned; its conflicting deps are out of scope here (the file
        # excludes them on purpose). Only a genuine fetch failure is an error.
        nd_cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            *_PIP_DOWNLOAD_PIN_FLAGS,
            "--dest",
            dest,
            spec,
        ]
        try:
            nd = subprocess.run(nd_cmd, capture_output = True, text = True, timeout = 180, env = env)
        except subprocess.TimeoutExpired:
            download_errors.append(f"per-spec --no-deps timed out for {spec}")
            continue
        if nd.returncode == 0:
            print(
                f"  [INFO] {name}: full tree unresolvable; scanned the package "
                f"alone (--no-deps), recovering deps individually.",
                file = sys.stderr,
            )
            # The --with-deps failure may have been a sdist-only TRANSITIVE dep,
            # which --no-deps skips. Recover the declared deps so that class is
            # still scanned (each is fetched as a wheel or direct sdist below).
            if meta is not None:
                sdist_dep_followups.extend(_requires_dist_for(name, version, meta, download_errors))
            continue
        # --no-deps also failed: last-ditch sdist fetch at the pinned version.
        if meta is not None:
            fpath, _serr = _download_sdist_direct(name, version, dest, meta = meta)
            if fpath is not None:
                continue
        download_errors.append(
            f"per-spec failed for {spec} (with-deps and --no-deps): {nd.stderr.strip()[:240]}"
        )

    # Recover the transitive deps of sdist-only packages. A depth-bounded,
    # deduped worklist so a wheel dep whose own child is sdist-only is itself
    # fetched (--no-deps) and scanned -- not silently dropped -- and that child
    # is then recovered in turn. `dep` carries the version specifier so a pinned
    # version is fetched.
    seen: set[str] = set()
    worklist: list[tuple[str, int]] = [(d, 0) for d in sdist_dep_followups]
    while worklist:
        dep, depth = worklist.pop()
        dep_name = _extract_pkg_name(dep)
        key = _norm_pkg(dep_name)
        if key in seen:
            continue
        seen.add(key)
        dep_ver = _spec_pin_version(dep)
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            *_PIP_DOWNLOAD_PIN_FLAGS,
            "--dest",
            dest,
            dep,
        ]
        try:
            proc = subprocess.run(cmd, capture_output = True, text = True, timeout = 300, env = env)
        except subprocess.TimeoutExpired:
            print(f"  [WARN] dep download timed out for {dep}", file = sys.stderr)
            continue
        if proc.returncode == 0:
            continue
        meta = _pypi_json(dep_name)
        if meta is None:
            print(f"  [WARN] could not resolve indirect dep {dep}; skipping", file = sys.stderr)
            continue
        if not _release_has_wheel(meta, dep_ver):
            fpath, serr = _download_sdist_direct(dep_name, dep_ver, dest, meta = meta)
            if fpath is None:
                print(f"  [WARN] could not fetch sdist dep {dep}: {serr}", file = sys.stderr)
            elif depth < _MAX_DEP_FOLLOWUP_DEPTH:
                worklist.extend((d, depth + 1) for d in _requires_dist_for(dep_name, dep_ver, meta))
            continue
        # Wheel published but its tree won't co-resolve (a sdist-only child).
        # Fetch the dep alone so it is scanned, then chase its own declared deps.
        nd_cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            *_PIP_DOWNLOAD_PIN_FLAGS,
            "--dest",
            dest,
            dep,
        ]
        try:
            nd = subprocess.run(nd_cmd, capture_output = True, text = True, timeout = 180, env = env)
        except subprocess.TimeoutExpired:
            print(f"  [WARN] dep --no-deps timed out for {dep}", file = sys.stderr)
            continue
        if nd.returncode == 0:
            if depth < _MAX_DEP_FOLLOWUP_DEPTH:
                worklist.extend((d, depth + 1) for d in _requires_dist_for(dep_name, dep_ver, meta))
            continue
        fpath, _serr = _download_sdist_direct(dep_name, dep_ver, dest, meta = meta)
        if fpath is None:
            print(f"  [WARN] could not resolve indirect dep {dep}; skipping", file = sys.stderr)
        elif depth < _MAX_DEP_FOLLOWUP_DEPTH:
            worklist.extend((d, depth + 1) for d in _requires_dist_for(dep_name, dep_ver, meta))


def download_packages(
    specs: list[str],
    dest: str,
    *,
    with_deps: bool = False,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Download packages to dest using pip download. NEVER installs.

    Returns ``(results, download_errors)``: ``results`` is ``(spec_or_name,
    filepath)`` per archive; ``download_errors`` is one-line transport-failure
    summaries. A non-empty ``download_errors`` MUST make the caller exit
    non-zero so a partial scan can't masquerade as "0 findings, all clean".

    with_deps=True downloads the full transitive tree (flat dir); a bulk resolve
    failure (sdist-only package or version conflict) degrades to per-spec
    resolution + direct sdist fetch rather than blanking the shard.
    with_deps=False (default) downloads each spec individually with --no-deps,
    also falling back to a direct sdist fetch when no wheel exists.
    """
    results: list[tuple[str, str]] = []
    download_errors: list[str] = []
    env = _pip_download_env()

    if with_deps:
        os.makedirs(dest, exist_ok = True)
        # Fast path: resolve + download the whole transitive tree in one call.
        # `--only-binary :all:` refuses sdists so we never build for metadata.
        rc, stderr = _pip_download_with_deps(specs, dest, env)
        if rc != 0:
            # Atomic resolve failed -- a sdist-only package, or a cross-package
            # version conflict (ResolutionImpossible). Degrade to per-spec
            # resolution so one bad spec can't blank the shard, then direct-fetch
            # any sdist-only holdouts (no build). Genuine failures still record an
            # error so the caller exits 2.
            print(
                f"  [INFO] bulk --with-deps resolve failed "
                f"({stderr.strip()[:160]}); falling back to per-spec resolution "
                f"for {len(specs)} spec(s).",
                file = sys.stderr,
            )
            _resolve_per_spec_with_deps(specs, dest, env, download_errors)
        # Collect everything that landed (bulk OR per-spec OR direct sdist).
        _collect_flat_dir(dest, results)
    else:
        for spec in specs:
            raw_name = _extract_pkg_name(spec)
            # Sanitize before joining into `dest` to prevent path traversal
            safe_name = _RE_PKG_NAME_SANITIZE.sub("_", raw_name) or "_pkg"
            pkg_dir = os.path.join(dest, safe_name)
            os.makedirs(pkg_dir, exist_ok = True)
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--no-deps",
                *_PIP_DOWNLOAD_PIN_FLAGS,
                "--dest",
                pkg_dir,
                spec,
            ]
            try:
                proc = subprocess.run(cmd, capture_output = True, text = True, timeout = 120, env = env)
            except subprocess.TimeoutExpired:
                download_errors.append(f"pip download timed out for {spec}")
                continue
            if proc.returncode != 0:
                # No wheel? Direct-fetch the sdist (no build) before erroring.
                name = _extract_pkg_name(spec)
                version = _spec_pin_version(spec)
                meta = _pypi_json(name)
                if meta is not None and not _release_has_wheel(meta, version):
                    fpath, serr = _download_sdist_direct(name, version, pkg_dir, meta = meta)
                    if fpath is not None:
                        results.append((spec, fpath))
                        continue
                    download_errors.append(serr or f"sdist fetch failed for {name}")
                    continue
                download_errors.append(
                    f"pip download failed for {spec}: {proc.stderr.strip()[:300]}"
                )
                continue

            for fname in os.listdir(pkg_dir):
                fpath = os.path.join(pkg_dir, fname)
                if os.path.isfile(fpath):
                    results.append((spec, fpath))
    return results, download_errors


# Parse requirements files

_RE_NAME = re.compile(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)")


def _extract_pkg_name(spec: str) -> str:
    """Extract the package name from a pip spec string."""
    m = _RE_NAME.match(spec)
    return (
        m.group(1) if m else spec.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip()
    )


def parse_requirements(req_files: list[str]) -> list[dict]:
    """Parse requirements files into a list of dicts with source tracking.

    Each dict has keys: spec, name, source_file, line_num, raw_line, is_git.
    """
    results = []
    for req_file in req_files:
        abs_path = os.path.abspath(req_file)
        try:
            with open(req_file) as f:
                for line_num, raw_line in enumerate(f, 1):
                    line = raw_line.strip()
                    # Skip blanks, comments, options, nested -r
                    if not line or line.startswith("#") or line.startswith("-"):
                        continue
                    is_git = line.startswith("git+") or "git+" in line.split("#")[0]
                    # Strip inline comments and env markers
                    spec = line.split("#")[0].strip()
                    spec = spec.split(";")[0].strip()
                    if not spec:
                        continue
                    name = _extract_pkg_name(spec) if not is_git else spec
                    results.append(
                        {
                            "spec": spec,
                            "name": name,
                            "source_file": abs_path,
                            "line_num": line_num,
                            "raw_line": raw_line.rstrip("\n"),
                            "is_git": is_git,
                        }
                    )
        except FileNotFoundError:
            print(f"  [ERROR] Requirements file not found: {req_file}", file = sys.stderr)
    return results


def get_downloaded_version(archive_path: str) -> str | None:
    """Extract version from wheel/sdist filename.

    Wheel: {name}-{version}(-...).whl
    Sdist: {name}-{version}.tar.gz / .zip
    """
    basename = os.path.basename(archive_path)
    # Wheel: name-version-pytag-abitag-platform.whl
    if basename.endswith(".whl"):
        parts = basename[:-4].split("-")
        if len(parts) >= 2:
            return parts[1]
    # Sdist: name-version.<ext>
    for ext in (".tar.gz", ".tar.bz2", ".tar.xz", ".tar", ".zip"):
        if basename.endswith(ext):
            stem = basename[: -len(ext)]
            parts = stem.rsplit("-", 1)
            if len(parts) == 2:
                return parts[1]
    return None


# Display


def severity_color(sev: str) -> str:
    colors = {CRITICAL: "\033[91m", HIGH: "\033[93m", MEDIUM: "\033[33m"}
    return colors.get(sev, "")


RESET = "\033[0m"


def print_findings(findings: list[Finding]) -> None:
    if not findings:
        print("\n  All clean. No suspicious patterns found.")
        return

    findings.sort(key = lambda f: SEVERITY_ORDER.get(f.severity, 99))

    print(f"\n  {'=' * 72}")
    print(f"  SCAN RESULTS: {len(findings)} finding(s)")
    print(f"  {'=' * 72}")

    for i, f in enumerate(findings, 1):
        color = severity_color(f.severity)
        print(f"\n  [{i}] {color}{f.severity}{RESET}  {f.check}")
        print(f"      Package:  {f.package}")
        print(f"      File:     {f.filename}")
        if f.evidence:
            for eline in f.evidence.split("\n"):
                print(f"      Evidence: {eline}")

    print(f"\n  {'=' * 72}")
    crits = sum(1 for f in findings if f.severity == CRITICAL)
    highs = sum(1 for f in findings if f.severity == HIGH)
    meds = sum(1 for f in findings if f.severity == MEDIUM)
    parts = []
    if crits:
        parts.append(f"{crits} CRITICAL")
    if highs:
        parts.append(f"{highs} HIGH")
    if meds:
        parts.append(f"{meds} MEDIUM")
    print(f"  Summary: {', '.join(parts)}")


# PyPI version queries and --fix logic


def version_sort_key(v: str) -> tuple:
    """PEP 440-ish sort key using stdlib only.

    Handles: epoch!, major.minor.patch, pre/post/dev suffixes.
    Returns a tuple that sorts in ascending version order.
    """
    epoch = 0
    if "!" in v:
        epoch_str, v = v.split("!", 1)
        try:
            epoch = int(epoch_str)
        except ValueError:
            pass

    # Split off pre/post/dev suffixes
    v_clean = re.split(
        r"[-_.]?(a|alpha|b|beta|rc|c|pre|preview|dev|post)", v, maxsplit = 1, flags = re.I
    )
    base = v_clean[0]
    suffix = v[len(base) :]

    parts = []
    for seg in base.split("."):
        try:
            parts.append(int(seg))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:  # pad to at least 3 parts
        parts.append(0)

    # Suffix ordering: dev < alpha < beta < rc < (none) < post
    suffix_lower = suffix.lower().lstrip(".-_")
    if suffix_lower.startswith("dev"):
        suffix_rank = -4
    elif suffix_lower.startswith(("a", "alpha")):
        suffix_rank = -3
    elif suffix_lower.startswith(("b", "beta")):
        suffix_rank = -2
    elif suffix_lower.startswith(("rc", "c", "pre", "preview")):
        suffix_rank = -1
    elif suffix_lower.startswith("post"):
        suffix_rank = 1
    else:
        suffix_rank = 0  # stable

    return (epoch, tuple(parts), suffix_rank, suffix)


def fetch_pypi_versions(name: str) -> list[str]:
    """Fetch all available versions for a package from PyPI JSON API.

    Returns versions sorted ascending by version_sort_key.
    """
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        req = urllib.request.Request(url, headers = {"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout = 30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [ERROR] Failed to query PyPI for {name}: {e}", file = sys.stderr)
        return []

    versions = list(data.get("releases", {}).keys())
    versions.sort(key = version_sort_key)
    return versions


def find_safe_version(
    name: str,
    bad_ver: str,
    tmpdir: str,
    max_search: int = 10,
) -> str | None:
    """Search backward from bad_ver for a clean version.

    Downloads and scans up to max_search older versions.
    Returns the first clean version found, or None.
    """
    versions = fetch_pypi_versions(name)
    if not versions:
        print(f"  [WARN] No versions found on PyPI for {name}", file = sys.stderr)
        return None

    try:
        bad_idx = versions.index(bad_ver)
    except ValueError:
        # bad_ver may resolve to a different string; search by sort key
        bad_key = version_sort_key(bad_ver)
        bad_idx = None
        for i, v in enumerate(versions):
            if version_sort_key(v) >= bad_key:
                bad_idx = i
                break
        if bad_idx is None:
            bad_idx = len(versions) - 1

    # Search backward from the version before bad_ver
    candidates = versions[:bad_idx]
    candidates.reverse()  # newest-first among older versions
    candidates = candidates[:max_search]

    if not candidates:
        print(f"  [WARN] No older versions to scan for {name}", file = sys.stderr)
        return None

    print(f"  Searching {len(candidates)} older version(s) of {name}...")

    for ver in candidates:
        spec = f"{name}=={ver}"
        scan_dir = os.path.join(tmpdir, f"{name}_{ver}")
        os.makedirs(scan_dir, exist_ok = True)

        downloaded, download_errors = download_packages([spec], scan_dir)
        if not downloaded:
            for err in download_errors:
                print(f"    [WARN] {err}", file = sys.stderr)
            continue

        clean = True
        for _, archive_path in downloaded:
            findings = scan_archive(archive_path, name)
            # Delete archive immediately after scanning
            try:
                os.remove(archive_path)
            except OSError:
                pass
            crit_findings = [f for f in findings if f.severity == CRITICAL]
            if crit_findings:
                clean = False
                print(f"    {ver} -- CRITICAL finding(s), skipping")
                break

        shutil.rmtree(scan_dir, ignore_errors = True)

        if clean:
            print(f"    {ver} -- clean!")
            return ver

    return None


def update_req_line(raw_line: str, safe_ver: str, old_ver: str | None) -> str:
    """Rewrite a single requirements line to pin to safe_ver.

    Preserves env markers, inline comments, and line format.
    Appends a comment noting the pin.
    """
    # Split off inline comment
    comment = ""
    if " #" in raw_line:
        code_part, comment = raw_line.split(" #", 1)
        comment = " #" + comment
    else:
        code_part = raw_line

    # Split off env markers (after semicolon)
    marker = ""
    if ";" in code_part:
        code_part, marker = code_part.split(";", 1)
        marker = ";" + marker

    # Replace version specifier (==1.2.3, >=1.2, ~=1.0, !=1.1, or bare name)
    rewritten = re.sub(
        r"([A-Za-z0-9._-]+)\s*(?:[><=!~]=?[^;#,\s]*(?:\s*,\s*[><=!~]=?[^;#,\s]*)*)?",
        lambda m: f"{m.group(1)}=={safe_ver}",
        code_part.strip(),
        count = 1,
    )

    was_note = f" (was {old_ver})" if old_ver else ""
    pin_comment = f"  # pinned by pth_scanner{was_note}"

    return f"{rewritten}{marker}{pin_comment}"


def update_req_file(filepath: str, updates: dict[int, str]) -> None:
    """Apply line-level updates to a requirements file.

    updates: {line_num (1-indexed): new_line_text}

    Writes atomically (sibling tmp file, fsync, os.replace) so a crash mid-write
    never leaves a half-written file that re-introduces a malicious pin.
    """
    with open(filepath) as f:
        lines = f.readlines()

    for line_num, new_text in updates.items():
        idx = line_num - 1
        if 0 <= idx < len(lines):
            ending = "\n" if lines[idx].endswith("\n") else ""  # preserve line ending
            lines[idx] = new_text + ending

    dirpath = os.path.dirname(os.path.abspath(filepath)) or "."
    fd, tmp_path = tempfile.mkstemp(
        prefix = ".req_fix.",
        dir = dirpath,
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, filepath)
    except Exception:
        # Best effort cleanup; the destination was never touched.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _run_fix(critical_pkgs: set[str], entries: list[dict], max_search: int) -> None:
    """Run the --fix flow: find safe versions, update requirements files."""
    # Map package names to entries for source tracking
    pkg_entries: dict[str, list[dict]] = {}
    for e in entries:
        norm = e["name"].lower().replace("-", "_").replace(".", "_")
        pkg_entries.setdefault(norm, []).append(e)

    changes_summary: list[str] = []

    with tempfile.TemporaryDirectory(prefix = "pth_fix_") as tmpdir:
        for pkg_name in sorted(critical_pkgs):
            norm = pkg_name.lower().replace("-", "_").replace(".", "_")
            related = pkg_entries.get(norm, [])

            # Check if any are git deps
            git_entries = [e for e in related if e["is_git"]]
            if git_entries:
                for e in git_entries:
                    src = e["source_file"] or "CLI"
                    print(f"  [SKIP] {pkg_name} is a git URL dep in {src}, cannot auto-update")
                    changes_summary.append(f"  SKIP  {pkg_name} (git URL)")
                continue

            # Resolved version: try to extract from the spec (name==1.2.3)
            current_ver = None
            for e in related:
                spec = e["spec"]
                if "==" in spec:
                    current_ver = spec.split("==", 1)[1].split(";")[0].strip()
                    break

            if not current_ver:
                # If no pinned version, download to find what pip resolves
                dl_dir = os.path.join(tmpdir, f"resolve_{pkg_name}")
                os.makedirs(dl_dir, exist_ok = True)
                downloaded, download_errors = download_packages([pkg_name], dl_dir)
                if downloaded:
                    current_ver = get_downloaded_version(downloaded[0][1])
                else:
                    for err in download_errors:
                        print(f"  [WARN] {err}", file = sys.stderr)
                shutil.rmtree(dl_dir, ignore_errors = True)

            if not current_ver:
                print(f"  [WARN] Cannot determine current version of {pkg_name}, skipping fix")
                changes_summary.append(f"  SKIP  {pkg_name} (version unknown)")
                continue

            print(f"\n  Fixing {pkg_name} (current: {current_ver})...")
            safe_ver = find_safe_version(pkg_name, current_ver, tmpdir, max_search)

            if not safe_ver:
                print(
                    f"  [FAIL] No safe version found for {pkg_name} within {max_search} older versions"
                )
                changes_summary.append(
                    f"  FAIL  {pkg_name}=={current_ver} -> no safe version found"
                )
                continue

            print(f"  [OK]   {pkg_name}: {current_ver} -> {safe_ver}")
            changes_summary.append(f"  FIX   {pkg_name}=={current_ver} -> {pkg_name}=={safe_ver}")

            # Update all occurrences in requirements files
            file_updates: dict[str, dict[int, str]] = {}
            for e in related:
                if e["source_file"] is None:
                    # CLI arg, no file to update
                    print(f"         (CLI arg, no file to update)")
                    continue
                new_line = update_req_line(e["raw_line"], safe_ver, current_ver)
                file_updates.setdefault(e["source_file"], {})[e["line_num"]] = new_line
                print(f"         {e['source_file']}:{e['line_num']}")
                print(f"           - {e['raw_line']}")
                print(f"           + {new_line}")

            for filepath, updates in file_updates.items():
                update_req_file(filepath, updates)

    print(f"\n  {'=' * 72}")
    print(f"  FIX SUMMARY")
    print(f"  {'=' * 72}")
    for line in changes_summary:
        print(line)
    print(f"\n  Re-run without --fix to verify the scan is clean.")


# Directory scanning


def _find_requirements_files(root: str) -> list[str]:
    """Recursively find pip requirements files under root.

    Matches:
      - requirements*.txt (e.g. requirements.txt, requirements-dev.txt)
      - *.txt inside directories named 'requirements' (e.g. requirements/base.txt)
    Skips:
      - .egg-info dirs, venvs, hidden dirs, __pycache__, node_modules
    """
    import fnmatch

    skip_dirs = {"__pycache__", "node_modules", "venv", ".venv", "site-packages"}
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden and known non-requirement dirs
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and d not in skip_dirs and not d.endswith(".egg-info")
        ]
        dirname = os.path.basename(dirpath)
        for fname in sorted(filenames):
            if not fname.endswith(".txt"):
                continue
            if fnmatch.fnmatch(fname.lower(), "requirements*.txt"):
                results.append(os.path.join(dirpath, fname))
            # *.txt inside a directory named "requirements"
            elif dirname == "requirements":
                results.append(os.path.join(dirpath, fname))
    return sorted(results)


# Baseline allowlist: triaged known-good CRITICAL/HIGH findings so the gate can
# enforce without drowning in legitimate-library noise. Matched on
# (package, package-relative file, check, evidence hash); the hash strips
# ``L<NN>:`` markers so version bumps and line shifts do not reopen an entry,
# but changed flagged code does. Regenerate with ``--write-baseline``.

_DEFAULT_BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scan_packages_baseline.json"
)


def _norm_pkg(name: str) -> str:
    """PEP 503-style normalization so requests/Requests/req_uests collapse."""
    return re.sub(r"[-_.]+", "-", (name or "").strip().lower())


# Leading "<name>-<version>/" archive root of an sdist member, which carries the
# version. Stripping it (but keeping the rest of the path) gives a key that is
# stable across version bumps yet still distinguishes same-named files.
_RE_SDIST_ROOT = re.compile(r"^[^/]+-\d[^/]*/")


def _relpath_in_package(filename: str) -> str:
    """Package-relative path: drop an sdist's version-carrying archive root.

    Wheel members are already package-relative (``numba/cuda/utils.py``); sdist
    members sit under ``numba-0.60.0/...``, so strip that one leading segment.
    """
    return _RE_SDIST_ROOT.sub("", filename, count = 1)


# Evidence joins matched spans with " | " and a newline between labelled groups,
# each span tagged "L<NN>: ". Split only on those real delimiters (a " | " before
# a marker, or a newline), never on a bare "|" -- matched code may contain a
# bitwise-or or union type. The prefix strips only a genuine leading marker, an
# optional "Label: " then "L<NN>: "; a marker-like "L<NN>:" inside raw code (e.g.
# a .pth import line) has no leading marker and is left intact.
_RE_EVIDENCE_SPLIT = re.compile(r" \| (?=L\d+:)|\n")
_RE_EVIDENCE_PREFIX = re.compile(r"^(?:[A-Za-z][A-Za-z0-9 _/+.-]*:\s*)?L\d+:\s?")


def _canon_evidence(evidence: str) -> str:
    """Matched code lines in discovery order (markers removed), duplicates kept.

    Splits evidence on its real span delimiters, drops each span's leading
    label / line-number marker, and keeps the code with its indentation. Line
    shifts are absorbed by stripping the L<NN>: markers, not by sorting, so order
    stays significant: reordering matched lines (executable context, e.g. the
    arguments of a multi-line call) reopens the finding. Keeping duplicates means
    an appended identical occurrence still changes the key."""
    spans = []
    for s in _RE_EVIDENCE_SPLIT.split(evidence or ""):
        s = _RE_EVIDENCE_PREFIX.sub("", s, count = 1).rstrip()
        if s:
            spans.append(s)
    return "\n".join(spans)


def _evidence_hash(evidence: str) -> str:
    """Stable digest of the canonical matched evidence."""
    return hashlib.sha256(_canon_evidence(evidence).encode("utf-8", "replace")).hexdigest()


def _finding_key(f: Finding) -> tuple[str, str, str, str]:
    """Allowlist key: package, package-relative path, check, evidence hash.

    The evidence hash is over the set of matched code, so the key survives version
    bumps, line shifts and reordering but reopens when the flagged code changes --
    so a future payload in a baselined file/check is not auto-suppressed.
    """
    return (
        _norm_pkg(f.package),
        _relpath_in_package(f.filename),
        f.check,
        _evidence_hash(f.evidence),
    )


def _load_baseline(path: str) -> "dict[tuple[str, str, str, str], set[str] | None]":
    """Load an allowlist JSON into {match key: pinned file digests}.

    None means unpinned: the key alone suppresses, as before. A set of digests
    covers only those exact file contents, so any other edit to the file reopens
    the finding. For files whose danger sits outside the matched lines, e.g. a
    credential send whose evidence records the urlopen call but not its destination.
    """
    try:
        with open(path, "r", encoding = "utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [WARN] could not read baseline {path}: {exc}", file = sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"  [WARN] baseline {path} is not a JSON object", file = sys.stderr)
        return {}
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        print(f"  [WARN] baseline {path} entries is not a list", file = sys.stderr)
        return {}
    keys: dict[tuple[str, str, str, str], "set[str] | None"] = {}
    legacy = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        try:
            # Use the reviewed hash; else recompute it from the stored evidence.
            evidence_hash = e.get("evidence_hash") or _evidence_hash(e.get("evidence") or "")
            if not e.get("evidence_hash"):
                legacy += 1
            key = (
                _norm_pkg(e["package"]),
                _relpath_in_package(e["file"]),
                e["check"],
                evidence_hash,
            )
        except (KeyError, TypeError):
            continue
        # None = unpinned (key alone suppresses). A set = only those file digests.
        # An unpinned entry wins, since it already suppresses the key on its own.
        pin = e.get("file_sha256")
        if key not in keys:
            keys[key] = {pin} if pin else None
        elif not pin:
            keys[key] = None
        elif keys[key] is not None:
            keys[key].add(pin)
    if legacy:
        print(
            f"  [WARN] baseline {path}: {legacy} entries lack evidence_hash and may "
            f"not suppress until regenerated with --write-baseline (findings reopen "
            f"rather than risk hiding changed code under a coarse key)",
            file = sys.stderr,
        )
    return keys


def _write_baseline(
    path: str,
    findings: list[Finding],
    source: "str | None" = None,
) -> None:
    """Persist CRITICAL/HIGH findings as an allowlist for human triage.

    Pins are carried over from `source`, the baseline in effect for this run, so
    regenerating cannot silently widen a reviewed entry. Reading them from `path`
    instead would drop every pin whenever the output goes somewhere new.
    """
    pinned = {k for k, v in _load_baseline(source or path).items() if v is not None}
    entries = []
    seen: set[tuple[str, str, str, str]] = set()
    for f in sorted(findings, key = lambda f: SEVERITY_ORDER.get(f.severity, 99)):
        if f.severity not in (CRITICAL, HIGH):
            continue
        key = _finding_key(f)
        if key in seen:
            continue
        seen.add(key)
        entry = {
            "package": f.package,
            "file": _relpath_in_package(f.filename),
            "check": f.check,
            "severity": f.severity,
            "evidence": f.evidence,
            "evidence_hash": _evidence_hash(f.evidence),
        }
        if key in pinned and f.file_sha256:
            entry["file_sha256"] = f.file_sha256
        entries.append(entry)
    doc = {
        "_comment": (
            "scan_packages.py allowlist. Each entry is a CRITICAL/HIGH finding "
            "manually judged benign. Matched on (package, package-relative file, "
            "check, evidence_hash); evidence_hash is over the matched code with "
            "L<NN>: markers stripped, so version bumps and line shifts do not "
            "reopen an entry but changed code does. An optional file_sha256 pins an "
            "entry to that exact file, for danger sitting outside the matched lines "
            "(a credential send records the urlopen call, not its destination). "
            "severity and evidence are for review only. Regenerate with "
            "--write-baseline AFTER reviewing every line."
        ),
        "version": 1,
        "entries": entries,
    }
    with open(path, "w", encoding = "utf-8") as fh:
        json.dump(doc, fh, indent = 2, sort_keys = False)
        fh.write("\n")
    print(f"  Wrote {len(entries)} baseline entr(y/ies) to {path}")


def _partition_baseline(
    findings: list[Finding], baseline: "dict[tuple[str, str, str, str], set[str] | None]"
) -> tuple[list[Finding], list[Finding]]:
    """Split findings into (active, suppressed) by allowlist membership."""
    if not baseline:
        return list(findings), []
    active, suppressed = [], []
    for f in findings:
        key = _finding_key(f)
        hit = key in baseline
        if hit:
            pins = baseline[key]
            # A pinned entry only covers the file it was reviewed against.
            hit = pins is None or f.file_sha256 in pins
        (suppressed if hit else active).append(f)
    return active, suppressed


# Main


def main() -> int:
    parser = argparse.ArgumentParser(
        description = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "packages",
        nargs = "*",
        help = "Package specs (e.g. requests==2.32.5 fastapi)",
    )
    parser.add_argument(
        "-r",
        "--requirements",
        action = "append",
        default = [],
        metavar = "FILE",
        help = "Requirements file(s) to scan",
    )
    parser.add_argument(
        "-d",
        "--scan-dir",
        action = "append",
        default = [],
        metavar = "DIR",
        help = "Recursively find requirements*.txt files in DIR",
    )
    parser.add_argument(
        "--with-deps",
        action = "store_true",
        help = "Also download and scan transitive dependencies (full dependency tree)",
    )
    parser.add_argument(
        "--fix",
        action = "store_true",
        help = "Auto-search for safe versions and update requirements files",
    )
    parser.add_argument(
        "--max-search",
        type = int,
        default = 10,
        metavar = "N",
        help = "Max older versions to scan when searching for safe version (default: 10)",
    )
    parser.add_argument(
        "--baseline",
        metavar = "FILE",
        default = None,
        help = (
            "Allowlist JSON of triaged known-good findings to suppress. "
            f"Defaults to {os.path.basename(_DEFAULT_BASELINE_PATH)} next to this "
            "script if present."
        ),
    )
    parser.add_argument(
        "--no-baseline",
        action = "store_true",
        help = "Ignore the auto-discovered baseline allowlist.",
    )
    parser.add_argument(
        "--write-baseline",
        metavar = "FILE",
        default = None,
        help = (
            "Write the current CRITICAL/HIGH findings to FILE as an allowlist, "
            "then exit 0. Review every entry before committing it."
        ),
    )
    args = parser.parse_args()

    # --scan-dir: auto-discover requirements files
    req_files = list(args.requirements)
    for scan_dir in args.scan_dir:
        found = _find_requirements_files(scan_dir)
        if found:
            print(f"  Found {len(found)} requirements file(s) in {scan_dir}/")
            for f in found:
                print(f"    {f}")
            req_files.extend(found)
        else:
            print(f"  [WARN] No requirements files found in {scan_dir}/", file = sys.stderr)

    # Build unified entry list: list of dicts with source tracking
    entries: list[dict] = []

    # CLI args -> entries with no source file
    for pkg in args.packages or []:
        entries.append(
            {
                "spec": pkg,
                "name": _extract_pkg_name(pkg),
                "source_file": None,
                "line_num": None,
                "raw_line": pkg,
                "is_git": pkg.startswith("git+") or "git+" in pkg,
            }
        )

    # Requirements files -> entries with source tracking
    if req_files:
        entries.extend(parse_requirements(req_files))

    if not entries:
        parser.print_help()
        return 2

    # Deduplicate by normalized name, preserving first occurrence
    seen: set[str] = set()
    unique_entries: list[dict] = []
    for e in entries:
        key = e["name"].lower().replace("-", "_").replace(".", "_")
        if key not in seen:
            seen.add(key)
            unique_entries.append(e)

    specs = [e["spec"] for e in unique_entries]
    mode_label = " (with transitive deps)" if args.with_deps else ""
    print(f"  Scanning {len(specs)} package(s){mode_label}...")

    all_findings: list[Finding] = []

    # Hard pin-block: refuse to download known-malicious PyPI versions
    specs, blocked_findings = _check_blocked_pypi_versions(specs)
    all_findings.extend(blocked_findings)

    tmpdir = tempfile.mkdtemp(prefix = "pth_scan_")
    atexit.register(lambda d = tmpdir: shutil.rmtree(d, ignore_errors = True))
    download_errors: list[str] = []
    try:
        downloaded, download_errors = download_packages(
            specs,
            tmpdir,
            with_deps = args.with_deps,
        )
        print(f"  Downloaded {len(downloaded)} archive(s).")

        for spec, archive_path in downloaded:
            pkg_name = _extract_pkg_name(spec)
            findings = scan_archive(archive_path, pkg_name)
            all_findings.extend(findings)
            # Delete archive immediately after scanning
            try:
                os.remove(archive_path)
            except OSError:
                pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors = True)

    # Baseline allowlist: suppress triaged, known-good findings so the CI gate
    # can be enforcing without red-failing on legitimate-library noise.
    if args.no_baseline:
        baseline_path = None
    elif args.baseline:
        baseline_path = args.baseline
    elif os.path.isfile(_DEFAULT_BASELINE_PATH):
        baseline_path = _DEFAULT_BASELINE_PATH
    else:
        baseline_path = None
    baseline = _load_baseline(baseline_path) if baseline_path else {}

    active, suppressed = _partition_baseline(all_findings, baseline)

    print_findings(active)
    if suppressed:
        crit_s = sum(1 for f in suppressed if f.severity == CRITICAL)
        high_s = sum(1 for f in suppressed if f.severity == HIGH)
        med_s = sum(1 for f in suppressed if f.severity == MEDIUM)
        print(
            f"\n  {len(suppressed)} finding(s) suppressed by baseline "
            f"{baseline_path} "
            f"({crit_s} CRITICAL, {high_s} HIGH, {med_s} MEDIUM)."
        )

    # --fix mode: auto-search for safe versions (only real, non-baselined ones)
    if args.fix and active:
        critical_pkgs = {f.package for f in active if f.severity == CRITICAL}
        if critical_pkgs:
            print(
                f"\n  --fix: Searching for safe versions of {len(critical_pkgs)} CRITICAL package(s)..."
            )
            _run_fix(critical_pkgs, entries, args.max_search)

    # Surface pip-download failures BEFORE the exit code so a partial download
    # can't masquerade as "0 findings, all clean" (silent-failure hardening 4).
    # Also keeps us from writing a baseline from an incomplete scan.
    if download_errors:
        print(
            f"\n  {'=' * 72}\n"
            f"  SCAN INCOMPLETE: {len(download_errors)} pip download "
            f"failure(s):\n"
            f"  {'=' * 72}",
            file = sys.stderr,
        )
        for err in download_errors:
            print(f"  [ERROR] {err}", file = sys.stderr)
        print(
            "  Refusing to report 'all clean' on a partial scan; exiting 2.",
            file = sys.stderr,
        )
        return 2

    # --write-baseline: persist the full current CRITICAL/HIGH set as the new
    # allowlist (ignoring any loaded baseline), then exit 0. Only reached once
    # the scan is known complete.
    if args.write_baseline:
        _write_baseline(args.write_baseline, all_findings, source = baseline_path)
        return 0

    # Exit code: 1 only if a NON-baselined CRITICAL or HIGH remains. This is the
    # signal CI gates on once the baseline reaches a clean run.
    if any(f.severity in (CRITICAL, HIGH) for f in active):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
