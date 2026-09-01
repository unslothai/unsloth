# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Follow the dependencies of source sliced into a ``node`` harness.

The harnesses in this directory pin frontend behaviour by slicing the real source VERBATIM
and running it under ``node --experimental-strip-types``. A slice only carries the lines
between its two markers, so the moment a sliced function starts calling a helper that lives
somewhere else the harness dies with ``ReferenceError: <helper> is not defined`` -- which is
what happened when ``sanitizeAssistantReplayText`` picked up ``stripSearchImageTokens``.

Naming the missing helper in the harness prelude fixes that one call and nothing else: the
next helper a sliced function gains breaks the harness again. So instead of another name,
this resolves them. Every identifier a slice references is looked up the way the module
itself would resolve it -- a top-level declaration in the same file, then its imports and
re-exports -- and the declaration is sliced in too, recursively.

Deliberately conservative. A name it cannot resolve, or a declaration it is not confident is
side-effect free, is left alone rather than guessed at: that is exactly the state the harness
is in today, so a helper this cannot follow is no worse off than before. Names the harness
already defines are never pulled, so the hand-written prelude fixtures still win.
"""

from __future__ import annotations

import re
from pathlib import Path

# `@/x` in the studio frontend means `studio/frontend/src/x`.
_ALIAS = "@/"

_DECL_RE = re.compile(
    r"^(?:export\s+)?(?:async\s+)?"
    r"(function\s*\*?|class|const|let|var|type|interface|enum)\s+"
    r"([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)
_IDENT_RE = re.compile(r"(?<![\w$])([A-Za-z_$][\w$]*)")
_IMPORT_RE = re.compile(
    r"^import\s+(?P<clause>[^;]*?)\s+from\s+[\"'](?P<spec>[^\"']+)[\"']", re.MULTILINE
)
_REEXPORT_RE = re.compile(
    r"^export\s+(?P<clause>\{[^}]*\}|\*)\s+from\s+[\"'](?P<spec>[^\"']+)[\"']", re.MULTILINE
)
# Initialisers a `const` may carry and still be safe to lift:
_SAFE_INIT_RE = re.compile(
    r"^(?:/|[\"'`]|\d|\[|\{|!|new\s+(?:Set|Map|RegExp|Date)\b"
    r"|async\s+function\b|function\b|async\s*\(|\(|[A-Za-z_$][\w$]*\s*(?:=>|;))"
)
_KEYWORDS = frozenset(
    """await break case catch class const continue debugger default delete do else enum export
    extends false finally for function if import in instanceof new null return super switch this
    throw true try typeof var void while with yield let static async of as from
    string number boolean any unknown never void object symbol bigint undefined readonly keyof
    typeof infer extends satisfies""".split()
)


# Keywords a `/` may legally follow, where it opens a regex rather than dividing.
_REGEX_MAY_FOLLOW = frozenset(
    """return case typeof instanceof in of delete void yield await throw new do else""".split()
)


def _blank_noise(text: str, keep_strings: bool = False) -> str:
    """The source with comments, strings and regex literals blanked to same-length spaces.

    Identifier scanning and bracket matching both run over this, so a brace inside a string
    or a regex quantifier (``[0-9a-f]{12}``) cannot be mistaken for structure. Newlines are
    kept so every offset still lines up with the original.

    Template literals are scanned with a stack: the text is blanked, but a ``${...}`` hole is
    ordinary code and is scanned as such, so a helper called only inside one is still found.
    With ``keep_strings`` the quoted literals survive, which is what the import parser needs
    to read a module specifier out of a statement whose comments are gone.
    """
    out = list(text)
    i, n = 0, len(text)
    # Frames of the scanner: "code" is ordinary source, "template" is inside a backtick.
    frames: list[tuple[str, int]] = [("code", 0)]

    def blank(start: int, stop: int) -> None:
        for k in range(start, min(stop, n)):
            if out[k] != "\n":
                out[k] = " "

    prev_word = ""
    prev_significant = ""
    while i < n:
        ch = text[i]
        if frames[-1][0] == "template":
            if ch == "\\":
                blank(i, i + 2)
                i += 2
                continue
            if ch == "`":
                out[i] = " "
                frames.pop()
                i += 1
                prev_significant = "`"
                prev_word = ""
                continue
            if ch == "$" and i + 1 < n and text[i + 1] == "{":
                blank(i, i + 2)
                frames.append(("code", 0))
                i += 2
                prev_significant = "{"
                prev_word = ""
                continue
            blank(i, i + 1)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            end = text.find("\n", i)
            end = n if end == -1 else end
            blank(i, end)
            i = end
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            end = n if end == -1 else end + 2
            blank(i, end)
            i = end
            continue
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            if not keep_strings:
                blank(i, min(j + 1, n))
            i = min(j + 1, n)
            prev_significant = ch
            prev_word = ""
            continue
        if ch == "`":
            out[i] = " "
            frames.append(("template", 0))
            i += 1
            continue
        if ch == "/" and (
            prev_word in _REGEX_MAY_FOLLOW
            or not (
                prev_significant in ")]"
                or prev_significant.isalnum()
                or prev_significant in "_$`'\""
            )
        ):
            # keywords above cannot be division.
            # A regex literal: `/` after an operator, an opening bracket or one of the keywords above cannot be
            j = i + 1
            in_class = False
            while j < n:
                c = text[j]
                if c == "\\":
                    j += 2
                    continue
                if c == "\n":
                    break
                if c == "[":
                    in_class = True
                elif c == "]":
                    in_class = False
                elif c == "/" and not in_class:
                    j += 1
                    break
                j += 1
            while j < n and text[j].isalpha():
                j += 1
            blank(i, j)
            i = j
            prev_significant = "/"
            prev_word = ""
            continue
        if ch in "{([":
            frames[-1] = (frames[-1][0], frames[-1][1] + 1)
        elif ch in ")]":
            frames[-1] = (frames[-1][0], frames[-1][1] - 1)
        elif ch == "}":
            if frames[-1][1] == 0 and len(frames) > 1:
                # The `}` closing a `${...}` hole:
                out[i] = " "
                frames.pop()
                i += 1
                prev_significant = "`"
                prev_word = ""
                continue
            frames[-1] = (frames[-1][0], frames[-1][1] - 1)
        if not ch.isspace():
            prev_significant = ch
        if ch.isalnum() or ch in "_$":
            start = i
            while i < n and (text[i].isalnum() or text[i] in "_$"):
                i += 1
            prev_word = text[start:i]
            prev_significant = text[i - 1]
            continue
        if not ch.isspace():
            prev_word = ""
        i += 1
    return "".join(out)


def _balanced(blanked: str) -> bool:
    depth = 0
    for ch in blanked:
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _block_end(blanked: str, start: int) -> int:
    """Index just past the closing brace of the top-level block beginning at ``start``.

    The first ``{`` is not always the body -- a destructured parameter or an inline return
    type opens one first -- so this walks the column-0 ``}`` lines these prettier-formatted
    sources end declarations on and takes the first one that leaves the slice balanced.
    """
    at = start
    while True:
        found = blanked.find("\n}", at)
        if found == -1:
            return -1
        end = found + 2
        if _balanced(blanked[start:end]):
            return end
        at = found + 1


def _assignment(blanked: str, start: int) -> int:
    """Index of the `=` that opens the initialiser of the declaration at ``start``.

    Not simply the first `=`: a typed binding can carry a `(a: X) => Y` annotation first, and
    reading the `>` of that arrow as the initialiser refuses a helper that is fine to lift.
    """
    depth = 0
    for i in range(start, len(blanked)):
        ch = blanked[i]
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        elif ch == ";" and depth == 0:
            return -1
        elif (
            ch == "="
            and depth == 0
            and blanked[i + 1 : i + 2] != "="
            and blanked[i - 1 : i] not in "=!<>"
        ):
            return i
    return -1


def _statement_end(blanked: str, start: int) -> int:
    """Index just past the `;` that ends the statement beginning at ``start``."""
    depth = 0
    for i in range(start, len(blanked)):
        ch = blanked[i]
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        elif ch == ";" and depth == 0:
            return i + 1
    return -1


class _Module:
    """One TypeScript source: its top-level declarations, imports and re-exports."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.text = path.read_text(encoding = "utf-8")
        self.blanked = _blank_noise(self.text)
        # must not read one out of a commented-out statement.
        # Comments gone, quoted literals kept:
        self.uncommented = _blank_noise(self.text, keep_strings = True)
        self.declarations: dict[str, list[tuple[str, int]]] = {}
        for match in _DECL_RE.finditer(self.blanked):
            self.declarations.setdefault(match.group(2), []).append(
                (match.group(1).strip(), match.start())
            )
        self.imports = self._parse(_IMPORT_RE)
        self.reexports = self._parse(_REEXPORT_RE)

    def _parse(self, pattern: re.Pattern[str]) -> dict[str, tuple[str, str]]:
        """``local name -> (module specifier, exported name)`` for one statement kind."""
        found: dict[str, tuple[str, str]] = {}
        for match in pattern.finditer(self.uncommented):
            clause = match.group("clause")
            spec = match.group("spec")
            if clause.lstrip().startswith("type") or ("*" in clause and "{" not in clause):
                continue
            braces = re.search(r"\{([^}]*)\}", clause, re.DOTALL)
            if braces is None:
                default = clause.strip()
                if re.fullmatch(r"[A-Za-z_$][\w$]*", default):
                    found[default] = (spec, "default")
                # `import type {...}` is erased by node, and `export * from` names nothing here.
                continue
            default = clause[: braces.start()].rstrip().rstrip(",").strip()
            if re.fullmatch(r"[A-Za-z_$][\w$]*", default):
                # `import Default, { named } from ...` binds both.
                found[default] = (spec, "default")
            for entry in braces.group(1).split(","):
                entry = entry.strip()
                if not entry or entry.startswith("type "):
                    continue
                parts = [p.strip() for p in re.split(r"\bas\b", entry)]
                exported = parts[0]
                local = parts[-1]
                if re.fullmatch(r"[A-Za-z_$][\w$]*", local):
                    found[local] = (spec, exported)
        return found

    def kind(self, name: str) -> str | None:
        entries = self.declarations.get(name)
        return entries[-1][0] if entries else None

    def declaration_text(self, name: str) -> str | None:
        """The full source of top-level ``name``, or ``None`` when it is not safe to lift."""
        for kind, start in self.declarations.get(name, ()):
            text = self._one_declaration(kind, start)
            if text is not None:
                return text
        return None

    def _one_declaration(self, kind: str, start: int) -> str | None:
        line_start = self.text.rfind("\n", 0, start) + 1
        if kind.startswith("function") or kind in {"class", "interface", "enum"}:
            body = self.blanked.find("{", start)
            semicolon = _statement_end(self.blanked, start)
            if (
                kind.startswith("function")
                and body == -1
                or (kind.startswith("function") and semicolon != -1 and semicolon < body)
            ):
                return None
            end = _block_end(self.blanked, line_start)
        elif kind == "type":
            end = _statement_end(self.blanked, start)
        else:
            equals = _assignment(self.blanked, start)
            if equals == -1 or not _SAFE_INIT_RE.match(self.text[equals + 1 :].lstrip()):
                return None
            end = _statement_end(self.blanked, start)
        if end == -1:
            return None
        text = self.text[line_start:end].strip("\n")
        if not _balanced(_blank_noise(text)) or "\nexport " in text:
            # Over-sliced into whatever followed.
            return None
        return text

    def references(self, source: str) -> list[str]:
        """Every identifier ``source`` uses as a name, in the order it first appears.

        Member accesses are skipped, and prettier breaks a chain before the `.`, so the
        nearest preceding non-space character is what decides. The `.` of a spread is not a
        member access: `...defaults` reads the binding, and missing that put a pulled `const`
        above the one it spreads. Object and interface keys are skipped too -- `{ role: "x" }`
        names a field, not a binding, and treating those as references pulled in whole
        modules a harness never asked for.
        """
        blanked = _blank_noise(source)
        names: list[str] = []
        for match in _IDENT_RE.finditer(blanked):
            at, name = match.start(), match.group(1)
            if name in _KEYWORDS or name in names:
                continue
            before = blanked[:at].rstrip()
            if before.endswith(".") and not before.endswith("..."):
                continue
            after = blanked[match.end() :].lstrip()
            if after.startswith(":") and not after.startswith("::") and before[-1:] in "{,":
                continue
            names.append(name)
        return names


def _resolve_module(spec: str, importer: Path, root: Path) -> Path | None:
    if spec.startswith(_ALIAS):
        base = root / spec[len(_ALIAS) :]
    elif spec.startswith("."):
        base = (importer.parent / spec).resolve()
    else:
        return None  # A package, not our source.
    for candidate in (
        base,
        Path(f"{base}.ts"),
        Path(f"{base}.mts"),
        base / "index.ts",
    ):
        if candidate.is_file():
            return candidate
    return None


def _harness_bindings(harness_source: str) -> set[str]:
    """Every name the harness itself binds at the top level, fixtures included.

    Deliberately over-inclusive: a name counted here is one that will not be pulled, so a
    stray extra costs a resolution the harness did not need, while a missed one is a
    duplicate declaration and a hard ``SyntaxError``. Destructured and multi-declarator
    bindings (``const { only, ...rest } = ...``) are the ones a declaration regex misses.
    """
    blanked = _blank_noise(harness_source)
    names: set[str] = set()
    for match in _DECL_RE.finditer(blanked):
        names.add(match.group(2))
    for match in re.finditer(r"^(?:export\s+)?(?:const|let|var)\s", blanked, re.MULTILINE):
        end = _statement_end(blanked, match.start())
        if end == -1:
            continue
        for declarator in _split_declarators(blanked[match.end() : end - 1]):
            names.update(_IDENT_RE.findall(declarator))
    return names - _KEYWORDS


def _split_declarators(region: str) -> list[str]:
    """The binding half of each declarator in `a = 1, { b } = c`, so every name is seen."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(region):
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(region[start:i])
            start = i + 1
    parts.append(region[start:])
    return [part.split("=")[0] for part in parts]


def _frontend_root(sources: tuple[Path, ...]) -> Path | None:
    for source in sources:
        for parent in source.parents:
            if parent.name == "src" and parent.parent.name == "frontend":
                return parent
    return None


def resolve_dependencies(
    harness_source: str,
    sources: tuple[Path, ...],
    root: Path | None = None,
) -> str:
    """``harness_source`` with the declarations its slices reference prepended.

    ``sources`` are the files the slices came from, in the order the harness concatenates
    them; a reference is resolved against the module it was sliced out of. Names the harness
    already defines -- the hand-written prelude fixtures included -- are never pulled, and
    anything unresolvable is left as it is.
    """
    sources = tuple(Path(s) for s in sources if Path(s).is_file())
    root = root or _frontend_root(sources)
    if root is None or not sources:
        return harness_source

    modules: dict[Path, _Module] = {}

    def module(path: Path) -> _Module:
        if path not in modules:
            modules[path] = _Module(path)
        return modules[path]

    defined = _harness_bindings(harness_source)
    pulled: dict[str, tuple[str, str, set[str]]] = {}
    order: list[str] = []
    seen: set[tuple[Path, str]] = set()

    def pull(name: str, origin: Path) -> None:
        """Emit ``name`` as resolved from ``origin``, dependencies first."""
        if name in defined or (origin, name) in seen:
            return
        seen.add((origin, name))
        home = module(origin)
        if name not in home.declarations:
            target = home.imports.get(name) or home.reexports.get(name)
            if target is None:
                return
            spec, exported = target
            path = _resolve_module(spec, origin, root)
            if path is None or path.suffix == ".tsx":
                # A component file is JSX, which node's type stripping will not parse.
                return
            pull(exported, path)
            if exported != name and exported in defined:
                # Imported under another name:
                defined.add(name)
                pulled[name] = ("const", f"const {name} = {exported};", {exported})
                order.append(name)
            return
        text = home.declaration_text(name)
        if text is None:
            return
        defined.add(name)
        # Dependencies first: a hoisted `function` would not care, but a `const` read before its declaration is a TDZ
        wanted = set()
        for reference in home.references(text):
            if (
                reference in home.declarations
                or reference in home.imports
                or reference in home.reexports
            ):
                # A name this module really does resolve, so failing to follow it matters.
                wanted.add(reference)
            pull(reference, origin)
        pulled[name] = (
            home.kind(name) or "const",
            f"// sliced from {home.path.name}\n" + re.sub(r"^export\s+", "", text, count = 1),
            wanted,
        )
        order.append(name)

    for source in sources:
        home = module(source)
        for reference in home.references(harness_source):
            pull(reference, source)

    # A declaration evaluated at import time cannot be left half-resolved: `const A = [B]` with `B` refused crashes the
    # whole harness on load, which is worse than the lazy ReferenceError it replaced.
    # A fixture the harness defines does not rescue one either: the fixtures sit BELOW this block, so reading one from
    eager = {"const", "let", "var", "class"}
    while True:
        doomed = {
            name
            for name, (kind, _, wanted) in pulled.items()
            if kind in eager and not wanted <= set(pulled)
        }
        if not doomed:
            break
        for name in doomed:
            del pulled[name]

    emitted = [pulled[name][1] for name in order if name in pulled]
    if not emitted:
        return harness_source
    header = (
        "// ---- Declarations the slices below reference, followed out of the studio sources\n"
        "// by tests/studio/_ts_deps.py. Verbatim, same as the slices themselves.\n"
    )
    block = header + "\n".join(emitted) + "\n"
    if not _balanced(_blank_noise(block)):
        return harness_source  # Never hand node something worse than it had.
    return block + harness_source
