# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contract for the chat response-details action and metadata."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from _en_catalog import en_string

REPO = Path(__file__).resolve().parents[2]
THREAD_TSX = REPO / "studio/frontend/src/components/assistant-ui/thread.tsx"
MESSAGE_MENU_TIME_TSX = REPO / "studio/frontend/src/components/assistant-ui/message-menu-time.tsx"
DETAILS_TSX = (
    REPO / "studio/frontend/src/components/assistant-ui/message-response-details-sheet.tsx"
)
DOCUMENT_PREVIEW_TSX = (
    REPO / "studio/frontend/src/features/rag/components/document-preview-sheet.tsx"
)
SHEET_TSX = REPO / "studio/frontend/src/components/ui/sheet.tsx"
REASONING_TSX = REPO / "studio/frontend/src/components/assistant-ui/reasoning.tsx"
ADAPTER_TS = REPO / "studio/frontend/src/features/chat/api/chat-adapter.ts"
CHAT_PREFS_TS = REPO / "studio/frontend/src/features/chat/stores/chat-preferences-store.ts"
CHAT_TAB_TSX = REPO / "studio/frontend/src/features/settings/tabs/chat-tab.tsx"
EN_LOCALE_TS = REPO / "studio/frontend/src/i18n/locales/en.ts"


# A className this reader cannot resolve. Distinct from None, which means the element carries
# no className at all, because the two want opposite treatment: absent is a fact to assert on,
# unreadable is a stale guard that must not quietly pass.
_UNREADABLE = "\x00unreadable"


def _class_list(source: str, marker: str) -> str | None:
    """The className of the JSX element whose opening tag contains `marker`.

    The opening tag is found first and the attribute read out of it, rather than matching
    `marker` and `className` as neighbours. JSX attribute order carries no meaning, so a
    `ref`, an `aria-*` or a test id inserted between them changes nothing about the element
    and must not fail a guard that is here to stop unrelated refactors reddening main.

    Comments come out of the whole source first. A commented-out element still contains its
    own `<`, so slicing from the marker found the tag inside `{/* ... */}` and reported the
    class list of something the page does not render.
    """
    source = _without_block_comments(source)
    start = source.find(marker)
    if start == -1:
        return None
    # Back up to the `<` that opens the element before reading forward. Scanning only the
    # suffix after the marker misses `className` written BEFORE it, which is the same
    # attribute-order assumption one level down.
    opens = source.rfind("<", 0, start + len(marker))
    if opens == -1:
        return None
    # The first `>` is not the end of the tag. An expression prop can contain one, and an
    # arrow function is the ordinary case: `onClick={() => ...}` ends the tag early and the
    # class list disappears. Only a `>` outside the JSX expression braces closes it.
    depth, end = 0, None
    for index in range(opens, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == ">" and depth == 0:
            end = index
            break
    if end is None:
        return None
    opening = _without_comments(source[opens : end + 1])
    # A spread can carry className, and nothing here can say what is in it. An element that
    # spreads props is an element whose class list is unknown, which is not the same as one
    # that states none: returning None would send the caller back to the base classes and
    # let an override through.
    #
    # Only a spread AFTER the explicit attribute, though. JSX applies attributes in order, so
    # `{...props} className="min-w-0"` ends with the explicit one whatever the spread holds,
    # and refusing that shape would fail a safe refactor rather than catch anything.
    if _spread_overrides(opening, "className"):
        return _UNREADABLE
    # On an attribute boundary, so that the name has to be the whole attribute. Unanchored,
    # `data-className="flex min-w-0"` matched on its suffix and its tokens came back as the
    # element's rendered classes, which the browser never applies.
    literal = re.search(r'(?:^|[\s{])className="([^"]*)"', opening)
    if literal:
        return literal.group(1)
    # An expression-valued className, `className={cn(...)}` or `className={"min-w-max"}`.
    # Its quoted pieces are what tailwind-merge sees, in this order; anything else in the
    # expression is beyond a reader like this one. Returning None here would be worse than
    # useless: the caller would fall back on the base classes and pass while the call site
    # overrode them, so an expression with no readable piece has to say so instead.
    expression = re.search(r"(?:^|[\s{])className=\{", opening)
    if not expression:
        return None
    # Only the className expression, closed by its own brace. Reading to the end of the tag
    # swept up quoted strings belonging to later attributes and called them classes.
    depth, body = 1, None
    for index in range(expression.end(), len(opening)):
        if opening[index] == "{":
            depth += 1
        elif opening[index] == "}":
            depth -= 1
            if depth == 0:
                body = opening[expression.end() : index]
                break
    if body is None:
        return _UNREADABLE
    # Strip a single cn(...) wrapper, then require every argument to be a plain string
    # literal. Anything else is a value this reader cannot evaluate: a conditional picks one
    # branch and flattening both reads the wrong one, and a bare identifier could be
    # anything at all. Either way the honest answer is that the class list is unknown, not
    # that it is whatever literals happen to be lying around.
    body = body.strip()
    wrapper = re.fullmatch(r"cn\((.*)\)", body, re.S)
    if wrapper:
        body = wrapper.group(1)
    arguments = [part.strip() for part in _split_arguments(body)]
    if not arguments or any(not re.fullmatch(r'"[^"]*"', part) for part in arguments):
        return _UNREADABLE
    return " ".join(part[1:-1] for part in arguments)


def _split_arguments(body: str) -> list[str]:
    """`body` split on top-level commas, ignoring those inside brackets or strings."""
    parts, depth, quoted, current = [], 0, False, []
    for char in body:
        if quoted:
            current.append(char)
            if char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current))
    return [part for part in parts if part.strip()]


def _assert_only_shrinks(tokens: list[str], what: str, evidence: str) -> None:
    """Exactly one min-width utility, unqualified, and it is `min-w-0`.

    An earlier version of this tried to work out which min-width WINS: last in `cn` order,
    per responsive variant, with `!important` beating an ordinary utility written after it.
    Every rule it gained was correct and the next one was still missing, because deciding
    that question properly is tailwind-merge plus the cascade, and a test file is the wrong
    place to keep a second copy of either.

    So it does not decide. One min-width, no variants, no importance markers, and it has to
    be the shrinking one. That is stricter than the framework: `min-w-0 md:min-w-0` really
    does shrink everywhere and is refused anyway. It is refused LOUDLY, saying that this
    guard does not adjudicate precedence, which is a message someone can act on, and it
    cannot quietly approve a layout nobody checked. Between a guard that is occasionally
    inconvenient and one that is occasionally wrong, this picks the first.
    """
    # min-width is necessary but not sufficient. The trigger shrinks because it is a flex
    # item that is allowed to: `shrink-0` or `flex-none` alongside `min-w-0` stops it
    # shrinking inside the header and lets a long summary widen the thread, with every
    # min-width check here still green. Refused rather than weighed, like the rest of this.
    pinned = [
        token
        for token in tokens
        # Only the ones that actually disable shrinking. `grow-0` sets flex-grow and leaves
        # flex-shrink alone, so an element with min-w-0 still shrinks normally; refusing it
        # would fail a correct flex-growth change for a defect it does not have.
        if re.fullmatch(r"(?:\S*:)?!?(?:shrink-0|flex-none)!?", token)
    ]
    assert not pinned, (
        f"{what} carries {sorted(set(pinned))}, which stops it shrinking however low its "
        f"min-width goes, so the summary widens its row instead. {evidence}"
    )
    widths = [token for token in tokens if _is_min_width(token)]
    assert widths, (
        f"{what} states no min-width at all, so whether it shrinks below its content is left "
        f"to whatever the element defaults to. {evidence}"
    )
    # By distinct utility: the same class written on both the base and the call site is a
    # duplicate, not a conflict, and tailwind-merge collapsing it changes nothing.
    assert set(widths) == {"min-w-0"}, (
        f"{what} carries min-width utilities this guard will not adjudicate between: "
        f"{sorted(set(widths))}. Which one wins is tailwind-merge's answer and then the cascade's, and "
        f"getting that wrong in either direction is worse than refusing: a variant-qualified "
        f"or important min-width, or more than one of them, has to be reduced to a single "
        f"unqualified min-w-0 for this to pass. {evidence}"
    )


# `[min-width:max-content]` sets the same property by another spelling, and a guard that
# only knows `min-w-*` reads a class list containing it as stating nothing. Splitting on the
# LAST colon breaks it too, since the value contains one, so the arbitrary form is matched
# before any variant is stripped.
_ARBITRARY_MIN_WIDTH = re.compile(r"(?:^|:)!?\[min-width:[^\]]*\]!?$")


def _is_min_width(token: str) -> bool:
    if _ARBITRARY_MIN_WIDTH.search(token.removeprefix("!")):
        return True
    # Variants are separated by `:`, but an arbitrary value can hold one of its own:
    # `min-w-[length:max-content]` compiles to `min-width: max-content` and drops an earlier
    # `min-w-0` through tailwind-merge, while splitting on the last `:` left `max-content]`
    # and this reader saw no min-width at all. Bracketed spans are masked before the split.
    masked = re.sub(r"\[[^\]]*\]", lambda found: "\x00" * len(found.group(0)), token)
    _, _, tail = masked.rpartition(":")
    utility = token[len(masked) - len(tail) :]
    return utility.removeprefix("!").removesuffix("!").startswith("min-w-")


def _cn_literals(source: str, anchor: str) -> str | None:
    """The string literals of the `cn(...)` call containing `anchor`, joined in order.

    `className` itself is expected among the arguments: that is the caller's contribution,
    read separately. Any OTHER unresolved argument means this cannot say what the element
    composes to, which is _UNREADABLE rather than silence.

    Comments out first, for the same reason `_class_list` does it: a dead `cn(...)` left in a
    block comment sits before the live component, so the anchor was found there and this
    validated classes nothing composes.
    """
    source = _without_block_comments(source)
    at = source.find(anchor)
    if at == -1:
        return None
    opens = source.rfind("cn(", 0, at)
    if opens == -1:
        return None
    depth, closes = 0, None
    for index in range(opens + 2, len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                closes = index
                break
    if closes is None:
        return _UNREADABLE
    pieces, forwards = [], False
    for argument in _split_arguments(source[opens + len("cn(") : closes]):
        argument = argument.strip()
        literal = re.fullmatch(r'"([^"]*)"', argument)
        if literal:
            pieces.append(literal.group(1))
        elif argument == "className":
            forwards = True
        else:
            return _UNREADABLE
    # The caller's classes have to actually arrive. Without `className` among the arguments
    # the component composes from its base alone, and the caller's contribution, which the
    # test adds to the order afterwards, reaches nothing. Reading the base and crediting the
    # call site anyway is how a trigger that stopped forwarding would keep passing.
    if not forwards:
        return _UNREADABLE
    return " ".join(pieces)


_STRING_LITERAL = re.compile(
    r"""(?:"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|`(?:[^`\\]|\\.)*`)""", re.S
)


def _without_code_comments(source: str) -> str:
    """`source` with comments blanked, string-aware: `track("a // b")` keeps its text.

    `_without_block_comments` works line by line on whole files and cannot tell a `//` in a
    string from a comment, which cut a callback short at the quoted marker.
    """
    out, index = [], 0
    while index < len(source):
        char = source[index]
        literal = _STRING_LITERAL.match(source, index) if char in "\"'`" else None
        if literal:
            out.append(literal.group(0))
            index = literal.end()
        elif source.startswith("//", index):
            end = source.find("\n", index)
            index = len(source) if end == -1 else end
        elif source.startswith("/*", index):
            end = source.find("*/", index + 2)
            index = len(source) if end == -1 else end + 2
            out.append(" ")
        else:
            out.append(char)
            index += 1
    return "".join(out)


def _without_strings(text: str) -> str:
    """`text` with every quoted literal emptied, so words inside a string are not read as code."""
    return _STRING_LITERAL.sub(lambda m: m.group(0)[0] * 2, text)


def _opening_tags(source: str, marker: str) -> list[str]:
    """Every opening JSX tag beginning at `marker`, brace-aware."""
    tags, at = [], source.find(marker)
    while at != -1:
        tag = _opening_tag(source[at:], marker)
        if tag:
            tags.append(tag)
        at = source.find(marker, at + len(marker))
    return tags


def _opening_tag(source: str, marker: str) -> str | None:
    """The opening JSX tag beginning at `marker`, brace-aware.

    `[^>]*` ends at the first `>`, and an arrow function in an earlier prop supplies one, so
    the tag would come back truncated and the props after it invisible.
    """
    opens = source.find(marker)
    if opens == -1:
        return None
    depth = 0
    index = opens
    while index < len(source):
        char = source[index]
        # A quoted `>` or brace (`title="a > b"`, `track("}")`) is text, not tag syntax.
        literal = _STRING_LITERAL.match(source, index) if char in "\"'`" else None
        if literal:
            index = literal.end()
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == ">" and depth == 0:
            return _without_comments(source[opens : index + 1])
        index += 1
    return None


def _without_block_comments(source: str) -> str:
    """The source with comments removed, block and line alike.

    Line comments matter as much as block ones here: everything below locates JSX by
    searching the text, so a stale `// <ReasoningBody isStreaming={...} ...>` left above the
    render is found by `_opening_tags` and read as the live element. The contract then
    describes a tag that renders nothing while the real one has lost both props.

    `(?<!:)` keeps `https://` out of it, which is the one `//` in this file that is not a
    comment.
    """
    source = re.sub(r"\{?\s*/\*.*?\*/\s*\}?", " ", source, flags = re.S)
    return "\n".join(re.sub(r"(?<!:)//.*$", "", line) for line in source.splitlines())


def _spread_overrides(tag: str, attribute: str) -> bool:
    """True when `tag` spreads props in a position that can beat an explicit `attribute`.

    JSX applies attributes left to right and the last write wins, so `{...props} name={x}`
    ends with `x` whatever the spread holds, while `name={x} {...props}` does not. Refusing
    both would make the guard red on a safe refactor that forwards unrelated props, which is
    a worse failure than the one it is guarding: it stops correct work.

    A tag with no explicit attribute at all is unknown if it spreads anything, since the
    spread is then the only thing that could be supplying it.
    """
    # At the tag's own attribute level only. An object spread inside another prop, such as
    # `onClick={() => call({...payload})}`, cannot reach className, and treating it as if it
    # could made this refuse a tag that is perfectly readable.
    spreads = []
    depth = 0
    for index, char in enumerate(tag):
        if char == "{":
            if depth == 0 and re.match(r"\{\s*\.\.\.", tag[index:]):
                spreads.append(index)
            depth += 1
        elif char == "}":
            depth -= 1
    if not spreads:
        return False
    explicit = re.search(rf"(?:^|[\s{{]){re.escape(attribute)}=", tag)
    return explicit is None or max(spreads) > explicit.start()


def _without_comments(tag: str) -> str:
    """`tag` with commented-out lines removed.

    A prop commented out is a prop that is not passed, and every check here is a substring
    test, so leaving the text in place lets a disabled prop satisfy the guard that exists to
    notice it went away. Only a `//` that begins a line counts, so a `//` inside a value is
    left alone; `/* ... */` is removed wherever it sits, because between two attributes is
    exactly where it sits when it is being used to switch a prop off.
    """
    kept = [line for line in tag.splitlines() if not line.lstrip().startswith("//")]
    return re.sub(r"/\*.*?\*/", " ", "\n".join(kept), flags = re.S)


def _prop_value(tag: str, name: str) -> str | None:
    """The expression inside `name={...}` on an opening tag, brace-aware, wherever it sits."""
    at = re.search(rf"(?:^|[\s{{]){re.escape(name)}=\{{", tag)
    if at is None:
        return None
    depth, index = 1, at.end()
    while index < len(tag) and depth:
        # A quoted brace (`track("{")`) is data, not the end of the expression.
        literal = _STRING_LITERAL.match(tag, index) if tag[index] in "\"'`" else None
        if literal:
            index = literal.end()
            continue
        depth += {"{": 1, "}": -1}.get(tag[index], 0)
        index += 1
    return tag[at.end() : index - 1] if depth == 0 else None


_STATEMENT_START = re.compile(
    r"\s*(?:const|let|var|function|async\s+function|return|if|for|while|switch|export|import"
    r"|class|type|interface)\b"
)


def _definition_body(name: str, source: str) -> str | None:
    """The text of `const|let|var name = ...;` or `function name(...) {...}`, brace-aware.

    A regex up to the first semicolon stopped inside a multi-statement body
    (`() => { track(); setDetailsOpen(true); }`) and missed the call.
    """
    at = re.search(
        # An optional type annotation may itself hold `=>` (`: () => void =`), so it is skipped up
        # to the first `=` that does not open an arrow.
        rf"(?:\b(?:const|let|var)\s+{re.escape(name)}\s*(?::(?:[^=;]|=>)*?)?=(?!>)"
        rf"|\bfunction\s+{re.escape(name)}\s*\()",
        source,
    )
    if at is None:
        return None
    is_function = at.group(0).startswith("function")
    # A function match ends on the `(` of its parameters, so it starts one level in.
    depth, index = (1 if is_function else 0), at.end()
    while index < len(source):
        char = source[index]
        # Quoted text and comments are not syntax: `track("}")` must not close the body.
        if char in "\"'`":
            index += 1
            while index < len(source) and source[index] != char:
                index += 2 if source[index] == "\\" else 1
        elif source.startswith("//", index):
            index = source.find("\n", index)
            index = len(source) if index == -1 else index
        elif source.startswith("/*", index):
            index = source.find("*/", index + 2)
            index = len(source) if index == -1 else index + 1
        elif char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
            if depth < 0 or (is_function and char == "}" and depth == 0):
                return source[at.end() : index + 1]
        elif char == ";" and depth == 0 and not is_function:
            return source[at.end() : index]
        elif (
            char == "\n"
            and depth == 0
            and not is_function
            and source[at.end() : index].strip()
            and _STATEMENT_START.match(source, index + 1)
        ):
            # No semicolon, and the next line starts a new statement: automatic semicolon
            # insertion ends the declaration here.
            return source[at.end() : index]
        index += 1
    return source[at.end() :]


# `setDetailsOpen(true)`, or a functional updater that always answers true
# (`setDetailsOpen(() => true)`, `setDetailsOpen((open) => true)`).
_OPENS_SHEET = re.compile(
    r"(?<![\w$.])setDetailsOpen\(\s*(?:true|(?:\(\s*[\w$]*\s*\)|[\w$]+)\s*=>\s*(?:true|\{\s*return\s+true\s*;?\s*\}))\s*\)"
)


def _opens_details(value: str, source: str) -> bool:
    """Whether a callback opens the sheet: inline, or the name of one this file defines."""
    # `analytics("setDetailsOpen(true)")` names the setter inside a string without calling it.
    if _OPENS_SHEET.search(_without_strings(value)):
        return True
    name = re.fullmatch(r"\s*([A-Za-z_$][\w$]*)\s*", value)
    if name is None:
        return False
    # A commented-out setter inside the body does not open anything.
    body = _definition_body(name.group(1), _without_code_comments(source))
    return body is not None and bool(_OPENS_SHEET.search(_without_strings(body)))


DETAILS_LABEL = "See response details"


# The prop itself, not `analytics.onShowDetails()` or `x?.onShowDetails()`.
_CALLS_SHOW_DETAILS = re.compile(r"(?<![\w$.])onShowDetails\s*(?:\?\.)?\(")


def _calls_show_details(
    value: str | None,
    source: str = "",
    _seen: frozenset = frozenset(),
) -> bool:
    """Whether an `onSelect` value hands over `onShowDetails` or calls it.

    `onSelect={() => onShowDetails}` mentions the name without calling it, so a bare mention
    inside a larger expression does not count. A bare name of a local handler
    (`onSelect={handleShowDetails}`) is followed to its definition in `source`.
    """
    if value is None:
        return False
    # `analytics("onShowDetails()")` names the call inside a string without making it.
    value = _without_strings(value)
    if re.fullmatch(r"\s*onShowDetails\s*", value) or _CALLS_SHOW_DETAILS.search(value):
        return True
    name = re.fullmatch(r"\s*([A-Za-z_$][\w$]*)\s*", value)
    if name is None or name.group(1) in _seen:
        return False
    body = _definition_body(name.group(1), _without_code_comments(source))
    if body is None:
        return False
    # The body of `const h = () => onShowDetails();` or `function h() { onShowDetails(); }`,
    # or `const h = onShowDetails;` which is the handler itself under another name.
    return _calls_show_details(body, source, _seen | {name.group(1)}) or bool(
        _CALLS_SHOW_DETAILS.search(_without_strings(body))
    )


def _names_details(tag: str) -> bool:
    """Whether the tag's own accessible name is the details label, literal or by catalog key."""
    quoted = rf"""(?:"{DETAILS_LABEL}"|'{DETAILS_LABEL}')"""
    if re.search(rf"(?<![\w-])aria-label=(?:{quoted}|\{{\s*{quoted}\s*\}})", tag):
        return True
    key = re.search(r"""(?<![\w-])aria-label=\{\s*t\(\s*["']([\w.]+)["']\s*\)\s*\}""", tag)
    if key is None:
        return False
    try:
        return en_string(key.group(1)) == DETAILS_LABEL
    except (KeyError, ValueError):
        return False


def _details_item_opens_sheet(menu: str) -> bool:
    """Whether one element both carries the details label and calls `onShowDetails`.

    Checked on the same opening tag: the label also appears in the item's tooltip, so a label
    anywhere plus a handler anywhere would stay green after either left the item itself.
    """
    for at in re.finditer(r"<[A-Za-z][\w.]*", menu):
        tag = _opening_tag(menu[at.start() :], at.group(0))
        if (
            tag
            and _names_details(tag)
            and _calls_show_details(_prop_value(tag, "onSelect"), menu)
            # A later `{...props}` can replace either prop, so the explicit ones prove nothing.
            and not _spread_overrides(tag, "aria-label")
            and not _spread_overrides(tag, "onSelect")
        ):
            return True
    return False


def _message_menu_time_tags(src: str) -> list[str]:
    """Opening `<MessageMenuTime` tags by exact name: `<MessageMenuTimestamp` is another component."""
    return [
        tag
        for tag in _opening_tags(src, "<MessageMenuTime")
        if re.match(r"<MessageMenuTime(?![\w$.])", tag)
    ]


def test_only_the_exact_component_name_counts():
    src = (
        "<MessageMenuTimestamp onShowDetails={() => setDetailsOpen(true)} />\n"
        "<MessageMenuTime.Item />\n"
        "<MessageMenuTime onShowDetails={open} />"
    )
    assert _message_menu_time_tags(src) == ["<MessageMenuTime onShowDetails={open} />"]


def test_assistant_more_menu_exposes_response_details_action():
    """The More menu still opens the details sheet. Since #11928 the item that does it lives in
    MessageMenuTime, beside the response's timestamp, so the action is followed through the prop
    the thread hands it rather than looked for in thread.tsx itself. Comments are removed first,
    so a commented-out element or handler does not count, and the prop is read wherever it sits
    and however the callback is spelled."""
    src = _without_block_comments(THREAD_TSX.read_text(encoding = "utf-8"))
    assert "MessageResponseDetailsSheet" in src
    tags = _message_menu_time_tags(src)
    assert tags, "thread.tsx no longer renders MessageMenuTime"
    callbacks = [_prop_value(tag, "onShowDetails") for tag in tags]
    assert any(
        value is not None and _opens_details(value, src) for value in callbacks
    ), f"no MessageMenuTime is handed a callback that opens the details sheet: {callbacks}"
    menu = _without_block_comments(MESSAGE_MENU_TIME_TSX.read_text(encoding = "utf-8"))
    assert _details_item_opens_sheet(
        menu
    ), f"no element labelled {DETAILS_LABEL!r} calls onShowDetails from its own onSelect"


@pytest.mark.parametrize(
    "value, source, opens",
    [
        ("() => setDetailsOpen(true)", "", True),
        ("showDetails", "const showDetails = () => setDetailsOpen(true);", True),
        (
            "showDetails",
            "const showDetails = () => {\n  track();\n  setDetailsOpen(true);\n};",
            True,
        ),
        (
            "showDetails",
            "const showDetails = useCallback(() => {\n  track();\n  setDetailsOpen(true);\n}, []);",
            True,
        ),
        ("showDetails", "function showDetails() {\n  track();\n  setDetailsOpen(true);\n}", True),
        (
            "showDetails",
            "const showDetails = () => {\n  track();\n};\nsetDetailsOpen(true);",
            False,
        ),
        ("showDetails", "function showDetails() {\n  track();\n}\nsetDetailsOpen(true);", False),
        (
            "showDetails",
            "const showDetails: () => void = () => setDetailsOpen(true);",
            True,
        ),
        (
            "showDetails",
            "const showDetails: Handler<void> = () => {\n  track();\n  setDetailsOpen(true);\n};",
            True,
        ),
        (
            "showDetails",
            "const showDetails: () => void = () => setDetailsOpen(false);",
            False,
        ),
        (
            "showDetails",
            'const showDetails = () => {\n  track("}");\n  setDetailsOpen(true);\n};',
            True,
        ),
        (
            "showDetails",
            "const showDetails = () => {\n  track(`;)`); // }\n  setDetailsOpen(true);\n};",
            True,
        ),
        (
            "showDetails",
            "const showDetails = () => {\n  /* setDetailsOpen(true) */ track();\n};\nsetDetailsOpen(true);",
            False,
        ),
        ("showDetails", "const showDetails = () => setDetailsOpen(false);", False),
        ("missing", "const showDetails = () => setDetailsOpen(true);", False),
        ('() => analytics("setDetailsOpen(true)")', "", False),
        ("() => setDetailsOpen(() => true)", "", True),
        ("() => setDetailsOpen((open) => true)", "", True),
        ("() => setDetailsOpen(() => { return true; })", "", True),
        ("showDetails", "const showDetails = () => setDetailsOpen(() => true);", True),
        ("() => setDetailsOpen((open) => !open)", "", False),
        ("() => setDetailsOpen(() => false)", "", False),
        (
            "showDetails",
            'const showDetails = () => { track("a // b"); setDetailsOpen(true); };',
            True,
        ),
        (
            "showDetails",
            'const showDetails = () => { track("/* x"); setDetailsOpen(true); track("*/"); };',
            True,
        ),
        (
            "showDetails",
            "const showDetails = () => track()\nconst unrelated = () => setDetailsOpen(true);",
            False,
        ),
        (
            "showDetails",
            "const showDetails = () =>\n  setDetailsOpen(true)\nconst unrelated = 1;",
            True,
        ),
        ("() => panel.setDetailsOpen(true)", "", False),
        ("showDetails", 'const showDetails = () => analytics("setDetailsOpen(true)");', False),
    ],
)
def test_the_callback_reader_follows_a_named_callback_to_its_end(value, source, opens):
    assert _opens_details(value, source) is opens


@pytest.mark.parametrize(
    "menu, opens",
    [
        ('<Item onSelect={onShowDetails} aria-label="See response details">', True),
        ('<Item aria-label="See response details" onSelect={() => onShowDetails()}>', True),
        (
            '<Item onSelect={() => { track(); onShowDetails?.(); }} aria-label="See response details">',
            True,
        ),
        ('<Item onSelect={() => onShowDetails} aria-label="See response details">', False),
        ('<Item onSelect={() => track(onShowDetails)} aria-label="See response details">', False),
        ('<Item onClick={onShowDetails} aria-label="See response details">', False),
        # Label and handler on different elements: the tooltip keeps the text, another item
        # keeps the handler, and the details item itself has neither.
        (
            '<Other onSelect={onShowDetails}>x</Other><Item aria-label="Copy">'
            "<Tip>See response details</Tip>",
            False,
        ),
        ('<Item onSelect={onShowDetails} aria-label="Copy">', False),
        ('<Item data-aria-label="See response details" onSelect={onShowDetails}>', False),
        (
            '<Item title="Open > details" aria-label="See response details" onSelect={onShowDetails}>',
            True,
        ),
        (
            '<Item onSelect={() => { track("}"); onShowDetails(); }} aria-label="See response details">',
            True,
        ),
        (
            '<Item onSelect={() => analytics("onShowDetails()")} aria-label="See response details">',
            False,
        ),
        (
            '<Item aria-label="See response details" onSelect={onShowDetails} {...itemProps}>',
            False,
        ),
        (
            '<Item {...itemProps} aria-label="See response details" onSelect={onShowDetails}>',
            True,
        ),
        ("<Item aria-label='See response details' onSelect={onShowDetails}>", True),
        ("<Item aria-label={'See response details'} onSelect={onShowDetails}>", True),
        (
            '<Item onSelect={() => { track("{"); onShowDetails(); }} aria-label="See response details">',
            True,
        ),
        (
            "const handleShowDetails = () => onShowDetails();\n"
            '<Item aria-label="See response details" onSelect={handleShowDetails}>',
            True,
        ),
        (
            "function handleShowDetails() {\n  track();\n  onShowDetails();\n}\n"
            '<Item aria-label="See response details" onSelect={handleShowDetails}>',
            True,
        ),
        (
            "const handleShowDetails = onShowDetails;\n"
            '<Item aria-label="See response details" onSelect={handleShowDetails}>',
            True,
        ),
        (
            "const handleShowDetails = () => onShowDetails;\n"
            '<Item aria-label="See response details" onSelect={handleShowDetails}>',
            False,
        ),
        ('<Item aria-label="See response details" onSelect={handleShowDetails}>', False),
        (
            '<Item aria-label="See response details" onSelect={() => analytics.onShowDetails()}>',
            False,
        ),
        ('<Item aria-label="See response details" onSelect={() => props?.onShowDetails()}>', False),
    ],
)
def test_the_details_item_must_carry_both_the_label_and_the_call(menu, opens):
    assert _details_item_opens_sheet(menu) is opens


def test_response_details_sheet_uses_unsloth_sheet_and_key_sections():
    src = DETAILS_TSX.read_text(encoding = "utf-8")
    assert "SheetContent" in src
    assert "Response details" in src
    assert "MessageResponseModelBadge" in src
    assert "showResponseModel" in src
    assert "ChipIcon" not in src
    assert "s.params.checkpoint" not in src
    assert "Not recorded" in src
    assert "min-w-0 break-words font-heading" in src
    assert "toolCallsFromContent(message.content)" in src
    assert 'label="Called"' in src
    for section in ["Response", "Tokens", "Timing", "Tools"]:
        assert f'title="{section}"' in src
    for field in ["Model", "Provider", "Total", "Cache hits", "Enabled", "Called"]:
        assert f'label="{field}"' in src


def assert_sheet_close_button_tracks_title_center(src: str) -> None:
    content_start = src.index("<SheetContent")
    content_tail = src[content_start:]
    content_end = re.search(r"(?m)^\s*>\s*$", content_tail)
    assert content_end is not None
    content_open = content_tail[: content_end.end()]
    header = src[src.index("<SheetHeader") : src.index("</SheetHeader>")]
    close_start = header.index("<SheetCloseButton")
    close = header[close_start : header.index("/>", close_start)]
    class_name = re.search(r'className="([^"]+)"', close)

    assert "showCloseButton={false}" in content_open
    assert '<div className="relative">' in header
    assert class_name is not None
    class_tokens = class_name.group(1).split()
    for token in ["absolute", "top-1/2", "right-0", "-translate-y-1/2"]:
        assert token in class_tokens


def test_sheet_headers_center_the_shared_close_button_on_the_title():
    assert_sheet_close_button_tracks_title_center(
        DETAILS_TSX.read_text(encoding = "utf-8"),
    )
    assert_sheet_close_button_tracks_title_center(
        DOCUMENT_PREVIEW_TSX.read_text(encoding = "utf-8"),
    )

    sheet_src = SHEET_TSX.read_text(encoding = "utf-8")
    close_button = sheet_src[
        sheet_src.index("function SheetCloseButton") : sheet_src.index("function SheetPortal")
    ]
    assert 'variant="ghost"' in close_button
    assert 'size="icon-sm"' in close_button
    assert "Cancel01Icon" in close_button
    assert '<span className="sr-only">Close</span>' in close_button
    assert '<SheetCloseButton className="absolute top-4 right-4" />' in sheet_src


def test_response_model_badge_is_user_configurable_and_rendered_once_per_message():
    prefs_src = CHAT_PREFS_TS.read_text(encoding = "utf-8")
    chat_tab_src = CHAT_TAB_TSX.read_text(encoding = "utf-8")
    thread_src = THREAD_TSX.read_text(encoding = "utf-8")
    reasoning_src = REASONING_TSX.read_text(encoding = "utf-8")

    assert "showResponseModel: boolean" in prefs_src
    assert "showResponseModel: false" in prefs_src
    assert "showResponseModel: saved?.showResponseModel ?? false" in prefs_src
    # The visible label lives in the locale file; the tab holds only the key that resolves to it.
    # Read by key, not by wording: #11924 rewrote this label without changing where it lives.
    assert en_string("settings.chat.showResponseModel", EN_LOCALE_TS)
    assert 't("settings.chat.showResponseModel")' in chat_tab_src
    assert "setShowResponseModel" in chat_tab_src
    details_src = DETAILS_TSX.read_text(encoding = "utf-8")
    assert (
        "aui-response-model-badge pointer-events-none relative inline-flex min-h-5" in details_src
    )
    assert "cursor-text select-text" in details_src
    assert "leading-5" in details_src
    assert "after:top-full after:h-1" in details_src
    assert "hover:opacity-100" in details_src
    assert "group-hover/assistant-message:opacity-100" in details_src
    # Pointer events gated behind hover/focus so the hidden badge stays inert when idle.
    assert "group-hover/assistant-message:pointer-events-auto" in details_src
    assert "group-focus-within/assistant-message:pointer-events-auto" in details_src
    assert thread_src.count("<MessageResponseModelBadge") == 1
    assert "hasReasoningParts" not in thread_src
    assert "group/assistant-message aui-assistant-message-root" in thread_src
    assert "pointer-events-none relative h-0" in thread_src
    assert "MessageResponseModelBadge" not in reasoning_src
    # The trigger has to be able to shrink below its content, or a long summary pushes the row
    # wider than the thread. `min-w-0` grants that. It used to be written `min-w-0 flex-1` on
    # one element; #11373 moved the filling to a header wrapper and left the shrinking on the
    # trigger, so pinning the old pair asserted one commit's layout rather than the property.
    #
    # Where the utility is written is not the property either. It currently appears twice, in
    # ReasoningTrigger's base classes and again on the call site, so demanding the call-site
    # copy would fail a harmless deduplication while the trigger still shrank. Both are read
    # and either satisfies it.
    #
    # Whole class tokens, not a substring: `\b` treats the colon in `md:min-w-0` as a
    # boundary, so a variant-qualified utility would satisfy a loose match while leaving the
    # trigger unable to shrink at every width it was not qualified for.
    # The WHOLE composition inside the component, not its first quoted literal. A second
    # literal appended after `className` would be effective and invisible to a first-match
    # read, and an argument this cannot resolve means the composition is unknown.
    # Bound to the element's className, not merely present in the file. The composition below
    # is found by its first literal, and a literal says nothing about where it is applied:
    # `data-className={cn(...)}` is valid JSX that renders none of these classes, so neither
    # the base composition nor the call site's min-w-0 would reach the DOM while every check
    # below went on describing them.
    live = _without_block_comments(reasoning_src)
    base_tags = [
        tag for tag in _opening_tags(live, "<Trigger") if 'data-slot="reasoning-trigger"' in tag
    ]
    assert base_tags, (
        'reasoning.tsx no longer renders a Trigger with data-slot="reasoning-trigger", so the '
        "base min-w-0 this guard reads belongs to no element on the page"
    )
    for tag in base_tags:
        lookalike = re.search(r"[\w-]className=", tag)
        assert not lookalike, (
            f"the reasoning trigger carries {lookalike.group(0)!r} rather than a className, so "
            f"the classes this guard reads render on nothing: {tag!r}"
        )
        applied_to = re.search(r"(?:^|[\s{])className=\{(.*)", tag, re.S)
        assert applied_to and '"aui-reasoning-trigger' in applied_to.group(1), (
            f"the reasoning trigger's className is not the composition this guard reads, so "
            f"what it measures is not what renders: {tag!r}"
        )

    base = _cn_literals(reasoning_src, '"aui-reasoning-trigger')
    assert base != _UNREADABLE, (
        "ReasoningTrigger composes its className from something this guard cannot resolve, "
        "so it cannot tell what the trigger ends up with. Widen the reader before trusting it"
    )
    # That it is rendered at all, before reading what it is given. `_class_list` answers None
    # both for a call site that passes no className and for one that is not there, and the
    # component's own function can stay behind unused, so without this the header could stop
    # rendering the trigger entirely and the checks below would go on describing base classes
    # that reach nothing.
    assert "<ReasoningTrigger" in _without_block_comments(reasoning_src), (
        "ReasoningTrigger is no longer rendered, so the shrinking this test is about belongs "
        "to an element that is not on the page"
    )
    # An inline style beats every utility below it in the cascade, and none of them is read
    # here. `style={{ minWidth: "max-content" }}` on the trigger leaves both min-w-0 checks
    # green while long summaries widen the row again, which is the whole defect.
    header_tags = [
        tag for tag in _opening_tags(live, "<div") if 'data-slot="reasoning-header"' in tag
    ]
    assert header_tags, (
        'reasoning.tsx no longer renders a div with data-slot="reasoning-header", so this '
        "guard cannot tell which element the trigger has to shrink inside"
    )
    # The header is a plain div carrying data-slot, not a <ReasoningHeader> component. Naming
    # the component matched nothing at all, so half of this check was vacuous and an inline
    # width on the real header went straight through.
    # The <Trigger> that ReasoningTrigger renders is where the base min-w-0 actually lives, so
    # it belongs in this loop as much as the call site does: a style there overrides the very
    # class this test reads, and checking only the call site and the header left it out.
    # Spreads are refused on the call site and the header, not on the base Trigger: that one
    # forwards `{...props}` by design, which is how a caller reaches it at all, and the values
    # arriving through it are exactly what the call-site check above adjudicates. Refusing it
    # here would fail the shipped component for doing the right thing.
    for tag in [*_opening_tags(live, "<ReasoningTrigger"), *header_tags]:
        assert not _spread_overrides(tag, "style"), (
            f"an element the min-w-0 chain depends on takes a spread that may carry a style, "
            f"which would outrank the utilities this guard compares: {tag!r}"
        )
    for tag in [*_opening_tags(live, "<ReasoningTrigger"), *header_tags, *base_tags]:
        assert not re.search(r"(?:^|[\s{])style=", tag), (
            f"an element the min-w-0 chain depends on carries an inline style, which outranks "
            f"the utilities this guard compares, so the width it computes is not the width "
            f"that renders: {tag!r}"
        )
    call_site = _class_list(reasoning_src, "<ReasoningTrigger")
    assert call_site != _UNREADABLE, (
        "the ReasoningTrigger call site passes a className this guard cannot resolve, so it "
        "cannot tell whether the base min-w-0 survives tailwind-merge. Widen the reader "
        "before trusting it"
    )
    # In `cn(base, className)` order, and the LAST min-w-* wins: cn runs tailwind-merge, so a
    # call site passing min-w-full or min-w-max drops the base min-w-0 and the trigger stops
    # shrinking. A union of the two would still hold the base token and call that fine.
    ordered = (base.split() if base else []) + (call_site.split() if call_site else [])
    assert ordered, (
        "neither ReasoningTrigger's base classes nor its call site carries a class list this "
        "can read, so this guard cannot see the trigger's layout at all"
    )
    _assert_only_shrinks(
        ordered,
        "the reasoning trigger",
        f"Base classes: {base!r}. Call site: {call_site!r}",
    )
    header = _class_list(reasoning_src, 'data-slot="reasoning-header"')
    assert header is not None, "the reasoning header row no longer carries a className"
    assert header != _UNREADABLE, (
        "the reasoning header row carries a className this guard cannot resolve, so it "
        "cannot tell whether the row still shrinks. Widen the reader before trusting it"
    )
    assert "flex" in header.split(), (
        f"the header row holding the trigger is no longer a flex row, so the trigger's own "
        f"shrinking is not what decides the layout any more: {header!r}"
    )
    # Read the same way as the trigger: a header that shrinks everywhere except above one
    # breakpoint puts the overflow back one level up at exactly those widths.
    _assert_only_shrinks(
        header.split(), "the header row holding the trigger", f"Classes: {header!r}"
    )


def test_reasoning_uses_continuous_transcript_without_legacy_height_cap():
    src = _without_block_comments(REASONING_TSX.read_text(encoding = "utf-8"))
    assert "retainStreamingHeight" not in src
    assert "resolveReasoningHeightCap" not in src
    assert "<ReasoningTranscript" in src
    assert "useCollapseScrollLock(" in src
    assert "? ANIMATION_DURATION + CLOSE_FALLBACK_MARGIN_MS" in src
    assert ": ANIMATION_DURATION," in src


def test_reasoning_clears_manual_open_on_a_new_stream():
    """A hand-opened block must not stay pinned open when the stream restarts.

    A nullable manual override outranks the visibility preference for one round;
    the next stream must return control to that preference.
    """
    src = _without_block_comments(REASONING_TSX.read_text(encoding = "utf-8"))

    # Which state holds the manual answer is read, not assumed. The previous form of this test
    # pinned `setManualOpen(false)`, and when #11433 replaced the `manualOpen` /
    # `dismissedWhileStreaming` pair with one nullable override the rename read as the reset
    # having been deleted. The override is identified the way the component itself identifies
    # it: it is the value `resolveReasoningOpen` is given as its override.
    opener = re.search(r"resolveReasoningOpen\(\{(.*?)\}\)", src, re.S)
    assert opener, (
        "reasoning.tsx no longer resolves its open state through resolveReasoningOpen, so "
        "this guard cannot tell which state holds a hand toggle's answer"
    )
    field = re.search(
        r"(?:^|,)\s*override\s*(?::\s*([A-Za-z_$][\w$]*))?\s*(?:,|$)", opener.group(1)
    )
    assert field, (
        f"resolveReasoningOpen is no longer passed an override, so nothing here outranks the "
        f"visibility setting and a hand toggle has nowhere to live: {opener.group(1)!r}"
    )
    held = field.group(1) or "override"
    # The identifier holding the live streaming flag, read the same way, because the predicate
    # below has to be handed it first.
    streaming_field = re.search(
        r"(?:^|,)\s*isStreaming\s*(?::\s*([A-Za-z_$][\w$]*))?\s*(?:,|$)", opener.group(1)
    )
    assert streaming_field, (
        f"resolveReasoningOpen is no longer passed an isStreaming, so this guard cannot tell "
        f"which value is the live stream: {opener.group(1)!r}"
    )
    held_streaming = streaming_field.group(1) or "isStreaming"
    setter = re.search(rf"const \[{re.escape(held)},\s*(set\w+)\]\s*=\s*useState", src)
    assert setter, (
        f"{held!r} reaches resolveReasoningOpen but is not a useState in reasoning.tsx, so "
        f"this guard cannot tell what writing it looks like"
    )
    writes = setter.group(1)

    # It is the hand toggle that writes it. Without this the override could be some derived
    # value the reader never sets, and clearing it would say nothing about a pinned block.
    handler = re.search(
        r"const handleOpenChange = useCallback\(\s*\(open: boolean\) => \{(.*?)\n    \}",
        src[src.index("const [override,") :],
        re.S,
    )
    assert handler and f"{writes}(open);" in handler.group(
        1
    ), "the hand toggle must store the requested open state directly"

    # And a new round clears it. Regenerate reuses this component instance, so without this a
    # block opened by hand over the last answer stays pinned open over the next one.
    round_at = src.find("startsNewReasoningRound(")
    assert round_at != -1, (
        "reasoning.tsx no longer asks whether a new reasoning round started, so nothing "
        "distinguishes a fresh stream from the end of the last one"
    )
    # Used positively, and as the whole condition. Locating the call and then reading the block
    # after it says nothing about the sense in which it was asked:
    # `if (!startsNewReasoningRound(...)) { setOverride(null); }` clears the override on every
    # transition EXCEPT a new round, which is the exact inverse of the contract, and it read as
    # green. Requiring `if (` to sit immediately before the call rejects the negation and any
    # other prefix; a condition that grows a second term has to teach this guard about it
    # rather than slipping past.
    assert re.search(r"if\s*\(\s*$", src[:round_at]), (
        "the new reasoning round is not asked as `if (startsNewReasoningRound(...))`. Negated "
        "or combined with another term it can clear the override on the transitions that are "
        "not a new round, and leave the one that is untouched, which is the inverse of what "
        "this guards"
    )
    # The brace has to be this condition's own. `_without_block_comments` strips the braces
    # around a comment along with it, so a branch whose body is only a comment loses its `{}`
    # entirely, and a scan for the next `{` then runs on into whatever block follows and reads
    # that one instead. Found by sabotage: emptying the new-round branch left this guard
    # reading the visibility-change branch, which clears the override too, so the guard passed
    # while the reset it exists for was gone.
    # In the order the predicate reads them. It is `isStreaming && !wasStreaming`, so swapping
    # the two type-checks and inverts the meaning: it then fires when a round ENDS, leaving a
    # hand-set override alive into the next one, which is the defect this whole test is about.
    # The current flag is the one resolveReasoningOpen is given; the previous one is the state
    # seeded from it.
    previous = re.search(rf"const \[(\w+), set\w+\] = useState\({re.escape(held_streaming)}\)", src)
    assert previous, (
        f"reasoning.tsx no longer keeps the previous streaming value in a useState seeded "
        f"from {held_streaming!r}, so this guard cannot tell which argument is which"
    )
    arguments = re.match(r"startsNewReasoningRound\(([^()]*)\)", src[round_at:])
    assert arguments and [part.strip() for part in arguments.group(1).split(",")] == [
        held_streaming,
        previous.group(1),
    ], (
        f"startsNewReasoningRound is called with "
        f"{arguments.group(1).strip() if arguments else 'arguments this guard cannot read'!r}. "
        f"It reads (current, previous) and returns true only when a round begins; the other "
        f"order type-checks and fires when one ends, leaving the override alive into the next"
    )
    # Nothing between the call and the brace but the `)` that closes the `if`. That rejects the
    # other way to invert it, `if (startsNewReasoningRound(...) === false) {`, which the older
    # form of this check let through because it only refused a statement or block boundary.
    opened = src.find("{", round_at)
    assert opened != -1 and re.fullmatch(
        r"startsNewReasoningRound\([^()]*\)\s*\)\s*", src[round_at:opened]
    ), (
        "the branch taken when a new reasoning round starts is not a block this guard can "
        "read: something sits between the predicate and the brace, so the block that follows "
        "runs under a condition other than the plain question this expects"
    )
    depth, closed = 0, None
    for index in range(opened, len(src)):
        if src[index] == "{":
            depth += 1
        elif src[index] == "}":
            depth -= 1
            if depth == 0:
                closed = index
                break
    assert closed is not None, "the new-round branch in reasoning.tsx is unterminated"
    # As a statement of the branch itself, not merely somewhere inside it. `if (false)
    # setOverride(null)` and a nested block both satisfy a substring search while a
    # hand-opened block stays pinned across a regenerate, which is the whole contract.
    statements = [piece.strip() for piece in src[opened + 1 : closed].split(";")]
    assert f"{writes}(null)" in statements, (
        f"a new reasoning round does not clear {held!r}. A block the reader opened by hand "
        f"during the previous round keeps its override, so it stays pinned open over the "
        f"next answer whatever the Thinking setting says"
    )


def test_response_details_metadata_is_persisted_without_backend_schema_change():
    src = ADAPTER_TS.read_text(encoding = "utf-8")
    assert "interface ResponseDetailsMetadata" in src
    assert "buildResponseDetails" in src
    assert "responseDetails: buildResponseDetails(finishedAt)" in src
    assert "toolCalls: Array.from(" in src
    assert "!isExternalRequest && supportsTools && toolsEnabled" in src
    assert re.search(r"selectedModelSummary\?\.name\s*\|\|\s*responseModelId", src)
    assert "providerName" in src
    assert "cancelId" in src
    metadata_block = src[
        src.find("interface ResponseDetailsMetadata") : src.find("type RunMessages")
    ]
    builder_block = src[
        src.find("const buildResponseDetails") : src.find("const externalCapabilities")
    ]
    # #11628: Code is recorded from the placement the request actually sends, a hosted sandbox or Studio's local
    # python/terminal/edit_file, so an external connection without a sandbox still reports Code when it runs locally.
    # Read inside the builder: the request payload further down tests studioLocalCodeTools too.
    assert re.search(
        r"code:\s*hostedCodeToolsForThisTurn\.length > 0\s*\|\|\s*"
        r"\(\s*supportsStudioToolsForThisTurn\s*&&\s*studioLocalCodeTools\.length > 0\s*\)",
        builder_block,
    ), "Response details no longer record Code from the hosted sandbox or Studio's local tools"
    for forbidden in [
        "encrypted_api_key",
        "externalApiKey",
        "apiKey",
        "providerKey",
        "secret",
    ]:
        assert forbidden not in metadata_block
        assert forbidden not in builder_block
