# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Contract coverage for preset load settings (#7347)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    path = ROOT / relative
    if not path.exists():
        path = ROOT / "unsloth_repo" / relative
    return path.read_text(encoding = "utf-8")


def test_preset_interface_includes_load_config():
    policy = _read("studio/frontend/src/features/chat/presets/preset-policy.ts")
    assert "loadConfig?: PresetLoadConfig" in policy


def test_preset_save_captures_load_config():
    sheet = _read("studio/frontend/src/features/chat/chat-settings-sheet.tsx")
    assert "capturePresetLoadConfig()" in sheet
    assert "applyPresetLoadConfig" in sheet


def test_preset_apply_restores_load_config():
    sheet = _read("studio/frontend/src/features/chat/chat-settings-sheet.tsx")
    assert "if (p.loadConfig)" in sheet
    assert "applyPresetLoadConfig(p.loadConfig)" in sheet


def test_persisted_preset_serializes_load_config():
    storage = _read("studio/frontend/src/features/chat/utils/chat-settings-storage.ts")
    assert "normalizePresetLoadConfig(item.loadConfig)" in storage
    api = _read("studio/frontend/src/features/chat/api/chat-settings-api.ts")
    assert "loadConfig?: Record<string, unknown>" in api


def test_capture_reads_gguf_loaded_context():
    source = _read("studio/frontend/src/features/chat/presets/preset-load-config.ts")
    assert "store.loadedContextLength" in source
    assert "effectiveContextLength" in source


def test_apply_skips_missing_load_config():
    source = _read("studio/frontend/src/features/chat/presets/preset-load-config.ts")
    assert "if (config == null)" in source
    assert "selectedGpuIds: store.selectedGpuIds" in source
    sheet = _read("studio/frontend/src/features/chat/chat-settings-sheet.tsx")
    assert "if (p.loadConfig)" in sheet


def test_hydration_does_not_replay_preset_load_config():
    store = _read("studio/frontend/src/features/chat/stores/chat-runtime-store.ts")
    assert "applyPresetLoadConfig(activeDefinition.loadConfig)" not in store


def test_capture_coalesces_default_load_knobs():
    source = _read("studio/frontend/src/features/chat/presets/preset-load-config.ts")
    assert "coalesceDefaultLoadKnobs" in source
    assert "DEFAULT_MAX_SEQ_LENGTH" in source


def test_backend_chat_preset_accepts_load_config():
    routes = _read("studio/backend/routes/chat_history.py")
    assert "class ChatPresetLoadConfig" in routes
    assert "loadConfig: Optional[ChatPresetLoadConfig]" in routes


def test_preset_load_config_carries_parallel_slots():
    # Captured, clamped on read, applied, and accepted by the extra="forbid"
    # backend model (a missing backend field would 422 every settings sync).
    source = _read("studio/frontend/src/features/chat/presets/preset-load-config.ts")
    assert '| "nParallel"' in source
    assert "nParallel: snapshot.nParallel ?? null" in source
    assert "nParallel: config.nParallel ?? null" in source
    assert "N_PARALLEL_MAX, Math.round(partial.nParallel)" in source
    routes = _read("studio/backend/routes/chat_history.py")
    assert (
        "nParallel: Optional[int] = Field(default = None, ge = PARALLEL_MIN, le = PARALLEL_MAX)"
        in routes
    )


def test_preset_load_config_carries_reasoning_budget():
    source = _read("studio/frontend/src/features/chat/presets/preset-load-config.ts")
    assert '| "reasoningBudget"' in source
    assert '| "reasoningBudgetMessage"' in source
    assert "reasoningBudget: capturesReasoning ? snapshot.reasoningBudget : -1" in source
    assert "? snapshot.reasoningBudgetMessage" in source
    routes = _read("studio/backend/routes/chat_history.py")
    # NotABoolean: bool subclasses int, so a lax parse would take `true` for a budget of 1.
    assert "reasoningBudget: NotABoolean" in routes
    assert "reasoningBudgetMessage: Optional[str]" in routes


def test_diffusion_suppresses_reasoning_without_dropping_gguf_context():
    """loadedIsDiffusion gates the reasoning fields only, never the GGUF test.

    A loaded DiffusionGemma reports is_gguf and is_diffusion, so folding the
    diffusion check into isGguf made effectiveContextLength fall back to null and
    stopped capturing store.ggufContextLength. On auto sizing that is the whole
    load config, so the preset saved none at all.
    """
    source = _read("studio/frontend/src/features/chat/presets/preset-load-config.ts")
    capture = source[source.index("export function capturePresetLoadConfig") :]
    capture = capture[: capture.index("\n}")]
    gguf_test = capture[capture.index("const isGguf") : capture.index("const capturesReasoning")]
    assert (
        "loadedIsDiffusion" not in gguf_test
    ), "a diffusion GGUF is still a GGUF; its resolved context has to capture"
    assert "const capturesReasoning = isGguf && !store.loadedIsDiffusion" in capture


def test_preset_summary_marks_a_budget_message():
    """hasPresetLoadConfig() counts the message, so the summary has to as well.

    perModelConfigsEqual compares reasoningBudgetMessage, so a preset that sets only
    the message is non-default and does change llama-server behaviour. With no part
    for it the formatter returned null, and the sheet hides both "Active now" and
    "Saved in preset" on null. A marker, never the text: it can reach 8 KiB.
    """
    source = _read("studio/frontend/src/features/chat/presets/preset-load-config.ts")
    body = source[source.index("export function formatPresetLoadConfigSummary") :]
    body = body[: body.index("\n}")]
    assert "config.reasoningBudgetMessage" in body, (
        "a message-only preset summarises to null, so the Preset section shows no "
        "load settings at all for a config that is not default"
    )
    assert (
        "${config.reasoningBudgetMessage}" not in body
    ), "the message is free prose up to 8 KiB; the summary takes a marker only"


def _balanced(
    source: str,
    open_at: int,
    opener: str = "(",
    closer: str = ")",
) -> str:
    """The text between `open_at`'s bracket and its match, so a nested one does not end it."""
    depth = 0
    for index in range(open_at, len(source)):
        if source[index] == opener:
            depth += 1
        elif source[index] == closer:
            depth -= 1
            if depth == 0:
                return source[open_at + 1 : index]
    raise AssertionError(f"unbalanced {opener} at {open_at}")


_TOP_LEVEL_DECLARATION = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:function|const|class)\s+(\w+)", re.MULTILINE
)


def _component_body(source: str, name: str) -> str:
    """The module from `name`'s declaration up to the next top-level one.

    The sheet holds several components, and later ones subscribe to the runtime store too. A
    module-wide search would let `ChatSettingsPanel`'s selector be repointed at another field
    while a sibling's subscription kept the guard green, leaving exactly the stale memos this
    test exists to catch.
    """
    starts = [(match.start(), match.group(1)) for match in _TOP_LEVEL_DECLARATION.finditer(source)]
    for index, (offset, declared) in enumerate(starts):
        if declared == name:
            end = starts[index + 1][0] if index + 1 < len(starts) else len(source)
            return source[offset:end]
    raise AssertionError(f"no top-level declaration of {name}; was the component renamed?")


def _store_selectors(source: str) -> list:
    """Every `useChatRuntimeStore(...)` argument, whatever shape the selector takes."""
    out, needle = [], "useChatRuntimeStore("
    start = source.find(needle)
    while start != -1:
        out.append(_balanced(source, start + len(needle) - 1))
        start = source.find(needle, start + 1)
    return out


def _split_ternary(expression: str, guards: tuple = ()) -> list:
    """`cond ? a : b` as `[(a, conditions), (b, conditions)]`, recursively.

    Each result is paired with every condition governing whether it is the one returned, so a
    caller can tell a constant the field decides from a constant it has no say in. Depth aware,
    and `??` / `?.` are not ternaries.
    """

    # A wholly wrapped arm hides its own ternary at depth 1, where the scan below never looks.
    # A comma expression returns only its last operand; the others are evaluated and dropped.
    while True:
        expression = expression.strip()
        if expression.startswith("(") and _balanced(expression, 0) == expression[1:-1]:
            expression = expression[1:-1]
        elif len(_top_level_operands(expression, ",")) > 1:
            expression = _top_level_operands(expression, ",")[-1]
        else:
            break

    # Literals blanked: a `?` or `:` inside a message is not an operator.
    scan = _outside_literals(expression)
    depth, question = 0, -1
    index = 0
    while index < len(expression):
        char = scan[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char == "?":
            following = scan[index + 1 : index + 2]
            if following in ("?", "."):
                index += 2
                continue
            question = index
            break
        index += 1
    if question == -1:
        # `a && b` returns `a` whenever `a` is falsy, and `a || b` whenever it is truthy, so the
        # left operand is a path of its own. `||` binds looser, so it is split first.
        # `??` is deliberately not split. The sheet's own selector returns
        # `s.loadedReasoningBudgetRequested ?? s.reasoningBudget` while the budget equals the
        # loaded one, which source alone cannot prove distinct (see the sheet test's docstring).
        for operator, head_taken in (("||", True), ("&&", False)):
            operands = _top_level_operands(expression, operator)
            if len(operands) > 1:
                head, tail = operands[0], f" {operator} ".join(operands[1:])
                return _split_ternary(head, guards + ((head, head_taken),)) + _split_ternary(
                    tail, guards + ((head, not head_taken),)
                )
        return [(expression.strip(), guards)]

    # An unparenthesised nested ternary in the true arm owns the next colon, so count `?` here
    # too: taking the first one at bracket depth 0 cuts `a ? b ? c : d : e` into `a ? b` and
    # `d : e`, and a field named anywhere in that second blob would look like every arm reading it.
    condition = expression[:question]
    depth, nested = 0, 0
    index = question + 1
    while index < len(expression):
        char = scan[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char == "?":
            if scan[index + 1 : index + 2] in ("?", "."):
                index += 2
                continue
            nested += 1
        elif depth == 0 and char == ":":
            if not nested:
                # Direction matters: only `budget === -1` taken true pins the budget.
                return _split_ternary(
                    expression[question + 1 : index], guards + ((condition, True),)
                ) + _split_ternary(expression[index + 1 :], guards + ((condition, False),))
            nested -= 1
        index += 1
    return [(expression[question + 1 :].strip(), guards + ((condition, True),))]


def _without_comments(source: str) -> str:
    """`//` and `/* */` removed, string literals left alone: a comment is not part of an arm's
    value, so two arms differing only by one are the same expression."""
    out, index, quote = [], 0, None
    while index < len(source):
        char = source[index]
        if quote is not None:
            out.append(char)
            if char == "\\":
                out.append(source[index + 1 : index + 2])
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in "\"'`":
            quote = char
            out.append(char)
        elif source.startswith("//", index):
            index = source.find("\n", index)
            if index == -1:
                break
            continue
        elif source.startswith("/*", index):
            end = source.find("*/", index + 2)
            index = len(source) if end == -1 else end + 2
            out.append(" ")
            continue
        else:
            out.append(char)
        index += 1
    return "".join(out)


_FUNCTION_BODY_OPENS = re.compile(r"=>\s*\{|\bfunction\b[^(){};]*\([^()]*\)\s*\{")


def _nested_function_spans(block: str) -> list:
    """Where functions declared inside this block begin and end.

    Only these are another scope: an `if` or `for` body is the selector's own, so excluding by
    brace depth alone would read `if (s.enabled) { return s.other; }` as always returning the
    field.
    """
    spans, index = [], 0
    while True:
        match = _FUNCTION_BODY_OPENS.search(block, index)
        if match is None:
            return spans
        opener = block.index("{", match.start())
        depth = 0
        for offset in range(opener, len(block)):
            if block[offset] == "{":
                depth += 1
            elif block[offset] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((match.start(), offset))
                    index = offset
                    break
        else:
            spans.append((match.start(), len(block)))
            return spans


# A keyword, not a property of the same name: `s.switch` and `s.default` are reads.
_KEYWORD = r"(?<![\w$.])"


def _consume_statement(block: str, index: int):
    """The statement starting at `index`, and where the one after it starts."""
    while index < len(block) and block[index].isspace():
        index += 1
    if index >= len(block):
        return "", len(block)
    if block[index] == "{":
        body = _balanced(block, index, "{", "}")
        end = index + len(body) + 2
        return block[index:end], end
    if re.match(r"do\b(?!\s*:)", block[index:]):
        _, cursor = _consume_statement(block, index + 2)
        following = re.match(r"\s*while\s*(?=\()", block[cursor:])
        if following is not None:
            cursor += following.end()
            cursor += len(_balanced(block, cursor, "(", ")")) + 2
            semicolon = re.match(r"\s*;", block[cursor:])
            cursor += semicolon.end() if semicolon else 0
        return block[index:cursor], cursor
    keyword = re.match(r"\b(if|for|while|switch|catch|try|else|function)\b", block[index:])
    if keyword is not None:
        cursor = index + keyword.end()
        while cursor < len(block) and block[cursor] != "(" and block[cursor] != "{":
            cursor += 1
        if cursor < len(block) and block[cursor] == "(":
            cursor += len(_balanced(block, cursor, "(", ")")) + 2
        if keyword.group(1) == "try":
            # All clauses are one statement, or an exhaustive `try` reads as a fall-through.
            _, cursor = _consume_statement(block, cursor)
            while True:
                following = re.match(r"\s*\b(?:catch|finally)\b", block[cursor:])
                if following is None:
                    break
                cursor += following.end()
                while cursor < len(block) and block[cursor].isspace():
                    cursor += 1
                if cursor < len(block) and block[cursor] == "(":
                    cursor += len(_balanced(block, cursor, "(", ")")) + 2
                _, cursor = _consume_statement(block, cursor)
            return block[index:cursor], cursor
        if keyword.group(1) == "if":
            _, cursor = _consume_statement(block, cursor)
            tail = block[cursor:]
            following = re.match(r"\s*\belse\b", tail)
            if following is not None:
                _, cursor = _consume_statement(block, cursor + following.end())
            return block[index:cursor], cursor
        _, cursor = _consume_statement(block, cursor)
        return block[index:cursor], cursor
    # Without a `;` (automatic insertion), a statement keyword is where the next one begins:
    # none of them can continue an expression.
    next_statement = re.compile(
        rf"{_KEYWORD}(?:if|for|while|do|switch|return|break|continue|try|throw|const|let|var)\b"
    )
    depth = 0
    for cursor in range(index, len(block)):
        char = block[cursor]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char == ";":
            return block[index : cursor + 1], cursor + 1
        elif depth == 0 and cursor > index and next_statement.match(block, cursor):
            return block[index:cursor], cursor
    return block[index:], len(block)


def _labels_at_top_level(body: str) -> list:
    """Where each `case`/`default` arm of this switch body starts, ignoring nested switches."""
    scan = _outside_literals(body)
    depth, ends = 0, []
    for match in re.finditer(rf"[(\[{{}}\])]|{_KEYWORD}(?:case\b[^:]*|default\s*):", scan):
        token = match.group(0)
        if token in "([{":
            depth += 1
        elif token in ")]}":
            depth -= 1
        elif depth == 0:
            ends.append(match.end())
    return ends


def _switch_always_returns(statement: str) -> bool:
    """A switch with its own `default:` whose every arm returns or falls into one that does."""
    body = statement[statement.index("{") :] if "{" in statement else ""
    if not body:
        return False
    body = _balanced(body, 0, "{", "}")
    labels = _labels_at_top_level(body)
    scan = _outside_literals(body)
    if not any(re.search(r"\bdefault\s*:\s*$", scan[:end]) for end in labels):
        return False
    if not labels:
        return False
    for position, start in enumerate(labels):
        end = labels[position + 1] if position + 1 < len(labels) else len(body)
        arm = re.sub(r"\b(?:case\b[^:]*|default\s*):\s*$", "", body[start:end]).strip()
        if _breaks_out(arm):
            return False
        if _block_always_returns(arm):
            continue
        # Any other arm falls into the next label; the last one has none.
        if position + 1 == len(labels):
            return False
    return True


def _breaks_out(arm: str) -> bool:
    """A `break` or `continue` that leaves this switch arm without reaching a return.

    Not bracket depth: `if (stop) { break; }` still leaves. A nested switch absorbs `break` only.
    """
    return _jumps_out(arm, "break", absorbed_by_switch = True) or _jumps_out(
        arm, "continue", absorbed_by_switch = False
    )


def _jumps_out(text: str, keyword: str, *, absorbed_by_switch: bool) -> bool:
    """A `keyword` jump in `text` that nothing nested inside `text` absorbs."""
    scan = _outside_literals(text)
    # A statement, not an object key: `{ switch: 1 }` absorbs nothing.
    loops = r"(?:for|while)\s*\(|do\b(?!\s*:)"
    absorbers = rf"{loops}|switch\s*\(" if absorbed_by_switch else loops
    absorbing = []
    for match in re.finditer(rf"{_KEYWORD}(?:{absorbers})", scan):
        # Inside a span already taken, as the `while` closing a `do` is: it absorbs nothing more.
        if not any(begin <= match.start() < end for begin, end in absorbing):
            absorbing.append((match.start(), _consume_statement(scan, match.start())[1]))
    absorbing += _nested_function_spans(scan)
    return any(
        not any(begin <= match.start() < end for begin, end in absorbing)
        for match in re.finditer(rf"{_KEYWORD}{keyword}\b", scan)
    )


def _loop_always_enters(statement: str) -> bool:
    """A loop whose body is certain to run: `do`, or a test written as a literal true."""
    keyword = re.match(r"\b(for|while|do)\b", statement.strip())
    if keyword is None:
        return False
    if keyword.group(1) == "do":
        return True
    body = statement.strip()[keyword.end() :]
    opening = body.find("(")
    if opening == -1:
        return False
    test = _balanced(body, opening, "(", ")")
    if keyword.group(1) == "while":
        return test.strip() in ("true", "1")
    clauses = _top_level_split(test, ";")
    return len(clauses) == 3 and clauses[1].strip() in ("", "true", "1")


def _top_level_split(text: str, separator: str) -> list:
    """`text` split on `separator` at bracket depth zero."""
    scan = _outside_literals(text)
    parts, depth, start = [], 0, 0
    for index, char in enumerate(scan):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char == separator:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _loop_body_start(statement: str, keyword) -> int:
    """Where a loop's body begins: after `do`, or after the parenthesised test."""
    if keyword.group(0) == "do":
        return keyword.end()
    opening = statement.find("(", keyword.end())
    return opening + len(_balanced(statement, opening, "(", ")")) + 2


def _try_always_returns(statement: str) -> bool:
    """A `try` whose `finally` returns, or whose `try` and `catch` both do."""
    # Walked clause by clause, not searched: a nested try's clauses would stand in for these.
    scan = _outside_literals(statement)
    bodies, cursor = {}, 0
    while cursor < len(statement):
        while cursor < len(statement) and statement[cursor].isspace():
            cursor += 1
        clause = re.match(r"\b(try|catch|finally)\b", scan[cursor:])
        if clause is None:
            break
        cursor += clause.end()
        while cursor < len(statement) and statement[cursor].isspace():
            cursor += 1
        if cursor < len(statement) and statement[cursor] == "(":
            cursor += len(_balanced(statement, cursor, "(", ")")) + 2
            while cursor < len(statement) and statement[cursor].isspace():
                cursor += 1
        if cursor >= len(statement) or statement[cursor] != "{":
            return False
        body = _balanced(statement, cursor, "{", "}")
        bodies.setdefault(clause.group(1), body)
        cursor += len(body) + 2
    if "finally" in bodies and _block_always_returns(bodies["finally"]):
        return True
    return (
        "try" in bodies
        and "catch" in bodies
        and _block_always_returns(bodies["try"])
        and _block_always_returns(bodies["catch"])
    )


def _block_always_returns(block: str) -> bool:
    """Does every path out of this block go through a `return`?

    Falling off the end returns `undefined`, which holds steady while the field moves.
    """
    index = 0
    while index < len(block):
        statement, index = _consume_statement(block, index)
        stripped = statement.strip()
        if not stripped:
            break
        if re.match(r"\breturn\b", stripped):
            return True
        if stripped.startswith("{") and _block_always_returns(_balanced(stripped, 0, "{", "}")):
            return True
        if re.match(r"\bswitch\b", stripped) and _switch_always_returns(stripped):
            return True
        if re.match(r"\btry\b", stripped) and _try_always_returns(stripped):
            return True
        loop = re.match(r"\b(?:for|while|do)\b", stripped)
        if loop is not None and _loop_always_enters(stripped):
            repeated, _ = _consume_statement(stripped, _loop_body_start(stripped, loop))
            # A `continue` reaches the test too, and `do ... while (false)` then falls out.
            if _arm_always_returns(repeated) and not _breaks_out(repeated):
                return True
        branch = re.match(r"\bif\b\s*", stripped)
        if branch is None:
            continue
        cursor = branch.end()
        cursor += len(_balanced(stripped, cursor, "(", ")")) + 2
        taken, cursor = _consume_statement(stripped, cursor)
        following = re.match(r"\s*\belse\b", stripped[cursor:])
        if following is None:
            continue
        missed, _ = _consume_statement(stripped, cursor + following.end())
        if _arm_always_returns(taken) and _arm_always_returns(missed):
            return True
    return False


def _arm_always_returns(arm: str) -> bool:
    """One branch of an `if`, braced or not, seen as a block."""
    arm = arm.strip()
    if arm.startswith("{"):
        arm = _balanced(arm, 0, "{", "}")
    return _block_always_returns(arm)


def _own_scope_returns(block: str) -> list:
    """The `return` expressions of this block, not of a function nested in it: a helper returns
    its own value, which is not what zustand compares."""
    nested = _nested_function_spans(block)
    out = []
    for match in re.finditer(rf"{_KEYWORD}return\b", block):
        start = match.start()
        if any(begin <= start < end for begin, end in nested):
            continue
        end = len(block)
        for offset in range(match.end(), len(block)):
            if block[offset] in ";}":
                end = offset
                break
        out.append(block[match.end() : end])
    return out


# `\\[\s\S]`, not `\\.`: an escaped line break is a line continuation, still inside the literal.
_QUOTED = r"'(?:[^'\\]|\\[\s\S])*'|\"(?:[^\"\\]|\\[\s\S])*\""
# Blanking must pair every template, substitutions included, or it spans two of them.
_TEMPLATE = r"`(?:[^`\\]|\\[\s\S])*`"
# Only a template without `${}` is a constant a guard can pin to.
_TEMPLATE_CONSTANT = r"`(?:[^`\\$]|\\[\s\S]|\$(?!\{))*`"
_STRING_BODY = rf"{_QUOTED}|{_TEMPLATE_CONSTANT}"
_STRING_LITERAL = re.compile(rf"{_QUOTED}|{_TEMPLATE}")


def _dotted(expression: str) -> str:
    """`s["x"]` written as `s.x`, so one access has one spelling."""
    # Only a bracket outside literals is an access; inside one it is part of a value.
    scan = _outside_literals(expression)
    return re.sub(
        r"\[\s*(['\"])([A-Za-z_$][\w$]*)\1\s*\]",
        lambda match: f".{match.group(2)}" if scan[match.start()] == "[" else match.group(0),
        expression,
    )


def _normalised(expression: str) -> str:
    """Whitespace outside literals removed, a trailing `,`/`;` dropped, and `s["x"]` as `s.x`."""
    expression = _dotted(expression)
    pieces, last = [], 0
    for match in _STRING_LITERAL.finditer(expression):
        pieces.append(re.sub(r"\s+", "", expression[last : match.start()]))
        pieces.append(match.group(0))
        last = match.end()
    pieces.append(re.sub(r"\s+", "", expression[last:]))
    return "".join(pieces).rstrip(",;")


def _bindings(block: str, name: str) -> int:
    """How many places in `block` bind `name`: declarations (destructuring too), arrow, function
    and `catch` parameters. Scopes are not modelled, so any count past the expected one means a
    use cannot be resolved and the caller has to refuse rather than guess."""
    reference = rf"(?<![\w$.]){re.escape(name)}(?![\w$])"
    patterns = (
        rf"\b(?:const|let|var|function|class)\s+{reference}",
        rf"\b(?:const|let|var)\s*[{{\[][^=;]*{reference}",
        rf"{reference}\s*=>",
        rf"\([^()]*{reference}[^()]*\)\s*=>",
        rf"\b(?:function\b[^(]*|catch\s*)\([^()]*{reference}",
    )
    return sum(len(re.findall(pattern, block)) for pattern in patterns)


def _selector_signature(selector: str, field: str):
    """Where the selector's body starts, and the pattern that finds `field` being read in it.

    A destructured parameter is the other way to write the same subscription, so
    `({ reasoningBudget }) => reasoningBudget` and `({ reasoningBudget: budget }) => budget`
    are read through their local name. Returns (0, None, None) when the field cannot be found,
    or when the body binds the parameter's name again: a shadowing local is not the store.
    """
    plain = re.match(r"\s*\(?\s*([A-Za-z_$][\w$]*)\s*\)?\s*(?::[^=]*)?=>", selector)
    if plain is not None:
        if _bindings(selector[plain.end() :], plain.group(1)):
            return 0, None, None
        access = rf"{re.escape(plain.group(1))}\.{field}"
        return plain.end(), re.compile(rf"(?<![\w$.]){access}\b"), access

    destructured = re.match(r"\s*\(?\s*\{([^}]*)\}\s*\)?\s*(?::[^=]*)?=>", selector)
    if destructured is None:
        return 0, None, None
    for entry in destructured.group(1).split(","):
        name, _, alias = entry.partition(":")
        if name.strip() == field:
            local = alias.strip() or field
            if _bindings(selector[destructured.end() :], local):
                return 0, None, None
            access = re.escape(local)
            # Not preceded by `.`: `s.other.reasoningBudget` is some other object's property.
            return destructured.end(), re.compile(rf"(?<![\w$.]){access}(?![\w$])"), access
    return 0, None, None


_NUMBER = r"-?(?:0[xXbBoO][\da-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
# The lookahead stops `1e3` matching as `1`.
_LITERAL = rf"(?:null|undefined|true|false|{_NUMBER}|{_STRING_BODY})(?![\w$.])"


def _outside_literals(expression: str) -> str:
    """`expression` with the contents of quoted literals blanked, offsets unchanged."""
    out, last = [], 0
    for match in _STRING_LITERAL.finditer(expression):
        out.append(expression[last : match.start()])
        out.append(match.group(0)[0] + " " * (len(match.group(0)) - 2) + match.group(0)[-1])
        last = match.end()
    out.append(expression[last:])
    return "".join(out)


def _unwrapped(expression: str) -> str:
    """`expression` without parentheses that wrap all of it; `(a) === (b)` keeps both."""
    while expression.startswith("(") and expression.endswith(")"):
        depth = 0
        for index, char in enumerate(expression):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    break
        if index != len(expression) - 1:
            break
        expression = expression[1:-1].strip()
    return expression


def _top_level_operands(guard: str, operator: str = "&&") -> list:
    """`guard` split on the `operator`s that are not inside brackets."""
    parts, depth, start = [], 0, 0
    scan = _outside_literals(guard)
    index = 0
    while index < len(guard):
        char = scan[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and scan.startswith(operator, index):
            parts.append(guard[start:index])
            index += 2
            start = index
            continue
        index += 1
    parts.append(guard[start:])
    return [part.strip() for part in parts if part.strip()]


_STRING_ESCAPES = {
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "b": "\b",
    "f": "\f",
    "v": "\v",
    "0": "\0",
}


def _decoded(text: str) -> str:
    """A string literal's body as the runtime string it denotes: `"\\x61"` is `"a"`."""
    out, index = [], 0
    while index < len(text):
        char = text[index]
        if char != "\\" or index + 1 >= len(text):
            out.append(char)
            index += 1
            continue
        marker = text[index + 1]
        if marker == "x" and re.fullmatch(r"[0-9a-fA-F]{2}", text[index + 2 : index + 4]):
            out.append(chr(int(text[index + 2 : index + 4], 16)))
            index += 4
        elif marker == "u" and text[index + 2 : index + 3] == "{":
            close = text.find("}", index + 3)
            digits = text[index + 3 : close] if close != -1 else ""
            if close != -1 and re.fullmatch(r"[0-9a-fA-F]+", digits):
                out.append(chr(int(digits, 16)))
                index = close + 1
            else:
                out.append(marker)
                index += 2
        elif marker == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", text[index + 2 : index + 6]):
            out.append(chr(int(text[index + 2 : index + 6], 16)))
            index += 6
        elif marker in "\r\n\u2028\u2029":
            # A line continuation contributes no character.
            index += 3 if text[index + 1 : index + 3] == "\r\n" else 2
        else:
            out.append(_STRING_ESCAPES.get(marker, marker))
            index += 2
    # Join UTF-16 surrogate pairs.
    return re.sub(
        r"[\ud800-\udbff][\udc00-\udfff]",
        lambda pair: chr(
            0x10000 + ((ord(pair.group(0)[0]) - 0xD800) << 10) + (ord(pair.group(0)[1]) - 0xDC00)
        ),
        "".join(out),
    )


def _literal_value(text: str):
    """What a literal denotes (`0x10` is `16`, `'x'` is `"x"`); anything else is its own text."""
    text = text.strip()
    if re.fullmatch(_NUMBER, text):
        try:
            return float(int(text, 0))
        except ValueError:
            return float(text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"`":
        if text[0] == "`" and re.search(r"\$\{", text):
            return text
        return ("string", _decoded(text[1:-1]))
    return text


def _pinned_literal(guard: str, taken: bool, access: str, field: str):
    """The one value `field` can hold on this branch, or None if the branch does not fix it.

    `budget === -1` taken true pins -1 and `budget !== null` taken false pins null; `budget > 0`
    pins nothing. The comparison must be on the selector's own parameter, as one operand pair.
    """
    # Bounded, or `s.reasoningBudget` matches inside `defaults.reasoningBudget`.
    start, end = r"(?<![\w$.])", r"(?![\w$])"
    bound = rf"{start}{access}{end}"
    # Strict only: `msg == 0` is taken by both "" and "0".
    equal = rf"(?:{bound}===({_LITERAL})|({_LITERAL})==={bound})"
    unequal = rf"(?:{bound}!==({_LITERAL})|({_LITERAL})!=={bound})"

    # Matching a whole top-level conjunct excludes a negated or disjoined comparison without
    # refusing an unrelated `!` elsewhere in the guard.
    def _separates(literal: str) -> bool:
        """`===` tells this literal apart from every other value.

        Not zero: `=== 0` also takes -0, which zustand's Object.is tells apart, and the preset
        normaliser can produce -0 through Math.trunc.
        """
        value = _literal_value(literal)
        return not (isinstance(value, float) and value == 0.0)

    if taken:
        # The comparison must BE a conjunct: `(budget === -1) === false` pins nothing, and
        # neither does `budget === -1 && a || b`, which `b` alone can make true.
        if len(_top_level_operands(_unwrapped(guard), "||")) > 1:
            return None
        for conjunct in _top_level_operands(_unwrapped(guard)):
            match = re.fullmatch(equal, _unwrapped(conjunct))
            if match is not None:
                found = match.group(1) or match.group(2)
                return found if _separates(found) else None
        return None
    match = re.fullmatch(unequal, _unwrapped(guard))
    if match is None:
        return None
    found = match.group(1) or match.group(2)
    return found if _separates(found) else None


def _is_boolean(expression: str) -> bool:
    """A comparison or negation at the top level: `s.budget > 0` is `true` for 1 and for 2."""
    scan = _outside_literals(_unwrapped(expression.strip().rstrip(",;")))
    if re.match(rf"\s*(?:!|{_KEYWORD}typeof\b)", scan):
        return True
    depth, flat = 0, []
    for char in scan:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        flat.append(char if depth == 0 else " ")
    top_level = "".join(flat).replace("=>", "  ")
    return re.search(r"[=!]==?|[<>]|\b(?:instanceof|in)\b", top_level) is not None


def _selector_reads(selector: str, field: str) -> bool:
    """Does every value this selector can return depend on `field`?

    Zustand re-renders on the RESULT, not on a property the selector happened to touch, so one
    that tests `s.<field>` and returns something else either way tracks nothing. A result the
    field does not appear in still counts when the field decides whether it is returned at all,
    as in `s.<field> !== null ? s.<field> : null`. The parameter name comes from the signature
    rather than being assumed to be `s`.
    """

    selector = _without_comments(selector)
    # A `/` outside strings is a division or a regex literal, and a regex can spell `default:`
    # or `break`. Neither is read here, so the selector is refused rather than misread.
    if "/" in _outside_literals(selector):
        return False
    # Every literal is parked behind a placeholder while the structure is read, so a `;`, a
    # bracket or a keyword inside a message is never taken for code. Values come back only to
    # compare pins against results.
    literals = []

    def _park(match) -> str:
        literals.append(match.group(0))
        placeholder = f"'@{len(literals) - 1}'"
        # A template substitution is code, so it stays readable beside the placeholder.
        text, substitutions = match.group(0), []
        start = text.find("${")
        while start != -1:
            inner = _balanced(text, start + 1, "{", "}")
            if text[start - 1] != "\\":
                # Parked too: a message inside a substitution is no more code than any other.
                substitutions.append(_STRING_LITERAL.sub(_park, _without_comments(inner)))
            start = text.find("${", start + len(inner) + 3)
        return f"({placeholder}, {', '.join(substitutions)})" if substitutions else placeholder

    def _restored(text: str) -> str:
        return re.sub(r"'@(\d+)'", lambda match: literals[int(match.group(1))], text)

    selector = _STRING_LITERAL.sub(_park, _dotted(selector))
    # `s. switch` and `s .x` are member accesses like `s.switch`: one spelling for every scan.
    selector = re.sub(r"\s*\.\s*(?=[A-Za-z_$])", ".", selector)
    signature, read, access = _selector_signature(selector, field)
    if read is None:
        return False
    body = selector[signature:].strip()
    if body.startswith("{"):
        block = _balanced(body, 0, "{", "}")
        # Inline `const` bindings only: a `let` can be reassigned before it is returned.
        # Brace-free right-hand sides only, or a function body is cut.
        for name, expression in re.findall(r"\bconst\s+(\w+)\s*=\s*([^;{}]+);", block):
            # A reference, not a property that happens to share the name: `s.other.value`.
            # A name bound anywhere else (again, as a parameter, in a `catch`) or used as an
            # object key is left uninlined, since which binding a use resolves to is not read.
            reference = rf"(?<![\w$.]){re.escape(name)}(?![\w$])"
            if _bindings(block, name) > 1 or re.search(rf"[{{,]\s*{reference}\s*:", block):
                continue
            # `const a = x, b = y` declares two names; its text past the comma is not `a`'s value.
            if len(_top_level_operands(expression, ",")) > 1:
                continue
            block = re.sub(reference, f"({expression})", block)
        results = [
            result
            for expression in _own_scope_returns(block)
            for result in _split_ternary(expression)
        ]
        if results and not _block_always_returns(block):
            results.append(("undefined", ()))
    else:
        # An arrow's expression body ends at its first top-level comma: what follows is the
        # next argument of the call, such as an equality function.
        results = _split_ternary((_top_level_operands(body, ",") or [""])[0])
    if not results or any(_is_boolean(result) for result, _ in results):
        return False
    # The field is looked for while literals are still parked: `"s.budget"` is not a read.
    results = [
        (
            read.search(_normalised(result)) is not None,
            _restored(_normalised(result)),
            tuple((_restored(_normalised(guard)), taken) for guard, taken in guards),
        )
        for result, guards in results
    ]
    # Every path returns the field, or returns exactly the value a guard pins it to:
    # `s.budget === -1 ? 0 : s.budget` returns 0 for both -1 and 0.
    return all(
        reads_field
        or any(
            _pinned_literal(guard, taken, access, field) is not None
            and _literal_value(_pinned_literal(guard, taken, access, field))
            == _literal_value(result)
            for guard, taken in guards
        )
        for reads_field, result, guards in results
    )


def _memo_dependency_lists(source: str) -> dict:
    """Each `const X = useMemo(...)`'s dependency array, keyed by the name it is bound to.

    Keyed rather than counted: the memos that capture the preset config are named ones, and a
    bare tally cannot tell a field moving out of `currentLoadSummary` and into some unrelated
    memo from it never moving at all.
    """
    out = {}
    for match in re.finditer(r"const\s+([A-Za-z_$][\w$]*)\s*=\s*useMemo\(", source):
        body = _balanced(source, match.end() - 1)
        bracket = body.rfind("[")
        if bracket != -1:
            names = _balanced(body, bracket, "[", "]")
            out[match.group(1)] = [name.strip() for name in names.split(",") if name.strip()]
    return out


# The guard above is only as good as this predicate, and it is the part a refactor of the sheet
# will walk into. Accepted and rejected forms, stated as cases rather than left to the one
# spelling the sheet happens to use today.
SELECTOR_CASES = [
    ("(s) => s.reasoningBudget", True),
    ("(store) => store.reasoningBudget", True),
    ("(s: RuntimeState) => s.reasoningBudget", True),
    ("(s) => (s.reasoningBudget)", True),
    ("(s) => s.reasoningBudget ?? s.fallback", True),
    # The sheet's own shape, accepted by the contract rather than proved.
    (
        "(s) => s.reasoningBudget === s.loadedReasoningBudget "
        "? (s.loadedReasoningBudgetRequested ?? s.reasoningBudget) : s.reasoningBudget",
        True,
    ),
    ("(s) => formatBudget(s.reasoningBudget)", True),
    # Loose comparison pins nothing: `!= null` is false for both null and undefined.
    ("(s) => s.reasoningBudget != null ? s.reasoningBudget : null", False),
    ("(s) => s.reasoningBudget !== null ? s.reasoningBudget : null", True),
    ("(s) => s.reasoningBudget == 0 ? 0 : s.reasoningBudget", False),
    # Steering between arms is not tracking: the result is the same for every non-null
    # budget, so the sheet never re-renders on a change between two of them.
    ("(s) => s.reasoningBudget !== null ? s.other : null", False),
    # Whitespace inside a literal is part of the value: "a b" and "ab" are two messages.
    ('(s) => s.reasoningBudget === "a b" ? "ab" : s.reasoningBudget', False),
    ('(s) => s.reasoningBudget === "a b" ? "a b" : s.reasoningBudget', True),
    ("(s) => s.reasoningBudget > 0 ? s.other : null", False),
    # The literal has to be compared against the field, not merely to sit in the same guard.
    ('(s) => s.reasoningBudget > 0 && s.mode === "x" ? s.other : s.reasoningBudget', False),
    ('(s) => s.reasoningBudget === -1 && s.mode === "x" ? -1 : s.reasoningBudget', True),
    ('(s) => s.reasoningBudget === -1 || s.mode === "x" ? -1 : s.reasoningBudget', False),
    ("(s) => s.reasoningBudget === -1 && s.enabled || s.override ? -1 : s.reasoningBudget", False),
    ('(s) => s.reasoningBudget !== null && s.mode === "x" ? s.reasoningBudget : null', False),
    # The comparison must be on the field read off the selector's own parameter.
    ("(s) => defaults.reasoningBudget === -1 ? -1 : s.reasoningBudget", False),
    ("(s) => -1 === s.reasoningBudgetExtra ? -1 : s.reasoningBudget", False),
    ("(s) => -1 === s.reasoningBudget ? -1 : s.reasoningBudget", True),
    # A negated condition beside the pin is an ordinary extra condition, not a negated pin.
    ("(s) => s.reasoningBudget === -1 && !s.disabled ? -1 : s.reasoningBudget", True),
    # A comparison under `!` says the opposite of what it reads.
    ("(s) => !(s.reasoningBudget === -1) ? -1 : s.reasoningBudget", False),
    # The literal is read whole: `1e3` is not a pin to 1.
    ("(s) => s.reasoningBudget === 1e3 ? 1 : s.reasoningBudget", False),
    ("(s) => s.reasoningBudget === 1e3 ? 1e3 : s.reasoningBudget", True),
    ("(s) => s.reasoningBudget === 0x10 ? 0x10 : s.reasoningBudget", True),
    # Pins compare by value: rewriting a literal is a reformatting, not a change of meaning.
    ("(s) => s.reasoningBudget === 0x10 ? 16 : s.reasoningBudget", True),
    ("(s) => s.reasoningBudget === 1e3 ? 1000 : s.reasoningBudget", True),
    ("(s) => s.reasoningBudget === 'x' ? \"x\" : s.reasoningBudget", True),
    # `===` cannot separate -0 from 0, but zustand's Object.is can, so zero pins nothing.
    ("(s) => s.reasoningBudget === -0 ? 0 : s.reasoningBudget", False),
    ("(s) => s.reasoningBudget === 0 ? 0 : s.reasoningBudget", False),
    # A `:` or a `?` inside a message is a character, not a ternary operator.
    ('(s) => s.reasoningBudget === "a:b" ? "a:b" : s.reasoningBudget', True),
    ('(s) => s.reasoningBudget === "a?b" ? "a?b" : s.reasoningBudget', True),
    # An escape spells a character; `"\\x61"` and `"a"` are one value.
    ('(s) => s.reasoningBudget === "\\x61" ? "a" : s.reasoningBudget', True),
    ('(s) => s.reasoningBudget === "\\u0061" ? "a" : s.reasoningBudget', True),
    ('(s) => s.reasoningBudget === "\\x61" ? "b" : s.reasoningBudget', False),
    # A template with no substitution is a constant like any other string.
    ("(s) => s.reasoningBudget === `thinking` ? `thinking` : s.reasoningBudget", True),
    ('(s) => s.reasoningBudget === `thinking` ? "thinking" : s.reasoningBudget', True),
    ("(s) => s.reasoningBudget === `thinking` ? `other` : s.reasoningBudget", False),
    # One with a substitution is an expression, not a constant.
    ("(s) => s.reasoningBudget === `a${x}b` ? `a${x}b` : s.reasoningBudget", False),
    # A surrogate pair is one character, written the way JavaScript stores it.
    ('(s) => s.reasoningBudget === "😀" ? "\\uD83D\\uDE00" : s.reasoningBudget', True),
    ('(s) => s.reasoningBudget === "😀" ? "\\uD83D" : s.reasoningBudget', False),
    # A line continuation adds no character, and the literal it sits in still ends the arm.
    ('(s) => s.reasoningBudget === "a\\\nb" ? "a\\nb" : s.reasoningBudget', False),
    ('(s) => s.reasoningBudget === "a\\\nb" ? "zz" : s.reasoningBudget', False),
    ('(s) => s.reasoningBudget === "a\\\nb" ? "ab" : s.reasoningBudget', True),
    # An escaped quote is a character in the message, not the end of the literal.
    ('(s) => s.reasoningBudget === "a\\"b" ? "a\\"b" : s.reasoningBudget', True),
    ('(s) => s.reasoningBudget === "a\\"b" ? "ab" : s.reasoningBudget', False),
    # Distinct values stay distinct, whatever they are spelled with.
    ("(s) => s.reasoningBudget === null ? undefined : s.reasoningBudget", False),
    ("(s) => s.reasoningBudget === true ? 1 : s.reasoningBudget", False),
    # A pinned arm must return the pinned value, or -1 becoming 0 returns 0 both times.
    ("(s) => s.reasoningBudget === -1 ? 0 : s.reasoningBudget", False),
    ("({ reasoningBudget: b }) => b === -1 ? -1 : b", True),
    # A guard cannot rescue a subpath it does not pin: `enabled` false holds s.other steady.
    ("(s) => s.reasoningBudget !== null ? (s.enabled ? s.reasoningBudget : s.other) : null", False),
    # Falling off the end of a block returns undefined, which tracks nothing.
    ("(s) => { if (s.enabled) return s.reasoningBudget; }", False),
    # An exhaustive if/else returns on every path, so there is no fall-through to invent.
    ("(s) => { if (s.enabled) return s.reasoningBudget; else return s.reasoningBudget; }", True),
    (
        "(s) => { if (s.enabled) { return s.reasoningBudget; } "
        "else { return s.reasoningBudget; } }",
        True,
    ),
    ("(s) => { if (s.enabled) return s.reasoningBudget; else return s.other; }", False),
    # `else` binds to the nearest `if`, so the outer one is not exhaustive here.
    (
        "(s) => { if (s.a) { if (s.b) return s.reasoningBudget; } "
        "else return s.reasoningBudget; }",
        False,
    ),
    (
        "(s) => { if (s.a) { if (s.b) return s.reasoningBudget; else return s.reasoningBudget; } "
        "else return s.reasoningBudget; }",
        True,
    ),
    # A loop body may never run, so it is not a path that always returns.
    ("(s) => { while (s.on) return s.reasoningBudget; }", False),
    ("(s) => { { return s.reasoningBudget; } }", True),
    ("(s) => { if (s.enabled) { return s.reasoningBudget; } return s.reasoningBudget; }", True),
    ("(s) => s.reasoningBudget === -1 ? -1 : s.reasoningBudget", True),
    # An operator inside a literal is part of a value: a message may spell `!` or `||`.
    ('(s) => s.reasoningBudget === "a!b" ? "a!b" : s.reasoningBudget', True),
    ('(s) => s.reasoningBudget === "a||b" ? "a||b" : s.reasoningBudget', True),
    ('(s) => s.reasoningBudget === "a&&b" ? "x" : s.reasoningBudget', False),
    # Wrapping the whole conjunction keeps its `&&` above depth zero until the guard is unwrapped.
    ("(s) => (s.reasoningBudget === -1 && s.enabled) ? -1 : s.reasoningBudget", True),
    ("(s) => (s.reasoningBudget === -1 || s.enabled) ? -1 : s.reasoningBudget", False),
    # Parenthesising a comparison, or the value it pins to, is a reformatting and nothing more.
    ("(s) => (s.reasoningBudget === -1) ? -1 : s.reasoningBudget", True),
    ("(s) => s.reasoningBudget === -1 ? (-1) : s.reasoningBudget", True),
    ("(s) => ((s.reasoningBudget === -1)) ? -1 : s.reasoningBudget", True),
    # A negation spelled as a second comparison pins nothing either.
    ("(s) => (s.reasoningBudget === -1) === false ? -1 : s.reasoningBudget", False),
    # Read but not returned: zustand compares results, so these subscribe to something else.
    ("(s) => s.enabled ? s.reasoningBudget : s.fallback", False),
    ("(s) => s.mode === 'x' ? (s.on ? s.reasoningBudget : s.q) : s.reasoningBudget", False),
    # The same shape without brackets: the nested arm owns the first colon, not the outer one.
    ("(s) => s.mode ? s.on ? s.reasoningBudget : s.q : s.reasoningBudget", False),
    ("(s) => s.mode ? s.on ? s.reasoningBudget : s.q : s.other", False),
    ("(s) => s.mode ? s.on ? s.reasoningBudget : s.reasoningBudget : s.reasoningBudget", True),
    ("(s) => s.enabled ? s.other : s.fallback", False),
    # A guard that steers nothing: every arm returns the same expression regardless.
    ("(s) => s.reasoningBudget ? s.other : s.other", False),
    ("(s) => s.reasoningBudget === 1 ? null : null", False),
    ("(s) => s.reasoningBudget ? (s.on ? null : null) : null", False),
    # Block bodies: what is returned, not what is mentioned on the way there.
    ("(s) => { return s.reasoningBudget; }", True),
    # A truthiness test pins nothing: a budget of 0 and a budget of null both land on -1.
    ("(s) => { const v = s.reasoningBudget; return v ? s.reasoningBudget : -1; }", False),
    (
        "(s) => { const v = s.reasoningBudget; return v ? s.reasoningBudget : s.reasoningBudget; }",
        True,
    ),
    ("(s) => { void s.reasoningBudget; return null; }", False),
    ("(s) => { s.reasoningBudget; }", False),
    # A control block is the selector's own scope; a function declared inside it is not.
    ("(s) => { if (s.enabled) { return s.other; } return s.reasoningBudget; }", False),
    ("(s) => { if (s.enabled) { return s.reasoningBudget; } return s.reasoningBudget; }", True),
    ("(s) => { for (const x of s.list) { return s.other; } return s.reasoningBudget; }", False),
    # A defaulted switch whose every arm returns the field leaves no path to undefined.
    (
        '(s) => { switch (s.mode) { case "x": return s.reasoningBudget; '
        "default: return s.reasoningBudget; } }",
        True,
    ),
    ('(s) => { switch (s.mode) { case "x": default: return s.reasoningBudget; } }', True),
    # A nested switch has labels of its own; they belong to it, not to the outer arm list.
    (
        '(s) => { switch (s.mode) { case "x": switch (s.sub) { default: return s.reasoningBudget; } '
        "default: return s.reasoningBudget; } }",
        True,
    ),
    # Fall-through is what any arm does when it neither returns nor breaks, empty or not.
    (
        '(s) => { switch (s.mode) { case "x": sideEffect(); case "y": return s.reasoningBudget; '
        "default: return s.reasoningBudget; } }",
        True,
    ),
    (
        '(s) => { switch (s.mode) { case "x": sideEffect(); default: return s.reasoningBudget; } }',
        True,
    ),
    (
        '(s) => { switch (s.mode) { default: return s.reasoningBudget; case "x": sideEffect(); } }',
        False,
    ),
    # `try` returns whichever way it goes only when both clauses do, or when `finally` does.
    ("(s) => { try { return s.reasoningBudget; } catch { return s.reasoningBudget; } }", True),
    ("(s) => { try { return s.reasoningBudget; } catch (e) { return s.reasoningBudget; } }", True),
    ("(s) => { try { return s.reasoningBudget; } catch { return s.other; } }", False),
    # A `try` alone leaves the throwing path uncovered.
    ("(s) => { try { return s.reasoningBudget; } finally { cleanup(); } }", False),
    # A nested try's clauses are its own: with `enabled` false the outer try falls out.
    (
        "(s) => { try { if (s.enabled) { try { return s.reasoningBudget; } "
        "catch { return s.reasoningBudget; } } } catch { return s.reasoningBudget; } }",
        False,
    ),
    # A loop with a test written as true always enters, so a return inside it always happens.
    ("(s) => { for (;;) return s.reasoningBudget; }", True),
    ("(s) => { while (true) return s.reasoningBudget; }", True),
    # Unless it can break out of itself first.
    ("(s) => { for (;;) { if (s.stop) break; return s.reasoningBudget; } }", False),
    ("(s) => { do { if (s.stop) continue; return s.reasoningBudget; } while (false); }", False),
    # A test read off the store is a condition, not a certainty.
    ("(s) => { while (s.on) return s.reasoningBudget; }", False),
    ("(s) => { for (const x of s.l) return s.reasoningBudget; }", False),
    # `do` is the one loop whose body runs before the test; the others may not run at all.
    ("(s) => { do { return s.reasoningBudget; } while (s.on); }", True),
    ("(s) => { while (s.on) { return s.reasoningBudget; } }", False),
    # A nested switch's `default:` does not make the outer one exhaustive.
    (
        '(s) => { switch (s.mode) { case "x": switch (s.sub) { default: return s.reasoningBudget; } } }',
        False,
    ),
    # `continue` jumps to the surrounding loop's test, so that arm never reaches the next label.
    (
        '(s) => { do { switch (s.mode) { case "x": continue; '
        "default: return s.reasoningBudget; } } while (false); }",
        False,
    ),
    # A nested switch absorbs a `break`, but never a `continue`: that goes to the loop.
    (
        '(s) => { do { switch (s.mode) { case "x": switch (s.sub) { default: continue; } '
        "default: return s.reasoningBudget; } } while (false); }",
        False,
    ),
    # A property named like a keyword is a read, not a nested switch that absorbs the `break`.
    (
        '(s) => { switch (s.mode) { case "x": s.switch; if (s.stop) break; '
        "return s.reasoningBudget; default: return s.reasoningBudget; } }",
        False,
    ),
    # A regex could spell a label, so any `/` outside a string refuses the selector.
    (
        '(s) => { switch (s.mode) { case "x": /default:/.test(s.name); '
        "return s.reasoningBudget; } }",
        False,
    ),
    ("(s) => s.reasoningBudget / 1", False),
    # A comparison collapses the field to a boolean that holds while the field moves.
    ("(s) => s.reasoningBudget > 0", False),
    ("(s) => !s.reasoningBudget", False),
    ("(s) => (s.reasoningBudget !== null)", False),
    # The left operand of `&&` or `||` is returned on its own path, and holds there.
    ("(s) => s.enabled && s.reasoningBudget", False),
    ("(s) => s.reasoningBudget || -1", False),
    ("(s) => s.reasoningBudget && s.reasoningBudget.toString()", True),
    ("(s) => (s.reasoningBudget && s.other) || s.reasoningBudget", False),
    ("(s) => (s.enabled ? s.reasoningBudget : s.other) || s.reasoningBudget", False),
    # A `let` can be reassigned, so it is not inlined as its initializer.
    ("(s) => { let value = s.reasoningBudget; value = s.other; return value; }", False),
    # A `continue` inside a nested loop belongs to that loop, not to the switch.
    (
        '(s) => { switch (s.mode) { case "x": for (const q of s.l) { continue; } '
        "return s.reasoningBudget; default: return s.reasoningBudget; } }",
        True,
    ),
    # A `break` in a plain block still leaves the switch; one inside a loop belongs to the loop.
    (
        '(s) => { switch (s.mode) { case "x": if (s.stop) { break; } return s.reasoningBudget; '
        "default: return s.reasoningBudget; } }",
        False,
    ),
    (
        '(s) => { switch (s.mode) { case "x": for (;;) { break; } return s.reasoningBudget; '
        "default: return s.reasoningBudget; } }",
        True,
    ),
    # An empty LAST label has nothing to fall into, so that value leaves the switch.
    ('(s) => { switch (s.mode) { default: return s.reasoningBudget; case "x": } }', False),
    # Undefaulted: a mode matching nothing falls past the switch and returns undefined.
    ('(s) => { switch (s.mode) { case "x": return s.reasoningBudget; } }', False),
    # `break` leaves the switch with no value, which is the fall-through again.
    ('(s) => { switch (s.mode) { case "x": break; default: return s.reasoningBudget; } }', False),
    (
        '(s) => { switch (s.mode) { case "x": return s.other; default: return s.reasoningBudget; } }',
        False,
    ),
    # A loop header's semicolons do not end a statement, and the body may never run.
    ("(s) => { for (let i = 0; i < s.list.length; i++) return s.reasoningBudget; }", False),
    (
        "(s) => { for (let i = 0; i < s.list.length; i++) return s.reasoningBudget; "
        "return s.reasoningBudget; }",
        True,
    ),
    ("(s) => { const f = (v) => { return v; }; return f(s.reasoningBudget); }", True),
    ("(s) => s.reasoningBudgetMessage", False),
    ("(s) => s.reasoningBudgets", False),
    ("{ budget: state.reasoningBudget }", False),
    # A destructured parameter subscribes to exactly the same field.
    ("({ reasoningBudget }) => reasoningBudget", True),
    ("({ reasoningBudget: budget }) => budget", True),
    ("({ reasoningBudget, loaded }) => (loaded ? reasoningBudget : reasoningBudget)", True),
    ("({ loadedReasoningBudget }) => loadedReasoningBudget", False),
    ("({ reasoningBudget, other }) => other", False),
    # One access, two spellings: the guard steers nothing if both arms mean the same read.
    ('(s) => s.reasoningBudget ? s.other : s["other"]', False),
    ('(s) => s["reasoningBudget"]', True),
    ('(s) => "s.reasoningBudget"', False),
    ("(s) => `${s.reasoningBudget}`", True),
    ("(s) => `\\${s.reasoningBudget}`", False),
    ('(s) => `${"s.reasoningBudget"}`', False),
    ("(s) => { const value = s.reasoningBudget; return s.other.value; }", False),
    ("(s) => { const value = s.reasoningBudget; return value; }", True),
    ("(s) => { const value = s.reasoningBudget; return { value: s.other }; }", False),
    (
        "(s) => { const value = s.reasoningBudget; { const value = s.other; return value; } }",
        False,
    ),
    ("(s) => { const v = s.reasoningBudget; const f = (v) => v; return f(s.other); }", False),
    # A local that shadows the parameter is not the store.
    (
        "({ reasoningBudget, enabled }) => { if (enabled) { const reasoningBudget = 1; "
        "return reasoningBudget; } else { const reasoningBudget = 2; return reasoningBudget; } }",
        False,
    ),
    ("(s) => { { const s = { reasoningBudget: 1 }; return s.reasoningBudget; } }", False),
    ("({ reasoningBudget }) => s.other.reasoningBudget", False),
    ("(s) => { const value = s.reasoningBudget; return s.other. value; }", False),
    # A comma expression returns its last operand, and a declarator list declares two names.
    ("(s) => (s.reasoningBudget, s.other)", False),
    ("(s) => (s.other, s.reasoningBudget)", True),
    ("(s) => { return s.reasoningBudget, s.other; }", False),
    ("(s) => { const value = s.other, ignored = s.reasoningBudget; return value; }", False),
    ("(s) => `${/* s.reasoningBudget */ 1}`", False),
    # Automatic semicolon insertion: the loop ends at `while (...)`, not at the next `;`.
    (
        '(s) => { switch (s.mode) { case "x": do {} while (false)\n if (s.stop) break; '
        "return s.reasoningBudget; default: return s.reasoningBudget; } }",
        False,
    ),
    (
        '(s) => { switch (s.mode) { case "x": sideEffect()\n if (s.stop) break; '
        "return s.reasoningBudget; default: return s.reasoningBudget; } }",
        False,
    ),
    (
        '(s) => { switch (s.mode) { case "x": const o = { switch: 1 }; if (s.stop) break; '
        "return s.reasoningBudget; default: return s.reasoningBudget; } }",
        False,
    ),
    (
        '(s) => { switch (s.mode) { case "x": s. switch; if (s.stop) break; '
        "return s.reasoningBudget; default: return s.reasoningBudget; } }",
        False,
    ),
    ("(s) => s .reasoningBudget", True),
    ('(s) => s.enabled ? "s.reasoningBudget" : s.reasoningBudget', False),
    # Inside a literal a bracket is part of the value, not an access to rewrite.
    ('(s) => s.reasoningBudget === \'s["x"]\' ? "s.x" : s.reasoningBudget', False),
    ("(s) => s.reasoningBudget === 's[\"x\"]' ? 's[\"x\"]' : s.reasoningBudget", True),
    # A comment is not part of the value an arm returns.
    ("(s) => s.reasoningBudget ? s.other : /* same value */ s.other", False),
    ("(s) => s.reasoningBudget ? s.other : s.another // differs", False),
    ("(s) => s.reasoningBudget /* the effective one */", True),
    ("(s) => s.enabled ? s.reasoningBudget : /* same */ s.reasoningBudget", True),
    # A helper's own return is not what zustand compares.
    ("(s) => { function n(v) { return v ?? -1; } return n(s.reasoningBudget); }", True),
    ("(s) => { function n(v) { return v ?? -1; } return n(s.other); }", False),
]


def test_the_subscription_predicate_accepts_refactors_and_rejects_non_subscriptions():
    for selector, expected in SELECTOR_CASES:
        assert _selector_reads(selector, "reasoningBudget") is expected, selector


def test_preset_sheet_reacts_to_a_reasoning_budget_change():
    """capturePresetLoadConfig() reads the runtime store through getState().

    A captured field the sheet neither subscribes to nor lists as a memo dependency
    cannot move the Update button or the summary: with the sheet open, changing only
    the reasoning budget left both stale until some unrelated setting changed.

    Asserted as behaviour, not spelling. Requiring the literal `(s) => s.reasoningBudget` at
    two fixed indentations made af4e98e2f red for writing the same subscription as a
    multi-line conditional. A guard a legal refactor breaks says nothing about what it guards.

    Source alone cannot prove the returned value is distinct for distinct field values; a
    selector mapping every budget to one constant would pass here.
    """
    sheet = _read("studio/frontend/src/features/chat/chat-settings-sheet.tsx")
    # The component that holds the capture memos, not the module: its siblings subscribe to the
    # runtime store too, and only this one's re-render moves the Update button and the summary.
    # Comments out before anything is discovered, not just inside the arms: a commented-out
    # `// useChatRuntimeStore((s) => s.reasoningBudget)` left beside a selector repointed at
    # another field is still a call as far as a text scan is concerned, and the marker sits
    # outside the extracted argument, so stripping later cannot reach it.
    panel = _without_comments(_component_body(sheet, "ChatSettingsPanel"))
    selectors = _store_selectors(panel)
    assert selectors, "ChatSettingsPanel makes no useChatRuntimeStore() call"
    dependency_lists = _memo_dependency_lists(panel)
    # The two memos that call capturePresetLoadConfig(): the dirty state behind the Update
    # button, and the summary. Named, so the field cannot leave one for an unrelated memo.
    capturing = ("hasUnsavedPresetChanges", "currentLoadSummary")
    for memo in capturing:
        assert memo in dependency_lists, (
            f"no `const {memo} = useMemo(...)` with a dependency list in the sheet; if it was "
            "renamed, rename it here too, and check it still captures the preset config"
        )

    for field in ("reasoningBudget", "reasoningBudgetMessage"):
        assert any(_selector_reads(text, field) for text in selectors), (
            f"no useChatRuntimeStore selector returns a value derived from {field}, so a "
            "change to it does not re-render the component whose memos capture it"
        )
        for memo in capturing:
            assert field in dependency_lists[memo], (
                f"{memo} captures {field} through capturePresetLoadConfig() but does not list "
                f"it as a dependency, so it keeps a value computed before {field} changed"
            )


def test_a_preset_records_a_self_sizing_load_s_pin_and_not_its_window():
    """A preset must reproduce the setup that ran. A window nobody pinned is reached again
    on replay, so only the pin is stored -- in the one field that means "pinned"."""
    source = _read("studio/frontend/src/features/chat/presets/preset-load-config.ts")
    body = source[source.index("export function capturePresetLoadConfig") :]
    body = body[: body.index("\n}\n")]
    # Here: capture asks the rule, off classifiers rather than constants, and bounds it.
    assert "requestableContextLength(\n    capturedContextLength(" in body, body
    assert "isServedByLlamaCpp({\n    loadedIsGguf: store.loadedIsGguf," in body, body
    assert not re.search(r"LlamaCpp\(\{[^}]*?(\w+): (?!store\.[\w.]*\1,)", body), body
    assert "isServedByMlx(isGguf, platform.deviceType, platform.chatOnlyReason)" in body, body
    # Compared as a pin, not as a field: another backend holds it in the other field.
    compare = source[source.index("function toComparablePerModelConfig(") :]
    compare = compare[: compare.index("\n}\n")]
    assert "const pin = savedContextPin(config);" in compare, compare
    assert "customContextLength: pin,\n    maxSeqLength: null," in compare, compare
    # Both bounds, in the one rule capture and storage share, else the replays disagree.
    rule = source[source.index("function requestableContextLength(") :]
    rule = rule[: rule.index("\n}\n")]
    assert "Math.min(MAX_SEQ_LENGTH_MAX, Math.max(CONTEXT_LENGTH_MIN," in rule, rule
    assert source.count("requestableContextLength(") == 3, source
    assert "loadedContextLength: store.loadedContextLength," in body, body
    assert "controlPin: snapshot.customContextLength," in body, body
    # A self-sized window is never captured: bounded, it replays a wider one narrower.
    assert re.search(r"capturedContextLength\(\{\n\s*isGguf,\n\s*controlPin:", body), body
    assert "maxSeqLength: isMlx ? null : normalizeMaxSeqLength(" in body, body
