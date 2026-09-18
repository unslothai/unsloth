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


def _without_comments(source: str) -> str:
    """`//` and `/* */` removed, leaving string literals alone.

    A comment is not part of the value an arm returns, so two arms that differ only by one are
    the same expression and the condition between them steers nothing.
    """
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


_RESERVED = frozenset(
    "await break case catch class const continue debugger default delete do else export extends "
    "finally for function if import in instanceof let new return super switch this throw try "
    "typeof var void while with yield".split()
)
_LITERAL_NAMES = frozenset({"null", "undefined", "true", "false"})
_TOKEN = re.compile(
    r"\s*(?:(?P<string>'[^'\\\n]*'|\"[^\"\\\n]*\")"
    r"|(?P<number>\d+(?:\.\d+)?(?![\w$.]))"
    r"|(?P<name>[A-Za-z_$][\w$]*)"
    r"|(?P<op>\?\?|\?\.(?!\d)|===|!==|==|!=|<=|>=|&&|\|\||[-?:()\[\].,<>!]))"
)
# Operators that end the value a selector hands zustand at a comparison or a branch: an arm
# holding one at its own level returns a boolean, or picks between values unscored.
_COLLAPSING = frozenset({"===", "!==", "==", "!=", "<", ">", "<=", ">=", "&&", "||", "!", "?"})
_UNARY_AFTER = _COLLAPSING | {None, ":", "(", ",", "??"}


def _tokens(expression: str):
    """`expression` as (kind, text) tokens, or None if it steps outside what is read here.

    Reads, calls, literals, ternaries, `??` and comparisons only. Assignment, arrow functions,
    blocks, templates, regexes, escapes, arithmetic and the comma operator are refused rather
    than modelled, which leaves nothing that can bind, write or hide a statement. Parentheses
    come back as "call" or "group"; `s["x"]` comes back as `s.x`.
    """
    raw, index, end = [], 0, len(expression.rstrip())
    while index < end:
        match = _TOKEN.match(expression, index)
        if match is None:
            return None
        raw.append((match.lastgroup, match.group(match.lastgroup)))
        index = match.end()
    out, stack = [], []
    index = 0
    while index < len(raw):
        kind, text = raw[index]
        previous = out[-1] if out else (None, None)
        member = previous[1] in (".", "?.")
        if kind == "name" and text in _RESERVED and not member:
            return None
        if text == "-":
            # Only a sign on a number literal: arithmetic is outside the subset.
            following = raw[index + 1] if index + 1 < len(raw) else (None, None)
            if previous[1] not in _UNARY_AFTER or following[0] != "number":
                return None
            out.append(("number", "-" + following[1]))
            index += 2
            continue
        if text == "[":
            # Only `x["key"]`, a member access spelled with a string. An array is refused.
            if index + 2 >= len(raw) or raw[index + 2][1] != "]" or raw[index + 1][0] != "string":
                return None
            key = raw[index + 1][1][1:-1]
            if previous[0] != "name" and previous[1] != ")":
                return None
            if not re.fullmatch(r"[A-Za-z_$][\w$]*", key):
                return None
            out += [("op", "."), ("name", key)]
            index += 3
            continue
        if text == "(":
            callee = previous[0] == "name" and previous[1] not in _LITERAL_NAMES
            stack.append("call" if callee or previous[1] == ")" else "group")
            out.append((stack[-1], "("))
        elif text == ")":
            if not stack:
                return None
            out.append((stack.pop(), ")"))
        elif text == "," and (not stack or stack[-1] != "call"):
            return None
        else:
            out.append((kind, text))
        index += 1
    return out if not stack else None


def _unwrapped(tokens: list) -> list:
    """`tokens` without grouping parentheses that enclose all of them."""
    while tokens and tokens[0] == ("group", "("):
        depth = 0
        for index, (kind, text) in enumerate(tokens):
            depth += text == "(" and kind in ("call", "group")
            depth -= text == ")" and kind in ("call", "group")
            if depth == 0:
                break
        if index != len(tokens) - 1:
            return tokens
        tokens = tokens[1:-1]
    return tokens


def _top_level(tokens: list):
    """(index, text) of each token outside every parenthesis."""
    depth = 0
    for index, (kind, text) in enumerate(tokens):
        if kind in ("call", "group"):
            depth += 1 if text == "(" else -1
        elif depth == 0:
            yield index, text


def _paths(tokens: list, guards: tuple = ()) -> list:
    """Each value the expression can return, with the (guard, taken) pairs that lead to it."""
    tokens = _unwrapped(tokens)
    marks = [index for index, text in _top_level(tokens) if text in ("?", ":")]
    if not marks or tokens[marks[0]][1] != "?":
        return [(tokens, guards)]
    # The colon that closes the first `?` is the first one no nested `?` is still waiting for.
    open_questions = 0
    for index in marks:
        open_questions += 1 if tokens[index][1] == "?" else -1
        if open_questions == 0:
            condition = tokens[: marks[0]]
            return _paths(tokens[marks[0] + 1 : index], guards + ((condition, True),)) + _paths(
                tokens[index + 1 :], guards + ((condition, False),)
            )
    return [([], guards)]


def _value(token):
    """What a literal token denotes, or None for anything else."""
    kind, text = token
    if kind == "number":
        return float(text)
    if kind == "string":
        return ("string", text[1:-1])
    return text if kind == "name" and text in _LITERAL_NAMES else None


def _pinned(guard: list, taken: bool, access: list):
    """The one value the field holds where `guard` went this way, or None.

    `budget === -1` taken, or `budget !== -1` not taken, pins -1. Anything looser (`==`, `>`,
    a disjunction, a negation) holds the field to more than one value and pins nothing. Zero
    pins nothing either: `=== 0` also takes -0, which zustand's Object.is tells apart.
    """
    guard = _unwrapped(guard)
    if taken:
        if any(text == "||" for _, text in _top_level(guard)):
            return None
        cuts = [-1] + [index for index, text in _top_level(guard) if text == "&&"] + [len(guard)]
        conjuncts = [_unwrapped(guard[a + 1 : b]) for a, b in zip(cuts, cuts[1:])]
    else:
        conjuncts = [guard]
    operator = "===" if taken else "!=="
    size = len(access)
    for conjunct in conjuncts:
        if len(conjunct) != size + 2:
            continue
        if conjunct[:size] == access and conjunct[size] == ("op", operator):
            literal = conjunct[size + 1]
        elif conjunct[2:] == access and conjunct[1] == ("op", operator):
            literal = conjunct[0]
        else:
            continue
        value = _value(literal)
        if value is not None and value != 0.0:
            return value
    return None


def _reads_field(arm: list, access: list) -> bool:
    """Does `arm` return the field itself, or a call on it?

    The field has to be read off the parameter, not off some other object, and be the value
    itself: `.length` after it, a comparison, a logical operator or a branch anywhere in the arm
    returns something the field does not decide. A call is the one transform taken on trust,
    as source cannot see into it.
    """
    if any(text in _COLLAPSING for _, text in arm):
        return False
    size = len(access)
    for index in range(len(arm) - size + 1):
        if arm[index : index + size] != access:
            continue
        before = arm[index - 1][1] if index else None
        after = arm[index + size] if index + size < len(arm) else None
        if before in (".", "?."):
            continue
        if after is None or after[1] in (",", "??") or after == ("call", ")"):
            return True
    return False


def _selector_reads(selector: str, field: str) -> bool:
    """Does every value this selector can return depend on `field`?

    Zustand re-renders on the RESULT, so a selector that tests the field and returns something
    else tracks nothing. Every path has to return the field (or a call on it), or return
    exactly the literal a guard pins the field to on that path, as in
    `s.budget === -1 ? -1 : s.budget`.

    Only the subset `_tokens` accepts is read, and anything else is refused. That is
    deliberate: a refusal fails this test loudly and asks for a plainer spelling, while
    modelling more JavaScript is how a stale selector slips through.
    """
    selector = _without_comments(selector)
    plain = re.match(r"\s*\(?\s*([A-Za-z_$][\w$]*)\s*\)?\s*(?::[^=]*)?=>", selector)
    destructured = re.match(r"\s*\(?\s*\{([^}]*)\}\s*\)?\s*(?::[^=]*)?=>", selector)
    if plain is not None:
        access = [("name", plain.group(1)), ("op", "."), ("name", field)]
        body = selector[plain.end() :]
    elif destructured is not None:
        locals_ = {}
        for entry in destructured.group(1).split(","):
            name, _, alias = entry.partition(":")
            locals_[name.strip()] = alias.strip() or name.strip()
        if field not in locals_:
            return False
        access = [("name", locals_[field])]
        body = selector[destructured.end() :]
    else:
        return False
    # The argument list's trailing comma rides along on a multi-line call.
    body = body.strip().removesuffix(",").strip()
    if body.startswith("{"):
        # A single `return` only. A line break straight after `return` returns undefined.
        match = re.fullmatch(r"\{\s*return(?![\w$])[ \t]*(?=[^\s;}])([^;{}]*?)\s*;?\s*\}", body)
        if match is None:
            return False
        body = match.group(1)
    tokens = _tokens(body)
    if not tokens:
        return False
    return all(
        _reads_field(arm, access)
        or (
            len(arm) == 1
            and _value(arm[0]) is not None
            and any(_pinned(guard, taken, access) == _value(arm[0]) for guard, taken in guards)
        )
        for arm, guards in _paths(tokens)
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
    ("s => s.reasoningBudget", True),
    ("(s) => (s.reasoningBudget)", True),
    ('(s) => s["reasoningBudget"]', True),
    ("(s) => s.reasoningBudget /* the effective one */", True),
    ("(s) => s.reasoningBudget ?? s.fallback", True),
    ("(s) => formatBudget(s.reasoningBudget)", True),
    ("(s) => formatBudget(s.reasoningBudget, 2)", True),
    ("(s) => String(s.reasoningBudget)", True),
    ("(s) => { return s.reasoningBudget; }", True),
    ("(s) => { return s.enabled ? s.reasoningBudget : s.reasoningBudget }", True),
    ("(s) => s.reasoningBudget,", True),
    # The sheet's own shape. The `??` arm is accepted by the contract, not proved: while the
    # budget equals the loaded one it returns the requested value (see the sheet test).
    (
        "(s) => s.reasoningBudget === s.loadedReasoningBudget "
        "? (s.loadedReasoningBudgetRequested ?? s.reasoningBudget) : s.reasoningBudget",
        True,
    ),
    # A destructured parameter subscribes to exactly the same field.
    ("({ reasoningBudget }) => reasoningBudget", True),
    ("({ reasoningBudget: budget }) => budget", True),
    ("({ reasoningBudget, loaded }) => (loaded ? reasoningBudget : reasoningBudget)", True),
    ("({ loadedReasoningBudget }) => loadedReasoningBudget", False),
    ("({ reasoningBudget, other }) => other", False),
    ("(s) => s.reasoningBudgetMessage", False),
    ("(s) => s.reasoningBudgets", False),
    ("(s) => other.s.reasoningBudget", False),
    ("({ reasoningBudget }) => s.other.reasoningBudget", False),
    ('(s) => "s.reasoningBudget"', False),
    ("{ budget: state.reasoningBudget }", False),
    # Read but not returned: zustand compares results, so these subscribe to something else.
    ("(s) => s.enabled ? s.reasoningBudget : s.fallback", False),
    ("(s) => s.mode ? s.on ? s.reasoningBudget : s.q : s.reasoningBudget", False),
    ("(s) => s.mode ? s.on ? s.reasoningBudget : s.reasoningBudget : s.reasoningBudget", True),
    ("(s) => s.reasoningBudget ? s.other : s.other", False),
    ("(s) => s.reasoningBudget > 0 ? s.other : null", False),
    ("(s) => s.reasoningBudget !== null ? s.other : null", False),
    ("(s) => (s.enabled ? s.other : s.reasoningBudget).toString()", False),
    ("(s) => String(s.enabled ? s.reasoningBudget : s.nBatch)", False),
    # A property of the field is another value: two messages of one length compare equal.
    ("(s) => s.reasoningBudget.length", False),
    ("(s) => s.reasoningBudget.toString()", False),
    ("(s) => (s.reasoningBudget).length", False),
    # A constant arm counts only when a guard pins the field to that very value there.
    ("(s) => s.reasoningBudget === -1 ? -1 : s.reasoningBudget", True),
    ("(s) => -1 === s.reasoningBudget ? -1 : s.reasoningBudget", True),
    ("(s) => s.reasoningBudget !== null ? s.reasoningBudget : null", True),
    ("(s) => s.reasoningBudget === 'x' ? \"x\" : s.reasoningBudget", True),
    ('(s) => s.reasoningBudget === -1 && s.mode === "x" ? -1 : s.reasoningBudget', True),
    ("({ reasoningBudget: b }) => b === -1 ? -1 : b", True),
    # -1 becoming 0 returns 0 both times.
    ("(s) => s.reasoningBudget === -1 ? 0 : s.reasoningBudget", False),
    # Loose, negated, disjoined or nested comparisons hold the field to more than one value.
    ("(s) => s.reasoningBudget != null ? s.reasoningBudget : null", False),
    ("(s) => s.reasoningBudget == 0 ? 0 : s.reasoningBudget", False),
    ("(s) => !(s.reasoningBudget === -1) ? -1 : s.reasoningBudget", False),
    ('(s) => s.reasoningBudget === -1 || s.mode === "x" ? -1 : s.reasoningBudget', False),
    ("(s) => s.reasoningBudget === -1 && s.on || s.x ? -1 : s.reasoningBudget", False),
    ("(s) => (s.reasoningBudget === -1) === false ? -1 : s.reasoningBudget", False),
    ("(s) => defaults.reasoningBudget === -1 ? -1 : s.reasoningBudget", False),
    # `=== 0` also takes -0, which zustand's Object.is tells apart from 0.
    ("(s) => s.reasoningBudget === 0 ? 0 : s.reasoningBudget", False),
    # A comparison or a logical operator returns something the field does not decide.
    ("(s) => s.reasoningBudget > 0", False),
    ("(s) => !s.reasoningBudget", False),
    ("(s) => s.enabled && s.reasoningBudget", False),
    ("(s) => s.reasoningBudget || -1", False),
    # A line break after `return` returns undefined.
    ("(s) => { return\ns.reasoningBudget; }", False),
    # Outside the subset read here, so refused rather than guessed at: statements, bindings,
    # writes, nested functions, the comma operator, templates, regexes, escapes, arithmetic,
    # a second call argument.
    ("(s) => { if (s.enabled) return s.reasoningBudget; return s.reasoningBudget; }", False),
    ("(s) => { const v = s.reasoningBudget; return v; }", False),
    ("(s) => { switch (s.mode) { default: return s.reasoningBudget; } }", False),
    ("(s) => (s = other).reasoningBudget", False),
    ("(s) => ((s) => s.reasoningBudget)(other)", False),
    ("(s) => (s.reasoningBudget, s.other)", False),
    ("(s) => `${s.reasoningBudget}`", False),
    ("(s) => /x/.test(s.name) ? s.other : s.reasoningBudget", False),
    ('(s) => s.reasoningBudget === "a\\nb" ? "a\\nb" : s.reasoningBudget', False),
    ("(s) => s.reasoningBudget / 1", False),
    ("(s) => s.reasoningBudget === 1e3 ? 1e3 : s.reasoningBudget", False),
    ("(s) => ({ budget: s.reasoningBudget })", False),
    ("(s) => [s.reasoningBudget][0]", False),
    ("(s) => s.reasoningBudget, shallow", False),
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
