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


def _store_selectors(source: str) -> list:
    """Every `useChatRuntimeStore(...)` argument, whatever shape the selector takes."""
    out, needle = [], "useChatRuntimeStore("
    start = source.find(needle)
    while start != -1:
        out.append(_balanced(source, start + len(needle) - 1))
        start = source.find(needle, start + 1)
    return out


def _split_ternary(
    expression: str,
    guards: tuple = (),
) -> list:
    """`cond ? a : b` as `[(a, conditions), (b, conditions)]`, recursively.

    Each result is paired with every condition governing whether it is the one returned, so a
    caller can tell a constant the field decides from a constant it has no say in. Depth aware,
    and `??` / `?.` are not ternaries.
    """

    # A wholly wrapped arm hides its own ternary at depth 1, where the scan below never looks.
    expression = expression.strip()
    while expression.startswith("(") and _balanced(expression, 0) == expression[1:-1]:
        expression = expression[1:-1].strip()

    depth, question = 0, -1
    index = 0
    while index < len(expression):
        char = expression[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char == "?":
            following = expression[index + 1 : index + 2]
            if following in ("?", "."):
                index += 2
                continue
            question = index
            break
        index += 1
    if question == -1:
        return [(expression.strip(), guards)]

    # An unparenthesised nested ternary in the true arm owns the next colon, so count `?` here
    # too: taking the first one at bracket depth 0 cuts `a ? b ? c : d : e` into `a ? b` and
    # `d : e`, and a field named anywhere in that second blob would look like every arm reading it.
    inner = guards + (expression[:question],)
    depth, nested = 0, 0
    index = question + 1
    while index < len(expression):
        char = expression[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char == "?":
            if expression[index + 1 : index + 2] in ("?", "."):
                index += 2
                continue
            nested += 1
        elif depth == 0 and char == ":":
            if not nested:
                return _split_ternary(expression[question + 1 : index], inner) + _split_ternary(
                    expression[index + 1 :], inner
                )
            nested -= 1
        index += 1
    return [(expression[question + 1 :].strip(), inner)]


def _selector_reads(selector: str, field: str) -> bool:
    """Does every value this selector can return depend on `field`?

    Zustand re-renders on the RESULT, not on a property the selector happened to touch, so one
    that tests `s.<field>` and returns something else either way tracks nothing. A result the
    field does not appear in still counts when the field decides whether it is returned at all,
    as in `s.<field> != null ? s.<field> : null`. The parameter name comes from the signature
    rather than being assumed to be `s`.
    """

    signature = re.match(r"\s*\(?\s*([A-Za-z_$][\w$]*)\s*\)?\s*(?::[^=]*)?=>", selector)
    if signature is None:
        return False
    read = re.compile(rf"\b{re.escape(signature.group(1))}\.{field}\b")
    results = _split_ternary(selector[signature.end() :])
    if not results:
        return False
    if all(read.search(result) for result, _ in results):
        return True
    # A guard only justifies an arm it can steer away from: if every arm returns the same
    # expression the condition decides nothing, so `s.budget ? s.other : s.other` subscribes
    # to `s.other` however prominently it names the budget.
    if len({re.sub(r"\s+", "", result) for result, _ in results}) < 2:
        return False
    return all(
        read.search(result) or any(read.search(guard) for guard in guards)
        for result, guards in results
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
    ("(s) => formatBudget(s.reasoningBudget)", True),
    # A constant arm the field itself decides between still moves when the field moves.
    ("(s) => s.reasoningBudget != null ? s.reasoningBudget : null", True),
    ("(s) => s.reasoningBudget != null ? s.other : null", True),
    ("(s) => s.reasoningBudget === -1 ? -1 : s.reasoningBudget", True),
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
    ("(s) => s.reasoningBudgetMessage", False),
    ("(s) => s.reasoningBudgets", False),
    ("{ budget: state.reasoningBudget }", False),
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
    selectors = _store_selectors(sheet)
    assert selectors, "no useChatRuntimeStore() call found; has the sheet been renamed?"
    dependency_lists = _memo_dependency_lists(sheet)
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
