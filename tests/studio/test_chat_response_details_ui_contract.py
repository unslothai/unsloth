# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contract for the chat response-details action and metadata."""

from __future__ import annotations

import re
from pathlib import Path

from _en_catalog import en_string

REPO = Path(__file__).resolve().parents[2]
THREAD_TSX = REPO / "studio/frontend/src/components/assistant-ui/thread.tsx"
MESSAGE_MENU_TIME_TSX = REPO / "studio/frontend/src/components/assistant-ui/message-menu-time.tsx"
DETAILS_TSX = (
    REPO / "studio/frontend/src/components/assistant-ui/message-response-details-sheet.tsx"
)
SHEET_TSX = REPO / "studio/frontend/src/components/ui/sheet.tsx"
REASONING_TSX = REPO / "studio/frontend/src/components/assistant-ui/reasoning.tsx"
ADAPTER_TS = REPO / "studio/frontend/src/features/chat/api/chat-adapter.ts"
CHAT_PREFS_TS = REPO / "studio/frontend/src/features/chat/stores/chat-preferences-store.ts"
CHAT_TAB_TSX = REPO / "studio/frontend/src/features/settings/tabs/chat-tab.tsx"
EN_LOCALE_TS = REPO / "studio/frontend/src/i18n/locales/en.ts"


# Distinct from None (no className): an unreadable className must not quietly pass.
_UNREADABLE = "\x00unreadable"


def _class_list(source: str, marker: str) -> str | None:
    """Comments are stripped first, or a commented-out element matching the marker is read instead."""
    source = _without_block_comments(source)
    start = source.find(marker)
    if start == -1:
        return None
    # Back up to the opening `<`: className may be written before the marker.
    opens = source.rfind("<", 0, start + len(marker))
    if opens == -1:
        return None
    # Only a `>` outside JSX expression braces closes the tag (`onClick={() => ...}`).
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
    # A spread after className makes the class list unknown; one before it is overridden.
    if _spread_overrides(opening, "className"):
        return _UNREADABLE
    # Anchored so `data-className=` does not match on its suffix.
    literal = re.search(r'(?:^|[\s{])className="([^"]*)"', opening)
    if literal:
        return literal.group(1)
    # Returning None for an unreadable expression would make the caller pass on base classes.
    expression = re.search(r"(?:^|[\s{])className=\{", opening)
    if not expression:
        return None
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
    # Only plain string literals inside one cn(...): anything else makes the list unknown.
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
    """Precedence is not decided: exactly one unqualified min-w-0 passes, and anything else is refused."""
    # min-w-0 is not enough: `shrink-0` or `flex-none` still stops the trigger shrinking.
    pinned = [
        token
        for token in tokens
        # `grow-0` leaves flex-shrink alone, so it is not refused.
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
    assert set(widths) == {"min-w-0"}, (
        f"{what} carries min-width utilities this guard will not adjudicate between: "
        f"{sorted(set(widths))}. Which one wins is tailwind-merge's answer and then the cascade's, and "
        f"getting that wrong in either direction is worse than refusing: a variant-qualified "
        f"or important min-width, or more than one of them, has to be reduced to a single "
        f"unqualified min-w-0 for this to pass. {evidence}"
    )


# `[min-width:...]` sets the same property; matched before variants are stripped.
_ARBITRARY_MIN_WIDTH = re.compile(r"(?:^|:)!?\[min-width:[^\]]*\]!?$")


def _is_min_width(token: str) -> bool:
    if _ARBITRARY_MIN_WIDTH.search(token.removeprefix("!")):
        return True
    # Arbitrary values can contain `:`, so bracketed spans are masked before the variant split.
    masked = re.sub(r"\[[^\]]*\]", lambda found: "\x00" * len(found.group(0)), token)
    _, _, tail = masked.rpartition(":")
    utility = token[len(masked) - len(tail) :]
    return utility.removeprefix("!").removesuffix("!").startswith("min-w-")


def _cn_literals(source: str, anchor: str) -> str | None:
    """Comments are removed first, or a dead cn(...) in a block comment matches before the live one."""
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
    # Without a forwarded className the call site's classes reach nothing.
    if not forwards:
        return _UNREADABLE
    return " ".join(pieces)


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
    """Brace-aware: a `>` inside an earlier prop's arrow function would truncate a [^>]* match."""
    opens = source.find(marker)
    if opens == -1:
        return None
    depth = 0
    for index in range(opens, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
        elif source[index] == ">" and depth == 0:
            return _without_comments(source[opens : index + 1])
    return None


def _without_block_comments(source: str) -> str:
    """Line comments count too: a stale `// <ReasoningBody>` would be found and read as live."""
    source = re.sub(r"\{?\s*/\*.*?\*/\s*\}?", " ", source, flags = re.S)
    return "\n".join(re.sub(r"(?<!:)//.*$", "", line) for line in source.splitlines())


def _spread_overrides(tag: str, attribute: str) -> bool:
    """JSX takes the last write, so only a spread placed after the attribute can override it."""
    # Only spreads at the tag's own attribute level can reach className.
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
    """Commented-out props are removed, since every check is a substring match that would count them."""
    kept = [line for line in tag.splitlines() if not line.lstrip().startswith("//")]
    return re.sub(r"/\*.*?\*/", " ", "\n".join(kept), flags = re.S)


def test_assistant_more_menu_exposes_response_details_action():
    """Matches today's wiring literally: a refactor that respells it must update this test."""
    src = _without_block_comments(THREAD_TSX.read_text(encoding = "utf-8"))
    assert "MessageResponseDetailsSheet" in src
    assert re.search(
        r"""import\s*\{\s*MessageMenuTime\s*\}\s*from\s*["']@/components/assistant-ui/message-menu-time["']""",
        src,
    ), "thread.tsx no longer imports MessageMenuTime from message-menu-time.tsx"
    caller = _opening_tag(src, "<MessageMenuTime ")
    assert (
        caller
        and re.match(
            r"<MessageMenuTime\s+onShowDetails=\{\s*\(\)\s*=>\s*setDetailsOpen\(\s*true\s*\)\s*\}",
            caller,
        )
    ), f"thread.tsx no longer hands MessageMenuTime a callback that opens the details sheet: {caller}"
    assert not _spread_overrides(caller, "onShowDetails"), caller
    sheet = _opening_tag(src, "<MessageResponseDetailsSheet")
    assert sheet and re.search(r"(?<![\w-])open=\{\s*detailsOpen\s*\}", sheet), sheet
    assert not _spread_overrides(sheet, "open"), sheet
    menu = _without_block_comments(MESSAGE_MENU_TIME_TSX.read_text(encoding = "utf-8"))
    item = _opening_tag(menu, "<ActionBarMorePrimitive.Item")
    assert item, "message-menu-time.tsx no longer renders a More-menu item"
    assert re.search(r'(?<![\w-])aria-label="See response details"', item), item
    assert re.search(r"(?<![\w-])onSelect=\{\s*onShowDetails\s*\}", item), item
    assert not _spread_overrides(item, "aria-label"), item
    assert not _spread_overrides(item, "onSelect"), item
    disabled = re.search(r"(?<![\w-])disabled(?:=\{\s*([^{}]*?)\s*\})?(?=[\s/>])", item)
    assert disabled is None or disabled.group(1) == "false", item


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
    # By key, not wording: the English label text has changed before.
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
    assert "group-hover/assistant-message:pointer-events-auto" in details_src
    assert "group-focus-within/assistant-message:pointer-events-auto" in details_src
    assert thread_src.count("<MessageResponseModelBadge") == 1
    assert "hasReasoningParts" not in thread_src
    assert "group/assistant-message aui-assistant-message-root" in thread_src
    assert "pointer-events-none relative h-0" in thread_src
    assert "MessageResponseModelBadge" not in reasoning_src
    # Whole tokens on the element's className, base and call site both read: `md:min-w-0`
    # or `data-className` would otherwise pass while the trigger cannot shrink.
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
    # _class_list returns None for an absent call site too, so check it renders at all.
    assert "<ReasoningTrigger" in _without_block_comments(reasoning_src), (
        "ReasoningTrigger is no longer rendered, so the shrinking this test is about belongs "
        "to an element that is not on the page"
    )
    # An inline style beats every utility, so a `minWidth` style is refused.
    header_tags = [
        tag for tag in _opening_tags(live, "<div") if 'data-slot="reasoning-header"' in tag
    ]
    assert header_tags, (
        'reasoning.tsx no longer renders a div with data-slot="reasoning-header", so this '
        "guard cannot tell which element the trigger has to shrink inside"
    )
    # The header is a plain data-slot div. The base Trigger forwards `{...props}` by design,
    # so spreads are refused only on the call site and header.
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
    # cn runs tailwind-merge, so the LAST min-w-* wins.
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
    """A manual open outranks the visibility preference for one round, so a new stream must clear it."""
    src = _without_block_comments(REASONING_TSX.read_text(encoding = "utf-8"))

    # The override is identified as the value given to resolveReasoningOpen, not by name.
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

    handler = re.search(
        r"const handleOpenChange = useCallback\(\s*\(open: boolean\) => \{(.*?)\n    \}",
        src[src.index("const [override,") :],
        re.S,
    )
    assert handler and f"{writes}(open);" in handler.group(
        1
    ), "the hand toggle must store the requested open state directly"

    # Regenerate reuses this component instance, so a new round must clear the override.
    round_at = src.find("startsNewReasoningRound(")
    assert round_at != -1, (
        "reasoning.tsx no longer asks whether a new reasoning round started, so nothing "
        "distinguishes a fresh stream from the end of the last one"
    )
    # `if (` must sit right before the call, which rejects a negated condition.
    assert re.search(r"if\s*\(\s*$", src[:round_at]), (
        "the new reasoning round is not asked as `if (startsNewReasoningRound(...))`. Negated "
        "or combined with another term it can clear the override on the transitions that are "
        "not a new round, and leave the one that is untouched, which is the inverse of what "
        "this guards"
    )
    # _without_block_comments drops braces around a comment-only body, so the brace is
    # located exactly. Predicate is `isStreaming && !wasStreaming`; swapped, it inverts.
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
    # Only the closing `)` may sit between the call and the brace (rejects `=== false`).
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
    # Code is recorded from the placement the request sends; read inside the builder only.
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
