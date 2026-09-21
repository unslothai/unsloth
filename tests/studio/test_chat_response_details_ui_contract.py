# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contract for the chat response-details action and metadata."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
THREAD_TSX = REPO / "studio/frontend/src/components/assistant-ui/thread.tsx"
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
    _, _, utility = token.rpartition(":")
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
    pieces = []
    for argument in _split_arguments(source[opens + len("cn(") : closes]):
        argument = argument.strip()
        literal = re.fullmatch(r'"([^"]*)"', argument)
        if literal:
            pieces.append(literal.group(1))
        elif argument != "className":
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
    """The opening JSX tag beginning at `marker`, brace-aware.

    `[^>]*` ends at the first `>`, and an arrow function in an earlier prop supplies one, so
    the tag would come back truncated and the props after it invisible.
    """
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
    """`source` with `/* ... */` and its JSX `{...}` wrapper removed.

    Kept separate from `_without_comments`, which takes a single opening tag: this one runs
    over whole files, where a commented-out element has to disappear entirely rather than
    have its attributes tidied.
    """
    return re.sub(r"\{?\s*/\*.*?\*/\s*\}?", " ", source, flags = re.S)


def _spread_overrides(tag: str, attribute: str) -> bool:
    """True when `tag` spreads props in a position that can beat an explicit `attribute`.

    JSX applies attributes left to right and the last write wins, so `{...props} name={x}`
    ends with `x` whatever the spread holds, while `name={x} {...props}` does not. Refusing
    both would make the guard red on a safe refactor that forwards unrelated props, which is
    a worse failure than the one it is guarding: it stops correct work.

    A tag with no explicit attribute at all is unknown if it spreads anything, since the
    spread is then the only thing that could be supplying it.
    """
    spreads = [match.start() for match in re.finditer(r"\{\s*\.\.\.", tag)]
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


def test_assistant_more_menu_exposes_response_details_action():
    src = THREAD_TSX.read_text(encoding = "utf-8")
    assert "MessageResponseDetailsSheet" in src
    assert "See response details" in src
    assert "setDetailsOpen(true)" in src


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
    assert 'showResponseModel: "Show response model"' in EN_LOCALE_TS.read_text(encoding = "utf-8")
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


def test_reasoning_keeps_streaming_height_cap_through_automatic_collapse():
    src = REASONING_TSX.read_text(encoding = "utf-8")

    assert "const [retainStreamingHeight, setRetainStreamingHeight]" in src
    assert "setRetainStreamingHeight(false)" in src
    assert "setRetainStreamingHeight(isReasoningStreaming)" in src
    # Still zero while streaming, and still the animation's length once it stops. The two
    # collapse mechanisms differ: the height keyframes animate a height captured at toggle
    # time, so releasing the cap cannot change what they animate and ANIMATION_DURATION is
    # right for them, while `1fr` resolves against the live content every frame and has to
    # outlast a transition that only starts a render after this timer is armed.
    assert "isReasoningStreaming ? 0 : closeDelay" in src
    assert "const closeDelay = GRID_COLLAPSE_REASONING_ENABLED" in src
    assert "? ANIMATION_DURATION + CLOSE_FALLBACK_MARGIN_MS" in src
    assert ": ANIMATION_DURATION;" in src
    # The cap is `streaming || retained`, and #11373 moved the render down a level: the state
    # lives here and is handed to the child as a prop, which then ORs it with its own
    # streaming flag. Which component evaluates it is layout; that it is still ORed, and that
    # the retained flag actually reaches the evaluation, is the claim.
    # On the element that receives it, not anywhere in the file. The same text in a comment,
    # or on some other element, reads identically to a file-wide search and hands nothing
    # over; ReasoningText would then OR in its own default-false prop and the cap is lost.
    # ReasoningBody is rendered more than once, and the one that matters is the one holding
    # this state: it passes isStreaming={isReasoningStreaming}, the flag whose end is what
    # the retained height is bridging. Every such call has to hand the retained flag over.
    holders = [
        tag
        for tag in _opening_tags(_without_block_comments(src), "<ReasoningBody")
        if "isStreaming={isReasoningStreaming}" in tag
    ]
    assert holders, (
        "no ReasoningBody receives isReasoningStreaming any more, so this guard cannot tell "
        "which render is the one whose collapse the retained height exists to smooth"
    )
    # A spread can supply the same prop and, written after it, wins. Nothing here can say
    # what is in one, so a holder that spreads is refused rather than read.
    spreading = [tag for tag in holders if _spread_overrides(tag, "retainStreamingHeight")]
    assert not spreading, (
        f"a ReasoningBody holding this state spreads props, so whether the retained flag it "
        f"is handed survives depends on what the spread contains, which this guard cannot "
        f"resolve: {spreading!r}"
    )
    # On an attribute boundary. Both props are optional, so `data-retainStreamingHeight=`
    # satisfies a substring search while the component falls back on its default of false.
    missing = [
        tag
        for tag in holders
        if not re.search(r"(?:^|[\s{])retainStreamingHeight=\{retainStreamingHeight\}", tag)
    ]
    assert not missing, (
        f"the retained-height flag no longer reaches the component that renders the block, "
        f"so nothing can OR it into the streaming cap: {missing!r}"
    )
    # On ReasoningText specifically. It is the element that writes data-streaming and so owns
    # the height cap; the same OR on a sibling reads identically here and caps nothing.
    tag = _opening_tag(src, "<ReasoningText")
    assert tag, "ReasoningText is no longer rendered, so nothing here caps the height"
    # Same ground as the holder above, and it has to be said again here: JSX takes the last
    # write of a prop, so a spread after `streaming=` decides the cap and the explicit text
    # this guard reads goes on satisfying it.
    assert not _spread_overrides(tag, "streaming"), (
        f"ReasoningText spreads props after its streaming prop, so whether the cap it is "
        f"given survives "
        f"depends on what the spread contains, which this guard cannot resolve: {tag!r}"
    )
    # The left operand has to be the component's own streaming input. `\w+` accepted any
    # identifier, so `streaming={somethingElse || retainStreamingHeight}` passed while an
    # actively streaming block went uncapped whenever the retained flag was false.
    assert re.search(r"(?:^|[\s{])streaming=\{isStreaming \|\| retainStreamingHeight\}", tag), (
        f"ReasoningText's streaming prop no longer ORs in retainStreamingHeight, so the block "
        f"collapses to its idle height the moment streaming stops, which is the jump this "
        f"test exists for: {tag!r}"
    )


def test_reasoning_clears_manual_open_on_a_new_stream():
    """A hand-opened block must not stay pinned open when the stream restarts.

    isOpen is `(streaming && !dismissed) || manualOpen` and manualOpen is only
    settable while idle, so the new-stream reset has to clear it too.
    """
    src = REASONING_TSX.read_text(encoding = "utf-8")

    marker = "setDismissedWhileStreaming(false)"
    start = src.find(marker)
    assert start != -1, "new-stream reset effect is missing"
    effect = src[src.rfind("useEffect(() => {", 0, start) : src.find("});", start)]
    assert "setManualOpen(false)" in effect


def test_response_details_metadata_is_persisted_without_backend_schema_change():
    src = ADAPTER_TS.read_text(encoding = "utf-8")
    assert "interface ResponseDetailsMetadata" in src
    assert "buildResponseDetails" in src
    assert "responseDetails: buildResponseDetails(finishedAt)" in src
    assert "toolCalls: Array.from(" in src
    assert "!isExternalRequest && supportsTools && toolsEnabled" in src
    assert "!isExternalRequest && supportsTools && codeToolsEnabled" in src
    assert re.search(r"selectedModelSummary\?\.name\s*\|\|\s*responseModelId", src)
    assert "providerName" in src
    assert "cancelId" in src
    metadata_block = src[
        src.find("interface ResponseDetailsMetadata") : src.find("type RunMessages")
    ]
    builder_block = src[
        src.find("const buildResponseDetails") : src.find("const externalCapabilities")
    ]
    for forbidden in [
        "encrypted_api_key",
        "externalApiKey",
        "apiKey",
        "providerKey",
        "secret",
    ]:
        assert forbidden not in metadata_block
        assert forbidden not in builder_block
