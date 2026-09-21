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
    """
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
    opening = source[opens : end + 1]
    literal = re.search(r'className="([^"]*)"', opening)
    if literal:
        return literal.group(1)
    # An expression-valued className, `className={cn(...)}` or `className={"min-w-max"}`.
    # Its quoted pieces are what tailwind-merge sees, in this order; anything else in the
    # expression is beyond a reader like this one. Returning None here would be worse than
    # useless: the caller would fall back on the base classes and pass while the call site
    # overrode them, so an expression with no readable piece has to say so instead.
    expression = re.search(r"className=\{", opening)
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


def _effective_widths(tokens: list[str]) -> dict[str, str]:
    """Variant -> the min-width utility that survives for it, in `cn` order.

    tailwind-merge resolves each variant separately and the last one wins, so `min-w-0
    md:min-w-max` keeps both and the element stops shrinking above `md`. A scan that only
    looks at unqualified utilities reads min-w-0 there and calls it fine.
    """
    widths: dict[str, str] = {}
    for token in tokens:
        variant, _, utility = token.rpartition(":")
        if utility.startswith("min-w-"):
            widths[variant] = utility
    return widths


def _assert_shrinks(widths: dict[str, str], what: str, evidence: str) -> None:
    offenders = {variant: utility for variant, utility in widths.items() if utility != "min-w-0"}
    assert not offenders, (
        f"{what} stops shrinking below its content at some width, so a long summary widens "
        f"the row past the thread there: {offenders} as variant -> effective min-width. "
        f"{evidence}"
    )
    assert "" in widths, (
        f"{what} states no unqualified min-width, so whether it shrinks at the smallest "
        f"widths is left to whatever the element defaults to. {evidence}"
    )


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
    base = re.search(r'"(aui-reasoning-trigger[^"]*)"', reasoning_src)
    call_site = _class_list(reasoning_src, "<ReasoningTrigger")
    assert call_site != _UNREADABLE, (
        "the ReasoningTrigger call site passes a className this guard cannot resolve, so it "
        "cannot tell whether the base min-w-0 survives tailwind-merge. Widen the reader "
        "before trusting it"
    )
    # In `cn(base, className)` order, and the LAST min-w-* wins: cn runs tailwind-merge, so a
    # call site passing min-w-full or min-w-max drops the base min-w-0 and the trigger stops
    # shrinking. A union of the two would still hold the base token and call that fine.
    ordered = (base.group(1).split() if base else []) + (call_site.split() if call_site else [])
    assert ordered, (
        "neither ReasoningTrigger's base classes nor its call site carries a class list this "
        "can read, so this guard cannot see the trigger's layout at all"
    )
    _assert_shrinks(
        _effective_widths(ordered),
        "the reasoning trigger",
        f"Base classes: {base.group(1) if base else None!r}. Call site: {call_site!r}",
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
    # Resolved the same way as the trigger: a header that shrinks everywhere except above one
    # breakpoint puts the overflow back one level up at exactly those widths.
    _assert_shrinks(
        _effective_widths(header.split()),
        "the header row holding the trigger",
        f"Classes: {header!r}",
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
    assert "retainStreamingHeight={retainStreamingHeight}" in src, (
        "the retained-height flag no longer reaches the component that renders the block, so "
        "nothing can OR it into the streaming cap"
    )
    # On ReasoningText specifically. It is the element that writes data-streaming and so owns
    # the height cap; the same OR on a sibling reads identically here and caps nothing.
    text_element = re.search(r"<ReasoningText\b[^>]*>", src, re.S)
    assert text_element, "ReasoningText is no longer rendered, so nothing here caps the height"
    # The left operand has to be the component's own streaming input. `\w+` accepted any
    # identifier, so `streaming={somethingElse || retainStreamingHeight}` passed while an
    # actively streaming block went uncapped whenever the retained flag was false.
    assert re.search(
        r"streaming=\{isStreaming \|\| retainStreamingHeight\}", text_element.group(0)
    ), (
        f"ReasoningText's streaming prop no longer ORs in retainStreamingHeight, so the block "
        f"collapses to its idle height the moment streaming stops, which is the jump this "
        f"test exists for: {text_element.group(0)!r}"
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
