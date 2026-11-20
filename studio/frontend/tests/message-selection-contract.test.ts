// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Source-level contracts for the native selection milestone: one layer per
// thread, no global selection blocking, and the approved action utilities.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const THREAD = readSrc("components/assistant-ui/thread.tsx");
const LAYER = readSrc("components/assistant-ui/message-selection-layer.tsx");
const MODEL = readSrc("components/assistant-ui/message-selection-model.ts");
const COLLECTION = readSrc(
  "components/assistant-ui/message-quote-collection.tsx",
);
const DIALOG = readSrc("components/assistant-ui/message-quote-dialog.tsx");
const CONTEXT = readSrc("components/assistant-ui/message-quote-context.ts");
const STORE = readSrc("components/assistant-ui/message-quote-store.ts");
const DISPATCH = readSrc("components/assistant-ui/message-quote-dispatch.ts");
const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");
const RUNTIME = readSrc("features/chat/runtime-provider.tsx");
const HISTORY_STORAGE = readSrc("features/chat/utils/chat-history-storage.ts");
const EN_LOCALE = readSrc("i18n/locales/en.ts");
const EDIT_COMPOSER = THREAD.slice(
  THREAD.indexOf("const EditComposer: FC"),
  THREAD.indexOf("const BranchPicker: FC"),
);

test("the thread mounts the selection and quote collection layers", () => {
  assert.match(THREAD, /<MessageSelectionLayer/);
  assert.match(THREAD, /<MessageQuoteCollection/);
  assert.match(THREAD, /viewportEl=\{viewportEl\}/);
  assert.match(THREAD, /enabled=\{!hideComposer\}/);
  assert.match(THREAD, /createEphemeralQuoteThreadId/);
  assert.match(THREAD, /resolveQuoteCollectionKey/);
  assert.match(THREAD, /threadId=\{quoteThreadId\}/);
  assert.match(LAYER, /useAui\(\)/);
});

test("the quote collection key survives unsaved-to-saved initialization", () => {
  assert.match(STORE, /runtimeThreadId/);
  assert.match(STORE, /activeThreadId/);
  assert.match(STORE, /ephemeralThreadId/);
  assert.match(THREAD, /resolveQuoteCollectionKey\(\s*runtimeThreadId,\s*threadId,\s*ephemeralQuoteThreadId/s);
  assert.doesNotMatch(THREAD, /threadId \|\| runtimeThreadId/);
});

test("the layer captures quotes and reuses clipboard utilities", () => {
  assert.match(LAYER, /captureQuoteAnchor/);
  assert.match(LAYER, /addQuote/);
  assert.doesNotMatch(LAYER, /insertSavedPrompt/);
  assert.match(LAYER, /copyToClipboard/);
  assert.match(LAYER, /data-selection-toolbar/);
  assert.match(LAYER, /onMouseDown=\{preserveSelection\}/);
  assert.match(LAYER, /removeAllRanges/);
  // The dock composer is outside message rows; a row edit composer is not.
  assert.match(LAYER, /closest\("\[data-message-id\]"\)/);
});

test("the floating quote pill uses measured geometry and row navigation", () => {
  assert.match(COLLECTION, /createPortal/);
  assert.match(COLLECTION, /fixed/);
  assert.match(COLLECTION, /menu-soft-surface/);
  assert.match(COLLECTION, /quotePillMode/);
  assert.match(COLLECTION, /clampQuotePillPosition/);
  assert.match(COLLECTION, /placeQuoteDropdown/);
  assert.match(COLLECTION, /quotePillDragExceeded/);
  assert.match(COLLECTION, /setPointerCapture/);
  assert.match(COLLECTION, /onPointerDown=\{beginDrag\}/);
  assert.match(COLLECTION, /endDrag\(event, true\)/);
  assert.match(COLLECTION, /endDrag\(event, false\)/);
  assert.match(COLLECTION, /event\.detail === 0/);
  assert.doesNotMatch(COLLECTION, /suppressClickRef/);
  assert.match(COLLECTION, /aria-expanded=\{expanded\}/);
  assert.match(COLLECTION, /DragDropVerticalIcon/);
  assert.match(COLLECTION, /<ul/);
  assert.match(COLLECTION, /<li/);
  assert.match(COLLECTION, /data-quote-remove/);
  assert.match(COLLECTION, /removeQuote/);
  assert.match(COLLECTION, /aui-thread-composer-dock/);
  assert.match(COLLECTION, /unsloth-composer-surface/);
  assert.match(COLLECTION, /getBoundingClientRect/);
  assert.match(COLLECTION, /moveQuoteCollection/);
  assert.doesNotMatch(COLLECTION, /revealProgressiveRow/);
  assert.doesNotMatch(COLLECTION, /scrollIntoView/);
  assert.doesNotMatch(COLLECTION, /CSS\.highlights/);
  assert.match(COLLECTION, /data-quote-navigate/);
  assert.match(COLLECTION, /onNavigate=\{navigate\}/);
});

test("the layer never sends a message", () => {
  assert.doesNotMatch(LAYER, /\.send\s*\(/);
  assert.doesNotMatch(LAYER, /composer\(\)\.send/);
  assert.doesNotMatch(LAYER, /setText\s*\(/);
});

test("Quote opens an anchored compact panel and only Save appends a quote", () => {
  assert.match(LAYER, /MessageQuoteDialog/);
  assert.match(LAYER, /setDraftAnchor\(anchor\)/);
  assert.match(LAYER, /setDraftAnchorRect\(snapshot\.rect\)/);
  assert.match(LAYER, /anchorRect=\{draftAnchorRect\}/);
  assert.match(LAYER, /setQuoteDialogOpen\(true\)/);
  assert.match(LAYER, /addQuote\(threadId, draftAnchor, \{ comment \}\)/);
  assert.doesNotMatch(LAYER, /addQuote\(threadId, anchor\)/);

  assert.match(DIALOG, /createPortal/);
  assert.match(DIALOG, /placeToolbar/);
  assert.match(DIALOG, /menu-soft-surface/);
  assert.match(DIALOG, /data-quote-comment-panel/);
  assert.match(DIALOG, /data-quote-dialog-passage/);
  assert.match(DIALOG, /type="text"/);
  assert.match(DIALOG, /Tick02Icon/);
  assert.match(DIALOG, /onSave\(comment\)/);
  assert.match(DIALOG, /onCancel\(\)/);
  // The panel appears instantly where measured, never sliding in from the
  // viewport origin while the first measurement settles.
  assert.doesNotMatch(DIALOG, /animate-in|zoom-in-95|fade-in-0/);
  assert.doesNotMatch(DIALOG, /left:\s*0\b/);
  assert.match(DIALOG, /PANEL_ESTIMATED_HEIGHT/);
  assert.doesNotMatch(DIALOG, /DialogContent/);
  assert.doesNotMatch(DIALOG, /Textarea/);
  assert.doesNotMatch(DIALOG, /onCloseAutoFocus/);
  assert.doesNotMatch(DIALOG, /common\.cancel/);
  assert.doesNotMatch(DIALOG, /composer\(\)/);
  assert.doesNotMatch(DIALOG, /\.send\s*\(/);
});

test("pending quote dispatches have a persistent confirmed release path", () => {
  assert.match(DISPATCH, /releaseQuoteDispatchManually/);
  assert.doesNotMatch(DISPATCH, /setTimeout|expiresAt|tombstone/i);
  assert.match(COLLECTION, /data-quote-pending-banner/);
  assert.match(COLLECTION, /releaseQuoteDispatchManually/);
  assert.match(COLLECTION, /quoteSendPendingWarning/);
  assert.match(COLLECTION, /releaseQuotesAnyway/);
  assert.match(COLLECTION, /keepWaitingQuotes/);
  assert.match(COLLECTION, /subscribeQuoteDispatches/);
  assert.doesNotMatch(COLLECTION, /<output/);
  assert.match(COLLECTION, /role="status"/);
  assert.match(COLLECTION, /aria-live="polite"/);
  assert.match(COLLECTION, /single generic warning deliberately/);
  assert.match(EN_LOCALE, /i18n follow-up/);
});

test("every EditComposer send is routed through sendEdit", () => {
  assert.match(EDIT_COMPOSER, /const sendEdit = useCallback/);
  assert.match(
    EDIT_COMPOSER,
    /setQuote\(quoteInfoFromMetadata\(sourceQuoteValue\)\)/,
  );
  assert.equal(
    (EDIT_COMPOSER.match(/aui\.composer\(\)\.send\(/g) ?? []).length,
    1,
  );
  assert.equal((EDIT_COMPOSER.match(/sendEdit\(\)/g) ?? []).length, 2);
});

test("normal send paths attach quotes and track send/history lifecycle", () => {
  assert.match(THREAD, /reserveQuoteDispatch\(quoteThreadId\)/);
  assert.match(THREAD, /attachQuotes: false/);
  assert.match(THREAD, /markQuoteDispatchSendStarted/);
  assert.match(THREAD, /markQuoteDispatchSendSettled/);
  assert.match(THREAD, /markQuoteDispatchSendFailed/);
  assert.match(THREAD, /getState\(\)\.quote !== undefined/);
  assert.match(THREAD, /sourceQuoteValue/);
  assert.match(
    THREAD,
    /setQuote\(quoteInfoFromMetadata\(sourceQuoteValue\)\)/,
  );
  assert.match(THREAD, /quoteInfoForDispatch\(dispatch\)/);
  assert.match(THREAD, /parseQuoteContext\(existingQuote\)/);
  const attached = THREAD.match(
    /sendReservedComposer\(\{ attachQuotes: true \}\)/g,
  );
  assert.ok((attached?.length ?? 0) >= 3);
});

test("the adapter appends validated quote context once after user text", () => {
  assert.match(ADAPTER, /appendQuoteContextToText\(baseText/);
  assert.match(ADAPTER, /message\.role === "user"/);
  assert.match(ADAPTER, /quoteValueForMessage/);
  assert.match(ADAPTER, /buildReplayContent\(textContent, imageParts\)/);
  assert.match(CONTEXT, /appendQuoteContextToText/);
  assert.doesNotMatch(ADAPTER, /JSON\.stringify\(quote|quote\.text/);
});

test("history persistence accepts exact messages and preserves quote metadata", () => {
  assert.match(RUNTIME, /quote: m\.metadata\.quote/);
  assert.match(RUNTIME, /quoteContextIdForMessage\(message\)/);
  assert.match(RUNTIME, /markQuoteDispatchHistoryPending/);
  assert.match(RUNTIME, /markQuoteDispatchHistoryOutcome/);
  assert.match(RUNTIME, /markQuoteDispatchHistoryFailed/);
  assert.match(RUNTIME, /acceptQuoteDispatchForMessage\(message\)/);
  assert.match(RUNTIME, /releaseQuoteDispatchForMessage\(message\)/);
  assert.match(RUNTIME, /historyWriteOutcome\(record, saved\)/);
  assert.match(
    RUNTIME,
    /outcome === "persisted" \|\| outcome === "delivered"/,
  );
  assert.doesNotMatch(RUNTIME, /Boolean\(outcome\)|outcome === true/);
  assert.doesNotMatch(RUNTIME, /acceptQuoteDispatchesForMessages/);
  assert.doesNotMatch(RUNTIME, /"discarded"/);
  // Temporary and normal threads share the same tracked append path. Temporary
  // messages return the input record from storage and are classified delivered.
  assert.match(HISTORY_STORAGE, /isThreadIncognito\(message\.threadId\)\) return message/);
  assert.match(RUNTIME, /const tracked = write\.then/);
  assert.match(RUNTIME, /return trackHistoryAppend\(message\.id, tracked\)/);
});

test("the context envelope only formats validated structured data", () => {
  assert.match(CONTEXT, /QUOTE_CONTEXT_MESSAGE_ID/);
  assert.match(CONTEXT, /QUOTE_CONTEXT_PREFIX/);
  assert.match(CONTEXT, /parseQuoteContext/);
  assert.match(CONTEXT, /formatQuoteContextForModel/);
  assert.match(CONTEXT, /kind: "not-ours"/);
  assert.match(CONTEXT, /kind: "invalid"/);
  // The formatter works from typed items, not from the encoded JSON string.
  assert.doesNotMatch(
    CONTEXT.slice(CONTEXT.indexOf("export function formatQuoteContextForModel")),
    /JSON\.parse/,
  );
});

test("the model reuses the existing Markdown and selection serializers", () => {
  assert.match(MODEL, /htmlNodeToMarkdown/);
  assert.match(MODEL, /faithfulSelectionText/);
  assert.match(MODEL, /turnNumberAt/);
  assert.match(MODEL, /\[data-role='assistant'\]\[data-message-id\]/);
  assert.match(MODEL, /QUOTE_PRUNE_SELECTOR/);
  assert.match(MODEL, /isAssistantSelectionOrigin/);
  assert.match(MODEL, /sameSelectionPosition/);
  assert.match(MODEL, /placeToolbar/);
  assert.match(MODEL, /rangeBoundsRect/);
  assert.match(MODEL, /unionSelectionRects/);
  assert.match(MODEL, /getClientRects/);
});

test("keyboard selection uses the focused target and thread selection context", () => {
  assert.match(LAYER, /shouldTrackKeyboardSelection/);
  assert.match(LAYER, /keyboardSelectionContext/);
  assert.match(LAYER, /keyboardSelectionTarget/);
  assert.match(LAYER, /keyboardTargetBlocksDocumentSelection/);
  assert.match(LAYER, /selectionEndpointsInside/);
  assert.match(LAYER, /targetIsDocumentBody/);
  assert.match(MODEL, /selectionEndpointsInside/);
  assert.match(MODEL, /KEYBOARD_SELECTION_EXTEND_KEYS/);
  assert.match(MODEL, /keyboardSelectionTarget/);
  assert.match(MODEL, /keyboardTargetBlocksDocumentSelection/);
  assert.doesNotMatch(LAYER, /event\.shiftKey\s*\|\|/);
});

test("pointer origin gates selection refreshes and outside presses keep it dismissed", () => {
  assert.match(LAYER, /isAssistantSelectionOrigin/);
  assert.match(LAYER, /pointerSelectionRef/);
  assert.match(LAYER, /ignoreSelectionChangeRef/);
  assert.match(LAYER, /pointercancel/);
  assert.match(LAYER, /pointerWasEligible/);
  assert.match(LAYER, /event\.button === 0/);
  assert.match(LAYER, /selectionChangeAllowed/);
  assert.match(MODEL, /selectionChangeAllowed/);
});

test("the toolbar is measured before it is placed", () => {
  assert.match(LAYER, /attachToolbar/);
  assert.match(LAYER, /toolbarSize/);
  assert.match(LAYER, /getBoundingClientRect/);
  assert.match(LAYER, /ResizeObserver/);
  assert.match(LAYER, /placeToolbar/);
  assert.match(LAYER, /viewportEl\.getBoundingClientRect\(\)/);
  assert.match(LAYER, /maxWidth/);
  assert.match(LAYER, /visibility: "hidden"/);
});

test("the toolbar reuses the Glass surface and composer pill controls", () => {
  assert.match(LAYER, /menu-soft-surface/);
  assert.match(LAYER, /menu-soft-edgeless/);
  assert.match(LAYER, /composer-pill-btn/);
  assert.match(LAYER, /composer-pill-glyph/);
  assert.match(LAYER, /QuoteIcon/);
  assert.match(LAYER, /Copy01Icon/);
});

test("copied state follows the range identity, not the selected text", () => {
  assert.match(LAYER, /selectionIdentityRef/);
  assert.match(LAYER, /sameSelectionPosition/);
  assert.match(MODEL, /startContainer/);
  assert.match(MODEL, /startOffset/);
  assert.match(MODEL, /endContainer/);
  assert.match(MODEL, /endOffset/);
});

test("no new global selection blocker is introduced", () => {
  assert.doesNotMatch(LAYER, /selectstart/);
  assert.doesNotMatch(LAYER, /user-select:\s*none/);
  assert.doesNotMatch(MODEL, /selectstart/);
  assert.doesNotMatch(MODEL, /style\.userSelect\s*=/);
});
