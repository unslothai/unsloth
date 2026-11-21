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
  assert.match(
    COLLECTION,
    /aria-expanded=\{mode === "multiple" \? expanded : undefined\}/,
  );
  assert.match(
    COLLECTION,
    /aria-controls=\{\s*mode === "multiple" && expanded \? dropdownId : undefined\s*\}/,
  );
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

test("the quote source label never exposes the internal message id", () => {
  assert.doesNotMatch(COLLECTION, /shortMessageId/);
  assert.match(COLLECTION, /quoteSourceLabel\(quote\.anchor\.turn/);
});

test("the quote collection width comes from the hidden sizer", () => {
  assert.doesNotMatch(
    COLLECTION,
    /QUOTE_COLLECTION_WIDTH|DROPDOWN_WIDTH|pillMaxWidth/,
  );
  assert.match(COLLECTION, /sizerRef/);
  assert.match(COLLECTION, /useMeasuredSize\(sizerRef/);
  assert.match(
    COLLECTION,
    /useMeasuredSize\(\s*pillRef,\s*enabled &&[\s\S]*?collectionWidth !== null/,
  );
  assert.match(COLLECTION, /data-quote-pill-sizer=""/);
  assert.match(
    COLLECTION,
    /pointer-events-none invisible fixed left-0 top-0 flex w-max items-center gap-1\.5 whitespace-nowrap rounded-full px-3 py-2/,
  );
  assert.match(COLLECTION, /sizerSize === null \|\| geometry === null/);
  assert.match(
    COLLECTION,
    /clampQuoteCollectionWidth\(\s*sizerSize\.width/,
  );
  assert.ok((COLLECTION.match(/width: collectionWidth/g) ?? []).length >= 2);
  assert.match(COLLECTION, /const pillStyle =[\s\S]*?width: collectionWidth/);
  assert.match(COLLECTION, /\{pillStyle !== null &&/);
  assert.match(
    COLLECTION,
    /\{dropdownOpen &&[\s\S]*?collectionWidth !== null[\s\S]*?collectionWidth > 0/,
  );
});

test("the compact pill keeps a stable count and adds remove only in single mode", () => {
  assert.match(COLLECTION, /const QuoteCount: FC/);
  assert.match(COLLECTION, /data-quote-count=""/);
  assert.match(COLLECTION, /\(00\)/);
  assert.match(COLLECTION, /\(\{count\}\)/);
  assert.match(COLLECTION, /tabular-nums/);
  assert.match(COLLECTION, /col-start-1/);
  assert.match(COLLECTION, /<QuoteCount count=\{2\} \/>/);
  assert.match(COLLECTION, /<QuoteCount count=\{quotes\.length\} \/>/);

  assert.match(COLLECTION, /gap-1\.5/);
  assert.match(
    COLLECTION,
    /<span className="flex min-w-0 flex-1 items-center gap-1">/,
  );
  assert.doesNotMatch(COLLECTION, /justify-between|ml-auto/);
  assert.match(COLLECTION, /mode === "single" && "pr-14"/);
  assert.match(
    COLLECTION,
    /const singleQuote = mode === "single" \? quotes\[0\] : null/,
  );
  assert.match(COLLECTION, /data-quote-single-remove=""/);
  assert.match(COLLECTION, /absolute right-1\.5 top-1\/2 z-10/);
  assert.match(
    COLLECTION,
    /onPointerDown=\{\(event\) => event\.stopPropagation\(\)\}/,
  );
  assert.match(
    COLLECTION,
    /onPointerUp=\{\(event\) => event\.stopPropagation\(\)\}/,
  );
  assert.match(COLLECTION, /remove\(singleQuote\.id\)/);
  assert.match(COLLECTION, /aria-disabled=\{singleReserved\}/);
  assert.match(
    COLLECTION,
    /if \(singleReserved\) \{\s*return;\s*\}/,
  );
  assert.match(
    COLLECTION,
    /singleReserved && "cursor-not-allowed opacity-30"/,
  );
  assert.doesNotMatch(COLLECTION, /\sdisabled=\{singleReserved\}/);

  const pillStart = COLLECTION.indexOf("ref={pillRef}");
  const pill = COLLECTION.slice(
    pillStart,
    COLLECTION.indexOf("</button>", pillStart),
  );
  assert.doesNotMatch(pill, /data-quote-single-remove/);
  assert.doesNotMatch(pill, /preventDefault/);
  assert.match(pill, /flex w-full touch-none select-none/);
  assert.match(
    COLLECTION,
    /min-w-0 flex-1 truncate whitespace-nowrap text-ui-12 leading-snug text-foreground/,
  );
  assert.doesNotMatch(
    COLLECTION,
    /min-w-0 flex-1 break-words text-ui-12 leading-snug text-foreground line-clamp-2/,
  );
  assert.match(COLLECTION, /rounded-full/);
  assert.doesNotMatch(COLLECTION, /ChevronDownStandardIcon/);
  assert.doesNotMatch(COLLECTION, /rotate-180/);

  assert.ok(
    pill.indexOf("icon={DragDropVerticalIcon}") < pill.indexOf("{label}"),
  );
  assert.ok(pill.indexOf("{label}") < pill.indexOf("<QuoteCount"));
  assert.ok(pill.indexOf("<QuoteCount") < pill.indexOf("icon={QuoteIcon}"));
  assert.ok(
    pill.indexOf("icon={QuoteIcon}") < pill.indexOf("data-quote-pending"),
  );

  assert.match(
    COLLECTION,
    /aria-expanded=\{mode === "multiple" \? expanded : undefined\}/,
  );
  assert.match(
    COLLECTION,
    /aria-controls=\{\s*mode === "multiple" && expanded \? dropdownId : undefined\s*\}/,
  );
  assert.match(COLLECTION, /title=\{mode === "multiple" \? expandLabel : undefined\}/);
  assert.match(COLLECTION, /data-quote-pending=""/);
  assert.match(COLLECTION, /sr-only/);
});

test("expanded rows are pill-sized strips with always-visible removal", () => {
  const row = COLLECTION.slice(
    COLLECTION.indexOf("const QuotePillRow"),
    COLLECTION.indexOf("export const MessageQuoteCollection"),
  );

  assert.match(
    row,
    /className="flex w-full min-w-0 shrink-0 list-none items-center gap-1\.5 rounded-full bg-popover px-1\.5 text-popover-foreground/,
  );
  assert.doesNotMatch(row, /menu-soft-surface|menu-soft-edgeless/);
  assert.doesNotMatch(row, /shadow-/);
  assert.match(row, /style=\{\{ height: rowHeight \}\}/);
  assert.match(COLLECTION, /rowHeight=\{pillSize\.height\}/);
  assert.match(
    row,
    /truncate whitespace-nowrap text-ui-12 leading-snug text-foreground\/85/,
  );
  assert.doesNotMatch(row, /line-clamp/);
  assert.doesNotMatch(row, /icon=\{QuoteIcon\}/);
  assert.doesNotMatch(row, /mt-1 block truncate text-ui-10/);
  assert.doesNotMatch(row, /opacity-0|group-hover|pointer-coarse/);

  assert.match(row, /data-quote-navigate=""/);
  assert.match(row, /data-quote-remove=""/);
  assert.match(row, /<TooltipTrigger/);
  assert.ok(row.indexOf("</TooltipTrigger>") < row.indexOf("data-quote-remove"));
  assert.match(row, /aria-disabled=\{reserved\}/);
  assert.doesNotMatch(row, /\sdisabled=\{reserved\}/);
  assert.match(row, /if \(reserved\) \{\s*return;\s*\}/);
  assert.match(row, /reserved && "cursor-not-allowed opacity-30"/);

  assert.match(
    COLLECTION,
    /className="fixed z-30 m-0 flex list-none flex-col gap-1 overflow-y-auto overscroll-contain bg-transparent p-0 shadow-none"/,
  );
  assert.doesNotMatch(COLLECTION, /scroll-rounded rounded-xl p-1\.5/);
  assert.match(COLLECTION, /className="list-none shrink-0"/);
});

test("row previews show the full quote and source in a scrollable surface", () => {
  const preview = COLLECTION.slice(
    COLLECTION.indexOf("const QUOTE_PREVIEW_CLASS"),
    COLLECTION.indexOf("const QuotePillRow"),
  );

  assert.match(
    preview,
    /pointer-events-auto overflow-y-auto overscroll-contain/,
  );
  assert.match(
    preview,
    /maxWidth: "min\(20rem, calc\(100vw - 1\.5rem\)\)"/,
  );
  assert.match(
    preview,
    /maxHeight: "min\(24rem, calc\(100vh - 1\.5rem\)\)"/,
  );
  assert.match(preview, /<span className="block">\{text\}<\/span>/);
  assert.match(
    preview,
    /<span className="mt-1 block text-ui-10 text-muted-foreground">\s*\{source\}/,
  );
  assert.doesNotMatch(preview, /line-clamp/);

  const row = COLLECTION.slice(
    COLLECTION.indexOf("const QuotePillRow"),
    COLLECTION.indexOf("export const MessageQuoteCollection"),
  );

  assert.match(row, /QUOTE_PREVIEW_CLASS/);
  assert.match(row, /QUOTE_PREVIEW_STYLE/);
  assert.match(
    row,
    /<QuotePreviewContent text=\{quote\.anchor\.text\} source=\{source\} \/>/,
  );
  assert.match(row, /quoteSourceLabel\(quote\.anchor\.turn/);
  assert.doesNotMatch(row, /line-clamp/);
  assert.match(COLLECTION, /<TooltipProvider/);
  assert.match(COLLECTION, /delayDuration=\{120\}/);
  assert.match(COLLECTION, /skipDelayDuration=\{300\}/);
  assert.match(COLLECTION, /disableHoverableContent=\{false\}/);
});

test("single quotes preview on the pill without a duplicate dropdown row", () => {
  assert.match(
    COLLECTION,
    /const dropdownOpen = mode === "multiple" && expanded/,
  );
  assert.match(
    COLLECTION,
    /mode === "single" && singleQuote !== null \? \(/,
  );
  assert.match(COLLECTION, /<Tooltip\s+open=\{singlePreviewOpen\}/);
  assert.match(
    COLLECTION,
    /<TooltipTrigger asChild=\{true\} disableClickToggle=\{true\}>/,
  );
  assert.match(COLLECTION, /data-quote-single-preview=""/);
  assert.match(COLLECTION, /side="top"/);
  assert.match(
    COLLECTION,
    /<QuotePreviewContent\s+text=\{singleQuote\.anchor\.text\}\s+source=\{singleSource\}/,
  );
  assert.match(COLLECTION, /quoteSourceLabel\(singleQuote\.anchor\.turn/);
  assert.match(COLLECTION, /style=\{QUOTE_PREVIEW_STYLE\}/);

  const singlePill = COLLECTION.slice(
    COLLECTION.indexOf('mode === "single" && singleQuote !== null ? ('),
    COLLECTION.indexOf("{dropdownOpen &&"),
  );
  assert.ok(
    singlePill.indexOf("</TooltipTrigger>") <
      singlePill.indexOf("data-quote-single-remove"),
    "the remove control must stay outside the preview trigger",
  );
  assert.match(
    singlePill,
    /onPointerDown=\{\(event\) => event\.stopPropagation\(\)\}/,
  );
  assert.match(
    singlePill,
    /onPointerUp=\{\(event\) => event\.stopPropagation\(\)\}/,
  );

  // Single mode has no disclosure and no native-title duplicate.
  assert.doesNotMatch(COLLECTION, /title=\{mode === "single" \? label/);
  assert.match(
    COLLECTION,
    /event\.detail === 0 && mode === "multiple"/,
  );
  assert.match(
    COLLECTION,
    /if \(isTouchClick\(event\)\) \{\s*const pinned = singlePinnedRef\.current === singleQuoteId;\s*singlePinnedRef\.current = toggledPinnedQuoteId\(/,
  );
  assert.match(COLLECTION, /const closeOnOutside = \(event: PointerEvent\)/);
  assert.match(COLLECTION, /singlePreviewRef\.current\?\.contains\(target\)/);
  assert.match(
    COLLECTION,
    /onPointerCancel=\{\(event\) => endDrag\(event, false\)\}/,
  );
  assert.match(
    COLLECTION,
    /if \(!allowToggle\) \{\s*closeSinglePreview\(\)/,
  );
  assert.match(
    COLLECTION,
    /if \(mode === "single"\) \{\s*\/\/ Touch has no hover/,
  );
  assert.match(COLLECTION, /const singlePinnedRef = useRef<string \| null>\(null\)/);
  assert.match(
    COLLECTION,
    /shouldApplyQuotePreviewOpen\(\s*singlePinnedRef\.current,\s*singleQuoteId,\s*open,?\s*\)/,
  );
  assert.match(
    COLLECTION,
    /const closeSinglePreview = useCallback\(\(\) => \{\s*singlePinnedRef\.current = null;\s*setSinglePreview\(null\);/,
  );
});

test("a single quote keeps an always-rendered source navigation control", () => {
  const singleBranch = COLLECTION.slice(
    COLLECTION.indexOf('mode === "single" && singleQuote !== null ? ('),
    COLLECTION.indexOf("{dropdownOpen &&"),
  );
  assert.match(singleBranch, /data-quote-single-navigate=""/);
  assert.match(
    singleBranch,
    /aria-label=\{t\("messageSelection.quoteOpenSource"\)\}/,
  );
  assert.match(
    singleBranch,
    /title=\{t\("messageSelection.quoteOpenSource"\)\}/,
  );
  assert.match(singleBranch, /void navigate\(singleQuote\)/);
  assert.match(singleBranch, /absolute right-7 top-1\/2 z-10/);
  assert.match(singleBranch, /size-6/);
  assert.match(singleBranch, /size-5/);

  const triggerEnd = singleBranch.indexOf("</TooltipTrigger>");
  const sourceStart = singleBranch.indexOf("data-quote-single-navigate");
  const removeStart = singleBranch.indexOf("data-quote-single-remove");
  assert.ok(triggerEnd >= 0 && sourceStart > triggerEnd);
  assert.ok(removeStart > sourceStart);
  assert.ok(
    sourceStart < removeStart,
    "the source control must sit outside the prompt and before remove",
  );
  assert.match(
    singleBranch,
    /onPointerDown=\{\(event\) => event\.stopPropagation\(\)\}/,
  );
  assert.match(
    singleBranch,
    /onPointerUp=\{\(event\) => event\.stopPropagation\(\)\}/,
  );
  assert.match(EN_LOCALE, /quoteOpenSource: "Go to quoted passage"/);
  assert.match(
    COLLECTION,
    /\{mode === "multiple" && \(\s*<HugeiconsIcon\s+icon=\{QuoteIcon\}/,
  );
});

test("geometry observers resync when the composer dock mounts late", () => {
  const hook = COLLECTION.slice(
    COLLECTION.indexOf("function useCollectionGeometry"),
    COLLECTION.indexOf("function useMeasuredSize"),
  );
  assert.match(hook, /const syncObservers = \(\) =>/);
  assert.match(hook, /dock !== observedDock/);
  assert.match(hook, /composerSurface !== observedSurface/);
  assert.match(hook, /new MutationObserver\(sync\)/);
  assert.match(
    hook,
    /dockTreeObserver = new MutationObserver\(syncObservers\)/,
  );
  assert.match(
    hook,
    /parentObserver\?\.observe\(parent as Node, \{ childList: true \}\)/,
  );
  assert.match(hook, /dockObserver\?\.disconnect\(\)/);
  assert.match(hook, /surfaceObserver\?\.disconnect\(\)/);
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
  assert.match(DIALOG, /onCancel\("escape"\)/);
  assert.match(DIALOG, /onCancel\("outside"\)/);
  assert.match(LAYER, /closeQuoteDialog\("save"\)/);
  assert.match(LAYER, /reason === "save" \|\| reason === "escape"/);
  const closeQuoteDialogSource = LAYER.slice(
    LAYER.indexOf("const closeQuoteDialog"),
    LAYER.indexOf("const saveQuoteDraft"),
  );
  assert.match(
    closeQuoteDialogSource,
    /if \(reason === "save" \|\| reason === "escape"\) \{[\s\S]*?composerInputFor\(viewportEl\)\?\.focus\(\{ preventScroll: true \}\)/,
  );
  assert.doesNotMatch(
    closeQuoteDialogSource,
    /querySelector<HTMLElement>\(COMPOSER_INPUT_SELECTOR\)/,
  );
  assert.match(LAYER, /input\.closest\("\[data-message-id\]"\) !== null/);
  assert.match(LAYER, /focus\(\{ preventScroll: true \}\)/);
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
