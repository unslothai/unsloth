// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

// One layer per Thread. The browser owns the native selection; this component
// snapshots it, shows contextual actions, and leaves Ctrl+C untouched.

import { useT } from "@/i18n";
import { COMPOSER_INPUT_SELECTOR } from "@/features/settings";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Copy01Icon } from "@/lib/copy-icon";
import { toast } from "@/lib/toast";
import { useAui } from "@assistant-ui/react";
import { QuoteIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type MouseEvent as ReactMouseEvent,
  type ReactElement,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import {
  assistantContentForMessage,
  captureQuoteAnchor,
} from "./message-quote-capture";
import { MessageQuoteDialog } from "./message-quote-dialog";
import type { QuoteAnchor } from "./message-quote-model";
import { addQuote } from "./message-quote-store";
import {
  buildSelectionSnapshot,
  isAssistantSelectionOrigin,
  keyboardSelectionTarget,
  keyboardTargetBlocksDocumentSelection,
  placeToolbar,
  rangeBoundsRect,
  sameSelectionPosition,
  selectionChangeAllowed,
  selectionEndpointsInside,
  shouldTrackKeyboardSelection,
  viewportUsable,
  type KeyboardSelectionInput,
  type SelectionRect,
  type SelectionSnapshot,
  type ToolbarSize,
} from "./message-selection-model";

type MessageSelectionLayerProps = {
  threadId: string;
  viewportEl: HTMLElement | null;
  enabled: boolean;
};

const SELECTION_SETTLE_MS = 60;
const COPIED_RESET_MS = 1600;
const TOOLBAR_MARGIN = 8;
const KEYBOARD_SELECTION_BLOCKED_TARGET = [
  "input",
  "textarea",
  "select",
  "button",
  "[role='button']",
  "a",
  "[contenteditable='']",
  "[contenteditable='true']",
  "[contenteditable='plaintext-only']",
].join(",");

function keyboardSelectionContext(
  event: KeyboardEvent,
  viewportEl: HTMLElement,
): KeyboardSelectionInput {
  const target = keyboardSelectionTarget(event.target);
  const targetBlocksDocumentSelection = keyboardTargetBlocksDocumentSelection(
    target,
    KEYBOARD_SELECTION_BLOCKED_TARGET,
  );
  const targetInThread = target !== null && viewportEl.contains(target);
  const targetIsDocumentBody =
    target === document.body || target === document.documentElement;

  const selection = window.getSelection();
  const selectionInThread =
    selection !== null &&
    selectionEndpointsInside(
      selection.anchorNode,
      selection.focusNode,
      (node) => viewportEl.contains(node),
    );

  return {
    key: event.key,
    shiftKey: event.shiftKey,
    ctrlOrMetaKey: event.ctrlKey || event.metaKey,
    targetBlocksDocumentSelection,
    targetInThread,
    targetIsDocumentBody,
    selectionInThread,
  };
}

function composerInputFor(
  viewport: HTMLElement | null,
): HTMLTextAreaElement | null {
  if (viewport === null) {
    return null;
  }
  const root = viewport.closest(".aui-thread-root");
  if (root === null) {
    return null;
  }
  // A user-message edit composer also carries the composer-input class, and it
  // sits earlier in the DOM than the dock. Only the dock composer is outside a
  // message row, so skip row-owned inputs.
  for (const input of root.querySelectorAll<HTMLTextAreaElement>(
    COMPOSER_INPUT_SELECTOR,
  )) {
    if (input.closest("[data-message-id]") !== null) {
      continue;
    }
    if (input.disabled || input.closest("[aria-disabled='true']") !== null) {
      continue;
    }
    return input;
  }
  return null;
}

export function MessageSelectionLayer({
  threadId,
  viewportEl,
  enabled,
}: MessageSelectionLayerProps): ReactElement | null {
  const t = useT();
  const aui = useAui();
  const [snapshot, setSnapshot] = useState<SelectionSnapshot | null>(null);
  const [copied, setCopied] = useState(false);
  const [toolbarSize, setToolbarSize] = useState<ToolbarSize | null>(null);
  const [draftAnchor, setDraftAnchor] = useState<QuoteAnchor | null>(null);
  const [draftAnchorRect, setDraftAnchorRect] =
    useState<SelectionRect | null>(null);
  const [quoteDialogOpen, setQuoteDialogOpen] = useState(false);
  const toolbarRef = useRef<HTMLDivElement>(null);
  const toolbarObserverRef = useRef<ResizeObserver | null>(null);
  const suppressSelectionClearRef = useRef(false);
  // null: no pointer gesture; true: pointerdown began in eligible assistant
  // text; false: pointerdown began elsewhere. The last non-eligible press keeps
  // selectionchange ignored until a genuine selection gesture starts.
  const pointerSelectionRef = useRef<boolean | null>(null);
  const ignoreSelectionChangeRef = useRef(false);
  const selectionIdentityRef = useRef<{
    messageId: string;
    range: Range;
  } | null>(null);
  const updateTimerRef = useRef<number | null>(null);
  const rectFrameRef = useRef<number | null>(null);
  const copiedTimerRef = useRef<number | null>(null);
  // A ref keeps the listener/effect lifecycle stable even if the host hands
  // us a fresh runtime object after a render.
  const auiRef = useRef(aui);
  auiRef.current = aui;

  const cancelScheduledUpdate = useCallback(() => {
    if (updateTimerRef.current !== null) {
      window.clearTimeout(updateTimerRef.current);
      updateTimerRef.current = null;
    }
  }, []);

  const clearSnapshot = useCallback(() => {
    cancelScheduledUpdate();
    selectionIdentityRef.current = null;
    setSnapshot(null);
  }, [cancelScheduledUpdate]);

  // A callback ref measures the real toolbar on mount and whenever its box
  // changes; selection scroll only moves the already-measured box.
  const attachToolbar = useCallback((node: HTMLDivElement | null) => {
    toolbarRef.current = node;
    toolbarObserverRef.current?.disconnect();
    toolbarObserverRef.current = null;
    if (node === null) {
      setToolbarSize(null);
      return;
    }
    const measure = () => {
      const rect = node.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        return;
      }
      setToolbarSize((current) =>
        current !== null &&
        current.width === rect.width &&
        current.height === rect.height
          ? current
          : { width: rect.width, height: rect.height },
      );
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    toolbarObserverRef.current = observer;
  }, []);

  const update = useCallback(() => {
    if (!enabled || !viewportUsable(viewportEl)) {
      clearSnapshot();
      return;
    }
    // The composer is the destination for Quote. When it is unavailable
    // (Compare panes, dictation, disabled composer) show no unusable toolbar.
    if (composerInputFor(viewportEl) === null) {
      clearSnapshot();
      return;
    }

    const selection = window.getSelection();
    if (
      selection === null ||
      selection.isCollapsed ||
      selection.rangeCount !== 1
    ) {
      if (!suppressSelectionClearRef.current) {
        clearSnapshot();
      }
      return;
    }

    let messages: readonly { id: string; role: string }[];
    try {
      messages = auiRef.current.thread().getState().messages;
    } catch {
      clearSnapshot();
      return;
    }

    const next = buildSelectionSnapshot(selection, viewportEl, messages);
    if (next === null) {
      clearSnapshot();
      return;
    }
    // A re-read of the same range must not erase the "Copied" state, but the
    // same words selected at another position must.
    const previous = selectionIdentityRef.current;
    const changedSelection =
      previous === null ||
      previous.messageId !== next.messageId ||
      !sameSelectionPosition(previous.range, next.range);
    selectionIdentityRef.current = {
      messageId: next.messageId,
      range: next.range.cloneRange(),
    };
    if (changedSelection) {
      setCopied(false);
    }
    setSnapshot(next);
  }, [clearSnapshot, enabled, viewportEl]);

  const scheduleUpdate = useCallback(() => {
    if (updateTimerRef.current !== null) {
      window.clearTimeout(updateTimerRef.current);
    }
    updateTimerRef.current = window.setTimeout(() => {
      updateTimerRef.current = null;
      update();
    }, SELECTION_SETTLE_MS);
  }, [update]);

  const refreshRect = useCallback(() => {
    setSnapshot((current) => {
      if (current === null) {
        return current;
      }
      const rect = rangeBoundsRect(current.range);
      if (rect === null) {
        selectionIdentityRef.current = null;
        return null;
      }
      return { ...current, rect };
    });
  }, []);

  const scheduleRectRefresh = useCallback(() => {
    if (rectFrameRef.current !== null) {
      return;
    }
    rectFrameRef.current = window.requestAnimationFrame(() => {
      rectFrameRef.current = null;
      refreshRect();
    });
  }, [refreshRect]);

  useEffect(() => {
    if (!enabled || viewportEl === null) {
      clearSnapshot();
      return;
    }

    const onSelectionChange = () => {
      if (suppressSelectionClearRef.current) {
        return;
      }
      const pointer = pointerSelectionRef.current;
      // Mid-gesture only an eligible assistant-text press may update. After a
      // press elsewhere, the browser's stale selection must stay dismissed.
      if (
        !selectionChangeAllowed(
          pointer,
          ignoreSelectionChangeRef.current,
        )
      ) {
        return;
      }
      scheduleUpdate();
    };
    const onPointerUp = (event: PointerEvent) => {
      const target = event.target;
      const targetNode = target instanceof Node ? target : null;
      const onToolbar =
        targetNode !== null &&
        toolbarRef.current?.contains(targetNode) === true;
      const pointerWasEligible = pointerSelectionRef.current === true;
      pointerSelectionRef.current = null;

      if (onToolbar) {
        window.setTimeout(() => {
          suppressSelectionClearRef.current = false;
        }, 0);
        return;
      }

      if (pointerWasEligible) {
        // Snapshot the finished drag now; the timer is the fallback for
        // selectionchange ordering and for a click that collapses it.
        const selection = window.getSelection();
        if (selection !== null && !selection.isCollapsed) {
          update();
        }
        ignoreSelectionChangeRef.current = false;
        scheduleUpdate();
      } else {
        ignoreSelectionChangeRef.current = true;
        clearSnapshot();
      }

      // The toolbar's mousedown preventDefault keeps the selection; release
      // the guard after the click has settled.
      window.setTimeout(() => {
        suppressSelectionClearRef.current = false;
      }, 0);
    };
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target;
      const targetNode = target instanceof Node ? target : null;
      if (
        targetNode !== null &&
        toolbarRef.current?.contains(targetNode) === true
      ) {
        cancelScheduledUpdate();
        pointerSelectionRef.current = false;
        ignoreSelectionChangeRef.current = true;
        return;
      }

      const eligible =
        event.button === 0 && isAssistantSelectionOrigin(targetNode);
      pointerSelectionRef.current = eligible;
      ignoreSelectionChangeRef.current = !eligible;
      clearSnapshot();
    };
    const onPointerCancel = () => {
      pointerSelectionRef.current = null;
      ignoreSelectionChangeRef.current = true;
      suppressSelectionClearRef.current = false;
      clearSnapshot();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        clearSnapshot();
        return;
      }
      if (
        shouldTrackKeyboardSelection(
          keyboardSelectionContext(event, viewportEl),
        )
      ) {
        pointerSelectionRef.current = null;
        ignoreSelectionChangeRef.current = false;
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (
        shouldTrackKeyboardSelection(
          keyboardSelectionContext(event, viewportEl),
        )
      ) {
        pointerSelectionRef.current = null;
        ignoreSelectionChangeRef.current = false;
        scheduleUpdate();
      }
    };

    document.addEventListener("selectionchange", onSelectionChange);
    document.addEventListener("pointerup", onPointerUp, true);
    document.addEventListener("pointercancel", onPointerCancel, true);
    document.addEventListener("keyup", onKeyUp, true);
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("keydown", onKeyDown, true);
    viewportEl.addEventListener("scroll", scheduleRectRefresh, {
      passive: true,
    });
    // The chat viewport changes size independently of the window when the
    // sidebar, header, or composer resizes; bounds must follow it.
    const viewportObserver = new ResizeObserver(() =>
      scheduleRectRefresh(),
    );
    viewportObserver.observe(viewportEl);
    window.addEventListener("resize", scheduleRectRefresh);

    return () => {
      document.removeEventListener("selectionchange", onSelectionChange);
      document.removeEventListener("pointerup", onPointerUp, true);
      document.removeEventListener("pointercancel", onPointerCancel, true);
      document.removeEventListener("keyup", onKeyUp, true);
      document.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("keydown", onKeyDown, true);
      viewportEl.removeEventListener("scroll", scheduleRectRefresh);
      viewportObserver.disconnect();
      window.removeEventListener("resize", scheduleRectRefresh);
      if (updateTimerRef.current !== null) {
        window.clearTimeout(updateTimerRef.current);
        updateTimerRef.current = null;
      }
      if (rectFrameRef.current !== null) {
        window.cancelAnimationFrame(rectFrameRef.current);
        rectFrameRef.current = null;
      }
      if (copiedTimerRef.current !== null) {
        window.clearTimeout(copiedTimerRef.current);
        copiedTimerRef.current = null;
      }
      clearSnapshot();
    };
  }, [
    cancelScheduledUpdate,
    clearSnapshot,
    enabled,
    scheduleRectRefresh,
    scheduleUpdate,
    update,
    viewportEl,
  ]);

  useEffect(
    () => () => {
      if (copiedTimerRef.current !== null) {
        window.clearTimeout(copiedTimerRef.current);
      }
      toolbarObserverRef.current?.disconnect();
      toolbarObserverRef.current = null;
    },
    [],
  );

  const preserveSelection = useCallback(
    (event: ReactMouseEvent<HTMLElement>) => {
      event.preventDefault();
      suppressSelectionClearRef.current = true;
    },
    [],
  );

  const handleQuote = useCallback(() => {
    if (snapshot === null || viewportEl === null) {
      return;
    }
    const content = assistantContentForMessage(
      viewportEl,
      snapshot.messageId,
    );
    if (content === null) {
      toast.error(t("messageSelection.quoteCaptureFailed"));
      clearSnapshot();
      return;
    }
    const anchor = captureQuoteAnchor({
      messageId: snapshot.messageId,
      turn: snapshot.turn,
      markdown: snapshot.markdown,
      range: snapshot.range,
      content,
    });
    if (anchor === null) {
      toast.error(t("messageSelection.quoteCaptureFailed"));
      clearSnapshot();
      return;
    }

    // The anchor and its screen rectangle are the persistent source
    // reference. The panel owns the draft until Save; no composer or thread
    // mutation happens here.
    setDraftAnchor(anchor);
    setDraftAnchorRect(snapshot.rect);
    setQuoteDialogOpen(true);
    window.getSelection()?.removeAllRanges();
    clearSnapshot();
  }, [clearSnapshot, snapshot, t, viewportEl]);

  const closeQuoteDialog = useCallback(() => {
    setQuoteDialogOpen(false);
    setDraftAnchor(null);
    setDraftAnchorRect(null);
  }, []);

  const saveQuoteDraft = useCallback(
    (comment: string) => {
      if (draftAnchor !== null) {
        addQuote(threadId, draftAnchor, { comment });
      }
      closeQuoteDialog();
    },
    [closeQuoteDialog, draftAnchor, threadId],
  );

  useEffect(() => {
    // A draft belongs to one conversation; never add it to another thread.
    setQuoteDialogOpen(false);
    setDraftAnchor(null);
    setDraftAnchorRect(null);
  }, [threadId]);

  const handleCopy = useCallback(async () => {
    if (snapshot === null) {
      return;
    }
    if (snapshot.markdown.trim().length === 0) {
      toast.info(t("messageSelection.copyEmpty"));
      return;
    }
    if (!(await copyToClipboard(snapshot.markdown))) {
      toast.error(t("messageSelection.copyFailed"));
      return;
    }

    setCopied(true);
    if (copiedTimerRef.current !== null) {
      window.clearTimeout(copiedTimerRef.current);
    }
    copiedTimerRef.current = window.setTimeout(() => {
      copiedTimerRef.current = null;
      setCopied(false);
    }, COPIED_RESET_MS);
  }, [snapshot, t]);

  if (!enabled) {
    return null;
  }

  let toolbar: ReactElement | null = null;
  if (
    snapshot !== null &&
    viewportEl !== null &&
    typeof document !== "undefined"
  ) {
    const viewportRect = viewportEl.getBoundingClientRect();
    const maxWidth = `${Math.max(
      0,
      viewportRect.width - TOOLBAR_MARGIN * 2,
    )}px`;
    const placement =
      toolbarSize === null
        ? null
        : placeToolbar(
            snapshot.rect,
            toolbarSize,
            viewportRect,
            TOOLBAR_MARGIN,
          );
    // A null placement means the toolbar cannot fit inside the chat viewport,
    // or neither side can hold it without covering the selection. Keep the
    // measured toolbar hidden until scrolling or a new selection creates a
    // valid slot; never clamp it over the highlighted text.
    const style =
      placement === null
        ? {
            left: 0,
            top: 0,
            maxWidth,
            visibility: "hidden" as const,
          }
        : {
            left: placement.left,
            top: placement.top,
            maxWidth,
          };

    toolbar = createPortal(
      <div
        ref={attachToolbar}
        data-selection-toolbar=""
        role="toolbar"
        aria-label={t("messageSelection.toolbar")}
        className="menu-soft-surface menu-soft-edgeless fixed z-50 flex flex-wrap items-center gap-0.5 rounded-full p-0.5 font-sans"
        style={style}
        onMouseDown={preserveSelection}
      >
        <button
          type="button"
          className="composer-pill-btn whitespace-nowrap"
          onMouseDown={preserveSelection}
          onClick={handleQuote}
        >
          <span className="composer-pill-glyph">
            <HugeiconsIcon
              icon={QuoteIcon}
              strokeWidth={2}
              aria-hidden={true}
              className="size-[calc(15px*var(--ui-space-scale,1))]"
            />
          </span>
          <span>{t("messageSelection.quote")}</span>
        </button>
        <button
          type="button"
          title={t("messageSelection.copyAsMarkdown")}
          className="composer-pill-btn whitespace-nowrap"
          onMouseDown={preserveSelection}
          onClick={() => void handleCopy()}
        >
          <span className="composer-pill-glyph">
            <HugeiconsIcon
              icon={Copy01Icon}
              strokeWidth={1.75}
              aria-hidden={true}
              className="size-[calc(15px*var(--ui-space-scale,1))]"
            />
          </span>
          <span>
            {copied
              ? t("messageSelection.copied")
              : t("messageSelection.copy")}
          </span>
        </button>
      </div>,
      document.body,
    );
  }

  return (
    <>
      {toolbar}
      <MessageQuoteDialog
        anchor={draftAnchor}
        anchorRect={draftAnchorRect}
        viewportEl={viewportEl}
        open={quoteDialogOpen}
        onCancel={closeQuoteDialog}
        onSave={saveQuoteDraft}
      />
    </>
  );
}
