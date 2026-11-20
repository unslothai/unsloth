// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

// Draggable, session-scoped quote pill. It lives in a body portal outside the
// virtualized message list, so dragging or expanding it never changes the
// thread's scroll geometry.

import { useAui } from "@assistant-ui/react";
import { useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  Clock01Icon,
  DragDropVerticalIcon,
  QuoteIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type FC,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { createPortal } from "react-dom";
import {
  getPendingQuoteDispatchesForThread,
  releaseQuoteDispatchManually,
  shouldShowQuoteDispatchInPill,
  subscribeQuoteDispatches,
} from "./message-quote-dispatch";
import { announceQuoteNavigation } from "./message-quote-highlight";
import type { CollectedQuote } from "./message-quote-model";
import {
  cancelQuoteNavigation,
  navigateToQuote,
} from "./message-quote-navigation";
import {
  type QuotePillPosition,
  type QuotePillSize,
  type QuotePillViewport,
  clampQuotePillPosition,
  placeQuoteDropdown,
  quotePillDragExceeded,
} from "./message-quote-position";
import {
  getQuotePillPosition,
  setQuotePillPosition,
} from "./message-quote-position-store";
import {
  quotePillExpandedAfterCountChange,
  quotePillMode,
} from "./message-quote-presentation";
import {
  getQuotes,
  getReservedQuoteIds,
  removeQuote,
  subscribeQuotes,
} from "./message-quote-store";
import { shortMessageId } from "./message-selection-model";
import { useDetachThreadFromBottom } from "./use-intent-aware-autoscroll";

const EDGE_INSET = 12;
const TOP_INSET = 8;
const COMPOSER_GAP = 8;
const DROPDOWN_WIDTH = 340;
const DROPDOWN_GAP = 6;
const DRAG_THRESHOLD = 4;

type CollectionGeometry = {
  readonly viewport: QuotePillViewport;
  readonly defaultLeft: number;
};

type PositionState = {
  readonly threadId: string;
  readonly position: QuotePillPosition | null;
};

type DragState = {
  readonly pointerId: number;
  readonly startX: number;
  readonly startY: number;
  readonly origin: QuotePillPosition;
  last: QuotePillPosition;
  moved: boolean;
};

type MessageQuoteCollectionProps = {
  readonly threadId: string;
  readonly viewportEl: HTMLElement | null;
  readonly enabled: boolean;
};

function measureGeometry(viewportEl: HTMLElement): CollectionGeometry | null {
  if (!viewportEl.isConnected) {
    return null;
  }
  const viewport = viewportEl.getBoundingClientRect();
  const paddingTop =
    Number.parseFloat(getComputedStyle(viewportEl).paddingTop) || 0;
  const dock =
    viewportEl.parentElement?.querySelector<HTMLElement>(
      ".aui-thread-composer-dock",
    ) ?? null;
  const composerSurface =
    dock?.querySelector<HTMLElement>(".unsloth-composer-surface") ?? null;
  const dockRect = dock?.getBoundingClientRect() ?? null;
  const surfaceRect = composerSurface?.getBoundingClientRect() ?? null;
  const composerTop = dockRect?.top ?? viewport.bottom;
  const top = viewport.top + paddingTop + TOP_INSET;
  const height = Math.max(0, composerTop - COMPOSER_GAP - top);
  if (viewport.width <= 0 || height <= 0) {
    return null;
  }
  return {
    viewport: {
      left: viewport.left,
      top,
      width: viewport.width,
      height,
    },
    defaultLeft:
      surfaceRect?.left ?? dockRect?.left ?? viewport.left + EDGE_INSET,
  };
}

function useCollectionGeometry(
  viewportEl: HTMLElement | null,
  enabled: boolean,
): CollectionGeometry | null {
  const [geometry, setGeometry] = useState<CollectionGeometry | null>(null);
  const measure = useCallback(() => {
    if (!enabled || viewportEl === null) {
      setGeometry(null);
      return;
    }
    setGeometry(measureGeometry(viewportEl));
  }, [enabled, viewportEl]);

  useLayoutEffect(() => {
    measure();
    if (!enabled || viewportEl === null) {
      return;
    }
    const viewportObserver = new ResizeObserver(measure);
    viewportObserver.observe(viewportEl);
    const dock =
      viewportEl.parentElement?.querySelector<HTMLElement>(
        ".aui-thread-composer-dock",
      ) ?? null;
    const composerSurface =
      dock?.querySelector<HTMLElement>(".unsloth-composer-surface") ?? null;
    const dockObserver = dock ? new ResizeObserver(measure) : null;
    if (dockObserver !== null && dock !== null) {
      dockObserver.observe(dock);
    }
    const surfaceObserver = composerSurface
      ? new ResizeObserver(measure)
      : null;
    if (surfaceObserver !== null && composerSurface !== null) {
      surfaceObserver.observe(composerSurface);
    }
    const dockTreeObserver =
      dock === null && viewportEl.parentElement !== null
        ? new MutationObserver(measure)
        : null;
    if (dockTreeObserver !== null) {
      dockTreeObserver.observe(viewportEl.parentElement as Node, {
        childList: true,
      });
    }
    window.addEventListener("resize", measure);

    return () => {
      viewportObserver.disconnect();
      dockObserver?.disconnect();
      surfaceObserver?.disconnect();
      dockTreeObserver?.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [enabled, measure, viewportEl]);

  return geometry;
}

function useMeasuredSize(
  ref: RefObject<HTMLElement | null>,
  enabled: boolean,
): QuotePillSize | null {
  const [size, setSize] = useState<QuotePillSize | null>(null);
  const measure = useCallback(() => {
    const node = ref.current;
    if (node === null) {
      return;
    }
    const rect = node.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      return;
    }
    setSize((current) =>
      current !== null &&
      current.width === rect.width &&
      current.height === rect.height
        ? current
        : { width: rect.width, height: rect.height },
    );
  }, [ref]);

  useLayoutEffect(() => {
    if (!enabled) {
      setSize(null);
      return;
    }
    measure();
    const node = ref.current;
    if (node === null) {
      return;
    }
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    window.addEventListener("resize", measure);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [enabled, measure, ref]);

  return size;
}

function useCollectedQuotes(threadId: string): readonly CollectedQuote[] {
  const subscribe = useCallback(
    (listener: () => void) => subscribeQuotes(threadId, listener),
    [threadId],
  );
  const getSnapshot = useCallback(() => getQuotes(threadId), [threadId]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

function usePendingQuoteDispatches(threadId: string) {
  const subscribe = useCallback(
    (listener: () => void) => subscribeQuoteDispatches(threadId, listener),
    [threadId],
  );
  const getSnapshot = useCallback(
    () => getPendingQuoteDispatchesForThread(threadId),
    [threadId],
  );
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

function quoteSource(
  quote: CollectedQuote,
  turnLabel: (turn: number) => string,
): string {
  const shortId = shortMessageId(quote.anchor.messageId);
  return quote.anchor.turn > 0
    ? `${turnLabel(quote.anchor.turn)} · ${shortId}`
    : shortId;
}

const QuotePillRow: FC<{
  readonly quote: CollectedQuote;
  readonly index: number;
  readonly reserved: boolean;
  readonly reservedByQueuedPrompt: boolean;
  readonly onNavigate: (quote: CollectedQuote) => void;
  readonly onRemove: (quoteId: string) => void;
}> = ({
  quote,
  index,
  reserved,
  reservedByQueuedPrompt,
  onNavigate,
  onRemove,
}) => {
  const t = useT();
  const source = quoteSource(quote, (turn) =>
    t("turns.label", { number: turn }),
  );
  const removeLabel = `${t("messageSelection.removeQuote")} ${index + 1}`;

  return (
    <li
      data-quote-id={quote.id}
      className="group flex min-w-0 list-none items-start gap-2 rounded-lg px-1 py-1 transition-colors"
    >
      <button
        type="button"
        data-quote-navigate=""
        onClick={() => onNavigate(quote)}
        className="flex min-w-0 flex-1 items-start gap-2 rounded-md px-1 py-1 text-left transition-colors hover:bg-muted/50 focus-visible:bg-muted/50 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      >
        <span className="mt-0.5 w-5 shrink-0 text-ui-10 tabular-nums text-muted-foreground">
          {index + 1}
        </span>
        <span className="min-w-0 flex-1">
          <span className="line-clamp-2 break-words text-ui-12 leading-snug text-foreground/85">
            {quote.anchor.text}
          </span>
          <span className="mt-1 block truncate text-ui-10 text-muted-foreground">
            {source}
          </span>
        </span>
      </button>
      <button
        type="button"
        data-quote-remove=""
        aria-label={removeLabel}
        aria-description={
          reservedByQueuedPrompt
            ? t("messageSelection.quoteReservedByQueuedPrompt")
            : undefined
        }
        title={
          reservedByQueuedPrompt
            ? t("messageSelection.quoteReservedByQueuedPrompt")
            : undefined
        }
        aria-disabled={reserved}
        disabled={reserved}
        onClick={(event) => {
          event.stopPropagation();
          onRemove(quote.id);
        }}
        className="mt-1 flex size-5 shrink-0 items-center justify-center rounded-[4px] text-muted-foreground opacity-0 transition-opacity hover:bg-destructive/10 hover:text-destructive focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring group-hover:opacity-100 disabled:cursor-not-allowed disabled:opacity-30 pointer-coarse:opacity-100"
      >
        <HugeiconsIcon
          icon={Cancel01Icon}
          strokeWidth={2}
          aria-hidden={true}
          className="size-3"
        />
      </button>
    </li>
  );
};

export const MessageQuoteCollection: FC<MessageQuoteCollectionProps> = ({
  threadId,
  viewportEl,
  enabled,
}) => {
  const t = useT();
  const aui = useAui();
  const detachFromBottom = useDetachThreadFromBottom();
  const dropdownId = useId();
  const highlightedQuoteRef = useRef<string | null>(null);
  const quotes = useCollectedQuotes(threadId);
  const allDispatches = usePendingQuoteDispatches(threadId);
  const pendingDispatches = allDispatches.filter(
    shouldShowQuoteDispatchInPill,
  );
  const queuedReservedQuoteIds = new Set(
    allDispatches
      .filter(
        (dispatch) =>
          dispatch.origin === "queue" &&
          dispatch.sendState === "not-started",
      )
      .flatMap((dispatch) => dispatch.quoteIds),
  );
  const reservedQuoteIds = getReservedQuoteIds(threadId);
  const mode = quotePillMode(quotes.length);
  const [expanded, setExpanded] = useState(false);
  const [releaseConfirmId, setReleaseConfirmId] = useState<string | null>(null);
  const [positionState, setPositionState] = useState<PositionState>(() => ({
    threadId,
    position: getQuotePillPosition(threadId),
  }));
  const [dragging, setDragging] = useState(false);
  const pillRef = useRef<HTMLButtonElement>(null);
  const dropdownRef = useRef<HTMLUListElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const previousThreadRef = useRef(threadId);
  const previousCountRef = useRef(quotes.length);
  // Navigations await frames; a Thread switch must invalidate the captured key.
  const threadRef = useRef(threadId);
  useLayoutEffect(() => {
    threadRef.current = threadId;
  }, [threadId]);

  const geometry = useCollectionGeometry(
    viewportEl,
    enabled && quotes.length > 0,
  );
  const pillSize = useMeasuredSize(pillRef, enabled && quotes.length > 0);

  const storedPosition =
    positionState.threadId === threadId
      ? positionState.position
      : getQuotePillPosition(threadId);
  const pillPosition =
    geometry !== null && pillSize !== null
      ? clampQuotePillPosition(
          storedPosition ?? {
            left: geometry.defaultLeft,
            top: geometry.viewport.top + geometry.viewport.height,
          },
          pillSize,
          geometry.viewport,
          EDGE_INSET,
        )
      : null;
  const pillMaxWidth =
    geometry === null
      ? 320
      : Math.max(0, Math.min(320, geometry.viewport.width - EDGE_INSET * 2));
  const dropdownWidth =
    geometry === null
      ? 0
      : Math.max(
          0,
          Math.min(DROPDOWN_WIDTH, geometry.viewport.width - EDGE_INSET * 2),
        );
  const dropdownSize = useMeasuredSize(
    dropdownRef,
    enabled && expanded && quotes.length > 0 && pillPosition !== null,
  );
  const dropdownPlacement =
    expanded &&
    geometry !== null &&
    pillPosition !== null &&
    pillSize !== null &&
    dropdownSize !== null &&
    dropdownWidth > 0
      ? placeQuoteDropdown(
          pillPosition,
          pillSize,
          { width: dropdownWidth, height: dropdownSize.height },
          geometry.viewport,
          EDGE_INSET,
          DROPDOWN_GAP,
        )
      : null;

  useEffect(() => {
    const threadChanged = previousThreadRef.current !== threadId;
    const previousCount = threadChanged
      ? quotes.length
      : previousCountRef.current;
    previousThreadRef.current = threadId;
    previousCountRef.current = quotes.length;
    setExpanded((current) =>
      threadChanged
        ? false
        : quotePillExpandedAfterCountChange(
            previousCount,
            quotes.length,
            current,
          ),
    );
  }, [threadId, quotes.length]);

  useEffect(() => {
    setPositionState({
      threadId,
      position: getQuotePillPosition(threadId),
    });
    setReleaseConfirmId(null);
    setDragging(false);
    dragRef.current = null;
  }, [threadId]);

  useEffect(() => {
    if (
      releaseConfirmId !== null &&
      !pendingDispatches.some(
        (dispatch) => dispatch.contextId === releaseConfirmId,
      )
    ) {
      setReleaseConfirmId(null);
    }
  }, [pendingDispatches, releaseConfirmId]);

  const remove = useCallback(
    (quoteId: string) => {
      if (highlightedQuoteRef.current === quoteId) {
        cancelQuoteNavigation(threadId);
        highlightedQuoteRef.current = null;
      }
      removeQuote(threadId, quoteId);
    },
    [threadId],
  );

  useEffect(() => {
    highlightedQuoteRef.current = null;
    cancelQuoteNavigation(threadId);
    return () => cancelQuoteNavigation(threadId);
  }, [enabled, threadId]);

  useEffect(() => {
    const highlighted = highlightedQuoteRef.current;
    if (
      highlighted !== null &&
      !quotes.some((quote) => quote.id === highlighted)
    ) {
      cancelQuoteNavigation(threadId);
      highlightedQuoteRef.current = null;
    }
  }, [quotes, threadId]);

  const navigate = useCallback(
    async (quote: CollectedQuote) => {
      setExpanded(false);
      detachFromBottom();
      highlightedQuoteRef.current = quote.id;
      const result = await navigateToQuote({
        viewport: viewportEl,
        threadKey: threadId,
        anchor: quote.anchor,
        messages: aui.thread().getState().messages,
        readThreadKey: () => threadRef.current,
        isCurrentScope: () => viewportEl?.isConnected === true,
      });
      if (result === "highlighted") {
        announceQuoteNavigation(
          t("messageSelection.quoteNavigationAnnounce"),
        );
        return;
      }
      // A newer navigation owns the ref; only clear our own failed attempt.
      if (result !== "cancelled" && highlightedQuoteRef.current === quote.id) {
        highlightedQuoteRef.current = null;
      }
      if (result === "source-missing") {
        toast.info(t("messageSelection.quoteSourceMissing"));
      } else if (result === "passage-missing") {
        toast.info(t("messageSelection.quotePassageMissing"));
      }
      if (result === "source-missing" || result === "passage-missing") {
        // The activated row was unmounted with the dropdown; leave keyboard
        // users on the persistent pill rather than dropping focus to <body>.
        pillRef.current?.focus();
      }
    },
    [aui, detachFromBottom, t, threadId, viewportEl],
  );

  if (!enabled || mode === "hidden" || typeof document === "undefined") {
    return null;
  }

  const label =
    mode === "single"
      ? quotes[0].anchor.text
      : t("messageSelection.multipleQuotes");
  const expandLabel = expanded
    ? t("messageSelection.collapseQuotes")
    : t("messageSelection.expandQuotes");
  const pillStyle =
    pillPosition !== null
      ? { left: pillPosition.left, top: pillPosition.top }
      : {
          left: geometry?.viewport.left ?? 0,
          top: geometry?.viewport.top ?? 0,
          visibility: "hidden" as const,
        };

  const beginDrag = (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (
      event.button !== 0 ||
      !event.isPrimary ||
      pillPosition === null ||
      pillSize === null ||
      geometry === null
    ) {
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      origin: pillPosition,
      last: pillPosition,
      moved: false,
    };
  };

  const moveDrag = (event: ReactPointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current;
    if (
      drag === null ||
      drag.pointerId !== event.pointerId ||
      pillSize === null ||
      geometry === null
    ) {
      return;
    }
    if (!drag.moved) {
      const exceeded = quotePillDragExceeded(
        drag.startX,
        drag.startY,
        event.clientX,
        event.clientY,
        DRAG_THRESHOLD,
      );
      if (!exceeded) {
        return;
      }
    }
    if (!drag.moved) {
      drag.moved = true;
      setDragging(true);
    }
    event.preventDefault();
    const next = clampQuotePillPosition(
      {
        left: drag.origin.left + event.clientX - drag.startX,
        top: drag.origin.top + event.clientY - drag.startY,
      },
      pillSize,
      geometry.viewport,
      EDGE_INSET,
    );
    drag.last = next;
    setPositionState({ threadId, position: next });
  };

  const endDrag = (
    event: ReactPointerEvent<HTMLButtonElement>,
    allowToggle: boolean,
  ) => {
    const drag = dragRef.current;
    if (drag === null || drag.pointerId !== event.pointerId) {
      return;
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const moved = drag.moved;
    dragRef.current = null;
    setDragging(false);
    if (moved) {
      setQuotePillPosition(threadId, drag.last);
      return;
    }
    if (allowToggle) {
      setExpanded((current) => !current);
    }
  };

  return createPortal(
    <>
      <section
        data-quote-collection=""
        aria-label={t("messageSelection.quoteCollection")}
        style={pillStyle}
        className="fixed z-30"
      >
        <button
          ref={pillRef}
          type="button"
          data-quote-pill=""
          data-dragging={dragging ? "true" : undefined}
          aria-expanded={expanded}
          aria-controls={expanded ? dropdownId : undefined}
          title={expandLabel}
          style={{ maxWidth: pillMaxWidth }}
          onClick={(event) => {
            // Mouse and touch activation is handled on pointerup, where the
            // drag threshold is known. Keyboard activation has no pointerup.
            if (event.detail === 0) {
              setExpanded((current) => !current);
            }
          }}
          onPointerDown={beginDrag}
          onPointerMove={moveDrag}
          onPointerUp={(event) => endDrag(event, true)}
          onPointerCancel={(event) => endDrag(event, false)}
          className={cn(
            "menu-soft-surface menu-soft-edgeless flex touch-none select-none items-center gap-2 rounded-full px-3 py-2 text-left",
            dragging ? "cursor-grabbing" : "cursor-grab",
          )}
        >
          <span
            aria-hidden={true}
            title={t("messageSelection.moveQuoteCollection")}
            className="flex size-3.5 shrink-0 items-center justify-center text-muted-foreground/60"
          >
            <HugeiconsIcon
              icon={DragDropVerticalIcon}
              strokeWidth={1.75}
              className="size-3.5"
            />
          </span>
          <HugeiconsIcon
            icon={QuoteIcon}
            strokeWidth={1.75}
            aria-hidden={true}
            className="size-3.5 shrink-0 text-muted-foreground"
          />
          <span
            className={cn(
              "min-w-0 truncate text-ui-12 text-foreground",
              mode === "multiple"
                ? "font-medium"
                : "max-w-[calc(190px*var(--ui-space-scale,1))]",
            )}
            title={mode === "single" ? label : undefined}
          >
            {label}
          </span>
          {mode === "multiple" && (
            <span className="shrink-0 text-ui-11 tabular-nums text-muted-foreground">
              {quotes.length}
            </span>
          )}
          {pendingDispatches.length > 0 && (
            <>
              <span className="sr-only">
                {t("messageSelection.quoteSendPending")}
              </span>
              <span
                data-quote-pending=""
                title={t("messageSelection.quoteSendPending")}
                aria-hidden={true}
                className="flex size-3.5 shrink-0 items-center justify-center text-muted-foreground"
              >
                <HugeiconsIcon
                  icon={Clock01Icon}
                  strokeWidth={1.75}
                  className="size-3.5"
                />
              </span>
            </>
          )}
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={1.75}
            aria-hidden={true}
            className={cn(
              "size-3.5 shrink-0 text-muted-foreground transition-transform",
              expanded && "rotate-180",
            )}
          />
        </button>
      </section>

      {expanded && pillPosition !== null && dropdownWidth > 0 && (
        <ul
          ref={dropdownRef}
          id={dropdownId}
          data-quote-dropdown=""
          aria-label={t("messageSelection.quoteCollection")}
          style={
            dropdownPlacement !== null
              ? {
                  left: dropdownPlacement.left,
                  top: dropdownPlacement.top,
                  width: dropdownWidth,
                  maxHeight: dropdownPlacement.maxHeight,
                }
              : {
                  left: pillPosition.left,
                  top: pillPosition.top,
                  width: dropdownWidth,
                  visibility: "hidden" as const,
                }
          }
          className="menu-soft-surface menu-soft-edgeless fixed z-30 m-0 list-none overflow-y-auto overscroll-contain scroll-rounded rounded-xl p-1.5"
        >
          {pendingDispatches.map((dispatch) => {
            const confirming = releaseConfirmId === dispatch.contextId;
            return (
              <li key={`pending-${dispatch.contextId}`} className="list-none">
                {/* A single generic warning deliberately covers every visible
                    pending state. The only safe-to-release state, reserved,
                    never paints before send() starts; every visible state can
                    still finish and duplicate if the user releases and resends. */}
                <div
                  data-quote-pending-banner=""
                  className="mb-1 rounded-lg border border-border/60 bg-muted/30 p-2 text-ui-11p5 leading-snug"
                >
                  <div className="flex items-start gap-2">
                    <HugeiconsIcon
                      icon={Clock01Icon}
                      strokeWidth={1.75}
                      aria-hidden={true}
                      className="mt-0.5 size-3.5 shrink-0 text-muted-foreground"
                    />
                    <div
                      role="status"
                      aria-live="polite"
                      className="min-w-0 flex-1"
                    >
                      <p className="font-medium text-foreground">
                        {t("messageSelection.quoteSendPending")}
                      </p>
                      <p className="mt-1 text-muted-foreground">
                        {confirming
                          ? t("messageSelection.quoteSendPendingWarning")
                          : t("messageSelection.quoteSendPendingSaving")}
                      </p>
                    </div>
                  </div>
                  <div className="mt-2 flex flex-wrap justify-end gap-1.5">
                    {confirming ? (
                      <>
                        <button
                          type="button"
                          onClick={() => setReleaseConfirmId(null)}
                          className="rounded-md px-2 py-1 text-ui-11 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                        >
                          {t("messageSelection.keepWaitingQuotes")}
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            releaseQuoteDispatchManually(dispatch.contextId);
                            setReleaseConfirmId(null);
                          }}
                          className="rounded-md bg-destructive/10 px-2 py-1 text-ui-11 font-medium text-destructive transition-colors hover:bg-destructive/20 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                        >
                          {t("messageSelection.releaseQuotesAnyway")}
                        </button>
                      </>
                    ) : (
                      <button
                        type="button"
                        onClick={() => setReleaseConfirmId(dispatch.contextId)}
                        className="rounded-md px-2 py-1 text-ui-11 text-muted-foreground underline-offset-2 transition-colors hover:bg-muted hover:text-foreground hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      >
                        {t("messageSelection.releaseQuotes")}
                      </button>
                    )}
                  </div>
                </div>
              </li>
            );
          })}
          {quotes.map((quote, index) => (
            <QuotePillRow
              key={quote.id}
              quote={quote}
              index={index}
              reserved={reservedQuoteIds.has(quote.id)}
              reservedByQueuedPrompt={queuedReservedQuoteIds.has(quote.id)}
              onNavigate={navigate}
              onRemove={remove}
            />
          ))}
        </ul>
      )}
    </>,
    document.body,
  );
};
