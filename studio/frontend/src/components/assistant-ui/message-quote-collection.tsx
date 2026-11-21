// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

// Draggable, session-scoped quote pill. It lives in a body portal outside the
// virtualized message list, so dragging or expanding it never changes the
// thread's scroll geometry.

import { useAui } from "@assistant-ui/react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { isTouchClick } from "@/components/ui/touch-click";
import { useT } from "@/i18n";
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
  clampQuoteCollectionWidth,
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
  quoteSourceLabel,
  shouldApplyQuotePreviewOpen,
  toggledPinnedQuoteId,
} from "./message-quote-presentation";
import {
  getQuotes,
  getReservedQuoteIds,
  removeQuote,
  subscribeQuotes,
} from "./message-quote-store";
import { useDetachThreadFromBottom } from "./use-intent-aware-autoscroll";

const EDGE_INSET = 12;
const TOP_INSET = 8;
const COMPOSER_GAP = 8;
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

    let observedDock: HTMLElement | null = null;
    let observedSurface: HTMLElement | null = null;
    let dockObserver: ResizeObserver | null = null;
    let surfaceObserver: ResizeObserver | null = null;
    let dockTreeObserver: MutationObserver | null = null;

    // The composer dock can mount after this effect (welcome screen, compare
    // mode) and can be replaced. Re-sync the observers whenever the tree says
    // the dock or its surface changed instead of observing only once.
    const syncObservers = () => {
      const dock =
        viewportEl.parentElement?.querySelector<HTMLElement>(
          ".aui-thread-composer-dock",
        ) ?? null;
      const composerSurface =
        dock?.querySelector<HTMLElement>(".unsloth-composer-surface") ?? null;

      if (dock !== observedDock) {
        dockObserver?.disconnect();
        dockObserver = null;
        dockTreeObserver?.disconnect();
        dockTreeObserver = null;
        observedDock = dock;
        if (dock !== null) {
          dockObserver = new ResizeObserver(measure);
          dockObserver.observe(dock);
          dockTreeObserver = new MutationObserver(syncObservers);
          dockTreeObserver.observe(dock, { childList: true });
        }
      }

      if (composerSurface !== observedSurface) {
        surfaceObserver?.disconnect();
        surfaceObserver = null;
        observedSurface = composerSurface;
        if (composerSurface !== null) {
          surfaceObserver = new ResizeObserver(measure);
          surfaceObserver.observe(composerSurface);
        }
      }
    };

    const sync = () => {
      measure();
      syncObservers();
    };

    sync();
    const viewportObserver = new ResizeObserver(measure);
    viewportObserver.observe(viewportEl);
    const parent = viewportEl.parentElement;
    const parentObserver = parent !== null ? new MutationObserver(sync) : null;
    parentObserver?.observe(parent as Node, { childList: true });
    window.addEventListener("resize", measure);

    return () => {
      viewportObserver.disconnect();
      dockObserver?.disconnect();
      surfaceObserver?.disconnect();
      dockTreeObserver?.disconnect();
      parentObserver?.disconnect();
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

const QuoteCount: FC<{ readonly count: number }> = ({ count }) => (
  <span
    data-quote-count=""
    className="inline-grid shrink-0 whitespace-nowrap text-ui-12 tabular-nums leading-snug text-muted-foreground"
  >
    <span aria-hidden={true} className="invisible col-start-1 row-start-1">
      (00)
    </span>
    <span className="col-start-1 row-start-1">({count})</span>
  </span>
);

const QUOTE_PREVIEW_CLASS =
  "pointer-events-auto overflow-y-auto overscroll-contain whitespace-pre-line break-words text-left";
const QUOTE_PREVIEW_STYLE = {
  maxWidth: "min(20rem, calc(100vw - 1.5rem))",
  maxHeight: "min(24rem, calc(100vh - 1.5rem))",
} as const;

const QuotePreviewContent: FC<{
  readonly text: string;
  readonly source: string;
}> = ({ text, source }) => (
  <>
    <span className="block">{text}</span>
    {source.length > 0 && (
      <span className="mt-1 block text-ui-10 text-muted-foreground">
        {source}
      </span>
    )}
  </>
);

const QuotePillRow: FC<{
  readonly quote: CollectedQuote;
  readonly index: number;
  readonly reserved: boolean;
  readonly reservedByQueuedPrompt: boolean;
  readonly rowHeight: number;
  readonly onNavigate: (quote: CollectedQuote) => void;
  readonly onRemove: (quoteId: string) => void;
}> = ({
  quote,
  index,
  reserved,
  reservedByQueuedPrompt,
  rowHeight,
  onNavigate,
  onRemove,
}) => {
  const t = useT();
  const source = quoteSourceLabel(quote.anchor.turn, (turn) =>
    t("turns.label", { number: turn }),
  );
  const removeLabel = `${t("messageSelection.removeQuote")} ${index + 1}`;

  return (
    <li
      data-quote-id={quote.id}
      className="flex w-full min-w-0 shrink-0 list-none items-center gap-1.5 rounded-full bg-popover px-1.5 text-popover-foreground transition-colors hover:bg-muted/50 focus-within:bg-muted/50"
      style={{ height: rowHeight }}
    >
      <Tooltip>
        <TooltipTrigger asChild={true} disableClickToggle={true}>
          <button
            type="button"
            data-quote-navigate=""
            onClick={() => onNavigate(quote)}
            className="flex h-full min-w-0 flex-1 items-center gap-1.5 rounded-full px-1 text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <span className="w-5 shrink-0 text-ui-10 tabular-nums text-muted-foreground">
              {index + 1}
            </span>
            <span className="min-w-0 flex-1 truncate whitespace-nowrap text-ui-12 leading-snug text-foreground/85">
              {quote.anchor.text}
            </span>
          </button>
        </TooltipTrigger>
        <TooltipContent
          side="bottom"
          align="start"
          sideOffset={6}
          collisionPadding={12}
          className={QUOTE_PREVIEW_CLASS}
          style={QUOTE_PREVIEW_STYLE}
        >
          <QuotePreviewContent text={quote.anchor.text} source={source} />
        </TooltipContent>
      </Tooltip>
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
        onClick={(event) => {
          event.stopPropagation();
          if (reserved) {
            return;
          }
          onRemove(quote.id);
        }}
        className={cn(
          "flex size-5 shrink-0 items-center justify-center rounded-[4px] text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
          reserved && "cursor-not-allowed opacity-30",
        )}
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
  const singleQuote = mode === "single" ? quotes[0] : null;
  const [expanded, setExpanded] = useState(false);
  const [singlePreview, setSinglePreview] = useState<{
    readonly quoteId: string;
    readonly open: boolean;
  } | null>(null);
  const [releaseConfirmId, setReleaseConfirmId] = useState<string | null>(null);
  const [positionState, setPositionState] = useState<PositionState>(() => ({
    threadId,
    position: getQuotePillPosition(threadId),
  }));
  const [dragging, setDragging] = useState(false);
  const pillRef = useRef<HTMLButtonElement>(null);
  const sizerRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLUListElement>(null);
  const singlePreviewRef = useRef<HTMLDivElement>(null);
  const singlePinnedRef = useRef<string | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const previousThreadRef = useRef(threadId);
  const previousCountRef = useRef(quotes.length);
  // Navigations await frames; a Thread switch must invalidate the captured key.
  const threadRef = useRef(threadId);
  useLayoutEffect(() => {
    threadRef.current = threadId;
  }, [threadId]);

  const singleQuoteId = singleQuote?.id ?? null;
  // A preview opened for an older quote or thread is stale by identity, not by
  // an effect, so returning to a thread can never resurrect it.
  const singlePreviewOpen =
    singleQuoteId !== null &&
    singlePreview !== null &&
    singlePreview.quoteId === singleQuoteId &&
    singlePreview.open;
  const setSinglePreviewOpen = useCallback(
    (open: boolean) => {
      // Radix closes on pointerdown/click. While a touch pin owns the preview,
      // those temporary closes must not clear the pinned identity.
      if (
        !shouldApplyQuotePreviewOpen(
          singlePinnedRef.current,
          singleQuoteId,
          open,
        )
      ) {
        return;
      }
      if (singleQuoteId === null) {
        setSinglePreview(null);
        return;
      }
      setSinglePreview({ quoteId: singleQuoteId, open });
    },
    [singleQuoteId],
  );
  const closeSinglePreview = useCallback(() => {
    singlePinnedRef.current = null;
    setSinglePreview(null);
  }, []);

  const geometry = useCollectionGeometry(
    viewportEl,
    enabled && quotes.length > 0,
  );
  const sizerSize = useMeasuredSize(sizerRef, enabled && quotes.length > 0);
  const collectionWidth =
    sizerSize === null || geometry === null
      ? null
      : clampQuoteCollectionWidth(
          sizerSize.width,
          geometry.viewport.width,
          EDGE_INSET,
        );
  const pillSize = useMeasuredSize(
    pillRef,
    enabled &&
      quotes.length > 0 &&
      collectionWidth !== null &&
      collectionWidth > 0,
  );

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
  const dropdownOpen = mode === "multiple" && expanded;
  const dropdownSize = useMeasuredSize(
    dropdownRef,
    enabled && dropdownOpen && quotes.length > 0 && pillPosition !== null,
  );
  const dropdownPlacement =
    dropdownOpen &&
    geometry !== null &&
    pillPosition !== null &&
    pillSize !== null &&
    dropdownSize !== null &&
    collectionWidth !== null &&
    collectionWidth > 0
      ? placeQuoteDropdown(
          pillPosition,
          pillSize,
          { width: collectionWidth, height: dropdownSize.height },
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
      threadChanged || quotePillMode(quotes.length) !== "multiple"
        ? false
        : quotePillExpandedAfterCountChange(
            previousCount,
            quotes.length,
            current,
          ),
    );
    // A preview belonging to a different collection size cannot be reused.
    closeSinglePreview();
  }, [closeSinglePreview, threadId, quotes.length]);

  useEffect(() => {
    setPositionState({
      threadId,
      position: getQuotePillPosition(threadId),
    });
    setReleaseConfirmId(null);
    setDragging(false);
    dragRef.current = null;
    closeSinglePreview();
  }, [closeSinglePreview, threadId]);

  useEffect(() => {
    if (!singlePreviewOpen) {
      return;
    }
    // Hover/focus opens the preview through Radix; a controlled preview also
    // needs the dismissal paths Radix would otherwise own.
    const closeOnOutside = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) {
        return;
      }
      if (pillRef.current?.contains(target)) {
        return;
      }
      if (singlePreviewRef.current?.contains(target)) {
        return;
      }
      closeSinglePreview();
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeSinglePreview();
      }
    };
    const closeOnBlur = () => closeSinglePreview();
    document.addEventListener("pointerdown", closeOnOutside, true);
    document.addEventListener("keydown", closeOnEscape, true);
    window.addEventListener("blur", closeOnBlur);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutside, true);
      document.removeEventListener("keydown", closeOnEscape, true);
      window.removeEventListener("blur", closeOnBlur);
    };
  }, [closeSinglePreview, singlePreviewOpen]);

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
      closeSinglePreview();
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
    [aui, closeSinglePreview, detachFromBottom, t, threadId, viewportEl],
  );

  if (!enabled || mode === "hidden" || typeof document === "undefined") {
    return null;
  }

  const label =
    mode === "single"
      ? quotes[0].anchor.text
      : t("messageSelection.multipleQuotes");
  const singleReserved =
    singleQuote !== null && reservedQuoteIds.has(singleQuote.id);
  const singleReservedByQueuedPrompt =
    singleQuote !== null && queuedReservedQuoteIds.has(singleQuote.id);
  const singleSource =
    singleQuote === null
      ? ""
      : quoteSourceLabel(singleQuote.anchor.turn, (turn) =>
          t("turns.label", { number: turn }),
        );
  const expandLabel = expanded
    ? t("messageSelection.collapseQuotes")
    : t("messageSelection.expandQuotes");
  const pillStyle =
    collectionWidth === null || collectionWidth <= 0
      ? null
      : pillPosition !== null
        ? {
            left: pillPosition.left,
            top: pillPosition.top,
            width: collectionWidth,
          }
        : {
            left: geometry?.viewport.left ?? 0,
            top: geometry?.viewport.top ?? 0,
            width: collectionWidth,
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
      closeSinglePreview();
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
      // A cancelled pointer must not leave a tapped preview pinned even when
      // the gesture never began a drag.
      if (!allowToggle) {
        closeSinglePreview();
      }
      return;
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const moved = drag.moved;
    dragRef.current = null;
    setDragging(false);
    if (moved) {
      closeSinglePreview();
      setQuotePillPosition(threadId, drag.last);
      return;
    }
    if (!allowToggle) {
      closeSinglePreview();
      return;
    }
    if (mode === "single") {
      // Touch has no hover, so a non-dragged tap pins the same preview the
      // mouse and keyboard get. A second tap unpins it.
      if (isTouchClick(event)) {
        const pinned = singlePinnedRef.current === singleQuoteId;
        singlePinnedRef.current = toggledPinnedQuoteId(
          singlePinnedRef.current,
          singleQuoteId ?? "",
        );
        setSinglePreview({
          quoteId: singleQuoteId ?? "",
          open: !pinned,
        });
      }
      return;
    }
    setExpanded((current) => !current);
  };

  const pillButton = (
    <button
      ref={pillRef}
      type="button"
      data-quote-pill=""
      data-dragging={dragging ? "true" : undefined}
      aria-expanded={mode === "multiple" ? expanded : undefined}
      aria-controls={mode === "multiple" && expanded ? dropdownId : undefined}
      title={mode === "multiple" ? expandLabel : undefined}
      onClick={(event) => {
        // Mouse and touch activation is handled on pointerup, where the drag
        // threshold is known. Keyboard activation has no pointerup.
        if (event.detail === 0 && mode === "multiple") {
          setExpanded((current) => !current);
        }
      }}
      onPointerDown={beginDrag}
      onPointerMove={moveDrag}
      onPointerUp={(event) => endDrag(event, true)}
      onPointerCancel={(event) => endDrag(event, false)}
      className={cn(
        "menu-soft-surface menu-soft-edgeless flex w-full touch-none select-none items-center gap-1.5 rounded-full py-2 pl-3 pr-3 text-left",
        mode === "single" && "pr-14",
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
      <span className="flex min-w-0 flex-1 items-center gap-1">
        <span
          className={cn(
            "min-w-0 flex-1 truncate whitespace-nowrap text-ui-12 leading-snug text-foreground",
            mode === "multiple" && "font-medium",
          )}
        >
          {label}
        </span>
        {mode === "multiple" && <QuoteCount count={quotes.length} />}
      </span>
      {mode === "multiple" && (
        <HugeiconsIcon
          icon={QuoteIcon}
          strokeWidth={1.75}
          aria-hidden={true}
          className="size-3.5 shrink-0 text-muted-foreground"
        />
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
    </button>
  );

  return createPortal(
    <>
      <div
        ref={sizerRef}
        aria-hidden={true}
        data-quote-pill-sizer=""
        className="pointer-events-none invisible fixed left-0 top-0 flex w-max items-center gap-1.5 whitespace-nowrap rounded-full px-3 py-2 text-left text-ui-12 leading-snug"
      >
        <span className="flex size-3.5 shrink-0 items-center justify-center text-muted-foreground/60">
          <HugeiconsIcon
            icon={DragDropVerticalIcon}
            strokeWidth={1.75}
            className="size-3.5"
          />
        </span>
        <span className="flex items-center gap-1">
          <span className="font-medium text-ui-12 leading-snug text-foreground">
            {t("messageSelection.multipleQuotes")}
          </span>
          <QuoteCount count={2} />
        </span>
        <HugeiconsIcon
          icon={QuoteIcon}
          strokeWidth={1.75}
          aria-hidden={true}
          className="size-3.5 shrink-0 text-muted-foreground"
        />
        {pendingDispatches.length > 0 && (
          <span className="flex size-3.5 shrink-0 items-center justify-center text-muted-foreground">
            <HugeiconsIcon
              icon={Clock01Icon}
              strokeWidth={1.75}
              className="size-3.5"
            />
          </span>
        )}
      </div>

      {pillStyle !== null && (
        <section
          data-quote-collection=""
          aria-label={t("messageSelection.quoteCollection")}
          style={pillStyle}
          className="fixed z-30"
        >
          {mode === "single" && singleQuote !== null ? (
            <TooltipProvider
              delayDuration={120}
              skipDelayDuration={300}
              disableHoverableContent={false}
            >
              <Tooltip
                open={singlePreviewOpen}
                onOpenChange={setSinglePreviewOpen}
              >
                <TooltipTrigger asChild={true} disableClickToggle={true}>
                  {pillButton}
                </TooltipTrigger>
                <TooltipContent
                  ref={singlePreviewRef}
                  data-quote-single-preview=""
                  side="top"
                  align="start"
                  sideOffset={6}
                  collisionPadding={12}
                  className={QUOTE_PREVIEW_CLASS}
                  style={QUOTE_PREVIEW_STYLE}
                >
                  <QuotePreviewContent
                    text={singleQuote.anchor.text}
                    source={singleSource}
                  />
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          ) : (
            pillButton
          )}
          {singleQuote !== null && (
            <button
              type="button"
              data-quote-single-navigate=""
              aria-label={t("messageSelection.quoteOpenSource")}
              title={t("messageSelection.quoteOpenSource")}
              onPointerDown={(event) => event.stopPropagation()}
              onPointerUp={(event) => event.stopPropagation()}
              onClick={(event) => {
                event.stopPropagation();
                void navigate(singleQuote);
              }}
              className="absolute right-7 top-1/2 z-10 flex size-6 -translate-y-1/2 items-center justify-center rounded-[4px] text-muted-foreground transition-colors hover:bg-muted/50 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              <HugeiconsIcon
                icon={QuoteIcon}
                strokeWidth={1.75}
                aria-hidden={true}
                className="size-3.5"
              />
            </button>
          )}
          {singleQuote !== null && (
            <button
              type="button"
              data-quote-single-remove=""
              aria-label={t("messageSelection.removeQuote")}
              aria-description={
                singleReservedByQueuedPrompt
                  ? t("messageSelection.quoteReservedByQueuedPrompt")
                  : undefined
              }
              title={
                singleReservedByQueuedPrompt
                  ? t("messageSelection.quoteReservedByQueuedPrompt")
                  : undefined
              }
              aria-disabled={singleReserved}
              onPointerDown={(event) => event.stopPropagation()}
              onPointerUp={(event) => event.stopPropagation()}
              onClick={(event) => {
                event.stopPropagation();
                if (singleReserved) {
                  return;
                }
                remove(singleQuote.id);
              }}
              className={cn(
                "absolute right-1.5 top-1/2 z-10 flex size-5 -translate-y-1/2 items-center justify-center rounded-[4px] text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                singleReserved && "cursor-not-allowed opacity-30",
              )}
            >
              <HugeiconsIcon
                icon={Cancel01Icon}
                strokeWidth={2}
                aria-hidden={true}
                className="size-3"
              />
            </button>
          )}
        </section>
      )}

      {dropdownOpen &&
        geometry !== null &&
        pillPosition !== null &&
        pillSize !== null &&
        collectionWidth !== null &&
        collectionWidth > 0 && (
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
                  width: collectionWidth,
                  maxHeight: dropdownPlacement.maxHeight,
                }
              : {
                  left: pillPosition.left,
                  top: pillPosition.top,
                  width: collectionWidth,
                  visibility: "hidden" as const,
                }
          }
          className="fixed z-30 m-0 flex list-none flex-col gap-1 overflow-y-auto overscroll-contain bg-transparent p-0 shadow-none"
        >
          {pendingDispatches.map((dispatch) => {
            const confirming = releaseConfirmId === dispatch.contextId;
            return (
              <li
                key={`pending-${dispatch.contextId}`}
                className="list-none shrink-0"
              >
                {/* A single generic warning deliberately covers every visible
                    pending state. The only safe-to-release state, reserved,
                    never paints before send() starts; every visible state can
                    still finish and duplicate if the user releases and resends. */}
                <div
                  data-quote-pending-banner=""
                  className="rounded-lg border border-border/60 bg-muted/30 p-2 text-ui-11p5 leading-snug"
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
          <TooltipProvider
            delayDuration={120}
            skipDelayDuration={300}
            disableHoverableContent={false}
          >
            {quotes.map((quote, index) => (
              <QuotePillRow
                key={quote.id}
                quote={quote}
                index={index}
                reserved={reservedQuoteIds.has(quote.id)}
                reservedByQueuedPrompt={queuedReservedQuoteIds.has(quote.id)}
                rowHeight={pillSize.height}
                onNavigate={navigate}
                onRemove={remove}
              />
            ))}
          </TooltipProvider>
        </ul>
      )}
    </>,
    document.body,
  );
};
