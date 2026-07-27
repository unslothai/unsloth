// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
} from "react";

const DRAG_THRESHOLD_PX = 4;
const AUTO_SCROLL_EDGE_PX = 28;
const AUTO_SCROLL_STEP_PX = 10;

// Stamped by DialogContent on the one dialog that confirms a bulk delete, over
// both its overlay and its content. Exempting every [role="dialog"] instead
// hands the exemption to unrelated overlays: the Ctrl/Cmd+K chat search opens
// with no outside press in front of it, and picking a result is a press inside
// a dialog followed by a chat-to-chat navigation that leaves Recents mounted,
// so a forgotten destructive batch would ride along to the new chat.
export const SIDEBAR_SELECTION_BOUNDARY = "sidebar-bulk-delete";
const SELECTION_BOUNDARY_SELECTOR = `[data-selection-boundary="${SIDEBAR_SELECTION_BOUNDARY}"]`;
// Escape belongs to the topmost layer it was delivered into, and a row's kebab
// is a layer too: Radix portals its content out of the list and gives it
// role="menu", so a menu-shaped Escape has to stop here the same way a dialog
// one does, or dismissing the menu also throws the batch away.
const OVERLAY_LAYER_SELECTOR =
  '[role="dialog"], [role="alertdialog"], [role="menu"]';

type DragState = {
  anchorIndex: number;
  // By id: the list can reorder mid-drag and shift every index.
  anchorId: string | null;
  pointerId: number;
  pointerType: string;
  startX: number;
  startY: number;
  lastClientY: number;
  dragging: boolean;
  // Set when the row the drag started from leaves the list. The range has no
  // anchor to rebuild from at that point, so the drag stops extending.
  cancelled: boolean;
};

function rangeIndices(anchor: number, current: number): number[] {
  const start = Math.min(anchor, current);
  const end = Math.max(anchor, current);
  const out: number[] = [];
  for (let i = start; i <= end; i += 1) out.push(i);
  return out;
}

function parseSelectionIndex(row: HTMLElement | null | undefined): number | null {
  const raw = row?.dataset.selectionIndex;
  if (raw == null) return null;
  const index = Number.parseInt(raw, 10);
  return Number.isFinite(index) ? index : null;
}

function indexFromPointerTarget(
  target: EventTarget | null,
  listRoot: HTMLElement | null,
): number | null {
  if (!listRoot || !(target instanceof Element)) return null;
  const row = target.closest<HTMLElement>("[data-selection-index]");
  if (!row || !listRoot.contains(row)) return null;
  return parseSelectionIndex(row);
}

// Dragging past an end extends to it instead of collapsing back to the anchor.
function edgeRowIndex(listRoot: HTMLElement, clientY: number): number | null {
  const rows = listRoot.querySelectorAll<HTMLElement>("[data-selection-index]");
  if (rows.length === 0) return null;
  const first = rows[0];
  const last = rows[rows.length - 1];
  if (clientY < first.getBoundingClientRect().top) return parseSelectionIndex(first);
  if (clientY > last.getBoundingClientRect().bottom) return parseSelectionIndex(last);
  return null;
}

export function useSidebarListSelection({
  itemIds,
  scrollContainerRef,
  listRootRef,
  routeKey,
}: {
  itemIds: string[];
  scrollContainerRef: RefObject<HTMLElement | null>;
  listRootRef: RefObject<HTMLElement | null>;
  // The current location, pathname and search together. Navigating is the one
  // signal every way of leaving has in common; the input that triggered it is
  // not. Optional so a caller with no router still gets the rest of the hook.
  routeKey?: string;
}) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const [isBulkPending, setIsBulkPending] = useState(false);
  const [pendingIds, setPendingIds] = useState<Set<string>>(() => new Set());
  const bulkPendingRef = useRef(false);
  const dragRef = useRef<DragState | null>(null);
  const autoScrollRef = useRef<number | null>(null);
  const autoScrollDirectionRef = useRef<-1 | 1 | null>(null);
  // By id, not index: the list reorders, so an index would silently re-point
  // a later shift+click at a different row.
  const anchorIdRef = useRef<string | null>(null);
  const suppressClickRef = useRef(false);
  const updateDragSelectionRef = useRef<(clientY: number) => void>(() => undefined);

  const indexOfId = useCallback(
    (id: string | null) => (id == null ? -1 : itemIds.indexOf(id)),
    [itemIds],
  );

  const clearSelection = useCallback(() => {
    // Keep the empty set when it is already empty: this now runs on every
    // navigation, and a fresh Set each time would re-render the whole sidebar
    // for nothing.
    setSelectedIds((prev) => (prev.size === 0 ? prev : new Set()));
    anchorIdRef.current = null;
  }, []);

  // Drops just the rows handed in. A bulk delete only disables its own Delete
  // button, so the rows stay clickable for the whole loop; clearing everything
  // at the end would throw away a selection the user started while waiting,
  // which was never part of the batch being deleted.
  const deselectIds = useCallback((ids: Iterable<string>) => {
    const removed = new Set(ids);
    if (removed.size === 0) return;
    setSelectedIds((prev) => {
      const next = new Set<string>();
      for (const id of prev) {
        if (!removed.has(id)) next.add(id);
      }
      return next.size === prev.size ? prev : next;
    });
    // Only if the anchor itself went: otherwise a later shift+click would lose
    // the range the surviving selection is still anchored on.
    if (anchorIdRef.current != null && removed.has(anchorIdRef.current)) {
      anchorIdRef.current = null;
    }
  }, []);

  // One batch at a time. The confirm dialog closes before the deletes finish,
  // so the still-live bar would otherwise re-submit the same rows and delete
  // each one twice. The ref guards the second call that lands before the state
  // update; the flag drives the bar. Both reset on the failure path too, or a
  // throwing batch would leave the bar dead until it remounts.
  const runBulkAction = useCallback(
    async (
      action: () => Promise<void>,
      // The rows this batch is going to delete. They are reported as pending so
      // each one's own Delete can be disabled for the duration: the rest of the
      // sidebar deliberately stays live during a slow batch, so the lock has to
      // be the batch's own rows rather than the whole list.
      capturedIds?: Iterable<string>,
    ): Promise<boolean> => {
      if (bulkPendingRef.current) return false;
      bulkPendingRef.current = true;
      setIsBulkPending(true);
      const captured = new Set(capturedIds ?? []);
      setPendingIds((prev) =>
        prev.size === 0 && captured.size === 0 ? prev : captured,
      );
      try {
        await action();
        return true;
      } finally {
        bulkPendingRef.current = false;
        setIsBulkPending(false);
        // On the failure path too, or a throwing batch leaves its rows
        // permanently undeletable.
        setPendingIds((prev) => (prev.size === 0 ? prev : new Set()));
      }
    },
    [],
  );

  // Clearing on the way out, by route rather than by input. A chat can be left
  // by mouse, by keyboard, or by the command palette, which navigates straight
  // out of cmdk's key handling and never produces a MouseEvent for a click or
  // pointer listener to see. Chat-to-chat navigation keeps Recents mounted, so
  // the unmount path below does not cover it either.
  const previousRouteKeyRef = useRef(routeKey);
  useEffect(() => {
    if (previousRouteKeyRef.current === routeKey) return;
    previousRouteKeyRef.current = routeKey;
    // Not while a batch is running: deleting the open chat navigates away by
    // itself, and treating the batch's own redirect as the user leaving would
    // discard a selection started while waiting for it.
    if (bulkPendingRef.current) return;
    clearSelection();
  }, [clearSelection, routeKey]);

  // The sidebar outlives the list: a route change, browser history, or
  // collapsing the section unmounts the rows while the hook keeps its state.
  // Drop the batch then, or coming back restores it and the next plain row
  // click toggles selection instead of opening the row. No dep array: the ref
  // has no identity to key on.
  useEffect(() => {
    if (selectedIds.size === 0) return;
    if (listRootRef.current == null) clearSelection();
  });

  // Drop stale selections when the list changes. Bail out unless truly stale:
  // itemIds is rebuilt most renders, so an unconditional setState loops.
  useEffect(() => {
    if (selectedIds.size === 0) return;
    const valid = new Set(itemIds);
    let hasStale = false;
    for (const id of selectedIds) {
      if (!valid.has(id)) {
        hasStale = true;
        break;
      }
    }
    if (!hasStale) return;
    setSelectedIds((prev) => {
      const next = new Set<string>();
      for (const id of prev) {
        if (valid.has(id)) next.add(id);
      }
      return next;
    });
  }, [itemIds, selectedIds]);

  const applyRangeSelection = useCallback(
    (anchorIndex: number, currentIndex: number) => {
      const indices = rangeIndices(anchorIndex, currentIndex);
      setSelectedIds(() => {
        const next = new Set<string>();
        for (const index of indices) {
          const id = itemIds[index];
          if (id) next.add(id);
        }
        return next;
      });
    },
    [itemIds],
  );

  const toggleId = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const stopAutoScroll = useCallback(() => {
    if (autoScrollRef.current != null) {
      window.clearInterval(autoScrollRef.current);
      autoScrollRef.current = null;
    }
    autoScrollDirectionRef.current = null;
  }, []);

  const startAutoScroll = useCallback(
    (direction: -1 | 1) => {
      if (
        autoScrollRef.current != null &&
        autoScrollDirectionRef.current === direction
      ) {
        return;
      }
      stopAutoScroll();
      autoScrollDirectionRef.current = direction;
      autoScrollRef.current = window.setInterval(() => {
        const container = scrollContainerRef.current;
        const drag = dragRef.current;
        if (!container || !drag) return;
        container.scrollTop += direction * AUTO_SCROLL_STEP_PX;
        updateDragSelectionRef.current(drag.lastClientY);
      }, 16);
    },
    [scrollContainerRef, stopAutoScroll],
  );

  const updateDragSelection = useCallback(
    (clientY: number) => {
      const drag = dragRef.current;
      const listRoot = listRootRef.current;
      const container = scrollContainerRef.current;
      if (!drag || drag.cancelled || !listRoot) return;

      // The row the drag started from can vanish mid-drag: deleted from another
      // window, or pushed out of the limited Recents window by a new chat. The
      // old numeric index now addresses whichever row slid into its place, so
      // rebuilding the range from it would sweep up rows the user never touched
      // and arm them for the delete that usually follows. Stop instead, and
      // keep whatever of the selection is still valid.
      const liveAnchor = indexOfId(drag.anchorId);
      if (drag.anchorId != null && liveAnchor === -1) {
        drag.cancelled = true;
        drag.anchorId = null;
        stopAutoScroll();
        return;
      }

      const index = document
        .elementsFromPoint(
          listRoot.getBoundingClientRect().left + 8,
          clientY,
        )
        .map((el) => indexFromPointerTarget(el, listRoot))
        .find((value) => value != null);
      const anchorIndex = liveAnchor === -1 ? drag.anchorIndex : liveAnchor;
      const currentIndex =
        index ?? edgeRowIndex(listRoot, clientY) ?? anchorIndex;
      applyRangeSelection(anchorIndex, currentIndex);

      if (container) {
        const rect = container.getBoundingClientRect();
        if (clientY < rect.top + AUTO_SCROLL_EDGE_PX) startAutoScroll(-1);
        else if (clientY > rect.bottom - AUTO_SCROLL_EDGE_PX) startAutoScroll(1);
        else stopAutoScroll();
      }
    },
    [
      applyRangeSelection,
      indexOfId,
      listRootRef,
      scrollContainerRef,
      startAutoScroll,
      stopAutoScroll,
    ],
  );

  updateDragSelectionRef.current = updateDragSelection;

  useEffect(() => {
    const onPointerMove = (event: PointerEvent) => {
      const drag = dragRef.current;
      if (!drag || event.pointerId !== drag.pointerId) return;
      if (drag.pointerType !== "mouse") return;
      // Cancelled by its anchor disappearing: the press is still held, but it
      // no longer selects anything.
      if (drag.cancelled) return;

      const dx = event.clientX - drag.startX;
      const dy = event.clientY - drag.startY;
      if (!drag.dragging) {
        if (Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) return;
        drag.dragging = true;
      }

      drag.lastClientY = event.clientY;
      event.preventDefault();
      // Through the ref, not the closure: a callback dep would re-run this
      // effect every render and kill the auto-scroll interval on cleanup.
      updateDragSelectionRef.current(event.clientY);
    };

    const onPointerUp = (event: PointerEvent) => {
      const drag = dragRef.current;
      if (!drag || event.pointerId !== drag.pointerId) return;
      if (drag.dragging) {
        suppressClickRef.current = true;
        // Continue a later shift+click from where the drag started.
        if (drag.anchorId != null) anchorIdRef.current = drag.anchorId;
      }
      dragRef.current = null;
      stopAutoScroll();
    };

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
      stopAutoScroll();
    };
  }, [stopAutoScroll]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      // An open dialog or menu holds focus, so an Escape aimed at it is
      // delivered inside it and dismisses that layer alone. Clearing here as
      // well would make keyboard cancel the one path that loses the batch,
      // while Cancel and a backdrop press both keep it.
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest(OVERLAY_LAYER_SELECTOR)) return;
      clearSelection();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [clearSelection]);

  useEffect(() => {
    if (selectedIds.size === 0) return;
    const isInsideSelectionBoundary = (target: Element | null) => {
      if (!target) return false;
      if (listRootRef.current?.contains(target)) return true;
      // The confirm dialog is portaled out of the list, so clearing on it would
      // drop the batch it is confirming. Its overlay is a portal sibling of the
      // content, not a descendant, which is why both carry the marker.
      return target.closest(SELECTION_BOUNDARY_SELECTOR) != null;
    };
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      if (isInsideSelectionBoundary(target)) return;
      clearSelection();
    };
    // Keyboard activation of an outside control fires no pointerdown at all.
    // detail 0 is what marks it: pointer clicks are already handled above,
    // while the overlay they landed on is still mounted.
    const onClick = (event: MouseEvent) => {
      if (event.detail !== 0) return;
      const target = event.target instanceof Element ? event.target : null;
      if (isInsideSelectionBoundary(target)) return;
      clearSelection();
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("click", onClick);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("click", onClick);
    };
  }, [clearSelection, listRootRef, selectedIds.size]);

  const handleItemPointerDown = useCallback(
    (index: number, event: ReactPointerEvent) => {
      // A drag released off the list never reaches a click; a stale flag would
      // then swallow the next real one.
      suppressClickRef.current = false;
      if (event.button !== 0 || event.pointerType !== "mouse") return;
      dragRef.current = {
        anchorIndex: index,
        anchorId: itemIds[index] ?? null,
        pointerId: event.pointerId,
        pointerType: event.pointerType,
        startX: event.clientX,
        startY: event.clientY,
        lastClientY: event.clientY,
        dragging: false,
        cancelled: false,
      };
    },
    [itemIds],
  );

  const handleItemClick = useCallback(
    (
      index: number,
      id: string,
      event: {
        metaKey: boolean;
        ctrlKey: boolean;
        shiftKey: boolean;
        detail?: number;
      },
    ): boolean => {
      // Only the pointer-generated click of a drag can be the one to swallow.
      // A drag released off the row produces no row click at all and leaves the
      // flag set, and keyboard or assistive-technology activation, which carries
      // detail 0, arrives with no pointerdown in front of it to clear the flag;
      // consuming that would leave the row doing nothing.
      if (suppressClickRef.current && (event.detail ?? 1) > 0) {
        suppressClickRef.current = false;
        return true;
      }

      const modifier = event.metaKey || event.ctrlKey;
      const anchorIndex = indexOfId(anchorIdRef.current);
      if (event.shiftKey && anchorIndex !== -1) {
        applyRangeSelection(anchorIndex, index);
        return true;
      }
      if (modifier || selectedIds.size > 0) {
        toggleId(id);
        anchorIdRef.current = id;
        return true;
      }
      anchorIdRef.current = id;
      return false;
    },
    [applyRangeSelection, indexOfId, selectedIds.size, toggleId],
  );

  const isItemSelected = useCallback(
    (id: string) => selectedIds.has(id),
    [selectedIds],
  );

  // True only for the rows a running batch already owns, so a row's own Delete
  // cannot issue a second request for something the batch is mid-way through.
  const isItemPending = useCallback(
    (id: string) => pendingIds.has(id),
    [pendingIds],
  );

  return {
    selectedIds,
    selectedCount: selectedIds.size,
    isSelectionActive: selectedIds.size > 0,
    isBulkPending,
    runBulkAction,
    clearSelection,
    deselectIds,
    handleItemPointerDown,
    handleItemClick,
    isItemSelected,
    isItemPending,
  };
}
