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
}: {
  itemIds: string[];
  scrollContainerRef: RefObject<HTMLElement | null>;
  listRootRef: RefObject<HTMLElement | null>;
}) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const [isBulkPending, setIsBulkPending] = useState(false);
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
    setSelectedIds(new Set());
    anchorIdRef.current = null;
  }, []);

  // One batch at a time. The confirm dialog closes before the deletes finish,
  // so the still-live bar would otherwise re-submit the same rows and delete
  // each one twice. The ref guards the second call that lands before the state
  // update; the flag drives the bar. Both reset on the failure path too, or a
  // throwing batch would leave the bar dead until it remounts.
  const runBulkAction = useCallback(
    async (action: () => Promise<void>): Promise<boolean> => {
      if (bulkPendingRef.current) return false;
      bulkPendingRef.current = true;
      setIsBulkPending(true);
      try {
        await action();
        return true;
      } finally {
        bulkPendingRef.current = false;
        setIsBulkPending(false);
      }
    },
    [],
  );

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
      if (!drag || !listRoot) return;

      const index = document
        .elementsFromPoint(
          listRoot.getBoundingClientRect().left + 8,
          clientY,
        )
        .map((el) => indexFromPointerTarget(el, listRoot))
        .find((value) => value != null);
      const liveAnchor = indexOfId(drag.anchorId);
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
      if (event.key === "Escape") clearSelection();
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
      // role-bearing content, not a descendant, so a backdrop dismiss only
      // counts as inside if the overlay is matched on its own.
      return (
        target.closest(
          '[role="dialog"], [role="alertdialog"], [data-slot$="-overlay"]',
        ) != null
      );
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
      };
    },
    [itemIds],
  );

  const handleItemClick = useCallback(
    (
      index: number,
      id: string,
      event: { metaKey: boolean; ctrlKey: boolean; shiftKey: boolean },
    ): boolean => {
      if (suppressClickRef.current) {
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

  return {
    selectedIds,
    selectedCount: selectedIds.size,
    isSelectionActive: selectedIds.size > 0,
    isBulkPending,
    runBulkAction,
    clearSelection,
    handleItemPointerDown,
    handleItemClick,
    isItemSelected,
  };
}
