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

function indexFromPointerTarget(
  target: EventTarget | null,
  listRoot: HTMLElement | null,
): number | null {
  if (!listRoot || !(target instanceof Element)) return null;
  const row = target.closest<HTMLElement>("[data-selection-index]");
  if (!row || !listRoot.contains(row)) return null;
  const raw = row.dataset.selectionIndex;
  if (raw == null) return null;
  const index = Number.parseInt(raw, 10);
  return Number.isFinite(index) ? index : null;
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
  const dragRef = useRef<DragState | null>(null);
  const autoScrollRef = useRef<number | null>(null);
  const autoScrollDirectionRef = useRef<-1 | 1 | null>(null);
  const anchorIndexRef = useRef<number | null>(null);
  const suppressClickRef = useRef(false);
  const updateDragSelectionRef = useRef<(clientY: number) => void>(() => undefined);

  const clearSelection = useCallback(() => {
    setSelectedIds(new Set());
    anchorIndexRef.current = null;
  }, []);

  // Drop stale selections when the visible list changes.
  useEffect(() => {
    const valid = new Set(itemIds);
    setSelectedIds((prev) => {
      if (prev.size === 0) return prev;
      const next = new Set<string>();
      for (const id of prev) {
        if (valid.has(id)) next.add(id);
      }
      return next.size === prev.size ? prev : next;
    });
  }, [itemIds]);

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
      const currentIndex = index ?? drag.anchorIndex;
      applyRangeSelection(drag.anchorIndex, currentIndex);

      if (container) {
        const rect = container.getBoundingClientRect();
        if (clientY < rect.top + AUTO_SCROLL_EDGE_PX) startAutoScroll(-1);
        else if (clientY > rect.bottom - AUTO_SCROLL_EDGE_PX) startAutoScroll(1);
        else stopAutoScroll();
      }
    },
    [
      applyRangeSelection,
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
      updateDragSelection(event.clientY);
    };

    const onPointerUp = (event: PointerEvent) => {
      const drag = dragRef.current;
      if (!drag || event.pointerId !== drag.pointerId) return;
      if (drag.dragging) {
        suppressClickRef.current = true;
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
  }, [stopAutoScroll, updateDragSelection]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") clearSelection();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [clearSelection]);

  useEffect(() => {
    if (selectedIds.size === 0) return;
    const onPointerDown = (event: PointerEvent) => {
      const listRoot = listRootRef.current;
      if (!listRoot) return;
      if (event.target instanceof Element && listRoot.contains(event.target)) {
        return;
      }
      clearSelection();
    };
    window.addEventListener("pointerdown", onPointerDown);
    return () => window.removeEventListener("pointerdown", onPointerDown);
  }, [clearSelection, listRootRef, selectedIds.size]);

  const handleItemPointerDown = useCallback(
    (index: number, event: ReactPointerEvent) => {
      if (event.button !== 0 || event.pointerType !== "mouse") return;
      dragRef.current = {
        anchorIndex: index,
        pointerId: event.pointerId,
        pointerType: event.pointerType,
        startX: event.clientX,
        startY: event.clientY,
        lastClientY: event.clientY,
        dragging: false,
      };
    },
    [],
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
      if (event.shiftKey && anchorIndexRef.current != null) {
        applyRangeSelection(anchorIndexRef.current, index);
        return true;
      }
      if (modifier) {
        toggleId(id);
        anchorIndexRef.current = index;
        return true;
      }
      if (selectedIds.size > 0) {
        toggleId(id);
        anchorIndexRef.current = index;
        return true;
      }
      anchorIndexRef.current = index;
      return false;
    },
    [applyRangeSelection, selectedIds.size, toggleId],
  );

  const isItemSelected = useCallback(
    (id: string) => selectedIds.has(id),
    [selectedIds],
  );

  return {
    selectedIds,
    selectedCount: selectedIds.size,
    isSelectionActive: selectedIds.size > 0,
    clearSelection,
    handleItemPointerDown,
    handleItemClick,
    isItemSelected,
  };
}
