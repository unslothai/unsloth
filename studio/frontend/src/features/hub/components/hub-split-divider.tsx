// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
  type RefObject,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  acquireDragOverlay,
  releaseDragOverlay,
} from "@/components/ui/panel-drag-overlay";

import {
  clampSplitListWidth,
  keyboardSplitWidth,
  SPLIT_LIST_MIN_PX,
  splitListMax,
} from "../lib/split-width";

const SPLIT_LIST_WIDTH_STORAGE_KEY = "unsloth.hub.splitListWidth";
const DRAG_SLOP = 4;

export const HUB_SPLIT_WIDTH_VAR = "--hub-split-list-width";
// Re-clamped in CSS so a stored width stays valid when the window narrows.
export const HUB_SPLIT_CUSTOM_WIDTH_CLASS =
  "lg:w-[clamp(340px,var(--hub-split-list-width),calc(100%_-_480px))]";

function readStoredWidth(): number | null {
  try {
    const value = Number(
      window.localStorage.getItem(SPLIT_LIST_WIDTH_STORAGE_KEY),
    );
    return Number.isFinite(value) && value >= SPLIT_LIST_MIN_PX ? value : null;
  } catch {
    return null;
  }
}

/** The split list pane's width in px, or null for the default responsive width. */
export function useHubSplitWidth(): {
  width: number | null;
  setWidth: (px: number | null) => void;
  style: CSSProperties | undefined;
} {
  const [width, setWidthState] = useState<number | null>(readStoredWidth);
  const setWidth = useCallback((px: number | null) => {
    const next = px === null ? null : Math.round(px);
    setWidthState(next);
    try {
      if (next === null) {
        window.localStorage.removeItem(SPLIT_LIST_WIDTH_STORAGE_KEY);
      } else {
        window.localStorage.setItem(SPLIT_LIST_WIDTH_STORAGE_KEY, String(next));
      }
    } catch {
      // Storage unavailable: the width still applies for this session.
    }
  }, []);
  const style =
    width === null
      ? undefined
      : ({ [HUB_SPLIT_WIDTH_VAR]: `${width}px` } as CSSProperties);
  return { width, setWidth, style };
}

type HubSplitDividerProps = {
  paneRef: RefObject<HTMLElement | null>;
  width: number | null;
  onWidthChange: (px: number | null) => void;
};

/**
 * Draggable edge between the split list and detail panes. Arrow keys resize;
 * Home or a double-click restores the default width.
 */
export function HubSplitDivider({
  paneRef,
  width,
  onWidthChange,
}: HubSplitDividerProps) {
  const dragRef = useRef<{
    startX: number;
    startWidth: number;
    moved: boolean;
  } | null>(null);
  const frameRef = useRef(0);
  const pendingRef = useRef(0);
  const [bounds, setBounds] = useState({ now: 0, max: 0, total: 0 });

  const limits = useCallback(() => {
    const pane = paneRef.current;
    const total = pane?.parentElement?.clientWidth ?? 0;
    return {
      now: pane?.getBoundingClientRect().width ?? 0,
      max: splitListMax(total),
      total,
    };
  }, [paneRef]);

  // Keeps the separator's aria values in step with the rendered width.
  useEffect(() => {
    const pane = paneRef.current;
    if (!pane?.parentElement) return;
    const update = () => {
      const next = limits();
      setBounds((prev) =>
        Math.round(prev.now) === Math.round(next.now) && prev.max === next.max
          ? prev
          : next,
      );
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(pane);
    observer.observe(pane.parentElement);
    return () => observer.disconnect();
  }, [paneRef, limits]);

  const clamp = useCallback(
    (px: number) => clampSplitListWidth(px, limits().total),
    [limits],
  );

  const endDrag = useCallback(() => {
    dragRef.current = null;
    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = 0;
    }
    paneRef.current?.removeAttribute("data-resizing");
    releaseDragOverlay();
  }, [paneRef]);

  useEffect(
    () => () => {
      if (dragRef.current) endDrag();
    },
    [endDrag],
  );

  const handlePointerDown = (event: PointerEvent<HTMLButtonElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const start = limits().now;
    dragRef.current = {
      startX: event.clientX,
      startWidth: start,
      moved: false,
    };
    pendingRef.current = start;
    paneRef.current?.setAttribute("data-resizing", "true");
    acquireDragOverlay();
  };

  const handlePointerMove = (event: PointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const delta = event.clientX - drag.startX;
    if (!drag.moved) {
      if (Math.abs(delta) < DRAG_SLOP) return;
      drag.moved = true;
      // The default width is CSS-only; pin it so the var below takes over.
      if (width === null) onWidthChange(drag.startWidth);
    }
    pendingRef.current = clamp(drag.startWidth + delta);
    if (frameRef.current) return;
    // Painted straight onto the pane, one write per frame; the page re-renders only on release.
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = 0;
      paneRef.current?.style.setProperty(
        HUB_SPLIT_WIDTH_VAR,
        `${pendingRef.current}px`,
      );
    });
  };

  // Release, cancel and lost capture all commit what was painted, so the pane
  // never keeps a width that state and storage do not hold.
  const finishDrag = (event: PointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    endDrag();
    if (drag.moved) onWidthChange(pendingRef.current);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    const { now, total } = limits();
    const next = keyboardSplitWidth(now, event.key, total);
    if (next !== null) {
      event.preventDefault();
      onWidthChange(next);
    } else if (event.key === "Home") {
      event.preventDefault();
      onWidthChange(null);
    }
  };

  return (
    <div className="relative hidden w-0 shrink-0 lg:block">
      <button
        type="button"
        data-slot="hub-split-divider"
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize model list"
        aria-valuenow={Math.round(bounds.now)}
        aria-valuemin={SPLIT_LIST_MIN_PX}
        aria-valuemax={Math.round(bounds.max)}
        title="Drag to resize, double-click to reset"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={finishDrag}
        onPointerCancel={finishDrag}
        onLostPointerCapture={finishDrag}
        onDoubleClick={() => onWidthChange(null)}
        onKeyDown={handleKeyDown}
        className={
          // Centred on the list's 1px border, which the hover line recolours.
          "absolute inset-y-0 -left-1.5 z-30 w-3 cursor-col-resize! touch-none select-none outline-none after:absolute after:inset-y-0 after:left-1/2 after:-translate-x-full after:w-px after:bg-transparent after:transition-colors after:duration-150 hover:after:bg-sidebar-ring/40 focus-visible:after:bg-sidebar-ring/70 active:after:bg-sidebar-ring/40"
        }
      />
    </div>
  );
}
