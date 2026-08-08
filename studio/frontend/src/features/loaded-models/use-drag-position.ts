// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pointer drag and resize for the indicator, in the Live monitor's idiom:
// anchored to its corner until the user moves it, then kept where they left it.
// Absolute viewport coordinates rather than a transform, so the geometry
// survives a reload and can be clamped when the window changes size.

import { useCallback, useEffect, useRef, useState } from "react";

export type DragPosition = { left: number; top: number };
export type PanelSize = { width: number; height: number };

/** Keeps the panel fully on screen, and off the very edge. */
const MARGIN = 8;

// Below this the press is a click, so the collapsed pill can be both a drag
// handle and the button that expands it.
const DRAG_THRESHOLD_PX = 4;

// Small enough to be worth shrinking to, wide enough that a row still fits its
// label and eject button, tall enough to keep the header and one row.
const MIN_WIDTH = 216;
const MIN_HEIGHT = 116;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export type Viewport = { width: number; height: number };

/** Keeps the whole panel on screen. Exported pure for the node suite. */
export function clampToViewport(
  position: DragPosition,
  width: number,
  height: number,
  viewport: Viewport,
): DragPosition {
  return {
    left: clamp(
      position.left,
      MARGIN,
      Math.max(MARGIN, viewport.width - width - MARGIN),
    ),
    top: clamp(
      position.top,
      MARGIN,
      Math.max(MARGIN, viewport.height - height - MARGIN),
    ),
  };
}

/** Whether a press has moved far enough to be a drag rather than a click. */
export function passedDragThreshold(dx: number, dy: number): boolean {
  return Math.hypot(dx, dy) >= DRAG_THRESHOLD_PX;
}

/** Keeps a resized panel between its floor and the room it was given. */
export function clampSize(
  size: PanelSize,
  maxWidth: number,
  maxHeight: number,
): PanelSize {
  return {
    width: clamp(size.width, MIN_WIDTH, Math.max(MIN_WIDTH, maxWidth)),
    height: clamp(size.height, MIN_HEIGHT, Math.max(MIN_HEIGHT, maxHeight)),
  };
}

export type ResizeStart = DragPosition & PanelSize;

/**
 * Resize from the top-left grip. The card is anchored bottom-right, where there
 * is no room to grow, so that corner stays put and the box opens up and to the
 * left instead. Exported pure for the node suite.
 */
export function resizeFromTopLeft(
  start: ResizeStart,
  dx: number,
  dy: number,
): { position: DragPosition; size: PanelSize } {
  const right = start.left + start.width;
  const bottom = start.top + start.height;
  // Only what lies that side of the held corner is available, less the margin.
  const size = clampSize(
    { width: start.width - dx, height: start.height - dy },
    right - MARGIN,
    bottom - MARGIN,
  );
  // Derived from the clamped size, so the held corner cannot drift.
  return {
    position: { left: right - size.width, top: bottom - size.height },
    size,
  };
}

function viewport(): Viewport {
  return { width: window.innerWidth, height: window.innerHeight };
}

/** One stored `{ left, top }` or `{ width, height }`, or null if unusable. */
function readStored<T extends DragPosition | PanelSize>(
  key: string,
  keys: (keyof T)[],
): T | null {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<Record<keyof T, unknown>>;
    return keys.every((name) => typeof parsed[name] === "number")
      ? (parsed as T)
      : null;
  } catch {
    return null;
  }
}

function store(key: string, value: object | null): void {
  try {
    if (value) {
      localStorage.setItem(key, JSON.stringify(value));
    } else {
      localStorage.removeItem(key);
    }
  } catch {
    // storage unavailable
  }
}

export type UseDragPosition = {
  /** null until the user moves it, so the default anchor still applies. */
  position: DragPosition | null;
  /** null until the user resizes it, so the card keeps its natural size. */
  size: PanelSize | null;
  panelRef: React.RefObject<HTMLDivElement | null>;
  startDrag: (event: React.PointerEvent<HTMLElement>) => void;
  startResize: (event: React.PointerEvent<HTMLElement>) => void;
  dragging: boolean;
  resizing: boolean;
  /** Back to the natural size and the default corner. */
  reset: () => void;
  /** True once, for the click that ends a drag, so a handle can also be a
   *  button. Reading it clears the flag, so a later keyboard activation on the
   *  same button is not swallowed too. */
  justDragged: () => boolean;
};

export type PanelStorageKeys = { position: string; size: string };

export function useDragPosition(keys: PanelStorageKeys): UseDragPosition {
  const [position, setPosition] = useState<DragPosition | null>(() =>
    typeof window === "undefined"
      ? null
      : readStored<DragPosition>(keys.position, ["left", "top"]),
  );
  const [size, setSize] = useState<PanelSize | null>(() =>
    typeof window === "undefined"
      ? null
      : readStored<PanelSize>(keys.size, ["width", "height"]),
  );
  // Held from pointerdown to pointerup; dragging only once past the threshold.
  const [pressing, setPressing] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [resizing, setResizing] = useState(false);
  const movedRef = useRef(false);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const sessionRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    left: number;
    top: number;
    width: number;
    height: number;
  } | null>(null);
  const resizeSessionRef = useRef<
    (ResizeStart & { pointerId: number; startX: number; startY: number }) | null
  >(null);

  // Returning the same object when nothing changed matters: this also runs from
  // a ResizeObserver, and a fresh object every time would re-render forever.
  const reclamp = useCallback((width: number, height: number) => {
    setPosition((current) => {
      if (!current) return current;
      const next = clampToViewport(current, width, height, viewport());
      return next.left === current.left && next.top === current.top
        ? current
        : next;
    });
  }, []);

  // A window that shrank, or a panel that grew when expanded, would otherwise
  // strand it off screen with nothing able to bring it back.
  // Skipped mid-resize: resizeFromTopLeft already holds the box on screen, and
  // clamping against a size that is still changing would fight it.
  useEffect(() => {
    if (!position || resizing) return;
    const panel = panelRef.current;
    const measure = () => {
      const box = panel?.getBoundingClientRect();
      if (box) reclamp(box.width, box.height);
    };
    window.addEventListener("resize", measure);
    const observer = panel ? new ResizeObserver(measure) : null;
    if (panel && observer) observer.observe(panel);
    return () => {
      window.removeEventListener("resize", measure);
      observer?.disconnect();
    };
  }, [position, resizing, reclamp]);

  const startDrag = useCallback((event: React.PointerEvent<HTMLElement>) => {
    const panel = panelRef.current;
    if (event.button !== 0 || !panel) return;
    const box = panel.getBoundingClientRect();
    sessionRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      left: box.left,
      top: box.top,
      width: box.width,
      height: box.height,
    };
    movedRef.current = false;
    setPressing(true);
  }, []);

  useEffect(() => {
    if (!pressing) return;
    const onMove = (event: PointerEvent) => {
      const session = sessionRef.current;
      if (!session || session.pointerId !== event.pointerId) return;
      const dx = event.clientX - session.startX;
      const dy = event.clientY - session.startY;
      if (!movedRef.current) {
        if (!passedDragThreshold(dx, dy)) return;
        movedRef.current = true;
        setDragging(true);
      }
      // Text selection would otherwise start mid-drag on the pill's label.
      event.preventDefault();
      setPosition(
        clampToViewport(
          { left: session.left + dx, top: session.top + dy },
          session.width,
          session.height,
          viewport(),
        ),
      );
    };
    const onEnd = (event: PointerEvent) => {
      const session = sessionRef.current;
      if (session && session.pointerId !== event.pointerId) return;
      sessionRef.current = null;
      setPressing(false);
      setDragging(false);
    };
    window.addEventListener("pointermove", onMove, { passive: false });
    window.addEventListener("pointerup", onEnd);
    window.addEventListener("pointercancel", onEnd);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onEnd);
      window.removeEventListener("pointercancel", onEnd);
    };
  }, [pressing]);

  const startResize = useCallback((event: React.PointerEvent<HTMLElement>) => {
    const panel = panelRef.current;
    if (event.button !== 0 || !panel) return;
    // The grip sits on the card, which the header drag handle does not cover,
    // but stop here anyway so a resize can never also start a drag.
    event.preventDefault();
    event.stopPropagation();
    const box = panel.getBoundingClientRect();
    resizeSessionRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      left: box.left,
      top: box.top,
      width: box.width,
      height: box.height,
    };
    // Pin it: the card flows in the bottom-right stack until now, and holding a
    // corner still means owning both the position and the size.
    setPosition({ left: box.left, top: box.top });
    setSize({ width: box.width, height: box.height });
    setResizing(true);
  }, []);

  useEffect(() => {
    if (!resizing) return;
    const onMove = (event: PointerEvent) => {
      const session = resizeSessionRef.current;
      if (!session || session.pointerId !== event.pointerId) return;
      event.preventDefault();
      const next = resizeFromTopLeft(
        session,
        event.clientX - session.startX,
        event.clientY - session.startY,
      );
      setPosition(next.position);
      setSize(next.size);
    };
    const onEnd = (event: PointerEvent) => {
      const session = resizeSessionRef.current;
      if (session && session.pointerId !== event.pointerId) return;
      resizeSessionRef.current = null;
      setResizing(false);
    };
    window.addEventListener("pointermove", onMove, { passive: false });
    window.addEventListener("pointerup", onEnd);
    window.addEventListener("pointercancel", onEnd);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onEnd);
      window.removeEventListener("pointercancel", onEnd);
    };
  }, [resizing]);

  // Persist the resting geometry only, not every frame of a drag or resize.
  useEffect(() => {
    if (pressing || resizing) return;
    store(keys.position, position);
    store(keys.size, size);
  }, [pressing, resizing, position, size, keys.position, keys.size]);

  const reset = useCallback(() => {
    setPosition(null);
    setSize(null);
  }, []);

  const justDragged = useCallback(() => {
    const moved = movedRef.current;
    movedRef.current = false;
    return moved;
  }, []);

  return {
    position,
    size,
    panelRef,
    startDrag,
    startResize,
    dragging,
    resizing,
    reset,
    justDragged,
  };
}
