// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pointer drag for the indicator, in the Live monitor's idiom: anchored to its
// corner until the user moves it, then kept where they left it. Absolute
// viewport coordinates rather than a transform, so the position survives a
// reload and can be clamped when the window changes size.

import { useCallback, useEffect, useRef, useState } from "react";

export type DragPosition = { left: number; top: number };

/** Keeps the panel fully on screen, and off the very edge. */
const MARGIN = 8;

// Below this the press is a click, so the collapsed pill can be both a drag
// handle and the button that expands it.
const DRAG_THRESHOLD_PX = 4;

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

function viewport(): Viewport {
  return { width: window.innerWidth, height: window.innerHeight };
}

function readStored(key: string): DragPosition | null {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<DragPosition>;
    return typeof parsed.left === "number" && typeof parsed.top === "number"
      ? { left: parsed.left, top: parsed.top }
      : null;
  } catch {
    return null;
  }
}

function store(key: string, position: DragPosition | null): void {
  try {
    if (position) {
      localStorage.setItem(key, JSON.stringify(position));
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
  /** Callback ref, not a RefObject: the reclamp effect has to re-run when the
   *  node appears, and a RefObject mutation does not re-render. */
  panelRef: (node: HTMLDivElement | null) => void;
  startDrag: (event: React.PointerEvent<HTMLElement>) => void;
  dragging: boolean;
  /** True once, for the click that ends a drag, so a handle can also be a
   *  button. Reading it clears the flag, so a later keyboard activation on the
   *  same button is not swallowed too. */
  justDragged: () => boolean;
};

export function useDragPosition(storageKey: string): UseDragPosition {
  const [position, setPosition] = useState<DragPosition | null>(() => {
    if (typeof window === "undefined") return null;
    const stored = readStored(storageKey);
    // Clamp on read as well as on resize. The observer below cannot fire until
    // after the first paint, so a position saved on a wider screen would flash
    // off screen once; zero width/height keeps at least the top-left corner in
    // view, and the first measurement refines it.
    return stored ? clampToViewport(stored, 0, 0, viewport()) : null;
  });
  // Held from pointerdown to pointerup; dragging only once past the threshold.
  const [pressing, setPressing] = useState(false);
  const [dragging, setDragging] = useState(false);
  const movedRef = useRef(false);
  const panelRef = useRef<HTMLDivElement | null>(null);
  // Mirrored into state because the reclamp effect must re-subscribe when the
  // node mounts: the card renders nothing until the first poll returns a row,
  // so the effect's first run sees a null ref.
  const [panelEl, setPanelEl] = useState<HTMLDivElement | null>(null);
  const attachPanel = useCallback((node: HTMLDivElement | null) => {
    panelRef.current = node;
    setPanelEl(node);
  }, []);
  const sessionRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    left: number;
    top: number;
    width: number;
    height: number;
  } | null>(null);

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
  useEffect(() => {
    if (!position || !panelEl) return;
    const measure = () => {
      const box = panelEl.getBoundingClientRect();
      reclamp(box.width, box.height);
    };
    window.addEventListener("resize", measure);
    const observer =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(measure);
    observer?.observe(panelEl);
    // No observer (an engine old enough to lack it) still gets the resize path,
    // plus this one measurement of whatever is on screen right now.
    if (!observer) measure();
    return () => {
      window.removeEventListener("resize", measure);
      observer?.disconnect();
    };
  }, [position, panelEl, reclamp]);

  const startDrag = useCallback((event: React.PointerEvent<HTMLElement>) => {
    const panel = panelRef.current;
    if (event.button !== 0 || !panel) return;
    // Without capture a pointerup over another window is never delivered, so
    // the card would keep tracking the cursor. Same as the Live monitor's drag.
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Capture is best effort; the window listeners below still drive the drag.
    }
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
      // The button was released somewhere we never saw it: end the drag rather
      // than following the cursor around.
      if (event.buttons === 0) {
        sessionRef.current = null;
        setPressing(false);
        setDragging(false);
        return;
      }
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

  // Persist the resting place only, not every frame of the drag.
  useEffect(() => {
    if (pressing) return;
    store(storageKey, position);
  }, [pressing, position, storageKey]);

  const justDragged = useCallback(() => {
    const moved = movedRef.current;
    movedRef.current = false;
    return moved;
  }, []);

  return { position, panelRef: attachPanel, startDrag, dragging, justDragged };
}
