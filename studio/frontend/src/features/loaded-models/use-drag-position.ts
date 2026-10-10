// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Absolute viewport coordinates, not a transform, so the position survives reloads and can be clamped.

import { useCallback, useEffect, useRef, useState } from "react";

export type DragPosition = { left: number; top: number };

const MARGIN = 8;

// Below this a press is a click, so the pill can be both drag handle and expand button.
const DRAG_THRESHOLD_PX = 4;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export type Viewport = { width: number; height: number };

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
  position: DragPosition | null;
  /** Callback ref: the reclamp effect must re-run when the node appears. */
  panelRef: (node: HTMLDivElement | null) => void;
  startDrag: (event: React.PointerEvent<HTMLElement>) => void;
  dragging: boolean;
  /** True once, for the click ending a drag; reading clears it so keyboard activation still works. */
  justDragged: () => boolean;
};


export function useDragPosition(storageKey: string): UseDragPosition {
  const [position, setPosition] = useState<DragPosition | null>(() => {
    if (typeof window === "undefined") return null;
    const stored = readStored(storageKey);
    // Clamp on read too: the observer fires only after first paint, so avoid an off-screen flash.
    return stored ? clampToViewport(stored, 0, 0, viewport()) : null;
  });
  const [pressing, setPressing] = useState(false);
  const [dragging, setDragging] = useState(false);
  const movedRef = useRef(false);
  const panelRef = useRef<HTMLDivElement | null>(null);
  // The card renders nothing until the first poll, so the effect's first run sees a null ref.
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
    lastLeft: number;
    lastTop: number;
  } | null>(null);
  // Drag paints via the DOM, not state, so a frame costs no render.
  const frameRef = useRef(0);
  const pendingRef = useRef<{ dx: number; dy: number } | null>(null);

  // Return the same object when unchanged: this runs from a ResizeObserver and would loop forever.
  const reclamp = useCallback((width: number, height: number) => {
    setPosition((current) => {
      if (!current) return current;
      const next = clampToViewport(current, width, height, viewport());
      return next.left === current.left && next.top === current.top
        ? current
        : next;
    });
  }, []);

  // Not keyed on `position`, or every drag frame rebuilds the observer and forces a layout.
  useEffect(() => {
    if (!panelEl) return;
    const measure = () => {
      const box = panelEl.getBoundingClientRect();
      reclamp(box.width, box.height);
    };
    window.addEventListener("resize", measure);
    const observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(measure);
    observer?.observe(panelEl);
    if (!observer) measure();
    return () => {
      window.removeEventListener("resize", measure);
      observer?.disconnect();
    };
  }, [panelEl, reclamp]);

  const startDrag = useCallback(
    (event: React.PointerEvent<HTMLElement>) => {
      const panel = panelRef.current;
      if (event.button !== 0 || !panel) return;
      // Without capture a pointerup over another window is never delivered.
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
        lastLeft: box.left,
        lastTop: box.top,
      };
      movedRef.current = false;
      setPressing(true);
    },
    [],
  );

  // One paint per frame via translate3d: trackpads outpace the display and left/top repaints the shadow.
  const applyPending = useCallback(() => {
    const session = sessionRef.current;
    const move = pendingRef.current;
    const panel = panelRef.current;
    pendingRef.current = null;
    if (!(session && move && panel)) {
      return;
    }
    const next = clampToViewport(
      { left: session.left + move.dx, top: session.top + move.dy },
      session.width,
      session.height,
      viewport(),
    );
    session.lastLeft = next.left;
    session.lastTop = next.top;
    panel.style.transform = `translate3d(${next.left - session.left}px, ${
      next.top - session.top
    }px, 0)`;
  }, []);

  const paint = useCallback(() => {
    frameRef.current = 0;
    applyPending();
  }, [applyPending]);

  // Written to the node as well as state so dropping cannot flash at the old spot.
  const settle = useCallback(() => {
    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = 0;
    }
    // Fold in an owed frame, or a mid-flick release lands behind the pointer.
    applyPending();
    const session = sessionRef.current;
    const panel = panelRef.current;
    const moved = Boolean(session && movedRef.current);
    if (panel) {
      if (moved && session) {
        panel.style.left = `${session.lastLeft}px`;
        panel.style.top = `${session.lastTop}px`;
      }
      panel.style.transform = "";
    }
    if (moved && session) {
      const landed = { left: session.lastLeft, top: session.lastTop };
      setPosition(landed);
      // Persist only user-chosen positions; persisting a reclamp let a small window overwrite a large one's.
      store(storageKey, landed);
    }
    sessionRef.current = null;
    setPressing(false);
    setDragging(false);
  }, [applyPending, storageKey]);

  // Pin the panel where it sits so the first move does not jump.
  const beginDrag = useCallback((left: number, top: number) => {
    movedRef.current = true;
    setDragging(true);
    setPosition((current) => current ?? { left, top });
  }, []);

  const onMove = useCallback(
    (event: PointerEvent) => {
      const session = sessionRef.current;
      if (!session || session.pointerId !== event.pointerId) return;
      // Released somewhere unseen: end the drag.
      if (event.buttons === 0) {
        settle();
        return;
      }
      const dx = event.clientX - session.startX;
      const dy = event.clientY - session.startY;
      if (!movedRef.current) {
        if (!passedDragThreshold(dx, dy)) return;
        beginDrag(session.left, session.top);
      }
      event.preventDefault();
      pendingRef.current = { dx, dy };
      if (!frameRef.current) {
        frameRef.current = requestAnimationFrame(paint);
      }
    },
    [settle, beginDrag, paint],
  );

  const onEnd = useCallback(
    (event: PointerEvent) => {
      const session = sessionRef.current;
      if (session && session.pointerId !== event.pointerId) return;
      settle();
    },
    [settle],
  );

  useEffect(() => {
    if (!pressing) return;
    window.addEventListener("pointermove", onMove, { passive: false });
    window.addEventListener("pointerup", onEnd);
    window.addEventListener("pointercancel", onEnd);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onEnd);
      window.removeEventListener("pointercancel", onEnd);
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = 0;
      }
    };
  }, [pressing, onMove, onEnd]);

  const justDragged = useCallback(() => {
    const moved = movedRef.current;
    movedRef.current = false;
    return moved;
  }, []);

  return { position, panelRef: attachPanel, startDrag, dragging, justDragged };
}
