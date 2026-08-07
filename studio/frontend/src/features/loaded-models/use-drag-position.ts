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

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function clampToViewport(
  position: DragPosition,
  width: number,
  height: number,
): DragPosition {
  return {
    left: clamp(position.left, MARGIN, Math.max(MARGIN, innerWidth - width - MARGIN)),
    top: clamp(position.top, MARGIN, Math.max(MARGIN, innerHeight - height - MARGIN)),
  };
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
  panelRef: React.RefObject<HTMLDivElement | null>;
  startDrag: (event: React.PointerEvent<HTMLElement>) => void;
  dragging: boolean;
  /** Send it back to its corner. */
  reset: () => void;
};

export function useDragPosition(storageKey: string): UseDragPosition {
  const [position, setPosition] = useState<DragPosition | null>(() =>
    typeof window === "undefined" ? null : readStored(storageKey),
  );
  const [dragging, setDragging] = useState(false);
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

  // A window that shrank while the panel sat near an edge would strand it
  // off screen, and nothing else can move it back.
  useEffect(() => {
    if (!position) return;
    const onResize = () => {
      const box = panelRef.current?.getBoundingClientRect();
      if (!box) return;
      setPosition((current) =>
        current ? clampToViewport(current, box.width, box.height) : current,
      );
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [position]);

  const startDrag = useCallback((event: React.PointerEvent<HTMLElement>) => {
    const panel = panelRef.current;
    if (event.button !== 0 || !panel) return;
    event.preventDefault();
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
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
    // Anchor to the measured box first, so the first move is not a jump from
    // the corner the panel was flowing in.
    setPosition(clampToViewport({ left: box.left, top: box.top }, box.width, box.height));
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (event: PointerEvent) => {
      const session = sessionRef.current;
      if (!session || session.pointerId !== event.pointerId) return;
      setPosition(
        clampToViewport(
          {
            left: session.left + (event.clientX - session.startX),
            top: session.top + (event.clientY - session.startY),
          },
          session.width,
          session.height,
        ),
      );
    };
    const onEnd = (event: PointerEvent) => {
      const session = sessionRef.current;
      if (session && session.pointerId !== event.pointerId) return;
      sessionRef.current = null;
      setDragging(false);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onEnd);
    window.addEventListener("pointercancel", onEnd);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onEnd);
      window.removeEventListener("pointercancel", onEnd);
    };
  }, [dragging]);

  // Persist the resting place only, not every frame of the drag.
  useEffect(() => {
    if (dragging) return;
    store(storageKey, position);
  }, [dragging, position, storageKey]);

  const reset = useCallback(() => setPosition(null), []);

  return { position, panelRef, startDrag, dragging, reset };
}
