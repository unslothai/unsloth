// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

interface UseResizableWidthOptions {
  storageKey: string;
  defaultWidth: number;
  minWidth: number;
  /** 0..1 fraction of viewport.innerWidth used as max width. Default 0.8. */
  maxWidthFraction?: number;
  /** Persist + listen for viewport-resize clamping only while true. */
  enabled?: boolean;
}

interface UseResizableWidthResult {
  width: number;
  isResizing: boolean;
  startResize: (event: ReactPointerEvent<HTMLElement>) => void;
  adjustWidth: (delta: number) => void;
  resetWidth: () => void;
}

function readStored(key: string, fallback: number): number {
  if (typeof window === "undefined") {
    return fallback;
  }
  try {
    const raw = window.localStorage.getItem(key);
    if (raw == null) {
      return fallback;
    }
    const parsed = Number.parseInt(raw, 10);
    return Number.isFinite(parsed) ? parsed : fallback;
  } catch {
    return fallback;
  }
}

/** Drag-to-resize hook for a right-anchored panel. The handle is on the
 *  panel's LEFT edge, so width grows as the pointer moves toward x=0.
 *  Persists to localStorage and re-clamps on viewport resize so a wide
 *  panel cannot eclipse the host content. */
export function useResizablePanelWidth({
  storageKey,
  defaultWidth,
  minWidth,
  maxWidthFraction = 0.8,
  enabled = true,
}: UseResizableWidthOptions): UseResizableWidthResult {
  const [width, setWidth] = useState<number>(() =>
    readStored(storageKey, defaultWidth),
  );
  const [isResizing, setIsResizing] = useState(false);
  const rafRef = useRef<number | null>(null);

  const clampWidth = useCallback(
    (next: number): number => {
      if (typeof window === "undefined") {
        return Math.max(minWidth, next);
      }
      const max = Math.floor(window.innerWidth * maxWidthFraction);
      return Math.max(minWidth, Math.min(max, next));
    },
    [minWidth, maxWidthFraction],
  );

  useEffect(() => {
    if (!enabled || typeof window === "undefined") {
      return;
    }
    try {
      window.localStorage.setItem(storageKey, String(width));
    } catch {
      // localStorage may be unavailable (private mode, quota); best-effort.
    }
  }, [width, storageKey, enabled]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const onResize = () => {
      setWidth((w) => clampWidth(w));
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [clampWidth]);

  const startResize = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => {
      if (!enabled) {
        return;
      }
      event.preventDefault();
      const target = event.currentTarget;
      const pointerId = event.pointerId;
      try {
        target.setPointerCapture(pointerId);
      } catch {
        // Pointer-capture isn't available everywhere (e.g. test envs).
      }
      setIsResizing(true);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";

      const onMove = (e: PointerEvent) => {
        if (rafRef.current !== null) {
          cancelAnimationFrame(rafRef.current);
        }
        rafRef.current = requestAnimationFrame(() => {
          setWidth(clampWidth(window.innerWidth - e.clientX));
        });
      };
      const cleanup = (e: PointerEvent) => {
        setIsResizing(false);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        try {
          target.releasePointerCapture(e.pointerId);
        } catch {
          // Already released or not captured.
        }
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", cleanup);
        window.removeEventListener("pointercancel", cleanup);
        if (rafRef.current !== null) {
          cancelAnimationFrame(rafRef.current);
          rafRef.current = null;
        }
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", cleanup);
      window.addEventListener("pointercancel", cleanup);
    },
    [enabled, clampWidth],
  );

  const adjustWidth = useCallback(
    (delta: number) => {
      setWidth((w) => clampWidth(w + delta));
    },
    [clampWidth],
  );

  const resetWidth = useCallback(() => {
    setWidth(clampWidth(defaultWidth));
  }, [clampWidth, defaultWidth]);

  return { width, isResizing, startResize, adjustWidth, resetWidth };
}
