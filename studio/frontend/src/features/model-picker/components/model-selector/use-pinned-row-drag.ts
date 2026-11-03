// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pointer events, not HTML5 drag: the desktop webview never forwards dragover or drop.

import { useCallback, useEffect, useRef, useState } from "react";

import {
  DRAG_THRESHOLD_PX,
  liftCopy,
  markDragging,
  placeCue,
  placeGhost,
  type RowGhost,
} from "@/features/chat";
import { prefersReducedMotion } from "@/features/settings";

export type PinnedDropEdge = "top" | "bottom";

const SCOPE_ATTR = "data-pinned-drop-scope";
const KEY_ATTR = "data-pinned-drop-key";

const NO_DRAG_SELECTOR =
  "button:not([data-model-picker-option]), a[href], [role='menuitem']";

const ROW_GHOST_CLASS = "model-picker-row-ghost";

const GHOST_DROPPED_ATTRS = [
  "id",
  SCOPE_ATTR,
  KEY_ATTR,
  "data-model-picker-option",
  "data-model-picker-active-option",
  "aria-current",
  "data-state",
  "data-slot",
] as const;

const SETTLE_MS = 180;

const NEAR_ROW_PX = 12;

const EDGE_PX = 40;
const EDGE_STEP_PX = 10;

export interface PinnedRowDrop {
  key: string;
  edge: PinnedDropEdge;
}

function scrollerOf(element: Element | null): HTMLElement | null {
  for (let node = element; node; node = node.parentElement) {
    if (!(node instanceof HTMLElement)) continue;
    const overflow = getComputedStyle(node).overflowY;
    if (
      (overflow === "auto" || overflow === "scroll") &&
      node.scrollHeight > node.clientHeight
    ) {
      return node;
    }
  }
  return null;
}

function viewOf(row: HTMLElement): Element {
  for (let node = row.parentElement; node; node = node.parentElement) {
    const overflow = getComputedStyle(node).overflowY;
    if (overflow === "auto" || overflow === "scroll") return node;
  }
  return row.parentElement ?? document.documentElement;
}

const ROW_FACE_ATTR = "data-pinned-row-face";

const faceOf = (row: Element): HTMLElement =>
  row.querySelector<HTMLElement>(`[${ROW_FACE_ATTR}]`) ??
  (row.firstElementChild as HTMLElement | null) ??
  (row as HTMLElement);

function settleRow(scope: string, key: string, from: number) {
  // Two frames: measured after the drop re-renders the list.
  requestAnimationFrame(() =>
    requestAnimationFrame(() => {
      const row = document.querySelector<HTMLElement>(
        `[${SCOPE_ATTR}="${CSS.escape(scope)}"][${KEY_ATTR}="${CSS.escape(key)}"]`,
      );
      if (!row) return;
      const delta = from - faceOf(row).getBoundingClientRect().top;
      if (Math.abs(delta) < 2) return;
      row.style.transition = "none";
      row.style.transform = `translateY(${delta}px)`;
      row.style.zIndex = "1";
      row.getBoundingClientRect();
      row.style.transition = `transform ${SETTLE_MS}ms cubic-bezier(0.2, 0.8, 0.2, 1)`;
      row.style.transform = "";
      window.setTimeout(() => {
        row.style.transition = "";
        row.style.zIndex = "";
      }, SETTLE_MS + 20);
    }),
  );
}

function edgeAt(rect: DOMRect, y: number): PinnedDropEdge {
  return y < rect.top + rect.height / 2 ? "top" : "bottom";
}

function rowUnder(scope: string, x: number, y: number): PinnedRowDrop | null {
  if (typeof document === "undefined") return null;
  for (const element of document.elementsFromPoint(x, y)) {
    const row = element.closest(`[${KEY_ATTR}]`);
    if (!row || row.getAttribute(SCOPE_ATTR) !== scope) continue;
    const key = row.getAttribute(KEY_ATTR);
    if (key) return { key, edge: edgeAt(row.getBoundingClientRect(), y) };
  }
  let nearest: { drop: PinnedRowDrop; distance: number } | null = null;
  for (const row of document.querySelectorAll(
    `[${SCOPE_ATTR}="${CSS.escape(scope)}"]`,
  )) {
    const key = row.getAttribute(KEY_ATTR);
    const rect = row.getBoundingClientRect();
    if (!key || x < rect.left || x > rect.right) continue;
    const distance =
      y < rect.top ? rect.top - y : y > rect.bottom ? y - rect.bottom : 0;
    if (distance > NEAR_ROW_PX) continue;
    if (!nearest || distance < nearest.distance) {
      nearest = { drop: { key, edge: edgeAt(rect, y) }, distance };
    }
  }
  return nearest?.drop ?? null;
}

function staysPut(
  order: readonly string[],
  fromKey: string,
  drop: PinnedRowDrop,
): boolean {
  const from = order.indexOf(fromKey);
  const target = order.indexOf(drop.key);
  if (from < 0 || target < 0) return true;
  const slot = target + (drop.edge === "bottom" ? 1 : 0);
  return slot === from || slot === from + 1;
}

export interface UsePinnedRowDragOptions {
  scope: string;
  order: () => readonly string[];
  onDrop: (fromKey: string, drop: PinnedRowDrop) => void;
}

export interface PinnedRowDragApi {
  draggingKey: string | null;
  lineEdge: (key: string) => PinnedDropEdge | undefined;
  rowProps: (key: string) => {
    onPointerDown: (event: React.PointerEvent) => void;
    onKeyDownCapture: (event: React.KeyboardEvent) => void;
    onDragStart: (event: React.DragEvent) => void;
    [SCOPE_ATTR]: string;
    [KEY_ATTR]: string;
  };
}

export function usePinnedRowDrag(
  options: UsePinnedRowDragOptions,
): PinnedRowDragApi {
  const [draggingKey, setDraggingKey] = useState<string | null>(null);
  const [target, setTarget] = useState<PinnedRowDrop | null>(null);
  const optionsRef = useRef(options);
  useEffect(() => {
    optionsRef.current = options;
  });
  const draggingRef = useRef<string | null>(null);
  const targetRef = useRef<PinnedRowDrop | null>(null);
  const scroller = useRef<HTMLElement | null>(null);
  const ghost = useRef<RowGhost | null>(null);
  const press = useRef<{ end: () => void } | null>(null);

  const showTarget = useCallback((next: PinnedRowDrop | null) => {
    const current = targetRef.current;
    if (current?.key === next?.key && current?.edge === next?.edge) return;
    targetRef.current = next;
    setTarget(next);
  }, []);

  const clear = useCallback(() => {
    draggingRef.current = null;
    scroller.current = null;
    ghost.current?.element.remove();
    for (const overlay of ghost.current?.cues ?? []) overlay.remove();
    ghost.current = null;
    markDragging(null, false);
    setDraggingKey(null);
    showTarget(null);
  }, [showTarget]);

  useEffect(
    () => () => {
      press.current?.end();
    },
    [],
  );

  const aim = useCallback((x: number, y: number): PinnedRowDrop | null => {
    const fromKey = draggingRef.current;
    if (!fromKey) return null;
    const { scope, order } = optionsRef.current;
    const drop = rowUnder(scope, x, y);
    if (!drop) return null;
    const keys = order();
    if (staysPut(keys, fromKey, drop)) return null;
    // One line per gap: a row's bottom is drawn as the next row's top.
    const next = drop.edge === "bottom" ? keys[keys.indexOf(drop.key) + 1] : undefined;
    return next ? { key: next, edge: "top" } : drop;
  }, []);

  const rowProps = useCallback(
    (key: string) => ({
      [SCOPE_ATTR]: optionsRef.current.scope,
      [KEY_ATTR]: key,
      // A native drag of the row's link or text would cancel the pointer stream.
      onDragStart: (event: React.DragEvent) => event.preventDefault(),
      onKeyDownCapture: (event: React.KeyboardEvent) => {
        if (!event.altKey || event.shiftKey || event.metaKey || event.ctrlKey) {
          return;
        }
        if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
        // Before the list's own arrow handling moves focus.
        event.preventDefault();
        event.stopPropagation();
        const order = optionsRef.current.order();
        const index = order.indexOf(key);
        const up = event.key === "ArrowUp";
        const neighbour = order[up ? index - 1 : index + 1];
        if (index < 0 || !neighbour) return;
        optionsRef.current.onDrop(key, {
          key: neighbour,
          edge: up ? "top" : "bottom",
        });
        const scope = optionsRef.current.scope;
        requestAnimationFrame(() => {
          const row = document.querySelector(
            `[${SCOPE_ATTR}="${CSS.escape(scope)}"][${KEY_ATTR}="${CSS.escape(key)}"]`,
          );
          const option = row?.querySelector<HTMLElement>(
            "[data-model-picker-option]",
          );
          if (option && document.activeElement !== option) option.focus();
        });
      },
      onPointerDown: (event: React.PointerEvent) => {
        if (event.button !== 0 || event.pointerType === "touch") return;
        if (!event.isPrimary) return;
        if ((event.target as Element | null)?.closest?.(NO_DRAG_SELECTOR)) {
          return;
        }
        // Drop a stale gesture whose release was missed, or it blocks every later drag.
        press.current?.end();

        const startX = event.clientX;
        const startY = event.clientY;
        const row = event.currentTarget as HTMLElement;
        const pointerId = event.pointerId;
        const at = { x: startX, y: startY };
        let started = false;
        let escaped = false;
        let frame = 0;

        const detach = () => {
          if (press.current === self) press.current = null;
          if (frame) cancelAnimationFrame(frame);
          frame = 0;
          window.removeEventListener("pointermove", onMove);
          window.removeEventListener("pointerup", onUp);
          window.removeEventListener("pointercancel", onCancel);
          window.removeEventListener("keydown", onKey, true);
          if (document.body.hasPointerCapture?.(pointerId)) {
            document.body.releasePointerCapture(pointerId);
          }
        };
        // Re-aim every frame: the list can scroll under a resting pointer.
        const onFrame = () => {
          if (!draggingRef.current) {
            frame = 0;
            return;
          }
          frame = requestAnimationFrame(onFrame);
          const list = scroller.current;
          if (list) {
            const rect = list.getBoundingClientRect();
            if (at.y < rect.top + EDGE_PX) list.scrollTop -= EDGE_STEP_PX;
            else if (at.y > rect.bottom - EDGE_PX) {
              list.scrollTop += EDGE_STEP_PX;
            }
          }
          if (ghost.current) placeGhost(ghost.current, at.y);
          showTarget(aim(at.x, at.y));
          // Redraw the drop line above the copy.
          if (ghost.current) placeCue(ghost.current);
        };
        const self = {
          end: () => {
            detach();
            if (started) clear();
          },
        };
        press.current = self;

        // Swallow the release's click, which would load the row it lands on.
        const swallowClick = () => {
          const stop = (clicked: MouseEvent) => {
            clicked.preventDefault();
            clicked.stopPropagation();
          };
          window.addEventListener("click", stop, { capture: true, once: true });
          window.setTimeout(() => {
            window.removeEventListener("click", stop, { capture: true });
          }, 0);
        };

        function onMove(moved: PointerEvent) {
          if (moved.pointerId !== pointerId || escaped) return;
          at.x = moved.clientX;
          at.y = moved.clientY;
          if (!started) {
            if (
              Math.abs(moved.clientX - startX) < DRAG_THRESHOLD_PX &&
              Math.abs(moved.clientY - startY) < DRAG_THRESHOLD_PX
            ) {
              return;
            }
            started = true;
            scroller.current = scrollerOf(row);
            ghost.current = liftCopy(
              faceOf(row),
              startY,
              scroller.current ?? viewOf(row),
              ROW_GHOST_CLASS,
              GHOST_DROPPED_ATTRS,
            );
            placeGhost(ghost.current, moved.clientY);
            markDragging(scroller.current ?? row.parentElement, true);
            try {
              document.body.setPointerCapture(pointerId);
            } catch {
              // No capture: window listeners still carry the drag inside the window.
            }
            draggingRef.current = key;
            setDraggingKey(key);
            frame = requestAnimationFrame(onFrame);
          }
          moved.preventDefault();
          showTarget(aim(moved.clientX, moved.clientY));
        }

        function onUp(released: PointerEvent) {
          if (released.pointerId !== pointerId) return;
          detach();
          if (!started) return;
          swallowClick();
          if (escaped) return;
          const drop = aim(released.clientX, released.clientY);
          const from = ghost.current?.element.getBoundingClientRect().top ?? null;
          clear();
          if (!drop) return;
          optionsRef.current.onDrop(key, drop);
          if (from !== null && !prefersReducedMotion()) {
            settleRow(optionsRef.current.scope, key, from);
          }
        }

        function onCancel(aborted: PointerEvent) {
          if (aborted.pointerId !== pointerId) return;
          detach();
          if (started) clear();
        }

        function onKey(pressed: KeyboardEvent) {
          if (pressed.key !== "Escape" || escaped) return;
          if (!started) {
            detach();
            return;
          }
          pressed.preventDefault();
          pressed.stopPropagation();
          // Keep listening so the pending release is still swallowed.
          escaped = true;
          clear();
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        window.addEventListener("pointercancel", onCancel);
        window.addEventListener("keydown", onKey, true);
      },
    }),
    [aim, clear, showTarget],
  );

  const lineEdge = useCallback(
    (key: string): PinnedDropEdge | undefined =>
      target?.key === key ? target.edge : undefined,
    [target],
  );

  return { draggingKey, lineEdge, rowProps };
}
