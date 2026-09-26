// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Drag to reorder a Pinned group in the model selector, like pinned chats in the sidebar.
// Pointer events, not HTML5 drag: the desktop webview never forwards dragover or drop.

import { useCallback, useEffect, useRef, useState } from "react";

import { DRAGGING_BODY_CLASS, DRAG_THRESHOLD_PX } from "@/features/chat";

export type PinnedDropEdge = "top" | "bottom";

/** The row's group, so two Pinned groups never swap rows. */
const SCOPE_ATTR = "data-pinned-drop-scope";
/** Carries the row's pin key. */
const KEY_ATTR = "data-pinned-drop-key";

/** Controls with their own press: row buttons other than the row itself, and links. */
const NO_DRAG_SELECTOR =
  "button:not([data-model-picker-option]), a[href], [role='menuitem']";

/** How far outside a row the pointer still aims at it. */
const NEAR_ROW_PX = 12;

/** Edge auto-scroll zone and step per frame. */
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

function edgeAt(rect: DOMRect, y: number): PinnedDropEdge {
  return y < rect.top + rect.height / 2 ? "top" : "bottom";
}

/** The row of `scope` under the pointer, else the nearest one in reach. */
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

/** Whether dropping `fromKey` on this spot leaves it where it already is. */
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
  /** Tells this group's rows apart from other groups'. */
  scope: string;
  /** The group's pin keys as drawn. Read at event time. */
  order: () => readonly string[];
  /** Lands `fromKey` on the given edge of `drop.key`. */
  onDrop: (fromKey: string, drop: PinnedRowDrop) => void;
}

export interface PinnedRowDragApi {
  /** The lifted row, for painting it faded. Null between drags. */
  draggingKey: string | null;
  /** The edge the insertion line is drawn on for this row, if any. */
  lineEdge: (key: string) => PinnedDropEdge | undefined;
  /** Props for a row's wrapper: pick-up, hit-test marks and Alt+arrow reorder. */
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
  // Handlers are made once and read the latest options at event time.
  const optionsRef = useRef(options);
  useEffect(() => {
    optionsRef.current = options;
  });
  const draggingRef = useRef<string | null>(null);
  const targetRef = useRef<PinnedRowDrop | null>(null);
  const scroller = useRef<HTMLElement | null>(null);
  /** The gesture in flight, so a second press cannot overlap it. */
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
    document.body.classList.remove(DRAGGING_BODY_CLASS);
    setDraggingKey(null);
    showTarget(null);
  }, [showTarget]);

  useEffect(
    () => () => {
      press.current?.end();
    },
    [],
  );

  /** The landing spot under the pointer, or null where a drop would change nothing. */
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
        // The row can lose focus on re-render; restore it.
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
        // Touch scrolls the list, and the right button opens the row menu.
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
          showTarget(aim(at.x, at.y));
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
            document.body.classList.add(DRAGGING_BODY_CLASS);
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
          clear();
          if (drop) optionsRef.current.onDrop(key, drop);
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
          // Cancel the drag without closing the selector.
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
