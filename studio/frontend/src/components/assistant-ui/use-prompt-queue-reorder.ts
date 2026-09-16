// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type PointerEvent,
} from "react";
import { useReducedMotionConfig } from "motion/react";
import type { PromptQueueUIItem } from "@/features/chat";

const SETTLE_MS = 180;
const EASING = "cubic-bezier(0.2, 0.8, 0.2, 1)";
const ROW_SELECTOR = "[data-queue-item-id]";

type Row = {
  id: string;
  element: HTMLElement;
  top: number;
  height: number;
  movable: boolean;
};
type Drag = {
  id: string;
  pointerId: number;
  handle: HTMLButtonElement;
  originY: number;
  scrollTop: number;
  scale: number;
  x: number;
  y: number;
  from: number;
  target: string | null;
  rows: Row[];
  signature: string;
  moved: boolean;
  frame: number;
  lastTime: number;
};

function getQueueScale(list: HTMLElement | null) {
  return list ? list.getBoundingClientRect().width / list.offsetWidth || 1 : 1;
}

export function usePromptQueueReorder(
  items: PromptQueueUIItem[],
  disabled: boolean,
  onMove: (id: string, targetId: string) => boolean,
  announce: (message: string) => void,
) {
  const listRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<Drag | null>(null);
  const animations = useRef<Animation[]>([]);
  const pendingPositions = useRef<Map<string, number> | null>(null);
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const reducedMotion = useReducedMotionConfig();
  const signature = JSON.stringify([
    disabled,
    items.map((item) => [item.id, item.canEdit, item.canRemove]),
  ]);

  const stopAnimations = useCallback(() => {
    for (const animation of animations.current) animation.cancel();
    animations.current = [];
  }, []);

  const capturePositions = useCallback(() => {
    const list = listRef.current;
    const scale = getQueueScale(list);
    return new Map(
      Array.from(list?.querySelectorAll<HTMLElement>(ROW_SELECTOR) ?? []).map(
        (row) => [
          row.dataset.queueItemId!,
          row.getBoundingClientRect().top / scale + (list?.scrollTop ?? 0),
        ],
      ),
    );
  }, []);

  const clearDrag = useCallback((drag: Drag) => {
    cancelAnimationFrame(drag.frame);
    dragRef.current = null;
    for (const row of drag.rows) {
      row.element.style.removeProperty("transform");
      row.element.style.removeProperty("transition");
    }
    if (drag.handle.hasPointerCapture(drag.pointerId)) {
      drag.handle.releasePointerCapture(drag.pointerId);
    }
  }, []);

  const cancelDrag = useCallback(() => {
    const drag = dragRef.current;
    if (!drag) return false;
    if (drag.moved) pendingPositions.current = capturePositions();
    clearDrag(drag);
    setDraggingId(null);
    return true;
  }, [capturePositions, clearDrag]);

  function move(id: string, targetId: string | undefined) {
    if (!targetId || id === targetId) return;
    pendingPositions.current ??= capturePositions();
    if (onMove(id, targetId)) {
      const position = items.findIndex((item) => item.id === targetId) + 1;
      announce(`Prompt moved to position ${position} of ${items.length}.`);
    } else {
      if (!draggingId) pendingPositions.current = null;
      announce(
        "The queue changed before this prompt could be moved. Try again.",
      );
    }
  }

  useLayoutEffect(() => {
    const drag = dragRef.current;
    if (drag && (drag.signature !== signature || disabled)) {
      clearDrag(drag);
      setDraggingId(null);
      announce(
        "The queue changed. Drag again to reorder the remaining prompts.",
      );
    }
    const previous = pendingPositions.current;
    pendingPositions.current = null;
    if (!previous) return;
    stopAnimations();
    if (reducedMotion) return;
    const list = listRef.current;
    const scale = getQueueScale(list);
    for (const row of list?.querySelectorAll<HTMLElement>(ROW_SELECTOR) ?? []) {
      const before = previous.get(row.dataset.queueItemId!);
      if (before === undefined) continue;
      const offset =
        before - row.getBoundingClientRect().top / scale - (list?.scrollTop ?? 0);
      if (Math.abs(offset) < 0.5) continue;
      animations.current.push(
        row.animate(
          [
            { transform: `translateY(${offset}px)` },
            { transform: "translateY(0)" },
          ],
          { duration: SETTLE_MS, easing: EASING },
        ),
      );
    }
  }, [
    signature,
    disabled,
    reducedMotion,
    announce,
    draggingId,
    clearDrag,
    stopAnimations,
  ]);

  useEffect(() => {
    const cancel = () => cancelDrag();
    window.addEventListener("blur", cancel);
    window.addEventListener("resize", cancel);
    return () => {
      window.removeEventListener("blur", cancel);
      window.removeEventListener("resize", cancel);
      const drag = dragRef.current;
      if (drag) clearDrag(drag);
      stopAnimations();
    };
  }, [cancelDrag, clearDrag, stopAnimations]);

  function updateDrag(time: number) {
    const drag = dragRef.current;
    const list = listRef.current;
    if (!drag || !drag.moved || !list) return;
    const bounds = list.getBoundingClientRect();
    const insideX = drag.x >= bounds.left && drag.x <= bounds.right;
    const elapsed = Math.min(time - drag.lastTime, 32);
    drag.lastTime = time;
    // Keep scrolling when the pointer rests at an edge.
    if (insideX) {
      const edge = Math.min(36, bounds.height / 4);
      const speed =
        drag.y < bounds.top + edge
          ? -Math.min(1, (bounds.top + edge - drag.y) / edge)
          : drag.y > bounds.bottom - edge
            ? Math.min(1, (drag.y - bounds.bottom + edge) / edge)
            : 0;
      list.scrollTop += (speed * elapsed * 0.45) / drag.scale;
    }
    const source = drag.rows[drag.from];
    // Pointer coordinates include interface scaling; transforms do not.
    const delta =
      (drag.y - drag.originY) / drag.scale + list.scrollTop - drag.scrollTop;
    const first = drag.rows[0];
    const last = drag.rows[drag.rows.length - 1];
    const offset = Math.max(
      first.top - source.top,
      Math.min(delta, last.top + last.height - source.top - source.height),
    );
    const center = source.top + source.height / 2 + offset;
    const inside = insideX && drag.y >= bounds.top && drag.y <= bounds.bottom;
    let to = drag.from;
    let distance = Infinity;
    if (inside) {
      drag.rows.forEach((row, index) => {
        const nextDistance = Math.abs(center - row.top - row.height / 2);
        if (row.movable && nextDistance < distance) {
          to = index;
          distance = nextDistance;
        }
      });
    }
    drag.target = inside ? drag.rows[to].id : null;
    drag.rows.forEach((row, index) => {
      let shift = 0;
      if (index === drag.from) shift = offset;
      else if (index > drag.from && index <= to) shift = -source.height;
      else if (index < drag.from && index >= to) shift = source.height;
      row.element.style.transform = `translate3d(0, ${shift}px, 0)`;
    });
    drag.frame = requestAnimationFrame(updateDrag);
  }

  function onPointerDown(event: PointerEvent<HTMLButtonElement>, id: string) {
    if (disabled || event.button !== 0 || !event.isPrimary || dragRef.current)
      return;
    const list = listRef.current;
    if (!list) return;
    stopAnimations();
    const bounds = list.getBoundingClientRect();
    const scale = getQueueScale(list);
    const rows = Array.from(
      list.querySelectorAll<HTMLElement>(ROW_SELECTOR),
    ).map((element) => {
      const rect = element.getBoundingClientRect();
      const item = items.find(
        (item) => item.id === element.dataset.queueItemId,
      )!;
      return {
        id: item.id,
        element,
        top: (rect.top - bounds.top) / scale + list.scrollTop,
        height: rect.height / scale,
        movable: item.canEdit && item.canRemove,
      };
    });
    const from = rows.findIndex((row) => row.id === id);
    if (
      from < 0 ||
      !rows[from].movable ||
      rows.filter((row) => row.movable).length < 2
    )
      return;
    event.preventDefault();
    event.currentTarget.focus({ preventScroll: true });
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      id,
      pointerId: event.pointerId,
      handle: event.currentTarget,
      originY: event.clientY,
      scrollTop: list.scrollTop,
      scale,
      x: event.clientX,
      y: event.clientY,
      from,
      target: null,
      rows,
      signature,
      moved: false,
      frame: 0,
      lastTime: performance.now(),
    };
  }

  function onPointerMove(event: PointerEvent<HTMLButtonElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    drag.x = event.clientX;
    drag.y = event.clientY;
    if (drag.moved || Math.abs(drag.y - drag.originY) < 5) return;
    drag.moved = true;
    for (const row of drag.rows) {
      row.element.style.transition =
        row.id === drag.id || reducedMotion
          ? "none"
          : `transform ${SETTLE_MS}ms ${EASING}`;
    }
    setDraggingId(drag.id);
    drag.frame = requestAnimationFrame(updateDrag);
  }

  function onPointerUp(event: PointerEvent<HTMLButtonElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (drag.moved) {
      drag.x = event.clientX;
      drag.y = event.clientY;
      cancelAnimationFrame(drag.frame);
      updateDrag(performance.now());
      pendingPositions.current = capturePositions();
    }
    clearDrag(drag);
    if (drag.moved && drag.target) move(drag.id, drag.target);
    setDraggingId(null);
  }

  return {
    listRef,
    draggingId,
    move,
    cancelDrag,
    onPointerDown,
    onPointerMove,
    onPointerUp,
  };
}
