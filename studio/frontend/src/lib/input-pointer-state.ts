// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function createInputPointerState(target: Window, document: Document): {
  held: () => boolean;
  dispose: () => void;
} {
  const pointers = new Set<number>();
  const cleanups: (() => void)[] = [];
  let mouseHeld = false;
  let touchHeld = false;
  let dragging = false;

  const listen = (node: EventTarget, type: string, listener: EventListener): void => {
    node.addEventListener(type, listener, { capture: true, passive: true });
    cleanups.push(() => node.removeEventListener(type, listener, { capture: true }));
  };
  const release = (): void => {
    pointers.clear();
    mouseHeld = false;
    touchHeld = false;
    dragging = false;
  };

  if (typeof (target as Window & { PointerEvent?: unknown }).PointerEvent === "function") {
    listen(target, "pointerdown", ((event: PointerEvent) => {
      if (event.button === 0 || event.pointerType === "touch") pointers.add(event.pointerId);
    }) as EventListener);
    listen(target, "pointermove", ((event: PointerEvent) => {
      if (event.pointerType === "touch") return;
      if ((event.buttons & 1) !== 0) pointers.add(event.pointerId);
      else pointers.delete(event.pointerId);
    }) as EventListener);
    const pointerReleased = ((event: PointerEvent) => {
      pointers.delete(event.pointerId);
    }) as EventListener;
    listen(target, "pointerup", pointerReleased);
    listen(target, "pointercancel", pointerReleased);
  } else {
    listen(target, "mousedown", ((event: MouseEvent) => {
      mouseHeld = event.button === 0 || (event.buttons & 1) !== 0;
    }) as EventListener);
    const mouseChanged = ((event: MouseEvent) => {
      mouseHeld = (event.buttons & 1) !== 0;
    }) as EventListener;
    listen(target, "mousemove", mouseChanged);
    listen(target, "mouseup", mouseChanged);
    const touchesChanged = ((event: TouchEvent) => {
      touchHeld = event.touches.length > 0;
    }) as EventListener;
    for (const type of ["touchstart", "touchmove", "touchend", "touchcancel"]) {
      listen(target, type, touchesChanged);
    }
  }

  // A cancelled dragstart is never followed by dragend or drop.
  listen(target, "dragstart", (event) => {
    dragging = true;
    setTimeout(() => {
      if (event.defaultPrevented) dragging = false;
    }, 0);
  });
  listen(target, "dragend", release);
  listen(target, "drop", release);
  listen(target, "blur", (event) => {
    if (event.target === target) release();
  });
  listen(target, "pagehide", release);
  listen(document, "visibilitychange", () => {
    if (document.hidden) release();
  });

  return {
    held: () => pointers.size > 0 || mouseHeld || touchHeld || dragging,
    dispose: () => {
      for (const cleanup of cleanups.splice(0)) cleanup();
      release();
    },
  };
}

const inputPointerState = typeof window === "undefined" || typeof document === "undefined"
  ? null
  : createInputPointerState(window, document);

export const inputPointerHeld = (): boolean => inputPointerState?.held() ?? false;
