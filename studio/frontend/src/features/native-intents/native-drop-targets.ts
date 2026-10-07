// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { nativeDropPointToCss } from "./native-drop-position.ts";

export interface NativeDropTargetHandlers {
  onDrop: (paths: string[]) => void;
  onDragOver?: (over: boolean) => void;
}

// Tauri delivers OS drops window-wide and suppresses webview drop events, so targets are hit-tested here.
const targets = new Map<HTMLElement, NativeDropTargetHandlers>();

let scaleFactor =
  typeof window === "undefined" ? 1 : window.devicePixelRatio || 1;
let hovered: HTMLElement | null = null;
let listening = false;
let ready = !isTauri;

export function nativeDropTargetAt(position: {
  x: number;
  y: number;
}): HTMLElement | null {
  if (!ready || targets.size === 0 || typeof document === "undefined")
    return null;
  const { x, y } = nativeDropPointToCss(position, scaleFactor);
  let node: Element | null = document.elementFromPoint(x, y);
  while (node !== null) {
    if (node instanceof HTMLElement && targets.has(node)) return node;
    node = node.parentElement;
  }
  return null;
}

function setHovered(next: HTMLElement | null): void {
  if (hovered === next) return;
  if (hovered) targets.get(hovered)?.onDragOver?.(false);
  hovered = next;
  if (next) targets.get(next)?.onDragOver?.(true);
}

function listen(): void {
  if (listening || !isTauri) return;
  listening = true;
  void import("@tauri-apps/api/window")
    .then(async ({ getCurrentWindow }) => {
      const currentWindow = getCurrentWindow();
      await currentWindow.onDragDropEvent(({ payload }) => {
        if (payload.type === "leave") {
          setHovered(null);
          return;
        }
        const target = nativeDropTargetAt(payload.position);
        if (payload.type !== "drop") {
          setHovered(target);
          return;
        }
        setHovered(null);
        if (target) targets.get(target)?.onDrop(payload.paths);
      });
      // Only now can a target be claimed, else window-wide handlers defer to a dead listener.
      ready = true;

      // Failing here must not reset `listening`, or a second listener doubles every drop.
      let scaleReported = false;
      await currentWindow
        .onScaleChanged(({ payload }) => {
          scaleReported = true;
          scaleFactor = payload.scaleFactor;
        })
        .catch(() => undefined);
      const initialScale = await currentWindow.scaleFactor().catch(() => null);
      if (!scaleReported && initialScale !== null) scaleFactor = initialScale;
    })
    .catch(() => {
      listening = false;
    });
}

// Install at app start, not under the user's first drop.
listen();

export function registerNativeDropTarget(
  element: HTMLElement,
  handlers: NativeDropTargetHandlers,
): () => void {
  targets.set(element, handlers);
  listen();
  return () => {
    if (hovered === element) hovered = null;
    targets.delete(element);
  };
}
