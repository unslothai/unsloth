// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { nativeDropPointToCss } from "./native-drop-position.ts";

export interface NativeDropTargetHandlers {
  onDrop: (paths: string[]) => void;
  onDragOver?: (over: boolean) => void;
}

// Tauri delivers OS drops window-wide with a physical position and suppresses
// the webview's own drop events, so which element was dropped on has to be
// resolved here instead of by DOM event routing.
const targets = new Map<HTMLElement, NativeDropTargetHandlers>();

let scaleFactor =
  typeof window === "undefined" ? 1 : window.devicePixelRatio || 1;
let hovered: HTMLElement | null = null;
let listening = false;

/** The innermost registered target under `position`, or null. */
export function nativeDropTargetAt(position: {
  x: number;
  y: number;
}): HTMLElement | null {
  if (targets.size === 0 || typeof document === "undefined") return null;
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
      let scaleReported = false;
      await currentWindow.onScaleChanged(({ payload }) => {
        scaleReported = true;
        scaleFactor = payload.scaleFactor;
      });
      const initialScale = await currentWindow.scaleFactor();
      if (!scaleReported) scaleFactor = initialScale;

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
    })
    .catch(() => {
      listening = false;
    });
}

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
