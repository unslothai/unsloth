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
      // Before the scale setup: until this resolves a registered target is
      // claimed with nothing behind it, and the chat-wide handler has already
      // stepped aside, so a drop in that window is lost outright. The
      // devicePixelRatio seed keeps the hit test usable until scale reports.
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

      // Scale is a refinement, not a prerequisite, and the drop listener is
      // already installed by now. Failing here must not reset `listening`: the
      // next registration would retry and stack a second drop listener, so
      // every later drop would be delivered twice. The seed above holds.
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

// Start the install now rather than on first registration, so the async window
// above is spent during app start instead of under the user's first drop.
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
