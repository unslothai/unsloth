// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Matched on `code` too, so non-Latin layouts and Shift's uppercase key still land. */
export function historyAction(event: {
  key: string;
  code?: string;
  metaKey: boolean;
  ctrlKey: boolean;
  shiftKey: boolean;
  altKey?: boolean;
}): "undo" | "redo" | null {
  if (!(event.metaKey || event.ctrlKey) || event.altKey) return null;
  const key = event.key.toLowerCase();
  const z = key === "z" || event.code === "KeyZ";
  const y = key === "y" || event.code === "KeyY";
  if (z) return event.shiftKey ? "redo" : "undo";
  if (y && !event.shiftKey) return "redo";
  return null;
}

export function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  return (
    el.isContentEditable ||
    el.tagName === "INPUT" ||
    el.tagName === "TEXTAREA" ||
    el.tagName === "SELECT"
  );
}

/** R (or Shift+R) with the reroute tool armed turns the reroute; an R typed in a field stays in the field. */
export function turnsArmedReroute(
  event: {
    key: string;
    target: EventTarget | null;
    metaKey: boolean;
    ctrlKey: boolean;
    altKey: boolean;
  },
  rerouteArmed: boolean,
): boolean {
  return (
    !isTyping(event.target) &&
    rerouteArmed &&
    (event.key === "r" || event.key === "R") &&
    !(event.metaKey || event.ctrlKey || event.altKey)
  );
}
