// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Ctrl/Cmd+Z undoes; Ctrl/Cmd+Shift+Z and Ctrl/Cmd+Y redo. Matched on `code` too, so a
 *  non-Latin layout or Shift's uppercase key still lands. */
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
