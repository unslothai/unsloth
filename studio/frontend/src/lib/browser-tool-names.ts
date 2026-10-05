// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** mirrors BROWSER_TOOLS in studio/backend/core/inference/browser_tools.py, in offer order. */
export const BROWSER_TOOL_NAMES = [
  "browser_navigate",
  "browser_snapshot",
  "browser_click",
  "browser_type",
  "browser_select",
  "browser_press_key",
  "browser_scroll",
  "browser_read",
  "browser_find",
  "browser_handoff",
  "browser_screenshot",
] as const;

export type BrowserToolName = (typeof BROWSER_TOOL_NAMES)[number];

const NAMES: ReadonlySet<string> = new Set(BROWSER_TOOL_NAMES);

export function isBrowserToolName(name: unknown): name is BrowserToolName {
  return typeof name === "string" && NAMES.has(name);
}

export function browserToolsForTurn(readsImages: boolean): string[] {
  return BROWSER_TOOL_NAMES.filter(
    (name) => readsImages || name !== "browser_screenshot",
  );
}
