// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { browserPanelAvailable } from "../panel-availability";
import { useBrowserAgentStore } from "./agent-store";

/** The browser tools (core/inference/browser_agent.py). This tab runs them, so a turn that offers
 *  them must keep its stream to this tab. */
export const BROWSER_AGENT_TOOLS = [
  "browser_navigate",
  "browser_snapshot",
  "browser_find",
  "browser_click",
  "browser_move_mouse",
  "browser_drag",
  "browser_type",
  "browser_select",
  "browser_scroll",
  "browser_press_key",
] as const;

/** Captured from the panel without a prompt only in the desktop app; the server keeps it for vision models. */
export const BROWSER_SCREENSHOT_TOOL = "browser_screenshot";

export const BROWSER_HISTORY_TOOL = "browser_history";

export const BROWSER_SCRIPT_TOOL = "browser_evaluate";

/** The browser tools to offer this turn: none unless the Browser tool is on and the panel can show. */
export function browserAgentToolsForTurn(): string[] {
  const { enabled, historyAccess, scriptAccess } = useBrowserAgentStore.getState();
  if (!enabled || !browserPanelAvailable()) return [];
  return [
    ...BROWSER_AGENT_TOOLS,
    ...(isTauri ? [BROWSER_SCREENSHOT_TOOL] : []),
    ...(historyAccess === "never" ? [] : [BROWSER_HISTORY_TOOL]),
    ...(scriptAccess === "never" ? [] : [BROWSER_SCRIPT_TOOL]),
  ];
}
