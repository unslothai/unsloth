// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useComputerAgentStore } from "./agent-store";

/** The visual computer use tools (core/inference/computer_agent.py, #15532). */
export const COMPUTER_AGENT_TOOLS = [
  "computer_screenshot",
  "computer_click",
  "computer_move_mouse",
  "computer_drag",
  "computer_mouse_down",
  "computer_mouse_up",
  "computer_type",
  "computer_press_key",
  "computer_scroll",
  "computer_cursor_position",
  "computer_interval_screenshot",
] as const;

export const COMPUTER_SCREENSHOT_TOOL = "computer_screenshot";

/** Tools to offer this turn when Visual Computer Use mode is enabled. */
export function computerAgentToolsForTurn(): string[] {
  const { enabled } = useComputerAgentStore.getState();
  if (!enabled) return [];
  return [...COMPUTER_AGENT_TOOLS];
}
