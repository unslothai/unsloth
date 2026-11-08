// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Types for Visual Computer Use mode (#15532). */

export type ComputerPermissionMode = "restricted" | "automatic";

export type ComputerAction = {
  id: string;
  sessionId: string;
  op:
    | "screenshot"
    | "click"
    | "move"
    | "drag"
    | "mouse_down"
    | "mouse_up"
    | "type"
    | "press_key"
    | "scroll"
    | "cursor_position"
    | "interval_screenshot";
  x?: number;
  y?: number;
  toX?: number;
  toY?: number;
  button?: "left" | "right" | "middle";
  clickCount?: number;
  modifiers?: string[];
  text?: string;
  pressEnter?: boolean;
  clearFirst?: boolean;
  key?: string;
  direction?: "up" | "down" | "left" | "right";
  amount?: number;
  display?: number;
  region?: { x: number; y: number; width: number; height: number };
  enabled?: boolean;
  intervalSeconds?: number;
  folderPath?: string;
  timeoutS?: number;
};

export type ComputerActionResult = {
  ok: boolean;
  error?: string;
  image?: string;
  width?: number;
  height?: number;
  message?: string;
  x?: number;
  y?: number;
  screenWidth?: number;
  screenHeight?: number;
  enabled?: boolean;
  folder?: string;
};
