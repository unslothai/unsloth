// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
/**
 * Lightweight, pure preference constants. Do not import the virtual renderer,
 * height ledger or DOM-cost index from the eager chat/settings entry points.
 */
export const AGENT_DEFAULT_ROWS = 5;
export const AGENT_INITIAL_ROWS = 32;
export const AGENT_MAX_ROWS = 96;
export const AGENT_WINDOW_SIZES = [2, 5, 8, 16, 32] as const;
export type AgentWindowSize = (typeof AGENT_WINDOW_SIZES)[number];
export function isAgentWindowSize(value: unknown): value is AgentWindowSize {
  return typeof value === "number" && (AGENT_WINDOW_SIZES as readonly number[]).includes(value);
}
export const AGENT_BUFFER_MODES = [
  "dynamic", "dynamic-low", "dynamic-high", "fixed-low", "fixed-high",
] as const;
export type AgentBufferMode = (typeof AGENT_BUFFER_MODES)[number];
export type AgentDomLevel = "off" | "low" | "high" | AgentBufferMode;
export function isAgentBufferMode(value: unknown): value is AgentBufferMode {
  return typeof value === "string" && (AGENT_BUFFER_MODES as readonly string[]).includes(value);
}
export function agentBufferRowLimit(mode: AgentBufferMode): 5 | 16 | 32 {
  return mode === "dynamic" ? 16 : mode.endsWith("-low") ? 5 : 32;
}
export function agentBufferDomLevel(mode: AgentBufferMode): "low" | "medium" | "high" | undefined {
  return mode === "dynamic" ? "medium"
    : mode === "dynamic-low" ? "low" : mode === "dynamic-high" ? "high" : undefined;
}
