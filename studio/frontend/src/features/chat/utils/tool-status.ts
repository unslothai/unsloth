// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Mirrors NUDGE_TOOL_CALLS_STATUS in
 * studio/backend/core/inference/tool_call_parser.py. The GGUF and safetensors
 * tool loops send it while a plan-without-action re-prompt regenerates; that
 * turn's output is hidden, so the badge is the only sign of life.
 */
export const NUDGE_TOOL_CALLS_STATUS = "Nudging tool calls";

export type ToolStatusKind = "nudge" | "terminal" | "web";

/**
 * Which glyph the composer status badge shows. The backend sends prose, not a
 * kind, so this is the one place that reads it: an exact match for the nudge,
 * then the long-standing "Running ..." prefix for the local sandbox tools
 * (status_for_tool in core/inference/tool_loop_controller.py), and the globe
 * for everything else ("Searching: ...", "Reading: ...", "Calling: ...").
 */
export function toolStatusKind(status: string): ToolStatusKind {
  if (status === NUDGE_TOOL_CALLS_STATUS) {
    return "nudge";
  }
  return status.startsWith("Running") ? "terminal" : "web";
}
