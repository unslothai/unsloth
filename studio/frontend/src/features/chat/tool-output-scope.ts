// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { createContext, useContext } from "react";
import type { ModelType } from "./types";

/**
 * Pane scope prefix for the transient tool-output store keys.
 *
 * Local GGUF tool ids are only unique within one response ("call_0", "call_1",
 * ...), and panes stream concurrently (compare mode mounts two runtimes; the
 * main view stays CSS-hidden off-route), so a bare id would let one pane's
 * stdout bleed into another's same-id card. Derived from static props
 * (`modelType` + `pairId`) shared by writer (adapter) and reader (components)
 * via one `ChatRuntimeProvider`, so they can never disagree.
 */
export function toolPaneScope(modelType?: ModelType, pairId?: string): string {
  return `${modelType ?? "base"}\u0000${pairId ?? ""}`;
}

export const ToolPaneScopeContext = createContext<string>(toolPaneScope());

export function useToolPaneScope(): string {
  return useContext(ToolPaneScopeContext);
}

/** Store key for the live/full tool output maps: pane scope + tool call id. */
export function toolOutputKey(paneScope: string, toolCallId: string): string {
  return `${paneScope}\u0000${toolCallId}`;
}

// Footer the backend appends when it truncates a result to protect the context
// window (see backend tools._truncate). Marks where the result stops being a
// copy of the stream, so it distinguishes "just truncated" from "carries
// failure/exit status the stream never produced".
const TRUNCATION_FOOTER_MARKER = "\n\n... (truncated";

/**
 * Whether the live stdout holds more real output than the model-visible
 * `result` and should be preserved for the finished card. Shared by writer
 * (retain?) and reader (display?) so they agree.
 *
 * True when the result is truncated, OR the stream is longer. Truncation can't
 * fall back to length: a truncated result may be longer by byte count once its
 * footer / `Exit code N:` / `__IMAGES__` blob is appended, yet the stream still
 * holds more stdout. Also true when a short stream is absent from the result: a
 * timed-out/cancelled tool returns only a status line, so length alone would
 * drop the partial stdout the stream captured.
 */
export function shouldPreserveFullOutput(full: string, result: string): boolean {
  if (!full) {
    return false;
  }
  if (result.includes(TRUNCATION_FOOTER_MARKER)) {
    return true;
  }
  if (full.length > result.length) {
    return true;
  }
  // Stream no longer than the result, but a timed-out/cancelled tool's status
  // line never echoes the captured stdout: preserve the stream whenever its
  // content is absent from the result (trimmed to ignore trailing-newline drift).
  const core = full.trim();
  return core.length > 0 && !result.includes(core);
}

/**
 * Pick what a finished python/terminal card shows. Prefer the fuller live
 * stream over the truncated `result`, but the result can carry failure/exit
 * text that never reached stdout ("Exit code N: ...", timeouts), so show the
 * stream when the result is just a truncated prefix of it, else append the
 * result so its status survives (and the copy button copies both).
 */
export function preferFullToolOutput(full: string, result: string): string {
  if (!shouldPreserveFullOutput(full, result)) {
    return result;
  }
  const marker = result.indexOf(TRUNCATION_FOOTER_MARKER);
  const core = marker === -1 ? result : result.slice(0, marker);
  if (!core || full === result || full.startsWith(core)) {
    return full;
  }
  // Failed executions prefix the result (not the stream) with "Exit code N:\n",
  // so `full.startsWith(core)` above misses and a plain append would duplicate
  // the stdout. Re-attach just the exit prefix (and any missing-path hint) to
  // the fuller stream so the status survives without duplicating stdout.
  const exitMatch = core.match(/^(Exit code -?\d+:\n)([\s\S]*)$/);
  if (exitMatch && full.startsWith(exitMatch[2])) {
    const hint = result.match(/\nHint:[\s\S]*$/)?.[0] ?? "";
    return `${exitMatch[1]}${full}${hint}`;
  }
  return `${full.replace(/\s+$/, "")}\n\n${result}`;
}
