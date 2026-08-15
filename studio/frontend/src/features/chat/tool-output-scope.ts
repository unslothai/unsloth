// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useAuiState } from "@assistant-ui/react";
import { createContext, useContext } from "react";

import { stripAnsi } from "../../lib/strip-ansi";
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

/**
 * Narrow a pane scope to one conversation: two threads in a pane can both be mid "call_0",
 * so without the thread in the key they share a store entry and swap outputs.
 */
export function toolThreadScope(paneScope: string, threadId?: string): string {
  return `${paneScope}\u0000${threadId ?? ""}`;
}

export const ToolPaneScopeContext = createContext<string>(toolPaneScope());

/**
 * Store-key scope for the conversation this component renders in, taken from the surrounding
 * runtime so reader and writer agree without a prop.
 *
 * `remoteId`, not `id`: the adapter gets `unstable_threadId`, which assistant-ui sources from
 * `remoteId`, and an uninitialized thread has `id` but no `remoteId`. Reading `id` split the
 * keys apart for the first turn of every New Chat, so live tool output never reached the card.
 */
export function useToolPaneScope(): string {
  const paneScope = useContext(ToolPaneScopeContext);
  const threadId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  return toolThreadScope(paneScope, threadId);
}

/**
 * Read a tool-output map for one call, tolerating a run that started before its thread had an id.
 *
 * The adapter captures its scope once at run start, so a first turn writes under the unresolved
 * scope for its whole life. The autosave can assign `remoteId` mid-run, which moves this
 * component's key but not the writer's, and the card went blank. Falling back to the pane-wide
 * scope keeps those entries reachable; only an unpersisted first turn can be filed there.
 */
/** The scope a run that started before its thread had an id writes under. */
export function useUnresolvedToolPaneScope(): string {
  return toolThreadScope(useContext(ToolPaneScopeContext), undefined);
}

export function useToolOutputFor(
  map: Record<string, string>,
  paneScope: string,
  toolCallId: string,
): string {
  // Unconditional: hooks cannot sit behind the early return below.
  const unresolvedScope = useUnresolvedToolPaneScope();
  // Only a thread mid-run can be the one that just gained its id. Local ids repeat
  // ("call_0"), so an unconditional fallback showed a live first turn's stdout in every
  // older conversation whose own entry had been cleared.
  const isRunning = useAuiState(({ thread }) => thread.isRunning);
  const own = map[toolOutputKey(paneScope, toolCallId)];
  if (own !== undefined) return own;
  if (!isRunning) return "";
  return map[toolOutputKey(unresolvedScope, toolCallId)] ?? "";
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


/** Normalize both sources before deciding whether the live stream is fuller. */
export function preferSanitizedFullToolOutput(
  full: string,
  result: string,
): string {
  return preferFullToolOutput(stripAnsi(full), stripAnsi(result));
}
