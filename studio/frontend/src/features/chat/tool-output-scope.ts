// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useAuiState } from "@assistant-ui/react";
import { createContext, useContext } from "react";

import type { ModelType } from "./types";

/** Pane scope prefix for the transient tool-output store keys. Local GGUF tool ids are only
 *  unique within one response ("call_0", "call_1", ...), and panes stream concurrently, so a
 *  bare id would let one pane's stdout bleed into another's same-id card. Derived from static
 *  props shared by writer and reader via one `ChatRuntimeProvider`, so they cannot disagree. */
export function toolPaneScope(modelType?: ModelType, pairId?: string): string {
  return `${modelType ?? "base"}\u0000${pairId ?? ""}`;
}

/** Narrow a pane scope to one conversation: two threads in a pane can both be mid "call_0", so
 *  without the thread in the key they share a store entry and swap outputs. */
export function toolThreadScope(paneScope: string, threadId?: string): string {
  return `${paneScope}\u0000${threadId ?? ""}`;
}

export const ToolPaneScopeContext = createContext<string>(toolPaneScope());

/** Store-key scope for the conversation this component renders in, taken from the surrounding
 *  runtime so reader and writer agree without a prop. `remoteId`, not `id`: the adapter gets
 *  `unstable_threadId`, which assistant-ui sources from `remoteId`, and an uninitialized
 *  thread has `id` but no `remoteId`. Reading `id` split the keys apart on every New Chat. */
export function useToolPaneScope(): string {
  const paneScope = useContext(ToolPaneScopeContext);
  const threadId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  return toolThreadScope(paneScope, threadId);
}

/** Read a tool-output map for one call, tolerating a run that started before its thread had an
 *  id. The adapter captures its scope once at run start, so a first turn writes under the
 *  unresolved scope for its whole life, and an autosave assigning `remoteId` mid-run moves
 *  this component's key but not the writer's. Falling back to the pane-wide scope keeps those
 *  entries reachable; only an unpersisted first turn can be filed there. */
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
  // Only a thread mid-run can be the one that just gained its id. Local ids repeat ("call_0"), so
  // an unconditional fallback showed a live first turn's stdout in every older conversation.
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

// The output-preference rules moved to a dependency-free module so the recovery replay can fold
// persisted tool frames into parts without importing this module's React-bound scope helpers.
// Re-exported here unchanged: writer and reader keep reading one source through this path too.
export {
  preferFullToolOutput,
  preferSanitizedFullToolOutput,
  shouldPreserveFullOutput,
} from "./utils/tool-output-preference";
