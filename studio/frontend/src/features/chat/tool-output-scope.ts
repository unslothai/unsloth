// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useAuiState } from "@assistant-ui/react";
import { createContext, useContext } from "react";

import type { ModelType } from "./types";

/** Local GGUF tool ids repeat per response and panes stream concurrently, so keys need a pane scope. */
export function toolPaneScope(modelType?: ModelType, pairId?: string): string {
  return `${modelType ?? "base"}\u0000${pairId ?? ""}`;
}

export function toolThreadScope(paneScope: string, threadId?: string): string {
  return `${paneScope}\u0000${threadId ?? ""}`;
}

export const ToolPaneScopeContext = createContext<string>(toolPaneScope());

/** Uses `remoteId`, not `id`: the adapter's unstable_threadId comes from remoteId. */
export function useToolPaneScope(): string {
  const paneScope = useContext(ToolPaneScopeContext);
  const threadId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  return toolThreadScope(paneScope, threadId);
}

export function useUnresolvedToolPaneScope(): string {
  return toolThreadScope(useContext(ToolPaneScopeContext), undefined);
}

export function useToolOutputFor(
  map: Record<string, string>,
  paneScope: string,
  toolCallId: string,
): string {
  const unresolvedScope = useUnresolvedToolPaneScope();
  const own = map[toolOutputKey(paneScope, toolCallId)];
  // Only a running thread may fall back; local ids repeat, so others would show its stdout.
  // Select the fallback string, not isRunning: every tool card in the history would otherwise
  // re-render when a run starts or ends, though almost none of them has a fallback entry.
  const fallback = useAuiState(({ thread }) =>
    own === undefined && thread.isRunning
      ? (map[toolOutputKey(unresolvedScope, toolCallId)] ?? "")
      : "",
  );
  return own ?? fallback;
}

export function toolOutputKey(paneScope: string, toolCallId: string): string {
  return `${paneScope}\u0000${toolCallId}`;
}

// Kept React-free so node tests can import it without the runner hanging.
export {
  preferFullToolOutput,
  preferSanitizedFullToolOutput,
  shouldPreserveFullOutput,
  toolResultText,
} from "./tool-output-result";
