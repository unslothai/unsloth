// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useAuiState } from "@assistant-ui/react";
import { createContext, useContext } from "react";

import { useChatRuntimeStore } from "./stores/chat-runtime-store";

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

/** Selects only this card's keys, not the whole map, so another tool's write leaves it alone. */
export function useToolOutput(
  field: "toolFullOutput" | "toolLiveOutput",
  toolCallId: string,
): string {
  const paneScope = useToolPaneScope();
  const unresolvedScope = useUnresolvedToolPaneScope();
  const own = useChatRuntimeStore(
    (s) => s[field][toolOutputKey(paneScope, toolCallId)],
  );
  const unresolved = useChatRuntimeStore(
    (s) => s[field][toolOutputKey(unresolvedScope, toolCallId)],
  );
  // Only a running thread may fall back (local ids repeat). Select the string, not isRunning,
  // so a run starting or ending does not re-render every tool card in the history.
  const fallback = useAuiState(({ thread }) =>
    own === undefined && thread.isRunning ? (unresolved ?? "") : "",
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
