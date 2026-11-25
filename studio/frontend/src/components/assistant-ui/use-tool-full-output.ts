// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useAuiState } from "@assistant-ui/react";
import {
  toolOutputKey,
  useChatRuntimeStore,
  useToolPaneScope,
  useUnresolvedToolPaneScope,
} from "@/features/chat";

/** useToolOutputFor over toolFullOutput, selecting this card's two keys as strings, not the whole
 *  map, so a long-output tool end re-renders only its own card instead of every card in history. */
export function useToolFullOutput(toolCallId: string): string {
  const paneScope = useToolPaneScope();
  const unresolvedScope = useUnresolvedToolPaneScope();
  const own = useChatRuntimeStore(
    (s) => s.toolFullOutput[toolOutputKey(paneScope, toolCallId)],
  );
  const unresolved = useChatRuntimeStore(
    (s) => s.toolFullOutput[toolOutputKey(unresolvedScope, toolCallId)],
  );
  // Only a running thread may fall back (local ids repeat); select the string, not isRunning.
  const fallback = useAuiState(({ thread }) =>
    own === undefined && thread.isRunning ? (unresolved ?? "") : "",
  );
  return own ?? fallback;
}
