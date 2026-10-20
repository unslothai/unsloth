// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import {
  buildLocalTokenCountExtras,
  buildLocalTokenCountHistory,
  buildLocalTokenCountReasoning,
} from "./chat-adapter";
import { type ChatCountTokensResponse, countChatInputTokens } from "./chat-api";

/** /context: the header recount's payload for this branch, split into its parts and with the
 *  rendered prompt. Only a count: nothing here reaches the conversation or the next completion.
 *  Null when the total belongs to another model, because the endpoint counts with whatever is
 *  resident. */
export async function countContext(
  branch: readonly ThreadMessage[],
  threadId: string | undefined,
): Promise<ChatCountTokensResponse | null> {
  const start = useChatRuntimeStore.getState();
  const checkpoint = start.params.checkpoint;
  const history = await buildLocalTokenCountHistory(branch, threadId);
  const extras = await buildLocalTokenCountExtras(threadId, history.messages);
  const counted = await countChatInputTokens({
    model: checkpoint,
    ...history,
    ...buildLocalTokenCountReasoning(),
    ...extras,
    breakdown: true,
    ...(start.params.maxTokens ? { max_tokens: start.params.maxTokens } : {}),
  });
  if (counted.model != null && counted.model !== checkpoint) return null;
  // The header ring shows the same number, so the two never disagree on screen. The recount's
  // own guards: not after a switch, a load, or a run that wrote fresher usage meanwhile.
  const now = useChatRuntimeStore.getState();
  if (
    now.params.checkpoint === checkpoint &&
    now.activeThreadId === (threadId ?? null) &&
    now.contextUsage === start.contextUsage &&
    !Object.values(now.runningByThreadId).some(Boolean)
  ) {
    now.setContextUsage({
      promptTokens: counted.input_tokens,
      completionTokens: 0,
      totalTokens: counted.input_tokens,
      cachedTokens: 0,
      cacheWriteTokens: 0,
    });
  }
  return counted;
}
