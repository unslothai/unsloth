// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { branchSignature } from "../utils/refresh-context-usage";
import type { ThreadMessage } from "@assistant-ui/react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import {
  buildLocalTokenCountExtras,
  buildLocalTokenCountHistory,
  buildLocalTokenCountReasoning,
  findLatestUserAudioBase64,
  findLatestUserVideoBase64,
  messagesContainImage,
} from "./chat-adapter";
import { type ChatCountTokensResponse, countChatInputTokens } from "./chat-api";

/** /context: the header recount's payload for this branch, split into its parts and with the
 *  rendered prompt. Only a count: nothing here reaches the conversation or the next completion.
 *  Null when the total belongs to another model, because the endpoint counts with whatever is
 *  resident. */
export async function countContext(
  branch: readonly ThreadMessage[],
  threadId: string | undefined,
  readBranch: () => readonly ThreadMessage[],
): Promise<ChatCountTokensResponse | null> {
  const start = useChatRuntimeStore.getState();
  const refusal = uncountable(branch, start.loadedIsGguf);
  if (refusal) throw new Error(refusal);
  const checkpoint = start.params.checkpoint;
  const signature = branchSignature(branch);
  const history = await buildLocalTokenCountHistory(branch, threadId);
  const extras = await buildLocalTokenCountExtras(threadId, history.messages);
  const counted = await countChatInputTokens({
    model: checkpoint,
    ...history,
    ...buildLocalTokenCountReasoning(),
    ...extras,
    breakdown: true,
  });
  if (counted.model != null && counted.model !== checkpoint) return null;
  const now = useChatRuntimeStore.getState();
  if (
    now.params.checkpoint !== checkpoint ||
    branchSignature(readBranch()) !== signature
  ) {
    return null;
  }
  if (
    now.activeThreadId === (threadId ?? null) &&
    now.contextUsage === start.contextUsage &&
    !Object.values(now.runningByThreadId).some(Boolean)
  ) {
    const usage = now.contextUsage;
    if (
      usage &&
      !usage.estimated &&
      (!usage.modelId || usage.modelId === checkpoint)
    ) {
      now.setContextUsage({ ...usage, recountedTokens: counted.input_tokens });
    } else {
      now.setContextUsage({
        promptTokens: counted.input_tokens,
        completionTokens: 0,
        totalTokens: counted.input_tokens,
        cachedTokens: 0,
      });
    }
  }
  return counted;
}

/** Chats whose count would be wrong, which the header recount (refreshContextUsage) declines too:
 *  it would leave out the media, or price a non-GGUF chat as if it were never compacted. */
function uncountable(
  branch: readonly ThreadMessage[],
  loadedIsGguf: boolean | null,
): string | null {
  if (messagesContainImage(branch)) {
    return "Cannot count tokens for messages containing images.";
  }
  // A clip still in the composer is not part of the chat being counted.
  if (findLatestUserAudioBase64(branch, false)) {
    return "Cannot count tokens for messages containing audio.";
  }
  if (findLatestUserVideoBase64(branch)) {
    return "Cannot count tokens for messages containing video.";
  }
  if (
    !loadedIsGguf &&
    branch.some((message) => message.metadata?.custom?.contextTruncation)
  ) {
    return "Cannot count tokens for a compacted chat on this model.";
  }
  return null;
}
