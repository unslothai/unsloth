// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  OpenAIChatCompletionsRequest,
  OpenAIChatMessage,
} from "../types/api";

export interface ChatKvPrefillAvailability {
  isExternalModel: boolean;
  residentCheckpoint: string | null | undefined;
  ggufContextLength: number | null;
  loadedIsDiffusion: boolean;
  loadedIsAudio: boolean;
}

/** Keep the visible control and the post-response dispatch on the same runtime gate. */
export function isChatKvPrefillAvailable({
  isExternalModel,
  residentCheckpoint,
  ggufContextLength,
  loadedIsDiffusion,
  loadedIsAudio,
}: ChatKvPrefillAvailability): boolean {
  return (
    !isExternalModel &&
    residentCheckpoint != null &&
    ggufContextLength != null &&
    !loadedIsDiffusion &&
    !loadedIsAudio
  );
}

export function buildChatKvPrefillPayload(
  request: OpenAIChatCompletionsRequest,
  finalizedAssistantMessages: OpenAIChatMessage[],
): OpenAIChatCompletionsRequest | null {
  const hasMeaningfulReplay = finalizedAssistantMessages.some((message) => {
    if (message.role === "tool" || (message.tool_calls?.length ?? 0) > 0) {
      return true;
    }
    if (message.reasoning_content?.trim()) {
      return true;
    }
    if (typeof message.content === "string") {
      return message.content.trim().length > 0;
    }
    return Array.isArray(message.content) && message.content.length > 0;
  });
  if (!hasMeaningfulReplay) {
    return null;
  }

  const messages = [...request.messages];
  if (request.continue_final_message && messages.at(-1)?.role === "assistant") {
    messages.pop();
  }
  messages.push(...finalizedAssistantMessages);

  return {
    ...request,
    messages,
    stream: false,
    continue_final_message: false,
  };
}

type ChatKvPrefillSender = (
  payload: OpenAIChatCompletionsRequest,
  signal: AbortSignal,
) => Promise<void>;

export function createChatKvPrefillCoordinator(send: ChatKvPrefillSender): {
  cancel: () => void;
  start: (payload: OpenAIChatCompletionsRequest) => void;
} {
  let activeController: AbortController | null = null;

  const cancel = (): void => {
    const controller = activeController;
    activeController = null;
    controller?.abort();
  };

  const start = (payload: OpenAIChatCompletionsRequest): void => {
    cancel();
    const controller = new AbortController();
    activeController = controller;
    void send(payload, controller.signal)
      // Prefill is a latency optimization; it must never surface as a chat error.
      .catch(() => {})
      .finally(() => {
        if (activeController === controller) {
          activeController = null;
        }
      });
  };

  return { cancel, start };
}
