// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  MessageRecord,
  ThreadRecord,
} from "../../../src/features/chat/types.ts";

export const state = {
  model: "local-model",
  contextLength: 4096,
  threads: [] as ThreadRecord[],
  messages: [] as MessageRecord[],
  requests: [] as {
    model: string;
    messages: { content: string }[];
    enable_tools: boolean;
    enable_thinking: boolean;
    stream: boolean;
    reasoning_effort?: string;
    provider_id?: string;
    external_model?: string;
    provider_type?: string;
  }[],
  writes: [] as string[],
  response: {
    choices: [
      { message: { content: "Contract Negotiation" }, finish_reason: "stop" },
    ],
  },
  status: 200,
  wait: undefined as Promise<void> | undefined,
  providers: [] as {
    id: string;
    providerType: string;
    baseUrl: string;
    hasApiKey: boolean;
  }[],
  connectionsEnabled: true,
};
export const useChatRuntimeStore = {
  getState: () => ({
    params: { checkpoint: state.model, maxSeqLength: 4096 },
    loadedContextLength: state.contextLength,
    autoTitle: false,
  }),
};
export const useExternalProvidersStore = { getState: () => state };
export async function getStoredChatThread(id: string) {
  return state.threads.find((thread) => thread.id === id);
}
export async function listStoredChatThreads({ pairId }: { pairId: string }) {
  return state.threads.filter((thread) => thread.pairId === pairId);
}
export async function listStoredChatMessages(id: string) {
  return state.messages.filter((message) => message.threadId === id);
}
export async function updateChatThread(
  id: string,
  patch: { title: string },
  options: { expectedTitle?: string },
) {
  const thread = await getStoredChatThread(id);
  if (!thread) throw new Error("Chat deleted");
  if (thread.title !== options.expectedTitle) throw new Error("Title changed");
  state.threads = state.threads.map((row) =>
    row.id === id ? { ...row, ...patch } : row,
  );
  state.writes.push(id);
}
export async function authFetch(_url: string, init: RequestInit) {
  state.requests.push(JSON.parse(init.body as string));
  const request = state.requests.at(-1)!;
  if (request.provider_type === "openai_codex" && !request.stream) {
    return new Response("ChatGPT subscription chat requires stream=true.", {
      status: 400,
    });
  }
  if (request.provider_type) {
    return new Response('data: {"choices":[]}\n\ndata: [DONE]\n\n', {
      headers: { "Content-Type": "text/event-stream" },
    });
  }
  await state.wait;
  return new Response(JSON.stringify(state.response), { status: state.status });
}
export async function encryptProviderApiKey(value: string) {
  return `encrypted:${value}`;
}

export async function* streamChatCompletions(payload: unknown) {
  state.requests.push(JSON.parse(JSON.stringify(payload)));
  if (state.status !== 200) throw new Error("Subscription unavailable");
  const choice = state.response.choices[0];
  yield {
    choices: [{ delta: { content: choice.message.content.slice(0, 9) } }],
  };
  yield {
    choices: [
      {
        delta: { content: choice.message.content.slice(9) },
        finish_reason: choice.finish_reason,
      },
    ],
  };
}
