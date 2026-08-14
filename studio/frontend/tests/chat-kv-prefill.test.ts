// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { OpenAIChatCompletionsRequest } from "../src/features/chat/types/api.ts";
import {
  buildChatKvPrefillPayload,
  createChatKvPrefillCoordinator,
  isChatKvPrefillAvailable,
} from "../src/features/chat/utils/chat-kv-prefill.ts";

function request(
  updates: Partial<OpenAIChatCompletionsRequest> = {},
): OpenAIChatCompletionsRequest {
  return {
    model: "local-model",
    messages: [{ role: "user", content: "Question" }],
    stream: true,
    max_tokens: 128,
    enable_thinking: true,
    ...updates,
  };
}

test("prefill is available only for a resident llama.cpp chat model", () => {
  const loadedGguf = {
    isExternalModel: false,
    residentCheckpoint: "local-model",
    ggufContextLength: 4096,
    loadedIsDiffusion: false,
    loadedIsAudio: false,
  };

  assert.equal(isChatKvPrefillAvailable(loadedGguf), true);
  assert.equal(
    isChatKvPrefillAvailable({ ...loadedGguf, residentCheckpoint: null }),
    false,
  );
  assert.equal(
    isChatKvPrefillAvailable({ ...loadedGguf, residentCheckpoint: undefined }),
    false,
  );
  assert.equal(
    isChatKvPrefillAvailable({ ...loadedGguf, ggufContextLength: null }),
    false,
  );
  assert.equal(
    isChatKvPrefillAvailable({ ...loadedGguf, isExternalModel: true }),
    false,
  );
  assert.equal(
    isChatKvPrefillAvailable({ ...loadedGguf, loadedIsDiffusion: true }),
    false,
  );

  assert.equal(
    isChatKvPrefillAvailable({ ...loadedGguf, loadedIsAudio: true }),
    false,
  );
});

test("prefill payload appends the finalized assistant replay without changing controls", () => {
  const payload = buildChatKvPrefillPayload(request(), [
    {
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: "call-1",
          type: "function",
          function: { name: "python", arguments: '{"code":"1+1"}' },
        },
      ],
    },
    {
      role: "tool",
      tool_call_id: "call-1",
      name: "python",
      content: "2",
    },
    {
      role: "assistant",
      content: "Done",
    },
  ]);

  assert.ok(payload);
  assert.equal(payload.stream, false);
  assert.equal(payload.continue_final_message, false);
  assert.equal(payload.max_tokens, 128);
  assert.equal(payload.enable_thinking, true);
  assert.deepEqual(payload.messages.slice(1), [
    {
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: "call-1",
          type: "function",
          function: { name: "python", arguments: '{"code":"1+1"}' },
        },
      ],
    },
    {
      role: "tool",
      tool_call_id: "call-1",
      name: "python",
      content: "2",
    },
    { role: "assistant", content: "Done" },
  ]);
});

test("continued responses replace the partial trailing assistant", () => {
  const payload = buildChatKvPrefillPayload(
    request({
      messages: [
        { role: "user", content: "Count" },
        { role: "assistant", content: "One, two" },
      ],
      continue_final_message: true,
    }),
    [{ role: "assistant", content: "One, two, three" }],
  );

  assert.deepEqual(payload?.messages, [
    { role: "user", content: "Count" },
    { role: "assistant", content: "One, two, three" },
  ]);
});

test("empty text is skipped while tool-only responses are prefilled", () => {
  assert.equal(
    buildChatKvPrefillPayload(request(), [
      { role: "assistant", content: "" },
    ]),
    null,
  );

  const payload = buildChatKvPrefillPayload(request(), [
    {
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: "call-1",
          type: "function",
          function: { name: "python", arguments: "{}" },
        },
      ],
    },
    { role: "tool", tool_call_id: "call-1", content: "2" },
  ]);
  assert.ok(payload);
  assert.equal(payload.messages.at(-1)?.role, "tool");
});

test("new work aborts the previous prefill and failures stay best effort", async () => {
  const signals: AbortSignal[] = [];
  const resolvers: Array<() => void> = [];
  const coordinator = createChatKvPrefillCoordinator(
    (_payload, signal) =>
      new Promise<void>((resolve) => {
        signals.push(signal);
        resolvers.push(resolve);
      }),
  );

  coordinator.start(request());
  coordinator.start(request({ model: "newer" }));
  assert.equal(signals[0].aborted, true);
  assert.equal(signals[1].aborted, false);

  resolvers[0]();
  await Promise.resolve();
  coordinator.cancel();
  assert.equal(signals[1].aborted, true);
  resolvers[1]();
});
