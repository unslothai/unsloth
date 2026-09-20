// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { beforeEach, test } from "node:test";
import type { SidebarItem } from "../src/features/chat/hooks/use-chat-sidebar-items.ts";
import type { MessageRecord } from "../src/features/chat/types.ts";
register("./refresh-chat-title-resolver.mjs", import.meta.url);
const { refreshChatTitle } = await import(
  "../src/features/chat/utils/refresh-chat-title.ts"
);
const { registerLiveThreadView } = await import(
  "../src/features/chat/utils/live-thread-head.ts"
);
const { state } = await import("./helpers/store-stubs/refresh-chat-title.ts");
const item: SidebarItem = {
  id: "chat",
  type: "single",
  title: "Follow Up Email",
  createdAt: 1,
  updatedAt: 1,
};
function message(
  id: string,
  role: MessageRecord["role"],
  text: string,
  parentId?: string,
): MessageRecord {
  return {
    id,
    role,
    parentId,
    threadId: "chat",
    content: [{ type: "text", text }],
    createdAt: state.messages.length + 1,
  };
}
beforeEach(() => {
  state.model = "local-model";
  state.contextLength = 4096;
  state.threads = [{ ...item, modelType: "base", archived: false }];
  state.messages = [];
  state.messages.push(message("u1", "user", "Help write a follow-up email"));
  state.messages.push(message("a1", "assistant", "Here is the email", "u1"));
  state.messages.push(
    message("u2", "user", "Now negotiate the contract liability clause", "a1"),
  );
  state.messages.push(
    message("a2", "assistant", "Cap liability and agree renewal terms", "u2"),
  );
  state.requests = [];
  state.writes = [];
  state.status = 200;
  state.response = {
    choices: [
      { message: { content: "Contract Negotiation" }, finish_reason: "stop" },
    ],
  };
  state.wait = undefined;
  state.providers = [];
  state.connectionsEnabled = true;
});

test("refresh uses the entire branch with auto-title off and disables tools", async () => {
  await refreshChatTitle(item);
  assert.equal(state.threads[0].title, "Contract Negotiation");
  const request = state.requests[0];
  for (const row of state.messages)
    assert.ok(
      request.messages[1].content.includes(
        (row.content[0] as { text: string }).text,
      ),
    );
  assert.equal(request.enable_tools, false);
  assert.equal(request.enable_thinking, false);
  assert.equal(request.stream, false);
  assert.equal(request.model, "local-model");
});

test("refresh uses the visible branch rather than a newer alternate reply", async () => {
  state.messages.push(
    message("alternate", "assistant", "Discarded answer", "u1"),
  );
  const unregister = registerLiveThreadView({
    threadListItem: () => ({ getState: () => ({ remoteId: "chat" }) }),
    thread: () => ({
      getState: () => ({
        messages: [{ id: "u1" }, { id: "a1" }, { id: "u2" }, { id: "a2" }],
      }),
    }),
  });
  try {
    await refreshChatTitle(item);
    assert.ok(state.requests[0].messages[1].content.includes("liability"));
    assert.ok(!state.requests[0].messages[1].content.includes("Discarded"));
  } finally {
    unregister();
  }
});

test("legacy flat history and pasted text contribute to the title", async () => {
  state.messages = state.messages.map((row) => ({
    ...row,
    parentId: undefined,
  }));
  state.messages[2].attachments = [
    {
      id: "paste",
      type: "file",
      name: "notes.txt",
      contentType: "text/plain",
      status: { type: "complete" },
      content: [{ type: "text", text: "[File: notes.txt]\nNegotiation notes" }],
    },
  ];
  await refreshChatTitle(item);
  assert.ok(
    state.requests[0].messages[1].content.includes("Negotiation notes"),
  );
  assert.ok(state.requests[0].messages[1].content.includes("renewal"));
});

test("comparison panes share one title request and both titles update", async () => {
  state.threads[0].pairId = "pair";
  state.threads.push({ ...state.threads[0], id: "right" });
  state.messages.push({
    ...message("right-answer", "assistant", "Other pane context"),
    threadId: "right",
  });
  await refreshChatTitle({ ...item, id: "pair", type: "compare" });
  assert.equal(state.requests.length, 1);
  assert.ok(
    state.requests[0].messages[1].content.includes("Other pane context"),
  );
  assert.deepEqual(state.writes.sort(), ["chat", "right"]);
});

for (const failure of [
  "http",
  "blank",
  "truncated",
  "thinking",
  "echo",
] as const) {
  test(`${failure} generation preserves the existing title`, async () => {
    if (failure === "http") state.status = 503;
    if (failure === "blank") state.response.choices[0].message.content = "";
    if (failure === "truncated")
      state.response.choices[0].finish_reason = "length";
    if (failure === "thinking")
      state.response.choices[0].message.content = "<think>thoughts</think>";
    if (failure === "echo")
      state.response.choices[0].message.content = "user: help";
    await assert.rejects(refreshChatTitle(item));
    assert.equal(state.threads[0].title, item.title);
    assert.equal(state.writes.length, 0);
  });
}

test("missing model and empty history do not send a generation request", async () => {
  state.model = "";
  await assert.rejects(refreshChatTitle(item), /Select a model/);
  state.model = "local-model";
  state.messages = [];
  await assert.rejects(refreshChatTitle(item), /no text/);
  assert.equal(state.requests.length, 0);
});

for (const action of ["rename", "delete"] as const) {
  test(`a ${action} during generation wins and duplicate clicks share the request`, async () => {
    let release!: () => void;
    state.wait = new Promise<void>((resolve) => {
      release = resolve;
    });
    const request = refreshChatTitle(item);
    assert.equal(refreshChatTitle(item), request);
    while (!state.requests.length)
      await new Promise((resolve) => setImmediate(resolve));
    state.threads =
      action === "delete"
        ? []
        : [{ ...state.threads[0], title: "Manual title" }];
    release();
    await assert.rejects(request);
    assert.equal(state.writes.length, 0);
    assert.equal(state.requests.length, 1);
    if (action === "rename")
      assert.equal(state.threads[0].title, "Manual title");
  });
}

test("saved connections route to the selected provider and respect disabled connections", async () => {
  state.model = "external::provider::chat-model";
  state.providers = [
    {
      id: "provider",
      providerType: "openai",
      baseUrl: "https://example.invalid/v1",
      hasApiKey: true,
    },
  ];
  await refreshChatTitle(item);
  assert.equal(state.requests[0].provider_id, "provider");
  assert.equal(state.requests[0].external_model, "chat-model");
  assert.equal(state.requests[0].provider_type, "openai");
  state.connectionsEnabled = false;
  await assert.rejects(refreshChatTitle(item));
  assert.equal(state.requests.length, 1);
});

test("refresh works without native AbortSignal.timeout and releases its timer", async () => {
  const nativeTimeout = AbortSignal.timeout;
  const originalSetTimeout = globalThis.setTimeout;
  const originalClearTimeout = globalThis.clearTimeout;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let disposed = false;
  Object.defineProperty(AbortSignal, "timeout", {
    configurable: true,
    value: undefined,
  });
  globalThis.setTimeout = ((...args: Parameters<typeof setTimeout>) => {
    timer = originalSetTimeout(...args);
    return timer;
  }) as typeof setTimeout;
  globalThis.clearTimeout = ((id: ReturnType<typeof setTimeout>) => {
    if (id === timer) disposed = true;
    return originalClearTimeout(id);
  }) as typeof clearTimeout;
  try {
    await refreshChatTitle(item);
    assert.equal(state.writes.length, 1);
    assert.equal(disposed, true);
  } finally {
    if (timer) originalClearTimeout(timer);
    globalThis.setTimeout = originalSetTimeout;
    globalThis.clearTimeout = originalClearTimeout;
    Object.defineProperty(AbortSignal, "timeout", {
      configurable: true,
      value: nativeTimeout,
    });
  }
});

test("long multilingual transcripts fit the loaded context and keep the latest topic", async () => {
  state.contextLength = 2048;
  state.messages[1].content = [
    { type: "text", text: "Older details 🦥 ".repeat(5000) },
  ];
  await refreshChatTitle(item);
  const transcript = state.requests[0].messages[1].content;
  assert.ok(new TextEncoder().encode(transcript).length <= 1536);
  assert.ok(transcript.includes("follow-up email"));
  assert.ok(transcript.includes("liability clause"));
  assert.ok(transcript.includes("renewal terms"));
});

test("legacy string content contributes to the refreshed title", async () => {
  state.messages = JSON.parse(
    JSON.stringify(
      state.messages.map((row) => ({
        ...row,
        content: row.content
          .filter((part) => part.type === "text")
          .map((part) => part.text)
          .join("\n"),
      })),
    ),
  );
  await refreshChatTitle(item);
  const transcript = state.requests[0].messages[1].content;
  assert.ok(transcript.includes("follow-up email"));
  assert.ok(transcript.includes("liability clause"));
  assert.ok(transcript.includes("renewal terms"));
  assert.equal(state.threads[0].title, "Contract Negotiation");
});

test("ChatGPT subscription titles use streaming and collect the text deltas", async () => {
  state.model = "external::subscription::gpt-5.3-codex";
  state.providers = [
    {
      id: "subscription",
      providerType: "openai_codex",
      baseUrl: "",
      hasApiKey: true,
    },
  ];
  await refreshChatTitle(item);
  assert.equal(state.requests[0].stream, true);
  assert.equal(state.requests[0].enable_tools, false);
  assert.equal(state.threads[0].title, "Contract Negotiation");
});

test("a truncated subscription title preserves the existing title", async () => {
  state.model = "external::subscription::gpt-5.3-codex";
  state.providers = [
    {
      id: "subscription",
      providerType: "openai_codex",
      baseUrl: "",
      hasApiKey: true,
    },
  ];
  state.response.choices[0].finish_reason = "length";
  await assert.rejects(refreshChatTitle(item));
  assert.equal(state.threads[0].title, item.title);
});

test("subscription failures return null for the automatic title fallback", async () => {
  const { generateChatTitle } = await import(
    "../src/features/chat/utils/generate-chat-title.ts"
  );
  state.providers = [
    {
      id: "subscription",
      providerType: "openai_codex",
      baseUrl: "",
      hasApiKey: true,
    },
  ];
  state.status = 401;
  assert.equal(
    await generateChatTitle(
      "user: Help write an email",
      "external::subscription::gpt-5.3-codex",
    ),
    null,
  );
});
