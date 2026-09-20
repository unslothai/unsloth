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
  state.guardSupport = true;
  state.rotationFailures = 0;
  state.encryptions = [];
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
  assert.match(
    request.messages[0].content,
    /Reflect how the topic has evolved\./,
  );
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

test("a comparison conflict restores the successfully refreshed sibling", async () => {
  state.threads[0].pairId = "pair";
  state.threads.push({ ...state.threads[0], id: "right" });
  let release!: () => void;
  state.wait = new Promise<void>((resolve) => {
    release = resolve;
  });
  const refresh = refreshChatTitle({ ...item, id: "pair", type: "compare" });
  while (state.requests.length === 0)
    await new Promise((resolve) => setTimeout(resolve, 0));
  state.threads = state.threads.map((thread) =>
    thread.id === "right" ? { ...thread, title: "Manual title" } : thread,
  );
  release();
  await assert.rejects(refresh, /Title changed/);
  assert.equal(
    state.threads.find((thread) => thread.id === "chat")?.title,
    item.title,
  );
  assert.equal(
    state.threads.find((thread) => thread.id === "right")?.title,
    "Manual title",
  );
});

for (const providerType of ["openai", "anthropic", "vllm"]) {
  test(`${providerType} connection titles consume the backend SSE response`, async () => {
    state.model = "external::connection::chat-model";
    state.providers = [
      {
        id: "connection",
        providerType,
        baseUrl: "https://example.invalid/v1",
        hasApiKey: true,
      },
    ];
    await refreshChatTitle(item);
    assert.equal(state.requests[0].stream, true);
    assert.equal(state.requests[0].reasoning_effort, "none");
    assert.equal(state.threads[0].title, "Contract Negotiation");
  });
}

test("older backends cannot receive unguarded title refreshes", async () => {
  state.guardSupport = false;
  await assert.rejects(refreshChatTitle(item), /Update Studio/);
  assert.equal(state.requests.length, 0);
  assert.equal(state.writes.length, 0);
});

for (const failures of [1, 2]) {
  test(`legacy key rotation retries once with ${failures} decryption failures`, async () => {
    const previousWindow = globalThis.window;
    const previousStorage = globalThis.localStorage;
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: {},
    });
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: { getItem: () => JSON.stringify({ legacy: "fixture-key" }) },
    });
    try {
      state.model = "external::legacy::chat-model";
      state.providers = [
        { id: "legacy", providerType: "openai", baseUrl: "", hasApiKey: false },
      ];
      state.rotationFailures = failures;
      if (failures === 1) await refreshChatTitle(item);
      else await assert.rejects(refreshChatTitle(item));
      assert.equal(state.requests.length, 2);
      assert.deepEqual(state.encryptions, [false, true]);
      assert.equal(
        state.threads[0].title,
        failures === 1 ? "Contract Negotiation" : item.title,
      );
    } finally {
      Object.defineProperty(globalThis, "window", {
        configurable: true,
        value: previousWindow,
      });
      Object.defineProperty(globalThis, "localStorage", {
        configurable: true,
        value: previousStorage,
      });
    }
  });
}

test("long comparisons preserve both panes within one context budget", async () => {
  state.threads[0].pairId = "pair";
  state.threads.push({ ...state.threads[0], id: "right" });
  state.messages = [
    message(
      "left",
      "user",
      `Opening left topic ${"older details ".repeat(1000)} Latest left topic`,
    ),
    {
      ...message(
        "right",
        "user",
        `Opening right topic ${"older details ".repeat(1000)} Latest right topic`,
      ),
      threadId: "right",
    },
  ];
  await refreshChatTitle({ ...item, id: "pair", type: "compare" });
  const transcript = state.requests[0].messages[1].content;
  for (const fragment of [
    "Opening left topic",
    "Latest left topic",
    "Opening right topic",
    "Latest right topic",
    "Another comparison pane:",
  ])
    assert.ok(transcript.includes(fragment), fragment);
  assert.ok(new TextEncoder().encode(transcript).length <= 3584);
});
