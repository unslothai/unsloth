// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register, stripTypeScriptTypes } from "node:module";
import { beforeEach, test } from "node:test";
register("./refresh-chat-title-resolver.mjs", import.meta.url);
const { refreshChatTitle } = await import(
  "../src/features/chat/utils/refresh-chat-title.ts"
);
const { queueChatTitle } = await import(
  "../src/features/chat/utils/chat-title-queue.ts"
);
const storage = await import("./helpers/store-stubs/refresh-chat-title.ts");
const { state } = storage;
const source = readFileSync(
  new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
  "utf8",
);
const start = source.indexOf("    async generateTitle(remoteId:");
const end = source.indexOf("\n    },\n  };\n}", start) + "\n    },".length;
assert.ok(start >= 0 && end > start);
const method = source.slice(start, end);
let release: () => void;
let started: Promise<void>;
function adapter(pairId?: string) {
  let begin!: () => void;
  started = new Promise<void>((resolve) => {
    begin = resolve;
  });
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const dependencies = {
    pairId,
    modelType: "base",
    projectId: undefined,
    queueChatTitle,
    useChatRuntimeStore: {
      getState: () => ({ autoTitle: true, runningByThreadId: {} }),
    },
    ensureStoredChatThread: storage.getStoredChatThread,
    getStoredChatThread: storage.getStoredChatThread,
    listStoredChatThreads: storage.listStoredChatThreads,
    updateChatThread: storage.updateChatThread,
    updateStoredChatThread: async (id: string, patch: { title: string }) => {
      state.threads = state.threads.map((row) =>
        row.id === id ? { ...row, ...patch } : row,
      );
    },
    createAssistantStream: (
      write: (controller: {
        appendText: (text: string) => void;
        close: () => void;
      }) => void,
    ) => {
      let value = "";
      write({
        appendText: (text) => {
          value += text;
        },
        close: () => {},
      });
      return value;
    },
    titleTextOf: () => "Opening email",
    extractTextParts: () => "Initial answer",
    fallbackTitleFromUserText: (text: string) => text,
    generateTitleWithModel: async () => {
      begin();
      await gate;
      return "Opening Email";
    },
  };
  return new Function(
    ...Object.keys(dependencies),
    stripTypeScriptTypes(`function make() { return {${method}}; }`) +
      "\nreturn make();",
  )(...Object.values(dependencies));
}
beforeEach(() => {
  state.model = "local-model";
  state.contextLength = 4096;
  state.threads = [
    {
      id: "chat",
      title: "New Chat",
      modelType: "base",
      createdAt: 1,
      updatedAt: 1,
      archived: false,
    },
  ];
  state.messages = [
    {
      id: "m",
      threadId: "chat",
      role: "user",
      content: [{ type: "text", text: "Negotiate contract terms" }],
      createdAt: 1,
    },
  ];
  state.requests = [];
  state.writes = [];
  state.status = 200;
  state.guardSupport = true;
  state.response = {
    choices: [
      { message: { content: "Contract Negotiation" }, finish_reason: "stop" },
    ],
  };
  state.wait = undefined;
  state.providers = [];
});
test("explicit refresh finishes after an already-running automatic title", async () => {
  const automatic = adapter().generateTitle("chat", state.messages);
  await started;
  const refreshing = refreshChatTitle({
    id: "chat",
    type: "single",
    title: "New Chat",
    createdAt: 1,
    updatedAt: 1,
  });
  await new Promise((resolve) => setTimeout(resolve, 20));
  release();
  await Promise.all([automatic, refreshing]);
  assert.equal(state.threads[0].title, "Contract Negotiation");
});
for (const compare of [false, true]) {
  test(`automatic title preserves a newer ${compare ? "comparison" : "single"} title from another client`, async () => {
    if (compare) {
      state.threads[0].pairId = "pair";
      state.threads.push({ ...state.threads[0], id: "right" });
    }
    const automatic = adapter(compare ? "pair" : undefined).generateTitle(
      "chat",
      state.messages,
    );
    await started;
    state.threads = state.threads.map((row) => ({
      ...row,
      title: "Contract Negotiation",
    }));
    release();
    assert.equal(await automatic, "Contract Negotiation");
    assert.ok(
      state.threads.every((row) => row.title === "Contract Negotiation"),
    );
  });
}

test("an automatic title request times out and releases its timer", async () => {
  const start = source.indexOf("async function generateTitleWithModel(");
  const end = source.indexOf("\nfunction cloneContent", start);
  const declaration = source.slice(start, end);
  let disposed = false;
  const controller = new AbortController();
  const generate = new Function(
    "useChatRuntimeStore",
    "clip",
    "disposableTimeoutSignal",
    "generateChatTitle",
    stripTypeScriptTypes(declaration) + "\nreturn generateTitleWithModel;",
  )(
    { getState: () => ({ params: { checkpoint: "local-model" } }) },
    (text: string, limit: number) => text.slice(0, limit),
    (milliseconds: number) => {
      assert.equal(milliseconds, 60_000);
      return {
        signal: controller.signal,
        dispose: () => {
          disposed = true;
        },
      };
    },
    (_conversation: string, _model: string, signal: AbortSignal) =>
      new Promise((_resolve, reject) => {
        signal.addEventListener("abort", () => reject(new Error("Timed out")), {
          once: true,
        });
      }),
  );
  const result = generate({
    userText: "Opening email",
    assistantText: "Initial reply",
  });
  controller.abort();
  assert.equal(await result, null);
  assert.equal(disposed, true);
});
