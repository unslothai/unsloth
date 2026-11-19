// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { modelVisibleMessage, withHeardCut } from "../src/features/chat/api/mcp-image.ts";
import {
  clearPendingHeardCut,
  pendingHeardCut,
  setPendingHeardCut,
} from "../src/features/chat/voice/voice-loop-bridge.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const REPLY = "Paris is the capital. It sits on the Seine.";
const message = (metadata?: unknown) => ({
  id: "a1",
  role: "assistant",
  metadata,
  content: [{ type: "text", text: REPLY }],
});

test("the next request is cut from memory while the old run is still cancelling", () => {
  clearPendingHeardCut("a1");
  assert.equal(modelVisibleMessage(message()).content[0].text, REPLY);
  setPendingHeardCut("a1", { offset: "Paris is the capital.".length });
  try {
    assert.equal(modelVisibleMessage(message()).content[0].text, "Paris is the capital.");
    const saved = message({ custom: { voiceHeard: { offset: REPLY.length, at: 1 } } });
    assert.equal(modelVisibleMessage(saved).content[0].text, REPLY);
  } finally {
    clearPendingHeardCut("a1");
  }
  assert.equal(pendingHeardCut("a1"), null);
});

test("a playback-only barge-in is cut too: nothing depends on a run being cancelled", () => {
  const done = { ...message({ custom: { voiceHeard: { offset: 0, at: 1 } } }), status: { type: "complete" } };
  assert.equal(withHeardCut(done).content[0].text, "");
});

type Store = typeof import("../src/features/chat/utils/update-thread-message.ts");

function storeWith(save: (record: unknown, options: unknown) => Promise<unknown>) {
  class ChatMessageProtectedError extends Error {}
  const saves: unknown[] = [];
  const mod = loadWithStubs<Store>(new URL("../src/features/chat/utils/update-thread-message.ts", import.meta.url), {
    "../api/chat-api": {
      saveChatMessage: (record: unknown, options: unknown) => {
        saves.push({ record, options });
        return save(record, options);
      },
    },
    "./delete-thread-message": {
      exportedItemToRecord: (threadId: string, parentId: string | null, m: { id: string; metadata?: unknown }) => ({
        id: m.id,
        threadId,
        parentId,
        metadata: m.metadata,
      }),
    },
    "./research-message-sync": { RESEARCH_METADATA_KEYS: [] },
    "./saved-context-usage": { withMessageRevision: (m: unknown) => m },
    "../voice/voice-loop-bridge": { clearPendingHeardCut, noteWrittenHeardCut: () => {}, pendingHeardCut, setPendingHeardCut },
  });
  return { mod, saves, ChatMessageProtectedError };
}

function thread() {
  let data = {
    headId: "a1",
    messages: [{ parentId: "u1", message: { id: "a1", role: "assistant", content: [], metadata: { custom: { kept: 1 } } } }],
  };
  return {
    export: () => data as never,
    import: (next: unknown) => {
      data = next as typeof data;
    },
    custom: () => (data.messages[0].message.metadata as { custom: Record<string, unknown> }).custom,
  };
}

test("the cut is saved through the edit path, and undo clears it the same way", async () => {
  const { mod, saves } = storeWith(async () => ({}));
  const t = thread();
  await mod.setThreadMessageVoiceHeard({ thread: t, messageId: "a1", remoteId: "th", isIncognito: false, heard: { offset: 9, at: 5 } });
  assert.deepEqual(t.custom(), { kept: 1, voiceHeard: { offset: 9, at: 5 } });
  assert.deepEqual((saves[0] as { options: unknown }).options, { allowGenerationEdit: true });
  await mod.setThreadMessageVoiceHeard({ thread: t, messageId: "a1", remoteId: "th", isIncognito: false, heard: null });
  assert.deepEqual(t.custom(), { kept: 1 });
  assert.equal(saves.length, 2);
  assert.equal(modelVisibleMessage(message({ custom: t.custom() })).content[0].text, REPLY);
});

test("a protected row rolls back and is not retried", async () => {
  const { mod, saves, ChatMessageProtectedError } = storeWith(async () => {
    throw new ChatMessageProtectedError("protected");
  });
  const t = thread();
  await assert.rejects(
    mod.setThreadMessageVoiceHeard({ thread: t, messageId: "a1", remoteId: "th", isIncognito: false, heard: { offset: 9, at: 5 } }),
    ChatMessageProtectedError,
  );
  assert.deepEqual(t.custom(), { kept: 1 });
  assert.equal(saves.length, 1);
});

test("a failed save keeps an edit made while it was in flight", async () => {
  let edit: () => void = () => {};
  const { mod } = storeWith(async () => {
    edit();
    throw new Error("network");
  });
  let data = {
    headId: "a1",
    messages: [
      { parentId: "u1", message: { id: "a1", role: "assistant", content: [{ type: "text", text: REPLY }], metadata: { custom: {} } } },
    ],
  };
  const t = { export: () => data as never, import: (next: unknown) => { data = next as typeof data; } };
  edit = () => {
    data = { ...data, messages: [{ ...data.messages[0], message: { ...data.messages[0].message, content: [{ type: "text", text: "Edited." }], metadata: { custom: {} } } }] };
  };
  await assert.rejects(mod.setThreadMessageVoiceHeard({ thread: t, messageId: "a1", remoteId: "th", isIncognito: false, heard: { offset: 9, at: 5 } }));
  assert.deepEqual(data.messages[0].message.content, [{ type: "text", text: "Edited." }]);
});

test("a failed save keeps a newer cut, and metadata it did not set, made while it was in flight", async () => {
  let during: () => void = () => {};
  const { mod } = storeWith(async () => {
    during();
    throw new Error("network");
  });
  const t = thread();
  during = () => {
    const current = t.export() as unknown as { messages: { message: { metadata: unknown } }[] };
    current.messages[0].message.metadata = { custom: { ...t.custom(), voiceHeard: { offset: 20, at: 6 }, other: 2 } };
  };
  await assert.rejects(mod.setThreadMessageVoiceHeard({ thread: t, messageId: "a1", remoteId: "th", isIncognito: false, heard: { offset: 9, at: 5 } }));
  assert.deepEqual(t.custom(), { kept: 1, voiceHeard: { offset: 20, at: 6 }, other: 2 });
});

test("editing a cut reply drops its cut, saved and pending", async () => {
  const { mod } = storeWith(async () => ({}));
  let data = {
    headId: "a1",
    messages: [
      {
        parentId: "u1",
        message: {
          id: "a1",
          role: "assistant",
          content: [{ type: "text", text: REPLY }],
          metadata: { custom: { voiceHeard: { offset: 21, at: 1 } } },
        },
      },
    ],
  };
  const t = { export: () => data as never, import: (next: unknown) => { data = next as typeof data; } };
  setPendingHeardCut("a1", { offset: 21 });
  await mod.updateThreadMessage({ thread: t, messageId: "a1", remoteId: "th", newText: "Rome is the capital.", isIncognito: false });
  const custom = (data.messages[0].message.metadata as { custom?: Record<string, unknown> }).custom ?? {};
  assert.equal("voiceHeard" in custom, false);
  assert.equal(pendingHeardCut("a1"), null);
  setPendingHeardCut("a1", { offset: 5 });
  await mod.updateThreadMessage({ thread: t, messageId: "a1", remoteId: "th", newText: "Rome is the capital.", isIncognito: false });
  assert.deepEqual(pendingHeardCut("a1"), { offset: 5 });
  clearPendingHeardCut("a1");
});

test("a reply edit that fails to save gives back the cut that pointed into its old text", async () => {
  const { mod } = storeWith(async () => {
    throw new Error("offline");
  });
  let data = {
    headId: "a1",
    messages: [
      {
        parentId: "u1",
        message: {
          id: "a1",
          role: "assistant",
          content: [{ type: "text", text: REPLY }],
          metadata: { custom: { voiceHeard: { offset: 21, at: 1 } } },
        },
      },
    ],
  };
  const t = { export: () => data as never, import: (next: unknown) => { data = next as typeof data; } };
  setPendingHeardCut("a1", { offset: 21 });
  const error = console.error;
  console.error = () => {};
  try {
    await assert.rejects(
      mod.updateThreadMessage({ thread: t, messageId: "a1", remoteId: "th", newText: "Rome is the capital.", isIncognito: false }),
      /offline/,
    );
  } finally {
    console.error = error;
  }
  assert.equal((data.messages[0].message.content[0] as { text: string }).text, REPLY);
  assert.deepEqual(pendingHeardCut("a1"), { offset: 21 });
  clearPendingHeardCut("a1");
});

test("an incognito thread keeps the cut on the thread only", async () => {
  const { mod, saves } = storeWith(async () => ({}));
  const t = thread();
  await mod.setThreadMessageVoiceHeard({ thread: t, messageId: "a1", remoteId: "th", isIncognito: true, heard: { offset: 1, at: 2 } });
  assert.equal(saves.length, 0);
  assert.deepEqual(t.custom(), { kept: 1, voiceHeard: { offset: 1, at: 2 } });
});
