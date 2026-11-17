// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Phase 4: when a heard cut is taken, where it is read, how it is saved and undone.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { modelVisibleMessage, withHeardCut } from "../src/features/chat/api/mcp-image.ts";
import {
  clearPendingHeardCut,
  pendingHeardCut,
  setPendingHeardCut,
} from "../src/features/chat/voice/voice-loop-bridge.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const src = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");
const engine = src("../src/features/chat/voice/voice-engine.tsx");
const adapter = src("../src/features/chat/api/chat-adapter.ts");

const REPLY = "Paris is the capital. It sits on the Seine.";
const message = (metadata?: unknown) => ({
  id: "a1",
  role: "assistant",
  metadata,
  content: [{ type: "text", text: REPLY }],
});

test("the snapshot is taken before the pause stops the clock", () => {
  const handler = engine.slice(engine.indexOf("registerVoiceBargeIn(() => {"));
  const snap = handler.indexOf("heardSnapshotRef.current = takeHeardSnapshot();");
  const pause = handler.indexOf("pausePlayback();");
  assert.ok(snap > 0 && snap < pause);
  // A false interruption drops it; a confirmed one records the cut before the turn is sent.
  assert.match(engine, /registerVoiceBargeAbort\(\(\) => \{[\s\S]*?heardSnapshotRef\.current = null;[\s\S]*?resumePlayback\(\);/);
  // The cut is recorded where the barge-in takes effect, so an ignored or queued one cuts nothing.
  assert.match(engine, /registerVoiceSubmit\(\(text, meta\) => \{\s*submitTranscript\(text, false, meta\);/);
  assert.match(
    engine,
    /if \(cutRun \|\| isSpeakingRef\.current \|\| isPlayingRef\.current\) heardCutRecorderRef\.current\(\);\s*else heardSnapshotRef\.current = null;/,
  );
  assert.match(engine, /heardSnapshotRef\.current = null;\s*\/\/ Nothing cut the reply the barge-in held[^\n]*\n\s*resumePlayback\(\);/);
  // Only the reply still at the tail: a snapshot from an earlier reply never cuts it later.
  assert.match(engine, /if \(!msg \|\| msg\.id !== snap\.messageId\) return;/);
});

test("the next request is cut from memory while the old run is still cancelling", () => {
  clearPendingHeardCut("a1");
  assert.equal(modelVisibleMessage(message()).content[0].text, REPLY);
  setPendingHeardCut("a1", { offset: "Paris is the capital.".length });
  try {
    assert.equal(modelVisibleMessage(message()).content[0].text, "Paris is the capital.");
    // The saved cut and the pending one: the later (larger) one wins.
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
  // Recorded from the snapshot alone; only a finished reply heard to its end is skipped.
  assert.match(engine, /if \(!cut \|\| \(settled && cut\.offset >= text\.trimEnd\(\)\.length\)\) return;/);
});

test("every request path reads the cut, pruning included, so the token count matches the send", () => {
  assert.match(adapter, /function toOpenAIMessages\([\s\S]*?\{\s*message = modelVisibleMessage\(message\);/);
  assert.match(adapter, /isAbandonedAssistantTurn\(withHeardCut\(message\), includeReasoningContent\)/);
  // The count and the send both prune, then serialize through toOpenAIMessages.
  assert.equal(adapter.match(/pruneOutboundHistory\(messages, /g)?.length, 2);
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
  // Undone: the model gets the full reply again.
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
  // The engine traces the refusal and keeps the cut in memory; only another failure is tried again.
  const save = engine.slice(engine.indexOf("const saveHeardCut"), engine.indexOf("const recordHeardCut"));
  const onError = save.slice(save.indexOf("(error: unknown) =>"), save.lastIndexOf("    tick();"));
  assert.match(onError, /voice\.heardSaveRefused/);
  assert.match(onError, /if \(!refused\) later\(/);
  assert.doesNotMatch(onError, /clearPendingHeardCut/);
});

test("a cut whose thread is not on screen waits for it instead of being dropped", () => {
  const save = engine.slice(engine.indexOf("const saveHeardCut"), engine.indexOf("const recordHeardCut"));
  assert.match(save, /if \(!thread\.getState\(\)\.messages\.some\(\(m\) => m\.id === messageId\)\) \{\s*later\(\d+\);\s*return;/);
  // Every wait counts against one budget, so a deleted reply stops trying.
  assert.match(save, /waited \+= ms;\s*if \(waited <= HEARD_SAVE_PATIENCE_MS\) setTimeout\(tick, ms\);\s*else \{[^}]*?voiceTrace\("voice\.heardSaveGaveUp", \{ messageId \}\);\s*done\(\);/);
  assert.doesNotMatch(save, /setTimeout\(tick, 500\)/);
});

test("one save loop runs per reply, and a cut taken while it saved goes next", () => {
  const save = engine.slice(engine.indexOf("const saveHeardCut"), engine.indexOf("const recordHeardCut"));
  assert.match(save, /if \(_heardSaving\.has\(messageId\)\) return;\s*_heardSaving\.add\(messageId\);/);
  // Every way the loop ends lets the next barge-in start one.
  assert.match(save, /if \(!cut\) \{\s*done\(\);\s*return;/);
  assert.match(save, /if \(pendingHeardCut\(messageId\) !== cut\) return tick\(\);\s*clearPendingHeardCut\(messageId\);\s*done\(\);/);
  assert.match(save, /if \(!refused\) later\([^\n]*\);\s*else done\(\);/);
});

test("a barge-in dropped by a Stop or reset gives the paused reply back", () => {
  const adapterSrc = src("../src/features/chat/adapters/studio-voice-loop-adapter.ts");
  const drops = [...adapterSrc.matchAll(/voiceTrace\("stt\.drop", \{ transcript, why: "cleared" \}\);([\s\S]*?)finish\("stopped"\);/g)];
  assert.equal(drops.length, 2);
  for (const [, between] of drops) assert.match(between, /if \(bargedIn\) requestVoiceBargeAbort\(\);/);
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
  // Saved with the same text: a pending cut still means something and is kept.
  setPendingHeardCut("a1", { offset: 5 });
  await mod.updateThreadMessage({ thread: t, messageId: "a1", remoteId: "th", newText: "Rome is the capital.", isIncognito: false });
  assert.deepEqual(pendingHeardCut("a1"), { offset: 5 });
  clearPendingHeardCut("a1");
});

test("an incognito thread keeps the cut on the thread only", async () => {
  const { mod, saves } = storeWith(async () => ({}));
  const t = thread();
  await mod.setThreadMessageVoiceHeard({ thread: t, messageId: "a1", remoteId: "th", isIncognito: true, heard: { offset: 1, at: 2 } });
  assert.equal(saves.length, 0);
  assert.deepEqual(t.custom(), { kept: 1, voiceHeard: { offset: 1, at: 2 } });
});
