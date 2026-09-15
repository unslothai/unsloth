// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  appendChatAudioTranscript,
  chatAudioUploadFenceMatches,
  chatAudioUploadFileError,
  completeChatAudioUpload,
} from "../src/features/chat/utils/chat-audio-upload.ts";

test("audio upload rejects empty and oversized files", () => {
  assert.equal(chatAudioUploadFileError({ size: 0 }), "The selected audio file is empty.");
  assert.equal(chatAudioUploadFileError({ size: 25 * 1024 * 1024 }), null);
  assert.match(
    chatAudioUploadFileError({ size: 25 * 1024 * 1024 + 1 }) ?? "",
    /smaller than 25MB/,
  );
});

test("audio upload accepts only its current owner, generation and auth session", () => {
  const started = { generation: 3, owner: "thread-a", authSessionEpoch: 7 };
  assert.equal(chatAudioUploadFenceMatches(started, { ...started }), true);
  assert.equal(
    chatAudioUploadFenceMatches(started, { ...started, generation: 4 }),
    false,
  );
  assert.equal(
    chatAudioUploadFenceMatches(started, { ...started, owner: "thread-b" }),
    false,
  );
  assert.equal(
    chatAudioUploadFenceMatches(started, { ...started, authSessionEpoch: 8 }),
    false,
  );
});

test("audio transcript appends once without replacing a draft typed meanwhile", () => {
  assert.equal(appendChatAudioTranscript("", " hello "), "hello");
  assert.equal(appendChatAudioTranscript("draft", " hello "), "draft hello");
  assert.equal(appendChatAudioTranscript("draft  ", "hello"), "draft hello");
  assert.equal(appendChatAudioTranscript("draft", "   "), "draft");
});

test("an ignored abort cannot commit a late transcript", async () => {
  let resolveTranscript: (value: string) => void = () => {};
  const transcript = new Promise<string>((resolve) => {
    resolveTranscript = resolve;
  });
  const controller = new AbortController();
  const fence = { generation: 1, owner: "thread-a", authSessionEpoch: 2 };
  let commits = 0;
  const completion = completeChatAudioUpload({
    started: fence,
    current: () => fence,
    signal: controller.signal,
    blocked: () => false,
    transcribe: () => transcript,
    commit: () => {
      commits += 1;
    },
  });
  controller.abort();
  resolveTranscript("late words");
  assert.equal(await completion, "stale");
  assert.equal(commits, 0);
});

test("an A to B to A generation change still rejects the first result", async () => {
  const started = { generation: 1, owner: "thread-a", authSessionEpoch: 2 };
  let current = started;
  let commits = 0;
  const completion = completeChatAudioUpload({
    started,
    current: () => current,
    signal: new AbortController().signal,
    blocked: () => false,
    transcribe: async () => {
      current = { ...started, generation: 3 };
      return "late words";
    },
    commit: () => {
      commits += 1;
    },
  });
  assert.equal(await completion, "stale");
  assert.equal(commits, 0);
});

test("a current result commits exactly once and empty audio commits nothing", async () => {
  const fence = { generation: 1, owner: "thread-a", authSessionEpoch: 2 };
  const committed: string[] = [];
  const run = (value: string) =>
    completeChatAudioUpload({
      started: fence,
      current: () => fence,
      signal: new AbortController().signal,
      blocked: () => false,
      transcribe: async () => value,
      commit: (result) => committed.push(result),
    });
  assert.equal(await run(" hello "), "committed");
  assert.deepEqual(committed, ["hello"]);
  assert.equal(await run("   "), "empty");
  assert.deepEqual(committed, ["hello"]);
});
