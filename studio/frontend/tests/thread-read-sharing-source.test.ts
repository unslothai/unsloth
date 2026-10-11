// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const API = readSrc("features/chat/api/chat-api.ts");
const STORAGE = readSrc("features/chat/utils/chat-history-storage.ts");
const BAR = readSrc("features/rag/components/thread-documents-bar.tsx");

function slice(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  assert.ok(start !== -1, `not found: ${from}`);
  const end = source.indexOf(to, start + from.length);
  assert.ok(end !== -1, `not found: ${to}`);
  return source.slice(start, end);
}

test("only the UI readers share a thread read; every post-write reader stays private", () => {
  const uiRead = slice(STORAGE, "export async function getStoredChatThreadReadResult", "\n}");
  assert.match(uiRead, /share: true/);
  const history = slice(STORAGE, "export async function readStoredChatMessages", "\n}");
  assert.match(history, /getChatThread\(threadId, \{ share: true \}\)/);
  for (const name of [
    "export async function ensureStoredChatThread",
    "async function retryFailedThreadRecord",
    "async function publishForkBoundary",
    "export async function syncStoredChatMessages",
    "export async function deleteStoredChatThreads",
    "export async function readBackendChatThread",
  ]) {
    assert.doesNotMatch(slice(STORAGE, name, "\n}"), /share: true/, name);
  }
});

test("every thread write through chat-api drops the thread's in-flight read when it settles", () => {
  const write = slice(API, "async function threadWriteFetch", "\n}");
  assert.equal((write.match(/forgetThreadReads\(affected\)/g) ?? []).length, 2);
  assert.match(slice(API, "export async function saveChatThread", "\n}"), /\[thread\.id\],/);
  assert.match(slice(API, "export async function updateChatThread", "\n}"), /\[threadId\],\s*options\.signal,/);
  assert.match(slice(API, "export async function deleteChatThreads", "\n}"), /threadIds,\s*\);/);
  assert.match(slice(API, "export async function syncChatMessages", "\n}"), /\[threadId\],\s*\);/);
  assert.match(slice(API, "export async function clearBackendChats", "\n}"), /null,\s*\);/);
  assert.match(slice(API, "export async function saveChatMessage", "\n}"), /forgetThreadReads\(\[message\.threadId\]\)/);
  assert.match(slice(API, "export async function forkChatThread", "\n}"), /forgetThreadReads\(\[threadId, args\.newThreadId\]\)/);
  assert.match(slice(API, "export async function deleteChatProject", "\n}"), /forgetThreadReads\(null\)/);
});

test("the shared read keeps the deadline and abort handling the pairing relies on", () => {
  const get = slice(API, "export async function getChatThread", "\n}");
  assert.match(get, /options\.timeoutMs !== undefined/);
  assert.match(get, /combineAbortSignals\(\[timeout\.signal, options\.signal\]\)/);
  assert.match(get, /options\.share\s*\?\s*await shareThreadRead\(threadId, signal\)/);
  const share = slice(API, "function shareThreadRead", "\n}");
  assert.match(share, /\|\| performance\.now\(\) - shared\.startedAt > THREAD_READ_JOIN_MS/, "only a read started inside the join window is shared");
  assert.match(share, /\|\| shared\.authEpoch !== authEpoch/, "a read started under another auth session is never joined");
  assert.match(share, /if \(joined\.readers === 0\) joined\.controller\.abort\(signal\.reason\)/);
  assert.match(share, /reject\(abortError\(signal\)\)/);
});

test("the documents bar reads the row once per chat and follows the history event for moves", () => {
  const hook = slice(BAR, "function useThreadProjectId", "\n}");
  assert.match(hook, /\}, \[threadId\]\);/);
  assert.doesNotMatch(hook, /\[threadId, activeProjectId\]/);
  assert.match(hook, /subscribeChatHistoryUpdated\(\(\{ thread \}\) => \{/);
  assert.match(hook, /thread\.id !== threadId\) return;/);
  assert.match(hook, /useChatRuntimeStore\.getState\(\)\.activeProjectId/);
});
