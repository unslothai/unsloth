// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register, stripTypeScriptTypes } from "node:module";
import test from "node:test";
import { installLocalStorageFake } from "./helpers/kit.ts";

const browser = installLocalStorageFake();
browser.store.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);
const { threadRows } = await import(
  "./helpers/store-stubs/chat-history-storage.ts"
);
const { sanitizeThreadScopedSettings } = await import(
  "../src/features/chat/utils/thread-scoped-settings.ts"
);
const STORE = new URL(
  "../src/features/chat/stores/chat-runtime-store.ts",
  import.meta.url,
).href;
const THREAD = "chat";
const PAINT = () => new Promise<void>((resolve) => setImmediate(resolve));
function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

async function start(name: string) {
  const mod = await import(`${STORE}?scenario=${name}`);
  const state = () => mod.useChatRuntimeStore.getState();
  state().setPermissionMode("auto");
  state().setActiveThreadId(THREAD);
  state().applyThreadScopedSettings(
    THREAD,
    sanitizeThreadScopedSettings(threadRows.rows.get(THREAD)),
  );
  return { mod, state };
}

function params(
  tab: Awaited<ReturnType<typeof start>>,
  edits: Record<string, unknown>,
) {
  tab.state().setParams({ ...tab.state().params, ...edits });
}

const provider = readFileSync(
  new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
  "utf8",
);
function remoteSync(
  tab: Awaited<ReturnType<typeof start>>,
  read: () => unknown,
) {
  const source = provider.slice(
    provider.indexOf("    let remoteRead: AbortController"),
    provider.indexOf('    window.addEventListener("storage", syncRemote)'),
  );
  const boundary = source.lastIndexOf("    sync();");
  return new Function(
    "activeThreadId",
    "awaitThreadScopedSettingsWrite",
    "getThreadSettingsEditVersion",
    "THREAD_SETTINGS_REVISION_KEY",
    "getStoredChatThreadReadResult",
    "applyThreadScopedSettings",
    "sanitizeThreadScopedSettings",
    "return " +
      stripTypeScriptTypes(`(function() {
      const paired = true;
      const cancelled = false;
      const THREAD_READ_TIMEOUT_MS = 8000;
      const forgetThreadReads = () => {};
      ${source.slice(0, boundary)}
      return () => syncRemote({ key: THREAD_SETTINGS_REVISION_KEY });
    })()`),
  )(
    THREAD,
    tab.mod.awaitThreadScopedSettingsWrite,
    tab.mod.getThreadSettingsEditVersion,
    tab.mod.THREAD_SETTINGS_REVISION_KEY,
    read,
    tab.state().applyThreadScopedSettings,
    sanitizeThreadScopedSettings,
  );
}

test("refresh guards local edits during the preceding PATCH wait", async () => {
  threadRows.reset();
  threadRows.rows.set(THREAD, {
    systemPrompt: "before",
    temperature: 0.5,
    topP: 0.9,
  });
  const tab = await start("before-wait");
  const aCommitted = deferred();
  const aAck = deferred();
  threadRows.afterWrite = () => {
    aCommitted.resolve();
    return aAck.promise;
  };
  params(tab, { temperature: 1.1 });
  const a = tab.mod.awaitThreadScopedSettingsWrite(THREAD);
  await aCommitted.promise;
  threadRows.rows.set(THREAD, { ...threadRows.rows.get(THREAD), topP: 0.75 });
  remoteSync(tab, () => ({
    thread: { settings: threadRows.rows.get(THREAD) },
    cacheable: true,
  }))();
  params(tab, { systemPrompt: "local edit" });
  aAck.resolve();
  await a;
  await PAINT();
  await tab.mod.awaitThreadScopedSettingsWrite(THREAD);
  assert.equal(
    threadRows.rows.get(THREAD)?.systemPrompt,
    "local edit",
    "local prompt persisted",
  );
  assert.equal(
    tab.state().params.systemPrompt,
    "local edit",
    "local prompt remained live",
  );
});

test("remote refresh retries after an intervening local edit", async () => {
  threadRows.reset();
  threadRows.rows.set(THREAD, { systemPrompt: "before", temperature: 0.5 });
  const tab = await start("during-read");
  threadRows.rows.set(THREAD, {
    ...threadRows.rows.get(THREAD),
    systemPrompt: "remote edit",
  });
  const readStarted = deferred();
  const readResponse = deferred();
  let reads = 0;
  const dispatch = remoteSync(tab, async () => {
    const settings = threadRows.rows.get(THREAD);
    if (++reads === 1) {
      readStarted.resolve();
      await readResponse.promise;
    }
    return { thread: { settings }, cacheable: true };
  });
  dispatch();
  await readStarted.promise;
  params(tab, { temperature: 1.1 });
  readResponse.resolve();
  await PAINT();
  await tab.mod.awaitThreadScopedSettingsWrite(THREAD);
  await PAINT();
  assert.equal(threadRows.rows.get(THREAD)?.systemPrompt, "remote edit");
  assert.equal(
    tab.state().params.systemPrompt,
    "remote edit",
    "remote prompt applied",
  );
  assert.equal(tab.state().params.temperature, 1.1, "local temp remained live");
});

test("queued temperature write does not retransmit acknowledged prompt", async () => {
  threadRows.reset();
  threadRows.rows.set(THREAD, { systemPrompt: "before", temperature: 0.5 });
  const tab = await start("local-two-writes");
  const other = await start("other-write");
  const aCommitted = deferred();
  const aAck = deferred();
  threadRows.afterWrite = () => {
    aCommitted.resolve();
    return aAck.promise;
  };
  params(tab, { systemPrompt: "first local prompt" });
  const a = tab.mod.awaitThreadScopedSettingsWrite(THREAD);
  await aCommitted.promise;
  params(tab, { temperature: 1.1 });
  const b = tab.mod.awaitThreadScopedSettingsWrite(THREAD);
  params(other, { systemPrompt: "more recent remote prompt" });
  await other.mod.awaitThreadScopedSettingsWrite(THREAD);
  assert.equal(
    threadRows.rows.get(THREAD)?.systemPrompt,
    "more recent remote prompt",
    "other tab succeeded",
  );
  aAck.resolve();
  await Promise.all([a, b]);
  assert.deepEqual(threadRows.writesFor(THREAD).at(-1)?.settingsPatch, {
    temperature: 1.1,
  });
  assert.equal(
    threadRows.rows.get(THREAD)?.systemPrompt,
    "more recent remote prompt",
  );
});

test("a remote revocation drops Full access established in the still-open tab", async () => {
  threadRows.reset();
  threadRows.rows.set(THREAD, { permissionMode: "auto", fullAccess: false });
  const tab = await start("retained-full");
  tab.state().grantChatFullAccess({
    runId: "grant",
    threadId: THREAD,
    sessionId: "session",
  });
  await tab.mod.awaitThreadScopedSettingsWrite(THREAD);
  assert.equal(tab.state().permissionMode, "full");
  const other = await start("revoke-full");
  assert.equal(other.state().permissionMode, "full");
  other.state().setPermissionMode("auto");
  await other.mod.awaitThreadScopedSettingsWrite(THREAD);
  assert.equal(threadRows.rows.get(THREAD)?.fullAccess, false);
  remoteSync(tab, () => ({
    thread: { settings: threadRows.rows.get(THREAD) },
    cacheable: true,
  }))();
  await PAINT();
  assert.equal(tab.state().permissionMode, "auto");
});

test("a fresh row retires failed edits before the next unrelated write", async () => {
  threadRows.reset();
  threadRows.rows.set(THREAD, { systemPrompt: "before", temperature: 0.5 });
  const tab = await start("failed-write-then-refresh");
  threadRows.failNext = true;
  params(tab, { systemPrompt: "failed local edit" });
  assert.equal(await tab.mod.awaitThreadScopedSettingsWrite(THREAD), false);

  threadRows.rows.set(THREAD, {
    systemPrompt: "saved remotely",
    temperature: 0.5,
  });
  remoteSync(tab, () => ({
    thread: { settings: threadRows.rows.get(THREAD) },
    cacheable: true,
  }))();
  await PAINT();
  assert.equal(tab.state().params.systemPrompt, "saved remotely");

  params(tab, { temperature: 1.1 });
  assert.equal(await tab.mod.awaitThreadScopedSettingsWrite(THREAD), true);
  assert.deepEqual(threadRows.writesFor(THREAD).at(-1)?.settingsPatch, {
    temperature: 1.1,
  });
  assert.equal(threadRows.rows.get(THREAD)?.systemPrompt, "saved remotely");
});

function mountUnpaired(
  tab: Awaited<ReturnType<typeof start>>,
  read: () => unknown,
) {
  const start = provider.indexOf(
    "    const { applyThreadScopedSettings }",
    provider.indexOf("function ThreadScopedSettingsSync"),
  );
  const end = provider.indexOf(
    "  }, [activeThreadId, enabled, pendingNewThreadId, settingsHydrated]);",
    start,
  );
  const source = provider.slice(start, end);
  const listeners = new Map<string, (event: { key: string }) => void>();
  const window = {
    addEventListener: (key: string, value: (event: { key: string }) => void) =>
      listeners.set(key, value),
    removeEventListener: (key: string) => listeners.delete(key),
  };
  const cleanup = new Function(
    "deps",
    "return " +
      stripTypeScriptTypes(`(function() {
    const { activeThreadId, useChatRuntimeStore, beginThreadScopedPairing, awaitThreadScopedSettingsWrite,
      releaseHeldThreadScopedEdits, commitHeldThreadScopedEditsToTheirThread,
      getThreadSettingsEditVersion, THREAD_SETTINGS_REVISION_KEY, getStoredChatThreadReadResult, sanitizeThreadScopedSettings,
      window } = deps;
    const [pendingNewThreadId, enabled, settingsHydrated] = [null, true, true];
    const [THREAD_READ_TIMEOUT_MS, THREAD_READ_RETRIES] = [8000, 2];
    const awaitStoredChatThreadWrites = async () => {};
    const setTimeout = () => 0;
    const CHAT_HISTORY_UPDATED_EVENT = "history";
    const forgetThreadReads = () => {};
    ${source}
  })()`),
  )({
    ...tab.mod,
    activeThreadId: THREAD,
    getStoredChatThreadReadResult: read,
    sanitizeThreadScopedSettings,
    window,
  });
  return {
    dispatch: () =>
      listeners.get("storage")!({ key: tab.mod.THREAD_SETTINGS_REVISION_KEY }),
    cleanup,
  };
}

test("a remote change during the first read cannot be lost to its stale result", async () => {
  threadRows.reset();
  threadRows.rows.set(THREAD, { systemPrompt: "before", temperature: 0.5 });
  const tab = await start("during-first-read");
  tab.state().applyThreadScopedSettings(null, null);
  const firstStarted = deferred();
  const firstResponse = deferred();
  const secondResponse = deferred();
  let reads = 0;
  const { dispatch, cleanup } = mountUnpaired(tab, async () => {
    const settings = threadRows.rows.get(THREAD);
    if (++reads === 1) {
      firstStarted.resolve();
      await firstResponse.promise;
    } else {
      await secondResponse.promise;
    }
    return { thread: { settings }, cacheable: true };
  });
  await firstStarted.promise;
  threadRows.rows.set(THREAD, {
    ...threadRows.rows.get(THREAD),
    systemPrompt: "remote edit",
  });
  dispatch();
  await PAINT();
  firstResponse.resolve();
  await PAINT();
  secondResponse.resolve();
  await PAINT();
  assert.equal(
    tab.state().params.systemPrompt,
    "remote edit",
    "new remote prompt applied during first pairing",
  );
  cleanup();
});

test("tab-close keepalive preserves an edit that fits once within the 64 KiB browser budget", async () => {
  threadRows.reset();
  threadRows.rows.set(THREAD, { systemPrompt: "before", temperature: 0.5 });
  const tab = await start("beacon-budget");
  const prompt = "x".repeat(40 * 1024);
  params(tab, { systemPrompt: prompt });
  browser.fireWindowEvent("pagehide", {});
  const raw = browser.store.get("unsloth_chat_thread_settings_replay");
  const body = JSON.parse(raw!)[THREAD];
  assert.equal(body.settingsPatch.systemPrompt, prompt);
  assert.ok(
    Buffer.byteLength(JSON.stringify(body)) <= 64 * 1024,
    `pagehide sent ${Buffer.byteLength(JSON.stringify(body))} bytes for a ${prompt.length}-byte prompt`,
  );
});

test("an acknowledged write signals other tabs without a secure context", async () => {
  threadRows.reset();
  threadRows.rows.set(THREAD, { systemPrompt: "before", temperature: 0.5 });
  const tab = await start("insecure-context");
  const original = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  Object.defineProperty(globalThis, "crypto", { value: {}, configurable: true });
  try {
    browser.store.delete("unsloth_chat_thread_settings_revision");
    params(tab, { temperature: 1.1 });
    await tab.mod.awaitThreadScopedSettingsWrite(THREAD);
    assert.ok(browser.store.get("unsloth_chat_thread_settings_revision"));
  } finally {
    if (original) Object.defineProperty(globalThis, "crypto", original);
  }
});
