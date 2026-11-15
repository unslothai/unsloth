// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

const browser = installLocalStorageFake();
browser.store.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);

const { threadRows } = await import("./helpers/store-stubs/chat-history-storage.ts");
const { setLoginMode } = await import("../src/features/auth/login-client.ts");
const STORE = new URL("../src/features/chat/stores/chat-runtime-store.ts", import.meta.url).href;
const { sanitizeThreadScopedSettings } = await import("../src/features/chat/utils/thread-scoped-settings.ts");
type Runtime = typeof import("../src/features/chat/stores/chat-runtime-store.ts");

const CHAT = "chat-with-full-access";
const OTHER = "chat-without-full-access";

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 900));
}

async function start(name: string) {
  const module = (await import(`${STORE}?scenario=${name}`)) as Runtime;
  const state = () => module.useChatRuntimeStore.getState();
  state().setPermissionMode("auto");
  return Object.assign(state, { module });
}

type State = Awaited<ReturnType<typeof start>>;

/** As runtime-provider opens a chat: its stored row, through the same sanitizer. */
function open(state: State, threadId: string): void {
  state().setActiveThreadId(threadId);
  state().applyThreadScopedSettings(threadId, sanitizeThreadScopedSettings(threadRows.rows.get(threadId) ?? {}));
}

function logOutAndBackIn(): void {
  browser.fireWindowEvent("unsloth:auth-session-cleared", {});
  browser.fireWindowEvent("unsloth:auth-session-stored", {});
}

async function grantChat(state: State, threadId: string): Promise<void> {
  open(state, threadId);
  state().grantChatFullAccess({ runId: `run-${threadId}`, threadId, sessionId: "session-1" });
  await settle();
}

test("a chat granted Full access keeps it after logging out and back in, and after a restart", async () => {
  threadRows.reset();
  threadRows.rows.set(OTHER, { permissionMode: "ask" });
  const state = await start("saved-grant");
  await grantChat(state, CHAT);
  assert.equal(state().permissionMode, "full");
  assert.equal(threadRows.rows.get(CHAT)?.fullAccess, true, "saved with the chat");

  logOutAndBackIn();
  open(state, CHAT);
  assert.equal(state().permissionMode, "full", "same chat after logging back in");

  open(state, OTHER);
  assert.equal(state().permissionMode, "ask", "another chat keeps its own level");
  assert.equal(state().bypassPermissions, false);
  open(state, CHAT);
  assert.equal(state().permissionMode, "full");

  const restarted = await start("saved-grant-restart");
  open(restarted, CHAT);
  assert.equal(restarted().permissionMode, "full", "same chat after a restart");
  assert.equal(restarted().bypassPermissions, true);
});

test("leaving Full access in that chat is saved too", async () => {
  threadRows.reset();
  const state = await start("saved-grant-leave");
  await grantChat(state, CHAT);
  state().setPermissionMode("ask");
  await settle();
  assert.equal(threadRows.rows.get(CHAT)?.fullAccess, false);
  assert.equal(threadRows.rows.get(CHAT)?.permissionMode, "ask");

  const restarted = await start("saved-grant-leave-restart");
  open(restarted, CHAT);
  assert.equal(restarted().permissionMode, "ask");
  assert.equal(restarted().bypassPermissions, false);
});

test("a saved chat Full access is not restored on an install with other accounts", async () => {
  threadRows.reset();
  threadRows.rows.set(CHAT, { permissionMode: "auto", fullAccess: true });
  try {
    setLoginMode("multi");
    const state = await start("saved-grant-multi");
    open(state, CHAT);
    assert.equal(state().permissionMode, "auto");
    assert.equal(state().bypassPermissions, false);
    await settle();
    assert.equal(threadRows.rows.get(CHAT)?.fullAccess, false, "the saved copy is cleared");
  } finally {
    setLoginMode("single", true);
  }
});

test("leaving Full access while the chat's settings load is not overridden by the saved flag", async () => {
  const exits: Array<[string, (state: State) => void]> = [
    ["pick-ask", (state) => state().setPermissionMode("ask")],
    ["deep-research", (state) => state().setDeepResearchEnabled(true)],
  ];
  for (const [name, leave] of exits) {
    threadRows.reset();
    threadRows.rows.set(CHAT, { permissionMode: "auto", fullAccess: true });
    const state = await start(`saved-grant-leave-during-read-${name}`);
    state().setActiveThreadId(CHAT);
    state().applyThreadScopedSettings(null, null);
    state.module.beginThreadScopedPairing(CHAT);
    leave(state);
    state().applyThreadScopedSettings(CHAT, sanitizeThreadScopedSettings(threadRows.rows.get(CHAT)));
    assert.notEqual(state().permissionMode, "full", name);
    assert.equal(state().bypassPermissions, false, name);
    await settle();
    assert.notEqual(state().permissionMode, "full", name);
    assert.equal(threadRows.rows.get(CHAT)?.fullAccess, false, name);
  }
});

test("reopening a chat whose saved Full access was revoked elsewhere drops the restored grant", async () => {
  threadRows.reset();
  threadRows.rows.set(CHAT, { permissionMode: "auto", fullAccess: true });
  const state = await start("saved-grant-revoked-elsewhere");
  open(state, CHAT);
  assert.equal(state().permissionMode, "full");
  threadRows.rows.set(CHAT, { permissionMode: "auto", fullAccess: false });
  open(state, OTHER);
  open(state, CHAT);
  assert.equal(state().permissionMode, "auto");
  assert.equal(state().bypassPermissions, false);
});
