// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import { installLocalStorageFake } from "./helpers/kit.ts";

const browser = installLocalStorageFake();
browser.store.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);
const STORE = new URL("../src/features/chat/stores/chat-runtime-store.ts", import.meta.url).href;
const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { setLoginMode } = await import("../src/features/auth/login-client.ts");
type Runtime = typeof import("../src/features/chat/stores/chat-runtime-store.ts");

const FLAG = "unsloth_chat_full_access";
const LEVEL = "unsloth_chat_permission_mode";

/** A fresh store module over the current fake storage, like a Studio restart. */
async function restart(name: string, server: Record<string, unknown> = {}) {
  settingsHttp.settings = server;
  settingsHttp.puts.length = 0;
  const module = await import(`${STORE}?scenario=${name}`) as Runtime;
  const state = () => module.useChatRuntimeStore.getState();
  return { ...module, state };
}

function resetStorage(level = "auto") {
  browser.store.delete(FLAG);
  browser.store.delete("unsloth_chat_full_access_code_off");
  browser.store.set(LEVEL, level);
}

function assertAppFullAccess(state: Awaited<ReturnType<typeof restart>>["state"]) {
  assert.equal(state().permissionMode, "full");
  assert.equal(state().bypassPermissions, true);
  assert.equal(state().confirmToolCalls, false);
  assert.equal(state().deepResearchEnabled, false);
  assert.equal(state().chatFullAccessPreviousMode, null);
}

test("Full access picked in settings is restored after a restart", async () => {
  resetStorage("off");
  const first = await restart("full-pick");
  first.state().setPermissionMode("full");
  assert.equal(browser.store.get(FLAG), "true");
  assert.equal(browser.store.get(LEVEL), "off", "the level to return to is kept");

  const second = await restart("full-pick-restart");
  assertAppFullAccess(second.state);

  second.state().setPermissionMode("off");
  assert.equal(second.state().confirmToolCalls, false);
  assert.equal(second.state().bypassPermissions, false);
});

test("Code switched off under Full access stays off after a restart", async () => {
  resetStorage();
  const first = await restart("code-declined");
  first.state().setPermissionMode("full");
  first.state().setCodeToolsEnabled(false);
  assert.equal(first.state().codeToolsDeclinedUnderFullAccess, true);

  const second = await restart("code-declined-restart");
  assert.equal(second.state().permissionMode, "full");
  assert.equal(second.state().codeToolsDeclinedUnderFullAccess, true);

  second.state().setCodeToolsEnabled(true);
  const third = await restart("code-enabled-restart");
  assert.equal(third.state().codeToolsDeclinedUnderFullAccess, false);
});

test("Code that Full access turned on comes back on after a restart", async () => {
  resetStorage();
  browser.store.set("unsloth_chat_code_tools_enabled", "false");
  const first = await restart("code-implicit");
  first.state().setPermissionMode("full");
  assert.equal(first.state().codeToolsDeclinedUnderFullAccess, false);
  const second = await restart("code-implicit-restart");
  assert.equal(second.state().permissionMode, "full");
  assert.equal(second.state().codeToolsDeclinedUnderFullAccess, false);
});

test("a server restore of Full access keeps Code off when its switch is off", async () => {
  resetStorage();
  browser.store.set("unsloth_chat_code_tools_enabled", "false");
  const { state } = await restart("server-code-off", { fullAccess: true, codeToolsEnabled: false });
  await state().hydratePersistedSettings();
  assert.equal(state().permissionMode, "full");
  assert.equal(state().codeToolsDeclinedUnderFullAccess, true);
});

test("ask, auto and off still restore as before and never turn on Full access", async () => {
  for (const level of ["ask", "auto", "off"] as const) {
    resetStorage();
    const first = await restart(`plain-${level}`);
    first.state().setPermissionMode(level);
    assert.equal(browser.store.get(FLAG), "false");
    const second = await restart(`plain-${level}-restart`);
    assert.equal(second.state().permissionMode, level);
    assert.equal(second.state().bypassPermissions, false);
    assert.equal(second.state().confirmToolCalls, level !== "off");
  }
});

test("leaving Full access by any route is remembered, even with a chat open", async () => {
  const exits: Array<[string, (state: Awaited<ReturnType<typeof restart>>["state"]) => void]> = [
    ["level-in-chat", (state) => {
      state().setActiveThreadId("chat-a");
      state().applyThreadScopedSettings("chat-a", {});
      state().setPermissionMode("ask");
    }],
    ["deep-research", (state) => state().setDeepResearchEnabled(true)],
    ["bypass-off", (state) => state().setBypassPermissions(false)],
  ];
  for (const [name, exit] of exits) {
    resetStorage();
    browser.store.set(FLAG, "true");
    const first = await restart(`exit-${name}`);
    assertAppFullAccess(first.state);
    exit(first.state);
    assert.notEqual(first.state().permissionMode, "full", name);
    assert.equal(browser.store.get(FLAG), "false", name);
    const second = await restart(`exit-${name}-restart`);
    assert.notEqual(second.state().permissionMode, "full", name);
    assert.equal(second.state().bypassPermissions, false, name);
  }
});

test("the legacy bypass switch saves Full access too", async () => {
  resetStorage();
  const first = await restart("bypass-on");
  first.state().setBypassPermissions(true);
  assert.equal(browser.store.get(FLAG), "true");
  const second = await restart("bypass-on-restart");
  assertAppFullAccess(second.state);
});

test("the server's saved Full access applies on hydration, and its false turns it off", async () => {
  resetStorage();
  const on = await restart("server-on", { fullAccess: true, deepResearchEnabled: true });
  assert.equal(on.state().permissionMode, "auto");
  await on.state().hydratePersistedSettings();
  assertAppFullAccess(on.state);

  resetStorage();
  browser.store.set(FLAG, "true");
  const off = await restart("server-off", { fullAccess: false, permissionMode: "ask" });
  assertAppFullAccess(off.state);
  await off.state().hydratePersistedSettings();
  assert.equal(off.state().permissionMode, "ask");
  assert.equal(off.state().bypassPermissions, false);
  assert.equal(off.state().confirmToolCalls, true);
});

test("a per-chat Full access grant is never saved", async () => {
  resetStorage();
  const { state } = await restart("chat-grant");
  state().setActiveThreadId("chat-a");
  state().grantChatFullAccess({ runId: "run-a", threadId: "chat-a", sessionId: "s" });
  assert.equal(state().permissionMode, "full");
  assert.equal(browser.store.has(FLAG), false);
});

test("an install with other accounts neither saves nor restores Full access", async () => {
  try {
    setLoginMode("multi");
    resetStorage();
    const first = await restart("multi-pick");
    first.state().setPermissionMode("full");
    assert.equal(first.state().permissionMode, "full", "still usable for the session");
    assert.equal(browser.store.has(FLAG), false);

    browser.store.set(FLAG, "true");
    const second = await restart("multi-restart", { fullAccess: true });
    assert.equal(second.state().permissionMode, "auto");
    await second.state().hydratePersistedSettings();
    assert.equal(second.state().permissionMode, "auto");
    assert.equal(browser.store.get(FLAG), "false", "the cached server copy is cleared too");
    await new Promise((resolve) => setTimeout(resolve, 600));
    assert.ok(
      settingsHttp.puts.some((put) => put.fullAccess === false),
      "the server copy is cleared",
    );
  } finally {
    setLoginMode("single", true);
  }
});
