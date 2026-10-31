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
type Runtime = typeof import("../src/features/chat/stores/chat-runtime-store.ts");

async function boot(name: string) {
  settingsHttp.settings = {};
  settingsHttp.puts.length = 0;
  const module = await import(`${STORE}?scenario=${name}`) as Runtime;
  const state = () => module.useChatRuntimeStore.getState();
  state().setPermissionMode("auto");
  return { ...module, state };
}

const grant = { runId: "run-a", threadId: "chat-a", sessionId: "project-shared" };

test("a confirmed shortcut affects its chat, survives revisit, and never persists Full access", async () => {
  const { state } = await boot("chat-full-scope");
  state().setActiveThreadId("chat-a");
  state().applyThreadScopedSettings("chat-a", { permissionMode: "ask" });
  const before = new Map(browser.store);
  const writes = settingsHttp.puts.length;
  state().grantChatFullAccess(grant);
  assert.equal(state().permissionMode, "full");
  assert.equal(state().bypassPermissions, true);
  assert.equal(state().confirmToolCalls, false);
  assert.equal(state().chatFullAccessPreviousMode, "ask");
  assert.deepEqual(browser.store, before);
  assert.equal(settingsHttp.puts.length, writes);

  state().setActiveThreadId("chat-b");
  state().applyThreadScopedSettings("chat-b", { permissionMode: "off" });
  assert.equal(state().permissionMode, "off");
  assert.equal(state().bypassPermissions, false);
  state().setActiveThreadId("chat-a");
  state().applyThreadScopedSettings("chat-a", { permissionMode: "ask" });
  assert.equal(state().permissionMode, "full");
  state().setPermissionMode("ask");
  assert.equal(state().bypassPermissions, false);
  assert.equal(Object.keys(state().chatFullAccessGrants).length, 0);
  state().setActiveThreadId("chat-b");
  state().setActiveThreadId("chat-a");
  assert.equal(state().permissionMode, "ask");
});

test("an acknowledgment after navigation grants the original chat only", async () => {
  const { state } = await boot("chat-full-late-ack");
  state().setActiveThreadId("chat-b");
  state().grantChatFullAccess(grant);
  assert.equal(state().permissionMode, "auto");
  state().setActiveThreadId("chat-a");
  assert.equal(state().permissionMode, "full");
});

test("parallel unsaved runs are bound separately when their first turns are adopted", async () => {
  const { state } = await boot("chat-full-adoption");
  state().grantChatFullAccess({ ...grant, threadId: "" });
  state().grantChatFullAccess({ ...grant, threadId: "", runId: "run-b" });
  assert.equal(state().permissionMode, "auto", "an unknown thread cannot grant every new chat");
  state().adoptChatFullAccessRun("run-a", "chat-a");
  assert.equal(state().chatFullAccessGrants["run-a"]!.threadId, "chat-a");
  assert.equal(state().chatFullAccessGrants["run-b"]!.threadId, "");
  state().setActiveThreadId("chat-a");
  assert.equal(state().permissionMode, "full");
  state().clearUnadoptedChatFullAccessRun("run-a");
  assert.ok(state().chatFullAccessGrants["run-a"], "completed saved chats retain the grant for future turns");
  state().clearUnadoptedChatFullAccessRun("run-b");
  assert.equal(state().chatFullAccessGrants["run-b"], undefined);
  state().setActiveThreadId(null);
  assert.equal(state().permissionMode, "auto");
});

test("lowering an unsaved chat revokes its own active run only", async () => {
  const { state } = await boot("chat-full-revoke-unbound");
  state().grantChatFullAccess({ ...grant, threadId: "", runId: "background-run" });
  state().grantChatFullAccess({ ...grant, threadId: "", runId: "visible-run" });
  state().setPermissionMode("ask");
  assert.ok(state().chatFullAccessGrants["background-run"]);
  assert.equal(state().chatFullAccessGrants["visible-run"], undefined);
});

test("authentication changes and a reload discard ephemeral chat grants", async () => {
  const { state } = await boot("chat-full-account");
  state().setActiveThreadId("chat-a");
  state().grantChatFullAccess(grant);
  browser.fireWindowEvent("unsloth:auth-session-cleared", {});
  assert.deepEqual(state().chatFullAccessGrants, {});
  assert.notEqual(state().permissionMode, "full");
  const reload = await boot("chat-full-after-reload");
  assert.deepEqual(reload.state().chatFullAccessGrants, {});
  assert.notEqual(reload.state().permissionMode, "full");
});

test("the existing dropdown Full access remains session-wide", async () => {
  const { state } = await boot("chat-full-global-dropdown");
  state().setPermissionMode("full");
  state().setActiveThreadId("chat-a");
  state().setActiveThreadId("chat-b");
  assert.equal(state().permissionMode, "full");
  assert.equal(state().chatFullAccessPreviousMode, null);
});


test("an acknowledgment after first-turn adoption still binds to the original chat", async () => {
  const { state } = await boot("chat-full-late-adoption-ack");
  state().adoptChatFullAccessRun("run-a", "chat-a");
  state().setActiveThreadId("chat-b");
  state().grantChatFullAccess({ ...grant, threadId: "" });
  assert.equal(state().chatFullAccessGrants["run-a"]!.threadId, "chat-a");
  assert.equal(state().permissionMode, "auto");
  state().setActiveThreadId("chat-a");
  assert.equal(state().permissionMode, "full");
});


test("only explicit permission changes advance the version used by pending upgrades", async () => {
  const { state } = await boot("chat-full-choice-version");
  const first = state().chatPermissionChoiceVersion;
  state().setActiveThreadId("chat-a");
  state().grantChatFullAccess(grant);
  state().adoptChatFullAccessRun("run-a", "chat-a");
  assert.equal(state().chatPermissionChoiceVersion, first);
  state().setPermissionMode("off");
  assert.equal(state().chatPermissionChoiceVersion, first + 1);
  state().setBypassPermissions(false);
  assert.equal(state().chatPermissionChoiceVersion, first + 2);
});

test("a lower mode chosen before first-turn adoption is not undone by the adoption", async () => {
  const { state } = await boot("chat-full-revoke-before-adopt");
  state().grantChatFullAccess({ runId: "run-new", threadId: "", sessionId: "project-shared" });
  assert.ok(state().chatFullAccessGrants["run-new"]);
  state().setActiveThreadId("chat-a");
  state().setPermissionMode("off");
  state().adoptChatFullAccessRun("run-new", "chat-a");
  assert.equal(state().permissionMode, "off");
  assert.equal(state().bypassPermissions, false);
  assert.equal(Object.keys(state().chatFullAccessGrants).length, 0);
});

test("runs that never used the shortcut do not churn queued settings", async () => {
  const { state } = await boot("chat-full-run-cleanup");
  state().setActiveThreadId("chat-a");
  const epoch = state().queuedSettingsEpoch;
  for (let i = 0; i < 3; i++) {
    state().adoptChatFullAccessRun(`run-${i}`, "chat-a");
    state().clearUnadoptedChatFullAccessRun(`run-${i}`);
  }
  assert.equal(state().queuedSettingsEpoch, epoch);
  assert.deepEqual(state().chatFullAccessGrants, {});
});

test("a first-turn upgrade acknowledged after its run ended still binds to the adopted chat", async () => {
  const { state } = await boot("chat-full-late-ack-after-run");
  state().adoptChatFullAccessRun("run-late", "chat-a");
  state().clearUnadoptedChatFullAccessRun("run-late");
  state().grantChatFullAccess({ runId: "run-late", threadId: "", sessionId: "project-shared" });
  assert.equal(state().chatFullAccessGrants["run-late"]!.threadId, "chat-a");
  state().setActiveThreadId("chat-a");
  assert.equal(state().permissionMode, "full");
});
