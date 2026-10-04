// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";
import {
  drainMockedTimers,
  enableCountedTimers,
} from "./helpers/mock-timer-drain.ts";

const { store: localStorageFake } = installLocalStorageFake();
localStorageFake.set("unsloth_chat_settings_imported_to_studio_db", "true");
register("./thread-sampling-resolver.mjs", import.meta.url);

const { settingsHttp } = await import("./helpers/store-stubs/settings-http.ts");
const { threadRows } = await import(
  "./helpers/store-stubs/chat-history-storage.ts"
);

const STORE_URL = new URL(
  "../src/features/chat/stores/chat-runtime-store.ts",
  import.meta.url,
).href;
const CODE_KEY = "unsloth_chat_code_tools_enabled";
const LOCAL = "unsloth/Qwen3-1.7B-GGUF";
const EXTERNAL = "external::anthropic::claude-opus-5";

type Store = Record<string, (...args: never[]) => unknown> & {
  params: { checkpoint: string };
  codeToolsEnabled: boolean;
  permissionMode: string;
};

interface StoreModule {
  useChatRuntimeStore: {
    getState: () => Store;
    setState: (patch: Record<string, unknown>) => void;
  };
  codeToolsOn: (state: Store) => boolean;
  beginThreadScopedPairing: (threadId: string) => void;
  awaitStartedThreadScopedSettingsWrites: () => Promise<void>;
}

let scenario = 0;

/** A fresh store with a tool-capable model selected and Code stored as `codeOn`. */
async function freshStore(codeOn: boolean, checkpoint = LOCAL) {
  scenario += 1;
  localStorageFake.set(CODE_KEY, String(codeOn));
  settingsHttp.settings = { codeToolsEnabled: codeOn };
  settingsHttp.puts.length = 0;
  threadRows.reset();
  const mod: StoreModule = await import(
    `${STORE_URL}?scenario=fullaccess${scenario}`
  );
  const store = mod.useChatRuntimeStore;
  store.setState({
    params: { ...store.getState().params, checkpoint },
    supportsTools: true,
  });
  const on = () => mod.codeToolsOn(store.getState());
  return { mod, store, on, state: () => store.getState() };
}

function pick(state: Store, mode: string) {
  (state.setPermissionMode as (mode: string) => void)(mode);
}

function clickCode(state: Store, next: boolean) {
  (state.setCodeToolsEnabled as (enabled: boolean) => void)(next);
}

test("picking Full access turns Code on, and leaving it turns Code back off", async () => {
  const { on, state } = await freshStore(false);
  assert.equal(on(), false);
  pick(state(), "full");
  assert.equal(on(), true);
  // The user's own choice is untouched, so nothing is stored.
  assert.equal(state().codeToolsEnabled, false);
  assert.equal(localStorageFake.get(CODE_KEY), "false");
  for (const mode of ["auto", "ask", "off"]) {
    pick(state(), "full");
    assert.equal(on(), true);
    pick(state(), mode);
    assert.equal(on(), false, `leaving Full access for ${mode}`);
  }
});

test("the legacy bypass toggle behaves the same way", async () => {
  const { on, state } = await freshStore(false);
  (state().setBypassPermissions as (on: boolean) => void)(true);
  assert.equal(on(), true);
  (state().setBypassPermissions as (on: boolean) => void)(false);
  assert.equal(on(), false);
});

test("Code the user already had on stays on after Full access", async () => {
  const { on, state } = await freshStore(true);
  pick(state(), "full");
  assert.equal(on(), true);
  pick(state(), "auto");
  assert.equal(on(), true);
  assert.equal(localStorageFake.get(CODE_KEY), "true");
});

test("a Code click under Full access is the user's and survives leaving it", async () => {
  const off = await freshStore(false);
  pick(off.state(), "full");
  clickCode(off.state(), !off.on());
  assert.equal(off.on(), false, "the click turns the pill off");
  // Re-picking the level already on does not undo the click.
  pick(off.state(), "full");
  assert.equal(off.on(), false);
  pick(off.state(), "auto");
  assert.equal(off.on(), false);

  const offThenOn = await freshStore(false);
  pick(offThenOn.state(), "full");
  clickCode(offThenOn.state(), false);
  clickCode(offThenOn.state(), true);
  pick(offThenOn.state(), "auto");
  assert.equal(offThenOn.on(), true);
  assert.equal(localStorageFake.get(CODE_KEY), "true");
});

test("Full access does not turn on an external provider's code sandbox", async () => {
  const { on, state } = await freshStore(false, EXTERNAL);
  pick(state(), "full");
  assert.equal(on(), false);
});

test("the turn-on reaches neither the installation nor the open chat's snapshot", async (t) => {
  enableCountedTimers(t);
  const tick = (ms: number) => t.mock.timers.tick(ms);
  const { mod, on, state } = await freshStore(false);
  const drain = () =>
    drainMockedTimers(tick, {
      label: "full access drain",
      barrier: () => mod.awaitStartedThreadScopedSettingsWrites(),
    });
  await (state().hydratePersistedSettings as () => Promise<void>)();
  await drain();
  (state().setActiveThreadId as (id: string) => void)("A");
  mod.beginThreadScopedPairing("A");
  (state().applyThreadScopedSettings as (id: string, s: null) => void)(
    "A",
    null,
  );
  pick(state(), "full");
  assert.equal(on(), true);
  // Any per-chat edit writes the chat's whole snapshot, Code included.
  (state().setToolsEnabled as (on: boolean) => void)(true);
  await drain();
  const row = threadRows.rows.get("A") as Record<string, unknown> | undefined;
  assert.ok(row, "the edit wrote chat A's snapshot");
  assert.notEqual(row.codeToolsEnabled, true);
  assert.equal(localStorageFake.get(CODE_KEY), "false");
  for (const put of settingsHttp.puts) {
    assert.notEqual(put.codeToolsEnabled, true);
  }
});

test("a chat opened after Full access was picked still gets the code tools", async (t) => {
  // Full access outlives a chat switch but Code does not, so a grant armed once
  // on entry left the incoming chat on Full access with no code tools.
  enableCountedTimers(t);
  const tick = (ms: number) => t.mock.timers.tick(ms);
  // Code already on where Full access is picked: the case that arms nothing.
  const { mod, store, on, state } = await freshStore(true);
  const drain = () =>
    drainMockedTimers(tick, {
      label: "cross-chat drain",
      barrier: () => mod.awaitStartedThreadScopedSettingsWrites(),
    });
  await (state().hydratePersistedSettings as () => Promise<void>)();
  await drain();
  (state().setActiveThreadId as (id: string) => void)("A");
  mod.beginThreadScopedPairing("A");
  (state().applyThreadScopedSettings as (id: string, s: null) => void)("A", null);
  pick(state(), "full");
  assert.equal(on(), true);

  // An older chat whose own settings are Code off and the ordinary level.
  (state().setActiveThreadId as (id: string) => void)("B");
  mod.beginThreadScopedPairing("B");
  (
    state().applyThreadScopedSettings as (
      id: string,
      s: Record<string, unknown>,
    ) => void
  )("B", { codeToolsEnabled: false, permissionMode: "auto" });
  await drain();
  assert.equal(state().permissionMode, "full", "Full access survives the switch");
  assert.equal(store.getState().params.checkpoint, LOCAL);
  assert.equal(on(), true, "so the code tools come with it");
});

test("a Code click under Full access survives a chat switch too", async (t) => {
  enableCountedTimers(t);
  const tick = (ms: number) => t.mock.timers.tick(ms);
  const { mod, on, state } = await freshStore(false);
  const drain = () =>
    drainMockedTimers(tick, {
      label: "cross-chat decline drain",
      barrier: () => mod.awaitStartedThreadScopedSettingsWrites(),
    });
  await (state().hydratePersistedSettings as () => Promise<void>)();
  await drain();
  (state().setActiveThreadId as (id: string) => void)("A");
  mod.beginThreadScopedPairing("A");
  (state().applyThreadScopedSettings as (id: string, s: null) => void)("A", null);
  pick(state(), "full");
  clickCode(state(), false); // the user's own no
  assert.equal(on(), false);

  (state().setActiveThreadId as (id: string) => void)("B");
  mod.beginThreadScopedPairing("B");
  (
    state().applyThreadScopedSettings as (
      id: string,
      s: Record<string, unknown>,
    ) => void
  )("B", { codeToolsEnabled: false, permissionMode: "auto" });
  await drain();
  assert.equal(on(), false, "a switch does not undo it");
});
