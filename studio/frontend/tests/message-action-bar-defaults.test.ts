// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  readSrc,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();
const { useChatPreferencesStore } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);

const KEY = "unsloth_chat_preferences";
const THREAD = readSrc("components/assistant-ui/thread.tsx");

function component(name: string): string {
  const start = THREAD.indexOf(`const ${name}: FC = () => {`);
  assert.notEqual(start, -1, `${name} is gone; this test needs rewriting`);
  return THREAD.slice(start, THREAD.indexOf("\n};", start));
}

test("Edit response is inline by default", () => {
  assert.equal(
    useChatPreferencesStore.getInitialState().showInlineEditResponse,
    true,
  );
});

test("a pre-version record carrying the old off default is moved to on", async () => {
  store.set(
    KEY,
    JSON.stringify({ state: { showInlineEditResponse: false }, version: 0 }),
  );
  await useChatPreferencesStore.persist.rehydrate();
  assert.equal(useChatPreferencesStore.getState().showInlineEditResponse, true);
});

test("an off choice saved after the migration is kept", async () => {
  store.set(
    KEY,
    JSON.stringify({ state: { showInlineEditResponse: false }, version: 1 }),
  );
  await useChatPreferencesStore.persist.rehydrate();
  assert.equal(
    useChatPreferencesStore.getState().showInlineEditResponse,
    false,
  );
});

test("a reply's bar carries Delete; its More menu does not", () => {
  const bar = component("AssistantActionBar");
  const menuAt = bar.indexOf("<ActionBarMorePrimitive.Root");
  assert.ok(bar.slice(0, menuAt).includes("<DeleteMessageButton />"));
  assert.ok(!bar.slice(menuAt).includes("Delete"));
});

test("a prompt's bar carries Fork and Delete; its More menu keeps Rewind only", () => {
  const bar = component("UserActionBar");
  assert.match(
    bar,
    /<ForkMessageButton \/>\s*<DeleteMessageButton \/>\s*<UserMoreMenu \/>/,
  );
  const menu = component("UserMoreMenu");
  assert.ok(menu.includes("<RewindMessageMenuItem />"));
  assert.ok(!/Fork|Delete/.test(menu));
});
