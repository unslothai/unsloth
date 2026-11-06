// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

const store = new Map<string, string>();
const localStorage = {
  getItem: (key: string) => store.get(key) ?? null,
  setItem: (key: string, value: string) => void store.set(key, value),
  removeItem: (key: string) => void store.delete(key),
};
Object.assign(globalThis, {
  localStorage,
  window: { localStorage, location: { protocol: "http:", host: "127.0.0.1", hostname: "127.0.0.1", pathname: "/", href: "" } },
});

const { markPending, readPending, writePending } = await import(
  "../src/features/settings/lib/client-prefs-pending.ts"
);

const none = {
  scale: false,
  browser: false,
  chat: false,
  plusMenu: false,
  timeFormat: false,
  spellcheck: false,
};

test("an unconfirmed preference edit survives a reload until the server takes it", () => {
  assert.deepEqual(readPending(), none);
  writePending({ ...none, chat: true });
  assert.deepEqual(readPending(), { ...none, chat: true });
  writePending(none);
  assert.equal(store.size, 0);
  assert.deepEqual(readPending(), none);
});

test("a corrupt marker reads as nothing pending", () => {
  writePending({ ...none, scale: true, browser: true });
  const [key] = [...store.keys()];
  store.set(key, "{");
  assert.deepEqual(readPending(), none);
});

test("one tab's pending flags survive another tab marking or clearing its own", () => {
  writePending(none);
  markPending(["chat"], true);
  markPending(["scale"], true);
  assert.deepEqual(readPending(), { ...none, chat: true, scale: true });
  markPending(["scale"], false);
  assert.deepEqual(readPending(), { ...none, chat: true });
  markPending(["chat"], false);
  assert.equal(store.size, 0);
});
