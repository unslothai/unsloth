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

const { readPending, writePending } = await import(
  "../src/features/settings/lib/client-prefs-pending.ts"
);

test("an unconfirmed preference edit survives a reload until the server takes it", () => {
  assert.deepEqual(readPending(), { scale: false, browser: false });
  writePending({ scale: true, browser: false });
  assert.deepEqual(readPending(), { scale: true, browser: false });
  writePending({ scale: false, browser: false });
  assert.equal(store.size, 0);
  assert.deepEqual(readPending(), { scale: false, browser: false });
});

test("a corrupt marker reads as nothing pending", () => {
  writePending({ scale: true, browser: true });
  const [key] = [...store.keys()];
  store.set(key, "{");
  assert.deepEqual(readPending(), { scale: false, browser: false });
});

test("preference saves run one at a time and send only the sections edited here", async () => {
  const { readSrc } = await import("./helpers/kit.ts");
  const hook = readSrc("features/settings/hooks/use-device-settings-sync.ts");
  assert.match(hook, /saving = saving\n\s*\.then\(async \(\) => \{\n\s*const sent = edits;/);
  assert.match(hook, /interfaceScale: touched\.scale\n\s*\? useInterfaceScaleStore\.getState\(\)\.scale\n\s*: undefined,/);
  assert.match(hook, /browser: touched\.browser \? browserPrefs\(\) : undefined,/);
});
