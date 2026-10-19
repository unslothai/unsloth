// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

import {
  installLocalStorageFake,
  readSrc,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store, fireWindowEvent } = installLocalStorageFake();

// Just enough <html> to read back what the preference wrote.
const attrs = new Map<string, string>();
Object.assign(globalThis.document, {
  documentElement: {
    getAttribute: (name: string) => attrs.get(name) ?? null,
    setAttribute: (name: string, value: string) => attrs.set(name, value),
    removeAttribute: (name: string) => attrs.delete(name),
  },
});
const root = () => attrs.get("spellcheck") ?? null;

const {
  SPELLCHECK_STORAGE_KEY,
  getSpellCheck,
  setSpellCheck,
  watchSpellCheck,
} = await import("../src/lib/spellcheck.ts");

test("an untouched install leaves spell check to the engine", () => {
  watchSpellCheck(globalThis.window);
  assert.equal(getSpellCheck(), true);
  assert.equal(root(), null);
});

test("turning it off marks <html> and persists", () => {
  setSpellCheck(false);
  assert.equal(getSpellCheck(), false);
  assert.equal(root(), "false");
  assert.equal(store.get(SPELLCHECK_STORAGE_KEY), "false");
});

test("turning it back on removes the attribute and the stored value", () => {
  setSpellCheck(true);
  assert.equal(root(), null);
  assert.equal(store.has(SPELLCHECK_STORAGE_KEY), false);
});

test("a toggle in another tab is applied here", () => {
  store.set(SPELLCHECK_STORAGE_KEY, "false");
  fireWindowEvent("storage", { key: SPELLCHECK_STORAGE_KEY });
  assert.equal(root(), "false");
  store.delete(SPELLCHECK_STORAGE_KEY);
  fireWindowEvent("storage", { key: null }); // localStorage.clear()
  assert.equal(root(), null);
});

test("no field opts back in, so the root setting reaches every input", () => {
  // One spellCheck={true} would keep underlines in that field with the switch off.
  const src = new URL("../src/", import.meta.url);
  const optIns = readdirSync(src, { recursive: true, encoding: "utf8" })
    .filter((file) => /\.tsx?$/.test(file))
    .filter((file) =>
      /spell[cC]heck=(\{true\}|"true")/.test(
        readFileSync(new URL(file, src), "utf8"),
      ),
    );
  assert.deepEqual(optIns, []);
});

test("the app applies it before the first render", () => {
  assert.match(readSrc("main.tsx"), /watchSpellCheck\(window\);/);
});

test("an account switch keeps the choice, like the display language", async () => {
  const { ACCOUNT_CHROME_KEYS } = await import(
    "../src/lib/account-transition.ts"
  );
  assert.ok(ACCOUNT_CHROME_KEYS.has(SPELLCHECK_STORAGE_KEY));
});

test("Reset all preferences turns spell check back on", () => {
  const prefsKeys = readSrc("features/settings/tabs/general-tab.tsx").match(
    /const PREFS_KEYS: string\[\] = \[([\s\S]*?)\n\];/,
  )?.[1];
  assert.match(prefsKeys ?? "", /^\s*SPELLCHECK_STORAGE_KEY,$/m);
});
