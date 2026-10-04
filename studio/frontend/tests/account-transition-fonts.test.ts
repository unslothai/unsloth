// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  BROWSER_ACCOUNT_KEY,
  transitionBrowserAccount,
} from "../src/lib/account-transition.ts";

const APPEARANCE_KEY = "unsloth_appearance_customization";
const ALICE_FONT_BYTES = "data:font/woff2;base64,QUxJQ0VTRUNSRVQ=";

function appearanceValue(): string {
  return JSON.stringify({
    state: {
      customization: {
        uiFont: "Alice Sans",
        headingFont: null,
        chatFont: null,
        codeFont: null,
        importedFonts: [{ name: "Alice Sans", dataUrl: ALICE_FONT_BYTES }],
        contrast: 40,
      },
    },
    version: 7,
  });
}

function browserWith(values: Record<string, string>) {
  const data = new Map(Object.entries(values));
  const storage = {
    get length() {
      return data.size;
    },
    key: (index: number) => [...data.keys()][index] ?? null,
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value);
    },
    removeItem: (key: string) => {
      data.delete(key);
    },
  } as Storage;
  const browser = {
    localStorage: storage,
    sessionStorage: { length: 0, key: () => null, removeItem: () => {} },
    indexedDB: {
      deleteDatabase: () => {
        const request = {} as IDBOpenDBRequest;
        queueMicrotask(() =>
          request.onsuccess?.call(request, {} as IDBVersionChangeEvent),
        );
        return request;
      },
    },
    location: { replace: () => {} },
  } as unknown as Window;
  return { browser, data };
}

test("an account switch drops the previous account's uploaded font bytes", async () => {
  const b = browserWith({
    [BROWSER_ACCOUNT_KEY]: "alice",
    [APPEARANCE_KEY]: appearanceValue(),
  });
  assert.equal(
    await transitionBrowserAccount("bob", "/chat", () => {}, b.browser),
    true,
  );
  const persisted = b.data.get(APPEARANCE_KEY) ?? "";
  assert.equal(
    persisted.includes(ALICE_FONT_BYTES),
    false,
    "Alice's uploaded font bytes are still readable by Bob's session",
  );
  const parsed = JSON.parse(persisted);
  assert.deepEqual(parsed.state.customization.importedFonts, []);
  // A selection naming a purged font would render as a missing family.
  assert.equal(parsed.state.customization.uiFont, null);
  // Real chrome still survives the switch.
  assert.equal(parsed.state.customization.contrast, 40);
});

test("appearance chrome that carries no fonts is left byte-identical", async () => {
  const plain = JSON.stringify({
    state: { customization: { contrast: 40, uiFont: null } },
    version: 7,
  });
  const b = browserWith({
    [BROWSER_ACCOUNT_KEY]: "alice",
    [APPEARANCE_KEY]: plain,
    theme: "dark",
  });
  await transitionBrowserAccount("bob", "/chat", () => {}, b.browser);
  assert.equal(b.data.get(APPEARANCE_KEY), plain);
  assert.equal(b.data.get("theme"), "dark");
});
