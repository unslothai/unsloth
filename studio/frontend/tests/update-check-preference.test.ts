// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

const PREF_URL = new URL(
  "../src/hooks/use-unsloth-update-pref.ts",
  import.meta.url,
);

function withStorage(t: import("node:test").TestContext): void {
  const values = new Map<string, string>();
  const storage = {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => {
      values.set(key, value);
    },
    removeItem: (key: string) => {
      values.delete(key);
    },
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  };
  const original = Object.getOwnPropertyDescriptor(globalThis, "window");
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: { localStorage: storage },
  });
  t.after(() => {
    if (original) {
      Object.defineProperty(globalThis, "window", original);
    } else {
      Reflect.deleteProperty(globalThis, "window");
    }
  });
}

test("unsloth automatic update checks default to enabled", async (t) => {
  withStorage(t);
  const { getShowUnslothUpdateBanner } = await import(PREF_URL.href);
  assert.equal(getShowUnslothUpdateBanner(), true);
});

test("an explicit opt-out persists and can be reverted", async (t) => {
  withStorage(t);
  const { getShowUnslothUpdateBanner, setShowUnslothUpdateBanner } =
    await import(PREF_URL.href);

  setShowUnslothUpdateBanner(false);
  assert.equal(getShowUnslothUpdateBanner(), false);

  setShowUnslothUpdateBanner(true);
  assert.equal(getShowUnslothUpdateBanner(), true);
});
