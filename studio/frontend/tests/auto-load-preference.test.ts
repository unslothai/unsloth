// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  CHAT_AUTO_LOAD_MODEL_KEY,
  DEFAULT_CHAT_AUTO_LOAD_MODEL,
  readAutoLoadModelPreference,
  saveAutoLoadModelPreference,
} from "../src/features/chat/utils/auto-load-preference.ts";

function withStorage(run: (storage: Map<string, string>) => void): void {
  const storage = new Map<string, string>();
  const previousWindow = (globalThis as { window?: unknown }).window;
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      localStorage: {
        getItem: (key: string) => storage.get(key) ?? null,
        setItem: (key: string, value: string) => storage.set(key, value),
      },
    },
  });
  try {
    run(storage);
  } finally {
    if (previousWindow === undefined) {
      Object.defineProperty(globalThis, "window", {
        configurable: true,
        value: undefined,
      });
    } else {
      Object.defineProperty(globalThis, "window", {
        configurable: true,
        value: previousWindow,
      });
    }
  }
}

test("auto-load preference defaults on and persists an explicit opt-out", () => {
  withStorage((storage) => {
    assert.equal(readAutoLoadModelPreference(), DEFAULT_CHAT_AUTO_LOAD_MODEL);
    saveAutoLoadModelPreference(false);
    assert.equal(storage.get(CHAT_AUTO_LOAD_MODEL_KEY), "false");
    assert.equal(readAutoLoadModelPreference(), false);
    saveAutoLoadModelPreference(true);
    assert.equal(readAutoLoadModelPreference(), true);
  });
});

test("malformed preference values fall back to the safe default", () => {
  withStorage((storage) => {
    storage.set(CHAT_AUTO_LOAD_MODEL_KEY, "not-a-boolean");
    assert.equal(readAutoLoadModelPreference(), DEFAULT_CHAT_AUTO_LOAD_MODEL);
  });
});
