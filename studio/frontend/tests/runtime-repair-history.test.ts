// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  recordRuntimeRepair,
  wasRuntimeRepairedRecently,
} from "../src/hooks/runtime-repair-history.ts";

test("a successful runtime repair is remembered across launches, then expires", () => {
  const values = new Map<string, string>();
  const original = globalThis.localStorage;
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => { values.set(key, value); },
    },
  });
  try {
    const now = 1_000_000_000;
    recordRuntimeRepair("backend_outdated", now);
    assert.equal(wasRuntimeRepairedRecently("llama_runtime_dir_missing", now), false);

    recordRuntimeRepair("llama_runtime_dir_missing", now);
    assert.equal(wasRuntimeRepairedRecently("llama_runtime_binaries_missing", now + 1000), true);
    assert.equal(wasRuntimeRepairedRecently("backend_outdated", now + 1000), false);
    assert.equal(wasRuntimeRepairedRecently("llama_runtime_dir_missing", now - 1), false);
    assert.equal(wasRuntimeRepairedRecently("llama_runtime_dir_missing", now + 8 * 86400_000), false);
  } finally {
    Object.defineProperty(globalThis, "localStorage", { configurable: true, value: original });
  }
});

test("blocked storage never prevents a runtime repair", () => {
  const original = globalThis.localStorage;
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    get: () => { throw new Error("storage blocked"); },
  });
  try {
    assert.doesNotThrow(() => recordRuntimeRepair("llama_runtime_incomplete"));
    assert.equal(wasRuntimeRepairedRecently("llama_runtime_incomplete"), false);
  } finally {
    Object.defineProperty(globalThis, "localStorage", { configurable: true, value: original });
  }
});
