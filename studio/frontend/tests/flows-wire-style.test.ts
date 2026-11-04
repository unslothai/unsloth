// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const stored = new Map<string, string>();
Object.defineProperty(globalThis, "localStorage", {
  configurable: true,
  value: {
    getItem: (key: string) => stored.get(key) ?? null,
    setItem: (key: string, value: string) => void stored.set(key, value),
    removeItem: (key: string) => void stored.delete(key),
  },
});

const STORE = "../src/features/flows/stores/ui-store.ts";

test("wire style defaults to Bus and the choice persists per user, not per flow", async () => {
  const { useUiStore, storedWireStyle, WIRE_STYLE_STORAGE_KEY } = await import(STORE);
  assert.equal(useUiStore.getState().wireStyle, "bus");
  useUiStore.getState().setWireStyle("curved");
  assert.equal(stored.get(WIRE_STYLE_STORAGE_KEY), "curved");
  // Clearing the editor for another flow keeps it.
  useUiStore.getState().reset();
  assert.equal(useUiStore.getState().wireStyle, "curved");
  // What the next page load starts from.
  assert.equal(storedWireStyle(), "curved");
  stored.set(WIRE_STYLE_STORAGE_KEY, "nonsense");
  assert.equal(storedWireStyle(), "bus");
});

test("the saved document never carries the wire style", async () => {
  const { useDocStore, documentFromState } = await import(
    "../src/features/flows/stores/doc-store.ts"
  );
  const { useUiStore } = await import(STORE);
  useUiStore.getState().setWireStyle("curved");
  const saved = JSON.stringify(documentFromState(useDocStore.getState()));
  assert.ok(!saved.includes("curved") && !saved.includes("wireStyle"));
});
