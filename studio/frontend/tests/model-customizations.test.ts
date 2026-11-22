// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

const key = "unsloth_model_customizations";
const icon = "data:image/png;base64,iVBORw0KGgo=";
const data = new Map<string, string>([
  [
    key,
    JSON.stringify({
      state: {
        byId: {
          "Org/Saved": { name: "  Spark  ", icon },
          "org/bad-icon": { icon: "javascript:alert(1)" },
          "org/huge-icon": {
            icon: `data:image/png;base64,${"A".repeat(70_000)}`,
          },
          "org/not-object": "x",
        },
      },
      version: 0,
    }),
  ],
]);
let fullStorage = false;
const listeners: ((event: { key: string | null }) => void)[] = [];
const localStorage = {
  getItem: (name: string) => data.get(name) ?? null,
  setItem: (name: string, value: string) => {
    if (fullStorage) throw new Error("QuotaExceededError");
    data.set(name, value);
  },
  removeItem: (name: string) => void data.delete(name),
};
Object.assign(globalThis, {
  localStorage,
  window: {
    localStorage,
    addEventListener: (
      type: string,
      listener: (event: { key: string | null }) => void,
    ) => {
      if (type === "storage") listeners.push(listener);
    },
  },
});

const { modelCustomizationKey, useModelCustomizationsStore } = await import(
  "../src/features/chat/model-customizations.ts"
);

const byId = () => useModelCustomizationsStore.getState().byId;
const set = (id: string, value: { name?: string; icon?: string }) =>
  useModelCustomizationsStore.getState().setCustomization(id, value);

test("stored entries are keyed case-insensitively and malformed ones dropped", () => {
  assert.deepEqual(byId(), { "org/saved": { name: "Spark", icon } });
  assert.equal(modelCustomizationKey(" Org/Saved "), "org/saved");
  assert.equal(
    modelCustomizationKey(
      "/home/u/.cache/huggingface/hub/models--unsloth--Qwen3-4B-GGUF/snapshots/abc123",
    ),
    "unsloth/qwen3-4b-gguf",
  );
});

test("saving is idempotent and an empty save clears the entry", () => {
  set("unsloth/Qwen3-4B-GGUF", { name: "Flint", icon });
  const first = byId();
  set("unsloth/qwen3-4b-gguf", { name: " Flint ", icon });
  assert.equal(byId(), first);
  assert.deepEqual(
    JSON.parse(data.get(key) ?? "{}").state.byId["unsloth/qwen3-4b-gguf"],
    {
      name: "Flint",
      icon,
    },
  );

  set("unsloth/Qwen3-4B-GGUF", { name: "  " });
  assert.equal(byId()["unsloth/qwen3-4b-gguf"], undefined);
});

test("names are capped and non-image icons are ignored", () => {
  set("org/long", { name: "x".repeat(200), icon: "https://example.com/a.png" });
  assert.deepEqual(byId()["org/long"], { name: "x".repeat(80) });
});

test("a full storage keeps the change in memory instead of throwing", () => {
  fullStorage = true;
  try {
    assert.doesNotThrow(() => set("org/full", { name: "Kept" }));
    assert.equal(byId()["org/full"]?.name, "Kept");
  } finally {
    fullStorage = false;
  }
});

test("another window's change is picked up", async () => {
  const saved = JSON.parse(data.get(key) ?? "{}");
  saved.state.byId = { ...saved.state.byId, "org/other": { name: "Other" } };
  data.set(key, JSON.stringify(saved));
  for (const listener of listeners) listener({ key });
  await Promise.resolve();
  assert.equal(byId()["org/other"]?.name, "Other");
});
