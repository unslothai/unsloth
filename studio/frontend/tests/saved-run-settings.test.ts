// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const saved = await import(
  "../src/features/model-picker/model-config/saved-run-settings.ts"
);
const { DEFAULT_PER_MODEL_CONFIG, resolveInitialConfig } = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

const MODEL = "unsloth/Repo-GGUF";
const config = (customContextLength: number | null) => ({
  ...DEFAULT_PER_MODEL_CONFIG,
  customContextLength,
});

test.beforeEach(() => store.clear());

test("one model keeps several named settings, listed by name", () => {
  assert.ok(saved.saveRunSettings(MODEL, "Long context", config(32768)));
  assert.ok(saved.saveRunSettings(MODEL, "  Fast  ", config(4096)));
  const list = saved.listSavedRunSettings(MODEL);
  assert.deepEqual(
    list.map((entry: { name: string }) => entry.name),
    ["Fast", "Long context"],
  );
  assert.equal(list[1].config.customContextLength, 32768);
  // Keyed per model, case-insensitively.
  assert.equal(saved.listSavedRunSettings("UNSLOTH/repo-gguf").length, 2);
  assert.equal(saved.listSavedRunSettings("unsloth/Other-GGUF").length, 0);
});

test("saving under an existing name replaces it, and delete drops only that one", () => {
  saved.saveRunSettings(MODEL, "Fast", config(4096));
  saved.saveRunSettings(MODEL, "Long", config(32768));
  assert.ok(saved.saveRunSettings(MODEL, "fast", config(2048)));
  let list = saved.listSavedRunSettings(MODEL);
  assert.equal(list.length, 2);
  assert.equal(
    saved.findSavedRunSettings(list, "FAST")?.config.customContextLength,
    2048,
  );
  assert.ok(saved.deleteSavedRunSettings(MODEL, "Fast"));
  list = saved.listSavedRunSettings(MODEL);
  assert.deepEqual(
    list.map((entry: { name: string }) => entry.name),
    ["Long"],
  );
});

test("blank names and a full list are refused", () => {
  assert.equal(saved.saveRunSettings(MODEL, "   ", config(4096)), false);
  for (let i = 0; i < saved.SAVED_RUN_SETTINGS_PER_MODEL_MAX; i++) {
    assert.ok(saved.saveRunSettings(MODEL, `Set ${i}`, config(4096 + i)));
  }
  assert.equal(saved.saveRunSettings(MODEL, "One more", config(4096)), false);
  assert.ok(saved.saveRunSettings(MODEL, "Set 0", config(8192)));
});

test("a future-schema entry is kept but not offered", () => {
  store.set(
    saved.SAVED_RUN_SETTINGS_STORAGE_KEY,
    JSON.stringify({
      [MODEL.toLowerCase()]: [
        {
          name: "From the future",
          savedAt: 1,
          config: { version: 999, customContextLength: 1 },
        },
      ],
    }),
  );
  assert.equal(saved.listSavedRunSettings(MODEL).length, 0);
});

test("a panel opened again reads what another tab saved while none was open", () => {
  saved.saveRunSettings(MODEL, "Fast", config(4096));
  assert.equal(saved.listSavedRunSettings(MODEL).length, 1);
  // Another tab's write, with nothing here listening.
  const map = JSON.parse(
    store.get(saved.SAVED_RUN_SETTINGS_STORAGE_KEY) ?? "{}",
  );
  const [key] = Object.keys(map);
  map[key] = [];
  store.set(saved.SAVED_RUN_SETTINGS_STORAGE_KEY, JSON.stringify(map));
  const unsubscribe = saved.subscribeSavedRunSettings(() => undefined);
  assert.equal(saved.listSavedRunSettings(MODEL).length, 0);
  unsubscribe();
});

test("saved settings never touch what a load applies: that stays Remember's record", () => {
  saved.saveRunSettings(MODEL, "Long", config(32768));
  assert.equal(resolveInitialConfig(MODEL, null).remembered, false);
});

test("run settings offer the saved list and keep Forget next to Save without loading", () => {
  const page = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-config-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(page, /<SavedRunSettingsRow\s/);
  assert.ok(
    page.includes(
      '{remember ? "Save without loading" : "Forget without loading"}',
    ),
  );
});
