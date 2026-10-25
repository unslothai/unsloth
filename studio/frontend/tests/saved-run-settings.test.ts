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

test("saving a set alone never changes what a load applies", () => {
  saved.saveRunSettings(MODEL, "Long", config(32768));
  assert.equal(resolveInitialConfig(MODEL, null).remembered, false);
});

test("each quant remembers its selection; unchosen, Default and a set read apart", () => {
  assert.equal(saved.getSelectedRunSettings(MODEL, "Q4_K_M"), undefined);
  assert.ok(saved.setSelectedRunSettings(MODEL, "Q4_K_M", "Long"));
  assert.ok(saved.setSelectedRunSettings(MODEL, "Q8_0", null));
  assert.equal(saved.getSelectedRunSettings(MODEL, "q4_k_m"), "Long");
  assert.equal(saved.getSelectedRunSettings(MODEL, "Q8_0"), null);
  assert.equal(
    saved.getSelectedRunSettings("unsloth/Other-GGUF", "Q4_K_M"),
    undefined,
  );
});

test("rename carries the selection; a taken name is refused", () => {
  saved.saveRunSettings(MODEL, "Long", config(32768));
  saved.saveRunSettings(MODEL, "Fast", config(4096));
  saved.setSelectedRunSettings(MODEL, "Q4_K_M", "Long");
  assert.equal(saved.renameSavedRunSettings(MODEL, "Long", "fast"), false);
  assert.ok(saved.renameSavedRunSettings(MODEL, "Long", "Very long"));
  assert.equal(saved.getSelectedRunSettings(MODEL, "Q4_K_M"), "Very long");
  assert.equal(
    saved.findSavedRunSettings(saved.listSavedRunSettings(MODEL), "Very long")
      ?.config.customContextLength,
    32768,
  );
});

test("deleting a set unchooses every quant that loaded with it", () => {
  saved.saveRunSettings(MODEL, "Long", config(32768));
  saved.setSelectedRunSettings(MODEL, "Q4_K_M", "Long");
  saved.setSelectedRunSettings(MODEL, "Q8_0", "Long");
  saved.setSelectedRunSettings("unsloth/Other-GGUF", "Q4_K_M", "Long");
  assert.ok(saved.deleteSavedRunSettings(MODEL, "Long"));
  assert.equal(saved.getSelectedRunSettings(MODEL, "Q4_K_M"), undefined);
  assert.equal(saved.getSelectedRunSettings(MODEL, "Q8_0"), undefined);
  assert.equal(
    saved.getSelectedRunSettings("unsloth/Other-GGUF", "Q4_K_M"),
    "Long",
  );
});

test("new settings are named after the model and numbered past taken names", () => {
  assert.equal(
    saved.newRunSettingsName([], "SmolLM"),
    "SmolLM custom settings",
  );
  const other = "unsloth/Named-GGUF";
  saved.saveRunSettings(other, "SmolLM custom settings", config(2048));
  assert.equal(
    saved.newRunSettingsName(saved.listSavedRunSettings(other), "SmolLM"),
    "SmolLM custom settings 2",
  );
});

test("settings remembered before named sets show up as one, and are not duplicated", () => {
  assert.equal(
    saved.adoptRememberedRunSettings(MODEL, "Q4_K_M", config(16384), "SmolLM"),
    "SmolLM custom settings",
  );
  assert.equal(
    saved.getSelectedRunSettings(MODEL, "Q4_K_M"),
    "SmolLM custom settings",
  );
  // Another quant with the same record reuses the set; a different one gets its own name.
  assert.equal(
    saved.adoptRememberedRunSettings(MODEL, "Q8_0", config(16384), "SmolLM"),
    "SmolLM custom settings",
  );
  assert.equal(
    saved.adoptRememberedRunSettings(MODEL, "BF16", config(2048), "SmolLM"),
    "SmolLM custom settings 2",
  );
  assert.equal(saved.listSavedRunSettings(MODEL).length, 2);
});

test("sets named by earlier builds take the model's custom settings name", () => {
  const legacy = "unsloth/Legacy-GGUF";
  saved.saveRunSettings(legacy, "Saved settings", config(4096));
  saved.saveRunSettings(legacy, "Custom setting for SmolLM 2", config(2048));
  saved.saveRunSettings(
    legacy,
    "Custom setting for SmolLM tuned",
    config(1024),
  );
  saved.setSelectedRunSettings(legacy, "Q4_K_M", "Saved settings");
  assert.ok(
    saved.hasLegacyRunSettings(saved.listSavedRunSettings(legacy), "SmolLM"),
  );
  saved.renameLegacyRunSettings(legacy, "SmolLM");
  // A name the user typed that only starts the same way is theirs, not an old default.
  assert.deepEqual(
    saved
      .listSavedRunSettings(legacy)
      .map((entry: { name: string }) => entry.name)
      .sort(),
    [
      "Custom setting for SmolLM tuned",
      "SmolLM custom settings",
      "SmolLM custom settings 2",
    ],
  );
  assert.equal(
    saved.getSelectedRunSettings(legacy, "Q4_K_M"),
    "SmolLM custom settings",
  );
  assert.ok(
    !saved.hasLegacyRunSettings(saved.listSavedRunSettings(legacy), "SmolLM"),
  );
});

test("run settings use one dropdown and one Save, with no Remember checkbox", () => {
  const page = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-config-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(page, /<RunSettingsPicker\s/);
  assert.match(page, /onClick=\{onSaveClick\}/);
  assert.doesNotMatch(page, /Remember for this model/);
  assert.doesNotMatch(page, /Save without loading|Forget without loading/);
  // Picking a set (or the recommended settings) is what the model loads with next.
  assert.match(
    page,
    /const pickSet = [\s\S]*?persistConfig\(next, entry !== null\)[\s\S]*?keepSelection\(/,
  );
  // Load saves into the selected set (an edited Default becomes one), so the set and what loads
  // never drift apart.
  assert.match(
    page,
    /const loadSet =\s*selectedSet \?\?[\s\S]*?saveRunSettings\(configId, loadSet, forSetStore\(effectiveRuntimeConfig\)\);[\s\S]*?persistConfig\(effectiveRuntimeConfig, loadSet !== null\)/,
  );
  // Editing Default shows the name it will be saved under; Save uses it without asking.
  assert.match(
    page,
    /selectedSet === null && !atDefault \? newRunSettingsName\(savedSets, modelName\)/,
  );
  assert.match(page, /\} else \{\s*handleSave\(customName\);/);
  // Reset goes back to Default settings at once, not just the panel values.
  assert.match(
    page,
    /disabled=\{atDefault && selectedSet === null\}[\s\S]{0,120}onClick=\{\(\) => pickSet\(null\)\}/,
  );
  // The pencil beside the dropdown renames a set, or names an edited Default before it exists.
  assert.match(
    page,
    /const editName =\s*selectedSet !== null[\s\S]*?mode: "rename"[\s\S]*?customName !== null[\s\S]*?mode: "new", name: customName[\s\S]*?: null;/,
  );
});

test("row icons never pick their row", () => {
  const row = readFileSync(
    new URL(
      "../src/features/model-picker/components/saved-run-settings-row.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Radix clicks the item on pointerup when it missed the pointerdown.
  assert.equal(row.match(/onPointerDown=\{keepRow\}/g)?.length, 2);
  assert.equal(row.match(/onPointerUp=\{keepRow\}/g)?.length, 2);
  // New settings come from editing Default, so the menu has no create entry.
  assert.doesNotMatch(row, /Save as new/);
});
