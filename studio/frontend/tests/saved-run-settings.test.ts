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
const perModel = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { DEFAULT_PER_MODEL_CONFIG, resolveInitialConfig } = perModel;

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

test("the quants that load with a set keep the picker's spelling", () => {
  saved.saveRunSettings(MODEL, "Long", config(32768));
  saved.setSelectedRunSettings(MODEL, "Q4_K_M", "Long");
  saved.setSelectedRunSettings(MODEL, "UD-Q8_K_XL", "Long");
  saved.setSelectedRunSettings(MODEL, "BF16", null);
  saved.setSelectedRunSettings("unsloth/Other-GGUF", "Q4_K_M", "Long");
  assert.deepEqual(saved.quantsSelectingRunSettings(MODEL, "long").sort(), [
    "Q4_K_M",
    "UD-Q8_K_XL",
  ]);
});

test("deleting a set moves every quant that loaded with it to Default", () => {
  saved.saveRunSettings(MODEL, "Long", config(32768));
  saved.setSelectedRunSettings(MODEL, "Q4_K_M", "Long");
  saved.setSelectedRunSettings(MODEL, "Q8_0", "Long");
  saved.setSelectedRunSettings("unsloth/Other-GGUF", "Q4_K_M", "Long");
  assert.ok(saved.deleteSavedRunSettings(MODEL, "Long"));
  // Default, not unchosen: an unchosen quant would adopt its remembered record back into a set.
  assert.equal(saved.getSelectedRunSettings(MODEL, "Q4_K_M"), null);
  assert.equal(saved.getSelectedRunSettings(MODEL, "Q8_0"), null);
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

test("a model name at the length limit still gets a unique numbered name", () => {
  const longName = "x".repeat(90);
  const first = saved.newRunSettingsName([], longName);
  assert.equal(first.length, saved.SAVED_RUN_SETTINGS_NAME_MAX);
  const other = "unsloth/Long-GGUF";
  saved.saveRunSettings(other, first, config(2048));
  const second = saved.newRunSettingsName(
    saved.listSavedRunSettings(other),
    longName,
  );
  assert.ok(second.endsWith(" 2"));
  assert.ok(second.length <= saved.SAVED_RUN_SETTINGS_NAME_MAX);
  assert.ok(saved.saveRunSettings(other, second, config(4096)));
  assert.ok(
    saved
      .newRunSettingsName(saved.listSavedRunSettings(other), longName)
      .endsWith(" 3"),
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

test("a name the user picks is kept as typed", () => {
  saved.saveRunSettings(MODEL, "Fast", config(4096));
  assert.ok(saved.renameSavedRunSettings(MODEL, "Fast", "Saved settings"));
  saved.saveRunSettings(MODEL, "Custom setting for Repo-GGUF", config(2048));
  assert.deepEqual(
    saved
      .listSavedRunSettings(MODEL)
      .map((entry: { name: string }) => entry.name),
    ["Custom setting for Repo-GGUF", "Saved settings"],
  );
  assert.equal(
    "renameLegacyRunSettings" in saved || "hasLegacyRunSettings" in saved,
    false,
  );
});

test("the per-model limit is reported only for a new name", () => {
  for (let n = 0; n < saved.SAVED_RUN_SETTINGS_PER_MODEL_MAX; n++) {
    assert.ok(saved.saveRunSettings(MODEL, `Set ${n}`, config(1024 + n * 128)));
  }
  assert.equal(saved.isRunSettingsLimitReached(MODEL, "One more"), true);
  assert.equal(saved.saveRunSettings(MODEL, "One more", config(4096)), false);
  // Overwriting an existing set is still allowed.
  assert.equal(saved.isRunSettingsLimitReached(MODEL, "set 3"), false);
  assert.ok(saved.saveRunSettings(MODEL, "Set 3", config(4096)));
});

test("forgetting a quant's server settings puts its selection back on Default", () => {
  const { savePerModelConfig } = perModel;
  saved.saveRunSettings(MODEL, "Fast", config(4096));
  assert.ok(savePerModelConfig(MODEL, "Q4_K_M", config(4096)));
  assert.ok(savePerModelConfig(MODEL, "Q8_0", config(4096)));
  saved.setSelectedRunSettings(MODEL, "Q4_K_M", "Fast");
  saved.setSelectedRunSettings(MODEL, "Q8_0", "Fast");
  assert.ok(saved.forgetPerModelConfigsForOverrideKeys([`${MODEL}:q4_k_m`]));
  assert.equal(resolveInitialConfig(MODEL, "Q4_K_M").remembered, false);
  assert.equal(saved.getSelectedRunSettings(MODEL, "Q4_K_M"), null);
  // Another quant, and the set itself, are left alone.
  assert.equal(saved.getSelectedRunSettings(MODEL, "Q8_0"), "Fast");
  assert.deepEqual(saved.quantsSelectingRunSettings(MODEL, "Fast"), ["Q8_0"]);
  assert.equal(saved.listSavedRunSettings(MODEL).length, 1);
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
    /const pickSet = [\s\S]*?persistConfig\(applyToPanel\(next\), entry !== null\)[\s\S]*?keepSelection\(/,
  );
  // Load saves into the selected set (an edited Default becomes one), so the set and what loads
  // never drift apart.
  assert.match(
    page,
    /let loadSet =\s*selectedSet \?\?[\s\S]*?!saveSharedSet\(loadSet, effectiveRuntimeConfig\)[\s\S]*?if \(loadSet !== selectedSet\) \{\s*loadSet = null;[\s\S]*?persistConfig\(effectiveRuntimeConfig, loadSet !== null\)/,
  );
  // Saving an edited Default creates a set, so it takes the same extra-argument gate as a set.
  assert.match(
    page,
    /\(\(selectedSet !== null \|\| customName !== null\) && extraArgsBlocked\)/,
  );
  assert.match(page, /if \(keep && extraArgsBlocked\) \{/);
  // The name box's Save and Enter reach handleSave, so it holds the Save button's readiness gates.
  assert.match(
    page,
    /const handleSave = [\s\S]{0,400}if \(stagedMetadataPending \|\| budgetSettling\) \{\s*return false;/,
  );
  // No pick or Reset before the server copy is read: it could not forget a row it has not seen.
  assert.match(
    page,
    /disabled=\{sharedVariantUnresolved \|\| extraArgsHydrating \|\| budgetSettling\}/,
  );
  // Editing Default shows the name it will be saved under; Save uses it without asking.
  assert.match(
    page,
    /selectedSet === null && !atDefault \? newRunSettingsName\(savedSets, modelName\)/,
  );
  assert.match(page, /\} else \{\s*handleSave\(customName\);/);
  // Reset goes back at once to what loads every time (the pinned set, else Default), not just the panel values.
  assert.match(
    page,
    /\(pinnedSet !== null\s*\? selectedSet === pinnedSet && !setUnsaved\s*: atDefault && selectedSet === null\) \|\|\s*extraArgsHydrating \|\|\s*budgetSettling\s*\}\s*onClick=\{\(\) => pickSet\(pinnedSet\)\}/,
  );
  // A number the Save or Load click itself committed is not in customName yet, so both name the
  // new set from the committed config.
  assert.match(
    page,
    /const target = into \?\? committedCustomName\(effectiveRuntimeConfig\);/,
  );
  assert.match(
    page,
    /let loadSet = selectedSet \?\? committedCustomName\(effectiveRuntimeConfig\);/,
  );
  // A load settling its budget captured its config already; a pick then would disagree with it.
  assert.match(
    page,
    /const pickSet = [\s\S]{0,200}if \(budgetSettling\) \{\s*return;/,
  );
  // A shared set: every save into it reaches the other quants that load with it, and a delete
  // forgets their records too.
  assert.match(
    page,
    /if \(keep && !saveSharedSet\(target, effectiveRuntimeConfig\)\)/,
  );
  assert.match(
    page,
    /if \(loadSet !== null && !saveSharedSet\(loadSet, effectiveRuntimeConfig\)\)/,
  );
  assert.match(
    page,
    /const otherQuants = quantsSelectingRunSettings\(configId, name\);[\s\S]*?syncOtherQuants\(otherQuants, null\);/,
  );
  // Syncing other quants can evict another model to fit the budget; its server override goes too.
  assert.match(
    page,
    /savePerModelConfig\(configId, variant, stored, evicted\)[\s\S]{0,400}for \(const dropped of evicted\) \{\s*syncModelOverride\(dropped\.modelId, dropped\.ggufVariant, null/,
  );
  // A pick persists what the panel shows, so a set without arguments clears the server's old flags,
  // and it lifts a stale objection to arguments the pick replaced.
  assert.match(page, /persistConfig\(applyToPanel\(next\), entry !== null\)/);
  assert.match(
    page,
    /setConfig\(applied\);[\s\S]{0,300}setExtraArgsLoadable\(true\);\s*return applied;/,
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

test("a set whose record was forgotten is not offered, and deleting one asks first", () => {
  const page = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-config-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Only once the server copy is read, and never for a set at defaults, which keeps no record.
  assert.match(
    page,
    /const selectionStale =\s*persistedEntry !== null &&\s*!extraArgsHydrating &&\s*!storedRecord\.remembered &&\s*!savedRemember &&\s*!isDefaultConfig\(forSetStore\(persistedEntry\.config\)\);/,
  );
  assert.match(
    page,
    /if \(selectionStale\) \{\s*setSelectedRunSettings\(configId, target\.ggufVariant, null\);/,
  );
  assert.match(page, /selectionStale\s*\? null\s*: \(persistedSet \?\? null\)/);
  // The trash icon opens a confirmation; only its Delete removes the set.
  assert.match(page, /onDelete=\{setPendingDelete\}/);
  assert.match(
    page,
    /if \(pendingDelete !== null\) \{\s*deleteSet\(pendingDelete\);/,
  );
  // The limit is named rather than a bare failure.
  assert.match(page, /isRunSettingsLimitReached\(configId, name\)/);
  assert.doesNotMatch(page, /persistenceOnly|rememberChanged/);
});

test("rows can be renamed and deleted from the keyboard", () => {
  const row = readFileSync(
    new URL(
      "../src/features/model-picker/components/saved-run-settings-row.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(row, /aria-keyshortcuts="F2 Delete"/);
  // Escape in the name box must reach only the box: Radix overlays dismiss from a document capture listener
  // unless the event is already prevented, so the guard sits on window capture.
  assert.match(row, /window\.addEventListener\("keydown", keepPanel, true\)/);
  assert.match(
    row,
    /closest\("\[data-run-settings-name\]"\)[\s\S]{0,40}event\.preventDefault\(\)/,
  );
  assert.match(row, /event\.key === "F2"[\s\S]{0,80}rename\(entry\.name\)/);
  assert.match(row, /event\.key === "Delete"[\s\S]{0,80}remove\(entry\.name\)/);
});

test("a pinned set follows renames and goes with its set or a forget", () => {
  saved.saveRunSettings(MODEL, "Mine", config(4096));
  saved.saveRunSettings(MODEL, "Trial", config(2048));
  assert.equal(saved.getPinnedRunSettings(MODEL, "Q4_K_M"), null);
  assert.ok(saved.setPinnedRunSettings(MODEL, "q4_k_m", "Mine"));
  // Keyed like the selection: quant case folds, other quants are separate.
  assert.equal(saved.getPinnedRunSettings(MODEL, "Q4_K_M"), "Mine");
  assert.equal(saved.getPinnedRunSettings(MODEL, "Q8_0"), null);
  assert.ok(saved.renameSavedRunSettings(MODEL, "Mine", "Long context"));
  assert.equal(saved.getPinnedRunSettings(MODEL, "Q4_K_M"), "Long context");
  // Deleting another set leaves the pin alone; deleting the pinned one unpins.
  assert.ok(saved.deleteSavedRunSettings(MODEL, "Trial"));
  assert.equal(saved.getPinnedRunSettings(MODEL, "Q4_K_M"), "Long context");
  assert.ok(saved.deleteSavedRunSettings(MODEL, "Long context"));
  assert.equal(saved.getPinnedRunSettings(MODEL, "Q4_K_M"), null);

  saved.saveRunSettings(MODEL, "Mine", config(4096));
  saved.setPinnedRunSettings(MODEL, "Q4_K_M", "Mine");
  assert.ok(perModel.savePerModelConfig(MODEL, "Q4_K_M", config(4096)));
  saved.forgetPerModelConfigsForOverrideKeys([`${MODEL}:Q4_K_M`]);
  assert.equal(saved.getPinnedRunSettings(MODEL, "Q4_K_M"), null);
  assert.ok(saved.setPinnedRunSettings(MODEL, "Q4_K_M", "Mine"));
  assert.ok(saved.setPinnedRunSettings(MODEL, "Q4_K_M", null));
  assert.equal(saved.getPinnedRunSettings(MODEL, "Q4_K_M"), null);
});

test("while a set is pinned, picking or loading another uses it once", () => {
  const page = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-config-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // A pick of another set only fills the panel: no record, override or selection is written.
  assert.match(
    page,
    /const pickSet = \(name: string \| null, pinned = pinnedSet\) => \{[\s\S]{0,400}if \(pinned !== null && entry\?\.name !== pinned\) \{\s*applyToPanel\(next\);\s*setPickedSet\(entry\?\.name \?\? null\);\s*return;/,
  );
  // Load saves into the set it shows, but leaves the pinned record and override for the next load.
  assert.match(
    page,
    /const onceOnly = pinnedSet !== null && loadSet !== pinnedSet;\s*const \{ saved \} = onceOnly\s*\? \{ saved: true \}\s*: persistConfig\(effectiveRuntimeConfig, loadSet !== null\);/,
  );
  // Save into another set does the same, and never forgets while pinned.
  assert.match(page, /if \(!keep && pinned !== null\) \{\s*return false;/);
  assert.match(
    page,
    /if \(pinned !== null && target !== pinned\) \{[\s\S]{0,120}setPickedSet\(target\);/,
  );
  // Ticking the box pins what is shown and makes it what loads, saving its edits first.
  assert.match(
    page,
    /const setAlwaysLoad = [\s\S]*?setPinnedRunSettings\(configId, target\.ggufVariant, selectedSet\)[\s\S]*?if \(setUnsaved\) \{\s*if \(!handleSave\(selectedSet, message, selectedSet\)\)[\s\S]*?pickSet\(selectedSet, selectedSet\);/,
  );
  assert.match(page, /Always load with these settings/);
  // The box and the status share the one row under the dropdown.
  assert.match(
    page,
    /<Checkbox\s+id=\{alwaysLoadId\}[\s\S]{0,1600}<\/label>\s*<p\s+id=\{saveStatusId\}[\s\S]{0,500}\{saveStatus\}\s*<\/p>\s*<\/div>/,
  );
  // The server row is the pinned set, so it must not replace what a resident model loaded once.
  assert.match(
    page,
    /if \(!\(loadedConfig && getPinnedRunSettings\(configId, target\.ggufVariant\) !== null\)\) \{\s*setExtraArgsLoadable\(hydratedIsLoadable\);\s*replaceModelConfigDraft\(/,
  );
  // Deleting the pinned set unpins it, and what the panel shows becomes what loads.
  assert.match(
    page,
    /if \(name === pinnedSet\) \{[\s\S]{0,120}pickSet\(wasSelected \? null : selectedSet, null\);/,
  );
});
