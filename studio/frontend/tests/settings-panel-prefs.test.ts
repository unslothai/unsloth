// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const KEY = "unsloth_settings_panel_prefs";

// A record written before the sanitiser existed, holding every field.
store.set(
  KEY,
  JSON.stringify({
    state: {
      agentsAgent: "codex",
      agentsModel: "unsloth/Foo-GGUF",
      agentsVariant: "UD-Q4_K_XL",
      agentsVariantModel: "unsloth/Foo-GGUF",
      apiExampleLang: "pythonTools",
      apiExampleOs: "windows",
      apiExampleAgent: "codex",
      resourcesLiveUpdates: false,
      fineTuneAction: "recipes",
    },
    version: 0,
  }),
);

const { useSettingsPanelPrefsStore, SETTINGS_PANEL_PREFS_STORAGE_KEY } =
  await import("../src/features/settings/stores/settings-panel-prefs-store.ts");

test("a version 0 record hydrates every field", () => {
  const s = useSettingsPanelPrefsStore.getState();
  assert.equal(s.agentsAgent, "codex");
  assert.equal(s.agentsModel, "unsloth/Foo-GGUF");
  assert.equal(s.agentsVariant, "UD-Q4_K_XL");
  assert.equal(s.apiExampleOs, "windows");
  assert.equal(s.resourcesLiveUpdates, false);
  assert.equal(s.fineTuneAction, "recipes");
});

test("picking a model carries its quant, and clearing it clears the quant", () => {
  const s = useSettingsPanelPrefsStore.getState();
  s.setAgentsModel("unsloth/Bar-GGUF", "Q4_K_M");
  let next = useSettingsPanelPrefsStore.getState();
  assert.equal(next.agentsModel, "unsloth/Bar-GGUF");
  assert.equal(next.agentsVariantModel, "unsloth/Bar-GGUF");
  next.setAgentsModel(null, null);
  next = useSettingsPanelPrefsStore.getState();
  assert.equal(next.agentsModel, null);
  assert.equal(next.agentsVariant, null);
  assert.equal(next.agentsVariantModel, null);
});

test("a quant can be dropped without disturbing the model", () => {
  const s = useSettingsPanelPrefsStore.getState();
  s.setAgentsModel("unsloth/Baz-GGUF", "Q4_K_M");
  useSettingsPanelPrefsStore.getState().clearAgentsVariant();
  const next = useSettingsPanelPrefsStore.getState();
  assert.equal(next.agentsModel, "unsloth/Baz-GGUF");
  assert.equal(next.agentsVariant, null);
  assert.equal(next.agentsVariantModel, null);
});

test("a setter write round-trips through localStorage", () => {
  useSettingsPanelPrefsStore.getState().setFineTuneAction("export");
  const raw = store.get(KEY);
  assert.ok(raw, "nothing was written");
  assert.equal(JSON.parse(raw as string).state.fineTuneAction, "export");
});

// The reason the sanitiser exists: agentsModel reaches `.toLowerCase()` and the
// path checks in agents-tab, so a non-string takes the whole app down.
test("a non-string model is refused rather than handed to the tab", () => {
  const merged = useSettingsPanelPrefsStore.persist.getOptions().merge;
  assert.ok(merged, "merge must be supplied, or untrusted JSON reaches the UI");
  const out = merged(
    { agentsModel: 42, agentsVariant: [], fineTuneAction: "obliterate" },
    useSettingsPanelPrefsStore.getState(),
  ) as { agentsModel: unknown; fineTuneAction: string };
  assert.equal(out.agentsModel, null);
  assert.equal(out.fineTuneAction, "train");
});

test("a persisted blob cannot replace the store actions", () => {
  const merged = useSettingsPanelPrefsStore.persist.getOptions().merge;
  assert.ok(merged);
  const out = merged(
    { setFineTuneAction: 5 },
    useSettingsPanelPrefsStore.getState(),
  ) as { setFineTuneAction: unknown };
  assert.equal(typeof out.setFineTuneAction, "function");
});

// Reset-all is the only in-app escape hatch from a bad pinned model.
test("Reset all local preferences clears this key", async () => {
  const source = await readFile(
    new URL("../src/features/settings/tabs/general-tab.tsx", import.meta.url),
    "utf8",
  );
  const keys = source.slice(
    source.indexOf("const PREFS_KEYS"),
    source.indexOf("];", source.indexOf("const PREFS_KEYS")),
  );
  assert.ok(
    keys.includes("SETTINGS_PANEL_PREFS_STORAGE_KEY") ||
      keys.includes(`"${SETTINGS_PANEL_PREFS_STORAGE_KEY}"`),
    `${SETTINGS_PANEL_PREFS_STORAGE_KEY} missing from PREFS_KEYS`,
  );
});
