// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

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
  assert.equal(s.agentsOs, null);
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

test("a quant remembered for one model does not follow onto another", () => {
  const s = useSettingsPanelPrefsStore.getState();
  s.setAgentsVariant("unsloth/Baz-GGUF", "Q4_K_M");
  const next = useSettingsPanelPrefsStore.getState();
  assert.equal(next.agentsVariant, "Q4_K_M");
  assert.equal(next.agentsVariantModel, "unsloth/Baz-GGUF");
  assert.equal(next.agentsModel, null, "a quant pick must not pin the model");
});

test("a setter write round-trips through localStorage", () => {
  useSettingsPanelPrefsStore.getState().setFineTuneAction("export");
  const raw = store.get(KEY);
  assert.ok(raw, "nothing was written");
  assert.equal(JSON.parse(raw as string).state.fineTuneAction, "export");
});

test("the Agents command shell override persists and rejects unknown values", () => {
  useSettingsPanelPrefsStore.getState().setAgentsOs("unix");
  assert.equal(useSettingsPanelPrefsStore.getState().agentsOs, "unix");
  const raw = store.get(KEY);
  assert.ok(raw);
  assert.equal(JSON.parse(raw as string).state.agentsOs, "unix");

  const merged = useSettingsPanelPrefsStore.persist.getOptions().merge;
  assert.ok(merged);
  const out = merged(
    { agentsOs: "fish" },
    useSettingsPanelPrefsStore.getState(),
  ) as { agentsOs: unknown };
  assert.equal(out.agentsOs, null);
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

// The case agentsVariantModel exists for: the tab keeps following the resident
// model, but the quant picked against it still survives the unmount.
test("a quant picked while following the resident model does not pin a model", () => {
  const s = useSettingsPanelPrefsStore.getState();
  s.setAgentsModel(null, null);
  s.setAgentsVariant("unsloth/Qux-GGUF", "Q6_K");
  const next = useSettingsPanelPrefsStore.getState();
  assert.equal(next.agentsModel, null);
  assert.equal(next.agentsVariant, "Q6_K");
  assert.equal(next.agentsVariantModel, "unsloth/Qux-GGUF");
});

// Half a pair is unusable: a quant with no model can never be scoped to one.
test("a quant with no model to scope it to is dropped", () => {
  const merged = useSettingsPanelPrefsStore.persist.getOptions().merge;
  assert.ok(merged);
  const out = merged(
    { agentsVariant: "Q6_K" },
    useSettingsPanelPrefsStore.getState(),
  ) as { agentsVariant: unknown; agentsVariantModel: unknown };
  assert.equal(out.agentsVariant, null);
  assert.equal(out.agentsVariantModel, null);
});

// A downgrade must not read a newer record: the field names may have been
// reused with different meaning.
test("a record from a newer build falls back to defaults", () => {
  const { migrate, version } = useSettingsPanelPrefsStore.persist.getOptions();
  assert.ok(migrate);
  assert.equal(version, 1);
  assert.deepEqual(migrate({ agentsModel: "unsloth/Foo-GGUF" }, 2), {});
  assert.deepEqual(migrate({ agentsModel: "unsloth/Foo-GGUF" }, 0), {
    agentsModel: "unsloth/Foo-GGUF",
  });
});

// Settling in a .finally let a superseded or failed poll release the retire
// with no resident model recorded, which erased the saved model and quant.
test("the status poll settles only on the read that applied", async () => {
  const source = await readFile(
    new URL("../src/features/settings/tabs/agents-tab.tsx", import.meta.url),
    "utf8",
  );
  const sync = source.slice(
    source.indexOf("const sync = ()"),
    source.indexOf("const timer = window.setInterval"),
  );
  assert.ok(sync, "the status poll moved; this contract needs updating");
  assert.ok(
    !sync.includes(".finally("),
    "a stale or failed poll must not settle",
  );
  const applied = sync.slice(
    sync.indexOf("seq === statusSeq.current"),
    sync.indexOf(".catch("),
  );
  assert.match(applied, /setStatusSettled\(true\)/);
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

// A quant is scoped to the repo it was picked for, and the catalog, cache and
// status endpoints can disagree on repo-id casing, so that scope check has to
// normalize or the user's quant is dropped when the spelling differs.
test("the remembered quant is scoped through modelKey, not an exact compare", async () => {
  const source = await readFile(
    new URL("../src/features/settings/tabs/agents-tab.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    source,
    /modelKey\(chosen\.model\) === modelKey\(model\)/,
    "rememberedVariant must compare through modelKey",
  );
  const exact = source
    .split("\n")
    .filter((line) => line.includes("chosenVariant.current?.model ==="));
  assert.deepEqual(
    exact,
    [],
    "an exact repo-id compare on chosenVariant drops the quant on a casing difference",
  );
});

// An unreadable record leaves the pre-PR defaults, so a mangled blob never
// changes how the tabs behave.
test("an unreadable record leaves the defaults", () => {
  const merged = useSettingsPanelPrefsStore.persist.getOptions().merge;
  assert.ok(merged);
  const out = merged(null, useSettingsPanelPrefsStore.getState());
  assert.equal(out.agentsModel, null);
  assert.equal(out.agentsVariant, null);
  assert.equal(out.resourcesLiveUpdates, true);
  assert.equal(out.fineTuneAction, "train");
});

// Last: it rehydrates the store. Corrupt JSON must not take settings down.
test("corrupt JSON does not break the store", async () => {
  store.set(KEY, "{not json");
  await assert.doesNotReject(async () => {
    await useSettingsPanelPrefsStore.persist.rehydrate();
  });
  assert.equal(
    typeof useSettingsPanelPrefsStore.getState().setAgentsModel,
    "function",
  );
});
