// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Dismissal and the Settings toggle must use separate flags: a load reopens a dismissed card.

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

const LOADED_MODELS_INDICATOR = readSrc(
  "features/loaded-models/loaded-models-indicator.tsx",
);

const { store } = installLocalStorageFake();

const {
  LOADED_MODELS_PREFERENCE_KEYS,
  getLoadedModelsDismissed,
  getShowLoadedModels,
  setLoadedModelsDismissed,
  setShowLoadedModels,
} = await import("../src/features/loaded-models/show-loaded-models-pref.ts");

function reset(): void {
  store.clear();
}

test("the indicator is off until it is switched on", () => {
  reset();
  assert.equal(getShowLoadedModels(), false);
});

// Never removed: a pre-update tab reads a missing key as on.
test("both toggles store a value the older reader also honours", () => {
  reset();
  setShowLoadedModels(true);
  assert.equal(store.get(LOADED_MODELS_PREFERENCE_KEYS.show), "true");
  setShowLoadedModels(false);
  assert.equal(store.get(LOADED_MODELS_PREFERENCE_KEYS.show), "false");
  assert.equal(getShowLoadedModels(), false);
});

test("an older explicit false still reads as off", () => {
  reset();
  store.set(LOADED_MODELS_PREFERENCE_KEYS.show, "false");
  assert.equal(getShowLoadedModels(), false);
});

test("a cleared key falls back to off, not on", () => {
  reset();
  setShowLoadedModels(true);
  store.delete(LOADED_MODELS_PREFERENCE_KEYS.show);
  assert.equal(getShowLoadedModels(), false);
});

test("the card is open until something closes it", () => {
  reset();
  assert.equal(getLoadedModelsDismissed(), false);
});

test("closing it stores the dismissal, reopening removes the key", () => {
  reset();
  setLoadedModelsDismissed(true);
  assert.equal(getLoadedModelsDismissed(), true);
  assert.equal(store.get(LOADED_MODELS_PREFERENCE_KEYS.dismissed), "true");
  setLoadedModelsDismissed(false);
  assert.equal(getLoadedModelsDismissed(), false);
  assert.equal(store.has(LOADED_MODELS_PREFERENCE_KEYS.dismissed), false);
});

test("closing the card does not switch the setting off", () => {
  reset();
  setShowLoadedModels(true);
  setLoadedModelsDismissed(true);
  assert.equal(getShowLoadedModels(), true);
});

test("switching the setting off is not a dismissal a load can undo", () => {
  reset();
  setShowLoadedModels(false);
  setLoadedModelsDismissed(false);
  assert.equal(
    getShowLoadedModels(),
    false,
    "a load reopening the card must not re-enable a disabled one",
  );
});

// Every load start sets this, so a no-op set must not re-render.
test("setting the dismissal to what it already is changes nothing", () => {
  reset();
  setLoadedModelsDismissed(false);
  assert.equal(store.size, 0, "an unchanged set must not write");
  setLoadedModelsDismissed(true);
  assert.equal(store.size, 1);
  setLoadedModelsDismissed(true);
  assert.equal(store.size, 1);
});

test("the dismissal key is cleared by Reset all local preferences", () => {
  const generalTab = readSrc("features/settings/tabs/general-tab.tsx");
  assert.match(generalTab, /LOADED_MODELS_PREFERENCE_KEYS\.dismissed,/);
});

test("the card carries a close button, and a load brings it back", () => {
  assert.match(LOADED_MODELS_INDICATOR, /aria-label="Close loaded models"/);
  assert.match(
    LOADED_MODELS_INDICATOR,
    /onClick=\{\(\) => setLoadedModelsDismissed\(true\)\}/,
  );
  assert.match(
    LOADED_MODELS_INDICATOR,
    /subscribeModelLifecycle\(\(\{ loading \}\) => \{\s*if \(loading\) \{\s*setLoadedModelsDismissed\(false\);/,
  );
  assert.match(
    LOADED_MODELS_INDICATOR,
    /const enabled = showIndicator && !dismissed && reachable;/,
  );
});

// Requested icon: hugeicons sparkle (single), not SparklesIcon or lib/sparkles-icon.
test("the card is badged with the single sparkle, not the brain", () => {
  assert.match(LOADED_MODELS_INDICATOR, /icon=\{SparkleIcon\}/);
  assert.match(LOADED_MODELS_INDICATOR, /from "@\/lib\/sparkle-icon"/);
  assert.doesNotMatch(LOADED_MODELS_INDICATOR, /AiBrain01Icon|SparklesIcon/);
});

test("a row ejects with the eject glyph, the header closes with an X", () => {
  const row = LOADED_MODELS_INDICATOR.slice(
    LOADED_MODELS_INDICATOR.indexOf("function LoadedModelRow"),
    LOADED_MODELS_INDICATOR.indexOf("export function LoadedModelsIndicator"),
  );
  assert.match(row, /icon=\{RemoveCircleIcon\}/);
  assert.doesNotMatch(row, /icon=\{Cancel01Icon\}/);
  const header = LOADED_MODELS_INDICATOR.slice(
    LOADED_MODELS_INDICATOR.indexOf('aria-label="Close loaded models"'),
  );
  assert.match(header, /icon=\{Cancel01Icon\}/);
});

// Tracking must respect the auth gate, or /login polls protected endpoints every 5s.
test("recording follows the route and auth gate, but not the dismissal", () => {
  assert.match(
    LOADED_MODELS_INDICATOR,
    /const reachable = canShowIndicator\(pathname\);/,
  );
  assert.match(
    LOADED_MODELS_INDICATOR,
    /hasAuthToken\(\) && !mustChangePassword\(\)/,
    "canShowIndicator must still be the thing that gates on auth",
  );
  const call = LOADED_MODELS_INDICATOR.slice(
    LOADED_MODELS_INDICATOR.indexOf("useLoadedModels("),
    LOADED_MODELS_INDICATOR.indexOf("useLoadedModels(") + 200,
  );
  assert.match(call, /showIndicator && reachable/, "track must be gated too");
  assert.doesNotMatch(
    call,
    /\n\s*showIndicator,\n/,
    "the preference alone is what polled /login",
  );
  // Dismissal stays out, or a closed card could never reopen.
  assert.doesNotMatch(call, /dismissed/);
});
