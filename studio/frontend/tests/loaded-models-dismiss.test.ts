// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Closing the card and switching it off in Settings are two different wishes,
// and they must not share a flag: the next model load reopens a card that was
// waved away, and would otherwise also reopen one deliberately turned off.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

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
  // Removed, not stored as "false", so the default stays open.
  assert.equal(store.has(LOADED_MODELS_PREFERENCE_KEYS.dismissed), false);
});

// The point of keeping them apart.
test("closing the card does not switch the setting off", () => {
  reset();
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

// Every load start reopens the card, so a set to the value it already holds
// must be inert: otherwise each load re-renders the whole overlay stack.
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
  const generalTab = readFileSync(
    new URL("../src/features/settings/tabs/general-tab.tsx", import.meta.url),
    "utf8",
  );
  assert.match(generalTab, /LOADED_MODELS_PREFERENCE_KEYS\.dismissed,/);
});

test("the card carries a close button, and a load brings it back", () => {
  const indicator = readFileSync(
    new URL(
      "../src/features/loaded-models/loaded-models-indicator.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(indicator, /aria-label="Close loaded models"/);
  assert.match(
    indicator,
    /onClick=\{\(\) => setLoadedModelsDismissed\(true\)\}/,
  );
  // Reopened on the START of a load, so the card is up for as long as the toast.
  assert.match(
    indicator,
    /subscribeModelLifecycle\(\(\{ loading \}\) => \{\s*if \(loading\) \{\s*setLoadedModelsDismissed\(false\);/,
  );
  // And the gate reads it.
  assert.match(
    indicator,
    /const enabled = showIndicator && !dismissed && canShowIndicator\(pathname\)/,
  );
});

// Requested by name: hugeicons.com/icon/sparkle, which the free set exports as
// SparklesIcon.
test("the card is badged with the sparkle, not the brain", () => {
  const indicator = readFileSync(
    new URL(
      "../src/features/loaded-models/loaded-models-indicator.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(indicator, /icon=\{SparklesIcon\}/);
  assert.doesNotMatch(indicator, /AiBrain01Icon/);
});
