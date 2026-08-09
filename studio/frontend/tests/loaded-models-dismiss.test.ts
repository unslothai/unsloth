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
  // And the gate reads it. Reachability is hoisted so tracking can share it.
  assert.match(
    indicator,
    /const enabled = showIndicator && !dismissed && reachable;/,
  );
});

// Requested by name: hugeicons.com/icon/sparkle. Singular, so NOT the free
// set's SparklesIcon (two stars) nor lib/sparkles-icon, which is a shield.
test("the card is badged with the single sparkle, not the brain", () => {
  const indicator = readFileSync(
    new URL(
      "../src/features/loaded-models/loaded-models-indicator.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(indicator, /icon=\{SparkleIcon\}/);
  assert.match(indicator, /from "@\/lib\/sparkle-icon"/);
  assert.doesNotMatch(indicator, /AiBrain01Icon|SparklesIcon/);
});

// Releasing the weights is not the same act as closing the card, so it must not
// wear the same X.
test("a row ejects with the eject glyph, the header closes with an X", () => {
  const indicator = readFileSync(
    new URL(
      "../src/features/loaded-models/loaded-models-indicator.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const row = indicator.slice(
    indicator.indexOf("function LoadedModelRow"),
    indicator.indexOf("export function LoadedModelsIndicator"),
  );
  assert.match(row, /icon=\{RemoveCircleIcon\}/);
  assert.doesNotMatch(row, /icon=\{Cancel01Icon\}/);
  // The close button keeps the X, as the Live monitor's does.
  const header = indicator.slice(
    indicator.indexOf('aria-label="Close loaded models"'),
  );
  assert.match(header, /icon=\{Cancel01Icon\}/);
});

// Recording had to widen past `enabled` so a card the user closed still hears
// the load that reopens it. It widened one step too far: `canShowIndicator`
// carries the auth gate as well as the hidden routes, so tracking on the
// preference alone polled four protected endpoints every 5s on /login, and each
// 401 ran authFetch's refresh-then-redirect ladder against no session at all.
// Asserted by reading the source, since the node suite has no DOM to mount in.
test("recording follows the route and auth gate, but not the dismissal", () => {
  const INDICATOR = readFileSync(
    new URL(
      "../src/features/loaded-models/loaded-models-indicator.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // The auth gate lives in canShowIndicator, so `reachable` is what carries it.
  assert.match(INDICATOR, /const reachable = canShowIndicator\(pathname\);/);
  assert.match(
    INDICATOR,
    /hasAuthToken\(\) && !mustChangePassword\(\)/,
    "canShowIndicator must still be the thing that gates on auth",
  );
  const call = INDICATOR.slice(
    INDICATOR.indexOf("useLoadedModels("),
    INDICATOR.indexOf("useLoadedModels(") + 200,
  );
  assert.match(call, /showIndicator && reachable/, "track must be gated too");
  assert.doesNotMatch(
    call,
    /\n\s*showIndicator,\n/,
    "the preference alone is what polled /login",
  );
  // And dismissal stays out of it, or a closed card could never reopen itself.
  assert.doesNotMatch(call, /dismissed/);
});
