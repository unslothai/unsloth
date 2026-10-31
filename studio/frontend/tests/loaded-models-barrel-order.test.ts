// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Import cycle: loaded-models/index -> indicator -> settings -> general-tab -> loaded-models.
// Exporting the preference module first avoids a TDZ error on its top-level const.

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

const BARREL = readText("../src/features/loaded-models/index.ts");
const GENERAL_TAB = readText("../src/features/settings/tabs/general-tab.tsx");
const INDICATOR = readText(
  "../src/features/loaded-models/loaded-models-indicator.tsx",
);

test("the preference module is exported before the indicator", () => {
  const pref = BARREL.indexOf("./show-loaded-models-pref");
  const indicator = BARREL.indexOf("./loaded-models-indicator");
  assert.ok(pref !== -1 && indicator !== -1, "expected both exports");
  assert.ok(
    pref < indicator,
    "show-loaded-models-pref must be evaluated first, or general-tab reads the keys in the temporal dead zone",
  );
});

test("the cycle this order defends against is still present", () => {
  assert.match(INDICATOR, /from "@\/features\/settings"/);
  assert.match(GENERAL_TAB, /from "@\/features\/loaded-models"/);
  assert.match(GENERAL_TAB, /LOADED_MODELS_PREFERENCE_KEYS\./);
});

test("general-tab reads the keys at module scope, not inside a component", () => {
  const keysAt = GENERAL_TAB.indexOf("LOADED_MODELS_PREFERENCE_KEYS.show");
  const firstComponent = GENERAL_TAB.search(/\nexport function |\nfunction \w+\(/);
  assert.ok(keysAt !== -1, "expected the reset list to name the keys");
  assert.ok(
    firstComponent === -1 || keysAt < firstComponent,
    "the keys are read at module scope, so evaluation order decides the outcome",
  );
});
