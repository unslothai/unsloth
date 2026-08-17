// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// There is a genuine import cycle here, and the barrel's export order is what
// keeps it harmless:
//
//   loaded-models/index -> loaded-models-indicator -> features/settings/index
//     -> settings-dialog -> tabs/general-tab -> loaded-models/index
//
// general-tab's reset list dereferences LOADED_MODELS_PREFERENCE_KEYS at module
// scope, so if the barrel hands control to the indicator before the preference
// module has initialised, that read hits the temporal dead zone and startup
// fails with "Cannot access 'LOADED_MODELS_PREFERENCE_KEYS' before
// initialization". Reproduced in Vite dev, which serves native ESM: entering
// the loaded-models barrel first threw exactly that.
//
// Exporting the preference module first evaluates the constant before the
// indicator is touched, so either entry order is safe. A bundler may hide this,
// which is why it is pinned here rather than left to the build.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");
}

const BARREL = read("../src/features/loaded-models/index.ts");
const GENERAL_TAB = read("../src/features/settings/tabs/general-tab.tsx");
const INDICATOR = read(
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
  // If any of these three links is ever cut the ordering stops mattering, but
  // while they hold, the order above is the only thing making startup safe.
  assert.match(INDICATOR, /from "@\/features\/settings"/);
  assert.match(GENERAL_TAB, /from "@\/features\/loaded-models"/);
  assert.match(GENERAL_TAB, /LOADED_MODELS_PREFERENCE_KEYS\./);
});

test("general-tab reads the keys at module scope, not inside a component", () => {
  // A read inside a component body would run long after both modules settled,
  // and the ordering would be cosmetic. It is not: this is a top-level const.
  const keysAt = GENERAL_TAB.indexOf("LOADED_MODELS_PREFERENCE_KEYS.show");
  const firstComponent = GENERAL_TAB.search(/\nexport function |\nfunction \w+\(/);
  assert.ok(keysAt !== -1, "expected the reset list to name the keys");
  assert.ok(
    firstComponent === -1 || keysAt < firstComponent,
    "the keys are read at module scope, so evaluation order decides the outcome",
  );
});
