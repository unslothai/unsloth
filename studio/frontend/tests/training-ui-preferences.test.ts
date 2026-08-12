// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  LEGACY_TRAINING_PARAM_MODE_STORAGE_KEY,
  TRAINING_DATASET_PICKER_TAB_STORAGE_KEY,
  TRAINING_MODEL_PICKER_TAB_STORAGE_KEY,
  TRAINING_PARAM_MODE_STORAGE_KEY,
  TRAINING_UI_PREFERENCE_KEYS,
} from "../src/features/training/lib/training-ui-preferences.ts";

test("the training UI preference list contains every owned storage key once", () => {
  assert.deepEqual(TRAINING_UI_PREFERENCE_KEYS, [
    TRAINING_MODEL_PICKER_TAB_STORAGE_KEY,
    TRAINING_DATASET_PICKER_TAB_STORAGE_KEY,
    TRAINING_PARAM_MODE_STORAGE_KEY,
    LEGACY_TRAINING_PARAM_MODE_STORAGE_KEY,
  ]);
  assert.equal(
    new Set(TRAINING_UI_PREFERENCE_KEYS).size,
    TRAINING_UI_PREFERENCE_KEYS.length,
  );
});

test("reset all local preferences includes the training UI preference list", async () => {
  const source = await readFile(
    new URL("../src/features/settings/tabs/general-tab.tsx", import.meta.url),
    "utf8",
  );
  const start = source.indexOf("const PREFS_KEYS");
  const keys = source.slice(start, source.indexOf("];", start));
  assert.ok(keys.includes("...TRAINING_UI_PREFERENCE_KEYS"));
});
