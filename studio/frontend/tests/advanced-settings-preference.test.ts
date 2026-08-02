// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const {
  ADVANCED_SETTINGS_OPEN_KEY,
  readAdvancedSettingsOpen,
  saveAdvancedSettingsOpen,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

test("a fresh profile keeps the advanced section closed", () => {
  store.delete(ADVANCED_SETTINGS_OPEN_KEY);
  assert.equal(readAdvancedSettingsOpen(), false);
});

test("opening it once is remembered", () => {
  saveAdvancedSettingsOpen(true);
  // What the next model, quant, or reload reads.
  assert.equal(readAdvancedSettingsOpen(), true);
});

test("closing it again is remembered too", () => {
  saveAdvancedSettingsOpen(true);
  saveAdvancedSettingsOpen(false);
  assert.equal(readAdvancedSettingsOpen(), false);
});

test("an unreadable value falls back to closed", () => {
  store.set(ADVANCED_SETTINGS_OPEN_KEY, "yes");
  assert.equal(readAdvancedSettingsOpen(), false);
});
