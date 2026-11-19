// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { ACCOUNT_CHROME_KEYS } from "../src/lib/account-transition.ts";
import { readSrcAsync } from "./helpers/kit.ts";

const KEY = "unsloth_model_picker_intro_dismissed";

test("the announcement owns the dismissal key", async () => {
  const source = await readSrcAsync(
    "components/assistant-ui/model-picker-announcement.tsx",
  );
  assert.ok(source.includes(`"${KEY}"`));
});

test("Reset preferences brings the card back", async () => {
  const source = await readSrcAsync("features/settings/tabs/general-tab.tsx");
  const start = source.indexOf("const PREFS_KEYS");
  const keys = source.slice(start, source.indexOf("];", start));
  assert.ok(keys.includes(`"${KEY}"`), `${KEY} missing from PREFS_KEYS`);
});

test("a dismissal survives switching accounts", () => {
  assert.ok(ACCOUNT_CHROME_KEYS.has(KEY));
});

test("the welcome screen mounts it under the composer", async () => {
  const source = await readSrcAsync("components/assistant-ui/thread.tsx");
  assert.ok(source.includes("<ModelPickerAnnouncement"));
});
