// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { ACCOUNT_KEYED_PREFIXES } from "../src/lib/account-transition.ts";
import { readSrcAsync } from "./helpers/kit.ts";

const PREFIX = "unsloth_model_picker_intro_dismissed:";
const SOURCE = "components/assistant-ui/model-picker-announcement.tsx";

test("the dismissal is stored per account", async () => {
  const source = await readSrcAsync(SOURCE);
  assert.ok(source.includes(`"${PREFIX}"`));
  assert.ok(source.includes("browserAccountKey()"));
});

test("a dismissal survives signing out and back in", () => {
  assert.ok((ACCOUNT_KEYED_PREFIXES as readonly string[]).includes(PREFIX));
});

test("Reset preferences does not bring a closed card back", async () => {
  const source = await readSrcAsync("features/settings/tabs/general-tab.tsx");
  assert.ok(!source.includes("unsloth_model_picker_intro_dismissed"));
});

test("only the X dismisses: opening the picker leaves the card", async () => {
  const source = await readSrcAsync(SOURCE);
  const uses = source.match(/\bdismiss\b(?!ed)/g) ?? [];
  // The definition and the X's onClick, nothing else.
  assert.equal(uses.length, 2);
  assert.ok(source.includes("onClick={dismiss}"));
});

test("the welcome screen mounts it under the composer", async () => {
  const source = await readSrcAsync("components/assistant-ui/thread.tsx");
  assert.ok(source.includes("<ModelPickerAnnouncement"));
});
