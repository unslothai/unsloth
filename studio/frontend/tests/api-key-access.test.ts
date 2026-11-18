// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { en } from "../src/i18n/locales/en.ts";
import { readSrc } from "./helpers/kit.ts";

const TAB = readSrc("features/settings/tabs/api-keys-tab.tsx");
const FORM = readSrc("features/settings/components/create-key-form.tsx");
const REVEAL = readSrc("features/settings/components/key-reveal-card.tsx");
const KEYS_API = readSrc("features/settings/api/api-keys.ts");
const ROW = readSrc("features/settings/components/api-key-row.tsx");

test("the create form is unchanged: every new key starts as full access", () => {
  assert.doesNotMatch(FORM, /ApiKeyAccess|accessInference/);
  assert.match(FORM, /createApiKey\(name\.trim\(\) \|\| defaultName, expiry\)/);
  assert.match(KEYS_API, /access: ApiKeyAccess = "full"/);
});

test("the access toggle sits beside the copy-now note and updates the shown key", () => {
  assert.match(
    REVEAL,
    /settings\.apiKeys\.copyNow"\)\}\s*<\/p>\s*<div[^>]*>\s*\{access && onAccessChange \?/,
  );
  assert.match(TAB, /updateApiKeyAccess\(revealedKey\.id, access\)/);
  assert.match(KEYS_API, /method: "PATCH"/);
});

test("regenerating a key keeps its access", () => {
  assert.match(TAB, /expiryDays\(revokeTarget\),\s*revokeTarget\.access,/);
});

test("an inference key is labelled Inference in the list", () => {
  assert.match(ROW, /apiKey\.access === "inference" &&/);
  assert.match(ROW, /settings\.apiKeys\.accessInferenceBadge/);
  assert.equal(en.settings.apiKeys.accessInferenceBadge, "Inference");
});
