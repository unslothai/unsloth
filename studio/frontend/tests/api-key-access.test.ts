// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { en } from "../src/i18n/locales/en.ts";
import { readSrc } from "./helpers/kit.ts";

const TAB = readSrc("features/settings/tabs/api-keys-tab.tsx");
const FORM = readSrc("features/settings/components/create-key-form.tsx");
const KEYS_API = readSrc("features/settings/api/api-keys.ts");
const ROW = readSrc("features/settings/components/api-key-row.tsx");

test("a new key is full access unless Inference only is picked", () => {
  assert.match(FORM, /useState<ApiKeyAccess>\("full"\)/);
  assert.match(KEYS_API, /access: ApiKeyAccess = "full"/);
  assert.match(
    KEYS_API,
    /\.\.\.\(access === "inference" \? \{ access \} : \{\}\)/,
  );
  assert.equal(en.settings.apiKeys.accessInference, "Inference only");
});

test("regenerating a key keeps its access", () => {
  assert.match(TAB, /expiryDays\(revokeTarget\),\s*revokeTarget\.access,/);
});

test("an inference only key is labelled in the list", () => {
  assert.match(ROW, /apiKey\.access === "inference" &&/);
});
