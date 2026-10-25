// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const form = readSrc("features/settings/components/create-key-form.tsx");

// The helper as written in the form, run here without pulling React into the test.
const helperSource = form.slice(
  form.indexOf("export function nextDefaultTokenNumber("),
  form.indexOf("\nexport function CreateKeyForm("),
);
const nextDefaultTokenNumber = new Function(
  `${helperSource
    .replace("export function", "function")
    .replace(/\(\s*names: readonly string\[\],\s*pattern: RegExp,?\s*\): number/, "(names, pattern)")}
  return nextDefaultTokenNumber;`,
)() as (names: readonly string[], pattern: RegExp) => number;

const NEW_TOKEN = /^New Token (\d+)$/;

test("an unnamed token takes the number after every New Token N in use", () => {
  assert.equal(nextDefaultTokenNumber([], NEW_TOKEN), 1);
  assert.equal(nextDefaultTokenNumber(["production", "ci"], NEW_TOKEN), 1);
  assert.equal(nextDefaultTokenNumber(["New Token 1", "New Token 2"], NEW_TOKEN), 3);
  // A gap is not refilled: a revoked "New Token 2" never comes back as a new token's name.
  assert.equal(nextDefaultTokenNumber(["New Token 1", "New Token 3"], NEW_TOKEN), 4);
  assert.equal(nextDefaultTokenNumber(["New Token 12x", "my New Token 9"], NEW_TOKEN), 1);
});

test("an empty name submits the default, and the button is never disabled for it", () => {
  assert.match(form, /createApiKey\(name\.trim\(\) \|\| defaultName, expiry\)/);
  assert.match(form, /placeholder=\{t\("settings.apiKeys.tokenNamePlaceholder"\)\}/);
  assert.match(form, /disabled=\{loading\}/);
  assert.match(form, /variant="dark"/);
});
