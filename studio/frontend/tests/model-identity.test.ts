// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { normalizeModelIdentity } from "../src/features/hub/lib/model-identity.ts";

test("normalizes relative Windows separators without changing path case", () => {
  assert.equal(
    normalizeModelIdentity(String.raw`.\models\demo`),
    normalizeModelIdentity("./models/demo"),
  );
  assert.equal(
    normalizeModelIdentity(String.raw`..\Models\Demo\\`),
    "../Models/Demo",
  );
  assert.notEqual(
    normalizeModelIdentity("./Models/Demo"),
    normalizeModelIdentity("./models/demo"),
  );
});

test("preserves existing platform and Hub identity rules", () => {
  assert.equal(
    normalizeModelIdentity(String.raw`C:\Models\Demo\\`),
    "c:/models/demo",
  );
  assert.equal(
    normalizeModelIdentity(String.raw`\\Server\Share\Models\Demo\\`),
    "//server/share/models/demo",
  );
  assert.equal(normalizeModelIdentity("Org/Model"), "org/model");
});
