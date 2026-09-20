// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { stripTypeScriptTypes } from "node:module";
import test from "node:test";
import type { ApiModelOverride } from "../src/features/model-picker/api/model-overrides.ts";
import { readSrc } from "./helpers/kit.ts";

const source = readSrc(
  "features/api-monitor/components/saved-model-settings.tsx",
);
const describeOverride = new Function(
  `${stripTypeScriptTypes(source.slice(source.indexOf("function plural("), source.indexOf("export function SavedModelSettingsPanel")))}\nreturn describeOverride;`,
)() as (override: ApiModelOverride) => string[];

for (const budget of [0, 32]) {
  test(`saved reasoning budget ${budget} is visible in the API load summary`, () => {
    assert.ok(
      describeOverride({ reasoning_budget: budget }).includes(
        `reasoning budget ${budget}`,
      ),
    );
  });
}

test("a saved budget message alone does not appear as app defaults", () => {
  assert.ok(
    describeOverride({
      reasoning_budget_message: "  Conclude now.  ",
    }).includes("custom reasoning budget message"),
  );
});

test("absent reasoning overrides leave the default summary unchanged", () => {
  assert.deepEqual(describeOverride({}), []);
});
