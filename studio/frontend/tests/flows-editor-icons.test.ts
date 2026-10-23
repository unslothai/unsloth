// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import * as hugeicons from "@hugeicons/core-free-icons";
import * as local from "../src/features/flows/editor/icon-data.ts";

test("the Flows icon copies match the package glyphs", () => {
  const names = Object.keys(local);
  assert.ok(names.length > 20);
  for (const name of names) {
    assert.deepEqual(
      (local as Record<string, unknown>)[name],
      (hugeicons as Record<string, unknown>)[name],
      `${name} drifted from @hugeicons/core-free-icons; regenerate icon-data.ts`,
    );
  }
});
