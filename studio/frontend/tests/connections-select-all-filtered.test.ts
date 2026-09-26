// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const dialog = readSrc("features/chat/chat-providers-dialog.tsx");

test("Select all only adds the models the search shows", () => {
  const body = dialog.match(
    /function selectAllModels\(\) \{([\s\S]*?)\n {2}\}/,
  )?.[1];
  assert.ok(body, "selectAllModels not found");
  assert.match(body, /filteredAvailableModels/);
  assert.match(body, /\.\.\.prev/);
  assert.doesNotMatch(body, /\.\.\.availableModels\b/);
});
