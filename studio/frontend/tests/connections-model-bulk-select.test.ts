// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Select all and Clear act on the models the search shows, and keep hidden picks.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const dialog = readSrc("features/chat/chat-providers-dialog.tsx");

function body(name: string): string {
  const match = dialog.match(
    new RegExp(`function ${name}\\(\\) \\{([\\s\\S]*?)\\n {2}\\}`),
  )?.[1];
  assert.ok(match, `${name} not found`);
  return match;
}

test("Select all only adds the models the search shows", () => {
  const selectAll = body("selectAllModels");
  assert.match(
    selectAll,
    /new Set\(\[\.\.\.prev, \.\.\.filteredAvailableModels\]\)/,
  );
  assert.doesNotMatch(selectAll, /availableModels\]/);
});

test("Clear only removes the models the search shows", () => {
  const clear = body("clearModelSelection");
  assert.match(clear, /new Set\(filteredAvailableModels\)/);
  assert.match(clear, /prev\.filter\(\(id\) => !visible\.has\(id\)\)/);
  assert.doesNotMatch(clear, /setSelectedModelIds\(\[\]\)/);
});

test("both model lists wire their buttons to the scoped handlers", () => {
  assert.equal(dialog.match(/onClick=\{selectAllModels\}/g)?.length, 2);
  assert.equal(
    dialog.match(/onClick=\{clearModelSelection\}|clearModelSelection\(\);/g)
      ?.length,
    2,
  );
});
