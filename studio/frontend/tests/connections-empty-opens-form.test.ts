// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const dialog = readSrc("features/chat/chat-providers-dialog.tsx");

test("an empty connection list opens the add-connection form", () => {
  assert.match(
    dialog,
    /if \(syncedProviders\.length === 0 && selectableRegistry\.length > 0\) \{\s*(?:\/\/[^\n]*\n\s*)*setPage\("form"\);/,
  );
  assert.match(
    dialog,
    /const selectableRegistry = registryRows\.filter\(\(entry\) => !entry\.hidden\);\s*setRegistry\(selectableRegistry\);/,
  );

  assert.doesNotMatch(
    dialog,
    /if \(providers\.length === 0 &&[^\n]*\)\s*\{\s*(?:\/\/[^\n]*\n\s*)*setPage\("form"\);/,
  );
});

test("the add-connection form only opens itself once", () => {
  assert.match(
    dialog,
    /if \(!autoOpenedAddFormRef\.current\) \{\s*autoOpenedAddFormRef\.current = true;/,
  );
  assert.match(dialog, /const autoOpenedAddFormRef = useRef\(false\);/);

  assert.doesNotMatch(dialog, /autoOpenedAddFormRef\.current = false;/);
});

test("the form's back arrow still returns to the list", () => {
  assert.match(
    dialog,
    /function closeForm\(\) \{\s*resetForm\(\);\s*autoOpenedAddFormRef\.current = true;\s*setPage\("list"\);/,
  );
  assert.match(dialog, /No connections yet/);
});

test("navigating before the sync lands consumes the auto-open", () => {
  const calls = [...dialog.matchAll(/setPage\("(?:list|form)"\)/g)];
  assert.equal(calls.length, 6);
  for (const call of calls) {
    const before = dialog.slice(
      Math.max(0, (call.index ?? 0) - 200),
      call.index,
    );
    assert.match(before, /autoOpenedAddFormRef\.current = true;/);
  }
});
