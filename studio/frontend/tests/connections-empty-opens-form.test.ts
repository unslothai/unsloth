// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > Connections used to open onto an empty list that never said what
// the page was for. It now opens the form when the first sync comes back with
// no connections. These pin the parts that make it land where a user expects.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const dialog = readFileSync(
  new URL("../src/features/chat/chat-providers-dialog.tsx", import.meta.url),
  "utf8",
);

test("an empty connection list opens the add-connection form", () => {
  // Both operands matter: `syncedProviders` is the backend answer, and
  // `selectableRegistry` stops a form with an empty dropdown from opening.
  assert.match(
    dialog,
    /if \(syncedProviders\.length === 0 && selectableRegistry\.length > 0\) \{\s*(?:\/\/[^\n]*\n\s*)*setPage\("form"\);/,
  );
  assert.match(
    dialog,
    /const selectableRegistry = registryRows\.filter\(\(entry\) => !entry\.hidden\);\s*setRegistry\(selectableRegistry\);/,
  );

  // Reading `providers` would decide against the previous render's props.
  assert.doesNotMatch(
    dialog,
    /if \(providers\.length === 0 &&[^\n]*\)\s*\{\s*(?:\/\/[^\n]*\n\s*)*setPage\("form"\);/,
  );
});

test("the add-connection form only opens itself once", () => {
  // Without the latch, the focus re-sync would pull the user back to the form.
  assert.match(
    dialog,
    /if \(!autoOpenedAddFormRef\.current\) \{\s*autoOpenedAddFormRef\.current = true;/,
  );
  assert.match(dialog, /const autoOpenedAddFormRef = useRef\(false\);/);

  // Resetting it would make the latch per render instead of per visit.
  assert.doesNotMatch(dialog, /autoOpenedAddFormRef\.current = false;/);
});

test("the form's back arrow still returns to the list", () => {
  // A starting point, not a trap: the way back and its copy both stay.
  assert.match(
    dialog,
    /function closeForm\(\) \{\s*resetForm\(\);\s*autoOpenedAddFormRef\.current = true;\s*setPage\("list"\);/,
  );
  assert.match(dialog, /No connections yet/);
});

test("navigating before the sync lands consumes the auto-open", () => {
  // The Add connection row stays live while the first sync runs, so a user can
  // open the form and go back before it lands. Every path that moves the page
  // latches the one-shot, or the sync would drag them back into the form.
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
