// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  type LockSet,
  acquire,
  release,
} from "../src/features/chat/prompt-storage/mutation-lock.ts";

const empty: LockSet = new Set<string>();

test("a second caller cannot take a lock that is already held", () => {
  const [held, took] = acquire(empty, "p1");
  assert.equal(took, true);
  const [again, tookAgain] = acquire(held, "p1");
  assert.equal(tookAgain, false, "the delete ran while the save was in flight");
  assert.equal(again, held, "the loser must not replace the set and re-render");
});

test("locks are per row, so one row's save does not block another", () => {
  const [one] = acquire(empty, "p1");
  const [two, took] = acquire(one, "p2");
  assert.equal(took, true);
  assert.deepEqual([...two].sort(), ["p1", "p2"]);
});

// The bug this file exists for: the detail pane is keyed by row id, so selecting
// another row unmounts it. A lock held there came back false, and because the
// save is an unconditional PUT it could land after the DELETE and resurrect the
// row. Holding it above the key is what makes this sequence safe.
test("a lock survives the row switch that unmounts the pane", () => {
  let held: LockSet = empty;
  [held] = acquire(held, "p1"); // Save on p1 starts.
  // User selects p2, then p1 again. The pane remounts; the set does not.
  assert.equal(held.has("p1"), true, "the remounted pane would see no lock");
  const [, tookDelete] = acquire(held, "p1");
  assert.equal(tookDelete, false, "delete slipped past a save still in flight");
  held = release(held, "p1"); // The PUT settles.
  const [, tookAfter] = acquire(held, "p1");
  assert.equal(tookAfter, true, "the lock never came back");
});

test("releasing an id nobody holds is a no-op on the same set", () => {
  const [held] = acquire(empty, "p1");
  assert.equal(release(held, "p2"), held);
  assert.equal(release(empty, "p1"), empty);
});

test("the detail panes do not own a mutation lock", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Both panes are mounted with key={entry.id}, so a useState lock inside one
  // resets on every row switch. That is the defect; keep it from coming back.
  assert.doesNotMatch(
    source,
    /const \[pending, setPending\] = useState/,
    "a detail pane owns its lock again, which a row switch resets",
  );
  for (const prop of ["pending={mutatingIds.has(", "runMutation={runMutation}"]) {
    assert.equal(
      source.split(prop).length - 1,
      2,
      `${prop} should reach both PromptDetail and PromptListDetail`,
    );
  }
});
