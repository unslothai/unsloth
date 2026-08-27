// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  type LockSet,
  acquire,
  release,
  sameListDraft,
  samePromptDraft,
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

// Codex, on the first version of the lock: reading the outcome of a functional
// updater straight after scheduling it is not sound, because React may defer the
// updater. The caller then skips the request while the id still gets acquired
// later during render, with no finally left to release it, and the row's Save and
// Delete stay disabled for good. The ref is the authority for that reason.
test("the lock decides from the ref, not from a scheduled updater", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(source, /const mutatingRef = useRef<ReadonlySet<string>>/);
  assert.match(
    source,
    /const \[held, started\] = acquire\(mutatingRef\.current, id\);/,
    "the lock is decided from state again, which can be stale",
  );
  assert.doesNotMatch(
    source,
    /let started = false;/,
    "the deferred-updater pattern is back",
  );
});

// A save is async and the editor stays usable while it runs, so clearing the
// draft on success can discard whatever was typed in the meantime.
test("a draft that moved on while saving is not cleared", () => {
  const submitted = { name: "notes", text: "first" };
  assert.equal(samePromptDraft({ ...submitted }, submitted), true);
  assert.equal(
    samePromptDraft({ name: "notes", text: "first, then more" }, submitted),
    false,
    "the newer edit would be thrown away",
  );
  assert.equal(
    samePromptDraft({ name: "renamed", text: "first" }, submitted),
    false,
  );
});

test("list drafts compare by items, not by identity", () => {
  const submitted = { name: "l", items: ["a", "b"] };
  assert.equal(sameListDraft({ name: "l", items: ["a", "b"] }, submitted), true);
  assert.equal(sameListDraft({ name: "l", items: ["a", "c"] }, submitted), false);
  assert.equal(
    sameListDraft({ name: "l", items: ["a", "b", "c"] }, submitted),
    false,
    "an item appended while saving would be thrown away",
  );
  assert.equal(sameListDraft({ name: "l", items: ["a"] }, submitted), false);
});

// Creating a row cannot use the by-id lock, because the id does not exist until
// the request is built. Both New forms awaited an unguarded PUT: a second click
// minted a second id and stored a duplicate, and a rejection was unhandled, so a
// failed create looked exactly like a successful one.
test("both create paths are guarded and report failure", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.equal(
    source.split("const { creating, create } = useCreateGuard();").length - 1,
    2,
    "a New form can still start a second create over the first",
  );
  assert.equal(
    source.split("disabled={creating ").length - 1,
    2,
    "a Save button stays live while its create is in flight",
  );
  for (const message of ["Could not create prompt", "Could not create list"]) {
    assert.ok(source.includes(message), `a failed create is silent: ${message}`);
  }
  // The ref decides, for the reason the mutation lock's does.
  assert.match(source, /if \(creatingRef\.current\) return;/);
});

// The draft is what covers the entry the pane still holds, which is the pre-save
// copy until the list is refetched. Clearing it first flashes the old text back.
test("a save clears its draft only after the refreshed entry is in", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.equal(
    source.split("await onRefresh();\n        onSaved(submitted);").length - 1,
    2,
    "a save pane drops the draft before the refresh lands",
  );
  assert.doesNotMatch(
    source,
    /onSaved\(submitted\);\n\s+onRefresh\(\);/,
    "the unawaited refresh is back",
  );
});
