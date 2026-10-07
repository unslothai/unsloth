// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  type LockSet,
  acquire,
  lockKey,
  release,
  sameListDraft,
  samePromptDraft,
} from "../src/features/chat/prompt-storage/mutation-lock.ts";

import { readSrc } from "./helpers/kit.ts";

const PROMPT_STORAGE_DIALOG = readSrc("features/chat/prompt-storage/prompt-storage-dialog.tsx");

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

// The pane is keyed by row id, so a lock held inside it resets and a late PUT resurrects rows.
test("a lock survives the row switch that unmounts the pane", () => {
  let held: LockSet = empty;
  [held] = acquire(held, "p1");
  // Selecting p2 then p1 remounts the pane; the set does not.
  assert.equal(held.has("p1"), true, "the remounted pane would see no lock");
  const [, tookDelete] = acquire(held, "p1");
  assert.equal(tookDelete, false, "delete slipped past a save still in flight");
  held = release(held, "p1");
  const [, tookAfter] = acquire(held, "p1");
  assert.equal(tookAfter, true, "the lock never came back");
});

test("releasing an id nobody holds is a no-op on the same set", () => {
  const [held] = acquire(empty, "p1");
  assert.equal(release(held, "p2"), held);
  assert.equal(release(empty, "p1"), empty);
});

test("the detail panes do not own a mutation lock", async () => {
  // Panes are keyed by entry.id, so a useState lock resets on every row switch.
  assert.doesNotMatch(
    PROMPT_STORAGE_DIALOG,
    /const \[pending, setPending\] = useState/,
    "a detail pane owns its lock again, which a row switch resets",
  );
  for (const prop of ["pending={mutatingIds.has(", "runMutation={runMutation}"]) {
    assert.equal(
      PROMPT_STORAGE_DIALOG.split(prop).length - 1,
      2,
      `${prop} should reach both PromptDetail and PromptListDetail`,
    );
  }
});

// React may defer a functional updater, so the ref, not the updater's outcome, decides.
test("the lock decides from the ref, not from a scheduled updater", async () => {
  assert.match(PROMPT_STORAGE_DIALOG, /const mutatingRef = useRef<ReadonlySet<string>>/);
  assert.match(
    PROMPT_STORAGE_DIALOG,
    /const \[held, started\] = acquire\(mutatingRef\.current, id\);/,
    "the lock is decided from state again, which can be stale",
  );
  assert.doesNotMatch(
    PROMPT_STORAGE_DIALOG,
    /let started = false;/,
    "the deferred-updater pattern is back",
  );
});

// The editor stays usable during a save, so only clear an unchanged draft.
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

// Creates have no id to lock on yet; unguarded they duplicated and swallowed failures.
test("both create paths are guarded and report failure", async () => {
  // Forms are conditionally mounted, so the guard lives above them.
  assert.equal(
    PROMPT_STORAGE_DIALOG.split("= useCreateGuard();").length - 1,
    2,
    "the create guard is not owned once per kind above the New forms",
  );
  const [beforeForms] = PROMPT_STORAGE_DIALOG.split("function NewPromptForm");
  assert.doesNotMatch(
    beforeForms,
    /const \{ creating, create \} = useCreateGuard\(\);/,
    "a New form owns its guard again, which a row switch resets",
  );
  for (const prop of ["creating={promptCreate.creating}", "creating={listCreate.creating}"]) {
    assert.ok(PROMPT_STORAGE_DIALOG.includes(prop), `${prop} should reach its New form`);
  }
  assert.equal(
    PROMPT_STORAGE_DIALOG.split("disabled={creating ").length - 1,
    2,
    "a Save button stays live while its create is in flight",
  );
  for (const message of ["Could not create prompt", "Could not create list"]) {
    assert.ok(PROMPT_STORAGE_DIALOG.includes(message), `a failed create is silent: ${message}`);
  }
  assert.match(PROMPT_STORAGE_DIALOG, /if \(creatingRef\.current\) return;/);
});

// Clearing the draft before refetch flashes the pre-save text.
test("a save clears its draft only after the refreshed entry is in", async () => {
  assert.equal(
    PROMPT_STORAGE_DIALOG.split("await onRefresh();\n        onSaved(submitted);").length - 1,
    2,
    "a save pane drops the draft before the refresh lands",
  );
  assert.doesNotMatch(
    PROMPT_STORAGE_DIALOG,
    /onSaved\(submitted\);\n\s+onRefresh\(\);/,
    "the unawaited refresh is back",
  );
});

// The parent reselects during render while the deleted row is still listed.
test("a delete clears its selection only after the row is gone", async () => {
  const leadIns = PROMPT_STORAGE_DIALOG.split("onDeleted(entry.id);").slice(0, -1);
  assert.equal(leadIns.length, 2, "both detail panes should clear a deleted row");
  for (const before of leadIns) {
    assert.ok(
      before.lastIndexOf("await onRefresh();") >
        before.lastIndexOf("await runMutation("),
      "a delete pane clears the selection before its refresh lands",
    );
  }
  assert.doesNotMatch(
    PROMPT_STORAGE_DIALOG,
    /onDeleted\(entry\.id\);\n\s+onRefresh\(\);/,
    "the unawaited refresh is back",
  );
});

// DialogContent is overflow-hidden, so a guessed min height clips controls on narrow dialogs.
test("the dialog body claims no height it has to guess", async () => {
  assert.doesNotMatch(
    PROMPT_STORAGE_DIALOG,
    /min-h-\[[^\]]*dvh/,
    "the body floor is measured against the viewport again",
  );
  assert.match(
    PROMPT_STORAGE_DIALOG,
    /flex-1 min-h-0 overflow-y-auto px-4 sm:px-6/,
    "the body no longer shrinks to whatever the chrome leaves it",
  );
  assert.match(PROMPT_STORAGE_DIALOG, /grid-rows-\[minmax\(132px,30%\)_minmax\(272px,1fr\)\]/);
});

// Fields stay editable during a create, so only reset a draft that still matches.
test("a create resets its draft only if it still holds what was sent", async () => {
  assert.match(PROMPT_STORAGE_DIALOG, /samePromptDraft\(prev, submitted\) \? emptyPromptDraft\(\) : prev/);
  assert.match(PROMPT_STORAGE_DIALOG, /sameListDraft\(prev, submitted\) \? emptyListDraft\(\) : prev/);
  assert.doesNotMatch(
    PROMPT_STORAGE_DIALOG,
    /onCreated\([^)]*\);\n\s+onClose\(\);/,
    "the created path closes through Cancel again, which resets unconditionally",
  );
  assert.equal(
    PROMPT_STORAGE_DIALOG.split("onCreated(id, submitted, mounted.current);").length - 1,
    2,
    "both create paths should hand the submitted snapshot up",
  );
});

test("an empty draft does not match a submitted one", () => {
  assert.equal(samePromptDraft({ name: "", text: "" }, { name: "n", text: "t" }), false);
  assert.equal(sameListDraft({ name: "", items: ["", ""] }, { name: "l", items: ["a"] }), false);
});

// One textarea per item scales badly and lists allow 10000 items.
test("an oversized list waits to be asked before mounting its editor", async () => {
  assert.match(PROMPT_STORAGE_DIALOG, /const EDITOR_ROW_LIMIT = \d+;/);
  const limit = Number(/const EDITOR_ROW_LIMIT = (\d+);/.exec(PROMPT_STORAGE_DIALOG)?.[1]);
  assert.ok(limit > 0 && limit < 500, `${limit} is not a limit that avoids the freeze`);
  // Latched so Add prompt at the limit does not unmount the active editor.
  assert.match(
    PROMPT_STORAGE_DIALOG,
    /const \[editorMounted, setEditorMounted\] = useState\(\n\s+\(\) => items\.length <= EDITOR_ROW_LIMIT,\n\s+\);/,
  );
  assert.doesNotMatch(
    PROMPT_STORAGE_DIALOG,
    /const editorMounted = \w+ \|\| items\.length <= EDITOR_ROW_LIMIT;/,
    "the mount decision is recomputed from the live length again",
  );
  for (const readsFullItems of [
    "const filtered = items.filter((t) => t.trim());",
    "const runnableItems = items.filter((t) => t.trim());",
  ]) {
    assert.ok(PROMPT_STORAGE_DIALOG.includes(readsFullItems), `truncated: ${readsFullItems}`);
  }
});

// A create outlives its form, so completion only touches the view its form still owns.
test("a finished create only moves the view its own form still owns", async () => {
  assert.equal(
    PROMPT_STORAGE_DIALOG.split("onCreated(id, submitted, mounted.current);").length - 1,
    2,
    "a create path does not say whether its form is still on screen",
  );
  assert.equal(
    PROMPT_STORAGE_DIALOG.split("if (!fromOpenForm) return;").length - 1,
    2,
    "a completion still navigates after the user left the form",
  );
  const [, afterGuard] = PROMPT_STORAGE_DIALOG.split("const selectCreatedPrompt");
  assert.ok(
    afterGuard.indexOf("setNewPromptDraft(") < afterGuard.indexOf("if (!fromOpenForm) return;"),
    "the guard skips the draft reset, leaving a saved prompt marked unsaved",
  );
});

// searchQuery is shared across tabs, so only correct the visible tab's selection.
test("only the visible tab's selection is corrected", async () => {
  assert.match(PROMPT_STORAGE_DIALOG, /if \(activeTab === "prompts"\) \{\n\s+if \(filteredPrompts\.length === 0\)/);
  assert.match(PROMPT_STORAGE_DIALOG, /const selectTab = useCallback\(\(tab: Tab\) => \{/);
  assert.doesNotMatch(
    PROMPT_STORAGE_DIALOG,
    /\}, \[activeTab\]\);/,
    "the per-tab reset is an effect again, which renders once with the old query",
  );
  assert.doesNotMatch(PROMPT_STORAGE_DIALOG, /onClick=\{\(\) => setActiveTab\(tab\)\}/);
});

// StrictMode replays effect setup/cleanup, so the mounted flag must be set in setup.
test("the New form's mounted flag is set in effect setup", async () => {
  assert.equal(
    PROMPT_STORAGE_DIALOG.split("mounted.current = true;").length - 1,
    2,
    "a New form only sets its mounted flag at the ref, which StrictMode clears",
  );
  assert.doesNotMatch(
    PROMPT_STORAGE_DIALOG,
    /useEffect\(\(\) => \(\) => \{ mounted\.current = false; \}, \[\]\);/,
    "the cleanup-only effect is back",
  );
});

// Prompts and lists have independent ids, so lock keys are prefixed.
test("a prompt and a list with one id do not share a lock", () => {
  let held: LockSet = new Set<string>();
  [held] = acquire(held, lockKey("prompt", "x"));
  assert.equal(held.has(lockKey("list", "x")), false, "the list is locked too");
  const [, tookList] = acquire(held, lockKey("list", "x"));
  assert.equal(tookList, true, "the list could not start its own mutation");
  const [, tookPromptAgain] = acquire(held, lockKey("prompt", "x"));
  assert.equal(tookPromptAgain, false, "the prompt's own lock stopped working");
});

test("no id can be crafted to collide across the two kinds", () => {
  assert.notEqual(lockKey("prompt", "list:abc"), lockKey("list", "abc"));
  assert.notEqual(lockKey("list", "prompt:abc"), lockKey("prompt", "abc"));
});

test("both panes take their lock through lockKey", async () => {
  assert.equal(PROMPT_STORAGE_DIALOG.split('runMutation(lockKey("prompt", entry.id)').length - 1, 2);
  assert.equal(PROMPT_STORAGE_DIALOG.split('runMutation(lockKey("list", entry.id)').length - 1, 2);
  assert.doesNotMatch(
    PROMPT_STORAGE_DIALOG,
    /runMutation\(entry\.id,/,
    "a raw id reaches the shared lock set again",
  );
  assert.doesNotMatch(
    PROMPT_STORAGE_DIALOG,
    /mutatingIds\.has\(selected(Prompt|List)\.id\)/,
    "a pane's pending state is read off a raw id again",
  );
});
