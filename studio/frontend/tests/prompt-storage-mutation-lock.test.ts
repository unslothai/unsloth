// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  type LockSet,
  acquire,
  lockKey,
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
  // Above the forms, like the row locks: the forms are conditionally mounted, so
  // selecting a rail row while a create is out would otherwise hand a reopened
  // form a fresh false guard and let it mint a second id for the same draft.
  assert.equal(
    source.split("= useCreateGuard();").length - 1,
    2,
    "the create guard is not owned once per kind above the New forms",
  );
  const [beforeForms] = source.split("function NewPromptForm");
  assert.doesNotMatch(
    beforeForms,
    /const \{ creating, create \} = useCreateGuard\(\);/,
    "a New form owns its guard again, which a row switch resets",
  );
  for (const prop of ["creating={promptCreate.creating}", "creating={listCreate.creating}"]) {
    assert.ok(source.includes(prop), `${prop} should reach its New form`);
  }
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

// The parent keeps a row selected during render, so clearing the selection while
// the deleted row is still in promptEntries reselects it. The pane then renders
// an entry the backend no longer has until the refetch lands.
test("a delete clears its selection only after the row is gone", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const leadIns = source.split("onDeleted(entry.id);").slice(0, -1);
  assert.equal(leadIns.length, 2, "both detail panes should clear a deleted row");
  for (const before of leadIns) {
    assert.ok(
      before.lastIndexOf("await onRefresh();") >
        before.lastIndexOf("await runMutation("),
      "a delete pane clears the selection before its refresh lands",
    );
  }
  assert.doesNotMatch(
    source,
    /onDeleted\(entry\.id\);\n\s+onRefresh\(\);/,
    "the unawaited refresh is back",
  );
});

// DialogContent is overflow-hidden, so anything the dialog's own children add up
// to past its height is gone, not scrollable. A minimum height on the body has
// to predict the header and search block above it, and that block gets taller
// when its text wraps on a narrow dialog. At 320x320 the guess left the body
// 57px too tall and Use, Save and Run fell outside the clip.
test("the dialog body claims no height it has to guess", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.doesNotMatch(
    source,
    /min-h-\[[^\]]*dvh/,
    "the body floor is measured against the viewport again",
  );
  assert.match(
    source,
    /flex-1 min-h-0 overflow-y-auto px-4 sm:px-6/,
    "the body no longer shrinks to whatever the chrome leaves it",
  );
  // The row minimums are what actually keeps each pane usable.
  assert.match(source, /grid-rows-\[minmax\(132px,30%\)_minmax\(272px,1fr\)\]/);
});

// The New form's fields stay editable while its create is out, and Cancel can
// start a fresh draft during one. Resetting unconditionally when the request
// lands discarded text that never reached the server, which is the defect
// samePromptDraft already guards on the edit panes.
test("a create resets its draft only if it still holds what was sent", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(source, /samePromptDraft\(prev, submitted\) \? emptyPromptDraft\(\) : prev/);
  assert.match(source, /sameListDraft\(prev, submitted\) \? emptyListDraft\(\) : prev/);
  // The created path must not run the Cancel callback, which discards outright.
  assert.doesNotMatch(
    source,
    /onCreated\([^)]*\);\n\s+onClose\(\);/,
    "the created path closes through Cancel again, which resets unconditionally",
  );
  assert.equal(
    source.split("onCreated(id, submitted, mounted.current);").length - 1,
    2,
    "both create paths should hand the submitted snapshot up",
  );
});

// The comparison is the same one the edit panes use, so the empty-draft case has
// to behave: a create that lands after Cancel must not resurrect an empty form.
test("an empty draft does not match a submitted one", () => {
  assert.equal(samePromptDraft({ name: "", text: "" }, { name: "n", text: "t" }), false);
  assert.equal(sameListDraft({ name: "", items: ["", ""] }, { name: "l", items: ["a"] }), false);
});

// Selecting the Lists tab auto-selects its first row, so the detail pane mounts
// the editor with no click, and one controlled textarea per item makes that cost
// grow faster than the item count. The backend takes 10000 items in one list, so
// the editor has to be able to wait.
test("an oversized list waits to be asked before mounting its editor", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(source, /const EDITOR_ROW_LIMIT = \d+;/);
  const limit = Number(/const EDITOR_ROW_LIMIT = (\d+);/.exec(source)?.[1]);
  assert.ok(limit > 0 && limit < 500, `${limit} is not a limit that avoids the freeze`);
  // Latched, not recomputed: Add prompt on a list at the limit would otherwise
  // take it past and unmount the editor the user is typing in.
  assert.match(
    source,
    /const \[editorMounted, setEditorMounted\] = useState\(\n\s+\(\) => items\.length <= EDITOR_ROW_LIMIT,\n\s+\);/,
  );
  assert.doesNotMatch(
    source,
    /const editorMounted = \w+ \|\| items\.length <= EDITOR_ROW_LIMIT;/,
    "the mount decision is recomputed from the live length again",
  );
  // Deferring the editor must not narrow what a save, run or export carries.
  for (const readsFullItems of [
    "const filtered = items.filter((t) => t.trim());",
    "const runnableItems = items.filter((t) => t.trim());",
  ]) {
    assert.ok(source.includes(readsFullItems), `truncated: ${readsFullItems}`);
  }
});

// A create outlives the form that started it, and completion used to clear the
// search, move the selection and close whatever New form was open by then.
test("a finished create only moves the view its own form still owns", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.equal(
    source.split("onCreated(id, submitted, mounted.current);").length - 1,
    2,
    "a create path does not say whether its form is still on screen",
  );
  assert.equal(
    source.split("if (!fromOpenForm) return;").length - 1,
    2,
    "a completion still navigates after the user left the form",
  );
  // The draft still resets on a match, wherever the user went.
  const [, afterGuard] = source.split("const selectCreatedPrompt");
  assert.ok(
    afterGuard.indexOf("setNewPromptDraft(") < afterGuard.indexOf("if (!fromOpenForm) return;"),
    "the guard skips the draft reset, leaving a saved prompt marked unsaved",
  );
});

// searchQuery is shared by both tabs, so filtering one collection filters the
// hidden one too. Correcting the hidden tab's selection against that dropped the
// row it had, and clearing the query in an effect left one render to do it in.
test("only the visible tab's selection is corrected", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(source, /if \(activeTab === "prompts"\) \{\n\s+if \(filteredPrompts\.length === 0\)/);
  assert.match(source, /const selectTab = useCallback\(\(tab: Tab\) => \{/);
  assert.doesNotMatch(
    source,
    /\}, \[activeTab\]\);/,
    "the per-tab reset is an effect again, which renders once with the old query",
  );
  assert.doesNotMatch(source, /onClick=\{\(\) => setActiveTab\(tab\)\}/);
});

// main.tsx wraps the app in StrictMode, which replays an effect as setup,
// cleanup, setup on mount. A flag only initialised at the ref stays false from
// that first cleanup, so every create reported an unmounted form and the New
// form never closed on success.
test("the New form's mounted flag is set in effect setup", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.equal(
    source.split("mounted.current = true;").length - 1,
    2,
    "a New form only sets its mounted flag at the ref, which StrictMode clears",
  );
  assert.doesNotMatch(
    source,
    /useEffect\(\(\) => \(\) => \{ mounted\.current = false; \}, \[\]\);/,
    "the cleanup-only effect is back",
  );
});

// Prompts and prompt lists live in separate tables with independent ids, so an
// import can give one of each the same id. One set keyed on the raw id let a
// prompt's save disable the unrelated list's Save and Delete.
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
  // The prefix is part of the key, so reaching a list key needs a list.
  assert.notEqual(lockKey("prompt", "list:abc"), lockKey("list", "abc"));
  assert.notEqual(lockKey("list", "prompt:abc"), lockKey("prompt", "abc"));
});

test("both panes take their lock through lockKey", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.equal(source.split('runMutation(lockKey("prompt", entry.id)').length - 1, 2);
  assert.equal(source.split('runMutation(lockKey("list", entry.id)').length - 1, 2);
  assert.doesNotMatch(
    source,
    /runMutation\(entry\.id,/,
    "a raw id reaches the shared lock set again",
  );
  assert.doesNotMatch(
    source,
    /mutatingIds\.has\(selected(Prompt|List)\.id\)/,
    "a pane's pending state is read off a raw id again",
  );
});
