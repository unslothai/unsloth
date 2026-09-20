// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The project Edit dialog: one stack of name, instructions and a grouped list of source folders.
// What it must not do is lose an edit: the two writes are separate, and the draft belongs to the
// project that is open, not the one that was.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrcAsync } from "./helpers/kit.ts";

const DIALOG = await readSrcAsync(
  "features/chat/components/edit-project-dialog.tsx",
);
const FOLDERS = await readSrcAsync(
  "features/rag/components/linked-folders-manager.tsx",
);

// The folders used to be a settings panel (heading, blurb and its own button) dropped inside a
// muted box under two fields that were bordered cards. Now all three read as one stack.
test("the dialog shows its folders as a grouped card, not a nested panel", () => {
  assert.match(
    DIALOG,
    /<LinkedFoldersManager\n\s*scope=\{\{ type: "project", id: project\.id \}\}\n\s*variant="card"\n\s*\/>/,
  );
  assert.ok(
    !DIALOG.includes('compact={true}'),
    "the dialog still asks for the panel layout",
  );
  assert.match(DIALOG, /<h3 className="[^"]*">\n\s*Source folders\n\s*<\/h3>/);
});

// The card's last row adds a folder, where the panel had a button in its header.
test("the card ends in an Add folder row that says when it cannot", () => {
  const card = FOLDERS.slice(
    FOLDERS.indexOf('if (variant === "card") {'),
    FOLDERS.indexOf("  return (\n    <section"),
  );
  assert.ok(card.length > 0, "the card layout moved");
  assert.match(card, /<span>Add folder<\/span>/);
  assert.match(card, /disabled=\{!manager\.desktopSupported \|\| manager\.mutating\}/);
  assert.match(card, /onClick=\{\(\) => void manager\.link\(\)\}/);
  // A folder list with no scope has nothing to link into, so it shows no row that cannot work.
  assert.match(card, /\{scope \? \(\n\s*<button/);
  // And the reason the row is dead is only spelled out when it is.
  assert.match(card, /\{!manager\.desktopSupported \? \(/);
});

// Both layouts drive the same hook and the same actions: a card row that dropped Sync changes or
// the two unlinks would be a quieter list that can do less.
test("a card row keeps every action the panel row has", () => {
  assert.match(FOLDERS, /function folderMenu\(/);
  assert.equal(
    (FOLDERS.match(/\{folderMenu\(folder, running\)\}/g) ?? []).length,
    2,
    "one of the two layouts stopped using the shared row menu",
  );
  // One confirmation, mounted by whichever layout rendered.
  assert.match(FOLDERS, /const removeIndexConfirm = \(/);
  assert.equal(
    (FOLDERS.match(/\{removeIndexConfirm\}/g) ?? []).length,
    2,
    "a layout can unlink-and-remove without confirming it",
  );
});

// The panel is what Settings, the knowledge-base dialog and the Sources tab render, and none of
// them asked for a card.
test("the other folder lists keep the panel layout", async () => {
  for (const path of [
    "features/settings/tabs/data-tab.tsx",
    "features/rag/components/knowledge-base-dialog.tsx",
    "features/rag/components/project-sources-panel.tsx",
  ]) {
    const source = await readSrcAsync(path);
    assert.ok(
      !source.includes('variant="card"'),
      `${path} was switched to the dialog's card layout`,
    );
  }
  assert.match(FOLDERS, /variant = "panel",/);
});

// A failed rename must not take the instructions with it, and neither write should run when its
// own field is untouched.
test("name and instructions are saved as separate writes", () => {
  assert.match(DIALOG, /if \(nameChanged\) await renameChatProject\(target\.id, trimmedName\);/);
  assert.match(
    DIALOG,
    /if \(instructionsChanged\) \{\n\s*await updateChatProjectInstructions\(target\.id, trimmedInstructions\);/,
  );
  // Nothing edited is not a write at all, just a close.
  assert.match(DIALOG, /if \(!dirty\) \{\n\s*onOpenChange\(false\);/);
  // An empty name is not a rename: it would leave the folder row with nothing to show.
  assert.match(DIALOG, /if \(busy \|\| !trimmedName\) return;/);
  assert.match(DIALOG, /disabled=\{busy \|\| !trimmedName\}/);
});

// The draft is the open project's. Reseeding on id keeps the last project's text from being saved
// over this one when the dialog is reopened from another row.
test("the draft follows whichever project is open", () => {
  assert.match(
    DIALOG,
    /if \(\(project\?\.id \?\? null\) !== seededFor\) \{\n\s*setSeededFor\(project\?\.id \?\? null\);\n\s*setName\(project\?\.name \?\? ""\);\n\s*setInstructions\(project\?\.instructions \?\? ""\);/,
  );
  // And a save in flight cannot be escaped out from under.
  assert.match(DIALOG, /function close\(\) \{\n\s*if \(busy\) return;/);
});

// Enter commits the name field; the instructions box needs its newlines, so the chord commits
// from either.
test("the dialog commits on Enter and on the chord", () => {
  assert.match(
    DIALOG,
    /if \(e\.key === "Enter" && \(e\.metaKey \|\| e\.ctrlKey\)\) \{\n\s*e\.preventDefault\(\);\n\s*void save\(\);/,
  );
  assert.match(
    DIALOG,
    /onKeyDown=\{\(e\) => \{\n\s*if \(e\.key === "Enter"\) \{\n\s*e\.preventDefault\(\);\n\s*void save\(\);/,
  );
});

// Delete is the one action here that cannot be undone, so it reads as the tinted destructive it
// is rather than as plain red text beside Cancel.
test("delete is a destructive button, and the caller still confirms it", () => {
  assert.match(
    DIALOG,
    /<Button\n\s*type="button"\n\s*variant="destructive"\n\s*disabled=\{busy\}\n\s*onClick=\{\(\) => \{\n\s*onOpenChange\(false\);\n\s*onDelete\(project\);/,
  );
});
