// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrcAsync } from "./helpers/kit.ts";

const DIALOG = await readSrcAsync(
  "features/chat/components/edit-project-dialog.tsx",
);
const FOLDERS = await readSrcAsync(
  "features/rag/components/linked-folders-manager.tsx",
);

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

test("the card ends in an Add folder row that says when it cannot", () => {
  const card = FOLDERS.slice(
    FOLDERS.indexOf('if (variant === "card") {'),
    FOLDERS.indexOf("  return (\n    <section"),
  );
  assert.ok(card.length > 0, "the card layout moved");
  assert.match(card, /<span>Add folder<\/span>/);
  assert.match(card, /disabled=\{!manager\.desktopSupported \|\| manager\.mutating\}/);
  assert.match(card, /onClick=\{\(\) => void manager\.link\(\)\}/);
  assert.match(card, /\{scope \? \(\n\s*<button/);
  assert.match(card, /\{!manager\.desktopSupported \? \(/);
});

test("a card row keeps every action the panel row has", () => {
  assert.match(FOLDERS, /function folderMenu\(/);
  assert.equal(
    (FOLDERS.match(/\{folderMenu\(folder, running\)\}/g) ?? []).length,
    2,
    "one of the two layouts stopped using the shared row menu",
  );
  assert.match(FOLDERS, /const removeIndexConfirm = \(/);
  assert.equal(
    (FOLDERS.match(/\{removeIndexConfirm\}/g) ?? []).length,
    2,
    "a layout can unlink-and-remove without confirming it",
  );
});

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

test("name and instructions are saved as separate writes", () => {
  assert.match(DIALOG, /if \(nameChanged\) await renameChatProject\(target\.id, trimmedName\);/);
  assert.match(
    DIALOG,
    /if \(instructionsChanged\) \{\n\s*await updateChatProjectInstructions\(target\.id, trimmedInstructions\);/,
  );
  assert.match(DIALOG, /if \(!dirty\) \{\n\s*onOpenChange\(false\);/);
  assert.match(DIALOG, /if \(busy \|\| !trimmedName\) return;/);
  assert.match(DIALOG, /disabled=\{busy \|\| !trimmedName\}/);
});

test("the draft follows whichever project is open", () => {
  assert.match(
    DIALOG,
    /if \(\(project\?\.id \?\? null\) !== seededFor\) \{\n\s*setSeededFor\(project\?\.id \?\? null\);\n\s*setName\(project\?\.name \?\? ""\);\n\s*setInstructions\(project\?\.instructions \?\? ""\);/,
  );
  assert.match(DIALOG, /function close\(\) \{\n\s*if \(busy\) return;/);
});

test("the dialog commits on Enter and on the chord", () => {
  assert.match(
    DIALOG,
    /if \(e\.key === "Enter" && \(e\.metaKey \|\| e\.ctrlKey\)\) \{\n\s*e\.preventDefault\(\);\n\s*void save\(\);/,
  );
  assert.match(
    DIALOG,
    /onKeyDown=\{\(e\) => \{\n\s*if \(e\.key === "Enter"\) \{\n\s*e\.preventDefault\(\);\n(?:\s*\/\/.*\n)*\s*e\.stopPropagation\(\);\n\s*void save\(\);/,
  );
});

test("delete is a destructive button, and the caller still confirms it", () => {
  assert.match(
    DIALOG,
    /<Button\n\s*type="button"\n\s*variant="destructive"\n\s*disabled=\{busy\}\n\s*onClick=\{\(\) => \{\n\s*onOpenChange\(false\);\n\s*onDelete\(project\);/,
  );
});
