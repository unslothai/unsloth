// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Skills dialog creates, edits and deletes skills in ~/.agents/skills, the same folder and
// the same file the create_skill tool writes. What it must not do is offer those on a skill it
// cannot write (a ~/.claude or bundled one), or send a name the backend would refuse.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const API = readSrc("features/chat/api/skills-api.ts");
const DIALOG = readSrc("features/chat/components/chat-skills-dialog.tsx");

// The module imports the auth layer, so the check is lifted from source; the assertion keeps
// the copy honest.
const declaration = /export const SKILL_NAME_PATTERN = (\/.*\/);/.exec(API);
assert.ok(declaration, "SKILL_NAME_PATTERN declaration not found");
const SKILL_NAME_PATTERN = new RegExp(declaration[1].slice(1, -1));
assert.match(
  API,
  /return SKILL_NAME_PATTERN\.test\(name\) && !name\.includes\("--"\);/,
);
const isValidSkillName = (name: string): boolean =>
  SKILL_NAME_PATTERN.test(name) && !name.includes("--");

// The same rule as the backend's _normalize_skill_name, so a name the form accepts is one the
// route accepts, and the folder it becomes is a safe one.
test("the name check agrees with the backend", () => {
  for (const name of [
    "a",
    "a1",
    "release-notes",
    "code-arithmetic-2",
    "x".repeat(64),
  ]) {
    assert.ok(isValidSkillName(name), `${name} should be accepted`);
  }
  for (const name of [
    "",
    "-a",
    "a-",
    "a--b",
    "Release",
    "release notes",
    "notes.md",
    "../escape",
    "x".repeat(65),
    "ünder",
  ]) {
    assert.ok(!isValidSkillName(name), `${name} should be refused`);
  }
});

test("the api speaks to the four skill routes", () => {
  assert.match(API, /authFetch\("\/api\/skills", \{\n\s*method: "POST",/);
  assert.match(
    API,
    /authFetch\(`\/api\/skills\/\$\{encodeURIComponent\(name\)\}`\);\n\s*return parseResponse<SkillManifest>/,
  );
  assert.match(
    API,
    /authFetch\(`\/api\/skills\/\$\{encodeURIComponent\(name\)\}`, \{\n\s*method: "PUT",/,
  );
  assert.match(
    API,
    /authFetch\(`\/api\/skills\/\$\{encodeURIComponent\(name\)\}`, \{\n\s*method: "DELETE",/,
  );
});

// A new or changed skill changes what @ offers and what the system prompt lists, in this window
// and the others, so every write settles through the one helper the toggle already uses.
test("every write re-reads the folders and tells the other windows", () => {
  assert.equal((API.match(/await skillsMutated\(\);/g) ?? []).length, 3);
  assert.match(
    API,
    /async function skillsMutated\(\): Promise<void> \{\n\s*channel\?\.postMessage\("changed"\);\n\s*void refreshContextUsage\(\{ invalidate: true \}\);\n\s*await listSkills\(true\)/,
  );
});

// ~/.claude and bundled skills are read-only to the dialog, a linked entry would be a write
// somewhere else, and an invalid folder has nothing to seed the editor from. The backend refuses
// all of those anyway; the editor stays read-only on them and says why.
test("save and delete are offered only on skills the dialog can write", () => {
  assert.match(
    DIALOG,
    /function isEditable\(skill: SkillRecord\): boolean \{\n\s*return skill\.valid && skill\.source === "agents" && !skill\.linked;/,
  );
  assert.match(DIALOG, /const editable = selected !== null && isEditable\(selected\);/);
  assert.match(DIALOG, /readOnly=\{!editable\}/);
  assert.match(
    DIALOG,
    /onDelete=\{editable \? \(\) => setConfirmingDelete\(selected\) : undefined\}/,
  );
  // Deleting removes a folder, so it goes through a confirmation with a destructive action.
  assert.match(
    DIALOG,
    /<AlertDialogAction\n\s*variant="destructive"\n\s*onClick=\{\(\) => \{\n\s*const skill = confirmingDelete;/,
  );
  assert.match(DIALOG, /if \(skill\) void remove\(skill\);/);
  // A card says which of the two it opens as.
  assert.match(DIALOG, /\{isEditable\(skill\) \? t\("skills\.edit"\) : t\("skills\.view"\)\}/);
});

// The form cannot submit what the route would bounce, and shows why while the name is typed.
test("the new-skill form holds back an invalid or incomplete draft", () => {
  assert.match(
    DIALOG,
    /const nameInvalid = trimmedName\.length > 0 && !isValidSkillName\(trimmedName\);/,
  );
  assert.match(DIALOG, /aria-invalid=\{nameInvalid \|\| undefined\}/);
  assert.match(
    DIALOG,
    /hint=\{nameInvalid \? t\("skills\.nameInvalid"\) : t\("skills\.nameHint"\)\}/,
  );
  assert.match(DIALOG, /disabled=\{creating \|\| !canCreate\}/);
  assert.match(DIALOG, /if \(creating \|\| !canCreate\) return;/);
  // Editing keeps the name: renaming would move the folder, which is not what the editor is for.
  assert.doesNotMatch(DIALOG, /id="skill-edit-name"/);
});

// An edit is only sent once it differs from the file, and never with a field emptied out.
test("save is held back until the draft differs from the file and is complete", () => {
  assert.match(
    DIALOG,
    /const canSave =\n\s*editable &&\n\s*draft !== null &&\n\s*draft\.description\.trim\(\)\.length > 0 &&\n\s*draft\.instructions\.trim\(\)\.length > 0;/,
  );
  assert.match(
    DIALOG,
    /if \(!selected \|\| !manifest \|\| !draft \|\| !canSave \|\| pending !== null\) return;/,
  );
  // A draft that returns to the file's content is dropped rather than kept as a no-op change.
  assert.match(
    DIALOG,
    /return next\.description === manifest\.description &&\n\s*next\.instructions === manifest\.instructions\n\s*\? null\n\s*: next;/,
  );
});

// A save, create or delete in flight must finish before the dialog can go away under it; a
// draft asks before it is dropped; and each open starts on the library, not on a draft left
// from last time.
test("the dialog guards writes and drafts, and reopens on the library", () => {
  assert.match(DIALOG, /if \(!next && busy\) return;/);
  assert.match(
    DIALOG,
    /if \(!next && dirty\) \{\n\s*setConfirmingDiscard\(\(\) => \(\) => onOpenChange\(false\)\);\n\s*return;/,
  );
  assert.match(
    DIALOG,
    /const leaveTo = \(next: \(\) => void\) => \{\n\s*if \(dirty\) setConfirmingDiscard\(\(\) => next\);/,
  );
  assert.match(
    DIALOG,
    /if \(open !== seenOpen\) \{\n\s*setSeenOpen\(open\);\n\s*if \(open\) \{\n\s*setSearchQuery\(""\);\n\s*setEnabledOnly\(false\);\n\s*setView\(LIBRARY\);/,
  );
});
