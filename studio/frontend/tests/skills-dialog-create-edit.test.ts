// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const API = readSrc("features/chat/api/skills-api.ts");
const DIALOG = readSrc("features/chat/components/chat-skills-dialog.tsx");

const declaration = /export const SKILL_NAME_PATTERN = (\/.*\/);/.exec(API);
assert.ok(declaration, "SKILL_NAME_PATTERN declaration not found");
const SKILL_NAME_PATTERN = new RegExp(declaration[1].slice(1, -1));
assert.match(
  API,
  /return SKILL_NAME_PATTERN\.test\(name\) && !name\.includes\("--"\);/,
);
const isValidSkillName = (name: string): boolean =>
  SKILL_NAME_PATTERN.test(name) && !name.includes("--");

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

test("every write re-reads the folders and tells the other windows", () => {
  assert.equal((API.match(/await skillsMutated\(\);/g) ?? []).length, 3);
  assert.match(
    API,
    /async function skillsMutated\(\): Promise<void> \{\n\s*channel\?\.postMessage\("changed"\);\n\s*void refreshContextUsage\(\{ invalidate: true \}\);\n\s*await listSkills\(true\)/,
  );
});

test("save and delete are offered only on skills the dialog can write", () => {
  assert.match(
    DIALOG,
    /function isEditable\(skill: SkillRecord\): boolean \{\n\s*return skill\.valid && skill\.source === "agents" && !skill\.linked;/,
  );
  assert.match(DIALOG, /const editable = selected !== null && isEditable\(selected\);/);
  assert.match(DIALOG, /readOnly=\{!editable\}/);
  assert.match(DIALOG, /\) : editable && selected \? \([^]*?onClick=\{\(\) => setConfirmingDelete\(selected\)\}/);
  assert.match(
    DIALOG,
    /<AlertDialogAction\n\s*variant="destructive"\n\s*onClick=\{\(\) => \{\n\s*const skill = confirmingDelete;/,
  );
  assert.match(DIALOG, /if \(skill\) void remove\(skill\);/);
});

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
  assert.doesNotMatch(DIALOG, /id="skill-edit-name"/);
});

test("save is held back until the draft differs from the file and is complete", () => {
  assert.match(
    DIALOG,
    /const canSave =\n\s*editable &&\n\s*draft !== null &&\n\s*draft\.description\.trim\(\)\.length > 0 &&\n\s*draft\.instructions\.trim\(\)\.length > 0;/,
  );
  assert.match(
    DIALOG,
    /if \(!selected \|\| !manifest \|\| !draft \|\| !canSave \|\| pending !== null\) return;/,
  );
  assert.match(
    DIALOG,
    /return next\.description === manifest\.description &&\n\s*next\.instructions === manifest\.instructions\n\s*\? null\n\s*: next;/,
  );
});

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
    /if \(open !== seenOpen\) \{\n\s*setSeenOpen\(open\);\n\s*if \(open\) \{\n\s*setSearchQuery\(""\);\n\s*setView\(LIBRARY\);/,
  );
});
