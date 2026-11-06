// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const API = readSrc("features/chat/api/skills-api.ts");
const DIALOG = readSrc("features/chat/components/chat-skills-dialog.tsx");

const IMPORT = /export async function importSkills\([^]*?\n\}\n/.exec(API)?.[0] ?? "";
const EXPORT = /export async function exportSkill\([^]*?\n\}\n/.exec(API)?.[0] ?? "";
const IMPORT_FILE = /async function importFile\([^]*?\n {2}\}\n/.exec(DIALOG)?.[0] ?? "";
const RESULT_LIST = /function ImportResultList\([^]*?\n\}\n/.exec(DIALOG)?.[0] ?? "";

test("import posts multipart without a hand-set Content-Type and re-reads the folders", () => {
  assert.match(IMPORT, /form\.append\("file", file\);/);
  assert.match(IMPORT, /authFetch\("\/api\/skills\/import", \{\n\s*method: "POST",\n\s*body: form,/);
  // A hand-set Content-Type would drop the multipart boundary.
  assert.doesNotMatch(IMPORT, /headers:/);
  assert.match(IMPORT, /await skillsMutated\(\);/);
});

test("export reads a blob and surfaces the backend's error", () => {
  assert.match(EXPORT, /`\/api\/skills\/\$\{encodeURIComponent\(name\)\}\/export`/);
  assert.match(EXPORT, /if \(!response\.ok\) await parseResponse<unknown>\(response\);/);
  assert.match(EXPORT, /return response\.blob\(\);/);
  assert.match(DIALOG, /await downloadFile\(blob, `\$\{name\}\.zip`, "application\/zip"\);/);
  assert.match(DIALOG, /if \(!isDownloadCancelled\(cause\)\)/);
});

test("only a clean one-skill import changes the view, and never out of an open skill", () => {
  assert.match(DIALOG, /event\.target\.value = "";/);
  assert.equal(IMPORT_FILE.match(/setView\(/g)?.length, 1);
  assert.match(
    IMPORT_FILE,
    /if \(failed\.length === 0\) \{[^]*?if \(count === 1\) \{[^]*?current\.kind === "library" \? \{ kind: "skill", key: keyOf\(imported\[0\]\) \} : current/,
  );
  assert.match(IMPORT_FILE, /const notify = count > 0 \? toast\.warning : toast\.error;/);
});

test("the result list puts failures, with their reason, before the added skills", () => {
  assert.ok(RESULT_LIST.indexOf("failed.map(") > 0);
  assert.ok(RESULT_LIST.indexOf("failed.map(") < RESULT_LIST.indexOf("imported.map("));
  assert.match(RESULT_LIST, /\{failure\.error\}/);
});

test("export is offered on every readable skill, not only editable ones", () => {
  assert.match(DIALOG, /\{readable \? \(\n\s*<Button[^]*?onClick=\{\(\) => void exportSelected\(\)\}/);
});

test("on phones search gets its own row and Import keeps its name as an icon", () => {
  assert.match(DIALOG, /max-sm:flex-wrap max-sm:justify-end">\n\s*<div className="relative min-w-0 flex-1 max-sm:basis-full">/);
  assert.match(DIALOG, /aria-label=\{t\("skills\.import"\)\}/);
  assert.match(DIALOG, /<span className="max-sm:hidden">\{t\("skills\.import"\)\}<\/span>/);
});
