// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const API = readSrc("features/chat/api/skills-api.ts");
const DIALOG = readSrc("features/chat/components/chat-skills-dialog.tsx");
const EN = readSrc("i18n/locales/en.ts");

const IMPORT = /export async function importSkills\([^]*?\n\}\n/.exec(API)?.[0] ?? "";
const EXPORT = /export async function exportSkill\([^]*?\n\}\n/.exec(API)?.[0] ?? "";

test("import posts the zip as multipart and re-reads the folders", () => {
  assert.match(IMPORT, /form\.append\("file", file\);/);
  assert.match(IMPORT, /authFetch\("\/api\/skills\/import", \{\n\s*method: "POST",\n\s*body: form,/);
  // A hand-set Content-Type would drop the multipart boundary.
  assert.doesNotMatch(IMPORT, /headers:/);
  assert.match(IMPORT, /await skillsMutated\(\);/);
  // One ZIP may hold several skills, so the backend always answers with a list.
  assert.match(IMPORT, /importSkills\(file: File\): Promise<SkillRecord\[\]>/);
  assert.match(IMPORT, /parseResponse<SkillRecord\[\]>\(response\)/);
});

test("export reads the archive as a blob and throws the backend's detail", () => {
  assert.match(EXPORT, /authFetch\(\n\s*`\/api\/skills\/\$\{encodeURIComponent\(name\)\}\/export`,\n\s*\);/);
  assert.match(EXPORT, /if \(!response\.ok\) await parseResponse<unknown>\(response\);/);
  assert.match(EXPORT, /return response\.blob\(\);/);
  assert.doesNotMatch(EXPORT, /skillsMutated/);
});

test("the dialog imports a .zip through a hidden input and stays open meanwhile", () => {
  assert.match(
    DIALOG,
    /<input\n\s*ref=\{fileInputRef\}\n\s*type="file"\n\s*accept="\.zip,application\/zip"\n\s*className="hidden"/,
  );
  assert.match(DIALOG, /onClick=\{\(\) => fileInputRef\.current\?\.click\(\)\}/);
  assert.match(DIALOG, /event\.target\.value = "";/);
  assert.match(DIALOG, /const imported = await importSkills\(file\);/);
  assert.match(DIALOG, /const busy = [^;]*\bimporting \|\| exporting;/);
});

test("an import toasts every skill it added, and opens it only when there is one", () => {
  const importFile = /async function importFile\([^]*?\n {2}\}\n/.exec(DIALOG)?.[0] ?? "";
  assert.match(
    importFile,
    /toast\.success\(t\(count === 1 \? "skills\.importedOne" : "skills\.importedMany", \{ count \}\), \{\n\s*description: <ImportedSkillsList skills=\{imported\} \/>,/,
  );
  // A skill opened while the upload ran keeps the view (and its unsaved draft).
  assert.match(
    importFile,
    /if \(count === 1\) \{[^]*?current\.kind === "library" \? \{ kind: "skill", key: keyOf\(imported\[0\]\) \} : current[^]*?readManifest\(imported\[0\]\);\n\s*\}/,
  );
  // Several stay on the library: nothing else in the handler changes the view.
  assert.equal(importFile.match(/setView\(/g)?.length, 1);

  const list = /function ImportedSkillsList\([^]*?\n\}\n/.exec(DIALOG)?.[0] ?? "";
  assert.match(list, /className="[^"]*rounded-md border[^"]*"/);
  assert.match(
    list,
    /skills\.map\(\(skill\) => \([^]*?icon=\{Tick02Icon\}[^]*?text-emerald-600[^]*?<span className="truncate">\{skill\.name\}<\/span>/,
  );
  assert.match(DIALOG, /import \{ Tick02Icon \} from "@\/lib\/tick-icon";/);
  assert.match(EN, /importedOne: "\{count\} skill added",\n\s*importedMany: "\{count\} skills added",/);
});

test("export saves through downloadFile and stays quiet on a cancelled save", () => {
  assert.match(DIALOG, /await downloadFile\(blob, `\$\{name\}\.zip`, "application\/zip"\);/);
  assert.match(DIALOG, /if \(!isDownloadCancelled\(cause\)\) \{\n\s*toast\.error\(t\("skills\.exportError"\)/);
});

test("export is offered on every readable skill, not only editable ones", () => {
  assert.match(DIALOG, /\{readable \? \(\n\s*<Button[^]*?onClick=\{\(\) => void exportSelected\(\)\}/);
  const editableBranch = /\) : editable && selected \? \([^]*?\) : null\}/.exec(DIALOG)?.[0];
  assert.ok(editableBranch, "editable footer branch not found");
  assert.doesNotMatch(editableBranch, /exportSelected/);
});
