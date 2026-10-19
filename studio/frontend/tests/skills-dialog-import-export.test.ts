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
const IMPORT_FILE = /async function importFile\([^]*?\n {2}\}\n/.exec(DIALOG)?.[0] ?? "";
// The handler's three outcomes, split at its top-level if / else if / else.
const ALL_OK = /if \(failed\.length === 0\) \{[^]*?\n {6}\} else if/.exec(IMPORT_FILE)?.[0] ?? "";
const SINGLE_FAILED =
  /\} else if \(count === 0 && failed\.length === 1\) \{[^]*?\n {6}\} else \{/.exec(IMPORT_FILE)?.[0] ?? "";
const PARTIAL = /\n {6}\} else \{[^]*?\n {6}\}\n/.exec(IMPORT_FILE)?.[0] ?? "";
const RESULT_LIST = /function ImportResultList\([^]*?\n\}\n/.exec(DIALOG)?.[0] ?? "";

test("import posts the zip as multipart and re-reads the folders", () => {
  assert.match(IMPORT, /form\.append\("file", file\);/);
  assert.match(IMPORT, /authFetch\("\/api\/skills\/import", \{\n\s*method: "POST",\n\s*body: form,/);
  // A hand-set Content-Type would drop the multipart boundary.
  assert.doesNotMatch(IMPORT, /headers:/);
  assert.match(IMPORT, /await skillsMutated\(\);/);
  // Each skill in the ZIP installs or fails on its own, so a 200 carries both lists.
  assert.match(
    API,
    /export type SkillImportResult = \{\n\s*imported: SkillRecord\[\];\n\s*failed: \{ name: string; error: string \}\[\];\n\};/,
  );
  assert.match(IMPORT, /importSkills\(file: File\): Promise<SkillImportResult>/);
  assert.match(IMPORT, /parseResponse<SkillImportResult>\(response\)/);
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
  assert.match(DIALOG, /const \{ imported, failed \} = await importSkills\(file\);/);
  assert.match(DIALOG, /const busy = [^;]*\bimporting \|\| exporting;/);
});

test("an import with no failures toasts every skill it added, and opens it only when there is one", () => {
  assert.match(
    ALL_OK,
    /toast\.success\(t\(count === 1 \? "skills\.importedOne" : "skills\.importedMany", \{ count \}\), \{\n\s*description: <ImportResultList imported=\{imported\} failed=\{failed\} \/>,/,
  );
  // A skill opened while the upload ran keeps the view (and its unsaved draft).
  assert.match(
    ALL_OK,
    /if \(count === 1\) \{[^]*?current\.kind === "library" \? \{ kind: "skill", key: keyOf\(imported\[0\]\) \} : current[^]*?readManifest\(imported\[0\]\);\n\s*\}/,
  );
  // Several stay on the library, and so does any import with a failure: only this branch
  // changes the view.
  assert.equal(IMPORT_FILE.match(/setView\(/g)?.length, 1);
  assert.equal(IMPORT_FILE.match(/readManifest\(/g)?.length, 1);
  assert.match(EN, /importedOne: "\{count\} skill added",\n\s*importedMany: "\{count\} skills added",/);
});

test("a ZIP of one skill that fails shows the backend's reason, as a single import did", () => {
  assert.match(
    SINGLE_FAILED,
    /toast\.error\(t\("skills\.importError"\), \{ description: failed\[0\]\.error \}\);/,
  );
  assert.doesNotMatch(SINGLE_FAILED, /setView\(|readManifest\(/);
});

test("a folder with failures warns when some landed, errors when none did, and opens nothing", () => {
  assert.match(PARTIAL, /const notify = count > 0 \? toast\.warning : toast\.error;/);
  assert.match(
    PARTIAL,
    /notify\(t\("skills\.importPartial", \{ added: count, total: count \+ failed\.length \}\), \{\n\s*description: <ImportResultList imported=\{imported\} failed=\{failed\} \/>,/,
  );
  assert.doesNotMatch(PARTIAL, /setView\(|readManifest\(/);
  assert.match(EN, /importPartial: "Added \{added\} of \{total\} skills",/);
});

test("the result list ticks each added skill and crosses each failed one with its reason", () => {
  assert.match(RESULT_LIST, /function ImportResultList\(\{ imported, failed \}: SkillImportResult\)/);
  assert.match(RESULT_LIST, /className="[^"]*rounded-md border[^"]*"/);
  assert.match(
    RESULT_LIST,
    /imported\.map\(\(skill\) => \([^]*?icon=\{Tick02Icon\}[^]*?text-emerald-600[^]*?<span className="truncate">\{skill\.name\}<\/span>/,
  );
  assert.match(
    RESULT_LIST,
    /failed\.map\(\(failure\) => \([^]*?icon=\{Cancel01Icon\}[^]*?text-destructive[^]*?<span className="truncate">\{failure\.name\}<\/span>/,
  );
  // The reason wraps in smaller muted text instead of being cut off.
  const reason = /<p className="([^"]*)">\{failure\.error\}<\/p>/.exec(RESULT_LIST)?.[1] ?? "";
  assert.match(reason, /\btext-xs\b/);
  assert.match(reason, /\btext-muted-foreground\b/);
  assert.match(reason, /\bbreak-words\b/);
  assert.doesNotMatch(reason, /\btruncate\b/);
  // Failures come first so the list's scroll cap cannot hide them.
  assert.ok(RESULT_LIST.indexOf("failed.map(") < RESULT_LIST.indexOf("imported.map("));
  assert.match(DIALOG, /import \{ Tick02Icon \} from "@\/lib\/tick-icon";/);
  assert.match(DIALOG, /import \{\n\s*Cancel01Icon,[^}]*\} from "@hugeicons\/core-free-icons";/);
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
