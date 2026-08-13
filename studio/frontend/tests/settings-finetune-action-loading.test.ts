// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SRC = fileURLToPath(new URL("../src", import.meta.url));
const DATA_TAB = path.join(SRC, "features/settings/tabs/data-tab.tsx");

const source = await readFile(DATA_TAB, "utf8");
const STATIC_FINE_TUNE_IMPORT = /finetune-recipe/;
const ACTION_FINE_TUNE_IMPORT =
  /await import\(\s*"\.\.\/components\/finetune-recipe"\s*\)/g;

test("fine-tuning workflow dependencies load only when their actions run", () => {
  // Bundle membership follows the static import graph, so this structural
  // assertion guards startup loading more directly than exercising the actions.
  const imports = source.slice(0, source.indexOf("export function DataTab"));
  assert.doesNotMatch(
    imports,
    STATIC_FINE_TUNE_IMPORT,
    "the Data tab must not pull fine-tuning workflows into Studio startup",
  );

  const actionImports = source.match(ACTION_FINE_TUNE_IMPORT);
  assert.equal(actionImports?.length, 2);
});

async function* walk(dir: string): AsyncGenerator<string> {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(full);
    else if (/\.tsx?$/.test(entry.name)) yield full;
  }
}

test("no module statically imports the fine-tuning workflow", async () => {
  // The check above holds one FILE. The property being bought is repo-wide: a
  // static import added to any other eagerly reached module would put the whole
  // Recipe Studio chunk back into startup with that assertion still green, which
  // is exactly the regression this PR exists to prevent.
  const offenders: string[] = [];
  for await (const file of walk(SRC)) {
    const text = await readFile(file, "utf8");
    for (const line of text.split("\n")) {
      // Static only: `await import(...)` and `import(...)` are the deferred forms
      // this PR introduces and are what we want everyone to use.
      if (!/finetune-recipe/.test(line)) continue;
      if (/\bimport\s*\(/.test(line)) continue;
      if (/^\s*(import|export)\b/.test(line)) {
        offenders.push(`${path.relative(SRC, file)}: ${line.trim()}`);
      }
    }
  }
  assert.deepEqual(
    offenders,
    [],
    `fine-tuning workflow reachable from startup via:\n${offenders.join("\n")}`,
  );
});
