// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const SRC = fileURLToPath(new URL("../src", import.meta.url));
const DATA_TAB = path.join(SRC, "features/settings/tabs/data-tab.tsx");
const RECIPE_MODULE = /finetune-recipe/;
const ACTION_FINE_TUNE_IMPORT =
  /await import\(\s*"\.\.\/components\/finetune-recipe"\s*\)/g;

test("both Data tab fine-tuning actions defer their workflow import", async () => {
  const source = await readFile(DATA_TAB, "utf8");
  assert.equal(source.match(ACTION_FINE_TUNE_IMPORT)?.length, 2);
});

async function* walk(dir: string): AsyncGenerator<string> {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(full);
    else if (/\.tsx?$/.test(entry.name)) yield full;
  }
}

/**
 * Module specifiers of the `import`/`export ... from` declarations in a file,
 * which is the edge set that fixes bundle membership. A deferred `import(...)`
 * parses as a call expression rather than a declaration, so it is never
 * collected and the pattern this PR adopts stays allowed.
 */
const staticSpecifiers = (file: string, text: string): string[] => {
  const parsed = ts.createSourceFile(
    file,
    text,
    ts.ScriptTarget.ESNext,
    // Parent pointers are only needed for getText(); StringLiteral.text is enough.
    false,
    file.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
  const specifiers: string[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) {
      // `export { x }` with no `from` has no specifier and adds no edge.
      const specifier = node.moduleSpecifier;
      if (specifier && ts.isStringLiteral(specifier))
        specifiers.push(specifier.text);
    }
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(parsed, visit);
  return specifiers;
};

test("no module statically imports the fine-tuning workflow", async () => {
  // Bundle membership follows the static import graph, and the property being
  // bought is repo-wide: a static import added to any eagerly reached module
  // would put the whole Recipe Studio chunk back into startup. Parsed rather
  // than scanned by line, because the import this PR removed spanned four lines
  // and a line-oriented check keyed on `import` cannot see a specifier that
  // sits on the closing `} from "..."` line.
  const offenders: string[] = [];
  for await (const file of walk(SRC)) {
    const text = await readFile(file, "utf8");
    for (const specifier of staticSpecifiers(file, text)) {
      if (RECIPE_MODULE.test(specifier))
        offenders.push(`${path.relative(SRC, file)}: ${specifier}`);
    }
  }
  assert.deepEqual(
    offenders,
    [],
    `fine-tuning workflow reachable from startup via:\n${offenders.join("\n")}`,
  );
});
