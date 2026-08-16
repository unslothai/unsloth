// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The settings dialog is mounted at the app root and is closed for the whole
 * launch, but its twelve tab panels were static imports, so the browser fetched,
 * parsed and executed every one of them before the first paint.
 *
 * Bundle membership follows the static import graph, so a single static
 * `./tabs/...` edge from anywhere reachable at startup puts them all back. That
 * is what these assert, rather than the dialog's rendered output.
 */

import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const SRC = fileURLToPath(new URL("../src", import.meta.url));
const SETTINGS = path.join(SRC, "features/settings");
const DIALOG = path.join(SETTINGS, "settings-dialog.tsx");
const TABS_DIR = path.join(SETTINGS, "tabs");

async function* walk(dir: string): AsyncGenerator<string> {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(full);
    else if (/\.tsx?$/.test(entry.name)) yield full;
  }
}

/**
 * Module specifiers of the `import`/`export ... from` declarations in a file. A
 * deferred `import(...)` parses as a call expression, so it is never collected,
 * which is exactly the distinction being enforced.
 */
const staticSpecifiers = (file: string, text: string): string[] => {
  const parsed = ts.createSourceFile(
    file,
    text,
    ts.ScriptTarget.ESNext,
    false,
    file.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
  const specifiers: string[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) {
      const specifier = node.moduleSpecifier;
      if (specifier && ts.isStringLiteral(specifier)) {
        specifiers.push(specifier.text);
      }
    }
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(parsed, visit);
  return specifiers;
};

/** A tab panel, however the importer spelled the path. */
const isTabPanel = (specifier: string): boolean =>
  /(^|\/)tabs\/[\w-]+-tab$/.test(specifier);

test("the dialog loads every tab panel on demand", async () => {
  const source = await readFile(DIALOG, "utf8");

  const statics = staticSpecifiers(DIALOG, source).filter(isTabPanel);
  assert.deepEqual(statics, [], `settings-dialog still statically imports: ${statics}`);

  // One loader per panel on disk, so a tab added later cannot quietly go missing
  // from the map and take the whole switch down with it.
  const panels = (await readdir(TABS_DIR)).filter((f) => /-tab\.tsx$/.test(f));
  assert.ok(panels.length >= 12, `only found ${panels.length} tab panels`);
  for (const file of panels) {
    const specifier = `./tabs/${file.replace(/\.tsx$/, "")}`;
    assert.ok(
      source.includes(`import("${specifier}")`),
      `no deferred import for ${specifier}`,
    );
  }
});

test("nothing else in src statically imports a tab panel", async () => {
  const offenders: string[] = [];
  for await (const file of walk(SRC)) {
    if (file.startsWith(TABS_DIR)) {
      // A panel importing a sibling is its own business; it is already lazy.
      continue;
    }
    const text = await readFile(file, "utf8");
    for (const specifier of staticSpecifiers(file, text)) {
      if (isTabPanel(specifier)) {
        offenders.push(`${path.relative(SRC, file)}: ${specifier}`);
      }
    }
  }
  assert.deepEqual(
    offenders,
    [],
    `settings tab panels are back on the startup path via:\n${offenders.join("\n")}`,
  );
});

test("the panels are prefetched once the dialog opens", async () => {
  // Without this the first click on a tab waits on a network round trip, which
  // would trade a startup cost for an interaction one.
  const source = await readFile(DIALOG, "utf8");
  assert.match(source, /scheduleIdleTask/);
  assert.match(source, /Object\.values\(TAB_LOADERS\)/);
});
