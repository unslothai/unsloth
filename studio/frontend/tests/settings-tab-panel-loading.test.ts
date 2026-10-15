// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Tab panels must stay lazy; one static ./tabs edge pulls them all back into startup. */

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

/** Parsed, not grepped: a deferred import(...) is a call expression and is not collected. */
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

const isTabPanel = (specifier: string): boolean =>
  /(^|\/)tabs\/[\w-]+-tab$/.test(specifier);

test("the dialog loads every tab panel on demand", async () => {
  const source = await readFile(DIALOG, "utf8");

  const statics = staticSpecifiers(DIALOG, source).filter(isTabPanel);
  assert.deepEqual(statics, [], `settings-dialog still statically imports: ${statics}`);

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

test("a panel that fails to load cannot take the app down with it", async () => {
  // Nothing above the root-mounted dialog catches, so a render throw unmounts the whole tree.
  const source = await readFile(DIALOG, "utf8");
  const parsed = ts.createSourceFile(
    DIALOG,
    source,
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );

  const boundaries = new Set<string>();
  const collect = (node: ts.Node): void => {
    if (ts.isClassDeclaration(node) && node.name) {
      const catches = node.members.some(
        (member) =>
          (ts.isMethodDeclaration(member) || ts.isPropertyDeclaration(member)) &&
          member.name !== undefined &&
          ts.isIdentifier(member.name) &&
          (member.name.text === "getDerivedStateFromError" ||
            member.name.text === "componentDidCatch"),
      );
      if (catches) boundaries.add(node.name.text);
    }
    ts.forEachChild(node, collect);
  };
  ts.forEachChild(parsed, collect);
  assert.ok(
    boundaries.size > 0,
    "settings-dialog defines no error boundary for the lazy panels",
  );

  const tagName = (node: ts.Node): string | null => {
    if (ts.isJsxElement(node)) return node.openingElement.tagName.getText(parsed);
    if (ts.isJsxSelfClosingElement(node)) return node.tagName.getText(parsed);
    return null;
  };

  let guarded = 0;
  let total = 0;
  const check = (node: ts.Node): void => {
    if (tagName(node) === "Suspense") {
      total += 1;
      for (
        let parent: ts.Node | undefined = node.parent;
        parent;
        parent = parent.parent
      ) {
        const name = tagName(parent);
        if (name && boundaries.has(name)) {
          guarded += 1;
          break;
        }
      }
    }
    ts.forEachChild(node, check);
  };
  ts.forEachChild(parsed, check);
  assert.ok(total > 0, "settings-dialog has no Suspense boundary around the panels");
  assert.equal(
    guarded,
    total,
    `${total - guarded} of ${total} panel Suspense boundaries are not inside ` +
      `one of ${[...boundaries].join(", ")}`,
  );
});

test("the panels are prefetched once the dialog opens", async () => {
  const source = await readFile(DIALOG, "utf8");
  assert.match(source, /scheduleIdleTask/);
  assert.match(source, /Object\.values\(TAB_LOADERS\)/);
  // It warms unselected panels, so a failed chunk must not surface as a rejection.
  assert.match(source, /load\(\)\.catch\(/);
});
