// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Tab panels must stay lazy; one static ./tabs edge pulls them all back into startup. */

import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "@typescript/typescript6";

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

test("the panels are prefetched: the opening tab after startup, the rest after an open", async () => {
  const source = await readFile(DIALOG, "utf8");
  assert.match(source, /scheduleIdleTask/);
  // Startup: the tab Settings opens on, plus the shortcuts list Mod+/ opens.
  assert.match(
    source,
    /prefetchTab\(resolveSettingsTab\(useSettingsDialogStore\.getState\(\)\.activeTab, isOwner\)\);\s*prefetchTab\("keyboard-shortcuts"\);/,
  );
  // After an open: the rest, once per page and one per idle slot, nothing while closed.
  assert.match(source, /if \(!open \|\| tabsLeftToPrefetch\.length === 0\)/);
  assert.match(source, /tabsLeftToPrefetch\.shift\(\)/);
  // It warms unselected panels, so a failed chunk must not surface as a rejection.
  assert.match(source, /loadTab\(id\)\.catch\(/);
  // One request per panel, shared by the prefetch and the lazy component; a failed one is dropped.
  assert.match(source, /lazy\(\(\) => loadTab\(id\)\)/);
  assert.match(source, /tabRequests\[id\] = undefined;/);
});

test("an open renders the requested panel, never the one shown last", async () => {
  // The deferred tab lives in the pane, which mounts with the dialog's content, so only tab
  // switches made while open lag. A tab's first view in the page paints the dialog first (null,
  // no panel) and renders the panel right after; once shown, an open renders it in the same pass.
  const source = await readFile(DIALOG, "utf8");
  const pane = source.slice(
    source.indexOf("const SettingsPanelPane = memo("),
    source.indexOf("export function SettingsDialog()"),
  );
  const dialog = source.slice(source.indexOf("export function SettingsDialog()"));
  assert.match(
    pane,
    /useDeferredValue<SettingsTab \| null>\(\s*requestedTab,\s*shownTabs\.has\(requestedTab\) \? undefined : null,\s*\)/,
  );
  assert.match(source, /shownTabs\.add\(tab\)/);
  assert.doesNotMatch(dialog, /useDeferredValue\(/);
  assert.match(dialog, /<SettingsPanelPane\b/);
});

test("opening focuses the active tab once and forces no layout from the rail ref", async () => {
  const source = await readFile(DIALOG, "utf8");
  // Radix's default would focus the search box first, then the tab a frame later.
  assert.match(source, /onOpenAutoFocus=\{\(e\) => \{\s*if \(focusTabButton\(activeTab\)\) \{\s*e\.preventDefault\(\);/);
  // The rail's scroll bar width comes from the ResizeObserver only, never a read mid-commit.
  const rail = source.slice(source.indexOf("const railRef = useCallback("), source.indexOf("const visibleTabs = useMemo("));
  assert.match(rail, /new ResizeObserver\(sync\)/);
  assert.doesNotMatch(rail, /^\s*sync\(\);/m);
});
