// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A value read at module-evaluation time must come from a module that cannot be
 * mid-initialization when the read happens.
 *
 * `general-tab.tsx` builds a top-level `const` array containing
 * `SIDEBAR_ORGANIZATION_STORAGE_KEY`. That key used to be defined in
 * `sidebar-organization-store.ts`, which imports zustand and sits in an import
 * cycle running through the chat barrel. Entering that cycle from the settings
 * side left the binding in its temporal dead zone, and the read threw
 * `Cannot access 'SIDEBAR_ORGANIZATION_STORAGE_KEY' before initialization`.
 * Nothing catches that: there is no error boundary above the router, so the
 * whole tree unmounts and the app shows a white screen on launch.
 *
 * The app only avoided it by accident. `app-sidebar.tsx` imports the theme
 * toggler above its own chat import, which happened to evaluate the store
 * first; swapping those two lines reproduced the white screen. That is not a
 * property anyone can be expected to preserve by hand, so it is asserted here.
 *
 * The fix is a keys module with no imports of its own. A module with no imports
 * is always fully evaluated before its importer, cycle or not.
 */

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const SRC = fileURLToPath(new URL("../src", import.meta.url));
const KEYS = path.join(SRC, "features/chat/stores/sidebar-organization-keys.ts");
const GENERAL_TAB = path.join(SRC, "features/settings/tabs/general-tab.tsx");

const parse = (file: string, text: string) =>
  ts.createSourceFile(
    file,
    text,
    ts.ScriptTarget.ESNext,
    true,
    file.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );

/** Module specifiers of `import`/`export ... from` declarations. */
const staticSpecifiers = (file: string, text: string): string[] => {
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
  ts.forEachChild(parse(file, text), visit);
  return specifiers;
};

test("the sidebar organization keys module imports nothing", async () => {
  // This is the whole reason it is safe to read at module scope. An import here
  // puts it back in the cycle and the white screen comes back.
  const text = await readFile(KEYS, "utf8");
  assert.deepEqual(
    staticSpecifiers(KEYS, text),
    [],
    "sidebar-organization-keys.ts must not import anything",
  );
  assert.match(text, /export const SIDEBAR_ORGANIZATION_STORAGE_KEY/);
});

test("general-tab reads the key from the keys module, not the store or the barrel", async () => {
  const text = await readFile(GENERAL_TAB, "utf8");
  const source = parse(GENERAL_TAB, text);

  let specifier: string | null = null;
  const visit = (node: ts.Node): void => {
    if (ts.isImportDeclaration(node) && node.importClause?.namedBindings) {
      const bindings = node.importClause.namedBindings;
      if (
        ts.isNamedImports(bindings) &&
        bindings.elements.some(
          (element) => element.name.text === "SIDEBAR_ORGANIZATION_STORAGE_KEY",
        ) &&
        ts.isStringLiteral(node.moduleSpecifier)
      ) {
        specifier = node.moduleSpecifier.text;
      }
    }
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(source, visit);

  assert.equal(
    specifier,
    "@/features/chat/stores/sidebar-organization-keys",
    "general-tab must not reach the key through the store or the chat barrel; " +
      "both are in an import cycle with this file",
  );
});

test("the key is still read at module scope, so the guard above is load-bearing", async () => {
  // If this ever stops being a module-scope read the two tests above are
  // pointless and should go, rather than sit here passing for no reason.
  const text = await readFile(GENERAL_TAB, "utf8");
  const source = parse(GENERAL_TAB, text);

  let readAtModuleScope = false;
  for (const statement of source.statements) {
    if (!ts.isVariableStatement(statement)) {
      continue;
    }
    const visit = (node: ts.Node): void => {
      if (
        ts.isIdentifier(node) &&
        node.text === "SIDEBAR_ORGANIZATION_STORAGE_KEY"
      ) {
        readAtModuleScope = true;
      }
      ts.forEachChild(node, visit);
    };
    ts.forEachChild(statement, visit);
  }

  assert.ok(
    readAtModuleScope,
    "general-tab no longer reads the key at module scope; drop this file's guards",
  );
});
