// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The run-settings panel is the one surface in the tree where a control that
// commits on POINTERDOWN is co-visible with a menu and is not itself inside a
// dismissable layer. Radix Slider commits in onPointerDown -> onSlideStart ->
// updateValues -> onValueChange, so on a non-modal menu the press that dismisses
// the menu also lands a value, and ParamSlider persists it to chat_settings.
// MenuDismissGuard cannot help: it cancels the later click, which is too late,
// and cancelling the pointerdown itself was measured on this branch and rejected
// because it kills every drag that starts from a React onPointerDown.
//
// So: every menu that renders inside ChatSettingsPanel stays modal. This test is
// the guard against a future sweep flipping one of them back, and against a new
// non-modal menu component being dropped into that panel.
//
// It also pins the other half, that the composer pill stays NON-modal, so a
// future "make menus modal again" sweep cannot quietly undo the branch.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SRC = fileURLToPath(new URL("../src/", import.meta.url));

const read = (relative: string): string =>
  readFileSync(path.join(SRC, relative), "utf8");

const LINE_COMMENT = /^[ \t]*\/\/.*$/gm;
const BLOCK_COMMENT = /\/\*[\s\S]*?\*\//g;
const JSX_COMMENT = /\{\s*\/\*[\s\S]*?\*\/\s*\}/g;

/** Strip comments so a code sample inside a rationale cannot satisfy a check. */
function code(source: string): string {
  return source
    .replace(JSX_COMMENT, "")
    .replace(BLOCK_COMMENT, "")
    .replace(LINE_COMMENT, "");
}

/**
 * Body of a top-level `function NAME(` / `const NAME = (` declaration, by brace
 * matching, so a check reads only the component it names.
 */
function declaration(source: string, name: string): string {
  const start = source.search(
    new RegExp(`(?:export\\s+)?(?:function|const)\\s+${name}\\b`),
  );
  assert.notEqual(start, -1, `${name} not found`);
  // Skip the parameter list: a destructured signature is full of braces, so the
  // body starts at the first `{` after the matching `)`.
  const paren = source.indexOf("(", start);
  assert.notEqual(paren, -1, `${name} has no parameter list`);
  let parens = 0;
  let afterParams = -1;
  for (let i = paren; i < source.length; i++) {
    if (source[i] === "(") parens++;
    else if (source[i] === ")") {
      parens--;
      if (parens === 0) {
        afterParams = i;
        break;
      }
    }
  }
  assert.notEqual(afterParams, -1, `unbalanced parentheses in ${name}`);
  const open = source.indexOf("{", afterParams);
  assert.notEqual(open, -1, `${name} has no body`);
  let depth = 0;
  for (let i = open; i < source.length; i++) {
    if (source[i] === "{") depth++;
    else if (source[i] === "}") {
      depth--;
      if (depth === 0) return source.slice(start, i + 1);
    }
  }
  throw new Error(`unbalanced braces in ${name}`);
}

const NON_MODAL = /modal=\{false\}/;

const PERMISSION = code(read("features/chat/permission-mode-select.tsx"));
const SHEET = code(read("features/chat/chat-settings-sheet.tsx"));

test("the settings-panel permission dropdown keeps its modal shield", () => {
  const dropdown = declaration(PERMISSION, "PermissionModeDropdown");
  assert.doesNotMatch(
    dropdown,
    NON_MODAL,
    "PermissionModeDropdown renders in ChatSettingsPanel's Tools section, directly " +
      "above Max Tool Calls Per Message and Max Tool Call Duration. Without the modal " +
      "shield the press that dismisses it commits one of those sliders on pointerdown.",
  );
  assert.match(
    dropdown,
    /<MenuDismissGuard\s*\/>/,
    "the guard stays mounted alongside the shield",
  );
});

test("the composer permission pill stays off the modal layer", () => {
  const pill = declaration(PERMISSION, "PermissionModeComposerPill");
  assert.match(
    pill,
    NON_MODAL,
    "the composer pill is on the hot chat path and must stay non-modal",
  );
  assert.match(pill, /<MenuDismissGuard\s*\/>/);
});

test("no menu declared inside the run-settings panel is non-modal", () => {
  assert.doesNotMatch(
    SHEET,
    NON_MODAL,
    "chat-settings-sheet.tsx renders sliders; every menu it declares stays modal",
  );
  assert.match(
    SHEET,
    /<DropdownMenu modal=\{true\}>/,
    "the preset menu is the other modal menu on this panel",
  );
});

/** Resolve a module specifier seen in `from` to candidate source paths. */
function candidates(spec: string, fromDir: string): string[] {
  const rel = spec.startsWith("@/")
    ? spec.slice(2)
    : path.relative(SRC, path.resolve(path.join(SRC, fromDir), spec));
  return [`${rel}.tsx`, `${rel}/index.tsx`, `${rel}/index.ts`, `${rel}.ts`];
}

/**
 * Body of `name` as exported by `spec`, following one level of barrel re-export
 * so `@/features/rag` cannot hide a menu behind an index file.
 */
function exportedDeclaration(
  spec: string,
  fromDir: string,
  name: string,
  depth = 0,
): { body: string; file: string } | null {
  if (depth > 2) return null;
  for (const file of candidates(spec, fromDir)) {
    let source: string;
    try {
      source = code(read(file));
    } catch {
      continue;
    }
    try {
      return { body: declaration(source, name), file };
    } catch {
      // Not declared here: follow a re-export that names it.
      const dir = path.dirname(file);
      for (const [, names, next] of source.matchAll(
        /export\s+(?:type\s+)?\{([^}]+)\}\s+from\s+"([^"]+)"/g,
      )) {
        const exported = names
          .split(",")
          .map((n) => (n.split(" as ").pop() ?? "").trim());
        if (!exported.includes(name)) continue;
        const found = exportedDeclaration(next, dir, name, depth + 1);
        if (found) return found;
      }
      for (const [, next] of source.matchAll(/export\s+\*\s+from\s+"([^"]+)"/g)) {
        const found = exportedDeclaration(next, dir, name, depth + 1);
        if (found) return found;
      }
    }
  }
  return null;
}

test("no non-modal menu component is rendered into the run-settings panel", () => {
  const imports = [
    ...SHEET.matchAll(
      /import\s+(?:type\s+)?\{([^}]+)\}\s+from\s+"(\.[^"]+|@\/[^"]+)"/g,
    ),
  ];
  const offenders: string[] = [];
  let checked = 0;
  for (const [, names, spec] of imports) {
    for (const raw of names.split(",")) {
      const name = raw.split(" as ").pop()?.trim() ?? "";
      if (!name || !/^[A-Z]/.test(name)) continue;
      // Only components the panel actually renders.
      if (!new RegExp(`<${name}[\\s/>]`).test(SHEET)) continue;
      const found = exportedDeclaration(spec, "features/chat", name);
      if (!found) continue;
      checked++;
      if (NON_MODAL.test(found.body)) {
        offenders.push(`${name} (${found.file})`);
      }
    }
  }
  // A sweep that resolves nothing would pass without measuring anything.
  assert.ok(
    checked >= 10,
    `only resolved ${checked} rendered components; the sweep is not reaching the tree`,
  );
  assert.deepEqual(
    offenders,
    [],
    "these components render a non-modal menu inside the run-settings panel, where a " +
      `dismissing pointerdown can commit a slider: ${offenders.join(", ")}`,
  );
});
