// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A rail floor may only be reserved when what it protects is on screen: the rail is
 * bottom-anchored, so dead space lifts every visible card. Geometry lives in Playwright.
 */

import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import { openingTag } from "./helpers/tsx-ast.ts";

const src = (relative: string): URL =>
  new URL(`../src/${relative}`, import.meta.url);

const parse = (path: URL, name: string): ts.SourceFile =>
  ts.createSourceFile(
    name,
    readFileSync(path, "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );

const provider = parse(src("app/provider.tsx"), "provider.tsx");

const RAIL_ANCHOR = "pointer-events-none fixed bottom-0 right-4";

/** Literal rail children miss Tauri `{children}`; `positioned={false}` misses prop-less cards. */
function railChildren(): string[] {
  const names = new Set<string>();
  const visit = (node: ts.Node): void => {
    if (ts.isJsxElement(node)) {
      const className = node.openingElement.attributes.properties.find(
        (p): p is ts.JsxAttribute =>
          ts.isJsxAttribute(p) && p.name.getText() === "className",
      );
      if (className?.getText().includes(RAIL_ANCHOR)) {
        for (const child of node.children) {
          const name = openingTag(child)?.tagName.getText();
          if (name && /^[A-Z]/.test(name)) names.add(name);
        }
      }
    }
    const tag = openingTag(node);
    if (tag && /^[A-Z]/.test(tag.tagName.getText())) {
      const stacked = tag.attributes.properties.some(
        (p) =>
          ts.isJsxAttribute(p) &&
          p.name.getText() === "positioned" &&
          p.initializer &&
          ts.isJsxExpression(p.initializer) &&
          p.initializer.expression?.kind === ts.SyntaxKind.FalseKeyword,
      );
      if (stacked) names.add(tag.tagName.getText());
    }
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(provider, visit);
  return [...names].sort();
}

function sourceOf(name: string): { path: URL; label: string } | null {
  let specifier: string | null = null;
  for (const statement of provider.statements) {
    if (!ts.isImportDeclaration(statement)) continue;
    const bindings = statement.importClause?.namedBindings;
    if (!bindings || !ts.isNamedImports(bindings)) continue;
    if (bindings.elements.some((el) => el.name.getText() === name)) {
      specifier = (statement.moduleSpecifier as ts.StringLiteral).text;
    }
  }
  if (!specifier?.startsWith("@/")) return null;

  const resolve = (relative: string): URL | null => {
    for (const candidate of [`${relative}.tsx`, `${relative}.ts`]) {
      const url = src(candidate);
      if (existsSync(fileURLToPath(url))) return url;
    }
    return null;
  };

  const base = specifier.slice(2);
  const direct = resolve(base);
  if (direct) return { path: direct, label: base };

  const index = resolve(`${base}/index`);
  if (!index) return null;
  const barrel = readFileSync(index, "utf8");
  const re = new RegExp(
    `export\\s*\\{[^}]*\\b${name}\\b[^}]*\\}\\s*from\\s*"\\./([^"]+)"`,
  );
  const hop = barrel.match(re);
  if (!hop) return null;
  const followed = resolve(`${base}/${hop[1]}`);
  return followed ? { path: followed, label: `${base}/${hop[1]}` } : null;
}

/** Any spelling counts; `min-h-0` only removes the flex auto default, so it is no floor. */
const FLOOR_TOKEN = /(^|:)min-h-(?!0$)\S+/;

/** getText() keeps the quotes, which are not class tokens. */
const classTokens = (literal: ts.Node): string[] =>
  literal.getText().replace(/["'`]/g, " ").split(/\s+/).filter(Boolean);

const isFloor = (token: string): boolean => FLOOR_TOKEN.test(token);

function floorLiterals(file: ts.SourceFile): ts.Node[] {
  const found: ts.Node[] = [];
  const visit = (node: ts.Node): void => {
    if (
      (ts.isStringLiteral(node) ||
        ts.isNoSubstitutionTemplateLiteral(node) ||
        ts.isTemplateExpression(node)) &&
      classTokens(node).some(isFloor)
    ) {
      found.push(node);
    }
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(file, visit);
  return found;
}

/** Read from the tree, not by name: substring matches accept toggles and footer slots. */
function notesPanels(): { components: Set<string>; slots: Set<string> } {
  const components = new Set<string>();
  const slots = new Set<string>();
  const dir = fileURLToPath(src("components"));
  for (const entry of readdirSync(dir, {
    recursive: true,
    withFileTypes: true,
  })) {
    if (!entry.isFile() || !entry.name.endsWith(".tsx")) continue;
    const text = readFileSync(join(entry.parentPath, entry.name), "utf8");
    if (!text.includes("UPDATE_NOTES_ROOT_CLASS")) continue;
    for (const hit of text.matchAll(/export function ([A-Z]\w*)/g)) {
      components.add(hit[1]);
    }
    for (const hit of text.matchAll(/data-slot="([^"]+)"/g)) slots.add(hit[1]);
  }
  return { components, slots };
}

const NOTES = notesPanels();

function gatedByHas(literal: ts.Node): boolean {
  const tokens = classTokens(literal).filter(isFloor);
  return (
    tokens.length > 0 &&
    tokens.every((token) => {
      const slot = token.match(/has-\[\[data-slot=([^\]]+)\]\]:min-h-/);
      return slot !== null && NOTES.slots.has(slot[1]);
    })
  );
}

/** `whenTrue` only: a floor in the else branch applies whenever the predicate is false. */
function branchConditions(literal: ts.Node): string[] {
  const conditions: string[] = [];
  let node: ts.Node = literal;
  let parent = node.parent;
  while (parent && !ts.isJsxAttribute(parent)) {
    if (ts.isConditionalExpression(parent) && parent.whenTrue === node) {
      conditions.push(parent.condition.getText());
    }
    node = parent;
    parent = parent.parent;
  }
  return conditions;
}

/** The className painted inside this floor's own slot: its element's first child. */
function paintedSurface(literal: ts.Node): string | null {
  let node: ts.Node | undefined = literal;
  while (node && !ts.isJsxAttribute(node)) node = node.parent;
  const opening = node?.parent?.parent;
  if (!opening || !ts.isJsxOpeningElement(opening)) return null;
  const owner = opening.parent;
  if (!ts.isJsxElement(owner)) return null;
  for (const child of owner.children) {
    const tag = openingTag(child);
    if (!tag) continue;
    const className = tag.attributes.properties.find(
      (p): p is ts.JsxAttribute =>
        ts.isJsxAttribute(p) && p.name.getText() === "className",
    );
    return className?.initializer?.getText() ?? "";
  }
  return null;
}

/** The predicate must render the notes, not just the toggle, which shows while closed. */
function gatesTheNotes(source: string, condition: string): boolean {
  const name = condition.trim();
  if (!/^[A-Za-z_$][\w$]*$/.test(name)) return false;
  const rendered = new RegExp(
    `\\b${name}\\b[^;{}]{0,160}?(\\?|&&)\\s*\\(?\\s*<([A-Za-z][\\w.]*)`,
    "g",
  );
  for (const hit of source.matchAll(rendered)) {
    if (NOTES.components.has(hit[2])) return true;
  }
  return false;
}

const CHILDREN = railChildren();

test("the rail's children are all resolvable, or this file proves nothing", () => {
  assert.ok(
    CHILDREN.length >= 3,
    `provider.tsx: found ${CHILDREN.length} cards in a rail anchored on
"${RAIL_ANCHOR}". Either the rail moved or its children are no longer
rendered inline, and every check below just skipped silently.`,
  );
  for (const name of CHILDREN) {
    assert.ok(
      sourceOf(name),
      `provider.tsx: cannot follow <${name} /> to a source file, so its floor
cannot be checked. Add the import path it resolves through.`,
    );
  }
});

for (const name of CHILDREN) {
  test(`${name} reserves no height it may not paint`, () => {
    const resolved = sourceOf(name);
    assert.ok(resolved, `<${name} /> did not resolve`);
    const file = parse(resolved.path, `${name}.tsx`);
    const source = readFileSync(resolved.path, "utf8");
    // Invisible to the analysis below; the rail keeps its geometry in CSS.
    assert.doesNotMatch(
      source,
      /\bminHeight\s*:/,
      `${resolved.label}: a floor is set from JS. The rail's cards floor in CSS
so the gate can be read here and by the browser suite; an inline minHeight is
reserved unconditionally and nothing checks it.`,
    );

    const literals = floorLiterals(file);
    if (literals.length === 0) return;

    for (const literal of literals) {
      const gated =
        gatedByHas(literal) ||
        branchConditions(literal).some((condition) =>
          gatesTheNotes(source, condition),
        );
      assert.ok(
        gated,
        `${resolved.label}: a min-h floor is applied without being gated on the
panel it exists to protect, so the card reserves height it may paint none of.
In a bottom-anchored rail that dead space lifts every visible card off the
corner. This is PR #10117, fixed by PR #10229.

Gate it one of the two ways already in use:
  has-[[data-slot=update-release-notes]]:min-h-[...]   (CSS)
  changelogPanelOpen ? "min-h-[...]" : "shrink-0"      (React, using the same
    predicate that renders the notes or changelog panel)

A predicate that renders something else does not count: changelogAvailable
renders the changelog toggle and is true while the panel is still closed, which
is the state that reserved 64.9px in #10117.

Floor found in: ${literal.getText().slice(0, 200)}
Ternary conditions around it: ${branchConditions(literal).join(", ") || "(none)"}`,
      );

      // This floor's own painted child: another branch's `grow` would vouch for the wrong surface.
      const surface = paintedSurface(literal);
      assert.ok(
        surface,
        `${resolved.label}: cannot find the element painted inside the floored
slot, so nothing checks that the floor is filled.`,
      );
      assert.match(
        surface,
        /\bgrow\b/,
        `${resolved.label}: the floored slot's painted child has no \`grow\`, so
short content leaves an unpainted gap inside the height the floor reserved.

Painted child: ${surface.slice(0, 160)}`,
      );
    }
  });
}
