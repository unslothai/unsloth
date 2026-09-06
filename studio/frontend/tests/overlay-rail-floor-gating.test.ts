// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A card in the bottom-right rail may not reserve height it does not paint.
 *
 * The rail is a bottom-anchored column, so reserved-but-unpainted height inside
 * a card does not disappear: it lifts everything the reader can see off the
 * corner. #10117 copied the desktop updater's
 * `min-h-[calc(117px+93px*var(--ui-font-scale,1))]` onto the llama.cpp banner
 * while making its changelog render only once opened, so the default state
 * reserved 204.2px and painted 147.3px and the whole stack sat 64.9px high.
 * #10229 fixed it.
 *
 * Both suites ran on #10117 and both went green, and the reason is worth
 * keeping in mind while reading this file. The source suite of the day asserted
 * that the llama card CARRIED the floor and, in the same commit, that it did
 * not carry `shrink-0` - the bug was pinned as a contract. Any test that only
 * compares one card's class string against another's will agree that two cards
 * are consistent while both are wrong.
 *
 * So this file does not check for a spelling. It enumerates the rail's children
 * out of provider.tsx, follows each to its source, and applies one rule:
 *
 *   a floor may only be reserved when the thing it protects is on screen.
 *
 * Two gates satisfy it, and both are in use. `has-[[data-slot=...]]:` makes CSS
 * ask whether the panel rendered; a React predicate works if it is the same
 * predicate that renders the panel. A floor sitting in the `positioned` ternary
 * with no other gate - exactly what #10117 shipped - satisfies neither.
 *
 * Read from the source: the node suite has no DOM to compute styles in. The
 * geometry itself is checked in tests/studio/playwright_update_banner_layout.py,
 * which measures each slot against the surface painted inside it.
 */

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
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

/** The rails, matched on the corner they are anchored to. */
const RAIL_ANCHOR = "pointer-events-none fixed bottom-0 right-4";

/** Every component rendered directly into a rail, by tag name. */
function railChildren(): string[] {
  const names = new Set<string>();
  const visit = (node: ts.Node): void => {
    if (ts.isJsxElement(node)) {
      const opening = node.openingElement;
      const className = opening.attributes.properties.find(
        (p): p is ts.JsxAttribute =>
          ts.isJsxAttribute(p) && p.name.getText() === "className",
      );
      if (className?.getText().includes(RAIL_ANCHOR)) {
        for (const child of node.children) {
          const tag = openingTag(child);
          const name = tag?.tagName.getText();
          // {children} in the Tauri layer is followed to its own rail by the
          // second match; a lower-case tag is a plain element, not a card.
          if (name && /^[A-Z]/.test(name)) names.add(name);
        }
      }
    }
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(provider, visit);
  return [...names].sort();
}

/**
 * Where `name` is imported from in provider.tsx, resolved to a file on disk.
 * Barrels are followed one hop, which is as far as the rail's children go.
 */
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

  // A barrel: find `export { name } from "./x"` and follow it.
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

/** Class tokens carrying a `min-h-[calc(...)]` floor, per string literal. */
function floorLiterals(file: ts.SourceFile): ts.Node[] {
  const found: ts.Node[] = [];
  const visit = (node: ts.Node): void => {
    if (
      (ts.isStringLiteral(node) ||
        ts.isNoSubstitutionTemplateLiteral(node) ||
        ts.isTemplateExpression(node)) &&
      node.getText().includes("min-h-[calc(")
    ) {
      found.push(node);
    }
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(file, visit);
  return found;
}

/** Is every floor in `literal` gated by the CSS `:has()` on a rendered slot? */
function gatedByHas(literal: ts.Node): boolean {
  const tokens = literal
    .getText()
    .split(/\s+/)
    .filter((token) => token.includes("min-h-[calc("));
  return (
    tokens.length > 0 &&
    tokens.every((token) =>
      /has-\[\[data-slot=[^\]]+\]\]:min-h-\[calc\(/.test(token),
    )
  );
}

/**
 * The conditions this literal is the `whenTrue` branch of, innermost first.
 *
 * Only `whenTrue`. A floor on the other side of a ternary applies when its
 * predicate is FALSE, which is the opposite of a gate: `showFailure ?
 * "shrink-0" : "<floor>"` - the shape #8367 shipped - floors every state
 * except the one state named, including the states with no notes to protect.
 */
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

/**
 * Does `condition` also decide whether an element renders?
 *
 * The point of the gate is that the floor and the panel it protects turn on
 * together. A condition that only picks between two class strings - `positioned`
 * being the one that mattered - reserves height for a panel it has no opinion
 * about.
 */
function gatesAnElement(source: string, condition: string): boolean {
  const name = condition.trim();
  if (!/^[A-Za-z_$][\w$]*$/.test(name)) return false;
  const rendered = new RegExp(
    `\\b${name}\\b[^;{}]{0,160}?(\\?|&&)\\s*\\(?\\s*<[A-Za-z]`,
  );
  return rendered.test(source);
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
    const literals = floorLiterals(file);
    if (literals.length === 0) return; // No floor, nothing to gate.

    for (const literal of literals) {
      const gated =
        gatedByHas(literal) ||
        branchConditions(literal).some((condition) =>
          gatesAnElement(source, condition),
        );
      assert.ok(
        gated,
        `${resolved.label}: a min-h-[calc(...)] floor is applied without being
gated on the panel it exists to protect, so the card reserves height it may
paint none of. In a bottom-anchored rail that dead space lifts every visible
card off the corner. This is PR #10117, fixed by PR #10229.

Gate it one of the two ways already in use:
  has-[[data-slot=update-release-notes]]:min-h-[calc(...)]   (CSS)
  changelogPanelOpen ? "min-h-[calc(...)]" : "shrink-0"      (React, using the
    same predicate that renders the panel)

Floor found in: ${literal.getText().slice(0, 200)}
Ternary conditions around it: ${branchConditions(literal).join(", ") || "(none)"}`,
      );
    }

    // A gated floor still has to be filled, or the card paints short inside a
    // slot it legitimately asked for and the gap comes back the other way.
    assert.match(
      source,
      /className="relative flex [^"]*\bgrow\b/,
      `${resolved.label}: the card carries a floor but its painted surface has
no \`grow\`, so short content leaves an unpainted gap inside the slot the
floor reserved.`,
    );
  });
}
