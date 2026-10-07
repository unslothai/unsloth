// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * No descendant-argument `:has()` on an ancestor of the thread: Chromium re-walks the whole
 * thread on every insertion (measured as the entire per-append cost). Child form `has-[>...]`
 * is cheap. Source test only; other ancestors are guarded by the perf ladder.
 */

import assert from "node:assert/strict";
import test from "node:test";

import ts from "typescript";

import { openingTag } from "./helpers/tsx-ast.ts";

import { readText } from "./helpers/kit.ts";

const parse = (rel: string): ts.SourceFile =>
  ts.createSourceFile(rel, readText(rel), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);

const stringLiterals = (source: ts.SourceFile): string[] => {
  const out: string[] = [];
  const walk = (node: ts.Node): void => {
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      out.push(node.text);
    }
    ts.forEachChild(node, walk);
  };
  walk(source);
  return out;
};

/*
 * has-[>[x]] is child (cheap); has-[[x]], has-data-*, has-aria-*, group-has-* are descendant.
 * Match on the class token so `has-[>...]` is not misread by a substring test.
 */
const descendantHasUtilities = (className: string): string[] =>
  className
    .split(/\s+/)
    .filter((token) => {
      const variant = token.startsWith("group-has-") ? token.slice("group-".length) : token;
      if (!variant.startsWith("has-")) return false;
      if (variant.startsWith("has-[>")) return false;
      return true;
    });

test("the sidebar wrapper, an ancestor of the whole app, has no descendant-argument :has()", () => {
  const source = parse("../src/components/ui/sidebar.tsx");
  const wrappers = stringLiterals(source).filter((s) => s.includes("group/sidebar-wrapper"));
  assert.equal(
    wrappers.length,
    1,
    "expected exactly one className carrying group/sidebar-wrapper; if this element was "
      + "restructured the assertion below is no longer looking at the ancestor of the thread",
  );
  assert.deepEqual(
    descendantHasUtilities(wrappers[0]),
    [],
    `sidebar-wrapper className carries a descendant-argument :has(): ${wrappers[0]}`,
  );
  // The rule must remain in child form; dropping it would change the inset sidebar.
  assert.ok(
    wrappers[0].includes("has-[>[data-variant=inset]]:bg-sidebar"),
    `sidebar-wrapper lost the inset background rule: ${wrappers[0]}`,
  );
});

test("the chat wrapper that contains the thread has no descendant-argument :has()", () => {
  const source = parse("../src/features/chat/chat-page.tsx");
  const wrappers = stringLiterals(source).filter((s) =>
    s.includes("--studio-chat-notice-height:"),
  );
  assert.equal(
    wrappers.length,
    1,
    "expected exactly one className declaring --studio-chat-notice-height",
  );
  assert.deepEqual(
    descendantHasUtilities(wrappers[0]),
    [],
    `the chat wrapper carries a descendant-argument :has(): ${wrappers[0]}`,
  );
  assert.ok(
    wrappers[0].includes("has-[>[data-chat-model-notice]]:[--studio-chat-notice-height:2.25rem]"),
    `the chat wrapper lost the notice-height rule: ${wrappers[0]}`,
  );
});

/* The child combinator only matches if the notice is a direct child. */
test("ChatModelNotice renders a DIRECT child of the element declaring the notice height", () => {
  const source = parse("../src/features/chat/chat-page.tsx");

  const declaringDiv = ((): ts.JsxElement | null => {
    let found: ts.JsxElement | null = null;
    const walk = (node: ts.Node): void => {
      if (found) return;
      if (ts.isJsxElement(node)) {
        const tag = node.openingElement;
        for (const attr of tag.attributes.properties) {
          if (!ts.isJsxAttribute(attr)) continue;
          if (attr.name.getText() !== "className") continue;
          const init = attr.initializer;
          if (!init || !ts.isStringLiteral(init)) continue;
          if (init.text.includes("--studio-chat-notice-height:")) {
            found = node;
            return;
          }
        }
      }
      ts.forEachChild(node, walk);
    };
    walk(source);
    return found;
  })();

  assert.ok(declaringDiv, "could not find the element declaring --studio-chat-notice-height");

  // A JSX expression container is not a DOM node, so unwrap one level of `{...}` and `&&` only.
  const directChildTagNames = declaringDiv.children.flatMap((child): string[] => {
    const fromNode = (node: ts.Node): string[] => {
      const tag = openingTag(node);
      if (tag) return [tag.tagName.getText()];
      if (ts.isParenthesizedExpression(node)) return fromNode(node.expression);
      if (
        ts.isBinaryExpression(node)
        && node.operatorToken.kind === ts.SyntaxKind.AmpersandAmpersandToken
      ) {
        return fromNode(node.right);
      }
      if (ts.isConditionalExpression(node)) {
        return [...fromNode(node.whenTrue), ...fromNode(node.whenFalse)];
      }
      return [];
    };
    if (ts.isJsxExpression(child)) {
      return child.expression ? fromNode(child.expression) : [];
    }
    return fromNode(child);
  });

  assert.ok(
    directChildTagNames.includes("ChatModelNotice"),
    "ChatModelNotice is no longer a direct child of the element whose "
      + "has-[>[data-chat-model-notice]] rule reserves its height, so the height is never "
      + `reserved. Direct children seen: ${directChildTagNames.join(", ")}`,
  );
});

/* If the data attribute moved off the root, the child combinator would stop matching. */
test("data-chat-model-notice is on the root element ChatModelNotice returns", () => {
  const source = parse("../src/features/chat/components/chat-model-notice.tsx");
  let rootHasAttribute = false;
  const walk = (node: ts.Node): void => {
    if (rootHasAttribute) return;
    if (ts.isReturnStatement(node) && node.expression) {
      let expression: ts.Node = node.expression;
      while (ts.isParenthesizedExpression(expression)) expression = expression.expression;
      const tag = openingTag(expression);
      if (tag) {
        for (const attr of tag.attributes.properties) {
          if (!ts.isJsxAttribute(attr)) continue;
          if (attr.name.getText() === "data-chat-model-notice") rootHasAttribute = true;
        }
      }
    }
    ts.forEachChild(node, walk);
  };
  walk(source);
  assert.ok(
    rootHasAttribute,
    "no return in chat-model-notice.tsx yields a root element carrying "
      + "data-chat-model-notice, so has-[>[data-chat-model-notice]] cannot match",
  );
});

test("index.css has no descendant-argument :has() on the panels that hold the thread", () => {
  const css = readText("../src/index.css");
  const offenders =
    css.match(/(?:#chat-thread|\.chat-artifact-split|\.chat-thread-pane)[^{},]*:has\((?!\s*>)/g) ?? [];
  assert.deepEqual(offenders, [], `descendant-argument :has() on a thread ancestor: ${offenders.join(" | ")}`);
});
