// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * NO DESCENDANT-ARGUMENT `:has()` ON AN ANCESTOR OF THE THREAD.
 *
 * A `:has()` whose argument is a descendant selector has to be re-checked whenever anything is
 * inserted or removed anywhere inside the subject, and answering it means WALKING the subject's
 * subtree. On an ancestor of the chat thread that walk is the whole thread, and it happens for
 * every message that mounts, every token that streams and every deferred code fence that
 * upgrades.
 *
 * It is a traversal, not a restyle. Blink's `UpdateLayoutTree.elementCount` for one inserted span
 * is 1 with both rules in their child form, 2 with one still in descendant form and 3 with both:
 * only the subjects are restyled. That is why no amount of `contain:` helps, and why
 * `content-visibility: auto` on the message roots does not either (measured at -7%): the argument
 * re-check walks skipped content too. The only lever is the combinator.
 *
 * CHROMIUM ONLY. On a synthetic thread with the same ancestor chain and the built Studio
 * stylesheet, at 300,464 elements, one inserted span costs 1.20 / 1.29 / 5.63 / 10.30 ms for
 * plain / child / one descendant rule / both in Chromium, against 4.33 / 4.58 / 4.58 / 4.33 ms in
 * WebKitGTK and 4.65 / 4.72 / 4.45 / 5.10 ms in Firefox. The other two engines are flat, so this
 * is free where it does not help. This test still guards the shape in every engine, because the
 * regression it prevents is a Chromium one and Studio runs in a browser too.
 *
 * Measured at the 500K rung, corpus 23cd2464, on a 357,843-element thread, as the cost of
 * appending ONE EMPTY span inside a message, in two concurrent arms:
 *
 *   every rule in place                                        17.5 / 18.6 ms
 *   sidebar wrapper's rule alone deleted                        8.7 /  9.0 ms
 *   chat wrapper's rule alone deleted                           9.8 /  9.3 ms
 *   both deleted                                               0.10 / 0.10 ms
 *   the other eleven `:has()` rules the bisect kept, deleted   17.2 / 19.2 ms
 *   all 142 `:has()` rules in the bundle deleted               0.10 / 0.10 ms
 *   the same span appended to <body> instead                   0.10 / 0.10 ms
 *
 * So those two rules were the whole cost and no other rule contributed. This test is what stops
 * a third one being added, because the symptom is a frame rate on a long thread and nothing
 * about writing `has-[[data-x]]` looks expensive.
 *
 * IT IS A SOURCE TEST AND THAT IS A REAL LIMIT. It reads the two elements it knows about. A
 * `:has()` utility introduced on some other ancestor of the thread would not be caught here, and
 * the honest guard for that is the perf ladder, not this file.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import { openingTag } from "./helpers/tsx-ast.ts";

const read = (rel: string): string =>
  readFileSync(fileURLToPath(new URL(rel, import.meta.url)), "utf8");

const parse = (rel: string): ts.SourceFile =>
  ts.createSourceFile(rel, read(rel), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);

/** Every string literal in the file, so a className built by `cn(...)` is covered too. */
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
 * Tailwind's `has-*` variants, and which of them carry a DESCENDANT argument.
 *
 *   has-[>[data-x]]        child      cheap: only a change to the subject's own child list can
 *                                     change the answer, so a mutation deep in the thread is
 *                                     skipped outright
 *   has-[[data-x]]         descendant expensive: any insertion or removal in the subtree
 *   has-data-[variant=v]   descendant expensive: shorthand for has-[[data-variant=v]]
 *   has-aria-invalid       descendant expensive
 *   group-has-[...]        descendant expensive, on whichever element carries the `group`
 *
 * Matching is done on the class token so that `has-[>...]` is not read as a descendant form by a
 * looser substring test, which is the one way this check could pass while the defect is present.
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
  // The rule is still THERE, in its child form. Dropping it entirely would also pass the
  // assertion above and would silently change what an inset sidebar looks like.
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

/*
 * THE CHILD COMBINATOR IS ONLY EQUIVALENT IF THE NOTICE IS A CHILD.
 *
 * Without this the previous test is satisfied by a selector that matches nothing, the height is
 * never reserved, and the first message reads under the opaque header bar. That is a visible
 * regression with no failing test, which is exactly the shape this file exists to prevent.
 */
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

  // A JSX expression container is not a DOM node, so `{cond && <ChatModelNotice/>}` still renders
  // a direct child. Unwrapping one level of `{...}` and of `&&` is therefore correct, and going
  // deeper than that would start accepting real wrappers.
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

/*
 * And the notice really is the element the selector names. If `data-chat-model-notice` moved off
 * the component's root onto something inside it, the child combinator would stop matching while
 * every assertion above still passed.
 */
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
