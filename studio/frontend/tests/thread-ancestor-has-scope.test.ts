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
 * CHROMIUM ONLY. On a synthetic thread with the same ancestor chain and the built Unsloth
 * stylesheet, at 300,464 elements, one inserted span costs 1.20 / 1.29 / 5.63 / 10.30 ms for
 * plain / child / one descendant rule / both in Chromium, against 4.33 / 4.58 / 4.58 / 4.33 ms in
 * WebKitGTK and 4.65 / 4.72 / 4.45 / 5.10 ms in Firefox. The other two engines are flat, so this
 * is free where it does not help. This test still guards the shape in every engine, because the
 * regression it prevents is a Chromium one and Unsloth runs in a browser too.
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
import test from "node:test";

import ts from "@typescript/typescript6";

import { openingTag } from "./helpers/tsx-ast.ts";

import { readText } from "./helpers/kit.ts";

const parse = (rel: string): ts.SourceFile =>
  ts.createSourceFile(rel, readText(rel), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);

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
  const wrappers: string[] = [];
  const walk = (node: ts.Node): void => {
    const tag = openingTag(node);
    if (tag?.attributes.properties.some((attr) =>
      ts.isJsxAttribute(attr) && attr.name.getText() === "data-chat-content-root",
    )) {
      for (const attr of tag.attributes.properties) {
        if (ts.isJsxAttribute(attr) && attr.name.getText() === "className") {
          assert.ok(attr.initializer && ts.isStringLiteral(attr.initializer));
          wrappers.push(attr.initializer.text);
        }
      }
    }
    ts.forEachChild(node, walk);
  };
  walk(source);
  assert.equal(wrappers.length, 1, "expected exactly one chat content root");
  assert.deepEqual(
    descendantHasUtilities(wrappers[0]),
    [],
    `the chat wrapper carries a descendant-argument :has(): ${wrappers[0]}`,
  );
});

test("index.css has no descendant-argument :has() on the panels that hold the thread", () => {
  const css = readText("../src/index.css");
  const offenders =
    css.match(/(?:#chat-thread|\.chat-artifact-split|\.chat-thread-pane)[^{},]*:has\((?!\s*>)/g) ?? [];
  assert.deepEqual(offenders, [], `descendant-argument :has() on a thread ancestor: ${offenders.join(" | ")}`);
});
