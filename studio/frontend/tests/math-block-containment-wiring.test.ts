// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { MATH_BLOCK_CLASS } from "../src/components/assistant-ui/math-block-marker.ts";
import {
  MATH_BLOCK_CONTAINMENT_ATTRIBUTE,
  MATH_BLOCK_CONTAINMENT_ON,
} from "../src/components/assistant-ui/math-block-mode.ts";

/*
 * THE THREE PIECES ONLY WORK TOGETHER, and nothing in the type system joins them:
 *
 *   1. the marker has to be composed onto the MATHS plugin, not passed as a `rehypePlugins` prop,
 *      because that prop switches off Streamdown's `allowedTags` sanitizer;
 *   2. the stylesheet has to name the class the marker writes and the attribute the resolver sets;
 *   3. `main.tsx` has to set that attribute before the first render.
 *
 * Each is checked against the real source. Where a check is a string search it says what it is
 * defending, so a rename that breaks the join fails here by name rather than by measuring as a
 * change that does nothing.
 */

const read = (relative: string): string =>
  readFileSync(new URL(relative, import.meta.url), "utf8");

const MARKDOWN_TEXT = read("../src/components/assistant-ui/markdown-text.tsx");
const INDEX_CSS = read("../src/index.css");
const MAIN_TSX = read("../src/main.tsx");

test("the marker is composed onto the maths plugin", () => {
  assert.ok(
    MARKDOWN_TEXT.includes("createMathPlugin({ singleDollarTextMath: true })"),
    "PRECONDITION: the chat renderer still builds its own maths plugin",
  );
  assert.ok(
    MARKDOWN_TEXT.includes(
      "rehypePlugin: withMathBlockMarker(baseMath.rehypePlugin)",
    ),
    "the marker wraps the maths plugin's own rehype pass",
  );
});

test("the chat renderer does NOT pass a rehypePlugins prop", () => {
  // Passing one makes Streamdown skip its `allowedTags` sanitizer schema, because it only installs
  // that schema while `rehypePlugins === defaultRehypePlugins`. This is the reason the marker is
  // composed rather than appended, and it is worth more than the class it buys.
  assert.ok(
    MARKDOWN_TEXT.includes("allowedTags={STREAMDOWN_ALLOWED_TAGS}"),
    "PRECONDITION: the chat renderer relies on the allowedTags sanitizer",
  );
  assert.ok(
    !/\brehypePlugins=\{/.test(MARKDOWN_TEXT),
    "no rehypePlugins prop, or the sanitizer schema above is silently dropped",
  );
});

test("the stylesheet rule names the class, the attribute and both declarations", () => {
  const rule = INDEX_CSS.slice(
    INDEX_CSS.indexOf(`html[${MATH_BLOCK_CONTAINMENT_ATTRIBUTE}=`),
  ).slice(0, 400);
  assert.ok(rule.length > 0, "PRECONDITION: the gated rule is present at all");

  assert.ok(
    rule.includes(
      `html[${MATH_BLOCK_CONTAINMENT_ATTRIBUTE}="${MATH_BLOCK_CONTAINMENT_ON}"]`,
    ),
    "the rule is gated on the attribute the resolver sets",
  );
  assert.ok(rule.includes(".aui-thread-root"), "scoped to the chat thread");
  assert.ok(rule.includes(`.${MATH_BLOCK_CLASS}`), "names the marker class");
  assert.ok(
    rule.includes(".katex-display"),
    "names display maths, which needs no marker",
  );
  assert.ok(
    rule.includes("content-visibility: auto"),
    "the declaration under test",
  );
  assert.ok(
    rule.includes("contain-intrinsic-size: auto 7.5rem"),
    "the placeholder, with the `auto` keyword so a rendered block remembers its real size",
  );
});

test("the rule is armed by nothing except that attribute", () => {
  // PRECONDITION: the stylesheet already carries an UNGATED `content-visibility: visible` on code
  // blocks, put there to stop a flicker. So "no content-visibility anywhere else" would be false
  // and this test has to be specific about which declaration it is defending.
  assert.ok(
    INDEX_CSS.includes("content-visibility: visible !important"),
    "the code-block flicker rule is still there",
  );

  // A second, ungated copy of `auto` would turn the flag into decoration.
  const occurrences = INDEX_CSS.split("content-visibility: auto;").length - 1;
  assert.equal(
    occurrences,
    1,
    "exactly one `content-visibility: auto` declaration in the whole stylesheet",
  );
  const gateAt = INDEX_CSS.indexOf(`html[${MATH_BLOCK_CONTAINMENT_ATTRIBUTE}=`);
  const declarationAt = INDEX_CSS.indexOf("content-visibility: auto;");
  assert.ok(
    gateAt >= 0 && declarationAt > gateAt,
    "and it sits inside the gated rule",
  );
});

test("startup applies the mode before the first render", () => {
  assert.ok(
    MAIN_TSX.includes("applyMathBlockContainment()"),
    "the attribute is applied at startup",
  );
  const applyAt = MAIN_TSX.indexOf("applyMathBlockContainment()");
  const renderAt = MAIN_TSX.indexOf("function renderApp");
  assert.ok(renderAt > 0, "PRECONDITION: main.tsx still defines renderApp");
  assert.ok(
    applyAt < renderAt,
    "before the render, or the first thread that mounts relayouts when it is armed",
  );
});
