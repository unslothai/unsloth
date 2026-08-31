// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  MATH_BLOCK_CLASS,
  MATH_DISPLAY_CLASS,
} from "../src/components/assistant-ui/math-block-marker.ts";
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
const MATH_BLOCK_MODE = read(
  "../src/components/assistant-ui/math-block-mode.ts",
);
const CONTAINMENT = read(
  "../src/components/assistant-ui/math-block-containment.ts",
);

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
  const gate = `html[${MATH_BLOCK_CONTAINMENT_ATTRIBUTE}="${MATH_BLOCK_CONTAINMENT_ON}"]`;
  const rules = INDEX_CSS.split(gate)
    .slice(1)
    .map((part) => part.slice(0, 220));
  assert.equal(
    rules.length,
    2,
    "PRECONDITION: two gated rules, one per population, because their heights differ 3x",
  );

  const [marked, display] = rules;
  for (const rule of rules) {
    assert.ok(rule.includes(".aui-thread-root"), "scoped to the chat thread");
    assert.ok(
      rule.includes("content-visibility: auto"),
      "the declaration under test",
    );
  }

  assert.ok(marked.includes(`.${MATH_BLOCK_CLASS}`), "names the marker class");
  assert.ok(
    display.includes(`.${MATH_DISPLAY_CLASS}`),
    "names the display class the renderer adds after KaTeX has run",
  );
  assert.equal(
    display.includes(".katex-display"),
    false,
    "and NOT `.katex-display` itself: a display carrying an equation number must not take " +
      "containment, because style containment scopes `katexEqnNo` and Chromium then renders " +
      "every numbered equation as (1). Measured, and engine dependent: the same fixture on " +
      "WebKitGTK 2.50.4 does not reproduce it.",
  );
  for (const rule of rules) {
    // Scoped to THESE rules rather than to the file, which carries other `:has()` rules that #9669
    // deliberately narrowed to direct children rather than removing.
    assert.equal(
      rule.includes(":has("),
      false,
      "the exemption is NOT expressed with `:has()` here, which was the measured owner of the " +
        "whole 500K scroll cost on Chromium (#9669); the renderer decides instead",
    );
  }

  /*
   * THE TWO PLACEHOLDERS ARE DIFFERENT, AND THAT IS THE POINT. A marked block is a PARAGRAPH of
   * prose holding a formula and measured a 138.04px mean; a display formula is one line of maths
   * and measured 49.13px. One shared value was tried and was 18px short on every paragraph and
   * 71px too tall on every formula, which queued up under a first-time scroller as a 3,995px
   * scrollbar excursion. If these two ever become equal again, that regression is back.
   */
  assert.ok(
    marked.includes("contain-intrinsic-size: auto 8.5rem"),
    "the paragraph placeholder, near the measured 138px mean",
  );
  assert.ok(
    display.includes("contain-intrinsic-size: auto 3rem"),
    "the formula placeholder, near the measured 49px mean",
  );
  assert.notEqual(
    marked.match(/contain-intrinsic-size: auto [\d.]+rem/)?.[0],
    display.match(/contain-intrinsic-size: auto [\d.]+rem/)?.[0],
    "one shared placeholder is the thing this pair replaced",
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

  // Two gated copies of `auto`, one per population. A third, ungated one would turn the flag
  // into decoration.
  const declarations = INDEX_CSS.split("content-visibility: auto;").length - 1;
  assert.equal(
    declarations,
    2,
    "exactly two `content-visibility: auto` declarations in the whole stylesheet",
  );
  const gateAt = INDEX_CSS.indexOf(`html[${MATH_BLOCK_CONTAINMENT_ATTRIBUTE}=`);
  assert.ok(gateAt >= 0, "PRECONDITION: the gate is present");
  assert.ok(
    INDEX_CSS.indexOf("content-visibility: auto;") > gateAt,
    "and the first of them sits after the gate",
  );
});

test("a print turns the containment off, for both populations", () => {
  /*
   * WHAT IS AT STAKE. Skipped content is not painted and the box keeps its placeholder height, so a
   * block still skipped when the reader prints comes out as an EMPTY 8.5rem or 3rem rectangle. The
   * Gecko report that got printing fixed there describes precisely that: "the contents are not
   * visible in the print preview. However, the height is reserved, resulting in a few empty pages."
   * (bugzilla.mozilla.org/show_bug.cgi?id=1907081, fixed in Firefox 130.)
   *
   * AND THE ENGINE DOES NOT DO IT FOR US. css-contain-2 lists on-screen, focused, selected and top
   * layer as the ways to be "relevant to the user" and says nothing about printing; the CSSWG
   * resolved to add a print carve-out on 2024-06-13 (w3c/csswg-drafts#10347) and the spec is still
   * unedited. Chromium and Gecko each fixed it themselves. WebKit has not: `ContentRelevancy` is
   * `OnScreen | Focused | IsInTopLayer | Selected` and nothing in its
   * `ContentVisibilityDocumentState` mentions printing. WebKit is what Studio renders through on
   * Linux, and it is an engine this feature ARMS on, since the `anchor-name` probe passes on Safari
   * 26 and on WebKitGTK 2.50.4. So this rule, not the engine, is what keeps the equations on the
   * page -- which makes it worth a test rather than a comment.
   *
   * NOT a `beforeprint` hook, unlike `code-fence-defer.tsx`, which has to script its print upgrade
   * because what it defers is not in the DOM. Here the maths is in the DOM and only its rendering
   * is skipped, so a media query is the whole fix and it covers `page.pdf()` too.
   */
  // Balanced-brace slices, so the assertions below are about a print block and not about whatever
  // the next 200 characters of the file happen to be, and every `@media print` in the file is a
  // candidate rather than just the first one.
  const printBlocks: string[] = [];
  for (
    let start = INDEX_CSS.indexOf("@media print");
    start >= 0;
    start = INDEX_CSS.indexOf("@media print", start + 1)
  ) {
    let depth = 0;
    for (let i = INDEX_CSS.indexOf("{", start); i < INDEX_CSS.length; i += 1) {
      if (INDEX_CSS[i] === "{") depth += 1;
      else if (INDEX_CSS[i] === "}") {
        depth -= 1;
        if (depth === 0) {
          printBlocks.push(INDEX_CSS.slice(start, i + 1));
          break;
        }
      }
    }
  }
  const printBlock =
    printBlocks.find((block) => block.includes(`.${MATH_BLOCK_CLASS}`)) ?? "";
  assert.notEqual(
    printBlock,
    "",
    "no `@media print` block in the stylesheet mentions the marked-block class, so a thread " +
      "printed before the reader has scrolled every formula into view loses them",
  );

  for (const cls of [MATH_BLOCK_CLASS, MATH_DISPLAY_CLASS]) {
    assert.ok(
      printBlock.includes(`.${cls}`),
      `the print override names .${cls}; a paragraph of prose holding a formula is lost by the ` +
        "same mechanism as the formula itself, so both populations need it",
    );
  }
  assert.ok(
    printBlock.includes("content-visibility: visible !important"),
    "`visible`, and important: the gated rules above are more specific than any unprefixed selector",
  );
  assert.ok(
    printBlock.includes("contain-intrinsic-size: none !important"),
    "and the placeholder height cleared, or a rendered block still prints at its fallback size",
  );

  // ANTI-VACUITY: the two gated `auto` rules really are the thing being overridden, and they are
  // still there to override. Without this the block above could be defending nothing.
  assert.equal(
    INDEX_CSS.split("content-visibility: auto;").length - 1,
    2,
    "PRECONDITION: the two gated declarations this print block exists to switch off",
  );
});

test("no comment anywhere claims this feature ships off", () => {
  // THIS HAS NOW GONE WRONG THREE TIMES. Flipping `SHIP_DEFAULT` to "contain" left prose in
  // `math-block-mode.ts`, `main.tsx` and `index.css` still saying the feature defaults to off;
  // fixing "all three" then missed a SECOND block in `index.css`, because the fix was a grep for
  // the passage already known about rather than for every statement of the default. A comment
  // that documents the opposite of the shipped behaviour is what someone diagnosing a rendering
  // problem or attempting a rollback will read and believe, so it is worth a test rather than
  // another round of care.
  //
  // Deliberately a search of the WHOLE text of each file, comments included, rather than of the
  // one block a previous fix touched.
  const STALE = [
    /OFF BY DEFAULT/i,
    /never arms this rule/i,
    /which is why this ships off/i,
    /`SHIP_DEFAULT`[^.]{0,80}is\s+"off"/i,
  ];
  for (const [name, text] of [
    ["index.css", INDEX_CSS],
    ["main.tsx", MAIN_TSX],
    ["math-block-mode.ts", MATH_BLOCK_MODE],
  ] as const) {
    for (const pattern of STALE) {
      assert.ok(
        !pattern.test(text),
        `${name} still documents the old off-by-default behaviour (${pattern})`,
      );
    }
  }

  // ANTI-VACUITY. The four patterns above are only meaningful if this file's text is actually
  // being searched; a bad path would make every assertion above pass on an empty string.
  for (const [name, text] of [
    ["index.css", INDEX_CSS],
    ["main.tsx", MAIN_TSX],
    ["math-block-mode.ts", MATH_BLOCK_MODE],
  ] as const) {
    assert.ok(text.length > 500, `PRECONDITION: ${name} was actually read`);
  }
  assert.ok(
    /SHIP_DEFAULT[^\n]*=[^\n]*"contain"/.test(MATH_BLOCK_MODE),
    "PRECONDITION: the shipped default really is `contain`, or this test is defending the wrong claim",
  );
});

test("every override name a comment advertises is one the code actually reads", () => {
  // Same failure family as the test above, and it cost a review round of its own: `index.css`
  // documented the rollback switches as `VITE_UNSLOTH_MATH_BLOCK` and `__UNSLOTH_MATH_BLOCK__`,
  // but the resolver reads the `_CONTAINMENT`-suffixed names. Vite substitutes the LITERAL
  // property name at build time (vite.dev/guide/env-and-mode), so the shorter build flag is never
  // consulted, and the shorter global is never read either. An operator rolling the feature back
  // by the documented names would set them, see containment stay on, and have no signal why.
  //
  // So: gather every override-looking token out of the prose and require each to be the real one.
  // A truncated or renamed variant fails by name.
  const BUILD = "VITE_UNSLOTH_MATH_BLOCK_CONTAINMENT";
  const RUNTIME = "__UNSLOTH_MATH_BLOCK_CONTAINMENT__";
  assert.ok(
    CONTAINMENT.includes(`import.meta.env.${BUILD}`),
    "PRECONDITION: the build flag really is read under this name",
  );
  assert.ok(
    CONTAINMENT.includes(RUNTIME),
    "PRECONDITION: the runtime flag really is read under this name",
  );

  for (const [name, text] of [
    ["index.css", INDEX_CSS],
    ["main.tsx", MAIN_TSX],
    ["math-block-mode.ts", MATH_BLOCK_MODE],
  ] as const) {
    for (const token of text.match(/\bVITE_UNSLOTH_MATH_BLOCK\w*/g) ?? []) {
      assert.equal(
        token,
        BUILD,
        `${name} advertises a build flag the code never reads`,
      );
    }
    for (const token of text.match(/\b__UNSLOTH_MATH_BLOCK\w*?__/g) ?? []) {
      assert.equal(
        token,
        RUNTIME,
        `${name} advertises a runtime flag the code never reads`,
      );
    }
  }

  // ANTI-VACUITY. The loops above pass trivially if nothing matched, which is also what a bad
  // path or a wholesale rename looks like.
  assert.ok(
    INDEX_CSS.includes(BUILD) && INDEX_CSS.includes(RUNTIME),
    "PRECONDITION: the stylesheet still documents both overrides",
  );
});

test("startup applies the mode before the first render", () => {
  // A LINE, not a substring. `includes` is satisfied by a commented-out call, which is exactly the
  // shape of the mutation that first slipped past this test.
  const lines = MAIN_TSX.split("\n");
  const callLine = lines.findIndex(
    (line) => line.trim() === "applyMathBlockContainment();",
  );
  assert.ok(
    callLine >= 0,
    "the attribute is applied at startup, on a line of its own and not in a comment",
  );
  const applyAt = MAIN_TSX.indexOf("\napplyMathBlockContainment();");
  const renderAt = MAIN_TSX.indexOf("function renderApp");
  assert.ok(renderAt > 0, "PRECONDITION: main.tsx still defines renderApp");
  assert.ok(
    applyAt < renderAt,
    "before the render, or the first thread that mounts relayouts when it is armed",
  );
});
