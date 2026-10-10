// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  MATH_BLOCK_CLASS,
  MATH_DISPLAY_CLASS,
} from "../src/components/assistant-ui/math-block-marker.ts";
import {
  MATH_BLOCK_CONTAINMENT_ATTRIBUTE,
  MATH_BLOCK_CONTAINMENT_ON,
} from "../src/components/assistant-ui/math-block-mode.ts";

import { readText } from "./helpers/kit.ts";

/* Marker on the maths rehype pass, stylesheet class/attribute, and main.tsx setup must agree. */

const MARKDOWN_TEXT = readText(
  "../src/components/assistant-ui/markdown-text.tsx",
);
const INDEX_CSS = readText("../src/index.css");
const MAIN_TSX = readText("../src/main.tsx");
const MATH_BLOCK_MODE = readText(
  "../src/components/assistant-ui/math-block-mode.ts",
);
const CONTAINMENT = readText(
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

test("the chat renderer's rehypePlugins pipeline carries the allowedTags merge itself", () => {
  // Streamdown merges allowedTags only for its default pipeline, so a passed one must carry it.
  assert.ok(
    MARKDOWN_TEXT.includes("allowedTags={STREAMDOWN_ALLOWED_TAGS}"),
    "PRECONDITION: the chat renderer relies on the allowedTags sanitizer",
  );
  assert.ok(
    /withDataImageSupport\(STREAMDOWN_ALLOWED_TAGS,/.test(
      MARKDOWN_TEXT,
    ) && MARKDOWN_TEXT.includes("rehypePlugins={rehypePlugins}"),
    "the passed pipeline must be derived from the defaults AND carry the allowedTags merge",
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
    assert.equal(
      rule.includes(":has("),
      false,
      "the exemption is NOT expressed with `:has()` here, which was the measured owner of the " +
        "whole 500K scroll cost on Chromium (#9669); the renderer decides instead",
    );
  }

  /* The two placeholder heights differ on purpose (paragraph vs display); equal values regress scroll. */
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
  // An ungated content-visibility: visible already exists on code blocks.
  assert.ok(
    INDEX_CSS.includes("content-visibility: visible !important"),
    "the code-block flicker rule is still there",
  );

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
  /* WebKit does not render skipped content when printing, so a print media rule is required. */
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

  assert.equal(
    INDEX_CSS.split("content-visibility: auto;").length - 1,
    2,
    "PRECONDITION: the two gated declarations this print block exists to switch off",
  );
});

test("no comment anywhere claims this feature ships off", () => {
  // Searches whole files, comments included, for stale statements of the default.
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
  // Vite substitutes the literal name, so docs must use the exact _CONTAINMENT override names.
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

  assert.ok(
    INDEX_CSS.includes(BUILD) && INDEX_CSS.includes(RUNTIME),
    "PRECONDITION: the stylesheet still documents both overrides",
  );
});

test("startup applies the mode before the first render", () => {
  // Line match, not substring: includes() accepts a commented-out call.
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
