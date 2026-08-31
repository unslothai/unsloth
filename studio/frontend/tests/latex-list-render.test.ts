// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createMathPlugin } from "@streamdown/math";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";
import { stabilizeStreamingMarkdown } from "../src/components/assistant-ui/streaming-markdown.ts";
import { IncrementalMarkdownCache } from "../src/components/assistant-ui/streaming-render-schedule.ts";
import { normalizeEscapedInlineMath } from "../src/lib/escaped-inline-math.ts";
import { preprocessLaTeX } from "../src/lib/latex.ts";

const math = createMathPlugin({ singleDollarTextMath: true });

function renderResponse(
  source: string,
  isStreaming: boolean,
  cache = new IncrementalMarkdownCache(),
): string {
  const processed = stabilizeStreamingMarkdown(
    preprocessLaTeX(normalizeEscapedInlineMath(source)),
    isStreaming,
  );
  const incremental = isStreaming ? cache.update(processed) : null;
  return renderToStaticMarkup(
    React.createElement(
      Streamdown,
      {
        mode: "streaming",
        parseIncompleteMarkdown: !incremental,
        parseMarkdownIntoBlocksFn: incremental?.parseMarkdownIntoBlocks,
        isAnimating: isStreaming,
        plugins: { math },
      },
      incremental?.markdown ?? processed,
    ),
  );
}

function assertMath(html: string, sources: string[]): void {
  for (const source of sources) {
    assert.ok(
      html.includes(
        `<annotation encoding="application/x-tex">${source}</annotation>`,
      ),
      source,
    );
  }
}

const REALISTIC_RESPONSE = [
  "$$",
  "ds^2 = -c^2dt^2 + (dx-v_sf(r_s)dt)^2 + dy^2 + dz^2",
  "$$",
  "",
  "where:",
  "",
  "- \\$v_s\\$ is the velocity of the bubble,",
  "- \\$f(r_s)\\$ is a shape function, with \\$f \\to 0\\$ far away and \\$f \\to 1\\$ inside,",
  "- \\$r_s\\$ is the radial coordinate.",
].join("\n");

const REALISTIC_MATH = ["v_s", "f(r_s)", "f \\to 0", "f \\to 1", "r_s"];

test("completed generated lists reach KaTeX through the chat pipeline", () => {
  const html = renderResponse(REALISTIC_RESPONSE, false);

  assertMath(html, REALISTIC_MATH);
  assert.ok(!html.includes("$v_s$"));
  assert.ok(!html.includes("katex-error"));
});

test("streaming generated lists recover math after the escaped closer arrives", () => {
  const cache = new IncrementalMarkdownCache();
  const opener = REALISTIC_RESPONSE.indexOf("\\$v_s\\$");
  const openTail = opener + "\\$v_s".length;
  const beforeCloser = renderResponse(
    REALISTIC_RESPONSE.slice(0, openTail),
    true,
    cache,
  );
  assert.ok(
    !beforeCloser.includes(
      '<annotation encoding="application/x-tex">v_s</annotation>',
    ),
  );

  let html = beforeCloser;
  for (let end = openTail + 7; end < REALISTIC_RESPONSE.length; end += 23) {
    html = renderResponse(REALISTIC_RESPONSE.slice(0, end), true, cache);
    assert.ok(!html.includes("katex-error"), `stream prefix ${end}`);
  }
  html = renderResponse(REALISTIC_RESPONSE, true, cache);

  assertMath(html, REALISTIC_MATH);
  assert.ok(!html.includes("$v_s$"));
  assert.ok(!html.includes("katex-error"));
});

test("markdown literals and nested escaped math keep their rendered semantics", () => {
  const markdown = [
    "Normal prose renders \\$x_1\\$ inline.",
    "",
    "Price: $5, budget: $1,200, literal escaped dollar: \\$5.",
    "",
    "Code: `\\$code_1\\$` and an incomplete delimiter \\$unfinished",
    "",
    "[linked \\$y_2\\$](https://e.test/\\$literal\\$)",
    "",
    "> - **nested \\$z^2\\$ math**",
    "",
    "$$",
    "E=mc^2",
    "$$",
  ].join("\n");
  for (const isStreaming of [false, true]) {
    const html = renderResponse(markdown, isStreaming);

    assertMath(html, ["x_1", "y_2", "z^2", "E=mc^2"]);
    for (const literal of ["code_1", "literal", "unfinished"]) {
      assert.ok(
        !html.includes(
          `<annotation encoding="application/x-tex">${literal}</annotation>`,
        ),
        literal,
      );
    }
    assert.ok(html.includes('data-streamdown="inline-code"'));
    assert.ok(html.includes("\\$code_1\\$"));
    assert.ok(html.includes("$5"));
    assert.ok(html.indexOf("<blockquote") < html.indexOf(">z^2</annotation>"));
    assert.ok(html.indexOf("<li") < html.indexOf(">z^2</annotation>"));
    assert.ok(html.indexOf("<strong") < html.indexOf(">z^2</annotation>"));
    assert.ok(!html.includes("katex-error"));
  }
});

test("loose-list continuations reach KaTeX through completed and streaming paths", () => {
  for (const isStreaming of [false, true]) {
    const html = renderResponse("- item\n\n    \\$x\\$", isStreaming);

    assertMath(html, ["x"]);
    assert.ok(html.indexOf("<li") < html.indexOf("<annotation"));
    assert.ok(html.indexOf("<annotation") < html.indexOf("</li>"));
    assert.ok(!html.includes("$x$"));
  }
});

test("long existing display math renders as one intact KaTeX node", () => {
  const displayBody = `${"z+".repeat(2050)}\\$w\\$`;
  const html = renderResponse(`$$\n${displayBody}\n$$`, false);

  assert.equal(html.match(/application\/x-tex/g)?.length, 1);
  assert.ok(html.includes(`${displayBody}</annotation>`));
  assert.ok(!html.includes("katex-error"));
});

test("normalization precedes streaming repair and currency escaping", () => {
  const cases = [
    {
      markdown: String.raw`value \$v_{s}\$`,
      annotation: "v_{s}",
    },
    {
      markdown: String.raw`comparison \$x<y\$`,
      annotation: String.raw`x\lt y`,
    },
  ];
  for (const isStreaming of [false, true]) {
    for (const { markdown, annotation } of cases) {
      const html = renderResponse(markdown, isStreaming);
      assert.ok(
        html.includes(
          `<annotation encoding="application/x-tex">${annotation}</annotation>`,
        ),
        `${markdown} in ${isStreaming ? "streaming" : "completed"} mode`,
      );
      assert.ok(!html.includes("katex-error"));
      if (markdown.includes("x<y")) {
        assert.ok(html.includes("<mo>&lt;</mo>"));
      }
    }

    const subscript = renderResponse(String.raw`value \$v_{s}\$`, isStreaming);
    const withoutAnnotation = subscript.replace(
      /<annotation encoding="application\/x-tex">.*?<\/annotation>/g,
      "",
    );
    assert.ok(!withoutAnnotation.includes("_"));

    const currency = renderResponse(
      "The package is $5 + a $10 add-on",
      isStreaming,
    );
    assert.equal(currency.match(/application\/x-tex/g)?.length ?? 0, 0);
    assert.ok(currency.includes("$5 + a $10 add-on"));

    const mixedCurrency = renderResponse(
      String.raw`Cost $5; variable \$x\$; cap $10`,
      isStreaming,
    );
    assertMath(mixedCurrency, ["x"]);
    assert.ok(mixedCurrency.includes("Cost $5; variable"));
    assert.ok(mixedCurrency.includes("; cap $10"));
  }
});
