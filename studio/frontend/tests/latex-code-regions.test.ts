// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { preprocessLaTeX } from "../src/lib/latex.ts";

test("currency inside inline code survives unrelated later code spans", () => {
  // An inline span containing a `~~~...~~~` pair used to yield overlapping
  // spans, so the binary search hit the inner one and missed the outer.
  const span = "`~~~a~~~ $5`";
  assert.equal(preprocessLaTeX(span), span);

  for (const trailer of [
    "\n\n`x`",
    "\n\n`x`\n\n`y`",
    "\n\n`x`\n\n`y`\n\n`z`",
  ]) {
    assert.equal(
      preprocessLaTeX(`${span}${trailer}`),
      `${span}${trailer}`,
      `currency inside inline code was rewritten by ${JSON.stringify(trailer)}`,
    );
  }
});

test("a code span's own text decides its escaping, whatever follows it", () => {
  // Appending an unrelated code span must never change an earlier one.
  const heads = [
    "`~~~a~~~ $5`",
    "`~~~ $5 ~~~`",
    "`a ``` b $5`",
    "`plain $5 here`",
    "```\nfenced $5\n```",
    "~~~\nfenced $5\n~~~",
    "`~~~a~~~ \\(x\\)`",
  ];
  for (const head of heads) {
    const alone = preprocessLaTeX(head);
    for (let count = 1; count <= 4; count += 1) {
      const trailer = "\n\n`t`".repeat(count);
      assert.equal(
        preprocessLaTeX(`${head}${trailer}`),
        `${alone}${trailer}`,
        `${JSON.stringify(head)} changed after ${count} trailing span(s)`,
      );
    }
  }
});

test("LaTeX inside inline code stays literal whatever follows it", () => {
  // The same lookup guards `convertLatexDelimiters`.
  const span = "`~~~a~~~ \\(x\\)`";
  assert.equal(preprocessLaTeX(span), span);
  assert.equal(preprocessLaTeX(`${span}\n\n\`x\``), `${span}\n\n\`x\``);
});

test("thematic and setext breaks stop cross-block code spans", () => {
  for (const boundary of ["***", "---", "==="]) {
    assert.equal(
      preprocessLaTeX(`\`open\n${boundary}\n\\(x\\)\``),
      `\`open\n${boundary}\n$x$\``,
      boundary,
    );
  }
});

test("ordinary code spans and fences are unchanged", () => {
  // Nothing that was already non-overlapping may move.
  const cases: [string, string][] = [
    ["`costs $5`", "`costs $5`"],
    ["``a ` \\$x\\$ b``", "``a ` \\$x\\$ b``"],
    ["`costs $5`\n\n`x`", "`costs $5`\n\n`x`"],
    ["```\ncosts $5\n```", "```\ncosts $5\n```"],
    ["```\ncosts $5\n```\n\n`x`", "```\ncosts $5\n```\n\n`x`"],
    ["a `b` c `d` e", "a `b` c `d` e"],
    ["costs $5 outside", "costs \\$5 outside"],
    ["costs $5 outside\n\n`x`", "costs \\$5 outside\n\n`x`"],
    ["```\n`inner`\n```", "```\n`inner`\n```"],
    ["```tex\n\\(x\\)\n", "```tex\n\\(x\\)\n"],
    ["```\n    ```\n\\(x\\)\n```", "```\n    ```\n\\(x\\)\n```"],
    ["- ```\n  \\(x\\)\n  ```", "- ```\n  \\(x\\)\n  ```"],
    ["> ```\n> \\(x\\)\n> ```", "> ```\n> \\(x\\)\n> ```"],
    [
      "- - ```\n    costs $5 and \\(x\\)\n    ```",
      "- - ```\n    costs $5 and \\(x\\)\n    ```",
    ],
    ["```\n- ```\n\\(x\\)\n```", "```\n- ```\n\\(x\\)\n```"],
    [
      "- item\n    ~~~tex\n    \\(x\\)\n    ~~~",
      "- item\n    ~~~tex\n    \\(x\\)\n    ~~~",
    ],
    [
      "- item\n    ```tex\n    \\(x\\)\n    ````",
      "- item\n    ```tex\n    \\(x\\)\n    ````",
    ],
    ["- item\n    ~~~tex\n    \\(x\\)", "- item\n    ~~~tex\n    \\(x\\)"],
    ["# heading\n    \\(x\\)", "# heading\n    \\(x\\)"],
    ["# heading\n \t\\(x\\)", "# heading\n \t\\(x\\)"],
    ["`\\(x\\)` and \\(y\\)", "`\\(x\\)` and $y$"],
    ["```\n\\(x\\)\n```\n\nthen \\(y\\)", "```\n\\(x\\)\n```\n\nthen $y$"],
  ];
  for (const [input, expected] of cases) {
    assert.equal(
      preprocessLaTeX(input),
      expected,
      `changed for ${JSON.stringify(input)}`,
    );
  }
});

test("fences stop at their Markdown container boundary", () => {
  const cases: [string, string][] = [
    ["- ```txt\n  code\noutside \\(x\\)", "- ```txt\n  code\noutside $x$"],
    ["> ```txt\n> code\noutside \\(x\\)", "> ```txt\n> code\noutside $x$"],
    [
      "paragraph\n2. ``` literal\n\\(x\\) after",
      "paragraph\n2. ``` literal\n$x$ after",
    ],
  ];

  for (const [input, expected] of cases) {
    assert.equal(preprocessLaTeX(input), expected, input);
  }
});

test("same-line nested items set the continuation content column", () => {
  assert.equal(
    preprocessLaTeX("- - item\n\n      \\(x\\)"),
    "- - item\n\n      $x$",
  );
  assert.equal(
    preprocessLaTeX("-   - item\n\n        \\(x\\)"),
    "-   - item\n\n        $x$",
  );
});
