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
    ["```tex\n\\$x\\$\n", "```tex\n\\$x\\$\n"],
    ["```\n    ```\n\\$x\\$\n```", "```\n    ```\n\\$x\\$\n```"],
    ["- ```\n  \\$x\\$\n  ```", "- ```\n  \\$x\\$\n  ```"],
    ["> ```\n> \\$x\\$\n> ```", "> ```\n> \\$x\\$\n> ```"],
    ["```\n- ```\n\\$x\\$\n```", "```\n- ```\n\\$x\\$\n```"],
    [
      "- item\n    ~~~tex\n    \\$x\\$\n    ~~~",
      "- item\n    ~~~tex\n    \\$x\\$\n    ~~~",
    ],
    [
      "- item\n    ```tex\n    \\$x\\$\n    ````",
      "- item\n    ```tex\n    \\$x\\$\n    ````",
    ],
    ["- item\n    ~~~tex\n    \\$x\\$", "- item\n    ~~~tex\n    \\$x\\$"],
    ["# heading\n    \\$x\\$", "# heading\n    \\$x\\$"],
    ["# heading\n \t\\$x\\$", "# heading\n \t\\$x\\$"],
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
    ["- ```txt\n  code\noutside \\$x\\$", "- ```txt\n  code\noutside $x$"],
    ["> ```txt\n> code\noutside \\$x\\$", "> ```txt\n> code\noutside $x$"],
    [
      "paragraph\n2. ``` literal\n\\$x\\$ after",
      "paragraph\n2. ``` literal\n$x$ after",
    ],
  ];

  for (const [input, expected] of cases) {
    assert.equal(preprocessLaTeX(input), expected, input);
  }
});

test("incomplete code spans stay literal without crossing block boundaries", () => {
  const cases: [string, string][] = [
    ["Use `code \\$x\\$", "Use `code \\$x\\$"],
    ["Use `code\n\\$x\\$", "Use `code\n\\$x\\$"],
    ["`open\n\n\\$x\\$\n\nclose`", "`open\n\n$x$\n\nclose`"],
    ["`open\n# heading\n\\$x\\$\nclose`", "`open\n# heading\n$x$\nclose`"],
    ["`open\nsoft \\$x\\$ close`", "`open\nsoft \\$x\\$ close`"],
  ];

  for (const [input, expected] of cases) {
    assert.equal(preprocessLaTeX(input), expected, input);
  }
});

test("escaped inline math from local models is recovered inside lists", () => {
  const input = [
    String.raw`- \$v_s\$ is the velocity of the bubble,`,
    String.raw`- \$f(r_s)\$ is a shape function, with \$f \to 0\$ far away and \$f \to 1\$ inside,`,
    String.raw`- \$r_s\$ is the radial coordinate.`,
  ].join("\n");
  const expected = [
    "- $v_s$ is the velocity of the bubble,",
    "- $f(r_s)$ is a shape function, with $f \\to 0$ far away and $f \\to 1$ inside,",
    "- $r_s$ is the radial coordinate.",
  ].join("\n");

  assert.equal(preprocessLaTeX(input), expected);
});

test("escaped math recovery preserves literal and non-math dollars", () => {
  const longBody = "a".repeat(201);
  const existingInlineBody = String.raw`\text{Revenue: \$USD\$}`.padEnd(
    200,
    "x",
  );
  const cases: [string, string][] = [
    [
      String.raw`\$5\$ is intentionally literal currency`,
      String.raw`\$5\$ is intentionally literal currency`,
    ],
    [String.raw`\$5 to 10\$ is prose`, String.raw`\$5 to 10\$ is prose`],
    [
      String.raw`\$5\$ + \$10\$ and \$v_s\$`,
      String.raw`\$5\$ + \$10\$ and $v_s$`,
    ],
    [String.raw`$5 + \$v_s\$ costs $10`, String.raw`\$5 + $v_s$ costs \$10`],
    ["$ v_s $ already works", "$ v_s $ already works"],
    ["$5 to $10 is a price range", "\\$5 to \\$10 is a price range"],
    ["`\\$v_s\\$` is code", "`\\$v_s\\$` is code"],
    [String.raw`    \$v_s\$`, String.raw`    \$v_s\$`],
    [String.raw`>     \$v_s\$`, String.raw`>     \$v_s\$`],
    [String.raw`-     \$v_s\$`, String.raw`-     \$v_s\$`],
    [
      String.raw`- item
    \$v_s\$`,
      "- item\n    $v_s$",
    ],
    [
      String.raw`- item

    \$v_s\$`,
      "- item\n\n    $v_s$",
    ],
    [
      String.raw`10. item

    \$v_s\$`,
      "10. item\n\n    $v_s$",
    ],
    [
      String.raw`- item

      \$v_s\$`,
      String.raw`- item

      \$v_s\$`,
    ],
    [
      String.raw`[\$v_s\$](https://example.com/\$literal\$)`,
      String.raw`[$v_s$](https://example.com/\$literal\$)`,
    ],
    ["$$\n v_s \n$$", "$$\n v_s \n$$"],
    [
      String.raw`$$
\text{Revenue: \$USD\$}
$$`,
      String.raw`$$
\text{Revenue: \$USD\$}
$$`,
    ],
    [String.raw`\$a\$\$b\$`, "$a$ $b$"],
    [`$${existingInlineBody}$`, `$${existingInlineBody}$`],
    [
      String.raw`\(\text{Revenue: \$USD\$}\)`,
      String.raw`$\text{Revenue: \$USD\$}$`,
    ],
    [String.raw`\$${longBody}\$ + \$x\$`, String.raw`\$${longBody}\$ + $x$`],
    [String.raw`\$v_s is incomplete`, String.raw`\$v_s is incomplete`],
    [String.raw`literal \$ then \$v_s\$`, String.raw`literal \$ then $v_s$`],
    ["\\$a\nb\\$ crosses a line", "\\$a\nb\\$ crosses a line"],
  ];

  for (const [input, expected] of cases) {
    assert.equal(preprocessLaTeX(input), expected, input);
  }
});
