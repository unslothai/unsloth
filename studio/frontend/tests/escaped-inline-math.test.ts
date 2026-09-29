// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createMathPlugin } from "@streamdown/math";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { Streamdown } from "streamdown";
import { parseMarkdownIntoRenderableBlocks } from "../src/components/assistant-ui/streaming-render-schedule.ts";
import {
  looksLikeEscapedInlineMath,
  normalizeEscapedInlineMath,
} from "../src/lib/escaped-inline-math.ts";
import { preprocessLaTeX } from "../src/lib/latex.ts";

const math = createMathPlugin({ singleDollarTextMath: true });

function render(
  markdown: string,
  mode: "static" | "streaming" = "static",
  normalizeEscapedMath = true,
): string {
  return renderToStaticMarkup(
    React.createElement(
      Streamdown,
      {
        mode,
        plugins: { math },
        parseMarkdownIntoBlocksFn: parseMarkdownIntoRenderableBlocks,
      },
      preprocessLaTeX(
        normalizeEscapedMath ? normalizeEscapedInlineMath(markdown) : markdown,
      ),
    ),
  );
}

function annotations(html: string): string[] {
  return Array.from(
    html.matchAll(
      /<annotation encoding="application\/x-tex">(.*?)<\/annotation>/g,
    ),
    (match) => match[1],
  );
}

test("the escaped-math classifier stays conservative", () => {
  for (const body of ["x", "v_s", String.raw`f \to 0`, "x + y", "f(x)"]) {
    assert.equal(looksLikeEscapedInlineMath(body), true, body);
  }
  for (const body of ["5", "5 to 10", "read-only", "USD/EUR", "not math"]) {
    assert.equal(looksLikeEscapedInlineMath(body), false, body);
  }
});

test("escaped math is parsed in ordinary and list text", () => {
  const html = render(
    [
      String.raw`- \$v_s\$ and \$f(r_s)\$`,
      String.raw`- \$f \to 0\$ and \$x + y\$`,
      String.raw`after \$f(x)\$`,
    ].join("\n"),
  );
  assert.deepEqual(annotations(html), [
    "v_s",
    "f(r_s)",
    "f \\to 0",
    "x + y",
    "f(x)",
  ]);
});

test("Markdown syntax decides where escaped math is allowed", () => {
  const literalCases = [
    "`\\$v_s\\$`",
    "``a ` \\$v_s\\$ b``",
    "    \\$v_s\\$",
    "# heading\n    \\$v_s\\$",
    " \t\\$v_s\\$",
    "```tex\n\\$v_s\\$\n```",
    "```tex\n\\$v_s\\$",
    "```\n    ```\n\\$v_s\\$\n```",
    "- ```\n  \\$v_s\\$\n  ```",
    "> ```\n> \\$v_s\\$\n> ```",
    "```\n- ```\n\\$v_s\\$\n```",
    ">     \\$v_s\\$",
    "-     \\$v_s\\$",
    "- item\n\n      \\$v_s\\$",
    String.raw`<https://example.com/\$v_s\$>`,
  ];
  for (const markdown of literalCases) {
    assert.deepEqual(annotations(render(markdown)), [], markdown);
  }

  const linked = render(String.raw`[\$v_s\$](https://example.com/\$literal\$)`);
  assert.deepEqual(annotations(linked), ["v_s"]);

  const rawCode = render(String.raw`<code>\$x_1\$</code> after \$y_2\$`);
  assert.deepEqual(annotations(rawCode), ["y_2"]);
  assert.match(rawCode, /data-streamdown="inline-code">\$x_1\$<\/code>/);

  const quotedAttribute = render(
    String.raw`<code title="x > y">\$v_s\$</code>`,
  );
  assert.deepEqual(annotations(quotedAttribute), []);
  assert.ok(quotedAttribute.includes("$v_s$"));
});

test("escaped candidates cannot consume nested Markdown", () => {
  for (const markdown of [
    `${String.raw`Before \$x + `}\`y\`${String.raw`\$ then \$z\$`}`,
    String.raw`Before \$x + [y](https://e.test)\$ then \$z\$`,
    String.raw`Before \$x_ + [y][ref]\$ then \$z\$

[ref]: https://example.com/reference`,
    String.raw`Before \$x_ + [y][]\$ then \$z\$

[y]: https://example.com/reference`,
    String.raw`Before \$x + <code>y</code>\$ then \$z\$`,
  ]) {
    assert.deepEqual(annotations(render(markdown)), ["z"], markdown);
  }

  const referenceLink = render(String.raw`Before \$x_ + [y][ref]\$ then \$z\$

[ref]: https://example.com/reference`);
  assert.ok(referenceLink.includes('data-streamdown="link"'));
  assert.ok(referenceLink.includes(">y</button>"));
});

test("rejected candidates do not consume later math openers", () => {
  const cases: [string, string][] = [
    [String.raw`literal \$ then \$v_s\$`, "v_s"],
    [String.raw`literal \$ not math \$v_s\$`, "v_s"],
    [String.raw`\$5\$ + \$10\$ and \$v_s\$`, "v_s"],
    [String.raw`\$\$ then \$v_s\$`, "v_s"],
    [`${String.raw`\$`}${"a".repeat(201)}${String.raw`\$ + \$x\$`}`, "x"],
  ];
  for (const [markdown, expected] of cases) {
    assert.deepEqual(annotations(render(markdown)), [expected]);
  }
});

test("the maximum body length is inclusive", () => {
  const body = `v_${"a".repeat(198)}`;
  assert.equal(body.length, 200);
  assert.deepEqual(annotations(render(`\\$${body}\\$ and \\$x\\$`)), [
    body,
    "x",
  ]);
});

test("escaped prose and currency remain literal", () => {
  for (const markdown of [
    String.raw`the label \$read-only\$`,
    String.raw`the pair \$USD/EUR\$`,
    String.raw`\$5\$ is intentionally literal currency`,
    String.raw`\$5 to 10\$ is prose`,
  ]) {
    assert.deepEqual(annotations(render(markdown)), [], markdown);
  }
});

test("non-interrupting list markers cannot hide later escaped math", () => {
  const html = render("paragraph\n2. ```\n\\$v_s\\$");
  assert.deepEqual(annotations(html), ["v_s"]);
});

test("adjacent escaped spans remain separate math nodes", () => {
  assert.deepEqual(annotations(render(String.raw`\$a\$\$b\$`)), ["a", "b"]);
});

test("escaped display delimiters remain literal", () => {
  const markdown = String.raw`\$\$x^2\$\$`;
  assert.equal(normalizeEscapedInlineMath(markdown), markdown);
  assert.deepEqual(annotations(render(markdown)), []);
});

test("escaped currency inside a braced math command is not a closer", () => {
  const markdown = String.raw`\$\text{Revenue: \$5}\$`;
  assert.equal(
    normalizeEscapedInlineMath(markdown),
    String.raw`$\text{Revenue: {\char"24}5}$`,
  );
  const html = render(markdown);
  assert.deepEqual(annotations(html), [
    String.raw`\text{Revenue: {\char&quot;24}5}`,
  ]);
  assert.ok(html.includes("<mtext>Revenue:\u00a0$5</mtext>"));
  assert.ok(!html.includes("katex-error"));
});

test("currency markers do not hide escaped math between them", () => {
  const markdown = String.raw`Cost $5; variable \$x\$; cap $10`;
  assert.equal(
    normalizeEscapedInlineMath(markdown),
    "Cost $5; variable $x$; cap $10",
  );
  const html = render(markdown);
  assert.deepEqual(annotations(html), ["x"]);
  assert.ok(html.includes("Cost $5; variable"));
  assert.ok(html.includes("; cap $10"));
});

test("less-than signs remain valid inside TeX text", () => {
  const html = render(String.raw`\$\text{x<y}\$`);
  assert.deepEqual(annotations(html), [
    String.raw`\text{x{\char&quot;3C}y}`,
  ]);
  assert.match(html, /<mtext>x&lt;y<\/mtext>/);
  assert.ok(!html.includes("katex-error"));
});

test("escaped dollars inside existing math stay inside that math", () => {
  const exactBoundary = String.raw`\text{Revenue: \$USD\$}`.padEnd(200, "x");
  const cases = [
    String.raw`$\text{Revenue: \$USD\$}$`,
    String.raw`$30^\circ + \text{\$x\$}$`,
    `$${exactBoundary}$`,
    String.raw`\(\text{Revenue: \$x\$}\)`,
    String.raw`$$
\text{Revenue: \$USD\$}
$$`,
    String.raw`\[a + \$x\$\]`,
  ];
  for (const markdown of cases) {
    assert.equal(normalizeEscapedInlineMath(markdown), markdown);
    assert.equal(render(markdown), render(markdown, "static", false), markdown);
  }
});

test("unmatched bracket math does not escape its Markdown block", () => {
  const markdown = String.raw`unfinished \(

- \$v_s\$`;
  assert.equal(
    normalizeEscapedInlineMath(markdown),
    String.raw`unfinished \(

- $v_s$`,
  );
  assert.deepEqual(annotations(render(markdown)), ["v_s"]);
});

test("streaming waits for the escaped closer", () => {
  const incomplete = String.raw`- \$v_s`;
  assert.deepEqual(annotations(render(incomplete, "streaming")), []);
  assert.deepEqual(annotations(render(`${incomplete}\\$`, "streaming")), [
    "v_s",
  ]);
});

test("block boundaries and nested list continuations keep escaped math active", () => {
  for (const mode of ["static", "streaming"] as const) {
    assert.deepEqual(
      annotations(render("- - item\n\n      \\$nested_s\\$", mode)),
      ["nested_s"],
    );
    assert.deepEqual(
      annotations(render("`open\n***\n\\$boundary_s\\$`", mode)),
      ["boundary_s"],
    );
  }
});

test("incomplete code spans preserve escaped dollars while streaming", () => {
  for (const opener of ["`", "``"]) {
    const incomplete = `${opener}formula \\$x\\$`;
    assert.equal(normalizeEscapedInlineMath(incomplete), incomplete);
    const streamingHtml = render(incomplete, "streaming");
    assert.deepEqual(annotations(streamingHtml), []);
    if (opener === "`") {
      assert.ok(streamingHtml.includes(String.raw`formula \$x\$`));
    }
    assert.deepEqual(annotations(render(incomplete)), []);

    const completed = `${incomplete}${opener} after \\$y\\$`;
    for (const mode of ["streaming", "static"] as const) {
      const completedHtml = render(completed, mode);
      assert.deepEqual(annotations(completedHtml), ["y"]);
      assert.ok(completedHtml.includes(String.raw`formula \$x\$`));
    }
  }
});
