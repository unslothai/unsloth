// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// preprocessLaTeX is not prefix-stable, so at every prefix the cache's block list must
// equal a whole-document parse. Hand-picked cases miss the counterexamples; sweep a corpus.

import assert from "node:assert/strict";
import test from "node:test";
import remend from "remend";
import { parseMarkdownIntoBlocks } from "streamdown";

import { stabilizeStreamingMarkdown } from "../src/components/assistant-ui/streaming-markdown.ts";
import {
  IncrementalMarkdownCache,
  repairStreamingMarkdown,
} from "../src/components/assistant-ui/streaming-render-schedule.ts";
import { preprocessLaTeX } from "../src/lib/latex.ts";

const processStreamingText = (text: string): string =>
  stabilizeStreamingMarkdown(preprocessLaTeX(text), true);

const NAMED_CASES: Array<[string, string]> = [
  ["inline-paren", "The value \\(x^2\\) is positive.\n\n"],
  ["inline-dollar", "The value $x^2$ is positive.\n\n"],
  ["display-bracket", "Show \\[E = mc^2\\] which is famous.\n\n"],
  ["display-dollar", "Show\n\n$$\nE = mc^2\n$$\n\nwhich is famous.\n\n"],
  // A display body spanning blank lines: the `\n$$\n` rewrite meets the preceding blank line.
  ["display-multi-paragraph", "L0\n\n\\[s0\n\ns1\n\ns2\n\ns3\n\n\\]\n\n"],
  [
    "display-multi-paragraph-long",
    "Intro line.\n\nMore prose here.\n\n\\[a = 1\n\nb = 2\n\nc = 3\n\nd = 4\n\ne = 5\n\n\\]\n\nAfter the block $5.\n\n",
  ],
  ["lone-currency", "It costs $1,200 per epoch.\n\n"],
  ["two-dollars", "Prices run from $5 to $10 in that range.\n\n"],
  ["dollar-in-code-span", "Run `echo $5` to print it.\n\n"],
  ["dollar-in-fence", "```sh\nrun --seed $1 --limit $2\n```\n\n"],
  ["math-in-fence", "```tex\n\\(x^2\\) and \\[y\\]\n```\n\n"],
  ["math-in-tilde-fence", "~~~tex\n\\(x^2\\) and $5\n~~~\n\n"],
  ["unclosed-fence", "Here is code:\n\n```python\ndef f(x):\n    return $5 + x\n"],
  ["escaped-dollar", "A literal \\$5 stays literal.\n\n"],
  ["escaped-backslash", "A path C:\\\\temp and \\\\(not math\\\\) here.\n\n"],
  ["currency-suffix", "Around $5K then $5Ki then $3.5M.\n\n"],
  [
    "shell-variables",
    "Add $PATH, then check $HOME/.bashrc\nand run `echo $CUDA_HOME` with $x$.\n\n",
  ],
  ["cjk", "\u4fa1\u683c\u306f $5 \u3067\u3059\u3002\u6570\u5f0f \\(x^2\\) \u3082\u3042\u308a\u307e\u3059\u3002\n\n"],
  ["rtl-arabic", "\u0627\u0644\u0633\u0639\u0631 $5 \u0648\u0627\u0644\u0645\u0639\u0627\u062f\u0644\u0629 \\(x^2\\) \u0647\u0646\u0627.\n\n"],
  ["rtl-hebrew", "\u05d4\u05de\u05d7\u05d9\u05e8 $10 \u05d5\u05d4\u05e0\u05d5\u05e1\u05d7\u05d4 \\(y\\).\n\n"],
  ["emoji", "Cost \ud83d\udcb0 $5 and math \ud83e\uddee \\(x\\) done \ud83c\udf89\n\n"],
  [
    "table-dollars",
    "| item | cost |\n| --- | --- |\n| a | $5 |\n| b | $1,200 |\n\nAfter the table.\n\n",
  ],
  ["nested-code-span", "Use `` `literal` `` and `a ``b`` c` here.\n\n"],
  ["code-span-backticks", "The token `` ` `` and ``a ` b`` cost $5.\n\n"],
  ["code-span-contains-fence", "`~~~a~~~ $5`\n\n`x`\n\n`y`\n\n"],
  ["adjacent-spans", "`a $5` `b $6` `c $7`\n\n"],
  ["link-parens", "See [docs](https://e.com/a_(b)) and $5.\n\n"],
  ["list-with-math", "- rate \\(\\alpha\\)\n- budget $250\n- decay \\(\\lambda\\)\n\n"],
  [
    "list-with-escaped-dollar-math",
    "- rate \\$\\alpha\\$\n- budget $250\n- decay \\$\\lambda\\$\n\n",
  ],
  ["loose-list", "- item one\n\n  continued body $5\n\n- item two\n\n  more \\(x\\)\n\n"],
  ["blockquote", "> quoted \\(x\\) and $5\n> more\n\nafter\n\n"],
  ["setext", "Heading text\n===\n\nBody with $5.\n\n"],
  ["indented-code", "    code $5 here\n\n    more code \\(x\\)\n\nprose\n\n"],
  ["html-comment", "<!-- a comment $5\n\nstill comment \\(x\\) -->\n\nafter $6\n\n"],
  ["crlf", "Line one $5\r\n\r\nLine two \\(x\\)\r\n\r\n"],
  ["empty-math", "Empty \\(\\) span and \\[\\] here with $5.\n\n"],
  ["bold-math", "The **$30^\\circ$** angle and **$90 - x$** too.\n\n"],
  ["display-in-list", "1. first\n\n   \\[a + b\\]\n\n2. second\n\n"],
  ["multiline-inline-span", "Open \\(a\n\nb\n\nc\\) closed.\n\n"],
  ["footnote", "Text[^1] here $5.\n\n[^1]: note \\(x\\)\n\n"],
  ["one-char", "$"],
];

const FRAGMENTS: string[] = [
  "Plain prose sentence.\n\n",
  "Inline \\(x^2 + y\\) math.\n\n",
  "Display \\[a = b + c\\] block.\n\n",
  "\\[a0\n\na1\n\na2\n\na3\n\na4\n\n\\]\n\n",
  "Costs $5 today.\n\n",
  "From $5 to $10 range.\n\n",
  "Token `echo $5` span.\n\n",
  "`~~~a~~~ $5` span.\n\n",
  "```sh\nrun $1 $2\n```\n\n",
  "```\nunclosed $5\n",
  "Escaped \\$5 and \\\\ here.\n\n",
  "Opener \\(a only.\n\n",
  "Closer b\\) only.\n\n",
  "- bullet $5\n- bullet \\(x\\)\n\n",
  "> quote $5 \\(x\\)\n\n",
  "| a | b |\n| --- | --- |\n| $5 | \\(x\\) |\n\n",
  "\u4fa1\u683c $5 \u3068 \\(x\\)\u3002\n\n",
  "\ud83d\ude80 $5 \ud83e\uddee \\(z\\)\n\n",
  "**$30^\\circ$** bold math.\n\n",
  "    indented $5 code\n\n",
  "<!-- html $5 comment -->\n\n",
  "$$\nblock $5\n$$\n\n",
  "A \\(long ",
  "span\\) close.\n\n",
];

function makeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function generateDocument(random: () => number, maxFragments: number): string {
  const count = 1 + Math.floor(random() * maxFragments);
  let out = "";
  for (let index = 0; index < count; index += 1) {
    out += FRAGMENTS[Math.floor(random() * FRAGMENTS.length)];
  }
  return out;
}

function tighten(document: string, random: () => number, tightness: number): string {
  let out = "";
  for (let index = 0; index < document.length; index += 1) {
    if (
      document[index] === "\n" &&
      document[index + 1] === "\n" &&
      random() < tightness
    ) {
      out += "\n";
      index += 1;
      continue;
    }
    out += document[index];
  }
  return out;
}

function assertMatchesFullSplit(
  name: string,
  source: string,
  step: number,
  requireRetention = false,
): void {
  const cache = new IncrementalMarkdownCache();
  let everRetained = 0;
  for (let length = 0; length <= source.length; length += step) {
    const input = processStreamingText(source.slice(0, length));
    const render = cache.update(input);
    everRetained = Math.max(
      everRetained,
      render.parseMarkdownIntoBlocks("").length,
    );
    assert.deepEqual(
      render.parseMarkdownIntoBlocks(render.markdown),
      parseMarkdownIntoBlocks(repairStreamingMarkdown(input)),
      `block mismatch at prefix ${length} of ${name}: ${JSON.stringify(source.slice(0, 200))}`,
    );
  }
  if (requireRetention) {
    assert.ok(
      everRetained > 0,
      `${name} never retained a block, so this sweep asserted nothing about the ` +
        "retained prefix",
    );
  }
}

const LEAD = Array.from(
  { length: 6 },
  (_, index) => `Lead paragraph ${index}.\n\n`,
).join("");

test("the retained block list matches a full split at every prefix of every construct", () => {
  for (const [name, source] of NAMED_CASES) {
    assertMatchesFullSplit(name, source, 1);
    assertMatchesFullSplit(`${name} behind a retained prefix`, LEAD + source, 1, true);
  }
});

test("the same holds for generated replies, loose and tight", () => {
  const random = makeRandom(20260817);
  for (let index = 0; index < 40; index += 1) {
    const document = generateDocument(random, 8);
    assertMatchesFullSplit(`fuzz-${index}`, document, 1);
    assertMatchesFullSplit(
      `fuzz-${index}-tight`,
      tighten(document, random, 0.6),
      1,
    );
  }
});

test("rebuilds of the retained prefix do not grow with the reply", () => {
  const units = [
    "The residual \\(r_i = y_i - \\hat{y}_i\\) shrinks as the fit improves.\n\n",
    "Rewriting gives\n\n\\[ L(\\theta) = \\sum_i (y_i - \\theta x_i)^2 \\]\n\nwhich is convex.\n\n",
    "At that batch size the run costs about $1,200 per epoch.\n\n",
    "- learning rate three ten-thousandths\n- budget $250\n\n",
    "```python\ndef step(theta, grad, lr):\n    return theta - lr * grad\n```\n\n",
  ];
  const buildReply = (count: number): string => {
    let out = "";
    for (let index = 0; index < count; index += 1) {
      out += units[index % units.length];
    }
    return out;
  };

  const stream = (reply: string) => {
    const cache = new IncrementalMarkdownCache();
    let retained = 0;
    for (let length = 24; length <= reply.length; length += 24) {
      const render = cache.update(processStreamingText(reply.slice(0, length)));
      retained = render.parseMarkdownIntoBlocks("").join("").length;
    }
    return {
      rebuilds: (cache as unknown as { retainedPrefixRebuilds: number })
        .retainedPrefixRebuilds,
      retained,
      length: reply.length,
    };
  };

  const short = stream(buildReply(80));
  const long = stream(buildReply(320));
  // Both sides read a private field; fail on a rename instead of comparing undefined.
  assert.equal(
    typeof short.rebuilds,
    "number",
    "the rebuild counter was renamed or removed; this test measures nothing",
  );
  assert.ok(long.length > short.length * 3, "the long reply is not longer");
  assert.equal(
    long.rebuilds,
    short.rebuilds,
    `rebuilds grew with the reply: ${short.rebuilds} at ${short.length} ` +
      `characters, ${long.rebuilds} at ${long.length}`,
  );
  for (const run of [short, long]) {
    assert.ok(
      run.retained > run.length * 0.8,
      `retained only ${run.retained} of ${run.length} characters`,
    );
  }
});

test("preprocessLaTeX is not prefix-stable, which is why the above is not free", () => {
  const cases: Array<[string, string]> = [
    ["closing an inline span", "The value \\(x^2\\)"],
    ["a currency dollar", "Cost is $1"],
    ["closing a display span", "Show \\[E = mc^2\\]"],
  ];
  for (const [why, source] of cases) {
    let sawRewrite = false;
    let previous = preprocessLaTeX("");
    for (let length = 1; length <= source.length; length += 1) {
      const current = preprocessLaTeX(source.slice(0, length));
      if (!current.startsWith(previous)) {
        sawRewrite = true;
      }
      previous = current;
    }
    assert.ok(sawRewrite, `${why} no longer rewrites already-emitted text`);
  }
});
