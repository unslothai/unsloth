// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  composerSchema,
  docPosToOffset,
  markdownSlice,
  offsetToDocPos,
  parseMarkdown,
  serializeMarkdown,
  sliceToMarkdown,
} from "../src/features/chat/rich-composer/markdown.ts";
import type { Node } from "../src/vendor/prosemirror-model/index.js";

const {
  paragraph,
  code_block: codeBlock,
  hard_break: hardBreak,
} = composerSchema.nodes;
const doc = (...blocks: Node[]) => composerSchema.node("doc", null, blocks);
const p = (...content: (Node | string)[]) =>
  paragraph.create(
    null,
    content.map((part) =>
      typeof part === "string" ? composerSchema.text(part) : part,
    ),
  );
const t = (text: string, ...marks: string[]) =>
  composerSchema.text(
    text,
    marks.map((name) => composerSchema.marks[name].create()),
  );

// What a block's styled content is, as [text, marks] pairs and "\n" for breaks.
function inline(node: Node): (string | [string, string])[] {
  const out: (string | [string, string])[] = [];
  for (const child of node.children) {
    if (child.type === hardBreak) out.push("\n");
    else
      out.push([
        child.text ?? "",
        child.marks.map((m) => m.type.name).join("+"),
      ]);
  }
  return out;
}

const ROUND_TRIP = [
  "",
  "hello",
  "line one\nline two",
  "\n\nblank lines\n\n\nkept\n",
  "**bold** and *italic* and ~~strike~~ and `code`",
  "**a *b* c**",
  "*a **b** c*",
  "***both***",
  "**bold with `code` inside**",
  "2 * 3 * 4 and a*b*c and snake_case_name and __dunder__",
  "** not bold ** and * not italic * and ``double`` and `unclosed",
  "\\*escaped\\* and \\`tick\\`",
  "# Heading\n- list item\n> quote\n| a | b |\n| --- | --- |\n[link](https://example.com)",
  "```\ncode\n```",
  "```ts\nconst x = 1;\n\nconsole.log(`x`);\n```\nafter",
  "before\n```py\nprint(1)\n```",
  "```\n```",
  "```\n\n```",
  "```js\nunclosed fence",
  "  ```\nindented fence\n  ```",
  "````\nfour ticks\n````",
  "```\na\n```\n```\nb\n```",
  "```\nx\n```\n",
  "@skill-name please *fix* this",
  "emoji 😀 and RTL שלום **מודגש** and CJK 你好 **粗体**",
  "tab\there  and  double  spaces ",
  "**trailing space **",
  "*a*b*c*",
  "**a**b**c**",
  "~~~\nnot a fence\n~~~",
  "\r\nwindows line",
];

test("every Markdown string reads back to itself", () => {
  for (const markdown of ROUND_TRIP) {
    assert.equal(
      serializeMarkdown(parseMarkdown(markdown)),
      markdown,
      JSON.stringify(markdown),
    );
  }
});

test("random marker soup reads back to itself", () => {
  const alphabet = [
    "*",
    "**",
    "~~",
    "`",
    "```",
    "\n",
    " ",
    "a",
    "b",
    "\\",
    "_",
    "@x",
  ];
  let seed = 7;
  const next = () => {
    seed = (seed * 1103515245 + 12345) % 2 ** 31;
    return seed;
  };
  for (let n = 0; n < 3000; n++) {
    let markdown = "";
    const length = next() % 14;
    for (let i = 0; i < length; i++)
      markdown += alphabet[next() % alphabet.length];
    assert.equal(
      serializeMarkdown(parseMarkdown(markdown)),
      markdown,
      JSON.stringify(markdown),
    );
  }
});

test("markers become marks and fences become code blocks", () => {
  const parsed = parseMarkdown("**b** *i* ~~s~~ `c`\n**a *b* c**");
  assert.equal(parsed.childCount, 1);
  assert.deepEqual(inline(parsed.child(0)), [
    ["b", "bold"],
    [" ", ""],
    ["i", "italic"],
    [" ", ""],
    ["s", "strike"],
    [" ", ""],
    ["c", "code"],
    "\n",
    ["a ", "bold"],
    ["b", "bold+italic"],
    [" c", "bold"],
  ]);
  const fenced = parseMarkdown("intro\n```ts\nlet a;\n```\nouter");
  assert.deepEqual(
    fenced.content.content.map((block) => block.type.name),
    ["paragraph", "code_block", "paragraph"],
  );
  assert.equal(fenced.child(1).attrs.language, "ts");
  assert.equal(fenced.child(1).textContent, "let a;");
  // Underscores, headings and lists are not styled; they stay text.
  assert.deepEqual(inline(parseMarkdown("_x_ # h").child(0)), [
    ["_x_ # h", ""],
  ]);
});

test("a styled document serialises to the Markdown it shows", () => {
  assert.equal(
    serializeMarkdown(
      doc(
        p(
          t("bold", "bold"),
          " ",
          t("both", "bold", "italic"),
          " ",
          t("x", "strike"),
        ),
      ),
    ),
    "**bold** ***both*** ~~x~~",
  );
  // A mark's edge spaces go outside its markers.
  assert.equal(
    serializeMarkdown(doc(p(t("bold ", "bold"), "next"))),
    "**bold** next",
  );
  assert.equal(serializeMarkdown(doc(p("a", t(" b", "italic")))), "a *b*");
  // Code keeps its spaces and gets a longer fence around backticks.
  assert.equal(serializeMarkdown(doc(p(t(" a ", "code")))), "` a `");
  assert.equal(serializeMarkdown(doc(p(t("a`b", "code")))), "``a`b``");
  assert.equal(serializeMarkdown(doc(p(t("x", "bold", "code")))), "**`x`**");
  // Marks close at a line break.
  assert.equal(
    serializeMarkdown(
      doc(p(t("a", "bold"), hardBreak.create(), t("b", "bold"))),
    ),
    "**a**\n**b**",
  );
  assert.equal(
    serializeMarkdown(
      doc(codeBlock.create({ language: "py" }, composerSchema.text("x\ny"))),
    ),
    "```py\nx\ny\n```",
  );
  assert.equal(serializeMarkdown(doc(codeBlock.create(), p())), "```\n```\n");
  assert.equal(serializeMarkdown(doc(p("a"), p("b"))), "a\nb");
});

test("caret positions map to Markdown offsets and back", () => {
  const markdown = "ab **cd** @sk\n```\nxy\n```\nz";
  const parsed = parseMarkdown(markdown);
  // Positions: p opens at 0, text starts at 1.
  assert.equal(docPosToOffset(parsed, 1), 0);
  assert.equal(docPosToOffset(parsed, 3), 2); // after "ab"
  // Run boundaries map to the earlier run's end: before an opening or closing marker.
  assert.equal(docPosToOffset(parsed, 4), 3);
  assert.equal(docPosToOffset(parsed, 5), 6);
  assert.equal(docPosToOffset(parsed, 6), 7);
  const mention = markdown.indexOf("@sk") + 3;
  assert.equal(
    markdown.slice(0, docPosToOffset(parsed, offsetToDocPos(parsed, mention))),
    "ab **cd** @sk",
  );
  // Code block text and the paragraph after it.
  const x = markdown.indexOf("xy");
  assert.equal(docPosToOffset(parsed, offsetToDocPos(parsed, x + 1)), x + 1);
  assert.equal(
    docPosToOffset(parsed, offsetToDocPos(parsed, markdown.length)),
    markdown.length,
  );
  // An offset inside a fence line snaps forward to the code text.
  assert.equal(
    docPosToOffset(parsed, offsetToDocPos(parsed, markdown.indexOf("```") + 1)),
    x,
  );
  for (let offset = 0; offset <= markdown.length; offset++) {
    const pos = offsetToDocPos(parsed, offset);
    assert.ok(pos >= 0 && pos <= parsed.content.size, `offset ${offset}`);
  }
});

test("pasted Markdown slices join the caret's line when they are one paragraph", () => {
  const one = markdownSlice("**x** y");
  assert.equal(one.openStart, 0);
  assert.equal(one.content.firstChild?.marks[0]?.type.name, "bold");
  const blocks = markdownSlice("a\n```\nb\n```\nc");
  assert.deepEqual([blocks.openStart, blocks.openEnd], [1, 1]);
  const codeFirst = markdownSlice("```\nb\n```");
  assert.deepEqual([codeFirst.openStart, codeFirst.openEnd], [0, 0]);
  assert.equal(sliceToMarkdown(blocks), "a\n```\nb\n```\nc");
  assert.equal(sliceToMarkdown(one), "**x** y");
});

test("a huge line of unmatched markers reads back quickly, as literal text", () => {
  const markdown = `${"~~a ".repeat(30000)}**end**`;
  const started = performance.now();
  const parsed = parseMarkdown(markdown);
  assert.equal(serializeMarkdown(parsed), markdown);
  assert.ok(performance.now() - started < 1000);
  assert.equal(parsed.firstChild?.firstChild?.marks.length, 0);
});
