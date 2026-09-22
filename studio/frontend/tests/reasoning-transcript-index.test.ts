// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { fromMarkdown } from "mdast-util-from-markdown";
import remend from "remend";
import {
  ReasoningTranscriptIndex,
  findReasoningAnchor,
  REASONING_FRAGMENT_CHARACTERS,
} from "../src/components/assistant-ui/reasoning-transcript-index.ts";

test("reading anchors resolve rendered Markdown, not formatting delimiters", () => {
  const fragments = new ReasoningTranscriptIndex().update([
    "Before\n\nA **bold** thought.\n\nAfter",
  ]);
  assert.equal(findReasoningAnchor(fragments, "A bold thought."), 1);
  assert.equal(findReasoningAnchor(fragments, "Missing"), -1);
});

test("long prose remains lossless and completed fragments keep their identity", () => {
  const index = new ReasoningTranscriptIndex();
  const source = Array.from(
    { length: 1000 },
    (_, i) => `Paragraph ${i}: **thoughts** about the game.\n\n`,
  ).join("");
  const before = index.update([source]);
  assert.equal(before.map((part) => part.text).join(""), source);
  assert.ok(
    before.every((part) => part.text.length <= REASONING_FRAGMENT_CHARACTERS),
  );
  const after = index.update([source + "Another thought"]);
  assert.equal(after[0], before[0]);
  assert.equal(after[0].key, before[0].key);
  assert.equal(
    after.map((part) => part.text).join(""),
    source + "Another thought",
  );
});

test("a giant fence does not swallow the surrounding prose", () => {
  const body = Array.from(
    { length: 5000 },
    (_, i) => `const bird${i} = ${i};`,
  ).join("\n");
  const source = `**Before**\n\n\`\`\`javascript\n${body}\n\`\`\`\n\n**After**`;
  const rows = new ReasoningTranscriptIndex().update([source]);
  assert.equal(rows[0].code, undefined);
  assert.equal(rows.at(-1)!.code, undefined);
  assert.match(rows.at(-1)!.text, /\*\*After\*\*/);
  const code = rows.filter((row) => row.code);
  assert.ok(code.length > 100);
  assert.equal(
    code.flatMap((row) => row.code!.lines.map((line) => line.text)).join("\n"),
    body,
  );
  assert.ok(
    code.every(
      (row) => row.code!.source === body && row.code!.language === "javascript",
    ),
  );
  assert.equal(code.filter((row) => row.first).length, 1);
  assert.equal(code.filter((row) => row.last).length, 1);
});

test("fence grammar and independent document boundaries survive streaming", () => {
  const index = new ReasoningTranscriptIndex();
  const open = "````html\n<div>\n```example\n" + "code\n".repeat(5000);
  const before = index.update([open, "# Another document"]);
  assert.equal(before.at(-1)!.text, "# Another document");
  assert.equal(before.at(-1)!.code, undefined);
  const after = index.update([
    open + "</div>\n````\n\nNormal **prose**",
    "# Another document",
  ]);
  assert.equal(after[0].key, before[0].key);
  assert.ok(
    after.some((row) => !row.code && row.text.includes("Normal **prose**")),
  );
  assert.ok(
    after
      .filter((row) => row.code)
      .every((row) => row.code!.source.includes("```example")),
  );
});

test("replacement and truncation discard stale fragments", () => {
  const index = new ReasoningTranscriptIndex();
  const old = index.update(["old thoughts\n\n".repeat(2000)]);
  const next = index.update(["New **thought**"]);
  assert.equal(next.length, 1);
  assert.equal(next[0].text, "New **thought**");
  assert.notEqual(next[0].key, old[0].key);
  assert.deepEqual(index.update([""]), []);
});

test("long unbroken lines stay bounded without losing surrogate pairs or CRLF", () => {
  for (const text of [
    "a😀".repeat(10000),
    "word\r\n".repeat(5000),
    "x".repeat(50000),
    "\n\n  Leading whitespace\n\n" + "word ".repeat(5000),
  ]) {
    const rows = new ReasoningTranscriptIndex().update([text]);
    assert.equal(rows.map((row) => row.text).join(""), text);
    assert.ok(
      rows.every((row) => row.text.length <= REASONING_FRAGMENT_CHARACTERS),
    );
    assert.ok(rows.every((row) => !/[\uD800-\uDBFF]$/.test(row.text)));
  }
});

test("an enormous code line is fragmented, including when the language is unknown", () => {
  const source = "x😀".repeat(50000);
  const rows = new ReasoningTranscriptIndex().update([
    "~~~unknown\n" + source + "\n~~~",
  ]);
  assert.ok(
    rows.every(
      (row) => row.code && row.text.length <= REASONING_FRAGMENT_CHARACTERS,
    ),
  );
  assert.equal(rows.map((row) => row.text).join(""), source);
});

test("a closing delimiter split across appends cannot eat later prose", () => {
  const index = new ReasoningTranscriptIndex();
  for (const suffix of ["`", "``", "```", "```\n", "```\n\nAfter"]) {
    const rows = index.update(["```js\nconst bird = true;\n" + suffix]);
    if (suffix.endsWith("After"))
      assert.ok(rows.some((row) => !row.code && row.text.includes("After")));
    else assert.ok(rows.some((row) => row.code));
  }
});

test("a bold paragraph keeps its formatting across bounded continuations", () => {
  const text = "**" + "long thought ".repeat(3000).trimEnd() + "**";
  const rows = new ReasoningTranscriptIndex().update([text]);
  assert.equal(rows.map((row) => row.text).join(""), text);
  for (const row of rows) {
    const parsed = fromMarkdown(remend(row.renderText ?? row.text));
    assert.equal(parsed.children[0].type, "paragraph");
    const paragraph = parsed.children[0];
    assert.ok(
      "children" in paragraph &&
        paragraph.children.some((child) => child.type === "strong"),
    );
  }
});

test("a quoted fence remains code when it crosses a rendering boundary", () => {
  const text =
    "> ```python\n" + "> print('bird')\n".repeat(3000) + "> ```\n\nAfter";
  const rows = new ReasoningTranscriptIndex().update([text]);
  assert.equal(rows.map((row) => row.text).join(""), text);
  for (const row of rows.filter((part) =>
    part.text.includes("print('bird')"),
  )) {
    const parsed = fromMarkdown(row.renderText ?? row.text);
    const quote = parsed.children[0];
    assert.equal(quote.type, "blockquote");
    assert.ok(
      "children" in quote &&
        quote.children.some((child) => child.type === "code"),
    );
  }
  assert.match(rows.at(-1)!.text, /After/);
});
