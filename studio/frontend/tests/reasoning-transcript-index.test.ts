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

test("reference links retain document definitions across fragments and appends", () => {
  const index = new ReasoningTranscriptIndex();
  const source = "Earlier paragraph.\n\n".repeat(1000) + "[Link][Ref]\n\n";
  const before = index.update([source]);
  const complete = source + '[ref]: https://example.org "Title"\n\nDone';
  const rows = index.update([complete]);
  assert.equal(rows.map((row) => row.text).join(""), complete);
  assert.equal(rows[0], before[0]);
  const link = rows.find((row) => row.text.includes("[Link]"))!;
  const parsed = fromMarkdown(link.renderText ?? link.text);
  assert.ok(parsed.children.some((node) => node.type === "paragraph"));
  assert.ok(JSON.stringify(parsed).includes('"type":"linkReference"'));
  assert.ok(JSON.stringify(parsed).includes('"url":"https://example.org"'));
  assert.equal(rows[0].renderText, undefined);
  const independent = index.update([complete, "[Link][ref]"]);
  assert.equal(independent.at(-1)!.renderText, undefined);
});

test("reference definitions update pending URLs, honor first wins, and reset", () => {
  const index = new ReasoningTranscriptIndex();
  const source = "[ref]\n\n[ref]: https://example.org";
  const row = () => index.update([source])[0];
  assert.match(row().renderText!, /https:\/\/example.org/);
  const updated = index.update([
    source + "/page\n\n[ref]: https://ignored.org",
  ]);
  assert.match(updated[0].renderText!, /https:\/\/example.org\/page/);
  assert.doesNotMatch(updated[0].renderText!, /ignored/);
  assert.equal(
    index.update(["[ref]\n\nNo definition"])[0].renderText,
    undefined,
  );
});

test("definitions inside quotes and lists retain document-wide scope", () => {
  for (const definition of [
    "> [ref]: https://example.org",
    "- [ref]: https://example.org",
    '> [ref]:\n>   https://example.org\n>   "Title"',
  ]) {
    const rows = new ReasoningTranscriptIndex().update([
      "[Link][ref]\n\nEarlier paragraph.\n\n" + definition,
    ]);
    const parsed = fromMarkdown(rows[0].renderText ?? rows[0].text);
    assert.ok(JSON.stringify(parsed).includes('"type":"linkReference"'));
    assert.ok(JSON.stringify(parsed).includes('"url":"https://example.org"'));
  }
});

test("synthesized definitions preserve entities and document-first precedence", () => {
  const rows = new ReasoningTranscriptIndex().update([
    '[Link][ref]\n\n[ref]: https://example.org/?x=&amp;amp; "&amp;amp;"',
  ]);
  const definition = fromMarkdown(rows[0].renderText!).children.find(
    (node) => node.type === "definition",
  )!;
  assert.equal(definition.url, "https://example.org/?x=&amp;");
  assert.equal(definition.title, "&amp;");
  const duplicates = new ReasoningTranscriptIndex().update([
    "[ref]: https://first.org\n\n> [ref]: https://second.org\n>\n> [Link][ref]",
  ]);
  assert.match(
    duplicates.at(-1)!.renderText!,
    /^\[ref\]: <https:\/\/first.org>/,
  );
});

test("oversized quoted items keep their enclosing blockquote", () => {
  const source = "> 1. " + "Long item ".repeat(3000);
  const rows = new ReasoningTranscriptIndex().update([source]);
  assert.equal(rows.map((row) => row.text).join(""), source);
  for (const row of rows) {
    const node = fromMarkdown(row.renderText ?? row.text).children[0];
    assert.equal(node.type, "blockquote");
    if (node.type === "blockquote") assert.equal(node.children[0].type, "list");
  }
});

test("oversized ordered, unordered, and nested items retain list containers", () => {
  for (const marker of ["1. ", "7) ", "- ", "- Parent\n  1. "]) {
    const source = marker + "Long item ".repeat(3000) + "\n";
    const rows = new ReasoningTranscriptIndex().update([source]);
    assert.equal(rows.map((row) => row.text).join(""), source);
    for (const [i, row] of rows.entries()) {
      const parsed = fromMarkdown(row.renderText ?? row.text);
      assert.equal(parsed.children[0].type, "list", row.key);
      assert.equal(
        row.listContinuationDepth ?? 0,
        i === 0 ? 0 : marker.includes("Parent") ? 2 : 1,
      );
      assert.ok(row.text.length <= REASONING_FRAGMENT_CHARACTERS);
    }
  }
});

test("a list split between items does not invent a nested continuation", () => {
  const source = Array.from(
    { length: 2000 },
    (_, i) => `${i + 1}. Item ${i}\n`,
  ).join("");
  const rows = new ReasoningTranscriptIndex().update([source]);
  assert.equal(rows.map((row) => row.text).join(""), source);
  for (const row of rows) {
    const list = fromMarkdown(row.renderText ?? row.text).children[0];
    assert.equal(list.type, "list");
    if (list.type !== "list") continue;
    assert.equal(list.start, Number.parseInt(row.text));
    assert.equal(row.listContinuationDepth ?? 0, 0);
    assert.ok(
      list.children.every((item) => item.children[0].type === "paragraph"),
    );
  }
});
